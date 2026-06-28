"""
Label-only MIA audit with HAMP and filtering defenses.

Trains shadow models and audits privacy using a label-only attack that:
1. Queries the model on 18 augmentations of the canary
2. Records binary correctness vector
3. Fits logistic regression (LOO CV) to score membership

Supports three defense types:
- none: standard CE training
- hamp: entropy regularization + label smoothing (train-time) + confidence randomization (test-time)
- filter: gradient-norm filtering during training

Bug fixes applied:
1. compute_eps_lower_from_mia return values were unpacked in wrong order (max_t, emp_eps).
2. drop_mask mutations inside clip_and_accum_grads were not propagated back to the
   outer drop_mask array because batch_drop_mask was only a view/copy.
3. Epoch-end per-class filter could include samples still flagged as 1 (pending ascent)
   in the active set; all 1s are now flushed to 2 before the filter runs.
4. generate_augmentations used torch.roll which wraps pixel content across boundaries;
   replaced with zero-padded shifting so augmentations match real pipeline behaviour.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from models import Models
from utils.data import load_data
from utils.training import (
    xavier_init_model, init_wideresnet, IndexedTensorDataset, test_model
)
from utils.dpsgd import clip_and_accum_grads, DefenseConfig
from utils.audit import compute_eps_lower_from_mia
from utils.args import build_parser
from sklearn.linear_model import LogisticRegression

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# ============================================================================
# HAMP Helper Functions
# ============================================================================

def compute_p_from_target_entropy(gamma, num_classes):
    """
    Binary search for p such that entropy of [p, q, q, ..., q] equals gamma * log(C).

    Args:
        gamma: target entropy as fraction of max entropy (0-1)
        num_classes: number of classes C

    Returns:
        p: probability for correct class
    """
    target_entropy = gamma * np.log(num_classes)

    def entropy(p, C):
        if p <= 0 or p > 1:
            return 0
        q = (1 - p) / (C - 1)
        if q < 0:
            return 0
        h = -p * np.log(p + 1e-10)
        if C > 1 and q > 0:
            h -= (C - 1) * q * np.log(q + 1e-10)
        return h

    lo, hi = 1.0 / num_classes, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        h = entropy(mid, num_classes)
        if h < target_entropy:
            hi = mid
        else:
            lo = mid
    return lo


def generate_soft_labels(y, num_classes, p):
    """
    Generate soft labels for HAMP training.

    For each sample, set true class to p and others to (1-p)/(C-1).

    Args:
        y: class indices, shape (B,)
        num_classes: C
        p: probability for true class

    Returns:
        soft labels, shape (B, C)
    """
    B = len(y)
    soft = torch.full((B, num_classes), (1 - p) / (num_classes - 1), dtype=torch.float32)
    soft[torch.arange(B), y] = p
    return soft


def kl_divergence_with_entropy_regularization(logits, soft_labels, alpha_entropy):
    """
    HAMP training loss: KL(soft_labels || softmax(logits)) - alpha * H(softmax(logits)).

    The entropy term is subtracted (maximized) to encourage high-entropy predictions.
    """
    probs = F.softmax(logits, dim=1)

    # KL divergence
    kl_loss = F.kl_div(
        F.log_softmax(logits, dim=1),
        soft_labels,
        reduction='batchmean'
    )

    # Entropy regularization (negative entropy since we subtract)
    entropy = -(probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

    return kl_loss - alpha_entropy * entropy


def rank_preserving_score_replacement(model, logits, batch_shape, rng, device):
    """
    HAMP test-time defense: randomize confidence by replacement with sorted logits 
    from predictions on random noise, preserving argmax class prediction order.
    Replicates the exact confidence randomization logic from experiments/hamp.py.
    """
    # 1. Generate random sample of the same batch shape
    batch_xs_random = []
    for _ in range(batch_shape[0]):
        batch_xs_random.append(rng.integers(0, 256, size=batch_shape[1:], dtype=np.uint8))
    batch_xs_random = torch.from_numpy(np.stack(batch_xs_random)).float()
    
    # 2. Scale and normalize (CIFAR-10 stats)
    CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    CIFAR10_STD_DEV = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
    batch_xs_random = batch_xs_random.to(device) / 255.0
    batch_xs_random = (batch_xs_random - CIFAR10_MEAN) / CIFAR10_STD_DEV
    
    # 3. Predict on random sample to get noise logits
    with torch.no_grad():
        pred_logits_random = model(batch_xs_random).cpu()
    
    # 4. Apply test-time defense
    pred_logits_traintime = logits.cpu().clone()
    pred_logits_random = torch.sort(pred_logits_random, stable=True, dim=-1).values
    assert pred_logits_random.dtype == pred_logits_random.dtype
    assert torch.all(pred_logits_random.max(-1).values == pred_logits_random[..., -1])

    # Make sure maxima of raw and random logits are unique per-sample to ensure test-time defense does not change label
    # NB: This does not necessarily preserve top-k order for k > 1, but we only care about top-1 here
    pred_logits_random[..., -1] += torch.finfo(pred_logits_random.dtype).eps  # min. pos. value s.t. x + eps > x
    assert torch.all(pred_logits_random[..., -1].unsqueeze(-1) > pred_logits_random[..., :-1])
    pred_labels_traintime = pred_logits_traintime.argmax(-1, keepdim=True)
    pred_logits_traintime.scatter_add_(
        -1,
        index=pred_labels_traintime,
        src=torch.tensor(torch.finfo(pred_logits_traintime.dtype).eps, dtype=pred_logits_traintime.dtype).expand(
            pred_labels_traintime.size()
        ),
    )
    num_classes = pred_logits_traintime.size(-1)
    assert torch.all(
        (pred_logits_traintime.max(-1, keepdim=True).values > pred_logits_traintime).int().sum(-1) == num_classes - 1
    )

    # Calculate defended predictions by reordering random logits
    pred_label_order = torch.argsort(pred_logits_traintime, stable=True, dim=-1)
    pred_logits_testtime = torch.empty_like(pred_logits_random)
    pred_logits_testtime.scatter_(dim=-1, index=pred_label_order, src=pred_logits_random)
    assert pred_logits_testtime.size() == pred_logits_traintime.size()
    # This should always be true since both the maxima of raw and random logits are unique per-sample
    assert torch.all(pred_logits_testtime.argmax(-1) == pred_logits_traintime.argmax(-1))
    
    return pred_logits_testtime.to(device)


# ============================================================================
# Augmentation and Attack Functions
# ============================================================================

def shift_image(x, shift_h, shift_w):
    """
    Shift a (C, H, W) image tensor by (shift_h, shift_w) pixels using zero-padding.

    Positive shift_h moves content DOWN (rows shift toward higher indices).
    Positive shift_w moves content RIGHT (cols shift toward higher indices).

    Unlike torch.roll, pixels shifted beyond the boundary are replaced with zeros
    rather than wrapping around, matching standard data-augmentation behaviour.

    Args:
        x: tensor of shape (..., H, W)
        shift_h: vertical shift in pixels (may be negative)
        shift_w: horizontal shift in pixels (may be negative)

    Returns:
        shifted tensor of the same shape as x
    """
    if shift_h == 0 and shift_w == 0:
        return x.clone()

    pad_h = abs(shift_h)
    pad_w = abs(shift_w)

    # F.pad order: (left, right, top, bottom)
    x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

    # Crop back to original spatial size, offset by the shift direction
    H_orig = x.shape[-2]
    W_orig = x.shape[-1]

    # If shift_h > 0 we want content moved DOWN, so crop starting from top=0 (not pad_h)
    top  = pad_h - shift_h   # shift_h > 0 → top < pad_h → crop starts earlier (moves down)
    left = pad_w - shift_w

    return x_padded[..., top:top + H_orig, left:left + W_orig].clone()


def generate_augmentations(x, num_augmentations, use_flip=True):
    """
    Generate fixed augmentations: zero-padded shifts + horizontal flips.

    FIX (Bug 4): replaced torch.roll (wrap-around) with zero-padded shifting so
    augmented images do not contain wrap-around pixel artefacts.

    For CIFAR (use_flip=True): 2 flips × 3 shifts × 3 shifts = 18 augmentations
    For MNIST (use_flip=False): first 18 of 5×5 shifts

    Args:
        x: input tensor, shape (C, H, W)
        num_augmentations: number to generate
        use_flip: whether to apply horizontal flips

    Returns:
        list of augmented tensors, each shape (C, H, W)
    """
    augmentations = []
    shifts = [0, -4, 4] if use_flip else list(range(-2, 3))
    flip_opts = [False, True] if use_flip else [False]

    for do_flip in flip_opts:
        for shift_h in shifts:
            for shift_w in shifts:
                if len(augmentations) >= num_augmentations:
                    return augmentations

                aug = x.clone()
                if do_flip:
                    aug = torch.flip(aug, dims=[-1])

                # BUG FIX: use zero-padded shift instead of torch.roll
                aug = shift_image(aug, shift_h, shift_w)
                augmentations.append(aug)

    return augmentations


def generate_binary_correctness_vector(model, x, y, augmentations, device, defense_type='none', rng=None):
    """
    Label-only attack: query model on augmentations and record correctness.

    For each augmentation, returns 1 if argmax == y, else 0.
    This binary vector is the attack signal.

    Args:
        model: trained model (in eval mode)
        x: original input (unused here; augmentations already prepared)
        y: true class (int)
        augmentations: list of augmented inputs, each shape (C, H, W)
        device: torch device
        defense_type: defense type to determine if test-time HAMP randomization is applied
        rng: np.random.Generator used for HAMP test-time randomization

    Returns:
        binary vector, shape (len(augmentations),) as numpy float32
    """
    model.eval()
    binary_vector = []

    with torch.no_grad():
        for aug in augmentations:
            aug = aug.to(device)
            logits = model(aug.unsqueeze(0))
            
            # Apply HAMP test-time confidence randomization if active
            if defense_type in ('hamp', 'hamp_testonly') and rng is not None:
                batch_shape = (1, aug.shape[0], aug.shape[1], aug.shape[2])
                logits = rank_preserving_score_replacement(model, logits, batch_shape, rng, device)
                
            pred = logits.argmax(dim=1).item()
            binary_vector.append(1.0 if pred == y else 0.0)

    return np.array(binary_vector, dtype=np.float32)


# ============================================================================
# Training Functions
# ============================================================================

def train_model(model, X, y, canary_x, canary_y, device, args, defense_type='none', momentum=None, weight_decay=None):
    """
    Train a model with optional HAMP or gradient-norm filter defense.

    Args:
        model: PyTorch model
        X, y: training data and labels (numpy), already includes canary if in-world
        canary_x, canary_y: unused here, kept for signature consistency
        device: torch device
        args: argument namespace
        defense_type: 'none', 'hamp', 'hamp_testonly', or 'filter'
        momentum: SGD momentum (optional)
        weight_decay: SGD weight decay (optional)

    Returns:
        trained model
    """
    model.train()
    batch_size = args.batch_size if args.batch_size else 256
    if momentum is None:
        momentum = getattr(args, 'momentum', None)
    if weight_decay is None:
        weight_decay = getattr(args, 'weight_decay', None)

    sgd_kwargs = {}
    if momentum is not None:
        sgd_kwargs['momentum'] = momentum
    if weight_decay is not None:
        sgd_kwargs['weight_decay'] = weight_decay

    optimizer = optim.SGD(model.parameters(), lr=args.lr, **sgd_kwargs)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    dataset = IndexedTensorDataset(X_tensor, y_tensor)
    num_classes = int(y_tensor.max().item()) + 1

    sampling = getattr(args, 'sampling', 'poisson')
    n_samples = len(X_tensor)
    if sampling == 'poisson':
        _poisson_q = batch_size / n_samples
        _poisson_n_batches = (n_samples + batch_size - 1) // batch_size
        loader = None
        num_steps_per_epoch = _poisson_n_batches
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        num_steps_per_epoch = len(loader)

    num_steps_per_epoch = max(1, num_steps_per_epoch)

    import warnings
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torch.optim.lr_scheduler")
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / num_steps_per_epoch,
                end_factor=1.0,
                total_iters=num_steps_per_epoch,
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[epoch * num_steps_per_epoch for epoch in (60, 120, 160)], gamma=0.2
            ),
        ],
        milestones=[1 * num_steps_per_epoch],
    )

    if defense_type in ('none', 'hamp', 'hamp_testonly'):
        hamp_gamma = getattr(args, 'hamp_gamma', 0.95)
        hamp_alpha_entropy = getattr(args, 'hamp_alpha_entropy', 1.0)

        if defense_type == 'hamp':
            p = compute_p_from_target_entropy(hamp_gamma, num_classes)

        for epoch in range(args.n_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            if sampling == 'poisson':
                def _poisson_iter():
                    for _ in range(_poisson_n_batches):
                        mask = torch.rand(n_samples) < _poisson_q
                        idx = torch.where(mask)[0]
                        if len(idx) == 0:
                            continue
                        yield X_tensor[idx], y_tensor[idx], idx
                batch_iter = _poisson_iter()
            else:
                batch_iter = loader

            for X_b, y_b, _ in batch_iter:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                logits = model(X_b)

                if defense_type == 'hamp':
                    soft_labels = generate_soft_labels(y_b, num_classes, p).to(device)
                    loss = kl_divergence_with_entropy_regularization(logits, soft_labels, hamp_alpha_entropy)
                else:
                    # 'none' and 'hamp_testonly' both train with standard CE.
                    # hamp_testonly applies rank-preserving confidence randomization at
                    # inference time only, which preserves argmax and therefore has no
                    # effect on the binary correctness vector — demonstrating that
                    # HAMP's test-time defense cannot hide membership from a label-only attack.
                    loss = F.cross_entropy(logits, y_b)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                batch_size_actual = X_b.size(0)
                epoch_loss += loss.item() * batch_size_actual
                epoch_correct += (logits.argmax(dim=1) == y_b).sum().item()
                epoch_samples += batch_size_actual

            if epoch_samples > 0:
                epoch_loss /= epoch_samples
                epoch_acc = epoch_correct / epoch_samples
            else:
                epoch_loss = 0.0
                epoch_acc = 0.0
            current_lr = lr_scheduler.get_last_lr()[0]
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{args.n_epochs}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}  lr={current_lr:.6f}")

    elif defense_type == 'filter':
        block_size = args.block_size if args.block_size else batch_size
        block_size = min(block_size, batch_size)
        max_grad_norm = args.max_grad_norm
        defense_k = args.defense_k
        defense_apply_ascent = args.defense_apply_ascent
        defense_score_fn = args.defense_score_fn
        defense_score_norm = args.defense_score_norm
        defense_filter_every = getattr(args, 'defense_filter_every', 1)

        criterion = nn.CrossEntropyLoss()
        scores = np.zeros(len(dataset), dtype=np.float32)
        # 0 = active, 1 = flagged for gradient ascent, 2 = permanently dropped
        drop_mask = np.zeros(len(dataset), dtype=np.int8)

        sampling = getattr(args, 'sampling', 'poisson')
        n_samples = len(dataset)
        if sampling == 'poisson':
            _poisson_q = batch_size / n_samples
            _poisson_n_batches = (n_samples + batch_size - 1) // batch_size
            loader = None
        else:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Lazy projection matrix state (populated inside clip_and_accum_grads)
        grad_dir_proj = rand_proj_mat = maxmin_proj_mat = None
        alignment_proj_mat = grad_accel_proj = grad_jerk_proj = None
        grad_norm_hist = grad_norm_hist_pos = None
        grad_dir_hist = grad_dir_hist_pos = None
        grad_accel_hist = grad_accel_hist_pos = None
        grad_jerk_hist = grad_jerk_hist_pos = None
        dir_unique_hist = dir_unique_hist_pos = None
        prev_losses = loss_hist = loss_hist_pos = None

        for epoch in range(args.n_epochs):
            optimizer.zero_grad()
            epoch_loss = 0.0
            epoch_samples = 0

            if sampling == 'poisson':
                def _poisson_iter():
                    for _ in range(_poisson_n_batches):
                        mask = torch.rand(n_samples) < _poisson_q
                        idx = torch.where(mask)[0]
                        if len(idx) == 0:
                            continue
                        yield X_tensor[idx], y_tensor[idx], idx
                batch_iter = _poisson_iter()
            else:
                batch_iter = loader

            for curr_X, curr_y, global_indices in batch_iter:
                curr_X = curr_X.to(device)
                curr_y = curr_y.to(device)
                global_indices = global_indices.to(device)

                defense_cfg = DefenseConfig(
                    score_fn=defense_score_fn,
                    score_norm=defense_score_norm,
                    grad_norm_hist=grad_norm_hist,
                    grad_norm_hist_pos=grad_norm_hist_pos,
                    grad_norm_percentile_k=args.grad_norm_percentile_k,
                    grad_dir_hist=grad_dir_hist,
                    grad_dir_hist_pos=grad_dir_hist_pos,
                    grad_dir_volatility_k=args.grad_dir_volatility_k,
                    grad_dir_proj=grad_dir_proj,
                    rand_proj_mat=rand_proj_mat,
                    rand_proj_var_m=args.rand_proj_var_m,
                    maxmin_proj_mat=maxmin_proj_mat,
                    maxmin_proj_k=args.maxmin_proj_k,
                    grad_rank_mode=args.grad_rank_mode,
                    grad_rank_eps=args.grad_rank_eps,
                    grad_accel_hist=grad_accel_hist,
                    grad_accel_hist_pos=grad_accel_hist_pos,
                    grad_accel_proj=grad_accel_proj,
                    grad_jerk_hist=grad_jerk_hist,
                    grad_jerk_hist_pos=grad_jerk_hist_pos,
                    grad_jerk_proj=grad_jerk_proj,
                    alignment_proj_mat=alignment_proj_mat,
                    alignment_proj_k=args.alignment_proj_k,
                    dir_unique_hist=dir_unique_hist,
                    dir_unique_hist_pos=dir_unique_hist_pos,
                    dir_unique_k=args.dir_unique_k,
                    grad_scatter_k=args.grad_scatter_k,
                    prev_losses=prev_losses,
                    loss_hist=loss_hist,
                    loss_hist_pos=loss_hist_pos,
                    loss_volatility_k=args.loss_volatility_k,
                )

                batch_indices = global_indices.cpu().numpy()

                # BUG FIX (Bug 2): take an explicit copy so that mutations made
                # inside clip_and_accum_grads are written back into batch_drop_mask
                # and then propagated back into the global drop_mask array below.
                batch_drop_mask = drop_mask[batch_indices].copy()

                curr_accumulated_gradients, scores = clip_and_accum_grads(
                    model, curr_X, curr_y, optimizer, criterion, max_grad_norm,
                    drop_mask=batch_drop_mask,
                    block_size=block_size,
                    scores=scores,
                    device=device,
                    global_indices=global_indices,
                    world_size=1,
                    rank=0,
                    batch_size=batch_size,
                    defense_cfg=defense_cfg,
                    defense_apply_ascent=defense_apply_ascent,
                )

                # BUG FIX (Bug 2 cont.): write batch_drop_mask mutations back to
                # the global drop_mask *before* we use it for the ascent promotion.
                drop_mask[batch_indices] = batch_drop_mask

                # Samples that received gradient ascent (1) are now permanently dropped (2)
                drop_mask[batch_indices[batch_drop_mask == 1]] = 2

                # Propagate lazily created projection matrices back from defense_cfg
                if defense_cfg.grad_dir_proj is not None:
                    grad_dir_proj = defense_cfg.grad_dir_proj
                if defense_cfg.rand_proj_mat is not None:
                    rand_proj_mat = defense_cfg.rand_proj_mat
                if defense_cfg.maxmin_proj_mat is not None:
                    maxmin_proj_mat = defense_cfg.maxmin_proj_mat
                if defense_cfg.alignment_proj_mat is not None:
                    alignment_proj_mat = defense_cfg.alignment_proj_mat
                if defense_cfg.grad_accel_proj is not None:
                    grad_accel_proj = defense_cfg.grad_accel_proj
                if defense_cfg.grad_jerk_proj is not None:
                    grad_jerk_proj = defense_cfg.grad_jerk_proj

                # Apply the accumulated gradients
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name not in curr_accumulated_gradients:
                            continue
                        grad = curr_accumulated_gradients[name].to(device)
                        grad.div_(float(batch_size))
                        if param.grad is None:
                            param.grad = grad.clone()
                        else:
                            param.grad.copy_(grad)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Track loss for logging (use criterion on the batch)
                with torch.no_grad():
                    batch_logits = model(curr_X)
                    batch_loss = criterion(batch_logits, curr_y)
                    epoch_loss += batch_loss.item() * curr_X.size(0)
                    epoch_samples += curr_X.size(0)

            # BUG FIX (Bug 3): flush all pending-ascent entries (1 → 2) before
            # building active_mask for the epoch-end per-class filter.  Without
            # this, samples flagged in the last batch of the epoch are still 1
            # and would be incorrectly included in the active set.
            drop_mask[drop_mask == 1] = 2

            # Filter top-k samples per class at end of each epoch
            if epoch % defense_filter_every == 0:
                active_mask = torch.from_numpy(drop_mask == 0)
                for cls in torch.unique(y_tensor):
                    cls_indices = (
                        (y_tensor == cls.item()) & active_mask
                    ).nonzero(as_tuple=True)[0]
                    if len(cls_indices) == 0:
                        continue
                    cls_scores = torch.tensor(scores[cls_indices.numpy()])
                    _, topk_idx = torch.topk(cls_scores, min(defense_k, len(cls_scores)))
                    drop_mask[cls_indices[topk_idx].numpy()] = 1
                scores.fill(0)

            n_dropped = int((drop_mask == 2).sum())
            if epoch_samples > 0:
                epoch_loss /= epoch_samples
            current_lr = lr_scheduler.get_last_lr()[0]
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{args.n_epochs}  loss={epoch_loss:.4f}  dropped={n_dropped}  lr={current_lr:.6f}")

    return model


# ============================================================================
# Audit Function
# ============================================================================

def audit_multi_canary(correctness_full, shadow_membership_mask, canary_indices, logreg_C, seed, alpha, delta):
    """
    Audit membership inference using LOO logistic regression per canary * per shadow model.

    Args:
        correctness_full: shape (num_canaries, num_shadow, 18)
        shadow_membership_mask: shape (num_samples, num_shadow)
        canary_indices: array/list of canary sample indices
        logreg_C: logistic regression C parameter
        seed: random seed for reproducibility
        alpha: significance level for empirical epsilon
        delta: DP delta parameter

    Returns:
        dict with metrics and scores
    """
    import sklearn.metrics
    num_canaries = len(canary_indices)
    num_shadow = correctness_full.shape[1]

    # Extract the membership mask specifically for the canary indices (1 for IN, 0 for OUT)
    canary_membership = shadow_membership_mask[canary_indices].astype(int)

    scores_raw = np.zeros((num_canaries, num_shadow), dtype=np.float32)

    for target_model_idx in range(num_shadow):
        for sample_idx in range(num_canaries):
            # train_ys: membership of this canary across all other shadow models (length num_shadow - 1)
            train_ys = np.delete(canary_membership[sample_idx], target_model_idx, axis=0)

            # train_xs: correctness features of this canary across all other shadow models (shape (num_shadow - 1, 18))
            train_xs = np.delete(correctness_full[sample_idx], target_model_idx, axis=0)

            # test_xs: correctness features of this canary on the target shadow model (shape (1, 18))
            test_xs = correctness_full[sample_idx, target_model_idx].reshape(1, -1)

            if len(np.unique(train_ys)) < 2:
                # Fallback if there is only one class in the training labels (e.g. during small sanity checks)
                scores_raw[sample_idx, target_model_idx] = float(train_ys[0])
            else:
                clf = LogisticRegression(
                    C=logreg_C,
                    penalty="l2",
                    random_state=seed,
                    warm_start=False,
                    max_iter=1000,
                    solver="lbfgs"
                )
                clf.fit(train_xs, train_ys)
                scores_raw[sample_idx, target_model_idx] = clf.predict_proba(test_xs)[0, 1]

    all_scores = scores_raw.flatten()
    all_labels = canary_membership.flatten()

    max_t, emp_eps = compute_eps_lower_from_mia(
        all_scores,
        all_labels,
        alpha,
        delta,
        method='GDP'
    )

    # Balanced accuracy calculation: threshold at median of all scores
    prediction_threshold = np.median(all_scores)
    pred_membership = all_scores > prediction_threshold
    balanced_accuracy = np.mean(pred_membership == all_labels)

    # TPR at FPR calculation
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true=all_labels, y_score=all_scores)
    tpr_at_fpr = {}
    target_fprs = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
    for target_fpr in target_fprs:
        valid_tpr = tpr[fpr <= target_fpr]
        tpr_at_fpr[target_fpr] = valid_tpr[-1] if len(valid_tpr) > 0 else 0.0

    return {
        'emp_eps': float(emp_eps) if emp_eps is not None else 0.0,
        'max_t': float(max_t) if max_t is not None else 0.0,
        'balanced_accuracy': float(balanced_accuracy),
        'tpr_at_fpr': tpr_at_fpr,
        'scores_in': all_scores[all_labels == 1],
        'scores_out': all_scores[all_labels == 0],
        'scores': all_scores,
        'labels': all_labels,
        'correctness_in': correctness_full[canary_membership == 1],
        'correctness_out': correctness_full[canary_membership == 0],
    }


# ============================================================================
# Utility Functions
# ============================================================================

def distribute_reps(n_reps, world_size):
    """Distribute reps across ranks (round-robin)."""
    reps_per_rank = [[] for _ in range(world_size)]
    for i in range(n_reps):
        reps_per_rank[i % world_size].append(i)
    return reps_per_rank


# ============================================================================
# Main
# ============================================================================

def main():
    parser = build_parser()

    # Add HAMP-specific arguments
    parser.add_argument('--defense_type', type=str, default='none',
                        choices=['none', 'hamp', 'hamp_testonly', 'filter'],
                        help='Defense type: none, hamp (train+test), hamp_testonly (test-time only), or filter')
    parser.add_argument('--hamp_gamma', type=float, default=0.95,
                        help='HAMP target entropy as fraction of max (0-1)')
    parser.add_argument('--hamp_alpha_entropy', type=float, default=1.0,
                        help='HAMP entropy regularization weight')
    
    # Paper-style multi-canary audit arguments
    parser.add_argument('--num_shadow', type=int, default=64,
                        help='Number of shadow models to train')
    parser.add_argument('--num_canaries', type=int, default=500,
                        help='Number of canaries to audit')
    parser.add_argument('--logreg_c', type=float, default=1.0,
                        help='Logistic regression regularization parameter C')
    parser.add_argument('--momentum', type=float, default=None,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='SGD weight decay')

    args = parser.parse_args()

    # Distributed setup
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    is_rank_zero = (rank == 0)

    seed = args.seed
    np.random.seed(seed + rank)

    # Load data
    n_df = None if args.n_df == 0 else args.n_df
    X, y, out_dim = load_data(args.data_name, n_df=n_df)
    X_test, y_test, _ = load_data(args.data_name, None, split='test')
    X = X.numpy() if isinstance(X, torch.Tensor) else X
    y = y.numpy() if isinstance(y, torch.Tensor) else y

    # Select canary indices and generate labels (mislabeled/label-noise or clean)
    rng = np.random.default_rng(seed)
    num_samples = len(X)
    
    canary_order = rng.permutation(num_samples)
    canary_indices = canary_order[:args.num_canaries]
    
    # Confirm canary generation and setup targets
    y_noisy = y.copy()
    if args.target_type in ('mislabeled', 'label_noise'):
        if is_rank_zero:
            print(f"[Confirm] Generating {args.num_canaries} mislabeled (label-noise) canaries by keeping features unchanged and randomly flipping labels to incorrect classes.")
        for idx in canary_indices:
            true_label = y[idx]
            available = [cls for cls in range(out_dim) if cls != true_label]
            y_noisy[idx] = rng.choice(available)
    elif args.target_type == 'blank':
        if is_rank_zero:
            print(f"[Confirm] Generating {args.num_canaries} blank image canaries.")
        for idx in canary_indices:
            X[idx] = np.zeros_like(X[idx])
            y_noisy[idx] = 0
    else:
        if is_rank_zero:
            print(f"[Confirm] Generating {args.num_canaries} clean canaries (no changes to features or labels).")

    # Generate shadow model membership splits (each sample IN for exactly half the models)
    rng_splits = np.random.default_rng(seed + 42)
    assert args.num_shadow % 2 == 0, "num_shadow must be even"
    uniforms = rng_splits.uniform(size=(args.num_shadow, num_samples))
    shadow_in_indices_t = np.argsort(uniforms, axis=0)[:args.num_shadow // 2].T
    
    shadow_membership_mask = np.zeros((num_samples, args.num_shadow), dtype=bool)
    for sample_idx in range(num_samples):
        shadow_membership_mask[sample_idx, shadow_in_indices_t[sample_idx]] = True
        
    # Force non-canaries to be IN for all shadow models
    canary_mask = np.zeros(num_samples, dtype=bool)
    canary_mask[canary_indices] = True
    shadow_membership_mask[~canary_mask] = True

    if is_rank_zero:
        print(f"Data: {args.data_name}, shape={X.shape}, out_dim={out_dim}")
        print(f"Canaries: {args.num_canaries} ({args.target_type})")
        print(f"Shadow Models: {args.num_shadow}")
        print(f"Defense: {args.defense_type}")
        print(f"Distributed: rank={rank}, world_size={world_size}")

    # Fixed init for synchronization across ranks
    torch.manual_seed(seed)
    init_model = Models[args.model_name](X.shape, out_dim=out_dim)
    if args.model_name == 'wideresnet':
        init_wideresnet(init_model)
    elif args.model_name == 'wideresnet_np':
        pass  # Use PyTorch default init to match misleading-privacy-evals
    else:
        xavier_init_model(init_model)
    torch.manual_seed(seed + rank)

    # Distribute shadow models
    reps_per_rank = distribute_reps(args.num_shadow, world_size)
    my_shadow_models = reps_per_rank[rank]

    local_binary_vectors = []
    local_train_accs = []
    local_test_accs = []

    for shadow_idx in my_shadow_models:
        if is_rank_zero:
            print(f"Training shadow model {shadow_idx + 1}/{args.num_shadow}...")

        # Get training subset where membership is True
        train_idx = np.where(shadow_membership_mask[:, shadow_idx])[0]
        X_train = X[train_idx]
        y_train = y_noisy[train_idx]

        model = copy.deepcopy(init_model)
        model.to(device)

        train_model(model, X_train, y_train, None, None, device, args,
                    defense_type=args.defense_type,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)

        # Compute train/test accuracy of this shadow model
        train_acc = test_model(model, torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        test_acc = test_model(model, X_test, y_test)
        local_train_accs.append(train_acc)
        local_test_accs.append(test_acc)

        # Evaluate correctness vector over 18 augmentations for all canaries
        model.eval()
        correctness_this_model = []
        use_flip = (args.data_name != 'mnist')
        # Match the exact random seed and spawning sequence of experiments/hamp.py
        shadow_seed = args.seed * (args.num_shadow + 1) + (shadow_idx + 1) # implemented this way in their codebase base.py in get_setting_seed()
        parent_rng = np.random.default_rng(seed=shadow_seed)
        rng_testtime, _ = parent_rng.spawn(2)

        with torch.no_grad():
            for c_idx in canary_indices:
                cx = torch.from_numpy(X[c_idx]).float()
                cy = int(y_noisy[c_idx])
                augmentations = generate_augmentations(cx, 18, use_flip=use_flip)
                binary_vec = generate_binary_correctness_vector(
                    model, cx, cy, augmentations, device,
                    defense_type=args.defense_type, rng=rng_testtime
                )
                correctness_this_model.append(binary_vec)

        local_binary_vectors.append(np.array(correctness_this_model, dtype=np.float32))

    # Save per-rank results
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(local_binary_vectors) > 0:
        np.save(output_dir / f'binary_vectors_rank{rank}.npy', np.array(local_binary_vectors))
        np.save(output_dir / f'shadow_indices_rank{rank}.npy', np.array(my_shadow_models))
        np.save(output_dir / f'train_accs_rank{rank}.npy', np.array(local_train_accs))
        np.save(output_dir / f'test_accs_rank{rank}.npy', np.array(local_test_accs))

    if world_size > 1:
        dist.barrier()

    # Rank 0: aggregate and audit
    if is_rank_zero:
        correctness_full = np.zeros((args.num_canaries, args.num_shadow, 18), dtype=np.float32)
        all_train_accs = np.zeros(args.num_shadow, dtype=np.float32)
        all_test_accs = np.zeros(args.num_shadow, dtype=np.float32)

        for r in range(world_size):
            path_vectors = output_dir / f'binary_vectors_rank{r}.npy'
            path_indices = output_dir / f'shadow_indices_rank{r}.npy'
            path_train_accs = output_dir / f'train_accs_rank{r}.npy'
            path_test_accs = output_dir / f'test_accs_rank{r}.npy'

            if path_vectors.exists() and path_indices.exists():
                local_vectors = np.load(path_vectors)
                local_indices = np.load(path_indices)

                for i, shadow_idx in enumerate(local_indices):
                    correctness_full[:, shadow_idx, :] = local_vectors[i]

                if path_train_accs.exists() and path_test_accs.exists():
                    local_train_accs = np.load(path_train_accs)
                    local_test_accs = np.load(path_test_accs)
                    for i, shadow_idx in enumerate(local_indices):
                        all_train_accs[shadow_idx] = local_train_accs[i]
                        all_test_accs[shadow_idx] = local_test_accs[i]

        # Audit with tuned C
        tuned_result = audit_multi_canary(
            correctness_full, shadow_membership_mask, canary_indices,
            logreg_C=args.logreg_c, seed=args.seed, alpha=args.alpha, delta=args.delta
        )

        # Audit with default C = 1.0
        default_result = audit_multi_canary(
            correctness_full, shadow_membership_mask, canary_indices,
            logreg_C=1.0, seed=args.seed, alpha=args.alpha, delta=args.delta
        )

        # Print results exactly like the paper's output
        print("=" * 60)
        print("AUDIT RESULTS (Leave-One-Out CV Logistic Regression)")
        print("=" * 60)
        print(f"Mean Train Accuracy of Shadow Models: {all_train_accs.mean() * 100:.2f}%")
        print(f"Mean Test Accuracy of Shadow Models:  {all_test_accs.mean() * 100:.2f}%")
        print("-" * 40)
        print(f"Tuned C ({args.logreg_c}):")
        print(f"  Balanced Accuracy : {tuned_result['balanced_accuracy']:.4f}")
        print(f"  Empirical Epsilon : {tuned_result['emp_eps']:.6f}")
        for fpr_val, tpr_val in tuned_result['tpr_at_fpr'].items():
            print(f"  TPR at FPR {fpr_val*100:.1f}% : {tpr_val*100:.4f}%")
            
        print("-" * 40)
        print("Default C (1.0):")
        print(f"  Balanced Accuracy : {default_result['balanced_accuracy']:.4f}")
        print(f"  Empirical Epsilon : {default_result['emp_eps']:.6f}")
        for fpr_val, tpr_val in default_result['tpr_at_fpr'].items():
            print(f"  TPR at FPR {fpr_val*100:.1f}% : {tpr_val*100:.4f}%")
        print("=" * 60)

        # Save files for compatibility with plotting / downstream scripts
        np.save(output_dir / 'binary_vectors_in.npy', tuned_result['correctness_in'])
        np.save(output_dir / 'binary_vectors_out.npy', tuned_result['correctness_out'])
        np.save(output_dir / 'emp_eps.npy', np.array(tuned_result['emp_eps']))
        np.save(output_dir / 'mia_scores.npy', tuned_result['scores'])
        np.save(output_dir / 'mia_labels.npy', tuned_result['labels'])
        np.save(output_dir / 'scores_in.npy',  tuned_result['scores_in'].astype(np.float32))
        np.save(output_dir / 'scores_out.npy', tuned_result['scores_out'].astype(np.float32))
        np.save(output_dir / 'shadow_train_accs.npy', all_train_accs)
        np.save(output_dir / 'shadow_test_accs.npy', all_test_accs)

        print(f"Outputs saved to {output_dir}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
