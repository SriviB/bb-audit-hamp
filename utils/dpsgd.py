"""
Utility functions to execute DP-SGD
"""
import torch
import torch.distributed as dist
import numpy as np
from torch.func import functional_call, vmap, grad
import matplotlib.pyplot as plt
import threading
import copy
from models.lstm import LSTM

try:
    from opacus.grad_sample import GradSampleModule as _GradSampleModule
    _OPACUS_AVAILABLE = True
except ImportError:
    _GradSampleModule = None
    _OPACUS_AVAILABLE = False
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

def preaugment_batch_vectorized(X, y, aug_fn, aug_mult):
    if aug_mult == 1:
        return aug_fn(X), y
    
    X_rep = X.repeat_interleave(aug_mult, dim=0)
    X_aug = aug_fn(X_rep)
    
    # Repeat labels: [B] -> [B * aug_mult]
    y_aug = y.repeat_interleave(aug_mult)
    
    return X_aug, y_aug


def average_grads_over_augmentations_optimized(ps_grads, batch_size, aug_mult):
    if aug_mult == 1:
        return ps_grads
    
    ps_grads_avg = {}
    for name, grad in ps_grads.items():
        param_dims = grad.shape[1:]
        grad_reshaped = grad.reshape(batch_size, aug_mult, *param_dims)
        ps_grads_avg[name] = grad_reshaped.mean(dim=1)
        
    return ps_grads_avg


# Drop-in replacement for your original functions
def preaugment_batch(X, y, aug_fn, aug_mult):
    """Original function signature maintained for compatibility"""
    return preaugment_batch_vectorized(X, y, aug_fn, aug_mult)


def average_grads_over_augmentations(ps_grads, batch_size, aug_mult):
    """Original function signature maintained for compatibility"""
    return average_grads_over_augmentations_optimized(ps_grads, batch_size, aug_mult)


def get_per_sample_grads(model, X, y, criterion):
    """Compute per-sample gradients"""
    # Check if model is DDP-wrapped
    is_ddp = hasattr(model, 'module')
    
    # Get model parameters, handling DDP case
    if is_ddp:
        # For DDP, we need to use the module's parameters but with the original names
        model_to_use = model.module
    else:
        model_to_use = model
    
    # map of parameter names : parameter values (without module prefix)
    params = {k: v.detach() for k, v in model_to_use.named_parameters()}
    # map of buffer names : buffer values (without module prefix)
    buffers = {k: v.detach() for k, v in model_to_use.named_buffers()}

    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        
        # Forward pass - no no_grad() here to allow gradient computation
        predictions = functional_call(model_to_use, (params, buffers), (batch,))
        loss = criterion(predictions, targets)
        return loss
    
    # Compute gradients
    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
    
    # Get gradients with consistent naming (without module prefix)
    ps_grads = ft_compute_sample_grad(params, buffers, X, y)
    
    return ps_grads


def get_per_sample_grads_and_losses(model, X, y, criterion):
    """Compute per-sample gradients and the per-sample losses used to compute them."""
    # Check if model is DDP-wrapped
    is_ddp = hasattr(model, 'module')

    # Get model parameters, handling DDP case
    if is_ddp:
        model_to_use = model.module
    else:
        model_to_use = model

    # map of parameter names : parameter values (without module prefix)
    params = {k: v.detach() for k, v in model_to_use.named_parameters()}
    # map of buffer names : buffer values (without module prefix)
    buffers = {k: v.detach() for k, v in model_to_use.named_buffers()}

    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = functional_call(model_to_use, (params, buffers), (batch,))
        loss = criterion(predictions, targets)
        return loss, loss

    ft_compute_grad = grad(compute_loss, has_aux=True)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    ps_grads, ps_losses = ft_compute_sample_grad(params, buffers, X, y)
    ps_losses = ps_losses.squeeze(-1).detach()

    return ps_grads, ps_losses


def compute_per_sample_losses_from_logits(logits, y):
    # Handle both classification (B, C) and sequence modeling (B, T, C)
    if y.ndim == 2 and logits.ndim == 3:
        b, t, c = logits.shape
        token_losses = F.cross_entropy(
            logits.reshape(b * t, c),
            y.reshape(b * t),
            reduction='none'
        ).reshape(b, t)
        return token_losses.mean(dim=1)

    return F.cross_entropy(logits, y, reduction='none')


def compute_per_sample_inverse_confidence_from_logits(logits):
    # Handle both classification (B, C) and sequence modeling (B, T, C)
    if logits.ndim == 3:
        probs = logits.softmax(dim=-1)
        max_probs = probs.max(dim=-1).values
        return (1.0 - max_probs).mean(dim=1)

    probs = logits.softmax(dim=-1)
    max_probs = probs.max(dim=-1).values
    return 1.0 - max_probs


def compute_per_sample_prediction_margin_from_logits(logits, y):
    # Handle both classification (B, C) and sequence modeling (B, T, C)
    # Returns INVERTED margin: high scores = uncertain samples (small margin)
    if logits.ndim == 3:
        # y: (B, T)
        probs = logits.softmax(dim=-1)
        p_true = probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)

        # max over classes excluding the true class
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(dim=-1, index=y.unsqueeze(-1), value=True)
        max_other = probs.masked_fill(mask, float('-inf')).max(dim=-1).values

        margin = p_true - max_other
        # Invert: high scores for uncertain samples (small margin)
        return -margin.mean(dim=1)

    # logits: (B, C), y: (B,)
    probs = logits.softmax(dim=-1)
    p_true = probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)

    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(dim=-1, index=y.unsqueeze(-1), value=True)
    max_other = probs.masked_fill(mask, float('-inf')).max(dim=-1).values

    # Invert: high scores for uncertain samples (small margin)
    return -(p_true - max_other)


def compute_per_sample_prediction_entropy_from_logits(logits):
    # Handle both classification (B, C) and sequence modeling (B, T, C)
    if logits.ndim == 3:
        probs = logits.softmax(dim=-1)
        log_probs = torch.log(probs + 1e-12)
        ent = -(probs * log_probs).sum(dim=-1)
        return ent.mean(dim=1)

    probs = logits.softmax(dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return -(probs * log_probs).sum(dim=-1)


@dataclass
class DefenseConfig:
    score_fn: str = 'grad_norm'
    score_norm: str = 'linf'
    delta_theta: Optional[torch.Tensor] = None
    theta_t_minus_theta0: Optional[torch.Tensor] = None
    grad_norm_hist: Optional[np.ndarray] = None
    grad_norm_hist_pos: Optional[np.ndarray] = None
    grad_norm_percentile_k: int = 20
    grad_dir_hist: Optional[np.ndarray] = None
    grad_dir_hist_pos: Optional[np.ndarray] = None
    grad_dir_volatility_k: int = 5
    grad_dir_proj: Optional[torch.Tensor] = None
    rand_proj_mat: Optional[torch.Tensor] = None
    rand_proj_var_m: int = 10
    maxmin_proj_mat: Optional[torch.Tensor] = None
    maxmin_proj_k: int = 10
    grad_rank_mode: str = 'effdim'
    grad_rank_eps: float = 1e-12
    grad_accel_hist: Optional[np.ndarray] = None
    grad_accel_hist_pos: Optional[np.ndarray] = None
    grad_accel_proj: Optional[torch.Tensor] = None
    grad_jerk_hist: Optional[np.ndarray] = None
    grad_jerk_hist_pos: Optional[np.ndarray] = None
    grad_jerk_proj: Optional[torch.Tensor] = None
    dir_unique_hist: Optional[np.ndarray] = None
    alignment_proj_mat: Optional[torch.Tensor] = None
    alignment_proj_k: int = 10
    dir_unique_hist_pos: Optional[np.ndarray] = None
    dir_unique_k: int = 5
    prev_grad_dir: Optional[np.ndarray] = None
    grad_scatter_hist: Optional[np.ndarray] = None
    grad_scatter_hist_pos: Optional[np.ndarray] = None
    grad_scatter_k: int = 5
    prev_losses: Optional[np.ndarray] = None
    loss_hist: Optional[np.ndarray] = None
    loss_hist_pos: Optional[np.ndarray] = None
    loss_volatility_k: int = 5


def _norm_p_from_name(score_norm: str):
    if score_norm == 'linf':
        return float('inf')
    if score_norm == 'l2':
        return 2
    if score_norm == 'l1':
        return 1
    raise ValueError(f"Unsupported defense_score_norm: {score_norm}")


def _flatten_per_sample_grads(ps_grads):
    """
    Flatten per-sample gradients in a sequence-aware manner.
    
    For regular models: gradients have shape (B, ...) -> flatten to (B, D)
    For sequence models (LSTM): gradients have shape (B, T, ...) -> average over T, then flatten to (B, D)
    
    Note: CNN conv layer gradients have shape (B, C_out, C_in, H, W) which should NOT be averaged.
    Only LSTM/RNN gradients with explicit sequence dimension should be averaged.
    
    Args:
        ps_grads: dict of {param_name: gradient_tensor}
    
    Returns:
        Tensor of shape (B, D) where D is the total flattened gradient dimension
    """
    flattened_grads = []
    for name, g in ps_grads.items():
        # Only average over sequence dimension for LSTM/RNN parameters
        # LSTM parameters typically have 'lstm' or 'rnn' in their name
        # and their gradients have shape (B, T, ...)
        if g.ndim >= 3 and ('lstm' in name.lower() or 'rnn' in name.lower()):
            # Sequence model: (B, T, ...) -> average over sequence dimension, then flatten
            # This gives us a per-sample gradient by averaging over the sequence
            g_avg = g.mean(dim=1)  # (B, T, ...) -> (B, ...)
            flattened_grads.append(g_avg.reshape(g.shape[0], -1))
        else:
            # Regular model (including CNN): (B, ...) -> flatten
            flattened_grads.append(g.reshape(g.shape[0], -1))
    
    return torch.cat(flattened_grads, dim=1)


def _get_flattened_param_shape(ps_grads):
    """
    Get the shape that parameters should be flattened to, matching gradient flattening.
    This is needed because sequence model gradients are averaged over time dimension.
    
    Args:
        ps_grads: dict of {param_name: gradient_tensor} - used to determine which params have time dimension
    
    Returns:
        List of (param_name, flattened_size) tuples
    """
    param_shapes = []
    for name, g in ps_grads.items():
        if g.ndim >= 3:
            # Sequence model gradient: (B, T, ...) -> after averaging becomes (B, ...)
            # So parameter should be flattened to size of (...)
            flat_size = g.shape[2:].numel() if len(g.shape) > 2 else g.shape[1]
        else:
            # Regular model: (B, ...) -> parameter flattened to size of (...)
            flat_size = g.shape[1:].numel() if len(g.shape) > 1 else 1
        param_shapes.append((name, flat_size))
    return param_shapes


def _flatten_param_diff_like_grads(param_diff_dict, ps_grads):
    """
    Flatten parameter differences in the same way as gradients.
    This ensures dimensional compatibility when comparing param diffs with gradients.
    
    Args:
        param_diff_dict: dict of {param_name: parameter_difference_tensor}
        ps_grads: dict of {param_name: gradient_tensor} - used to determine flattening strategy
    
    Returns:
        Flattened tensor of shape (D,) where D matches _flatten_per_sample_grads output
    """
    flattened = []
    for name in ps_grads.keys():
        if name not in param_diff_dict:
            continue
        param_diff = param_diff_dict[name]
        
        # For LSTM/RNN parameters, we don't need special handling since parameter
        # differences don't have batch or time dimensions - they're just parameter tensors
        # Just flatten everything
        flattened.append(param_diff.reshape(-1))
    
    return torch.cat(flattened, dim=0)


def compute_defense_scores(ps_grads, ps_grads_clipped, y, defense_cfg: DefenseConfig, ps_losses=None, ps_logits=None):
    score_fn = defense_cfg.score_fn

    # Any gradient-based score function accepts an _unclipped suffix to use raw
    # (pre-clip) per-sample gradients instead of the clipped ones.
    # Loss- and logit-based functions are unaffected by this suffix.
    use_unclipped = score_fn.endswith('_unclipped')
    if use_unclipped:
        score_fn = score_fn[:-len('_unclipped')]
    ps_grads_used = ps_grads if use_unclipped else ps_grads_clipped

    if score_fn == 'loss':
        if ps_losses is None:
            raise RuntimeError("defense_score_fn='loss' requires per-sample losses")
        return ps_losses.to(dtype=torch.float32)

    if score_fn == 'loss_momentum':
        if ps_losses is None:
            raise RuntimeError("defense_score_fn='loss_momentum' requires per-sample losses")
        return ps_losses.to(dtype=torch.float32)

    if score_fn == 'loss_volatility':
        if ps_losses is None:
            raise RuntimeError("defense_score_fn='loss_volatility' requires per-sample losses")
        return ps_losses.to(dtype=torch.float32)

    if score_fn == 'grad_norm_x_loss':
        if ps_losses is None:
            raise RuntimeError("defense_score_fn='grad_norm_x_loss' requires per-sample losses")
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads_used)
        grad_norms = per_sample_flat_grads.norm(2, dim=1)
        return (grad_norms * ps_losses).to(dtype=torch.float32)

    if score_fn == 'inv_confidence':
        if ps_logits is None:
            raise RuntimeError("defense_score_fn='inv_confidence' requires logits")
        return compute_per_sample_inverse_confidence_from_logits(ps_logits).to(dtype=torch.float32)

    if score_fn == 'prediction_margin':
        if ps_logits is None:
            raise RuntimeError("defense_score_fn='prediction_margin' requires logits")
        return compute_per_sample_prediction_margin_from_logits(ps_logits, y).to(dtype=torch.float32)

    if score_fn == 'pred_entropy':
        if ps_logits is None:
            raise RuntimeError("defense_score_fn='pred_entropy' requires logits")
        return compute_per_sample_prediction_entropy_from_logits(ps_logits).to(dtype=torch.float32)

    if score_fn == 'cos_update':
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads_used)
        delta_theta_dict = defense_cfg.delta_theta
        if delta_theta_dict is None:
            return torch.zeros((per_sample_flat_grads.shape[0],), device=per_sample_flat_grads.device, dtype=torch.float32)

        # Flatten delta_theta in the same way as gradients
        delta_theta = _flatten_param_diff_like_grads(delta_theta_dict, ps_grads_used)
        delta_theta = delta_theta.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        delta_norm = delta_theta.norm(2)
        if float(delta_norm) < 1e-8:
            return torch.zeros((per_sample_flat_grads.shape[0],), device=per_sample_flat_grads.device, dtype=torch.float32)

        per_norms = per_sample_flat_grads.norm(2, dim=1) + 1e-12
        cos_sims = (per_sample_flat_grads @ delta_theta) / (per_norms * delta_norm)
        return cos_sims.abs().to(dtype=torch.float32)

    if score_fn == 'cos_theta0':
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads_used)
        theta_t_minus_theta0_dict = defense_cfg.theta_t_minus_theta0
        if theta_t_minus_theta0_dict is None:
            return torch.zeros((per_sample_flat_grads.shape[0],), device=per_sample_flat_grads.device, dtype=torch.float32)

        # Flatten theta_t_minus_theta0 in the same way as gradients
        theta_t_minus_theta0 = _flatten_param_diff_like_grads(theta_t_minus_theta0_dict, ps_grads_used)
        theta_t_minus_theta0 = theta_t_minus_theta0.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        delta_norm = theta_t_minus_theta0.norm(2)
        if float(delta_norm) < 1e-8:
            return torch.zeros((per_sample_flat_grads.shape[0],), device=per_sample_flat_grads.device, dtype=torch.float32)

        per_norms = per_sample_flat_grads.norm(2, dim=1) + 1e-12
        cos_sims = (per_sample_flat_grads @ theta_t_minus_theta0) / (per_norms * delta_norm)
        return cos_sims.abs().to(dtype=torch.float32)

    if score_fn == 'fisher':
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads_used)
        return (per_sample_flat_grads ** 2).sum(dim=1).to(dtype=torch.float32)

    if score_fn == 'rand_proj_var':
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads_used)

        # Create projection matrix lazily based on actual gradient dimensions
        if defense_cfg.rand_proj_mat is None:
            flat_dim = per_sample_flat_grads.shape[1]
            m = defense_cfg.rand_proj_var_m
            gen = torch.Generator(device='cpu')
            gen.manual_seed(0)
            defense_cfg.rand_proj_mat = torch.randn((flat_dim, m), generator=gen, dtype=torch.float32)
            defense_cfg.rand_proj_mat = defense_cfg.rand_proj_mat / (defense_cfg.rand_proj_mat.norm(2, dim=0, keepdim=True) + 1e-12)

        proj_mat = defense_cfg.rand_proj_mat.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        projections = per_sample_flat_grads @ proj_mat
        mean_abs = projections.abs().mean(dim=1)
        std = projections.std(dim=1, unbiased=False)
        return (mean_abs * std).to(dtype=torch.float32)

    if score_fn == 'maxmin_proj_ratio':
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads_used)

        # Create projection matrix lazily based on actual gradient dimensions
        if defense_cfg.maxmin_proj_mat is None:
            flat_dim = per_sample_flat_grads.shape[1]
            k = defense_cfg.maxmin_proj_k
            gen = torch.Generator(device='cpu')
            gen.manual_seed(0)
            defense_cfg.maxmin_proj_mat = torch.randn((flat_dim, k), generator=gen, dtype=torch.float32)
            defense_cfg.maxmin_proj_mat = defense_cfg.maxmin_proj_mat / (defense_cfg.maxmin_proj_mat.norm(2, dim=0, keepdim=True) + 1e-12)

        proj_mat = defense_cfg.maxmin_proj_mat.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        projections = (per_sample_flat_grads @ proj_mat).abs()
        max_proj = projections.max(dim=1).values
        min_proj = projections.min(dim=1).values
        # Clamp min_proj to prevent unbounded explosion and handle near-zero case
        min_proj_clamped = torch.clamp(min_proj, min=1e-6)
        ratio = max_proj / min_proj_clamped
        # Cap the ratio to prevent extreme outliers from dominating
        ratio = torch.clamp(ratio, max=1e6)
        return ratio.to(dtype=torch.float32)

    if score_fn == 'gradient_rank':
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads_used)
        d = float(per_sample_flat_grads.shape[1])
        eps = float(defense_cfg.grad_rank_eps)

        abs_g = per_sample_flat_grads.abs()
        l1 = abs_g.sum(dim=1)
        l2 = per_sample_flat_grads.norm(2, dim=1)

        mode = str(defense_cfg.grad_rank_mode)
        if mode == 'effdim':
            effdim = (l1 / (l2 + eps)) ** 2
            return (effdim / (d + eps)).to(dtype=torch.float32)

        if mode == 'entropy':
            # Handle zero-gradient case: when l1 is very small, entropy is undefined
            zero_grad = l1 < eps
            probs = abs_g / (l1[:, None] + eps)
            entropy = -(probs * torch.log(probs + eps)).sum(dim=1)
            # Set entropy to 0 for zero-gradient samples
            entropy = torch.where(zero_grad, torch.zeros_like(entropy), entropy)
            return (l2 * entropy).to(dtype=torch.float32)

        raise ValueError(f"Unsupported grad_rank_mode: {mode}")

    if score_fn == 'alignment_with_rand_proj':
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads_used)

        # Create projection matrix lazily based on actual gradient dimensions
        if defense_cfg.alignment_proj_mat is None:
            flat_dim = per_sample_flat_grads.shape[1]
            k = defense_cfg.alignment_proj_k
            gen = torch.Generator(device='cpu')
            gen.manual_seed(0)
            defense_cfg.alignment_proj_mat = torch.randn((flat_dim, k), generator=gen, dtype=torch.float32)
            defense_cfg.alignment_proj_mat = defense_cfg.alignment_proj_mat / (defense_cfg.alignment_proj_mat.norm(2, dim=0, keepdim=True) + 1e-12)

        proj_mat = defense_cfg.alignment_proj_mat.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)

        grad_norms = per_sample_flat_grads.norm(2, dim=1, keepdim=True) + 1e-12
        normalized_grads = per_sample_flat_grads / grad_norms

        cos_sims = normalized_grads @ proj_mat
        alignment_std = cos_sims.std(dim=1, unbiased=False)
        return alignment_std.to(dtype=torch.float32)

    if score_fn == 'gradient_sparsity':
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads_used)
        l1_norm = per_sample_flat_grads.abs().sum(dim=1)
        l2_norm = per_sample_flat_grads.norm(2, dim=1)
        return (l1_norm / (l2_norm + 1e-12)).to(dtype=torch.float32)

    if score_fn == 'gradient_kurtosis':
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads_used)
        mean_g = per_sample_flat_grads.mean(dim=1, keepdim=True)
        std_g = per_sample_flat_grads.std(dim=1, keepdim=True, unbiased=False)

        # Avoid overflow for near-constant gradients by using a larger epsilon and clamping
        std_threshold = 1e-6
        normalized = (per_sample_flat_grads - mean_g) / torch.clamp(std_g, min=std_threshold)
        # Clamp normalized values to prevent overflow in ** 4
        normalized = torch.clamp(normalized, min=-100.0, max=100.0)
        kurtosis = (normalized ** 4).mean(dim=1)
        return kurtosis.to(dtype=torch.float32)

    if score_fn == 'grad_dir_change_rate':
        raise RuntimeError(
            "defense_score_fn='grad_dir_change_rate' is history-based and must be computed in clip_and_accum_grads(...), keyed by global_indices"
        )

    if score_fn == 'norm_x_trajectory_orth':
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads_used)
        theta_t_minus_theta0_dict = defense_cfg.theta_t_minus_theta0
        if theta_t_minus_theta0_dict is None:
            return torch.zeros((per_sample_flat_grads.shape[0],), device=per_sample_flat_grads.device, dtype=torch.float32)

        # Flatten theta_t_minus_theta0 in the same way as gradients
        theta_t_minus_theta0 = _flatten_param_diff_like_grads(theta_t_minus_theta0_dict, ps_grads_used)
        theta_t_minus_theta0 = theta_t_minus_theta0.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        trajectory_norm = theta_t_minus_theta0.norm(2)
        # Use threshold instead of exact zero check to catch near-zero trajectories
        if float(trajectory_norm) < 1e-8:
            return torch.zeros((per_sample_flat_grads.shape[0],), device=per_sample_flat_grads.device, dtype=torch.float32)

        overall_direction = theta_t_minus_theta0 / trajectory_norm
        grad_norms = per_sample_flat_grads.norm(2, dim=1)
        normalized_grads = per_sample_flat_grads / (grad_norms[:, None] + 1e-12)

        cos_sims = (normalized_grads @ overall_direction).abs()
        orthogonality = 1.0 - cos_sims

        return (grad_norms * orthogonality).to(dtype=torch.float32)

    if score_fn == 'gradient_scatter':
        raise RuntimeError(
            "defense_score_fn='gradient_scatter' is history-based and must be computed in clip_and_accum_grads(...), keyed by global_indices"
        )

    # Default: grad_norm — uses clipped or unclipped depending on ps_grads_used
    per_sample_flat_grads = _flatten_per_sample_grads(ps_grads_used)
    p = _norm_p_from_name(defense_cfg.score_norm)
    return per_sample_flat_grads.norm(p, dim=1).to(dtype=torch.float32)

def get_per_sample_grad_norms(per_sample_grads):
    """
    Compute L2 norms of per-sample gradients in a sequence-aware manner.
    
    For regular models: gradients have shape (B, ...) -> compute norm over all dims except batch
    For sequence models (LSTM): gradients have shape (B, T, ...) -> average over T first, then compute norm
    
    Note: CNN conv layer gradients have shape (B, C_out, C_in, H, W) which should NOT be averaged.
    Only LSTM/RNN gradients with explicit sequence dimension should be averaged.
    """
    norms_per_param = []
    for name, curr_grad in per_sample_grads.items():
        # Only average over sequence dimension for LSTM/RNN parameters
        if curr_grad.ndim >= 3 and ('lstm' in name.lower() or 'rnn' in name.lower()):
            # Sequence model: average over sequence dimension first
            curr_grad = curr_grad.mean(dim=1)  # (B, T, ...) -> (B, ...)
        # Flatten and compute norm for this parameter
        norms_per_param.append(curr_grad.flatten(start_dim=1).norm(2, dim=1))
    
    # Combine norms across all parameters
    return torch.vstack(norms_per_param).norm(2, dim=0)


def clip_per_sample_grads(per_sample_grads, max_grad_norm):
    """Clip per-sample gradients to clipping norm"""
    ps_grad_norms = get_per_sample_grad_norms(per_sample_grads)

    ps_grad_scales = 1 / torch.maximum(
        torch.ones_like(ps_grad_norms),
        ps_grad_norms / max_grad_norm
    )

    ps_grads_clipped = {
        # broadcast
        name: curr_grad * ps_grad_scales[(...,) + (None,) * (curr_grad.dim() - 1)]
        for name, curr_grad in per_sample_grads.items() 
    }

    ps_grad_norms_clipped = get_per_sample_grad_norms(ps_grads_clipped)

    return ps_grads_clipped, { 'before': ps_grad_norms.cpu().numpy(), 'after': ps_grad_norms_clipped.cpu().numpy() }

def _get_per_sample_grads(model, X, y, criterion):
    model.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    ps_grads = {name: param.grad_sample for name, param in model.named_parameters()}
    return ps_grads

def clip_and_accum_grads_block(model, X, y, optimizer, criterion, max_grad_norm, device='cuda', aug_fn=None, aug_mult=1, 
                             is_gradient_space_canary=False, crafted_gradient=None, canary_local_idx=None, curr_gradient_ascent_indices=None, defense_cfg: Optional[DefenseConfig] = None, defense_score_norm='linf', defense_score_fn='grad_norm', delta_theta=None, theta_t_minus_theta0=None, defense_apply_ascent=True):
    """
    Add aug_fn and aug_mult params to support augmentation multiplicity outside vmap.

    If aug_mult > 1, apply augmentation multiplicity outside, then average grads.
    """
    optimizer.zero_grad()
    
    # Check if model is DDP-wrapped
    is_ddp = hasattr(model, 'module')
    
    # Get the actual model (unwrapped if DDP)
    model_to_use = model.module if is_ddp else model
    
    # Get parameter names without 'module.' prefix
    param_names = [name.replace('module.', '') for name in model.state_dict().keys() 
                  if not name.startswith('_forward_hooks') and not name.startswith('_backward_hooks')]
    
    if defense_cfg is None:
        defense_cfg = DefenseConfig(
            score_fn=defense_score_fn,
            score_norm=defense_score_norm,
            delta_theta=delta_theta,
            theta_t_minus_theta0=theta_t_minus_theta0
        )

    ps_losses = None
    ps_logits = None
    if len(X) == 0:
        # Initialize zero gradients with correct names
        ps_grads = {name: torch.zeros_like(param).unsqueeze(dim=0) 
                   for name, param in model_to_use.named_parameters()}
    else:
        # Compute per-sample gradients
        # Pre-augment outside vmap if aug_mult > 1
        if aug_mult > 1 and aug_fn is not None:
            X_aug, y_aug = preaugment_batch(X, y, aug_fn, aug_mult)
            X_aug = X_aug.to(device)
            y_aug = y_aug.to(device)

            if defense_cfg.score_fn == 'inv_confidence':
                logits = model(X_aug)
                aug_inv_conf = compute_per_sample_inverse_confidence_from_logits(logits)
                ps_logits = logits
                ps_losses = None

            if defense_cfg.score_fn == 'prediction_margin':
                logits = model(X_aug)
                aug_margin = compute_per_sample_prediction_margin_from_logits(logits, y_aug)
                ps_logits = logits
                ps_losses = None

            if defense_cfg.score_fn == 'pred_entropy':
                logits = model(X_aug)
                aug_entropy = compute_per_sample_prediction_entropy_from_logits(logits)
                ps_logits = logits
                ps_losses = None

            if defense_cfg.score_fn in {'loss', 'loss_momentum', 'loss_volatility', 'grad_norm_x_loss'}:
                logits = model(X_aug)
                aug_losses = compute_per_sample_losses_from_logits(logits, y_aug)
                ps_losses = aug_losses
            else:
                ps_losses = None

            if isinstance(model_to_use, LSTM):
                if not _OPACUS_AVAILABLE:
                    raise RuntimeError("LSTM per-sample gradients require opacus: pip install opacus")
                ps_grads = _get_per_sample_grads(_GradSampleModule(model), X_aug, y_aug, criterion)
            else:
                ps_grads = get_per_sample_grads(model, X_aug, y_aug, criterion)

            if defense_cfg.score_fn == 'inv_confidence':
                ps_losses = None
                ps_logits = None
                ps_inv_conf = aug_inv_conf.reshape(len(X), aug_mult).mean(dim=1).detach()
                ps_margin = None
                ps_entropy = None
            elif defense_cfg.score_fn == 'prediction_margin':
                ps_losses = None
                ps_logits = None
                ps_margin = aug_margin.reshape(len(X), aug_mult).mean(dim=1).detach()
                ps_inv_conf = None
                ps_entropy = None
            elif defense_cfg.score_fn == 'pred_entropy':
                ps_losses = None
                ps_logits = None
                ps_entropy = aug_entropy.reshape(len(X), aug_mult).mean(dim=1).detach()
                ps_inv_conf = None
                ps_margin = None
            elif defense_cfg.score_fn in {'loss', 'loss_momentum', 'loss_volatility', 'grad_norm_x_loss'}:
                ps_losses = aug_losses.reshape(len(X), aug_mult).mean(dim=1).detach()
                ps_logits = None
                ps_inv_conf = None
                ps_margin = None
                ps_entropy = None
            else:
                ps_losses = None
                ps_logits = None
                ps_inv_conf = None
                ps_margin = None
                ps_entropy = None

            ps_grads = average_grads_over_augmentations(ps_grads, batch_size=len(X), aug_mult=aug_mult)
        else:
            X = X.to(device)
            y = y.to(device)

            if defense_cfg.score_fn == 'inv_confidence':
                logits = model(X)
                ps_inv_conf = compute_per_sample_inverse_confidence_from_logits(logits).detach()
            else:
                ps_inv_conf = None

            if defense_cfg.score_fn == 'prediction_margin':
                logits = model(X)
                ps_margin = compute_per_sample_prediction_margin_from_logits(logits, y).detach()
            else:
                ps_margin = None

            if defense_cfg.score_fn == 'pred_entropy':
                logits = model(X)
                ps_entropy = compute_per_sample_prediction_entropy_from_logits(logits).detach()
            else:
                ps_entropy = None

            if defense_cfg.score_fn in {'loss', 'loss_momentum', 'loss_volatility', 'grad_norm_x_loss'}:
                logits = model(X)
                ps_losses = compute_per_sample_losses_from_logits(logits, y).detach()
            else:
                ps_losses = None

            if isinstance(model_to_use, LSTM):
                if not _OPACUS_AVAILABLE:
                    raise RuntimeError("LSTM per-sample gradients require opacus: pip install opacus")
                ps_grads = _get_per_sample_grads(_GradSampleModule(model), X, y, criterion)
            else:
                ps_grads = get_per_sample_grads(model, X, y, criterion)
        
        # Apply gradient-space audit after getting the gradients but before clipping
        if is_gradient_space_canary and crafted_gradient is not None:
            # crafted_gradient is now a dict mapping local indices to gradient dicts
            if isinstance(crafted_gradient, dict):
                for local_idx, grad_dict in crafted_gradient.items():
                    # Replace the gradient for this canary
                    # Assume grad_dict has all the same keys as ps_grads for efficiency
                    for name, grad_tensor in grad_dict.items():
                        ps_grads[name][local_idx] = grad_tensor
            
    if max_grad_norm is not None:
        ps_grads_clipped, _ = clip_per_sample_grads(ps_grads, max_grad_norm)
    else:
        ps_grads_clipped = ps_grads

    # last_layer_name = list(model.net.named_modules())[-1][0]
    # last_w_name = 'net.' + last_layer_name + '.weight'
    # last_b_name = 'net.' + last_layer_name + '.bias'

    if defense_cfg.score_fn == 'inv_confidence':
        scores_block = ps_inv_conf.to(dtype=torch.float32)
    elif defense_cfg.score_fn == 'prediction_margin':
        scores_block = ps_margin.to(dtype=torch.float32)
    elif defense_cfg.score_fn == 'pred_entropy':
        scores_block = ps_entropy.to(dtype=torch.float32)
    elif defense_cfg.score_fn in {'norm_x_dir_uniqueness', 'norm_x_dir_uniqueness_unclipped'}:
        use_unclipped = defense_cfg.score_fn.endswith('_unclipped')
        p = _norm_p_from_name(defense_cfg.score_norm)
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads if use_unclipped else ps_grads_clipped)
        scores_block = per_sample_flat_grads.norm(p=p, dim=1).to(dtype=torch.float32)
    elif defense_cfg.score_fn in {'grad_dir_change_rate', 'grad_dir_change_rate_unclipped',
                                   'gradient_scatter', 'gradient_scatter_unclipped'}:
        scores_block = torch.zeros((len(X),), device=device, dtype=torch.float32)
    else:
        scores_block = compute_defense_scores(ps_grads, ps_grads_clipped, y, defense_cfg, ps_losses=ps_losses, ps_logits=ps_logits)

    aux_embeds_block = None
    if defense_cfg is not None and defense_cfg.score_fn in {
        'grad_dir_volatility', 'grad_dir_volatility_unclipped',
        'norm_x_dir_uniqueness', 'norm_x_dir_uniqueness_unclipped',
    }:
        use_unclipped = defense_cfg.score_fn.endswith('_unclipped')
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads if use_unclipped else ps_grads_clipped)
        per_sample_flat_grads = per_sample_flat_grads / (per_sample_flat_grads.norm(2, dim=1, keepdim=True) + 1e-12)

        # Create projection matrix lazily based on actual gradient dimensions
        if defense_cfg.grad_dir_proj is None:
            flat_dim = per_sample_flat_grads.shape[1]
            proj_dim = defense_cfg.grad_dir_hist.shape[2] if defense_cfg.grad_dir_hist is not None else 64
            gen = torch.Generator(device='cpu')
            gen.manual_seed(0)  # Use consistent seed
            defense_cfg.grad_dir_proj = torch.randn((flat_dim, proj_dim), generator=gen, dtype=torch.float32)
            defense_cfg.grad_dir_proj = defense_cfg.grad_dir_proj / (defense_cfg.grad_dir_proj.norm(2, dim=0, keepdim=True) + 1e-12)

        proj = defense_cfg.grad_dir_proj.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        embeds = per_sample_flat_grads @ proj
        embeds = embeds / (embeds.norm(2, dim=1, keepdim=True) + 1e-12)
        aux_embeds_block = embeds.detach().cpu().numpy().astype(np.float32, copy=False)

    if defense_cfg is not None and defense_cfg.score_fn in {'grad_dir_change_rate', 'grad_dir_change_rate_unclipped'}:
        use_unclipped = defense_cfg.score_fn.endswith('_unclipped')
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads if use_unclipped else ps_grads_clipped)
        per_sample_flat_grads = per_sample_flat_grads / (per_sample_flat_grads.norm(2, dim=1, keepdim=True) + 1e-12)
        aux_embeds_block = per_sample_flat_grads.detach().cpu().numpy().astype(np.float32, copy=False)

    if defense_cfg is not None and defense_cfg.score_fn in {'gradient_scatter', 'gradient_scatter_unclipped'}:
        use_unclipped = defense_cfg.score_fn.endswith('_unclipped')
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads if use_unclipped else ps_grads_clipped)
        aux_embeds_block = per_sample_flat_grads.detach().cpu().numpy().astype(np.float32, copy=False)

    if defense_cfg is not None and defense_cfg.score_fn in {'grad_accel', 'grad_accel_unclipped'}:
        use_unclipped = defense_cfg.score_fn.endswith('_unclipped')
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads if use_unclipped else ps_grads_clipped)
        
        # Create projection matrix lazily based on actual gradient dimensions
        if defense_cfg.grad_accel_proj is None:
            flat_dim = per_sample_flat_grads.shape[1]
            proj_dim = defense_cfg.grad_accel_hist.shape[2] if defense_cfg.grad_accel_hist is not None else 64
            gen = torch.Generator(device='cpu')
            gen.manual_seed(0)
            defense_cfg.grad_accel_proj = torch.randn((flat_dim, proj_dim), generator=gen, dtype=torch.float32)
            defense_cfg.grad_accel_proj = defense_cfg.grad_accel_proj / (defense_cfg.grad_accel_proj.norm(2, dim=0, keepdim=True) + 1e-12)
        
        proj = defense_cfg.grad_accel_proj.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        embeds = per_sample_flat_grads @ proj
        aux_embeds_block = embeds.detach().cpu().numpy().astype(np.float32, copy=False)

    if defense_cfg is not None and defense_cfg.score_fn in {'grad_jerk', 'grad_jerk_unclipped'}:
        use_unclipped = defense_cfg.score_fn.endswith('_unclipped')
        per_sample_flat_grads = _flatten_per_sample_grads(ps_grads if use_unclipped else ps_grads_clipped)
        
        # Create projection matrix lazily based on actual gradient dimensions
        if defense_cfg.grad_jerk_proj is None:
            flat_dim = per_sample_flat_grads.shape[1]
            proj_dim = defense_cfg.grad_jerk_hist.shape[2] if defense_cfg.grad_jerk_hist is not None else 64
            gen = torch.Generator(device='cpu')
            gen.manual_seed(0)
            defense_cfg.grad_jerk_proj = torch.randn((flat_dim, proj_dim), generator=gen, dtype=torch.float32)
            defense_cfg.grad_jerk_proj = defense_cfg.grad_jerk_proj / (defense_cfg.grad_jerk_proj.norm(2, dim=0, keepdim=True) + 1e-12)
        
        proj = defense_cfg.grad_jerk_proj.to(device=per_sample_flat_grads.device, dtype=per_sample_flat_grads.dtype)
        embeds = per_sample_flat_grads @ proj
        aux_embeds_block = embeds.detach().cpu().numpy().astype(np.float32, copy=False)

    # flat_last_weights = ps_grads[last_w_name].flatten(start_dim=1)
    # last_biases = ps_grads[last_b_name]
    # last_layer_grads = torch.cat((flat_last_weights, last_biases), dim=1)
    
    # all_norms = torch.zeros_like(y, dtype=torch.float32)
    # for k in range(10):
    #     # center each class
    #     k_last_layer_grads = last_layer_grads[y == k]
    #     centered_k_last_layer_grads = k_last_layer_grads - k_last_layer_grads.mean(dim=0, keepdim=True)
    #     # take norm of each class
    #     centered_k_last_layer_norms = centered_k_last_layer_grads.norm(2, dim=1)
    #     all_norms[y == k] = centered_k_last_layer_norms

    # Apply gradient ascent if enabled
    if defense_apply_ascent and curr_gradient_ascent_indices.any():
        for name in ps_grads_clipped:
            ps_grads_clipped[name][curr_gradient_ascent_indices] *= -1


    with torch.no_grad():
        accum_grad_block = {name: grad.sum(dim=0) for name, grad in ps_grads_clipped.items()}

    return accum_grad_block, None, scores_block.cpu().numpy(), aux_embeds_block





def clip_and_accum_grads(model, X, y, optimizer, criterion, max_grad_norm,
                         block_size=1024, scores=None, device='cuda',
                         global_indices=None, aug_mult: int = 1, aug_fn=None,
                         world_size=1, rank=0, batch_size=None, drop_mask=None,
                         is_gradient_space_canary=False, global_idx_to_grad=None, canary_indices=None, defense_cfg: Optional[DefenseConfig] = None, defense_score_norm='linf', defense_score_fn='grad_norm', delta_theta=None, theta_t_minus_theta0=None, defense_apply_ascent=True):

    if scores is None:
        raise ValueError("scores array must be provided")
    
    if drop_mask is not None and len(drop_mask) != len(X):
        raise ValueError(f"drop_mask length ({len(drop_mask)}) must match X length ({len(X)})")
    
    batch_size_in = len(X)

    # Get indices of non-dropped samples
    active_indices = (torch.tensor(drop_mask, device=device) != 2)

    gradient_ascent_indices = torch.tensor(drop_mask, device=device)[active_indices] == 1

    # Filter out dropped samples
    X = X[active_indices]
    y = y[active_indices]
    global_indices = global_indices[active_indices]
    
    # Check if any canaries are in this batch and we should apply gradient space canary
    canary_mask = None
    if is_gradient_space_canary and canary_indices is not None:
        # Create a mask for which samples in the batch are canaries
        canary_indices_set = set(canary_indices.tolist() if hasattr(canary_indices, 'tolist') else canary_indices)
        canary_mask = torch.tensor([idx.item() in canary_indices_set for idx in global_indices], device=device)
    
    if len(X) == 0:
        return {}, scores
    
    # Process in blocks for memory efficiency
    accum_grad = None
    n_samples = len(X)
    
    for block_idx, i in enumerate(range(0, n_samples, block_size)):
        # Get current block
        idx_block = slice(i, min(i + block_size, n_samples))
        curr_X = X[idx_block]
        curr_y = y[idx_block]
        curr_global_indices = global_indices[idx_block]

        curr_gradient_ascent_indices = gradient_ascent_indices[idx_block]
        
        # Skip if no samples in this block
        if len(curr_X) == 0:
            continue
        
        # Check if this block contains any canaries and build mapping
        curr_canary_mask = canary_mask[idx_block] if canary_mask is not None else None
        block_contains_canary = curr_canary_mask is not None and curr_canary_mask.any()
        
        # Build mapping from local indices to crafted gradients for canaries in this block
        canary_gradient_map = None
        if block_contains_canary and global_idx_to_grad is not None:
            canary_gradient_map = {}
            # Only iterate over canaries in this block (where mask is True)
            canary_local_indices = torch.where(curr_canary_mask)[0]
            for local_idx in canary_local_indices:
                global_idx = curr_global_indices[local_idx].item()
                grad = global_idx_to_grad.get(global_idx)
                if grad is not None:
                    canary_gradient_map[int(local_idx.item())] = grad
        
        # Compute per-block gradients with clipping
        accum_grad_block, _, score_aux_block, dir_embeds_block = clip_and_accum_grads_block(
            model, curr_X, curr_y, optimizer, criterion, max_grad_norm,
            device=device, aug_mult=aug_mult, aug_fn=aug_fn,
            is_gradient_space_canary=block_contains_canary,
            crafted_gradient=canary_gradient_map if block_contains_canary else None,
            canary_local_idx=None,  # No longer used, replaced by canary_gradient_map
            curr_gradient_ascent_indices=curr_gradient_ascent_indices,
            defense_cfg=defense_cfg,
            defense_score_norm=defense_score_norm,
            defense_score_fn=defense_score_fn,
            delta_theta=delta_theta,
            theta_t_minus_theta0=theta_t_minus_theta0,
            defense_apply_ascent=defense_apply_ascent
        )

        # Accumulate gradients
        if accum_grad is None:
            accum_grad = accum_grad_block
        else:
            with torch.no_grad():
                for name in accum_grad:
                    accum_grad[name] += accum_grad_block[name]
        
        # Update scores for this block
        if defense_cfg is not None and defense_cfg.score_fn == 'grad_dir_change_rate':
            if dir_embeds_block is None:
                raise RuntimeError("dir_embeds_block must be returned from clip_and_accum_grads_block when score_fn='grad_dir_change_rate'")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_dirs = dir_embeds_block.astype(np.float32, copy=False)

            if defense_cfg.prev_grad_dir is None:
                defense_cfg.prev_grad_dir = np.full((len(scores), curr_dirs.shape[1]), np.nan, dtype=np.float32)

            if defense_cfg.prev_grad_dir.shape[1] != curr_dirs.shape[1]:
                raise ValueError(
                    f"defense_cfg.prev_grad_dir has shape {defense_cfg.prev_grad_dir.shape}, expected second dim == {curr_dirs.shape[1]}"
                )

            prev_dirs = defense_cfg.prev_grad_dir[curr_idx_np]
            valid = ~np.isnan(prev_dirs).any(axis=1)
            cos = np.sum(curr_dirs * prev_dirs, axis=1)
            # Clip cosine to [-1, 1] to handle numerical precision issues
            cos = np.clip(cos, -1.0, 1.0)

            score = np.zeros((curr_dirs.shape[0],), dtype=np.float32)
            score[valid] = (1.0 - cos[valid]).astype(np.float32, copy=False)
            scores[curr_idx_np] = score

            defense_cfg.prev_grad_dir[curr_idx_np] = curr_dirs

        elif defense_cfg is not None and defense_cfg.score_fn == 'gradient_scatter':
            if dir_embeds_block is None:
                raise RuntimeError("dir_embeds_block must be returned from clip_and_accum_grads_block when score_fn='gradient_scatter'")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_grads = dir_embeds_block.astype(np.float32, copy=False)
            k = int(defense_cfg.grad_scatter_k)
            if k <= 0:
                raise ValueError(f"grad_scatter_k must be > 0, got {k}")

            if defense_cfg.grad_scatter_hist is None or defense_cfg.grad_scatter_hist_pos is None:
                defense_cfg.grad_scatter_hist = np.full((len(scores), k, curr_grads.shape[1]), np.nan, dtype=np.float32)
                defense_cfg.grad_scatter_hist_pos = np.zeros((len(scores),), dtype=np.int32)

            if defense_cfg.grad_scatter_hist.shape[1] != k or defense_cfg.grad_scatter_hist.shape[2] != curr_grads.shape[1]:
                raise ValueError(
                    f"defense_cfg.grad_scatter_hist has shape {defense_cfg.grad_scatter_hist.shape}, expected (_, {k}, {curr_grads.shape[1]})"
                )

            scatter_scores = np.zeros((curr_grads.shape[0],), dtype=np.float32)
            for bi, gi in enumerate(curr_idx_np):
                pos = int(defense_cfg.grad_scatter_hist_pos[gi])
                defense_cfg.grad_scatter_hist[gi, pos, :] = curr_grads[bi]
                defense_cfg.grad_scatter_hist_pos[gi] = (pos + 1) % k

                recent = defense_cfg.grad_scatter_hist[gi]
                valid = ~np.isnan(recent).any(axis=1)
                if not np.any(valid):
                    scatter_scores[bi] = 0.0
                    continue

                recent_valid = recent[valid]
                centroid = recent_valid.mean(axis=0)
                diffs = recent_valid - centroid
                scatter_scores[bi] = np.mean(np.sum(diffs * diffs, axis=1), dtype=np.float32)

            scores[curr_idx_np] = scatter_scores

        elif defense_cfg is not None and defense_cfg.score_fn == 'grad_norm_percentile':
            if defense_cfg.grad_norm_hist is None or defense_cfg.grad_norm_hist_pos is None:
                raise ValueError("defense_cfg.grad_norm_hist and defense_cfg.grad_norm_hist_pos must be provided when score_fn='grad_norm_percentile'")

            k = int(defense_cfg.grad_norm_percentile_k)
            if k <= 0:
                raise ValueError(f"grad_norm_percentile_k must be > 0, got {k}")
            if defense_cfg.grad_norm_hist.shape[1] != k:
                raise ValueError(f"defense_cfg.grad_norm_hist has shape {defense_cfg.grad_norm_hist.shape}, expected second dim == {k}")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_norms = score_aux_block.astype(np.float32, copy=False)

            hist = defense_cfg.grad_norm_hist[curr_idx_np]
            valid = ~np.isnan(hist)
            counts = valid.sum(axis=1).astype(np.int32, copy=False)

            # Percentile rank in [0, 1]. If no history yet, define percentile as 0.
            leq = (hist <= curr_norms[:, None]) & valid
            pct = np.zeros_like(curr_norms, dtype=np.float32)
            nonzero = counts > 0
            pct[nonzero] = (leq[nonzero].sum(axis=1) / counts[nonzero]).astype(np.float32, copy=False)
            scores[curr_idx_np] = pct

            pos = defense_cfg.grad_norm_hist_pos[curr_idx_np].astype(np.int64, copy=False)
            defense_cfg.grad_norm_hist[curr_idx_np, pos] = curr_norms
            defense_cfg.grad_norm_hist_pos[curr_idx_np] = (pos + 1) % k
        elif defense_cfg is not None and defense_cfg.score_fn == 'grad_dir_volatility':
            if defense_cfg.grad_dir_hist is None or defense_cfg.grad_dir_hist_pos is None:
                raise ValueError("defense_cfg.grad_dir_hist and defense_cfg.grad_dir_hist_pos must be provided when score_fn='grad_dir_volatility'")
            if dir_embeds_block is None:
                raise RuntimeError("dir_embeds_block must be returned from clip_and_accum_grads_block when score_fn='grad_dir_volatility'")

            k = int(defense_cfg.grad_dir_volatility_k)
            if k <= 0:
                raise ValueError(f"grad_dir_volatility_k must be > 0, got {k}")
            if defense_cfg.grad_dir_hist.shape[1] != k:
                raise ValueError(f"defense_cfg.grad_dir_hist has shape {defense_cfg.grad_dir_hist.shape}, expected second dim == {k}")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_dirs = dir_embeds_block.astype(np.float32, copy=False)
            
            if defense_cfg.grad_dir_hist.shape[2] != curr_dirs.shape[1]:
                raise ValueError(
                    f"defense_cfg.grad_dir_hist has shape {defense_cfg.grad_dir_hist.shape}, expected third dim == {curr_dirs.shape[1]}"
                )

            hist = defense_cfg.grad_dir_hist[curr_idx_np]
            valid = ~np.isnan(hist[..., 0])
            counts = valid.sum(axis=1).astype(np.int32, copy=False)

            # cos_sim between current dir and each historical dir (dirs are already unit norm)
            dots = (hist * curr_dirs[:, None, :]).sum(axis=2)
            diffs = 1.0 - dots

            score = np.zeros((curr_dirs.shape[0],), dtype=np.float32)
            nonzero = counts > 0
            if np.any(nonzero):
                diffs_masked = np.where(valid, diffs, np.nan)
                score[nonzero] = np.nanmean(diffs_masked[nonzero], axis=1).astype(np.float32, copy=False)
            scores[curr_idx_np] = score

            pos = defense_cfg.grad_dir_hist_pos[curr_idx_np].astype(np.int64, copy=False)
            defense_cfg.grad_dir_hist[curr_idx_np, pos, :] = curr_dirs
            defense_cfg.grad_dir_hist_pos[curr_idx_np] = (pos + 1) % k
        elif defense_cfg is not None and defense_cfg.score_fn == 'grad_accel':
            if defense_cfg.grad_accel_hist is None or defense_cfg.grad_accel_hist_pos is None:
                raise ValueError("defense_cfg.grad_accel_hist and defense_cfg.grad_accel_hist_pos must be provided when score_fn='grad_accel'")
            if dir_embeds_block is None:
                raise RuntimeError("Projected gradients must be returned from clip_and_accum_grads_block when score_fn='grad_accel'")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_embeds = dir_embeds_block.astype(np.float32, copy=False)
            
            if defense_cfg.grad_accel_hist.shape[2] != curr_embeds.shape[1]:
                raise ValueError(
                    f"defense_cfg.grad_accel_hist has shape {defense_cfg.grad_accel_hist.shape}, expected third dim == {curr_embeds.shape[1]}"
                )

            hist = defense_cfg.grad_accel_hist[curr_idx_np]
            pos = defense_cfg.grad_accel_hist_pos[curr_idx_np].astype(np.int64, copy=False)

            prev1 = hist[np.arange(hist.shape[0]), (pos - 1) % 3, :]
            prev2 = hist[np.arange(hist.shape[0]), (pos - 2) % 3, :]
            valid = (~np.isnan(prev1).any(axis=1)) & (~np.isnan(prev2).any(axis=1))

            accel = np.zeros_like(curr_embeds, dtype=np.float32)
            accel[valid] = curr_embeds[valid] - 2.0 * prev1[valid] + prev2[valid]
            scores[curr_idx_np] = np.linalg.norm(accel, axis=1).astype(np.float32, copy=False)

            defense_cfg.grad_accel_hist[curr_idx_np, pos, :] = curr_embeds
            defense_cfg.grad_accel_hist_pos[curr_idx_np] = (pos + 1) % 3
        elif defense_cfg is not None and defense_cfg.score_fn == 'grad_jerk':
            if defense_cfg.grad_jerk_hist is None or defense_cfg.grad_jerk_hist_pos is None:
                raise ValueError("defense_cfg.grad_jerk_hist and defense_cfg.grad_jerk_hist_pos must be provided when score_fn='grad_jerk'")
            if dir_embeds_block is None:
                raise RuntimeError("Projected gradients must be returned from clip_and_accum_grads_block when score_fn='grad_jerk'")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_embeds = dir_embeds_block.astype(np.float32, copy=False)
            
            if defense_cfg.grad_jerk_hist.shape[2] != curr_embeds.shape[1]:
                raise ValueError(
                    f"defense_cfg.grad_jerk_hist has shape {defense_cfg.grad_jerk_hist.shape}, expected third dim == {curr_embeds.shape[1]}"
                )

            hist = defense_cfg.grad_jerk_hist[curr_idx_np]
            pos = defense_cfg.grad_jerk_hist_pos[curr_idx_np].astype(np.int64, copy=False)

            prev1 = hist[np.arange(hist.shape[0]), (pos - 1) % 4, :]
            prev2 = hist[np.arange(hist.shape[0]), (pos - 2) % 4, :]
            prev3 = hist[np.arange(hist.shape[0]), (pos - 3) % 4, :]
            valid = (~np.isnan(prev1).any(axis=1)) & (~np.isnan(prev2).any(axis=1)) & (~np.isnan(prev3).any(axis=1))

            jerk = np.zeros_like(curr_embeds, dtype=np.float32)
            jerk[valid] = curr_embeds[valid] - 3.0 * prev1[valid] + 3.0 * prev2[valid] - prev3[valid]
            scores[curr_idx_np] = np.linalg.norm(jerk, axis=1).astype(np.float32, copy=False)

            defense_cfg.grad_jerk_hist[curr_idx_np, pos, :] = curr_embeds
            defense_cfg.grad_jerk_hist_pos[curr_idx_np] = (pos + 1) % 4
        elif defense_cfg is not None and defense_cfg.score_fn == 'norm_x_dir_uniqueness':
            if defense_cfg.dir_unique_hist is None or defense_cfg.dir_unique_hist_pos is None:
                raise ValueError("defense_cfg.dir_unique_hist and defense_cfg.dir_unique_hist_pos must be provided when score_fn='norm_x_dir_uniqueness'")
            if dir_embeds_block is None:
                raise RuntimeError("Projected directions must be returned from clip_and_accum_grads_block when score_fn='norm_x_dir_uniqueness'")

            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_dirs = dir_embeds_block.astype(np.float32, copy=False)

            # Magnitude of clipped per-sample gradients (computed in clip_and_accum_grads_block)
            magnitude = score_aux_block.astype(np.float32, copy=False)

            k = int(defense_cfg.dir_unique_k)
            if defense_cfg.dir_unique_hist.shape[2] != curr_dirs.shape[1]:
                raise ValueError(
                    f"defense_cfg.dir_unique_hist has shape {defense_cfg.dir_unique_hist.shape}, expected third dim == {curr_dirs.shape[1]}"
                )
            hist = defense_cfg.dir_unique_hist[curr_idx_np]
            pos = defense_cfg.dir_unique_hist_pos[curr_idx_np].astype(np.int64, copy=False)

            # Gather previous K directions in recency order (t-1..t-K)
            past_dirs = []
            for j in range(1, k + 1):
                past_dirs.append(hist[np.arange(hist.shape[0]), (pos - j) % k, :])
            past_dirs = np.stack(past_dirs, axis=1)  # (B, K, D)

            valid = ~np.isnan(past_dirs).any(axis=2)  # (B, K)
            # Since directions are normalized, cos_sim is dot product
            cos = np.einsum('bd,bkd->bk', curr_dirs, past_dirs)
            cos = np.where(valid, cos, np.nan)

            # std over available past cos sims (need at least 2 to have nonzero std)
            counts = np.sum(~np.isnan(cos), axis=1)
            dir_vol = np.zeros((curr_dirs.shape[0],), dtype=np.float32)
            enough = counts >= 2
            if np.any(enough):
                dir_vol[enough] = np.nanstd(cos[enough], axis=1).astype(np.float32, copy=False)

            scores[curr_idx_np] = magnitude * dir_vol

            defense_cfg.dir_unique_hist[curr_idx_np, pos, :] = curr_dirs
            defense_cfg.dir_unique_hist_pos[curr_idx_np] = (pos + 1) % k
        elif defense_cfg is not None and defense_cfg.score_fn == 'loss_momentum':
            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_losses = score_aux_block.astype(np.float32, copy=False)
            if defense_cfg.prev_losses is None:
                raise ValueError("defense_cfg.prev_losses must be provided for score_fn='loss_momentum'")
            prev = defense_cfg.prev_losses[curr_idx_np]
            valid = ~np.isnan(prev)
            score = np.zeros_like(curr_losses)
            score[valid] = curr_losses[valid] - prev[valid]
            scores[curr_idx_np] = score
            defense_cfg.prev_losses[curr_idx_np] = curr_losses

        elif defense_cfg is not None and defense_cfg.score_fn == 'loss_volatility':
            curr_idx_np = curr_global_indices.detach().cpu().numpy()
            curr_losses = score_aux_block.astype(np.float32, copy=False)
            if defense_cfg.loss_hist is None or defense_cfg.loss_hist_pos is None:
                raise ValueError("defense_cfg.loss_hist and defense_cfg.loss_hist_pos must be provided for score_fn='loss_volatility'")
            k = defense_cfg.loss_volatility_k
            pos = defense_cfg.loss_hist_pos[curr_idx_np].astype(np.int64, copy=False)
            defense_cfg.loss_hist[curr_idx_np, pos] = curr_losses
            defense_cfg.loss_hist_pos[curr_idx_np] = (pos + 1) % k
            hist = defense_cfg.loss_hist[curr_idx_np]
            valid_counts = (~np.isnan(hist)).sum(axis=1)
            vol = np.zeros_like(curr_losses)
            enough = valid_counts >= 2
            if np.any(enough):
                vol[enough] = np.nanstd(hist[enough], axis=1).astype(np.float32, copy=False)
            scores[curr_idx_np] = vol

        else:
            scores[curr_global_indices.cpu().numpy()] = score_aux_block

    # Note: Division by batch_size is now done in the calling code after noise addition
    # This ensures standard DP-SGD: noise is added to the sum, then divided
    return accum_grad, scores


