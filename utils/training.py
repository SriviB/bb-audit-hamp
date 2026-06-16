"""
Shared training helpers used across audit scripts.

Centralises classes and functions that were previously copy-pasted into every
audit entry-point (AugmentationFunction, IndexedTensorDataset, model initialisation,
accuracy evaluation, and per-sample loss computation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from torch.utils.data import TensorDataset, DataLoader, Dataset

from models.wideresnet import WSConv2d


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

class IndexedTensorDataset(Dataset):
    """TensorDataset that also returns the sample's global index."""
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == t.size(0) for t in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors) + (index,)

    def __len__(self):
        return self.tensors[0].size(0)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class AugmentationFunction:
    """Random crop + horizontal flip for image datasets."""
    def __init__(self, image_size=32, channels=3):
        self.base_transforms = v2.Compose([
            v2.RandomCrop(image_size, padding=4),
            v2.RandomHorizontalFlip(p=0.5),
        ])

    def __call__(self, x):
        return self.base_transforms(x)


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def xavier_init_model(model: nn.Module) -> None:
    """Xavier initialisation for Linear and Conv2d layers."""
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    model.apply(_init)


def init_wideresnet(model: nn.Module) -> None:
    """Kaiming (He) initialisation for WideResNet."""
    for m in model.modules():
        if isinstance(m, WSConv2d):
            m._initialize_weights()
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 128) -> float:
    """Return classification accuracy on (X, y)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for curr_X, curr_y in loader:
            curr_X, curr_y = curr_X.to(device), curr_y.to(device)
            correct += (model(curr_X).argmax(dim=1) == curr_y).sum().item()
            total += len(curr_y)
    model.train()
    return correct / total if total > 0 else 0.0


def compute_per_sample_losses(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> "np.ndarray":
    """Return per-sample cross-entropy losses as a numpy array."""
    import numpy as np
    model = model.to(device)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    losses = []

    model.eval()
    with torch.no_grad():
        for curr_X, curr_y in loader:
            curr_X, curr_y = curr_X.to(device), curr_y.to(device)
            logits = model(curr_X)

            # Sequence models: (B, T, C) logits, (B, T) targets
            if curr_y.ndim == 2 and logits.ndim == 3:
                b, t, c = logits.shape
                batch_losses = F.cross_entropy(
                    logits.reshape(b * t, c),
                    curr_y.reshape(b * t),
                    reduction='none',
                ).reshape(b, t).mean(dim=1)
            else:
                batch_losses = F.cross_entropy(logits, curr_y, reduction='none')

            losses.append(batch_losses.detach().cpu())

    model.train()
    return torch.cat(losses, dim=0).numpy()
