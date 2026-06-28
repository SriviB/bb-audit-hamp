"""
Wide Residual Network (WRN-16-4) for non-private CIFAR training.

Matches the implementation in misleading-privacy-evals/src/models.py:
  - Standard torch.nn.Conv2d (no Weight Standardization)
  - BatchNorm2d (not GroupNorm)
  - Pre-activation ResNet blocks: BN → ReLU → Conv
  - No dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBlock(nn.Module):
    """Pre-activation residual block: BN → ReLU → Conv → BN → ReLU → Conv."""

    def __init__(
        self,
        features_in: int,
        features: int,
        stride: int,
        expand_features: bool,
    ):
        super().__init__()

        self.norm_in = nn.BatchNorm2d(features_in)
        self.conv_in = nn.Conv2d(
            features_in, features,
            kernel_size=3, stride=1, padding=1,
        )
        self.norm_out = nn.BatchNorm2d(features)
        self.conv_out = nn.Conv2d(
            features, features,
            kernel_size=3, stride=stride, padding=1,
        )

        if stride != 1 or expand_features:
            self.identity = nn.Conv2d(
                features_in, features,
                kernel_size=1, stride=stride,
            )
        else:
            self.identity = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(torch.relu(self.norm_in(inputs)))
        x = self.conv_out(torch.relu(self.norm_out(x)))
        x += self.identity(inputs)
        return x


class WideResNetNonPrivate(nn.Module):
    """
    Wide Residual Network matching the misleading-privacy-evals implementation.

    Uses BatchNorm2d + standard Conv2d (no Weight Standardization or GroupNorm).
    Default configuration: depth=16, width=4 → WRN-16-4.

    Constructor signature is compatible with the bb-audit-hamp Models registry:
        model = WideResNetNonPrivate(X.shape, out_dim=10)
    """

    def __init__(
        self,
        input_shape,
        out_dim: int = 10,
        depth: int = 16,
        width: int = 4,
    ):
        super().__init__()

        assert (depth - 4) % 6 == 0 and depth > 4
        assert width > 0
        num_blocks = (depth - 4) // 6
        features_per_block = (
            16,
            16 * width,
            32 * width,
            64 * width,
        )

        # Determine input channels from input_shape
        if isinstance(input_shape, (tuple, list, torch.Size)):
            # input_shape could be (N, C, H, W) or (C, H, W)
            in_channels = input_shape[1] if len(input_shape) == 4 else input_shape[0]
        else:
            in_channels = 3

        features_in = features_per_block[0]
        self.conv_in = nn.Conv2d(
            in_channels, features_in,
            kernel_size=3, stride=1, padding=1,
        )

        self.group_1, features_in = self._wide_layer(
            features_in, features_per_block[1],
            stride=1, num_blocks=num_blocks,
        )
        self.group_2, features_in = self._wide_layer(
            features_in, features_per_block[2],
            stride=2, num_blocks=num_blocks,
        )
        self.group_3, features_in = self._wide_layer(
            features_in, features_per_block[3],
            stride=2, num_blocks=num_blocks,
        )

        self.norm_out = nn.BatchNorm2d(features_in)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(features_in, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.group_1(x)
        x = self.group_2(x)
        x = self.group_3(x)
        x = torch.relu(self.norm_out(x))
        x = self.flatten(self.pool(x))
        x = self.dense(x)
        return x

    @staticmethod
    def _wide_layer(
        features_in: int,
        features: int,
        num_blocks: int,
        stride: int,
    ):
        blocks = []
        for block_idx in range(num_blocks):
            block = WideBlock(
                features_in=features_in,
                features=features,
                stride=stride if block_idx == 0 else 1,
                expand_features=(features_in != features),
            )
            blocks.append(block)
            features_in = features
        return nn.Sequential(*blocks), features_in
