# """PyTorch implementation of Wide Residual Network for CIFAR."""

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from dataclasses import dataclass
# from typing import Optional


# class WideResNet(nn.Module):
#     """Wide Residual Network implementation in PyTorch."""
    
#     def __init__(
#         self,
#         input_shape,
#         out_dim: int = 10,
#         depth: int = 16,
#         width: int = 4,
#         dropout_rate: float = 0.0
#     ):
#         super().__init__()
        
#         self.num_classes = out_dim
#         self.width = width
#         self.dropout_rate = dropout_rate
        
#         # Calculate number of blocks per residual group
#         # Formula: (depth - 4) // 6 accounts for initial conv + 3 groups + final layers
#         self.resnet_blocks = (depth - 4) // 6
        
#         if (depth - 4) % 6 != 0:
#             raise ValueError(f"Depth {depth} is not compatible with WideResNet. "
#                            f"Should be 6n+4 for some integer n.")
        
#         # Initial convolution
#         self.first_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        
#         # Three residual block groups
#         self.block1 = self._make_residual_group(16, 16 * width, stride=1)
#         self.block2 = self._make_residual_group(16 * width, 32 * width, stride=2)
#         self.block3 = self._make_residual_group(32 * width, 64 * width, stride=2)
        
#         # Final layers
#         self.final_norm = nn.GroupNorm(16, 64 * width)
#         self.classifier = nn.Linear(64 * width, out_dim)
        
    
#     def _make_residual_group(self, in_channels: int, out_channels: int, stride: int) -> nn.ModuleList:
#         """Create a group of residual blocks."""
#         blocks = nn.ModuleList()
        
#         for i in range(self.resnet_blocks):
#             # Only apply stride on the first block of the group
#             block_stride = stride if i == 0 else 1
#             blocks.append(
#                 ResidualBlock(
#                     in_channels if i == 0 else out_channels,
#                     out_channels,
#                     stride=block_stride,
#                     dropout_rate=self.dropout_rate
#                 )
#             )
        
#         return blocks
    
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass through the network."""
#         # Initial convolution
#         x = self.first_conv(x)
        
#         # Residual block groups
#         for block in self.block1:
#             x = block(x)
        
#         for block in self.block2:
#             x = block(x)
        
#         for block in self.block3:
#             x = block(x)
        
#         # Final processing
#         x = F.relu(x)
#         x = self.final_norm(x)
        
#         # Global average pooling
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = x.view(x.size(0), -1)  # Flatten
        
#         # Apply dropout if specified
#         if self.dropout_rate > 0.0:
#             x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
#         # Classification layer
#         x = self.classifier(x)
        
#         return x


# class ResidualBlock(nn.Module):
#     """A single residual block for Wide ResNet."""
    
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         stride: int = 1,
#         dropout_rate: float = 0.0
#     ):
#         super().__init__()
        
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.stride = stride
#         self.dropout_rate = dropout_rate
        
#         # Skip connection projection (if needed)
#         self.skip_projection = None
#         if stride != 1 or in_channels != out_channels:
#             self.skip_projection = nn.Sequential(
#                 nn.ReLU(inplace=True),
#                 nn.GroupNorm(16, in_channels),
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
#             )
        
#         # Main residual path - first conv
#         self.norm1 = nn.GroupNorm(16, in_channels)
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
#                               stride=stride, padding=1, bias=False)
        
#         # Main residual path - second conv  
#         self.norm2 = nn.GroupNorm(16, out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
#                               stride=1, padding=1, bias=False)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass through the residual block."""
#         # Skip connection
#         if self.skip_projection is not None:
#             # Apply skip projection on the first block of each group
#             skip = self.skip_projection(x)
#         else:
#             skip = x
        
#         # Main residual path - first convolution
#         out = F.relu(x)
#         out = self.norm1(out)
#         out = self.conv1(out)
        
#         # # Apply dropout if specified
#         # if self.dropout_rate > 0.0:
#         #     out = F.dropout(out, p=self.dropout_rate, training=self.training)
        
#         # Main residual path - second convolution
#         out = F.relu(out)
#         out = self.norm2(out)
#         out = self.conv2(out)
        
#         # # Apply dropout if specified
#         # if self.dropout_rate > 0.0:
#         #     out = F.dropout(out, p=self.dropout_rate, training=self.training)
        
#         # Add skip connection
#         out = out + skip
        
#         return out




"""PyTorch implementation of Wide Residual Network for CIFAR with Weight Standardization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math


class WSConv2d(nn.Module):
    """2D Convolution with Weight Standardization and affine gain+bias."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        eps: float = 1e-4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.eps = eps
        
        # Weight parameter
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size
        ))
        
        # Bias parameter (always included in WSConv2d)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Gain parameter for weight standardization
        self.gain = nn.Parameter(torch.ones(out_channels))
    
    def _initialize_weights(self):
        """Initialize weights using fan-in variance scaling."""
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(self.weight, mean=0.0, std=std)
    
    def standardize_weight(self):
        """Apply weight standardization with affine gain."""
        # Compute mean and variance over spatial and input channel dimensions
        # weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        mean = self.weight.mean(dim=(1, 2, 3), keepdim=True)  # Shape: (out_channels, 1, 1, 1)
        var = self.weight.var(dim=(1, 2, 3), keepdim=True, unbiased=False)  # Shape: (out_channels, 1, 1, 1)
        
        # Fan-in calculation
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        
        # Compute scale factor: gain / sqrt(fan_in * var + eps)
        scale = self.gain.view(-1, 1, 1, 1) * torch.rsqrt(var * fan_in + self.eps)
        
        # Compute shift: mean * scale
        shift = mean * scale
        
        # Return standardized weight: (w * scale - shift)
        return self.weight * scale - shift
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight standardization."""
        # Get standardized weights
        weight = self.standardize_weight()
        
        # Apply convolution
        out = F.conv2d(
            x, weight, bias=self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
        
        return out



class WideResNet(nn.Module):
    """Wide Residual Network implementation in PyTorch."""
    
    def __init__(
        self,
        input_shape,
        out_dim: int = 10,
        depth: int = 16,
        width: int = 4,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        self.num_classes = out_dim
        self.width = width
        self.dropout_rate = dropout_rate
        
        # Calculate number of blocks per residual group
        # Formula: (depth - 4) // 6 accounts for initial conv + 3 groups + final layers
        self.resnet_blocks = (depth - 4) // 6
        
        if (depth - 4) % 6 != 0:
            raise ValueError(f"Depth {depth} is not compatible with WideResNet. "
                           f"Should be 6n+4 for some integer n.")
        
        # Initial convolution (with bias since we're using WSConv2d)
        self.first_conv = WSConv2d(3, 16, kernel_size=3, stride=1, padding=1)
        
        # Three residual block groups
        self.block1 = self._make_residual_group(16, 16 * width, stride=1)
        self.block2 = self._make_residual_group(16 * width, 32 * width, stride=2)
        self.block3 = self._make_residual_group(32 * width, 64 * width, stride=2)
        
        # Final layers
        self.final_norm = nn.GroupNorm(16, 64 * width)
        self.classifier = nn.Linear(64 * width, out_dim)
    
    
    def _make_residual_group(self, in_channels: int, out_channels: int, stride: int) -> nn.ModuleList:
        """Create a group of residual blocks."""
        blocks = nn.ModuleList()
        
        for i in range(self.resnet_blocks):
            # Only apply stride on the first block of the group
            block_stride = stride if i == 0 else 1
            blocks.append(
                ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride=block_stride,
                    dropout_rate=self.dropout_rate
                )
            )
        
        return blocks
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Initial convolution
        x = self.first_conv(x)
        
        # Residual block groups
        for block in self.block1:
            x = block(x)
        
        for block in self.block2:
            x = block(x)
        
        for block in self.block3:
            x = block(x)
        
        # Final processing
        x = self.final_norm(x)
        x = F.relu(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Apply dropout if specified
        if self.dropout_rate > 0.0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Classification layer
        x = self.classifier(x)
        
        return x


class ResidualBlock(nn.Module):
    """A single residual block for Wide ResNet."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dropout_rate = dropout_rate
        
        # Skip connection projection (if needed)
        self.skip_projection = None
        if stride != 1 or in_channels != out_channels:
            self.skip_projection = WSConv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        
        # Main residual path - first conv
        self.norm1 = nn.GroupNorm(16, in_channels)
        self.conv1 = WSConv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1)
        
        # Main residual path - second conv  
        self.norm2 = nn.GroupNorm(16, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block."""
        # Skip connection
        if self.skip_projection is not None:
            # Apply skip projection on the first block of each group
            skip = self.skip_projection(x)
        else:
            skip = x
        
        # Main residual path - first convolution
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)

        # Apply dropout if specified
        if self.dropout_rate > 0.0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        # Main residual path - second convolution
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        
        # Apply dropout if specified
        if self.dropout_rate > 0.0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        
        # Add skip connection
        out = out + skip
        
        return out
