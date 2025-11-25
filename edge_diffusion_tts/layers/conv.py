"""
Convolutional layers for Edge Diffusion TTS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution - significantly fewer parameters.
    
    Standard Conv: C_in × C_out × K parameters
    Depthwise Sep: C_in × K + C_in × C_out parameters
    For K=3: ~3x fewer parameters
    
    Args:
        in_ch: Input channels
        out_ch: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
    """
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1
    ):
        super().__init__()
        padding = kernel_size // 2
        
        # Depthwise convolution (separate filter per channel)
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=False
        )
        
        # Pointwise convolution (1x1 conv for channel mixing)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=True)
        
        # Group normalization (more stable than BatchNorm for small batches)
        num_groups = min(8, out_ch)
        self.norm = nn.GroupNorm(num_groups, out_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C_in, T]
        
        Returns:
            Output tensor [B, C_out, T']
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return F.gelu(x)


class ConvBlock(nn.Module):
    """
    Standard convolution block with normalization and activation.
    
    Args:
        in_ch: Input channels
        out_ch: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
    """
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1
    ):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        num_groups = min(8, out_ch)
        self.norm = nn.GroupNorm(num_groups, out_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return F.gelu(x)
