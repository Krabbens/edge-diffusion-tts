"""
Neural network layers for Edge Diffusion TTS.

Optimized components for edge device inference:
- Depthwise separable convolutions
- Memory-efficient attention
- Efficient transformer blocks
"""

from .conv import DepthwiseSeparableConv
from .attention import EfficientAttention
from .transformer import DiffusionTransformerBlock, AdaLayerNorm, SwiGLU
from .embeddings import SinusoidalTimeEmb, SinusoidalPositionalEmb, RotaryEmbedding

__all__ = [
    "DepthwiseSeparableConv",
    "EfficientAttention",
    "DiffusionTransformerBlock",
    "AdaLayerNorm",
    "SwiGLU",
    "SinusoidalTimeEmb",
    "SinusoidalPositionalEmb",
    "RotaryEmbedding",
]
