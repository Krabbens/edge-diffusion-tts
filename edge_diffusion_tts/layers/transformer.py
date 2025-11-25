"""
Transformer layers for Edge Diffusion TTS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import EfficientAttention
from .mla import MultiHeadLatentAttention, RMSNorm


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    
    Sigmoid-Weighted Linear Unit with Gating.
    Standard in LLaMA / DeepSeek.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class FeedForward(nn.Module):
    """
    Feed-forward network with GEGLU activation.
    
    Args:
        dim: Input/output dimension
        mult: Hidden dimension multiplier
        dropout: Dropout probability
    """
    
    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden_dim = dim * mult
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),  # 2x for SwiGLU split
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class AdaLayerNorm(nn.Module):
    """
    Adaptive RMS Normalization (AdaRMSNorm).
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: [B, C]
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        x = self.norm(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusionTransformerBlock(nn.Module):
    """
    Transformer block optimized for diffusion with AdaLN and Cross-Attention.
    
    Features:
    - AdaLN for robust timestep conditioning
    - Cross-Attention for semantic conditioning
    - Efficient Self-Attention (Flash) with optional local attention
    - GEGLU FFN
    """
    
    def __init__(
        self,
        dim: int,
        context_dim: int = None,
        cond_dim: int = None,
        heads: int = 4,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        use_adaln: bool = True,
        window_size: int = None  # Local attention window (None = full attention)
    ):
        super().__init__()
        from .attention import CrossAttention
        
        self.use_adaln = use_adaln
        context_dim = context_dim or dim
        cond_dim = cond_dim or dim
        
        # 1. Self-Attention with optional local attention window
        if use_adaln:
            self.norm1 = AdaLayerNorm(dim, cond_dim)
        else:
            self.norm1 = RMSNorm(dim)
            
        self.attn = EfficientAttention(dim, heads, dropout, window_size=window_size)
        
        # 2. Cross-Attention (MLA) - full attention for conditioning
        self.norm2 = RMSNorm(dim)
        self.cross_attn = MultiHeadLatentAttention(
            dim=dim, 
            heads=heads, 
            kv_lora_rank=dim // 2,
            dropout=dropout,
            window_size=None  # Full attention for cross-attn (context is usually short)
        )
        
        # 3. FFN
        if use_adaln:
            self.norm3 = AdaLayerNorm(dim, cond_dim)
        else:
            self.norm3 = RMSNorm(dim)
            
        self.ffn = FeedForward(dim, ffn_mult, dropout)
        
        # Zero-init output of attention/ffn for stability
        # (Optional, but often helps DiT)
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: torch.Tensor, 
        cond: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input [B, T, D]
            context: Cross-attn context [B, S, D_ctx] (Semantic tokens)
            cond: AdaLN conditioning [B, D_cond] (Timestep emb)
        """
        # Self-Attention
        if self.use_adaln:
            h = self.norm1(x, cond)
        else:
            h = self.norm1(x)
        x = x + self.attn(h)
        
        # Cross-Attention
        # Cross-Attention (MLA)
        # MLA takes x (query) and context (kv)
        x = x + self.cross_attn(self.norm2(x), context=context)
        
        # FFN
        if self.use_adaln:
            h = self.norm3(x, cond)
        else:
            h = self.norm3(x)
        x = x + self.ffn(h)
        
        return x
