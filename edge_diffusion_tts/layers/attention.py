"""
Attention layers for Edge Diffusion TTS.
Supports sliding window local attention for memory efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_local_attention_mask(
    seq_len: int, 
    window_size: int, 
    device: torch.device,
) -> torch.Tensor:
    """
    Create a sliding window attention mask.
    
    Args:
        seq_len: Sequence length
        window_size: Size of attention window (each side)
        device: Target device
        
    Returns:
        Attention mask [1, 1, seq_len, seq_len] where True = attend, False = mask
    """
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = (cols - rows).abs() <= window_size
    return mask.unsqueeze(0).unsqueeze(0)


class EfficientAttention(nn.Module):
    """
    Memory-efficient attention for edge devices.
    
    Features:
    - Uses PyTorch 2.0 Flash Attention when available
    - Falls back to standard attention otherwise
    - Optimized for inference speed
    - Optional sliding window local attention
    
    Args:
        dim: Model dimension
        heads: Number of attention heads
        dropout: Dropout probability
        window_size: Local attention window size (None = full attention)
    """
    
    def __init__(
        self, 
        dim: int, 
        heads: int = 4, 
        dropout: float = 0.1,
        window_size: int = None
    ):
        super().__init__()
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Check for Flash Attention (PyTorch 2.0+)
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
        
        # Cache for attention mask
        self._cached_mask = None
        self._cached_mask_seq_len = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, D]
        
        Returns:
            Output tensor [B, T, D]
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Prepare local attention mask if needed
        attn_mask = None
        if self.window_size is not None:
            if T != self._cached_mask_seq_len or self._cached_mask is None:
                self._cached_mask = create_local_attention_mask(
                    T, self.window_size, x.device
                )
                self._cached_mask_seq_len = T
            attn_mask = self._cached_mask
        
        if self.use_flash:
            # Use PyTorch 2.0 Flash Attention (memory-efficient)
            dropout_p = self.dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask,
                dropout_p=dropout_p
            )
        else:
            # Manual attention (fallback)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = attn @ v
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class CrossAttention(nn.Module):
    """
    Cross-attention layer for conditioning.
    
    Args:
        dim: Model dimension
        context_dim: Context dimension (if different from dim)
        heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        dim: int,
        context_dim: int = None,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        context_dim = context_dim or dim
        assert dim % heads == 0
        
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(context_dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Query tensor [B, T, D]
            context: Key/Value tensor [B, S, D_ctx]
        
        Returns:
            Output tensor [B, T, D]
        """
        B, T, C = x.shape
        S = context.shape[1]
        
        q = self.q(x).reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).reshape(B, S, 2, self.heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        if self.use_flash:
            dropout_p = self.dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = attn @ v
        
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)
