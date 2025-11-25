"""
Multi-Head Latent Attention (MLA) with Local Attention.

DeepSeek-V2/V3 style attention with Low-Rank Key-Value Compression.
Supports sliding window local attention for memory efficiency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


from .embeddings import RotaryEmbedding


def create_local_attention_mask(
    seq_len: int, 
    window_size: int, 
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create a sliding window attention mask.
    
    Args:
        seq_len: Sequence length
        window_size: Size of attention window (each side)
        device: Target device
        dtype: Target dtype
        
    Returns:
        Attention mask [1, 1, seq_len, seq_len] where True = attend, False = mask
    """
    # Create position indices
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Local window: |i - j| <= window_size
    mask = (cols - rows).abs() <= window_size
    
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) with optional Local Attention.
    
    Compresses KV into a lower-dimensional latent vector to reduce memory usage.
    Optional sliding window attention reduces memory from O(n²) to O(n*w).
    
    Args:
        dim: Model dimension
        heads: Number of attention heads
        kv_lora_rank: Rank for KV compression (default: dim // 2)
        dropout: Dropout probability
        use_flash: Use Flash Attention when available
        window_size: Local attention window size (None = full attention)
                     Each position attends to window_size tokens on each side
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        kv_lora_rank: int = None,
        dropout: float = 0.1,
        use_flash: bool = True,
        window_size: int = None  # None = full attention, e.g. 256 for local
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size  # Local attention window
        
        # KV Compression Rank (DeepSeek style: typically much smaller than dim)
        self.kv_lora_rank = kv_lora_rank or (dim // 2)
        
        # Query Projection
        self.q_proj = nn.Linear(dim, dim, bias=False)
        
        # KV Compression (Down-projection)
        self.kv_down_proj = nn.Linear(dim, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        
        # KV Up-projection (generate Keys and Values from latent)
        self.kv_up_proj = nn.Linear(self.kv_lora_rank, dim * 2, bias=False)
        
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        
        # RoPE (for queries and keys)
        self.rope = RotaryEmbedding(self.head_dim)
        
        # Cache for attention mask (avoid recomputing)
        self._cached_mask = None
        self._cached_mask_seq_len = 0
        
    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Query input [B, T, D]
            context: Key/Value input [B, S, D] (if cross-attention)
            cond: Conditioning vector [B, D] (added to query)
        """
        B, T, C = x.shape
        kv_input = context if context is not None else x
        S = kv_input.shape[1]
        
        # 1. Query Processing
        q = x
        if cond is not None:
            q = q + cond.unsqueeze(1)
        
        q = self.q_proj(q).reshape(B, T, self.heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, h, T, d]
        
        # 2. Key-Value Compression (MLA)
        # Project down to latent space
        c_kv = self.kv_down_proj(kv_input)  # [B, S, rank]
        c_kv = self.kv_norm(c_kv)
        
        # Project up to generate Heads
        # In optimized inference, we would cache c_kv. 
        # Here we re-compute for training simplicity.
        kv = self.kv_up_proj(c_kv)  # [B, S, 2*D]
        kv = kv.reshape(B, S, 2, self.heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, h, S, d]
        k, v = kv[0], kv[1]
        
        # Apply RoPE (Rotary Positional Embeddings)
        # DeepSeek V2/V3 applies RoPE to Q and K (decoupled strategy usually, but here standard RoPE)
        # Only apply RoPE if sequence lengths match or if causal (Self-Attention)
        if context is None:
            # Self-Attention: Apply RoPE to both Q and K
            q, k = self.rope(q, k)
        
        # 3. Prepare Local Attention Mask (if needed)
        attn_mask = None
        if self.window_size is not None and context is None:  # Only for self-attention
            # Cache mask if sequence length unchanged
            if T != self._cached_mask_seq_len or self._cached_mask is None:
                self._cached_mask = create_local_attention_mask(
                    T, self.window_size, x.device, x.dtype
                )
                self._cached_mask_seq_len = T
            attn_mask = self._cached_mask
        
        # 4. Attention
        if self.use_flash:
            # Flash Attention 2.0 with optional local mask
            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            # Apply local attention mask
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask, float('-inf'))
            
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = attn @ v
            
        # 5. Output Projection
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)
