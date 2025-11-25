"""
Embedding layers for Edge Diffusion TTS.
"""

import math

import torch
import torch.nn as nn
from typing import Tuple


class SinusoidalTimeEmb(nn.Module):
    """
    Sinusoidal timestep embedding for diffusion models.
    
    Creates fixed sinusoidal embeddings similar to positional encodings
    in transformers, but for diffusion timesteps.
    
    Args:
        dim: Embedding dimension
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal embeddings for timesteps.
        
        Args:
            t: Timesteps [B]
        
        Returns:
            Embeddings [B, dim]
        """
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class LearnedTimeEmb(nn.Module):
    """
    Learned timestep embedding with sinusoidal initialization.
    
    Combines sinusoidal embeddings with learnable MLP projection.
    
    Args:
        dim: Output embedding dimension
        hidden_dim: Hidden layer dimension (default: 4 * dim)
    """
    
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.sinusoidal = SinusoidalTimeEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            t: Timesteps [B]
        
        Returns:
            Embeddings [B, dim]
        """
        emb = self.sinusoidal(t)
        return self.mlp(emb)


class LearnedPositionalEmb(nn.Module):
    """
    Learned positional embeddings.
    
    Args:
        max_len: Maximum sequence length
        dim: Embedding dimension
    """
    
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(max_len, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input.
        
        Args:
            x: Input tensor [B, T, D]
        
        Returns:
            x + positional embeddings
        """
        T = x.shape[1]
        positions = torch.arange(T, device=x.device)
        return x + self.emb(positions)


class SinusoidalPositionalEmb(nn.Module):
    """
    Fixed sinusoidal positional embeddings.
    
    Args:
        dim: Embedding dimension
        max_len: Maximum sequence length for precomputation
    """
    
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input.
        
        Args:
            x: Input tensor [B, T, D]
        
        Returns:
            x + positional embeddings
        """
        return x + self.pe[:x.shape[1]]


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    
    Encodes relative positions by rotating the query and key vectors.
    Crucial for long-context performance and generalization.
    
    Args:
        dim: Embedding dimension (head_dim)
        max_len: Maximum sequence length
    """
    
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to queries and keys.
        
        Args:
            q: Queries [B, H, T, D]
            k: Keys [B, H, T, D]
        """
        # Truncate to sequence length
        T = q.shape[2]
        cos = self.cos_cached[:, :, :T, :]
        sin = self.sin_cached[:, :, :T, :]
        
        return self.apply_rotary_pos_emb(q, cos, sin), self.apply_rotary_pos_emb(k, cos, sin)
    
    def apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate vector x."""
        return (x * cos) + (self.rotate_half(x) * sin)
        
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
