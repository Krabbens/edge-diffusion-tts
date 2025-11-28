"""
Finite Scalar Quantization (FSQ).

From "Finite Scalar Quantization: VQ-VAE Made Simple" (DeepMind, 2023)

Key advantages over VQ:
- No codebook collapse
- No EMA needed
- No commitment loss needed
- Simpler implementation
- Better codebook utilization
"""

import torch
import torch.nn as nn
from typing import List


class FSQ(nn.Module):
    """
    Finite Scalar Quantization.
    
    Instead of learning a codebook, we quantize each dimension 
    to a fixed number of levels.
    
    Example: levels=[8, 5, 5, 5] gives 8*5*5*5 = 1000 possible codes.
    
    Args:
        levels: Number of quantization levels per dimension
                e.g., [8, 6, 5] gives 8*6*5 = 240 codes
    """
    
    def __init__(self, levels: List[int]):
        super().__init__()
        self.levels = levels
        self.dim = len(levels)
        
        # Precompute for indexing
        self.register_buffer(
            '_levels', 
            torch.tensor(levels, dtype=torch.int32)
        )
        self.register_buffer(
            '_basis',
            torch.cumprod(
                torch.tensor([1] + levels[:-1], dtype=torch.int64), 
                dim=0
            )
        )
        
        self.codebook_size = 1
        for l in levels:
            self.codebook_size *= l
    
    @property
    def num_codes(self) -> int:
        return self.codebook_size
    
    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound z to [-1, 1] using tanh."""
        return torch.tanh(z)
    
    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantize bounded z to discrete levels.
        
        Input z should be in [-1, 1] (after bound()).
        """
        # z is in [-1, 1]
        # Shift to [0, 1] then scale to [0, L-1] then round
        half_levels = (self._levels.float() - 1) / 2  # [3.5, 2.0, 2.0] for [8,5,5]
        
        # Scale from [-1, 1] to [0, L-1]
        z_scaled = (z + 1) * half_levels  # Now in [0, L-1]
        z_quantized = torch.round(z_scaled)
        
        # Clamp element-wise (each dim has its own max)
        max_vals = self._levels.float() - 1
        z_quantized = torch.clamp(z_quantized, min=0)
        z_quantized = torch.minimum(z_quantized, max_vals)
        
        # Scale back to [-1, 1]
        z_out = z_quantized / half_levels - 1
        
        return z_out
    
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with quantization.
        
        Args:
            z: Input features [..., dim] where dim = len(levels)
        
        Returns:
            z_q: Quantized features (same shape as z)
            indices: Codebook indices [...] (flattened codes)
        """
        # Bound to [-1, 1]
        z_bounded = self.bound(z)
        
        # Quantize
        z_q = self.quantize(z_bounded)
        
        # Straight-through gradient
        z_q = z_bounded + (z_q - z_bounded).detach()
        
        # Compute indices
        indices = self.codes_to_indices(z_q)
        
        return z_q, indices
    
    def codes_to_indices(self, z_q: torch.Tensor) -> torch.Tensor:
        """Convert quantized codes to flat indices."""
        # z_q is in [-1, 1], convert to [0, L-1]
        half_levels = (self._levels.float() - 1) / 2
        codes = ((z_q + 1) * half_levels).round().long()
        
        # Flatten using basis
        indices = (codes * self._basis).sum(dim=-1)
        return indices
    
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert flat indices back to codes."""
        codes = []
        for i in range(self.dim - 1, -1, -1):
            codes.append(indices % self._levels[i])
            indices = indices // self._levels[i]
        codes = torch.stack(codes[::-1], dim=-1)
        
        # Convert to [-1, 1]
        half_levels = (self._levels.float() - 1) / 2
        return codes.float() / half_levels - 1


class FSQEncoder(nn.Module):
    """
    Wrapper that projects to FSQ dimension and quantizes.
    
    Replaces VectorQuantizer in the semantic encoder.
    
    Args:
        input_dim: Input feature dimension
        levels: FSQ levels per dimension
    """
    
    def __init__(self, input_dim: int, levels: List[int] = [8, 6, 5, 5, 5]):
        super().__init__()
        self.fsq = FSQ(levels)
        self.fsq_dim = len(levels)
        
        # Project to FSQ dimension
        self.proj_down = nn.Linear(input_dim, self.fsq_dim)
        self.proj_up = nn.Linear(self.fsq_dim, input_dim)
        
    @property
    def codebook_size(self) -> int:
        return self.fsq.codebook_size
    
    def forward(
        self, 
        z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass matching VectorQuantizer interface.
        
        Args:
            z: Input features [B, T, D]
        
        Returns:
            z_q: Quantized features [B, T, D]
            idx: Codebook indices [B, T]
            loss: Always 0 (no VQ loss needed)
            perplexity: Codebook usage perplexity
            used: Number of used codes in batch
        """
        # Project down
        z_low = self.proj_down(z)
        
        # Quantize
        z_q_low, indices = self.fsq(z_low)
        
        # Project back up
        z_q = self.proj_up(z_q_low)
        
        # NOTE: No outer STE here. The STE is already inside the fsq object for z_q_low.
        # This allows gradients to flow back through proj_up and proj_down.
        
        # Compute metrics - tuned for MPS using scatter_add (no CPU sync)
        with torch.no_grad():
            counts = self.fused_count_usage(indices)
            probs = counts / counts.sum().clamp_min(1.0)
            perplexity = torch.exp(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
            used = (counts > 0).sum()
        
        # FSQ doesn't need a reconstruction loss
        loss = torch.tensor(0.0, device=z.device)
        
        return z_q, indices, loss, perplexity, used
    
    def fused_count_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """GPU-resident counting using scatter_add (avoids CPU sync)."""
        flat_idx = indices.flatten()
        counts = torch.zeros(self.fsq.num_codes, device=indices.device, dtype=torch.float32)
        ones = torch.ones(flat_idx.shape[0], device=indices.device, dtype=torch.float32)
        
        # Parallel atomic add on GPU
        counts.scatter_add_(0, flat_idx, ones)
        
        return counts
        

    
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encode to indices."""
        z_low = self.proj_down(z)
        _, indices = self.fsq(z_low)
        return indices
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from indices."""
        z_q_low = self.fsq.indices_to_codes(indices)
        return self.proj_up(z_q_low)
