"""
Vector Quantization module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with straight-through gradient estimation.
    
    Quantizes continuous representations to discrete codebook entries.
    Uses EMA codebook updates for stability.
    
    Args:
        dim: Feature dimension
        codebook_size: Number of codebook entries
        commit: Commitment loss weight
        decay: EMA decay for codebook updates
        epsilon: Small constant for numerical stability
        reset_unused_every: Reset unused codes every N updates (0 = disabled)
    """
    
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        commit: float = 0.25,
        decay: float = 0.99,  # Changed: Enable EMA by default
        epsilon: float = 1e-5,
        reset_unused_every: int = 100,  # New: Reset dead codes
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commit = commit
        self.decay = decay
        self.epsilon = epsilon
        self.reset_unused_every = reset_unused_every
        
        # Codebook - initialize with larger variance for diversity
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0)  # Changed: Normal init
        
        # EMA tracking (always enabled now)
        self.register_buffer('ema_cluster_size', torch.ones(codebook_size))
        self.register_buffer('ema_w', self.codebook.weight.clone())
        self.register_buffer('update_count', torch.tensor(0))

    
    def forward(
        self,
        z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with quantization.
        
        Args:
            z: Input features [B, T, D]
        
        Returns:
            z_q: Quantized features [B, T, D]
            idx: Codebook indices [B, T]
            vq_loss: VQ loss (codebook + commitment)
            perplexity: Codebook usage perplexity
            used: Number of used codebook entries
        """
        B, T, D = z.shape
        flat = z.reshape(-1, D)
        
        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1, keepdim=True).t()
        )
        
        # Find nearest codebook entry
        idx = dist.argmin(dim=1)
        z_q = self.codebook(idx).view(B, T, D)
        
        # Compute losses
        if self.training:
            codebook_loss = F.mse_loss(z_q, z.detach())
            commit_loss = F.mse_loss(z_q.detach(), z)
            vq_loss = codebook_loss + self.commit * commit_loss
            
            # Optional EMA update
            if self.decay > 0:
                self._ema_update(flat, idx)
        else:
            vq_loss = torch.tensor(0.0, device=z.device)
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # Compute metrics
        with torch.no_grad():
            counts = torch.bincount(idx, minlength=self.codebook_size).float()
            probs = counts / counts.sum().clamp_min(1.0)
            perplexity = torch.exp(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
            used = (counts > 0).sum()
        
        return z_q, idx.view(B, T), vq_loss, perplexity, used
    
    def _ema_update(self, flat: torch.Tensor, idx: torch.Tensor) -> None:
        """Update codebook with exponential moving average and dead code reset."""
        # One-hot encoding
        encodings = F.one_hot(idx, self.codebook_size).float()
        
        # Update cluster sizes
        n = encodings.sum(0)
        self.ema_cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        
        # Update embeddings
        dw = encodings.t() @ flat
        self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        
        # Normalize
        n_norm = self.ema_cluster_size.clamp_min(self.epsilon)
        self.codebook.weight.data.copy_(self.ema_w / n_norm.unsqueeze(1))
        
        # Dead code reset
        self.update_count += 1
        if self.reset_unused_every > 0 and self.update_count % self.reset_unused_every == 0:
            # Find codes that haven't been used recently (low cluster size)
            dead_mask = self.ema_cluster_size < 1.0
            num_dead = dead_mask.sum().item()
            
            if num_dead > 0 and flat.shape[0] > 0:
                # Sample random vectors from current batch to replace dead codes
                num_replace = min(num_dead, flat.shape[0])
                replace_idx = torch.randperm(flat.shape[0], device=flat.device)[:num_replace]
                new_vectors = flat[replace_idx]
                
                # Find which dead codes to replace
                dead_indices = dead_mask.nonzero(as_tuple=True)[0][:num_replace]
                
                # Reset the dead codes
                self.codebook.weight.data[dead_indices] = new_vectors
                self.ema_w[dead_indices] = new_vectors
                self.ema_cluster_size[dead_indices] = 1.0

    
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encode features to codebook indices."""
        B, T, D = z.shape
        flat = z.reshape(-1, D)
        
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1, keepdim=True).t()
        )
        
        return dist.argmin(dim=1).view(B, T)
    
    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices to features."""
        return self.codebook(idx)
