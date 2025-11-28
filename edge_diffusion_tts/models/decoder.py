"""
Edge-optimized Diffusion Decoder.
"""

from typing import Optional

import torch
import torch.nn as nn

from ..config import CFG
from ..layers import DiffusionTransformerBlock, SinusoidalTimeEmb, SinusoidalPositionalEmb


class EdgeDiffusionDecoder(nn.Module):
    """Edge-optimized diffusion decoder with AdaLN and Cross-Attention."""
    
    def __init__(self, cfg: CFG):
        super().__init__()
        H = cfg.hidden
        self.cfg = cfg
        
        self.token_emb = nn.Embedding(cfg.codebook_size, H)
        self.sem_proj = nn.Linear(cfg.semantic_dim, H)
        
        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmb(H), 
            nn.Linear(H, H), 
            nn.GELU(), 
            nn.Linear(H, H),
        )
        self.step_emb = nn.Embedding(16, H)
        
        # Input projection
        self.in_proj = nn.Linear(cfg.n_mels, H)
        
        # Positional embedding for mel sequence (CRITICAL for temporal structure)
        self.pos_emb = SinusoidalPositionalEmb(H, max_len=1000)
        
        # Positional embedding for semantic context (CRITICAL for alignment)
        self.context_pos_emb = SinusoidalPositionalEmb(H, max_len=512)
        
        # Transformer backbone with local attention
        self.layers = nn.ModuleList([
            DiffusionTransformerBlock(
                dim=H,
                context_dim=H,     # Semantic tokens dimension
                cond_dim=H,        # Timestep embedding dimension
                heads=cfg.heads,
                ffn_mult=cfg.ffn_mult,
                dropout=cfg.dropout,
                use_adaln=cfg.use_adaln,
                window_size=cfg.attn_window_size  # Local attention for memory efficiency
            )
            for _ in range(cfg.layers)
        ])
        
        # Output projection
        self.final_norm = nn.LayerNorm(H)
        self.out_proj = nn.Linear(H, cfg.n_mels)
        
        # Zero-init output projection
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, sem_idx: Optional[torch.Tensor] = None,
                step_idx: Optional[torch.Tensor] = None, sem_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_t: Noisy mels [B, T, n_mels]
            t: Timesteps [B]
            sem_idx: Semantic tokens [B, S]
            step_idx: Consistency step indices [B]
            sem_features: Continuous semantic features [B, S, semantic_dim]
        """
        # 1. Prepare conditioning
        t_cond = self.time_emb(t)  # [B, H]
        
        if step_idx is not None:
            t_cond = t_cond + self.step_emb(step_idx)
            
        # 2. Prepare context (semantic tokens)
        if sem_features is not None:
            # Pass continuous features for gradient flow (STE)
            context = self.sem_proj(sem_features)
        elif sem_idx is not None:
            # Use discrete indices (inference)
            context = self.token_emb(sem_idx)
        else:
            raise ValueError("Either sem_idx or sem_features must be provided")
            
        # Add positional information to context for alignment
        context = self.context_pos_emb(context)
        
        # 3. Input projection + positional encoding
        h = self.in_proj(x_t)  # [B, T, H]
        h = self.pos_emb(h)    # Add positional information
        
        # 4. Transformer layers
        for layer in self.layers:
            h = layer(h, context=context, cond=t_cond)
            
        # 5. Output projection
        # Use final norm if not using AdaLN for everything? 
        # Usually AdaLN replaces all LayerNorms.
        # But we kept LayerNorm in DiTBlock as backup or mixed.
        # Let's apply standard LayerNorm before output.
        h = self.final_norm(h)
        return self.out_proj(h)
