"""
Semantic Encoder module.

Uses HuBERT for feature extraction and VQ/FSQ for discretization.
"""

import torch
import torch.nn as nn

from transformers import HubertModel

from ..config import CFG
from .vq import VectorQuantizer
from .fsq import FSQEncoder


class SemanticEncoder(nn.Module):
    """
    Semantic encoder using HuBERT features with VQ/FSQ discretization.
    
    Pipeline:
    1. Extract HuBERT features from waveform
    2. Project to semantic dimension
    3. Quantize with VQ or FSQ
    
    Args:
        cfg: Configuration object
    """
    
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        
        # Load HuBERT (frozen)
        self.hubert = HubertModel.from_pretrained(cfg.hubert_id)
        self.hubert.eval()
        for p in self.hubert.parameters():
            p.requires_grad = False
        
        # Projection layers
        self.proj = nn.Sequential(
            nn.Linear(768, cfg.semantic_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.semantic_dim),
            nn.Linear(cfg.semantic_dim, cfg.semantic_dim),
        )
        
        # Quantizer: FSQ or VQ based on config
        if getattr(cfg, 'use_fsq', False):
            self.vq = FSQEncoder(cfg.semantic_dim, cfg.fsq_levels)
            self.codebook_size = self.vq.codebook_size
        else:
            self.vq = VectorQuantizer(
                cfg.semantic_dim,
                cfg.codebook_size,
                commit=cfg.vq_commit
            )
            self.codebook_size = cfg.codebook_size
    
    @torch.no_grad()
    def extract_hubert(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """
        Extract HuBERT features from waveform.
        
        Args:
            wav_16k: Waveform at 16kHz [B, T_audio]
        
        Returns:
            HuBERT features [B, T_feat, 768]
        """
        out = self.hubert(wav_16k, output_hidden_states=True)
        return out.hidden_states[self.cfg.hubert_layer]
    
    def forward(
        self,
        wav_16k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: extract and quantize semantic features.
        
        Args:
            wav_16k: Waveform at 16kHz [B, T_audio]
        
        Returns:
            z_q: Quantized features [B, T_feat, semantic_dim]
            idx: Codebook indices [B, T_feat]
            vq_loss: VQ loss
            perplexity: Codebook perplexity
            used: Number of used codebook entries
        """
        # Extract HuBERT features (frozen, but ensure detached)
        h = self.extract_hubert(wav_16k).detach()
        
        # Project to semantic dimension
        z = self.proj(h)
        
        # Quantize
        z_q, idx, vq_loss, perplexity, used = self.vq(z)
        
        return z_q, idx, vq_loss, perplexity, used
    
    def encode(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """
        Encode waveform to semantic token indices.
        
        Args:
            wav_16k: Waveform at 16kHz [B, T_audio]
        
        Returns:
            Codebook indices [B, T_feat]
        """
        with torch.no_grad():
            h = self.extract_hubert(wav_16k)
            z = self.proj(h)
            return self.vq.encode(z)
    
    def decode_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Decode token indices to continuous features.
        
        Args:
            idx: Codebook indices [B, T_feat]
        
        Returns:
            Continuous features [B, T_feat, semantic_dim]
        """
        return self.vq.decode(idx)
    
    def get_trainable_params(self) -> list:
        """Get list of trainable parameters (excludes frozen HuBERT)."""
        return list(self.proj.parameters()) + list(self.vq.parameters())
