"""
Inference module for Edge Diffusion TTS.
"""

import torch

from .config import CFG
from .schedule import DiffusionSchedule
from .models import SemanticEncoder, EdgeDiffusionDecoder


class EdgeInference:
    """Optimized inference for edge devices with 1-4 steps."""
    
    def __init__(self, cfg: CFG, schedule: DiffusionSchedule,
                 encoder: SemanticEncoder, decoder: EdgeDiffusionDecoder):
        self.cfg = cfg
        self.schedule = schedule
        self.encoder = encoder
        self.decoder = decoder
        self.device = cfg.device
    
    @torch.no_grad()
    def generate_mel(self, sem_idx: torch.Tensor, num_steps: int = 4,
                     temperature: float = 1.0) -> torch.Tensor:
        """Generate mel-spectrogram from semantic tokens."""
        self.encoder.eval()
        self.decoder.eval()
        
        B, T_sem = sem_idx.shape[0], sem_idx.shape[1]
        T_out = T_sem * 2
        
        x = torch.randn(B, T_out, self.cfg.n_mels, device=self.device) * temperature
        
        stride = self.cfg.diff_steps // num_steps
        timesteps = list(range(self.cfg.diff_steps - 1, 0, -stride))[:num_steps]
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
            step_idx = torch.full((B,), i, device=self.device, dtype=torch.long)
            t_prev = max(t - stride, 0)
            t_prev_tensor = torch.full((B,), t_prev, device=self.device, dtype=torch.long)
            
            eps_pred = self.decoder(x, t_tensor, sem_idx, step_idx)
            
            if eps_pred.shape[1] != x.shape[1]:
                min_len = min(eps_pred.shape[1], x.shape[1])
                x = x[:, :min_len, :]
                eps_pred = eps_pred[:, :min_len, :]
            
            x, x0_pred = self.schedule.get_ddim_step(x, t_tensor, t_prev_tensor, eps_pred, eta=0.0)
        
        return x0_pred
    
    @torch.no_grad()
    def generate_from_audio(self, wav: torch.Tensor, num_steps: int = 4) -> torch.Tensor:
        """Generate mel from reference audio."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.device)
        _, sem_idx, _, _, _ = self.encoder(wav)
        return self.generate_mel(sem_idx, num_steps)
