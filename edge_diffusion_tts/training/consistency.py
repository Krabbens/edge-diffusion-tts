"""
Consistency and Progressive Distillation Trainer.
"""

import copy
from typing import Tuple

import torch
import torch.nn.functional as F

from ..config import CFG
from ..schedule import DiffusionSchedule
from ..models import SemanticEncoder, EdgeDiffusionDecoder
from ..utils.audio import normalize_mel


class ConsistencyTrainer:
    """
    Implements progressive distillation and consistency training
    for few-step inference on edge devices.
    
    Reference: "Consistency Models" (Song et al., 2023)
    """
    
    def __init__(self, cfg: CFG, schedule: DiffusionSchedule,
                 encoder: SemanticEncoder, decoder: EdgeDiffusionDecoder):
        self.cfg = cfg
        self.schedule = schedule
        self.encoder = encoder
        self.decoder = decoder
        self.device = cfg.device
        
        self.teacher = None
        self.ema_decay = 0.999
        self.current_steps = cfg.diff_steps
    
    def init_teacher(self):
        """Initialize EMA teacher from current decoder."""
        self.teacher = copy.deepcopy(self.decoder)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def update_teacher(self):
        """Update teacher with EMA of student weights."""
        if self.teacher is None:
            return
        for t_param, s_param in zip(self.teacher.parameters(), self.decoder.parameters()):
            t_param.data.lerp_(s_param.data, 1 - self.ema_decay)
    
    def get_timestep_pairs(self, batch_size: int, num_steps: int):
        """Get pairs of timesteps for progressive distillation."""
        stride = self.cfg.diff_steps // num_steps
        step_indices = torch.randint(0, num_steps, (batch_size,), device=self.device)
        t = (step_indices + 1) * stride - 1
        t_prev = (t - stride).clamp(min=0)
        return t.long(), t_prev.long(), step_indices
    
    def progressive_distillation_loss(self, mel: torch.Tensor, sem_idx: torch.Tensor,
                                       num_steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Progressive distillation: train student to match teacher's output."""
        B = mel.shape[0]
        mel_n, mean, std = normalize_mel(mel)
        
        t, t_prev, step_idx = self.get_timestep_pairs(B, num_steps)
        noise = torch.randn_like(mel_n)
        x_t, _ = self.schedule.q_sample(mel_n, t, noise)
        
        # Model predicts v (velocity), not epsilon!
        v_student = self.decoder(x_t.float(), t, sem_idx, step_idx).float()
        
        Tm = min(v_student.shape[1], mel_n.shape[1])
        v_student, x_t, mel_n = v_student[:, :Tm, :], x_t[:, :Tm, :], mel_n[:, :Tm, :]
        
        # Use v-prediction to get x0
        x0_student = self.schedule.predict_x0_from_v(x_t, t, v_student)
        
        if self.teacher is not None and num_steps < self.cfg.diff_steps:
            with torch.no_grad():
                v_teacher = self.teacher(x_t, t, sem_idx, step_idx)[:, :Tm, :]
                x0_teacher = self.schedule.predict_x0_from_v(x_t, t, v_teacher)
            loss = F.mse_loss(x0_student, x0_teacher.detach())
        else:
            # v-prediction target: v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0
            v_target = self.schedule.get_v_target(mel_n, noise[:, :Tm, :], t)
            loss = F.mse_loss(v_student, v_target)
        
        return loss, x0_student, mel_n

    
    def consistency_loss(self, mel: torch.Tensor, sem_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Consistency loss: model should predict same x0 from any timestep."""
        B = mel.shape[0]
        mel_n, mean, std = normalize_mel(mel)
        
        t1 = torch.randint(1, self.cfg.diff_steps, (B,), device=self.device)
        t2 = torch.randint(1, self.cfg.diff_steps, (B,), device=self.device)
        noise = torch.randn_like(mel_n)
        
        x_t1, _ = self.schedule.q_sample(mel_n, t1, noise)
        x_t2, _ = self.schedule.q_sample(mel_n, t2, noise)
        
        step_idx = torch.zeros(B, device=self.device, dtype=torch.long)
        
        # Model predicts v (velocity), not epsilon!
        v1 = self.decoder(x_t1.float(), t1, sem_idx, step_idx).float()
        v2 = self.decoder(x_t2.float(), t2, sem_idx, step_idx).float()
        
        Tm = min(v1.shape[1], v2.shape[1], mel_n.shape[1])
        x_t1, x_t2 = x_t1[:, :Tm, :], x_t2[:, :Tm, :]
        v1, v2 = v1[:, :Tm, :], v2[:, :Tm, :]
        
        # Use v-prediction to get x0
        x0_pred1 = self.schedule.predict_x0_from_v(x_t1, t1, v1)
        x0_pred2 = self.schedule.predict_x0_from_v(x_t2, t2, v2)
        
        consistency_loss = F.mse_loss(x0_pred1, x0_pred2.detach())
        mel_n_trimmed = mel_n[:, :Tm, :]
        recon_loss = 0.5 * (F.mse_loss(x0_pred1, mel_n_trimmed) + F.mse_loss(x0_pred2, mel_n_trimmed))
        
        return consistency_loss + recon_loss, x0_pred1, mel_n_trimmed

