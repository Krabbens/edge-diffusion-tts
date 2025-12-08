"""
Visualization and evaluation utilities.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..config import CFG
from .audio import normalize_mel, denormalize_mel


@torch.no_grad()
def evaluate_model(val_loader, encoder, decoder, schedule, cfg: CFG, max_batches: int = 4) -> float:
    """Evaluate model on validation set."""
    encoder.eval()
    decoder.eval()
    
    total_loss, n = 0.0, 0
    
    for i, (wav, mel, _) in enumerate(val_loader):
        if i >= max_batches:
            break
        
        wav, mel = wav.to(cfg.device), mel.to(cfg.device)
        mel_n, _, _ = normalize_mel(mel)
        _, sem_idx, _, _, _ = encoder(wav)
        
        B = wav.shape[0]
        t = torch.randint(1, cfg.diff_steps, (B,), device=cfg.device)
        noise = torch.randn_like(mel_n)
        x_t, _ = schedule.q_sample(mel_n, t, noise)
        
        eps_pred = decoder(x_t, t, sem_idx)
        Tm = min(eps_pred.shape[1], noise.shape[1])
        loss = F.mse_loss(eps_pred[:, :Tm, :], noise[:, :Tm, :])
        total_loss += loss.item()
        n += 1
    
    return total_loss / max(n, 1)


@torch.no_grad()
def visualize_generation(val_loader, encoder, decoder, schedule, inference,
                         cfg: CFG, step: int, run_dir: str, 
                         num_steps_list: List[int] = None) -> str:
    """Visualize generation quality for different number of steps."""
    if num_steps_list is None:
        num_steps_list = [1, 2, 4]
    
    encoder.eval()
    decoder.eval()
    
    wav, mel, fids = next(iter(val_loader))
    wav, mel = wav[:1].to(cfg.device), mel[:1].to(cfg.device)
    mel_n, mean, std = normalize_mel(mel)
    _, sem_idx, _, _, _ = encoder(wav)
    
    mel_gt = mel.squeeze(0).transpose(0, 1).cpu().numpy()
    
    n_plots = 1 + len(num_steps_list)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))
    
    axes[0].imshow(mel_gt, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title(f"Ground Truth | File: {fids[0]}", fontweight="bold")
    axes[0].set_ylabel("Mel bin")
    
    for i, num_steps in enumerate(num_steps_list):
        mel_gen = inference.generate_mel(sem_idx, num_steps=num_steps)
        mel_gen = denormalize_mel(mel_gen, mean, std)
        mel_gen_np = mel_gen.squeeze(0).transpose(0, 1).cpu().numpy()
        
        min_len = min(mel_gt.shape[1], mel_gen_np.shape[1])
        mse = np.mean((mel_gt[:, :min_len] - mel_gen_np[:, :min_len]) ** 2)
        
        ax = axes[i + 1]
        ax.imshow(mel_gen_np, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"Generated ({num_steps} step{'s' if num_steps > 1 else ''}) | MSE: {mse:.4f}")
        ax.set_ylabel("Mel bin")
    
    axes[-1].set_xlabel("Frame")
    plt.suptitle(f"Step {step}: Few-Step Generation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    out_path = os.path.join(run_dir, "samples", f"gen_step_{step:06d}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return out_path
