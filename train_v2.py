#!/usr/bin/env python3
"""
IMPROVED Training Script v2 for Edge Diffusion TTS.

Fixes from deep analysis:
1. ✅ v-prediction (not ε-prediction)
2. ✅ Edge architecture (160H, 4L, 4H)
3. ✅ Cosine LR schedule with warmup
4. ✅ CFG dropout (10%)
5. ✅ DPM-Solver++ for validation
6. ✅ Better regularization (dropout 0.2)
"""

import sys
sys.path.insert(0, ".")

import os
import random
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from edge_diffusion_tts.config import CFG
from edge_diffusion_tts.models import EdgeDiffusionDecoder
from edge_diffusion_tts.models.fsq import FSQEncoder
from edge_diffusion_tts.schedule import DiffusionSchedule, DPMSolverPP
from edge_diffusion_tts.data import LJSpeechDataset, ensure_ljspeech, Collate
from edge_diffusion_tts.utils.audio import normalize_mel


class FastSemanticEncoder(nn.Module):
    """Fast semantic encoder using DistilHuBERT + FSQ."""
    
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        
        from transformers import HubertModel
        # Use HuBERT Base (Layer 9) for full phonetic content
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.hubert_dim = 768
        self.hubert_layer = 9  # Layer 9 is standard for speech resynthesis
        
        self.hubert.eval()
        for p in self.hubert.parameters():
            p.requires_grad = False
        
        self.proj = nn.Sequential(
            nn.Linear(self.hubert_dim, cfg.semantic_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.semantic_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.semantic_dim, cfg.semantic_dim),
        )
        
        self.fsq = FSQEncoder(cfg.semantic_dim, cfg.fsq_levels)
        self.codebook_size = self.fsq.codebook_size
    
    @torch.no_grad()
    def extract_hubert(self, wav_16k: torch.Tensor) -> torch.Tensor:
        out = self.hubert(wav_16k, output_hidden_states=True)
        if hasattr(out, 'hidden_states'):
            feat = out.hidden_states[min(self.hubert_layer, len(out.hidden_states)-1)]
        else:
            feat = out.last_hidden_state
        return feat.float()
    
    def forward(self, wav_16k: torch.Tensor):
        h = self.extract_hubert(wav_16k).detach()
        z = self.proj(h)
        z_q, idx, vq_loss, perplexity, used = self.fsq(z)
        return z_q, idx, vq_loss, perplexity, used
    
    def get_trainable_params(self):
        return list(self.proj.parameters()) + list(self.fsq.parameters())


def cosine_lr_schedule(optimizer, step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    """Cosine LR schedule with linear warmup."""
    if step < warmup_steps:
        lr = base_lr * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_step(
    batch: Tuple[torch.Tensor, torch.Tensor, list],
    encoder: FastSemanticEncoder,
    decoder: EdgeDiffusionDecoder,
    schedule: DiffusionSchedule,
    cfg: CFG,
    cfg_dropout_prob: float = 0.1
) -> Tuple[torch.Tensor, dict]:
    """Single training step with v-prediction and CFG dropout."""
    
    wav, mel, fids = batch
    device = cfg.device
    wav, mel = wav.to(device), mel.to(device)
    
    B = mel.shape[0]
    mel_n, mean, std = normalize_mel(mel)
    
    # Encode semantic features
    z_q, sem_idx, vq_loss, perplexity, used = encoder(wav)
    
    # CFG dropout: randomly drop conditioning
    if random.random() < cfg_dropout_prob:
        z_q = torch.zeros_like(z_q)
    
    # Sample timestep
    max_t = cfg.max_timestep
    t = torch.randint(1, max_t, (B,), device=device)
    
    # Add noise
    noise = torch.randn_like(mel_n)
    x_t, _ = schedule.q_sample(mel_n, t, noise)
    
    step_idx = torch.zeros(B, device=device, dtype=torch.long)
    
    # Forward pass - predicts v
    v_pred = decoder(x_t, t, sem_features=z_q, step_idx=step_idx)
    
    # Align sequence lengths
    min_len = min(v_pred.shape[1], mel_n.shape[1])
    v_pred = v_pred[:, :min_len, :]
    mel_n = mel_n[:, :min_len, :]
    noise = noise[:, :min_len, :]
    x_t = x_t[:, :min_len, :]
    
    # v-prediction target
    v_target = schedule.get_v_target(mel_n, noise, t)
    
    # Loss
    diff_loss = F.mse_loss(v_pred, v_target)
    loss = diff_loss + vq_loss * cfg.vq_commit
    
    # Metrics
    with torch.no_grad():
        x0_pred = schedule.predict_x0_from_v(x_t, t, v_pred)
        x0_mse = F.mse_loss(x0_pred, mel_n).item()
        x0_cos = F.cosine_similarity(
            x0_pred.flatten(1), mel_n.flatten(1), dim=1
        ).mean().item()
    
    metrics = {
        'loss': loss.item(),
        'diff_loss': diff_loss.item(),
        'perplexity': perplexity.item(),
        'x0_cos': x0_cos,
    }
    
    return loss, metrics


@torch.no_grad()
def validate(loader, encoder, decoder, schedule, cfg, num_batches=4):
    """Validation using DPM-Solver++."""
    encoder.eval()
    decoder.eval()
    
    device = cfg.device
    dpm = DPMSolverPP(schedule, order=2, predict_x0=False)
    
    total_cos = 0
    count = 0
    
    for i, (wav, mel, fids) in enumerate(loader):
        if i >= num_batches:
            break
        
        wav, mel = wav.to(device), mel.to(device)
        mel_n, mean, std = normalize_mel(mel)
        
        z_q, sem_idx, _, perplexity, _ = encoder(wav)
        
        B = wav.shape[0]
        T_out = z_q.shape[1] * 2
        x_T = torch.randn(B, T_out, cfg.n_mels, device=device)
        
        x0_pred = dpm.sample(decoder, x_T, z_q, num_steps=4)
        
        min_len = min(x0_pred.shape[1], mel_n.shape[1])
        cos = F.cosine_similarity(
            x0_pred[:, :min_len].flatten(1),
            mel_n[:, :min_len].flatten(1),
            dim=1
        ).mean().item()
        
        total_cos += cos
        count += 1
    
    encoder.train()
    decoder.train()
    
    return {'val_cos': total_cos / count if count > 0 else 0}


def main():
    print("=" * 60)
    print("  IMPROVED Edge Diffusion TTS Training v2")
    print("=" * 60)
    
    cfg = CFG()
    cfg.use_fsq = True
    cfg.use_v_prediction = True
    cfg.batch_size = 8
    cfg.grad_accumulation = 4
    device = cfg.device
    
    print(f"\n📊 Config: {cfg.hidden}H, {cfg.layers}L, {cfg.heads}H, dropout={cfg.dropout}")
    print(f"   Batch: {cfg.batch_size} x {cfg.grad_accumulation} = {cfg.batch_size * cfg.grad_accumulation}")
    
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    
    # Data
    cfg.ljspeech_dir = ensure_ljspeech(cfg.data_root)
    train_ds = LJSpeechDataset(cfg.ljspeech_dir, split="train")
    val_ds = LJSpeechDataset(cfg.ljspeech_dir, split="val")
    
    collate = Collate(cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, drop_last=True, collate_fn=collate
    )
    
    print(f"📂 Train: {len(train_loader)} batches | Val: {len(val_ds)} samples")
    
    # Models
    encoder = FastSemanticEncoder(cfg).to(device)
    cfg.codebook_size = encoder.codebook_size
    decoder = EdgeDiffusionDecoder(cfg).to(device)
    schedule = DiffusionSchedule(cfg.diff_steps, cfg.beta_start, cfg.beta_end, device)
    
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"🏗️ Decoder: {dec_params:,} params ({dec_params * 4 / 1e6:.2f} MB)")
    
    if hasattr(torch, 'compile') and device.startswith('cuda'):
        decoder = torch.compile(decoder, mode="reduce-overhead")
    
    # Optimizer
    trainable_params = encoder.get_trainable_params() + list(decoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=0.05)
    
    # AMP
    use_amp = device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Logging
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(run_dir, "tb"))
    
    epochs = cfg.diffusion_epochs
    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    
    print(f"⚙️ Training: {epochs} epochs, {total_steps} steps, warmup={warmup_steps}")
    print("=" * 60 + "\n")
    
    global_step = 0
    best_val_cos = 0
    
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        epoch_count = 0
        
        for batch_idx, batch in enumerate(pbar):
            lr = cosine_lr_schedule(optimizer, global_step, total_steps, warmup_steps, cfg.lr)
            
            with torch.amp.autocast(device, dtype=torch.float16, enabled=use_amp):
                loss, metrics = train_step(batch, encoder, decoder, schedule, cfg, cfg_dropout_prob=0.1)
                loss = loss / cfg.grad_accumulation
            
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % cfg.grad_accumulation == 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                    optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += metrics['loss']
            epoch_count += 1
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.3f}",
                'cos': f"{metrics['x0_cos']:.3f}",
                'lr': f"{lr:.1e}"
            })
            
            if global_step % cfg.log_every_steps == 0:
                writer.add_scalar('train/loss', metrics['loss'], global_step)
                writer.add_scalar('train/x0_cos', metrics['x0_cos'], global_step)
                writer.add_scalar('lr', lr, global_step)
        
        # Validation
        val_metrics = validate(val_loader, encoder, decoder, schedule, cfg)
        
        print(f"\n📊 Epoch {epoch+1} | loss={epoch_loss/epoch_count:.4f} | val_cos={val_metrics['val_cos']:.4f}")
        writer.add_scalar('val/cos', val_metrics['val_cos'], epoch)
        
        if val_metrics['val_cos'] > best_val_cos:
            best_val_cos = val_metrics['val_cos']
            print(f"  ✨ New best! cos={best_val_cos:.4f}")
            
            torch.save({
                'epoch': epoch + 1,
                'encoder_proj': encoder.proj.state_dict(),
                'encoder_fsq': encoder.fsq.state_dict(),
                'decoder': decoder.state_dict() if not hasattr(decoder, '_orig_mod') else decoder._orig_mod.state_dict(),
                'val_cos': best_val_cos,
            }, os.path.join(run_dir, "best_model.pt"))
    
    print(f"\n✅ Done! Best val_cos: {best_val_cos:.4f}")
    print(f"   Saved to: {run_dir}")
    writer.close()


if __name__ == "__main__":
    main()
