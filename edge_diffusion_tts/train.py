"""
Main training module for Edge Diffusion TTS.
"""

import gc
import json
import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .config import CFG
from .schedule import DiffusionSchedule
from .models import SemanticEncoder, EdgeDiffusionDecoder
from .training import ConsistencyTrainer
from .inference import EdgeInference
from .data import LJSpeechDataset, ensure_ljspeech, Collate
from .utils import normalize_mel, evaluate_model, visualize_generation, export_for_edge


def train(
    config_path: Optional[str] = None,
    resume_path: Optional[str] = None,
    export_onnx: bool = False,
    device_override: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    epochs_override: Optional[int] = None,
):
    """Main training function."""
    
    # Load or create config
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = CFG.from_dict(json.load(f))
    else:
        cfg = CFG()
    
    # Apply overrides
    if device_override:
        cfg.device = device_override
    if batch_size_override:
        cfg.batch_size = batch_size_override
    if epochs_override:
        cfg.diffusion_epochs = epochs_override
    
    cfg.setup_environment()
    cfg.print_config()
    
    # Setup data
    cfg.ljspeech_dir = ensure_ljspeech(cfg.data_root)
    
    print("Creating datasets...")
    train_ds = LJSpeechDataset(cfg.ljspeech_dir, split="train")
    val_ds = LJSpeechDataset(cfg.ljspeech_dir, split="val")
    
    collate = Collate(cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        drop_last=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, drop_last=True, collate_fn=collate
    )
    print(f"✓ Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # Initialize models
    print("\nInitializing models...")
    encoder = SemanticEncoder(cfg).to(cfg.device)
    decoder = EdgeDiffusionDecoder(cfg).to(cfg.device)
    
    enc_params = sum(p.numel() for p in encoder.proj.parameters()) + \
                 sum(p.numel() for p in encoder.vq.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Encoder (trainable): {enc_params:,} params")
    print(f"  Decoder: {dec_params:,} params ({dec_params * 4 / 1024 / 1024:.2f} MB FP32)")
    
    # Compile decoder for faster training (PyTorch 2.0+)
    if hasattr(torch, 'compile') and cfg.device != "mps":  # MPS doesn't support compile well
        print("  Compiling decoder with torch.compile...")
        decoder = torch.compile(decoder, mode="reduce-overhead")
    
    # Diffusion schedule
    schedule = DiffusionSchedule(cfg.diff_steps, cfg.beta_start, cfg.beta_end, cfg.device)
    
    # Training components
    consistency_trainer = ConsistencyTrainer(cfg, schedule, encoder, decoder)
    inference = EdgeInference(cfg, schedule, encoder, decoder)

    
    # Optimizer
    trainable_params = encoder.get_trainable_params() + list(decoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # AMP
    use_amp = cfg.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    # Logging
    run_dir = cfg.get_run_dir()
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
    tb = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
    
    print(f"\n  Output: {run_dir}")
    print(f"  TensorBoard: tensorboard --logdir {os.path.join(run_dir, 'tb')}")
    
    # Resume if specified
    global_step = 0
    best_val = float("inf")
    
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=cfg.device)
        encoder.load_state_dict(ckpt.get("encoder", {}), strict=False)
        decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
        global_step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"✓ Resumed from step {global_step}")
    
    # ===== PHASE 1: Standard Diffusion Training =====
    print(f"\n{'='*60}")
    print("  PHASE 1: Standard Diffusion Training")
    print(f"{'='*60}\n")
    
    for epoch in range(1, cfg.diffusion_epochs + 1):
        encoder.train()
        decoder.train()
        
        pbar = tqdm(train_loader, desc=f"[Diffusion] Epoch {epoch}/{cfg.diffusion_epochs}",
                    leave=True, ncols=120)
        epoch_loss = 0.0
        epoch_steps = 0
        
        for wav, mel, _ in pbar:
            global_step += 1
            epoch_steps += 1
            
            wav, mel = wav.to(cfg.device), mel.to(cfg.device)
            mel_n, _, _ = normalize_mel(mel)
            
            with torch.autocast(device_type=cfg.device.split(':')[0] if ':' in cfg.device else cfg.device,
                               enabled=use_amp):
                _, sem_idx, vq_loss, vq_ppl, vq_used = encoder(wav)
                
                B = wav.shape[0]
                t = torch.randint(1, cfg.diff_steps, (B,), device=cfg.device)
                noise = torch.randn_like(mel_n)
                x_t, _ = schedule.q_sample(mel_n, t, noise)
                
                eps_pred = decoder(x_t, t, sem_idx)
                Tm = min(eps_pred.shape[1], noise.shape[1])
                recon_loss = F.mse_loss(eps_pred[:, :Tm, :], noise[:, :Tm, :])
                loss = recon_loss + 0.1 * vq_loss
            
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "vq_ppl": f"{vq_ppl.item():.0f}", "vq_used": f"{vq_used.item()}"})
            
            # Log VQ health EVERY step (critical for debugging)
            tb.add_scalar("vq/perplexity", vq_ppl.item(), global_step)
            tb.add_scalar("vq/used_codes", vq_used.item(), global_step)
            
            if global_step % cfg.log_every_steps == 0:
                tb.add_scalar("phase1/loss", loss.item(), global_step)
                tb.add_scalar("phase1/recon_loss", recon_loss.item(), global_step)
                tb.add_scalar("phase1/vq_loss", vq_loss.item(), global_step)
            
            if global_step % cfg.plot_every_steps == 0:
                visualize_generation(val_loader, encoder, decoder, schedule,
                                   inference, cfg, global_step, run_dir, [4, 8, 16])
                encoder.train()
                decoder.train()
            
            if global_step % cfg.val_every_steps == 0:
                val_loss = evaluate_model(val_loader, encoder, decoder, schedule, cfg)
                tb.add_scalar("val/loss", val_loss, global_step)
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                              "step": global_step, "best_val": best_val},
                              os.path.join(run_dir, "best_diffusion.pt"))
                encoder.train()
                decoder.train()
            
            if global_step % 100 == 0:
                gc.collect()
                if cfg.device == "cuda":
                    torch.cuda.empty_cache()
    
    # Save Phase 1
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                "step": global_step, "phase": "diffusion_complete"},
               os.path.join(run_dir, "checkpoint_phase1.pt"))
    
    # ===== PHASE 2: Progressive Distillation =====
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Progressive Distillation (→ {cfg.progressive_target_steps} steps)")
    print(f"{'='*60}\n")
    
    consistency_trainer.init_teacher()
    
    step_schedule = []
    s = cfg.diff_steps
    while s > cfg.progressive_target_steps:
        s = max(s // 2, cfg.progressive_target_steps)
        step_schedule.append(s)
    
    for pg in optimizer.param_groups:
        pg['lr'] = cfg.lr_consistency
    
    for target_steps in step_schedule:
        print(f"\n  Training for {target_steps} steps...")
        
        for epoch in range(1, cfg.progressive_epochs_per_halving + 1):
            encoder.train()
            decoder.train()
            
            pbar = tqdm(train_loader, desc=f"[Prog {target_steps}] Ep {epoch}",
                       leave=True, ncols=120)
            
            for wav, mel, _ in pbar:
                global_step += 1
                wav, mel = wav.to(cfg.device), mel.to(cfg.device)
                _, sem_idx, vq_loss, _, _ = encoder(wav)
                
                loss, _, _ = consistency_trainer.progressive_distillation_loss(mel, sem_idx, target_steps)
                loss = loss + 0.05 * vq_loss
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                optimizer.step()
                consistency_trainer.update_teacher()
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "steps": target_steps})
        
        consistency_trainer.init_teacher()
    
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                "step": global_step, "phase": "progressive_complete"},
               os.path.join(run_dir, "checkpoint_phase2.pt"))
    
    # ===== PHASE 3: Consistency Fine-tuning =====
    print(f"\n{'='*60}")
    print("  PHASE 3: Consistency Fine-tuning (1-4 step generation)")
    print(f"{'='*60}\n")
    
    for epoch in range(1, cfg.consistency_epochs + 1):
        encoder.train()
        decoder.train()
        
        pbar = tqdm(train_loader, desc=f"[Consistency] Ep {epoch}/{cfg.consistency_epochs}",
                   leave=True, ncols=120)
        
        for wav, mel, _ in pbar:
            global_step += 1
            wav, mel = wav.to(cfg.device), mel.to(cfg.device)
            _, sem_idx, vq_loss, _, _ = encoder(wav)
            
            loss, _, _ = consistency_trainer.consistency_loss(mel, sem_idx)
            loss = loss + 0.05 * vq_loss
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            optimizer.step()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # ===== SAVE FINAL =====
    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE!")
    print(f"{'='*60}\n")
    
    final_path = os.path.join(run_dir, "edge_model_final.pt")
    torch.save({
        "encoder_proj": encoder.proj.state_dict(),
        "encoder_vq": encoder.vq.state_dict(),
        "decoder": decoder.state_dict(),
        "cfg": cfg.to_dict(),
    }, final_path)
    print(f"✓ Final model: {final_path}")
    print(f"  Size: {os.path.getsize(final_path) / 1024 / 1024:.2f} MB")
    
    visualize_generation(val_loader, encoder, decoder, schedule, inference, cfg, global_step, run_dir, [1, 2, 4])
    
    if export_onnx:
        export_for_edge(decoder, cfg, os.path.join(run_dir, "edge_decoder.onnx"))
    
    tb.close()
    return encoder, decoder, inference
