
import sys
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import torchaudio.transforms as T
import torchaudio.functional as AF
from edge_diffusion_tts.config import CFG
from edge_diffusion_tts.models import EdgeDiffusionDecoder
from edge_diffusion_tts.schedule import DiffusionSchedule, DPMSolverPP
from edge_diffusion_tts.models.fsq import FSQEncoder
from edge_diffusion_tts.utils.audio import normalize_mel, denormalize_mel
import os
import tqdm

class FastSemanticEncoder(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        from transformers import HubertModel
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.hubert_dim = 768
        self.hubert_layer = 9
        self.hubert.eval()
        for p in self.hubert.parameters(): p.requires_grad = False
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
        feat = out.hidden_states[min(self.hubert_layer, len(out.hidden_states)-1)] if hasattr(out, 'hidden_states') else out.last_hidden_state
        return feat.float()
    
    def forward(self, wav_16k: torch.Tensor):
        h = self.extract_hubert(wav_16k).detach()
        z = self.proj(h)
        z_q, idx, vq_loss, perplexity, used = self.fsq(z)
        return z_q

def main():
    cfg = CFG()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg.device = device
    print(f"Device: {device}")
    
    # --- Configuration ---
    chunk_seconds = 2.0  # Match training
    overlap_seconds = 0.5
    refine_strength = 1.0
    refine_steps = 150  # Maximum quality
    cfg_scale = 1.0  # No CFG
    
    # 1. Load Checkpoints
    teacher_runs_dir = "run_edge_diffusion"
    latest_run = sorted([d for d in os.listdir(teacher_runs_dir) if d.startswith("run_")])[-1]
    teacher_ckpt_path = os.path.join(teacher_runs_dir, latest_run, "best_model.pt")
    teacher_state = torch.load(teacher_ckpt_path, map_location=device)
    
    encoder = FastSemanticEncoder(cfg).to(device)
    encoder.proj.load_state_dict(teacher_state["encoder_proj"])
    encoder.fsq.load_state_dict(teacher_state["encoder_fsq"])
    encoder.eval()
    cfg.codebook_size = encoder.codebook_size
    
    teacher_decoder = EdgeDiffusionDecoder(cfg).to(device)
    teacher_decoder.load_state_dict(teacher_state["decoder"])
    teacher_decoder.eval()
    
    student_ckpt_path = "distilled_stage_4.pt"
    student_decoder = EdgeDiffusionDecoder(cfg).to(device)
    student_decoder.load_state_dict(torch.load(student_ckpt_path, map_location=device))
    student_decoder.eval()
    
    schedule = DiffusionSchedule(cfg.diff_steps, cfg.beta_start, cfg.beta_end, device)
    
    # DPM-Solver++ for fast sampling
    dpm_solver = DPMSolverPP(schedule, order=2, predict_x0=False)  # v-prediction model
    
    # --- Frequency Coherence Network ---
    from frequency_coherence_net import FrequencyCoherenceNet
    fcn = FrequencyCoherenceNet(n_mels=cfg.n_mels, hidden=32, n_blocks=4).to(device)
    fcn.load_state_dict(torch.load("fcn_best.pt", map_location=device))
    fcn.eval()
    print("Loaded Frequency Coherence Network")
    
    # --- Transforms ---
    inv_mel = T.InverseMelScale(n_stft=cfg.n_fft//2+1, n_mels=cfg.n_mels, sample_rate=cfg.sample_rate).to("cpu")
    griffin = T.GriffinLim(n_fft=cfg.n_fft, n_iter=32, win_length=cfg.win_length, hop_length=cfg.hop_length).to("cpu")
    mel_transform = T.MelSpectrogram(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.win_length, 
        hop_length=cfg.hop_length, f_min=cfg.f_min, f_max=cfg.f_max, 
        n_mels=cfg.n_mels, power=2.0, normalized=False
    ).to(device)

    # --- In-painting Sampling ---
    def inpaint_student_sample(x_shape, sem_features, known_mel=None, overlap_len=0, num_steps=4):
        B = 1
        x_curr = torch.randn(x_shape, device=device)
        
        times = torch.linspace(cfg.diff_steps-1, 0, num_steps+1, device=device).long()
        times = times[:-1] 
        # Calculate conditioning mask
        # If known_mel is provided, it's the TAIL of previous chunk.
        # We enforce it on the HEAD of current chunk.
        mask = torch.zeros_like(x_curr)
        if known_mel is not None:
             mask[:, :overlap_len, :] = 1.0
        
        s_idx = torch.full((B,), 3, device=device, dtype=torch.long) # Stage 4
        
        for i in range(num_steps):
            t_curr = times[i]
            t_next = times[i+1] if i < num_steps-1 else torch.tensor(0, device=device)
            t_curr_tensor = torch.full((B,), t_curr, device=device, dtype=torch.long)
            
            # --- In-painting Injection ---
            # Replace masked region with noisy version of known_mel
            if known_mel is not None:
                # noise at t_curr
                # known_mel is x0.
                noise = torch.randn_like(known_mel)
                known_noisy, _ = schedule.q_sample(known_mel, t_curr_tensor, noise)
                # Apply mask
                x_curr[:, :overlap_len, :] = known_noisy
            
            v_pred = student_decoder(x_curr, t_curr_tensor, sem_features=sem_features, step_idx=s_idx)
            
            x0_pred = schedule.predict_x0_from_v(x_curr, t_curr_tensor, v_pred)
            x0_pred = torch.clamp(x0_pred, -3, 3)
            eps = schedule.predict_eps_from_v(x_curr, t_curr_tensor, v_pred)
            
            alpha_next = schedule.alpha_bar[t_next]
            x_curr = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1-alpha_next) * eps
        
        # Final Force (at t=0)
        if known_mel is not None:
             x_curr[:, :overlap_len, :] = known_mel
             
        return x_curr

    # Null conditioning for CFG
    z_null = None  # Will be created per-chunk
    
    def inpaint_teacher_refine(x_coarse, sem_features, known_mel=None, overlap_len=0, strength=0.2, steps=10, cfg_scale=1.0):
        B = x_coarse.shape[0]
        t_start = int(cfg.diff_steps * strength)
        
        # Create null conditioning for CFG
        z_null_local = torch.zeros_like(sem_features)
        
        # Mask
        mask = torch.zeros_like(x_coarse)
        if known_mel is not None:
             mask[:, :overlap_len, :] = 1.0
        
        # Diffuse
        noise = torch.randn_like(x_coarse)
        t_start_tensor = torch.full((B,), t_start, device=device, dtype=torch.long)
        x_curr, _ = schedule.q_sample(x_coarse, t_start_tensor, noise)
        
        times = torch.linspace(t_start, 0, steps+1, device=device).long()
        times = times[:-1]
        s_idx = torch.full((B,), 0, device=device, dtype=torch.long) 
        
        for i in range(len(times)):
            t_curr = times[i]
            t_next = times[i+1] if i < len(times)-1 else torch.tensor(0, device=device)
            t_curr_tensor = torch.full((B,), t_curr, device=device, dtype=torch.long)
            
            # --- In-painting Injection ---
            if known_mel is not None:
                noise_k = torch.randn_like(known_mel)
                known_noisy, _ = schedule.q_sample(known_mel, t_curr_tensor, noise_k)
                x_curr[:, :overlap_len, :] = known_noisy
            
            # CFG: v = v_uncond + scale * (v_cond - v_uncond)
            v_cond = teacher_decoder(x_curr, t_curr_tensor, sem_features=sem_features, step_idx=s_idx)
            
            if cfg_scale != 1.0:
                v_uncond = teacher_decoder(x_curr, t_curr_tensor, sem_features=z_null_local, step_idx=s_idx)
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = v_cond
            
            x0_pred = schedule.predict_x0_from_v(x_curr, t_curr_tensor, v_pred)
            x0_pred = torch.clamp(x0_pred, -3, 3)
            eps = schedule.predict_eps_from_v(x_curr, t_curr_tensor, v_pred)
            
            alpha_next = schedule.alpha_bar[t_next]
            x_curr = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1-alpha_next) * eps
            
        if known_mel is not None:
             x_curr[:, :overlap_len, :] = known_mel
             
        return x_curr
    
    # --- Input Audio ---
    ljspeech_wavs = os.path.join(cfg.ljspeech_dir, "wavs")
    demo_file = "LJ001-0010.wav" 
    wav_path = os.path.join(ljspeech_wavs, demo_file)
    wav_np, orig_sr = sf.read(wav_path)
    wav = torch.from_numpy(wav_np).float()
    if wav.dim() == 1: wav = wav.unsqueeze(0)
    else: wav = wav.t()
    if orig_sr != cfg.sample_rate:
        wav = AF.resample(wav, orig_sr, cfg.sample_rate)
    
    wav = wav.to(device)
    total_samples = wav.shape[1]
    print(f"Loaded WAV: {wav.shape}, min={wav.min():.4f}, max={wav.max():.4f}, mean={wav.mean():.4f}")
    
    # Calculate full target for stats
    full_mel_target = mel_transform(wav)
    total_frames = full_mel_target.shape[2]
    
    # --- Context-Aware Sliding Window (Large Chunks + Overlap Crossfade) ---
    chunk_seconds = 2.0 # Match training segment_secs=2.0
    overlap_seconds = 0.5 # 25% overlap
    
    chunk_samples = int(chunk_seconds * cfg.sample_rate)
    overlap_samples = int(overlap_seconds * cfg.sample_rate)
    hop_samples = chunk_samples - overlap_samples
    
    num_chunks = int(np.ceil((total_samples - overlap_samples) / hop_samples))
    
    estimated_frames = total_frames + 1000
    final_mel = torch.zeros(cfg.n_mels, estimated_frames, device=device)
    final_weights = torch.zeros(1, estimated_frames, device=device)
    
    # Crossfade Window (Mel)
    chunk_ref = mel_transform(torch.zeros(1, chunk_samples, device=device))
    chunk_frames = chunk_ref.shape[2]
    
    overlap_ref = mel_transform(torch.zeros(1, overlap_samples, device=device))
    overlap_frames = overlap_ref.shape[2]
    
    hop_frames = chunk_frames - overlap_frames
    
    # Window function: Linear fade in/out at edges
    # For 50% overlap, we want a triangle window or trapezoid?
    # Triangle is good for 50%. Bartlett window.
    # But let's stick to trapezoid (flat center, fade edges) if hop < overlap?
    # Here hop = 1s, overlap = 1s.
    # So strictly: Fade In (1s) -> Fade Out (0s)? No.
    # Overlap region is 1s. Frame 0..overlap is Fade In. Frame chunk-overlap..chunk is Fade Out.
    # Since hop=overlap, the center "flat" region is 0 length?
    # chunk=2s. overlap=1s. hop=1s.
    # 0..1s: Fade In. 1..2s: Fade Out.
    # 2s total.
    # Peak at 1s.
    
    window_mask = torch.ones(1, chunk_frames, device=device)
    fade_len = overlap_frames
    
    fade_in = torch.linspace(0, 1, fade_len, device=device).unsqueeze(0)
    fade_out = torch.linspace(1, 0, fade_len, device=device).unsqueeze(0)
    
    window_mask[0, :fade_len] = fade_in
    window_mask[0, -fade_len:] = fade_out
    
    # To ensure sum=1 in overlap, we need careful window.
    # Bartlett (Triangle):
    # Overlap region (0 to 1s) of Chunk 2 overlaps (1 to 2s) of Chunk 1.
    # Chunk 1 (1..2s) is failing from 1.0 to 0.0.
    # Chunk 2 (0..1s) is rising from 0.0 to 1.0.
    # Sum = 1.0. Perfect.
    
    # --- Global Semantic Encoding ---
    print("Extracting Global Semantic Features...")
    # Resample to 16k for Hubert if needed
    if cfg.sample_rate != 16000:
        wav_16k = AF.resample(wav, cfg.sample_rate, 16000)
    else:
        wav_16k = wav
    
    # Pad to match latent alignment (320 samples)
    # Ensure divisible by 320
    if wav_16k.shape[1] % 320 != 0:
        pad_len = 320 - (wav_16k.shape[1] % 320)
        wav_16k = F.pad(wav_16k, (0, pad_len))
        
    with torch.no_grad():
        z_q_global = encoder(wav_16k)
    
    print(f"Global z_q shape: {z_q_global.shape}")
    
    # Compute global denormalization stats from full audio
    full_mel_log = torch.log(torch.clamp(full_mel_target, min=1e-5)).transpose(1, 2)  # [1, T, 80]
    _, global_mean, global_std = normalize_mel(full_mel_log)
    print(f"Global Mel stats: mean={global_mean.mean():.3f}, std={global_std.mean():.3f}")
    
    # Sliding Window
    print(f"50% Overlap Sliding: {num_chunks} chunks.")

    prev_mel_tail = None

    for i in tqdm.tqdm(range(num_chunks)):
        start_sample = i * hop_samples
        end_sample = start_sample + chunk_samples
        
        chunk_wav = wav[:, start_sample:end_sample]
        # Pad last chunk
        if chunk_wav.shape[1] < chunk_samples:
             chunk_wav = F.pad(chunk_wav, (0, chunk_samples - chunk_wav.shape[1]))
        
        # Check Mel length
        t_mel = mel_transform(chunk_wav)
        
        # Slicing Global z_q
        # We need to map start_sample (at cfg.sample_rate) to z_q index (at 16k / 320)
        # Factor = (16000 / cfg.sample_rate) / 320 ?
        # No, let's convert start_sample to time, then to 16k samples, then dive by 320
        
        start_sec = start_sample / cfg.sample_rate
        end_sec = end_sample / cfg.sample_rate
        
        start_idx_16k = int(start_sec * 16000)
        end_idx_16k = int(end_sec * 16000)
        
        start_lat = start_idx_16k // 320
        end_lat = end_idx_16k // 320
        
        # Current decoder expects z_q to match Mel length approx?
        # Decoder uses adaptive layer norm or cross attn?
        # If Cross-Attn, length doesn't strictly matter but should cover content.
        # Let's slice.
        
        z_q_chunk = z_q_global[:, start_lat:end_lat, :]
        
        # In-painting Generation
        with torch.no_grad():
            x_T = torch.randn(1, chunk_frames, cfg.n_mels, device=device)
            
            # Linear sampling with in-painting
            x_coarse = torch.randn(1, chunk_frames, cfg.n_mels, device=device)
            
            x_refined = inpaint_teacher_refine(
                 x_coarse, 
                 z_q_chunk, 
                 known_mel=prev_mel_tail, 
                 overlap_len=overlap_frames,
                 strength=refine_strength, 
                 steps=refine_steps,
                 cfg_scale=cfg_scale
            )
            
            # Update prev_mel_tail for NEXT chunk
            # We want the LAST overlap_frames of THIS generated chunk
            # to be the START (known_mel) of the NEXT chunk.
            prev_mel_tail = x_refined[:, -overlap_frames:, :].clone()
            
            # Apply Frequency Coherence Network to fix vertical artifacts
            with torch.no_grad():
                x_refined = fcn(x_refined)
            
            # Denorm using per-chunk GT stats (achieves 0.9+)
            mel_chunk_log = torch.log(torch.clamp(mel_transform(wav[:, start_sample:end_sample]), min=1e-5)).transpose(1, 2)
            _, real_mean, real_std = normalize_mel(mel_chunk_log)
            mel_denorm = denormalize_mel(x_refined, real_mean, real_std)
            lin_mel = torch.exp(mel_denorm).transpose(1, 2)
            output_chunk = lin_mel.squeeze(0)
            
            if output_chunk.shape[1] != chunk_frames:
                output_chunk = output_chunk[:, :chunk_frames]
                
            start_frame = i * hop_frames
            end_frame = start_frame + chunk_frames
            
            final_mel[:, start_frame:end_frame] += output_chunk * window_mask
            final_weights[:, start_frame:end_frame] += window_mask

    # Normalize Stitched Mel (Linear Domain)
    final_weights = torch.clamp(final_weights, min=1e-5)
    final_mel = final_mel / final_weights
    
    # Trim to original length
    final_mel = final_mel[:, :total_frames]
    
    print("Running Linear Scale & Griffin-Lim...")
    # final_mel is already Linear Mel [Mel, T]
    lin_mel = final_mel.unsqueeze(0).cpu()
    
    # 2D Smoothing to fix missing frequency coherence + temporal jitter
    # lin_mel shape: [1, 80, T]
    lin_mel_2d = lin_mel.unsqueeze(0)  # [1, 1, 80, T]
    
    # Apply 2D average pooling (frequency x time smoothing)
    kernel_h = 5  # Frequency smoothing (vertical)
    kernel_w = 3  # Time smoothing (horizontal)
    lin_mel_smoothed = F.avg_pool2d(
        lin_mel_2d, 
        kernel_size=(kernel_h, kernel_w), 
        stride=1, 
        padding=(kernel_h//2, kernel_w//2)
    ).squeeze(0)  # [1, 80, T]
    
    # Invert Mel Scale to get Linear Spectrogram
    lin_spec = inv_mel(lin_mel_smoothed)
    
    # More iterations for better phase
    griffin_hq = T.GriffinLim(n_fft=cfg.n_fft, n_iter=100, win_length=cfg.win_length, hop_length=cfg.hop_length).to("cpu")
    waveform = griffin_hq(lin_spec).squeeze()
    
    out_path = "final_50overlap.wav"
    sf.write(out_path, waveform.numpy(), cfg.sample_rate)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
