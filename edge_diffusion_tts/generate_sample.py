
import os
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import soundfile as sf
# Force soundfile backend
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

from edge_diffusion_tts.config import CFG
from edge_diffusion_tts.models import SemanticEncoder, EdgeDiffusionDecoder
from edge_diffusion_tts.inference import EdgeInference
from edge_diffusion_tts.schedule import DiffusionSchedule
from edge_diffusion_tts.utils.audio import normalize_mel, denormalize_mel

def main():
    device = "cpu"  # Force CPU to avoid any MPS/backend weirdness during debug
    print(f"Using device: {device}")
    
    # Paths
    run_dir = "run_edge_diffusion/run_20260104_174357" # From user metadata
    ckpt_path = os.path.join(run_dir, "edge_model_final.pt")
    
    if not os.path.exists(ckpt_path):
        # Fallback to verify logic
        print(f"Checkpoint not found at {ckpt_path}, searching...")
        # Just use the one user mentioned if possible, but let's try to find it
        # Actually, let's just use the 'run_edge_diffusion' dir which is in the metadata
        dirs = sorted([d for d in os.listdir("run_edge_diffusion") if d.startswith("run_")])
        if dirs:
            latest = dirs[-1] # Assuming the latest run is the one
            ckpt_path = os.path.join("run_edge_diffusion", latest, "edge_model_final.pt")
            print(f"Found checkpoint: {ckpt_path}")
    
    # Load Config and Model
    print("Loading model...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg_dict = checkpoint["cfg"]
    cfg = CFG.from_dict(cfg_dict)
    cfg.device = device # Override device
    
    encoder = SemanticEncoder(cfg).to(device)
    encoder.proj.load_state_dict(checkpoint["encoder_proj"])
    encoder.vq.load_state_dict(checkpoint["encoder_vq"])
    
    decoder = EdgeDiffusionDecoder(cfg).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    
    schedule = DiffusionSchedule(cfg.diff_steps, cfg.beta_start, cfg.beta_end, device)
    inference = EdgeInference(cfg, schedule, encoder, decoder)
    
    # Get a sample audio
    ljspeech_wavs = os.path.join(cfg.ljspeech_dir, "wavs")
    if not os.path.exists(ljspeech_wavs):
        print(f"LJSpeech wavs not found at {ljspeech_wavs}")
        return

    # Pick a random file or specific one
    demo_file = sorted(os.listdir(ljspeech_wavs))[0] # LJ001-0001.wav usually
    wav_path = os.path.join(ljspeech_wavs, demo_file)
    print(f"Processing: {wav_path}")
    
    # Load with soundfile
    wav_np, orig_sr = sf.read(wav_path)
    wav = torch.from_numpy(wav_np).float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0) # [1, T] for mono
    else:
        wav = wav.t() # [C, T]
        
    # Resample if needed using torchaudio functional (hopefully works)
    if orig_sr != cfg.sample_rate:
        try:
            import torchaudio.functional as AF
            wav = AF.resample(wav, orig_sr, cfg.sample_rate)
        except Exception as e:
            print(f"Resampling failed: {e}")
            # Naive resampling or error out
            return
    
    # Trim to 5 seconds max
    max_len = cfg.sample_rate * 5
    if wav.shape[1] > max_len:
        wav = wav[:, :max_len]
        
    wav = wav.to(device)
    
    # 1. Reconstruction (4 steps)
    print("Generating (4 steps)...")
    mel_gen = inference.generate_from_audio(wav, num_steps=4)
    
    # Calculate stats from input audio for denormalization
    mel_transform = T.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        n_mels=cfg.n_mels,
        power=2.0,
        normalized=False
    ).to(device)
    
    mel_orig = mel_transform(wav)
    mel_orig = torch.log(torch.clamp(mel_orig, min=1e-5))
    # Transpose to [B, T, n_mels] for normalization stats matches training
    mel_orig = mel_orig.transpose(1, 2)
    _, mean, std = normalize_mel(mel_orig)
    
    # Denormalize generated
    mel_out = denormalize_mel(mel_gen, mean, std)
    
    # Convert back from log scale
    lin_mel = torch.exp(mel_out)
    
    # Transpose to [B, n_mels, T] for transform
    lin_mel = lin_mel.transpose(1, 2)
    
    # Inverse Mel Scale: Mel -> Linear Spectrogram
    inverse_mel = T.InverseMelScale(
        n_stft=cfg.n_fft // 2 + 1,
        n_mels=cfg.n_mels,
        sample_rate=cfg.sample_rate,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        norm=None, # Match collate.py which uses default (None for T.MelSpectrogram)
    ).to(device)
    
    # Griffin-Lim: Linear Spectrogram -> Waveform
    griffin_lim = T.GriffinLim(
        n_fft=cfg.n_fft,
        n_iter=32,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        power=2.0, 
    ).to(device)
    
    try:
        linear_spec = inverse_mel(lin_mel)
        wav_gen = griffin_lim(linear_spec)
    except Exception as e:
        print(f"Vocoding failed: {e}")
        return

    except Exception as e:
        print(f"Vocoding failed: {e}")
        return

    # To numpy
    wav_out_np = wav_gen.cpu().squeeze().numpy()
    
    # Save raw output
    out_path_raw = "generated_sample_4steps_raw.wav"
    sf.write(out_path_raw, wav_out_np, cfg.sample_rate)
    print(f"Saved {out_path_raw}")

    # Apply Noise Reduction
    print("Applying noise reduction...")
    try:
        import noisereduce as nr
        # Assume noise is stationary, use statistics from the audio itself or a silence patch if available
        # Since we don't have a noise profile, we use stationary noise reduction
        wav_denoised = nr.reduce_noise(y=wav_out_np, sr=cfg.sample_rate, prop_decrease=0.75, stationary=True)
        
        out_path_denoised = "generated_sample_4steps_denoised.wav"
        sf.write(out_path_denoised, wav_denoised, cfg.sample_rate)
        print(f"Saved {out_path_denoised}")
    except Exception as e:
        print(f"Noise reduction failed: {e}")

    # Also save original for comparison (resampled)
    sf.write("original_sample.wav", wav.cpu().squeeze().numpy(), cfg.sample_rate)
    print(f"Saved original_sample.wav")

    # --- ORACLE RECONSTRUCTION DEBUG ---
    print("Generating Oracle reconstruction (Wav -> Mel -> Wav)...")
    try:
        # Mel from original
        mel_oracle = mel_transform(wav) # [1, n_mels, T]
        mel_oracle = torch.log(torch.clamp(mel_oracle, min=1e-5))
        
        # Invert
        lin_oracle = torch.exp(mel_oracle)
        # linear expects [B, n_mels, T], which it is.
        
        linear_spec_oracle = inverse_mel(lin_oracle)
        wav_oracle = griffin_lim(linear_spec_oracle)
        
        wav_oracle_np = wav_oracle.cpu().squeeze().numpy()
        sf.write("oracle_sample.wav", wav_oracle_np, cfg.sample_rate)
        print(f"Saved oracle_sample.wav (If this sounds bad, the vocoder parameters are wrong)")
        
        # Print duration stats
        dur_orig = wav.shape[1] / cfg.sample_rate
        dur_gen = wav_gen.shape[1] / cfg.sample_rate
        print(f"Duration - Original: {dur_orig:.2f}s, Generated: {dur_gen:.2f}s")
        
    except Exception as e:
        print(f"Oracle reconstruction failed: {e}")

if __name__ == "__main__":
    main()
