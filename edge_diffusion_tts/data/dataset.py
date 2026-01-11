"""
LJSpeech Dataset module.
"""

import os
from typing import Tuple

import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset


def ensure_ljspeech(data_root: str) -> str:
    """Check LJSpeech is present (manual download required)."""
    ljspeech_dir = os.path.join(data_root, "LJSpeech-1.1")
    wavs_dir = os.path.join(ljspeech_dir, "wavs")
    meta_path = os.path.join(ljspeech_dir, "metadata.csv")
    
    if os.path.exists(wavs_dir) and os.path.exists(meta_path):
        print("✓ LJSpeech already present.")
        return ljspeech_dir
    
    raise FileNotFoundError(
        f"LJSpeech not found at {ljspeech_dir}. "
        "Please download from https://keithito.com/LJ-Speech-Dataset/ "
        "and extract to ./data/LJSpeech-1.1"
    )


class LJSpeechDataset(Dataset):
    """LJSpeech dataset wrapper with train/val split."""
    
    def __init__(self, root: str, split: str = "train", val_ratio: float = 0.05, max_samples: int = None):
        self.root = root
        meta = os.path.join(root, "metadata.csv")
        wavs = os.path.join(root, "wavs")
        
        with open(meta, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        
        ids = [ln.split("|")[0] for ln in lines if "|" in ln]
        
        rng = np.random.RandomState(1234)
        perm = rng.permutation(len(ids))
        n_val = int(len(ids) * val_ratio)
        val_idx = set(perm[:n_val].tolist())
        
        if split == "train":
            self.ids = [ids[i] for i in range(len(ids)) if i not in val_idx]
        else:
            self.ids = [ids[i] for i in range(len(ids)) if i in val_idx]
        
        # Limit dataset size for faster iteration
        if max_samples and len(self.ids) > max_samples:
            rng2 = np.random.RandomState(42)
            indices = rng2.choice(len(self.ids), max_samples, replace=False)
            self.ids = [self.ids[i] for i in sorted(indices)]
        
        self.wav_dir = wavs
        print(f"  {split} set: {len(self.ids)} samples")
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, str]:
        fid = self.ids[i]
        path = os.path.join(self.wav_dir, f"{fid}.wav")
        
        # Use soundfile instead of torchaudio (avoids FFmpeg dependency)
        wav_np, sr = sf.read(path)
        wav = torch.from_numpy(wav_np).float()
        
        # Handle stereo
        if wav.dim() == 2:
            wav = wav.mean(dim=1)  # [T, C] -> [T]
        
        return wav, sr, fid

