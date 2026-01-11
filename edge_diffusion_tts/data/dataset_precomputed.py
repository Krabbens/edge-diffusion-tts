"""
LJSpeech Dataset with pre-computed HuBERT features.

Uses pre-extracted features for ~10-20x faster training.
"""

import os
from typing import Tuple, Optional

import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset
import torch.nn.functional as F


class LJSpeechPrecomputedDataset(Dataset):
    """LJSpeech dataset using pre-computed HuBERT features."""
    
    def __init__(
        self, 
        root: str, 
        split: str = "train", 
        val_ratio: float = 0.05,
        max_samples: int = None,
        cfg = None
    ):
        self.root = root
        self.cfg = cfg
        self.features_dir = os.path.join(root, "hubert_features")
        self.wavs_dir = os.path.join(root, "wavs")
        
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(
                f"Pre-computed features not found at {self.features_dir}. "
                "Run `python precompute_hubert.py` first!"
            )
        
        meta = os.path.join(root, "metadata.csv")
        with open(meta, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        
        ids = [ln.split("|")[0] for ln in lines if "|" in ln]
        
        # Filter to only files with pre-computed features
        available_features = set(f.replace(".pt", "") for f in os.listdir(self.features_dir))
        ids = [fid for fid in ids if fid in available_features]
        
        rng = np.random.RandomState(1234)
        perm = rng.permutation(len(ids))
        n_val = int(len(ids) * val_ratio)
        val_idx = set(perm[:n_val].tolist())
        
        if split == "train":
            self.ids = [ids[i] for i in range(len(ids)) if i not in val_idx]
        else:
            self.ids = [ids[i] for i in range(len(ids)) if i in val_idx]
        
        if max_samples and len(self.ids) > max_samples:
            rng2 = np.random.RandomState(42)
            indices = rng2.choice(len(self.ids), max_samples, replace=False)
            self.ids = [self.ids[i] for i in sorted(indices)]
        
        print(f"  {split} set: {len(self.ids)} samples (pre-computed features)")
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            hubert_features: Pre-computed HuBERT features [T_feat, 768]
            mel: Mel spectrogram (computed on-the-fly, fast) [T_mel, n_mels]
            fid: File ID
        """
        fid = self.ids[i]
        
        # Load pre-computed HuBERT features (fast: just disk read)
        feat_path = os.path.join(self.features_dir, f"{fid}.pt")
        hubert_feat = torch.load(feat_path, map_location="cpu", weights_only=True)
        
        # Load wav for mel computation (still needed)
        wav_path = os.path.join(self.wavs_dir, f"{fid}.wav")
        wav_np, sr = sf.read(wav_path)
        wav = torch.from_numpy(wav_np).float()
        
        if wav.dim() == 2:
            wav = wav.mean(dim=1)
        
        return hubert_feat, wav, sr, fid


class CollatePrecomputed:
    """Collate function for pre-computed features dataset."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        # Mel spectrogram transform
        from torchaudio.transforms import MelSpectrogram, Resample
        self.mel_tfm = MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
        )
    
    def __call__(self, batch):
        cfg = self.cfg
        
        feats = []
        mels = []
        fids = []
        
        for hubert_feat, wav, sr, fid in batch:
            # Resample if needed
            if sr != cfg.sample_rate:
                ratio = cfg.sample_rate / sr
                new_len = int(len(wav) * ratio)
                wav = F.interpolate(
                    wav.unsqueeze(0).unsqueeze(0),
                    size=new_len,
                    mode='linear',
                    align_corners=False
                ).squeeze()
            
            # Random crop for training
            target_len = cfg.segment_len
            if len(wav) > target_len:
                start = torch.randint(0, len(wav) - target_len, (1,)).item()
                wav = wav[start:start + target_len]
                
                # Corresponding HuBERT crop
                # HuBERT downsamples by 320x (20ms frames at 16kHz)
                feat_per_sample = 1 / 320
                feat_start = int(start * feat_per_sample)
                feat_len = int(target_len * feat_per_sample)
                hubert_feat = hubert_feat[feat_start:feat_start + feat_len]
            else:
                # Pad if too short
                wav = F.pad(wav, (0, target_len - len(wav)))
            
            # Compute mel (fast operation, ~1ms)
            mel = self.mel_tfm(wav)  # [n_mels, T]
            mel = mel.transpose(0, 1)  # [T, n_mels]
            
            feats.append(hubert_feat)
            mels.append(mel)
            fids.append(fid)
        
        # Pad features to same length
        max_feat_len = max(f.shape[0] for f in feats)
        max_mel_len = max(m.shape[0] for m in mels)
        
        feats_padded = torch.zeros(len(batch), max_feat_len, feats[0].shape[-1])
        mels_padded = torch.zeros(len(batch), max_mel_len, self.cfg.n_mels)
        
        for i, (f, m) in enumerate(zip(feats, mels)):
            feats_padded[i, :f.shape[0]] = f
            mels_padded[i, :m.shape[0]] = m
        
        return feats_padded, mels_padded, fids
