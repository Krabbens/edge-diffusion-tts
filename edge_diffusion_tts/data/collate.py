
"""
Collate function for batching audio data.
"""

import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as T

from ..config import CFG


class Collate:
    """Collate function for batching audio with mel spectrogram extraction."""
    
    def __init__(self, cfg: CFG):
        self.cfg = cfg
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            n_mels=cfg.n_mels,
            power=2.0,
            normalized=False,
        )
    
    def _resample(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.cfg.sample_rate:
            return wav
        return AF.resample(wav, orig_freq=sr, new_freq=self.cfg.sample_rate)
    
    def _crop_pad(self, wav: torch.Tensor) -> torch.Tensor:
        L = wav.numel()
        tgt = self.cfg.segment_len
        if L >= tgt:
            start = random.randint(0, L - tgt)
            return wav[start:start + tgt]
        else:
            return F.pad(wav, (0, tgt - L))
    
    def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        wavs, fids = [], []
        for wav, sr, fid in batch:
            wav = self._resample(wav, sr)
            wav = self._crop_pad(wav)
            wav = torch.clamp(wav, -1.0, 1.0)
            wavs.append(wav)
            fids.append(fid)
        
        wav = torch.stack(wavs, dim=0)
        mel = self.mel(wav)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.transpose(1, 2).contiguous()
        return wav, mel, fids
