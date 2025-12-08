"""
Audio processing utilities.
"""

from typing import Tuple

import torch


def normalize_mel(mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize mel-spectrogram per-batch."""
    mean = mel.mean(dim=1, keepdim=True)
    std = mel.std(dim=1, keepdim=True).clamp_min(1e-5)
    return (mel - mean) / std, mean, std


def denormalize_mel(mel_n: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Denormalize mel-spectrogram."""
    return mel_n * std + mean
