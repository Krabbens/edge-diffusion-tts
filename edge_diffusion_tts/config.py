"""
Configuration module for Edge Diffusion TTS.

Contains the main configuration dataclass and training phase enum.
"""

import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import torch


def get_device() -> str:
    """Detect the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    
    # Check for TPU (XLA)
    try:
        import torch_xla.core.xla_model as xm
        return "xla"
    except ImportError:
        pass
        
    return "cpu"


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainPhase(Enum):
    """Training phases for progressive distillation."""
    DIFFUSION = "diffusion"       # Standard DDPM training
    PROGRESSIVE = "progressive"   # Progressive distillation (halving steps)
    CONSISTENCY = "consistency"   # Consistency distillation (1-4 steps)


@dataclass
class CFG:
    """
    Main configuration for Edge Diffusion TTS.
    
    Attributes are organized into logical groups:
    - System: device, paths, seed
    - Data: dataset paths, audio parameters
    - Mel spectrogram: FFT settings
    - HuBERT + VQ: semantic encoding
    - Model: architecture hyperparameters
    - Diffusion: schedule parameters
    - Progressive distillation: step reduction
    - Training: optimization settings
    - Logging: checkpointing and visualization
    """
    
    # ===== SYSTEM =====
    seed: int = 42
    device: str = field(default_factory=get_device)
    out_dir: str = "run_edge_diffusion"
    run_name: str = field(default_factory=lambda: time.strftime("run_%Y%m%d_%H%M%S"))
    
    # ===== DATA =====
    data_root: str = "./data"
    ljspeech_dir: str = "./data/LJSpeech-1.1"
    sample_rate: int = 16000
    orig_sr: int = 22050
    segment_secs: float = 2.0  # Shorter for M1 memory
    segment_len: int = 32000
    segment_len: int = 32000
    num_workers: int = 0  # CRITICAL for MPS: avoids "BrokenPipe" and IPC deadlocks on macOS
    pin_memory: bool = False # Usually False for MPS to avoid unnecessary CPU syncs
    
    # ===== MEL SPECTROGRAM =====
    n_fft: int = 1024
    hop_length: int = 160
    win_length: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0
    
    # ===== HUBERT + VQ/FSQ =====
    hubert_id: str = "facebook/hubert-base-ls960"
    hubert_layer: int = 9
    semantic_dim: int = 128
    codebook_size: int = 512
    vq_commit: float = 1.0
    use_fsq: bool = True  # Use FSQ instead of VQ (simpler, no collapse)
    fsq_levels: list = field(default_factory=lambda: [4, 4, 3, 3, 2, 2, 2, 2])  # 2304 codes, 8D latent space (better resolution)
    
    # ===== EDGE-OPTIMIZED MODEL =====
    hidden: int = 160           # Edge-optimized (~2.5MB decoder)
    layers: int = 4             # Edge target: 4 layers
    heads: int = 4              # Edge target: 4 heads
    ffn_mult: int = 2
    use_depthwise: bool = True  # Depthwise separable conv
    use_flash_attn: bool = True # Memory-efficient attention
    use_adaln: bool = True      # Adaptive LayerNorm for timestep
    dropout: float = 0.2        # Increased for regularization (was 0.1)
    attn_window_size: int = 64  # Smaller window (was 256, too large)
    
    # ===== DIFFUSION SCHEDULE =====
    diff_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    use_v_prediction: bool = True  # v-prediction is more stable than ε
    max_timestep: int = 950  # Avoid numerical explosion at t>950
    
    # ===== TRAINING PHASE =====
    phase: TrainPhase = TrainPhase.DIFFUSION
    
    # Phase 1 (Diffusion)
    diffusion_epochs: int = 50  # Long training session
    
    # Phase 2 (Progressive) - target step count
    progressive_epochs_per_halving: int = 5
    progressive_target_steps: int = 4  # Reduce to 4 steps
    
    # Phase 3 (Consistency)
    consistency_epochs: int = 10
    consistency_weight: float = 1.0
    
    # ===== TRAINING =====
    batch_size: int = 4  # Small physical batch for MPS memory
    grad_accumulation: int = 8    # Effective batch = 32
    lr: float = 2e-4
    lr_consistency: float = 1e-4  # Lower for fine-tuning
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # grad_accumulation: int = 2  # Effective batch = 8 (consolidated above)
    
    # ===== LOGGING / EVAL =====
    log_every_steps: int = 50
    val_every_steps: int = 200
    plot_every_steps: int = 100
    val_batches: int = 4
    
    # ===== INFERENCE =====
    inference_steps: int = 4  # Target: 4 steps on edge!
    
    # ===== CHECKPOINT =====
    ckpt_path: str = ""
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Calculate segment length from seconds
        self.segment_len = int(self.sample_rate * self.segment_secs)
        
        # Align to LCM for model compatibility
        lcm = 320
        self.segment_len = (self.segment_len // lcm) * lcm
        
        # Create directories
        os.makedirs(self.data_root, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Set checkpoint path
        if not self.ckpt_path:
            self.ckpt_path = os.path.join(self.out_dir, "checkpoint_latest.pt")
    
    def setup_environment(self) -> None:
        """Setup reproducibility and performance settings."""
        set_seed(self.seed)
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    
    def print_config(self) -> None:
        """Print configuration summary."""
        print(f"{'='*60}")
        print(f"   EDGE-OPTIMIZED DIFFUSION TTS")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Segment: {self.segment_len} samples ({self.segment_len/self.sample_rate:.2f}s)")
        print(f"Model hidden: {self.hidden} (edge-optimized)")
        print(f"Target inference steps: {self.inference_steps}")
        print(f"{'='*60}\n")
    
    def get_run_dir(self) -> str:
        """Get the run directory path."""
        return os.path.join(self.out_dir, self.run_name)
    
    @classmethod
    def from_dict(cls, d: dict) -> "CFG":
        """Create config from dictionary."""
        # Handle TrainPhase enum
        if "phase" in d and isinstance(d["phase"], str):
            d["phase"] = TrainPhase(d["phase"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        d = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, TrainPhase):
                v = v.value
            d[k] = v
        return d
