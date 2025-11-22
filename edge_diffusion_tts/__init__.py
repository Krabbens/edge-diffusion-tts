"""
Edge-Optimized Diffusion TTS

A lightweight diffusion-based text-to-speech system optimized for edge devices
with progressive distillation for few-step (1-4) inference.

Key Features:
- Depthwise separable convolutions for reduced parameters
- Memory-efficient attention with Flash Attention support
- Progressive distillation: 1000 → 4 steps
- Consistency training for single-step generation
- HuBERT-based semantic encoding with VQ
"""

__version__ = "0.1.0"

from .config import CFG, TrainPhase, get_device, set_seed
from .schedule import DiffusionSchedule
from .models import SemanticEncoder, EdgeDiffusionDecoder, VectorQuantizer
from .inference import EdgeInference
from .training import ConsistencyTrainer

__all__ = [
    "CFG",
    "TrainPhase",
    "get_device",
    "set_seed",
    "DiffusionSchedule",
    "SemanticEncoder",
    "EdgeDiffusionDecoder",
    "VectorQuantizer",
    "EdgeInference",
    "ConsistencyTrainer",
]
