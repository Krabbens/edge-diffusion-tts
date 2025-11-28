"""
Model components for Edge Diffusion TTS.

Main models:
- VectorQuantizer: VQ for semantic token discretization
- FSQ, FSQEncoder: Finite Scalar Quantization (simpler, no collapse)
- SemanticEncoder: HuBERT-based semantic encoding with VQ/FSQ
- EdgeDiffusionDecoder: Optimized diffusion decoder for edge devices
"""

from .vq import VectorQuantizer
from .fsq import FSQ, FSQEncoder
from .encoder import SemanticEncoder
from .decoder import EdgeDiffusionDecoder

__all__ = [
    "VectorQuantizer",
    "FSQ",
    "FSQEncoder",
    "SemanticEncoder",
    "EdgeDiffusionDecoder",
]

