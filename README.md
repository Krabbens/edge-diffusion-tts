# Edge Diffusion TTS

**Edge-optimized Diffusion Text-to-Speech with Progressive Distillation**

A lightweight diffusion-based TTS system optimized for edge device inference with **1-4 step generation**.

## Features

- 🚀 **Few-step inference**: Generate speech in 1-4 denoising steps (vs 1000 standard)
- 📱 **Edge-optimized**: Depthwise separable convolutions, efficient attention
- 🎯 **Progressive distillation**: Systematic step reduction 1000→4
- 🔊 **HuBERT semantic encoding**: High-quality semantic representation
- ⚡ **Flash Attention**: Memory-efficient when available (PyTorch 2.0+)
- 🍎 **M1/M2 support**: Runs on Apple Silicon with MPS backend

## Project Structure

```
edge_diffusion_tts/
├── __init__.py           # Package exports
├── config.py             # Configuration dataclass
├── schedule.py           # Diffusion schedules (DDPM/DDIM)
├── inference.py          # Few-step inference engine
├── train.py              # Main training loop
├── cli.py                # CLI entry point
│
├── layers/               # Neural network layers
│   ├── conv.py          # Depthwise separable convolutions
│   ├── attention.py     # Efficient attention
│   ├── transformer.py   # Transformer blocks
│   └── embeddings.py    # Time/position embeddings
│
├── models/               # Main models
│   ├── vq.py            # Vector quantizer
│   ├── encoder.py       # HuBERT semantic encoder
│   └── decoder.py       # Edge diffusion decoder
│
├── training/             # Training components
│   └── consistency.py   # Progressive/consistency distillation
│
├── data/                 # Data loading
│   ├── dataset.py       # LJSpeech dataset
│   └── collate.py       # Batch collation
│
└── utils/                # Utilities
    ├── audio.py         # Mel normalization
    ├── visualization.py # Plotting & evaluation
    └── export.py        # ONNX export
```

## Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Training

```bash
# Train with default settings
uv run train.py

# With custom settings
uv run train.py --device cuda --batch-size 8 --epochs 50

# Resume from checkpoint
uv run train.py --resume run_edge_diffusion/run_xxx/checkpoint_phase1.pt

# Export ONNX after training
uv run train.py --export
```

### Using the Package

```python
from edge_diffusion_tts import CFG, DiffusionSchedule, SemanticEncoder, EdgeDiffusionDecoder, EdgeInference

# Initialize
cfg = CFG()
schedule = DiffusionSchedule(cfg.diff_steps, device=cfg.device)
encoder = SemanticEncoder(cfg).to(cfg.device)
decoder = EdgeDiffusionDecoder(cfg).to(cfg.device)
inference = EdgeInference(cfg, schedule, encoder, decoder)

# Load trained weights
checkpoint = torch.load("edge_model_final.pt")
encoder.proj.load_state_dict(checkpoint["encoder_proj"])
encoder.vq.load_state_dict(checkpoint["encoder_vq"])
decoder.load_state_dict(checkpoint["decoder"])

# Generate in 4 steps!
mel = inference.generate_from_audio(waveform, num_steps=4)
```

## Training Phases

The training uses 3-phase progressive distillation:

### Phase 1: Standard Diffusion (30 epochs)
Train a full 1000-step diffusion model.

### Phase 2: Progressive Distillation (5 epochs per halving)
Progressively halve steps: 1000 → 500 → 250 → 125 → 64 → 32 → 16 → 8 → 4

### Phase 3: Consistency Distillation (10 epochs)
Fine-tune for 1-4 step generation with consistency loss.

## Configuration

Key hyperparameters in `CFG`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden` | 160 | Model hidden dimension (edge-optimized) |
| `layers` | 4 | Number of transformer layers |
| `heads` | 4 | Attention heads |
| `diff_steps` | 1000 | Total diffusion steps |
| `inference_steps` | 4 | Target inference steps |
| `batch_size` | 4 | Training batch size |
| `use_depthwise` | True | Use depthwise separable convs |

## Model Size

- **Decoder**: ~2.5 MB (FP32)
- **Semantic encoder (trainable)**: ~150KB
- **Total inference model**: ~3 MB

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchaudio
- transformers (for HuBERT)
- tensorboard, tqdm, matplotlib

## License

MIT
