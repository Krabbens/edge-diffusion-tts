#!/usr/bin/env python3
"""
Train Edge Diffusion TTS model.

Usage:
    uv run train.py                     # Train with defaults
    uv run train.py --device cuda       # Force CUDA
    uv run train.py --batch-size 8      # Override batch size
    uv run train.py --resume ckpt.pt    # Resume from checkpoint
    uv run train.py --export            # Export ONNX after training
"""

import sys
sys.path.insert(0, ".")

from edge_diffusion_tts.train import train

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Edge Diffusion TTS")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--export", action="store_true", help="Export ONNX")
    parser.add_argument("--device", type=str, help="Device override")
    parser.add_argument("--batch-size", type=int, help="Batch size override")
    parser.add_argument("--epochs", type=int, help="Epochs override")
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        resume_path=args.resume,
        export_onnx=args.export,
        device_override=args.device,
        batch_size_override=args.batch_size,
        epochs_override=args.epochs,
    )
