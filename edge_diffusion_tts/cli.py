"""
CLI entry point for Edge Diffusion TTS.
"""

import argparse


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Edge Diffusion TTS Training")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--export", action="store_true", help="Export to ONNX after training")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override diffusion epochs")
    
    args = parser.parse_args()
    
    # Import here to delay heavy imports
    from .train import train
    
    train(
        config_path=args.config,
        resume_path=args.resume,
        export_onnx=args.export,
        device_override=args.device,
        batch_size_override=args.batch_size,
        epochs_override=args.epochs,
    )


if __name__ == "__main__":
    main()
