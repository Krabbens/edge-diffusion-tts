"""
Export utilities for edge deployment.
"""

import os

import torch

from ..config import CFG


def export_for_edge(decoder, cfg: CFG, output_path: str = "edge_decoder.onnx"):
    """Export decoder to ONNX for edge deployment."""
    print(f"\nExporting to ONNX: {output_path}")
    
    decoder.eval()
    decoder.cpu()
    
    B, T, T_sem = 1, 200, 100
    x_t = torch.randn(B, T, cfg.n_mels)
    t = torch.tensor([500], dtype=torch.long)
    sem_idx = torch.randint(0, cfg.codebook_size, (B, T_sem))
    step_idx = torch.tensor([0], dtype=torch.long)
    
    torch.onnx.export(
        decoder,
        (x_t, t, sem_idx, step_idx),
        output_path,
        input_names=["x_t", "t", "sem_idx", "step_idx"],
        output_names=["eps_pred"],
        dynamic_axes={
            "x_t": {0: "batch", 1: "time"},
            "sem_idx": {0: "batch", 1: "sem_time"},
            "eps_pred": {0: "batch", 1: "time"}
        },
        opset_version=14
    )
    
    print(f"✓ Exported: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
