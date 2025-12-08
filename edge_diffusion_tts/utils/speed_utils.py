"""
Speed optimization utilities for Edge Diffusion TTS.

Includes:
- Gradient checkpointing
- Memory-efficient training helpers
- Benchmark utilities
"""

import gc
import time
from functools import wraps
from typing import Callable, Optional

import torch
import torch.nn as nn


def enable_gradient_checkpointing(model: nn.Module, checkpoint_layers: bool = True) -> None:
    """
    Enable gradient checkpointing on transformer layers.
    
    Reduces memory by ~40% at cost of ~20% slower backward pass.
    Net effect: can use 40% larger batch size = faster training.
    
    Args:
        model: Model to enable checkpointing on
        checkpoint_layers: Whether to checkpoint transformer layers
    """
    if hasattr(model, 'layers') and checkpoint_layers:
        for layer in model.layers:
            layer.use_checkpoint = True
            # Wrap forward to use checkpointing
            if not hasattr(layer, '_original_forward'):
                layer._original_forward = layer.forward
                
                def make_checkpointed_forward(layer):
                    def checkpointed_forward(x, context, cond=None):
                        if layer.training:
                            return torch.utils.checkpoint.checkpoint(
                                layer._original_forward,
                                x, context, cond,
                                use_reentrant=False
                            )
                        return layer._original_forward(x, context, cond)
                    return checkpointed_forward
                
                layer.forward = make_checkpointed_forward(layer)


def setup_memory_efficient_training(cfg, model: nn.Module) -> dict:
    """
    Setup memory-efficient training configurations.
    
    Returns:
        dict with scaler and autocast context
    """
    settings = {
        'use_amp': False,
        'scaler': None,
        'autocast_dtype': torch.float32,
        'autocast_context': None,
    }
    
    device_type = cfg.device.split(':')[0] if ':' in cfg.device else cfg.device
    
    if device_type == 'cuda':
        settings['use_amp'] = True
        settings['scaler'] = torch.amp.GradScaler('cuda', enabled=True)
        settings['autocast_dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device_type == 'mps':
        # MPS supports float16 but not full AMP
        settings['autocast_dtype'] = torch.float16
    
    # Enable memory-efficient attention patterns
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Set optimal memory allocator settings
    if torch.cuda.is_available():
        # Reduce fragmentation
        torch.cuda.set_per_process_memory_fraction(0.95)
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.memory.set_per_process_memory_fraction(0.95)
    
    return settings


def get_fused_optimizer(params, lr: float, weight_decay: float = 0.01, betas=(0.9, 0.999)):
    """
    Get the fastest optimizer available.
    
    Uses fused AdamW on CUDA, regular AdamW elsewhere.
    Fused optimizer is ~10-20% faster.
    """
    if torch.cuda.is_available():
        try:
            # Fused AdamW is significantly faster on CUDA
            return torch.optim.AdamW(
                params, 
                lr=lr, 
                weight_decay=weight_decay,
                betas=betas,
                fused=True  # CUDA fused implementation
            )
        except (TypeError, RuntimeError):
            pass
    
    # Fallback to regular AdamW
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def compile_model(model: nn.Module, mode: str = "reduce-overhead", device: str = "cuda") -> nn.Module:
    """
    Compile model with torch.compile for faster training.
    
    Args:
        model: Model to compile
        mode: Compilation mode
            - "reduce-overhead": Fastest for small batches (default)
            - "max-autotune": Best for large batches, slower compile time
            - "default": Balanced
        device: Target device
        
    Returns:
        Compiled model (or original if compilation not supported)
    """
    if not hasattr(torch, 'compile'):
        return model
    
    if device == "mps":
        # MPS has limited torch.compile support
        return model
    
    try:
        # Use inductor backend for best performance
        compiled = torch.compile(
            model, 
            mode=mode,
            backend="inductor",
            fullgraph=False,  # Allow graph breaks for flexibility
        )
        return compiled
    except Exception as e:
        print(f"  ⚠ torch.compile failed: {e}")
        return model


def memory_cleanup():
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()


class TimingContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, sync_cuda: bool = True):
        self.name = name
        self.sync_cuda = sync_cuda
        self.start = 0
        self.duration = 0
    
    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.duration = time.perf_counter() - self.start
    
    def __str__(self):
        return f"{self.name}: {self.duration*1000:.2f}ms"


def benchmark_model(model: nn.Module, sample_input: tuple, warmup: int = 5, runs: int = 20) -> dict:
    """
    Benchmark model inference speed.
    
    Args:
        model: Model to benchmark
        sample_input: Tuple of sample inputs
        warmup: Number of warmup runs
        runs: Number of timed runs
        
    Returns:
        dict with timing statistics
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*sample_input)
    
    # Sync
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(*sample_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    import statistics
    return {
        'mean_ms': statistics.mean(times) * 1000,
        'std_ms': statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'throughput': 1 / statistics.mean(times),
    }


class GradientAccumulator:
    """
    Efficient gradient accumulation helper.
    
    Handles scaling and optimizer steps correctly with mixed precision.
    """
    
    def __init__(self, optimizer, scaler=None, accumulation_steps: int = 1, max_grad_norm: float = 1.0):
        self.optimizer = optimizer
        self.scaler = scaler
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0
    
    def backward(self, loss: torch.Tensor, parameters=None):
        """
        Scale and backward loss.
        
        Args:
            loss: Loss tensor
            parameters: Parameters for gradient clipping (optional)
        """
        scaled_loss = loss / self.accumulation_steps
        
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        self.step_count += 1
    
    def step(self, parameters) -> bool:
        """
        Perform optimizer step if accumulation is complete.
        
        Args:
            parameters: Parameters for gradient clipping
            
        Returns:
            True if optimizer step was performed
        """
        if self.step_count < self.accumulation_steps:
            return False
        
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
        self.step_count = 0
        return True
    
    def zero_grad(self):
        """Zero gradients with memory efficiency."""
        self.optimizer.zero_grad(set_to_none=True)
        self.step_count = 0
