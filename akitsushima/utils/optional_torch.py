"""Optional PyTorch detection, isolated in its own module.

Kept separate from optional_matplotlib.py so that a caller who only needs
one of these optional, heavyweight (and, for torch, OpenMP-based -- see
fitting/models/__init__.py's docstring for a concrete instability this
isolation avoids) dependencies never eagerly imports the other.
"""
try:
    import torch
    import torch.nn as _torch_nn
    HAS_TORCH = True
except ImportError:
    torch = None
    _torch_nn = None
    HAS_TORCH = False

__all__ = ["HAS_TORCH", "torch", "_torch_nn"]
