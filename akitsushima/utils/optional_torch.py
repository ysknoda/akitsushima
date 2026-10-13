"""Optional PyTorch detection, isolated in its own module.

Kept separate from optional_matplotlib.py so that a caller who only needs
one of these optional, heavyweight (and, for torch, OpenMP-based -- see
fitting/models/__init__.py's docstring for a concrete instability this
isolation avoids) dependencies never eagerly imports the other.
"""
# numba's threading layer must be up before torch loads its own OpenMP
# runtime (see utils/jit.py); importing jit starts it.
from . import jit as _jit  # noqa: F401

try:
    import torch
    import torch.nn as _torch_nn
    HAS_TORCH = True
except ImportError:
    torch = None
    _torch_nn = None
    HAS_TORCH = False

__all__ = ["HAS_TORCH", "torch", "_torch_nn"]
