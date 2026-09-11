"""Optional matplotlib detection, isolated in its own module so that
importing it never drags in any *other* optional dependency (torch,
lightgbm, optuna) -- see optional_lightgbm.py / optional_torch.py for why
that isolation matters.
"""
import warnings

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not found -- trajectory plots will be skipped.")

__all__ = ["HAS_MATPLOTLIB", "plt"]
