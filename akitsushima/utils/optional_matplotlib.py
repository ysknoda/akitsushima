"""Optional matplotlib detection, isolated in its own module so that
importing it never drags in any *other* optional dependency (torch) --
see optional_torch.py for why
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

_warned = False


def have_matplotlib() -> bool:
    """True when matplotlib is installed.  Called where a plot is about to
    be made: without matplotlib it warns once, then plots are skipped.
    (It used to warn on every ``import akitsushima``, plot or no plot --
    ADR-0023 O1.)"""
    global _warned
    if not HAS_MATPLOTLIB and not _warned:
        _warned = True
        warnings.warn("matplotlib not found -- plots are skipped "
                      '(pip install "akitsushima[plotting]").', stacklevel=2)
    return HAS_MATPLOTLIB


__all__ = ["HAS_MATPLOTLIB", "plt", "have_matplotlib"]
