"""Scalar helper functions shared by the Stillinger-Weber and Tersoff numba
kernels. JIT-compiled here (once) so both ``stillinger_weber.py`` and
``tersoff.py`` can import the *already-compiled* callable -- numba's
nopython mode needs any function referenced inside a jitted function's
global namespace to itself already be jitted at compile time.
"""
import math

from ..utils.jit import _HAS_NUMBA, _numba_njit


def _nb_exp(x):
    """Scalar exp, clamped to avoid overflow -- inline-able by Numba."""
    if x > 500.0:
        return _nb_exp_clamp_hi
    if x < -500.0:
        return 0.0
    return math.exp(x)


# Saturation value returned for x > 500.  Computed rather than written as
# a literal so it cannot drift from the threshold it belongs to: the
# literal previously here, 5.184705528587072e+217, is exp(501.3067), not
# exp(500) as its comment claimed -- a factor of 3.69 -- which left the
# clamp discontinuous at its own cut-in point.  Physically this only ever
# matters deep in an already-unphysical repulsive regime (both values are
# effectively infinite; float64 does not overflow until ~exp(709)), so
# this is a self-consistency fix, not a change to any meaningful result.
_nb_exp_clamp_hi = math.exp(500.0)   # 1.4035922178528375e+217

if _HAS_NUMBA:
    _nb_exp = _numba_njit(_nb_exp)  # noqa: F811 -- reassign to the JIT version
