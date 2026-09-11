"""Numba JIT setup (optional; transparent fall-back without numba).

When numba is installed, inner pair/triplet loops in the engine can be
compiled to native machine code (10-100x speed-up for large N). If numba is
not installed, ``_numba_njit``/``_numba_par`` degrade to identity decorators
so callers can always write ``fn_jit = _numba_par(fn_numpy)`` unconditionally
and get a working (plain Python) callable either way -- no per-kernel
``if _HAS_NUMBA: ... else: ...`` branching is needed at the call site.

Activation
----------
Pass ``use_numba=True`` to a potential's ``compute()``, or set the
environment variable ``NUMBA_ENABLE=1`` for a global default (the legacy
alias ``BUCKINGHAM_NUMBA=1`` is also accepted for backward compatibility).
Thread count for ``parallel=True`` kernels is controlled by the
``NUMBA_NUM_THREADS`` environment variable (numba's own convention).
"""
import os

try:
    import numba as _numba
    _HAS_NUMBA = True
except ImportError:
    _numba = None
    _HAS_NUMBA = False

# Auto-enable via environment variable (NUMBA_ENABLE=1 or legacy BUCKINGHAM_NUMBA=1)
_NUMBA_ENV = (
    os.environ.get("NUMBA_ENABLE", "0").strip() not in ("0", "false", "False", "")
    or os.environ.get("BUCKINGHAM_NUMBA", "0").strip() not in ("0", "false", "False", "")
)

if _HAS_NUMBA:
    _numba_njit = _numba.njit(cache=True, fastmath=True)
    _numba_par = _numba.njit(cache=True, fastmath=True, parallel=True)
else:
    def _numba_njit(fn):
        """Identity decorator used when numba is unavailable."""
        return fn

    _numba_par = _numba_njit
