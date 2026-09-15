"""Numba JIT setup (optional; transparent fall-back without numba).

When numba is installed, inner pair/triplet loops in the engine can be
compiled to native machine code (10-100x speed-up for large N). If numba is
not installed, ``_numba_njit``/``_numba_par`` degrade to identity decorators
so callers can always write ``fn_jit = _numba_par(fn_numpy)`` unconditionally
and get a working (plain Python) callable either way -- no per-kernel
``if _HAS_NUMBA: ... else: ...`` branching is needed at the call site.

Activation
----------
The JIT kernels are used automatically whenever numba is importable.
Set ``NUMBA_ENABLE=0`` to force the NumPy path -- for a like-for-like
comparison, or to sidestep the macOS duplicate-OpenMP conflict between
numba and pip's torch wheel (see the README).  ``use_numba=True`` on a
potential's ``compute()`` still forces the JIT path regardless, and the
legacy alias ``BUCKINGHAM_NUMBA=1`` is still accepted.

This used to default to OFF, so a user who installed the ``numba``
extra got the NumPy path anyway unless they happened to know about the
environment variable, which the README never mentioned -- while the
README did advertise the extra as a 10-100x speed-up.  Measured on
64-atom silicon: Stillinger-Weber 66 -> 4.2 ms/step (16x), Tersoff
117 -> 2.5 ms/step (47x).  The two paths are checked against each other
in tests/test_engine_forces.py and tests/test_neighborlist.py.

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

# On by default when numba is importable; NUMBA_ENABLE=0 opts out.
_OFF = ("0", "false", "no", "off", "")


def _env_on(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in _OFF


_NUMBA_ENV = _env_on("NUMBA_ENABLE", "1") or _env_on("BUCKINGHAM_NUMBA", "0")

if _HAS_NUMBA:
    _numba_njit = _numba.njit(cache=True, fastmath=True)
    _numba_par = _numba.njit(cache=True, fastmath=True, parallel=True)
else:
    def _numba_njit(fn):
        """Identity decorator used when numba is unavailable."""
        return fn

    _numba_par = _numba_njit
