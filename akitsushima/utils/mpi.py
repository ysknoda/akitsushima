"""MPI support (optional; transparent fall-back without mpi4py).

Launch:  mpirun -n 4 python3 -m akitsushima ...
Hybrid:  NUMBA_NUM_THREADS=8 mpirun -n 4 python3 -m akitsushima ...

All MPI-aware ``compute()`` methods call :func:`_mpi_allreduce_sum` for
forces/stress and use :func:`_mpi_domain_split` to partition atoms/pairs
across ranks.

Nested parallelism and :func:`local_only`
------------------------------------------
Every engine-level potential (``pair_potentials.py``, ``stillinger_weber.py``,
``tersoff.py``, ``eam.py``, ``coulomb/ewald.py``) unconditionally splits its
OWN internal work (pairs / atoms / k-vectors) across every rank on the
single, global ``COMM_WORLD`` communicator and Allreduces the result at the
end of every ``compute()`` call. That is the right granularity when one
rank set is cooperatively evaluating a SINGLE shared structure (e.g. one MD
trajectory, one relaxation).

Some callers instead want to distribute independent ITEMS across ranks --
different NEB images, different phonon finite-displacements, different
fitting structures -- so that each rank computes a complete, independent
result for its own item(s). Naively combining both levels on the same
communicator is wrong: if rank 0 is evaluating item A while rank 1 is
evaluating item B, an unconditional Allreduce inside ``compute()`` would
average/sum together two unrelated items' partial sums, silently producing
garbage (discovered via a bit-for-bit divergence between a 1-rank and a
3-rank run of the same phonon IFC calculation, both of which "succeeded"
with no error, crash, or hang -- only the resulting numbers were wrong).

:func:`local_only` lets an outer, item-splitting caller suppress the
INNER per-call splitting/reduction for the duration of a ``with`` block,
without touching ``_HAS_MPI``/``_mpi_rank``/``_mpi_size`` themselves (other,
unrelated code outside the block is unaffected). Enter it ONCE around an
entire per-rank batch of local work (not once per item/thread) so that
concurrent reads of the suppression flag from worker threads (e.g. under
``ThreadPoolExecutor``) never race with the enter/exit writes, which only
ever happen on the single calling thread.
"""
import contextlib

import numpy as np

try:
    from mpi4py import MPI as _MPI
    _mpi_comm = _MPI.COMM_WORLD
    _mpi_rank = _mpi_comm.Get_rank()
    _mpi_size = _mpi_comm.Get_size()
    _HAS_MPI = (_mpi_size > 1)
except ImportError:
    _MPI = None
    _mpi_comm = None
    _mpi_rank = 0
    _mpi_size = 1
    _HAS_MPI = False


_mpi_suppress_depth = 0


def _mpi_is_active() -> bool:
    """True when inner-level MPI splitting/reduction should apply here:
    real multi-rank MPI is running AND no enclosing :func:`local_only`
    block has suppressed it."""
    return _HAS_MPI and _mpi_suppress_depth == 0


@contextlib.contextmanager
def local_only():
    """
    Suppress :func:`_mpi_domain_split`/:func:`_mpi_allreduce_sum`/
    :func:`_mpi_allreduce_sum_scalar` for the duration of this ``with``
    block (see the module docstring for why this is needed).

    Re-entrant: nesting increments/decrements a depth counter, so a
    ``local_only()`` block calling code that itself opens another
    ``local_only()`` block behaves correctly (suppressed until the
    OUTERMOST block exits).
    """
    global _mpi_suppress_depth
    _mpi_suppress_depth += 1
    try:
        yield
    finally:
        _mpi_suppress_depth -= 1


def _mpi_allreduce_sum(arr: np.ndarray) -> np.ndarray:
    """Sum a NumPy array across all MPI ranks (Allreduce SUM)."""
    if not _mpi_is_active():
        return arr
    out = np.empty_like(arr)
    _mpi_comm.Allreduce(arr, out, op=_MPI.SUM)
    return out


def _mpi_allreduce_sum_scalar(value: float) -> float:
    """Sum a Python/NumPy scalar across all MPI ranks (Allreduce SUM)."""
    if not _mpi_is_active():
        return value
    return float(_mpi_comm.allreduce(float(value), op=_MPI.SUM))


def _mpi_bcast(obj, root: int = 0):
    """Broadcast a Python object from root to all ranks."""
    if not _HAS_MPI:
        return obj
    return _mpi_comm.bcast(obj, root=root)


def _mpi_domain_split(N: int) -> tuple:
    """
    Evenly split N items across MPI ranks.

    Returns (i_start, i_end); the last rank absorbs any remainder.
    When MPI is not active (or suppressed by :func:`local_only`) returns
    (0, N).
    """
    if not _mpi_is_active():
        return 0, N
    chunk = N // _mpi_size
    i_start = _mpi_rank * chunk
    i_end = N if _mpi_rank == _mpi_size - 1 else i_start + chunk
    return int(i_start), int(i_end)
