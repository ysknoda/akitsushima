"""Shared numba kernel for LJ/Morse/BornMayer/Buckingham pair potentials.

``pair_energy_force()`` pre-evaluates V(r)/dV-dr(r) in vectorised NumPy (the
formulas differ per potential); this kernel only does the generic
force/stress scatter, shared by every :class:`~.pair_potentials.BasePairPotential`
subclass.
"""
import numpy as np

from ..utils.jit import _HAS_NUMBA, _numba, _numba_par


def _pair_body_numpy(idx_i, idx_j, rij_arr, r_arr, forces, stress,
                      V_arr, dVdr_arr):
    """
    Generic pair-potential force scatter (serial).

    V_arr and dVdr_arr are pre-evaluated for each pair (by the Python caller).
    This kernel accumulates forces and stress; returns energy sum.
    """
    energy = 0.0
    for p in range(len(idx_i)):
        ii = idx_i[p]
        jj = idx_j[p]
        energy += V_arr[p]
        cx = (dVdr_arr[p] / r_arr[p])
        for d in range(3):
            fv = cx * rij_arr[p, d]
            forces[ii, d] += fv
            forces[jj, d] += -fv
        for di in range(3):
            for dj in range(3):
                stress[di, dj] -= cx * rij_arr[p, di] * rij_arr[p, dj]
    return energy


def _pair_body_numpy_par(idx_i, idx_j, rij_arr, r_arr, forces, stress,
                          V_arr, dVdr_arr):
    """
    Parallel version of _pair_body_numpy using per-thread accumulator
    buffers to avoid the write-write race on forces[ii]/forces[jj] (two
    different pairs can share an atom, so a naive prange over pairs would
    have multiple threads incrementing the SAME forces row simultaneously).

    Each thread accumulates into its own private slice of a
    (n_threads, N, 3) forces buffer and a (n_threads, 3, 3) stress buffer,
    indexed by numba.get_thread_id() -- so distinct threads never touch the
    same memory. After the parallel loop, the per-thread slices are summed
    into the caller-provided forces/stress arrays.

    Bug-fix note (memory safety, found via a segfault under the full test
    suite -- reproducible only after many other @njit(parallel=True)
    kernels had already run in the same process, never in isolation): the
    buffer used to be sized with ``numba.get_num_threads()``, the number of
    threads *currently* configured for a parallel region. But
    ``numba.get_thread_id()`` is not contractually bounded by whatever
    ``get_num_threads()`` last returned in a different call -- if the
    active thread count changes between the two (set_num_threads() is
    global/process-wide, and this codebase's own MPI+numba hybrid mode
    document expects callers to change it), get_thread_id() can return an
    index at or beyond the buffer's first dimension, corrupting adjacent
    memory. Sizing the buffer from ``numba.config.NUMBA_NUM_THREADS`` (the
    static maximum thread count numba was configured with -- see
    NUMBA_NUM_THREADS in utils/jit.py -- which get_thread_id() *is*
    contractually bounded by) removes that window entirely, at the cost of
    a few unused rows when fewer threads are active.
    """
    N = forces.shape[0]
    n_threads = _numba.config.NUMBA_NUM_THREADS
    forces_t = np.zeros((n_threads, N, 3))
    stress_t = np.zeros((n_threads, 3, 3))
    energy = 0.0

    for p in _numba.prange(len(idx_i)):
        tid = _numba.get_thread_id()
        ii = idx_i[p]
        jj = idx_j[p]
        r = r_arr[p]
        energy += V_arr[p]
        cx = dVdr_arr[p] / r
        for d in range(3):
            fv = cx * rij_arr[p, d]
            forces_t[tid, ii, d] += fv
            forces_t[tid, jj, d] -= fv
        for di in range(3):
            for dj in range(3):
                stress_t[tid, di, dj] -= cx * rij_arr[p, di] * rij_arr[p, dj]

    for t in range(n_threads):
        forces += forces_t[t]
        stress += stress_t[t]
    return energy


_pair_body_jit = _numba_par(_pair_body_numpy_par) if _HAS_NUMBA else _pair_body_numpy
