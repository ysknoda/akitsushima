"""Tersoff bond-order potential for covalent systems (multi-species).

Reference: Tersoff, Phys. Rev. Lett. 56, 632 (1986);
           Phys. Rev. B 37, 6991 (1988); Phys. Rev. B 38, 9902 (1988).

Pair-sum convention (four duplicated kernels depend on it)
----------------------------------------------------------
The total energy is

    E = (1/2) sum_{i != j} fc(r_ij) [ fR(r_ij) + b_ij fA(r_ij) ]

and every kernel here iterates over ``NeighbourList.get_candidate_pairs()``,
which yields each pair EXACTLY ONCE (i < j), with the bond order already
symmetrised as ``b_avg = (b_ij + b_ji) / 2``.  Summing
``fc * (fR + b_avg * fA)`` over those unique pairs therefore reproduces the
double sum with a prefactor of 1, not 1/2 -- the 1/2 is already spent on
symmetrising ``b``.  For the same reason the three-body force correction
carries ``d(b_avg)/d(zeta_ij) = (1/2) db_ij/d(zeta_ij)``, i.e. ``coeff =
0.5 * fc_ij * fA * dbij``.

This is spelled out because an earlier revision carried an extra 0.5 at
all sixteen of those sites, returning exactly HALF the correct energy,
forces and stress for every Tersoff calculation.  It was invisible to the
finite-difference tests, which compare forces against the derivative of
this same (uniformly halved) energy, and to the numba-vs-NumPy tests,
which compare two paths that were halved identically.  What caught it was
an absolute check against a published reference value -- Tersoff T3
silicon, whose parameter set is fit to a0 = 5.432 A and E_coh = 4.63
eV/atom; see tests/test_engine_forces.py::
test_tersoff_t3_silicon_reproduces_published_cohesive_energy.
"""
import math
from typing import Optional, Tuple

import numpy as np

from ..utils.jit import _HAS_NUMBA, _numba, _numba_njit, _numba_par
from ..utils.mpi import (
    _HAS_MPI,
    _mpi_allreduce_sum,
    _mpi_allreduce_sum_scalar,
    _mpi_domain_split,
)
from ._numba_common import _nb_exp
from .neighborlist import NeighbourList


def _te_fc(r, R, D):
    """Tersoff smooth cutoff function (scalar)."""
    if r < R - D:
        return 1.0
    if r > R + D:
        return 0.0
    return 0.5 - 0.5 * math.sin(0.5 * math.pi * (r - R) / D)


if _HAS_NUMBA:
    _te_fc = _numba_njit(_te_fc)  # noqa: F811 -- reassign to the JIT version


# -- Tersoff: bond-order (zeta) pre-computation kernel -------------------

def _te_zeta_numpy(P, idx_i, idx_j, rij_arr, r_arr,
                    nb_start, nb_idx, nb_rij, nb_r,
                    zeta_ij, zeta_ji,
                    l3, c_t, d_t, h_t, R_t, D_t):
    """Compute zeta_ij and zeta_ji for all pairs."""
    for p in range(P):
        ii = idx_i[p]
        jj = idx_j[p]
        r_ij = r_arr[p]
        # zeta_ij (central = i)
        for nb in range(nb_start[ii], nb_start[ii + 1]):
            kk = nb_idx[nb]
            if kk == jj:
                continue
            r_ik = nb_r[nb]
            fc_ik = _te_fc(r_ik, R_t, D_t)
            if fc_ik == 0.0:
                continue
            dot = 0.0
            for d in range(3):
                dot += rij_arr[p, d] * nb_rij[nb, d]
            cos_t = dot / (r_ij * r_ik)
            g_t = 1.0 + c_t * c_t / (d_t * d_t) - c_t * c_t / (d_t * d_t + (h_t - cos_t)**2)
            zeta_ij[p] += fc_ik * g_t * _nb_exp(l3**3 * (r_ij - r_ik)**3)
        # zeta_ji (central = j)
        for nb in range(nb_start[jj], nb_start[jj + 1]):
            kk = nb_idx[nb]
            if kk == ii:
                continue
            r_jk = nb_r[nb]
            fc_jk = _te_fc(r_jk, R_t, D_t)
            if fc_jk == 0.0:
                continue
            dot = 0.0
            for d in range(3):
                dot += (-rij_arr[p, d]) * nb_rij[nb, d]
            cos_t = dot / (r_arr[p] * r_jk)
            g_t = 1.0 + c_t * c_t / (d_t * d_t) - c_t * c_t / (d_t * d_t + (h_t - cos_t)**2)
            zeta_ji[p] += fc_jk * g_t * _nb_exp(l3**3 * (r_arr[p] - r_jk)**3)


def _te_zterm_scalar(rijx, rijy, rijz, rikx, riky, rikz,
                      R, D, c_t, d_t, h_t, l3, m):
    """
    Scalar Tersoff zeta angular/exponential term for one (i;j,k) triplet,
    given the relative-displacement vectors r_ij and r_ik.

    This is a plain, Numba-nopython-compatible scalar function used as the
    building block for the numerical (central-difference) 3-body force
    derivation in _te_3body_force_numpy / _te_3body_force_numpy_par.
    """
    r_ij = math.sqrt(rijx * rijx + rijy * rijy + rijz * rijz)
    r_ik = math.sqrt(rikx * rikx + riky * riky + rikz * rikz)
    fc_ik = _te_fc(r_ik, R, D)
    if fc_ik == 0.0:
        return 0.0
    cos_t = (rijx * rikx + rijy * riky + rijz * rikz) / (r_ij * r_ik)
    g_t = 1.0 + c_t * c_t / (d_t * d_t) - c_t * c_t / (d_t * d_t + (h_t - cos_t)**2)
    return fc_ik * g_t * _nb_exp(l3**m * (r_ij - r_ik)**m)


if _HAS_NUMBA:
    _te_zterm_scalar = _numba_njit(_te_zterm_scalar)  # noqa: F811


def _te_dfc_scalar(r, R, D):
    """Scalar derivative of the Tersoff smooth cutoff function _te_fc.
    Needed (alongside _te_fc) by the multi-species JIT kernels' 2-body
    force term; the legacy single-species path instead differentiates a
    vectorised NumPy _dfc(r_) closure inline, which Numba's nopython mode
    cannot JIT directly."""
    if r < R - D or r > R + D:
        return 0.0
    return -0.25 * math.pi / D * math.cos(0.5 * math.pi * (r - R) / D)


def _te_zterm_scalar_multi(rijx, rijy, rijz, rikx, riky, rikz,
                           R, D, c_t, d_t, h_t, l3, m, gam):
    """Table-driven analogue of _te_zterm_scalar: unlike the legacy
    single-species term (implicit gamma=1), the multi-species format's
    zeta sum (see TersoffPotential._compute_multi_species) includes an
    explicit per-triple gamma prefactor, so this scalar reproduces that
    exact formula (gam * fc_ik * g_t * exp(...)), matching _sw_h3_scalar_multi's
    analogous gam_t note in stillinger_weber.py."""
    r_ij = math.sqrt(rijx * rijx + rijy * rijy + rijz * rijz)
    r_ik = math.sqrt(rikx * rikx + riky * riky + rikz * rikz)
    fc_ik = _te_fc(r_ik, R, D)
    if fc_ik == 0.0:
        return 0.0
    cos_t = (rijx * rikx + rijy * riky + rijz * rikz) / (r_ij * r_ik)
    g_t = 1.0 + c_t * c_t / (d_t * d_t) - c_t * c_t / (d_t * d_t + (h_t - cos_t)**2)
    return gam * fc_ik * g_t * _nb_exp(l3**m * (r_ij - r_ik)**m)


if _HAS_NUMBA:
    _te_dfc_scalar = _numba_njit(_te_dfc_scalar)  # noqa: F811
    _te_zterm_scalar_multi = _numba_njit(_te_zterm_scalar_multi)  # noqa: F811


def _te_3body_force_numpy(p_start, p_end,
                           idx_i, idx_j, rij_arr, r_arr,
                           fc_arr, fa_arr, dbij_ij, dbij_ji,
                           nb_start, nb_idx, nb_rij, nb_r,
                           forces, stress,
                           R, D, c_t, d_t, h_t, l3, m):
    """
    Tersoff 3-body force/stress correction from d(bond-order)/d(position),
    computed via central finite differences of _te_zterm_scalar.
    """
    hh = 1e-6
    for p in range(p_start, p_end):
        ii = idx_i[p]
        jj = idx_j[p]
        rijx = rij_arr[p, 0]
        rijy = rij_arr[p, 1]
        rijz = rij_arr[p, 2]
        fc_ij = fc_arr[p]
        fa_ij = fa_arr[p]
        dbi = dbij_ij[p]
        dbj = dbij_ji[p]

        if abs(dbi) > 1e-30:
            for nb in range(nb_start[ii], nb_start[ii + 1]):
                kk = nb_idx[nb]
                if kk == jj:
                    continue
                rikx = nb_rij[nb, 0]
                riky = nb_rij[nb, 1]
                rikz = nb_rij[nb, 2]
                r_ik = nb_r[nb]
                fc_ik = _te_fc(r_ik, R, D)
                if fc_ik == 0.0:
                    continue
                coeff = 0.5 * fc_ij * fa_ij * dbi

                dzj0 = (_te_zterm_scalar(rijx + hh, rijy, rijz, rikx, riky, rikz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rijx - hh, rijy, rijz, rikx, riky, rikz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzj1 = (_te_zterm_scalar(rijx, rijy + hh, rijz, rikx, riky, rikz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rijx, rijy - hh, rijz, rikx, riky, rikz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzj2 = (_te_zterm_scalar(rijx, rijy, rijz + hh, rikx, riky, rikz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rijx, rijy, rijz - hh, rikx, riky, rikz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzk0 = (_te_zterm_scalar(rijx, rijy, rijz, rikx + hh, riky, rikz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rijx, rijy, rijz, rikx - hh, riky, rikz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzk1 = (_te_zterm_scalar(rijx, rijy, rijz, rikx, riky + hh, rikz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rijx, rijy, rijz, rikx, riky - hh, rikz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzk2 = (_te_zterm_scalar(rijx, rijy, rijz, rikx, riky, rikz + hh, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rijx, rijy, rijz, rikx, riky, rikz - hh, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)

                Fj0 = -coeff * dzj0
                Fj1 = -coeff * dzj1
                Fj2 = -coeff * dzj2
                Fk0 = -coeff * dzk0
                Fk1 = -coeff * dzk1
                Fk2 = -coeff * dzk2
                Fi0 = coeff * (dzj0 + dzk0)
                Fi1 = coeff * (dzj1 + dzk1)
                Fi2 = coeff * (dzj2 + dzk2)

                forces[ii, 0] += Fi0
                forces[ii, 1] += Fi1
                forces[ii, 2] += Fi2
                forces[jj, 0] += Fj0
                forces[jj, 1] += Fj1
                forces[jj, 2] += Fj2
                forces[kk, 0] += Fk0
                forces[kk, 1] += Fk1
                forces[kk, 2] += Fk2

                stress[0, 0] += Fj0 * rijx + Fk0 * rikx
                stress[0, 1] += Fj0 * rijy + Fk0 * riky
                stress[0, 2] += Fj0 * rijz + Fk0 * rikz
                stress[1, 0] += Fj1 * rijx + Fk1 * rikx
                stress[1, 1] += Fj1 * rijy + Fk1 * riky
                stress[1, 2] += Fj1 * rijz + Fk1 * rikz
                stress[2, 0] += Fj2 * rijx + Fk2 * rikx
                stress[2, 1] += Fj2 * rijy + Fk2 * riky
                stress[2, 2] += Fj2 * rijz + Fk2 * rikz

        if abs(dbj) > 1e-30:
            rjix = -rijx
            rjiy = -rijy
            rjiz = -rijz
            for nb in range(nb_start[jj], nb_start[jj + 1]):
                kk = nb_idx[nb]
                if kk == ii:
                    continue
                rjkx = nb_rij[nb, 0]
                rjky = nb_rij[nb, 1]
                rjkz = nb_rij[nb, 2]
                r_jk = nb_r[nb]
                fc_jk = _te_fc(r_jk, R, D)
                if fc_jk == 0.0:
                    continue
                coeff = 0.5 * fc_ij * fa_ij * dbj

                dzi0 = (_te_zterm_scalar(rjix + hh, rjiy, rjiz, rjkx, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rjix - hh, rjiy, rjiz, rjkx, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzi1 = (_te_zterm_scalar(rjix, rjiy + hh, rjiz, rjkx, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rjix, rjiy - hh, rjiz, rjkx, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzi2 = (_te_zterm_scalar(rjix, rjiy, rjiz + hh, rjkx, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rjix, rjiy, rjiz - hh, rjkx, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzk0 = (_te_zterm_scalar(rjix, rjiy, rjiz, rjkx + hh, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rjix, rjiy, rjiz, rjkx - hh, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzk1 = (_te_zterm_scalar(rjix, rjiy, rjiz, rjkx, rjky + hh, rjkz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rjix, rjiy, rjiz, rjkx, rjky - hh, rjkz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzk2 = (_te_zterm_scalar(rjix, rjiy, rjiz, rjkx, rjky, rjkz + hh, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rjix, rjiy, rjiz, rjkx, rjky, rjkz - hh, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)

                Fi0 = -coeff * dzi0
                Fi1 = -coeff * dzi1
                Fi2 = -coeff * dzi2
                Fk0 = -coeff * dzk0
                Fk1 = -coeff * dzk1
                Fk2 = -coeff * dzk2
                Fj0 = coeff * (dzi0 + dzk0)
                Fj1 = coeff * (dzi1 + dzk1)
                Fj2 = coeff * (dzi2 + dzk2)

                forces[jj, 0] += Fj0
                forces[jj, 1] += Fj1
                forces[jj, 2] += Fj2
                forces[ii, 0] += Fi0
                forces[ii, 1] += Fi1
                forces[ii, 2] += Fi2
                forces[kk, 0] += Fk0
                forces[kk, 1] += Fk1
                forces[kk, 2] += Fk2

                stress[0, 0] += Fi0 * rjix + Fk0 * rjkx
                stress[0, 1] += Fi0 * rjiy + Fk0 * rjky
                stress[0, 2] += Fi0 * rjiz + Fk0 * rjkz
                stress[1, 0] += Fi1 * rjix + Fk1 * rjkx
                stress[1, 1] += Fi1 * rjiy + Fk1 * rjky
                stress[1, 2] += Fi1 * rjiz + Fk1 * rjkz
                stress[2, 0] += Fi2 * rjix + Fk2 * rjkx
                stress[2, 1] += Fi2 * rjiy + Fk2 * rjky
                stress[2, 2] += Fi2 * rjiz + Fk2 * rjkz


def _te_3body_force_numpy_par(p_start, p_end,
                               idx_i, idx_j, rij_arr, r_arr,
                               fc_arr, fa_arr, dbij_ij, dbij_ji,
                               nb_start, nb_idx, nb_rij, nb_r,
                               forces, stress,
                               R, D, c_t, d_t, h_t, l3, m):
    """
    Parallel version of _te_3body_force_numpy, using per-thread
    accumulator buffers (force contributions to neighbour atoms j/k can
    collide across different central pairs handled by different threads).
    See _pair_kernels._pair_body_numpy_par for why the buffer is sized
    from numba.config.NUMBA_NUM_THREADS rather than numba.get_num_threads().
    """
    N = forces.shape[0]
    n_threads = _numba.config.NUMBA_NUM_THREADS
    forces_t = np.zeros((n_threads, N, 3))
    stress_t = np.zeros((n_threads, 3, 3))
    hh = 1e-6

    for p in _numba.prange(p_start, p_end):
        tid = _numba.get_thread_id()
        ii = idx_i[p]
        jj = idx_j[p]
        rijx = rij_arr[p, 0]
        rijy = rij_arr[p, 1]
        rijz = rij_arr[p, 2]
        fc_ij = fc_arr[p]
        fa_ij = fa_arr[p]
        dbi = dbij_ij[p]
        dbj = dbij_ji[p]

        if abs(dbi) > 1e-30:
            for nb in range(nb_start[ii], nb_start[ii + 1]):
                kk = nb_idx[nb]
                if kk == jj:
                    continue
                rikx = nb_rij[nb, 0]
                riky = nb_rij[nb, 1]
                rikz = nb_rij[nb, 2]
                r_ik = nb_r[nb]
                fc_ik = _te_fc(r_ik, R, D)
                if fc_ik == 0.0:
                    continue
                coeff = 0.5 * fc_ij * fa_ij * dbi

                dzj0 = (_te_zterm_scalar(rijx + hh, rijy, rijz, rikx, riky, rikz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rijx - hh, rijy, rijz, rikx, riky, rikz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzj1 = (_te_zterm_scalar(rijx, rijy + hh, rijz, rikx, riky, rikz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rijx, rijy - hh, rijz, rikx, riky, rikz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzj2 = (_te_zterm_scalar(rijx, rijy, rijz + hh, rikx, riky, rikz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rijx, rijy, rijz - hh, rikx, riky, rikz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzk0 = (_te_zterm_scalar(rijx, rijy, rijz, rikx + hh, riky, rikz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rijx, rijy, rijz, rikx - hh, riky, rikz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzk1 = (_te_zterm_scalar(rijx, rijy, rijz, rikx, riky + hh, rikz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rijx, rijy, rijz, rikx, riky - hh, rikz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzk2 = (_te_zterm_scalar(rijx, rijy, rijz, rikx, riky, rikz + hh, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rijx, rijy, rijz, rikx, riky, rikz - hh, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)

                Fj0 = -coeff * dzj0
                Fj1 = -coeff * dzj1
                Fj2 = -coeff * dzj2
                Fk0 = -coeff * dzk0
                Fk1 = -coeff * dzk1
                Fk2 = -coeff * dzk2
                Fi0 = coeff * (dzj0 + dzk0)
                Fi1 = coeff * (dzj1 + dzk1)
                Fi2 = coeff * (dzj2 + dzk2)

                forces_t[tid, ii, 0] += Fi0
                forces_t[tid, ii, 1] += Fi1
                forces_t[tid, ii, 2] += Fi2
                forces_t[tid, jj, 0] += Fj0
                forces_t[tid, jj, 1] += Fj1
                forces_t[tid, jj, 2] += Fj2
                forces_t[tid, kk, 0] += Fk0
                forces_t[tid, kk, 1] += Fk1
                forces_t[tid, kk, 2] += Fk2

                stress_t[tid, 0, 0] += Fj0 * rijx + Fk0 * rikx
                stress_t[tid, 0, 1] += Fj0 * rijy + Fk0 * riky
                stress_t[tid, 0, 2] += Fj0 * rijz + Fk0 * rikz
                stress_t[tid, 1, 0] += Fj1 * rijx + Fk1 * rikx
                stress_t[tid, 1, 1] += Fj1 * rijy + Fk1 * riky
                stress_t[tid, 1, 2] += Fj1 * rijz + Fk1 * rikz
                stress_t[tid, 2, 0] += Fj2 * rijx + Fk2 * rikx
                stress_t[tid, 2, 1] += Fj2 * rijy + Fk2 * riky
                stress_t[tid, 2, 2] += Fj2 * rijz + Fk2 * rikz

        if abs(dbj) > 1e-30:
            rjix = -rijx
            rjiy = -rijy
            rjiz = -rijz
            for nb in range(nb_start[jj], nb_start[jj + 1]):
                kk = nb_idx[nb]
                if kk == ii:
                    continue
                rjkx = nb_rij[nb, 0]
                rjky = nb_rij[nb, 1]
                rjkz = nb_rij[nb, 2]
                r_jk = nb_r[nb]
                fc_jk = _te_fc(r_jk, R, D)
                if fc_jk == 0.0:
                    continue
                coeff = 0.5 * fc_ij * fa_ij * dbj

                dzi0 = (_te_zterm_scalar(rjix + hh, rjiy, rjiz, rjkx, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rjix - hh, rjiy, rjiz, rjkx, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzi1 = (_te_zterm_scalar(rjix, rjiy + hh, rjiz, rjkx, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rjix, rjiy - hh, rjiz, rjkx, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzi2 = (_te_zterm_scalar(rjix, rjiy, rjiz + hh, rjkx, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rjix, rjiy, rjiz - hh, rjkx, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzk0 = (_te_zterm_scalar(rjix, rjiy, rjiz, rjkx + hh, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rjix, rjiy, rjiz, rjkx - hh, rjky, rjkz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzk1 = (_te_zterm_scalar(rjix, rjiy, rjiz, rjkx, rjky + hh, rjkz, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rjix, rjiy, rjiz, rjkx, rjky - hh, rjkz, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)
                dzk2 = (_te_zterm_scalar(rjix, rjiy, rjiz, rjkx, rjky, rjkz + hh, R, D, c_t, d_t, h_t, l3, m)
                        - _te_zterm_scalar(rjix, rjiy, rjiz, rjkx, rjky, rjkz - hh, R, D, c_t, d_t, h_t, l3, m)) / (2 * hh)

                Fi0 = -coeff * dzi0
                Fi1 = -coeff * dzi1
                Fi2 = -coeff * dzi2
                Fk0 = -coeff * dzk0
                Fk1 = -coeff * dzk1
                Fk2 = -coeff * dzk2
                Fj0 = coeff * (dzi0 + dzk0)
                Fj1 = coeff * (dzi1 + dzk1)
                Fj2 = coeff * (dzi2 + dzk2)

                forces_t[tid, jj, 0] += Fj0
                forces_t[tid, jj, 1] += Fj1
                forces_t[tid, jj, 2] += Fj2
                forces_t[tid, ii, 0] += Fi0
                forces_t[tid, ii, 1] += Fi1
                forces_t[tid, ii, 2] += Fi2
                forces_t[tid, kk, 0] += Fk0
                forces_t[tid, kk, 1] += Fk1
                forces_t[tid, kk, 2] += Fk2

                stress_t[tid, 0, 0] += Fi0 * rjix + Fk0 * rjkx
                stress_t[tid, 0, 1] += Fi0 * rjiy + Fk0 * rjky
                stress_t[tid, 0, 2] += Fi0 * rjiz + Fk0 * rjkz
                stress_t[tid, 1, 0] += Fi1 * rjix + Fk1 * rjkx
                stress_t[tid, 1, 1] += Fi1 * rjiy + Fk1 * rjky
                stress_t[tid, 1, 2] += Fi1 * rjiz + Fk1 * rjkz
                stress_t[tid, 2, 0] += Fi2 * rjix + Fk2 * rjkx
                stress_t[tid, 2, 1] += Fi2 * rjiy + Fk2 * rjky
                stress_t[tid, 2, 2] += Fi2 * rjiz + Fk2 * rjkz

    for t in range(n_threads):
        forces += forces_t[t]
        stress += stress_t[t]


def _te_zeta_numpy_par(P, idx_i, idx_j, rij_arr, r_arr,
                        nb_start, nb_idx, nb_rij, nb_r,
                        zeta_ij, zeta_ji,
                        l3, c_t, d_t, h_t, R_t, D_t):
    """
    Parallel version of _te_zeta_numpy, using an explicit prange over the
    pair index p.

    This loop is SAFE to parallelize without any thread-local buffers:
    each iteration p writes ONLY to zeta_ij[p] and zeta_ji[p], which are
    unique per-iteration slots that no other iteration ever touches.
    """
    for p in _numba.prange(P):
        ii = idx_i[p]
        jj = idx_j[p]
        r_ij = r_arr[p]
        # zeta_ij (central = i)
        for nb in range(nb_start[ii], nb_start[ii + 1]):
            kk = nb_idx[nb]
            if kk == jj:
                continue
            r_ik = nb_r[nb]
            fc_ik = _te_fc(r_ik, R_t, D_t)
            if fc_ik == 0.0:
                continue
            dot = 0.0
            for d in range(3):
                dot += rij_arr[p, d] * nb_rij[nb, d]
            cos_t = dot / (r_ij * r_ik)
            g_t = 1.0 + c_t * c_t / (d_t * d_t) - c_t * c_t / (d_t * d_t + (h_t - cos_t)**2)
            zeta_ij[p] += fc_ik * g_t * _nb_exp(l3**3 * (r_ij - r_ik)**3)
        # zeta_ji (central = j)
        for nb in range(nb_start[jj], nb_start[jj + 1]):
            kk = nb_idx[nb]
            if kk == ii:
                continue
            r_jk = nb_r[nb]
            fc_jk = _te_fc(r_jk, R_t, D_t)
            if fc_jk == 0.0:
                continue
            dot = 0.0
            for d in range(3):
                dot += (-rij_arr[p, d]) * nb_rij[nb, d]
            cos_t = dot / (r_arr[p] * r_jk)
            g_t = 1.0 + c_t * c_t / (d_t * d_t) - c_t * c_t / (d_t * d_t + (h_t - cos_t)**2)
            zeta_ji[p] += fc_jk * g_t * _nb_exp(l3**3 * (r_arr[p] - r_jk)**3)


# -- Tersoff: multi-species kernels (Phase 5) -----------------------------
#
# Same "resolve dict lookups into dense per-species-id arrays ONCE,
# outside the JIT boundary" principle as stillinger_weber.py's
# _sw_multi_2body_numpy/_sw_multi_3body_numpy -- see that module's
# comment for the full rationale. Tersoff's multi-species path needs TWO
# separate tables: a (n_sp, n_sp) pair table (A, B, lambda1, lambda2, R,
# D, beta, n -- everything self._two_body(si, sj) can return) and a
# (n_sp, n_sp, n_sp) triple table (c, d, h, lambda3, m, gamma -- from
# self._three_body(si, sj, sk)), since the zeta sum and the 3-body force
# correction both need a triple's angular parameters AND its k-neighbour
# pair's own (R, D) cutoff simultaneously. Scope note: unlike the legacy
# single-species kernels (_te_zeta_jit/_te_3body_force_jit), these are
# JIT-compiled WITHOUT numba.prange multi-threading (a deliberate,
# smaller-scope choice given the added complexity two interacting
# per-species-id tables would add on top of thread-safe accumulator
# buffers) -- still a substantial speed-up over pure Python/dict lookups
# from JIT compilation alone, matching the same scope decision made for
# stillinger_weber.py's multi-species kernels.

def _te_multi_zeta_numpy(P, idx_i, idx_j, rij_arr, r_arr, type_ids,
                         nb_start, nb_idx, nb_rij, nb_r,
                         zeta_ij, zeta_ji,
                         tb_valid, tb_R, tb_D,
                         tri_valid, tri_c, tri_d, tri_h, tri_l3, tri_m, tri_gam):
    """Table-driven multi-species analogue of _te_zeta_numpy. Reproduces
    the original Python loop's exact gating: if tb_valid[si, sj] is
    False (no (i,j) pair table entry), BOTH zeta_ij[p] and zeta_ji[p]
    are left at zero for this pair -- matching the original's single
    `if tb_ij is None: continue` gating both computations, not just the
    zeta_ij one, even though the k-loop for zeta_ji conceptually depends
    only on the (j,k) pair. This is a property of the EXISTING NumPy
    path being replicated exactly, not something introduced here."""
    for p in range(P):
        ii = idx_i[p]
        jj = idx_j[p]
        si = type_ids[ii]
        sj = type_ids[jj]
        if not tb_valid[si, sj]:
            continue
        r_ij = r_arr[p]
        rijx = rij_arr[p, 0]
        rijy = rij_arr[p, 1]
        rijz = rij_arr[p, 2]

        # zeta_ij: sum over k neighbouring i (excluding j)
        for nb in range(nb_start[ii], nb_start[ii + 1]):
            kk = nb_idx[nb]
            if kk == jj:
                continue
            sk = type_ids[kk]
            if not tb_valid[si, sk] or not tri_valid[si, sj, sk]:
                continue
            R_ik = tb_R[si, sk]
            D_ik = tb_D[si, sk]
            r_ik = nb_r[nb]
            fc_ik = _te_fc(r_ik, R_ik, D_ik)
            if fc_ik == 0.0:
                continue
            dot = 0.0
            for d in range(3):
                dot += rij_arr[p, d] * nb_rij[nb, d]
            cos_t = dot / (r_ij * r_ik)
            c_t = tri_c[si, sj, sk]
            d_t = tri_d[si, sj, sk]
            h_t = tri_h[si, sj, sk]
            l3 = tri_l3[si, sj, sk]
            m_t = tri_m[si, sj, sk]
            gam = tri_gam[si, sj, sk]
            g_t = 1.0 + c_t * c_t / (d_t * d_t) - c_t * c_t / (d_t * d_t + (h_t - cos_t)**2)
            zeta_ij[p] += gam * fc_ik * g_t * _nb_exp(l3**m_t * (r_ij - r_ik)**m_t)

        # zeta_ji: sum over k neighbouring j (excluding i)
        rjix = -rijx
        rjiy = -rijy
        rjiz = -rijz
        for nb in range(nb_start[jj], nb_start[jj + 1]):
            kk = nb_idx[nb]
            if kk == ii:
                continue
            sk = type_ids[kk]
            if not tb_valid[sj, sk] or not tri_valid[sj, si, sk]:
                continue
            R_jk = tb_R[sj, sk]
            D_jk = tb_D[sj, sk]
            r_jk = nb_r[nb]
            fc_jk = _te_fc(r_jk, R_jk, D_jk)
            if fc_jk == 0.0:
                continue
            dot = rjix * nb_rij[nb, 0] + rjiy * nb_rij[nb, 1] + rjiz * nb_rij[nb, 2]
            cos_t = dot / (r_ij * r_jk)
            c_t = tri_c[sj, si, sk]
            d_t = tri_d[sj, si, sk]
            h_t = tri_h[sj, si, sk]
            l3 = tri_l3[sj, si, sk]
            m_t = tri_m[sj, si, sk]
            gam = tri_gam[sj, si, sk]
            g_t = 1.0 + c_t * c_t / (d_t * d_t) - c_t * c_t / (d_t * d_t + (h_t - cos_t)**2)
            zeta_ji[p] += gam * fc_jk * g_t * _nb_exp(l3**m_t * (r_ij - r_jk)**m_t)


def _te_multi_pass3_numpy(p_start, p_end,
                          idx_i, idx_j, rij_arr, r_arr, type_ids,
                          bij_ij, bij_ji, dbij_ij, dbij_ji,
                          nb_start, nb_idx, nb_rij, nb_r,
                          forces, stress,
                          tb_valid, tb_A, tb_B, tb_l1, tb_l2, tb_R, tb_D,
                          tri_valid, tri_c, tri_d, tri_h, tri_l3, tri_m, tri_gam):
    """Table-driven multi-species analogue combining the original's
    2-body energy/force loop and its 3-body (d(bond-order)/d(position))
    force correction -- kept as one kernel since the original interleaves
    them within a single per-pair loop body (the 3-body correction reuses
    fc_ij/fa computed for the 2-body term of the SAME pair)."""
    energy = 0.0
    hh = 1e-6
    for p in range(p_start, p_end):
        ii = idx_i[p]
        jj = idx_j[p]
        si = type_ids[ii]
        sj = type_ids[jj]
        if not tb_valid[si, sj]:
            continue
        A_t = tb_A[si, sj]
        B_t = tb_B[si, sj]
        l1 = tb_l1[si, sj]
        l2 = tb_l2[si, sj]
        R_t = tb_R[si, sj]
        D_t = tb_D[si, sj]
        r_ij = r_arr[p]
        rijx = rij_arr[p, 0]
        rijy = rij_arr[p, 1]
        rijz = rij_arr[p, 2]
        fc_ij = _te_fc(r_ij, R_t, D_t)
        dfc_ij = _te_dfc_scalar(r_ij, R_t, D_t)
        if fc_ij == 0.0 and dfc_ij == 0.0:
            continue
        fr = A_t * _nb_exp(-l1 * r_ij)
        fa = -B_t * _nb_exp(-l2 * r_ij)
        b_avg = 0.5 * (bij_ij[p] + bij_ji[p])

        energy += fc_ij * (fr + b_avg * fa)
        dEdr = (dfc_ij * (fr + b_avg * fa)
                + fc_ij * (-l1 * fr + b_avg * (-l2 * fa)))
        cx = dEdr / r_ij
        for d in range(3):
            fv = cx * rij_arr[p, d]
            forces[ii, d] += fv
            forces[jj, d] += -fv
        for di in range(3):
            for dj in range(3):
                stress[di, dj] -= cx * rij_arr[p, di] * rij_arr[p, dj]

        dbi = dbij_ij[p]
        dbj = dbij_ji[p]

        if abs(dbi) > 1e-30:
            for nb in range(nb_start[ii], nb_start[ii + 1]):
                kk = nb_idx[nb]
                if kk == jj:
                    continue
                sk = type_ids[kk]
                if not tb_valid[si, sk] or not tri_valid[si, sj, sk]:
                    continue
                R_ik = tb_R[si, sk]
                D_ik = tb_D[si, sk]
                rikx = nb_rij[nb, 0]
                riky = nb_rij[nb, 1]
                rikz = nb_rij[nb, 2]
                r_ik = nb_r[nb]
                fc_ik = _te_fc(r_ik, R_ik, D_ik)
                if fc_ik == 0.0:
                    continue
                c_t = tri_c[si, sj, sk]
                d_t = tri_d[si, sj, sk]
                h_t = tri_h[si, sj, sk]
                l3 = tri_l3[si, sj, sk]
                m_t = tri_m[si, sj, sk]
                gam = tri_gam[si, sj, sk]
                coeff = 0.5 * fc_ij * fa * dbi

                dzj0 = (_te_zterm_scalar_multi(rijx + hh, rijy, rijz, rikx, riky, rikz, R_ik, D_ik, c_t, d_t, h_t, l3, m_t, gam)
                        - _te_zterm_scalar_multi(rijx - hh, rijy, rijz, rikx, riky, rikz, R_ik, D_ik, c_t, d_t, h_t, l3, m_t, gam)) / (2 * hh)
                dzj1 = (_te_zterm_scalar_multi(rijx, rijy + hh, rijz, rikx, riky, rikz, R_ik, D_ik, c_t, d_t, h_t, l3, m_t, gam)
                        - _te_zterm_scalar_multi(rijx, rijy - hh, rijz, rikx, riky, rikz, R_ik, D_ik, c_t, d_t, h_t, l3, m_t, gam)) / (2 * hh)
                dzj2 = (_te_zterm_scalar_multi(rijx, rijy, rijz + hh, rikx, riky, rikz, R_ik, D_ik, c_t, d_t, h_t, l3, m_t, gam)
                        - _te_zterm_scalar_multi(rijx, rijy, rijz - hh, rikx, riky, rikz, R_ik, D_ik, c_t, d_t, h_t, l3, m_t, gam)) / (2 * hh)
                dzk0 = (_te_zterm_scalar_multi(rijx, rijy, rijz, rikx + hh, riky, rikz, R_ik, D_ik, c_t, d_t, h_t, l3, m_t, gam)
                        - _te_zterm_scalar_multi(rijx, rijy, rijz, rikx - hh, riky, rikz, R_ik, D_ik, c_t, d_t, h_t, l3, m_t, gam)) / (2 * hh)
                dzk1 = (_te_zterm_scalar_multi(rijx, rijy, rijz, rikx, riky + hh, rikz, R_ik, D_ik, c_t, d_t, h_t, l3, m_t, gam)
                        - _te_zterm_scalar_multi(rijx, rijy, rijz, rikx, riky - hh, rikz, R_ik, D_ik, c_t, d_t, h_t, l3, m_t, gam)) / (2 * hh)
                dzk2 = (_te_zterm_scalar_multi(rijx, rijy, rijz, rikx, riky, rikz + hh, R_ik, D_ik, c_t, d_t, h_t, l3, m_t, gam)
                        - _te_zterm_scalar_multi(rijx, rijy, rijz, rikx, riky, rikz - hh, R_ik, D_ik, c_t, d_t, h_t, l3, m_t, gam)) / (2 * hh)

                Fj0 = -coeff * dzj0
                Fj1 = -coeff * dzj1
                Fj2 = -coeff * dzj2
                Fk0 = -coeff * dzk0
                Fk1 = -coeff * dzk1
                Fk2 = -coeff * dzk2
                Fi0 = coeff * (dzj0 + dzk0)
                Fi1 = coeff * (dzj1 + dzk1)
                Fi2 = coeff * (dzj2 + dzk2)

                forces[ii, 0] += Fi0
                forces[ii, 1] += Fi1
                forces[ii, 2] += Fi2
                forces[jj, 0] += Fj0
                forces[jj, 1] += Fj1
                forces[jj, 2] += Fj2
                forces[kk, 0] += Fk0
                forces[kk, 1] += Fk1
                forces[kk, 2] += Fk2

                stress[0, 0] += Fj0 * rijx + Fk0 * rikx
                stress[0, 1] += Fj0 * rijy + Fk0 * riky
                stress[0, 2] += Fj0 * rijz + Fk0 * rikz
                stress[1, 0] += Fj1 * rijx + Fk1 * rikx
                stress[1, 1] += Fj1 * rijy + Fk1 * riky
                stress[1, 2] += Fj1 * rijz + Fk1 * rikz
                stress[2, 0] += Fj2 * rijx + Fk2 * rikx
                stress[2, 1] += Fj2 * rijy + Fk2 * riky
                stress[2, 2] += Fj2 * rijz + Fk2 * rikz

        if abs(dbj) > 1e-30:
            rjix = -rijx
            rjiy = -rijy
            rjiz = -rijz
            for nb in range(nb_start[jj], nb_start[jj + 1]):
                kk = nb_idx[nb]
                if kk == ii:
                    continue
                sk = type_ids[kk]
                if not tb_valid[sj, sk] or not tri_valid[sj, si, sk]:
                    continue
                R_jk = tb_R[sj, sk]
                D_jk = tb_D[sj, sk]
                rjkx = nb_rij[nb, 0]
                rjky = nb_rij[nb, 1]
                rjkz = nb_rij[nb, 2]
                r_jk = nb_r[nb]
                fc_jk = _te_fc(r_jk, R_jk, D_jk)
                if fc_jk == 0.0:
                    continue
                c_t = tri_c[sj, si, sk]
                d_t = tri_d[sj, si, sk]
                h_t = tri_h[sj, si, sk]
                l3 = tri_l3[sj, si, sk]
                m_t = tri_m[sj, si, sk]
                gam = tri_gam[sj, si, sk]
                coeff = 0.5 * fc_ij * fa * dbj

                dzi0 = (_te_zterm_scalar_multi(rjix + hh, rjiy, rjiz, rjkx, rjky, rjkz, R_jk, D_jk, c_t, d_t, h_t, l3, m_t, gam)
                        - _te_zterm_scalar_multi(rjix - hh, rjiy, rjiz, rjkx, rjky, rjkz, R_jk, D_jk, c_t, d_t, h_t, l3, m_t, gam)) / (2 * hh)
                dzi1 = (_te_zterm_scalar_multi(rjix, rjiy + hh, rjiz, rjkx, rjky, rjkz, R_jk, D_jk, c_t, d_t, h_t, l3, m_t, gam)
                        - _te_zterm_scalar_multi(rjix, rjiy - hh, rjiz, rjkx, rjky, rjkz, R_jk, D_jk, c_t, d_t, h_t, l3, m_t, gam)) / (2 * hh)
                dzi2 = (_te_zterm_scalar_multi(rjix, rjiy, rjiz + hh, rjkx, rjky, rjkz, R_jk, D_jk, c_t, d_t, h_t, l3, m_t, gam)
                        - _te_zterm_scalar_multi(rjix, rjiy, rjiz - hh, rjkx, rjky, rjkz, R_jk, D_jk, c_t, d_t, h_t, l3, m_t, gam)) / (2 * hh)
                dzk0 = (_te_zterm_scalar_multi(rjix, rjiy, rjiz, rjkx + hh, rjky, rjkz, R_jk, D_jk, c_t, d_t, h_t, l3, m_t, gam)
                        - _te_zterm_scalar_multi(rjix, rjiy, rjiz, rjkx - hh, rjky, rjkz, R_jk, D_jk, c_t, d_t, h_t, l3, m_t, gam)) / (2 * hh)
                dzk1 = (_te_zterm_scalar_multi(rjix, rjiy, rjiz, rjkx, rjky + hh, rjkz, R_jk, D_jk, c_t, d_t, h_t, l3, m_t, gam)
                        - _te_zterm_scalar_multi(rjix, rjiy, rjiz, rjkx, rjky - hh, rjkz, R_jk, D_jk, c_t, d_t, h_t, l3, m_t, gam)) / (2 * hh)
                dzk2 = (_te_zterm_scalar_multi(rjix, rjiy, rjiz, rjkx, rjky, rjkz + hh, R_jk, D_jk, c_t, d_t, h_t, l3, m_t, gam)
                        - _te_zterm_scalar_multi(rjix, rjiy, rjiz, rjkx, rjky, rjkz - hh, R_jk, D_jk, c_t, d_t, h_t, l3, m_t, gam)) / (2 * hh)

                Fi0 = -coeff * dzi0
                Fi1 = -coeff * dzi1
                Fi2 = -coeff * dzi2
                Fk0 = -coeff * dzk0
                Fk1 = -coeff * dzk1
                Fk2 = -coeff * dzk2
                Fj0 = coeff * (dzi0 + dzk0)
                Fj1 = coeff * (dzi1 + dzk1)
                Fj2 = coeff * (dzi2 + dzk2)

                forces[jj, 0] += Fj0
                forces[jj, 1] += Fj1
                forces[jj, 2] += Fj2
                forces[ii, 0] += Fi0
                forces[ii, 1] += Fi1
                forces[ii, 2] += Fi2
                forces[kk, 0] += Fk0
                forces[kk, 1] += Fk1
                forces[kk, 2] += Fk2

                stress[0, 0] += Fi0 * rjix + Fk0 * rjkx
                stress[0, 1] += Fi0 * rjiy + Fk0 * rjky
                stress[0, 2] += Fi0 * rjiz + Fk0 * rjkz
                stress[1, 0] += Fi1 * rjix + Fk1 * rjkx
                stress[1, 1] += Fi1 * rjiy + Fk1 * rjky
                stress[1, 2] += Fi1 * rjiz + Fk1 * rjkz
                stress[2, 0] += Fi2 * rjix + Fk2 * rjkx
                stress[2, 1] += Fi2 * rjiy + Fk2 * rjky
                stress[2, 2] += Fi2 * rjiz + Fk2 * rjkz
    return energy


if _HAS_NUMBA:
    _te_zeta_jit = _numba_par(_te_zeta_numpy_par)
    _te_3body_force_jit = _numba_par(_te_3body_force_numpy_par)

    _te_multi_zeta_jit = _numba_njit(_te_multi_zeta_numpy)
    _te_multi_pass3_jit = _numba_njit(_te_multi_pass3_numpy)
else:
    _te_zeta_jit = _te_zeta_numpy
    _te_3body_force_jit = _te_3body_force_numpy

    _te_multi_zeta_jit = _te_multi_zeta_numpy
    _te_multi_pass3_jit = _te_multi_pass3_numpy


class TersoffPotential:
    """
    Tersoff bond-order potential for covalent systems (multi-species).

    Reference: Tersoff, Phys. Rev. Lett. 56, 632 (1986);
               Phys. Rev. B 37, 6991 (1988); Phys. Rev. B 38, 9902 (1988).

    Formulation
    -----------
    E = sum_{i<j} fc(r_ij) [ A_ij exp(-lambda1_ij r_ij)
                              - b_ij B_ij exp(-lambda2_ij r_ij) ]

    b_ij = (1 + beta_i^n_i zeta_ij^n_i)^{-1/(2 n_i)}

    zeta_ij = sum_{k!=i,j} gamma_ijk fc(r_ik) g_ijk(theta_ijk)
              exp(lambda3_ijk^m_ijk (r_ij - r_ik)^m_ijk)

    g_ijk(theta) = 1 + c_ijk^2/d_ijk^2 - c_ijk^2/[d_ijk^2 + (h_ijk - costheta)^2]

    fc(r; R_ij, D_ij) : smooth cut-off.

    Parameter format (multi-species, LAMMPS Tersoff-file convention)
    -------------------------------------------------------------
    params : dict keyed by ordered element triples (i, j, k) -> dict with:
        A, B, lambda1, lambda2, R, D  : two-body / cutoff parameters.
            Only the DIAGONAL entry (i, j, j) is read for the i-j pair
            term; the caller must supply symmetric values in both
            (i, j, j) and (j, i, i) (this matches the LAMMPS .tersoff
            file format exactly).
        beta, n                       : bond-order parameters, also read
            from the diagonal entry (i, j, j).
        lambda3, c, d, h              : angular (three-body) parameters,
            read from the FULL triple (i, j, k).
        m       : three-body exponent (optional, default 3).
        gamma   : three-body prefactor (optional, default 1.0).

    Any (i, j, k) combination absent from `params` is treated as
    NON-INTERACTING (the pair or angular term is simply skipped).

    Backward-compatible legacy format
    ----------------------------------
    A FLAT scalar dict (keys 'A','B','lambda1','lambda2','lambda3','beta',
    'n','c','d','h','R','D' at the top level, as used for single-element
    systems like Si) is still accepted.

    Parameters
    ----------
    params  : dict (see above).
    species : list of species symbols.  Required for the multi-species
              (nested-dict) parameter format; optional (defaults to a
              single dummy species) for the legacy flat format.

    Notes on performance
    --------------------
    Both the legacy single-species path and the general multi-species
    path have a JIT-compiled fast path (``use_numba=True`` or the
    ``NUMBA_ENABLE`` environment variable); the multi-species one
    resolves its dict-keyed parameter tables into dense, per-species-id
    NumPy arrays once per ``compute()`` call before dispatching (see
    ``_compute_multi_species``'s own docstring), rather than JIT-
    compiling the dict lookups themselves. Unlike the legacy path's
    JIT kernels, the multi-species ones are not additionally
    multi-threaded (``numba.prange``) -- a smaller-scope choice, see
    ``_te_multi_pass3_numpy``'s module-level comment.
    """

    def __init__(self,
                 params: dict,
                 species: Optional[list] = None):
        self.species = list(species) if species else None

        # Detect legacy flat single-species format (top-level scalar keys).
        is_legacy_flat = ('A' in params and 'lambda1' in params
                          and not any(isinstance(k, tuple) for k in params))

        if is_legacy_flat:
            sp0 = (self.species[0] if self.species else '__X__')
            self.species = [sp0]
            self._multi = False
            self._flat = dict(params)
            self.R = float(params.get('R', 2.85))
            self.D = float(params.get('D', 0.15))
            self.r_cut = self.R + self.D
            # Also build the nested-triple form so the generic (slow)
            # path works identically if ever invoked (e.g. for testing).
            self.p = {(sp0, sp0, sp0): dict(params)}
        else:
            if not self.species:
                raise ValueError(
                    "TersoffPotential: 'species' is required when using "
                    "the multi-species (nested-triple) parameter format.")
            self._multi = True
            self.p = params
            # r_cut = max over all supplied two-body (i,j,j) cutoffs
            rc_max = 0.0
            for (i, j, k), v in params.items():
                if j == k and 'R' in v and 'D' in v:
                    rc_max = max(rc_max, float(v['R']) + float(v['D']))
            self.r_cut = rc_max if rc_max > 0 else 3.0
            self.R = None
            self.D = None

    # ------------------------------------------------------------------
    def _two_body(self, si: str, sj: str) -> Optional[dict]:
        """Return two-body/bond-order params for the (i,j) pair, or None."""
        return self.p.get((si, sj, sj))

    def _three_body(self, si: str, sj: str, sk: str) -> Optional[dict]:
        """Return angular params for the (i; j,k) triple, or None."""
        return self.p.get((si, sj, sk))

    @staticmethod
    def _zeta_grad(rij_v, rik_v, gam, R, D, c_t, d_t, h_t, l3, m,
                    fc_func):
        """
        Gradient of a single zeta angular/exponential term with respect
        to the positions of atoms j and k (position of i follows from
        translation invariance: dz/dpos_i = -(dz/dpos_j + dz/dpos_k)).

        Computed via central finite differences of the SCALAR zeta term
        directly with respect to the components of the relative vectors
        rij_v, rik_v (rather than via an error-prone hand-derived
        analytic chain-rule expression through r_ij, r_ik, cos(theta)).

        Parameters
        ----------
        fc_func : callable(r) -> (fc, dfc) for the i-k pair cutoff.
        """
        def _zterm(rij, rik):
            r_ij = float(np.linalg.norm(rij))
            r_ik = float(np.linalg.norm(rik))
            fc_ik, _ = fc_func(r_ik)
            if fc_ik == 0.0:
                return 0.0
            cos_t = float(np.dot(rij, rik)) / (r_ij * r_ik)
            g_t = 1.0 + c_t**2 / d_t**2 - c_t**2 / (d_t**2 + (h_t - cos_t)**2)
            return gam * fc_ik * g_t * math.exp(l3**m * (r_ij - r_ik)**m)

        h = 1e-6
        dz_dposj = np.zeros(3)
        dz_dposk = np.zeros(3)
        for d_ in range(3):
            rp = rij_v.copy()
            rp[d_] += h
            rm = rij_v.copy()
            rm[d_] -= h
            dz_dposj[d_] = (_zterm(rp, rik_v) - _zterm(rm, rik_v)) / (2 * h)
        for d_ in range(3):
            rp = rik_v.copy()
            rp[d_] += h
            rm = rik_v.copy()
            rm[d_] -= h
            dz_dposk[d_] = (_zterm(rij_v, rp) - _zterm(rij_v, rm)) / (2 * h)
        return dz_dposj, dz_dposk

    # ------------------------------------------------------------------
    def compute(self,
                lattice: np.ndarray,
                positions: np.ndarray,
                atom_types: list,
                neigh_list: Optional[NeighbourList] = None,
                use_numba: bool = False,
                ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute Tersoff energy, forces, and virial stress.

        Dispatches to the legacy JIT-accelerated single-species kernel
        when exactly one species is present and the flat parameter
        format was supplied; otherwise uses the general multi-species
        implementation (see class docstring), which also honours
        ``use_numba``/``NUMBA_ENABLE`` as of Phase 5.
        """
        if not self._multi:
            return self._compute_single_species(
                lattice, positions, atom_types, neigh_list, use_numba)
        return self._compute_multi_species(
            lattice, positions, atom_types, neigh_list, use_numba)

    # ------------------------------------------------------------------
    #  Legacy single-species path (JIT-accelerated, unchanged behaviour)
    # ------------------------------------------------------------------
    def _compute_single_species(self,
                                 lattice, positions, atom_types, neigh_list, use_numba):
        N = len(positions)
        volume = float(np.abs(np.linalg.det(lattice)))
        inv_lat = np.linalg.inv(lattice)

        p_ = self._flat
        A_t = float(p_['A'])
        B_t = float(p_['B'])
        l1 = float(p_['lambda1'])
        l2 = float(p_['lambda2'])
        l3 = float(p_['lambda3'])
        bet = float(p_['beta'])
        n_t = float(p_['n'])
        c_t = float(p_['c'])
        d_t = float(p_['d'])
        h_t = float(p_['h'])
        R_t = float(p_['R'])
        D_t = float(p_['D'])
        rc = R_t + D_t

        # Vectorised MIC rij
        if neigh_list is not None:
            # Candidates only: get_pairs_arrays' build-time ``<= r_cut``
            # mask permanently dropped skin-resident pairs that later
            # migrated inside r_cut.  This routine already recomputes its
            # own distances below, so it needs the unmasked candidate set.
            # See NeighbourList.get_pairs_arrays' bug-fix note.
            idx_i_raw, idx_j_raw = neigh_list.get_candidate_pairs()
        else:
            ii_all, jj_all = np.triu_indices(N, k=1)
            idx_i_raw = ii_all.astype(np.int32)
            idx_j_raw = jj_all.astype(np.int32)

        if len(idx_i_raw) > 0:
            dr = positions[idx_j_raw] - positions[idx_i_raw]
            s = dr @ inv_lat
            s -= np.round(s)
            rij_all = s @ lattice
            r_all = np.linalg.norm(rij_all, axis=1)
            mask = (r_all < rc) & (r_all > 1e-12)
            idx_i = idx_i_raw[mask].astype(np.int32)
            idx_j = idx_j_raw[mask].astype(np.int32)
            rij_arr = np.ascontiguousarray(rij_all[mask])
            r_arr = np.ascontiguousarray(r_all[mask])
        else:
            idx_i = np.empty(0, np.int32)
            idx_j = np.empty(0, np.int32)
            rij_arr = np.empty((0, 3))
            r_arr = np.empty(0)

        if len(idx_i) == 0:
            return 0.0, np.zeros((N, 3)), np.zeros((3, 3))

        # CSR neighbour structure (vectorised)
        src_all = np.concatenate([idx_i, idx_j])
        dst_all = np.concatenate([idx_j, idx_i])
        rij_dir = np.concatenate([rij_arr, -rij_arr])
        r_dir = np.concatenate([r_arr, r_arr])
        order = np.argsort(src_all, kind='stable')
        nb_cnt = np.bincount(src_all[order], minlength=N).astype(np.int32)
        nb_start = np.zeros(N + 1, dtype=np.int32)
        nb_start[1:] = np.cumsum(nb_cnt)
        nb_idx = dst_all[order].astype(np.int32)
        nb_rij = np.ascontiguousarray(rij_dir[order])
        nb_r = np.ascontiguousarray(r_dir[order])

        def _fc(r_):
            return np.where(r_ < R_t - D_t, 1.0, np.where(r_ > R_t + D_t, 0.0,
                             0.5 - 0.5 * np.sin(0.5 * np.pi * (r_ - R_t) / D_t)))

        def _dfc(r_):
            return np.where(r_ < R_t - D_t, 0.0, np.where(r_ > R_t + D_t, 0.0,
                             -0.25 * np.pi / D_t * np.cos(0.5 * np.pi * (r_ - R_t) / D_t)))

        def _fc1(r):
            return float(_fc(np.array([r]))[0])

        # Pre-compute zeta_ij / zeta_ji -- dispatch to JIT or NumPy
        from ..utils.jit import _NUMBA_ENV
        numba_on = (use_numba or _NUMBA_ENV) and _HAS_NUMBA
        P = len(idx_i)
        zeta_ij = np.zeros(P)
        zeta_ji = np.zeros(P)

        if numba_on:
            # Numba parallel kernel (NUMBA_NUM_THREADS threads)
            _te_zeta_jit(
                P, idx_i, idx_j, rij_arr, r_arr,
                nb_start, nb_idx, nb_rij, nb_r,
                zeta_ij, zeta_ji,
                l3, c_t, d_t, h_t, R_t, D_t)
        else:
            for p in range(P):
                ii = int(idx_i[p])
                jj = int(idx_j[p])
                r_ij = float(r_arr[p])
                rij_v = rij_arr[p]
                for nb in range(int(nb_start[ii]), int(nb_start[ii + 1])):
                    kk = int(nb_idx[nb])
                    if kk == jj:
                        continue
                    r_ik = float(nb_r[nb])
                    fc_ik = _fc1(r_ik)
                    if fc_ik == 0.0:
                        continue
                    rik_v = nb_rij[nb]
                    cos_t = float(np.dot(rij_v, rik_v)) / (r_ij * r_ik)
                    g_t = 1.0 + c_t**2 / d_t**2 - c_t**2 / (d_t**2 + (h_t - cos_t)**2)
                    zeta_ij[p] += fc_ik * g_t * np.exp(l3**3 * (r_ij - r_ik)**3)
                rji_v = -rij_v
                for nb in range(int(nb_start[jj]), int(nb_start[jj + 1])):
                    kk = int(nb_idx[nb])
                    if kk == ii:
                        continue
                    r_jk = float(nb_r[nb])
                    fc_jk = _fc1(r_jk)
                    if fc_jk == 0.0:
                        continue
                    rjk_v = nb_rij[nb]
                    cos_t = float(np.dot(rji_v, rjk_v)) / (r_arr[p] * r_jk)
                    g_t = 1.0 + c_t**2 / d_t**2 - c_t**2 / (d_t**2 + (h_t - cos_t)**2)
                    zeta_ji[p] += fc_jk * g_t * np.exp(l3**3 * (r_arr[p] - r_jk)**3)

        # NOTE: zeta_ij / zeta_ji are computed above via a loop over ALL
        # P pairs (not domain-split across MPI ranks), so every rank
        # already holds the complete, correct zeta arrays. Do NOT
        # Allreduce them: Allreduce(SUM) of N ranks each holding an
        # IDENTICAL copy of the same array multiplies its values by N,
        # not 1 -- this inflated bond-order zeta values in an earlier
        # version, suppressing the attractive term and yielding a
        # dramatically wrong (far less negative) total energy that grew
        # increasingly wrong with MORE MPI ranks (confirmed by direct
        # testing under real mpirun: n=2: -1071.9 eV, n=4: -894.8 eV,
        # n=8: -633.6 eV, vs. the correct single-process value of
        # -1185.2 eV for the same 512-atom Si Tersoff system). Only the
        # genuinely domain-split (p_start:p_end) 2-body and 3-body sums
        # need cross-rank reduction (handled at the end of this method).
        def _bij(z):
            return np.where(z > 0, (1.0 + (bet * z)**n_t)**(-0.5 / n_t), np.ones_like(z))

        def _dbij(z, b):
            safe = np.where(z > 0, z, np.ones_like(z))
            bz = (bet * safe)**n_t
            return np.where(z > 0, -0.5 * b * bz / (safe * (1.0 + bz)), np.zeros_like(z))

        bij_ij = _bij(zeta_ij)
        bij_ji = _bij(zeta_ji)
        dbij_ij = _dbij(zeta_ij, bij_ij)
        dbij_ji = _dbij(zeta_ji, bij_ji)

        # Vectorised 2-body energy+forces (MPI pair-split)
        p_start, p_end = _mpi_domain_split(P)
        forces = np.zeros((N, 3))
        stress = np.zeros((3, 3))
        energy = 0.0

        r_loc = r_arr[p_start:p_end]
        rij_loc = rij_arr[p_start:p_end]
        ii_loc = idx_i[p_start:p_end]
        jj_loc = idx_j[p_start:p_end]
        bij_avg = 0.5 * (bij_ij[p_start:p_end] + bij_ji[p_start:p_end])
        fc_l = _fc(r_loc)
        dfc_l = _dfc(r_loc)
        fr_l = A_t * np.exp(-l1 * r_loc)
        fa_l = -B_t * np.exp(-l2 * r_loc)

        energy += float((fc_l * (fr_l + bij_avg * fa_l)).sum())
        dEdr = (dfc_l * (fr_l + bij_avg * fa_l) + fc_l * (-l1 * fr_l + bij_avg * (-l2 * fa_l)))
        fij = (dEdr / r_loc)[:, None] * rij_loc
        np.add.at(forces, ii_loc, fij)
        np.add.at(forces, jj_loc, -fij)
        stress -= np.einsum('ki,kj->ij', fij, rij_loc)

        # 3-body force corrections from dbij. See _te_3body_force_numpy's
        # docstring for why the position gradient of the zeta term is
        # computed via central finite differences (of the scalar zeta
        # term w.r.t. the relative-vector components) rather than a
        # hand-derived analytic chain-rule expression, and why this
        # correction is dispatched to the JIT-parallel kernel whenever
        # numba_on is True (it is typically the dominant cost of a
        # Tersoff force evaluation).
        if numba_on:
            fc_full = np.zeros(P)
            fa_full = np.zeros(P)
            fc_full[p_start:p_end] = fc_l
            fa_full[p_start:p_end] = fa_l
            _te_3body_force_jit(
                p_start, p_end,
                idx_i, idx_j, rij_arr, r_arr,
                fc_full, fa_full, dbij_ij, dbij_ji,
                nb_start, nb_idx, nb_rij, nb_r,
                forces, stress,
                R_t, D_t, c_t, d_t, h_t, l3, 3.0)
        else:
            def _fc_pair(r_):
                return _fc1(r_), 0.0

            for p_loc in range(p_end - p_start):
                p = p_start + p_loc
                ii = int(idx_i[p])
                jj = int(idx_j[p])
                r_ij = float(r_arr[p])
                rij_v = rij_arr[p]
                fc_ij = float(fc_l[p_loc])
                fa_ij = float(fa_l[p_loc])
                dbi = float(dbij_ij[p])
                dbj = float(dbij_ji[p])

                if abs(dbi) > 1e-30:
                    for nb in range(int(nb_start[ii]), int(nb_start[ii + 1])):
                        kk = int(nb_idx[nb])
                        if kk == jj:
                            continue
                        r_ik = float(nb_r[nb])
                        rik_v = nb_rij[nb]
                        fc_ik = _fc1(r_ik)
                        if fc_ik == 0.0:
                            continue
                        coeff = 0.5 * fc_ij * fa_ij * dbi
                        dz_dposj, dz_dposk = self._zeta_grad(
                            rij_v, rik_v, 1.0, R_t, D_t, c_t, d_t, h_t, l3, 3.0,
                            _fc_pair)
                        Fi = coeff * (dz_dposj + dz_dposk)
                        Fj = -coeff * dz_dposj
                        Fk = -coeff * dz_dposk
                        forces[ii] += Fi
                        forces[jj] += Fj
                        forces[kk] += Fk
                        # Sign: see the bug-fix note in stillinger_weber.py's
                        # _sw_3body_numpy docstring -- must be +=, not -=.
                        stress += np.outer(Fj, rij_v) + np.outer(Fk, rik_v)

                if abs(dbj) > 1e-30:
                    rji_v = -rij_v
                    for nb in range(int(nb_start[jj]), int(nb_start[jj + 1])):
                        kk = int(nb_idx[nb])
                        if kk == ii:
                            continue
                        r_jk = float(nb_r[nb])
                        rjk_v = nb_rij[nb]
                        fc_jk = _fc1(r_jk)
                        if fc_jk == 0.0:
                            continue
                        coeff = 0.5 * fc_ij * fa_ij * dbj
                        # Derivation roles: "i"=j(real), "j"=i(real), "k"=k(real)
                        dz_dposi, dz_dposk = self._zeta_grad(
                            rji_v, rjk_v, 1.0, R_t, D_t, c_t, d_t, h_t, l3, 3.0,
                            _fc_pair)
                        Fj = coeff * (dz_dposi + dz_dposk)
                        Fi = -coeff * dz_dposi
                        Fk = -coeff * dz_dposk
                        forces[jj] += Fj
                        forces[ii] += Fi
                        forces[kk] += Fk
                        stress += np.outer(Fi, rji_v) + np.outer(Fk, rjk_v)

        # Bug-fix note (found in code-check, same class as F16): this used
        # to be a raw `_mpi_comm.allreduce(energy, op=_MPI.SUM)` call, which
        # bypasses `local_only()` (it never checks `_mpi_is_active()`, so an
        # outer item-splitting caller such as
        # `fitting/loss.py::generic_loss_function` or `solvers/neb.py`
        # wrapped in `with local_only():` could not suppress it) -- while
        # `forces`/`stress` below already went through the
        # `local_only()`-aware `_mpi_allreduce_sum()`. Under real multi-rank
        # `mpirun`, this silently summed each rank's own-structure/image
        # energy together with every OTHER rank's DIFFERENT structure/image
        # energy. Fixed by routing through the same
        # `_mpi_allreduce_sum_scalar()` helper `pair_potentials.py` and
        # `coulomb/ewald.py` already use for their own scalar energy.
        if _HAS_MPI:
            energy = _mpi_allreduce_sum_scalar(energy)
            forces = _mpi_allreduce_sum(forces)
            stress = _mpi_allreduce_sum(stress)
        stress /= volume
        return energy, forces, stress

    # ------------------------------------------------------------------
    #  General multi-species path (NumPy, per-pair/per-triple lookup)
    # ------------------------------------------------------------------
    def _compute_multi_species(self, lattice, positions, atom_types,
                               neigh_list, use_numba: bool = False):
        """
        Bug-fix note (Phase 5, numba for the multi-species path): mirrors
        StillingerWeberPotential._compute_multi_species's own Phase 5
        note -- this path previously had no JIT option at all (unlike
        the legacy single-species path). See _te_multi_zeta_numpy's and
        _te_multi_pass3_numpy's module-level comments for how the
        dict-keyed parameter lookups are resolved into dense,
        per-species-id tables once here before dispatching to them.
        """
        N = len(positions)
        volume = float(np.abs(np.linalg.det(lattice)))
        inv_lat = np.linalg.inv(lattice)
        sp = list(atom_types)

        rc_global = self.r_cut
        if rc_global <= 0.0:
            return 0.0, np.zeros((N, 3)), np.zeros((3, 3))

        if neigh_list is not None:
            # Candidates only: get_pairs_arrays' build-time ``<= r_cut``
            # mask permanently dropped skin-resident pairs that later
            # migrated inside r_cut.  This routine already recomputes its
            # own distances below, so it needs the unmasked candidate set.
            # See NeighbourList.get_pairs_arrays' bug-fix note.
            idx_i_raw, idx_j_raw = neigh_list.get_candidate_pairs()
        else:
            ii_all, jj_all = np.triu_indices(N, k=1)
            idx_i_raw = ii_all.astype(np.int32)
            idx_j_raw = jj_all.astype(np.int32)

        if len(idx_i_raw) == 0:
            return 0.0, np.zeros((N, 3)), np.zeros((3, 3))

        dr = positions[idx_j_raw] - positions[idx_i_raw]
        s = dr @ inv_lat
        s -= np.round(s)
        rij_all = s @ lattice
        r_all = np.linalg.norm(rij_all, axis=1)
        mask = (r_all < rc_global) & (r_all > 1e-12)
        idx_i = idx_i_raw[mask].astype(np.int64)
        idx_j = idx_j_raw[mask].astype(np.int64)
        rij_arr = np.ascontiguousarray(rij_all[mask])
        r_arr = np.ascontiguousarray(r_all[mask])
        P = len(idx_i)
        if P == 0:
            return 0.0, np.zeros((N, 3)), np.zeros((3, 3))

        # CSR neighbour structure (all pairs within rc_global, species-agnostic;
        # the correct per-species (R,D) cutoff is re-applied when each
        # neighbour k is actually used inside the angular sums below).
        src_all = np.concatenate([idx_i, idx_j])
        dst_all = np.concatenate([idx_j, idx_i])
        rij_dir = np.concatenate([rij_arr, -rij_arr])
        r_dir = np.concatenate([r_arr, r_arr])
        order = np.argsort(src_all, kind='stable')
        nb_cnt = np.bincount(src_all[order], minlength=N).astype(np.int64)
        nb_start = np.zeros(N + 1, dtype=np.int64)
        nb_start[1:] = np.cumsum(nb_cnt)
        nb_idx = dst_all[order]
        nb_rij = np.ascontiguousarray(rij_dir[order])
        nb_r = np.ascontiguousarray(r_dir[order])

        forces = np.zeros((N, 3))
        stress = np.zeros((3, 3))
        energy = 0.0

        p_start, p_end = _mpi_domain_split(P)

        from ..utils.jit import _NUMBA_ENV
        numba_on = (use_numba or _NUMBA_ENV) and _HAS_NUMBA

        if numba_on:
            # Resolve every (species_i, species_j[, species_k]) combination
            # into dense integer-indexed tables ONCE (see
            # _te_multi_zeta_numpy's/_te_multi_pass3_numpy's module-level
            # comment); the JIT kernels then only ever index these with
            # plain integers.
            n_sp = len(self.species)
            sp_index = {s: idx for idx, s in enumerate(self.species)}
            type_ids = np.array([sp_index[t] for t in sp], dtype=np.int64)

            tb_valid = np.zeros((n_sp, n_sp), dtype=np.bool_)
            tb_A = np.zeros((n_sp, n_sp))
            tb_B = np.zeros((n_sp, n_sp))
            tb_l1 = np.zeros((n_sp, n_sp))
            tb_l2 = np.zeros((n_sp, n_sp))
            tb_R = np.ones((n_sp, n_sp))
            tb_D = np.ones((n_sp, n_sp))
            tb_beta = np.zeros((n_sp, n_sp))
            tb_n = np.ones((n_sp, n_sp))
            for i_sp, si in enumerate(self.species):
                for j_sp, sj in enumerate(self.species):
                    tb = self._two_body(si, sj)
                    if tb is None:
                        continue
                    tb_valid[i_sp, j_sp] = True
                    tb_A[i_sp, j_sp] = float(tb.get('A', 0.0))
                    tb_B[i_sp, j_sp] = float(tb.get('B', 0.0))
                    tb_l1[i_sp, j_sp] = float(tb.get('lambda1', 0.0))
                    tb_l2[i_sp, j_sp] = float(tb.get('lambda2', 0.0))
                    tb_R[i_sp, j_sp] = float(tb.get('R', 1.0))
                    tb_D[i_sp, j_sp] = float(tb.get('D', 1.0))
                    tb_beta[i_sp, j_sp] = float(tb.get('beta', 0.0))
                    tb_n[i_sp, j_sp] = float(tb.get('n', 1.0))

            tri_valid = np.zeros((n_sp, n_sp, n_sp), dtype=np.bool_)
            tri_c = np.zeros((n_sp, n_sp, n_sp))
            tri_d = np.ones((n_sp, n_sp, n_sp))
            tri_h = np.zeros((n_sp, n_sp, n_sp))
            tri_l3 = np.zeros((n_sp, n_sp, n_sp))
            tri_m = np.full((n_sp, n_sp, n_sp), 3.0)
            tri_gam = np.ones((n_sp, n_sp, n_sp))
            for i_sp, si in enumerate(self.species):
                for j_sp, sj in enumerate(self.species):
                    for k_sp, sk in enumerate(self.species):
                        tri = self._three_body(si, sj, sk)
                        if tri is None:
                            continue
                        tri_valid[i_sp, j_sp, k_sp] = True
                        tri_c[i_sp, j_sp, k_sp] = float(tri['c'])
                        tri_d[i_sp, j_sp, k_sp] = float(tri['d'])
                        tri_h[i_sp, j_sp, k_sp] = float(tri['h'])
                        tri_l3[i_sp, j_sp, k_sp] = float(tri['lambda3'])
                        tri_m[i_sp, j_sp, k_sp] = float(tri.get('m', 3.0))
                        tri_gam[i_sp, j_sp, k_sp] = float(tri.get('gamma', 1.0))

            # ---- Pass 1: zeta_ij / zeta_ji ------------------------------
            zeta_ij = np.zeros(P)
            zeta_ji = np.zeros(P)
            _te_multi_zeta_jit(
                P, idx_i, idx_j, rij_arr, r_arr, type_ids,
                nb_start, nb_idx, nb_rij, nb_r,
                zeta_ij, zeta_ji,
                tb_valid, tb_R, tb_D,
                tri_valid, tri_c, tri_d, tri_h, tri_l3, tri_m, tri_gam)

            # ---- Pass 2: bond order b_ij, b_ji (vectorised, O(P) only --
            # no neighbour loop, so plain NumPy fancy-indexing already
            # suffices; matches the ORIGINAL's quirky gating exactly:
            # a pair with no FORWARD (i,j) table entry gets NEITHER
            # direction's bond order updated, even if the (j,i) entry
            # exists -- see this method's own Pass 2 in the else-branch
            # below, and _te_multi_zeta_numpy's docstring for the same
            # gating pattern in Pass 1.
            si_arr = type_ids[idx_i]
            sj_arr = type_ids[idx_j]
            valid_ij = tb_valid[si_arr, sj_arr]

            bij_ij = np.ones(P)
            bij_ji = np.ones(P)
            dbij_ij = np.zeros(P)
            dbij_ji = np.zeros(P)

            beta_i_arr = tb_beta[si_arr, sj_arr]
            n_i_arr = tb_n[si_arr, sj_arr]
            mask_i = valid_ij & (zeta_ij > 0) & (beta_i_arr != 0.0)
            z = zeta_ij[mask_i]
            beta_i = beta_i_arr[mask_i]
            n_i = n_i_arr[mask_i]
            bz = (beta_i * z) ** n_i
            b = (1.0 + bz) ** (-0.5 / n_i)
            bij_ij[mask_i] = b
            dbij_ij[mask_i] = -0.5 * b * bz / (z * (1.0 + bz))

            beta_j_arr = tb_beta[sj_arr, si_arr]
            n_j_arr = tb_n[sj_arr, si_arr]
            mask_j = valid_ij & (zeta_ji > 0) & (beta_j_arr != 0.0)
            z = zeta_ji[mask_j]
            beta_j = beta_j_arr[mask_j]
            n_j = n_j_arr[mask_j]
            bz = (beta_j * z) ** n_j
            b = (1.0 + bz) ** (-0.5 / n_j)
            bij_ji[mask_j] = b
            dbij_ji[mask_j] = -0.5 * b * bz / (z * (1.0 + bz))

            # ---- Pass 3: 2-body energy/forces + 3-body correction -------
            energy += _te_multi_pass3_jit(
                p_start, p_end, idx_i, idx_j, rij_arr, r_arr, type_ids,
                bij_ij, bij_ji, dbij_ij, dbij_ji,
                nb_start, nb_idx, nb_rij, nb_r,
                forces, stress,
                tb_valid, tb_A, tb_B, tb_l1, tb_l2, tb_R, tb_D,
                tri_valid, tri_c, tri_d, tri_h, tri_l3, tri_m, tri_gam)

        else:
            def _fc1(r, R, D):
                if r < R - D:
                    return 1.0
                if r > R + D:
                    return 0.0
                return 0.5 - 0.5 * math.sin(0.5 * math.pi * (r - R) / D)

            def _dfc1(r, R, D):
                if r < R - D or r > R + D:
                    return 0.0
                return -0.25 * math.pi / D * math.cos(0.5 * math.pi * (r - R) / D)

            # ---- Pass 1: zeta_ij / zeta_ji (per-pair angular sums) --------
            zeta_ij = np.zeros(P)
            zeta_ji = np.zeros(P)

            for p in range(P):
                ii = int(idx_i[p])
                jj = int(idx_j[p])
                si, sj = sp[ii], sp[jj]
                tb_ij = self._two_body(si, sj)
                if tb_ij is None:
                    continue   # this species pair does not interact
                r_ij = float(r_arr[p])
                rij_v = rij_arr[p]

                # zeta_ij: sum over k neighbouring i (excluding j)
                for nb in range(int(nb_start[ii]), int(nb_start[ii + 1])):
                    kk = int(nb_idx[nb])
                    if kk == jj:
                        continue
                    sk = sp[kk]
                    tb_ik = self._two_body(si, sk)
                    tri = self._three_body(si, sj, sk)
                    if tb_ik is None or tri is None:
                        continue
                    r_ik = float(nb_r[nb])
                    fc_ik = _fc1(r_ik, float(tb_ik['R']), float(tb_ik['D']))
                    if fc_ik == 0.0:
                        continue
                    rik_v = nb_rij[nb]
                    cos_t = float(np.dot(rij_v, rik_v)) / (r_ij * r_ik)
                    c_t = float(tri['c'])
                    d_t = float(tri['d'])
                    h_t = float(tri['h'])
                    l3 = float(tri['lambda3'])
                    m_t = float(tri.get('m', 3.0))
                    gam = float(tri.get('gamma', 1.0))
                    g_t = 1.0 + c_t**2 / d_t**2 - c_t**2 / (d_t**2 + (h_t - cos_t)**2)
                    zeta_ij[p] += gam * fc_ik * g_t * math.exp(l3**m_t * (r_ij - r_ik)**m_t)

                # zeta_ji: sum over k neighbouring j (excluding i)
                rji_v = -rij_v
                for nb in range(int(nb_start[jj]), int(nb_start[jj + 1])):
                    kk = int(nb_idx[nb])
                    if kk == ii:
                        continue
                    sk = sp[kk]
                    tb_jk = self._two_body(sj, sk)
                    tri = self._three_body(sj, si, sk)
                    if tb_jk is None or tri is None:
                        continue
                    r_jk = float(nb_r[nb])
                    fc_jk = _fc1(r_jk, float(tb_jk['R']), float(tb_jk['D']))
                    if fc_jk == 0.0:
                        continue
                    rjk_v = nb_rij[nb]
                    cos_t = float(np.dot(rji_v, rjk_v)) / (r_ij * r_jk)
                    c_t = float(tri['c'])
                    d_t = float(tri['d'])
                    h_t = float(tri['h'])
                    l3 = float(tri['lambda3'])
                    m_t = float(tri.get('m', 3.0))
                    gam = float(tri.get('gamma', 1.0))
                    g_t = 1.0 + c_t**2 / d_t**2 - c_t**2 / (d_t**2 + (h_t - cos_t)**2)
                    zeta_ji[p] += gam * fc_jk * g_t * math.exp(l3**m_t * (r_ij - r_jk)**m_t)

            # (No Allreduce needed for zeta_ij/zeta_ji -- see the identical,
            # detailed explanation in _compute_single_species above.)

            # ---- Pass 2: bond order b_ij, two-body energy/forces -----------
            bij_ij = np.ones(P)
            bij_ji = np.ones(P)
            dbij_ij = np.zeros(P)
            dbij_ji = np.zeros(P)

            for p in range(P):
                ii = int(idx_i[p])
                jj = int(idx_j[p])
                si, sj = sp[ii], sp[jj]
                tb_ij = self._two_body(si, sj)
                if tb_ij is None:
                    continue
                beta_i = float(tb_ij.get('beta', 0.0))
                n_i = float(tb_ij.get('n', 1.0))
                z = zeta_ij[p]
                if z > 0 and beta_i != 0.0:
                    bz = (beta_i * z) ** n_i
                    bij_ij[p] = (1.0 + bz) ** (-0.5 / n_i)
                    dbij_ij[p] = -0.5 * bij_ij[p] * bz / (z * (1.0 + bz))

                tb_ji = self._two_body(sj, si)
                beta_j = float(tb_ji.get('beta', 0.0)) if tb_ji else 0.0
                n_j = float(tb_ji.get('n', 1.0)) if tb_ji else 1.0
                z = zeta_ji[p]
                if z > 0 and beta_j != 0.0:
                    bz = (beta_j * z) ** n_j
                    bij_ji[p] = (1.0 + bz) ** (-0.5 / n_j)
                    dbij_ji[p] = -0.5 * bij_ji[p] * bz / (z * (1.0 + bz))

            for p in range(p_start, p_end):
                ii = int(idx_i[p])
                jj = int(idx_j[p])
                si, sj = sp[ii], sp[jj]
                tb_ij = self._two_body(si, sj)
                if tb_ij is None:
                    continue
                A_t = float(tb_ij['A'])
                B_t = float(tb_ij['B'])
                l1 = float(tb_ij['lambda1'])
                l2 = float(tb_ij['lambda2'])
                R_t = float(tb_ij['R'])
                D_t = float(tb_ij['D'])
                r_ij = float(r_arr[p])
                rij_v = rij_arr[p]
                fc_ij = _fc1(r_ij, R_t, D_t)
                dfc_ij = _dfc1(r_ij, R_t, D_t)
                if fc_ij == 0.0 and dfc_ij == 0.0:
                    continue
                fr = A_t * math.exp(-l1 * r_ij)
                fa = -B_t * math.exp(-l2 * r_ij)
                b_avg = 0.5 * (bij_ij[p] + bij_ji[p])

                energy += fc_ij * (fr + b_avg * fa)
                dEdr = (dfc_ij * (fr + b_avg * fa)
                        + fc_ij * (-l1 * fr + b_avg * (-l2 * fa)))
                fij = (dEdr / r_ij) * rij_v
                forces[ii] += fij
                forces[jj] -= fij
                stress -= np.outer(fij, rij_v)

                # 3-body force corrections from d(bij)/d(zeta). See
                # _te_3body_force_numpy's docstring for why a naive F_j=-F_i
                # assumption (used in an earlier, buggy version) is incorrect
                # whenever the angular (cos theta) dependence is present.
                dbi = dbij_ij[p]
                dbj = dbij_ji[p]
                if abs(dbi) > 1e-30:
                    for nb in range(int(nb_start[ii]), int(nb_start[ii + 1])):
                        kk = int(nb_idx[nb])
                        if kk == jj:
                            continue
                        sk = sp[kk]
                        tb_ik = self._two_body(si, sk)
                        tri = self._three_body(si, sj, sk)
                        if tb_ik is None or tri is None:
                            continue
                        r_ik = float(nb_r[nb])
                        rik_v = nb_rij[nb]
                        R_ik = float(tb_ik['R'])
                        D_ik = float(tb_ik['D'])
                        fc_ik = _fc1(r_ik, R_ik, D_ik)
                        if fc_ik == 0.0:
                            continue
                        c_t = float(tri['c'])
                        d_t = float(tri['d'])
                        h_t = float(tri['h'])
                        l3 = float(tri['lambda3'])
                        m_t = float(tri.get('m', 3.0))
                        gam = float(tri.get('gamma', 1.0))
                        coeff = 0.5 * fc_ij * fa * dbi
                        dz_dposj, dz_dposk = self._zeta_grad(
                            rij_v, rik_v, gam, R_ik, D_ik, c_t, d_t, h_t, l3, m_t,
                            lambda r_: (_fc1(r_, R_ik, D_ik), _dfc1(r_, R_ik, D_ik)))
                        Fi = coeff * (dz_dposj + dz_dposk)
                        Fj = -coeff * dz_dposj
                        Fk = -coeff * dz_dposk
                        forces[ii] += Fi
                        forces[jj] += Fj
                        forces[kk] += Fk
                        stress += np.outer(Fj, rij_v) + np.outer(Fk, rik_v)

                if abs(dbj) > 1e-30:
                    rji_v = -rij_v
                    for nb in range(int(nb_start[jj]), int(nb_start[jj + 1])):
                        kk = int(nb_idx[nb])
                        if kk == ii:
                            continue
                        sk = sp[kk]
                        tb_jk = self._two_body(sj, sk)
                        tri = self._three_body(sj, si, sk)
                        if tb_jk is None or tri is None:
                            continue
                        r_jk = float(nb_r[nb])
                        rjk_v = nb_rij[nb]
                        R_jk = float(tb_jk['R'])
                        D_jk = float(tb_jk['D'])
                        fc_jk = _fc1(r_jk, R_jk, D_jk)
                        if fc_jk == 0.0:
                            continue
                        c_t = float(tri['c'])
                        d_t = float(tri['d'])
                        h_t = float(tri['h'])
                        l3 = float(tri['lambda3'])
                        m_t = float(tri.get('m', 3.0))
                        gam = float(tri.get('gamma', 1.0))
                        coeff = 0.5 * fc_ij * fa * dbj
                        # Derivation roles: "i"=j(real), "j"=i(real), "k"=k(real)
                        dz_dposi, dz_dposk = self._zeta_grad(
                            rji_v, rjk_v, gam, R_jk, D_jk, c_t, d_t, h_t, l3, m_t,
                            lambda r_: (_fc1(r_, R_jk, D_jk), _dfc1(r_, R_jk, D_jk)))
                        Fj = coeff * (dz_dposi + dz_dposk)
                        Fi = -coeff * dz_dposi
                        Fk = -coeff * dz_dposk
                        forces[jj] += Fj
                        forces[ii] += Fi
                        forces[kk] += Fk
                        stress += np.outer(Fi, rji_v) + np.outer(Fk, rjk_v)

        # Bug-fix note (found in code-check, same class as F16): this used
        # to be a raw `_mpi_comm.allreduce(energy, op=_MPI.SUM)` call, which
        # bypasses `local_only()` (it never checks `_mpi_is_active()`, so an
        # outer item-splitting caller such as
        # `fitting/loss.py::generic_loss_function` or `solvers/neb.py`
        # wrapped in `with local_only():` could not suppress it) -- while
        # `forces`/`stress` below already went through the
        # `local_only()`-aware `_mpi_allreduce_sum()`. Under real multi-rank
        # `mpirun`, this silently summed each rank's own-structure/image
        # energy together with every OTHER rank's DIFFERENT structure/image
        # energy. Fixed by routing through the same
        # `_mpi_allreduce_sum_scalar()` helper `pair_potentials.py` and
        # `coulomb/ewald.py` already use for their own scalar energy.
        if _HAS_MPI:
            energy = _mpi_allreduce_sum_scalar(energy)
            forces = _mpi_allreduce_sum(forces)
            stress = _mpi_allreduce_sum(stress)
        stress /= volume
        return energy, forces, stress
