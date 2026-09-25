"""Stillinger-Weber (SW) potential for covalently bonded systems (multi-species).

Reference: Stillinger & Weber, Phys. Rev. B 31, 5262 (1985).
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


# -- Stillinger-Weber: 2-body kernel --------------------------------------

def _sw_2body_numpy(idx_i, idx_j, rij_arr, r_arr, forces, stress,
                     eps, sig, A_sw, B_sw, pp, qq, a_sw):
    """
    SW 2-body energy and forces.  Returns energy; modifies forces/stress in-place.
    Numba-compatible signature (no Python objects, only arrays and scalars).
    """
    energy = 0.0
    for p in range(len(idx_i)):
        ii = idx_i[p]
        jj = idx_j[p]
        r = r_arr[p]
        xi = r / sig
        if xi >= a_sw:
            continue
        t = 1.0 / (xi - a_sw)
        ex = _nb_exp(t)
        V2 = eps * A_sw * (B_sw * xi**(-pp) - xi**(-qq)) * ex
        dV2 = eps * A_sw * ex * ((-pp * B_sw * xi**(-pp - 1) + qq * xi**(-qq - 1))
                                  + (B_sw * xi**(-pp) - xi**(-qq)) * (-t * t))
        dV2dr = dV2 / sig
        energy += V2
        cx = (dV2dr / r)
        for d in range(3):
            fv = cx * rij_arr[p, d]
            forces[ii, d] += fv
            forces[jj, d] += -fv
        for di in range(3):
            for dj in range(3):
                stress[di, dj] -= cx * rij_arr[p, di] * rij_arr[p, dj]
    return energy


def _sw_h3_scalar(rijx, rijy, rijz, rikx, riky, rikz,
                   sig, a_sw, gam, lam, eps, costh0):
    """
    Scalar SW three-body energy term h(theta_jik) for one (j, k) pair
    of neighbours of central atom i, given the two relative-displacement
    vectors r_ij = (rijx, rijy, rijz) and r_ik = (rikx, riky, rikz).

    This is a plain, Numba-nopython-compatible scalar function (no
    array slicing) used as the building block for a numerical
    (central-difference) force derivation -- see _sw_3body_numpy for
    why this replaces a previous hand-derived analytic formula that
    was found to be incorrect (verified via finite-difference
    comparison against the total energy).

    Bug-fix note (gamma): the exponent used to read ``1.0 / (x - a_sw)``,
    i.e. gamma hard-coded to 1, even though the class docstring documents
    the term as ``exp(gamma.sigma/(r - a.sigma))`` and lists 'gamma' as an
    accepted key of the legacy flat parameter format.  A flat dict's
    'gamma' was therefore read by nothing and discarded in silence.  The
    defect is invisible in a perfect diamond lattice -- every bond angle
    is exactly tetrahedral there, so ``(cos t - cos t0)^2`` vanishes and
    the whole three-body term with it -- and only shows up once the
    angles are distorted: on a 0.15 A randomly displaced 8-atom Si cell,
    gamma = 1.00 and gamma = 1.20 both gave exactly -28.92004161 eV
    before the fix, against -30.39236397 eV for the (already correct)
    multi-species path at gamma = 1.20.  Canonical Stillinger-Weber Si
    has gamma = 1.20, so the legacy path was not running SW Si.
    """
    r_ij = math.sqrt(rijx * rijx + rijy * rijy + rijz * rijz)
    r_ik = math.sqrt(rikx * rikx + riky * riky + rikz * rikz)
    xij = r_ij / sig
    xik = r_ik / sig
    if xij >= a_sw or xik >= a_sw:
        return 0.0
    exp_ij = _nb_exp(gam / (xij - a_sw))
    exp_ik = _nb_exp(gam / (xik - a_sw))
    cos_th = (rijx * rikx + rijy * riky + rijz * rikz) / (r_ij * r_ik)
    dcos = cos_th - costh0
    return lam * eps * exp_ij * exp_ik * dcos * dcos


def _sw_3body_numpy(n_start, n_end,
                     nb_start, nb_idx, nb_rij, nb_r,
                     forces, stress,
                     eps, sig, a_sw, gam, lam, costh0):
    """
    SW 3-body energy and forces.  Returns energy; modifies forces/stress
    in-place.  Iterates over central atoms in [n_start, n_end).

    Forces are obtained via central finite differences of the scalar
    h3 term (_sw_h3_scalar) with respect to the 6 components of the two
    relative-displacement vectors r_ij, r_ik, rather than a hand-derived
    analytic chain-rule expression. A previous version of this kernel
    used such an analytic formula, which was found (via a
    finite-difference check against the total energy on a displaced,
    symmetry-broken structure) to be INCORRECT -- the bug had gone
    undetected because it only manifests for a genuinely asymmetric
    local environment, and pristine-crystal validation has zero force by
    symmetry regardless of the sign/magnitude error.

    Bug-fix note (stress sign): an earlier version accumulated
    ``stress -= outer(Fj, rij) + outer(Fk, rik)`` (matching the sign used
    by the validated 2-body kernel's ``stress -= outer(fij, rij)``), but
    for a 3-body cluster this is the WRONG sign. Direct finite-difference
    validation (isolating the 3-body term alone, via A_sw=0 to silence
    the 2-body term) showed the 3-body force matched to ~1e-10 while the
    stress was the exact negative of the correct value component-by-
    component. Re-deriving the general atom-force-to-virial relationship
    by calibrating against the ALREADY-TRUSTED 2-body formula (rather
    than assuming the naive ``sigma = -(1/V) sum r_i (x) F_i`` textbook
    form, which turns out to have the opposite sign convention from what
    this codebase's 2-body kernels actually implement) gives
    ``stress = +(1/V) sum_atoms F_atom (x) r_atom``; substituting
    ``F_i = -(F_j+F_k)`` (translational invariance of h3) yields
    ``stress += outer(Fj, rij) + outer(Fk, rik)`` -- i.e. ``+=``, not
    ``-=``. The same derivation and fix applies to Tersoff's analogous
    3-body bond-order force correction (tersoff.py).
    """
    energy = 0.0
    hh = 1e-6
    for i in range(n_start, n_end):
        si = nb_start[i]
        ei = nb_start[i + 1]
        if ei - si < 2:
            continue
        for pj in range(si, ei):
            j_at = nb_idx[pj]
            r_ij = nb_r[pj]
            xij = r_ij / sig
            if xij >= a_sw:
                continue
            rijx = nb_rij[pj, 0]
            rijy = nb_rij[pj, 1]
            rijz = nb_rij[pj, 2]
            # Each unordered pair {j, k} once (Stillinger & Weber 1985,
            # eq. 5: Phi3 = sum_{i<j<k} of three h terms, one per centre).
            for pk in range(pj + 1, ei):
                k_at = nb_idx[pk]
                r_ik = nb_r[pk]
                xik = r_ik / sig
                if xik >= a_sw:
                    continue
                rikx = nb_rij[pk, 0]
                riky = nb_rij[pk, 1]
                rikz = nb_rij[pk, 2]

                h3 = _sw_h3_scalar(rijx, rijy, rijz, rikx, riky, rikz,
                                    sig, a_sw, gam, lam, eps, costh0)
                energy += h3

                dzj0 = (_sw_h3_scalar(rijx + hh, rijy, rijz, rikx, riky, rikz, sig, a_sw, gam, lam, eps, costh0)
                        - _sw_h3_scalar(rijx - hh, rijy, rijz, rikx, riky, rikz, sig, a_sw, gam, lam, eps, costh0)) / (2 * hh)
                dzj1 = (_sw_h3_scalar(rijx, rijy + hh, rijz, rikx, riky, rikz, sig, a_sw, gam, lam, eps, costh0)
                        - _sw_h3_scalar(rijx, rijy - hh, rijz, rikx, riky, rikz, sig, a_sw, gam, lam, eps, costh0)) / (2 * hh)
                dzj2 = (_sw_h3_scalar(rijx, rijy, rijz + hh, rikx, riky, rikz, sig, a_sw, gam, lam, eps, costh0)
                        - _sw_h3_scalar(rijx, rijy, rijz - hh, rikx, riky, rikz, sig, a_sw, gam, lam, eps, costh0)) / (2 * hh)
                dzk0 = (_sw_h3_scalar(rijx, rijy, rijz, rikx + hh, riky, rikz, sig, a_sw, gam, lam, eps, costh0)
                        - _sw_h3_scalar(rijx, rijy, rijz, rikx - hh, riky, rikz, sig, a_sw, gam, lam, eps, costh0)) / (2 * hh)
                dzk1 = (_sw_h3_scalar(rijx, rijy, rijz, rikx, riky + hh, rikz, sig, a_sw, gam, lam, eps, costh0)
                        - _sw_h3_scalar(rijx, rijy, rijz, rikx, riky - hh, rikz, sig, a_sw, gam, lam, eps, costh0)) / (2 * hh)
                dzk2 = (_sw_h3_scalar(rijx, rijy, rijz, rikx, riky, rikz + hh, sig, a_sw, gam, lam, eps, costh0)
                        - _sw_h3_scalar(rijx, rijy, rijz, rikx, riky, rikz - hh, sig, a_sw, gam, lam, eps, costh0)) / (2 * hh)

                fj0 = -dzj0
                fj1 = -dzj1
                fj2 = -dzj2
                fk0 = -dzk0
                fk1 = -dzk1
                fk2 = -dzk2
                fi0 = dzj0 + dzk0
                fi1 = dzj1 + dzk1
                fi2 = dzj2 + dzk2

                forces[i, 0] += fi0
                forces[i, 1] += fi1
                forces[i, 2] += fi2
                forces[j_at, 0] += fj0
                forces[j_at, 1] += fj1
                forces[j_at, 2] += fj2
                forces[k_at, 0] += fk0
                forces[k_at, 1] += fk1
                forces[k_at, 2] += fk2

                stress[0, 0] += fj0 * rijx + fk0 * rikx
                stress[0, 1] += fj0 * rijy + fk0 * riky
                stress[0, 2] += fj0 * rijz + fk0 * rikz
                stress[1, 0] += fj1 * rijx + fk1 * rikx
                stress[1, 1] += fj1 * rijy + fk1 * riky
                stress[1, 2] += fj1 * rijz + fk1 * rikz
                stress[2, 0] += fj2 * rijx + fk2 * rikx
                stress[2, 1] += fj2 * rijy + fk2 * riky
                stress[2, 2] += fj2 * rijz + fk2 * rikz
    return energy


def _sw_2body_numpy_par(idx_i, idx_j, rij_arr, r_arr, forces, stress,
                         eps, sig, A_sw, B_sw, pp, qq, a_sw):
    """
    Parallel version of _sw_2body_numpy. See _pair_kernels._pair_body_numpy_par
    for the per-thread accumulator pattern used to avoid the write-write
    race on forces[ii]/forces[jj], and for the bug-fix note on why the
    buffer is sized from numba.config.NUMBA_NUM_THREADS rather than
    numba.get_num_threads() (a segfault-causing memory-safety fix).
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
        xi = r / sig
        if xi >= a_sw:
            continue
        t = 1.0 / (xi - a_sw)
        ex = _nb_exp(t)
        V2 = eps * A_sw * (B_sw * xi**(-pp) - xi**(-qq)) * ex
        dV2 = eps * A_sw * ex * ((-pp * B_sw * xi**(-pp - 1) + qq * xi**(-qq - 1))
                                  + (B_sw * xi**(-pp) - xi**(-qq)) * (-t * t))
        dV2dr = dV2 / sig
        energy += V2
        cx = (dV2dr / r)
        for d in range(3):
            fv = cx * rij_arr[p, d]
            forces_t[tid, ii, d] += fv
            forces_t[tid, jj, d] += -fv
        for di in range(3):
            for dj in range(3):
                stress_t[tid, di, dj] -= cx * rij_arr[p, di] * rij_arr[p, dj]

    for t in range(n_threads):
        forces += forces_t[t]
        stress += stress_t[t]
    return energy


def _sw_3body_numpy_par(n_start, n_end,
                         nb_start, nb_idx, nb_rij, nb_r,
                         forces, stress,
                         eps, sig, a_sw, gam, lam, costh0):
    """
    Parallel version of _sw_3body_numpy, parallelizing over the central
    atom index i, using per-thread force/stress accumulator buffers
    (force contributions to neighbour atoms j_at/k_at can collide across
    different central atoms handled by different threads).  See
    _pair_kernels._pair_body_numpy_par for why the buffer is sized from
    numba.config.NUMBA_NUM_THREADS rather than numba.get_num_threads().
    """
    N = forces.shape[0]
    n_threads = _numba.config.NUMBA_NUM_THREADS
    forces_t = np.zeros((n_threads, N, 3))
    stress_t = np.zeros((n_threads, 3, 3))
    energy = 0.0
    hh = 1e-6

    for i in _numba.prange(n_start, n_end):
        tid = _numba.get_thread_id()
        si = nb_start[i]
        ei = nb_start[i + 1]
        if ei - si < 2:
            continue
        for pj in range(si, ei):
            j_at = nb_idx[pj]
            r_ij = nb_r[pj]
            xij = r_ij / sig
            if xij >= a_sw:
                continue
            rijx = nb_rij[pj, 0]
            rijy = nb_rij[pj, 1]
            rijz = nb_rij[pj, 2]
            # Each unordered pair {j, k} once (Stillinger & Weber 1985,
            # eq. 5: Phi3 = sum_{i<j<k} of three h terms, one per centre).
            for pk in range(pj + 1, ei):
                k_at = nb_idx[pk]
                r_ik = nb_r[pk]
                xik = r_ik / sig
                if xik >= a_sw:
                    continue
                rikx = nb_rij[pk, 0]
                riky = nb_rij[pk, 1]
                rikz = nb_rij[pk, 2]

                h3 = _sw_h3_scalar(rijx, rijy, rijz, rikx, riky, rikz,
                                    sig, a_sw, gam, lam, eps, costh0)
                energy += h3

                dzj0 = (_sw_h3_scalar(rijx + hh, rijy, rijz, rikx, riky, rikz, sig, a_sw, gam, lam, eps, costh0)
                        - _sw_h3_scalar(rijx - hh, rijy, rijz, rikx, riky, rikz, sig, a_sw, gam, lam, eps, costh0)) / (2 * hh)
                dzj1 = (_sw_h3_scalar(rijx, rijy + hh, rijz, rikx, riky, rikz, sig, a_sw, gam, lam, eps, costh0)
                        - _sw_h3_scalar(rijx, rijy - hh, rijz, rikx, riky, rikz, sig, a_sw, gam, lam, eps, costh0)) / (2 * hh)
                dzj2 = (_sw_h3_scalar(rijx, rijy, rijz + hh, rikx, riky, rikz, sig, a_sw, gam, lam, eps, costh0)
                        - _sw_h3_scalar(rijx, rijy, rijz - hh, rikx, riky, rikz, sig, a_sw, gam, lam, eps, costh0)) / (2 * hh)
                dzk0 = (_sw_h3_scalar(rijx, rijy, rijz, rikx + hh, riky, rikz, sig, a_sw, gam, lam, eps, costh0)
                        - _sw_h3_scalar(rijx, rijy, rijz, rikx - hh, riky, rikz, sig, a_sw, gam, lam, eps, costh0)) / (2 * hh)
                dzk1 = (_sw_h3_scalar(rijx, rijy, rijz, rikx, riky + hh, rikz, sig, a_sw, gam, lam, eps, costh0)
                        - _sw_h3_scalar(rijx, rijy, rijz, rikx, riky - hh, rikz, sig, a_sw, gam, lam, eps, costh0)) / (2 * hh)
                dzk2 = (_sw_h3_scalar(rijx, rijy, rijz, rikx, riky, rikz + hh, sig, a_sw, gam, lam, eps, costh0)
                        - _sw_h3_scalar(rijx, rijy, rijz, rikx, riky, rikz - hh, sig, a_sw, gam, lam, eps, costh0)) / (2 * hh)

                fj0 = -dzj0
                fj1 = -dzj1
                fj2 = -dzj2
                fk0 = -dzk0
                fk1 = -dzk1
                fk2 = -dzk2
                fi0 = dzj0 + dzk0
                fi1 = dzj1 + dzk1
                fi2 = dzj2 + dzk2

                forces_t[tid, i, 0] += fi0
                forces_t[tid, i, 1] += fi1
                forces_t[tid, i, 2] += fi2
                forces_t[tid, j_at, 0] += fj0
                forces_t[tid, j_at, 1] += fj1
                forces_t[tid, j_at, 2] += fj2
                forces_t[tid, k_at, 0] += fk0
                forces_t[tid, k_at, 1] += fk1
                forces_t[tid, k_at, 2] += fk2

                stress_t[tid, 0, 0] += fj0 * rijx + fk0 * rikx
                stress_t[tid, 0, 1] += fj0 * rijy + fk0 * riky
                stress_t[tid, 0, 2] += fj0 * rijz + fk0 * rikz
                stress_t[tid, 1, 0] += fj1 * rijx + fk1 * rikx
                stress_t[tid, 1, 1] += fj1 * rijy + fk1 * riky
                stress_t[tid, 1, 2] += fj1 * rijz + fk1 * rikz
                stress_t[tid, 2, 0] += fj2 * rijx + fk2 * rikx
                stress_t[tid, 2, 1] += fj2 * rijy + fk2 * riky
                stress_t[tid, 2, 2] += fj2 * rijz + fk2 * rikz

    for t in range(n_threads):
        forces += forces_t[t]
        stress += stress_t[t]
    return energy


# -- Stillinger-Weber: multi-species kernels (Phase 5) --------------------
#
# The multi-species path looks up per-(species-triple) parameters from a
# Python dict keyed by element-symbol tuples (self.p), which Numba's
# nopython mode cannot touch directly. Instead of JIT-compiling the dict
# lookup itself, _compute_multi_species (below) resolves every possible
# (species_i, species_j[, species_k]) combination ONCE into dense
# integer-indexed NumPy tables (a "valid" boolean flag plus one float
# array per parameter) before calling these kernels -- the kernels then
# do only integer array indexing (type_ids[atom] -> species id -> table
# lookup), which Numba compiles exactly like the single-species kernels'
# plain scalar parameters. This mirrors, for a per-pair/per-triple
# parameter set, the same "resolve dict lookups outside the JIT boundary"
# principle the legacy single-species path already uses by extracting
# eps/sig/A_sw/... as plain floats before calling _sw_2body_jit/_sw_3body_jit.

def _sw_multi_2body_numpy(idx_i, idx_j, rij_arr, r_arr, type_ids, forces, stress,
                          tb_valid, tb_eps, tb_sig, tb_a, tb_A, tb_B, tb_p, tb_q):
    """Table-driven multi-species analogue of _sw_2body_numpy. tb_* are
    (n_species, n_species) dense arrays; tb_valid[si, sj] mirrors
    `self._two_body(species[si], species[sj]) is not None and A != 0`."""
    energy = 0.0
    for p in range(len(idx_i)):
        ii = idx_i[p]
        jj = idx_j[p]
        si = type_ids[ii]
        sj = type_ids[jj]
        if not tb_valid[si, sj]:
            continue
        sig = tb_sig[si, sj]
        a_sw = tb_a[si, sj]
        r = r_arr[p]
        xi = r / sig
        if xi >= a_sw:
            continue
        eps = tb_eps[si, sj]
        A_sw = tb_A[si, sj]
        B_sw = tb_B[si, sj]
        pp = tb_p[si, sj]
        qq = tb_q[si, sj]
        t = 1.0 / (xi - a_sw)
        ex = _nb_exp(t)
        V2 = eps * A_sw * (B_sw * xi**(-pp) - xi**(-qq)) * ex
        dV2 = eps * A_sw * ex * ((-pp * B_sw * xi**(-pp - 1) + qq * xi**(-qq - 1))
                                  + (B_sw * xi**(-pp) - xi**(-qq)) * (-t * t))
        dV2dr = dV2 / sig
        energy += V2
        cx = dV2dr / r
        for d in range(3):
            fv = cx * rij_arr[p, d]
            forces[ii, d] += fv
            forces[jj, d] += -fv
        for di in range(3):
            for dj in range(3):
                stress[di, dj] -= cx * rij_arr[p, di] * rij_arr[p, dj]
    return energy


def _sw_h3_scalar_multi(rijx, rijy, rijz, rikx, riky, rikz,
                        sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t):
    """Table-driven analogue of _sw_h3_scalar, differing only in where
    gamma comes from: here it is looked up per (i, j, k) triple from
    tri_gam, whereas _sw_h3_scalar takes the single flat-format scalar.
    (This function's explicit ``gam_t / (x - a_t)`` is what made the
    legacy path's hard-coded ``1 / (x - a_sw)`` detectable as a defect;
    see _sw_h3_scalar's bug-fix note.)"""
    r_ij = math.sqrt(rijx * rijx + rijy * rijy + rijz * rijz)
    r_ik = math.sqrt(rikx * rikx + riky * riky + rikz * rikz)
    xij = r_ij / sigma_t
    xik = r_ik / sigma_t
    if xij >= a_t or xik >= a_t:
        return 0.0
    exp_ij = _nb_exp(gam_t / (xij - a_t))
    exp_ik = _nb_exp(gam_t / (xik - a_t))
    cos_th = (rijx * rikx + rijy * riky + rijz * rikz) / (r_ij * r_ik)
    dcos = cos_th - costh0_t
    return eps_t * lam_t * exp_ij * exp_ik * dcos * dcos


def _sw_multi_3body_numpy(n_start, n_end,
                          nb_start, nb_idx, nb_rij, nb_r, type_ids,
                          forces, stress,
                          tri_valid, tri_eps, tri_sig, tri_a, tri_lam, tri_gam, tri_cos):
    """Table-driven multi-species analogue of _sw_3body_numpy. tri_* are
    (n_species, n_species, n_species) dense arrays indexed [i_sp, j_sp,
    k_sp]; tri_valid mirrors `self._three_body(si, sj, sk) is not None`.
    Same central-finite-difference force/stress derivation as
    _sw_3body_numpy (see that function's docstring for why), and the
    same early `if h3 == 0.0: continue` skip the existing multi-species
    NumPy path uses to avoid 6 wasted finite-difference evaluations once
    a triple is out of its (per-triple) cutoff."""
    energy = 0.0
    hh = 1e-6
    for i in range(n_start, n_end):
        si = nb_start[i]
        ei = nb_start[i + 1]
        if ei - si < 2:
            continue
        i_sp = type_ids[i]
        for pj in range(si, ei):
            j_at = nb_idx[pj]
            j_sp = type_ids[j_at]
            rijx = nb_rij[pj, 0]
            rijy = nb_rij[pj, 1]
            rijz = nb_rij[pj, 2]
            for pk in range(si, ei):
                if pk == pj:
                    continue
                k_at = nb_idx[pk]
                k_sp = type_ids[k_at]
                if not tri_valid[i_sp, j_sp, k_sp]:
                    continue
                rikx = nb_rij[pk, 0]
                riky = nb_rij[pk, 1]
                rikz = nb_rij[pk, 2]

                sigma_t = tri_sig[i_sp, j_sp, k_sp]
                a_t     = tri_a[i_sp, j_sp, k_sp]
                gam_t   = tri_gam[i_sp, j_sp, k_sp]
                eps_t   = tri_eps[i_sp, j_sp, k_sp]
                # Both orders (j, k) and (k, j) are visited, each with its
                # own record, so each weighs one half: an unordered pair
                # contributes (h_ijk + h_ikj) / 2, which is Stillinger &
                # Weber's single h when the two records agree and does not
                # depend on neighbour order when they do not.
                lam_t   = 0.5 * tri_lam[i_sp, j_sp, k_sp]
                costh0_t = tri_cos[i_sp, j_sp, k_sp]

                h3 = _sw_h3_scalar_multi(rijx, rijy, rijz, rikx, riky, rikz,
                                         sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t)
                if h3 == 0.0:
                    continue
                energy += h3

                dzj0 = (_sw_h3_scalar_multi(rijx + hh, rijy, rijz, rikx, riky, rikz, sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t)
                        - _sw_h3_scalar_multi(rijx - hh, rijy, rijz, rikx, riky, rikz, sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t)) / (2 * hh)
                dzj1 = (_sw_h3_scalar_multi(rijx, rijy + hh, rijz, rikx, riky, rikz, sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t)
                        - _sw_h3_scalar_multi(rijx, rijy - hh, rijz, rikx, riky, rikz, sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t)) / (2 * hh)
                dzj2 = (_sw_h3_scalar_multi(rijx, rijy, rijz + hh, rikx, riky, rikz, sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t)
                        - _sw_h3_scalar_multi(rijx, rijy, rijz - hh, rikx, riky, rikz, sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t)) / (2 * hh)
                dzk0 = (_sw_h3_scalar_multi(rijx, rijy, rijz, rikx + hh, riky, rikz, sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t)
                        - _sw_h3_scalar_multi(rijx, rijy, rijz, rikx - hh, riky, rikz, sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t)) / (2 * hh)
                dzk1 = (_sw_h3_scalar_multi(rijx, rijy, rijz, rikx, riky + hh, rikz, sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t)
                        - _sw_h3_scalar_multi(rijx, rijy, rijz, rikx, riky - hh, rikz, sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t)) / (2 * hh)
                dzk2 = (_sw_h3_scalar_multi(rijx, rijy, rijz, rikx, riky, rikz + hh, sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t)
                        - _sw_h3_scalar_multi(rijx, rijy, rijz, rikx, riky, rikz - hh, sigma_t, a_t, gam_t, eps_t, lam_t, costh0_t)) / (2 * hh)

                fj0 = -dzj0
                fj1 = -dzj1
                fj2 = -dzj2
                fk0 = -dzk0
                fk1 = -dzk1
                fk2 = -dzk2
                fi0 = dzj0 + dzk0
                fi1 = dzj1 + dzk1
                fi2 = dzj2 + dzk2

                forces[i, 0] += fi0
                forces[i, 1] += fi1
                forces[i, 2] += fi2
                forces[j_at, 0] += fj0
                forces[j_at, 1] += fj1
                forces[j_at, 2] += fj2
                forces[k_at, 0] += fk0
                forces[k_at, 1] += fk1
                forces[k_at, 2] += fk2

                stress[0, 0] += fj0 * rijx + fk0 * rikx
                stress[0, 1] += fj0 * rijy + fk0 * riky
                stress[0, 2] += fj0 * rijz + fk0 * rikz
                stress[1, 0] += fj1 * rijx + fk1 * rikx
                stress[1, 1] += fj1 * rijy + fk1 * riky
                stress[1, 2] += fj1 * rijz + fk1 * rikz
                stress[2, 0] += fj2 * rijx + fk2 * rikx
                stress[2, 1] += fj2 * rijy + fk2 * riky
                stress[2, 2] += fj2 * rijz + fk2 * rikz
    return energy


if _HAS_NUMBA:
    # _sw_3body_numpy_par references _sw_h3_scalar as a module-level global;
    # numba's nopython mode requires that global to already be a compiled
    # (njit) callable at the time _sw_3body_numpy_par itself gets compiled,
    # so _sw_h3_scalar must be JIT-compiled and reassigned FIRST.
    _sw_h3_scalar = _numba_njit(_sw_h3_scalar)  # noqa: F811
    _sw_2body_jit = _numba_par(_sw_2body_numpy_par)
    _sw_3body_jit = _numba_par(_sw_3body_numpy_par)

    _sw_h3_scalar_multi = _numba_njit(_sw_h3_scalar_multi)  # noqa: F811
    _sw_multi_2body_jit = _numba_njit(_sw_multi_2body_numpy)
    _sw_multi_3body_jit = _numba_njit(_sw_multi_3body_numpy)
else:
    _sw_2body_jit = _sw_2body_numpy
    _sw_3body_jit = _sw_3body_numpy

    _sw_multi_2body_jit = _sw_multi_2body_numpy
    _sw_multi_3body_jit = _sw_multi_3body_numpy


def _effective_two_body(rec):
    """The two-body inputs a record actually supplies, exactly as both
    evaluation paths read them -- or None when it supplies no two-body
    term (record absent, or A == 0: both paths skip the pair then)."""
    if rec is None or float(rec.get('A', 0.0)) == 0.0:
        return None
    return {'epsilon': float(rec.get('epsilon', 1.0)),
            'sigma': float(rec['sigma']), 'a': float(rec['a']),
            'A': float(rec['A']), 'B': float(rec.get('B', 0.0)),
            'p': float(rec.get('p', 4)), 'q': float(rec.get('q', 0))}


def _check_two_body_symmetry(params: dict) -> None:
    """Refuse multi-species SW parameters whose two-body term depends on
    the order the atoms are listed in.

    Each unordered pair is evaluated ONCE, from the diagonal record
    (s_i, s_j, s_j) of whichever atom of the pair has the LOWER index.
    A Ge-Sn pair therefore reads (Ge,Sn,Sn) when the Ge comes first and
    (Sn,Ge,Ge) when the Sn does.  If those two records supply different
    two-body terms, the energy of one fixed structure changes when its
    atoms are merely relabelled: 7.98 eV on the 128-atom example/GeSn
    cell with (Ge,Sn,Sn).A scaled by 1.2.  That is not a physical
    potential, so it is rejected here rather than evaluated.

    Only what is actually EVALUATED has to agree: a pair where neither
    record has A != 0 has no two-body term, and the records' sigma/a
    then serve their own three-body terms only, which may differ.
    """
    species = sorted({e for t in params for e in t})
    for x, si in enumerate(species):
        for sj in species[x + 1:]:
            r1 = _effective_two_body(params.get((si, sj, sj)))
            r2 = _effective_two_body(params.get((sj, si, si)))
            if r1 is None and r2 is None:
                continue
            if (r1 is not None and r2 is not None
                    and all(math.isclose(r1[k], r2[k], rel_tol=1e-12, abs_tol=0.0)
                            for k in r1)):
                continue
            if r1 is None or r2 is None:
                have, miss = (((si, sj, sj), (sj, si, si)) if r2 is None
                              else ((sj, si, si), (si, sj, sj)))
                raise ValueError(
                    f"StillingerWeberPotential: {have} supplies a two-body "
                    f"{si}-{sj} term but {miss} does not (absent, or A = 0). "
                    f"The pair is evaluated from whichever of the two belongs "
                    f"to the lower-indexed atom, so the energy would depend "
                    f"on atom order.  Give {miss} the same two-body "
                    f"parameters (epsilon, sigma, a, A, B, p, q).")
            diff = [k for k in r1
                    if not math.isclose(r1[k], r2[k], rel_tol=1e-12, abs_tol=0.0)]
            raise ValueError(
                f"StillingerWeberPotential: ({si},{sj},{sj}) and "
                f"({sj},{si},{si}) disagree on the {si}-{sj} two-body "
                f"parameter(s) {diff}: "
                + ", ".join(f"{k} {r1[k]} vs {r2[k]}" for k in diff)
                + ".  The pair is evaluated from whichever record belongs "
                f"to the lower-indexed atom, so the energy would depend on "
                f"atom order.  Their lambda/gamma/costheta0 may differ; "
                f"these may not.")


class StillingerWeberPotential:
    """
    Stillinger-Weber (SW) potential for covalently bonded systems
    (multi-species).

    Reference: Stillinger & Weber, Phys. Rev. B 31, 5262 (1985).

    Energy decomposition
    --------------------
    E = sum_{i<j} V2(r_ij) + sum_{i<j<k} V3(r_ij, r_ik, theta_ijk)

    2-body term (repulsion + short-range attraction):
        V2(r) = epsilon . A (B(sigma/r)^p - (sigma/r)^q) exp(sigma/(r - a.sigma))
               x fc(r; R=a.sigma, D=0)   (hard cut at a.sigma)

    3-body term (angle penalty --- favours tetrahedral angle):
        V3(r_ij, r_ik, theta) = epsilon . lambda (cos theta - cos theta_0)^2
                            x exp(gamma.sigma/(r_ij - a.sigma))
                              exp(gamma.sigma/(r_ik - a.sigma))

    Legacy single-species format (backward compatible)
    ----------------------------------------------------
    A flat scalar dict (keys 'epsilon','sigma','A','B','p','q','a',
    'lambda','gamma','costheta0' at the top level, as used for Si) is
    accepted directly: it is automatically expanded so the single
    species in `species` maps to this one parameter set.  Every key
    falls back to Stillinger & Weber's own Si value when omitted, so an
    empty dict is SW Si -- including 'gamma', whose fallback is 1.20
    (it used to be hard-coded to 1 inside the three-body exponent and
    read from the dict by nothing at all; see _sw_h3_scalar).

    Multi-species format (LAMMPS .sw-file-like convention)
    ----------------------------------------------------------
    params : dict keyed by ordered element triples (i, j, k) -> dict with:
        sigma, a           : pair-cutoff-scale parameters.  For a
                              diagonal entry (i, j, j) these define the
                              i-j pair's two-body AND three-body cutoff.
                              For a mixed entry (i, j, k) with j != k,
                              these are used (per this potential's own
                              convention, matching the source table the
                              user extracted) for BOTH exponential decay
                              legs of that specific triplet's angular
                              term -- i.e. sigma_ij and sigma_ik are not
                              looked up separately from other rows; the
                              (i, j, k) row's own sigma/a apply
                              uniformly to both r_ij and r_ik in that
                              triplet's h(i;j,k) contribution.
        A, B, p, q          : two-body parameters.  Only meaningful (and
                              only read) for a diagonal entry (i, j, j);
                              for a mixed entry (j != k) they should be
                              0 (or omitted) since no two-body term is
                              evaluated on a mixed row.
        lambda, gamma       : three-body parameters for this specific
                              (i, j, k) angular term.
        epsilon             : energy prefactor (defaults to 1.0, matching
                              the common convention of folding epsilon
                              into A, B, lambda directly).
        costheta0           : equilibrium cos(angle) (defaults to -1/3,
                              the tetrahedral angle).

    Any (i, j, k) combination absent from `params` is treated as
    NON-INTERACTING for that role (two-body pair skipped if (i,j,j) is
    missing; a given three-body angular contribution skipped if the
    specific (i,j,k) entry is missing).

    Parameters
    ----------
    params  : dict (see above).
    r_cut   : optional override (legacy single-species only; ignored
              for the multi-species format, where each pair's own a*sigma
              determines its cutoff).
    species : list of species symbols.  Required for the multi-species
              (nested-dict) parameter format; optional (defaults to a
              single dummy species) for the legacy flat format.
    """

    def __init__(self,
                 params: dict,
                 r_cut: Optional[float] = None,
                 species: Optional[list] = None):
        self.species = list(species) if species else None

        is_legacy_flat = ('epsilon' in params or 'sigma' in params) \
            and not any(isinstance(k, tuple) for k in params)

        if is_legacy_flat:
            sp0 = (self.species[0] if self.species else '__X__')
            self.species = [sp0]
            self._multi = False
            self.p = params
            eps = params.get('epsilon', 2.1682)
            sig = params.get('sigma', 2.0951)
            a = params.get('a', 1.80)
            self.r_cut = r_cut if r_cut is not None else a * sig
        else:
            if not self.species:
                raise ValueError(
                    "StillingerWeberPotential: 'species' is required "
                    "when using the multi-species (nested-triple) "
                    "parameter format.")
            self._multi = True
            self.p = params
            _check_two_body_symmetry(params)
            rc_max = 0.0
            for (i, j, k), v in params.items():
                if j == k:
                    rc_max = max(rc_max, float(v['a']) * float(v['sigma']))
            self.r_cut = rc_max if rc_max > 0 else 4.0

    # ------------------------------------------------------------------
    def _two_body(self, si: str, sj: str) -> Optional[dict]:
        """Return two-body params for the (i,j) pair (diagonal entry)."""
        return self.p.get((si, sj, sj))

    def _three_body(self, si: str, sj: str, sk: str) -> Optional[dict]:
        """Return angular params for the (i; j,k) triple."""
        return self.p.get((si, sj, sk))

    # ------------------------------------------------------------------
    def compute(self,
                lattice: np.ndarray,
                positions: np.ndarray,
                atom_types: list,
                neigh_list: Optional[NeighbourList] = None,
                use_numba: bool = False,
                ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute SW energy, forces, and virial stress.

        Dispatches to the legacy single-species implementation when the
        flat parameter format was supplied, or the general multi-species
        implementation otherwise. Both accept ``use_numba`` (or honour
        the ``NUMBA_ENABLE`` environment variable) to JIT-compile their
        per-pair/per-triple loops; see ``_compute_multi_species``'s own
        docstring for how its dict-keyed parameter table is resolved
        into JIT-compatible dense arrays before dispatching.
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

        eps = self.p.get('epsilon', 2.1682)
        sig = self.p.get('sigma', 2.0951)
        A_sw = self.p.get('A', 7.049556)
        B_sw = self.p.get('B', 0.6022246)
        pp = float(self.p.get('p', 4))
        qq = float(self.p.get('q', 0))
        a_sw = self.p.get('a', 1.80)
        lam = self.p.get('lambda', 21.0)
        # Defaults to Stillinger & Weber's own Si value, like every
        # other fallback in this block.  Until this was read, the
        # three-body exponent hard-coded gamma = 1 and silently
        # discarded whatever the caller supplied (see _sw_h3_scalar).
        gam = self.p.get('gamma', 1.20)
        costh0 = self.p.get('costheta0', -1.0 / 3.0)
        rc = a_sw * sig

        # 1. Vectorised MIC rij
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
            dr_raw = positions[idx_j_raw] - positions[idx_i_raw]
            s_raw = dr_raw @ inv_lat
            s_raw -= np.round(s_raw)
            rij_all = s_raw @ lattice
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

        # 2. Build CSR neighbour structure (fully vectorised -- no Python loop)
        # Each undirected pair (i<j) contributes two directed entries: i->j and j->i.
        # We sort by the source atom index to build a contiguous CSR layout.
        src_all = np.concatenate([idx_i, idx_j])          # source atom index
        dst_all = np.concatenate([idx_j, idx_i])          # destination atom index
        rij_dir = np.concatenate([rij_arr, -rij_arr])     # directed displacement
        r_dir = np.concatenate([r_arr, r_arr])       # directed distance

        order = np.argsort(src_all, kind='stable')
        src_s = src_all[order]
        rij_s = rij_dir[order]
        r_s = r_dir[order]

        nb_cnt = np.bincount(src_s, minlength=N).astype(np.int32)
        nb_start = np.zeros(N + 1, dtype=np.int32)
        nb_start[1:] = np.cumsum(nb_cnt)
        nb_idx = dst_all[order].astype(np.int32)
        nb_rij = np.ascontiguousarray(rij_s)
        nb_r = np.ascontiguousarray(r_s)

        # 3. MPI domain split
        P = len(idx_i)
        p_start, p_end = _mpi_domain_split(P)
        n_start, n_end = _mpi_domain_split(N)

        forces = np.zeros((N, 3))
        stress = np.zeros((3, 3))
        energy = 0.0

        # 4 + 5: 2-body and 3-body -- dispatch to Numba JIT or NumPy
        from ..utils.jit import _NUMBA_ENV
        numba_on = (use_numba or _NUMBA_ENV) and _HAS_NUMBA

        if numba_on:
            # Numba path: pass CSR arrays to pre-compiled JIT kernels.
            # _sw_2body_jit uses numba.prange (NUMBA_NUM_THREADS threads).
            e2 = _sw_2body_jit(
                idx_i[p_start:p_end].copy(), idx_j[p_start:p_end].copy(),
                rij_arr[p_start:p_end].copy(), r_arr[p_start:p_end].copy(),
                forces, stress,
                eps, sig, A_sw, B_sw, pp, qq, a_sw)
            energy += e2
            e3 = _sw_3body_jit(
                n_start, n_end,
                nb_start, nb_idx, nb_rij, nb_r,
                forces, stress,
                eps, sig, a_sw, gam, lam, costh0)
            energy += e3
        else:
            # NumPy path (MPI pair-split for 2-body)
            r_loc = r_arr[p_start:p_end]
            rij_loc = rij_arr[p_start:p_end]
            ii_loc = idx_i[p_start:p_end]
            jj_loc = idx_j[p_start:p_end]
            xi_loc = r_loc / sig
            in_cut = xi_loc < a_sw
            if np.any(in_cut):
                xic = xi_loc[in_cut]
                rc2 = rij_loc[in_cut]
                rii = ii_loc[in_cut]
                rjj = jj_loc[in_cut]
                r_in = r_loc[in_cut]
                t = 1.0 / (xic - a_sw)
                ex = np.exp(t)
                V2 = eps * A_sw * (B_sw * xic**(-pp) - xic**(-qq)) * ex
                dV2dx = eps * A_sw * ex * ((-pp * B_sw * xic**(-pp - 1) + qq * xic**(-qq - 1))
                                           + (B_sw * xic**(-pp) - xic**(-qq)) * (-t**2))
                energy += float(V2.sum())
                fij = (dV2dx / sig / r_in)[:, None] * rc2
                np.add.at(forces, rii, fij)
                np.add.at(forces, rjj, -fij)
                stress -= np.einsum('ki,kj->ij', fij, rc2)

            # 3-body (MPI atom-split). See _sw_3body_numpy's docstring for
            # why finite differences replace a hand-derived analytic force.
            def _h3_term(rij_v, rik_v):
                r_ij_ = float(np.linalg.norm(rij_v))
                r_ik_ = float(np.linalg.norm(rik_v))
                xij_ = r_ij_ / sig
                xik_ = r_ik_ / sig
                if xij_ >= a_sw or xik_ >= a_sw:
                    return 0.0
                e_ij = math.exp(gam / (xij_ - a_sw))
                e_ik = math.exp(gam / (xik_ - a_sw))
                cos_ = float(np.dot(rij_v, rik_v)) / (r_ij_ * r_ik_)
                return lam * eps * e_ij * e_ik * (cos_ - costh0)**2

            for i in range(n_start, n_end):
                si = int(nb_start[i])
                ei = int(nb_start[i + 1])
                if ei - si < 2:
                    continue
                nbj = nb_idx[si:ei]
                rij_i = nb_rij[si:ei]
                r_i = nb_r[si:ei]
                for pj in range(len(nbj)):
                    j_at = int(nbj[pj])
                    rij_v = rij_i[pj]
                    r_ij = float(r_i[pj])
                    xij = r_ij / sig
                    if xij >= a_sw:
                        continue
                    # each unordered pair once (see _sw_3body_numpy)
                    for pk in range(pj + 1, len(nbj)):
                        k_at = int(nbj[pk])
                        rik_v = rij_i[pk]
                        r_ik = float(r_i[pk])
                        xik = r_ik / sig
                        if xik >= a_sw:
                            continue
                        h3 = _h3_term(rij_v, rik_v)
                        energy += h3
                        hh = 1e-6
                        dz_dposj = np.zeros(3)
                        dz_dposk = np.zeros(3)
                        for d_ in range(3):
                            rp = rij_v.copy()
                            rp[d_] += hh
                            rm = rij_v.copy()
                            rm[d_] -= hh
                            dz_dposj[d_] = (_h3_term(rp, rik_v) - _h3_term(rm, rik_v)) / (2 * hh)
                        for d_ in range(3):
                            rp = rik_v.copy()
                            rp[d_] += hh
                            rm = rik_v.copy()
                            rm[d_] -= hh
                            dz_dposk[d_] = (_h3_term(rij_v, rp) - _h3_term(rij_v, rm)) / (2 * hh)
                        Fj = -dz_dposj
                        Fk = -dz_dposk
                        Fi = dz_dposj + dz_dposk
                        forces[i] += Fi
                        forces[j_at] += Fj
                        forces[k_at] += Fk
                        # Sign: see the module-level bug-fix note in
                        # _sw_3body_numpy's docstring -- this must be +=,
                        # not -=.
                        stress += np.outer(Fj, rij_v) + np.outer(Fk, rik_v)

        # 6. MPI reduction
        #
        # Bug-fix note (found in code-check, same class as F16): the energy
        # reduction below used to be a raw `_mpi_comm.allreduce(energy,
        # op=_MPI.SUM)` call, gated only on the static `_HAS_MPI` flag. That
        # bypasses `local_only()` entirely, because `local_only()` only
        # suppresses calls that go through `_mpi_is_active()` -- which a raw
        # `_mpi_comm.allreduce()` call never checks. So whenever this
        # `compute()` was invoked from an OUTER item-splitting caller (e.g.
        # `fitting/loss.py::generic_loss_function`'s per-structure loop,
        # `solvers/neb.py`'s per-image loop) wrapped in `with local_only():`,
        # each rank's already-locally-correct `energy` for its OWN structure/
        # image got silently summed together with every OTHER rank's energy
        # for its (different!) structure/image -- while `forces`/`stress`
        # stayed correct, because they already went through the
        # `local_only()`-aware `_mpi_allreduce_sum()` wrapper below. This
        # combination (SW/Tersoff/EAM potential + outer `local_only()`
        # splitting + real multi-rank `mpirun`) was never previously
        # exercised by `test_mpi.py` (which only tests SW/Tersoff/EAM
        # directly on a single shared structure -- the one scenario where
        # the raw call and the wrapper behave identically -- and tests
        # `generic_loss_function` only with an LJ potential, which already
        # used the wrapper correctly). Fixed by routing through the same
        # `_mpi_allreduce_sum_scalar()` helper `pair_potentials.py` and
        # `coulomb/ewald.py` already use for their own scalar energy.
        if _HAS_MPI:
            energy = _mpi_allreduce_sum_scalar(energy)
            forces = _mpi_allreduce_sum(forces)
            stress = _mpi_allreduce_sum(stress)
        stress /= volume
        return energy, forces, stress

    # ------------------------------------------------------------------
    #  General multi-species path (NumPy, per-pair/per-triple lookup;
    #  optionally JIT-compiled -- see the dense-table note below)
    # ------------------------------------------------------------------
    def _compute_multi_species(self, lattice, positions, atom_types,
                               neigh_list, use_numba: bool = False):
        """
        Bug-fix note (Phase 5, numba for the multi-species path): this
        path used to be pure Python/NumPy only (unlike the legacy
        single-species path, which has had a JIT-compiled option since
        Phase 2) -- confirmed via grep that ``_compute_multi_species``
        never referenced ``_HAS_NUMBA``. Its two nested per-pair/per-
        triple loops (Pass 1, Pass 2 below) dominate runtime for
        multi-species systems the same way the single-species loops did
        before their own JIT kernels were added.

        The obstacle to JIT-compiling this path directly is that its
        parameters come from ``self.p``, a dict keyed by tuples of
        element-symbol STRINGS (``self._two_body``/``self._three_body``)
        -- not something Numba's nopython mode can look up. Rather than
        attempting that, every possible (species_i, species_j[,
        species_k]) combination is resolved into dense, integer-indexed
        NumPy tables ONCE here (plain Python, outside the hot loop;
        negligible cost relative to the O(pairs)/O(triples) work below),
        then handed to ``_sw_multi_2body_jit``/``_sw_multi_3body_jit``,
        which only ever index into these tables with plain integers
        (``type_ids[atom] -> species id``) -- exactly the same "resolve
        once outside, then pure-array-math inside" principle the legacy
        path already uses for its plain scalar parameters.
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
        n_start, n_end = _mpi_domain_split(N)

        from ..utils.jit import _NUMBA_ENV
        numba_on = (use_numba or _NUMBA_ENV) and _HAS_NUMBA

        if numba_on:
            # Resolve every (species_i, species_j[, species_k]) combination
            # into dense integer-indexed tables ONCE (see this method's own
            # docstring); the JIT kernels then only ever index these with
            # plain integers.
            n_sp = len(self.species)
            sp_index = {s: idx for idx, s in enumerate(self.species)}
            type_ids = np.array([sp_index[t] for t in sp], dtype=np.int64)

            tb_valid = np.zeros((n_sp, n_sp), dtype=np.bool_)
            tb_eps = np.zeros((n_sp, n_sp))
            tb_sig = np.ones((n_sp, n_sp))
            tb_a = np.zeros((n_sp, n_sp))
            tb_A = np.zeros((n_sp, n_sp))
            tb_B = np.zeros((n_sp, n_sp))
            tb_p = np.zeros((n_sp, n_sp))
            tb_q = np.zeros((n_sp, n_sp))
            for i_sp, si in enumerate(self.species):
                for j_sp, sj in enumerate(self.species):
                    tb = self._two_body(si, sj)
                    if tb is None or float(tb.get('A', 0.0)) == 0.0:
                        continue
                    tb_valid[i_sp, j_sp] = True
                    tb_eps[i_sp, j_sp] = float(tb.get('epsilon', 1.0))
                    tb_sig[i_sp, j_sp] = float(tb['sigma'])
                    tb_a[i_sp, j_sp] = float(tb['a'])
                    tb_A[i_sp, j_sp] = float(tb['A'])
                    tb_B[i_sp, j_sp] = float(tb.get('B', 0.0))
                    tb_p[i_sp, j_sp] = float(tb.get('p', 4))
                    tb_q[i_sp, j_sp] = float(tb.get('q', 0))

            tri_valid = np.zeros((n_sp, n_sp, n_sp), dtype=np.bool_)
            tri_eps = np.zeros((n_sp, n_sp, n_sp))
            tri_sig = np.ones((n_sp, n_sp, n_sp))
            tri_a = np.zeros((n_sp, n_sp, n_sp))
            tri_lam = np.zeros((n_sp, n_sp, n_sp))
            tri_gam = np.zeros((n_sp, n_sp, n_sp))
            tri_cos = np.full((n_sp, n_sp, n_sp), -1.0 / 3.0)
            for i_sp, si in enumerate(self.species):
                for j_sp, sj in enumerate(self.species):
                    for k_sp, sk in enumerate(self.species):
                        tri = self._three_body(si, sj, sk)
                        if tri is None:
                            continue
                        tri_valid[i_sp, j_sp, k_sp] = True
                        tri_eps[i_sp, j_sp, k_sp] = float(tri.get('epsilon', 1.0))
                        tri_sig[i_sp, j_sp, k_sp] = float(tri['sigma'])
                        tri_a[i_sp, j_sp, k_sp] = float(tri['a'])
                        tri_lam[i_sp, j_sp, k_sp] = float(tri['lambda'])
                        tri_gam[i_sp, j_sp, k_sp] = float(tri['gamma'])
                        tri_cos[i_sp, j_sp, k_sp] = float(tri.get('costheta0', -1.0 / 3.0))

            e2 = _sw_multi_2body_jit(
                idx_i[p_start:p_end].copy(), idx_j[p_start:p_end].copy(),
                rij_arr[p_start:p_end].copy(), r_arr[p_start:p_end].copy(),
                type_ids, forces, stress,
                tb_valid, tb_eps, tb_sig, tb_a, tb_A, tb_B, tb_p, tb_q)
            energy += e2
            e3 = _sw_multi_3body_jit(
                n_start, n_end, nb_start, nb_idx, nb_rij, nb_r, type_ids,
                forces, stress,
                tri_valid, tri_eps, tri_sig, tri_a, tri_lam, tri_gam, tri_cos)
            energy += e3

        else:
            def _h3_term(sign_i, rij_v, rik_v, tri):
                """Scalar three-body term for a given (i,j,k) parameter set."""
                r_ij_ = float(np.linalg.norm(rij_v))
                r_ik_ = float(np.linalg.norm(rik_v))
                sigma_t = float(tri['sigma'])
                a_t = float(tri['a'])
                xij_ = r_ij_ / sigma_t
                xik_ = r_ik_ / sigma_t
                if xij_ >= a_t or xik_ >= a_t:
                    return 0.0
                eps_t = float(tri.get('epsilon', 1.0))
                # one half: both orders (j, k) and (k, j) are visited, see
                # _sw_multi_3body_numpy
                lam_t = 0.5 * float(tri['lambda'])
                gam_t = float(tri['gamma'])
                costh0_t = float(tri.get('costheta0', -1.0 / 3.0))
                e_ij = math.exp(gam_t / (xij_ - a_t))
                e_ik = math.exp(gam_t / (xik_ - a_t))
                cos_ = float(np.dot(rij_v, rik_v)) / (r_ij_ * r_ik_)
                return eps_t * lam_t * e_ij * e_ik * (cos_ - costh0_t)**2

            # ---- Pass 1: two-body energy/forces (diagonal (i,j,j) entries) --
            for p in range(p_start, p_end):
                ii = int(idx_i[p])
                jj = int(idx_j[p])
                si, sj = sp[ii], sp[jj]
                tb = self._two_body(si, sj)
                if tb is None:
                    continue
                eps_p = float(tb.get('epsilon', 1.0))
                sig_p = float(tb['sigma'])
                a_p = float(tb['a'])
                A_p = float(tb.get('A', 0.0))
                B_p = float(tb.get('B', 0.0))
                p_p = float(tb.get('p', 4))
                q_p = float(tb.get('q', 0))
                r_ij = float(r_arr[p])
                rij_v = rij_arr[p]
                x = r_ij / sig_p
                if x >= a_p or A_p == 0.0:
                    continue
                t = 1.0 / (x - a_p)
                ex = math.exp(t)
                V2 = eps_p * A_p * (B_p * x**(-p_p) - x**(-q_p)) * ex
                dV2dx = eps_p * A_p * ex * ((-p_p * B_p * x**(-p_p - 1) + q_p * x**(-q_p - 1))
                                            + (B_p * x**(-p_p) - x**(-q_p)) * (-t**2))
                energy += V2
                fij = (dV2dx / sig_p / r_ij) * rij_v
                forces[ii] += fij
                forces[jj] -= fij
                stress -= np.outer(fij, rij_v)

            # ---- Pass 2: three-body energy/forces -----------------------
            for i in range(n_start, n_end):
                si_start, si_end = int(nb_start[i]), int(nb_start[i + 1])
                if si_end - si_start < 2:
                    continue
                si = sp[i]
                nbrs = range(si_start, si_end)
                for pj in nbrs:
                    j_at = int(nb_idx[pj])
                    rij_v = nb_rij[pj]
                    r_ij = float(nb_r[pj])
                    sj = sp[j_at]
                    for pk in nbrs:
                        if pk == pj:
                            continue
                        k_at = int(nb_idx[pk])
                        rik_v = nb_rij[pk]
                        r_ik = float(nb_r[pk])
                        sk = sp[k_at]
                        tri = self._three_body(si, sj, sk)
                        if tri is None:
                            continue
                        h3 = _h3_term(si, rij_v, rik_v, tri)
                        if h3 == 0.0:
                            continue
                        # Both orders (j, k) and (k, j) are visited and
                        # _h3_term carries a factor 1/2, so each unordered
                        # pair contributes (h_ijk + h_ikj) / 2.  Bug-fix note
                        # (2026-09-25): this used to sum both orders WITHOUT
                        # the 1/2, under a comment claiming that literature
                        # SW parameters assume that convention.  They do
                        # not: Stillinger & Weber (PRB 31, 5262, eq. 5) count
                        # each (centre, unordered pair) once, as LAMMPS
                        # pair_sw does (jj < kk).  The doubled three-body term
                        # gave SW Si C11/C12/C44 = 201.4/51.4/90.5 GPa instead
                        # of 151.4/76.4/56.4 (tests/test_sw_three_body_counting.py).
                        energy += h3
                        hh = 1e-6
                        dz_dposj = np.zeros(3)
                        dz_dposk = np.zeros(3)
                        for d_ in range(3):
                            rp = rij_v.copy()
                            rp[d_] += hh
                            rm = rij_v.copy()
                            rm[d_] -= hh
                            dz_dposj[d_] = (_h3_term(si, rp, rik_v, tri)
                                            - _h3_term(si, rm, rik_v, tri)) / (2 * hh)
                        for d_ in range(3):
                            rp = rik_v.copy()
                            rp[d_] += hh
                            rm = rik_v.copy()
                            rm[d_] -= hh
                            dz_dposk[d_] = (_h3_term(si, rij_v, rp, tri)
                                            - _h3_term(si, rij_v, rm, tri)) / (2 * hh)
                        Fj = -dz_dposj
                        Fk = -dz_dposk
                        Fi = dz_dposj + dz_dposk
                        forces[i] += Fi
                        forces[j_at] += Fj
                        forces[k_at] += Fk
                        stress += np.outer(Fj, rij_v) + np.outer(Fk, rik_v)

        # Bug-fix note: same `_mpi_allreduce_sum_scalar` fix as the
        # legacy single-species path above (see that path's `compute()`
        # for the full explanation) -- the multi-species path had an
        # identical raw-`_mpi_comm.allreduce()` bug for `energy`.
        if _HAS_MPI:
            energy = _mpi_allreduce_sum_scalar(energy)
            forces = _mpi_allreduce_sum(forces)
            stress = _mpi_allreduce_sum(stress)
        stress /= volume
        return energy, forces, stress
