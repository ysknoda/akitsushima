"""Lattice thermal conductivity from third-order force constants in the
relaxation-time approximation (ADR-0042).

Pipeline
--------
1. fc2: ``PhononCalculator`` on the fc2 supercell (or a saved ifc.npz).
2. fc3: two-step finite displacements on the fc3 supercell.  The first
   displacements are those of the ADR-0040 symmetry plan (one atom per
   orbit, Cartesian directions until the site-symmetry images span 3D,
   -u skipped when it is an image).  For each of them, every supercell
   atom is displaced again by +/- delta, giving the second-order force
   constants of the displaced cell; their difference along the first
   displacement is fc3.  The images of every computed slab under the
   space group fill the other atoms of the orbit through per-atom normal
   equations (as ADR-0040 does for fc2), so the closure under u -> -u
   makes the first difference central.
3. fc3 is symmetrised over the six index permutations and the acoustic
   sum rule sum_k Phi(i, j, k) = 0 is imposed (``sym_fc3``).
4. On a Gamma-centred mesh: frequencies, eigenvectors and group
   velocities everywhere; at the irreducible q points the imaginary part
   of the self energy (Togo, Chaput and Tanaka, PRB 91, 094306 (2015);
   the normalisation of phono3py):

       gamma_lambda = 18 pi / hbar^2  sum_{lambda' lambda''} |Phi|^2
           [ (n' + n'' + 1) delta(w - w' - w'')
             + (n' - n'') (delta(w + w' - w'') - delta(w - w' + w'')) ]

   |Phi_{-lambda lambda' lambda''}|^2 = (hbar / 2)^3 / 36 / N
       |V|^2 / (w w' w''),
   V = sum fc3(0k, l'k', l''k'') e*(k, q) e(k', q') e(k'', q'')
       / sqrt(m m' m'') <exp(i q'.R')> <exp(i q''.R'')>,
   with q' + q'' = q + G and a Gaussian delta of width sigma (THz).
   The phase uses lattice translations only, as ``PhononCalculator``'s
   dynamical matrix does, averaged over the minimum images of each atom
   seen from the first one.
5. RTA: tau = 1 / (2 Gamma) = 1 / (4 pi gamma) with gamma in THz;
   kappa = 1 / (N V) sum_lambda C_lambda v (x) v tau_lambda, symmetrised
   over the point group.

Optional scattering: mass disorder (Tamura, PRB 27, 858 (1983)),
gamma_iso = pi f^2 / (4 N) sum g_k |e*(k).e'(k)|^2 delta(f - f'), and
boundary scattering 1 / tau_b = |v| / L.

Units: fc3 eV/A^3, masses amu, frequencies THz, kappa W/m/K.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

from ..utils.jit import _HAS_NUMBA
from ..utils.mpi import _mpi_allreduce_sum, _mpi_domain_split, _mpi_rank, local_only
from .phonon import PhononCalculator

__all__ = ["BTECalculator", "mass_variance"]

# -- physical constants (SI 2019 exact where defined) ---------------------
_HBAR = 1.054571817e-34  # J s
_KB = 1.380649e-23  # J/K
_H = 6.62607015e-34  # J s
_EV = 1.602176634e-19  # J
_AMU = 1.66053906660e-27  # kg
_ANG = 1e-10  # m
_THZ = 1e12
# eigenvalue of D [eV / (A^2 amu)] -> omega^2 [rad^2 / s^2]
_W2_SI = _EV / (_ANG**2 * _AMU)
# gamma [THz] = _GAMMA_UNIT * sum |V|^2 / (f f' f'') * delta_f, with V in
# eV / A^3 / amu^(3/2), f in THz and delta_f in 1/THz (derivation in the
# module docstring; the same constant as phono3py's two unit factors).
_GAMMA_UNIT = (
    18.0
    * np.pi
    * _HBAR
    / (8.0 * 36.0)
    * (_EV / _ANG**3) ** 2
    / _AMU**3
    / (2.0 * np.pi * _THZ) ** 5
)
_CUTOFF_THZ = 1e-4  # modes below this frequency carry no kappa (phono3py)


def mass_variance(masses, fractions) -> float:
    """g = sum_i f_i (1 - m_i / m_bar)^2 with m_bar = sum_i f_i m_i."""
    m = np.asarray(masses, dtype=float)
    f = np.asarray(fractions, dtype=float)
    f = f / f.sum()
    mbar = float(f @ m)
    return float(f @ (1.0 - m / mbar) ** 2)


# ======================================================================
#  numba kernels
# ======================================================================


def _dD_dq_numpy(IFC, masses, sc_to_prim, img_R, img_n, origin, q_cart, N):
    """d D_raw / d q_alpha (3, 3N, 3N): the derivative of
    PhononCalculator's _build_D_raw_numpy, i R_alpha inside the phase."""
    N_sc = sc_to_prim.shape[0]
    dD = np.zeros((3, N * 3, N * 3), dtype=np.complex128)
    for ia_loc in range(N):
        ia_glob = origin[ia_loc]
        mi = masses[ia_loc]
        for j in range(N_sc):
            ib = sc_to_prim[j]
            n_im = img_n[ia_loc, j]
            ph0 = complex(0.0, 0.0)
            ph1 = complex(0.0, 0.0)
            ph2 = complex(0.0, 0.0)
            for k in range(n_im):
                R0 = img_R[ia_loc, j, k, 0]
                R1 = img_R[ia_loc, j, k, 1]
                R2 = img_R[ia_loc, j, k, 2]
                dot = q_cart[0] * R0 + q_cart[1] * R1 + q_cart[2] * R2
                e = complex(-np.sin(dot), np.cos(dot))  # i exp(i q.R)
                ph0 += e * R0
                ph1 += e * R1
                ph2 += e * R2
            s = 1.0 / (n_im * np.sqrt(mi * masses[ib]))
            for di in range(3):
                for dj in range(3):
                    v = IFC[ia_glob * 3 + di, j * 3 + dj] * s
                    dD[0, ia_loc * 3 + di, ib * 3 + dj] += v * ph0
                    dD[1, ia_loc * 3 + di, ib * 3 + dj] += v * ph1
                    dD[2, ia_loc * 3 + di, ib * 3 + dj] += v * ph2
    return dD


def _gamma_kernel_numpy(
    f0,
    e0c,
    freqs,
    evecs,
    q2_of,
    bk,
    bj,
    bkk,
    bval,
    phase,
    basis,
    inv_sqrt_m,
    occ,
    sigma,
    cutoff,
    unit,
    plus_one,
):
    """Imaginary self energy (gamma, THz) of the bands of one q point,
    summed over all q' of the mesh (q'' = q - q').

    f0 (nb,), e0c (3n, nb) = conj(e(q)); freqs (Nq, nb); evecs
    (Nq, 3n, nb); q2_of (Nq,); fc3 blocks: first atom bk (origin basis),
    supercell atoms bj, bkk, values bval (nbl, 27); phase (Nq, n, N_sc)
    = <exp(i q.R)> of supercell atom j seen from origin atom kappa;
    occ (nT, Nq, nb).  Returns (Nq, nT, nb): one row per q', summed by
    the caller (keeps the parallel loop free of races).
    """
    Nq, nb = freqs.shape
    n = inv_sqrt_m.shape[0]
    n3 = 3 * n
    nT = occ.shape[0]
    nbl = bk.shape[0]
    out = np.zeros((Nq, nT, nb))
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    inv2s2 = 0.5 / (sigma * sigma)
    e0t = np.ascontiguousarray(e0c.T)  # (s0, x)
    for g1 in _prange(Nq):
        g2 = q2_of[g1]
        # (q', q'') and (q'', q') contribute equally: compute g1 <= g2 only
        if g1 > g2:
            continue
        wq = 1.0 if g1 == g2 else 2.0
        F = np.zeros((n3, n3, n3), dtype=np.complex128)
        for b in range(nbl):
            ka = bk[b]
            j = bj[b]
            k = bkk[b]
            kb = basis[j]
            kc = basis[k]
            p = phase[g1, ka, j] * phase[g2, ka, k]
            p = p * (inv_sqrt_m[ka] * inv_sqrt_m[kb] * inv_sqrt_m[kc])
            for a in range(3):
                for be in range(3):
                    for c in range(3):
                        F[3 * ka + a, 3 * kb + be, 3 * kc + c] += (
                            bval[b, 9 * a + 3 * be + c] * p
                        )
        # V[s0, s1, s2] = sum_xyz F[x, y, z] e0c[x, s0] e1[y, s1] e2[z, s2],
        # as three matrix products (BLAS through numba's np.dot).
        T1 = np.dot(F.reshape(n3 * n3, n3), evecs[g2])        # (x y, s2)
        T1 = T1.reshape(n3, n3, nb)
        T2 = np.empty((n3, nb, nb), dtype=np.complex128)
        e1t = np.ascontiguousarray(evecs[g1].T)                # (s1, y)
        for x in range(n3):
            T2[x] = np.dot(e1t, np.ascontiguousarray(T1[x]))   # (s1, s2)
        V = np.dot(e0t, T2.reshape(n3, nb * nb)).reshape(nb, nb, nb)
        for s1 in range(nb):
            f1 = freqs[g1, s1]
            if f1 < cutoff:
                continue
            for s2 in range(nb):
                f2 = freqs[g2, s2]
                if f2 < cutoff:
                    continue
                for s0 in range(nb):
                    if f0[s0] < cutoff:
                        continue
                    v = V[s0, s1, s2]
                    w = wq * (v.real * v.real + v.imag * v.imag) * unit / (
                        f0[s0] * f1 * f2)
                    d1 = f0[s0] - f1 - f2
                    d2 = f0[s0] + f1 - f2
                    d3 = f0[s0] - f1 + f2
                    g_1 = norm * np.exp(-d1 * d1 * inv2s2)
                    g_23 = norm * (
                        np.exp(-d2 * d2 * inv2s2) - np.exp(-d3 * d3 * inv2s2)
                    )
                    for t in range(nT):
                        n1 = occ[t, g1, s1]
                        n2 = occ[t, g2, s2]
                        out[g1, t, s0] += w * ((n1 + n2 + plus_one) * g_1 + (n1 - n2) * g_23)
    return out


if _HAS_NUMBA:
    import numba as _nb

    _prange = _nb.prange
    _dD_dq = _nb.njit(cache=True)(_dD_dq_numpy)
    _gamma_kernel = _nb.njit(parallel=True, cache=True)(_gamma_kernel_numpy)
else:  # pragma: no cover - numba is a core dependency
    _prange = range
    _dD_dq = _dD_dq_numpy
    _gamma_kernel = _gamma_kernel_numpy


# ======================================================================
#  Calculator
# ======================================================================


class BTECalculator:
    """Lattice thermal conductivity in the RTA (ADR-0042).

    Parameters
    ----------
    lattice, positions, atom_types : the unit cell (rows = vectors, A).
    force_fn : the akitsushima force callable.
    supercell : diagonal fc3 supercell (int or 3 ints).
    supercell_fc2 : diagonal fc2 supercell; default ``supercell``.
    delta : fc3 displacement amplitude [A] (both steps).
    delta_fc2 : fc2 displacement amplitude [A] (PhononCalculator's).
    masses : per-atom masses [amu]; default from ATOMIC_MASS.
    symmetry, symprec : ADR-0040 reduction of the displacements and the
        irreducible q points; on by default here.
    cutoff_pair : [A] second displacements farther than this from the
        first displaced atom (minimum image) are skipped; their fc3 is 0.
    sym_fc3 : symmetrise fc3 over index permutations and impose the
        acoustic sum rule.
    fc2_calc : an existing PhononCalculator with IFC (e.g. loaded from
        ifc.npz); then supercell_fc2 / delta_fc2 are not used.
    """

    def __init__(
        self,
        lattice,
        positions,
        atom_types,
        force_fn: Callable,
        supercell=2,
        supercell_fc2=None,
        delta: float = 0.03,
        delta_fc2: float = 0.01,
        masses=None,
        symmetry: bool = True,
        symprec: float = 1e-5,
        cutoff_pair: float | None = None,
        sym_fc3: bool = True,
        fc2_calc: PhononCalculator | None = None,
    ):
        self.lattice = np.asarray(lattice, dtype=float).copy()
        self.positions = np.asarray(positions, dtype=float).copy()
        self.atom_types = list(atom_types)
        self.force_fn = force_fn
        self.delta = float(delta)
        self.symmetry = bool(symmetry)
        self.symprec = float(symprec)
        self.cutoff_pair = None if cutoff_pair is None else float(cutoff_pair)
        self.sym_fc3 = bool(sym_fc3)
        self.n_basis = len(self.atom_types)
        kw = {"masses": masses, "symmetry": self.symmetry, "symprec": self.symprec}
        self.ph3 = PhononCalculator(
            self.lattice,
            self.positions,
            self.atom_types,
            force_fn,
            supercell=supercell,
            delta=self.delta,
            **kw,
        )
        if fc2_calc is not None:
            self.ph2 = fc2_calc
        else:
            self.ph2 = PhononCalculator(
                self.lattice,
                self.positions,
                self.atom_types,
                force_fn,
                supercell=supercell if supercell_fc2 is None else supercell_fc2,
                delta=delta_fc2,
                **kw,
            )
        self.masses = np.asarray(self.ph2.masses, dtype=float)
        self.fc3: np.ndarray | None = None  # (n_basis, N_sc, N_sc, 3, 3, 3)
        self.fc3_info: dict = {}
        self.mass_variances = np.zeros(self.n_basis)
        self.boundary_length_m: float | None = None

    # ------------------------------------------------------------------
    #  fc2 / fc3
    # ------------------------------------------------------------------

    def compute_fc2(self, n_jobs: int = 1, verbose: bool = False) -> None:
        if self.ph2.IFC is None:
            self.ph2.compute_ifc(n_jobs=n_jobs, verbose=verbose)

    def _first_displacements(self, verbose: bool):
        """(plan, tasks): the ADR-0040 plan of the fc3 supercell, or an
        identity-only plan with +/- delta along x, y, z of every origin
        atom when symmetry is off or leaves only the identity."""
        ph = self.ph3
        plan = ph._symmetry_plan(verbose) if self.symmetry else None
        if plan is not None:
            return plan, plan["tasks"]
        cells = np.asarray(ph._sc_cell, dtype=np.int64)
        basis = np.asarray(ph._sc_to_prim, dtype=np.int64)
        origin = ph._origin_indices()
        tasks = [
            (j, s * self.delta * np.eye(3)[a])
            for j in origin
            for a in range(3)
            for s in (1.0, -1.0)
        ]

        class _Id:
            n_ops = 1
            rot_cart = np.eye(3)[None]

        plan = {
            "ops": _Id(),
            "sc_perm": np.arange(ph.N_sc)[None],
            "cells": cells,
            "basis": basis,
            "origin_of": {int(basis[j]): j for j in origin},
        }
        return plan, tasks

    def _second_tasks(self, i0: int) -> list[int]:
        """Supercell atoms displaced second for a first atom i0."""
        ph = self.ph3
        if self.cutoff_pair is None:
            return list(range(ph.N_sc))
        a = int(ph._sc_to_prim[i0])
        R = ph._img_R[a, :, 0]  # one minimum-image translation per atom
        d = ph._pos_sc - ph._pos_sc[i0]
        # minimum-image vector = position difference + (image R - own R)
        dmin = d + R - ph._R_sc
        dist = np.linalg.norm(dmin, axis=1)
        return [int(j) for j in np.flatnonzero(dist <= self.cutoff_pair + 1e-8)]

    def compute_fc3(self, n_jobs: int = 1, verbose: bool = False) -> np.ndarray:
        """Third-order force constants, compact form (n_basis, N_sc, N_sc,
        3, 3, 3): first index the origin-cell atom of each basis atom."""
        ph = self.ph3
        N_sc = ph.N_sc
        u = self.delta
        plan, first = self._first_displacements(verbose)
        # flat list of force calls: (first task, second atom, axis, sign)
        calls = []
        for t, (i0, _d) in enumerate(first):
            for j in self._second_tasks(i0):
                for b in range(3):
                    for s in (1.0, -1.0):
                        calls.append((t, j, b, s))
        n_calls = len(calls)
        if verbose and _mpi_rank == 0:
            print(
                f"  fc3: {len(first)} first displacements, {n_calls} "
                f"force calls (supercell {ph.supercell}, {N_sc} atoms, "
                f"delta {u} A)"
            )

        def _force(c):
            t, j, b, s = calls[c]
            i0, d = first[t]
            pos = ph._pos_sc.copy()
            pos[i0] += d
            pos[j, b] += s * u
            _, F, _ = self.force_fn(ph.lattice_sc, pos, ph._types_sc)
            return np.asarray(F, dtype=float)

        i_start, i_end = _mpi_domain_split(n_calls)
        # fc2 of each first-displaced cell: rows (j, beta), cols (k, gamma)
        fc2d = np.zeros((len(first), N_sc * 3, N_sc * 3))
        with local_only():
            idx = range(i_start, i_end)
            if n_jobs == 1:
                results = (_force(c) for c in idx)
            else:
                from concurrent.futures import ThreadPoolExecutor

                ex = ThreadPoolExecutor(max_workers=n_jobs)
                results = ex.map(_force, idx)
            for c, F in zip(idx, results):
                t, j, b, s = calls[c]
                fc2d[t, 3 * j + b] -= s * F.ravel() / (2.0 * u)
            if n_jobs != 1:
                ex.shutdown()
        fc2d = _mpi_allreduce_sum(fc2d)

        fc3 = self._reconstruct(plan, first, fc2d)
        self.fc3_info = {
            "n_first_displacements": len(first),
            "n_force_calls": n_calls,
            "n_force_calls_full": 6 * self.n_basis * 6 * N_sc,
            "symmetry": ph.symmetry_info,
            "cutoff_pair_A": self.cutoff_pair,
        }
        if self.sym_fc3:
            fc3, res = symmetrize_fc3(fc3, ph)
            self.fc3_info["asr_residual"] = res
        self.fc3 = fc3
        return fc3

    def _reconstruct(self, plan, first, fc2d) -> np.ndarray:
        """Normal equations per origin basis atom over all images:
        sum_a (R d)_a Phi3(i, a; j', k') = R fc2^(d)(j, k) R^T."""
        ph = self.ph3
        N_sc, n = ph.N_sc, self.n_basis
        ops, sc_perm = plan["ops"], plan["sc_perm"]
        cells, basis = plan["cells"], plan["basis"]
        S = np.asarray(ph.supercell, dtype=np.int64)
        A = np.zeros((n, 3, 3))
        B = np.zeros((n, 3, N_sc, 3, N_sc, 3))
        for t, (i0, d) in enumerate(first):
            M = fc2d[t].reshape(N_sc, 3, N_sc, 3)
            for k in range(ops.n_ops):
                R = ops.rot_cart[k]
                p = sc_perm[k]
                a_img = p[i0]
                p = ph._sc_index(basis[p], (cells[p] - cells[a_img]) % S)
                b_img = int(basis[a_img])
                d_img = R @ d
                Mr = np.einsum("ab,ibjc,dc->iajd", R, M, R, optimize=True)
                Mi = np.empty_like(Mr)
                Mi[np.ix_(p, np.arange(3), p, np.arange(3))] = Mr
                A[b_img] += np.outer(d_img, d_img)
                B[b_img] += d_img[:, None, None, None, None] * Mi[None]
        fc3 = np.zeros((n, N_sc, N_sc, 3, 3, 3))
        for b in range(n):
            sol = np.linalg.solve(A[b], B[b].reshape(3, -1)).reshape(
                3, N_sc, 3, N_sc, 3
            )
            fc3[b] = sol.transpose(1, 3, 0, 2, 4)
        return fc3

    # ------------------------------------------------------------------
    #  scattering options
    # ------------------------------------------------------------------

    def set_mass_variance(self, g_per_atom) -> None:
        """Tamura mass-variance parameter g of every basis atom."""
        g = np.asarray(g_per_atom, dtype=float)
        if g.shape != (self.n_basis,):
            raise ValueError("mass variance: one g per basis atom expected")
        self.mass_variances = g

    def set_boundary_length(self, length_um: float | None) -> None:
        """Casimir length L [micrometre]: 1 / tau_b = |v| / L."""
        self.boundary_length_m = None if length_um is None else float(length_um) * 1e-6

    # ------------------------------------------------------------------
    #  harmonic part on the mesh
    # ------------------------------------------------------------------

    def _mesh(self, mesh):
        """Grid addresses (spglib order), irreducible points and weights,
        point-group rotations (Cartesian)."""
        mesh = np.asarray(np.broadcast_to(mesh, 3), dtype=np.int64)
        n_q = int(np.prod(mesh))
        a0, a1, a2 = np.meshgrid(*[np.arange(m) for m in mesh], indexing="ij")
        # spglib order: index = a0 + m0 * (a1 + m1 * a2)
        addr = np.stack(
            [
                a0.transpose(2, 1, 0).ravel(),
                a1.transpose(2, 1, 0).ravel(),
                a2.transpose(2, 1, 0).ravel(),
            ],
            axis=1,
        )
        rots = np.eye(3)[None]
        if self.symmetry:
            from .symmetry import _spglib_numbers, require_spglib

            spglib = require_spglib()
            import warnings

            frac = self.positions @ np.linalg.inv(self.lattice)
            cell = (
                self.lattice,
                frac - np.floor(frac),
                _spglib_numbers(self.atom_types, self.masses),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                mapping, grid = spglib.get_ir_reciprocal_mesh(
                    mesh, cell, is_shift=[0, 0, 0], symprec=self.symprec
                )
                sym = spglib.get_symmetry(cell, symprec=self.symprec)
            grid = np.asarray(grid, dtype=np.int64)
            assert np.all(grid % mesh == addr), "spglib grid order changed"
            mapping = np.asarray(mapping, dtype=np.int64)
            ir, counts = np.unique(mapping, return_counts=True)
            LT = self.lattice.T
            rf = np.unique(np.asarray(sym["rotations"]).reshape(-1, 9), axis=0)
            rots = np.einsum(
                "ij,njk,kl->nil", LT, rf.reshape(-1, 3, 3), np.linalg.inv(LT)
            )
        else:
            ir = np.arange(n_q)
            counts = np.ones(n_q, dtype=np.int64)
        return mesh, addr, ir, counts, rots

    def _harmonic(self, q_cart_all):
        """Frequencies (THz), eigenvectors and D-derivatives on the mesh."""
        ph = self.ph2
        nb = 3 * self.n_basis
        Nq = len(q_cart_all)
        freqs = np.zeros((Nq, nb))
        evecs = np.zeros((Nq, nb, nb), dtype=np.complex128)
        for g, q in enumerate(q_cart_all):
            D = ph.build_dynamical_matrix(q)
            w2, e = np.linalg.eigh(D)
            freqs[g] = np.sign(w2) * np.sqrt(np.abs(w2) * _W2_SI) / (2 * np.pi * _THZ)
            evecs[g] = e
        return freqs, evecs

    def _dD(self, q):
        """dD/dq (3, 3n, 3n) [eV / (A amu)] at Cartesian q."""
        ph = self.ph2
        IFC = ph.IFC if ph._nac is None else ph._nac["IFC_SR"]
        origin = np.asarray(ph._origin_indices(), dtype=np.int64)
        dD = _dD_dq(
            IFC,
            ph.masses,
            np.asarray(ph._sc_to_prim, dtype=np.int64),
            ph._img_R,
            ph._img_n,
            origin,
            np.asarray(q, dtype=float),
            self.n_basis,
        )
        if ph._nac is not None and np.linalg.norm(q) > 1e-8:
            h = 1e-5
            for a in range(3):
                dq = np.zeros(3)
                dq[a] = h
                dD[a] += (ph._nac_dd(q + dq) - ph._nac_dd(q - dq)) / (2 * h)
        return 0.5 * (dD + dD.conj().transpose(0, 2, 1))

    # direction that fixes the basis of a degenerate subspace (phonopy)
    _DEG_DIR = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)

    def _velocities(self, q, f, e, little):
        """Group velocities (nb, 3) [m/s], as phonopy computes them.

        Within a degenerate set the basis is the one that diagonalises
        dD along (1, 2, 3) / sqrt(14); v_a is the diagonal of dD_a in that
        basis divided by 2 omega.  The result is averaged over the little
        group of q (Cartesian rotations ``little``), so that v (x) v of a
        degenerate mode is that of phono3py."""
        dD = self._dD(q) * (_W2_SI * _ANG)  # d(omega^2)/dq [s^-2 m]
        nb = len(f)
        v = np.zeros((nb, 3))
        s = 0
        while s < nb:
            t = s + 1
            while t < nb and abs(f[t] - f[s]) < 1e-4:
                t += 1
            if f[s] > _CUTOFF_THZ:
                w = 2 * np.pi * _THZ * f[s]
                E = e[:, s:t]
                Pn = E.conj().T @ np.einsum("a,axy->xy", self._DEG_DIR, dD) @ E
                _, U = np.linalg.eigh(Pn)
                Er = E @ U
                for a in range(3):
                    v[s:t, a] = np.real(np.einsum("xi,xy,yi->i", Er.conj(), dD[a], Er)) / (2 * w)
            s = t
        return np.einsum("rab,sb->sa", little, v) / len(little)

    # ------------------------------------------------------------------
    #  gamma and kappa
    # ------------------------------------------------------------------

    def _fc3_blocks(self):
        """Non-zero fc3 blocks (first basis atom, j, k, 27 values).  Blocks
        below 1e-10 of the largest |fc3| are dropped (round-off of forces on
        atoms the displacements do not reach)."""
        fc3 = self.fc3
        mx = np.abs(fc3).max(axis=(3, 4, 5))
        keep = np.argwhere(mx > 1e-10 * max(float(mx.max()), 1e-300))
        bval = fc3[keep[:, 0], keep[:, 1], keep[:, 2]].reshape(len(keep), 27)
        return (
            keep[:, 0].astype(np.int64),
            keep[:, 1].astype(np.int64),
            keep[:, 2].astype(np.int64),
            np.ascontiguousarray(bval),
        )

    def _phases(self, q_cart_all):
        """<exp(i q.R)> over the minimum images of supercell atom j seen
        from origin atom kappa, (Nq, n, N_sc)."""
        ph = self.ph3
        R, nim = ph._img_R, ph._img_n
        mask = np.arange(R.shape[2])[None, None, :] < nim[:, :, None]
        out = np.zeros((len(q_cart_all), self.n_basis, ph.N_sc), dtype=np.complex128)
        for g, q in enumerate(q_cart_all):
            e = np.exp(1j * (R @ q)) * mask
            out[g] = e.sum(axis=2) / nim
        return out

    def compute_kappa(
        self,
        mesh,
        sigma: float = 0.1,
        temperatures=(300.0,),
        occupation: str = "quantum",
        verbose: bool = False,
    ) -> dict:
        """kappa(T) in the RTA on a Gamma-centred mesh; sigma in THz."""
        if self.fc3 is None or self.ph2.IFC is None:
            raise RuntimeError("BTECalculator: compute fc2 and fc3 first")
        if occupation not in ("quantum", "classical"):
            raise ValueError("occupation must be 'quantum' or 'classical'")
        T = np.atleast_1d(np.asarray(temperatures, dtype=float))
        mesh, addr, ir, counts, rots = self._mesh(mesh)
        Nq = len(addr)
        B = 2 * np.pi * np.linalg.inv(self.lattice).T  # rows: reciprocal vectors
        q_frac = addr / mesh
        q_cart = q_frac @ B
        freqs, evecs = self._harmonic(q_cart)
        nb = freqs.shape[1]

        # q'' = q - q' on the grid
        def gindex(a):
            a = a % mesh
            return a[..., 0] + mesh[0] * (a[..., 1] + mesh[1] * a[..., 2])

        # occupations (nT, Nq, nb)
        fpos = np.where(freqs > _CUTOFF_THZ, freqs, np.inf)
        x = (_H * _THZ * fpos)[None] / (_KB * T[:, None, None])
        occ = (1.0 / x) if occupation == "classical" else 1.0 / np.expm1(x)
        occ = np.ascontiguousarray(np.where(np.isfinite(occ), occ, 0.0))
        bk, bj, bkk, bval = self._fc3_blocks()
        phase = self._phases(q_cart)
        basis = np.asarray(self.ph3._sc_to_prim, dtype=np.int64)
        inv_sqrt_m = 1.0 / np.sqrt(self.masses)
        V_m3 = abs(np.linalg.det(self.lattice)) * _ANG**3

        n_ir = len(ir)
        gam3 = np.zeros((n_ir, len(T), nb))
        giso = np.zeros((n_ir, nb))
        vel = np.zeros((n_ir, nb, 3))
        gvgv = np.zeros((n_ir, nb, 3, 3))
        for i, g in enumerate(ir):
            q2_of = gindex(addr[g][None] - addr).astype(np.int64)
            out = _gamma_kernel(
                freqs[g],
                np.ascontiguousarray(evecs[g].conj()),
                freqs,
                evecs,
                q2_of,
                bk,
                bj,
                bkk,
                bval,
                phase,
                basis,
                inv_sqrt_m,
                occ,
                float(sigma),
                _CUTOFF_THZ,
                _GAMMA_UNIT / Nq,
                # classical limit: the spontaneous-decay "+1" is O(hbar)
                0.0 if occupation == "classical" else 1.0,
            )
            gam3[i] = _degenerate_average(freqs[g], out.sum(axis=0))
            if np.any(self.mass_variances > 0):
                giso[i] = _degenerate_average(
                    freqs[g], self._gamma_iso(g, freqs, evecs, sigma)[None]
                )[0]
            vel[i] = self._velocities(q_cart[g], freqs[g], evecs[g],
                                      _little_group(rots, q_frac[g], self.lattice))
            gvgv[i] = vel[i][:, :, None] * vel[i][:, None, :]
            if verbose and _mpi_rank == 0 and (i + 1) % max(1, n_ir // 10) == 0:
                print(f"    gamma: {i + 1}/{n_ir} irreducible q points", flush=True)

        f_ir = freqs[ir]
        gam_b = np.zeros((n_ir, nb))
        if self.boundary_length_m is not None:
            gam_b = (
                np.linalg.norm(vel, axis=2)
                / self.boundary_length_m
                / (4 * np.pi * _THZ)
            )
        # (nT, n_ir, nb) from here on
        gam3 = gam3.transpose(1, 0, 2)
        gam_tot = gam3 + giso[None] + gam_b[None]
        ok = (f_ir > _CUTOFF_THZ)[None] & (gam_tot > 0)
        tau = np.where(ok, 1.0 / (4 * np.pi * _THZ * np.where(ok, gam_tot, 1.0)), 0.0)
        xx = (_H * _THZ * np.where(f_ir > _CUTOFF_THZ, f_ir, np.inf))[None] / (
            _KB * T[:, None, None]
        )
        if occupation == "classical":
            cv = np.where(np.isfinite(xx), _KB, 0.0)
        else:
            ex = np.exp(np.where(np.isfinite(xx), xx, 0.0))
            cv = np.where(
                np.isfinite(xx),
                _KB * xx**2 * ex / np.expm1(np.where(np.isfinite(xx), xx, 1.0)) ** 2,
                0.0,
            )
        # mode kappa (nT, n_ir, nb, 3, 3), weighted by the star size
        kmode = (cv * tau)[..., None, None] * gvgv[None] / (Nq * V_m3)
        kraw = np.einsum("i,tisab->tab", counts.astype(float), kmode)
        kappa = np.einsum("rax,txy,rby->tab", rots, kraw, rots) / len(rots)
        self.result = {
            "temperatures_K": T,
            "kappa_W_mK": kappa,
            "mesh": mesh.tolist(),
            "sigma_THz": float(sigma),
            "occupation": occupation,
            "n_irreducible_q": int(n_ir),
            "weights": counts,
            "q_frac_ir": q_frac[ir],
            "frequencies_THz": f_ir,
            "group_velocity_m_s": vel,
            "gamma_THz": gam3,
            "gamma_isotope_THz": giso,
            "gamma_boundary_THz": gam_b,
            "tau_s": tau,
            "heat_capacity_J_K": cv,
            "mode_kappa_W_mK": kmode * counts[None, :, None, None, None],
        }
        return self.result

    def _gamma_iso(self, g, freqs, evecs, sigma):
        """Tamura gamma (THz) of the bands at grid point g."""
        Nq, nb = freqs.shape
        n = self.n_basis
        f0 = freqs[g]
        e0 = evecs[g].reshape(n, 3, nb)
        e = evecs.reshape(Nq, n, 3, nb)
        # overlap[q', s0, s', kappa] = sum_alpha conj(e0) e'
        ov = np.einsum("kas,qkat->qstk", e0.conj(), e)
        w = np.einsum("qstk,k->qst", np.abs(ov) ** 2, self.mass_variances)
        d = f0[None, :, None] - freqs[:, None, :]
        delta = np.exp(-0.5 * (d / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        valid = (freqs > _CUTOFF_THZ)[:, None, :]
        s = np.sum(w * delta * valid, axis=(0, 2))
        out = np.pi * f0**2 / (4 * Nq) * s
        return np.where(f0 > _CUTOFF_THZ, out, 0.0)

    # ------------------------------------------------------------------
    #  output
    # ------------------------------------------------------------------

    def save_results(self, output_dir: str, settings: dict | None = None) -> None:
        """bte_results.json, bte_modes.npz, cumulative CSVs, plots."""
        import json

        r = self.result
        os.makedirs(output_dir, exist_ok=True)
        T = r["temperatures_K"]
        out = {
            "temperatures_K": T.tolist(),
            "kappa_W_mK": r["kappa_W_mK"].tolist(),
            "kappa_trace_over_3_W_mK": [
                float(np.trace(k) / 3) for k in r["kappa_W_mK"]
            ],
            "mesh": r["mesh"],
            "sigma_THz": r["sigma_THz"],
            "occupation": r["occupation"],
            "n_irreducible_q": r["n_irreducible_q"],
            "fc3": self.fc3_info,
            "mass_variances": self.mass_variances.tolist(),
            "boundary_length_um": (
                None if self.boundary_length_m is None else self.boundary_length_m * 1e6
            ),
        }
        if settings:
            out["settings"] = settings
        with open(os.path.join(output_dir, "bte_results.json"), "w") as fh:
            json.dump(out, fh, indent=2, default=_json_default)
        np.savez(
            os.path.join(output_dir, "bte_modes.npz"),
            **{k: v for k, v in r.items() if isinstance(v, np.ndarray)},
        )
        # cumulative kappa (trace / 3) vs frequency and vs mean free path
        kt = np.einsum("tisaa->tis", r["mode_kappa_W_mK"]) / 3.0
        f = np.broadcast_to(r["frequencies_THz"], kt.shape[1:]).ravel()
        mfp = np.linalg.norm(r["group_velocity_m_s"], axis=2)[None] * r["tau_s"]
        for name, key in (("frequency_THz", None), ("mfp_m", mfp)):
            rows = []
            for it in range(len(T)):
                x = f if key is None else key[it].ravel()
                o = np.argsort(x)
                rows.append((x[o], np.cumsum(kt[it].ravel()[o])))
            path = os.path.join(output_dir, f"bte_cumulative_{name.split('_')[0]}.csv")
            with open(path, "w") as fh:
                fh.write(f"T_K,{name},kappa_cum_W_mK\n")
                for it, (x, c) in enumerate(rows):
                    fh.writelines(f"{T[it]:g},{xi:.6e},{ci:.6e}\n" for xi, ci in zip(x, c))
        self._plot(output_dir, kt, mfp)

    def _plot(self, output_dir, kt, mfp) -> None:
        from ..utils.optional_matplotlib import have_matplotlib, plt

        if not have_matplotlib():
            return
        r = self.result
        T = r["temperatures_K"]
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        k = r["kappa_W_mK"]
        for a, lab in enumerate("xyz"):
            ax[0].plot(T, k[:, a, a], "o-", label=f"kappa_{lab}{lab}")
        ax[0].set_xlabel("T (K)")
        ax[0].set_ylabel("kappa (W/m/K)")
        ax[0].legend()
        it = int(np.argmin(np.abs(T - 300.0)))
        ax[1].semilogy(
            r["frequencies_THz"].ravel(), r["gamma_THz"][it].ravel(), ".", ms=3
        )
        ax[1].set_xlabel("frequency (THz)")
        ax[1].set_ylabel(f"gamma (THz), T = {T[it]:g} K")
        x = mfp[it].ravel() * 1e9
        o = np.argsort(x)
        ax[2].semilogx(np.maximum(x[o], 1e-3), np.cumsum(kt[it].ravel()[o]))
        ax[2].set_xlabel("mean free path (nm)")
        ax[2].set_ylabel("cumulative kappa (W/m/K)")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "bte_kappa.png"), dpi=120)
        plt.close(fig)


# ======================================================================
#  helpers
# ======================================================================


def _little_group(rots, q_frac, lattice):
    """Cartesian rotations R of the point group with R q = q exactly, q
    taken in the zone as q - rint(q) (phonopy's site symmetry of a q
    point: an operation that maps q onto q + G, G != 0, is not used, so
    on an even mesh the zone-boundary points average over fewer
    operations, as in phono3py)."""
    B = 2 * np.pi * np.linalg.inv(lattice).T
    qf = np.asarray(q_frac, dtype=float)
    q = (qf - np.rint(qf)) @ B
    Binv = np.linalg.inv(B)
    keep = [R for R in rots if np.all(np.abs((R @ q - q) @ Binv) < 1e-8)]
    return np.array(keep)


def _degenerate_average(f, g):
    """Average the last axis of g over degenerate bands of f (THz)."""
    g = np.array(g, dtype=float)
    nb = len(f)
    s = 0
    while s < nb:
        t = s + 1
        while t < nb and abs(f[t] - f[s]) < 1e-4:
            t += 1
        g[..., s:t] = g[..., s:t].mean(axis=-1, keepdims=True)
        s = t
    return g


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def _translation_maps(ph: PhononCalculator) -> np.ndarray:
    """T[x, y]: the supercell atom y shifted by minus the cell of atom x."""
    S = np.asarray(ph.supercell, dtype=np.int64)
    cells = np.asarray(ph._sc_cell, dtype=np.int64)
    basis = np.asarray(ph._sc_to_prim, dtype=np.int64)
    T = np.empty((ph.N_sc, ph.N_sc), dtype=np.int64)
    for x in range(ph.N_sc):
        T[x] = ph._sc_index(basis, (cells - cells[x]) % S)
    return T


def symmetrize_fc3(
    fc3: np.ndarray, ph: PhononCalculator, n_iter: int = 200, tol: float = 1e-10
):
    """Index-permutation symmetrisation and the acoustic sum rule
    sum_k Phi(i, j, k) = 0, alternated until the sum rule holds to tol
    (relative to max |fc3|) or n_iter rounds.

    The sum-rule residual of each (i, j) is put on the k = i block, so the
    support of fc3 stays that of the pairs it already has (no uniform
    spreading over the supercell); the next symmetrisation distributes it
    over the permutations.  The residual falls geometrically (Si, 64 atoms:
    1.7e-7 after 20 rounds, 8e-14 after 100).  Returns (fc3, final
    relative residual)."""
    basis = np.asarray(ph._sc_to_prim, dtype=np.int64)
    origin = np.array(ph._origin_indices())  # origin atom of basis b
    T = _translation_maps(ph)
    scale = max(float(np.abs(fc3).max()), 1e-300)
    res = np.inf
    for _ in range(n_iter):
        fc3 = _permute_average(fc3, basis, origin, T)
        r = fc3.sum(axis=2)  # (n, N_sc, 3, 3, 3)
        res = float(np.abs(r).max()) / scale
        if res < tol:
            break
        for b in range(fc3.shape[0]):
            fc3[b, :, origin[b]] -= r[b]
    fc3 = _permute_average(fc3, basis, origin, T)
    res = float(np.abs(fc3.sum(axis=2)).max()) / scale
    return fc3, res


def _permute_average(fc3, basis, origin, T):
    """Average fc3 over the six permutations of (i, j, k), translating so
    that the first index is always an origin-cell atom."""
    n, N_sc = fc3.shape[0], fc3.shape[1]
    out = fc3 + fc3.transpose(0, 2, 1, 3, 5, 4)  # (i,j,k) + (i,k,j)
    for b in range(n):
        i = origin[b]
        for j in range(N_sc):
            bj = basis[j]
            Tj = T[j]
            ti, tk = Tj[i], Tj  # i and all k seen from j
            # (j, i, k): fc3[bj, T_j(i), T_j(k)][beta, alpha, gamma]
            out[b, j] += fc3[bj, ti, tk].transpose(0, 2, 1, 3)  # [k, b, a, c] -> [k, a, b, c]
            # (j, k, i): fc3[bj, T_j(k), T_j(i)][beta, gamma, alpha]
            out[b, j] += fc3[bj, tk, ti].transpose(0, 3, 1, 2)
    for b in range(n):
        i = origin[b]
        for k in range(N_sc):
            bk = basis[k]
            Tk = T[k]
            ti, tj = Tk[i], Tk
            # (k, i, j): fc3[bk, T_k(i), T_k(j)][gamma, alpha, beta]
            out[b, :, k] += fc3[bk, ti, tj].transpose(0, 2, 3, 1)
            # (k, j, i): fc3[bk, T_k(j), T_k(i)][gamma, beta, alpha]
            out[b, :, k] += fc3[bk, tj, ti].transpose(0, 3, 2, 1)
    return out / 6.0
