"""Volume-dependent phonon properties (ADR-0018 stage 2b): structures
relaxed at fixed volume, mode Grueneisen parameters from dD/dV, and the
quasi-harmonic approximation.

Conventions follow phonopy so that its PhonopyGruneisen / PhonopyQHA can
serve as test oracles: the same reduced q at every volume, gamma_qv =
-(V / 2 w^2) <e|dD/dV|e> with degenerate subspaces diagonalised, the
thermal expansion by central differences of V(T), and C_p from C_v and S
fitted as fourth-order polynomials in V.
"""
import contextlib
import io
from typing import Optional, Sequence

import numpy as np

from ..constants import EV_ANG3_TO_GPA
from .elastic import fit_equation_of_state
from .geometry_optimiser import GeometryOptimiser
from .phonon import (
    _KB_EV,
    _R_J_K_MOL,
    _THZ_TO_CM,
    PhononCalculator,
)

__all__ = ["build_supercell", "relax_at_volume", "phonons_at",
           "mode_gruneisen", "mean_gruneisen", "quasi_harmonic"]


def build_supercell(lattice, positions, atom_types, supercell):
    """(lattice_sc, positions_sc, types_sc), cell-major: atom k of cell c
    is index c * N + k, so the first N atoms are the primitive cell."""
    S = [int(s) for s in np.broadcast_to(supercell, 3)]
    lattice = np.asarray(lattice, dtype=float)
    pos, types = [], []
    for i in range(S[0]):
        for j in range(S[1]):
            for k in range(S[2]):
                pos.append(np.asarray(positions) + np.array([i, j, k]) @ lattice)
                types += list(atom_types)
    return np.diag(S) @ lattice, np.vstack(pos), types


def relax_at_volume(lattice, positions, atom_types, force_fn, volume,
                    supercell, relax_kw: Optional[dict] = None,
                    output_dir: str = ".") -> dict:
    """
    Scale the primitive cell isotropically to ``volume`` [A^3], build the
    supercell (wide enough for the cutoff), relax shape and ions at that
    fixed volume (GeometryOptimiser fix_volume), and return the primitive
    cell back.  ``volume=None``: a full (variable-volume) relaxation of
    the unscaled cell instead.

    Returns dict: lattice, positions (primitive), energy [eV per primitive
    cell], pressure_GPa (standard sign: positive under compression),
    converged, n_steps.
    """
    lattice = np.asarray(lattice, dtype=float)
    s = (1.0 if volume is None
         else (volume / abs(np.linalg.det(lattice))) ** (1.0 / 3.0))
    lat, pos = lattice * s, np.asarray(positions, dtype=float) * s
    L, P, T = build_supercell(lat, pos, atom_types, supercell)
    kw = dict(method="fire", fmax=1e-4, pmax=1e-3, max_steps=5000)
    kw.update(relax_kw or {})
    r = GeometryOptimiser(L, P, T, force_fn=force_fn, relax_cell=True,
                          fix_volume=volume is not None, verbose=False,
                          output_dir=output_dir, **kw).run()
    n_cell = int(np.prod(np.broadcast_to(supercell, 3)))
    n = len(atom_types)
    S = np.diag([int(x) for x in np.broadcast_to(supercell, 3)])
    lat_p = np.linalg.inv(S) @ r["lattice"]
    _, _, stress = force_fn(r["lattice"], r["positions"], T)
    return {"lattice": lat_p, "positions": r["positions"][:n].copy(),
            "energy": float(r["energy"]) / n_cell,
            # compressive-positive stress (ADR-0001): P = +tr / 3
            "pressure_GPa": float(np.trace(stress) / 3.0 * EV_ANG3_TO_GPA),
            "converged": bool(r["converged"]), "n_steps": int(r["n_steps"])}


def phonons_at(lattice, positions, atom_types, force_fn, supercell,
               delta=0.01, masses=None, n_jobs=1) -> PhononCalculator:
    """Force constants of one (relaxed) primitive cell."""
    with contextlib.redirect_stdout(io.StringIO()):
        pc = PhononCalculator(lattice, positions, atom_types, force_fn,
                              supercell=supercell, delta=delta, masses=masses)
        pc.compute_ifc(n_jobs=n_jobs, verbose=False)
    return pc


def mode_gruneisen(pcs: Sequence[PhononCalculator],
                   q_reduced: np.ndarray,
                   degeneracy_THz: float = 1e-4,
                   acoustic_THz: float = 1e-3) -> dict:
    """
    Mode Grueneisen parameters at reduced q-points from three volumes
    (V-, V0, V+): gamma_qv = -(V0 / 2 w_qv^2) <e_qv| dD/dV |e_qv>,
    dD/dV = (D(V+) - D(V-)) / (V+ - V-), the same reduced q at every
    volume, eigenvectors of D(V0).  Within a degenerate set (frequencies
    within ``degeneracy_THz``) dD/dV is diagonalised in that subspace and
    its eigenvalues assigned in ascending order, as phonopy does.  Modes
    with |f| <= ``acoustic_THz`` give NaN.

    Returns dict: frequencies_THz (n_q, n_modes) at V0, gamma (n_q, n_modes).
    """
    vols = [abs(np.linalg.det(pc.lattice_prim)) for pc in pcs]
    recs = [2.0 * np.pi * np.linalg.inv(pc.lattice_prim).T for pc in pcs]
    q_reduced = np.atleast_2d(np.asarray(q_reduced, dtype=float))
    n = 3 * pcs[1].N_basis
    freqs = np.zeros((len(q_reduced), n))
    gam = np.full((len(q_reduced), n), np.nan)
    for iq, q in enumerate(q_reduced):
        Dm, D0, Dp = (pc.build_dynamical_matrix(q @ r) for pc, r in zip(pcs, recs))
        dD = (Dp - Dm) / (vols[2] - vols[0])
        w2, e = np.linalg.eigh(D0)
        f = PhononCalculator._eig_to_freq(w2) / _THZ_TO_CM
        freqs[iq] = f
        i = 0
        while i < n:
            j = i + 1
            while j < n and abs(f[j] - f[i]) < degeneracy_THz:
                j += 1
            sub = e[:, i:j]
            dw2 = np.linalg.eigvalsh(sub.conj().T @ dD @ sub)
            for k in range(i, j):
                if abs(f[k]) > acoustic_THz:
                    gam[iq, k] = -vols[1] / (2.0 * w2[k]) * dw2[k - i]
            i = j
    return {"frequencies_THz": freqs, "gamma": gam}


def mean_gruneisen(freqs_THz, gamma, temperatures) -> np.ndarray:
    """C_v-weighted mean sum(gamma_i c_i) / sum(c_i) over modes with a
    finite gamma (NaN at T = 0)."""
    from .phonon import _THZ_TO_EV
    ok = np.isfinite(gamma) & (freqs_THz > 0)
    e, g = freqs_THz[ok] * _THZ_TO_EV, gamma[ok]
    out = []
    for t in np.atleast_1d(temperatures):
        if t <= 0:
            out.append(np.nan)
            continue
        x = e / (_KB_EV * t)
        nb = 1.0 / np.expm1(x)
        c = x * x * nb * (nb + 1.0)
        out.append(float(np.sum(g * c) / np.sum(c)) if np.sum(c) > 0 else np.nan)
    return np.array(out)


def quasi_harmonic(volumes, energies, temperatures, F_vib, S_vib, Cv_vib,
                   eos: str = "birch-murnaghan") -> dict:
    """
    Quasi-harmonic thermodynamics from E(V) and the harmonic F, S, C_v at
    each volume (ADR-0018).

    Parameters
    ----------
    volumes      : (n_V,) [A^3], energies (n_V,) [eV], per primitive cell.
    temperatures : (n_T,) [K], uniformly spaced.
    F_vib        : (n_T, n_V) [eV]; S_vib, Cv_vib (n_T, n_V) [k_B].
    eos          : 'birch-murnaghan' or 'vinet'.

    Returns dict (arrays over the temperatures kept): temperatures, V_A3,
    G_eV (minimum of F(V, T)), B_GPa, Bp, alpha_per_K (central difference,
    NaN at both ends), Cv_J_K_mol and S_J_K_mol at V(T) (fourth-order
    polynomials in V), Cp_J_K_mol = C_v + T (dV/dT) (dS/dV),
    gamma (= alpha B V / C_v), and t_max_K: the highest temperature whose
    V(T) lies inside the sampled volume range -- no point above it is
    returned (no extrapolation).
    """
    V = np.asarray(volumes, dtype=float)
    T = np.asarray(temperatures, dtype=float)
    E = np.asarray(energies, dtype=float)
    fit = []
    for k in range(len(T)):
        r = fit_equation_of_state(V, E + np.asarray(F_vib)[k], eos=eos)
        if not (V.min() < r["V0"] < V.max()):
            break
        fit.append(r)
    nT = len(fit)
    if nT == 0:
        raise ValueError("quasi_harmonic: the free-energy minimum is outside "
                         "the sampled volumes already at the lowest "
                         "temperature; widen the volume range.")
    T = T[:nT]
    Veq = np.array([r["V0"] for r in fit])
    B = np.array([r["B0"] for r in fit])
    alpha = np.full(nT, np.nan)
    dVdT = np.full(nT, np.nan)
    for i in range(1, nT - 1):
        dVdT[i] = (Veq[i + 1] - Veq[i - 1]) / (T[i + 1] - T[i - 1])
        alpha[i] = dVdT[i] / Veq[i]
    cv = np.array([np.polyval(np.polyfit(V, np.asarray(Cv_vib)[i], 4), Veq[i])
                   for i in range(nT)])
    s = np.array([np.polyval(np.polyfit(V, np.asarray(S_vib)[i], 4), Veq[i])
                  for i in range(nT)])
    dsdv = np.array([np.polyval(np.polyder(np.polyfit(V, np.asarray(S_vib)[i], 4)),
                                Veq[i]) for i in range(nT)])
    cp = cv + T * dVdT * dsdv
    # gamma = alpha B V / C_v with B V in eV (GPa A^3 / 160.2) and C_v in eV/K
    gamma = alpha * (B / EV_ANG3_TO_GPA) * Veq / (cv * _KB_EV)
    return {"temperatures": T, "V_A3": Veq,
            "G_eV": np.array([r["E0"] for r in fit]),
            "B_GPa": B, "Bp": np.array([r["B0_prime"] for r in fit]),
            "alpha_per_K": alpha, "Cv_J_K_mol": cv * _R_J_K_MOL,
            "S_J_K_mol": s * _R_J_K_MOL, "Cp_J_K_mol": cp * _R_J_K_MOL,
            "gamma": gamma, "t_max_K": float(T[-1]),
            "truncated": nT < len(np.asarray(temperatures))}
