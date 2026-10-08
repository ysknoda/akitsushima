"""Born effective charges, the Gonze-Lee dipole-dipole term (NAC), the
static dielectric tensor, infrared mode analysis and the piezoelectric
tensor (ADR-0019).

Units: charges in e, positions in A, masses in amu, force constants in
eV/A^2.  e^2 / (4 pi eps_0) = K_E = 14.3996 eV A, so the Gaussian-unit
formulas (Gonze and Lee, Phys. Rev. B 55, 10355 (1997)) carry K_E where
they have e^2.
"""
from typing import Optional, Sequence

import numpy as np

from ..constants import K_E

__all__ = ["born_from_config", "dipole_dipole", "dielectric_response",
           "piezoelectric_tensor"]

_E_C   = 1.602176634e-19        # C
_EPS0  = 8.8541878128e-12       # F/m (CODATA 2018)


def _as_tensor(v, what: str) -> np.ndarray:
    """Scalar -> s*I, length-3 -> diag, 3x3 -> as is."""
    a = np.asarray(v, dtype=float)
    if a.ndim == 0:
        return float(a) * np.eye(3)
    if a.shape == (3,):
        return np.diag(a)
    if a.shape == (3, 3):
        return a.copy()
    raise ValueError(f"{what}: give a scalar, 3 diagonal values or a 3x3 "
                     f"tensor, got shape {a.shape}.")


def born_from_config(born_cfg: Optional[dict], atom_types: Sequence[str],
                     charges: Optional[dict] = None) -> Optional[dict]:
    """
    Z* (N, 3, 3) and eps_inf (3, 3) from the YAML ``born:`` block
    (ADR-0019, Q12, Q13, Q22):

        born:
          epsilon_inf: 4.0 | [ex, ey, ez] | [[...], [...], [...]]
          born_charges:
            per_species: {Zn: 2.1, O: [..3..] or 3x3}
            per_atom:    {4: 3x3, ...}        # 1-based POSCAR index; wins
          clamped_ion_piezo: 3x6 (C/m^2)      # optional, for piezo

    Missing pieces fall back to the potential's point charges, Z* = q I and
    eps_inf = 1 (the rigid-ion model: no electronic polarisation; LO-TO
    splitting and eps_0 are then those of that model, warned about by the
    caller).  None when there is neither a born block nor charges.

    The charge-neutrality sum rule sum_k Z*_k = 0 is imposed by removing
    the mean (per Cartesian component); the largest removed component is
    returned as ``sum_rule_correction``.
    """
    born_cfg = born_cfg or {}
    n = len(atom_types)
    bc = born_cfg.get("born_charges") or {}
    if not bc and charges is None:
        return None
    Z = np.zeros((n, 3, 3))
    src = []
    per_sp = bc.get("per_species") or {}
    per_at = {int(k): v for k, v in (bc.get("per_atom") or {}).items()}
    for i, t in enumerate(atom_types):
        if (i + 1) in per_at:
            Z[i] = _as_tensor(per_at[i + 1], f"born_charges.per_atom[{i + 1}]")
            src.append("per_atom")
        elif t in per_sp:
            Z[i] = _as_tensor(per_sp[t], f"born_charges.per_species[{t}]")
            src.append("per_species")
        elif charges is not None and t in charges:
            Z[i] = float(charges[t]) * np.eye(3)
            src.append("point_charge")
        else:
            raise ValueError(f"born: no Born charge for atom {i + 1} ({t}) -- "
                             f"give it under born_charges.")
    bad = [k for k in per_at if not 1 <= k <= n]
    if bad:
        raise ValueError(f"born_charges.per_atom: indices {bad} outside 1..{n}.")
    mean = Z.mean(axis=0)
    Z -= mean
    eps_given = "epsilon_inf" in born_cfg
    eps = (_as_tensor(born_cfg["epsilon_inf"], "epsilon_inf") if eps_given
           else np.eye(3))
    if np.any(np.linalg.eigvalsh(0.5 * (eps + eps.T)) < 1.0 - 1e-12):
        raise ValueError("born.epsilon_inf must be >= 1 in every direction.")
    e0 = born_cfg.get("clamped_ion_piezo")
    return {"Z": Z, "epsilon_inf": eps, "sources": src,
            "epsilon_inf_given": eps_given,
            "rigid_ion_default": (not eps_given) or ("point_charge" in src),
            "sum_rule_correction": float(np.max(np.abs(mean))),
            "clamped_ion_piezo": (None if e0 is None
                                  else np.asarray(e0, dtype=float).reshape(3, 6))}


def g_vectors(lattice: np.ndarray, n_points: int = 300):
    """Reciprocal vectors (2 pi included) inside the sphere that holds about
    ``n_points`` of them, and its radius -- phonopy's choice
    (G_cutoff = (3 n / 4 pi / V)^(1/3), times 2 pi here)."""
    lattice = np.asarray(lattice, dtype=float)
    vol = abs(np.linalg.det(lattice))
    g_cut = 2.0 * np.pi * (3.0 * n_points / (4.0 * np.pi) / vol) ** (1.0 / 3.0)
    rec = 2.0 * np.pi * np.linalg.inv(lattice).T
    m = [int(np.ceil(g_cut * np.linalg.norm(a) / (2.0 * np.pi))) + 1
         for a in lattice]
    idx = np.array(np.meshgrid(*[np.arange(-k, k + 1) for k in m],
                               indexing="ij")).reshape(3, -1).T
    G = idx @ rec
    return G[np.linalg.norm(G, axis=1) < g_cut], g_cut


def dipole_dipole(q_cart, G, positions, Z, eps, volume, lam,
                  q_direction=None, drift=None) -> np.ndarray:
    """
    Reciprocal-space dipole-dipole force constants C(q) [eV/A^2], shape
    (N, 3, N, 3), in the cell gauge of PhononCalculator (phase from lattice
    translations only), Gonze and Lee (1997) eq. (71) without its
    real-space part (as phonopy's default):

        (4 pi K_E / V) sum_{G, K = q + G != 0} (K.Z_a)(K.Z_b)^T / (K.eps.K)
            exp(-K.eps.K / 4 Lambda^2) exp(i K.(tau_a - tau_b))
        - delta_ab  drift_a

    ``drift`` (N, 3, 3) is the q = 0 sum over b (it makes the term obey the
    acoustic sum rule); computed here when None.  K = 0 is skipped unless
    ``q_direction`` is given, when the non-analytic limit along it is added.
    """
    K = np.asarray(q_cart, dtype=float)[None, :] + G
    keep = np.linalg.norm(K, axis=1) > 1e-10
    K = K[keep]
    KZ = np.einsum("gi,aij->gaj", K, Z)
    den = np.einsum("gi,ij,gj->g", K, eps, K)
    w = np.exp(-den / (4.0 * lam * lam)) / den
    ph = np.exp(1j * K @ np.asarray(positions).T)          # (g, a)
    A = KZ * ph[:, :, None]
    C = np.einsum("g,gai,gbj->aibj", w, A, A.conj())
    if q_direction is not None and not np.all(keep):
        n = np.asarray(q_direction, dtype=float)
        n = n / np.linalg.norm(n)
        nz = np.einsum("i,aij->aj", n, Z)
        C += np.einsum("ai,bj->aibj", nz, nz) / float(n @ eps @ n)
    C *= 4.0 * np.pi * K_E / volume
    if drift is None:
        drift = dipole_dipole(np.zeros(3), G, positions, Z, eps, volume, lam,
                              drift=np.zeros((len(Z), 3, 3))).sum(axis=2)
    for a in range(len(Z)):
        C[a, :, a, :] -= drift[a]
    return C


def dielectric_response(D_gamma: np.ndarray, masses, Z, eps_inf, volume,
                        acoustic_tol: float = 1e-6) -> dict:
    """
    Static dielectric tensor and infrared mode analysis from the analytic
    Gamma-point dynamical matrix D [eV/(A^2 amu)] (ADR-0019, Q3, Q31):

        p_m = sum_k Z_k e_m(k) / sqrt(M_k)            [e / sqrt(amu)]
        S_m = p_m p_m^T                              (oscillator strength)
        eps_0 = eps_inf + (4 pi K_E / V) sum_m S_m / w_m^2
        Z*_m = sum_k Z_k U_m(k) / |U_m|,  U_m(k) = e_m(k) / sqrt(M_k)
        I_m  = |p_m|^2                               (IR intensity, e^2/amu)

    The three acoustic modes (|w^2| below ``acoustic_tol``) are left out;
    an imaginary optical mode raises (eps_0 is then undefined).
    """
    from .phonon import PhononCalculator, _THZ_TO_CM
    m = np.asarray(masses, dtype=float)
    n = len(m)
    w2, e = np.linalg.eigh(0.5 * (D_gamma + D_gamma.conj().T))
    inv_sqm = np.repeat(1.0 / np.sqrt(m), 3)
    U = e * inv_sqm[:, None]                                  # (3N, modes)
    Zbig = np.asarray(Z, float).transpose(1, 0, 2).reshape(3, 3 * n)  # (a, k b)
    p = Zbig @ U                                              # (3, modes)
    freqs = PhononCalculator._eig_to_freq(w2) / _THZ_TO_CM
    optical = np.abs(w2) > acoustic_tol
    if np.any(w2[optical] < 0):
        raise ValueError("dielectric_response: imaginary optical mode at Gamma "
                         f"({freqs.min():.3f} THz); eps_0 is undefined.")
    pref = 4.0 * np.pi * K_E / volume
    eps0 = np.array(eps_inf, dtype=float).copy()
    modes = []
    for k in range(3 * n):
        S = np.real(np.outer(p[:, k], p[:, k].conj()))
        inten = float(np.real(p[:, k] @ p[:, k].conj()))
        u = U[:, k]
        zm = np.real(Zbig @ u) / np.linalg.norm(u)
        d_eps = pref * S / w2[k] if optical[k] else np.zeros((3, 3))
        if optical[k]:
            eps0 += d_eps
        modes.append({"frequency_THz": float(freqs[k]),
                      "acoustic": bool(not optical[k]),
                      "oscillator_strength_e2_amu": S,
                      "delta_epsilon": d_eps,
                      "mode_effective_charge_e": zm,
                      "ir_intensity_e2_amu": inten})
    return {"epsilon_0": eps0, "epsilon_inf": np.array(eps_inf, float),
            "modes": modes}


def piezoelectric_tensor(Z, internal_strain, volume, compliance_per_GPa,
                         epsilon_static, clamped_ion=None) -> dict:
    """
    Relaxed-ion piezoelectric response of a rigid-ion (fixed Z*) model
    (ADR-0019, Q15, Q29):

        e_{i j} = e0_{ij} + (e / V) sum_k Z_k,ib du_kb / d eta_j    [C/m^2]
        d = e S^E                                                 [pC/N]
        k_{ij}^2 = d_ij^2 / (S^E_jj eps^T_ii),
        eps^T = eps_vac eps_static + e d^T   (stress-free permittivity)

    Voigt j, engineering shear strain (as the internal strain of
    ElasticConstantsCalculator).  e0: the clamped-ion (electronic) term,
    zero for point charges, from born.clamped_ion_piezo otherwise.
    """
    Z = np.asarray(Z, float)
    u = np.asarray(internal_strain, float)                     # (N, 3, 6)
    e_ion = np.einsum("kib,kbj->ij", Z, u) * _E_C / (volume * 1e-30) * 1e-10
    e = e_ion + (0.0 if clamped_ion is None else np.asarray(clamped_ion, float))
    S = np.asarray(compliance_per_GPa, float) * 1e-9           # 1/Pa
    d = e @ S                                                  # C/N
    epsT = _EPS0 * np.asarray(epsilon_static, float) + e @ d.T # F/m
    with np.errstate(divide="ignore", invalid="ignore"):
        k = d / np.sqrt(np.outer(np.diag(epsT), np.diag(S)))
    return {"e_C_m2": e, "e_ionic_C_m2": e_ion, "d_pC_N": d * 1e12,
            "k": k, "epsilon_T_relative": epsT / _EPS0}


def static_dielectric_by_field(force_fn, lattice, positions, atom_types, Z,
                               epsilon_inf, field: float = 5e-3,
                               relax_kw: Optional[dict] = None,
                               output_dir: str = ".") -> np.ndarray:
    """
    Relaxed-ion (strain-clamped) static dielectric tensor of a cell by
    relaxing the ions in a uniform field (ADR-0019): the force Z_k^T E is
    added to every atom (energy - sum_k (Z_k^T E) . r_k), the ions are
    relaxed at fixed cell for E = +-field along each axis, and

        eps_ab = eps_inf_ab + (4 pi K_E / V) sum_k Z_k,ac du_kc / dE_b.

    ``field`` in V/A.  5e-3 is on the plateau for rocksalt MgO
    (eps_0 9.3801 at 5e-3 and 1e-2 V/A against 9.3809 from force constants
    extrapolated to zero displacement); weaker fields amplify the
    relaxation tolerance (error ~ fmax / field), stronger ones the
    anharmonicity.  Equivalent to eps_inf + (4 pi K_E / V) Z Phi^+ Z^T
    (dielectric_response), without building Phi; used by ``piezo`` on the
    elastic cell.
    """
    from .geometry_optimiser import GeometryOptimiser
    Z = np.asarray(Z, float)
    lattice = np.asarray(lattice, float)
    pos0 = np.asarray(positions, float)
    vol = abs(float(np.linalg.det(lattice)))
    kw = dict(method="fire", fmax=1e-5, max_steps=10000)
    kw.update(relax_kw or {})
    dudE = np.zeros((len(Z), 3, 3))                     # (k, c, b)
    for b in range(3):
        u = []
        for sgn in (1.0, -1.0):
            E = np.zeros(3)
            E[b] = sgn * field
            Fext = np.einsum("kab,a->kb", Z, E)           # eV/A

            def fn(lat, pos, types, _F=Fext, **k):
                e, f, s = force_fn(lat, pos, types, **k)
                return e - float(np.sum(_F * pos)), f + _F, s
            fn.r_cut = getattr(force_fn, "r_cut", None)
            r = GeometryOptimiser(lattice, pos0, atom_types, force_fn=fn,
                                  relax_cell=False, verbose=False,
                                  output_dir=output_dir, **kw).run()
            d = r["positions"] - pos0
            u.append(d - d.mean(axis=0))
        dudE[:, :, b] = (u[0] - u[1]) / (2.0 * field)
    return np.asarray(epsilon_inf, float) + 4.0 * np.pi * K_E / vol * np.einsum(
        "kac,kcb->ab", Z, dudE)
