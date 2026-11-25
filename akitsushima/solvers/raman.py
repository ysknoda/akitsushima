"""First-order Raman spectra in the harmonic approximation (ADR-0046,
ADR-0049).

From the Gamma dynamical matrix of a cell and a polarisability derivative
A[k, a, b, c] = d alpha_ab / d u_kc [A^2] (solvers.response):

    mode Raman tensor   R_m = sum_k A_k . e_m(k) / sqrt(M_k)   [A^2 amu^-1/2]
    single crystal      I_m(e_i, e_s) = (e_s . R_m . e_i)^2
    powder              a = tr R / 3,  g^2 = (3/2) tr(R' R'^T),  R' = R - a I
                        I_par = 45 a^2 + 4 g^2,  I_perp = 3 g^2,
                        powder intensity = I_par + I_perp = 45 a^2 + 7 g^2,
                        rho = I_perp / I_par
                        (D. A. Long, The Raman Effect, Wiley 2002, ch. 5)
    thermal factor      Stokes (n + 1) / w, anti-Stokes n / w,
                        times (w_L -+ w)^4 when a laser wavelength is given

Degenerate partners (frequencies within ``degeneracy_cm``) are summed.
Only relative intensities are meaningful.

Activity is decided from characters, without character tables: for each
space-group operation g (spglib, optional) the character of a degenerate
set S is chi_S(g) = sum_{m in S} e_m^T G(g) e_m, with G(g) moving atom i
onto perm(i) and rotating its displacement.  The multiplicities of the
vector representation (IR) and of the symmetric rank-2 representation
(Raman) in S are

    n_IR    = (1/|G|) sum_g chi_S(g) tr R_g
    n_Raman = (1/|G|) sum_g chi_S(g) (tr(R_g)^2 + tr(R_g^2)) / 2

and the set is active when its multiplicity is non-zero.  The
multiplicities must be integers; the largest deviation is reported (a
degenerate set split by the tolerance, or a structure that is not
symmetric to symprec, shows up there).
"""

import warnings
from collections.abc import Sequence

import numpy as np

from ..constants import EV_ANG_TO_AMU_ANG_FS2, FREQ_RAD_FS_TO_WAVENUMBER_CM

__all__ = [
    "RAMAN_DEGENERACY_CM",
    "activities",
    "degenerate_sets",
    "gamma_modes",
    "lorentzian",
    "mode_raman_tensors",
    "powder_invariants",
    "raman_modes",
    "thermal_factor",
]

# Frequencies closer than this [cm^-1] are one degenerate set.  The
# finite-difference IFC of a hexagonal cell splits E modes by an O(delta^2)
# term (ADR-0040); merging too much is harmless for the characters (the
# set then holds two irreps and the multiplicities stay integers), while
# splitting a true pair makes them non-integer.
RAMAN_DEGENERACY_CM = 0.1

# Second radiation constant h c / k_B [cm K] (CODATA 2018: 1.438776877e-2 m K).
_C2_CM_K = 1.438776877


def gamma_modes(D: np.ndarray):
    """Frequencies [cm^-1, negative for imaginary] and mass-weighted
    eigenvectors (columns) of a Gamma dynamical matrix [eV/(A^2 amu)]."""
    D = np.asarray(D)
    if np.iscomplexobj(D):
        if np.abs(D.imag).max() > 1e-8 * max(np.abs(D.real).max(), 1e-30):
            raise ValueError("gamma_modes: the Gamma dynamical matrix is not real")
        D = D.real
    w2, E = np.linalg.eigh(0.5 * (D + D.T))
    w = np.sqrt(np.abs(w2) * EV_ANG_TO_AMU_ANG_FS2) * FREQ_RAD_FS_TO_WAVENUMBER_CM
    return np.where(w2 < 0.0, -w, w), E


def degenerate_sets(freqs: np.ndarray, tol: float = RAMAN_DEGENERACY_CM) -> list:
    """Index arrays of consecutive (sorted) frequencies within ``tol``."""
    order = np.argsort(freqs)
    sets, cur = [], [order[0]]
    for i in order[1:]:
        if freqs[i] - freqs[cur[-1]] <= tol:
            cur.append(i)
        else:
            sets.append(np.array(cur))
            cur = [i]
    sets.append(np.array(cur))
    return sets


def _characters(ops, E: np.ndarray, sets: list) -> np.ndarray:
    """chi[s, g] of every degenerate set under every operation."""
    N = E.shape[0] // 3
    chi = np.zeros((len(sets), ops.n_ops))
    for g in range(ops.n_ops):
        R, perm = ops.rot_cart[g], ops.perm[g]
        for s, idx in enumerate(sets):
            e = E[:, idx].reshape(N, 3, len(idx))
            ge = np.empty_like(e)
            ge[perm] = np.einsum("ab,ibm->iam", R, e)
            chi[s, g] = float(np.sum(e * ge))
    return chi


def activities(ops, E: np.ndarray, sets: list) -> dict:
    """IR and Raman multiplicities of every set (see the module docstring)
    and the largest deviation of a multiplicity from an integer."""
    chi = _characters(ops, E, sets)
    tr = np.trace(ops.rot_cart, axis1=1, axis2=2)
    tr2 = np.trace(
        np.einsum("gab,gbc->gac", ops.rot_cart, ops.rot_cart), axis1=1, axis2=2
    )
    n_ir = chi @ tr / ops.n_ops
    n_raman = chi @ (0.5 * (tr**2 + tr2)) / ops.n_ops
    dev = float(
        max(
            np.abs(n_ir - np.round(n_ir)).max(),
            np.abs(n_raman - np.round(n_raman)).max(),
        )
    )
    return {"n_ir": n_ir, "n_raman": n_raman, "max_deviation": dev, "characters": chi}


def mode_raman_tensors(A: np.ndarray, E: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """R[m] = sum_kc A[k, :, :, c] e_m[k, c] / sqrt(M_k), shape (3N, 3, 3)."""
    N = len(masses)
    e = E.reshape(N, 3, -1) / np.sqrt(np.asarray(masses, float))[:, None, None]
    R = np.einsum("kabc,kcm->mab", np.asarray(A, float), e)
    return 0.5 * (R + R.transpose(0, 2, 1))


def powder_invariants(R: np.ndarray):
    """(a^2, g^2) of each (symmetric) tensor R[m]."""
    a = np.trace(R, axis1=1, axis2=2) / 3.0
    Rp = R - a[:, None, None] * np.eye(3)
    g2 = 1.5 * np.einsum("mab,mab->m", Rp, Rp)
    return a**2, g2


def thermal_factor(
    freq_cm,
    temperature: float,
    laser_nm: float | None = None,
    anti_stokes: bool = False,
) -> np.ndarray:
    """(n + 1) / w (Stokes) or n / w (anti-Stokes), times (w_L -+ w)^4 when
    ``laser_nm`` is given; w in cm^-1, positive."""
    w = np.asarray(freq_cm, float)
    if temperature > 0.0:
        n = 1.0 / np.expm1(_C2_CM_K * w / temperature)
    else:
        n = np.zeros_like(w)
    f = (n if anti_stokes else n + 1.0) / w
    if laser_nm:
        wl = 1.0e7 / float(laser_nm)
        f = f * (wl + w if anti_stokes else wl - w) ** 4
    return f


def lorentzian(
    grid: np.ndarray, centres: np.ndarray, weights: np.ndarray, fwhm: float
) -> np.ndarray:
    """Sum of area-normalised Lorentzians of full width ``fwhm``."""
    hw = 0.5 * fwhm
    x = grid[:, None] - centres[None, :]
    return (hw / np.pi / (x**2 + hw**2)) @ weights


def _unit(v, name):
    v = np.asarray(v, float).reshape(3)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError(f"raman geometry {name}: zero polarisation vector")
    return v / n


def raman_modes(
    D: np.ndarray,
    masses: np.ndarray,
    A: np.ndarray | None = None,
    ops=None,
    geometries: Sequence[dict] = (),
    temperature: float = 300.0,
    laser_nm: float | None = None,
    anti_stokes: bool = False,
    degeneracy_cm: float = RAMAN_DEGENERACY_CM,
    acoustic_cm: float = 1.0,
) -> dict:
    """Gamma modes of one cell grouped into degenerate sets, with activity
    (when ``ops``, spglib operations of the cell, are given) and Raman
    intensities (when A is given).

    geometries : [{"name", "incident", "scattered"}], Cartesian vectors.

    Returns dict with ``sets`` (one dict per set, ascending frequency:
    frequency_cm, degeneracy, acoustic, imaginary, n_ir, n_raman,
    raman_active, ir_active, and with A: powder_intensity, I_par, I_perp,
    rho, thermal_factor and per-geometry intensities ``geometry``), the
    eigenvectors, the frequencies, the activity deviation and the mode
    Raman tensors.
    """
    masses = np.asarray(masses, float)
    freqs, E = gamma_modes(D)
    sets = degenerate_sets(freqs, degeneracy_cm)
    order = np.argsort(np.abs(freqs))
    acoustic = set(order[:3].tolist())
    worst_acoustic = float(np.abs(freqs[order[:3]]).max())
    if worst_acoustic > acoustic_cm:
        warnings.warn(
            f"raman: an acoustic Gamma mode is at "
            f"{worst_acoustic:.3f} cm^-1 (> {acoustic_cm} cm^-1); "
            "the cell may not be relaxed",
            stacklevel=2,
        )
    act = activities(ops, E, sets) if ops is not None else None
    geos = [
        {
            "name": str(g["name"]),
            "incident": _unit(g["incident"], g["name"]),
            "scattered": _unit(g["scattered"], g["name"]),
        }
        for g in geometries
    ]
    R = mode_raman_tensors(A, E, masses) if A is not None else None
    if R is not None:
        a2, g2 = powder_invariants(R)
    out = []
    for s, idx in enumerate(sets):
        f = float(np.mean(freqs[idx]))
        row = {
            "frequency_cm": f,
            "degeneracy": len(idx),
            "modes": idx.tolist(),
            "acoustic": bool(acoustic & set(idx.tolist())),
            "imaginary": f < -acoustic_cm,
        }
        if act is not None:
            row["n_ir"] = float(act["n_ir"][s])
            row["n_raman"] = float(act["n_raman"][s])
            row["ir_active"] = bool(round(row["n_ir"]) > 0)
            row["raman_active"] = bool(round(row["n_raman"]) > 0)
        if R is not None:
            ipar = float(np.sum(45.0 * a2[idx] + 4.0 * g2[idx]))
            iperp = float(np.sum(3.0 * g2[idx]))
            row["I_par"], row["I_perp"] = ipar, iperp
            row["powder_intensity"] = ipar + iperp
            row["rho"] = iperp / ipar if ipar > 0.0 else float("nan")
            row["geometry"] = {
                g["name"]: float(
                    np.sum(
                        np.einsum("a,mab,b->m", g["scattered"], R[idx], g["incident"])
                        ** 2
                    )
                )
                for g in geos
            }
            ok = not row["acoustic"] and f > acoustic_cm
            row["thermal_factor"] = (
                float(thermal_factor([f], temperature, laser_nm, anti_stokes)[0])
                if ok
                else 0.0
            )
        out.append(row)
    return {
        "sets": out,
        "frequencies_cm": freqs,
        "eigenvectors": E,
        "activity_deviation": (act["max_deviation"] if act else None),
        "raman_tensors": R,
        "geometries": geos,
        "temperature_K": float(temperature),
        "laser_nm": laser_nm,
        "anti_stokes": bool(anti_stokes),
    }


def spectrum_columns(
    result: dict, grid: np.ndarray, fwhm: float, per_atom: int | None = None
) -> dict:
    """Broadened spectra of one raman_modes result on ``grid``: powder
    total / par / perp and one column per geometry, each weighted by the
    thermal factor; divided by ``per_atom`` when given."""
    rows = [
        r
        for r in result["sets"]
        if "powder_intensity" in r and r["thermal_factor"] > 0.0
    ]
    if not rows:
        return {}
    w = np.array([r["frequency_cm"] for r in rows])
    t = np.array([r["thermal_factor"] for r in rows])
    cols = {
        "powder": np.array([r["powder_intensity"] for r in rows]),
        "powder_par": np.array([r["I_par"] for r in rows]),
        "powder_perp": np.array([r["I_perp"] for r in rows]),
    }
    for g in result["geometries"]:
        cols[g["name"]] = np.array([r["geometry"][g["name"]] for r in rows])
    scale = 1.0 / per_atom if per_atom else 1.0
    return {k: scale * lorentzian(grid, w, v * t, fwhm) for k, v in cols.items()}
