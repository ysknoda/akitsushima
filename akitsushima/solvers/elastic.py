"""Elastic constant tensor (via strain-stress fitting) and equation-of-
state (Birch-Murnaghan/Vinet) fitting.

Sign convention note (verified numerically, not assumed)
----------------------------------------------------------
Hooke's law in the standard continuum-mechanics convention is
``sigma_standard = C . eps`` (both stress and strain tensile-positive).
This codebase's own stress, however -- exactly as established while
investigating the NPT barostat sign bug (see
``solvers/molecular_dynamics.py::NPTSimulation._pressure_inst``'s
bug-fix note) -- behaves as the NEGATIVE of that standard convention
when related to a physical pressure/stability criterion: a cell
compressed below its own equilibrium volume has a POSITIVE ``sigma``
trace in this codebase (verified via ``dE/dT`` and a direct finite-
difference/energy-minimum argument), the opposite of the textbook
"compression is negative stress" convention. Consequently
``C_ij = -d(sigma_i)/d(eps_j)`` here (WITH the extra minus sign), not
``+d(sigma_i)/d(eps_j)`` as a naive reading of Hooke's law using this
codebase's own ``sigma`` directly would suggest.

This was confirmed empirically, not merely argued: computing both
candidate signs for a real, relaxed (near-zero-stress) LJ FCC crystal
gives a NEGATIVE bulk modulus for the un-flipped formula and a POSITIVE
one (matching the physically-required mechanical-stability criterion
that a genuine energy minimum must have a positive bulk modulus) for
the flipped formula -- see the module's own test suite
(tests/test_elastic.py) for the corresponding regression test.
"""
import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit

from ..constants import EV_ANG3_TO_GPA
from ..engine.neighborlist import _check_min_image
from .geometry_optimiser import GeometryOptimiser

__all__ = ["ElasticConstantsCalculator", "elastic_moduli",
           "fit_equation_of_state", "DEFAULT_STRAINS"]

# Voigt notation: (1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy), 0-indexed here.
_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

# Strain amplitudes (dimensionless), each applied with both signs
# (ADR-0017 decision 3: de Jong et al., Sci. Data 2, 150009 (2015) use
# 0.5 % and 1 %; 0.25 % is added to see the linear regime from below).
DEFAULT_STRAINS = (0.0025, 0.005, 0.01)

# For the sound velocities / Debye temperature: h and k_B exact (SI 2019),
# amu CODATA 2018.
_AMU_KG = 1.66053906660e-27
_H_SI   = 6.62607015e-34
_KB_SI  = 1.380649e-23


def _strain_matrix(voigt_idx: int, magnitude: float) -> np.ndarray:
    """3x3 symmetric strain tensor for Voigt component `voigt_idx`
    (0-5), at the given (dimensionless) magnitude. Off-diagonal (shear)
    components use the engineering-strain Voigt convention (the
    symmetric tensor entry is half the Voigt value), matching how
    :math:`C_{i4}, C_{i5}, C_{i6}` are conventionally defined."""
    eps = np.zeros((3, 3))
    a, b = _VOIGT_PAIRS[voigt_idx]
    if a == b:
        eps[a, b] = magnitude
    else:
        eps[a, b] = eps[b, a] = magnitude / 2.0
    return eps


def _voigt(sigma: np.ndarray) -> np.ndarray:
    return np.array([sigma[a, b] for a, b in _VOIGT_PAIRS])


class ElasticConstantsCalculator:
    """
    Compute the 6x6 elastic constant tensor C_ij (Voigt notation) [GPa]
    from small strain deformations, via linear stress-strain fitting
    (ADR-0017).

    Each of the 6 Voigt strain modes is applied at +-each amplitude in
    ``strains`` with fractional coordinates fixed (the clamped-ion cell),
    and the ions are then relaxed at that FIXED strained cell shape (the
    relaxed-ion cell).  Each column of C is the slope of a straight-line
    least-squares fit of the six stress components against the strain,
    over those points and the unstrained cell.  With a single amplitude
    this is exactly the old +-delta central difference.

    Parameters
    ----------
    lattice, positions, atom_types : reference (to-be-strained) structure,
                     ideally already relaxed to its own equilibrium (a
                     nonzero starting stress shows up as the fit's
                     intercept, ``residual_stress_GPa``, and biases C at
                     the order of that stress).
    force_fn        : callable (lattice, positions, atom_types) -> (E, F, S).
    strains         : strain amplitudes (dimensionless), each used with
                     both signs.  Default ``DEFAULT_STRAINS``.
    delta           : a single amplitude -- the old interface, the same as
                     ``strains=(delta,)``.  Give at most one of the two.
    relax_ions      : if True (default), relax ionic positions at each
                     strained cell via GeometryOptimiser(relax_cell=False)
                     before reading off the stress -- needed whenever the
                     strained cell has inequivalent atomic sites that
                     would otherwise feel a nonzero internal force (e.g.
                     a shear strain on a multi-atom-basis cell).  With one
                     atom per cell no relaxation is done.
    fmax            : force convergence criterion for the ionic
                     relaxation, when relax_ions=True [eV/A].
    skin_width      : the relaxation's neighbour-list skin [A]; part of
                     the minimum-image check below.
    n_jobs          : worker threads over the strained cells (-1: all
                     cores).  force_fn must be thread-safe for n_jobs > 1
                     (ExternalFileEngine and ELMModel.compute are not).

    Minimum cell size
    -----------------
    The cutoff (plus ``skin_width`` when ions are relaxed) must stay below
    half the smallest perpendicular width of EVERY strained cell; this is
    checked before any force call and raises ValueError otherwise.  Past
    that bound a direct O(N^2) evaluation keeps one periodic image per
    pair and silently drops the others: a 4-atom FCC cell with r_cut = 6 A
    once gave a bulk modulus wrong by a factor of about 5.  Use a supercell.
    """

    def __init__(self,
                 lattice    : np.ndarray,
                 positions  : np.ndarray,
                 atom_types : list,
                 force_fn   : Callable,
                 strains    : Optional[Sequence[float]] = None,
                 delta      : Optional[float] = None,
                 relax_ions : bool  = True,
                 fmax       : float = 1e-4,
                 max_steps  : int   = 1000,
                 skin_width : float = 0.5,
                 n_jobs     : int   = 1):
        if strains is not None and delta is not None:
            raise ValueError("ElasticConstantsCalculator: give strains= or "
                             "delta=, not both.")
        if strains is None:
            strains = (delta,) if delta is not None else DEFAULT_STRAINS
        self.strains = tuple(sorted(abs(float(s)) for s in strains))
        if not self.strains or min(self.strains) <= 0.0:
            raise ValueError("ElasticConstantsCalculator: strain amplitudes "
                             f"must be positive, got {strains}.")
        self.lattice0    = np.asarray(lattice, dtype=float).copy()
        self.positions0  = np.asarray(positions, dtype=float).copy()
        self.atom_types  = list(atom_types)
        self.force_fn    = force_fn
        self.relax_ions  = bool(relax_ions) and len(self.positions0) > 1
        self.fmax        = fmax
        self.max_steps   = max_steps
        self.skin_width  = float(skin_width)
        self.n_jobs      = (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)
        self._frac0      = self.positions0 @ np.linalg.inv(self.lattice0)
        from ..fitting.force_fn import check_esm_for_solver
        from ..fitting.mlip_engine import check_mlip_for_solver
        check_esm_for_solver(force_fn, "ElasticConstantsCalculator",
                             refuse="straining the cell moves the ESM "
                                    "boundaries (ADR-0017)")
        check_mlip_for_solver(force_fn, "ElasticConstantsCalculator",
                              species=set(self.atom_types))
        # One fresh directory per instance, and a subdirectory per strained
        # cell: GeometryOptimiser truncates its relax.log / XDATCAR_relax on
        # every call, so a shared path let concurrent relaxations clobber
        # each other's scratch files.
        self._scratch_dir = tempfile.mkdtemp(prefix="akitsushima_elastic_")
        self._check_cells()

    def _points(self) -> list:
        """(mode j, signed strain) pairs; mode None is the unstrained cell."""
        pts = [(None, 0.0)]
        for j in range(6):
            for s in self.strains:
                pts += [(j, +s), (j, -s)]
        return pts

    def _strained_lattice(self, j, s) -> np.ndarray:
        if j is None:
            return self.lattice0.copy()
        return self.lattice0 @ (np.eye(3) + _strain_matrix(j, s))

    def _check_cells(self) -> None:
        r_cut = getattr(self.force_fn, "r_cut", None)
        if r_cut is None:
            return
        r_list = float(r_cut) + (self.skin_width if self.relax_ions else 0.0)
        for j, s in self._points():
            _check_min_image(self._strained_lattice(j, s), r_list,
                             f"ElasticConstantsCalculator (strain mode "
                             f"{'none' if j is None else j + 1}, {s:+.4f})")

    def _evaluate(self, k: int, j, s) -> tuple:
        """(clamped stress, relaxed stress, affine positions, relaxed
        positions, converged) at one strain point."""
        lattice_s   = self._strained_lattice(j, s)
        positions_s = self._frac0 @ lattice_s
        _, _, stress0 = self.force_fn(lattice_s, positions_s, self.atom_types)
        if not self.relax_ions:
            return stress0, stress0, positions_s, positions_s, True
        out = os.path.join(self._scratch_dir, f"point_{k:02d}")
        os.makedirs(out, exist_ok=True)
        go = GeometryOptimiser(
            lattice_s, positions_s, self.atom_types,
            force_fn=self.force_fn, relax_cell=False, method="fire",
            fmax=self.fmax, max_steps=self.max_steps, verbose=False,
            skin_width=self.skin_width, output_dir=out)
        res = go.run()
        _, _, stress = go._get_forces_and_stress()
        return (stress0, stress, positions_s, res["positions"],
                bool(res["converged"]))

    def compute(self) -> dict:
        """
        Returns
        -------
        dict with keys:
            C                  : (6, 6) relaxed-ion elastic tensor [GPa],
                                 Voigt order, symmetrised.
            C_clamped          : (6, 6) clamped-ion tensor [GPa].
            internal_strain    : (N, 3, 6) du_ia / d eta_j [A per unit
                                 Voigt strain]; zeros when the ions are
                                 not relaxed.
            fit_rms_GPa        : {"relaxed": (6,), "clamped": (6,)} RMS
                                 residual of each mode's straight-line fit.
            residual_stress_GPa: (6,) the fits' intercept, Voigt order,
                                 standard (tensile-positive) sign.
            unconverged        : [(mode 1-6, strain)] whose ionic
                                 relaxation hit max_steps.
            strains            : the amplitudes used.
            bulk_modulus       : Voigt bulk modulus [GPa] (the old
                                 interface; see elastic_moduli).
        """
        pts = self._points()
        if self.n_jobs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
                res = list(ex.map(lambda a: self._evaluate(a[0], *a[1]),
                                  enumerate(pts)))
        else:
            res = [self._evaluate(k, j, s) for k, (j, s) in enumerate(pts)]

        unconverged = [(j + 1, s) for (j, s), r in zip(pts, res)
                       if j is not None and not r[4]]
        if unconverged:
            warnings.warn(
                f"ElasticConstantsCalculator: ionic relaxation did not reach "
                f"fmax={self.fmax} eV/A within {self.max_steps} steps at "
                f"(mode, strain) {unconverged}; the relaxed-ion C and the "
                f"internal strain there are less accurate.", RuntimeWarning)

        n = len(self.positions0)
        C, C0 = np.zeros((6, 6)), np.zeros((6, 6))
        dudeta = np.zeros((n, 3, 6))
        rms = {"relaxed": np.zeros(6), "clamped": np.zeros(6)}
        intercepts = []
        # Sign: this codebase's stress is compressive-positive (ADR-0001),
        # the negative of the standard Hooke's-law stress, so the standard
        # stress is -sigma and C_ij = -d sigma_i / d eta_j.
        for j in range(6):
            idx = [0] + [k for k, (jj, _) in enumerate(pts) if jj == j]
            eta = np.array([pts[k][1] for k in idx])
            A = np.column_stack([eta, np.ones_like(eta)])
            for key, col, target in (("clamped", 0, C0), ("relaxed", 1, C)):
                y = np.array([-_voigt(res[k][col]) for k in idx]) * EV_ANG3_TO_GPA
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                target[:, j] = coef[0]
                rms[key][j] = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
                if key == "relaxed":
                    intercepts.append(coef[1])
            if self.relax_ions:
                # Displacement from the affine position, minimum image in
                # the strained cell, uniform translation removed.
                u = []
                for k in idx:
                    lat_s = self._strained_lattice(*pts[k])
                    f = (res[k][3] - res[k][2]) @ np.linalg.inv(lat_s)
                    d = (f - np.round(f)) @ lat_s
                    u.append((d - d.mean(axis=0)).ravel())
                coef, *_ = np.linalg.lstsq(A, np.array(u), rcond=None)
                dudeta[:, :, j] = coef[0].reshape(n, 3)

        if not np.any(C0) and not np.any(C):
            raise ValueError(
                "ElasticConstantsCalculator: the stress is identically zero "
                "at every strain -- the force function returns no stress "
                "(e.g. an external engine without one).")
        C  = 0.5 * (C + C.T)     # symmetrise away finite-difference noise
        C0 = 0.5 * (C0 + C0.T)
        bulk_modulus = (C[0, 0] + C[1, 1] + C[2, 2]
                       + 2.0 * (C[0, 1] + C[1, 2] + C[0, 2])) / 9.0
        return {"C": C, "C_clamped": C0, "internal_strain": dudeta,
                "fit_rms_GPa": rms,
                "residual_stress_GPa": np.mean(intercepts, axis=0),
                "unconverged": unconverged, "strains": self.strains,
                "bulk_modulus": float(bulk_modulus)}


def elastic_moduli(C      : np.ndarray,
                   volume : Optional[float] = None,
                   masses : Optional[Sequence[float]] = None) -> dict:
    """
    Polycrystalline moduli and stability measures from a Voigt C [GPa]
    (ADR-0017 decisions 6 and 7).

    Parameters
    ----------
    C      : (6, 6) elastic tensor [GPa], Voigt order, engineering shear.
    volume : cell volume [A^3]; with
    masses : per-atom masses [amu] -- both needed for the density, the
             sound velocities and the Debye temperature.

    Returns
    -------
    dict: ``S`` (6x6, 1/GPa), ``eigenvalues`` [GPa], ``born_stable``
    (all eigenvalues > 0; Mouhat and Coudert, PRB 90, 224104 (2014));
    ``B``, ``G``, ``E`` [GPa] and ``nu`` as {"voigt", "reuss", "hill"}
    (Hill 1952; E and nu of each scheme from that scheme's B and G);
    ``pugh_ratio`` = B_H / G_H; ``universal_anisotropy``
    A^U = 5 G_V/G_R + B_V/B_R - 6; with volume and masses also
    ``density_kg_m3``, ``v_l_m_s``, ``v_t_m_s``, ``v_m_m_s`` and
    ``debye_temperature_K`` (Anderson 1963, per atom).

    A singular C gives NaN for every Reuss-derived value.
    """
    C = np.asarray(C, dtype=float)
    eig = np.linalg.eigvalsh(0.5 * (C + C.T))
    try:
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        S = np.full((6, 6), np.nan)

    def diag(M):  return M[0, 0] + M[1, 1] + M[2, 2]
    def off(M):   return M[0, 1] + M[1, 2] + M[0, 2]
    def shear(M): return M[3, 3] + M[4, 4] + M[5, 5]

    B = {"voigt": (diag(C) + 2.0 * off(C)) / 9.0,
         "reuss": 1.0 / (diag(S) + 2.0 * off(S))}
    G = {"voigt": (diag(C) - off(C) + 3.0 * shear(C)) / 15.0,
         "reuss": 15.0 / (4.0 * diag(S) - 4.0 * off(S) + 3.0 * shear(S))}
    for M in (B, G):
        M["hill"] = 0.5 * (M["voigt"] + M["reuss"])
    E, nu = {}, {}
    for k in ("voigt", "reuss", "hill"):
        b, g = B[k], G[k]
        E[k]  = 9.0 * b * g / (3.0 * b + g)
        nu[k] = (3.0 * b - 2.0 * g) / (2.0 * (3.0 * b + g))
    out = {
        "S": S, "eigenvalues": eig, "born_stable": bool(np.all(eig > 0.0)),
        "B": B, "G": G, "E": E, "nu": nu,
        "pugh_ratio": B["hill"] / G["hill"],
        "universal_anisotropy": (5.0 * G["voigt"] / G["reuss"]
                                 + B["voigt"] / B["reuss"] - 6.0),
    }
    if volume is not None and masses is not None:
        rho = float(np.sum(masses)) * _AMU_KG / (volume * 1e-30)     # kg/m^3
        v_l = np.sqrt((B["hill"] + 4.0 * G["hill"] / 3.0) * 1e9 / rho)
        v_t = np.sqrt(G["hill"] * 1e9 / rho)
        v_m = ((2.0 / v_t**3 + 1.0 / v_l**3) / 3.0) ** (-1.0 / 3.0)
        n_dens = len(masses) / (volume * 1e-30)                         # 1/m^3
        theta = (_H_SI / _KB_SI) * (3.0 * n_dens / (4.0 * np.pi)) ** (1.0 / 3.0) * v_m
        out.update(density_kg_m3=rho, v_l_m_s=float(v_l), v_t_m_s=float(v_t),
                   v_m_m_s=float(v_m), debye_temperature_K=float(theta))
    return out


def _birch_murnaghan_energy(V, E0, V0, B0, B0_prime):
    """3rd-order Birch-Murnaghan equation of state: E(V)."""
    eta = (V0 / V) ** (2.0 / 3.0)
    return E0 + (9.0 * V0 * B0 / 16.0) * (
        (eta - 1.0) ** 3 * B0_prime
        + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta))


def fit_equation_of_state(volumes: np.ndarray,
                          energies: np.ndarray,
                          eos: str = "birch-murnaghan") -> dict:
    """
    Fit an E(V) curve (from a volume scan of relaxed energies) to the
    3rd-order Birch-Murnaghan equation of state.

    Parameters
    ----------
    volumes  : (n,) cell volumes [A^3] (a scan around the expected
              equilibrium volume; each entry's energy should come from a
              structure relaxed at FIXED volume, e.g. via
              ``GeometryOptimiser(relax_cell=False)`` after isotropically
              scaling the cell to that volume).
    energies : (n,) corresponding total energies [eV].
    eos      : equation-of-state model; currently only
              ``'birch-murnaghan'`` is implemented.

    Returns
    -------
    dict with keys:
        E0         : equilibrium energy [eV].
        V0         : equilibrium volume [A^3].
        B0         : bulk modulus at equilibrium [GPa].
        B0_prime   : pressure derivative of the bulk modulus (dimensionless).
    """
    if eos != "birch-murnaghan":
        raise ValueError(
            f"fit_equation_of_state: unknown eos '{eos}'. "
            "Only 'birch-murnaghan' is implemented.")

    volumes  = np.asarray(volumes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if len(volumes) < 4:
        raise ValueError(
            "fit_equation_of_state: need at least 4 (volume, energy) "
            f"points to fit 4 parameters, got {len(volumes)}.")

    # Initial guesses: E0/V0 from the minimum-energy sample, B0 from a
    # quadratic fit's curvature, B0_prime a typical solid-state value.
    i_min = int(np.argmin(energies))
    V0_guess = volumes[i_min]
    E0_guess = energies[i_min]
    poly = np.polyfit(volumes, energies, 2)
    B0_guess = float(max(2.0 * poly[0] * V0_guess, 1e-3)) * EV_ANG3_TO_GPA
    p0 = [E0_guess, V0_guess, B0_guess, 4.0]

    popt, _ = curve_fit(_birch_murnaghan_energy, volumes, energies,
                        p0=p0, maxfev=20000)
    E0, V0, B0_eV_ang3, B0_prime = popt
    return {
        "E0"      : float(E0),
        "V0"      : float(V0),
        "B0"      : float(B0_eV_ang3 * EV_ANG3_TO_GPA),
        "B0_prime": float(B0_prime),
    }
