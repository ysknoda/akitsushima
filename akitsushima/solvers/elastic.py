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
import tempfile
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit

from ..constants import EV_ANG3_TO_GPA
from .geometry_optimiser import GeometryOptimiser

__all__ = ["ElasticConstantsCalculator", "fit_equation_of_state"]

# Voigt notation: (1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy), 0-indexed here.
_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


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


class ElasticConstantsCalculator:
    """
    Compute the 6x6 elastic constant tensor C_ij (Voigt notation) [GPa]
    from small strain deformations, via linear stress-strain fitting.

    For each of the 6 independent Voigt strain modes, applies a
    +-``delta`` strain to the reference cell (holding fractional
    coordinates fixed, then optionally relaxing ionic positions at that
    FIXED strained cell shape), and central-differences the resulting
    stress with respect to the strain magnitude.

    Parameters
    ----------
    lattice, positions, atom_types : reference (to-be-strained) structure,
                     ideally already relaxed to its own equilibrium (a
                     nonzero starting stress will bias the linear
                     stress-strain fit).
    force_fn        : callable (lattice, positions, atom_types) -> (E, F, S).
    delta           : strain magnitude for the +-delta finite difference
                     (dimensionless, default 0.005 = 0.5%).
    relax_ions      : if True (default), relax ionic positions at each
                     strained cell via GeometryOptimiser(relax_cell=False)
                     before reading off the stress -- needed whenever the
                     strained cell has inequivalent atomic sites that
                     would otherwise feel a nonzero internal force (e.g.
                     a shear strain on a multi-atom-basis cell). Skipping
                     this (relax_ions=False) is only valid for structures
                     with no free internal coordinates under strain (e.g.
                     a single atom per primitive cell).
    fmax            : force convergence criterion for the ionic
                     relaxation, when relax_ions=True [eV/A].

    Minimum cell size (important)
    ------------------------------
    ``lattice``/``positions`` must already be large enough that
    ``force_fn``'s own r_cut is comfortably below half the shortest
    cell dimension -- exactly the same constraint
    ``NeighbourList.build()`` enforces for any force_fn that uses one.
    If ``force_fn`` internally falls back to a bare O(N^2) minimum-image
    sum with NO ``NeighbourList`` (e.g. calling a potential directly
    with ``neigh_list=None``), THAT path performs no equivalent
    validation and will silently return WRONG (and not obviously so --
    still finite, still smooth) forces/stress once r_cut exceeds this
    bound, since more than one periodic image of a neighbour then lies
    within the cutoff and only one is (arbitrarily) kept. This was
    confirmed directly during development: a bare 4-atom conventional
    FCC cell with r_cut=6.0 A (roughly 3x too large for that cell) gave
    a nonzero shear-strain force (~0.018 eV/A) on what is, by inversion
    symmetry, an EXACTLY force-free configuration for any true single-
    atom-basis Bravais lattice under any affine strain -- and fed
    through to a bulk modulus wrong by close to a factor of 5. Switching
    to an adequately-sized supercell reduced that same force to ~2e-13
    eV/A (machine-precision zero) and gave a physically sound,
    Cauchy-relation-respecting result. Use a supercell large enough for
    your ``force_fn``'s r_cut before calling this class.
    """

    def __init__(self,
                 lattice    : np.ndarray,
                 positions  : np.ndarray,
                 atom_types : list,
                 force_fn   : Callable,
                 delta      : float = 0.005,
                 relax_ions : bool  = True,
                 fmax       : float = 1e-4,
                 max_steps  : int   = 1000):
        self.lattice0    = np.asarray(lattice, dtype=float).copy()
        self.positions0  = np.asarray(positions, dtype=float).copy()
        self.atom_types  = list(atom_types)
        self.force_fn    = force_fn
        self.delta       = float(delta)
        self.relax_ions  = relax_ions
        self.fmax        = fmax
        self.max_steps   = max_steps
        self._frac0      = self.positions0 @ np.linalg.inv(self.lattice0)
        # Bug-fix note (code-check follow-up): this used to be a single
        # hardcoded path shared by every ElasticConstantsCalculator
        # instance/process. GeometryOptimiser truncates (mode "w") the
        # relax.log/XDATCAR_relax files it opens there on every call, so
        # two instances running concurrently (e.g. under different MPI
        # ranks each analysing a different structure, or just two
        # instances in the same process) could interleave/clobber each
        # other's scratch log files -- harmless to the returned stress
        # (which is read from the in-memory GeometryOptimiser result, not
        # from these log files), but a real nuisance for anyone trying to
        # inspect them for debugging. A fresh, uniquely-named directory
        # per instance removes that collision risk entirely.
        self._scratch_dir = tempfile.mkdtemp(prefix="akitsushima_elastic_")

    def _stress_at_strain(self, eps: np.ndarray) -> np.ndarray:
        T = np.eye(3) + eps
        lattice_s   = self.lattice0 @ T
        positions_s = self._frac0 @ lattice_s
        if self.relax_ions and len(self.positions0) > 1:
            go = GeometryOptimiser(
                lattice_s, positions_s, self.atom_types,
                force_fn=self.force_fn, relax_cell=False, method="fire",
                fmax=self.fmax, max_steps=self.max_steps, verbose=False,
                output_dir=self._scratch_dir)
            go.run()
            _, _, stress = go._get_forces_and_stress()
        else:
            _, _, stress = self.force_fn(lattice_s, positions_s, self.atom_types)
        return stress

    def compute(self) -> dict:
        """
        Returns
        -------
        dict with keys:
            C            : (6, 6) elastic constant tensor [GPa], Voigt order.
            bulk_modulus : Voigt-averaged bulk modulus [GPa]
                          ``(C11+C22+C33 + 2(C12+C13+C23)) / 9``.
        """
        C = np.zeros((6, 6))
        for j in range(6):
            eps_p = _strain_matrix(j, +self.delta)
            eps_m = _strain_matrix(j, -self.delta)
            s_p = self._stress_at_strain(eps_p)
            s_m = self._stress_at_strain(eps_m)
            ds  = (s_p - s_m) / (2.0 * self.delta)
            dsigma_voigt = np.array([ds[a, b] for a, b in _VOIGT_PAIRS])
            # Sign: see module docstring -- this codebase's sigma is the
            # negative of the standard Hooke's-law stress convention.
            C[:, j] = -dsigma_voigt * EV_ANG3_TO_GPA

        C = 0.5 * (C + C.T)   # symmetrise away finite-difference noise
        bulk_modulus = (C[0, 0] + C[1, 1] + C[2, 2]
                       + 2.0 * (C[0, 1] + C[1, 2] + C[0, 2])) / 9.0
        return {"C": C, "bulk_modulus": float(bulk_modulus)}


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
