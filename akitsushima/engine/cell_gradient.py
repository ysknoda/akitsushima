"""Generalised force conjugate to the lattice matrix (Parrinello-Rahman-
style variable-cell relaxation), shared by GeometryOptimiser and
ScaledForceEngine.

Single source of truth for this formula.  It was previously duplicated
independently in ``solvers/geometry_optimiser.py`` and
``fitting/external.py`` and the two copies had DIVERGED into two
different, both wrong, matrix orderings -- discovered via a direct
numerical finite-difference check (perturbing each of the 9 raw lattice
matrix components independently and comparing against the analytic
formula, for a deliberately non-orthogonal cell with a real shear
stress, since a diagonal cell/stress cannot distinguish the two
orderings):

    +vol*(stress @ h_inv_T)   (the geometry_optimiser.py formula):
        max|candidate - (-dE/dh)| = 17.0   -- wrong
    +vol*(h_inv_T @ stress)   (h_inv_T on the LEFT instead):
        max|candidate - (-dE/dh)| = 1.24e-7  -- matches to FD precision

Derivation: under the affine deformation h' = h @ T (this codebase's own
verified convention, see test_engine_forces.py's finite_diff_stress),
with positions r' = frac @ h' at fixed fractional coordinates, the
independently-verified stress definition is
``sigma_ab = -(1/V) dE/dT_ab``, i.e. dE/dT = -V*sigma (as a matrix).
Since T = h^-1 @ h', dE/dh = h^-T @ (dE/dT) = -V * (h^-T @ sigma) by the
chain rule.  The force for a gradient-descent-style minimiser is
f_cell = -dE/dh = +V * (h^-T @ sigma) -- h^-T LEFT-multiplies sigma, not
the other way around.

This module sits in ``engine/`` (not ``solvers/``) because the actual
import DAG is ``engine -> fitting -> solvers -> cli`` (confirmed:
``solvers/geometry_optimiser.py``/``molecular_dynamics.py``/
``config_bridge.py`` import from ``fitting``, and neither ``fitting``
nor ``engine`` ever import from ``solvers``) -- placing the shared
helper in ``solvers`` would make ``fitting/external.py`` import from
``solvers``, a reverse/cyclic dependency against the project's
one-directional-DAG rule.
"""
import numpy as np

__all__ = ["cell_force_from_stress"]


def cell_force_from_stress(stress: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """
    Generalised force conjugate to the lattice matrix h (row-vector
    convention: ``lattice[i, :]`` is lattice vector i).

    f_cell = -dE/dh = +V * (h^-T @ sigma),   V = |det(h)|

    Parameters
    ----------
    stress  : (3, 3) stress tensor [eV/A^3], tensile-positive convention.
    lattice : (3, 3) lattice matrix [A], rows are lattice vectors.

    Returns
    -------
    (3, 3) array [eV/A] -- the raw (unmasked, unconstrained) cell force.
    Units follow from the formula itself: V [A^3] * h^-T [A^-1] * sigma
    [eV/A^3] = eV/A (NOT eV -- the h^-T factor carries an inverse
    length).  eV/A is also required for consistency downstream, where
    GeometryOptimiser._generalised_force concatenates this array
    directly with the ionic forces [eV/A] into one generalised force
    vector.
    """
    stress = np.asarray(stress, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    vol = np.abs(np.linalg.det(lattice))
    h_inv_T = np.linalg.inv(lattice).T
    return vol * (h_inv_T @ stress)
