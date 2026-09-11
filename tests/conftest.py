"""Shared pytest fixtures: small synthetic crystal structures for
engine/solver/fitting regression tests.

All structures use the conventional (not primitive) cubic cell, which keeps
the fractional coordinates easy to verify by eye. Cell edges are only a few
angstrom, so tests that want to exercise the neighbour list's small-cell
(``n_cells[axis] == 1``) code path should simply pick an ``r_cut`` comparable
to or larger than half the cell edge -- no need for primitive cells to hit it.
"""
import numpy as np
import pytest


def _cubic(a: float) -> np.ndarray:
    return np.eye(3) * a


def _structure(lattice, frac_positions, atom_types, charges=None) -> dict:
    frac = np.asarray(frac_positions, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    return {
        "lattice": lattice,
        "positions": frac @ lattice,
        "atom_types": list(atom_types),
        "charges": charges,
    }


def fcc_al_dict() -> dict:
    """Conventional cubic FCC aluminium, a = 4.05 A, 4 atoms."""
    a = 4.05
    frac = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    return _structure(_cubic(a), frac, ["Al"] * 4)


def rocksalt_nacl_dict() -> dict:
    """Conventional cubic rock-salt NaCl, a = 5.64 A, 4 Na + 4 Cl."""
    a = 5.64
    na_frac = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    cl_frac = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5)]
    frac = na_frac + cl_frac
    atom_types = ["Na"] * 4 + ["Cl"] * 4
    charges = {"Na": 1.0, "Cl": -1.0}
    return _structure(_cubic(a), frac, atom_types, charges=charges)


def diamond_si_dict() -> dict:
    """Conventional cubic diamond silicon, a = 5.43 A, 8 atoms."""
    a = 5.43
    base = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    shift = np.array([0.25, 0.25, 0.25])
    frac = base + [tuple(np.array(p) + shift) for p in base]
    return _structure(_cubic(a), frac, ["Si"] * 8)


def toy_dimer_dict() -> dict:
    """Two well-separated identical atoms in a large box (no PBC ambiguity)."""
    lattice = _cubic(20.0)
    positions = np.array([[0.0, 0.0, 0.0], [3.2, 0.0, 0.0]])
    return {
        "lattice": lattice,
        "positions": positions,
        "atom_types": ["Ar", "Ar"],
        "charges": None,
    }


def randomly_perturbed(structure: dict, magnitude: float = 0.05, seed: int = 0) -> dict:
    """Return a copy of ``structure`` with positions displaced by random
    noise (uniform in [-magnitude, +magnitude] A per Cartesian component).

    Symmetric/equilibrium structures have zero force by symmetry regardless
    of a sign error in the force formula -- every force/stress regression
    test in this suite must run on a perturbed copy, never the bare fixture.
    """
    rng = np.random.default_rng(seed)
    out = dict(structure)
    disp = rng.uniform(-magnitude, magnitude, size=structure["positions"].shape)
    out["positions"] = structure["positions"] + disp
    return out


@pytest.fixture
def fcc_al() -> dict:
    return fcc_al_dict()


@pytest.fixture
def rocksalt_nacl() -> dict:
    return rocksalt_nacl_dict()


@pytest.fixture
def diamond_si() -> dict:
    return diamond_si_dict()


@pytest.fixture
def toy_dimer() -> dict:
    return toy_dimer_dict()
