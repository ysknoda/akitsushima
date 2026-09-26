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

from akitsushima.engine.neighborlist import allow_minimum_image_violation


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


def rocksalt_nacl_dict(n: int = 1) -> dict:
    """Conventional cubic rock-salt NaCl, a = 5.64 A, an n x n x n
    supercell of the 4 Na + 4 Cl cell (each cell lists Na before Cl)."""
    a = 5.64
    na_frac = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    cl_frac = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5)]
    cell = np.array(na_frac + cl_frac)
    frac = np.concatenate([(cell + [i, j, k]) / n for i in range(n)
                           for j in range(n) for k in range(n)])
    atom_types = (["Na"] * 4 + ["Cl"] * 4) * n ** 3
    charges = {"Na": 1.0, "Cl": -1.0}
    return _structure(_cubic(n * a), frac, atom_types, charges=charges)


def diamond_si_dict(n: int = 1) -> dict:
    """Conventional cubic diamond silicon, a = 5.43 A, an n x n x n
    supercell of the 8-atom cell.

    n = 2 (64 atoms, half-width 5.43 A) is the smallest that hosts the SW
    (a * sigma = 3.77 A) and Tersoff (R + D = 3.0 A) cutoffs under the
    minimum-image convention; the bare cell's half-width is 2.715 A.
    """
    a = 5.43
    base = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    shift = np.array([0.25, 0.25, 0.25])
    cell = np.array(base + [tuple(np.array(p) + shift) for p in base])
    frac = np.concatenate([(cell + [i, j, k]) / n for i in range(n)
                           for j in range(n) for k in range(n)])
    return _structure(_cubic(n * a), frac, ["Si"] * (8 * n ** 3))


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


@pytest.fixture
def small_cell_ok():
    """Opt this test out of the minimum-image precondition.

    BasePairPotential.compute()'s O(N^2) fallback refuses to run when
    r_cut reaches past half the smallest perpendicular cell width,
    because beyond that limit `s -= np.round(s)` keeps one image per
    pair and silently drops the others.

    Several fixtures in this suite deliberately pair a small reference
    cell with a generous cutoff.  That is legitimate here precisely
    because these tests do not depend on the absolute physics: they
    check that two code paths agree, that an analytic derivative matches
    a finite difference of the same truncated energy, or that a
    save/load round-trip preserves what it was given -- all of which are
    affected identically on both sides.  Requesting this fixture says so
    explicitly; `grep -rl small_cell_ok tests/` then lists every test
    that depends on a cell too small for its cutoff.

    Do NOT use it to make a test that asserts a physical value pass.
    """
    with allow_minimum_image_violation():
        yield
