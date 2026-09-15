"""Regression tests for solvers/elastic.py (ElasticConstantsCalculator,
fit_equation_of_state).

Both elastic-constant tests are analytic/physical-law checks needing no
external reference data, matching this project's established
verification style: (1) the Cauchy relation C12 == C44, which holds
EXACTLY for any purely pairwise (no three-body/angular term) potential
in a cubic lattice -- a textbook fact, independent of this module's own
sign convention; (2) the bulk modulus of a genuine (near-zero-stress)
equilibrium crystal must be POSITIVE (mechanical stability) -- this is
the actual regression test for the sign bug caught during development
(see elastic.py's module docstring: the un-flipped formula gave a
NEGATIVE bulk modulus for a real relaxed LJ FCC crystal, which is
unphysical, decisively settling which sign is correct).

Bug-fix note (minimum-cell-size, found while developing these tests):
the FIRST attempt at both tests below used the BARE 4-atom conventional
FCC cell (a~4.05 A) directly with r_cut=6.0 A, calling `fn(...)`
directly (no NeighbourList) -- i.e. going through
BasePairPotential.compute()'s neigh_list=None O(N^2) fallback path,
which (unlike NeighbourList.build(), see its own `_check_min_cell`)
performs NO validation that r_cut is small enough for the minimum-image
convention to be valid. With r_cut=6.0 A vastly exceeding half this
cell's ~2.0 A shortest dimension, the naive minimum-image reduction
silently used only ONE (arbitrarily chosen) periodic image per pair
where several lie within r_cut, giving forces at a SHEARED
configuration of a true single-atom-basis Bravais lattice (which must
be EXACTLY zero, by inversion symmetry about every lattice site, for
ANY affine strain) of ~0.018 eV/A -- clearly wrong, and this fed through
into a bulk modulus wrong not just in sign but by close to a factor of
5 (46 GPa vs. the correct ~250 GPa recovered below). Confirmed directly:
switching to a properly-sized supercell (with r_cut comfortably below
half the shortest cell length) reduces that same sheared-configuration
force to ~2e-13 eV/A (exact zero, to machine precision) as physically
required, and gives a Cauchy-relation-respecting, stable-sign result.
This is a general hazard of calling any force_fn directly on a
too-small cell without a NeighbourList to enforce the same check --
not specific to this module -- so every test below uses an
appropriately-sized supercell, exactly like the rest of this project's
test suite already does for MD/GCMC (see test_solvers.py's own
_fcc_al_supercell helper and its docstring for the identical concern).
"""
import numpy as np
import pytest

from akitsushima.engine import LennardJonesPotential, PotentialEngine
from akitsushima.fitting import make_force_fn
from akitsushima.solvers import (
    ElasticConstantsCalculator,
    GeometryOptimiser,
    fit_equation_of_state,
)

from conftest import fcc_al_dict


def _fcc_al_supercell(n=3):
    base = fcc_al_dict()
    lattice = base["lattice"] * n
    positions, atom_types = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                shift = np.array([i, j, k], dtype=float) @ base["lattice"]
                positions.append(base["positions"] + shift)
                atom_types.extend(base["atom_types"])
    return {
        "lattice": lattice,
        "positions": np.concatenate(positions, axis=0),
        "atom_types": atom_types,
    }


def _relaxed_lj_fcc_al():
    """A genuinely-relaxed (near-zero-stress) LJ FCC Al structure --
    shared setup for both elastic-constant tests below, since a nonzero
    starting stress would bias the linear stress-strain fit. Uses a
    n=3 supercell (see module docstring).  r_cut is 5.0 rather than 6.0
    for two reasons that only became visible once GeometryOptimiser
    started building a neighbour list.  First, the bound NeighbourList
    enforces is on r_cut + skin_width, not r_cut alone, so 6.0 + 0.5
    already exceeds the unrelaxed cell's 6.075 A half-width.  Second,
    this test RELAXES the cell before straining it, and the relaxed cell
    shrinks to 11.9955 A (half-width 5.9978 A) -- so even 5.5 + 0.5
    lands 0.0022 A over the line at the point where it matters.
    5.0 + 0.5 = 5.5 leaves real headroom and keeps
    below half the shortest cell length."""
    engine = PotentialEngine().add(
        LennardJonesPotential({frozenset({"Al"}): {"epsilon": 0.4, "sigma": 2.5}},
                              r_cut=5.0, shift=True))
    fn = make_force_fn(engine)
    al = _fcc_al_supercell(n=3)
    go = GeometryOptimiser(al["lattice"], al["positions"], al["atom_types"],
                          force_fn=fn, relax_cell=True, method="fire",
                          fmax=0.001, pmax=0.005, max_steps=3000, verbose=False,
                          output_dir="/tmp/akitsushima_test_elastic_eq")
    result = go.run()
    assert result["converged"]
    return fn, result["lattice"], result["positions"], al["atom_types"]


def test_cauchy_relation_c12_equals_c44_for_pairwise_potential_cubic_lattice():
    """For any purely pairwise (central-force) potential in a cubic
    crystal with a centrosymmetric lattice site (FCC qualifies), the
    Cauchy relation C12 == C44 holds exactly -- independent of which
    overall sign convention this module uses (a uniform sign flip
    preserves the equality), so this specifically checks the elastic
    tensor's internal physical consistency, complementing (not
    replacing) the separate bulk-modulus sign check below."""
    fn, lattice, positions, atom_types = _relaxed_lj_fcc_al()
    calc = ElasticConstantsCalculator(lattice, positions, atom_types, fn,
                                      delta=0.005, relax_ions=True,
                                      fmax=1e-4, max_steps=1000)
    result = calc.compute()
    C = result["C"]
    assert C[0, 1] == pytest.approx(C[3, 3], rel=1e-2), (
        f"Cauchy relation C12==C44 must hold for a pairwise potential "
        f"in a cubic lattice; got C12={C[0,1]:.3f}, C44={C[3,3]:.3f}")
    # Cubic symmetry sanity checks (all three diagonal/off-diagonal
    # blocks should also agree with each other, up to numerical noise).
    assert C[0, 0] == pytest.approx(C[1, 1], rel=1e-2)
    assert C[1, 1] == pytest.approx(C[2, 2], rel=1e-2)


def test_bulk_modulus_is_positive_for_stable_equilibrium_crystal():
    """Sign regression: a genuinely relaxed (near-zero-stress) crystal
    must have a POSITIVE bulk modulus (basic mechanical-stability
    requirement -- a real energy minimum cannot have negative
    curvature). This is the actual test that determined the sign in
    elastic.py's C_ij = -d(sigma)/d(eps) formula (the un-flipped
    version fails this exact check)."""
    fn, lattice, positions, atom_types = _relaxed_lj_fcc_al()
    calc = ElasticConstantsCalculator(lattice, positions, atom_types, fn,
                                      delta=0.005, relax_ions=True,
                                      fmax=1e-4, max_steps=1000)
    result = calc.compute()
    assert result["bulk_modulus"] > 0.0, (
        f"bulk modulus of a stable equilibrium crystal must be positive, "
        f"got {result['bulk_modulus']:.3f} GPa")
    # All diagonal elastic constants must also be positive (necessary,
    # though not sufficient, for the full Born stability criteria).
    C = result["C"]
    assert C[0, 0] > 0.0 and C[3, 3] > 0.0


def test_fit_equation_of_state_recovers_known_birch_murnaghan_parameters():
    """Synthetic-data test: generate E(V) points EXACTLY from a known
    Birch-Murnaghan curve, fit, and confirm the original parameters are
    recovered to high precision (a clean numerical-correctness check on
    the curve_fit wiring, independent of any physics)."""
    from akitsushima.solvers.elastic import _birch_murnaghan_energy

    E0_true, V0_true, B0_true_eV_ang3, B0p_true = -2.5, 66.0, 0.3, 4.5
    volumes = np.linspace(V0_true * 0.85, V0_true * 1.15, 12)
    energies = _birch_murnaghan_energy(volumes, E0_true, V0_true,
                                       B0_true_eV_ang3, B0p_true)

    result = fit_equation_of_state(volumes, energies)
    assert result["E0"] == pytest.approx(E0_true, abs=1e-6)
    assert result["V0"] == pytest.approx(V0_true, abs=1e-6)
    assert result["B0"] == pytest.approx(B0_true_eV_ang3 * 160.2176621, rel=1e-4)
    assert result["B0_prime"] == pytest.approx(B0p_true, rel=1e-4)


def test_fit_equation_of_state_rejects_unknown_eos():
    with pytest.raises(ValueError, match="birch-murnaghan"):
        fit_equation_of_state(np.array([1.0, 2.0, 3.0, 4.0]),
                              np.array([1.0, 0.5, 0.6, 1.2]), eos="vinet")


def test_two_instances_get_distinct_scratch_dirs(tmp_path):
    """Code-check follow-up regression test: ElasticConstantsCalculator
    used to hand every instance's internal GeometryOptimiser the SAME
    hardcoded output_dir ('/tmp/akitsushima_elastic_relax_scratch').
    GeometryOptimiser truncates (mode 'w') the relax.log/XDATCAR_relax
    files it opens there on every call, so two instances running
    concurrently could interleave/clobber each other's scratch log files.
    Each instance must now get its own freshly-created, uniquely-named
    scratch directory instead."""
    import os

    engine = PotentialEngine().add(
        LennardJonesPotential({frozenset({"Al"}): {"epsilon": 0.4, "sigma": 2.5}},
                              r_cut=5.0, shift=True))
    fn = make_force_fn(engine)
    al = _fcc_al_supercell(n=3)

    calc1 = ElasticConstantsCalculator(al["lattice"], al["positions"],
                                       al["atom_types"], force_fn=fn)
    calc2 = ElasticConstantsCalculator(al["lattice"], al["positions"],
                                       al["atom_types"], force_fn=fn)

    assert calc1._scratch_dir != calc2._scratch_dir
    assert os.path.isdir(calc1._scratch_dir)
    assert os.path.isdir(calc2._scratch_dir)
