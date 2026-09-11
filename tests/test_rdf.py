"""Regression tests for solvers/rdf.py (RadialDistributionFunction).

Both tests are analytic/self-consistency checks that need no external
reference data, matching this project's established verification style
(see test_engine_forces.py's finite-difference oracles, or
test_mie_reduces_to_lj_at_12_6's cross-check): (1) for a genuinely
single-species system, the total and (sole) partial g(r) must be
numerically identical -- a direct algebraic consequence of rdf.py's own
normalisation derivation (see its module docstring); (2) for a perfect
FCC crystal, integrating g(r) up to just past the first coordination
shell must recover the textbook FCC nearest-neighbour count of exactly
12, a well-known, independently-verifiable fact.
"""
import numpy as np
import pytest

from akitsushima.solvers import RadialDistributionFunction

from conftest import fcc_al_dict, rocksalt_nacl_dict


def _supercell(struct: dict, n: int) -> dict:
    """n x n x n periodic replication of a conventional-cell structure
    dict, needed because NeighbourList (unlike GeometryOptimiser's O(N^2)
    fallback) enforces r_cut + skin_width < half the shortest cell
    length, and these fixtures' bare conventional cells are too small
    for an RDF r_max of several Angstrom."""
    lattice = struct["lattice"] * n
    positions, atom_types = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                shift = np.array([i, j, k], dtype=float) @ struct["lattice"]
                positions.append(struct["positions"] + shift)
                atom_types.extend(struct["atom_types"])
    return {
        "lattice": lattice,
        "positions": np.concatenate(positions, axis=0),
        "atom_types": atom_types,
    }


def test_rdf_single_species_partial_equals_total():
    al = _supercell(fcc_al_dict(), n=4)
    rdf = RadialDistributionFunction(r_max=6.0, n_bins=100)
    rdf.add_frame(al["lattice"], al["positions"], al["atom_types"])
    result = rdf.compute()

    assert len(result["g_partial"]) == 1
    (pair, g_aa), = result["g_partial"].items()
    assert pair == frozenset({"Al"})
    assert np.allclose(g_aa, result["g_total"], atol=1e-12), (
        "For a single-species system, the sole partial g(r) must be "
        "numerically identical to the total g(r) -- see rdf.py's module "
        "docstring for the algebraic derivation of why.")


def test_rdf_fcc_coordination_number_matches_known_nearest_neighbor_count():
    """FCC has exactly 12 nearest neighbours at distance a/sqrt(2) -- a
    textbook fact, independent of anything this module itself computes."""
    al_base = fcc_al_dict()
    a = float(np.linalg.norm(al_base["lattice"][0]))
    nn_dist = a / np.sqrt(2.0)
    al = _supercell(al_base, n=3)

    # Fine bins so the step in the running coordination number is
    # resolved sharply around the (exact, delta-function-like) first
    # shell distance.
    rdf = RadialDistributionFunction(r_max=nn_dist + 1.0, n_bins=2000)
    rdf.add_frame(al["lattice"], al["positions"], al["atom_types"])
    result = rdf.compute()

    # Evaluate the running coordination number just past the first shell
    # (a small margin beyond nn_dist) and just before it.
    r = result["r"]
    idx_after  = np.searchsorted(r, nn_dist + 0.15)
    idx_before = np.searchsorted(r, nn_dist - 0.15)
    n_after  = result["coordination_number"][idx_after]
    n_before = result["coordination_number"][idx_before]

    assert n_before == pytest.approx(0.0, abs=0.05), (
        "no neighbours should be counted before the first FCC shell")
    assert n_after == pytest.approx(12.0, abs=0.05), (
        f"FCC has exactly 12 nearest neighbours; got {n_after:.3f}")


def test_rdf_rocksalt_partial_pairs_present():
    """Sanity check for the multi-species path: a rocksalt (NaCl-type)
    structure must produce all three species-pair combinations
    (Na-Na, Cl-Cl, Na-Cl) with finite, non-negative g(r) everywhere."""
    nacl = _supercell(rocksalt_nacl_dict(), n=3)
    rdf = RadialDistributionFunction(r_max=6.0, n_bins=100)
    rdf.add_frame(nacl["lattice"], nacl["positions"], nacl["atom_types"])
    result = rdf.compute()

    pairs_found = set(result["g_partial"].keys())
    assert pairs_found == {frozenset({"Na"}), frozenset({"Cl"}),
                          frozenset({"Na", "Cl"})}
    for pair, g in result["g_partial"].items():
        assert np.all(np.isfinite(g))
        assert np.all(g >= 0.0)
