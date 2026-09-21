"""Finite-difference force/stress validation for every engine potential.

This is the most important regression test in the whole test suite: several
CRITICAL bugs found during the pre-release review were sign/formula errors
that are *invisible* at a symmetric equilibrium structure (force is zero
there by symmetry regardless of a sign flip) and only show up once you
perturb the structure away from equilibrium. Every test below therefore
runs on a randomly perturbed copy of a fixture, never the bare symmetric
fixture.

Stress convention note
-----------------------
The stress-vs-finite-difference relationship used here (``sigma_ab =
-(1/V) dE/dT_ab`` under an affine deformation ``h' = h @ T``, positions
``r' = frac @ h'`` with fixed fractional coordinates) was independently
derived from the chain rule for a generic pair potential and cross-checked
against ``ewald_sum``'s real-space contribution (whose force sign is itself
independently validated two other ways -- see ``engine/coulomb/ewald.py``),
giving an exact match (ratio 1.0 to 8 significant figures). This is the
same check that caught a genuine reciprocal-space stress *formula* bug
(not just a sign flip) that the original code review had not found.
"""
import numpy as np
import pytest

from akitsushima.utils.jit import _HAS_NUMBA
from conftest import (
    diamond_si_dict,
    fcc_al_dict,
    randomly_perturbed,
    rocksalt_nacl_dict,
    toy_dimer_dict,
)


def _charges_array(struct):
    if struct.get("charges") is None:
        return None
    return np.array([struct["charges"][t] for t in struct["atom_types"]])


def finite_diff_forces(energy_fn, positions, h=1e-4):
    """Central-difference force estimate: F_i,a = -dE/dr_i,a.

    ``energy_fn(positions) -> float``; lattice is assumed fixed (this is a
    force check, not a stress check -- use :func:`finite_diff_stress` for
    the latter, which must also vary the lattice).
    """
    N = positions.shape[0]
    F = np.zeros((N, 3))
    for i in range(N):
        for a in range(3):
            pp = positions.copy()
            pm = positions.copy()
            pp[i, a] += h
            pm[i, a] -= h
            F[i, a] = -(energy_fn(pp) - energy_fn(pm)) / (2.0 * h)
    return F


def finite_diff_stress(energy_fn, lattice, positions, h=1e-5):
    """
    Central-difference stress estimate via a small linear-deformation probe
    on the cell, holding *fractional* coordinates fixed.

    ``energy_fn(lattice, positions) -> float``.

    For each of the 9 independent entries of a deformation matrix
    ``T = I + eps`` (not pre-symmetrized -- see module docstring), applies
    ``lattice' = lattice @ T``, ``positions' = frac @ lattice'`` (affine,
    fractional coordinates fixed), and returns
    ``sigma_ab = -(1/V) dE/dT_ab``. For a correctly rotationally-invariant
    potential the result is automatically symmetric (a useful built-in
    sanity check) even though the probe itself is not.
    """
    volume = abs(np.linalg.det(lattice))
    inv_lat = np.linalg.inv(lattice)
    frac = positions @ inv_lat
    M = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            Tp = np.eye(3)
            Tp[a, b] += h
            Tm = np.eye(3)
            Tm[a, b] -= h
            lat_p = lattice @ Tp
            lat_m = lattice @ Tm
            pos_p = frac @ lat_p
            pos_m = frac @ lat_m
            M[a, b] = (energy_fn(lat_p, pos_p) - energy_fn(lat_m, pos_m)) / (2.0 * h)
    return -M / volume


def assert_forces_match(f_analytic, f_numeric, atol=2e-3, rtol=1e-3, label=""):
    diff = np.abs(f_analytic - f_numeric)
    tol = atol + rtol * np.abs(f_numeric)
    assert np.all(diff <= tol), (
        f"{label} force mismatch: max abs diff={diff.max():.6g}\n"
        f"analytic=\n{f_analytic}\nnumeric=\n{f_numeric}")


def assert_stress_match(s_analytic, s_numeric, atol=2e-3, rtol=1e-3, label=""):
    diff = np.abs(s_analytic - s_numeric)
    tol = atol + rtol * np.abs(s_numeric)
    assert np.all(diff <= tol), (
        f"{label} stress mismatch: max abs diff={diff.max():.6g}\n"
        f"analytic=\n{s_analytic}\nnumeric=\n{s_numeric}")


def assert_translationally_invariant(forces, atol=1e-7):
    assert np.allclose(np.sum(forces, axis=0), 0.0, atol=atol)


# ===========================================================================
# Ewald
# ===========================================================================

def test_ewald_force_matches_finite_difference():
    from akitsushima.engine.coulomb.ewald import ewald_sum

    struct = randomly_perturbed(rocksalt_nacl_dict(), magnitude=0.08, seed=7)
    charges = _charges_array(struct)
    lattice = struct["lattice"]
    r_cut = 4.0

    def energy_of(positions):
        e, _, _ = ewald_sum(lattice, positions, charges, r_cut=r_cut)
        return e

    _, f_analytic, _ = ewald_sum(lattice, struct["positions"], charges, r_cut=r_cut)
    f_numeric = finite_diff_forces(energy_of, struct["positions"])
    assert_forces_match(f_analytic, f_numeric, label="ewald_sum")
    assert_translationally_invariant(f_analytic)


def test_ewald_stress_matches_finite_difference():
    from akitsushima.engine.coulomb.ewald import ewald_sum

    struct = randomly_perturbed(rocksalt_nacl_dict(), magnitude=0.08, seed=7)
    charges = _charges_array(struct)
    r_cut = 4.0

    def energy_of(lattice, positions):
        e, _, _ = ewald_sum(lattice, positions, charges, r_cut=r_cut)
        return e

    _, _, s_analytic = ewald_sum(struct["lattice"], struct["positions"], charges, r_cut=r_cut)
    s_numeric = finite_diff_stress(energy_of, struct["lattice"], struct["positions"])
    assert_stress_match(s_analytic, s_numeric, label="ewald_sum")


def test_ewald_matches_pppm_real_space_with_neighbor_list():
    """Cross-check: ewald_sum's real-space contribution and PPPMSolver's
    neighbour-list real-space branch must agree exactly (both call the
    same shared ``real_space_coulomb`` implementation)."""
    from akitsushima.engine.coulomb.ewald import real_space_coulomb
    from akitsushima.engine.neighborlist import NeighbourList

    struct = randomly_perturbed(rocksalt_nacl_dict(), magnitude=0.08, seed=7)
    charges = _charges_array(struct)
    lattice = struct["lattice"]
    r_cut = 2.7
    alpha = 5.0 / r_cut

    e_direct, f_direct, s_direct = real_space_coulomb(
        lattice, struct["positions"], charges, alpha, r_cut, neigh_list=None)

    nl = NeighbourList(r_cut=r_cut, skin_width=0.1, backend="cll")
    nl.build(lattice, struct["positions"])
    e_nl, f_nl, s_nl = real_space_coulomb(
        lattice, struct["positions"], charges, alpha, r_cut, neigh_list=nl)

    assert abs(e_direct - e_nl) < 1e-10
    assert np.allclose(f_direct, f_nl, atol=1e-10)
    assert np.allclose(s_direct, s_nl, atol=1e-10)


# ===========================================================================
# PPPM
# ===========================================================================

def _pppm_solver(lattice, r_cut=2.7, mesh=64, order=8):
    from akitsushima.engine.coulomb.pppm import PPPMSolver
    alpha = 5.0 / r_cut
    return PPPMSolver(lattice, alpha=alpha, mesh=mesh, order=order, r_cut=r_cut)


def test_pppm_force_matches_finite_difference():
    struct = randomly_perturbed(rocksalt_nacl_dict(), magnitude=0.08, seed=7)
    charges = _charges_array(struct)
    lattice = struct["lattice"]

    def energy_of(positions):
        solver = _pppm_solver(lattice)
        e, _, _ = solver.compute(positions, charges)
        return e

    solver = _pppm_solver(lattice)
    _, f_analytic, _ = solver.compute(struct["positions"], charges)
    f_numeric = finite_diff_forces(energy_of, struct["positions"])
    assert_forces_match(f_analytic, f_numeric, atol=5e-3, rtol=2e-3, label="PPPM")
    assert_translationally_invariant(f_analytic, atol=1e-6)


def test_pppm_stress_matches_finite_difference():
    struct = randomly_perturbed(rocksalt_nacl_dict(), magnitude=0.08, seed=7)
    charges = _charges_array(struct)
    lattice = struct["lattice"]

    def energy_of(lat, positions):
        solver = _pppm_solver(lat)
        e, _, _ = solver.compute(positions, charges)
        return e

    solver = _pppm_solver(lattice)
    _, _, s_analytic = solver.compute(struct["positions"], charges)
    s_numeric = finite_diff_stress(energy_of, lattice, struct["positions"])
    assert_stress_match(s_analytic, s_numeric, atol=5e-3, rtol=2e-3, label="PPPM")


def test_pppm_independent_of_neighbor_list_usage():
    """PPPM's real-space branch must give bit-for-bit-close results whether
    or not a pre-built NeighbourList is supplied (both routes go through the
    same shared real_space_coulomb implementation)."""
    from akitsushima.engine.neighborlist import NeighbourList

    struct = randomly_perturbed(rocksalt_nacl_dict(), magnitude=0.08, seed=7)
    charges = _charges_array(struct)
    lattice = struct["lattice"]
    r_cut = 2.7

    solver = _pppm_solver(lattice, r_cut=r_cut)
    e0, f0, s0 = solver.compute(struct["positions"], charges)

    nl = NeighbourList(r_cut=r_cut, skin_width=0.1, backend="cll")
    nl.build(lattice, struct["positions"])
    e1, f1, s1 = solver.compute(struct["positions"], charges, neigh_list=nl)

    assert abs(e0 - e1) < 1e-10
    assert np.allclose(f0, f1, atol=1e-10)
    assert np.allclose(s0, s1, atol=1e-10)


def test_pppm_matches_ewald_sum():
    """PPPM (mesh-based) and ewald_sum (direct k-sum) approximate the same
    physics and must agree closely for well-converged parameters."""
    from akitsushima.engine.coulomb.ewald import ewald_sum

    struct = randomly_perturbed(rocksalt_nacl_dict(), magnitude=0.08, seed=7)
    charges = _charges_array(struct)
    lattice = struct["lattice"]
    r_cut = 2.7
    alpha = 5.0 / r_cut

    solver = _pppm_solver(lattice, r_cut=r_cut, mesh=64, order=8)
    e_pppm, f_pppm, s_pppm = solver.compute(struct["positions"], charges)
    e_ew, f_ew, s_ew = ewald_sum(lattice, struct["positions"], charges,
                                  alpha=alpha, k_max=12, r_cut=r_cut)

    assert abs(e_pppm - e_ew) < 5e-3
    assert np.allclose(f_pppm, f_ew, atol=1e-2)
    assert np.allclose(s_pppm, s_ew, atol=5e-3)


# ===========================================================================
# ESM (Effective Screening Medium)
# ===========================================================================
#
# ESM found THREE bugs beyond the one already known (E4, the z-index hard
# wrap): (1) the xy charge-assignment phase factor had the wrong sign,
# placing each atom's charge density at the *mirrored* (Lx-rx, Ly-ry)
# position instead of its true position -- confirmed by comparing
# Nx*Ny*ifft2(rho_Gz) for a single test charge against a direct (no-FFT)
# evaluation of the same Gaussian, which showed the FFT-recovered peak sitting
# exactly at the mirrored grid coordinates; (2) np.fft.ifft2 (used to convert
# the manually-phase-constructed G-space potential/field back to real space in
# _compute_forces/_compute_stress) was missing the Nx*Ny renormalisation that
# its own built-in 1/(Nx*Ny) division requires when its input did not come
# from an actual fft2 call -- confirmed because the analytic/numeric force
# ratio scaled as almost exactly 1/(n_grid_xy**2) when n_grid_xy was varied,
# independent of n_grid_z; (3) the G||=0 (planar-averaged) 1-D Poisson solve
# used a plain np.cumsum (O(dz) Riemann-sum accuracy) instead of a proper
# trapezoidal cumulative integral (O(dz^2)) -- confirmed because the residual
# converged only *linearly* (not quadratically) with n_grid_z before the fix,
# and quadratically after.


def _esm_test_structure():
    """A small, clean 2-ion slab -- deliberately not derived from a bulk
    fixture's periodic images, to avoid any incidental alignment with the
    ESM z-grid that could mask or exaggerate discretisation error."""
    lattice = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 16.0]])
    positions = np.array([[1.234, 2.345, -1.111],
                           [3.456, 4.567, 1.222]])
    atom_types = ["Na", "Cl"]
    charges = {"Na": 1.0, "Cl": -1.0}
    return lattice, positions, atom_types, charges


@pytest.mark.parametrize("bc", ["bc1", "bc2", "bc3"])
def test_esm_force_matches_finite_difference(bc):
    from akitsushima.engine.coulomb.esm import ESMCalculator

    lattice, positions, atom_types, charges = _esm_test_structure()
    esm = ESMCalculator(lattice, positions, atom_types, charges=charges, bc=bc,
                         sigma=0.3, n_grid_xy=96, n_grid_z=384)

    def energy_of(pos):
        e, _, _ = esm(lattice, pos, atom_types)
        return e

    _, f_analytic, _ = esm(lattice, positions, atom_types)
    f_numeric = finite_diff_forces(energy_of, positions)
    # Mesh-based method with trilinear force interpolation (not matched to the
    # Gaussian charge-assignment kernel the way PPPM's B-splines are) -- a
    # looser, resolution-appropriate tolerance than Ewald/PPPM's near-machine
    # precision is expected and was confirmed to shrink further with an even
    # finer grid; this tolerance is set from the observed converged residual.
    assert_forces_match(f_analytic, f_numeric, atol=3e-2, rtol=2e-2, label=f"ESM {bc}")


def test_esm_charge_density_recovers_atom_position():
    """Direct regression test for the phase-sign (mirroring) bug: the
    charge density recovered from rho_Gz via ifft2 must peak at the atom's
    true position, not its mirror image (Lx - rx, Ly - ry)."""
    from akitsushima.engine.coulomb.esm import ESMCalculator

    Nx = Ny = 64
    Nz = 32
    Lx = Ly = 6.0
    Lz = 12.0
    sigma = 0.4
    z_grid = np.linspace(-Lz / 2, Lz / 2, Nz, endpoint=False)
    gx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    gy = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')

    rx, ry, rz = 1.03, 2.07, 0.5
    positions = np.array([[rx, ry, rz]])
    charges = np.array([1.0])

    rho_Gz = ESMCalculator._build_charge_density_2d(
        positions, charges, Nx, Ny, Nz, Lz, Lx, Ly, sigma, z_grid, GX, GY)
    rho_real = (Nx * Ny) * np.fft.ifft2(rho_Gz, axes=(0, 1)).real

    x_grid = np.arange(Nx) * (Lx / Nx)
    y_grid = np.arange(Ny) * (Ly / Ny)
    idx = np.unravel_index(np.argmax(rho_real), rho_real.shape)
    peak_x, peak_y = x_grid[idx[0]], y_grid[idx[1]]

    assert abs(peak_x - rx) < (Lx / Nx)
    assert abs(peak_y - ry) < (Ly / Ny)
    # the mirrored (buggy) location would be near (Lx-rx, Ly-ry) -- make sure
    # we did NOT land there instead
    assert abs(peak_x - (Lx - rx)) > Lx / 4


def test_esm_gpar0_poisson_solve_converges_quadratically():
    """Direct regression test for the cumsum -> trapezoidal fix: the
    G||=0 z-force residual (vs. finite difference) must shrink roughly by
    4x (not ~2x) each time n_grid_z doubles."""
    from akitsushima.engine.coulomb.esm import ESMCalculator

    lattice, positions, atom_types, charges = _esm_test_structure()

    def fz_error(nz):
        esm = ESMCalculator(lattice, positions, atom_types, charges=charges,
                             bc="bc1", sigma=0.3, n_grid_xy=48, n_grid_z=nz)
        e0, f_analytic, _ = esm(lattice, positions, atom_types)
        h = 1e-4
        pp = positions.copy()
        pp[0, 2] += h
        pm = positions.copy()
        pm[0, 2] -= h
        ep, _, _ = esm(lattice, pp, atom_types)
        em, _, _ = esm(lattice, pm, atom_types)
        fz_numeric = -(ep - em) / (2 * h)
        return abs(f_analytic[0, 2] - fz_numeric)

    err_192 = fz_error(192)
    err_384 = fz_error(384)
    # linear (buggy) convergence would give a ratio near 2; quadratic
    # (fixed) convergence gives a ratio near 4 -- assert clearly closer to
    # quadratic to guard against a regression back to cumsum.
    assert err_192 / err_384 > 3.0


def test_esm_rejects_bad_lattice():
    from akitsushima.engine.coulomb.esm import ESMCalculator

    lattice, positions, atom_types, charges = _esm_test_structure()
    bad_lattice = lattice.copy()
    bad_lattice[2, 0] = 1.0   # c-vector no longer along z
    with pytest.raises(ValueError):
        ESMCalculator(bad_lattice, positions, atom_types, charges=charges)


def test_esm_z_out_of_bounds_raises_not_wraps():
    """Direct regression test for E4: an atom placed outside the
    [-Lz/2, Lz/2) domain must raise, not silently wrap to the opposite,
    physically disconnected side of a non-periodic boundary."""
    from akitsushima.engine.coulomb.esm import ESMCalculator

    lattice, positions, atom_types, charges = _esm_test_structure()
    esm = ESMCalculator(lattice, positions, atom_types, charges=charges,
                         bc="bc1", sigma=0.3, n_grid_xy=24, n_grid_z=64)
    bad_positions = positions.copy()
    bad_positions[0, 2] = lattice[2, 2] / 2.0 + 1.0   # well outside the domain
    with pytest.raises(ValueError):
        esm(lattice, bad_positions, atom_types)


# ===========================================================================
# Simple pair potentials: Lennard-Jones, Morse, Born-Mayer, Buckingham
# ===========================================================================

def test_lj_force_stress_match_finite_difference():
    from akitsushima.engine import LennardJonesPotential

    struct = randomly_perturbed(fcc_al_dict(), magnitude=0.1, seed=3)
    pot = LennardJonesPotential({frozenset({"Al", "Al"}): {"epsilon": 0.4, "sigma": 2.6}}, r_cut=6.0,
    allow_minimum_image_violation=True)

    def e_of_pos(pos):
        e, _, _ = pot.compute(struct["lattice"], pos, struct["atom_types"])
        return e

    def e_of_lat(lat, pos):
        e, _, _ = pot.compute(lat, pos, struct["atom_types"])
        return e

    _, f_analytic, s_analytic = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert_forces_match(f_analytic, finite_diff_forces(e_of_pos, struct["positions"]), label="LJ")
    assert_stress_match(s_analytic, finite_diff_stress(e_of_lat, struct["lattice"], struct["positions"]), label="LJ")
    assert_translationally_invariant(f_analytic)


def test_morse_force_stress_match_finite_difference():
    from akitsushima.engine import MorsePotential

    struct = randomly_perturbed(fcc_al_dict(), magnitude=0.1, seed=3)
    pot = MorsePotential({frozenset({"Al", "Al"}): {"D_e": 0.3, "alpha": 1.5, "r0": 2.86}}, r_cut=6.0,
    allow_minimum_image_violation=True)

    def e_of_pos(pos):
        e, _, _ = pot.compute(struct["lattice"], pos, struct["atom_types"])
        return e

    def e_of_lat(lat, pos):
        e, _, _ = pot.compute(lat, pos, struct["atom_types"])
        return e

    _, f_analytic, s_analytic = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert_forces_match(f_analytic, finite_diff_forces(e_of_pos, struct["positions"]), label="Morse")
    assert_stress_match(s_analytic, finite_diff_stress(e_of_lat, struct["lattice"], struct["positions"]), label="Morse")
    assert_translationally_invariant(f_analytic)


def test_bornmayer_force_stress_match_finite_difference():
    from akitsushima.engine import BornMayerPotential

    struct = randomly_perturbed(fcc_al_dict(), magnitude=0.1, seed=3)
    pot = BornMayerPotential({frozenset({"Al", "Al"}): {"A": 1000.0, "rho": 0.3}}, r_cut=6.0,
    allow_minimum_image_violation=True)

    def e_of_pos(pos):
        e, _, _ = pot.compute(struct["lattice"], pos, struct["atom_types"])
        return e

    def e_of_lat(lat, pos):
        e, _, _ = pot.compute(lat, pos, struct["atom_types"])
        return e

    _, f_analytic, s_analytic = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert_forces_match(f_analytic, finite_diff_forces(e_of_pos, struct["positions"]), label="BornMayer")
    assert_stress_match(s_analytic, finite_diff_stress(e_of_lat, struct["lattice"], struct["positions"]), label="BornMayer")
    assert_translationally_invariant(f_analytic)


def test_buckingham_force_stress_match_finite_difference():
    from akitsushima.engine import BuckinghamPotential

    struct = randomly_perturbed(fcc_al_dict(), magnitude=0.1, seed=3)
    pot = BuckinghamPotential({frozenset({"Al", "Al"}): {"A": 1000.0, "rho": 0.3, "C": 10.0}}, r_cut=6.0,
    allow_minimum_image_violation=True)

    def e_of_pos(pos):
        e, _, _ = pot.compute(struct["lattice"], pos, struct["atom_types"])
        return e

    def e_of_lat(lat, pos):
        e, _, _ = pot.compute(lat, pos, struct["atom_types"])
        return e

    _, f_analytic, s_analytic = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert_forces_match(f_analytic, finite_diff_forces(e_of_pos, struct["positions"]), label="Buckingham")
    assert_stress_match(s_analytic, finite_diff_stress(e_of_lat, struct["lattice"], struct["positions"]), label="Buckingham")
    assert_translationally_invariant(f_analytic)


def test_buckingham_c_zero_matches_bornmayer():
    """BuckinghamPotential(C=0,
    allow_minimum_image_violation=True) must be numerically identical to
    BornMayerPotential with the same A/rho (design cross-check noted in
    the plan's Buckingham-consolidation design decision)."""
    from akitsushima.engine import BornMayerPotential, BuckinghamPotential

    struct = randomly_perturbed(fcc_al_dict(), magnitude=0.1, seed=3)
    buck = BuckinghamPotential({frozenset({"Al", "Al"}): {"A": 1000.0, "rho": 0.3, "C": 0.0}}, r_cut=6.0,
    allow_minimum_image_violation=True)
    bm = BornMayerPotential({frozenset({"Al", "Al"}): {"A": 1000.0, "rho": 0.3}}, r_cut=6.0,
    allow_minimum_image_violation=True)

    e1, f1, s1 = buck.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    e2, f2, s2 = bm.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert abs(e1 - e2) < 1e-12
    assert np.allclose(f1, f2, atol=1e-12)
    assert np.allclose(s1, s2, atol=1e-12)


def test_mie_force_stress_match_finite_difference():
    """Phase 2: MiePotential, using exponents (n=14, m=7) deliberately
    different from the LJ-reducing (12, 6) case, so this test actually
    exercises the general Mie formula rather than a value that happens
    to coincide with LJ's own already-verified formula."""
    from akitsushima.engine import MiePotential

    struct = randomly_perturbed(fcc_al_dict(), magnitude=0.1, seed=3)
    pot = MiePotential({frozenset({"Al", "Al"}): {"epsilon": 0.4, "sigma": 2.6}},
                       n=14.0, m=7.0, r_cut=6.0,
                                               allow_minimum_image_violation=True)

    def e_of_pos(pos):
        e, _, _ = pot.compute(struct["lattice"], pos, struct["atom_types"])
        return e

    def e_of_lat(lat, pos):
        e, _, _ = pot.compute(lat, pos, struct["atom_types"])
        return e

    _, f_analytic, s_analytic = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert_forces_match(f_analytic, finite_diff_forces(e_of_pos, struct["positions"]), label="Mie")
    assert_stress_match(s_analytic, finite_diff_stress(e_of_lat, struct["lattice"], struct["positions"]), label="Mie")
    assert_translationally_invariant(f_analytic)


def test_mie_reduces_to_lj_at_12_6():
    """MiePotential(n=12, m=6,
    allow_minimum_image_violation=True) must be numerically identical to
    LennardJonesPotential with the same epsilon/sigma -- the free,
    strong correctness check enabled by Mie's own normalisation
    constant C(n,m) (see MiePotential's docstring), and a direct
    demonstration that the Phase 1 registry's "add a new potential"
    recipe produces a genuinely correct implementation, not just one
    that passes its own from-scratch finite-difference check in
    isolation."""
    from akitsushima.engine import LennardJonesPotential, MiePotential

    struct = randomly_perturbed(fcc_al_dict(), magnitude=0.1, seed=3)
    mie = MiePotential({frozenset({"Al", "Al"}): {"epsilon": 0.4, "sigma": 2.6}},
                       n=12.0, m=6.0, r_cut=6.0,
                                               allow_minimum_image_violation=True)
    lj = LennardJonesPotential({frozenset({"Al", "Al"}): {"epsilon": 0.4, "sigma": 2.6}}, r_cut=6.0,
    allow_minimum_image_violation=True)

    e1, f1, s1 = mie.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    e2, f2, s2 = lj.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert abs(e1 - e2) < 1e-10
    assert np.allclose(f1, f2, atol=1e-10)
    assert np.allclose(s1, s2, atol=1e-10)


def test_yukawa_force_stress_match_finite_difference():
    """Phase 2: YukawaPotential (screened Coulomb), with a negative A
    (attractive, as for an opposite-charge pair) to exercise the signed
    parameter regime YukawaParameterSet also allows."""
    from akitsushima.engine import YukawaPotential

    struct = randomly_perturbed(fcc_al_dict(), magnitude=0.1, seed=3)
    pot = YukawaPotential({frozenset({"Al", "Al"}): {"A": -20.0}}, kappa=1.2, r_cut=6.0,
    allow_minimum_image_violation=True)

    def e_of_pos(pos):
        e, _, _ = pot.compute(struct["lattice"], pos, struct["atom_types"])
        return e

    def e_of_lat(lat, pos):
        e, _, _ = pot.compute(lat, pos, struct["atom_types"])
        return e

    _, f_analytic, s_analytic = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert_forces_match(f_analytic, finite_diff_forces(e_of_pos, struct["positions"]), label="Yukawa")
    assert_stress_match(s_analytic, finite_diff_stress(e_of_lat, struct["lattice"], struct["positions"]), label="Yukawa")
    assert_translationally_invariant(f_analytic)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
def test_pair_potential_numba_matches_numpy():
    from akitsushima.engine import LennardJonesPotential

    struct = randomly_perturbed(fcc_al_dict(), magnitude=0.1, seed=3)
    pot = LennardJonesPotential({frozenset({"Al", "Al"}): {"epsilon": 0.4, "sigma": 2.6}}, r_cut=6.0,
    allow_minimum_image_violation=True)
    e0, f0, s0 = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"], use_numba=False)
    e1, f1, s1 = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"], use_numba=True)
    assert abs(e0 - e1) < 1e-8
    assert np.allclose(f0, f1, atol=1e-8)
    assert np.allclose(s0, s1, atol=1e-8)


# ===========================================================================
# Stillinger-Weber
# ===========================================================================

def test_sw_force_stress_match_finite_difference():
    from akitsushima.engine import StillingerWeberPotential

    struct = randomly_perturbed(diamond_si_dict(), magnitude=0.05, seed=4)
    # A minimal dict with a recognized top-level key ('sigma') triggers the
    # legacy flat (default Si) parameter path; the rest fall back to their
    # hardcoded Si defaults via .get(key, default) inside the class.
    pot = StillingerWeberPotential({"sigma": 2.0951}, species=["Si"])

    def e_of_pos(pos):
        e, _, _ = pot.compute(struct["lattice"], pos, struct["atom_types"])
        return e

    def e_of_lat(lat, pos):
        e, _, _ = pot.compute(lat, pos, struct["atom_types"])
        return e

    _, f_analytic, s_analytic = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert_forces_match(f_analytic, finite_diff_forces(e_of_pos, struct["positions"]), label="SW")
    assert_stress_match(s_analytic, finite_diff_stress(e_of_lat, struct["lattice"], struct["positions"]), label="SW")
    assert_translationally_invariant(f_analytic)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
def test_sw_numba_matches_numpy():
    from akitsushima.engine import StillingerWeberPotential

    struct = randomly_perturbed(diamond_si_dict(), magnitude=0.05, seed=4)
    pot = StillingerWeberPotential({"sigma": 2.0951}, species=["Si"])
    e0, f0, s0 = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"], use_numba=False)
    e1, f1, s1 = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"], use_numba=True)
    assert abs(e0 - e1) < 1e-6
    assert np.allclose(f0, f1, atol=1e-6)
    assert np.allclose(s0, s1, atol=1e-6)


def test_sw_multi_species_path():
    """Exercise the nested-triple multi-species parameter format (not just
    the legacy flat single-species default)."""
    from akitsushima.engine import StillingerWeberPotential

    struct = randomly_perturbed(diamond_si_dict(), magnitude=0.05, seed=4)
    params = {
        ("Si", "Si", "Si"): {
            "epsilon": 2.1682, "sigma": 2.0951, "A": 7.049556, "B": 0.6022246,
            "p": 4, "q": 0, "a": 1.80, "lambda": 21.0, "gamma": 1.20,
            "costheta0": -1.0 / 3.0,
        },
    }
    pot = StillingerWeberPotential(params, species=["Si"])

    def e_of_pos(pos):
        e, _, _ = pot.compute(struct["lattice"], pos, struct["atom_types"])
        return e

    _, f_analytic, _ = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert_forces_match(f_analytic, finite_diff_forces(e_of_pos, struct["positions"]), label="SW-multi")
    assert_translationally_invariant(f_analytic)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
def test_sw_multi_species_numba_matches_numpy():
    """Phase 5: the multi-species (nested-triple) path gained its own JIT
    kernel, table-driven off dense per-species-id arrays resolved from
    the dict-keyed parameter set (see StillingerWeberPotential's
    _compute_multi_species docstring). Uses a genuinely heteroatomic
    (Na/Cl rock-salt) structure -- not just a single species dressed up
    in the multi-species dict format, as the pre-existing
    test_sw_multi_species_path does -- with DELIBERATELY DIFFERENT
    parameter values for the (Na,Cl,Cl) vs (Cl,Na,Na) entries, so that a
    species-id indexing mistake in the JIT table lookup (e.g. a
    transposed [i_sp, j_sp]) would show up as a mismatch here rather
    than being masked by both directions happening to share one value."""
    from akitsushima.engine import StillingerWeberPotential

    struct = randomly_perturbed(rocksalt_nacl_dict(), magnitude=0.05, seed=2)
    # a*sigma = 1.80*2.0951 = 3.771 A cuts off just short of the a/sqrt(2)
    # = 3.99 A same-species shell, leaving only the a/2 = 2.82 A
    # Na-Cl nearest-neighbour shell inside cutoff -- i.e. every 2-body/
    # 3-body term actually evaluated is heteroatomic, exactly what this
    # test wants to exercise.
    params = {
        ("Na", "Cl", "Cl"): {
            "epsilon": 2.0, "sigma": 2.0951, "A": 7.05, "B": 0.60,
            "p": 4, "q": 0, "a": 1.80, "lambda": 15.0, "gamma": 1.2,
            "costheta0": -1.0 / 3.0,
        },
        ("Cl", "Na", "Na"): {
            "epsilon": 1.5, "sigma": 2.0951, "A": 5.5, "B": 0.55,
            "p": 4, "q": 0, "a": 1.80, "lambda": 12.0, "gamma": 1.1,
            "costheta0": -1.0 / 3.0,
        },
    }
    pot = StillingerWeberPotential(params, species=["Na", "Cl"])

    e0, f0, s0 = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"], use_numba=False)
    e1, f1, s1 = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"], use_numba=True)
    assert abs(e0 - e1) < 1e-8
    assert np.allclose(f0, f1, atol=1e-8)
    assert np.allclose(s0, s1, atol=1e-8)

    # Also confirm the numba path itself matches a finite-difference
    # force oracle (not just the NumPy path -- catches a bug that could
    # affect both paths identically, though none is expected here since
    # the NumPy path is unchanged and already covered by
    # test_sw_multi_species_path).
    def e_of_pos(pos):
        e, _, _ = pot.compute(struct["lattice"], pos, struct["atom_types"], use_numba=True)
        return e

    assert_forces_match(f1, finite_diff_forces(e_of_pos, struct["positions"]), label="SW-multi-numba")
    assert_translationally_invariant(f1)


# ===========================================================================
# Tersoff
# ===========================================================================

_TERSOFF_SI_PARAMS = dict(
    A=1830.8, B=471.18, lambda1=2.4799, lambda2=1.7322, lambda3=1.7322,
    beta=1.1e-6, n=0.78734, c=100390.0, d=16.217, h=-0.59825, R=2.85, D=0.15,
)


def test_tersoff_force_stress_match_finite_difference():
    from akitsushima.engine import TersoffPotential

    struct = randomly_perturbed(diamond_si_dict(), magnitude=0.05, seed=4)
    pot = TersoffPotential(dict(_TERSOFF_SI_PARAMS), species=["Si"])

    def e_of_pos(pos):
        e, _, _ = pot.compute(struct["lattice"], pos, struct["atom_types"])
        return e

    def e_of_lat(lat, pos):
        e, _, _ = pot.compute(lat, pos, struct["atom_types"])
        return e

    _, f_analytic, s_analytic = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert_forces_match(f_analytic, finite_diff_forces(e_of_pos, struct["positions"]),
                         atol=5e-3, rtol=2e-3, label="Tersoff")
    assert_stress_match(s_analytic, finite_diff_stress(e_of_lat, struct["lattice"], struct["positions"]),
                         atol=5e-3, rtol=2e-3, label="Tersoff")
    assert_translationally_invariant(f_analytic)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
def test_tersoff_numba_matches_numpy():
    from akitsushima.engine import TersoffPotential

    struct = randomly_perturbed(diamond_si_dict(), magnitude=0.05, seed=4)
    pot = TersoffPotential(dict(_TERSOFF_SI_PARAMS), species=["Si"])
    e0, f0, s0 = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"], use_numba=False)
    e1, f1, s1 = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"], use_numba=True)
    assert abs(e0 - e1) < 5e-3
    assert np.allclose(f0, f1, atol=5e-3)
    assert np.allclose(s0, s1, atol=5e-3)


def test_tersoff_multi_species_path():
    """Exercise the nested-triple multi-species parameter format."""
    from akitsushima.engine import TersoffPotential

    struct = randomly_perturbed(diamond_si_dict(), magnitude=0.05, seed=4)
    diag = dict(_TERSOFF_SI_PARAMS)
    params = {("Si", "Si", "Si"): diag}
    pot = TersoffPotential(params, species=["Si"])

    def e_of_pos(pos):
        e, _, _ = pot.compute(struct["lattice"], pos, struct["atom_types"])
        return e

    _, f_analytic, _ = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert_forces_match(f_analytic, finite_diff_forces(e_of_pos, struct["positions"]),
                         atol=5e-3, rtol=2e-3, label="Tersoff-multi")
    assert_translationally_invariant(f_analytic)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
def test_tersoff_multi_species_numba_matches_numpy():
    """Phase 5: the multi-species path gained its own JIT kernels
    (_te_multi_zeta_jit / _te_multi_pass3_jit), table-driven off dense
    per-species-id arrays resolved from the dict-keyed parameter set
    (see TersoffPotential._compute_multi_species's docstring). Uses a
    genuinely heteroatomic zincblende-like structure -- relabelling
    diamond silicon's two interpenetrating FCC sublattices (each atom's
    4 nearest neighbours all sit on the OTHER sublattice, the defining
    property of the diamond/zincblende structure -- the same way real
    binary zincblende compounds like SiC are built from the diamond
    parent lattice) as species "X"/"Y", with DELIBERATELY DIFFERENT
    parameter values for the (X,Y,Y) vs (Y,X,X) table entries, so that a
    transposed species-id table lookup would show up as a mismatch
    rather than being masked by both directions sharing one value."""
    from akitsushima.engine import TersoffPotential

    base = diamond_si_dict()
    struct = randomly_perturbed(base, magnitude=0.05, seed=4)
    struct["atom_types"] = ["X"] * 4 + ["Y"] * 4   # 2 diamond sublattices -> 2 species

    params = {
        ("X", "Y", "Y"): {
            "A": 1830.8, "B": 471.18, "lambda1": 2.4799, "lambda2": 1.7322,
            "R": 2.85, "D": 0.15, "beta": 1.1e-6, "n": 0.78734,
            "c": 100390.0, "d": 16.217, "h": -0.59825, "lambda3": 1.7322,
        },
        ("Y", "X", "X"): {
            "A": 1500.0, "B": 400.0, "lambda1": 2.2, "lambda2": 1.5,
            "R": 2.85, "D": 0.15, "beta": 0.9e-6, "n": 0.7,
            "c": 90000.0, "d": 14.0, "h": -0.55, "lambda3": 1.5,
        },
    }
    pot = TersoffPotential(params, species=["X", "Y"])

    e0, f0, s0 = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"], use_numba=False)
    e1, f1, s1 = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"], use_numba=True)
    assert abs(e0 - e1) < 1e-6
    assert np.allclose(f0, f1, atol=1e-6)
    assert np.allclose(s0, s1, atol=1e-6)

    def e_of_pos(pos):
        e, _, _ = pot.compute(struct["lattice"], pos, struct["atom_types"], use_numba=True)
        return e

    assert_forces_match(f1, finite_diff_forces(e_of_pos, struct["positions"]),
                        atol=5e-3, rtol=2e-3, label="Tersoff-multi-numba")
    assert_translationally_invariant(f1)


# ===========================================================================
# EAM
# ===========================================================================

def test_eam_force_stress_match_finite_difference():
    from akitsushima.engine import EAMPotential

    struct = randomly_perturbed(fcc_al_dict(), magnitude=0.1, seed=3)

    # Simple analytic callables (no spline needed): a Gaussian-decaying
    # density and a soft-core embedding function, generic enough that
    # neither is linear (a linear F(rho) would make the embedding-force
    # cross term trivially zero and not actually exercise dF/drho).
    def rho_fn(r):
        return np.exp(-r / 1.5)

    def F_fn(rho):
        return -0.5 * np.sqrt(np.maximum(rho, 1e-12))

    def phi_fn(r):
        return 500.0 * np.exp(-r / 0.4)

    pot = EAMPotential(F_fn, rho_fn, phi_fn, r_cut=6.0, species=["Al"])

    def e_of_pos(pos):
        e, _, _ = pot.compute(struct["lattice"], pos, struct["atom_types"])
        return e

    def e_of_lat(lat, pos):
        e, _, _ = pot.compute(lat, pos, struct["atom_types"])
        return e

    _, f_analytic, s_analytic = pot.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert_forces_match(f_analytic, finite_diff_forces(e_of_pos, struct["positions"]),
                         atol=5e-3, rtol=2e-3, label="EAM")
    assert_stress_match(s_analytic, finite_diff_stress(e_of_lat, struct["lattice"], struct["positions"]),
                         atol=5e-3, rtol=2e-3, label="EAM")
    assert_translationally_invariant(f_analytic)


def test_eam_tabulated_spline_matches_callable():
    """Cubic-spline (tabulated-array) construction path must agree with
    the equivalent analytic callable to within spline interpolation
    error."""
    from akitsushima.engine import EAMPotential

    struct = randomly_perturbed(fcc_al_dict(), magnitude=0.1, seed=3)

    def rho_fn(r):
        return np.exp(-r / 1.5)

    def F_fn(rho):
        return -0.5 * np.sqrt(np.maximum(rho, 1e-12))

    def phi_fn(r):
        return 500.0 * np.exp(-r / 0.4)

    pot_callable = EAMPotential(F_fn, rho_fn, phi_fn, r_cut=6.0, species=["Al"])

    r_grid = np.linspace(0.05, 6.5, 400)
    rho_grid = np.linspace(1e-6, rho_fn(r_grid.min()), 400)
    pot_tab = EAMPotential(F_fn(rho_grid), rho_fn(r_grid), phi_fn(r_grid),
                           rho_grid=rho_grid, r_grid=r_grid, r_cut=6.0, species=["Al"])

    e0, f0, s0 = pot_callable.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    e1, f1, s1 = pot_tab.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    assert abs(e0 - e1) < 1e-3
    assert np.allclose(f0, f1, atol=1e-2)


# ===========================================================================
# PotentialEngine + compute_all_general
# ===========================================================================

def test_potential_engine_sums_multiple_potentials():
    from akitsushima.engine import BornMayerPotential, LennardJonesPotential, PotentialEngine

    struct = randomly_perturbed(fcc_al_dict(), magnitude=0.1, seed=3)
    lj = LennardJonesPotential({frozenset({"Al", "Al"}): {"epsilon": 0.4, "sigma": 2.6}}, r_cut=6.0,
    allow_minimum_image_violation=True)
    bm = BornMayerPotential({frozenset({"Al", "Al"}): {"A": 100.0, "rho": 0.3}}, r_cut=6.0,
    allow_minimum_image_violation=True)

    e_lj, f_lj, s_lj = lj.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    e_bm, f_bm, s_bm = bm.compute(struct["lattice"], struct["positions"], struct["atom_types"])

    engine = PotentialEngine().add(lj).add(bm)
    e, f, s = engine.compute(struct["lattice"], struct["positions"], struct["atom_types"])

    assert abs(e - (e_lj + e_bm)) < 1e-10
    assert np.allclose(f, f_lj + f_bm, atol=1e-10)
    assert np.allclose(s, s_lj + s_bm, atol=1e-10)


def test_compute_all_general_matches_engine_plus_ewald():
    from akitsushima.engine import BornMayerPotential, PotentialEngine, compute_all_general
    from akitsushima.engine.coulomb.ewald import ewald_sum

    struct = randomly_perturbed(rocksalt_nacl_dict(), magnitude=0.05, seed=7)
    bm = BornMayerPotential({frozenset({"Na", "Cl"}): {"A": 1000.0, "rho": 0.3},
                             frozenset({"Na", "Na"}): {"A": 0.0, "rho": 1.0},
                             frozenset({"Cl", "Cl"}): {"A": 0.0, "rho": 1.0}}, r_cut=2.7,
                                                                                        allow_minimum_image_violation=True)
    engine = PotentialEngine().add(bm)

    e, f, s = compute_all_general(struct["lattice"], struct["positions"], struct["atom_types"],
                                   engine, charges=struct["charges"], r_cut=2.7,
                                   coulomb_method="ewald")

    e_bm, f_bm, s_bm = bm.compute(struct["lattice"], struct["positions"], struct["atom_types"])
    q = np.array([struct["charges"][t] for t in struct["atom_types"]])
    e_ew, f_ew, s_ew = ewald_sum(struct["lattice"], struct["positions"], q, r_cut=2.7)

    assert abs(e - (e_bm + e_ew)) < 1e-10
    assert np.allclose(f, f_bm + f_ew, atol=1e-10)
    assert np.allclose(s, s_bm + s_ew, atol=1e-10)


def test_compute_all_general_pppm_cache_reused_and_correct():
    """Regression test for bug E6: the relocated PPPM LRU cache must
    still produce numerically-correct, lattice-consistent results across
    repeated calls with varying lattices (not just avoid crashing)."""
    from akitsushima.engine import PotentialEngine, compute_all_general
    from akitsushima.engine.potential_engine import _pppm_cache

    _pppm_cache.clear()
    base = randomly_perturbed(rocksalt_nacl_dict(), magnitude=0.05, seed=7)
    engine = PotentialEngine()

    results = []
    for scale in [1.0, 1.01, 1.02, 1.0]:   # revisit the original lattice last
        lat = base["lattice"] * scale
        pos = base["positions"] * scale
        e, f, s = compute_all_general(lat, pos, base["atom_types"], engine,
                                       charges=base["charges"], r_cut=2.5,
                                       coulomb_method="pppm", pppm_mesh=32)
        results.append(e)
    # revisiting the original (scale=1.0) lattice must reproduce the exact
    # same energy as the first call, not a stale/incorrectly-cached one
    assert abs(results[0] - results[3]) < 1e-8
    assert len(_pppm_cache) <= 8


def test_pppm_cache_key_distinguishes_r_cut():
    """
    Regression: the PPPM solver cache key omitted ``r_cut``, so a second
    call at a different real-space cutoff silently got back the solver
    built with the FIRST one.

    ``PPPMSolver`` takes r_cut as a constructor argument and uses it for
    both its real-space erfc sum and its ``r_cut < min_i(d_i)/2``
    minimum-image precondition check, so a stale value is doubly harmful.
    With the default ``ewald_alpha=None`` the bug was masked, because
    alpha is then derived from r_cut and alpha *was* in the key; it is
    reachable with an EXPLICIT ``ewald_alpha``, i.e. exactly a real-space
    cutoff convergence study at fixed Ewald splitting.  Measured on a
    1024-atom MgO cell at alpha=0.35, warming at r_cut=5.0 and then
    asking for 7.0 returned the 5.0 energy bit-for-bit: 730 eV of error
    out of 12271 eV.

    The invariant needs no reference data: a cache is only legitimate if
    it is invisible, so the warmed-cache result must equal the
    cold-cache result for the same inputs.
    """
    from akitsushima.engine import PotentialEngine, compute_all_general
    from akitsushima.engine.potential_engine import _pppm_cache

    # A 2x2x2 supercell is required, not the bare conventional cell: NaCl
    # has a = 5.64 A with its FIRST neighbour shell at 2.82 A, so the
    # single cell's minimum-image limit (r_cut < 2.82 A) leaves no room
    # for two cutoffs that bracket any real pair -- both would enclose an
    # empty real-space sum and give identical energies for the wrong
    # reason.  The 11.28 A supercell admits r_cut up to 5.64 A, so 3.0 and
    # 5.0 straddle the 2.82 / 3.99 / 4.88 A shells.
    unit = randomly_perturbed(rocksalt_nacl_dict(), magnitude=0.05, seed=3)
    a_cell = unit["lattice"][0, 0]
    positions = np.concatenate([
        unit["positions"] + np.array([i, j, k]) * a_cell
        for i in range(2) for j in range(2) for k in range(2)])
    lattice = unit["lattice"] * 2.0
    atom_types = unit["atom_types"] * 8

    engine = PotentialEngine()
    # Explicit alpha -> alpha no longer tracks r_cut, which is what
    # exposes the bug.  0.5 A^-1 is chosen so the erfc-damped real-space
    # tail between the two cutoffs is genuinely significant
    # (erfc(0.5*3.0) = 3.4e-2 vs erfc(0.5*5.0) = 4.1e-4); a strongly
    # damping alpha would make both cutoffs numerically identical and the
    # premise assertion below would (correctly) reject the fixture.
    alpha_fixed = 0.5

    def run(r_cut):
        return compute_all_general(
            lattice, positions, atom_types, engine,
            charges=unit["charges"], r_cut=r_cut, ewald_alpha=alpha_fixed,
            coulomb_method="pppm", pppm_mesh=32)[0]

    _pppm_cache.clear()
    e_small_cold = run(3.0)
    _pppm_cache.clear()
    e_large_cold = run(5.0)

    assert abs(e_large_cold - e_small_cold) > 1e-6, (
        "test premise: r_cut must actually change the energy at this "
        "fixed alpha, or the check below cannot detect a stale solver")

    # Warm the cache at the SMALLER cutoff, then ask for the larger one.
    _pppm_cache.clear()
    run(3.0)
    e_large_warm = run(5.0)

    assert e_large_warm == pytest.approx(e_large_cold, abs=1e-9), (
        f"a warmed cache must not change the answer: got "
        f"{e_large_warm} with the cache warmed at r_cut=3.0, but "
        f"{e_large_cold} from a cold cache (the r_cut=3.0 energy is "
        f"{e_small_cold})")
    assert len(_pppm_cache) == 2, (
        "the two distinct r_cut values must occupy two distinct cache "
        "entries")


# ===========================================================================
# Absolute reference-value checks
#
# Every other test in this file is RELATIVE: finite-difference oracles
# compare forces against the derivative of the same code's energy, and the
# numba-vs-NumPy tests compare two paths against each other.  A constant
# factor applied uniformly to energy, forces and stress is invisible to
# both -- the FD check still agrees, and both paths are wrong identically.
#
# Exactly that happened: every Tersoff kernel carried a spurious 0.5 on
# the pair term (and the matching 0.25 on the three-body correction) and
# returned HALF the correct energy, forces and stress, while the whole
# suite stayed green.  The tests below close that hole by pinning the two
# many-body potentials to numbers their authors published, which no
# internal consistency check can reproduce by construction.
# ===========================================================================

def _diamond_supercell(a: float, species: str, n: int = 2):
    """Conventional diamond cell repeated n^3 times, lattice constant a."""
    base = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25],
                     [0.0, 0.5, 0.5], [0.25, 0.75, 0.75],
                     [0.5, 0.0, 0.5], [0.75, 0.25, 0.75],
                     [0.5, 0.5, 0.0], [0.75, 0.75, 0.25]])
    lattice = np.eye(3) * (a * n)
    frac = np.concatenate([base + [i, j, k]
                           for i in range(n) for j in range(n)
                           for k in range(n)]) / n
    return lattice, frac @ lattice, [species] * len(frac)


def _energy_volume_minimum(pot, a_grid, species, r_cut):
    """(a_at_minimum, E_per_atom_at_minimum) over a lattice-constant scan."""
    from akitsushima.engine import NeighbourList

    out = {}
    for a in a_grid:
        lattice, positions, types = _diamond_supercell(a, species)
        nl = NeighbourList(r_cut=r_cut, skin_width=0.0)
        nl.build(lattice, positions)
        e, _, _ = pot.compute(lattice, positions, types, neigh_list=nl)
        out[a] = e / len(types)
    a_min = min(out, key=out.get)
    return a_min, out[a_min]


def test_tersoff_t3_silicon_reproduces_published_cohesive_energy():
    """Tersoff T3 silicon was fit to two measured numbers, which any
    correct implementation of the T3 parameter set must return:

        a0 = 5.432 A,  E_coh = 4.63 eV/atom

    (Tersoff, Phys. Rev. B 38, 9902 (1988), Table I and the surrounding
    text; the same pair of values LAMMPS' shipped Si.tersoff reproduces.)

    This is an ABSOLUTE check, and that is the whole point -- see the
    section header above.  When this file's Tersoff kernels carried an
    extra factor of 1/2, the lattice constant below still came out at
    exactly 5.432 A (a uniform scale factor cannot move the minimum of
    E(a)) while the cohesive energy came out at 2.3148 eV/atom, half of
    the published 4.63.  Only the energy assertion catches that."""
    from akitsushima.engine import TersoffPotential

    pot = TersoffPotential(dict(_TERSOFF_SI_PARAMS), species=["Si"])
    a_min, e_min = _energy_volume_minimum(
        pot, np.round(np.arange(5.35, 5.51, 0.002), 3), "Si", r_cut=3.0)

    assert a_min == pytest.approx(5.432, abs=0.005), (
        f"Tersoff T3 silicon must minimise at the published a0 = 5.432 A, "
        f"got {a_min:.3f} A")
    assert abs(e_min) == pytest.approx(4.63, rel=0.01), (
        f"Tersoff T3 silicon must give the published E_coh = 4.63 eV/atom, "
        f"got {abs(e_min):.4f} eV/atom (ratio {abs(e_min) / 4.63:.4f}) -- a "
        f"ratio near 0.5 means the pair sum is being halved twice: the "
        f"candidate-pair list already holds each pair once and b_avg "
        f"already carries the 1/2, so the pair prefactor is 1")


def test_sw_silicon_reproduces_published_cohesive_energy():
    """The original Stillinger-Weber silicon parameter set is constructed
    so that the diamond structure sits at

        a0 = 5.431 A,  E_coh = 2 * epsilon = 4.3366 eV/atom

    (Stillinger and Weber, Phys. Rev. B 31, 5262 (1985): epsilon is
    defined as 50 kcal/mol = 2.16826 eV, and the remaining parameters are
    chosen to put the diamond lattice at exactly -2 epsilon per atom.)

    SW was correct when the Tersoff halving was found, so this test went
    in green.  It stays because it is the companion check that tells the
    two apart: if a future change to the shared pair-list convention
    halves one of them, exactly one of these two tests moves."""
    from akitsushima.engine import StillingerWeberPotential

    epsilon = 2.16826
    params = dict(epsilon=epsilon, sigma=2.0951, a=1.80, A=7.049556277,
                  B=0.6022245584, p=4.0, q=0.0, lam=21.0, gamma=1.20,
                  costheta0=-1.0 / 3.0)
    pot = StillingerWeberPotential(params=params, species=["Si"])
    a_min, e_min = _energy_volume_minimum(
        pot, np.round(np.arange(5.35, 5.51, 0.002), 3), "Si",
        r_cut=1.80 * 2.0951)

    assert a_min == pytest.approx(5.431, abs=0.005), (
        f"SW silicon must minimise at the published a0 = 5.431 A, got "
        f"{a_min:.3f} A")
    assert abs(e_min) == pytest.approx(2.0 * epsilon, rel=0.001), (
        f"SW silicon must give E_coh = 2*epsilon = {2 * epsilon:.4f} "
        f"eV/atom, got {abs(e_min):.4f} eV/atom")


def _rocksalt_charges(a: float, n):
    """Rocksalt AB supercell: (lattice, positions, charges = +1/-1)."""
    lattice = np.diag([a * n[0], a * n[1], a * n[2]])
    sub = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    base = np.concatenate([sub, sub + [.5, 0, 0]])
    frac, q = [], []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                frac.append((base + [i, j, k]) / np.array(n))
                q += [1.0] * 4 + [-1.0] * 4
    frac = np.concatenate(frac)
    return lattice, frac @ lattice, np.array(q)


def test_ewald_energy_per_atom_is_supercell_invariant():
    """One infinite crystal, several supercells -- one energy per atom.

    This is reference-free: 2x2x2 and 2x2x4 rocksalt describe the SAME
    crystal, so whatever the Ewald sum returns per atom must not depend
    on which one is handed to it.  It is the sharpest statement of what
    went wrong with the old reciprocal-space cutoff.

    ``k_max`` was a fixed integer shell count -- a cutoff on the INDEX,
    not on |k|.  With k = h b1 + k b2 + l b3 and
    |b_i| = 2 pi / (perpendicular width along i), eight shells reach a
    physical cutoff that SHRINKS as an axis lengthens, so stretching c
    quietly stopped resolving the c direction at all.  The second half of
    this test pins that: the legacy path disagrees with itself by 0.086
    eV/atom, and the error is always negative, which is what drove a
    variable-cell ZnO relaxation to inflate its cell 7x while reporting
    an energy drop that a converged sum puts at a rise.

    The Madelung check at the end is the absolute one, since a sum can be
    self-consistent and still wrong: rocksalt's energy per atom is
    -alpha_M k_e / (2 r_nn) with the Madelung constant alpha_M =
    1.7475646, and nothing inside the implementation knows that number.
    """
    from akitsushima.constants import K_E
    from akitsushima.engine.coulomb.ewald import ewald_sum

    a = 5.64                                    # NaCl
    cells = [(2, 2, 2), (2, 2, 3), (2, 2, 4)]

    auto, legacy = [], []
    for n in cells:
        lattice, positions, q = _rocksalt_charges(a, n)
        auto.append(ewald_sum(lattice, positions, q, alpha=1.0, r_cut=5.0,
                              accuracy=1e-8)[0] / len(q))
        legacy.append(ewald_sum(lattice, positions, q, alpha=1.0, r_cut=5.0,
                                k_max=8)[0] / len(q))

    spread = max(auto) - min(auto)
    assert spread < 1e-8, (
        f"the same crystal gave different energies per atom across "
        f"supercells {cells}: {auto} (spread {spread:.3e} eV/atom)")

    # Control: the old fixed-shell cutoff does NOT satisfy the invariant.
    # Without this the test above could pass on a build where the shells
    # happen to be generous, and would have no teeth.
    legacy_spread = max(legacy) - min(legacy)
    assert legacy_spread > 1e-3, (
        f"the legacy fixed-shell path was expected to break supercell "
        f"invariance (that is the defect being guarded against), but its "
        f"spread was only {legacy_spread:.3e} eV/atom -- if this ever "
        f"fires, the fixture is too small to exercise the anisotropy")

    madelung = 1.7475645946
    expected = -madelung * K_E / (2.0 * (a / 2.0))
    assert auto[0] == pytest.approx(expected, rel=1e-4), (
        f"rocksalt Ewald energy must reproduce the Madelung constant: "
        f"expected {expected:.6f} eV/atom, got {auto[0]:.6f}")


def test_ewald_points_at_pppm_instead_of_silently_truncating():
    """A cell too large for an explicit Ewald sum must say so.

    The failure mode being replaced was silence: the old cutoff simply
    stopped resolving the long axis and returned a number.  Refusing, and
    naming the mesh solver that does scale, is the behaviour pinned here.

    The trigger is a plain large box, not a contrived one -- the number
    of k-vectors inside a fixed |k| cutoff grows with the cell volume as
    V k_cut^3 / (6 pi^2), so a 50 A cube at the default accuracy already
    needs over a million of them.  That is a real system, and an explicit
    Ewald sum is simply the wrong tool for it.
    """
    from akitsushima.engine.coulomb.ewald import ewald_sum

    lattice = np.eye(3) * 50.0
    positions = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
    q = np.array([1.0, -1.0])

    with pytest.raises(ValueError, match="pppm"):
        ewald_sum(lattice, positions, q, alpha=1.0, r_cut=5.0, accuracy=1e-8)

    # The same call with an explicit k_max is the caller overriding on
    # purpose, and is how a brute-force reference value gets computed, so
    # it must NOT be capped.
    e, _, _ = ewald_sum(lattice, positions, q, alpha=1.0, r_cut=5.0, k_max=4)
    assert np.isfinite(e)


def test_pppm_accuracy_knob_actually_buys_accuracy():
    """Asking PPPM for more accuracy must not deliver less.

    ``pppm_accuracy`` sets alpha, and a larger alpha means narrower
    Gaussian clouds, which the mesh then has to resolve: the error is
    governed by alpha*h with h = L/M, not by M alone.  The mesh rule used
    to be M >= 2 alpha L / pi, which after rounding up to a power of two
    and a floor of 16 returned M = 16 for every accuracy from 1e-4 to
    1e-7 on a 300-atom wurtzite ZnO cell.  So alpha climbed, the mesh
    stood still, and each tightening of the knob made the forces
    monotonically WORSE -- 0.019, 0.057, 0.125, 0.225 eV/A rms against an
    rms force of 2.37.  A 20 ps NVT run of that cell drifted 17 eV, more
    than its whole kinetic energy.

    The oracle is the Ewald sum, which is independently pinned by
    test_ewald_energy_per_atom_is_supercell_invariant.  The cell has to
    be big enough that the old rule saturates at its floor of 16, or the
    test passes on a mesh that was accidentally generous.
    """
    from akitsushima.engine.coulomb.ewald import ewald_sum
    from akitsushima.engine.coulomb.pppm import (PPPMSolver,
                                                 _pppm_alpha_from_accuracy,
                                                 _pppm_mesh_from_alpha)

    a, n = 5.64, (3, 3, 3)
    lattice, positions, q = _rocksalt_charges(a, n)
    positions = positions + np.random.default_rng(0).normal(
        0.0, 0.08, positions.shape)                  # break the symmetry
    r_cut = 5.0

    errors, meshes = [], []
    for accuracy in (1e-4, 1e-5, 1e-6):
        alpha = _pppm_alpha_from_accuracy(r_cut, accuracy)
        mesh = _pppm_mesh_from_alpha(lattice, alpha, 6, accuracy)
        meshes.append(mesh)

        _, f_ref, _ = ewald_sum(lattice, positions, q, alpha=alpha,
                                r_cut=r_cut, accuracy=1e-10)
        _, f_pppm, _ = PPPMSolver(lattice=lattice, alpha=alpha, mesh=mesh,
                                  order=6, r_cut=r_cut).compute(positions, q)
        df = f_pppm - f_ref
        errors.append(float(np.sqrt((df ** 2).sum(axis=1)).mean()
                            / np.sqrt((f_ref ** 2).sum(axis=1)).mean()))

    assert meshes[0] > 16, (
        f"test premise: this cell must need a mesh above the floor of 16, "
        f"or the old saturating rule is not being exercised; got {meshes}")
    assert meshes == sorted(meshes), (
        f"a tighter accuracy must not ask for a coarser mesh: {meshes}")
    for tight, loose in zip(errors[1:], errors[:-1]):
        assert tight < loose, (
            f"tightening pppm_accuracy made the forces worse: {errors} "
            f"for meshes {meshes} -- this is the defect being guarded "
            f"against, not a tolerance failure")
    assert errors[0] < 1e-2, (
        f"even the loosest setting should be well under 1% relative force "
        f"error, got {errors[0]:.3e}")
