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
    elastic_moduli,
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


# ---------------------------------------------------------------------------
# ADR-0017: multi-amplitude fit, clamped/relaxed-ion tensors, internal
# strain, polycrystalline moduli and refusals.
# ---------------------------------------------------------------------------

# Tersoff, Phys. Rev. B 38, 9902 (1988), Si(C) (also Tersoff 1989, Table I
# Si); lambda3 = 0 as published.  R +- D = 2.85 +- 0.15 A is the paper's
# R = 2.7, S = 3.0.
_TERSOFF_SI = dict(A=1830.8, B=471.18, lambda1=2.4799, lambda2=1.7322,
                   lambda3=0.0, beta=1.1e-6, n=0.78734, c=100390.0,
                   d=16.217, h=-0.59825, R=2.85, D=0.15)


def _diamond_si_tersoff():
    """64-atom diamond Si (2x2x2 conventional cells; r_cut 3.0 A + 0.5 A
    skin < half the 10.86 A width) at its Tersoff lattice constant."""
    from scipy.optimize import minimize_scalar
    from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml
    fn, *_ = _build_force_fn_from_yaml(
        {"type": "Tersoff", "species": ["Si"], "params": _TERSOFF_SI})
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    basis = np.vstack([basis, basis + 0.25])
    frac = np.array([(b + [i, j, k]) / 2.0 for i in range(2)
                     for j in range(2) for k in range(2) for b in basis])
    lat0 = np.eye(3) * 2 * 5.432
    types = ["Si"] * len(frac)
    s = minimize_scalar(lambda s: fn(lat0 * s, frac @ lat0 * s, types)[0],
                        bracket=(0.99, 1.01), tol=1e-10).x
    lat = lat0 * s
    return fn, lat, frac @ lat, types, frac


@pytest.fixture(scope="module")
def si_elastic():
    fn, lat, pos, types, frac = _diamond_si_tersoff()
    res = ElasticConstantsCalculator(lat, pos, types, fn, fmax=1e-5,
                                     max_steps=5000).compute()
    return fn, lat, pos, types, frac, res


def test_tersoff_si_reproduces_published_constants(si_elastic):
    """Tersoff (1988) quotes for Si(C): C11 142.5, C12 75.4, C44 69,
    unrelaxed C44^0 118.8 GPa and Kleinman zeta 0.67 (UNCERTAIN ~85 %:
    values recalled, not re-read from the paper; the clamped tensor is
    independently checked against ASE below).  zeta from the internal
    strain: the two fcc sublattices move apart along x by
    zeta * a * gamma_yz / 4."""
    *_, res = si_elastic
    C, C0 = res["C"], res["C_clamped"]
    assert C[0, 0] == pytest.approx(142.5, rel=2e-3)
    assert C[0, 1] == pytest.approx(75.4, rel=2e-3)
    assert C[3, 3] == pytest.approx(69.0, rel=2e-3)
    assert C0[3, 3] == pytest.approx(118.8, rel=2e-3)
    # The normal-strain block has no internal relaxation in diamond.
    assert np.allclose(C[:3, :3], C0[:3, :3], atol=0.05)
    a = si_elastic[1][0, 0] / 2.0
    u = res["internal_strain"]
    zeta = 2.0 * abs(u[0, 0, 3]) * 4.0 / a
    assert zeta == pytest.approx(0.67, abs=0.01)
    assert not res["unconverged"]
    assert np.all(res["fit_rms_GPa"]["relaxed"] < 0.05)


def test_clamped_tensor_matches_ase_tersoff(si_elastic):
    """Oracle: ASE's own Tersoff implementation (independent engine and
    stress code; ASE stress is tensile-positive, so C = +d sigma / d eta)."""
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.tersoff import Tersoff, TersoffParameters
    from ase.units import GPa
    from akitsushima.solvers.elastic import _strain_matrix
    fn, lat, pos, types, frac, res = si_elastic
    calc = Tersoff({("Si", "Si", "Si"):
                    TersoffParameters(m=3.0, gamma=1.0, **_TERSOFF_SI)})
    C = np.zeros((6, 6))
    d = 0.005
    for j in range(6):
        st = []
        for sg in (1, -1):
            L = lat @ (np.eye(3) + _strain_matrix(j, sg * d))
            at = Atoms("Si%d" % len(types), cell=L, scaled_positions=frac,
                       pbc=True)
            at.calc = calc
            st.append(at.get_stress(voigt=True))
        C[:, j] = (st[0] - st[1]) / (2 * d) / GPa
    # Same +-0.5 % central difference on both sides: agree to the
    # engines' rounding (2e-8 GPa when written).
    ours = ElasticConstantsCalculator(lat, pos, types, fn, delta=d,
                                      relax_ions=False).compute()
    assert np.max(np.abs(ours["C_clamped"] - 0.5 * (C + C.T))) < 1e-5
    # The default multi-amplitude fit differs from it only by the
    # stress-strain curvature picked up at +-1 % (0.025 GPa on C12).
    assert np.max(np.abs(res["C_clamped"] - ours["C_clamped"])) < 0.05


def test_internal_strain_and_relaxed_tensor_match_force_constants(si_elastic):
    """ADR-0017 decision 5: du/deta = Phi^+ Lambda and
    C = C^0 - Lambda^T Phi^+ Lambda / V, with Lambda_{ia,j} = dF_ia/deta_j
    at clamped ions and Phi the cell's force-constant matrix, both by
    central differences.  The pseudo-inverse drops the three uniform
    translations (the computed internal strain has its mean removed)."""
    from akitsushima.constants import EV_ANG3_TO_GPA
    from akitsushima.solvers.elastic import _strain_matrix
    fn, lat, pos, types, frac, res = si_elastic
    n = len(types)
    h = 1e-4
    Phi = np.zeros((3 * n, 3 * n))
    for k in range(3 * n):
        dp = np.zeros(3 * n); dp[k] = h
        Fp = fn(lat, pos + dp.reshape(n, 3), types)[1].ravel()
        Fm = fn(lat, pos - dp.reshape(n, 3), types)[1].ravel()
        Phi[:, k] = -(Fp - Fm) / (2 * h)
    Phi = 0.5 * (Phi + Phi.T)
    Lam = np.zeros((3 * n, 6))
    for j in range(6):
        Fs = []
        for sg in (1, -1):
            L = lat @ (np.eye(3) + _strain_matrix(j, sg * h))
            Fs.append(fn(L, frac @ L, types)[1].ravel())
        Lam[:, j] = (Fs[0] - Fs[1]) / (2 * h)
    P = np.linalg.pinv(Phi, rcond=1e-8)
    u = (P @ Lam).reshape(n, 3, 6)
    u -= u.mean(axis=0)
    assert np.max(np.abs(u - res["internal_strain"])) < 2e-3   # A / strain
    V = np.linalg.det(lat)
    C_rel = res["C_clamped"] - Lam.T @ P @ Lam / V * EV_ANG3_TO_GPA
    assert np.max(np.abs(C_rel - res["C"])) < 0.1               # GPa


def test_elastic_moduli_isotropic_identities():
    """An isotropic C (Lame lambda, mu): Voigt = Reuss = Hill, A^U = 0,
    G = mu, B = lambda + 2 mu / 3, E and nu from the textbook relations."""
    lam, mu = 60.0, 30.0
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] = lam + 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu
    m = elastic_moduli(C)
    for k in ("voigt", "reuss", "hill"):
        assert m["B"][k] == pytest.approx(lam + 2 * mu / 3, rel=1e-12)
        assert m["G"][k] == pytest.approx(mu, rel=1e-12)
        assert m["E"][k] == pytest.approx(mu * (3 * lam + 2 * mu) / (lam + mu),
                                          rel=1e-12)
        assert m["nu"][k] == pytest.approx(lam / (2 * (lam + mu)), rel=1e-12)
    assert m["universal_anisotropy"] == pytest.approx(0.0, abs=1e-12)
    assert m["born_stable"]
    assert np.allclose(m["S"] @ C, np.eye(6))


@pytest.mark.parametrize("name", ["cubic", "hexagonal", "triclinic"])
def test_elastic_moduli_match_pymatgen(name):
    """Oracle: pymatgen's ElasticTensor (Voigt C in GPa)."""
    pytest.importorskip("pymatgen")
    from pymatgen.analysis.elasticity import ElasticTensor
    if name == "cubic":        # Cu, room temperature (illustrative values)
        C = np.zeros((6, 6)); C[:3, :3] = 121.4
        C[np.arange(3), np.arange(3)] = 168.4
        C[np.arange(3, 6), np.arange(3, 6)] = 75.4
    elif name == "hexagonal":  # wurtzite-like
        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = 209.7; C[2, 2] = 210.9
        C[0, 1] = C[1, 0] = 121.1
        C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = 105.1
        C[3, 3] = C[4, 4] = 42.5; C[5, 5] = (C[0, 0] - C[0, 1]) / 2
    else:                      # a random positive-definite tensor
        rng = np.random.default_rng(7)
        M = rng.normal(size=(6, 6))
        C = M @ M.T * 10 + np.eye(6) * 50
    et = ElasticTensor.from_voigt(C)
    m = elastic_moduli(C)
    assert m["B"]["voigt"] == pytest.approx(et.k_voigt, rel=1e-10)
    assert m["B"]["reuss"] == pytest.approx(et.k_reuss, rel=1e-10)
    assert m["G"]["voigt"] == pytest.approx(et.g_voigt, rel=1e-10)
    assert m["G"]["reuss"] == pytest.approx(et.g_reuss, rel=1e-10)
    assert m["B"]["hill"] == pytest.approx(et.k_vrh, rel=1e-10)
    assert m["G"]["hill"] == pytest.approx(et.g_vrh, rel=1e-10)
    assert m["E"]["hill"] * 1e9 == pytest.approx(et.y_mod, rel=1e-10)
    assert m["nu"]["hill"] == pytest.approx(et.homogeneous_poisson, rel=1e-10)
    assert m["universal_anisotropy"] == pytest.approx(et.universal_anisotropy,
                                                      rel=1e-8, abs=1e-12)


def test_debye_temperature_of_aluminium():
    """Al at 0 K: C11 114.3, C12 61.92, C44 31.62 GPa (Kamm and Alers,
    J. Appl. Phys. 35, 327 (1964)), a = 4.032 A; elastic Debye temperature
    428 K (Kittel, Introduction to Solid State Physics, Table 5.1, and
    Anderson 1963).  UNCERTAIN ~80 %: the inputs are recalled, hence the
    2 % tolerance; the formula itself is fixed by the isotropic check."""
    C = np.zeros((6, 6)); C[:3, :3] = 61.92
    C[np.arange(3), np.arange(3)] = 114.3
    C[np.arange(3, 6), np.arange(3, 6)] = 31.62
    m = elastic_moduli(C, volume=4.032 ** 3, masses=[26.9815385] * 4)
    assert m["density_kg_m3"] == pytest.approx(2734.1, rel=1e-3)
    assert m["debye_temperature_K"] == pytest.approx(428.0, rel=0.02)


def test_debye_temperature_isotropic_closed_form():
    """Isotropic solid: v_t = sqrt(mu/rho), v_l = sqrt((lambda+2mu)/rho)
    and Theta_D = (h/k_B)(3n/4pi)^(1/3) v_m, by hand."""
    lam, mu = 60.0, 30.0
    C = np.zeros((6, 6)); C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] = lam + 2 * mu
    C[np.arange(3, 6), np.arange(3, 6)] = mu
    V, masses = 50.0, [20.0, 30.0]
    m = elastic_moduli(C, volume=V, masses=masses)
    rho = 50.0 * 1.66053906660e-27 / (V * 1e-30)
    vt = np.sqrt(mu * 1e9 / rho); vl = np.sqrt((lam + 2 * mu) * 1e9 / rho)
    vm = (1 / 3 * (2 / vt ** 3 + 1 / vl ** 3)) ** (-1 / 3)
    th = 6.62607015e-34 / 1.380649e-23 * (3 * 2 / (4 * np.pi * V * 1e-30)) ** (1 / 3) * vm
    assert m["v_t_m_s"] == pytest.approx(vt, rel=1e-12)
    assert m["v_l_m_s"] == pytest.approx(vl, rel=1e-12)
    assert m["debye_temperature_K"] == pytest.approx(th, rel=1e-12)


def test_born_stability_flags_negative_shear():
    C = np.zeros((6, 6)); C[:3, :3] = 80.0
    C[np.arange(3), np.arange(3)] = 100.0          # C11 - C12 > 0
    C[np.arange(3, 6), np.arange(3, 6)] = -5.0     # C44 < 0
    assert not elastic_moduli(C)["born_stable"]


def test_single_amplitude_is_central_difference():
    """strains=(d,) and delta=d are the same thing; both not allowed."""
    fn, lat, pos, types, _ = _diamond_si_tersoff()
    a = ElasticConstantsCalculator(lat, pos, types, fn, delta=0.004,
                                   relax_ions=False).compute()
    b = ElasticConstantsCalculator(lat, pos, types, fn, strains=[0.004],
                                   relax_ions=False).compute()
    assert np.array_equal(a["C"], b["C"])
    with pytest.raises(ValueError, match="not both"):
        ElasticConstantsCalculator(lat, pos, types, fn, delta=0.004,
                                   strains=[0.004])


def test_refuses_cell_too_thin_for_cutoff():
    """Conventional 8-atom Si cell: 3.0 A cutoff + 0.5 A skin > 2.716 A."""
    from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml
    fn, *_ = _build_force_fn_from_yaml(
        {"type": "Tersoff", "species": ["Si"], "params": _TERSOFF_SI})
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    basis = np.vstack([basis, basis + 0.25])
    lat = np.eye(3) * 5.432
    with pytest.raises(ValueError, match="perpendicular"):
        ElasticConstantsCalculator(lat, basis @ lat, ["Si"] * 8, fn)


def test_refuses_esm_and_zero_stress():
    lat = np.eye(3) * 20.0
    pos = np.array([[0.0, 0, 0], [1.0, 1, 1]])

    def esm_fn(lattice, positions, atom_types, **kw):
        return 0.0, np.zeros_like(positions), np.zeros((3, 3))
    esm_fn.coulomb = "esm"
    esm_fn.r_cut = 3.0
    with pytest.raises(ValueError, match="ESM"):
        ElasticConstantsCalculator(lat, pos, ["Si", "Si"], esm_fn)

    def no_stress(lattice, positions, atom_types, **kw):
        return 0.0, np.zeros_like(positions), np.zeros((3, 3))
    with pytest.raises(ValueError, match="identically zero"):
        ElasticConstantsCalculator(lat, pos, ["Si", "Si"], no_stress,
                                   relax_ions=False).compute()


def test_cli_elastic_writes_results_json(tmp_path, monkeypatch):
    """`akitsushima elastic` from the shipped template on a jittered 64-atom
    Si cell: pre-relaxation recovers the reference, and the JSON carries
    C, the Hill moduli and the input hash (ADR-0017, Q25, Q30)."""
    import json
    import yaml
    from test_cli_smoke import _ns
    from test_cli_templates import _write_si
    from akitsushima.cli.parser import _DISPATCH
    from akitsushima.cli.templates import _DUMP_MAP
    monkeypatch.chdir(tmp_path)
    cfg = yaml.safe_load(_DUMP_MAP["elastic"])
    _write_si(tmp_path / "POSCAR", jitter=0.03)
    (tmp_path / "e.yaml").write_text(yaml.safe_dump(cfg))
    assert _DISPATCH["elastic"](_ns(tmp_path / "e.yaml")) == 0
    out = json.loads((tmp_path / "elastic_output" / "elastic_results.json").read_text())
    C = np.array(out["C_GPa"])
    assert C[0, 0] == pytest.approx(142.5, rel=2e-3)
    assert C[3, 3] == pytest.approx(69.0, rel=2e-3)
    assert out["born_stable"] is True
    assert out["B_GPa"]["hill"] == pytest.approx((142.53 + 2 * 75.41) / 3, rel=2e-3)
    assert len(out["input_hash"]) == 64
    assert np.array(out["internal_strain_A"]).shape == (64, 3, 6)
