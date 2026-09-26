"""CFCMC (osmc rebuilt, ADR-0009).

- Forces of U(lambda) = U_N + g(lambda, U_N+1 - U_N) (energy-space soft
  core, ADR-0009 Q18) are its gradient, on both branches of g; g is exact
  at lambda = 0 and 1 and bounded for overlaps.
- Exactly solvable muVT system: non-interacting atoms in an external field
  u(x) = c cos(2 pi x / L), so <N> = z I0(beta c) and beta mu_ex =
  -ln I0(beta c) (z = V exp(beta mu) / Lambda^3).  This checks the
  insertion / deletion rules and both lambda-end estimators with a
  nonzero dU; the ideal gas (c = 0) checks the combinatorics alone.
- After insertions and deletions the cached end energies match a fresh
  evaluation, and the arrays stay consistent.
"""

import numpy as np
import pytest
from scipy.special import i0

from akitsushima.constants import K_B
from akitsushima.engine import LennardJonesPotential, PotentialEngine
from akitsushima.fitting.force_fn import make_force_fn
from akitsushima.solvers import CFCMCSimulation

L, T, MASS = 20.0, 300.0, 39.948


def _lambda_dB():
    h, kB, amu = 6.62607015e-34, 1.380649e-23, 1.66053906660e-27
    return h / np.sqrt(2 * np.pi * MASS * amu * kB * T) * 1e10


def _field_fn(c):
    k = 2 * np.pi / L

    def fn(lat, pos, types):
        F = np.zeros_like(pos)
        F[:, 0] = c * k * np.sin(k * pos[:, 0])
        return float(np.sum(c * np.cos(k * pos[:, 0]))), F, np.zeros((3, 3))

    fn.r_cut = 5.0
    return fn


def _sim(fn, tmp_path, z=5.0, n0=5, **kw):
    mu = K_B * T * np.log(z * _lambda_dB() ** 3 / L**3)
    pos = np.random.default_rng(0).uniform(0, L, (n0, 3))
    args = dict(
        temperature=T,
        mu=mu,
        species="Ar",
        mass=MASS,
        n_cycles=6000,
        md_steps_per_cycle=0,
        gcmc_attempts_per_cycle=50,
        max_disp=3.0,
        verbose=False,
        output_dir=str(tmp_path),
        seed=3,
    )
    args.update(kw)
    return CFCMCSimulation(np.eye(3) * L, pos, ["Ar"] * n0, fn, **args)


@pytest.mark.parametrize("beta_c", [0.0, 2.0])
def test_exactly_solvable_field_gives_the_muvt_averages(beta_c, tmp_path):
    r = _sim(_field_fn(beta_c * K_B * T), tmp_path).run()
    N_ex, bmu_ex = 5.0 * i0(beta_c), -np.log(i0(beta_c))
    for key in ("N_mean", "N_mean_lambda1"):
        assert abs(r[key] - N_ex) < 3 * r[key + "_se"], (key, r[key], N_ex)
    est = r["cfcmc"]
    assert abs(est["beta_mu_ex"] - bmu_ex) < 3 * est["beta_mu_ex_se"]
    assert est["N0_se"] < 0.1 * N_ex  # the test has resolving power
    assert r["acc_ins"] > 0.1 and r["acc_del"] > 0.1


@pytest.mark.parametrize(
    "x_frac,soft",
    [
        ([6.0, 6.0, 2.0], 10.0),  # dU = -0.016 eV: linear branch of g
        ([4.0, 4.0, 5.0], 10.0),  # dU = 5.7 eV (~550 kT): soft branch
        ([2.6, 2.4, 2.3], 10.0),  # dU = 1.9e6 eV: deep overlap, g bounded
        ([4.0, 4.0, 5.0], 0.0),  # linear coupling
    ],
    ids=["attractive", "overlap", "deep-overlap", "linear"],
)
def test_forces_are_the_gradient_of_the_coupled_energy(x_frac, soft, tmp_path):
    lj = PotentialEngine().add(
        LennardJonesPotential(
            {frozenset({"Ar"}): {"epsilon": 0.0103, "sigma": 3.405}},
            r_cut=6.0,
            shift=True,
        )
    )
    fn = make_force_fn(lj)
    pos = np.array(
        [[2, 2, 2], [5.6, 2.3, 2.1], [2.4, 5.8, 2.2], [5.5, 5.4, 5.9]], float
    )
    sim = CFCMCSimulation(
        np.eye(3) * 15.0, pos, ["Ar"] * 4, fn, temperature=120.0, mu=-0.2,
        species="Ar", verbose=False, output_dir=str(tmp_path), softcore_kT=soft,
    )
    sim.positions[-1] = x_frac
    sim._lam = 0.37
    U, F, _ = sim._get_forces()
    assert (sim._E1 - sim._E0 > 0) == (x_frac[0] != 6.0)
    assert F.shape == (5, 3)
    assert np.abs(F.sum(axis=0)).max() < 1e-9 * max(1.0, np.abs(F).max())
    h = 1e-6
    for k in range(5):
        for a in range(3):
            sim.positions[k, a] += h
            sim._refresh_energies()
            Up = sim._U()
            sim.positions[k, a] -= 2 * h
            sim._refresh_energies()
            Um = sim._U()
            sim.positions[k, a] += h
            fd = -(Up - Um) / (2 * h)
            assert F[k, a] == pytest.approx(fd, rel=1e-5, abs=1e-7)
    if x_frac[0] != 2.6:
        assert abs(F[-1]).max() > 1e-3  # the fractional atom is coupled
    else:  # the soft core all but switches off a deep overlap's force
        assert abs(F[-1]).max() < 1e-6


def test_softcore_is_exact_at_the_ends_and_bounded_between(tmp_path):
    sim = _sim(_field_fn(0.0), tmp_path)
    kT = K_B * T
    for dU in (-3 * kT, 0.5 * kT, 1e6 * kT):
        assert sim._g(0.0, dU) == 0.0
        assert sim._g(1.0, dU) == pytest.approx(dU)
    # an overlap costs at most lambda E_c / (1 - lambda)
    for lam in (0.01, 0.3, 0.9):
        assert sim._g(lam, 1e12 * kT) <= lam * 10 * kT / (1 - lam) * (1 + 1e-9)


def test_crossings_keep_arrays_and_cached_energies_consistent(tmp_path):
    sim = _sim(_field_fn(2 * K_B * T), tmp_path, n_cycles=0, max_dlambda=1.0)
    n_ins = n_del = 0
    for _ in range(400):
        N_before = sim.N
        sim._lambda_move()
        n_ins += sim.N > N_before
        n_del += sim.N < N_before
        assert len(sim.positions) == len(sim.atom_types) == sim.N + 1
        assert len(sim.velocities) == len(sim._masses) == len(sim._fix_mask)
        E0, E1 = sim._E0, sim._E1
        sim._refresh_energies()
        assert (sim._E0, sim._E1) == pytest.approx((E0, E1), abs=1e-12)
        assert 0.0 <= sim._lam <= 1.0
    assert n_ins > 10 and n_del > 10
    assert sim._nhc["n_dof"] == sim._compute_n_dof(sim.N + 1, sim._fix_mask)


def test_fixed_atoms_of_the_exchanged_species_are_refused(tmp_path):
    fix = np.zeros((5, 3), bool)
    fix[0] = True
    with pytest.raises(ValueError, match="may not be fixed"):
        _sim(_field_fn(0.0), tmp_path, fix_atoms=fix)


def test_a_bare_esm_calculator_is_refused(tmp_path):
    fn = _field_fn(0.0)
    fn.kind = "esm"
    with pytest.raises(ValueError, match="coulomb_method: esm"):
        _sim(fn, tmp_path)
