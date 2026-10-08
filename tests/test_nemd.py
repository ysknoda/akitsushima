"""ADR-0020: per-atom heat currents, the Green-Kubo kappa decomposition,
and NEMD with Langevin heat baths."""
import numpy as np
import pytest

from akitsushima.engine import LennardJonesPotential, PotentialEngine
from akitsushima.fitting import make_force_fn
from akitsushima.solvers import ThermalConductivityCalculator
from akitsushima.solvers.nemd import NEMDSimulation
from akitsushima.solvers.qha import build_supercell

_FCC = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])


def write_fcc_ar(path, n=(4, 4, 16), a=5.30):
    """fcc Ar for the nemd template (LJ cutoff 8.5 A < half of 21.2 A)."""
    from akitsushima.io.poscar import write_poscar
    L, P, T = build_supercell(np.eye(3) * a, _FCC * a, ["Ar"] * 4, list(n))
    write_poscar(str(path), L, P, T)


def _lj_small(species=("X",), n=(3, 3, 6)):
    """A small LJ crystal (sigma 2 A, cutoff 4 A, fcc a = 3.17 A) whose
    3x3 cross-section is wide enough for the cutoff; two species
    alternate when given two."""
    sig, eps = 2.0, 0.01
    pairs = {}
    for i, a in enumerate(species):
        for b in species[i:]:
            pairs[frozenset({a, b})] = {"epsilon": eps, "sigma": sig}
    fn = make_force_fn(PotentialEngine().add(
        LennardJonesPotential(pairs, r_cut=4.0, shift=True)))
    a0 = 2 ** (1 / 6) * sig * np.sqrt(2)
    L, P, _ = build_supercell(np.eye(3) * a0, _FCC * a0, ["X"] * 4, list(n))
    T = [species[k % len(species)] for k in range(len(P))]
    return fn, L, P, T


# --------------------------------------------------- per-atom heat current

@pytest.mark.parametrize("kind", ["lj", "tersoff"])
def test_per_atom_heat_currents_add_up_to_the_heat_flux(kind):
    """sum_i J_i = V J of _compute_heat_flux, for a pair and a many-body
    potential, on a rattled configuration with random velocities."""
    rng = np.random.default_rng(3)
    if kind == "lj":
        fn, L, P, T = _lj_small(("X", "Y"))
        masses = np.where(np.array(T) == "X", 40.0, 20.0)
    else:
        from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml
        from test_phonon_thermal import _SI
        fn = _build_force_fn_from_yaml({"type": "Tersoff", "species": ["Si"],
                                        "params": _SI})[0]
        base = np.vstack([_FCC, _FCC + 0.25])
        L, P, T = build_supercell(np.eye(3) * 5.432, base * 5.432, ["Si"] * 8, 2)
        masses = None
    P = P + rng.normal(0, 0.05, P.shape)
    tc = ThermalConductivityCalculator(L, P, T, fn, masses=masses)
    vel = rng.normal(0, 0.01, P.shape)
    E, F, S = fn(L, P, T)
    pdata = tc._gather_pair_data(P)
    e_i = tc._per_atom_energy(P, F, float(E), pdata)
    J = tc._compute_heat_flux(P, vel, F, S, e_i, pdata)
    Ji = tc._per_atom_heat_current(vel, e_i, pdata, S)
    V = abs(np.linalg.det(L))
    assert np.allclose(Ji.sum(axis=0) / V, J, rtol=1e-12, atol=1e-16)


def test_kappa_decomposition_adds_up_to_kappa(tmp_path):
    """Species and regions of a binary LJ crystal: each decomposition sums
    to compute_kappa's tensor (the integral is linear in J_g)."""
    fn, L, P, T = _lj_small(("X", "Y"))
    masses = np.where(np.array(T) == "X", 40.0, 20.0)
    tc = ThermalConductivityCalculator(L, P, T, fn, masses=masses, temperature=20.0)
    tc.run_md(n_steps=400, timestep=2.0, equil_steps=0, heat_flux_every=2,
              traj_every=10**6, thermo_every=10**6, output_dir=str(tmp_path),
              verbose=False, decompose={"species": True, "save_atom_flux": True,
                                        "regions": {"axis": "c",
                                                    "bounds": [0, 0.3, 0.7, 1.0]}})
    kappa = tc.compute_kappa(tau_max=0.1, n_blocks=2, verbose=False)
    d = tc.compute_kappa_decomposition(tau_max=0.1, verbose=False)
    assert set(d["species"]) == {"X", "Y"} and len(d["regions"]) == 3
    for name in ("species", "regions"):
        total = sum(d[name].values())
        assert np.allclose(total, kappa, rtol=1e-9, atol=1e-12 * np.max(np.abs(kappa)))
        assert d["sum_rule_residual"][name] <= 1e-9 * np.max(np.abs(kappa))
    flux = np.load(tmp_path / "atom_heat_flux.npy")
    assert flux.shape == (200, len(T), 3)
    assert np.allclose(flux.sum(axis=1), tc._J_series, rtol=1e-5, atol=1e-9)


# --------------------------------------------------- NEMD

def _nemd(tmp_path=None, **kw):
    fn, L, P, T = _lj_small(n=(3, 3, 8))
    args = dict(T_hot=30.0, T_cold=10.0, axis="c", hot=(0.0, 0.125),
                cold=(0.5, 0.625), damping=50.0, timestep=2.0, n_bins=8, seed=7)
    args.update(kw)
    return NEMDSimulation(L, P, T, fn, masses=np.full(len(T), 40.0), **args), fn


def test_energy_changes_only_by_the_bath_heat():
    """E(t) - E(0) = Q_hot + Q_cold up to the Verlet error (dt 0.5 fs)."""
    sim, _ = _nemd(timestep=0.5)
    st = sim.run(50, 300, sample_every=5, verbose=False)
    Q = abs(st["Q_hot"]) + abs(st["Q_cold"])
    assert Q > 1e-3
    assert abs(st["E_minus_Q_drift_eV"]) < 2e-3 * Q


def test_restart_continues_bit_for_bit(tmp_path, monkeypatch):
    """A run killed after a checkpoint and re-run with the same arguments
    ends exactly where an uninterrupted run does (positions, velocities,
    bath heat, profiles): the RNG state is part of the checkpoint."""
    ref, _ = _nemd()
    ref.run(40, 160, sample_every=5, verbose=False)
    sim, fn = _nemd()
    calls = {"n": 0}

    def dying(lat, pos, types, **k):
        calls["n"] += 1
        if calls["n"] == 131:
            raise KeyboardInterrupt
        return fn(lat, pos, types, **k)
    sim.fn = dying
    ck = str(tmp_path / "ck.npz")
    with pytest.raises(KeyboardInterrupt):
        sim.run(40, 160, sample_every=5, checkpoint=ck, checkpoint_every=25,
                verbose=False)
    sim2, _ = _nemd()
    st = sim2.run(40, 160, sample_every=5, checkpoint=ck, checkpoint_every=25,
                  verbose=False)
    for k in ("pos", "vel", "ke_bin", "n_bin", "j_bin", "q_block"):
        assert np.array_equal(st[k], ref.state[k]), k
    assert st["Q_hot"] == ref.state["Q_hot"] and st["Q_cold"] == ref.state["Q_cold"]


def test_baths_thermalise_to_their_temperature():
    """Both baths at 20 K over the whole cell: the mean temperature of the
    sampled profile is 20 K within its statistical error."""
    sim, _ = _nemd(T_hot=20.0, T_cold=20.0, hot=(0.0, 0.5), cold=(0.5, 1.0),
                   damping=20.0)
    sim.run(300, 1200, sample_every=4, verbose=False)
    r = sim.results()
    T_mean = np.sum(r["T_K"] * r["atoms_per_bin"]) / np.sum(r["atoms_per_bin"])
    assert T_mean == pytest.approx(20.0, rel=0.05)


def test_results_recover_kappa_and_kapitza_from_a_known_profile():
    """Fed a piecewise-linear profile with a 2 K jump at z = 0.25 and a
    known bath rate, results() returns the gradient, kappa = J / |dT/dz|
    and R = dT / J exactly."""
    sim, _ = _nemd(n_bins=32, hot=(0.0, 0.0625), cold=(0.5, 0.5625))
    nb = 32
    z = (np.arange(nb) + 0.5) / nb
    Lz = sim.length
    g = -0.5                                            # K per A
    T = np.where(z < 0.5, 40 + g * z * Lz, 40 + g * (1 - z) * Lz)
    T = T - np.where((z > 0.25) & (z < 0.5), 2.0, 0.0)  # jump at 0.25
    from akitsushima.constants import K_B
    ke = 1.5 * K_B * T * 10.0                           # 10 atoms per bin
    rate = 1e-4                                          # eV/fs
    sim.state = {"ke_bin": np.tile(ke / 4, (4, 1)), "n_bin": np.full((4, nb), 10 / 4),
                 "j_bin": np.zeros((4, nb)), "n_jsamp": np.zeros(4),
                 "n_prod": 1000, "n_samples": 1, "Q_hot_prod": rate * 1000 * sim.dt,
                 "Q_cold_prod": -rate * 1000 * sim.dt}
    r = sim.results(fit_fraction=0.2, interfaces=[0.25], kapitza_window=0.1,
                    kapitza_skip=0)
    J = rate / (2 * sim.area)
    seg = r["segments"]
    assert seg[1]["gradient_K_per_A"] == pytest.approx(-g, rel=1e-9)
    assert seg[0]["gradient_sign_ok"] and seg[1]["gradient_sign_ok"]
    assert seg[1]["kappa_W_mK"] == pytest.approx(J / abs(g) * 1.602176634e6, rel=1e-9)
    assert r["kapitza"][0]["delta_T_K"] == pytest.approx(2.0, rel=1e-9)
    assert r["kapitza"][0]["R_m2K_per_W"] == pytest.approx(
        2.0 / (J * 1.602176634e16), rel=1e-9)


def test_nemd_refusals():
    fn, L, P, T = _lj_small(n=(3, 3, 8))
    with pytest.raises(ValueError, match="no atoms"):
        NEMDSimulation(L, P, T, fn, hot=(0.01, 0.02), cold=(0.5, 0.6))
    with pytest.raises(ValueError, match="overlap"):
        NEMDSimulation(L, P, T, fn, hot_atoms=[0, 1], cold_atoms=[1, 2])

    def esm(lattice, positions, atom_types, **kw):
        return 0.0, np.zeros_like(positions), np.zeros((3, 3))
    esm.coulomb = "esm"
    with pytest.raises(ValueError, match="ESM"):
        NEMDSimulation(L, P, T, esm)
