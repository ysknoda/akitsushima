"""Langevin thermostat of md, BAOAB (ADR-0032).

1. Independent harmonic oscillators: <x^2> and the kinetic temperature
   against the analytic canonical values, and <x^2> the same at two
   time steps (BAOAB samples a harmonic configuration exactly at any
   stable dt).  With the centre-of-mass velocity removed and the start
   at the minima, the centre of mass stays there, so each component has
   <x^2> = (kT/k)(1 - 1/N).
2. SW silicon: mean temperature at the target, centre of mass at rest.
3. Fixed atoms keep zero velocity; no centre-of-mass removal then.
4. Refusals: velocity_rescale_steps > 0, damping <= 0, and a restart
   with another damping.  (The bitwise restart is in test_md_restart.py.)
"""

import numpy as np
import pytest
from test_md_restart import SW_SI, _structure

import akitsushima as ak
from akitsushima.constants import EV_ANG_TO_AMU_ANG_FS2, K_B

T = 300.0
K_SPRING = 1.0  # eV/A^2
N_SIDE = 4  # 64 oscillators on a simple cubic grid, 4 A apart


class _Harmonic:
    """E = k/2 sum |r_i - r0_i|^2 (minimum image), recording sum x^2."""

    uses_neighbour_list = False

    def __init__(self, r0, lattice):
        self.r0, self.inv = r0, np.linalg.inv(lattice)
        self.lattice = lattice
        self.record = False
        self.sum_x2, self.n = [], 0

    def __call__(self, lattice, positions, atom_types, neigh_list=None):
        d = positions - self.r0
        f = d @ self.inv
        d = (f - np.round(f)) @ lattice
        if self.record:
            self.sum_x2.append(float(np.mean(d * d)))
        return 0.5 * K_SPRING * float(np.sum(d * d)), -K_SPRING * d, np.zeros((3, 3))


def _oscillators():
    g = np.arange(N_SIDE) * 4.0
    r0 = np.array([[x, y, z] for x in g for y in g for z in g], float)
    lat = np.eye(3) * 4.0 * N_SIDE
    return lat, r0


def _run_harmonic(tmp_path, dt, n_steps, seed=11):
    lat, r0 = _oscillators()
    fn = _Harmonic(r0, lat)
    sim = ak.LangevinSimulation(
        lat,
        r0.copy(),
        ["Si"] * len(r0),
        force_fn=fn,
        r_cut=1.0,
        temperature=T,
        timestep=dt,
        n_steps=n_steps,
        equil_steps=0,
        damping=100.0,
        output_dir=str(tmp_path),
        verbose=False,
        thermo_every=1,
        traj_every=10**9,
        seed=seed,
        timing=False,
        save_plots=False,
    )
    # equilibrate 5 damping times, then sample
    n_eq = int(500.0 / dt)
    calls = {"n": 0}
    real = fn.__call__

    def counting(*a, **k):
        calls["n"] += 1
        fn.record = calls["n"] > n_eq
        return real(*a, **k)

    sim._force_fn = counting
    res = sim.run()
    x2 = np.array(fn.sum_x2)
    Ts = np.array([r["T_inst"] for r in res["thermo_log"]])[n_eq:]
    return x2, Ts, len(r0)


def _mean_and_error(series, n_blocks=20):
    blocks = np.array_split(series, n_blocks)
    means = np.array([b.mean() for b in blocks])
    return float(series.mean()), float(means.std(ddof=1) / np.sqrt(n_blocks))


@pytest.mark.slow
def test_harmonic_oscillators_sample_the_canonical_distribution(tmp_path):
    x2_a, Ts, N = _run_harmonic(tmp_path / "a", dt=2.0, n_steps=40000)
    want = K_B * T / K_SPRING * (1.0 - 1.0 / N)
    m, e = _mean_and_error(x2_a)
    assert abs(m - want) < 4 * e and abs(m / want - 1) < 0.03, (m, want, e)
    # kinetic temperature (n_dof = 3N - 3, the COM velocity is removed)
    mT, eT = _mean_and_error(Ts)
    assert abs(mT - T) < 4 * eT and abs(mT / T - 1) < 0.03, (mT, eT)
    # BAOAB: the configurational average does not depend on dt
    # dt = 16 fs is omega dt = 0.30: a velocity-Verlet + O splitting would
    # be off by (omega dt / 2)^2 = 2.2 % here, BAOAB by nothing
    x2_b, _, _ = _run_harmonic(tmp_path / "b", dt=16.0, n_steps=5000, seed=12)
    mb, eb = _mean_and_error(x2_b)
    assert abs(mb - m) < 4 * np.hypot(e, eb), (m, mb, e, eb)
    assert abs(mb - want) < 4 * eb, (mb, want, eb)


@pytest.mark.slow
def test_sw_silicon_holds_the_target_temperature_and_rest_frame(tmp_path):
    lat, pos, types = _structure()
    sim = ak.LangevinSimulation(
        lat,
        pos,
        types,
        force_fn=ak.force_fn_from_config(SW_SI),
        r_cut=3.77,
        temperature=600.0,
        timestep=1.0,
        n_steps=2500,
        equil_steps=500,
        damping=100.0,
        output_dir=str(tmp_path),
        verbose=False,
        thermo_every=1,
        traj_every=10**9,
        seed=4,
        timing=False,
        save_plots=False,
    )
    res = sim.run()
    Ts = np.array([r["T_inst"] for r in res["thermo_log"]])[500:]
    mT, eT = _mean_and_error(Ts)
    assert abs(mT - 600.0) < 4 * eT, (mT, eT)
    p = (sim.masses[:, None] * res["velocities"]).sum(axis=0)
    assert np.abs(p).max() < 1e-10
    # E_tot - Q is the integrator check: small drift at dt = 1 fs
    emq = np.array([r["e_minus_q"] for r in res["thermo_log"]])
    assert np.ptp(emq) < 0.02 * len(pos) * K_B * 600.0


def test_fixed_atoms_stay_fixed(tmp_path):
    lat, pos, types = _structure()
    fix = np.zeros((len(pos), 3), bool)
    fix[:4] = True
    sim = ak.LangevinSimulation(
        lat,
        pos,
        types,
        force_fn=ak.force_fn_from_config(SW_SI),
        r_cut=3.77,
        temperature=600.0,
        timestep=1.0,
        n_steps=30,
        damping=50.0,
        output_dir=str(tmp_path),
        verbose=False,
        fix_atoms=fix,
        seed=2,
        timing=False,
        save_plots=False,
    )
    res = sim.run()
    assert np.all(res["velocities"][:4] == 0.0)
    assert np.allclose(
        res["positions"][:4], pos[:4] - np.floor(pos[:4] @ np.linalg.inv(lat)) @ lat
    )


def test_refusals(tmp_path):
    lat, pos, types = _structure()
    kw = dict(
        force_fn=ak.force_fn_from_config(SW_SI),
        r_cut=3.77,
        n_steps=5,
        output_dir=str(tmp_path),
        verbose=False,
        timing=False,
        save_plots=False,
    )
    with pytest.raises(ValueError, match="velocity_rescale_steps"):
        ak.LangevinSimulation(lat, pos, types, velocity_rescale_steps=10, **kw)
    with pytest.raises(ValueError, match="damping"):
        ak.LangevinSimulation(lat, pos, types, damping=0.0, **kw)
    ck = str(tmp_path / "md_checkpoint.npz")
    ak.LangevinSimulation(lat, pos, types, damping=50.0, **kw).run(checkpoint=ck)
    with pytest.raises(ValueError, match="damping"):
        ak.LangevinSimulation(lat, pos, types, damping=80.0, **kw).run(checkpoint=ck)
    # a Nose-Hoover checkpoint is not continued as Langevin
    ck2 = str(tmp_path / "nhc.npz")
    ak.NVTSimulation(lat, pos, types, **kw).run(checkpoint=ck2)
    with pytest.raises(ValueError, match="ensemble"):
        ak.LangevinSimulation(lat, pos, types, **kw).run(checkpoint=ck2)
    _ = EV_ANG_TO_AMU_ANG_FS2  # unit used by the analytic values above


def test_cli_selects_langevin_and_refuses_misuse(tmp_path):
    from test_cli_smoke import _LJ_POT_FITTED, _al_supercell_poscar, _ns, _write_yaml

    from akitsushima.cli.cmd_md import cmd_md

    poscar = str(_al_supercell_poscar(tmp_path))

    def run(**md):
        sec = {"ensemble": "nvt", "thermostat": "langevin", "damping": 50.0,
               "output_dir": str(tmp_path / "out"), "temperature": 300.0,
               "n_steps": 20, "equil_steps": 0, "verbose": False,
               "thermo_every": 5, "traj_every": 5}
        sec.update(md)
        path = tmp_path / "md.yaml"
        _write_yaml(path, {"structure": {"poscar": poscar},
                           "potential": _LJ_POT_FITTED, "md": sec})
        return cmd_md(_ns(path))

    assert run() == 0
    head = (tmp_path / "out" / "thermo.log").read_text().splitlines()[0]
    assert "E-Q(eV)" in head
    with pytest.raises(ValueError, match="velocity_rescale_steps"):
        run(velocity_rescale_steps=100, restart=False)
    with pytest.raises(ValueError, match="nvt only"):
        run(ensemble="npt", restart=False)
    with pytest.raises(ValueError, match="thermostat must be"):
        run(thermostat="berendsen", restart=False)
