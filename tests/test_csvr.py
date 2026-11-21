"""CSVR thermostat of md (ADR-0039).

1. The rescaling alone (no dynamics) propagates the kinetic energy by the
   exact process of Bussi, Donadio and Parrinello, J. Chem. Phys. 126,
   014101 (2007): stationary gamma distribution (mean N_f kT/2, variance
   N_f (kT)^2/2) and conditional mean c K + (1 - c) K_t.
2. SW silicon, 64 atoms, 600 K: mean T, Var(K) against N_f (kT)^2/2 and
   <E_pot> against the Langevin reference within 4 sigma; total momentum
   conserved; small E - Q drift.  The same run with the noise switched
   off (a deterministic, Berendsen-like rescaling) fails the variance.
   Not on independent harmonic oscillators: a global thermostat cannot
   make them ergodic (ADR-0039 decision 6).
3. Fixed atoms keep zero velocity.
4. Refusals and the CLI.  (The bitwise restart is in test_md_restart.py,
   the MPI agreement in test_mpi_cli.py.)
"""

import numba
import numpy as np
import pytest
from test_md_restart import SW_SI, _structure

import akitsushima as ak
from akitsushima.constants import K_B

T = 600.0
# <E_pot>/N of SW Si (64 atoms, the rattled start of test_md_restart) at
# 600 K from a 200 ps Langevin run, 40-block error (ADR-0039 result).
E_REF, E_REF_ERR = -4.25902, 0.00021


@pytest.fixture
def one_thread():
    """64 atoms are too few for threads: 0.8 ms per SW call on one numba
    thread, 5.3 ms on ten."""
    n = numba.get_num_threads()
    numba.set_num_threads(1)
    yield
    numba.set_num_threads(n)


def _sim(tmp_path, cls=ak.CSVRSimulation, **over):
    lat, pos, types = _structure()
    kw = {
        "force_fn": ak.force_fn_from_config(SW_SI),
        "r_cut": 3.77,
        "temperature": T,
        "timestep": 1.0,
        "n_steps": 10,
        "equil_steps": 0,
        "damping": 100.0,
        "output_dir": str(tmp_path),
        "verbose": False,
        "thermo_every": 1,
        "traj_every": 10**9,
        "seed": 7,
        "timing": False,
        "save_plots": False,
    }
    kw.update(over)
    return cls(lat, pos.copy(), types, **kw)


def _mean_and_error(series, n_blocks=20):
    blocks = np.array_split(np.asarray(series), n_blocks)
    means = np.array([b.mean() for b in blocks])
    return float(np.mean(series)), float(means.std(ddof=1) / np.sqrt(n_blocks))


def test_rescaling_alone_samples_the_gamma_distribution(tmp_path):
    sim = _sim(tmp_path, damping=2.0)  # c = exp(-0.25): short correlation
    sim._prepare()
    nf, kT = sim.n_dof, K_B * T
    Kt = 0.5 * nf * kT
    K = []
    for _ in range(40000):
        sim._rescale()
        K.append(sim._kinetic_energy())
    K = np.array(K)[1000:]
    m, e = _mean_and_error(K)
    assert abs(m - Kt) < 4 * e, (m, Kt, e)
    v, ev = _mean_and_error((K - K.mean()) ** 2 / (nf * kT**2 / 2))
    assert abs(v - 1.0) < 4 * ev and abs(v - 1.0) < 0.1, (v, ev)
    # E[K' | K] = c K + (1 - c) K_t: the regression slope of K' on K is c
    c = sim._c_half
    slope, icpt = np.polyfit(K[:-1], K[1:], 1)
    assert slope == pytest.approx(c, abs=0.02)
    assert icpt == pytest.approx((1 - c) * Kt, rel=0.1)


def _kinetic_stats(res, sim, n_eq):
    rows = res["thermo_log"][n_eq:]
    K = np.array([r["e_kin"] for r in rows])
    nf, kT = sim.n_dof, K_B * T
    Tm, eT = _mean_and_error(2 * K / (nf * K_B))
    v, ev = _mean_and_error((K - K.mean()) ** 2 / (nf * kT**2 / 2))
    U = np.array([r["e_pot"] for r in rows]) / len(sim.atom_types)
    u, eu = _mean_and_error(U)
    return Tm, eT, v, ev, u, eu


@pytest.mark.slow
def test_sw_silicon_samples_the_canonical_kinetic_energy(tmp_path, one_thread):
    n_eq = 2000
    sim = _sim(tmp_path / "csvr", n_steps=20000, equil_steps=n_eq)
    res = sim.run()
    Tm, eT, v, ev, u, eu = _kinetic_stats(res, sim, n_eq)
    assert abs(Tm - T) < 4 * eT, (Tm, eT)
    assert abs(v - 1.0) < 4 * ev, (v, ev)
    assert abs(u - E_REF) < 4 * np.hypot(eu, E_REF_ERR), (u, eu, E_REF)
    # one factor for every atom: the total momentum is conserved
    p = (sim.masses[:, None] * res["velocities"]).sum(axis=0)
    assert np.abs(p).max() < 1e-10
    # E_tot - Q, the effective energy of the paper: small drift at 1 fs
    emq = np.array([r["e_minus_q"] for r in res["thermo_log"]])
    assert np.ptp(emq) < 0.02 * len(sim.atom_types) * K_B * T

    # Control: without the noise the rescaling is deterministic
    # (Berendsen-like) and the kinetic energy fluctuates far too little.
    ctl = _sim(tmp_path / "ctl", n_steps=6000, equil_steps=n_eq)
    ctl._noise_scale = 0.0
    _, _, v0, ev0, _, _ = _kinetic_stats(ctl.run(), ctl, n_eq)
    assert abs(v0 - 1.0) > 4 * ev0 and v0 < 0.6, (v0, ev0)


def test_fixed_atoms_stay_fixed(tmp_path):
    fix = np.zeros((64, 3), bool)
    fix[:4] = True
    sim = _sim(tmp_path, n_steps=30, damping=50.0, fix_atoms=fix)
    assert sim.n_dof == 3 * 64 - 12
    res = sim.run()
    assert np.all(res["velocities"][:4] == 0.0)


def test_refusals(tmp_path):
    with pytest.raises(ValueError, match="velocity_rescale_steps"):
        _sim(tmp_path, velocity_rescale_steps=10)
    with pytest.raises(ValueError, match="damping"):
        _sim(tmp_path, damping=0.0)
    ck = str(tmp_path / "md_checkpoint.npz")
    _sim(tmp_path, damping=50.0).run(checkpoint=ck)
    with pytest.raises(ValueError, match="damping"):
        _sim(tmp_path, damping=80.0).run(checkpoint=ck)
    # a Langevin checkpoint is not continued as CSVR
    ck2 = str(tmp_path / "lang.npz")
    _sim(tmp_path, cls=ak.LangevinSimulation, damping=50.0).run(checkpoint=ck2)
    with pytest.raises(ValueError, match="ensemble"):
        _sim(tmp_path, damping=50.0).run(checkpoint=ck2)


def test_cli_selects_csvr_and_refuses_misuse(tmp_path):
    from test_cli_smoke import _LJ_POT_FITTED, _al_supercell_poscar, _ns, _write_yaml

    from akitsushima.cli.cmd_md import cmd_md

    poscar = str(_al_supercell_poscar(tmp_path))

    def run(**md):
        sec = {
            "ensemble": "nvt",
            "thermostat": "csvr",
            "damping": 50.0,
            "output_dir": str(tmp_path / "out"),
            "temperature": 300.0,
            "n_steps": 20,
            "equil_steps": 0,
            "verbose": False,
            "thermo_every": 5,
            "traj_every": 5,
        }
        sec.update(md)
        path = tmp_path / "md.yaml"
        _write_yaml(
            path,
            {"structure": {"poscar": poscar}, "potential": _LJ_POT_FITTED, "md": sec},
        )
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
