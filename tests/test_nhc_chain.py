"""M-link Nose-Hoover chain (ADR-0044).

* nhc_chain = 2 repeats the former 2-link propagator bit for bit, and a
  checkpoint written by 0.3.0a7 (scalar chain state) continues to the
  same state as the uninterrupted 0.3.0a7 run.
* M > 2: H_NHC is conserved, a restart is bit for bit, and a restart with
  another chain length is refused.
* The heating time from the equipartition-halved start shrinks with
  three links (the reason for ADR-0044).
"""

import os
import shutil

import numpy as np
import pytest
from akitsushima.constants import K_B
from akitsushima.solvers.molecular_dynamics import (
    CSVRSimulation,
    NPTSimulation,
    NVTSimulation,
)
from test_solvers import _SW_SI_RCUT, _perturbed_si_supercell, _sw_si_force_fn

LEGACY = os.path.join(os.path.dirname(__file__), "data", "nhc_legacy_0.3.0a7")


def _kw(**over):
    kw = dict(
        force_fn=_sw_si_force_fn(),
        r_cut=_SW_SI_RCUT,
        temperature=600.0,
        timestep=1.0,
        thermo_every=10,
        traj_every=50,
        seed=7,
        verbose=False,
        save_plots=False,
    )
    kw.update(over)
    return kw


def _si():
    return _perturbed_si_supercell(1.0, n=2, sigma=0.05, seed=1)


def _old_two_link_step(sim):
    """The 0.3.0a7 body of _nhc_step on scalar state (reference copy)."""
    dt = sim.dt
    Q1, Q2 = sim.Q_nhc
    xi1, xi2 = sim.xi_nhc
    vxi1, vxi2 = sim.vxi_nhc
    kT = K_B * sim.temperature
    e_kin = sim._kinetic_energy()
    G1 = (2.0 * e_kin - sim.n_dof * kT) / Q1
    G2 = (Q1 * vxi1**2 - kT) / Q2
    vxi2 += G2 * (dt * 0.25)
    s_arg = float(np.clip(-vxi2 * dt * 0.125, -350.0, 350.0))
    s = np.exp(s_arg)
    vxi1 = s * (s * vxi1 + G1 * dt * 0.25)
    sc_arg = float(np.clip(-vxi1 * dt * 0.5, -350.0, 350.0))
    sim.velocities *= np.exp(sc_arg)
    sim.velocities *= ~sim._fix_mask
    xi1 += vxi1 * dt * 0.5
    xi2 += vxi2 * dt * 0.5
    e_kin = sim._kinetic_energy()
    G1 = (2.0 * e_kin - sim.n_dof * kT) / Q1
    vxi1 = s * (s * vxi1 + G1 * dt * 0.25)
    G2 = (Q1 * vxi1**2 - kT) / Q2
    vxi2 += G2 * (dt * 0.25)
    sim.xi_nhc = [xi1, xi2]
    sim.vxi_nhc = [vxi1, vxi2]


def test_two_links_repeat_the_former_propagator_bit_for_bit(tmp_path):
    lat, pos, types = _si()
    a = NVTSimulation(lat, pos, types, output_dir=str(tmp_path), n_steps=1, **_kw())
    b = NVTSimulation(lat, pos, types, output_dir=str(tmp_path), n_steps=1, **_kw())
    rng = np.random.default_rng(3)
    for sim in (a, b):
        sim.vxi_nhc = [1e-3, -2e-3]
        sim.xi_nhc = [0.1, -0.2]
        sim.velocities = sim.velocities * (1.0 + 0.3 * rng.random())
    b.velocities = a.velocities.copy()
    for _ in range(50):
        a._nhc_step()
        _old_two_link_step(b)
    assert np.array_equal(a.velocities, b.velocities)
    assert a.xi_nhc == b.xi_nhc and a.vxi_nhc == b.vxi_nhc


@pytest.mark.parametrize("bad", [1, 0, 2.5, True, -3])
def test_bad_chain_length_is_refused(tmp_path, bad):
    lat, pos, types = _si()
    with pytest.raises(ValueError, match="nhc_chain"):
        NVTSimulation(
            lat, pos, types, output_dir=str(tmp_path), n_steps=1, nhc_chain=bad, **_kw()
        )


def test_scalar_aliases_address_the_first_two_links(tmp_path):
    lat, pos, types = _si()
    sim = NVTSimulation(
        lat,
        pos,
        types,
        output_dir=str(tmp_path),
        n_steps=1,
        nhc_chain=4,
        tau_nhc=50.0,
        **_kw(),
    )
    assert len(sim.Q_nhc) == 4 and sim.nhc_chain == 4
    kT = K_B * 600.0
    assert sim.Q1 == pytest.approx(sim.n_dof * kT * 50.0**2, rel=1e-14)
    assert sim.Q_nhc[1:] == [sim.Q2] * 3
    assert sim.Q2 == pytest.approx(kT * 50.0**2, rel=1e-14)
    sim.Q2 = 7.0
    sim.vxi1 = 0.5
    assert sim.Q_nhc[1] == 7.0 and sim.vxi_nhc[0] == 0.5
    bare = NVTSimulation.__new__(NVTSimulation)
    bare.Q1 = 1.0
    assert bare.Q_nhc == [1.0, 0.0]


@pytest.mark.parametrize(
    "name,cls,extra",
    [
        ("nvt", NVTSimulation, {}),
        ("npt", NPTSimulation, dict(pressure=0.0, tau_baro=500.0)),
    ],
)
def test_legacy_checkpoint_continues_like_the_uninterrupted_run(
    tmp_path, name, cls, extra
):
    """A 0.3.0a7 checkpoint (chain as scalars xi1 .. vxi2) at step 150,
    carried on to step 300, lands on the uninterrupted 0.3.0a7 run."""
    d = tmp_path / name
    shutil.copytree(os.path.join(LEGACY, name), d)
    lat, pos, types = _si()
    sim = cls(
        lat, pos, types, output_dir=str(d), n_steps=300, nhc_chain=2, **_kw(), **extra
    )
    sim.run(checkpoint=str(d / "md_checkpoint.npz"))
    ref = np.load(os.path.join(LEGACY, "final_300.npz"))
    for key, got in (
        ("pos", sim.positions),
        ("vel", sim.velocities),
        ("lat", sim.lattice),
    ):
        want = ref[f"{name}_{key}"]
        assert np.max(np.abs(got - want)) <= 1e-12 * np.max(np.abs(want)), key


def test_legacy_checkpoint_refuses_another_chain_length(tmp_path):
    d = tmp_path / "nvt"
    shutil.copytree(os.path.join(LEGACY, "nvt"), d)
    lat, pos, types = _si()
    sim = NVTSimulation(
        lat, pos, types, output_dir=str(d), n_steps=300, nhc_chain=3, **_kw()
    )
    with pytest.raises(ValueError, match="nhc_chain"):
        sim.run(checkpoint=str(d / "md_checkpoint.npz"))


@pytest.mark.parametrize(
    "cls,extra",
    [
        (NVTSimulation, {}),
        (NPTSimulation, dict(pressure=0.0, tau_baro=500.0)),
        (NPTSimulation, dict(pressure=0.0, tau_baro=500.0, cell_dof="aniso")),
    ],
)
def test_three_link_restart_is_bit_for_bit(tmp_path, cls, extra):
    lat, pos, types = _si()
    full = cls(
        lat,
        pos,
        types,
        output_dir=str(tmp_path / "full"),
        n_steps=200,
        nhc_chain=3,
        **_kw(),
        **extra,
    )
    full.run()
    ck = str(tmp_path / "part" / "ck.npz")
    cls(
        lat,
        pos,
        types,
        output_dir=str(tmp_path / "part"),
        n_steps=100,
        nhc_chain=3,
        **_kw(),
        **extra,
    ).run(checkpoint=ck)
    meta = np.load(ck)
    assert meta["state_xi_nhc"].shape == (3,)
    part = cls(
        lat,
        pos,
        types,
        output_dir=str(tmp_path / "part"),
        n_steps=200,
        nhc_chain=3,
        **_kw(),
        **extra,
    )
    part.run(checkpoint=ck)
    assert np.array_equal(part.positions, full.positions)
    assert np.array_equal(part.velocities, full.velocities)
    assert part.vxi_nhc == full.vxi_nhc and part.xi_nhc == full.xi_nhc
    other = cls(
        lat,
        pos,
        types,
        output_dir=str(tmp_path / "part"),
        n_steps=300,
        nhc_chain=4,
        **_kw(),
        **extra,
    )
    with pytest.raises(ValueError, match="nhc_chain"):
        other.run(checkpoint=ck)


def test_stochastic_thermostat_checkpoint_has_no_chain(tmp_path):
    lat, pos, types = _si()
    ck = str(tmp_path / "ck.npz")
    CSVRSimulation(
        lat, pos, types, output_dir=str(tmp_path), n_steps=20, nhc_chain=3, **_kw()
    ).run(checkpoint=ck)
    z = np.load(ck)
    assert "state_xi_nhc" not in z.files
    sim = CSVRSimulation(lat, pos, types, output_dir=str(tmp_path), n_steps=40, **_kw())
    sim.run(checkpoint=ck)  # chain length is not a CSVR setting


@pytest.mark.parametrize("M", [2, 3, 4])
def test_h_nhc_is_conserved_for_any_chain_length(tmp_path, M):
    """2 ps of SW Si at 600 K, tau 50 fs: H_NHC moves by less than 1 % of
    the thermal energy n_dof kT / 2 (measured: below 0.05 %)."""
    lat, pos, types = _si()
    sim = NVTSimulation(
        lat,
        pos,
        types,
        output_dir=str(tmp_path),
        n_steps=2000,
        nhc_chain=M,
        tau_nhc=50.0,
        **_kw(thermo_every=20),
    )
    res = sim.run()
    h = np.array([r["H_nhc"] for r in res["thermo_log"]])
    thermal = 0.5 * sim.n_dof * K_B * 600.0
    assert np.ptp(h) < 0.01 * thermal, (M, np.ptp(h), thermal)


def _t95(sim):
    res = sim.run()
    t = np.array([r["time_ps"] for r in res["thermo_log"]])
    T = np.array([r["T_inst"] for r in res["thermo_log"]])
    w = int(round(0.5 / (t[1] - t[0])))
    run = np.convolve(T, np.ones(w) / w, mode="valid")
    hit = np.nonzero(run >= 0.95 * sim.temperature)[0]
    return t[hit[0] + w - 1] if len(hit) else np.inf


@pytest.mark.slow
def test_three_links_heat_from_the_halved_start_faster(tmp_path):
    """SW Si 512 atoms from its minimum with velocities at T0: equipartition
    halves T at once.  The 2-link chain needs about ln(1.9) tau
    sqrt(n_dof) to recover 0.95 T0; three links lift that cap."""
    from test_solvers import _A_EQ_SI

    n = 4
    basis = np.array(
        [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ],
        float,
    )
    frac = np.concatenate(
        [(basis + [i, j, k]) / n for i in range(n) for j in range(n) for k in range(n)]
    )
    lat = np.eye(3) * _A_EQ_SI * n
    pos = frac @ lat
    types = ["Si"] * len(pos)
    t = {}
    for M in (2, 3):
        sim = NVTSimulation(
            lat,
            pos,
            types,
            output_dir=str(tmp_path / str(M)),
            n_steps=6000,
            nhc_chain=M,
            tau_nhc=100.0,
            **_kw(temperature=300.0, thermo_every=10, traj_every=6000),
        )
        t[M] = _t95(sim)
    est = np.log(1.9) * 100.0 * np.sqrt(3 * 512 - 3) / 1000.0
    assert t[2] > 0.8 * est, t
    assert t[3] < 0.5 * t[2], t
