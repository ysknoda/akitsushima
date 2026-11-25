"""Thermo rows and trajectory frames labelled by completed steps (ADR-0048).

The row labelled k used to hold the state after k + 1 steps (k the
0-based loop index): time 0 was the state after one step and the last
row was n_steps - thermo_every.  Now, as in LAMMPS, row 0 is the initial
state and the row labelled c the state after c completed steps; frames
start at c = equil_steps.  The integration and the averages are
unchanged; a restart from a checkpoint written before ADR-0048 goes on
with the new labels after one comment line.
"""

import os
import re

import numpy as np
import pytest
from test_nemd import _lj_small
from test_npt_aniso import _si, _sw

import akitsushima as ak
from akitsushima.constants import K_B
from akitsushima.solvers import CFCMCSimulation
from akitsushima.solvers._checkpoint import load_checkpoint, save_checkpoint
from akitsushima.solvers.gcmc import GCMCMDSimulation
from akitsushima.solvers.nemd import NEMDSimulation
from akitsushima.solvers.thermal_conductivity import ThermalConductivityCalculator

S = _si()
CLASSES = {
    "nve": (ak.NVESimulation, {}),
    "nvt": (ak.NVTSimulation, {"tau_nhc": 50.0}),
    "langevin": (ak.LangevinSimulation, {"damping": 50.0}),
    "csvr": (ak.CSVRSimulation, {"damping": 50.0}),
    "npt": (ak.NPTSimulation, {"pressure": 1.0, "tau_nhc": 50.0, "tau_baro": 200.0}),
}


def _md(name, out, **kw):
    cls, extra = CLASSES[name]
    args = dict(
        force_fn=_sw(),
        r_cut=3.8,
        temperature=600.0,
        timestep=1.0,
        n_steps=6,
        equil_steps=2,
        seed=7,
        verbose=False,
        output_dir=str(out),
        thermo_every=1,
        traj_every=1,
    )
    if name != "nve":
        args["save_plots"] = False
    args.update(extra)
    args.update(kw)
    return cls(S["lattice"], S["positions"], S["atom_types"], **args)


def _rows(path):
    return [ln.split() for ln in open(path) if re.match(r"\s*\d", ln)]


def _frames(out):
    traj = [f for f in os.listdir(out) if f.startswith("XDATCAR")][0]
    return re.findall(
        r"^Direct configuration=\s*(\d+)",
        open(os.path.join(out, traj)).read(),
        flags=re.MULTILINE,
    )


@pytest.mark.parametrize("name", list(CLASSES))
def test_rows_and_frames_are_labelled_by_completed_steps(name, tmp_path):
    sim = _md(name, tmp_path / "a")
    sim.run()
    rows = _rows(tmp_path / "a" / "thermo.log")
    assert [int(r[0]) for r in rows] == list(range(9))  # 0 .. 8 steps
    assert [float(r[1]) for r in rows] == pytest.approx(np.arange(9) * 1e-3)
    assert len(_frames(tmp_path / "a")) == 7  # c = 2 .. 8
    # row 0 is the initial state: its E_pot is the start structure's
    e0 = _sw()(S["lattice"], S["positions"], S["atom_types"])[0]
    assert float(rows[0][2]) == pytest.approx(e0, abs=1e-6)

    _md(name, tmp_path / "b", thermo_every=3, traj_every=4).run()
    assert [int(r[0]) for r in _rows(tmp_path / "b" / "thermo.log")] == [0, 3, 6]
    assert len(_frames(tmp_path / "b")) == 2  # c = 4, 8


def test_averages_are_over_the_states_after_equilibration(tmp_path):
    """T_mean is unchanged: the mean over the states after steps
    equil_steps + 1 .. total, i.e. the rows with c > equil_steps."""
    sim = _md("nvt", tmp_path)
    r = sim.run()
    T = [row["T_inst"] for row in sim._thermo_log if row["step"] > 2]
    assert len(T) == 6
    assert r["T_mean"] == pytest.approx(np.mean(T), rel=1e-14)


def _strip_labels(path):
    meta, arrays = load_checkpoint(path)
    assert meta.pop("labels") == "completed"
    save_checkpoint(path, meta, arrays)


@pytest.mark.parametrize("name", ["nvt", "csvr", "npt"])
def test_restart_from_a_checkpoint_written_before_adr_0048(name, tmp_path):
    ref = _md(name, tmp_path / "ref", n_steps=8)
    ref.run()
    ref_rows = open(tmp_path / "ref" / "thermo.log").read().splitlines()

    ck = str(tmp_path / "ck.npz")
    _md(name, tmp_path / "new", n_steps=3).run(checkpoint=ck, checkpoint_every=5)
    sim = _md(name, tmp_path / "new", n_steps=8)
    sim.run(checkpoint=ck, checkpoint_every=5)
    # a new-format checkpoint continues bit for bit, without a note
    assert open(tmp_path / "new" / "thermo.log").read().splitlines() == ref_rows

    ck2 = str(tmp_path / "ck2.npz")
    _md(name, tmp_path / "old", n_steps=3).run(checkpoint=ck2, checkpoint_every=5)
    _strip_labels(ck2)
    old = _md(name, tmp_path / "old", n_steps=8)
    old.run(checkpoint=ck2, checkpoint_every=5)
    lines = open(tmp_path / "old" / "thermo.log").read().splitlines()
    note = [i for i, ln in enumerate(lines) if ln.startswith("# ADR-0048")]
    assert len(note) == 1
    assert lines[: note[0]] == ref_rows[: note[0]]  # rows up to step 5
    assert lines[note[0] + 1 :] == ref_rows[note[0] :]  # steps 6 .. 10
    np.testing.assert_array_equal(old.positions, ref.positions)
    np.testing.assert_array_equal(old.velocities, ref.velocities)
    # readers that skip '#' lines see the same rows
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / "old" / "thermo.log", skiprows=1),
        np.loadtxt(tmp_path / "ref" / "thermo.log", skiprows=1),
    )


def test_green_kubo_rows_and_heat_flux_labels(tmp_path):
    fn, L, P, T = _lj_small()
    tc = ThermalConductivityCalculator(
        L, P, T, fn, masses=np.full(len(T), 40.0), temperature=20.0
    )
    tc.run_md(
        n_steps=12,
        timestep=2.0,
        equil_steps=0,
        heat_flux_every=4,
        thermo_every=3,
        traj_every=6,
        output_dir=str(tmp_path),
        verbose=False,
        seed=7,
    )
    assert [int(r[0]) for r in _rows(tmp_path / "thermo_nve.log")] == [0, 3, 6, 9, 12]
    # the flux is sampled after steps 1, 5, 9 as before (kappa unchanged)
    assert [int(r[0]) for r in _rows(tmp_path / "heat_flux.dat")] == [1, 5, 9]
    steps = re.findall(r"step=(\d+)", open(tmp_path / "trajectory_tc.xyz").read())
    assert steps == ["0", "6", "12"]


def test_nemd_has_an_initial_row(tmp_path):
    fn, L, P, T = _lj_small(n=(3, 3, 8))
    sim = NEMDSimulation(
        L,
        P,
        T,
        fn,
        masses=np.full(len(T), 40.0),
        T_hot=30.0,
        T_cold=10.0,
        axis="c",
        hot=(0.0, 0.125),
        cold=(0.5, 0.625),
        damping=50.0,
        timestep=2.0,
        n_bins=8,
        seed=7,
    )
    th = tmp_path / "nemd.log"
    sim.run(
        equil_steps=10, n_steps=20, sample_every=1, thermo_path=str(th), verbose=False
    )
    rows = _rows(th)
    assert [int(r[0]) for r in rows] == [0, 10, 20, 30]
    assert [float(x) for x in rows[0][3:]] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("cls", [GCMCMDSimulation, CFCMCSimulation])
def test_gcmc_rows_are_labelled_by_completed_cycles(cls, tmp_path):
    fn, L, P, T = _lj_small()
    sim = cls(
        L,
        P,
        T,
        force_fn=fn,
        temperature=20.0,
        mu=K_B * 20.0 * np.log(0.02),
        species="X",
        mass=40.0,
        n_cycles=6,
        md_steps_per_cycle=2,
        gcmc_attempts_per_cycle=5,
        max_disp=0.2,
        thermo_every=2,
        traj_every=3,
        verbose=False,
        output_dir=str(tmp_path),
        seed=3,
    )
    r = sim.run()
    rows = _rows(tmp_path / "gcmc_md_thermo.log")
    assert [int(x[0]) for x in rows] == [0, 2, 4, 6]
    assert rows[0][5:] == ["n/a", "n/a", "n/a"]  # no move yet
    n_lines = _rows(tmp_path / "N_vs_cycle.dat")
    assert [int(x[0]) for x in n_lines] == list(range(7))
    assert len(r["N_history"]) == 6  # averages unchanged
    assert [row["cycle"] for row in r["thermo_log"]] == list(range(1, 7))
    if cls is CFCMCSimulation:
        lam = _rows(tmp_path / "lambda_vs_cycle.dat")
        assert [int(x[0]) for x in lam] == list(range(1, 7))
