"""Ensemble of atom-centred ELMs (ADR-0033).

1. The mean is the force field: forces and stress are finite-difference
   derivatives of the mean energy, and equal the mean of the members'.
2. Members that are the same model give zero spread; different seeds
   give a positive one.
3. Seeds: seed + 1000 m by default, an explicit list when given, and the
   first member's chosen hyperparameters reused by the others.
4. Save / load round trip through load_mlip_model; pickling the force
   field.
5. An md run with an ensemble restarts bit for bit.
"""

import numpy as np
import pytest
from test_elm_options import _acsf
from test_mlip_atom_centred import _fd4_check, _random_structs

from akitsushima.fitting import AtomicELMEnsemble, AtomicELMModel
from akitsushima.fitting.mlip_engine import MLIPForceField, load_mlip_model


def _fit(seeds=None, n=3, **kw):
    kw.setdefault("n_hidden", 24)
    ens = AtomicELMEnsemble(
        n_members=n, seeds=seeds, r0=0.1, eps0=0.01, reg=1e-6, verbose=False, **kw
    )
    return ens.fit(_random_structs(10, 0), _acsf())


@pytest.fixture(scope="module")
def ens():
    return _fit()


def test_mean_is_a_conservative_force_field(ens):
    ff = MLIPForceField(ens)
    s = _random_structs(3, 5)[1]
    lat, pos, types = s["lattice"], s["positions"], s["atom_types"]
    E, F, S = ff(lat, pos, types)
    # the mean of the members, member by member
    singles = [MLIPForceField(m)(lat, pos, types) for m in ens.members]
    assert E == pytest.approx(np.mean([e for e, _, _ in singles]), rel=1e-12)
    np.testing.assert_allclose(
        F, np.mean([f for _, f, _ in singles], axis=0), rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(
        S, np.mean([x for _, _, x in singles], axis=0), rtol=0, atol=1e-12
    )
    _fd4_check(ff, lat, pos, types, 1e-4, h_strain=3e-5)


def test_spread_is_zero_for_identical_members_and_positive_otherwise(ens):
    ff = MLIPForceField(ens)
    assert ff.uncertainty() is None  # before any call
    s = _random_structs(3, 5)[0]
    ff(s["lattice"], s["positions"], s["atom_types"])
    u = ff.uncertainty()
    assert set(u) == {"sigma_E", "sigma_F_max", "sigma_S"}
    assert all(v > 0 for v in u.values())

    same = AtomicELMEnsemble(seeds=[1, 2], verbose=False)
    same.members = [ens.members[0], ens.members[0]]
    same.physics_config, same._fitted = "", True
    ff2 = MLIPForceField(same)
    ff2(s["lattice"], s["positions"], s["atom_types"])
    assert all(v == 0.0 for v in ff2.uncertainty().values())
    # a single model has no spread
    single = MLIPForceField(ens.members[0])
    single(s["lattice"], s["positions"], s["atom_types"])
    assert single.uncertainty() is None


def test_seeds_and_shared_hyperparameters():
    e = _fit(n=3, seed=7, reg_search=[1e-8, 1e-6, 1e-4])
    assert e.seeds == [7, 1007, 2007]
    assert [m.seed for m in e.members] == [7, 1007, 2007]
    assert len({m.reg for m in e.members}) == 1
    assert e.members[1].reg_search is None  # the search ran once
    e2 = _fit(seeds=[42, 1, 2])
    assert [m.seed for m in e2.members] == [42, 1, 2]
    # the first member is the single model of that seed
    single = AtomicELMModel(
        n_hidden=24, r0=0.1, eps0=0.01, reg=1e-6, seed=42, verbose=False
    ).fit(_random_structs(10, 0), _acsf())
    np.testing.assert_array_equal(e2.members[0].units[0]._beta, single.units[0]._beta)
    with pytest.raises(ValueError, match="distinct seeds"):
        AtomicELMEnsemble(seeds=[1, 1])
    with pytest.raises(ValueError, match="member_class"):
        AtomicELMEnsemble(member_class="AtomicNNModel")


def test_save_load_round_trip(ens, tmp_path):
    path = str(tmp_path / "ens.npz")
    ens.save(path)
    back = load_mlip_model(path)
    assert isinstance(back, AtomicELMEnsemble) and back.seeds == ens.seeds
    s = _random_structs(3, 5)[2]
    a = MLIPForceField(ens)(s["lattice"], s["positions"], s["atom_types"])
    b = MLIPForceField(back)(s["lattice"], s["positions"], s["atom_types"])
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@pytest.mark.slow
def test_md_with_an_ensemble_restarts_bit_for_bit(ens, tmp_path):
    from test_md_restart import _Killed, _kill_after

    from akitsushima.solvers.molecular_dynamics import NVTSimulation

    s = _random_structs(3, 5)[0]
    ff = MLIPForceField(ens)

    def make(out):
        return NVTSimulation(
            s["lattice"], s["positions"], s["atom_types"], force_fn=ff,
            timestep=0.5, n_steps=20, temperature=300.0, output_dir=str(out),
            thermo_every=2, traj_every=2, verbose=False, save_plots=False,
        )

    ref = make(tmp_path / "ref").run()
    out = tmp_path / "ck"
    ck = str(out / "md.npz")
    sim = make(out)
    _kill_after(sim, 14)
    with pytest.raises(_Killed):
        sim.run(checkpoint=ck, checkpoint_every=5)
    got = make(out).run(checkpoint=ck, checkpoint_every=5)
    np.testing.assert_array_equal(ref["positions"], got["positions"])
    np.testing.assert_array_equal(ref["velocities"], got["velocities"])
    a = (tmp_path / "ref" / "thermo.log").read_text().splitlines()
    b = (out / "thermo.log").read_text().splitlines()
    assert "sigF_max(eV/A)" in a[0]
    # NL_builds (column 8 of 9) counts the builds in each process

    def drop_nl(rows):
        return [r.split()[:7] + r.split()[8:] for r in rows]

    assert drop_nl(a) == drop_nl(b)
    assert all(float(r.split()[8]) > 0 for r in a[1:])  # sigF_max


def test_anneal_ledger_and_relax_log_carry_the_spread(ens, tmp_path):
    import json

    from akitsushima.solvers.global_opt import GlobalOptimiser

    s = _random_structs(3, 5)[0]
    ff = MLIPForceField(ens)
    GlobalOptimiser(
        s["lattice"], s["positions"], s["atom_types"], ff, ff.r_cut,
        method="basin_hopping", output_dir=str(tmp_path),
        relax={"method": "fire", "fmax": 0.05, "max_steps": 30},
        n_trials=2, temperature=300.0, step=0.1, seed=1, verbose=False,
        plot=False,
    ).run()
    recs = [json.loads(x) for x in open(tmp_path / "anneal_log.jsonl")]
    ok = [r for r in recs if r.get("status") == "ok"]
    assert ok and all(r["sigma_F_max"] > 0 and r["sigma_E"] >= 0 for r in ok)
    logs = list(tmp_path.rglob("relax.log"))
    assert logs and "sigF_max(eV/A)" in logs[0].read_text().splitlines()[0]


def test_mlip_cli_fits_an_ensemble_and_evaluates_its_spread(tmp_path):
    from test_cli_smoke import _ns, _write_sw_si_training_set, _write_yaml

    from akitsushima.cli.cmd_mlip import cmd_mlip

    csv = _write_sw_si_training_set(tmp_path / "ts")
    out = tmp_path / "out"
    cfg = {
        "structure": {"training_csv": str(csv)},
        "potential": {"species": ["Si"]},
        "mlip": {"output_dir": str(out), "form": "atom_centred", "model": "elm",
                 "n_hidden": 32, "val_fraction": 0.0, "ensemble": 3,
                 "ensemble_seeds": [42, 1, 2], "verbose": False,
                 "descriptor": {"type": "acsf", "r_cut": 3.77,
                                "g2": {"eta": [4.0], "rs": [2.3, 2.6, 3.0]}}},
    }
    _write_yaml(tmp_path / "fit.yaml", cfg)
    assert cmd_mlip(_ns(tmp_path / "fit.yaml")) == 0
    model = out / "atomic_elm_model.npz"
    back = load_mlip_model(str(model))
    assert isinstance(back, AtomicELMEnsemble) and back.seeds == [42, 1, 2]

    ev = {"mlip": {"evaluate": {
        "model": str(model), "poscars": str(tmp_path / "ts" / "*" / "POSCAR"),
        "csv": str(csv), "output": str(tmp_path / "unc.csv")}}}
    _write_yaml(tmp_path / "ev.yaml", ev)
    assert cmd_mlip(_ns(tmp_path / "ev.yaml")) == 0
    lines = (tmp_path / "unc.csv").read_text().splitlines()
    head = lines[0].split(",")
    assert head[:6] == ["name", "n_atoms", "E_per_atom", "sigma_E",
                        "sigma_F_max", "sigma_S"]
    assert len(lines) == 1 + 3 + 3  # three POSCARs, three labelled structures
    labelled = [ln.split(",") for ln in lines[4:]]
    assert all(float(r[4]) > 0 and r[6] != "" for r in labelled)
