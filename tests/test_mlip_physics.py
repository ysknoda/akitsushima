"""Physics-informed MLIP parts (ADR-0014): the pair baseline
(Delta-learning, ZBL / Born-Mayer with a C2 switch) and the isolated-atom
limit.

- Baseline: ZBL values against the formula; forces and stress are
  finite-difference derivatives of its energy; the switch is C1 at
  r_inner / r_outer and zero beyond; subtract() removes exactly compute().
- Engine: a model trained on baseline-subtracted labels, with the
  baseline in its physics_config, returns learned part + baseline; its
  forces / stress are derivatives of that total; site energies add up;
  save / load keeps the baseline (tabular and atom-centred).
- Isolated atom: eps_s(G = 0) = E_atom,s exactly for atom-centred ELM,
  QELM and NN, f(0) = E_atom for the one-species tabular ELM; an atom
  far from the others has that energy and no force; refusals.
- CLI: mlip.baseline + mlip.isolated_atom_energy, then the saved model
  drives the engine with the baseline.
"""

import importlib.util
import json

import numpy as np
import pytest
from test_mlip_atom_centred import _fd4_check, _fd_check, _random_structs
from test_mlip_chain_rule import _acsf, _sw_si_structures, _write_layout_b

from akitsushima.constants import K_E
from akitsushima.fitting.descriptors import ACSF, build_table
from akitsushima.fitting.mlip_engine import MLIPForceField, load_mlip_model
from akitsushima.fitting.models import ELMModel, QuantizedELMModel
from akitsushima.fitting.models.atomic import AtomicELMModel, AtomicQELMModel
from akitsushima.fitting.physics import PairBaseline

HAS_TORCH = importlib.util.find_spec("torch") is not None
ZBL = {"type": "zbl", "r_inner": 1.0, "r_outer": 2.4}
BM = {
    "type": "born_mayer",
    "r_inner": 1.5,
    "r_outer": 2.6,
    "pairs": {"Sn-Ge": {"A": 800.0, "rho": 0.3}, "Si-Si": {"A": 500.0, "rho": 0.25}},
}


def _si_acsf():
    return ACSF.from_grid(
        ["Si"],
        3.771,
        g2={"eta": [4.0], "rs": [2.2, 2.6, 3.0]},
        g5={"eta": [0.05], "zeta": [1, 2], "lambda": [-1, 1]},
    )


def _close_cell(seed=3):
    rng = np.random.default_rng(seed)
    lat = np.array([[4.1, 0.3, -0.2], [0.4, 3.9, 0.5], [-0.3, 0.2, 4.3]])
    return lat, rng.random((6, 3)) @ lat, ["Sn", "Ge", "Ge", "Si", "Ge", "Si"]


# ---------------------------------------------------------------- baseline


def test_zbl_matches_its_formula_and_the_switch_is_smooth():
    b = PairBaseline(["Si"], ZBL)
    z = np.zeros(1, dtype=int)
    for r in (0.5, 1.0):  # below r_inner: plain ZBL
        a = 0.46850 / (2 * 14**0.23)
        x = r / a
        ref = (
            K_E
            * 196
            / r
            * (
                0.18175 * np.exp(-3.19980 * x)
                + 0.50986 * np.exp(-0.94229 * x)
                + 0.28022 * np.exp(-0.40290 * x)
                + 0.02817 * np.exp(-0.20162 * x)
            )
        )
        assert b.phi(np.array([r]), z, z)[0][0] == pytest.approx(ref, rel=1e-12)
    e, de = b.phi(np.array([2.4, 2.5, 3.0]), np.zeros(3, int), np.zeros(3, int))
    assert np.all(e == 0) and np.all(de == 0)
    for r0 in (1.0, 2.4):  # value and slope continuous at both ends
        h = 1e-7
        e, de = b.phi(np.array([r0 - h, r0 + h]), np.zeros(2, int), np.zeros(2, int))
        # a jump would exceed the slope's own change over 2h
        assert abs(e[1] - e[0]) <= 2 * h * abs(de[0]) * 1.01 + 1e-12
        assert abs(de[1] - de[0]) < 1e-4 * max(1.0, abs(de[0]))
    # dE/dr equals the finite difference of the energy inside the switch
    r = np.array([1.7])
    h = 1e-6
    fd = (b.phi(r + h, z, z)[0] - b.phi(r - h, z, z)[0]) / (2 * h)
    assert b.phi(r, z, z)[1][0] == pytest.approx(fd[0], rel=1e-7)


def test_baseline_forces_and_stress_are_derivatives_of_its_energy():
    b = PairBaseline(["Sn", "Ge", "Si"], [ZBL, BM])
    lat, pos, types = _close_cell()
    E, F, S = b.compute(lat, pos, types)
    assert E > 1.0 and np.abs(F).max() > 1.0  # the cell has close pairs
    _fd_check(b.compute, lat, pos, types, 1e-6, 1e-6 * np.abs(F).max(), 1e-7)
    s = dict(
        lattice=lat,
        positions=pos,
        atom_types=types,
        ref_energy=1.0,
        ref_forces=np.ones((6, 3)),
        ref_stress=np.eye(3),
    )
    (t,) = b.subtract([s])
    assert t["ref_energy"] == pytest.approx(1.0 - E / 6)
    np.testing.assert_allclose(t["ref_forces"], 1.0 - F)
    np.testing.assert_allclose(t["ref_stress"], np.eye(3) - S)


def test_engine_adds_the_baseline_atom_centred_and_tabular(tmp_path):
    structs = _random_structs(10, 0)
    base = PairBaseline(["Ge", "Sn", "Si"], [ZBL, BM])
    phys = json.dumps({"baseline": base.config()})
    m = AtomicELMModel(n_hidden=16, r0=0.1, eps0=0.01, verbose=False)
    m.physics_config = phys
    m.fit(base.subtract(structs), _acsf())
    path = str(tmp_path / "a.npz")
    m.save(path)
    m2 = load_mlip_model(path)
    assert m2.physics_config == phys
    ff = MLIPForceField(m2)
    lat, pos, types = _close_cell()
    J = _acsf().atomic_jacobian(lat, pos, types)
    E_ml = m2.atomic_energy_and_gradient(J.G, J.sp)[0].sum()
    E, F, S = ff(lat, pos, types)
    assert E == pytest.approx(E_ml + base.compute(lat, pos, types)[0], rel=1e-12)
    st = ff.site_terms(lat, pos, types)
    assert st.energy == pytest.approx(E, rel=1e-12)
    np.testing.assert_allclose(st.forces(), F, atol=1e-9 * np.abs(F).max())
    np.testing.assert_allclose(st.stress(abs(np.linalg.det(lat))), S, atol=1e-9)
    _fd4_check(ff, lat, pos, types, 3e-5, tol_f=1e-6 * np.abs(F).max(), tol_s=1e-8 * np.abs(S).max())

    si = _sw_si_structures(12, 0)
    sb = PairBaseline(["Si"], {"type": "zbl", "r_inner": 1.0, "r_outer": 2.0})
    y, lab, w, X = build_table(sb.subtract(si), _si_acsf(), r0=0.1, eps0=0.01)
    t = ELMModel(n_hidden=20, verbose=False)
    t.descriptor_config = _si_acsf().to_json()
    t.physics_config = json.dumps({"baseline": sb.config()})
    t.fit(y, lab, X, weights=w)
    p = str(tmp_path / "t.npz")
    t.save(p)
    ft = MLIPForceField(load_mlip_model(p))
    s = si[0]
    pos = s["positions"].copy()
    pos[1] = pos[0] + [1.2, 0.3, 0.2]  # a 1.25 A contact
    E, F, S = ft(s["lattice"], pos, s["atom_types"])
    Eb, Fb, _ = sb.compute(s["lattice"], pos, s["atom_types"])
    assert Eb > 1.0
    d, *_ = _si_acsf().structure_table(s["lattice"], pos, s["atom_types"])
    assert E == pytest.approx(8 * t.energy_and_gradient(d)[0] + Eb, rel=1e-12)
    _fd_check(
        ft, s["lattice"], pos, s["atom_types"], 1e-6, 1e-5 * np.abs(F).max(), 1e-7
    )


def test_baseline_beyond_the_descriptor_cutoff_is_refused():
    m = AtomicELMModel(n_hidden=8, verbose=False).fit(_random_structs(4, 0), _acsf())
    m.physics_config = json.dumps(
        {
            "baseline": PairBaseline(
                ["Ge", "Sn", "Si"], {"type": "zbl", "r_inner": 1, "r_outer": 5.0}
            ).config()
        }
    )
    with pytest.raises(ValueError, match="exceeds the descriptor cutoff"):
        MLIPForceField(m)


# ------------------------------------------------------------ isolated atom

ISO = {"Ge": -1.5, "Sn": -2.25, "Si": -3.0}


def _isolated(species, box=30.0):
    """One atom per species, far apart (no neighbour within r_cut)."""
    lat = np.eye(3) * box
    pos = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], float)[: len(species)]
    return lat, pos, list(species)


@pytest.mark.parametrize("cls", [AtomicELMModel, AtomicQELMModel])
def test_atomic_elm_obeys_the_isolated_atom_limit_exactly(cls, tmp_path):
    structs = _random_structs(12, 0)
    m = cls(n_hidden=24, r0=0.1, eps0=0.01, reg_search=[1e-6, 1e-3], verbose=False)
    m.physics_config = json.dumps({"isolated_atom_energy": ISO})
    m.fit(structs, _acsf())
    eps, _ = m.atomic_energy_and_gradient(
        np.zeros((3, _acsf().n_per_atom)), np.arange(3)
    )
    np.testing.assert_allclose(eps, [ISO[s] for s in m.species], atol=1e-9)
    path = str(tmp_path / "m.npz")
    m.save(path)
    E, F, S = MLIPForceField(load_mlip_model(path))(*_isolated(["Ge", "Sn", "Si"]))
    assert E == pytest.approx(sum(ISO.values()), abs=1e-9)
    assert np.abs(F).max() < 1e-12 and np.abs(S).max() < 1e-12
    # the unconstrained fit does not satisfy it: the test discriminates
    u = cls(n_hidden=24, r0=0.1, eps0=0.01, verbose=False).fit(structs, _acsf())
    e_u = u.atomic_energy_and_gradient(np.zeros((3, _acsf().n_per_atom)), np.arange(3))[
        0
    ]
    assert np.abs(e_u - [ISO[s] for s in u.species]).max() > 1e-3


def test_tabular_elm_isolated_atom_limit_one_species_only():
    si = _sw_si_structures(12, 0)
    y, lab, w, X = build_table(si, _si_acsf(), r0=0.1, eps0=0.01)
    for cls in (ELMModel, QuantizedELMModel):
        m = cls(n_hidden=20, verbose=False)
        m.descriptor_config = _si_acsf().to_json()
        m.physics_config = json.dumps({"isolated_atom_energy": {"Si": -0.7}})
        m.fit(y, lab, X, weights=w)
        assert m.energy_and_gradient(np.zeros(X.shape[1]))[0] == pytest.approx(
            -0.7, abs=1e-9
        )
    y3, lab3, w3, X3 = build_table(_random_structs(6, 0), _acsf(), r0=0.1, eps0=0.01)
    m = ELMModel(n_hidden=8, verbose=False)
    m.descriptor_config = _acsf().to_json()
    m.physics_config = json.dumps({"isolated_atom_energy": ISO})
    with pytest.raises(ValueError, match="one species"):
        m.fit(y3, lab3, X3, weights=w3)
    m = ELMModel(n_hidden=8, solver="lsqr", verbose=False)
    m.descriptor_config = _si_acsf().to_json()
    m.physics_config = json.dumps({"isolated_atom_energy": {"Si": 0.0}})
    with pytest.raises(ValueError, match="direct solver"):
        m.fit(y, lab, X, weights=w)
    a = AtomicELMModel(n_hidden=8, verbose=False)
    a.physics_config = json.dumps({"isolated_atom_energy": {"Si": 0.0}})
    with pytest.raises(ValueError, match="no entry"):
        a.fit(_random_structs(4, 0), _acsf())


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_atomic_nn_obeys_the_isolated_atom_limit(tmp_path):
    import torch

    from akitsushima.fitting.models.atomic_nn import AtomicNNModel
    from akitsushima.fitting.models.nn import NNModel

    m = AtomicNNModel(
        hidden_layers=[8],
        epochs=20,
        r0=0.1,
        eps0=0.01,
        deterministic=True,
        verbose=False,
    )
    m.physics_config = json.dumps({"isolated_atom_energy": ISO})
    m.fit(_random_structs(8, 0), _acsf())
    assert m._iso_t is None  # folded into the output bias
    path = str(tmp_path / "n.pt")
    m.save(path)
    E, F, _ = MLIPForceField(load_mlip_model(path))(*_isolated(["Ge", "Sn", "Si"]))
    assert E == pytest.approx(sum(ISO.values()), abs=1e-5)  # float32 network
    assert np.abs(F).max() < 1e-6
    del torch
    n = NNModel(hidden_layers=[4], epochs=1, verbose=False)
    n.physics_config = json.dumps({"isolated_atom_energy": {"Si": 0.0}})
    with pytest.raises(ValueError, match="atom-centred nn"):
        n.fit(np.zeros(1), np.array(["E"]), np.zeros((1, 2)))


# ---------------------------------------------------------------- CLI


def test_cli_mlip_with_baseline_and_isolated_atom(tmp_path, monkeypatch):
    import yaml
    from test_cli_smoke import _ns

    from akitsushima.cli.parser import _DISPATCH

    monkeypatch.chdir(tmp_path)
    _write_layout_b(tmp_path / "train", _sw_si_structures(12, 1))
    desc = {
        "type": "acsf",
        "r_cut": 3.771,
        "g2": {"eta": [4.0], "rs": [2.2, 2.6, 3.0]},
        "g5": {"eta": [0.05], "zeta": [1, 2], "lambda": [-1, 1]},
    }
    for form, fname in (
        ("atom_centred", "atomic_qelm_model.npz"),
        ("tabular", "qelm_model.npz"),
    ):
        ml = {
            "output_dir": f"out_{form}",
            "form": form,
            "model": "qelm",
            "descriptor": desc,
            "r0": 0.1,
            "eps0": 0.01,
            "n_hidden": 16,
            "baseline": {"type": "zbl", "r_inner": 1.0, "r_outer": 2.0},
            "isolated_atom_energy": {"Si": 0.0},
        }
        (tmp_path / "mlip.yaml").write_text(
            yaml.safe_dump(
                {
                    "structure": {"training_csv": "train/structures.csv"},
                    "potential": {"species": ["Si"]},
                    "mlip": ml,
                }
            )
        )
        assert _DISPATCH["mlip"](_ns(tmp_path / "mlip.yaml")) == 0
        m = load_mlip_model(str(tmp_path / f"out_{form}" / fname))
        phys = json.loads(m.physics_config)
        assert phys["baseline"]["terms"][0]["type"] == "zbl"
        ff = MLIPForceField(m)
        assert ff.baseline is not None
        E, F, _ = ff(*_isolated(["Si"]))
        assert E == pytest.approx(0.0, abs=1e-8)


# ------------------------------------------------------ ACSF inner cutoff


def _inner_acsf(inner=(2.0, 2.8)):
    return ACSF.from_grid(
        ["Ge", "Sn", "Si"],
        4.2,
        g2={"eta": [0.5, 2.0], "rs": [0.0, 2.5]},
        g4={"eta": [0.02, 0.1], "zeta": [1, 2, 4], "lambda": [-1, 1]},
        g5={"eta": [0.05], "zeta": [1, 4], "lambda": [-1, 1]},
        inner_cutoff=inner,
    )


def test_acsf_inner_cutoff_derivatives_and_invisibility():
    from test_mlip_chain_rule import _distorted_cell

    lat, frac, types = _distorted_cell()  # pairs at 0.99 ... 2.12 ... A
    ac = _inner_acsf()
    c = np.random.default_rng(1).normal(size=(len(types), ac.n_per_atom))

    def ff(lat_, pos_, types_):
        J = ac.atomic_jacobian(lat_, pos_, types_)
        F, S = J.forces_and_stress(c)
        return float(np.sum(c * J.G)), F, S

    _fd_check(ff, lat, frac @ lat, types, 1e-6, 1e-7, 1e-8)
    # a dimer closer than r_off is invisible; inside the switch it is not
    lat = np.eye(3) * 20.0
    for r, zero in ((1.9, True), (2.4, False)):
        G = ac.atomic(lat, np.array([[0, 0, 0], [r, 0, 0.0]]), ["Si", "Si"])
        assert (np.abs(G).max() == 0.0) == zero
    # the switch is C1 at r_off and r_on (G2 of a dimer)
    for r0 in (2.0, 2.8):
        g = [ac.atomic(lat, np.array([[0, 0, 0], [r, 0, 0.0]]), ["Si", "Si"])[0]
             for r in (r0 - 1e-6, r0, r0 + 1e-6)]
        assert np.abs(g[2] - g[0]).max() < 1e-5
    # config round trip; a config without the key has none
    assert ACSF.from_config(ac.config()).inner_cutoff == [2.0, 2.8]
    cfg = ac.config()
    cfg.pop("inner_cutoff")
    assert ACSF.from_config(cfg).inner_cutoff is None
    with pytest.raises(ValueError, match="inner_cutoff"):
        _inner_acsf((2.8, 2.0))


def test_inner_cutoff_with_baseline_engine_is_conservative(tmp_path):
    """Atom-centred ELM on an inner-cutoff descriptor plus a baseline that
    covers the blind range: engine forces / stress are derivatives of
    the total energy across the switch."""
    structs = _random_structs(10, 0)
    ac = _inner_acsf((1.4, 2.0))
    base = PairBaseline(["Ge", "Sn", "Si"], {"type": "zbl", "r_inner": 1.0, "r_outer": 2.0})
    m = AtomicELMModel(n_hidden=16, r0=0.1, eps0=0.01, verbose=False)
    m.physics_config = json.dumps({"baseline": base.config()})
    m.fit(base.subtract(structs), ac)
    path = str(tmp_path / "m.npz")
    m.save(path)
    ff = MLIPForceField(load_mlip_model(path))
    assert ff.descriptor.inner_cutoff == [1.4, 2.0]
    lat, pos, types = _close_cell()
    E, F, S = ff(lat, pos, types)
    _fd4_check(ff, lat, pos, types, 3e-5, tol_f=1e-6 * np.abs(F).max(), tol_s=1e-8 * np.abs(S).max())


# ------------------------------------------- fitted Born-Mayer (follow-up)


def test_fit_born_mayer_recovers_known_pair_terms():
    from akitsushima.fitting.physics import fit_born_mayer

    rng = np.random.default_rng(0)
    true = PairBaseline(["Ge", "Si"], {
        "type": "born_mayer", "r_inner": 1.8, "r_outer": 2.4,
        "pairs": {"Si-Si": {"A": 2000.0, "rho": 0.25}, "Ge-Si": {"A": 4000.0, "rho": 0.22}}})
    sr = []
    for _ in range(30):
        lat, pos = np.eye(3) * 8.0, rng.random((6, 3)) * 8.0
        t = ["Si", "Ge", "Si", "Si", "Ge", "Si"]
        e = true.compute(lat, pos, t)[0] + 1.0  # + the "model" energy
        sr.append(dict(lattice=lat, positions=pos, atom_types=t, ref_energy=e / 6))
    term, info = fit_born_mayer(sr, lambda *a: 1.0, ["Si", "Ge"], 1.8, 2.4)
    assert term["pairs"]["Si-Si"]["A"] == pytest.approx(2000.0, rel=1e-6)
    assert term["pairs"]["Si-Si"]["rho"] == pytest.approx(0.25, rel=1e-6)
    assert term["pairs"]["Si-Ge"]["A"] == pytest.approx(4000.0, rel=1e-6)
    assert term["pairs"]["Si-Ge"]["rho"] == pytest.approx(0.22, rel=1e-6)
    assert info["rms_after_eV"] < 1e-6 < info["rms_before_eV"]


def _sw_pushed(structs, rs):
    """8-atom SW Si cells with atom 0 pushed to r from its nearest
    neighbour, labelled by SW on the 2x2x2 replica (as _sw_si_structures)."""
    import itertools

    from akitsushima import PotentialEngine, StillingerWeberPotential

    sw = {"epsilon": 2.1683, "sigma": 2.0951, "a": 1.80, "gamma": 1.20,
          "lambda": 21.0, "A": 7.049556277, "B": 0.6022245584, "p": 4.0,
          "q": 0.0, "costheta0": -1.0 / 3.0}
    eng = PotentialEngine().add(
        StillingerWeberPotential({("Si", "Si", "Si"): sw}, species=["Si"]))
    out = []
    for s in structs:
        lat, pos = s["lattice"], s["positions"]
        d = pos - pos[0]
        d -= np.round(d @ np.linalg.inv(lat)) @ lat
        dist = np.linalg.norm(d, axis=1)
        dist[0] = np.inf
        j = int(np.argmin(dist))
        for r in rs:
            p = pos.copy()
            p[0] = pos[0] + d[j] - d[j] / dist[j] * r
            shifts = np.array(list(itertools.product(range(2), repeat=3))) @ lat
            big = (p[None] + shifts[:, None]).reshape(-1, 3)
            E, F, S = eng.compute(2 * lat, big, ["Si"] * 64)
            out.append(dict(lattice=lat, positions=p, atom_types=["Si"] * 8,
                            ref_energy=E / 64, ref_forces=F[:8], ref_stress=S))
    return out


@pytest.mark.parametrize("form,r_outer", [("atom_centred", 2.0), ("tabular", 2.3)])
def test_cli_fits_born_mayer_to_short_range_data(form, r_outer, tmp_path, monkeypatch):
    import yaml
    from test_cli_smoke import _ns

    from akitsushima.cli.parser import _DISPATCH

    monkeypatch.chdir(tmp_path)
    train = _sw_si_structures(16, 1)
    _write_layout_b(tmp_path / "train", train)
    _write_layout_b(tmp_path / "sr", _sw_pushed(train[:6], (1.3, 1.5, 1.7, 1.9)))
    desc = {"type": "acsf", "r_cut": 3.771, "inner_cutoff": [1.8, 2.0],
            "g2": {"eta": [4.0], "rs": [2.2, 2.6, 3.0]},
            "g5": {"eta": [0.05], "zeta": [1, 2], "lambda": [-1, 1]}}
    ml = {"output_dir": "out", "form": form, "model": "elm", "descriptor": desc,
          "r0": 0.1, "eps0": 0.01, "n_hidden": 16,
          "isolated_atom_energy": {"Si": 0.0},
          "baseline": [{"type": "born_mayer", "r_inner": 1.8, "r_outer": r_outer,
                        "fit_to": "sr/structures.csv"}]}
    (tmp_path / "mlip.yaml").write_text(yaml.safe_dump(
        {"structure": {"training_csv": "train/structures.csv"},
         "potential": {"species": ["Si"]}, "mlip": ml}))
    assert _DISPATCH["mlip"](_ns(tmp_path / "mlip.yaml")) == 0
    summ = json.loads((tmp_path / "out" / "mlip_summary.json").read_text())
    fit = summ["baseline_fit"]
    assert fit["rms_after_eV"] < 0.5 * fit["rms_before_eV"]
    # the rattled training cells have pairs below 2.3 A, none below 2.0 A
    assert fit["refit"] is (r_outer > 2.2)
    name = ("atomic_" if form == "atom_centred" else "") + "elm_model.npz"
    m = load_mlip_model(str(tmp_path / "out" / name))
    term = json.loads(m.physics_config)["baseline"]["terms"][0]
    assert term["type"] == "born_mayer" and set(term["pairs"]) == {"Si-Si"}
    # the engine is conservative with the fitted term in place
    ff = MLIPForceField(m)
    (s,) = _sw_pushed(train[6:7], (1.6,))
    E, F, S = ff(s["lattice"], s["positions"], s["atom_types"])
    _fd4_check(ff, s["lattice"], s["positions"], s["atom_types"], 3e-5,
               tol_f=1e-6 * np.abs(F).max(), tol_s=1e-8 * np.abs(S).max())
