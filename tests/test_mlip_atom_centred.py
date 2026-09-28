"""ADR-0005: atom-centred (Behler-Parrinello) models E = sum_i eps_s(G_i).

Every derivative is checked against finite differences (ADR-0002); the
training rows of each model are checked against its own engine, and the
whole training chain against a teacher it must recover exactly.  The NN
tests come last and import torch only when they run (see
test_mlip_chain_rule.py on the numba/torch OpenMP conflict).
"""

import importlib.util
import json

import numpy as np
import pytest
from test_mlip_chain_rule import (
    _acsf,
    _distorted_cell,
    _sw_si_structures,
    _write_layout_b,
)

from akitsushima.fitting import make_force_fn
from akitsushima.fitting.descriptors import ACSF
from akitsushima.fitting.mlip_engine import MLIPForceField, load_mlip_model
from akitsushima.fitting.models.atomic import (
    AtomicData,
    AtomicELMModel,
    AtomicQELMModel,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _si_acsf():
    return ACSF.from_grid(
        ["Si"],
        3.771,
        g2={"eta": [4.0], "rs": [2.2, 2.6, 3.0]},
        g5={"eta": [0.05], "zeta": [1, 2], "lambda": [-1, 1]},
    )


def _random_structs(n, seed):
    """Three-species cells (~5 A, thinner than 2 r_c) with arbitrary
    labels: the derivative tests do not depend on what was learned."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        lat, frac, types = _distorted_cell(seed + k)
        N = len(types)
        out.append(
            dict(
                lattice=lat,
                positions=frac @ lat,
                atom_types=types,
                ref_energy=rng.normal() - 4.0,
                ref_forces=rng.normal(size=(N, 3)),
                ref_stress=0.01 * rng.normal(size=(3, 3)),
            )
        )
    return out


def _fd_check(ff, lat, pos, types, h, tol_f, tol_s):
    E0, F0, S0 = ff(lat, pos, types)
    for j in range(len(types)):
        for a in range(3):
            p = pos.copy()
            p[j, a] += h
            Ep = ff(lat, p, types)[0]
            p[j, a] -= 2 * h
            Em = ff(lat, p, types)[0]
            assert -(Ep - Em) / (2 * h) == pytest.approx(F0[j, a], abs=tol_f), (j, a)
    V = abs(np.linalg.det(lat))
    frac = pos @ np.linalg.inv(lat)
    for a in range(3):
        for b in range(3):
            T = np.eye(3)
            T[a, b] += h
            Ep = ff(lat @ T, frac @ lat @ T, types)[0]
            T = np.eye(3)
            T[a, b] -= h
            Em = ff(lat @ T, frac @ lat @ T, types)[0]
            assert -(Ep - Em) / (2 * h) / V == pytest.approx(S0[a, b], abs=tol_s), (
                a,
                b,
            )
    np.testing.assert_allclose(F0.sum(axis=0), 0.0, atol=1e-9)


def _fd4_check(ff, lat, pos, types, h, tol_f=1e-7, tol_s=1e-8):
    """Forces and stress against the fourth-order central difference
    (8 [f(h) - f(-h)] - [f(2h) - f(-2h)]) / 12h."""
    E0, F0, S0 = ff(lat, pos, types)

    def d4(fun):
        return (8 * (fun(h) - fun(-h)) - (fun(2 * h) - fun(-2 * h))) / (12 * h)

    for j in (0, 3):
        for a in range(3):

            def e_pos(x, j=j, a=a):
                p = pos.copy()
                p[j, a] += x
                return ff(lat, p, types)[0]

            assert -d4(e_pos) == pytest.approx(F0[j, a], abs=tol_f), (j, a)
    V = abs(np.linalg.det(lat))
    frac = pos @ np.linalg.inv(lat)
    for a, b in [(0, 0), (1, 2), (2, 1)]:

        def e_cell(x, a=a, b=b):
            T = np.eye(3)
            T[a, b] += x
            return ff(lat @ T, frac @ lat @ T, types)[0]

        assert -d4(e_cell) / V == pytest.approx(S0[a, b], abs=tol_s), (a, b)
    np.testing.assert_allclose(F0.sum(axis=0), 0.0, atol=1e-9)


# ---------------------------------------------------------------- ACSF


def test_atomic_jacobian_contraction_matches_finite_differences():
    """E = sum_i c_i . G_i with random c: forces and stress from the
    per-entry Jacobians against finite differences, on a distorted
    three-species cell thinner than 2 r_c (several images per pair)."""
    lat, frac, types = _distorted_cell()
    ac = _acsf()
    c = np.random.default_rng(1).normal(size=(len(types), ac.n_per_atom))

    def ff(lat_, pos_, types_):
        J = ac.atomic_jacobian(lat_, pos_, types_)
        F, S = J.forces_and_stress(c)
        return float(np.sum(c * J.G)), F, S

    _fd_check(ff, lat, frac @ lat, types, 1e-5, 1e-7, 1e-8)


def test_structure_table_is_the_per_atom_average_of_the_atomic_jacobian():
    lat, frac, types = _distorted_cell()
    ac = _acsf()
    d, XF, XS = ac.structure_table(lat, frac @ lat, types)
    J = ac.atomic_jacobian(lat, frac @ lat, types)
    N, Dc = J.G.shape
    np.testing.assert_allclose(d, ac._blocks(J.G, J.sp), atol=1e-14)
    # A linear energy E = N w . d is atom-centred: both routes agree.
    w = np.random.default_rng(2).normal(size=len(d))
    g = w.reshape(len(ac.species), Dc)[J.sp]
    F, S = J.forces_and_stress(g)
    np.testing.assert_allclose(N * XF @ w, F, atol=1e-12)
    np.testing.assert_allclose(N * 0.5 * (XS @ w + (XS @ w).T), S, atol=1e-13)


# ---------------------------------------------------------------- ELM / QELM


@pytest.mark.parametrize(
    "cls,kw",
    [
        (AtomicELMModel, {"activation": "tanh"}),
        (AtomicELMModel, {"activation": "rff", "direct_link": False}),
        (AtomicELMModel, {"activation": "softplus"}),
        (AtomicQELMModel, {"activation": "tanh"}),
    ],
)
def test_atomic_elm_rows_equal_its_engine_and_engine_is_conservative(cls, kw, tmp_path):
    structs = _random_structs(12, 0)
    m = cls(n_hidden=24, r0=0.1, eps0=0.01, reg=1e-6, verbose=False, **kw)
    m.fit(structs, _acsf())
    path = str(tmp_path / "m.npz")
    m.save(path)
    m2 = load_mlip_model(path)
    assert type(m2) is cls and m2.atom_centred
    ff = MLIPForceField(m2)
    s = structs[1]
    J = _acsf().atomic_jacobian(s["lattice"], s["positions"], s["atom_types"])
    coef = np.concatenate([u._coef() for u in m.units])
    np.testing.assert_allclose(
        m._design(J) @ coef, m2._predict_structure(J), atol=1e-10
    )
    # default PCA whitening keeps near-null directions (gain up to
    # 1/sqrt(pca_eps)): strongly curved energy, fourth-order difference
    _fd4_check(ff, s["lattice"], s["positions"], s["atom_types"], 3e-5)


def test_atomic_elm_recovers_a_teacher_of_its_own_form_exactly():
    """Labels from an atom-centred ELM with known output weights; a
    student with the same random layer (same seed) must recover them from
    E, F and S rows together -- the whole chain: rows, weights, solve."""
    ac = _acsf()
    train = _random_structs(30, 10)
    teacher = AtomicELMModel(n_hidden=20, reg=1e-6, verbose=False).fit(train, ac)
    rng = np.random.default_rng(3)
    for u in teacher.units:
        u._beta = rng.normal(size=u._beta.shape)
        u._gamma = 0.1 * rng.normal(size=u._gamma.shape)
    ff = MLIPForceField(teacher)
    for s in train:
        E, F, S = ff(s["lattice"], s["positions"], s["atom_types"])
        s["ref_energy"], s["ref_forces"], s["ref_stress"] = E / len(F), F, S
    st = AtomicELMModel(
        n_hidden=20, reg=1e-12, verbose=False, w_forces=1.0, w_stress=1.0
    ).fit(train, ac)
    test = _random_structs(3, 99)
    np.testing.assert_allclose(
        st.predict_structures(test, ac),
        teacher.predict_structures(test, ac),
        rtol=1e-6,
        atol=1e-6,
    )


def test_atomic_model_is_local_supercell_energy_is_extensive():
    """Locality: a 2x2x2 supercell has 8x the energy, the same forces
    per atom and the same stress."""
    structs = _sw_si_structures(20, 0)
    m = AtomicELMModel(n_hidden=20, r0=0.1, eps0=0.01, verbose=False)
    m.fit(structs, _si_acsf())
    ff = MLIPForceField(m)
    s = structs[0]
    E, F, S = ff(s["lattice"], s["positions"], s["atom_types"])
    sh = (
        np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
        @ s["lattice"]
    )
    pos = (s["positions"][None] + sh[:, None]).reshape(-1, 3)
    E8, F8, S8 = ff(2 * s["lattice"], pos, ["Si"] * len(pos))
    assert E8 == pytest.approx(8 * E, rel=1e-12)
    np.testing.assert_allclose(F8, np.tile(F, (8, 1)), atol=1e-10)
    np.testing.assert_allclose(S8, S, atol=1e-12)


def test_atomic_engine_is_invariant_under_rotation_translation_permutation():
    structs = _random_structs(10, 5)
    m = AtomicELMModel(n_hidden=16, verbose=False).fit(structs, _acsf())
    ff = MLIPForceField(m)
    lat, pos, types = (structs[0][k] for k in ("lattice", "positions", "atom_types"))
    E, F, S = ff(lat, pos, types)
    th = 0.9
    R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    Er, Fr, Sr = ff(lat @ R.T, pos @ R.T, types)
    assert Er == pytest.approx(E, rel=1e-12)
    np.testing.assert_allclose(Fr, F @ R.T, atol=1e-10)
    np.testing.assert_allclose(Sr, R @ S @ R.T, atol=1e-12)
    assert ff(lat, pos + 0.77, types)[0] == pytest.approx(E, rel=1e-12)
    perm = np.random.default_rng(0).permutation(len(types))
    Ep, Fp, _ = ff(lat, pos[perm], [types[k] for k in perm])
    assert Ep == pytest.approx(E, rel=1e-12)
    np.testing.assert_allclose(Fp, F[perm], atol=1e-10)


def test_make_force_fn_takes_a_model_and_refuses_descriptor_fn():
    structs = _sw_si_structures(10, 0)
    m = AtomicELMModel(n_hidden=10, verbose=False).fit(structs, _si_acsf())
    fn = make_force_fn(m)
    assert fn.kind == "mlip" and fn.r_cut == pytest.approx(3.771)
    with pytest.raises(ValueError, match="descriptor_fn"):
        make_force_fn(m, descriptor_fn=lambda *a: None)
    assert not hasattr(__import__("akitsushima").ELMModel, "compute")


def test_atomic_data_rows_follow_the_tabular_table():
    from akitsushima.fitting.descriptors import build_table

    structs = _random_structs(4, 0)
    ac = _acsf()
    y, lab, w, _ = build_table(structs, ac, r0=0.1, eps0=0.01)
    ya, laba, wa, sid, ptr = AtomicData(structs, ac).rows(0.1, 0.01)
    np.testing.assert_array_equal(lab, laba)
    np.testing.assert_allclose(y, ya)
    np.testing.assert_allclose(w, wa)
    assert ptr[-1] == len(y) and sid[-1] == 3


# ---------------------------------------------------------------- CLI


def test_cli_mlip_atom_centred_then_relax_and_md(tmp_path, monkeypatch):
    import yaml
    from test_cli_smoke import _ns

    from akitsushima.cli.parser import _DISPATCH
    from akitsushima.io.poscar import write_poscar

    monkeypatch.chdir(tmp_path)
    structs = _sw_si_structures(16, 1)
    _write_layout_b(tmp_path / "train", structs)
    _write_layout_b(tmp_path / "test", _sw_si_structures(4, 2))
    desc = {
        "type": "acsf",
        "r_cut": 3.771,
        "g2": {"eta": [4.0], "rs": [2.2, 2.6, 3.0]},
        "g5": {"eta": [0.05], "zeta": [1, 2], "lambda": [-1, 1]},
    }
    ml = {
        "output_dir": "out",
        "form": "atom_centred",
        "model": "elm",
        "descriptor": desc,
        "r0": 0.1,
        "eps0": 0.01,
        "n_hidden": 20,
        "test_csv": "test/structures.csv",
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
    summ = json.loads((tmp_path / "out" / "mlip_summary.json").read_text())
    assert summ["engine"] is True and summ["form"] == "atom_centred"
    assert set(summ["rmse_test"]) == {"E", "F", "S"}
    model = load_mlip_model(str(tmp_path / "out" / "atomic_elm_model.npz"))
    assert model.atom_centred

    s = structs[0]
    sh = (
        np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
        @ s["lattice"]
    )
    pos = (s["positions"][None] + sh[:, None]).reshape(-1, 3)
    write_poscar(str(tmp_path / "POSCAR"), 2 * s["lattice"], pos, ["Si"] * len(pos))
    pot = {"type": "MLIP", "model": "out/atomic_elm_model.npz"}
    (tmp_path / "relax.yaml").write_text(
        yaml.safe_dump(
            {
                "structure": {"poscar": "POSCAR"},
                "potential": pot,
                "relax": {
                    "output_dir": "relax_out",
                    "max_steps": 5,
                    "relax_cell": True,
                },
            }
        )
    )
    assert _DISPATCH["relax"](_ns(tmp_path / "relax.yaml")) == 0
    (tmp_path / "md.yaml").write_text(
        yaml.safe_dump(
            {
                "structure": {"poscar": "POSCAR"},
                "potential": pot,
                "md": {
                    "output_dir": "md_out",
                    "ensemble": "nve",
                    "n_steps": 5,
                    "temperature": 300.0,
                    "dt": 1.0,
                },
            }
        )
    )
    assert _DISPATCH["md"](_ns(tmp_path / "md.yaml")) == 0


def test_cli_atom_centred_refuses_models_without_forces(tmp_path, monkeypatch, capsys):
    import yaml
    from test_cli_smoke import _ns

    from akitsushima.cli.parser import _DISPATCH

    monkeypatch.chdir(tmp_path)
    _write_layout_b(tmp_path / "train", _sw_si_structures(4, 1))
    (tmp_path / "m.yaml").write_text(
        yaml.safe_dump(
            {
                "structure": {"training_csv": "train/structures.csv"},
                "mlip": {
                    "form": "atom_centred",
                    "model": "lgbm",
                    "descriptor": {
                        "type": "acsf",
                        "r_cut": 3.771,
                        "g2": {"eta": [4.0], "rs": [2.4]},
                    },
                },
            }
        )
    )
    # (lgbm itself was removed, ADR-0012; any non-force model is refused)
    assert _DISPATCH["mlip"](_ns(tmp_path / "m.yaml")) == 1
    assert "elm, qelm or nn" in capsys.readouterr().out


# ---------------------------------------------------------------- NN (last)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_atomic_nn_rows_equal_its_engine_and_engine_is_conservative(tmp_path):
    import torch

    from akitsushima.fitting.models.atomic_nn import AtomicNNModel

    structs = _random_structs(10, 0)
    ac = _acsf()
    m = AtomicNNModel(
        hidden_layers=[12, 12],
        epochs=5,
        batch_structures=4,
        r0=0.1,
        eps0=0.01,
        verbose=False,
    )
    m.fit(structs, ac)
    path = str(tmp_path / "m.pt")
    m.save(path)
    m2 = load_mlip_model(path)
    assert type(m2) is AtomicNNModel
    T = m._tensors(AtomicData(structs, ac), np.arange(len(structs)))
    with torch.enable_grad():
        p = m._rows_pred(T, [2], False).detach().numpy()
    s = structs[2]
    q = m2._predict_structure(
        ac.atomic_jacobian(s["lattice"], s["positions"], s["atom_types"])
    )
    np.testing.assert_allclose(p, q, atol=2e-5 * max(1.0, np.abs(q).max()))
    # float32 network: a larger step and tolerance (as the tabular NN test)
    _fd_check(
        MLIPForceField(m2),
        s["lattice"],
        s["positions"],
        s["atom_types"],
        1e-3,
        5e-3,
        5e-4,
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_atomic_nn_learns_and_is_deterministic():
    from akitsushima.fitting.models.atomic_nn import AtomicNNModel

    structs = _sw_si_structures(30, 3)
    ac = _si_acsf()
    kw = dict(
        hidden_layers=[16],
        epochs=30,
        batch_structures=5,
        r0=0.1,
        eps0=0.01,
        deterministic=True,
        verbose=False,
        w_forces=1.0,
        w_stress=1.0,
    )
    a = AtomicNNModel(**kw).fit(structs, ac)
    b = AtomicNNModel(**kw).fit(structs, ac)
    np.testing.assert_array_equal(
        a.predict_structures(structs[:2], ac), b.predict_structures(structs[:2], ac)
    )
    # 30 epochs from scratch: forces clearly below their spread
    y, lab, *_ = AtomicData(structs, ac).rows(0.1, 0.01)
    assert a._train_sc["F"]["RMSE"] < 0.5 * np.std(y[lab == "F"])
