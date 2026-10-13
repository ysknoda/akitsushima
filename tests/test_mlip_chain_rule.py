"""ADR-0004: chain-rule training on tabular E/F/S rows, one weighting rule,
linear models, in-package ACSF, and the MLIP relax/md engine.

Every derivative here is checked against finite differences (ADR-0002,
decision 4); the weighting test is discriminating against the
pre-0.2.0a3 ELM formula (ADR-0003 finding M1).
"""

import importlib.util
import itertools

import numpy as np
import pytest

from akitsushima.fitting.descriptors import ACSF, build_table
from akitsushima.fitting.mlip_engine import MLIPForceField, load_mlip_model
from akitsushima.fitting.models import ELMModel, LinearModel, QuantizedELMModel
from akitsushima.fitting.models.csv_io import build_structure_index, load_mlip_csv

# Detect torch WITHOUT importing it: importing torch at collection time,
# before numba has started its OpenMP thread pool, aborts numba's first
# parallel kernel in a later test module (the numba/torch OpenMP conflict
# documented in utils/jit.py).  The NN tests import it when they run.
HAS_TORCH = importlib.util.find_spec("torch") is not None

# ---------------------------------------------------------------- fixtures


def _chain_table(n_struct=60, n_atoms=3, D=5, seed=0, noise=0.0):
    """Rows exactly consistent with E/N = f(d): F/S rows = N grad f . X."""
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=D), rng.normal(size=D)

    def grad(d):
        return (1 - np.tanh(d @ w1) ** 2) * w1 + 0.5 * w2

    y, lab, w, X = [], [], [], []
    for _ in range(n_struct):
        d = rng.normal(size=D)
        y.append(np.tanh(d @ w1) + 0.5 * d @ w2 - 4.0)
        lab.append("E")
        w.append(float(n_atoms))
        X.append(d)
        for c, k, s in (("F", 3 * n_atoms, 1.0), ("S", 6, 0.1)):
            for _ in range(k):
                x = rng.normal(size=D) * s
                y.append(n_atoms * grad(d) @ x)
                lab.append(c)
                w.append(0.3 if c == "F" else 0.5)
                X.append(x)
    y = np.array(y) + noise * rng.normal(size=len(y))
    return y, np.array(lab), np.array(w), np.array(X)


def _fd_rows_match(model, lab, X, n_atoms=3, h=1e-5, rtol=1e-5, atol=1e-6):
    """F/S rows of predict_table == N * directional derivative of predict()
    along the row's descriptor, at the structure's d (row 0 = E row)."""
    pt = model.predict_table(lab, X)
    d = X[0]
    for k in range(1, 3 * n_atoms + 7):
        x = X[k]
        fd = (
            n_atoms
            * (
                model.predict((d + h * x)[None])[0]
                - model.predict((d - h * x)[None])[0]
            )
            / (2 * h)
        )
        assert pt[k] == pytest.approx(fd, rel=rtol, abs=atol), k
    assert pt[0] == pytest.approx(model.predict(d[None])[0], rel=1e-12)


# ---------------------------------------------------------------- csv / index


def test_structure_index_ties_rows_to_structures():
    lab = list("E" + "F" * 6 + "S" * 6 + "E" + "F" * 3)
    sid, e_row, n_at = build_structure_index(lab)
    assert list(e_row) == [0, 13]
    assert list(n_at) == [2, 1]
    assert (sid[:13] == 0).all() and (sid[13:] == 1).all()


@pytest.mark.parametrize(
    "lab, msg",
    [
        (list("EFF"), "multiple of 3"),
        (list("EFFFSSS"), "expected 0 or 6"),
        (list("FFFE"), "not 'E'"),
        (list("ESSSSSS"), "n_atoms column"),
    ],
)
def test_structure_index_rejects_malformed_structures(lab, msg):
    with pytest.raises(ValueError, match=msg):
        build_structure_index(lab)


def test_csv_metadata_columns_are_not_descriptors(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(
        "target,label,weight,struct_id,rdf_a-a_b0000,n_atoms\n"
        "-4.0,E,2,7,0.5,1\n"
        "0.1,S,1,7,0.2,1\n" * 1
    )
    y, lab, w, X, meta = load_mlip_csv(str(p), verbose=False, return_meta=True)
    assert X.shape == (2, 1)
    assert list(meta["struct_id"]) == [7, 7]
    assert list(meta["n_atoms"]) == [1, 1]
    assert meta["columns"] == ["rdf_a-a_b0000"]
    # A Layout-B `fit` CSV's struct_idx column stays a descriptor column
    # (cmd_fit reads it from X).
    q = tmp_path / "b.csv"
    q.write_text("target,label,weight,struct_idx\n-4.0,E,1.0,0\n")
    assert load_mlip_csv(str(q), verbose=False)[3].shape == (1, 1)


# ---------------------------------------------------------------- chain rule


@pytest.mark.parametrize("activation", ["tanh", "silu", "rff", "gelu"])
def test_elm_force_rows_are_derivatives_of_its_energy(activation):
    y, lab, w, X = _chain_table()
    m = ELMModel(n_hidden=40, activation=activation, reg=1e-6, verbose=False)
    m.fit(y, lab, X, weights=w)
    _fd_rows_match(m, lab[:16], X[:16])


def test_qelm_force_rows_are_derivatives_of_its_energy():
    y, lab, w, X = _chain_table()
    m = QuantizedELMModel(n_hidden=40, reg=1e-6, verbose=False)
    m.fit(y, lab, X, weights=w)
    _fd_rows_match(m, lab[:16], X[:16])


def test_elm_activation_derivatives_match_finite_differences():
    z = np.linspace(-3, 3, 13)
    for name in (
        "tanh",
        "relu",
        "sigmoid",
        "silu",
        "mish",
        "gelu",
        "softplus",
        "snake",
    ):
        m = ELMModel(activation=name, verbose=False)
        h = 1e-6
        fd = (m._act(z + h) - m._act(z - h)) / (2 * h)
        ok = np.abs(z) > 1e-3 if name == "relu" else slice(None)
        assert np.allclose(m._dact(z)[ok], fd[ok], atol=1e-6), name


def test_elm_learns_chain_consistent_data_and_direct_equals_lsqr():
    y, lab, w, X = _chain_table(n_struct=150)
    kw = dict(n_hidden=150, reg=1e-8, verbose=False, w_forces=1.0, w_stress=1.0)
    direct = ELMModel(solver="direct", **kw).fit(y, lab, X, weights=w)
    lsqr = ELMModel(solver="lsqr", lsqr_maxiter=20000, lsqr_atol=1e-14, **kw).fit(
        y, lab, X, weights=w
    )
    # Same minimiser: LSQR's residual cannot be below the direct solve's.
    assert lsqr._solver_info["rnorm"] >= direct._solver_info["rnorm"] * (1 - 1e-6)
    # On the training rows the direct fit reproduces E and F closely.
    assert direct._train_sc["E"]["RMSE"] < 0.02
    assert direct._train_sc["F"]["RMSE"] < 0.2


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_nn_force_rows_are_derivatives_of_its_energy(tmp_path):
    from akitsushima.fitting.models.nn import NNModel

    y, lab, w, X = _chain_table()
    m = NNModel(hidden_layers=[16], epochs=20, verbose=False)
    m.fit(y, lab, X, weights=w)
    # The finite difference runs on a float64 copy of the network (float32
    # round-off of E ~ -4 would swamp it); predict_table itself is float32.
    import copy

    import torch
    net64 = copy.deepcopy(m._net).double()

    def f64(d):
        t = torch.as_tensor(m._normalise(d[None]), dtype=torch.float64)
        return float(net64(t))

    pt = m.predict_table(lab[:16], X[:16])
    d, h = X[0], 1e-5
    for k in range(1, 16):
        fd = 3 * (f64(d + h * X[k]) - f64(d - h * X[k])) / (2 * h)
        assert pt[k] == pytest.approx(fd, rel=1e-4, abs=1e-6), k
    m.save(str(tmp_path / "nn.pt"))
    m2 = NNModel.load(str(tmp_path / "nn.pt"))
    assert not m2.row_independent
    np.testing.assert_allclose(m2.predict_table(lab, X), m.predict_table(lab, X))


def test_row_independent_mode_predicts_each_row_alone():
    y, lab, w, X = _chain_table(n_struct=20)
    m = ELMModel(n_hidden=20, verbose=False, row_independent=True)
    m.fit(y, lab, X, weights=w)
    np.testing.assert_allclose(m.predict_table(lab, X), m.predict(X))


def test_pre_0_2_0a3_elm_file_loads_as_row_independent(tmp_path):
    y, lab, w, X = _chain_table(n_struct=10)
    m = ELMModel(n_hidden=8, verbose=False).fit(y, lab, X, weights=w)
    p = str(tmp_path / "m.npz")
    m.save(p)
    z = dict(np.load(p))
    for k in ("beta0", "row_independent", "descriptor_config"):
        z.pop(k)
    np.savez(p, **z)
    assert ELMModel.load(p).row_independent


# ---------------------------------------------------------------- weighting


def _scaled_rows(n_struct, seed):
    """ADR-0003 T3b layout: realistic label scales (S ~ 0.05) with a
    one-hot label indicator; every row is scale * (f(x) + noise)."""
    wr = np.random.default_rng(99)
    w1, w2 = wr.normal(size=8) / np.sqrt(8), wr.normal(size=8) / np.sqrt(8)
    rng = np.random.default_rng(seed)
    scale = {"E": 1.0, "F": 1.0, "S": 0.05}
    unit = {"E": 4.0, "F": 1.0, "S": 0.01 * 64.0}
    y, lab, w, X = [], [], [], []
    for _ in range(n_struct):
        for c in ["E"] + ["F"] * 12 + ["S"] * 6:
            x = rng.normal(size=8)
            y.append(scale[c] * (np.tanh(x @ w1) + 0.5 * x @ w2
                                 + 0.05 * rng.normal()))
            lab.append(c); w.append(unit[c])
            X.append([float(c == k) for k in "EFS"] + list(x))
    return np.array(y), np.array(lab), np.array(w), np.array(X)


def _old_uw_row_weights(self, labels, w_unit, w_user, targets):
    """The pre-0.2.0a3 ELM weighting, for the control: rows multiplied by
    w_cat * w_unit * norm / sigma2_c with sigma2_c = mean((w_unit y s_c)^2),
    s_c = 1, r0, eps0 -- i.e. squared-residual weight (that)^2."""
    sc = {"E": 1.0, "F": self.r0, "S": self.eps0}
    s2 = {c: max(float(np.mean((w_unit[labels == c] * targets[labels == c]
                                * sc[c]) ** 2)), 1e-30) for c in "EFS"}
    nf = np.exp(np.mean([np.log(s2[c]) for c in "EFS"]))
    wc = {"E": self.w_energy, "F": self.w_forces, "S": self.w_stress}
    mult = np.array([wc[c] * nf / s2[c] for c in labels]) * w_unit * w_user
    return mult ** 2


def test_weighting_rule_is_wcat_times_wunit_squared():
    m = ELMModel(w_energy=2.0, w_forces=0.5, w_stress=3.0, verbose=False)
    w = m._row_weights(np.array(["E", "F", "S"]), np.array([4.0, 0.1, 0.6]),
                       np.array([1.0, 2.0, 1.0]))
    np.testing.assert_allclose(w, [2 * 16, 0.5 * 0.01 * 2, 3 * 0.36])


def test_elm_energy_is_fitted_near_its_noise_with_stress_rows_present():
    """M1 regression (ADR-0003): with realistic label scales the former
    Uncertainty Weighting (1/sigma^2 on the residual, unit weight squared
    again) left E far above its noise; w_cat * w_unit^2 does not.  The
    control run with the old formula shows the test discriminates."""
    y, lab, w, X = _scaled_rows(300, 0)
    yt, lt, _, Xt = _scaled_rows(100, 1)
    e = lt == "E"

    def ratio_E(model):
        model.fit(y, lab, X, weights=w)
        p = model.predict(Xt)
        return np.sqrt(np.mean((p[e] - yt[e]) ** 2)) / 0.05

    # z-score: the test is about the weighting rule, measured (below)
    # with the standardisation it was written for
    kw = dict(n_hidden=256, reg=1e-6, verbose=False, row_independent=True,
              r0=1.0, eps0=0.01, standardize="zscore")
    new = ratio_E(ELMModel(**kw))
    old_model = ELMModel(**kw)
    old_model._row_weights = (lambda lab_, wu, wus:
                              _old_uw_row_weights(old_model, lab_, wu, wus, y))
    old = ratio_E(old_model)
    # Measured: new 3.3 sigma, old 73 sigma (136 sigma at n_hidden 512).
    assert new < 5.0
    assert old > 10.0 * new


# ---------------------------------------------------------------- linear


@pytest.mark.parametrize(
    "method, kw",
    [
        ("ols", {}),
        ("ridge", {"alpha": 1e-8}),
        ("smooth_ridge", {"alpha": 1e-8}),
        ("lasso", {"alpha": 1e-7}),
        ("elastic_net", {"alpha": 1e-7}),
        ("pls", {"n_components": 6}),
        ("bayes_ridge", {}),
        ("ard", {}),
    ],
)
def test_linear_methods_recover_a_linear_potential(method, kw):
    rng = np.random.default_rng(1)
    D, N = 6, 4
    wt, bt = rng.normal(size=D), -4.3
    y, lab, w, X = [], [], [], []
    for _ in range(80):
        d = rng.normal(size=D)
        y.append(d @ wt + bt)
        lab.append("E")
        w.append(N)
        X.append(d)
        for c, k in (("F", 3 * N), ("S", 6)):
            for _ in range(k):
                x = rng.normal(size=D)
                y.append(N * wt @ x)
                lab.append(c)
                w.append(0.3)
                X.append(x)
    m = LinearModel(method=method, verbose=False, **kw)
    m.fit(np.array(y), np.array(lab), np.array(X), weights=np.array(w))
    np.testing.assert_allclose(m.coef_, wt, atol=1e-3)
    assert m.intercept_ == pytest.approx(bt, abs=1e-3)


def test_linear_intercept_does_not_leak_into_force_rows():
    """A constant energy offset must not change F/S predictions: the
    intercept column is 1 on E rows only."""
    y, lab, w, X = _chain_table(n_struct=40)
    a = LinearModel(method="ridge", alpha=1e-6, verbose=False)
    b = LinearModel(method="ridge", alpha=1e-6, verbose=False)
    a.fit(y, lab, X, weights=w)
    b.fit(y + 7.0 * (lab == "E"), lab, X, weights=w)
    np.testing.assert_allclose(a.coef_, b.coef_, rtol=1e-8, atol=1e-10)
    assert b.intercept_ - a.intercept_ == pytest.approx(7.0, abs=1e-8)


def test_linear_alpha_search_picks_from_the_validation_split():
    y, lab, w, X = _chain_table(n_struct=60, noise=0.05)
    m = LinearModel(method="ridge", alpha_search=[1e-8, 1e-3, 1e2], verbose=False).fit(
        y, lab, X, weights=w
    )
    assert m.alpha in (1e-8, 1e-3, 1e2)
    assert len(m.search_results_) == 3


# ---------------------------------------------------------------- ACSF


def _distorted_cell(seed=3):
    rng = np.random.default_rng(seed)
    lat = np.array([[5.1, 0.3, -0.2], [0.4, 4.8, 0.5], [-0.3, 0.2, 5.4]])
    frac = rng.random((7, 3))
    types = ["Sn", "Ge", "Ge", "Sn", "Ge", "Si", "Ge"]
    return lat, frac, types


def _acsf():
    return ACSF.from_grid(
        ["Ge", "Sn", "Si"],
        4.2,
        g2={"eta": [0.5, 2.0], "rs": [0.0, 2.5]},
        g4={"eta": [0.02, 0.1], "zeta": [1, 2, 4], "lambda": [-1, 1]},
        g5={"eta": [0.05], "zeta": [1, 4], "lambda": [-1, 1]},
    )


def test_acsf_species_follow_atomic_number():
    assert _acsf().species == ["Si", "Ge", "Sn"]


def test_acsf_derivatives_match_finite_differences():
    """Cell (~5 A) smaller than 2 r_c: several periodic images of each
    neighbour enter.  Positions and all nine cell components."""
    lat, frac, types = _distorted_cell()
    pos = frac @ lat
    ac = _acsf()
    d, XF, XS = ac.structure_table(lat, pos, types)
    h = 1e-5
    for j in range(len(types)):
        for a in range(3):
            p = pos.copy()
            p[j, a] += h
            dp = ac.structure(lat, p, types)
            p[j, a] -= 2 * h
            dm = ac.structure(lat, p, types)
            np.testing.assert_allclose(-(dp - dm) / (2 * h), XF[j, a], atol=1e-8)
    V = abs(np.linalg.det(lat))
    for a in range(3):
        for b in range(3):
            T = np.eye(3)
            T[a, b] += h
            dp = ac.structure(lat @ T, frac @ lat @ T, types)
            T = np.eye(3)
            T[a, b] -= h
            dm = ac.structure(lat @ T, frac @ lat @ T, types)
            np.testing.assert_allclose(-(dp - dm) / (2 * h) / V, XS[a, b], atol=1e-9)
    np.testing.assert_allclose(XF.sum(axis=0), 0.0, atol=1e-12)


def test_acsf_is_invariant_under_rotation_translation_permutation():
    lat, frac, types = _distorted_cell()
    pos = frac @ lat
    ac = _acsf()
    d = ac.structure(lat, pos, types)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    np.testing.assert_allclose(ac.structure(lat @ R.T, pos @ R.T, types), d, atol=1e-12)
    np.testing.assert_allclose(ac.structure(lat, pos + 1.3, types), d, atol=1e-12)
    perm = np.random.default_rng(0).permutation(len(types))
    np.testing.assert_allclose(
        ac.structure(lat, pos[perm], [types[k] for k in perm]), d, atol=1e-12
    )


def test_acsf_numba_and_python_kernels_agree():
    import akitsushima.fitting.descriptors.acsf as A

    if not hasattr(A._acsf_kernel, "py_func"):
        pytest.skip("numba not installed")
    lat, frac, types = _distorted_cell()
    ac = _acsf()
    ref = ac.structure_table(lat, frac @ lat, types)
    orig = A._acsf_kernel
    try:
        A._acsf_kernel = orig.py_func
        py = ac.structure_table(lat, frac @ lat, types)
    finally:
        A._acsf_kernel = orig
    for a, b in zip(ref, py):
        np.testing.assert_allclose(a, b, atol=1e-13)


# ---------------------------------------------------------------- engine


def _sw_si_structures(n, seed):
    from akitsushima import PotentialEngine, StillingerWeberPotential

    # Stillinger & Weber, PRB 31, 5262 (1985), Si
    sw = {
        "epsilon": 2.1683,
        "sigma": 2.0951,
        "a": 1.80,
        "gamma": 1.20,
        "lambda": 21.0,
        "A": 7.049556277,
        "B": 0.6022245584,
        "p": 4.0,
        "q": 0.0,
        "costheta0": -1.0 / 3.0,
    }
    eng = PotentialEngine().add(
        StillingerWeberPotential({("Si", "Si", "Si"): sw}, species=["Si"])
    )
    base = np.array(
        [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ]
    )
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        e = rng.uniform(-0.02, 0.02, (3, 3))
        lat = np.eye(3) * 5.431 @ (np.eye(3) + 0.5 * (e + e.T))
        pos = base @ lat + rng.normal(0, 0.05, (8, 3))
        types = ["Si"] * 8
        # Labels from the 2x2x2 replica: the 8-atom cell's half-width
        # (2.7 A) cannot host SW's 3.77 A cutoff under the minimum-image
        # convention, and the replica's E / 8, first 8 forces and stress
        # are exactly the periodic 8-atom cell's.
        shifts = np.array(list(itertools.product(range(2), repeat=3))) @ lat
        big = (pos[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        E, F, S = eng.compute(2 * lat, big, types * 8)
        E, F = E / 8, F[:8]
        out.append(
            dict(
                lattice=lat,
                positions=pos,
                atom_types=types,
                ref_energy=E / 8,
                ref_forces=F,
                ref_stress=S,
            )
        )
    return out


@pytest.mark.parametrize("kind", ["elm", "linear", "nn"])
def test_mlip_engine_forces_and_stress_are_derivatives_of_its_energy(kind, tmp_path):
    if kind == "nn" and not HAS_TORCH:
        pytest.skip("torch not installed")
    structs = _sw_si_structures(12, 0)
    ac = ACSF.from_grid(
        ["Si"],
        3.771,
        g2={"eta": [4.0], "rs": [2.2, 2.6, 3.0]},
        g4={"eta": [0.05], "zeta": [1, 2], "lambda": [-1, 1]},
    )
    y, lab, w, X = build_table(structs, ac, r0=0.1, eps0=0.01)
    if kind == "elm":
        m = ELMModel(n_hidden=30, verbose=False)
    elif kind == "linear":
        m = LinearModel(method="ridge", alpha=1e-6, verbose=False)
    else:
        from akitsushima.fitting.models.nn import NNModel

        m = NNModel(hidden_layers=[8], epochs=5, verbose=False)
    m.fit(y, lab, X, weights=w)
    m.descriptor_config = ac.to_json()
    path = str(tmp_path / ("m.pt" if kind == "nn" else "m.npz"))
    m.save(path)
    ff = MLIPForceField(load_mlip_model(path))
    s = structs[0]
    lat, pos, types = s["lattice"], s["positions"], s["atom_types"]
    E0, F0, S0 = ff(lat, pos, types)
    # float32 NN energies: a larger step and tolerance
    h, tol = (1e-3, 2e-3) if kind == "nn" else (1e-5, 1e-6)
    for j in (0, 5):
        for a in range(3):
            p = pos.copy()
            p[j, a] += h
            Ep = ff(lat, p, types)[0]
            p[j, a] -= 2 * h
            Em = ff(lat, p, types)[0]
            assert -(Ep - Em) / (2 * h) == pytest.approx(F0[j, a], abs=tol)
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
            assert -(Ep - Em) / (2 * h) / V == pytest.approx(S0[a, b], abs=tol / 10)
    np.testing.assert_allclose(F0.sum(axis=0), 0.0, atol=1e-9)


def test_mlip_engine_refuses_models_that_are_not_force_fields():
    y, lab, w, X = _chain_table(n_struct=10)
    m = ELMModel(n_hidden=8, verbose=False, row_independent=True)
    m.fit(y, lab, X, weights=w)
    m.descriptor_config = _acsf().to_json()
    with pytest.raises(ValueError, match="row_independent"):
        MLIPForceField(m)
    m2 = ELMModel(n_hidden=8, verbose=False).fit(y, lab, X, weights=w)
    with pytest.raises(ValueError, match="external CSV"):
        MLIPForceField(m2)


# ---------------------------------------------------------------- CLI


def _write_layout_b(root, structs):
    """The classical `fit` layout: structures.csv + <k>/POSCAR."""
    from akitsushima.io.poscar import write_poscar
    rows = ["target,label,weight,struct_idx"]
    for k, s in enumerate(structs):
        (root / str(k)).mkdir(parents=True)
        write_poscar(str(root / str(k) / "POSCAR"), s["lattice"],
                     s["positions"], s["atom_types"])
        S = s["ref_stress"]
        rows.append(f"{s['ref_energy']:.12f},E,1.0,{k}")
        rows += [f"{f:.12f},F,1.0,{k}" for f in s["ref_forces"].ravel()]
        rows += [f"{S[a, b]:.12e},S,1.0,{k}"
                 for a, b in ((0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1))]
    (root / "structures.csv").write_text("\n".join(rows) + "\n")


@pytest.mark.parametrize("model", ["elm", "linear"])
def test_cli_mlip_with_acsf_then_relax_and_md_on_the_model(model, tmp_path,
                                                           monkeypatch):
    import yaml
    from test_cli_smoke import _ns

    from akitsushima.cli.parser import _DISPATCH
    from akitsushima.io.poscar import write_poscar

    monkeypatch.chdir(tmp_path)
    structs = _sw_si_structures(16, 1)
    _write_layout_b(tmp_path / "train", structs)
    desc = {"type": "acsf", "r_cut": 3.771,
            "g2": {"eta": [4.0], "rs": [2.2, 2.6, 3.0]},
            "g4": {"eta": [0.05], "zeta": [1, 2], "lambda": [-1, 1]}}
    ml = {"output_dir": "out", "model": model, "descriptor": desc,
          "r0": 0.1, "eps0": 0.01, "n_hidden": 20, "method": "ridge",
          "alpha": 1e-6, "write_table": True}
    (tmp_path / "mlip.yaml").write_text(yaml.safe_dump(
        {"structure": {"training_csv": "train/structures.csv"},
         "potential": {"species": ["Si"]}, "mlip": ml}))
    assert _DISPATCH["mlip"](_ns(tmp_path / "mlip.yaml")) == 0
    fname = {"elm": "elm_model.npz", "linear": "linear_model.npz"}[model]
    import json
    summ = json.loads((tmp_path / "out" / "mlip_summary.json").read_text())
    assert summ["engine"] is True
    assert (tmp_path / "out" / "descriptor_table.csv").is_file()

    # relax/md build their own neighbour list, which needs every cell width
    # > 2 (r_cut + skin); a 2x2x2 supercell of a training cell satisfies
    # it (the ACSF itself handles the 5.4 A training cells).
    s = structs[0]
    shifts = np.array([[i, j, k] for i in range(2) for j in range(2)
                       for k in range(2)]) @ s["lattice"]
    pos = (s["positions"][None] + shifts[:, None]).reshape(-1, 3)
    write_poscar(str(tmp_path / "POSCAR"), 2 * s["lattice"], pos,
                 ["Si"] * len(pos))
    pot = {"type": "MLIP", "model": f"out/{fname}"}
    (tmp_path / "relax.yaml").write_text(yaml.safe_dump(
        {"structure": {"poscar": "POSCAR"}, "potential": pot,
         "relax": {"output_dir": "relax_out", "max_steps": 5,
                   "relax_cell": True}}))
    assert _DISPATCH["relax"](_ns(tmp_path / "relax.yaml")) == 0
    (tmp_path / "md.yaml").write_text(yaml.safe_dump(
        {"structure": {"poscar": "POSCAR"}, "potential": pot,
         "md": {"output_dir": "md_out", "ensemble": "nve", "n_steps": 5,
                "temperature": 300.0, "dt": 1.0}}))
    assert _DISPATCH["md"](_ns(tmp_path / "md.yaml")) == 0


@pytest.mark.parametrize("method", ["bayes_ridge", "ard"])
def test_bayesian_linear_converges_with_noise_optimal_weights(method):
    """Weights 1/sigma^2 of order 1e4-1e5 (the noise-optimal choice).  The
    first evidence implementation judged convergence on the coefficients,
    stopped while a strong initial prior still held them near 0, and left
    SW-Si forces at 270 sigma; the fixed-point iteration must now settle
    the hyperparameters and recover the data at the noise level."""
    rng = np.random.default_rng(5)
    D, N, sig_E, sig_F = 8, 8, 0.002, 0.05
    wt, bt = rng.normal(size=D), -4.3
    y, lab, w, X = [], [], [], []
    for _ in range(150):
        d = rng.normal(size=D)
        y.append(d @ wt + bt + sig_E * rng.normal()); lab.append("E")
        w.append(N); X.append(d)
        for _ in range(3 * N):
            x = rng.normal(size=D)
            y.append(N * wt @ x + sig_F * rng.normal()); lab.append("F")
            w.append(0.1); X.append(x)
    y, lab, w, X = map(np.array, (y, lab, w, X))
    m = LinearModel(method=method, verbose=False,
                    w_energy=1 / (sig_E * N) ** 2, w_forces=1 / (sig_F * 0.1) ** 2)
    m.fit(y, lab, X, weights=w, val_fraction=0.0)
    assert m._evidence_iter < m.max_iter
    p = m.predict_table(lab, X)
    f = lab == "F"
    assert np.sqrt(np.mean((p[f] - y[f]) ** 2)) / sig_F < 1.1
    np.testing.assert_allclose(m.coef_, wt, atol=5e-3)


def test_bayesian_ridge_ignores_directions_the_data_cannot_resolve():
    """Exactly dependent descriptor columns (the SW-Si ACSF set has five:
    e.g. G5 with eta = 0 and lambda = +1 plus lambda = -1 loses its angle)
    give G = A^T A eigenvalues at round-off level, which the evidence
    iteration now ignores; dependent columns must not change the fit.
    This is a property check: the degenerate fixed point it guards
    against (prior precision 1.5e-11, SW-Si v2 at 14 / 38 / 13 sigma) was
    only reproduced on the full SW-Si v2 set, by
    example/mlip/adr0004/sw_si_verify.py --dataset v2 --models bayes."""
    rng = np.random.default_rng(6)
    D, N, sig_E, sig_F = 6, 8, 0.002, 0.05
    wt, bt = rng.normal(size=D), -4.3
    y, lab, w, X = [], [], [], []
    for _ in range(150):
        d = rng.normal(size=D)
        y.append(d @ wt + bt + sig_E * rng.normal()); lab.append("E")
        w.append(N); X.append(d)
        for _ in range(3 * N):
            x = rng.normal(size=D)
            y.append(N * wt @ x + sig_F * rng.normal()); lab.append("F")
            w.append(0.1); X.append(x)
    y, lab, w, X = map(np.array, (y, lab, w, X))
    kw = dict(method="bayes_ridge", verbose=False,
              w_energy=1 / (sig_E * N) ** 2, w_forces=1 / (sig_F * 0.1) ** 2)
    ref = LinearModel(**kw).fit(y, lab, X, weights=w, val_fraction=0.0)
    mix = rng.normal(size=(D, 4))
    Xz = np.hstack([X, X @ mix])            # four exactly dependent columns
    m = LinearModel(**kw).fit(y, lab, Xz, weights=w, val_fraction=0.0)
    np.testing.assert_allclose(m.predict_table(lab, Xz), ref.predict_table(lab, X),
                               rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------- RVFL


@pytest.mark.parametrize("cls, kw", [
    (ELMModel, {}), (ELMModel, {"activation": "rff"}), (QuantizedELMModel, {}),
])
def test_rvfl_rows_are_derivatives_of_its_energy(cls, kw, tmp_path):
    y, lab, w, X = _chain_table()
    m = cls(n_hidden=30, reg=1e-6, verbose=False, direct_link=True, **kw)
    m.fit(y, lab, X, weights=w)
    _fd_rows_match(m, lab[:16], X[:16])
    f, g = m.energy_and_gradient(X[0])
    assert f == pytest.approx(m.predict(X[:1])[0], rel=1e-12)
    assert 3 * g @ X[1] == pytest.approx(m.predict_table(lab[:16], X[:16])[1],
                                         rel=1e-10)
    p = str(tmp_path / "m.npz")
    m.save(p)
    m2 = cls.load(p)
    assert m2.direct_link
    np.testing.assert_allclose(m2.predict_table(lab, X), m.predict_table(lab, X))


def test_rvfl_with_large_reg_reduces_to_the_linear_model():
    """reg acts on the hidden weights only, so reg -> infinity leaves the
    direct-link part: the ridge-free linear model on the same rows."""
    rng = np.random.default_rng(1)
    D, N = 6, 4
    wt, bt = rng.normal(size=D), -4.3
    y, lab, w, X = [], [], [], []
    for _ in range(60):
        d = rng.normal(size=D)
        y.append(d @ wt + bt); lab.append("E"); w.append(N); X.append(d)
        for _ in range(3 * N):
            x = rng.normal(size=D)
            y.append(N * wt @ x); lab.append("F"); w.append(0.3); X.append(x)
    y, lab, w, X = map(np.array, (y, lab, w, X))
    m = ELMModel(n_hidden=50, reg=1e12, direct_link=True, verbose=False)
    m.fit(y, lab, X, weights=w)
    assert np.abs(m._beta).max() < 1e-6
    np.testing.assert_allclose(m._to_raw(m._gamma), wt, atol=1e-6)


def test_elm_reg_search_picks_from_the_validation_split():
    y, lab, w, X = _chain_table(n_struct=80, noise=0.05)
    m = ELMModel(n_hidden=60, verbose=False, direct_link=True,
                 reg_search=[1e-6, 1e-2, 1e2]).fit(y, lab, X, weights=w)
    losses = dict(m.reg_search_results_)
    assert m.reg == min(losses, key=losses.get)


def test_nn_direct_link_round_trip(tmp_path):
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    from akitsushima.fitting.models.nn import NNModel
    y, lab, w, X = _chain_table()
    m = NNModel(hidden_layers=[8], epochs=5, verbose=False, direct_link=True)
    m.fit(y, lab, X, weights=w)
    p = str(tmp_path / "nn.pt")
    m.save(p)
    m2 = NNModel.load(p)
    assert m2.direct_link
    np.testing.assert_allclose(m2.predict_table(lab, X), m.predict_table(lab, X))


def test_nn_deterministic_training_is_bit_reproducible():
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    from akitsushima.fitting.models.nn import NNModel
    y, lab, w, X = _chain_table(n_struct=40)
    preds = []
    for _ in range(2):
        m = NNModel(hidden_layers=[16, 16], epochs=15, batch_size=64,
                    verbose=False, deterministic=True)
        m.fit(y, lab, X, weights=w)
        preds.append(m.predict_table(lab, X))
    np.testing.assert_array_equal(preds[0], preds[1])


def test_model_defaults_follow_the_adr_0004_follow_up():
    """ELM/QELM: direct link on; NN: softplus with a direct link."""
    assert ELMModel(verbose=False).direct_link
    assert QuantizedELMModel(verbose=False).direct_link
    if HAS_TORCH:
        from akitsushima.fitting.models.nn import NNModel
        m = NNModel(verbose=False)
        assert m.activation == "softplus" and m.direct_link
        assert not m.deterministic
        assert m.batch_structures == 30 and m.optimizer == "adam"
        assert m.early_stopping_patience == 150 and m.epochs == 1500


def test_nn_batch_structures_and_lbfgs_train_and_stay_consistent():
    """batch_structures and the full-batch L-BFGS optimiser run, keep the
    F rows derivatives of the energy, and L-BFGS is reproducible."""
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    import copy

    import torch

    from akitsushima.fitting.models.nn import NNModel
    y, lab, w, X = _chain_table(n_struct=40)
    for kw in ({"batch_structures": 4},
               {"optimizer": "lbfgs", "learning_rate": 1.0}):
        preds = []
        for _ in range(2):
            m = NNModel(hidden_layers=[8], epochs=5, verbose=False,
                        deterministic=True, **kw).fit(y, lab, X, weights=w)
            preds.append(m.predict_table(lab, X))
        np.testing.assert_array_equal(preds[0], preds[1])
        net64 = copy.deepcopy(m._net).double()

        def f64(d):
            t = torch.as_tensor(m._normalise(d[None]), dtype=torch.float64)
            return float(net64(t))
        d, h = X[0], 1e-5
        pt = preds[1]
        for k in range(1, 16):
            fd = 3 * (f64(d + h * X[k]) - f64(d - h * X[k])) / (2 * h)
            assert pt[k] == pytest.approx(fd, rel=1e-4, abs=1e-6)
    with pytest.raises(ValueError, match="optimizer"):
        NNModel(optimizer="sgd", verbose=False)


@pytest.mark.parametrize("model", ["elm", "linear"])
def test_written_descriptor_tables_retrain_to_the_same_model(model, tmp_path,
                                                            monkeypatch):
    """write_table round trip (ADR-0021 plan, F2): the training and test
    tables written by an ACSF run, read back as a descriptor CSV, give the
    same model and the same test score."""
    import json

    import yaml
    from test_cli_smoke import _ns

    from akitsushima.cli.parser import _DISPATCH
    from akitsushima.fitting.models.csv_io import load_mlip_csv

    monkeypatch.chdir(tmp_path)
    _write_layout_b(tmp_path / "train", _sw_si_structures(16, 1))
    _write_layout_b(tmp_path / "test", _sw_si_structures(6, 2))
    desc = {"type": "acsf", "r_cut": 3.771,
            "g2": {"eta": [4.0], "rs": [2.2, 2.6, 3.0]},
            "g4": {"eta": [0.05], "zeta": [1, 2], "lambda": [-1, 1]}}
    common = {"model": model, "r0": 0.1, "eps0": 0.01, "n_hidden": 20,
              "method": "ridge", "alpha": 1e-6, "val_fraction": 0.2}
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(
        {"structure": {"training_csv": "train/structures.csv"},
         "potential": {"species": ["Si"]},
         "mlip": {**common, "output_dir": "a", "descriptor": desc,
                  "write_table": True, "test_csv": "test/structures.csv"}}))
    assert _DISPATCH["mlip"](_ns(tmp_path / "a.yaml")) == 0
    assert (tmp_path / "a" / "descriptor_table_test.csv").is_file()
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(
        {"structure": {"training_csv": "a/descriptor_table.csv"},
         "mlip": {**common, "output_dir": "b",
                  "test_csv": "a/descriptor_table_test.csv"}}))
    assert _DISPATCH["mlip"](_ns(tmp_path / "b.yaml")) == 0

    sa = json.loads((tmp_path / "a" / "mlip_summary.json").read_text())
    sb = json.loads((tmp_path / "b" / "mlip_summary.json").read_text())
    for c in ("E", "F", "S"):
        assert sb["rmse_test"][c] == pytest.approx(sa["rmse_test"][c], rel=1e-6)
    from akitsushima.fitting import ELMModel, LinearModel
    cls = ELMModel if model == "elm" else LinearModel
    fname = {"elm": "elm_model.npz", "linear": "linear_model.npz"}[model]
    ma, mb = (cls.load(str(tmp_path / d / fname)) for d in ("a", "b"))
    _, lab, _, X = load_mlip_csv(str(tmp_path / "a" / "descriptor_table_test.csv"),
                                 verbose=False)
    pa, pb = ma.predict_table(lab, X), mb.predict_table(lab, X)
    assert np.max(np.abs(pa - pb)) < 1e-8


def test_write_table_csv_round_trips_bit_for_bit(tmp_path):
    from akitsushima.fitting.descriptors import write_table_csv
    from akitsushima.fitting.models.csv_io import load_mlip_csv

    rng = np.random.default_rng(3)
    X = rng.normal(size=(7, 4)) * 10.0 ** rng.integers(-12, 4, size=(7, 4))
    y, w = rng.normal(size=7), rng.uniform(0.1, 70.0, size=7)
    lab = np.array(["E"] + ["F"] * 6)
    write_table_csv(tmp_path / "t.csv", y, lab, w, X)
    y2, lab2, w2, X2 = load_mlip_csv(str(tmp_path / "t.csv"), verbose=False)
    assert np.array_equal(y2, y) and np.array_equal(w2, w) and np.array_equal(X2, X)
    assert list(lab2) == list(lab)
