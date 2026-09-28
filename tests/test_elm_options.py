"""ELM / QELM options of ADR-0013: scale search, data-centred bias, PCA
whitening, radially correlated weights, smoothness penalty on the direct
link, QELM level sets and the sparse product.

- Each option: training rows equal the engine and the engine's forces and
  stress are finite-difference derivatives of its energy, after save / load
  (tabular and atom-centred).
- Each option does what it says (whitened covariance, weight correlation,
  bias centres, smoothed gamma, level sets, search results).
- Refusals: PCA with channel-layout options, external descriptors with
  radial correlation, bad level settings; n_bits is a deprecated alias.
"""

import numpy as np
import pytest
from test_mlip_atom_centred import _fd4_check, _fd_check, _random_structs
from test_mlip_chain_rule import _acsf, _sw_si_structures

from akitsushima.fitting.descriptors import ACSF, build_table
from akitsushima.fitting.mlip_engine import MLIPForceField, load_mlip_model
from akitsushima.fitting.models import ELMModel, QuantizedELMModel
from akitsushima.fitting.models.atomic import AtomicELMModel, AtomicQELMModel


def _si_acsf():
    return ACSF.from_grid(
        ["Si"],
        3.771,
        g2={"eta": [4.0, 12.5], "rs": [2.2, 2.4, 2.6, 2.8, 3.0]},
        g5={"eta": [0.05], "zeta": [1, 2], "lambda": [-1, 1]},
    )


@pytest.fixture(scope="module")
def si_table():
    structs = _sw_si_structures(16, 0)
    ac = _si_acsf()
    y, lab, w, X = build_table(structs, ac, r0=0.1, eps0=0.01)
    return structs, ac, y, lab, w, X


TABULAR = [
    (ELMModel, dict(bias_init="data")),
    (ELMModel, dict(standardize="pca")),
    (
        ELMModel,
        dict(standardize="pca", pca_eps=0.0, activation="rff", direct_link=False),
    ),
    (ELMModel, dict(w_correlation="radial", w_corr_length=0.3)),
    (ELMModel, dict(standardize="zscore")),
    (ELMModel, dict(w_init_scale_search=True, reg_search=[1e-6, 1e-3])),
    (
        ELMModel,
        dict(
            w_correlation="radial",
            w_corr_length_search=[0.1, 0.4],
            reg_search=[1e-6, 1e-3],
        ),
    ),
    (QuantizedELMModel, dict(levels="ternary")),
    (QuantizedELMModel, dict(levels="uniform", n_levels=7)),
    (QuantizedELMModel, dict(matmul="sparse", p_zero=0.8)),
    (QuantizedELMModel, dict(levels="ternary", bias_init="data", standardize="pca")),
    (QuantizedELMModel, dict(w_correlation="radial", w_init_scale_search=[1, 4])),
]


@pytest.mark.parametrize("cls,kw", TABULAR)
def test_tabular_option_rows_equal_engine_and_engine_is_conservative(
    si_table, cls, kw, tmp_path
):
    structs, ac, y, lab, w, X = si_table
    m = cls(n_hidden=30, verbose=False, **kw)
    m.descriptor_config = ac.to_json()
    m.fit(y, lab, X, weights=w)
    path = str(tmp_path / "m.npz")
    m.save(path)
    m2 = load_mlip_model(path)
    assert type(m2) is cls
    np.testing.assert_allclose(
        m2.predict_table(lab, X), m.predict_table(lab, X), rtol=0, atol=1e-10
    )
    ff = MLIPForceField(m2)
    s = structs[0]
    lat, pos, types = s["lattice"], s["positions"], s["atom_types"]
    E0, F0, S0 = ff(lat, pos, types)
    # the training rows of structure 0 are the engine's E/N, F, S
    p = m.predict_table(lab, X)[: 1 + 3 * len(types)]
    assert p[0] == pytest.approx(E0 / len(types), abs=1e-10)
    np.testing.assert_allclose(p[1:], F0.ravel(), atol=1e-9)
    h, tol = 1e-5, 1e-6
    for j in (0, 5):
        for a in range(3):
            q = pos.copy()
            q[j, a] += h
            Ep = ff(lat, q, types)[0]
            q[j, a] -= 2 * h
            Em = ff(lat, q, types)[0]
            assert -(Ep - Em) / (2 * h) == pytest.approx(F0[j, a], abs=tol)
    V = abs(np.linalg.det(lat))
    frac = pos @ np.linalg.inv(lat)
    for a, b in [(0, 0), (1, 2), (2, 0)]:
        T = np.eye(3)
        T[a, b] += h
        Ep = ff(lat @ T, frac @ lat @ T, types)[0]
        T[a, b] -= 2 * h
        Em = ff(lat @ T, frac @ lat @ T, types)[0]
        assert -(Ep - Em) / (2 * h) / V == pytest.approx(S0[a, b], abs=tol / 10)
    np.testing.assert_allclose(F0.sum(axis=0), 0.0, atol=1e-9)


ATOMIC = [
    (AtomicELMModel, dict(bias_init="data")),
    (AtomicELMModel, dict(standardize="pca")),
    (
        AtomicELMModel,
        dict(w_correlation="radial", w_corr_length=0.5),
    ),
    (
        AtomicELMModel,
        dict(
            w_init_scale_search=[0.5, 2.0],
            standardize="zscore",
            reg_search=[1e-6, 1e-3],
        ),
    ),
    (AtomicQELMModel, dict(levels="uniform", n_levels=5, matmul="sparse")),
    (AtomicQELMModel, dict(levels="ternary", standardize="pca", bias_init="data")),
]


@pytest.mark.parametrize("cls,kw", ATOMIC)
def test_atomic_option_rows_equal_engine_and_engine_is_conservative(cls, kw, tmp_path):
    structs = _random_structs(12, 0)
    m = cls(n_hidden=24, r0=0.1, eps0=0.01, reg=1e-6, verbose=False, **kw)
    m.fit(structs, _acsf())
    path = str(tmp_path / "m.npz")
    m.save(path)
    m2 = load_mlip_model(path)
    assert type(m2) is cls
    s = structs[1]
    J = _acsf().atomic_jacobian(s["lattice"], s["positions"], s["atom_types"])
    coef = np.concatenate([u._coef() for u in m.units])
    np.testing.assert_allclose(
        m._design(J) @ coef, m2._predict_structure(J), atol=1e-10
    )
    ff = MLIPForceField(m2)
    if kw.get("standardize", "pca") == "pca" and "w_correlation" not in kw:
        # PCA keeps near-null directions (gain up to 1/sqrt(pca_eps)): the
        # fitted energy is strongly curved and the central difference is
        # off by 2e-6 at h = 1e-5 (falling as h^2), so a fourth-order
        # difference is used
        _fd4_check(ff, s["lattice"], s["positions"], s["atom_types"], 3e-5)
    else:
        _fd_check(ff, s["lattice"], s["positions"], s["atom_types"], 1e-5, 1e-6, 1e-7)


# ------------------------------------------------------------ what they do


def test_pca_whitens_the_training_inputs_and_drops_null_directions(si_table):
    structs, ac, y, lab, w, X = si_table
    X2 = np.hstack([X, X[:, :1]])  # an exactly duplicated column
    m = ELMModel(n_hidden=20, standardize="pca", pca_eps=0.0, verbose=False)
    m.fit(y, lab, X2, weights=w, val_fraction=0.0)
    e = lab == "E"
    Xn = m._normalise(X2[e])
    assert m._P.shape[1] <= X.shape[1]  # the duplicate adds no direction
    np.testing.assert_allclose(Xn.T @ Xn / len(Xn), np.eye(Xn.shape[1]), atol=1e-6)


def test_radial_correlation_follows_the_kernel_and_keeps_the_variance():
    ac = _si_acsf()
    m = ELMModel(
        n_hidden=20000,
        w_correlation="radial",
        w_corr_length=0.2,
        direct_link=False,
        verbose=False,
    )
    m._groups = ac.radial_groups(per_atom=False)
    D = ac.n_features
    m._init_hidden(D)
    W = m._W / np.sqrt(2.0 / (D + m.n_hidden))
    idx, rs = m._groups[0]
    C = np.corrcoef(W[idx])
    K = np.exp(-((rs[:, None] - rs[None, :]) ** 2) / (2 * 0.2**2))
    np.testing.assert_allclose(C, K, atol=0.03)
    np.testing.assert_allclose(W.var(axis=1), 1.0, atol=0.05)
    # G5 channels stay independent of the G2 ones
    g5 = D - 1
    assert abs(np.corrcoef(W[idx[0]], W[g5])[0, 1]) < 0.03


def test_data_bias_centres_every_unit_on_a_training_input(si_table):
    structs, ac, y, lab, w, X = si_table
    m = ELMModel(n_hidden=40, bias_init="data", verbose=False)
    m.fit(y, lab, X, weights=w)
    Z = m._preact(m._normalise(X[lab == "E"]))
    assert np.abs(Z).min(axis=0).max() < 1e-10


def test_pca_is_the_default_and_old_files_load_as_zscore(si_table, tmp_path):
    structs, ac, y, lab, w, X = si_table
    assert ELMModel().standardize == "pca"
    assert QuantizedELMModel().standardize == "pca"
    assert ELMModel(w_correlation="radial").standardize == "zscore"
    m = ELMModel(n_hidden=10, verbose=False).fit(y, lab, X, weights=w)
    assert m._P is not None and m._P.shape == (X.shape[1], X.shape[1])
    a = AtomicELMModel(n_hidden=8, verbose=False).fit(structs[:6], ac)
    assert all(u.standardize == "pca" and u._P is not None for u in a.units)
    # a file written before ADR-0013 (no X_P / elm_options) is a z-score model
    z = ELMModel(n_hidden=10, standardize="zscore", verbose=False)
    z.fit(y, lab, X, weights=w)
    path = str(tmp_path / "old.npz")
    z.save(path)
    d = dict(np.load(path))
    for k in ("X_P", "elm_options"):
        d.pop(k)
    np.savez(path, **d)
    old = ELMModel.load(path)
    assert old.standardize == "zscore" and old._P is None
    np.testing.assert_allclose(old.predict_table(lab, X), z.predict_table(lab, X))


def test_scale_search_reports_every_candidate_and_keeps_the_best(si_table):
    structs, ac, y, lab, w, X = si_table
    m = ELMModel(
        n_hidden=20, w_init_scale_search=True, reg_search=[1e-6, 1e-2], verbose=False
    )
    m.fit(y, lab, X, weights=w)
    res = m.hyper_search_results_
    assert [r[0] for r in res] == [0.5, 1.0, 2.0, 4.0, 8.0]
    assert m._w_scale_factor == min(res, key=lambda r: r[-1])[0]


def test_qelm_level_sets():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(50, 30))
    t = QuantizedELMModel(n_hidden=30, levels="ternary", p_zero=0.5, verbose=False)
    Wq = t._quantize_W(W)
    for j in range(30):
        v = np.unique(np.abs(Wq[:, j]))
        assert len(v) == 2 and v[0] == 0.0
        assert np.sum(Wq[:, j] ** 2) == pytest.approx(np.sum(W[:, j] ** 2))
    u = QuantizedELMModel(
        n_hidden=30, levels="uniform", n_levels=7, p_zero=0.0, verbose=False
    )
    Wq = u._quantize_W(W)
    k = Wq / u._W_scale
    np.testing.assert_allclose(k, np.round(k), atol=1e-12)
    assert np.abs(k).max() == 3
    p = QuantizedELMModel(n_hidden=30, max_exponent=2, p_zero=0.0, verbose=False)
    k = np.abs(p._quantize_W(W) / p._W_scale)
    assert set(np.unique(np.round(k, 12))) <= {0.0, 1.0, 2.0, 4.0}


def test_refusals_and_deprecated_alias(si_table):
    structs, ac, y, lab, w, X = si_table
    with pytest.raises(ValueError, match="PCA whitening mixes"):
        ELMModel(standardize="pca", w_correlation="radial")
    with pytest.raises(ValueError, match="in-package"):
        ELMModel(n_hidden=10, w_correlation="radial", verbose=False).fit(
            y, lab, X, weights=w
        )  # no descriptor_config: external table
    with pytest.raises(ValueError, match="rff"):
        ELMModel(activation="rff", bias_init="data")
    with pytest.raises(ValueError, match="levels"):
        QuantizedELMModel(levels="int4")
    with pytest.raises(ValueError, match="n_levels"):
        QuantizedELMModel(levels="uniform", n_levels=4)
    with pytest.warns(DeprecationWarning, match="max_exponent"):
        q = QuantizedELMModel(n_bits=2)
    assert q.max_exponent == 2 and q.n_bits == 2
    with pytest.warns(DeprecationWarning, match="max_exponent"):
        a = AtomicQELMModel(n_bits=2)
    assert a.unit_kw["max_exponent"] == 2


def test_cli_passes_the_options_to_the_models():
    from akitsushima.cli.cmd_mlip import _build_atomic_model, _build_model

    ml = dict(
        n_hidden=10,
        standardize="pca",
        bias_init="data",
        levels="ternary",
        matmul="sparse",
    )
    m = _build_model("qelm", ml, False)
    assert (m.standardize, m.bias_init, m.levels, m.matmul) == (
        "pca",
        "data",
        "ternary",
        "sparse",
    )
    e = _build_model("elm", dict(w_correlation="radial"), False)
    assert e.w_correlation == "radial" and not hasattr(e, "levels")
    a = _build_atomic_model("qelm", ml, False)
    assert a.unit_kw["levels"] == "ternary" and a.unit_kw["standardize"] == "pca"
