"""min_feature_std (ADR-0010, open finding A4): input columns whose training
std lies below the threshold are masked in every model.

* feature_std: std < 1e-12 keeps the historical unit scale, std in
  [1e-12, min_feature_std) becomes +inf (input and gradient exactly 0).
* Tabular ELM / QELM / linear / NN: a masked column cannot change any E, F
  or S prediction; the mask survives save / load; the default masks
  nothing.
* Atom-centred: masking every ACSF column leaves forces and stress exactly 0.
* The CLI passes the key to every model.

torch is imported inside the tests (numba / torch OpenMP note in
test_mlip_chain_rule.py).
"""

import importlib.util

import numpy as np
import pytest
from test_mlip_atom_centred import _random_structs
from test_mlip_chain_rule import _acsf, _chain_table, _fd_rows_match

from akitsushima.fitting.mlip_engine import MLIPForceField
from akitsushima.fitting.models.atomic import AtomicELMModel
from akitsushima.fitting.models.elm import ELMModel, feature_std
from akitsushima.fitting.models.linear import LinearModel
from akitsushima.fitting.models.quantized_elm import QuantizedELMModel

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_feature_std_masks_only_the_band_below_the_threshold():
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    X = np.column_stack([np.full(200, 3.0), 1e-8 * base, 1e-3 * base, base])
    sd = X.std(axis=0)
    np.testing.assert_array_equal(feature_std(X), [1.0, sd[1], sd[2], sd[3]])
    out = feature_std(X, 1e-4)
    assert out[0] == 1.0 and np.isinf(out[1])
    np.testing.assert_array_equal(out[2:], sd[2:])


def _table_with_quiet_column(seed=0):
    """_chain_table plus a column that hardly varies (std ~ 1e-7)."""
    y, lab, w, X = _chain_table(seed=seed)
    rng = np.random.default_rng(seed + 1)
    quiet = 1.0 + 1e-7 * rng.normal(size=len(y))
    quiet[lab != "E"] = 1e-7 * rng.normal(size=int(np.sum(lab != "E")))
    return y, lab, w, np.column_stack([X, quiet])


def _tabular_models():
    out = [
        lambda **k: ELMModel(n_hidden=24, verbose=False, **k),
        lambda **k: ELMModel(n_hidden=24, standardize="zscore", verbose=False, **k),
        lambda **k: QuantizedELMModel(n_hidden=24, verbose=False, **k),
        lambda **k: LinearModel(method="ridge", alpha=1e-6, verbose=False, **k),
    ]
    if HAS_TORCH:
        from akitsushima.fitting.models.nn import NNModel

        out.append(lambda **k: NNModel(hidden_layers=[8], epochs=5, verbose=False, **k))
    return out


@pytest.mark.parametrize("make", _tabular_models())
def test_masked_column_cannot_change_any_prediction(make, tmp_path):
    y, lab, w, X = _table_with_quiet_column()
    m = make(min_feature_std=1e-4)
    m.fit(y, lab, X, weights=w)
    assert np.isinf(m._X_std[-1]) and np.all(np.isfinite(m._X_std[:-1]))
    p0 = m.predict_table(lab, X)
    assert np.all(np.isfinite(p0))
    X2 = X.copy()
    X2[:, -1] += 10.0 * np.random.default_rng(5).normal(size=len(X))
    np.testing.assert_array_equal(m.predict_table(lab, X2), p0)
    ext = ".pt" if type(m).__name__ == "NNModel" else ".npz"
    m.save(str(tmp_path / ("m" + ext)))
    m2 = type(m).load(str(tmp_path / ("m" + ext)))
    assert np.isinf(m2._X_std[-1])
    np.testing.assert_allclose(m2.predict_table(lab, X2), p0, rtol=1e-6, atol=1e-8)
    # the default masks nothing
    d = make()
    d.fit(y, lab, X, weights=w)
    assert np.all(np.isfinite(d._X_std))


def test_masked_elm_rows_are_still_derivatives_of_its_energy():
    y, lab, w, X = _table_with_quiet_column()
    m = ELMModel(n_hidden=24, verbose=False, min_feature_std=1e-4)
    m.fit(y, lab, X, weights=w)
    _fd_rows_match(m, lab[:16], X[:16])


def test_atom_centred_mask_of_every_column_leaves_no_force():
    structs = _random_structs(8, 0)
    m = AtomicELMModel(
        n_hidden=12, r0=0.1, eps0=0.01, verbose=False, min_feature_std=1e9
    )
    m.fit(structs, _acsf())
    # constant columns (no neighbour of that species) keep the unit scale
    assert all(np.all(np.isinf(u._X_std) | (u._X_std == 1.0)) for u in m.units)
    assert any(np.any(np.isinf(u._X_std)) for u in m.units)
    s = structs[1]
    E, F, S = MLIPForceField(m)(s["lattice"], s["positions"], s["atom_types"])
    assert np.isfinite(E)
    assert np.all(F == 0.0) and np.all(S == 0.0)


def test_cli_passes_min_feature_std_to_every_model():
    from akitsushima.cli.cmd_mlip import _build_atomic_model, _build_model

    ml = dict(min_feature_std=1e-4)
    for t in ("elm", "qelm", "linear"):
        assert _build_model(t, ml, False).min_feature_std == 1e-4
    assert _build_atomic_model("elm", ml, False).unit_kw["min_feature_std"] == 1e-4
    if HAS_TORCH:
        assert _build_model("nn", ml, False).min_feature_std == 1e-4
        assert _build_atomic_model("nn", ml, False).unit_kw["min_feature_std"] == 1e-4
