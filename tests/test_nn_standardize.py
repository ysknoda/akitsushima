"""NN input standardisation (ADR-0013 addendum, 0.2.0a12): NNModel and
AtomicNNModel take ``standardize: zscore | pca`` with ``pca_eps`` /
``pca_drop``, as the ELM family does.

* With 'pca' the force / stress rows are still derivatives of the energy
  (finite differences on a float64 copy, as in test_mlip_chain_rule).
* The whitening map is the ELM's (pca_whitening) and pca_drop removes
  directions; the network input width follows.
* Save / load keeps the map; files without one load as z-score models.
* 'zscore' stays the default; the bit-for-bit check against the code
  before this change was run once when it was made (ADR-0013 addendum).
* The CLI passes the keys to both NN forms.

torch is imported inside the tests (numba / torch OpenMP note in
test_mlip_chain_rule.py).
"""

import copy
import importlib.util

import numpy as np
import pytest
from test_mlip_atom_centred import _random_structs
from test_mlip_chain_rule import _acsf, _chain_table, _sw_si_structures

from akitsushima.fitting.mlip_engine import MLIPForceField, load_mlip_model
from akitsushima.fitting.models.atomic import AtomicData
from akitsushima.fitting.models.elm import pca_whitening

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


def _correlated_table(D=6, seed=0):
    """_chain_table with two descriptor columns made nearly collinear, so
    the whitening map has a direction pca_drop can remove."""
    y, lab, w, X = _chain_table(D=D, seed=seed)
    X = X.copy()
    X[:, 1] = X[:, 0] + 1e-4 * X[:, 1]
    return y, lab, w, X


def test_pca_nn_rows_are_derivatives_of_its_energy(tmp_path):
    import torch

    from akitsushima.fitting.models.nn import NNModel

    y, lab, w, X = _chain_table()
    m = NNModel(hidden_layers=[16], epochs=20, verbose=False, standardize="pca")
    m.fit(y, lab, X, weights=w)
    assert m._P is not None and m._P.shape == (X.shape[1], X.shape[1])
    net64 = copy.deepcopy(m._net).double()

    def f64(d):
        with torch.no_grad():
            return float(net64(torch.as_tensor(m._normalise(d[None]), dtype=torch.float64)))

    pt = m.predict_table(lab[:16], X[:16])
    d, h = X[0], 1e-5
    for k in range(1, 16):
        fd = 3 * (f64(d + h * X[k]) - f64(d - h * X[k])) / (2 * h)
        assert pt[k] == pytest.approx(fd, rel=1e-4, abs=1e-6), k
    # energy_and_gradient (the MLIP engine's entry) maps back to raw d
    e, g = m.energy_and_gradient(d)
    for k in range(4):
        u = np.zeros_like(d)
        u[k] = 1.0
        fd = (f64(d + h * u) - f64(d - h * u)) / (2 * h)
        assert g[k] == pytest.approx(fd, rel=1e-4, abs=1e-6)
    m.save(str(tmp_path / "nn.pt"))
    m2 = type(m).load(str(tmp_path / "nn.pt"))
    assert m2.standardize == "pca"
    np.testing.assert_allclose(m2._P, m._P)
    np.testing.assert_allclose(m2.predict_table(lab, X), m.predict_table(lab, X))


def test_pca_map_is_the_elm_whitening_and_pca_drop_narrows_the_input():
    from akitsushima.fitting.models.csv_io import (
        build_structure_index,
        structure_grouped_split,
    )
    from akitsushima.fitting.models.nn import NNModel

    y, lab, w, X = _correlated_table()
    full = NNModel(hidden_layers=[8], epochs=2, verbose=False, standardize="pca")
    full.fit(y, lab, X, weights=w)
    cut = NNModel(hidden_layers=[8], epochs=2, verbose=False, standardize="pca",
                  pca_drop=1e-6)
    cut.fit(y, lab, X, weights=w)
    # the statistics are the training structures' E rows (as ELMModel)
    tr, _ = structure_grouped_split(lab, val_fraction=0.2, val_seed=0, caller="t")
    sid, e_row, _ = build_structure_index(lab, None, None, caller="t")
    Xs = X[e_row[np.unique(sid[tr])]]
    np.testing.assert_allclose(full._X_mean, Xs.mean(axis=0))
    P_ref = pca_whitening((Xs - full._X_mean) / full._X_std, 1e-3, 0.0)
    np.testing.assert_allclose(full._P, P_ref)
    assert full._P.shape[1] == X.shape[1]
    # the near-collinear pair leaves one direction below 1e-6 max(Lambda)
    assert cut._P.shape[1] == X.shape[1] - 1
    for m, width in ((full, X.shape[1]), (cut, X.shape[1] - 1)):
        assert m._net.lin.in_features == width          # direct link
        assert m._net.mlp[0].in_features == width


def test_zscore_is_the_default_and_bad_values_are_refused():
    from akitsushima.fitting.models.atomic_nn import AtomicNNModel
    from akitsushima.fitting.models.nn import NNModel

    y, lab, w, X = _chain_table()
    m = NNModel(hidden_layers=[8], epochs=2, verbose=False).fit(y, lab, X, weights=w)
    assert m.standardize == "zscore" and m._P is None
    with pytest.raises(ValueError, match="standardize"):
        NNModel(standardize="whiten")
    with pytest.raises(ValueError, match="standardize"):
        AtomicNNModel(standardize="whiten")


def test_file_without_a_map_loads_as_zscore(tmp_path):
    import torch

    from akitsushima.fitting.models.nn import NNModel

    y, lab, w, X = _chain_table()
    m = NNModel(hidden_layers=[8], epochs=2, verbose=False).fit(y, lab, X, weights=w)
    path = str(tmp_path / "nn.pt")
    m.save(path)
    ck = torch.load(path, weights_only=False)
    for k in ("standardize", "pca_eps", "pca_drop", "X_P"):
        ck.pop(k)
    torch.save(ck, path)
    m2 = NNModel.load(path)
    assert m2.standardize == "zscore" and m2._P is None
    np.testing.assert_allclose(m2.predict_table(lab, X), m.predict_table(lab, X))


def test_pca_atomic_nn_rows_equal_its_engine_and_engine_is_conservative(tmp_path):
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
        standardize="pca",
        pca_drop=1e-8,
    )
    m.fit(structs, ac)
    assert m._X_P is not None and len(m._X_P) == len(m.species)
    for s, net in enumerate(m._nets):
        assert net.P.shape[1] == m._X_P[s].shape[1] <= ac.n_per_atom
    path = str(tmp_path / "m.pt")
    m.save(path)
    m2 = load_mlip_model(path)
    assert type(m2) is AtomicNNModel and m2.unit_kw["standardize"] == "pca"
    T = m._tensors(AtomicData(structs, ac), np.arange(len(structs)))
    with torch.enable_grad():
        p = m._rows_pred(T, [2], False).detach().numpy()
    s = structs[2]
    q = m2._predict_structure(
        ac.atomic_jacobian(s["lattice"], s["positions"], s["atom_types"])
    )
    np.testing.assert_allclose(p, q, atol=2e-5 * max(1.0, np.abs(q).max()))
    from test_mlip_atom_centred import _fd_check

    _fd_check(
        MLIPForceField(m2),
        s["lattice"],
        s["positions"],
        s["atom_types"],
        1e-3,
        5e-3,
        5e-4,
    )


def test_pca_atomic_nn_learns_and_keeps_the_isolated_atom_constraint():
    """Without the constraint, 20 epochs bring the training forces well
    below their spread (z-score 0.61, pca 0.47 eV/A against 1.15).  With
    isolated_atom_energy, eps_s(G = 0) = E_atom holds exactly through the
    fixed whitening layer (x0 is mapped like every other input)."""
    import json

    from test_mlip_atom_centred import _si_acsf

    from akitsushima.fitting.models.atomic_nn import AtomicNNModel

    structs = _sw_si_structures(20, 3)
    ac = _si_acsf()
    kw = dict(hidden_layers=[16], epochs=20, batch_structures=5, r0=0.1, eps0=0.01,
              verbose=False, standardize="pca", w_forces=1.0, deterministic=True)
    m = AtomicNNModel(**kw).fit(structs, ac)
    y, lab, *_ = AtomicData(structs, ac).rows(0.1, 0.01)
    assert m._train_sc["F"]["RMSE"] < 0.7 * np.std(y[lab == "F"])
    c = AtomicNNModel(**kw)
    c.physics_config = json.dumps({"isolated_atom_energy": {"Si": -0.5}})
    c.fit(structs, ac)
    eps, _ = c.atomic_energy_and_gradient(np.zeros((1, ac.n_per_atom)), np.array([0]))
    assert eps[0] == pytest.approx(-0.5, abs=1e-5)


def test_cli_passes_the_standardize_keys_to_both_nn_forms():
    from akitsushima.cli.cmd_mlip import _build_atomic_model, _build_model

    ml = {"standardize": "pca", "pca_eps": 1e-2, "pca_drop": 1e-5}
    tab = _build_model("nn", ml, verbose=False)
    assert (tab.standardize, tab.pca_eps, tab.pca_drop) == ("pca", 1e-2, 1e-5)
    atom = _build_atomic_model("nn", ml, verbose=False)
    assert atom.unit_kw == {"standardize": "pca", "pca_eps": 1e-2, "pca_drop": 1e-5}
    assert _build_model("nn", {}, verbose=False).standardize == "zscore"
    assert _build_atomic_model("nn", {}, verbose=False).unit_kw["standardize"] == "zscore"
