"""Output files: text tables and the JSON summary (ADR-0007)."""

import json

import numpy as np

from akitsushima.mdspec import output


def _res(nf=6):
    rng = np.random.default_rng(0)
    t = lambda: rng.normal(size=(nf, 3, 3))
    return {
        "eps_re": t(),
        "eps_im": t(),
        "eps_re_err": np.abs(t()),
        "eps_im_err": np.abs(t()),
        "acf": t(),
        "dt_s": 2e-15,
    }


def test_write_eps_columns(tmp_path):
    res = _res()
    freq = np.linspace(0, 5, 6)
    output.write_eps(tmp_path / "e.dat", freq, "THz", res)
    head = (tmp_path / "e.dat").read_text().splitlines()[0].split()[1:]
    assert head[:3] == ["freq_THz", "eps_re_xx", "eps_im_xx"]
    assert head[-4:] == ["eps_re_iso", "eps_im_iso", "err_re_iso", "err_im_iso"]
    d = np.loadtxt(tmp_path / "e.dat")
    assert d.shape == (6, 11)
    assert np.allclose(d[:, 5], res["eps_re"][:, 2, 2])
    assert np.allclose(d[:, 7], np.trace(res["eps_re"], axis1=1, axis2=2) / 3)


def test_write_acf_lags(tmp_path):
    res = _res()
    output.write_acf(tmp_path / "a.dat", res)
    d = np.loadtxt(tmp_path / "a.dat")
    assert np.allclose(d[:, 0], np.arange(6) * 2e-3)  # ps
    assert np.allclose(d[:, 2], res["acf"][:, 1, 1])


def test_json_numpy_types(tmp_path):
    summary = {
        "a": np.float64(1.5),
        "b": np.arange(3),
        "c": [np.int64(2), (np.eye(2),)],
        "d": {"e": np.float32(0.25)},
    }
    output.write_json(tmp_path / "s.json", summary)
    got = json.loads((tmp_path / "s.json").read_text())
    assert got == {"a": 1.5, "b": [0, 1, 2], "c": [2, [[[1.0, 0.0], [0.0, 1.0]]]], "d": {"e": 0.25}}


def test_json_nan_becomes_null(tmp_path):
    summary = {
        "a": float("nan"),
        "b": np.array([1.0, np.inf]),
        "c": np.float64(-np.inf),
        "d": np.bool_(True),
    }
    output.write_json(tmp_path / "s.json", summary)
    text = (tmp_path / "s.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"a": None, "b": [1.0, None], "c": None, "d": True}
