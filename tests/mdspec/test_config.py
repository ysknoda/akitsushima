"""YAML schema, defaults, path resolution and validation errors (ADR-0007, ADR-0011 A6)."""

import warnings

import pytest

from akitsushima.mdspec import config


def _raw(**over):
    raw = {
        "trajectory": {"files": ["XDATCAR"], "timestep_fs": 1.0, "frame_every": 1},
        "temperature": 300.0,
        "dipole": {"model": "point_charge", "charges": {"Mg": 2.0, "O": -2.0}},
        "epsilon_inf": 1.0,
    }
    for key, val in over.items():
        sec, _, sub = key.partition("__")
        if sub:
            raw.setdefault(sec, {})[sub] = val
        else:
            raw[sec] = val
    return raw


def test_defaults_and_relative_paths(tmp_path):
    raw = _raw()
    raw["trajectory"]["files"] = "run/XDATCAR"  # a string is accepted
    cfg = config.validate(raw, tmp_path)
    assert cfg["trajectory"]["files"] == [str((tmp_path / "run/XDATCAR").resolve())]
    assert cfg["output"]["dir"] == str((tmp_path / "mdspec_output").resolve())
    assert cfg["spectrum"]["estimator"] == "auto" and cfg["spectrum"]["n_blocks"] == 5
    assert cfg["trajectory"]["mode"] == "concatenate" and cfg["masses"] == {}
    assert cfg["modes"] is None and cfg["forces"] is None and cfg["fit"] is None


def test_absolute_paths_kept(tmp_path):
    raw = _raw(trajectory__files=["/abs/XDATCAR"])
    assert config.validate(raw, tmp_path)["trajectory"]["files"] == ["/abs/XDATCAR"]


@pytest.mark.parametrize(
    "over, match",
    [
        ({"spectrum": {"window": "kaiser"}}, "spectrum.window"),
        ({"trajectory__mode": "merge"}, "trajectory.mode"),
        ({"temperature": -5}, "temperature"),
        ({"temperature": "auto"}, "temperature_source"),
        ({"trajectory__timestep_fs": 0}, "timestep_fs"),
        ({"trajectory__frame_every": 0}, "frame_every"),
        ({"trajectory__skip_frames": -1}, "skip_frames"),
        ({"spectrum": {"n_blocks": 1}}, "n_blocks"),
        ({"spectrum": {"max_lag_ps": -1}}, "max_lag_ps"),
        ({"dipole": {"model": "point_charge"}}, "charges"),
        ({"dipole": {"model": "born_tensor", "born": {"outcar": "OUTCAR"}}}, "unit_cell"),
        ({"dipole": {"model": "born_tensor", "born": {}}}, "per_species"),
        ({"dipole": {"model": "external"}}, "external_file"),
        ({"epsilon_inf": {"file": "x"}}, "outcar"),
        ({"fit": {"terms": [{"type": "gauss"}]}}, "fit term"),
        ({"fit": {"terms": [{"type": "debye"}], "use_component": "xy"}}, "use_component"),
        ({"fit": {"model": "gp", "terms": [{"type": "debye"}]}}, "fit.model"),
        ({"fit": {"model": "gervais_piriou", "terms": [{"f_to": 1.0}]}}, "f_lo"),
    ],
)
def test_invalid_inputs(tmp_path, over, match):
    with pytest.raises(config.ConfigError, match=match):
        config.validate(_raw(**over), tmp_path)


def test_external_forces_raw_and_refuses_modes(tmp_path):
    raw = _raw(dipole={"model": "external", "external_file": "m.dat"})
    raw["spectrum"] = {"estimator": "fsum"}
    with pytest.warns(UserWarning, match="estimator set"):
        cfg = config.validate(raw, tmp_path)
    assert cfg["spectrum"]["estimator"] == "raw"
    raw["modes"] = {"file": "m.npz", "unit_cell": "P", "supercell": [1, 1, 1]}
    with pytest.raises(config.ConfigError, match="modes"), pytest.warns(UserWarning):
        config.validate(raw, tmp_path)


def test_epsilon_inf_default_warns(tmp_path):
    raw = _raw()
    del raw["epsilon_inf"]
    with pytest.warns(UserWarning, match="epsilon_inf"):
        assert config.validate(raw, tmp_path)["epsilon_inf"] == 1.0


def test_implicit_modes_from_born_outcar(tmp_path):
    raw = _raw(
        dipole={
            "model": "born_tensor",
            "born": {"outcar": "dfpt/OUTCAR", "unit_cell": "dfpt/POSCAR", "supercell": [2, 2, 2]},
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = config.validate(raw, tmp_path)
    m = cfg["modes"]
    assert m["implicit"] and m["outcar"] == str((tmp_path / "dfpt/OUTCAR").resolve())
    assert m["supercell"] == [2, 2, 2]


def test_masses_are_floats(tmp_path):
    cfg = config.validate(_raw(masses={"Mg": 24}), tmp_path)
    assert cfg["masses"] == {"Mg": 24.0}


# --------------------------------------------------------------------------
# A6: types and shapes (ADR-0011)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "over, match",
    [
        ({"trajectory__timestep_fs": "abc"}, "timestep_fs must be a number"),
        ({"trajectory__frame_every": 1.5}, "frame_every must be an integer"),
        ({"trajectory__files": [1, 2]}, "trajectory.files"),
        ({"trajectory__species": "Mg O"}, None),  # a single string is a one-element list
        ({"temperature": True}, "temperature"),
        ({"masses": {"Mg": -1}}, "masses.Mg"),
        ({"masses": [24.3]}, "masses must be a mapping"),
        ({"dipole": {"model": "point_charge", "charges": {"Mg": "two"}}}, "charges.Mg"),
        ({"epsilon_inf": [1.0, 2.0]}, "epsilon_inf"),
        ({"spectrum": {"n_blocks": 2.5}}, "n_blocks"),
        ({"spectrum": {"freq_max": 0}}, "freq_max"),
        ({"spectrum": {"kk_check": "yes"}}, "kk_check"),
        ({"output": {"plot": "yes"}}, "output.plot"),
        ({"performance": {"chunk_mb": 0}}, "chunk_mb"),
        ({"fit": {"terms": [{"type": "debye"}], "freq_range": [5, 2]}}, "low < high"),
        ({"fit": {"terms": [{"type": "debye"}], "freq_range": [5]}}, "freq_range"),
    ],
)
def test_types_and_shapes(tmp_path, over, match):
    if match is None:
        config.validate(_raw(**over), tmp_path)
        return
    with pytest.raises(config.ConfigError, match=match):
        config.validate(_raw(**over), tmp_path)


@pytest.mark.parametrize(
    "supercell, ok",
    [
        ([2, 2, 1], True),
        ([[1, 1, 0], [-1, 1, 0], [0, 0, 2]], True),
        ([2, 2], False),
        ([2, 0, 1], False),
        ([2.5, 2, 1], False),
        ([[1, 1, 0], [1, 1, 0], [0, 0, 1]], False),
        ("2 2 1", False),
    ],
)
def test_supercell_shapes(tmp_path, supercell, ok):
    born = {"outcar": "OUTCAR", "unit_cell": "POSCAR", "supercell": supercell}
    raw = _raw(dipole={"model": "born_tensor", "born": born})
    if ok:
        assert config.validate(raw, tmp_path)["dipole"]["born"]["supercell"] == supercell
    else:
        with pytest.raises(config.ConfigError, match="supercell"):
            config.validate(raw, tmp_path)


def test_numeric_strings_accepted(tmp_path):
    # PyYAML reads 1e-6 (no dot) as a string
    cfg = config.validate(
        _raw(
            dipole={
                "model": "point_charge",
                "charges": {"Mg": 2, "O": -2},
                "neutrality_tol": "1e-6",
            }
        ),
        tmp_path,
    )
    assert cfg["dipole"]["neutrality_tol"] == 1e-6
    assert cfg["trajectory"]["frame_every"] == 1 and isinstance(
        cfg["trajectory"]["frame_every"], int
    )
