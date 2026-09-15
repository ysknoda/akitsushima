"""Regression tests for solvers/config_bridge.py: the YAML/JSON-config to
force_fn bridge.

Covers S19 (LJ/Morse/BornMayer could never actually be constructed from a
YAML config: an invalid `species=` constructor kwarg, plus -- even past
that crash -- params dict keys in the wrong format for _get_params to
ever match a pair) and F11 (YAML configs were never back-filled with
default values the way JSON configs are).
"""
import numpy as np
import pytest

from akitsushima.engine import BuckinghamPotential, LennardJonesPotential
from akitsushima.fitting.io_utils import save_potential
from akitsushima.solvers.config_bridge import (
    _build_force_fn_from_yaml,
    _load_cfg,
)


# ===========================================================================
# S19: LJ / Morse / BornMayer via YAML config
# ===========================================================================

def test_lj_multi_species_list_form_matches_direct_construction():
    """
    S19 regression: potential.params as a list of {i, j, epsilon, sigma}
    records (required for multi-species systems) must produce a force_fn
    numerically identical to constructing LennardJonesPotential directly
    with the equivalent frozenset-keyed params -- before the fix this
    branch raised TypeError immediately (invalid species= kwarg).
    """
    cfg = {
        "type": "LJ",
        "species": ["Al", "O"],
        "r_cut": 6.0,
        "params": [
            {"i": "Al", "j": "Al", "epsilon": 0.4, "sigma": 2.5},
            {"i": "Al", "j": "O",  "epsilon": 0.1, "sigma": 3.0},
            {"i": "O",  "j": "O",  "epsilon": 0.2, "sigma": 2.8},
        ],
    }
    fn, r_cut, buck, charges = _build_force_fn_from_yaml(cfg)
    assert r_cut == 6.0
    assert buck is None and charges is None

    direct = LennardJonesPotential(
        params={
            frozenset({"Al"}):      {"epsilon": 0.4, "sigma": 2.5},
            frozenset({"Al", "O"}): {"epsilon": 0.1, "sigma": 3.0},
            frozenset({"O"}):       {"epsilon": 0.2, "sigma": 2.8},
        }, r_cut=6.0, shift=True)

    lattice = np.eye(3) * 15.0
    positions = np.array([[5.0, 5.0, 5.0], [7.2, 5.0, 5.0]])
    atom_types = ["Al", "O"]

    e1, f1, s1 = fn(lattice, positions, atom_types)
    e2, f2, s2 = direct.compute(lattice, positions, atom_types)

    assert e1 == pytest.approx(e2, rel=1e-10)
    assert np.allclose(f1, f2, rtol=1e-10)
    assert np.linalg.norm(f1) > 1e-8, "sanity: the pair must actually interact"


def test_lj_single_species_flat_form():
    """
    S19 regression: a flat {epsilon, sigma} dict (no i/j records) must be
    accepted when species: names exactly one species, converted to a
    self-pair frozenset key.
    """
    cfg = {"type": "lj", "species": ["Al"],
           "params": {"epsilon": 0.4, "sigma": 2.5, "r_cut": 6.0}}
    fn, r_cut, _, _ = _build_force_fn_from_yaml(cfg)
    assert r_cut == 6.0

    lattice = np.eye(3) * 15.0
    positions = np.array([[5.0, 5.0, 5.0], [7.5, 5.0, 5.0]])
    e, f, s = fn(lattice, positions, ["Al", "Al"])
    assert np.isfinite(e)
    assert np.linalg.norm(f) > 1e-8


def test_lj_multi_species_requires_list_form():
    """A flat dict with more than one species listed is ambiguous (which
    pair do the parameters belong to?) and must raise a clear error, not
    silently apply to the wrong pair or crash obscurely."""
    cfg = {"type": "LJ", "species": ["Al", "O"],
           "params": {"epsilon": 0.4, "sigma": 2.5}}
    with pytest.raises(ValueError, match="list of"):
        _build_force_fn_from_yaml(cfg)


def test_mie_and_yukawa_via_yaml_work():
    """Phase 2 registry regression: Mie/Yukawa's YAML bridge
    (fitting/registry.py's build_from_yaml functions) must work through
    the exact same solvers/config_bridge.py dispatch as the original
    potentials, having required zero edits to config_bridge.py itself."""
    cfg_mie = {"type": "mie", "species": ["Al"], "n": 14.0, "m": 7.0,
               "params": {"epsilon": 0.4, "sigma": 2.5, "r_cut": 6.0}}
    fn_mie, r_cut_mie, _, _ = _build_force_fn_from_yaml(cfg_mie)
    assert r_cut_mie == 6.0

    cfg_yukawa = {"type": "yukawa", "species": ["Al"], "kappa": 1.2,
                 "params": {"A": -15.0, "r_cut": 6.0}}
    fn_yukawa, r_cut_yukawa, _, _ = _build_force_fn_from_yaml(cfg_yukawa)
    assert r_cut_yukawa == 6.0

    lattice = np.eye(3) * 15.0
    positions = np.array([[5.0, 5.0, 5.0], [7.5, 5.0, 5.0]])
    for fn in (fn_mie, fn_yukawa):
        e, f, s = fn(lattice, positions, ["Al", "Al"])
        assert np.isfinite(e)
        assert np.linalg.norm(f) > 1e-8


def test_morse_and_bornmayer_via_yaml_run_without_crashing(small_cell_ok):
    """S19 regression (Morse/BornMayer branches)."""
    lattice = np.eye(3) * 15.0
    positions = np.array([[5.0, 5.0, 5.0], [7.5, 5.0, 5.0]])

    morse_cfg = {"type": "Morse", "species": ["Al"],
                 "params": {"D_e": 0.3, "alpha": 1.2, "r0": 2.8}, "r_cut": 6.0}
    fn_m, r_cut_m, _, _ = _build_force_fn_from_yaml(morse_cfg)
    e_m, f_m, _ = fn_m(lattice, positions, ["Al", "Al"])
    assert np.isfinite(e_m) and np.linalg.norm(f_m) > 1e-8

    bm_cfg = {"type": "BornMayer", "species": ["Na", "Cl"],
              "params": [{"i": "Na", "j": "Cl", "A": 1000.0, "rho": 0.3},
                        {"i": "Na", "j": "Na", "A": 500.0,  "rho": 0.3},
                        {"i": "Cl", "j": "Cl", "A": 500.0,  "rho": 0.3}],
              "r_cut": 8.0}
    fn_b, r_cut_b, _, _ = _build_force_fn_from_yaml(bm_cfg)
    e_b, f_b, _ = fn_b(lattice, np.array([[5.0, 5.0, 5.0], [7.0, 5.0, 5.0]]),
                       ["Na", "Cl"])
    assert np.isfinite(e_b) and np.linalg.norm(f_b) > 1e-8


def test_sw_tersoff_via_yaml_still_work():
    """Regression guard: the already-correct SW/Tersoff branches must be
    unaffected by the LJ/Morse/BornMayer (S19) changes above."""
    sw_cfg = {"type": "SW", "species": ["Si"],
              "params": {"epsilon": 2.1682, "sigma": 2.0951, "A": 7.049556,
                        "B": 0.6022246, "p": 4, "q": 0, "a": 1.80,
                        "lambda": 21.0, "gamma": 1.20, "costheta0": -0.333333}}
    fn, r_cut, _, _ = _build_force_fn_from_yaml(sw_cfg)
    assert r_cut == pytest.approx(1.80 * 2.0951)

    tersoff_cfg = {"type": "Tersoff", "species": ["Si"],
                   "params": {"A": 1830.8, "B": 471.18, "lambda1": 2.4799,
                             "lambda2": 1.7322, "lambda3": 1.7322,
                             "beta": 1.1e-6, "n": 0.78734, "c": 100390.0,
                             "d": 16.218, "h": -0.59826, "R": 2.85, "D": 0.15}}
    fn2, r_cut2, _, _ = _build_force_fn_from_yaml(tersoff_cfg)
    assert r_cut2 == pytest.approx(2.85 + 0.15)


def test_eam_raises_clear_error_not_typeerror():
    """
    S19 regression: the original called EAMPotential(params=params,
    species=species), but EAMPotential.__init__ takes F_table/rho_table/
    phi_table (tabulated per-species functions), not a flat params dict
    -- raising a confusing TypeError. A clear, actionable ValueError is
    expected instead, since this project has no YAML-level loader for
    tabulated EAM potential files.
    """
    with pytest.raises(ValueError, match="EAM"):
        _build_force_fn_from_yaml(
            {"type": "EAM", "species": ["Al"], "params": {"foo": 1}})


def test_buckingham_legacy_path_loads_json(tmp_path):
    """The legacy Buckingham branch returns (None, r_cut, buck_params,
    charges) for the caller's legacy-kwargs constructor path; confirms
    S13's _load_potential_json is wired in correctly here too."""
    pair = frozenset({"Mg", "O"})
    result = {"params": {"buck": {pair: {"A": 1000.0, "rho": 0.30, "C": 30.0}},
                        "charges": {"Mg": 2.0, "O": -2.0}},
              "loss": 1e-9}
    path = str(tmp_path / "mgo_buck.json")
    save_potential(result, path, potential_type="buckingham", r_cut=10.0,
                  species=["Mg", "O"])

    cfg = {"type": "buckingham", "potential_json": path}
    force_fn, r_cut, buck_params, charges = _build_force_fn_from_yaml(cfg)
    assert force_fn is None
    assert r_cut == 10.0
    assert buck_params[pair]["C"] == pytest.approx(30.0)
    assert charges == {"Mg": 2.0, "O": -2.0}


def test_unknown_potential_type_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown potential type"):
        _build_force_fn_from_yaml({"type": "not_a_real_potential"})


# ===========================================================================
# F11: YAML configs must be back-filled with defaults, like JSON configs
# ===========================================================================

def test_load_cfg_backfills_missing_keys_for_yaml(tmp_path):
    """
    F11 regression: _load_cfg's YAML branch used to return immediately
    from _load_yaml() without ever applying the default-fill loop (which
    only ran for the JSON branch), so a YAML config omitting an optional
    key would KeyError the first time a caller indexed cfg[key] directly,
    while the equivalent JSON config would not.
    """
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("a: 1\nb: 2\n")

    cfg = _load_cfg(str(yaml_path), default={"a": 999, "c": 42, "r_cut": 10.0})
    assert cfg["a"] == 1     # present in file: not overwritten
    assert cfg["b"] == 2     # present in file, not in default: preserved
    assert cfg["c"] == 42    # missing from file: filled from default
    assert cfg["r_cut"] == 10.0   # missing from file: filled from default


def test_load_cfg_json_still_backfills(tmp_path):
    """Sanity check that the pre-existing JSON behaviour is unchanged."""
    import json
    json_path = tmp_path / "cfg.json"
    json_path.write_text(json.dumps({"a": 1}))

    cfg = _load_cfg(str(json_path), default={"a": 999, "c": 42})
    assert cfg["a"] == 1
    assert cfg["c"] == 42
