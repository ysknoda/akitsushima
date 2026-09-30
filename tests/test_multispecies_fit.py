"""ADR-0015: multi-species classical fitting -- Tersoff (1989 mixing and
(i, j, k) records) and the analytic Zhou et al. (2004) alloy EAM.

Seconds-long counterparts of example/fit/{SiC_tersoff,AlN_tersoff,NiAl_eam}.
"""
import copy
import itertools
import json
import math

import numpy as np
import pytest

from akitsushima.engine import PotentialEngine, TersoffPotential
from akitsushima.engine.eam_zhou import (
    continuous, density, embedding, pair_aa, read_setfl, write_setfl, zhou_eam)
from akitsushima.engine.published_parameters import (
    TERSOFF_1989, ZHOU_2004, ZHOU_2004_KEYS)
from akitsushima.engine.tersoff import tersoff1989_to_triplets
from akitsushima.fitting import (
    MultiSpeciesTersoffParameterSet, Tersoff1989ParameterSet, ZhouEAMParameterSet)

FCC = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]], float)


def _cell(a, basis, types, n):
    cells = np.array(list(itertools.product(range(n), repeat=3)))
    frac = np.concatenate([(np.asarray(basis, float) + c) / n for c in cells])
    lat = np.eye(3) * a * n
    return lat, frac @ lat, list(types) * len(cells)


def _zincblende(a, x="Si", y="C", n=2):
    return _cell(a, np.vstack([FCC, FCC + .25]), [x] * 4 + [y] * 4, n)


def _fd4(fun, h=1e-3):
    return (fun(-2 * h) - 8 * fun(-h) + 8 * fun(h) - fun(2 * h)) / (12 * h)


def _rattled(cell, seed, amp=0.06, shear=0.02):
    lat, pos, types = cell
    rng = np.random.default_rng(seed)
    e = rng.uniform(-shear, shear, (3, 3))
    t = np.eye(3) + 0.5 * (e + e.T)
    return lat @ t, pos @ t + rng.normal(0, amp, pos.shape), types


# ---------------------------------------------------------------------------
#  Published tables
# ---------------------------------------------------------------------------

def test_published_tables_are_complete():
    """All rows of both source tables are present (CLAUDE.md): Tersoff (1989)
    Table I 3 x 11 + 2 chi = 35 values here (41 printed numbers include the
    6 R, S of the 3 elements, counted in the 11); Zhou (2004) Table III
    16 x 20 = 320."""
    assert sorted(TERSOFF_1989["elements"]) == ["C", "Ge", "Si"]
    assert all(len(v) == 11 for v in TERSOFF_1989["elements"].values())
    assert TERSOFF_1989["chi"] == {"C-Si": 0.9776, "Si-Ge": 1.00061}
    assert len(ZHOU_2004) == 16
    assert all(tuple(v) == ZHOU_2004_KEYS for v in ZHOU_2004.values())
    assert sum(len(v) for v in ZHOU_2004.values()) == 320


# ---------------------------------------------------------------------------
#  Tersoff (1989) mixing form
# ---------------------------------------------------------------------------

def test_tersoff1989_expansion_matches_eq_1e_by_hand():
    el, chi = TERSOFF_1989["elements"], TERSOFF_1989["chi"]
    t = tersoff1989_to_triplets(el, chi, ["Si", "C"])
    si, c = el["Si"], el["C"]
    sic = t[("Si", "C", "C")]
    assert sic["A"] == pytest.approx(math.sqrt(1830.8 * 1393.6), rel=1e-14)
    assert sic["B"] == pytest.approx(0.9776 * math.sqrt(471.18 * 346.7), rel=1e-14)
    assert sic["lambda1"] == pytest.approx((2.4799 + 3.4879) / 2, rel=1e-14)
    assert sic["lambda2"] == pytest.approx((1.7322 + 2.2119) / 2, rel=1e-14)
    r_ij, s_ij = math.sqrt(2.7 * 1.8), math.sqrt(3.0 * 2.1)
    assert sic["R"] == pytest.approx((r_ij + s_ij) / 2, rel=1e-14)
    assert sic["D"] == pytest.approx((s_ij - r_ij) / 2, rel=1e-14)
    # bond order and angular terms of the centre (Si), lambda3 = 0, omega = 1
    assert (sic["beta"], sic["n"]) == (si["beta"], si["n"])
    for trip in (("Si", "C", "C"), ("Si", "C", "Si"), ("Si", "Si", "C")):
        assert (t[trip]["c"], t[trip]["d"], t[trip]["h"]) == (si["c"], si["d"], si["h"])
        assert t[trip]["lambda3"] == 0.0 and t[trip]["gamma"] == 1.0
    cs = t[("C", "Si", "Si")]
    assert (cs["beta"], cs["c"]) == (c["beta"], c["c"])
    assert cs["B"] == sic["B"] and cs["A"] == sic["A"]
    assert t[("Si", "Si", "Si")]["B"] == pytest.approx(471.18, rel=1e-14)
    with pytest.raises(ValueError, match="chi for the"):
        tersoff1989_to_triplets(el, {}, ["C", "Ge"])


@pytest.mark.parametrize("el, a0, e_coh, tol", [
    ("C", 3.566, 7.37, 0.005),     # Tersoff 1989 text; d_C erratum
    ("Si", 5.432, 4.63, 0.005),
])
def test_tersoff1989_elements_reproduce_quoted_cohesive_energies(el, a0, e_coh, tol):
    from scipy.optimize import minimize_scalar
    pot = TersoffPotential(tersoff1989_to_triplets(TERSOFF_1989["elements"], {}, [el]), [el])
    e = lambda a: pot.compute(*_cell(a, np.vstack([FCC, FCC + .25]), [el] * 8, 2))[0] / 64
    r = minimize_scalar(e, bounds=(0.95 * a0, 1.05 * a0), method="bounded",
                        options={"xatol": 1e-7})
    assert abs(-r.fun - e_coh) < tol
    assert abs(r.x - a0) < 0.001


def test_sic_reproduces_tersoff1989_table_ii():
    """3C-SiC: a = 4.32 A (Table II), E_coh = 6.165 eV/atom (text) within the
    propagated rounding of Table I (+-0.0058, example/fit/SiC_tersoff/
    step0_rounding.py), B = 2.2 Mbar to its printed precision."""
    from scipy.optimize import minimize_scalar
    pot = TersoffPotential(tersoff1989_to_triplets(
        TERSOFF_1989["elements"], TERSOFF_1989["chi"], ["Si", "C"]), ["Si", "C"])
    e = lambda a: pot.compute(*_zincblende(a))[0] / 64
    r = minimize_scalar(e, bounds=(4.2, 4.45), method="bounded", options={"xatol": 1e-8})
    a = r.x
    assert abs(a - 4.32) < 0.005
    assert abs(-r.fun - 6.165) < 0.0058
    h = 1e-3
    ee = [e(a * (1 + k * h)) for k in (-1, 0, 1)]
    B = a * a * (ee[0] - 2 * ee[1] + ee[2]) / (a * h) ** 2 / (9 * a ** 3 / 8) * 160.21766208
    assert abs(B - 220.0) <= 5.0


def test_tersoff1989_forces_stress_are_energy_derivatives():
    pot = TersoffPotential(tersoff1989_to_triplets(
        TERSOFF_1989["elements"], TERSOFF_1989["chi"], ["Si", "C"]), ["Si", "C"])
    lat, pos, types = _rattled(_zincblende(4.32), 3)
    _, f, s = pot.compute(lat, pos, types)
    for i in (0, 33):
        for d in range(3):
            def e(x):
                p = pos.copy(); p[i, d] += x
                return pot.compute(lat, p, types)[0]
            # h = 5e-4: at 1e-3 the fourth-order truncation of stiff C-C
            # bonds is 1.3e-8 (it falls 16x per halving, to 7.7e-10 here)
            assert abs(f[i, d] + _fd4(e, 5e-4)) < 5e-9
    frac, vol = pos @ np.linalg.inv(lat), abs(np.linalg.det(lat))
    for a_, b_ in ((0, 0), (1, 2)):
        def e(x):
            t = np.eye(3); t[a_, b_] += x
            return pot.compute(lat @ t, frac @ (lat @ t), types)[0]
        # h = 1e-4: rattled Si-Si second neighbours (2.78 A) sit near the
        # switch onset R - D = 2.7 A, where fc'' jumps; larger strain steps
        # cross it (1e-6 to 2e-5 at h >= 2.5e-4, 1e-11 at 1.25e-4)
        assert abs(s[a_, b_] + _fd4(e, 1e-4) / vol) < 1e-9


def test_tersoff1989_parameter_set_round_trip_and_chi_only():
    el, chi = TERSOFF_1989["elements"], TERSOFF_1989["chi"]
    ps = Tersoff1989ParameterSet(el, chi, ["C", "Si"])
    assert len(ps.x0) == 15          # beta, n fixed by default (Q17)
    assert not any(n.endswith((':beta', ':n')) for n in ps.variable_names)
    dec = ps.decode(ps.x0)
    ref = tersoff1989_to_triplets(el, chi, ["C", "Si"])
    for t, p in ref.items():
        for k, v in p.items():
            assert dec["tersoff"][t][k] == pytest.approx(v, rel=1e-12)
    only = Tersoff1989ParameterSet(el, chi, ["C", "Si"], fit_keys=["chi"])
    assert only.variable_names == ["chi:C-Si"]
    with pytest.raises(ValueError, match="chi for C-Ge"):
        Tersoff1989ParameterSet(el, chi, ["C", "Ge"])


# ---------------------------------------------------------------------------
#  Tersoff (i, j, k) records
# ---------------------------------------------------------------------------

def _aln_records():
    from akitsushima.fitting.registry import _TERSOFF_ALN_RECORDS
    return copy.deepcopy(_TERSOFF_ALN_RECORDS)


def test_record_parameter_set_ties_pairs_and_centres():
    ps = MultiSpeciesTersoffParameterSet(_aln_records(), ["N", "Al"])
    names = ps.variable_names
    assert len(names) == 10          # beta, n fixed by default (Q17)
    assert [n for n in names if n.startswith("Al-N:")] == [
        "Al-N:A", "Al-N:B", "Al-N:lambda1", "Al-N:lambda2"]
    x = ps.x0.copy()
    x[names.index("Al-N:B")] += 0.1
    d = ps.decode(x)["tersoff"]
    assert d[("Al", "N", "N")]["B"] == d[("N", "Al", "Al")]["B"] != 453.228512
    tied = MultiSpeciesTersoffParameterSet(_aln_records(), ["N", "Al"], tie="centre")
    assert "Al;*:c" in tied.variable_names
    bad = _aln_records()
    bad[1]["A"] *= 1.01
    with pytest.raises(ValueError, match="disagree on 'A'"):
        MultiSpeciesTersoffParameterSet(bad, ["N", "Al"])
    with pytest.raises(ValueError, match="no partner"):
        MultiSpeciesTersoffParameterSet(_aln_records()[:1], ["N", "Al"])


def test_flat_tersoff_is_refused_for_two_species_and_kept_for_one():
    from akitsushima.fitting import registry
    spec = registry.get("tersoff")
    flat = dict(initial_tersoff=dict(spec.default_cfg["initial_tersoff"]))
    with pytest.raises(ValueError, match="single flat parameter set"):
        spec.build_param_set(flat, ["C", "Si"], [], 3.0)
    from akitsushima.fitting import TersoffParameterSet
    assert isinstance(spec.build_param_set(dict(flat, species=["C"]), ["C"], [], 2.1),
                      TersoffParameterSet)


# ---------------------------------------------------------------------------
#  Zhou et al. (2004) EAM
# ---------------------------------------------------------------------------

def test_zhou_functions_equal_eqs_a3_to_a8():
    p = continuous(ZHOU_2004["Ni"])
    r = np.array([2.1, 2.6, 3.3, 4.7])
    x = r / p["r_e"]
    phi = (p["A"] * np.exp(-p["alpha"] * (x - 1)) / (1 + (x - p["kappa"]) ** 20)
           - p["B"] * np.exp(-p["beta"] * (x - 1)) / (1 + (x - p["lambda"]) ** 20))
    f = p["f_e"] * np.exp(-p["beta"] * (x - 1)) / (1 + (x - p["lambda"]) ** 20)
    assert np.allclose(pair_aa(r, p)[0], phi, rtol=1e-12, atol=0)
    assert np.allclose(density(r, p)[0], f, rtol=1e-12, atol=0)
    rho = np.array([0.5, 1.0, 1.3]) * p["rho_e"]
    rn, re, rs = 0.85 * p["rho_e"], p["rho_e"], p["rho_s"]
    expect = [sum(p[f"Fn{i}"] * (rho[0] / rn - 1) ** i for i in range(4)),
              sum(p[f"F{i}"] * (rho[1] / re - 1) ** i for i in range(4)),
              p["F_e"] * (1 - p["eta"] * math.log(rho[2] / rs)) * (rho[2] / rs) ** p["eta"]]
    assert np.allclose(embedding(rho, p)[0], expect, rtol=1e-12, atol=0)


@pytest.mark.parametrize("el", ["Ni", "Al", "Cu"])
def test_zhou_embedding_continuous_and_matches_printed(el):
    """Value and slope match at rho_n and rho_0 (1e-12); the re-derived Fn0,
    Fn1, F0, F_e equal the printed ones to 3e-5 (F0 is printed to 2
    decimals, so to its rounding: 5e-3)."""
    p = continuous(ZHOU_2004[el])
    for rho in (0.85 * p["rho_e"], 1.15 * p["rho_e"]):
        lo, hi = embedding(np.array([rho * (1 - 1e-13), rho]), p)
        assert abs(lo[0] - lo[1]) < 1e-11 and abs(hi[0] - hi[1]) < 1e-10
    for k, tol in (("Fn0", 3e-5), ("Fn1", 3e-5), ("F_e", 3e-5), ("F0", 5e-3)):
        assert abs(p[k] - ZHOU_2004[el][k]) < tol, k


def test_zhou_al_erratum_gives_equilibrium_at_r_e():
    """Zhou (2004) defines r_e as the equilibrium nearest-neighbour spacing.
    With A = 0.314873 (ZHOU_2004_ERRATA; printed 0.134873) fcc Al has a0 =
    sqrt(2) r_e to 1e-4 A."""
    from scipy.optimize import minimize_scalar
    p = ZHOU_2004["Al"]
    assert p["A"] == 0.314873
    pot = zhou_eam({"Al": p}, ["Al"])
    e = lambda a: pot.compute(*_cell(a, FCC, ["Al"] * 4, 4))[0]
    a0 = minimize_scalar(e, bounds=(3.9, 4.2), method="bounded", options={"xatol": 1e-7}).x
    assert abs(a0 - math.sqrt(2) * p["r_e"]) < 1e-4


def _ni3al(a=3.646, n=4):
    return _cell(a, FCC, ["Al", "Ni", "Ni", "Ni"], n)


def test_zhou_eam_forces_stress_are_energy_derivatives():
    pot = zhou_eam({s: ZHOU_2004[s] for s in ("Al", "Ni")}, ["Al", "Ni"])
    lat, pos, types = _rattled(_ni3al(), 5, amp=0.08)
    _, f, s = pot.compute(lat, pos, types)
    assert np.abs(f.sum(axis=0)).max() < 1e-12
    for i in (0, 77):
        for d in range(3):
            def e(x):
                p = pos.copy(); p[i, d] += x
                return pot.compute(lat, p, types)[0]
            assert abs(f[i, d] + _fd4(e)) < 2e-9
    frac, vol = pos @ np.linalg.inv(lat), abs(np.linalg.det(lat))
    for a_, b_ in ((0, 0), (2, 1)):
        def e(x):
            t = np.eye(3); t[a_, b_] += x
            return pot.compute(lat @ t, frac @ (lat @ t), types)[0]
        assert abs(s[a_, b_] + _fd4(e) / vol) < 1e-9


def test_zhou_cutoff_switch_changes_lattice_energy_below_1e5():
    """The C2 switch over the last 0.5 A before r_c (1e-6 decay criterion)
    changes the L1_2 Ni3Al energy by < 1e-5 eV/atom against a 3 A longer
    cutoff (ADR-0015, Q10)."""
    el = {s: ZHOU_2004[s] for s in ("Al", "Ni")}
    pot = zhou_eam(el, ["Al", "Ni"])
    big = zhou_eam(el, ["Al", "Ni"], r_cut=pot.r_cut + 3.0)
    lat, pos, types = _ni3al()
    e1 = pot.compute(lat, pos, types)[0] / len(types)
    lat2, pos2, types2 = _ni3al(n=6)
    e2 = big.compute(lat2, pos2, types2)[0] / len(types2)
    assert abs(e1 - e2) < 1e-5


def test_setfl_round_trip(tmp_path):
    pot = zhou_eam({s: ZHOU_2004[s] for s in ("Al", "Ni")}, ["Al", "Ni"])
    path = tmp_path / "NiAl.eam.alloy"
    write_setfl(str(path), pot, ["Al", "Ni"])
    tab = read_setfl(str(path))
    assert tab.species == ["Al", "Ni"] and tab.r_cut == pytest.approx(pot.r_cut)
    lat, pos, types = _rattled(_ni3al(), 6, amp=0.08)
    a, b = pot.compute(lat, pos, types), tab.compute(lat, pos, types)
    # Cubic-spline interpolation on the 5000-point grids (as LAMMPS does):
    # measured 5e-13 eV/atom, forces 2.5e-6 eV/A at worst (near the knee of
    # the 20th-power denominators), stress 1.3e-8 eV/A^3 (2e-6 GPa).  A format error
    # (wrong order, r*phi vs phi, grid offset) is of order 0.1.
    assert abs(a[0] - b[0]) / len(types) < 1e-10
    assert np.abs(a[1] - b[1]).max() < 1e-5
    assert np.abs(a[2] - b[2]).max() < 1e-7


def test_zhou_parameter_set_derives_continuity_and_refuses_derived_keys():
    ps = ZhouEAMParameterSet({s: ZHOU_2004[s] for s in ("Al", "Ni")}, ["Al", "Ni"])
    assert len(ps.x0) == 18 and ps.r_cut == pytest.approx(6.6904, abs=1e-3)
    x = ps.x0.copy()
    x[ps.variable_names.index("Ni:F2")] *= 1.1
    p = ps.decode(x)["eam"]["elements"]["Ni"]
    assert p == continuous(p)
    with pytest.raises(ValueError, match="continuity"):
        ZhouEAMParameterSet({s: ZHOU_2004[s] for s in ("Al", "Ni")}, ["Al", "Ni"],
                            fit_keys=["F0"])
    chi = ZhouEAMParameterSet({s: ZHOU_2004[s] for s in ("Al", "Ni")}, ["Al", "Ni"],
                              fit_keys=["chi"])
    assert chi.variable_names == ["chi:Al-Ni"]


# ---------------------------------------------------------------------------
#  I/O, YAML route and CLI
# ---------------------------------------------------------------------------

def test_fitted_json_reloads_to_the_same_potential(tmp_path):
    from akitsushima.fitting.io_utils import load_potential, save_potential
    cases = [
        ("tersoff", Tersoff1989ParameterSet(TERSOFF_1989["elements"], TERSOFF_1989["chi"], ["C", "Si"]),
         _rattled(_zincblende(4.32), 1)),
        ("tersoff", MultiSpeciesTersoffParameterSet(_aln_records(), ["N", "Al"]),
         _rattled(_zincblende(4.78, "Al", "N"), 2)),
        ("eam", ZhouEAMParameterSet({s: ZHOU_2004[s] for s in ("Al", "Ni")}, ["Al", "Ni"],
                                    chi={"Al-Ni": 1.07}),
         _rattled(_ni3al(), 3)),
    ]
    for k, (pt, ps, cell) in enumerate(cases):
        params = ps.decode(ps.x0)
        path = tmp_path / f"p{k}.json"
        save_potential({"params": params, "loss": 0.0}, str(path), potential_type=pt,
                       r_cut=5.0, species=ps.species)
        engine, _ = load_potential(str(path))
        a = ps.build_engine(params).compute(*cell)
        b = engine.compute(*cell)
        assert a[0] == pytest.approx(b[0], rel=1e-13)
        assert np.allclose(a[1], b[1], rtol=0, atol=1e-11)
    saved = json.load(open(tmp_path / "p0.json"))
    assert saved["tersoff1989"]["chi"] == {"C-Si": 0.9776}


def test_yaml_route_builds_mixing_tersoff_and_eam(tmp_path):
    from akitsushima.fitting import registry
    t = registry.get_by_yaml_type("tersoff").build_from_yaml(
        {"species": ["Si", "C"],
         "params": {"mixing": "tersoff1989", "elements": TERSOFF_1989["elements"],
                    "chi": TERSOFF_1989["chi"]}})
    ref = TersoffPotential(tersoff1989_to_triplets(
        TERSOFF_1989["elements"], TERSOFF_1989["chi"], ["Si", "C"]), ["Si", "C"])
    cell = _rattled(_zincblende(4.32), 4)
    assert t[0](*cell)[0] == pytest.approx(ref.compute(*cell)[0], rel=1e-13)
    assert t[1] == pytest.approx(math.sqrt(3.0 * 2.1) + 0.0, abs=0.6)
    eam_fn, rc = registry.get_by_yaml_type("eam").build_from_yaml(
        {"form": "zhou2004", "elements": ["Al", "Ni"]})
    pot = zhou_eam({s: ZHOU_2004[s] for s in ("Al", "Ni")}, ["Al", "Ni"])
    cell = _rattled(_ni3al(), 5)
    assert eam_fn(*cell)[0] == pytest.approx(pot.compute(*cell)[0], rel=1e-13)
    write_setfl(str(tmp_path / "x.eam.alloy"), pot, ["Al", "Ni"])
    tab_fn, rc2 = registry.get_by_yaml_type("eam").build_from_yaml(
        {"setfl": str(tmp_path / "x.eam.alloy")})
    assert tab_fn(*cell)[0] == pytest.approx(pot.compute(*cell)[0], rel=1e-10)
    with pytest.raises(ValueError, match="setfl"):
        registry.get_by_yaml_type("eam").build_from_yaml({"params": {"a": 1}})


def _write_training(root, force_fn, cells):
    from akitsushima.io import write_poscar
    rows = ["target,label,weight,struct_idx"]
    for k, (lat, pos, types) in enumerate(cells):
        order = sorted(range(len(types)), key=lambda i: types[i])
        pos, types = pos[order], [types[i] for i in order]
        E, F, S = force_fn(lat, pos, types)
        (root / str(k)).mkdir(parents=True)
        write_poscar(str(root / str(k) / "POSCAR"), lat, pos, types)
        rows.append(f"{E / len(types):.14f},E,1.0,{k}")
        rows += [f"{f:.14f},F,1.0,{k}" for f in F.ravel()]
        rows += [f"{v:.14e},S,1.0,{k}" for v in
                 (S[0, 0], S[1, 1], S[2, 2], S[1, 2], S[0, 2], S[0, 1])]
    (root / "structures.csv").write_text("\n".join(rows) + "\n")


@pytest.mark.parametrize("potential, template", [
    ("tersoff", "tersoff1989"), ("tersoff", "records"), ("eam", None)])
def test_cli_template_runs_and_recovers_its_own_potential(tmp_path, potential, template):
    """--write-config (and --template) writes a config that `fit` runs as
    written; on data labelled by the template's own potential, a fit of one
    variable (chi, or B of the Al-N pair) started 5 % away returns to it."""
    import yaml
    from akitsushima.cli.cmd_fit import cmd_fit
    from akitsushima.cli.parser import build_parser
    from akitsushima.fitting import registry

    def cli_main(argv):
        return cmd_fit(build_parser().parse_args(argv))
    cfg_path = tmp_path / "cfg.json"
    argv = ["fit", "-p", potential, "--write-config", str(cfg_path)]
    if template:
        argv += ["--template", template]
    assert cli_main(argv) == 0
    cfg = json.load(open(cfg_path))
    spec = registry.get(potential)
    if potential == "eam":
        species, cell = ["Al", "Ni"], _rattled(_ni3al(), 7, amp=0.05, shear=0.01)
        key, tv = "chi", "chi:Al-Ni"
    elif template == "records":
        species, cell = ["Al", "N"], _rattled(_zincblende(4.78, "Al", "N"), 7, amp=0.05, shear=0.01)
        key, tv = "B", "Al-N:B"
    else:
        species, cell = ["C", "Si"], _rattled(_zincblende(4.32), 7, amp=0.05, shear=0.01)
        key, tv = "chi", "chi:C-Si"
    ps = spec.build_param_set(dict(cfg, fit_keys=[key]), species, [], cfg["r_cut"])
    truth = ps.decode(ps.x0)
    _write_training(tmp_path / "train", lambda *c: ps.build_engine(truth).compute(*c), [cell])
    x = ps.x0.copy()
    x[0] += math.log(1.05)
    start = ps.decode(x)
    if potential == "eam":
        cfg["initial_eam"]["chi"] = start["eam"]["chi"]
    elif template == "records":
        cfg["initial_tersoff"] = [dict(i=t[0], j=t[1], k=t[2], **p)
                                  for t, p in start["tersoff"].items()]
    else:
        cfg["initial_tersoff"]["chi"] = start["tersoff1989"]["chi"]
    cfg.update(training_csv=str(tmp_path / "train" / "structures.csv"),
               output_dir=str(tmp_path / "out"), fit_keys=[key], val_fraction=0.0)
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(cfg))
    assert cli_main(["fit", "-p", potential, "--config", str(tmp_path / "run.yaml"), "-q"]) == 0
    out = json.load(open(tmp_path / "out" / "fitted_potential.json"))
    if potential == "eam":
        got = out["params"]["chi"]["Al-Ni"]
        assert (tmp_path / "out" / "fitted.eam.alloy").exists()
    elif template == "records":
        got = [r["B"] for r in out["params"] if (r["i"], r["j"]) == ("Al", "N")][0]
    else:
        got = out["tersoff1989"]["chi"]["C-Si"]
    want = math.exp(ps.x0[0]) if key in ps._LOG_KEYS else ps.x0[0]
    assert got == pytest.approx(want, rel=1e-6)
    assert cli_main(["fit", "-p", "tersoff", "--write-config", str(tmp_path / "z.json"),
                     "--template", "nope"]) == 1


def test_fit_restarts_continue_from_the_best_point_and_stop_early(tmp_path):
    """ADR-0015: `restarts` reruns the local search from the best point (a
    rotated simplex for Nelder-Mead); with a 10-iteration cap per run the
    loss keeps falling across restarts, the history is non-increasing, and
    restarts=0 is the former single run."""
    from akitsushima.fitting.driver import fit_potential
    el, chi = TERSOFF_1989["elements"], TERSOFF_1989["chi"]
    truth = Tersoff1989ParameterSet(el, chi, ["C", "Si"], fit_keys=["chi", "A"])
    cell = _rattled(_zincblende(4.32), 8, amp=0.05, shear=0.01)
    E, F, S = truth.build_engine(truth.decode(truth.x0)).compute(*cell)
    struct = {"lattice": cell[0], "positions": cell[1], "atom_types": cell[2],
              "natoms": 64, "ref_energy": E / 64, "ref_forces": F, "ref_stress": S}
    start = copy.deepcopy(el)
    start["Si"]["A"] *= 1.03
    ps = Tersoff1989ParameterSet(start, {"C-Si": 0.95}, ["C", "Si"], fit_keys=["chi", "A"])
    kw = dict(r_cut=3.0, weights={"energy": 1, "forces": 1, "stress": 1, "r0": .1, "eps0": .01},
              val_fraction=0.0, verbose=False, output_dir=str(tmp_path), local_maxiter=10)
    one = fit_potential(ps, [struct], restarts=0, **kw)
    many = fit_potential(ps, [struct], restarts=4, restart_rtol=1e-12, **kw)
    assert len(one["loss_history"]) == 1
    h = many["loss_history"]
    assert len(h) == 5 and all(b <= a for a, b in zip(h, h[1:]))
    assert h[-1] < 0.5 * one["loss"]
    with pytest.raises(ValueError, match="restarts"):
        fit_potential(ps, [struct], restarts=-1, **kw)


def test_rotated_restart_simplex_is_nondegenerate_and_within_bounds():
    from akitsushima.fitting.driver import _rotated_simplex
    x = np.array([1.0, -2.0, 0.0, 5.0])
    b = [(-10, 10), (-10, 10), (-1, 1), (4.9, 5.1)]
    sim = _rotated_simplex(x, b, np.random.default_rng(0))
    assert sim.shape == (5, 4) and np.allclose(sim[0], x)
    assert np.linalg.matrix_rank(sim[1:] - sim[0]) == 4
    assert all(lo <= v <= hi for row in sim for v, (lo, hi) in zip(row, b))
    other = _rotated_simplex(x, b, np.random.default_rng(1))
    assert not np.allclose(sim, other)


def test_fit_time_limit_keeps_the_best_point_so_far(tmp_path):
    """ADR-0015: time_limit_hours stops the search at the next evaluation
    past the budget and returns the best point evaluated so far."""
    from akitsushima.fitting.driver import fit_potential
    el, chi = TERSOFF_1989["elements"], TERSOFF_1989["chi"]
    truth = Tersoff1989ParameterSet(el, chi, ["C", "Si"], fit_keys=["chi"])
    cell = _rattled(_zincblende(4.32), 9, amp=0.05, shear=0.01)
    E, F, S = truth.build_engine(truth.decode(truth.x0)).compute(*cell)
    struct = {"lattice": cell[0], "positions": cell[1], "atom_types": cell[2],
              "natoms": 64, "ref_energy": E / 64, "ref_forces": F, "ref_stress": S}
    ps = Tersoff1989ParameterSet(el, {"C-Si": 0.95}, ["C", "Si"], fit_keys=["chi"])
    kw = dict(r_cut=3.0, weights={"energy": 1, "forces": 1, "stress": 1, "r0": .1, "eps0": .01},
              val_fraction=0.0, verbose=False, output_dir=str(tmp_path))
    free = fit_potential(ps, [struct], **kw)
    cut = fit_potential(ps, [struct], time_limit_hours=1e-7, restarts=3, **kw)
    assert cut["time_limited"] and not free["time_limited"]
    assert cut["loss"] <= fit_potential(ps, [struct], time_limit_hours=0.0, **kw)["loss"]
    assert cut["loss"] >= free["loss"]
    assert np.isfinite(cut["loss"]) and "tersoff" in cut["params"]


def _eam_struct(seed=11):
    el = {s: ZHOU_2004[s] for s in ("Al", "Ni")}
    truth = ZhouEAMParameterSet(el, ["Al", "Ni"])
    cell = _rattled(_ni3al(), seed, amp=0.05, shear=0.01)
    E, F, S = truth.build_engine(truth.decode(truth.x0)).compute(*cell)
    return truth, {"lattice": cell[0], "positions": cell[1], "atom_types": cell[2],
                   "natoms": len(cell[2]), "ref_energy": E / len(cell[2]),
                   "ref_forces": F, "ref_stress": S}


def test_residual_vector_squares_sum_to_the_loss():
    from akitsushima.fitting.loss import generic_loss_function, generic_residuals, residual_length
    truth, struct = _eam_struct()
    w = {"energy": 1.0, "forces": 1.0, "stress": 1.0, "r0": 0.1, "eps0": 0.01}
    s2 = dict(struct, weight=2.5, scale=0.7)
    x = truth.x0 + np.random.default_rng(0).normal(0, 1e-3, len(truth.x0))
    for s in (struct, s2):
        r = generic_residuals(x, truth, [s, struct], w, truth.r_cut, None)
        assert len(r) == residual_length([s, struct]) == 2 * (1 + 3 * 256 + 9)
        assert r @ r == pytest.approx(generic_loss_function(x, truth, [s, struct], w, truth.r_cut, None),
                                      rel=1e-12)


def test_least_squares_recovers_the_reference(tmp_path):
    """ADR-0015 Q21: local_method 'least_squares' (dogbox on the residual
    vector) returns to the Zhou Ni-Al parameters from 2 % off,
    on one L1_2 cell, fitting the pair terms and the embedding slope terms."""
    from akitsushima.fitting.driver import fit_potential, resolve_local_method
    assert resolve_local_method("LEAST_SQUARES")[0] == "least_squares"
    truth, struct = _eam_struct()
    keys = ["A", "B", "F2", "eta"]
    el = {s: dict(ZHOU_2004[s]) for s in ("Al", "Ni")}
    for s in el:
        for k in keys:
            el[s][k] *= 1.02
    ps = ZhouEAMParameterSet(el, ["Al", "Ni"], fit_keys=keys, r_cut=truth.r_cut)
    res = fit_potential(ps, [struct], r_cut=truth.r_cut,
                        weights={"energy": 1, "forces": 1, "stress": 1, "r0": .1, "eps0": .01},
                        method="least_squares", val_fraction=0.0, verbose=False,
                        output_dir=str(tmp_path))
    assert res["loss"] < 1e-12
    for s in ("Al", "Ni"):
        for k in keys:
            assert res["params"]["eam"]["elements"][s][k] == pytest.approx(ZHOU_2004[s][k], rel=1e-5)
