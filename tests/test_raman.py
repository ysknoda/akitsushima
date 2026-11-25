"""Raman spectra in the harmonic approximation (ADR-0046 stage S1,
ADR-0049), and the parts ported from nozuchi 0.2.0a2.

Checks 1-4 of ADR-0046 decision 14:
  1. selection-rule counts (diamond, wurtzite, rock salt) from characters;
  2. diamond tensor form, powder rho = 0.75, rotation and permutation;
  3. BPM derivative against central differences, translation sum;
  4. a 16-atom diamond supercell gives the primitive Gamma spectrum per
     atom, and its folded modes are inactive with zero intensity.
Check 4 uses Tersoff Si (cited parameters in test_phonon_regressions)
rather than Ge: the structure and the test are the same.

The rock-salt case uses a short-range Born-Mayer model with made-up
parameters: only its symmetry matters.
"""

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from akitsushima.io.outcar_dielectric import (
    map_to_unit_cell,
    read_outcar_born,
    read_outcar_eps_inf,
)
from akitsushima.solvers.raman import (
    gamma_modes,
    lorentzian,
    powder_invariants,
    raman_modes,
    thermal_factor,
)
from akitsushima.solvers.response import (
    BondPolarizabilityModel,
    FixedResponse,
    polarizability_from_config,
)
from test_phonon_regressions import _ALN, _SI

from akitsushima.io.poscar import parse_poscar
from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml
from akitsushima.solvers.dielectric import born_from_config
from akitsushima.solvers.phonon import PhononCalculator

DATA = Path(__file__).parent / "data" / "nozuchi"
OUTCAR = DATA / "sic_dfpt_outcar_excerpt.txt"
POSCAR_SIC = DATA / "sic_unit_POSCAR.vasp"

A_SI = 5.431988  # Tersoff Si equilibrium (as test_phonon_regressions)

BONDS = {
    "Si-Si": dict(r_cut=2.8, r0=2.35, alpha_diff=1.5, d_alpha_l=2.5, d_alpha_p=0.5),
    "Al-N": dict(r_cut=2.3, r0=1.9, alpha_diff=1.0, d_alpha_l=2.0, d_alpha_p=0.6),
    "Na-Cl": dict(r_cut=3.2, r0=2.8, alpha_diff=1.0, d_alpha_l=2.0, d_alpha_p=0.6),
}


def _sio(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _fn(pot):
    return _build_force_fn_from_yaml(pot)[0]


def _si_fn():
    return _fn({"type": "Tersoff", "species": ["Si"], "params": _SI})


def _si_primitive():
    lat = 0.5 * A_SI * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    return lat, np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ lat, ["Si"] * 2


def _gamma(lat, pos, types, fn, sc):
    pc = PhononCalculator(lat, pos, types, fn, supercell=sc)
    _sio(pc.compute_ifc, verbose=False)
    return pc, pc.build_dynamical_matrix(np.zeros(3))


def _ops(lat, pos, types, masses):
    pytest.importorskip("spglib")
    from akitsushima.solvers.symmetry import spglib_operations

    return spglib_operations(lat, pos, types, masses)


def _run(lat, pos, types, fn, sc, geometries=()):
    pc, D = _gamma(lat, pos, types, fn, sc)
    A = BondPolarizabilityModel(BONDS).polarizability_derivative(lat, pos, types)
    return raman_modes(
        D, pc.masses, A, _ops(lat, pos, types, pc.masses), geometries=geometries
    )


def _optical(res):
    return [r for r in res["sets"] if not r["acoustic"]]


# ------------------------------------------------ ported nozuchi parts


def test_outcar_readers():
    z = read_outcar_born(OUTCAR)
    assert z.shape == (8, 3, 3)
    assert np.allclose(z[:4], -2.6898 * np.eye(3), atol=1e-5)
    assert np.allclose(z[4:], 2.6898 * np.eye(3), atol=1e-5)
    assert np.allclose(read_outcar_eps_inf(OUTCAR), 7.004568 * np.eye(3), atol=1e-6)


def test_eps_inf_rows_starting_with_negative_zero(tmp_path):
    p = tmp_path / "OUTCAR"
    p.write_text(
        " MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)\n"
        " ------------------------------------------------------\n"
        "           9.552689     0.000000    -0.000000\n"
        "          -0.000000     9.552689    -0.000000\n"
        "          -0.000000     0.000000     9.552689\n"
        " ------------------------------------------------------\n"
    )
    assert np.allclose(read_outcar_eps_inf(p), 9.552689 * np.eye(3), atol=1e-6)


def test_born_block_stops_at_next_table(tmp_path):
    extra = (
        "\n other table\n ion    1\n    1  9.0 9.0 9.0\n    2  9.0 9.0 9.0\n"
        "    3  9.0 9.0 9.0\n"
    )
    (tmp_path / "OUTCAR").write_text(OUTCAR.read_text() + extra)
    z = read_outcar_born(tmp_path / "OUTCAR")
    assert len(z) == 8 and not np.any(z == 9.0)


def _sic_unit():
    u = parse_poscar(str(POSCAR_SIC))
    lat = np.asarray(u["lattice"])
    return lat, list(u["atom_types"]), np.asarray(u["positions"]) @ np.linalg.inv(lat)


def test_site_mapping_supercell_shuffled_and_displaced():
    lat_u, sym_u, frac_u = _sic_unit()
    rng = np.random.default_rng(5)
    reps = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
    frac_sc = ((frac_u[None] + reps[:, None]) / 2.0).reshape(-1, 3)
    lat_sc = 2 * lat_u
    frac_sc = frac_sc + rng.normal(scale=0.15, size=frac_sc.shape) @ np.linalg.inv(
        lat_sc
    )
    perm = rng.permutation(len(frac_sc))
    sym_sc = sym_u * 8
    site = map_to_unit_cell(
        frac_sc[perm],
        lat_sc,
        [sym_sc[i] for i in perm],
        lat_u,
        sym_u,
        frac_u,
        [2, 2, 2],
    )
    assert np.array_equal(site, np.tile(np.arange(8), 8)[perm])


def test_site_mapping_removes_rigid_translation():
    lat_u, sym_u, frac_u = _sic_unit()
    rng = np.random.default_rng(6)
    reps = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
    frac_sc = ((frac_u[None] + reps[:, None]) / 2.0).reshape(-1, 3)
    lat_sc = 2 * lat_u
    shift = np.array([0.55, -0.45, 0.5])
    frac_sc = frac_sc + (
        rng.normal(scale=0.1, size=frac_sc.shape) + shift
    ) @ np.linalg.inv(lat_sc)
    with pytest.warns(
        UserWarning, match="rigid translation of the whole crystal of 0.8"
    ):
        site = map_to_unit_cell(
            frac_sc, lat_sc, sym_u * 8, lat_u, sym_u, frac_u, [2, 2, 2]
        )
    assert np.array_equal(site, np.tile(np.arange(8), 8))


def test_born_from_config_outcar_priority_and_eps():
    """per_atom > per_species > outcar; the sum rule; eps_inf from the
    OUTCAR; a 2x1x1 cell maps onto the 8-atom DFT cell."""
    lat_u, sym_u, frac_u = _sic_unit()
    frac = np.vstack([frac_u * [0.5, 1, 1], frac_u * [0.5, 1, 1] + [0.5, 0, 0]])
    lat = lat_u * [[2], [1], [1]]
    types = sym_u * 2
    cfg = {
        "epsilon_inf": {"outcar": str(OUTCAR)},
        "born_charges": {
            "outcar": str(OUTCAR),
            "unit_cell": str(POSCAR_SIC),
            "supercell": [2, 1, 1],
            "per_atom": {16: 3.0},
        },
    }
    b = born_from_config(cfg, types, None, lattice=lat, positions=frac @ lat)
    assert np.allclose(b["epsilon_inf"], 7.004568 * np.eye(3), atol=1e-6)
    assert b["sources"][:15] == ["outcar"] * 15 and b["sources"][15] == "per_atom"
    assert np.allclose(b["Z"].sum(axis=0), 0.0, atol=1e-12)
    # 3.0 instead of 2.6898 on one of 16 atoms: mean removed = 0.3102 / 16
    assert b["sum_rule_correction"] == pytest.approx((3.0 - 2.6898) / 16, abs=1e-5)
    cfg["born_charges"]["per_species"] = {"C": -1.0}
    b = born_from_config(cfg, types, None, lattice=lat, positions=frac @ lat)
    assert b["sources"].count("per_species") == 8
    with pytest.raises(ValueError, match="unit_cell"):
        born_from_config(
            {"born_charges": {"outcar": str(OUTCAR)}},
            types,
            lattice=lat,
            positions=frac @ lat,
        )


# ------------------------------------------------ check 3: BPM


@pytest.mark.parametrize("cell", ["alloy", "primitive"])
def test_bpm_derivative_matches_central_differences(cell):
    rng = np.random.default_rng(0)
    if cell == "alloy":
        f = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        f = np.vstack([f, f + 0.25])
        lat = np.eye(3) * 5.6 + rng.normal(scale=0.05, size=(3, 3))
        pos = f @ lat + rng.normal(scale=0.08, size=(8, 3))
        types = ["Ge", "Sn"] * 4
    else:  # bonds through periodic images only
        lat, pos, _ = _si_primitive()
        pos = pos + rng.normal(scale=0.05, size=(2, 3))
        types = ["Ge", "Ge"]
    m = BondPolarizabilityModel(
        {
            "Ge-Ge": dict(
                r_cut=2.9, r0=2.45, alpha_diff=1.3, d_alpha_l=2.1, d_alpha_p=0.4
            ),
            "Ge-Sn": dict(
                r_cut=3.0, r0=2.6, alpha_diff=1.7, d_alpha_l=2.8, d_alpha_p=0.6
            ),
            "Sn-Sn": dict(
                r_cut=3.1, r0=2.8, alpha_diff=2.0, d_alpha_l=3.0, d_alpha_p=0.9
            ),
        }
    )
    A = m.polarizability_derivative(lat, pos, types)
    h, Afd = 1e-5, np.zeros_like(A)
    for k in range(len(pos)):
        for c in range(3):
            p = pos.copy()
            p[k, c] += h
            ap = m.polarizability(lat, p, types)
            p[k, c] -= 2 * h
            Afd[k, :, :, c] = (ap - m.polarizability(lat, p, types)) / (2 * h)
    assert np.abs(A).max() > 0.1
    assert np.allclose(A, Afd, atol=1e-8)
    assert np.allclose(A.sum(axis=0), 0.0, atol=1e-14)  # translation
    assert np.allclose(A, A.transpose(0, 2, 1, 3))  # alpha symmetric
    n_bonds = len(m._bonds(lat, pos, types)[0]) // 2
    assert n_bonds == (16 if cell == "alloy" else 4)


def test_bpm_and_config_refusals(tmp_path):
    with pytest.raises(ValueError, match="missing keys"):
        BondPolarizabilityModel({"Si-Si": {"r_cut": 2.8}})
    with pytest.raises(ValueError, match="given twice"):
        BondPolarizabilityModel({"Si-Ge": BONDS["Si-Si"], "Ge-Si": BONDS["Si-Si"]})
    with pytest.raises(ValueError, match="S2"):
        polarizability_from_config({"source": "vasp_fd"})
    with pytest.raises(ValueError, match="exactly one"):
        polarizability_from_config({"source": "bpm"})
    (tmp_path / "p.json").write_text(json.dumps({"bonds": {"Si-Si": BONDS["Si-Si"]}}))
    m = polarizability_from_config({"params_file": str(tmp_path / "p.json")})
    assert set(m.bonds) == {("Si", "Si")}
    assert polarizability_from_config(None) is None


def test_fixed_response_dipole_derivative_is_z():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(3, 3, 3))
    r = FixedResponse(born_charges=Z)
    ref = rng.normal(size=(3, 3))
    for k in range(3):
        for b in range(3):
            p = ref.copy()
            p[k, b] += 0.01
            dM = r.dipole(None, p, None, ref) / 0.01
            assert np.allclose(dM, Z[k, :, b])
    with pytest.raises(NotImplementedError):
        r.polarizability_derivative(None, ref, None)


# ------------------------------------------------ check 1: selection rules


def test_selection_rules_diamond():
    res = _run(*_si_primitive(), _si_fn(), 3)
    opt = _optical(res)
    assert len(opt) == 1 and opt[0]["degeneracy"] == 3
    assert opt[0]["raman_active"] and not opt[0]["ir_active"]
    assert res["activity_deviation"] < 1e-8


def test_selection_rules_wurtzite():
    """A1 + E1 + 2E2 Raman, A1 + E1 IR, 2B1 silent.  u and c/a are moved
    off the ideal tetrahedron, where A1 and E1 (TO) are almost degenerate
    and a BPM gives the A1 tensor no trace."""
    a, c, u = 3.381244, 5.45, 0.382
    lat = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    frac = np.array(
        [
            [1 / 3, 2 / 3, 0],
            [2 / 3, 1 / 3, 0.5],
            [1 / 3, 2 / 3, u],
            [2 / 3, 1 / 3, 0.5 + u],
        ]
    )
    fn = _fn({"type": "Tersoff", "species": ["Al", "N"], "params": _ALN})
    res = _run(lat, frac @ lat, ["Al", "Al", "N", "N"], fn, [4, 4, 3])
    kinds = sorted(
        (r["degeneracy"], r["ir_active"], r["raman_active"]) for r in _optical(res)
    )
    assert kinds == sorted(
        [
            (2, False, True),
            (2, False, True),  # 2 E2
            (1, True, True),
            (2, True, True),  # A1, E1
            (1, False, False),
            (1, False, False),
        ]
    )  # 2 B1
    assert res["activity_deviation"] < 1e-6
    for r in _optical(res):  # silent and IR-only modes carry no intensity
        if not r["raman_active"]:
            # 1e-10 of the largest is seen: the Cartesian displacements
            # break the hexagonal symmetry at O(delta^2) (ADR-0040).
            assert r["powder_intensity"] < 1e-8 * max(
                q["powder_intensity"] for q in _optical(res)
            )


def test_selection_rules_rocksalt():
    a = 5.64
    lat = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    pos = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) @ lat
    fn = _fn(
        {
            "type": "BornMayer",
            "species": ["Na", "Cl"],
            "r_cut": 4.5,
            "params": [
                {"i": "Na", "j": "Cl", "A": 1000.0, "rho": 0.3},
                {"i": "Na", "j": "Na", "A": 500.0, "rho": 0.3},
                {"i": "Cl", "j": "Cl", "A": 500.0, "rho": 0.3},
            ],
        }
    )
    res = _run(lat, pos, ["Na", "Cl"], fn, 3)
    opt = _optical(res)
    assert len(opt) == 1 and opt[0]["degeneracy"] == 3
    assert opt[0]["ir_active"] and not opt[0]["raman_active"]
    assert opt[0]["powder_intensity"] < 1e-20


# ------------------------------------------------ check 2: tensor form


_GEOS = [
    {"name": "xy", "incident": [1, 0, 0], "scattered": [0, 1, 0]},
    {"name": "xx", "incident": [1, 0, 0], "scattered": [1, 0, 0]},
    {"name": "d", "incident": [1, 1, 0], "scattered": [0, 0, 1]},
]


def test_diamond_tensor_form_powder_rho_rotation_and_permutation():
    lat, pos, types = _si_primitive()
    fn = _si_fn()
    res = _run(lat, pos, types, fn, 3, _GEOS)
    (opt,) = _optical(res)
    R = res["raman_tensors"][opt["modes"]]
    # Each mode tensor: no diagonal; the set sum of R R^T is d^2 (J - I)-like
    assert np.abs(np.einsum("maa->ma", R)).max() < 1e-12 * np.abs(R).max()
    assert opt["rho"] == pytest.approx(0.75, abs=1e-12)
    assert opt["geometry"]["xx"] < 1e-20
    # Cartesian rotation of the whole crystal: powder invariants and the
    # rotated geometries give the same numbers.
    th = 0.4
    Q = np.array(
        [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    ) @ np.array(
        [[1, 0, 0], [0, np.cos(0.3), -np.sin(0.3)], [0, np.sin(0.3), np.cos(0.3)]]
    )
    geos_rot = [
        {
            "name": g["name"],
            "incident": Q @ np.array(g["incident"], float),
            "scattered": Q @ np.array(g["scattered"], float),
        }
        for g in _GEOS
    ]
    rot = _run(lat @ Q.T, pos @ Q.T, types, fn, 3, geos_rot)
    (opt_r,) = _optical(rot)
    # The Cartesian displacements turn with the axes, not with the crystal:
    # the O(delta^2) term moves the frequency by ~4e-3 cm^-1.
    assert opt_r["frequency_cm"] == pytest.approx(opt["frequency_cm"], abs=1e-2)
    assert opt_r["powder_intensity"] == pytest.approx(opt["powder_intensity"], rel=1e-8)
    for g in ("xy", "d"):
        assert opt_r["geometry"][g] == pytest.approx(opt["geometry"][g], rel=1e-8)
    # Atom permutation changes nothing.
    perm = _run(lat, pos[::-1].copy(), types, fn, 3, _GEOS)
    (opt_p,) = _optical(perm)
    for k in ("powder_intensity", "rho"):
        assert opt_p[k] == pytest.approx(opt[k], rel=1e-10)
    assert opt_p["geometry"]["xy"] == pytest.approx(opt["geometry"]["xy"], rel=1e-10)


def test_powder_invariants_and_factors():
    R = np.array([np.diag([1.0, 0.0, 0.0]), np.diag([1.0, 1.0, 1.0])])
    a2, g2 = powder_invariants(R)
    assert np.allclose(a2, [1 / 9, 1.0]) and np.allclose(g2, [1.0, 0.0])
    # rho: 3 g2 / (45 a2 + 4 g2) = 3 / 9 for diag(1, 0, 0); 0 for isotropic
    assert 3 * g2[0] / (45 * a2[0] + 4 * g2[0]) == pytest.approx(1 / 3)
    w = np.array([100.0, 500.0])
    assert np.allclose(thermal_factor(w, 0.0), 1.0 / w)
    n = 1.0 / np.expm1(1.438776877 * w / 300.0)
    assert np.allclose(thermal_factor(w, 300.0), (n + 1) / w)
    assert np.allclose(thermal_factor(w, 300.0, anti_stokes=True), n / w)
    wl = 1e7 / 532.0
    assert np.allclose(
        thermal_factor(w, 300.0, laser_nm=532.0), (n + 1) / w * (wl - w) ** 4
    )
    grid = np.arange(-2000.0, 2000.0, 0.01)
    L = lorentzian(grid, np.array([0.0]), np.array([2.0]), 3.0)
    assert L.sum() * 0.01 == pytest.approx(2.0, rel=2e-3)
    assert L.max() == pytest.approx(2.0 * 2.0 / (np.pi * 3.0), rel=1e-6)


# ------------------------------------------------ check 4: supercell


def test_supercell_folds_onto_the_primitive_gamma_spectrum():
    """16-atom 2x2x2 replica of the primitive cell: the Gamma triplet has
    the primitive frequency and the same intensity per atom; every other
    (folded) set is inactive by characters and has no intensity.  The
    dynamical matrix of the cell itself is exact at Gamma from any valid
    supercell (here 3x3x3 against 2x2x2 of the primitive cell)."""
    lat, pos, types = _si_primitive()
    fn = _si_fn()
    prim = _run(lat, pos, types, fn, 3)
    _, D2 = _gamma(lat, pos, types, fn, 2)
    _, D4 = _gamma(lat, pos, types, fn, 4)
    # optical modes; the acoustic ones are sqrt(round-off), ~1e-5 cm^-1
    assert np.allclose(gamma_modes(D2)[0][3:], gamma_modes(D4)[0][3:], atol=1e-8)
    cells = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
    frac = pos @ np.linalg.inv(lat)
    f16 = ((frac[None] + cells[:, None]) / 2.0).reshape(-1, 3)
    lat16 = 2 * lat
    sup = _run(lat16, f16 @ lat16, types * 8, fn, 2)
    (p,) = _optical(prim)
    active = [r for r in _optical(sup) if r["raman_active"]]
    assert len(active) == 1
    (s,) = active
    assert s["degeneracy"] == 3
    assert s["frequency_cm"] == pytest.approx(p["frequency_cm"], abs=1e-3)
    assert s["powder_intensity"] / 16 == pytest.approx(
        p["powder_intensity"] / 2, rel=1e-6
    )
    folded = [r for r in _optical(sup) if r is not s]
    assert sum(r["degeneracy"] for r in folded) == 3 * 16 - 6
    assert not any(r["ir_active"] or r["raman_active"] for r in folded)
    assert max(r["powder_intensity"] for r in folded) < 1e-12 * s["powder_intensity"]


# ------------------------------------------------ CLI


@pytest.mark.slow
def test_cli_structures_average_lattice_basis_and_refusals(tmp_path, monkeypatch):
    """Two identical configurations average to themselves with zero std;
    a [110] lattice-basis geometry equals its Cartesian equivalent; the
    forbidden z(xx) column stays at zero after normalisation."""
    import yaml
    from akitsushima.cli.cmd_raman import cmd_raman
    from test_cli_smoke import _ns
    from test_cli_templates import _write_si_primitive

    from akitsushima.cli.templates import _DUMP_MAP

    monkeypatch.chdir(tmp_path)
    _write_si_primitive(tmp_path / "POSCAR", a=A_SI)
    _write_si_primitive(tmp_path / "POSCAR2", a=A_SI)
    cfg = yaml.safe_load(_DUMP_MAP["raman"])
    lat = 0.5 * A_SI * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    cart = np.array([1.0, 1.0, 0.0])
    uvw = cart @ np.linalg.inv(lat)  # the same direction in a_i
    cfg["raman"]["geometries"] += [
        {"name": "c110", "incident": cart.tolist(), "scattered": [0, 0, 1]},
        {
            "name": "l110",
            "incident": uvw.tolist(),
            "scattered": [0, 0, 1],
            "basis": "lattice",
        },
    ]
    cfg["raman"]["structures"] = ["POSCAR", "POSCAR2"]
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(cfg))
    assert _sio(cmd_raman, _ns(tmp_path / "r.yaml")) == 0
    out = tmp_path / "raman_output"
    res = json.loads((out / "raman_results.json").read_text())
    assert len(res["structures"]) == 2
    s0 = [r for r in res["structures"][0]["sets"] if not r["acoustic"]][0]
    assert s0["geometry"]["c110"] == pytest.approx(s0["geometry"]["l110"], rel=1e-10)
    head = (out / "raman_spectrum_normalised.dat").read_text().splitlines()[2]
    cols = head.lstrip("# ").split()
    data = np.loadtxt(out / "raman_spectrum_normalised.dat")
    assert np.abs(data[:, cols.index("powder_std")]).max() < 1e-12
    assert data[:, cols.index("powder")].max() == pytest.approx(1.0)
    assert np.abs(data[:, cols.index("z(xx)-z")]).max() < 1e-12
    assert (out / "raman_modes_000.dat").exists() and (
        out / "raman_modes_001.dat"
    ).exists()
    cfg["raman"]["bogus"] = 1
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="unknown keys"):
        cmd_raman(_ns(tmp_path / "r.yaml"))


def test_mode_tensor_is_d_alpha_d_normal_coordinate():
    """R_m = d alpha / dQ_m with u_k = e_m(k) Q / sqrt(M_k): central
    differences of the BPM alpha along each optical mode of wurtzite AlN
    (two species, so the mass weighting is tested too)."""
    a, c, u = 3.381244, 5.45, 0.382
    lat = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    frac = np.array([[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, .5],
                     [1 / 3, 2 / 3, u], [2 / 3, 1 / 3, .5 + u]])
    pos, types = frac @ lat, ["Al", "Al", "N", "N"]
    fn = _fn({"type": "Tersoff", "species": ["Al", "N"], "params": _ALN})
    pc, D = _gamma(lat, pos, types, fn, [4, 4, 3])
    m = BondPolarizabilityModel(BONDS)
    res = raman_modes(D, pc.masses, m.polarizability_derivative(lat, pos, types))
    E = res["eigenvectors"]
    h = 1e-4
    for mode in range(3, 12):
        disp = (E[:, mode].reshape(4, 3) / np.sqrt(pc.masses)[:, None])
        fd = (m.polarizability(lat, pos + h * disp, types)
              - m.polarizability(lat, pos - h * disp, types)) / (2 * h)
        assert np.allclose(res["raman_tensors"][mode], fd, atol=1e-8)
