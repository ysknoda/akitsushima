"""ADR-0019: Born charges from YAML, Gonze-Lee NAC, the static dielectric
tensor, IR modes and the piezoelectric tensor.

Rocksalt MgO, Buckingham (Lewis and Catlow, J. Phys. C 18, 1149 (1985):
Mg-O A 1428.5 eV, rho 0.2945 A; O-O A 22764 eV, rho 0.149 A, C 27.88 eV
A^6) with formal charges and PPPM, 8-atom conventional cell, 3x3x3
supercell (r_cut 5 A).
"""
import contextlib
import io
import json

import numpy as np
import pytest

from akitsushima.constants import K_E
from akitsushima.solvers.dielectric import (
    born_from_config,
    dielectric_response,
    piezoelectric_tensor,
    static_dielectric_by_field,
)
from akitsushima.solvers.phonon import _THZ_TO_CM, PhononCalculator
from akitsushima.solvers.qha import build_supercell, relax_at_volume

MGO_JSON = {"potential_type": "buckingham", "r_cut": 5.0, "species": ["Mg", "O"],
            "charges": {"Mg": 2.0, "O": -2.0},
            "params": {"Mg-O": {"A": 1428.5, "rho": 0.2945, "C": 0.0},
                       "O-O": {"A": 22764.0, "rho": 0.149, "C": 27.88},
                       "Mg-Mg": {"A": 0.0, "rho": 0.1, "C": 0.0}}}
_TYPES = ["Mg"] * 4 + ["O"] * 4


def write_mgo(directory, a=4.21, n=1):
    """MgO json and an n^3 supercell POSCAR of the conventional cell."""
    from akitsushima.io.poscar import write_poscar
    (directory / "mgo_buckingham.json").write_text(json.dumps(MGO_JSON))
    f = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    L, P, T = build_supercell(np.eye(3) * a, np.vstack([f, f + [.5, 0, 0]]) * a,
                              _TYPES, n)
    write_poscar(str(directory / "POSCAR"), L, P, T)


@pytest.fixture(scope="module")
def mgo(tmp_path_factory):
    d = tmp_path_factory.mktemp("mgo")
    (d / "mgo_buckingham.json").write_text(json.dumps(MGO_JSON))
    from akitsushima.cli.common import _cli_force_fn
    fn, _ = _cli_force_fn({"type": "Buckingham",
                           "potential_json": str(d / "mgo_buckingham.json"),
                           "r_cut": 5.0, "coulomb_method": "pppm"}, "test")
    f = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    lat = np.eye(3) * 4.21
    r = relax_at_volume(lat, np.vstack([f, f + [.5, 0, 0]]) @ lat, _TYPES, fn,
                        None, 3, {"fmax": 1e-5, "pmax": 1e-4}, output_dir=str(d))
    with contextlib.redirect_stdout(io.StringIO()):
        pc = PhononCalculator(r["lattice"], r["positions"], _TYPES, fn, supercell=3)
        pc.compute_ifc(verbose=False)
    return fn, r, pc


def _f(pc, q, qd=None):
    return np.sort(PhononCalculator._eig_to_freq(
        np.linalg.eigvalsh(pc.build_dynamical_matrix(q, qd))) / _THZ_TO_CM)


# ------------------------------------------------------------- YAML input

def test_born_from_config_shapes_override_and_sum_rule():
    types = ["Zn", "Zn", "O", "O"]
    b = born_from_config({"epsilon_inf": [3.6, 3.6, 3.8],
                          "born_charges": {"per_species": {"Zn": [2.1, 2.1, 2.2],
                                                           "O": -2.0},
                                           "per_atom": {4: [[-2.1, 0, 0], [0, -2.1, 0],
                                                            [0, 0, -2.3]]}}}, types)
    assert np.allclose(b["epsilon_inf"], np.diag([3.6, 3.6, 3.8]))
    assert b["sources"] == ["per_species", "per_species", "per_species", "per_atom"]
    # sum rule: raw zz sum 2.2 + 2.2 - 2.0 - 2.3 = 0.1 -> 0.025 removed each
    assert b["sum_rule_correction"] == pytest.approx(0.025)
    assert np.allclose(b["Z"].sum(axis=0), 0.0, atol=1e-14)
    assert b["Z"][0, 2, 2] == pytest.approx(2.2 - 0.025)
    # point charges complete what the block leaves out; eps_inf -> 1
    b = born_from_config(None, types, {"Zn": 2.0, "O": -2.0})
    assert b["rigid_ion_default"] and np.allclose(b["epsilon_inf"], np.eye(3))
    assert np.allclose(b["Z"][2], -2.0 * np.eye(3))
    assert born_from_config(None, types, None) is None
    with pytest.raises(ValueError, match="no Born charge"):
        born_from_config({"born_charges": {"per_species": {"Zn": 2.0}}}, types)
    with pytest.raises(ValueError, match="outside"):
        born_from_config({"born_charges": {"per_atom": {5: 1.0}}}, types,
                         {"Zn": 2.0, "O": -2.0})
    with pytest.raises(ValueError, match=">= 1"):
        born_from_config({"epsilon_inf": 0.5}, types, {"Zn": 2.0, "O": -2.0})


# ------------------------------------------------------------- NAC

def test_nac_leaves_commensurate_points_and_matches_phonopy(mgo):
    """Oracle: phonopy's Gonze-Lee NAC with the same Z* and eps_inf on its
    own force constants.  A non-symmetric, anisotropic Z* and eps_inf
    test the index conventions (Z[k, a, b] = dP_a / du_kb, as phonopy and
    VASP's OUTCAR): phonopy then agrees to 5e-5 THz, and to 0.08 THz off
    with Z transposed.  phonopy is run with is_symmetry=False: otherwise
    it projects Z* and eps_inf onto the cubic site symmetry, which these
    deliberately non-cubic tensors do not have."""
    pytest.importorskip("phonopy")
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    fn, r, pc = mgo
    Zmg = np.array([[2.0, 0.3, 0.0], [-0.1, 1.9, 0.2], [0.0, 0.1, 2.2]])
    born = born_from_config({"epsilon_inf": [[3.0, 0.1, 0], [0.1, 2.8, 0], [0, 0, 3.2]],
                             "born_charges": {"per_species": {"Mg": Zmg, "O": -Zmg}}},
                            _TYPES)
    rec = 2 * np.pi * np.linalg.inv(r["lattice"]).T
    q_comm = np.array([[1 / 3, 0, 0], [1 / 3, 1 / 3, 2 / 3]])
    before = [_f(pc, q @ rec) for q in q_comm]
    pc.set_nac(born["Z"], born["epsilon_inf"])
    try:
        # Exact up to the D-ASR correction, which is recomputed from the
        # short-range IFC (differs at ~1e-6 THz).
        for q, f0 in zip(q_comm, before):
            assert np.max(np.abs(_f(pc, q @ rec) - f0)) < 1e-5
        u = PhonopyAtoms(symbols=_TYPES, cell=r["lattice"], positions=r["positions"],
                         masses=list(pc.masses))
        p = Phonopy(u, supercell_matrix=np.diag([3, 3, 3]), primitive_matrix="P",
                    is_symmetry=False)
        p.generate_displacements(distance=0.01)
        p.forces = np.array([fn(np.array(s.cell), np.array(s.positions),
                                list(s.symbols))[1]
                             for s in p.supercells_with_displacements])
        p.produce_force_constants()
        p.nac_params = {"born": born["Z"], "dielectric": born["epsilon_inf"],
                        "factor": K_E}
        for q in ([0.1, 0.05, 0.2], [0.25, 0.0, 0.0], [0.41, 0.17, 0.33]):
            ours = _f(pc, np.array(q) @ rec)
            assert np.max(np.abs(ours - np.sort(p.get_frequencies(q)))) < 2e-3
        for d in ([0, 0, 1], [1, 1, 0]):
            ours = _f(pc, np.zeros(3), np.array(d, float) @ rec)
            p.run_qpoints([[0, 0, 0]], nac_q_direction=d)
            ref = np.sort(p.get_qpoints_dict()["frequencies"][0])
            assert np.max(np.abs(ours - ref)) < 2e-3
    finally:
        pc._nac = None


# ------------------------------------------------------------- dielectric

def test_lyddane_sachs_teller_and_two_routes_to_eps0(mgo, tmp_path):
    """Rocksalt has one IR-active triplet, so eps_0 / eps_inf =
    (w_LO / w_TO)^2 exactly (Lyddane-Sachs-Teller); and eps_0 from the
    Gamma modes equals eps_0 from relaxing the ions in a field (the route
    of piezo), on the same structure."""
    fn, r, pc = mgo
    born = born_from_config({"epsilon_inf": 2.95}, _TYPES, {"Mg": 2.0, "O": -2.0})
    vol = abs(np.linalg.det(r["lattice"]))
    res = dielectric_response(pc.build_dynamical_matrix(np.zeros(3)), pc.masses,
                              born["Z"], born["epsilon_inf"], vol)
    eps0 = res["epsilon_0"]
    assert np.allclose(eps0, eps0[0, 0] * np.eye(3), atol=1e-6)
    ir = [m for m in res["modes"] if m["ir_intensity_e2_amu"] > 1e-6]
    assert len(ir) == 3
    w_to = ir[0]["frequency_THz"]
    f_to = _f(pc, np.zeros(3))
    pc.set_nac(born["Z"], born["epsilon_inf"])
    try:
        f_nac = _f(pc, np.zeros(3), np.array([0, 0, 1.0]))
    finally:
        pc._nac = None
    # the one frequency the q -> 0 limit adds: the LO mode
    new = [f for f in f_nac if np.min(np.abs(f_to - f)) > 1e-3]
    assert len(new) == 1
    w_lo = new[0]
    assert eps0[0, 0] / 2.95 == pytest.approx((w_lo / w_to) ** 2, rel=2e-4)
    # mode-resolved: the IR triplet carries all of eps_0 - eps_inf
    assert np.allclose(sum(m["delta_epsilon"] for m in ir), eps0 - 2.95 * np.eye(3))
    L, P, T = build_supercell(r["lattice"], r["positions"], _TYPES, 3)
    eps_f = static_dielectric_by_field(fn, L, P, T, np.tile(born["Z"], (27, 1, 1)),
                                       born["epsilon_inf"], field=5e-3,
                                       relax_kw={"fmax": 1e-6},
                                       output_dir=str(tmp_path))
    # The mode route carries the O(delta^2) error of the 0.01 A force
    # constants: 9.3768 here, 9.3809 extrapolated to delta -> 0; the field
    # route gives 9.3801 (written down while choosing the default field).
    assert np.allclose(eps_f, eps0, rtol=5e-4, atol=5e-4)


def test_dielectric_response_refuses_imaginary_optical_mode():
    D = np.diag([0, 0, 0, -1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="imaginary"):
        dielectric_response(D, [1.0, 1.0], np.zeros((2, 3, 3)), np.eye(3), 10.0)


# ------------------------------------------------------------- piezo

def test_piezoelectric_tensor_units_and_definitions():
    """By hand: one Z = +-1 pair moving apart along z by 0.5 A per unit
    eta_3 in a 10 A^3 cell gives e_33 = (1 x 0.5 + 1 x 0.5) e A / 10 A^3
    = 0.1 e/A^2 = 1.602 C/m^2; d = e S; k from d, S and eps^T."""
    Z = np.array([np.eye(3), -np.eye(3)])
    u = np.zeros((2, 3, 6))
    u[0, 2, 2], u[1, 2, 2] = 0.5, -0.5
    S = np.diag([0.01, 0.01, 0.01, 0.02, 0.02, 0.02])          # 1/GPa
    r = piezoelectric_tensor(Z, u, 10.0, S, 5.0 * np.eye(3))
    assert r["e_C_m2"][2, 2] == pytest.approx(1.602176634, rel=1e-12)
    assert np.count_nonzero(np.abs(r["e_C_m2"]) > 1e-12) == 1
    assert r["d_pC_N"][2, 2] == pytest.approx(1.602176634 * 0.01e-9 * 1e12, rel=1e-12)
    eps0 = 8.8541878128e-12
    epsT = 5.0 * eps0 + r["e_C_m2"][2, 2] * r["d_pC_N"][2, 2] * 1e-12
    k33 = r["d_pC_N"][2, 2] * 1e-12 / np.sqrt(epsT * 0.01e-9)
    assert r["k"][2, 2] == pytest.approx(k33, rel=1e-12)
    r2 = piezoelectric_tensor(Z, u, 10.0, S, 5.0 * np.eye(3),
                              clamped_ion=np.full((3, 6), 0.1))
    assert np.allclose(r2["e_C_m2"] - r["e_C_m2"], 0.1)


def test_piezo_command_reuses_elastic_by_hash_and_refuses_others(tmp_path, monkeypatch):
    """MgO (centrosymmetric): e = 0 and the note; load_elastic is accepted
    for the same POSCAR + potential and refused for another potential."""
    import yaml
    from test_cli_smoke import _ns
    from akitsushima.cli.cmd_piezo import cmd_piezo
    from akitsushima.cli.templates import _DUMP_MAP
    monkeypatch.chdir(tmp_path)
    write_mgo(tmp_path, n=3)
    cfg = yaml.safe_load(_DUMP_MAP["piezo"])
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(cfg))
    assert cmd_piezo(_ns(tmp_path / "p.yaml")) == 0
    out = json.loads((tmp_path / "piezo_output" / "piezo_results.json").read_text())
    assert out["centrosymmetric_like"] and out["max_abs_e_C_m2"] < 1e-4
    cfg["piezo"]["load_elastic"] = "./elastic_output"
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(cfg))
    assert cmd_piezo(_ns(tmp_path / "p.yaml")) == 0
    cfg["potential"]["r_cut"] = 4.9
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="input hash"):
        cmd_piezo(_ns(tmp_path / "p.yaml"))


def test_dielectric_refuses_a_potential_without_charges(tmp_path, monkeypatch):
    import yaml
    from test_cli_smoke import _ns
    from test_cli_templates import _write_si_primitive
    from akitsushima.cli.cmd_dielectric import cmd_dielectric
    from akitsushima.cli.templates import _DUMP_MAP
    monkeypatch.chdir(tmp_path)
    _write_si_primitive(tmp_path / "POSCAR")
    cfg = yaml.safe_load(_DUMP_MAP["dielectric"])
    cfg["potential"] = yaml.safe_load(_DUMP_MAP["phonon"])["potential"]
    cfg.pop("born", None)
    cfg["dielectric"]["supercell"] = [4, 4, 4]
    (tmp_path / "d.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="no Born charges"):
        cmd_dielectric(_ns(tmp_path / "d.yaml"))
