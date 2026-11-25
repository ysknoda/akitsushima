"""OUTCAR parsing, Born-charge site mapping, sum rule, masses (ADR-0002)."""

import numpy as np
import pytest

from akitsushima.mdspec import charges as ch

from .conftest import DATA

OUTCAR = DATA / "sic_dfpt_outcar_excerpt.txt"
POSCAR = DATA / "sic_unit_POSCAR.vasp"


def test_outcar_readers():
    z = ch.read_outcar_born(OUTCAR)
    assert z.shape == (8, 3, 3)
    assert np.allclose(z[:4], -2.6898 * np.eye(3), atol=1e-5)
    assert np.allclose(z[4:], 2.6898 * np.eye(3), atol=1e-5)
    assert np.allclose(ch.read_outcar_eps_inf(OUTCAR), 7.004568 * np.eye(3), atol=1e-6)
    assert ch.read_outcar_pomass(OUTCAR) == [12.011, 28.085]
    assert np.allclose(ch.read_outcar_temperatures(OUTCAR), [301.25, 299.75])


def test_eps_inf_rows_starting_with_negative_zero(tmp_path):
    # regression: rows such as "-0.000000  9.55 ..." were taken for the dashed separator
    p = tmp_path / "OUTCAR"
    p.write_text(
        " MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)\n"
        " ------------------------------------------------------\n"
        "           9.552689     0.000000    -0.000000\n"
        "          -0.000000     9.552689    -0.000000\n"
        "          -0.000000     0.000000     9.552689\n"
        " ------------------------------------------------------\n"
    )
    assert np.allclose(ch.read_outcar_eps_inf(p), 9.552689 * np.eye(3), atol=1e-6)


def test_site_mapping_supercell_shuffled_and_displaced():
    lat_u, sym_u, frac_u = ch.read_poscar(POSCAR)
    rng = np.random.default_rng(5)
    reps = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
    frac_sc = ((frac_u[None] + reps[:, None]) / 2.0).reshape(-1, 3)
    sym_sc = sym_u * 8
    site_true = np.tile(np.arange(len(sym_u)), 8)
    lat_sc = 2 * lat_u
    # thermal-size displacements (0.15 A rms) and a shuffled atom order
    frac_sc = frac_sc + rng.normal(scale=0.15, size=frac_sc.shape) @ np.linalg.inv(lat_sc)
    perm = rng.permutation(len(frac_sc))
    site = ch.map_to_unit_cell(
        frac_sc[perm], lat_sc, [sym_sc[i] for i in perm], lat_u, sym_u, frac_u, [2, 2, 2]
    )
    assert np.array_equal(site, site_true[perm])


def test_site_mapping_removes_rigid_translation():
    # regression: a Langevin NVT moved a whole urea crystal by 0.4 A, beyond map_tolerance
    lat_u, sym_u, frac_u = ch.read_poscar(POSCAR)
    rng = np.random.default_rng(6)
    reps = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
    frac_sc = ((frac_u[None] + reps[:, None]) / 2.0).reshape(-1, 3)
    lat_sc = 2 * lat_u
    shift = np.array([0.55, -0.45, 0.5])  # 0.87 A, less than half the Si-Si distance
    frac_sc = frac_sc + (rng.normal(scale=0.1, size=frac_sc.shape) + shift) @ np.linalg.inv(lat_sc)
    with pytest.warns(UserWarning, match="rigid translation of the whole crystal of 0.8"):
        site = ch.map_to_unit_cell(frac_sc, lat_sc, sym_u * 8, lat_u, sym_u, frac_u, [2, 2, 2])
    assert np.array_equal(site, np.tile(np.arange(len(sym_u)), 8))


def test_born_charges_sum_rule_and_overrides():
    lat_u, sym_u, frac_u = ch.read_poscar(POSCAR)
    cfg = {
        "outcar": str(OUTCAR),
        "unit_cell": str(POSCAR),
        "supercell": [1, 1, 1],
        "per_species": None,
        "per_atom": {1: [-1.0, -1.0, -1.0]},  # mean shift 1.69/8 = 0.21 e > 0.1
        "map_tolerance": 0.6,
    }
    with pytest.warns(UserWarning, match="sum rule"):
        z = ch.born_charges(cfg, sym_u, frac_u, lat_u)
    assert np.allclose(z.sum(axis=0), 0, atol=1e-12)


def test_point_charges_neutrality():
    with pytest.raises(ValueError, match="not neutral"):
        ch.point_charges(["Mg", "O", "O"], {"Mg": 2.0, "O": -2.0})
    z = ch.point_charges(["Mg", "O"], {"Mg": 2.0, "O": -2.0})
    assert np.allclose(z[0], 2 * np.eye(3))


def test_mass_priority():
    m, src = ch.atom_masses(["C", "Si"], ["C", "Si"], {"Si": 30.0}, [str(OUTCAR)])
    assert m.tolist() == [12.011, 30.0]
    assert src == {"C": "OUTCAR POMASS", "Si": "YAML"}
    m, src = ch.atom_masses(["H"], ["H"], {}, None)
    assert m[0] == 1.008 and src["H"] == "standard atomic weight"


def test_born_convention_matches_vasp_finite_difference():
    """Tier 1 (ADR-0002 item 3): urea H7, VASP LCALCPOL central differences (+-0.02 A,
    2026-10-17) gave D[a, b] = dp_a/du_b equal to the OUTCAR tensor as printed (max
    deviation 0.0098 e) and not to its transpose (0.066 e).  nozuchi must therefore
    build M_a = sum_b Z[k, a, b] r_b with Z exactly as read from the OUTCAR."""
    from akitsushima.mdspec.dipole import dipole_from_chunk

    z_outcar = np.array(
        [[0.3102, -0.0824, 0.0318], [-0.0824, 0.3102, 0.0318], [0.097, 0.097, 0.6557]]
    )
    d_vasp = np.array([[0.314, -0.083, 0.031], [-0.083, 0.314, 0.031], [0.0935, 0.0935, 0.6655]])
    lat = np.eye(3)[None] * 10.0
    for b in range(3):
        u = np.zeros((2, 1, 3))
        u[1, 0, b] = 0.002  # fractional, i.e. 0.02 A
        m = dipole_from_chunk(u, np.repeat(lat, 2, axis=0), z_outcar[None], None)
        dm_du = (m[1] - m[0]) / 0.02
        assert np.allclose(dm_du, d_vasp[:, b], atol=0.012)
        assert np.allclose(dm_du, z_outcar[:, b])


def test_born_block_stops_at_next_table(tmp_path):
    text = (DATA / "sic_dfpt_outcar_excerpt.txt").read_text()
    n = len(ch.read_outcar_born(DATA / "sic_dfpt_outcar_excerpt.txt"))
    extra = (
        "\n other table\n ion    1\n    1  9.0 9.0 9.0\n    2  9.0 9.0 9.0\n    3  9.0 9.0 9.0\n"
    )
    (tmp_path / "OUTCAR").write_text(text + extra)
    z = ch.read_outcar_born(tmp_path / "OUTCAR")
    assert len(z) == n and not np.any(z == 9.0)


def test_poscar_negative_scale_cartesian(tmp_path):
    lat = np.diag([2.0, 3.0, 4.0])
    frac = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
    vol = 8.0 * 24.0  # target volume: the cell doubled in each direction
    cart = frac @ lat  # written in the unscaled frame
    (tmp_path / "POSCAR").write_text(
        f"t\n{-vol}\n2 0 0\n0 3 0\n0 0 4\nSi C\n1 1\nCartesian\n"
        + "".join(f"{x} {y} {z}\n" for x, y, z in cart)
    )
    got_lat, sym, got_frac = ch.read_poscar(tmp_path / "POSCAR")
    assert np.allclose(got_lat, 2 * lat) and sym == ["Si", "C"]
    assert np.allclose(got_frac, frac)
    (tmp_path / "P4").write_text("t\n1.0\n2 0 0\n0 3 0\n0 0 4\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n")
    with pytest.raises(ValueError, match="VASP 4"):
        ch.read_poscar(tmp_path / "P4")


def test_debye_constant():
    from akitsushima.mdspec.constants import DEBYE_TO_E_ANG

    assert DEBYE_TO_E_ANG == pytest.approx(0.20819434, rel=1e-7)
