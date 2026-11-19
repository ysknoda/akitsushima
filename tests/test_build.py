"""Structure builders (ADR-0035, decision 12).

1. bulk against ase.build.bulk and analytic nearest-neighbour distances.
2. supercell: counts, volume, no overlaps, energy per atom, ASE, min_width,
   qha's build_supercell unchanged.
3. substitute / vacancy: exact counts, seeds, uniform site frequencies.
4. slab against ase.build.surface; geometry; terminations; ESM checks.
5. rattle and strain.
6. CLI: template, repeat, reproducibility, JSON record.
"""

import json
import warnings

import numpy as np
import pytest

import akitsushima as ak
from akitsushima import build as B

A = {
    "diamond": 5.431,
    "zincblende": 5.653,
    "rocksalt": 4.212,
    "fcc": 4.05,
    "bcc": 2.87,
    "hcp": 3.21,
    "wurtzite": 3.25,
}
SP = {
    "diamond": "Si",
    "zincblende": ["Ga", "As"],
    "rocksalt": ["Mg", "O"],
    "fcc": "Al",
    "bcc": "Fe",
    "hcp": "Mg",
    "wurtzite": ["Zn", "O"],
}
ASE_NAME = {
    "diamond": "Si",
    "zincblende": "GaAs",
    "rocksalt": "MgO",
    "fcc": "Al",
    "bcc": "Fe",
    "hcp": "Mg",
    "wurtzite": "ZnO",
}


def _neigh_shells(lat, pos, types, n=16):
    """Per species: sorted distances to the n nearest neighbours of each
    atom (pooled, rounded), an orientation-free fingerprint."""
    lat, pos = np.asarray(lat), np.asarray(pos)
    sh = (
        np.array(
            [
                [i, j, k]
                for i in range(-2, 3)
                for j in range(-2, 3)
                for k in range(-2, 3)
            ]
        )
        @ lat
    )
    out = {}
    for sp in sorted(set(types)):
        d = []
        for i in np.flatnonzero(np.array(types) == sp):
            r = np.linalg.norm((pos[None, :, :] + sh[:, None, :]) - pos[i], axis=-1)
            r = np.sort(r.ravel())[1 : n + 1]
            d.append(r)
        out[sp] = np.round(np.sort(np.concatenate(d)), 8)
    return out


# ---------------------------------------------------------------- 1. bulk


@pytest.mark.parametrize("t", B.BULK_TYPES)
@pytest.mark.parametrize("cell", ["conventional", "primitive"])
def test_bulk_matches_ase(t, cell):
    ase_build = pytest.importorskip("ase.build")
    if t in ("hcp", "wurtzite") and cell == "conventional":
        cell = "primitive"
    s = B.bulk(t, SP[t], A[t], cell=cell)
    kw = {"cubic": True} if cell == "conventional" else {}
    if t in ("hcp", "wurtzite"):  # ASE's default c/a is the element's measured one
        kw["c"] = A[t] * np.sqrt(8 / 3)
    if t == "wurtzite":
        kw["u"] = 0.375
    ref = ase_build.bulk(ASE_NAME[t], t, a=A[t], **kw)
    assert s["natoms"] == len(ref)
    G = s["lattice"] @ s["lattice"].T
    np.testing.assert_allclose(G, ref.cell[:] @ ref.cell[:].T, atol=1e-10)
    a = _neigh_shells(s["lattice"], s["positions"], s["atom_types"])
    b = _neigh_shells(ref.cell[:], ref.positions, ref.get_chemical_symbols())
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=1e-7)


def test_bulk_nearest_neighbours_are_analytic():
    nn = {
        "diamond": np.sqrt(3) / 4,
        "zincblende": np.sqrt(3) / 4,
        "fcc": 1 / np.sqrt(2),
        "bcc": np.sqrt(3) / 2,
        "rocksalt": 0.5,
        "hcp": 1.0,
        "wurtzite": np.sqrt(3.0 / 8.0)
        * np.sqrt(8.0 / 3.0)
        * 0.375
        * np.sqrt(8.0 / 3.0),
    }
    for t in B.BULK_TYPES:
        for cell in ("conventional", "primitive"):
            if t in ("hcp", "wurtzite") and cell == "conventional":
                continue
            s = B.bulk(t, SP[t], 4.0, cell=cell)
            got = min(v for v in B.summary(s)["min_distance"].values() if v)
            # ideal wurtzite: bond u c = 3/8 sqrt(8/3) a
            want = 4.0 * (0.375 * np.sqrt(8 / 3) if t == "wurtzite" else nn[t])
            assert got == pytest.approx(want, rel=1e-12), (t, cell)
    with pytest.raises(ValueError, match="needs 2 species"):
        B.bulk("rocksalt", "Mg", 4.2)
    with pytest.raises(ValueError, match="type must be"):
        B.bulk("perovskite", "Sr", 3.9)


# ---------------------------------------------------------- 2. supercell


def _sw_si():
    from test_md_restart import SW_SI

    return ak.force_fn_from_config(SW_SI)


@pytest.mark.slow
@pytest.mark.parametrize(
    "M",
    [[2, 2, 2], [[1, 1, 0], [-1, 1, 0], [0, 0, 2]], [[2, 1, 0], [0, 2, 1], [1, 0, 2]]],
)
def test_supercell_counts_volume_and_energy(M):
    from conftest import randomly_perturbed

    base = B.supercell(B.bulk("diamond", "Si", 5.431), [2, 2, 2])  # wide enough for SW
    p = randomly_perturbed(base, magnitude=0.05, seed=2)
    big = B.supercell(p, M)
    det = int(round(np.linalg.det(np.asarray(M) if np.ndim(M) == 2 else np.diag(M))))
    assert big["natoms"] == det * 64
    vol = abs(np.linalg.det(big["lattice"]))
    assert vol == pytest.approx(det * abs(np.linalg.det(p["lattice"])), rel=1e-13)
    f = big["positions"] @ np.linalg.inv(big["lattice"])
    f = np.mod(f + 1e-9, 1.0)
    key = {tuple(np.round(x, 7) % 1.0) for x in f}
    assert len(key) == big["natoms"]  # no overlaps
    ff = _sw_si()
    e0 = ff(p["lattice"], p["positions"], p["atom_types"])[0] / 64
    e1 = ff(big["lattice"], big["positions"], big["atom_types"])[0] / big["natoms"]
    assert e1 == pytest.approx(e0, rel=1e-12)


def test_supercell_matches_ase_and_min_width():
    ase_build = pytest.importorskip("ase.build")
    prim = B.bulk("diamond", "Si", 5.431, cell="primitive")
    M = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]  # primitive fcc -> cubic
    s = B.supercell(prim, M)
    ref = ase_build.make_supercell(
        ak.to_ase_atoms(prim["lattice"], prim["positions"], prim["atom_types"]), M
    )
    np.testing.assert_allclose(s["lattice"], ref.cell[:], atol=1e-12)
    fa = np.round(np.mod(s["positions"] @ np.linalg.inv(s["lattice"]) + 1e-9, 1), 8)
    fb = np.round(np.mod(ref.get_scaled_positions() + 1e-9, 1), 8)
    assert {tuple(x) for x in fa} == {tuple(x) for x in fb}
    np.testing.assert_allclose(s["lattice"], np.eye(3) * 5.431, atol=1e-12)

    conv = B.bulk("diamond", "Si", 5.431)
    w = B.supercell(conv, min_width=12.0)
    assert np.allclose(np.diag(w["lattice"]), 3 * 5.431)  # 2 * 5.431 < 12
    hexa = B.bulk("hcp", "Mg", 3.21)
    w = B.supercell(hexa, min_width=10.0)
    widths = B._perp_widths(w["lattice"])
    assert np.all(widths >= 10.0)
    n = np.round(np.diag(w["lattice"] @ np.linalg.inv(hexa["lattice"]))).astype(int)
    for d in range(3):  # one fewer repetition would be too thin
        fewer = n.copy()
        fewer[d] -= 1
        if fewer[d] >= 1:
            thin = B._perp_widths(np.diag(fewer) @ hexa["lattice"])
            assert thin[d] < 10.0


def test_qha_build_supercell_is_unchanged():
    from akitsushima.solvers.qha import build_supercell

    prim = B.bulk("diamond", "Si", 5.431, cell="primitive")
    pos = prim["positions"] + 0.0137
    L, P, T = build_supercell(prim["lattice"], pos, prim["atom_types"], [3, 2, 4])
    # the former implementation, verbatim
    ref_pos, ref_t = [], []
    for i in range(3):
        for j in range(2):
            for k in range(4):
                ref_pos.append(np.asarray(pos) + np.array([i, j, k]) @ prim["lattice"])
                ref_t += list(prim["atom_types"])
    assert np.array_equal(L, np.diag([3, 2, 4]) @ prim["lattice"])
    assert np.array_equal(P, np.vstack(ref_pos)) and T == ref_t


# --------------------------------------------------- 3. substitute, vacancy


def test_substitute_counts_seed_and_uniformity():
    s = B.supercell(B.bulk("diamond", "Ge", 5.66), [2, 2, 2])
    a = B.substitute(s, "Ge", {"Sn": 8, "Si": 0.0625}, seed=3)
    assert a["atom_types"].count("Sn") == 8 and a["atom_types"].count("Si") == 4
    assert s["atom_types"].count("Ge") == 64  # input untouched
    b = B.substitute(s, "Ge", {"Sn": 8, "Si": 0.0625}, seed=3)
    assert a["atom_types"] == b["atom_types"]
    c = B.substitute(s, "Ge", {"Sn": 8, "Si": 0.0625}, seed=4)
    assert a["atom_types"] != c["atom_types"]
    # each site equally likely: chi-square over 2000 seeds
    counts = np.zeros(64)
    for seed in range(2000):
        t = B.substitute(s, "Ge", {"Sn": 8}, seed=seed)["atom_types"]
        counts += np.array(t) == "Sn"
    expected = 2000 * 8 / 64
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    from scipy.stats import chi2 as dist

    # 63 dof; the draws are not independent across sites (exactly 8 per
    # seed), which only makes the statistic smaller
    assert dist.sf(chi2, 63) > 1e-3, chi2
    d = B.substitute(s, None, "Sn", indices=[0, 5])
    assert [i for i, t in enumerate(d["atom_types"]) if t == "Sn"] == [0, 5]
    with pytest.raises(ValueError, match="seed is required"):
        B.substitute(s, "Ge", {"Sn": 8})
    with pytest.raises(ValueError, match="more than"):
        B.substitute(s, "Ge", {"Sn": 60, "Si": 10}, seed=1)


def test_vacancy_counts_seed_and_indices():
    s = B.supercell(B.bulk("rocksalt", ["Mg", "O"], 4.212), [2, 2, 2])
    v = B.vacancy(s, "O", 3, seed=1)
    assert v["natoms"] == 61 and v["atom_types"].count("O") == 29
    assert v["atom_types"] == B.vacancy(s, "O", 3, seed=1)["atom_types"]
    _, rec = B._vacancy(s, "O", 3, seed=1)
    keep = [i for i in range(64) if i not in rec["removed"]]
    np.testing.assert_array_equal(v["positions"], s["positions"][keep])
    w = B.vacancy(s, indices=[0, 1])
    np.testing.assert_array_equal(w["positions"], s["positions"][2:])


# ---------------------------------------------------------------- 4. slab


@pytest.mark.parametrize(
    "case",
    [
        ("diamond", (1, 0, 0)),
        ("diamond", (1, 1, 0)),
        ("diamond", (1, 1, 1)),
        ("wurtzite", (0, 0, 1)),
        ("wurtzite", (1, 0, 0)),
    ],
)
def test_slab_matches_ase_surface(case):
    ase_build = pytest.importorskip("ase.build")
    t, hkl = case
    s0 = B.bulk(t, SP[t], A[t])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # oblique a, b: the warning is tested below
        s = B.slab(s0, hkl, layers=3, vacuum=12.0)
    ref = ase_build.surface(
        ak.to_ase_atoms(s0["lattice"], s0["positions"], s0["atom_types"]),
        hkl,
        3,
        vacuum=6.0,
    )
    assert s["natoms"] == len(ref)
    area = np.linalg.norm(np.cross(s["lattice"][0], s["lattice"][1]))
    ref_area = np.linalg.norm(np.cross(ref.cell[0], ref.cell[1]))
    assert area == pytest.approx(ref_area, rel=1e-10)
    L = s["lattice"]
    assert L[0, 2] == 0 and L[1, 2] == 0 and L[2, 0] == 0 and L[2, 1] == 0
    z = s["positions"][:, 2]
    assert abs(z.min() + z.max()) < 1e-9  # centred on z = 0
    assert L[2, 2] == pytest.approx(z.max() - z.min() + 12.0, abs=1e-9)
    # the interior is bulk: the shortest distance is the bulk bond
    bulk_d = min(v for v in B.summary(s0)["min_distance"].values() if v)
    assert min(v for v in B.summary(s)["min_distance"].values() if v) == pytest.approx(
        bulk_d, rel=1e-9
    )


def test_slab_terminations_and_esm_cell():
    zno = B.bulk("wurtzite", ["Zn", "O"], 3.25, c=5.21, u=0.382)
    planes = B.slab_planes(zno, (0, 0, 0, 1))
    assert [list(p["composition"]) for p in planes] == [["Zn"], ["O"], ["Zn"], ["O"]]
    for k in range(4):
        s = B.slab(
            zno, (0, 0, 0, 1), layers=2, vacuum=10.0, termination=k, orthogonal_ab=True
        )
        top = s["atom_types"][int(np.argmax(s["positions"][:, 2]))]
        assert top == list(planes[k]["composition"])[0]
        L = s["lattice"]
        assert abs(L[0] @ L[1]) < 1e-9 and s["natoms"] == 16  # rectangle, 2 x area
    si = B.bulk("diamond", "Si", 5.431)
    assert len(B.slab_planes(si, (1, 0, 0))) == 4
    assert len(B.slab_planes(si, (1, 1, 1))) == 2
    with pytest.raises(ValueError, match="termination 5"):
        B.slab(si, (1, 0, 0), 2, 10.0, termination=5)
    mid = B.slab(si, (1, 0, 0), 2, 10.0, center="middle")
    z = mid["positions"][:, 2]
    assert (z.min() + z.max()) / 2 == pytest.approx(mid["lattice"][2, 2] / 2)

    # the rectangular (111) slab passes ESMCalculator's geometry checks
    s = B.slab(si, (1, 1, 1), layers=2, vacuum=15.0, orthogonal_ab=True)
    from akitsushima.engine.coulomb.esm import ESMCalculator

    ESMCalculator._validate_lattice(s["lattice"])  # raises on a bad cell
    with pytest.warns(UserWarning, match="not orthogonal"):
        B.slab(si, (1, 1, 1), layers=2, vacuum=15.0)
    L = s["lattice"]
    assert (
        abs(L[0] @ L[1]) < 1e-9
        and np.allclose(L[2, :2], 0)
        and np.allclose(L[:2, 2], 0)
    )


# ------------------------------------------------------ 5. rattle, strain


def test_rattle_and_strain():
    s = B.supercell(B.bulk("fcc", "Al", 4.05), [4, 4, 4])
    r = B.rattle(s, 0.05, seed=1)
    d = (r["positions"] - s["positions"]).ravel()
    assert np.std(d) == pytest.approx(0.05, rel=0.05)  # 768 samples, 2.6 % error
    assert abs(np.mean(d)) < 4 * 0.05 / np.sqrt(d.size)
    for kw in (
        {"volume": 0.03, "shear": 0.02, "seed": 2},
        {"volume": -0.06},
        {"epsilon": [[0.01, 0.002, 0], [0.002, -0.01, 0], [0, 0, 0.005]]},
    ):
        t = B.strain(s, **kw)
        V0, V1 = (abs(np.linalg.det(x["lattice"])) for x in (s, t))
        if "volume" in kw:
            assert V1 / V0 == pytest.approx(1 + kw["volume"], rel=1e-13)
        else:
            assert V1 / V0 == pytest.approx(np.linalg.det(np.eye(3) + kw["epsilon"]))
        f0 = s["positions"] @ np.linalg.inv(s["lattice"])
        f1 = t["positions"] @ np.linalg.inv(t["lattice"])
        np.testing.assert_allclose(f1, f0, atol=1e-13)
    sc = B.scale(s, volume=2 * abs(np.linalg.det(s["lattice"])))
    assert abs(np.linalg.det(sc["lattice"])) == pytest.approx(
        2 * abs(np.linalg.det(s["lattice"]))
    )
    with pytest.raises(ValueError, match="seed is required"):
        B.strain(s, shear=0.02)


# ------------------------------------------------------------------ 6. CLI


def test_build_cli_repeat_reproducible_and_record(tmp_path):
    from test_cli_smoke import _ns, _write_yaml

    from akitsushima.cli.cmd_build import cmd_build
    from akitsushima.io.structure import read_extxyz

    cfg = {
        "build": {
            "start": {"bulk": {"type": "diamond", "species": ["Ge"], "a": 5.7624}},
            "steps": [
                {"supercell": [2, 2, 2]},
                {"substitute": {"from": "Ge", "to": {"Sn": 8}, "seed": 1}},
                {"rattle": {"sigma": 0.05, "seed": 2}},
            ],
            "output": str(tmp_path / "a" / "gesn.extxyz"),
            "repeat": 3,
        }
    }
    (tmp_path / "a").mkdir()
    _write_yaml(tmp_path / "a.yaml", cfg)
    assert cmd_build(_ns(tmp_path / "a.yaml")) == 0
    frames = read_extxyz(tmp_path / "a" / "gesn.extxyz")
    assert len(frames) == 3
    assert frames[0]["atom_types"] != frames[1]["atom_types"]
    assert all(f["atom_types"].count("Sn") == 8 for f in frames)
    log = json.loads((tmp_path / "a" / "gesn_build.json").read_text())
    rec = log["structures"][1]["steps"]
    assert rec[1]["options"]["seed"] == 2 and rec[2]["options"]["seed"] == 3
    sn = [i for i, t in enumerate(frames[1]["atom_types"]) if t == "Sn"]
    assert rec[1]["indices"]["Sn"] == sn
    assert log["structures"][0]["composition"] == {"Ge": 56, "Sn": 8}

    # the same input gives the same bytes; POSCAR output gets numbered files
    first = (tmp_path / "a" / "gesn.extxyz").read_bytes()
    assert cmd_build(_ns(tmp_path / "a.yaml")) == 0
    assert (tmp_path / "a" / "gesn.extxyz").read_bytes() == first
    cfg["build"]["output"] = str(tmp_path / "a" / "POSCAR_g")
    _write_yaml(tmp_path / "b.yaml", cfg)
    assert cmd_build(_ns(tmp_path / "b.yaml")) == 0
    for k in range(3):
        s = ak.read_structure(str(tmp_path / "a" / f"POSCAR_g_{k:03d}"))
        assert s["natoms"] == 64
    assert (tmp_path / "a" / "POSCAR_g_build.json").exists()

    cfg["build"]["steps"] = [{"twist": {}}]
    _write_yaml(tmp_path / "c.yaml", cfg)
    assert cmd_build(_ns(tmp_path / "c.yaml")) == 1
