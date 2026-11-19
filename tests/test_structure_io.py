"""Structure and trajectory I/O, ASE adapter (ADR-0034, decision 9).

1. Own round trips: extxyz, LAMMPS data, CIF (P1).
2. Against ASE, both directions: extxyz, lammps-data, lammps-dump-text,
   CIF.
3. The LAMMPS orientation of a non-orthogonal GeSn cell: a rotation that
   leaves distances and the energy unchanged and turns forces and stress
   with the cell.
4. AkitsushimaCalculator: equals the force function, matches ASE's
   numerical forces and stress, carries an ensemble's sigma.
5. md restart with extxyz / lammps-dump trajectories (bit for bit).
6. The CLI gives the same md run from POSCAR and from extxyz.
7. A training set from extxyz equals the one from vasprun.xml.
"""

import os
import warnings

import numpy as np
import pytest

import akitsushima as ak
from akitsushima.io.structure import (
    lammps_orientation,
    read_extxyz,
    read_lammps_data,
    read_structure,
    write_cif,
    write_extxyz,
    write_lammps_data,
    write_structure,
)
from akitsushima.io.trajectory import TrajectoryWriter, traj_filename

LAT = np.array([[5.6, 0.3, 0.1], [1.2, 5.9, -0.2], [0.4, 0.7, 6.1]])
TYPES = ["Sn", "Ge", "Ge", "Sn", "Ge", "Ge", "Ge", "Ge"]


def _sample(seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.random((8, 3)) @ LAT
    F = rng.normal(size=(8, 3))
    S = rng.normal(size=(3, 3))
    return pos, F, 0.5 * (S + S.T)


def _gram(L):
    return L @ L.T


# --------------------------------------------------------------- 1. own


def test_extxyz_round_trip_and_frame_selection(tmp_path):
    pos, F, S = _sample()
    p = tmp_path / "a.extxyz"
    write_extxyz(p, LAT, pos, TYPES, energy=-12.25, forces=F, stress=S)
    write_extxyz(p, LAT, pos + 0.1, TYPES, append=True)
    fr = read_extxyz(p)
    assert len(fr) == 2
    np.testing.assert_allclose(fr[0]["lattice"], LAT, atol=1e-13)
    np.testing.assert_allclose(fr[0]["positions"], pos, atol=1e-13)
    np.testing.assert_allclose(fr[0]["forces"], F, atol=1e-13)
    np.testing.assert_allclose(fr[0]["stress"], S, atol=1e-14)  # akitsushima sign
    assert fr[0]["energy"] == -12.25 and fr[0]["atom_types"] == TYPES
    assert "energy" not in fr[1] and "forces" not in fr[1]
    last = read_structure(str(p))  # default: the last frame, as ASE
    np.testing.assert_allclose(last["positions"], pos + 0.1, atol=1e-13)
    first = read_structure(str(p), index=0)
    np.testing.assert_allclose(first["positions"], pos, atol=1e-13)
    assert first["species"] == ["Sn", "Ge"] and first["counts"] == [2, 6]
    with pytest.raises(ValueError, match="2 frame"):
        read_structure(str(p), index=5)


def test_extxyz_refuses_non_periodic_frames(tmp_path):
    p = tmp_path / "m.xyz"
    p.write_text("1\nProperties=species:S:1:pos:R:3\nSi 0 0 0\n")
    with pytest.raises(ValueError, match="no Lattice"):
        read_extxyz(p)
    p.write_text('1\nLattice="5 0 0 0 5 0 0 0 5" pbc="T T F"\nSi 0 0 0\n')
    with pytest.raises(ValueError, match="periodic"):
        read_extxyz(p)


def test_lammps_data_round_trip_and_species(tmp_path):
    pos, _, _ = _sample()
    p = tmp_path / "a.data"
    write_lammps_data(p, LAT, pos, TYPES)
    d = read_lammps_data(str(p))
    Lp, R = lammps_orientation(LAT)
    np.testing.assert_allclose(d["lattice"], Lp, atol=1e-13)
    np.testing.assert_allclose(d["positions"], pos @ R, atol=1e-13)
    assert d["atom_types"] == TYPES
    text = p.read_text()
    assert "1 72.63" in text and "# Ge" in text and "Atoms # atomic" in text
    # types by atomic number: Ge (32) is 1, Sn (50) is 2
    assert text.splitlines()[-8].split()[1] == "2"

    # no element comments: species from the caller, never from the mass
    bare = text.replace(" # Ge", "").replace(" # Sn", "")
    (tmp_path / "b.data").write_text(bare)
    with pytest.raises(ValueError, match="not guessed"):
        read_lammps_data(str(tmp_path / "b.data"))
    d2 = read_structure(str(tmp_path / "b.data"), species=["Ge", "Sn"])
    assert d2["atom_types"] == TYPES

    # atom_style charge: read, charges dropped with a warning
    lines = text.splitlines()
    k = lines.index("Atoms # atomic")
    charged = (
        lines[:k]
        + ["Atoms # charge"]
        + lines[k + 1 : k + 2]
        + [
            f"{r.split()[0]} {r.split()[1]} 0.5 {' '.join(r.split()[2:])}"
            for r in lines[k + 2 :]
        ]
    )
    (tmp_path / "c.data").write_text("\n".join(charged) + "\n")
    with pytest.warns(UserWarning, match="charges are not"):
        d3 = read_lammps_data(str(tmp_path / "c.data"))
    np.testing.assert_allclose(d3["positions"], d["positions"], atol=0)

    # a left-handed cell cannot be written
    with pytest.raises(ValueError, match="left-handed"):
        write_lammps_data(tmp_path / "x.data", LAT[[1, 0, 2]], pos, TYPES)


def test_cif_round_trip(tmp_path):
    pytest.importorskip("ase")
    pos, _, _ = _sample()
    p = tmp_path / "a.cif"
    write_cif(p, LAT, pos, TYPES)
    s = read_structure(str(p))
    # a CIF keeps the cell parameters and fractional coordinates
    np.testing.assert_allclose(_gram(s["lattice"]), _gram(LAT), atol=1e-12)
    frac = s["positions"] @ np.linalg.inv(s["lattice"])
    np.testing.assert_allclose(frac, pos @ np.linalg.inv(LAT), atol=1e-12)
    assert s["atom_types"] == TYPES


def test_format_detection_and_dispatch(tmp_path):
    pos, _, _ = _sample()
    for name in ("POSCAR_x", "a.vasp", "a.extxyz", "a.xyz", "a.lmp", "a.data"):
        write_structure(str(tmp_path / name), LAT, pos, TYPES)
        s = read_structure(str(tmp_path / name))
        assert s["natoms"] == 8 and sorted(s["atom_types"]) == sorted(TYPES)
    with pytest.raises(ValueError, match="cannot tell"):
        read_structure(str(tmp_path / "a.vasp").replace("a.vasp", "weird.txt"))
    with pytest.raises(ValueError, match="one structure"):
        read_structure(str(tmp_path / "a.lmp"), index=2)
    with pytest.warns(UserWarning, match="POSCAR only"):
        write_structure(
            str(tmp_path / "f.extxyz"), LAT, pos, TYPES, fix_atoms=np.ones((8, 3), bool)
        )


# --------------------------------------------------------------- 2. ASE


def test_extxyz_against_ase_both_ways(tmp_path):
    ase_io = pytest.importorskip("ase.io")
    pos, F, S = _sample()
    write_extxyz(
        tmp_path / "a.extxyz", LAT, pos, TYPES, energy=-3.5, forces=F, stress=S
    )
    a = ase_io.read(tmp_path / "a.extxyz")
    np.testing.assert_allclose(a.cell[:], LAT, atol=1e-13)
    np.testing.assert_allclose(a.positions, pos, atol=1e-13)
    assert a.get_chemical_symbols() == TYPES
    assert a.get_potential_energy() == pytest.approx(-3.5, abs=1e-12)
    np.testing.assert_allclose(a.get_forces(), F, atol=1e-13)
    # ASE's sign: positive under tension
    np.testing.assert_allclose(a.get_stress(voigt=False), -S, atol=1e-14)

    ase_io.write(tmp_path / "b.extxyz", a)
    fr = read_extxyz(tmp_path / "b.extxyz")[0]
    np.testing.assert_allclose(fr["positions"], pos, atol=1e-7)  # ASE: 8 decimals
    np.testing.assert_allclose(fr["forces"], F, atol=1e-7)
    np.testing.assert_allclose(fr["stress"], S, atol=1e-12)
    assert fr["energy"] == pytest.approx(-3.5) and fr["atom_types"] == TYPES


def test_lammps_against_ase_both_ways(tmp_path):
    ase_io = pytest.importorskip("ase.io")
    pos, F, _ = _sample()
    Lp, R = lammps_orientation(LAT)
    write_lammps_data(tmp_path / "a.data", LAT, pos, TYPES)
    b = ase_io.read(tmp_path / "a.data", format="lammps-data", atom_style="atomic")
    np.testing.assert_allclose(b.cell[:], Lp, atol=1e-12)
    np.testing.assert_allclose(b.positions, pos @ R, atol=1e-12)
    assert b.get_chemical_symbols() == TYPES

    from akitsushima.io.ase_adapter import to_ase_atoms

    ase_io.write(
        tmp_path / "c.data",
        to_ase_atoms(LAT, pos, TYPES),
        format="lammps-data",
        masses=True,
        specorder=["Ge", "Sn"],
    )
    c = read_lammps_data(str(tmp_path / "c.data"))
    assert c["atom_types"] == TYPES
    np.testing.assert_allclose(_gram(c["lattice"]), _gram(LAT), atol=1e-10)
    frac = c["positions"] @ np.linalg.inv(c["lattice"])
    d = frac - pos @ np.linalg.inv(LAT)
    np.testing.assert_allclose(d - np.round(d), 0, atol=1e-10)

    with open(tmp_path / "t.lammpstrj", "w") as fh:
        w = TrajectoryWriter(fh, "lammps-dump")
        w.frame(10, LAT, pos, TYPES, forces=F)
        w.frame(20, LAT, pos + 0.05, TYPES, forces=2 * F)
    frames = ase_io.read(tmp_path / "t.lammpstrj", index=":", format="lammps-dump-text")
    assert len(frames) == 2
    f = frames[1]
    np.testing.assert_allclose(f.cell[:], Lp, atol=1e-7)
    np.testing.assert_allclose(f.positions, (pos + 0.05) @ R, atol=1e-7)
    np.testing.assert_allclose(f.get_forces(), 2 * F @ R, atol=1e-7)
    assert f.get_chemical_symbols() == TYPES


def test_cif_from_ase_and_partial_occupancy(tmp_path):
    ase_io = pytest.importorskip("ase.io")
    from ase.build import bulk

    si = bulk("Si", "diamond", a=5.431, cubic=True)
    ase_io.write(tmp_path / "si.cif", si)
    s = read_structure(str(tmp_path / "si.cif"))
    assert s["natoms"] == 8 and set(s["atom_types"]) == {"Si"}
    np.testing.assert_allclose(_gram(s["lattice"]), _gram(si.cell[:]), atol=1e-10)

    (tmp_path / "mix.cif").write_text(
        "data_mix\n_cell_length_a 5.6\n_cell_length_b 5.6\n_cell_length_c 5.6\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'P 1'\nloop_\n_atom_site_label\n"
        "_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
        "_atom_site_fract_z\n_atom_site_occupancy\n"
        "A1 Ge 0 0 0 0.5\nA2 Sn 0 0 0 0.5\nB1 Ge 0.5 0.5 0.5 1.0\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="partially occupied"):
            read_structure(str(tmp_path / "mix.cif"))


# --------------------------------------------------- 3. LAMMPS orientation


def _gesn_sheared():
    from conftest import diamond_si_dict, randomly_perturbed
    from test_sw_three_body_counting import _tomita

    s = randomly_perturbed(diamond_si_dict(2), magnitude=0.08, seed=5)
    shear = np.array([[1.0, 0.0, 0.0], [0.12, 1.0, 0.0], [-0.07, 0.09, 1.0]])
    f = 5.9 / 5.431
    lat = np.asarray(s["lattice"]) @ shear.T * f
    pos = np.asarray(s["positions"]) @ shear.T * f
    types = list(np.random.default_rng(3).choice(["Ge", "Sn"], len(pos)))
    ff = ak.make_force_fn(
        ak.PotentialEngine().add(ak.StillingerWeberPotential(_tomita(), species=["Ge", "Sn"]))
    )
    return lat, pos, types, ff


def test_lammps_orientation_is_a_rotation_of_a_gesn_cell():
    lat, pos, types, ff = _gesn_sheared()
    assert abs(lat[0, 1]) + abs(lat[0, 2]) + abs(lat[1, 2]) > 0.5  # not LAMMPS form
    Lp, R = lammps_orientation(lat)
    assert Lp[0, 1] == Lp[0, 2] == Lp[1, 2] == 0.0
    assert Lp[0, 0] > 0 and Lp[1, 1] > 0 and Lp[2, 2] > 0
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(lat @ R, Lp, atol=1e-12)
    pr = pos @ R
    d0 = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    d1 = np.linalg.norm(pr[:, None] - pr[None], axis=-1)
    np.testing.assert_allclose(d1, d0, atol=1e-12)
    E0, F0, S0 = ff(lat, pos, types)
    E1, F1, S1 = ff(Lp, pr, types)
    assert E1 == pytest.approx(E0, rel=1e-12, abs=1e-10)
    np.testing.assert_allclose(F1, F0 @ R, atol=1e-10)
    np.testing.assert_allclose(S1, R.T @ S0 @ R, atol=1e-12)
    assert np.abs(F0).max() > 0.1  # a non-trivial check


# ---------------------------------------------------------- 4. calculator


@pytest.mark.slow
def test_calculator_matches_force_fn_and_finite_differences():
    pytest.importorskip("ase")
    from ase.calculators.fd import (
        calculate_numerical_forces,
        calculate_numerical_stress,
    )

    lat, pos, types, ff = _gesn_sheared()
    atoms = ak.to_ase_atoms(lat, pos, types, calculator=ak.AkitsushimaCalculator(ff))
    E, F, S = ff(lat, pos, types)
    assert atoms.get_potential_energy() == E
    assert atoms.get_potential_energy(force_consistent=True) == E
    np.testing.assert_array_equal(atoms.get_forces(), F)
    st = atoms.get_stress()  # Voigt xx yy zz yz xz xy, ASE sign
    np.testing.assert_array_equal(
        st, -np.array([S[0, 0], S[1, 1], S[2, 2], S[1, 2], S[0, 2], S[0, 1]])
    )
    np.testing.assert_allclose(
        calculate_numerical_forces(atoms, 1e-4), F, rtol=0, atol=2e-6
    )
    np.testing.assert_allclose(
        calculate_numerical_stress(atoms, 1e-5), st, rtol=0, atol=2e-7
    )
    assert np.abs(st).max() > 1e-3
    back = ak.from_ase_atoms(atoms)
    np.testing.assert_allclose(back["positions"], pos, atol=1e-13)  # after FD moves
    assert back["atom_types"] == types

    atoms.pbc = [True, True, False]
    with pytest.raises(ValueError, match="periodic"):
        atoms.get_potential_energy()


@pytest.mark.slow
def test_calculator_carries_the_ensemble_spread():
    pytest.importorskip("ase")
    from test_ensemble_elm import _fit
    from test_mlip_atom_centred import _random_structs

    from akitsushima.fitting.mlip_engine import MLIPForceField

    ff = MLIPForceField(_fit(n=2))
    s = _random_structs(3, 5)[0]
    calc = ak.AkitsushimaCalculator(ff)
    atoms = ak.to_ase_atoms(
        s["lattice"], s["positions"], s["atom_types"], calculator=calc
    )
    atoms.get_potential_energy()
    u = ff.uncertainty()
    assert u is not None
    for k, v in u.items():
        assert calc.results[k] == v
    single = ak.AkitsushimaCalculator(MLIPForceField(ff.model.members[0]))
    atoms.calc = single
    atoms.get_potential_energy()
    assert "sigma_E" not in single.results


# ------------------------------------------------- 5. trajectories, restart


@pytest.mark.slow
@pytest.mark.parametrize("fmt", ["extxyz", "lammps-dump"])
def test_md_restart_with_trajectory_format(tmp_path, fmt):
    from test_md_restart import EVERY, STOP_AT, _kill_after, _Killed, _make

    ref = _make("nvt", tmp_path / "ref", traj_format=fmt).run()
    out = tmp_path / "ck"
    ck = str(out / "md_checkpoint.npz")
    sim = _make("nvt", out, traj_format=fmt)
    _kill_after(sim, STOP_AT + 1)
    with pytest.raises(_Killed):
        sim.run(checkpoint=ck, checkpoint_every=EVERY)
    got = _make("nvt", out, traj_format=fmt).run(checkpoint=ck, checkpoint_every=EVERY)
    np.testing.assert_array_equal(got["positions"], ref["positions"])
    name = traj_filename("nvt", fmt)
    a = (tmp_path / "ref" / name).read_bytes()
    assert a and a == (out / name).read_bytes()
    assert not (out / "XDATCAR_nvt").exists()
    # a checkpoint of another format is refused
    with pytest.raises(ValueError, match="traj_format"):
        _make("nvt", out, traj_format="xdatcar").run(checkpoint=ck)


@pytest.mark.slow
def test_trajectory_frames_hold_the_frame_energy_forces_and_stress(tmp_path):
    from test_md_restart import SW_SI, _make

    _make("npt", tmp_path, n_steps=10, traj_format="extxyz").run()
    frames = read_extxyz(tmp_path / "traj_npt.extxyz")
    assert [int(f["info"]["step"]) for f in frames] == [20, 25]
    ff = ak.force_fn_from_config(SW_SI)
    for f in frames:
        E, F, S = ff(f["lattice"], f["positions"], f["atom_types"])
        # positions are written with 8 decimals
        assert f["energy"] == pytest.approx(E, abs=1e-6)
        np.testing.assert_allclose(f["forces"], F, atol=1e-6)
        np.testing.assert_allclose(f["stress"], S, atol=1e-8)


@pytest.mark.slow
def test_relax_and_gcmc_write_the_chosen_format(tmp_path):
    from conftest import diamond_si_dict, randomly_perturbed
    from test_md_restart import SW_SI

    s = randomly_perturbed(diamond_si_dict(2), magnitude=0.05, seed=1)
    ak.GeometryOptimiser(
        s["lattice"],
        s["positions"],
        s["atom_types"],
        force_fn=ak.force_fn_from_config(SW_SI),
        method="fire",
        max_steps=6,
        output_dir=str(tmp_path / "r"),
        verbose=False,
        traj_format="extxyz",
    ).run()
    frames = read_extxyz(tmp_path / "r" / "traj_relax.extxyz")
    log = (tmp_path / "r" / "relax.log").read_text().splitlines()[1:]
    rows = [r for r in log if r.split() and r.split()[0].isdigit()]
    assert len(frames) == len(rows)
    for f, r in zip(frames, rows):
        assert f["energy"] == pytest.approx(float(r.split()[1]), abs=1e-6)
    assert not (tmp_path / "r" / "XDATCAR_relax").exists()

    from test_cli_templates import _write_ar_gas

    _write_ar_gas(tmp_path / "POSCAR_ar")
    ar = read_structure(str(tmp_path / "POSCAR_ar"))
    lj = ak.force_fn_from_config(
        {"type": "LJ", "species": ["Ar"],
         "params": {"epsilon": 0.01032, "sigma": 3.405, "r_cut": 10.0}}
    )
    ak.GCMCMDSimulation(
        ar["lattice"], ar["positions"], ar["atom_types"], force_fn=lj,
        species="Ar", mu=-0.1, temperature=300.0, n_cycles=4,
        md_steps_per_cycle=2, gcmc_attempts_per_cycle=10, traj_every=1,
        output_dir=str(tmp_path / "g"), verbose=False, traj_format="lammps-dump",
    ).run()
    text = (tmp_path / "g" / "traj_gcmc.lammpstrj").read_text()
    assert text.count("ITEM: TIMESTEP") >= 4 and " Ar " in text


# ------------------------------------------------------------------ 6. CLI


def _exact_si(path_poscar, path_xyz):
    """Diamond Si at a = 5.375 (= 43/8, exact in binary): fractional and
    Cartesian coordinates are exact in both files, so both inputs give
    the same floats."""
    from conftest import diamond_si_dict

    d = diamond_si_dict(2)
    lat = np.eye(3) * 10.75
    frac = np.asarray(d["positions"]) @ np.linalg.inv(np.asarray(d["lattice"]))
    frac = np.round(frac * 8) / 8
    pos = frac @ lat
    ak.write_poscar(str(path_poscar), lat, pos, d["atom_types"])
    write_structure(str(path_xyz), lat, pos, d["atom_types"])


@pytest.mark.slow
def test_cli_md_from_extxyz_equals_poscar(tmp_path):
    from test_cli_smoke import _ns, _write_yaml
    from test_md_restart import SW_SI

    from akitsushima.cli.cmd_md import cmd_md

    _exact_si(tmp_path / "POSCAR", tmp_path / "si.extxyz")
    a = read_structure(str(tmp_path / "POSCAR"))
    b = read_structure(str(tmp_path / "si.extxyz"))
    np.testing.assert_array_equal(a["positions"], b["positions"])
    np.testing.assert_array_equal(a["lattice"], b["lattice"])
    outs = {}
    for name, st in (
        ("p", {"poscar": str(tmp_path / "POSCAR")}),
        ("x", {"file": str(tmp_path / "si.extxyz"), "format": "auto"}),
    ):
        out = tmp_path / name
        cfg = {
            "structure": st,
            "potential": SW_SI,
            "md": {
                "ensemble": "nvt",
                "temperature": 600.0,
                "timestep": 1.0,
                "n_steps": 30,
                "equil_steps": 0,
                "velocity_rescale_steps": 0,
                "thermo_every": 5,
                "traj_every": 10,
                "output_dir": str(out),
                "traj_format": "extxyz",
                "save_plots": False,
            },
        }
        _write_yaml(tmp_path / f"{name}.yaml", cfg)
        assert cmd_md(_ns(tmp_path / f"{name}.yaml")) == 0
        outs[name] = out
    for f in ("thermo.log", "POSCAR_final_MD", "traj_nvt.extxyz"):
        assert (outs["p"] / f).read_bytes() == (outs["x"] / f).read_bytes(), f


def test_cli_structure_errors(tmp_path, capsys):
    from akitsushima.cli.common import _check_structure

    pos, _, _ = _sample()
    write_structure(str(tmp_path / "a.extxyz"), LAT, pos, TYPES)
    assert _check_structure({"file": str(tmp_path / "a.extxyz")})["natoms"] == 8
    assert _check_structure({"file": "x", "poscar": "y"}) is None
    assert "not both" in capsys.readouterr().out
    assert _check_structure({"file": str(tmp_path / "missing.cif")}) is None
    assert "not found" in capsys.readouterr().out
    (tmp_path / "s.txt").write_text("x")
    assert _check_structure({"file": str(tmp_path / "s.txt")}) is None
    assert "cannot tell" in capsys.readouterr().out
    # `poscar:` stays POSCAR whatever the name, unless format says otherwise
    assert (
        _check_structure({"poscar": str(tmp_path / "a.extxyz"), "format": "extxyz"})[
            "natoms"
        ]
        == 8
    )


# ------------------------------------------------------- 7. training data


def test_training_set_from_extxyz_equals_vasprun(tmp_path):
    from test_vasp_dataset import _runs

    from akitsushima.io import build_training_set, load_training_set, read_vasprun

    _runs(tmp_path / "runs", n_runs=4)
    kw = dict(
        isolated_atom_energy={"Ge": -0.75, "Sn": -0.6}, test_fraction=0.25, seed=1
    )
    ref = build_training_set(
        str(tmp_path / "runs"), output_dir=str(tmp_path / "v"), **kw
    )
    xyz = tmp_path / "xyz"
    xyz.mkdir()
    for run in sorted(os.listdir(tmp_path / "runs")):
        frames = read_vasprun(str(tmp_path / "runs" / run / "vasprun.xml"))
        for k, f in enumerate(frames):
            write_extxyz(
                xyz / f"{run}.extxyz",
                f["lattice"],
                f["positions"],
                f["atom_types"],
                energy=f["energy_wo_entropy"],
                forces=f["forces"],
                stress=f["stress"],
                append=k > 0,
            )
    got = build_training_set(
        str(xyz), output_dir=str(tmp_path / "x"), reader="extxyz", **kw
    )
    assert got["is_test"] == ref["is_test"]
    assert (got["n_train"], got["n_test"]) == (ref["n_train"], ref["n_test"])
    for split in ("train", "test"):
        a = load_training_set(str(tmp_path / "v" / split / "structures.csv"))
        b = load_training_set(str(tmp_path / "x" / split / "structures.csv"))
        assert len(a) == len(b) > 0
        for sa, sb in zip(a, b):
            assert sa["atom_types"] == sb["atom_types"]
            for key in ("lattice", "positions"):
                np.testing.assert_allclose(sb[key], sa[key], rtol=0, atol=1e-12)
            for key in ("energy", "forces", "stress"):
                if sa.get(key) is not None:
                    np.testing.assert_allclose(sb[key], sa[key], rtol=1e-12, atol=1e-12)

    # a frame without forces is refused
    (xyz / "zz.extxyz").write_text(
        '1\nLattice="5 0 0 0 5 0 0 0 5" energy=-1.0\nGe 0 0 0\n'
    )
    with pytest.raises(ValueError, match="no forces"):
        build_training_set(
            str(xyz),
            output_dir=str(tmp_path / "y"),
            reader="extxyz",
            isolated_atom_energy={"Ge": -0.75, "Sn": -0.6},
        )
