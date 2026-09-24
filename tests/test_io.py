"""Round-trip tests for akitsushima.io (POSCAR / XDATCAR / EFS)."""
import numpy as np

import pytest

from akitsushima.constants import VASP_KB_TO_EV_ANG3
from akitsushima.io import parse_poscar, read_efs_file, write_efs_file, write_poscar
from akitsushima.io.xdatcar import _write_xdatcar_frame, _write_xdatcar_header


def test_poscar_roundtrip(tmp_path, fcc_al):
    path = tmp_path / "POSCAR"
    write_poscar(str(path), fcc_al["lattice"], fcc_al["positions"], fcc_al["atom_types"])
    parsed = parse_poscar(str(path))
    assert np.allclose(parsed["lattice"], fcc_al["lattice"], atol=1e-10)
    assert np.allclose(parsed["positions"], fcc_al["positions"], atol=1e-10)
    assert parsed["atom_types"] == fcc_al["atom_types"]
    assert parsed["natoms"] == 4
    assert not parsed["fix_atoms"].any()


def test_poscar_roundtrip_multi_species(tmp_path, rocksalt_nacl):
    path = tmp_path / "POSCAR"
    write_poscar(str(path), rocksalt_nacl["lattice"], rocksalt_nacl["positions"],
                 rocksalt_nacl["atom_types"])
    parsed = parse_poscar(str(path))
    assert parsed["species"] == ["Na", "Cl"]
    assert parsed["counts"] == [4, 4]
    assert np.allclose(parsed["positions"], rocksalt_nacl["positions"], atol=1e-10)


def test_poscar_roundtrip_interleaved_species_without_fix_atoms(tmp_path):
    """Bug fix (found in code-check): write_poscar's fix_atoms=None branch
    used to write the coordinate block in raw input order instead of
    grouping by species like the fix_atoms-given branch already did.
    POSCAR carries no per-atom species label -- a reader recovers species
    purely from position-in-file via the header's species/counts lines --
    so for INTERLEAVED atom_types (not already grouped by species) this
    silently produced a file that decodes to a different, wrong species-
    to-position assignment on re-read, with no crash or warning.

    This is directly reachable in practice: GCMCMDSimulation._accept_
    insertion appends each newly-inserted atom's species to the END of
    self.atom_types regardless of species, so a multi-species GCMC/OSMMC
    run's atom_types becomes interleaved after the first insertion of a
    species that is not already the group at the end of the array, and
    solvers/gcmc.py writes the final structure via
    write_poscar(..., self.atom_types) with no fix_atoms=. Reproduces
    that exact interleaving pattern directly (rather than running a full
    GCMC simulation) for a fast, deterministic regression test."""
    lattice = np.eye(3) * 10.0
    # Interleaved on purpose: Li, Li, O, O, Li (a 3rd Li "inserted" after
    # the O block, exactly what _accept_insertion would produce).
    atom_types = ["Li", "Li", "O", "O", "Li"]
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0], [4.0, 0.0, 0.0],
    ])
    li_positions = {tuple(positions[i]) for i, a in enumerate(atom_types) if a == "Li"}
    o_positions  = {tuple(positions[i]) for i, a in enumerate(atom_types) if a == "O"}

    path = tmp_path / "POSCAR"
    write_poscar(str(path), lattice, positions, atom_types)   # fix_atoms=None
    parsed = parse_poscar(str(path))

    assert parsed["species"] == ["Li", "O"]
    assert parsed["counts"] == [3, 2]
    # Every position the file re-labels as "Li" must be one of the
    # ORIGINAL Li positions (and likewise for "O") -- this is the actual
    # correctness property a POSCAR file must have, independent of
    # exactly which order same-species atoms end up listed in.
    reread_li = {tuple(parsed["positions"][i])
                 for i, a in enumerate(parsed["atom_types"]) if a == "Li"}
    reread_o = {tuple(parsed["positions"][i])
                for i, a in enumerate(parsed["atom_types"]) if a == "O"}
    assert reread_li == li_positions
    assert reread_o == o_positions


def test_poscar_selective_dynamics_roundtrip(tmp_path, diamond_si):
    path = tmp_path / "POSCAR"
    fix_atoms = np.zeros((8, 3), dtype=bool)
    fix_atoms[0] = True          # freeze all 3 DoF of atom 0
    fix_atoms[1, 2] = True       # freeze only z of atom 1
    write_poscar(str(path), diamond_si["lattice"], diamond_si["positions"],
                 diamond_si["atom_types"], fix_atoms=fix_atoms)
    parsed = parse_poscar(str(path))
    assert np.array_equal(parsed["fix_atoms"], fix_atoms)
    assert np.allclose(parsed["positions"], diamond_si["positions"], atol=1e-10)


def test_xdatcar_frames(tmp_path, fcc_al):
    path = tmp_path / "XDATCAR"
    with open(path, "w") as fh:
        _write_xdatcar_header(fh, fcc_al["lattice"], fcc_al["atom_types"])
        _write_xdatcar_frame(fh, fcc_al["positions"], fcc_al["lattice"],
                              fcc_al["atom_types"], frame_idx=1)
        _write_xdatcar_frame(fh, fcc_al["positions"] + 0.01, fcc_al["lattice"],
                              fcc_al["atom_types"], frame_idx=2)
    text = path.read_text()
    assert text.count("Direct configuration=") == 2
    assert "Al" in text.splitlines()[5]


def test_xdatcar_frame_groups_interleaved_species_to_match_header(tmp_path):
    """Bug fix (found in code-check): _write_xdatcar_frame's coordinate
    block used to be written in raw input order while the header
    (species_order/counts) assumes species-grouped order -- silently
    wrong for interleaved atom_types, exactly like the write_poscar bug
    this mirrors. Reachable via GCMCMDSimulation/OSMMCSimulation, whose
    _accept_insertion appends each newly-inserted atom's species to the
    END of self.atom_types regardless of species, and which write every
    trajectory frame with variable_header=True (see solvers/gcmc.py)."""
    lattice = np.eye(3) * 10.0
    atom_types = ["Li", "Li", "O", "O", "Li"]   # interleaved (GCMC-style)
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0], [4.0, 0.0, 0.0],
    ])
    li_positions = {tuple(positions[i]) for i, a in enumerate(atom_types) if a == "Li"}
    o_positions  = {tuple(positions[i]) for i, a in enumerate(atom_types) if a == "O"}

    path = tmp_path / "XDATCAR"
    with open(path, "w") as fh:
        _write_xdatcar_frame(fh, positions, lattice, atom_types, frame_idx=1,
                              variable_header=True, comment="interleaved test")

    # A single-frame XDATCAR block has the same header+coordinate layout
    # as a POSCAR, so parse_poscar can read it back directly.
    parsed = parse_poscar(str(path))
    assert parsed["species"] == ["Li", "O"]
    assert parsed["counts"] == [3, 2]
    reread_li = {tuple(parsed["positions"][i])
                 for i, a in enumerate(parsed["atom_types"]) if a == "Li"}
    reread_o = {tuple(parsed["positions"][i])
                for i, a in enumerate(parsed["atom_types"]) if a == "O"}
    assert reread_li == li_positions
    assert reread_o == o_positions


def test_efs_roundtrip():
    import tempfile
    import os
    rng = np.random.default_rng(0)
    n_atoms = 5
    energy = -42.123456        # total, eV
    forces = rng.uniform(-2, 2, size=(n_atoms, 3))
    stress = rng.uniform(-0.1, 0.1, size=(3, 3))
    stress = 0.5 * (stress + stress.T)   # symmetrize, matches Voigt round-trip

    fd, path = tempfile.mkstemp(suffix=".efs")
    os.close(fd)
    try:
        write_efs_file(path, energy, forces, stress, n_atoms=n_atoms)
        e_read, f_read, s_read = read_efs_file(path, n_atoms=n_atoms)
        assert abs(e_read - energy) < 1e-6
        assert np.allclose(f_read, forces, atol=1e-6)
        assert np.allclose(s_read, stress, atol=1e-6)
    finally:
        os.remove(path)


def test_efs_read_applies_the_vasp_kb_unit_factor():
    """read_efs_file(..., stress_to_eV_Ang3=VASP_KB_TO_EV_ANG3) must use
    the kbar factor 1/1602.176621 -- not 1/160.2176621, the eV/A^3->GPa
    factor an earlier version of this codebase recommended (10x off).

    This test is about MAGNITUDE only.  Its predecessor also claimed to
    pin the sign, but it built its "raw VASP" input as
    ``target / VASP_KB_TO_EV_ANG3`` and checked that multiplying by
    ``VASP_KB_TO_EV_ANG3`` gave ``target`` back -- true for either sign,
    so it passed with the wrong one.  The sign needs an independent
    oracle; see test_vasp_kb_sign_matches_ases_production_reader below.
    """
    import os
    import tempfile

    rng = np.random.default_rng(1)
    n_atoms = 3
    raw_kb = rng.uniform(-50.0, 50.0, size=(3, 3))
    raw_kb = 0.5 * (raw_kb + raw_kb.T)

    fd, path = tempfile.mkstemp(suffix=".efs")
    os.close(fd)
    try:
        write_efs_file(path, -12.5, rng.uniform(-1, 1, (n_atoms, 3)), raw_kb,
                       n_atoms=n_atoms)
        _, _, s_read = read_efs_file(path, n_atoms=n_atoms,
                                     stress_to_eV_Ang3=VASP_KB_TO_EV_ANG3)
        assert np.allclose(np.abs(s_read), np.abs(raw_kb) / 1602.176621,
                           rtol=1e-9)
        assert not np.allclose(np.abs(s_read), np.abs(raw_kb) / 160.2176621,
                               rtol=1e-3)
    finally:
        os.remove(path)


def test_vasp_kb_sign_matches_ases_production_reader():
    """VASP_KB_TO_EV_ANG3 must turn a raw OUTCAR "in kB" line into THIS
    engine's stress convention -- checked end to end against ASE, which
    is independent of this package and whose VASP reader is production
    code used far more widely than anything here.

    The chain, each link measured rather than argued:

      1. ASE's own ``Vasp.read_stress`` converts a raw "in kB" line to
         ASE's stress convention.
      2. ASE's own LennardJones calculator and this package's
         LennardJonesPotential, on the same compressed FCC cell, give
         stresses of opposite sign and equal magnitude.  So ASE's
         convention is the negative of the engine's.
      3. Therefore the raw line, converted to the ENGINE's convention,
         must equal minus ASE's reading of it.

    Discriminating power: the previous value of the constant,
    -1/1602.176621, fails link 3 -- asserted explicitly below.  That
    value came from reading this codebase's former label for its own
    convention, "tensile-positive", in ASE's sense; the label was wrong
    (CONTEXT.md, "Compressive-positive").
    """
    ase = pytest.importorskip("ase")
    from ase.build import bulk
    from ase.calculators.lj import LennardJones
    from ase.calculators.vasp import Vasp

    from akitsushima.constants import EV_ANG3_TO_KBAR
    from akitsushima.engine import LennardJonesPotential, PotentialEngine
    from akitsushima.fitting import make_force_fn

    # -- link 2: the engine's convention is the negative of ASE's --------
    eps, sig, rc = 0.0104, 3.40, 8.5
    fn = make_force_fn(PotentialEngine().add(LennardJonesPotential(
        {frozenset({"Ar"}): {"epsilon": eps, "sigma": sig}},
        r_cut=rc, shift=False)))
    atoms = bulk("Ar", "fcc", a=5.26 * 0.95, cubic=True).repeat((4, 4, 4))
    atoms.calc = LennardJones(epsilon=eps, sigma=sig, rc=rc, smooth=False)
    s_ase = atoms.get_stress(voigt=False)
    _, _, s_eng = fn(np.array(atoms.cell), atoms.get_positions(),
                     ["Ar"] * len(atoms))
    assert s_eng[0, 0] > 0.0, "a compressed cell must be positive here"
    assert np.allclose(s_eng, -s_ase, rtol=1e-6, atol=1e-12)

    # -- link 1 + 3: raw "in kB" line through ASE's own reader -----------
    raw = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])   # VASP: xx yy zz xy yz zx
    line = "  in kB  " + "".join(f"{v:12.5f}" for v in raw) + "\n"
    ase_voigt = Vasp().read_stress(lines=[line])        # ASE: xx yy zz yz xz xy
    ours = raw[[0, 1, 2, 4, 5, 3]] * VASP_KB_TO_EV_ANG3  # same order as ASE's
    assert np.allclose(ours, -ase_voigt, rtol=1e-9)

    # The former constant fails the same check.
    old = raw[[0, 1, 2, 4, 5, 3]] * (-1.0 / EV_ANG3_TO_KBAR)
    assert not np.allclose(old, -ase_voigt, rtol=1e-3)
