"""Round-trip tests for akitsushima.io (POSCAR / XDATCAR / EFS)."""
import numpy as np

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


def test_efs_read_applies_vasp_kb_conversion_with_correct_unit_and_sign():
    """Bug A/B regression: read_efs_file(..., stress_to_eV_Ang3=...) must
    apply BOTH the correct kB->eV/A^3 unit factor (1/1602.176621, not the
    eV/A^3->GPa factor 1/160.2176621 that a previous version of this
    codebase's own documentation mistakenly recommended -- a 10x error)
    AND the sign flip VASP's raw OUTCAR stress needs to match this
    codebase's tensile-positive convention. Exercises the exact
    ``constants.VASP_KB_TO_EV_ANG3`` preset this docstring fix points
    users at, since test_efs_roundtrip only ever exercises the default
    stress_to_eV_Ang3=1.0 path and never touches this factor at all."""
    import os
    import tempfile

    rng = np.random.default_rng(1)
    n_atoms = 3
    energy = -12.5
    forces = rng.uniform(-1, 1, size=(n_atoms, 3))

    # A known target stress, already in this codebase's eV/A^3
    # tensile-positive convention.
    target_stress_eV_ang3 = rng.uniform(-0.05, 0.05, size=(3, 3))
    target_stress_eV_ang3 = 0.5 * (target_stress_eV_ang3 + target_stress_eV_ang3.T)

    # What a real VASP OUTCAR "in kB" line would contain to represent
    # that same physical stress: invert the conversion the fixed
    # docstrings now document (raw_kB * VASP_KB_TO_EV_ANG3 == target).
    raw_vasp_kb_stress = target_stress_eV_ang3 / VASP_KB_TO_EV_ANG3

    fd, path = tempfile.mkstemp(suffix=".efs")
    os.close(fd)
    try:
        write_efs_file(path, energy, forces, raw_vasp_kb_stress, n_atoms=n_atoms)
        _, _, s_read = read_efs_file(path, n_atoms=n_atoms,
                                     stress_to_eV_Ang3=VASP_KB_TO_EV_ANG3)
        assert np.allclose(s_read, target_stress_eV_ang3, atol=1e-8), (
            "VASP_KB_TO_EV_ANG3 did not recover the correct tensile-positive "
            "eV/A^3 stress from a simulated raw VASP-kB input.")

        # The old, wrong factor (1/160.2176621, no sign flip) must NOT
        # match -- otherwise this test would not actually discriminate
        # the bug from the fix.
        old_wrong_factor = 1.0 / 160.2176621
        _, _, s_read_wrong = read_efs_file(path, n_atoms=n_atoms,
                                           stress_to_eV_Ang3=old_wrong_factor)
        assert not np.allclose(s_read_wrong, target_stress_eV_ang3, atol=1e-8)
    finally:
        os.remove(path)
