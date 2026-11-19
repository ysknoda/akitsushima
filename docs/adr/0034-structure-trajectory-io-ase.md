---
status: proposed (2026-11-19)
---

# Structure and trajectory I/O (extxyz, LAMMPS, CIF) and an ASE adapter

Tier A step 9 of ADR-0028 (Q14, Q17).  Agreed with the user on
2026-11-19 (grilling, Q1-Q12).

## Starting point (2026-11-19, 0.2.0a23)

- **Structure input.**  Only VASP POSCAR / CONTCAR (`io/poscar.py`).
  Every CLI command reads `structure: poscar:` and calls `_check_poscar`
  in `cli/common.py`.  `parse_poscar` returns a dict with `lattice`,
  `positions` (Cartesian), `atom_types`, `species`, `counts`, `natoms`
  and `fix_atoms` (Selective Dynamics).
- **Trajectory output.**  VASP XDATCAR (`io/xdatcar.py`) from md (NVT,
  Langevin, NVE, NPT: `XDATCAR_<task>`), relax (`XDATCAR_relax`) and
  gcmc.  nemd writes its own `trajectory_tc.xyz`.  The restart of
  ADR-0031 truncates the trajectory to the byte size recorded in the
  checkpoint, so any append-only format can be continued.
- **ASE.**  Used only by `tools/ase_efs.py` (the file route of the
  external engine, ADR-0011) and by tests.  It is not an extra in
  `pyproject.toml`.  ase 3.28 is installed in the development
  environment.
- **Training data.**  `build_training_set` reads `vasprun.xml` or
  `OUTCAR` (`reader: vasprun | outcar`).  Each frame is a dict with
  lattice, positions, atom types, energies, forces and stress.
- **Conventions.**  Stress is positive under compression in akitsushima
  (ADR-0001) and positive under tension in ASE; units eV/A^3 in both.

## Decisions

1. **Formats and implementation (Q1).**
   - extxyz (read and write), LAMMPS data (read and write) and LAMMPS
     dump (write) are implemented in akitsushima with numpy only.
   - CIF is read through ASE: expanding the symmetry operations, partial
     occupancies and the dialects of the format are easy to get wrong.
     CIF is written in P1 by akitsushima.
   - `ase` becomes an optional extra (`ase>=3.22`).  Only CIF reading
     and the calculator need it.
2. **Structure input in the CLI (Q2).**
   - `structure: {file: ..., format: auto, index: ...}`.  `auto` decides
     from the file name: POSCAR, CONTCAR, `*.vasp` -> poscar;
     `.xyz`, `.extxyz` -> extxyz; `.data`, `.lmp` -> lammps-data;
     `.cif` -> cif.
   - The `poscar:` key keeps working as an alias.
   - For a multi-frame file `index` selects the frame; the default is the
     last frame, as in ASE.
   - One `read_structure` serves every command.
   - Fixed atoms (Selective Dynamics) come from POSCAR only.  Output
     structures (`POSCAR_final_MD` etc.) stay POSCAR.
3. **Trajectory output (Q3).**  `traj_format: xdatcar | extxyz |
   lammps-dump`, one per run, default `xdatcar` (outputs unchanged), for
   md (NVT, Langevin, NVE, NPT), relax and gcmc.  nemd and neb keep their
   output.  All three formats are append-only, so the restart of ADR-0031
   applies unchanged.
4. **Content of a frame (Q4).**
   - extxyz: lattice, positions, species, energy, forces and stress.  The
     stress is written with ASE's sign (positive under tension), so that
     ASE reads it correctly.  No velocities (their units invite
     confusion).
   - LAMMPS dump: `id type element x y z fx fy fz`, metal units.  The
     cell is rotated to LAMMPS's restricted triclinic orientation (a
     along x, b in the xy plane); positions and forces are rotated with
     it.
   - The ensemble uncertainty is already in `thermo.log` (`sigF_max`)
     and is not repeated in the trajectory.
5. **LAMMPS data species (Q5).**
   - Reading: the element of each type comes from the comment after its
     mass (`1 72.63 # Ge`), else from the `species:` list (type order),
     else an error.  No guessing from the mass.
   - atom_style atomic and charge are read; charges are dropped with a
     warning.
   - Writing: atom_style atomic with the element comments; the cell is
     rotated to the LAMMPS orientation, so a written and re-read structure
     is the same crystal in another orientation (documented).
6. **ASE adapter (Q6).**
   - `ak.AkitsushimaCalculator(force_fn)`, a subclass of ASE's
     `Calculator`: energy, free_energy, forces and stress (Voigt, ASE
     sign).  pbc must be True in all three directions.
   - For an ensemble force field (ADR-0033) `results` also holds
     `sigma_E`, `sigma_F_max` and `sigma_S`.
   - `to_ase_atoms` and `from_ase_atoms` convert structures.
   - The reverse direction (an ASE calculator as an akitsushima force
     function) is not part of this step.
7. **Training data from extxyz (Q7).**  `reader: extxyz` in
   `build_training_set` reads energy, forces and stress labels into the
   same frame dicts as `read_vasprun`.  The keys default to ASE's
   (`energy`, `forces`, `stress`, ASE sign) and can be changed.
8. **Precision (Q8).**  Structure files: 14 decimals, as POSCAR.
   Trajectories: 8 decimals for positions and forces, as ASE; energy 10
   decimals, stress 10 significant digits.
9. **Validation (Q9).**  Tests that need ASE use `importorskip`.
   1. Own round trip of extxyz, LAMMPS data and CIF (P1) within 1e-12.
   2. Against ASE: own files read by `ase.io.read`, and ASE's files read
      by akitsushima, for extxyz, lammps-data and lammps-dump-text; CIF
      written by ASE read through `read_structure`.  Cell, positions,
      species, energy, forces and the stress sign are compared.
   3. A non-orthogonal GeSn cell rotated to the LAMMPS orientation: the
      lattice is lower triangular, distances and energy are unchanged,
      the forces rotate with the cell.
   4. Calculator: equals the force function; agrees with ASE's
      `calculate_numerical_forces` and `calculate_numerical_stress`
      (sign and Voigt order); for an ensemble the sigmas equal
      `uncertainty()`.
   5. An interrupted md run with `traj_format` extxyz and lammps-dump
      continues bit for bit.
   6. The CLI gives bit-identical md results from the same structure as
      POSCAR and as extxyz.
   7. A training set built with `reader: extxyz` from vasprun frames
      written as extxyz equals the one built with `reader: vasprun`.
   LAMMPS itself is not installed; ASE's LAMMPS readers stand in.
10. **File names (Q10).**  `traj_<task>.extxyz` and
    `traj_<task>.lammpstrj`; `XDATCAR_<task>` stays for xdatcar.
11. **No conversion command (Q11).**  `ak.read_structure` and
    `ak.write_structure` in the Python API only; the CLI stays YAML
    driven.
12. **Recording (Q12).**  This ADR is written before the implementation
    and accepted with the result.  ADR-0028 step 9 links here.  README and
    templates show `file`, `format`, `index` and `traj_format`.  The
    version goes to 0.2.0a24.
