---
status: accepted (2026-11-19; implemented in 0.2.0a24 -- see Result)
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

## Result (2026-11-19, 0.2.0a24)

**Implementation.**
- `io/structure.py`:
  - `read_structure` / `write_structure` (exported as
    `ak.read_structure`, `ak.write_structure`) dispatch on the format;
    every reader returns the dict of `parse_poscar`.
  - extxyz: `read_extxyz` (all frames; Lattice required, pbc must be
    T T T; stress from 6 or 9 values or a `virial`, returned in
    akitsushima's sign) and `write_extxyz`.
  - LAMMPS data: `read_lammps_data` (atomic or charge; the style from the
    `Atoms # <style>` comment, `atom_style`, or 5 / 8 columns) and
    `write_lammps_data` (types by atomic number unless `species` is given,
    `# <element>` after each mass).
  - `lammps_orientation(lattice)` returns the restricted-triclinic cell and
    the rotation R (`r -> r @ R`); a left-handed cell is refused.
  - CIF: read through `ase.io.read`, with partially occupied or mixed sites
    refused; `write_cif` writes P1.
- `io/trajectory.py`: `TrajectoryWriter` (xdatcar, extxyz, lammps-dump)
  and `traj_filename`.  md (`_BaseMDSimulation._traj_frame`), relax
  (`_log_row`) and gcmc / osmc write through it.
  - md and relax frames carry the energy, forces and stress of the
    frame's positions, which the force call of that step already has.
  - gcmc frames carry the energy only.  osmc frames leave out the
    fractional atom and therefore its U.
  - The LAMMPS type map of gcmc includes the inserted species from the
    first frame.
  - The checkpoint settings carry `traj_format` only when it is not
    xdatcar, so checkpoints of 0.2.0a21-a23 still match.
- `io/ase_adapter.py`: `to_ase_atoms`, `from_ase_atoms` and
  `AkitsushimaCalculator`.  The calculator class is built on first access,
  and the three names are lazy attributes of `akitsushima`.  They are left
  out of `__all__`, so that `from akitsushima import *` does not need ASE.
- CLI: `_check_structure` reads `structure: {file | poscar, format, index,
  species}` for every single-structure command.
  - `file` without `format` uses auto detection.
  - `poscar` keeps meaning POSCAR whatever the name, so existing inputs
    with names like `Si.pos` still work.
  - `file` and `poscar` together are refused.
  - md, relax, gcmc and osmc read `traj_format`.
  - The templates show `traj_format: xdatcar` and a commented `file` /
    `format` / `index`.
- `build_training_set(reader="extxyz", extxyz_keys=...)`: a source file,
  or each `*.xyz` / `*.extxyz` of a source directory, is one run.  Frames
  without an energy or forces label are refused.  `energy` and
  `cross_check` do not apply.
- `pyproject.toml`: extra `ase = ["ase>=3.22"]`, included in `all`.

**Validation** (`tests/test_structure_io.py`, 18 tests; ASE 3.28).
1. Own round trips are within 1e-13: extxyz (energy, forces, stress
   sign, multi-frame `index`), LAMMPS data (in the LAMMPS orientation,
   with species from the comments or the caller, and the charge style)
   and CIF (cell Gram matrix and fractional coordinates).
2. ASE in both directions:
   - extxyz: ASE reads akitsushima's file exactly, and akitsushima reads
     ASE's file within ASE's 8 decimals.
   - lammps-data: both directions agree within 1e-10.
   - lammps-dump-text: ASE reads the frames, cell, rotated positions,
     forces and elements.
   - CIF: a cell written by ASE is read back.
3. A sheared 64-atom GeSn cell (Tomita SW):
   - Lp is lower-triangular and R is a proper rotation (R R^T = I to
     1e-14).
   - Distances agree to 1e-12 and E to 1e-12 relative.
   - F becomes F R (1e-10) and S becomes R^T S R (1e-12).
4. Calculator:
   - The energy, forces and stress equal the force function's
     exactly.
   - They agree with ASE's `calculate_numerical_forces` (eps 1e-4, 2e-6
     eV/A) and `calculate_numerical_stress` (eps 1e-5, 2e-7 eV/A^3).
   - For an ensemble, `results` holds the sigmas of `uncertainty()`.
   - A pbc of F F T is refused.
5. NVT with extxyz and lammps-dump trajectories, interrupted and
   restarted, gives the same positions and a byte-identical trajectory.
   A restart with another `traj_format` is refused.
6. The CLI md run from a POSCAR and from an extxyz of the same exactly
   representable structure (a = 5.375 A) writes byte-identical
   `thermo.log`, `POSCAR_final_MD` and trajectory.
7. A training set built from extxyz files written from the vasprun.xml
   frames equals the vasprun one: the same split, and lattice,
   positions, E, F and S within 1e-12.

Also checked by hand: in each md ensemble (NVT, Langevin, NVE, NPT), the
last extxyz frame matches the force function evaluated on that frame's
positions.  The differences (E 5e-8 eV, F 2e-7 eV/A, S 6e-10 eV/A^3)
come from writing the positions with 8 decimals.

LAMMPS itself was not run; ASE's readers stand in for it (decision 9).
A LAMMPS data file is written without reducing the tilt factors, so a
strongly sheared cell may need `box tilt large` in older LAMMPS.

Full suite: 910 passed, 1 skipped (14.9 min).
