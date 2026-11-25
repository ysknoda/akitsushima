---
status: accepted (2026-11-25, 0.4.0a2)
---

# mdspec P2: trajectory formats, velocities, in-house Gamma modes, charge model auto

Second stage of the nozuchi port (ADR-0050 decisions 3, 5, 9, 10, 11).
It connects `mdspec` to akitsushima's own MD so that an MLIP or
classical run needs no DFT input.

## Decisions

1. **extxyz and LAMMPS dump** (mdspec/trajectory.py).
   - Streaming readers behind nozuchi's API (`read_header`,
     `iter_chunks`, `count_frames`), so the pipeline is unchanged.
   - The format comes from the name (.extxyz / .xyz, .lammpstrj /
     .dump), else from the content.
   - Cartesian frames become wrapped fractional coordinates.  LAMMPS
     tilted boxes follow the LAMMPS bound convention, atoms are sorted
     by id, and x / xu / xs columns are accepted.  The element comes
     from the `element` column or, by type, from `trajectory.species`.
   - Atoms of one species must be contiguous (POSCAR order, as
     akitsushima writes them): the header's species / counts and the
     per_atom indices depend on it.  An interleaved file is refused.
   - A truncated last frame is dropped with a warning, as for XDATCAR.
   - Each frame is parsed in Python per line.  That is fine for
     akitsushima MD sizes, but slower than the numba XDATCAR parser
     (UNCERTAIN for multi-GB files; numba parsing would be a later
     step).
2. **LAMMPS orientation.**
   - akitsushima writes dumps in the LAMMPS orientation (a along x), so
     their frames and the eps tensor are in that frame.
   - When Born tensors come from an OUTCAR, or Gamma modes from a file,
     for a unit cell whose orientation differs from the one implied by
     the dump (more than 1 %), the run is refused.  Otherwise the
     Cartesian inputs would be silently misaligned.
   - Point charges and the in-house modes (computed on the MD lattice)
     need no check.
3. **Velocities** (`md.traj_velocities: true`).
   - extxyz: the property `velocities:R:3`, in A/fs.
   - lammps-dump: `vx vy vz` in A/ps (metal units), rotated with the
     positions.
   - Refused with xdatcar.  Stored in the checkpoint settings when on.
   - The readers convert to A/fs; P4 (VDOS) uses them.
4. **Forces from the trajectory.**  `mdspec.forces: trajectory` takes
   the forces of the extxyz / LAMMPS-dump trajectory files, one force
   frame per trajectory frame, for the force-based static value.
5. **`dipole.model: auto`** (the default in the akitsushima YAML) takes,
   in this order (ADR-0050 decision 9):
   1. `born.born_charges` (born_tensor);
   2. the potential's point charges (Buckingham); `dipole.charges` is
      then ignored, with a warning;
   3. `mdspec.dipole.charges`.

   With none of them the run is refused.  The choice is logged.  Every
   point_charge / born_tensor run warns that electronic-polarisation
   fluctuations are not in M(t).
6. **Gamma modes from the potential** (mdspec/prepare.py), when a
   `potential:` block is present and no mode file is given.
   - With `modes: {unit_cell, supercell}`: the unit cell is the one
     implied by the MD lattice (inv(S) L_MD, so the MD volume and
     orientation), filled with the POSCAR's fractional positions.
   - Without them, and N <= 1000: the MD cell itself, at its mean
     positions (first 200 frames).  Above 1000 atoms the mode estimator
     is not offered (warned) and fsum / raw are used.
   - The ions are relaxed at fixed cell (`modes.relax: true`, default),
     on a supercell valid for the potential.  The Gamma matrix comes from
     `PhononCalculator` with `auto_supercell` (exact at Gamma,
     ADR-0049).
   - The masses are those of the run, so the projection is orthonormal.
   - The modes are written as `<prefix>_gamma_modes.npz` and
     `<prefix>_gamma_unit.vasp`, and `modes.file` points to them: the
     ported pipeline runs unchanged and the files can be inspected or
     reused.
   - `modes: off` disables the modes.  With no potential, nozuchi's
     behaviour is unchanged.
   - `auto_supercell` moved from cli/cmd_raman.py to solvers/raman.py
     (mdspec must not import the CLI).
7. **Paths.**  mdspec paths are read against the YAML file's directory;
   the potential section against the working directory, as in every
   command.

## Verification

- **Readers** (tests/mdspec/test_p2.py).
  - The same 300 frames of a sheared Mg / O cell as XDATCAR, extxyz and
    LAMMPS dump: fractional coordinates within 1e-7 (8 written
    decimals); the lattice metric is equal (the dump is rotated).
  - Frame counts, forces and velocities (A/ps -> A/fs) read back;
    several blocks per file; `max_frames` honoured.
  - Interleaved species are refused; a truncated last frame is dropped
    with a warning.
  - The whole analysis on the three files: eps_s equal to 1e-5 (the
    rounding of the written decimals, amplified by the fluctuation
    estimator).  For the dump, the tensor invariants agree.
- **Orientation guard.** Refused for a rotated unit cell; accepted for
  the dump's own orientation and for other formats.
- **Velocities.** A 72-atom AlN md with extxyz: the kinetic energy from
  the written velocities equals thermo.log's E_kin at the same steps
  (|diff| < 2e-6 eV).  `traj_velocities` with xdatcar is refused.
- **auto.**  The priority order, the override warning and the refusal.
- **In-house Gamma modes, end to end** (slow test).
  - Setup: Tersoff AlN (test_phonon_regressions._ALN), 3 x 3 x 2
    wurtzite (72 atoms), fictitious charges +-1, eps_inf = 1, Langevin
    30 K, 6 ps.
  - Paths: the modes from the unit cell (tiled) and from the MD cell
    itself.
  - Against the harmonic eps_0 of the same model (`dielectric_response`
    on the written unit cell), Delta eps agrees: force-based within 2 %,
    mode estimator within 5 %.
  - The two mode paths give the same force-based tensor: diagonals to
    3e-5, off-diagonals within 2e-4.
  - Above the size limit (patched to 10) the modes are not offered.
  - A longer by-hand run (20 ps, 30 K): harmonic eps_0 1.33253; force
    based 1.3333 (+0.06 %); mode estimator 1.3356 +- 0.0009 (+0.23 %).
- **Equivalence** with nozuchi 0.2.0a2 on its 58 validation inputs,
  rerun with this code: 58 / 58 bit for bit (largest relative
  difference 0.0); the XDATCAR path is unchanged.
- **Full suite.** 1362 passed, 1 skipped.
  - The first run had 2 failures: moving `auto_supercell` had left
    cmd_raman.py without its import (`raman` template and CLI test).
  - Fixed, and the suite was rerun on the final code.

## Self-critique

| value | source | confidence |
|---|---|---|
| N <= 1000 for the MD-cell modes | ADR-0046 eigh timings | ~80 % as a guide |
| first 200 frames for the mean positions | nozuchi's mapping default (mean_positions) | ~85 %; enough for a crystal; a diffusing system has no meaningful mean anyway |
| 1 % orientation tolerance | nozuchi's cell-mismatch warning | ~85 % |
| contiguous species only | akitsushima writes POSCAR order | > 95 % for akitsushima files; other writers (ASE) may interleave -- refused, not misread |

## Consequences

- An MLIP / classical MD in extxyz (with forces and velocities) is
  analysed with no DFT input: charges from the potential or the YAML,
  Gamma modes from the same potential.
- Next: P3 (nozuchi 0.3.0a1 items B1-B8), on the user's go-ahead.
