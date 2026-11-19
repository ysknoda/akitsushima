---
status: proposed (2026-11-19)
---

# Structure builders: bulk, supercell, alloy, vacancy, slab, perturbations

Tier A step 10 of ADR-0028 (Q16; SQS stays tier C, Q19).  Agreed with
the user on 2026-11-19 (grilling, Q1-Q14).

## Starting point (2026-11-19, 0.2.0a24)

- No general structure builder.  `solvers/qha.py` has `build_supercell`
  (diagonal repetition, cell-major order: atom k of cell c is c * N + k);
  `PhononCalculator` builds its own supercell.
- The GeSn work built structures with one-off scripts:
  `example/GeSn/tutorial/make_structure.py` (diamond cell by hand, exact
  Sn count, Vegard lattice constant) and
  `example/GeSn/dft_requests/b_round0/make_round0.py` (rattle, volume
  strain and random shear for training data).
- `ESMCalculator` needs c along z, a and b orthogonal, and the slab
  centred on z = 0 (`engine/coulomb/esm.py`).
- ADR-0034 gives `read_structure` / `write_structure` for POSCAR, extxyz,
  LAMMPS data and CIF; a structure is the dict of `parse_poscar`.

## Decisions

1. **Interface (Q1).**  Python API plus a CLI command `akitsushima build
   build.yaml` that writes a structure file (any format of
   `write_structure`).  The `structure:` section of other commands takes
   no transformations: each command keeps a concrete file as input.
2. **Operations (Q2).**  bulk, supercell, substitute (random alloy),
   vacancy, slab, rattle, strain, scale.  No interstitials; the
   point-defect workflow stays tier B.
3. **Random alloy (Q3).**
   - The number of substituted atoms is exact: `substitute: {from: Ge,
     to: {Sn: 8}, seed}`; a fraction gives round(x * N_from), and the
     composition it gives is reported.
   - Several targets in one step are allowed.
   - `seed` is required.
   - No automatic Vegard scaling: `bulk` takes `a` and `scale` adjusts
     the cell; the template shows a Vegard example.
4. **Slab (Q4, Q12).**
   - General (hkl) with integer in-plane bases (extended Euclid).
   - `layers` counts repeat units along the normal (the thickness is
     reported).
   - Atomic planes of one repeat unit (heights within 0.01 A along the
     normal) are listed with their composition; `termination: k` puts
     plane k at the top (default 0).
   - `vacuum` is the total vacuum thickness [A].
   - The output cell has c along z, perpendicular to a and b;
     `orthogonal_ab: true` searches a rectangular in-plane cell, with a
     warning when a and b are not orthogonal and it is not set.
   - `center: zero` (default; slab centred on z = 0, as ESM needs) or
     `middle` (z = Lz / 2).
   - No polarity correction, no automatic symmetric slab.
5. **Vacancy (Q5).**  Random `count` of a `species` (seed required) or
   explicit `indices`.  No charge compensation.
6. **Supercell (Q6).**
   - A diagonal `[n1, n2, n3]` or a general 3x3 integer matrix (lattice
     points of the new cell enumerated).
   - `min_width: w` picks the smallest diagonal repetition whose
     perpendicular widths are all >= w.
   - Cell-major order as in qha; qha's `build_supercell` is replaced by
     it with bit-identical results.
7. **YAML and repetition (Q7).**

   ```yaml
   build:
     start: {bulk: {...}}      # or {file: path}
     steps: [{supercell: ...}, {substitute: ...}, {rattle: ...}]
     output: POSCAR_GeSn       # format from the name, or format:
     repeat: 1                 # n > 1: seeds + k, k = 0..n-1
   ```

   With `repeat: n` every seed is offset by k.  An extxyz output holds the
   n structures as frames; other formats get numbered files
   (`POSCAR_GeSn_000`, ...).
8. **API (Q8).**  `akitsushima/build.py`, exported as `ak.build`:
   `bulk`, `supercell`, `substitute`, `vacancy`, `slab`, `rattle`,
   `strain`, `scale`.  Each takes the structure dict and returns a new one
   (the input is not modified).  numpy only.
9. **bulk (Q9).**
   - Types: diamond, fcc, bcc, hcp (one species); zincblende, wurtzite,
     rocksalt (two).
   - `cell: conventional` (default) or `primitive`; hcp and wurtzite are
     hexagonal only.
   - `a` [A] required; `c` defaults to the ideal c/a = sqrt(8/3) for hcp
     and wurtzite, `u` to 3/8 for wurtzite.
   - For two species the first sits on the origin site; the order given
     is kept.
10. **rattle, strain, scale (Q10).**
    - rattle: N(0, sigma^2) on every Cartesian component, seed required.
    - strain: a symmetric `epsilon`, or `volume` (relative change) with
      `shear` (off-diagonal components from U(-shear, shear), seed
      required); the cell becomes L (I + eps)^T and fractional
      coordinates are kept.
    - scale: `factor` (isotropic, on the lattice) or a target `volume`.
11. **Record (Q11).**  `<output>_build.json` beside the output: start,
    steps, seeds, chosen indices, final composition, volume and the
    shortest distance per species pair.  The console shows composition,
    atom count and shortest distance.  No distance threshold warning.
12. **Validation (Q13).**  ASE-based tests use `importorskip`.
    1. bulk against `ase.build.bulk` for all types, conventional and
       primitive (Gram matrix, positions modulo the lattice), and
       nearest-neighbour distances against the analytic values.
    2. supercell: atom count det(M) N, volume ratio, no overlaps, SW
       energy per atom equal to the cell's within 1e-12 for diagonal and
       non-diagonal M, the same set as `ase.build.make_supercell`,
       `min_width` minimal, qha results unchanged bit for bit.
    3. substitute and vacancy: exact counts, seed reproducibility,
       uniform site frequencies over 2000 seeds (chi-square), explicit
       indices.
    4. slab against `ase.build.surface` for Si (100), (110), (111) and
       wurtzite ZnO (0001), (10-10): atoms per repeat unit and in-plane
       area; c perpendicular to a and b; interior nearest-neighbour
       distances equal to bulk; the number of terminations;
       `orthogonal_ab` gives a.b = 0 and passes ESMCalculator's checks.
    5. rattle standard deviation, strain volume and kept fractional
       coordinates.
    6. CLI: the template runs; `repeat: 3` gives three distinct extxyz
       frames; the same input gives byte-identical output; the JSON
       record.
    7. Full suite.
13. **Recording (Q14).**  This ADR is written before the implementation
    and accepted with the result.  ADR-0028 step 10 links here.  README
    section and `--dump-default build` template (GeSn with a Vegard
    lattice constant).  The version goes to 0.2.0a25.
