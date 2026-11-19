---
status: accepted (2026-11-19; implemented in 0.2.0a25 -- see Result)
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

## Result (2026-11-19, 0.2.0a25)

**Implementation.**
- `akitsushima/build.py` (`ak.build`):
  - `bulk`, `supercell`, `substitute`, `vacancy`, `slab`, `slab_planes`,
    `rattle`, `strain`, `scale` and `summary`.
  - Private variants (`_supercell`, `_substitute`, ...) also return what
    they chose; the CLI records it.
- **supercell.**
  - A diagonal matrix keeps the loop and arithmetic of the former
    `qha.build_supercell`, which is now a wrapper of it.  Its output is
    byte-identical to the former implementation.
  - A general matrix enumerates the lattice points n with n M^-1 in
    [0, 1)^3 and wraps the atoms into the new cell.
- **slab.**
  - Integer column reduction of (h, k, l) gives a unimodular W.  Its first
    two columns span the plane; the third has hkl . v3 = 1.
  - The in-plane pair is Lagrange-Gauss reduced, and v3 is shifted by
    integer in-plane vectors towards the normal.
  - The re-based cell is turned to the LAMMPS orientation of ADR-0034, so
    the normal is z and one repeat unit is d_hkl high.
  - Atomic planes are the heights within 0.01 A, periodic in d_hkl.
  - The slab keeps the atoms of the stacked repeat units whose height lies
    in (z_k - layers d, z_k], which puts plane k on top.
  - c becomes (0, 0, thickness + vacuum).
  - `orthogonal_ab` picks the smallest-area rectangular combination of
    a and b, with coefficients up to 8 (a vectorised search).
  - Four-index (h k i l) input is accepted.
- **summary.**  Shortest distances per species pair through a k-d tree
  over the 27 neighbouring images of the wrapped cell.
- **CLI.**  `akitsushima build` (`cli/cmd_build.py`) and the template
  `--dump-default build`: a 64-atom Ge56Sn8 cell with the Vegard constant
  5.7624 A from a(Ge) 5.658 A and a(alpha-Sn) 6.493 A.
  - `start` is `{bulk: ...}`, `{file: path}` or `{file: {path, format,
    index, species}}`.
  - A slab step prints its planes and marks the top one.

**Validation** (`tests/test_build.py`, 30 tests; ASE 3.28).
1. bulk, all seven types, conventional and primitive:
   - The Gram matrix and the per-species distances to the 16 nearest
     neighbours equal those of `ase.build.bulk`.  ASE gets c and u
     explicitly, since it defaults to measured c/a values.
   - The shortest distances equal the analytic values (a sqrt(3)/4,
     a/sqrt(2), a sqrt(3)/2, a/2, a, and 3/8 c for ideal wurtzite) to
     1e-12.
2. supercell:
   - With diag(2, 2, 2) and two non-diagonal matrices (det 2 and 9) on a
     perturbed 64-atom Si cell, the atom counts and volumes are right and
     no atoms overlap.
   - The SW energy per atom equals the cell's to 1e-12.
   - The primitive-to-cubic matrix gives `ase.build.make_supercell`'s
     cell and fractional positions.
   - `min_width` is minimal in each direction.
   - qha's wrapper equals the former loop bit for bit.
3. substitute and vacancy:
   - The counts are exact, and several targets get disjoint sites.
   - The same seed gives the same structure and a different seed a
     different one.
   - Over 2000 seeds the per-site Sn frequencies pass chi-square (63
     dof, p > 1e-3).
   - Explicit indices and the errors (no seed, too many sites) work.
4. slab:
   - Si (100), (110), (111) and wurtzite ZnO (0001), (10-10), 3 layers:
     the atom count and in-plane area equal `ase.build.surface`'s.
   - c is along z with a and b in the xy plane, the slab is centred on
     z = 0, and Lz = thickness + vacuum.
   - The shortest distance in the slab equals the bulk bond.
   - ZnO (0001) has 4 planes (Zn, O, Zn, O), and `termination: k` puts
     plane k's species on top.  Si has 4 planes in (100) and 2 in (111).
   - The rectangular Si(111) cell passes
     `ESMCalculator._validate_lattice`.
   - An oblique cell without `orthogonal_ab` warns.
5. rattle and strain:
   - rattle: the standard deviation of 768 displacements is within 5 %
     of sigma, with a mean near zero.
   - strain: the volume is exact to 1e-13 for a volume change, with or
     without shear, and equals det(I + eps) for an explicit epsilon.
     Fractional coordinates are kept to 1e-13.
6. CLI:
   - `repeat: 3` to extxyz gives three frames with different Sn sites.
     The JSON holds seeds offset by k, the chosen indices and the
     composition.
   - A rerun gives byte-identical output.
   - POSCAR output gets `_000` to `_002`.
   - An unknown step is refused.
   - The template runs as shipped (`test_cli_templates`).

The template's Vegard constant is an approximation, as its comment says.
Whether, and by how much, GeSn deviates from Vegard's law was not checked
here (UNCERTAIN).

Full suite: 941 passed, 1 skipped (14.3 min).
