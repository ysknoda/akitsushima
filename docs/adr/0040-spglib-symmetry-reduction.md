---
status: proposed (2026-11-22)
---

# Symmetry reduction of phonon displacements and elastic strains (spglib)

Tier B item 5 of ADR-0028 (addendum 5), re-scoped by the user on
2026-11-22.  Item 5 had three parts: phonopy force-constant output,
spglib reduction of displacements and strains, and LAMMPS potential
files.  The user judged phonopy output and LAMMPS potential files not
urgent and moved both to tier C (ADR-0028 addendum 6).  This ADR covers
only the spglib reduction.  Agreed with the user on 2026-11-22
(grilling, two rounds; round 2, Q1-Q8, all as recommended).

## Purpose

- **Fewer force calls for ordered crystals.**
  - Phonons: 12 force calls become 1 for diamond Si; AlN wurtzite drops
    from 24.
  - Elastic constants: fewer strained cells, each of which needs an
    ionic relaxation.
  - The saving is largest where one force call is expensive: an MLIP on
    a large supercell, or an external DFT engine.
- **Prerequisite of tier B item 6.**  The third-order force constants of
  item 6 need pairs of displacements, which are impractical without
  symmetry.  The operations, the atom mapping and the reconstruction
  built here are reused there.
- **Not a gain for the random alloys.**  A GeSn random cell has no
  symmetry beyond the identity, so its cost does not change.  This is
  why the user ranked the item's direct value as moderate.

## Starting point (2026-11-22, 0.3.0a3)

- **`PhononCalculator`:**
  - Each atom of the origin cell is displaced by +delta and -delta along
    x, y and z: 6 N_basis force calls.  The other columns of the IFC are
    filled by lattice translation.
  - Afterwards: IFC <- (IFC + IFC^T) / 2, the acoustic sum rule, and the
    D-ASR correction at Gamma.
  - The supercell is diagonal (`supercell` int or 3 ints).  Its atoms are
    ordered cell-major: lattice point outer, basis atom inner.
  - The displacement tasks are split across MPI ranks, and within a rank
    across `n_jobs` threads.
  - No symmetry is used anywhere.
- **Users of `PhononCalculator`:**
  - `phonon`.
  - `qha` and `gruneisen`, through `solvers.qha.phonons_at`.
  - `kappa`, on its Slack / Debye path.
  - `dielectric`, through `phonons_at`, for LO-TO.
- **`ElasticConstantsCalculator`:**
  - The unstrained cell, plus each of the 6 Voigt modes at +-each
    amplitude of `strains` (default 0.0025, 0.005, 0.01): 37 strained
    cells.
  - With more than one atom, each strained cell has its ions relaxed
    (`relax_ions`).
  - Each column of C (and of C_clamped and of the internal strain) is a
    straight-line least-squares fit over its own mode's points.
  - C is then symmetrised, (C + C^T) / 2.
  - `elastic` is the only CLI user; `piezo` and the Born charges of
    `dielectric` apply their own strains and displacements.
- **Environment:** spglib 2.7.0 and phonopy 4.1.0 are installed but are
  not in the extras of `pyproject.toml`.

## Decisions

1. **Scope and record (Q1, Q7).**
   - `PhononCalculator` and `ElasticConstantsCalculator` get `symmetry`
     (default False) and `symprec` (default 1e-5 A).
     `solvers.qha.phonons_at` passes both through.
   - The CLI key goes into the sections of `phonon`, `qha`,
     `gruneisen`, `kappa` (the Slack section) and `elastic`.
   - Not in this item, recorded only: the Born charges of `dielectric`
     (and its LO-TO phonons, which keep the full method) and the
     strains of `piezo`.  Both have their own displacement schemes.
   - The move of phonopy output and LAMMPS potential files to tier C is
     recorded in ADR-0028 (tier C list and addendum 6), in the same
     commit as this ADR.
2. **Own implementation on spglib (Q2).**
   - spglib only supplies the space-group operations of the input cell
     (`spglib.get_symmetry`).
   - The following are written in this package:
     - the choice of displacements and strains,
     - the rotation and permutation of forces and stresses,
     - the reconstruction of the IFC and of C.
   - phonopy is used in the tests only, as an independent reference
     (skipped when it is not installed).
3. **Off by default (Q3).**
   - `symmetry: false` is the default.  The present path is then
     untouched, which keeps ADR-0028 decision 5 (1e-10).
   - With `symmetry: true` the results differ from the full method:
     - With Cartesian displacements related by cubic operations, they
       agree to round-off.
     - With rotations by 60 or 120 degrees, the central differences of
       the rotated directions carry a different O(delta^2) error, so a
       relative difference of order 1e-4 is expected.
4. **Symmetry operations (Q6).**
   - Input to spglib:
     - the cell, the fractional positions, and one number per distinct
       (species, mass) pair, so that unequal masses of one species break
       the symmetry;
     - `symprec` in A, default 1e-5 (spglib's default).
   - The structure is not symmetrised.  Positions that are only nearly
     symmetric give a lower group: fewer savings, but no wrong result.
   - The phonon method keeps only the operations whose rotation maps the
     supercell lattice onto itself.
   - Each operation gives a permutation of the supercell atoms, found by
     matching positions (minimum image, tolerance `symprec`).  An
     operation that does not map every atom onto an atom is an error, not
     a skip.
   - The log and the results file record:
     - the space group (international symbol and number);
     - the number of operations, and of those kept;
     - the number of force calls or strained cells, against the full
       method.
   - `symmetry: true` without spglib stops with
     `pip install akitsushima[symmetry]`.
   - `pyproject.toml` gets the extra `symmetry = ["spglib>=2.0"]`, and
     `all` includes it.
   - `symmetry: true` with ESM electrostatics is refused, because the
     boundary condition need not share the operations that act on z.
     Elastic already refuses ESM.
5. **Phonon displacements (Q4).**
   - **Which atoms:** the origin-cell atoms are grouped into orbits under
     the operations.  Only one representative per orbit is displaced.
   - **Which directions, per representative:**
     - The Cartesian axes x, y and z are tried in that order.
     - A direction is kept when its images under the site-symmetry
       operations raise the rank of the set of directions.
     - The search stops at rank 3.
   - **Signs:**
     - -u is not computed when it is the image of +u under a
       site-symmetry operation; its forces then follow from that
       operation exactly.
     - Otherwise both +u and -u are computed.
     - Either way each kept direction gives the same central difference
       (F(+u) - F(-u)) / 2.
   - **Reconstruction:**
     - Each operation maps a displacement u of atom a to R u of atom
       pi(a), with forces F'_{pi(i)} = R F_i.
     - Collecting the images gives, for every origin-cell atom, at least
       three independent displacement directions with their force
       differences.
     - Its 3 x 3 N_sc column block of the IFC is the least-squares
       solution.
     - The other columns follow by lattice translation, as now.
     - The symmetrisation, the acoustic sum rule and D-ASR stay as they
       are.
   - **Identity only:** when the operations are the identity alone (and
     no -u is an image), the present code path is taken.  The result is
     then identical to `symmetry: false` bit for bit.
   - **Parallel work:** the reduced displacement list is split across
     MPI ranks and `n_jobs` threads as the full list is now.
   - **Expected counts:**
     - diamond Si: 12 -> 1 (one orbit, site symmetry -43m, -x an image
       of +x);
     - AlN wurtzite: 24 -> between 4 and 8, depending on whether -x is an
       image at the site 3m; the measured count goes into the result.
6. **Elastic strains (Q5).**
   - **Strain images:** an operation R maps the strain eps to
     R eps R^T, and the stress sigma to R sigma R^T.  This holds for the
     clamped-ion stress and for the relaxed-ion stress.
   - **Modes:**
     - The Voigt modes 1 to 6 are tried in order.
     - A mode is kept when its strain images raise the rank of the set
       of strains, written as 6-vectors.
     - The search stops at rank 6.
   - **Signs:** as for displacements, the -s point of a mode is not
     computed when it is an image of the +s point.  An example is the
     shear yz under a rotation by 90 degrees about x.
   - **Fit:**
     - C (and C_clamped) comes from one least-squares fit over every
       computed point and all its images.
     - The unknowns are the 36 entries of C and a 6-vector intercept
       (the residual stress).
     - The result is then symmetrised as now.
     - With images included, C has the symmetry of the group to
       round-off; for example cubic C11 = C22 = C33.
   - **Internal strain:** it is rebuilt with the same operations.  The
     displacement of atom i under strain eps maps to R u of atom pi(i)
     under R eps R^T.
   - **Fit residuals:** `fit_rms_GPa` is reported for the computed modes
     and NaN for the modes taken from symmetry.  The results file lists
     which modes were computed.
   - **Identity only:** the present path is taken, bit for bit.
   - **Expected counts (default 3 amplitudes):** 37 strained cells
     become 10 for cubic (modes 1 and 4; the -s of the shear is an
     image) and about 19 for hexagonal (modes 1, 3 and 4).
7. **Validation and gate (Q8).**
   - **In the suite** (new `slow` tests at most 60 s in all):
     1. SW Si, diamond: the IFC, the frequencies and C with
        `symmetry: true` against `symmetry: false`, relative 1e-10.  C is
        also cubic to round-off.
     2. hcp Ti, Zhou 2004 EAM (hexagonal, operations by 60 and 120
        degrees): the frequencies and C against the full method, within
        the O(delta^2) bound measured for that case.
     3. The same hcp Ti cell against phonopy: phonopy builds the force
        constants from the same force function, and the frequencies at
        Gamma, M, K and A are compared (skipped without phonopy).
     4. A GeSn random cell: the operations are the identity, the counts
        equal the full method and the results are identical bit for bit.
     5. Refusals: missing spglib (simulated), ESM, an operation that
        does not map the atoms.
     6. The CLI keys of `phonon`, `qha`, `gruneisen`, `kappa` and
        `elastic`.
   - **Outside the suite**, under nohup, recorded in the result: the AlN
     MLIP on wurtzite.
     - Compared between the full and the reduced method: the
       frequencies on a q-path, C, the force calls and strained cells,
       and the wall time.
   - **Gate:** ADR-0028's 1.5 times, judged on the number of force calls
     or strained cells.  The cost of one call is the same either way.
     - The fixed cases are diamond Si (phonon and elastic) and AlN
       wurtzite (phonon and elastic).
     - The wall times are recorded beside the counts.
8. **Recording and version (Q1).**
   - This ADR is committed as Proposed, then implemented, then accepted
     with its result.
   - The version goes to 0.3.0a4; commits are local.
   - The default path (`symmetry: false`) of every command stays equal
     to 0.3.0a3 within 1e-10; it is in fact unchanged code.

## Consequences

- Phonon, QHA, Grüneisen and elastic runs on ordered crystals need
  several times fewer force calls when `symmetry: true` is given.  This
  matters most for MLIPs on large supercells and for external engines.
  On a P1 cell nothing changes.
- The operations, the atom permutation and the reconstruction are ready
  for the third-order force constants of item 6.
- spglib becomes an optional extra; numpy and scipy stay the only
  required dependencies (ADR-0028 decision 6).
- The reduced results carry the symmetry of the found group exactly.
  If `symprec` is so loose that it accepts a symmetry the structure does
  not quite have, the error is of the order of that departure.  The log
  shows the group so that the user can see this.

## Not decided here

- phonopy force-constant output (`FORCE_SETS`, `FORCE_CONSTANTS`) and
  reading or writing LAMMPS `.sw` / `.tersoff` files: tier C (ADR-0028
  addendum 6).
- Symmetry in the Born charges of `dielectric`, in `piezo`, and in the
  third-order force constants (item 6, its own ADR).
- Symmetrising an input structure (spglib's refinement): left to the
  user's own preparation of the structure.
