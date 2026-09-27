---
status: accepted
---

# External codes as a potential: `potential: {type: external}`

`ExternalFileEngine` (akitsushima/fitting/external.py) lets an external
code act as the force function: each call writes the structure (POSCAR),
runs a shell command and reads energy, forces and stress back from a
plain-text EFS file (akitsushima/io/efs.py).  It was reachable from the
Python API only, had no test at all, and was not in the ADR-0003 CLI plan.

## Decisions (agreed with the user, 2026-09-27, Q22-Q27)

1. **Stand-in external codes** (Q22): no VASP / QE / LAMMPS on this
   machine, so (a) akitsushima itself in a separate process
   (`python -m akitsushima.tools.akitsushima_efs pot.yaml`), compared with
   the same potential in process -- the file route is the only
   difference -- and (b) an ASE calculator (EMT) in a separate process
   (`python -m akitsushima.tools.ase_efs --calc emt`), an independent code
   whose conventions must be converted.  Real DFT runs are for the user.
2. **Scope** (Q23): unit checks (EFS round trip, stress factor, cache,
   failures), physical consistency (F and S = derivatives of the E
   received), solvers (relax with cell, NVE, elastic constants).
3. **YAML route** (Q24): `potential: {type: external, cmd, work_dir,
   input_file, output_file, stress_to_eV_Ang3, timeout, mpi_prefix,
   cache_tol, r_cut, verbose}` in the registry; unknown keys are errors.
4. **EFS only** (Q25): the YAML route reads EFS files; a wrapper converts
   the code's output.  `tools.ase_efs --read FILE` converts anything ASE
   reads with energy, forces and stress (vasprun.xml, OUTCAR,
   espresso.pwo) -- UNVERIFIED on real output files, none were available.
   Built-in VASP / QE readers wait for real files.
5. **Solver rules** (Q26): relax, md, neb, tsa, phonon, elastic take it;
   gcmc / osmc take it with a start-up warning (one external run per
   trial move); kappa refuses it (no per-atom energies); coulomb_method /
   charges / esm with it are refused (the external total energy already
   contains the electrostatics).  `kind = "external"`,
   `per_atom_energy = False` on the engine; `check_external_for_solver`.
6. **Tests** (Q27): tests/test_external_engine.py (ASE part skipped
   without ASE) and example/external/check_external.py.

## Conventions settled

- **ASE stress sign.**  ASE's `get_stress()` is +(1/V) dE/d(strain)
  (positive under tension); akitsushima's is -(1/V) dE/d(strain) (positive
  under compression, ADR-0001).  `tools.ase_efs` flips the sign; checked
  against finite differences of EMT energies (xx, yz, xy agree to 1e-9
  eV/A^3) and in the test suite.
- **Energy** from ASE sources: the force-consistent (free) energy when
  available, since the forces are its derivatives.
- **r_cut** (default 1.0 A, must be positive): it only sizes the solvers'
  own Verlet list (md, neb, gcmc build one unconditionally and refused
  r_cut = 0); the external code ignores that list.

## Defects fixed on the way

- The cache ignored the species list (a swap of species at the same
  positions returned the old result) and handed out its own arrays (a
  caller mutating F corrupted the cache).  The key now includes the
  species; copies are returned.
- An `output_reader` returning lists was passed on unconverted; E, F, S
  are now float / (N, 3) / (3, 3) arrays.

## Verification

tests/test_external_engine.py (8 tests, ~60 s: every external call starts
a Python process):
- EFS round trip; stress factor; wrong atom count, missing file, non-zero
  exit and timeout are errors.
- akitsushima SW Si through the external route equals the in-process
  values to the EFS print precision (E 64e-10 eV, F and S 2e-10); forces
  and sigma_xx equal finite differences of the external energy.
- Cache: hit within cache_tol, miss beyond it and for another cell; a
  mutated result does not reach the cache.
- A FIRE relax through the external engine follows the in-process one
  (positions within 1e-7 A after 8 steps); relax through the CLI with
  `type: external`; YAML refusals; kappa refuses, gcmc warns.
- ASE EMT Cu through the external route equals EMT in process after the
  conversion; sigma_yy equals -(1/V) dE/d(eps_yy).

Solver comparison (example/external/check_external.py): see Results.

## Results (2026-09-27, example/external/check_external.py)

External route vs the same physics in process (SW Si: 64 atoms, relaxed
from a 1% expanded, rattled cell; EMT Cu: 32 atoms, same):

| case | stage | difference external - direct | external calls | time ext / direct | verdict |
|---|---|---|---|---|---|
| SW Si | relax (cell + ions, L-BFGS, fmax 1e-4) | E 3.5e-11 eV/atom, a 5.8e-11 A, positions 1.1e-9 A; a = 5.430950 A (SW 5.430950) | 25 | 58 s / 0.3 s | PASS |
| SW Si | NVE (200 fs, 600 K) | E_tot identical at the 1e-6 eV print precision; range 0.062 meV/atom, drift 0.009 meV/atom both | 201 | 456 s / 1.0 s | PASS |
| SW Si | elastic (relaxed ions) | max dC 1.0e-3 GPa; C11 / C12 / C44 = 151.4 / 76.4 / 56.4 GPa | 174 | 398 s / 0.9 s | PASS |
| EMT Cu | relax | E 2.1e-11 eV/atom, a 2.8e-12 A, positions 5.6e-10 A | 15 | 32 s / 0.1 s | PASS |
| EMT Cu | NVE (200 fs, 300 K) | E_tot identical at print precision; range 0.006 meV/atom | 201 | 428 s / 1.5 s | PASS |
| EMT Cu | elastic (clamped ions, fcc Bravais) | max dC 1.0e-6 GPa; 172.6 / 115.4 / 89.9 GPa | 12 | 26 s / 0.09 s | PASS |

The file route costs ~2 s per call here (a Python start-up and import per
call); an external DFT call dwarfs that.  No cache hits occurred in these
solvers (every call is a new geometry), so the cache matters only for
repeated evaluations of one structure.
