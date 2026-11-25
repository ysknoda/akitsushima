---
status: accepted (2026-11-25, 0.4.0a3)
---

# FIRE: a mass for the cell coordinates (`fire_cell_mass`, default N^(1/3))

The follow-up of ADR-0025's finding 2 ("FIRE on SW at 108864 atoms
stalls"), after the relax re-measurement on 0.3.0a10 (ADR-0025
addendum).  Grilled with the user on 2026-11-25 in three rounds: Q1-Q2
(write the addendum; prototype a cell mass before an ADR), Q3 (adopt
N^(1/3) as the default), Q4-Q8 (key, scope, checkpoints, criteria,
process).

## Starting point (0.4.0a2)

FIRE with `relax_cell` gave all 3N + 9 coordinates mass 1.  The cell
coordinate is x = L eps with L = V0^(1/3) (ADR-0041 decision 4; raw
lattice components scale the same way), so the cell force is about
sigma L^2 and the cell stiffness about B L, while the atoms' stiffness
does not depend on N.  Above a few thousand atoms the cell is the
stiffest coordinate and sets FIRE's time step.  Measured on the
ADR-0025 ladder (fit_sw, Ge0.875Sn0.125 with 1/64 vacancies, 0.05 A
rattle, start at about 4 GPa; `relax_cell`, fmax 0.01 eV/A, pmax 0.1
GPa, 2000 steps):
- the median FIRE time step falls as L^(-1/2): 0.116, 0.089, 0.071,
  0.059, 0.048 fs from 504 to 108864 atoms;
- with the cell fixed it is 0.118 fs at 108864 atoms;
- FIRE stops at 2000 steps at 108864 atoms (fmax 0.153 eV/A), while
  L-BFGS converges in 673 steps and FIRE with the cell fixed in 904.

## Decisions

1. **Mass (Q3).**  In FIRE the 9 cell coordinates get mass m; the atoms
   keep mass 1.  Only the two half-kicks divide by it; the power
   P = v . g, the velocity mixing and the 0.2 A step clamp are as in
   Bitzek et al. (PRL 97, 170201, 2006).  Default m = N^(1/3): the cell's
   stiffness over mass is then about B V / (L^2 N^(1/3)) = B Omega^(1/3)
   (Omega the volume per atom), independent of N, with no reference size
   to choose.
2. **Key (Q4).**  `relax.fire_cell_mass` and
   `GeometryOptimiser(fire_cell_mass=...)`: `auto` (default, N^(1/3)) or
   a positive number; 1 is 0.4.0a2 bit for bit.  Used only by FIRE with
   `relax_cell`; ignored otherwise.  The relax template explains it.
3. **Scope (Q5).**  Strain and lattice cell coordinates; `anneal` passes
   the key through its relax section.  Every FIRE variable-cell
   relaxation takes the default, including `elastic`'s pre-relaxation and
   the QHA relaxations.  NEB's FIRE has no cell degrees of freedom.
4. **Checkpoints (Q6).**  The mass is a checkpoint setting when FIRE
   relaxes the cell with m != 1; an older checkpoint has no such item and
   means m = 1.  A 0.4.0a2 FIRE variable-cell checkpoint is therefore
   refused under `auto`; the message names `fire_cell_mass` (set it to 1
   to continue).
5. **Criteria (Q7).**  m = 1 reproduces 0.4.0a2 bit for bit (63-atom
   GeSn fixture, strain and lattice coordinates).  The final code
   reproduces prototype runs bit for bit, so the prototype's
   measurements are this ADR's results (108864 atoms is not run again).
   Existing FIRE tests pass unchanged; full suite.
6. **Process (Q8).**  ADR-0053, 0.4.0a3, one commit on branch
   fire-cell-mass; ADR-0048 (thermo labels) follows.

## Result (2026-11-25)

### Prototype measurements

Prototype: the same change driven by an environment variable
(`example/GeSn/large/adr0047/fire_cell_proto.py`, ledger
`fire_cell_proto_ledger.jsonl`), on 0.3.0a10 with the ladder's start
structures; one thread, load 3-7 (another project's jobs held two
cores).  Steps to convergence, one start each:

| Run | m = 1 | m = n/4 | m = N^(1/3) |
|---|---|---|---|
| SW 504 | 178 | (= 1) | 157 |
| SW 4032 | 360 | 274 | 228 |
| SW 13608 | 781 | 862 | 538 |
| SW 108864 | not converged (2000) | 1215 | 816 |
| Tersoff 108864 | 126 | 97 | 73 |
| MLIP (ac_elm_iso_r2) 4032 | 293 | 293 | 488 |

(n is the number of conventional cells per edge; m = n/4 is 1 at 504
atoms.)  At 108864 atoms N^(1/3) = 47.7 brings the median time step back
to 0.122 fs (0.118 with the cell fixed); the run takes 1834 s against
1878 s for L-BFGS and ends 0.07 meV/atom below L-BFGS's minimum.

Five starts at 4032 atoms (the ladder's seed 42 and seeds 1-4, built the
ladder's way):

| Model | m = 1 | m = N^(1/3) | Paired ratio |
|---|---|---|---|
| SW | mean 431 (sd 90) | mean 334 (sd 65) | 0.63-1.05, fewer in 4 of 5 |
| MLIP | mean 366 (sd 101) | mean 378 (sd 106) | 0.57-1.67, within the spread |

The MLIP's 488 steps above is within the spread between starts.  The
4032-atom MLIP cell is not yet large enough for the cell to limit the
time step, so the mass changes little there.  m = n/4 helped less (SW
13608 slower than m = 1) and was not pursued.

### Final code

`example/GeSn/large/adr0053/gates.py`, ledger `gates_ledger.jsonl`.

- m = 1 against 0.4.0a2, 63-atom GeSn fixture: steps, energy, positions
  and lattice equal (224 steps strain, 209 lattice); the prototype's
  m = 1 run at 504 atoms has the same relax.log as 0.3.0a10 bit for bit.
- Prototype runs again with the key (SW 504 at m = 1 and auto, SW 4032,
  MLIP 4032, Tersoff 108864 at auto): relax.log rows equal to the
  prototype's bit for bit in all five.
- Lattice coordinates, SW 13608 atoms: m = 1 885 steps (median dt 0.071
  fs), auto 583 steps (0.110 fs); both converge, auto 0.47 meV/atom
  lower.
- Tests: `tests/test_fire_cell_mass.py` (m = 1 against 0.4.0a2 on both
  coordinates; `auto` = N^(1/3) and bad values refused; `auto` converges
  to the same basin; no effect without FIRE cell coordinates; checkpoint
  refusals).  Full suite: 1372 passed, 1 skipped; no existing test needed a
  change.

## Consequences

- Every FIRE relaxation with `relax_cell` takes a different path from
  0.4.0a2 (relax, anneal quench, elastic pre-relaxation, QHA).  It ends
  at a minimum within the same tolerances; in a rugged landscape (a
  random alloy with vacancies) it may be a different nearby minimum, as
  between any two starts.  `fire_cell_mass: 1` restores the old path.
- 0.4.0a2 FIRE variable-cell checkpoints need `fire_cell_mass: 1` to
  continue.
- FIRE's atoms-only part also slows with N (904 steps at 108864 atoms
  with the cell fixed, 193 at 504); not changed here.
