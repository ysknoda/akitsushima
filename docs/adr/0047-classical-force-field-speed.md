---
status: accepted (2026-11-25, 0.3.0a10)
---

# Classical force fields: analytic SW three-body forces, threaded multi-species SW / Tersoff, numpy pair grouping

The first of three follow-ups to the open findings of ADR-0025 (ladder
findings 1-4 and finding 7 of the parallel matrix), agreed with the user
on 2026-11-25 (grilling, three rounds, Q1-Q17; Q9 changed by the user
to include Tersoff).  The other two are the relax re-measurement
(findings 1, 2) and ADR-0048 (finding 3, thermo labels).  This ADR
answers findings 4 and 7: "the multi-species SW kernel is serial and
17 times slower per atom than Tersoff; threads do not help".

## Starting point (0.3.0a9)

One force call of `fit_sw` on the 4032-atom GeSn ladder cell took
0.156 s on one thread, 92 % of it in the three-body kernel, for three
reasons:

1. **Finite differences.**  The SW three-body forces (single- and
   multi-species numba kernels) were central differences of the term
   h3 (step 1e-6 A): 12 extra evaluations per triple.  An earlier
   hand-derived formula had been wrong in a way a perfect crystal does
   not show, and the differences replaced it.
2. **Serial kernels.**  The multi-species SW kernel and both
   multi-species Tersoff kernels were serial `njit`; only the
   single-species ones had `prange`.
3. **Cutoff.**  `fit_sw` reaches 4.33-4.53 A (sigma a per leg), past
   the second-neighbour shell of GeSn (about 4.07 A): 16 neighbours and
   about 120 triples per atom instead of 4 and 6.  The published Tomita
   et al. (2018) set reaches 3.95 A on Ge legs and 4.46 A on Sn legs, so
   it also counts some second neighbours.

Re-checking every classical family with multi-species cells at 1 and 4
threads (load 5-6, indicative): EAM (Zhou Cu-Ni) gained 1.3 times, as
its single-species path does (ADR-0043); SW and Tersoff gained nothing;
the pair potentials gained nothing either, because 88 % of a 4096-atom
MgO call was the per-pair Python loop of
`BasePairPotential._build_pair_groups`; Buckingham + PPPM gained
nothing, and Buckingham + Ewald took 33 s.

## Decisions

1. **SW three-body forces (Q2).**  Analytic gradient of h3
   (`_sw_h3_grad`) in the single-species serial and parallel kernels and
   in the multi-species kernel.  The NumPy fallback (numba off) keeps
   the differences and serves as the reference in the tests.
2. **Threads (Q9, changed by the user; Q15).**  The multi-species SW
   three-body kernel and both multi-species Tersoff kernels get prange
   twins with per-thread force and stress buffers.  One thread calls the
   serial kernels, so it is bit for bit the former path.  The two-body
   terms and EAM are unchanged.
3. **Pair grouping (Q14).**  `_build_pair_groups` groups the pairs with
   numpy: one stable sort of a species-pair key (16-bit integers, a radix
   sort in numpy), same keys, same key order and same pair order as the
   former loop, so every pair potential (LJ, Morse, Born-Mayer,
   Buckingham, Mie, Yukawa) is bit for bit unchanged.
4. **Not changed (Q3, Q16).**  The `fit_sw` cutoff (a model property;
   refitting belongs to the MLIP side) and the Coulomb solvers (PPPM was
   reworked in ADR-0043; Ewald is known to be slow for large cells).
5. **Criteria (Q7).**  SW energy bit for bit; forces and stress against
   central differences of the energy on distorted Si and GeSn to a
   relative 1e-7, forces summing to zero; against the former differences
   within 1e-8 eV/A (forces) and a relative 1e-8 (stress).  Tersoff and
   pair potentials bit for bit on one thread.  Threads 1 / 4 / 10 to a
   relative 1e-12.  This is an exception to "the default path stays
   within 1e-10" (ADR-0043), as in ADR-0041: removing the
   difference error is the point.
6. **Speed gates (Q8, Q14, Q15).**  SW GeSn 4032 atoms, one thread:
   three-body kernel 3 times faster.  MgO 4096 atoms (pair path), one
   thread: whole call 3 times faster.  SW 13608 atoms: 4 threads 2.5
   times one thread.  Tersoff 13608 atoms: 4 threads 2 times one
   thread.  A missed gate is recorded and discussed, apart from
   correctness.
7. **Process (Q17).**  Four commits (analytic SW, threads, pair grouping,
   ADR + benchmarks + 0.3.0a10); `bench_force` gains `sw_gesn_tomita`
   (Tomita et al. 2018, Table I), `tersoff_xy` (two-species test values
   on the diamond sublattices, not a material) and `eam_cuni` (Zhou et
   al. 2004).

## Result (2026-11-25)

### Correctness

Distorted cells (rattled and strained; 64-atom Si, 504-atom GeSn ladder
cell), one thread, against 0.3.0a9:

| Potential | Energy | Forces | Stress |
|---|---|---|---|
| SW Si | bit for bit | max diff 2.7e-10 eV/A | relative 5.5e-11 |
| SW GeSn (`fit_sw`) | bit for bit | max diff 2.4e-9 eV/A | relative 2.2e-10 |
| Tersoff Si, GeSn (`fit_tersoff`) | bit for bit | bit for bit | bit for bit |
| Buckingham MgO | bit for bit | bit for bit | bit for bit |

Threads 4 and 10 against 1 (4032-atom GeSn ladder cell, rattled): SW energy
5.6e-15, forces 1.8e-15, stress 4.7e-13 (relative); Tersoff 8.2e-16,
4.1e-16, 1.2e-14.

Tests: `tests/test_sw_analytic.py` (h3 bit for bit against the former
term in plain Python; its gradient against differences of h3 on 500
random triples up to 2 % below the cutoffs; forces and stress against
energy differences on Si and on GeSn with asymmetric records to a
relative 1e-7; the numba path against the difference fallback within
1e-8) and `tests/test_multi_species_threads.py` (threads, and the
grouping against the former loop with four species named out of
alphabetical order).  `test_nhc_chain`'s two legacy-checkpoint tests
compare a continuation with 0.3.0a7's numbers on SW Si; the forces moved
by ~1e-9 eV/A and the step-300 state by 3e-12 (relative), so their
tolerance went from 1e-12 to 1e-9.  Full suite: 1189 passed, 1 skipped.

Note: the kernels are compiled with fastmath.  The h3 values were equal
bit for bit in the kernels measured here, but fastmath may reorder
operations, so a standalone compiled `_sw_h3_grad` can differ from
`_sw_h3_scalar_multi` in the last bit; the test compares the plain
Python functions bit for bit and the compiled ones to 1e-15.

### Speed gates

`example/GeSn/large/adr0047/gates.py`, ledger `gates_ledger.jsonl`; old =
main at 2e5e075 (0.3.0a9), new = this branch; the ladder cells of
ADR-0025 and MgO with zero charges.  Load 4.1-6.5 throughout (another
project's jobs held two cores).

| Gate | old | new | Ratio | Verdict |
|---|---|---|---|---|
| SW GeSn 4032, 1 thread, three-body kernel | 0.143 s | 0.047 s | 3.0 | pass |
| MgO 4096, 1 thread, whole call | 0.138 s | 0.036 s | 3.8 | pass |
| SW GeSn 13608, new, 4 threads vs 1 | 0.202 s | 0.135 s | 1.50 | **missed** (2.5) |
| Tersoff GeSn 13608, new, 4 threads vs 1 | 27.5 ms | 23.5 ms | 1.17 | **missed** (2.0) |

Further numbers (whole call, `fit_sw`, 13608 atoms): old 0.528 / 0.677 /
0.606 s at 1 / 4 / 10 threads (threads made it slower), new 0.202 /
0.135 / 0.102 s.  New at 10 threads is 5.2 times old at one.

Why the thread gates are missed:
- Only the kernels are threaded.  The numpy work before them -- minimum
  image displacements, the CSR sort of the neighbour list, the parameter
  tables, and for Tersoff the bond orders between its two kernels --
  stays serial: 0.044 s of the 0.202 s SW call and about 8 of the
  27.5 ms Tersoff call.  With perfect kernel scaling 4 threads would
  give at most about 2.4 (SW) and 2.2 (Tersoff).
- The M4 has four performance cores, and two were busy with another
  project's jobs during every measurement.
The user decided to record the gates as missed and close (2026-11-25);
threading the preparation is not part of this ADR.

### bench_force (`benchmarks/adr0047.jsonl`, tag `loaded`)

New code at 1 / 4 / 10 threads; the old baseline is the same script run
against main at 2e5e075 (`PYTHONPATH`; kept out of the ledger, whose
`commit` field names the script's checkout).  Whole call, 4096 / 4000
atoms, one thread, old / new: `sw_si` 1.56, `sw_gesn_tomita` 1.74,
`buck_pppm_mgo` 1.91, `tersoff_si` 1.01, `tersoff_xy` 1.07, `eam_ni`
0.97, `eam_cuni` 1.06.  At 4 / 10 threads `sw_gesn_tomita` takes 25 / 21
ms against 34 ms on one.  Tersoff at 4096 atoms (8-10 ms per call) does
not gain from threads, its kernels being a small part of the call.

## Findings

1. Thread scaling of the classical families is limited by serial numpy
   preparation, not by the kernels any more; threading it (numba CSR
   build, MIC) is the next step if large classical runs need it.
2. `fit_sw` and, on Sn legs, Tomita's SW count second neighbours in the
   three-body sum; `fit_sw` has about 20 times the triples a
   nearest-neighbour SW would.  A model matter, recorded only.
3. Coulomb: PPPM does not gain from threads; Ewald takes 33 s for 4096
   MgO atoms.  Not changed.
4. The NumPy fallback of SW still uses differences; it is the test
   reference and is used only with numba off.

## Consequences

- SW forces differ from 0.3.0a9 by up to ~1e-9 eV/A (the former
  differences' error); energies are unchanged.  Tersoff and pair
  potentials are bit for bit on one thread.
- Multi-species SW and Tersoff runs now use numba's threads.
