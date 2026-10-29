---
status: accepted (plan only, 2026-10-29)
---

# Large-scale GeSn test of relax and md: scaling and durability

Agreed with the user on 2026-10-29 (grilling, rounds 1-2, Q1-Q16).  The
GeSn tests so far (ADR-0021, ADR-0010 A1) used cells of 63-128 atoms.
This test asks how `relax` and `md` behave with many atoms: how run time
and memory grow with N, and whether long runs finish cleanly.

## Starting point (2026-10-29, 0.2.0a15)

Machine: Apple M4 (4 performance + 6 efficiency cores), 24 GB RAM.
mpi4py 4.1.2, numba 0.65.1 (on by default), torch 2.13.  The GPU backend
(ADR-0016) is not implemented.

One force call, warm, one process, measured on the 63-atom test cell
`mlip/training_set/test/0` replicated n x n x n:

| N | SW (`fit_sw`) | `ac_elm_iso_r2` |
|---|---|---|
| 63 | 7.8 ms | 59 ms |
| 504 | 51 ms | 0.57 s |
| 1701 | 0.16 s | 8.3 s (peak RSS 2.5 GB) |
| 4032 | 0.41 s | not run (estimate ~47 s, ~14 GB) |
| 13608 | 1.4 s | not run (estimate: out of memory) |

- SW is linear at about 0.1 ms per atom, with RSS below 0.3 GB.  E/atom
  of every replica equals the 63-atom value (SW -3.968956, MLIP
  -3.966687 eV/atom).
- The MLIP is quadratic.  `ACSF._neighbours`
  (`fitting/descriptors/acsf.py`) builds a dense (N, N, 27, 3) array of
  displacement vectors, so time and memory are O(N^2).  The ACSF kernel
  is single-threaded (`njit`, not `parallel`), and the MLIP force field
  has no MPI path.  SW, Tersoff and EAM split work across MPI ranks with
  replicated data.
- `relax` method `rfo` builds a finite-difference Hessian of size
  (3N+9)^2 from 2(3N+9) force calls, so it cannot be used at large N.
- Neither `md` nor `relax` can be restarted from a checkpoint.

## Decisions

1. **Goal (Q1).**  Scaling (time and memory against N) and durability
   (no crash, no NaN, no memory growth, complete output, energy
   conservation in NVE).  Physical sanity checks serve as pass criteria
   (decision 13).  No comparison with literature values.
2. **Target size (Q2).**  SW up to about 1.1e5 atoms, the MLIP up to
   1e4 atoms, on this machine.
3. **ACSF neighbour list first (Q3, Q10).**  The values above are kept
   as the "before" record.  `_neighbours` is replaced by a numba cell
   list inside `acsf.py` that returns every periodic image within r_cut
   for any cell, including cells thinner than 2 r_cut; the dense path is
   removed.  Acceptance: on every training and test structure the CSR
   neighbour list (after sorting) is identical, and G, forces and
   stress agree with the old code to a relative 1e-12.  The kernel gets
   `prange` only if, after the fix, profiling shows it takes more than
   50 % of the call.
4. **Potentials (Q4).**  SW with the published Tomita 2018 parameters is
   the main one (published, and continuous with the existing 128-atom
   examples; `fit_sw` uses the same engine, so its timing is the same).
   `ac_elm_iso_r2` is second.  Tersoff is timed only; its Sn parameters
   are placeholders.
5. **Machine (Q5).**  This Mac only.  Parallelism is measured inside it
   with numba threads and `mpirun`.
6. **Budget (Q6).**  About 48 h in total, at most 12 h per run, under
   `caffeinate -is nohup` with a JSONL ledger so that an interruption
   loses at most one run.
7. **Structures (Q7, Q9).**
   - Exactness check: the 63-atom test cell replicated n = 1, 2, 4.
   - Workload: random Ge0.875Sn0.125 diamond supercells of n^3 8-atom
     conventional cells with 1/64 of the sites vacant (n even), which
     keeps the MLIP inside its training distribution (63-atom vacancy
     cells).  Vegard lattice constant, seed 42.

     | n | N | SW per step (estimate) | MLIP per step (estimate, UNCERTAIN) |
     |---|---|---|---|
     | 4 | 504 | 0.05 s | 0.5 s |
     | 8 | 4032 | 0.4 s | 4 s |
     | 12 | 13608 | 1.4 s | 14 s |
     | 16 | 32256 | 3.3 s | -- |
     | 24 | 108864 | 11 s | -- |

     SW and Tersoff run n = 4, 8, 12, 16, 24; the MLIP runs n = 4, 8, 12.
     The MLIP estimate (about 1 ms per atom) comes from the 63-atom call
     and is re-measured after decision 3.
8. **Parallel matrix (Q11).**  SW at n = 12 (13608 atoms): numba threads
   1 / 4 / 10, and `mpirun -n` 1 / 2 / 4 with
   `NUMBA_NUM_THREADS = 10 / n`.  The MLIP varies threads only.  Time
   comes from the CLI's own timing output, memory from
   `/usr/bin/time -l` (peak RSS).
9. **relax (Q12).**  FIRE and L-BFGS; Vegard start plus a 0.05 A
   Gaussian rattle; `relax_cell: true`, `fmax` 0.01 eV/A, `pmax`
   0.1 GPa, `max_steps` 2000.  The main observable is the number of
   steps against N (`fmax` is a maximum over atoms, so it gets harder as
   N grows).  `rfo`, `diis`, `rmmdiis` and `sd` are left out; the `rfo`
   limit is recorded.
10. **md (Q13).**  Time step 1 fs.
    - Scaling: NVT 300 K, 200 steps from the relaxed cell at every rung.
    - Durability, one run at a time:
      1. SW, 4032 atoms, NVE 100 ps (about 11 h);
      2. SW, 4032 atoms, NPT 300 K / 0 GPa, 50 ps (about 6 h);
      3. SW, 13608 atoms, NVT 900 K, 20 ps (about 8 h);
      4. MLIP, 4032 atoms, NVT 600 K, as far as 12 h allows;
      5. MLIP, 4032 atoms, NVE 5 ps.
    - MLIP temperatures stay within its training range (MD frames up
      to 800 K).
    - `traj_every` 100 for scaling runs and 1000 for durability runs,
      keeping each XDATCAR under about 1 GB.
11. **No restart (Q14).**  Not implemented for this test; runs stay
    under 12 h and the missing restart is recorded as a finding.
12. **Findings (Q8).**  Bugs are fixed at once with a regression test and
    a full-suite run.  Design limits (the O(N^2) ACSF list until fixed,
    no restart, the `rfo` Hessian) are recorded and decided in a separate
    grilling.
13. **Pass criteria (Q15).**
    - Every run: exit 0, finite values, the expected number of frames.
    - Replicas against the 63-atom cell: E within 1e-10 eV/atom, forces
      on corresponding atoms within 1e-8 eV/A, stress within 1e-10.
    - Scaling: log-log slope of time against N at most 1.15 for SW and
      for the fixed MLIP (UNCERTAIN; reviewed once results exist).
    - Memory: peak RSS under 20 GB; in durability runs (`ps` every
      60 s) RSS grows less than 5 % after the first 10 % of the run.
    - NVE: |dE_tot| below 0.1 meV/atom/ps (UNCERTAIN; the fastmath ELM
      may need a looser value).
    - NVT: mean T within 2 % of the target.  NPT: mean P within 0.1 GPa.
    - MLIP not collapsed: shortest interatomic distance always above
      2.0 A, and E/atom never more than 0.05 eV/atom below the relaxed
      value.
    - relax: converged within `max_steps`; step counts are recorded.
14. **Order (Q16).**  This plan -> ACSF fix, full suite, commit ->
    replica exactness -> size ladder (about 8 h) -> parallel matrix
    (about 2 h) -> durability runs (about 37 h, one at a time) -> results
    in this ADR with figures checked as PNG (time and RSS against N on
    log-log axes, RSS against time, NVE E_tot against time, relax steps
    against N).  Runs are serial so that timings are not disturbed.
    Version 0.2.0a16 at the end; local commits only, pushes on request.

Scripts and outputs go to `example/GeSn/large/` (gitignored); results
are recorded here.
