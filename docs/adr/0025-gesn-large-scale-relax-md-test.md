---
status: accepted (plan 2026-10-29; size ladder 2026-11-21; parallel matrix and durability runs 2026-11-22/23; all results written 2026-11-25)
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

## Results

### ACSF neighbour list (decision 3, 2026-10-29)

`_neighbours` is now a numba cell list (`_cell_scan`, two passes: count,
then fill).  It scans every periodic image for any cell: bins are at
least r_cut wide, and a cell thinner than r_cut is covered by scanning
several image copies of its bins.  Each atom's entries are sorted into
the former order (neighbour index, then image in the old `meshgrid`
order), and the displacement is computed in the old order of operations
without fastmath.  The ACSF kernel's loop over atoms is now `prange`:
atom i writes only `G[i]` and its own neighbour entries, so there is no
race.

Checks:

- CSR (ptr, nbj, vec) bitwise equal to the dense search on 666 cells at
  r_cut 3.1, 5.0 and 7.3 A: the 564 training and 83 test structures,
  the 2-atom primitive diamond cell, and 20 random thin and sheared
  triclinic cells with atoms outside the cell.
- `ac_elm_iso_r2` on the 83 test cells and one 504-atom replica: E and G
  bitwise equal; F and S within a relative 3.4e-14 (the parallel kernel
  is compiled differently).  Q10 asked for 1e-12.
- Full suite: 798 passed (31 min at 2 threads under load).
- `tests/test_acsf_neighbours.py`: the dense search kept as the reference,
  replicas giving the same G, and a 3456-atom cell.

Profile (32256 atoms, before `prange`): the ACSF kernel took 86 % of the
call and the cell list 0.23 s, so the `prange` condition of decision 3
was met.

Force call of `ac_elm_iso_r2` with the cell list, before `prange`, 2
numba threads while another job loaded the machine (indicative only; the
ladder re-measures):

| N | before | after |
|---|---|---|
| 504 | 0.57 s | 0.24 s |
| 1701 | 8.3 s, 2.5 GB | 0.78 s |
| 4032 | ~47 s, ~14 GB (estimate) | 2.0 s, 0.41 GB |
| 32256 | out of memory | 16 s, 1.65 GB |

### Size ladder (decisions 7, 9, 10, 13; 2026-11-20 to 2026-11-21)

Run as ADR-0028 addendum 5 allowed: one core, numba and BLAS limited to
one thread, one CLI run at a time, tagged `loaded` (the user worked on
the machine, and the ADR-0038 benchmarks used a second core for part of
the time).  Script `example/GeSn/large/ladder.py`, ledger
`ladder_ledger.jsonl` (41 units), figure `ladder.png` (md time per step
and peak RSS against N, relax steps against N; checked as PNG).  Wall
time 24.1 h (2026-11-20 18:50 to 2026-11-21 18:56), mostly the L-BFGS
runs that did not converge.  The code was main at 546d2a9 (0.2.0b1)
until 2026-11-21 02:06 and a8c83e5 (0.3.0a1) after it, from
`relax/fit_sw/n24/fire` on; ADR-0037 does not touch the force or
integrator paths of these runs.  The ledger's `akitsushima` field is the
version the driver imported at the start.  ADR-0038 (moment-tensor G5)
is not in these numbers.

**Replicas (63-atom test cell, n = 1, 2, 4): PASS** for all three
models.  Largest differences against n = 1: E 1.8e-13 eV/atom, forces
4.5e-10 eV/A (SW; 3.5e-13 for Tersoff and the MLIP), stress 3.5e-11
relative.

**md, NVT 300 K, 1 fs, 200 steps from the FIRE-relaxed cell**

| Model | N | time per step | per atom | peak RSS | d_min end |
|---|---|---|---|---|---|
| SW (`fit_sw`) | 504 | 32 ms | 64 us | 0.24 GB | 2.39 A |
| | 4032 | 0.26 s | 64 us | 0.37 GB | 2.37 A |
| | 13608 | 0.93 s | 68 us | 0.58 GB | 2.31 A |
| | 32256 | 2.16 s | 67 us | 0.83 GB | 2.34 A |
| | 108864 | 6.99 s | 64 us | 1.25 GB | 2.33 A |
| Tersoff (`fit_tersoff`) | 504 | 1.9 ms | 3.7 us | 0.22 GB | 2.31 A |
| | 4032 | 13 ms | 3.3 us | 0.26 GB | 2.29 A |
| | 13608 | 45 ms | 3.3 us | 0.37 GB | 2.25 A |
| | 32256 | 0.11 s | 3.3 us | 0.63 GB | 2.29 A |
| | 108864 | 0.40 s | 3.7 us | 1.12 GB | 2.29 A |
| MLIP (`ac_elm_iso_r2`) | 504 | 52 ms | 104 us | 0.32 GB | 2.38 A |
| | 4032 | 0.44 s | 109 us | 0.57 GB | 2.36 A |
| | 13608 | 1.43 s | 105 us | 1.27 GB | 2.32 A |

Peak RSS is the largest of the three runs of a rung (FIRE, L-BFGS, md).
Log-log slope of md time per step against N: SW 1.00, Tersoff 1.00,
MLIP 1.01 (0.97 for N >= 4032).  The decision 7 estimates were 0.4 s
(SW) and 4 s (MLIP) per step at 4032 atoms; the MLIP is 9 times faster
than estimated after the cell list, `prange` and tier A.  The drift of
the NHC conserved quantity is 0.03 (MLIP) to 0.07 (Tersoff) meV/atom/ps
and does not grow with N.

**relax, `relax_cell`, fmax 0.01 eV/A, pmax 0.1 GPa, 2000 steps**

| Model | N | FIRE steps | FIRE time | L-BFGS steps | L-BFGS time |
|---|---|---|---|---|---|
| SW | 504 | 192 | 9.9 s | 92 | 6.5 s |
| | 4032 | 353 | 103 s | 1540 | 1877 s |
| | 13608 | 854 | 788 s | not converged | 3644 s |
| | 32256 | 1644 | 3822 s | not converged | 11942 s |
| | 108864 | not converged | 15192 s | not converged | 31632 s |
| Tersoff | 504 | 68 | 3.3 s | 25 | 2.8 s |
| | 4032 | 72 | 4.6 s | 54 | 4.8 s |
| | 13608 | 107 | 11 s | 234 | 32 s |
| | 32256 | 90 | 21 s | 1749 | 1337 s |
| | 108864 | 128 | 84 s | 386 | 388 s |
| MLIP | 504 | 102 | 8.8 s | 40 | 5.6 s |
| | 4032 | 489 | 227 s | not converged | 1843 s |
| | 13608 | 501 | 804 s | not converged | 8778 s |

**Pass criteria (decision 13), judged again.**  The ledger marks all 13
md runs and 6 relax runs as failed.  The md flags come from the
script's own check, not from the runs:
- It asked for a mean T within 6 K of 300 K.  A 200-step run that starts
  at a minimum with 300 K velocities shares the kinetic energy with the
  potential energy and settles near 150 K (T_mean 153-162 K, T_inst
  oscillating between about 80 and 300 K), and `tau_nhc` 100 fs cannot
  restore 300 K within 200 fs.  The 2 % criterion of decision 13 is meant
  for the durability runs; it does not apply to these scaling runs.
- It asked for the last thermo row at step 200.  Rows are labelled 0 to
  `n_steps - thermo_every` (finding 3 below).

With those two checks replaced by "20 thermo rows, 2 frames, d_min above
2.0 A", the results are:
- exit 0 and finite values: all 41 runs, PASS;
- 20 thermo rows and 2 XDATCAR frames in every md run: PASS;
- md MLIP not collapsed: d_min 2.32-2.38 A, and the lowest E_pot per atom
  stays 0.1 meV above the relaxed value (allowed: 50 meV below): PASS;
- scaling, slope at most 1.15: PASS for all three models;
- memory under 20 GB: PASS (largest 1.27 GB);
- relax converged within `max_steps`: FIRE fails only for SW at 108864
  atoms; L-BFGS fails for SW from 13608 atoms and for the MLIP from 4032
  atoms: FAIL (findings 1 and 2).

`ladder.py` now uses the replaced md checks; the ledger is kept as run.

**Findings (decision 12: design limits are decided in a separate
grilling).**
1. L-BFGS with `relax_cell` stalls on large cells.  The MLIP runs stop
   moving: E changes by less than 1e-5 eV over the last steps, with fmax 0.45
   and 0.66 eV/A and pmax 0.49 and 0.22 GPa.  SW at 13608-108864 atoms
   ends with fmax 0.17-0.30 eV/A.  L-BFGS steps cost 1.3 to 4.2 times a
   FIRE step (line-search calls), and where they stop the energy lies up
   to 5.6 meV/atom above FIRE's (MLIP, 4032 atoms).  Tersoff converges with both.
2. FIRE on SW at 108864 atoms stalls at fmax 0.121 eV/A (time step down
   to 0.02 fs, E changing by about 1e-4 eV per step); its step count grows with N (192,
   353, 854, 1644 steps), as decision 9 expected for a maximum over
   atoms.  Tersoff (68-128 steps) and the MLIP (102-501) grow much less.
3. The md thermo log labels the row written after step k+1 as step k at
   time k dt (`step` is the 0-based loop index), so the first row (step
   0, time 0) is the state after one step and the last row is
   `n_steps - thermo_every`.  The same holds for all four integrators.
   Not a crash; changing it changes every thermo file.
4. The multi-species SW kernel (`_compute_multi_species`, serial
   `njit`) takes 0.24 s for 4032 GeSn atoms, against 0.011 s for the
   multi-species Tersoff and 0.018 s for single-species SW Si at 4096
   atoms (`bench_force`, one thread): 64 us per atom in md, 17 times
   Tersoff.  A candidate for tier B item 8 (speed-ups).

### Parallel matrix (decision 8; 2026-11-22, 0.3.0a7)

`example/GeSn/large/campaign.py`, ledger `campaign_ledger.jsonl`, stage
`parallel`.  NVT 300 K, 200 steps from the FIRE-relaxed 13608-atom cell
(n = 12), as the ladder's md runs; time per step from the CLI, memory
from `ps` on the process tree.  Load average 2.0-4.2 (macOS background
work), tagged `loaded`.

| Model | Run | time per step | speed-up | peak RSS, one process / tree |
|---|---|---|---|---|
| SW (`fit_sw`) | 1 thread | 0.520 s | 1 | 0.40 / 0.38 GB |
| | 4 threads | 0.520 s | 1.00 | 0.42 / 0.36 GB |
| | 10 threads | 0.518 s | 1.00 | 0.44 / 0.43 GB |
| | `mpirun -n 1`, 10 threads | 0.517 s | 1.01 | 0.44 / 0.43 GB |
| | `mpirun -n 2`, 5 threads each | 0.279 s | 1.86 | 0.34 / 0.59 GB |
| | `mpirun -n 4`, 2 threads each | 0.239 s | 2.18 | 0.46 / 1.63 GB |
| MLIP (`ac_elm_iso_r2`) | 1 thread | 0.641 s | 1 | 1.08 / 1.01 GB |
| | 4 threads | 0.268 s | 2.39 | 1.07 / 1.05 GB |
| | 10 threads | 0.227 s | 2.82 | 1.11 / 1.08 GB |

- All nine runs pass decision 13 (exit 0, finite, 20 rows, 2 frames,
  RSS under 20 GB).  T_mean is the same to every printed digit across
  the six SW runs (144.13 K) and across the three MLIP runs (138.85 K),
  so neither threads nor ranks change the trajectory at this length.
- SW gains nothing from threads: the multi-species SW kernel is a serial
  `njit` (ladder finding 4).  Only MPI splits it, 2.2 times at four
  ranks; replicated data costs about 0.4 GB per rank.
- The MLIP's `prange` ACSF kernel gives 2.4 times on four threads and
  2.8 times on ten (the M4 has four performance cores).
- The ladder's 13608-atom md timings (SW 0.93 s, MLIP 1.43 s per step)
  were inflated by a load of 5-8; at a load of about 2 the same runs
  take 0.52 s and 0.64 s.

### Durability (decision 10; 2026-11-22 to 2026-11-24, 0.3.0a7)

Stage `durability` of `campaign.py`: one run at a time, one thread, md
checkpoints every chunk (ADR-0031, which replaced the "no restart" of
decision 11), `ps` every 60 s.  Figure `durability.png` (RSS against wall
time, NVE E_tot, T of the thermostatted runs, unit 2's P and V; checked
as PNG), made by `plot_durability.py`.

| Unit | Run | wall | per step | peak RSS | Result | Verdict |
|---|---|---|---|---|---|---|
| 1 | SW 4032, NVE, 100 ps | 4.2 h | 0.153 s | 0.30 GB | E_tot drift -1.2e-6 meV/atom/ps, largest deviation 0.010 meV/atom | pass |
| 2 | SW 4032, NPT 300 K / 0 GPa, 50 ps | 2.1 h | 0.149 s | 0.28 GB | T_mean 298.0 K, P_mean -0.0018 GPa, V 25.08 A^3/atom | pass |
| 3 | SW 13608, NVT 900 K, 20 ps | 2.7 h | 0.481 s | 0.45 GB | T_mean 774.2 K | **fail** |
| 3b | as 3, template burn-in (ADR-0044) | 3.2 h | 0.519 s | 0.49 GB | T_mean 899.7 K over the 20 ps production | pass |
| 4 | MLIP 4032, NVT 600 K, 200 ps | 9.3 h | 0.168 s | 0.49 GB | T_mean 600.01 K, d_min 2.149 A, lowest E_pot 0.1 meV/atom above relaxed | pass |
| 5 | MLIP 4032, NVE, 5 ps | 0.24 h | 0.171 s | 0.49 GB | E_tot drift -5.6e-5 meV/atom/ps, largest deviation 0.023 meV/atom, d_min 2.253 A | pass |

Verdicts leave out the RSS-growth criterion (below).  Unit 4 was planned
"as far as 12 h allows" and set to 200 ps by the user.  Every run exited
with 0, stayed finite and wrote the expected rows and frames.

- **NVE.**  Both drifts are three to five orders of magnitude below the
  0.1 meV/atom/ps of decision 13, so the UNCERTAIN bound is not tight
  for SW or for the fastmath ELM.  The largest deviation is set by the
  first 0.2 ps (panel b): E_tot steps by 0.008 (SW) / 0.013 (MLIP)
  meV/atom while the start at a minimum halves the kinetic energy, the
  O(dt^2) offset of velocity Verlet's conserved energy following T.
  Afterwards it is flat.
- **NPT.**  The volume settles within about 8 ps, 0.6 % above the 0 K
  relaxed cell (24.93 A^3/atom), and stays there.
- **Unit 3: slow heating.**  The campaign started every thermostatted run
  at the relaxed cell without the template's burn-in (`equil_steps` 0,
  `velocity_rescale_steps` 0).  Equipartition halves T at once, and the
  two-link Nose-Hoover chain then took 7.8 / 13.9 / 8.1 ps to reach
  0.95 T0 in units 2 / 3 / 4 (panel c).  20 ps of unit 3 was too short
  to average it out.  Cause and remedy: ADR-0044 (three links by default
  from 0.3.0a8; unit 3b with the burn-in passes on 0.3.0a7), and for
  anneal ADR-0045.
- **RSS growth.**  The criterion "less than 5 % growth after the first
  10 %" (the largest sample after 10 % against the sample at 10 %)
  fails on units 1-4 and 3b (8.4 / 7.7 / 31.8 / 12.2 / 21.7 %) and passes
  on unit 5 (0.5 %).  The samples are noise, not growth (panel a): macOS
  resident memory jumps between 0.13 and 0.18 GB (units 1, 2), 0.22 and
  0.41 GB (3, 3b) and 0.17 and 0.41 GB (4) as pages are compressed and
  brought back.  Least-squares trends over the same windows are -10 /
  +17 / +11 / -6 / +7 / +9 % (units 1, 2, 3, 3b, 4, 5): both signs, and
  no run ends above 0.49 GB.  No leak is seen; the criterion as written
  measures paging, not use.  A leak check on macOS should sample the
  physical footprint (`footprint` / `phys_footprint`), not RSS.
- **Driver.**  `campaign.py` stopped once in its own judge (a numpy bool
  that `json` could not write) after unit 1 had finished.  It was fixed
  and restarted from unit 2 without rerunning unit 1.  Later it was
  swapped while unit 2 ran, attaching to the running process.  Neither
  touched akitsushima.

### Pass criteria, all stages

- exit 0, finite values, rows and frames: every run of the ladder (41),
  the parallel matrix (9) and the durability stage (6): **pass**.
- Replicas: **pass**.  Scaling slope at most 1.15: **pass** (1.00-1.01).
  Peak RSS under 20 GB: **pass** (largest 1.63 GB, four MPI ranks).
- NVE drift, NPT pressure, MLIP not collapsed: **pass**.
- NVT mean T within 2 %: **fail** for unit 3 as run, **pass** for 3b and
  unit 4 (ADR-0044).
- relax within `max_steps`: **fail** for L-BFGS on large cells and FIRE
  on SW at 108864 atoms (ladder findings 1 and 2).
- RSS growth under 5 %: not met as written; no growth seen (above).

### Findings of the parallel and durability stages

5. The two-link Nose-Hoover chain heats a cold start in about
   ln(1.9) tau sqrt(n_dof): fixed in ADR-0044 (0.3.0a8).
6. The RSS-growth criterion measures macOS paging; a future leak check
   should use the physical footprint.
7. Threads do not speed up multi-species SW (ladder finding 4, still
   open); MPI does, 2.2 times at four ranks.

Findings 1-4 (ladder) and 7 went to the separate grilling of decision
12 (2026-11-25); finding 5 is closed by ADR-0044 and ADR-0045.

### Addendum: relax re-measured on 0.3.0a10 (findings 1, 2; 2026-11-25)

Agreed in the grilling of the open findings (Q4, Q11): the relax runs
that did not converge are run again on the current code, with the
ladder's own start structures and settings (`relax_cell`, fmax 0.01
eV/A, pmax 0.1 GPa, 2000 steps, one thread).  Since the ladder the cell
moves in strain coordinates by default (ADR-0041, 0.3.0a5) and the SW
three-body forces are analytic (ADR-0047, 0.3.0a10).  Script
`example/GeSn/large/adr0047/relax_recheck.py`, ledger
`relax_recheck_ledger.jsonl`; load 4.5-7.6 (another project's jobs held
two cores), so the times are indicative.

| Run | Ladder (0.2.0b1) | 0.3.0a10 |
|---|---|---|
| L-BFGS, MLIP 4032 | not converged, 1843 s | 119 steps, 38 s |
| L-BFGS, MLIP 13608 | not converged, 8778 s | 188 steps, 166 s |
| L-BFGS, SW 13608 | not converged, 3644 s | 237 steps, 90 s |
| L-BFGS, SW 32256 | not converged, 11942 s | 513 steps, 368 s |
| L-BFGS, SW 108864 | not converged, 31632 s | 673 steps, 1878 s |
| FIRE, SW 108864 | not converged, fmax 0.121, 15192 s | not converged, fmax 0.153, 4046 s |
| FIRE, SW 108864, cell fixed | - | 904 steps, 2250 s (P stays 1.83 GPa) |

**Finding 1 (L-BFGS) is closed.**  All five L-BFGS runs that stalled
now converge.  The cause is the strain cell coordinates of ADR-0041,
not ADR-0047: the 4032-atom MLIP run (no SW) with `cell_coordinates:
lattice` on 0.3.0a10 stops moving after about 250 steps at fmax 0.354
eV/A and stays there (E -16137.135 eV to within 1e-5 eV up to step
2000), as in the ladder.  Where both
methods converge the minima differ slightly: L-BFGS ends 0.23 and 1.24
meV/atom above the ladder's FIRE for the MLIP at 4032 and 13608 atoms,
0.007 above for SW at 13608, and 0.089 below for SW at 32256; at 108864
atoms it ends 0.145 meV/atom below where FIRE stops.

**Finding 2 (FIRE) is not closed; its cause is the cell coordinate.**
FIRE on SW at 108864 atoms still stops at 2000 steps, much as in the
ladder (173 time-step resets against 175, energy still falling by about
0.006 eV per step over the last 100 steps; 16 eV above the L-BFGS minimum).  Facts:
- The median FIRE time step on SW falls with the cell edge L as
  L^(-1/2): 0.116, 0.089, 0.071, 0.059, 0.048 fs at 504 to 108864
  atoms (ratios to 504 atoms 1.97 at 32256 and 2.42 at 108864, against
  sqrt(4) = 2 and sqrt(6) = 2.45).
- With the cell fixed the median is 0.118 fs at 108864 atoms, the
  504-atom value, and FIRE converges in 904 steps.
- The cell coordinate is x = L eps (ADR-0041 decision 4), so its
  force is about sigma L^2 and its stiffness about B L, while the
  atoms' stiffness does not depend on N.  FIRE gives every coordinate
  the same mass, so above a few thousand atoms the cell is the stiffest
  coordinate and sets the time step.  Raw lattice coordinates (the
  ladder) scale the same way.
- Tersoff (128 steps at 108864 atoms) starts at about 1 GPa, SW at
  about 4 GPa, so SW needs a much larger cell change.
- Even with the cell fixed FIRE needs 904 steps against 193 at 504
  atoms: the atoms alone also relax more slowly in the large cell.

The fix was grilled on 2026-11-25 (Q2): a mass for the cell coordinates
of FIRE is prototyped and measured before an ADR is written.
Closed by ADR-0053 (0.4.0a3): with `fire_cell_mass: auto` (N^(1/3))
the 108864-atom run converges in 816 steps.
