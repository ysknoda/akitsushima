---
status: accepted (2026-11-22; implemented in 0.3.0a3 -- see Result)
---

# CSVR thermostat in `md` and a thermostat choice in `anneal`

Tier B item 4 of ADR-0028 (addendum 5): the CSVR thermostat of Q14 and
the Langevin thermostat in anneal's `md_quench`, which ADR-0032
(decision 6) left for later.  Agreed with the user on 2026-11-22
(grilling, Q1-Q11, all as recommended; Q11 replaced part of Q5).

## Starting point (2026-11-22, 0.3.0a2)

- **`md`, NVT.**  `thermostat: nhc | langevin`.
  - `nhc` is `NVTSimulation`: velocity Verlet with a Nose-Hoover chain of
    length 2, whose half-steps sit before the first and after the second
    velocity half-step.  `thermo.log` shows `H_NHC`.
  - `langevin` is `LangevinSimulation(NVTSimulation)` (ADR-0032): BAOAB,
    `damping` in fs (default 100), heat Q accumulated, `E-Q` in
    `thermo.log`, Q and the random-number state in the checkpoint.
- **`md`, NPT.**  The particle thermostat is the Nose-Hoover chain only;
  the cell has `baro_thermostat: nhc | langevin` (ADR-0036).
- **`anneal`, md_quench (ADR-0022).**
  - `_md_chunk` builds an `NVTSimulation` or `NPTSimulation` for every
    chunk of `quench_every` steps.  The positions, velocities and cell
    carry over, the thermostat restarts, and the seed of a chunk is
    `seed + 7919 (cycle + 1)`.
  - The md section accepts the keys of `_MD_KEYS` in `cmd_anneal.py`;
    there is no `thermostat` key.
  - `anneal_log.jsonl` has no measured temperature of the MD chunk.
- **Other users of `NVTSimulation`.**  The Green-Kubo equilibration of
  `kappa` and `dynamics_analysis`.
- **CSVR** exists nowhere in the package.

## Decisions

1. **Scope (Q1).**
   - `md` NVT gets `thermostat: csvr`.
   - The anneal md section gets `thermostat: nhc | langevin | csvr` and
     `damping`.
   - Not in this item, recorded only: CSVR or Langevin as the particle
     thermostat of NPT (it would need its own check against MTTK), and
     the thermostat of the `kappa` and `dynamics_analysis`
     equilibration.  The Green-Kubo production phase must stay without a
     stochastic thermostat (ADR-0032 decision 6).
2. **Integrator (Q2).**  Canonical sampling through velocity rescaling,
   Bussi, Donadio and Parrinello, J. Chem. Phys. 126, 014101 (2007).
   - The kinetic energy K is propagated exactly over a time h by the
     stochastic process of that paper (its appendix).  With
     c = exp(-h / tau), the target K_t = N_f k_B T / 2, one Gaussian
     number R_1 and S, the sum of N_f - 1 squared Gaussian numbers
     (drawn as one chi-squared number with N_f - 1 degrees of freedom):

         K' = c K + (1 - c) K_t (R_1^2 + S) / N_f
              + 2 R_1 sqrt(c (1 - c) K K_t / N_f)

     All velocities are multiplied by alpha = sqrt(K' / K), with the
     sign of R_1 + sqrt(c N_f K / ((1 - c) K_t)) as in the paper.  A
     negative alpha needs h much larger than tau and is kept for
     exactness.
   - The rescaling is applied twice per step, with h = dt / 2, where
     the Nose-Hoover half-steps of `NVTSimulation` sit: before the
     first and after the second velocity half-step.  The step is then
     symmetric in time, as the NVT integrator is now.
   - All atoms share one factor, so the total momentum is conserved:
     no centre-of-mass removal is needed.  Fixed components are zero and
     stay zero.  N_f follows `_compute_n_dof` (3N - 3, or 3N minus the
     fixed components).
3. **Parameter (Q3).**  `damping` in fs, default 100, the same key as
   Langevin; LAMMPS also calls both damping parameters "damp".  The two
   act differently at the same value: the kinetic energy relaxes with
   time constant tau under CSVR and damping / 2 under Langevin.  The
   template and the docstrings say so.
4. **Class, CLI and diagnostic (Q4).**
   - `CSVRSimulation(NVTSimulation)`, exported as `ak.CSVRSimulation`.
     It shares the output files (`XDATCAR_nvt`, `thermo.log`,
     `POSCAR_final_MD`), the checkpoint and the settings check of NVT.
   - The energy the rescaling adds is accumulated as Q, and
     `thermo.log` shows `E-Q(eV)` where NVT shows `H_NHC(eV)`: the
     effective energy of the paper, whose drift is the integrator
     check.  Q and the random-number state go into the checkpoint; the
     settings check adds `damping` and drops `tau_nhc`, and its ensemble
     is `csvr`, so a checkpoint of another thermostat is not continued.
   - The CLI refuses, with the reason: `thermostat: csvr` with `npt` or
     `nve`, an unknown thermostat, a damping of 0 or less, and
     `velocity_rescale_steps > 0` (as for Langevin, ADR-0032
     decision 8).
   - The NVT template lists `csvr` in the comment of `thermostat`.
5. **anneal (Q7, Q8, Q9).**
   - The md section takes `thermostat` (default `nhc`) and `damping`;
     `_md_chunk` builds `NVTSimulation`, `LangevinSimulation` or
     `CSVRSimulation`.  With the default the run is unchanged.
   - `ensemble: npt` with `thermostat: langevin | csvr` stops with the
     reason; the cell keeps `baro_thermostat`.
   - Each chunk keeps its own seed, `seed + 7919 (cycle + 1)`, so a
     restarted run still reproduces the uninterrupted one.
   - Every quench row of `anneal_log.jsonl` gets `T_md`, the mean
     temperature of its MD chunk, for every thermostat.
   - The template shows `thermostat: nhc   # nhc | langevin | csvr` and
     a commented `# damping: 100.0`.
6. **Validation of CSVR (Q5, replaced in part by Q11).**
   - Not on independent harmonic oscillators, the check ADR-0032 used
     for Langevin.  CSVR rescales all atoms by one factor and samples
     the canonical ensemble only when the underlying dynamics is
     ergodic.  Independent oscillators are not: with one frequency,
     every component goes through the same linear map and the same
     factor, so ratios such as those of x_i v_j - x_j v_i never change.
     A correct CSVR could fail that test.
   - SW silicon, 64 atoms, 600 K (interacting atoms):
     1. the mean temperature within 4 sigma of the target;
     2. the variance of the kinetic energy equal to N_f (k_B T)^2 / 2
        within 4 sigma (block averages).  This holds exactly in the
        canonical ensemble whatever the potential.  The same run with the
        noise set to zero (a Berendsen-like rescaling) must fail it;
     3. the mean potential energy equal to that of a Langevin run at the
        same temperature within the combined error (configurational
        sampling);
     4. the total momentum below 1e-10 and the drift of E - Q recorded.
   - Restart: an interrupted and resumed run equals the uninterrupted
     one bit for bit, random numbers included (the test of ADR-0031,
     `test_md_restart.py`).
   - MPI: a `csvr` case in `test_mpi_cli.py` agrees on 1, 2 and 4 ranks.
   - In the suite the runs are short (`slow` tests at most 60 s in
     all).  A long run (about 200 ps) goes outside the suite under nohup
     and its variance, with a tighter error, goes into the result.
7. **Validation of anneal (Q10).**
   - In the suite: the CLI anneal with `langevin` and with `csvr`; a
     restarted run equal to the uninterrupted one bit for bit; refusal
     of `npt` with a stochastic thermostat and of unknown values.
   - Outside the suite, under nohup, recorded here: SW Si, 64 atoms, a
     geometric schedule from 1500 to 300 K in 8 segments of at least
     2000 steps each, with `nhc`, `langevin` and `csvr`.  Compared: the
     mean temperature of each segment and the energies of the minima
     found.  Pass: the segment means of `langevin` and `csvr` within
     4 sigma of their targets.  `nhc` is recorded for comparison only.
   - Not in this item: a re-run of the GeSn MLIP anneal (ADR-0027 A1),
     which stays a separate research decision.
8. **Recording and version (Q6).**  This ADR is committed as Proposed,
   then implemented, then accepted with its result.  The version goes to
   0.3.0a3; commits are local.  The default `nhc` path of `md` and
   `anneal` must stay equal to 0.3.0a2 within 1e-10.

## Consequences

- `md` gets a global stochastic thermostat that disturbs the dynamics
  less than Langevin (one random number per half-step for the whole
  system instead of one per degree of freedom), for NVT runs whose time
  correlations matter.
- anneal can cool with Langevin or CSVR, which relax to each new
  segment temperature without the oscillations of a freshly restarted
  Nose-Hoover chain.
- NPT keeps the Nose-Hoover chain on the particles.

## Result (2026-11-22, 0.3.0a3)

**Implementation.**
- `_StochasticNVTSimulation(NVTSimulation)` in
  `solvers/molecular_dynamics.py` now holds the loop that
  `LangevinSimulation` had: output files, `E-Q`, the checkpoint with Q
  and the random-number state, the result dict.  A subclass supplies one
  step.  Langevin's BAOAB step moved there unchanged; a 300-step SW Si
  run is bit for bit the same as in 0.3.0a2 (the first run after a fresh
  numba compile differed by 5e-12 A, a compile effect, not the code).
- `CSVRSimulation(_StochasticNVTSimulation)`, exported as
  `ak.CSVRSimulation`: `_rescale` draws R_1 and a chi-squared number with
  N_f - 1 degrees of freedom and applies the exact propagation of
  decision 2 over dt/2; `_step` is R, B, A (full drift), wrap, force, B,
  R.  The checkpoint ensemble is `csvr`.
- `md` CLI: `thermostat: nhc | langevin | csvr`; csvr with npt or nve, an
  unknown name and `velocity_rescale_steps > 0` stop with the reason.
- `anneal`: `thermostat` and `damping` in the md section
  (`GlobalOptimiser.thermostat`), the refusals of decision 5, `T_md` in
  every quench row.  Templates of `md` (NVT) and `anneal` updated.
- **Bug found and fixed on the way.**  The anneal template's md section
  has the NPT-only `tau_baro: 1000.0`.  With `ensemble: nvt` it went to
  `NVTSimulation`, so every MD chunk of a template md_quench run failed
  ("unexpected keyword argument 'tau_baro'"), while the run still ended
  with exit 0 and only the initial relaxation as its minimum.  No test
  ran md_quench from the template.  `tau_baro` is now dropped for nvt;
  `test_cli_md_quench_from_the_template` (nhc and langevin) requires
  every cycle to be ok.
- The default `nhc` path is unchanged: md NVT (200 steps) and anneal
  md_quench (4 cycles) are bit for bit equal to 0.3.0a2.

**Validation, CSVR (SW Si, 64 atoms, 600 K, damping 100 fs, dt 1 fs, one
numba thread).**  Long runs outside the suite:
`example/thermostats/validate_adr0039.py`, ledger
`validate_ledger.jsonl`; 2 ps equilibration, 200 ps production, 40-block
errors; N_f = 189.

| Run | <T> (K) | Var(K) / [N_f (kT)^2 / 2] | <E_pot>/N (eV) | max \|P\| | E-Q drift |
|---|---|---|---|---|---|
| CSVR | 600.7 +- 1.9 | 1.019 +- 0.027 | -4.25882 +- 0.00025 | 7.9e-14 | 3.6 meV |
| Langevin | 597.4 +- 1.5 | 0.968 +- 0.015 | -4.25902 +- 0.00021 | 2.8e-16 | 4.4 meV |
| CSVR without noise | 596.8 +- 0.03 | 0.269 +- 0.008 | -4.25929 +- 0.00003 | 4.8e-13 | 2.3 meV |

- CSVR passes every criterion of decision 6: T at 0.4 sigma, the
  variance at 0.7 sigma, <E_pot> 0.6 sigma from Langevin, the momentum
  conserved to round-off.
- The control fails as required: a deterministic rescaling keeps only
  27 % of the canonical variance of K (90 sigma).
- In the suite (`tests/test_csvr.py`): the rescaling alone, iterated
  40000 times, gives the gamma distribution (mean, variance) and the
  conditional mean c K + (1 - c) K_t; a 22 ps run with the same checks
  against the 200 ps Langevin <E_pot> (23 s); the zero-noise control
  (6 ps); fixed atoms; refusals; the CLI.  `test_md_restart.py` (both
  restart tests) and `test_mpi_cli.py` have a `csvr` case.

**Validation, anneal (decision 7).**  SW Si, 64 atoms, 0.8 A jitter,
geometric 1500 -> 300 K, 8 segments of 4000 steps, one quench per
segment.  Segment means over the last 3000 steps; the 40-block errors
(75 fs blocks) are shorter than the correlation time of T, so the table
uses 10 blocks of 300 fs.

| Thermostat | max \|z\| over 8 segments | rms z | best E/N (eV) |
|---|---|---|---|
| nhc (tau_nhc 20 fs) | 2.78 | 1.51 | -4.33640 |
| langevin (damping 100 fs) | 2.27 | 1.35 | -4.33640 |
| csvr (damping 100 fs) | 2.40 | 1.45 | -4.33640 |

- Pass: the segment means of `langevin` and `csvr` lie within 4 sigma
  of their targets (with the 40-block errors also: max 2.76 and 3.35).
- All three reach the diamond minimum (-4.3364 eV/atom) from the second
  quench on.  In this small SW test the Nose-Hoover chain with
  tau_nhc 20 fs follows the schedule as well as the stochastic
  thermostats; the advantage expected for Langevin in quenching
  (ADR-0032 decision 6) does not show here and was not measured on an
  MLIP.
- The `slow` tests added by this item take about 32 s in all.

**Found on the way, fixed in 0.3.0a3 at the user's request: numba
caches froze the thread count.**
- Four parallel kernels sized their per-thread buffers with
  `numba.config.NUMBA_NUM_THREADS` read inside the jitted function:
  `_pair_kernels._pair_body_numpy_par` (all pair potentials), the SW
  2-body and 3-body kernels and the Tersoff 3-body kernel.  With
  `cache=True` the value was frozen when the cache was built.  A cache
  built by a process with NUMBA_NUM_THREADS=1 (the size ladder,
  `bench_force.py --threads 1`, any one-thread run) and loaded by a
  process with 10 threads got thread ids 0-9 for a buffer of one row:
  out-of-bounds writes.
- Shown on the Tersoff kernel (64-atom Si, 400 calls): cache built at
  10 threads, then 10 threads: fine twice.  Cache built at 1 thread, then
  10 threads: "Abort trap" and "Segmentation fault", and one process hung
  in the abort handler.  The first full-suite run of this item died the
  same way in `test_template_runs_as_shipped` (elastic, Tersoff), after a
  one-thread timing run in the worktree had built the Tersoff cache.
- The same out-of-bounds writes could change results without crashing:
  the extra threads' rows were never summed.
- Fix: the four kernels take `n_threads` as an argument.  Thin wrappers
  (`_pair_body_jit`, `_sw_2body_jit`, `_sw_3body_jit`,
  `_te_3body_force_jit`) pass `utils.jit._max_threads()`, which reads
  `numba.config.NUMBA_NUM_THREADS` in Python at call time.  Call sites and
  the serial fallbacks are unchanged.  E, F and S of LJ, SW and Tersoff Si
  at 10 threads are bit for bit the same as before the fix (fresh caches).
- `tests/test_numba_thread_cache.py`: a subprocess builds the caches with
  NUMBA_NUM_THREADS=1 (the test checks that the four kernels' caches
  exist), a second loads them with 4 threads; E, F and S must agree.
  Before the fix the second process died with a segmentation fault.

**Full suite.**  1059 passed, 1 skipped and 21 errors in 17.8 min; the
errors were all of `test_mpi_cli.py`, whose `mpirun python -m akitsushima`
subprocesses imported the editable install of the main checkout
(0.3.0a2, without csvr) instead of this worktree.  With `PYTHONPATH` set
to the worktree, `test_mpi_cli.py` gives 23 passed: 1080 passed,
1 skipped in all.
After the numba fix, with `PYTHONPATH` set to the worktree for the whole
run: 1081 passed, 1 skipped in 19.7 min, 0.3 min under the 20 min
threshold at which xdist gets its own grilling (ADR-0028 addendum 5).
