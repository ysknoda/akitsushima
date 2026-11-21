---
status: proposed (2026-11-22)
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
