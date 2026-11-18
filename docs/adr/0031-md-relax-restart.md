---
status: proposed (2026-11-18)
---

# Restart of `md` and `relax` from a checkpoint

Tier A step 6 of ADR-0028.  Agreed with the user on 2026-11-18
(grilling, Q1-Q11).  ADR-0025 recorded that neither `md` nor `relax`
can be restarted, so a run is limited to what finishes inside one job
(12 h there).

## Starting point (2026-11-18, 0.2.0a20)

- **md.**  `NVTSimulation` (Nose-Hoover chain), `NVESimulation` and
  `NPTSimulation` (MTTK with a barostat chain) in
  `solvers/molecular_dynamics.py`.  The state that carries over between
  steps:
  - positions and velocities;
  - the thermostat chain;
  - for NPT, the cell, `p_eps` and the barostat chain;
  - the step number;
  - the per-step lists behind the averages (T, E_pot, E_kin);
  - the neighbour list.

  `thermo.log` and `XDATCAR_*` are opened in "w" mode and written as
  the run goes; `POSCAR_final_MD` is written once at the end.  Random
  numbers are drawn only for the initial velocities.  The Langevin
  thermostat of step 7 will draw them every step.
- **relax.**  Six methods (fire, lbfgs, diis, rmmdiis, rfo, sd) in
  `solvers/geometry_optimiser.py`.  Each keeps its own state inside its
  loop: FIRE its velocity, dt and alpha; LBFGS its history.
  `XDATCAR_relax` gets every step's structure.
- **Existing restarts, with two different conventions.**
  - `nemd` (ADR-0020) continues automatically from its checkpoint when
    the file exists; `restart: false` discards it.  It stores the
    random-number state, and a test checks that a restarted run
    continues bit for bit.
  - `anneal` (ADR-0022) continues only with `restart: true`.  It drops
    the ledger lines past the checkpoint.
  - Both write an npz through a temporary file and `os.replace`, on
    rank 0 only.
- The `NeighbourList` that md passes to the force function is not put
  in the canonical pair order of ADR-0028 step 3.  A list rebuilt at
  the restart point would order the pairs differently, and the rounding
  would differ.

## Decisions

1. **Scope (Q1).**
   - md stores its full state and continues exactly.
   - relax continues from the last checkpointed structure with the
     method's state started afresh, keeping the steps already used.
     FIRE recovers in a few steps and LBFGS refills its history in
     about 20, so exact per-method state is not worth its cost.
2. **Bit for bit (Q2).**  An md run interrupted and restarted gives the
   same bits as the run that was not interrupted.  The checkpoint holds:
   - positions and velocities;
   - the thermostat chain and, for NPT, the cell, `p_eps` and the
     barostat chain;
   - the per-step lists behind the averages and the thermo log
     records;
   - the step number;
   - the random-number state, a slot for step 7.

   The neighbour list is not stored: it is rebuilt, and its candidate
   pairs are put in canonical order on every build (decision 11), so the
   forces do not depend on when it was built.  A test interrupts NVT,
   NVE and NPT runs with an exception, restarts them, and compares them
   bitwise with uninterrupted runs.
3. **When to continue (Q3).**  The `nemd` convention: if the checkpoint
   exists, the run continues from it; `restart: false` deletes it and
   starts afresh.  Resubmitting the same job script after a time limit
   therefore just continues.  The `anneal` default is left as it is.
4. **Frequency (Q4).**  `checkpoint_every` in steps, default 1000, as
   in `nemd`.  A step count fixes where a run resumes, which the bitwise
   test needs.
5. **Output files (Q5).**  The checkpoint records the byte length of
   `thermo.log` and the trajectory at the time it was written.  A
   restart truncates both files to that length and appends, so the
   finished files equal those of an uninterrupted run.
6. **Extension (Q6).**  A finished run can be extended by raising
   `n_steps`.  The new steps are production steps and enter the
   averages.
7. **API (Q7).**  `run(checkpoint=..., checkpoint_every=...)` on the
   three md classes and on the relax optimiser, as in `nemd`.  The CLI
   calls these (ADR-0024: the CLI and the API offer the same features).
8. **Recording (Q8).**  This ADR is written before the implementation
   and accepted with the result.  The version goes to 0.2.0a21.
9. **Consistency check (Q9).**  A restart compares these with the
   checkpoint and stops with the reason if any differs:
   - the number and order of atom types;
   - the ensemble;
   - dt and `equil_steps`;
   - temperature, pressure, `tau_nhc` and `tau_baro`;
   - `thermo_every` and `traj_every`;
   - in the CLI, a hash of the potential: the text of its config
     section and the bytes of any model file it reads.

   `n_steps` may differ if it is not below the checkpoint's step
   (decision 6).  Through the API the force function cannot be
   compared, so the other items are checked.
10. **relax checkpoint (Q10).**  An npz written every `checkpoint_every`
    steps.  It holds:
    - positions and cell;
    - the steps used;
    - the best structure so far (`save_best`);
    - the byte lengths of `relax.log` and `XDATCAR_relax`.

    The md consistency check and file truncation are shared.
11. **Neighbour-list order (Q11).**  `NeighbourList` puts its candidate
    pairs in canonical order (i < j, sorted by (i, j)) after every
    build.  The added cost per build and per md step is measured and
    recorded here.  If it slows an md step by more than 5 %, the
    ordering is applied only to runs that write a checkpoint.

## Result

(to be appended)
