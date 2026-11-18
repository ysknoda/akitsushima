---
status: accepted (2026-11-18; implemented in 0.2.0a21 -- see Result)
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

## Result (2026-11-18, 0.2.0a21)

**Neighbour-list order (decision 11).**  `NeighbourList.build` now puts
its candidates in canonical order (option `canonical_order`, default
on).  The owned lists of ADR-0028 step 3 get it from there.  The sort
uses one int64 key per pair, which orders like `lexsort` because the
list is minimum-image, so each (i, j) occurs once.  Cost with SW Si,
cell-list back-end, 1 thread, the two orders alternated:

| | without | with |
|---|---|---|
| one build, 4096 atoms | 15.7 ms | 17.2 ms (+10 %) |
| NVT step, 4096 atoms (21 builds in 300 steps) | 27.0-28.0 ms | 27.2-29.7 ms |
| NVT step, 512 atoms | 3.7-4.0 ms | 3.8-4.0 ms |

The expected cost per step is 21 x 1.5 ms / 300 = 0.1 ms, below 1 %.
The measured step times vary by +-5 % from load alone, so the ordering
stays on for every run.  `lexsort` took +40 % per build.  Switching
the ordering off makes the restarted NVT, NVE and NPT runs of the test
below differ from the uninterrupted ones, so it is what makes them
equal.

**md.**
- The new module `solvers/_checkpoint.py` holds what md and relax
  share:
  - atomic npz writing on rank 0;
  - the settings check (both ways: an item on one side only is a
    difference);
  - file lengths and truncation;
  - the potential hash.
- `_BaseMDSimulation` gains `_ck_settings`, `_ck_begin` and `_ck_save`.
- The three `run` methods take `checkpoint`, `checkpoint_every` and
  `checkpoint_meta`.
- The checkpoint stores:
  - positions, velocities and cell;
  - the chain variables (NVT: 4; NPT: 9 with `p_eps` and the barostat
    chain);
  - the per-step lists behind the averages and the NVT thermo records;
  - the reference energy of the drift (NVE `e_tot_ref`, NPT `H_ref`);
  - the random-number state;
  - the next XDATCAR frame index;
  - the byte lengths of `thermo.log` and the XDATCAR.
- A checkpoint is written every `checkpoint_every` steps and at the
  end.  The final one makes extension possible: equal settings with a
  larger `n_steps` continue a finished run.
- `velocity_rescale_steps` was added to the settings that must match
  (it changes the dynamics during equilibration).

`tests/test_md_restart.py`:
- For NVT, NVE and NPT, a run killed at step 37 by an exception from
  the force call (after the step-25 checkpoint, with lines past it
  already in the files) and restarted equals the uninterrupted run.
  Positions, velocities, cell, every scalar of the result, the XDATCAR
  and `thermo.log` are bit-identical.  The one exception is the NVT
  column `NL_builds`, which counts the builds of the neighbour list in
  each process.
- An extended run equals the longer run.
- Different temperature, ensemble or potential hash, an absent hash,
  and `n_steps` below the checkpoint are refused.

**relax.**
- `GeometryOptimiser.run` takes the same three arguments.
- Every `checkpoint_every` steps it saves, before the row is logged:
  the structure about to be logged, the steps used, the best structure
  and the file lengths.  A restart logs that structure again as the
  same step, with the same energy.
- `_log_row`, `_update_best` and `_finalise` add the step offset, and
  `run` lowers `max_steps` by it for the method loop.
- The method headers are written only on a fresh start.
- At the end a final checkpoint is written with the result, and with
  the file lengths taken before the `Converged:` summary line.
- A restart from a final checkpoint:
  - returns the stored result, without new steps or file changes, if
    the run converged or `max_steps` is not larger;
  - otherwise continues from the last structure with rows appended
    where the summary was.
- The settings compared are:
  - method and atoms;
  - `relax_cell`;
  - the convergence switches and thresholds;
  - `save_best`;
  - `cell_update_scheme`;
  - the fixed degrees of freedom.

`tests/test_relax_restart.py`, for all six methods:
- the rows before the checkpoint are kept;
- the checkpointed structure is logged again with the same energy;
- the steps run 0..n-1 without gaps or repeats;
- the XDATCAR has one frame per row;
- a finished run is not redone, and raising `max_steps` continues it;
- different settings are refused;
- writing checkpoints does not change the path.

**CLI.**
- `md` and `relax` read `checkpoint` (default `md_checkpoint.npz` and
  `relax_checkpoint.npz` in `output_dir`), `checkpoint_every` (default
  1000 for md and 10 for relax) and `restart` (default true).
- `restart: false` deletes the checkpoint.
- The potential hash covers the potential section and every file it
  names.
- The templates show the keys.

`tests/test_cli_restart.py` checks four things:
- a finished md run is extended through the CLI;
- `restart: false` starts afresh;
- a changed LJ epsilon is refused;
- relax continues after `max_steps` is raised.

The first full-suite run had one failure.  The template-key guard
`test_relax_template_keys_are_all_read_by_cmd_relax` looks for
`rel_cfg.get("<key>"` in `cmd_relax.py`, and the keys were read inside
the shared helper.  The commands now read the keys themselves and pass
the values.

Full suite after the fix: 875 passed, 1 skipped (14.5 min).

**Not done here.**
- The md classes of `gcmc` and `osmc` have no checkpoint.
- `anneal` keeps its own `restart: true` convention.
