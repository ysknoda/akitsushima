---
status: accepted (implemented 2026-10-14)
---

# The `anneal` command: MD-quench annealing and basin hopping

Agreed with the user on 2026-10-14 (grilling, rounds 1-2, Q5-Q17).  The
package could relax a structure and run MD, but had no way to repeat
"heat, then quench" or to hop between minima.  Users strung `md` and
`relax` runs together by hand.

## Decisions

1. **One command, two methods (Q5).** `akitsushima anneal` has two
   methods, set with `method: md_quench | basin_hopping`.
   - `md_quench` follows a temperature schedule (Q11): `constant`,
     `linear`, `geometric` (the default, 1500 -> 300 K over 8 segments)
     or `list`.
   - `basin_hopping` follows Wales & Doye, J. Phys. Chem. A 101, 5111
     (1997).
2. **Reuse the solvers (Q9).** `solvers/global_opt.py`
   (`GlobalOptimiser`) drives `NVTSimulation`, `NPTSimulation` and
   `GeometryOptimiser`. It adds no integrator and no optimiser of its
   own. The MD classes gained only two parameters:
   - `initial_velocities`, to continue a trajectory;
   - `save_plots`, so a chunk can skip its time-series figure.
3. **md_quench is one continuous trajectory (Q10).**
   - The trajectory is cut into chunks of `quench_every` steps. Each
     chunk continues the previous one's positions, velocities and cell;
     only the NHC chain restarts.
   - After every chunk a copy of the structure is relaxed. The quench
     never feeds back into the MD.
   - The temperature is constant within a segment.
4. **Cell degrees of freedom (Q6).** The MD may be NVT or NPT, and the
   quench may relax the cell (`relax.relax_cell`).
5. **Basin-hopping moves (Q12).**
   - Every atom is displaced uniformly inside a sphere of radius `step`
     (default 0.3 A).
   - With `relax_cell`, the cell also gets a random symmetric strain of
     at most `cell_step` (default 0.02).
   - Every 10 trials the step is multiplied by 0.9, or divided by it,
     towards an acceptance of 0.5.
   - Acceptance is Metropolis on the relaxed energy at `temperature`
     (default 1000 K). With `pressure`, it compares the enthalpy.
6. **Minima (Q13).**
   - Two minima are the same when |dE| <= 1e-4 eV/atom and
     |dV|/V <= 1e-3.
   - Up to `keep_best` (10) distinct minima are written as
     `POSCAR_rankNN`, the lowest also as `POSCAR_best`, and all are
     listed in `minima.csv`.
   - The history is plotted in `anneal.png`.
7. **YAML (Q14).** `anneal.md` and `anneal.relax` take the key names of
   the `md` and `relax` configs. Unknown keys are refused, so a typo
   cannot silently fall back to a default.
8. **Restart (Q7).**
   - One JSON line per cycle goes to `anneal_log.jsonl`, and
     `checkpoint.npz` is rewritten atomically.
   - The checkpoint holds the MD state, or the walker's structure, step
     and random-number state.
   - `restart: true` continues after the checkpoint's cycle. Ledger
     lines past that cycle are dropped.
   - The ledger line is written before the checkpoint. If a run is
     interrupted between the two, the extra line is trimmed on restart
     and its cycle recomputed; no line is lost.
9. **Failures (Q8).** A cycle fails when the MD or the quench raises, the
   energy is not finite, or the energy drops below
   `min_energy_per_atom`.
   - The last case is an MLIP collapsing under extrapolation, which is
     the common failure in ADR-0021's relax runs.
   - The energy checks apply to the last frame of every MD chunk as well
     as to the quench. An MLIP can collapse during the MD without
     raising: the GeSn ELM at 800 K went from -4.1 to -6.7 eV/atom and
     to 8600 K within 450 steps.
   - After a failure, md_quench returns to the last good MD state with
     freshly drawn velocities.
   - After a failure, basin hopping keeps its current structure.
   - `max_consecutive_failures` (5) failures in a row stop the run. The
     minima found so far are still written, and the command exits 1.

## Choices made during implementation

These were not in the agreed plan.

- **The MD starts from the relaxed input (cycle 0), not the raw POSCAR.**
  We tried the raw POSCAR first: 64-atom SW Si with 0.8 A random
  displacements disordered completely within its first 1000-step chunk
  at 1300 K, below SW's melting point of 1691 K. A strained input
  releases its excess potential energy as heat faster than the
  thermostat (tau = 20 fs) removes it. Started from the relaxed input,
  the same run healed to diamond.
- **Pressure is applied through the force function.**
  `GeometryOptimiser` has no target pressure. `enthalpy_force_fn` wraps
  the potential to return E + PV and sigma - P*I. Stress is
  compression-positive (ADR-0001), so d(PV)/d(eps_ab) = P V delta_ab.
  Every relaxation method then minimises H without any change of its
  own.
- **Unconverged quenches.** In basin hopping, a trial whose relaxation
  did not converge is rejected. Such a quench never enters the list of
  minima.
- **Random draws.** The walker draws its Metropolis number on every
  trial, including failed ones. Its random-number stream therefore does
  not depend on the outcome, and a restart lines up with the
  uninterrupted run.

## Verification

tests/test_anneal.py (19 tests) and the `anneal` template case in
tests/test_cli_templates.py.  Unless a citation is given, each reference
value below was computed for this work.

**Basin hopping**
- LJ13: 60 trials from a random compact start reach
  E = -44.326801 epsilon (Wales & Doye 1997), the Mackay icosahedron.
  The unshifted LJ relaxes the ideal icosahedron to -44.3268014.
- With the YAML default force-shifted LJ, the search finds that
  potential's icosahedron energy, -43.9626666.
- At kT = 0.8 epsilon nearly every move is accepted, and the step grows
  0.36 -> 0.61 A in steps of 1/0.9.

**md_quench**
- 64-atom SW Si with 0.8 A jitter relaxes to -4.2890 eV/atom.
- One 1000-step chunk at 1300 K, then a quench, gives -4.33634 eV/atom.
  Ideal diamond is -4.3366 eV/atom at a = 5.431 A.
- With NPT MD and `relax_cell`, the cell returns to a = 5.431 A.

**Pressure**
- The wrapped stress equals -(1/V) dH/d(eps) by finite differences, for
  all six components.
- A cell relaxed at 5 GPa ends with the raw potential's diagonal stress
  at 5.000 +- 0.002 GPa.

**Restart and failures**
- A run of 7 trials, restarted to 16, matches the uninterrupted 16-trial
  run in every energy, acceptance and step.
- Injected solver failures are logged with their stage and skipped.
  Three failures in a row stop the run, and the minima found so far are
  still written.
- `min_energy_per_atom` turns a too-deep quench into a failure.

**MD continuation**
- `initial_velocities` continues an NVT run: 15 + 15 steps equal 30
  steps to 1e-9 A, with the chain made inert (tau = 1e9 fs).
- Inside md_quench, a chunk starts from the previous chunk's exact
  positions and velocities.
- After an MD failure, the next chunk starts from the last good
  positions with a fresh Maxwell-Boltzmann draw.

## GeSn demonstration (Q16)

The script is example/GeSn/anneal/run_anneal_demo.py, and its results
are in summary.csv (example/ is not tracked).  The setup:
- Force fields: fitted Tersoff, and atom-centred ELM with
  `isolated_atom_energy`.
- Structures: test runs 00022 and 00038.
- Short settings: md_quench 800 -> 300 K with 6 quenches; basin hopping
  with 20 trials.
- Relax: FIRE with `relax_cell`.

Tersoff results:
- All four runs finish in about 5 s each.
- Every quench returns to the relax-only basin, within 1.3e-4 eV/atom.
  A single vacancy in a 63-atom cell does not migrate in 3 ps at
  <= 800 K.

The ELM results show the guard working, not the search. With the
`min_energy_per_atom` floor at the DFT energy - 0.3 eV/atom:
- **800 K MD and 0.3 A moves.** All four runs stopped after 5
  consecutive failures. The MD collapsed, or a quench fell 0.4-1.2
  eV/atom below the DFT energy.
- **300 -> 100 K MD.** About 5 of 6 chunks per structure still
  collapsed below the floor.
- **Moves of 0.1 A at 300 K.**
  - On test00022, every trial converged and the walker drifted to
    -4.27 eV/atom. That is 0.27 below the DFT final energy and just
    above the floor, and is almost certainly a hole in the MLIP rather
    than a real minimum.
  - On test00038 the walker stayed in the starting basin.

The model is trained on near-equilibrium vacancy cells only (ADR-0021),
and a global search drives it out of that domain within picoseconds.
`anneal` is therefore also a quick stress test of an MLIP. A floor
relative to a trusted energy is necessary, but it is not sufficient: a
spurious minimum just above the floor passes.
