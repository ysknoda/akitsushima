---
status: accepted (stage 1 implemented 2026-10-15)
---

# Closing the Python API gaps and aligning the API defaults with the CLI

Agreed with the user on 2026-10-15 (grilling Q20-Q27, all
recommendations accepted) after the Python API verification of ADR-0023,
whose findings (G1-G9 gaps, O1-O10 observations, the default mismatches
and the clean-room stuck points) this ADR acts on.

## Decisions

1. **Order (Q20, Q27).** O7 first, because it bears on A1 (ADR-0010):
   fix, then re-judge the ADR-0021 relaxations, release 0.2.0a14.  Then
   Q22 (defaults), Q23 (MLIP API), Q24 (dataset and YAML potential as
   library functions), Q25 (documentation), Q26 (small findings), released
   together as 0.2.0a15.  Commits are local; push and tags are asked for.
2. **O7: a force function may declare that it does not use the solver's
   neighbour list (Q21).** `uses_neighbour_list = False` on
   `MLIPForceField` (its ACSF does its own image search),
   `ExternalFileEngine` and `ScaledForceEngine` (they call their engine
   without a list).  The solvers build their list through
   `engine.neighborlist.neighbour_list_for(force_fn, r_cut, skin)`, which
   returns a `NullNeighbourList` (same interface, no pairs, never refuses
   a cell) for such a function; `ElasticConstantsCalculator` skips its
   strained-cell check for it; `enthalpy_force_fn` (anneal under
   pressure) carries the declaration through.  Every other force function,
   so every classical potential, keeps the minimum-image precondition.
3. **Q22.** The API defaults take the CLI's values: NVT `tau_nhc` 20 fs,
   tabular ELM / QELM and `build_table` `eps0` 0.01, `fit_potential`
   `de_maxiter` / `de_popsize` 300 / 10, GeometryOptimiser `timing` True
   and an `output_dir` that does not litter the working directory.
4. **Q23.** `ELMModel.fit(..., descriptor=)` (and the other tabular
   models) sets `descriptor_config` itself; the MLIP constructors take
   `baseline=` and `isolated_atom_energy=` and `fit` subtracts the
   baseline; `akitsushima.io.load_training_set()` reads the fit layout.
   The CLI uses the same functions.
5. **Q24.** The `dataset` logic moves into a library function
   `build_training_set()` and the YAML `potential:` bridge becomes the
   public `force_fn_from_config()`; G6-G9 stay recorded only.
6. **Q25.** A Python MLIP workflow example in the README (executed by a
   test), the dict keys / units / stress convention of `read_vasprun`,
   `read_outcar` and the training structure dicts in their docstrings,
   and self-contained docstring examples where a few lines suffice.
7. **Q26.** matplotlib warns only when a plot is asked for (O1); `fit`
   keeps the config's `potential:` unless `--potential` is given (O4); the
   fit table prints E RMSE in eV/atom (O8); the `fix_cell` format is
   checked and documented consistently (O9); the docstrings say what
   `seed` and `val_seed` each fix.

## Stage 1 result (O7, 0.2.0a14)

`tests/test_mlip_thin_cells.py`: on an 8-atom Si cell (half width 2.7 A
against ACSF r_cut 3.771 + skin 0.5 A) an MLIP relax and NVT run, and the
relax equals the former workaround (`r_cut=1.0`) bit for bit; Stillinger-
Weber on the same cell is still refused.  The pre-fix code refused the
MLIP run.

**ADR-0021 relaxations re-judged** (example/api/rejudge, same models and
YAML, fixed code).  The minimum-image check had stopped 25 of the 228
FIRE runs and 61 of the 228 L-BFGS runs; all but one (an L-BFGS SW run,
a correct refusal) were MLIPs.  Rerun:

- FIRE, the judged set: of the 17 refused test-structure runs, 4 now
  pass (all `tab_lin_pls`), 3 do not converge, 10 converge with |dV/V|
  > 5 % (-18 to -73 %).  The 8 bulk runs (recorded only) all shrink by
  more than 5 %.  The FIRE pass count goes from 138 to 142 of 190.
- L-BFGS (recorded): of 48 refused test runs, 1 passes, 13 do not
  converge, 34 shrink by more than 5 %.

So the check mostly stopped runs that were already collapsing; it did
not hide working models.  The A1 model (atom-centred ELM with
`isolated_atom_energy`) was never refused, and A1's conclusion, a
data-coverage problem, stands.

During the full test run 23 MPI tests failed to start `mpirun` while
other MPI-free jobs ran; rerun alone they pass (environment, not code).
