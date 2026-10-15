---
status: accepted (implemented 2026-10-15, 0.2.0a14 and 0.2.0a15)
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

## Stage 2 (Q22, defaults) -- partly done

Changed to the CLI's values: tabular `ELMModel` / `QuantizedELMModel`
`eps0` 0.01 (recorded with the model only; training weights come from
the table), `build_table` / `ACSF.table_rows` `eps0` 0.01 (the S-row
weight eps0*V), `fit_potential` `de_maxiter` / `de_popsize` 300 / 10,
`GeometryOptimiser` `timing` True and `output_dir` ./relax_output.  The
timing summary now prints only when `verbose` is on, so the internal
relaxations of elastic, qha, dielectric and tsa stay quiet.
`tests/test_api_workflows.py` pins each default to the CLI's.

NVT `tau_nhc` was then decided the other way (user, 2026-10-15): the
CLI's 20 fs fallback is below the 50-200 fs NVTSimulation documents as
typical, and kappa, the diffusion analyser and anneal rely on 100 fs, so
the CLI `md` fallback and the nvt / npt templates became 100 fs; the API
is unchanged.  (UNCERTAIN which coupling time suits solids best; a
reference for Nose-Hoover chain coupling times would settle it.  The
anneal template keeps its tuned 20 fs.)

## Stage 3 (Q23-Q26)

- **MLIP API (Q23).** Every model constructor takes `baseline=`
  (a PairBaseline) and `isolated_atom_energy=`, written into the same
  `physics_config` the CLI writes; `model.baseline`, `model.set_physics()`.
  Tabular models: `fit(..., descriptor=)` sets `descriptor_config`, and
  `fit_structures(structures, descriptor)` (user decision, 2026-10-15)
  builds the table minus the baseline and fits -- a tabular `fit` gets a
  table, so it cannot subtract the baseline itself.  Atom-centred models
  subtract their baseline in `fit` and `score_structures`; the CLI now
  passes raw structures.  One class decorator per kind
  (fitting/models/_api.py) adds all of this without touching each
  signature.  `akitsushima.io.load_training_set()` is the fit-layout loader
  (moved from cli/cmd_fit.py).  Tests: the CLI and the API agree bit for
  bit with a baseline that reaches the data, and, independently of the
  CLI, a model with `baseline=` equals a baseline-free model on
  subtracted labels plus the baseline (subtracted exactly once).
- **Q24.** `akitsushima.io.build_training_set()` does the dataset work
  (moved from cli/cmd_dataset.py; the CLI keeps reading the YAML and
  printing).  On the GeSn data the CLI's training_set and log are
  byte-identical to the baseline run.  `akitsushima.force_fn_from_config()`
  is the YAML `potential:` bridge, with `fn.r_cut` the CLI's cutoff.
- **Q25.** README: a self-contained Python MLIP example (SW silicon as
  stand-in data, ACSF, AtomicELMModel, save, relax), executed by
  tests/test_public_api.py (relaxed energy within 0.02 eV/atom of SW's
  own diamond minimum).  `read_vasprun` / `read_outcar` docstrings list
  the frame keys, units and stress convention; `fit_potential` and the
  MLIP `fit` docstrings point at `load_training_set` for the structure
  dicts.  The RDF, TimingCollector and Buckingham `make_force_fn`
  docstring examples are self-contained.
- **Q26.** matplotlib: `have_matplotlib()` warns once, when a plot is
  asked for, not on import (O1).  `fit --config` takes the config's
  `potential:` unless `--potential` is given (O4).  The ELM score table
  prints E in eV/atom (O8).  The relax template's `fix_cell` example is
  the (3, 3) mask the code takes (O9; the 6-element "Voigt mask" was
  rejected).  `seed` vs `val_seed` is stated in the fit docstrings.
