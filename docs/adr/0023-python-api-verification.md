---
status: accepted (plan only, 2026-10-15)
---

# Verification plan for the Python API

Agreed with the user on 2026-10-15 (grilling, rounds 1-3, Q1-Q19).  The
CLI has been verified command by command (ADR-0003, ADR-0017..0022) and
on real VASP data (ADR-0021).  The Python API has not been checked from
a user's point of view: the 735 tests in `tests/` call it in-process, so
"it runs" is covered, but nobody has tried to do a CLI workflow from
Python using only the documentation.

## Starting point (2026-10-15)

- The top-level `akitsushima.__all__` has about 60 names.  From those
  alone a user can reproduce `relax`, `md`, `gcmc`, `osmc`, `neb`, `tsa`,
  `phonon`, `kappa`, `elastic` and classical `fit`.
- Only in submodules: `GlobalOptimiser` (in `solvers`, not top level),
  `solvers.qha`, `solvers.dielectric`, `solvers.nemd.NEMDSimulation`,
  `io.read_vasprun` / `read_outcar`, `fitting.descriptors.ACSF` /
  `build_table`, `fitting.mlip_engine.MLIPForceField`,
  `fitting.physics.PairBaseline`.
- Only as private helpers in `cli/`: the run split and isolated-atom
  subtraction of `dataset` (`cmd_dataset._split_runs` ...), the baseline
  and short-range assembly of `mlip` (`cmd_mlip._fit_short_range` ...),
  the CSV-to-structures step of `fit` (`cmd_fit._structures_from_csv`).
- The README's Python API example has drifted from the docstring one
  that `tests/test_public_api.py` executes: r_cut = 5.0 A (1.47 sigma),
  the two atoms 2.5 A apart (E = +1.43 eV, strongly repulsive), and the
  neighbour list is built but never passed.
- `pip show akitsushima` reports 0.1.0b2 (stale egg-info);
  `__version__` is 0.2.0a13.  `dist/` holds only a 0.2.0a2 wheel.

## Decisions

1. **Goal (Q1).** Can a user finish each CLI workflow from Python using
   only the README, `help()` and docstrings?  Each workflow passes when
   the API reproduces the CLI's numbers (decision 7).  Physical
   correctness is not re-judged; the CLI tests own it.  A smoke test of
   every exported name is not repeated (`tests/` covers it).
2. **What counts as public (Q2).** Top-level names plus non-underscore
   names in submodules.  Anything that needs a private function or
   `cli/` code is an **API gap**; the list of gaps is the main result.
3. **Environments (Q3, Q13).** The main runs use the development
   environment (miniconda base, editable install, all extras).  A small
   extra check runs in a fresh venv in the scratchpad:
   - wheel built from HEAD with `python -m build --wheel --no-isolation`;
   - numpy and scipy from PyPI first, then pyyaml; torch, numba and
     mpi4py deliberately absent;
   - checks: import, `pip show` against `__version__`, torch not
     imported eagerly, a clear error from `NNModel` without torch, the
     README example on the pure-Python fallback, a clear CLI error
     without yaml.
   Afterwards `pip install -e . --no-deps` refreshes the stale metadata.
4. **Findings (Q4).** Bugs (crash, wrong number, docstring contradicting
   behaviour) are fixed at once with a regression test.  API gaps and
   default mismatches are only recorded; whether to add public API is
   decided in a separate grilling once the full list is known.
5. **Documentation is specification (Q5).** Code examples in the README
   and in docstrings of the public names are run.  The README example is
   aligned with the docstring one and executed by a test in the same way
   as `test_public_api.py`.
6. **Scope (Q6, Q14).**
   - GeSn chain (ADR-0021 data): `dataset`; `fit` SW and Tersoff (the
     Tersoff fit capped at about 5 min of evaluations); `mlip` one case
     per model family: tab_lin_ridge, tab_elm, tab_qelm, tab_nn, ac_elm,
     ac_qelm, ac_nn, ac_elm_iso_zbl, tab_elm_fromtable, ext21_elm;
     `relax` with SW, Tersoff and ac_elm_iso on the 10 test structures;
     a short `anneal` (md_quench and basin hopping, SW) and a short NVT
     `md` (SW).
   - One case for each other command, from the existing example inputs,
     shrunk to under about 5 min per run:

     | command | input | system |
     |---|---|---|
     | phonon | example/phonon/Si | Si Tersoff |
     | elastic | example/elastic/Si | Si Tersoff |
     | gruneisen, qha | example/qha/Si | Si Tersoff |
     | dielectric | example/dielectric/MgO | MgO |
     | piezo | example/dielectric/ZnO/piezo (after elastic) | ZnO Buckingham |
     | neb, tsa | example/neb/Si_vac_2x2x2 | Si SW |
     | gcmc | example/gcmc/ideal_gcmc | LJ |
     | osmc | example/gcmc/ideal_cfcmc | LJ |
     | esm | example/esm/solvers_cli/relax | Buckingham |
     | kappa | example/kappa/T50/seed11 | LJ Ar |
     | nemd | example/nemd/L12 | LJ |

     CLI and API get the same shrunk settings.
7. **CLI-API agreement (Q7).** The old GeSn outputs predate the NN
   standardisation (04964f9), `min_feature_std` (9d8801e) and the FIRE
   fix (2e2c92a), so every compared case is rerun with the CLI on HEAD,
   twice.
   - If the two CLI runs agree bit for bit, the API must too.
   - Otherwise deterministic results (parameters, predictions, relaxed
     E and V, frequencies, C_ij) agree to a relative
     max(1e-8, 10 x the CLI-CLI difference).
   - Trajectories (md, anneal, gcmc, osmc, nemd, kappa): the first 50
     steps to 1e-8, then statistics only (mean T, mean E) within their
     own standard error.
   - UNCERTAIN: 1e-8 and 50 steps are first estimates, revisited once
     the CLI-CLI noise is measured.  `engine/_pair_kernels.py` uses
     numba `parallel=True`, so bitwise reproducibility is not assumed.
8. **Defaults (Q8).** For the comparison the API script sets every
   parameter to the CLI's value.  Separately, a table lists, per
   command, the CLI template default against the API constructor
   default; differences are findings (decision 4).
9. **How the scripts are written (Q9, Q17).** I write them from the
   README, `help()` and docstrings first, and log every time I have to
   read the source (what I needed, where I found it).  Because I know the
   internals, a clean-room subagent calibrates that log:
   - task: train tab_elm and ac_elm_iso_zbl, relax the 10 test
     structures;
   - may read: README, `help()` / `pydoc`, `--dump-default` templates,
     the GeSn data and `dataset` output;
   - may not read: `akitsushima/` source, `tests/`, `docs/adr/`,
     `example/` scripts and logs, my scripts;
   - same model as mine; it reports its script, every point where it got
     stuck and every file it read.  Isolation is by instruction only.
   The user approves the subagent again right before it starts.
10. **Gaps that block a workflow (Q15).** Up to about 50 lines of user
    code: reimplement in the script, compare that step with the CLI too,
    record the line count.  Larger: use the CLI's output file for that
    step, mark the step "blocked", record it.
11. **Data flow (Q16).** Steps pass objects in memory (fit result ->
    force function -> relax).  Each model family also gets one
    save/load round trip (`save_potential` / `load_potential`,
    `model.save` / `load`, `load_mlip_model`).
12. **Layout (Q10).** `example/api/` (not tracked): one script per
    workflow, a resumable runner `run_api_tests.py` with a JSONL ledger,
    and a judge.  Scripts run with a working directory outside the
    repository and import the installed package (no `sys.path` edits).
    Checks that take seconds move to `tests/test_api_workflows.py`.
13. **Order (Q18, Q12).**
    1. this plan;
    2. README example and documentation tests;
    3. clean venv;
    4. CLI baselines, two runs per case, in the background;
    5. GeSn chain through the API, defaults table;
    6. subagent in the background, the other commands through the API;
    7. promote fast checks to `tests/`, results and gap list here, then
       a grilling on the gaps.
    A1 (ADR-0010) stays the top priority: when the user's VASP results
    (b_round0, i_reference_relax) arrive, work switches at the next
    workflow boundary.
14. **Commits (Q19).** This plan first; each fix with its regression
    test as its own commit; the version goes to 0.2.0a14 once, with the
    results.  Commits are local; push and tags are asked for each time.

## Results

Pending.
