---
status: proposed (2026-11-20)
---

# Active learning driven by the ensemble uncertainty

Tier B item 2 of ADR-0028 (addendum 5, Q15 of the roadmap grilling).
Agreed with the user on 2026-11-20 (grilling, Q1-Q17).

## Starting point (2026-11-20, 0.2.0b1)

- ADR-0033: `AtomicELMEnsemble` (atom-centred ELM and QELM, members
  differ only in the seed).  The mean is the force field; the spread
  gives sigma_E [eV/atom], sigma_F,max [eV/A] and sigma_S [GPa] through
  `MLIPForceField.uncertainty()`.  md and relax log a `sigF_max` column on
  the logged steps, the anneal ledger carries the three sigmas of every
  minimum, and `mlip.evaluate` writes them to CSV for a list of
  structures.  On the GeSn round-5 data the spread flagged the members'
  disagreements at the DFT-checked anneal minima.
- ADR-0027 rounds 3-5: every round of new GeSn data was assembled by
  hand -- choosing structures, running VASP, `dataset`, retraining.  A
  63-atom VASP single point takes about 1 h at np 4 and about 10 h on one
  core.
- ADR-0021: VASP readers and the `dataset` command (split by run,
  stratified by composition).  ADR-0011: `type: external` potentials read
  the energy, forces and stress of an external code.
- The re-run of ADR-0027 A1 with an ensemble is outside the roadmap
  (ADR-0028 addendum 5, Q1) and outside this ADR.

## Decisions

1. **Scope (Q1).**  A staged loop: explore -> select -> label -> train,
   one directory per round, each stage recorded in a ledger so that a
   rerun resumes where it stopped.  Labelling either writes the inputs
   and stops (default; the user runs DFT elsewhere and reruns the
   command) or runs a command and continues unattended.
2. **Acceptance (Q2).**  By a synthetic oracle: an SW potential plays
   the role of DFT, and active learning is compared with random
   selection from the same candidates (decision 13).  A round on real
   GeSn data is a separate research decision.
3. **Explorers (Q3).**  md (NVT at several temperatures) and anneal
   (`md_quench`), with their existing settings nested in the
   configuration.  The candidates are their saved trajectory frames and,
   for anneal, the minima.  The explorer list is open, so relax
   trajectories and `build` perturbations can be added later.
4. **Selection (Q4).**  A trust window on sigma_F,max as in DP-GEN
   (Zhang et al., Comput. Phys. Commun. 253, 107206 (2020)):
   - sigma < lo: accurate, not selected;
   - lo <= sigma <= hi: candidate;
   - sigma > hi: failed (unphysical), not selected.
   The candidates are then thinned to at most K per round by
   farthest-point sampling (decision 16), so that near-identical frames
   of one trajectory are not all sent to DFT.
5. **Thresholds (Q5).**  By default from the distribution of
   sigma_F,max over the training structures: lo = its 99th percentile,
   hi = 10 lo.  Absolute values (eV/A) override them.  The seed-only
   ensemble is not calibrated in absolute terms (ADR-0033 decision 3),
   and a relative scale carries over between systems.
6. **Cell size (Q6).**  Exploration runs directly in cells that DFT can
   label (64 atoms or fewer).  Carving small cells out of large MD cells
   around uncertain atoms is not part of this ADR.
7. **Retraining (Q7).**  Each round refits the ensemble from scratch on
   the enlarged training set with the same settings and seeds (5 members
   by default).  The regularisation and scale searches run on the first
   member as in ADR-0033 decision 8.  Incremental updates (OS-ELM) stay
   in tier C.
8. **Command (Q8).**  `akitsushima active active.yaml`, and `ak.active`
   in Python.  Sections:
   - `model`: the starting model (an npz, or an `mlip` training section)
     and its data;
   - `explore`: a list of explorers, each an md or anneal section with
     its start structures;
   - `select`: lo, hi (relative or absolute), K;
   - `label`: decision 11;
   - `train`: overrides of the training settings, if any;
   - `rounds`: the maximum number of rounds.
   Output: `round_000/`, `round_001/`, ... each with `explore/`,
   `select/`, `label/` and `train/`, and `active_ledger.jsonl` at the
   top.  Running the same command again resumes.
9. **Stopping a trajectory (Q9).**  md and anneal get a general option
   `stop_sigma: {threshold, consecutive}`: a trajectory (for anneal, the
   current quench) stops when sigma_F,max exceeds `threshold` on
   `consecutive` logged steps in a row (default 3).  The stop is written
   to the log and to the ledger.  In `active`, threshold defaults to hi.
   The option is usable outside active learning.
10. **Ending the loop (Q10).**  At `rounds`, or earlier when the explored
    frames are at least 99 % accurate (sigma < lo) and none failed
    (sigma > hi).  The 99 % is the value commonly used with DP-GEN and is
    UNCERTAIN; it is reviewed with the validation result.
11. **Labelling (Q11).**  Two modes:
    - `vasp_inputs`: an INCAR template, the POTCAR path for each element
      (referenced, never copied) and the k-point density K [A], with
      k_i = ceil(K / L_i) as in ADR-0027.  The command writes one
      directory per structure and stops; on the next run it reads the
      results with the ADR-0021 readers.
    - `external`: an ADR-0011 `type: external` potential section, run
      for each selected structure (`n_jobs`, decision 17).  The
      validation uses `akitsushima.tools.akitsushima_efs` with SW; a
      local VASP can be wrapped the same way.
    A structure whose SCF did not converge, or whose output cannot be
    read, is recorded in the ledger and dropped.
12. **New data (Q12).**  Labelled structures are added to the training
    split only; validation and test keep the first split, so errors are
    comparable between rounds.  They enter the data set as runs tagged
    `al_round`.
13. **Validation (Q13).**  Run under nohup outside the test suite, the
    result reported here.
    - Oracle: the GeSn multi-species SW of the CLI tests (`fit_sw`).
    - Start: 20 frames of 300 K SW md of a 64-atom Ge0.875Sn0.125 cell.
    - Explorers: md NVT at 300, 900 and 1500 K, and anneal.  K = 10 per
      round, 5 rounds.
    - Baseline: the same number of structures drawn at random from the
      same candidate pool each round.  Three seeds of the start set.
    - Metrics: RMSE of E and F on held-out frames of SW md at 300-1500 K;
      anneal collapse rate (shortest distance below 2.0 A, or E more
      than 0.05 eV/atom below the oracle).
    - Pass: in the last round the F RMSE of active learning is below
      that of random selection by more than the standard deviation over
      the seeds (UNCERTAIN).  The collapse rate is reported only.
    - Tests added to the suite: at most 60 s of `slow` tests in total (a
      small SW case of one round, resume, `stop_sigma`, both label
      modes).  Cores for the validation run are asked for at the time if
      the ADR-0025 ladder is still running.
14. **Recording and version (Q14).**  This ADR is written before the
    implementation and accepted with the result.  ADR-0028 addendum 5
    item 2 and ADR-0033 link here.  The version goes to 0.3.0a1.  Local
    commits; pushes and tags are asked for.
15. **Uncertainty of the candidates (Q15).**  At selection every saved
    frame is evaluated again with the ensemble, so the candidates do not
    depend on the logging interval.
16. **Farthest-point space (Q16).**  One vector per structure: the
    per-species mean of the model's ACSF (the species blocks
    concatenated), standardised with the mean and standard deviation over
    the training structures.  The first point is the candidate with the
    largest sigma_F,max.  K defaults to 10.
17. **Parallel labelling (Q17).**  `n_jobs` for the external mode,
    default 1; one ledger row per structure, so an interruption loses at
    most one labelling.
