---
status: accepted (2026-11-21; implemented in 0.3.0a1 -- see Result)
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

## Implementation (2026-11-20)

- `akitsushima/active.py`: `ak.active(config)` (a YAML path or a dict)
  and the stages; `akitsushima active active.yaml` (`cli/cmd_active.py`)
  exits 0 when the loop ended, 2 while VASP labels are awaited, 1 on an
  error; `--dump-default active`.
- Rounds: round r explores with the model of round r - 1 (round 0: the
  initial model, from `model.file` or trained into `initial/`), and its
  `train/` holds the next model.  The ledger has one row per explorer run
  and per labelled structure, so an interruption loses at most one of
  them.
- Explorers run the md and anneal commands in process with
  `traj_format: extxyz`; their YAML and output stay in
  `round_NNN/explore/<name>/<k>/`.  Seeds: `seed + 1000 r + k` added to the
  explorer's own.
- Selection writes `select/frames.csv` (every frame, its sigma and class)
  and one `POSCAR` per selected structure.  `strategy: random` draws from
  the frames that did not fail (sigma <= hi): the failed frames are
  unphysical and both strategies discard them.  Decided while writing the
  validation, where a draw from all frames put collapsed MLIP-md frames
  into the random arm's training set.
- `stop_sigma` (decision 9): `set_stop_sigma(threshold, consecutive)` on
  the four MD classes, checked on logged steps; a stopped run writes no
  final checkpoint and sets `sim.stopped`.  `md.stop_sigma` in the md
  command, `stop_sigma` in the anneal section, where a stop fails the
  cycle at stage md like any failed chunk.
- `train_only` (decision 12): `_prepare` of the atom-centred models draws
  the validation split over the structures without the flag, so the
  split is the one they get alone; unflagged data take the old path
  unchanged.
- VASP read-back keeps the geometry of `vasprun.xml`, not the POSCAR
  written for it; a run whose last ionic step used NELM electronic steps
  counts as not converged.
- Tests: `tests/test_active.py` (10 tests, 7 slow), and the
  template test runs `--dump-default active` as shipped (rc 2, waiting).
  The template's ACSF lacked radial and angular functions and did not
  run; it now carries those of the CLI tests.
- First validation run (2026-11-20): from 20 frames of 300 K md the
  initial ensemble had sigma_F,max p99 0.50 eV/A on its training data but
  63 eV/A on the explorers' start structure (zscore whitening 27, n_hidden
  100: 21), so every explored frame was above hi = 10 lo, nothing was
  selected and the loop retrained the same data.  The user chose (over a
  larger default hi_factor, or a broader start set for the validation
  only): when fewer than k frames can be chosen, both strategies fill up
  from the failed frames with the smallest sigma, skipping exact
  duplicates; `select` records `n_filled_from_failed` and frames.csv the
  class `failed_filled`.  A round with nothing selected that has not
  converged ends the loop with status `stalled`.
- Second validation run (2026-11-20, 1.2 h, fill from failed, no
  distance guard): failed the pass criterion.  Seed 1 worked (last-round
  test F RMSE 0.123 eV/A with active learning against 0.188 with random
  selection; anneal collapses of the final model 0/24 against 13/24).
  Seeds 2 and 3 diverged (test F RMSE up to 1e20 eV/A).  The cause was in
  the labels: structures from collapsed MLIP trajectories, with atoms
  0.004-0.9 A apart, for which SW returns E up to 1e13 eV/atom and forces
  up to 1e26 eV/A; one such structure is enough to wreck the
  least-squares fit.  In seed 2 five of them came in through the fill in
  round 0.  The random arm drew one with sigma below hi (shortest distance
  0.94 A): a poor model's sigma does not flag every collapse.  Results
  kept in `example/GeSn/active_validation/*_unguarded*`.
- The user chose a distance guard: a frame whose shortest interatomic
  distance (all images) is below `select.min_distance` -- by default
  `min_distance_factor` 0.75 times the shortest distance over the training
  structures -- is class `unphysical`: never a candidate, never drawn by
  the random arm, never filled, and it counts against convergence.  One
  value for all species pairs, not one per pair: a dilute species (8 Sn in
  64) may have no nearest-neighbour pair of its own in the training data,
  and its reference would then be a second-neighbour distance.  With the
  validation's start sets the training shortest distance is 2.28 A, so the
  guard is 1.71 A; the equilibrium bonds are Ge-Ge 2.450 A and Sn-Sn
  2.811 A (a sqrt(3) / 4 with a = 5.658 and 6.493 A).  The factor 0.75 is
  UNCERTAIN; it is reviewed with the third run.
- Offline check before the third run (retraining the unguarded runs'
  labels without the guarded structures): the divergence goes (seed 2
  round 4: test F RMSE 0.72 eV/A with 4 of 45 labels kept), but seed 3
  stays poor -- 16.5 eV/A after round 1 with 18 of 20 kept, whose
  structures are 1.95-2.1 A apart at -2 to -3.4 eV/atom (SW minimum about
  -4.04); an additional cap of 1 eV/atom above the training maximum still
  gave 9.2 eV/A.  Seed 3's random arm, with only physical labels, had a
  test E RMSE of 0.8-2.0 eV/atom in rounds 0-3 against 4-6 meV/atom for
  seed 1: from 20 frames of one 300 K cell the ELM's extrapolation depends
  on the start set.  The offline check does not reproduce the guarded
  selection, which picks other structures; the third run decides.
- Third validation run (2026-11-21, 1.35 h, with the guard): no run
  diverged; the guard removed 5 frames in seed 2 round 0 and 3-4 in
  seed 3.  Test F RMSE (eV/A) after the last round:

  | seed | active learning | random | random - AL |
  |---|---|---|---|
  | 1 | 0.123 | 0.188 | 0.065 |
  | 2 | 0.293 | 0.436 | 0.143 |
  | 3 | 0.343 | 0.566 | 0.223 |

  Active learning is lower on every seed (mean difference 0.144 eV/A),
  but the pre-agreed criterion (mean difference larger than the standard
  deviation over seeds of either arm: 0.116 for AL, 0.192 for random) is
  not met, because the spread between seeds is larger than the
  difference.  A paired view (the difference has standard deviation
  0.079, t = 3.2 with 2 degrees of freedom, one-sided p about 0.04) is a
  criterion chosen after the data and is reported only.  Test E RMSE
  (meV/atom) is 2.3 / 3.8 for seed 1 but 48 / 227 for seed 2 and 773 / 114
  for seed 3 (AL / random), and already 2100 and 160 for the initial
  models of seeds 2 and 3 against 20 for seed 1: the energy
  extrapolation of an ensemble trained on 20 frames of one 300 K cell
  depends on the start set, in both arms.  Anneal collapses of the final
  models (out of 24 minima, AL / random): 0 / 13, 16 / 8, 16 / 19; every
  initial model collapsed 24 / 24.  Figure:
  `example/GeSn/active_validation/validation.png` (log scale).
- Revised criterion (user, 2026-11-21, fixed before seeds 4-6 were run):
  the seed-to-seed spread measures how much the start set matters, not
  whether active learning helps on a given start set, so the arms are
  compared in pairs.  Three more start sets (seeds 4, 5, 6; the test set
  stays that of cell 999) are run with the unchanged loop.  With
  d_s = F RMSE(random) - F RMSE(AL) after the last round of seed s, the
  validation passes when both hold:
  1. over seeds 1-6, mean(d) / (sd(d) / sqrt(6)) > 2.015 (one-sided
     paired t-test at 5 %, 5 degrees of freedom; Student-t quantile
     t_0.95,5 = 2.015);
  2. over the fresh seeds 4-6 alone, mean(d) > 0, so the criterion
     chosen after seeing seeds 1-3 is checked on data it has not seen.
  The anneal collapse rate stays reported only.

## Result (2026-11-21, 0.3.0a1)

Synthetic-oracle validation (decision 13, with the paired criterion of
the implementation section), six start sets, 5 rounds of K = 10, the
distance guard on.  Test F RMSE [eV/A] after the last round, and test E
RMSE [meV/atom]:

| seed | F, AL | F, random | d = random - AL | E, AL | E, random |
|---|---|---|---|---|---|
| 1 | 0.123 | 0.188 | 0.065 | 2.3 | 3.8 |
| 2 | 0.293 | 0.436 | 0.143 | 48 | 227 |
| 3 | 0.343 | 0.566 | 0.223 | 773 | 114 |
| 4 | 0.346 | 0.385 | 0.038 | 234 | 8.5 |
| 5 | 0.110 | 0.130 | 0.019 | 2.2 | 3.0 |
| 6 | 0.109 | 0.136 | 0.028 | 182 | 87 |

- Criterion 1: mean d 0.086, sd 0.081, t = 2.61 > 2.015.  Criterion 2:
  the fresh seeds 4-6 have mean d 0.028 > 0 (each positive).  **The
  validation passes**; F is lower with active learning on all six start
  sets.  The old criterion (mean d above the spread over seeds) is still
  not met: the start set changes the result more than the selection.
- The gain on the fresh seeds is small (0.02-0.04 eV/A, 10-20 %); on
  seeds 1-3 it is 0.07-0.22 eV/A.
- Energy is not better: the selection uses sigma_F,max only, and E RMSE
  is lower with active learning on seeds 1, 2 and 5 and higher on 3, 4
  and 6.  The start sets with a poor initial E (160-2600 meV/atom; seeds
  1 and 5: 14-20) stay poor in both arms: 20 frames of one 300 K cell
  fix the forces well enough but not the energy far from that cell.
- Anneal collapses of the final models (24 minima each, AL / random):
  0 / 13, 16 / 8, 16 / 19, 0 / 24, 0 / 0, 18 / 0; 50 against 64 in
  all.  Every initial model collapsed 24 / 24.  Not a clear difference.
- Decision 10 (stop at 99 % accurate and none failed): no arm reached it
  in 5 rounds, so the 99 % stays untested.
- The fill from the failed frames was used in round 0 of 11 of the 12
  arms (the initial models put most explored frames above hi; all but
  seed 3's random arm), and later in rounds 1-2 of seed 2's AL arm and
  round 1 of seed 6's random arm.  The guard removed 3-5 frames in round
  0 of seeds 2-5 (both arms) and 1 frame in round 1 of seed 3's random
  arm; with it, round 0 of seeds 2, 4 and 5 labelled only 3-6 structures
  instead of 10.
- Run time: 1.2-1.35 h per three seeds on one core (oracle labels are
  free; with DFT the labels dominate).
- Figure: `example/GeSn/active_validation/validation.png`; tables:
  `report.md` there.  The unguarded run is kept beside them
  (`*_unguarded*`).

Consequences for use: the command is fit for driving GeSn DFT rounds
(selection with a trust window, the distance guard, resumable labels),
but its benefit over random picks from the same explorers is modest per
round, and it does not repair a narrow start set's energy.  A start set
that spans the explorers' temperatures, and an energy-aware selection
(sigma_E as a second criterion), are the obvious next steps; neither is
decided here.

Full suite: 982 passed, 1 skipped (13.3 min, one thread) after the
guard.
