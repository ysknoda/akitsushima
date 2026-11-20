---
status: accepted (2026-11-18; implemented in 0.2.0a23 -- see Result)
---

# Ensemble ELM: mean as the force field, spread as the uncertainty

Tier A step 8 of ADR-0028.  It realises ADR-0010 item 7.  Agreed with
the user on 2026-11-18 (grilling, Q1-Q11).

## Starting point (2026-11-18, 0.2.0a22)

- **ADR-0010 item 7** (agreed 2026-09-28): several ELMs that differ
  only in the random seed (W, b), trained on the same data; the spread
  of their predictions estimates the model uncertainty.  Three points
  were left open: the number of members, whether md and relax report
  the spread, and how the members are stored.
- **ADR-0027 (rounds 3-5).**  The spurious anneal minima of the GeSn
  atom-centred ELM are random-feature noise of each model, and more
  data does not remove them.
  - Other seeds put the same cells within +-5.4 meV/atom.
  - On test00026 the MLIP-DFT error ranges from +5.6 to -44.1 meV/atom
    across models.
  - 18 anneal minima of single models have been checked against DFT
    (`example/GeSn/dft_local/check_r*`).
- **Cost.**  The members share the ACSF descriptor, which is most of a
  force call, and add only their ELM heads.  With 400 hidden units the
  head took 44 ms of a 4096-atom call that takes about 440 ms after
  ADR-0030.
- **Fitting.**  One `AtomicData` keeps the ACSF Jacobians of the
  training set (`cache=True`), so each further member costs one
  normal-equation build and one linear solve.
- **The engine interface.**  `MLIPForceField` asks the model for
  `atomic_energy_and_gradient(G, sp)` and contracts the gradient with
  the stored Jacobian.  The pair baseline and the isolated-atom limit
  come from the model's `physics_config` (ADR-0014).

## Decisions

1. **Role (Q1).**  The mean of the members is the force field.  Its
   forces and stress are the exact derivatives of the mean energy, so
   it stays conservative.  The spread of the members is the
   uncertainty.
2. **Models (Q2).**  Atom-centred ELM and QELM only.  All members share
   the descriptor and its Jacobian.
3. **What differs (Q3).**  Only the seed of the random hidden layer;
   the data are the same for every member.  A bootstrap of the
   structures is considered only if the evaluation shows the spread to
   be poorly calibrated.
4. **Members (Q4).**  Default 5.
5. **Reported quantities (Q5).**
   - sigma_E, the standard deviation of the members' energy per atom
     [eV/atom];
   - sigma_F,max, the largest over atoms of the RMS deviation of the
     members' force vectors from their mean [eV/A];
   - sigma_S, the largest standard deviation over the stress components
     [GPa].

   Where they appear:
   - a `sigF_max` column in the md and relax logs, filled on the logged
     steps only;
   - sigma in the anneal ledger for every minimum;
   - an evaluation mode of the `mlip` command that writes the three
     values of a list of structures to CSV, the minimal tool for
     choosing the next DFT batch.
6. **Storage (Q6).**  One npz holds all members, with one descriptor
   configuration and one physics configuration.  `load_mlip_model`
   recognises it.
7. **Validation (Q7).**
   - Unit tests:
     - the mean forces and stress match finite differences of the mean
       energy;
     - identical members give zero spread;
     - a saved and reloaded ensemble gives the same results;
     - an md run with an ensemble still restarts bit for bit.
   - Evaluation on the GeSn round-5 data, comparing the ensemble with
     single members:
     - the test RMSE;
     - the error at the 18 DFT-checked anneal minima;
     - whether sigma at those minima stands out from its distribution
       over the training structures.

   The feature is adopted whatever the evaluation shows.  If the spread
   does not flag the erroneous minima, that is recorded as its
   limitation.
8. **Hyperparameters (Q8).**  The regularisation and scale searches run
   on the first member only.  The other members reuse the chosen
   values, so they differ in the seed alone.
9. **Configuration and seeds (Q9).**
   - `ensemble: 5` in the model settings.
   - The members' seeds are `seed + 1000 m`, which keeps clear of the
     per-species offset `seed + s`.
   - `ensemble_seeds: [...]` gives them explicitly.  The evaluation
     includes the seeds of the existing single models (42, 1 and 2).
10. **ADR-0027 A1 (Q10).**  Repeating the anneal acceptance check with
    the ensemble is not part of this step.  It is decided separately
    after the evaluation.
11. **Recording (Q11).**  This ADR is written before the implementation
    and accepted with the result.  ADR-0010 item 7 links here.  The
    version goes to 0.2.0a23.

## Result (2026-11-18, 0.2.0a23)

**Implementation.**
- `AtomicELMEnsemble` (`fitting/models/ensemble.py`, exported as
  `ak.AtomicELMEnsemble`).
  - `fit` builds one `AtomicData`.  The first member runs the searches;
    the others reuse the chosen reg, scale factor and correlation
    length.
  - `atomic_energy_and_gradient` returns the mean of the members, and
    `member_energies_and_gradients` all of them.
  - The train and val scores are those of the mean.
  - `save` stores each member's own npz as bytes inside one npz;
    `AtomicELMModel.load` now also accepts a file-like object for this.
    `load_mlip_model` recognises the class.
- `MLIPForceField`:
  - For an ensemble it evaluates the members, returns the mean and keeps
    the last call per thread.
  - `uncertainty()` returns sigma_E, sigma_F_max and sigma_S (sample
    standard deviations, ddof 1) of that call, or None for a single
    model.
  - The members' forces are contracted only when `uncertainty()` is
    called.
- md (NVT, Langevin, NVE, NPT) and relax append a `sigF_max(eV/A)`
  column to their logs when the force field is an ensemble.
- anneal adds sigma_E, sigma_F_max and sigma_S to every ledger record of
  a minimum.  It costs one extra force call per quench.
- CLI `mlip`:
  - `ensemble` and `ensemble_seeds` select the ensemble.
  - `mlip.evaluate: {model, poscars, csv, poscar_dir, output}` writes a
    CSV with E per atom and the three sigmas.  For labelled structures
    it adds the errors against the reference.  The template shows both.
- `tests/test_ensemble_elm.py` (7 tests):
  - the mean equals the members' mean and passes the fourth-order
    finite-difference check;
  - identical members give zero spread, a single model gives None;
  - default and explicit seeds, with one search shared;
  - save and load;
  - an interrupted md run with an ensemble continues bit for bit,
    `sigF_max` included;
  - the anneal ledger and relax log carry sigma;
  - the CLI fits an ensemble and evaluates it.

**Evaluation on GeSn round 5** (`example/GeSn/ensemble/`, not tracked):
- Settings: those of `ac_elm_iso_r5` (n_hidden 1600, isolated-atom
  limit, 656 training structures) with seeds 42, 1, 2, 3 and 4.
  Seeds 42, 1 and 2 are the single models r5, r5s1 and r5s2.
- Fitting: 45 s for the first member (searches and Jacobians), then
  113-118 s per member.
- Member 0 has the same W as r5.  Its beta differs by 8e-9 relative,
  and its energies by 6e-8 meV/atom at the checks.  The solve is
  ill-conditioned (rcond 7.5e-17) and amplifies the rounding change of
  ADR-0030.

RMSE (E meV/atom, F eV/A, S meV/A^3):

| Model | val E / F / S | test E / F / S |
|---|---|---|
| r5 (seed 42) | 2.67 / 0.132 / 1.57 | 1.19 / 0.0787 / 1.48 |
| r5s1 (seed 1) | 3.19 / 0.129 / 1.52 | 1.18 / 0.0868 / 1.38 |
| r5s2 (seed 2) | 3.82 / 0.125 / 1.52 | 1.10 / 0.0874 / 1.47 |
| ensemble mean | 2.49 / 0.093 / 1.34 | 1.11 / 0.0657 / 1.32 |

The force error falls by 17-30 % against every single member.

The 18 DFT-checked anneal minima (errors in meV/atom against DFT,
referenced as the training data):

- Mean absolute error: 2.31 for the ensemble mean.  Per member: 3.78
  (seed 42), 5.19 (seed 1), 2.29, 1.92 and 2.24.
  - These minima were found by single models.  Six of them were found
    by seeds 42 and 1, which are members, so their errors weigh on the
    mean.  Seeds 2-4 did not find them and are low here for that
    reason.
- Checks beyond 10 meV/atom: 1 for the mean (-12.6 at
  r5s1_test00026).  For the members: 3 (seed 42: -11.5, -23.1, -10.7)
  and 1 (seed 1: -44.1), the rest 0.
- sigma against the errors:
  - Spearman correlation of sigma_E with the spread of the members'
    errors: 0.89; with the absolute error of the mean: 0.57.
  - Over the 656 training structures, sigma_E has median 0.46,
    95th percentile 3.2 and 99th percentile 6.6 meV/atom; sigma_F_max
    has 0.19, 0.62 and 1.20 eV/A.
  - At the checks, sigma_E is above the training 95th percentile for
    6 of 18, and sigma_F_max for 15 of 18.  The two largest errors of
    the mean have sigma_E 17.9 meV/atom (r5s1_test00026) and 10.3
    meV/atom (r5_test00046, mean error -6.7), both above the training
    99th percentile.

The ensemble removes most of the seed noise found in ADR-0027, and the
spread flags the minima where members disagree.  It does not correct a
minimum that its own members helped find: there, the mean carries a
part of their error, and sigma marks it as uncertain.  Whether the
ensemble passes ADR-0027 A1 needs its own anneal runs (decision 10).

**Cost** (one force call, 1 numba thread, GeSn cells built from the
63-atom test00038 check):

| N | single r5 | ensemble of 5 | `uncertainty()` |
|---|---|---|---|
| 504 | 60 ms | 136 ms | 5 ms |
| 4032 | 573 ms | 1290 ms | 39 ms |

At n_hidden 1600 the ELM heads, not the descriptor, dominate, so five
members cost 2.25 times one.  The +40 % estimated in the starting point
was for 400 hidden units.

Full suite: 892 passed, 1 skipped (14.4 min).

Follow-up: ADR-0037 (0.3.0a1) drives active learning with this
spread (`akitsushima active`, trust window on sigma_F,max).
