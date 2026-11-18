---
status: proposed (2026-11-18)
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

## Result

(to be appended)
