---
status: accepted
---

# Remove LightGBM; document the direct linear term

Decided with the user 2026-09-28 (Q28-Q32).

## LightGBM removed completely (Q28)

The MLIP comparisons (ADR-0004, ADR-0005) established that
`LightGBMModel` cannot be a force field and is the weakest regressor in
the package:

- Trees are piecewise constant, so -dE/dr is zero almost everywhere: no
  conservative force (ADR-0004 decision 3); it was refused as an engine.
- Least accurate on every data set: Si test 4.5 / 1.3 / 2.8 sigma
  (E / F / S) against 1.03 / 1.01 / 0.97 for linear / ELM; 3.7-75 sigma on
  transfer sets.
- ~40 min per fit (Optuna search, 30 min cap) against seconds.
- lightgbm and torch cannot share a process (OpenMP runtime conflict).
- Its one relative strength, a row-independent regressor on the user's
  RDF / ADF / ODF descriptor tables (4.4 / 2.8 / 2.3 sigma), still stayed
  far from the noise floor; descriptor screening can use LightGBM
  directly, outside this package.

Removed: `fitting/models/lightgbm_model.py`, `utils/optional_lightgbm.py`,
the `lgbm` model of `akitsushima mlip` (and its template block and
settings print), the lazy-import entries and public name `LightGBMModel`,
the `lightgbm` extra (lightgbm, optuna) of pyproject.toml, its tests and
README entries.  Kept: historical notes in ADR-0003 / 0004 / 0005 and in
bug-fix comments.  `mlip.model: lgbm` and loading an old LightGBM model
file now fail with an error naming this ADR.

## README: the direct linear term (Q30)

The README only mentioned `direct_link` in the release banner.  A
paragraph now gives the forms (ELM / QELM f = beta . g(Wx + b) + gamma . x
+ beta_0, the RVFL network; NN f = MLP(x) + w . x), why (the linear part is
fitted exactly, extrapolation falls back to it), how each model
regularises it -- checked in the code: ELM `_penalty` puts reg on beta only
and a 1e-12 relative jitter on gamma and beta_0 (same in the atom-centred
solver); the NN's Adam weight_decay covers all parameters and the linear
weights start at zero -- and `direct_link: false`.  No RVFL citation is
given: the reference that came to mind (Pao, Park & Sobajic, 1994) was not
verified.

## Deferred (Q29, Q32)

The atom-centred linear / quadratic model and a sparse Gaussian-process
(GAP-type) model are recorded in ADR-0010 (items 5 and 6) for later.
