---
status: accepted
---

# MLIP: chain-rule training on tabular E/F/S data, one weighting rule, and MLIP as a force-field engine

ADR-0003 left finding **M1** open: the ELM/QELM uncertainty weighting squares
the unit weight a second time and applies 1/sigma^2 to the residual, so with
realistic label scales the S rows outweigh the E rows by ~1e17, the LSQR does
not converge, and E is fitted to ~200x its noise (T3b). Looking at why led to a
deeper defect, a weighting rule that differs between models, a LightGBM search
space that caps its accuracy, and the absence of any way to run `relax` or `md`
on a trained MLIP. This ADR records the decisions that fix those. The
atom-centred (Behler-Parrinello) model that was discussed at the same time is
recorded separately in ADR-0005, because it is implemented second.

## Background: what the tabular data format means

The `mlip` CSV (`load_mlip_csv`; produced by the user's
`atomic_descriptor_ver2.0.0.py`, see `docs/future-work.md` FW-1) holds, per
structure with N atoms, one E row, 3N F rows and 6 S rows, in that order:

- E row: target E/N [eV/atom]; descriptor d = the per-atom-averaged structure
  descriptor.
- F rows: target F_{i,a} [eV/A]; descriptor X_{ia} = -(d d)/(d r_{ia}), the
  exact gradient of the E-row descriptor.
- S rows: target sigma_{ab} [eV/A^3], Voigt, compressive-positive (ADR-0001);
  descriptor X_{ab} = the matching cell derivative of d.

The derivatives are in the descriptor, so E, F and S can be learned from one
table by any tabular model. That is exact for a model **linear** in d: with
E/N = w.d + b, F_{ia} = N w.X_{ia} (and likewise for S). For a **nonlinear**
model E/N = f(d), the correct force is

    F_{ia} = N grad f(d) . X_{ia}                                   (1)

which depends on the structure's own d, not only on the row's X_{ia}.

**The defect.** Every model in the package (ELM, QELM, NN, LightGBM) applies
the same function to every row and uses the label only to pick a weight: an F
row is predicted as f(X_{ia}), not as (1). The E and F predictions of one
trained model are therefore not derivatives of one energy, and the model is
not a force field. This, together with the weighting formula below, is the
root of M1. (`ELMModel.compute()` meanwhile evaluates an atom-centred
E = sum_i g(X_i), a third form that matches neither the training nor the
data.)

## Decisions

1. **ELM, QELM and NN train with the chain rule (1) by default.**
   - F and S rows are tied to their structure. The structure id is recovered
     from the row order (each E row opens a structure; the order
     `structure_grouped_split` already relies on); an explicit `struct_id`
     column, if present, takes precedence. A structure whose row count is not
     1 + 3N + 6 (or 1 + 3N when S rows are absent) is rejected with a message.
   - ELM/QELM: with hidden layer h(d) = g(W d + b) and E/N = beta . h(d),
     an F row is N beta . [g'(W d + b) (*) (W X_{ia})] and an S row the same
     with X_{ab}. Every row is **linear in beta**, so E, F and S rows are
     stacked into one weighted least-squares system and solved in closed form,
     as today. The bias acts on E rows only.
   - NN: F and S rows are the Jacobian-vector product grad f(d) . X, computed
     with `torch.func.jvp` (or autograd), and enter the same loss.
   - The factor N and the sign of each row type are fixed by a finite-
     difference test on a synthetic structure (decision 7), not by reading
     the formulas.
   - The old row-independent behaviour stays available as
     `row_independent: true` (to reproduce earlier results and for the ADR-0005
     comparison). It is no longer the default: **a behaviour change**, listed
     in the 0.2.0a3 notes.

2. **One weighting rule for every model.** The squared residual of each row is
   multiplied by

       w = w_cat * w_unit^2 * w_user

   where w_cat is the category weight (`w_energy`, `w_forces`, `w_stress`),
   w_unit the CSV weight column (N for E, r0 for F, eps0*V for S, which makes
   every term eV^2), and w_user the optional block-wise user weight. This is
   the convention the classical `fit` already uses (w_cat x w_unit^2). For the
   least-squares models the row multiplier is its square root,
   sqrt(w_cat) * w_unit * sqrt(w_user).
   - The ELM/QELM uncertainty weighting (`_compute_sigma2`, 1/sigma^2 on the
     residual) is **removed**; this closes M1 together with decision 1.
   - NN and LightGBM change from w_cat * w_unit to w_cat * w_unit^2.
   - Iterative noise-based reweighting (estimating each category's noise from
     the residuals) is deferred; it can be added on top of this rule later.

3. **LightGBM stays a row-independent regressor, with an improved search.**
   Trees have zero gradient almost everywhere, so (1) cannot be used and a
   tree model gives no conservative force; LightGBM is **not** offered as a
   `relax`/`md` engine. Within that role:
   - The label (E/F/S) is passed as a categorical feature, so the trees can
     learn separate functions for energies, forces and stresses.
   - Targets are standardised per label before training (and restored on
     prediction), so split gains no longer depend on the units of the targets
     or on the weights.
   - Optuna search space, set from the ablation below:

     | parameter | old | new |
     |---|---|---|
     | learning_rate | 0.01-0.1 log | unchanged |
     | num_leaves | 15-255 | unchanged |
     | max_depth | 3-20 | unchanged |
     | min_data_in_leaf | 2-30 | unchanged |
     | bagging_fraction, feature_fraction | 0.5-1 | unchanged |
     | min_gain_to_split | 0.1-10 log | **fixed 0** |
     | lambda_l1, lambda_l2 | 1-100 log | **1e-8-10 log** |
     | max_bin | 200-500 | unchanged |
     | num_iterations cap | 10000 | raised; early stopping decides |

   - `optuna_timeout` (seconds) bounds the search, and the study is stored in
     SQLite so an interrupted search resumes (the ablation's best run took
     85 min for 30 trials).

4. **In-package ACSF descriptors.** Behler-Parrinello symmetry functions,
   radial G2 and angular G4 (J. Behler, J. Chem. Phys. 134, 074106 (2011)),
   with analytic derivatives with respect to atomic positions and the cell,
   written in numba. For the tabular models of this ADR the structure
   descriptor d is the per-atom average of the atomic ACSF vectors, and
   X_{ia}, X_{ab} its derivatives, so a trained model can be evaluated on any
   structure. The same per-atom vectors and Jacobians serve ADR-0005. Species
   axes are ordered by atomic number. The descriptor interface (per-atom
   vectors, Jacobians, serialisable config) is kept pluggable so that FW-1
   (the user's own descriptors) can be added later.

5. **Stress from the start.** S rows are trained and the engine returns a
   stress tensor, in the ADR-0001 convention (compressive-positive), so
   variable-cell `relax` and NPT `md` work.

6. **MLIP as a `relax`/`md` engine.** YAML:

       potential:
         type: MLIP
         model: path/to/model.npz

   The model file stores the model, the descriptor type and its full config,
   and the element list (atomic-number order). Only the path is given in the
   YAML, so the descriptor used at inference cannot differ from the one used
   in training. The engine plugs into the existing force-function interface
   (`make_force_fn`), returning energy, forces and stress. Available for
   ELM/QELM/NN trained with in-package descriptors; a model trained with
   `row_independent: true` or on external descriptors is refused as an engine
   with a message saying why.

7. **Verification**, to the ADR-0002 standard:
   - Finite differences: forces and stress of the engine, and the F/S rows of
     decision 1 against the E-row prediction, on small synthetic structures;
     plus sum_i F_i = 0 as a first check. ACSF Jacobians against finite
     differences, and invariance under translation, rotation and permutation
     of like atoms.
   - Weighting: a test that fails under the old ELM formula (E fitted far
     above its noise when S rows are present) and passes under decision 2.
   - Learning on labels from a known potential: Si Stillinger-Weber (PRB 31,
     5262 (1985)); the trained model is judged on E/F/S RMSE against the
     noise sigma added to the labels (ratio ~1 is the target, as in T3).
   - Engine: NVE energy conservation and the relaxed Si lattice constant
     against SW's own value.
   - LightGBM: the ablation data set (T3a) must reach the raw-library level
     (<= 1.5 sigma on E/F/S) with the new search space.
   - Real DFT data is out of scope here; the user will ask for it separately.

8. **Release.** Bump to 0.2.0a3 after the implementation; the notes list the
   default change of decision 1, the weighting change of decision 2, and the
   LightGBM search-space change.

## Evidence: LightGBM ablation (2026-09-25)

Data: T3a shared data set (`example/mlip/shared_{train,test}.csv`, noise
sigma = 0.05); same rows, weights and structure-grouped 80/20 split
throughout; 30 Optuna trials; test RMSE / sigma for E / F / S.

| case | E | F | S | note |
|---|---|---|---|---|
| A raw lightgbm, same weights | 1.60 | 1.49 | 1.50 | 9511 iterations (cap 10000) |
| A0 raw lightgbm, unweighted | 1.30 | 1.32 | 1.33 | |
| B package wrapper pinned to A | = A | = A | = A | the wrapper is not the problem |
| C package, shipped ranges | 2.03 | 1.84 | 1.77 | min_gain 0.22, lambda_l2 11.5 |
| D C with min_gain_to_split = 0 | 1.46 | 1.45 | 1.47 | main cause |
| E C with lambda in [1e-8, 10] | 1.78 | 1.63 | 1.61 | secondary cause |
| F D + E | 1.45 | 1.40 | 1.42 | 5076 s |

The shipped ranges are the ones from the user's own earlier script. There the
lower bound 0.1 for `min_gain_to_split` suited that data; a split gain is a
loss decrease in the units of the weighted squared target, so the same bound
stops splitting on eV/A-scaled rows with eV-unit weights. Per-label
standardisation (decision 3) removes that dependence. The remaining gap to
the unweighted fit (A0) is the weighting, which decision 2 changes.

## Out of scope / later

- Atom-centred model E = sum_i eps(X_i): ADR-0005.
- The comparison of the tabular and atom-centred approaches: ADR-0005.
- External descriptors / porting `atomic_descriptor_ver2.0.0.py`: FW-1.
- An additive per-atom LightGBM energy model (custom objective, gradient of
  each atom = its structure's residual), usable for energy-only
  Delta-learning: FW-2.
- A linear (ridge) model on the tabular data: exact without the chain rule,
  but only useful as an engine once FW-1 provides the descriptor at
  inference; to be considered with FW-1.

## Consequences

- Every ELM/QELM/NN fit changes (chain rule and weighting); results before
  0.2.0a3 are reproducible only with `row_independent: true` and are not
  comparable on E/F/S RMSE.
- Existing CSV files keep working without change; only a malformed row order
  is now an error.
- LightGBM becomes a better regressor but is explicitly not a force field.
- The package gains its first MLIP engine for `relax`/`md`.
