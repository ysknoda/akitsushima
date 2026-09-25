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

9. **Linear models** (`mlip.model: linear`; added 2026-09-25 after the user
   pointed out that the tabular format suits them). With E/N = w.d + b, every
   F and S row is N w.X exactly -- no chain rule is needed, E, F and S rows
   are solved together, and because d is a per-atom average,
   E = sum_i w.G_i: the linear tabular model *is* an atom-centred linear
   model (the SNAP/MTP family) and keeps locality.
   - Methods (`method:`): `ols`, `ridge`, `lasso`, `elastic_net`, `pls`,
     `bayes_ridge`, `ard`, `smooth_ridge` (generalised Tikhonov: a
     first-difference penalty between adjacent bins of the same descriptor
     block, blocks read from the CSV column names, e.g. `RDF_Si-O_b0000`).
   - The intercept and the centring act on E rows only: F and S rows are
     derivatives, in which a constant vanishes. Library implementations that
     centre every row (e.g. scikit-learn's) are silently wrong here, so the
     methods are written in the package with numpy/scipy (no new
     dependency): direct solve for OLS/ridge/smooth_ridge, coordinate descent
     for LASSO/Elastic Net, NIPALS for PLS, evidence maximisation for Bayesian
     ridge/ARD.
   - Same weighting rule as decision 2. Unweighted OLS is the special case
     w_energy = w_forces = w_stress = 1 with a unit weight column.
   - `alpha`, `l1_ratio`, `n_components` may be given, or chosen from
     `alpha_search` / a component range on the structure-grouped validation
     split; the Bayesian methods set their own regularisation and also return
     a predictive variance.
   - With in-package ACSF the linear model is a `relax`/`md` engine like
     ELM/NN (decision 6).

10. **Defaults and acceptance criteria after the verification** (decided by
    the user, 2026-09-25, from the follow-up in the appendix):
    - `ELMModel` / `QuantizedELMModel`: `direct_link=True` (RVFL) by
      default; `reg` then acts on the hidden weights only, and
      `reg_search` picks it on the validation split.
    - `NNModel`: `activation='softplus'` and `direct_link=True` by
      default; `deterministic=True` (torch deterministic algorithms, one
      CPU thread during fit) is available for bit-reproducible training,
      off by default for speed.
    - Decision 7's learning criterion is judged on SW-Si data whose test
      cells lie inside the training strain range (data set v2).  Test
      cells with strains outside it (data set v1) measure transferability
      and move to the ADR-0005 comparison, item (e).
    - LightGBM's T3a criterion is judged with noise-optimal weights
      (1/sigma^2 per row), the rule used for SW-Si; the template's
      category weights are a user choice, not part of the test.

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
- Group LASSO, principal-component regression and Huber regression: FW-3.

## Consequences

- Every ELM/QELM/NN fit changes (chain rule and weighting); results before
  0.2.0a3 are reproducible only with `row_independent: true` and are not
  comparable on E/F/S RMSE.
- Existing CSV files keep working without change; only a malformed row order
  is now an error.
- LightGBM becomes a better regressor but is explicitly not a force field.
- The package gains its first MLIP engine for `relax`/`md`.

## Appendix: implementation notes and verification (2026-09-25)

### Implementation notes (deviations and additions)

- **ELM direct solver.** The chain-rule system mixes rows of very different
  scale, and LSQR stopped at its iteration limit (flag 7) short of the
  minimum. `ELMModel(solver='auto')` now solves the normal equations
  (A^T A + reg I) beta = A^T t by Cholesky when there are at most 6000
  output weights -- the same minimiser as LSQR with damp = sqrt(reg) -- and
  falls back to LSQR above that. On the SW-Si data both gave identical
  predictions; the direct solve took 1.5 s against 238 s.
- **ELM output bias.** In chain-rule mode the ELM has an explicit output
  bias beta0 acting on E rows only (per-atom energies of about -4.3 eV are
  otherwise left to the hidden units).
- **G5 added to ACSF.** With r_c = 3.771 A (SW's own range a*sigma) two
  neighbours of a Si atom are 3.84 A apart, so G4's fc(r_jk) removes almost
  every bonded triplet and the descriptor loses the bond angles SW's
  three-body term depends on (force RMSE 0.58-1.2 eV/A with G4 only). G5
  (Behler 2011, eq. 9) drops the r_jk terms; with it the linear model
  reached 0.029 eV/A on noise-free labels.
- **Bayesian ridge / ARD.** The first implementation judged convergence on
  the coefficients and stopped while its initial prior still held them
  near 0 (SW-Si forces at 270 sigma). It now iterates MacKay's fixed point
  on the hyperparameters (Bayesian ridge in the eigenbasis of A^T A) and
  computes the residual directly rather than by the cancellation-prone
  expanded form.
- **LightGBM `label_feature`** (default true) can be switched off; see the
  T3a result below.
- **Torch at test collection.** Importing torch while pytest collects, before
  numba has started its OpenMP thread pool, aborts numba's first parallel
  kernel. The new test module detects torch with `importlib.util.find_spec`
  and imports it only inside the NN tests.
- **relax / md cell size.** ACSF handles cells thinner than 2 r_c itself
  (all periodic images), but relax / md still build their own neighbour
  list, which needs every cell width > 2 (r_c + skin).
- **Descriptor convention of `atomic_descriptor_ver2.0.0.py`.** From its
  own finite-difference tests and normalisation code: E rows are
  E_raw / N, F rows -dE_raw/dr / N and S rows -dE_raw/deps / (N V), i.e.
  X_F = -dd/dr and X_S = -(1/V) dd/deps, the convention the chain rule
  assumes, so its CSVs train correctly as they are. (The sign of its S
  *targets*, read from OUTCAR, was not checked here.)
- The T3 synthetic sets of ADR-0003 are row-independent by construction
  (every row is f(x) of its own descriptor); ELM/QELM/NN reproduce T3 only
  with `row_independent: true`. They are not chain-rule data.

### Tests

`tests/test_mlip_chain_rule.py` (40 tests): structure indexing and CSV
metadata; F/S rows equal N x the finite-difference derivative of the E-row
prediction for ELM (tanh, silu, rff, gelu), QELM and NN (float64 copy);
every ELM activation derivative against finite differences; direct = LSQR
minimiser; the weighting rule; the M1 regression, which runs the old
formula as a control (E: new 3.3 sigma, old 73 sigma; 136 sigma at 512
hidden units); every linear method recovering a linear potential; the
intercept not leaking into F rows; Bayesian convergence with noise-optimal
weights; ACSF derivatives against finite differences for positions and all
nine cell components on a distorted three-species cell thinner than 2 r_c,
sum_j X_F = 0, rotation/translation/permutation invariance, numba = Python;
engine forces and stress against finite differences of the engine's own
energy (ELM, linear, NN); refusal of non-force-field models; and the CLI
chain `mlip` (ACSF) -> `relax` (variable cell) -> `md` (NVE). Full suite:
445 passed.

### SW-Si verification (decision 7; `example/mlip/adr0004/sw_si_verify.py`)

Data: 64-atom diamond cells, NVT MD with SW (Stillinger & Weber, PRB 31,
5262 (1985)) at 300/600/900/1200/1500 K in the unstrained cell and in
randomly strained cells (up to +/-3%); 450 training structures (3 cells)
and 200 test structures (2 cells) from independent trajectories. Gaussian
label noise sigma: E 2 meV/atom, F 0.05 eV/A, S 0.002 eV/A^3. Row weights
1/sigma^2 (w_cat chosen accordingly). ACSF, r_c = 3.771 A: G2 at 20 centres
x 2 widths, G5 5 eta x 5 zeta x 2 lambda (90 features). Pass: test RMSE /
sigma <= 1.5 for E, F and S; relaxed a within 0.1% of SW's 5.430950 A; NVE
(600 K, 3000 fs) energy fluctuation and drift < 1 meV/atom. The same NVE
run on SW itself: 0.016 meV/atom fluctuation, 0.001 meV/atom drift.

| model | E | F | S | fit | a [A] (err) | NVE std / drift [meV/atom] | verdict |
|---|---|---|---|---|---|---|---|
| linear (ridge) | 1.02 | 1.02 | 0.99 | 0.5 s | 5.42981 (-0.021%) | 0.016 / 0.001 | PASS |
| bayes_ridge | 1.02 | 1.01 | 0.98 | 0.3 s | 5.42975 (-0.022%) | 0.016 / 0.001 | PASS |
| ard | 1.03 | 1.03 | 1.01 | 7.5 s | 5.42986 (-0.020%) | 0.016 / 0.001 | PASS |
| NN [64, 64] | 1.29 | 1.20 | 1.55 | 36 s | 5.42855 (-0.044%) | 0.016 / 0.002 | S just above 1.5 |
| ELM (400) | 10.6 | 7.3 | 20.4 | 1.4 s | 5.41541 (-0.29%) | 0.015 / 0.001 | FAIL |

Every engine conserves energy as well as SW itself, because forces and
stress are exact derivatives of each model's energy. Accuracy separates the
models:

- The linear models (E = sum_i w . G_i, an atom-centred linear potential)
  learn SW to the noise level.
- ELM fits the training set to the noise (1.01 / 1.04 / 1.05) but not the
  test set; no regularisation strength (1e-2 to 1e6) or width (100 to 2000
  hidden units) fixes it. The test set uses a strain the training set
  does not contain, so the random-feature f(d) on the structure descriptor
  extrapolates in d. This is a property of the model form, not a defect
  in the implementation (the derivatives are verified); a remedy -- e.g.
  a direct linear term in the ELM output (random-vector functional link)
  -- is a design change for the user to decide.
- The NN generalises better than the ELM but is limited by the same
  extrapolation in S.

### LightGBM on T3a (decision 7; `example/mlip/adr0004/check_lgbm_t3a.py`)

30 Optuna trials, new search space, w_unit^2 weights, per-label
standardisation.

| run | E | F | S | wall |
|---|---|---|---|---|
| label feature on (default) | 1.59 | 1.41 | 1.43 | 21 min |
| label feature off (template weights) | 1.42 | 1.35 | 1.36 | 3.7 h |
| label feature on, noise-optimal weights (1/sigma^2 per row: equal for T3a) | 1.25 | 1.27 | 1.29 | 43 min |

(Wall times of the later runs overlap other jobs on the same machine.)
The ablation's best (old weights, no label feature, fixed min_gain 0 and
lambda in [1e-8, 10]) was 1.45 / 1.40 / 1.42 in 85 min.  With the label
feature, E misses the 1.5 criterion narrowly while F and S pass.  T3a gives
every label the same function, so a label split only separates the E rows
(1/19 of the data) from the F and S rows they could learn from; the run
without it tests that.  Without the label feature single trials early-stop
far later (trial 0: 1066 s against 87 s), so the raised iteration cap makes
`optuna_timeout` the practical bound on search time.

### Follow-up (2026-09-25): RVFL, NN, activation functions, LightGBM weights

Changes, all opt-in (defaults unchanged, pending the user's decision):

- `ELMModel(direct_link=True)`: random-vector functional link (Pao, Park
  & Sobajic, Neurocomputing 6, 163 (1994)), E/N = beta . h(d) +
  gamma . d_n + beta0; the F/S rows stay linear in (beta, gamma, beta0).
  With the direct solver, `reg` penalises the hidden weights beta only
  (gamma and beta0 get a 1e-12 relative jitter), so a large reg shrinks
  the model to the linear model it contains, not to zero.
- `ELMModel(reg_search=[...])`: reg chosen on the structure-grouped
  validation split; A^T A is built once and re-solved per candidate.
  Needed because the scale of reg follows that of the row weights (useful
  values were 1e4-1e8 with the SW-Si weights of ~1e4 per row).
- `NNModel(direct_link=True)`: MLP(x) + gamma . x_n.
- Tests: RVFL rows = finite-difference derivatives (tanh, rff, QELM);
  RVFL with reg 1e12 equals the linear model; reg_search picks the
  lowest validation loss; NN direct-link save/load.  46 tests in
  `tests/test_mlip_chain_rule.py`.

**Two SW-Si data sets.**  v1 (above) trains on 3 cells and tests on a cell
with an unseen strain, so its test mixes learning SW with extrapolating in
strain.  v2 trains on 15 cells (unstrained + 14 random strains up to
+/-3%) and tests on 5 new cells from the same strain distribution
(interpolation); otherwise identical (`sw_si_verify.py --dataset v2`).

v2 (test RMSE / sigma, lattice error, NVE all < 0.02 meV/atom):

| model | E | F | S | a err | verdict |
|---|---|---|---|---|---|
| linear / bayes_ridge / ard | 1.02 | 1.01-1.02 | 0.97-0.98 | -0.02% | PASS |
| ELM tanh (plain) | 1.03 | 1.08 | 1.00 | +0.03% | PASS |
| RVFL tanh | 1.02 | 1.02 | 0.97 | -0.02% | PASS |
| NN silu, 3 seeds | 1.16-1.33 | 1.15-1.24 | 1.36-1.44 | -0.36 to +0.14% | 2 of 3 fail the lattice check |
| NN silu + direct link, 3 seeds | 1.05-1.15 | 1.10-1.13 | 1.03-1.17 | -0.10 to +0.03% | PASS |

So the v1 failures of ELM and NN were extrapolation, not a defect: with
training data covering the strain range, the plain ELM passes.  The NN
results are not reproducible run to run even at a fixed seed (test E at
seed 42: 50.6 then 59.4 sigma on v1), presumably non-deterministic CPU
reductions in torch; its validation loss, drawn from the training cells,
cannot see the extrapolation failure, so early stopping cannot guard
against it.  The same holds for `reg_search` on v1: it selected reg = 100
and RVFL stayed at 2.6 / 2.1 / 4.9 sigma, while a large reg (1e6-1e8)
gives the linear model's 1.02.

**Activation functions** (`activation_scan.py`; ELM 400 units with
reg_search; NN [64, 64], seeds 42/1/2; relu excluded -- forces are its
first derivative and would jump):

- v2 (interpolation): all ELM activations 1.02-1.09 sigma plain (rff
  1.07 / 1.16 / 1.07) and 1.02 / 1.02 / 0.97 with the direct link; all NN
  activations F 1.09-1.15, S 1.0-1.5, E 1.03-1.69 with the spread over
  seeds larger than that over activations.  The activation hardly matters.
- v1 (extrapolation): it matters a great deal.  NN ranges over seeds:
  softplus + direct link 1.07-1.47 / 1.14-1.22 / 1.08-1.27 (the only
  configuration near the criterion), softplus 1.14-2.11 / 1.24-1.38 /
  1.25-1.72, tanh 1.17-2.30 / 1.22-1.37 / 1.17-3.95; silu 2.7-12.8 (E),
  gelu 19-89, mish 18-101, and with the direct link silu 17-132.  Two NN
  runs relaxed to a lattice constant 28-30% off.  ELM: 1.5-35 sigma for
  every activation; best mish + direct link (1.51 / 2.38 / 3.82).  A
  plausible reading (not established): softplus and tanh have monotone,
  bounded first derivatives, while silu, gelu and mish have a
  non-monotone dip near zero, so gradients far from the data -- which
  set the extrapolated forces and stresses -- behave less predictably.

**LightGBM.**  With the template's weights T3a puts 160x more weight on an
E row than on an F row, although all rows have the same noise.  With
noise-optimal (equal) weights the fit passes: 1.25 / 1.27 / 1.29 sigma,
better than the bare library unweighted (1.30 / 1.32 / 1.33).  The E miss
was the weighting of the test, not the model.

**Verification with the decision-10 defaults** (`sw_si_verify.py`,
models `elm_def` = ELM 400 units with the direct link and reg_search,
`nn_def` = NN [64, 64], softplus, direct link, deterministic):

| model | data | E | F | S | a err | verdict |
|---|---|---|---|---|---|---|
| ELM default | v2 | 1.02 | 1.02 | 0.97 | -0.02% | PASS |
| NN default, patience 40 (3 seeds) | v2 | 1.09-1.86 | 1.09-1.10 | 1.00-1.11 | -0.08 to -0.04% | 2 of 3 PASS (E of seed 42) |
| NN default, patience 150, <= 1500 epochs | v2 | 1.04-1.11 | 1.06-1.07 | 1.00-1.01 | | 3 of 3 PASS |
| ELM default | v1 | 2.58 | 2.14 | 4.91 | -0.07% | transferability (ADR-0005) |
| NN default, patience 40 | v1 | 1.12-1.76 | 1.15-1.27 | 1.10-1.50 | -0.07 to +0.07% | transferability |
| NN default, patience 150, <= 1500 epochs | v1 | 1.04-1.15 | 1.10-1.12 | 1.01-1.08 | | transferability |

The deterministic runs repeat exactly (patience-40 numbers identical in two
separate runs).  E is the slowest quantity for the NN to converge (one E
row per 199 rows): an early-stopping patience of 150 epochs brings every
seed to the noise level on both data sets.  The NN's defaults
(`early_stopping_patience=30`, `epochs=500`) are unchanged here, pending
the user's decision; the recommendation is patience 150 with up to 1500
epochs.
