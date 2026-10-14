---
status: accepted (implemented and evaluated 2026-09-28; PCA whitening made the default, gamma_smooth removed)
---

# ELM / QELM: random-matrix scale and bias, PCA whitening, correlated weights, QELM levels

Decided with the user 2026-09-28 (Q33-Q47).  Every item is an option;
defaults stay as they are until the evaluation below shows a gain.

## Starting point (checked in the code)

- **Random matrix.**  `ELMModel._init_hidden` already draws W and b from
  N(0, s^2) with s = sqrt(2 / (D + n_hidden)) (Glorot / Xavier variance);
  `w_init_scale` overrides s.
- **Standardisation.**  Per-feature z-score (training mean and standard
  deviation, stored in the model); tabular: from the E-row descriptors,
  atom-centred: per species from the atoms' G.
- **NN.**  torch default initialisation; the direct-link weights start at
  zero, the atom-centred output bias at the per-species reference energy.
- **QELM.**  `n_bits = 3`: each column of W is rounded to
  {0, +-1, +-2, +-4, +-8} * s_j, s_j = max|column| / 2^n_bits (9 levels);
  `p_zero = 0.5` zeroes the smaller half of each column.  Ternary is
  `n_bits = 0`.  `n_bits` is the largest power-of-two exponent, not a bit
  count (9 levels need 4 bits).  QELM matched ELM in accuracy and time
  (ADR-0005); "multiplication-free" shows no speed gain in numpy.

**Why Xavier / He do not transfer.**  They keep signal and gradient
magnitudes through layers trained by back-propagation.  An ELM trains no
hidden layer; what matters is how much of the activation's non-linear range
z = W.x + b covers and how spread the units' centres are.  With the
default s, Si (D = 90, n_hidden = 400) gives std(z) ~ 0.6 -- tanh nearly
linear -- and std(b) ~ 0.06, so all units are centred at almost the same
point.  That is what the options below address.

## Decisions

1. **Scale search** (Q33a, Q41): `w_init_scale_search: true` picks the
   factor on the default s from {0.5, 1, 2, 4, 8} on the validation split,
   jointly with reg (the `reg_search` machinery).
2. **Data-centred bias** (Q33a, Q41): `bias_init: normal | data`
   (default normal); `data` picks a random training point x_k per unit and
   sets b = -W.x_k, so the unit is centred on the data.
3. **No named Xavier / He / LeCun options** (Q33): no ELM rationale (He is
   a ReLU rule for back-propagation).
4. **PCA whitening** (Q34, Q39): `standardize: zscore | pca` (default
   zscore), `pca_eps`.  mu, the rotation U and the variances lambda come
   from the training data and are stored in the model; every prediction
   uses x~ = (Lambda + eps)^(-1/2) U^T (x_z), x_z the z-scored input (PCA of
   the correlation matrix).  No cut by explained variance (it could drop
   small components that matter for forces).  **Changed after the first
   test (see Results): directions are no longer dropped at 1e-6 *
   max(lambda) -- all are kept, eps alone bounds the gain; `pca_drop`
   (default 0) restores a relative cut.**  F and S rows (descriptor
   derivatives) get the same matrix without the mean, so chain-rule forces
   and stress stay energy gradients.  Atom-centred: one transform per
   species.  Caveats: a direction barely varying in training is amplified
   by 1/sqrt(lambda + eps); dropped directions are ignored for new data too.
   Min-max scaling to [-1, 1] is not added (sensitive to outliers).
5. **Correlated random weights** (Q45): `w_correlation: none | radial`
   (default none), `w_corr_length` (A), searchable over
   {0.1, 0.2, 0.4, 0.8} A.  Each unit's weight column is drawn from a
   block-diagonal covariance: inside one (species, eta) group of G2
   channels K_kl = exp(-(r_s,k - r_s,l)^2 / (2 l^2)); G5 channels and
   everything across blocks stay independent; the per-element variance is
   unchanged.  Rationale: adjacent G2 channels (r_s 0.1 A apart) are
   nearly identical; weights of opposite sign on them make a unit see
   their difference, a low-variance, noise-sensitive direction.  Smooth
   weights along r_s make each unit a smooth radial filter -- a prior that
   lowers the effective number of degrees of freedom, so it works against
   overfitting; the risk is the opposite, lost resolution when l exceeds
   the spacing of features to be told apart (Si first / second neighbour
   2.35 / 3.84 A), hence l is chosen on validation.  It is the opposite of
   PCA whitening (which decorrelates and amplifies exactly those
   difference directions); the two are alternatives.  Needs the channel
   layout, so in-package ACSF only; an external descriptor table is an
   error.  QELM quantises the correlated draw.  No MLIP literature on this
   was checked; the evaluation decides.
6. **Smoothness penalty on gamma** (Q46): `gamma_smooth` (default 0) added
   the squared differences of the direct-link weights of adjacent r_s
   channels (same groups as item 5) to the normal equations -- the idea of
   the tabular `smooth_ridge`.  **Removed after the evaluation** (no gain;
   worse where selected, see Decisions after the evaluation).
7. **QELM levels** (Q42, Q44): `levels: pow2 | ternary | uniform`
   (default pow2).  ternary: {-1, 0, 1} * s_j, zero fraction from p_zero;
   uniform: {-k, ..., k} * s_j (integer arithmetic, e.g. int8), with
   `n_levels`.  `n_bits` is renamed `max_exponent`; `n_bits` stays as a
   deprecated alias.  A sparse random projection with discrete values is
   expected to behave like a Gaussian one when the variance matches
   (recalled, literature not checked), so the level set should matter for
   hardware efficiency more than accuracy.
8. **QELM sparse / integer path** (Q43): compute Wx with a sparse W (large
   p_zero) and measure time and memory; if no gain appears, deprecating
   QELM is put to the user.
9. **NN unchanged** (Q35, Q47): no new initialisation options; the
   smoothness penalty on the first NN layer waits for the ELM result.
10. **Recorded only** (Q36): ensemble ELM and OS-ELM in ADR-0010 (items 7,
    8); multilayer ELM only if single-layer is shown insufficient; kernel
    ELM overlaps ADR-0010 item 6.

## Evaluation (Q37, Q40, Q47)

The SW data sit at the noise floor, so the options are compared where they
can differ:

- **Data**: SW-Si and SW-GeSn with the label noise sigma reduced to 1/10
  (existing structures, labels regenerated), and Tersoff-Si (the package
  template's Tersoff (1988) parameters), newly sampled.  EAM is not used
  (no YAML route for its parameters).  Real DFT data when available.
- **Capacity**: n_hidden in {25, 50, 100, 400}.
- **Axes, one at a time** (others at default), then combined only where
  they helped: scale (fixed / searched), bias (normal / data),
  standardisation (zscore / pca / correlated weights), gamma penalty
  (off / on), QELM levels (pow2 / ternary / uniform) and the sparse path.
  Tabular and atom-centred.
- **Measure**: test RMSE / sigma for E / F / S; training time; QELM time and
  memory.
- **Run**: `example/mlip/adr0013/`, one ledger line per condition
  (resumable), nohup, estimated 1.5-2.5 h.
- **Tests**: finite-difference forces and stress per option, save / load,
  defaults unchanged (existing suite passes), full suite.

## Implementation

- `ELMModel`: `_normalise` / `_scale_dir` / `_to_raw` carry the input map
  (z-score, then the whitening map P for pca) for E rows, derivative rows
  and engine gradients; `_init_hidden(D, X_ref)` draws W (correlated with
  `_correlate`), passes it through `_finish_W` (QELM quantises there) and
  sets b (normal or data-centred); `fit` loops over the (scale factor,
  correlation length) candidates and refits the winner; `_solve_direct`
  searches (reg, gamma_smooth) jointly on the normal equations.
  `pca_whitening` is module level.  Model files gain `X_P` and
  `elm_options`.
- `ACSF.radial_groups(per_atom)`: G2 channels of one (neighbour species,
  eta) (and central-species block for the structure descriptor), sorted
  by r_s.
- `QuantizedELMModel`: `levels`, `max_exponent` (`n_bits` deprecated alias,
  DeprecationWarning; old files load), `n_levels`, `matmul` (dense | sparse
  product in the chain rows and the engine); `W_mem_int8_MB` in the
  quantisation stats.
- Atom-centred models: per-species whitening maps (`X_P_s` in the file),
  per-species column offsets (whitened widths may differ), the same
  searches over the per-species units.
- CLI: `mlip` passes the options (elm, qelm, atom_centred) when given;
  `descriptor_config` is set before the tabular fit; template block.
- Before the default was changed (Decisions after the evaluation), the
  defaults were checked unchanged: HEAD and the new code gave the same predictions
  for tabular ELM / QELM / reg_search and atom-centred ELM / QELM to 2e-9
  eV (HEAD against itself differs by the same 2e-9: BLAS threading).

## Verification

tests/test_elm_options.py (26 tests): for 12 tabular and 6 atom-centred
option sets, training rows = engine after save / load, engine forces and
stress = finite differences of its energy (central, h = 1e-5; for PCA
atom-centred a fourth-order difference, since the central one is off by
2e-6 at h = 1e-5 and falls as h^2 -- strong curvature along the amplified
near-null directions), sum F = 0.  Whitened training inputs have identity
covariance; an exactly duplicated column adds no direction (pca_eps 0);
the radial weight correlation matches the kernel (20 000 units, within
0.03) with unit variance and independent G5 rows; every data-centred unit
has a zero pre-activation at a training input; (gamma_smooth 1e6 cut the
r_s roughness of gamma by > 1e3 -- test removed with the option); the scale search reports all 5 factors
and keeps the best; ternary / uniform / pow2 level sets; refusals; CLI
pass-through.  Full suite: 585 passed (559 before + 26).

## Results (2026-09-28, example/mlip/adr0013/evaluate.py, report.md)

216 fits (3 data sets x tabular / atom-centred x N_h 25, 50, 100, 400 x
9 conditions), ~15 min on 3 processes.  Test RMSE / sigma, sigma = ADR-0004
sigma / 10 (E 0.2 meV/atom, F 0.005 eV/A, S 0.0002 eV/A^3); reg on the
validation split from 1e2..1e12.

**First test of PCA with the agreed drop rule** (Si, tabular, N_h 25):
structure descriptors vary in few directions (correlation eigenvalues
fall from 1 to 1e-6 within 18 directions and to 1e-16 by 55), but the
force rows -- derivatives of the descriptor -- carry weight in the
directions the E-row statistics call negligible:

| drop below (x max lambda) | directions kept (of 90) | E / F / S |
|---|---|---|
| 1e-6 (agreed) | 19 | 4.77 / 5.21 / 6.49 |
| 1e-9 | 32 | 2.13 / 3.27 / 1.95 |
| 1e-12 | 43 | 1.47 / 2.40 / 1.38 |
| none (eps 1e-3 only) | 90 | 1.13 / 1.75 / 1.11 |
| z-score (default) | 90 | 1.19 / 1.91 / 1.17 |

eps between 1e-3 and 1 changes little.  The default became "keep all"
(`pca_drop: 0`); the whole evaluation uses it.

**One axis at a time: F RMSE option / default (geometric mean over N_h)**

| data / form | scale search | data bias | PCA | radial corr. | gamma smooth | QELM pow2 | ternary | uniform |
|---|---|---|---|---|---|---|---|---|
| SW Si / tab | 0.990 | 0.996 | **0.924** | 1.003 | 1.000 | 0.998 | 0.998 | 0.996 |
| SW Si / ac | 0.975 | 0.997 | **0.963** | 1.019 | 1.000 | 1.008 | 1.004 | 1.007 |
| Tersoff / tab | 1.010 | 0.998 | 1.011 | 1.003 | 1.151 | 1.002 | 1.002 | 1.002 |
| Tersoff / ac | 1.004 | 0.980 | **0.808** | 0.966 | 1.061 | 0.998 | 0.980 | 0.991 |
| SW GeSn / tab | 0.999 | 0.994 | **0.941** | 0.997 | 1.000 | 0.999 | 0.996 | 1.001 |
| SW GeSn / ac | 0.993 | 1.003 | **0.948** | 1.004 | 1.000 | 1.002 | 1.000 | 1.001 |
| all | 0.995 | 0.995 | **0.930** | 0.998 | 1.034 | | | |

- **PCA whitening** is the only axis above the 2 % threshold: F improves
  in 22 of 24 (data, form, N_h) cells and E, F and S together in 19, e.g.
  Si ac N_h 25 F 1.88 -> 1.72, GeSn ac E 1.59 -> 1.36, Tersoff ac N_h 400
  F 9.71 -> 6.86.  F got worse in Si ac N_h 400 (1.57 -> 1.64, where the
  default is already closest to the floor) and Tersoff tab N_h 400
  (9.71 -> 11.27, reg picked 1e2 on a validation split dominated by
  outliers, see below).  Cost: none
  (one D x D eigendecomposition).  So the combination stage had nothing to
  combine.
- **Scale search / data bias**: < 1 % on average; the search picked 0.5
  (12 of 24), 8 (8) or 1 (4) -- no consistent direction; 4-5x fit time.
- **Radial correlation**: no gain (0.998); l picks spread over the whole
  grid (0.1: 8, 1.6: 6, 0.2: 5, 0.4: 4, 0.8: 1; 1.6 A was added because
  the first Si run picked the agreed grid's end, 0.8).
- **gamma smooth**: never helps; 0 picked in 16 of 24 cells (no effect),
  and where the validation split picked 1e2 / 1e4 (Tersoff) the test got
  worse (tab 9.7 -> 11.2): the penalty fits the validation outliers.
- **QELM level sets**: pow2, ternary and uniform (15 levels) match the ELM
  within 1-2 % everywhere, as expected for random features of matching
  variance.
- **Tersoff data caveat**: at 1500 K some frames hold rare close contacts
  (a 1.99 A bond, reference force 6.0 eV/A on one atom, predicted ~0) and
  pairs inside Tersoff's cutoff switch (R - D .. R + D = 2.70 .. 3.00 A;
  51 of 300 training and 18 of 100 test frames).  These dominate the RMSE
  (validation F 16-18 sigma against test 9-10 sigma) and push reg to the
  top of the grid in the default runs (the hidden layer then contributes
  little).  Tersoff numbers compare options under outliers; they are not
  a clean capacity test.

**QELM product and memory** (engine call on 64 atoms, one process):

| case | dense product | CSC product | W dense / CSC / int8 |
|---|---|---|---|
| Si ac N_h 400, p_zero 0.9 | 4.94 ms | 5.46 ms | 0.29 / 0.045 / 0.039 MB |
| GeSn tab N_h 2000, p_zero 0.5 | 31.7 ms | 78.2 ms | 7.0 / 5.2 / 0.89 MB |
| GeSn ac N_h 2000, p_zero 0.9 | 18.8 ms | 20.5 ms | 3.5 / 0.54 / 0.45 MB |

The engine call is dominated by the ACSF (the product is a small part);
the CSC product was never faster.  Product alone (numpy / scipy, 64 x 218
by 218 x 2000): dense 0.8 / 0.25 / 0.27 ms, CSC 3.1 / 0.66 / 0.11 ms at
p_zero 0.5 / 0.9 / 0.99 -- CSC wins only at 99 % zeros.  Memory does
shrink (CSC at p_zero 0.9: 6.5x; int8 levels: 8x), and accuracy at
p_zero 0.9 equals that at 0.5.  In this numpy implementation QELM offers
smaller W storage, not speed.

## Decisions after the evaluation (user, 2026-09-28)

1. **PCA whitening is the ELM / QELM default** (tabular and atom-centred):
   `standardize` defaults to `pca` (all directions kept, `pca_eps` 1e-3),
   or to `zscore` when `w_correlation: radial` is set (the channel layout
   needs unmixed inputs).  Model files written before ADR-0013 (no `X_P`
   / `elm_options`) load as z-score models; `standardize: zscore` gives
   the former behaviour.  Tests that were about z-score specifics now
   say so (the M1 weighting regression test, the RVFL -> linear model
   test compares `_to_raw(gamma)`); the atom-centred engine test uses a
   fourth-order difference (curvature along the whitened near-null
   directions, as above).
2. **gamma_smooth removed** (option, search, CLI key, template, tests).
3. **QELM kept** for its smaller W storage; memory comparison with the
   ELM under the same conditions: see below.
4. Scale search, data-centred bias and radial correlation stay options
   without a recommendation.

## ELM vs QELM model storage (2026-09-28, example/mlip/adr0013/memory.py, memory.md)

Same data, descriptor, N_h, seed, reg (1e6) and PCA default; only the
model class (and the QELM level set / p_zero) differs.  MB = 1e6 bytes.

| data / form | N_h | ELM file / in memory | QELM (pow2, p_zero 0.5) file / in memory | QELM W as int8 + scales | ELM W (float64) |
|---|---|---|---|---|---|
| Si / ac | 400 | 0.372 / 0.363 | 0.373 / 0.872 | 0.039 | 0.288 |
| Si / ac | 2000 | 1.550 / 1.540 | 1.550 / 4.084 | 0.196 | 1.440 |
| GeSn / ac | 400 | 2.192 / 2.186 | 2.192 / 4.637 | 0.181 | 1.395 |
| GeSn / ac | 2000 | 7.824 / 7.818 | 7.824 / 20.074 | 0.904 | 6.976 |
| GeSn / tab | 400 | 2.946 / 2.933 | 4.347 / 5.379 | 0.178 | 1.395 |
| GeSn / tab | 2000 | 8.552 / 8.539 | 15.547 / 20.771 | 0.888 | 6.976 |

**As implemented, QELM saves no memory:**

- Files: every QELM file stores W as float64.  The atom-centred file
  equals the ELM's; the tabular file is 1.5-1.8x larger, because it
  writes both `W` and `W_q_dense` (the same matrix).
- In memory: 2.3-2.6x the ELM, because QELM keeps the full-precision
  draw `_W_fp` (for `quantization_error`) and a CSC copy next to the
  dense quantised W.
- Level sets and p_zero do not change the stored size; only the CSC copy
  shrinks with p_zero.

**What the level structure would allow:** the levels are integers
(|k| <= 2^max_exponent, or <= (n_levels - 1) / 2) times a per-column
scale.  Stored as int8 plus one float64 scale per column, W takes 7-8x
less than the float64 W (e.g. GeSn ac N_h 2000: 0.90 MB against
6.98 MB).  The whole file would then shrink from 7.8 MB to about 1.8 MB;
the PCA map (0.76 MB) then becomes a large share.  This needs a compact
save format (int8 levels + scales, no duplicate W, `_W_fp` optional), a
change of the QELM file format that has not been made yet.

## QELM integer storage (user request 2026-09-28)

**Implementation.**  The quantised W is held as integer levels K and
column scales s, W = K * s exactly (K int8; int16 when a level exceeds
127, i.e. pow2 with max_exponent 7..14, now the allowed range).

- Files store `W_int` and `W_scale` only: no float W, no duplicate, no
  full-precision draw.  The atom-centred file stores `W_int_s` /
  `W_scale_s` per species.
- `w_memory: float64` (default) keeps a dense float W for the products,
  as before.  `w_memory: int8` keeps K and s only and forms X W as
  (X K) * s on every call.  numpy casts K to float inside the product,
  so a transient float copy exists during a call.
- The full-precision draw is kept only with `keep_fp: true`, which only
  `quantization_error` needs; without it that method now raises instead
  of comparing the model with itself.  The CSC copy is built only for
  `matmul: sparse`.
- Files with a float W still load.  In the tabular form, `W_q_dense` is
  converted with its stored scales.  In the atom-centred form, which
  stored no scales, s_j = max|W[:, j]| / kmax: the largest entry of a
  column always sits on the top level.  The conversion checks that the
  result is exactly integer.

**Verification** (tests/test_elm_options.py, +6 tests, 32 in total):
- For pow2, ternary and uniform: W equals K * s bit for bit, and the file
  holds int8 levels and no float W.
- The reloaded model predicts the same to 1e-12.
- `w_memory: int8` keeps no float W, and its predictions, engine E / F / S
  match float64 mode to 1e-9.
- Atom-centred int8 mode:
  - The same fitted model gives the same result through the int path and
    the float path (1e-10).
  - Training in the two modes differs only by product rounding amplified
    by the solve (2e-7 on random labels).
  - After save / load, forces and stress equal fourth-order finite
    differences.
- Old-format files load to the same integer levels (tabular and
  atom-centred).
- max_exponent 8 gives int16; `quantization_error` needs `keep_fp`.

**Storage and speed, same conditions** (example/mlip/adr0013/memory.md;
data, descriptor, N_h, seed, reg 1e6 and PCA equal; MB = 1e6 bytes; one
process):

| data / form | N_h | ELM file / memory | QELM int8 file / memory | ELM / QELM int8 engine ms | F / sigma ELM / QELM |
|---|---|---|---|---|---|
| Si / ac | 400 | 0.372 / 0.363 | 0.124 / 0.114 | 5.79 / 5.54 | 1.71 / 1.71 |
| Si / ac | 2000 | 1.550 / 1.540 | 0.307 / 0.296 | 7.07 / 7.18 | 1.79 / 1.71 |
| GeSn / ac | 400 | 2.192 / 2.186 | 0.978 / 0.971 | 18.25 / 17.39 | 1.87 / 1.81 |
| GeSn / ac | 2000 | 7.824 / 7.818 | 1.753 / 1.746 | 18.67 / 20.29 | 1.96 / 2.01 |
| GeSn / tab | 400 | 2.946 / 2.933 | 1.731 / 1.715 | 23.23 / 22.64 | 1.92 / 1.89 |
| GeSn / tab | 2000 | 8.552 / 8.539 | 2.467 / 2.451 | 32.63 / 36.68 | 2.81 / 2.94 |

- Files and in-memory models are 1.7-5.2x smaller than the ELM's.
- The W itself is 7.4x smaller (int8 plus one scale per column).
- The fit time is unchanged, and an engine call is at most 12 % slower
  (GeSn tab N_h 2000).
- The float64 memory mode is 6-13 % larger than the ELM, because it keeps
  both copies of W; its file is as small as in int8 mode.
- Once W is compressed, the PCA whitening map (float64, D x D per species)
  is the largest part: 1.5 of 2.5 MB in GeSn tab.  Storing it in float32
  would halve that; not done, since it is part of every ELM model.

## Quantised atom-centred NN: study (2026-09-28, example/mlip/adr0013/nn_quant.py)

**Method.**  Post-training quantisation (PTQ) of a trained
`AtomicNNModel` ([64, 64], softplus, deterministic, default PCA
standardisation not applied: the NN keeps z-score).  Every hidden Linear
weight matrix is quantised per output row, with a symmetric scale.  The
biases and the direct link stay in float.  The quantised network is
still smooth in its input, so its forces remain exact derivatives of its
energy; only accuracy is at stake.  Data as in the evaluation (sigma /
10), one seed.  Results are test RMSE / sigma, E / F / S:

| data | float | int8 (+-127) | pow2 (max_exponent 3, p_zero 0.5) | ternary (p_zero 0.5) | weights float64 / int8 | all parameters |
|---|---|---|---|---|---|---|
| SW Si | 2.18 / 2.83 / 1.89 | 6.48 / 2.85 / 2.00 | 144 / 13.2 / 29.3 | 461 / 12.5 / 14.8 | 79 / 10 KB | 81 KB |
| SW GeSn | 2.99 / 2.56 / 1.49 | 3.49 / 2.56 / 1.46 | 130 / 9.5 / 6.6 | 254 / 10.7 / 19.0 | 290 / 36 KB | 295 KB |

**Findings.**
- **int8 PTQ keeps forces and stress** (F unchanged) but shifts
  energies: E grows 3x on Si (to 1.3 meV/atom).  The shift is a nearly
  constant energy offset from the rounding.  A cheap fix is to re-solve
  the last layer (and the direct link) by linear least squares with the
  quantised hidden layers frozen -- the ELM solve.  Not tried here.
- **pow2 and ternary PTQ destroy the trained network.**  Unlike the ELM,
  where W is random and quantised before the output layer is solved, an
  NN's trained weights carry the fit.  Coarse levels would need
  quantisation-aware training (straight-through estimator), with the
  forces as double-backprop through the quantiser -- not justified for
  the gain below.
- **Little to save.**  The whole NN is 81-295 KB, already 4-7x smaller
  than an ELM with N_h 400 (0.37-2.2 MB); int8 would save 70-260 KB.  The
  memory case for quantisation is the ELM's large random W, which QELM
  now covers.

**Conclusion.**  A quantised NN is feasible (int8 PTQ, plus a last-layer
re-solve for the energy offset), but its memory benefit is small.  It is
not implemented; recorded as an option.

**Decision (user, 2026-09-28): a quantised NN is not recommended and not
implemented.**  The reasons are the study above:
- The NN's parameters are already small.
- Coarse levels would need quantisation-aware training.
- int8 PTQ shifts the energy.
Quantisation stays the QELM's role.  `w_memory: int8` is the QELM
default (same date).

**PCA and memory.**  With W in int8, the whitening map is the largest part
of a QELM file: D x D float64 per species, e.g. 1.52 of 2.47 MB for GeSn
tabular N_h 2000.  `standardize: zscore` stores two D-vectors instead,
which brings that file to about 0.95 MB.  The cost is the PCA accuracy
gain (7 % force error on average here).  Storing the map in float32
would halve it without giving up whitening; not done.

## Addendum (2026-10-14, 0.2.0a12): the same standardisation for the NN

User decision: where the ELM family (ELMModel, QuantizedELMModel,
AtomicELMModel, AtomicQELMModel) has `standardize` / `pca_eps` /
`pca_drop`, the NN family (NNModel, AtomicNNModel) takes them too.
Before this change the NN read none of these keys, so a config that set
them ran with the z-score default.

**Implementation**
- `standardize: zscore | pca`. The NN default stays `zscore`, so
  existing results and model files are unchanged.
- `NNModel` applies one map to the whole table, as `ELMModel` does. E
  rows are z-scored and then multiplied by `pca_whitening`'s P. F and S
  rows get the same map without the mean. The network and its direct
  link see the D' whitened inputs. `energy_and_gradient` maps the
  gradient back with P^T / std.
- `AtomicNNModel` uses the per-species maps that `atomic._prepare`
  already built for the atom-centred ELM. Each map is a fixed first
  layer (a buffer x P_s) in front of that species' MLP. Everything
  outside the networks stays in z-score space: the Jacobians, the
  isolated-atom point x0, and the engine gradient.
- The map is saved with the model. Files written without it load as
  z-score models.

**Verification** (tests/test_nn_standardize.py)
- The F and S rows of a `pca` model equal finite differences of its
  energy.
- The map equals `pca_whitening` of the training structures' E rows.
- `pca_drop` narrows the input of both the MLP and the direct link.
- Save / load round-trips, and the atom-centred engine stays
  conservative.
- The isolated-atom constraint holds exactly.
- The CLI passes the keys to both NN forms.
- A one-off comparison against the previous commit showed bit-for-bit
  equal predictions with the z-score default: 960 tabular rows and 93
  atom-centred values.

**Use on the GeSn ver2.1.0 tables**
PCA did not rescue the tabular NN on the 1801-column external tables
(ADR-0021 addendum).  The test force error depended on `pca_drop`:

| pca_drop | directions kept | test F (eV/A) |
|---|---|---|
| 0 | 1801 | 74 |
| 1e-6 | 163 | 8.7 |
| 1e-4 | 95 | 1.1 |
| 1e-3 | 56 | 3.6 |
| 1e-2 | 23 | 0.78 |

The error is not monotonic in `pca_drop`. Training itself is unstable
here: the training-set F error tracks the test error.
