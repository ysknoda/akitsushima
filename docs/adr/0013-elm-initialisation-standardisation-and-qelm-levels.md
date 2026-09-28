---
status: accepted (implementation and evaluation in progress)
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
   uses x~ = (Lambda + eps)^(-1/2) U^T (x - mu).  Directions with lambda below
   1e-6 * max(lambda) are dropped (numerically meaningless); the rest are
   kept, with eps added (no cut by explained variance, which could drop
   small components that matter for forces).  F and S rows (descriptor
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
6. **Smoothness penalty on gamma** (Q46): `gamma_smooth` (default 0) adds
   the squared differences of the direct-link weights of adjacent r_s
   channels (same groups as item 5) to the normal equations -- the idea of
   the tabular `smooth_ridge`.
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

## Results

(to be filled in after the run)
