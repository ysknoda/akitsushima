---
status: accepted
---

# MLIP: atom-centred energy model, and its comparison with the tabular approach

ADR-0004 makes the tabular approach (structure descriptor d with its
derivatives stored as E/F/S rows, trained with the chain rule) exact and usable
as a `relax`/`md` engine. This ADR adds the conventional atom-centred approach
beside it -- the high-dimensional neural network potential of Behler and
Parrinello (PRL 98, 146401 (2007)) -- and fixes how the two are compared. It
is implemented after ADR-0004 and reuses its ACSF descriptors, weighting rule,
model file and engine interface.

The two approaches differ in one essential way. The tabular model sees one
descriptor per structure; the atom-centred model sums contributions that each
depend only on one atom's neighbourhood within the cutoff. Locality is what
lets a model trained on one cell size be used on another, or on surfaces and
defects. Whether the tabular approach, which the user designed for its
convenience (one table, any tabular learner), gives up accuracy or
transferability for that is an open, measurable question.

## Decisions

1. **Model form.** E = sum_i eps_{s(i)}(X_i), with X_i the ACSF vector of
   atom i (ADR-0004, decision 4) and eps_s one function per species s.
   F_j = -dE/dr_j via the per-pair Jacobians dX_i/dr_j; stress from the cell
   derivative (virial), compressive-positive (ADR-0001), from the start.

2. **Models: ELM (with QELM through inheritance) and NN.**
   - ELM: eps_s(X) = beta_s . g(W_s X + b_s). E, F and S are all linear in
     beta, so E, F and S rows form one weighted least-squares system, solved
     in closed form.
   - NN: forces and stress by torch autograd.
   - LightGBM is not included: trees give no usable gradient (ADR-0004,
     decision 3).
   - `ELMModel.compute()`, which already evaluates an atom-centred form, is
     replaced by this model's evaluation.

3. **Training data** in the format of the classical `fit`: `structures.csv`
   with E/F/S rows and `<k>/POSCAR` per structure. Descriptors are computed
   in the package, so the CSV has no descriptor columns. Data labelled for T8
   can be reused, and importing VASP results later shares its path with the
   classical fit.

4. **Weighting**: the ADR-0004 rule, w_cat x w_unit^2 x w_user.

5. **Force/stress-row subsampling** is to be considered for this model.
   Unlike the tabular models (network evaluated once per structure, an F
   row is one dot product), the atom-centred model evaluates the network
   per atom and needs per-atom Jacobian contractions for the forces, so
   using a fraction of the F rows (or atoms) per step can cut training time
   substantially (ADR-0004 follow-up, Q4).

6. **Engine**: the same `potential: {type: MLIP, model: ...}` entry as
   ADR-0004; the model file records that it is atom-centred.

7. **Verification**: as ADR-0004 decision 7 -- finite-difference forces and
   stress, invariances, learning SW-labelled Si to the noise level, NVE
   conservation and the relaxed lattice constant.

## Comparison plan (after this ADR is implemented)

Accuracy measurements of the tabular side (items a and d) may be taken as
soon as ADR-0004 is implemented; the full comparison runs after this ADR.

**Descriptors** -- both of:
- (i) ACSF on both sides (tabular: per-atom-averaged ACSF; atom-centred:
  per-atom ACSF, same G2/G4 parameters). Isolates the model form.
- (ii) Tabular with the user's `atomic_descriptor_ver2.0.0.py` descriptors
  (RDF/ADF/ODF) against atom-centred ACSF: the combination each would be used
  with. Accuracy and training time only; an engine for (ii) needs FW-1.

**Models**: tabular ELM, QELM, NN (chain rule) and LightGBM (row-independent);
atom-centred ELM, QELM, NN. LightGBM is left out of item (d).

**Items measured**:
- (a) accuracy: test E/F/S RMSE relative to the label noise sigma;
- (b) training time, and inference time per structure (one md step);
- (c) learning curve: 100, 300, 1000 training structures;
- (d) force-field quality: NVE energy conservation, relaxed lattice constant,
  and reproduction of the reference potential's lattice constant and elastic
  constants;
- (e) transferability: cells of a size not in training (train 64 atoms, test
  216 atoms), surface-containing structures, and strains outside the
  training range -- ADR-0004's SW-Si data set v1, on which the tabular
  nonlinear models extrapolated poorly (ELM 10 sigma; NN 2-132 sigma
  depending on seed and activation; linear 1.0 sigma).

**Data**, in this order:
1. Si labelled with Stillinger-Weber (PRB 31, 5262 (1985)): 64-atom diamond
   cells, snapshots from 300-1500 K MD and cells strained by up to +/-3%.
   SW has a three-body term, so a descriptor without angular information
   cannot reproduce it. Labels are exact, so the accuracy limit is measurable;
   Gaussian noise sigma may be added to make ratio ~1 the target.
2. GeSn with the known potential used in T8, for two species.

The results are recorded as an appendix to this ADR.

## Consequences

- The package offers both approaches with one engine interface; the user
  chooses per system, informed by the comparison.
- A second training-data path (structures + POSCAR) exists next to the CSV
  path; it is the one real VASP data will enter by.

## Appendix: implementation and verification (2026-09-25)

### Implementation notes

- **One ACSF kernel, per-entry Jacobians.**  The ACSF kernel now returns,
  for every neighbour entry m of atom i (vector v_m = r_j + shift - r_i),
  dG_i/dv_m (`AtomicJacobian`).  Everything else is a contraction of it:
  dG_i/dr_j = sum_{m of i -> j} dG_i/dv_m - [i = j] sum_{m of i} dG_i/dv_m
  and dG_i/dT_ab = sum_{m of i} v_{m,a} dG_i/dv_{m,b}.  The tabular
  `structure_table` is built from the same Jacobians (so both approaches
  use one derivative code), and the ADR-0004 ACSF tests pass unchanged.
- **Models.**  `AtomicELMModel` holds one ELM unit per species (an
  `ELMModel`, seeds seed + s), so activations, the direct link (default
  on, as ADR-0004 decision 10) and the output bias come from the tabular
  class; `AtomicQELMModel` only swaps the unit class for
  `QuantizedELMModel` (QELM through inheritance).  The hidden-layer draw
  moved into `ELMModel._init_hidden`, which `QuantizedELMModel` overrides
  to quantise; its `fit` no longer needs the `seed = None` trick.  The
  normal equations are accumulated structure by structure, and
  `reg_search` scores each candidate from accumulated validation
  moments, so no design matrix is stored.  `AtomicNNModel` is one
  NNModel-built MLP per species (softplus, direct link, [64, 64],
  batches of 30 structures, patience 150, at most 1500 epochs, as the
  tabular NN), with forces and stress from autograd g_i contracted with
  the stored Jacobians (float32 on the device).
- **NN energy offset.**  The NN starts each species' output bias at its
  least-squares reference energy (total energies against composition).
  Without it the first tens of epochs only learn the -4.3 eV/atom
  offset: on 30 small SW-Si cells after 30 epochs, force RMSE 8.7 eV/A
  without and 0.76 eV/A with it.
- **Engine and CLI.**  The model file records `atom_centred`; the MLIP
  engine evaluates E = sum_i eps_i and contracts g_i = d eps_i/dG_i with
  the Jacobians.  `mlip.form: atom_centred` (models elm, qelm, nn) trains
  from the `fit`-layout structure list; the model is written as
  `atomic_<model>_model.npz` (`.pt`).  `ELMModel.compute()` and
  `make_force_fn(..., descriptor_fn=)` are removed (decision 2);
  `make_force_fn(model)` now wraps any MLIP model in `MLIPForceField`.
- **Decision 5 (force/stress-row subsampling) measured, not adopted.**
  On SW-Si v2 (450 structures): atom-centred ELM fit 3.6 s with all,
  half or a quarter of the F rows (the per-entry design rows, not the
  row count, set the cost); atom-centred NN 0.307 against 0.279 s per
  epoch with a quarter of the F rows (-9%).  The energy already needs
  eps_i and its gradient g_i for every atom, and the Jacobian
  contraction is cheap (profile: 10.6 ms forward + forces, 19.2 ms with
  the backward pass, per 30-structure batch).  The option was removed
  again.

### Tests

`tests/test_mlip_atom_centred.py` (15 tests): the Jacobian contraction
against finite differences for positions and all nine cell components
(three species, cell thinner than 2 r_c); the tabular table as the
per-atom average of the atomic Jacobian; atom-centred ELM (tanh, rff
without direct link, softplus) and QELM: training design rows equal the
engine's E/F/S to 1e-10, engine forces and stress against finite
differences, sum F = 0; recovery of a teacher of the same form from E, F
and S rows to 1e-6 (rows, weights and solve together); locality (a 2x2x2
supercell has 8x the energy, the same forces and stress); invariance
under rotation, translation and permutation; `make_force_fn(model)`;
row layout identical to `build_table`; CLI `mlip` (form atom_centred,
with a test set) -> `relax` (variable cell) -> `md` (NVE); refusal of
LightGBM; NN training path equal to its engine and engine forces and
stress against finite differences (float32 tolerances); NN learning and
bit-reproducibility with `deterministic`.

### Comparison (`example/mlip/adr0005/compare.py`; full tables in `compare_report.md`)

**Reference potential.**  All labels were made with the engine *before*
ADR-0006: Stillinger-Weber with the three-body term counted twice (Si:
C11 / C12 / C44 = 201.8 / 51.6 / 90.6 GPa instead of SW's 151.4 / 76.4 /
56.4), and the Tomita GeSn set counted the same way.  The comparison
measures how well each approach learns a known three-body potential,
which this still is; the numbers are not those of SW itself.  Re-running
on the corrected engine is an open decision (ADR-0006).

**Setup.**  Noise sigma, weights (1/sigma^2 per row) and model settings as
ADR-0004 decision 10 (ELM / QELM 400 units per species or in total,
reg_search 1e2-1e10; NN [64, 64] softplus + direct link, 3 seeds,
deterministic; LightGBM 20 Optuna trials, 30 min cap).  Si: pool of 1000
64-atom structures (20 cells, unstrained + 19 strains up to +/-3%, 300-1500
K), nested training subsets of 100 / 300 / 1000; test = ADR-0004's v2 test
set (200).  216-atom cells (60) and Si(001) slabs (96 atoms, 12 A vacuum,
24) test transfer; v1 (train 450, test 200 with an unseen strain) as in
ADR-0004.  GeSn: 64-atom random alloys x_Sn = 0 ... 1, 300-700 K,
train 300, test 100 (same compositions) and 80 (x_Sn = 0.125 ... 0.875,
absent from training).  Values are test RMSE / sigma (E / F / S); each test
set's own noise floor (the ratio of a perfect model) is Si test 1.02 /
1.00 / 0.97, 216-atom 1.22 / 1.00 / 1.02, slab 0.92 / 1.00 / 0.97, GeSn
1.06 / 1.00 / 0.98 and 0.96 / 1.00 / 1.02.  Ranges are over NN seeds.

**(a, c) Accuracy and learning curve, Si, descriptor (i) ACSF:**

| model | n = 100 | n = 300 | n = 1000 |
|---|---|---|---|
| tabular linear | 1.08 / 1.26 / 0.98 | 1.02 / 1.01 / 0.97 | 1.02 / 1.01 / 0.97 |
| tabular ELM / QELM | 1.10 / 1.41-1.43 / 1.00 | 1.02 / 1.01 / 0.97 | 1.02 / 1.01 / 0.97 |
| tabular NN | 1.06-1.31 / 1.19-1.23 / 1.04-1.06 | 1.06-1.34 / 1.08 / 1.00-1.01 | 1.04-1.09 / 1.06-1.07 / 1.00-1.02 |
| tabular LightGBM | 7.09 / 1.73 / 2.09 | 6.64 / 1.46 / 1.67 | 5.80 / 1.33 / 2.49 |
| atom-centred ELM / QELM | 1.10 / 1.48-1.49 / 1.00 | 1.02 / 1.01 / 0.97 | 1.02 / 1.01 / 0.97 |
| atom-centred NN | 1.08-1.24 / 1.08-1.14 / 0.99-1.02 | 1.03 / 1.05 / 0.99-1.00 | 1.20 / 1.04 / 1.01 |

**(b) Time** (Si, n = 1000; shared ACSF table 19 s, atomic Jacobians
comparable): fit linear 2 s, tabular ELM / QELM 11 s, atom-centred ELM /
QELM 19-21 s, tabular NN 389-554 s, atom-centred NN 1719 s (one seed;
all runs shared the machine, so times carry contention), LightGBM 30 min
(capped).  One engine call on 64 atoms: 9-21 ms for every force-field
model; LightGBM 85 ms (n = 100) - 918 ms (v1) per structure's rows.

**(d) Force-field quality** (Si, n = 1000; relaxed 64-atom cell,
`ElasticConstantsCalculator` with relaxed ions; reference relaxed the same
way: a = 5.42955 A, C11 / C12 / C44 = 201.8 / 51.6 / 90.6 GPa):

| model | a [A] | C11 | C12 | C44 | NVE std / drift [meV/atom] |
|---|---|---|---|---|---|
| tabular linear | 5.42953 | 201.7 | 51.9 | 89.8 | 0.016 / 0.001 |
| tabular ELM / QELM | 5.42931 | 202.6 | 52.5 | 90.2 | 0.016 / 0.000 |
| tabular NN | 5.42755 | 206.6-209.3 | 51.8-53.8 | 92.1-92.8 | 0.016 / 0.000 |
| atom-centred ELM | 5.42927 | 201.5 | 51.5 | 90.4 | 0.016 / 0.001 |
| atom-centred QELM | 5.42940 | 201.3 | 51.3 | 90.4 | 0.016 / 0.001 |
| atom-centred NN | 5.42745 | 205.5 | 51.8 | 91.8 | 0.016 / 0.001 |

GeSn x = 0.5 (reference a = 6.07981 A, 103.1 / 21.8 / 45.7 GPa): atom-
centred ELM / QELM 6.0780 / 6.0776 A, 104.4-104.5 / 21.8-22.2 / 45.2;
tabular linear 6.0728 A, 103.8 / 21.2 / 45.6; tabular ELM / QELM
6.0704 / 6.0610 A, C12 16.4-17.2; tabular NN 6.049 A, C12 14.8-17.0;
atom-centred NN 6.0762 A, C12 18.8-20.6.  Every engine conserves energy
as the reference does (NVE std 0.016 meV/atom Si, 0.002 GeSn).

Decision 7 on ADR-0004's v2 split (train 450): atom-centred ELM 1.02 /
1.02 / 0.97, QELM 1.02 / 1.01 / 0.97, NN 1.04-1.16 / 1.04-1.05 /
1.00-1.01 -- all pass (<= 1.5); lattice within 0.05% and NVE as SW.

**(e) Transferability.**
- 216-atom cells (trained on 64): every model at the floor (E 1.21-1.27,
  F 1.01-1.02, S 1.02-1.05; floor 1.22 / 1.00 / 1.02), atom-centred NN E
  1.35; LightGBM 3.86 / 15.8 / 6.6.  At this cutoff the per-atom-averaged tabular descriptor of a
  larger cell of the same phase lies where the 64-atom ones do, so size
  alone does not separate the approaches.
- Si(001) slabs: no model has seen a surface; E 30-41 sigma for all (tabular
  NN 32-35, atom-centred NN 30, the others 37-41, LightGBM 82), F 1.5-1.6
  (tabular NN 2.6-2.8, LightGBM 8.0), S 1.0-1.1 (tabular NN 1.4-1.6,
  LightGBM 2.2).
- Strain outside the training range (v1): tabular ELM 2.57 / 2.14 / 4.90
  (as in ADR-0004) and LightGBM 8.69 / 1.56 / 9.71 fail; tabular QELM 1.02
  / 1.03 / 1.07 and NN 1.07-1.30 / 1.10-1.19 / 1.04-1.43 pass; every
  atom-centred model passes (ELM / QELM 1.02 / 1.02 / 0.98, NN 1.10-1.15 /
  1.07-1.08 / 0.99-1.01).
- GeSn, compositions absent from training: atom-centred ELM / QELM 1.18-
  1.24 / 1.46-1.48 / 1.08-1.09 and tabular linear 1.18 / 1.60 / 1.11 are
  closest to the floor; tabular ELM / QELM 1.91-2.61 / 1.73-1.77 /
  1.13-1.20; tabular NN 4.0-6.1 / 2.7-2.9 / 1.4-3.3; atom-centred NN
  1.30-2.02 / 1.67-1.73 / 1.09-1.18; LightGBM 18.0 / 4.0 / 3.8.  On the
  same compositions (gesn_test) the atom-centred ELM is also the best
  nonlinear model (1.13 / 1.26 / 1.00 against tabular ELM 1.36 / 1.27 /
  1.01 and tabular NN 3.0-3.3 / 1.8 / 1.1-1.3).

**Descriptor (ii): the user's RDF / ADF / ODF descriptors** (tabular
models; accuracy and time only):

| data | linear | ELM / QELM | NN | LightGBM |
|---|---|---|---|---|
| Si n = 100 | 14.3 / 43.0 / 16.0 | 27 / 161 / 37 | 15-54 / 71-396 / 19-84 | 8.9 / 4.9 / 3.8 |
| Si n = 300 | 15.0 / 16.2 / 14.7 | 15.1 / 19.8 / 15.0 | 14.7-15.9 / 13-39 / 14-17 | 8.3 / 4.1 / 5.6 |
| Si n = 1000 | 14.6 / 9.8 / 14.0 | 14.6 / 10.1-10.2 / 14.0 | 13.9-14.5 / 9.2-9.4 / 13.1-13.6 | 6.5 / 3.8 / 4.3 |
| GeSn test | 588 / 6258 / 787 | 765-2420 / ... | 100-1020 / ... | 9.8 / 19.1 / 3.3 |
| GeSn unseen x | 31 / 602 / 61 | 41-127 / ... | 99-969 / ... | 11.9 / 24.8 / 4.6 |

Settings: Si rc1 / rc2 = 1.5 / 1.8855 A per element (pair 3.0 / 3.771 A,
the ACSF r_c), dr 0.05 A, dtheta 1 deg, sigma_r 0.1 A, sigma_theta 2 deg
(824 columns); GeSn the user's `config_GeSn.yaml.example` (5006 columns);
file route (POSCAR -> *.dat, 7 significant digits).  Checks made before
recording these:
- The F and S rows are the derivatives of the E row: finite differences
  on a 64-atom Si cell agree to 0.9995-0.99999 (projection ratio), so the
  file route and the conventions are right.
- The Si floor is not regularisation: the training error is already
  10 / 9 / 13-14 sigma with the smallest alpha (1e-10), for the linear
  model and for the NN.  A plausible reading: SW's three-body term
  couples each leg's radial factor to the angle, which an angle histogram
  with one radial weight plus a separate distance histogram cannot
  express per triplet; ACSF G5 with several eta does.
- GeSn: one test structure (x_Sn = 0.25, 600 K) is off by -5874 sigma in
  E (median 7.7 sigma).  Two ADF3 Ge-Sn-Sn bins (126-127 deg) are empty in
  every training structure and filled in the test, so the models
  extrapolate along unconstrained columns; alpha / reg ended at the top of
  the grid (1e-4 / 1e10).  With alpha up to 1e4 the outlier disappears
  but validation picks the top again and the fit settles at 55 / 25 / 6
  sigma (train 56 / 20 / 5).  The grids were sized for ACSF; the result is
  an upper bound for these descriptor settings, not for the descriptor
  family.

### Reading of the comparison

- With ACSF on both sides and training data covering the test conditions,
  the tabular and the atom-centred forms are equally accurate (both at the
  noise floor from n = 300) and equally good force fields; at n = 100 the
  tabular NN and the atom-centred NN learn forces fastest.
- Where the test leaves the training distribution, the atom-centred ELM /
  QELM are the most robust nonlinear models: unseen strain (v1) and unseen
  GeSn compositions, where the tabular ELM and NN degrade (the tabular
  linear model, itself atom-centred, is as robust).  LightGBM is the least
  accurate everywhere and is no force field.
- Surfaces are outside every model's training data and no approach
  transfers there (30-41 sigma in E); training data must contain them.
- Cost: the closed-form ELM fits take seconds to tens of seconds either
  way; the atom-centred NN is 3-4x slower to train than the tabular NN.
  Engine calls cost the same (9-21 ms per 64-atom step).
- The user's RDF / ADF / ODF descriptors, as configured here, do not reach
  the noise level on SW-type data with any tabular model (Si 10-15 sigma,
  GeSn much worse with sparse-bin extrapolation).  The limit is the
  descriptor, not the tabular form: ACSF in the same tabular models
  reaches the floor.
