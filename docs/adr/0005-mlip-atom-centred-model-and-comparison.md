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

**Reference potentials** (after ADR-0006): Stillinger-Weber Si (PRB 31,
5262 (1985); reproduces C11 / C12 / C44 = 151.7 / 76.7 / 56.4 GPa and
a = 5.4295 A with this procedure) and the Tomita GeSn set (Jpn. J. Appl.
Phys. 57, 04FB04 (2018), LAMMPS leg convention).  A first run of this
comparison used the engine before ADR-0006 (three-body term doubled,
GeSn legs from the mixed rows); its ledgers are kept in
`example/mlip/adr0005/archive_pre_adr0006/`, and how its conclusions
differ is noted at the end.

**Setup.**  Noise sigma, weights (1/sigma^2 per row) and model settings as
ADR-0004 decision 10 (ELM / QELM 400 units per species or in total,
reg_search 1e2-1e10; NN [64, 64] softplus + direct link, 3 seeds,
deterministic; LightGBM 20 Optuna trials, 30 min cap).  Si: pool of 1000
64-atom structures (20 cells, unstrained + 19 strains up to +/-3%,
300-1500 K), nested training subsets of 100 / 300 / 1000; test = the v2
test set (200).  216-atom cells (60) and Si(001) slabs (96 atoms, 12 A
vacuum, 24) test transfer; v1 (train 450, test 200 with an unseen strain)
as in ADR-0004.  GeSn: 64-atom random alloys x_Sn = 0 ... 1, 300-700 K,
train 300, test 100 (same compositions) and 80 (x_Sn = 0.125 ... 0.875,
absent from training).  Values are test RMSE / sigma (E / F / S); each
test set's own noise floor (the ratio of a perfect model) is Si test
1.02 / 1.00 / 0.97, 216-atom 1.22 / 1.00 / 1.02, slab 0.92 / 1.00 / 0.97,
GeSn 1.06 / 1.00 / 0.98 and 0.96 / 1.00 / 1.02.  Ranges are over NN seeds.

**(a, c) Accuracy and learning curve, Si, descriptor (i) ACSF:**

| model | n = 100 | n = 300 | n = 1000 |
|---|---|---|---|
| tabular linear / ELM / QELM | 1.08 / 1.02 / 0.97-0.98 | 1.03 / 1.01-1.02 / 0.97 | 1.03 / 1.01-1.02 / 0.97-0.98 |
| tabular NN | 1.18-1.53 / 1.19-1.20 / 1.06-1.13 | 1.06-1.18 / 1.06 / 0.98-0.99 | 1.07-1.12 / 1.05 / 0.98-1.03 |
| tabular LightGBM | 6.12 / 1.64 / 1.95 | 4.86 / 1.36 / 1.62 | 4.52 / 1.26 / 2.77 |
| atom-centred ELM / QELM | 1.08 / 1.02 / 0.98 | 1.03 / 1.01 / 0.97 | 1.03 / 1.01 / 0.97 |
| atom-centred NN | 1.12-1.25 / 1.06 / 0.98-1.01 | 1.10-1.11 / 1.04 / 0.98-1.00 | 1.03-1.09 / 1.03 / 0.97-0.98 |

**(b) Time** (Si, n = 1000; all runs shared the machine, so times carry
contention): shared ACSF table 19 s; fit linear 4 s, tabular ELM / QELM
14-16 s, atom-centred ELM / QELM 31-32 s, tabular NN 364-502 s,
atom-centred NN 1466-3100 s, LightGBM 40 min (30 min search cap).  One
engine call on 64 atoms: 8-21 ms for every force-field model;
LightGBM 75-358 ms per structure's rows (not an engine).

**(d) Force-field quality** (relaxed 64-atom cell; `ElasticConstantsCalculator`
with relaxed ions; reference relaxed the same way):

| model (Si, n = 1000) | a [A] | C11 | C12 | C44 [GPa] | NVE std / drift [meV/atom] |
|---|---|---|---|---|---|
| reference SW | 5.42955 | 151.7 | 76.7 | 56.4 | 0.014 / 0.001 |
| tabular linear | 5.42990 | 151.2 | 77.1 | 55.0 | 0.014 / 0.001 |
| tabular ELM / QELM | 5.42977-5.42982 | 151.1-151.6 | 76.4-76.9 | 55.6 | 0.014 / 0.001 |
| tabular NN | 5.42653 | 154.2-155.5 | 75.9-77.2 | 57.5-58.0 | 0.014 / 0.000 |
| atom-centred ELM / QELM | 5.42940-5.42943 | 152.2-152.5 | 77.2-77.4 | 56.0 | 0.014 / 0.001 |
| atom-centred NN | 5.42905 | 154.5 | 76.8 | 57.7 | 0.014 / 0.000 |

GeSn x = 0.5 (reference a = 6.07293 A, 73.8 / 38.7 / 26.6 GPa): linear,
ELM and QELM of both forms 6.0701-6.0704 A, 73.1-73.4 / 38.4-38.6 /
26.3-26.5; tabular NN 6.0637 A, C11 73.5-75.6; atom-centred NN 6.0710 A,
C11 74.7-75.3.  Every engine conserves energy as the reference does.

Decision 7 on ADR-0004's v2 split (train 450): atom-centred ELM 1.03 /
1.01 / 0.97, QELM 1.02 / 1.01 / 0.97, NN 1.07-1.08 / 1.04 / 0.98-1.01 --
all pass (<= 1.5); lattice within 0.05%, elastic constants within 1.5
(ELM / QELM) and 4 GPa (NN), NVE as SW.

**(e) Transferability.**
- 216-atom cells (trained on 64): every force-field model at the floor
  (E 1.21-1.25, F 1.01-1.02, S 1.02-1.06); LightGBM 3.7 / 14.4 / 6.3.
- Si(001) slabs (no surface in training): E 12-27 for every model
  (tabular linear 12.2, ELM / QELM 14.5-15.0 either form, tabular NN
  20.9-21.5, atom-centred NN 13.7-26.8, LightGBM 75); F 1.16-1.66, S
  about 1.0.  No approach transfers to surfaces; they must be trained.
- Strain outside the training range (v1): every model except LightGBM
  passes (tabular linear / ELM / QELM 1.02-1.03 / 1.04-1.05 / 0.99-1.06,
  NN 1.04-1.09 / 1.10-1.14 / 0.99-1.06; atom-centred ELM / QELM 1.01 /
  1.02 / 0.98, NN 1.04-1.16 / 1.06-1.08 / 0.99); LightGBM 10.2 / 1.5 /
  7.2.
- GeSn, compositions absent from training: linear, ELM and QELM of both
  forms at the floor (0.97 / 1.02 / 1.02, also 1.10 / 1.01 / 0.98 on the
  same compositions); atom-centred NN 1.16-1.19 / 1.07 / 1.02-1.05; the
  tabular NN degrades (E 2.1, 11.6 and 20.9 for the three seeds, F
  1.14-1.59, S 1.12-1.33); LightGBM 14.5 / 2.0 / 2.3.

**Descriptor (ii): the user's RDF / ADF / ODF descriptors** (tabular
models; accuracy and time only; test RMSE / sigma):

| data | linear | ELM / QELM | NN | LightGBM |
|---|---|---|---|---|
| Si n = 100 | 9.8 / 5.5 / 9.3 | 9.7 / 6.6 / 9.4 | 9.4-33.9 / 6.4-50.7 / 9.3-12.4 | 6.1 / 4.2 / 3.2 |
| Si n = 300 | 10.4 / 5.3 / 10.0 | 10.3 / 5.3-5.4 / 10.0 | 10.0-10.2 / 5.4-5.5 / 9.7-9.9 | 5.8 / 3.0 / 2.9 |
| Si n = 1000 | 9.9 / 5.4 / 9.6 | 9.9 / 5.4 / 9.5 | 9.0-9.7 / 5.4-6.6 / 8.6-9.4 | 4.4 / 2.8 / 2.3 |
| GeSn test | 18.6 / 109 / 10.7 | 19-20 / 132-150 / 14-16 | 26-62 / 45-186 / 5-19 | 11.7 / 16.4 / 2.5 |
| GeSn unseen x | 37 / 555 / 101 | 246-250 / 4319-4454 / 813-842 | 400-554 / 5229-7570 / 875-1194 | 23.1 / 20.9 / 3.3 |

Settings: Si rc1 / rc2 = 1.5 / 1.8855 A per element (pair 3.0 / 3.771 A,
the ACSF r_c), dr 0.05 A, dtheta 1 deg, sigma_r 0.1 A, sigma_theta 2 deg
(824 columns); GeSn the user's `config_GeSn.yaml.example` (5006 columns);
file route (POSCAR -> *.dat, 7 significant digits).  Checks:
- The F and S rows are the derivatives of the E row (finite differences
  on a 64-atom Si cell agree to 0.9995-0.99999 as a projection ratio), so
  the file route and the conventions are right.
- The Si limit is not regularisation: the training error is already
  6.8-7.7 / 5.5-5.7 / 9.4-10.1 with alpha 1e-10-1e-6, and the NN is no
  better.  A plausible reading: SW's three-body term couples each leg's
  radial factor to the angle, which an angle histogram with one radial
  weight plus a separate distance histogram cannot express per triplet;
  ACSF G5 with several eta can.  LightGBM, which regresses rows
  independently, is the best model on these descriptors (4.4 / 2.8 / 2.3
  at n = 1000) and still far from the floor.
- GeSn (first run, same behaviour): test structures fill ADF bins that
  are empty in every training structure (e.g. ADF3 Ge-Sn-Sn 126-127 deg),
  so the fits extrapolate along unconstrained columns; alpha / reg end at
  the top of their grids (1e-4 / 1e10).  With alpha up to 1e4 the
  extreme errors disappear but validation picks the top again and the
  fit settles at about 55 / 25 / 6.  The grids were sized for ACSF; the
  result bounds these descriptor settings, not the descriptor family.

### Reading of the comparison

- With ACSF on both sides the tabular and the atom-centred forms of the
  closed-form models (linear, ELM, QELM) are equally accurate and equally
  good force fields: at the noise floor from n = 100-300, on larger cells,
  on unseen strains and on unseen GeSn compositions, with elastic
  constants within 1.5 GPa of the reference.  Nothing measured here
  favours the atom-centred ELM over the tabular one.
- The NN is where the forms differ: the atom-centred NN stays near the
  floor on unseen GeSn compositions (E 1.16-1.19) where the tabular NN
  does not (2.1-20.9, seed-dependent), and it has the lower Si training
  error at n = 1000.  It trains 3-6x slower.
- Surfaces are outside every model's training data and no approach
  transfers there; LightGBM is the least accurate everywhere and is no
  force field.
- The user's RDF / ADF / ODF descriptors, as configured here, do not
  reach the noise level on SW-type data with any tabular model (Si
  5-10 sigma); the limit is the descriptor, since ACSF in the same models
  reaches it.

**Against the first run (doubled three-body term, other GeSn legs).**
There the tabular ELM failed v1 (2.57 / 2.14 / 4.90) and the tabular ELM /
QELM trailed the atom-centred ones on unseen GeSn compositions (1.91-2.61
against 1.18-1.24 in E), which suggested an advantage of the atom-centred
ELM out of distribution.  With the correct potentials both gaps close.
Which reference is learned (here a three-body term twice as strong, and
for GeSn a different leg convention) evidently changes how far the
averaged-descriptor ELM extrapolates; the two changes were not separated.
Conclusions of this kind should be re-checked on the target data (real
DFT) before being relied on.
