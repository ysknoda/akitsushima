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
