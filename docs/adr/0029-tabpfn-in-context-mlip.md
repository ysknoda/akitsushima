---
status: rejected (2026-11-15; spike done, gate not passed -- see Result)
---

# TabPFN as an in-context MLIP: feasibility spike and adoption gate

Agreed with the user on 2026-11-15 (grilling, Q1-Q21).  The user wants the
regression mode of TabPFN -- a tabular foundation model that predicts from
a table of labelled rows given at inference time, with no training step --
used as a machine-learning interatomic potential on the package's tabular
E / F / S descriptor data (ADR-0004), in the hope of a workflow like that
of a universal potential.  Nothing is implemented in the package by this
ADR: it fixes the model form, a feasibility spike under `example/`, and
the gate that decides whether a package implementation follows.  Results
are appended as an appendix.

## Starting point (2026-11-15, 0.2.0a16)

**The package.**
- The tabular route exists (ADR-0004): per structure one E row, 3N F rows
  and 6 S rows, the structure descriptor d and its derivatives X, built
  from the in-package ACSF by `build_table`.
- ADR-0004 decision 1: for a nonlinear model E / N = f(d) the force is
  F_ia = N grad f(d) . X_ia, which depends on the structure's own d.
  Predicting an F row as f(X_ia), row by row, is not the derivative of
  one energy; the model is then not a force field.
- ADR-0012 removed LightGBM for exactly that (no conservative force) and
  for its accuracy: 4.5-6.1 / 1.3-1.6 / 1.6-2.8 sigma (E / F / S) on Si
  against about 1.0 for the tabular linear model and ELM (ADR-0005).

**TabPFN** (looked up on 2026-11-15; not installed before this ADR).
- PyPI `tabpfn` 10.0.0 (released 2026-11-13); default weights TabPFN-3.5.
- TabPFN-3.5: up to 1,000,000 rows and 20,000 features; on a CPU a guard
  stops at 5,000 rows and the project itself calls CPU inference slow.
- Code Apache-2.0.  The weights of 2.5 and later are under non-commercial
  licences and need a PriorLabs login and licence acceptance on first use;
  the v2 weights are under the Prior Labs License (Apache 2.0 with an
  attribution clause).
- The regressor returns a predictive distribution (mean, median,
  quantiles), so an uncertainty comes with every prediction.
- `fit_with_differentiable_input` exists, but `predict` runs in inference
  mode: a gradient with respect to the test input is not a documented
  feature.  It has to come from the internal torch model or from finite
  differences (read from the source, UNVERIFIED).
- `tabpfn` depends on lightgbm.  lightgbm and torch could not share a
  process here (OpenMP runtime, ADR-0012); whether importing tabpfn loads
  lightgbm is the first thing the spike checks.
- No published use of TabPFN as an interatomic potential (forces, MD) was
  found; property prediction from composition or structure descriptors
  exists.

## Decisions

### Frame

1. **What "like a universal potential" means (Q1).**  (a) The user gives
   their own DFT data as the table and gets E / F / S for that system
   with no fit and no hyperparameter tuning.  TabPFN knows nothing about
   atoms; the system's DFT data are still needed.
   - (b) A context table shipped with the package that covers many
     elements, so that a user needs no DFT data, is **not in scope now
     and is kept as a future option** (the user may adopt it later).  Its
     prerequisites (Q17): a descriptor whose column count does not grow
     with the number of species (ACSF grows with its square); the choice
     and the licence of a public data set for the context; covering many
     elements inside the context-row limit (1,000,000 for TabPFN-3.5).
     The timing is the user's.
2. **Model form (Q2, Q10).**  An energy model with chain-rule forces and
   stress, so the result is a conservative force field usable by every
   solver:

       E / N = w . d + b + t(d)

   with w . d + b the existing tabular linear model (ADR-0004 decision 9:
   F and S rows exact, equivalent to an atom-centred linear model, 1.03 /
   1.01 / 0.97 sigma on SW Si alone) and t the TabPFN regressor on the
   residual.  F_ia = N (w + grad t(d)) . X_ia, and likewise for the
   stress, compressive-positive (ADR-0001).  This is the `direct_link`
   idea of the ELM and NN defaults: TabPFN carries a small quantity, so
   float32 matters less, and extrapolation falls back to the linear part.
   The linear fit is closed-form (seconds) and has nothing to tune.
   - TabPFN alone (no linear part) is measured as a comparison.
   - The row-independent form (every E / F / S row predicted on its own)
     is measured as a reference only and is not to be shipped.
3. **Descriptor (Q3).**  The in-package ACSF through `build_table` (the
   request's "ASDF").  The structure-averaged tabular form: an
   atom-centred form E = sum_i eps_i has no per-atom labels to give
   TabPFN.  External descriptors at inference stay deprioritised
   (ADR-0010 item 9).
4. **Force information in the context (Q2, Q11).**  A table of E rows
   only would discard more than 99 % of the labels.  F rows enter as
   energies of displaced structures to first order (the Taylor-expansion
   approach of A. M. Cooper, J. Kaestner, A. Urban and N. Artrith, npj
   Comput. Mater. 6, 54 (2020)): for a displacement u of all atoms,
   target E - sum_ia F_ia u_ia, descriptor recomputed for the displaced
   positions.  Collective displacements, because one atom moved by
   0.01 A under 1 eV/A changes a 64-atom cell by 0.16 meV/atom, far
   below the 2 meV/atom energy noise.  S rows enter the same way as
   strained cells.
5. **Weights and licence (Q4).**  Academic, non-commercial use.
   TabPFN-3.5 by default, the checkpoint selectable.  In a later package
   implementation `tabpfn` is an optional extra, the weights are never
   bundled, and the model file records the tabpfn version and the
   checkpoint.  The login and licence acceptance are the user's own act.
6. **Hardware (Q5).**  The spike runs on this Mac (torch 2.13, MPS);
   production use is aimed at the H100 machine of ADR-0016.  Each MD step
   needs a transformer forward pass and a gradient, so the use for now is
   small-cell `relax`, short `md` and static prediction; it will be orders
   of magnitude slower than the numba ELM (0.30 ms/atom, ADR-0028).
7. **Order of work (Q6, Q7, Q9, Q16).**  This ADR first, as `proposed`;
   then the spike; a package implementation only if the gate passes, and
   then after a short grilling whose result is added here.  The spike is
   independent of ADR-0028's tier A, whose order does not change.

### The spike

8. **Installation (Q8).**  `tabpfn==10.0.0` into the base environment.  A
   dry run on 2026-11-15 showed 11 new packages (tabpfn, safetensors,
   einops, huggingface_hub, hf-xet, httpx2, httpcore2, skrub, pydot, mlx,
   mlx-metal) and two upgrades (click 8.2.1 -> 8.5.1, idna 3.11 -> 3.20);
   numpy, scipy, torch, numba and scikit-learn stay as they are.
9. **Script (Q15).**  One self-contained script in
   `example/mlip/adr0029/` (argparse, English, one JSONL record per
   condition so that an interrupted run resumes), with the
   finite-difference and sum-of-forces checks built in.  Package code is
   not touched and the version is not bumped; the full test suite still
   runs after every edit.  Runs longer than about 5 minutes go under
   nohup with a log file.
10. **Variants (Q11, Q14).**
    1. E rows only (the lower bound).
    2. E rows + collective-displacement rows: M in {4, 16, 64} per
       structure, |u| per atom delta in {0.01, 0.03} A (both sets are
       first estimates, UNCERTAIN).
    3. Row-independent (reference).
    4. Variant 2 + strained-cell rows from the S labels.
    - Each with and without the linear part (decision 2).
    - TabPFN with its default preprocessing and with the identity
      preprocessing of the differentiable path; `random_state=0`.
    - Gradient by autograd first, cross-checked against finite
      differences.
    - With a reference potential the Taylor target is compared with the
      exact energy of the displaced structure, which separates the error
      of the approximation from TabPFN's.
    - 300 training structures first (at most about 20,000 context rows);
      1000 only for variants that pass (about 65,000 rows at M = 64;
      whether that fits in MPS memory is unknown).
11. **Systems (Q18-Q21).**  Most MLIP use is for compounds (the user,
    2026-11-15), so the adoption is judged on two-species systems.
    - **Screen: Si**, Stillinger-Weber (PRB 31, 5262 (1985)), the sets of
      ADR-0004 / 0005 (64 atoms; 216-atom cells and slabs for transfer).
      Used for the stop rules only.
    - **Adoption sets, exact labels:**
      - GeSn, Stillinger-Weber of Tomita et al., Jpn. J. Appl. Phys. 57,
        04FB04 (2018), the sets of ADR-0005: 64-atom random alloys, 300
        training structures (x_Sn = 0, 0.25, 0.5, 0.75, 1), 100 test
        structures, 80 structures at compositions absent from training
        (x_Sn = 0.125 ... 0.875).  An alloy: tests interpolation in
        composition.
      - SiC, Tersoff (PRB 39, 5566 (1989)), a stoichiometric compound.
        The data set is new, made by the Si recipe (64 atoms, MD at
        300-1500 K, strains up to +/-3 %); the linear and ELM baselines
        are measured on it first.
      - NiAl (EAM) is left out: the main target is covalent
        semiconductors (ADR-0028).  Ionic oxides are unsuitable here, as
        a short-range descriptor cannot carry the long-range Coulomb
        part.
    - **Real DFT, only after the adoption sets pass:** AlN (`AlN_V01` +
      round 0) first, then GeSn round 2 (563 structures).  DFT has no
      known noise sigma, so the comparison is the E / F / S RMSE of the
      current atom-centred ELM on the same split.
    - **Three species, after that:** a ternary alloy labelled with the
      EAM of Zhou, Johnson and Wadley (PRB 69, 144113 (2004)), to measure
      accuracy and time against the column count only (ACSF per atom: 90
      columns for Si, 218 for GeSn).  The elements are proposed once the
      parameter table has been checked.  A ternary covalent compound
      needs real DFT data and stays an open item.

### Gate and stop rules

12. **Adoption gate (Q12, Q19)**, applied to both adoption sets, with the
    label noise of ADR-0004 (E 2 meV/atom, F 0.05 eV/A, S 0.002 eV/A^3):
    - correctness (required): forces and stress agree with finite
      differences of the energy; sum_i F_i = 0; the same input gives the
      same result;
    - accuracy: test E, F and S each <= 2 sigma;
    - transfer: no worse than the tabular ELM -- on GeSn at the
      compositions absent from training, on Si for the 216-atom cells and
      slabs;
    - force-field quality: NVE without a systematic energy drift on 64
      atoms; relaxed lattice constant within 0.1 % of the reference
      potential's;
    - cost: one 64-atom force call <= 1 s on MPS.

    The thresholds 2 sigma, 0.1 % and 1 s are proposals with a weak basis
    (UNCERTAIN).  Accuracy merely equal to the ELM's still counts as
    adoption if the absence of tuning and the uncertainty output are
    confirmed.
13. **Stop rules (Q13)**, judged on Si.  Stop and report if
    - the lightgbm / torch clash cannot be avoided;
    - neither autograd nor finite differences gives forces that agree
      with the finite difference of the energy;
    - F exceeds 5 sigma in every variant of decision 10.

## Open items (decided after the gate, Q16)

- Model class, `mlip.model: tabpfn` in the YAML, model-file format
  (whether the context table is stored), thinning the context above the
  row limit, the uncertainty API, which solvers are supported.
- Column-count scaling for three and more species in covalent compounds.
- Option (b) of decision 1.

## Consequences

- A negative result is recorded here as LightGBM's was, with the
  measurements.
- If adopted, the default path's 1e-10 reproducibility standard
  (ADR-0028) cannot hold for this model (float32 transformer); it stays
  opt-in.

## Appendix A: environment and gradient check (2026-11-15)

Scripts: `example/mlip/adr0029/check_coexistence.py`,
`check_gradient.py`, `check_smoothness.py` (example/ is not tracked).
Test function y = sin(2 x0) + x1 x2 + 0.5 x3^2 on 300 uniform points in
[-1, 1]^5, CPU, float32 unless stated; y std 0.86.

- **Installation.**  `tabpfn==10.0.0` in base, the packages of
  decision 8 and nothing else.
- **Coexistence.**  `import tabpfn` and a fit / predict do not load
  lightgbm; numba parallel kernels (omp layer) work before and after
  torch and tabpfn.  Stop rule 1 does not apply.
- **Gradient route.**  `differentiable_input=True`,
  `fit_with_differentiable_input`, then the private
  `_iter_forward_executor` with a test tensor that requires grad; the
  prediction is the mean over the ensemble of the bar-distribution means.
  A test row's prediction does not depend on the other test rows.
- **TabPFN-3.5 is not a continuous function of its input.**
  - With the default `FINGERPRINT_FEATURE` (a hash of the row) the
    prediction jumps by about 1e-2 between points 1e-6 apart.  It can be
    switched off (`inference_config`).
  - Without it jumps of 1e-3 to 3e-3 remain, about one per 1e-3 of path.
    Cause: the architecture feeds every cell's midrank against the train
    rows (ECDF, `_ecdf_midrank_counts` in `tabpfn_v3_5.py`) to the
    model, a step function that moves by 1 / n_train whenever a test
    value crosses a train value.  Between two crossings the function is
    smooth and autograd is exact (float64, step 1e-6: slope -0.18282
    against autograd -0.18282), but the autograd gradient misses the
    jumps: against central differences its rms error is 0.08-0.09 at
    h = 1e-2 and 1e-3, for a gradient of rms 0.72.
  - `ecdf_num_buckets` reduced to 64 or 16 leaves fewer jumps of the
    same size and doubles the prediction error (0.014 -> 0.025).
  - This is part of the pretrained model and cannot be configured away:
    as it stands, TabPFN-3.5 gives an energy surface with small steps,
    not a conservative force field.
- **TabPFN v2 is smooth.**  The v2 architecture has no rank feature.
  With identity preprocessing and no fingerprint: second differences
  along the 1e-4 scan 6.6e-7 (float32 rounding), autograd against
  central differences 2.3e-4 rms at h = 1e-3, prediction rmse 0.0115
  (3.5: 0.0139), 200 test rows in 1.1 s (3.5: 5.3 s).  Its documented
  limits are 10,000 rows and 500 features (Hollmann et al., Nature 637,
  319 (2025)), and its weights are under the Prior Labs License.
- **v2.5, v2.6 and v3 are smooth too** (tested after the user accepted
  their licences).  Same test; limits read from each version's
  `inference_config_`:

  | version | max abs 2nd difference, 1e-4 scan | autograd - FD rms, h = 1e-3 | rmse | 200 rows (s) | rows | features |
  |---|---|---|---|---|---|---|
  | v2 | 6.6e-7 | 2.3e-4 | 0.0115 | 1.1 | 10,000 | 500 |
  | v2.5 | 9.8e-7 | 2.6e-4 | 0.0121 | 1.2 | 50,000 | 2,000 |
  | v2.6 | 7.5e-7 | 9.6e-5 | 0.0331 | 1.7 | 100,000 | 2,000 |
  | v3 | 6.6e-7 | 9.1e-4 | 0.0123 | 2.2 | 1,000,000 | 2,000 |
  | v3.5 | 2.7e-3 | 9.3e-2 | 0.0139 | 5.3 | 1,000,000 | 20,000 |

  Only 3.5 is discontinuous.
- **Cost, first sight.**  CPU, 300 context rows, 6 test rows, forward +
  backward: 10 s for 3.5.  The same on MPS ran out of memory (11.2 GiB);
  not investigated.

Consequence for decision 5 (weights): open, to be decided with the user.

## Appendix B: Si screen, first results (2026-11-15)

`example/mlip/adr0029/tabpfn_spike.py` (commands `taylor`, `run`,
`fdcheck`, `report`; ledger.jsonl).  SW Si, 300 training structures of 64
atoms, 90 ACSF columns, TabPFN v2 on the CPU, identity preprocessing, no
fingerprint feature.  Test RMSE / sigma on `si_test`.

**Taylor targets against exact SW energies** (20 structures, exact
labels; meV/atom):

| rows | error mean | error rms | first-order signal rms |
|---|---|---|---|
| displacement, delta 0.01 A | -0.87 | 0.88 | 1.03 |
| displacement, delta 0.03 A | -7.85 | 7.88 | 3.09 |
| strain, eta 0.002 | -0.04 | 0.05 | 0.58 |
| strain, eta 0.005 | -0.27 | 0.28 | 1.46 |
| strain, eta 0.01 | -1.09 | 1.13 | 2.92 |

The error is the second-order term (every atom moved: about
0.5 k delta^2 per atom) and is the same for +u and -u.  delta = 0.03 A is
unusable; decision 10's set becomes {0.003, 0.01} A.

**Accuracy** (E / F / S):

| model | context rows | ensemble | test structures | E / F / S |
|---|---|---|---|---|
| linear alone | - | - | 50 | 1.08 / 1.01 / 0.99 |
| linear + TabPFN, E rows | 300 | 8 | 50 | 1.09 / 1.01 / 0.99 |
| TabPFN alone, E rows | 300 | 8 | 50 | 1.19 / 1.89 / 1.60 |
| TabPFN alone, E rows | 300 | 1 | 16 | 1.30 / 1.55 / 1.47 |
| TabPFN alone, E + 4 displacement rows, delta 0.01 | 1500 | 1 | 16 | 1.48 / 3.98 / 3.90 |
| TabPFN alone, E + 4 displacement rows, delta 0.003 | 1500 | 1 | 16 | 1.85 / 5.12 / 2.94 |
| TabPFN alone, E + 4 strain rows, eta 0.005 | 1500 | 1 | 16 | 1.75 / 5.30 / 2.14 |

- The linear model alone is at the noise floor, and the residual left
  for TabPFN (std 1.96 meV/atom) is the 2 meV/atom label noise: on SW
  data the main variant of decision 2 cannot differ from the linear
  model, so these sets cannot show whether TabPFN adds anything to it.
- TabPFN alone from 300 energies gives forces at 1.5-1.9 sigma, inside
  the 2 sigma gate, without any force label.
- The Taylor rows make every variant tried WORSE (F 1.55 -> 4.0-5.3
  sigma at equal settings).  Not explained yet; the suspicion is that
  clusters of nearly identical rows mislead the in-context estimate of
  noise and smoothness.  Decision 4 does not work as designed.

**Correctness** (one test structure, E-row context): sum of forces
1e-12 eV/A.  Chain-rule forces against central differences of the model
energy: with the linear part 1.1e-4 / 9.7e-6 / 5.1e-6 eV/A at h = 0.01 /
0.003 / 0.001 A, stress 2.4e-6 eV/A^3 at h = 0.001; TabPFN alone 9.0e-4
eV/A at h = 0.01 A, growing to 8.3e-3 at h = 0.001 A -- float32 rounding
of an energy that now carries the full 60 meV/atom spread, which is what
the linear part was meant to remove.  Stop rule 2 does not apply.

**Cost** (CPU, 24 GiB machine): 0.85-0.89 s per 64-atom structure with
300 context rows and 8 ensemble members (gate: 1 s), 0.16 s with one
member.  Memory is the problem: with autograd, 1500 context rows x 90
columns peak at 15-16 GiB with ONE member and 16 test rows, and the
8-member run was killed by the system.  Larger contexts need either
finite differences in the descriptor (no graph) or a GPU.

## Appendix C: AlN, real DFT (2026-11-15)

Done before the SW two-species sets at the user's choice: on SW data the
linear model is at the noise floor (appendix B), so only real data can
show whether TabPFN adds to it.  `tabpfn_spike.py real`: the AlN tutorial
set (440 training structures of 63-64 atoms, 69 held-out test structures),
the tutorial's ACSF (144 table columns), `akitsushima mlip` default
weights; TabPFN v2, CPU, context = the 440 E rows.  Absolute test RMSE:

| model | E (meV/atom) | F (eV/A) | S (eV/A^3) |
|---|---|---|---|
| atom-centred ELM (tutorial log, 04_mlip_elm) | 1.12 | 0.140 | 0.00198 |
| tabular linear | 1.46 | 0.161 | 0.00227 |
| tabular ELM (n_hidden 512) | 1.46 | 0.161 | 0.00227 |
| linear + TabPFN residual | 1.15 | 0.191 | 0.0388 |
| TabPFN alone | 1.47 | 0.253 | 0.132 |

- TabPFN on the residual improves the energy to the atom-centred ELM's
  level and makes forces 19 % and stress 17 times worse than the linear
  model it is added to.  Alone it is worse still.
- Reading (not verified): the E rows fix t(d) only on the set of
  structure descriptors; its gradient in the directions the F and S rows
  point along is unconstrained, and the stress rows point furthest out of
  that set.  The force and stress labels are needed, and the one way
  tried to give them to TabPFN (Taylor rows) made things worse on Si.
- Cost: 3.7 s per structure for the residual model, 13.8 s alone, with
  about 17 GiB resident and swapping (peak footprint 61 GiB).

**State of the gate:** no variant passes on real data; the correctness
part passes (appendix B).  Not adoptable as designed.  Open: why the
Taylor rows hurt.

## Appendix D: why the Taylor rows do not help (2026-11-15)

Three bounded tests, agreed with the user as the last step before
closing.  SW Si, 300 training structures, TabPFN alone (no linear part),
one ensemble member, 16 test structures, CPU (`sweep2.sh`).  From here on
the scores also include the absolute RMSE against the NOISE-FREE reference
labels, which the 0.05 eV/A force noise had been hiding.

| weights | training labels | context | rows | F vs exact (eV/A) | E vs exact (meV/atom) | S vs exact (eV/A^3) |
|---|---|---|---|---|---|---|
| v2 | exact | E rows | 300 | 0.083 | 0.50 | 0.0021 |
| v2 | exact | + 4 Taylor rows, delta 0.01 A | 1500 | 0.106 | 0.59 | 0.0035 |
| v2 | exact | + 4 rows with TRUE energies, delta 0.01 A | 1500 | 0.099 | 0.83 | 0.0031 |
| v2 | exact | + 4 rows with true energies, delta 0.05 A | 1500 | 0.098 | 0.28 | 0.0021 |
| v2 | exact | + 4 rows with true energies, delta 0.15 A | 1500 | 0.094 | 0.49 | 0.0011 |
| v2 | noisy | + 4 rows with true energies, delta 0.05 A | 1500 | 0.077 | 0.61 | 0.0025 |
| v3 | noisy | E rows | 300 | 0.050 | 0.69 | 0.0015 |
| v3 | noisy | + 4 Taylor rows, delta 0.01 A | 1500 | 0.383 | 2.22 | 0.0100 |
| linear + TabPFN v2 residual (= linear) | noisy | E rows | 300 | 0.0030 | 0.16 | 0.00007 |

- **Not the label noise:** with exact training labels the Taylor rows
  still make the forces worse (0.083 -> 0.106).
- **Not the Taylor error:** true energies of the displaced structures in
  place of the Taylor estimate change nothing (0.099).
- **Not the closeness of the rows:** true energies at 0.05 and 0.15 A,
  i.e. five times as many exact, well separated energies, leave the force
  error at 0.094-0.098, above the 300-row value.
- **v3 is not the answer:** better from E rows alone (0.050), far worse
  with the Taylor rows (0.383).
- **The measure that matters:** against noise-free labels TabPFN alone is
  at 0.05-0.10 eV/A in every variant, the linear model at 0.003 eV/A --
  17 to 33 times better.  The "1.5-1.9 sigma, inside the gate" of
  appendix B came from the 0.05 eV/A label noise dominating the ratio.

Reading: the force error of TabPFN is not limited by the data it is given.
More and better energies do not sharpen the gradient of its in-context
fit; a model that fits F and S rows directly (linear, ELM) uses the same
structures an order of magnitude better.

## Result (2026-11-15)

**Not adopted.**  Against the gate of decision 12:

| item | outcome |
|---|---|
| correctness | passes: TabPFN v2 / v2.5 / v2.6 / v3 are smooth and autograd forces match finite differences; **TabPFN-3.5, the current default, is discontinuous in its input** and cannot be a force field |
| accuracy | fails: on real AlN data the residual model improves E (1.46 -> 1.15 meV/atom) but worsens F (0.161 -> 0.191 eV/A) and S (0.0023 -> 0.039 eV/A^3); alone it is 17-33 times worse than the linear model in true force error on SW Si |
| force labels in the context | fails: Taylor rows, and even true displaced energies, do not improve forces |
| cost | 0.16-0.9 s per 64-atom structure at 300 context rows on the CPU; 15-17 GiB of memory with autograd from 440-1500 rows |

Not run, as the gate failed before them: SW GeSn, Tersoff SiC, GeSn round
2, the ternary column test, NVE and lattice constants, the row-independent
reference (variant 3), MPS / GPU timings.

What stays:
- The package is unchanged.  `tabpfn` 10.0.0 remains installed in base;
  the scripts and ledgers are in `example/mlip/adr0029/`.
- One positive finding: 440 energies alone give TabPFN a residual
  correction that brings the tabular linear model's energy error to the
  atom-centred ELM's.  An energy-only use (ranking structures, an energy
  correction with a predictive distribution) is a possible future item;
  it is not a force field.
- Option (b) of decision 1 and its prerequisites stay recorded as a
  future option.  It would additionally need a tabular model that can use
  force labels, which this ADR did not find.
