---
status: proposed (plan, 2026-11-15)
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
