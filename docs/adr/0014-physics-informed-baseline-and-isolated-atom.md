---
status: accepted (A, B and the ACSF inner cutoff implemented 2026-09-28)
---

# Physics-informed MLIP parts: pair baseline (A) and isolated-atom limit (B)

Requested by the user on 2026-09-28: implement A (a short-range repulsive
baseline, Delta-learning) and B (the physical limit of an isolated atom)
now.  The attractive and oscillating terms are a study only (below).
C (Hessian / elastic-tensor losses) is recorded as future work in
ADR-0010 item 11.

The ELM / QELM / NN are already physics-informed in one sense: E, F and S
come from one energy function, and forces and stress are trained as its
exact derivatives (chain rule, ADR-0004).  Symmetries come from the
descriptor; locality and extensivity from the atom-centred form.  A and B
add prior knowledge where the training data are silent.

## A. Pair baseline (Delta-learning)

    E = E_ML + E_base,   E_base = 1/2 sum_i sum_j phi_{s_i s_j}(r_ij) s(r_ij)

The labels are E - E_base (and F, S likewise) at training time; the MLIP
engine adds E_base back.

- **Terms** (`fitting/physics.py`, `PairBaseline`):
  - `zbl`: the universal screened Coulomb repulsion (Ziegler, Biersack &
    Littmark 1985).  The coefficients (0.18175, 0.50986, 0.28022,
    0.02817) and exponents (3.19980, 0.94229, 0.40290, 0.20162), and
    a = 0.46850 / (Z_i^0.23 + Z_j^0.23) A, were checked against the
    LAMMPS pair_style zbl documentation.  K_E comes from
    akitsushima.constants.
  - `born_mayer`: A exp(-r / rho) per species pair.
- **Switch.**  s(r) is a C2 quintic: 1 below r_inner, 0 above r_outer.
  The baseline is multiplied by s, so it vanishes exactly beyond r_outer.
  LAMMPS's zbl instead adds a polynomial shift.
- **Storage and use.**  The baseline is stored in the model's
  `physics_config` (JSON), which every model class now saves.  The
  engine adds it to E, F, S and to the per-atom site terms (gcmc, kappa).
  r_outer must not exceed the descriptor cutoff; that is checked.
- **CLI.**  `mlip.baseline: {type: zbl, r_inner, r_outer}` (or a list of
  terms).  The shortest distance in the training data is printed next to
  r_outer.
- **Verification** (tests/test_mlip_physics.py):
  - ZBL equals its formula (rel 1e-12).
  - The switch is continuous in value and slope, and zero beyond r_outer.
  - Baseline forces and stress equal finite differences on a cell with
    close contacts (|F| up to 1200 eV/A).
  - The engine returns learned part + baseline.  Its site terms rebuild
    the same E, F, S.  Forces and stress are derivatives of the total, in
    the atom-centred and the tabular form.
  - A baseline beyond the descriptor cutoff is refused.

## B. Isolated-atom limit

With `isolated_atom_energy: {Si: E_atom, ...}`, eps_s(G = 0) = E_atom,s
holds exactly.  An atom with no neighbour inside the cutoff then has the
isolated-atom energy and no force.

- **ELM / QELM** (atom-centred; tabular for one species only, since d = 0
  does not fix the composition): an equality constraint on the
  least-squares solve.  It is solved as a KKT system, with the
  constraint rows scaled to the normal matrix (`solve_constrained`), and
  applied inside the reg search.  The LSQR solver is refused.
- **Atom-centred NN**: by construction during training
  (net(x) - net(x0) + E_atom), then folded into the output bias, so
  inference is unchanged.
- **Tabular NN and linear models** refuse the option.
- **Verification**:
  - eps_s(0) = E_atom to 1e-9 for ELM and QELM (the unconstrained fit
    misses it by > 1e-3, so the test discriminates).
  - An isolated-atom configuration gives E = sum E_atom, F = 0, S = 0.
  - Tabular one-species f(0) = E_atom.
  - NN to 1e-5 (float32).
  - Refusals; CLI with both options.
- **Full suite:** 600 passed (591 + 9 in tests/test_mlip_physics.py); 602 after the inner cutoff (+2); 605 after the fitted Born-Mayer term (+3).

## Evaluation (example/mlip/adr0014/evaluate.py, report.md)

**Setup.**
- Data: SW Si and Tersoff Si (ADR-0013 data, sigma / 10), atom-centred.
- Models: ELM and QELM (N_h 400, PCA and int8 defaults, reg searched);
  NN [64, 64].
- Conditions:
  - A1: ZBL switched 1.0 -> 1.8 A, acting only below the data, whose
    shortest distance is 2.0 A.
  - A2: ZBL switched 1.5 -> 2.1 A.
  - B: E_atom = 0.  Both reference potentials vanish for a free atom.

**Probes.**
- Bond push: one atom of the 64-atom diamond cell is moved along a bond
  towards its neighbour, and E(r) - E(r0) is recorded.
- Expansion: the crystal is expanded to 1.3 and 1.7 a0.

| data | model | condition | E / F / S | push at 1.8 / 1.5 / 1.2 / 1.0 A [eV] | monotonic | expanded 1.3 / 1.7 a0 [eV/atom] |
|---|---|---|---|---|---|---|
| Si | reference | | | 3.8 / 12.7 / 38.6 / 85.6 | yes | -1.410 / 0.000 |
| Si | ELM | none | 1.04 / 1.64 / 1.05 | 11 / 286 / 555 / 379 | no | -1.998 / -1.328 |
| Si | ELM | A1+B | 1.06 / 1.67 / 1.08 | 6 / 84 / 181 / 165 | no | -1.430 / 0.000 |
| Si | NN | none | 2.18 / 2.83 / 1.89 | 3.3 / 5.1 / 8.5 / 11.5 | yes | -1.271 / -0.751 |
| Si | NN | A1+B | 2.44 / 3.06 / 1.81 | 3.2 / 7.4 / 30.7 / 62.5 | yes | -1.482 / 0.000 |
| Tersoff | reference | | | 4.1 / 16.2 / 52.8 / 91.9 | yes | 0.000 / 0.000 |
| Tersoff | ELM | none | 2.65 / 6.86 / 2.24 | 47 / 848 / 744 / 311 | no | 0.639 / 1.071 |
| Tersoff | ELM | A1+B | 2.65 / 6.85 / 2.22 | 47 / 802 / 612 / 240 | no | 0.170 / 0.000 |
| Tersoff | QELM | none | 2.60 / 6.97 / 2.19 | 42 / 393 / -390 / -518 | no | 0.598 / 1.211 |
| Tersoff | NN | none | 8.03 / 7.65 / 3.43 | 3.3 / 5.0 / 8.7 / 12.2 | yes | -0.087 / 3.108 |
| Tersoff | NN | A1+B | 8.08 / 8.82 / 2.52 | 3.0 / 6.6 / 28.6 / 58.0 | yes | 0.526 / 0.000 |

(All rows, A1 / A2 / B separately and QELM: report.md.)

**Findings.**
- **B works as intended.**
  - The expanded crystal (1.7 a0, no neighbour in the cutoff) now has
    exactly the isolated-atom energy.  Without B the models give between
    -1.3 and +3.1 eV/atom there.
  - Cost for ELM / QELM: the test error changes by at most 3 %.
  - For the NN only A1+B was run.  F got 8 % worse on Si and 15 % worse
    on Tersoff; S got 27 % better on Tersoff.
- **A works with the NN, not with the ELM / QELM as they stand.**
  - The NN alone is too soft under compression: 8.5 eV at 1.2 A against
    38.6.  With A1+B it gives 30.7 (Si) and 28.6 (Tersoff, reference
    52.8), and the curve stays monotonic.
  - The ELM / QELM run away under compression: +555 eV, or -390 eV
    (collapse) for the Tersoff QELM.  A cannot correct that.
  - Cause, by a check that switched one factor at a time:
    - The direct-link term gamma . x is linear in the whitened
      descriptor.  Beyond the data it grows without bound, and the PCA
      gain (1 / sqrt(pca_eps) on near-null directions) amplifies it.
    - With direct_link off, the push stays bounded (Si: 2.3 / 2.9 / 2.4 /
      2.0 eV).  With A1+B it becomes monotonic and physical (2.7 / 5.8 /
      25.9 / 54.5 eV).  But the test error rises from 1.6 to 31 sigma in
      F on these data.
    - With z-score and the direct link, Tersoff collapses (-379 eV at
      1.5 A).
  - So the README's statement that the direct link "extrapolates more
    gracefully" does not hold for close contacts.  It has been corrected.
- **A2 (switch overlapping the data's edge) is worse than A1** for the
  ELM (+2238 eV at 1.2 A on Si).  The baseline then changes the labels
  where the model is fitted, and does not help the extrapolation.

## Decision: inner cutoff in the ACSF (user, 2026-09-28, option 2)

The user chose option (2) of the three put to them:
1. Soft saturation of the direct-link inputs.
2. An inner cutoff in the descriptor.
3. Recommending the NN, or direct_link false with A.

**Implementation.**
- `ACSF(..., inner_cutoff=(r_off, r_on))` multiplies every cutoff factor
  fc(r) by a C2 switch.  The switch is 0 for r <= r_off and 1 for
  r >= r_on (quintic between: `_fcut` in the numba kernel).  It acts on
  the G2 pair, on r_ij and r_ik of G4 / G5, and on r_jk of G4.
- A pair closer than r_off, and every triplet containing it, does not
  enter G.  There the ML part sees an undercoordinated environment and
  stays bounded; the pair baseline alone supplies the repulsion.
- The setting is stored in the descriptor config (absent in older
  configs, meaning none) and set with `mlip.descriptor.inner_cutoff`.
- Rule of use: put r_on at or below the shortest distance in the
  training data, so the fitted range is untouched.  The baseline's
  r_outer should reach r_on, so the baseline covers the blind range.

**Verification.**  tests/test_mlip_physics.py (+2):
- Atom-centred Jacobian forces / stress equal finite differences with
  pairs inside the switch: 8.5e-10 eV/A, 1.3e-11 eV/A^3.
- A dimer below r_off gives G = 0; inside the switch it does not.
- The switch is continuous at both ends.
- The config round-trips.
- An ELM on an inner-cutoff ACSF plus a ZBL baseline is conservative
  across the switch (fourth-order finite differences).

**Evaluation** (same probes; I: inner cutoff 1.6 -> 1.95 A alone;
I+A3+B: with ZBL 1.0 -> 1.95 A and E_atom = 0; I2+A4+B: inner 1.8 ->
2.0 A, ZBL 1.2 -> 2.0 A, E_atom = 0; data start at 2.0 A):

| data | model | condition | E / F / S | push at 1.8 / 1.5 / 1.2 / 1.0 A [eV] | expanded 1.7 a0 |
|---|---|---|---|---|---|
| Si | reference | | | 3.8 / 12.7 / 38.6 / 85.6 | 0.000 |
| Si | ELM | none | 1.04 / 1.64 / 1.05 | 11 / 286 / 555 / 379 | -1.328 |
| Si | ELM | I | 1.04 / 1.64 / 1.05 | (bounded, 5.2 at 1.2 A) | -1.328 |
| Si | ELM | I2+A4+B | 1.06 / 1.67 / 1.08 | monotonic, 29.8 at 1.2 A | 0.000 |
| Si | QELM | I2+A4+B | 1.06 / 1.68 / 1.08 | monotonic, 29.9 at 1.2 A | 0.000 |
| Si | NN | I+A3+B | 2.44 / 3.06 / 1.81 | monotonic, 28.9 at 1.2 A | 0.000 |
| Tersoff | reference | | | 4.1 / 16.2 / 52.8 / 91.9 | 0.000 |
| Tersoff | ELM | none | 2.65 / 6.86 / 2.24 | 47 / 848 / 744 / 311 | 1.071 |
| Tersoff | ELM | I2+A4+B | 2.65 / 6.85 / 2.22 | 29.4 at 1.2 A; one 1.1 eV dip at 1.85 -> 1.80 A | 0.000 |
| Tersoff | QELM | I2+A4+B | 2.59 / 6.96 / 2.19 | 29.5 at 1.2 A | 0.000 |
| Tersoff | NN | I+A3+B | 8.08 / 8.82 / 2.52 | monotonic, 31.4 at 1.2 A | 0.000 |

**Findings.**
- **The runaway is gone.**  The inner cutoff alone bounds the ELM /
  QELM under compression: from +555 eV (Si) and +744 / -390 eV (Tersoff
  ELM / QELM) at 1.2 A down to 5.2 and 3.4 eV.
- **With the ZBL baseline and the isolated-atom limit** all models give
  a repulsive curve: 28-31 eV at 1.2 A, against 38.6 (SW) and 52.8
  (Tersoff).
- **Accuracy inside the data is unchanged** (ELM / QELM F within 2 %),
  because r_on sits at the data's edge.
- **Tersoff ELM / QELM keep one small non-monotonic step** (1.1 eV
  between 1.85 and 1.80 A).  The ML part still extrapolates a little
  above the reference just outside the data (4.7 against 2.4 eV at
  1.9 A) before the switch removes it.
- **The resulting curves are softer than the references at 1.0-1.2 A.**
  - SW Si: 29.8 against 38.6 eV at 1.2 A (-23 %), 57 against 86 eV at
    1.0 A (-33 %).
  - Tersoff: 29.4 against 52.8 eV at 1.2 A (-44 %), 46 against 92 eV at
    1.0 A (-50 %).
  - ZBL is the universal nuclear-collision repulsion, not a fit to Si.
    A Born-Mayer term fitted to short-range reference data would match
    better when such data exist.

## Follow-up study: remedies for the remaining short-range issues (2026-09-29)

The two remaining issues -- ZBL too soft at 1.0-1.5 A, and the 1.1 eV
dip in the Tersoff switch -- have one cause: no training data closer
than 2.0 A.  Three remedies were compared (example/mlip/adr0014/
short_range.py and short_range.md).

**Short-range (SR) data.**  From training snapshots, one atom is pushed
towards its nearest neighbour and labelled by the reference potential
(with the evaluation's noise).  In practice these are a few dozen DFT
single points.  Held out: pushes from test snapshots.

| remedy | what it does | result |
|---|---|---|
| (a) current | inner cutoff 1.8 -> 2.0 A + ZBL 1.2 -> 2.0 A | as above: soft, Tersoff dip 1.1 eV |
| (b) fitted Born-Mayer | same inner cutoff; A exp(-r / rho), switched 1.8 -> 2.0 A, least-squares fitted to the SR residual energies E_ref - E_ML of the trained model | **best** (table below) |
| (c) SR data in training | SR data added to the fit, inner cutoff lowered to 1.1 -> 1.25 A, ZBL below | SR configurations reproduced (0.3 meV/atom on Si), but the model runs away again below the SR data (16-22 eV dips), and Tersoff test accuracy collapses (F 6.85 -> 31.7 sigma) |
| (d) b + c | a BM under the SR-trained model | degenerate: the residual is already ~0, the fit does not move |

(b) with SR data at different distances.  The ELM is unchanged; the BM
term is 0 at the data (>= 2.0 A), so test accuracy is untouched (Si
F 1.67 sigma, Tersoff 6.85 sigma, as in (a)):

| data | SR distances [A] | A [eV] / rho [A] | push 1.8 / 1.5 / 1.2 / 1.0 A [eV] | reference | largest dip |
|---|---|---|---|---|---|
| SW Si | (ZBL, no SR data) | | 3.3 / 10.4 / 29.8 / 57.3 | 3.8 / 12.7 / 38.6 / 85.6 | 0 |
| SW Si | 1.3-1.9 | 10491 / 0.211 | 5.0 / 12.2 / 40.8 / 99.0 | | 0 |
| SW Si | 1.1-1.9 | 5830 / 0.232 | 5.4 / 12.6 / 38.1 / 85.3 | | 0 |
| Tersoff | (ZBL, no SR data) | | 4.1 / 11.6 / 29.4 / 46.1 | 4.1 / 16.2 / 52.8 / 91.9 | 1.14 eV |
| Tersoff | 1.3-1.9 | 32740 / 0.190 | 6.1 / 16.6 / 62.7 / 162.8 | | 0.61 eV |
| Tersoff | 1.1-1.9 | 6785 / 0.242 | 7.6 / 18.4 / 51.9 / 103.9 | | 0.29 eV |
| Tersoff | 1.0-1.8 | 4616 / 0.261 | 8.4 / 19.5 / 51.2 / 95.7 | | 0.14 eV |

**Findings.**
- **A Born-Mayer term fitted to SR residuals fixes the softness** where
  the SR data reach: Si within 1.5 % at 1.0-1.5 A, Tersoff within 2 % at
  1.2 A.  It shrinks the dip by 4-8x and leaves the fitted range alone.
- **It extrapolates only as far as the SR data go.**  Fitted on 1.3-1.9
  A it overshoots at 1.0 A on Tersoff (163 against 92 eV); with data to
  1.1 A it is within 13 %.
- **One exponential cannot match both ends.**  It is too stiff
  at 1.8 A, the edge of the switch (Si 5.4 against 3.8 eV, Tersoff 7.6
  against 4.1 eV; 40-105 % too stiff).
- **Teaching the ML the short range (c) moves the problem inward** and
  hurts the fit on the harder data.  The baseline is the better place
  for the short range.

**Implemented (user, 2026-09-29).**
- `fitting.physics.fit_born_mayer(structures, energy_fn, species,
  r_inner, r_outer)`: one (A, rho) per species pair found within r_outer
  in the short-range (SR) structures.  Bounded least squares on the
  residual energies E_ref - energy_fn (the trained model with every
  fixed baseline term).
- CLI: a baseline term `{type: born_mayer, r_inner, r_outer, fit_to:
  sr.csv}` (or `fit_to: {training_csv, poscar_dir}`), with SR data in the
  `fit` layout.  After training, the term is fitted and appended to the
  fixed terms in physics_config; `mlip_summary.json` records
  `baseline_fit` (pairs, rms before / after, refit).
- If the fitted term is non-zero on any training structure (> 1e-6 eV),
  the model is refitted once on labels minus the full baseline, so the
  training labels and the stored baseline stay consistent.
- ZBL remains the choice without SR data.

**Verification** (tests/test_mlip_physics.py, +3):
- Synthetic SR data from known Si-Si and Si-Ge terms: both pairs are
  recovered (rel 1e-6), residual < 1e-6 eV.
- CLI, atom-centred and tabular ELM, with SW Si pushes as SR data: the
  residual rms falls by more than half; the stored term has the fitted
  Si-Si pair.
- The refit happens exactly when r_outer reaches the training data
  (2.3 A: yes; 2.0 A: no).
- The engine with the fitted term is conservative (fourth-order finite
  differences).

## Attractive and oscillating terms (study, not implemented)

- **Attraction that vanishes at long range** (e.g. -C6 / r^6 with damping,
  a Morse well).
  - Inside the ML cutoff it is redundant: the model learns the
    attraction from the data, and a fixed one only has to be learned
    away where it is wrong.
  - Its physical value is beyond the cutoff: dispersion (vdW) or
    electrostatics, the same Delta-learning pattern as the Coulomb plan
    (ADR-0010 item 1).
  - Worth adding as a baseline term type (`dispersion`) when vdW or
    layered systems are in scope.  Not needed for covalent Si / GeSn.
- **Wells at the 1st, 2nd, 3rd ... neighbour distances, with small
  barriers between** (an oscillating pair function).
  - The model already contains such a function.  The direct-link weights
    on the G2 channels form a pair potential sum_k gamma_k exp(-eta
    (r - r_s,k)^2) fc(r), learned with any sign pattern along r.
  - A fixed one would need the shell distances of one crystal, a
    phase-specific prior that hurts transfer to liquids, surfaces, other
    polymorphs and strained cells.
  - Not recommended as a fixed term.  If wanted, it can be read off (and
    plotted) from the fitted gamma as a diagnostic.
