---
status: accepted (A and B implemented 2026-09-28; the direct-link extrapolation found here is open, see Open question)
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
- **Full suite:** 600 passed (591 + 9 in tests/test_mlip_physics.py).

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

## Open question (for the user)

How to bound the ELM / QELM direct-link extrapolation while keeping its
accuracy inside the data.  Candidates, not implemented:
1. **Soft input saturation for the direct link only.**  x -> c tanh(x / c)
   per whitened component, with c from the training range.  The energy
   stays smooth, so the forces stay conservative.
2. **Inner cutoff in the descriptor.**  The ML part does not see pairs
   below r_inner (an inner switch on the ACSF pair and triplet terms), so
   the baseline alone acts there.
3. **Accept it and recommend the NN (or direct_link: false with A)** for
   runs that reach close contacts.

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
