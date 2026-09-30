---
status: accepted (implemented and verified 2026-09-30)
---

# Multi-species classical fits: Tersoff (1989 mixing and records) and Zhou EAM

Requested by the user on 2026-09-29: every classical potential other than
SW that could not be fitted for more than one species is to be made
fittable, and checked on real crystal structures.  Found by reading the
code: the pair potentials (LJ, Mie, Morse, Born-Mayer, Yukawa,
Buckingham) were already multi-species (one record per "A-B" pair; T8
passed on Cu3Au and ZnO).  Tersoff's engine read the multi-species (i, j,
k) table but `fit` took only the flat one-element form; EAM had no fit
and no YAML route at all (registry S19).

## Decisions (agreed with the user, 2026-09-29, Q1-Q16)

1. **Scope (Q1):** Tersoff multi-species fit, EAM fit, and the
   multi-species Tersoff forces made analytic.  Pair potentials unchanged
   (no mixing-rule option).
2. **Tersoff, two forms (Q2, Q6, Q7).**
   - *1989 mixing* (default for a new multi-species fit): Tersoff, Phys.
     Rev. B 39, 5566 (1989), eqs. (1a)-(1e).  Per element A, B, lambda,
     mu, beta, n, c, d, h (fitted) and R, S (fixed); chi per pair of
     distinct elements (fitted); omega = 1.  `fit_keys: [chi]` is the
     paper's own procedure.
   - *Records*: the LAMMPS (i, j, k) table.  Pair keys A, B, lambda1,
     lambda2 (and R, D) are one variable per unordered pair ((i, j, j) and
     (j, i, i) must agree; the engine reads the lower-indexed atom's);
     beta, n per diagonal record; c, d, h per record, or per centre with
     `tie: centre`; R, D, m, gamma, lambda3 fixed.
   - YAML `initial_tersoff` / `potential.params`: flat dict (one element,
     unchanged), record list, or `{mixing: tersoff1989, elements, chi}`.
3. **EAM: Zhou, Johnson & Wadley (Q3, Q8-Q10).**  Phys. Rev. B 69, 144113
   (2004), eqs. (A3)-(A8).  Cross pair by eq. (A5), optionally scaled by
   chi_ab (not in the paper).  Continuity of F and dF/drho at rho_n and
   rho_0 is imposed exactly: Fn0, Fn1, F0, F_e follow from Fn2, Fn3, F2,
   F3, eta (F1 = 0); per element A, B, alpha, beta, Fn2, Fn3, F2, F3, eta
   are fitted, r_e, f_e, rho_e, rho_s, kappa, lambda fixed.  Cutoff: the
   distance beyond which |phi_aa| / max(A, B) and f / f_e stay below 1e-6,
   with a C2 quintic switch over its last 0.5 A.  setfl files are written
   (after every EAM fit, `fitted.eam.alloy`) and read (`potential: {type:
   EAM, setfl: file}`), besides `{type: EAM, form: zhou2004, elements}`.
4. **Verification (Q4, Q5, Q11, Q13-Q15).**  Data labelled by a published
   potential on real crystal structures, noise-free; each fitted variable
   started within +-10 %; Nelder-Mead, no DE.  Judged on properties of the
   reference potential itself (ADR-0003 decision 3), tolerances fixed
   before any run and never loosened: V0, B +-0.5 %; C11 / C12 / C44
   (C13, C33) +-2 %; formation / phase energies +-5 meV/atom; a relaxed
   defect energy +-1 %.  Parameters for information only.
   - SiC (1989 form): trained on 3C-SiC, diamond Si and C (+ antisite
     cells); unseen rocksalt SiC and the V_Si + V_C pair.
   - AlN (records, Karaaslan et al., Phys. Rev. Applied 13, 034027
     (2020)): trained on wurtzite; unseen zinc blende and rocksalt.
   - Ni-Al (Zhou EAM): trained on fcc Ni, fcc Al, L1_2 Ni3Al; unseen B2
     NiAl and L1_2 Al3Ni.
5. **Recording (Q12, Q16):** this ADR; version 0.2.0a4 (Tersoff forces
   change at the 1e-7 eV/A level); one commit per stage.

## Stage 1: analytic Tersoff forces (commit af4b9b1)

Every Tersoff force kernel differentiated the zeta term by central
differences (step 1e-6 A).  `_te_zterm_grad` gives the gradient in closed
form.  On a rattled, sheared 64-atom cell, forces against fourth-order
differences of the energy: 2.0e-7 -> 1.0e-9 eV/A (one species), 1.0e-7
-> 7.7e-11 (AlN, two species); stress 9e-10 -> 4e-11 eV/A^3.  Energies
unchanged.

## Published tables and two misprints

Both tables are transcribed in full (akitsushima/engine/
published_parameters.py): Tersoff (1989) Table I, 3 elements x 11 + 2
chi; Zhou (2004) Table III, 16 x 20 = 320 values.  Two printed values
are wrong, each shown by the paper's own numbers:

- **Tersoff (1989), carbon d.**  Printed 4.384.  With it diamond C gives
  7.422 eV/atom and 3C-SiC 6.187, against the paper's 7.37 and 6.165.
  With 4.3484 (Tersoff's 1988 carbon value, as carried by this package's
  Tersoff template; the 1988 paper was not re-read): 7.3705 eV/atom at
  3.5656 A, and 3C-SiC 6.1647 at 4.3211 A (paper 4.32).  Silicon, not
  affected, gives 4.6296 eV/atom at 5.4320 A (paper 4.63).
- **Zhou (2004), Al A.**  Printed 0.134873.  The paper defines r_e as the
  equilibrium nearest-neighbour spacing; every other fcc / bcc row gives
  a0 = sqrt(2) r_e (fcc) or 2 r_e / sqrt(3) (bcc) to 4e-6 relative, with
  the standard cohesive energies (Cu 3.54, Ni 4.45, Fe 4.29 eV ...).
  With 0.134873 fcc Al has no minimum there (a0 3.44 A against 4.0502,
  5.42 eV/atom).  With 0.314873 -- recalled from the parameter file of
  Zhou's LAMMPS generator, UNVERIFIED as a source -- a0 = 4.05019 A =
  sqrt(2) r_e, E_coh 3.58 eV, B 75.9 GPa.  The self-consistency is the
  evidence, not the recollection.

Continuity of Zhou's embedding function as printed (re-deriving Fn0,
Fn1, F0, F_e from the other coefficients): Ni, Al, Cu, Ag, Au, Pd, Pb, Fe
agree with the printed values to <= 3e-5 (F0 is printed to 2 decimals);
Mo, Ta to 3e-4; Mg, Ti, Zr (kappa 0.5, lambda 1) to 1-2e-3; **Pt's
printed Fn0 / Fn1 leave a 2.6 eV step at rho_n, and W's negative eta
cannot meet the slope condition at rho_0 with the printed F_e (0.9 eV
apart).**  These rows are flagged (ZHOU_2004_CONTINUITY_ISSUES); the
engine enforces continuity, so Pt and W differ from the printed table.

## Step 0: the engine against Tersoff (1989) Table II

3C-SiC from Table I (carbon d corrected), 64-atom cell, relaxed ions
for C44 (example/fit/SiC_tersoff/step0_*.py):

| quantity | paper | engine | verdict |
|---|---|---|---|
| a [A] | 4.32 | 4.3212 | PASS |
| E_coh [eV/atom] | 6.165 | 6.1636 +/- 0.0058 (Table I rounding propagated) | PASS |
| B [GPa] | 220 (2.2 Mbar) | 224.1 | PASS (printed precision) |
| C11 [GPa] | 420 | 436.5 | **differs (+3.9 %)** |
| C12 [GPa] | 120 | 118.0 | PASS |
| C44 [GPa] (relaxed ions) | 260 | 256.5 (clamped: 310.7) | PASS |
| rocksalt a [A] | 4.25 | 4.2390 +/- 0.0001 | **differs (-0.26 %)** |
| E(rocksalt) - E(3C) [eV/atom] | 1.5 | 1.49 | PASS |

C11 and the rocksalt lattice constant are not explained by the rounding
of Table I.  In rocksalt the Si-Si distance a/sqrt(2) = 3.00 A sits at
S_Si = 3.0 A, so it is cutoff-sensitive.  Later recomputations of this
potential are recalled as C11 / C12 / C44 = 437 / 118 / 256 GPa (C44
unrelaxed 311), matching the engine to the digit -- UNVERIFIED (from
memory of a comparison table, e.g. Erhart & Albe, Phys. Rev. B 71,
035211 (2005)); the paper that would settle it has not been read.  The
fit tests below do not depend on this: their reference is the potential
itself.

## Zhou EAM engine checks

Ni-Al, r_c = 6.690 A (set by Al): forces and stress against fourth-order
differences 1.8e-10 eV/A and 8.7e-11 eV/A^3; the cutoff switch changes
lattice energies by at most 8.8e-6 eV/atom (fcc Al) against a cutoff 3 A
longer; setfl round trip (5000-point grids, cubic splines) 5e-13
eV/atom, forces <= 2.5e-6 eV/A, stress 1.3e-8 eV/A^3.

## Verification results

Judge scripts: example/fit/{SiC_tersoff,AlN_tersoff,NiAl_eam}/check_fit.py
(criteria in example/fit/judge_common.py, fixed before any run).  Every
row below is the full judgement of one fit; "unseen" phases were never in
the training data.

### Tersoff

| run | training data | optimiser | final loss | failed rows |
|---|---|---|---|---|
| AlN, beta and n fitted | wurtzite | Nelder-Mead | 3.3e-4 | 10 of 19: wurtzite B +1.09 %, C33 +2.18 %; rocksalt (unseen) V0 +4.6 %, E - E(wz) +214 meV |
| AlN, beta and n fixed (Q17) | wurtzite | Nelder-Mead | 1.6e-5 | 2 of 19: rocksalt (unseen) C44 -8.3 %, E - E(wz) -16 meV |
| SiC, beta and n fixed | 3C, diamond Si and C | Nelder-Mead | 2.1e-3 | 7 of 23: diamond C C12 -4.2 %; rocksalt (unseen) B -2.4 %, E - E(3C) +55 meV; V_Si + V_C (unseen) -9.2 % |
| AlN (Q19b) | + rocksalt | Nelder-Mead | 1.2e-5 | 2 of 23: rocksalt C44 -2.4 %; honeycomb (unseen) E +5.9 meV |
| SiC (Q19b) | + rocksalt | Nelder-Mead | 3.6e-4 | 2 of 27: honeycomb (unseen) C12 -19 % (of -5.4 GPa), E +5.5 meV |
| AlN | + rocksalt | NM, restarts (same simplex) | 6.5e-6 | 1: honeycomb E +5.06 meV |
| SiC | + rocksalt | NM, restarts (same simplex) | 2.9e-5 | 3: rocksalt B +0.68 %; honeycomb C12 +2.0 %, E -6.2 meV |
| AlN | + rocksalt | NM, restarts (rotated simplex) | 1.3e-7 | **none (23 of 23)** |
| SiC | + rocksalt | NM, restarts (rotated simplex) | 3.5e-15 | **none (27 of 27)** |
| AlN | + rocksalt | least_squares | 2.4e-7 | **none** |
| SiC | + rocksalt | least_squares | 3.5e-15 | **none** |

The honeycomb sheet (planar, 120 deg angles only; judged on area per atom,
in-plane C11 / C12 and E - E(ground)) replaced the rocksalt phase as the
unseen test once rocksalt was trained.  CsCl was considered and dropped:
its minimum bond (SiC 2.29 A, AlN 2.44 A) lies inside the cutoff switch, so
it would test the fixed cutoff function, not the fit.

Diagnosis (AlN, wurtzite only): restarting from the fitted point hardly
moved the loss (3.3e-4 -> 7e-5); from a start 1 % off the reference,
beta moved by -93 % / +3384 % and d by -50 % / +177 % while the loss fell
to 4e-6; with rocksalt added the pair terms returned to 0.5 % but beta, c,
d did not.  With near-tetrahedral angles g(theta) depends on c and d only
through c / d^2 (d >> |h - cos theta|), and beta rescales zeta: the
degeneracy of ADR-0003 FT1.  Fixing beta, n (Q17) and training on a second
coordination (Q19b) made the parameters identifiable.

Two caveats.  beta and n are fixed at the reference values, which favours
the test; in real use they are fixed at the published values of the
starting potential.  A relative tolerance on a quantity near zero (the
honeycomb C12 of SiC, -5.4 GPa beside C11 = 177 GPa) is strict; it was
kept.  The AlN least_squares run stopped at 2.4e-7 with the final
Jacobian (a forward difference, step 1e-7 max(1, |x|)), 2e-15 with
scipy's relative step before that step was replaced (below); both pass.

### Zhou EAM, Ni-Al

| run | training data | optimiser | final loss | failed rows |
|---|---|---|---|---|
| 1 | fcc Ni, fcc Al, L1_2 Ni3Al (15 cells) | NM, 20 restarts, local_maxiter 1e4 | 5.2e-4 | 9 of 28: B / C11 / C12 of Ni3Al (+2.7 %), B2 NiAl (unseen, -19 %), Al3Ni (unseen, +7.9 %) |
| 2 | + expanded (x1.06, 1.10) and compressed (x0.98) cells (24) | NM, restarts, 8.8 h limit | 2.7e-4 | 1 of 28: B2 NiAl B -0.67 % |
| 3 | as 2 | least_squares | **7.5e-14** (about 3 min) | **none (28 of 28)**; every parameter to the printed digits |
| 4 | as 1 | least_squares | 9.3e-3 after 27 000 evaluations, stopped | (not judged) |

Formation energies and C44 passed in every run.  Run 1's failures are all
volume-type moduli, which the curvature of F sets.  The Ni of B2 NiAl sits
at 0.85 rho_e and of Al3Ni at 0.79 rho_e, in the low-density branch
(rho < 0.85 rho_e) that only 3 %-expanded Ni3Al reached in the training
data; its Fn2, Fn3 came out at -380 % / +317 % (Ni).

**Sensitivity of the loss** (example/NiAl_eam/sensitivity.py): H = 2 J^T J
from the residual Jacobian at the reference (the exact Hessian there; a
first attempt differencing the scalar loss with step 1e-3 gave a gradient
of 3e2 at the minimum and negative eigenvalues, third-order terms of the
stiff directions).  Change of one parameter alone that raises the loss to
1e-3 (15-cell data): Al F2 1.2e-6, Ni eta / Ni F2 / Al F3 / Al eta 4-8e-6
(absolute); A, B 0.001-0.003 %; alpha, beta 0.004-0.01 %; but Ni Fn2 0.04,
Ni Fn3 0.55, Al Fn2 5.2, Al Fn3 283.  Eigenvalues 1.5e9 down to ~0
(singular, Al Fn3); with the 24-cell data 2.1e9 to 1.3e-2, condition
number 1.6e11 (the Fn diagonal rose by 1e3-1e9).  The second-stiffest
direction is Ni eta - Ni F2 (+0.67 / -0.67): a diagonal valley.  Run 1's
result lay 3e-5 off the reference along stiff directions and ~7 along the
soft Fn3 ones; the quadratic model predicts loss 0.15 there against 5.2e-4
measured, i.e. a strongly curved (Rosenbrock-like) valley.  Stiff and soft
directions together (not stiffness alone) are what defeat a derivative-
free simplex; a Gauss-Newton method uses the curvature and rescales the
variables by the Jacobian (run 3).  Run 4 shows the converse: with the
low-density branch undetermined by the data, least_squares is slow too --
the data, not only the optimiser, must fix the parameters.

## Optimiser additions (Q21 and the restart work)

- `restarts: N`, `restart_rtol` (default 1e-3), `restart_seed`: further
  local runs from the best point.  A Nelder-Mead restart from scipy's own
  axis-aligned simplex retraces the previous run exactly (SiC stuck at
  2.9e-5; the same point, perturbed only in its last digits by a JSON round
  trip, reached 3.5e-15), so each restart starts from a randomly rotated
  simplex of scipy's step sizes.  It stops after two consecutive restarts
  below restart_rtol.
- `local_maxiter`: caps each local run.
- `time_limit_hours`: wall-clock budget; the best point evaluated so far is
  kept and saved.  Measured with time.time(): time.monotonic stops during
  macOS sleep, which let an 8.8 h run go on through 3.7 h of low-power
  sleep.  Under MPI the decision is reduced over ranks.
- `local_method: least_squares`: scipy.optimize.least_squares, method
  `dogbox` (bounds honoured), on `loss.generic_residuals` (the residual
  vector, sum of squares = the loss, rank-order MPI allgather), with
  `x_scale='jac'` and the driver's own forward-difference Jacobian (step
  1e-7 max(1, |x_i|); scipy's relative step 1e-7 |x_i| is zero at x_i = 0,
  where a chi = 1 fit stopped on gtol at the wrong point).  On an 8-
  variable Ni3Al test: dogbox 5 iterations / 1 s to 2e-25; trf 177
  iterations / 50 s to only 1e-10; lm (no bounds) 2e-32.  The multi-species
  `--write-config` templates now use it.

## Consequences

- `fit --potential tersoff` fits any number of elements in either form;
  `fit --potential eam` fits the Zhou form and writes a setfl table; YAML
  takes both (`mixing: tersoff1989`, `type: EAM` with `form: zhou2004` or
  `setfl:`).
- Multi-species Tersoff forces change by ~1e-7 eV/A (analytic zeta
  gradient); version 0.2.0a4.
- Guidance for real data: train on more than one coordination (Tersoff) and
  on expanded and compressed cells (EAM), and use `least_squares`.
- Out of scope, noticed: under MPI every rank prints the fit log (and
  prterun reports an abnormal exit after a normal finish).
