---
status: accepted
---

# Verification plan for every CLI command

Every one of the eleven CLI commands (`relax`, `md`, `gcmc`, `neb`, `tsa`,
`osmc`, `esm`, `phonon`, `kappa`, `mlip`, `fit`) already has an in-process
smoke test in `tests/test_cli_smoke.py`, so every one of them "runs". That is
not the standard this project holds itself to (ADR-0002, decision 4), and
before this plan only two commands had been judged against a known answer:
`relax` (Si, Stillinger-Weber, all six optimisers, `example/Si/check_relaxed.py`)
and `fit` (single- and two-species SW, `example/fit/*/check_fit.py`). This ADR
records what "verified" means, what is still unverified, and the structure,
potential, conditions, judged quantity and tolerance for each outstanding test.

## Decisions

1. **"Verified" means judged against a known answer by a script with a stated
   tolerance.** Completing without a crash (the smoke level) or looking
   plausible by eye does not count. The model is `example/Si/check_relaxed.py`.
2. **Two layers.** Heavy physics runs live in `example/<system>/<command>/`
   together with a `check_*.py` judge (`example/` is gitignored and run by
   hand, with `nohup` and a log file for anything over ~5 minutes). Each one
   gets a shrunken, seconds-long regression counterpart in `tests/`, so that
   the PostToolUse pytest hook stays fast and regressions are still caught
   under version control.
3. **Reference values come from the potential itself.** Exact or analytic
   results of the same potential (its own E(V) minimum, the ideal-gas limit,
   an image-charge series, an independent phonopy calculation with the same
   force function), or literature values computed *with the same potential
   and settings*. Experimental values measure the potential, not the CLI,
   and are not used as pass criteria.
4. **Coverage is every major branch within a command, not every
   potential x command combination.** Per-potential correctness of energy,
   forces and stress is already owned by the finite-difference oracle in
   `tests/test_engine_forces.py`; re-testing it through every command would
   duplicate that without discriminating anything new.
5. **Existing `example/` outputs are judged, not rerun, where their
   timestamps postdate every fix that could change them.** Checked on
   2026-09-24 against the commit history: every AlN / Cu3Au / GeSn / ZnO `md`
   output postdates the Tersoff halving fix (`db6fc31`), the production-window
   drift fix (`31185dd`) and the Born-Mayer/Buckingham force shift
   (`b259879`). The GeSn SW parameter set passes the (i,j,j)/(j,i,i)
   consistency check added in 0.2.0a2, so its results are unaffected by that
   release. The one exception is `example/ZnO/relax/`, whose outputs carry the
   same minute (09-22 08:40) as `b259879` and cannot be placed on either side
   of it, so it is **rerun**.
6. **Execution order** (cheap and foundational first, with `mlip` moved up at
   the user's request to get an early answer on it):
   1. `tests/` layer: all templates, MPI rank-count agreement
   2. `relax` / `md`: judge existing outputs (rerun ZnO relax)
   3. `mlip`: synthetic data
   4. `phonon`
   5. `neb` / `tsa`
   6. `esm`
   7. `gcmc` / `osmc`
   8. `fit`: the five remaining potential types
   9. `kappa`
7. **Real DFT data is out of scope here.** The VASP outputs in
   `example/fit/GeSnVac_SW/GeSn_V01/` (61 structures) will be used for a
   separate verification when the user asks for it; physically meaningful
   `mlip` descriptors will be designed at that point, not now.

## Test plan

Tolerances marked **UNCERTAIN** are first estimates (under ~90% confidence)
to be revisited against the first real data, not silently loosened.

### T1 `tests/` layer (all commands)

- **Templates.** For each of the 13 `--dump-default` targets (`relax`, `md`,
  `nvt`, `nve`, `npt`, `gcmc`, `neb`, `tsa`, `osmc`, `esm`, `phonon`, `kappa`,
  `mlip`), substitute only the input files (POSCAR / CSV / JSON) and shrink the
  step counts, then assert that the command completes and writes its outputs.
  This extends the partial key checks already in `test_cli_smoke.py`
  (`test_relax_template_keys_are_all_read_by_cmd_relax` and others).
- **MPI.** For `relax`, `md`, `gcmc`, `osmc`: 1, 2 and 4 ranks under `mpirun`
  must agree. Follows the existing `tests/test_mpi.py` pattern.
  *Revised on implementation* (`tests/test_mpi_cli.py`): the planned 1e-10
  cannot hold for trajectories. Allreduce summation-order noise flips the last
  printed digit of an NVT log (1.879160 vs 1.879159 eV), and 50 NPT steps
  amplify it to ~1e-8 A in positions. Logs are therefore compared token by
  token to 1e-5 absolute (non-numeric tokens exactly) and structure files to
  1e-6 A. Either is far below what a rank that double-counts or drops its
  share would produce. Repeat runs at a fixed rank count are bit-identical.

### T2 `relax` and `md` (existing examples)

Scope: 7 relax runs (AlN Tersoff; Cu3Au LJ / Mie / Morse; GeSn SW; ZnO
Born-Mayer / Buckingham) and 42 md runs (the same systems x NVT 300/600/900 K,
NVE, NPT 0/5 GPa). Si relax is already verified and excluded.

| ensemble | judged quantity | pass criterion |
|---|---|---|
| relax | final max force and stress; lattice (a; a, c, u for wurtzite) | below the run's convergence thresholds; lattice within 1e-4 relative of an independent `scipy` minimisation of the same potential on the unit cell |
| NVT | mean T; kinetic-energy fluctuation | mean T within 3 sigma (block-averaged standard error) of target; Var(K) within +/-20% of the canonical 2<K>^2/(3N) |
| NVE | conserved-quantity drift over production | < 1 meV/atom/ns (the bar already used in `31185dd`) |
| NPT | mean P; volume fluctuation | mean P within 3 sigma of target; B = kT<V>/<dV^2> within **+/-25% (UNCERTAIN)** of B from a Birch-Murnaghan fit to the 0 K E(V) curve |
| NPT 5 GPa | V(5 GPa)/V(0 GPa) | informational: compared with the same E(V) fit, allowing for thermal expansion |

The NPT fluctuation test is the one that checks MTTK samples the
isothermal-isobaric distribution rather than only controlling its average.

### T3 `mlip` (synthetic data)

`mlip` only accepts a CSV of pre-computed descriptors; the package has no
descriptor generator. So the test uses synthetic regression data: descriptors
X drawn at random, targets from a known function y = f(X) plus Gaussian noise
of known sigma, laid out in the E / 3N x F / 6 x S row structure
`load_mlip_csv` expects, with a separate `test_csv` split by structure (never
by row).

- Models: all four (`elm`, `qelm`, `nn`, `lgbm`).
- Pass: training, saving and test evaluation complete; test-set RMSE <= 1.5 x
  the known noise sigma for each of E, F and S.
- This checks the CLI wiring and learner health, not physics. A fitted model
  cannot be driven by `relax` / `md` from YAML (the path exists only through
  the Python API); that is recorded as out of scope, not as a defect.

### T4 `phonon`

Oracle: phonopy 4.1.0 (already installed) using the same akitsushima force
function.

| system | why |
|---|---|
| Si, Tersoff (template default) | cubic reference case |
| AlN, Tersoff, wurtzite | non-orthogonal cell, 4 atoms per cell |

Pass: frequencies at every q-path point within 0.01 THz of phonopy; DOS in
agreement; F(T) within 0.1 meV/atom; Gamma acoustic modes < 1e-3 THz (acoustic
sum rule). The Debye temperature is definition-dependent, so it is checked by
recomputing it from the same DOS with the code's own stated definition.
LO-TO splitting is out of scope: `solvers/phonon.py` has no non-analytic
correction.

### T5 `neb` and `tsa`

System: single vacancy migration in SW Si. 63 atoms (2x2x2 conventional cell
minus one; perpendicular width 10.86 A > 2 x (3.77 + 0.5) = 8.54 A), plus
215 atoms (3x3x3 minus one) for a size check.

Pass (all internal consistency):
0. (added 2026-09-26) The relaxed initial state is a minimum (no negative
   mode of an independent Hessian beyond the translations) and E_IS = E_FS.
   Relaxing the vacancy from ideal sites keeps its Td symmetry; with SW's own
   three-body term that Td vacancy is a local minimum 1.5-1.6 eV above the
   C3v one, and a NEB between two Td endpoints falls into the lower basin.
   The hop atom now starts 40% of the way along the hop.
1. `neb`: energy profile symmetric about the midpoint to 1 meV; climbing-image
   max force < fmax.
2. `tsa` on the climbing image: exactly one imaginary mode.
3. IRC endpoints match the NEB endpoints (RMSD < 0.05 A) and the barrier
   matches NEB to 1 meV.
4. The TST prefactor matches a Vineyard ratio (product of initial-state over
   saddle frequencies) computed independently with phonopy, to +/-1%.

Literature SW vacancy-migration barriers are **UNCERTAIN** and shown for
information only.

### T6 `esm`

- **(i) Ion pair.** A +/-1 charge pair in vacuum, Buckingham with A = C = 0 so
  only the charges remain. bc2 and bc3 against the analytic infinite
  image-charge series; bc3 with an external field: the potential slope in the
  vacuum region equals the applied field.
- **(ii) Symmetric MgO(001) slab.** bc1 energy against 3D Ewald with a large
  vacuum gap (a symmetric slab has no dipole). No `relax` section: the
  structure is fixed, so the short-range parameters do not enter the
  electrostatics being tested and no MgO parameter source is needed.

### T7 `gcmc` and `osmc`

LJ Ar. Cell perpendicular width > 2 x (10 + 0.5) = 21 A for r_cut = 10 A.

- **(A) Ideal-gas limit** (epsilon effectively 0): <N> = V exp(beta mu) /
  lambda^3 exactly; pass within 3 sigma.
- **(B) Interacting supercritical LJ Ar** (T = 200 K, mu near rho* ~ 0.4):
  `gcmc` and `osmc` at the same mu must agree on <N> within 3 sigma.
- **(C) NIST Standard Reference Simulation Website** LJ-fluid benchmarks
  (mu, rho, P per state point at 3 sigma cutoff, with long-range correction or
  cut-and-shift). State point and truncation of (B) are matched to an SRSW
  entry. The tables are fetched during the execution phase (at most one
  request every few seconds, cached to disk), and all rows and footnotes are
  reported when they are.

The Johnson-Zollweg-Gubbins EOS (Mol. Phys. 78, 591 (1993)) was the first
choice for (C) but the paper could not be obtained; it is **pending** and not
reconstructed from memory.

### T8 `fit` (remaining potential types)

SW is done. For the other five, training data is labelled with a known
parameter set and the fit is judged on **properties, not parameters**
(parameters can be degenerate, notably for Tersoff): E(V) curve, equilibrium
lattice constant and bulk modulus within +/-0.5% of the generating potential.
Parameter recovery is reported for information.

| potential | system (reused from `example/`) |
|---|---|
| LJ, Morse | Cu3Au |
| Born-Mayer, Buckingham (+Coulomb) | ZnO |
| Tersoff | AlN |

### T9 `kappa`

Reference: McGaughey and Kaviany, Phys. Rev. B 69, 094303 (2004), Table I
(GK), with the MD procedure in McGaughey and Kaviany, Int. J. Heat Mass
Transfer 47, 1783 (2004) §3. Both PDFs are in `references/` (gitignored).

**Conditions**, as stated in the papers:
- LJ Ar fcc, sigma = 3.40 A, epsilon = 1.67e-21 J = **0.010423 eV**. The
  template's 0.01032 eV (119.8 K) is ~1% different and is not used.
- Cutoff 2.5 sigma = 8.5 A, **truncated and shifted** (IJHMT §3; the PRB lists
  only the fixed 2.5 sigma cutoff and the dropped pressure correction as
  changes). That is an energy shift only, whose forces are identical to plain
  truncation. akitsushima's `shift: true` is a *force* shift, which changes the
  forces, so the run uses **`shift: false`**. That makes the equations of
  motion identical to the paper's. The only remaining difference is a per-pair
  constant in the per-atom energies of the convective heat-flux term, which is
  expected to be negligible in a crystal. The NVE energy jumps at cutoff
  crossings are recorded against the paper's reported 0.005% conservation.
- NVE production after NVT equilibration (the `kappa` solver's own sequence),
  dt = 4.285 fs, lattice constant per temperature from Table I (zero
  pressure).
- h = 4 (256 atoms, L = 21.6 A at 50 K > 2 x (8.5 + 0.5) = 18 A).
- The paper's procedure: 5e5-step initialisation, 1e6-step production, heat
  flux every 5 steps, correlation length 5e4 steps (214 ps), five independent
  runs with the HCACFs **averaged before integration**.

**Plan:** T = 20, 50, 80 K; five seeds. Benchmark a ~20 ps run first. The
target production is 2 ns per run with tau_max ~ 10 x tau_ac,lg (~80 ps at
20 K), raised towards the paper's 1e6 steps if the benchmark allows, reduced
(length or seeds) if one case exceeds a few hours. HCACFs are averaged across
seeds before integration, as in the paper.

**Pass:** five-seed kappa within **+/-15% (UNCERTAIN)** of the h = 4 value
(paper's +/-5%, our statistical error, and residual truncation differences),
and kappa_xx ~ kappa_yy ~ kappa_zz within statistical error. In addition, the
Slack-Debye term must equal its formula evaluated with the T4 Debye
temperature.

## Out of scope / pending

- JZG LJ EOS: the paper could not be obtained (T7).
- Tersoff Si `kappa`: too expensive locally and literature values are widely
  scattered; a candidate for a later cluster run.
- LO-TO splitting: no non-analytic correction implemented.
- Using a fitted MLIP from `relax` / `md` YAML: no CLI path exists.
- Real DFT data (`example/fit/GeSnVac_SW/`), including physical `mlip`
  descriptors: when the user asks.

## Appendix A: McGaughey and Kaviany, PRB 69, 094303 (2004), Table I

All 16 rows extracted: one T = 0 K row plus 5 temperatures x h = 4, 5, 6.
Extracted as text from the PDF, then cross-checked row by row:
kappa_ac,sh + kappa_ac,lg reproduces kappa_GK to within 0.001 W/m-K in every
row. Ambiguities are flagged:
- In the T = 0 row the lone value "1" is read as c_v/k_B (the classical
  harmonic limit), not as h.
- c_v/k_B is given only for h = 4, except at 50 K, where it is given for all
  three h.

| T (K) | a (A) | h | c_v/k_B | kappa_GK | kappa_ac,sh | kappa_ac,lg | tau_ac,sh (ps) | tau_ac,lg (ps) |
|---|---|---|---|---|---|---|---|---|
| 0 | 5.269 | -- | 1 | -- | -- | -- | -- | -- |
| 20 | 5.315 | 4 | 0.976 | 1.216 | 0.080 | 1.137 | 0.289 | 8.161 |
| 20 | 5.315 | 5 | -- | 1.217 | 0.074 | 1.143 | 0.275 | 7.776 |
| 20 | 5.315 | 6 | -- | 1.218 | 0.069 | 1.149 | 0.263 | 7.617 |
| 35 | 5.355 | 4 | 0.957 | 0.580 | 0.090 | 0.491 | 0.290 | 4.438 |
| 35 | 5.355 | 5 | -- | 0.587 | 0.086 | 0.501 | 0.281 | 4.183 |
| 35 | 5.355 | 6 | -- | 0.593 | 0.088 | 0.505 | 0.286 | 4.279 |
| 50 | 5.401 | 4 | 0.944 | 0.323 | 0.100 | 0.223 | 0.289 | 2.669 |
| 50 | 5.401 | 5 | 0.941 | 0.348 | 0.091 | 0.257 | 0.271 | 2.518 |
| 50 | 5.401 | 6 | 0.942 | 0.336 | 0.089 | 0.246 | 0.266 | 2.400 |
| 65 | 5.455 | 4 | 0.930 | 0.219 | 0.095 | 0.124 | 0.261 | 1.560 |
| 65 | 5.455 | 5 | -- | 0.223 | 0.093 | 0.130 | 0.254 | 1.601 |
| 65 | 5.455 | 6 | -- | 0.236 | 0.098 | 0.138 | 0.264 | 1.826 |
| 80 | 5.527 | 4 | 0.924 | 0.162 | 0.082 | 0.080 | 0.225 | 0.894 |
| 80 | 5.527 | 5 | -- | 0.164 | 0.084 | 0.080 | 0.227 | 0.919 |
| 80 | 5.527 | 6 | -- | 0.177 | 0.099 | 0.078 | 0.247 | 1.313 |

kappa in W/m-K. Mean kappa_GK over h (PRB Table III): 1.217 (20 K), 0.587
(35 K), 0.335 (50 K), 0.226 (65 K), 0.167 (80 K). Averaging the rounded
Table I values gives 0.336 and 0.168 at 50 and 80 K instead, a 0.001
rounding difference; the pass criterion uses the h = 4 row, not this mean.
The paper states a GK
uncertainty of +/-5%. Experimental values quoted in the paper (1.36, 0.65,
0.46, 0.35, 0.30 W/m-K) are for information only (Decision 3).

## Consequences

Tests T2-T9 each produce a judge script and a result that can be re-run
after any change touching that command. The cost is wall time. `kappa` alone
may take hours per temperature, which is why it runs last and starts with a
benchmark. The tolerances marked UNCERTAIN will need a decision once the first
data is in: they may be tightened, but are never loosened just to make a run
pass. As with ADR-0002, a failure is a finding to reproduce and diagnose, not
a tolerance to adjust.

## Appendix B: Execution results (2026-09-24/25)

Run in the order of Decision 6. Judge scripts and their JSON live under
`example/` (gitignored); regression tests added under `tests/`. Fixes applied
follow ADR-0002 Decision 3: a defect that produces silently wrong numbers or a
crash is fixed immediately with a discriminating regression test (each was
checked to FAIL on the pre-fix code). Changes to a method's design, and
anything whose root cause is not yet established, are listed under
"Decisions needed" instead. No version bump has been made; the fixes below
change computed values (phonon, gcmc, kappa), which by ADR-0002 Decision 6
calls for one. Nothing is committed yet. Full suite after all fixes: 407
passed.

### Deviations from the plan, and why

- **T1 MPI tolerance** revised (see T1 above): 1e-5 on logs, 1e-6 A on
  structures.
- **T2 relax 1e-4 criterion** is inconsistent with the examples' own
  thresholds (pmax 0.1 GPa, fmax 0.01 eV/A): the residual lattice error
  equals |sigma_res|/(3B) almost exactly (Cu3Au LJ 7.84e-6 vs 7.83e-6).
  Examples were therefore also rerun at pmax 1e-3 GPa / fmax 1e-4 eV/A.
- **T3 synthetic data** redesigned: the first design (200 structures, 2%
  noise, three-term f) could not be fitted to 1.5 sigma by ELM even on E rows
  alone, so it tested the dataset rather than the code. Final: 1000
  structures, 5% noise, f = tanh(x.w1) + 0.5 x.w2, one shared f for all labels
  (T3a, judged) plus a label-scaled set (T3b, information).
- **T5 criterion 4**: tsa's rate uses Q_TS/Q_IS = 1 by design (tsa.py S11
  note), so it is checked against its own stated formula; the Vineyard rate is
  reported for information.
- **T8 Tersoff** on Si, not AlN: `fit` supports only the single-species flat
  Tersoff form. DE budget reduced to 100 iterations (template 500).
- **T9** production shortened to 1 ns per run (benchmark: ~64 steps/s,
  i.e. ~2 h per 2 ns run x 15 runs).

### Results

| test | verdict | evidence |
|---|---|---|
| T1 templates | PASS 14/14 (after fix F1) | `tests/test_cli_templates.py` |
| T1 MPI | PASS 15/15 | `tests/test_mpi_cli.py`; repeat runs bit-identical |
| T2 relax | PASS at tight thresholds (a, c within 6e-6, u within 6e-6 of an energy-only scipy minimum); default-threshold examples miss 1e-4 exactly by the residual-stress amount | `example/check_relax.py`, ZnO relax rerun reproduced its old output bit-for-bit |
| T2 NVE | PASS 5/5 (drift <= 0.13 meV/atom/ns) | `example/check_md.py` |
| T2 NVT | 11/15 on the 20 ps examples. AlN (3 runs): Var(K) excess decays from 1.22-1.28 to 0.95-1.01 over a 100 ps run and <T> = 600.8 +/- 1.0 K -- insufficient equilibration in the examples, not a thermostat defect. Cu3Au LJ 600 K: <T> 601.3-605.4 K over 4 runs, not a timestep effect (dt 0.5 fs gives 605.4) and not COM drift (sum F = 0 to 1e-11): **unexplained** | block SE underestimates true error by ~5x (seed spread) |
| T2 NPT | (GeSn rows: SW before ADR-0006; the re-run is in ADR-0006 -- 200 ps powered test ratio 1.03) <P> PASS 14/14. B from volume fluctuations: 12/14 within 25%, but per-run SE 15-100% gives the test no power; pooled chi^2 = 4.2 / 14 dof. Powered test (GeSn, 200 ps): B_T = 67.0 +/- 3.4 GPa vs static B(V_T) = 54.5 GPa, ratio 1.23 +/- 0.06: PASS but marginal; the static reference ignores the thermal term, sizeable for this potential (B' = 12) | |
| T3a mlip | ELM 1.23-1.26, QELM 1.23-1.31, NN 0.98-1.04 x sigma: PASS. LightGBM 1.72-1.92: FAIL; raw lightgbm with the same row weights reaches 1.48-1.58, unweighted 1.24-1.33 | `example/mlip/check_mlip.py` |
| T3b mlip (info) | NN 1.0-1.4; ELM E 201x, QELM E 130x (finding M1) | |
| T4 phonon | PASS after fixes P1-P7: frequencies within 1e-3 THz of phonopy on every path point (Si and AlN), F_vib within 0.06 meV/atom, theta_D within 0.1% | `example/phonon/check_phonon.py`, `tests/test_phonon_regressions.py` |
| T5 neb/tsa | PASS all 8 checks (criterion 0 added: endpoint a minimum), 63 and 215 atoms, with the corrected SW (ADR-0006) and endpoints rebuilt 2026-09-26 (C3v vacancy, E_f 2.835 / 2.694 eV). Barrier 0.517 / 0.513 eV; imaginary mode -178.09 vs -178.10 and -171.80 vs -171.81 cm^-1 (independent Hessian); IRC ends within 0.004 A of the endpoints. Literature, for information: 0.43 eV for SW from vacancy vs split-vacancy formation energies (Maroudas & Brown, PRB 47, 15562 (1993), as quoted by El-Mellouhi et al., arXiv cond-mat/0308289, sec. IV.2.1; the original not re-read). Before ADR-0006 (three-body doubled): 0.949 / 0.909 eV | `example/neb/check_neb_tsa.py` |
| T6 esm | FAIL (findings E1-E3) | `example/esm/check_esm.py`; reference Ewald self-tested on the rock-salt Madelung constant (1.747565) |
| T7 gcmc | PASS after fix G1 (5000 cycles, last 80%): ideal gas <N> = 39.57 +/- 0.25 (exact 40), Var(N)/<N> = 0.984; NIST TMMC vapour (T* = 0.90, p/p_sat = 0.80, LFS 2.5 sigma, V = 512 sigma^3) <N> = 42.88 +/- 1.41 vs 41.90 (ideal gas at the same z: 22.5). Before G1: 18.8 | `example/gcmc/check_gcmc.py`; NIST table cached in `references/nist_srsw/vaporEOS_rc25LFS.csv` (all 100 rows x 6 temperatures; the page quotes no uncertainties) |
| T7 osmc | FAIL: <N> = 0.74 +/- 0.63 against 41.90; gcmc vs osmc differ by 42.1 +/- 1.5 (findings O1-O2) | |
| T8 fit | LJ, Morse, Born-Mayer, Buckingham PASS (V0, B0, E0 within 6e-4; Buckingham after fix F5); Tersoff FAIL (finding FT1) | `example/fit/others/check_fit_others.py` |
| T9 kappa | After fixes K1-K3, 5 seeds x 1 ns, HCACFs averaged: 50 K 0.302 +/- 0.036 vs 0.323 (-6.6%) PASS; 80 K 0.141 +/- 0.013 vs 0.162 (-13.3%) PASS; 20 K 0.928 +/- 0.116 vs 1.216 (-23.7%, 2.5 SE) FAIL -- the seed-to-seed SE alone is 12.5% at 20 K (tau_ac,lg = 8 ps against 1 ns runs; the paper used 4.3 ns x 5), so the +/-15% bar is not resolvable there; the running integral at tau_max/2 is 1.13 (-7%). Isotropy PASS at all T. NVE E_tot range <= 0.14% (paper 0.005%; plain truncation steps). Before K2/K3 all five seeds were identical (0.837 / 0.281 / 0.105) | `example/kappa/check_kappa.py`, `run_kappa.py` |

### Defects fixed (with regression tests)

- **F1** `cmd_esm`: any config with a `relax:` section -- including the shipped
  template -- crashed in `update_geometry` (missing `atom_types`).
- **P1** phonon: the Fourier phase used each supercell atom's raw cell offset;
  exact only for q commensurate with the supercell (10 THz errors elsewhere).
  Now minimum-image translations, ties averaged.
- **P2** phonon: `eigvalsh(D.real)` dropped the imaginary part of D(q)
  (1.2-4 THz errors).
- **P3** phonon: `phonon_free_energy` read the cm^-1 DOS axis as THz
  (F_vib 2.35 instead of 0.046 eV/atom for Si at 300 K).
- **P4** phonon: `debye_temperature_from_vsound` read cm^-1 slopes as THz.
- **P5** phonon: reduced -> Cartesian q used the columns of the reciprocal
  matrix (wrong points on hexagonal lattices, band path and DOS mesh).
- **P6** phonon: the acoustic-sum-rule correction ignored mass weighting and
  moved AlN Gamma optical modes by up to 1.8 THz.
- **P7** phonon: v_T was taken from the other directions' longitudinal slopes
  (v_T = v_L; theta_D 685 K instead of 376 K for Tersoff Si).
- **G1** gcmc: deletion acceptance used exp(+beta dU) instead of
  exp(-beta dU); invisible for an ideal gas, it put LJ Ar vapour below the
  ideal-gas density (N = 18.8 vs NIST 41.9).
- **K1** kappa: per-atom kinetic energy in the heat-flux convective term was
  amu A^2/fs^2 used as eV (103.6x too small); the NVE log's T, E_tot and the
  sign of P were wrong too (log showed 0.48 K for a 50 K run).
- **K2** kappa: `seed` never reached the NVT equilibration, which drew its
  velocities from NVTSimulation's default seed -- every seed gave the same
  trajectory (the first 15-run production was 3 runs repeated 5 times; kept
  under `example/kappa/pre_K2K3/`).
- **K3** kappa: the pair virial term of the heat flux had the wrong sign
  (`compute_pairs()` returns r_ij = r_j - r_i). Oracle: J V = d/dt sum_i r_i
  E_i by central differences on an isolated LJ cluster -- 230% off before,
  1e-8 after.
- **F5** fit: `print_fit_result` crashed on any like-species Buckingham pair
  (every oxide), after the optimisation and before saving.

### Findings not fixed (decisions needed)

- **M1** mlip ELM/QELM uncertainty weighting: sigma^2_c multiplies the unit
  weight by r0 or eps0 a second time, and 1/sigma^2 is applied to the residual
  (so 1/sigma^4 on its square). With realistic label scales S outweighs E by
  ~1e17 on the squared residual; the LSQR does not converge (flag 7) and E is
  fitted to 200x its noise. A formula change alters every ELM fit.
  -> Decided in ADR-0004 (chain-rule training and one weighting rule).
- **E1-E3** esm: E1 the Gaussian self-energy is divided by the cell area
  (dimensionally wrong; ~8 eV per unit charge left in); E2 the external field
  enters 14.40x too strongly (e^2/4 pi eps0); E3 after correcting E1 the pair
  energies still differ from the Ewald reference by 0.5-3 eV, growing with the
  dipole length -- the G|| = 0 term is suspected, root cause not established.
  Also, the `relax:` section relaxes on ESM electrostatics alone (no
  Buckingham short range): ions collapse. Forces are consistent with ESM's own
  energy (finite differences, 3e-3 eV/A).
- **O1-O2** osmc: O1 the lambda <= 0 wall sets lambda-dot negative (its comment
  says upward), so lambda sticks at 0 and deletes repeatedly (N -> 0); O2 the
  lambda force -dV/dlambda - mu has no ideal-gas translational term, so even
  with O1 fixed it cannot reproduce <N>(mu) of the muVT ensemble (SUSPECTED,
  design-level).
- **FT1** fit Tersoff: from +10% the fit does not recover the potential (B0 off
  by 4.6-5.1%, beta collapses to its lower bound 0); from the true parameters
  it stays put (loss 2e-11), so the loss is right and the optimisation is
  ill-conditioned (beta ~ 1e-6 beside parameters of order 1-1e5).
- **fit output JSON**: Born-Mayer nests its parameters under `params.bm`, and
  neither Born-Mayer nor Buckingham records `r_cut`, so passing the file to
  `potential_json` silently builds a different potential (r_cut defaults to
  10 A; for Born-Mayer the nested keys are also misread as pairs).
- **ESM input convention**: z must lie in [-Lz/2, Lz/2); a slab centred the
  usual POSCAR way is rejected. Undocumented in the template and CLI.
- **tsa**: the printed "k (TST)" is (kT/h) exp(-Ea/kT); harmonic TST is 24x
  larger for the Si vacancy hop. Not labelled as an approximation in the output.
