---
status: accepted (2026-11-24, 0.3.0a8)
---

# Nose-Hoover chain length (`nhc_chain`, default 3)

Follow-up to the ADR-0025 durability campaign (2026-11-23).  Unit 3 (SW
GeSn, 13608 atoms, NVT 900 K, 20 ps) failed its temperature criterion
with T_mean 774 K.  The cause was investigated on 2026-11-24, and the
remedy was agreed with the user the same day (grilling, three rounds,
Q1-Q15, all as recommended).

## Starting point (0.3.0a7)

### What was wrong

- `NVTSimulation` and `NPTSimulation` used a Nose-Hoover chain of
  exactly two links, held in the scalars `Q1`, `Q2`, `xi1`, `xi2`,
  `vxi1`, `vxi2`.  The masses were the standard ones (Martyna, Klein &
  Tuckerman 1992): Q_1 = n_dof kT tau^2 and Q_2 = kT tau^2.
- A run that starts at a minimum with velocities drawn at T0 drops to
  about T0/2 at once, because equipartition moves half of the kinetic
  energy into the potential energy.  The thermostat must then supply
  n_dof kT0/2.
- With two links, the second link keeps Q_1 vxi_1^2 near kT.  That caps
  |vxi_1| near 1/(tau sqrt(n_dof)), so the heating rate is at most
  dT/dt ~ T/(tau sqrt(n_dof)).  The time to recover 0.95 T0 is about
  ln(1.9) tau sqrt(n_dof): it grows with the square root of the system
  size.

### Investigation (2026-11-24)

The scripts and ledgers are in `example/GeSn/large/`: `nhc_toy.py`,
`nhc_probe.py`, `ase_nhc.py`, `nhc_probe_ledger.jsonl`.

t95 is the first time the 0.5 ps running mean of T_inst reaches 0.95 T0.
All runs below start at the FIRE-relaxed ladder cell (Ge0.875Sn0.125, 1/64
vacancies) unless stated otherwise, with tau = 100 fs and dt = 1 fs.

| Factor varied | Case | t95 (ps) | ln(1.9) tau sqrt(n_dof) (ps) |
|---|---|---|---|
| system size (Tersoff) | 504 / 4032 / 13608 atoms | 3.2 / 8.2 / 14.7 | 2.5 / 7.1 / 13.0 |
| tau_nhc (4032 atoms) | 25 / 100 / 400 fs | 2.3 / 8.2 / 31.5 | 1.8 / 7.1 / 28.2 |
| T0 (4032 atoms) | 300 / 900 K | 8.2 / 8.4 | 7.1 |
| potential (504 atoms) | Tersoff / SW / MLIP | 3.2 / 3.1 / 2.7 | 2.5 |
| other code | ASE 3.28 `NoseHooverChainNVT`, EMT Cu 500 / 2048 atoms, tchain 2 | 3.2 / 6.0 | 2.5 / 5.0 |
| chain (4032 atoms) | plain Nose-Hoover (Q_2 -> infinity) | 0.5 | overshoots to 1.7 T0 |
| | Q_2 = n_dof kT tau^2 | 0.5 | no overshoot |
| | ASE, three links (500 atoms) | 0.8 | |
| thermostat (4032 atoms) | CSVR / Langevin, damping 100 fs | 0.75 / 0.57 | |
| burn-in (4032 atoms) | velocity_rescale_steps 1000 / start at 2 T0 | 0.5 / 0.5 | |
| campaign (ADR-0025) | SW 4032 NPT 300 K / MLIP 4032 NVT 600 K / SW 13608 NVT 900 K | 7.8 / 8.1 / 13.9 | 7.1 / 7.1 / 13.0 |

- Every two-link case lies 1.1-1.3 times above the estimate,
  independently of the potential, the temperature and the code.  An
  independent RK4 integration of the NHC equations on harmonic
  oscillators (`nhc_toy.py`) gives the same.
- Chain length on harmonic oscillators (n_dof 1500 / 12000): three and
  five links heat in 0.8-1.4 ps, four links in 1.4-2.4 ps, two links in
  3.1-8.1 ps.  With an even number of links the top link pins the one
  below it, and the cap moves down the chain; with an odd number the
  first link is free.
- No integrator or unit error was found.  The two-link masses are the
  standard ones, and ASE uses the same masses.
- Neither SW nor the burn-in setting is the cause.  The campaign ran
  with `equil_steps: 0` and `velocity_rescale_steps: 0`, while the
  shipped templates set both to 1000 (NVT) or 2000 (NPT): that is what
  exposed the slow heating.

## Decisions

1. **Scope (Q1).**  Both: unit 3 is measured again with the template's
   burn-in, and the NHC itself is fixed.

2. **M-link chain (Q2).**  A chain of M links (Martyna, Tuckerman &
   Klein, Mol. Phys. 87, 1117 (1996)), with the new setting `nhc_chain`
   (an integer >= 2).  Rejected:
   - Q_2 = n_dof kT tau^2: fast and stable, but outside the standard form
     with no published basis.
   - Start velocities at 2 T0: assumes a harmonic solid and overheats a
     liquid or a disordered start.
   - An automatic burn-in rescale: silently changes the default
     dynamics.

3. **Unit 3 again (Q3).**  `velocity_rescale_steps: 1000`,
   `equil_steps: 2000`, 20 ps production; the +-2 % temperature criterion
   is judged on the production window only.

4. **Anneal (Q4, Q13, Q14).**  md_quench starts every chunk with a fresh
   chain, so a step down in the schedule must be followed by the
   thermostat.  The lag is measured in this ADR on SW GeSn, 4032 atoms
   (geometric 1300 -> 300 K, 6 segments, segment_steps 2000,
   quench_every 500, one seed) for two links, three links and CSVR.  A
   remedy beyond the chain length is decided after the measurement.

5. **Default 3 (Q5).**  `nhc_chain: 2` reproduces 0.3.0a7 bit for bit.
   The rule "the default path stays within 1e-10" (ADR-0043) applies to
   M = 2 here, since changing the default dynamics is the point.

6. **Barostat chain unchanged (Q6).**  It acts on 1 to 6 degrees of
   freedom, where sqrt(n_dof) does not slow anything.

7. **Integrator (Q7).**  The MTTK factorisation of `_nhc_step` extended
   to M links: one sub-step, no Suzuki-Yoshida.  M = 2 must match the
   former body to 1e-12.

8. **State and compatibility (Q8).**
   - Lists `Q_nhc`, `xi_nhc`, `vxi_nhc` of length M.
   - `Q1`, `Q2`, `xi1`, `xi2`, `vxi1`, `vxi2` remain as aliases of the
     first two links.
   - Checkpoints carry the chain as arrays.  A checkpoint written before
     this ADR loads as M = 2.
   - `nhc_chain` joins the restart settings when it is not 2.

9. **Where it is set (Q9).**  The Python API (`NVTSimulation`,
   `NPTSimulation`), the `md` command and the `anneal` md section.  The
   templates explain it.  The thermo.log columns are unchanged.

10. **Gates (Q10).**
    - M = 2 matches 0.3.0a7 to 1e-12 (NVT and NPT).
    - M = 3: t95 <= 1.5 ps for Tersoff 504 / 4032 / 13608 atoms.
    - var(T_inst) within +-10 % of 2 T0^2 / n_dof.
    - H_NHC drift of the order of M = 2.
    - The full suite passes.
    - New tests: M = 3 conservation, restart, and loading an old
      checkpoint.

11. **Process (Q11, Q12, Q15).**
    - ADR-0044, version 0.3.0a8, one worktree, three commits:
      generalisation (M = 2 identical), default 3, CLI and templates.
    - Unit 3 runs at once on 0.3.0a7; the anneal measurement runs after
      the fix.
    - Regression tests with fixed numbers pin `nhc_chain=2`; physics
      tests use the new default.

## Result (2026-11-24, 0.3.0a8)

### Implementation

- `_nhc_step` runs the MTTK half step on a chain of M links: a quarter
  kick of the top link, then down the chain damp by exp(-v_{j+1} dt/8),
  kick by G_j dt/4, damp; scale the velocities by exp(-v_1 dt/2);
  advance xi; then back up the chain reusing the same damping factors.
  For M = 2 the operations are the former ones in the same order.
- `NVTSimulation` and `NPTSimulation` take `nhc_chain` (default 3; an
  integer >= 2, `bool` rejected).  Langevin and CSVR ignore it like
  `tau_nhc`, and their checkpoints carry no chain.
- `md` reads `nhc_chain` (default 3) and prints `NHC chain : M links`;
  `anneal` passes it through its md section (`_MD_KEYS`).  The NVT, NPT
  and anneal templates set `nhc_chain: 3` with a comment.
- Other solvers that build an `NVTSimulation` (thermal conductivity,
  elastic, dynamics analysis) now get three links; their tests pass
  unchanged.  GCMC / OSMC keep their own two-link chain in `gcmc.py`,
  which this ADR does not touch.

### Compatibility

`nhc_ref.py` (SW Si 64 atoms; NVT, NPT iso, NPT aniso; 300 steps)
compares positions, velocities, cell and chain state with references
written by 0.3.0a7.  `nhc_chain=2`: the largest relative difference is
0.0 in all three ensembles, both before and after the default changed.

Tests (`tests/test_nhc_chain.py`, 21 tests; fixture
`tests/data/nhc_legacy_0.3.0a7/`, written by 0.3.0a7 at step 150 of
300):
- two links repeat the former propagator bit for bit (pins
  `nhc_chain=2`);
- a 0.3.0a7 checkpoint restarts as M = 2 and reaches the 0.3.0a7 final
  state within 1e-12 (NVT, NPT); with `nhc_chain=3` it is refused;
- a three-link restart is bit for bit (NVT, NPT iso, NPT aniso), and a
  restart with another chain length is refused;
- H_NHC is conserved for M = 2, 3, 4;
- SW Si 512 atoms: t95 with three links is below half of two links
  (slow);
- `md` and `anneal` pass the setting through.

The full suite: 1174 passed, 1 skipped (0.3.0a8, `NUMBA_NUM_THREADS=2`,
10.6 min).

### Gates (Tersoff ladder cells, 300 K, tau 100 fs, dt 1 fs)

`nhc_gate.py`, ledger `nhc_gate_ledger.jsonl`.  The ledger's
`akitsushima` field reads 0.3.0a7 because the version was bumped after
the runs; its `module` field shows the worktree's code was used.

| Atoms | n_dof | t95, M = 2 (ps) | t95, M = 3 (ps) | Gate |
|---|---|---|---|---|
| 504 | 1509 | 3.15 | 0.77 | <= 1.5 ps: pass |
| 4032 | 12093 | 8.22 | 0.95 | pass |
| 13608 | 40821 | 14.71 | 1.09 | pass |

| Run | var(T) / (2 T0^2 / n_dof) | H drift (meV/atom/ps) | H peak-to-peak (meV/atom) |
|---|---|---|---|
| 504 atoms, 60 ps, M = 2 (t >= 10 ps) | 0.989 | 5e-6 | 0.0050 |
| 504 atoms, 60 ps, M = 3 (t >= 10 ps) | 1.011 | 7e-6 | 0.0051 |
| 13608 atoms, 30 ps, M = 3 (t >= 10 ps) | 1.009 | 3e-5 | 0.0013 |

All gates pass.  The 8 ps heating runs gave variance ratios 0.94 / 1.06
/ 1.53 (504 / 4032 / 13608 atoms) over their second half; the 13608-atom
case was therefore run to 30 ps:

| Window (ps) | 2-4 | 4-6 | 6-8 | 8-10 | 10-15 | 15-20 | 20-25 | 25-30 |
|---|---|---|---|---|---|---|---|---|
| var ratio | 2.35 | 1.58 | 1.48 | 0.89 | 0.89 | 0.96 | 1.03 | 1.10 |

The excess before 8 ps is the system settling after the heating (mean T
is 299.9-300.0 K in every window), not the three-link chain.

### Unit 3 again (0.3.0a7, two links, template burn-in)

Unit `durability/3b_fit_sw_13608_nvt900_20ps_burnin`:
`velocity_rescale_steps: 1000`, `equil_steps: 2000`, 20 ps production,
one thread.

| | Unit 3 (no burn-in) | Unit 3b (burn-in) |
|---|---|---|
| T_mean, production | 774.2 K | 899.7 K (+-2 %: pass) |
| per step | 0.48 s (load 1.3) | 0.52 s (load 2.5) |
| wall | 2.7 h | 3.2 h (22 ps with burn-in) |

Every other check passes except RSS growth, which fails on every unit of
the campaign (macOS RSS noise, see ADR-0025).  So with the shipped
templates' burn-in even two links reach the target; the slow heating
only shows when a run starts cold.

### Anneal lag

`anneal_lag.py`, ledger `anneal_lag_ledger.jsonl`: SW GeSn 4032 atoms,
NVT, geometric 1300 -> 300 K in 6 segments of 2000 steps (2 ps),
quench_every 500, tau_nhc / CSVR damping 100 fs, one seed.  Each cell:
segment mean T / mean T over the second half of the segment (K); last
column: fraction of the first 0.5 ps within +-5 % of the target, over the
six segments.

| Target (K) | 1300 | 969.6 | 723.1 | 539.3 | 402.2 | 300 | first 0.5 ps | wall |
|---|---|---|---|---|---|---|---|---|
| NHC, M = 2 | 814 / 894 | 971 / 975 | 828 / 770 | 623 / 578 | 466 / 433 | 348 / 323 | 0.02, 1.0, 0, 0, 0, 0 | 2757 s |
| NHC, M = 3 | 1158 / 1302 | 1005 / 964 | 754 / 727 | 561 / 540 | 420 / 405 | 313 / 302 | 0.02, 0.20, 0.22, 0.24, 0.22, 0.20 | 2682 s |
| CSVR | 1237 / 1304 | 1002 / 974 | 754 / 725 | 563 / 541 | 415 / 405 | 311 / 302 | 0.10, 0.22, 0.12, 0.06, 0.30, 0.20 | 2071 s |

- Two links: the first segment never reaches 1300 K, and from the third
  segment on the system stays 6-8 % above the target even in the second
  half.  (The second segment's 1.0 is the first segment's lag landing
  on the next target.)  Every anneal run with the NHC thermostat before
  0.3.0a8 followed its schedule this loosely.
- Three links: the second half of every segment is within 1 % of the
  target.  The segment mean is 3-5 % high because the first ~0.5 ps
  follows the step down.
- CSVR: the same as three links within the scatter of one seed.  The
  remaining lag is the response of the whole system (the potential
  energy must also relax) to a step in the target at a 100 fs coupling,
  not the chain.

### Findings

1. The remaining anneal lag is common to three-link NHC and CSVR, so
   neither candidate of Q14 (carry the chain state across chunks, or
   CSVR as the anneal default) would remove it: the chain state matters
   little once the first link is free, and CSVR already lags the same.
   What would shorten it is a shorter coupling time or longer segments.
   No further change is made; the decision stays with the user (Q14).
2. Reaching T0 from a minimum takes about ln(1.9) tau sqrt(n_dof) with
   two links.  Users of older versions who started NVT or anneal runs
   cold without burn-in should discard about that much of each run
   (13 ps at 13608 atoms with tau 100 fs).
3. `example/GeSn/run_cli_tests.py` puts the main repository first on
   `sys.path`, which hides a worktree's `akitsushima` from scripts that
   import it.  The gate scripts import `akitsushima` first and record the
   module path.
4. The LAMMPS default (`fix nvt` tchain 3) is quoted from memory and is
   UNCERTAIN; ASE 3.28 `NoseHooverChainNVT` (tchain 3) was checked.

## Consequences

- NVT and NPT runs started without the template's burn-in reach their
  target in about 1 ps instead of tau sqrt(n_dof); anneal segments follow
  their schedule within 1 % in their second half.
- The default NHC trajectory differs from 0.3.0a7.  `nhc_chain: 2`
  restores it bit for bit, and 0.3.0a7 checkpoints load as M = 2.
- A checkpoint written with M != 2 records `nhc_chain` and is refused
  by a run with another chain length.
