---
status: accepted (2026-11-24, 0.3.0a9)
---

# Temperature response of anneal segments; CSVR as the md_quench default

Follow-up to ADR-0044 (decision 4, Q14).  With a three-link
Nose-Hoover chain the second half of every anneal segment was within
1 % of its target, but the segment mean was still 3-5 % high, and CSVR
behaved the same.  ADR-0044 left the remedy open.  The verification
was agreed with the user on 2026-11-24 (grilling, Q1-Q13), and the
remedy after its result in a second grilling the same day (Q1-Q10).

## Hypothesis

A thermostat acts on the kinetic energy only, while a solid holds about
as much heat again in its potential energy, exchanged on the vibrational
time scale.  After a step in the target the whole system should
therefore relax with

| Thermostat | Kinetic energy relaxes in | Predicted tau_r |
|---|---|---|
| CSVR (Bussi, Donadio & Parrinello 2007) | damping | 2 damping |
| Langevin (BAOAB) | damping / 2 | damping |
| NHC, 3 links | about tau_nhc | about 2 tau_nhc |

and reach +-5 % of the target after t5 = tau_r ln(r0 / 0.05), with r0
the relative initial excess (T_prev / T - 1 when cooling; about 0.5 on
the first segment, which starts at the relaxed cell and heats).  The
picture needs the exchange between kinetic and potential energy to be
faster than the thermostat.

Re-fitting the ADR-0044 runs (4032 atoms, tau = damping = 100 fs) gave
tau_r 223 fs (NHC) and 221 fs (CSVR) and t5 0.34-0.51 ps against
2 tau ln(0.34 / 0.05) = 0.38 ps.

## Verification (agreed Q1-Q13)

- Runs: SW GeSn, FIRE-relaxed ladder cells (Ge0.875Sn0.125, 1/64
  vacancies), NVT md_quench, geometric 1300 -> 300 K, segment_steps
  2000, quench_every 500, dt 1 fs, T every 5 fs, seeds 42 and 43, quench
  capped at 50 FIRE steps.
  - 4032 atoms: tau (damping) 25 / 50 / 100 / 200 fs, NHC 3 links and
    CSVR.
  - 4032 atoms, 100 fs: Langevin; 3 and 12 segments (NHC).
  - 504 atoms, 100 fs: NHC and CSVR.
  - 26 runs, four processes at a time, resumable ledger.
- Per segment: tau_r from a fit of A exp(-t / tau_r) + c to r(t) = T /
  T_target - 1 (cooling: from the segment start; heating: from the
  minimum of r); t5 from the 0.05 ps moving average; the segment-mean
  excess; overshoot recorded only.
- The chain restarts every chunk (quench_every), three times inside each
  segment: the 0.1 ps mean T across those boundaries against the same
  statistic at mid-chunk times.
- Criteria, fixed before the runs: tau_r / prediction 0.8-1.25 (mean of
  the two seeds), t5 within +-25 % of the prediction, 504 and 4032 atoms
  within 15 % of each other; applied for tau >= 50 fs.  25 fs checks
  whether the vibrational exchange holds the response back.
- Not checked: whether the lag changes the minima an anneal finds (Q5).
- If the criteria fail, no code or documentation change; report and
  grill again (Q11).

Scripts and ledgers: `example/GeSn/large/adr0045/` (`anneal_response.py`,
`analyse_response.py`, `vib_period.py`, `ase_step.py`,
`anneal_response_ledger.jsonl`, `response_summary.json`,
`ase_step_ledger.jsonl`).

## Result

### Vibrational time scale

504 atoms, NVE 2 ps after 1 ps of CSVR, velocities every 2 fs: the
velocity autocorrelation first crosses zero at 44 fs (300 K) / 42 fs
(1000 K); the vibrational density of states has its mean at 6.1 THz
(period 164 fs) and its peak at 9.0 THz (111 fs).  Kinetic and potential
energy therefore exchange in about 40 fs, comparable to 2 tau at 25 fs.

### Cooling segments

tau_r / prediction, mean of 2 seeds x 5 segments (r0 = 0.34):

| tau or damping (fs) | 25 | 50 | 100 | 200 |
|---|---|---|---|---|
| NHC, 3 links | 1.03 | 1.04 | 1.11 | **1.51** |
| CSVR | 1.26 | 1.14 | 1.06 | 0.97 |
| Langevin | | | 1.10 | |

t5 / prediction: NHC 1.10 / 1.02 / 1.01 / 1.20, CSVR 1.30 / 1.04 / 1.04
/ 1.03, Langevin 1.10.

| Further cases (100 fs) | tau_r (fs) | tau_r / prediction | t5 / prediction |
|---|---|---|---|
| NHC, 3 segments (r0 1.08) | 273 | **1.36** | 1.03 |
| NHC, 12 segments (r0 0.14) | 179 | 0.89 | 1.14 |
| NHC, 504 atoms | 157 | 0.78 | 0.73 |
| NHC, 4032 atoms | 222 | 1.11 | 1.01 |
| CSVR, 504 atoms | 210 | 1.05 | 1.03 |
| CSVR, 4032 atoms | 212 | 1.06 | 1.04 |

Size: NHC 504 vs 4032 atoms differ by **30 %**, CSVR by 1 %.

### Heating (first segment)

t5 / prediction: NHC 1.06 / 1.09 / **1.42** / **1.82** (25 / 50 / 100 /
200 fs), CSVR 1.07 / **0.67** / 0.95 / 0.91, Langevin 1.11.  The NHC
heating is not exponential: its first link starts from rest, and the
fitted tau_r (486 fs at 100 fs, 2.5 ps at 200 fs) only describes the
slow tail.

### Chain restarts inside a segment

Jump of the 0.1 ps mean T across a chunk boundary, relative to the
target: NHC -0.2 +- 1.0 % (180 boundaries; control +0.1 +- 1.4 %), CSVR
-0.1 +- 1.4 % (control 0.0 +- 1.3 %), Langevin +0.3 +- 0.9 % (control
+0.2 +- 0.8 %).  Restarting the chain at a chunk boundary does not
disturb T.

### Verdict

- CSVR and Langevin meet the criteria for tau >= 50 fs, with one
  exception: the CSVR heating at 50 fs is faster than predicted (t5
  0.67).  At 25 fs CSVR is 26 % slower than 2 damping, as the 40 fs
  energy exchange predicts.
- The three-link NHC meets them only for cooling at 50-100 fs.  It fails
  at 200 fs, for the large step of 3 segments, in heating at 100 and
  200 fs, and between 504 and 4032 atoms.  The criteria are not
  adjusted after the fact.

## Why the Nose-Hoover chain drifts

The chain's first link is driven by (2K - n_dof kT) / Q_1, the second by
(Q_1 vxi_1^2 - kT) / Q_2.  With the standard masses (Q_1 = n_dof kT
tau^2, Q_j = kT tau^2; Martyna, Klein & Tuckerman 1992) the second force
grows with n_dof: as soon as the first link moves, the second pulls it
back, more strongly the larger the system.  This is the mechanism that
made two links scale with sqrt(n_dof) (ADR-0044), weakened but not
removed by a third link.

- Harmonic oscillators (1-9 THz, RK4, no akitsushima code), cooling step
  r0 = 0.34, three links: tau_r 177 / 221 / 279 fs at n_dof 1500 / 12000
  / 40000 (tau 100 fs) and 324 / 430 / 544 fs (tau 200 fs).  Five links
  are slower still (226 / 387 / 638 fs).  The 3-link values at 1500 and
  12000 match the MD (157 and 222 fs).
- ASE 3.28 `NoseHooverChainNVT` (tchain 3, tdamp 100 fs, same masses),
  EMT Cu, cooling step 402 -> 300 K after 2 ps of Bussi at 402 K:
  tau_r (fs):

  | Atoms (n_dof) | NHC, 3 links | Bussi (control) |
  |---|---|---|
  | 500 (1500) | 157 / 158 (2 seeds) | 153 / 175 (2 seeds) |
  | 2048 (6144) | 188 | 173 |
  | 5324 (15972) | 218 | 181 |

  The chain slows by 39 % from 500 to 5324 atoms, close to the harmonic
  model over the same n_dof (177 -> about 235 fs, interpolated in log
  n_dof: 1.33x).  Bussi changes by 10 %, within its 14 % seed scatter at
  500 atoms.  On EMT Cu tau_r / 2 tdamp is 0.77-1.09; the size trend,
  not the absolute value, is the check here.  Fit errors 3-12 fs; one seed at 2048 and
  5324 atoms (wall time 0.26 h and 0.6 h per run).

The drift is therefore a property of the standard chain, not of the
akitsushima integrator.  Changing the masses (e.g. Q_j = n_dof kT tau^2)
stays rejected for want of a published basis (ADR-0044, Q2).

## Decisions (second grilling, Q1-Q10)

1. **anneal default (Q1, Q6).**  md_quench uses CSVR at nvt when no
   thermostat is given, and the Nose-Hoover chain at npt (CSVR is nvt
   only, ADR-0039).  The anneal follows a schedule, and CSVR's response
   is predictable and size-independent.  The chain restarts every chunk
   anyway, so the anneal never used the NHC's continuous deterministic
   dynamics.
2. **damping 50 fs (Q5).**  The shortest coupling for which 2 damping
   holds (25 fs is held back by the vibrational exchange): T is within
   5 % about 0.2 ps into each segment of the template's 8-segment
   schedule.  The default applies whenever md_quench runs CSVR without a
   damping; Langevin keeps its own default (100 fs).
3. **Old inputs (Q6).**  `tau_nhc` or `nhc_chain` without `thermostat`
   at nvt is an error that names the fix (`thermostat: nhc`, or remove
   them).  Inputs from the 0.3.0a3+ template name their thermostat and
   are unaffected.
4. **Restart.**  The checkpoint records the thermostat, and a restart
   with another one is refused; checkpoints written before 0.3.0a9 carry
   none and are not checked.  (Added during the implementation: a run
   started without a thermostat before 0.3.0a9 would otherwise switch to
   CSVR on restart.)
5. **API (Q7).**  `GlobalOptimiser` has the same default.
6. **md NVT stays NHC (Q3).**  Its template now states the response
   times of the three thermostats and recommends CSVR for reaching a
   target quickly.  NVT runs start with the template's burn-in
   (ADR-0044), where the chain's response matters little.
7. **Tests (Q8).**  Tests with fixed numbers pin `thermostat: nhc`; new
   tests cover the defaults, the error and the restart check; the
   template test runs csvr, nhc and langevin.  No new verification run
   (Q9): CSVR at 50 fs was measured above (4032 atoms, 2 seeds).
8. **Carrying the chain state across chunks** is dropped: restarts do
   not disturb T.

## Consequences

- md_quench runs without a thermostat setting change dynamics: numbers
  differ from 0.3.0a8.  `thermostat: nhc` (with the former `tau_nhc`)
  restores them.
- In each segment the first ~2 damping ln(r0 / 0.05) (0.16 ps for the
  template's schedule) is still on its way; segment lengths of five
  times that or more keep the schedule meaningful.
- The NHC remains the `md` NVT / NPT default, with its size- and
  tau-dependent response documented.
