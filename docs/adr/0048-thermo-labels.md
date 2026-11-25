---
status: accepted (2026-11-25, 0.4.0a4)
---

# Thermo rows and trajectory frames labelled by completed steps

ADR-0025's ladder finding 3, grilled with the user on 2026-11-25 among
the open findings (Q5: LAMMPS-style labels; Q12: older checkpoints;
Q13: scope and process).  The number 0048 was reserved then; ADR-0049
to ADR-0053 were written in the meantime.

## Starting point (0.4.0a3)

Every MD loop ran `for step in range(start, total)`, integrated one step
and then wrote a thermo row when `step % thermo_every == 0` and a
trajectory frame when `step >= equil_steps and step % traj_every == 0`.
`step` is the 0-based loop index, so the row labelled k held the state
after k + 1 steps, at "time" k dt: time 0 was the state after one step,
the initial state was never written, and the last row was
`n_steps - thermo_every` (ADR-0025 finding 3).  The same held for the
four integrators (NVE, NVT with NHC, the stochastic NVT of Langevin and
CSVR, NPT), for the Green-Kubo `run_md` of `kappa`, and for GCMC / CFCMC
(`osmc`) per cycle.  NEMD already counted completed steps but had no
row for the initial state.

## Decisions

1. **Labels (Q5 (a)).**  As in LAMMPS: row 0 is the initial state; then
   the state after c completed steps is written when
   `c % thermo_every == 0`, labelled c at time c dt.  Frames likewise,
   from c = equil_steps (the end of equilibration) when
   `c % traj_every == 0`; with `equil_steps: 0` the initial structure
   is the first frame.  The integration, the running averages (still
   over the states after steps equil_steps + 1 .. total), the
   checkpoints' state and every returned number are unchanged.  The
   stress is read on the steps whose row or frame needs it
   (ADR-0043), now counted the same way.
2. **Scope (Q13).**  `md` (four integrators), the `kappa` Green-Kubo
   run (thermo and trajectory), NEMD (initial row), GCMC and CFCMC
   (thermo, `N_vs_cycle.dat`, trajectory; `thermo_log` entries and
   `lambda_vs_cycle.dat` labelled by completed cycles).  GCMC has the
   same shift, so it is included.
3. **Heat flux.**  `heat_flux.dat` keeps sampling the states after steps
   1, 1 + heat_flux_every, ... (so the flux series and kappa are bit for
   bit unchanged) and labels them by completed steps (1, 5, 9, ... for
   every 4; it read 0, 4, 8 before).
4. **Older checkpoints (Q12 (a)).**  A checkpoint written before
   0.4.0a4 continues with the new labels; one comment line
   (`# ADR-0048: rows below are labelled by completed steps ...`) marks
   the change in thermo.log.  New checkpoints carry
   `"labels": "completed"`, which is not a setting, so either kind is
   accepted.
5. **Process (Q13).**  ADR-0048, 0.4.0a4, one commit on branch
   adr-0048-thermo-labels.

## Result (2026-11-25)

`example/GeSn/large/adr0048/ref_outputs.py` runs every touched loop on
small cells (SW Si 64 atoms rattled: NVE, NVT, Langevin, CSVR, NPT iso
and aniso, 5 + 20 steps; LJ: Green-Kubo 6 + 40 steps, NEMD 10 + 40,
GCMC and CFCMC 12 cycles), twice: every interval 1, and intervals 3-9.
`compare.py` against the same script on 0.4.0a3: all 35 checks pass.
- Every interval 1: the new thermo rows, without row 0, equal the old
  rows with Step + 1 and Time + dt, all other columns textually equal
  (md, Green-Kubo, GCMC, CFCMC); NEMD is row 0 plus the old rows; the
  XDATCARs hold one more frame (c = equil_steps) followed by the old
  frames.
- Both intervals: final positions, velocities, cell and all scalar
  results (averages, kappa, the flux series, N history, CFCMC
  estimates) bit for bit.

Tests: `tests/test_thermo_labels.py` (labels and frame counts for the
five classes; T_mean is the mean over rows c > equil_steps; a new
checkpoint continues bit for bit without a note; one written before
ADR-0048 continues with one comment line, the same rows otherwise and
the same final state; Green-Kubo, NEMD, GCMC and CFCMC labels).
Four existing tests counted the old rows or frames and were updated:
`test_cli_restart.test_md_cli_restart_false_starts_afresh` (rows 0-25
of a 25-step run, 0-20 before), `test_speedups_adr0043` (the stress is
read after loop steps 3, 7, 11 for frames c = 4, 8, 12; 0, 4, 8 before),
`test_structure_io` (NPT frames 20, 25, 30) and `test_active` (each
explorer's frame 0 is now the perfect start crystal, sigma 0, counted
accurate: 2 of 8 frames).  Full suite: 1385 passed, 1 skipped.

## Consequences

- Every thermo.log has one more row (row 0) and its rows hold other
  states (c instead of c + 1); trajectories gain the frame at
  c = equil_steps.  Scripts that skip the first row or assume
  `n_steps - thermo_every` as the last label need updating; `mdspec`
  aligns its temperature rows to the end of the log and is unaffected.
- `stop_sigma` (ADR-0037) checks the logged states, now including the
  initial one.
- `active` explorers' trajectories include their start structure as
  frame 0 when `equil_steps` is 0; it is classified like any frame.
