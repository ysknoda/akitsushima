---
status: proposed (2026-11-18)
---

# Langevin thermostat for `md` (BAOAB)

Tier A step 7 of ADR-0028.  The user moved this item from tier B to A on
2026-11-12, right after the restart of step 6, so that the random-number
state is part of the checkpoint from the start.  Agreed with the user on
2026-11-18 (grilling, Q1-Q10).

## Starting point (2026-11-18, 0.2.0a21)

- **`md` thermostats.**  The NVT ensemble of `md` has only the
  Nose-Hoover chain (`NVTSimulation`).  `anneal` (md_quench), the
  Green-Kubo equilibration of `kappa` and `dynamics_analysis` use that
  class too.
- **Langevin in `nemd` (ADR-0020).**  `nemd` already has Langevin
  baths, but only on its hot and cold slabs.
  - Each step is velocity Verlet followed by one O step,
    v <- c v + s xi with c = exp(-dt / damping).
  - Its parameter is `damping` (fs, default 100).
  - It accumulates the heat Q the baths add and reports the drift of
    E - Q.
- **The checkpoint (ADR-0031)** already stores the random-number state.
  Every MPI rank holds the same data and draws from the same seed, so
  the random numbers agree across ranks.

## Decisions

1. **Form (Q1).**  A new class `LangevinSimulation`, one class per
   ensemble as with NVE and NPT.  It shares the output files,
   checkpoint and settings check of the other md classes.  In the CLI
   the ensemble stays `nvt` and `thermostat: langevin` selects it;
   the default is `nhc`.
2. **Integrator (Q2).**  BAOAB (Leimkuhler and Matthews, Appl. Math.
   Res. Express 2013, 34; J. Chem. Phys. 138, 174102 (2013)).  Each
   step is B (half kick), A (half drift), O (exact
   Ornstein-Uhlenbeck step over dt), A, force, B.  Its configurational
   error is small, and for a harmonic potential <x^2> is exact for
   every stable dt.  The integrator of `nemd` is not changed, because
   that would change its results.
3. **Centre of mass (Q3).**  After each O step the centre-of-mass
   velocity is removed, unless atoms are fixed.  The number of degrees
   of freedom follows `_compute_n_dof` (3N - 3, or 3N minus the fixed
   components).  Independent noise on each atom would otherwise make
   the whole crystal random-walk and mix into MSD analyses.  This is
   the idea of LAMMPS `fix langevin ... zero yes`.
4. **Parameter (Q4).**  `damping` in fs (the relaxation time 1/gamma),
   default 100 fs, the same name and unit as `nemd`.
5. **Diagnostic (Q5).**  The heat Q added by the O steps (and removed
   with the centre-of-mass velocity) is accumulated.  `thermo.log`
   shows E_tot - Q in place of H_NHC; its drift is the integrator check.
   Q is part of the checkpoint.
6. **Scope (Q6).**  `md` only.  `anneal` (where Langevin would suit
   quenching) and the equilibration of `kappa` are left for later.  The
   Green-Kubo production phase must never use it: the friction and
   noise change the heat-current autocorrelation.
7. **Validation (Q7).**
   - A harmonic potential (independent oscillators): <x^2> = kT/k and
     <v^2> = kT/m within the statistical error, and <x^2> the same at
     two time steps.
   - SW silicon: the mean temperature equals the target within its
     error, and the centre-of-mass velocity stays zero.
   - An interrupted and restarted run equals the uninterrupted one bit
     for bit, random numbers included (the test of ADR-0031).
   - The MPI CLI test gets a Langevin case that must agree across rank
     counts.
8. **`velocity_rescale_steps` (Q8).**  Not accepted with Langevin: a
   value above 0 stops the CLI with the reason.  The thermostat relaxes
   to the target within a few damping times by itself, and a key that
   is silently ignored misleads.
9. **Template (Q9).**  The NVT template gets `thermostat: nhc` and a
   commented `# damping: 100.0`.
10. **Recording (Q10).**  This ADR is written before the implementation
    and accepted with the result.  The version goes to 0.2.0a22.

## Result

(to be appended)
