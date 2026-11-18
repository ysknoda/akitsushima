---
status: accepted (2026-11-18; implemented in 0.2.0a22 -- see Result)
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

## Result (2026-11-18, 0.2.0a22)

**Implementation.**  `LangevinSimulation(NVTSimulation)` in
`solvers/molecular_dynamics.py`, exported as `ak.LangevinSimulation`.
- It reuses the constructor, the neighbour list and the output files of
  NVT (`XDATCAR_nvt`, `thermo.log`, `POSCAR_final_MD`).
- `run` is a BAOAB loop: B, A, O, A, wrap, force, B.
- The O step uses `self.rng`, the generator seeded in the constructor,
  so its state is what the checkpoint of ADR-0031 already saves.  Each
  O step adds its kinetic-energy change to Q, then removes the
  centre-of-mass velocity, or zeroes the fixed components when atoms
  are fixed.
- `thermo.log` has `E-Q(eV)` where NVT has `H_NHC(eV)`, and the time
  series plot labels it accordingly.
- The checkpoint carries Q.  The settings check adds `damping` and
  drops `tau_nhc`, and the ensemble is `langevin`, so a Nose-Hoover
  checkpoint is not continued as Langevin.
- The CLI reads `thermostat` (nhc | langevin) and `damping`.
  `thermostat: langevin` with `npt` or `nve`, an unknown thermostat, and
  `velocity_rescale_steps > 0` all stop with the reason.  The NVT
  template shows `thermostat: nhc` and a commented `damping`.

**Validation.**

1. Harmonic oscillators.  64 independent oscillators, k = 1 eV/A^2,
   Si mass, 300 K, damping 100 fs; omega = 0.0186 /fs, period 338 fs.
   Errors are from 20 blocks.  Expected <x^2> = (kT/k)(1 - 1/N): the
   centre of mass stays at the minima.

   | Run | <x^2> / expected | <T> |
   |---|---|---|
   | dt 2 fs, 80 ps | 1.0006 +- 0.0043 | 299.9 +- 1.1 K |
   | dt 16 fs (omega dt = 0.30), 80 ps | 0.9938 +- 0.0037 | - |

   - A velocity-Verlet + O splitting (the `nemd` form) would be about
     +2.2 % off at dt 16 fs, about 8 sigma.  BAOAB is not.
   - With the noise planted 3 % too large, <x^2> came out at
     1.0615 +- 0.0046 and <T> at 318.2 +- 1.1 K.  The test fails such
     an error by more than 10 sigma.
   - The test asserts 4 sigma and at most 3 %.
2. SW silicon, 64 atoms, 600 K, 2500 production steps.
   - The mean temperature agrees with the target within 4 sigma.  A
     2000-step trial gave 594.7 +- 62.2 K, against the expected
     fluctuation T sqrt(2 / n_dof) = 61 K.
   - The total momentum stays below 1e-10 (8e-17 in the trial).
   - E_tot - Q varies by 2.9 meV over 2500 steps at dt 1 fs.
3. Restart.  The Langevin case of `test_md_restart.py` passes bit for
   bit for both the interruption and the extension.  If the checkpoint's
   random-number state is replaced by a fresh seed, the test fails.
4. MPI.  The case `langevin` of `test_mpi_cli.py` gives the same
   `thermo.log` (to 1e-5) and final structure (to 1e-6 A) on 1, 2 and 4
   ranks.

Other tests: fixed atoms keep zero velocity; a damping of 0, a positive
`velocity_rescale_steps` and a restart with another damping are
refused; the CLI selects Langevin and refuses misuse.

Full suite: 885 passed, 1 skipped (14.2 min).
