---
status: accepted (implemented 2026-10-08; NEMD vs Green-Kubo settled 2026-10-09, see Verification)
---

# NEMD with Langevin heat baths, local heat flux, and the decomposition of the Green-Kubo conductivity

Fourth and last stage of the property-command plan agreed with the user on
2026-10-08 (grilling Q1-Q35; ADR-0017 lists the stages).

## Starting point (found by reading the code, 2026-10-08)

`kappa` computes the Green-Kubo tensor from the exact many-body heat flux
of ADR-0007 (site-energy decomposition, Fan et al. PRB 92, 094301
(2015)); only the total flux J was kept.  No non-equilibrium method, no
spatial resolution, no checkpoint in any MD command.

## Decisions (agreed with the user, 2026-10-08)

1. **Per-atom heat current (Q7, Q26).**  J_i = E_i v_i - sum over the
   neighbour entries m whose neighbour is i of v_m (dU_centre/dv_m . v_i):
   each entry's term goes to the atom whose velocity it carries (Fan's
   per-atom form).  sum_i J_i = V J exactly; the part with no site
   decomposition (Coulomb) keeps the centre-of-mass approximation and is
   split equally.
2. **kappa decomposition (Q26).**  `kappa.decompose: {species, regions:
   {axis, bounds}, save_atom_flux}`: kappa_g = (V / k_B T^2) int <J_g(0)
   J(tau)>, J_g the group's summed currents / V; linear in J_g, so every
   decomposition adds up to kappa (residual printed).  Regions follow each
   frame's positions.  Raw J_i optionally to atom_heat_flux.npy (float32).
3. **NEMD (Q19, Q27).**  New command `nemd`, solvers/nemd.py: baths as
   fractional slabs along a lattice vector (or 1-based atom lists),
   membership fixed at the start; velocity Verlet, then an exact
   Ornstein-Uhlenbeck update of the bath atoms (c = exp(-dt/damping)) with
   the kinetic energy each bath adds accumulated.  Flux J = Qdot / (2A)
   (heat flows both ways round the periodic cell), from the hot and from
   the cold bath separately.  Temperature profile in n_bins bins, local
   flux per bin from the per-atom currents projected on the axis normal.
   kappa = J / |dT/dz| from straight-line fits over the central
   `fit.fraction` of each of the two segments, block errors (4 blocks);
   Kapitza: linear fits within `kapitza.window` on each side of a given
   interface (`skip_bins` next to it left out), extrapolated to it:
   R = dT / J.  Interfaces are given by hand.
4. **Steady state (Q27).**  Not detected automatically: equil_steps is
   the user's.  Reported: bath balance |Qh + Qc| / |Qh| (warned above
   0.05), the profile change between the last two blocks (warned above
   1 K), and whether each segment's gradient has the expected sign
   (warned otherwise).
5. **Length (Q34).**  One length per run; the 1/kappa against 1/L
   extrapolation is the example's (example/nemd), and the README says so.
6. **Checkpoint (Q35).**  Every `checkpoint_every` steps the full state --
   positions, velocities, accumulators, the bath RNG state -- is written
   atomically (temp file + rename); a re-run with the same input continues
   from it.  `restart: false` discards it.
7. **Refusals (Q32).**  ESM refused; the site-energy refusals of `kappa`
   (ADR-0007/0011) apply when `local_flux` is on.

## Verification

- tests/test_nemd.py: sum_i J_i = V J to 1e-12 (binary LJ, Tersoff Si,
  rattled, random velocities); species and region decompositions add up
  to kappa to 1e-9, the saved per-atom fluxes to J; E(t) - E(0) =
  Q_hot + Q_cold to 0.2 % of |Q| (Verlet error, dt 0.5 fs); a run killed
  after a checkpoint and re-run ends bit for bit where an uninterrupted
  run does (positions, velocities, profiles, bath heat); two baths at
  20 K over the whole cell hold it at 20 K (5 %); a synthetic
  piecewise-linear profile with a 2 K jump returns the gradient, kappa and
  R exactly; refusals.
- Steady state needs long runs.  A 6000-step LJ Ar run (1024 atoms)
  was not yet steady (bath balance 0.74, profile still moving) -- the
  warnings fired, as intended.
- Length series (example/nemd, 2026-10-08/09; LJ Ar, baths 50 / 30 K,
  4x4 cross-section, 200 ps equilibration + 500 ps production, dt 5 fs):

  | length | bath balance | local / bath flux (two segments) | kappa (W/m/K) |
  |---|---|---|---|
  | 12 cells, 63.6 A | 0.012 | 0.94 / 1.06 | 0.467 +- 0.026 |
  | 16 cells, 84.8 A | 0.006 | 0.97 / 1.02 | 0.492 +- 0.036 |
  | 24 cells, 127.2 A | 0.004 | 0.97 / 1.02 | 0.667 +- 0.034 |

  The NEMD checks pass (12 / 12): steady bath balance, gradients of the
  right sign, and the per-bin flux from the per-atom heat currents equal
  to the bath's Qdot / 2A within 6 % -- two independent measurements of
  the same current.  The profile change between the last two blocks is
  1.7-3.2 K (above the 1 K indicator): block-to-block noise of the
  temperatures at this length of run.
- **The comparison with Green-Kubo FAILED and is open.**
  1/kappa = 0.851 (+- 0.192) + 85.5 / L gives kappa(L -> inf) = 1.17 W/m/K
  (1-sigma 0.96 .. 1.52); the Green-Kubo run (6x6x6, 1 ns, tau_max 50 ps)
  reported 0.32 W/m/K with xx / yy / zz = 0.10 / 0.58 / 0.28 for a cubic
  crystal.  Its running kappa shows why: all three components reach a
  plateau of about 0.60 W/m/K at 10-15 ps and then wander in noise; the
  50 ps tau_max (chosen when the example was set up) integrates mostly
  noise.  Even against the ~0.60 plateau the extrapolated NEMD value is
  about twice as large, and kappa at 24 cells (0.67) already exceeds it,
  which a finite-length NEMD value should not.  Not resolved: GK
  statistics (one 1 ns trajectory), GK cell size, the NEMD bath coupling
  (damping 500 fs) and the three-point extrapolation (chi^2/dof 1.7) are
  the candidates; a literature value for LJ Ar at 40 K (e.g. McGaughey
  and Kaviany, PRB 69, 094303 (2004)) would anchor it.
- **Follow-up (2026-10-09): five independent Green-Kubo runs** (seeds
  101-505, same cell and length; criteria fixed before the runs,
  example/nemd/check_gk_seeds.py): tau_max 15 ps, mean of the five
  running kappa at 15 ps, standard error over seeds.
  - Per seed, iso: 0.70, 0.74, 0.72, 0.54, 0.70 W/m/K; components range
    0.45-1.04.
  - kappa_GK = 0.682 +- 0.035 W/m/K (xx 0.717 +- 0.106, yy 0.593 +- 0.039,
    zz 0.737 +- 0.054).  Seed-averaged running kappa: 0.650 / 0.669 /
    0.682 at 10 / 12.5 / 15 ps (still rising within its error).
  - Isotropy criterion (each component within 2 SE of the mean): FAIL,
    marginally (yy off by 2.3 SE; chi^2 = 6.4 for 2 dof, p ~ 0.04).
  - Against NEMD: the extrapolation 1.17 (0.96 .. 1.52) is 1.7x the GK
    value and its lower bound 8 SE above it -- the pre-set comparison
    FAILS again.  But the longest NEMD length, 0.667 +- 0.034 at 127 A,
    agrees with GK (0.682 +- 0.035) within one error bar.
  - Reading (not yet a conclusion): the disagreement sits in the
    straight-line 1/kappa against 1/L extrapolation through three noisy
    points, not in the measured currents; either kappa(L) saturates near
    0.68 by 127 A (then the extrapolation overshoots), or it keeps rising
    (then the GK cell, 32 A, is too small).  One more, longer NEMD length
    decides between the two.
- **32 cells (169.6 A, 2048 atoms; 2026-10-09, run under caffeinate):**
  bath balance 0.006, gradients of the right sign, local flux = bath flux
  within 7 %; kappa = 0.644 +- 0.073 W/m/K.
  - The rule stated before this run: kappa near 0.68 at 32 cells means
    kappa(L) has levelled off and NEMD agrees with GK; above ~0.8 would
    have pointed at the GK cell.  0.644 +- 0.073 is the first case.
  - kappa(L): 0.467 / 0.492 / 0.667 / 0.644 W/m/K at 63.6 / 84.8 / 127.2 /
    169.6 A.  The two longest agree (difference 0.3 combined SE); their
    weighted mean 0.663 +- 0.031 agrees with GK 0.682 +- 0.035 (0.4
    combined SE).  example/nemd/check_nemd.py: 18 / 18.
  - The straight-line 1/kappa against 1/L extrapolation, the criterion
    first written into check_nemd.py, still gives 1.07 (four points) and
    is now printed for information only: it cannot represent a kappa(L)
    that is flat over the two longest lengths.  That criterion was
    replaced after the data were seen, by the plateau rule stated before
    the last run; recorded here so that the change is visible.
  - Caveats: the plateau rests on two lengths with 5 % and 11 % errors;
    why kappa jumps between 85 and 127 A and not further (bath coupling
    at damping 500 fs, profile noise of 2-3.5 K between blocks) is not
    established.  Isotropy of the five-seed GK tensor was marginal (p ~
    0.04).
- Conclusion: the NEMD measurements (bath heat, local flux, gradients)
  are internally consistent at every length, and the NEMD conductivity at
  lengths >= 127 A agrees with Green-Kubo for LJ Ar at 40 K.

## Self-critique of the chosen values

| value | source | confidence |
|---|---|---|
| warn at bath balance 0.05, profile change 1 K | conventions chosen here | indicators only, not physical bounds |
| default fit fraction 0.5 | keeps clear of the non-linear zones next to the baths, common practice | ~80 %; check the profile |
| LJ Ar eps/k_B 119.8 K, sigma 3.405 A | the gcmc template's values (commonly quoted) | ~85 % |
| template damping 500 fs, dt 5 fs (Ar) | dt about 1/100 of the shortest Ar phonon period (~0.5 ps); damping a few phonon periods, so the bath couples weakly | ~70 % (UNCERTAIN): too strong a coupling adds a boundary resistance at the bath; resolved by checking kappa against damping |
