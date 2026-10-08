---
status: accepted (implemented 2026-10-08; length-series example pending, see below)
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
- Length series (example/nemd, launched 2026-10-08, several hours):
  LJ Ar 40 K, lengths 12 / 16 / 24 cells, against Green-Kubo; judged by
  check_nemd.py (steady state, local flux = bath flux within 10 %, kappa
  extrapolated to infinite length within 25 % of GK).  Results to be
  recorded here.

## Self-critique of the chosen values

| value | source | confidence |
|---|---|---|
| warn at bath balance 0.05, profile change 1 K | conventions chosen here | indicators only, not physical bounds |
| default fit fraction 0.5 | keeps clear of the non-linear zones next to the baths, common practice | ~80 %; check the profile |
| LJ Ar eps/k_B 119.8 K, sigma 3.405 A | the gcmc template's values (commonly quoted) | ~85 % |
| template damping 500 fs, dt 5 fs (Ar) | dt about 1/100 of the shortest Ar phonon period (~0.5 ps); damping a few phonon periods, so the bath couples weakly | ~70 % (UNCERTAIN): too strong a coupling adds a boundary resistance at the bath; resolved by checking kappa against damping |
