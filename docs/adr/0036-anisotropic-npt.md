---
status: accepted (2026-11-20; implemented in 0.2.0a26 -- see Addendum and Result)
---

# Anisotropic NPT: per-axis, full-cell and in-plane barostats

Tier A step 11 of ADR-0028 (Q14), including the in-plane barostat of
ADR-0010 item 2 (Q24).  Agreed with the user on 2026-11-20 (grilling,
Q1-Q12).

## Starting point (2026-11-20, 0.2.0a25)

- `NPTSimulation` is the MTTK integrator (Martyna, Tuckerman, Tobias &
  Klein, Mol. Phys. 87, 1117 (1996)) with an isotropic barostat: one
  momentum `p_eps`, mass W = (N_f + 3) kT tau_b^2, a 2-chain NHC on it,
  and the finite-N terms (3/N_f) 2 E_kin and (1 + 3/N_f).  The conserved
  quantity H_NPT is reported; tau -> infinity reduces it to NVE.
- ESM refuses NPT (ADR-0008): the isotropic barostat would move Lz, which
  places the ESM boundaries.  ADR-0010 item 2 left open whether the
  in-plane barostat should be isotropic or per axis.
- relax has a 9-component raw lattice mask (`lattice_constraints`).
- The neighbour list judges the skin against any affine cell change
  through the smallest singular value of the deformation.

## Decisions

1. **Modes (Q1).**  `NPTSimulation(cell_dof=...)`:
   - `iso` (default): volume only, as before.
   - `aniso`: Lx, Ly, Lz independent; shear fixed.
   - `tri`: all six strain components.
   - `xy`: Lx, Ly independent; Lz and shear fixed (ESM).
   - `xy_iso`: Lx and Ly scaled together.
   - `z`: Lz only (an epitaxial film with the in-plane cell held at the
     substrate's).
   Internally each mode is a list of strain generators (symmetric 3x3
   matrices: a diagonal unit, a shear pair e_ab + e_ba, or the sum of
   coupled diagonals).
2. **Algorithm (Q2).**  The MTTK fully flexible cell, Trotter-split as
   the current integrator (Tuckerman et al., J. Phys. A 39, 5629
   (2006)).  Not Parrinello-Rahman (no finite-N terms) and not stochastic
   cell rescaling (no conserved quantity to check).
3. **Target (Q3).**  A scalar hydrostatic P for every mode.
   Non-hydrostatic targets need the reference-cell term (Shinoda, Shiga &
   Mikami, PRB 69, 134103 (2004)) and stay future work.
4. **Frame (Q4).**  The cell moves as h_dot = G h with G = sum_k
   (p_k / W) E_k symmetric: no rotation, the input orientation is kept,
   and x, y, z are Cartesian.  On an oblique cell `aniso` changes angles
   too (affine, as LAMMPS).
5. **ESM (Q5).**  Only `xy` and `xy_iso`, and only P = 0 (the pressure
   of a cell that includes vacuum is ambiguous otherwise).  Other modes
   stay refused.  ADR-0008's table is updated; ADR-0010 item 2 is done.
6. **iso (Q6).**  `cell_dof: iso` keeps the current code path: results,
   thermo.log and checkpoints are bit-identical.  The other modes are a
   new path in the same class.  Checkpoint settings carry `cell_dof` only
   when it is not iso.
7. **Masses and chains (Q7).**
   - W = (N_f + 3) kT tau_b^2 per generator, the iso W.  LAMMPS `fix nh`
     also uses one undivided mass per component
     (`omega_mass = (natoms + 1) kT / p_freq^2`).  The stationary
     distribution does not depend on W.
   - One 2-chain NHC over all barostat momenta, target n_free kT:
     Q1_b = n_free kT tau_b^2, Q2_b = kT tau_b^2 (equal to the iso chain
     for n_free = 1).
   - Equations (alpha = 1/N_f, P_int = K/V + stress with K = sum m v v^T):
     - r_dot = v + G r
     - v_dot = F/m - (G + alpha Tr(G) I) v - (chain) v
     - p_k_dot = V Tr(E_k (P_int - P I)) + alpha Tr(E_k) sum m v^2
       - (chain) p_k
     - H = E_kin + E_pot + P V + sum p_k^2 / 2W + chains.
     For E = I this is the current iso integrator.
8. **Output (Q8).**  For modes other than iso, thermo.log adds Pxx, Pyy,
   Pzz [GPa, kinetic included] and a, b, c [A]; `tri` adds Pyz, Pxz, Pxy
   and alpha, beta, gamma.  The result adds the mean lattice, the mean
   pressure tensor and the mean and std of a, b, c; the plot adds
   per-axis pressure and length panels.
9. **Skewed cells (Q9).**  No flips.  In `tri` a warning is given once if
   a cell angle leaves [45, 135] degrees.
10. **Interface (Q10).**  `cell_dof` in Python and in the `md:` section
    (template lists the modes); anneal's md_quench passes it through;
    relax unchanged; `fix_atoms` freezes fractional coordinates as
    before.
11. **Validation (Q11).**  SW Si, 64-216 atoms; slow marks where needed.
    1. iso unchanged bit for bit.
    2. tau_nhc, tau_baro -> infinity reduces every mode to NVE.
    3. H drift small and ~ dt^2.
    4. <P_aa> of free components at the target within 3 standard errors
       (block averages); fixed components constant bit for bit.
    5. Cubic Si: aniso and tri give the iso <V> and <a> = <b> = <c>.
    6. tri from a sheared cell at P = 0 relaxes the shear.
    7. z with Lx, Ly compressed: <c> at low T matches relax with only c
       free.
    8. ESM slab, xy at P = 0: Lz and a perpendicular to b kept bit for
       bit, mean in-plane pressure 0; P != 0 and other modes refused.
    9. Checkpoint restart bit-identical for a non-iso mode.
    10. Not a test, run under nohup and reported here: the bulk modulus
        from iso volume fluctuations and C11, C12, C44 from tri strain
        fluctuations against `akitsushima elastic`.  No tolerance set in
        advance.
    11. Full suite.
12. **Recording (Q12).**  This ADR is written before the implementation
    and accepted with the result.  ADR-0010 item 2, ADR-0008's table and
    ADR-0028 step 11 link here; README section.  The version goes to
    0.2.0a26.  0.2.0b1 is a separate commit, asked for afterwards.

## Addendum (2026-11-20): Langevin piston option

Validation item 10 showed that one Nose-Hoover chain on the barostat
(decision 7) does not equipartition the strain degrees of freedom of a
crystal (numbers in the Result).  The strain modes are nearly harmonic and
exchange energy with each other and with the atoms only through weak
anharmonic coupling; the chain fixes their total kinetic energy, not how
it is shared.  Means are unaffected; fluctuations are wrong.  Agreed with
the user (option C):

- `baro_thermostat: nhc | langevin`, default `nhc` (decisions 2 and 7
  unchanged; iso with nhc stays bit-identical).
- `langevin` applies to every mode, iso included: each barostat momentum
  gets the half-step Ornstein-Uhlenbeck update
  p <- c p + sqrt((1 - c^2) W kT) xi, c = exp(-dt / (2 tau_baro)), in
  place of the barostat chain (Langevin piston: Feller, Zhang, Pastor &
  Brooks, J. Chem. Phys. 103, 4613 (1995); Quigley & Probert, J. Chem.
  Phys. 120, 11432 (2004) for a flexible cell).  The particles keep their
  Nose-Hoover chain.
- The kinetic energy the noise adds is accumulated (`baro_heat`) and
  subtracted from H_NPT, which therefore remains a drift diagnostic.
- The noise uses the run's random generator, which a checkpoint already
  restores, so restarts stay bit for bit; `baro_heat` is checkpointed and
  `baro_thermostat` enters the checkpoint settings when it is not nhc.
- CLI `md.baro_thermostat`, anneal md section; refused with
  ensemble nvt.

## Result (2026-11-20, 0.2.0a26)

**Implementation** (`solvers/molecular_dynamics.py`).
- `CELL_DOF_MODES` and `_cell_generators(mode)`.
- `NPTSimulation(cell_dof=..., baro_thermostat=...)`:
  - With iso the loop calls the former methods through `_baro_nhc`,
    `_baro_half` and `_baro_scale`.  Results, thermo.log and XDATCAR of an
    iso run were compared with the previous commit: identical bit for bit.
  - Other modes: `_pressure_tensor`, `_barostat_half_step_g`,
    `_nhc_baro_step_g` (target n_free kT), `_scale_step_g` (element-wise
    exponential for a diagonal G, so an absent axis is multiplied by
    exactly 1.0; eigen-decomposition otherwise) and `_npt_conserved_g`.
  - `_baro_langevin` for the option of the addendum.
- The base class gained `_ck_arrays()` (array state in a checkpoint:
  `p_g`, `baro_heat`); empty for the other ensembles.
- The NPT time-series plot used thermo_every * dt per entry although the
  lists hold one entry per production step; its time axis was stretched
  by thermo_every.  Fixed (plot only).
- W: one mass (N_f + 3) kT tau_b^2 per generator, the iso mass.  LAMMPS
  `fix nh` likewise uses one undivided mass per component
  (`omega_mass[i] = nkt / (p_freq[i]^2)`, nkt = (natoms + 1) kT; read from
  the LAMMPS source), so the UNCERTAIN /d of the grilling is not used.

**Validation** (`tests/test_npt_aniso.py`, 30 tests; SW Si, 64 atoms).
1. iso unchanged bit for bit (manual comparison with 0.2.0a25 above; the
   existing NPT tests pass).  A generator E = I with the particle chain
   off reproduces the iso integrator to 6e-12 A in the positions.
2. tau_nhc = tau_baro = 1e8 reduces every mode, and the Langevin piston,
   to NVE within 1e-8 A.
3. H drift over 400 fs at 600 K, 1 GPa: 5-6 meV at dt = 1 fs; halving dt
   divides it by 3.7-4.4 in every mode.  With the Langevin piston the
   effective H (heat subtracted) drifts less than 20 meV and less at the
   smaller dt (iso and tri); the Ornstein-Uhlenbeck step alone takes
   10,000 momenta to N(0, W kT) and books its heat exactly.
4. 300 K, 1 GPa, 8 ps after 1 ps: Pxx, Pyy, Pzz of aniso and tri at
   1.000 +- 0.004 GPa, the tri shears at 0 within their errors; in xy and
   z the free components reach 1 GPa while the fixed ones do not.  Fixed
   components (Lz in xy / xy_iso, a and b in z, shears outside tri) stay
   bit for bit, and a = b exactly in xy_iso.
5. <V> of iso, aniso and tri: 1273.70 +- 0.03, 1273.75 +- 0.06 and
   1273.70 +- 0.07 A^3; a = b = c within 0.002 A.
6. tri from gamma = 86.7 degrees at 50 K, P = 0: gamma 90.001 degrees and
   Pxy 0.005 GPa (from -6.2) over the last ps.
7. z with Lx, Ly compressed 2 %: <c> 11.0844 A at 10 K against 11.0834 A
   from relax with only c free (0.4 % of the 0.22 A Poisson expansion).
8. ESM MgO(001) slab (Buckingham, bc1): Lz and the rectangular cell bit
   for bit in xy and xy_iso; iso, aniso, tri and z refused, P != 0
   refused.  relax with a_x, b_y free gives a = 16.0213 A; xy_iso at
   10 K gives <a> = 16.0217 +- 0.0027 A and <Pxx>, <Pyy> within
   5e-4 GPa of 0 (manual; the test starts at a = 16.02 and checks the
   mean in-plane pressure).
9. A tri run, with nhc and with the Langevin piston, killed after its
   checkpoint continues bit for bit; a different `cell_dof` is refused.
10. Fluctuations against `ElasticConstantsCalculator` at 0 K (relaxed ion:
    C11 151.6, C12 76.6, C44 56.4, B 101.6 GPa; SW Si, 64 atoms, P = 0,
    tau_baro 200 fs; iso 200 ps, tri 300 ps after 20 ps; errors from 10
    blocks):

    | barostat | T | B (iso, volume) | C11 | C12 | C44 (tri, strain) |
    |---|---|---|---|---|---|
    | nhc | 100 K | 139 +- 9 | -- | -- | -- |
    | nhc | 300 K | 104 +- 5 | 146 +- 10 | 59 +- 10 | 85 +- 6 |
    | langevin | 100 K | 103 +- 4 | 143 +- 4 | 71 +- 5 | 55.3 +- 2.4 |
    | langevin | 300 K | 100 +- 5 | 147 +- 6 | 72 +- 5 | 54.9 +- 1.3 |

    Diagnosis of the nhc rows (80 ps runs, tri, 300 K): per-component
    <p_k^2 / W> / kT of 0.47, 0.47, 0.46, 1.97, 1.24, 0.90 at
    tau_baro 1000 fs and 0.78-1.35 at 600 K, with each strain variance
    tracking its own kinetic energy (so the stiffness each mode feels is
    the relaxed-ion one); iso at 100 K had <p_eps^2 / W> = 0.63 kT.  One
    chain per barostat momentum (length 3) equalised the kinetic energies
    (0.97-1.03) but not the strain variances (0.64-1.20).  The Langevin
    piston gives 0.94-1.18 in 70 ps and the table above.
    Not resolved: C11 with the Langevin piston is 1-2 sigma below the 0 K
    value at both temperatures (finite temperature, 64 atoms or
    statistics).
11. Full suite: 971 passed, 1 skipped.
