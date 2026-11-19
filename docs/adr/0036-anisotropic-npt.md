---
status: proposed (2026-11-20)
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
