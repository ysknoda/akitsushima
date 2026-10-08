---
status: accepted (stages 2a and 2b implemented and verified 2026-10-08)
---

# Phonon thermodynamics, Grueneisen parameters and the quasi-harmonic approximation

Second stage of the property-command plan agreed with the user on
2026-10-08 (grilling Q1-Q35; ADR-0017 lists the stages).  Two commits:
2a, harmonic thermodynamics in `phonon`; 2b, constant-volume cell
relaxation, `gruneisen` and `qha`.

## Starting point (found by reading the code, 2026-10-08)

`phonon` gave F_vib(T) only, by integrating the Gaussian-smeared DOS
histogram (bin width and smearing errors, worst at low T), and a Debye
temperature from the acoustic slopes.  The DOS q-grid,
(i + 1/2)/n - 1/2, is Gamma-centred only for odd n.  No C_v, S or U.

## Decisions (agreed with the user, 2026-10-08)

1. **Where (Q2).**  A `thermal:` block of `phonon` (t_min, t_max, t_step,
   q_mesh, cutoff_THz); `gruneisen` and `qha` are new commands (2b).
2. **Mode sum (Q5).**  F, U, S and C_v by a direct sum over the modes of a
   Gamma-centred mesh, q_frac = i/n (folded to [-1/2, 1/2)), the grid of
   phonopy's `is_gamma_center=True`; `phonon_free_energy` uses the same
   sum.  Modes at or below `cutoff_THz` (default 1e-3 THz: the three
   acoustic modes at Gamma, which are 2e-7 THz after the ASR, and
   imaginary modes) are left out; imaginary ones are counted and warned
   about.  phonopy leaves out f <= 0 by default, so comparisons pass it
   the same cutoff.
3. **Units (Q16).**  Per primitive cell (eV, k_B), per atom, and per mole
   of cells (kJ/mol, J/K/mol, as phonopy).  Constants: h, e, k_B, N_A
   exact (SI 2019); phonopy's differ from them by about 6e-7.
4. **Theta_D(T) (Q16).**  The Debye temperature whose Debye C_v equals
   the computed one at each T: C_v / (3 N k_B) = D(Theta/T),
   D(y) = (3/y^3) int_0^y x^4 e^x / (e^x - 1)^2 dx, solved by bisection
   (D falls monotonically from 1 to 0).  NaN at T = 0 and where C_v is not
   inside (0, 3 N k_B).  Its low-T limit is meaningful only on a mesh fine
   enough for the long-wavelength acoustic modes (below).
5. **Output (Q25).**  `phonon_results.json` (with the input hash of
   ADR-0017) and `thermal_properties.dat`.

## Bug found: the sound-velocity Debye temperature was 0.69x

`debye_temperature_from_vsound` averaged v_D = (1/v_L^3 + 2/v_T^3)^-1/3.
The Debye mean is 3/v_D^3 = 1/v_L^3 + 2/v_T^3 (Anderson, J. Phys. Chem.
Solids 24, 909 (1963), eq. (5)), so v_D and theta_D were 3^(-1/3) = 0.693
times too small (Tersoff Si: 375.7 K, now 541.9 K).
`example/phonon/check_phonon.py` had copied the code's definition and so
could not see it; it now uses the Debye mean.

## Verification (stage 2a)

- `tests/test_phonon_thermal.py` (Tersoff Si, published set):
  U = F + TS to 1e-12; S = -dF/dT and C_v = dU/dT against central
  differences to 1e-6; T = 0 gives the zero-point energy; C_v at 1e7 K is
  3 k_B per atom less the three Gamma acoustic modes; the sums against
  phonopy's per-mode formulas on the same frequencies to 1e-5 (its
  constants); end to end against phonopy's own force constants on a
  12^3 mesh, F / C_v to 1e-4 and S to 3e-4 (10 K is the most sensitive);
  D(1) = 0.9517; Theta_D inversion round trip to 1e-8.
- Theta_D(T) at 15 K against the corrected sound-velocity value 541.9 K:
  957 / 666 / 600 K at 4 K on 20^3 / 32^3 / 48^3 meshes, 571 / 565 / 563 K
  at 15 K -- the mesh limits the low-T end; within 5 % at 15 K on 32^3.
- `example/phonon/check_phonon.py` (Si and AlN against phonopy): F_vib
  per atom within 0.003 meV (tolerance 0.1), sound-velocity theta_D
  541.9 / 782.1 K against phonopy-derived 541.9 / 782.1 K.

## Stage 2b decisions (agreed 2026-10-08, Q6, Q17, Q18, Q33)

6. **Constant-volume relaxation (Q17, Q23).**  `GeometryOptimiser(
   fix_volume=True)` (YAML `relax.fix_volume`): the cell force is
   projected orthogonal to dV/dh (prop. to h^-T, within the free
   components), every step is followed by an isotropic rescale back to the
   starting volume, and the stress criterion tests the deviatoric stress:
   at a constant-volume minimum the stress is purely hydrostatic.  Works
   with every method; refused with ESM and without relax_cell.
7. **Volume points.**  Each volume is reached by isotropic scaling of the
   primitive cell, then shape and ions are relaxed at fixed volume in the
   phonon supercell (cell-major, so the primitive cell is read back from
   its first N atoms); force constants of that relaxed cell.  Always the
   fixed-volume relaxation, also for cubic cells, where it reproduces the
   isotropic scaling (Q17).
8. **Mode Grueneisen (Q6, Q18).**  `gruneisen`: V0 (1 -+ dv), dv = 0.01;
   gamma = -(V0 / 2 w^2) <e| dD/dV |e> with dD/dV the central difference
   at the same reduced q, eigenvectors at V0, degenerate sets
   (|df| < 1e-4 THz) diagonalised in their subspace and assigned in
   ascending order, as phonopy does (it groups by eigenvalue rather
   than frequency); Gamma acoustic modes NaN.
   Output: the q-path, a mesh, and the C_v-weighted mean gamma(T); the
   three pressures also give B for information.
9. **QHA (Q6, Q17, Q33).**  `qha`: n_volumes (7, >= 5) in V0 [1 -+ 0.05];
   F(V, T) = E(V) + F_vib(V, T) fitted per T with third-order
   Birch-Murnaghan (default) or Vinet (added to `fit_equation_of_state`);
   V(T), B(T), G(T); alpha by central differences of V(T) (one extra
   temperature is computed so alpha exists at t_max, as phonopy);
   C_v and S at V(T) from fourth-order polynomials in V;
   C_p = C_v + T (dV/dT)(dS/dV); gamma = alpha B V / C_v.
   A volume with a mode below -imaginary_tol_THz (0.05 THz) stops the run
   unless `skip_unstable_volumes: true` drops it; nothing is reported
   above the temperature at which V(T) leaves the sampled volumes.

## Verification (stage 2b)

- fix_volume (tests/test_solvers.py): wurtzite AlN started 3 % off in
  c/a and sheared, at 97 % volume: FIRE, L-BFGS and SD keep the volume to
  1e-12, end hydrostatic (deviatoric < 1e-3 GPa) and reach the c/a of an
  independent golden-section search over c/a (ions relaxed at each) to
  2e-5; a rattled, compressed Si cube stays the scaled cube.
- Mode gamma (tests/test_qha.py, Tersoff Si): against -d ln w / d ln V
  from the eigenfrequencies to 2e-3; against phonopy's GruneisenBase on
  phonopy force constants of the same three structures to 3e-4; the
  Gamma optical triplet has one gamma.  Si's TA gamma at X is negative
  (-0.20), as known for Si.
- QHA against PhonopyQHA fed the same E(V), F, S, C_v (both EOS): V(T) to
  1e-7, B(T) and C_p to 1e-5, alpha and gamma to 1e-3 of their maxima
  from 50 K (below, alpha ~1e-8 /K is at the level of the EOS-fit
  rounding).  Synthetic limits: no vibrations give V(T) = V0 and
  alpha = 0; a free energy that drives V(T) out of the scan stops the
  output there.  The imaginary-mode stop and skip are tested through the
  CLI.
- The thermodynamic gamma = alpha B V / C_v at 300 K against the
  C_v-weighted mean mode gamma: Si 0.797 / 0.780, AlN 0.596 / 0.581
  (within 5 %; they coincide only if B were volume-independent).
- Bulk modulus of Tersoff Si: 97.81 GPa from the gruneisen pressures and
  from the static E(V) of the qha scan, against 97.8 from `elastic`.
  QHA B(T = 0) is 95.8 GPa: the zero-point motion expands V0 from 40.07
  to 40.32 A^3.
- Shape relaxation in a real scan: the AlN Tersoff of Karaaslan et al.
  reaches only Al-N first neighbours and relaxes to the ideal wurtzite,
  so its c/a stays sqrt(8/3) at every volume (symmetry: an isotropic
  scaling keeps all bonds equal and the stress hydrostatic) -- the first
  version of the example check expected it to move and was wrong.  ZnO
  (Buckingham + PPPM, non-ideal c/a) is the case where it does move;
  see example/qha/check_qha.py.

## Self-critique of the chosen values

| value | source | confidence |
|---|---|---|
| cutoff 1e-3 THz | Gamma acoustic after the ASR is ~2e-7 THz; the lowest non-Gamma mode on a 48^3 mesh is 0.29 THz | > 95 % that it removes exactly the Gamma acoustic modes for the meshes used |
| default mesh 20^3 | 2-atom Si, 20^3 -> 48^3: F per atom changes by 0.0001 / 0.014 / 0.063 meV at 0 / 300 / 1000 K (high T weights the low-frequency modes through ln(hw/kT)) | a convenience default, not a convergence claim: converge q_mesh for the quantity used; low-T C_v needs more (documented in the template) |
| Debye mean | Anderson 1963 eq. (5) | > 95 % |
| D(1) = 0.9517 | Debye function tables | > 95 % |
| dv = 0.01 (gruneisen) | agreed (Q18); phonopy has no default (the user supplies the three volumes) | the log-derivative check agrees to 2e-3 at this step |
| 7 volumes, +-5 % (qha) | agreed (Q17); phonopy examples use similar scans | a default: the run reports t_max where V(T) leaves the scan |
| imaginary_tol_THz 0.05 | 250x the Gamma-acoustic residue after the ASR (2e-7 THz), far below any real soft mode | ~85 %; a soft mode between 0 and 0.05i THz passes unnoticed (UNCERTAIN; raise it per system) |
| degeneracy 1e-4 THz | phonopy's degenerate_sets cutoff is also 1e-4 (checked in its source), but phonopy applies it to the eigenvalues of D, here to frequencies in THz | the two criteria group the same modes for the Si and AlN cases checked (oracle agreement 3e-4) |
