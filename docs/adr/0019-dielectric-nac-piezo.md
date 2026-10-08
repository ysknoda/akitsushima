---
status: accepted (implemented and verified 2026-10-08)
---

# Born charges, NAC, dielectric tensor and piezoelectric response

Third stage of the property-command plan agreed with the user on
2026-10-08 (grilling Q1-Q35; ADR-0017 lists the stages).

## Starting point (found by reading the code, 2026-10-08)

The only charged model reachable from YAML is Buckingham + Coulomb with
fixed point charges (potential_json; Ewald, PPPM or ESM).  There is no
shell model or charge equilibration, so the model's own Born charges are
Z* = q I and its high-frequency permittivity is 1.  MLIP, EAM, Tersoff and
SW have no dielectric response.  No LO-TO splitting anywhere.

## Decisions (agreed with the user, 2026-10-08)

1. **Model (Q3).**  Rigid ion: Z* and eps_inf are inputs; no shell model
   or QEq (they would change the engine, the fits and ADR-0010 item 1).
2. **Input (Q12, Q22).**  A top-level YAML `born:` block only (no OUTCAR
   or BORN readers yet): `epsilon_inf` (scalar, three diagonal values or
   3x3); `born_charges.per_species` and `per_atom` (1-based POSCAR index,
   wins), each scalar / diagonal / 3x3; `clamped_ion_piezo` (3x6 C/m^2).
   Convention Z[k, a, b] = dP_a / du_kb = dF_kb / dE_a, the same as
   phonopy and VASP's OUTCAR (checked numerically, below).  The
   charge-neutrality sum rule is imposed by removing the mean of each
   component, and the correction is printed.
3. **Default (Q13).**  Missing charges come from the potential's point
   charges, a missing eps_inf is 1, with a printed warning (LO-TO
   overestimated by about the true eps_inf, eps_0 without the electronic
   part).  `dielectric` and `piezo` use this default; with no charges at
   all they refuse.  `phonon`, `gruneisen` and `qha` apply NAC only when
   a `born:` key is present (`born: {}` asks for the rigid-ion default),
   so existing Buckingham phonon runs do not change silently.
4. **NAC (Q14, Q24).**  Gonze and Lee (PRB 55, 10355 (1997)) with only
   the reciprocal-space Ewald term, as phonopy's default: the dipole-dipole
   force constants at the q-points commensurate with the supercell
   (shortest representatives) are Fourier-transformed and removed from
   the IFC, and every D(q) adds the dipole-dipole term back.  G-vectors:
   phonopy's 300-point sphere; Lambda from exp(-G_cut^2 tr(eps)/3 / 4
   Lambda^2) = 1e-10.  At Gamma the band path supplies the direction of
   approach.  Applied to bands, DOS, thermal properties, gruneisen and qha
   (Z*, eps_inf volume independent, as phonopy).  phonopy additionally
   symmetrises Z* and eps_inf to the crystal symmetry; this code uses
   them as given.
5. **`dielectric` (Q2, Q31).**  From the analytic Gamma matrix:
   eps_0 = eps_inf + (4 pi K_E / V) sum_m p_m p_m^T / w_m^2,
   p_m = sum_k Z_k e_m(k) / sqrt(M_k); per mode the oscillator strength,
   the eps_0 contribution, the mode effective charge and the IR intensity
   |p_m|^2; LO frequencies along x, y, z.  eps(omega) is deferred.
6. **`piezo` (Q15, Q29, Q30).**  e = e0 + (e/V) sum_k Z_k du_k/deta from
   the internal strain of `elastic` (run, or reused through `load_elastic`
   when the input hash -- POSCAR and potential -- matches); d = e S
   (pC/N); k_ij = d_ij / sqrt(S_jj eps^T_ii) with the stress-free
   eps^T = eps_vac eps^S + e d^T.  eps^S (strain-clamped, relaxed-ion) of
   the elastic cell by relaxing the ions in a +-5e-3 V/A field along each
   axis.  |e| < 1e-4 C/m^2 everywhere is reported as consistent with a
   centrosymmetric cell (no symmetry library).  e0 is zero for point
   charges.
7. **Reuse (Q30).**  `dielectric.load_ifc` takes a `phonon` output
   directory, `piezo.load_elastic` an `elastic` one; both check the input
   hash.  The elastic hash is now that of the input POSCAR (it was the
   pre-relaxed structure in 0.2.0a6), the same as phonon's.

## Verification (tests/test_dielectric.py, example/dielectric)

- Index convention: with a deliberately non-symmetric Z* and anisotropic
  eps_inf on MgO, NAC frequencies agree with phonopy (is_symmetry=False)
  to 5e-5 THz; with Z transposed they are 0.08 THz off.  With symmetry on,
  phonopy projected those tensors onto cubic symmetry and disagreed by
  0.39 THz -- the reason for the flag.
- NAC leaves D unchanged at commensurate q (to 1e-6 THz: the D-ASR
  correction is recomputed from the short-range IFC); LO at Gamma along
  [001] and [110] agrees with phonopy to 2e-3 THz; ZnO (diagonal Z*):
  six q-points including non-commensurate ones within 2.3e-4 THz.
- MgO: Lyddane-Sachs-Teller eps_0/eps_inf = (w_LO/w_TO)^2 to 2e-4; the IR
  triplet carries all of eps_0 - eps_inf; eps_0 from the modes equals
  eps_0 from field relaxation (9.3768 with 0.01 A force constants, 9.3809
  extrapolated to zero displacement, 9.3801 by field at 5e-3 and 1e-2
  V/A).  Against experiment (information): TO 12.40 THz (12.0), LO 22.11
  (21.5), eps_0 9.38 (9.8).
- piezo: units and definitions by hand; MgO e = 0 with the
  centrosymmetric note; load_elastic accepted for the same input, refused
  for another r_cut; dielectric refused for Tersoff Si (no charges).
- ZnO: see example/dielectric/check_dielectric.py (eps_0 uniaxial, LO
  direction dependence, the 6mm form of e).

## Self-critique of the chosen values

| value | source | confidence |
|---|---|---|
| 300 G-vectors, Lambda from 1e-10 | phonopy DynamicalMatrixGL defaults (read in its source) | > 95 % that they are phonopy's; only interpolation between commensurate points depends on them |
| field 5e-3 V/A | plateau found on MgO (5e-3 and 1e-2 agree to 4e-6) | ~85 % for other systems: softer lattices move further at the same field (UNCERTAIN; a field scan resolves it) |
| centrosymmetric note at 1e-4 C/m^2 | numerical noise of e on MgO < 1e-6 | ~80 % as a general threshold |
| MgO Lewis-Catlow parameters | Lewis and Catlow, J. Phys. C 18, 1149 (1985) | ~85 % (recalled; the test and template use them only as a model) |
| MgO eps_inf 2.95, TO 401, LO 718 cm^-1, eps_0 9.8 | experiment as commonly quoted | ~85 %, information only |
| ZnO e33 0.96, e31 -0.62, e15 -0.37 C/m^2; d33 12.4, d31 -5.0 pC/N | experiment as commonly quoted (Kobiakov 1980; Wang et al.) | ~80 %, UNCERTAIN, information only |

## Consequences

- `PhononCalculator.set_nac`, `build_dynamical_matrix(q, q_direction)`;
  solvers/dielectric.py.  Version 0.2.0a9.
- Found while timing: `compute_ifc` displaced every supercell atom (6 N_sc
  force calls) though only the origin-cell rows are used -- 27x the
  necessary work for a 3x3x3 supercell.  Fixed in a follow-up commit:
  only the origin-cell atoms are displaced (6 N_basis calls) and the
  other columns are filled by lattice translation,
  Phi[(a, M), (b, L)] = Phi[(a, M - L), (b, 0)], before the unchanged
  symmetrisation and ASR.  Against the old implementation: IFC equal to
  2.6e-12 eV/A^2, frequencies to 1.7e-7 THz (Tersoff Si 4x4x4, AlN 5x5x3,
  five q-points each); check_phonon.py (phonopy) unchanged.  Time: Si
  1.53 -> 0.34 s, AlN 4.20 -> 0.28 s, MgO 3x3x3 Ewald 147 -> 4.8 s.
