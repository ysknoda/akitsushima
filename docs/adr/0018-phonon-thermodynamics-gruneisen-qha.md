---
status: accepted (stage 2a implemented and verified 2026-10-08; stage 2b open)
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

## Self-critique of the chosen values

| value | source | confidence |
|---|---|---|
| cutoff 1e-3 THz | Gamma acoustic after the ASR is ~2e-7 THz; the lowest non-Gamma mode on a 48^3 mesh is 0.29 THz | > 95 % that it removes exactly the Gamma acoustic modes for the meshes used |
| default mesh 20^3 | 2-atom Si, 20^3 -> 48^3: F per atom changes by 0.0001 / 0.014 / 0.063 meV at 0 / 300 / 1000 K (high T weights the low-frequency modes through ln(hw/kT)) | a convenience default, not a convergence claim: converge q_mesh for the quantity used; low-T C_v needs more (documented in the template) |
| Debye mean | Anderson 1963 eq. (5) | > 95 % |
| D(1) = 0.9517 | Debye function tables | > 95 % |
