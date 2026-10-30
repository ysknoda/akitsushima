---
status: accepted (revised 2026-10-31 after the VASP check; supersedes the 2026-10-30 plan)
---

# Isolated-atom reference for ISPIN = 1 training data

Agreed with the user on 2026-10-30 / 31 (grilling, Q1-Q20).  The first
plan (Q1: switch the reference to the non-spin-polarised atom) was
reversed by the calculations below (Q16).

## Question

The GeSn and AlN tutorials subtract isolated-atom energies computed
spin-polarised (ISPIN = 2), while every training structure is
non-spin-polarised (ISPIN = 1; GeSn 50 V01 OUTCARs, AlN 192 V01 + 802
round OUTCARs).  ENCUT 500 eV, GGA = PS, LASPH, PREC = Normal and the
POTCARs (PAW_PBE Ge 05Jan2001, Sn 08Apr2002, Al 04Jan2001, N 08Apr2002)
agree.  `dataset` subtracts sum_s n_s E_atom,s and the MLIP is fitted with
`isolated_atom_energy: 0`, so eps_s(G = 0) is pinned to E_atom,s
(ADR-0014 B): the choice of atom fixes the model's energy for an atom
with no neighbour inside the cutoff.

## Isolated atoms (example/*/vasp_ref, VASP 6.6.1, one core)

15 x 15.5 x 16 A box, Gamma only, ISYM = 0.  Energy without entropy, eV:

| | U-sph | U-axial | P-sph | P-int | U-sph - P-int |
|---|---|---|---|---|---|
| Al | -0.0011 | +0.0372 | -0.1921 | -0.2807 | 0.280 |
| N | -0.0058 | +0.6224 | -3.1870 | -3.1870 | 3.181 |
| Ge | -0.0068 | +0.4216 | -0.7597 | -0.7664 | 0.760 |
| Sn | -0.0037 | +0.3975 | -0.6568 | -0.6514 | 0.648 |

- U-sph: ISPIN = 1, ISMEAR = 0, SIGMA = 0.01; the p shell shared evenly
  (spherical).  U-axial: ISPIN = 1, ISMEAR = -2, the p electrons in one
  (N: one and a half) p orbital.  P-sph / P-int: ISPIN = 2 with NUPDOWN
  fixed, smeared / integer occupation (Hund).
- U-axial is not an aufbau state: the filled p orbital is pushed above
  the empty ones by its own Hartree term, so FERWE "1 1" (aufbau order)
  flipped between the p orbitals for 90 steps; FERWE on the highest p
  band converged in 20 steps (Ge: occupied p at -3.44 eV, empty p at
  -4.06 eV).
- N P-sph and P-int are the same configuration (p3 up) and agree.
- Sn P-sph is 5.4 meV below P-int.  The tutorials' Ge (P-int) / Sn
  (P-sph) mix was `vasp_ref.py`'s rule "the lowest ISPIN = 2 state", not
  an error.

## Uniform expansion (ISPIN = 1, the training data's method)

Relaxed alpha-Ge (2 atoms) and wurtzite AlN (4 atoms) scaled by s, with
the k-point density of s = 1.  E / atom (eV) against the ELM of the
tutorials (r_cut 5.0 A, fitted on the P-int reference), both on VASP's
absolute scale:

| s | Ge nn (A) | Ge VASP | Ge ELM | AlN nn (A) | AlN VASP | AlN ELM |
|---|---|---|---|---|---|---|
| 1.0 | 2.47 | -4.898 | -4.895 | 1.89 | -7.828 | -7.820 |
| 1.2 | 2.96 | -4.011 | -3.322 | 2.27 | -6.462 | -6.334 |
| 1.5 | 3.70 | -2.287 | -2.053 | 2.84 | -3.739 | -6.865 |
| 1.7 | 4.20 | -1.487 | -1.417 | 3.22 | -2.617 | -5.292 |
| 2.0 / 2.2 | 4.94 | -0.745 | -0.770 | 4.16 | -1.347 | -2.311 |
| 2.3 / 3.0 | 5.68 | -0.359 | -0.766 | 5.67 | -0.679 | -1.734 |
| 3.0 | 7.41 | -0.063 | -0.766 | | | |
| 4.0 | 9.88 | -0.008 | -0.766 | | | |

AlN s = 3.5 did not converge (ISPIN = 1 energy swinging by ~1 eV/atom over
200 steps) and is left out.

## Findings

1. The ISPIN = 1 surface does dissociate to U-sph, but only near 10 A
   (Ge).  The long tail is the fractional-occupation (static-correlation)
   error of non-spin-polarised GGA; a real atom spin-polarises as it
   leaves, so the physical limit is the spin-polarised atom.
2. At the cutoff (5 A) the ISPIN = 1 energy is close to P-int for Ge
   (-0.745 vs -0.766) and between the two for AlN (about -0.97 against
   -1.73 (P-int) and -0.003 (U-sph)).  Pinning to U-sph would put a step
   of about 0.74 eV/atom (Ge) at the cutoff in the other direction.
3. The large model errors are inside the cutoff (s = 1.2 - 1.7; AlN
   -6.9 vs -3.7 eV/atom at s = 1.5): stretched bonds are missing from the
   training data.  That is a data-coverage problem, not a reference one.

## Decisions

1. **Reference (Q16).**  For ISPIN = 1 training data use the lowest
   spin-polarised atom (the tutorials' existing values).  The tutorials'
   numbers stay valid; neither the tutorials' chapters nor the ADR-0010 A1
   model are re-run (Q18).
2. **Input (Q2, Q17, Q20).**  `dataset.isolated_atom_energy` takes, per
   species, a number or the path of the atom's VASP run (a directory with
   OUTCAR, or the OUTCAR); a path gives the last step's energy with the
   dataset's `energy` key and must hold one atom of that species.  The
   settings comparison and its warning, built first, were removed: the
   choice of reference is the user's call.  Version 0.2.0a16.
3. **Tutorials (Q8 - Q12, Q18).**  Both get an appendix on the
   isolated-atom calculations (the table above, occupations, a level
   diagram), the expansion figure (VASP, the P-int ELM and, for the
   comparison only, an ELM fitted on the U-sph reference), cohesive
   energies against both references, and a troubleshooting entry ("making
   the atom ISPIN = 1 to match the data backfires with a cutoff").
4. **Follow-up (Q19).**  Adding stretched structures (uniformly expanded
   bulk, slabs, stretched bonds) to the training data is separate work.

## Execution

All runs on one core (`mpirun -np 1`; OMP / NUMBA / OPENBLAS / MKL /
VECLIB threads = 1), one job at a time, at the user's request (another
session used 6 cores).  `example/` is not tracked by git, so this ADR is
the record of the results.
