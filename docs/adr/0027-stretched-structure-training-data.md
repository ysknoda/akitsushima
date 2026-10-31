---
status: accepted (plan, 2026-10-31)
---

# Stretched-structure training data for the GeSn and AlN MLIPs (round 3)

Agreed with the user on 2026-10-31 (grilling, Q1-Q11).  Follow-up of
ADR-0026: on uniformly expanded alpha-Ge and wurtzite AlN the tutorial ELM
misses ISPIN = 1 VASP by up to 0.7 eV/atom (Ge) and 3 eV/atom (AlN) at
1.2 - 1.7 times the bond length, because stretched bonds are missing from
the training data.

## Where ISPIN = 1 data are physical (Q5, Q11)

The expansion scans were repeated with ISPIN = 2 (ferromagnetic start,
NUPDOWN free; example/*/vasp_ref, `--expand-spin`).  E / atom, eV:

| | s | nn (A) | ISPIN = 1 | ISPIN = 2 | diff (meV) | moment (muB / cell) |
|---|---|---|---|---|---|---|
| Ge | 1.50 | 3.70 | -2.2874 | -2.2874 | 0.0 | 0.01 |
| Ge | 1.70 | 4.20 | -1.4870 | -1.4893 | -2.2 | 3.40 |
| Ge | 1.75 | 4.32 | -1.3298 | -1.4133 | -83.6 | 4.00 |
| Ge | 2.00 | 4.94 | -0.7446 | -1.1198 | -375 | 4.00 |
| Ge | 4.00 | 9.88 | -0.0079 | -0.7600 | -752 | 4.00 |
| AlN | 1.50 | 2.84 | -3.7393 | -3.7393 | 0.0 | 0.00 |
| AlN | 1.55 | 2.93 | -3.4059 | -3.3319 | +74.0 | 4.00 |
| AlN | 1.60 | 3.03 | -3.1097 | -3.1757 | -66.0 | 4.00 |
| AlN | 2.20 | 4.16 | -1.3471 | -2.0933 | -746 | 4.00 |

- The ISPIN = 2 Ge curve goes to the spin-polarised atom (-0.766), which
  confirms ADR-0026 from the other side.
- AlN s = 1.55: the ISPIN = 2 run stopped in a metastable magnetic state
  74 meV/atom ABOVE the non-magnetic one; the ground state there is
  non-magnetic.
- Rule: s_max is the largest s at which the ground states (the lower of
  ISPIN = 1 / 2) differ by at most 10 meV/atom (UNCERTAIN, a judgement
  call).  **Ge s_max = 1.70 (nn 4.20 A); AlN s_max = 1.55 (nn 2.93 A).**
- AlN s = 3.5 did not converge in either spin setting and is left out.

## Decisions

1. **Aim (Q1).**  Uniformly expanded bulk, plus stretched and rattled
   cells.  Slabs are separate work.
2. **Systems (Q2).**  GeSn and AlN.
3. **VASP (Q3, Q8).**  Run by Claude with the local VASP 6.6.1 (the
   training data used 6.4.2 / 6.4.3; ADR-0010 A6 measured <= 0.32
   meV/atom between them).  Training-data settings: PBEsol (GGA = PS),
   ENCUT 500 eV, PREC Normal, LASPH, ISPIN 1, ISMEAR 0, SIGMA 0.01, same
   POTCARs; EDIFF 1E-4 (the round-0 rule); NSW 0.  k mesh at the
   training data's density: k_i = ceil(K / L_i) with K = 44 A (GeSn,
   4 x 4 x 4 on ~11.1-11.5 A) and 37 A (AlN, 6 x 4 x 4 on 6.2 / 10.8 /
   10.0 A).  NPAR is dropped (one rank per job).
4. **Cells and counts (Q6, Q7).**  Small cells only (a 63-atom single
   point takes about 10 h on one core).  Validation stays separate: the
   2-atom alpha-Ge and 4-atom wurtzite scans of ADR-0026 are never
   trained on, so cells identical to them use offset s values.
   - GeSn (a0 from Vegard between alpha-Ge 5.701 and alpha-Sn 6.546 A,
     vasp_ref):
     - uniform: 8-atom Ge8 at s = 1.05, 1.15, ..., 1.65 (7); 8-atom
       Ge7Sn1 at s = 1.10, 1.20, ..., 1.70 (7);
     - stretched + rattled N(0, 0.1 A): 16-atom cells at s = 1.05 -
       1.70 in steps of 0.05, three per s (Ge16, Ge14Sn2, Ge12Sn4): 42.
   - AlN (the relaxed wurtzite of vasp_ref):
     - uniform: 4-atom cell at s = 1.05, 1.15, ..., 1.55 (6);
       c/a +3 % / -3 % alternating at s = 1.10, ..., 1.50 (5);
     - stretched + rattled N(0, 0.1 A): 16-atom 2 x 2 x 1 cells at s =
       1.05 - 1.55 in steps of 0.05, three per s: 33.
   - About 100 single points in all.
5. **Models (Q4, Q10).**  The data become round 3 (train only).  The
   tutorials' ELM and NN and chapters 4-6 and appendix D are rebuilt;
   the ADR-0010 A1 model (`ac_elm_iso_r2`) is retrained and re-judged.
6. **Criteria (Q9, UNCERTAIN thresholds).**
   - The validation scans: |ELM - VASP| <= 50 meV/atom for s <= s_max.
   - Tutorial test errors (E, F) no more than 10 % worse.
   - A1: relax 10/10, anneal 10/10, DFT check 3/3 again.

## Execution

From 2026-10-31 the user allows six cores in total: six one-core VASP
jobs at a time, Python with up to six threads.  `example/` is not tracked
by git; this ADR records the results.
