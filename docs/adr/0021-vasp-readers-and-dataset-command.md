---
status: accepted (implemented 2026-10-13)
---

# VASP output readers and the `dataset` command

Agreed with the user on 2026-10-13 (grilling, rounds 1-3) as the first stage
of a CLI test of `fit` and `mlip` on real VASP data: 50 GeSn relaxations
(63 atoms, one vacancy, Ge48-62 Sn1-15, PBEsol, ISIF=3; 405 ionic steps)
in example/GeSn/{fit,mlip}/GeSn_V01.

## Starting point

`fit` and `mlip` read the `fit` layout (`structures.csv` with
target,label,weight,struct_idx plus `<k>/POSCAR`), but nothing in the
package read VASP output.  The closest was `tools/ase_efs.py --read`, which
needs ASE (not a dependency), handles one structure, and was marked
unverified on real VASP output.  `fit` compares energies as absolute
values with no offset term, so DFT total energies (about -4.74 eV/atom
here) cannot be fitted by a potential whose zero is the isolated atom.

## Decisions

1. **Readers (`akitsushima/io/vasp.py`)**, standard library only:
   `read_vasprun` and `read_outcar` return one dict per ionic step with
   lattice, Cartesian positions, atom types, the three energies (free,
   without entropy, sigma->0), forces and stress.  Stress: VASP's numbers
   (OUTCAR `in kB` = vasprun `stress`, checked to be identical on the GeSn
   run) times `VASP_KB_TO_EV_ANG3`, no sign flip (ADR-0001).  The OUTCAR
   line is xx yy zz xy yz zx.  vasprun's two triangles differ in the last
   digit (~1e-6 kB), so the tensor is symmetrised.  A truncated vasprun.xml
   yields its complete steps with a warning.
2. **Cross-check.**  `dataset` reads one file and, by default, compares
   every step with the other one, stopping on a mismatch beyond OUTCAR's
   printed precision.  On run 00001 the two agree to 5e-9 A (lattice),
   5e-6 A (positions), 5e-7 eV/A (forces), 5e-6 kB (stress) and exactly
   in energy.
3. **`akitsushima dataset dataset.yaml`** (Q1-b, option a): sources (a
   run directory, or a parent of run directories), reader, energy choice
   (default without entropy), frames (all / last / first_last / stride),
   `isolated_atom_energy` (target = (E - sum_s n_s E_s) / N), split, and
   output.  Atoms are regrouped by species in the order `write_poscar`
   writes, and the forces move with them.  `manifest.csv` records run,
   step, composition, raw energy, target, largest force and half the
   smallest cell width (with an optional `r_cut_check` warning).
4. **Split by run, stratified by composition (Q4).**  The steps of one
   relaxation are strongly correlated, so they are never split between
   train and test.  Runs are sorted by composition and cut into n_test
   consecutive groups, and one run is drawn from each group (seeded).
   `mlip`'s internal `val_fraction` still splits by frame (one frame = one
   structure), so its validation score is optimistic on trajectory data.
   The held-out test set is the honest number.
5. **One data set for `fit` and `mlip` (Q11).**  The isolated-atom shift
   is applied once in `dataset`.  An MLIP trained on it uses
   `isolated_atom_energy: {Ge: 0, Sn: 0}`.

## Verification

tests/test_vasp_dataset.py covers four things:
- Synthetic vasprun.xml and OUTCAR (written in VASP's layout,
  tests/vasp_synth.py) are recovered exactly.  The off-diagonal stresses
  are distinct, so a component mix-up would show.
- A truncated file is handled.
- A two-step excerpt of the real GeSn run (tests/data/vasp_gesn) gives the
  OUTCAR's printed values, and both readers agree.
- On interleaved species, every F row stays with its atom after
  regrouping.  The energy shift, whole-run splits, stratification,
  refusal on disagreeing files and missing isolated-atom energies are
  also tested.

The `--dump-default dataset` template runs as shipped
(tests/test_cli_templates.py).
