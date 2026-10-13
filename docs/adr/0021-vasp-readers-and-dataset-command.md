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

## CLI test on the GeSn data (2026-10-13)

Runner: example/GeSn/run_cli_tests.py.  The ledger is runs/ledger.jsonl
(example/ is not tracked).

**Stages that passed**
- dataset: 3/3.  405 structures, 40/10 runs; the fit and mlip copies are
  byte-identical.
- fit: SW and Tersoff both pass, loss below the initial one.
  - SW: test RMSE E 13 meV/atom, F 0.32 eV/A.
  - Tersoff: test RMSE E 5.8 meV/atom, F 0.20 eV/A.
- mlip: 23/23.  ACSF tabular and atom-centred models, the physics options
  and the write_table round trip all pass.  The external-descriptor cases
  pass after the fixes below.  Test RMSE: E 0.5-3.8 meV/atom, F 0.06-0.23
  eV/A.

**References.**  VASP references are in example/GeSn/vasp_ref.
E_coh is Ge 4.13 and Sn 3.59 eV/atom against 3.85 / 3.14 measured; the
calculations are PBEsol without SOC.

**Findings and fixes**
- `write_table_csv` now writes 17 significant digits.  With 12 digits an
  ELM retrained from the written table differed by up to 2e-4 in its
  predictions: rounding of 5e-12 was amplified by the ill-conditioned
  solve.  Bit-for-bit round trip is now tested.
- `write_table` also writes `descriptor_table_test.csv`.
- Outside this package, atomic_descriptor_ver2.0.0.py had three problems:
  - Its OUTCAR energy-line rule also matched VASP 6's per-SCF lines.  The
    ionic-step line is now recognised by "entropy=".
  - It wrote weights F -> 1 and S -> V, where `load_mlip_csv` reads r0 and
    eps0*V, so stress rows were 1e4 times too heavy.  It now writes r0 and
    eps0*V.
  - Stale numba caches in its __pycache__ made every batch directory fail
    with "No module named '<dynamic>'".
- External descriptors.  Of the 5006 columns, only about 190 vary over the
  training E rows.  ADF1/2 and ODF1/2 are constant: the ADF/ODF subtypes
  use the central atom's per-element rc1 (1.5 A for Ge), whereas RDF and
  CN use the pair sum.  Z-scoring or whitening the near-constant columns
  made ELM, QELM and NN blow up, with test F of 12-26 eV/A.  Cutting the
  table to columns with std > 1e-4, with z-score plus reg_search for ELM
  and QELM, gives test F of 0.17-0.23 eV/A (mlip/ext_sweep).
- relax with FIRE and relax_cell falls into limit cycles on this data,
  while L-BFGS converges on the same input.  L-BFGS is the judged run, and
  the FIRE fault was handed off as a separate task.  With L-BFGS, 117/190
  test relaxations are within 5 % of the DFT final volume.  Most failures
  are MLIP collapse under extrapolation: the energy runs away and the cell
  shrinks until the minimum-image guard stops the run.  The training data
  hold near-equilibrium vacancy cells only.  Atom-centred ELM with
  isolated_atom_energy passed 9/10, against 2/10 without it.
- Stress (example/GeSn/stress_check.py, test set, DFT std 2.79 GPa):
  - ACSF MLIPs: RMSE 0.17-0.35 GPa, R^2 0.985-0.996.
  - External-descriptor models: 0.26-0.42 GPa.
  - SW: 0.91 GPa, R^2 0.89.
  - Tersoff: 1.33 GPa, R^2 0.77, with a systematic negative offset of
    about 2 GPa on the diagonal.  This matches its 3-8 % volume shrinkage
    in relax.
