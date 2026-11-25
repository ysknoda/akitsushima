---
status: accepted (2026-11-25, 0.4.0a1)
---

# mdspec P1: equivalence port of nozuchi 0.2.0a2

First stage of the nozuchi port (ADR-0050 decision 18).  The code of
nozuchi 0.2.0a2 (commit 84dec4c) moves to `akitsushima/mdspec/` without
behaviour changes.  The new command `akitsushima mdspec` reads
akitsushima-style YAML.

## Decisions

1. **Modules.**
   - Copied as they are: charges, check, config, constants, dipole,
     fitting, forces, modes, modespec, optics, output, spectrum and
     trajectory.
   - The package `__init__` keeps nozuchi's `run`, which takes the
     internal (nozuchi-format) configuration.
   - nozuchi's `cli` is not ported; cli/cmd_mdspec.py replaces it.
2. **One OUTCAR reader.**
   - mdspec/charges.py re-exports `read_outcar_eps_inf`,
     `read_outcar_born`, `map_to_unit_cell` and `rigid_shift` from
     io/outcar_dielectric.py, which ADR-0049 ported unchanged in
     numbers.
   - The two warning texts of the site mapping, which ADR-0049 had
     shortened, are nozuchi's again.
   - `read_outcar_eps_inf` also accepts the LCALCEPS header "...
     (including local field effects)" when the DFPT one ("... in DFT")
     is absent (ADR-0050 decision 7).
3. **YAML** (mdspec/yaml_io.py).
   - The akitsushima form puts eps_inf in `born.epsilon_inf` and the Born
     charges in `born.born_charges`, as every other command does.
     `map_frames` moves to `mdspec.dipole`.  Everything else is nozuchi's
     schema under `mdspec:`.
   - `to_internal` and `from_nozuchi` are inverses.
   - Refused: `mdspec.epsilon_inf`, `mdspec.dipole.born`, and
     `map_frames` without `born_charges`.
   - `born.born_charges` with the point_charge model is warned about as
     unused.
   - Relative paths are read against the YAML file's directory, as in
     nozuchi.
4. **Command.**
   - `akitsushima mdspec input.yaml`; `--check [--energy ...]
     [--target]` (exit 1 on WARN); `--convert-nozuchi` (prints the
     akitsushima YAML); `--dump-default mdspec`.
   - Input errors (ConfigError is a ValueError) end as akitsushima's one
     ERROR line, exit 1; run-time errors keep their traceback.  nozuchi
     used exit 2 and a one-line message.
   - Under MPI, rank 0 only.
5. **Small adaptations (no numerical effect).**
   - Default output `mdspec_output/mdspec`.
   - The results JSON carries `akitsushima_version` and
     `nozuchi_base: "0.2.0a2"` instead of `nozuchi_version`.
   - The numba switch is akitsushima's `NUMBA_ENABLE=0` (was
     `NOZUCHI_NUMBA=0`).
   - Without matplotlib the figure is skipped with one warning
     (akitsushima's optional-plotting rule) instead of an ImportError.
6. **Tests.**
   - nozuchi's suite is in tests/mdspec/, as a package; its data are
     shared with ADR-0049's tests/data/nozuchi.
   - test_cli is rewritten for the akitsushima command: converter round
     trip for the three nozuchi templates, the translation refusals, the
     shipped template, the end-to-end run with the dipole cache, and the
     error paths.
   - test_check runs `--check` on the converted input.
   - Tests of 0.5 s or more, or sharing a costly fixture, are marked
     `slow` in tests/mdspec/conftest.py: 23 of 142.
7. **Glossary.**  nozuchi's terms are merged into CONTEXT.md ("MD
   spectra").  Mapping where akitsushima already had a term:
   "Unwrapping" -> unwrapped coordinates; "check" -> `mdspec --check`;
   "Born effective charge" -> Born charge (Z*); "eps_inf" as in
   `born:`.  nozuchi's "Trajectory" and "frame" are kept, and "Run"
   gets an mdspec qualifier.

## Verification

- **Ported suite.** tests/mdspec: 142 passed (nozuchi's 138, with
  test_cli reshaped) in 42 s; 119 in 1.3 s without the slow ones.
- **Template.** test_cli_templates runs `--dump-default mdspec` on a
  generated Langevin Mg/O XDATCAR and thermo.log.
- **Equivalence gate** (example/mdspec/equivalence_nozuchi.py; ADR-0050
  decision 16).
  - Scope: every nozuchi validation input (58 units).
    - Tier 2: 27 akitsushima classical MD runs, MgO / ZnO / TiO2,
      2400-4800 atoms, mode estimator with the npz Gamma modes.
    - Tier 3: 15 VASP AIMD seeds x 2 DFPT meshes, SiC / ZnO / ice XI /
      urea, born_tensor from OUTCAR.
    - The SPC/E water run.
  - Method: each unit is run by nozuchi 0.2.0a2 (its input as is) and by
    mdspec (the converted input, the user's path).
  - Comparison: every number of the results JSON and every array of the
    .npz, against |a - b| <= 1e-10 max(|a|, |b|).  Wall time is
    excluded.
  - Result: 58 / 58 pass, bit for bit (largest relative difference 0.0
    in both JSON and npz).  Wall time: nozuchi 14.6 min, mdspec 14.5 min
    in total.
  - Found on the first pass: 18 units (the Tier 3 runs with a rigid
    shift) differed only in a warning text.  ADR-0049's port of the
    site mapping had shortened two messages.  Both were restored to
    nozuchi's wording, and the 18 units were rerun and passed.
- **Full suite.** 1351 passed, 1 skipped (16 min; was 1208 + the 142 ported + the template case).

## Consequences

- `akitsushima mdspec` gives everything nozuchi 0.2.0a2 gave.
- nozuchi is frozen.  Its README pointer to akitsushima waits for the
  user's go (ADR-0050 decision 1).
- Next: P2 (in-house Gamma modes, extxyz / lammps-dump readers, `md`
  velocities, `dipole.model: auto` and the response models), on the
  user's go-ahead.
