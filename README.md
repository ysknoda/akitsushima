# akitsushima

> **Alpha release.** This project is under active development.
> Interfaces, behavior, and output formats may change without notice
> between releases, and it has not yet been broadly tested. Do not
> rely on it for production use. Computed values are not comparable
> across these pre-releases. 0.2.0a1 stopped halving Tersoff energies,
> forces and stresses, gave Buckingham and Born-Mayer the force-shifted
> truncation their sibling pair potentials already had, and chose the
> Ewald and PPPM cutoffs from the accuracy you ask for. 0.2.0a2 makes the
> single-species Stillinger-Weber format honour `gamma` (it was ignored;
> the default is now SW's own 1.20), removes a spurious sign flip from
> `VASP_KB_TO_EV_ANG3`, fixes variable-cell RFO, and now refuses
> multi-species SW parameters and `fit` optimiser choices it used to
> accept and get wrong. 0.2.0a3 fixes phonon frequencies (minimum-image
> force constants, the ASR and the unit conversions), the GCMC deletion
> acceptance, and the Green-Kubo heat flux and seeding (ADR-0003); and
> changes every MLIP fit (ADR-0004): ELM / QELM / NN now learn one energy
> whose derivatives give the F and S rows (chain rule; `row_independent:
> true` restores the old per-row form), all models weight the squared
> residual by w_cat x w_unit^2 x w_user (the ELM Uncertainty Weighting is
> gone). ELM / QELM now add a direct linear term by default
> (`direct_link`), and NN defaults to softplus with a direct linear term.
> LightGBM has since been removed (ADR-0012): trees give no conservative
> force, so it could never drive a solver. 0.2.0a4 makes Tersoff forces
> analytic (they change by ~1e-7 eV/A) and fits multi-species Tersoff and
> the Zhou (2004) alloy EAM (ADR-0015). 0.2.0a5 makes an MPI `fit` print
> and write its files on rank 0 only. 0.2.0a6 adds the `elastic` command
> (ADR-0017): clamped- and relaxed-ion elastic tensors from a
> multi-amplitude stress-strain fit, B / G / E / nu as Voigt, Reuss and
> Hill averages, Born stability, sound velocities, Debye temperature and
> the internal strain. 0.2.0a7 adds harmonic thermodynamics to `phonon`
> (ADR-0018): F, U, S, C_v and Theta_D(T) by a Gamma-centred q-mesh mode
> sum (F_vib too), and fixes the sound-velocity Debye temperature, which
> lacked the 1/3 of the Debye mean and was 0.69x too low. 0.2.0a8 adds
> constant-volume cell relaxation (`relax.fix_volume`), the `gruneisen`
> command (mode Grueneisen parameters from +-1 % volume force constants)
> and the `qha` command (thermal expansion, B(T), C_p, thermodynamic
> gamma), and the Vinet equation of state. 0.2.0a9 adds Born charges
> (`born:` block), the Gonze-Lee non-analytic correction (LO-TO splitting)
> in phonon / gruneisen / qha, and the `dielectric` (eps_0, IR modes) and
> `piezo` (relaxed-ion e, d, k) commands (ADR-0019). 0.2.0a10 adds the
> `nemd` command (Langevin heat baths, temperature profile, local heat
> flux, kappa and Kapitza resistance, checkpoint / restart) and the
> species / region decomposition of the Green-Kubo kappa (ADR-0020).
> 0.2.0a11 adds VASP readers (vasprun.xml / OUTCAR, every ionic step) and
> the `dataset` command, which turns VASP runs into a `fit` / `mlip`
> training set split by run (ADR-0021). It also changes `relax` with FIRE,
> the default method. FIRE no longer shrinks its velocity when a step is
> clamped: that had defeated its uphill reset and trapped variable-cell
> runs in limit cycles. Runs whose steps are never clamped are unchanged.
> `write_table` now writes 17 significant digits plus a test-set table,
> and numba no longer aborts when torch is imported first.
> 0.2.0a12 adds the `anneal` command (ADR-0022): MD-quench annealing on a
> temperature schedule, or basin hopping, with every quench a `relax`,
> optional pressure (enthalpy), checkpoint / restart and a list of
> distinct minima. `md`'s NVT / NPT classes take `initial_velocities`.
> The NN models (tabular and atom-centred) take the ELM's input
> standardisation keys (`standardize: zscore | pca`, `pca_eps`,
> `pca_drop`). The NN default stays z-score.
> 0.2.0a13 adds `min_feature_std` to every MLIP model (linear, ELM /
> QELM, NN, tabular and atom-centred): input columns whose training std
> is below it are masked (stored as an infinite std, so they add nothing
> to E, F or S).  The default 1e-12 masks nothing; 1e-4 replaces the
> external column reduction needed for near-constant columns of external
> descriptor tables (ADR-0010, open finding A4).
> 0.2.0a14: a machine-learning potential (and an external code) is no
> longer held to the solvers' neighbour-list minimum-image check, which
> it never used: relax, md, neb, gcmc and elastic refused cells thinner
> than 2 (r_cut + skin) although the ACSF handles them (ADR-0024).  The
> CLI's `--version` prints the version; the Python API is verified
> against the CLI (ADR-0023).
> 0.2.0a15 (ADR-0024): every MLIP model takes `baseline=` and
> `isolated_atom_energy=`; tabular models have `fit_structures()`;
> `akitsushima.io.load_training_set()` / `build_training_set()` and
> `akitsushima.force_fn_from_config()` are public.  API defaults follow
> the CLI's (tabular ELM `eps0` 0.01, relax `output_dir` ./relax_output,
> ...); the CLI's NVT `tau_nhc` falls back to 100 fs like the API.
> 0.2.0a16 (ADR-0026): `dataset.isolated_atom_energy` also takes the
> path of the atom's VASP run (the energy of its last step, read with
> `dataset.energy`).  Which atom to use is the user's choice; for
> ISPIN = 1 training data the lowest spin-polarised atom is recommended
> (ADR-0026 shows why the ISPIN = 1 atom is the worse choice for a
> finite-cutoff MLIP).

A classical force-field and machine-learning potential toolkit for atomistic
simulation. `akitsushima` fits interatomic potentials, relaxes structures,
runs molecular dynamics, and computes phonon / thermal-transport properties
for crystalline and molecular systems — from the command line or as a Python
library.

## Features

### Force fields

**Pair potentials** (ionic / van der Waals)
- Buckingham — `A exp(-r/rho) - C/r^6`
- Born-Mayer — `A exp(-r/rho)`
- Lennard-Jones — `4epsilon [(sigma/r)^12 - (sigma/r)^6]`
- Morse — `D_e [1 - exp(-alpha(r-r0))]^2`

**Many-body potentials** (covalent / metallic)
- Stillinger-Weber — 2-body + 3-body angle term (Si, Ge, H2O)
- Tersoff — bond-order (`b_ij`) with coordination (Si, C, Ge, GaAs);
  multi-species as a LAMMPS-style (i, j, k) table or through the
  Tersoff (1989) mixing rules
- EAM — embedding `F(rho_i)` + pair `phi(r)` (metals): the analytic
  alloy form of Zhou, Johnson & Wadley (2004), or a setfl table
  (LAMMPS `eam/alloy`)

**Long-range Coulomb**
- Ewald summation — vectorised, `O(N^1.5)` for N ≲ 2,000
- PPPM / SPME — FFT mesh, `O(N log N)` for N ≳ 500
- ESM — Effective Screening Medium (open / metallic boundary conditions)

All three are chosen in the potential section and reach every solver
through the same force function (ADR-0008):

```yaml
potential:
  type: buckingham
  potential_json: ./fitting_output/fitted_potential.json
  r_cut: 5.5
  coulomb_method: esm          # ewald | pppm | esm
  esm:
    bc: bc1                    # bc1 vacuum|slab|vacuum, bc2 metal|slab|metal,
                               # bc3 vacuum|slab|metal (Otani-Sugino / QE)
    sigma: 0.5
    n_grid_xy: 32
    n_grid_z: 512
    external_field: 0.0        # V/A (bc2)
```

With ESM the slab is centred at fractional z = 0 (the medium starts at
fractional z = 1/2), as in Quantum ESPRESSO.  relax may relax only a_x and
b_y; NPT and kappa refuse ESM; phonon needs supercell [N, N, 1]; gcmc
inserts charged species only with bc2 / bc3.  The z-grid error is
O(dz^2 q^2): ~20 meV per +-1 ion at dz = 0.125 A, 1.3 at dz = 0.031 A.

**Machine-learning potentials**
- `ELMModel` — Extreme Learning Machine (single hidden layer)
- `QuantizedELMModel` — ELM with quantised hidden-layer weights
- `LinearModel` — linear model: OLS, ridge, smoothness-penalised ridge, LASSO, Elastic Net, PLS, Bayesian ridge, ARD
- `NNModel` — PyTorch feed-forward network (lazily imported)
- `AtomicELMModel` / `AtomicQELMModel` / `AtomicNNModel` — atom-centred (Behler-Parrinello) models E = sum_i eps_s(G_i), one function per species (ADR-0005)
- `ACSF` — atom-centred symmetry functions G2 / G4 / G5 with analytic position and cell derivatives

### Simulation capabilities

| Task | Description |
|---|---|
| `fit`   | Fit classical or ML potential parameters against DFT reference data (POSCAR geometry + CSV energy/force/stress). |
| `relax` | Geometry optimisation: FIRE, L-BFGS, DIIS, RMM-DIIS, RFO — ionic or variable-cell, with transition-state refinement. |
| `md`    | Molecular dynamics: NVE, NVT (Nosé-Hoover chain), NPT. |
| `anneal` | Global structure search: MD-quench annealing on a temperature schedule, or basin hopping (ADR-0022). |
| `gcmc` / `osmc` | Grand-canonical (μVT) MC/MD: discrete insertion/deletion (`gcmc`) or a continuous fractional atom, CFCMC (`osmc`, ADR-0009). |
| `elastic` | Elastic tensor (clamped- and relaxed-ion), B / G / E / ν as Voigt, Reuss and Hill averages, Born stability, sound velocities, Debye temperature, internal strain (ADR-0017). |

### Post-processing

- **CI-NEB** — Climbing-image nudged elastic band
- **`TransitionStateAnalyser`** — Hessian, IRC, saddle-point verification, TST reaction rate
- **`PhononCalculator`** — Finite-displacement force constants, band structure, DOS, Debye temperature, `thermal_properties` (F, U, S, C_v, Theta_D(T); ADR-0018)
- **`ThermalConductivityCalculator`** — Green-Kubo MD + Slack-Debye estimate
- **`ElasticConstantsCalculator`, `elastic_moduli`** — elastic tensor and polycrystalline moduli (ADR-0017)

### Performance

- **`NeighbourList`** — cell-linked list / SciPy `cKDTree` back-end, with a Verlet skin for lazy rebuilds during MD.
- **Numba JIT** — optional `@njit` compilation of the pair / many-body loops,
  used automatically whenever numba is importable, with a transparent NumPy
  fallback when it isn't. Measured on 64-atom silicon: Stillinger-Weber
  66 → 4.2 ms/step, Tersoff 117 → 2.5 ms/step. Set `NUMBA_ENABLE=0` to force
  the NumPy path.
- **MPI (mpi4py)** — domain-split + `Allreduce` parallelism across ranks, available for every potential, for NEB image evaluation, phonon finite displacements, and the fitting loss function — with a transparent single-process fallback when mpi4py isn't installed.

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .
```

The only hard dependencies are NumPy and SciPy. Everything else is an
optional extra, installed as needed:

```bash
pip install -e ".[numba]"       # JIT-compiled potential kernels
pip install -e ".[mpi]"         # multi-rank parallelism (requires mpi4py/mpirun)
pip install -e ".[torch]"       # NNModel
pip install -e ".[yaml]"        # YAML config files for the CLI
pip install -e ".[plotting]"    # matplotlib output (NEB/phonon/relax plots)
pip install -e ".[all]"         # everything above
```

`NNModel` / `AtomicNNModel` are imported lazily — `import akitsushima`,
and every other class in the package, never pulls in `torch` unless you
actually reference one of them.

## Quick start

### Command line

Every task reads a single YAML input file (`fit` instead reads a training
CSV plus a JSON/YAML fitting config). Generate a starting template with
`--dump-default`:

```bash
akitsushima --dump-default relax > relax.yaml
akitsushima relax relax.yaml

akitsushima --dump-default nvt   > md_nvt.yaml
akitsushima md md_nvt.yaml
```

Run `akitsushima --help` for the full list of tasks and options.

### Python API

```python
import numpy as np
from akitsushima import (
    PotentialEngine,
    LennardJonesPotential,
    NeighbourList,
    make_force_fn,
)

# Lennard-Jones argon: sigma = 3.405 A, epsilon = 0.0103 eV
# (Verlet, Phys. Rev. 159, 98 (1967)); r_cut = 10 A = 2.94 sigma.
lj_params = {frozenset({"Ar"}): {"epsilon": 0.0103, "sigma": 3.405}}

# The cell edge must exceed 2 * (r_cut + skin) = 2 * (10.0 + 0.5) = 21 A.
lattice = np.eye(3) * 24.0
# Two atoms 4.0 A apart, just beyond the minimum at 2^(1/6) sigma = 3.82 A.
positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
atom_types = ["Ar", "Ar"]

engine = PotentialEngine().add(LennardJonesPotential(lj_params, r_cut=10.0))
fn = make_force_fn(engine)

nl = NeighbourList(r_cut=10.0)
nl.build(lattice, positions)

energy, forces, stress = fn(lattice, positions, atom_types, nl)
# energy = -9.4170e-03 eV (force-shifted LJ, the default shift=True)
```

### Python API: a machine-learning potential

Training data, an atom-centred ELM on ACSF descriptors, and the model as a
force field (the same steps as `akitsushima mlip` and `akitsushima relax`):

```python
import numpy as np
import akitsushima as ak
from akitsushima.fitting import AtomicELMModel
from akitsushima.fitting.descriptors import ACSF

# Training data.  Real data come from DFT: akitsushima.io.build_training_set
# turns VASP runs into a training set and akitsushima.io.load_training_set
# reads it back as the structure dicts below.  Here Stillinger-Weber silicon
# (Stillinger & Weber, PRB 31, 5262 (1985)) stands in for DFT.
sw = ak.force_fn_from_config({"type": "SW", "species": ["Si"], "params": {
    "epsilon": 2.1683, "sigma": 2.0951, "a": 1.80, "gamma": 1.20, "lambda": 21.0,
    "A": 7.049556277, "B": 0.6022245584, "p": 4.0, "q": 0.0, "costheta0": -1 / 3}})
diamond = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                    [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])
cells = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
frac = (diamond[None] + cells[:, None]).reshape(-1, 3) / 2       # 64-atom 2x2x2 cell
rng = np.random.default_rng(0)
structures = []
for _ in range(8):
    lattice = np.eye(3) * 10.862 * rng.uniform(0.98, 1.02)
    positions = frac @ lattice + rng.normal(0.0, 0.05, frac.shape)
    types = ["Si"] * len(positions)
    e, f, s = sw(lattice, positions, types)
    structures.append({"lattice": lattice, "positions": positions, "atom_types": types,
                       "ref_energy": e / len(types),        # eV/atom
                       "ref_forces": f,                     # eV/A, (N, 3)
                       "ref_stress": s})                    # eV/A^3, compressive-positive

# Descriptor and model.  A pair baseline (e.g. ZBL) and isolated-atom
# energies go in the constructor: AtomicELMModel(baseline=..., isolated_atom_energy=...).
acsf = ACSF.from_grid(["Si"], 3.771, g2={"eta": [4.0], "rs": [2.2, 2.6, 3.0, 3.4]},
                      g4={"eta": [0.05], "zeta": [1, 2], "lambda": [-1, 1]})
model = AtomicELMModel(n_hidden=64, seed=42, verbose=False).fit(structures, acsf)
print(model.score_structures(structures, acsf)["F"]["RMSE"], "eV/A training force RMSE")
model.save("si_model.npz")

# The model is a force field for every solver.
fn = ak.make_force_fn(model)
result = ak.GeometryOptimiser(structures[0]["lattice"], structures[0]["positions"],
                              structures[0]["atom_types"], force_fn=fn, relax_cell=True,
                              fmax=0.01, verbose=False).run()
print(result["energy"] / 64, "eV/atom after relaxation")
```

## CLI reference

| Command | Purpose |
|---|---|
| `relax`  | Geometry optimisation (FIRE / L-BFGS / DIIS / RMM-DIIS / RFO) |
| `md`     | Molecular dynamics (NVT / NVE / NPT) |
| `anneal` | MD-quench annealing or basin hopping; distinct minima, restartable |
| `gcmc`   | Grand-canonical Monte Carlo MD (μVT) |
| `neb`    | Climbing-image NEB transition-state search |
| `tsa`    | Transition-state analysis (Hessian / IRC / TST rate) |
| `osmc`   | CFCMC grand-canonical MC/MD (fractional atom) |
| `esm`    | Effective screening medium |
| `phonon` | Phonon band structure, DOS, Debye temperature, F / U / S / C_v (mode sum) |
| `kappa`  | Thermal conductivity (Green-Kubo + optional Slack-Debye) |
| `elastic` | Elastic tensor, B / G / E / ν (Voigt / Reuss / Hill), Born stability, Debye temperature |
| `gruneisen` | Mode Grüneisen parameters (q-path, mesh, C_v-weighted mean γ(T)) |
| `qha`    | Quasi-harmonic thermal expansion, B(T), C_p, thermodynamic γ(T) |
| `dielectric` | Static dielectric tensor, LO-TO splitting, IR modes (Born charges) |
| `piezo`  | Relaxed-ion piezoelectric e, d and coupling k (rigid ion) |
| `nemd`   | Non-equilibrium MD: T profile, local heat flux, kappa, Kapitza resistance |
| `mlip`   | Fit a machine-learning potential (ELM / QELM / NN / linear) |
| `fit`    | Fit a classical potential (LJ / Mie / Morse / Born-Mayer / Yukawa / Buckingham / Stillinger-Weber / Tersoff / EAM) |
| `dataset` | `fit` / `mlip` training set from VASP runs (vasprun.xml / OUTCAR), split by run |

### Elastic constants (ADR-0017)

```bash
akitsushima --dump-default elastic > elastic.yaml   # Tersoff Si template
akitsushima elastic elastic.yaml
```

The cell and ions are relaxed first (`pre_relax`), then each of the six
Voigt strains is applied at +-0.25, 0.5 and 1 % (`strains`).  Each column
of C is the slope of a straight-line fit of the stress (RMS residual
reported per mode); the clamped-ion C^0 uses affinely strained cells, the
relaxed-ion C relaxes the ions in each.  The POSCAR must already be a
supercell with r_cut + skin below half its smallest perpendicular width
at every strain -- this is checked and refused, never extended
automatically.  ESM electrostatics is refused (straining moves its
boundaries).

Output (`elastic_output/`): `elastic_results.json` (C, C^0, S, eigenvalues
and Born stability, B / G / E / nu for Voigt, Reuss and Hill -- E and nu
from each scheme's own B and G, as pymatgen -- Pugh ratio, universal
anisotropy A^U, density, v_l / v_t / v_m, Debye temperature, internal
strain du/deta in A per unit strain, and an input hash),
`elastic_tensor.dat`, `internal_strain.dat` and `POSCAR_reference`.
`example/elastic/check_elastic.py` checks Tersoff Si against Tersoff's
published C11 / C12 / C44 / C44^0 and Kleinman zeta, ASE's own Tersoff
and pymatgen, and ZnO (Buckingham + PPPM) against the hexagonal relations.

### Grüneisen parameters and the quasi-harmonic approximation (ADR-0018)

```bash
akitsushima --dump-default gruneisen > g.yaml   # Tersoff Si, primitive POSCAR
akitsushima gruneisen g.yaml
akitsushima --dump-default qha > q.yaml
akitsushima qha q.yaml
```

Both start from the primitive cell (as `phonon`), relax it fully
(`pre_relax`), then relax shape and ions at each fixed volume in the
phonon supercell (`GeometryOptimiser(fix_volume=True)`, also available as
`relax.fix_volume`).  `gruneisen` uses V0 (1 -+ 0.01): gamma =
-(V/2w^2)<e|dD/dV|e> along the q-path and on a mesh, with the
C_v-weighted mean gamma(T).  `qha` scans 7 volumes over +-5 %, fits
E(V) + F_vib(V, T) per temperature (Birch-Murnaghan or Vinet) and reports
V(T), B(T), alpha(T), C_v, C_p and gamma(T); it stops on an imaginary mode
(below -0.05 THz) unless `skip_unstable_volumes: true`, and reports
nothing above the temperature where V(T) leaves the scanned volumes.
Both match phonopy's own Grueneisen and QHA (tests/test_qha.py).

### Born charges, dielectric tensor and piezoelectric response (ADR-0019)

```yaml
born:                         # top level; also turns on NAC in phonon/gruneisen/qha
  epsilon_inf: 3.7            # scalar, [xx, yy, zz] or 3x3; default 1 (warned)
  born_charges:               # default: the potential's point charges
    per_species: {Zn: 2.1, O: -2.1}       # scalar, diagonal or 3x3
    per_atom: {4: [[...], [...], [...]]}  # 1-based POSCAR index, wins
  clamped_ion_piezo: [[...6...], [...], [...]]   # C/m^2, piezo only
```

Z[k, a, b] = dP_a / du_kb (phonopy's and VASP's order); the sum rule is
imposed.  `akitsushima dielectric` gives eps_0 = eps_inf + (4 pi / V)
sum_m S_m / w_m^2 with the per-mode oscillator strengths, mode effective
charges, IR intensities and LO frequencies along x, y, z.  `akitsushima
piezo` takes the internal strain of `elastic` (or `load_elastic`) and
gives e (C/m^2), d (pC/N) and k.  The model is rigid ion: with point
charges only, eps_inf = 1 and the electronic (clamped-ion) piezoelectric
term is zero unless given.  Without charges or a `born:` block (MLIP,
EAM, Tersoff, SW) both commands refuse.

### NEMD and the decomposition of kappa (ADR-0020)

`akitsushima nemd` places a hot and a cold Langevin bath (fractional slabs
along a lattice vector) in a periodic cell, so heat flows both ways
round: J = Qdot / (2A).  It reports the temperature profile, the local
heat flux per bin (from the per-atom heat currents), kappa = J / |dT/dz|
from each of the two segments with block errors, the Kapitza resistance
of interfaces you name, and steady-state indicators (bath balance,
profile drift, gradient sign).  kappa depends on the length: run several
and extrapolate 1/kappa against 1/L (example/nemd).  Long runs: launch
detached; the checkpoint (`checkpoint_every`) lets a re-run continue bit
for bit.

`kappa.decompose: {species: true, regions: {axis: c, bounds: [0, 0.5, 1]}}`
splits the Green-Kubo tensor into kappa_g = (V/k_B T^2) int <J_g(0) J(t)>,
which add up to kappa exactly.

### Global structure search (ADR-0022)

`akitsushima anneal anneal.yaml` (template: `--dump-default anneal`).
Cycle 0 relaxes the input, which is the relax-only reference. Then:

- `method: md_quench` runs one continuous NVT or NPT trajectory from that
  relaxed structure. The trajectory follows a temperature schedule
  (constant, linear, geometric or a list), and a copy of the structure
  is relaxed every `quench_every` steps.
- `method: basin_hopping` (Wales & Doye 1997) displaces every atom
  inside a sphere and, with `relax_cell`, strains the cell. It relaxes
  the trial and accepts it by the Metropolis rule on the relaxed energy.
  The step adapts towards 50 % acceptance.

The `anneal.relax` and `anneal.md` blocks take the `relax` and `md` key
names. With `pressure` set, every quench minimises H = E + PV.

A cycle fails when the solver raises, the energy is not finite, or the
energy drops below `min_energy_per_atom`, which guards against an MLIP
that collapses under extrapolation. A failed cycle is logged and
skipped, and `max_consecutive_failures` failures in a row stop the run.

Every cycle is appended to `anneal_log.jsonl` and checkpointed, so
`restart: true` continues an interrupted run and reproduces the
uninterrupted one. Distinct minima go to `POSCAR_rank01..` and
`minima.csv`, the lowest to `POSCAR_best`, and the history to
`anneal.png`.

### Fitting a classical potential

```bash
akitsushima fit --potential lj --write-config fit_lj.yaml
# edit fit_lj.yaml, then:
akitsushima fit --potential lj --config fit_lj.yaml --training-csv training.csv
```

The training CSV has one row per energy/force/stress component:

```
col 0  target value  [eV/atom | eV/A | eV/A^3]
col 1  category       "E" | "F" | "S"
col 2  struct_idx     integer structure index
```

Geometry for each `struct_idx` is read from `poscar_dir/<struct_idx>/POSCAR`.

#### Optimisation

Fitting runs in two stages, both driven from the config file:

1. **Global search (optional)** -- `use_de: true` runs scipy's
   Differential Evolution over the parameter bounds first
   (`de_maxiter`, `de_popsize`; `--use-de` on the command line does the
   same).  The `--write-config` templates switch it on for `sw` only:
   for `tersoff` it settled in a wrong basin along the beta / c / d
   directions the data cannot resolve, where the local search alone
   recovers the potential (ADR-0007).  Its cost is `de_popsize x n_params` loss evaluations
   per generation, so it is practical for a handful of parameters, not
   for tens.
2. **Local refinement (always)** -- `local_method` picks a bounded
   scipy optimiser (default `Nelder-Mead`, case-insensitive):

| `local_method` | uses a gradient | notes |
|---|---|---|
| `L-BFGS-B`    | finite-difference | former default; stalls in narrow valleys |
| `TNC`         | finite-difference | truncated Newton |
| `SLSQP`       | finite-difference | |
| `Nelder-Mead` | none | default; adaptive simplex; many evaluations |
| `COBYLA`      | none | linear trust-region model |
| `COBYQA`      | none | quadratic trust-region model |
| `least_squares` | finite-difference Jacobian of the residuals | scipy `least_squares` (`dogbox`) on the residual vector the loss sums; squared loss only |

All six respect the parameter bounds and receive the same stopping
intent in their own option names (at most 5000 iterations -- 100 000
for Nelder-Mead -- and 100 000 loss evaluations, loss tolerance 1e-12
eV^2, gradient and step tolerances 1e-8).  Nelder-Mead stops on the step
tolerance alone (a simplex ~1e-8 relative in every parameter): scipy's
loss tolerance is absolute, so a fit whose minimum is not zero never
met it and ran all 100 000 evaluations.  Any other name is refused
before any data is loaded: `BFGS` and `CG` ignore bounds, `Newton-CG`,
`trust-*` and `dogleg` need an analytic gradient, which `fit` does not
provide, and scipy's bounded `Powell` returned points worse than ones
it had itself evaluated on real fit losses.

`least_squares` (ADR-0015) minimises the vector of E / F / S residuals
whose squares the loss sums, with a Jacobian from forward differences and
the variables rescaled by it.  It is the method for ill-conditioned fits:
on the Ni-Al Zhou-EAM test (loss Hessian condition number ~1e11 in a
curved valley) it reached loss 7e-14 in about 3 minutes, where
Nelder-Mead with restarts stopped at 2.7e-4 after 8.8 hours; on the
multi-species Tersoff tests it reached ~1e-15.  The multi-species
`--write-config` templates use it.

Three settings apply to any method:

- `restarts: N` reruns the local search from the best point (Nelder-Mead
  from a randomly rotated simplex, `restart_seed`); it stops after two
  consecutive restarts that improve the loss by less than `restart_rtol`
  (default 1e-3, relative).
- `local_maxiter` caps each local run, so a slow run is restarted.
- `time_limit_hours` ends the search on wall-clock time (sleep included)
  and keeps the best point evaluated so far.

`Nelder-Mead` is the default because it was the only method that
reached the minimum (loss ~1e-14) on every real fit loss measured: SW Si
from the DE result and from a 10-25% perturbed start, and the 19-variable
GeSn SW fit.  `L-BFGS-B`, the former default, stopped at 6.6e-5, 2.3e-3
and 1.9e4 on those (it also stalled at 14 on an LJ test where the others
reached 1e-14): its finite-difference gradient and line search end early
in narrow curved valleys and at points the loss scores as failed.  The
price is evaluations: 800-1700 for 5 SW parameters, ~32 000 for 19.
`COBYQA` is the next choice for a handful of parameters.

#### Multi-species Tersoff and EAM (ADR-0015)

The pair potentials take one record per `A-B` pair and were always
multi-species.  Tersoff and EAM now are too:

```bash
akitsushima fit -p tersoff --write-config t.json --template tersoff1989  # Si-C
akitsushima fit -p tersoff --write-config t.json --template records      # AlN
akitsushima fit -p eam     --write-config e.json                         # Ni-Al
```

- **Tersoff, 1989 mixing** (`initial_tersoff: {mixing: tersoff1989,
  elements: {...}, chi: {...}}`): per-element A, B, lambda, mu, c, d, h
  and one chi per pair are fitted (Tersoff, Phys. Rev. B 39, 5566
  (1989), eqs. 1a-1e); `fit_keys: [chi]` is the paper's own procedure.
- **Tersoff, records** (a list of `i, j, k` records, the LAMMPS table):
  the pair terms of (i, j, j) and (j, i, i) are one variable; c, d, h per
  record, or per centre element with `tie: centre`.
- Train on more than one coordination: Tersoff's angular parameters c
  and d are only determined through c / d^2 when every angle is near
  tetrahedral, and an unseen rocksalt or sheet phase then fails; adding
  rocksalt cells to the training data fixed it (ADR-0015).  Likewise an
  EAM needs expanded and compressed cells: Zhou's low-density embedding
  branch (rho < 0.85 rho_e) is otherwise left undetermined.
- In both forms **beta and n are fixed by default**: with near-tetrahedral
  training angles they trade off almost exactly against c and d, and
  fitted they drifted by orders of magnitude without improving the loss
  (ADR-0015).  Add them to `fit_keys` to fit them anyway.
- **EAM** (`initial_eam: {form: zhou2004, elements: {...} | [Ni, Al]}`):
  A, B, alpha, beta and the embedding coefficients Fn2, Fn3, F2, F3, eta
  per element; Fn0, Fn1, F0, F_e follow from the continuity of F and
  dF/drho, which the paper requires.  Each fit also writes
  `fitted.eam.alloy` (setfl), usable as `potential: {type: EAM, setfl:
  fitted.eam.alloy}`.
- Published tables: Tersoff (1989) Table I and Zhou (2004) Table III in
  full (`akitsushima.engine.published_parameters`), with two misprints
  corrected (carbon d, Al A) and the rows whose embedding function is
  discontinuous as printed flagged (Pt, W; ADR-0015).

### Fitting a machine-learning potential

`akitsushima mlip mlip.yaml` (template: `akitsushima mlip --dump-default`)
trains on a table of E, F and S rows (`load_mlip_csv`):
`target, label, weight, <descriptors>`, where an E row holds the structure
descriptor d (target E/N), an F row X = -dd/dr and an S row
X = -(1/V) dd/dT.  Per `docs/adr/0004`:

- **ELM / QELM / NN** learn E/N = f(d) and fit every F and S row as
  N grad f(d) . X of its own structure (chain rule), so one energy function
  gives E, F and S.  `row_independent: true` restores the pre-0.2.0a3
  per-row regression, which is not a force field.
- **linear** is E/N = w . d + b; with d the per-atom average of atomic
  descriptors it is an atom-centred linear potential.
- Every model weights the squared residual by w_cat x w_unit^2 x w_user.

**Direct linear term (`direct_link`, on by default for ELM / QELM / NN).**
Besides the hidden layer, the input descriptor enters the output linearly:

    ELM / QELM :  f(x) = beta . g(W x + b) + gamma . x + beta_0
    NN         :  f(x) = MLP(x) + w . x

with x the standardised descriptor (d for the tabular form, the atom's
G_i for the atom-centred one, one set of weights per species).  For the
ELM this is the random-vector functional-link (RVFL) network: W and b are
random and fixed, and beta, gamma and beta_0 are solved together in one
least-squares system.  The linear part the data contain is then fitted
exactly, and the hidden layer only has to learn the non-linear remainder.
Outside the training data the model falls back towards that linear
model instead of towards the saturated hidden units.  That is gentle for
small excursions, but the linear term is unbounded: for close contacts
(a bond pushed to 1.2 A) it runs away by hundreds of eV, more so with
the PCA whitening (ADR-0014).  Regularisation differs by model:

- ELM / QELM: `reg` (and `reg_search`) acts on the hidden-layer weights
  beta only; gamma and beta_0 get only a relative jitter (1e-12 of the
  mean diagonal) that keeps the normal equations definite.  A large reg
  therefore shrinks the model towards the linear model it contains, not
  towards zero.
- NN: the linear weights w start at zero and are trained with the rest;
  Adam's `weight_decay` applies to all parameters alike.

`direct_link: false` gives the plain ELM (reg on every output weight) and
the plain MLP.  With the direct term the ELM has D + 1 more fitted
weights per species (D descriptor components), the NN D more.

**ELM / QELM input standardisation and options (`docs/adr/0013`).**
By default (since ADR-0013) the descriptor is z-scored and then whitened
with the training PCA (`standardize: pca`; all directions kept,
`pca_eps` bounds the gain; the map is stored in the model and applied to
F / S rows without the mean) -- in the ADR-0013 comparison this lowered
the force error in 22 of 24 cases (by 7 % on average, up to 30 %).
`standardize: zscore` gives the former per-feature standardisation;
models saved before ADR-0013 load as z-score models. Since 0.2.0a12 the NN
(tabular and atom-centred) takes the same keys, with `zscore` as its
default: `standardize: pca` feeds the network the whitened descriptor,
and `pca_drop` narrows its input.  `min_feature_std` (all models, default
1e-12) masks input columns that hardly vary in the training data.  Options without a
measurable gain there: `w_init_scale_search` (factors on the Glorot
scale of W, b, picked on the validation split), `bias_init: data` (each
unit centred on a training input), `w_correlation: radial` with
`w_corr_length(_search)` (G2 weights correlated along r_s; in-package
ACSF only; implies z-score).  QELM:
`levels: pow2 | ternary | uniform`, `max_exponent` (formerly `n_bits`),
`n_levels`, `p_zero`, `matmul: dense | sparse`, `w_memory: int8 | float64`
(default int8: W held and saved as integer levels plus column scales,
files and memory 1.7-5.2x smaller than the ELM's); accuracy matches the ELM,
W storage shrinks, the sparse product is not faster in numpy.

**Physics-informed parts (`docs/adr/0014`).**
`mlip.baseline: {type: zbl, r_inner, r_outer}` (or `born_mayer` with
per-pair `A`, `rho`; a list of terms) is a fixed short-range repulsion,
switched off smoothly between r_inner and r_outer.  It is subtracted
from the labels and added back by the engine (Delta-learning).  Choose
r_outer below the shortest distance in the data; it is printed.
`mlip.isolated_atom_energy: {Si: 0.0}` makes every atom-centred energy
equal the isolated-atom energy when an atom has no neighbour within the
cutoff: exactly for ELM / QELM (a constrained solve; tabular form for
one species) and the atom-centred NN.  In ADR-0014's tests the isolated-
atom limit was exact at little cost.
`mlip.descriptor.inner_cutoff: [r_off, r_on]` switches pairs closer
than r_off out of the ACSF, so the model cannot run away at close
contacts (the ELM / QELM direct link otherwise does, see above) and the
baseline alone acts there.  Put r_on at or below the data's shortest
distance and let the baseline's r_outer reach r_on.  For example, for Si
with data from 2.0 A: `inner_cutoff: [1.8, 2.0]`, `baseline: {type: zbl,
r_inner: 1.2, r_outer: 2.0}`, `isolated_atom_energy: {Si: 0.0}`.  All
models then gave a repulsive bond-compression curve with unchanged
accuracy inside the data.
ZBL is universal, not fitted, and was 20-50 % too soft at 1.0-1.2 A.
With a few dozen short-range reference points (one atom pushed towards
a neighbour, from about 1.1 A to the data's edge, in the same `fit`
layout), a Born-Mayer term can be fitted instead:
`baseline: [{type: born_mayer, r_inner: 1.8, r_outer: 2.0, fit_to:
sr/structures.csv}]`.  After training, A and rho (per species pair) are
fitted to the residual E_ref - E_model and stored in the model.  If the
term reaches the training data, the model is refitted once.  In
ADR-0014 this matched the reference within 2 % at 1.0-1.5 A (SW Si)
and 1.2 A (Tersoff).

With `mlip.descriptor` set (ACSF), `training_csv` is a structure list in the
`fit` layout above and the descriptors are computed in-package and stored
in the model file.  Such a model then drives `relax` / `md` / ... with

```yaml
potential:
  type:  MLIP
  model: ./mlip_output/elm_model.npz
```

`mlip.form: atom_centred` (ADR-0005) trains the conventional atom-centred
form instead, E = sum_i eps_s(G_i) on each atom's own ACSF vector G_i
(models `elm`, `qelm`, `nn`), from the same structure list.  The energy is
a sum of local terms, so a model trained on one cell size runs on others;
the model file (`atomic_<model>_model.npz` / `.pt`) records the form and
is used with the same `potential: {type: MLIP, model: ...}` entry.

### External codes (ADR-0011)

Any code that can be run from a shell drives every E/F/S solver through

```yaml
potential:
  type:     external
  cmd:      "mpirun -np 8 vasp_std > vasp.out && python -m akitsushima.tools.ase_efs --read vasprun.xml"
  work_dir: ./external_run
```

Each call writes `work_dir/POSCAR`, runs `cmd` in `work_dir` and reads
`work_dir/result.efs` (akitsushima EFS format: energy per atom, forces,
stress in eV/A^3, positive under compression).  Wrappers:
`python -m akitsushima.tools.ase_efs --calc emt|lj` (an ASE calculator),
`--read FILE` (any ASE-readable output with energy, forces and stress --
unverified on real VASP / QE files), and
`python -m akitsushima.tools.akitsushima_efs pot.yaml` (akitsushima itself,
for testing).  `relax`, `md`, `neb`, `tsa`, `phonon` and elastic constants
take it; `gcmc` / `osmc` warn (one external run per trial move); `kappa`
and `coulomb_method` / `charges` / `esm` are refused.

Which solvers take an MLIP (ADR-0007):

| command | tabular | atom-centred | note |
|---|---|---|---|
| `relax`, `md`, `neb`, `tsa`, `phonon`, `elastic`, `gruneisen`, `qha` | yes | yes | need E, F, S only |
| `dielectric`, `piezo` | with `born:` | with `born:` | Z* and eps_inf must be given (no charges of its own) |
| `gcmc` | refused | yes | the atom count changes: needs a sum of site energies; inserted elements must be among the model's |
| `kappa`, `nemd` (local flux) | refused | yes | exact heat flux from the site terms |
| `osmc` | refused | yes | CFCMC couples the fractional atom through total energies (ADR-0009): as `gcmc` |
| `esm` | refused | refused | electrostatics only |

A refusal is an error at start-up, not a silent approximation.  The heat
flux of `kappa` is exact for every potential with a site-energy
decomposition -- pair potentials, Stillinger-Weber, Tersoff, EAM and an
atom-centred MLIP (`akitsushima/engine/site_terms.py`); only a Coulomb
term still uses the centre-of-mass approximation, with a warning.

## Project layout

```
akitsushima/
├── constants.py        # physical constants
├── utils/              # timing, MPI helpers, numba JIT setup
├── io/                 # POSCAR / XDATCAR / EFS file I/O
├── engine/              # energy / force / stress: pair, many-body, and Coulomb potentials
├── fitting/             # parameter sets, loss function, fitting driver, ML models
├── solvers/             # geometry optimisation, MD, GCMC, NEB, TSA, phonons, kappa
└── cli/                 # argument parsing and one module per sub-command
tests/
```

Dependencies form a one-directional DAG: `cli → solvers/fitting → engine →
utils/io`.

## Testing

```bash
pip install -e ".[all,dev]"
pytest tests/
```

`tests/test_mpi.py` spawns real `mpirun -np 1/2/4` subprocesses to check
that results agree across rank counts; it is skipped automatically if
`mpirun` or `mpi4py` is not available.

torch ships its own OpenMP runtime (`libomp`), and so does numba's
default `omp` threading layer. When torch's copy is initialised first,
the first numba `parallel=True` kernel aborts the process with
`OMP: Error #15` and then `Fatal Python error: Aborted`. This was
observed on macOS, also under miniconda with pip's torch wheel. The
reverse order works. Since 0.2.0a11, akitsushima starts numba's
threading layer when it is imported, and every torch import inside the
package happens after that. If torch was imported before akitsushima,
the `workqueue` layer is chosen instead, which loads no OpenMP. So the
order of imports, and of test files, no longer matters
(tests/test_openmp_order.py). An explicit `NUMBA_THREADING_LAYER` is
respected.

## License

MIT — see [LICENSE](LICENSE).
