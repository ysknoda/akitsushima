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
> gone), and LightGBM uses the row label as a feature, standardises each
> label and searches a new space. ELM / QELM now add a direct linear term
> by default (`direct_link`), and NN defaults to softplus with a direct
> linear term. LightGBM files from earlier versions must be refitted.

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
- Tersoff — bond-order (`b_ij`) with coordination (Si, C, Ge, GaAs)
- EAM — embedding `F(rho_i)` + pair `phi(r)`, tabulated (metals)

**Long-range Coulomb**
- Ewald summation — vectorised, `O(N^1.5)` for N ≲ 2,000
- PPPM / SPME — FFT mesh, `O(N log N)` for N ≳ 500
- ESM — Effective Screening Medium (open / metallic boundary conditions)

**Machine-learning potentials**
- `ELMModel` — Extreme Learning Machine (single hidden layer)
- `QuantizedELMModel` — ELM with quantised hidden-layer weights
- `LinearModel` — linear model: OLS, ridge, smoothness-penalised ridge, LASSO, Elastic Net, PLS, Bayesian ridge, ARD
- `NNModel` — PyTorch feed-forward network (lazily imported)
- `LightGBMModel` — Gradient-boosted trees + Optuna search (lazily imported)
- `AtomicELMModel` / `AtomicQELMModel` / `AtomicNNModel` — atom-centred (Behler-Parrinello) models E = sum_i eps_s(G_i), one function per species (ADR-0005)
- `ACSF` — atom-centred symmetry functions G2 / G4 / G5 with analytic position and cell derivatives

### Simulation capabilities

| Task | Description |
|---|---|
| `fit`   | Fit classical or ML potential parameters against DFT reference data (POSCAR geometry + CSV energy/force/stress). |
| `relax` | Geometry optimisation: FIRE, L-BFGS, DIIS, RMM-DIIS, RFO — ionic or variable-cell, with transition-state refinement. |
| `md`    | Molecular dynamics: NVE, NVT (Nosé-Hoover chain), NPT. |
| `gcmc` / `osmc` | Grand-canonical (μVT) MD, including soft-core lambda-dynamics insertion/deletion. |

### Post-processing

- **CI-NEB** — Climbing-image nudged elastic band
- **`TransitionStateAnalyser`** — Hessian, IRC, saddle-point verification, TST reaction rate
- **`PhononCalculator`** — Finite-displacement force constants, band structure, DOS, Debye temperature
- **`ThermalConductivityCalculator`** — Green-Kubo MD + Slack-Debye estimate

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
pip install -e ".[lightgbm]"    # LightGBMModel + Optuna hyperparameter search
pip install -e ".[yaml]"        # YAML config files for the CLI
pip install -e ".[plotting]"    # matplotlib output (NEB/phonon/relax plots)
pip install -e ".[all]"         # everything above
```

`NNModel` and `LightGBMModel` are imported lazily — `import akitsushima`,
and every other class in the package, never pulls in `torch` or `lightgbm`
unless you actually reference one of those two classes.

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

lattice = np.eye(3) * 15.0
positions = np.array([[5.0, 5.0, 5.0], [7.5, 5.0, 5.0]])
atom_types = ["Ar", "Ar"]
lj_params = {frozenset({"Ar"}): {"epsilon": 0.0103, "sigma": 3.405}}

engine = PotentialEngine().add(LennardJonesPotential(lj_params, r_cut=5.0))
fn = make_force_fn(engine)

nl = NeighbourList(r_cut=5.0)
nl.build(lattice, positions)

energy, forces, stress = fn(lattice, positions, atom_types)
```

## CLI reference

| Command | Purpose |
|---|---|
| `relax`  | Geometry optimisation (FIRE / L-BFGS / DIIS / RMM-DIIS / RFO) |
| `md`     | Molecular dynamics (NVT / NVE / NPT) |
| `gcmc`   | Grand-canonical Monte Carlo MD (μVT) |
| `neb`    | Climbing-image NEB transition-state search |
| `tsa`    | Transition-state analysis (Hessian / IRC / TST rate) |
| `osmc`   | OS-MC soft-core lambda-integration GCMC |
| `esm`    | Effective screening medium |
| `phonon` | Phonon band structure, DOS, Debye temperature, free energy |
| `kappa`  | Thermal conductivity (Green-Kubo + optional Slack-Debye) |
| `mlip`   | Fit a machine-learning potential (ELM / QELM / NN / linear / LightGBM) |
| `fit`    | Fit a classical potential (LJ / Morse / Born-Mayer / Buckingham / Stillinger-Weber / Tersoff) |

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
   same).  The `--write-config` templates switch it on for `sw` and
   `tersoff` only.  Its cost is `de_popsize x n_params` loss evaluations
   per generation, so it is practical for a handful of parameters, not
   for tens.
2. **Local refinement (always)** -- `local_method` picks a bounded
   scipy optimiser (default `L-BFGS-B`, case-insensitive):

| `local_method` | uses a gradient | notes |
|---|---|---|
| `L-BFGS-B`    | finite-difference | default; the usual choice |
| `TNC`         | finite-difference | truncated Newton |
| `SLSQP`       | finite-difference | |
| `Nelder-Mead` | none | adaptive simplex; many evaluations |
| `COBYLA`      | none | linear trust-region model |
| `COBYQA`      | none | quadratic trust-region model |

All six respect the parameter bounds and receive the same stopping
intent in their own option names (at most 5000 iterations and 100 000
loss evaluations, loss tolerance 1e-12 eV^2, gradient and step
tolerances 1e-8).  Any other name is refused before any data is loaded:
`BFGS` and `CG` ignore bounds, `Newton-CG`, `trust-*` and `dogleg` need
an analytic gradient, which `fit` does not provide, and scipy's bounded
`Powell` returned points worse than ones it had itself evaluated on
real fit losses.

If `L-BFGS-B` stops early from a poor starting point -- it stalled at a
loss of 14 on an LJ test where the others reached 1e-14 -- rerun with
`Nelder-Mead` or `COBYQA`, which do not rely on a finite-difference
gradient.

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
- **lgbm** regresses each row on its own (label as a feature); trees give
  no conservative force, so it is not a force field.
- Every model weights the squared residual by w_cat x w_unit^2 x w_user.

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

On macOS, a pure-pip environment with both `[numba]` and `[torch]`
installed can abort at import time (`Fatal Python error: Aborted`):
numba and pip's torch wheel each bundle their own OpenMP runtime, and
loading both into one process is not supported. This is an ecosystem
packaging conflict, not an akitsushima bug -- it does not occur under
conda, which coordinates a single shared `libomp`. Work around it with:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
