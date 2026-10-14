---
status: accepted (plan only; no implementation)
---

# GPU support: scope, backend, precision and staging

Requested by the user on 2026-09-29: study whether akitsushima can be made
to run on GPUs, and record the decisions.  Implementation is deferred
("much later"); this ADR is the plan every later GPU stage refers to.
Version stays 0.2.0a4 (no code changes).

## Starting point (found by reading the code, 2026-09-29)

- The compute path is CPU only: numpy / scipy, with optional numba JIT
  (`utils/jit.py`, `parallel=True`) in the pair and many-body kernels,
  the neighbour list, ACSF, phonon and `fitting/force_fn.py`; optional
  mpi4py in replicated-data form (every rank holds all atoms and
  Allreduces at the end of each `compute()`, `utils/mpi.py`).
- torch is used only for training `NNModel` / `AtomicNNModel`; their
  `device='auto'` picks CUDA or CPU and never MPS (`models/nn.py:167`,
  `models/atomic_nn.py:78`).  MLIP inference (`MLIPForceField`), the ELM
  normal equations and Cholesky solve, the PPPM / ESM FFTs and phonon
  diagonalisation are numpy / scipy.
- Solvers (MD, relaxation, NEB, ...) call `engine.compute()` from Python
  once per step, so a GPU engine pays a host <-> device transfer per step.
- `AtomicJacobian` stores `dGdv (M, 3, Dc)` in float64
  (`descriptors/acsf.py:358`).  At N = 1e6 with ~80 neighbours and
  Dc = 100 this is 8e7 * 3 * 100 * 8 B ~ 190 GB, which does not fit on
  any single GPU (H100: 80 GB).
- Development machine: Apple M4 (Metal 3, MPS available in torch 2.13;
  no NVIDIA GPU).  torch's MPS backend has no float64.  numba's CUDA
  target is NVIDIA only (now split out as the `numba-cuda` package).

## Decisions (agreed with the user, 2026-09-29, Q1-Q17)

1. **Scope (Q1, Q3).**  Target system sizes N = 1e5 - 1e6.  GPU support
   is to be considered for almost every feature; highest priority:
   (c) MD with an MLIP (ACSF + ELM / QELM / NN inference) and (d) MLIP
   training (ELM A^T A accumulation and solve, NN training, ACSF table
   generation).  This ADR records the plan only; each stage below is
   implemented later under its own ADR (or an addendum here).
2. **Hardware (Q2, Q6).**  CUDA is the supported target: NVIDIA H100
   (data-centre, FP64 ~ 1/2 of FP32) is the primary machine, a GPU
   workstation (RTX / L40S class, FP64 ~ 1/32 - 1/64 of FP32 per the
   vendor datasheets) a possible later one.  MPS (Apple Silicon) is
   best-effort: it should run, float32 only, and is not a performance
   target.
3. **Adoption gate (Q4, Q14).**  A GPU path is adopted only if it is at
   least **3x faster** than the numba-parallel CPU path on the benchmark
   below; otherwise it stays opt-in or is dropped.
   - Metric: ms per NVE MD step, host <-> device transfer included.
   - CPU baseline: numba parallel on all cores of one socket of the same
     node.
   - Systems, each at N = 1e5 and 1e6 (1e5 only if the current CPU path
     runs out of memory at 1e6):
     1. Si, Tersoff (many-body classical);
     2. Ni-Al, Zhou EAM (ADR-0015);
     3. Si, atom-centred ACSF + ELM MLIP (priority (c));
     4. ZnO, Buckingham + PPPM (long-range Coulomb).
   - MLIP training (d) is judged separately on ELM fit wall time for
     ~1000 training structures.
4. **Backend (Q5, Q17).**  GPU paths are written in **torch** (runs on
   CUDA and MPS; autograd; `torch.fft` for PPPM / ESM; `index_add_` /
   scatter for pair and many-body accumulation).  Custom kernels via
   Triton (CUDA only) are allowed where a benchmark shows the torch
   version is the bottleneck.  CuPy / numba-cuda, JAX and vendor-specific
   stacks (Warp, MLX) are not used.  The numpy / numba CPU path stays as
   the **reference implementation** and is never removed.  A new extra
   `gpu = ["torch>=..."]` (Triton added there if used) is created; the
   minimum torch version is set in stage 1.  Known risk: the macOS
   duplicate-OpenMP conflict between numba and pip's torch wheel (see
   `utils/jit.py` and the README).
5. **Host / device boundary (Q7).**  First the engine alone moves to the
   GPU (positions in, energy / forces / stress out, ~24 MB per direction
   at N = 1e6, ~1 ms on PCIe 4/5).  A solver's whole loop (integrator,
   thermostat, neighbour-list rebuilds) moves to the GPU only where the
   benchmark shows the transfer limits it.  The array-abstraction
   boundary is fixed in stage 1 so that solvers do not grow new
   numpy-only assumptions in the meantime.
6. **ACSF Jacobian at large N (Q8).**  Inference computes G and the
   force / virial contraction tile by tile (atoms in blocks sized from
   free device memory; the partial `dGdv` is contracted and discarded),
   never holding the full `(M, 3, Dc)` array.  torch autograd on
   E(G(r)) is the correctness oracle for that path.  A fully fused
   on-the-fly kernel is considered only if the tiled path fails the
   gate.  Training keeps the current stored-Jacobian form (training
   structures are small).
7. **One GPU per run (Q9).**  Multi-GPU for a single structure needs a
   domain-decomposition rewrite of the replicated-data MPI and is left
   to a separate ADR.  Independent items (NEB images, phonon
   displacements, fitting structures) can already be spread over ranks,
   one GPU each, through the existing MPI item distribution
   (`local_only`).
8. **CPU-only features (Q10).**  GCMC / CFCMC trial moves, the direct
   Ewald sum (use PPPM or ESM at large N) and TSA stay on the CPU.  If a
   GPU device is requested for them, they warn and run on the CPU.  Their
   energy / force calls still go through the engine and benefit from a
   GPU engine indirectly.
9. **Precision (Q11).**  `precision: auto | float64 | mixed | float32`.
   - `float64`: everything double; the default on H100.
   - `mixed`: per-pair / descriptor terms in float32, accumulation of
     energy, forces and virial in float64 (the GROMACS / LAMMPS-KOKKOS
     sense); intended for workstation GPUs.
   - `float32`: everything single; the only option on MPS.
   - `auto`: `float64` on a CUDA device with fast FP64 (decided by a
     device-name allow-list), `mixed` on other CUDA devices, `float32` on
     MPS.  MPS `float32` is for training and trials only; it is not
     supported for long NVE MD (no energy-conservation guarantee).
10. **User interface (Q12).**  A top-level YAML block
    `compute: {device: auto|cpu|cuda|mps, precision: ...}`, overridable
    by `--device` / `--precision` on the CLI.  The existing `ml.device`
    stays as an alias; a conflict between the two is an error.
    `device: auto` uses the GPU only for paths that have passed the
    adoption gate (decision 3) and the CPU for everything else.
11. **CPU / GPU agreement tests (Q13).**  Tolerances fixed now, never
    loosened (ADR-0003 practice), GPU result versus the CPU reference on
    the same structure:

    | precision | energy [eV/atom] | forces [eV/A] | stress [eV/A^3] | NVE drift |
    |-----------|------------------|---------------|-----------------|-----------|
    | float64   | 1e-10            | 1e-8          | 1e-8            | --        |
    | mixed     | 1e-6             | 1e-4          | 1e-5            | <= 1e-5 eV/atom/ps over 10 ps |
    | float32   | 1e-6             | 1e-4          | 1e-5            | not required |

    Tests carry a `gpu` pytest marker.  MPS tests run on the development
    Mac; CUDA tests are run by hand on the H100 (no GPU in CI).

## Stages (Q15)

In dependency order; each stage is gated by decision 3 and recorded in
its own ADR or as an addendum here.

1. Infrastructure: the `compute:` block and CLI flags, device / precision
   resolution, numpy <-> torch array hand-off, the `gpu` extra, and MPS
   in `NNModel` / `AtomicNNModel` `device='auto'`.
2. Neighbour list on the GPU (cell list); a prerequisite for every path
   at N = 1e6.
3. MLIP inference: tiled ACSF (decision 6), ELM / QELM / NN evaluation.
4. MLIP training: ELM A^T A accumulation and Cholesky solve, ACSF table
   generation.
5. Classical potentials: pair potentials, EAM (incl. Zhou), SW, Tersoff.
6. Long-range Coulomb: PPPM and ESM via `torch.fft`.
7. Device-resident solver loops for MD and relaxation (decision 5,
   second step).
8. Phonon and thermal conductivity: batched displacements / q-points.

## Not decided here

- Domain decomposition and multi-GPU for one structure (decision 7).
- The exact FP64 allow-list and the minimum torch version (stage 1).
- Whether Triton kernels are needed at all (only after a gate failure).

## Recording (Q16)

This file; status accepted as a plan; no version bump.  Per the user,
ADR-0016 was to be committed on its own, separate from the ADR-0015
work; it was, in e111c14 (2026-10-08).  Implementation still waits for
the user.
