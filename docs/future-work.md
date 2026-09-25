# Future work (not yet scheduled)

Items recorded here are agreed to be worth doing but are deliberately left
out of the current ADR's scope.  Each one gets its own ADR when it is
picked up.

## FW-1  External / user-chosen descriptors for the MLIP models

Recorded 2026-09-25, during the MLIP design discussion that leads to
ADR-0004 (tabular chain-rule training, in-package ACSF descriptors) and
ADR-0005 (atom-centred model).

**Goal.**  Let the MLIP models use descriptors that the user chose outside
akitsushima -- in particular the ones produced by the user's own script
`atomic_descriptor_ver2.0.0.py` (`/Users/yusuke/Claude_Code/atomic_descriptor/`,
under development) -- as the model's explanatory variables, both for
training and when the model drives `relax` / `md` as a force field.

**Two candidate routes (to be decided when picked up).**

1. *Plug-in interface.*  akitsushima defines a descriptor protocol
   (per-atom descriptor matrix X_i and its Jacobians dX_i/dr_j and the
   cell derivative, plus a serialisable config stored inside the model
   file, as agreed for ADR-0004 Q10), and an adapter wraps the external
   script's kernels to satisfy it.
2. *Absorb the script.*  Port the contents of `atomic_descriptor_ver2.0.0.py`
   (RDF / ADF / ODF, optional CN / composition / dihedral, numba kernels,
   YAML config, superset remapping) into the package as a descriptor
   module alongside ACSF.

**Facts established so far (2026-09-25).**

- The script currently writes *structure-level* descriptors: the E row is
  the per-atom-averaged histogram summed over all atoms, the F rows are
  the exact per-atom gradients of that structure descriptor
  (`F_{i,a} = -dD/dr_{i,a}`), and the S rows its cell derivative.  Its
  `dataset_train.csv` is the `target,label,weight,...` layout that
  `load_mlip_csv` reads today (weights: E -> N_atoms, F -> 1, S -> V).
- With that layout, a model *linear* in D gives exactly conservative
  forces (F = w . F-row).  A nonlinear model is exact when trained with the
  chain rule F = N grad f(d) . X_{ia} (ADR-0004, decision 1), which needs
  only the structure id of each F/S row -- already recoverable from the row
  order.  So training on the script's CSV works as soon as ADR-0004 is
  implemented; what FW-1 adds is calling the descriptor *at inference*, so
  that such a model can drive `relax` / `md`.  Using the script's
  descriptors in the atom-centred model (ADR-0005) would additionally need
  the per-atom (not atom-summed) descriptors and per-pair Jacobians, which
  the script computes internally but does not write out.
- The script already validates its analytic gradients against finite
  differences (`tests/test_*_gradients.py`) and checks translation
  invariance on every call; those tests should come along with it.
- It has its own element-superset remapping; element ordering must follow
  the package convention (atomic number).
- Dependencies: numpy, scipy, numba (optional), mpi4py (optional),
  pyyaml -- all already used by akitsushima.

## FW-2  Additive per-atom LightGBM energy model (energy-only Delta-learning)

Recorded 2026-09-25 (ADR-0004, "Out of scope").  LightGBM can learn
E = sum_i f(X_i) on atom-centred descriptors with a custom objective: each
atom's gradient/hessian is its structure's weighted energy residual (and
hessian).  Trees are piecewise constant, so -dE/dr is zero almost everywhere
and jumps at split boundaries: this gives energies only, never forces for
`relax`/`md`.  Candidate use: an energy correction (Delta-learning) on top
of a force-field model.

## FW-3  Further linear methods for the tabular MLIP

Recorded 2026-09-25 (ADR-0004, decision 9 lists what is implemented now).

- **Group LASSO**: keep or drop whole descriptor blocks (RDF_A-B, ADF...),
  showing which element pairs / angle types matter.
- **Principal-component regression**: PLS's counterpart that picks
  components without looking at the targets; a reference for PLS.
- **Huber regression** (iteratively reweighted): robust to outlying
  structures in real DFT data (e.g. poorly converged SCF).  Its benefit
  cannot be shown on noise-free labels from a known potential, so it waits
  for the real-data stage.

## FW-4  Force/stress-row subsampling for LightGBM

Recorded 2026-09-25 (ADR-0004 follow-up, Q4).  In the chain-rule tabular
models an F or S row costs one dot product (the network runs once per
structure), so subsampling F/S rows saves little there.  LightGBM's cost
grows with the number of rows, so a per-label row fraction (e.g.
`force_row_fraction`) could shorten its training substantially.  Its
existing `bagging_fraction` (searched by Optuna) already subsamples rows
without regard to label; decide on a label-aware fraction once training
time on real DFT data shows it is needed.
