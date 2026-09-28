---
status: proposed (future work; timing decided by the user)
---

# Future work: MLIP with explicit electrostatics under ESM, and open items

Recorded 2026-09-27.  Nothing here is implemented; the user decides when.
Each item below needs its own ADR (or an update of this one) before
implementation, starting with the open questions listed under it.

## 1. MLIP + ESM through a fixed-charge Delta-learned MLIP (Q19-Q21, agreed)

### Why MLIP + ESM is refused today

`_build_from_yaml_mlip` (fitting/registry.py) refuses `coulomb_method`,
`charges` and `esm` with `type: MLIP` (ADR-0008):

1. **Double counting.**  The MLIP is trained on total DFT energies (forces,
   stress), which already contain the ionic electrostatics; adding an ESM
   Coulomb term counts it twice.
2. **Locality.**  An atom-centred MLIP sees only its cutoff sphere; the
   long-range 1/r interaction, electrode image charges and an applied
   field -- exactly what ESM describes -- are invisible to it.
3. **Boundary conditions.**  Whatever electrostatics the model absorbed was
   learned under the training data's (usually 3D periodic) boundary
   conditions and cannot be converted to vacuum / electrode slabs.

### It is not physically impossible

Split long and short range:

    E_MLIP(train) = E_DFT - E_Coul(q; boundary conditions of the DFT data)
    E(use)        = E_MLIP + E_Coul(q; ewald | pppm | esm)

The MLIP learns only the short-range remainder (repulsion, covalency,
in-cutoff polarisation); the explicit Coulomb term carries the long range
under whichever boundary conditions the run uses.  Double counting is
excluded by construction, and each Coulomb term matches its own boundary
conditions -- the structure of Buckingham + Ewald, with the MLIP as the
short-range part.

**Physical limits that stay:** fixed charges cannot describe charge
transfer or the electronic response to a field or an electrode (constant-
potential behaviour); that needs environment-dependent charges (item 1b).
Transferability of the short-range part to surfaces depends on surface
configurations in the training data.

### Decisions (2026-09-27)

- **Q19 -- scope (C):** first (a) the fixed-charge Delta-learned MLIP,
  built from the existing atom-centred MLIP and ESMCoulomb; (b)
  environment-dependent charges (charges predicted by the model,
  charge equilibration solved under the ESM boundary conditions) later,
  in a separate ADR.
- **Q20 -- target and data:** ionic oxide slabs, MgO first.  First
  verification as in ADR-0007: train on synthetic data from Buckingham +
  Ewald and check the MLIP + ESM results against Buckingham + ESM through
  the same solvers; real DFT data afterwards.
- **Q21 -- total-energy MLIPs:** an MLIP trained on total energies stays
  refused with ESM (and any coulomb_method).  The error message will
  point to the Delta-learned model.  The model file records the charges
  and the boundary conditions of the Coulomb baseline it was trained
  against, so a solver can tell mechanically whether a Coulomb term may be
  added.

### Open questions (next grilling round, before implementation)

- Charge source: user-given formal charges, charges from a classical fit,
  or charges fitted jointly with the MLIP.
- Baseline form: point charges, or Gaussian charges of width sigma_q (finite
  at r -> 0, so the MLIP need not learn a divergent attraction between
  opposite charges; ESM already works with Gaussian charges internally).
  A fixed short-range repulsive core as part of the baseline?
- Baseline boundary conditions per training structure (3D Ewald for
  periodic DFT, ESM for QE-ESM slab data) and how they are stored.
- Which solvers and which verification tolerances (sized by the model's
  force error, as in ADR-0007).

## 2. In-plane barostat (NPT with ESM)

ADR-0008 refuses NPT with ESM because the only barostat is isotropic and
would move the ESM boundaries (Lz).  Needed: a barostat acting on a_x and
b_y only (rectangular in-plane cell, Lz and shear fixed), i.e. the
"in-plane only" row of the agreed ADR-0008 table.  Open: isotropic in-plane
(a_x = b_y scaled together) vs independent a_x, b_y; target stress
(in-plane components of the pressure tensor only).

## 3. Multi-species CFCMC

ADR-0009 builds CFCMC for a single exchanged species (the old
multi-species `species_params` / `SpeciesParams` interface was removed
with OSMMCSimulation).  Needed for e.g. Li2O or salt insertion: one
fractional atom per species (or a species choice per lambda cycle), with
per-species mu, bias w(lambda) and estimators; for ionic species, charge-
neutral insertion (ion pairs as one fractional unit) or bc2 / bc3 ESM,
whose electrode takes the net charge.  Open: per-species fractional atoms
vs fractional neutral groups; verification (ideal mixture, NIST mixtures
if available, gcmc per species).

## 4. Constant-potential ESM

The present ESM (bc2 / bc3) is constant-charge: the electrode takes the
image charge of the slab's net charge for a fixed electron count.
Constant potential (fixed electrode potential / chemical potential of the
electrons, mu_e) needs the electrode charge (or the ions' charges) to
adjust -- a grand-canonical treatment of the electrons, related to item
1b.  Open: formulation for fixed point charges (electrode charge as a
dynamical or optimised variable) and its verification.

## Appendix: MLIPs that separate short- and long-range energies (literature, 2026-09-27)

Surveyed for item 1 (and items 1b, 4).  Only what was read in the source
(paper text via arXiv / ar5iv, or official documentation) is stated; items
marked UNVERIFIED were seen only in abstracts or secondary summaries.

| method | energy split | charges | short-range trained on | source read |
|---|---|---|---|---|
| **DPLR** (Deep Potential Long-Range) | E = E_DP + E_ele | Gaussian charges on ions (fixed, e.g. O 6, H 1) and on Wannier centroids (fixed charge, position predicted by a Deep Wannier network) | E* - E_ele (electrostatics subtracted from the labels) | DeePMD-kit documentation (DPLR page) |
| **4G-HDNNP** | E_total(R, Q) = E_elec(R, Q) + E_short(R, Q), E_short = sum of atomic energies | global charge equilibration: E_Qeq = E_elec + sum(chi_i Q_i + J_i Q_i^2 / 2), chi_i from atomic NNs, element hardness J_i, Gaussian widths sigma_i from covalent radii, total charge by a Lagrange multiplier; chi_i trained to reference (Hirshfeld) charges | E_ref - E_elec; the charges are also extra inputs to the short-range NNs | arXiv:2009.06484 (ar5iv text) |
| **3G-HDNNP** | E = E_elec + E_short | charges predicted directly by atomic NNs from the local environment (no equilibration, so no non-local charge transfer) | UNVERIFIED (the 2011 paper was not read; the description of its charges is from the 4G paper) | 4G paper's comparison |
| **LES** (latent Ewald summation) | E = E_sr + E_lr | latent charges q_i = Q(B_i) from the local descriptors, learned from energies and forces only (no charge labels), not constrained to neutrality or physical magnitude | jointly trained (no subtraction); E_lr = (1/V) sum_{0<k<kc} exp(-sigma^2 k^2 / 2) / k^2 abs(S(k))^2, k = 0 omitted, sigma ~ 1 A | arXiv:2408.15165 (ar5iv text) |
| **ec-MLP** (electrochemical MLP) | long-range electrostatics from the charge distribution + short-range MLP residual | electrolyte: Deep Wannier centroids (atomic dipoles); electrode: polarisable electrode, induced charge under applied bias; run with LAMMPS in constant-potential mode | the residual (details UNVERIFIED: from search-result summaries of Ai2-Kit, arXiv:2607.00613) | search summary only |

**What this means for item 1 (fixed-charge Delta-learned MLIP):**

- The plan "E_MLIP = E_DFT - E_Coul(q), then E_MLIP + E_Coul(q; esm)" is the
  DPLR / 4G-HDNNP pattern with fixed charges on the ions only -- DPLR
  without Wannier centroids.  It needs no new theory, only the baseline
  subtraction in training and bookkeeping in the model file.
- **Gaussian charges are the norm.**  DPLR and 4G-HDNNP both use Gaussian
  charge densities, which keeps E_ele finite at short range; in DPLR the
  width (beta, LAMMPS `gewald`) must be the same in training and in use.
  This favours the Gaussian option of the open question "baseline form",
  and ESM already works with Gaussian charges internally (ESMCoulomb adds
  the erfc complement only to reach point charges; a Gaussian baseline
  would skip it).  The width must then be stored in the model file.
- **Charge source:** DPLR uses fixed formal-like charges; 4G fits
  electronegativities to Hirshfeld charges; LES learns unconstrained
  latent charges from energies and forces.  With ESM, unconstrained latent
  charges are unsafe: bc1 needs a neutral cell and the electrode of
  bc2 / bc3 responds to the physical net charge, so charges used with ESM
  must be physical and sum to the true total charge (as in 4G's
  constraint).
- **Item 1b** (environment-dependent charges) is the 4G-HDNNP scheme; under
  ESM the charge equilibration would use the ESM Green's function instead
  of Ewald for E_elec.  **Item 4** (constant potential) is what ec-MLP
  does with a polarisable electrode on top of the short-range residual.

**Sources**
- DeePMD-kit documentation, "Deep potential long-range (DPLR)":
  https://docs.deepmodeling.com/projects/deepmd/en/r2/model/dplr.html ;
  L. Zhang, H. Wang, M. C. Muniz, A. Z. Panagiotopoulos, R. Car, W. E,
  J. Chem. Phys. 156, 124107 (2022) (bibliographic data from the journal
  page; paper text not read).
- T. W. Ko, J. A. Finkler, S. Goedecker, J. Behler, Nat. Commun. 12, 398
  (2021), DOI 10.1038/s41467-020-20427-2; text read as arXiv:2009.06484.
- N. Artrith, T. Morawietz, J. Behler, Phys. Rev. B 83, 153101 (2011)
  (3G-HDNNP; bibliographic data only).
- B. Cheng, npj Comput. Mater. 11, 80 (2025), DOI
  10.1038/s41524-025-01577-7; text read as arXiv:2408.15165.
- D. M. Anstine, O. Isayev, J. Phys. Chem. A 127, 2417 (2023) (review of
  MLIPs and long-range physics; bibliographic data only).
- ec-MLP: Ai2-Kit, arXiv:2607.00613 (search summary only, UNVERIFIED);
  related: J.-X. Zhu, J. Cheng, arXiv:2407.17740 (hybrid dielectric
  response at metal / electrolyte interfaces; abstract only).

## 5. Atom-centred linear and quadratic MLIP (Q29, Q31 agreed 2026-09-28)

The linear model exists only in the tabular form, so gcmc, osmc and kappa
(which need per-atom energies, ADR-0007) have no light model.  Planned:

    eps_s(G_i) = w_s . G_i + b_s  (+ G_i^T Q_s G_i with quadratic: true)
    E = sum_i eps_s(G_i)

- Solver: ridge only, reg chosen on the structure-grouped validation split
  (the atom-centred ELM machinery).  Other linear methods (lasso, PLS,
  Bayesian ridge, ...) only if a need appears.
- Quadratic terms off by default (`quadratic: true`); D(D+1)/2 features per
  species (Si D = 90: 4095; GeSn D = 218: 23 871), printed at start-up and
  refused above 10 000 per species.  The idea matches the quadratic SNAP
  extension (recalled as Wood & Thompson, J. Chem. Phys. 148, 241721
  (2018) -- UNVERIFIED).
- CLI: `mlip: {model: linear, form: atom_centred, quadratic: false}`.
- Verification: finite-difference forces and stress, invariances, other
  cell sizes, save / load, CLI, use in gcmc; the ADR-0005 comparison (Si,
  GeSn, ACSF) re-run for the new models only, rows added to its tables.
- The SW data already sit at the noise floor with the linear model, so
  any gain can only show on harder (real DFT) data.

## 6. Sparse Gaussian-process MLIP (GAP type) (Q29, deferred 2026-09-28)

ACSF per atom, squared-exponential kernel, a sparse set of representative
environments, closed-form solve (Bartok et al., Phys. Rev. Lett. 104,
136403 (2010) -- recalled, not re-checked in this session).  Moderate effort; its main value is the predictive
variance, for active learning (choosing structures to add to training).
Decide when real DFT data are at hand.  Note the ELM already offers
random Fourier features (`activation: rff`), a random-feature
approximation of the same Gaussian kernel, without the variance.

## 7. Ensemble ELM (Q36, agreed 2026-09-28)

Several ELMs that differ only in the random seed (W, b), trained on the
same data; the spread of their predictions estimates the model
uncertainty.  Cheap (one linear solve per member, no retraining of a
network), and useful for active learning: pick the structures with the
largest spread for the next DFT batch.  Open: number of members; whether
the engine reports the spread (per atom / per structure) in md and relax
logs; storage (one file with all members).

## 8. Online sequential ELM (OS-ELM) (Q36, agreed 2026-09-28)

Update beta when new structures are added, without re-solving from
scratch (recursive least squares on the normal equations).  Pairs with
item 7 in an active-learning loop.  Open: how the weighting rule and reg
selection (which uses a validation split) carry over to incremental
updates; the atom-centred solver.

Multilayer / hierarchical ELMs (stacked ELM auto-encoders) are not
planned: the chain-rule forces grow costly with every layer and no
evidence of a gain for force fields was found.  Reconsider only if a
hard data set shows that a single hidden layer is not enough (ADR-0013).

## 9. External / user-chosen descriptors for the MLIP (formerly FW-1)

Moved here from docs/future-work.md (recorded 2026-09-25, ADR-0004 /
ADR-0005 discussion).

**Goal.**  Let the MLIP models use descriptors chosen outside akitsushima --
in particular those of the user's own script `atomic_descriptor_ver2.0.0.py`
(`/Users/yusuke/Claude_Code/atomic_descriptor/`, under development) -- both
for training and when the model drives the solvers as a force field.

**Two candidate routes (to be decided when picked up).**

1. *Plug-in interface.*  A descriptor protocol (per-atom descriptor matrix
   X_i, its Jacobians dX_i/dr_j and cell derivative, and a serialisable
   config stored in the model file, ADR-0004 Q10); an adapter wraps the
   external script's kernels.
2. *Absorb the script.*  Port `atomic_descriptor_ver2.0.0.py` (RDF / ADF /
   ODF, optional CN / composition / dihedral, numba kernels, YAML config,
   superset remapping) into the package next to ACSF.

**Facts established (2026-09-25).**

- The script writes *structure-level* descriptors: the E row is the
  per-atom-averaged descriptor, the F rows its exact per-atom gradients
  (`F_{i,a} = -dD/dr_{i,a}`), the S rows its cell derivative; the CSV is the
  `target,label,weight,...` layout of `load_mlip_csv` (weights E -> N_atoms,
  F -> 1, S -> V).
- Training on that CSV already works (chain rule, ADR-0004 decision 1);
  what is missing is computing the descriptor at inference so that such a
  model can drive relax / md.  The atom-centred model would also need the
  per-atom descriptors and per-pair Jacobians, which the script computes
  internally but does not write out.
- The script validates its gradients against finite differences and checks
  translation invariance; those tests should come along.
- Element ordering must follow the package convention (atomic number).
- Dependencies (numpy, scipy, numba, mpi4py, pyyaml) are already used by
  akitsushima.

## 10. Further linear methods for the tabular MLIP (formerly FW-3)

Moved here from docs/future-work.md (recorded 2026-09-25, ADR-0004).

- **Group LASSO**: keep or drop whole descriptor blocks (RDF_A-B, ADF, ...),
  showing which element pairs / angle types matter.
- **Principal-component regression**: PLS's counterpart that picks
  components without looking at the targets.
- **Huber regression** (iteratively reweighted): robust to outlying
  structures in real DFT data (e.g. poorly converged SCF); its benefit
  cannot show on noise-free labels, so it waits for real data.

docs/future-work.md is removed.  Its FW-2 (additive per-atom LightGBM
energy model) and FW-4 (force / stress row subsampling for LightGBM) were
dropped with LightGBM (ADR-0012).
