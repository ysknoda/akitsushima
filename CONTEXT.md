# akitsushima

Atomistic simulation with classical force fields and machine-learning
interatomic potentials: fitting potentials, relaxing structures, running
molecular dynamics, and computing phonon and transport properties.

This is a glossary of the terms this project uses in a specific or
opinionated way. Terms that mean the same thing here as they do in any
textbook are left out; so are general programming concepts.

## Language

### Cells and periodicity

**Perpendicular width**:
The spacing between opposite faces of the cell, along each of the three
directions. Distinct from the length of a lattice vector, and smaller
than it for any sheared cell — a cell can look large enough and still be
too thin.
_Avoid_: cell width, box size, cell dimension

**Minimum image**:
The convention that each pair of atoms interacts through exactly one
periodic image, the nearest one. It is valid only while the cutoff stays
under half the smallest perpendicular width; past that it still returns
an answer, silently missing the other images inside the cutoff.
_Avoid_: nearest image, PBC convention

**Unwrapped coordinates**:
Positions that are allowed to leave the cell, so that a trajectory is
continuous across a boundary. Required wherever a displacement is
meaningful — diffusion, and an optimiser's step — as opposed to wrapped
coordinates, which are what gets written to a structure file.
_Avoid_: absolute coordinates, real coordinates, true positions

### Neighbour lists

**Cutoff** (`r_cut`):
The distance beyond which a potential contributes nothing. A property of
the potential.
_Avoid_: range, radius, r_max

**Skin**:
The extra distance a neighbour list reaches beyond the cutoff so that it
survives a few steps of atomic motion before it has to be rebuilt.
_Avoid_: buffer, margin, padding

**List radius** (`r_list`):
Cutoff plus skin — how far the neighbour list actually reaches. This, not
the cutoff alone, is what the minimum-image bound applies to.
_Avoid_: search radius, verlet radius

**Candidate pair**:
A pair of atom indices the neighbour list offers as possibly within the
cutoff. Only indices: the caller computes the geometry itself, from the
current positions, and applies its own cutoff. A candidate pair is not
yet an interacting pair.
_Avoid_: neighbour pair, pair, neighbour

### Forces, stress and the cell

**Compressive-positive**:
This project's stress sign convention, in which a cell compressed below
its equilibrium volume has a positive diagonal stress. The opposite of
the engineering-mechanics convention and of ASE's; the SAME sign as
VASP's and Quantum ESPRESSO's raw printed stress, so converting their
output needs a unit factor and no sign flip. The source of several sign
errors; recorded in
docs/adr/0001-stress-sign-and-multiplication-order-conventions.md.
_Avoid_: tensile-positive (this entry's own former name -- tensile-
positive means the OPPOSITE convention, and reading the old name at face
value is exactly how `VASP_KB_TO_EV_ANG3` acquired a spurious sign flip),
standard convention, physical sign, normal sign

**Cell gradient**:
The derivative of the energy with respect to the lattice matrix, taken
with the fractional coordinates held fixed — so the atoms move affinely
with the cell. The cell's counterpart to the force on an atom.
_Avoid_: cell force, lattice force, stress force

**Generalised force**:
The ionic forces and the cell gradient together, as the single vector a
variable-cell optimiser descends. Undefined when the cell is fixed.
_Avoid_: total force, combined force

**Variable-cell**:
A calculation in which the lattice relaxes alongside the atoms.
_Avoid_: full relaxation, cell relaxation, NPT relaxation

### Elastic properties

**Clamped-ion / relaxed-ion**:
The two elastic tensors of `elastic` (ADR-0017).  Clamped-ion: atoms
strained affinely with the cell (fractional coordinates fixed).
Relaxed-ion: atoms then relaxed inside the fixed strained cell.  The
physical static tensor is the relaxed-ion one; the two coincide with one
atom per primitive cell.
_Avoid_: unrelaxed / relaxed C (ambiguous with the pre-relaxation),
frozen-ion, ionic contribution

**Internal strain**:
The displacement of each atom away from its affine position per unit
strain, du_ia/d eta_j -- the part of the relaxed-ion response the
clamped-ion tensor misses, and the input of the relaxed-ion piezoelectric
tensor.
_Avoid_: internal relaxation (the process, not the tensor), Kleinman
parameter (one scalar of it for diamond)

**Voigt / Reuss / Hill average**:
Polycrystalline bulk and shear moduli from C (uniform strain), from
S = C^-1 (uniform stress), and their arithmetic mean.  E and nu of each
scheme come from that scheme's own B and G; E_Hill is not the mean of
E_Voigt and E_Reuss.
_Avoid_: VRH (when only one of the three is meant), isotropic average

**Born charge** (Z*):
Z[k, a, b] = dP_a / du_kb = dF_kb / dE_a -- polarisation index first, as
phonopy and VASP.  In this code's rigid-ion model it is the point charge
times the unit tensor unless given under `born:` (ADR-0019).
_Avoid_: effective charge (also means the mode effective charge), ionic
charge

**Rigid-ion default**:
Z* = point charge, eps_inf = 1: what the dielectric and piezo commands
assume without a `born:` block.  Overestimates LO-TO splitting and leaves
out the electronic parts of eps_0 and e.
_Avoid_: formal-charge model (the charges need not be formal)

**NAC** (non-analytic correction):
The Gonze-Lee dipole-dipole term added to D(q); what produces LO-TO
splitting at Gamma, which depends on the direction of approach.
_Avoid_: LO-TO correction, polar correction

**Per-atom heat current** (J_i):
The share of V J carried by atom i in ADR-0020's split of the exact
heat flux; sums to V J.  Its time average is zero in equilibrium -- the
meaningful per-group quantity in Green-Kubo is kappa_g, the
cross-correlation of the group's flux with the total.
_Avoid_: local kappa, atomic conductivity

**Heat bath** (NEMD):
A slab of atoms whose velocities get the Langevin update at a fixed
temperature; its energy input over time is the heat current.
_Avoid_: thermostat region, reservoir (also used for chemical potentials)

### Ensembles and dynamics

**Equilibration** / **production**:
The opening stretch of a trajectory, discarded because the system is
still settling, and the remainder, from which every average is taken. A
quantity reported from equilibration frames is not a measurement.
_Avoid_: warm-up, thermalisation, burn-in, sampling phase

**Chemical potential** (`mu`):
The full chemical potential, the control parameter of a grand-canonical
simulation. Not the excess chemical potential, which is what remains
after the ideal-gas part is removed and which cannot select a density on
its own.
_Avoid_: mu_ex, excess chemical potential

**Barostat momentum**:
The momentum conjugate to the cell's strain variable — the barostat's own
degree of freedom, which is thermostatted like any other.
_Avoid_: volume momentum, cell velocity

**Conserved quantity**:
The invariant of an extended-system integrator, which includes the
thermostat and barostat terms alongside the physical energy. Its drift is
the measure of whether a timestep is small enough; it is not the physical
energy and is not expected to equal it.
_Avoid_: total energy, Hamiltonian, conserved energy

### Fitting

**Structure**:
One reference configuration together with its target energy, forces and
stress. The unit a training set is counted in, and the unit a train/test
split must divide — never its individual rows, which all come from the
same configuration.
_Avoid_: sample, configuration, data point, frame

**Scale** / **weight**:
Two different per-structure multipliers in the loss. A scale is a unit
correction and enters squared; a weight expresses how much a data source
is trusted and enters linearly.
_Avoid_: using either name for the other

**Species pair**:
An unordered pair of elements, which is how every pairwise parameter is
keyed. Unordered is the point: the parameters for A–B and B–A are the
same entry.
_Avoid_: pair type, bond type, interaction pair

**Tabular model**:
An MLIP that sees one descriptor per structure (the per-atom average d)
and learns E/N = f(d) from a table of E, F and S rows, the F and S rows
holding derivatives of d (ADR-0004). Linear in d it is also atom-centred;
nonlinear it is not local.
_Avoid_: global model, structure model

**Atom-centred model**:
An MLIP whose energy is a sum of per-atom terms, E = sum_i eps_s(G_i),
each depending only on one atom's own neighbourhood within the cutoff
(Behler-Parrinello; ADR-0005). Locality is what lets it run on a cell
size, surface or defect it was not trained on.
_Avoid_: local model, per-atom model, HDNNP (for the ELM form)

### Spectroscopy

**Response model**:
What gives a structure's dipole and electronic polarisability and their
derivatives with respect to the atomic positions (Born charges Z*, the
polarisability derivative), in the role `force_fn` plays for forces:
fixed tensors from DFT, point charges, or the bond polarisability model
(ADR-0046, ADR-0049).
_Avoid_: charge model, dielectric model

**Polarisability derivative** (`A`):
A[k, a, b, c] = d alpha_ab / d u_kc, per atom, in A^2, with alpha the
polarisability volume (alpha = V (eps_inf - 1) / 4 pi). The input of
every Raman intensity; response indices first, as in Z*.
_Avoid_: Raman tensor, Raman susceptibility

**Raman tensor**:
The 3x3 tensor of one mode only, R_m = sum_k A_k e_m(k) / sqrt(M_k) — the
derivative of alpha along that mode's normal coordinate. Not the per-atom
quantity, which is the polarisability derivative.

**Activity**:
Whether a mode set can appear in a Raman (or IR) spectrum at all, decided
by symmetry from the characters of the space-group operations: a yes or
no. An active mode can still be weak; an inactive one is exactly zero.

**Powder intensity**:
The rotational average 45 a^2 + 7 g^2 = I_par + I_perp of one mode set,
in A^4/amu, before the thermal factor. Gaussian and Phonopy-Spectroscopy
call this scalar the "Raman activity"; here activity is the symmetry
yes/no.
_Avoid_: Raman activity (for this scalar)

**Polarisation geometry**:
A named pair of polarisation vectors (incident e_i, scattered e_s), given
in Cartesian or lattice directions. The propagation directions are not
part of it.
_Avoid_: Porto notation (it is not parsed)

**BPM** (bond polarisability model):
alpha as a sum over bonds of a uniaxial tensor whose parallel and
perpendicular parts depend linearly on the bond length: three parameters
per species pair enter the derivative.

**Two-phonon DOS**:
The density of (q, -q) phonon pairs that second-order Raman / IR can
reach, the photon momentum being about zero. Not the general three-phonon
q' + q'' = q phase space.
_Avoid_: joint DOS (phono3py's term for the general case)

**ADP** (atomic displacement parameter, U_ij):
The thermal mean-square displacement tensor <u_i u_j> of one atom, in
A^2, with U_iso and B = 8 pi^2 U_iso.
_Avoid_: Debye-Waller factor (the exponential exp(-2W) it enters)

### MD spectra (`mdspec`, merged from nozuchi's glossary, ADR-0051)

**Run** (mdspec):
One trajectory file. Runs are combined by **concatenation** (one
trajectory continued) or as **independent runs** (separate spectra,
averaged; their scatter is the honest error bar).
_Avoid_: seed, chunk

**Run temperature**:
The mean instantaneous temperature of a run over its kept frames, from
its own `temperature_source` (thermo.log or OUTCAR). It enters only the
raw estimator.

**Total dipole** (M(t)):
The cell dipole built by a dipole model in the displacement form
M = sum_k Z_k . u_k, u_k(t) = (f_k(t) - fbar_k) h(t), from unwrapped
fractional coordinates and their run average. For a fixed cell it differs
from sum_k Z_k . r_k by a constant only; for a variable cell it removes the
spurious term of the cell fluctuations.
_Avoid_: polarisation (that is M / V)

**Dipole model**:
How M(t) is built: point_charge (fixed scalar charges), born_tensor (Z*
mapped from a DFT cell), external (a series computed elsewhere). It is
the response model's charge part, used frame by frame.

**Estimator** (raw / fsum / mode):
How the static value and the spectrum are normalised: raw (none), fsum
(each Cartesian component by the exact kinetic constraint <Mdot^2>),
mode (each Gamma normal coordinate by its own kinetic energy; needs Gamma
eigenvectors). `auto` takes mode when the modes are known, else fsum.

**f-sum ratio**:
<Mdot^2>_MD / <Mdot^2>_exact, a diagnostic: 1 in a well-sampled canonical
run; it also catches a wrong timestep, charge or mass.

**Normal coordinate** (Q_m):
The projection of the supercell displacements on a mass-weighted Gamma
eigenvector of the unit cell; M = p Q exactly for constant charges.

**Force-based static value** (eps_static_force):
Delta eps from positions and forces through the effective harmonic
matrix D_eff = -<G dQ^T><dQ dQ^T>^-1; the Tier 2a / 3a reference, reported
alongside and never selected.

**Mode table**:
Delta eps per Gamma mode with every cross term between two modes shared
equally (so it sums to the total), with the diagonal-only value, the
peak frequency and the linewidth. The cross share is what cross terms
carry.

**KK check**:
eps' rebuilt from eps'' by Kramers-Kronig and compared with the direct
FDT eps'; a consistency diagnostic only.

**Validation tiers** (mdspec):
Tier 1 analytic signals; Tier 2 / 2a akitsushima classical MD against the
harmonic `dielectric` values and the force-based value; Tier 3 / 3a VASP
AIMD against DFPT and the force-based value; Tier 4 experiment, shown
only.
