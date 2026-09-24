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
