"""Effective Screening Medium (ESM) electrostatics for slab geometries.

Reference
----------
M. Otani and O. Sugino, Phys. Rev. B 73, 115407 (2006).

Overview
---------
ESM replaces the standard 3-D periodic Ewald / PPPM long-range electrostatics
with a 2-D periodic + open-boundary treatment along z.  The slab is assumed
to lie in the xy plane with the z axis pointing out of the surface.

The Poisson equation is solved in the mixed representation:

    V(G||, z) = (2pi / G||) int rho(G||, z') G(G||, z, z'; BC) dz'

where G|| = |G_xy| and G is the 1-D Green function whose form depends on
the boundary condition (BC).

Three boundary conditions (BC)
--------------------------------
BC1  vacuum | slab | vacuum
     Both sides are vacuum (dielectric constant 1).
     Requires a charge-neutral slab (total charge = 0).

BC2  vacuum | slab | metal
     The right boundary (z = +Lz/2) is a perfect metal (image charges).
     Supports charged slabs; models one electrode surface.

BC3  metal | slab | metal
     Both boundaries are perfect metals (parallel-plate capacitor).
     Supports charged slabs and external electric field; models
     electrochemical cells.

Charge density representation
-------------------------------
Each point charge q_I at r_I is smeared with a normalised Gaussian of
width sigma (default 0.5 A):

    rho_I(r) = q_I / (sigma sqrt(2pi))^3 x exp(-|r-r_I|^2 / (2sigma^2))

After Fourier transform in x and y the density becomes separable:

    rho_I(G||, z) = q_I x exp(-G||^2 sigma^2/2) / (sigma sqrt(2pi))
                 x exp(-(z-z_I)^2/(2sigma^2)) x exp(i G||.r||_I)

Numerical grid
---------------
xy plane  : regular N_gxy x N_gxy FFT grid (default 64).
z axis    : uniform N_gz points covering [-Lz/2, +Lz/2] (default 256).

Interface with other solvers
-----------------------------
ESMCalculator implements __call__(lattice, positions, atom_types)
-> (energy[eV], forces[eV/A], stress[eV/A^3])

It can therefore be used as force_fn for:
  GeometryOptimiser, NVTSimulation, NVESimulation, NPTSimulation,
  GCMCMDSimulation, NEBOptimiser, PhononCalculator, etc.

Additional helper methods
--------------------------
update_geometry(lattice, positions, atom_types)
    Rebuild the charge-density grid after insertion/deletion (GCMC).
get_potential_profile()
    Return the planar-averaged electrostatic potential V(z) [eV].
work_function(e_fermi=0.0)
    Vacuum level - Fermi level (BC2/BC3 only) [eV].
surface_dipole_correction()
    Dipole-layer correction energy [eV] (BC1 only).

Integration notes
------------------
* z-direction cell: for NPT simulations use fix_cell[2,:] = True.
* PhononCalculator: set supercell = [N, N, 1] to avoid compressing vacuum.
* GCMCMDSimulation: call esm.update_geometry() inside _accept_insertion /
  _accept_deletion (the default hook is automatic when force_fn is an
  ESMCalculator).
"""
import warnings
from typing import Optional, Tuple, Union

import numpy as np
from scipy.integrate import cumulative_trapezoid

_ESM_HARTREE_TO_EV = 27.2113860459           # 1 Ha = 27.211 eV
_ESM_BOHR_TO_ANG = 0.529177210903          # 1 bohr = 0.529177 A
_ESM_E_CHARGE_SI = 1.60217663e-19          # C
_ESM_EPS0_SI = 8.8541878128e-12          # F/m
_ESM_EV_PER_E2_ANG = 14.3996455              # e^2/A -> eV  (= 1/(4pi epsilon0) in eV.A/e^2)


def _trapz_mean(f, dx):
    """Trapezoid-rule mean of samples f spaced dx over their full span."""
    return float(np.trapezoid(f, dx=dx) / (dx * (len(f) - 1)))


class ESMCalculator:
    """
    Effective Screening Medium electrostatics for slab geometries.

    The slab must lie in the **xy plane** (z is the surface normal).
    Long-range Coulomb interactions are evaluated with 2-D periodicity
    in xy and the chosen open/metal boundary condition along z.

    Parameters
    ----------
    lattice    : (3,3) array [A].  Must be orthorhombic or at least have
                 the z-vector along Cartesian z: lattice[2] = [0, 0, Lz].
    positions  : (N,3) Cartesian coordinates [A].
    atom_types : list[str] of length N.
    charges    : dict {symbol: charge [e]} or (N,) array of charges.
    bc         : boundary condition -- ``'bc1'``, ``'bc2'``, or ``'bc3'``.

                 ``'bc1'`` vacuum | slab | vacuum  (neutral slab required)
                 ``'bc2'`` metal  | slab | metal   (both walls, field OK)
                 ``'bc3'`` vacuum | slab | metal   (right wall at +Lz/2)
                 -- the naming of Otani & Sugino (2006) and Quantum
                 ESPRESSO; swapped in akitsushima before 2026-09-26.

    sigma      : Gaussian smearing width [A].  Default 0.5.
    n_grid_xy  : FFT grid size per xy direction.  Default 64.
    n_grid_z   : Number of z-grid points.  Default 256.
    external_field : uniform external electric field along z [V/A].
                 Only meaningful for BC2 and BC3.  Default 0.0.
    r_cut      : stored as attribute for solver layer.  Does not affect
                 the ESM calculation itself.  Default 15.0.

    Example
    -------
    >>> esm = ESMCalculator(lattice, positions, atom_types,
    ...                     charges={'Na': +1., 'Cl': -1.},
    ...                     bc='bc1')
    >>> go  = GeometryOptimiser(lattice, positions, atom_types,
    ...                         force_fn=esm, relax_cell=False)
    >>> result = go.run()
    """

    kind = "esm"

    def __init__(self,
                 lattice: np.ndarray,
                 positions: np.ndarray,
                 atom_types: list,
                 charges: Union[dict, np.ndarray],
                 bc: str = 'bc1',
                 sigma: float = 0.5,
                 n_grid_xy: int = 64,
                 n_grid_z: int = 256,
                 external_field: float = 0.0,
                 r_cut: float = 15.0):

        if bc not in ('bc1', 'bc2', 'bc3'):
            raise ValueError(
                f"ESMCalculator: bc must be 'bc1', 'bc2', or 'bc3', got '{bc}'.")

        self.lattice = np.asarray(lattice, dtype=float).copy()
        self.positions = np.asarray(positions, dtype=float).copy()
        self.atom_types = list(atom_types)
        self.bc = bc
        # ADR-0003 T6 / ADR-0008: the names follow Otani & Sugino, PRB 73,
        # 115407 (2006) and Quantum ESPRESSO's esm_bc -- bc2 metal | slab |
        # metal, bc3 vacuum | slab | metal.  Until 2026-09-26 this class
        # had bc2 and bc3 the other way round.
        self._bc = {"bc1": "_vv", "bc2": "_mm", "bc3": "_vm"}[bc]
        if bc in ("bc2", "bc3"):
            warnings.warn(
                f"ESMCalculator: bc='{bc}' now means "
                + ("metal | slab | metal" if bc == "bc2" else "vacuum | slab | metal")
                + " (Otani-Sugino / Quantum ESPRESSO); before 2026-09-26 "
                "bc2 and bc3 were swapped in akitsushima.",
                UserWarning, stacklevel=2)
        self.sigma = float(sigma)
        self.n_grid_xy = int(n_grid_xy)
        self.n_grid_z = int(n_grid_z)
        self.external_field = float(external_field)
        self.r_cut = float(r_cut)

        self.N = len(atom_types)

        # Build charge array; also store original spec for re-resolution
        self._charges_orig = charges          # original dict or array (for GCMC re-use)
        self._charges = self._resolve_charges(charges, atom_types)

        # Validate lattice (z-vector must be along Cartesian z)
        self._validate_lattice(self.lattice)

        # Cached results (filled by compute)
        self._V_z: Optional[np.ndarray] = None   # V(z) planar-averaged [eV]
        self._V_Gz: Optional[np.ndarray] = None   # V(G||, z) [eV.A]
        self._rho_Gz: Optional[np.ndarray] = None   # rho(G||, z) [e/A^3]

        # Pre-compute grid quantities
        self._setup_grids()

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_charges(charges, atom_types):
        if isinstance(charges, dict):
            try:
                return np.array([charges[a] for a in atom_types], dtype=float)
            except KeyError as e:
                raise ValueError(
                    f"ESMCalculator: no charge defined for element {e}. "
                    "Provide all elements in the charges dict.") from e
        else:
            arr = np.asarray(charges, dtype=float)
            if arr.shape != (len(atom_types),):
                raise ValueError(
                    f"ESMCalculator: charges array must have shape "
                    f"({len(atom_types)},), got {arr.shape}.")
            return arr

    @staticmethod
    def _validate_lattice(lattice):
        """
        Validate that the lattice is compatible with ESM (z-axis slab geometry).

        Requirements
        ------------
        1. The c-vector (lattice[2]) must be parallel to Cartesian z:
           lattice[2] = [0, 0, Lz].
        2. The a- and b-vectors must lie in the xy plane, i.e.
           lattice[0, 2] ~= 0 and lattice[1, 2] ~= 0.
        3. The a- and b-vectors must be orthogonal to each other (rectangular
           slab).  Oblique lattices (monoclinic / triclinic ab plane) are not
           supported because the 2-D FFT grid assumes a rectangular unit cell
           in xy.  An oblique lattice will cause incorrect phase factors.

        Raises
        ------
        ValueError
            For any of the three conditions above.
        """
        tol = 1e-4
        # 1. c-vector along z
        if abs(lattice[2, 0]) > tol or abs(lattice[2, 1]) > tol:
            raise ValueError(
                "ESMCalculator: the c-vector (lattice[2]) must be parallel "
                "to Cartesian z, i.e. lattice[2] = [0, 0, Lz].  "
                f"Got lattice[2] = {lattice[2]}.")
        # 2. a- and b-vectors lie in xy plane
        if abs(lattice[0, 2]) > tol:
            raise ValueError(
                "ESMCalculator: the a-vector (lattice[0]) must lie in the "
                "xy plane, i.e. lattice[0, 2] = 0.  "
                f"Got lattice[0] = {lattice[0]}.")
        if abs(lattice[1, 2]) > tol:
            raise ValueError(
                "ESMCalculator: the b-vector (lattice[1]) must lie in the "
                "xy plane, i.e. lattice[1, 2] = 0.  "
                f"Got lattice[1] = {lattice[1]}.")
        # 3. a x b (rectangular slab)
        a_vec = lattice[0, :2]
        b_vec = lattice[1, :2]
        a_norm = np.linalg.norm(a_vec)
        b_norm = np.linalg.norm(b_vec)
        if a_norm > 1e-12 and b_norm > 1e-12:
            cos_ab = float(np.dot(a_vec, b_vec) / (a_norm * b_norm))
            if abs(cos_ab) > tol:
                angle = float(np.degrees(np.arccos(np.clip(cos_ab, -1, 1))))
                raise ValueError(
                    "ESMCalculator: the a- and b-vectors must be orthogonal "
                    "(rectangular slab in xy).  Oblique lattices are not "
                    "supported because the 2-D FFT grid assumes a rectangular "
                    f"unit cell.  Current angle = {angle:.2f} deg.  "
                    "Use a rectangular supercell instead.")

    def _setup_grids(self):
        """Pre-compute FFT grids in xy and the uniform z grid."""
        latt = self.lattice
        Lx = np.linalg.norm(latt[0])     # [A]
        Ly = np.linalg.norm(latt[1])     # [A]
        Lz = float(latt[2, 2])           # [A]
        self._Lx = Lx
        self._Ly = Ly
        self._Lz = Lz

        Nx = self.n_grid_xy
        Ny = self.n_grid_xy
        Nz = self.n_grid_z

        # Real-space grids [A]
        self._dx = Lx / Nx
        self._dy = Ly / Ny
        self._dz = Lz / Nz
        self._z_grid = np.linspace(-Lz / 2, Lz / 2, Nz, endpoint=False)

        # Reciprocal-space G|| vectors [A^-1] using FFT frequencies
        gx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)   # (Nx,)
        gy = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)   # (Ny,)
        GX, GY = np.meshgrid(gx, gy, indexing='ij')        # (Nx, Ny)
        self._Gpar = np.sqrt(GX**2 + GY**2)               # (Nx, Ny)
        self._GX = GX
        self._GY = GY

    # ------------------------------------------------------------------ #
    #  Main compute interface                                             #
    # ------------------------------------------------------------------ #

    def __call__(self,
                 lattice: np.ndarray,
                 positions: np.ndarray,
                 atom_types: list,
                 ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute ESM energy, forces, and stress.

        Parameters
        ----------
        lattice    : (3,3) [A]  -- updated lattice (must be same shape as init).
        positions  : (N,3) [A]  -- updated Cartesian positions.
        atom_types : list[str]  -- element symbols (charges resolved from init).

        Returns
        -------
        energy : float [eV]
        forces : (N,3) ndarray [eV/A]
        stress : (3,3) ndarray [eV/A^3]  -- xy components physically meaningful;
                                          z components set to 0 for BC1.
        """
        lattice = np.asarray(lattice, dtype=float)
        positions = self._to_esm_frame(lattice, positions)

        # Update geometry (rebuilds grids if lattice changed)
        if not np.allclose(lattice, self.lattice):
            self.lattice = lattice.copy()
            self._validate_lattice(lattice)
            self._setup_grids()

        # Detect geometry changes
        prev_types = self.atom_types
        prev_N = self.N

        self.positions = positions.copy()
        self.atom_types = list(atom_types)
        self.N = len(atom_types)

        # Re-resolve charges when atom_types or N changed (e.g. after GCMC)
        if list(atom_types) != prev_types or len(atom_types) != prev_N:
            self._charges = self._resolve_charges(
                self._charges_orig, atom_types)
        # We always use the per-atom charges array resolved at init unless
        # update_geometry is called explicitly.

        return self._compute_esm(lattice, positions)

    def _to_esm_frame(self, lattice, positions):
        """Map z into [-Lz/2, Lz/2) (ADR-0008, decision 3).

        Solvers wrap fractional coordinates into [0, 1); ESM's grid and
        boundaries are centred on z = 0, as in Quantum ESPRESSO: the slab
        sits around fractional z = 0 and the medium starts at z = +/- Lz/2.
        z -> z - Lz round(z / Lz) moves each atom by a lattice vector, so
        forces are unchanged.  An atom closer than 4 sigma to a boundary
        is refused: its Gaussian would wrap across the medium, and near a
        metal the point-charge complement of its image would matter."""
        positions = np.asarray(positions, dtype=float).copy()
        Lz = float(lattice[2, 2])
        positions[:, 2] -= Lz * np.round(positions[:, 2] / Lz)
        margin = 4.0 * self.sigma
        if positions.size and np.max(np.abs(positions[:, 2])) > Lz / 2 - margin:
            k = int(np.argmax(np.abs(positions[:, 2])))
            raise ValueError(
                f"ESM: atom {k} is at z = {positions[k, 2]:+.3f} A, within "
                f"4 sigma = {margin:.2f} A of the boundary at +/- Lz/2 = "
                f"{Lz / 2:.2f} A.  Centre the slab at fractional z = 0 (the "
                f"medium starts at fractional z = 1/2) and keep a vacuum gap.")
        return positions

    def update_geometry(self,
                         lattice: np.ndarray,
                         positions: np.ndarray,
                         atom_types: list,
                         charges: Union[dict, np.ndarray, None] = None,
                         ) -> None:
        """
        Rebuild internal state after geometry change (e.g. GCMC insertion /
        deletion).

        Call this whenever the number of atoms or the lattice changes.

        Parameters
        ----------
        lattice    : (3,3) [A]
        positions  : (N,3) [A]
        atom_types : list[str]
        charges    : new charge specification (dict or array).  If None,
                     the original charge dict is re-applied to atom_types.
        """
        self.lattice = np.asarray(lattice, dtype=float).copy()
        self.positions = np.asarray(positions, dtype=float).copy()
        self.atom_types = list(atom_types)
        self.N = len(atom_types)
        self._validate_lattice(self.lattice)
        self._setup_grids()
        if charges is not None:
            self._charges_orig = charges
            self._charges = self._resolve_charges(charges, atom_types)
        else:
            # Re-apply original charge spec (dict or array) to new atom_types
            self._charges = self._resolve_charges(self._charges_orig, atom_types)
        # Invalidate cached results
        self._V_z = self._V_Gz = self._rho_Gz = None

    # ------------------------------------------------------------------ #
    #  Core computation                                                    #
    # ------------------------------------------------------------------ #

    def _compute_esm(self,
                      lattice: np.ndarray,
                      positions: np.ndarray,
                      ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Main ESM pipeline: density -> potential -> energy/forces/stress."""

        Lz = self._Lz
        Nx = Ny = self.n_grid_xy
        Nz = self.n_grid_z
        dz = self._dz
        z_g = self._z_grid        # (Nz,) uniform z grid [A]
        Gpar = self._Gpar          # (Nx, Ny) [A^-1]
        sig = self.sigma

        # -- 1. Build charge density rho(G||, z) ---------------
        rho_Gz = self._build_charge_density_2d(positions, self._charges,
                                                Nx, Ny, Nz, Lz,
                                                self._Lx, self._Ly,
                                                sig, z_g,
                                                self._GX, self._GY)
        self._rho_Gz = rho_Gz      # (Nx, Ny, Nz) complex

        # -- 2. Solve Poisson equation: rho -> V(G||, z) -------
        V_Gz = self._solve_poisson_esm(rho_Gz, Gpar, z_g, dz, Lz)
        self._V_Gz = V_Gz          # (Nx, Ny, Nz) complex

        # -- 3. Planar-averaged V(z) ---------------------------
        self._V_z = V_Gz[0, 0, :].real   # G||=0 component [eV]

        # -- 4. Energy [eV] ------------------------------------
        energy = self._compute_energy(rho_Gz, V_Gz, dz)

        # -- 5. Forces [eV/A] ----------------------------------
        forces = self._compute_forces(positions, self._charges,
                                       V_Gz, Nx, Ny, Nz,
                                       self._Lx, self._Ly, Lz,
                                       self._GX, self._GY)

        # -- 6. Stress [eV/A^3] --------------------------------
        stress = self._compute_stress(rho_Gz, V_Gz, Gpar,
                                       self._GX, self._GY,
                                       lattice, dz)

        return float(energy), forces, stress

    # ------------------------------------------------------------------ #
    #  Step 1: Build rho(G||, z)                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_charge_density_2d(positions: np.ndarray,
                                  charges: np.ndarray,
                                  Nx: int,
                                  Ny: int,
                                  Nz: int,
                                  Lz: float,
                                  Lx: float,
                                  Ly: float,
                                  sigma: float,
                                  z_grid: np.ndarray,
                                  GX: np.ndarray,
                                  GY: np.ndarray,
                                  ) -> np.ndarray:
        """
        Compute rho(G||, z) by Gaussian-smearing each atom in z and
        accumulating the 2-D structure factor in the xy plane.

        All atoms are processed simultaneously using NumPy broadcasting,
        eliminating the Python loop over atoms.

        Shapes used in the einsum / broadcast:
            phases  : (N, Nx, Ny)  phase factors exp(-i G|| . r||_I)
            gauss_z : (N, Nz)      Gaussian envelope in z
            fform   : (Nx, Ny)     G-space Gaussian form factor (shared)

        Bug-fix note: this must build the array that ``np.fft.ifft2``
        (used throughout this class to go from G-space back to real
        space; convention ``f(r) = sum_k F(k) exp(+i k.r)``) correctly
        inverts to a charge density peaked at each atom's *actual*
        position. For a point charge at r0, that requires
        ``F(k) = exp(-i k.r0)`` (so that ``ifft2`` reconstructs
        ``exp(+i k.(r-r0))``, which sums to a peak at ``r=r0``) -- i.e. a
        **negative** exponent here, not positive. An earlier version used
        ``exp(+i G.r)``, which ``ifft2`` inverts to a charge density
        peaked at the *mirrored* position ``(Lx-rx, Ly-ry)`` instead of
        ``(rx, ry)``. Verified directly: building the charge density for a
        single off-grid-node test charge and comparing
        ``Nx*Ny*ifft2(rho_Gz)`` against a direct (no-FFT) evaluation of
        the same Gaussian showed the array's peak at the exact mirrored
        grid coordinates, with the value at the true atomic position
        being ~1e-11 (noise floor), not the expected near-peak value.
        Because the energy only depends on isotropic |G|||-dependent
        quantities, this sign error is invisible in the total energy
        (mirroring the whole charge distribution leaves all pairwise
        distances, hence the energy, unchanged) -- it only corrupts
        direction-dependent quantities (forces, stress).

        Returns
        -------
        rho_Gz : (Nx, Ny, Nz) complex array [e/A^3]
        """
        sig2 = sigma**2
        norm_z = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

        # Skip zero-charge atoms
        mask = charges != 0.0
        if not mask.any():
            return np.zeros((Nx, Ny, Nz), dtype=complex)

        q_masked = charges[mask]               # (N',)
        pos_masked = positions[mask]             # (N', 3)

        rx = pos_masked[:, 0]   # (N',)
        ry = pos_masked[:, 1]   # (N',)
        rz = pos_masked[:, 2]   # (N',)

        # 2-D phase factors: (N', Nx, Ny)
        # GX, GY: (Nx, Ny) -> broadcast with rx, ry: (N', 1, 1)
        phases = np.exp(-1j * (GX[None, :, :] * rx[:, None, None]
                               + GY[None, :, :] * ry[:, None, None]))  # (N', Nx, Ny)

        # G-space Gaussian form factor (same for all atoms): (Nx, Ny)
        Gpar2 = GX**2 + GY**2
        fform = np.exp(-0.5 * Gpar2 * sig2)    # (Nx, Ny)

        # z Gaussian: (N', Nz)
        dz_mat = z_grid[None, :] - rz[:, None]  # (N', Nz)
        # Minimum image in z
        dz_mat -= Lz * np.round(dz_mat / Lz)
        gauss_z = norm_z * np.exp(-0.5 * dz_mat**2 / sig2)   # (N', Nz)

        # Accumulate: rho(Gx, Gy, z) = sum_I q_I fform(Gx,Gy) phase_I(Gx,Gy) gauss_z_I(z)
        # Step 1: weighted phases (N', Nx, Ny)
        w_phases = q_masked[:, None, None] * phases   # (N', Nx, Ny)

        # Step 2: contract over atoms: (Nx, Ny, N') x (N', Nz) -> (Nx, Ny, Nz)
        wp_flat = w_phases.reshape(len(q_masked), Nx * Ny).T   # (Nx*Ny, N')
        rho_flat = wp_flat @ gauss_z                              # (Nx*Ny, Nz)
        rho_Gz = rho_flat.reshape(Nx, Ny, Nz)                  # (Nx, Ny, Nz)

        # Apply form factor (Nx, Ny, 1) * rho_Gz
        rho_Gz *= fform[:, :, None]

        # Normalise by 2D unit-cell area
        rho_Gz /= (Lx * Ly)

        return rho_Gz

    # ------------------------------------------------------------------ #
    #  Step 2: Solve Poisson equation                                     #
    # ------------------------------------------------------------------ #

    def _solve_poisson_esm(self,
                            rho_Gz: np.ndarray,
                            Gpar: np.ndarray,
                            z_grid: np.ndarray,
                            dz: float,
                            Lz: float,
                            ) -> np.ndarray:
        """
        Solve the 2-D Poisson equation for each G|| component.

        V(G||, z) = (2pi/G||) int rho(G||, z') G(G||, z, z'; BC) dz'

        Performance
        -----------
        For each unique |G||| value the integral is written as a
        matrix--vector product:

            V_z = (2pi/G||) dz x G_mat @ rho_z

        where G_mat[iz, iz'] = G(G||, z[iz], z_grid[iz']; BC) is the
        (NzxNz) Green-function kernel.  This replaces the O(Nx.Ny.Nz^2)
        triple loop with O(N_unique.Nz^2) matrix products.

        G|| = 0 is treated by the 1-D Poisson solver _solve_gpar0.

        Returns
        -------
        V_Gz : (Nx, Ny, Nz) complex [eV]
        """
        Nx, Ny, Nz = rho_Gz.shape
        V_Gz = np.zeros((Nx, Ny, Nz), dtype=complex)
        bc = self._bc
        K = _ESM_EV_PER_E2_ANG
        Lz_h = Lz / 2.0

        # Pre-build Green-function kernel matrices for each unique |G|||,
        # exploiting the Hermitian symmetry of the FFT.
        gpar_flat = Gpar.ravel()                  # (Nx*Ny,)
        # Use relative-precision rounding so that uniqueness detection
        # remains stable when the cell is very large (large Gpar values)
        # or very small (near-zero Gpar values).
        _gpar_max = float(np.max(np.abs(gpar_flat)))
        if _gpar_max > 0.0:
            _scale = 10 ** (8 - int(np.floor(np.log10(_gpar_max))) - 1)
            _rounded = np.round(gpar_flat * _scale) / _scale
        else:
            _rounded = gpar_flat.copy()
        unique_gp, inv_idx = np.unique(_rounded, return_inverse=True)

        # Cache: gp_value -> G_mat (Nz, Nz)
        kernel_cache: dict = {}

        for uid, gp in enumerate(unique_gp):
            if gp < 1e-10:
                continue
            # Build G_mat: rows=z (field), cols=z' (source)
            z_row = z_grid[:, None]           # (Nz, 1)
            z_col = z_grid[None, :]           # (1, Nz)
            G_mat = self._green_function_matrix(gp, z_row, z_col, bc, Lz_h)
            kernel_cache[uid] = G_mat

        # Apply kernel: for each unique G||, collect all (ix,iy) with that
        # value and apply one batched matrix product.
        for uid, gp in enumerate(unique_gp):
            if gp < 1e-10:
                continue
            G_mat = kernel_cache[uid]                      # (Nz, Nz)
            flat_idx = np.where(inv_idx == uid)[0]          # (N_same,)
            rho_batch = rho_Gz.reshape(-1, Nz)[flat_idx, :].T   # (Nz, N_same)
            prefactor = (2.0 * np.pi / gp) * dz * K
            V_batch = prefactor * (G_mat @ rho_batch)    # (Nz, N_same)
            V_Gz.reshape(-1, Nz)[flat_idx, :] = V_batch.T

        # G|| = 0: 1-D Poisson
        rho_z0 = rho_Gz[0, 0, :].real
        V_z0 = self._solve_gpar0(rho_z0, z_grid, dz, Lz, bc)
        V_Gz[0, 0, :] = V_z0 * K
        # The applied field's own potential, in the solver's gauge: the
        # energy counts it once (not halved like the ions' mutual term).
        if self.external_field != 0.0:
            self._V_ext0 = (V_z0 - self._solve_gpar0(rho_z0, z_grid, dz, Lz, bc,
                                                     field_on=False)) * K
        else:
            self._V_ext0 = np.zeros_like(V_z0)

        return V_Gz

    @staticmethod
    def _green_function_matrix(Gpar: float,
                                z_row: np.ndarray,
                                z_col: np.ndarray,
                                bc: str,
                                Lz_half: float,
                                ) -> np.ndarray:
        """
        Build the (Nz, Nz) Green-function kernel matrix for a given G||.

        G_mat[iz, iz'] = G(G||, z[iz], z[iz']; BC)

        Parameters
        ----------
        Gpar    : G|| [A^-1]  (scalar, > 0)
        z_row   : (Nz, 1) field-point z coordinates [A]
        z_col   : (1, Nz) source-point z coordinates [A]
        bc      : internal code '_vv', '_mm' or '_vm' (bc1 / bc2 / bc3)
        Lz_half : Lz/2 [A]

        Returns
        -------
        G_mat : (Nz, Nz) float array
        """
        Lz = 2.0 * Lz_half
        absz = np.abs(z_row - z_col)        # (Nz, Nz)
        gpLz = Gpar * Lz

        if bc == '_vv':
            # Open boundaries: no images (Otani, Review of Polarography 64,
            # 79 (2018), eq. 5: G = (4 pi / 2 g) exp(-g |z - z'|)).  Bug-fix
            # note (E3): two negative image terms at Lz +/- (z + z') were
            # included -- the images of grounded walls, not of vacuum.
            G_mat = np.exp(-Gpar * absz)
        elif bc == '_vm':
            G_mat = (np.exp(-Gpar * absz)
                      - np.exp(-Gpar * (Lz - z_row - z_col)))
        elif bc == '_mm':
            # Grounded plates at z = +/- Lz/2 (image method):
            #   G = 2 sinh(g(Lz/2 - z_>)) sinh(g(Lz/2 + z_<)) / sinh(g Lz)
            #     = [cosh(g(Lz - |dz|)) - cosh(g(z + z'))] / sinh(g Lz),
            # written with every exponent <= 0 (|z + z'| <= Lz), and
            # -> exp(-g |dz|) as Lz -> infinity.  Bug-fix note (ADR-0003 T6,
            # E3): this used [exp(g(|dz| - 2Lz)) + exp(-g|dz|)] / (1 -
            # exp(-2 g Lz)) / g -- an extra 1/g on top of the 2 pi / g
            # prefactor, and without the z + z' (image) terms.
            zs = z_row + z_col
            G_mat = ((np.exp(-Gpar * absz) + np.exp(-Gpar * (2.0 * Lz - absz))
                      - np.exp(-Gpar * (Lz - zs)) - np.exp(-Gpar * (Lz + zs)))
                     / (1.0 - np.exp(-2.0 * gpLz)))
        else:
            raise ValueError(f"Unknown bc '{bc}'.")

        return G_mat

    def _solve_gpar0(self,
                      rho_z: np.ndarray,
                      z_grid: np.ndarray,
                      dz: float,
                      Lz: float,
                      bc: str,
                      field_on: bool = True,
                      ) -> np.ndarray:
        """
        Solve the 1-D Poisson equation for the G|| = 0 component.

        d^2V/dz^2 = -4pi rho(z)

        Boundary conditions
        --------------------
        bc1 (_vv): E = 0 at z = -Lz/2 (and at +Lz/2 for a neutral slab).
             Requires int rho dz = 0  (charge neutrality).
        bc3 (_vm): E = 0 at z = -Lz/2 (vacuum), V = 0 at z = +Lz/2 (metal
             ground).  Supports net charge.
        bc2 (_mm): V equal at z = +/-Lz/2 (metal at both ends).
             External field E_ext added as linear potential.

        Parameters
        ----------
        rho_z  : (Nz,) charge density on z grid [e/A^3]
        z_grid : (Nz,) z coordinates [A]
        dz     : grid spacing [A]
        Lz     : cell length along z [A]
        bc     : boundary condition string

        Returns
        -------
        V_z : (Nz,) electrostatic potential [eV/e] (before K multiplication)

        Precision note
        ---------------
        Both integrations use ``scipy.integrate.cumulative_trapezoid``
        (O(dz^2) global error) rather than a plain ``np.cumsum`` Riemann
        sum (O(dz) global error, i.e. only linear convergence with grid
        refinement). An earlier version used the plain Riemann sum here;
        verified by direct finite-difference force checks that this was
        the dominant source of a slowly-converging (linear in dz, not the
        expected quadratic) discrepancy between the analytic and
        finite-difference z-force for the G||=0 (planar-averaged)
        contribution -- doubling ``n_grid_z`` roughly halved the residual
        rather than quartering it, the signature of a first-order
        quadrature. The (much smaller) G||!=0 contributions do not go
        through this cumulative integration and were not affected.
        """
        # The field here is in the Gaussian units of V_z (e/A^2, multiplied
        # by K = e^2/(4 pi eps0) = 14.40 eV A later); the user's field is
        # in V/A.  Bug-fix note (ADR-0003 T6, E2): it was added unconverted,
        # so the applied field came out K = 14.40 times too strong.
        E_ext = (self.external_field / _ESM_EV_PER_E2_ANG      # [e/A^2]
                 if field_on else 0.0)

        # First integration: E_z(z) = 4 pi int_{z_min}^{z} rho dz' + C1
        # E = -dV/dz and d2V/dz2 = -4 pi rho give dE/dz = +4 pi rho.
        # Bug-fix note (ADR-0003 T6, E3): the sign was negative, which
        # reversed the whole planar-average (G|| = 0) energy -- the pair
        # energies were off by twice their capacitor term, growing with
        # the dipole length (1.68 eV for a +-1 pair 3 A apart along z).
        E_field = 4.0 * np.pi * cumulative_trapezoid(rho_z, dx=dz, initial=0.0)   # (Nz,)

        if bc == '_vv':
            # BC1: open at both ends (C1 = 0, see below)
            Q_total = float(np.sum(rho_z) * dz)
            if abs(Q_total) > 1e-4:
                warnings.warn(
                    f"ESM BC1: slab is not charge-neutral "
                    f"(int rho dz = {Q_total:.3f} e/A^2).  "
                    "BC1 requires int rho dz = 0; consider using BC2 or BC3.",
                    UserWarning, stacklevel=3)
            # open boundary: E(z_min) = 0 by construction, and E(z_max) =
            # 4 pi Q = 0 for a neutral slab.  (Was: minus the cell mean of
            # E, a spurious uniform field proportional to the dipole.)

        elif bc == '_vm':
            # vacuum | slab | metal at +Lz/2: no field at the vacuum end,
            # E(z_min) = 0, which the integral from z_min already gives; the
            # metal takes whatever field the net charge needs.  (Was: the
            # field averaged over the right 1/8 of the cell set to zero --
            # wrong at the metal, where a charged slab induces a field.)
            E_field += E_ext

        elif bc == '_mm':
            # metal | slab | metal: both plates at the same potential, i.e.
            # int E dz = 0 over the cell, plus the applied field.  (Was:
            # the fields averaged over the two outer 1/8 of the cell.)
            E_field -= _trapz_mean(E_field, dz)
            E_field += E_ext

        # Second integration: V(z) = -int E_z dz + C2
        V_z = -cumulative_trapezoid(E_field, dx=dz, initial=0.0)

        # Fix integration constant C2 based on BC
        if bc == '_vv':
            # Zero-mean potential (arbitrary gauge)
            V_z -= float(np.mean(V_z))
        elif bc == '_vm':
            # V = 0 at right boundary (metal)
            V_z -= V_z[-1]
        elif bc == '_mm':
            # V = 0 at both metal walls -> set to average of endpoints
            V_z -= (V_z[0] + V_z[-1]) / 2.0

        return V_z

    # ------------------------------------------------------------------ #
    #  Step 3: Energy                                                     #
    # ------------------------------------------------------------------ #

    def _compute_energy(self,
                         rho_Gz: np.ndarray,
                         V_Gz: np.ndarray,
                         dz: float,
                         ) -> float:
        """
        Electrostatic energy E = (1/2) intint rho(r) V(r) d^3r

        In the G||-z representation (Parseval):
            E = (1/2) Lx Ly sum_{G||} int rho*(G||,z) V(G||,z) dz

        Self-energy of Gaussian charges is subtracted.

        Returns
        -------
        energy : float [eV]
        """
        Lx = self._Lx
        Ly = self._Ly
        area = Lx * Ly

        # Interaction energy (Parseval)
        integrand = (np.conj(rho_Gz) * V_Gz).real   # (Nx, Ny, Nz)
        E_int = 0.5 * area * float(np.sum(integrand)) * dz

        # Subtract the Gaussian self-energy E_self = K sum_I q_I^2 / (2 sqrt(pi)
        # sigma) of a normalised Gaussian with standard deviation sigma per
        # axis (_build_charge_density_2d).  Bug-fix note (ADR-0003 T6, E1):
        # it was also divided by the cell area Lx Ly, which has no place in
        # it (it made the term 1/area too small: ~8 eV per unit charge was
        # left in the energy of a 16 x 16 A cell).
        sig = self.sigma
        E_self = float(np.sum(self._charges**2)) / (
            2.0 * sig * np.sqrt(np.pi)) * _ESM_EV_PER_E2_ANG

        # (1/2) int rho V counted the applied field's potential with a
        # factor 1/2 as well; the interaction with an external potential
        # is int rho V_ext, so add the other half (ADR-0008).
        V_ext0 = getattr(self, "_V_ext0", None)
        if V_ext0 is not None:
            E_int += 0.5 * area * float(np.sum(rho_Gz[0, 0, :].real * V_ext0)) * dz

        return E_int - E_self

    # ------------------------------------------------------------------ #
    #  Step 4: Forces                                                     #
    # ------------------------------------------------------------------ #

    def _compute_forces(self,
                         positions: np.ndarray,
                         charges: np.ndarray,
                         V_Gz: np.ndarray,
                         Nx: int,
                         Ny: int,
                         Nz: int,
                         Lx: float,
                         Ly: float,
                         Lz: float,
                         GX: np.ndarray,
                         GY: np.ndarray,
                         ) -> np.ndarray:
        """
        Forces on atoms: F_I = -q_I grad V(r_I)

        Gradient in xy: via inverse FFT of iG_x V and iG_y V.
        Gradient in z : via central finite difference on V_Gz.

        The gradient fields are sampled at each atomic position using
        **trilinear interpolation** (8-point stencil) rather than
        nearest-neighbour snapping.  This ensures continuous forces as
        atoms move between grid cells, which is essential for stable MD.

        Bug-fix notes: (a) the z-index computed below (``fz``) hard-wraps
        via ``% Nz`` with no bounds check. If an atom's z ever leaves the
        padded ``[-Lz/2, +Lz/2)`` region (insufficient vacuum, a bad
        MD/relax step, a GCMC insertion placed too close to the boundary),
        the wrap silently aliases the interpolation stencil to the
        *opposite* side of an explicitly open/metal (non-periodic-in-z)
        boundary -- physically nonsensical for BC1/2/3. This is now
        checked explicitly and raises instead of silently aliasing.
        (b) ``V_Gz`` (and hence ``dV_d*_Gz``) is built by explicit
        ``exp(i*G.r)`` phase sums in :meth:`_build_charge_density_2d` /
        :meth:`_solve_poisson_esm`, never through an actual
        ``np.fft.fft2`` call -- it is an *unnormalised* forward-DFT-like
        quantity. ``np.fft.ifft2`` divides by ``Nx*Ny`` as part of its
        normal convention, which is only correct when its input truly came
        from an unnormalised ``fft2``; applied here it introduces a
        spurious extra ``1/(Nx*Ny)`` factor into every gradient component
        (x, y, *and* z, since all three go through the same ``ifft2`` to
        convert the xy G-space representation back to real space). An
        earlier version did not compensate for this: verified by finite
        difference that the analytic-to-numeric force ratio scaled as
        almost exactly ``1/(n_grid_xy**2)`` when ``n_grid_xy`` was varied
        (24 -> 48 -> 96 changed the ratio by almost exactly 4x each time),
        which is the signature of a missing ``Nx*Ny`` normalisation, not a
        discretisation/convergence artefact (the ratio was independent of
        the z-grid size ``n_grid_z``). Fixed by multiplying the ``ifft2``
        result by ``Nx*Ny`` below.

        Returns
        -------
        forces : (N, 3) [eV/A]

        Bug-fix note (Phase 5, vectorisation): this method used to loop
        over atoms one at a time in pure Python. There is no loop-carried
        dependency between atoms (each is an independent trilinear gather
        from the three precomputed gradient grids), so it is rewritten
        here as NumPy fancy-indexing over all atoms at once instead --
        the formulas themselves are unchanged, only replayed on arrays
        rather than scalars, and were confirmed to give bit-for-bit
        identical forces to the original per-atom loop on every existing
        ESM test structure.
        """
        # Energy-consistent forces (ADR-0008).  With E = (1/2) A sum_G int
        # rho* V dz and the atom's density rho_I(G, z) = (q_I / A) f(G)
        # exp(-i G.R_I) g(z - Z_I) as built in _build_charge_density_2d,
        #     F_I = -A sum_G int Re[(d rho_I* / d R_I) V] dz,
        # the Gaussian-weighted field at the atom: its own field cancels
        # exactly, and the z derivative is that of the sampled Gaussian, so
        # these are the gradient of the discretised energy.  Bug-fix note:
        # the field used to be interpolated trilinearly at the atom centre
        # from the total potential, own Gaussian included; that self-field
        # does not vanish on a grid, so the forces depended on the grid and
        # did not sum to zero (1.7 eV/A off in x for a +-1 quartet, 96 x 512
        # grid; only z-aligned pairs had been checked).
        dz = Lz / Nz
        sig = self.sigma
        z_g = self._z_grid
        Nxy = Nx * Ny
        Vg = V_Gz.reshape(Nxy, Nz)                                  # (G, z)
        fform = np.exp(-0.5 * (GX**2 + GY**2) * sig**2).reshape(Nxy)
        gx, gy = GX.reshape(Nxy), GY.reshape(Nxy)
        dzm = z_g[None, :] - positions[:, 2][:, None]               # (N, z)
        dzm -= Lz * np.round(dzm / Lz)
        g = np.exp(-0.5 * dzm**2 / sig**2) / (sig * np.sqrt(2.0 * np.pi))
        dg = g * dzm / sig**2                                       # d g / d Z
        W = dz * (g @ Vg.T)                                         # (N, G)
        U = dz * (dg @ Vg.T)
        ph = np.exp(1j * (positions[:, 0][:, None] * gx[None, :]
                          + positions[:, 1][:, None] * gy[None, :]))  # (N, G)
        a = fform[None, :] * ph
        forces = np.empty((len(charges), 3))
        forces[:, 0] = -charges * np.sum((1j * gx[None, :] * a * W).real, axis=1)
        forces[:, 1] = -charges * np.sum((1j * gy[None, :] * a * W).real, axis=1)
        forces[:, 2] = -charges * np.sum((a * U).real, axis=1)
        return forces

    # ------------------------------------------------------------------ #
    #  Step 5: Stress                                                     #
    # ------------------------------------------------------------------ #

    def _compute_stress(self,
                         rho_Gz: np.ndarray,
                         V_Gz: np.ndarray,
                         Gpar: np.ndarray,
                         GX: np.ndarray,
                         GY: np.ndarray,
                         lattice: np.ndarray,
                         dz: float,
                         ) -> np.ndarray:
        """
        In-plane stress, the strain derivative of this class's energy
        (ADR-0008); z rows and columns are zero (Lz places the boundaries,
        so the cell is not strained along z with ESM).

        Under an in-plane strain at fixed fractional coordinates G.R is
        unchanged; the density's 1/A factor and |G| (through the Gaussian
        form factor exp(-|G|^2 sigma^2 / 2) and the kernel) change::

            sigma_ab = (1/V) [E_ion delta_ab + sum_{G != 0} (G_a G_b / |G|)
                       (A/2) dz Re{rho* [(-1/|G| - 2 |G| sigma^2) V
                                         + K (2 pi / |G|) dz dG_bc/d|G| rho]}]

        E_ion excludes the applied field (its energy does not depend on A).
        Bug-fix note: this was a Maxwell-stress estimate, -(1/V) int
        dV/dr_a dV/dr_b / 4 pi, which does not match the Gaussian energy
        with its self terms removed (xx and yy off by 7x and 4x, xy of the
        wrong sign, on a +-1 quartet against a strain finite difference).
        """
        K = _ESM_EV_PER_E2_ANG
        area = self._Lx * self._Ly
        V = abs(np.linalg.det(lattice))
        Nz = rho_Gz.shape[2]
        V_ion = V_Gz.copy()
        V_ext0 = getattr(self, "_V_ext0", None)
        if V_ext0 is not None:
            V_ion[0, 0, :] -= V_ext0
        E_ion = 0.5 * area * float(np.sum((np.conj(rho_Gz) * V_ion).real)) * dz
        sig = self.sigma
        z = self._z_grid
        Lz_h = self._Lz / 2.0
        gp = Gpar.ravel()
        gx, gy = GX.ravel(), GY.ravel()
        rho = rho_Gz.reshape(-1, Nz)
        Vf = V_ion.reshape(-1, Nz)
        S2 = np.zeros((2, 2))
        _scale = 10 ** (8 - int(np.floor(np.log10(max(gp.max(), 1e-30)))) - 1)
        keys = np.round(gp * _scale) / _scale
        for k in np.unique(keys):
            if k < 1e-10:
                continue
            idx = np.where(keys == k)[0]
            dG = self._green_function_dmatrix(k, z[:, None], z[None, :],
                                              self._bc, Lz_h)
            r = rho[idx]                                      # (n, Nz)
            t1 = (np.conj(r) * Vf[idx]).real.sum(axis=1) * (-1.0 / k - 2.0 * k * sig**2)
            t2 = (np.conj(r) * (r @ dG.T)).real.sum(axis=1) * K * (2.0 * np.pi / k) * dz
            w = 0.5 * area * dz * (t1 + t2) / k                # per G
            S2[0, 0] += np.sum(w * gx[idx] ** 2)
            S2[1, 1] += np.sum(w * gy[idx] ** 2)
            S2[0, 1] += np.sum(w * gx[idx] * gy[idx])
        S2[1, 0] = S2[0, 1]
        stress = np.zeros((3, 3))
        stress[:2, :2] = (E_ion * np.eye(2) + S2) / V
        return stress

    @staticmethod
    def _green_function_dmatrix(Gpar, z_row, z_col, bc, Lz_half):
        """d/dG|| of _green_function_matrix (same bc codes)."""
        Lz = 2.0 * Lz_half
        absz = np.abs(z_row - z_col)
        zs = z_row + z_col
        e1 = np.exp(-Gpar * absz)
        if bc == '_vv':
            return -absz * e1
        if bc == '_vm':
            e3 = np.exp(-Gpar * (Lz - zs))
            return -absz * e1 + (Lz - zs) * e3
        if bc == '_mm':
            e2 = np.exp(-Gpar * (2.0 * Lz - absz))
            e3 = np.exp(-Gpar * (Lz - zs))
            e4 = np.exp(-Gpar * (Lz + zs))
            N = e1 + e2 - e3 - e4
            dN = -absz * e1 - (2.0 * Lz - absz) * e2 + (Lz - zs) * e3 + (Lz + zs) * e4
            D = 1.0 - np.exp(-2.0 * Gpar * Lz)
            dD = 2.0 * Lz * np.exp(-2.0 * Gpar * Lz)
            return (dN * D - N * dD) / D**2
        raise ValueError(f"Unknown bc '{bc}'.")

    # ------------------------------------------------------------------ #
    #  Diagnostic / post-processing helpers                               #
    # ------------------------------------------------------------------ #

    def get_potential_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the planar-averaged electrostatic potential profile V(z).

        Returns
        -------
        z_grid : (Nz,) float [A]
        V_z    : (Nz,) float [eV]

        Raises
        ------
        RuntimeError  if called before the first ``__call__``.
        """
        if self._V_z is None:
            raise RuntimeError(
                "ESMCalculator.get_potential_profile(): call compute first.")
        return self._z_grid.copy(), self._V_z.copy()

    def macroscopic_average(self,
                             lambda_avg: Optional[float] = None,
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the macroscopic average of the planar-averaged potential Vbar(z).

        The macroscopic average removes rapid oscillations on the scale of
        the lattice period (Baldereschi et al., 1988) by convolving V(z)
        with a rectangular window of width ``lambda_avg``.

        Used for band alignment, work function, and interface dipole
        analysis.

        Parameters
        ----------
        lambda_avg : averaging window width [A].
                     Default: Lz / 4 (covers roughly one lattice period for
                     typical slab+vacuum geometries).

        Returns
        -------
        z_grid  : (Nz,) float [A]
        V_macro : (Nz,) float [eV]  -- macroscopically averaged potential.

        Raises
        ------
        RuntimeError  if called before the first ``__call__``.
        """
        if self._V_z is None:
            raise RuntimeError(
                "ESMCalculator.macroscopic_average(): call compute first.")

        z_g = self._z_grid
        V_z = self._V_z
        Nz = len(V_z)
        dz = self._dz
        Lz = self._Lz

        if lambda_avg is None:
            lambda_avg = Lz / 4.0

        # Window width in grid points (must be odd for symmetry)
        n_win = max(1, int(round(lambda_avg / dz)))
        if n_win % 2 == 0:
            n_win += 1

        # Rectangular window kernel (normalised), zero-padded to Nz
        kernel = np.zeros(Nz)
        half = n_win // 2
        kernel[:half + 1] = 1.0 / n_win
        kernel[Nz - half:] = 1.0 / n_win    # wrap-around tail

        # Convolution via FFT
        V_macro = np.fft.ifft(np.fft.fft(V_z) * np.fft.fft(kernel)).real

        return z_g.copy(), V_macro

    def charge_density_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the planar-averaged charge density profile rho_i(z).

        This is simply the G|| = 0 component of rho(G||, z), scaled by the
        2-D unit-cell area A = Lx x Ly.

        Returns
        -------
        z_grid : (Nz,) float [A]
        rho_z  : (Nz,) float [e/A]   -- charge per unit length along z.

        Raises
        ------
        RuntimeError  if called before the first ``__call__``.
        """
        if self._rho_Gz is None:
            raise RuntimeError(
                "ESMCalculator.charge_density_profile(): call compute first.")
        area = self._Lx * self._Ly
        rho_z = self._rho_Gz[0, 0, :].real * area   # [e/A] (charge per A)
        return self._z_grid.copy(), rho_z

    def work_function(self, e_fermi: float = 0.0) -> float:
        """
        Compute the work function phi = V_vacuum - E_Fermi [eV].

        The vacuum level is estimated by extrapolating the macroscopic
        average potential into the vacuum region using a linear fit.
        For BC2 / BC3 the metal boundary potential is used as a reference.
        For BC1 the average of the macroscopic potential in both vacuum
        wings is used.

        Parameters
        ----------
        e_fermi : Fermi energy [eV].  Default 0.0.

        Returns
        -------
        phi : float [eV]

        Raises
        ------
        RuntimeError  if called before the first ``__call__``.
        """
        if self._V_z is None:
            raise RuntimeError(
                "ESMCalculator.work_function(): call compute first.")

        # Use macroscopic average to suppress atomic-scale oscillations
        _, V_mac = self.macroscopic_average()
        Nz = len(V_mac)
        n_win = max(Nz // 10, 1)

        if self._bc == '_vm':
            # Vacuum is on the left side (z ~= -Lz/2); fit linear trend
            V_left = V_mac[:n_win]
            z_left = self._z_grid[:n_win]
            # Linear fit to extrapolate to z = -Lz/2
            if len(z_left) >= 2:
                slope, intercept = np.polyfit(z_left, V_left, 1)
                V_vac = float(intercept + slope * self._z_grid[0])
            else:
                V_vac = float(np.mean(V_left))

        elif self._bc == '_mm':
            # Both walls are metal at V = 0 by construction
            V_vac = 0.0

        else:  # bc1
            # Average both vacuum wings (left and right)
            V_L = float(np.mean(V_mac[:n_win]))
            V_R = float(np.mean(V_mac[-n_win:]))
            V_vac = (V_L + V_R) / 2.0

        return float(V_vac - e_fermi)

    def surface_dipole_correction(self) -> float:
        """
        Compute the dipole correction energy for BC1 (vacuum | slab | vacuum).

        The correction accounts for the artificial electric field arising
        from the periodic image dipole and is given by:

            E_dip = -(mu_z / Lz) x E_macro

        where mu_z = int z rho(z) dz is the dipole moment per unit area and
        E_macro is the macroscopic field.

        Returns
        -------
        E_dip : float [eV]

        Notes
        -----
        For BC2 and BC3, the metal boundaries naturally screen the dipole;
        this correction is only needed for BC1.
        """
        if self._bc != '_vv':
            return 0.0
        if self._V_z is None:
            raise RuntimeError(
                "ESMCalculator.surface_dipole_correction(): call compute first.")

        z_g = self._z_grid
        rho_z = self._rho_Gz[0, 0, :].real if self._rho_Gz is not None \
            else np.zeros_like(z_g)
        dz = self._dz
        Lz = self._Lz
        area = self._Lx * self._Ly

        # Dipole moment per unit area [e.A/A^2]
        mu_z = float(np.sum(z_g * rho_z) * dz)

        # Macroscopic field: difference of vacuum potentials on two sides
        Nz = len(self._V_z)
        n_win = max(Nz // 8, 1)
        V_L = float(np.mean(self._V_z[:n_win]))
        V_R = float(np.mean(self._V_z[-n_win:]))
        E_mac = (V_R - V_L) / Lz    # [eV/A]

        E_dip = -mu_z * E_mac * area * _ESM_EV_PER_E2_ANG
        return float(E_dip)

    def electron_density_difference(self,
                                     ref_positions: np.ndarray,
                                     ref_charges: Union[dict, np.ndarray],
                                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the differential planar-averaged charge-density delta_rho_i(z).

        Positive values indicate charge accumulation; negative values
        indicate charge depletion relative to the reference.

        Typical use: adsorption-induced charge transfer, interface dipole.

        Parameters
        ----------
        ref_positions : (N_ref, 3) reference atomic positions [A].
        ref_charges   : dict {symbol: charge} or (N_ref,) float array.

        Returns
        -------
        z_grid    : (Nz,) float [A]
        delta_rho : (Nz,) float [e/A]

        Raises
        ------
        RuntimeError  if called before the first ``__call__``.
        """
        if self._rho_Gz is None:
            raise RuntimeError(
                "ESMCalculator.electron_density_difference(): "
                "call compute (via __call__) first.")

        _, rho_slab = self.charge_density_profile()   # [e/A]

        # Resolve reference charges
        ref_pos = np.asarray(ref_positions, dtype=float)
        N_ref = len(ref_pos)
        if isinstance(ref_charges, np.ndarray):
            ref_q = np.asarray(ref_charges, dtype=float)
            if ref_q.shape != (N_ref,):
                raise ValueError(
                    f"electron_density_difference: ref_charges array must "
                    f"have shape ({N_ref},), got {ref_q.shape}.")
        else:
            # dict: determine types from order of keys repeated to fill N_ref
            key_cycle = list(ref_charges.keys())
            ref_types = [key_cycle[i % len(key_cycle)] for i in range(N_ref)]
            ref_q = self._resolve_charges(ref_charges, ref_types)

        rho_Gz_ref = self._build_charge_density_2d(
            ref_pos, ref_q,
            self.n_grid_xy, self.n_grid_xy, self.n_grid_z,
            self._Lz, self._Lx, self._Ly, self.sigma,
            self._z_grid, self._GX, self._GY,
        )
        area = self._Lx * self._Ly
        rho_ref = rho_Gz_ref[0, 0, :].real * area   # [e/A]

        return self._z_grid.copy(), (rho_slab - rho_ref)

    def print_summary(self) -> None:
        """Print a diagnostic summary of the last ESM computation."""
        if self._V_z is None:
            print("ESMCalculator: not yet computed.")
            return
        q_tot = float(np.sum(self._charges))
        z_g, V_z = self.get_potential_profile()
        _, V_mac = self.macroscopic_average()
        _, rho_z = self.charge_density_profile()
        print(f"\n  ESM Summary  (bc={self.bc})")
        print(f"    N atoms       = {self.N}")
        print(f"    Total charge  = {q_tot:+.3f} e")
        print(f"    Lz            = {self._Lz:.2f} A  "
              f"  sigma = {self.sigma:.2f} A")
        print(f"    V(z) range    = [{V_z.min():.3f}, {V_z.max():.3f}] eV")
        print(f"    V_macro range = [{V_mac.min():.3f}, {V_mac.max():.3f}] eV")
        print(f"    rho_i(z) integral = {float(np.sum(rho_z) * self._dz):.3f} e/A^2")
        if self._bc in ('_vm', '_mm'):
            print(f"    Work function = {self.work_function():.3f} eV")
        if self._bc == '_vv':
            print(f"    Dipole corr.  = {self.surface_dipole_correction():.4f} eV")


# ---------------------------------------------------------------------------
#  Point-charge electrostatics with ESM boundary conditions (ADR-0008)
# ---------------------------------------------------------------------------

class ESMCoulomb:
    """
    Coulomb energy, forces and stress of POINT charges under ESM boundary
    conditions -- the ESM counterpart of ``ewald_sum`` / ``PPPMSolver``.

    ``ESMCalculator`` gives the interaction of Gaussian charges of width
    sigma (self terms removed).  The point-charge result adds the
    real-space complement, as in Ewald's split::

        E = E_ESM(sigma) + K sum_{i<j} q_i q_j erfc(r_ij / 2 sigma) / r_ij

    summed with the minimum image in x and y only (z is not periodic).
    erfc(r / 2 sigma) < 1e-7 beyond r = 11 sigma, so the pair sum is cut
    there, which must fit in the in-plane half-width.
    """

    kind = "esm"

    def __init__(self, bc="bc1", sigma=0.5, n_grid_xy=64, n_grid_z=256,
                 external_field=0.0):
        self.bc, self.sigma = bc, float(sigma)
        self.n_grid_xy, self.n_grid_z = int(n_grid_xy), int(n_grid_z)
        self.external_field = float(external_field)
        self.r_real = 11.0 * self.sigma
        self._esm = None

    def _calc(self, lattice, positions, atom_types, q):
        if self._esm is None:
            self._esm = ESMCalculator(
                lattice, positions, atom_types, np.asarray(q, float), bc=self.bc,
                sigma=self.sigma, n_grid_xy=self.n_grid_xy,
                n_grid_z=self.n_grid_z, external_field=self.external_field)
        elif len(q) != self._esm.N or not np.allclose(q, self._esm._charges):
            self._esm.update_geometry(lattice, positions, atom_types,
                                      charges=np.asarray(q, float))
        return self._esm

    def compute(self, lattice, positions, atom_types, q):
        """E [eV], F (N, 3) [eV/A], S (3, 3) [eV/A^3] (compressive-positive)."""
        lattice = np.asarray(lattice, float)
        q = np.asarray(q, float)
        esm = self._calc(lattice, positions, atom_types, q)
        E, F, S = esm(lattice, positions, atom_types)
        e2, f2, s2 = self._real_space(lattice, esm._to_esm_frame(lattice, positions), q)
        return E + e2, F + f2, S + s2

    def _real_space(self, lattice, pos, q):
        from scipy.special import erfc

        N = len(pos)
        F = np.zeros((N, 3))
        S = np.zeros((3, 3))
        widths = [abs(np.linalg.det(lattice)) / np.linalg.norm(np.cross(lattice[1], lattice[2])),
                  abs(np.linalg.det(lattice)) / np.linalg.norm(np.cross(lattice[2], lattice[0]))]
        if self.r_real >= min(widths) / 2:
            raise ValueError(
                f"ESMCoulomb: the real-space complement reaches 11 sigma = "
                f"{self.r_real:.2f} A, more than half the in-plane cell width "
                f"({min(widths) / 2:.2f} A); enlarge the cell in x/y or lower sigma.")
        if N < 2:
            return 0.0, F, S
        ii, jj = np.triu_indices(N, k=1)
        d = pos[jj] - pos[ii]
        s = d @ np.linalg.inv(lattice)
        s[:, :2] -= np.round(s[:, :2])
        d = s @ lattice
        r = np.linalg.norm(d, axis=1)
        m = (r < self.r_real) & (r > 1e-12)
        ii, jj, d, r = ii[m], jj[m], d[m], r[m]
        qq = q[ii] * q[jj] * _ESM_EV_PER_E2_ANG
        a = 1.0 / (2.0 * self.sigma)
        e = qq * erfc(a * r) / r
        # dE/dr of erfc(a r) / r
        dedr = -qq * (erfc(a * r) / r**2
                      + 2.0 * a / np.sqrt(np.pi) * np.exp(-(a * r) ** 2) / r)
        f = (dedr / r)[:, None] * d          # force on i (d = r_j - r_i)
        np.add.at(F, ii, f)
        np.add.at(F, jj, -f)
        V = abs(np.linalg.det(lattice))
        S = -np.einsum("ma,mb->ab", f, d) / V
        return float(e.sum()), F, 0.5 * (S + S.T)
