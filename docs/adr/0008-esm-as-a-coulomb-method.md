---
status: accepted
---

# ESM as a Coulomb method of the force field, not a solver

The Effective Screening Medium method (Otani & Sugino, Phys. Rev. B 73,
115407 (2006)) solves the Poisson equation of a charge distribution
analytically under slab boundary conditions: vacuum or grounded metal at
z = +/- Lz/2.  It describes the electrostatics of a system -- the same job
as Ewald or PPPM under 3D periodicity -- not an atomistic algorithm.  Until
now akitsushima offered it only as a separate command, `esm`, whose
`ESMCalculator` was the whole force function: no short-range potential,
so its `relax:` section collapsed the ions (ADR-0003 T6), and no other
solver could use it.

## Decisions (agreed with the user, 2026-09-26, Q16-1..Q16-4)

1. **`coulomb_method: esm`** joins `ewald` and `pppm` in the potential
   section, with an `esm:` block for its settings (bc, sigma, n_grid_xy,
   n_grid_z, external_field).  The short-range potential plus ESM
   electrostatics is one force function built by `make_force_fn` /
   `_build_force_fn_from_yaml`, so every solver takes it without solver
   code of its own -- the ADR-0007 pattern.  The YAML Buckingham path,
   which returned no force function and so worked only in relax and md,
   returns a complete one for every Coulomb method.
2. **Point charges.**  ESM gives the interaction energy of Gaussian charges
   of width sigma (self terms removed).  The point-charge energy adds the
   real-space complement, as in Ewald's split:
       E = E_ESM(sigma) + K sum_{i<j} q_i q_j erfc(r_ij / 2 sigma) / r_ij
   (lateral minimum image; erfc(r / 2 sigma) < 1e-7 beyond 11 sigma, so
   the pair sum needs r <= 11 sigma inside the in-plane half-width).
   Image charges of the metal walls contribute to that complement only for
   atoms within ~11 sigma / 2 of a wall, which is refused.
3. **z convention.**  Solvers wrap fractional coordinates into [0, 1);
   ESM's grid spans [-Lz/2, Lz/2).  Positions are mapped with
   z -> z - Lz round(z / Lz), i.e. the slab is centred at fractional z = 0
   and the boundaries (vacuum or electrode) sit at fractional z = 1/2 --
   the convention of Quantum ESPRESSO's ESM.  An atom that comes within the
   refusal distance of +/- Lz/2 is an error, not a wrap across the medium.
4. **`esm` command** becomes analysis-only: single point, planar-averaged
   potential, work function, charge density.  Its `relax:` section is
   refused with a pointer to `relax` + `coulomb_method: esm`.
5. **Solver constraints** (refusals are start-up errors):

   | solver | with ESM |
   |---|---|
   | relax | atoms free; `relax_cell` only for a_x and b_y (`lattice_constraints: '1 0 0 0 1 0 0 0 0'`): Lz places the boundaries, and ESMCalculator needs a rectangular in-plane cell, so shear is frozen too |
   | md | NVE, NVT; NPT refused -- the barostat is isotropic and would move the boundaries (an in-plane barostat does not exist yet; the agreed table had "in-plane only", which needs one) |
   | neb, tsa | allowed |
   | phonon | supercell along z must be 1 |
   | gcmc | bc1 refuses inserting a charged species (it requires neutrality); bc2 / bc3 allow it (the electrode compensates) |
   | kappa | refused (no site-energy decomposition; slabs out of scope) |
   | MLIP + ESM | refused for now (a model trained on total energies would count electrostatics twice) |

6. **Verification.**
   (i) point-charge energies and forces against 3D Ewald with a padded
   vacuum and the Yeh-Berkowitz correction (bc1) and against the image
   Ewald references (bc2, bc3) of `example/esm/check_esm.py`;
   (ii) an MgO(001) slab relaxed with Buckingham + ESM(bc1) against
   Buckingham + Ewald with a large vacuum and the slab correction;
   (iii) energy conservation in NVE with ESM;
   (iv) every solver through the CLI on `coulomb_method: esm`.

## What the integration found in ESMCalculator (fixed)

Checking forces and stress of the combined force function against
finite differences exposed three defects the T6 re-run (ADR-0003) had
not reached -- it checked the z force of z-aligned pairs only:

- **Forces** interpolated the field at the atom centre from the total
  potential, the atom's own Gaussian included.  That self-field does not
  vanish on a grid: forces depended on the grid and did not sum to zero
  (1.7 eV/A off in x for a +-1 quartet at 96 x 512).  They are now the
  exact gradient of the discretised energy, the Gaussian-weighted field
  F_I = -q_I sum_G f(G) Re[d(exp(i G.R_I)) int g(z - Z_I) V(G, z) dz]:
  within 6e-9 eV/A of central differences even at 32 x 256, and summing
  to zero.
- **Stress** was a Maxwell-stress estimate that does not match the
  Gaussian energy with its self terms removed (xx, yy off by 4-7x, xy of
  the wrong sign).  It is now the in-plane strain derivative of that
  energy (density 1/A factor, |G| in the form factor and the kernel,
  dG_bc/d|G| for each bc), matching finite differences within their
  noise (~3e-5 eV/A^3: the energy is a difference of ~1e3 eV sums); z rows
  and columns are zero.
- **The applied field's energy** was halved with the ions' mutual term;
  the interaction with an external potential is int rho V_ext.  Energy
  and forces now agree with a field on (6e-9 eV/A).

## Verification results (2026-09-26)

- ESMCoulomb point-charge energies: Gaussian reference + erfc complement
  within 5 meV per ion at n_grid_z 512 (all bc); Buckingham + ESM forces
  and in-plane stress = energy derivatives (tests/test_esm_coulomb.py).
- Symmetric MgO(001) slab, 72 ions: ESM bc1 (n_grid_z 1024) against 3D
  Ewald in a 120 A cell within 5 meV per ion.
- All six solvers through the CLI on `coulomb_method: esm`: relax, md
  (NVT), phonon (supercell [1, 1, 1]), gcmc (bc3, inserting Mg2+), neb,
  tsa -- exit status 0 (example/esm/check_esm_solvers.py, stage cli).
- (ii) The slab relaxed at fixed cell (L-BFGS, fmax 1e-3 eV/A, both
  converged) with ESM bc1 (n_grid_z 1024) and with Ewald in a 120 A cell:
  positions within 5.3e-5 A RMSD, energy within 4.4 meV per ion: PASS.
- (iii) NVE with ESM bc1 (n_grid_z 512), 300 K start, 0.49 ps at 1 fs:
  total-energy range 0.062 meV/atom, drift 0.006 meV/atom/ps: PASS.

## Consequences

- ESM results now include the short-range potential; the collapse of the
  old `esm` relax is gone with the section.
- Buckingham (with charges) works in gcmc, neb, tsa, phonon and kappa
  through YAML for the first time -- with Ewald or PPPM as well as ESM.
- An in-plane barostat and constant-potential (constant mu_e) ESM are
  natural next steps, not part of this ADR.
