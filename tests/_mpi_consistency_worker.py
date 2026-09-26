"""Standalone worker script for test_mpi.py: computes SW/Tersoff/EAM
energy/forces/stress and the generic fitting loss function on fixed,
small structures, then (rank 0 only) saves the results to the .npz path
given as sys.argv[1].

Run via:  mpirun -np N python3 _mpi_consistency_worker.py <out.npz>

Not collected by pytest itself (leading underscore, no `test_` prefix);
test_mpi.py invokes it as a subprocess for several -np values and
compares the saved results.
"""
import sys

import numpy as np

from akitsushima.engine import EAMPotential, StillingerWeberPotential, TersoffPotential
from akitsushima.fitting.loss import generic_loss_function
from akitsushima.fitting.parameter_sets import LJParameterSet, StillingerWeberParameterSet
from akitsushima.utils.mpi import _mpi_rank, local_only


def _supercell(lattice, frac, types, n):
    frac = np.concatenate([(frac + [i, j, k]) / n for i in range(n)
                           for j in range(n) for k in range(n)])
    lattice = n * lattice
    return lattice, frac @ lattice, list(types) * n ** 3


def _fcc_al(a=4.05):
    frac = np.array([(0.0, 0.0, 0.0), (0.5, 0.5, 0.0),
                     (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)])
    lattice = np.eye(3) * a
    return lattice, frac @ lattice, ["Al"] * 4


def _diamond_si(a=5.43, n=2):
    # n = 2: the 8-atom cell's half-width (2.715 A) cannot host the SW
    # (3.77 A) or Tersoff (3.0 A) cutoff under the minimum-image convention.
    base = np.array([(0.0, 0.0, 0.0), (0.5, 0.5, 0.0),
                     (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)])
    shift = np.array([0.25, 0.25, 0.25])
    frac = np.concatenate([base, base + shift], axis=0)
    return _supercell(np.eye(3) * a, frac, ["Si"] * 8, n)


def _perturb(positions, magnitude, seed):
    rng = np.random.default_rng(seed)
    return positions + rng.uniform(-magnitude, magnitude, size=positions.shape)


def _rocksalt_nacl(a=5.64, n=2):
    na_frac = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    cl_frac = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5)]
    frac = np.array(na_frac + cl_frac)
    return _supercell(np.eye(3) * a, frac, ["Na"] * 4 + ["Cl"] * 4, n)


def main():
    out_path = sys.argv[1]

    # ---- Stillinger-Weber (Si, diamond structure) ----------------------
    lat_si, pos_si, types_si = _diamond_si()
    pos_si = _perturb(pos_si, 0.05, seed=0)
    sw_params = {"epsilon": 2.1682, "sigma": 2.0951, "A": 7.049556,
                "B": 0.6022246, "p": 4, "q": 0, "a": 1.80,
                "lambda": 21.0, "gamma": 1.20, "costheta0": -0.333333}
    sw = StillingerWeberPotential(params=sw_params, species=["Si"])
    e_sw, f_sw, s_sw = sw.compute(lat_si, pos_si, types_si)

    # ---- Tersoff (Si, diamond structure) -------------------------------
    t_params = {"A": 1830.8, "B": 471.18, "lambda1": 2.4799,
               "lambda2": 1.7322, "lambda3": 1.7322, "beta": 1.1e-6,
               "n": 0.78734, "c": 100390.0, "d": 16.218, "h": -0.59826,
               "R": 2.85, "D": 0.15}
    tersoff = TersoffPotential(params=t_params, species=["Si"])
    e_te, f_te, s_te = tersoff.compute(lat_si, pos_si, types_si)

    # ---- EAM (Al, FCC structure) ----------------------------------------
    lat_al, pos_al, types_al = _fcc_al()
    pos_al = _perturb(pos_al, 0.1, seed=1)

    def rho_fn(r):
        return np.exp(-r / 1.5)

    def F_fn(rho):
        return -0.5 * np.sqrt(np.maximum(rho, 1e-12))

    def phi_fn(r):
        return 500.0 * np.exp(-r / 0.4)

    eam = EAMPotential(F_fn, rho_fn, phi_fn, r_cut=6.0, species=["Al"])
    e_eam, f_eam, s_eam = eam.compute(lat_al, pos_al, types_al)

    # ---- local_only() regression check (code-check finding, same class
    # as F16) ---------------------------------------------------------
    # SW/Tersoff/EAM's scalar `energy` used to be combined across ranks
    # with a raw `_mpi_comm.allreduce()` inside `compute()`, which (unlike
    # the `_mpi_allreduce_sum()` used for forces/stress) never checked
    # `_mpi_is_active()` and so ignored `local_only()` suppression. Inside
    # `with local_only():`, `_mpi_domain_split` returns the FULL range on
    # EVERY rank (no partial-sum splitting happens), so each rank
    # independently computes the complete, correct energy for the SAME
    # structure below -- the true value must therefore come out IDENTICAL
    # across -np 1/2/4. Before the fix, the leftover raw allreduce still
    # summed all ranks' (identical) energies together, inflating the
    # result by exactly n_ranks (2x under -np 2, 4x under -np 4).
    with local_only():
        e_sw_local, _, _ = sw.compute(lat_si, pos_si, types_si)
        e_te_local, _, _ = tersoff.compute(lat_si, pos_si, types_si)
        e_eam_local, _, _ = eam.compute(lat_al, pos_al, types_al)

        # Also exercise the multi-species (nested-triple) SW code path
        # (Phase 5's own `_compute_multi_species`, which had an
        # independent copy of the same bug) with a genuinely heteroatomic
        # NaCl structure.
        lat_nacl, pos_nacl, types_nacl = _rocksalt_nacl()
        pos_nacl = _perturb(pos_nacl, 0.05, seed=2)
        sw_multi_params = {
            ("Na", "Cl", "Cl"): {
                "epsilon": 2.0, "sigma": 2.0951, "A": 7.05, "B": 0.60,
                "p": 4, "q": 0, "a": 1.80, "lambda": 15.0, "gamma": 1.2,
                "costheta0": -1.0 / 3.0,
            },
            ("Cl", "Na", "Na"): {
                # Two-body inputs identical to (Na,Cl,Cl): both records
                # describe the one Na-Cl pair, and StillingerWeberPotential
                # refuses them otherwise (the energy would depend on atom
                # order).  lambda/gamma stay different, which is what keeps
                # a transposed three-body table lookup detectable.
                "epsilon": 2.0, "sigma": 2.0951, "A": 7.05, "B": 0.60,
                "p": 4, "q": 0, "a": 1.80, "lambda": 12.0, "gamma": 1.1,
                "costheta0": -1.0 / 3.0,
            },
        }
        sw_multi = StillingerWeberPotential(sw_multi_params, species=["Na", "Cl"])
        e_sw_multi_local, _, _ = sw_multi.compute(lat_nacl, pos_nacl, types_nacl)

    # ---- Fitting loss function (LJ, several perturbed Al structures) ---
    pair = frozenset({"Al"})
    param_set = LJParameterSet(
        pairs=[pair], species=["Al"],
        initial_lj={pair: {"epsilon": 0.4, "sigma": 2.5}},
        r_cut=6.0, shift=True)

    structures = []
    for seed in range(6):
        p = _perturb(pos_al, 0.05, seed=100 + seed)
        structures.append({
            "lattice": lat_al, "positions": p, "atom_types": types_al,
            "natoms": len(types_al),
            "volume": float(abs(np.linalg.det(lat_al))),
            "ref_energy": -0.4, "ref_forces": np.zeros((4, 3)),
            "ref_stress": np.zeros((3, 3)),
        })
    weights = {"energy": 1.0, "forces": 1.0, "stress": 0.1,
              "r0": 1.0, "eps0": 1.0}
    loss = generic_loss_function(
        param_set.x0, param_set, structures, weights,
        r_cut=6.0, ewald_alpha=None)

    # ---- Fitting loss function with SW (regression check for the same
    # local_only()-bypass bug, exercised through the REAL call path:
    # generic_loss_function's own per-structure loop is what wraps
    # compute_all_general (and hence SW.compute()) in `with local_only():`
    # -- see fitting/loss.py). Several distinct Si structures so that
    # different ranks own genuinely different structures under -np 2/4. --
    sw_param_set = StillingerWeberParameterSet(
        initial_sw=sw_params, species=["Si"])
    sw_structures = []
    for seed in range(6):
        p = _perturb(pos_si, 0.03, seed=200 + seed)
        sw_structures.append({
            "lattice": lat_si, "positions": p, "atom_types": types_si,
            "natoms": len(types_si),
            "volume": float(abs(np.linalg.det(lat_si))),
            "ref_energy": -4.0, "ref_forces": np.zeros((len(types_si), 3)),
            "ref_stress": np.zeros((3, 3)),
        })
    loss_sw = generic_loss_function(
        sw_param_set.x0, sw_param_set, sw_structures, weights,
        r_cut=4.0, ewald_alpha=None)

    if _mpi_rank == 0:
        np.savez(out_path,
                 e_sw=e_sw, f_sw=f_sw, s_sw=s_sw,
                 e_te=e_te, f_te=f_te, s_te=s_te,
                 e_eam=e_eam, f_eam=f_eam, s_eam=s_eam,
                 loss=loss,
                 e_sw_local=e_sw_local, e_te_local=e_te_local,
                 e_eam_local=e_eam_local, e_sw_multi_local=e_sw_multi_local,
                 loss_sw=loss_sw)


if __name__ == "__main__":
    main()
