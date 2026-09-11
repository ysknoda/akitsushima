"""``akitsushima neb`` -- Climbing-Image NEB reaction-path search from a
YAML config.
"""
import os
import time as _time

from ..io.poscar import parse_poscar
from ..solvers import NEBChain, NEBOptimiser
from ..solvers.config_bridge import _build_force_fn_from_yaml
from .common import (
    _check_poscar,
    _check_yaml,
    _print_header,
    _print_output_files,
    _print_results_header,
    _print_timing_summary,
)

__all__ = ["cmd_neb"]


def cmd_neb(args) -> int:
    """Climbing-Image NEB from YAML input."""
    cfg = _check_yaml(args.yaml)
    if cfg is None:
        return 1

    struct_cfg = cfg.get("structure", {})
    pot_cfg    = cfg.get("potential", {})
    neb_cfg    = cfg.get("neb",       {})
    verbose    = neb_cfg.get("verbose", True) and not getattr(args, "quiet", False)

    force_fn, r_cut, _, _ = _build_force_fn_from_yaml(pot_cfg)
    _print_header("Climbing-Image NEB", cfg, pot_cfg=pot_cfg)
    print(f"\n[Settings]")
    print(f"  n_images  : {neb_cfg.get('n_images',5)}")
    print(f"  k_spring  : {neb_cfg.get('k_spring',5.0)} eV/A^2")
    print(f"  climbing  : {neb_cfg.get('climbing',True)}")
    print(f"  fmax      : {neb_cfg.get('fmax',0.05)} eV/A")

    output_dir = neb_cfg.get("output_dir", "./neb_output")
    os.makedirs(output_dir, exist_ok=True)

    k_spring   = neb_cfg.get("k_spring", 5.0)
    climbing   = neb_cfg.get("climbing", True)
    skin_width = neb_cfg.get("skin_width", 0.5)

    explicit_images = struct_cfg.get("images", None)
    if explicit_images:
        images_pos = []
        for p in explicit_images:
            d = _check_poscar(p)
            if d is None: return 1
            images_pos.append(d["positions"])
        ref = parse_poscar(explicit_images[0])
        lattice    = ref["lattice"]
        atom_types = ref["atom_types"]
        chain = NEBChain(
            images_pos=images_pos, lattice=lattice, atom_types=atom_types,
            force_fn=force_fn, k_spring=k_spring, climbing=climbing,
            r_cut=r_cut, skin_width=skin_width)
    else:
        path_is = struct_cfg.get("poscar_initial", "POSCAR_initial")
        path_fs = struct_cfg.get("poscar_final",   "POSCAR_final")
        d_is = _check_poscar(path_is);  d_fs = _check_poscar(path_fs)
        if d_is is None or d_fs is None: return 1
        lattice    = d_is["lattice"]
        atom_types = d_is["atom_types"]
        n_img      = int(neb_cfg.get("n_images", 5))
        # Bug-fix note (S5): the original called
        # NEBChain.linear_interpolation(pos_init, pos_final, n_images=,
        # lattice=) -- missing the required atom_types/force_fn arguments
        # (an immediate TypeError) -- and even setting that error aside,
        # assigned the return value (a fully-constructed NEBChain) to
        # `images_pos`, which was then passed AGAIN into a second,
        # separate NEBChain(images_pos=...) call as if it were a plain
        # list of position arrays. Fixed by passing every required
        # argument and using the NEBChain linear_interpolation() already
        # builds directly, with no second construction.
        chain = NEBChain.linear_interpolation(
            d_is["positions"], d_fs["positions"], lattice, atom_types,
            force_fn, n_images=n_img, k_spring=k_spring, climbing=climbing,
            r_cut=r_cut, skin_width=skin_width)
        print(f"\n[Images]")
        print(f"  IS      : {path_is}")
        print(f"  FS      : {path_fs}")
        print(f"  n_images: {n_img} (interpolated)")

    opt = NEBOptimiser(
        chain=chain, fmax=neb_cfg.get("fmax",0.05),
        max_steps=neb_cfg.get("max_steps",500),
        output_dir=output_dir, verbose=verbose, timing=True)

    t0 = _time.perf_counter()
    result = opt.run()
    t_run = _time.perf_counter() - t0

    _print_results_header("neb")
    # Bug-fix note (S7): the original read result.get("image_energies",
    # []), a key NEBOptimiser.run() has never returned (its actual keys
    # are converged/n_steps/activation_energy/reverse_barrier/
    # climbing_image_pos/climbing_image_index/reaction_path/chain/
    # timing) -- so this whole pathway summary silently never printed.
    chain_result = result.get("chain")
    energies = list(chain_result.energies) if chain_result is not None else []
    ci_idx = result.get("climbing_image_index", -1)   # 0-indexed among intermediates
    if energies:
        print(f"\n[NEB pathway  ({len(energies)} images)]")
        for i, e in enumerate(energies):
            tag = " <- TS" if (i - 1) == ci_idx else ""
            print(f"  Image {i:2d}: E = {e:.6f} eV{tag}")
        print(f"\n[Barrier]")
        print(f"  E_IS    : {energies[0]:.6f} eV")
        print(f"  E_TS    : {energies[ci_idx + 1]:.6f} eV")
        print(f"  E_FS    : {energies[-1]:.6f} eV")
        print(f"  Ea_fwd  : {result['activation_energy']:.4f} eV")
        print(f"  Ea_rev  : {result['reverse_barrier']:.4f} eV")
    print(f"\n  Converged : {result.get('converged','?')}")
    print(f"  Steps     : {result.get('n_steps','?')}")

    _print_output_files(output_dir)
    _print_timing_summary(
        [(f"NEB optimisation ({result.get('n_steps','?')} steps)", t_run)], t_run)
    return 0
