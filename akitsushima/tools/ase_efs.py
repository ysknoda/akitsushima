"""ASE -> EFS file for `potential: {type: external}` (ADR-0011).

Two modes:
  --calc emt | lj   evaluate an ASE calculator on POSCAR (a stand-in for an
                    external code; EMT has analytic forces and stress)
  --read FILE       convert any ASE-readable output that carries energy,
                    forces and stress (e.g. vasprun.xml, OUTCAR,
                    espresso.pwo); the atom order must be POSCAR's
                    (UNVERIFIED on real VASP / QE output: no such files
                    were available when this was written)

Conventions (checked against finite differences, tests/test_external_engine.py):
ASE's stress is (1/V) dE/d(strain), positive under tension; akitsushima's
is positive under compression (ADR-0001), so the sign is flipped here.
Energy: the force-consistent (free) energy when the source provides it,
since the forces are its derivatives.

Usage:  python -m akitsushima.tools.ase_efs --calc emt [POSCAR] [OUT.efs]
        python -m akitsushima.tools.ase_efs --read vasprun.xml [POSCAR] [OUT.efs]
"""
import argparse
import sys

import numpy as np


def _energy(atoms) -> float:
    try:
        return float(atoms.get_potential_energy(force_consistent=True))
    except Exception:  # calculators without a free energy
        return float(atoms.get_potential_energy())


def ase_efs(atoms):
    """(E [eV], F [eV/A], S [eV/A^3], akitsushima sign) of an ASE Atoms
    object with a calculator (or read results) attached."""
    E = _energy(atoms)
    F = np.asarray(atoms.get_forces(), dtype=float)
    S = -np.asarray(atoms.get_stress(voigt=False), dtype=float)
    return E, F, S


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--calc", choices=["emt", "lj"])
    g.add_argument("--read", metavar="FILE")
    ap.add_argument("--lj-epsilon", type=float, default=0.0103)
    ap.add_argument("--lj-sigma", type=float, default=3.405)
    ap.add_argument("--lj-rc", type=float, default=8.5125)
    ap.add_argument("poscar", nargs="?", default="POSCAR")
    ap.add_argument("output", nargs="?", default="result.efs")
    args = ap.parse_args(argv)

    import ase.io

    from ..io.efs import write_efs_file

    if args.read:
        atoms = ase.io.read(args.read)
        ref = ase.io.read(args.poscar, format="vasp")
        if atoms.get_chemical_symbols() != ref.get_chemical_symbols():
            raise SystemExit(f"ase_efs: {args.read} lists the atoms in another "
                             f"order than {args.poscar}.")
    else:
        atoms = ase.io.read(args.poscar, format="vasp")
        if args.calc == "emt":
            from ase.calculators.emt import EMT
            atoms.calc = EMT()
        else:
            from ase.calculators.lj import LennardJones
            atoms.calc = LennardJones(epsilon=args.lj_epsilon,
                                      sigma=args.lj_sigma, rc=args.lj_rc,
                                      smooth=False)
    E, F, S = ase_efs(atoms)
    write_efs_file(args.output, E, F, S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
