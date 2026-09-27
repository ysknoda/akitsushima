"""akitsushima as an external code: POSCAR -> E, F, S of a YAML potential
-> EFS file.  Used to test `potential: {type: external}` against the same
potential evaluated in process (ADR-0011), and as a template for wrappers.

Usage:  python -m akitsushima.tools.akitsushima_efs POT.yaml [POSCAR] [OUT.efs]

POT.yaml holds a `potential:` section (or the section's keys at the top
level), exactly as in any akitsushima config.
"""
import argparse
import sys

import yaml


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("potential_yaml", help="YAML with a potential: section")
    ap.add_argument("poscar", nargs="?", default="POSCAR")
    ap.add_argument("output", nargs="?", default="result.efs")
    args = ap.parse_args(argv)

    from ..io.efs import write_efs_file
    from ..io.poscar import parse_poscar
    from ..solvers.config_bridge import _build_force_fn_from_yaml

    with open(args.potential_yaml) as fh:
        cfg = yaml.safe_load(fh)
    pot = cfg.get("potential", cfg)
    if str(pot.get("type", "")).lower() == "external":
        raise SystemExit("akitsushima_efs: the potential must not itself be "
                         "external (that would recurse).")
    fn = _build_force_fn_from_yaml(pot)[0]
    st = parse_poscar(args.poscar)
    E, F, S = fn(st["lattice"], st["positions"], st["atom_types"])
    write_efs_file(args.output, float(E), F, S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
