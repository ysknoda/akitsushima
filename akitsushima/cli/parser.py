"""CLI entry point: argument parsing and sub-command dispatch."""
import argparse

from ..fitting import registry
from ..fitting.driver import LOCAL_METHOD_OPTIONS
from .cmd_esm import cmd_esm
from .cmd_fit import cmd_fit
from .cmd_gcmc import cmd_gcmc
from .cmd_kappa import cmd_kappa
from .cmd_md import cmd_md
from .cmd_mlip import cmd_mlip
from .cmd_neb import cmd_neb
from .cmd_osmc import cmd_osmc
from .cmd_phonon import cmd_phonon
from .cmd_relax import cmd_relax
from .cmd_tsa import cmd_tsa
from .templates import _DUMP_MAP

__all__ = ["build_parser", "main"]

# Bug-fix note (F1, consolidation): the original also dispatched a
# separate "fitpot" sub-command here, backed by an implementation that
# never actually worked (see cmd_fit.py's module docstring for why) --
# removed per the project decision to fit every classical potential type
# (LJ/Morse/BornMayer/Buckingham/SW/Tersoff) through this single "fit"
# command instead of two overlapping, confusingly-similar ones.
_DISPATCH = {
    "relax":   cmd_relax,
    "md":      cmd_md,
    "gcmc":    cmd_gcmc,
    "neb":     cmd_neb,
    "tsa":     cmd_tsa,
    "osmc":    cmd_osmc,
    "esm":     cmd_esm,
    "phonon":  cmd_phonon,
    "kappa":   cmd_kappa,
    "mlip":    cmd_mlip,
    "fit":     cmd_fit,
}


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="akitsushima",
        description=(
            "akitsushima  --  classical / ML force-field simulation toolkit\n\n"
            "All sub-commands read a single YAML input file (except 'fit',\n"
            "which reads a training CSV + JSON/YAML fitting config).\n"
            "Generate a YAML template with:  akitsushima --dump-default <task>\n\n"
            "Tasks\n"
            "-----\n"
            "  relax     Geometry optimisation  (FIRE / L-BFGS / SD)\n"
            "  md        Molecular dynamics     (NVT / NVE / NPT)\n"
            "  gcmc      Grand-Canonical MD     (muVT)\n"
            "  neb       Climbing-Image NEB transition-state search\n"
            "  tsa       Transition State Analysis (Hessian / IRC / TST rate)\n"
            "  osmc      CFCMC grand-canonical MC/MD (fractional atom)\n"
            "  esm       Effective Screening Medium\n"
            "  phonon    Phonon band structure, DOS, Debye T, free energy\n"
            "  kappa     Thermal conductivity (Green-Kubo + Slack-Debye)\n"
            "  mlip      Fit a machine-learning potential (ELM/QELM/NN/LightGBM)\n"
            "  fit       Fit a classical potential "
            "(LJ/Morse/BornMayer/Buckingham/SW/Tersoff)\n\n"
            "Examples\n"
            "--------\n"
            "  akitsushima --dump-default nvt    > md_nvt.yaml\n"
            "  akitsushima md     md_nvt.yaml\n"
            "  akitsushima --dump-default relax  > relax.yaml\n"
            "  akitsushima relax  relax.yaml\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    root.add_argument(
        "--dump-default", metavar="TASK",
        help=("Print a default YAML template for TASK and exit.\n"
              "TASK: relax | md | nvt | nve | npt | gcmc | neb | tsa | osmc"
              " | esm | phonon | kappa | mlip"))
    root.add_argument("--version", action="store_true",
                      help="Print version and exit.")

    sub = root.add_subparsers(dest="command", metavar="COMMAND")

    def _add(name, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("yaml", metavar="INPUT.yaml")
        p.add_argument("--quiet", "-q", action="store_true")
        return p

    _add("relax",  "Geometry optimisation (FIRE / L-BFGS / SD)")
    _add("md",     "Molecular dynamics (NVT / NVE / NPT)")
    _add("gcmc",   "Grand-Canonical Monte Carlo MD (muVT)")
    _add("neb",    "Climbing-Image NEB transition-state search")
    _add("tsa",    "Transition State Analysis (Hessian / IRC / TST rate)")
    _add("osmc",   "CFCMC grand-canonical MC/MD (fractional atom)")
    _add("esm",    "Effective Screening Medium")
    _add("phonon", "Phonon band structure, DOS, Debye temperature, free energy")
    _add("kappa",  "Thermal conductivity (Green-Kubo + optional Slack-Debye)")
    _add("mlip",   "Fit a machine-learning potential (ELM/QELM/NN/LightGBM)")

    p_fit = sub.add_parser(
        "fit", help="Fit a classical potential "
                    "(LJ/Morse/BornMayer/Buckingham/SW/Tersoff)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Listed from the table fit_potential itself enforces, so the help
        # cannot drift from what is accepted.
        epilog=("optimisation (config file keys):\n"
                "  use_de: true|false         global Differential Evolution first\n"
                "                             (de_maxiter, de_popsize); --write-config\n"
                "                             templates set true for sw only\n"
                "  local_method: NAME         bounded local refinement, always run:\n"
                "                             " + ", ".join(LOCAL_METHOD_OPTIONS) + "\n"
                "                             (default Nelder-Mead; case-insensitive)\n"
                "  BFGS, CG, Newton-CG, trust-*, dogleg and Powell are refused: they\n"
                "  ignore bounds, need an analytic gradient (fit supplies none), or\n"
                "  (Powell, with bounds) returned worse points than they had found."))
    # Bug-fix note (F3): the original restricted --potential to
    # choices=["buckingham"] at the argparse level, rejecting
    # lj/morse/bornmayer/sw/tersoff before cmd_fit ever ran, even though
    # _build_param_set has always supported all six.
    #
    # Consolidation note (Phase 1, potential registry): choices are now
    # derived from fitting.registry.names() instead of a hardcoded list,
    # so a newly-registered potential (see registry.py's module
    # docstring) automatically appears here with no parser.py edit.
    p_fit.add_argument("--potential",    "-p", default="buckingham",
                        choices=registry.names())
    p_fit.add_argument("--config",       "-c", metavar="FILE")
    p_fit.add_argument("--write-config",       metavar="FILE")
    p_fit.add_argument("--training-csv", "-d", metavar="CSV")
    p_fit.add_argument("--output-dir",   "-o", metavar="DIR")
    p_fit.add_argument("--r-cut",        type=float, metavar="A")
    p_fit.add_argument("--r0",           type=float, metavar="A")
    p_fit.add_argument("--eps0",         type=float, metavar="STRAIN")
    p_fit.add_argument("--use-de",       action="store_true",
                        help="run Differential Evolution before the local "
                             "refinement (same as use_de: true)")
    p_fit.add_argument("--no-forces",    action="store_true")
    p_fit.add_argument("--no-stress",    action="store_true")
    p_fit.add_argument("--quiet", "-q",  action="store_true")

    return root


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    if getattr(args, "version", False):
        print("akitsushima  (development version)")
        return 0

    if getattr(args, "dump_default", None):
        key  = args.dump_default.lower()
        tmpl = _DUMP_MAP.get(key)
        if tmpl is None:
            print(f"ERROR: Unknown task '{key}'. "
                  f"Choose from: {list(_DUMP_MAP)}")
            return 1
        print(tmpl, end="")
        return 0

    if not args.command:
        parser.print_help()
        return 0

    handler = _DISPATCH.get(args.command)
    if handler is None:
        print(f"ERROR: Unknown command '{args.command}'.")
        return 1
    try:
        return handler(args)
    except ValueError as exc:
        # Configuration errors raised past the command's own checks (e.g.
        # an MLIP a solver cannot use, ADR-0007) end as a message, not a
        # traceback.
        print(f"ERROR: {exc}")
        return 1
