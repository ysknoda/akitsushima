"""``akitsushima mdspec`` -- spectra from MD trajectories: the complex
permittivity eps(omega) by the fluctuation-dissipation theorem, optics,
mode-resolved spectra, fits and run diagnostics (ADR-0050, ADR-0051;
ported from nozuchi 0.2.0a2).

    akitsushima mdspec input.yaml                    # the analysis
    akitsushima mdspec input.yaml --check            # run diagnostics
    akitsushima mdspec nozuchi.yaml --convert-nozuchi > input.yaml
"""

from pathlib import Path

from ..utils.mpi import _mpi_rank

__all__ = ["cmd_mdspec"]


def cmd_mdspec(args) -> int:
    """mdspec from YAML input (rank 0 only: a serial post-processing step)."""
    if _mpi_rank != 0:
        return 0
    from .. import __version__
    from ..mdspec import run, yaml_io

    if getattr(args, "convert_nozuchi", False):
        print(yaml_io.convert_file(args.yaml), end="")
        return 0
    cfg = yaml_io.load(args.yaml)
    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    quiet = getattr(args, "quiet", False)
    if getattr(args, "check", False):
        from ..mdspec.check import check
        from ..mdspec.output import write_json

        print(f"akitsushima {__version__} mdspec check: {Path(args.yaml).resolve()}")
        rep = check(
            cfg,
            energy_files=getattr(args, "energy", None),
            target=getattr(args, "target", 0.02),
        )
        write_json(out / f"{cfg['output']['prefix']}_check.json", rep)
        return 1 if rep["n_warn"] else 0
    log_path = out / f"{cfg['output']['prefix']}.log"
    with open(log_path, "w") as fh:

        def log(msg: str) -> None:
            if not quiet:
                print(msg, flush=True)
            fh.write(msg + "\n")

        log(f"akitsushima {__version__} mdspec: {Path(args.yaml).resolve()}")
        try:
            run(cfg, log=log)
        except Exception as exc:
            fh.write(f"[error] {type(exc).__name__}: {' '.join(str(exc).split())}\n")
            raise
    return 0
