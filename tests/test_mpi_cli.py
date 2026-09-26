"""The MPI-aware CLI commands must give the same answer on 1, 2 and 4 ranks.

ADR-0003, T1.  ``relax``, ``md`` (NVT and NPT), ``gcmc`` and ``osmc``
(CFCMC, ADR-0009) are each run end-to-end under ``mpirun -np 1/2/4`` through ``python -m
akitsushima``, from their shipped templates with only the step counts
shrunk, and the files they write are compared.

Two tolerances, because two different things are being compared:

* Everything a log prints (energies, temperatures, N per cycle) must agree
  to 1e-5 absolute, and every non-numeric token exactly.  Summation-order
  noise can flip the last printed digit (seen: 1.879160 vs 1.879159 eV in
  NVT), so exact text equality is too strict; a rank that double-counts or
  drops its share of a sum changes these in the leading digits.
* Structure files carry ~15 significant digits, where the summation order
  of the Allreduce does show.  After 50 NPT steps the barostat has
  amplified that noise to ~1e-8 A in positions (measured), so structures
  are compared at 1e-6 A: far above the noise, far below any real error.

Skipped when mpirun or mpi4py is unavailable, like tests/test_mpi.py.
"""

import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml
from test_cli_templates import _write_ar_gas, _write_si

from akitsushima.cli.templates import _DUMP_MAP
from akitsushima.io.poscar import parse_poscar

try:
    import mpi4py  # noqa: F401

    _HAS_MPI4PY = True
except ImportError:
    _HAS_MPI4PY = False

_MPIRUN = shutil.which("mpirun")

pytestmark = pytest.mark.skipif(
    _MPIRUN is None or not _HAS_MPI4PY,
    reason="mpirun and/or mpi4py not available in this environment",
)

_RANKS = (1, 2, 4)

# case -> (template key, command, section, step overrides, POSCAR builder)
_CASES = {
    "relax": ("relax", "relax", "relax", {"max_steps": 30}, "si"),
    "nvt": (
        "nvt",
        "md",
        "md",
        {
            "n_steps": 50,
            "equil_steps": 10,
            "velocity_rescale_steps": 10,
            "thermo_every": 5,
        },
        "si",
    ),
    "npt": (
        "npt",
        "md",
        "md",
        {
            "n_steps": 50,
            "equil_steps": 10,
            "velocity_rescale_steps": 10,
            "thermo_every": 5,
        },
        "si",
    ),
    "gcmc": ("gcmc", "gcmc", "gcmc", {"n_cycles": 10}, "ar"),
    "osmc": ("osmc", "osmc", "osmc", {"n_cycles": 10, "equil_cycles": 2}, "ar"),
}

# case -> (log files compared to 1e-5, structure files compared at 1e-6 A)
_FILES = {
    "relax": (["relax.log"], ["POSCAR_relaxed"]),
    "nvt": (["thermo.log"], ["POSCAR_final_MD"]),
    "npt": (["thermo.log"], ["POSCAR_final_NPT"]),
    "gcmc": (["N_vs_cycle.dat", "gcmc_md_thermo.log"], ["POSCAR_gcmc_final"]),
    "osmc": (["N_vs_cycle.dat", "lambda_vs_cycle.dat"], ["POSCAR_gcmc_final"]),
}


@pytest.fixture(scope="module")
def mpi_runs(tmp_path_factory):
    """Run every case on every rank count once; return {(case, np): dir}."""
    root = tmp_path_factory.mktemp("mpi_cli")
    _write_si(root / "POSCAR_si", jitter=0.05)
    _write_ar_gas(root / "POSCAR_ar")
    out = {}
    for case, (key, command, section, shrink, geo) in _CASES.items():
        for n in _RANKS:
            cfg = yaml.safe_load(_DUMP_MAP[key])
            cfg["structure"]["poscar"] = str(root / f"POSCAR_{geo}")
            cfg[section].update(shrink)
            cfg[section]["verbose"] = False
            out_dir = root / f"{case}_np{n}"
            cfg[section]["output_dir"] = str(out_dir)
            yaml_path = root / f"{case}_np{n}.yaml"
            yaml_path.write_text(yaml.safe_dump(cfg))
            res = subprocess.run(
                [
                    _MPIRUN,
                    "-np",
                    str(n),
                    sys.executable,
                    "-m",
                    "akitsushima",
                    command,
                    str(yaml_path),
                    "-q",
                ],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=root,
            )
            assert res.returncode == 0, (
                f"{case} on {n} ranks failed:\n{res.stdout}\n{res.stderr}"
            )
            out[(case, n)] = out_dir
    return out


def _tokens(path):
    return [line.split() for line in path.read_text().splitlines()]


def _as_float(tok):
    try:
        return float(tok)
    except ValueError:
        return None


@pytest.mark.parametrize("case", sorted(_CASES))
def test_logs_agree_across_ranks(mpi_runs, case):
    for name in _FILES[case][0]:
        ref = _tokens(mpi_runs[(case, 1)] / name)
        for n in _RANKS[1:]:
            got = _tokens(mpi_runs[(case, n)] / name)
            assert len(got) == len(ref), f"{case}: {name} line count, {n} ranks"
            for i, (lr, lg) in enumerate(zip(ref, got)):
                assert len(lr) == len(lg), f"{case}: {name} line {i}, {n} ranks"
                for tr, tg in zip(lr, lg):
                    fr, fg = _as_float(tr), _as_float(tg)
                    if fr is None:
                        assert tg == tr, f"{case}: {name} line {i}, {n} ranks"
                    else:
                        assert fg is not None and abs(fg - fr) <= 1e-5, (
                            f"{case}: {name} line {i}: {tr} (1 rank) vs "
                            f"{tg} ({n} ranks)")


@pytest.mark.parametrize("case", sorted(_CASES))
def test_structures_agree_across_ranks(mpi_runs, case):
    for name in _FILES[case][1]:
        ref = parse_poscar(str(mpi_runs[(case, 1)] / name))
        for n in _RANKS[1:]:
            got = parse_poscar(str(mpi_runs[(case, n)] / name))
            assert got["atom_types"] == ref["atom_types"], (
                f"{case}: {name} atom list differs on {n} ranks"
            )
            np.testing.assert_allclose(
                got["lattice"], ref["lattice"], rtol=0, atol=1e-6
            )
            np.testing.assert_allclose(
                got["positions"], ref["positions"], rtol=0, atol=1e-6
            )


@pytest.mark.parametrize("case", sorted(_CASES))
def test_outputs_written_once_by_rank_zero(mpi_runs, case):
    """Every rank writing the same log would interleave or duplicate its
    lines; the line count must not scale with the rank count."""
    for name in _FILES[case][0]:
        ref = len((mpi_runs[(case, 1)] / name).read_text().splitlines())
        for n in _RANKS[1:]:
            got = len((mpi_runs[(case, n)] / name).read_text().splitlines())
            assert got == ref
