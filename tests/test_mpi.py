"""MPI consistency tests: run the same calculation under `mpirun -np 1`,
`-np 2`, and `-np 4`, and confirm the results agree (up to floating-point
summation-order noise from Allreduce, not exact bit-for-bit equality
across different rank counts).

Covers the engine-level `_mpi_domain_split`/`_mpi_allreduce_sum` pattern
used by every pair/many-body potential (SW/Tersoff/EAM here) and by
`fitting/loss.py::generic_loss_function` (see the F16 bug-fix note in
utils/mpi.py for why this specific combination -- an engine potential's
OWN internal MPI split, invoked from a function that does no outer-level
splitting of its own -- needed dedicated verification: the *reverse*
combination, outer-level splitting invoking these same potentials, is
exactly the scenario F16 found broken and fixed via `local_only()`).

This module is skipped entirely if `mpirun` is not on PATH or mpi4py is
not installed, since MPI itself is an optional dependency of this
project (single-process operation is fully supported without it).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import mpi4py  # noqa: F401
    _HAS_MPI4PY = True
except ImportError:
    _HAS_MPI4PY = False

_MPIRUN = shutil.which("mpirun")
_WORKER = Path(__file__).parent / "_mpi_consistency_worker.py"

pytestmark = pytest.mark.skipif(
    _MPIRUN is None or not _HAS_MPI4PY,
    reason="mpirun and/or mpi4py not available in this environment")


def _run_worker(np_ranks: int, out_path: Path) -> None:
    cmd = [_MPIRUN, "-np", str(np_ranks), sys.executable,
           str(_WORKER), str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"mpirun -np {np_ranks} failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}")
    assert out_path.is_file(), (
        f"mpirun -np {np_ranks} did not produce {out_path} "
        f"(rank-0 guard may be broken)")


@pytest.fixture(scope="module")
def mpi_results(tmp_path_factory):
    """Run the worker script under -np 1/2/4 once, shared across every
    assertion test below (the worker itself takes a few seconds per
    invocation, and there is no reason to re-run it per assertion)."""
    tmp_dir = tmp_path_factory.mktemp("mpi_consistency")
    results = {}
    for n in (1, 2, 4):
        out_path = tmp_dir / f"np{n}.npz"
        _run_worker(n, out_path)
        results[n] = dict(np.load(out_path))
    return results


def _assert_all_close(results: dict, key: str, **kwargs):
    ref = results[1][key]
    for n in (2, 4):
        assert np.allclose(results[n][key], ref, **kwargs), (
            f"{key} differs between np=1 and np={n}: "
            f"np1={ref}, np{n}={results[n][key]}")


def test_sw_energy_forces_stress_consistent_across_ranks(mpi_results):
    _assert_all_close(mpi_results, "e_sw", rtol=1e-8, atol=1e-10)
    _assert_all_close(mpi_results, "f_sw", rtol=1e-8, atol=1e-10)
    _assert_all_close(mpi_results, "s_sw", rtol=1e-8, atol=1e-10)


def test_tersoff_energy_forces_stress_consistent_across_ranks(mpi_results):
    _assert_all_close(mpi_results, "e_te", rtol=1e-8, atol=1e-10)
    _assert_all_close(mpi_results, "f_te", rtol=1e-8, atol=1e-10)
    _assert_all_close(mpi_results, "s_te", rtol=1e-8, atol=1e-10)


def test_eam_energy_forces_stress_consistent_across_ranks(mpi_results):
    _assert_all_close(mpi_results, "e_eam", rtol=1e-8, atol=1e-10)
    _assert_all_close(mpi_results, "f_eam", rtol=1e-8, atol=1e-10)
    _assert_all_close(mpi_results, "s_eam", rtol=1e-8, atol=1e-10)


def test_fitting_loss_consistent_across_ranks(mpi_results):
    _assert_all_close(mpi_results, "loss", rtol=1e-8, atol=1e-10)


def test_fitting_loss_with_sw_consistent_across_ranks(mpi_results):
    """Regression test for the local_only()-bypass bug found in
    code-check: SW's `compute()` used a raw `_mpi_comm.allreduce()` for
    its scalar energy (both the legacy and the multi-species path), which
    ignored `local_only()` suppression -- unlike `_mpi_allreduce_sum()`,
    already correctly used for forces/stress. Because
    `generic_loss_function` wraps its own per-structure loop in `with
    local_only():` (see fitting/loss.py), fitting a many-body potential
    such as SW under real multi-rank `mpirun` used to silently return a
    wrong loss value (LJ, a `pair_potentials.py`-based potential, was
    already correct and unaffected -- see `test_fitting_loss_consistent
    _across_ranks` above, which is why this combination was never
    previously caught)."""
    _assert_all_close(mpi_results, "loss_sw", rtol=1e-8, atol=1e-10)


def test_sw_tersoff_eam_energy_respects_local_only_under_mpi(mpi_results):
    """Direct, isolated regression test for the same local_only()-bypass
    bug (see test_fitting_loss_with_sw_consistent_across_ranks above):
    wrapped in `with local_only():`, `_mpi_domain_split` returns the FULL
    range on every rank (no partial-sum splitting happens inside
    `compute()`), so each rank independently computes the complete,
    correct energy for the SAME fixed structure -- the true value must
    therefore come out EXACTLY identical regardless of -np, not just
    close. Before the fix, the leftover raw `_mpi_comm.allreduce()` call
    still summed every rank's (identical) energy together despite the
    suppression, inflating the reported energy by exactly the rank count
    (2x under -np 2, 4x under -np 4) -- a much larger, unambiguous
    discrepancy than ordinary floating-point summation-order noise, so a
    tight tolerance is deliberately used here to make sure a regression
    cannot hide behind it."""
    _assert_all_close(mpi_results, "e_sw_local", rtol=1e-10, atol=1e-12)
    _assert_all_close(mpi_results, "e_te_local", rtol=1e-10, atol=1e-12)
    _assert_all_close(mpi_results, "e_eam_local", rtol=1e-10, atol=1e-12)
    _assert_all_close(mpi_results, "e_sw_multi_local", rtol=1e-10, atol=1e-12)
