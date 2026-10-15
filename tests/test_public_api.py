"""Phase 7 acceptance test: the top-level `akitsushima` package's public
API, including the exact Quick Start example from its own module
docstring, must work -- and NNModel must stay lazy (F15/E15-adjacent:
importing akitsushima alone, or using any of its non-NN classes, must
never pull in torch).
"""
import os
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest


def test_package_version():
    import akitsushima
    assert akitsushima.__version__


# --------------------------------------------------------------------------
# Quick start example: executed straight out of akitsushima.__doc__.
#
# This test deliberately keeps NO copy of the example's parameters.  A
# previous version did, and the copy drifted: the docstring advertised
# r_cut = 10.0 A while the test quietly substituted r_cut = 5.0 A, so the
# documented example was unrunnable for two years' worth of readers while
# the acceptance test stayed green.  Reading the source from the live
# docstring makes that class of drift impossible rather than merely
# discouraged.
# --------------------------------------------------------------------------

_QUICK_START_MARKER = "# API example:"

# The extracted block must contain these.  Without this guard, a parsing
# failure that returned an empty string would make exec("") succeed and this
# test pass vacuously -- reintroducing the exact silent-success failure mode
# the test exists to prevent.
_QUICK_START_REQUIRED_TOKENS = (
    "import numpy as np",
    "PotentialEngine",
    "LennardJonesPotential",
    "NeighbourList",
    "make_force_fn",
    "nl.build(",
    "r_cut=",
)


def _extract_quick_start_api_example(doc):
    """Return the runnable Python block of the docstring's Quick start section.

    The block begins at the ``# API example:`` marker line and continues
    while lines are blank or indented at least as far as that marker, so it
    ends naturally at the next unindented section heading or at the end of
    the docstring.  The shell/CLI lines of the same section are excluded
    because they sit above the marker.
    """
    if not doc:
        raise AssertionError("akitsushima.__doc__ is empty or None")

    lines = doc.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if _QUICK_START_MARKER in line), None)
    if start is None:
        raise AssertionError(
            f"marker {_QUICK_START_MARKER!r} not found in akitsushima.__doc__ "
            "-- the Quick start example can no longer be located, so it can no "
            "longer be verified")

    marker_indent = len(lines[start]) - len(lines[start].lstrip())
    block = [lines[start]]
    for line in lines[start + 1:]:
        if not line.strip():
            block.append(line)
            continue
        if len(line) - len(line.lstrip()) < marker_indent:
            break
        block.append(line)

    source = textwrap.dedent("\n".join(block))
    missing = [tok for tok in _QUICK_START_REQUIRED_TOKENS if tok not in source]
    if missing:
        raise AssertionError(
            f"Quick start block extracted from akitsushima.__doc__ is missing "
            f"{missing} -- extraction is broken, or the documented example was "
            f"gutted.\n--- extracted ---\n{source}")
    return source


def _run_quick_start_api_example(doc):
    """Execute the extracted Quick start block, returning its namespace.

    ``exec`` is safe here: the input is this package's own module docstring,
    shipped in the same source tree as this test, not user-supplied text.
    Executing it rather than duplicating it is precisely what couples the
    test to the documentation.
    """
    source = _extract_quick_start_api_example(doc)
    namespace = {}
    exec(compile(source, "<akitsushima.__doc__ Quick start>", "exec"), namespace)
    return namespace


def test_quick_start_docstring_example_works():
    """Run the Quick start API example extracted from ``akitsushima.__doc__``
    at run time; this test holds no copy of its parameters.

    The documented example must therefore be genuinely self-contained -- it
    defines lj_params, lattice, positions and atom_types itself -- and must
    execute exactly as a reader would paste it.

    Beyond "it runs", this checks the documented values stay physically
    defensible: the cutoff must remain a conventional LJ-argon cutoff, and
    the documented cell must genuinely satisfy NeighbourList's minimum-image
    precondition rather than the cutoff having been shrunk to fit a box that
    is too small.
    """
    import akitsushima

    namespace = _run_quick_start_api_example(akitsushima.__doc__)

    energy = namespace["energy"]
    forces = np.asarray(namespace["forces"])
    stress = np.asarray(namespace["stress"])
    lattice = np.asarray(namespace["lattice"])
    positions = np.asarray(namespace["positions"])
    atom_types = namespace["atom_types"]
    neighbour_list = namespace["nl"]

    assert np.isfinite(energy)
    assert np.all(np.isfinite(forces))
    assert np.all(np.isfinite(stress))
    assert forces.shape == (len(atom_types), 3)
    assert stress.shape == (3, 3)

    # Newton's third law: the two-atom system's forces must cancel exactly.
    assert np.allclose(forces[0], -forces[1])
    # Translational invariance: the net force must vanish.
    assert np.allclose(forces.sum(axis=0), 0.0, atol=1e-12)

    # The documented cell must actually satisfy the minimum-image convention
    # NeighbourList.build() enforces, with the documented r_cut and the
    # default skin width -- not merely happen to pass today.  The documented
    # cell is orthogonal, so the shortest lattice-vector length and the
    # shortest perpendicular cell width coincide.
    shortest_cell_vector = float(np.min(np.linalg.norm(lattice, axis=1)))
    assert shortest_cell_vector > 2.0 * neighbour_list.r_list, (
        f"documented cell (shortest lattice vector {shortest_cell_vector} A) "
        f"violates edge > 2*(r_cut + skin_width) = "
        f"{2.0 * neighbour_list.r_list} A")

    # The documented cutoff must stay a physically sensible LJ-argon cutoff.
    # This is the guard against "fixing" a future minimum-image ValueError by
    # shrinking r_cut to fit a small box: conventional LJ argon is truncated
    # at 2.5-3.5 sigma (Verlet, Phys. Rev. 159, 98 (1967), uses 2.5 sigma).
    sigma = namespace["lj_params"][frozenset({"Ar"})]["sigma"]
    assert 2.5 <= neighbour_list.r_cut / sigma <= 3.5, (
        f"documented r_cut = {neighbour_list.r_cut} A is "
        f"{neighbour_list.r_cut / sigma:.2f} sigma, outside the conventional "
        "2.5-3.5 sigma truncation range for Lennard-Jones argon")

    # The documented separation sits beyond the LJ minimum r_min = 2^(1/6)
    # sigma, so the pair must be bound and mutually attractive.
    separation_vector = positions[1] - positions[0]
    separation = float(np.linalg.norm(separation_vector))
    assert separation > 2.0 ** (1.0 / 6.0) * sigma
    assert energy < 0.0, "atoms beyond the LJ minimum must be bound"
    assert float(forces[0] @ separation_vector) > 0.0, (
        "attraction must pull atom 0 towards atom 1")


@pytest.mark.parametrize("name", [
    "PotentialEngine", "NeighbourList", "BasePairPotential",
    "LennardJonesPotential", "MorsePotential", "BornMayerPotential",
    "BuckinghamPotential", "StillingerWeberPotential", "TersoffPotential",
    "EAMPotential", "ewald_sum", "PPPMSolver", "ESMCalculator",
    "compute_all_general", "make_force_fn", "compute_all", "fit_potential",
    "save_potential", "load_potential", "FittableParameterSet",
    "BuckinghamParameterSet", "LJParameterSet", "MorseParameterSet",
    "BornMayerParameterSet", "StillingerWeberParameterSet",
    "TersoffParameterSet", "ExternalFileEngine", "ScaledForceEngine",
    "ELMModel", "QuantizedELMModel", "GeometryOptimiser", "NVTSimulation",
    "NVESimulation", "NPTSimulation", "GCMCMDSimulation", "CFCMCSimulation",
    "NEBChain", "NEBOptimiser", "TransitionStateAnalyser",
    "PhononCalculator", "ThermalConductivityCalculator", "parse_poscar",
    "write_poscar", "ATOMIC_MASS", "K_B", "K_E", "EV_ANG_TO_AMU_ANG_FS2",
])
def test_top_level_symbol_is_exported(name):
    import akitsushima
    assert hasattr(akitsushima, name)
    assert name in akitsushima.__all__


def test_nn_is_lazy_at_top_level(tmp_path):
    """F15 regression, verified at the top-level package too: importing
    akitsushima, or touching any of its other classes, must not import
    torch; only explicitly accessing NNModel may.

    Run in a genuinely fresh subprocess rather than in-process: within
    the shared pytest session, other test modules (test_fitting.py's own
    NNModel tests) legitimately import torch for
    their own purposes, and Python's sys.modules cache is process-wide,
    not per-test -- so an in-process "assert torch not in sys.modules"
    here would depend on test execution order rather than on whether
    *this* package's lazy-loading is actually correct (confirmed by this
    exact test failing when run after test_fitting.py in a full `pytest
    tests/` run, despite passing in isolation).

    The subprocess is run with cwd=tmp_path (a directory that cannot
    contain the original akitsushima.py), not the repository root:
    pytest's own cwd IS the repository root, which subprocess.run
    inherits by default, and Python resolves a bare `import akitsushima`
    against the CURRENT directory before site-packages -- so running
    this from the repo root would silently import the OLD, pre-refactor
    akitsushima.py file instead of the new package (that old file
    imports torch/lightgbm/optuna unconditionally at module load time,
    which is one of the very things this port fixed, so testing against
    it would defeat the point of this test rather than exercise it).
    """
    # The second half -- that accessing NNModel DOES import torch -- can
    # only hold where torch is installed.  It is an optional extra, and in
    # a clean `pip install akitsushima[dev,yaml]` environment this test
    # used to fail on exactly that line while the property it exists for
    # (nothing imported eagerly) passed.
    import importlib.util
    has_torch = importlib.util.find_spec("torch") is not None
    script = (
        "import sys\n"
        "import akitsushima\n"
        "assert 'torch' not in sys.modules, 'torch imported too eagerly'\n"
        "_ = akitsushima.NNModel\n"
        + ("assert 'torch' in sys.modules, 'NNModel access did not import torch'\n"
           if has_torch else "")
        + "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=60,
                           cwd=str(tmp_path))
    assert result.returncode == 0, (
        f"stdout: {result.stdout}\nstderr: {result.stderr}")
    assert "OK" in result.stdout


def test_cli_main_and_build_parser_importable():
    from akitsushima.cli import build_parser, main
    parser = build_parser()
    ns = parser.parse_args(["--version"])
    assert ns.version is True
    assert callable(main)


# --------------------------------------------------------------------------
# README Python API example (ADR-0023, decision 5).  Like the docstring
# example above, it is executed from the file itself, so the README cannot
# drift again: before 2026-10-15 it used r_cut = 5.0 A (1.47 sigma) and two
# atoms 2.5 A apart, and built a neighbour list it never passed.
# --------------------------------------------------------------------------

_README = os.path.join(os.path.dirname(__file__), os.pardir, "README.md")


def _readme_python_api_block():
    with open(_README) as fh:
        text = fh.read()
    match = re.search(r"### Python API\n\n```python\n(.*?)```", text, re.S)
    assert match, "README.md has no '### Python API' python code block"
    return match.group(1)


@pytest.mark.skipif(not os.path.isfile(_README),
                    reason="README.md not shipped (installed package)")
def test_readme_python_api_example_matches_docstring():
    import akitsushima

    readme = {}
    exec(compile(_readme_python_api_block(), "<README.md Python API>", "exec"),
         readme)
    doc = _run_quick_start_api_example(akitsushima.__doc__)

    # Same physics as the docstring example, which carries the guards on
    # r_cut and separation; agreement here carries them over.
    assert readme["energy"] == pytest.approx(doc["energy"], rel=1e-12)
    np.testing.assert_allclose(readme["forces"], doc["forces"],
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(readme["stress"], doc["stress"],
                               rtol=1e-12, atol=1e-15)
    assert readme["energy"] < 0.0


def test_cli_version_prints_the_package_version(monkeypatch, capsys):
    """ADR-0023: `akitsushima --version` printed "(development version)"."""
    import akitsushima
    from akitsushima.cli import main
    monkeypatch.setattr(sys, "argv", ["akitsushima", "--version"])
    assert main() == 0
    assert capsys.readouterr().out.strip() == f"akitsushima {akitsushima.__version__}"


@pytest.mark.parametrize("module, build", [
    ("akitsushima.fitting.models.nn", lambda m: m.NNModel()),
    ("akitsushima.fitting.models.nn", lambda m: m.NNModel.load("x.pt")),
    ("akitsushima.fitting.models.atomic_nn", lambda m: m.AtomicNNModel()),
])
def test_missing_torch_message_names_the_extra(monkeypatch, module, build):
    """ADR-0023: without torch the error advised
    `pip install torch --break-system-packages`, wrong inside a venv."""
    import importlib
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, "HAS_TORCH", False)
    with pytest.raises(ImportError) as exc:
        build(mod)
    msg = str(exc.value)
    assert 'pip install "akitsushima[torch]"' in msg
    assert "break-system-packages" not in msg


@pytest.mark.skipif(not os.path.isfile(_README),
                    reason="README.md not shipped (installed package)")
def test_readme_mlip_example_runs(tmp_path, monkeypatch):
    """ADR-0024 Q25: the README's machine-learning-potential example, run as
    written.  Its relaxed energy must sit near the minimum of the
    Stillinger-Weber silicon it was trained on."""
    with open(_README) as fh:
        text = fh.read()
    match = re.search(r"### Python API: a machine-learning potential\n.*?```python\n(.*?)```",
                      text, re.S)
    assert match, "README.md has no MLIP example block"
    monkeypatch.chdir(tmp_path)
    ns = {}
    exec(compile(match.group(1), "<README.md MLIP example>", "exec"), ns)
    import akitsushima as ak
    a = 5.431   # SW Si equilibrium lattice constant [A] (Stillinger & Weber 1985)
    diamond = ns["frac"] @ (np.eye(3) * 2 * a)
    e_min = ns["sw"](np.eye(3) * 2 * a, diamond, ["Si"] * 64)[0] / 64
    assert abs(ns["result"]["energy"] / 64 - e_min) < 0.02
    assert (tmp_path / "si_model.npz").is_file()
    assert isinstance(ns["model"], ak.AtomicELMModel)


def test_import_without_matplotlib_does_not_warn(tmp_path):
    """ADR-0024 O1: without matplotlib every `import akitsushima` warned,
    plot or no plot; now only a requested plot does."""
    script = (
        "import sys, warnings\n"
        "sys.modules['matplotlib'] = None\n"      # import fails like an absent package
        "warnings.simplefilter('error')\n"
        "import akitsushima\n"
        "from akitsushima.utils.optional_matplotlib import have_matplotlib\n"
        "warnings.simplefilter('always')\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    assert have_matplotlib() is False and have_matplotlib() is False\n"
        "assert len(w) == 1, w\n"
        "print('OK')\n")
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                       timeout=120, cwd=str(tmp_path))
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr
