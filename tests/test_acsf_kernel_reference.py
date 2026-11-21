"""The ACSF kernel against its 0.2.0a19 form (ADR-0030).

ADR-0030 reorganised the angular loop of ``_acsf_kernel``.  Shared
exponentials, integer powers and per-neighbour cutoff factors change the
rounding but not the value.  ``tests/acsf_reference_kernel.py`` keeps the
former kernel unchanged.  Here both are run on the same structures and
grids, and G, dGdv and the E, F, S of a fitted atom-centred ELM must agree
to relative 1e-10, the default-path rule of ADR-0028.
"""

import acsf_reference_kernel as ref
import numpy as np
import pytest

from akitsushima.fitting import AtomicELMModel
from akitsushima.fitting.descriptors import ACSF
from akitsushima.fitting.descriptors import acsf as acsf_mod
from akitsushima.fitting.mlip_engine import MLIPForceField

RTOL = 1e-10
DIAMOND = np.array(
    [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ]
)
SHEAR = np.array([[1.0, 0.0, 0.0], [0.21, 0.97, 0.0], [0.12, -0.09, 1.03]])

# r_c 5 A as in ac_elm_iso_r2 (ADR-0010 A1)
G2 = {"eta": [12.5], "rs": [2.0 + 0.2 * k for k in range(15)]}
G5_ISO_R2 = {"eta": [0.0, 0.05], "zeta": [1, 2, 4], "lambda": [-1, 1]}
G4 = {"eta": [0.005, 0.05], "zeta": [1, 2, 16], "lambda": [-1, 1]}
NON_INTEGER = {"eta": [0.01], "zeta": [1.5, 3.7], "lambda": [-1, 1]}

GRIDS = {
    "g5_iso_r2": dict(g2=G2, g5=G5_ISO_R2),
    "g4": dict(g2=G2, g4=G4),
    "g4_g5_non_integer_zeta": dict(g2=G2, g4=NON_INTEGER, g5=NON_INTEGER),
    "g5_inner_cutoff": dict(g2=G2, g5=G5_ISO_R2, inner_cutoff=(1.9, 2.5)),
}


def _cell(name, triclinic, seed=0):
    """64-atom diamond cell, rattled; Si, or Ge/Sn at random (GeSn)."""
    rng = np.random.default_rng(seed)
    a = 5.431 if name == "Si" else 5.9
    cells = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
    frac = ((DIAMOND[None] + cells[:, None]).reshape(-1, 3)) / 2.0
    lat = np.eye(3) * 2 * a
    if triclinic:
        lat = lat @ SHEAR
    pos = frac @ lat + rng.normal(0.0, 0.08, frac.shape)
    if name == "Si":
        types = ["Si"] * len(pos)
    else:
        types = list(rng.choice(["Ge", "Sn"], len(pos)))
    return lat, pos, types


def _species(name):
    return ["Si"] if name == "Si" else ["Ge", "Sn"]


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-300)


@pytest.fixture
def with_reference(monkeypatch):
    """Run ``fn`` with the 0.2.0a19 kernel in place of the package one;
    the descriptors ``descs`` are switched to the pair loop meanwhile,
    since the reference computes every angular term by pairs (ADR-0038)."""

    def run(fn, *descs):
        with monkeypatch.context() as m:
            m.setattr(acsf_mod, "_acsf_kernel", ref._acsf_kernel)
            for d in descs:
                m.setattr(d, "angular_kernel", "pairs")
            return fn()

    return run


@pytest.mark.parametrize("kernel", ["auto", "pairs"])
@pytest.mark.parametrize("triclinic", [False, True], ids=["cubic", "triclinic"])
@pytest.mark.parametrize("grid", sorted(GRIDS))
@pytest.mark.parametrize("name", ["Si", "GeSn"])
@pytest.mark.slow
def test_g_and_dgdv_match_the_reference(with_reference, name, grid, triclinic, kernel):
    """``auto`` puts the G5 terms with integer zeta on the moment tensors
    of ADR-0038; ``pairs`` is the pair loop of ADR-0030."""
    d = ACSF.from_grid(_species(name), 5.0, **GRIDS[grid], angular_kernel=kernel)
    lat, pos, types = _cell(name, triclinic)
    new = d._run(lat, pos, types, True)
    old = with_reference(lambda: d._run(lat, pos, types, True), d)
    G, dGdv = new[4], new[5]
    assert np.all(np.isfinite(G)) and np.abs(G).max() > 0
    assert _rel(G, old[4]) < RTOL, f"G: {_rel(G, old[4]):.2e}"
    assert _rel(dGdv, old[5]) < RTOL, f"dGdv: {_rel(dGdv, old[5]):.2e}"
    # the descriptor-only path (no derivatives) gives the same G
    G_only = d._run(lat, pos, types, False)[4]
    assert _rel(G_only, old[4]) < RTOL


@pytest.mark.parametrize("grid", ["g5_iso_r2", "g5_inner_cutoff"])
@pytest.mark.slow
def test_elm_energy_forces_stress_match_the_reference(with_reference, grid):
    d = ACSF.from_grid(["Ge", "Sn"], 5.0, **GRIDS[grid])
    rng = np.random.default_rng(1)
    structs = []
    for k in range(4):
        lat, pos, types = _cell("GeSn", k % 2 == 1, seed=10 + k)
        N = len(types)
        structs.append(
            dict(
                lattice=lat,
                positions=pos,
                atom_types=types,
                ref_energy=rng.normal() - 4.0,
                ref_forces=rng.normal(size=(N, 3)),
                ref_stress=0.01 * rng.normal(size=(3, 3)),
            )
        )
    model = AtomicELMModel(n_hidden=48, seed=3, verbose=False).fit(structs, d)
    ff = MLIPForceField(model)
    lat, pos, types = _cell("GeSn", True, seed=99)
    new = ff(lat, pos, types)
    old = with_reference(lambda: ff(lat, pos, types), ff.descriptor)
    for a, b, what in zip(new, old, ("E", "F", "S")):
        assert _rel(a, b) < RTOL, f"{what}: {_rel(a, b):.2e}"


def test_the_reference_is_still_the_former_kernel():
    """The comparison above is empty if the reference is edited along
    with the package kernel; it must keep the general power per term."""
    import inspect

    src = inspect.getsource(
        ref._acsf_kernel.py_func
        if hasattr(ref._acsf_kernel, "py_func")
        else ref._acsf_kernel
    )
    assert "P = base_c**zeta" in src
    assert "K = 2.0 ** (1.0 - zeta)" in src
