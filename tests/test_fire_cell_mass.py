"""Mass of the cell coordinates in FIRE (ADR-0053).

FIRE gave every coordinate mass 1.  The cell coordinate's stiffness
grows with the cell edge (about B L for x = L eps), so in large cells
the cell set FIRE's time step and a 108864-atom GeSn SW relaxation
stopped at 2000 steps.  fire_cell_mass ('auto' = N^(1/3) by default)
divides the two half-kicks of the cell coordinates.

Fixture: tests/data/fire_gesn (63-atom GeSn vacancy cell, fitted SW), as
in test_solvers.test_fire_variable_cell_does_not_limit_cycle.
"""

import os

import numpy as np
import pytest
import yaml

from akitsushima import GeometryOptimiser
from akitsushima.cli.common import _cli_force_fn
from akitsushima.io import parse_poscar

D = os.path.join(os.path.dirname(__file__), "data", "fire_gesn")
P = parse_poscar(os.path.join(D, "POSCAR"))
FN, RC = _cli_force_fn(yaml.safe_load(open(os.path.join(D, "potential.yaml"))), "test")

# 0.4.0a2 (before ADR-0053), same machine-independent quantities as below:
# steps, energy, sum |positions|, sum |lattice|.
REF_042 = {
    "strain": (224, -251.67295828301113, 1085.4070838101982, 36.39604882663113),
    "lattice": (209, -251.67298669948056, 1085.4184717440226, 36.40375262129317),
}


def _opt(tmp_path, **kw):
    args = dict(
        force_fn=FN,
        r_cut=RC,
        relax_cell=True,
        fmax=0.01,
        pmax=0.1,
        max_steps=1000,
        verbose=False,
        method="fire",
        output_dir=str(tmp_path),
    )
    args.update(kw)
    return GeometryOptimiser(P["lattice"], P["positions"], P["atom_types"], **args)


@pytest.mark.parametrize("cc", ["strain", "lattice"])
def test_mass_one_reproduces_0_4_0a2(tmp_path, cc):
    r = _opt(tmp_path, cell_coordinates=cc, fire_cell_mass=1).run()
    n, e, sp, sl = REF_042[cc]
    assert r["converged"] and r["n_steps"] == n
    assert r["energy"] == pytest.approx(e, rel=1e-12, abs=0.0)
    assert float(np.abs(r["positions"]).sum()) == pytest.approx(sp, rel=1e-12, abs=0.0)
    assert float(np.abs(r["lattice"]).sum()) == pytest.approx(sl, rel=1e-12, abs=0.0)


def test_auto_is_cube_root_of_n_and_bad_values_are_refused(tmp_path):
    assert _opt(tmp_path).fire_cell_mass == pytest.approx(63 ** (1 / 3), rel=1e-15)
    assert _opt(tmp_path, fire_cell_mass="AUTO").fire_cell_mass == pytest.approx(
        63 ** (1 / 3), rel=1e-15
    )
    assert _opt(tmp_path, fire_cell_mass=2.5).fire_cell_mass == 2.5
    for bad in (0, -1.0, float("nan"), float("inf"), "heavy"):
        with pytest.raises(ValueError, match="fire_cell_mass"):
            _opt(tmp_path, fire_cell_mass=bad)


@pytest.mark.parametrize("cc", ["strain", "lattice"])
def test_auto_converges_to_the_same_minimum(tmp_path, cc):
    e0 = FN(P["lattice"], P["positions"], P["atom_types"])[0]
    r = _opt(tmp_path, cell_coordinates=cc).run()
    assert r["converged"], f"FIRE did not converge in {r['n_steps']} steps"
    assert r["energy"] < e0
    assert abs(r["energy"] - REF_042[cc][1]) < 0.01  # same basin, eV


@pytest.mark.parametrize(
    "kw", [dict(relax_cell=False), dict(method="lbfgs"), dict(method="sd")]
)
def test_mass_has_no_effect_without_fire_cell_dof(tmp_path, kw):
    kw = dict(kw, max_steps=60)
    a = _opt(tmp_path / "a", fire_cell_mass=1, **kw).run()
    b = _opt(tmp_path / "b", fire_cell_mass=37.0, **kw).run()
    assert a["n_steps"] == b["n_steps"]
    assert a["energy"] == b["energy"]
    np.testing.assert_array_equal(a["positions"], b["positions"])
    np.testing.assert_array_equal(a["lattice"], b["lattice"])


def test_checkpoint_records_the_mass(tmp_path):
    """An older checkpoint (FIRE + relax_cell, no fire_cell_mass item) is
    the mass-1 run: it continues with fire_cell_mass 1, and 'auto' (or any
    other mass) is refused with a message naming the setting."""
    ck = str(tmp_path / "ck.npz")
    _opt(tmp_path / "w", fire_cell_mass=1, max_steps=30).run(
        checkpoint=ck, checkpoint_every=10
    )
    with pytest.raises(ValueError, match="fire_cell_mass"):
        _opt(tmp_path / "w", max_steps=60).run(checkpoint=ck, checkpoint_every=10)
    r = _opt(tmp_path / "w", fire_cell_mass=1, max_steps=60).run(
        checkpoint=ck, checkpoint_every=10
    )
    assert r["n_steps"] == 60

    ck2 = str(tmp_path / "ck2.npz")
    _opt(tmp_path / "v", max_steps=30).run(checkpoint=ck2, checkpoint_every=10)
    with pytest.raises(ValueError, match="fire_cell_mass"):
        _opt(tmp_path / "v", fire_cell_mass=2.0, max_steps=60).run(
            checkpoint=ck2, checkpoint_every=10
        )
    r = _opt(tmp_path / "v", max_steps=60).run(checkpoint=ck2, checkpoint_every=10)
    assert r["n_steps"] == 60


def test_cli_keys():
    from akitsushima.cli import cmd_anneal

    assert "fire_cell_mass" in cmd_anneal._RELAX_KEYS
