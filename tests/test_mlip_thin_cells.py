"""ADR-0024 (ADR-0023 finding O7): a force function that finds its own
neighbours (an MLIP) is not held to the solver neighbour list's
minimum-image precondition, which still guards every classical potential.

The 8-atom Si cell is 5.4 A wide: half of it (2.7 A) is below the ACSF
cutoff plus skin (3.771 + 0.5 A), yet the ACSF handles it (its own image
search), so relax and MD must run -- and give exactly what the same MLIP
gives when the solver is handed a cutoff small enough to pass the check.
"""

import numpy as np
import pytest
from test_elm_options import _si_acsf
from test_mlip_chain_rule import _sw_si_structures

import akitsushima as ak
from akitsushima.engine.neighborlist import NullNeighbourList, uses_neighbour_list
from akitsushima.fitting.models.atomic import AtomicELMModel

SW_SI = {"epsilon": 2.1683, "sigma": 2.0951, "a": 1.80, "gamma": 1.20,
         "lambda": 21.0, "A": 7.049556277, "B": 0.6022245584, "p": 4.0,
         "q": 0.0, "costheta0": -1.0 / 3.0}


@pytest.fixture(scope="module")
def mlip_fn():
    model = AtomicELMModel(n_hidden=20, verbose=False).fit(_sw_si_structures(12, 0), _si_acsf())
    return ak.make_force_fn(model)


def _thin_cell():
    s = _sw_si_structures(1, 7)[0]
    return s["lattice"], s["positions"], s["atom_types"]


def test_mlip_declares_no_neighbour_list(mlip_fn):
    assert not uses_neighbour_list(mlip_fn)
    sw = ak.make_force_fn(ak.PotentialEngine().add(
        ak.StillingerWeberPotential(params=SW_SI, species=["Si"])))
    assert uses_neighbour_list(sw)


def test_relax_with_mlip_runs_on_a_cell_thinner_than_its_cutoff(mlip_fn, tmp_path):
    lat, pos, types = _thin_cell()
    kw = dict(force_fn=mlip_fn, method="fire", relax_cell=True, fmax=0.05,
              max_steps=30, verbose=False)
    go = ak.GeometryOptimiser(lat, pos, types, output_dir=str(tmp_path / "a"), **kw)
    assert isinstance(go._nl, NullNeighbourList)
    a = go.run()
    # the former workaround: a cutoff small enough to pass the check
    b = ak.GeometryOptimiser(lat, pos, types, r_cut=1.0, output_dir=str(tmp_path / "b"), **kw).run()
    np.testing.assert_array_equal(a["positions"], b["positions"])
    np.testing.assert_array_equal(a["lattice"], b["lattice"])


def test_md_with_mlip_runs_on_a_cell_thinner_than_its_cutoff(mlip_fn, tmp_path):
    lat, pos, types = _thin_cell()
    ak.NVTSimulation(lattice=lat, positions=pos, atom_types=types, force_fn=mlip_fn,
                     temperature=300.0, timestep=1.0, n_steps=5, thermo_every=1,
                     output_dir=str(tmp_path), verbose=False).run()


def test_classical_potential_is_still_refused_on_a_thin_cell(tmp_path):
    lat, pos, types = _thin_cell()
    sw = ak.make_force_fn(ak.PotentialEngine().add(
        ak.StillingerWeberPotential(params=SW_SI, species=["Si"])))
    with pytest.raises(ValueError, match="half the smallest perpendicular"):
        ak.GeometryOptimiser(lat, pos, types, force_fn=sw, max_steps=2, verbose=False,
                             output_dir=str(tmp_path)).run()
