"""ESM as a Coulomb method of the force field (ADR-0008).

- ESMCoulomb gives point-charge energies: the Gaussian energies of
  test_esm_electrostatics.py plus the analytic erfc complement.
- Buckingham + ESM through make_force_fn: forces and in-plane stress are
  derivatives of its energy.
- The z convention: positions wrapped into [0, Lz) give the same result as
  positions centred on z = 0; an atom near the boundary is refused.
- Every solver's ESM rule (ADR-0008, decision 5).
"""

import numpy as np
import pytest
from scipy.special import erfc

from akitsushima.engine import BuckinghamPotential, PotentialEngine
from akitsushima.engine.coulomb.esm import _ESM_EV_PER_E2_ANG as K
from akitsushima.engine.coulomb.esm import ESMCoulomb
from akitsushima.fitting.force_fn import make_force_fn, uses_esm

pytestmark = pytest.mark.filterwarnings("ignore:ESMCalculator. bc=:UserWarning")

# Gaussian-charge references of test_esm_electrostatics.py (check_esm.py)
REF_GAUSS = {
    ("bc1", 3.0): -4.6589353609869075,
    ("bc3", 3.0): -4.658945213307015,
    ("bc2", 3.0): -4.758354573667387,
}
BUCK = {
    frozenset({"Mg", "O"}): {"A": 1284.38, "rho": 0.29969, "C": 0.0},
    frozenset({"O"}): {"A": 9547.96, "rho": 0.21916, "C": 32.0},
}
CH = {"Mg": 2.0, "O": -2.0}


@pytest.mark.parametrize("bc", ["bc1", "bc2", "bc3"])
def test_point_charge_energy_is_gaussian_plus_erfc_complement(bc):
    lat = np.diag([16.0, 16.0, 32.0])
    d = 3.0
    pos = np.array([[8.0, 8.0, -d / 2], [8.0, 8.0, d / 2]])
    q = np.array([1.0, -1.0])
    E = ESMCoulomb(bc, 0.5, 32, 512).compute(lat, pos, ["Na", "Cl"], q)[0]
    ref = REF_GAUSS[(bc, d)] + K * (-1.0) * erfc(d / (2 * 0.5)) / d
    assert abs(E - ref) / 2 < 8e-3


def _mgo_slab(n=3, a=4.21, Lz=30.0, seed=1):
    pos, t = [], []
    for iz, z in enumerate((-a / 4, a / 4)):
        for i in range(n):
            for j in range(n):
                for fx, fy, sp in (
                    (0, 0, "Mg"),
                    (0.5, 0.5, "Mg"),
                    (0.5, 0, "O"),
                    (0, 0.5, "O"),
                ):
                    if iz == 1:
                        sp = "O" if sp == "Mg" else "Mg"
                    pos.append([(i + fx) * a, (j + fy) * a, z])
                    t.append(sp)
    lat = np.diag([n * a, n * a, Lz])
    pos = np.array(pos) + np.random.default_rng(seed).normal(0, 0.05, (len(t), 3))
    return lat, pos, t


def _fn(bc="bc1", nz=128):
    return make_force_fn(
        "buckingham",
        buck_params=BUCK,
        charges=CH,
        r_cut=6.0,
        coulomb_method="esm",
        esm={"bc": bc, "sigma": 0.5, "n_grid_xy": 32, "n_grid_z": nz},
    )


def test_buckingham_esm_forces_and_stress_are_energy_derivatives():
    lat, pos, t = _mgo_slab()
    fn = _fn()
    assert uses_esm(fn)
    _, F, S = fn(lat, pos, t)
    assert np.abs(F.sum(axis=0)).max() < 1e-10
    h = 1e-4
    for k, c in [(0, 0), (5, 2), (20, 1)]:
        p = pos.copy()
        p[k, c] += h
        Ep = fn(lat, p, t)[0]
        p[k, c] -= 2 * h
        Em = fn(lat, p, t)[0]
        assert F[k, c] == pytest.approx(-(Ep - Em) / (2 * h), abs=1e-6)
    V = abs(np.linalg.det(lat))
    fr = pos @ np.linalg.inv(lat)
    for a in (0, 1):
        T = np.eye(3)
        T[a, a] += h
        Ep = fn(lat @ T, fr @ lat @ T, t)[0]
        T = np.eye(3)
        T[a, a] -= h
        Em = fn(lat @ T, fr @ lat @ T, t)[0]
        # the ESM energy is a difference of ~1e3 eV sums; its FD noise is
        # ~3e-5 eV/A^3 in absolute terms (the total stress here is a small
        # difference of Buckingham +0.12 and ESM -0.13 eV/A^3)
        assert S[a, a] == pytest.approx(-(Ep - Em) / (2 * h) / V, abs=6e-5)


def test_positions_wrapped_into_the_cell_give_the_same_result():
    lat, pos, t = _mgo_slab()
    fn = _fn()
    E0, F0, _ = fn(lat, pos, t)
    wrapped = pos.copy()
    wrapped[:, 2] %= lat[2, 2]  # what a solver's wrap produces
    E1, F1, _ = fn(lat, wrapped, t)
    assert E1 == pytest.approx(E0, abs=1e-9)
    np.testing.assert_allclose(F1, F0, atol=1e-9)


def test_atom_near_the_boundary_is_refused():
    lat, pos, t = _mgo_slab()
    pos[0, 2] = lat[2, 2] / 2 - 1.0  # 1 A from the medium, < 4 sigma
    with pytest.raises(ValueError, match="boundary"):
        _fn()(lat, pos, t)


# ---------------------------------------------------------------- solvers


def test_relax_cell_only_in_plane():
    from akitsushima.solvers import GeometryOptimiser

    lat, pos, t = _mgo_slab()
    fn = _fn()
    with pytest.raises(ValueError, match="a_x and b_y"):
        GeometryOptimiser(lat, pos, t, force_fn=fn, relax_cell=True, verbose=False)
    GeometryOptimiser(
        lat,
        pos,
        t,
        force_fn=fn,
        relax_cell=True,
        verbose=False,
        lattice_constraints="1 0 0 0 1 0 0 0 0",
    )
    GeometryOptimiser(lat, pos, t, force_fn=fn, relax_cell=False, verbose=False)


def test_npt_refused_nvt_allowed(tmp_path):
    from akitsushima.solvers.molecular_dynamics import NPTSimulation, NVTSimulation

    lat, pos, t = _mgo_slab()
    fn = _fn()
    with pytest.raises(ValueError, match="NPTSimulation.*ESM"):
        NPTSimulation(
            lat, pos, t, force_fn=fn, n_steps=1, verbose=False, output_dir=str(tmp_path)
        )
    NVTSimulation(
        lat, pos, t, force_fn=fn, n_steps=1, verbose=False, output_dir=str(tmp_path)
    )


def test_phonon_refuses_a_z_supercell():
    from akitsushima.solvers.phonon import PhononCalculator

    lat, pos, t = _mgo_slab()
    with pytest.raises(ValueError, match="supercell"):
        PhononCalculator(lat, pos, t, _fn(), supercell=[1, 1, 2])


def test_gcmc_bc1_refuses_charged_insertion_bc3_allows(tmp_path):
    from akitsushima.solvers.gcmc import GCMCMDSimulation

    lat, pos, t = _mgo_slab()
    kw = dict(
        temperature=300.0, mu=-1.0, n_cycles=1, verbose=False, output_dir=str(tmp_path)
    )
    with pytest.raises(ValueError, match="neutral"):
        GCMCMDSimulation(lat, pos, t, force_fn=_fn("bc1"), species="Mg", **kw)
    GCMCMDSimulation(lat, pos, t, force_fn=_fn("bc3"), species="Mg", **kw)


def test_kappa_refuses_esm():
    from akitsushima.solvers.thermal_conductivity import ThermalConductivityCalculator

    lat, pos, t = _mgo_slab()
    with pytest.raises(ValueError, match="ESM"):
        ThermalConductivityCalculator(lat, pos, t, _fn())


def test_mlip_takes_no_coulomb_method():
    from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml

    with pytest.raises(ValueError, match="electrostatics"):
        _build_force_fn_from_yaml(
            {"type": "MLIP", "model": "x.npz", "coulomb_method": "esm"}
        )


def test_yaml_buckingham_with_esm_builds_a_complete_force_function(tmp_path):
    from akitsushima.fitting.io_utils import save_potential
    from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml

    path = str(tmp_path / "buck.json")
    save_potential(
        {"params": {"buck": BUCK, "charges": CH}, "loss": 0.0},
        path,
        potential_type="buckingham",
        r_cut=6.0,
        species=["Mg", "O"],
    )
    cfg = {
        "type": "buckingham",
        "potential_json": path,
        "r_cut": 6.0,
        "coulomb_method": "esm",
        "esm": {"bc": "bc1", "sigma": 0.5, "n_grid_xy": 32, "n_grid_z": 128},
    }
    fn = _build_force_fn_from_yaml(cfg)[0]
    lat, pos, t = _mgo_slab()
    E = fn(lat, pos, t)[0]
    assert fn.coulomb == "esm"
    assert E == pytest.approx(_fn()(lat, pos, t)[0], rel=1e-12)
    with pytest.raises(ValueError, match="esm is set"):
        _build_force_fn_from_yaml(dict(cfg, coulomb_method="ewald"))


def test_esm_matches_ewald_for_a_symmetric_slab_in_a_long_cell():
    """A dipole-free slab: ESM bc1 against 3D Ewald with a long vacuum (no
    slab correction needed), both point charges, same Buckingham."""
    lat, pos, t = _mgo_slab(Lz=30.0)
    # n_grid_z 1024: the z-grid error is O(dz^2 q^2), 17.6 meV per ion at
    # 512 for these +-2 charges, 4.4 at 1024 (ADR-0003 T6)
    E_esm = _fn(nz=1024)(lat, pos, t)[0]
    big = lat.copy()
    big[2, 2] = 120.0
    ew = make_force_fn(
        PotentialEngine().add(BuckinghamPotential(BUCK, r_cut=6.0)),
        charges=CH,
        r_cut=6.0,
    )
    E_ew = ew(big, pos, t)[0]
    assert abs(E_esm - E_ew) / len(t) < 5e-3
