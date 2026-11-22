"""EAM / PPPM / ESM speed-ups and need_stress (ADR-0043).

Each new path is checked against the path it replaced, kept in the code
as the reference: the EAM numpy path, the full complex-FFT PPPM methods
and the ESM Green-function matrices.  need_stress=False must leave
energy and forces unchanged and return None for the stress.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import akitsushima as ak
from akitsushima.fitting.force_fn import (
    accepts_need_stress,
    call_force_fn,
    make_force_fn,
)

pytestmark = pytest.mark.filterwarnings("ignore:ESMCalculator. bc=:UserWarning")

FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
ROCK = np.vstack([FCC, FCC + [0.5, 0, 0]])
BUCK = {
    frozenset({"Mg", "O"}): {"A": 1284.38, "rho": 0.29969, "C": 0.0},
    frozenset({"O"}): {"A": 9547.96, "rho": 0.21916, "C": 32.0},
}
CH = {"Mg": 2.0, "O": -2.0}


def cubic(basis, a, n, types, rattle=0.05, seed=0):
    rng = np.random.default_rng(seed)
    cells = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    frac = ((basis[None] + cells[:, None]).reshape(-1, 3)) / n
    lat = np.eye(3) * a * n
    return lat, frac @ lat + rng.normal(0, rattle, (len(frac), 3)), list(types) * n**3


def _rel(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(b).max(), 1e-300)


# ------------------------------------------------------------------ EAM


def _eam_cases():
    ni = ak.force_fn_from_config(
        {"type": "EAM", "form": "zhou2004", "elements": ["Ni"], "species": ["Ni"]}
    )
    nial = ak.force_fn_from_config(
        {
            "type": "EAM",
            "form": "zhou2004",
            "elements": ["Ni", "Al"],
            "species": ["Ni", "Al"],
        }
    )
    L, P, T = cubic(FCC, 3.6, 4, ["Ni"] * 4, seed=1)
    T = list(np.random.default_rng(3).choice(["Ni", "Al"], len(T)))
    out = [
        ("zhou_ni", ni, cubic(FCC, 3.52, 4, ["Ni"] * 4)),
        ("zhou_nial", nial, (L, P, T)),
    ]
    # a tabulated (setfl) potential written from the Zhou one
    return out


@pytest.mark.parametrize("name", ["zhou_ni", "zhou_nial", "setfl_nial"])
def test_eam_numba_path_matches_the_numpy_path(name, tmp_path):
    cases = dict((n, (f, s)) for n, f, s in _eam_cases())
    if name == "setfl_nial":
        from akitsushima.engine.eam_zhou import write_setfl

        fn, s = cases["zhou_nial"]
        pot = fn.engine._potentials[0]
        path = str(tmp_path / "NiAl.eam.alloy")
        write_setfl(path, pot, ["Ni", "Al"])
        fn = ak.force_fn_from_config(
            {"type": "EAM", "setfl": path, "species": ["Ni", "Al"]}
        )
    else:
        fn, s = cases[name]
    pot = fn.engine._potentials[0]
    assert pot._nb is not None  # every function has a known form
    E1, F1, S1 = fn(*s)
    keep = pot._nb
    pot._nb = None  # force the numpy path
    try:
        E0, F0, S0 = fn(*s)
    finally:
        pot._nb = keep
    assert abs(E1 - E0) / abs(E0) < 1e-12
    assert _rel(F1, F0) < 1e-10
    assert _rel(S1, S0) < 1e-10


def test_eam_bare_callables_keep_the_numpy_path():
    from akitsushima.engine import PotentialEngine
    from akitsushima.engine.eam import EAMPotential

    pot = EAMPotential(
        lambda rho: -np.sqrt(rho),
        lambda r: np.exp(-r),
        lambda r: 0.5 * np.exp(-2 * r),
        r_cut=5.0,
        species=["X"],
    )
    assert pot._nb is None
    fn = make_force_fn(PotentialEngine().add(pot))
    E, F, S = fn(*cubic(FCC, 3.0, 4, ["X"] * 4))
    assert np.isfinite(E) and np.abs(F.sum(axis=0)).max() < 1e-10


# ------------------------------------------------------------------ PPPM


@pytest.mark.parametrize("mesh", [None, 15])
def test_pppm_real_fft_matches_the_complex_fft_reference(mesh):
    from akitsushima.engine.coulomb.pppm import PPPMSolver
    from akitsushima.engine.potential_engine import _pppm_alpha_from_accuracy

    L, P, T = cubic(ROCK, 4.212, 3, ["Mg"] * 4 + ["O"] * 4)
    q = np.array([CH[t] for t in T])
    alpha = _pppm_alpha_from_accuracy(6.0, 1e-5)
    M = 16 if mesh is None else mesh  # even and odd meshes
    sol = PPPMSolver(L, alpha=alpha, mesh=M, order=6, r_cut=6.0)
    E, F, S = sol.compute(P, q)
    # full-FFT reference from the methods kept for it
    rho = sol._assign_charges(P, q)
    rho_k = np.fft.fftn(rho)
    e_mesh = sol._mesh_energy(rho_k, sol._G_mesh * rho_k)
    f_mesh = sol._mesh_forces(P, q, rho_k)
    s_mesh = sol._mesh_stress(rho_k)
    from akitsushima.engine.coulomb.ewald import real_space_coulomb

    e_r, f_r, s_r = real_space_coulomb(L, P, q, alpha, 6.0)
    E0 = (
        e_r
        + e_mesh
        - sol._self_energy(q, alpha)
        + sol._charged_system(q, alpha, sol.volume)
    )
    assert abs(E - E0) / abs(E0) < 1e-12
    assert _rel(F, f_r + f_mesh) < 1e-10
    assert _rel(S, s_r + s_mesh) < 1e-10


# ------------------------------------------------------------------ ESM


@pytest.mark.parametrize("bc", ["bc1", "bc2", "bc3"])
def test_esm_recursions_match_the_matrix_reference(bc):
    import test_esm_coulomb as TE

    from akitsushima.engine.coulomb.esm import ESMCalculator

    lat, pos, t = TE._mgo_slab()
    calc = ESMCalculator(
        lat,
        pos,
        t,
        charges=CH,
        bc=bc,
        sigma=0.5,
        n_grid_xy=24,
        n_grid_z=96,
        external_field=0.05 if bc == "bc2" else 0.0,
    )
    E, F, S = calc(lat, pos, t)
    V_new = calc._V_Gz.copy()
    rho = calc._rho_Gz
    V_old = calc._solve_poisson_esm_matrix(
        rho, calc._Gpar, calc._z_grid, calc._dz, calc._Lz
    )
    assert _rel(V_new, V_old) < 1e-10
    S_old = calc._compute_stress_matrix(
        rho, V_old, calc._Gpar, calc._GX, calc._GY, lat, calc._dz
    )
    assert _rel(S, S_old) < 1e-10


def test_esm_apply_against_dense_kernels_on_a_random_column():
    """The recursions against _green_function_(d)matrix for one g, all
    boundary conditions, a random complex column."""
    from akitsushima.engine.coulomb.esm import ESMCalculator, _esm_apply

    Lz, nz = 30.0, 64
    z = np.linspace(-Lz / 2, Lz / 2, nz, endpoint=False)
    dz = Lz / nz
    rng = np.random.default_rng(0)
    x = rng.normal(size=nz) + 1j * rng.normal(size=nz)
    for code, bc in enumerate(("_vv", "_vm", "_mm")):
        for g in (0.05, 0.7, 4.0):
            G = ESMCalculator._green_function_matrix(
                g, z[:, None], z[None, :], bc, Lz / 2
            )
            dG = ESMCalculator._green_function_dmatrix(
                g, z[:, None], z[None, :], bc, Lz / 2
            )
            y = _esm_apply(np.array([g]), z, dz, Lz, code, x[None], False)[0]
            dy = _esm_apply(np.array([g]), z, dz, Lz, code, x[None], True)[0]
            assert _rel(y, G @ x) < 1e-12, (bc, g)
            assert _rel(dy, dG @ x) < 1e-12, (bc, g)


# ------------------------------------------------------------------ need_stress


def _fns():
    import test_esm_coulomb as TE

    out = []
    for meth in ("pppm", "ewald"):
        out.append(
            (
                meth,
                make_force_fn(
                    "buckingham",
                    buck_params=BUCK,
                    charges=CH,
                    r_cut=6.0,
                    coulomb_method=meth,
                ),
                cubic(ROCK, 4.212, 3, ["Mg"] * 4 + ["O"] * 4),
            )
        )
    out.append(("esm", TE._fn("bc2", 96), TE._mgo_slab()))
    out.append(
        (
            "eam",
            ak.force_fn_from_config(
                {
                    "type": "EAM",
                    "form": "zhou2004",
                    "elements": ["Ni"],
                    "species": ["Ni"],
                }
            ),
            cubic(FCC, 3.52, 4, ["Ni"] * 4),
        )
    )
    return out


@pytest.mark.parametrize("which", ["pppm", "ewald", "esm", "eam"])
def test_need_stress_false_keeps_energy_and_forces_and_returns_none(which):
    name, fn, (L, P, T) = [c for c in _fns() if c[0] == which][0]
    assert accepts_need_stress(fn)
    E, F, S = fn(L, P, T)
    E2, F2, S2 = fn(L, P, T, need_stress=False)
    assert S is not None and S2 is None
    assert E2 == E
    assert np.array_equal(F2, F)


def test_call_force_fn_passes_the_flag_only_to_callables_that_take_it():
    seen = {}

    def plain(lat, pos, types):
        return 0.0, np.zeros((len(types), 3)), np.zeros((3, 3))

    def kw(lat, pos, types, **k):  # **kwargs does not count
        seen.update(k)
        return 0.0, np.zeros((len(types), 3)), np.zeros((3, 3))

    lat, pos = np.eye(3) * 5, np.zeros((1, 3))
    assert call_force_fn(plain, lat, pos, ["X"], need_stress=False)[2] is not None
    assert call_force_fn(kw, lat, pos, ["X"], need_stress=False)[2] is not None
    assert "need_stress" not in seen


def test_md_skips_the_stress_only_where_nobody_reads_it(tmp_path):
    """NVT with PPPM: the same trajectory with need_stress honoured or
    ignored (the flag never changes energies or forces), and the stress is
    asked for exactly on the trajectory steps."""
    from akitsushima.solvers.molecular_dynamics import NVTSimulation

    base = make_force_fn(
        "buckingham", buck_params=BUCK, charges=CH, r_cut=6.0, coulomb_method="pppm"
    )
    asked = []

    def counting(lat, pos, types, neigh_list=None, need_stress=True):
        asked.append(need_stress)
        return base(lat, pos, types, neigh_list=neigh_list, need_stress=need_stress)

    counting.r_cut = 6.0

    def ignoring(lat, pos, types, neigh_list=None):
        return base(lat, pos, types, neigh_list=neigh_list)

    ignoring.r_cut = 6.0
    L, P, T = cubic(ROCK, 4.212, 4, ["Mg"] * 4 + ["O"] * 4)
    res = []
    for k, fn in enumerate((counting, ignoring)):
        sim = NVTSimulation(
            L,
            P,
            T,
            fn,
            temperature=300.0,
            n_steps=12,
            equil_steps=0,
            timestep=1.0,
            traj_every=4,
            thermo_every=1,
            seed=1,
            verbose=False,
            output_dir=str(tmp_path / str(k)),
        )
        res.append(sim.run())
    a, b = res
    assert np.array_equal(a["positions"], b["positions"])
    # one call before the loop, then steps 0..11; trajectory at 0, 4, 8
    assert asked[0] is True
    assert [i for i, v in enumerate(asked[1:]) if v] == [0, 4, 8]


@pytest.mark.parametrize("M", [16, 17])
def test_pppm_half_spectrum_on_a_sheared_cell(M):
    """Hexagonal cell, even and odd mesh: on an even mesh the Nyquist
    index is its own negative and k(-k) != -k(k); the partner-averaged
    factors reproduce the full-mesh sums (and Re(ifftn)) exactly."""
    from akitsushima.engine.coulomb.ewald import real_space_coulomb
    from akitsushima.engine.coulomb.pppm import PPPMSolver

    rng = np.random.default_rng(1)
    a, c = 3.25, 5.2
    L = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]) * np.array([[4], [4], [3]])
    fr = np.array([[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0.5], [1 / 3, 2 / 3, 0.38], [2 / 3, 1 / 3, 0.88]])
    cells = np.array([[i, j, k] for i in range(4) for j in range(4) for k in range(3)])
    F = ((fr[None] + cells[:, None]) / np.array([4, 4, 3])).reshape(-1, 3)
    P = F @ L + rng.normal(0, 0.05, (len(F), 3))
    q = np.array([2.0, 2.0, -2.0, -2.0] * len(cells))
    sol = PPPMSolver(L, alpha=0.45, mesh=M, order=5, r_cut=5.0)
    E, Fo, S = sol.compute(P, q)
    rho = sol._assign_charges(P, q)
    rk = np.fft.fftn(rho)
    er, fr_, sr = real_space_coulomb(L, P, q, 0.45, 5.0)
    E0 = (er + sol._mesh_energy(rk, sol._G_mesh * rk) - sol._self_energy(q, 0.45)
          + sol._charged_system(q, 0.45, sol.volume))
    assert abs(E - E0) / abs(E0) < 1e-12
    assert _rel(Fo, fr_ + sol._mesh_forces(P, q, rk)) < 1e-10
    assert _rel(S, sr + sol._mesh_stress(rk)) < 1e-10
