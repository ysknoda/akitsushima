"""Strain coordinates for variable-cell relaxation (ADR-0041).

- the gradient in x = [r0, s (F - I)] against finite differences of E(x),
  with and without constraints and fix_volume;
- _get_x / _set_x round trip;
- space group kept after relaxing Si (fcc primitive) and AlN (wurtzite)
  from a strained lattice, for all six methods;
- frozen lattice components bit for bit (also with ESM);
- the corrected 'vasp' scheme on the lattice path, its deprecation with
  strain coordinates;
- checkpoint restart rules, recording of cell_coordinates.
"""

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from akitsushima.cli.templates import _DUMP_MAP
from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml
from akitsushima.solvers.geometry_optimiser import GeometryOptimiser
from akitsushima.solvers.qha import build_supercell


def _si_fn():
    # Tersoff Si of the qha template (r_cut 3.0 A)
    return _build_force_fn_from_yaml(yaml.safe_load(_DUMP_MAP["qha"])["potential"])[0]


def _aln_fn():
    import test_phonon_regressions as T

    return _build_force_fn_from_yaml(
        {"type": "Tersoff", "species": ["Al", "N"], "params": T._ALN}
    )[0]


def _si_fcc_prim(a=5.50, n=3):
    prim = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    frac = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    return build_supercell(prim, frac @ prim, ["Si"] * 2, [n, n, n])


def _aln_wurtzite(a=3.30, c=5.40, u=0.38):
    lat = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    frac = np.array(
        [
            [1 / 3, 2 / 3, 0],
            [2 / 3, 1 / 3, 0.5],
            [1 / 3, 2 / 3, u],
            [2 / 3, 1 / 3, 0.5 + u],
        ]
    )
    return build_supercell(lat, frac @ lat, ["Al", "Al", "N", "N"], [3, 3, 2])


def _sheared_si():
    """64 Si atoms in a low-symmetry, non-orthogonal cell, rattled."""
    rng = np.random.default_rng(3)
    lat = np.array([[5.6, 0.3, -0.2], [0.9, 5.2, 0.4], [-0.5, 0.7, 5.9]])
    f8 = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    f8 = np.vstack([f8, f8 + 0.25]) + rng.normal(0, 0.02, (8, 3))
    frac = np.vstack(
        [(f8 + [i, j, k]) / 2 for i in range(2) for j in range(2) for k in range(2)]
    )
    lat = 2 * lat
    return lat, frac @ lat, ["Si"] * 64


def _go(L, P, T, fn, tmp_path, **kw):
    return GeometryOptimiser(
        L,
        P,
        T,
        force_fn=fn,
        relax_cell=True,
        verbose=False,
        timing=False,
        output_dir=str(tmp_path),
        **kw,
    )


# ------------------------------------------------------------------ gradient


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"lattice_constraints": "1 1 0 0 1 1 1 0 1"},
        {"fix_volume": True},
        {"fix_volume": True, "fix_cell": np.eye(3, dtype=bool)},
    ],
    ids=["free", "constraints", "fix_volume", "fix_volume+fix_cell"],
)
def test_gradient_matches_finite_differences_of_E_of_x(kw, tmp_path):
    """-dE/dx along random directions of the free subspace (fix_volume:
    tangent to the volume constraint; the retraction is second order)."""
    L, P, T = _sheared_si()
    fn = _si_fn()
    go = _go(L, P, T, fn, tmp_path, **kw)
    _, f, s = fn(go.lattice, go.positions, T)
    g = go._generalised_force(f, s)
    x = go._get_x()
    n_ion = 3 * go.N
    rng = np.random.default_rng(0)

    def energy(xx):
        st = go._save_state()
        go._set_x(xx)
        e = float(fn(go.lattice, go.positions, T)[0])
        go._restore_state(st)
        return e

    for _ in range(6):
        d = rng.normal(size=x.size)
        d[n_ion:] = go._strain_project(d[n_ion:])
        d /= np.linalg.norm(d)
        h = 1e-5
        fd = -(energy(x + h * d) - energy(x - h * d)) / (2 * h)
        an = float(g @ d)
        assert abs(fd - an) <= 2e-6 * max(abs(an), 1e-3), (fd, an)


def test_set_x_is_a_pure_function_in_strain_coordinates(tmp_path):
    """Unlike the raw-lattice path, _set_x(x) then _get_x() returns x."""
    L, P, T = _sheared_si()
    go = _go(L, P, T, _si_fn(), tmp_path)
    x = go._get_x()
    assert np.abs(x[3 * go.N :]).max() < 1e-14  # F = I up to rounding
    rng = np.random.default_rng(1)
    dx = np.concatenate([rng.normal(0, 1e-3, 3 * go.N), rng.normal(0, 1e-2, 9)])
    go._set_x(x + dx)
    assert np.allclose(go._get_x(), x + dx, atol=1e-12, rtol=0)


def test_strain_gradient_is_the_covariant_cell_force(tmp_path):
    """At F = I the cell block is V sigma / s for any choice of basis: an
    isotropic stress gives an isotropic step on the fcc primitive cell."""
    L, P, T = _si_fcc_prim(n=3)
    go = _go(L, P, T, _si_fn(), tmp_path)
    sigma = 1e-3 * np.eye(3)
    g = go._generalised_force(np.zeros_like(P), sigma)
    V = abs(np.linalg.det(L))
    assert np.allclose(
        g[3 * go.N :].reshape(3, 3), V * sigma / V ** (1 / 3), rtol=1e-12
    )


# ------------------------------------------------------------------ symmetry


@pytest.mark.parametrize("method", ["fire", "lbfgs", "sd", "diis", "rmmdiis", "rfo"])
def test_relaxation_keeps_the_space_group(method, tmp_path):
    """From a strained lattice, Si stays Fd-3m and AlN P6_3mc at symprec
    1e-5.  The raw-lattice path gave R-3m / Cmc2_1 / Cm / P1 here.

    RMM-DIIS runs to tighter tolerances (ADR-0041 finding 4): its line
    search accepts any energy decrease, so a stiff shear mode can
    overshoot and grow from rounding noise (a 1e-10 seed reached 3.9e-6
    before decaying; raw-lattice path 1.2e-2).  With multi-threaded force
    sums the noise is ~1e-15, and at fmax 1e-3 / pmax 1e-2 the run stopped
    mid-growth with 3.4e-6 of shear (1.7e-4 GPa, inside pmax): P1 at
    1e-5.  fmax 1e-4 / pmax 1e-3 gives Fd-3m."""
    spglib = pytest.importorskip("spglib")
    tol = {"fmax": 1e-4, "pmax": 1e-3} if method == "rmmdiis" else {"fmax": 1e-3, "pmax": 1e-2}
    for (L, P, T), fn, want in (
        (_si_fcc_prim(), _si_fn(), 227),
        (_aln_wurtzite(), _aln_fn(), 186),
    ):
        r = _go(L, P, T, fn, tmp_path, method=method, max_steps=1000, **tol).run()
        assert r["converged"]
        num = [{"Si": 14, "Al": 13, "N": 7}[t] for t in T]
        Lr = r["lattice"]
        ds = spglib.get_symmetry_dataset(
            (Lr, r["positions"] @ np.linalg.inv(Lr), num), symprec=1e-5
        )
        assert ds.number == want, (method, ds.international)


def test_fix_volume_keeps_volume_and_space_group(tmp_path):
    spglib = pytest.importorskip("spglib")
    L, P, T = _aln_wurtzite()
    r = _go(
        L,
        P,
        T,
        _aln_fn(),
        tmp_path,
        method="fire",
        fix_volume=True,
        fmax=1e-3,
        pmax=1e-2,
        max_steps=400,
    ).run()
    assert r["converged"]
    assert abs(np.linalg.det(r["lattice"]) / np.linalg.det(L) - 1) < 1e-12
    Lr = r["lattice"]
    ds = spglib.get_symmetry_dataset(
        (Lr, r["positions"] @ np.linalg.inv(Lr), [13, 13, 7, 7] * (len(T) // 4)),
        symprec=1e-5,
    )
    assert ds.number == 186


def test_rfo_takes_the_cell_rotations_out(tmp_path):
    """The rotation directions have stress-sized negative curvature in
    linear-F coordinates; RFO's model must not contain them."""
    L, P, T = _aln_wurtzite()
    go = _go(L, P, T, _aln_fn(), tmp_path, method="rfo")
    Q = go._rotation_basis()
    assert Q.shape == (3 * go.N + 9, 3)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-12)
    assert np.abs(Q[: 3 * go.N]).max() < 1e-15
    # gradient orthogonal to them (E(QF) = E(F))
    _, f, s = go._get_forces_and_stress()
    g = go._generalised_force(f, s)
    assert np.abs(Q.T @ g).max() < 1e-10 * np.abs(g).max()
    # a frozen component that pins rotations about z leaves two
    go2 = _go(L, P, T, _aln_fn(), tmp_path, lattice_constraints="1 0 1 1 1 1 1 1 1")
    assert go2._rotation_basis().shape[1] == 2
    assert (
        _go(L, P, T, _aln_fn(), tmp_path, cell_coordinates="lattice")._rotation_basis()
        is None
    )


# ------------------------------------------------------------------ constraints


def test_frozen_components_stay_bit_for_bit(tmp_path):
    L, P, T = _si_fcc_prim(n=3)
    mask = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
    fc = np.zeros((3, 3), dtype=bool)
    fc[2, 2] = True
    r = _go(
        L,
        P,
        T,
        _si_fn(),
        tmp_path,
        method="lbfgs",
        max_steps=30,
        lattice_constraints=" ".join(str(int(m)) for m in mask),
        fix_cell=fc,
    ).run()
    frozen = ~mask.reshape(3, 3) | fc
    assert np.array_equal(r["lattice"][frozen], L[frozen])
    assert not np.allclose(r["lattice"][~frozen], L[~frozen])


def test_esm_in_plane_relaxation_keeps_frozen_components(tmp_path):
    from test_esm_coulomb import _fn, _mgo_slab

    lat, pos, t = _mgo_slab()
    r = GeometryOptimiser(
        lat,
        pos,
        t,
        force_fn=_fn(),
        relax_cell=True,
        verbose=False,
        timing=False,
        max_steps=3,
        skin_width=0.2,  # r_cut 6 A in a 12.63 A slab
        output_dir=str(tmp_path),
        lattice_constraints="1 0 0 0 1 0 0 0 0",
    ).run()
    frozen = ~np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=bool).reshape(3, 3)
    assert np.array_equal(r["lattice"][frozen], lat[frozen])
    assert r["cell_coordinates"] == "strain"


def test_residual_rotation_stays_small(tmp_path):
    """The gradient is orthogonal to rotations, but F picks up a rotation
    at second order in the strain along a curved path.  Relaxing this
    sheared cell stretches it by +21 / -10 / -12 %: the polar rotation of
    h_new^T h_old^-T was 0.255 deg (fire) and 0.290 deg (lbfgs) with
    strain coordinates, 0.228 / 0.210 deg on the raw-lattice path
    (ADR-0041).  Harmless (a rigid rotation), and no worse than before."""
    L, P, T = _sheared_si()
    r = _go(
        L, P, T, _si_fn(), tmp_path, method="fire", fmax=5e-3, pmax=0.05, max_steps=300
    ).run()
    A = r["lattice"].T @ np.linalg.inv(L.T)  # h_new^T = A h_old^T
    U, _, Vt = np.linalg.svd(A)
    Rm = U @ Vt
    angle = np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))
    assert angle < 0.5, angle


# ------------------------------------------------------------------ vasp scheme


def test_vasp_scheme_is_lattice_at_stress_on_the_lattice_path(tmp_path):
    L, P, T = _aln_wurtzite()
    go = _go(
        L,
        P,
        T,
        _aln_fn(),
        tmp_path,
        cell_coordinates="lattice",
        cell_update_scheme="vasp",
    )
    sigma = np.diag([1e-3, 1e-3, 2e-3])  # hexagonal symmetry
    f = go._generalised_force(np.zeros_like(P), sigma)[3 * go.N :].reshape(3, 3)
    assert np.array_equal(f, L @ sigma)
    h = L + 50 * f
    gam = np.degrees(
        np.arccos(h[0] @ h[1] / np.linalg.norm(h[0]) / np.linalg.norm(h[1]))
    )
    assert abs(gam - 120.0) < 1e-12  # stress @ lattice gave 107.5 deg-type distortions


def test_vasp_scheme_is_deprecated_and_inert_with_strain(tmp_path):
    L, P, T = _aln_wurtzite()
    with pytest.warns(DeprecationWarning, match="cell_update_scheme='vasp'"):
        gv = _go(L, P, T, _aln_fn(), tmp_path, cell_update_scheme="vasp")
    gg = _go(L, P, T, _aln_fn(), tmp_path)
    sigma = np.diag([1e-3, 2e-3, -1e-3])
    assert np.array_equal(
        gv._generalised_force(np.zeros_like(P), sigma),
        gg._generalised_force(np.zeros_like(P), sigma),
    )


def test_unknown_cell_coordinates_is_refused(tmp_path):
    L, P, T = _aln_wurtzite()
    with pytest.raises(ValueError, match="cell_coordinates"):
        _go(L, P, T, _aln_fn(), tmp_path, cell_coordinates="metric")


# ------------------------------------------------------------------ records, restart


def test_result_and_relax_log_record_cell_coordinates(tmp_path):
    L, P, T = _aln_wurtzite()
    r = _go(L, P, T, _aln_fn(), tmp_path, max_steps=2).run()
    assert r["cell_coordinates"] == "strain"
    log = (tmp_path / "relax.log").read_text()
    assert log.rstrip().endswith("Cell coordinates: strain")
    r = GeometryOptimiser(
        L,
        P,
        T,
        force_fn=_aln_fn(),
        verbose=False,
        timing=False,
        max_steps=1,
        output_dir=str(tmp_path / "fixed"),
    ).run()
    assert r["cell_coordinates"] is None
    assert "Cell coordinates" not in (tmp_path / "fixed" / "relax.log").read_text()


def test_restart_needs_the_same_cell_coordinates(tmp_path):
    from akitsushima.solvers._checkpoint import load_checkpoint, save_checkpoint

    L, P, T = _aln_wurtzite()
    ck = str(tmp_path / "ck.npz")
    _go(
        L, P, T, _aln_fn(), tmp_path / "a", max_steps=3, cell_coordinates="lattice"
    ).run(checkpoint=ck, checkpoint_every=1)
    with pytest.raises(ValueError, match="cell_coordinates: lattice"):
        _go(L, P, T, _aln_fn(), tmp_path / "a", max_steps=6).run(checkpoint=ck)
    # A checkpoint written before ADR-0041 has no cell_coordinates: it is
    # a lattice-coordinate run, continued by cell_coordinates='lattice'.
    meta, arr = load_checkpoint(ck)
    del meta["settings"]["cell_coordinates"]
    save_checkpoint(ck, meta, arr)
    with pytest.raises(ValueError, match="cell_coordinates: lattice"):
        _go(L, P, T, _aln_fn(), tmp_path / "a", max_steps=6).run(checkpoint=ck)
    r = _go(
        L, P, T, _aln_fn(), tmp_path / "a", max_steps=6, cell_coordinates="lattice"
    ).run(checkpoint=ck)
    assert r["n_steps"] == 6
    # strain run: restart re-references h0 on the restored cell
    ck2 = str(tmp_path / "ck2.npz")
    _go(L, P, T, _aln_fn(), tmp_path / "b", max_steps=3).run(
        checkpoint=ck2, checkpoint_every=1
    )
    go = _go(L, P, T, _aln_fn(), tmp_path / "b", max_steps=6)
    r = go.run(checkpoint=ck2)
    assert r["n_steps"] == 6
    assert not np.array_equal(go._h0, L)
