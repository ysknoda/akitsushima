"""Anisotropic NPT: cell_dof aniso | tri | xy | xy_iso | z (ADR-0036).

Oracles: the isotropic integrator (a generator E = I reproduces it), the
NVE limit (tau -> infinity), the dt^2 scaling of the conserved-quantity
drift, the target pressure in every free component, the cubic-symmetry
equalities of Si, a constrained relax (z mode) and ESM's geometric
requirements (xy modes).
"""

import os
import warnings

import numpy as np
import pytest
from conftest import diamond_si_dict, randomly_perturbed
from test_md_restart import (
    EVERY,
    STOP_AT,
    SW_SI,
    _files,
    _kill_after,
    _Killed,
    _make,
    _same_result,
)

import akitsushima as ak
from akitsushima.constants import K_B
from akitsushima.solvers.molecular_dynamics import (
    CELL_DOF_MODES,
    NPTSimulation,
    NVESimulation,
    _cell_generators,
)

MODES = ("aniso", "tri", "xy", "xy_iso", "z")


def _sw():
    return ak.force_fn_from_config(SW_SI)


def _si(perturb=0.05, seed=3):
    s = diamond_si_dict(2)  # 64 atoms, the smallest cell SW fits
    return randomly_perturbed(s, magnitude=perturb, seed=seed) if perturb else s


def _npt(s, out, **kw):
    base = dict(
        force_fn=_sw(),
        r_cut=3.8,
        temperature=600.0,
        pressure=1.0,
        timestep=1.0,
        n_steps=50,
        tau_nhc=50.0,
        tau_baro=200.0,
        traj_every=10**9,
        thermo_every=10**9,
        output_dir=str(out),
        verbose=False,
        save_plots=False,
    )
    base.update(kw)
    return NPTSimulation(s["lattice"], s["positions"], s["atom_types"], **base)


def _block(x, nb=10):
    """Mean and its standard error from nb block averages."""
    x = np.asarray(x, dtype=float)
    m = len(x) // nb * nb
    b = x[:m].reshape(nb, -1).mean(axis=1)
    return float(x.mean()), float(b.std(ddof=1) / np.sqrt(nb))


def _thermo(out):
    with open(os.path.join(out, "thermo.log")) as fh:
        names = fh.readline().split()
    rows = np.loadtxt(os.path.join(out, "thermo.log"), skiprows=1, ndmin=2)
    # header tokens: Step Time(ps) E_pot(eV) ... -> one name per column
    return {n.split("(")[0]: rows[:, i] for i, n in enumerate(names)}


# ------------------------------------------------------------ generators


def test_generators_are_symmetric_and_cover_the_modes():
    assert CELL_DOF_MODES == ("iso", "aniso", "tri", "xy", "xy_iso", "z")
    sizes = {"aniso": 3, "tri": 6, "xy": 2, "xy_iso": 1, "z": 1}
    for mode, n in sizes.items():
        E = _cell_generators(mode)
        assert E.shape == (n, 3, 3)
        assert np.array_equal(E, np.transpose(E, (0, 2, 1)))
    assert np.array_equal(_cell_generators("xy_iso")[0], np.diag([1.0, 1.0, 0.0]))
    assert np.array_equal(_cell_generators("z")[0], np.diag([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="cell_dof"):
        _cell_generators("iso")


def test_unknown_mode_is_refused_and_settings_record_it(tmp_path):
    s = _si()
    with pytest.raises(ValueError, match="cell_dof must be one of"):
        _npt(s, tmp_path, cell_dof="xz")
    iso = _npt(s, tmp_path)
    assert iso.cell_dof == "iso" and "cell_dof" not in iso._ck_settings()
    tri = _npt(s, tmp_path, cell_dof="TRI")
    assert tri._ck_settings()["cell_dof"] == "tri"
    assert tri.n_baro == 6 and tri.Q1_b == pytest.approx(6 * iso.Q1_b)
    assert tri.W == iso.W  # one undivided mass per generator (decision 7)


# ------------------------------------------------- 1. the isotropic limit


@pytest.mark.slow
def test_identity_generator_reproduces_the_isotropic_integrator(tmp_path):
    """With E = I the general equations are the iso ones.  The particle
    chain is switched off (tau_nhc huge) because the iso path takes the
    kinetic pressure of its first barostat half-step before that chain
    acts; the barostat chain stays on."""
    s = _si()
    kw = dict(pressure=2.0, n_steps=300, tau_nhc=1e8, tau_baro=200.0)
    a = _npt(s, tmp_path / "a", **kw).run()
    sim = _npt(s, tmp_path / "b", cell_dof="z", **kw)
    sim._gens = np.eye(3)[None]
    sim._gen_tr = np.array([3.0])
    sim.n_baro = 1
    sim.Q1_b = K_B * 600.0 * 200.0**2
    b = sim.run()
    assert np.max(np.abs(a["positions"] - b["positions"])) < 1e-10
    assert np.max(np.abs(a["lattice"] - b["lattice"])) < 1e-11
    assert b["H_npt_drift"] == pytest.approx(a["H_npt_drift"], abs=1e-9)


# ------------------------------------------------------- 2. the NVE limit


@pytest.mark.slow
@pytest.mark.parametrize("mode", MODES)
def test_every_mode_reduces_to_nve(tmp_path, mode):
    s = _si()
    common = dict(
        force_fn=_sw(),
        r_cut=3.8,
        timestep=1.0,
        n_steps=150,
        temperature=600.0,
        traj_every=10**9,
        thermo_every=10**9,
        seed=42,
        verbose=False,
    )
    nve = NVESimulation(
        s["lattice"],
        s["positions"],
        s["atom_types"],
        **common,
        output_dir=str(tmp_path / "nve"),
    ).run()
    npt = NPTSimulation(
        s["lattice"],
        s["positions"],
        s["atom_types"],
        **common,
        output_dir=str(tmp_path / "npt"),
        pressure=0.0,
        tau_nhc=1e8,
        tau_baro=1e8,
        save_plots=False,
        cell_dof=mode,
    ).run()
    assert np.max(np.abs(npt["positions"] - nve["positions"])) < 1e-8
    assert abs(npt["T_mean"] - nve["T_mean"]) < 1e-6


# ------------------------------------------ 3. conserved quantity, dt^2


@pytest.mark.slow
def test_drift_scales_with_dt_squared(tmp_path):
    s = _si()
    d = [
        _npt(
            s, tmp_path / str(dt), cell_dof="tri", timestep=dt, n_steps=int(400 / dt)
        ).run()["H_npt_drift"]
        for dt in (1.0, 0.5)
    ]
    assert abs(d[0]) < 0.02  # eV over 400 fs, 64 atoms
    assert 3.0 < d[0] / d[1] < 5.5


# --------------------------- 4. fixed components, and the logged columns


@pytest.mark.parametrize("mode", MODES)
def test_fixed_components_stay_bit_for_bit(tmp_path, mode):
    s = _si()
    L0 = s["lattice"].copy()
    sim = _npt(s, tmp_path, cell_dof=mode, n_steps=6, thermo_every=1)
    r = sim.run()
    L = r["lattice"]
    off = L - np.diag(np.diag(L))
    if mode != "tri":
        assert not np.any(off)
    else:
        assert np.any(off)
    if mode in ("xy", "xy_iso"):
        assert L[2, 2] == L0[2, 2]
    if mode == "xy_iso":
        assert L[0, 0] == L[1, 1] != L0[0, 0]
    if mode == "z":
        assert L[0, 0] == L0[0, 0] and L[1, 1] == L0[1, 1]
    cols = _thermo(tmp_path)
    want = ["Pxx", "Pyy", "Pzz", "a", "b", "c"]
    if mode == "tri":
        want += ["Pyz", "Pxz", "Pxy", "alpha", "beta", "gamma"]
    assert set(want) <= set(cols)
    assert cols["c"][-1] == pytest.approx(np.linalg.norm(L[2]), abs=1e-4)
    # the scalar pressure is the trace of the logged tensor
    P = (cols["Pxx"] + cols["Pyy"] + cols["Pzz"]) / 3
    assert np.allclose(P, cols["P_inst"], atol=2e-4)
    for k in ("lattice_mean", "P_tensor_mean_gpa", "abc_mean", "abc_std"):
        assert k in r
    assert r["cell_dof"] == mode


def test_tri_warns_once_when_an_angle_leaves_the_range(tmp_path, monkeypatch):
    sim = _npt(_si(), tmp_path, cell_dof="tri", n_steps=4)
    monkeypatch.setattr(sim, "_cell_angles", lambda: np.array([90.0, 90.0, 40.0]))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim.run()
    assert sum("left [45, 135]" in str(x.message) for x in w) == 1


# ------------------------------- 4/5. target pressure and cubic symmetry


@pytest.mark.slow
def test_free_components_reach_the_target_and_cubic_si_stays_cubic(tmp_path):
    """SW Si, 64 atoms, 300 K, 1 GPa: every free diagonal component
    averages to the target and every free shear to zero (3 standard
    errors, 10 blocks); aniso and tri give the iso volume and a = b = c."""
    s = _si(perturb=0.0)
    eq, ns = 1000, 4000
    res = {}
    for mode in ("iso", "aniso", "tri"):
        out = tmp_path / mode
        r = _npt(
            s,
            out,
            cell_dof=mode,
            temperature=300.0,
            n_steps=ns,
            equil_steps=eq,
            thermo_every=1,
            seed=5,
        ).run()
        c = {k: v[eq:] for k, v in _thermo(out).items()}
        res[mode] = (r, c)
        if mode == "iso":
            continue
        for k in ("Pxx", "Pyy", "Pzz"):
            m, se = _block(c[k])
            assert abs(m - 1.0) < 3 * se, (mode, k, m, se)
        if mode == "tri":
            for k in ("Pyz", "Pxz", "Pxy"):
                m, se = _block(c[k])
                assert abs(m) < 3 * se, (mode, k, m, se)
        a = [_block(c[k]) for k in "abc"]
        for i in range(3):
            for j in range(i):
                assert abs(a[i][0] - a[j][0]) < 3 * np.hypot(a[i][1], a[j][1])
    v_iso = _block(res["iso"][1]["V"])
    for mode in ("aniso", "tri"):
        v = _block(res[mode][1]["V"])
        assert abs(v[0] - v_iso[0]) < 3 * np.hypot(v[1], v_iso[1]), (mode, v, v_iso)


# ---------------------------------------------------- 6. shear relaxes


@pytest.mark.slow
def test_tri_relaxes_an_imposed_shear(tmp_path):
    s = _si(perturb=0.0)
    F = np.eye(3)
    F[0, 1] = F[1, 0] = 0.03  # gamma starts at 86.7 degrees
    sheared = dict(s, lattice=s["lattice"] @ F, positions=s["positions"] @ F)
    out = tmp_path / "shear"
    _npt(
        sheared,
        out,
        cell_dof="tri",
        temperature=50.0,
        pressure=0.0,
        n_steps=3000,
        thermo_every=1,
    ).run()
    c = _thermo(out)
    assert c["gamma"][0] < 87.0
    assert abs(np.mean(c["gamma"][-1000:]) - 90.0) < 0.05
    assert abs(np.mean(c["Pxy"][-1000:])) < 0.02  # GPa, from -6.2 at the start


# ------------------------------------------- 7. z mode against relax


@pytest.mark.slow
def test_z_mode_matches_the_constrained_relax(tmp_path):
    """Lx, Ly held 2 % compressed: Lz expands (Poisson).  At 10 K the
    mean Lz equals the relax with only c free to within 1 % of that
    expansion."""
    from akitsushima.solvers.geometry_optimiser import GeometryOptimiser

    s = _si(perturb=0.0)
    S = np.diag([0.98, 0.98, 1.0])
    comp = dict(s, lattice=s["lattice"] @ S, positions=s["positions"] @ S)
    rel = GeometryOptimiser(
        comp["lattice"],
        comp["positions"],
        comp["atom_types"],
        force_fn=_sw(),
        relax_cell=True,
        lattice_constraints="0 0 0 0 0 0 0 0 1",
        verbose=False,
        output_dir=str(tmp_path / "rel"),
        fmax=1e-4,
    ).run()
    c_rel = rel["lattice"][2, 2]
    r = _npt(
        comp,
        tmp_path / "md",
        cell_dof="z",
        temperature=10.0,
        pressure=0.0,
        n_steps=3000,
        equil_steps=1000,
    ).run()
    c0 = s["lattice"][2, 2]
    assert c_rel > c0 + 0.1
    assert abs(r["abc_mean"][2] - c_rel) < 0.01 * (c_rel - c0)
    assert r["lattice"][0, 0] == comp["lattice"][0, 0]  # held exactly
    assert r["abc_mean"][0] == pytest.approx(comp["lattice"][0, 0], abs=1e-12)


# --------------------------------------------------------------- 8. ESM


def _esm_slab(a=None):
    from test_esm_coulomb import _fn, _mgo_slab

    lat, pos, t = _mgo_slab(n=4)  # 16.8 A: the r_cut 6 A + skin fits
    if a is not None:
        S = np.diag([a / lat[0, 0], a / lat[1, 1], 1.0])
        lat, pos = lat @ S, pos @ S
    return lat, pos, t, _fn(nz=64)


def test_esm_allows_only_the_in_plane_modes_at_zero_pressure(tmp_path):
    lat, pos, t, fn = _esm_slab()
    kw = dict(force_fn=fn, n_steps=1, verbose=False, output_dir=str(tmp_path))
    for mode in ("iso", "aniso", "tri", "z"):
        with pytest.raises(ValueError, match="NPTSimulation.*ESM.*xy or xy_iso"):
            NPTSimulation(lat, pos, t, cell_dof=mode, **kw)
    with pytest.raises(ValueError, match="only pressure 0"):
        NPTSimulation(lat, pos, t, cell_dof="xy", pressure=0.5, **kw)
    for mode in ("xy", "xy_iso"):
        NPTSimulation(lat, pos, t, cell_dof=mode, **kw)


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["xy", "xy_iso"])
def test_esm_slab_keeps_its_boundaries(tmp_path, mode):
    lat, pos, t, fn = _esm_slab()
    r = NPTSimulation(
        lat,
        pos,
        t,
        force_fn=fn,
        temperature=300.0,
        n_steps=8,
        tau_nhc=50.0,
        tau_baro=100.0,
        verbose=False,
        save_plots=False,
        output_dir=str(tmp_path),
        cell_dof=mode,
    ).run()
    L = r["lattice"]
    assert L[2, 2] == lat[2, 2]
    assert not np.any(L - np.diag(np.diag(L)))  # a, b along x, y; c along z
    assert L[0, 0] != lat[0, 0]


@pytest.mark.slow
def test_esm_slab_in_plane_pressure_averages_to_zero(tmp_path):
    """Started near the in-plane equilibrium (relax with a_x, b_y free gives
    a = 16.0213 A for this slab; ADR-0036 result), xy_iso at 10 K and
    P = 0 keeps <Pxx> and <Pyy> at zero."""
    lat, pos, t, fn = _esm_slab(a=16.02)
    out = tmp_path / "md"
    eq = 100
    NPTSimulation(
        lat,
        pos,
        t,
        force_fn=fn,
        temperature=10.0,
        n_steps=300,
        equil_steps=eq,
        tau_nhc=50.0,
        tau_baro=100.0,
        thermo_every=1,
        verbose=False,
        save_plots=False,
        output_dir=str(out),
        cell_dof="xy_iso",
    ).run()
    c = _thermo(out)
    for k in ("Pxx", "Pyy"):
        m, se = _block(c[k][eq:])
        assert abs(m) < max(3 * se, 2e-3), (k, m, se)


# ---------------------------------------------------- 9. checkpoint


@pytest.mark.slow
def test_tri_restart_continues_bit_for_bit(tmp_path):
    ref = _make("npt", tmp_path / "ref", cell_dof="tri").run()
    out = tmp_path / "ck"
    ck = str(out / "md_checkpoint.npz")
    sim = _make("npt", out, cell_dof="tri")
    _kill_after(sim, STOP_AT + 1)
    with pytest.raises(_Killed):
        sim.run(checkpoint=ck, checkpoint_every=EVERY)
    got = _make("npt", out, cell_dof="tri").run(checkpoint=ck, checkpoint_every=EVERY)
    _same_result(got, ref)
    for k in ("lattice_mean", "P_tensor_mean_gpa", "abc_mean", "abc_std"):
        assert np.array_equal(got[k], ref[k]), k
    assert _files(out, "npt") == _files(tmp_path / "ref", "npt")
    with pytest.raises(ValueError, match="cell_dof"):  # the mode must match
        _make("npt", out, cell_dof="aniso").run(checkpoint=ck)


# ------------------------------------------------------------- anneal


def test_anneal_passes_cell_dof_to_npt_and_refuses_it_under_nvt():
    from akitsushima.solvers.global_opt import GlobalOptimiser

    s = _si()
    args = (s["lattice"], s["positions"], s["atom_types"], _sw(), 3.8)
    with pytest.raises(ValueError, match="cell_dof 'tri' needs ensemble npt"):
        GlobalOptimiser(
            *args, md={"ensemble": "nvt", "cell_dof": "tri"}, verbose=False, plot=False
        )
    g = GlobalOptimiser(
        *args, md={"ensemble": "nvt", "cell_dof": "iso"}, verbose=False, plot=False
    )
    assert "cell_dof" not in g.md_kw
    g = GlobalOptimiser(
        *args, md={"ensemble": "npt", "cell_dof": "tri"}, verbose=False, plot=False
    )
    assert g.md_kw["cell_dof"] == "tri"
    with pytest.raises(ValueError, match="baro_thermostat 'langevin' needs"):
        GlobalOptimiser(
            *args,
            md={"ensemble": "nvt", "baro_thermostat": "langevin"},
            verbose=False,
            plot=False,
        )


# ------------------------------------- Langevin piston (ADR-0036 addendum)


def test_langevin_piston_option_is_validated_and_recorded(tmp_path):
    s = _si()
    with pytest.raises(ValueError, match="baro_thermostat must be"):
        _npt(s, tmp_path, baro_thermostat="bussi")
    assert "baro_thermostat" not in _npt(s, tmp_path)._ck_settings()
    sim = _npt(s, tmp_path, baro_thermostat="Langevin", cell_dof="tri")
    assert sim._ck_settings()["baro_thermostat"] == "langevin"
    # ADR-0044: the particle chain is carried as arrays too
    assert set(sim._ck_arrays()) == {"p_g", "baro_heat", "xi_nhc", "vxi_nhc"}


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["iso", "tri"])
def test_langevin_piston_keeps_an_effective_conserved_quantity(tmp_path, mode):
    """H_NPT minus the heat the noise put in drifts only by the
    integration error, which shrinks with dt (same noise sequence per
    step count is not needed: the drift is far below the heat)."""
    s = _si()
    d, heat = [], []
    for dt in (1.0, 0.5):
        sim = _npt(
            s,
            tmp_path / f"{mode}{dt}",
            cell_dof=mode,
            baro_thermostat="langevin",
            timestep=dt,
            n_steps=int(400 / dt),
        )
        d.append(sim.run()["H_npt_drift"])
        heat.append(abs(sim.baro_heat[0]))
    assert abs(d[0]) < 0.02
    assert abs(d[1]) < abs(d[0])
    assert max(heat) > 0.0


@pytest.mark.slow
def test_langevin_piston_reduces_to_nve(tmp_path):
    s = _si()
    common = dict(
        force_fn=_sw(),
        r_cut=3.8,
        timestep=1.0,
        n_steps=150,
        temperature=600.0,
        traj_every=10**9,
        thermo_every=10**9,
        seed=42,
        verbose=False,
    )
    nve = NVESimulation(
        s["lattice"],
        s["positions"],
        s["atom_types"],
        **common,
        output_dir=str(tmp_path / "nve"),
    ).run()
    npt = NPTSimulation(
        s["lattice"],
        s["positions"],
        s["atom_types"],
        **common,
        output_dir=str(tmp_path / "npt"),
        pressure=0.0,
        tau_nhc=1e8,
        tau_baro=1e8,
        save_plots=False,
        cell_dof="tri",
        baro_thermostat="langevin",
    ).run()
    assert np.max(np.abs(npt["positions"] - nve["positions"])) < 1e-8


def test_langevin_piston_step_samples_the_maxwell_distribution(tmp_path):
    """The Ornstein-Uhlenbeck half-step drives any momenta to N(0, W kT)
    and books the kinetic energy it adds as heat.  10,000 momenta through
    2,000 half-steps (5 friction times) at tau_baro = 200 fs."""
    sim = _npt(
        _si(), tmp_path, cell_dof="tri", baro_thermostat="langevin", temperature=300.0
    )
    kT = K_B * 300.0
    sim.p_g = np.full(10000, 3.0 * np.sqrt(sim.W * kT))  # far too hot
    e0 = 0.5 * float(sim.p_g @ sim.p_g) / sim.W
    for _ in range(2000):
        sim._baro_langevin()
    p = sim.p_g / np.sqrt(sim.W * kT)
    assert abs(np.mean(p**2) - 1.0) < 0.05
    assert abs(np.mean(p)) < 0.05
    assert abs(np.mean(p**4) - 3.0) < 0.25  # Gaussian kurtosis
    e1 = 0.5 * float(sim.p_g @ sim.p_g) / sim.W
    assert sim.baro_heat[0] == pytest.approx(e1 - e0, rel=1e-9)


@pytest.mark.slow
def test_langevin_piston_restart_continues_bit_for_bit(tmp_path):
    kw = dict(cell_dof="tri", baro_thermostat="langevin")
    ref = _make("npt", tmp_path / "ref", **kw).run()
    out = tmp_path / "ck"
    ck = str(out / "md_checkpoint.npz")
    sim = _make("npt", out, **kw)
    _kill_after(sim, STOP_AT + 1)
    with pytest.raises(_Killed):
        sim.run(checkpoint=ck, checkpoint_every=EVERY)
    got = _make("npt", out, **kw).run(checkpoint=ck, checkpoint_every=EVERY)
    _same_result(got, ref)
    assert _files(out, "npt") == _files(tmp_path / "ref", "npt")
