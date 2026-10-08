"""ADR-0018 stage 2b: mode Grueneisen parameters and the quasi-harmonic
approximation, on Tersoff Si (published set; 2-atom primitive cell,
4x4x4 supercell)."""
import numpy as np
import pytest

from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml
from akitsushima.solvers.qha import (
    mean_gruneisen,
    mode_gruneisen,
    phonons_at,
    quasi_harmonic,
    relax_at_volume,
)

from test_phonon_thermal import _FRAC, _MASS, _PRIM, _SI

_TYPES = ["Si", "Si"]
_Q = np.array([[0, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0.5], [0.25, 0, 0.25],
               [0.375, 0.375, 0.75], [0.1, 0.2, 0.3]])


@pytest.fixture(scope="module")
def si(tmp_path_factory):
    fn = _build_force_fn_from_yaml(
        {"type": "Tersoff", "species": ["Si"], "params": _SI})[0]
    out = str(tmp_path_factory.mktemp("qha"))
    V0 = abs(np.linalg.det(_PRIM))
    st = {}
    for f in np.linspace(0.95, 1.05, 7).tolist() + [0.99, 1.01]:
        r = relax_at_volume(_PRIM, _FRAC @ _PRIM, _TYPES, fn, V0 * f, 4,
                            output_dir=out)
        st[round(f, 4)] = (r, phonons_at(r["lattice"], r["positions"],
                                         _TYPES, fn, 4, masses=_MASS), f)
    return fn, st


def test_relax_at_volume_keeps_volume_and_cubic_shape(si):
    _, st = si
    for r, _, f in st.values():
        assert r["converged"]
        V = abs(np.linalg.det(r["lattice"]))
        assert V == pytest.approx(f * abs(np.linalg.det(_PRIM)), rel=1e-12)
        # diamond stays diamond: the scaled fcc primitive vectors
        assert np.allclose(r["lattice"] / (V / abs(np.linalg.det(_PRIM))) ** (1 / 3),
                           _PRIM, atol=1e-6)
    # compressed -> positive pressure, expanded -> negative
    assert st[0.95][0]["pressure_GPa"] > 4 and st[1.05][0]["pressure_GPa"] < -3


def test_mode_gruneisen_matches_log_derivative_of_frequencies(si):
    """Independent of the dD/dV projection: for non-degenerate modes away
    from crossings, gamma = -d ln w / d ln V, by central differences of
    the eigenfrequencies themselves."""
    _, st = si
    pcs = [st[0.99][1], st[1.0][1], st[1.01][1]]
    g = mode_gruneisen(pcs, _Q[-1:])["gamma"][0]
    vols = [abs(np.linalg.det(p.lattice_prim)) for p in pcs]
    f = []
    for p in (pcs[0], pcs[2]):
        rec = 2 * np.pi * np.linalg.inv(p.lattice_prim).T
        f.append(np.sqrt(np.linalg.eigvalsh(p.build_dynamical_matrix(_Q[-1] @ rec))))
    ref = -(np.log(f[1]) - np.log(f[0])) / (np.log(vols[2]) - np.log(vols[0]))
    assert np.allclose(g, ref, atol=2e-3)


def test_mode_gruneisen_matches_phonopy(si):
    """Oracle: phonopy's GruneisenBase on phonopy force constants of the
    same three relaxed structures (same force function)."""
    pytest.importorskip("phonopy")
    from phonopy import Phonopy
    from phonopy.gruneisen.core import GruneisenBase
    from phonopy.structure.atoms import PhonopyAtoms
    fn, st = si

    def ph(r):
        u = PhonopyAtoms(symbols=_TYPES, cell=r["lattice"],
                         positions=r["positions"], masses=_MASS)
        p = Phonopy(u, supercell_matrix=np.diag([4, 4, 4]), primitive_matrix=None)
        p.generate_displacements(distance=0.01)
        p.forces = np.array([fn(np.array(s.cell), np.array(s.positions),
                                list(s.symbols))[1]
                             for s in p.supercells_with_displacements])
        p.produce_force_constants()
        return p
    P = [ph(st[f][0]) for f in (0.99, 1.0, 1.01)]
    ref = GruneisenBase(P[1].dynamical_matrix, P[2].dynamical_matrix,
                        P[0].dynamical_matrix, qpoints=_Q).get_gruneisen()
    ours = mode_gruneisen([st[f][1] for f in (0.99, 1.0, 1.01)], _Q)["gamma"]
    ok = np.isfinite(ours)
    assert ok.sum() == ours.size - 3          # only the Gamma acoustic modes
    assert np.max(np.abs(ours[ok] - ref[ok])) < 3e-4
    # Gamma optical triplet is degenerate: one gamma for all three
    assert np.ptp(ours[0, 3:]) < 1e-8


def _qha_inputs(st, T):
    keys = [k for k in sorted(st) if k not in (0.99, 1.01)]
    V, E, F, S, C = [], [], [], [], []
    for k in keys:
        r, pc, _ = st[k]
        th = pc.thermal_properties(T, q_mesh=8)
        V.append(abs(np.linalg.det(r["lattice"]))); E.append(r["energy"])
        F.append(th["F_eV"]); S.append(th["S_kB"]); C.append(th["Cv_kB"])
    return np.array(V), np.array(E), np.array(F).T, np.array(S).T, np.array(C).T


@pytest.mark.parametrize("eos,peos", [("birch-murnaghan", "birch_murnaghan"),
                                      ("vinet", "vinet")])
def test_quasi_harmonic_matches_phonopy(si, eos, peos):
    """Oracle: PhonopyQHA on the same E(V), F, S, C_v (converted to its
    kJ/mol and J/K/mol units)."""
    pytest.importorskip("phonopy")
    from phonopy import PhonopyQHA
    _, st = si
    T = np.arange(0.0, 1210.0, 10.0)
    V, E, F, S, C = _qha_inputs(st, T)
    q = quasi_harmonic(V, E, T, F, S, C, eos=eos)
    R = 1.380649e-23 * 6.02214076e23
    p = PhonopyQHA(volumes=V, electronic_energies=E, temperatures=T,
                   free_energy=F * 96.48533212331002, cv=C * R, entropy=S * R,
                   eos=peos)
    n = len(p.volume_temperature)
    sl = slice(1, n - 1)
    assert not q["truncated"]
    assert np.allclose(q["V_A3"][sl], p.volume_temperature[sl], rtol=1e-7)
    assert np.allclose(q["B_GPa"][sl], p.bulk_modulus_temperature[sl], rtol=1e-5)
    assert np.allclose(q["Cp_J_K_mol"][sl], p.heat_capacity_P_polyfit[sl], rtol=1e-5)
    # alpha and gamma are central differences of V(T): from 50 K, where
    # alpha (~1e-8 /K at 10 K) is large against the EOS-fit rounding.
    sl = slice(5, n - 1)
    a, pa = q["alpha_per_K"][sl], np.asarray(p.thermal_expansion)[sl]
    assert np.max(np.abs(a - pa)) < 1e-3 * np.max(np.abs(pa))
    g, pg = q["gamma"][sl], np.asarray(p.gruneisen_temperature)[sl]
    assert np.max(np.abs(g - pg)) < 1e-3 * np.max(np.abs(pg))


def test_thermodynamic_gamma_agrees_with_mean_mode_gamma(si):
    """gamma_th = alpha B V / C_v (QHA) against sum(gamma_i c_i) / sum(c_i)
    from the mode parameters at 300 K: equal in the quasi-harmonic limit
    up to the volume dependence of B and the mesh (0.797 vs 0.780 when
    written)."""
    _, st = si
    T = np.arange(0.0, 610.0, 10.0)
    q = quasi_harmonic(*_qha_inputs(st, T)[:2], T, *_qha_inputs(st, T)[2:])
    g_th = q["gamma"][30]
    m = 8
    grid = np.array([[i / m, j / m, k / m] for i in range(m) for j in range(m)
                     for k in range(m)])
    grid -= np.round(grid)
    g = mode_gruneisen([st[f][1] for f in (0.99, 1.0, 1.01)], grid)
    g_mode = mean_gruneisen(g["frequencies_THz"], g["gamma"], [300.0])[0]
    assert g_th == pytest.approx(g_mode, rel=0.05)


def test_quasi_harmonic_synthetic_limits():
    """No vibrations: V(T) = V0 and alpha = 0.  A free energy that drives
    the minimum out of the sampled range: output stops there."""
    from akitsushima.solvers.elastic import _birch_murnaghan_energy
    V = np.linspace(36.0, 44.0, 7)
    E = _birch_murnaghan_energy(V, 0.0, 40.0, 0.6, 4.5)
    T = np.arange(0.0, 501.0, 10.0)
    zero = np.zeros((len(T), len(V)))
    q = quasi_harmonic(V, E, T, zero, zero + 1.0, zero + 1.0)
    assert np.allclose(q["V_A3"], 40.0, atol=1e-6)
    assert np.allclose(q["alpha_per_K"][1:-1], 0.0, atol=1e-10)
    F = -np.outer(T, V) * 2e-4          # pushes V up linearly with T
    q = quasi_harmonic(V, E, T, F, zero + 1.0, zero + 1.0)
    assert q["truncated"] and q["t_max_K"] < 500.0
    assert q["V_A3"][-1] < V.max()


def test_qha_command_stops_on_imaginary_modes(tmp_path, monkeypatch):
    """A volume with a mode below -imaginary_tol_THz ends the run unless
    skip_unstable_volumes is set; then it is dropped and reported."""
    import json
    import yaml
    from test_cli_smoke import _ns
    from test_cli_templates import _write_si_primitive
    import akitsushima.cli.cmd_qha as mod
    from akitsushima.cli.templates import _DUMP_MAP

    real = mod.phonons_at
    calls = {"n": 0}

    def fake(*a, **k):
        pc = real(*a, **k)
        calls["n"] += 1
        if calls["n"] == 7:                      # the largest volume
            orig = pc.thermal_properties

            def tp(*aa, **kk):
                r = orig(*aa, **kk)
                r["min_frequency_THz"] = -1.0
                return r
            pc.thermal_properties = tp
        return pc
    monkeypatch.setattr(mod, "phonons_at", fake)
    monkeypatch.chdir(tmp_path)
    _write_si_primitive(tmp_path / "POSCAR")
    cfg = yaml.safe_load(_DUMP_MAP["qha"])
    cfg["qha"].update(q_mesh=6, t_max=300.0)
    (tmp_path / "q.yaml").write_text(yaml.safe_dump(cfg))
    assert mod.cmd_qha(_ns(tmp_path / "q.yaml")) == 1
    calls["n"] = 0
    cfg["qha"]["skip_unstable_volumes"] = True
    (tmp_path / "q.yaml").write_text(yaml.safe_dump(cfg))
    assert mod.cmd_qha(_ns(tmp_path / "q.yaml")) == 0
    out = json.loads((tmp_path / "qha_output" / "qha_results.json").read_text())
    assert len(out["volumes_A3"]) == 6 and len(out["dropped_unstable_volumes_A3"]) == 1
