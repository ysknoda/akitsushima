"""ADR-0018 stage 2a: harmonic thermodynamics by a q-mesh mode sum,
Theta_D(T) from C_v, and the corrected sound-velocity Debye temperature.

Tersoff Si (published Si(C) set, lambda3 = 0), 2-atom primitive cell,
4x4x4 supercell (r_cut 3.0 A).
"""
import contextlib
import io

import numpy as np
import pytest

from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml
from akitsushima.solvers.phonon import (
    PhononCalculator,
    _debye_cv_ratio,
    _debye_theta_from_cv,
)

_SI = dict(A=1830.8, B=471.18, lambda1=2.4799, lambda2=1.7322, lambda3=0.0,
           beta=1.1e-6, n=0.78734, c=100390.0, d=16.217, h=-0.59825,
           R=2.85, D=0.15)
_A0 = 5.432005          # Tersoff Si equilibrium (tests/test_elastic.py)
_PRIM = 0.5 * _A0 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
_FRAC = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
_MASS = [28.0855, 28.0855]


@pytest.fixture(scope="module")
def si():
    fn = _build_force_fn_from_yaml(
        {"type": "Tersoff", "species": ["Si"], "params": _SI})[0]
    with contextlib.redirect_stdout(io.StringIO()):
        pc = PhononCalculator(_PRIM, _FRAC @ _PRIM, ["Si", "Si"], fn,
                              supercell=4, masses=_MASS)
        pc.compute_ifc()
    return fn, pc


def test_mesh_is_gamma_centred_for_even_meshes(si):
    _, pc = si
    f = pc.mesh_frequencies(4)
    assert f.shape == (64, 6)
    # Gamma is the first grid point: three acoustic zeros (ASR).
    assert np.all(np.abs(f[0, :3]) < 1e-4)


def test_thermodynamic_identities(si):
    """U = F + TS, S = -dF/dT, C_v = dU/dT (central differences), T = 0
    gives the zero-point energy, and C_v -> 3 k_B per atom at high T."""
    _, pc = si
    T, h = 300.0, 0.5
    # 1e7 K: the quantum correction 1 - x^2/12 is ~1e-10 there.
    r = pc.thermal_properties([0.0, T - h, T, T + h, 1e7], q_mesh=8)
    F, U, S, Cv = r["F_eV"], r["U_eV"], r["S_kB"], r["Cv_kB"]
    kB = 8.617333262145179e-05
    assert F[0] == U[0] == pytest.approx(r["ZPE_eV"], rel=1e-14)
    assert U[2] == pytest.approx(F[2] + T * kB * S[2], rel=1e-12)
    assert -(F[3] - F[1]) / (2 * h) / kB == pytest.approx(S[2], rel=1e-6)
    assert (U[3] - U[1]) / (2 * h) / kB == pytest.approx(Cv[2], rel=1e-6)
    # 3 k_B per atom less the three acoustic modes at Gamma that are left
    # out: 6 x 512 modes, 3 excluded (phonopy does the same).
    assert r["Cv_kB_per_atom"][4] == pytest.approx(3.0 * (1 - 3 / 3072), rel=1e-6)
    assert r["n_imaginary"] == 0


def test_free_energy_is_the_mode_sum(si):
    _, pc = si
    with contextlib.redirect_stdout(io.StringIO()):
        F = pc.phonon_free_energy(300.0, n_q=8)
    assert F == pc.thermal_properties([300.0], q_mesh=8)["F_eV"][0]


def test_mode_sum_matches_phonopy_mode_functions(si):
    """Our sums against phonopy's own per-mode formulas on the same
    frequencies.  The residue is phonopy's CODATA constants (k_B, h, eV
    differ from the exact SI 2019 values by ~6e-7)."""
    pytest.importorskip("phonopy")
    import phonopy.phonon.thermal_properties as tp
    from phonopy.physical_units import get_physical_units
    u = get_physical_units()
    _, pc = si
    f = pc.mesh_frequencies(8)
    e = f[f > 1e-3] * u.THzToEv
    r = pc.thermal_properties([50.0, 300.0, 1000.0], q_mesh=8)
    for k, t in enumerate(r["temperatures"]):
        Fp = np.sum(tp.mode_F(t, e)) / len(f) * u.EvTokJmol
        Sp = np.sum(tp.mode_S(t, e)) / len(f) * u.EvTokJmol * 1e3
        Cp = np.sum(tp.mode_cv(t, e)) / len(f) * u.EvTokJmol * 1e3
        assert r["F_kJ_mol"][k] == pytest.approx(Fp, rel=1e-5)
        assert r["S_J_K_mol"][k] == pytest.approx(Sp, rel=1e-5)
        assert r["Cv_J_K_mol"][k] == pytest.approx(Cp, rel=1e-5)


def test_thermal_properties_match_phonopy(si):
    """End to end against phonopy (its own displacements and force
    constants, same force function, same Gamma-centred mesh and cutoff)."""
    pytest.importorskip("phonopy")
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    fn, pc = si
    unit = PhonopyAtoms(symbols=["Si", "Si"], cell=_PRIM,
                        scaled_positions=_FRAC, masses=_MASS)
    phn = Phonopy(unit, supercell_matrix=np.diag([4, 4, 4]),
                  primitive_matrix=None)
    phn.generate_displacements(distance=0.01)
    phn.forces = np.array([fn(np.array(s.cell), np.array(s.positions),
                              list(s.symbols))[1]
                           for s in phn.supercells_with_displacements])
    phn.produce_force_constants()
    T = [10.0, 100.0, 300.0, 1000.0]
    phn.run_mesh([12, 12, 12], is_gamma_center=True)
    phn.run_thermal_properties(temperatures=T, cutoff_frequency=1e-3)
    d = phn.get_thermal_properties_dict()
    r = pc.thermal_properties(T, q_mesh=12)
    assert np.allclose(r["F_kJ_mol"], d["free_energy"], rtol=1e-4, atol=1e-3)
    # S at 10 K is the most sensitive to the two force-constant builds
    # (1.1e-4 relative); 3e-4 covers it.
    assert np.allclose(r["S_J_K_mol"], d["entropy"], rtol=3e-4)
    assert np.allclose(r["Cv_J_K_mol"], d["heat_capacity"], rtol=1e-4)


def test_debye_function_and_inversion():
    """D(1) = 0.9517 (C_v at T = Theta_D, e.g. Kittel Fig. 5.x tables);
    D -> 1 - y^2/20 for small y; inversion recovers Theta_D."""
    assert _debye_cv_ratio(1.0) == pytest.approx(0.95173, abs=2e-5)
    assert _debye_cv_ratio(0.01) == pytest.approx(1 - 0.01**2 / 20, abs=1e-9)
    for theta, T in ((645.0, 30.0), (300.0, 300.0), (400.0, 2000.0)):
        ratio = _debye_cv_ratio(theta / T)
        assert _debye_theta_from_cv(ratio, T) == pytest.approx(theta, rel=1e-8)
    assert np.isnan(_debye_theta_from_cv(1.0, 300.0))
    assert np.isnan(_debye_theta_from_cv(0.5, 0.0))


def test_vsound_debye_temperature_uses_the_debye_mean(si):
    """Regression (ADR-0018): v_D lacked the 1/3 of 3/v_D^3 = 1/v_L^3 +
    2/v_T^3 and theta_D came out 0.69x.  With it, the low-temperature
    Theta_D(T) from C_v (mesh 32, 15 K) lands within 5 % of the
    sound-velocity value (542 K vs 565 K; the sound-velocity method
    averages three directions only, and the mesh is not converged for
    the long-wavelength modes)."""
    _, pc = si
    with contextlib.redirect_stdout(io.StringIO()):
        th_v = pc.debye_temperature_from_vsound(verbose=False)
    th_c = pc.thermal_properties([15.0], q_mesh=32)["theta_D_K"][0]
    assert th_v == pytest.approx(541.9, rel=2e-3)
    assert th_c == pytest.approx(th_v, rel=0.05)
