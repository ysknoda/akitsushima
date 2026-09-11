"""Regression tests for solvers/dynamics_analysis.py (DiffusionAnalyzer).

Two independent verification strategies, matching this project's
established style:

1. A REAL, short LJ FCC Al NVE trajectory, used only for a smoke test
   plus the equipartition-theorem check on VACF(0) (0.5 m <v_x^2> =
   0.5 k_B T per component, an exact physical law for any system in
   thermal equilibrium, independent of whether the system actually
   diffuses).

2. A SYNTHETIC, i.i.d.-Gaussian random-walk trajectory (bypassing
   run_md() entirely, isolating the MSD/VACF *machinery* the same way
   test_solvers.py's test_nhc_thermostat_isolated_is_stable_and_bounded
   isolates the NHC propagator from a full MD run), with an EXACTLY
   analytically-known diffusion coefficient derived independently for
   both the Einstein (MSD-slope) and Green-Kubo (VACF-integral) routes
   -- see the derivation in this file's own comments -- giving a clean,
   decisive cross-check that the two independent methods agree with
   each other AND with the known theoretical value.
"""
import numpy as np
import pytest

from akitsushima.constants import EV_ANG_TO_AMU_ANG_FS2, K_B
from akitsushima.engine import LennardJonesPotential, PotentialEngine
from akitsushima.fitting import make_force_fn
from akitsushima.solvers import DiffusionAnalyzer

from conftest import fcc_al_dict


def _fcc_al_supercell(n=2):
    base = fcc_al_dict()
    lattice = base["lattice"] * n
    positions, atom_types = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                shift = np.array([i, j, k], dtype=float) @ base["lattice"]
                positions.append(base["positions"] + shift)
                atom_types.extend(base["atom_types"])
    return {
        "lattice": lattice,
        "positions": np.concatenate(positions, axis=0),
        "atom_types": atom_types,
    }


def test_msd_vacf_smoke_and_equipartition():
    """Real (short) NVE trajectory: outputs must be finite/sane-shaped,
    and VACF(0) (the mean-squared velocity per component, averaged over
    atoms) must match the equipartition theorem's prediction
    k_B T / m for this single-species system, an exact statistical-
    mechanics identity independent of whether the system diffuses.

    On ``equil_steps``
    ------------------
    Comparing VACF(0) against ``k_B * T_target / m`` is only a valid test
    if the NVT stage has actually brought the system TO ``T_target``.  It
    had not: this test used ``equil_steps=200`` (20 fs), after which the
    production run sits near 245 K, not 500 K -- the textbook halving you
    get when a solid is started from Maxwell-Boltzmann velocities at T and
    then allowed to share that energy with its potential degrees of
    freedom.  The test nevertheless passed, because it was calibrated
    against dynamics driven by the stale/skin-truncated neighbour data
    described in NeighbourList.get_pairs_arrays' bug-fix note.

    With that defect fixed the thermostat converges monotonically, so the
    honest fix is to equilibrate properly.  Measured production
    temperature by equilibration length (fixed code):
    200 -> 245 K, 1000 -> 312 K, 3000 -> 389 K, 6000 -> 465 K,
    10000 -> 500 K, at which point VACF(0)/(k_B T/m) = 0.9999, 1.0102,
    1.0243, 1.0193, 1.0169 over seeds 1-5 -- equipartition recovered to
    within 2.4%, comfortably inside the rel=0.15 tolerance, at ~3.4 s.

    That monotonic convergence is itself notable: the same sweep BEFORE
    the neighbour-list fix ran 0.78 -> 2.14 -> 19.3 -> diverged (H_nhc
    ~1e18), which this project had recorded as an inherent chaotic
    instability of a small periodic LJ system rather than a defect.  It
    was the neighbour-list defect.
    """
    fn = make_force_fn(PotentialEngine().add(
        LennardJonesPotential({frozenset({"Al"}): {"epsilon": 0.4, "sigma": 2.5}},
                              r_cut=3.2, shift=True)))
    al = _fcc_al_supercell(n=2)
    T = 500.0
    da = DiffusionAnalyzer(al["lattice"], al["positions"], al["atom_types"],
                          fn, temperature=T)
    da.run_md(n_steps=800, timestep=0.1, equil_steps=10000, seed=1,
              verbose=False)

    msd_result = da.compute_msd()
    assert np.all(np.isfinite(msd_result["msd_total"]))
    assert np.all(msd_result["msd_total"] >= 0.0)
    assert set(msd_result["msd_per_species"].keys()) == {"Al"}

    vacf_result = da.compute_vacf_dos()
    assert np.all(np.isfinite(vacf_result["vacf"]))
    assert np.all(np.isfinite(vacf_result["dos"]))
    assert np.all(vacf_result["dos"] >= 0.0)

    mass_al = da.masses[0]
    vacf0_expected = K_B * T / mass_al * EV_ANG_TO_AMU_ANG_FS2
    assert vacf_result["vacf"][0] == pytest.approx(vacf0_expected, rel=0.15), (
        "VACF(0) = <v_x^2> must match the equipartition prediction "
        "k_B T / m (within statistical noise from a short trajectory)")


def test_msd_einstein_and_green_kubo_agree_for_synthetic_random_walk():
    """Bypasses run_md() entirely, directly injecting a synthetic i.i.d.
    Gaussian random-walk trajectory with an EXACTLY known diffusion
    coefficient, to isolate and verify compute_msd()/compute_vacf_dos()
    against ground truth rather than against an uncontrolled physical
    simulation.

    Derivation
    ----------
    Let each recorded frame's per-atom, per-component displacement step
    be i.i.d. ~ N(0, sigma^2). After `lag` steps,
    MSD(lag) = <|sum of `lag` iid steps|^2> = 3 sigma^2 lag (3 the
    number of Cartesian components), so MSD(t) = 3 sigma^2 (t/dt) is
    EXACTLY linear in t with slope 3 sigma^2 / dt. Matching this to the
    Einstein relation MSD(t) -> 6 D t gives D_theory = sigma^2/(2 dt).

    Constructing the matching velocity series as vel_k = step_k / dt
    (consistent with pos_k = pos_{k-1} + dt . vel_k, exactly how
    run_md()'s own position update works) makes vel_k i.i.d. -- so its
    VACF is (in the infinite-sample limit) a single nonzero spike
    VACF(0) = Var(step)/dt^2 = sigma^2/dt^2, and zero at every nonzero
    lag. The trapezoidal Green-Kubo integral of this one-sided spike is
    dt/2 . VACF(0) = sigma^2/(2 dt) -- the SAME D_theory, independently.

    Bug-fix note (max_lag vs. n_frames, found while writing this test):
    the FIRST version of this test used max_lag = n_frames/4 (1000 out
    of 4000), which gave a Green-Kubo D low by a factor of ~2 --
    confirmed (by scanning several (n_frames, max_lag) combinations) to
    be a genuine finite-sample statistical bias from _autocorr_fft's
    sample-mean subtraction, NOT an implementation bug: for i.i.d. data,
    subtracting the SAMPLE mean (rather than the unknown true mean)
    forces sum_{tau=0}^{n-1} C(tau) (with the tau=0 term counted once,
    lags weighted per the standard convention) to be exactly 0 -- a
    small negative bias gets distributed across every nonzero lag to
    exactly cancel C(0), and that bias becomes non-negligible for the
    *truncated* [0, max_lag) sum used here once max_lag is a
    significant fraction of n_frames. The fix is not a code change (the
    underlying computation is correct) but using a much smaller
    max_lag/n_frames ratio (a well-known best practice for correlation-
    function estimation in general, and already precedented in this
    project by ThermalConductivityCalculator.compute_kappa's own
    n_lag-vs-n_blocks handling, S18) -- see compute_vacf_dos's own
    docstring note added alongside this fix.
    """
    rng = np.random.default_rng(0)
    n_atoms  = 150
    n_frames = 10_000
    dt       = 0.5   # [fs]
    sigma    = 0.05  # [A] per-component step std-dev

    steps = rng.normal(0.0, sigma, size=(n_frames, n_atoms, 3))
    pos_series = np.cumsum(steps, axis=0)
    vel_series = steps / dt

    D_theory = sigma ** 2 / (2.0 * dt)

    fn = make_force_fn(PotentialEngine().add(
        LennardJonesPotential({frozenset({"Al"}): {"epsilon": 0.4, "sigma": 2.5}}, r_cut=3.0)))
    al = _fcc_al_supercell(n=2)
    da = DiffusionAnalyzer(al["lattice"], al["positions"][:1], ["Al"], fn)
    da._pos_series = pos_series
    da._vel_series = vel_series
    da._dt_record  = dt
    da.atom_types  = ["Al"] * n_atoms   # match the synthetic n_atoms, not the real structure

    # max_lag deliberately kept a small fraction (1%) of n_frames -- see
    # the bug-fix note above and compute_vacf_dos's own docstring.
    max_lag = 100
    msd_result = da.compute_msd(max_lag=max_lag)
    assert msd_result["D_einstein"] == pytest.approx(D_theory, rel=0.05), (
        f"Einstein-relation D from a known random walk must match "
        f"theory: got {msd_result['D_einstein']:.6f}, expected {D_theory:.6f}")

    # Direct check that MSD actually grows LINEARLY (not, e.g., as a
    # plateauing or superlinear curve): doubling the lag must
    # (statistically) double the MSD.
    msd = msd_result["msd_total"]
    ratio = msd[90] / msd[45]
    assert ratio == pytest.approx(2.0, rel=0.1), (
        f"MSD must grow linearly in time for a diffusive random walk; "
        f"MSD(90)/MSD(45) = {ratio:.3f}, expected ~2.0")

    vacf_result = da.compute_vacf_dos(max_lag=max_lag)
    assert vacf_result["D_green_kubo"] == pytest.approx(D_theory, rel=0.05), (
        f"Green-Kubo D from the same known random walk's velocities "
        f"must match theory: got {vacf_result['D_green_kubo']:.6f}, "
        f"expected {D_theory:.6f}")

    # The actual cross-check the plan asked for: two independent
    # methods (MSD slope vs. VACF integral) on the SAME trajectory must
    # agree with each other, not just with theory separately.
    assert msd_result["D_einstein"] == pytest.approx(
        vacf_result["D_green_kubo"], rel=0.1), (
        "Einstein-relation and Green-Kubo diffusion coefficients must "
        "agree with each other for the same trajectory")


def test_two_instances_get_distinct_scratch_dirs():
    """Code-check follow-up regression test: DiffusionAnalyzer used to
    hand every instance's internal NVT equilibration run the SAME
    hardcoded output_dir ('/tmp/akitsushima_diffusion_nvt_equil'), which
    could interleave/clobber trajectory/thermo log files between two
    instances running concurrently. Each instance must now get its own
    freshly-created, uniquely-named scratch directory instead."""
    import os

    fn = make_force_fn(PotentialEngine().add(
        LennardJonesPotential({frozenset({"Al"}): {"epsilon": 0.4, "sigma": 2.5}},
                              r_cut=3.2, shift=True)))
    al = _fcc_al_supercell(n=2)

    da1 = DiffusionAnalyzer(al["lattice"], al["positions"], al["atom_types"], fn)
    da2 = DiffusionAnalyzer(al["lattice"], al["positions"], al["atom_types"], fn)

    assert da1._scratch_dir != da2._scratch_dir
    assert os.path.isdir(da1._scratch_dir)
    assert os.path.isdir(da2._scratch_dir)
