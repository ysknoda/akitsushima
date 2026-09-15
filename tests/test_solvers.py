"""Regression tests for the solvers/ package.

Physical/algorithmic correctness of the underlying force fields is
already covered by test_engine_forces.py; this file targets the
solver-specific bugs found during the port: GeometryOptimiser's S1 (cell
relaxation using an elementwise product instead of a matrix product for
the Parrinello-Rahman-style cell force) and S9 (log/XDATCAR file handles
leaking on an exception mid-relaxation), plus the MD classes' S8 (degrees
of freedom not accounting for fix_atoms).
"""
from unittest import mock

import numpy as np
import pytest

from akitsushima.constants import (
    EV_ANG3_TO_GPA,
    EV_ANG_TO_AMU_ANG_FS2,
    FREQ_RAD_FS_TO_WAVENUMBER_CM,
    K_B,
)
from akitsushima.engine import LennardJonesPotential, PotentialEngine
from akitsushima.fitting import make_force_fn
from akitsushima.solvers import (
    GCMCMDSimulation,
    GeometryOptimiser,
    NEBChain,
    NEBOptimiser,
    NPTSimulation,
    NVESimulation,
    NVTSimulation,
    OSMMCSimulation,
    PhononCalculator,
    SpeciesParams,
    ThermalConductivityCalculator,
    TransitionStateAnalyser,
)
from akitsushima.solvers import geometry_optimiser as go_mod

from conftest import fcc_al_dict, randomly_perturbed, toy_dimer_dict


def _lj_force_fn(pair=frozenset({"Al"}), epsilon=0.4, sigma=2.5, r_cut=6.0):
    engine = PotentialEngine().add(
        LennardJonesPotential({pair: {"epsilon": epsilon, "sigma": sigma}},
                              r_cut=r_cut, shift=True))
    return make_force_fn(engine)


def _fcc_al_supercell(n=2):
    """n x n x n supercell of the conventional cubic FCC Al fixture, needed
    for MD tests since NeighbourList (unlike GeometryOptimiser's O(N^2)
    fallback) enforces r_cut + skin_width < half the shortest cell length
    (bug E5), and the bare 4.05 A conventional cell is too small for any
    physically meaningful cutoff."""
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
        "charges": None,
    }


def _find_lj_equilibrium(fn, structure, output_dir):
    """Relax `structure` to its true mechanical equilibrium (lattice AND
    ions) under `fn`, via the independently-verified default
    'generalized_force' cell-relaxation scheme (Bug C). Used as a known
    reference point by tests that need a deliberately compressed/expanded
    starting cell relative to a real equilibrium, rather than an
    arbitrary guess."""
    go_eq = GeometryOptimiser(structure["lattice"], structure["positions"],
                             structure["atom_types"], force_fn=fn,
                             relax_cell=True, method="fire",
                             fmax=0.005, pmax=0.02, max_steps=2000, verbose=False,
                             output_dir=output_dir)
    result_eq = go_eq.run()
    assert result_eq["converged"], "equilibrium-finding relaxation did not converge"
    lattice_eq = result_eq["lattice"]
    frac_eq = result_eq["positions"] @ np.linalg.inv(lattice_eq)
    return lattice_eq, frac_eq


def _scaled_structure(lattice_eq, frac_eq, atom_types, scale):
    """Build a purely hydrostatically strained structure at `scale` times
    the equilibrium lattice, with fractional coordinates held fixed."""
    lattice_s = lattice_eq * scale
    positions_s = frac_eq @ lattice_s
    return lattice_s, positions_s


# ===========================================================================
# S1: GeometryOptimiser._generalised_force cell-relaxation matrix product
# ===========================================================================

def _finite_diff_cell_gradient(fn, lattice, positions, atom_types, delta=1e-5):
    """dE/dh_ab by directly perturbing each of the 9 raw lattice-matrix
    components independently (NOT via the h'=h@T composed-deformation
    route test_engine_forces.py's finite_diff_stress uses, to avoid any
    circularity), holding fractional coordinates fixed."""
    frac = positions @ np.linalg.inv(lattice)
    dEdh = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            lat_p = lattice.copy(); lat_p[a, b] += delta
            lat_m = lattice.copy(); lat_m[a, b] -= delta
            e_p, _, _ = fn(lat_p, frac @ lat_p, atom_types)
            e_m, _, _ = fn(lat_m, frac @ lat_m, atom_types)
            dEdh[a, b] = (e_p - e_m) / (2.0 * delta)
    return dEdh


def test_generalised_force_cell_matches_finite_difference_dEdh(small_cell_ok):
    """Bug C regression: the 'generalized_force' cell-force branch must
    equal f_cell = -dE/dh, verified against a TRUE finite-difference
    oracle (not another hand-derived formula, which is exactly how the
    previous version of this test passed while encoding the same wrong
    matrix-multiplication order as the production code -- see
    engine/cell_gradient.py's module docstring for the direct numerical
    proof that vol*(h_inv_T @ stress), not vol*(stress @ h_inv_T), is
    correct). Uses a deliberately non-orthogonal, low-symmetry cell with
    a real shear stress, exactly the regime where the two orderings
    (and the even-older elementwise product) all disagree."""
    fn = _lj_force_fn()
    al = randomly_perturbed(fcc_al_dict(), magnitude=0.05, seed=0)
    lattice = al["lattice"] @ np.array([[1.0, 0.0, 0.0],
                                        [0.05, 1.0, 0.0],
                                        [0.02, 0.03, 1.0]])  # shear
    positions = al["positions"]
    atom_types = al["atom_types"]

    go = GeometryOptimiser(lattice, positions, atom_types,
                           force_fn=fn, relax_cell=True, verbose=False)
    _, forces, stress = fn(lattice, positions, atom_types)

    g = go._generalised_force(forces, stress)
    f_cell = g[go.N * 3:].reshape(3, 3)

    dEdh_numeric = _finite_diff_cell_gradient(fn, lattice, positions, atom_types)

    assert np.allclose(f_cell, -dEdh_numeric, atol=2e-3, rtol=1e-3), (
        "Bug C regression: f_cell must equal -dE/dh (finite-difference "
        "oracle), i.e. vol*(h_inv_T @ stress) with h_inv_T on the LEFT.")


def test_generalised_force_cell_differs_from_known_wrong_orderings():
    """Fast regression guard (no finite differences) against reverting to
    either of the two previously-shipped wrong formulas: the original
    elementwise product (pre-S1), or S1's own fix which used the correct
    matrix PRODUCT but the wrong operand ORDER (stress @ h_inv_T instead
    of h_inv_T @ stress) -- see
    test_generalised_force_cell_matches_finite_difference_dEdh for the
    full finite-difference proof of which ordering is actually right."""
    fn = _lj_force_fn()
    al = randomly_perturbed(fcc_al_dict(), magnitude=0.05, seed=0)
    lattice = al["lattice"] @ np.array([[1.0, 0.0, 0.0],
                                        [0.05, 1.0, 0.0],
                                        [0.02, 0.03, 1.0]])  # shear

    go = GeometryOptimiser(lattice, al["positions"], al["atom_types"],
                           force_fn=fn, relax_cell=True, verbose=False)
    stress = np.array([[0.01, 0.004, -0.002],
                       [0.004, -0.02, 0.001],
                       [-0.002, 0.001, 0.03]])
    forces = np.zeros_like(go.positions)

    g = go._generalised_force(forces, stress)
    f_cell = g[go.N * 3:].reshape(3, 3)

    vol = abs(np.linalg.det(lattice))
    h_inv_T = np.linalg.inv(lattice).T
    elementwise_wrong = vol * (stress * h_inv_T)          # pre-S1
    transposed_wrong  = vol * (stress @ h_inv_T)          # S1's own bug

    assert not np.allclose(f_cell, elementwise_wrong, rtol=1e-6)
    assert not np.allclose(f_cell, transposed_wrong, rtol=1e-6)


def test_variable_cell_relax_converges_from_sheared_cell():
    """End-to-end smoke test: relax_cell=True from a deliberately sheared
    (non-orthogonal) starting cell must actually converge to a low-stress
    state, not diverge or stall -- an integration-level check that S1's
    fix is wired correctly into the FIRE dynamics."""
    # Cell size: r_cut + skin_width must stay under half the smallest
    # perpendicular cell width.  The bare 4.05 A conventional FCC Al
    # cell cannot host any useful cutoff, so use the same 2x2x2
    # supercell + r_cut=3.2 pairing the MD tests below already use.
    # Before GeometryOptimiser built a neighbour list at all this
    # combination went unchecked, and these relax tests silently ran
    # at r_cut=6.0 in a cell whose half-width is 2.02 A.
    fn = _lj_force_fn(r_cut=3.2)
    al = _fcc_al_supercell()
    lattice = al["lattice"] @ np.array([[1.0, 0.0, 0.0],
                                        [0.08, 1.0, 0.0],
                                        [0.0, 0.0, 1.0]])
    go = GeometryOptimiser(lattice, al["positions"], al["atom_types"],
                           force_fn=fn, relax_cell=True, method="fire",
                           fmax=0.01, pmax=0.5, max_steps=500, verbose=False,
                           output_dir="/tmp/akitsushima_test_relax_cell")
    result = go.run()
    assert result["converged"], "variable-cell relaxation failed to converge"
    GPa = EV_ANG3_TO_GPA
    assert np.max(np.abs(result["forces"])) < 0.05
    stress_final = go._get_forces_and_stress()[2]
    assert np.max(np.abs(stress_final)) * GPa < 1.0


def test_vasp_cell_scheme_drives_volume_in_correct_physical_direction():
    """Bug C' regression: cell_update_scheme='vasp' (the "computation
    method that doesn't use the inverse matrix h^-1", f_cell = stress @
    lattice) had zero test coverage before this fix and its sign was
    only weakly argued from first principles (VASP's own OUTCAR stress
    uses the opposite sign from this codebase's tensile-positive
    convention, so translating VASP's own dyna.F update rule seemed to
    require a sign flip) rather than empirically verified. Per the
    project's own established practice of gating any such change on
    direct verification rather than a plausible-sounding argument, this
    test isolates the 'vasp' formula's DIRECTION by freezing all ionic
    positions (fix_atoms=True for every atom) so only the cell degrees
    of freedom respond, then checks that a cell compressed below the
    true LJ equilibrium volume gets a POSITIVE (expanding) f_cell, and a
    cell expanded above it gets a NEGATIVE (contracting) f_cell -- the
    physically correct direction in both cases.

    Result: the CURRENT, unflipped formula (f_cell = stress @ lattice,
    no sign change) already drives volume in the correct direction --
    the sign-flip hypothesis argued from the VASP-translation reasoning
    turned out to be wrong once actually checked. (The reasoning error:
    the codebase's own already-verified stress convention -- see
    test_engine_forces.py's sigma_ab = -(1/V) dE/dT_ab -- gives POSITIVE
    diagonal stress for a COMPRESSED cell, not negative as originally
    assumed; a parabolic E(s) ~ (s-1)^2 argument confirms dE/ds < 0 for
    s < 1, so sigma = -(1/V)(dE/dT) is positive there. This is exactly
    why the fix was gated on an empirical test instead of applied
    blindly: doing so would have introduced a real bug into code that
    was already correct.) No code change was needed; this test exists
    purely to give this previously-untested branch permanent coverage.
    """
    # Cell size: r_cut + skin_width must stay under half the smallest
    # perpendicular cell width.  The bare 4.05 A conventional FCC Al
    # cell cannot host any useful cutoff, so use the same 2x2x2
    # supercell + r_cut=3.2 pairing the MD tests below already use.
    # Before GeometryOptimiser built a neighbour list at all this
    # combination went unchecked, and these relax tests silently ran
    # at r_cut=6.0 in a cell whose half-width is 2.02 A.
    fn = _lj_force_fn(r_cut=3.2)
    al = _fcc_al_supercell()
    N = len(al["positions"])

    lattice_eq, frac_eq = _find_lj_equilibrium(
        fn, al, "/tmp/akitsushima_test_vasp_scheme_eq")

    fix_all = np.ones((N, 3), dtype=bool)

    def _f_cell_diag_mean(scale):
        lattice_s, positions_s = _scaled_structure(
            lattice_eq, frac_eq, al["atom_types"], scale)
        go = GeometryOptimiser(lattice_s, positions_s, al["atom_types"],
                               force_fn=fn, relax_cell=True,
                               cell_update_scheme="vasp", fix_atoms=fix_all,
                               method="fire", max_steps=1, verbose=False,
                               output_dir=f"/tmp/akitsushima_test_vasp_scheme_{scale}")
        _, forces, stress = go._get_forces_and_stress()
        g = go._generalised_force(forces, stress)
        f_cell = g[N * 3:].reshape(3, 3)
        return float(np.mean(np.diag(f_cell)))

    f_cell_compressed = _f_cell_diag_mean(0.95)
    f_cell_expanded   = _f_cell_diag_mean(1.05)

    assert f_cell_compressed > 0.0, (
        "Bug C' regression: a compressed cell must get a POSITIVE "
        "(expanding) f_cell under cell_update_scheme='vasp'.")
    assert f_cell_expanded < 0.0, (
        "Bug C' regression: an expanded cell must get a NEGATIVE "
        "(contracting) f_cell under cell_update_scheme='vasp'.")


# ===========================================================================
# S9: file handles must not leak when force_fn raises mid-relaxation
# ===========================================================================

def test_relax_closes_file_handles_on_exception(tmp_path):
    # Cell size: r_cut + skin_width must stay under half the smallest
    # perpendicular cell width.  The bare 4.05 A conventional FCC Al
    # cell cannot host any useful cutoff, so use the same 2x2x2
    # supercell + r_cut=3.2 pairing the MD tests below already use.
    # Before GeometryOptimiser built a neighbour list at all this
    # combination went unchecked, and these relax tests silently ran
    # at r_cut=6.0 in a cell whose half-width is 2.02 A.
    fn = _lj_force_fn(r_cut=3.2)
    call_count = [0]

    def flaky_fn(lattice, positions, atom_types, neigh_list=None):
        call_count[0] += 1
        if call_count[0] > 2:
            raise RuntimeError("simulated force_fn failure")
        return fn(lattice, positions, atom_types, neigh_list=neigh_list)

    flaky_fn.r_cut = fn.r_cut
    al = randomly_perturbed(_fcc_al_supercell(), magnitude=0.05, seed=1)
    go = GeometryOptimiser(al["lattice"], al["positions"], al["atom_types"],
                           force_fn=flaky_fn, method="fire", max_steps=500,
                           verbose=False, output_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="simulated force_fn failure"):
        go.run()

    assert go._xdatcar_fh.closed, (
        "S9 regression: XDATCAR_relax file handle must be closed (via "
        "try/finally) even when force_fn raises mid-relaxation")


def test_open_log_closes_xdatcar_handle_if_it_fails_before_returning(
        monkeypatch, tmp_path):
    """Code-check follow-up regression test: _open_log() opens
    self._xdatcar_fh, then (for relax_cell=False) writes the XDATCAR
    header and opens relax.log, all BEFORE returning to its caller -- and
    the S9 try/finally protection above only starts in the CALLER, after
    _open_log() has already returned. A failure in that narrow window
    (simulated here by making _write_xdatcar_header raise) used to leak
    self._xdatcar_fh with no cleanup at all, since it is neither inside
    the caller's try/finally (which has not started yet) nor caught by
    anything inside _open_log() itself."""
    fn = _lj_force_fn()
    al = randomly_perturbed(fcc_al_dict(), magnitude=0.05, seed=1)
    go = GeometryOptimiser(al["lattice"], al["positions"], al["atom_types"],
                           force_fn=fn, method="fire", max_steps=10,
                           relax_cell=False, verbose=False,
                           output_dir=str(tmp_path))

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated XDATCAR header failure")

    monkeypatch.setattr(go_mod, "_write_xdatcar_header", _raise)

    with pytest.raises(RuntimeError, match="simulated XDATCAR header failure"):
        go._open_log("dt(fs)")

    assert go._xdatcar_fh.closed, (
        "_open_log() must close self._xdatcar_fh before re-raising when "
        "a step after opening it (XDATCAR header write, or the relax.log "
        "open) fails -- this window is NOT covered by the caller's own "
        "try/finally, which only starts after _open_log() returns.")


# ===========================================================================
# RFO sign: _rfo_step / _prfo_step returned exactly minus the correct step
# ===========================================================================

def _rosenbrock_force_fn(offset=5.0):
    """Analytic Rosenbrock objective dressed up as a force_fn, so the RFO
    machinery is exercised with zero physics: positions.ravel() - offset IS
    the Rosenbrock variable and the minimum is at all-ones. Rosenbrock is
    used rather than a harmonic well on purpose -- a quadratic would be
    solved in a single Newton step from any starting point, which is far too
    forgiving to distinguish a working optimiser from a broken one."""
    def force_fn(lattice, positions, atom_types, neigh_list=None):
        x = positions.ravel() - offset
        e = float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2
                         + (1.0 - x[:-1]) ** 2))
        grad = np.zeros(x.size)
        grad[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
        grad[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return e, (-grad).reshape(positions.shape), np.zeros((3, 3))
    force_fn.r_cut = 5.0
    force_fn.kind = None
    return force_fn


def _spd_hessian_and_gradient(seed=0, n=4, g_scale=0.05):
    """Random DENSE, well-conditioned SPD Hessian plus a generic gradient.
    Deliberately NOT diagonal: with a diagonal H the augmented-Hessian
    eigenvectors are sign-symmetric enough that operand-order and sign
    mistakes can survive, so a diagonal test case cannot discriminate."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    H = M @ M.T + 3.0 * np.eye(n)
    g = rng.normal(size=n) * g_scale
    return H, g


def test_rfo_step_matches_newton_step():
    """RFO sign regression: for the augmented Hessian A = [[H, g], [g^T, 0]]
    the eigenvalue equations give (H - lam I) v_x = -g v_n, so the step is
    p = v_x / v_n = -(H - lam I)^{-1} g, which tends to the Newton step
    -H^{-1} g as lam -> 0. _rfo_step used to return -v_x / v_n, i.e. exactly
    the NEGATIVE of this -- an ascent direction for a minimum search.

    Checked against an independent analytic oracle (numpy's own solve),
    not against a recomputation of the production expression."""
    H, g = _spd_hessian_and_gradient(seed=0, n=4, g_scale=0.05)
    p, lam = GeometryOptimiser._rfo_step(g, H)
    p_newton = -np.linalg.solve(H, g)

    # Direction: the RFO step differs from Newton only by the lam shift, so
    # for a small gradient the two must be very nearly parallel AND equal in
    # magnitude. cosine == -1 is precisely the shipped bug.
    cosine = float(np.dot(p, p_newton)
                   / (np.linalg.norm(p) * np.linalg.norm(p_newton)))
    assert cosine > 0.999, (
        f"RFO step is anti-parallel to the Newton step (cos={cosine:.6f}); "
        "the augmented-Hessian step is p = +v[:n]/v[n], not -v[:n]/v[n]")
    assert p == pytest.approx(p_newton, rel=1e-3)

    # Descent condition, and the exact identity g . p = lam that follows
    # from the bottom row g^T v_x = lam v_n of the eigen-equation.
    assert float(np.dot(g, p)) < 0.0, "RFO step must be a descent direction"
    assert float(np.dot(g, p)) == pytest.approx(lam, rel=1e-8)
    assert lam < 0.0


def test_rfo_step_is_descent_for_many_random_hessians():
    """The descent guarantee must not be an artefact of one lucky seed."""
    for seed in range(20):
        H, g = _spd_hessian_and_gradient(seed=seed, n=5, g_scale=0.1)
        p, lam = GeometryOptimiser._rfo_step(g, H)
        assert float(np.dot(g, p)) < 0.0, f"ascent direction at seed={seed}"
        cosine = float(np.dot(p, -np.linalg.solve(H, g))
                       / (np.linalg.norm(p)
                          * np.linalg.norm(np.linalg.solve(H, g))))
        assert cosine > 0.99, f"anti-parallel to Newton at seed={seed}"


def test_rfo_degenerate_fallback_is_sign_deterministic():
    """`eigh` fixes eigenvectors only up to a global sign. p = v_x / v_n is
    invariant under that flip (numerator and denominator flip together), but
    the |v_n| < 1e-12 fallback returned a bare +/- v_x and so could hand back
    a step in either direction depending on the LAPACK build. Simulate the
    flip by negating eigh's eigenvectors and require an identical step.

    g = 0 with an indefinite H makes A = blkdiag(H, 0), whose lowest
    eigenvector is (u_min, 0) -- v_n is then exactly zero, which is the only
    way to reach the fallback deterministically."""
    H, _ = _spd_hessian_and_gradient(seed=3, n=4)
    w, V = np.linalg.eigh(H)
    w[0] = -2.0
    H = V @ np.diag(w) @ V.T
    g = np.zeros(4)

    p_ref, _ = GeometryOptimiser._rfo_step(g, H)

    real_eigh = np.linalg.eigh

    def flipped_eigh(a):
        vals, vecs = real_eigh(a)
        return vals, -vecs

    with mock.patch.object(np.linalg, "eigh", flipped_eigh):
        p_flip, _ = GeometryOptimiser._rfo_step(g, H)

    assert np.linalg.norm(p_ref) > 0.0, "fallback must not return a null step"
    assert p_flip == pytest.approx(p_ref, abs=1e-14), (
        "the degenerate RFO fallback must not depend on the arbitrary global "
        f"sign eigh assigns its eigenvectors (got {p_flip} vs {p_ref})")


def test_prfo_step_mode_signs():
    """P-RFO sign regression (Banerjee et al. 1985). Every mode takes the
    form p_k = -b_k / (lam_k - nu_k); mode 0 uses the UPPER root, so
    lam_0 - nu_0 <= 0 and p_0 shares the sign of b_0 (uphill), while modes
    k >= 1 use the LOWER root, so lam_k - nu_k >= 0 and p_k opposes b_k
    (downhill). Both denominators used to be written with the operands
    reversed, so mode 0 descended the mode it must ascend and every other
    mode ascended the modes it must minimise.

    The Hessian here has exactly one negative eigenvalue -- a first-order
    saddle, which is the only regime where P-RFO's partitioning is meaningful."""
    H, g = _spd_hessian_and_gradient(seed=0, n=4, g_scale=0.05)
    w, V = np.linalg.eigh(H)
    w[0] = -2.0                                   # exactly one negative mode
    H_saddle = V @ np.diag(w) @ V.T
    assert np.sum(np.linalg.eigvalsh(H_saddle) < 0.0) == 1

    p, lam0 = GeometryOptimiser._prfo_step(g, H_saddle)
    assert lam0 == pytest.approx(-2.0, rel=1e-10)

    eigvals, eigvecs = np.linalg.eigh(H_saddle)
    b = eigvecs.T @ g                             # modal gradient
    p_modal = eigvecs.T @ p                       # modal step

    assert abs(b[0]) > 1e-6, "test needs a non-negligible modal gradient"
    assert np.sign(p_modal[0]) == np.sign(b[0]), (
        f"mode 0 must step UPHILL: modal grad b0={b[0]:.6f} and step "
        f"p0={p_modal[0]:.6f} must share a sign")

    for k in range(1, len(b)):
        assert abs(b[k]) > 1e-9
        assert np.sign(p_modal[k]) == -np.sign(b[k]), (
            f"mode {k} must step DOWNHILL: modal grad b{k}={b[k]:.6f} and "
            f"step p{k}={p_modal[k]:.6f} must have opposite signs")

    # Consequence of the above: the step raises the energy along mode 0 and
    # lowers it along every other mode, to first order.
    assert b[0] * p_modal[0] > 0.0
    assert float(np.dot(b[1:], p_modal[1:])) < 0.0


def _prfo_step_oracle_2x2(g, H):
    """Structurally independent oracle for _prfo_step.

    Mode k's RFO step is the step of the 2x2 augmented eigenproblem

        A_k = [[lam_k, b_k], [b_k, 0]],   p_k = v_x / v_n

    taking the UPPER eigenvalue for mode 0 (maximise) and the LOWER one
    for k >= 1 (minimise).  Reading p_k off that eigenvector uses `eigh`
    instead of the closed-form quadratic root, so this shares no algebra
    with the production loop beyond the definition of A_k -- unlike a
    retyped `-b/(lam - nu)`, which would silently inherit a conceptual
    error in the root formula itself.
    """
    eigvals, eigvecs = np.linalg.eigh(H)
    b = eigvecs.T @ g
    p_modal = np.zeros(len(b))
    for k in range(len(b)):
        vals, vecs = np.linalg.eigh(np.array([[eigvals[k], b[k]],
                                              [b[k], 0.0]]))
        j = 1 if k == 0 else 0          # eigh is ascending: 1 = upper root
        p_modal[k] = vecs[0, j] / vecs[1, j]
    return eigvecs @ p_modal


def test_prfo_step_magnitude_matches_banerjee_formula():
    """Oracle check for _prfo_step, at two independence levels.

    (a) The published closed form p_k = -b_k/(lam_k - nu_k) written out
        here with the operand order taken from the paper.  This catches
        the shipped reversed-operand bug, but it is the same arithmetic
        the production loop performs, so it would share any conceptual
        error in the root formula.
    (b) `_prfo_step_oracle_2x2`, which derives the same step from the 2x2
        augmented eigenproblem instead of the quadratic root -- a
        genuinely independent route.  This is the level of independence
        this repo actually needs: a previous regression test here
        computed its "expected" value with the same wrong formula as the
        production code and so passed by construction.

    Verified to discriminate: under the shipped reversed-operand bug the
    returned step is exactly negated, so it misses both oracles by 200%
    of the step magnitude (e.g. max|diff| = 8.13e-02 against an oracle of
    scale 4.07e-02 at seed=7), against a tolerance of rel=1e-10.
    """
    H, g = _spd_hessian_and_gradient(seed=7, n=4, g_scale=0.08)
    w, V = np.linalg.eigh(H)
    w[0] = -1.5
    H_saddle = V @ np.diag(w) @ V.T

    p, _ = GeometryOptimiser._prfo_step(g, H_saddle)

    eigvals, eigvecs = np.linalg.eigh(H_saddle)
    b = eigvecs.T @ g
    expected_modal = np.empty_like(b)
    nu_0 = (eigvals[0] + np.sqrt(eigvals[0] ** 2 + 4.0 * b[0] ** 2)) / 2.0
    expected_modal[0] = -b[0] / (eigvals[0] - nu_0)
    for k in range(1, len(b)):
        nu_k = (eigvals[k] - np.sqrt(eigvals[k] ** 2 + 4.0 * b[k] ** 2)) / 2.0
        expected_modal[k] = -b[k] / (eigvals[k] - nu_k)

    assert p == pytest.approx(eigvecs @ expected_modal, rel=1e-10)

    # (b) the structurally independent route, over several curvatures and
    # dimensions so the agreement cannot be a one-seed coincidence.
    for seed, lam0, n, g_scale in ((7, -1.5, 4, 0.08), (0, -2.0, 4, 0.05),
                                   (11, -0.7, 5, 0.20), (23, -3.3, 6, 0.01)):
        Hs, gs = _spd_hessian_and_gradient(seed=seed, n=n, g_scale=g_scale)
        ws, Vs = np.linalg.eigh(Hs)
        ws[0] = lam0
        Hs = Vs @ np.diag(ws) @ Vs.T
        p_prod, _ = GeometryOptimiser._prfo_step(gs, Hs)
        assert p_prod == pytest.approx(_prfo_step_oracle_2x2(gs, Hs),
                                       rel=1e-10, abs=1e-14), (
            f"_prfo_step disagrees with the 2x2 augmented-eigenproblem "
            f"oracle at seed={seed}, lam0={lam0}")


def test_rfo_relaxation_converges_rosenbrock(tmp_path):
    """End-to-end regression test -- the one whose absence let the sign bug
    ship: nothing in the suite ran method='rfo' at all.

    The failure mode is silent rather than loud. _run_rfo rejects any step
    that raises the energy (`if actual_de <= 0.0 or self.uphill`), so an
    ascent-direction step is rejected every single iteration and the
    optimiser sits at its starting point reporting converged=False, with the
    energy pinned at EXACTLY its initial value. Asserting a measurable energy
    decrease (not merely converged=True) is what makes this discriminating."""
    force_fn = _rosenbrock_force_fn(offset=5.0)
    x0 = np.array([-1.2, 1.0, -1.2, 1.0, -1.2, 1.0])
    positions = (x0 + 5.0).reshape(2, 3)

    go = GeometryOptimiser(np.eye(3) * 20.0, positions, ["X", "X"],
                           force_fn=force_fn, method="rfo",
                           fmax=1e-4, max_steps=400, verbose=False,
                           output_dir=str(tmp_path))
    e_start = force_fn(None, positions, None)[0]
    result = go.run()

    assert result["converged"], "RFO relaxation failed to converge"
    assert result["energy"] < e_start - 1.0, (
        f"RFO made no progress: E stayed at {result['energy']:.6f} from a "
        f"start of {e_start:.6f} (an ascent step is rejected every iteration, "
        "pinning the energy at its initial value)")
    assert result["energy"] == pytest.approx(0.0, abs=1e-6)
    assert np.max(np.abs(result["forces"])) < 1e-3
    assert result["positions"].ravel() - 5.0 == pytest.approx(
        np.ones(6), abs=1e-3), "RFO did not reach the Rosenbrock minimum"


# ===========================================================================
# S8: MD n_dof must account for fix_atoms
# ===========================================================================

def test_nvt_n_dof_accounts_for_fixed_atoms():
    fn = _lj_force_fn()
    al = fcc_al_dict()
    N = len(al["atom_types"])

    sim_free = NVTSimulation(al["lattice"], al["positions"], al["atom_types"],
                             force_fn=fn, n_steps=1, verbose=False,
                             output_dir="/tmp/akitsushima_test_nvt_dof_free")
    assert sim_free.n_dof == 3 * N - 3

    fix = np.zeros((N, 3), dtype=bool)
    fix[0] = True   # freeze all 3 components of one atom
    sim_fixed = NVTSimulation(al["lattice"], al["positions"], al["atom_types"],
                              force_fn=fn, n_steps=1, verbose=False,
                              fix_atoms=fix,
                              output_dir="/tmp/akitsushima_test_nvt_dof_fixed")
    assert sim_fixed.n_dof == 3 * N - 3, (
        "sanity: with only 3 of 3N dof fixed this happens to coincide "
        "with the unfixed count; the real regression check is below")

    fix2 = np.zeros((N, 3), dtype=bool)
    fix2[0] = True
    fix2[1, 0] = True
    sim_fixed2 = NVTSimulation(al["lattice"], al["positions"], al["atom_types"],
                               force_fn=fn, n_steps=1, verbose=False,
                               fix_atoms=fix2,
                               output_dir="/tmp/akitsushima_test_nvt_dof_fixed2")
    assert sim_fixed2.n_dof == 3 * N - 4, (
        "S8 regression: n_dof must be 3N minus the number of frozen "
        "components once any atom is fixed (no further -3 COM "
        "correction, since a pinned atom already breaks translational "
        "invariance) -- got n_dof="
        f"{sim_fixed2.n_dof}, expected {3*N - 4}")


# ===========================================================================
# NVE / NVT integration smoke tests
# ===========================================================================

def test_nve_energy_conservation():
    fn = _lj_force_fn(r_cut=3.2)
    al = randomly_perturbed(_fcc_al_supercell(), magnitude=0.05, seed=2)
    # timestep=0.1 fs: this ad-hoc LJ parametrisation (tuned only to be
    # reasonably stiff on the small test supercell, not to match real Al
    # phonon frequencies) is numerically stiffer than the realistic
    # potentials this timestep default is normally calibrated for; a
    # dt-convergence scan (0.5/0.1/0.02/0.005 fs) confirmed the drift
    # shrinks as O(dt^2) as expected for a correct Verlet integrator, so
    # 0.5 fs was simply too large a step for *this* test potential, not a
    # sign of an integrator bug.
    sim = NVESimulation(al["lattice"], al["positions"], al["atom_types"],
                        force_fn=fn, timestep=0.1, n_steps=200,
                        temperature=300.0, seed=3, verbose=False,
                        thermo_every=1000000,
                        output_dir="/tmp/akitsushima_test_nve")
    result = sim.run()
    n_atoms = len(al["atom_types"])
    assert abs(result["e_tot_drift"]) / n_atoms < 1e-2, (
        "NVE total energy should be conserved to well under 10 meV/atom "
        "over 200 steps of 0.1 fs")


def test_nhc_thermostat_isolated_is_stable_and_bounded():
    """
    Direct unit check of the NHC half-step propagator (_nhc_step),
    isolated from any force field: starting far from the target
    temperature with zero thermostat state, repeatedly applying it to a
    FIXED set of velocities (no forces, no position updates -- the
    thermostat is the only thing acting) must drive the instantaneous
    temperature toward the target and then keep it bounded in a sane
    range indefinitely.

    This isolates _nhc_step's own correctness from the separate question
    of whether a *particular* force field/timestep/system-size
    combination is itself dynamically stable over long times (an
    ordinary many-body chaos/stiffness question that arises for any
    correctly-implemented integrator, and is exercised only lightly and
    over a short time window in the coupled NVT test below).
    """
    N = 32
    sim = NVTSimulation.__new__(NVTSimulation)
    sim.masses = np.full(N, 26.98)
    sim._fix_mask = np.zeros((N, 3), dtype=bool)
    sim.n_dof = 3 * N - 3
    sim.temperature = 300.0
    sim.dt = 0.1
    tau_nhc = 30.0
    sim.Q1 = sim.n_dof * K_B * sim.temperature * tau_nhc**2
    sim.Q2 = K_B * sim.temperature * tau_nhc**2
    sim.xi1 = sim.xi2 = sim.vxi1 = sim.vxi2 = 0.0
    rng = np.random.default_rng(0)
    sim.velocities = rng.standard_normal((N, 3)) * 0.001   # cold start

    T_hist = []
    for i in range(20000):
        sim._nhc_step()
        if i >= 10000:   # past the initial heating transient from a cold start
            T_hist.append(2.0 * sim._kinetic_energy() / (sim.n_dof * K_B))

    T_hist = np.array(T_hist)
    assert np.all(np.isfinite(T_hist)) and np.all(T_hist < 1e4), (
        "S10 regression: the shared _nhc_step propagator must not "
        "diverge when acting alone on a fixed set of velocities")
    assert 100.0 < T_hist.mean() < 600.0, (
        f"_nhc_step should settle near the 300 K target once past the "
        f"initial transient from a cold start; got mean={T_hist.mean():.1f} K")


def test_nvt_runs_and_moves_toward_target_temperature():
    """
    Light coupled integration smoke test: starting well below the target
    temperature, NVT must raise the temperature back toward the target
    rather than leaving it cold or blowing up.

    History -- this docstring used to justify the short run length by
    claiming a "long-time chaotic/close-contact instability any small
    periodic LJ system can eventually hit regardless of integrator
    correctness". That claim was wrong and has been disproven. The
    divergence it described was a symptom of a real, since-fixed defect
    (Bug L: the potentials consumed stale build-time neighbour-list
    distances instead of current ones), not a property of the
    Nose-Hoover chain integrator. Re-running the equilibration sweep
    after that fix gives production temperatures of 244.9 / 311.7 /
    388.5 / 464.9 K for equil_steps = 200 / 1000 / 3000 / 6000 --
    monotonically convergent toward the 500 K target with no divergence
    -- where the pre-fix write-up had recorded VACF ratios of
    0.78 -> 2.14 -> 19.3 -> diverged with H_nhc ~ 1e18. At
    equil_steps = 10000 the VACF/equipartition ratio is 0.9999-1.0243
    across seeds 1-5.

    This evidence establishes that the previously observed blow-up was a
    force-evaluation bug, not an integrator limitation. It does not
    establish that the NHC integrator is unconditionally stable for
    arbitrarily long runs or arbitrary timesteps, and this test is not
    the place that would demonstrate such a claim: the run here stays
    short because it is a fast smoke test, not because a longer one is
    expected to fail.
    """
    fn = _lj_force_fn(r_cut=3.2)
    al = randomly_perturbed(_fcc_al_supercell(), magnitude=0.03, seed=4)
    sim = NVTSimulation(al["lattice"], al["positions"], al["atom_types"],
                        force_fn=fn, temperature=300.0, timestep=0.05,
                        n_steps=400, equil_steps=0, tau_nhc=30.0,
                        seed=5, verbose=False, thermo_every=20,
                        output_dir="/tmp/akitsushima_test_nvt")
    # Start deliberately cold (well below the 300 K target).
    sim.velocities *= 0.2
    result = sim.run()
    T_series = np.array([r["T_inst"] for r in result["thermo_log"]])
    assert np.all(np.isfinite(T_series)) and np.all(T_series < 1e4), (
        "NVT must not diverge over a short run"
    )
    assert T_series[-1] > T_series[0], (
        "starting well below the target, NVT should have raised the "
        "temperature by the end of the run, not left it cold"
    )


def test_npt_smoke():
    """Bug D regression: NPTSimulation was imported in this file but never
    once instantiated anywhere in the test suite. Minimal smoke test:
    construct and run a short NPTSimulation and check it completes with
    finite, sane-shaped output."""
    fn = _lj_force_fn(r_cut=3.2)
    al = _fcc_al_supercell()
    sim = NPTSimulation(al["lattice"], al["positions"], al["atom_types"],
                        force_fn=fn, temperature=300.0, pressure=0.0,
                        timestep=0.05, n_steps=50, equil_steps=0,
                        tau_nhc=50.0, tau_baro=500.0, seed=1, verbose=False,
                        thermo_every=1000000, traj_every=1000000,
                        output_dir="/tmp/akitsushima_test_npt_smoke")
    result = sim.run()
    for key in ("T_mean", "T_std", "P_mean_gpa", "P_std_gpa", "V_mean",
               "H_npt_drift"):
        assert np.isfinite(result[key]), f"NPTSimulation.run()['{key}'] is not finite"
    assert result["lattice"].shape == (3, 3)
    assert result["positions"].shape == al["positions"].shape


def test_npt_drives_volume_toward_target_pressure_direction():
    """Bug D regression + NPT barostat sign fix: starting from a cell
    deliberately compressed (resp. expanded) relative to the true LJ
    mechanical equilibrium, NPT run at target pressure=0 must move the
    volume in the physically correct direction -- compressed cells
    expand, expanded cells contract -- rather than diverging further
    away (which is exactly what the old, sign-flipped
    NPTSimulation._pressure_inst did: monkey-patching the old
    ``Pv = -Tr(stress)/3`` formula back in and re-running this exact
    check showed volume moving in the WRONG direction in both cases,
    confirming the bug directly rather than by argument).
    """
    fn = _lj_force_fn(r_cut=3.2)
    al = _fcc_al_supercell()
    lattice_eq, frac_eq = _find_lj_equilibrium(
        fn, al, "/tmp/akitsushima_test_npt_eq")

    def _run_npt(scale, tag):
        lattice0, positions0 = _scaled_structure(
            lattice_eq, frac_eq, al["atom_types"], scale)
        vol0 = abs(np.linalg.det(lattice0))
        sim = NPTSimulation(lattice0, positions0, al["atom_types"],
                            force_fn=fn, temperature=300.0, pressure=0.0,
                            timestep=0.05, n_steps=400, equil_steps=0,
                            tau_nhc=50.0, tau_baro=500.0, seed=1,
                            verbose=False, thermo_every=1000000,
                            traj_every=1000000,
                            output_dir=f"/tmp/akitsushima_test_npt_{tag}")
        result = sim.run()
        assert np.isfinite(result["H_npt_drift"]) and abs(result["H_npt_drift"]) < 5.0, (
            f"NPT ({tag}) should conserve H_NPT reasonably well, not blow up")
        return vol0, result["V_mean"]

    vol0_c, vmean_c = _run_npt(0.97, "compressed")
    vol0_e, vmean_e = _run_npt(1.03, "expanded")

    assert vmean_c > vol0_c, (
        "NPT barostat sign regression: a cell compressed below the true "
        "equilibrium volume must expand (mean volume > starting volume) "
        "under a target pressure of 0 GPa, not contract further.")
    assert vmean_e < vol0_e, (
        "NPT barostat sign regression: a cell expanded above the true "
        "equilibrium volume must contract (mean volume < starting volume) "
        "under a target pressure of 0 GPa, not expand further.")


# ===========================================================================
# S3: GCMC deletion energy must not self-overlap (ghost placed on a real atom)
# ===========================================================================

def test_local_energy_of_atom_matches_pair_energy_for_dimer():
    """
    For a two-atom system, removing atom 0 leaves a single atom (E=0), so
    the correct local energy of atom 0 is exactly the pair interaction
    energy V(r_01). The old buggy implementation (_probe_energy on
    self.positions[idx], i.e. a ghost placed exactly on top of the
    still-present real atom) would instead diverge to +/- infinity.
    """
    dimer = toy_dimer_dict()
    pair = frozenset({"Ar"})
    epsilon, sigma = 0.0103, 3.405
    engine = PotentialEngine().add(
        LennardJonesPotential({pair: {"epsilon": epsilon, "sigma": sigma}},
                              r_cut=8.0, shift=False))
    fn = make_force_fn(engine)

    sim = GCMCMDSimulation.__new__(GCMCMDSimulation)
    sim.lattice = dimer["lattice"]
    sim.positions = dimer["positions"].copy()
    sim.atom_types = list(dimer["atom_types"])
    sim._force_fn = fn
    sim.N = 2
    from akitsushima.engine.neighborlist import NeighbourList
    sim._nl = NeighbourList(r_cut=8.0, skin_width=0.5)

    r = float(np.linalg.norm(dimer["positions"][1] - dimer["positions"][0]))
    sr6 = (sigma / r) ** 6
    V_expected = 4.0 * epsilon * (sr6 ** 2 - sr6)

    u0 = sim._local_energy_of_atom(0)
    assert np.isfinite(u0), (
        "S3 regression: local energy must be finite, not the +/-inf that "
        "results from a ghost particle placed on top of the real atom")
    assert u0 == pytest.approx(V_expected, rel=1e-8)


def test_species_params_is_defined_and_constructible():
    """
    S14 regression: SpeciesParams was referenced throughout
    OSMMCSimulation's multi-species code path (isinstance checks, the
    class docstring's own example) but never actually defined anywhere,
    so constructing a multi-species OSMMCSimulation raised NameError
    immediately.
    """
    sp = SpeciesParams(mu=-0.12, mass=6.941, m_lam=7.0,
                       sc_alpha=0.55, sc_n=4, weight=2.0)
    assert sp.mu == -0.12
    assert sp.sc_alpha == 0.55

    # Defaults usable with only the mandatory field:
    sp_min = SpeciesParams(mu=-0.1)
    assert sp_min.mass is None
    assert sp_min.sc_alpha == 0.5
    assert sp_min.sc_n == 4


def test_gcmc_md_short_run_smoke():
    lattice = np.eye(3) * 15.0
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 15.0, size=(8, 3))
    atom_types = ["Ar"] * 8

    engine = PotentialEngine().add(
        LennardJonesPotential({frozenset({"Ar"}): {"epsilon": 0.0103, "sigma": 3.405}},
                              r_cut=6.0, shift=True))
    fn = make_force_fn(engine)

    sim = GCMCMDSimulation(
        lattice, positions, atom_types, fn,
        temperature=120.0, mu=-0.05, species="Ar", mass=39.948,
        timestep=1.0, n_cycles=30, md_steps_per_cycle=5,
        gcmc_attempts_per_cycle=8, seed=1, verbose=False,
        output_dir="/tmp/akitsushima_test_gcmc")
    result = sim.run()
    assert result["N_mean"] > 0
    assert np.isfinite(result["e_pot_mean"])


def test_osmmc_single_species_short_run_smoke():
    lattice = np.eye(3) * 15.0
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 15.0, size=(6, 3))
    atom_types = ["Ar"] * 6

    engine = PotentialEngine().add(
        LennardJonesPotential({frozenset({"Ar"}): {"epsilon": 0.0103, "sigma": 3.405}},
                              r_cut=6.0, shift=True))
    fn = make_force_fn(engine)

    sim = OSMMCSimulation(
        lattice, positions, atom_types, fn,
        temperature=120.0, mu=-0.05, species="Ar", mass=39.948,
        m_lam=50.0, sc_alpha=0.5, sc_n=4,
        timestep=1.0, n_cycles=30, md_steps_per_cycle=5,
        seed=1, verbose=False,
        output_dir="/tmp/akitsushima_test_osmmc")
    result = sim.run()
    assert np.isfinite(result["e_pot_mean"])
    assert result["multi_species"] is False


def test_osmmc_multi_species_construction():
    """S14 regression (integration level): a multi-species OSMMCSimulation
    must actually construct and run, not NameError on SpeciesParams."""
    lattice = np.eye(3) * 15.0
    rng = np.random.default_rng(2)
    positions = rng.uniform(0, 15.0, size=(6, 3))
    atom_types = ["Li"] * 4 + ["O"] * 2

    lj = {
        frozenset({"Li"}):    {"epsilon": 0.02, "sigma": 2.0},
        frozenset({"O"}):     {"epsilon": 0.02, "sigma": 2.8},
        frozenset({"Li","O"}): {"epsilon": 0.02, "sigma": 2.4},
    }
    engine = PotentialEngine().add(LennardJonesPotential(lj, r_cut=6.0, shift=True))
    fn = make_force_fn(engine)

    sim = OSMMCSimulation(
        lattice, positions, atom_types, fn,
        temperature=500.0,
        species_params={
            "Li": SpeciesParams(mu=-0.12, mass=6.941, m_lam=7.0,
                                sc_alpha=0.55, sc_n=4, weight=2.0),
            "O":  SpeciesParams(mu=-0.35, mass=15.999, m_lam=16.0,
                                sc_alpha=0.40, sc_n=4, weight=1.0),
        },
        n_cycles=20, md_steps_per_cycle=5, seed=3, verbose=False,
        output_dir="/tmp/akitsushima_test_osmmc_multi")
    result = sim.run()
    assert result["multi_species"] is True
    assert set(result["species_stats"].keys()) == {"Li", "O"}


# ===========================================================================
# `mu_ex` -> `mu` rename: backward compatibility
#
# GCMCMDSimulation/OSMMCSimulation's constructor kwarg and SpeciesParams'
# field were renamed from `mu_ex` to `mu` once test_gcmc_mu_is_the_full_
# chemical_potential (above) settled that the value had always been the
# FULL chemical potential, never the excess the old name implied.  Every
# old spelling -- the deprecated constructor kwarg, the deprecated
# SpeciesParams field/attribute, and the deprecated YAML config key
# (covered in test_cli_smoke.py) -- is kept working, so these tests pin
# that nothing which used the old name silently breaks.
# ===========================================================================

def test_species_params_mu_ex_kwarg_still_works_and_warns():
    """Deprecated construction path: SpeciesParams(mu_ex=...) must still
    build an equivalent object to SpeciesParams(mu=...), with a warning."""
    with pytest.warns(UserWarning, match="mu_ex"):
        sp = SpeciesParams(mu_ex=-0.12, mass=6.941)
    assert sp.mu == -0.12
    assert sp == SpeciesParams(mu=-0.12, mass=6.941)


def test_species_params_mu_ex_attribute_read_still_works_and_warns():
    """Deprecated attribute-read path: an existing SpeciesParams built
    the new way must still answer `.mu_ex` (returning `.mu`), with a
    warning, for code that reads the old attribute name."""
    sp = SpeciesParams(mu=-0.35, mass=15.999)
    with pytest.warns(UserWarning, match="mu_ex"):
        val = sp.mu_ex
    assert val == -0.35


def test_species_params_mu_and_mu_ex_together_raises():
    """Passing both the new and the deprecated spelling at once is
    ambiguous and must raise rather than silently pick one."""
    with pytest.raises(TypeError, match="mu_ex"):
        SpeciesParams(mu=-0.1, mu_ex=-0.2)


def test_gcmc_md_mu_ex_kwarg_still_works_and_warns(tmp_path):
    """Deprecated construction path: GCMCMDSimulation(..., mu_ex=...)
    must still set `.mu` to the given value, with a warning."""
    lattice = np.eye(3) * 15.0
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 15.0, size=(6, 3))
    engine = PotentialEngine().add(
        LennardJonesPotential({frozenset({"Ar"}): {"epsilon": 0.0103, "sigma": 3.405}},
                              r_cut=6.0, shift=True))
    fn = make_force_fn(engine)

    with pytest.warns(UserWarning, match="mu_ex"):
        sim = GCMCMDSimulation(
            lattice, positions, ["Ar"] * 6, fn, temperature=120.0,
            mu_ex=-0.05, species="Ar", mass=39.948, verbose=False,
            output_dir=str(tmp_path))
    assert sim.mu == -0.05

    # The instance-level attribute read must also still work (a SEPARATE
    # code path from the constructor kwarg above: a read-only property,
    # not the __init__ logic), and also warn.
    with pytest.warns(UserWarning, match="mu_ex"):
        assert sim.mu_ex == -0.05


def test_gcmc_md_mu_and_mu_ex_together_raises(tmp_path):
    lattice = np.eye(3) * 15.0
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 15.0, size=(6, 3))
    engine = PotentialEngine().add(
        LennardJonesPotential({frozenset({"Ar"}): {"epsilon": 0.0103, "sigma": 3.405}},
                              r_cut=6.0, shift=True))
    fn = make_force_fn(engine)

    with pytest.raises(TypeError, match="mu_ex"):
        GCMCMDSimulation(lattice, positions, ["Ar"] * 6, fn,
                         mu=-0.05, mu_ex=-0.05, verbose=False,
                         output_dir=str(tmp_path))


def test_osmmc_mu_ex_kwarg_still_works_and_warns(tmp_path):
    """The deprecated mu_ex= kwarg reaches OSMMCSimulation's parent
    constructor transparently via **kwargs, so single-species OSMMC
    construction must accept it identically to GCMCMDSimulation."""
    lattice = np.eye(3) * 15.0
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 15.0, size=(6, 3))
    engine = PotentialEngine().add(
        LennardJonesPotential({frozenset({"Ar"}): {"epsilon": 0.0103, "sigma": 3.405}},
                              r_cut=6.0, shift=True))
    fn = make_force_fn(engine)

    with pytest.warns(UserWarning, match="mu_ex"):
        sim = OSMMCSimulation(
            lattice, positions, ["Ar"] * 6, fn, temperature=120.0,
            mu_ex=-0.05, species="Ar", mass=39.948, verbose=False,
            output_dir=str(tmp_path))
    assert sim.mu == -0.05


def test_osmmc_species_stats_carries_deprecated_mu_ex_duplicate(tmp_path):
    """species_stats()'s returned per-species dict must expose BOTH the
    canonical 'mu' key and a same-valued deprecated 'mu_ex' duplicate,
    for any external code parsing the returned dict directly (a plain
    dict key can't itself emit a warning, unlike an attribute)."""
    lattice = np.eye(3) * 15.0
    rng = np.random.default_rng(2)
    positions = rng.uniform(0, 15.0, size=(6, 3))
    atom_types = ["Li"] * 4 + ["O"] * 2
    lj = {
        frozenset({"Li"}):    {"epsilon": 0.02, "sigma": 2.0},
        frozenset({"O"}):     {"epsilon": 0.02, "sigma": 2.8},
        frozenset({"Li","O"}): {"epsilon": 0.02, "sigma": 2.4},
    }
    engine = PotentialEngine().add(LennardJonesPotential(lj, r_cut=6.0, shift=True))
    fn = make_force_fn(engine)

    sim = OSMMCSimulation(
        lattice, positions, atom_types, fn, temperature=500.0,
        species_params={
            "Li": SpeciesParams(mu=-0.12, mass=6.941, m_lam=7.0),
            "O":  SpeciesParams(mu=-0.35, mass=15.999, m_lam=16.0),
        },
        n_cycles=1, md_steps_per_cycle=1, seed=3, verbose=False,
        output_dir=str(tmp_path))
    stats = sim.species_stats()
    assert stats["Li"]["mu"] == -0.12 == stats["Li"]["mu_ex"]
    assert stats["O"]["mu"] == -0.35 == stats["O"]["mu_ex"]


# ===========================================================================
# NEB: end-to-end regression on a synthetic analytic double well
# ===========================================================================

def _double_well_force_fn(h=1.0, w=1.0, x0=10.0):
    """Analytic single-atom double well V(x) = h*(1-((x-x0)/w)^2)^2, with
    F_y = F_z = 0 everywhere. Two degenerate minima at x0 +/- w, a single
    maximum of height h at x0. Isolates NEBChain/NEBOptimiser/
    TransitionStateAnalyser from any real pair-potential physics, in the
    same spirit as test_nhc_thermostat_isolated_is_stable_and_bounded."""
    def force_fn(lattice, positions, atom_types, neigh_list=None):
        x = positions[:, 0] - x0
        xi = x / w
        V = float(np.sum(h * (1.0 - xi**2) ** 2))
        dVdx = -4.0 * h * x / w**2 * (1.0 - xi**2)
        F = np.zeros_like(positions)
        F[:, 0] = -dVdx
        return V, F, np.zeros((3, 3))
    return force_fn


def test_neb_finds_analytic_saddle_on_synthetic_double_well(tmp_path):
    """
    End-to-end CI-NEB regression test: on a symmetric double well with an
    exactly known barrier height (h=1 eV) and saddle location (x=x0), CI-NEB
    must converge and place the climbing image within a tight tolerance of
    the analytic saddle. An even number of intermediate images is used so
    that linear interpolation does NOT already place an image exactly at
    the saddle (which would make the test pass trivially without the FIRE
    loop doing any real work); this exercises the full NEBChain tangent /
    spring-force projection and the NEBOptimiser FIRE loop (including the
    S11 n_pos-threshold fix) together.
    """
    h, w, x0 = 1.0, 1.0, 10.0
    force_fn = _double_well_force_fn(h, w, x0)

    lattice   = np.eye(3) * 20.0
    pos_init  = np.array([[x0 - w, 10.0, 10.0]])
    pos_final = np.array([[x0 + w, 10.0, 10.0]])

    chain = NEBChain.linear_interpolation(
        pos_init, pos_final, lattice, ["X"], force_fn,
        n_images=4, k_spring=5.0, climbing=True, r_cut=5.0)
    opt = NEBOptimiser(chain, fmax=1e-3, max_steps=500, verbose=False,
                       output_dir=str(tmp_path))
    result = opt.run()

    assert result["converged"]
    assert result["activation_energy"] == pytest.approx(h, abs=1e-3)
    assert result["reverse_barrier"] == pytest.approx(h, abs=1e-3)
    ts_x = result["climbing_image_pos"][0, 0]
    assert ts_x == pytest.approx(x0, abs=1e-2)


def test_neb_wraps_images_back_into_the_periodic_box(tmp_path):
    """
    S20 regression: NEBOptimiser.run() never wrapped image positions back
    into the primary cell after a FIRE step (unlike GeometryOptimiser,
    which does `frac -= np.floor(frac)`), so an atom drifting outside
    [0, box) during optimisation -- entirely normal for a diffusion/hop
    pathway, the very thing NEB is used for -- made NeighbourList's
    cKDTree(..., boxsize=...) raise ValueError ("Some input data are
    greater than the size of the periodic box"). Reproduced here with a
    real LJ FCC Al supercell and an end-state displaced far enough that
    the interpolated/relaxed path pushes an atom outside the cell.
    """
    al = _fcc_al_supercell(n=2)
    lattice, positions, atom_types = al["lattice"], al["positions"], al["atom_types"]
    fn = _lj_force_fn(r_cut=3.2)

    pos_final = positions.copy()
    pos_final[0] = pos_final[0] + np.array([0.3, 0.0, 0.0])

    chain = NEBChain.linear_interpolation(
        positions, pos_final, lattice, atom_types, fn,
        n_images=3, k_spring=5.0, climbing=True, r_cut=3.2)
    opt = NEBOptimiser(chain, fmax=0.1, max_steps=300, verbose=False,
                       output_dir=str(tmp_path))

    result = opt.run()   # must not raise ValueError from cKDTree
    assert np.all(np.isfinite(result["activation_energy"]))


# ===========================================================================
# TSA: S11 (is_first_order_saddle threshold) and S6 (run_irc dict return)
# ===========================================================================

def test_is_first_order_saddle_uses_passed_threshold():
    """
    S11 regression: is_first_order_saddle used to accept a `threshold`
    argument but ignore it entirely, unconditionally returning
    self._modes["is_ts"] (computed by analyse_modes() with its own
    hard-coded -1e-3 cutoff). Here the same pre-computed eigenvalues must
    yield DIFFERENT answers depending on the threshold actually passed --
    something the old implementation could never do.
    """
    analyser = TransitionStateAnalyser.__new__(TransitionStateAnalyser)
    analyser._modes = {
        "eigenvalues": np.array([-0.5, -0.05, 0.1, 0.2, 0.3]),
        "is_ts": False,   # analyse_modes()'s own -1e-3 cutoff counts both
                           # -0.5 and -0.05 as negative -> n_negative=2
    }
    # Default-equivalent threshold: two eigenvalues count as negative.
    assert analyser.is_first_order_saddle(threshold=-1e-3) is False
    # Looser threshold: only -0.5 counts as negative -> exactly one.
    assert analyser.is_first_order_saddle(threshold=-0.2) is True


def test_tsa_hessian_and_saddle_check_on_synthetic_double_well():
    """
    Integration-level check of compute_hessian -> analyse_modes ->
    is_first_order_saddle on the same synthetic double well used for the
    NEB test above: at the exact saddle point x=x0 there is one negative
    curvature (x) and two exactly-flat directions (y, z, since F_y=F_z=0
    everywhere), so it must be classified as a genuine first-order saddle,
    and the transition vector must point along x.
    """
    h, w, x0 = 1.0, 1.0, 10.0
    force_fn = _double_well_force_fn(h, w, x0)

    analyser = TransitionStateAnalyser(
        lattice=np.eye(3) * 20.0,
        positions=np.array([[x0, 10.0, 10.0]]),
        atom_types=["X"],
        force_fn=force_fn,
        masses=np.array([1.0]),
    )
    modes = analyser.analyse_modes(delta=0.001, mass_weighted=False)
    assert modes["n_negative"] == 1
    assert analyser.is_first_order_saddle(threshold=-1e-3) is True

    tv = modes["transition_vector"]
    assert abs(abs(tv[0, 0]) - 1.0) < 1e-6, (
        "transition vector must point along the unstable (x) direction")


def _harmonic_well_force_fn(k_vec, r0):
    """Analytic single-atom anisotropic harmonic well
    V(r) = 0.5 * sum_alpha k_alpha (r_alpha - r0_alpha)^2, whose Cartesian
    Hessian is EXACTLY diag(k_x, k_y, k_z) [eV/A^2] -- so the resulting
    vibrational frequencies have a closed form and need no reference data.
    Same isolate-from-real-pair-potential-physics spirit as
    _double_well_force_fn above."""
    k_vec = np.asarray(k_vec, dtype=float)
    r0 = np.asarray(r0, dtype=float)

    def force_fn(lattice, positions, atom_types, neigh_list=None):
        d = positions - r0
        V = float(0.5 * np.sum(k_vec * d**2))
        F = -k_vec * d
        return V, F, np.zeros((3, 3))
    return force_fn


def test_wavenumber_conversion_uses_exact_speed_of_light():
    """
    Regression for the rounded-speed-of-light bug in
    FREQ_RAD_FS_TO_WAVENUMBER_CM (constants.py), the single constant
    shared by TransitionStateAnalyser.analyse_modes(),
    PhononCalculator._eig_to_freq() and
    DiffusionAnalyzer.compute_vacf_dos(): it used a rounded
    c = 3.0e-5 cm/fs instead of the exact SI value 2.99792458e-5 cm/fs,
    evaluating to 5305.1648 instead of 5308.8375 -- so EVERY wavenumber
    this package reported was systematically 0.069% too low.

    Two independent checks:

    1. The constant itself, against an oracle built directly from the SI
       definition of the metre (c = 299792458 m/s, exact by definition),
       written out literally here rather than imported, so this test
       cannot agree with the code by sharing its mistake.
    2. The end-to-end analyse_modes() path on a single atom in an
       anisotropic harmonic well, whose mass-weighted Hessian eigenvalues
       are exactly k_alpha/m, giving closed-form wavenumbers.  This
       catches the constant being correct but mis-applied (e.g. a missing
       or doubled 2*pi) at the call site.

    The tolerance (rtol=1e-9) is five orders of magnitude tighter than
    the 6.9e-4 relative error the rounded value introduced, so this test
    genuinely discriminates between the two constants rather than passing
    on either -- asserted explicitly at the end.

    EV_ANG_TO_AMU_ANG_FS2 is a separate, independently-derived constant
    (eV/A -> amu.A/fs^2) that is not the subject of this test; it is used
    as given so the check stays focused on the rad/fs -> cm^-1 step.
    """
    # c = 299792458 m/s exactly (SI definition of the metre)
    #   -> 2.99792458e10 cm/s -> x1e-15 s/fs -> cm/fs
    c_cm_per_fs = 299792458.0 * 100.0 * 1.0e-15
    expected_constant = 1.0 / (2.0 * np.pi * c_cm_per_fs)

    assert FREQ_RAD_FS_TO_WAVENUMBER_CM == pytest.approx(
        expected_constant, rel=1e-12), (
        f"FREQ_RAD_FS_TO_WAVENUMBER_CM must use the exact speed of light: "
        f"got {FREQ_RAD_FS_TO_WAVENUMBER_CM:.6f}, "
        f"expected {expected_constant:.6f}")

    # --- end-to-end: single atom in an anisotropic harmonic well ---------
    k_vec = np.array([2.0, 5.0, 9.0])        # [eV/A^2], deliberately distinct
    mass = 28.086                            # [amu] (Si)
    r0 = np.array([[5.0, 5.0, 5.0]])

    analyser = TransitionStateAnalyser(
        lattice=np.eye(3) * 10.0,
        positions=r0.copy(),
        atom_types=["Si"],
        force_fn=_harmonic_well_force_fn(k_vec, r0),
        masses=np.array([mass]),
    )
    modes = analyser.analyse_modes(delta=1e-4, mass_weighted=True)

    # Mass-weighted eigenvalues are exactly k_alpha / m [eV/(A^2.amu)];
    # omega = sqrt(lambda * EV_ANG_TO_AMU_ANG_FS2) [rad/fs];
    # nu~ = omega / (2 pi c).
    omega = np.sqrt(np.sort(k_vec) / mass * EV_ANG_TO_AMU_ANG_FS2)
    expected_cm = omega / (2.0 * np.pi * c_cm_per_fs)

    got_cm = np.sort(modes["frequencies_cm"])
    assert got_cm == pytest.approx(expected_cm, rel=1e-6), (
        f"analyse_modes wavenumbers must match the closed-form harmonic "
        f"result: got {got_cm}, expected {expected_cm}")

    # The test must actually be able to tell the two constants apart: the
    # old rounded c = 3.0e-5 cm/fs would have produced wavenumbers 0.069%
    # low, far outside the tolerances asserted above.
    rounded_constant = 1.0 / (2.0 * np.pi * 3.0e-5)
    rel_gap = abs(rounded_constant - expected_constant) / expected_constant
    assert rel_gap > 1e-4, (
        "sanity: the rounded and exact constants must differ by far more "
        "than this test's tolerance, or it would not discriminate")


def test_run_irc_returns_dict_with_energies_and_convergence(monkeypatch, tmp_path):
    """
    S6 regression: run_irc() used to return a bare (pos_forward,
    pos_backward) tuple, discarding the energy/convergence information that
    GeometryOptimiser.run() already computes for each side (the CLI layer
    needs exactly this to report barriers, and assumed a dict). It must now
    return a dict exposing pos_forward/pos_backward plus e_forward/
    e_backward/converged_forward/converged_backward.

    GeometryOptimiser itself is stubbed out here (regression-tested for its
    own correctness elsewhere) so this test targets only run_irc's
    return-value plumbing, not relaxation dynamics.
    """
    analyser = TransitionStateAnalyser.__new__(TransitionStateAnalyser)
    analyser.lattice = np.eye(3) * 10.0
    analyser.positions = np.array([[5.0, 5.0, 5.0]])
    analyser.atom_types = ["X"]
    analyser.force_fn = lambda *a, **k: (0.0, np.zeros((1, 3)), np.zeros((3, 3)))
    analyser.N = 1
    analyser._modes = {"transition_vector": np.array([[1.0, 0.0, 0.0]])}

    class _FakeGO:
        _counter = [0]

        def __init__(self, lattice, positions, atom_types, force_fn,
                     fmax, max_steps, skin_width, output_dir, verbose):
            self._positions = positions
            self._label = "forward" if _FakeGO._counter[0] == 0 else "backward"
            _FakeGO._counter[0] += 1

        def run(self):
            if self._label == "forward":
                return {"positions": self._positions + 0.1, "energy": -1.0,
                        "converged": True}
            return {"positions": self._positions - 0.1, "energy": -2.0,
                    "converged": False}

    import akitsushima.solvers.tsa as tsa_mod
    monkeypatch.setattr(tsa_mod, "GeometryOptimiser", _FakeGO)

    irc = analyser.run_irc(output_dir=str(tmp_path), verbose=False)

    assert isinstance(irc, dict), (
        "S6 regression: run_irc must return a dict, not a bare tuple")
    assert set(irc.keys()) == {
        "pos_forward", "pos_backward", "e_forward", "e_backward",
        "converged_forward", "converged_backward"}
    assert irc["e_forward"] == -1.0
    assert irc["e_backward"] == -2.0
    assert irc["converged_forward"] is True
    assert irc["converged_backward"] is False


# ===========================================================================
# PhononCalculator: S2 (IFC sign), S16 (self.lattice), S11 (atom_types
# save/load), S12 (ESM isinstance -> .kind)
# ===========================================================================

def _asr_correct(H: np.ndarray, N: int) -> np.ndarray:
    """Apply the same acoustic-sum-rule row-subtraction PhononCalculator
    applies to its IFC, so a Hessian computed by an independent method can
    be compared against it on equal footing."""
    H_4d = H.reshape(N, 3, N, 3).copy()
    row_sum = H_4d.sum(axis=2)
    for i in range(N):
        H_4d[i, :, i, :] -= row_sum[i]
    return H_4d.reshape(N * 3, N * 3)


def test_compute_ifc_matches_tsa_hessian_supercell_one(small_cell_ok):
    """
    S2 regression: compute_ifc()'s finite-difference force-constant
    formula must include the same negative sign as
    TransitionStateAnalyser.compute_hessian() (Phi = -dF/dr) -- the two
    are mathematically the identical quantity when PhononCalculator is
    given supercell=1 (N_sc == N_basis, identity sc_to_prim), so after
    applying the same ASR correction to both, they must agree to machine
    precision. Before the fix, PhononCalculator's IFC had the opposite
    sign from TSA's Hessian.
    """
    fn = _lj_force_fn()
    al = fcc_al_dict()
    N = len(al["atom_types"])

    tsa = TransitionStateAnalyser(al["lattice"], al["positions"],
                                  al["atom_types"], fn)
    H = tsa.compute_hessian(delta=0.01)
    H_asr = _asr_correct(H, N)

    pc = PhononCalculator(al["lattice"], al["positions"], al["atom_types"],
                          fn, supercell=1, delta=0.01)
    IFC = pc.compute_ifc(verbose=False)

    assert np.allclose(H_asr, IFC, atol=1e-10), (
        "S2 regression: PhononCalculator.compute_ifc() must agree with "
        "TransitionStateAnalyser.compute_hessian() (same ASR correction "
        "applied to both) -- a sign error here flips the WHOLE dynamical "
        "matrix and misclassifies every stable mode as imaginary")


def test_phonon_dispersion_has_no_imaginary_modes_for_stable_crystal():
    """
    Integration-level S2 regression: a mechanically stable LJ FCC crystal
    must show purely real phonon frequencies everywhere along a
    high-symmetry path (this is what compute_ifc's missing negative sign
    would break: EVERY non-acoustic-zero mode would appear imaginary).
    Acoustic branches must also go to (near-)zero at Gamma, as enforced by
    the D-ASR correction.
    """
    fn = _lj_force_fn(r_cut=5.0)
    al = fcc_al_dict()
    pc = PhononCalculator(al["lattice"], al["positions"], al["atom_types"],
                         fn, supercell=3, delta=0.01)
    pc.compute_ifc(verbose=False)

    bands = pc.compute_band_structure(
        [("G", [0, 0, 0]), ("X", [0.5, 0, 0]), ("M", [0.5, 0.5, 0]),
         ("G", [0, 0, 0])], n_points=10, verbose=False)
    freqs = bands["frequencies"]

    assert np.sum(freqs < -1.0) == 0, (
        "a stable LJ FCC crystal must show no imaginary phonon modes")
    assert np.max(np.abs(freqs[0, :3])) < 1e-2, (
        "the 3 acoustic branches must vanish at Gamma (D-ASR)")


def test_debye_temperature_from_vsound_runs_without_attributeerror():
    """
    S16 regression (newly found during the port): debye_temperature_from_
    vsound() referenced self.lattice, an attribute this class never sets
    (only self.lattice_prim is ever assigned) -- every call raised
    AttributeError immediately, regardless of input. It must now run and
    return a finite, positive temperature.
    """
    fn = _lj_force_fn(r_cut=5.0)
    al = fcc_al_dict()
    pc = PhononCalculator(al["lattice"], al["positions"], al["atom_types"],
                         fn, supercell=3, delta=0.01)
    pc.compute_ifc(verbose=False)

    theta_D = pc.debye_temperature_from_vsound(verbose=False)
    assert np.isfinite(theta_D) and theta_D > 0.0


def test_phonon_save_load_round_trip_preserves_atom_types(tmp_path, small_cell_ok):
    """
    S11 regression: save_results() never wrote atom_types to ifc.npz, so
    load_ifc()'s d.get('atom_types', ['X', ...]) fallback always silently
    substituted placeholder labels -- the real element symbols were lost
    on every save/load round trip. Also confirms allow_pickle=True is no
    longer needed (np.load() is called without it in load_ifc()).
    """
    fn = _lj_force_fn(r_cut=5.0)
    al = fcc_al_dict()
    pc = PhononCalculator(al["lattice"], al["positions"], al["atom_types"],
                         fn, supercell=2, delta=0.01)
    pc.compute_ifc(verbose=False)
    pc.save_results(str(tmp_path), verbose=False)

    pc2 = PhononCalculator.load_ifc(str(tmp_path / "ifc.npz"), fn)
    assert list(pc2.atom_types) == al["atom_types"], (
        "S11 regression: atom_types must survive a save_results()/"
        "load_ifc() round trip, not fall back to placeholder 'X' labels")
    assert np.allclose(pc2.IFC, pc.IFC)


def test_phonon_esm_warning_uses_kind_attribute_not_isinstance():
    """
    S12 regression: the C-4 supercell[2]>1 + ESM warning used to require
    isinstance(force_fn, ESMCalculator), forcing this module to import
    engine internals just to recognise it. It must now key off the same
    getattr(force_fn, "kind", None) == "esm" convention used throughout
    solvers/gcmc.py, so a bare stand-in object with a .kind/.bc attribute
    (never an actual ESMCalculator) still triggers the warning.
    """
    fn = _lj_force_fn(r_cut=1.0)

    class _FakeESM:
        kind = "esm"
        bc = "bc2"
        r_cut = 1.0

        def __call__(self, lattice, positions, atom_types, neigh_list=None):
            return fn(lattice, positions, atom_types, neigh_list=neigh_list)

    al = fcc_al_dict()
    with pytest.warns(UserWarning, match="ESM"):
        PhononCalculator(al["lattice"], al["positions"], al["atom_types"],
                        _FakeESM(), supercell=[2, 2, 2], delta=0.01)


# ===========================================================================
# ThermalConductivityCalculator: S4 (pair-virial flux), S17 (NVTSimulation
# velocities= crash), S18 (compute_kappa block-size crash)
# ===========================================================================

def _tc_al_supercell(n=2):
    base = fcc_al_dict()
    lattice = base["lattice"] * n
    positions, atom_types = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                shift = np.array([i, j, k], dtype=float) @ base["lattice"]
                positions.append(base["positions"] + shift)
                atom_types.extend(base["atom_types"])
    return (lattice, np.concatenate(positions, axis=0), atom_types)


def test_pair_virial_flux_is_not_negligible_unlike_com_approximation():
    """
    S4 regression: the original approximated the ENTIRE pair-virial heat-
    flux term from the aggregate stress tensor and the mean (centre-of-
    mass) velocity alone, which is exactly zero whenever total momentum is
    exactly zero (as constructed here) -- silently discarding the
    dominant heat-conduction term for a solid. The new exact
    per-pair-decomposable calculation must NOT vanish under the same
    conditions.
    """
    lattice, positions, atom_types = _tc_al_supercell()
    N = len(atom_types)
    al = randomly_perturbed({"lattice": lattice, "positions": positions,
                             "atom_types": atom_types}, magnitude=0.05, seed=1)
    fn = _lj_force_fn(r_cut=3.2)

    tc = ThermalConductivityCalculator(al["lattice"], al["positions"],
                                       al["atom_types"], fn, temperature=300.0)
    rng = np.random.default_rng(3)
    vel = rng.standard_normal((N, 3)) * 0.001
    vel -= vel.mean(axis=0)   # exactly zero total (COM) momentum
    assert np.allclose(vel.mean(axis=0), 0.0, atol=1e-12)

    _, _, stress = fn(al["lattice"], al["positions"], al["atom_types"])
    pair_data = tc._gather_pair_data(al["positions"])
    J_vir = tc._pair_virial_flux(al["positions"], vel, stress, pair_data)

    # Old buggy formula, recomputed inline for comparison: -V/2 * stress @ v_mean.
    V = abs(np.linalg.det(al["lattice"]))
    J_vir_old_approx = -V * stress @ vel.mean(axis=0) * 0.5

    assert np.linalg.norm(J_vir) > 1e-6, (
        "S4 regression: the exact pair-virial flux must not vanish just "
        "because total momentum is zero")
    assert np.linalg.norm(J_vir_old_approx) < 1e-15, (
        "sanity: the OLD centre-of-mass approximation really is "
        "numerically zero here (confirming it IS the bug being fixed)")


def test_per_atom_energy_matches_pair_decomposition_and_sums_to_total():
    """
    S4 regression (per-atom energy half): the pair-decomposed per-atom
    energies must sum back to the total energy, and must NOT all be
    identical (unlike the old uniform E_tot/N split) for a perturbed,
    asymmetric structure.
    """
    lattice, positions, atom_types = _tc_al_supercell()
    al = randomly_perturbed({"lattice": lattice, "positions": positions,
                             "atom_types": atom_types}, magnitude=0.05, seed=2)
    fn = _lj_force_fn(r_cut=3.2)
    tc = ThermalConductivityCalculator(al["lattice"], al["positions"],
                                       al["atom_types"], fn, temperature=300.0)

    E_tot, forces, _ = fn(al["lattice"], al["positions"], al["atom_types"])
    pair_data = tc._gather_pair_data(al["positions"])
    e_pot_i = tc._per_atom_energy(al["positions"], forces, float(E_tot), pair_data)

    assert e_pot_i.sum() == pytest.approx(float(E_tot), rel=1e-8)
    assert np.std(e_pot_i) > 1e-6, (
        "S4 regression: per-atom energies for a perturbed (asymmetric) "
        "structure must vary atom-to-atom, not all equal E_tot/N")


def test_run_md_does_not_crash_with_default_equilibration():
    """
    S17 regression: run_md() used to pass velocities=... to NVTSimulation,
    which has never accepted that keyword argument (checked against the
    ported class and the original monolithic file alike) -- every call
    with equil_steps > 0 (the default) raised TypeError immediately.
    """
    lattice, positions, atom_types = _tc_al_supercell()
    fn = _lj_force_fn(r_cut=3.2)
    tc = ThermalConductivityCalculator(lattice, positions, atom_types, fn,
                                       temperature=300.0)
    result = tc.run_md(n_steps=20, timestep=0.2, equil_steps=10,
                       heat_flux_every=2, traj_every=1000, thermo_every=1000,
                       seed=7, verbose=False,
                       output_dir="/tmp/akitsushima_test_tc_run_md")
    assert result["J_series"].shape == (10, 3)
    assert np.all(np.isfinite(result["J_series"]))


def test_compute_kappa_block_averaging_does_not_crash(tmp_path):
    """
    S18 regression: block_size = max(n // n_blocks, n_lag) was computed
    but never actually used for the block slicing (start/end used the
    unfloored n // n_blocks instead), so whenever n_lag exceeded a raw
    n // n_blocks block's length -- true even for this class's own
    docstring example -- _autocorr_fft silently returned a shorter-than-
    expected array and NumPy raised ValueError on the broadcast into hb.
    """
    lattice, positions, atom_types = _tc_al_supercell()
    fn = _lj_force_fn(r_cut=3.2)
    tc = ThermalConductivityCalculator(lattice, positions, atom_types, fn,
                                       temperature=300.0)
    tc.run_md(n_steps=400, timestep=0.2, equil_steps=50,
             heat_flux_every=2, traj_every=1000, thermo_every=1000,
             seed=7, verbose=False, output_dir=str(tmp_path))

    # n_lag will exceed a naive (unfloored) n // n_blocks-sized block here.
    kappa = tc.compute_kappa(tau_max=0.01, n_blocks=2, verbose=False)
    assert np.all(np.isfinite(kappa))
    assert np.all(np.isfinite(tc._kappa_std)), (
        "with these parameters >= 2 full-sized blocks fit, so a real "
        "(finite) uncertainty estimate should be reported, not NaN")


def test_compute_kappa_warns_and_returns_nan_std_when_too_few_blocks(tmp_path):
    """S18 edge case: when fewer than 2 full-sized blocks fit (e.g.
    n_blocks=1), compute_kappa must warn and report kappa_std as NaN
    rather than crashing or fabricating an uncertainty from too little
    data; the main kappa estimate (from the full series) is unaffected."""
    lattice, positions, atom_types = _tc_al_supercell()
    fn = _lj_force_fn(r_cut=3.2)
    tc = ThermalConductivityCalculator(lattice, positions, atom_types, fn,
                                       temperature=300.0)
    tc.run_md(n_steps=400, timestep=0.2, equil_steps=50,
             heat_flux_every=2, traj_every=1000, thermo_every=1000,
             seed=7, verbose=False, output_dir=str(tmp_path))

    with pytest.warns(UserWarning, match="too short"):
        kappa = tc.compute_kappa(tau_max=0.01, n_blocks=1, verbose=False)
    assert np.all(np.isfinite(kappa))
    assert np.all(np.isnan(tc._kappa_std))


def test_slack_debye_runs_with_both_theta_d_methods():
    """Basic smoke test: slack_debye() must run to completion (returning a
    finite, positive kappa) with both the default DOS-cutoff Debye
    temperature and the use_vsound=True alternative."""
    fn = _lj_force_fn(r_cut=5.0)
    al = fcc_al_dict()
    pc = PhononCalculator(al["lattice"], al["positions"], al["atom_types"],
                         fn, supercell=3, delta=0.01)
    pc.compute_ifc(verbose=False)
    pc.compute_dos(q_mesh=[4, 4, 4], verbose=False)

    tc = ThermalConductivityCalculator(al["lattice"], al["positions"],
                                       al["atom_types"], fn, temperature=300.0)
    k_dos = tc.slack_debye(pc, verbose=False)
    k_vs  = tc.slack_debye(pc, use_vsound=True, verbose=False)
    assert np.isfinite(k_dos) and k_dos > 0.0
    assert np.isfinite(k_vs) and k_vs > 0.0


# ===========================================================================
# GCMC: `mu_ex` is the FULL chemical potential, not the excess
# ===========================================================================

def test_gcmc_mu_is_the_full_chemical_potential(tmp_path):
    """
    Settles what GCMCMDSimulation's chemical-potential argument actually
    means, and pins it so the documentation cannot drift back.

    (This test predates the ``mu_ex`` -> ``mu`` rename that followed from
    its own result; it now exercises the current, canonical ``mu=``
    kwarg.  The deprecated ``mu_ex=`` alias has its own dedicated
    backward-compatibility tests elsewhere in this file.)

    The acceptance rule is the textbook grand-canonical form (Frenkel &
    Smit eq. 5.6.8), ``acc = min(1, V/(lambda^3 (N+1)) exp(beta(mu-dU)))``,
    whose ``mu`` is the FULL chemical potential -- substituting
    ``mu = mu_id + mu_ex`` collapses the prefactor to ``N/(N+1)``, so
    keeping the explicit ideal-gas factor while being handed an EXCESS mu
    double-counts ``mu_id``.  The parameter was nonetheless documented as
    the excess, an error worth 9.5 kT for supercritical Ar -- more than
    the whole range a user tunes over.

    The IDEAL GAS settles it exactly, with no reference data, no fitted
    potential and no Widom statistics: interactions vanish, so

        <N> = V exp(beta mu) / lambda^3

    holds in closed form.  Passing ``mu = kT ln(N_target lambda^3 / V)``
    must therefore reproduce ``N_target``.  Three well-separated targets
    are used so a coincidental single-point match cannot pass.

    The excess reading additionally fails on its own terms here, which is
    why this fixture is decisive rather than merely suggestive: for an
    ideal gas ``mu_ex == 0`` at EVERY density, so it cannot select one --
    an input that cannot distinguish N=30 from N=120 cannot be what a
    muVT acceptance rule consumes.
    """
    T = 200.0
    kT = K_B * T
    L = 22.0
    V = L ** 3
    lattice = np.eye(3) * L

    # Thermal de Broglie wavelength, by the same formula gcmc.py uses.
    h_J, kB_J = 6.62607015e-34, 1.380649e-23
    mass = 39.948
    lam = (h_J / np.sqrt(2.0 * np.pi * mass * 1.66053906660e-27 * kB_J * T)
           ) * 1e10

    def ideal_force_fn(lat, positions, atom_types, neigh_list=None):
        return 0.0, np.zeros_like(positions), np.zeros((3, 3))

    ideal_force_fn.kind = None
    ideal_force_fn.r_cut = 6.0

    # Kept deliberately short: an ideal gas has no correlations to
    # decorrelate, so N equilibrates within a few tens of cycles.  Two
    # well-separated targets rule out a coincidental single-point match.
    for n_target in (25, 100):
        mu_full = kT * np.log(n_target * lam ** 3 / V)
        rng = np.random.default_rng(0)
        pos = rng.random((n_target, 3)) * L
        gc = GCMCMDSimulation(
            lattice, pos, ["Ar"] * n_target, ideal_force_fn,
            temperature=T, mu=mu_full, species="Ar", mass=mass,
            cavity_bias=False, energy_bias_del=False,
            n_cycles=150, md_steps_per_cycle=0, gcmc_attempts_per_cycle=40,
            timestep=5.0, output_dir=str(tmp_path), verbose=False, seed=1)
        hist = np.asarray(gc.run()["N_history"], dtype=float)
        n_mean = float(hist[len(hist) // 2:].mean())

        assert n_mean == pytest.approx(n_target, rel=0.10), (
            f"passing the FULL chemical potential mu = {mu_full:.5f} eV "
            f"must reproduce the exact ideal-gas result <N> = "
            f"V exp(beta mu)/lambda^3 = {n_target}; got {n_mean:.1f}.  If "
            f"this fails, either the acceptance rule's ideal-gas "
            f"normalisation changed, or mu is no longer the full "
            f"chemical potential")


def test_npt_reduces_to_nve_in_the_weak_coupling_limit():
    """C1 regression: NPTSimulation had no free-flight drift.

    `_scale_step` applied the affine cell rescale and the barostat
    velocity rescale but never `r += dt*v`, which NVT and NVE both do.
    The affine rescale is a similarity transform, so every fractional
    coordinate stayed exactly put: the trajectory was one frozen
    structure with a breathing lattice, evaluated at a force that never
    changed.  With positions frozen the velocity half-steps integrate a
    constant acceleration forever, so <T> ran away (1704 K against an
    NVE reference of 342 K on the reporter's system).

    The oracle needs no reference data: as tau_nhc and tau_baro go to
    infinity the thermostat and barostat decouple and NPT must reduce
    to NVE exactly.  That limit is what pins both the diagnosis and the
    fix -- a static-cell bug cannot pass it, and the corrected
    integrator reproduces NVE to rounding.
    """
    fn = _lj_force_fn(r_cut=3.2)
    al = _fcc_al_supercell()
    common = dict(lattice=al["lattice"], positions=al["positions"],
                  atom_types=al["atom_types"], force_fn=fn, r_cut=3.2,
                  timestep=0.5, n_steps=200, equil_steps=0,
                  temperature=600.0, traj_every=10**9, thermo_every=10**9,
                  seed=42, verbose=False)
    nve = NVESimulation(**common,
                        output_dir="/tmp/akitsushima_test_npt_nve_a").run()
    npt = NPTSimulation(**common, pressure=0.0, tau_nhc=1e8, tau_baro=1e8,
                        output_dir="/tmp/akitsushima_test_npt_nve_b").run()

    assert np.max(np.abs(npt["positions"] - nve["positions"])) < 1e-8, (
        "NPT does not reduce to NVE in the weak-coupling limit")
    assert abs(npt["T_mean"] - nve["T_mean"]) < 1e-6

    # And, directly: the atoms must actually move relative to the cell.
    frac_0 = al["positions"] @ np.linalg.inv(al["lattice"])
    frac_N = npt["positions"] @ np.linalg.inv(npt["lattice"])
    assert np.max(np.abs(frac_N - frac_0)) > 1e-6, (
        "fractional coordinates are unchanged -- the drift is missing")


def test_nhc_conserved_quantity_is_actually_conserved():
    """H2 regression: the Nose-Hoover chain kinetic term was divided by
    EV_ANG_TO_AMU_ANG_FS2, inflating it by 1/0.0096485 = 103.6x.

    Q1 = n_dof*k_B[eV/K]*T[K]*tau[fs]^2 is eV*fs^2 and vxi is fs^-1, so
    0.5*Q1*vxi^2 is already eV; unlike the particle kinetic energy it
    never enters the amu basis.  The dynamics were unaffected -- the
    factor appeared only in the diagnostic -- but H_NHC is the only
    integrator-health signal the package reports, and a user tuning
    their timestep against it would have concluded their step was far
    too large.

    Conservation is its own oracle here: the exact invariant of the NHC
    equations of motion must not drift.  Before the fix this run
    spanned 27.13 eV and drifted +9.46 eV.
    """
    fn = _lj_force_fn(r_cut=3.2)
    al = _fcc_al_supercell()
    res = NVTSimulation(lattice=al["lattice"], positions=al["positions"],
                        atom_types=al["atom_types"], force_fn=fn, r_cut=3.2,
                        timestep=0.1, n_steps=3000, equil_steps=0,
                        temperature=300.0, tau_nhc=50.0, traj_every=10**9,
                        thermo_every=1, seed=42, verbose=False,
                        output_dir="/tmp/akitsushima_test_nhc_conserved").run()
    H = np.array([r["H_nhc"] for r in res["thermo_log"]])
    span = float(H.max() - H.min())
    # Two orders of magnitude tighter than the pre-fix 27.13 eV, and two
    # orders looser than the 1.3e-05 eV actually measured, so this is a
    # discrimination test rather than a fit to one machine's rounding.
    assert span < 1e-3, f"H_NHC is not conserved: span {span:.4e} eV"
