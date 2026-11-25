"""Shared helpers: synthetic XDATCAR writer and a Langevin harmonic-crystal model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from akitsushima.mdspec.constants import AMU, ATOMIC_MASS, KB

DATA = Path(__file__).parent.parent / "data" / "nozuchi"


def write_xdatcar(path, species, counts, frac, lattice, variable=False, comment="test", start=1):
    """Write frames frac (n, N, 3) with lattice (3, 3) or (n, 3, 3) in VASP 5 format."""
    frac = np.asarray(frac)
    lat = np.asarray(lattice)
    lats = np.broadcast_to(lat, (len(frac), 3, 3)) if lat.ndim == 2 else lat

    def header(lv):
        s = f"{comment}\n           1\n"
        s += "".join(f"    {v[0]:12.6f}{v[1]:12.6f}{v[2]:12.6f}\n" for v in lv)
        s += "   " + "   ".join(species) + "\n   " + "   ".join(str(c) for c in counts) + "\n"
        return s

    with open(path, "w") as fh:
        for k, fr in enumerate(frac):
            if k == 0 or variable:
                fh.write(header(lats[k]))
            fh.write(f"Direct configuration= {start + k:5d}\n")
            fh.writelines(f"  {x:.8f}  {y:.8f}  {z:.8f}\n" for x, y, z in fr)


def langevin_crystal(
    species=("Mg", "O"),
    charges=(2.0, -2.0),
    k_spring=(8.0, 8.0),  # eV / A^2 per species
    n_cells=2,
    a=4.2,
    temperature=300.0,
    dt_fs=1.0,
    n_steps=40000,
    gamma_ps=0.5,
    seed=0,
):
    """Independent charged 3D Einstein oscillators (rock-salt sites), BAOAB Langevin.

    Exact static response: <dM_a^2> = sum_i q_i^2 kB T / k_i, i.e.
    Delta eps = sum_i q_i^2 / (k_i eps0 V) (BAOAB samples the harmonic
    configurational distribution exactly).  Returns positions (n, N, 3) [A],
    lattice, symbols, masses [amu] and the exact Delta eps.
    """
    rng = np.random.default_rng(seed)
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array(
        [[i, j, k] for i in range(n_cells) for j in range(n_cells) for k in range(n_cells)]
    )
    site_a = ((base[None] + cells[:, None]).reshape(-1, 3)) / n_cells
    site_b = ((base[None] + 0.5 + cells[:, None]).reshape(-1, 3) % n_cells) / n_cells
    L = a * n_cells
    lattice = np.eye(3) * L
    sites = np.vstack([site_a, site_b]) * L
    n_a = len(site_a)
    symbols = [species[0]] * n_a + [species[1]] * len(site_b)
    q = np.array([charges[0]] * n_a + [charges[1]] * len(site_b))
    ksp = np.array([k_spring[0]] * n_a + [k_spring[1]] * len(site_b))  # eV/A^2
    m = np.array([ATOMIC_MASS[s] for s in symbols])  # amu
    # units: A, fs, amu; energy unit amu A^2 fs^-2 = AMU * 1e10 J
    e_unit = AMU * 1e10
    kT = KB * temperature / e_unit
    kk = ksp * 1.602176634e-19 / e_unit
    g = gamma_ps * 1e-3  # 1/fs
    c1 = np.exp(-g * dt_fs)
    c2 = np.sqrt((1 - c1**2) * kT / m)[:, None]
    x = rng.normal(size=sites.shape) * np.sqrt(kT / kk)[:, None]
    v = rng.normal(size=sites.shape) * np.sqrt(kT / m)[:, None]
    out = np.empty((n_steps, len(m), 3))
    for s in range(n_steps):
        v += 0.5 * dt_fs * (-kk[:, None] * x) / m[:, None]
        x += 0.5 * dt_fs * v
        v = c1 * v + c2 * rng.normal(size=v.shape)
        x += 0.5 * dt_fs * v
        v += 0.5 * dt_fs * (-kk[:, None] * x) / m[:, None]
        out[s] = sites + x
    from akitsushima.mdspec.constants import ANGSTROM, E_CHARGE, EPS0

    d_eps = (
        np.sum(q**2 / (ksp * 1.602176634e-19 / ANGSTROM**2))
        * E_CHARGE**2
        / (EPS0 * L**3 * ANGSTROM**3)
    )
    return out, lattice, symbols, m, q, d_eps


@pytest.fixture(scope="session")
def crystal_traj():
    """A cached Langevin trajectory (8 + 8 ions, 40 ps at 1 fs)."""
    return langevin_crystal()


# akitsushima's slow marker (ADR-0028 addendum 4: 0.5 s or more, or sharing
# a costly module / session fixture); the post-edit hook runs -m "not slow".
# Measured on the port, 2026-11-25 (ADR-0051).
_SLOW_TESTS = {
    "test_atom_order_point_charges",
    "test_end_to_end_cli",
    "test_rotation_raw_estimator",
    "test_force_estimator_exact_with_unequal_mode_energies",
    "test_rotation_and_order_mode_estimator",
    "test_phonopy_qpoints_matches_npz",
    "test_degenerate_modes_basis_independent",
    "test_mode_split_exact_with_correlated_modes",
    "test_mode_resolved_spectra_harmonic",
    "test_mode_estimator_exact_with_unequal_mode_energies",
    "test_kk_maclaurin_lorentz",
    "test_per_run_temperature",
    "test_independent_runs_write_per_run_outputs",
    "test_fsum_warns_when_modes_show_several_ir_modes",
    "test_ou_process_debye_static_and_peak",
    "test_force_estimator_from_outcar_with_nblock",
    "test_check_flags_non_equipartition_and_drift",
}
_COSTLY_FIXTURES = {"crystal_traj", "two_runs", "crystal", "harmonic", "small_crystal"}


def pytest_collection_modifyitems(config, items):
    here = str(Path(__file__).parent)
    for item in items:
        if not str(item.fspath).startswith(here):
            continue
        if item.originalname in _SLOW_TESTS or _COSTLY_FIXTURES & set(item.fixturenames):
            item.add_marker(pytest.mark.slow)
