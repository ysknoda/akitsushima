"""Regressions for the PhononCalculator defects found by ADR-0003 T4.

Found by comparing `akitsushima phonon` with phonopy on Tersoff Si (diamond,
2-atom cell) and Tersoff AlN (wurtzite, 4-atom cell):

P1  D(q) used each supercell atom's raw cell offset as the Fourier phase,
    exact only for q commensurate with the supercell.
P2  eigvalsh(D.real) discarded the imaginary part of the Hermitian D(q).
P3  phonon_free_energy read the cm^-1 DOS axis as THz (x33.4 in omega).
P4  debye_temperature_from_vsound read cm^-1 slopes as THz (x33.4 in v).
P5  reduced -> Cartesian q used the columns of the reciprocal matrix.
P6  the acoustic-sum-rule correction ignored the mass weighting of D.
P7  v_T was taken from the pooled slopes of all directions, i.e. from the
    longitudinal slopes of the other directions.

The phonopy comparison is the oracle that found them; the other tests are
phonopy-free consistency checks that each fail on the old code.
"""

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from akitsushima.engine import TersoffPotential
from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml
from akitsushima.solvers.phonon import PhononCalculator

_SI = {
    "A": 1830.8,
    "B": 471.18,
    "lambda1": 2.4799,
    "lambda2": 1.7322,
    "lambda3": 1.7322,
    "beta": 1.1e-6,
    "n": 0.78734,
    "c": 100390.0,
    "d": 16.218,
    "h": -0.59826,
    "R": 2.85,
    "D": 0.15,
}

# AlN Tersoff: the example/AlN parameter set (i, j, k records).
_ALN_COMMON = {
    "A": 1258.567263,
    "B": 453.228512,
    "lambda1": 2.434869,
    "lambda2": 1.717680,
    "R": 2.55,
    "D": 0.15,
}
_ALN = [
    dict(
        i="Al",
        j="N",
        k="N",
        beta=2.133047e-6,
        n=0.598233,
        lambda3=1.186759,
        c=19110.741778,
        d=10.882090,
        h=-0.972662,
        **_ALN_COMMON,
    ),
    dict(
        i="N",
        j="Al",
        k="Al",
        beta=2.496023e-6,
        n=1.220882,
        lambda3=1.100709,
        c=27568.039128,
        d=8.292436,
        h=-0.816226,
        **_ALN_COMMON,
    ),
]

A_SI = 5.431988  # Tersoff Si equilibrium (energy-only minimum)
THZ_TO_CM = 33.35640951981521


def _fn(pot):
    def f(lat, pos, types):
        return pot.compute(lat, pos, types)

    f.r_cut = 3.0
    return f


def _si_calc(sc):
    lat = 0.5 * A_SI * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    pos = np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ lat
    calc = PhononCalculator(
        lattice=lat,
        positions=pos,
        atom_types=["Si"] * 2,
        force_fn=_fn(TersoffPotential(params=_SI, species=["Si"])),
        supercell=sc,
        delta=0.01,
    )
    with redirect_stdout(io.StringIO()):
        calc.compute_ifc(verbose=False)
    return calc


def _aln_calc(sc=(4, 4, 3)):
    a, c, u = 3.381244, 5.521549, 0.375
    lat = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    frac = np.array(
        [
            [1 / 3, 2 / 3, 0],
            [2 / 3, 1 / 3, 0.5],
            [1 / 3, 2 / 3, u],
            [2 / 3, 1 / 3, 0.5 + u],
        ]
    )
    fn = _build_force_fn_from_yaml({"type": "Tersoff", "species": ["Al", "N"],
                                    "params": _ALN})[0]
    calc = PhononCalculator(
        lattice=lat,
        positions=frac @ lat,
        atom_types=["Al", "Al", "N", "N"],
        force_fn=fn,
        supercell=list(sc),
        delta=0.01,
    )
    with redirect_stdout(io.StringIO()):
        calc.compute_ifc(verbose=False)
    return calc


def _freqs(calc, q_cart):
    return calc._eig_to_freq(
        np.sort(np.linalg.eigvalsh(calc.build_dynamical_matrix(q_cart)))
    )


def _recip(calc):
    return 2 * np.pi * np.linalg.inv(calc.lattice_prim).T


@pytest.fixture(scope="module")
def si4():
    return _si_calc([4, 4, 4])


@pytest.fixture(scope="module")
def aln():
    return _aln_calc()


def test_interpolated_frequencies_do_not_depend_on_supercell(si4):
    """P1: Tersoff Si force constants reach well inside a 4x4x4 supercell,
    so a correct Fourier interpolation gives the same D(q) from 4x4x4 and
    5x5x5 at a q commensurate with neither.  The old raw-offset phases
    differed by several THz here."""
    si5 = _si_calc([5, 5, 5])
    q = np.array([0.13, 0.13, 0.13]) @ _recip(si4)
    np.testing.assert_allclose(_freqs(si4, q), _freqs(si5, q), atol=0.05)


def test_band_structure_uses_the_complex_dynamical_matrix(si4):
    """P2: at q = (1/4, 0, 0) -- commensurate, so P1 plays no part -- D is
    complex (|Im D| ~ 0.13) and D.real gave frequencies 2.4 THz off."""
    q_red = [0.25, 0.0, 0.0]
    D = si4.build_dynamical_matrix(np.array(q_red) @ _recip(si4))
    assert np.abs(D.imag).max() > 0.05
    res = si4.compute_band_structure(
        [("G", [0, 0, 0]), ("Q", q_red)], n_points=4, verbose=False
    )
    np.testing.assert_allclose(
        res["frequencies"][-1], _freqs(si4, np.array(q_red) @ _recip(si4)), atol=1e-6
    )


def test_free_energy_matches_a_direct_mode_sum(si4):
    """P3: F_vib from the smeared DOS must equal the Bose-Einstein mode sum
    over the same q-mesh; the old THz/cm^-1 mix-up made it ~35x too big."""
    hbar_kB = 7.638232577e-12  # hbar/k_B [K s]
    T = 300.0
    with redirect_stdout(io.StringIO()):
        F_code = si4.phonon_free_energy(T=T, n_q=8, verbose=False)
    B = _recip(si4)
    total = 0.0
    n = 8
    for i0 in range(n):
        for i1 in range(n):
            for i2 in range(n):
                q = (np.array([i0, i1, i2]) + 0.5) / n - 0.5
                nu_cm = _freqs(si4, q @ B)
                w = nu_cm[nu_cm > 0.01 * THZ_TO_CM] / THZ_TO_CM * 2e12 * np.pi
                x = hbar_kB * w / T
                total += np.sum(x / 2 + np.log1p(-np.exp(-x)))
    F_direct = 8.617333262e-5 * T * total / n**3
    assert F_code == pytest.approx(F_direct, rel=0.01)


def test_debye_velocities_are_physical_and_distinguish_l_from_t(si4):
    """P4 + P7: sound velocities of Tersoff Si are a few km/s (the old code
    gave 9.3e5 m/s), and v_T < v_L (the old pooling made them equal)."""
    out = io.StringIO()
    with redirect_stdout(out):
        theta = si4.debye_temperature_from_vsound(verbose=True)
    import re

    vL, vT = (
        float(x)
        for x in re.search(
            r"v_L = (\d+) m/s,\s+v_T = (\d+) m/s", out.getvalue()
        ).groups()
    )
    assert 5000 < vL < 12000
    assert vT < 0.8 * vL
    assert 200 < theta < 1000


def test_hexagonal_path_points_are_the_reduced_points(aln):
    """P5: M = (1/2, 0, 0) on a hexagonal lattice must be q_red @ B (rows of
    B are the reciprocal vectors), not B @ q_red."""
    res = aln.compute_band_structure(
        [("G", [0, 0, 0]), ("M", [0.5, 0, 0])], n_points=2, verbose=False
    )
    q_m = np.array([0.5, 0, 0]) @ _recip(aln)
    np.testing.assert_allclose(res["q_points"][-1], q_m, atol=1e-12)
    np.testing.assert_allclose(res["frequencies"][-1], _freqs(aln, q_m), atol=1e-6)


def test_asr_correction_leaves_optical_modes_alone_for_unequal_masses(aln):
    """P6: the IFC already satisfies the sum rule, so the Gamma correction
    must not move the AlN optical modes (the unweighted version shifted
    them by up to 1.8 THz)."""
    raw = aln._eig_to_freq(np.sort(np.linalg.eigvalsh(aln._build_D_raw(np.zeros(3)))))
    np.testing.assert_allclose(_freqs(aln, np.zeros(3))[3:], raw[3:], atol=0.05)


def test_matches_phonopy_at_general_q(aln):
    """The oracle itself, when phonopy is installed: every AlN frequency at
    a general (non-commensurate, complex-D) q within 0.01 THz."""
    phonopy = pytest.importorskip("phonopy")
    from phonopy.structure.atoms import PhonopyAtoms

    lat = aln.lattice_prim
    frac = aln.positions @ np.linalg.inv(lat)
    unit = PhonopyAtoms(
        symbols=list(aln.atom_types),
        cell=lat,
        scaled_positions=frac,
        masses=list(aln.masses),
    )
    phn = phonopy.Phonopy(
        unit, supercell_matrix=np.diag(aln.supercell), primitive_matrix=None
    )
    phn.generate_displacements(distance=0.01)
    fn = aln.force_fn
    phn.forces = np.array(
        [
            fn(np.array(sc.cell), np.array(sc.positions), list(sc.symbols))[1]
            for sc in phn.supercells_with_displacements
        ]
    )
    phn.produce_force_constants()
    q_red = np.array([0.17, 0.08, 0.21])
    ref = np.sort(phn.get_frequencies(q_red))
    got = _freqs(aln, q_red @ _recip(aln)) / THZ_TO_CM
    np.testing.assert_allclose(got, ref, atol=0.01)
