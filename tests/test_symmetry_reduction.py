"""Symmetry reduction of phonon displacements and elastic strains
(ADR-0040).

- Cubic (diamond Si): the reduced IFC and C equal the full method to
  round-off.
- Hexagonal (wurtzite AlN, Tersoff): the reduced results differ from the
  full method by the O(delta^2) / O(s^2) finite-difference term, which
  shrinks four-fold when the step is halved, and phonopy agrees.
- P1 (rattled cell): only the identity is found and the result is the
  full method's bit for bit.
- Refusals and the CLI keys.
"""

import io
import json
from contextlib import redirect_stdout

import numpy as np
import pytest

pytest.importorskip("spglib")

from test_phonon_regressions import _ALN, _SI, THZ_TO_CM

from akitsushima.solvers.config_bridge import _build_force_fn_from_yaml
from akitsushima.solvers.elastic import ElasticConstantsCalculator
from akitsushima.solvers.phonon import PhononCalculator
from akitsushima.solvers.symmetry import spglib_operations

A_SI = 5.431988  # Tersoff Si equilibrium (as test_phonon_regressions)


def _si_fn():
    return _build_force_fn_from_yaml(
        {"type": "Tersoff", "species": ["Si"], "params": _SI}
    )[0]


def _aln_fn():
    return _build_force_fn_from_yaml(
        {"type": "Tersoff", "species": ["Al", "N"], "params": _ALN}
    )[0]


def _si_primitive():
    lat = 0.5 * A_SI * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    return lat, np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ lat, ["Si"] * 2


def _aln_primitive():
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
    return lat, frac @ lat, ["Al", "Al", "N", "N"]


def _supercell(lat, pos, types, S):
    frac = pos @ np.linalg.inv(lat)
    S = np.asarray(S)
    f = np.array(
        [
            (x + [i, j, k]) / S
            for i in range(S[0])
            for j in range(S[1])
            for k in range(S[2])
            for x in frac
        ]
    )
    latS = lat * S[:, None]
    return latS, f @ latS, list(types) * int(np.prod(S))


def _phonons(lat, pos, types, fn, sc, symmetry, delta=0.01):
    pc = PhononCalculator(
        lat, pos, types, fn, supercell=sc, delta=delta, symmetry=symmetry
    )
    with redirect_stdout(io.StringIO()):
        pc.compute_ifc(verbose=False)
    return pc


def _freqs(pc, q_red):
    rec = 2 * np.pi * np.linalg.inv(pc.lattice_prim).T
    return np.concatenate(
        [
            pc._eig_to_freq(
                np.sort(
                    np.linalg.eigvalsh(pc.build_dynamical_matrix(np.asarray(q) @ rec))
                )
            )
            for q in q_red
        ]
    )


# General q-points (no Gamma: its acoustic modes are sqrt of round-off).
_Q = [[0.5, 0.0, 0.0], [1 / 3, 1 / 3, 0.0], [0.1, 0.2, 0.3], [0.0, 0.0, 0.5]]


# --------------------------------------------------------------------------- #
#  Operations                                                                  #
# --------------------------------------------------------------------------- #


def test_operations_and_atom_mapping():
    """Fd-3m (48 operations) for diamond, P6_3mc (12) for wurtzite; every
    operation maps each atom onto perm + shift exactly, also for positions
    given outside the unit cell."""
    for (lat, pos, types), symbol, n in (
        (_si_primitive(), "Fd-3m", 48),
        (_aln_primitive(), "P6_3mc", 12),
    ):
        pos = pos + np.array([1.0, -2.0, 1.0]) @ lat  # unwrapped input
        ops = spglib_operations(lat, pos, types)
        assert (ops.symbol, ops.n_ops) == (symbol, n)
        f = pos @ np.linalg.inv(lat)
        for k in range(ops.n_ops):
            np.testing.assert_allclose(
                f @ ops.rot_frac[k].T + ops.trans[k],
                f[ops.perm[k]] + ops.shift[k],
                atol=1e-12,
            )
            R = ops.rot_cart[k]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_unequal_masses_of_one_species_break_symmetry():
    lat, pos, types = _si_primitive()
    assert spglib_operations(lat, pos, types).n_ops == 48
    ops = spglib_operations(lat, pos, types, masses=np.array([28.0, 29.0]))
    assert ops.n_ops < 48


# --------------------------------------------------------------------------- #
#  Phonons                                                                     #
# --------------------------------------------------------------------------- #


def test_phonon_cubic_equals_full_method():
    """Diamond Si: one force evaluation instead of 12, IFC equal to 1e-10."""
    lat, pos, types = _si_primitive()
    fn = _si_fn()
    full = _phonons(lat, pos, types, fn, 2, False)
    red = _phonons(lat, pos, types, fn, 2, True)
    info = red.symmetry_info
    assert (info["n_force_calls"], info["n_force_calls_full"]) == (1, 12)
    assert full.symmetry_info is None
    scale = np.abs(full.IFC).max()
    assert np.abs(red.IFC - full.IFC).max() < 1e-10 * scale
    np.testing.assert_allclose(_freqs(red, _Q), _freqs(full, _Q), rtol=1e-9, atol=1e-6)


@pytest.mark.slow
def test_phonon_hexagonal_difference_is_the_delta_squared_term():
    """Wurtzite AlN: 6 force evaluations instead of 24.  Rotations by 120
    degrees give the central differences of other directions, so the IFC
    differs from the full method by O(delta^2): four times less at half
    the step, and below the full method's own change with delta."""
    lat, pos, types = _aln_primitive()
    fn = _aln_fn()
    sc = [3, 3, 2]
    d = {}
    for delta in (0.01, 0.005):
        full = _phonons(lat, pos, types, fn, sc, False, delta)
        red = _phonons(lat, pos, types, fn, sc, True, delta)
        assert (
            red.symmetry_info["n_force_calls"],
            red.symmetry_info["n_force_calls_full"],
        ) == (6, 24)
        d[delta] = (np.abs(red.IFC - full.IFC).max(), full)
    ratio = d[0.01][0] / d[0.005][0]
    assert 3.5 < ratio < 4.5, ratio
    own = np.abs(d[0.01][1].IFC - d[0.005][1].IFC).max()
    assert d[0.01][0] < own


@pytest.mark.slow
def test_phonon_hexagonal_matches_phonopy():
    """The reduced AlN frequencies agree with phonopy's (its own
    displacements and symmetrisation, the same force function) within
    0.01 THz at general q."""
    phonopy = pytest.importorskip("phonopy")
    from phonopy.structure.atoms import PhonopyAtoms

    lat, pos, types = _aln_primitive()
    fn = _aln_fn()
    sc = [3, 3, 2]
    red = _phonons(lat, pos, types, fn, sc, True)
    unit = PhonopyAtoms(
        symbols=types,
        cell=lat,
        scaled_positions=pos @ np.linalg.inv(lat),
        masses=list(red.masses),
    )
    phn = phonopy.Phonopy(unit, supercell_matrix=np.diag(sc), primitive_matrix=None)
    phn.generate_displacements(distance=0.01)
    phn.forces = np.array(
        [
            fn(np.array(s.cell), np.array(s.positions), list(s.symbols))[1]
            for s in phn.supercells_with_displacements
        ]
    )
    phn.produce_force_constants()
    for q in ([0.17, 0.08, 0.21], [0.5, 0.0, 0.0], [1 / 3, 1 / 3, 0.5]):
        ref = np.sort(phn.get_frequencies(q))
        got = _freqs(red, [q]) / THZ_TO_CM
        np.testing.assert_allclose(got, ref, atol=0.01)


@pytest.mark.slow
def test_identity_only_is_the_full_method_bit_for_bit():
    """A rattled 8-atom Si cell is P1 (two atoms would keep an inversion
    centre): the plan falls back to the full method, so the IFC is
    identical and the count is the full one."""
    conv = A_SI * np.eye(3)
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    pos = np.vstack([basis, basis + 0.25]) @ conv
    pos = pos + np.random.default_rng(3).normal(0.0, 0.02, pos.shape)
    types = ["Si"] * 8
    fn = _si_fn()
    full = _phonons(conv, pos, types, fn, 2, False)
    red = _phonons(conv, pos, types, fn, 2, True)
    assert red.symmetry_info["number"] == 1
    assert red.symmetry_info["n_force_calls"] == 48
    assert np.array_equal(red.IFC, full.IFC)


# --------------------------------------------------------------------------- #
#  Elastic constants                                                           #
# --------------------------------------------------------------------------- #


def _si64():
    conv = A_SI * np.eye(3)
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    frac8 = np.vstack([basis, basis + 0.25])
    return _supercell(conv, frac8 @ conv, ["Si"] * 8, [2, 2, 2])


@pytest.mark.slow
def test_elastic_cubic_equals_full_method():
    """Diamond Si (64 atoms): modes 1 and 4 only, the -s of the shear an
    image; 10 strained cells instead of 37.  C, C_clamped and the internal
    strain equal the full method to round-off, and C is exactly cubic."""
    lat, pos, types = _si64()
    fn = _si_fn()
    kw = {"fmax": 1e-5, "max_steps": 5000}
    full = ElasticConstantsCalculator(lat, pos, types, fn, **kw).compute()
    calc = ElasticConstantsCalculator(lat, pos, types, fn, symmetry=True, **kw)
    red = calc.compute()
    info = red["symmetry"]
    assert info["modes_computed"] == [1, 4]
    assert (info["n_cells"], info["n_cells_full"]) == (10, 37)
    assert len(calc._points()) == 10 and "symmetry" not in full
    for key in ("C", "C_clamped"):
        np.testing.assert_allclose(
            red[key], full[key], rtol=0, atol=1e-10 * np.abs(full[key]).max()
        )
    np.testing.assert_allclose(
        red["internal_strain"], full["internal_strain"], rtol=0, atol=1e-9
    )
    C = red["C"]
    assert np.ptp(np.diag(C)[:3]) < 1e-10 and np.ptp(np.diag(C)[3:]) < 1e-10
    assert np.isnan(red["fit_rms_GPa"]["relaxed"][[1, 2, 4, 5]]).all()
    assert np.isfinite(red["fit_rms_GPa"]["relaxed"][[0, 3]]).all()


@pytest.mark.slow
def test_elastic_hexagonal_difference_is_the_strain_squared_term():
    """Wurtzite AlN (72 atoms): modes 1, 3 and 4, 6 strained cells instead
    of 13 per amplitude.  The clamped-ion C differs from the full method by
    O(s^2) (four times less at half the strain); C66 = (C11 - C12) / 2
    holds exactly."""
    lat, pos, types = _supercell(*_aln_primitive(), [3, 3, 2])
    fn = _aln_fn()
    d = {}
    for s in (0.01, 0.005):
        kw = {"strains": [s], "relax_ions": False}
        full = ElasticConstantsCalculator(lat, pos, types, fn, **kw).compute()
        red = ElasticConstantsCalculator(
            lat, pos, types, fn, symmetry=True, **kw
        ).compute()
        assert red["symmetry"]["modes_computed"] == [1, 3, 4]
        assert (red["symmetry"]["n_cells"], red["symmetry"]["n_cells_full"]) == (6, 13)
        C = red["C_clamped"]
        assert abs(C[5, 5] - 0.5 * (C[0, 0] - C[0, 1])) < 1e-10 * C[0, 0]
        d[s] = np.abs(red["C_clamped"] - full["C_clamped"]).max()
    assert 3.5 < d[0.01] / d[0.005] < 4.5, d


def test_elastic_identity_only_is_the_full_method_bit_for_bit():
    lat, pos, types = _si64()
    pos = pos + np.random.default_rng(5).normal(0.0, 0.02, pos.shape)
    fn = _si_fn()
    kw = {"strains": [0.005], "relax_ions": False}
    full = ElasticConstantsCalculator(lat, pos, types, fn, **kw).compute()
    red = ElasticConstantsCalculator(lat, pos, types, fn, symmetry=True, **kw).compute()
    assert red["symmetry"]["number"] == 1 and red["symmetry"]["n_cells"] == 13
    for key in ("C", "C_clamped", "internal_strain"):
        assert np.array_equal(red[key], full[key])


# --------------------------------------------------------------------------- #
#  Refusals                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.filterwarnings("ignore:PhononCalculator. supercell side")
def test_refusals(monkeypatch):
    lat, pos, types = _si_primitive()

    def esm_fn(lattice, positions, atom_types, **kw):
        return 0.0, np.zeros_like(positions), np.zeros((3, 3))

    esm_fn.coulomb = "esm"
    esm_fn.r_cut = 3.0
    with pytest.raises(ValueError, match="ESM"):
        PhononCalculator(lat, pos, types, esm_fn, supercell=[2, 2, 1], symmetry=True)

    import sys

    monkeypatch.setitem(sys.modules, "spglib", None)
    with pytest.raises(ImportError, match=r"akitsushima\[symmetry\]"):
        spglib_operations(lat, pos, types)


def test_an_operation_that_does_not_map_the_atoms_is_an_error(monkeypatch):
    import spglib

    lat, pos, types = _si_primitive()
    real = spglib.get_symmetry

    def bogus(cell, symprec=1e-5):
        s = real(cell, symprec=symprec)
        s["translations"] = s["translations"].copy()
        s["translations"][1] += 0.1
        return s

    monkeypatch.setattr(spglib, "get_symmetry", bogus)
    with pytest.raises(ValueError, match="does not map every atom"):
        spglib_operations(lat, pos, types)


def test_cli_symmetry_keys_are_checked():
    from akitsushima.cli.common import _symmetry_kw

    assert _symmetry_kw({}) == {"symmetry": False, "symprec": 1e-5}
    assert _symmetry_kw({"symmetry": True, "symprec": 1e-3})["symprec"] == 1e-3
    with pytest.raises(ValueError, match="true or false"):
        _symmetry_kw({"symmetry": "yes"})
    with pytest.raises(ValueError, match="positive"):
        _symmetry_kw({"symmetry": True, "symprec": 0})


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #

_RESULTS = {
    "phonon": ("phonon", "phonon_results.json"),
    "elastic": ("elastic", "elastic_results.json"),
    "qha": ("qha", "qha_results.json"),
    "gruneisen": ("gruneisen", "gruneisen_results.json"),
}


@pytest.mark.slow
@pytest.mark.parametrize("key", ["phonon", "elastic", "qha", "gruneisen", "kappa"])
def test_cli_templates_run_with_symmetry(key, tmp_path, monkeypatch):
    """Each template with its symmetry key set to true runs, and the
    results file records the space group (kappa: the Slack section's
    PhononCalculator)."""
    import yaml
    from test_cli_smoke import _ns
    from test_cli_templates import _COMMAND, _DISPATCH, _SHRINK, _merge, _prepare_inputs

    from akitsushima.cli.templates import _DUMP_MAP

    cfg = yaml.safe_load(_DUMP_MAP[key])
    _merge(cfg, _SHRINK[key])
    sec = cfg[_COMMAND[key]]
    (sec["slack_debye"] if key == "kappa" else sec)["symmetry"] = True
    if key in ("qha", "gruneisen"):
        # Their variable-cell relaxations leave non-uniform displacements of
        # about 1e-5 A (within fmax), so symprec 1e-5 finds only R-3m
        # (ADR-0040 result); 1e-4 recovers Fd-3m.
        sec["symprec"] = 1e-4
    monkeypatch.chdir(tmp_path)
    _prepare_inputs(key, cfg, tmp_path)
    (tmp_path / f"{key}.yaml").write_text(yaml.safe_dump(cfg))
    seen = []
    if key == "kappa":
        import akitsushima.cli.cmd_kappa as kmod
        real = kmod.PhononCalculator

        class Spy(real):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                seen.append(self)
        monkeypatch.setattr(kmod, "PhononCalculator", Spy)
    with redirect_stdout(io.StringIO()):
        rc = _DISPATCH[_COMMAND[key]](_ns(tmp_path / f"{key}.yaml"))
    assert rc == 0
    if key == "kappa":
        assert seen and seen[0].symmetry_info["spacegroup"] == "Fd-3m"
        return
    _, fname = _RESULTS[key]
    out = json.loads((tmp_path / sec.get("output_dir", "") / fname).read_text())
    sym = out["symmetry"]
    first = sym[0] if isinstance(sym, list) else sym
    assert (first["spacegroup"], first["number"]) == ("Fd-3m", 227)


def test_dielectric_refuses_the_symmetry_key(tmp_path, monkeypatch):
    import yaml
    from test_cli_smoke import _ns
    from test_cli_templates import _prepare_inputs

    import akitsushima.cli.cmd_dielectric as mod
    from akitsushima.cli.templates import _DUMP_MAP

    cfg = yaml.safe_load(_DUMP_MAP["dielectric"])
    cfg["dielectric"]["symmetry"] = True
    monkeypatch.chdir(tmp_path)
    _prepare_inputs("dielectric", cfg, tmp_path)
    (tmp_path / "d.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="not supported"):
        mod.cmd_dielectric(_ns(tmp_path / "d.yaml"))
