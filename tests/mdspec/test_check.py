"""`akitsushima mdspec --check` diagnostics (nozuchi ADR-0010 F3)."""

import numpy as np
import pytest

from akitsushima.mdspec import check as chk
import argparse

from akitsushima.cli.cmd_mdspec import cmd_mdspec
from akitsushima.mdspec import yaml_io

from .test_modes import ENERGY, _cfg, _harmonic_crystal


@pytest.fixture(scope="module")
def small_crystal(tmp_path_factory):
    d = tmp_path_factory.mktemp("check")
    _harmonic_crystal(d, n_frames=6000)
    return d


def _status(rep, start):
    return [x["status"] for x in rep["items"] if x["item"].startswith(start)]


def test_check_flags_non_equipartition_and_drift(small_crystal, tmp_path):
    d = small_crystal
    # an OSZICAR whose conserved energy drifts by 0.5 eV over 6 ps for 6 atoms at 300 K
    n = 6000
    e = -100.0 + 0.5 * np.arange(n) / n
    (tmp_path / "OSZICAR").write_text(
        "".join(
            f"{i + 1:7d} T=   300. E= {x:.8E} F= {x:.8E} E0= {x:.8E}\n" for i, x in enumerate(e)
        )
    )
    rep = chk.check(_cfg(d, "auto"), log=lambda *a: None, energy_files=[tmp_path / "OSZICAR"])
    # energies 0.3-2.5 kT (mean 1.13): a canonical-looking spread, so the mean test passes
    kin = next(x for x in rep["items"] if x["item"].startswith("Gamma-mode"))
    assert kin["status"] == "OK" and kin["value"] == pytest.approx(np.mean(ENERGY), rel=0.03)
    assert "0.30-2.50" in kin["note"]
    assert _status(rep, "energy drift") == ["WARN"]
    assert _status(rep, "centre-of-mass") == ["OK"]  # pure optical motion
    assert _status(rep, "dipole ACF tail") == ["OK"]
    assert rep["n_runs"] == 1 and rep["n_warn"] >= 1


def test_check_mean_temperature_vs_tebeg(small_crystal, tmp_path):
    # a VASP-like run dir: <T> 60 K for TEBEG = 300 K (6 atoms: 3 sigma = 71 %, offset -80 %)
    (tmp_path / "INCAR").write_text("TEBEG = 300.0\n")
    (tmp_path / "OUTCAR").write_text(
        "".join(f" kinetic energy EKIN   = 1.0  (temperature  {t:.2f} K)\n" for t in [60.0] * 50)
    )
    cfg = _cfg(small_crystal, "auto")
    cfg["temperature_source"] = [str(tmp_path / "OUTCAR")]
    rep = chk.check(cfg, log=lambda *a: None)
    item = next(x for x in rep["items"] if x["item"].startswith("mean T vs INCAR TEBEG"))
    assert item["status"] == "WARN" and item["value"] == pytest.approx(-0.8)


def test_read_energy_formats(tmp_path):
    (tmp_path / "OSZICAR").write_text(
        "      1 T=    95. E= -.77149944E+03 F= -.77306285E+03 E0= -.77306285E+03  EK= 0.15634E+01\n"
        "      2 T=    96. E= -.77149985E+03 F= -.77308160E+03 E0= -.77308160E+03  EK= 0.15818E+01\n"
    )
    assert np.allclose(chk.read_energy(tmp_path / "OSZICAR"), [-771.49944, -771.49985])
    (tmp_path / "thermo.log").write_text(
        "Step Time(ps) E_pot(eV) E_kin(eV) E_tot(eV) T_inst(K)\n"
        "0 0.0 -10.0 1.0 -9.0 300.0\n10 0.01 -10.1 1.1 -9.0 301.0\n"
    )
    assert np.allclose(chk.read_energy(tmp_path / "thermo.log"), [-9.0, -9.0])


def test_cli_check_writes_json_and_exit_code(small_crystal, tmp_path):
    d = small_crystal
    (d / "in.yaml").write_text(
        "trajectory: {files: [XDATCAR], timestep_fs: 1.0, frame_every: 1}\n"
        "temperature: 300.0\n"
        "dipole: {model: point_charge, charges: {Mg: 2.0, O: -1.0}}\n"
        "epsilon_inf: 1.0\n"
        "modes: {file: modes.npz, unit_cell: POSCAR_unit, supercell: [2, 1, 1]}\n"
        "spectrum: {max_lag_ps: 2.0, kk_check: false}\n"
        "output: {dir: chk_out, prefix: c, plot: false}\n"
    )
    # the nozuchi input above, converted to the akitsushima YAML
    (d / "aki.yaml").write_text(yaml_io.convert_file(d / "in.yaml"))
    code = cmd_mdspec(argparse.Namespace(yaml=str(d / "aki.yaml"), quiet=True, check=True,
                                         convert_nozuchi=False, energy=None, target=0.02))
    assert code in (0, 1)
    assert (d / "chk_out" / "c_check.json").exists()
    assert not (d / "chk_out" / "c_eps.dat").exists()  # no permittivity output
