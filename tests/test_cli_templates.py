"""Every ``--dump-default`` template must run as shipped.

ADR-0003, T1.  Each template is taken verbatim from ``_DUMP_MAP``; the only
edits are (a) the input files it names are created in a temporary working
directory, sized to satisfy that template's own potential and cutoff, and
(b) step/cycle counts are shrunk so the run takes seconds.  Every other key
-- potential parameters, cutoffs, grids, thermostat settings -- is left
exactly as a user would receive it, so a template key the loader does not
read, or a value the solver rejects, fails here.

The ``tests/test_cli_smoke.py`` suite exercises the same commands with
hand-written configs; this module is the one that holds the templates
themselves to account.
"""

import numpy as np
import pytest
import yaml
from test_cli_smoke import _ns

from akitsushima.cli.parser import _DISPATCH
from akitsushima.cli.templates import _DUMP_MAP
from akitsushima.io.poscar import write_poscar

# Template key -> CLI command.  The md aliases all dispatch to cmd_md.
_COMMAND = {
    "relax": "relax",
    "md": "md",
    "nvt": "md",
    "nve": "md",
    "npt": "md",
    "gcmc": "gcmc",
    "neb": "neb",
    "tsa": "tsa",
    "osmc": "osmc",
    "esm": "esm",
    "phonon": "phonon",
    "kappa": "kappa",
    "mlip": "mlip",
}

# Only step/cycle counts are overridden (section -> {key: value}).
_SHRINK = {
    "relax": {"relax": {"max_steps": 5}},
    "md": {"md": {"n_steps": 20, "equil_steps": 5, "velocity_rescale_steps": 5}},
    "nvt": {"md": {"n_steps": 20, "equil_steps": 5, "velocity_rescale_steps": 5}},
    "nve": {"md": {"n_steps": 20}},
    "npt": {"md": {"n_steps": 20, "equil_steps": 5, "velocity_rescale_steps": 5}},
    "gcmc": {"gcmc": {"n_cycles": 3}},
    "neb": {"neb": {"max_steps": 5}},
    "tsa": {"tsa": {"irc_max_steps": 5}},
    "osmc": {"osmc": {"n_cycles": 3}},
    "esm": {"esm": {"relax": {"max_steps": 2}}},
    "phonon": {},
    "kappa": {"kappa": {"n_steps": 200, "equil_steps": 20, "tau_max": 0.02}},
    "mlip": {},
}


def test_every_template_key_is_covered():
    assert set(_COMMAND) == set(_DUMP_MAP)
    assert set(_SHRINK) == set(_DUMP_MAP)


# ---------------------------------------------------------------- inputs


def _diamond_si_conventional(n: int, a: float = 5.431):
    """n x n x n supercell of the 8-atom conventional diamond cell."""
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    frac = np.vstack([base, base + 0.25])
    cells = np.array(
        [[i, j, k] for i in range(n) for j in range(n) for k in range(n)], dtype=float
    )
    frac = (frac[None, :, :] + cells[:, None, :]).reshape(-1, 3) / n
    lattice = np.eye(3) * a * n
    return lattice, frac @ lattice


def _write_si(path, n=2, jitter=0.0, seed=0):
    lattice, pos = _diamond_si_conventional(n)
    if jitter:
        pos = pos + np.random.default_rng(seed).uniform(-jitter, jitter, pos.shape)
    write_poscar(str(path), lattice, pos, ["Si"] * len(pos))


def _write_si_primitive(path, a=5.431):
    lattice = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
    frac = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    write_poscar(str(path), lattice, frac @ lattice, ["Si", "Si"])


def _write_si_vacancy_hop(initial, final):
    """64-atom Si minus atom 0; the final state moves a first neighbour of
    the vacancy onto the vacant site."""
    lattice, pos = _diamond_si_conventional(2)
    vac = pos[0]
    rest = pos[1:]
    d = rest - vac
    d -= np.round(d @ np.linalg.inv(lattice)) @ lattice
    hop = int(np.argmin(np.linalg.norm(d, axis=1)))
    write_poscar(str(initial), lattice, rest, ["Si"] * len(rest))
    moved = rest.copy()
    moved[hop] = vac
    write_poscar(str(final), lattice, moved, ["Si"] * len(moved))


def _write_ar_gas(path, box=22.0, n=10, seed=0):
    """Dilute Ar in a box wider than 2 x (r_cut 10 + skin 0.5) = 21 A."""
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < n:
        p = rng.uniform(0, box, 3)
        if all(np.linalg.norm(p - q) > 4.0 for q in pos):
            pos.append(p)
    write_poscar(str(path), np.eye(3) * box, np.array(pos), ["Ar"] * n)


def _write_mgo_slab(path, n_xy=8, a=4.21, vacuum=30.0):
    """Two-layer MgO(001) slab, n_xy x n_xy conventional cells in plane.
    8 x 8 is the smallest the esm template's r_cut = 15 A allows (its own
    comment: every perpendicular width must exceed 31 A)."""
    layer_mg = np.array([[0, 0], [0.5, 0.5]])
    layer_o = np.array([[0.5, 0], [0, 0.5]])
    pos, types = [], []
    for iz in range(2):
        mg, o = (layer_mg, layer_o) if iz == 0 else (layer_o, layer_mg)
        for i in range(n_xy):
            for j in range(n_xy):
                for fx, fy in mg:
                    pos.append([(i + fx) * a, (j + fy) * a, iz * a / 2])
                    types.append("Mg")
                for fx, fy in o:
                    pos.append([(i + fx) * a, (j + fy) * a, iz * a / 2])
                    types.append("O")
    pos = np.array(pos)
    lz = a / 2 + vacuum
    # ESMCalculator's z-range is [-Lz/2, +Lz/2), centred on z = 0, and the
    # CLI passes POSCAR Cartesian z through unshifted.  A slab centred in
    # the cell the usual POSCAR way (z ~ Lz/2) is rejected, so the slab is
    # centred on z = 0 instead.
    pos[:, 2] -= a / 4
    order = np.argsort([0 if t == "Mg" else 1 for t in types], kind="stable")
    lattice = np.diag([n_xy * a, n_xy * a, lz])
    write_poscar(str(path), lattice, pos[order], [types[k] for k in order])


def _write_mlip_csv(path, n_struct=20, n_feat=8, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n_feat)
    lines = ["target,label,weight," + ",".join(f"f{k}" for k in range(n_feat))]
    for _ in range(n_struct):
        x = rng.normal(size=n_feat)
        lines.append(f"{np.tanh(x @ w):.10f},E,1.0," + ",".join(f"{v:.10f}" for v in x))
    path.write_text("\n".join(lines) + "\n")


def _prepare_inputs(key: str, cfg: dict, tmp_path) -> None:
    """Create exactly the files the template names, in the working dir."""
    st = cfg.get("structure", {})
    if key == "relax":
        _write_si(tmp_path / st["poscar"], jitter=0.05)
    elif key in ("md", "nvt", "nve", "npt"):
        _write_si(tmp_path / st["poscar"])
    elif key in ("gcmc", "osmc"):
        _write_ar_gas(tmp_path / st["poscar"])
    elif key == "neb":
        _write_si_vacancy_hop(
            tmp_path / st["poscar_initial"], tmp_path / st["poscar_final"]
        )
    elif key == "tsa":
        _write_si(tmp_path / st["poscar"], jitter=0.05)
    elif key == "esm":
        _write_mgo_slab(tmp_path / st["poscar"])
    elif key == "phonon":
        _write_si_primitive(tmp_path / st["poscar"])
    elif key == "kappa":
        _write_si(tmp_path / st["poscar"])
    elif key == "mlip":
        _write_mlip_csv(tmp_path / st["training_csv"])
    else:  # pragma: no cover - guarded by test_every_template_key_is_covered
        raise KeyError(key)


def _merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict):
            assert isinstance(dst.get(k), dict), (
                f"shrink key {k!r} is not a section in the template"
            )
            _merge(dst[k], v)
        else:
            assert k in dst, (
                f"shrink key {k!r} is not in the template -- the override "
                f"would add a key instead of shrinking one"
            )
            dst[k] = v


@pytest.mark.parametrize("key", sorted(_DUMP_MAP))
def test_template_runs_as_shipped(key, tmp_path, monkeypatch):
    cfg = yaml.safe_load(_DUMP_MAP[key])
    _merge(cfg, _SHRINK[key])
    monkeypatch.chdir(tmp_path)
    _prepare_inputs(key, cfg, tmp_path)
    yaml_path = tmp_path / f"{key}.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg))

    if key == "osmc":
        # disabled until rebuilt (ADR-0003 T7, O2): the shipped template
        # must reach the refusal, not an earlier error
        with pytest.raises(ValueError, match="disabled"):
            _DISPATCH[_COMMAND[key]](_ns(yaml_path))
        return
    rc = _DISPATCH[_COMMAND[key]](_ns(yaml_path))
    assert rc == 0, f"template {key!r} returned {rc}"

    section = cfg.get(_COMMAND[key], {})
    out_dir = tmp_path / section.get("output_dir", "")
    assert out_dir.is_dir() and any(out_dir.iterdir()), (
        f"template {key!r} wrote nothing to {out_dir}"
    )
