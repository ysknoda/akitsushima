"""Global structure search: MD-quench annealing and basin hopping (ADR-0022).

Both methods are drivers around solvers that already exist and are
verified on their own: :class:`NVTSimulation` / :class:`NPTSimulation`
for the dynamics and :class:`GeometryOptimiser` for every quench.  This
module adds no integrator and no optimiser of its own.

md_quench
    One continuous MD trajectory, started from the relaxed input (cycle
    0) and cut into chunks of ``quench_every`` steps.  The temperature is piecewise constant over ``n_segments``
    segments (``schedule``: constant / linear / geometric / list); each
    chunk continues the previous chunk's positions, velocities and cell
    (``initial_velocities``), only the thermostat chain restarts.  After
    every chunk a *copy* of the structure is relaxed and recorded -- the
    inherent structure of that part of the trajectory (Stillinger &
    Weber 1982).  The quench never feeds back into the MD.

basin_hopping
    Wales & Doye, J. Phys. Chem. A 101, 5111 (1997).  Every atom is
    displaced uniformly inside a sphere of radius ``step`` (with
    ``relax_cell``, the cell also gets a random symmetric strain of at
    most ``cell_step``); the trial is relaxed and accepted with the
    Metropolis rule on the relaxed enthalpy at ``temperature``.  Every
    ``adapt_every`` trials the step is multiplied by ``adapt_factor``
    (acceptance below ``target_acceptance``) or divided by it (above).
    A trial whose relaxation did not converge is rejected.

Pressure
    With ``pressure`` != 0 the quench minimises H = E + PV: the force
    function handed to GeometryOptimiser returns E + PV and the stress
    sigma - P (compression positive, ADR-0001), so every relaxation
    method sees dH/d(strain).  NPT MD uses the same pressure, and the
    Metropolis test compares H.

Failures
    An exception in an MD chunk or a quench, a non-finite energy, or an
    energy per atom below ``min_energy_per_atom`` (an MLIP collapsing
    under extrapolation) marks the cycle ``failed``.  The energy checks
    apply to the last frame of every MD chunk as well as to the quench.  md_quench then
    returns to the last good MD state with freshly drawn velocities;
    basin hopping keeps its current structure.  ``max_consecutive_failures``
    failures in a row stop the run.

Restart
    Every cycle appends one JSON line to ``anneal_log.jsonl`` and
    rewrites ``checkpoint.npz`` (atomically).  ``restart=True`` continues
    after the checkpoint's cycle; ledger lines past it are dropped.  The
    walker's random-number state is part of the checkpoint, so a resumed
    run reproduces the uninterrupted one.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable

import numpy as np

from ..constants import EV_ANG3_TO_GPA, K_B
from ..utils.mpi import _mpi_rank
from .geometry_optimiser import GeometryOptimiser, _accepts_neigh_list
from .molecular_dynamics import NPTSimulation, NVTSimulation

__all__ = ["GlobalOptimiser", "enthalpy_force_fn", "temperature_schedule"]

LEDGER = "anneal_log.jsonl"
CHECKPOINT = "checkpoint.npz"


# ----------------------------------------------------------------------
#  helpers
# ----------------------------------------------------------------------
def temperature_schedule(
    kind: str,
    n_segments: int,
    T_start: float = None,
    T_end: float = None,
    temperatures=None,
) -> np.ndarray:
    """Target temperature [K] of each segment.

    constant  : T_start for every segment
    linear    : evenly spaced from T_start to T_end
    geometric : T_start * (T_end / T_start)^(k / (n - 1))
    list      : ``temperatures`` as given (n_segments is its length)
    """
    kind = str(kind).lower()
    if kind == "list":
        T = np.asarray(temperatures, dtype=float)
        if T.ndim != 1 or T.size == 0:
            raise ValueError("schedule list: 'temperatures' must be a non-empty list")
    else:
        n = int(n_segments)
        if n < 1:
            raise ValueError("schedule: n_segments must be >= 1")
        if kind == "constant":
            T = np.full(n, float(T_start))
        elif kind == "linear":
            T = np.linspace(float(T_start), float(T_end), n)
        elif kind == "geometric":
            if T_start <= 0 or T_end <= 0:
                raise ValueError("schedule geometric: T_start and T_end must be > 0")
            T = float(T_start) * (float(T_end) / float(T_start)) ** (
                np.arange(n) / max(n - 1, 1)
            )
        else:
            raise ValueError(
                f"schedule: unknown type {kind!r} "
                "(constant | linear | geometric | list)"
            )
    if np.any(T <= 0) or not np.all(np.isfinite(T)):
        raise ValueError(f"schedule: temperatures must be finite and > 0, got {T}")
    return T


def enthalpy_force_fn(force_fn: Callable, pressure_gpa: float) -> Callable:
    """Wrap ``force_fn`` so it returns (E + PV, F, sigma - P*I).

    Stress is compression-positive (ADR-0001): sigma = -(1/V) dE/d(eps),
    so d(PV)/d(eps_ab) = P V delta_ab gives sigma_H = sigma - P delta_ab.
    Forces are unchanged.  P = 0 returns ``force_fn`` itself.
    """
    if not pressure_gpa:
        return force_fn
    P = float(pressure_gpa) / EV_ANG3_TO_GPA
    voigt_diag = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    def _shift(lat, out):
        e, f, s = out[0], out[1], np.array(out[2], dtype=float)
        V = abs(float(np.linalg.det(lat)))
        if s.shape == (3, 3):
            s = s - P * np.eye(3)
        else:
            s = s - P * voigt_diag
        return (e + P * V, f, s) + tuple(out[3:])

    if _accepts_neigh_list(force_fn):

        def h_fn(lattice, positions, atom_types, **kw):
            return _shift(lattice, force_fn(lattice, positions, atom_types, **kw))
    else:

        def h_fn(lattice, positions, atom_types):
            return _shift(lattice, force_fn(lattice, positions, atom_types))

    return h_fn


def _random_in_sphere(rng, n: int, radius: float) -> np.ndarray:
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    return d * (radius * rng.random(n) ** (1.0 / 3.0))[:, None]


class _CycleFailed(Exception):
    pass


# ----------------------------------------------------------------------
#  driver
# ----------------------------------------------------------------------
class GlobalOptimiser:
    """MD-quench annealing or basin hopping around existing solvers.

    Parameters
    ----------
    lattice, positions, atom_types : starting structure (A).
    force_fn, r_cut : a resolved force function (lat, pos, types) ->
        (E, F, S) and its cutoff, as from ``_resolve_force_fn``.
    method : "md_quench" or "basin_hopping".
    output_dir : everything is written below it.
    relax : keyword arguments for GeometryOptimiser (method, relax_cell,
        fmax, max_steps, ...); output_dir / verbose are set here.
    md : md_quench only -- ``ensemble`` (nvt | npt) plus keyword
        arguments for NVTSimulation / NPTSimulation (timestep, tau_nhc,
        tau_baro, traj_every, thermo_every, masses, ...).
    schedule : md_quench only -- dict for :func:`temperature_schedule`
        (type, n_segments, T_start, T_end, temperatures).
    segment_steps, quench_every : md_quench only -- MD steps per segment
        and per chunk (segment_steps must be a multiple of quench_every).
    n_trials : basin_hopping only -- number of trial moves.
    temperature, step, cell_step, adapt_every, adapt_factor,
    target_acceptance : basin_hopping settings (K, A, -, trials, -, -).
    pressure : GPa; NPT target and the P in H = E + PV.
    seed : walker seed (perturbations, MD velocity draws).
    restart : continue from ``checkpoint.npz`` in output_dir.
    max_consecutive_failures, min_energy_per_atom : failure handling.
    keep_best, dedup_energy, dedup_volume : distinct minima written as
        POSCAR_rank01.. (energy tolerance eV/atom, relative volume).
    fix_atoms : (N, 3) bool, frozen components (also not perturbed).
    """

    def __init__(
        self,
        lattice,
        positions,
        atom_types,
        force_fn,
        r_cut,
        *,
        method: str = "md_quench",
        output_dir: str = "./anneal_output",
        relax: dict | None = None,
        md: dict | None = None,
        schedule: dict | None = None,
        segment_steps: int = 2000,
        quench_every: int = 500,
        n_trials: int = 100,
        temperature: float = 1000.0,
        step: float = 0.3,
        cell_step: float = 0.02,
        adapt_every: int = 10,
        adapt_factor: float = 0.9,
        target_acceptance: float = 0.5,
        pressure: float = 0.0,
        seed: int = 42,
        restart: bool = False,
        max_consecutive_failures: int = 5,
        min_energy_per_atom: float | None = None,
        keep_best: int = 10,
        dedup_energy: float = 1e-4,
        dedup_volume: float = 1e-3,
        fix_atoms: np.ndarray | None = None,
        verbose: bool = True,
        plot: bool = True,
    ):
        self.method = str(method).lower()
        if self.method not in ("md_quench", "basin_hopping"):
            raise ValueError(
                f"anneal: unknown method {method!r} (md_quench | basin_hopping)"
            )
        self.lattice0 = np.array(lattice, dtype=float)
        self.positions0 = np.array(positions, dtype=float)
        self.atom_types = list(atom_types)
        self.N = len(self.atom_types)
        self.r_cut = r_cut
        self.raw_force_fn = force_fn
        self.pressure = float(pressure)
        self.h_fn = enthalpy_force_fn(force_fn, self.pressure)
        self.output_dir = output_dir
        self.relax_kw = dict(relax or {})
        self.md_kw = dict(md or {})
        self.ensemble = str(self.md_kw.pop("ensemble", "nvt")).lower()
        if self.ensemble not in ("nvt", "npt"):
            raise ValueError(
                f"anneal md: ensemble must be nvt or npt, got {self.ensemble!r}"
            )
        self.seed = int(seed)
        self.restart = bool(restart)
        self.max_fail = int(max_consecutive_failures)
        self.e_floor = (
            None if min_energy_per_atom is None else float(min_energy_per_atom)
        )
        self.keep_best = int(keep_best)
        self.dedup_energy = float(dedup_energy)
        self.dedup_volume = float(dedup_volume)
        self.fix_atoms = None if fix_atoms is None else np.asarray(fix_atoms, bool)
        self.verbose = bool(verbose) and _mpi_rank == 0
        self.plot = bool(plot)
        self._root = _mpi_rank == 0

        if self.method == "md_quench":
            sch = dict(schedule or {})
            self.T_seg = temperature_schedule(
                sch.get("type", "geometric"),
                sch.get("n_segments", 8),
                sch.get("T_start", 1500.0),
                sch.get("T_end", 300.0),
                sch.get("temperatures"),
            )
            self.segment_steps = int(segment_steps)
            self.quench_every = int(quench_every)
            if (
                self.quench_every < 1
                or self.segment_steps < self.quench_every
                or self.segment_steps % self.quench_every
            ):
                raise ValueError(
                    "anneal md_quench: segment_steps must be a positive multiple of "
                    f"quench_every (got {self.segment_steps} and {self.quench_every})"
                )
            self.chunks_per_segment = self.segment_steps // self.quench_every
            self.n_cycles = len(self.T_seg) * self.chunks_per_segment
        else:
            self.n_cycles = int(n_trials)
            self.T_bh = float(temperature)
            self.step0 = float(step)
            self.cell_step0 = float(cell_step)
            self.adapt_every = int(adapt_every)
            self.adapt_factor = float(adapt_factor)
            self.target_acc = float(target_acceptance)
            if self.T_bh <= 0 or self.step0 <= 0:
                raise ValueError(
                    "anneal basin_hopping: temperature and step must be > 0"
                )
            if not 0.0 < self.adapt_factor < 1.0:
                raise ValueError("anneal basin_hopping: adapt_factor must be in (0, 1)")
        if self.max_fail < 1:
            raise ValueError("anneal: max_consecutive_failures must be >= 1")

    # ------------------------------------------------------------ io
    def _path(self, *p):
        return os.path.join(self.output_dir, *p)

    def _cycle_dir(self, c):
        return self._path("cycles", f"c{c:05d}")

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _append_ledger(self, rec):
        if self._root:
            with open(self._path(LEDGER), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()

    def _save_checkpoint(self, cycle, state):
        if not self._root:
            return
        arrays = {k: v for k, v in state.items() if isinstance(v, np.ndarray)}
        meta = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
        meta.update(cycle=cycle, method=self.method, minima=self.minima)
        tmp = self._path(CHECKPOINT + ".tmp.npz")
        np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, self._path(CHECKPOINT))

    def _load_checkpoint(self):
        ck = self._path(CHECKPOINT)
        if not os.path.isfile(ck):
            return None
        with np.load(ck, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            arrays = {k: z[k].copy() for k in z.files if k != "meta"}
        if meta.get("method") != self.method:
            raise ValueError(
                f"anneal restart: checkpoint is for {meta.get('method')!r}, "
                f"the config asks for {self.method!r}"
            )
        meta.update(arrays)
        return meta

    def _trim_ledger(self, last_cycle):
        path = self._path(LEDGER)
        if not (self._root and os.path.isfile(path)):
            return
        keep = []
        with open(path) as fh:
            for line in fh:
                try:
                    if json.loads(line)["cycle"] <= last_cycle:
                        keep.append(line)
                except (ValueError, KeyError):
                    break
        with open(path, "w") as fh:
            fh.writelines(keep)

    # ------------------------------------------------------------ steps
    def _check_energy(self, e):
        if not np.isfinite(e):
            raise _CycleFailed("non-finite energy")
        if self.e_floor is not None and e / self.N < self.e_floor:
            raise _CycleFailed(
                f"energy {e / self.N:.4f} eV/atom below "
                f"min_energy_per_atom {self.e_floor}"
            )

    def _quench(self, lat, pos, out_dir):
        kw = dict(self.relax_kw)
        kw.update(output_dir=out_dir, verbose=False)
        kw.setdefault("timing", False)
        res = GeometryOptimiser(
            lat,
            pos,
            self.atom_types,
            force_fn=self.h_fn,
            r_cut=self.r_cut,
            fix_atoms=self.fix_atoms,
            **kw,
        ).run()
        H = float(res["energy"])
        V = abs(float(np.linalg.det(res["lattice"])))
        E = H - self.pressure / EV_ANG3_TO_GPA * V
        self._check_energy(E)
        return dict(
            E=E,
            H=H,
            V=V,
            converged=bool(res["converged"]),
            n_steps=int(res["n_steps"]),
            lattice=res["lattice"],
            positions=res["positions"],
            poscar=os.path.join(out_dir, "POSCAR_relaxed"),
        )

    def _md_chunk(self, lat, pos, vel, T, cycle, out_dir):
        kw = dict(self.md_kw)
        kw.update(
            n_steps=self.quench_every,
            equil_steps=0,
            temperature=T,
            output_dir=out_dir,
            verbose=False,
            save_plots=False,
            seed=self.seed + 7919 * (cycle + 1),
            initial_velocities=vel,
        )
        kw.setdefault("timing", False)
        common = dict(
            lattice=lat,
            positions=pos,
            atom_types=self.atom_types,
            force_fn=self.raw_force_fn,
            r_cut=self.r_cut,
            fix_atoms=self.fix_atoms,
        )
        if self.ensemble == "npt":
            sim = NPTSimulation(**common, pressure=self.pressure, **kw)
        else:
            sim = NVTSimulation(**common, **kw)
        res = sim.run()
        for key in ("positions", "velocities", "lattice"):
            if not np.all(np.isfinite(res[key])):
                raise _CycleFailed(f"non-finite MD {key}")
        # The same energy checks as a quench, on the chunk's last frame:
        # an MLIP collapsing during the MD (GeSn ELM at 800 K: E_pot
        # -4.1 -> -6.7 eV/atom and T -> 8600 K within 450 steps) must not
        # become the state the next chunk continues from.
        e_end = float(self.raw_force_fn(res["lattice"], res["positions"],
                                        self.atom_types)[0])
        self._check_energy(e_end)
        return res["lattice"], res["positions"], res["velocities"]

    # ------------------------------------------------------------ minima
    def _add_minimum(self, cycle, q):
        if not q["converged"]:
            return
        h = q["H"] / self.N
        for m in self.minima:
            if (
                abs(m["H"] / self.N - h) <= self.dedup_energy
                and abs(m["V"] - q["V"]) <= self.dedup_volume * m["V"]
            ):
                if q["H"] < m["H"]:
                    m.update(
                        cycle=cycle, H=q["H"], E=q["E"], V=q["V"], poscar=q["poscar"]
                    )
                break
        else:
            self.minima.append(
                dict(cycle=cycle, H=q["H"], E=q["E"], V=q["V"], poscar=q["poscar"])
            )
        self.minima.sort(key=lambda m: m["H"])

    def _write_minima(self):
        if not self._root:
            return
        for f in os.listdir(self.output_dir):
            if f.startswith("POSCAR_rank") or f == "POSCAR_best":
                os.remove(self._path(f))
        rows = ["rank,cycle,E_eV,E_per_atom_eV,H_eV,H_per_atom_eV,volume_A3,source"]
        for r, m in enumerate(self.minima[: self.keep_best], start=1):
            if os.path.isfile(m["poscar"]):
                shutil.copyfile(m["poscar"], self._path(f"POSCAR_rank{r:02d}"))
                if r == 1:
                    shutil.copyfile(m["poscar"], self._path("POSCAR_best"))
            rows.append(
                f"{r},{m['cycle']},{m['E']:.8f},{m['E'] / self.N:.8f},"
                f"{m['H']:.8f},{m['H'] / self.N:.8f},{m['V']:.6f},"
                f"{os.path.relpath(m['poscar'], self.output_dir)}"
            )
        with open(self._path("minima.csv"), "w") as fh:
            fh.write("\n".join(rows) + "\n")

    # ------------------------------------------------------------ run
    def run(self) -> dict:
        if self._root:
            os.makedirs(self._path("cycles"), exist_ok=True)
        self.minima = []
        ck = self._load_checkpoint() if self.restart else None
        if self.restart and ck is None:
            self._log(
                "  anneal: restart requested but no checkpoint found; starting fresh"
            )
        if ck is None and self._root:
            for f in (LEDGER, CHECKPOINT):
                if os.path.isfile(self._path(f)):
                    os.remove(self._path(f))
        if ck is not None:
            self.minima = list(ck["minima"])
            self._trim_ledger(ck["cycle"])
            self._log(f"  anneal: restarting after cycle {ck['cycle']}")
        if self.method == "md_quench":
            out = self._run_md_quench(ck)
        else:
            out = self._run_basin_hopping(ck)
        self._write_minima()
        if self.plot and self._root:
            self._plot()
        best = self.minima[0] if self.minima else None
        out.update(
            minima=list(self.minima),
            best=best,
            n_cycles=self.n_cycles,
            output_dir=self.output_dir,
        )
        return out

    def _initial_quench(self):
        """Cycle 0: relax the input structure (the relax-only reference)."""
        rec = dict(cycle=0, kind="initial")
        try:
            q = self._quench(self.lattice0, self.positions0, self._cycle_dir(0))
        except Exception as err:
            raise RuntimeError(
                f"anneal: the starting structure failed to relax: {err}"
            ) from err
        self._add_minimum(0, q)
        rec.update(
            status="ok",
            E=q["E"],
            H=q["H"],
            V=q["V"],
            E_per_atom=q["E"] / self.N,
            converged=q["converged"],
            relax_steps=q["n_steps"],
        )
        self._append_ledger(rec)
        self._log(
            f"  cycle     0  initial relax  H/N = {q['H'] / self.N:.6f} eV  "
            f"V = {q['V']:.3f} A^3  conv = {q['converged']}"
        )
        return q

    def _run_md_quench(self, ck):
        if ck is None:
            # The MD starts from the relaxed input, not the raw one: a
            # strongly strained input releases its excess potential energy
            # as heat faster than the thermostat removes it (SW Si with
            # 0.8 A jitter disordered within one 1300 K chunk).
            q0 = self._initial_quench()
            lat, pos, vel = q0["lattice"].copy(), q0["positions"].copy(), None
            n_fail, start, n_failed = 0, 1, 0
            self._save_checkpoint(
                0, dict(lat=lat, pos=pos, vel=np.zeros((0, 3)), n_fail=0, n_failed=0)
            )
        else:
            lat, pos = ck["lat"], ck["pos"]
            vel = ck["vel"] if ck["vel"].size else None
            n_fail, n_failed, start = (
                int(ck["n_fail"]),
                int(ck["n_failed"]),
                ck["cycle"] + 1,
            )
        aborted = False
        for c in range(start, self.n_cycles + 1):
            T = float(self.T_seg[(c - 1) // self.chunks_per_segment])
            cdir = self._cycle_dir(c)
            rec = dict(cycle=c, kind="quench", T=T)
            try:
                try:
                    lat_md, pos_md, vel_md = self._md_chunk(
                        lat, pos, vel, T, c, os.path.join(cdir, "md")
                    )
                except Exception as err:
                    rec["stage"] = "md"
                    raise _CycleFailed(str(err)) from err
                lat, pos, vel = lat_md, pos_md, vel_md
                try:
                    q = self._quench(lat, pos, os.path.join(cdir, "relax"))
                except Exception as err:
                    rec["stage"] = "relax"
                    raise _CycleFailed(str(err)) from err
            except _CycleFailed as err:
                n_fail += 1
                n_failed += 1
                if rec.get("stage") == "md":
                    vel = None  # same state, fresh Maxwell-Boltzmann draw
                rec.update(status="failed", error=str(err)[:300])
                self._log(
                    f"  cycle {c:5d}  T = {T:7.1f} K  FAILED ({rec['stage']}): "
                    f"{str(err)[:120]}"
                )
            else:
                n_fail = 0
                self._add_minimum(c, q)
                rec.update(
                    status="ok",
                    E=q["E"],
                    H=q["H"],
                    V=q["V"],
                    E_per_atom=q["E"] / self.N,
                    converged=q["converged"],
                    relax_steps=q["n_steps"],
                    best_H_per_atom=(
                        self.minima[0]["H"] / self.N if self.minima else None
                    ),
                )
                self._log(
                    f"  cycle {c:5d}  T = {T:7.1f} K  H/N = {q['H'] / self.N:.6f} eV  "
                    f"V = {q['V']:.3f}  conv = {q['converged']}"
                )
            # Ledger first, checkpoint second: an interruption in between
            # leaves a ledger line past the checkpoint, which a restart
            # trims and recomputes, rather than a lost line.
            self._append_ledger(rec)
            self._save_checkpoint(
                c,
                dict(
                    lat=lat,
                    pos=pos,
                    vel=np.zeros((0, 3)) if vel is None else vel,
                    n_fail=n_fail,
                    n_failed=n_failed,
                ),
            )
            if n_fail >= self.max_fail:
                aborted = True
                self._log(f"  anneal: {n_fail} consecutive failures, stopping")
                break
        return dict(aborted=aborted, n_failed=n_failed)

    def _perturb(self, rng, lat, pos, scale):
        if self.relax_kw.get("relax_cell", False):
            eps = rng.uniform(-1.0, 1.0, (3, 3)) * self.cell_step0 * scale
            eps = 0.5 * (eps + eps.T)
            frac = pos @ np.linalg.inv(lat)
            lat = lat @ (np.eye(3) + eps)
            pos = frac @ lat
        d = _random_in_sphere(rng, self.N, self.step0 * scale)
        if self.fix_atoms is not None:
            d[self.fix_atoms] = 0.0
        return lat, pos + d

    def _run_basin_hopping(self, ck):
        kT = K_B * self.T_bh
        if ck is None:
            q0 = self._initial_quench()
            rng = np.random.default_rng(self.seed)
            cur = dict(lat=q0["lattice"], pos=q0["positions"], H=q0["H"])
            scale, win_acc, win_n = 1.0, 0, 0
            n_acc, n_fail, n_failed, start = 0, 0, 0, 1
        else:
            rng = np.random.default_rng()
            rng.bit_generator.state = json.loads(ck["rng"])
            cur = dict(lat=ck["lat"], pos=ck["pos"], H=float(ck["H"]))
            scale, win_acc, win_n = (
                float(ck["scale"]),
                int(ck["win_acc"]),
                int(ck["win_n"]),
            )
            n_acc, n_fail, n_failed = (
                int(ck["n_acc"]),
                int(ck["n_fail"]),
                int(ck["n_failed"]),
            )
            start = ck["cycle"] + 1

        def state():
            return dict(
                lat=cur["lat"],
                pos=cur["pos"],
                H=cur["H"],
                scale=scale,
                win_acc=win_acc,
                win_n=win_n,
                n_acc=n_acc,
                n_fail=n_fail,
                n_failed=n_failed,
                rng=json.dumps(rng.bit_generator.state),
            )

        if ck is None:
            self._save_checkpoint(0, state())
        aborted = False
        last = start - 1
        for c in range(start, self.n_cycles + 1):
            last = c
            rec = dict(cycle=c, kind="trial", step=self.step0 * scale)
            lat_t, pos_t = self._perturb(rng, cur["lat"], cur["pos"], scale)
            u = rng.random()  # drawn every trial: the stream stays aligned
            try:
                q = self._quench(lat_t, pos_t, self._cycle_dir(c))
            except Exception as err:  # noqa: BLE001
                n_fail += 1
                n_failed += 1
                accepted = False
                rec.update(status="failed", error=str(err)[:300], accepted=False)
                self._log(f"  trial {c:5d}  FAILED: {str(err)[:120]}")
            else:
                n_fail = 0
                self._add_minimum(c, q)
                dH = q["H"] - cur["H"]
                accepted = q["converged"] and (dH <= 0.0 or u < np.exp(-dH / kT))
                if accepted:
                    cur = dict(lat=q["lattice"], pos=q["positions"], H=q["H"])
                    n_acc += 1
                rec.update(
                    status="ok",
                    E=q["E"],
                    H=q["H"],
                    V=q["V"],
                    E_per_atom=q["E"] / self.N,
                    converged=q["converged"],
                    relax_steps=q["n_steps"],
                    accepted=bool(accepted),
                    current_H_per_atom=cur["H"] / self.N,
                    best_H_per_atom=(
                        self.minima[0]["H"] / self.N if self.minima else None
                    ),
                )
                self._log(
                    f"  trial {c:5d}  step = {self.step0 * scale:.3f} A  "
                    f"H/N = {q['H'] / self.N:.6f}  "
                    f"{'accepted' if accepted else 'rejected'}"
                )
            win_n += 1
            win_acc += int(accepted)
            if win_n >= self.adapt_every:
                rate = win_acc / win_n
                scale = (
                    scale / self.adapt_factor
                    if rate > self.target_acc
                    else (
                        scale * self.adapt_factor if rate < self.target_acc else scale
                    )
                )
                win_acc, win_n = 0, 0
            self._append_ledger(rec)  # before the checkpoint, as in md_quench
            self._save_checkpoint(c, state())
            if n_fail >= self.max_fail:
                aborted = True
                self._log(f"  anneal: {n_fail} consecutive failures, stopping")
                break
        return dict(
            aborted=aborted,
            n_failed=n_failed,
            n_accepted=n_acc,
            acceptance=n_acc / max(last, 1),
            current_H=cur["H"],
        )

    # ------------------------------------------------------------ plot
    def _plot(self):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        recs = []
        if os.path.isfile(self._path(LEDGER)):
            with open(self._path(LEDGER)) as fh:
                recs = [json.loads(line) for line in fh if line.strip()]
        ok = [r for r in recs if r.get("status") == "ok"]
        if not ok:
            return
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax = axes[0]
        lab = "H" if self.pressure else "E"
        x = np.array([r["cycle"] for r in ok])
        y = np.array([r["H"] / self.N for r in ok])
        if self.method == "basin_hopping":
            acc = np.array(
                [bool(r.get("accepted")) or r["kind"] == "initial" for r in ok]
            )
            ax.plot(x[acc], y[acc], "o", ms=4, color="C0", label="accepted")
            ax.plot(
                x[~acc], y[~acc], "o", ms=4, mfc="none", color="C1", label="rejected"
            )
        else:
            ax.plot(x, y, "o", ms=4, color="C0", label="quenched")
        best = np.minimum.accumulate(y)
        ax.step(x, best, where="post", color="k", lw=1.2, label="best so far")
        fail = [r["cycle"] for r in recs if r.get("status") == "failed"]
        if fail:
            ax.plot(fail, np.full(len(fail), y.max()), "x", color="C3", label="failed")
        ax.set_ylabel(f"{lab} after relaxation (eV/atom)")
        ax.legend(fontsize=8)
        ax.set_title(f"anneal: {self.method}  (N = {self.N})")
        ax2 = axes[1]
        if self.method == "md_quench":
            q = [r for r in recs if r["kind"] == "quench"]
            ax2.step(
                [r["cycle"] for r in q], [r["T"] for r in q], where="mid", color="C2"
            )
            ax2.set_ylabel("MD temperature (K)")
        else:
            t = [r for r in recs if r["kind"] == "trial"]
            ax2.step([r["cycle"] for r in t], [r["step"] for r in t], where="post", color="C2")
            ax2.set_ylabel("step (A)")
        ax2.set_xlabel("cycle")
        fig.tight_layout()
        fig.savefig(self._path("anneal.png"), dpi=150)
        plt.close(fig)
