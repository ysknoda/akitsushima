"""Non-equilibrium MD with two Langevin heat baths (ADR-0020): temperature
profile, local (per-bin) heat flux, thermal conductivity from the
gradient, and the Kapitza resistance of given interfaces.

Geometry: transport along lattice vector ``axis``; the hot and cold baths
are slabs given as fractional intervals along it (or atom indices).  The
cell is periodic, so heat flows from the hot to the cold bath both ways
round: the flux through a cross-section of area A is Qdot / (2 A).

Integrator: velocity Verlet, then an exact Ornstein-Uhlenbeck update of
the bath atoms' velocities, v <- c v + sqrt((1 - c^2) k_B T / m) xi,
c = exp(-dt / damping); the kinetic energy each bath adds is accumulated
(Q_hot > 0, Q_cold < 0 in steady state).  The energy of the system then
changes only by Q_hot + Q_cold (checked in the tests).
"""
import json
import os
from typing import Callable, Optional, Sequence

import numpy as np

from ..constants import ATOMIC_MASS, EV_ANG_TO_AMU_ANG_FS2, K_B

__all__ = ["NEMDSimulation"]

_EV_ANG_K_FS_TO_W_MK = 1.602176634e-19 / (1e-10 * 1e-15)
_EV_ANG2_FS_TO_W_M2 = 1.602176634e-19 / (1e-20 * 1e-15)
_AX = {"a": 0, "b": 1, "c": 2, 0: 0, 1: 1, 2: 2}


class NEMDSimulation:
    """
    Parameters
    ----------
    lattice, positions, atom_types, force_fn : the system (positions in A).
    masses      : (N,) [amu]; default ATOMIC_MASS.
    T_hot, T_cold : bath temperatures [K].
    axis        : transport direction, lattice vector 'a' | 'b' | 'c'.
    hot, cold   : bath slabs as fractional intervals [lo, hi) along axis
                  (initial positions), or ``hot_atoms`` / ``cold_atoms``
                  (0-based indices) instead.  Membership is fixed at the
                  start (solids).
    damping     : Langevin relaxation time [fs].
    timestep    : [fs].
    n_bins      : bins along axis for the profiles.
    local_flux  : accumulate the per-bin heat flux from the per-atom heat
                  currents (needs a site-energy decomposition, ADR-0007;
                  the refusals of ThermalConductivityCalculator apply).
    seed        : RNG seed (initial velocities and bath noise).
    """

    def __init__(self, lattice, positions, atom_types, force_fn: Callable,
                 masses=None, T_hot=330.0, T_cold=270.0, axis="c",
                 hot=(0.0, 0.05), cold=(0.5, 0.55),
                 hot_atoms: Optional[Sequence[int]] = None,
                 cold_atoms: Optional[Sequence[int]] = None,
                 damping=100.0, timestep=1.0, n_bins=20, local_flux=True,
                 seed=42):
        from ..fitting.force_fn import check_esm_for_solver
        check_esm_for_solver(force_fn, "NEMDSimulation",
                             refuse="ESM boundaries and NEMD baths are not "
                                    "combined (ADR-0020, Q32)")
        self.lattice = np.asarray(lattice, float).copy()
        self.pos = np.asarray(positions, float).copy()
        self.types = list(atom_types)
        self.fn = force_fn
        self.N = len(self.types)
        self.masses = (np.asarray(masses, float) if masses is not None
                       else np.array([ATOMIC_MASS.get(t, 1.0) for t in self.types]))
        self.T_hot, self.T_cold = float(T_hot), float(T_cold)
        self.axis = _AX[axis]
        self.damping, self.dt = float(damping), float(timestep)
        self.n_bins = int(n_bins)
        self.local_flux = bool(local_flux)
        self.seed = int(seed)
        frac = self._frac(self.pos)
        def slab(iv):
            lo, hi = float(iv[0]) % 1.0, float(iv[1]) % 1.0
            # a layer at 0 can sit at -1e-17 and wrap to 0.99999...
            f = (frac[:, self.axis] + 1e-9) % 1.0
            return np.flatnonzero((f >= lo) & (f < hi) if lo < hi
                                  else (f >= lo) | (f < hi))
        self.hot = (np.asarray(hot_atoms, int) if hot_atoms is not None else slab(hot))
        self.cold = (np.asarray(cold_atoms, int) if cold_atoms is not None else slab(cold))
        if len(self.hot) == 0 or len(self.cold) == 0:
            raise ValueError("NEMDSimulation: a heat bath has no atoms.")
        if np.intersect1d(self.hot, self.cold).size:
            raise ValueError("NEMDSimulation: the hot and cold baths overlap.")
        self.hot_iv = (frac[self.hot, self.axis].min(), frac[self.hot, self.axis].max())
        self.cold_iv = (frac[self.cold, self.axis].min(), frac[self.cold, self.axis].max())
        V = abs(np.linalg.det(self.lattice))
        i, j = [k for k in range(3) if k != self.axis]
        self.area = float(np.linalg.norm(np.cross(self.lattice[i], self.lattice[j])))
        self.length = V / self.area                     # perpendicular width
        self.volume = V
        # unit normal of the cross-section, pointing along +axis
        n = np.cross(self.lattice[i], self.lattice[j])
        self.normal = n / np.linalg.norm(n) * np.sign(n @ self.lattice[self.axis])
        self._hf = None
        if self.local_flux:
            from .thermal_conductivity import ThermalConductivityCalculator
            self._hf = ThermalConductivityCalculator(
                self.lattice, self.pos, self.types, force_fn, masses=self.masses)

    # ------------------------------------------------------------------
    def _frac(self, pos):
        f = pos @ np.linalg.inv(self.lattice)
        return f - np.floor(f)

    def _ke(self, vel, idx=slice(None)):
        return 0.5 * np.sum(self.masses[idx, None] * vel[idx] ** 2) / EV_ANG_TO_AMU_ANG_FS2

    def _o_step(self, vel, idx, T, c):
        before = self._ke(vel, idx)
        s = np.sqrt((1.0 - c * c) * K_B * T * EV_ANG_TO_AMU_ANG_FS2 / self.masses[idx])
        vel[idx] = c * vel[idx] + s[:, None] * self.rng.standard_normal((len(idx), 3))
        return self._ke(vel, idx) - before

    def _fresh_state(self):
        self.rng = np.random.default_rng(self.seed)
        T0 = 0.5 * (self.T_hot + self.T_cold)
        sig = np.sqrt(K_B * T0 * EV_ANG_TO_AMU_ANG_FS2 / self.masses)
        vel = self.rng.standard_normal((self.N, 3)) * sig[:, None]
        vel -= (self.masses[:, None] * vel).sum(0) / self.masses.sum()
        nb = self.n_bins
        return {"step": 0, "pos": self.pos.copy(), "vel": vel,
                "Q_hot": 0.0, "Q_cold": 0.0, "Q_hot_prod": 0.0, "Q_cold_prod": 0.0,
                "E0": None, "n_prod": 0, "n_samples": 0,
                "ke_bin": np.zeros((4, nb)), "n_bin": np.zeros((4, nb)),
                "j_bin": np.zeros((4, nb)), "n_jsamp": np.zeros(4),
                "q_block": np.zeros((4, 2)), "t_block": np.zeros(4)}

    def _save(self, path, st):
        tmp = path + ".tmp.npz"
        arrays = {k: v for k, v in st.items() if isinstance(v, np.ndarray)}
        scal = {k: v for k, v in st.items() if not isinstance(v, np.ndarray)}
        np.savez(tmp, **arrays, scalars=json.dumps(scal),
                 rng=json.dumps(self.rng.bit_generator.state))
        os.replace(tmp, path)

    def _load(self, path):
        d = np.load(path, allow_pickle=False)
        st = json.loads(str(d["scalars"]))
        for k in d.files:
            if k not in ("scalars", "rng"):
                st[k] = d[k].copy()
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = json.loads(str(d["rng"]))
        return st

    # ------------------------------------------------------------------
    def run(self, equil_steps: int, n_steps: int, sample_every: int = 10,
            checkpoint: Optional[str] = None, checkpoint_every: int = 0,
            thermo_path: Optional[str] = None, verbose: bool = True) -> dict:
        """
        ``equil_steps`` with the baths on (not sampled), then ``n_steps``
        of production sampled every ``sample_every`` steps into four
        consecutive blocks (for error bars and the steady-state check).
        With ``checkpoint`` set, the full state -- positions, velocities,
        accumulators and the RNG state -- is written there every
        ``checkpoint_every`` steps (atomically) and, if the file exists at
        the start, the run continues from it: an interrupted run loses at
        most ``checkpoint_every`` steps and continues bit for bit.
        """
        st = (self._load(checkpoint) if checkpoint and os.path.isfile(checkpoint)
              else self._fresh_state())
        if verbose and st["step"]:
            print(f"  NEMD: restarting from {checkpoint} at step {st['step']}")
        pos, vel = st["pos"], st["vel"]
        conv = EV_ANG_TO_AMU_ANG_FS2 / self.masses[:, None]
        c = float(np.exp(-self.dt / self.damping))
        E, F, S = self.fn(self.lattice, pos, self.types)
        if st["E0"] is None:
            st["E0"] = float(E) + self._ke(vel)
        acc = F * conv
        total = equil_steps + n_steps
        th = open(thermo_path, "a") if thermo_path else None
        if th is not None and st["step"] == 0:
            th.write("# step T_hot_K T_cold_K Q_hot_eV Q_cold_eV E_minus_Q_drift_eV\n")
        inv = np.linalg.inv(self.lattice)
        try:
            while st["step"] < total:
                vel += 0.5 * self.dt * acc
                pos += self.dt * vel
                E, F, S = self.fn(self.lattice, pos, self.types)
                acc = F * conv
                vel += 0.5 * self.dt * acc
                dh = self._o_step(vel, self.hot, self.T_hot, c)
                dc = self._o_step(vel, self.cold, self.T_cold, c)
                st["Q_hot"] += dh
                st["Q_cold"] += dc
                st["step"] += 1
                k = st["step"] - equil_steps                  # production step
                if k > 0:
                    blk = min(4 * (k - 1) // max(n_steps, 1), 3)
                    st["Q_hot_prod"] += dh
                    st["Q_cold_prod"] += dc
                    st["n_prod"] += 1
                    st["q_block"][blk] += (dh, dc)
                    st["t_block"][blk] += self.dt
                    if k % sample_every == 0:
                        self._sample(st, blk, pos, vel, E, F, S)
                if th is not None and st["step"] % max(sample_every * 10, 1) == 0:
                    drift = float(E) + self._ke(vel) - st["E0"] - st["Q_hot"] - st["Q_cold"]
                    th.write(f"{st['step']} {self._temp(vel, self.hot):.3f} "
                             f"{self._temp(vel, self.cold):.3f} {st['Q_hot']:.8e} "
                             f"{st['Q_cold']:.8e} {drift:.6e}\n")
                    th.flush()            # progress visible during long runs
                if checkpoint and checkpoint_every and st["step"] % checkpoint_every == 0:
                    st["pos"], st["vel"] = pos, vel
                    self._save(checkpoint, st)
        finally:
            if th is not None:
                th.close()
        st["pos"], st["vel"] = pos, vel
        if checkpoint:
            self._save(checkpoint, st)
        st["E_minus_Q_drift_eV"] = (float(E) + self._ke(vel) - st["E0"]
                                    - st["Q_hot"] - st["Q_cold"])
        self.state = st
        return st

    def _temp(self, vel, idx):
        return 2.0 * self._ke(vel, idx) / (3.0 * len(idx) * K_B)

    def _sample(self, st, blk, pos, vel, E, F, S):
        f = self._frac(pos)[:, self.axis]
        b = np.minimum((f * self.n_bins).astype(int), self.n_bins - 1)
        ke = 0.5 * self.masses * np.sum(vel ** 2, axis=1) / EV_ANG_TO_AMU_ANG_FS2
        np.add.at(st["ke_bin"][blk], b, ke)
        np.add.at(st["n_bin"][blk], b, 1.0)
        st["n_samples"] += 1
        if self._hf is not None:
            hf = self._hf
            hf.lattice = self.lattice
            pdata = hf._gather_pair_data(pos)
            e_i = hf._per_atom_energy(pos, F, float(E), pdata)
            Ji = hf._per_atom_heat_current(vel, e_i, pdata, S) @ self.normal
            v_bin = self.volume / self.n_bins
            np.add.at(st["j_bin"][blk], b, Ji / v_bin)
            st["n_jsamp"][blk] += 1

    # ------------------------------------------------------------------
    def results(self, fit_fraction: float = 0.5, interfaces: Sequence[float] = (),
                kapitza_window: float = 0.1, kapitza_skip: int = 1) -> dict:
        """
        From the accumulated production data: T(z) per bin and block, the
        bath heat rates, the flux J = Qdot_hot / (2 A) (and from the cold
        bath), the per-bin local flux along +axis, kappa from a linear fit
        over the central ``fit_fraction`` of each of the two segments
        between the baths, block errors, the steady-state indicators, and
        for each interface position (fractional) the temperature jump and
        Kapitza resistance from linear fits within ``kapitza_window`` on
        each side, ``kapitza_skip`` bins next to it left out.
        """
        st = self.state
        nb = self.n_bins
        z = (np.arange(nb) + 0.5) / nb                       # fractional centres
        with np.errstate(invalid="ignore", divide="ignore"):
            T_blk = 2.0 * st["ke_bin"] / (3.0 * K_B * st["n_bin"])
            T = 2.0 * st["ke_bin"].sum(0) / (3.0 * K_B * st["n_bin"].sum(0))
            j_loc = (st["j_bin"].sum(0) / st["n_jsamp"].sum()
                     if st["n_jsamp"].sum() else np.full(nb, np.nan))
        t_prod = st["n_prod"] * self.dt
        rate_hot = st["Q_hot_prod"] / t_prod                  # eV/fs
        rate_cold = st["Q_cold_prod"] / t_prod
        J_hot = rate_hot / (2.0 * self.area)                  # eV/(A^2 fs)
        J_cold = -rate_cold / (2.0 * self.area)
        J = 0.5 * (J_hot + J_cold)

        def seg_bins(a_end, b_start):
            """Bins strictly between the end of one bath and the start of
            the next going +axis, central fit_fraction of them."""
            span = (b_start - a_end) % 1.0
            lo = a_end + 0.5 * (1.0 - fit_fraction) * span
            rel = (z - lo) % 1.0
            return np.flatnonzero(rel < fit_fraction * span), lo
        out_seg = []
        for name, a_end, b_start, sign in (
                ("hot->cold (+axis)", self.hot_iv[1], self.cold_iv[0], -1),
                ("cold->hot (+axis)", self.cold_iv[1], self.hot_iv[0], +1)):
            idx, lo = seg_bins(a_end, b_start)
            x = ((z[idx] - lo) % 1.0) * self.length          # A
            def fit(Tv):
                ok = np.isfinite(Tv[idx])
                return np.polyfit(x[ok], Tv[idx][ok], 1)[0] if ok.sum() >= 2 else np.nan
            g = fit(T)
            g_blk = np.array([fit(T_blk[k]) for k in range(4)])
            kap = J / abs(g) * _EV_ANG_K_FS_TO_W_MK if np.isfinite(g) and g else np.nan
            kap_blk = J / np.abs(g_blk) * _EV_ANG_K_FS_TO_W_MK
            out_seg.append({"segment": name, "bins": idx.tolist(),
                            "gradient_K_per_A": float(g),
                            "gradient_sign_expected": sign,
                            "gradient_sign_ok": bool(np.isfinite(g) and np.sign(g) == sign),
                            "kappa_W_mK": float(kap),
                            "kappa_block_W_mK": kap_blk.tolist(),
                            "kappa_err_W_mK": float(np.std(kap_blk, ddof=1) / 2.0)})
        kapitza = []
        w = int(round(kapitza_window * nb))
        for x0 in interfaces:
            i0 = int(np.floor(x0 * nb + 1e-9))               # first bin right of x0
            left = [(i0 - kapitza_skip - 1 - k) % nb for k in range(w)]
            right = [(i0 + kapitza_skip + k) % nb for k in range(w)]
            def ext(ix):
                xs = ((z[ix] - x0 + 0.5) % 1.0 - 0.5) * self.length
                p = np.polyfit(xs, T[ix], 1)
                return np.polyval(p, 0.0)
            dT = float(ext(left) - ext(right))
            jj = abs(J)
            kapitza.append({"interface_frac": float(x0), "delta_T_K": dT,
                            "R_m2K_per_W": abs(dT) / (jj * _EV_ANG2_FS_TO_W_M2)
                            if jj else float("nan")})
        prof = [np.nanmax(np.abs(T_blk[k] - T_blk[k - 1])) for k in (2, 3)]
        return {
            "z_frac": z, "z_A": z * self.length, "T_K": T, "T_block_K": T_blk,
            "atoms_per_bin": st["n_bin"].sum(0) / max(st["n_samples"], 1),
            "local_flux_eV_A2_fs": j_loc,
            "local_flux_W_m2": j_loc * _EV_ANG2_FS_TO_W_M2,
            "Q_hot_rate_eV_fs": rate_hot, "Q_cold_rate_eV_fs": rate_cold,
            "flux_hot_W_m2": J_hot * _EV_ANG2_FS_TO_W_M2,
            "flux_cold_W_m2": J_cold * _EV_ANG2_FS_TO_W_M2,
            "flux_W_m2": J * _EV_ANG2_FS_TO_W_M2,
            "bath_balance": float(abs(rate_hot + rate_cold) / max(abs(rate_hot), 1e-300)),
            "profile_change_last_blocks_K": [float(p) for p in prof],
            "segments": out_seg,
            "kappa_W_mK": float(np.nanmean([s["kappa_W_mK"] for s in out_seg])),
            "kapitza": kapitza,
            "area_A2": self.area, "length_A": self.length,
            "production_fs": t_prod,
            "E_minus_Q_drift_eV": st.get("E_minus_Q_drift_eV"),
        }
