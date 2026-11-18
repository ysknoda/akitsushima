"""Ensemble of atom-centred ELMs (ADR-0033; ADR-0010 item 7).

M atom-centred ELMs (or QELMs) that differ only in the seed of their
random hidden layer are fitted on the same data.  Their mean is the force
field -- forces and stress are the exact derivatives of the mean energy,
so it stays conservative -- and the spread of the members estimates the
model uncertainty (:meth:`MLIPForceField.uncertainty`).

The members share one ``AtomicData`` (the ACSF Jacobians are computed once)
and the hyperparameters chosen by the first member's regularisation and
scale search, so each further member costs one normal-equation build and
one linear solve.
"""

from __future__ import annotations

import io
import json

import numpy as np

from ._api import atomic_mlip
from .atomic import AtomicData, AtomicELMModel, AtomicQELMModel

__all__ = ["AtomicELMEnsemble"]

_MEMBER_CLASSES = {"AtomicELMModel": AtomicELMModel, "AtomicQELMModel": AtomicQELMModel}
SEED_STRIDE = 1000  # member m: seed + 1000 m, clear of the species offset seed + s


@atomic_mlip
class AtomicELMEnsemble:
    """
    Mean of ``n_members`` atom-centred ELMs (``member_class``
    ``"AtomicELMModel"`` or ``"AtomicQELMModel"``) with seeds
    ``seed + 1000 m`` or the explicit list ``seeds``.  Every other keyword
    goes to the members (n_hidden, reg, reg_search, activation, ...).
    """

    atom_centred = True
    row_independent = False

    def __init__(
        self,
        n_members: int = 5,
        member_class: str = "AtomicELMModel",
        seeds: list | None = None,
        verbose: bool = True,
        **member_kw,
    ):
        if member_class not in _MEMBER_CLASSES:
            raise ValueError(
                f"AtomicELMEnsemble: member_class must be one of "
                f"{sorted(_MEMBER_CLASSES)}, got {member_class!r} (ADR-0033: "
                "ELM and QELM only)."
            )
        base = int(member_kw.pop("seed", 42))
        if seeds is None:
            if int(n_members) < 2:
                raise ValueError("AtomicELMEnsemble: n_members must be >= 2.")
            seeds = [base + SEED_STRIDE * m for m in range(int(n_members))]
        seeds = [int(s) for s in seeds]
        if len(seeds) < 2 or len(set(seeds)) != len(seeds):
            raise ValueError(
                f"AtomicELMEnsemble: need at least two distinct seeds, got {seeds}."
            )
        self.member_class = member_class
        self.seeds = seeds
        self.member_kw = dict(member_kw)
        self.verbose = bool(verbose)
        self.members: list = []
        self.physics_config = ""
        self._fitted = False

    # -- what the engine and the scorer read ------------------------------
    @property
    def n_members(self) -> int:
        return len(self.seeds)

    def _first(self):
        if not self.members:
            raise RuntimeError("AtomicELMEnsemble: call fit() first.")
        return self.members[0]

    @property
    def descriptor_config(self) -> str:
        return self._first().descriptor_config if self.members else ""

    @property
    def species(self) -> list:
        return self._first().species

    @property
    def r0(self) -> float:
        return (
            self._first().r0 if self.members else float(self.member_kw.get("r0", 1.0))
        )

    @property
    def eps0(self) -> float:
        return (
            self._first().eps0
            if self.members
            else float(self.member_kw.get("eps0", 0.01))
        )

    # -- fitting -----------------------------------------------------------
    def _member(self, seed: int, **over):
        kw = dict(self.member_kw, **over)
        m = _MEMBER_CLASSES[self.member_class](seed=seed, verbose=False, **kw)
        m.physics_config = self.physics_config
        return m

    def fit(
        self,
        structures,
        descriptor=None,
        weight_csv=None,
        val_fraction: float = 0.2,
        val_seed: int = 0,
    ) -> AtomicELMEnsemble:
        """Fit every member on the same structures (see
        :meth:`AtomicELMModel.fit`).  The first member runs the
        regularisation and scale searches; the others reuse its choice."""
        data = (
            structures
            if isinstance(structures, AtomicData)
            else AtomicData(structures, descriptor, verbose=self.verbose)
        )
        first = self._member(self.seeds[0])
        first.verbose = self.verbose
        first.fit(data, None, weight_csv, val_fraction, val_seed)
        factor, length = first._trial
        chosen = {
            "reg": first.reg,
            "reg_search": None,
            "w_init_scale_search": [float(factor)],
            "w_corr_length": float(length),
            "w_corr_length_search": None,
        }
        self.chosen_hyperparameters_ = chosen
        self.members = [first]
        for s in self.seeds[1:]:
            m = self._member(s, **chosen)
            m.fit(data, None, weight_csv, val_fraction, val_seed)
            self.members.append(m)
            if self.verbose:
                print(
                    f"  ensemble member seed {s}: fitted ({m.fit_time_['total']:.1f} s)"
                )
        self._fitted = True
        # scores of the mean on the members' train / val split
        from .atomic import _prepare, _scores

        probe = self._member(self.seeds[0])
        _, y, lab, _, _, ptr, tr, val = _prepare(
            probe, data, None, weight_csv, val_fraction, val_seed)
        self._train_sc = _scores(self, data, y, lab, ptr, tr, "train")
        self._val_sc = _scores(self, data, y, lab, ptr, val, "val") if len(val) else None
        if self.verbose:
            from .elm import ELMModel

            print(f"\nEnsemble of {self.n_members} {self.member_class} "
                  f"(seeds {self.seeds}), mean:")
            ELMModel._print_scores(self._train_sc, self._val_sc)
        return self

    # -- prediction ----------------------------------------------------------
    def member_energies_and_gradients(self, G, sp):
        """eps (M, N) and g (M, N, Dc) of every member."""
        out = [m.atomic_energy_and_gradient(G, sp) for m in self.members]
        return np.array([e for e, _ in out]), np.array([g for _, g in out])

    def atomic_energy_and_gradient(self, G, sp):
        """Mean of the members: eps_i (N,) and g_i (N, Dc)."""
        if not self._fitted:
            raise RuntimeError("AtomicELMEnsemble: call fit() first.")
        eps, g = self.member_energies_and_gradients(G, sp)
        return eps.mean(axis=0), g.mean(axis=0)

    _predict_structure = AtomicELMModel._predict_structure
    predict_structures = AtomicELMModel.predict_structures
    score_structures = AtomicELMModel.score_structures.__wrapped__

    # -- storage ---------------------------------------------------------------
    def save(self, path: str) -> None:
        """One npz: every member's own npz as bytes, plus the seeds."""
        if not self._fitted:
            raise RuntimeError("AtomicELMEnsemble: model has not been fitted.")
        blobs = {}
        for k, m in enumerate(self.members):
            buf = io.BytesIO()
            m.save(buf)
            blobs[f"member_{k}"] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        np.savez(
            path,
            model_class=np.array("AtomicELMEnsemble"),
            atom_centred=np.array(True),
            member_class=np.array(self.member_class),
            seeds=np.array(self.seeds),
            member_kw=np.array(json.dumps(self.member_kw)),
            physics_config=np.array(self.physics_config),
            **blobs,
        )

    @classmethod
    def load(cls, path: str, verbose: bool = False):
        p = path if path.endswith(".npz") else path + ".npz"
        with np.load(p) as z:
            obj = cls(
                seeds=[int(s) for s in z["seeds"]],
                member_class=str(z["member_class"]),
                verbose=verbose,
                **json.loads(str(z["member_kw"])),
            )
            obj.physics_config = str(z["physics_config"])
            mcls = _MEMBER_CLASSES[obj.member_class]
            obj.members = [
                mcls.load(io.BytesIO(z[f"member_{k}"].tobytes()))
                for k in range(len(obj.seeds))
            ]
        obj._fitted = True
        return obj
