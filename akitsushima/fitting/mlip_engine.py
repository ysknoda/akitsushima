"""A trained MLIP as a solver force function (ADR-0004 decision 6;
every solver: ADR-0007).

YAML::

    potential:
      type: MLIP
      model: path/to/model.npz      # or .pt for NNModel

The model file carries its descriptor configuration (in-package ACSF) and
element list, so nothing else is needed and the descriptor at inference is
the one used in training.  For a structure with N atoms, descriptor d and
its derivative rows X_F = -dd/dr, X_S = -(1/V) dd/dT, and a model
E/N = f(d)::

    E     = N f(d)
    F_j   = N X_F[j] . grad f(d)
    sigma = N X_S . grad f(d)          (compressive-positive, ADR-0001)

An atom-centred model (ADR-0005; the file records ``atom_centred``) is
evaluated as E = sum_i eps_s(i)(G_i) from the per-atom ACSF vectors, with
forces and stress contracted from their Jacobians.
"""

import os

import numpy as np

from .descriptors import descriptor_from_config

__all__ = ["MLIPForceField", "load_mlip_model"]


def load_mlip_model(path: str):
    """Load an ELM / QELM / NN / linear model file by content."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"MLIP model file not found: '{path}'")
    if path.endswith(".pt"):
        from ..utils.optional_torch import torch

        if torch.load(path, map_location="cpu", weights_only=False).get(
            "atom_centred", False
        ):
            from .models.atomic_nn import AtomicNNModel

            return AtomicNNModel.load(path)
        from .models.nn import NNModel

        return NNModel.load(path)
    with np.load(path) as z:
        keys = set(z.files)
        cls = str(z["model_class"]) if "model_class" in keys else ""
    if cls in ("AtomicELMModel", "AtomicQELMModel"):
        from .models import atomic

        return getattr(atomic, cls).load(path)
    if "model_str" in keys:
        raise ValueError(
            f"'{path}' is a LightGBM model file; LightGBM was removed from "
            f"akitsushima (ADR-0012: trees give no conservative force).")
    if cls == "LinearModel":
        from .models.linear import LinearModel

        return LinearModel.load(path)
    if cls == "QuantizedELMModel":
        from .models.quantized_elm import QuantizedELMModel

        return QuantizedELMModel.load(path)
    from .models.elm import ELMModel

    return ELMModel.load(path)


class MLIPForceField:
    """
    Force function ``fn(lattice, positions, atom_types, neigh_list=None)
    -> (energy, forces, stress)`` around a chain-rule / linear model
    trained on in-package descriptors.
    """

    kind = "mlip"

    def __init__(self, model):
        name = type(model).__name__
        if getattr(model, "row_independent", True):
            raise ValueError(
                f"MLIP engine: this {name} was trained row-independently "
                f"(row_independent=True, or a file written before 0.2.0a3). "
                f"Its force rows are not derivatives of its energy, so it is "
                f"not a force field.  Retrain with the chain rule "
                f"(the default)."
            )
        if not getattr(model, "descriptor_config", ""):
            raise ValueError(
                f"MLIP engine: this {name} was trained on descriptors from an "
                f"external CSV; akitsushima cannot recompute them for a new "
                f"structure.  Train on in-package descriptors (mlip: "
                f"descriptor: ...) to use it as an engine "
                f"(external descriptors at inference: docs/adr/0010 item 9)."
            )
        self.model = model
        self.descriptor = descriptor_from_config(model.descriptor_config)
        self.r_cut = float(self.descriptor.r_cut)
        self.atom_centred = bool(getattr(model, "atom_centred", False))

    @classmethod
    def from_file(cls, path: str) -> "MLIPForceField":
        return cls(load_mlip_model(path))

    @property
    def species(self) -> list:
        """Elements the model was trained on."""
        return list(self.descriptor.species)

    @property
    def per_atom_energy(self) -> bool:
        """True when the model is a sum of site energies (atom-centred):
        it then supports a changing atom count (gcmc) and the exact heat
        flux (kappa).  A tabular model's E = N f(d) has no site split."""
        return self.atom_centred

    def site_terms(self, lattice, positions, atom_types):
        """Per-atom energies and per-neighbour gradients (ADR-0007)."""
        if not self.atom_centred:
            raise ValueError(
                "MLIP engine: a tabular model (E = N f(d) of a structure "
                "descriptor) has no per-atom energies; the exact heat flux "
                "and a changing atom count need an atom-centred model "
                "(mlip.form: atom_centred).")
        from ..engine.site_terms import SiteTerms
        J = self.descriptor.atomic_jacobian(lattice, positions, atom_types)
        eps, g = self.model.atomic_energy_and_gradient(J.G, J.sp)
        centre = J.centre
        dUdv = np.einsum("mak,mk->ma", J.dGdv, g[centre])
        return SiteTerms(np.asarray(eps, float), centre, J.nbj.astype(np.int64),
                         np.asarray(J.vec, float), dUdv)

    def __call__(self, lattice, positions, atom_types, neigh_list=None):
        if self.atom_centred:
            J = self.descriptor.atomic_jacobian(lattice, positions, atom_types)
            eps, g = self.model.atomic_energy_and_gradient(J.G, J.sp)
            forces, stress = J.forces_and_stress(g)
            return float(eps.sum()), forces, stress
        d, XF, XS = self.descriptor.structure_table(lattice, positions, atom_types)
        N = len(atom_types)
        f, g = self.model.energy_and_gradient(d)
        energy = N * f
        forces = N * (XF @ g)
        stress = N * (XS @ g)
        stress = 0.5 * (stress + stress.T)
        return float(energy), forces, stress


def check_mlip_for_solver(force_fn, solver, *, need_site_energies=False,
                          species=(), refuse=None):
    """Refuse, at construction, an MLIP a solver cannot use (ADR-0007).

    No-op for any other force function.  ``refuse``: a reason this solver
    takes no MLIP at all.  ``need_site_energies``: the solver needs a sum of
    site energies (a changing atom count, or the exact heat flux), which a
    tabular model does not have.  ``species``: elements the solver will
    place, which must be among the model's.
    """
    if getattr(force_fn, "kind", None) != "mlip":
        return
    if refuse:
        raise ValueError(f"{solver}: an MLIP cannot drive it -- {refuse}")
    if need_site_energies and not getattr(force_fn, "per_atom_energy", False):
        raise ValueError(
            f"{solver}: needs a sum of per-atom energies (it changes the atom "
            f"count or needs the exact heat flux); this tabular MLIP (E = N "
            f"f(d) of a structure descriptor) has none.  Train an atom-centred "
            f"model (mlip.form: atom_centred).")
    missing = sorted(set(species) - set(force_fn.species))
    if missing:
        raise ValueError(
            f"{solver}: element(s) {missing} are not among the MLIP's "
            f"trained elements {force_fn.species}.")
