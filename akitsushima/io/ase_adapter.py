"""ASE adapter (ADR-0034 decision 6).  ``ase`` is an optional extra.

* :func:`to_ase_atoms` / :func:`from_ase_atoms` convert structures.
* :class:`AkitsushimaCalculator` wraps an akitsushima force function
  ``force_fn(lattice, positions, atom_types) -> (E, F, S)`` as an ASE
  calculator: energy, free_energy, forces and stress (Voigt, ASE's sign:
  positive under tension; akitsushima's is positive under compression,
  ADR-0001).  For an ensemble MLIP (ADR-0033) ``results`` also holds
  ``sigma_E`` [eV/atom], ``sigma_F_max`` [eV/A] and ``sigma_S`` [GPa].

The reverse direction (an ASE calculator used as an akitsushima force
function) is not provided.
"""

from __future__ import annotations

import numpy as np

__all__ = ["AkitsushimaCalculator", "from_ase_atoms", "to_ase_atoms"]


def _ase():
    try:
        import ase
    except ImportError:
        raise ImportError("This needs ASE (pip install 'akitsushima[ase]').") from None
    return ase


def to_ase_atoms(lattice, positions, atom_types, calculator=None):
    """An ``ase.Atoms`` (pbc in all directions) of an akitsushima structure."""
    _ase()
    from ase import Atoms

    atoms = Atoms(
        symbols=list(atom_types),
        positions=np.asarray(positions, dtype=float),
        cell=np.asarray(lattice, dtype=float),
        pbc=True,
    )
    if calculator is not None:
        atoms.calc = calculator
    return atoms


def from_ase_atoms(atoms) -> dict:
    """The dict of :func:`parse_poscar` from an ``ase.Atoms``.  The atoms
    must be periodic in all three directions."""
    from .structure import structure_dict

    if not all(atoms.pbc):
        raise ValueError(
            f"from_ase_atoms: pbc={list(atoms.pbc)}; akitsushima structures "
            "are periodic in all three directions."
        )
    return structure_dict(
        np.array(atoms.cell[:], dtype=float),
        np.array(atoms.get_positions(), dtype=float),
        list(atoms.get_chemical_symbols()),
    )


def _calculator_base():
    _ase()
    from ase.calculators.calculator import Calculator, all_changes

    return Calculator, all_changes


def _make_calculator_class():
    Calculator, all_changes = _calculator_base()

    class AkitsushimaCalculator(Calculator):
        """ASE calculator around an akitsushima force function.

        Parameters
        ----------
        force_fn : callable ``(lattice, positions, atom_types) -> (E, F, S)``
            with E [eV], F (N, 3) [eV/A], S (3, 3) [eV/A^3, positive under
            compression] -- any akitsushima engine, ``MLIPForceField`` or
            ``force_fn_from_config`` result.
        """

        implemented_properties = ["energy", "free_energy", "forces", "stress"]
        nolabel = True

        def __init__(self, force_fn, **kwargs):
            super().__init__(**kwargs)
            self.force_fn = force_fn

        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            s = from_ase_atoms(self.atoms)
            E, F, S = self.force_fn(s["lattice"], s["positions"], s["atom_types"])
            S = -np.asarray(S, dtype=float)  # ASE sign: positive under tension
            self.results = {
                "energy": float(E),
                "free_energy": float(E),
                "forces": np.array(F, dtype=float),
                # Voigt order xx yy zz yz xz xy, as ASE
                "stress": np.array(
                    [S[0, 0], S[1, 1], S[2, 2], S[1, 2], S[0, 2], S[0, 1]]
                ),
            }
            unc = getattr(self.force_fn, "uncertainty", None)
            u = unc() if callable(unc) else None
            if u:
                self.results.update({k: float(v) for k, v in u.items()})

    AkitsushimaCalculator.__qualname__ = "AkitsushimaCalculator"
    AkitsushimaCalculator.__module__ = __name__
    return AkitsushimaCalculator


_CLASS = None


def __getattr__(name):
    global _CLASS
    if name == "AkitsushimaCalculator":
        if _CLASS is None:
            _CLASS = _make_calculator_class()
        return _CLASS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
