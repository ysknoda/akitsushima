"""Radial distribution function g(r), from one or more atomic
configurations (a single static structure, or accumulated over an MD
trajectory for better statistics).

Derivation (also serves as the basis for the exact single-species
self-consistency check in tests/test_solvers.py)
-------------------------------------------------
g(r) is defined so that the mean number of neighbour atoms found in a
shell [r, r+dr) around a given reference atom is
``rho . g(r) . 4 pi r^2 dr`` (rho = N/V, the bulk number density).

For a species-resolved partial g_ab(r), the analogous definition is
``rho_b . g_ab(r) . 4 pi r^2 dr`` = mean number of species-b atoms found
around a species-a reference atom (rho_b = N_b/V).  Writing M_ab for the
raw count of *unordered* species-(a,b) neighbour pairs found in a shell:

- Every unordered pair (i, j) contributes ONE "i sees a neighbour"
  event AND ONE "j sees a neighbour" event to the TOTAL (species-
  agnostic) count, regardless of species -- so the total histogram is
  always doubled: 2 M_total.
- For a HOMOATOMIC pair (a == b), both of those two events are also
  "an a-atom sees an a-neighbour" events, so the partial histogram is
  likewise doubled: 2 M_aa, normalised by N_a (both reference count and
  the density come from the same species).
- For a HETEROATOMIC pair (a != b), the two events are of DIFFERENT
  kinds ("a sees a b-neighbour" contributes once to that count; "b sees
  an a-neighbour" contributes once to a *different* count) -- so each
  unordered pair contributes exactly once to *each* count, i.e. no
  doubling, and g_ab = (M_ab / N_a) / (rho_b . shell_vol). One can show
  this is symmetric, g_ab == g_ba, exactly as the notation suggests (the
  same M_ab appears in both the a-referenced and b-referenced
  derivation).

A direct, no-reference-data-needed consequence used as a regression
test: for a genuinely single-species system, every pair is homoatomic,
N_a == N (the only species), rho_a == rho, and the partial and total
formulas above reduce to being numerically IDENTICAL.
"""
import numpy as np

from ..engine.neighborlist import NeighbourList

__all__ = ["RadialDistributionFunction"]


class RadialDistributionFunction:
    """
    Accumulate pair-distance statistics from one or more frames and
    compute the total and per-species-pair radial distribution
    functions, plus the running coordination number.

    Parameters
    ----------
    r_max   : maximum distance to histogram [A] (also used as the
              NeighbourList cutoff).
    n_bins  : number of histogram bins spanning [0, r_max).

    Usage
    -----
    >>> import numpy as np
    >>> from akitsushima import RadialDistributionFunction
    >>> base = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
    >>> cells = np.array([[i, j, k] for i in range(4) for j in range(4) for k in range(4)])
    >>> lattice = np.eye(3) * 4 * 5.256                  # 4x4x4 fcc Ar, a = 5.256 A
    >>> positions = (base[None] + cells[:, None]).reshape(-1, 3) * 5.256
    >>> atom_types = ["Ar"] * len(positions)
    >>> rdf = RadialDistributionFunction(r_max=8.0, n_bins=200)
    >>> rdf.add_frame(lattice, positions, atom_types)   # one or more frames
    >>> result = rdf.compute()
    >>> result["r"], result["g_total"], result["coordination_number"]
    """

    def __init__(self, r_max: float, n_bins: int = 200):
        if r_max <= 0.0:
            raise ValueError(
                f"RadialDistributionFunction: r_max must be positive, got {r_max}")
        if n_bins <= 0:
            raise ValueError(
                f"RadialDistributionFunction: n_bins must be positive, got {n_bins}")
        self.r_max  = float(r_max)
        self.n_bins = int(n_bins)
        self.r_edges   = np.linspace(0.0, self.r_max, self.n_bins + 1)
        self.r_centers = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])
        self._shell_vol = ((4.0 / 3.0) * np.pi
                          * (self.r_edges[1:] ** 3 - self.r_edges[:-1] ** 3))

        self._hist_total   = np.zeros(self.n_bins)
        self._hist_partial : dict = {}   # frozenset({sp1,sp2}) -> (n_bins,) hist
        self._n_frames        = 0
        self._n_atoms_total    = 0
        self._volume_sum       = 0.0
        self._species_count_sum: dict = {}   # species -> summed count across frames

    def add_frame(self,
                  lattice   : np.ndarray,
                  positions : np.ndarray,
                  atom_types: list) -> None:
        """Accumulate pair-distance statistics from one configuration."""
        N = len(positions)
        nl = NeighbourList(r_cut=self.r_max, skin_width=0.0)
        nl.build(lattice, positions)
        idx_i, idx_j, _, r = nl.get_pairs_arrays()

        h, _ = np.histogram(r, bins=self.r_edges)
        self._hist_total += 2.0 * h   # see module docstring: always doubled

        types_arr = np.array(atom_types)
        sp_i = types_arr[idx_i]
        sp_j = types_arr[idx_j]
        pair_keys = [frozenset({a, b}) for a, b in zip(sp_i, sp_j)]
        for pair in set(pair_keys):
            mask = np.array([k == pair for k in pair_keys])
            h_pair, _ = np.histogram(r[mask], bins=self.r_edges)
            factor = 2.0 if len(pair) == 1 else 1.0   # see module docstring
            self._hist_partial.setdefault(pair, np.zeros(self.n_bins))
            self._hist_partial[pair] += factor * h_pair

        for sp in set(atom_types):
            self._species_count_sum[sp] = (
                self._species_count_sum.get(sp, 0) + atom_types.count(sp))

        self._n_frames     += 1
        self._n_atoms_total += N
        self._volume_sum    += float(abs(np.linalg.det(lattice)))

    def compute(self) -> dict:
        """
        Finalise the accumulated statistics.

        Returns
        -------
        dict with keys:
            r                   : (n_bins,) bin-centre distances [A].
            g_total             : (n_bins,) total radial distribution function.
            g_partial           : dict frozenset({sp1,sp2}) -> (n_bins,) g_ab(r).
            coordination_number : (n_bins,) running integral N(r) = integral_0^r
                                  4 pi r'^2 rho g(r') dr' (mean number of
                                  neighbours within distance r of a reference atom).
            density             : mean number density N/V used for normalisation.
        """
        if self._n_frames == 0:
            raise RuntimeError(
                "RadialDistributionFunction.compute(): no frames added "
                "-- call add_frame() at least once first.")

        N_avg = self._n_atoms_total / self._n_frames
        V_avg = self._volume_sum / self._n_frames
        rho   = N_avg / V_avg

        norm_total = self._n_frames * N_avg * rho * self._shell_vol
        g_total = np.divide(self._hist_total, norm_total,
                            out=np.zeros_like(self._hist_total),
                            where=norm_total > 0)

        species_avg = {sp: c / self._n_frames
                      for sp, c in self._species_count_sum.items()}
        g_partial = {}
        for pair, hist in self._hist_partial.items():
            sp_list = sorted(pair)
            n_a = species_avg[sp_list[0]]
            n_b = species_avg[sp_list[-1]]   # == n_a for a homoatomic pair
            norm = self._n_frames * n_a * n_b / V_avg * self._shell_vol
            g_partial[pair] = np.divide(hist, norm,
                                        out=np.zeros_like(hist),
                                        where=norm > 0)

        coordination_number = np.cumsum(g_total * self._shell_vol * rho)

        return {
            "r"                  : self.r_centers,
            "g_total"            : g_total,
            "g_partial"          : g_partial,
            "coordination_number": coordination_number,
            "density"            : rho,
        }
