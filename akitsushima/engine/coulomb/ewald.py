"""Ewald summation -- fully vectorised (no Python loops over atoms or k)."""
import math
from typing import Optional, Tuple

import numpy as np
from scipy.special import erfc

from ...constants import K_E
from ...utils.mpi import _mpi_allreduce_sum, _mpi_allreduce_sum_scalar, _mpi_domain_split


def real_space_coulomb(lattice: np.ndarray,
                        positions: np.ndarray,
                        charges: np.ndarray,
                        alpha: float,
                        r_cut: float,
                        neigh_list=None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Real-space (erfc-damped) short-range Coulomb sum.

    Shared by :func:`ewald_sum` (always calls this with ``neigh_list=None``,
    i.e. the O(N^2) vectorised minimum-image path) and
    :class:`~akitsushima.engine.coulomb.pppm.PPPMSolver` (which may pass a
    pre-built :class:`~akitsushima.engine.neighborlist.NeighbourList` for an
    O(N) pair sum instead). Extracting this into a single implementation
    matters for correctness, not just for avoiding duplication: an earlier
    version of this codebase had two independent copies of this formula (one
    per caller) that had drifted apart -- the neighbour-list copy was correct
    and the O(N^2) copy had an extra spurious minus sign in the force (see
    the bug-fix notes on :func:`ewald_sum`). Keeping one implementation means
    a future fix here automatically applies to every caller.

    Parameters
    ----------
    lattice    : (3,3) lattice vectors [A], rows = a, b, c.
    positions  : (N,3) Cartesian positions [A].
    charges    : (N,)  formal charges [e].
    alpha      : Ewald splitting parameter [A^-1].
    r_cut      : real-space cutoff [A].
    neigh_list : optional pre-built NeighbourList; its
                 ``get_candidate_pairs()`` supplies the candidate
                 ``(idx_i, idx_j)`` and the minimum-image geometry is
                 recomputed here from the positions actually passed in.
                 When omitted, pairs are found by a fully-vectorised
                 O(N^2) minimum-image search instead.

    Returns
    -------
    energy : float        [eV]   (MPI-reduced across ranks)
    forces : (N,3) array  [eV/A] (MPI-reduced)
    stress : (3,3) array  [eV/A^3] (MPI-reduced, already divided by volume)
    """
    N = len(positions)
    volume = np.abs(np.linalg.det(lattice))
    energy = 0.0
    forces = np.zeros((N, 3))
    stress = np.zeros((3, 3))

    if neigh_list is not None:
        # Bug-fix note (stale-distance / defeated-skin defect): this used
        # to consume the rij/r cached by the last neighbour-list rebuild.
        # With the default skin_width > 0 the list is deliberately not
        # rebuilt every step, so the cached geometry -- and the ``< r_cut``
        # mask applied to it -- were both stale, and the mask additionally
        # discarded the skin-resident pairs the skin exists to preserve.
        # Candidates now come back as indices only and the minimum-image
        # geometry is recomputed from the CURRENT positions, matching the
        # neigh_list-free branch below.  See
        # NeighbourList.get_pairs_arrays' own bug-fix note.
        i_idx, j_idx = neigh_list.get_candidate_pairs()
        inv_lat = np.linalg.inv(lattice)
        dr_cand = positions[j_idx] - positions[i_idx]
        s_cand = dr_cand @ inv_lat
        s_cand -= np.round(s_cand)
        dr_cand = s_cand @ lattice
        r_cand = np.linalg.norm(dr_cand, axis=1)
        keep = (r_cand < r_cut) & (r_cand > 1e-12)
        i_idx, j_idx = i_idx[keep], j_idx[keep]
        dr_all, r_all = dr_cand[keep], r_cand[keep]
    else:
        inv_lat = np.linalg.inv(lattice)
        dr_raw = positions[None, :, :] - positions[:, None, :]   # (N, N, 3)
        s_raw = dr_raw @ inv_lat
        s_mi = s_raw - np.round(s_raw)
        dr_mi = s_mi @ lattice
        dist2 = np.einsum('ijk,ijk->ij', dr_mi, dr_mi)
        dist = np.sqrt(np.maximum(dist2, 1e-30))
        mask = (np.triu(np.ones((N, N), dtype=bool), k=1)
                & (dist < r_cut) & (dist > 1e-12))
        i_idx, j_idx = np.where(mask)
        dr_all = dr_mi[i_idx, j_idx]
        r_all = dist[i_idx, j_idx]

    p_start, p_end = _mpi_domain_split(len(i_idx))
    i_idx, j_idx = i_idx[p_start:p_end], j_idx[p_start:p_end]
    dr_m, r_m = dr_all[p_start:p_end], r_all[p_start:p_end]

    if len(i_idx) > 0:
        qi_qj = K_E * charges[i_idx] * charges[j_idx]        # (P,)

        ar = alpha * r_m
        erfc_ar = erfc(ar)
        exp_ar2 = np.exp(-ar**2)

        # Energy: E_real = sum_{i<j} q_i q_j erfc(alpha r) / r  (factor 2 absorbed)
        energy += np.dot(qi_qj, erfc_ar / r_m)

        # dV/dr per pair
        de_dr = qi_qj * (-erfc_ar / r_m**2
                         - 2.0 * alpha / np.sqrt(np.pi) * exp_ar2 / r_m)
        # Force on i: F_i = +(dV/dr) * (r_j - r_i) / r   (see ewald_sum docstring
        # note 1 -- an earlier version had a spurious extra minus sign here)
        fij = (de_dr / r_m)[:, None] * dr_m                 # (P, 3)
        np.add.at(forces, i_idx, fij)
        np.add.at(forces, j_idx, -fij)

        # Virial stress: sigma = -(1/V) sum_{i<j} F_ij (x) r_ij
        stress -= np.einsum('pi,pj->ij', fij, dr_m)

    energy = _mpi_allreduce_sum_scalar(energy)
    forces = _mpi_allreduce_sum(forces)
    stress = _mpi_allreduce_sum(stress)
    stress /= volume

    return energy, forces, stress


# Guard rails for the reciprocal-space sum.  Ewald's cost is N * Nk, and
# this implementation builds the (N, Nk) phase matrix, so a cell large
# enough to need more than this many k-vectors wants PPPM instead -- the
# mesh-based solver in coulomb/pppm.py, whose cost scales as M^3 log M
# with the mesh rather than with the number of k-vectors.
_K_TRIPLET_LIMIT = 5_000_000       # before the spherical mask
_K_VECTOR_LIMIT  = 1_000_000       # after it
_K_CHUNK_ELEMS   = 4_000_000       # bound the (N, chunk) phase matrix


def _auto_k_shells(recip: np.ndarray, alpha: float,
                   accuracy: float) -> Tuple[np.ndarray, float]:
    """Per-axis integer shell counts reaching a PHYSICAL |k| cutoff.

    The reciprocal sum's terms carry exp(-k^2 / 4 alpha^2), so truncating
    at ``k_cut = 2 alpha sqrt(-ln accuracy)`` leaves a relative residue of
    about ``accuracy``.  Converting that to index space needs one shell
    count per axis, because |b_i| = 2 pi / (perpendicular width along i)
    differs from axis to axis -- which is exactly what a single shared
    integer cutoff cannot express.
    """
    k_cut = 2.0 * alpha * math.sqrt(-math.log(accuracy))
    b_norm = np.linalg.norm(recip, axis=1)
    return np.maximum(1, np.ceil(k_cut / b_norm).astype(int)), k_cut


def ewald_sum(lattice: np.ndarray,
              positions: np.ndarray,
              charges: np.ndarray,
              alpha: Optional[float] = None,
              k_max: Optional[int] = None,
              r_cut: float = 12.0,
              accuracy: float = 1e-8) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Ewald summation -- fully vectorised (no Python loops over atoms or k).

    Vectorisation scheme
    --------------------
    Real-space  : see :func:`real_space_coulomb` (O(N^2) minimum-image path).

    Recip-space : Build all (K, 3) k-vectors at once via meshgrid.
                  Phase matrix phi = positions @ kvecs.T  has shape (N, K).
                  Structure factors S(k) = charges @ exp(i*phi)  shape (K,).
                  Force   = -(q[:,None]) * (weighted_imag_mat.T @ kvecs)
                  where weighted_imag_mat[k,i] = pref[k]*Im(e^{i phi_ik}*S*[k]).

    Complexity  : O(N^2) real-space, O(K.N) recip-space --- same asymptotic
                  as the loop version but ~100x faster in practice.

    MPI         : when running under mpirun with >1 rank, the flattened
                  real-space pair list (inside :func:`real_space_coulomb`)
                  and the flattened k-vector list here are each split across
                  ranks (:func:`_mpi_domain_split`); every rank still builds
                  the full (N,N) minimum-image matrix and the full k-vector
                  grid (cheap relative to the erfc/exp work), but only
                  evaluates its own slice of pairs/k-vectors before an
                  ``Allreduce(SUM)`` combines the partial energy, forces,
                  and stress. With a single rank (no MPI, or mpirun -n 1)
                  this reduces to exactly the serial algorithm.

    Constraint  : r_cut < min_i(d_i) / 2, where d_i = V / |a_j x a_k| is the
                  PERPENDICULAR width (interplanar spacing) of the cell along
                  axis i (minimum-image assumption of the real-space branch).
                  NOT ENFORCED HERE -- see "Unvalidated precondition" below.

    Bug-fix notes (kept here so a future edit does not silently reintroduce
    any of these issues)
    ----------------------------------------------------------------------
    1. Real-space force sign: for ``dr_m = r_j - r_i`` and ``de_dr = dV/dr``,
       the force on atom i is ``F_i = +(dV/dr) * dr_m / r`` (chain rule:
       ``F_i = -dV/dr_i = -(dV/dr)(dr/dr_i) = -(dV/dr)(-dr_m/r)``). An
       earlier version computed ``fij = (-de_dr/r) * dr_m``, an extra
       spurious minus sign that silently flipped the sign of every
       real-space Coulomb force and stress component (energy was
       unaffected, since it does not depend on direction). Verified two
       independent ways: (a) direct chain-rule derivation above, and (b) a
       repulsive-like-charges sanity check (for qi=qj>0 at short range,
       de_dr<0, so the correct force must point away from the other atom;
       the buggy formula pointed the other way, i.e. attractive). Now fixed
       in the single shared implementation, :func:`real_space_coulomb`.
    2. Stress volume normalisation: the real-space and reciprocal-space
       stress contributions must each be divided by the cell volume V
       exactly once. An earlier version accumulated both contributions
       into one running ``stress`` variable and called ``stress /= volume``
       after *each* section, which divides the real-space part by V a
       second time (V^2 total) while the reciprocal part is divided once
       -- a dimensionally-inconsistent result. This version keeps the two
       contributions in separate accumulators (:func:`real_space_coulomb`
       returns its own, already divided by V once) and divides the
       reciprocal-space part by V exactly once before summing them.
    3. Reciprocal-space stress tensor structure: an earlier version used
       ``delta_ab*(1 - k^2/(2 alpha^2)) - k_a k_b/k^2`` (per k-vector,
       weighted by ``coeff``), which is not merely a sign error -- the
       ``k^2/(2 alpha^2)`` correction was attached to the isotropic
       (``delta_ab``) term instead of the anisotropic (``k_a k_b``) term,
       and the ``k_a k_b/k^2`` term was missing a factor of 2. The correct
       formula (derived from ``sigma_ab = -(1/V) dE/dT_ab`` under an affine
       cell deformation ``h' = h @ T``, and cross-checked against the
       standard Ewald/Nose-Klein reciprocal-space virial) is
       ``delta_ab - (2 + k^2/(2 alpha^2)) * khat_a * khat_b`` with
       ``khat = k/|k|``, implemented below. Confirmed by finite-difference
       dE/dstrain agreement to ~1e-7 relative error (the old formula
       disagreed by factors of ~4 to ~12 depending on tensor component,
       i.e. not a subtle rounding issue). The reciprocal-space *energy* and
       *force* formulas were independently verified correct (unaffected).

    Parameters
    ----------
    lattice   : (3,3) lattice vectors [A], rows = a, b, c.
    positions : (N,3) Cartesian positions [A].
    charges   : (N,)  formal charges [e].
    alpha     : Ewald splitting [A^-1]; auto-set to 5/r_cut if None.
    k_max     : LEGACY manual override -- a fixed integer shell count,
                applied identically to all three reciprocal axes and with
                no spherical selection.  Leave it None (the default) to
                have the shells chosen from ``accuracy``.

                This parameter used to default to 8, and that was a
                defect rather than a tuning choice: a shell count is a
                cutoff on the INDEX, not on |k|.  Since
                |b_i| = 2 pi / (perpendicular width along axis i), eight
                shells reach a physical cutoff that SHRINKS as a cell
                axis lengthens, so the accuracy silently depended on cell
                size and shape.  Small fixtures were fine -- rocksalt
                NaCl at a = 5.6 A reaches |k| = 9.0 A^-1, residue 1.7e-9
                -- which is why the finite-difference tests, all of which
                compare forces against the derivative of the same
                truncated energy, never saw it.  A 300-atom wurtzite ZnO
                cell was already 39.2 eV (0.55%) off, and the error is
                always NEGATIVE and grows as a cell stretches, so a
                variable-cell relaxation or an NPT run sees a spurious
                energy gain from expanding.  One did: ZnO ran away from
                3572 to 24751 A^3, with c growing 4.7x, while reporting a
                1967 eV energy DROP that a converged sum puts at a rise.
    accuracy  : target relative residue of the reciprocal sum, used to
                pick the shell counts when ``k_max`` is None.
    r_cut     : real-space cutoff [A].

    Returns
    -------
    energy : float        [eV]
    forces : (N,3) array  [eV/A]
    stress : (3,3) array  [eV/A^3]
    """
    N = len(positions)
    volume = np.abs(np.linalg.det(lattice))
    alpha = 5.0 / r_cut if alpha is None else alpha
    inv_lat = np.linalg.inv(lattice)
    recip = 2.0 * np.pi * inv_lat.T   # rows = b1, b2, b3

    energy, forces, stress = real_space_coulomb(
        lattice, positions, charges, alpha, r_cut, neigh_list=None)

    # =========================================================================
    # Reciprocal-space part  (vectorised over all k-vectors simultaneously)
    # =========================================================================
    # Build the integer (h, k, l) triplets.  Shell counts come per axis
    # from a physical |k| cutoff (see _auto_k_shells and the k_max entry
    # in the docstring); a legacy integer k_max reproduces the old cubic
    # index-space box with no spherical selection.
    if k_max is None:
        n_shell, k_cut = _auto_k_shells(recip, alpha, accuracy)
    else:
        n_shell, k_cut = np.array([int(k_max)] * 3), np.inf

    n_triplet = int(np.prod(2 * n_shell + 1))
    if n_triplet > _K_TRIPLET_LIMIT:
        raise ValueError(
            f"ewald_sum: this cell needs {n_shell.tolist()} reciprocal "
            f"shells ({n_triplet} triplets) to reach |k| = {k_cut:.2f} A^-1 "
            f"at accuracy {accuracy:g}, which is past the practical limit "
            f"for an explicit Ewald sum.  Use coulomb_method='pppm' (the "
            f"mesh solver scales with the FFT mesh, not with the number of "
            f"k-vectors), or raise `accuracy`, or lower `alpha` to move "
            f"work into the real-space sum.")

    H, K, L = np.meshgrid(*[np.arange(-n, n + 1) for n in n_shell],
                          indexing='ij')
    H, K, L = H.ravel(), K.ravel(), L.ravel()

    kvecs_all = (H[:, None] * recip[0] + K[:, None] * recip[1]
                 + L[:, None] * recip[2])
    k2_all = np.einsum('ki,ki->k', kvecs_all, kvecs_all)
    # Spherical selection: (0,0,0) always drops out via k2 > 0.
    keep = (k2_all > 1e-12) & (k2_all <= k_cut ** 2)
    kvecs_all, k2_all = kvecs_all[keep], k2_all[keep]

    # Only the automatic path is capped: an explicit k_max is the caller
    # deliberately overriding, and is how a reference value is computed.
    if k_max is None and len(kvecs_all) > _K_VECTOR_LIMIT:
        raise ValueError(
            f"ewald_sum: {len(kvecs_all)} k-vectors needed at accuracy "
            f"{accuracy:g}; use coulomb_method='pppm' instead, or relax "
            f"`accuracy`.")

    k_start, k_end = _mpi_domain_split(len(kvecs_all))
    kvecs_all, k2_all = kvecs_all[k_start:k_end], k2_all[k_start:k_end]

    recip_energy = 0.0
    recip_forces = np.zeros((N, 3))
    recip_stress = np.zeros((3, 3))

    # Chunked over k: the phase matrix is (N, Nk), which for a large cell
    # is the memory bottleneck of the whole routine.  The arithmetic is a
    # plain sum over k, so chunking changes nothing but the peak
    # footprint.
    chunk = max(1, int(_K_CHUNK_ELEMS // max(N, 1)))
    for c0 in range(0, len(kvecs_all), chunk):
        kvecs = kvecs_all[c0:c0 + chunk]
        k2 = k2_all[c0:c0 + chunk]

        # Prefactors: pref[m] = 4pi k_e / V * exp(-k^2/4alpha^2) / k^2
        pref = (K_E * 4.0 * np.pi / volume
                * np.exp(-k2 / (4.0 * alpha**2)) / k2)             # (Nk,)

        # Phase matrix: phi[i, m] = k[m] . r[i]                    (N, Nk)
        phi = positions @ kvecs.T
        # Structure factor: S[m] = sum_i q_i exp(i phi[i,m])         (Nk,)
        ephi = np.exp(1j * phi)                                    # (N, Nk)
        SF = charges @ ephi                                      # (Nk,)
        sk2 = SF.real**2 + SF.imag**2                             # (Nk,)

        recip_energy += 0.5 * np.dot(pref, sk2)

        # Forces: F[i,alpha] = -(q_i) sum_k pref[k] Im(e^{i phi_{ik}} S*[k]) k_alpha
        # Im(e^{i phi} S*) = sin(phi)*Re(S) - cos(phi)*Im(S)
        imag_mat = (ephi.real * SF.imag[None, :]                    # (N, Nk)
                    - ephi.imag * SF.real[None, :])
        recip_forces -= charges[:, None] * (imag_mat * pref[None, :]) @ kvecs

        # Stress: sigma_ab = (1/V) sum_k coeff[k] * [delta_ab - w[k] * khat_a khat_b],
        # w[k] = 2 + k^2/(2 alpha^2),  khat = k / |k|  -- see bug-fix note 3 above.
        coeff = 0.5 * pref * sk2                                     # (Nk,)
        khat = kvecs / np.sqrt(k2[:, None])                          # (Nk, 3)
        weight = coeff * (2.0 + k2 / (2.0 * alpha**2))                # (Nk,)
        recip_stress += (np.sum(coeff) * np.eye(3)
                          - np.einsum('k,ka,kb->ab', weight, khat, khat))

    recip_energy = _mpi_allreduce_sum_scalar(recip_energy)
    recip_forces = _mpi_allreduce_sum(recip_forces)
    recip_stress = _mpi_allreduce_sum(recip_stress)

    energy += recip_energy
    forces += recip_forces
    stress += recip_stress / volume   # divide remaining (real-space already divided)

    # =========================================================================
    # Self-energy and charged-system corrections
    # =========================================================================
    energy -= K_E * alpha / np.sqrt(np.pi) * np.dot(charges, charges)
    q_tot = np.sum(charges)
    if abs(q_tot) > 1e-8:
        energy -= K_E * np.pi * q_tot**2 / (2.0 * alpha**2 * volume)

    return energy, forces, stress
