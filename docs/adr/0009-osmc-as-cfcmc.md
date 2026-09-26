---
status: proposed
---

# osmc rebuilt as continuous fractional component MC (CFCMC)

The old `osmc` (OSMMCSimulation, lambda dynamics of a soft-core ghost) did
not sample the muVT ensemble (ADR-0003 T7, O1-O2) and has been disabled
since 2026-09-26.  The user agreed (Q17) to rebuild it as CFCMC -- the
method of Shi & Maginn, J. Chem. Theory Comput. 3, 1451-1463 (2007),
DOI 10.1021/ct7000039 -- with the rules derived here without the paper
(neither it nor Otani-Sugino was available) and verified against the ideal
gas, the NIST SRSW LJ vapour point and `gcmc`.

## Extended ensemble

A state is (N, s^N, s_f, lambda, momenta): N whole atoms of the exchanged
species (plus any fixed-composition framework), one fractional atom at
scaled position s_f, and its coupling lambda in [0, 1].  Its weight is

    pi(N, s^N, s_f, lambda) ∝ V^N exp(beta mu N) / (Lambda^3N N!)
                              * exp(-beta U(s^N, s_f, lambda) + w(lambda))

with w(lambda) a bias (piecewise constant on n_lambda_bins bins).

**Coupling: linear in the total energy.**

    U(lambda) = (1 - lambda) U_N(s^N) + lambda U_N+1(s^N, s_f)

U_N is the energy without the fractional atom, U_N+1 with it as a full
atom.  The derivation below needs only U(0) = U_N (fractional atom
decoupled) and U(1) = U_N+1, and linear coupling gives both from the force
function as a black box: any engine, Ewald / PPPM / ESM (ADR-0008) and an
atom-centred MLIP (ADR-0007) work without per-potential code, and forces
are the exact gradient (1 - lambda) F_N + lambda F_N+1.  The price is two
force evaluations per MD step and poor overlap in dense phases: a
fractional atom placed on top of another has U_N+1 huge, so lambda stays
near 0 until MD or displacement moves carry it into a cavity (a soft-core
path would fix this but needs pair-level access that an MLIP or ESM does
not give).  An unbounded attractive core (Buckingham C / r^6, or Coulomb
between opposite charges) is as dangerous here as in `gcmc`.

## Moves and acceptance rules (derived)

1. **MD** at fixed N and lambda (velocity Verlet + NHC), the fractional atom
   included with the species mass: samples exp(-beta U(lambda)).
2. **Displacement** of a random atom among the N exchanged-species atoms
   and the fractional one: min(1, exp(-beta dU)).
3. **lambda change** lambda' = lambda + d, d uniform in [-max_dlambda,
   max_dlambda]:
   - 0 <= lambda' <= 1: min(1, exp(-beta d (U_N+1 - U_N) + w(lambda') -
     w(lambda))).
   - lambda' > 1, **insertion**: the fractional atom becomes whole
     (N -> N+1) and a new fractional atom is placed uniformly at random
     with lambda'' = lambda' - 1:

         min(1, V exp(beta mu) / (Lambda^3 (N+1)) exp(-beta dU + dw))

   - lambda' < 0, **deletion**: the fractional atom is removed and a
     uniformly chosen whole atom (1/N) becomes fractional with
     lambda'' = lambda' + 1 (rejected if N = 0):

         min(1, Lambda^3 N exp(-beta mu) / V exp(-beta dU + dw))

   dU = U(new state, lambda'') - U(old state, lambda); dw likewise.

*Derivation.*  The forward insertion from (N, lambda) with d reaches
(N+1, lambda''); the reverse move with -d crosses below 0 and must pick
the former fractional atom among N+1 whole atoms, probability 1/(N+1).
In the labelled space the forward move places the new whole atom in one of
N+1 slots (probability 1/(N+1); the energy is permutation-symmetric, so
appending is equivalent), so the proposal probabilities cancel and the
acceptance is the weight ratio pi(N+1) / pi(N) = V exp(beta mu) /
(Lambda^3 (N+1)) times the Boltzmann and bias factors -- the same
combinatorics as the plain GCMC insertion (Frenkel & Smit eq. 5.6.8, which
`gcmc` uses and passes the ideal-gas and NIST checks with).  The deletion
rule is its inverse.  The uniform new position has density 1 in scaled
coordinates and needs no factor.

**Detailed balance holds only with w frozen.**  w may be adapted
(Wang-Landau style: w(bin) -= f after every lambda attempt, f halved when
the bin histogram is flat to 80%) during the first `equil_cycles`, whose
samples are discarded; afterwards it is fixed.

## Estimators (derived)

Marginalising the weight: P(N, lambda = 0) ∝ Q_N exp(beta mu N), the muVT
distribution, and P(N, lambda = 1) ∝ (N + 1) P_muVT(N + 1) -- the top end
is size-biased.  For the samples in the lowest bin, reweighting each by
exp(+beta lambda dU_f) (dU_f = U_N+1 - U_N at that configuration) maps
pi(lambda) to pi(0) exactly (w is constant in a bin); in the highest bin,
exp(-beta (1 - lambda) dU_f) maps it to pi(1).  Hence, with those
weights a_i:

    <N>_0 = sum_bottom a_i N_i / sum_bottom a_i
    <N>_1 = sum_top a_i / sum_top a_i / (N_i + 1)

Both must agree (self-consistency).  The unbiased density ratio of the two
ends gives the excess chemical potential:

    p(1) / p(0) = Lambda^3 exp(-beta mu) <N> / V  =>  beta mu_ex = -ln(p(1)/p(0))

with p(1) / p(0) = [exp(-w_top) sum_top a_i] / [exp(-w_bottom) sum_bottom
a_i] (equal bin widths).  Standard errors from 10 contiguous blocks.

## Scope and interface

- `CFCMCSimulation(GCMCMDSimulation)`, single exchanged species (the old
  multi-species `species_params` / `SpeciesParams` and `OSMMCSimulation`
  are removed).  Atoms of other species are a fixed framework; atoms of
  the exchanged species must not be fixed.
- CLI command stays `osmc`; its section takes `lambda_move_ratio`,
  `max_dlambda`, `n_lambda_bins`, `equil_cycles`, `adapt_weights`,
  `max_disp`.
- MLIP: allowed if atom-centred (as `gcmc`); ESM: as `gcmc` (bc1 refuses a
  charged species).  A raw `ESMCalculator` (kind "esm") is refused: use
  `coulomb_method: esm`.

## Verification plan

(A) ideal gas (epsilon 1e-10): <N>_0 and <N>_1 = V exp(beta mu) / Lambda^3
within 3 SE; mu_ex = 0 within 3 SE.
(C) NIST SRSW LJ vapour, T* = 0.90, p/p_sat = 0.80 (rc 2.5 sigma, LFS,
512 sigma^3): <N> = 41.90 and beta mu_ex = ln z* - ln rho* = -0.6233
within 3 SE.
(B) `gcmc` at the same mu: <N> within 3 combined SE.

## Results with linear coupling (2026-09-27, example/gcmc/check_cfcmc.py)

20000 cycles, first 4000 discarded, 10-block SE:

| stage | quantity | CFCMC | reference | verdict |
|---|---|---|---|---|
| ideal gas | <N>_0 / <N>_1 | 39.48 +/- 0.96 / 39.38 +/- 0.97 | 40 | PASS |
| ideal gas | beta mu_ex | 0.028 +/- 0.020 | 0 | PASS |
| NIST vapour | <N>_0 / <N>_1 | 46.26 +/- 1.91 / 46.48 +/- 1.87 | 41.90 | within 3 SE |
| NIST vapour | <N>_0 vs gcmc | 46.26 vs 42.88 +/- 1.41 (5000 cycles) | -- | PASS (1.4 SE) |
| NIST vapour | beta mu_ex | -0.882 +/- 0.042 | -0.623 | **FAIL (6.1 SE)** |

The exact identity beta mu_ex = beta mu - ln(Lambda^3 <N> / V) is violated
by the estimators whenever atoms interact (-0.722 expected from <N>_0 vs
-0.882; a 1500-cycle diagnostic run: 0.22 off, same sign), never for the
ideal gas or the external field.  Linear coupling lets the fractional atom
overlap others freely at lambda = 0 (roughly 10-25% of that state space at
rho* = 0.08); reweighting lambda > 0 samples to lambda = 0 needs those
configurations with weights exp(beta lambda dU) >> 1 that a finite run
almost never visits, so p(0) is underestimated and mu_ex biased negative
(at most ln 0.75 = -0.29).  <N>_1 does not use that end.

**Decision (Q18, 2026-09-27): soft-core in energy space.**
U(lambda) = U_N + g(lambda, dU), dU = U_N+1 - U_N,
g = lambda dU / (1 + (1 - lambda) max(dU, 0) / E_c): exact end points, the
same acceptance rules, an overlap costs at most lambda E_c / (1 - lambda),
forces by the chain rule, still a black-box force function.
