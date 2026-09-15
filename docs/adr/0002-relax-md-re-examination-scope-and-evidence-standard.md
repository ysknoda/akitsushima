---
status: accepted
---

# Scope and evidence standard for the relax/md re-examination

A re-examination of the package for defects was opened with relax (geometry
optimisation) and md (molecular dynamics) as the stated priorities, on the
grounds that they are the base functionality everything else builds on. Six
decisions were settled before any investigation began; they are recorded here
because several of them cost real time, and because a later reviewer working to
a looser standard would produce findings that cannot be compared against these.

## Decisions

1. **Investigation is parallelised across read-only sub-agents**, one for the
   relax path and one for the md path. Each is given the verified conventions
   from ADR-0001 up front, so its effort goes into testing them rather than
   re-deriving them and re-reporting them as bugs.
2. **Scope includes CLI commands, YAML templates, and docstrings**, not just
   solver internals. A template key that does not match what the loader reads,
   or a docstring that contradicts the formula beside it, counts as a defect.
   This is not hypothetical: separate past defects had a GCMC template using
   `epsilon_Ar_Ar` where the code required `epsilon`, an SW/Tersoff fit template
   missing the `r_cut` key the command unconditionally reads, and a docstring
   describing an excess chemical potential that the acceptance formula never
   used.
3. **Findings are split by severity rather than all fixed or all deferred.**
   Defects that produce silently wrong numbers are fixed immediately with a
   discriminating regression test. Anything that changes physical output values,
   a default, or an API contract is reported for the user's decision instead.
4. **Every finding about energy, force, stress, or a cell gradient must be
   reproduced numerically** -- finite-difference oracle, multi-rank `mpirun`
   comparison, or refactoring-equivalence, matching the three patterns already
   established in this repository. Read-only observations are reported
   separately as SUSPECTED and never mixed in with confirmed defects. The reason
   is empirical: the reverse-space Ewald stress tensor, the three-body stress
   sign in Stillinger-Weber and Tersoff, the cell-gradient multiplication order,
   and the NPT pressure sign were all textbook-looking on the page and were all
   caught only by finite differences.
5. **"Bug G" is in scope.** `BasePairPotential.compute()` applies the minimum-
   image convention in its `neigh_list is None` fallback without checking that
   `r_cut` is under half the minimum perpendicular cell width, and returns
   silently wrong results when it is not. It was previously documented and
   deferred. It is relevant here because `GeometryOptimiser` obtains its force
   function through `_resolve_force_fn`, so a caller-supplied `force_fn` can
   reach the unguarded path.
6. **Behaviour and default changes are permitted at this release stage.** The
   package is at `0.1.0b1` and its README carries the standard beta notice
   stating that bugs may still be fixed before a stable release. Any such change
   bumps the version to `0.1.0b2` and is stated explicitly rather than folded in
   silently.

## Consequences

Decision 4 is the expensive one: it rules out reporting a defect on the strength
of a code read, which is slower and will produce fewer findings per hour than a
review that does not hold that line. That is the intended trade: in this
codebase, reading has a track record of both missing real sign errors and
producing confident false alarms about correct ones (see ADR-0001).
