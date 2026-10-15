# Upstream Pint follow-up

**Status:** Active
**Last updated:** 2026-10-15
**Scope:** Pint defects and extension opportunities that affect xpint's workarounds and supported range.

## Context

xpint supports Pint 0.25.3 through the 0.25 line and works around several Pint
defects. An upstream merge is not enough to remove a workaround: the fix must be
in a supported Pint release and xpint's public regressions must pass against it.
Pull-request states below were last checked on 2026-09-18 and must be refreshed
before acting.

## Current Decision

### Tracked contributions

| Issue / PR | State at check | xpint impact and next action |
| --- | --- | --- |
| [#2410](https://github.com/hgrecco/pint/pull/2410), [#2400](https://github.com/hgrecco/pint/issues/2400): `sum`/`nansum` quantity `initial` units | Merged 2026-09-10 | Find the first release containing it; keep xpint's conversion guards until then |
| [#2418](https://github.com/hgrecco/pint/pull/2418): bare numeric `sum` initial | Open | Its policy differs from xpint's in-unit scalar initial; do not adopt it implicitly |
| [#2422](https://github.com/hgrecco/pint/pull/2422), [#2254](https://github.com/hgrecco/pint/issues/2254): variadic `wraps` binding | Open | xpint's guard probes whether misplaced-slot protection is needed, and its adapter tests pass against the PR; retest the release that ships it |
| [#2423](https://github.com/hgrecco/pint/pull/2423): failed in-place power changes units | Open | Keep xpint's transactional NumPy staging until Pint's failure and alias contract is satisfied independently |
| [#2424](https://github.com/hgrecco/pint/pull/2424), [#1938](https://github.com/hgrecco/pint/issues/1938): stale dimensionality after mutation | Open | Retest power, multiplication/division, and context conversion before removing cache invalidation |
| [#2425](https://github.com/hgrecco/pint/pull/2425): nan-extrema quantity `initial` units | Open | Classified nan-extrema support still needs an xpint propagation contract |
| [#2426](https://github.com/hgrecco/pint/pull/2426): `check` variadic binding | Open draft | Fixed signatures remain the documented safe subset on 0.25.3 |

### Candidate reports

Search open and closed issues before drafting either report.

- **Offset equality at zero.** `PlainQuantity.__eq__` returns true for two
  all-zero magnitudes of equal dimensionality before converting, so
  `Q(0, "degC") == Q(0, "degF")` is true although the temperatures differ. xpint
  compares offset quantities in base units instead.
  ```python
  import pint

  u = pint.UnitRegistry()
  assert u.Quantity(0, "degC") != u.Quantity(0, "degF")  # fails on 0.25.3
  ```
- **Quantity `initial` for `prod`/`nanprod`.** The initial multiplies into the
  output unit, so it cannot share the additive fix. Recheck against current Pint
  first:
  ```python
  import numpy as np
  import pint

  u = pint.UnitRegistry()
  q = u.Quantity(np.array([2.0, 3.0]), "meter")
  np.prod(q, initial=4 * u.second)  # physically expected: 24 meter**2 * second
  ```

### Inherited limitations without an xpint fix

- Pint-left equality (`pint_quantity == xpint_quantity`) compares physically and
  bypasses xpint's ownership and semantic checks. A symmetric contract needs an
  upstream change or a composition-based representation.
- `GenericUnitRegistry` lacks some convenience adapters of the concrete
  `UnitRegistry`; xpint supplies its own guarded versions.

### Extension opportunities

Each needs a minimal Pint-only motivating example before any proposal:

1. Small-array performance in Pint's output-unit inference, preserving offset,
   logarithmic, and subclass behavior.
2. Registry selection during unpickling ([#2079](https://github.com/hgrecco/pint/issues/2079),
   [#1815](https://github.com/hgrecco/pint/issues/1815)); xpint already
   reconstructs against a scoped registry.
3. A stateful result hook, informed by the kind ([#1967](https://github.com/hgrecco/pint/pull/1967))
   and uncertainty ([#1797](https://github.com/hgrecco/pint/pull/1797)) work. A
   universal metadata framework is not the goal.

## Alternatives Considered

Duplicate fixes, numeric fixes bundled with policy changes, and a universal
metadata framework designed ahead of a concrete need are all avoided.

## Deferred Work

Contribution resourcing and any new public extension API are maintainer decisions.

## Next Steps

Refresh the PR states, then retest each candidate Pint release against xpint's
public regressions before widening the supported range. Submitting an issue or
PR needs specific maintainer authorization.
