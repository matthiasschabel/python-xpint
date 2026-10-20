# Unsupported behavior and future work

**Status:** Deferred
**Last updated:** 2026-10-19
**Scope:** xpint features outside the prepared beta contract and unresolved design choices.

## Context

The beta is a bounded semantic extension, not full compatibility with every Pint
or scientific-Python entry point. [Compatibility](../compatibility.md) records
the current boundaries; this document records what would justify extending them.
Items below are not promises or prerequisites for releasing the documented beta.

## Current Decision

Keep unsupported classified operations explicit. Ordinary physical operations
continue through Pint subject to the documented ownership and correctness guards.
Keep `pint>=0.25.3,<0.26` until a new compatibility matrix supports expansion.

### Deferred capabilities

| Area | Current boundary | Prerequisite for extending it |
| --- | --- | --- |
| Exponent identity | Designated strings parse powers as floats, as Pint does; a unit built with a non-dyadic `Fraction` power formats identically but parses back unequal | Normalize physical exponents or change unit equality |
| Formatting | Designated default/abbreviated text and xpint rich display; direct Pint/Babel formatting rejects designations | Explicit format-specific preservation rules, including rational physical exponents that Pint 0.25 cannot format |
| Group/root designations | Factor contributions only | Resolve scope and algebra separately from parser syntax |
| Measurement containers | Classified Measurement operands/targets reject; explicit Quantity construction can carry an uncertainty magnitude | Design propagation and serialization across Pint's separate Measurement hierarchy; sequence/container behavior needs its own contract |
| Discrete and broader NumPy arithmetic | Classified floor division, modulo, divmod, dot/prod and undefined dispatch reject | Define result kinds, designation algebra, affine rules, reductions and mutation before expanding allowlists |
| Scientific adapters | Bounded Dask/xarray/masked-array support; no broad pandas/JAX or arbitrary graph-transform claim | Backend-specific tests that check shape, dtype, laziness, ownership, and preserve/drop/reject behavior |
| pint-xarray metadata round trips | Ordinary and designated DataArray unit text round-trip; kinds are not serialized | Define an explicit kind-metadata schema and test broader Dataset/coordinate workflows before claiming general semantic interchange |
| Cross-language parity | JSON fixture has an xpint consumer only | Implement and run a MATLAB consumer and record versioned results from both libraries |
| Durable interchange | Python pickle requires compatible caller-owned registry definitions | Versioned schema and explicit registry/definition identity; consider external vocabulary mappings before claiming interoperability |
| Semantic vocabulary | No chemistry identity resolution, molar-mass inference, ontology lookup, or UCUM certification | Concrete use case; evaluate an existing UCUM bridge instead of replacing Pint's definition language |
| Other affine domains | Strict policy is thermodynamic-temperature only | Separate design for gauge pressure or other origins; do not infer it from temperature rules |
| Angle configuration | Stock definitions at construction; no supported custom contexts or runtime mode toggle | Consistent cache, catalog, conversion, and definition-compatibility model |

Native designated token strings shipped in 0.1.0b4;
universal exponent round-trip identity remains blocked as noted above. The
grammar, Pint extension boundary, and cache isolation are recorded in
[architecture](../architecture.md#native-designated-strings).

### Classified NumPy coverage

Offset-unit arrays are classified by their unit, so Celsius users meet the
classified allowlist without opting in (`np.sort`, `np.median`, `np.isclose`, and
`np.interp` raise; the README documents the `to("kelvin").as_unkinded()` exit).
Expanding coverage needs per-operation contracts, not family-wide rules: plain
Pint 0.25.3 itself returns `np.average(..., returned=True)` weight sums in degC
and fails on `np.linspace(..., retstep=True)`. Proposed order, each tier with
public tests for Absolute, Delta, kinded, and designated inputs, mixed scales,
and every option that returns or accepts a unit-bearing value:

| Tier | Operations | Contract |
| --- | --- | --- |
| 1. Single input, result keeps input state | `sort`, `nanmax`/`nanmin`, `median`/`nanmedian`, `percentile`/`quantile` (and `nan*`), `unique` values | Same state as input. Interpolating quantiles are affine combinations with weights summing to one, so Absolute is valid. `unique` auxiliary outputs (`return_index`, `return_counts`) are plain integers |
| 2. Same-state combination and selection | `append`, `maximum`/`minimum`/`fmax`/`fmin`, function `np.clip` | `append` follows `concatenate`: every operand must have compatible state and converts to the first operand's scale. Selection requires exact semantic state, as for ordering; `np.clip` routes to the governed method and keeps its unkinded-bound exception |
| 3. Mixed-role arguments | `isclose`/`allclose`, `interp`, `linspace`, `average`, `full_like`, `round` | For an Absolute input, `atol`, the `linspace` step, and the `interp` `period` are Deltas; `rtol` against a point is scale-dependent and should require `rtol=0` or reject. `interp` separates coordinate state (`x`, `xp`) from value state (`fp`). `average` needs normalized weights, and `returned=True` yields a plain weight sum. Rounding a point depends on its scale but is defined within one, so preserve state |
| Keep rejecting | `dot`, `prod`, `cross`, `trapezoid`, `linalg.norm` on Absolute values; `ones_like` on any classified input | Products and integrals of points have no affine meaning; a kinded `ones_like` invents a value. Revisit kinded (non-affine) products only with a kind-algebra decision |

`np.testing.assert_allclose` will still reject classified inputs because it
exports them to `ndarray`; once `allclose` is supported, document
`assert np.allclose(...)` or comparison of `m_as()` magnitudes for tests.

### Open design questions

**Designation scope remains a product decision.** The current signature equates
`m["cap"]**2` with `(m**2)["cap"]`: it labels a physical dimension contribution,
not an indivisible geometric measure. Whether users need a whole-measure
distinction is unresolved; a distinct representation needs a motivating case and
an explicit compatibility and migration decision.

**Kind-product retention is a possible stronger model.** Dropping the kind of
Force × Length loses the difference between a derived value and an originally
unkinded value. Keeping a product for later checked classification would require
relations, a hierarchy, and an ambiguity policy. Do not add automatic naming or
composite kinds without a real misuse case. `compatible_kinds()` discovery and
reference units already exist; they are not outstanding implementation tasks.

**Affine role is encoded through kinds.** Absolute and Delta roles live in the
built-in `Temperature` and `TemperatureDifference` kinds, so `define_kind()`
rejects other kinds over temperature dimensions. Supporting domain kinds such as
a sensor temperature needs the role stored separately from the kind, with
equality, hashing, conversion, and pickling updated together.

### Maintenance and performance

- Repeat the inherited public and private Pint surface inventory before changing
  dependency support. Context/parsing/registry coverage is targeted, not an
  exhaustive GenericUnitRegistry audit. A CI-visible surface diff is an option
  if dependency churn warrants it.
- Test NumPy/runtime dependency floors explicitly before claiming a broader
  matrix. Existing local beta results use Pint 0.25.3 and NumPy 2.5.2.
- Performance follow-ups from the pre-release audit, each to be profiled before
  changing: explicit `ndarray` export with a dtype change and `copy=True` copies the
  converted array a second time; `from_sequence` recomputes the first element's
  semantic state for every comparison; designation signatures are recomputed
  several times per operation, which matters for scalars and small arrays.
- Benchmark actual workloads with the retained harnesses. Persistent staging
  buffers might save allocation but need shape/dtype/mask, nesting, concurrency,
  and ownership rules. Preserve transaction guarantees when optimizing.
- Further splitting `registry.py` or improving static magnitude typing requires
  a concrete maintenance/type-checking benefit; dynamic Pint binding constrains
  both. Neither is a release blocker on file size alone.
- Consider pinning the remaining major-version GitHub Actions references to
  reviewed commit SHAs during CI maintenance; setup-uv is already SHA-pinned.
- Track ordinary-Pint comparison asymmetry and adapter signature bugs through
  [upstream work](upstream_pint.md), rather than loosening xpint's semantic checks.

## Alternatives Considered

Automatically accepting every NumPy/Pint operation would silently lose metadata.
A universal metadata framework would create policy and performance costs without
evidence that one abstraction fits wrappers, plotting, mutation, and serialization.
Broad integration claims based only on scalar smoke tests are insufficient.

## Deferred Work

The tables above are the maintained backlog. Prioritize one concrete consumer or
reproducer at a time; current release logistics live in the
[release checklist](release_checklist.md).

## Next Steps

Choose a scoped item when there is user demand, define observable acceptance
criteria, and move completed decisions into architecture/compatibility. Remove
completed backlog entries instead of retaining struck-through plans.
