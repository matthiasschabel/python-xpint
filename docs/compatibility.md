# Compatibility and extension boundaries

The prepared beta supports Python 3.12+ and Pint `>=0.25.3,<0.26`.
Local release evidence covers Python 3.12–3.14 with Pint 0.25.3 and NumPy 2.5.2;
see [validation](dev/validation.md). The bound is deliberate: xpint reuses Pint
private hooks as well as public protocols. It is not certification of all future
patches in that range or every optional backend.

This is the maintained extension inventory. The [README](../README.md) supplies
user examples; [architecture](architecture.md) explains decisions. An inherited
method need not be overridden when it only inspects physical state or already
reaches a governed xpint operation. Tests verify public behavior, not the mere
presence of an override.

## Core operations

| Surface | Contract | Public regression suites |
| --- | --- | --- |
| Construction, copying, `units`/`u`, iteration and indexing | Preserve semantic state; foreign registries reject; explicit target units follow conversion rules | [registry](../tests/test_registry.py), [kinds](../tests/test_kinds.py), [designations](../tests/test_designations.py) |
| `from_list` / `from_sequence` | Require homogeneous semantic states; uncertainty sequences are outside the scalar float64 sequence contract | [registry](../tests/test_registry.py) |
| Addition/subtraction | Physical compatibility first, then designation/affine policy and equal stored kinds (including both absent) | [affine temperature](../tests/test_affine_temperature.py), [kinds](../tests/test_kinds.py) |
| Multiplication/division/power | Defined kind drop or scalar retention; factor algebra; Absolute points and classified exponents reject | [kinds](../tests/test_kinds.py), [designations](../tests/test_designations.py), [NumPy](../tests/test_numpy.py) |
| Conversion, compact/preferred/unprefixed/reduced variants | Preserve source designations on plain targets; explicit designated targets must match; kinds cannot survive dimension-changing contexts | [registry](../tests/test_registry.py), [designations](../tests/test_designations.py), [Pint parity](../tests/test_pint_passthrough.py) |
| In-place arithmetic/conversion | NumPy casting and aliases preserved; stage changes before commit, refresh dimensionality after unit changes | [registry](../tests/test_registry.py), [Pint parity](../tests/test_pint_passthrough.py) |
| Angle mode | Fixed per registry, stock definitions, derived solid angle and conjugate rotational units | [angle dimension](../tests/test_angle_dimension.py) |

General preferred-unit optimization may require Pint's optional solver packages;
coverage of simple explicit targets does not certify every optimization path.
Raw physical dimension checks (`check` on quantities, `dimensionality`,
`dimensionless`) keep Pint's meaning. `compatible_units()` is a physical catalog,
not a semantic compatibility result. `compatible_kinds()` discovers classifications
without reclassifying the input.

## Comparison and extraction

Quantity equality and ordering require exact semantic state; mismatches broadcast
both magnitudes, preserving supported array/Dask shape. Arithmetic and ordering
reject foreign registries with `ValueError`, matching Pint's own cross-registry
error; equality between different xpint registries is false.
Offset-unit equality bypasses Pint's all-zero shortcut, which would equate
`0 degC` with `0 degF`, and converts before comparing.
Scalar hashes incorporate normalized physical value, registry identity, and
semantic state. Unclassified dimensionless values hash like equal plain numbers.
Unit equality is structural, as in Pint: `Unit("mmol[X]*min/L") == Unit("mM[X]*min")`
is false, like `Unit("mmol/L") == Unit("mM")`, while the quantities
`Quantity(1, "mmol[X]*min/L") == Quantity(1, "mM[X]*min")` compare equal. Compare
quantities, or use `is_compatible_with`, to test physical and semantic equivalence.

There is an inherited mixed-Pint limitation: xpint-left comparison with an ordinary
Pint quantity rejects, while Pint-left dispatch may compare physically and return
true. Hash/equality consistency is claimed within the xpint domain, not across
that asymmetric boundary. [Registry tests](../tests/test_registry.py) record it.

`is_compatible_with(quantity)` requires exact semantic state before physical or
context compatibility. With a unit or string target it asks whether the semantic
conversion is allowed: kinds need not disappear and a plain target adopts source
designations. This makes the predicate directional where conversion is directional,
such as Absolute Celsius to Kelvin versus Generic Kelvin to Celsius.

`.magnitude` and `.m` deliberately expose data. `m_as(target)` extracts after an
explicit semantic conversion. Registry `convert()` validates semantic unit objects,
but plain strings/physical containers select a physical path without metadata
checks. Scalars (`float`, `int`, `complex`) and implicit ndarray export reject
retained classification. `bool()` tests magnitude truthiness rather than extraction.

## Arrays and scientific adapters

[_policies.py](../src/xpint/_policies.py) holds the allowlists; an undefined
classified operation raises. Ordinary values delegate to Pint with xpint's
ownership checks and guards against unit-blind `out` and `initial` handling.

| Family | Current semantic behavior |
| --- | --- |
| Arithmetic, comparison, signs, shape methods, copy, `zeros_like`, composition (`concatenate`, `stack`, `where`) | Preserve metadata according to each operation's contract; Absolute restrictions still apply |
| `sum`/`nansum` and `cumsum`/`nancumsum` | Additive rules; quantity-valued initials, where supported, require matching stored kinds; Absolute inputs reject |
| `mean`/`nanmean`, extrema | Preserve classification; quantity-valued extrema initials require exact semantic state |
| `std`/`nanstd` | Preserve units/kind/designations; Absolute inputs and output targets reject |
| `var`/`nanvar` | Square units/designation powers, drop kind; Absolute inputs and output targets reject |
| `diff` | Preserve kind/designations, or Absolute to Delta with TemperatureDifference; zero-order retains input |
| Roots, square, reciprocal | Power semantics: drop kind, transform designation powers; Absolute rejects |
| Index operations (`argmax`, `argmin`, `argsort`, `nonzero`) | Return plain indices, including for Absolute inputs |
| Trigonometry | Drop input kinds, reject retained designations, delegate dimensional checks to Pint; `arctan2` requires exact semantics and rejects Absolute inputs; results unkinded |
| Masked `filled` / `compressed` | Preserve classified metadata and, in angle mode, angular units; ordinary Pint behavior may return a bare array |
| Dask `compute`, `persist`, `rechunk` | Preserve classified metadata; arbitrary classified `map_blocks` rejects |
| xarray | Tested reductions, composition, indexing/transpose, Dask computation and multiplication by ordinary xpint units |
| pint-xarray | Ordinary quantification and designated DataArray unit-text round trips with `force_ndarray_like=True`; default/abbreviated formatting; no quantity-kind serialization |

Supported reductions normalize positional/keyword options before validation.
Bare scalar initials and masked fills are magnitudes in the existing semantic
unit. Quantity-valued initials must convert physically and match stored kinds;
extrema require exact semantic state. Cumulative sums also reject output
targets. These are per-operation contracts, not blanket `out=` support.

Classified dynamic forwarding permits array metadata and governed methods;
other methods reject on call and ungoverned storage properties raise AttributeError.
Explicitly accessing `.magnitude` leaves this boundary. The guard inspects Pint's
returned attribute, so it is not a guarantee against arbitrary backend property
side effects. Mutation failure safety is NumPy-only.

Evidence: [NumPy](../tests/test_numpy.py), [duck arrays](../tests/test_duck_arrays.py),
[Pint parity](../tests/test_pint_passthrough.py). pandas and arbitrary backend
transforms have no supported semantic contract.

## Formatting, serialization, and registry adapters

| Surface | Contract | Tests |
| --- | --- | --- |
| Native designated strings (since 0.1.0b4) | Single-token labels in `Unit`, `Quantity`, the registry call, `parse_units`, `parse_expression`, `to`/`ito`/`m_as`/`to_compact` targets, and registry `convert`; other string entry points (`get_dimensionality`, `get_base_units`, `check`) reject them with Pint's `UndefinedUnitError`, and `unit == "mM[Fe]"` returns `False` (no string equality). `is_compatible_with` accepts a designated string as a conversion target: true for matching designations, false for mismatches. Default/abbreviated unit text round-trips; powers parse with the registry's `non_int_type`, as in Pint. Repeated identical labels cancel as repeated plain names do. Operator syntax only after a label: Pint's word forms (`m[cap] squared`, `square m[cap]`) on a designated token raise `ValueError` naming the operator form (since 0.1.0b5; earlier releases raise Pint's `UndefinedUnitError`) | [registry](../tests/test_registry.py), [conformance](../tests/test_conformance.py), [pint-xarray](../tests/test_duck_arrays.py) |
| Designated `str` / `format` / repr and rich display | Default and abbreviated text with numeric specs; unsupported unit specs reject; repr includes kind | [formatter](../tests/test_formatter.py), [designations](../tests/test_designations.py) |
| Direct Pint formatter/Babel APIs | Designated inputs reject, including delegate formatters; magnitude formatting and ordinary text still delegate | [formatter](../tests/test_formatter.py) |
| Pickle | All semantic registries can produce pickles; reconstruction uses the selected application registry for the pickle's angle mode, which must have matching definitions | [registry](../tests/test_registry.py), [kinds](../tests/test_kinds.py) |
| Physical tuple / timedelta | Reject metadata the representation cannot preserve; `from_tuple` does not infer semantics | [registry](../tests/test_registry.py) |
| `Measurement` / `plus_minus` | Unclassified inputs only; classified arithmetic/conversion rejects in both operand orders, equality is false; explicitly construct an uncertainty Quantity to use normal propagation | [measurement](../tests/test_measurement.py) |
| `wraps` | Inspect original converted values and declarations before magnitude extraction; probe variadic binding when decorating affected signatures and guard misplaced slots only on defective converters | [adapters](../tests/test_adapters.py) |
| Registry `check` | Physical dimensions and ownership only; preserve inputs. Use fixed signatures: inherited variadic/keyword-only validation remains defective on Pint 0.25.3 | [adapters](../tests/test_adapters.py) |
| `pi_theorem` / Matplotlib | Classified extraction rejects; ordinary physical behavior delegated; plotting import is lazy | [adapters](../tests/test_adapters.py) |

Kind-only text retains Pint's formatting; repr exposes the kind. Fraction-valued
physical unit exponents can fail in Pint 0.25 formatting, including quantity
`repr()`; designated unit `str()`/`format(unit, "~")` use xpint's formatter and
round-trip through `Unit`. Powers parse as floats, as in Pint's own parser, so a
unit built with a non-dyadic `Fraction` power parses back with the float power and
compares unequal to its source. Rational designation signatures do not eliminate
the other upstream formatting limitations. Optional uncertainties,
Babel, plotting and array packages are test/development dependencies, not direct
runtime requirements. Pint is the only direct runtime dependency.

## Pint dependency upgrade checklist

Before extending support, inventory newly inherited APIs and check the private
hooks below for signature and dispatch changes. Then run public regression and
ordinary-Pint parity tests on installed artifacts. Symbol presence alone is not
behavioral evidence.

| Pint boundary | xpint reliance |
| --- | --- |
| `_add_sub`, `_iadd_sub`, `_mul_div`, `_imul_div` | Arithmetic policy, physical operations, dtype and alias behavior |
| `_numpy_method_wrap`, `_dask_finalize` | Array dispatch and metadata restoration |
| `_repr_html_`, `_repr_latex_` | Rich display delegation |
| `_after_init`, `_build_cache` | Registry seeding and angle-definition cache rebuild |
| `_add_unit`, `_switch_context_cache_and_units` | Affine-role cache invalidation on definitions/context changes |
| `pint.compat.is_duck_array_type`, `pint.compat.np` | Optional NumPy and duck-array handling |
| `pint.util.UnitsContainer`, `to_units_container` | Physical units, factor storage and parsing |
| `registry_helpers.wraps/check`, `util.pi_theorem` | Registry adapter behavior and original-value guarding |
| `pint.matplotlib.PintConverter`, formatter delegates and Measurement hooks | Optional extraction boundaries |

This is a bounded inventory. Broader GenericUnitRegistry, custom contexts, and
optional third-party integrations need their own focused checks. Consult the
[upstream adoption tracker](dev/upstream_pint.md) before removing a workaround.
