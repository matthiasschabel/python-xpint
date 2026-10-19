# Architecture and design decisions

xpint adds meaning-sensitive checks to Pint without replacing its unit engine.
This document describes the implemented beta design. The [README](../README.md)
is the user contract; [compatibility](compatibility.md) describes integration
boundaries and [future work](dev/future_work.md) records unresolved extensions.

## Objectives and responsibilities

- Detect equal-dimensional mistakes: Energy versus Torque, one substance versus
  another, or an absolute temperature versus a temperature difference.
- Keep ordinary Pint dimensions, definitions, conversion, and array behavior.
- Make classification explicit. A kind is checked only against another kind;
  an unkinded quantity is not absorbed into a classified one.
- Preserve semantic state through supported operations; reject undefined
  propagation before numerical extraction or mutation.
- Keep registry ownership explicit without changing Pint's application registry.
- Keep method names mechanically paired with a MATLAB implementation of the same
  model (`asKind` ↔ `as_kind`). The shared conformance fixture is a contract
  candidate, not yet evidence of cross-language parity.

`SemanticRegistry` subclasses Pint's generic registry and owns dynamically bound
`SemanticQuantity`, `SemanticUnit`, and guarded Measurement classes. Registry
identity matters even if two registries happen to contain identical definitions.

| Module | Responsibility |
| --- | --- |
| [registry.py](../src/xpint/registry.py) | Registry, quantity/unit behavior, result propagation, conversion, array protocols, and pickle reconstruction |
| [designations.py](../src/xpint/designations.py) | Immutable factor expressions and canonical designation signatures |
| [_policies.py](../src/xpint/_policies.py) | Immutable built-in catalogs and semantic dispatch tables |
| [_adapters.py](../src/xpint/_adapters.py) | Guards around Pint decorators and dimensional analysis |
| [_formatter.py](../src/xpint/_formatter.py) | Registry-local guards on direct Pint formatter entry points |
| [_measurement.py](../src/xpint/_measurement.py) | Guards around Pint's separate uncertainty Measurement hierarchy |
| [_matplotlib.py](../src/xpint/_matplotlib.py) | Optional plotting converter with checks before metadata extraction |
| [units.py](../src/xpint/units.py) | Curated and lazy unit symbols from one default registry |

The Pint-bound classes remain together because their runtime references and
Pint's dynamic subclasses make a mechanical split risky. Independent policies and
adapters are already separate. Module size alone is not a reason to change that
boundary.

## Semantic model and validation precedence

Four distinct concerns govern an operation:

1. **Physical dimensions** determine whether the physical operation is valid.
   Addition/subtraction reports physical mismatch before semantic mismatch;
   Pint's logarithmic arithmetic retains its own physical rule.
2. **Designations** partition dimension contributions by referent identity.
3. **Affine role**, for thermodynamic temperature, gates arithmetic before
   stored-kind compatibility.
4. **Kind** constrains the meanings that remain compatible.

Registry ownership is a separate boundary check. Semantic checks precede the
Pint operation that would otherwise lose the distinction. Result construction
then validates or attaches the resulting state. For example, `20 degC + 5 K`
rejects the Generic Kelvin role before stored kinds are checked.

### Quantity kinds

A kind is optional quantity metadata with a registry-validated reference
unit/dimensionality. `as_kind()` classifies without converting units;
`as_unkinded()` removes that classification explicitly. `compatible_kinds()`
returns sorted dimension-matching names, filters intrinsic affine conflicts,
and ignores designations and the quantity's current kind. It is discovery, not
automatic inference.

Addition/subtraction requires equal stored kinds, including both being unkinded,
after dimension, designation, and affine handling. Affine arithmetic can therefore
combine an absolute temperature and a temperature difference regardless of whether
their roles came from units or stored kinds. Other kind mismatches require an
explicit `as_kind()` or `as_unkinded()` call. Equality and ordering require exact
semantic state.

Quantity-valued assignments, fills, searches, and additive-reduction initials also
require matching stored kinds. Kind errors precede mutation. `clip()` alone allows
a missing kind on a bound and preserves the input classification; it still rejects
two different non-null kinds. Bare scalar reduction initials remain magnitudes in
the existing semantic unit.

Multiplication preserves a kind when scaling by a bare number or an unkinded,
physically dimensionless factor without retained designations. Division retains
the numerator's kind for such a denominator. General products, quotients, and
powers drop kinds; they do not infer new ones. This is a defined metadata drop.
A kinded dimensionless exponent rejects rather than silently losing its kind.

Kinds catch an equal-dimension mistake only when both operands are classified:
an unkinded value is never absorbed into a kinded one. Earlier drafts adopted the
only kind present, which made `torque + 10 * J` silently a torque; that one-sided
adoption was removed before the first release rather than kept behind a
`strict_kinds` flag. Most built-in kinds have a unique dimensionality and add
nothing to dimensional checking; the catalogue earns its place in the collisions
(energy/torque, frequency/angular velocity/radioactivity, action/angular momentum,
pressure/energy density, absorbed/equivalent dose, entropy/heat capacity,
kinematic viscosity/diffusivity, and the dimensionless angles). Treat it as a
starter vocabulary, not an ontology.

Kinds answer *what quantity is this?* Designations answer *which referent?*
Frequency and Radioactivity remain different kinds, while `Bq["F18"]` distinguishes
an isotope. Plane angles cannot be recognized from an arc/radius designation:
inverse trigonometry creates the same physical concept without that construction.

### Factor designations

A designation is an opaque, case-sensitive label attached to a factor. xpint
retains an immutable factor expression for display and a canonical signature of
physical dimension, label, and rational exponent for matching/cancellation.
Consequently `mmol["O2"]` and `mol["O2"]` share identity despite differing scales.

Pint's `dimensionless` keeps its physical meaning. `mL["blood"]/mL["tissue"]`
is physically dimensionless but retains a designation signature, so
`is_semantic_unity` is false. Coercion and propagation use the stronger invariant.
An undeclared compound factor cannot guess its designation slot. Registry slots
make `mM["O2"]` target the substance numerator; the unambiguous general spelling
is `mmol["O2"]/L`.

Conversion to a plain unit target preserves source designations, while an
explicitly designated target must match. `as_undesignated()` is the explicit
removal operation. Unit reduction reconciles physical scale with retained factor
identity, so `1 m[a] / (1 cm[b])` becomes `100 m[a] / m[b]`.

The implemented signature labels dimension contributions: `m["cap"]**2` and
`(m**2)["cap"]` have the same signature. Whether a future API must distinguish
whole geometric measures is an [open design question](dev/future_work.md#open-design-questions),
not silently resolved by this consolidation.

### Thermodynamic temperature

Roles are Absolute, Delta, and Generic. Offset scales and their delta partners
follow Pint definitions; Rankine has an explicit absolute/delta policy. Bare
Kelvin and other linear temperature units are Generic unless classified. The role
is therefore encoded two ways: derived from the unit for offset and delta scales,
and carried by the stored kind once a value is in Kelvin, so
`(20 * degC).to(K) + 5 * K` still rejects. A point converted to Kelvin remains a
point. Because the
two built-in kinds encode the role, `define_kind()` rejects other kinds over
temperature dimensions; separating role from kind would be needed to support them.
`Temperature` and `TemperatureDifference` supply explicit Absolute and Delta
roles, including after conversion to Kelvin.

| Operands | Addition | Subtraction |
| --- | --- | --- |
| Absolute, Absolute | reject | Delta in the left scale |
| Absolute, Delta | Absolute | Absolute |
| Delta, Absolute | Absolute | reject |
| Delta, Delta | Delta | Delta |
| Generic, Generic | ordinary physical arithmetic | ordinary physical arithmetic |
| Generic with Absolute/Delta | reject | reject |

Absolute points reject multiplication, division, powers, negation, and absolute
value. Multiplication of an absolute **unit** by a magnitude is quantity
construction, not scaling an existing point. To use ratio-scale algebra, convert
to Kelvin and explicitly remove classification. Strict roles do not generalize
to gauge pressure or other affine domains in this release.

## Optional angle dimensions

`SemanticRegistry(angle_dimension=True)` gives radian dimension `[angle]` and
steradian `[angle]**2`. It is fixed at construction, uses stock Pint definitions,
and does not alter other registries. Definitions and photometric dimensions are
updated before kind registration, followed by Pint's cache rebuild. Temporary
internal redefinitions restore the caller's `on_redefinition` policy afterward.

The angle catalog uses conjugate rotational dimensions:

| Quantity | Reference unit in angle mode |
| --- | --- |
| Torque | joule / radian |
| AngularMomentum | joule * second / radian |
| MomentOfInertia | kilogram * meter**2 / radian**2 |

With `eta = 1 / registry.radian`, arc length is `eta * r * theta`, torque is
`eta * (r × F)`, angular momentum is `eta * (r × p)`, and moment of inertia is
`eta**2 * integral(r**2 dm)`. Porting Cartesian formulas must include these
factors. Dimensional analysis must include eta among its dimensional constants;
omitting it can suppress valid dimensionless groups. Pint constants keep their
ordinary definitions, so inserting angle dimensions into a formula requires an
explicit dimensional check.

An independent solid-angle base dimension was considered. The implemented
`sr = rad**2` choice avoids a second bridging constant; it does not distinguish
solid angle from every other squared-angle quantity. Kinds can distinguish
classified meanings. The default registry retains Pint's dimensionless angles.

Trigonometric ufuncs drop any input kind, reject retained designations, and
delegate domain checks to Pint's dimensionality. In the default registry even a
`SolidAngle`-kinded value can pass because it is physically dimensionless; angle
mode distinguishes it dimensionally. `arctan2` additionally requires exact
semantic state and rejects Absolute inputs. Inverse trigonometry returns unkinded
angular quantities. Automatic classification was rejected because it would impose new coercion restrictions
on ordinary Pint-style calls. Callers can classify explicitly. In angle mode,
`m_as("radian")` extracts an angular magnitude and multiplication by eta produces
a dimensionless quantity; neither operation guesses a unit. See the executable
[angle examples](../examples/angle_dimensions.py).

## Ownership, mutation, and serialization

Quantities carry their own metadata; immutable unit factors and signatures can
be shared. In-place NumPy operations stage values and candidate metadata before
committing, preserving aliases, dtype/casting behavior, and failure state.
Classified multiplication/division also stage automatic reduction with the
resulting semantics. This costs a full-size temporary buffer, sometimes a second
for reduction; it is not a constant-memory or performance-neutral guarantee.
Other duck-array backends retain Pint's mutation behavior.

Pickles store values, units, classification, factors, and angle mode, not whole
registry definitions. Values from any semantic registry can be serialized;
reconstruction uses `xpint.get_application_registry(angle_dimension=...)` for the
pickle's recorded angle mode. Each mode has its own slot, because a process may
hold both kinds of pickle; one slot made selecting an angle registry strand every
default-mode pickle. The caller selects compatible definitions with
`xpint.set_application_registry()`, which fills only its registry's mode. The
defaults are xpint's shared symbol registry and a lazily built shared angle
registry, each constructed once even under concurrent first use. Neither function
changes Pint's application registry. Pickle is not a versioned cross-language
interchange schema.

`.magnitude`, `m_as()` after explicit conversion, and physical registry conversion
are deliberate extraction boundaries. `m_as()` is the checked one: a designated value
needs a target naming the same designations, because a plain target would otherwise
adopt them and return a number nobody checked. Plain tuple and timedelta adapters reject
semantic state they cannot encode. Scalar hashing includes semantic state and
registry ownership; arrays remain unhashable. Mixed comparison with ordinary
Pint has an [inherited asymmetry](compatibility.md#comparison-and-extraction).

## Alternatives retained as rationale

| Alternative | Reason it was not selected |
| --- | --- |
| Fork Pint or build another unit engine | Duplicates definitions, conversion, and scientific interoperability |
| Composition around every Pint value | Avoids inheritance leakage but requires rebuilding operator/protocol integration; revisit only with evidence that guards cannot suffice |
| Kinds as dimensions or contexts | Confuses physical compatibility with optional persistent meaning |
| Auxiliary designation dimensions | Pollutes physical dimensionality, leaks generated unit names, and couples labels to registry mutation |
| Display-only labels or object-identity side tables | Cannot reliably enforce cancellation or survive copying/reconstruction |
| Infer named kinds from products | Ambiguous relations; the beta deliberately drops kind rather than inventing meaning |
| Mutate then roll back, or replace in-place arithmetic with out-of-place arithmetic | Can expose partial writes or change casting and alias contracts |
| Global registry replacement | Changes application behavior merely by importing a library |
| One-sided kind adoption, or a `strict_kinds` flag defaulting to it | Adoption fails the safety goal on the common path of classified code meeting unkinded values; a flag would freeze two behaviors for one rule |
| xpint as another Pint facet | Ties the package to Pint's internal facet order and every registry variant Pint builds |

New interfaces should begin with a concrete failing public example, a propagation
contract, and a focused test. Unsupported behavior and adoption prerequisites
belong in [future work](dev/future_work.md), not in completed implementation plans.
