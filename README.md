# xpint

xpint extends [Pint](https://pint.readthedocs.io/) with quantity kinds and unit
designations: semantic information that keeps energy distinct from torque and
mL[blood]/mL[tissue] distinct from a plain ratio.

> **Status:** 0.1.0b2 is a public beta. The API and the supported
> interoperability surface may change during beta.

Pint checks physical dimensions. Many mistakes pass that check: energy and
torque are both newton-metres, an oxygen concentration and a carbon dioxide
concentration are both mol/L, and 20 °C plus 20 °C is not 40 °C. xpint adds
three kinds of meaning on top of Pint and checks them before Pint can erase the
distinction:

- **Kinds** say what a quantity is (`Energy`, `Torque`, `Frequency`).
- **Designations** say which referent a unit factor belongs to (`mmol["O2"]`,
  `mL["blood"]`).
- **Temperature roles** separate an absolute temperature (a point on a scale)
  from a temperature difference.

Pint remains the unit engine: definitions, conversions, contexts, and NumPy
support are Pint's own. xpint values are Pint quantities, and importing xpint
does not change Pint's application registry.

## Installation

```bash
python -m pip install "xpint==0.1.0b2"
```

xpint requires Python 3.12 or later and Pint `>=0.25.3,<0.26`. The Pint range is
deliberately narrow because xpint hooks into Pint internals; Pint 0.26 is not
supported yet. Scalars need only Pint. Install NumPy for arrays (2.5.2 is
tested). Matplotlib, Dask, xarray, pint-xarray, Babel, and uncertainties are
optional and are not installed by xpint.

## Quick start

Unit symbols come from `xpint.units`, which binds them to one shared registry.
The module exports common symbols and resolves any other Pint unit name or
prefix on import, for example `from xpint.units import kg, microgram`.

```python
from xpint.units import J, L, N, degC, degF, delta_degC, m, mL, mmol

# Kinds: same dimensions, different meaning
energy = (10 * J).as_kind("Energy")
torque = (10 * N * m).as_kind("Torque")
energy + torque  # TypeError: cannot add quantity kinds 'Energy' and 'Torque'
energy.to("kWh")  # the kind survives conversion
(N * m).compatible_kinds()  # ('Energy', 'Torque')

# Designations: same units, different referent
oxygen = 5 * mmol["O2"] / L
oxygen + 5 * mmol["CO2"] / L  # TypeError: different factor designations
oxygen.to("mol/L")  # 0.005 mole[O2] / liter

ratio = 0.6 * mL["blood"] / mL["tissue"]
ratio.dimensionless  # True: physically a ratio
ratio.is_semantic_unity  # False: the designations do not cancel
float(ratio)  # TypeError: remove the designations explicitly first

# Temperature roles: points and differences
room = 20 * degC
room + 5 * delta_degC  # 25 degC
room + room  # TypeError: cannot add two absolute temperatures
25 * degC - 68 * degF  # about 5 delta_degC
```

## Quantity kinds

A kind is optional metadata naming what a quantity is. It is checked against the
quantity's dimensions when attached, and only against other kinds afterwards:

- `as_kind(name)` classifies a value without converting it; `as_unkinded()`
  removes the classification. Both return new quantities. You can also pass
  `kind=` to `ureg.Quantity(...)`.
- Addition, subtraction, assignment into arrays, and fills require equal kinds,
  including both being unkinded. A kinded value plus an unkinded one is rejected
  rather than silently adopting the kind; align them with `as_kind()` or
  `as_unkinded()`.
- Equality and ordering require exactly the same semantic state, so a kinded
  quantity is unequal to an otherwise identical unkinded one.
- Multiplying or dividing by a plain number or a plain dimensionless value keeps
  the kind. Other products, quotients, and powers drop it: xpint does not guess
  that force times length is torque. `energy / (2 * s)` is an unkinded
  `J / s`.
- Conversion keeps the kind. A context conversion that would change a kinded
  quantity's dimensions, such as wavelength to energy, requires `as_unkinded()`
  first.
- `compatible_kinds()` on a unit, quantity, or registry lists the registered
  kinds that match the dimensions. It is a lookup, not inference.

The built-in catalogue covers the SI base and derived quantities and the common
same-dimension collisions (energy/torque, frequency/angular
velocity/radioactivity, absorbed/equivalent dose, entropy/heat capacity, and
others). See `ureg.kinds` for the full list.

## Designations

A designation is an opaque, case-sensitive label attached to one factor of a
unit, such as the substance in `mmol["O2"] / L`:

- Matching designations cancel; different ones stay in the quantity's
  designation signature even when the physical units cancel, as in
  `mL["blood"] / mL["tissue"]`. `dimensionless` keeps Pint's physical meaning;
  `is_semantic_unity` is true only when a value is dimensionless *and* has no
  designation left. It does not look at kinds.
- Addition and ordering require matching designations, and equality with a
  mismatch is false. Scalar conversion (`float()`, `int()`) and undefined NumPy
  operations reject any retained designation.
- Converting to a plain unit keeps the source's designations:
  `oxygen.to("mol/L")` is `0.005 mole[O2] / liter`. A designated target must
  match exactly, so `oxygen.to(mol["CO2"] / L)` is rejected. The same rules apply
  to `ito()`, compact and preferred conversions, construction from another
  quantity, and registry `convert()`.
- `m_as()` returns a bare number, so it does not adopt designations: the target
  must name them. `oxygen.m_as(mol["O2"] / L)` is `0.005`, while
  `oxygen.m_as("mol/L")` raises `TypeError`. Mismatches that survive physical
  cancellation are caught too: a gadoteridol relaxivity times a gadobutrol
  concentration cannot be read with `.m_as(1 / s)`. `.magnitude` remains Pint's
  unchecked accessor; use `as_undesignated()` to drop designations deliberately.
- `with_designation(label)` labels a value whose unit is a single factor, such as
  `(5 * mmol).with_designation("O2")`; `as_undesignated()` removes every label.
- Unit reduction keeps designations and physical scale:
  `1 m["a"] / (1 cm["b"])` reduces to `100 m[a] / m[b]`.
- Compound units do not guess which factor a label belongs to. `mM["O2"]` works
  because the registry declares that molar concentration designates its
  substance; otherwise write the factor form, `mmol["O2"] / L`.

Designations are entered programmatically; there is no parser for strings such
as `"mmol[O2]/L"` yet.

## Temperature roles

Thermodynamic temperatures have one of three roles:

| Role | Comes from | Meaning |
| --- | --- | --- |
| Absolute | offset units such as `degC`, `degF`, and Réaumur; Rankine; or kind `Temperature` | a point on a scale |
| Delta | delta units (`delta_degC`, ...), or kind `TemperatureDifference` | a difference |
| Generic | plain Kelvin and other linear units, unclassified | ordinary Pint arithmetic |

| Operands | Addition | Subtraction |
| --- | --- | --- |
| Absolute, Absolute | rejected | Delta, in the left operand's scale |
| Absolute, Delta | Absolute | Absolute |
| Delta, Absolute | Absolute | rejected |
| Delta, Delta | Delta | Delta |
| Generic, Generic | Pint's rules | Pint's rules |
| Generic with Absolute or Delta | rejected | rejected |

- The role survives conversion: `(20 * degC).to(K)` is an absolute temperature
  of kind `Temperature`, so adding Generic `5 * K` to it is rejected.
- Absolute temperatures reject multiplication, division, powers, negation, and
  `abs()`. To use ratio-scale algebra, convert to Kelvin and remove the role:
  `(20 * degC).to(K).as_unkinded() ** 4`.
- `20 * degC` and `ureg.Quantity(20, "degC")` construct an absolute temperature.
  String parsing such as `ureg("20 degC")` multiplies by the unit and is rejected
  with a pointer to these spellings; `"5 delta_degC"` and `"300 kelvin"` parse
  normally.
- Equality between offset scales converts first. Unlike ordinary Pint,
  `0 * degC == 0 * degF` is false.
- Custom kinds cannot be defined over temperature dimensions, because the
  `Temperature` and `TemperatureDifference` kinds carry the role.

The strict roles apply to thermodynamic temperature only, not to gauge pressure
or other offset scales.

## Optional angle dimensions

By default plane angle is dimensionless, as in SI and Pint. A registry can
instead make it a base dimension, with solid angle derived as `rad ** 2`:

```python
from xpint import SemanticRegistry

u = SemanticRegistry(angle_dimension=True)

u.Quantity(1, "radian") + u.Quantity(1, "steradian")  # DimensionalityError
u.get_dimensionality("radian / second") == u.get_dimensionality("hertz")  # False
((1 * u.mrad) * (1 * u.mrad)).to("microsteradian")  # 1.0 microsteradian
torque = 5 * u.J / u.rad  # not interchangeable with energy
```

Rotational quantities then take their conjugate forms (`joule / radian` for
torque), and leaving the angle world is explicit: extract a magnitude with
`theta.m_as("radian")`, or multiply by `eta = 1 / u.radian`. Pint's constants
carry no angle dimension, so formulas such as `hbar * omega` need one `eta`. The
mode is fixed when the registry is created, and values from angle and ordinary
registries cannot be mixed. See
[examples/angle_dimensions.py](https://github.com/matthiasschabel/python-xpint/blob/main/examples/angle_dimensions.py).

In either mode, `float()`, `int()`, and functions that call them, such as
`math.sin`, reject a kinded value even when it is dimensionless, because the
conversion would drop the kind. Use `math.sin(angle.m_as("radian"))`; NumPy
trigonometric functions accept such angles directly and return unkinded results.

## Extending a registry

The built-in catalogue is a starting point. A registry can declare its own kinds
and designation slots:

```python
from xpint import SemanticRegistry

u = SemanticRegistry()

u.define_kind("SpecificActivity", "becquerel / gram")
activity = u.Quantity(4.2, "becquerel / gram", kind="SpecificActivity")

u.Quantity(1.0, "becquerel").as_kind("SpecificActivity")
# ValueError: kind 'SpecificActivity' requires dimensions 1 / [time] / [mass], ...

u.define_designation_slot("katal", "[substance]")
3 * u.katal["trypsin"]  # 3 katal[trypsin]
```

`define_kind()` constrains a name to the dimensions of a reference unit.
`define_designation_slot()` names the single factor that a label designates on a
one-token derived unit. Both affect only that registry. See
[examples/custom_semantics.py](https://github.com/matthiasschabel/python-xpint/blob/main/examples/custom_semantics.py).

## Working with Pint, NumPy, and other libraries

### Registries

Every xpint value belongs to one `SemanticRegistry`. Combining values from two
registries, or an xpint value with an ordinary Pint quantity, raises
`ValueError`; equality between two xpint registries is simply false. One caveat comes from Pint: `pint_quantity == xpint_quantity`
(ordinary Pint on the left) compares physically and can return true.

### What behaves like ordinary Pint

Values without kinds, designations, or a temperature role go through Pint's own
code paths, with two guards added: registry ownership checks, and rejection of
Pint 0.25.3's unit-blind NumPy `out=` and quantity-valued `initial=` handling
(including `ndarray += quantity`, which would strip the unit). A bare scalar
`initial=` is a magnitude in the quantity's own unit.

Celsius, Fahrenheit, and Rankine values are classified by their unit, so they
follow the semantic rules below rather than Pint's full NumPy surface. Functions
outside the list, such as `np.sort`, `np.median`, `np.percentile`, `np.isclose`,
and `np.interp`, raise for them. Convert explicitly with
`temperatures.to("kelvin").as_unkinded()` to use them as ordinary linear values,
or operate on `.magnitude`.

### NumPy arrays

For classified arrays the supported surface is deliberately bounded, and
anything outside it raises rather than silently dropping metadata:

- Arithmetic and comparisons follow the kind, designation, and temperature rules
  above.
- `mean`/`nanmean`, `max`/`min`/`amax`/`amin`, shape-only methods, `zeros_like`,
  `concatenate`, `stack`, and `where` preserve metadata, including absolute
  temperatures. Combined operands must have matching semantic state.
  `nanmax`/`nanmin` are not supported yet.
- `sum`/`nansum` and `cumsum`/`nancumsum` preserve kinds and designations and
  reject absolute temperatures, which cannot be added together.
- `sqrt`, `square`, `cbrt`, and `reciprocal` follow power rules: kinds are
  dropped and designation exponents scale.
- `std`/`nanstd` preserve units, kinds, and designations; `var`/`nanvar` square
  the units and drop the kind. Both reject absolute temperatures, and these and
  the cumulative sums reject output targets.
- `diff` preserves metadata, except that absolute temperatures produce
  temperature differences; `diff(n=0)` returns the input unchanged.
- `argmax`, `argmin`, `argsort`, and `nonzero` return plain indices.
- `clip()` keeps the input's classification and allows an unkinded bound.
- Reductions validate `out` and `initial` whether passed by position or keyword.
  A quantity `initial` must match the input's semantic state.
- Undefined operations, discrete arithmetic such as modulo, `dot`, `prod`, and
  implicit conversion to `ndarray` (including `np.asarray`) raise. Use
  `.magnitude` when you intend to leave xpint's checks.

In-place powers, unit conversions (`ito()`), temperature-role addition and
subtraction, and classified in-place multiplication and division stage a
temporary NumPy copy, so a failure in those operations leaves values and metadata
unchanged. This costs one extra full-size buffer, sometimes two. Other in-place
operations, including ordinary `+=` on a kinded array, write through Pint
directly, as in Pint, and a floating-point error raised midway can leave the
array partly updated. The staging covers NumPy arrays only. Classified quantities expose array metadata such as `shape`,
`dtype`, and `mask`, but forwarded magnitude methods such as `view()` raise.

### Other array libraries

- Masked arrays: `filled()` and `compressed()` preserve classified values.
- Dask: `compute()`, `persist()`, and `rechunk()` preserve metadata; classified
  `map_blocks()` raises because the callback could change values or dimensions.
- xarray: reductions, concatenation, `where`, transpose and indexing, Dask-backed
  computation, and multiplication by xpint units are tested.
- pint-xarray: ordinary units work with `SemanticRegistry(force_ndarray_like=True)`;
  designated round trips need a designation parser and are unsupported.
- pandas and arbitrary Dask graph transformations are not supported.

### Comparison and hashing

Scalar quantities can be dictionary keys. Hashes include the registry and the
semantic state, so equal values hash equally; arrays are unhashable.
`is_compatible_with()` has two meanings. With a unit or unit string it asks
whether converting to that unit would be accepted, so
`(20 * degC).is_compatible_with(K)` is true while `K.is_compatible_with(degC)` is
false. With a quantity it also requires the same kind, designations, and
temperature role, so an `Energy` value is not compatible with a `Torque` value
even though it converts to its units. A quantity used as a conversion target
supplies units and designations only; it never reclassifies the value.

### Formatting

`str()` and `format()` include designations and accept numeric specs such as
`f"{oxygen:.2f}"` and the abbreviated unit spec `~`; other unit specs (`P`, `L`,
`H`, `C`, `D`) are rejected for designated values. `repr()` shows the kind,
`kind='Energy'`, which `str()` omits. Pint's formatter and Babel APIs reject
designated values rather than drop their labels.

### Pickling

Quantities and units from any registry can be pickled. Unpickling reconstructs
against `xpint.get_application_registry()`, by default the registry behind
`xpint.units`. Registry definitions are not pickled: to unpickle values that use
custom units or kinds, or an angle-dimension registry, call
`xpint.set_application_registry(registry)` first, also in worker processes such
as multiprocessing or Dask distributed. A missing kind or mismatched angle mode
raises an error naming that function. Neither function changes Pint's
application registry.

`to_tuple()` and `to_timedelta()` reject kinds and designations, which those
representations cannot hold; remove them explicitly first.

### Registry adapters

- `ureg.check("[length]")` checks physical dimensions only and passes the
  original quantity through with its metadata.
- `ureg.wraps(...)` converts arguments to magnitudes for a numerical function.
  Converted arguments must be unclassified, including the implicit roles of
  Celsius and delta units; return units may construct those roles but cannot
  carry designations. With a fixed parameter list, a `None` argument
  specification passes the value through unchanged, metadata included.
- Use fixed parameter lists with both. Pint 0.25.3 binds variadic and
  keyword-only parameters incorrectly
  ([pint#2254](https://github.com/hgrecco/pint/issues/2254),
  [pint#2426](https://github.com/hgrecco/pint/pull/2426)). For `wraps`, xpint
  rejects classified values in slots Pint might convert by mistake, which can
  also reject a `None`-specified variadic argument; `check` inherits Pint's
  behavior unchanged.
- `ureg.pi_theorem({...})` and `ureg.setup_matplotlib()` accept unclassified
  values. For temperature data, convert to Kelvin and call `as_unkinded()` before
  plotting.
- Measurements (`plus_minus()`, `ureg.Measurement()`, which need the
  `uncertainties` package) accept only unclassified values. To combine
  uncertainty with semantic checks, construct
  `ureg.Quantity(measurement.magnitude, measurement.units)`.

## Examples

Runnable scripts that start from Pint's tutorial style and add xpint's checks:

| Example | Shows |
| --- | --- |
| [pint_style_basics.py](https://github.com/matthiasschabel/python-xpint/blob/main/examples/pint_style_basics.py) | Construction, arithmetic, and conversion with imported symbols |
| [designated_solutions.py](https://github.com/matthiasschabel/python-xpint/blob/main/examples/designated_solutions.py) | Concentrations, analyte mismatches, and compartment ratios |
| [affine_temperature.py](https://github.com/matthiasschabel/python-xpint/blob/main/examples/affine_temperature.py) | Absolute, Delta, and Generic temperature roles |
| [semantic_arrays.py](https://github.com/matthiasschabel/python-xpint/blob/main/examples/semantic_arrays.py) | Metadata through NumPy and where it stops |
| [angle_dimensions.py](https://github.com/matthiasschabel/python-xpint/blob/main/examples/angle_dimensions.py) | The two angle modes side by side |
| [custom_semantics.py](https://github.com/matthiasschabel/python-xpint/blob/main/examples/custom_semantics.py) | Custom kinds and designation slots |

## Reporting issues

Use the [issue tracker](https://github.com/matthiasschabel/python-xpint/issues)
and include a minimal reproducer, your Python, Pint, and NumPy versions, and the
expected and actual behavior.

## Development

From a source checkout:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests examples benchmarks tools/check_minimal_install.py
uv run pytest
```

The full suite needs the development dependency group, which installs every
optional dependency the tests use. CI tests the installed wheel on Python 3.12–3.14 and a
separate install without optional dependencies. Maintainer documentation is in
[docs/](https://github.com/matthiasschabel/python-xpint/blob/main/docs/README.md).

## License

[BSD 3-Clause](https://github.com/matthiasschabel/python-xpint/blob/main/LICENSE),
matching Pint's license. Copyright (c) 2026 Matthias C. Schabel.
