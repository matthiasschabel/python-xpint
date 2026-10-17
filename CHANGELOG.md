# Changelog

All notable public changes to xpint are documented here.

## [Unreleased]

### Changed

- `m_as()` on a designated value requires a target carrying the same
  designations. A plain target such as `"mol/L"` or `1 / s` used to adopt them
  and return the bare number unchecked, so `m_as()` could read a value whose
  designations had not cancelled (a gadoteridol relaxivity times a gadobutrol
  concentration) as an ordinary rate. It now raises `TypeError`. Pass a
  designated unit, such as `mol["O2"] / L`, or call `as_undesignated()` first.
  `to()`, `ito()` and `.magnitude` are unchanged.

## [0.1.0b1] - 2026-10-15

First public beta. No upgrade from a previous public version is required.

### Added

- Pint-backed semantic quantities with dimension-constrained kinds, explicit
  classification/removal methods, and `compatible_kinds()` on quantities, units,
  and the registry to discover kinds sharing a dimensionality.
- Factor-level, case-sensitive designations for substances and compartments,
  including concentration aliases and designation-preserving conversions.
- Strict Absolute, Delta, and Generic temperature roles, including Celsius
  construction and affine arithmetic across temperature scales.
  Absolute temperature units construct quantities only from magnitudes; multiplying
  one by a quantity or unit raises, including through NumPy.
- Offset-aware equality: absolute temperatures in different scales are compared
  after conversion, so `0 * degC == 0 * degF` is false, unlike in ordinary Pint,
  and equality agrees with hashing.
- `define_kind()` rejects new kinds over thermodynamic-temperature dimensions;
  the built-in `Temperature` and `TemperatureDifference` kinds carry affine roles,
  and conversion would otherwise replace a custom kind silently.
- Strict stored-kind matching: addition, subtraction, assignments, fills,
  searches, and additive-reduction quantity initials reject differing kinds,
  including a missing kind. Use `as_kind()` or `as_unkinded()` to align them.
  `clip()` permits an unkinded bound.
- Optional per-registry angle dimensions with corresponding rotational kinds.
- A documented subset of NumPy, masked-array, Dask, xarray, and pint-xarray
  behavior that preserves semantic state or explicitly rejects undefined operations.
- Registry ownership checks raise `ValueError` for values from another registry,
  including ordinary Pint quantities and a pickle whose `angle_dimension` differs
  from the application registry; unpickling an undefined kind raises a `KeyError`
  naming `set_application_registry()`.
- Registry ownership checks, scoped application-registry pickle reconstruction,
  and guarded Pint decorators, plotting, and measurement adapters. `wraps`
  preserves classified values under `None` variadic declarations when Pint binds
  them correctly, and keeps a protective guard on converters that do not.
- `define_kind()` and `define_designation_slot()` to extend a registry's kind
  catalogue and designation syntax beyond the built-in definitions.
- An `xpint.units` module exporting curated unit symbols and resolving any other
  Pint unit name or prefix through the shared registry on attribute access.
- Typed APIs, runnable examples, and the BSD 3-Clause license.

### Compatibility and limitations

- Python 3.12–3.14 and Pint 0.25.3 are tested. The dependency range remains
  `pint>=0.25.3,<0.26`; Pint 0.26 and later are not supported yet.
- Scalar use requires only Pint and its dependencies. Arrays and ecosystem
  adapters need their corresponding optional packages; NumPy 2.5.2 is tested.
- Registry `check` and `wraps` require fixed signatures. Inherited variadic and
  keyword-only validation remains defective on Pint 0.25.3, so xpint guards those
  declarations rather than trusting them.
- Measurement adapters reject kinds, designations, and affine roles. To use an
  ordinary measurement in semantic arithmetic, explicitly construct an xpint
  Quantity from its magnitude and units first.
- Designated values support xpint `str()` and the documented `format()` specs.
  Direct Pint/Babel formatter APIs reject them rather than omit their labels.
- Native designated-string parsing, pandas, arbitrary Dask graph transformations,
  and classified Measurement objects remain unsupported.
- The API may change during beta. See the
  [README](https://github.com/matthiasschabel/python-xpint/blob/main/README.md)
  for installation and detailed behavior.
