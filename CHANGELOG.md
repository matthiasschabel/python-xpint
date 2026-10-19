# Changelog

All notable public changes to xpint are documented here.

## [Unreleased]

## [0.1.0b3] - 2026-10-19

### Added

- The default registry defines a `MomentOfInertia` kind (`kilogram * meter ** 2`).
  It existed only in angle-dimension registries (`kilogram * meter ** 2 / radian ** 2`),
  so `as_kind("MomentOfInertia")` raised `KeyError` in default-mode code and a
  pickle carrying the kind could not be restored there.

### Changed

- `set_application_registry()` selects the pickle registry for its registry's
  angle mode only, and `get_application_registry()` takes a keyword
  `angle_dimension` selecting the mode. Without it, the getter returns the
  default-mode registry; it used to return whichever registry was selected last,
  so pass `angle_dimension=True` to get an angle-dimension one. Selecting an angle-dimension registry used
  to make every default-mode pickle fail with `ValueError`, so a process could not
  load pickles of both modes; they now load side by side. An angle-mode pickle with
  no angle registry selected reconstructs in a shared
  `SemanticRegistry(angle_dimension=True)` instead of raising.
- In an angle-dimension registry, conversion to a plain array (`np.asarray()`,
  `np.array()`, `np.asanyarray()`, assignment into an `ndarray`) raises
  `DimensionalityError` for a value carrying an angle, as `float()` already did.
  Pint used to strip the angle with only a `UnitStrippedWarning`. Use
  `m_as("radian")` or multiply by `1 / radian` first. Libraries that coerce
  before converting units, such as Matplotlib's `errorbar`, now raise for such
  values too. Masked `filled()`/`compressed()` and Dask `rechunk()` return an
  angle as a quantity instead of a bare magnitude. Default registries and
  non-angle units are unchanged.

### Fixed

- `get_default_registry()` returns one registry even when several threads make
  the first call at once. It used to build one per racing thread when
  `xpint.units` had not been imported yet, and values made in the extra
  registries could not be combined with later ones. The function no longer
  carries `functools.cache` attributes such as `cache_clear()`, which were never
  documented and would have left `xpint.units` bound to a discarded registry.

## [0.1.0b2] - 2026-10-17

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
