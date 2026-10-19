"""Semantic quantities and their owning Pint registry.

Pickles reconstruct against xpint's application registry, which defaults to the
shared registry. This setting never changes Pint's application registry. The
application registry must provide the serialized units and kinds and use the
same angle-dimension mode as the source registry.
"""

from __future__ import annotations

import copy
import datetime
import html
import operator
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from functools import cache, partial
from inspect import Signature, signature
from types import MappingProxyType
from typing import Any, TypeAlias, cast, override

import pint
from pint.compat import eq, is_duck_array_type, np  # type: ignore[attr-defined]
from pint.delegates.formatter._spec_helpers import split_format
from pint.errors import DimensionalityError
from pint.facets.measurement import Measurement
from pint.facets.nonmultiplicative.definitions import OffsetConverter
from pint.facets.plain.definitions import UnitDefinition
from pint.util import UnitsContainer, to_units_container

from ._formatter import _SemanticFormatter
from ._measurement import _SemanticMeasurement
from ._policies import (
    _ANGLE_DEFINITIONS,
    _ANGLE_KIND_OVERRIDES,
    _ANGLE_UFUNCS,
    _ANGULAR_UNITS,
    _DEFAULT_KINDS,
    _NUMPY_FUNCTION_ADDITIVE,
    _NUMPY_FUNCTION_COMPOSITION,
    _NUMPY_FUNCTION_LIKE_CONSTRUCTORS,
    _NUMPY_FUNCTION_REDUCTIONS,
    _NUMPY_MAGNITUDE_INDICES,
    _NUMPY_METHOD_ADDITIVE,
    _NUMPY_METHOD_REDUCTIONS,
    _NUMPY_METHOD_SUPPORTED,
    _NUMPY_STATISTICS,
    _NUMPY_UFUNC_BINARY_METHODS,
    _NUMPY_UFUNC_MAGNITUDE_PREDICATES,
    _NUMPY_UFUNC_POWERS,
    _NUMPY_VARIANCES,
)
from .designations import (
    DesignationSignature,
    DesignationTerm,
    _as_fraction,
    _SemanticFactors,
    canonical_factors,
    combine_factors,
    designated_factor,
    designation_signature,
    factors_from_units,
    format_factors,
    power_factors,
    undesignated_factors,
)


class AffineRole(StrEnum):
    """Semantic role of a thermodynamic-temperature quantity."""

    ABSOLUTE = "Absolute"
    DELTA = "Delta"
    GENERIC = "Generic"


_ABSOLUTE_LINEAR_ESCAPE = (
    "when the quantity is in kelvin, call as_unkinded() first to opt into linear scaling"
)
_MAGNITUDE_METADATA = frozenset(
    {
        "shape",
        "dtype",
        "ndim",
        "size",
        "nbytes",
        "itemsize",
        "strides",
        "flags",
        "mask",
        "recordmask",
        "fill_value",
        "hardmask",
        "sharedmask",
        "chunks",
        "chunksize",
        "numblocks",
        "name",
        "device",
    }
)
_AFFINE_KINDS = frozenset({"Temperature", "TemperatureDifference"})
_UNCLASSIFIED_STATE: tuple[DesignationSignature, str | None, AffineRole | None] = (
    (),
    None,
    None,
)


@cache
def _reduction_signature(name: str) -> Signature:
    return signature(getattr(np, name))


@dataclass(frozen=True, slots=True)
class KindDefinition:
    """A registered quantity kind constrained to one dimensionality.

    Attributes:
        name: Stable semantic identifier.
        reference_unit: Unit expression defining the required dimensionality.
    """

    name: str
    reference_unit: str


class SemanticQuantity(pint.UnitRegistry.Quantity[Any]):
    """A Pint quantity carrying xpint semantic metadata.

    Kinds distinguish meanings that share physical dimensions, factor designations
    preserve entity or compartment identity, and affine roles separate absolute
    thermodynamic temperatures from differences and generic Kelvin values. Magnitudes
    retain Pint's scalar or array shape and dtype behavior. Operations preserve, drop,
    or reject metadata according to explicit semantic rules before delegating physical
    conversion and arithmetic to Pint.

    Constructing from an existing quantity with target units applies the same
    semantic conversion as ``to()``, preserving the converted kind and affine role.
    Plain targets adopt the source's designations; designated targets must match
    exactly. Use ``with_designation()`` or ``as_undesignated()`` to change labels.
    """

    _kind: str | None
    _semantic_factors: _SemanticFactors

    def __new__(
        cls,
        value: Any,
        units: Any = None,
        *,
        kind: str | None = None,
    ) -> SemanticQuantity:
        registry = cast(SemanticRegistry, cls._REGISTRY)
        registry._require_owned(value, name="value")
        registry._require_owned(units, name="units")

        source = value
        if isinstance(value, SemanticQuantity) and units is not None:
            # Pint normalizes a constructor's target to physical units before
            # conversion, so convert while its semantic factors are still available.
            source = value.to(units)
            value = source.magnitude
            units = source.units
        instance = cast(SemanticQuantity, super().__new__(cls, value, units))
        inherited_kind = getattr(source, "_kind", None)
        instance._kind = inherited_kind if kind is None else kind
        inherited_factors = getattr(source, "_semantic_factors", None)
        unit_factors = getattr(units, "_semantic_factors", None)
        if unit_factors is not None:
            instance._semantic_factors = unit_factors
        elif units is None and inherited_factors is not None:
            instance._semantic_factors = inherited_factors
        else:
            instance._semantic_factors = registry._factors_for_units(instance._units)
        if instance._kind is not None:
            registry._validate_kind(instance._kind, instance.dimensionality)
        registry._validate_affine_kind(instance._units, instance._kind)
        return instance

    @classmethod
    def _from_parts(
        cls,
        value: Any,
        units: Any = None,
        *,
        kind: str | None = None,
        semantic_factors: _SemanticFactors | None,
    ) -> SemanticQuantity:
        # Internal constructor: replaces the factors derived from units after the
        # public constructor validates kind and affine role, which do not depend
        # on designations. Keeps the factor expression out of the public signature.
        instance = cls(value, units, kind=kind)
        if semantic_factors is not None:
            instance._semantic_factors = semantic_factors
        return instance

    @classmethod
    @override
    def from_sequence(cls, seq: Sequence[Any], units: Any = None) -> SemanticQuantity:
        """Combine scalar quantities with matching semantic state into an array.

        Pint converts each element to ``units`` (the first element's units by
        default) and produces a one-dimensional float64 array of length ``len(seq)``.
        Kinds, designations, and affine roles must match across all elements;
        the result retains the common state, including implicit temperature kinds.
        Target units must satisfy each element's ``to()`` conversion constraints,
        including adoption by plain targets and compatible affine roles.

        Raises:
            TypeError: If elements are not quantities or semantic states differ.
            ValueError: If a quantity or target belongs to another registry, or
                the sequence is empty and no target units are supplied.
            DimensionalityError: If element units cannot convert to the target.
        """
        registry = cast(SemanticRegistry, cls._REGISTRY)
        registry._require_owned(units, name="units")
        for quantity in seq:
            registry._require_owned(quantity, name="sequence element")
            if not isinstance(quantity, SemanticQuantity):
                raise TypeError(
                    "sequence elements must be quantities of the registry's Quantity class"
                )
        if len(seq):
            units = seq[0]._conversion_target(seq[0].units if units is None else units)
        result = cast(SemanticQuantity, super().from_sequence(seq, units))
        if len(seq):
            for quantity in seq[1:]:
                seq[0]._require_same_semantic_state(quantity, operation="combine in a sequence")
            result._kind = seq[0]._converted_kind()
        return result

    @property
    @override
    def units(self) -> SemanticUnit:
        """Return units carrying the quantity's semantic factor expression."""

        registry = cast(SemanticRegistry, self._REGISTRY)
        return registry.Unit._from_parts(self._units, semantic_factors=self._semantic_factors)

    @property
    @override
    def u(self) -> SemanticUnit:
        """Return the semantic units, matching Pint's concise unit alias."""

        return self.units

    @property
    def kind(self) -> str | None:
        """Return the semantic kind, or ``None`` when unkinded."""

        return self._kind

    @property
    def affine_role(self) -> AffineRole | None:
        """Return the thermodynamic-temperature role, if applicable."""

        registry = cast(SemanticRegistry, self._REGISTRY)
        return registry._affine_role_from_container(self._units, self._kind)

    @property
    def designation_signature(self) -> DesignationSignature:
        """Return the immutable canonical factor-designation signature."""

        return designation_signature(self._semantic_factors)

    @property
    def has_designation(self) -> bool:
        """Return whether a designation contribution remains after cancellation."""

        return bool(self.designation_signature)

    @property
    def is_semantic_unity(self) -> bool:
        """Return whether physical dimensions and designation residue are both absent.

        Scale-only dimensionless units remain semantic unity. This property asks
        whether the value is safe to treat as an undesignated scalar, not whether
        its unit scale is exactly one.
        """

        return self.dimensionless and not self.has_designation

    def has_kind(self) -> bool:
        """Return whether this quantity has a semantic kind."""

        return self._kind is not None

    def _has_designated_factors(self) -> bool:
        # Avoid generator setup in every classified arithmetic check.
        for factor in self._semantic_factors:
            if factor.designation is not None:
                return bool(self.designation_signature)
        return False

    def _has_affine_semantics(self) -> bool:
        registry = cast(SemanticRegistry, self._REGISTRY)
        return (
            self._kind in _AFFINE_KINDS or registry._intrinsic_affine_role(self._units) is not None
        )

    def _has_absolute_temperature_semantics(self) -> bool:
        registry = cast(SemanticRegistry, self._REGISTRY)
        return (
            self._kind == "Temperature"
            or registry._intrinsic_affine_role(self._units) is AffineRole.ABSOLUTE
        )

    def _is_semantically_classified(self) -> bool:
        registry = cast(SemanticRegistry, self._REGISTRY)
        return (
            self._kind is not None
            or self._has_designated_factors()
            or registry._intrinsic_affine_role(self._units) is not None
        )

    def _semantic_state(self) -> tuple[DesignationSignature, str | None, AffineRole | None]:
        return self.designation_signature, self._converted_kind(), self.affine_role

    def _quantity_operands_have_semantics(self, other: SemanticQuantity | None) -> bool:
        return self._is_semantically_classified() or (
            other is not None and other._is_semantically_classified()
        )

    @staticmethod
    def _semantic_values(value: Any) -> Iterator[SemanticQuantity | SemanticUnit]:
        if isinstance(value, (SemanticQuantity, SemanticUnit)):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from SemanticQuantity._semantic_values(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from SemanticQuantity._semantic_values(item)

    @classmethod
    def _operands_have_semantics(cls, *values: Any) -> bool:
        for value in values:
            for semantic_value in cls._semantic_values(value):
                if isinstance(semantic_value, SemanticQuantity):
                    if semantic_value._is_semantically_classified():
                        return True
                else:
                    registry = cast(SemanticRegistry, semantic_value._REGISTRY)
                    if (
                        semantic_value.has_designation
                        or registry._is_absolute_unit(semantic_value)
                        or registry._is_delta_unit(semantic_value)
                    ):
                        return True
        return False

    def compatible_kinds(self) -> tuple[str, ...]:
        """Return registered kinds compatible with this quantity's units.

        The current kind and factor designations do not filter the result: a
        Torque-kind newton-meter still lists both Energy and Torque in the default
        registry. Affine filtering follows the unit, so generic kelvin lists both
        temperature kinds. The magnitude, shape, dtype, and metadata are unchanged.

        Returns:
            Kind names sorted alphabetically, or an empty tuple if none match.
        """

        return cast(SemanticRegistry, self._REGISTRY).compatible_kinds(self)

    def as_kind(self, name: str) -> SemanticQuantity:
        """Return a copy explicitly asserted to have ``name`` as its kind.

        Args:
            name: Registered kind whose dimensionality must match this quantity.

        Returns:
            A value-equivalent quantity carrying the requested kind.

        Raises:
            KeyError: If ``name`` is not registered.
            ValueError: If the kind has incompatible dimensions or affine role.
        """

        return self.__class__._from_parts(
            self._magnitude,
            self._units,
            kind=name,
            semantic_factors=self._semantic_factors,
        )

    def as_unkinded(self) -> SemanticQuantity:
        """Return a value-equivalent quantity without a semantic kind."""

        result = self.__class__._from_parts(
            self._magnitude,
            self._units,
            semantic_factors=self._semantic_factors,
        )
        result._kind = None
        return result

    def with_designation(self, designation: str) -> SemanticQuantity:
        """Return this quantity with its single eligible unit factor designated."""

        return self.__class__(
            self._magnitude, self.units.with_designation(designation), kind=self._kind
        )

    def as_undesignated(self) -> SemanticQuantity:
        """Return a value-equivalent quantity with all factor designations removed."""

        return self.__class__._from_parts(
            self._magnitude,
            self._units,
            kind=self._kind,
            semantic_factors=undesignated_factors(self._semantic_factors),
        )

    @override
    def __copy__(self) -> SemanticQuantity:
        return self.__class__._from_parts(
            copy.copy(self._magnitude),
            self._units,
            kind=self._kind,
            semantic_factors=self._semantic_factors,
        )

    @override
    def __deepcopy__(self, memo: dict[int, Any]) -> SemanticQuantity:
        return self.__class__._from_parts(
            copy.deepcopy(self._magnitude, memo),
            copy.deepcopy(self._units, memo),
            kind=self._kind,
            semantic_factors=copy.deepcopy(self._semantic_factors, memo),
        )

    @override
    def __iter__(self) -> Iterator[Any]:
        elements = iter(self._magnitude)
        return (self._result(element, self._units, self._kind) for element in elements)

    @override
    def __getitem__(self, key: Any) -> SemanticQuantity:
        result = cast(
            SemanticQuantity,
            super().__getitem__(key),  # type: ignore[no-untyped-call]
        )
        result._kind = self._kind
        result._semantic_factors = self._semantic_factors
        return result

    @override
    def tolist(self) -> Any:
        """Return nested quantity scalars without discarding semantic metadata."""

        try:
            values = self._magnitude.tolist()
        except AttributeError:
            raise AttributeError(
                f"Magnitude '{type(self._magnitude).__name__}' does not support tolist."
            ) from None

        def wrap(value: Any) -> Any:
            if isinstance(value, list):
                return [wrap(element) for element in value]
            return self._result(value, self._units, self._kind)

        return wrap(values)

    @override
    def __reduce__(self) -> Any:
        registry = cast(SemanticRegistry, self._REGISTRY)
        return _unpickle_semantic_quantity, (
            self._magnitude,
            self._units,
            self._kind,
            self._semantic_factors,
            registry.angle_dimension,
        )

    def _result(
        self,
        magnitude: Any,
        units: Any,
        kind: str | None,
        semantic_factors: _SemanticFactors | None = None,
    ) -> SemanticQuantity:
        return self.__class__._from_parts(
            magnitude,
            units,
            kind=kind,
            semantic_factors=(
                self._semantic_factors if semantic_factors is None else semantic_factors
            ),
        )

    def _set_result_semantics(
        self,
        result: SemanticQuantity,
        *,
        kind: str | None,
        semantic_factors: _SemanticFactors,
    ) -> SemanticQuantity:
        registry = cast(SemanticRegistry, self._REGISTRY)
        if kind is not None:
            registry._validate_kind(kind, result.dimensionality)
        registry._validate_affine_kind(result._units, kind)
        return self._attach_result_semantics(
            result,
            kind=kind,
            semantic_factors=semantic_factors,
        )

    def _attach_result_semantics(
        self,
        result: SemanticQuantity,
        *,
        kind: str | None,
        semantic_factors: _SemanticFactors,
    ) -> SemanticQuantity:
        result._kind = kind
        result._semantic_factors = semantic_factors
        return result

    def _inplace_candidate(self) -> SemanticQuantity:
        magnitude = self._magnitude
        if np is not None and isinstance(magnitude, np.ndarray):
            # Stage mutable NumPy buffers so a failed Pint operation cannot leave
            # either the values or their semantic interpretation partly updated.
            magnitude = magnitude.copy(order="K")
            magnitude.flags.writeable = self._magnitude.flags.writeable
        return self._result(magnitude, self._units, self._kind)

    def _commit_inplace_result(
        self, result: SemanticQuantity, staged_magnitude: Any
    ) -> SemanticQuantity:
        if (
            np is not None
            and isinstance(self._magnitude, np.ndarray)
            and result._magnitude is staged_magnitude
        ):
            # Copy back through the caller's array/view when Pint kept its buffer.
            # Read-only candidates can succeed only without changing that buffer.
            if self._magnitude.flags.writeable:
                self._magnitude[...] = staged_magnitude
        else:
            self._magnitude = result._magnitude
        self._units = result._units
        self._kind = result._kind
        self._semantic_factors = result._semantic_factors
        self._dimensionality = None
        return self

    def _require_matching_designations(self, other: SemanticQuantity) -> None:
        if not self._has_designated_factors() and not other._has_designated_factors():
            return
        if self.designation_signature != other.designation_signature:
            raise TypeError(
                "cannot combine quantities with different factor designations; "
                "convert to matching designated units or explicitly remove designations first."
            )

    def _require_same_semantic_state(
        self,
        other: SemanticQuantity,
        *,
        operation: str,
    ) -> None:
        self._require_matching_designations(other)
        if self.affine_role != other.affine_role:
            raise TypeError(
                f"cannot {operation} quantities with different affine roles "
                f"{self.affine_role!r} and {other.affine_role!r}."
            )
        self_kind = self._converted_kind()
        other_kind = other._converted_kind()
        if self_kind != other_kind:
            raise TypeError(
                f"cannot {operation} quantities with different kinds "
                f"{self_kind!r} and {other_kind!r}."
            )

    def _require_compatible_quantity(self, other: Any) -> SemanticQuantity | None:
        if isinstance(other, Measurement):
            registry = cast(SemanticRegistry, self._REGISTRY)
            registry._require_owned(other, name="Measurement operand")
            if self._is_semantically_classified():
                raise TypeError(
                    "classified quantities cannot use Measurement operands directly; "
                    "construct Quantity(measurement.magnitude, measurement.units) first"
                )
            return registry.Quantity(other.magnitude, other.units)
        if not isinstance(other, pint.registry.Quantity):
            return None
        if not isinstance(other, SemanticQuantity):
            raise ValueError(
                "xpint semantic quantities cannot be combined with ordinary Pint quantities; "
                "construct both values from the same SemanticRegistry."
            )
        if self._REGISTRY is not other._REGISTRY:
            raise ValueError(
                "xpint quantities belong to different SemanticRegistry instances; "
                "convert both through one registry before operating on them."
            )
        return other

    def _require_semantic_value_compatibility(
        self,
        other: Any,
        *,
        operation: str,
        allow_missing_kind: bool = False,
    ) -> SemanticQuantity | None:
        semantic_other = self._require_compatible_quantity(other)
        if semantic_other is None:
            if self.has_designation:
                raise TypeError(
                    f"cannot {operation} a designated quantity using an ordinary scalar; "
                    "explicitly remove the designation first."
                )
            return None

        if self.dimensionality == semantic_other.dimensionality:
            self._require_matching_designations(semantic_other)
            if not (allow_missing_kind and None in (self._kind, semantic_other._kind)):
                self._require_matching_kinds(semantic_other, operation)
        roles = (self.affine_role, semantic_other.affine_role)
        if roles[0] != roles[1] and None not in roles:
            raise TypeError(
                f"cannot {operation} quantities with affine roles {roles[0]!r} and {roles[1]!r}."
            )
        return semantic_other

    def _require_discrete_arithmetic_supported(self, other: Any, operation: str) -> None:
        semantic_other = self._require_compatible_quantity(other)
        if self._is_semantically_classified() or (
            semantic_other is not None and semantic_other._is_semantically_classified()
        ):
            raise TypeError(
                f"{operation} has no xpint semantic propagation rule for classified "
                "quantities yet; explicitly remove the metadata first."
            )

    def _require_matching_kinds(self, other: SemanticQuantity, operation: str) -> None:
        if self._kind != other._kind:
            raise TypeError(
                f"cannot {operation} quantity kinds {self._kind!r} and {other._kind!r}; "
                "use as_kind() or as_unkinded() to make the kinds match."
            )

    def _affine_add_sub(self, other: SemanticQuantity, operation: Any) -> SemanticQuantity | None:
        if not self._has_affine_semantics() and not other._has_affine_semantics():
            return None
        role_self = self.affine_role
        role_other = other.affine_role
        if role_self is None and role_other is None:
            return None
        if role_self is AffineRole.GENERIC and role_other is AffineRole.GENERIC:
            return None
        if AffineRole.GENERIC in (role_self, role_other):
            raise TypeError(
                "cannot combine a classified temperature with Generic Kelvin; "
                "use as_kind('Temperature'), as_kind('TemperatureDifference'), or a delta unit."
            )

        registry = cast(SemanticRegistry, self._REGISTRY)
        if operation is operator.add:
            if role_self is AffineRole.ABSOLUTE and role_other is AffineRole.ABSOLUTE:
                raise TypeError(
                    "cannot add two absolute temperatures; shift an absolute temperature "
                    "with a temperature difference instead."
                )
            if role_self is AffineRole.DELTA and role_other is AffineRole.DELTA:
                converted = other.to(self.units)
                return self._result(
                    self._magnitude + converted._magnitude,
                    self._units,
                    "TemperatureDifference",
                )
            if role_self is AffineRole.ABSOLUTE:
                delta_units = registry._delta_units_for(self.units)
                converted = other.to(delta_units)
                return self._result(
                    self._magnitude + converted._magnitude,
                    self._units,
                    "Temperature",
                )

            delta_units = registry._delta_units_for(other.units)
            converted = self.to(delta_units)
            return self._result(
                converted._magnitude + other._magnitude,
                other._units,
                "Temperature",
            )

        if role_self is AffineRole.DELTA and role_other is AffineRole.ABSOLUTE:
            raise TypeError(
                "cannot subtract an absolute temperature from a temperature difference."
            )
        if role_self is AffineRole.ABSOLUTE and role_other is AffineRole.ABSOLUTE:
            other = other.to(self.units)
            delta_units = registry._delta_units_for(self.units)
            return self._result(
                self._magnitude - other._magnitude,
                delta_units,
                "TemperatureDifference",
            )
        if role_self is AffineRole.DELTA and role_other is AffineRole.DELTA:
            converted = other.to(self.units)
            return self._result(
                self._magnitude - converted._magnitude,
                self._units,
                "TemperatureDifference",
            )
        if role_self is AffineRole.ABSOLUTE and role_other is AffineRole.DELTA:
            delta_units = registry._delta_units_for(self.units)
            converted = other.to(delta_units)
            return self._result(
                self._magnitude - converted._magnitude,
                self._units,
                "Temperature",
            )
        raise AssertionError(f"unhandled affine roles: {role_self!r}, {role_other!r}")

    @override
    def _add_sub(self, other: Any, op: Any) -> Any:
        semantic_other = self._require_compatible_quantity(other)
        if not self._quantity_operands_have_semantics(semantic_other):
            return super()._add_sub(other, op)

        if semantic_other is None:
            if self._has_designated_factors():
                raise TypeError(
                    "cannot combine a designated quantity with an ordinary scalar; "
                    "explicitly remove the designation first."
                )
            result = super()._add_sub(other, op)
            if result is NotImplemented:
                return result
            return self._attach_result_semantics(
                result,
                kind=self._kind,
                semantic_factors=self._semantic_factors,
            )

        if self.dimensionality != semantic_other.dimensionality and not (
            self._is_logarithmic and semantic_other._is_logarithmic
        ):
            raise DimensionalityError(
                self._units,
                semantic_other._units,
                str(self.dimensionality),
                str(semantic_other.dimensionality),
            )
        self._require_matching_designations(semantic_other)
        affine_result = self._affine_add_sub(semantic_other, op)
        if affine_result is not None:
            return affine_result

        self._require_matching_kinds(semantic_other, "add" if op is operator.add else "subtract")
        kind = self._kind
        if self._has_designated_factors() or semantic_other._has_designated_factors():
            converted = semantic_other.to(self.units)
            return self._result(
                op(self._magnitude, converted._magnitude),
                self._units,
                kind,
                self._semantic_factors,
            )
        result = super()._add_sub(semantic_other, op)
        if result is NotImplemented:
            return result
        return self._attach_result_semantics(
            result,
            kind=kind,
            semantic_factors=self._semantic_factors,
        )

    @override
    def _iadd_sub(self, other: Any, op: Any) -> Any:
        if op not in (operator.iadd, operator.isub):
            raise AssertionError(f"unsupported in-place additive operation: {op!r}")

        semantic_other = self._require_compatible_quantity(other)
        if not self._quantity_operands_have_semantics(semantic_other):
            return super()._iadd_sub(other, op)

        kind = self._kind
        factors = self._semantic_factors
        if semantic_other is None:
            if self._has_designated_factors():
                raise TypeError(
                    "cannot combine a designated quantity with an ordinary scalar; "
                    "explicitly remove the designation first."
                )
        else:
            if self.dimensionality != semantic_other.dimensionality and not (
                self._is_logarithmic and semantic_other._is_logarithmic
            ):
                raise DimensionalityError(
                    self._units,
                    semantic_other._units,
                    str(self.dimensionality),
                    str(semantic_other.dimensionality),
                )
            self._require_matching_designations(semantic_other)
            affine_result = self._affine_add_sub(
                semantic_other,
                operator.add if op is operator.iadd else operator.sub,
            )
            if affine_result is not None:
                computed = affine_result._magnitude
                target = self._magnitude
                if (
                    np is not None
                    and type(target) is np.ndarray
                    and type(computed) is np.ndarray
                    and computed.shape == target.shape
                    and computed.dtype == target.dtype
                    and target.flags.writeable
                ):
                    # The affine result is already a complete buffer independent of
                    # the target, and an equal-dtype, equal-shape write cannot fail,
                    # so it can be committed without staging a second copy.
                    return self._commit_inplace_result(affine_result, computed)
                candidate = self._inplace_candidate()
                staged_magnitude = candidate._magnitude
                if np is not None and isinstance(staged_magnitude, np.ndarray):
                    np.copyto(staged_magnitude, affine_result._magnitude, casting="same_kind")
                    affine_result._magnitude = staged_magnitude
                return self._commit_inplace_result(affine_result, staged_magnitude)
            self._require_matching_kinds(
                semantic_other, "add" if op is operator.iadd else "subtract"
            )
            kind = self._kind

        result = super()._iadd_sub(other, op)
        if result is NotImplemented:
            return result
        return self._attach_result_semantics(result, kind=kind, semantic_factors=factors)

    @override
    def __iadd__(self, other: Any) -> Any:
        return super().__iadd__(other)

    @override
    def __isub__(self, other: Any) -> SemanticQuantity:
        return cast(SemanticQuantity, super().__isub__(other))  # type: ignore[no-untyped-call]

    def _multiplicative_kind(
        self,
        other: SemanticQuantity | None,
        unit_operation: Any,
    ) -> str | None:
        if other is None:
            return self._kind
        if unit_operation is operator.mul:
            if self._kind is not None and not other.has_kind() and other.is_semantic_unity:
                return self._kind
            if other._kind is not None and not self.has_kind() and self.is_semantic_unity:
                return other._kind
        if (
            unit_operation is operator.truediv
            and self._kind is not None
            and not other.has_kind()
            and other.is_semantic_unity
        ):
            return self._kind
        return None

    @override
    def _mul_div(self, other: Any, magnitude_op: Any, units_op: Any = None) -> Any:
        if isinstance(other, SemanticUnit):
            if self._REGISTRY is not other._REGISTRY:
                raise ValueError(
                    "xpint quantities and units belong to different SemanticRegistry "
                    "instances; use one registry for the entire expression."
                )
            other = self._REGISTRY.Quantity(1, other)
        semantic_other = self._require_compatible_quantity(other)
        if not self._quantity_operands_have_semantics(semantic_other):
            return super()._mul_div(other, magnitude_op, units_op)

        if self._has_absolute_temperature_semantics() or (
            semantic_other is not None and semantic_other._has_absolute_temperature_semantics()
        ):
            raise TypeError(
                "absolute temperatures cannot participate in multiplication or division; "
                "use a temperature difference or explicitly remove the absolute interpretation."
            )

        unit_operation = magnitude_op if units_op is None else units_op
        if semantic_other is None:
            factors = self._semantic_factors
            expected_units = self._units
        else:
            expected_units = unit_operation(self._units, semantic_other._units)
            factors = combine_factors(
                self._semantic_factors,
                semantic_other._semantic_factors,
                divide=unit_operation is operator.truediv,
            )
        result = super()._mul_div(other, magnitude_op, units_op)
        if result is NotImplemented:
            return result
        if result._units != expected_units:
            factors = canonical_factors(self._REGISTRY, result._units, factors)
        return self._attach_result_semantics(
            result,
            kind=self._multiplicative_kind(semantic_other, unit_operation),
            semantic_factors=factors,
        )

    @override
    def _imul_div(self, other: Any, magnitude_op: Any, units_op: Any = None) -> Any:
        if magnitude_op not in (operator.imul, operator.itruediv):
            raise AssertionError(f"unsupported in-place multiplicative operation: {magnitude_op!r}")

        if isinstance(other, SemanticUnit):
            if self._REGISTRY is not other._REGISTRY:
                raise ValueError(
                    "xpint quantities and units belong to different SemanticRegistry "
                    "instances; use one registry for the entire expression."
                )
            other = self._REGISTRY.Quantity(1, other)
        semantic_other = self._require_compatible_quantity(other)
        if not self._quantity_operands_have_semantics(semantic_other):
            result = super()._imul_div(other, magnitude_op, units_op)
            if result is not NotImplemented:
                result._dimensionality = None
                result._semantic_factors = cast(
                    SemanticRegistry, self._REGISTRY
                )._factors_for_units(result._units)
            return result

        if self._has_absolute_temperature_semantics() or (
            semantic_other is not None and semantic_other._has_absolute_temperature_semantics()
        ):
            raise TypeError(
                "absolute temperatures cannot participate in multiplication or division; "
                "use a temperature difference or explicitly remove the absolute interpretation."
            )

        unit_operation = magnitude_op if units_op is None else units_op
        kind_operation = (
            operator.mul if unit_operation in (operator.imul, operator.mul) else operator.truediv
        )
        kind = self._multiplicative_kind(semantic_other, kind_operation)
        factors = (
            self._semantic_factors
            if semantic_other is None
            else combine_factors(
                self._semantic_factors,
                semantic_other._semantic_factors,
                divide=kind_operation is operator.truediv,
            )
        )
        expected_units = (
            self._units
            if semantic_other is None
            else kind_operation(self._units, semantic_other._units)
        )
        candidate = self._inplace_candidate()
        staged_magnitude = candidate._magnitude
        # Pint's automatic reduction runs inside its arithmetic hook, after
        # changing physical units. Give it the result's semantic state in advance.
        candidate._kind = kind
        candidate._semantic_factors = factors
        result = super(SemanticQuantity, candidate)._imul_div(other, magnitude_op, units_op)
        if result is NotImplemented:
            return result
        result._dimensionality = None
        if result._units != expected_units:
            factors = canonical_factors(self._REGISTRY, result._units, factors)
        self._attach_result_semantics(result, kind=kind, semantic_factors=factors)
        return self._commit_inplace_result(result, staged_magnitude)

    @override
    def __imul__(self, other: Any) -> SemanticQuantity:
        return cast(SemanticQuantity, super().__imul__(other))  # type: ignore[no-untyped-call]

    @override
    def __itruediv__(self, other: Any) -> SemanticQuantity:
        return cast(SemanticQuantity, super().__itruediv__(other))  # type: ignore[no-untyped-call]

    @override
    def __rtruediv__(self, other: Any) -> Any:
        if self.affine_role is AffineRole.ABSOLUTE:
            raise TypeError(
                "absolute temperatures cannot participate in multiplication or division; "
                "use a temperature difference or explicitly remove the absolute interpretation."
            )
        result = super().__rtruediv__(other)  # type: ignore[no-untyped-call]
        if result is not NotImplemented:
            result._kind = None
            factors = power_factors(self._semantic_factors, -1)
            if result._units != self._units**-1:
                factors = canonical_factors(self._REGISTRY, result._units, factors)
            result._semantic_factors = factors
        return result

    @override
    def __floordiv__(self, other: Any) -> Any:
        self._require_discrete_arithmetic_supported(other, "floor division")
        return super().__floordiv__(other)

    @override
    def __rfloordiv__(self, other: Any) -> Any:
        self._require_discrete_arithmetic_supported(other, "floor division")
        return super().__rfloordiv__(other)

    @override
    def __ifloordiv__(self, other: Any) -> Any:
        self._require_discrete_arithmetic_supported(other, "floor division")
        return super().__ifloordiv__(other)  # type: ignore[no-untyped-call]

    @override
    def __mod__(self, other: Any) -> Any:
        self._require_discrete_arithmetic_supported(other, "modulo")
        return super().__mod__(other)

    @override
    def __rmod__(self, other: Any) -> Any:
        self._require_discrete_arithmetic_supported(other, "modulo")
        return super().__rmod__(other)

    @override
    def __imod__(self, other: Any) -> Any:
        self._require_discrete_arithmetic_supported(other, "modulo")
        return super().__imod__(other)

    @override
    def __divmod__(self, other: Any) -> Any:
        self._require_discrete_arithmetic_supported(other, "divmod")
        return super().__divmod__(other)

    @override
    def __rdivmod__(self, other: Any) -> Any:
        self._require_discrete_arithmetic_supported(other, "divmod")
        return super().__rdivmod__(other)

    def _requested_target(self, other: Any) -> SemanticUnit:
        # The target exactly as the caller wrote it, before any designation adoption.
        registry = cast(SemanticRegistry, self._REGISTRY)
        registry._require_owned(other, name="target units")
        target_container = (
            self.UnitsContainer()
            if other is None
            else to_units_container(other, cast(pint.UnitRegistry[Any], registry))
        )
        target_factors = getattr(other, "_semantic_factors", None)
        return registry.Unit._from_parts(
            target_container,
            semantic_factors=(
                registry._factors_for_units(target_container)
                if target_factors is None
                else target_factors
            ),
        )

    def _conversion_target(self, other: Any) -> SemanticUnit:
        registry = cast(SemanticRegistry, self._REGISTRY)
        target_units = self._requested_target(other)
        if not target_units.has_designation and self.has_designation:
            target_units = self._adopt_designations(target_units)
        if self.designation_signature != target_units.designation_signature:
            raise TypeError(
                "cannot convert between different factor designations; "
                "the target unit must carry the same designation signature."
            )
        target_role = registry._affine_role(target_units, self._kind)
        source_role = self.affine_role
        if source_role is AffineRole.GENERIC and target_role in (
            AffineRole.ABSOLUTE,
            AffineRole.DELTA,
        ):
            raise TypeError(
                "cannot convert Generic Kelvin to an absolute or delta scale; "
                "classify it with as_kind() first."
            )
        if (
            source_role in (AffineRole.ABSOLUTE, AffineRole.DELTA)
            and target_role in (AffineRole.ABSOLUTE, AffineRole.DELTA)
            and target_role is not source_role
        ):
            raise TypeError(
                f"cannot convert a {source_role.value} temperature to a {target_role.value} scale."
            )

        return target_units

    def _adopt_designations(self, target: SemanticUnit) -> SemanticUnit:
        # A single-token target such as ``mM`` or ``cm`` takes the label through its
        # own slot, so ``mmol[O2]/L -> mM`` reads ``mM[O2]`` rather than the physically
        # equivalent but opaque root-factor form canonical_factors would produce.
        registry = cast(SemanticRegistry, self._REGISTRY)
        signature = self.designation_signature
        if len(signature) == 1 and len(target._semantic_factors) == 1:
            try:
                candidate = target.with_designation(signature[0].designation)
            except ValueError:
                candidate = None
            if candidate is not None and candidate.designation_signature == signature:
                return candidate
        return registry.Unit._from_parts(
            target._units,
            semantic_factors=canonical_factors(registry, target._units, self._semantic_factors),
        )

    @override
    def to(self, other: Any = None, *contexts: Any, **ctx_kwargs: Any) -> SemanticQuantity:
        """Convert units, retaining shape, kind, and source designations.

        Plain targets adopt source designations; designated targets must match.
        Pint controls numeric scaling and dtype changes.

        Raises:
            TypeError: If designation or affine roles conflict, or a context
                changes a kinded value's dimensions; call ``as_unkinded()`` first.
        """
        target_units = self._conversion_target(other)
        result = super().to(target_units, *contexts, **ctx_kwargs)
        self._require_kind_conversion_dimensions(result)
        return self._set_result_semantics(
            result,
            kind=self._converted_kind(),
            semantic_factors=target_units._semantic_factors,
        )

    @override
    def ito(self, other: Any = None, *contexts: Any, **ctx_kwargs: Any) -> None:
        """Convert in place with Pint's dtype and buffer-alias behavior.

        Conversions performed in the existing NumPy buffer retain its shape and
        dtype. A temporary buffer is staged before committing, so casting or
        semantic failures leave the original values and metadata unchanged. Pint
        context conversions may replace the buffer and change dtype, as in Pint.
        The failure-safety guarantee applies to NumPy buffers; other duck arrays
        retain Pint's own mutation behavior. Plain targets adopt source designations.

        Raises:
            TypeError: If designation or affine roles conflict, or a context
                changes a kinded value's dimensions; call ``as_unkinded()`` first.
        """
        target_units = self._conversion_target(other)
        candidate = self._inplace_candidate()
        staged_magnitude = candidate._magnitude
        super(SemanticQuantity, candidate).ito(target_units, *contexts, **ctx_kwargs)
        candidate._dimensionality = None
        self._require_kind_conversion_dimensions(candidate)
        self._set_result_semantics(
            candidate, kind=self._converted_kind(), semantic_factors=target_units._semantic_factors
        )
        self._commit_inplace_result(candidate, staged_magnitude)

    @override
    def m_as(self, units: Any) -> Any:
        """Return the magnitude expressed in ``units``, which must name every designation.

        A bare magnitude is where designation checks end, so unlike ``to()`` a plain
        target does not adopt the source's designations: the caller must state what the
        number means. ``(5 * mmol["O2"] / L).m_as(mmol["O2"] / L)`` is ``5``, while
        ``m_as("mmol/L")`` on the same value raises. Values without designations behave
        as in Pint. Kinds are dropped with the units, as ``.magnitude`` drops them.

        Raises:
            TypeError: If the value carries designations and the target names none, or
                the target's designations or affine role conflict with the value's.
        """
        if self.has_designation and not self._requested_target(units).has_designation:
            raise TypeError(
                f"m_as() needs a target that carries the designations of {self.units}; "
                "a plain target would extract a number without checking them. Pass a unit "
                "object with the same designations, such as mmol['O2'] / L, or call "
                "as_undesignated() first to drop them deliberately."
            )
        return self.to(units).magnitude

    def _require_kind_conversion_dimensions(self, result: SemanticQuantity) -> None:
        if self._kind is not None and result.dimensionality != self.dimensionality:
            raise TypeError(
                "context conversion changes the dimensions of a kinded quantity; "
                "call as_unkinded() before converting"
            )

    @override
    def to_timedelta(self) -> datetime.timedelta:
        if self._is_semantically_classified():
            raise TypeError(
                "cannot convert a classified xpint quantity to datetime.timedelta because "
                "that type cannot represent kind or designation metadata; explicitly remove "
                "the metadata first."
            )
        return super().to_timedelta()

    @override
    def to_base_units(self) -> SemanticQuantity:
        result = cast(SemanticQuantity, super().to_base_units())
        result._kind = self._converted_kind()
        result._semantic_factors = canonical_factors(
            self._REGISTRY,
            result._units,
            self._semantic_factors,
        )
        return result

    @override
    def to_root_units(self) -> SemanticQuantity:
        result = cast(SemanticQuantity, super().to_root_units())
        result._kind = self._converted_kind()
        result._semantic_factors = canonical_factors(
            self._REGISTRY,
            result._units,
            self._semantic_factors,
        )
        return result

    @override
    def ito_base_units(self) -> None:
        """Convert in place to Pint's base units, following ``ito()`` semantics."""
        _, units = self._REGISTRY._get_base_units(self._units)
        factors = canonical_factors(self._REGISTRY, units, self._semantic_factors)
        self.ito(
            cast(SemanticRegistry, self._REGISTRY).Unit._from_parts(units, semantic_factors=factors)
        )

    @override
    def ito_root_units(self) -> None:
        """Convert in place to Pint's root units, following ``ito()`` semantics."""
        _, units = self._REGISTRY._get_root_units(self._units)
        factors = canonical_factors(self._REGISTRY, units, self._semantic_factors)
        self.ito(
            cast(SemanticRegistry, self._REGISTRY).Unit._from_parts(units, semantic_factors=factors)
        )

    def _reduced_target(self) -> SemanticUnit:
        # Ask Pint to choose the physical target without exposing designations
        # to an API that accepts only physical unit containers.
        probe = self.__class__(1, self._units)
        reduced = super(SemanticQuantity, probe).to_reduced_units()
        if reduced._units == self._units:
            return self.units
        factors = canonical_factors(self._REGISTRY, reduced._units, self._semantic_factors)
        return cast(SemanticRegistry, self._REGISTRY).Unit._from_parts(
            reduced._units, semantic_factors=factors
        )

    @override
    def to_reduced_units(self) -> SemanticQuantity:
        """Reduce units with Pint while retaining factor designations and scale.

        Shape and dtype follow ``to()``. As in Pint, an already reduced quantity
        may be returned unchanged.
        """
        if not self._is_semantically_classified():
            return cast(SemanticQuantity, super().to_reduced_units())
        target = self._reduced_target()
        return self if len(self._units) == 1 and not self.dimensionless else self.to(target)

    @override
    def ito_reduced_units(self) -> None:
        """Reduce units in place, retaining designations and ``ito()`` behavior."""
        # Pint also calls this immediately after mutating an arithmetic result.
        self._dimensionality = None
        if not self._is_semantically_classified():
            super().ito_reduced_units()
            return
        target = self._reduced_target()
        if target._units != self._units:
            self.ito(target)

    def _converted_kind(self) -> str | None:
        role = self.affine_role
        if role is AffineRole.ABSOLUTE:
            return "Temperature"
        if role is AffineRole.DELTA:
            return "TemperatureDifference"
        return self._kind

    @override
    def is_compatible_with(self, other: Any, *contexts: Any, **ctx_kwargs: Any) -> bool:
        """Return whether another value has matching physical and semantic state.

        Quantity operands must belong to this registry and have the same kind,
        designation signature, and affine role. Units and unit strings are
        compatible when they are accepted conversion targets.
        Semantic conversion failures return ``False`` even when a Pint context is active.
        """

        try:
            if isinstance(other, (SemanticUnit, str)):
                target = self._conversion_target(other)
                if self._kind is not None and target.dimensionality != self.dimensionality:
                    return False
                return bool(super().is_compatible_with(target, *contexts, **ctx_kwargs))
            if isinstance(other, SemanticQuantity):
                if self._REGISTRY is not other._REGISTRY:
                    return False
                if self._semantic_state() != other._semantic_state():
                    return False
            elif (
                isinstance(other, (pint.registry.Quantity, pint.registry.Unit))
                or self._semantic_state() != _UNCLASSIFIED_STATE
            ):
                return False
            return bool(super().is_compatible_with(other, *contexts, **ctx_kwargs))
        except (DimensionalityError, TypeError, ValueError):
            return False

    @override
    def __eq__(self, other: Any) -> Any:
        semantic_other: SemanticQuantity | None
        if isinstance(other, SemanticQuantity):
            if self._REGISTRY is not other._REGISTRY:
                return self._false_comparison(other)
            semantic_other = other
        elif isinstance(other, SemanticUnit):
            if self._REGISTRY is not other._REGISTRY:
                return self._false_comparison(other)
            semantic_other = cast(SemanticQuantity, self._REGISTRY.Quantity(1, other))
        elif isinstance(other, pint.registry.Quantity):
            raise ValueError(
                "xpint semantic quantities cannot be compared with ordinary Pint "
                "quantities; construct both values from the same SemanticRegistry."
            )
        elif isinstance(other, pint.registry.Unit):
            return self._false_comparison(other)
        else:
            semantic_other = None
        if semantic_other is not None:
            if self._semantic_state() != semantic_other._semantic_state():
                return self._false_comparison(other)
            if self._units != semantic_other._units and not (
                self._is_multiplicative and semantic_other._is_multiplicative
            ):
                # Pint treats two all-zero magnitudes as equal before applying
                # offsets, which would make 0 degC equal 0 degF. Compare base-unit
                # magnitudes instead: the result is symmetric and matches __hash__.
                if self.dimensionality != semantic_other.dimensionality:
                    return self._false_comparison(other)
                return eq(
                    self.to_base_units()._magnitude,
                    semantic_other.to_base_units()._magnitude,
                    False,
                )
        elif self._semantic_state() != _UNCLASSIFIED_STATE:
            return self._false_comparison(other)
        return super().__eq__(other)

    @override
    def __hash__(self) -> int:
        """Hash scalar physical values and semantics; array magnitudes are unhashable."""
        base = self.to_base_units()
        if base.dimensionless and base._semantic_state() == _UNCLASSIFIED_STATE:
            return hash(base.magnitude)
        return hash(
            (
                id(self._REGISTRY),
                base.magnitude,
                base._units,
                base._converted_kind(),
                base.designation_signature,
                base.affine_role,
            )
        )

    def _false_comparison(self, other: Any) -> Any:
        other_magnitude = getattr(other, "_magnitude", other)
        if not is_duck_array_type(type(self._magnitude)) and not is_duck_array_type(
            type(other_magnitude)
        ):
            return False
        template, _ = np.broadcast_arrays(self._magnitude, other_magnitude)
        return np.full_like(template, fill_value=False, dtype=np.bool_)

    @override
    def compare(self, other: Any, op: Any) -> Any:
        semantic_other = self._require_compatible_quantity(other)
        if semantic_other is not None:
            if self.dimensionality == semantic_other.dimensionality:
                self._require_same_semantic_state(
                    semantic_other,
                    operation="order",
                )
        elif self.dimensionless and self._semantic_state() != _UNCLASSIFIED_STATE:
            raise TypeError(
                "cannot order a classified quantity against an ordinary scalar; "
                "explicitly remove its semantic metadata first."
            )
        return super().compare(other, op)

    def _reject_angle_coercion(self, target: str) -> None:
        """Explain how to leave an angle registry instead of raising Pint's bare error.

        Fires only on a registry built with ``angle_dimension=True``, where an angle
        genuinely is not a number. Names no default convention on purpose: the caller
        states the unit, which is the distinction the switch exists to preserve.

        The advice is derived from the quantity's actual angular exponents rather than
        assumed. A steradian needs two factors, a torque needs one in the opposite
        direction, and a fixed suggestion naming one radian is wrong for both.
        """

        registry = cast(SemanticRegistry, self._REGISTRY)
        if not registry.angle_dimension:
            return
        angular = {
            name: exponent
            for name, exponent in self.dimensionality.items()
            if name in registry._angular_dimensions and exponent
        }
        if not angular:
            return
        carried = " * ".join(
            _format_unit_power(registry._angular_units[name], exponent)
            for name, exponent in sorted(angular.items())
        )
        clearing = " * ".join(
            _format_unit_power(registry._angular_units[name], -exponent)
            for name, exponent in sorted(angular.items())
        )
        # m_as needs a target of the same dimensionality, so a powered or mixed angular
        # value must be offered the whole expression rather than one of its factors.
        purely_angular = len(angular) == len(self.dimensionality)
        naming = (
            f' name a unit of the same kind with .m_as("{carried}"), or' if purely_angular else ""
        )
        raise DimensionalityError(
            self._units,
            "dimensionless",
            extra_msg=(
                f" - an angle is not a number under angle_dimension=True, so {target}() "
                f"cannot choose a convention for you. This quantity carries {carried};"
                f"{naming} clear it by multiplying by {clearing}."
            ),
        )

    @override
    def __float__(self) -> float:
        if self.has_designation:
            raise TypeError(
                "cannot coerce a quantity with retained factor designations to float; "
                "explicitly remove the designations first."
            )
        if self._kind is not None:
            raise TypeError(
                "cannot coerce a quantity with a retained kind to float; "
                "explicitly remove the kind first."
            )
        self._reject_angle_coercion("float")
        return super().__float__()

    @override
    def __int__(self) -> int:
        if self.has_designation:
            raise TypeError(
                "cannot coerce a quantity with retained factor designations to int; "
                "explicitly remove the designations first."
            )
        if self._kind is not None:
            raise TypeError(
                "cannot coerce a quantity with a retained kind to int; "
                "explicitly remove the kind first."
            )
        self._reject_angle_coercion("int")
        return super().__int__()

    @override
    def __complex__(self) -> complex:
        if self.has_designation:
            raise TypeError(
                "cannot coerce a quantity with retained factor designations to complex; "
                "explicitly remove the designations first."
            )
        if self._kind is not None:
            raise TypeError(
                "cannot coerce a quantity with a retained kind to complex; "
                "explicitly remove the kind first."
            )
        self._reject_angle_coercion("complex")
        return super().__complex__()

    def _power_exponent(self, other: Any) -> Fraction | None:
        cast(SemanticRegistry, self._REGISTRY)._require_owned(other, name="exponent")
        if self.affine_role is AffineRole.ABSOLUTE:
            raise TypeError(
                f"absolute temperatures cannot be exponentiated; {_ABSOLUTE_LINEAR_ESCAPE}."
            )
        if not self._operands_have_semantics(self, other):
            return None
        if isinstance(other, SemanticQuantity):
            if other._is_semantically_classified():
                raise TypeError("a classified quantity cannot be used as an exponent.")
            if not other.is_semantic_unity:
                raise TypeError("an exponent quantity must be dimensionless and undesignated.")
            exponent_value = other.to_root_units().magnitude
        else:
            exponent_value = other
        try:
            return _as_fraction(exponent_value)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise TypeError("classified quantities require a scalar rational exponent.") from exc

    @override
    def __pow__(self, other: Any) -> Any:
        exponent = self._power_exponent(other)
        result = super().__pow__(other)
        if exponent is None or result is NotImplemented or not isinstance(result, SemanticQuantity):
            return result
        return self._set_result_semantics(
            result,
            kind=self._kind if exponent == 1 else None,
            semantic_factors=power_factors(self._semantic_factors, exponent),
        )

    @override
    def __ipow__(self, other: Any) -> Any:
        if not is_duck_array_type(type(self._magnitude)):
            return self.__pow__(other)
        exponent = self._power_exponent(other)
        candidate = self._inplace_candidate()
        staged_magnitude = candidate._magnitude
        result = super(SemanticQuantity, candidate).__ipow__(other)
        if result is NotImplemented:
            return result
        result._dimensionality = None
        factors = (
            cast(SemanticRegistry, self._REGISTRY)._factors_for_units(result._units)
            if exponent is None
            else power_factors(self._semantic_factors, exponent)
        )
        self._set_result_semantics(
            result, kind=self._kind if exponent == 1 else None, semantic_factors=factors
        )
        return self._commit_inplace_result(result, staged_magnitude)

    @override
    def __rpow__(self, other: Any) -> Any:
        if self._is_semantically_classified():
            raise TypeError("a classified quantity cannot be used as an exponent.")
        return super().__rpow__(other)

    @override
    def __abs__(self) -> SemanticQuantity:
        if self.affine_role is AffineRole.ABSOLUTE:
            raise TypeError(
                "absolute temperatures cannot be passed through absolute value; "
                f"{_ABSOLUTE_LINEAR_ESCAPE}."
            )
        return self._result(abs(self._magnitude), self._units, self._kind)

    @override
    def __pos__(self) -> SemanticQuantity:
        return self._result(operator.pos(self._magnitude), self._units, self._kind)

    @override
    def __neg__(self) -> SemanticQuantity:
        if self.affine_role is AffineRole.ABSOLUTE:
            raise TypeError(f"absolute temperatures cannot be negated; {_ABSOLUTE_LINEAR_ESCAPE}.")
        return self._result(operator.neg(self._magnitude), self._units, self._kind)

    @override
    def __round__(self, ndigits: int | None = None) -> SemanticQuantity:
        return self._result(round(self._magnitude, ndigits), self._units, self._kind)

    def _prepare_reduction_kwargs(
        self, name: str, kwargs: dict[str, Any], *, method: bool
    ) -> tuple[dict[str, Any], str | None]:
        mean = kwargs.get("mean")
        if isinstance(mean, pint.registry.Quantity):
            semantic_mean = self._require_compatible_quantity(mean)
            assert semantic_mean is not None
            self._require_same_semantic_state(
                semantic_mean, operation="use as a reduction mean for"
            )
            kwargs = {**kwargs, "mean": semantic_mean.to(self.units).magnitude}
        initial = kwargs.get("initial")
        result_kind = self._kind
        wants_magnitude = method or name in _NUMPY_FUNCTION_ADDITIVE
        if initial is None or initial is np._NoValue:
            return kwargs, result_kind
        if isinstance(initial, pint.registry.Quantity):
            semantic_initial = self._require_compatible_quantity(initial)
            assert semantic_initial is not None
            converted_initial = semantic_initial.to(self.units)
            if name in _NUMPY_FUNCTION_ADDITIVE:
                self._require_semantic_value_compatibility(
                    semantic_initial, operation="use as a reduction initial value for"
                )
            else:
                self._require_same_semantic_state(
                    semantic_initial, operation="use as a reduction initial value for"
                )
        elif wants_magnitude:
            return kwargs, result_kind
        else:
            # Pint extrema expect a unit-bearing initial; xpint defines bare
            # initial values as magnitudes in the input's existing unit.
            converted_initial = self._result(initial, self._units, self._kind)
        initial = converted_initial.magnitude if wants_magnitude else converted_initial
        return {**kwargs, "initial": initial}, result_kind

    @override
    def __getattr__(self, name: str) -> Any:
        """Forward array metadata; reject undefined classified magnitude access.

        Pint's wrapped methods keep their semantic dispatch. Other methods and
        storage exports require explicit access through ``magnitude`` when this
        quantity has a kind, designation, or affine role. Returned ungoverned
        callables raise TypeError on invocation.

        Raises:
            AttributeError: If a classified quantity requests raw storage or an
                ungoverned magnitude property.
        """
        if name in _NUMPY_MAGNITUDE_INDICES:
            return getattr(self._magnitude, name)
        attribute = super().__getattr__(name)
        if "_semantic_factors" not in self.__dict__ or not self._is_semantically_classified():
            return attribute
        if name in _MAGNITUDE_METADATA or (
            isinstance(attribute, partial) and attribute.func == self._numpy_method_wrap
        ):
            return attribute
        message = (
            f"magnitude attribute {name!r} has no xpint semantic propagation rule; "
            "operate on magnitude explicitly"
        )
        if callable(attribute):

            def reject(*args: Any, **kwargs: Any) -> Any:
                raise TypeError(message)

            return reject
        raise AttributeError(message)

    @override
    def _numpy_method_wrap(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        name = func.__name__
        if name in _NUMPY_METHOD_REDUCTIONS and args:
            bound = _reduction_signature(name).bind(self, *args, **kwargs)
            kwargs = dict(bound.arguments)
            kwargs.pop("a")
            args = ()
        is_classified = self._is_semantically_classified()
        if not is_classified and name not in _NUMPY_METHOD_REDUCTIONS and kwargs.get("out") is None:
            return super()._numpy_method_wrap(func, *args, **kwargs)  # type: ignore[no-untyped-call]
        if (
            name in (_NUMPY_METHOD_ADDITIVE | _NUMPY_STATISTICS)
            and self.affine_role is AffineRole.ABSOLUTE
        ):
            raise TypeError(f"NumPy method {name!r} is undefined for absolute temperatures.")
        if is_classified and name not in _NUMPY_METHOD_SUPPORTED:
            raise TypeError(
                f"NumPy method {name!r} has no xpint semantic propagation rule yet; "
                "use as_unkinded() or operate on magnitude explicitly."
            )
        if kwargs.get("out") is not None:
            raise TypeError(
                f"NumPy method {name!r} with an output target is rejected because Pint "
                "does not validate or preserve the target quantity's units."
            )

        call_kwargs, result_kind = (
            self._prepare_reduction_kwargs(name, kwargs, method=True)
            if name in _NUMPY_METHOD_REDUCTIONS
            else (kwargs, self._kind)
        )

        result = super()._numpy_method_wrap(  # type: ignore[no-untyped-call]
            func, *args, **call_kwargs
        )
        if isinstance(result, SemanticQuantity) and name in _NUMPY_METHOD_SUPPORTED:
            return self._set_result_semantics(
                result,
                kind=None if name in _NUMPY_VARIANCES else result_kind,
                semantic_factors=(
                    power_factors(self._semantic_factors, 2)
                    if name in _NUMPY_VARIANCES
                    else self._semantic_factors
                ),
            )
        return result

    @override
    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Any:
        if self._is_semantically_classified():
            raise TypeError(
                "cannot downcast a classified xpint quantity to ndarray because the array "
                "cannot represent semantic metadata; operate on magnitude explicitly."
            )
        # Pint would strip the angle with only a warning; refuse it as float() does.
        self._reject_angle_coercion("numpy.asarray")
        result = super().__array__(dtype)
        if dtype is not None:
            converted = result.astype(dtype, copy=False)
            if copy is False and converted is not result:
                raise ValueError("copy=False cannot satisfy the requested dtype conversion.")
            result = converted
        if copy is True:
            result = result.copy()
        return result

    @override
    def clip(self, min: Any = None, max: Any = None, out: Any = None, **kwargs: Any) -> Any:
        if not self._operands_have_semantics(self, min, max, out):
            return super().clip(min=min, max=max, out=out, **kwargs)  # type: ignore[no-untyped-call]
        if min is not None:
            self._require_semantic_value_compatibility(
                min, operation="clip", allow_missing_kind=True
            )
        if max is not None:
            self._require_semantic_value_compatibility(
                max, operation="clip", allow_missing_kind=True
            )
        if out is not None and self._is_semantically_classified():
            raise TypeError("clip with an output target has no xpint semantic rule yet.")
        result = cast(
            SemanticQuantity,
            super().clip(min=min, max=max, out=out, **kwargs),  # type: ignore[no-untyped-call]
        )
        return self._set_result_semantics(
            result,
            kind=self._kind,
            semantic_factors=self._semantic_factors,
        )

    @override
    def fill(self, value: Any) -> None:
        if not self._operands_have_semantics(self, value):
            super().fill(value)
            self._dimensionality = None
            self._semantic_factors = cast(SemanticRegistry, self._REGISTRY)._factors_for_units(
                self._units
            )
            return

        semantic_value = self._require_semantic_value_compatibility(value, operation="fill")
        result_kind = self._kind
        if semantic_value is not None:
            # Probe only. Pint's fill replaces units; do not use the converted magnitude.
            semantic_value.to(self.units)
        super().fill(value)
        self._dimensionality = None
        factors = (
            semantic_value._semantic_factors
            if semantic_value is not None
            else cast(SemanticRegistry, self._REGISTRY)._factors_for_units(self._units)
        )
        self._set_result_semantics(self, kind=result_kind, semantic_factors=factors)

    @override
    def put(self, indices: Any, values: Any, mode: str = "raise") -> None:
        self._require_semantic_value_compatibility(
            values,
            operation="put values into",
        )
        super().put(indices, values, mode)

    @override
    def __setitem__(self, key: Any, value: Any) -> None:
        self._require_semantic_value_compatibility(
            value,
            operation="assign to",
        )
        super().__setitem__(key, value)  # type: ignore[no-untyped-call]

    def filled(self, fill_value: Any = None) -> Any:
        """Return a filled copy, treating a scalar fill as an in-unit magnitude."""

        magnitude_fill = fill_value
        if isinstance(fill_value, pint.registry.Quantity):
            semantic_fill = self._require_semantic_value_compatibility(
                fill_value,
                operation="fill masked values in",
            )
            assert semantic_fill is not None
            magnitude_fill = semantic_fill.to(self.units).magnitude
        try:
            magnitude = self._magnitude.filled(magnitude_fill)
        except AttributeError:
            raise AttributeError(
                f"Magnitude '{type(self._magnitude).__name__}' does not support filled."
            ) from None
        if not self._is_semantically_classified():
            return magnitude
        return self._result(magnitude, self._units, self._kind)

    def compressed(self) -> Any:
        """Return unmasked values with units and semantic metadata intact."""

        try:
            magnitude = self._magnitude.compressed()
        except AttributeError:
            raise AttributeError(
                f"Magnitude '{type(self._magnitude).__name__}' does not support compressed."
            ) from None
        if not self._is_semantically_classified():
            return magnitude
        return self._result(magnitude, self._units, self._kind)

    @override
    def searchsorted(self, value: Any, side: str = "left", sorter: Any = None) -> Any:
        self._require_semantic_value_compatibility(value, operation="search")
        return super().searchsorted(value, side, sorter)  # type: ignore[no-untyped-call]

    @property
    @override
    def real(self) -> SemanticQuantity:
        return self._result(self._magnitude.real, self._units, self._kind)

    @property
    @override
    def imag(self) -> SemanticQuantity:
        return self._result(self._magnitude.imag, self._units, self._kind)

    @property
    @override
    def T(self) -> SemanticQuantity:
        return self._result(self._magnitude.T, self._units, self._kind)

    @property
    @override
    def flat(self) -> Iterator[SemanticQuantity]:
        for value in self._magnitude.flat:
            yield self._result(value, self._units, self._kind)

    @override
    def __str__(self) -> str:
        if not self.has_designation:
            return super().__str__()
        return f"{self._magnitude} {format_factors(self._REGISTRY, self._semantic_factors, abbreviated=False)}"

    @override
    def __format__(self, spec: str) -> str:
        if not self.has_designation:
            return super().__format__(spec)
        magnitude_spec, unit_spec = split_format(spec, "")
        if unit_spec not in ("", "~"):
            raise TypeError("designated quantity formatting supports only unit specs '' and '~'")
        units = format_factors(
            self._REGISTRY,
            self._semantic_factors,
            abbreviated=unit_spec == "~",
        )
        magnitude = self._REGISTRY.formatter.get_formatter("").format_magnitude(
            self._magnitude, magnitude_spec
        )
        return f"{magnitude} {units}"

    @override
    def __repr__(self) -> str:
        result = super().__repr__()
        if self.has_designation:
            unit_text = format_factors(self._REGISTRY, self._semantic_factors, abbreviated=False)
            result = result.removesuffix(f"'{self._units}')>") + f"{unit_text!r})>"
        if self._kind is not None:
            result = result.removesuffix(")>") + f", kind={self._kind!r})>"
        return result

    @override
    def _repr_html_(self) -> str:
        if not self.has_designation:
            return super()._repr_html_()
        return html.escape(str(self))

    @override
    def _repr_latex_(self) -> str:
        if not self.has_designation:
            return super()._repr_latex_()
        return _plain_latex(str(self))

    @override
    def to_tuple(self) -> tuple[Any, tuple[tuple[str, ...]]]:
        if self._kind is not None or self.has_designation:
            raise TypeError(
                "Pint's plain quantity tuple schema cannot represent kind or factor-designation "
                "metadata; use pickle or explicitly remove semantic metadata first."
            )
        return super().to_tuple()

    @override
    def plus_minus(self, error: Any, relative: bool = False) -> Any:
        """Construct an ordinary Pint measurement without classified metadata.

        Requires Pint's optional ``uncertainties`` dependency. Kinds,
        designations, and Absolute/Delta temperature roles cannot be represented
        by Measurement. Remove them explicitly before calling this adapter.

        Raises:
            TypeError: If the value or error carries classified metadata.
            ValueError: If the error belongs to another registry.
        """
        from ._adapters import _require_plain

        _require_plain(self._REGISTRY, self, "plus_minus")
        _require_plain(self._REGISTRY, error, "plus_minus")
        return super().plus_minus(error, relative)  # type: ignore[no-untyped-call]

    @override
    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Apply semantic rules, delegating ordinary quantities directly to Pint."""

        name = ufunc.__name__
        if kwargs.get("out") is not None:
            raise TypeError(
                f"NumPy ufunc {name!r} with an output target is rejected because Pint "
                "0.25.3 can recurse or write through a target with incompatible units."
            )
        if not self._operands_have_semantics(inputs, kwargs):
            return super().__array_ufunc__(  # type: ignore[no-untyped-call]
                ufunc, method, *inputs, **kwargs
            )
        if method != "__call__":
            raise TypeError(
                f"NumPy ufunc method {method!r} has no xpint semantic propagation rule yet."
            )
        if kwargs:
            raise TypeError(f"NumPy ufunc {name!r} options do not yet have an xpint semantic rule.")
        if name in _NUMPY_UFUNC_BINARY_METHODS:
            left, right = inputs[:2]
            direct, reflected = _NUMPY_UFUNC_BINARY_METHODS[name]
            if isinstance(left, SemanticQuantity):
                return getattr(left, direct)(right)
            if isinstance(right, SemanticQuantity):
                return getattr(right, reflected)(left)
            return NotImplemented
        if name in _NUMPY_UFUNC_POWERS:
            value = inputs[0]
            # Power defines the metadata; the ufunc retains its numerical domain
            # and dtype rules, including real cube roots of negative magnitudes.
            result = value._result(1, value._units, value._kind) ** _NUMPY_UFUNC_POWERS[name]
            result._magnitude = ufunc(value._magnitude)
            return result
        if name == "absolute":
            return abs(inputs[0])
        if name == "negative":
            return operator.neg(inputs[0])
        if name == "positive":
            return operator.pos(inputs[0])
        if name in _NUMPY_UFUNC_MAGNITUDE_PREDICATES:
            return super().__array_ufunc__(  # type: ignore[no-untyped-call]
                ufunc, method, *inputs, **kwargs
            )
        if name in _ANGLE_UFUNCS:
            return self._apply_angle_ufunc(ufunc, method, inputs, kwargs)
        raise TypeError(
            f"NumPy ufunc {name!r} has no xpint semantic propagation rule yet; "
            "use as_unkinded() or operate on magnitude explicitly."
        )

    def _apply_angle_ufunc(
        self,
        ufunc: Any,
        method: str,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Delegate trigonometry to Pint after discarding kinds.

        A trigonometric result is a ratio, and an inverse result is an angle, so
        no input kind describes the output and every kind is dropped rather than
        propagated. Designations are rejected instead: a designated dimensionless
        ratio records how a value was built, which does not make it an angle, and
        silently discarding the label would lose the distinction the designation
        exists to keep. Pint's dimensionality check rejects non-angle unary
        arguments. Binary arctan2 accepts matching dimensions, so xpint must also
        reject absolute temperatures whose ratio depends on the scale's origin.
        """

        if ufunc.__name__ == "arctan2":
            left, right = inputs[:2]
            semantic_input = left if isinstance(left, SemanticQuantity) else right
            if isinstance(semantic_input, SemanticQuantity):
                other = right if semantic_input is left else left
                semantic_other = semantic_input._require_compatible_quantity(other)
                if semantic_other is None:
                    raise TypeError(
                        "cannot apply NumPy ufunc 'arctan2' to a classified quantity and "
                        "a plain value; both operands must carry matching xpint semantic "
                        "state."
                    )
                semantic_input._require_same_semantic_state(
                    semantic_other,
                    operation="apply NumPy ufunc 'arctan2' to",
                )
                if semantic_input.affine_role is AffineRole.ABSOLUTE:
                    raise TypeError(
                        "NumPy ufunc 'arctan2' is undefined for absolute temperatures; "
                        f"{_ABSOLUTE_LINEAR_ESCAPE}."
                    )

        plain = []
        for value in inputs:
            if isinstance(value, SemanticQuantity):
                if value.has_designation:
                    raise TypeError(
                        f"NumPy ufunc {ufunc.__name__!r} rejects a quantity with retained "
                        "factor designations because a designated ratio is not an angle; "
                        "remove the designations explicitly with as_undesignated() first."
                    )
                value = value.as_unkinded()
            plain.append(value)
        return super().__array_ufunc__(  # type: ignore[no-untyped-call]
            ufunc, method, *plain, **kwargs
        )

    @override
    def __array_function__(
        self,
        func: Any,
        types: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        """Apply semantic rules, delegating ordinary quantities directly to Pint."""

        name = func.__name__
        if name in (_NUMPY_FUNCTION_REDUCTIONS | _NUMPY_MAGNITUDE_INDICES) and (
            len(args) != 1 or "a" in kwargs
        ):
            bound = _reduction_signature(name).bind(*args, **kwargs)
            kwargs = dict(bound.arguments)
            args = (kwargs.pop("a"),)
        if kwargs.get("out") is not None:
            raise TypeError(
                f"NumPy function {name!r} with an output target is rejected because Pint "
                "does not validate or preserve the target quantity's units."
            )
        has_semantics = self._operands_have_semantics(args, kwargs)
        if not has_semantics and name not in _NUMPY_FUNCTION_REDUCTIONS:
            return super().__array_function__(func, types, args, kwargs)  # type: ignore[no-untyped-call]
        if (
            name in (_NUMPY_FUNCTION_ADDITIVE | _NUMPY_STATISTICS)
            and self.affine_role is AffineRole.ABSOLUTE
        ):
            raise TypeError(f"NumPy function {name!r} is undefined for absolute temperatures.")
        if name in _NUMPY_FUNCTION_REDUCTIONS:
            call_kwargs, result_kind = self._prepare_reduction_kwargs(name, kwargs, method=False)
            result = super().__array_function__(  # type: ignore[no-untyped-call]
                func, types, args, call_kwargs
            )
            if isinstance(result, SemanticQuantity):
                return self._set_result_semantics(
                    result,
                    kind=None if name in _NUMPY_VARIANCES else result_kind,
                    semantic_factors=(
                        power_factors(self._semantic_factors, 2)
                        if name in _NUMPY_VARIANCES
                        else self._semantic_factors
                    ),
                )
            return result
        if name in _NUMPY_MAGNITUDE_INDICES:
            return super().__array_function__(func, types, args, kwargs)  # type: ignore[no-untyped-call]
        if name == "diff":
            bound = _reduction_signature(name).bind(*args, **kwargs)
            value = bound.arguments["a"]
            if not isinstance(value, SemanticQuantity):
                return NotImplemented
            if bound.arguments.get("n", 1) == 0:
                return value
            for edge in ("prepend", "append"):
                other = bound.arguments.get(edge)
                if isinstance(other, pint.registry.Quantity):
                    semantic_other = value._require_compatible_quantity(other)
                    assert semantic_other is not None
                    value._require_same_semantic_state(semantic_other, operation=f"{edge} to diff")
                    bound.arguments[edge] = semantic_other.to(value.units).magnitude
            call_kwargs = dict(bound.arguments)
            call_kwargs.pop("a")
            result = super().__array_function__(func, types, (value,), call_kwargs)  # type: ignore[no-untyped-call]
            return value._set_result_semantics(
                result,
                kind="TemperatureDifference"
                if value.affine_role is AffineRole.ABSOLUTE
                else value._kind,
                semantic_factors=canonical_factors(
                    value._REGISTRY, result._units, value._semantic_factors
                ),
            )
        if name in _NUMPY_FUNCTION_LIKE_CONSTRUCTORS:
            result = super().__array_function__(func, types, args, kwargs)  # type: ignore[no-untyped-call]
            if isinstance(result, SemanticQuantity):
                return self._set_result_semantics(
                    result,
                    kind=self._kind,
                    semantic_factors=self._semantic_factors,
                )
            return result
        if name in _NUMPY_FUNCTION_COMPOSITION:
            quantities = [
                value
                for value in self._semantic_values((args, kwargs))
                if isinstance(value, SemanticQuantity)
            ]
            for quantity in quantities:
                self._require_compatible_quantity(quantity)
                self._require_same_semantic_state(quantity, operation=name)
            result = super().__array_function__(func, types, args, kwargs)  # type: ignore[no-untyped-call]
            if isinstance(result, SemanticQuantity):
                return self._set_result_semantics(
                    result,
                    kind=self._kind,
                    semantic_factors=self._semantic_factors,
                )
            return result
        if name == "result_type":
            return super().__array_function__(func, types, args, kwargs)  # type: ignore[no-untyped-call]
        raise TypeError(
            f"NumPy function {name!r} has no xpint semantic propagation rule yet; "
            "use as_unkinded() or operate on magnitude explicitly."
        )

    @override
    def _dask_finalize(self, results: Any, func: Any, args: Any, units: Any) -> Any:
        result = super()._dask_finalize(results, func, args, units)  # type: ignore[no-untyped-call]
        if isinstance(result, SemanticQuantity):
            return self._set_result_semantics(
                result,
                kind=self._kind,
                semantic_factors=self._semantic_factors,
            )
        return result

    @override
    def __dask_tokenize__(self) -> Any:
        return (
            *super().__dask_tokenize__(),  # type: ignore[no-untyped-call]
            self._kind,
            self.designation_signature,
        )

    def rechunk(self, *args: Any, **kwargs: Any) -> Any:
        """Rechunk a Dask-backed magnitude without dropping semantic metadata."""

        try:
            magnitude = self._magnitude.rechunk(*args, **kwargs)
        except AttributeError:
            raise AttributeError(
                f"Magnitude '{type(self._magnitude).__name__}' does not support rechunk."
            ) from None
        if not self._is_semantically_classified():
            return magnitude
        return self._result(magnitude, self._units, self._kind)

    def map_blocks(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate plain Dask block mapping and fail closed for classified data."""

        if self._is_semantically_classified():
            raise TypeError(
                "Dask map_blocks can change values or dimensions without an xpint semantic "
                "propagation rule; operate on magnitude and reclassify explicitly."
            )
        try:
            return self._magnitude.map_blocks(*args, **kwargs)
        except AttributeError:
            raise AttributeError(
                f"Magnitude '{type(self._magnitude).__name__}' does not support map_blocks."
            ) from None


class SemanticUnit(pint.UnitRegistry.Unit):
    """A Pint unit carrying xpint factor-designation metadata.

    Units remain owned by one :class:`SemanticRegistry`. Designated factors preserve
    their semantic identity through supported unit algebra, while absolute-temperature
    units reject compound expressions and scaling operations that have no affine
    meaning. Multiplying an absolute unit by a number or array constructs an
    absolute quantity, preserving the magnitude's shape and dtype. Quantity and
    unit operands raise TypeError; use quantity conversion explicitly instead.
    """

    _semantic_factors: _SemanticFactors

    def __init__(
        self,
        units: Any,
    ) -> None:
        registry = cast(SemanticRegistry, self._REGISTRY)
        registry._require_owned(units, name="units")
        super().__init__(units)
        inherited_factors = getattr(units, "_semantic_factors", None)
        if inherited_factors is not None:
            self._semantic_factors = inherited_factors
        else:
            self._semantic_factors = registry._factors_for_units(self._units)

    @classmethod
    def _from_parts(cls, units: Any, *, semantic_factors: _SemanticFactors | None) -> SemanticUnit:
        # Internal constructor: replaces the factors derived from units, keeping
        # the private factor expression out of the public constructor signature.
        unit = cls(units)
        if semantic_factors is not None:
            unit._semantic_factors = semantic_factors
        return unit

    @property
    def designation_signature(self) -> DesignationSignature:
        """Return the immutable canonical factor-designation signature."""

        return designation_signature(self._semantic_factors)

    def _semantic_state(self) -> tuple[DesignationSignature, str | None, AffineRole | None]:
        registry = cast(SemanticRegistry, self._REGISTRY)
        role = registry._affine_role(self, None)
        if role is AffineRole.ABSOLUTE:
            kind = "Temperature"
        elif role is AffineRole.DELTA:
            kind = "TemperatureDifference"
        else:
            kind = None
        return self.designation_signature, kind, role

    @property
    def has_designation(self) -> bool:
        """Return whether a designation contribution remains after cancellation."""

        return bool(self.designation_signature)

    @property
    def is_semantic_unity(self) -> bool:
        """Return whether physical dimensions and designation residue are both absent."""

        return self.dimensionless and not self.has_designation

    def compatible_kinds(self) -> tuple[str, ...]:
        """Return registered kinds compatible with this unit's dimensions.

        Factor designations are ignored. Absolute temperature units exclude
        TemperatureDifference, delta units exclude Temperature, and generic kelvin
        lists both. This query leaves the unit unchanged.

        Returns:
            Kind names sorted alphabetically, or an empty tuple if none match.
        """

        return cast(SemanticRegistry, self._REGISTRY).compatible_kinds(self)

    def with_designation(self, designation: str) -> SemanticUnit:
        """Return a copy with a designation on its one eligible semantic factor.

        ``designation`` is opaque and case-sensitive. Ordinary factors must have
        exactly one physical dimension. A registry may declare a unique semantic
        slot for a derived unit such as millimolar.
        """

        if not isinstance(designation, str):
            raise TypeError("designation must be a string.")
        if not designation:
            raise ValueError("designation must not be empty.")
        if self._absolute_involved():
            raise ValueError("absolute temperature units cannot carry factor designations.")
        if len(self._semantic_factors) != 1:
            raise ValueError(
                "with_designation requires one unit factor; designate eligible factors "
                "before building a compound expression."
            )

        factor = self._semantic_factors[0]
        slot = cast(SemanticRegistry, self._REGISTRY)._designation_slot_for(self)
        if slot is None:
            dimensionality = self.dimensionality
            if len(dimensionality) != 1:
                raise ValueError(
                    f"cannot designate {self!s}: it has {len(dimensionality)} physical "
                    "dimensions and no declared designation slot."
                )
            dimension, total_power = next(iter(dimensionality.items()))
            dimension_power = (
                total_power if isinstance(total_power, Fraction) else Fraction(str(total_power))
            )
            slot = (str(dimension), dimension_power / factor.exponent)
        dimension, designation_power = slot
        designated = designated_factor(
            factor,
            designation,
            dimension=dimension,
            designation_power=designation_power,
        )
        return self.__class__._from_parts(self._units, semantic_factors=(designated,))

    def as_undesignated(self) -> SemanticUnit:
        """Return a physically equivalent unit with all designations removed."""

        return self.__class__._from_parts(
            self._units,
            semantic_factors=undesignated_factors(self._semantic_factors),
        )

    def __getitem__(self, designation: str) -> SemanticUnit:
        """Apply concise factor-designation syntax, such as ``mmol["O2"]``."""

        return self.with_designation(designation)

    @override
    def __copy__(self) -> SemanticUnit:
        return self.__class__._from_parts(self._units, semantic_factors=self._semantic_factors)

    @override
    def __deepcopy__(self, memo: dict[int, Any]) -> SemanticUnit:
        return self.__class__._from_parts(
            copy.deepcopy(self._units, memo),
            semantic_factors=copy.deepcopy(self._semantic_factors, memo),
        )

    @override
    def __reduce__(self) -> Any:
        registry = cast(SemanticRegistry, self._REGISTRY)
        return _unpickle_semantic_unit, (
            self._units,
            self._semantic_factors,
            registry.angle_dimension,
        )

    @override
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SemanticUnit):
            if self._REGISTRY is not other._REGISTRY:
                return False
            if self.designation_signature != other.designation_signature:
                return False
        elif isinstance(other, SemanticQuantity):
            if self._REGISTRY is not other._REGISTRY:
                return False
            if self.designation_signature != other.designation_signature:
                return False
            return cast(bool, other == self._REGISTRY.Quantity(1, self))
        elif self.has_designation:
            return False
        return super().__eq__(other)

    @override
    def __hash__(self) -> int:
        return hash(self._REGISTRY.Quantity(1, self))

    @override
    def is_compatible_with(self, other: Any, *contexts: Any, **ctx_kwargs: Any) -> bool:
        """Return whether another value has matching physical and semantic state."""

        return bool(
            self._REGISTRY.Quantity(1, self).is_compatible_with(other, *contexts, **ctx_kwargs)
        )

    @override
    def __str__(self) -> str:
        if not self.has_designation:
            return super().__str__()
        return format_factors(self._REGISTRY, self._semantic_factors, abbreviated=False)

    @override
    def __format__(self, spec: str) -> str:
        if not self.has_designation:
            return super().__format__(spec)
        _, spec = split_format(spec, "")
        if spec not in ("", "~"):
            raise TypeError("designated unit formatting supports only unit specs '' and '~'")
        return format_factors(
            self._REGISTRY,
            self._semantic_factors,
            abbreviated=spec == "~",
        )

    @override
    def __repr__(self) -> str:
        if not self.has_designation:
            return super().__repr__()
        return f"<Unit({str(self)!r})>"

    @override
    def _repr_html_(self) -> str:
        if not self.has_designation:
            return super()._repr_html_()
        return html.escape(str(self))

    @override
    def _repr_latex_(self) -> str:
        if not self.has_designation:
            return super()._repr_latex_()
        return _plain_latex(str(self))

    @override
    def __float__(self) -> float:
        return float(self._REGISTRY.Quantity(1, self))

    @override
    def __int__(self) -> int:
        return int(self._REGISTRY.Quantity(1, self))

    @override
    def __complex__(self) -> complex:
        return complex(self._REGISTRY.Quantity(1, self))

    def _absolute_involved(self, other: Any = None) -> bool:
        registry = cast(SemanticRegistry, self._REGISTRY)
        if registry._is_absolute_unit(self):
            return True
        return isinstance(other, SemanticUnit) and registry._is_absolute_unit(other)

    @override
    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        if method != "__call__":
            return NotImplemented
        if ufunc.__name__ not in ("true_divide", "divide", "floor_divide", "multiply"):
            return NotImplemented
        if ufunc.__name__ == "multiply" and self._absolute_involved():
            if kwargs:
                raise TypeError("absolute-temperature array construction does not accept options.")
            other = inputs[1] if inputs[0] is self else inputs[0]
            if isinstance(other, pint.registry.Unit):
                raise TypeError("absolute temperature units cannot form compound units.")
            if isinstance(other, pint.registry.Quantity):
                raise TypeError(
                    "an absolute temperature unit cannot scale a quantity; multiply "
                    "a magnitude by the unit to construct one."
                )
            return self._REGISTRY.Quantity(other, self)
        semantic_inputs = tuple(
            self._REGISTRY.Quantity(1, self) if value is self else value for value in inputs
        )
        return ufunc(*semantic_inputs, **kwargs)

    @override
    def __mul__(self, other: Any) -> Any:
        if self._absolute_involved(other):
            if isinstance(other, pint.registry.Unit):
                raise TypeError("absolute temperature units cannot form compound units.")
            if isinstance(other, pint.registry.Quantity):
                raise TypeError(
                    "an absolute temperature unit cannot scale a quantity; multiply "
                    "a magnitude by the unit to construct one."
                )
            return self._REGISTRY.Quantity(other, self)
        if isinstance(other, SemanticUnit):
            if self._REGISTRY is not other._REGISTRY:
                raise ValueError(
                    "xpint units belong to different SemanticRegistry instances; "
                    "use one registry for the entire expression."
                )
            result = super().__mul__(other)  # type: ignore[no-untyped-call]
            result._semantic_factors = combine_factors(
                self._semantic_factors,
                other._semantic_factors,
            )
            return result
        return self._REGISTRY.Quantity(1, self) * other

    __rmul__ = __mul__

    @override
    def __truediv__(self, other: Any) -> Any:
        if self._absolute_involved(other):
            if not isinstance(other, SemanticUnit):
                raise TypeError(
                    "absolute temperature units cannot be divided by a value; multiply "
                    "the value by the unit to construct an absolute quantity."
                )
            raise TypeError("absolute temperature units cannot form compound units.")
        if isinstance(other, SemanticUnit):
            if self._REGISTRY is not other._REGISTRY:
                raise ValueError(
                    "xpint units belong to different SemanticRegistry instances; "
                    "use one registry for the entire expression."
                )
            result = super().__truediv__(other)  # type: ignore[no-untyped-call]
            result._semantic_factors = combine_factors(
                self._semantic_factors,
                other._semantic_factors,
                divide=True,
            )
            return result
        return self._REGISTRY.Quantity(1, self) / other

    @override
    def __rtruediv__(self, other: Any) -> Any:
        if self._absolute_involved():
            raise TypeError("absolute temperature units cannot form compound units.")
        result = super().__rtruediv__(other)  # type: ignore[no-untyped-call]
        if isinstance(result, SemanticQuantity):
            result._semantic_factors = power_factors(self._semantic_factors, -1)
        return result

    @override
    def __pow__(self, other: Any) -> SemanticUnit:
        if self._absolute_involved():
            raise TypeError("absolute temperature units cannot be exponentiated.")
        result = cast(SemanticUnit, super().__pow__(other))
        result._semantic_factors = power_factors(self._semantic_factors, other)
        return result


class SemanticRegistry(pint.registry.GenericUnitRegistry[SemanticQuantity, SemanticUnit]):
    """Own Pint definitions together with xpint semantic policy.

    Each registry owns dynamically bound quantity and unit classes. Values from
    different registries are rejected rather than silently combining definitions.
    Constructing a registry does not alter Pint's process-wide application registry.
    """

    Quantity: TypeAlias = SemanticQuantity
    Unit: TypeAlias = SemanticUnit
    Measurement = _SemanticMeasurement

    def __init__(self, *args: Any, angle_dimension: bool = False, **kwargs: Any) -> None:
        """Own Pint definitions together with xpint semantic policy.

        Args:
            angle_dimension: Make plane angle a base dimension, so that torque is
                not energy, angular velocity is not frequency, and a steradian is
                not interchangeable with a bare ratio. Solid angle follows as
                ``rad ** 2`` rather than being independent. Off by default, and settable
                only here: the definitions are injected before the kind catalog is
                seeded, and Pint caches dimensionality, so there is no runtime
                toggle. Values from an angle registry and an ordinary one are
                rejected against each other like any other cross-registry pair.

        Raises:
            TypeError: If ``angle_dimension`` is not a boolean.
        """

        if not isinstance(angle_dimension, bool):
            angle_type = type(angle_dimension)
            type_name = angle_type.__name__
            if angle_type.__module__ != "builtins":
                type_name = f"{angle_type.__module__}.{type_name}"
            raise TypeError(f"angle_dimension must be a bool, got {type_name}.")
        self._angle_dimension = angle_dimension
        self._angular_dimensions: frozenset[str] = frozenset()
        self._angular_units: MappingProxyType[str, str] = MappingProxyType({})
        self._kind_definitions: dict[str, KindDefinition] = {}
        self._kind_dimensionalities: dict[str, UnitsContainer] = {}
        self._kind_names_by_dimensionality: dict[UnitsContainer, tuple[str, ...]] = {}
        self._designation_slots: dict[str, tuple[str, Fraction]] = {}
        self._factor_cache: dict[UnitsContainer, _SemanticFactors] = {}
        self._last_factor_entry: tuple[UnitsContainer, _SemanticFactors] | None = None
        self._temperature_dimensionality = UnitsContainer()
        self._intrinsic_affine_roles: dict[UnitsContainer, AffineRole | None] = {}
        super().__init__(*args, **kwargs)
        formatter = _SemanticFormatter(cast(pint.UnitRegistry[Any], self))
        formatter.locale = self.formatter.locale
        self.formatter = formatter

    @override
    def _after_init(self) -> None:
        super()._after_init()
        self.define("delta_degree_Rankine = degree_Rankine = delta_degR")
        kinds: Any = _DEFAULT_KINDS
        if self._angle_dimension:
            # Must precede define_kind: each kind freezes its dimensionality when
            # defined, so kinds seeded before the redefinition would be stranded.
            # Pint logs a redefinition warning per overridden unit. These overrides are
            # the switch working as designed, so silence them for this call only rather
            # than changing the caller's on_redefinition policy for their own defines.
            policy = self._on_redefinition
            self._on_redefinition = "ignore"
            try:
                self.load_definitions(list(_ANGLE_DEFINITIONS))
            finally:
                self._on_redefinition = policy
            self._build_cache()
            self._angular_dimensions = frozenset(_ANGULAR_UNITS)
            self._angular_units = _ANGULAR_UNITS
            kinds = {**_DEFAULT_KINDS, **_ANGLE_KIND_OVERRIDES}
        for name, reference_unit in kinds.items():
            self.define_kind(name, reference_unit)
        self.define_designation_slot("molar", "[substance]")
        self.define_designation_slot("millimolar", "[substance]")
        self._temperature_dimensionality = self.get_dimensionality("kelvin")
        self._intrinsic_affine_roles.clear()

    @override
    def _add_unit(self, definition: UnitDefinition) -> None:
        try:
            super()._add_unit(definition)
        finally:
            # define/load_definitions can change existing unit roles. Lazy
            # prefixes bypass this hook but introduce only new containers.
            self._intrinsic_affine_roles.clear()

    @override
    def _switch_context_cache_and_units(self) -> None:
        try:
            super()._switch_context_cache_and_units()
        finally:
            self._intrinsic_affine_roles.clear()

    @property
    def angle_dimension(self) -> bool:
        """Whether plane angle is a base dimension in this registry.

        Solid angle is then derived from it, with ``steradian`` equal to ``radian ** 2``.
        """

        return self._angle_dimension

    @property
    def kinds(self) -> MappingProxyType[str, KindDefinition]:
        """Return the registered kinds as a read-only mapping."""

        return MappingProxyType(self._kind_definitions)

    def compatible_kinds(self, target: str | SemanticUnit | SemanticQuantity) -> tuple[str, ...]:
        """Return registered kinds with the target's dimensionality.

        Factor designations and a quantity's current kind are ignored. Absolute
        temperature units exclude TemperatureDifference, delta units exclude
        Temperature, and generic kelvin lists both. This query does not classify
        or otherwise modify the target.

        Args:
            target: Unit expression string, or a semantic unit or quantity owned
                by this registry.

        Returns:
            Kind names sorted alphabetically, or an empty tuple if none match.
            Use ``kinds[name]`` to inspect a definition or ``as_kind(name)`` to
            classify a quantity explicitly.

        Raises:
            TypeError: If the target is an ordinary Pint object or has an
                unsupported type.
            ValueError: If the target belongs to another semantic registry, or
                the unit string carries a scaling factor, as Pint raises.
            pint.errors.UndefinedUnitError: If the unit string is undefined.
        """

        if isinstance(target, str):
            target = self.Unit(target)
        elif isinstance(target, (SemanticUnit, SemanticQuantity)):
            if cast(SemanticRegistry, target._REGISTRY) is not self:
                raise ValueError(
                    "target belongs to another registry; use this registry's unit "
                    "or quantity, or pass a unit expression string"
                )
        else:
            raise TypeError(
                "target must be a unit expression string or a SemanticUnit or "
                "SemanticQuantity owned by this registry"
            )

        names = self._kind_names_by_dimensionality.get(target.dimensionality, ())
        if self._is_absolute_unit(target):
            return tuple(name for name in names if name != "TemperatureDifference")
        if self._is_delta_unit(target):
            return tuple(name for name in names if name != "Temperature")
        return names

    def define_kind(self, name: str, reference_unit: str) -> KindDefinition:
        """Register a kind constrained to the dimensions of ``reference_unit``.

        Args:
            name: Non-empty stable semantic identifier.
            reference_unit: Pint unit expression supplying the required dimensions.

        Returns:
            The registered immutable definition.

        Raises:
            TypeError: If either argument is not a string.
            ValueError: If either string is empty, an existing definition conflicts,
                or a new kind is constrained to thermodynamic temperature. Affine
                roles are carried by the built-in ``Temperature`` and
                ``TemperatureDifference`` kinds, which other names cannot refine.
        """

        if not isinstance(name, str) or not isinstance(reference_unit, str):
            raise TypeError("kind name and reference_unit must both be strings.")
        name = name.strip()
        reference_unit = reference_unit.strip()
        if not name or not reference_unit:
            raise ValueError("kind name and reference_unit must not be empty.")

        definition = KindDefinition(name, reference_unit)
        existing = self._kind_definitions.get(name)
        if existing is not None:
            if self._kind_dimensionalities[name] != self.Unit(reference_unit).dimensionality:
                raise ValueError(
                    f"kind {name!r} is already constrained to {existing.reference_unit!r}."
                )
            return existing
        dimensionality = self.Unit(reference_unit).dimensionality
        if name not in _AFFINE_KINDS and dimensionality == self.get_dimensionality("kelvin"):
            # Affine roles are encoded through the two built-in temperature kinds, so
            # Absolute/Delta normalization would silently replace any other name.
            raise ValueError(
                f"kind {name!r} cannot be constrained to thermodynamic temperature; "
                "use the built-in 'Temperature' or 'TemperatureDifference' kind."
            )
        self._kind_definitions[name] = definition
        self._kind_dimensionalities[name] = dimensionality
        names = self._kind_names_by_dimensionality.get(dimensionality, ())
        self._kind_names_by_dimensionality[dimensionality] = tuple(sorted((*names, name)))
        return definition

    def define_designation_slot(
        self,
        unit: str,
        dimension: str,
        *,
        exponent: int | Fraction = Fraction(1),
    ) -> None:
        """Declare the unique factor targeted by designation on a derived unit.

        For example, millimolar targets ``[substance]`` with exponent one,
        allowing ``mM["O2"]`` to mean a concentration of O2 rather than a
        designation applied ambiguously to the whole compound dimension.

        Args:
            unit: Single Pint unit token receiving the designation syntax.
            dimension: One physical dimension present in that unit.
            exponent: Contribution of the target dimension per unit exponent.

        Raises:
            TypeError: If text arguments or ``exponent`` have invalid types.
            ValueError: If the unit is compound, the dimension is absent, or a
                conflicting slot is already registered.
        """

        if not isinstance(unit, str) or not isinstance(dimension, str):
            raise TypeError("designation slot unit and dimension must both be strings.")
        if isinstance(exponent, bool) or not isinstance(exponent, (int, Fraction)):
            raise TypeError("designation slot exponent must be an integer or Fraction.")
        exponent = Fraction(exponent)
        if not unit or not dimension or not exponent:
            raise ValueError("designation slot arguments must not be empty or zero.")

        semantic_unit = self.Unit(unit)
        if len(semantic_unit._units) != 1:
            raise ValueError("a designation slot must be declared on one Pint unit token.")
        unit_name, unit_exponent = next(iter(semantic_unit._units.items()))
        if unit_exponent != 1:
            raise ValueError("a designation slot unit must have exponent one.")
        dimensionality = semantic_unit.dimensionality
        if dimension not in dimensionality:
            raise ValueError(
                f"designation slot dimension {dimension!r} is absent from unit {unit!r}."
            )

        slot = (dimension, exponent)
        existing = self._designation_slots.get(str(unit_name))
        if existing is not None and existing != slot:
            raise ValueError(f"unit {unit!r} already has designation slot {existing!r}.")
        self._designation_slots[str(unit_name)] = slot

    def _designation_slot_for(self, unit: SemanticUnit) -> tuple[str, Fraction] | None:
        if len(unit._units) != 1:
            return None
        unit_name, exponent = next(iter(unit._units.items()))
        if exponent == 0:
            return None
        slot = self._designation_slots.get(str(unit_name))
        if slot is None:
            return None
        dimension, designation_power = slot
        return dimension, designation_power

    def _require_owned(self, value: Any, *, name: str) -> None:
        owner = getattr(value, "_REGISTRY", self)
        if owner is not self:
            raise ValueError(
                f"{name} belongs to a different Pint registry; construct it from this "
                "SemanticRegistry before combining values."
            )

    def wraps(
        self, ret: Any, args: Any, strict: bool = True
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Wrap a numerical function using Pint's unit declarations.

        Converted inputs and input-unit declarations must be unclassified,
        including implicit Absolute/Delta temperature roles. Return units may
        construct those roles but cannot carry designations. ``None`` argument
        declarations pass values through unchanged, including metadata; ordinary
        non-quantity values in non-strict mode follow Pint's pass-through behavior.
        Defaults, keyword arguments, and relative unit expressions follow Pint.
        Every value under a non-``None`` variadic declaration is guarded, as are
        keyword-only values and defaults. Use fixed parameter lists for conversion:
        Pint 0.25.3 can convert a positional slot under the wrong declaration, and
        those slots are guarded even when their own declaration is ``None``.
        On converters with corrected binding, ``None`` variadic declarations
        preserve metadata; binding behavior is checked when decorating.

        Raises:
            TypeError: If conversion would strip classified metadata.
            ValueError: If a converted value or unit belongs to another registry.
        """
        from ._adapters import _wraps

        return _wraps(self, ret, args, strict)

    def check(self, *dimensions: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorate a function with Pint's physical dimensionality checks.

        Inputs reach the function unchanged, preserving kinds and designations.
        This decorator checks dimensions only; it does not assert semantic equality.
        ``None`` skips checking the corresponding argument.

        Raises:
            ValueError: If checked inputs or declarations belong to another registry.
            DimensionalityError: If an input has incompatible physical dimensions.
        """
        from ._adapters import _check

        return _check(self, dimensions)

    def pi_theorem(self, quantities: dict[str, Any]) -> list[dict[str, Any]]:
        """Return Pint's Buckingham-pi combinations for unclassified inputs.

        Kind, designation, and affine constraints cannot be represented by this
        physical exponent mapping; remove them explicitly before using the theorem.

        Raises:
            TypeError: If an input carries classified semantics.
            ValueError: If an input belongs to another registry.
        """
        from ._adapters import _pi_theorem

        return cast(list[dict[str, Any]], _pi_theorem(self, quantities))

    def setup_matplotlib(self, enable: bool = True) -> None:
        """Enable or disable Pint plotting for this registry's ordinary quantities.

        Requires matplotlib. Converting classified values for an axis fails closed;
        remove metadata explicitly before plotting. Celsius and delta-temperature
        units carry implicit roles; convert to Kelvin and remove the role to plot
        their magnitudes. Other registries are unaffected.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        from ._adapters import _setup_matplotlib

        _setup_matplotlib(self, enable)

    @override
    def parse_expression(
        self, input_string: str, case_sensitive: bool | None = None, **values: Any
    ) -> Any:
        """Parse a Pint expression, explaining offset-temperature construction.

        Raises:
            TypeError: If parsing multiplies an absolute temperature by a magnitude;
                use ``Quantity(20, "degC")`` or ``20 * degC`` instead.
        """
        try:
            return super().parse_expression(input_string, case_sensitive, **values)
        except TypeError as exc:
            if "absolute temperatures cannot participate in multiplication or division" not in str(
                exc
            ):
                raise
            raise TypeError(
                "string parsing multiplies by the unit; construct absolute temperatures "
                'with Quantity(20, "degC") or 20 * degC instead'
            ) from exc

    __call__ = parse_expression

    @override
    def convert(
        self, value: Any, src: Any, dst: Any, inplace: bool = False, **ctx_kwargs: Any
    ) -> Any:
        """Convert magnitudes while validating semantic unit-object inputs.

        Semantic source/target objects obey quantity conversion rules. When both
        source and target are strings or containers, conversion is explicitly
        physical and skips affine/metadata policy.
        Quantity source/target magnitudes are ignored, as in Pint. A source kind
        constrains result dimensions; a target quantity supplies its units and
        designations, not a replacement kind, just as in ``Quantity.to()``.
        Shape, dtype, and in-place aliases follow Pint's magnitude contract.

        Raises:
            ValueError: If a value, source, or target belongs to another registry.
            TypeError: If semantic source and target states are incompatible, or
                value is a quantity rather than a magnitude; use its ``to()`` method.
        """
        self._require_owned(value, name="value")
        if isinstance(value, pint.registry.Quantity):
            raise TypeError("convert value must be a magnitude; use quantity.to() instead")
        if isinstance(src, UnitsContainer) and isinstance(dst, UnitsContainer):
            return super().convert(value, src, dst, inplace, **ctx_kwargs)
        self._require_owned(src, name="source units")
        self._require_owned(dst, name="target units")
        if (
            not inplace
            and isinstance(src, SemanticUnit)
            and isinstance(dst, SemanticUnit)
            and not any(
                factor.designation is not None
                for unit in (src, dst)
                for factor in unit._semantic_factors
            )
            and self._intrinsic_affine_role(src._units) is None
            and self._intrinsic_affine_role(dst._units) is None
        ):
            return super().convert(value, src._units, dst._units, inplace, **ctx_kwargs)
        if not isinstance(src, (SemanticQuantity, SemanticUnit)) and not isinstance(
            dst, (SemanticQuantity, SemanticUnit)
        ):
            return super().convert(value, src, dst, inplace, **ctx_kwargs)
        source_kind = src.kind if isinstance(src, SemanticQuantity) else None
        source_units = src.units if isinstance(src, SemanticQuantity) else src
        target_units = dst.units if isinstance(dst, SemanticQuantity) else dst
        # Validate units on a scalar, leaving magnitude coercion to Pint's
        # registry converter rather than its different Quantity constructor.
        source_container = to_units_container(source_units, cast(pint.UnitRegistry[Any], self))
        quantity = self.Quantity._from_parts(
            1,
            source_container,
            kind=source_kind,
            semantic_factors=getattr(source_units, "_semantic_factors", None),
        )
        target = quantity._conversion_target(target_units)
        kind = quantity._converted_kind()
        self._validate_kind(kind, target.dimensionality)
        self._validate_affine_kind(target._units, kind)
        if inplace and np is not None and isinstance(value, np.ndarray):
            quantity._magnitude = value
            quantity.ito(target, **ctx_kwargs)
            return quantity.magnitude
        return super().convert(value, quantity._units, target._units, inplace, **ctx_kwargs)

    def _factors_for_units(self, units: UnitsContainer) -> _SemanticFactors:
        last_entry = self._last_factor_entry
        if last_entry is not None and (units is last_entry[0] or units == last_entry[0]):
            return last_entry[1]
        try:
            factors = self._factor_cache[units]
        except KeyError:
            factors = factors_from_units(units)
            self._factor_cache[units] = factors
        self._last_factor_entry = (units, factors)
        return factors

    def _validate_kind(self, kind: str | None, dimensionality: UnitsContainer) -> None:
        if kind is None:
            return
        if not isinstance(kind, str):
            raise TypeError("kind must be a registered string or None.")
        try:
            definition = self._kind_definitions[kind]
        except KeyError as exc:
            raise KeyError(
                f"unknown quantity kind {kind!r}; register it with define_kind() first."
            ) from exc
        expected = self._kind_dimensionalities[definition.name]
        if dimensionality != expected:
            raise ValueError(
                f"kind {kind!r} requires dimensions {expected}, received {dimensionality}."
            )

    def _validate_affine_kind(self, units: Any, kind: str | None) -> None:
        if kind not in _AFFINE_KINDS:
            return
        if self._is_absolute_unit(units) and kind == "TemperatureDifference":
            raise ValueError(
                "an absolute temperature unit cannot have kind 'TemperatureDifference'; "
                "use the matching delta unit."
            )
        if self._is_delta_unit(units) and kind == "Temperature":
            raise ValueError(
                "a delta temperature unit cannot have kind 'Temperature'; "
                "use the matching absolute unit."
            )

    def _affine_role(self, units: SemanticUnit, kind: str | None) -> AffineRole | None:
        return self._affine_role_from_container(units._units, kind)

    def _affine_role_from_container(
        self,
        units: UnitsContainer,
        kind: str | None,
    ) -> AffineRole | None:
        if self._get_dimensionality(units) != self._temperature_dimensionality:
            return None
        if self._is_absolute_unit(units):
            return AffineRole.ABSOLUTE
        if self._is_delta_unit(units):
            return AffineRole.DELTA
        if kind == "Temperature":
            return AffineRole.ABSOLUTE
        if kind == "TemperatureDifference":
            return AffineRole.DELTA
        return AffineRole.GENERIC

    def _intrinsic_affine_role(self, units: UnitsContainer) -> AffineRole | None:
        try:
            return self._intrinsic_affine_roles[units]
        except KeyError:
            role = self._resolve_intrinsic_affine_role(units)
            self._intrinsic_affine_roles[units] = role
            return role

    def _resolve_intrinsic_affine_role(self, units: UnitsContainer) -> AffineRole | None:
        if len(units) != 1:
            return None
        name, exponent = next(iter(units.items()))
        if exponent != 1 or self._get_dimensionality(units) != self._temperature_dimensionality:
            return None
        if name == "degree_Rankine":
            return AffineRole.ABSOLUTE
        if name == "delta_degree_Rankine":
            return AffineRole.DELTA
        definition = self._units.get(name)
        if definition is None:
            return None
        if isinstance(definition.converter, OffsetConverter) and not definition.is_multiplicative:
            return AffineRole.ABSOLUTE
        original = (
            self._units.get(name.removeprefix("delta_")) if name.startswith("delta_") else None
        )
        if (
            original is not None
            and isinstance(original.converter, OffsetConverter)
            and not original.is_multiplicative
        ):
            return AffineRole.DELTA
        # Linear aliases and prefixes of delta units retain the difference role.
        if (
            not definition.is_base
            and definition.reference is not None
            and self._intrinsic_affine_role(definition.reference) is AffineRole.DELTA
        ):
            return AffineRole.DELTA
        return None

    def _is_absolute_unit(self, units: Any) -> bool:
        container = getattr(units, "_units", units)
        return self._intrinsic_affine_role(container) is AffineRole.ABSOLUTE

    def _is_delta_unit(self, units: Any) -> bool:
        container = getattr(units, "_units", units)
        return self._intrinsic_affine_role(container) is AffineRole.DELTA

    def _delta_units_for(self, absolute_units: SemanticUnit) -> SemanticUnit:
        if self._intrinsic_affine_role(absolute_units._units) is AffineRole.ABSOLUTE:
            name = next(iter(absolute_units._units))
            return self.Unit(self.UnitsContainer({"delta_" + name: 1}))
        # Multiplicative scales use the same unit for a temperature difference.
        return absolute_units


@cache
def get_default_registry() -> SemanticRegistry:
    """Return xpint's shared semantic registry without installing it globally."""

    return SemanticRegistry()


_application_registry: SemanticRegistry | None = None


def get_application_registry() -> SemanticRegistry:
    """Return the registry used to reconstruct xpint pickles (default: shared registry)."""
    return get_default_registry() if _application_registry is None else _application_registry


def set_application_registry(registry: SemanticRegistry) -> None:
    """Choose the xpint pickle registry without changing Pint's application registry.

    Raises:
        TypeError: If registry is not a SemanticRegistry.
    """
    if not isinstance(registry, SemanticRegistry):
        raise TypeError("application registry must be a SemanticRegistry")
    global _application_registry
    _application_registry = registry


def _pickle_registry(angle_dimension: bool) -> SemanticRegistry:
    registry = get_application_registry()
    if registry.angle_dimension != angle_dimension:
        raise ValueError(
            "pickle angle_dimension does not match the application registry; "
            f"call set_application_registry() with a SemanticRegistry(angle_dimension={angle_dimension})"
        )
    return registry


def _unpickle_semantic_quantity(
    magnitude: Any,
    units: UnitsContainer,
    kind: str | None,
    semantic_factors: _SemanticFactors,
    angle_dimension: bool,
) -> SemanticQuantity:
    """Reconstruct a quantity against xpint's application registry."""

    registry = _pickle_registry(angle_dimension)
    if kind is not None and kind not in registry.kinds:
        # Workers in a fresh process (multiprocessing, Dask distributed) start with
        # the default registry, so name the setup step rather than only the kind.
        raise KeyError(
            f"cannot unpickle a quantity of kind {kind!r}: the xpint application "
            "registry does not define it; call xpint.set_application_registry() with "
            "a registry that defines the kind before unpickling"
        )
    return registry.Quantity._from_parts(
        magnitude,
        registry.Unit(str(units)),
        kind=kind,
        semantic_factors=semantic_factors,
    )


def _plain_latex(value: str) -> str:
    """Return an injection-safe plain-text LaTeX representation."""

    replacements = {
        "\\": r"\backslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "%": r"\%",
        "_": r"\_",
        "^": r"\^{}",
        "~": r"\~{}",
        " ": r"\ ",
    }
    escaped = "".join(replacements.get(character, character) for character in value)
    return rf"$\mathtt{{{escaped}}}$"


def _unpickle_semantic_unit(
    units: UnitsContainer,
    semantic_factors: _SemanticFactors,
    angle_dimension: bool,
) -> SemanticUnit:
    """Reconstruct a unit against xpint's application registry."""

    registry = _pickle_registry(angle_dimension)
    return registry.Unit._from_parts(
        registry.Unit(str(units)),
        semantic_factors=semantic_factors,
    )


def _format_unit_power(unit: str, exponent: Any) -> str:
    """Render ``unit ** exponent`` so Pint can parse it back to the same exponent.

    Deliberately does not round through ``Fraction.limit_denominator``: that silently
    changes an exponent whose denominator exceeds the default limit, and the whole point
    of this string is that multiplying by it clears the angle exactly.
    """

    if exponent == 1:
        return unit
    rendered = int(exponent) if float(exponent).is_integer() else repr(exponent)
    return f"{unit} ** {rendered}"


__all__ = [
    "AffineRole",
    "DesignationSignature",
    "DesignationTerm",
    "KindDefinition",
    "SemanticQuantity",
    "SemanticRegistry",
    "SemanticUnit",
    "get_application_registry",
    "get_default_registry",
    "set_application_registry",
]
