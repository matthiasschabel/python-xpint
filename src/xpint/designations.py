"""Immutable factor-designation metadata and algebra."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any

from pint.util import UnitsContainer


@dataclass(frozen=True, order=True, slots=True)
class DesignationTerm:
    """One canonical contribution to a quantity's designation signature.

    Attributes:
        dimension: Pint physical-dimension name, such as ``"[substance]"``.
        designation: Opaque application-defined label.
        exponent: Rational contribution of the dimension carrying that label.
    """

    dimension: str
    designation: str
    exponent: Fraction


@dataclass(frozen=True, order=True, slots=True)
class _SemanticFactor:
    """One display factor plus its optional designation target."""

    unit_name: str
    exponent: Fraction
    designation: str | None = None
    designated_dimension: str | None = None
    designation_power: Fraction | None = None


type DesignationSignature = tuple[DesignationTerm, ...]
type _SemanticFactors = tuple[_SemanticFactor, ...]


def _as_fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    exact = Fraction(str(value))
    rationalized = exact.limit_denominator()
    if math.isclose(float(exact), float(rationalized), rel_tol=1e-12, abs_tol=0.0):
        return rationalized
    return exact


def _factor_key(factor: _SemanticFactor) -> tuple[str, str | None, str | None, Fraction | None]:
    return (
        factor.unit_name,
        factor.designation,
        factor.designated_dimension,
        factor.designation_power,
    )


def factors_from_units(units: UnitsContainer) -> _SemanticFactors:
    """Build undesignated semantic factors from Pint's structural units."""

    return tuple(
        _SemanticFactor(str(unit_name), _as_fraction(exponent))
        for unit_name, exponent in units.items()
        if exponent != 0
    )


def designation_signature(factors: _SemanticFactors) -> DesignationSignature:
    """Reduce semantic factors to their canonical designation signature."""

    contributions: dict[tuple[str, str], Fraction] = {}
    for factor in factors:
        if factor.designation is None:
            continue
        assert factor.designated_dimension is not None
        assert factor.designation_power is not None
        key = (factor.designated_dimension, factor.designation)
        contributions[key] = contributions.get(key, Fraction()) + (
            factor.exponent * factor.designation_power
        )

    return tuple(
        DesignationTerm(dimension, designation, exponent)
        for (dimension, designation), exponent in sorted(contributions.items())
        if exponent
    )


def _merge_factors(factors: _SemanticFactors) -> _SemanticFactors:
    merged: dict[tuple[str, str | None, str | None, Fraction | None], Fraction] = {}
    for factor in factors:
        key = _factor_key(factor)
        merged[key] = merged.get(key, Fraction()) + factor.exponent

    return tuple(
        _SemanticFactor(unit_name, exponent, designation, dimension, designation_power)
        for (unit_name, designation, dimension, designation_power), exponent in sorted(
            merged.items(), key=lambda item: repr(item[0])
        )
        if exponent
    )


def _strip_cancelled_designations(factors: _SemanticFactors) -> _SemanticFactors:
    contributions: dict[tuple[str, str], Fraction] = {}
    for factor in factors:
        if factor.designation is None:
            continue
        assert factor.designated_dimension is not None
        assert factor.designation_power is not None
        key = (factor.designated_dimension, factor.designation)
        contributions[key] = contributions.get(key, Fraction()) + (
            factor.exponent * factor.designation_power
        )

    cancelled = {key for key, exponent in contributions.items() if not exponent}
    if not cancelled:
        return factors

    return _merge_factors(
        tuple(
            replace(
                factor,
                designation=None,
                designated_dimension=None,
                designation_power=None,
            )
            if factor.designation is not None
            and (factor.designated_dimension, factor.designation) in cancelled
            else factor
            for factor in factors
        )
    )


def combine_factors(
    left: _SemanticFactors,
    right: _SemanticFactors,
    *,
    divide: bool = False,
) -> _SemanticFactors:
    """Multiply or divide two semantic factor expressions."""

    sign = -1 if divide else 1
    combined = left + tuple(replace(factor, exponent=sign * factor.exponent) for factor in right)
    return _strip_cancelled_designations(_merge_factors(combined))


def power_factors(factors: _SemanticFactors, exponent: Any) -> _SemanticFactors:
    """Raise a semantic factor expression to a scalar rational power."""

    power = _as_fraction(exponent)
    if not power:
        return ()
    return _strip_cancelled_designations(
        _merge_factors(
            tuple(replace(factor, exponent=factor.exponent * power) for factor in factors)
        )
    )


def undesignated_factors(factors: _SemanticFactors) -> _SemanticFactors:
    """Remove all designation labels while retaining the physical expression."""

    return _merge_factors(
        tuple(
            replace(
                factor,
                designation=None,
                designated_dimension=None,
                designation_power=None,
            )
            for factor in factors
        )
    )


def designated_factor(
    factor: _SemanticFactor,
    designation: str,
    *,
    dimension: str,
    designation_power: Any,
) -> _SemanticFactor:
    """Return ``factor`` with one explicit designation target."""

    return replace(
        factor,
        designation=designation,
        designated_dimension=dimension,
        designation_power=_as_fraction(designation_power),
    )


def _single_dimension_factor(registry: Any, unit_name: str) -> tuple[str, Fraction] | None:
    dimensionality = registry.get_dimensionality(unit_name)
    if len(dimensionality) != 1:
        return None
    dimension, exponent = next(iter(dimensionality.items()))
    return str(dimension), _as_fraction(exponent)


def canonical_factors(
    registry: Any,
    target_units: UnitsContainer,
    source_factors: _SemanticFactors,
) -> _SemanticFactors:
    """Express designations without changing the target's physical unit scale."""

    signature = designation_signature(source_factors)
    if not signature:
        return factors_from_units(target_units)

    candidates: dict[str, tuple[str, Fraction]] = {}

    for unit_name in target_units:
        candidate = _single_dimension_factor(registry, str(unit_name))
        if candidate is not None:
            dimension, unit_power = candidate
            candidates.setdefault(dimension, (str(unit_name), unit_power))

    for factor in source_factors:
        if factor.designation is None:
            continue
        assert factor.designated_dimension is not None
        if factor.designated_dimension in candidates:
            continue
        _, root_units = registry.get_root_units(factor.unit_name)
        for unit_name in root_units._units:
            candidate = _single_dimension_factor(registry, str(unit_name))
            if candidate is not None and candidate[0] == factor.designated_dimension:
                candidates[factor.designated_dimension] = (str(unit_name), candidate[1])
                break

    result: list[_SemanticFactor] = []
    residual_units = target_units
    for term in signature:
        try:
            unit_name, unit_power = candidates[term.dimension]
        except KeyError as exc:
            raise TypeError(
                f"cannot preserve designation {term.designation!r} while reducing "
                f"dimension {term.dimension}; no root factor is available."
            ) from exc
        result.append(
            designated_factor(
                _SemanticFactor(unit_name, term.exponent / unit_power),
                term.designation,
                dimension=term.dimension,
                designation_power=unit_power,
            )
        )
        residual_units = residual_units / registry.UnitsContainer(
            {unit_name: term.exponent / unit_power}
        )

    # Unit algebra preserves scale and derived residual units; dimensional
    # subtraction alone cannot distinguish, for example, metres and centimetres.
    result.extend(factors_from_units(residual_units))
    return _merge_factors(tuple(result))


def _escaped_designation(designation: str) -> str:
    return designation.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def format_factors(registry: Any, factors: _SemanticFactors, *, abbreviated: bool) -> str:
    """Format a semantic factor expression with native bracket designations."""

    if not factors:
        return "dimensionless"

    def format_factor(factor: _SemanticFactor) -> str:
        unit = registry.Unit(factor.unit_name)
        name = format(unit, "~") if abbreviated else str(unit)
        if factor.designation is not None:
            name += f"[{_escaped_designation(factor.designation)}]"
        exponent = abs(factor.exponent)
        if exponent != 1:
            exponent_text = (
                str(exponent.numerator)
                if exponent.denominator == 1
                else f"({exponent.numerator}/{exponent.denominator})"
            )
            name += f" ** {exponent_text}"
        return name

    numerator = [format_factor(factor) for factor in factors if factor.exponent > 0]
    denominator = [format_factor(factor) for factor in factors if factor.exponent < 0]
    numerator_text = " * ".join(numerator) if numerator else "1"
    if not denominator:
        return numerator_text
    denominator_text = " * ".join(denominator)
    if len(denominator) > 1:
        denominator_text = f"({denominator_text})"
    return f"{numerator_text} / {denominator_text}"


__all__ = ["DesignationSignature", "DesignationTerm"]
