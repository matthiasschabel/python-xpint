"""Strict thermodynamic-temperature role and arithmetic tests."""

import operator
from typing import Any

import numpy as np
import pint
import pytest

from xpint import AffineRole, SemanticRegistry
from xpint.units import K, degC, degF, delta_degC, delta_degF, delta_degR, m, ureg


def test_temperature_roles_come_from_unit_or_kind() -> None:
    assert (20 * degC).affine_role is AffineRole.ABSOLUTE
    assert (5 * delta_degC).affine_role is AffineRole.DELTA
    assert (300 * K).affine_role is AffineRole.GENERIC
    assert (300 * K).as_kind("Temperature").affine_role is AffineRole.ABSOLUTE
    assert (5 * K).as_kind("TemperatureDifference").affine_role is AffineRole.DELTA
    assert (5 * m).affine_role is None


@pytest.mark.parametrize("operation", [operator.mul, np.multiply])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("other", [20 * degC, 68 * degF, degC, m])
def test_absolute_unit_multiplication_rejects_quantities_and_units(
    operation: Any, reverse: bool, other: Any
) -> None:
    left, right = (other, degC) if reverse else (degC, other)
    with pytest.raises(TypeError, match="absolute temperature"):
        operation(left, right)


@pytest.mark.parametrize("operation", [operator.mul, np.multiply])
@pytest.mark.parametrize("unit_operand", [False, True])
def test_absolute_unit_rejects_plain_pint_operands(operation: Any, unit_operand: bool) -> None:
    registry: pint.UnitRegistry[Any] = pint.UnitRegistry()
    other = registry.degF if unit_operand else registry.Quantity(68, "degF")
    match = "cannot form compound" if unit_operand else r"cannot scale a quantity.*magnitude"
    with pytest.raises(TypeError, match=match):
        operation(degC, other)


@pytest.mark.parametrize("operation", [operator.mul, np.multiply])
@pytest.mark.parametrize("reverse", [False, True])
def test_absolute_unit_constructs_from_scalar_and_array_magnitudes(
    operation: Any, reverse: bool
) -> None:
    for magnitude in (2, np.array([2, 3], dtype=np.int16)):
        left, right = (magnitude, degC) if reverse else (degC, magnitude)
        result = operation(left, right)
        np.testing.assert_array_equal(result.magnitude, magnitude)
        assert result.units == degC
        assert result.affine_role is AffineRole.ABSOLUTE
        if isinstance(magnitude, np.ndarray):
            assert result.magnitude.shape == magnitude.shape
            assert result.magnitude.dtype == magnitude.dtype


def test_contradictory_affine_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="absolute temperature unit"):
        ureg.Quantity(20, degC, kind="TemperatureDifference")
    with pytest.raises(ValueError, match="delta temperature unit"):
        ureg.Quantity(5, delta_degC, kind="Temperature")


def test_absolute_plus_absolute_is_rejected() -> None:
    with pytest.raises(TypeError, match="cannot add two absolute temperatures"):
        20 * degC + 15 * degC


def test_absolute_minus_absolute_converts_scales_and_returns_delta() -> None:
    difference = 20 * degC - 15 * degC

    assert difference.magnitude == 5
    assert difference.units == delta_degC
    assert difference.kind == "TemperatureDifference"
    across_scales = 25 * degC - 68 * degF
    assert across_scales.units == delta_degC
    assert across_scales.magnitude == pytest.approx(5, rel=0.0, abs=1e-12)


def test_absolute_and_delta_arithmetic_uses_scale_only_for_delta() -> None:
    warmed = 20 * degC + 9 * delta_degF
    warmed_reversed = 9 * delta_degF + 20 * degC
    cooled = 20 * degC - 9 * delta_degF

    assert warmed.magnitude == pytest.approx(25, rel=0.0, abs=1e-12)
    assert warmed_reversed.magnitude == pytest.approx(25, rel=0.0, abs=1e-12)
    assert cooled.magnitude == pytest.approx(15, rel=0.0, abs=1e-12)
    assert warmed.units == degC
    assert warmed.kind == "Temperature"


def test_delta_arithmetic_returns_temperature_difference() -> None:
    total = 5 * delta_degC + 9 * delta_degF
    difference = 8 * delta_degC - 3 * delta_degC

    assert total.magnitude == pytest.approx(10, rel=0.0, abs=1e-12)
    assert difference.magnitude == 5
    assert total.kind == "TemperatureDifference"
    assert difference.kind == "TemperatureDifference"


def test_delta_minus_absolute_is_rejected() -> None:
    with pytest.raises(TypeError, match="cannot subtract an absolute temperature"):
        5 * delta_degC - 20 * degC


def test_classified_and_generic_temperature_arithmetic_is_rejected() -> None:
    with pytest.raises(TypeError, match="Generic Kelvin"):
        20 * degC + 5 * K
    with pytest.raises(TypeError, match="Generic Kelvin"):
        (300 * K).as_kind("Temperature") - 5 * K


def test_generic_kelvin_remains_linear() -> None:
    result = 5 * K + 3 * K

    assert result.magnitude == 8
    assert result.kind is None
    assert result.affine_role is AffineRole.GENERIC


def test_affine_conversion_requires_explicit_role_and_preserves_it() -> None:
    kelvin_absolute = (20 * degC).to(K)
    celsius_absolute = (293.15 * K).as_kind("Temperature").to(degC)
    kelvin_delta = (5 * delta_degC).to(K)

    assert kelvin_absolute.magnitude == pytest.approx(293.15, rel=0.0, abs=1e-12)
    assert kelvin_absolute.kind == "Temperature"
    assert celsius_absolute.magnitude == pytest.approx(20, rel=0.0, abs=1e-12)
    assert celsius_absolute.kind == "Temperature"
    assert kelvin_delta.magnitude == pytest.approx(5, rel=0.0, abs=1e-12)
    assert kelvin_delta.kind == "TemperatureDifference"

    with pytest.raises(TypeError, match="Generic Kelvin"):
        (293.15 * K).to(degC)
    with pytest.raises(TypeError, match=r"Delta.*Absolute"):
        (5 * delta_degC).to(degC)


def test_absolute_temperature_cannot_be_multiplied_or_compounded() -> None:
    absolute = 20 * degC

    assert degC * 20 == absolute
    with pytest.raises(TypeError, match="cannot participate"):
        absolute * 2
    with pytest.raises(TypeError, match="cannot participate"):
        2 / absolute
    with pytest.raises(TypeError, match="cannot form compound"):
        degC * m
    with pytest.raises(TypeError, match="cannot form compound"):
        m / degC
    with pytest.raises(TypeError, match="cannot be exponentiated"):
        degC**0
    with pytest.raises(TypeError, match="cannot be divided by a value"):
        degC / 2


def test_absolute_temperature_rejects_origin_dependent_unary_operations_and_powers() -> None:
    absolute = 20 * degC
    kelvin_absolute = absolute.to(K)

    with pytest.raises(TypeError, match="in kelvin, call as_unkinded"):
        abs(absolute)
    with pytest.raises(TypeError, match="in kelvin, call as_unkinded"):
        _ = -absolute
    with pytest.raises(TypeError, match="in kelvin, call as_unkinded"):
        absolute**0
    with pytest.raises(TypeError, match="in kelvin, call as_unkinded"):
        kelvin_absolute**1

    assert (+absolute).affine_role is AffineRole.ABSOLUTE
    linear_power = kelvin_absolute.as_unkinded() ** 4
    assert linear_power.units == K**4


def test_temperature_difference_keeps_linear_unary_operations() -> None:
    difference = -5 * delta_degC

    absolute_value = abs(difference)
    negated = -difference

    assert absolute_value.magnitude == pytest.approx(5, rel=0.0, abs=1e-12)
    assert negated.magnitude == pytest.approx(5, rel=0.0, abs=1e-12)
    assert absolute_value.affine_role is AffineRole.DELTA
    assert negated.affine_role is AffineRole.DELTA


def test_rankine_has_explicit_delta_partner() -> None:
    difference = 5 * delta_degR

    assert difference.affine_role is AffineRole.DELTA


def test_absolute_temperature_comparison_applies_offsets() -> None:
    absolute = 20 * degC
    converted = absolute.to(degF)
    converted_kelvin = absolute.to(K)
    converted_delta = (5 * delta_degC).to(delta_degF)

    assert converted.magnitude == pytest.approx(68, rel=0.0, abs=1e-12)
    assert absolute == converted
    assert absolute == converted_kelvin
    assert 5 * delta_degC == converted_delta
    assert 30 * degC > (25 * degC).to(degF)


def test_zero_absolute_temperatures_in_different_scales_are_not_equal() -> None:
    # Pint equates any two zero magnitudes of equal dimensionality before
    # applying offsets; 0 degC is 273.15 K and 0 degF is about 255.37 K.
    celsius = 0.0 * degC
    fahrenheit = 0.0 * degF

    assert celsius != fahrenheit
    assert fahrenheit != celsius
    assert hash(celsius) != hash(fahrenheit)
    assert celsius == (0.0 * degC).to(K)
    assert hash(celsius) == hash((0.0 * degC).to(degC))


def test_zero_absolute_temperature_arrays_compare_after_offsets() -> None:
    celsius = ureg.Quantity(np.zeros(2), degC)
    # Equality is exact, so the second element is the converted freezing point
    # rather than a literal 32 that differs in the last bits.
    freezing = (0.0 * degC).to(degF).magnitude
    fahrenheit = ureg.Quantity(np.array([0.0, freezing]), degF)

    np.testing.assert_array_equal(celsius == fahrenheit, [False, True])
    np.testing.assert_array_equal(celsius != fahrenheit, [True, False])
    assert celsius.shape == (2,)


def test_offset_equality_is_symmetric_and_agrees_with_hashing() -> None:
    # Celsius conversion rounds a tiny Kelvin magnitude to zero, so comparing in
    # either operand's scale would make equality order-dependent.
    kelvin = ureg.Quantity(1e-14, "kelvin", kind="Temperature")
    celsius = kelvin.to(degC)

    assert (kelvin == celsius) == (celsius == kelvin)
    assert (kelvin != celsius) == (celsius != kelvin)
    for left, right in ((kelvin, celsius), (20 * degC, (20 * degC).to(degF))):
        if left == right:
            assert hash(left) == hash(right)
            assert {left: "value"}[right] == "value"


def test_offset_equality_broadcasts_and_propagates_nan() -> None:
    scalar = 0.0 * degC
    array = ureg.Quantity(np.array([0.0, np.nan, 1.0]), degF)

    np.testing.assert_array_equal(scalar == array, [False, False, False])
    np.testing.assert_array_equal(array != scalar, [True, True, True])
    np.testing.assert_array_equal(
        scalar == ureg.Quantity(np.array([[0.0], [0.0]]), degC).to(degF),
        [[True], [True]],
    )


def test_dimensional_temperature_ordering_against_scalar_uses_pint_error() -> None:
    with pytest.raises(ValueError, match="Cannot compare PlainQuantity"):
        _ = 20 * degC < 30


def test_compatibility_requires_exact_affine_role_and_is_total_in_contexts() -> None:
    absolute = 20 * degC
    equivalent_absolute = 68 * degF
    generic = 293.15 * K

    assert absolute.is_compatible_with(equivalent_absolute)
    assert not absolute.is_compatible_with(generic)
    assert not generic.is_compatible_with(absolute)
    assert absolute.is_compatible_with(degC)
    assert absolute.is_compatible_with(K)
    assert degC.is_compatible_with(absolute)
    assert not K.is_compatible_with(degC)
    assert degC.is_compatible_with(K)

    with ureg.context("spectroscopy"):
        assert not generic.is_compatible_with(absolute)
        assert not K.is_compatible_with(degC)


@pytest.mark.parametrize("angle_dimension", [False, True])
@pytest.mark.parametrize("scale", ["degRe", "custom_temperature"])
def test_definition_driven_offset_temperature_roles(angle_dimension: bool, scale: str) -> None:
    registry = SemanticRegistry(angle_dimension=angle_dimension)
    registry.define("custom_temperature = 2 * kelvin; offset: 100 = custom_t")
    quantity = registry.Quantity(20, scale)
    assert quantity.affine_role is AffineRole.ABSOLUTE
    assert quantity.to("kelvin").kind == "Temperature"
    difference = quantity - registry.Quantity(10, scale)
    assert difference.affine_role is AffineRole.DELTA
    assert difference.kind == "TemperatureDifference"
    assert difference.magnitude == 10
    assert difference.units == registry.Unit("delta_" + scale)
    assert (quantity + difference).magnitude == pytest.approx(30, rel=1e-15, abs=0.0)
    with pytest.raises(TypeError, match="cannot be negated"):
        _ = -quantity
    with pytest.raises(TypeError, match="multiplication or division"):
        quantity * 2
    with pytest.raises(ValueError, match="absolute temperature unit"):
        registry.Quantity(20, scale, kind="TemperatureDifference")


@pytest.mark.parametrize("scale", ["millikelvin", "microkelvin"])
def test_linear_absolute_temperature_subtraction_retains_scale(scale: str) -> None:
    registry = SemanticRegistry()
    a = registry.Quantity(300000, scale, kind="Temperature")
    b = registry.Quantity(290000, scale, kind="Temperature")
    difference = a - b
    assert difference.units == a.units
    assert difference.kind == "TemperatureDifference"
    assert difference.magnitude == 10000
    assert (b + difference).magnitude == 300000


def test_affine_roles_follow_new_definitions_and_delta_prefixes() -> None:
    registry = SemanticRegistry(on_redefinition="ignore")
    registry.define("custom_temperature = kelvin")
    quantity = registry.Quantity(20, "custom_temperature")
    assert quantity.affine_role is AffineRole.GENERIC
    registry.define("custom_temperature = kelvin; offset: 100 = custom_t")
    assert str(quantity.affine_role) == "Absolute"
    assert registry.Quantity(20, "custom_t").affine_role is AffineRole.ABSOLUTE
    assert registry.Quantity(1, "millidelta_custom_temperature").affine_role is AffineRole.DELTA
    with pytest.raises(ValueError, match="delta temperature unit"):
        registry.Quantity(1, "millidelta_custom_temperature", kind="Temperature")


def test_offset_length_keeps_pint_behavior() -> None:
    registry = SemanticRegistry(autoconvert_offset_to_baseunit=True)
    registry.define("shifted_length = meter; offset: 10")
    quantity = registry.Quantity(2, "shifted_length")
    assert quantity.affine_role is None
    assert quantity.to("meter").magnitude == 12


@pytest.mark.parametrize("use_unit", [False, True])
def test_offset_quantity_is_unequal_to_other_dimensions(use_unit: bool) -> None:
    registry = SemanticRegistry(autoconvert_offset_to_baseunit=True)
    registry.define("shifted_length = meter; offset: 10")
    quantity = registry.Quantity(2, "shifted_length")
    other = registry.second if use_unit else registry.Quantity(1, "second")

    assert bool(quantity == other) is False
    assert quantity != other
    if not use_unit:
        assert bool(other == quantity) is False
        assert other != quantity
    arrays = registry.Quantity(np.array([2.0, 3.0]), "shifted_length")
    np.testing.assert_array_equal(arrays == registry.Quantity(1, "second"), [False, False])


def test_affine_role_cache_tracks_context_redefinitions() -> None:
    registry = SemanticRegistry()
    registry.define("custom_temperature = kelvin")
    quantity = registry.Quantity(20, "custom_temperature")
    context = pint.Context("offset_role")
    context.redefine("custom_temperature = kelvin; offset: 100")
    assert str(quantity.affine_role) == "Generic"
    registry.add_context(context)
    with registry.context("offset_role"):
        assert str(quantity.affine_role) == "Absolute"
    assert str(quantity.affine_role) == "Generic"


def test_dimension_container_does_not_crash_affine_role_lookup() -> None:
    registry = SemanticRegistry()
    quantity = registry.Quantity(1, registry.UnitsContainer({"[temperature]": 1}))
    assert quantity.affine_role is AffineRole.GENERIC


def test_compound_linear_temperature_supports_interval_arithmetic() -> None:
    registry = SemanticRegistry()
    left = registry.Quantity(300, "joule/boltzmann_constant", kind="Temperature")
    right = registry.Quantity(290, left.units, kind="Temperature")
    difference = left - right
    assert difference.magnitude == 10
    assert difference.units == left.units
    assert difference.kind == "TemperatureDifference"
    assert (right + difference).magnitude == 300


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.iadd, operator.isub])
@pytest.mark.parametrize("reverse", [False, True])
def test_affine_dimension_mismatch_raises_dimensionality_error(
    operation: Any, reverse: bool
) -> None:
    absolute = ureg.Quantity(np.array([20.0, 25.0]), "degC")
    length = ureg.Quantity(np.array([1.0, 2.0]), "meter", kind="Length")
    left, right = (length, absolute) if reverse else (absolute, length)
    with pytest.raises(pint.DimensionalityError, match="Cannot convert"):
        operation(left, right)
    np.testing.assert_allclose(absolute.magnitude, [20, 25], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(length.magnitude, [1, 2], rtol=0.0, atol=0.0)

    assert absolute.units == ureg.degC
    assert absolute.kind is None
    assert absolute.affine_role is AffineRole.ABSOLUTE
    assert length.units == ureg.meter
    assert length.kind == "Length"
    assert length.affine_role is None


@pytest.mark.parametrize(
    (
        "unit",
        "kind",
        "right_unit",
        "right_value",
        "operation",
        "expected_unit",
        "expected_value",
        "expected_kind",
    ),
    [
        ("degC", None, "degF", 68.0, operator.isub, "delta_degC", 5.0, "TemperatureDifference"),
        (
            "kelvin",
            "Temperature",
            "degC",
            20.0,
            operator.isub,
            "kelvin",
            6.85,
            "TemperatureDifference",
        ),
        ("degC", None, "delta_degF", 9.0, operator.iadd, "degC", 30.0, "Temperature"),
        ("delta_degF", None, "degC", 20.0, operator.iadd, "degC", 25.0, "Temperature"),
    ],
)
def test_inplace_affine_arithmetic_matches_out_of_place_and_preserves_buffer(
    unit: str,
    kind: str | None,
    right_unit: str,
    right_value: float,
    operation: Any,
    expected_unit: str,
    expected_value: float,
    expected_kind: str,
) -> None:
    value = 300.0 if unit == "kelvin" else 9.0 if unit == "delta_degF" else 25.0
    original = np.array([value, value], dtype=np.float64)
    left = ureg.Quantity(original, unit, kind=kind)
    right = ureg.Quantity(right_value, right_unit)
    expected = left - right if operation is operator.isub else left + right
    result = operation(left, right)
    assert result is left
    assert result.magnitude is original
    assert result.units == ureg.Unit(expected_unit)
    assert result.kind == expected_kind
    assert result.affine_role == expected.affine_role
    np.testing.assert_allclose(result.magnitude, expected.magnitude, rtol=1e-15, atol=1e-12)
    np.testing.assert_allclose(result.magnitude, expected_value, rtol=1e-15, atol=1e-12)


def test_inplace_affine_casting_failure_keeps_values_and_semantics() -> None:
    original = np.array([25, 30], dtype=np.int64)
    left = ureg.Quantity(original, degC)
    with pytest.raises(TypeError, match="Cannot cast"):
        left -= 68 * degF
    assert left.magnitude is original
    np.testing.assert_array_equal(original, [25, 30])
    assert left.units == degC
    assert left.affine_role is AffineRole.ABSOLUTE


def test_inplace_affine_read_only_buffer_rejects_without_change() -> None:
    original = np.array([20.0, 21.0])
    original.flags.writeable = False
    left = ureg.Quantity(original, degC)
    with pytest.raises(ValueError, match="read-only"):
        left += 5 * delta_degC
    assert left.magnitude is original
    np.testing.assert_array_equal(original, [20.0, 21.0])
    assert left.affine_role is AffineRole.ABSOLUTE


def test_inplace_affine_writes_through_strided_views() -> None:
    base = np.zeros(6)
    view = base[::2]
    left = ureg.Quantity(view, degC)
    left += ureg.Quantity(np.array([1.0, 2.0, 3.0]), delta_degC)
    assert left.magnitude is view
    np.testing.assert_array_equal(base, [1.0, 0.0, 2.0, 0.0, 3.0, 0.0])


def test_inplace_affine_keeps_destination_dtype() -> None:
    original = np.array([20.0, 21.0], dtype=np.float32)
    left = ureg.Quantity(original, degC)
    left += ureg.Quantity(np.array([1.0, 2.0], dtype=np.float64), delta_degC)
    assert left.magnitude is original
    assert original.dtype == np.float32
    np.testing.assert_allclose(original, [21.0, 23.0], rtol=1e-6, atol=0.0)


def test_absolute_hashes_match_after_conversion_to_kelvin() -> None:
    celsius = 20 * degC
    kelvin = (293.15 * K).as_kind("Temperature")
    assert celsius == kelvin
    assert hash(celsius) == hash(kelvin)
    assert {celsius: "room"}[kelvin] == "room"
