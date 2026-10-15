"""Fail-closed NumPy and masked-array compatibility tests."""

import warnings
from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest
from pint.errors import DimensionalityError, UnitStrippedWarning

from xpint import AffineRole, SemanticQuantity, SemanticRegistry
from xpint.units import J, degC, m, ureg


def test_add_and_dimensionless_multiply_follow_quantity_semantics() -> None:
    energy = ureg.Quantity(np.array([1.0, 2.0]), J, kind="Energy")
    unkinded = ureg.Quantity(np.array([3.0, 4.0]), J)

    with pytest.raises(TypeError, match="quantity kinds"):
        np.add(energy, unkinded)
    scaled = cast(SemanticQuantity, np.multiply(energy, 2))

    np.testing.assert_allclose(
        cast(Any, scaled.magnitude),
        np.array([2.0, 4.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert scaled.kind == "Energy"


def test_sum_and_mean_preserve_kind() -> None:
    energy = ureg.Quantity(np.array([1.0, 2.0]), J, kind="Energy")

    total = cast(SemanticQuantity, np.sum(energy))
    average = cast(SemanticQuantity, np.mean(energy))

    assert total.magnitude == pytest.approx(3.0, rel=0.0, abs=0.0)
    assert average.magnitude == pytest.approx(1.5, rel=0.0, abs=0.0)
    assert total.kind == "Energy"
    assert average.kind == "Energy"


def test_direct_reductions_preserve_kind_and_designation() -> None:
    samples = ureg.Quantity(np.array([1.0, 2.0]), m["tissue"], kind="Length")

    total = samples.sum()
    average = samples.mean()

    assert total.kind == "Length"
    assert average.kind == "Length"
    assert total.designation_signature == samples.designation_signature
    assert average.designation_signature == samples.designation_signature


def test_array_accessors_preserve_semantics() -> None:
    samples = ureg.Quantity(
        np.array([[1.0 + 2.0j, 3.0 + 4.0j]]),
        m["tissue"],
        kind="Length",
    )

    results = [
        samples.copy(),
        samples.flatten(),
        samples.item(0),
        samples.real,
        samples.imag,
        samples.T,
        next(samples.flat),
        samples.clip(1 * m["tissue"], 2 * m["tissue"]),
    ]

    for result in results:
        assert result.kind == "Length"
        assert result.designation_signature == samples.designation_signature


def test_array_mutation_rejects_semantic_mismatch() -> None:
    energy = ureg.Quantity(np.array([1.0, 2.0]), J, kind="Energy")
    designated = ureg.Quantity(np.array([1.0, 2.0]), m["tissue"])

    with pytest.raises(TypeError, match=r"'Energy'.*'Torque'"):
        energy[0] = (3 * J).as_kind("Torque")
    with pytest.raises(TypeError, match="different factor designations"):
        designated.put([0], 3 * m["blood"])
    with pytest.raises(TypeError, match="different factor designations"):
        designated.fill(3 * m["blood"])


@pytest.mark.parametrize("operation", ["fill", "put", "setitem"])
@pytest.mark.parametrize("kinds", [("Energy", None), (None, "Energy"), ("Energy", "Torque")])
def test_mutation_rejects_kind_mismatch_before_changing_target(
    operation: str, kinds: tuple[str | None, str | None]
) -> None:
    magnitude = np.array([1, 2], dtype=np.int16)
    target = ureg.Quantity(magnitude, J, kind=kinds[0])
    source = ureg.Quantity(3, "kJ", kind=kinds[1])

    with pytest.raises(TypeError, match="quantity kinds"):
        if operation == "fill":
            target.fill(source)
        elif operation == "put":
            target.put([0], source)
        else:
            target[0] = source

    assert target.magnitude is magnitude
    np.testing.assert_array_equal(magnitude, [1, 2])
    assert magnitude.dtype == np.int16
    assert target.kind == kinds[0]
    assert target.units == J


def test_fill_with_matching_kind_keeps_pint_unit_replacement() -> None:
    target = ureg.Quantity(np.array([1, 2], dtype=np.int16), J, kind="Energy")
    target.fill(ureg.Quantity(3, "kJ", kind="Energy"))

    np.testing.assert_array_equal(target.magnitude, [3, 3])
    assert target.units == ureg.kilojoule
    assert target.kind == "Energy"


@pytest.mark.parametrize("kinds", [("Energy", None), (None, "Energy"), ("Energy", "Torque")])
def test_searchsorted_rejects_kind_mismatch(kinds: tuple[str | None, str | None]) -> None:
    target = ureg.Quantity([1, 2], J, kind=kinds[0])
    with pytest.raises(TypeError, match="quantity kinds"):
        target.searchsorted(ureg.Quantity(1, J, kind=kinds[1]))


@pytest.mark.parametrize("kinds", [("Energy", None), (None, "Energy"), ("Energy", "Torque")])
def test_masked_filled_rejects_kind_mismatch(kinds: tuple[str | None, str | None]) -> None:
    magnitude = np.ma.array([1, 2], mask=[False, True], dtype=np.int16)
    target = ureg.Quantity(magnitude, J, kind=kinds[0])
    with pytest.raises(TypeError, match="quantity kinds"):
        target.filled(ureg.Quantity(3, J, kind=kinds[1]))
    np.testing.assert_array_equal(magnitude.data, [1, 2])
    np.testing.assert_array_equal(magnitude.mask, [False, True])


def test_clip_accepts_unkinded_bound_and_preserves_kind() -> None:
    target = ureg.Quantity([1, 2, 3], J, kind="Energy")
    clipped = target.clip(ureg.Quantity(2, J), ureg.Quantity(2, J))
    np.testing.assert_array_equal(clipped.magnitude, [2, 2, 2])
    assert clipped.kind == "Energy"


def test_fill_rejects_invalid_final_kind_before_mutation() -> None:
    energy = ureg.Quantity(np.array([1.0, 2.0]), J, kind="Energy")
    _ = energy.dimensionality

    with pytest.raises(DimensionalityError):
        energy.fill(3 * m)

    assert energy.units == J
    assert energy.kind == "Energy"
    assert energy.dimensionality == J.dimensionality


def test_plain_fill_refreshes_cached_dimensionality() -> None:
    values = ureg.Quantity(np.array([1.0, 2.0]), J)
    _ = values.dimensionality

    values.fill(3 * m)

    assert values.units == m
    assert values.dimensionality == m.dimensionality


def test_implicit_ndarray_downcast_rejects_but_variance_method_is_supported() -> None:
    energy = ureg.Quantity(np.array([1.0, 2.0]), J, kind="Energy")

    with pytest.raises(TypeError, match="cannot downcast a classified xpint quantity"):
        np.asarray(energy)
    variance = energy.var()
    assert variance.units == J**2
    assert variance.kind is None
    assert variance.magnitude == pytest.approx(0.25, rel=1e-15, abs=0.0)


def test_plain_ndarray_export_supports_numpy_two_copy_keyword() -> None:
    values = ureg.Quantity(np.array([1.0, 2.0]), m)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        view = np.array(values, copy=False)
        copied = np.array(values, copy=True)

    assert any(isinstance(item.message, UnitStrippedWarning) for item in caught)
    assert not any(isinstance(item.message, DeprecationWarning) for item in caught)
    assert np.shares_memory(view, cast(Any, values.magnitude))
    assert not np.shares_memory(copied, cast(Any, values.magnitude))


def test_reduction_rejects_semantic_initial_and_output_mismatch() -> None:
    energy = ureg.Quantity(np.array([1.0, 2.0]), J, kind="Energy")
    torque = ureg.Quantity(1.0, J, kind="Torque")
    output = ureg.Quantity(np.zeros(()), m)

    with pytest.raises(TypeError, match=r"'Energy'.*'Torque'"):
        np.sum(cast(Any, energy), initial=cast(Any, torque))
    with pytest.raises(TypeError, match="does not validate or preserve"):
        np.sum(cast(Any, energy), out=cast(Any, output))

    assert output.magnitude == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert output.units == m


@pytest.mark.parametrize(
    ("name", "method"), [("sum", False), ("sum", True), ("nansum", False), ("cumsum", True)]
)
@pytest.mark.parametrize("kinds", [("Energy", None), (None, "Energy"), ("Energy", "Torque")])
def test_additive_reduction_initial_rejects_kind_mismatch(
    name: str, method: bool, kinds: tuple[str | None, str | None]
) -> None:
    values = ureg.Quantity([1, 2], J, kind=kinds[0])
    initial = ureg.Quantity(3, J, kind=kinds[1])
    function = getattr(values, name) if method else getattr(np, name)
    args = () if method else (values,)
    with pytest.raises(TypeError, match="quantity kinds"):
        function(*args, initial=initial)


def test_scalar_reduction_initial_is_magnitude_in_existing_semantic_unit() -> None:
    values = ureg.Quantity(np.array([1.0, 2.0]), m["tissue"], kind="Length")

    total = cast(SemanticQuantity, np.sum(values, initial=5.0))

    assert total.magnitude == pytest.approx(8.0, rel=0.0, abs=0.0)
    assert total.kind == "Length"
    assert total.designation_signature == values.designation_signature


def test_ufunc_output_is_rejected_before_pint_recursion_or_mutation() -> None:
    values = ureg.Quantity(np.array([1.0, 2.0]), m)
    output = ureg.Quantity(np.zeros(2), m)

    with pytest.raises(TypeError, match="can recurse or write through"):
        np.add(cast(Any, values), cast(Any, values), out=cast(Any, output))

    np.testing.assert_array_equal(cast(Any, output.magnitude), np.zeros(2))


def test_semantic_mismatch_equality_preserves_broadcast_shape() -> None:
    energy = ureg.Quantity(np.ones((2, 1)), J, kind="Energy")
    torque = ureg.Quantity(np.ones((1, 3)), J, kind="Torque")

    equal = energy == torque
    not_equal = energy != torque

    np.testing.assert_array_equal(equal, np.zeros((2, 3), dtype=bool))
    np.testing.assert_array_equal(not_equal, np.ones((2, 3), dtype=bool))


def test_kind_to_unkinded_numpy_comparison_uses_strict_policy() -> None:
    energy = ureg.Quantity(np.ones(2), J, kind="Energy")
    unkinded = ureg.Quantity(np.ones(2), J)

    np.testing.assert_array_equal(np.equal(energy, unkinded), np.zeros(2, dtype=bool))
    with pytest.raises(TypeError, match="different kinds"):
        np.less(energy, unkinded)


def test_equality_and_inequality_delegate_to_xarray() -> None:
    xr = pytest.importorskip("xarray")
    quantity = ureg.Quantity(1.0)
    data = xr.DataArray(np.array([1.0, 2.0]), dims="sample")

    equal = quantity == data
    not_equal = quantity != data

    assert isinstance(equal, xr.DataArray)
    assert isinstance(not_equal, xr.DataArray)
    np.testing.assert_array_equal(equal.data, np.array([True, False]))
    np.testing.assert_array_equal(not_equal.data, np.array([False, True]))


def test_plain_array_exponents_delegate_to_pint() -> None:
    values = ureg.Quantity(np.array([2.0, 3.0]))

    result = values ** np.array([2.0, 3.0])

    np.testing.assert_array_equal(cast(Any, result.magnitude), np.array([4.0, 27.0]))
    assert result.dimensionless


def test_classified_power_requires_scalar_rational_exponent() -> None:
    length = ureg.Quantity(2.0, m["tissue"], kind="Length")

    with pytest.raises(TypeError, match="scalar rational exponent"):
        length ** np.array([1.0, 2.0])

    restored = (length ** (1 / 3)) ** 3
    exact = (length ** Fraction(1, 3)) ** 3
    assert restored.units == length.units
    assert exact.units == length.units


def test_small_designated_power_does_not_round_to_undesignated() -> None:
    length = ureg.Quantity(2.0, m["tissue"])

    result = length**1e-7

    assert result.has_designation
    assert result.designation_signature[0].exponent == Fraction(1, 10_000_000)


def test_masked_array_extraction_preserves_semantics() -> None:
    magnitude = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
    values = ureg.Quantity(magnitude, m["tissue"], kind="Length")

    filled = values.filled(4.0)
    compressed = values.compressed()

    np.testing.assert_array_equal(cast(Any, filled.magnitude), np.array([1.0, 4.0, 3.0]))
    np.testing.assert_array_equal(cast(Any, compressed.magnitude), np.array([1.0, 3.0]))
    assert filled.kind == compressed.kind == "Length"
    assert filled.designation_signature == values.designation_signature
    assert compressed.designation_signature == values.designation_signature


def test_plain_masked_array_extraction_matches_pint_return_type() -> None:
    magnitude = np.ma.array([1.0, 2.0], mask=[False, True])
    values = ureg.Quantity(magnitude, m)

    assert isinstance(values.filled(3.0), np.ndarray)
    assert isinstance(values.compressed(), np.ndarray)


@pytest.mark.parametrize("method", [False, True])
def test_mean_preserves_absolute_role_but_sum_is_rejected(method: bool) -> None:
    temperatures = ureg.Quantity(np.array([20.0, 22.0]), degC)

    average = temperatures.mean() if method else cast(SemanticQuantity, np.mean(temperatures))

    assert average.magnitude == pytest.approx(21.0, rel=0.0, abs=0.0)
    assert average.affine_role is AffineRole.ABSOLUTE
    with pytest.raises(TypeError, match="undefined for absolute temperatures"):
        if method:
            temperatures.sum()
        else:
            np.sum(temperatures)


def test_undefined_semantic_ufunc_fails_closed() -> None:
    energy = ureg.Quantity(np.array([1.0, 2.0]), J, kind="Energy")

    with pytest.raises(TypeError, match="no xpint semantic propagation rule"):
        np.exp(energy)


def test_variance_drops_kind_and_squares_units() -> None:
    energy = ureg.Quantity(np.array([1.0, 2.0]), J, kind="Energy")

    variance = cast(SemanticQuantity, np.var(energy))
    assert variance.units == J**2
    assert variance.kind is None
    assert variance.magnitude == pytest.approx(0.25, rel=1e-15, abs=0.0)


def test_absolute_temperature_ufuncs_use_strict_affine_rules() -> None:
    absolute = 20 * degC

    with pytest.raises(TypeError, match="in kelvin, call as_unkinded"):
        np.absolute(absolute)
    with pytest.raises(TypeError, match="cannot be negated"):
        np.negative(absolute)
    with pytest.raises(TypeError, match="cannot be exponentiated"):
        np.power(absolute, 1)


@pytest.mark.parametrize("ufunc", [np.sin, np.cos, np.tan])
def test_trigonometry_accepts_a_kinded_angle_and_drops_the_kind(ufunc: Any) -> None:
    """sin of an angle is a ratio, so no input kind describes the result."""

    angle = ureg.Quantity(np.array([0.0, 1.0]), ureg.radian, kind="PlaneAngle")

    result = cast(SemanticQuantity, ufunc(angle))

    assert result.kind is None
    assert result.dimensionless
    assert result.shape == (2,)
    np.testing.assert_allclose(result.magnitude, ufunc(np.array([0.0, 1.0])), rtol=1e-15)


def test_trigonometry_honors_unit_scale_through_the_kinded_path() -> None:
    """Degrees must still convert, which is the behavior Pint supplies."""

    right_angle = ureg.Quantity(90.0, ureg.degree, kind="PlaneAngle")

    result = cast(SemanticQuantity, np.sin(right_angle))

    assert result.magnitude == pytest.approx(1.0, abs=1e-15)


def test_inverse_trigonometry_returns_an_unkinded_angle() -> None:
    ratio = ureg.Quantity(0.5, "", kind="Dimensionless")

    result = cast(SemanticQuantity, np.arcsin(ratio))

    assert result.kind is None
    assert result.units == ureg.radian
    assert result.magnitude == pytest.approx(np.pi / 6, rel=1e-15)


def test_arctan2_accepts_kinded_operands_and_returns_an_angle() -> None:
    side = ureg.Quantity(1.0, m, kind="Length")

    result = cast(SemanticQuantity, np.arctan2(side, side))

    assert result.kind is None
    assert result.units == ureg.radian
    assert result.magnitude == pytest.approx(np.pi / 4, rel=1e-15)


def test_arctan2_rejects_mismatched_semantic_inputs() -> None:
    energy = ureg.Quantity(1.0, J, kind="Energy")
    torque = ureg.Quantity(1.0, J, kind="Torque")
    unkinded_length = ureg.Quantity(1.0, m)
    kinded_length = unkinded_length.as_kind("Length")

    with pytest.raises(TypeError, match="different kinds"):
        np.arctan2(energy, torque)
    with pytest.raises(TypeError, match="different kinds"):
        np.arctan2(kinded_length, unkinded_length)
    with pytest.raises(TypeError, match="classified quantity and a plain value"):
        np.arctan2(energy, 1.0)


def test_arctan2_rejects_foreign_registry_and_affine_role_mismatch() -> None:
    other_registry = SemanticRegistry()
    other_energy = other_registry.Quantity(1.0, "joule", kind="Energy")

    with pytest.raises(ValueError, match="different SemanticRegistry"):
        np.arctan2(ureg.Quantity(1.0, J, kind="Energy"), other_energy)
    with pytest.raises(TypeError, match="different affine roles"):
        np.arctan2(20 * degC, 5 * ureg.delta_degC)


def test_trigonometry_rejects_retained_designations() -> None:
    """A designated ratio records how a value was built, which is not an angle."""

    ratio = (1.0 * m["arc"]) / (1.0 * m["radius"])

    with pytest.raises(TypeError, match="retained factor designations"):
        np.sin(ratio)


def test_trigonometry_leaves_the_dimensional_check_to_pint() -> None:
    energy = ureg.Quantity(np.array([1.0, 2.0]), J, kind="Energy")

    with pytest.raises(DimensionalityError):
        np.sin(energy)


@pytest.mark.parametrize("method", [False, True])
@pytest.mark.parametrize("name", ["sum", "max", "min"])
@pytest.mark.parametrize("positional", [False, True])
def test_reduction_initial_converts_units_across_call_styles(
    method: bool, name: str, positional: bool
) -> None:
    values = ureg.Quantity(np.array([1.0, 2.0]), m["tissue"], kind="Length")
    initial = ureg.Quantity(300, ureg.centimeter["tissue"], kind="Length")
    function = getattr(values, name) if method else getattr(np, name)
    args: list[Any] = [] if method else [values]
    if positional:
        args.extend(
            [None, None, None, False, initial] if name == "sum" else [None, None, False, initial]
        )
        result = function(*args)
    else:
        result = function(*args, initial=initial)
    assert result.kind == "Length"
    assert result.designation_signature == values.designation_signature
    assert result.magnitude == pytest.approx(
        {"sum": 6, "max": 3, "min": 1}[name], rel=1e-15, abs=0.0
    )


@pytest.mark.parametrize("method", [False, True])
@pytest.mark.parametrize("name", ["sum", "max", "min", "mean"])
def test_reduction_positional_output_rejects_before_mutation(method: bool, name: str) -> None:
    values = ureg.Quantity(np.array([1.0, 2.0]), "m")
    output = ureg.Quantity(np.array(7.0), "s")
    function = getattr(values, name) if method else getattr(np, name)
    args: list[Any] = [] if method else [values]
    args.extend([None, None, output] if name in ("sum", "mean") else [None, output])
    with pytest.raises(TypeError, match="output target"):
        function(*args)
    assert output.magnitude == pytest.approx(7, rel=0.0, abs=0.0)
    assert output.units == ureg.second


@pytest.mark.parametrize("method", [False, True])
@pytest.mark.parametrize("name", ["sum", "max", "min"])
def test_reduction_positional_initial_rejects_wrong_units(method: bool, name: str) -> None:
    values = ureg.Quantity(np.array([1.0, 2.0]), "m")
    function = getattr(values, name) if method else getattr(np, name)
    args: list[Any] = [] if method else [values]
    args.extend(
        [None, None, None, False, 100 * ureg.second]
        if name == "sum"
        else [None, None, False, 100 * ureg.second]
    )
    with pytest.raises(DimensionalityError):
        function(*args)


@pytest.mark.parametrize("method", [False, True])
def test_extrema_initial_requires_matching_semantic_state(method: bool) -> None:
    energy = ureg.Quantity(np.array([1.0, 2.0]), "J", kind="Energy")
    maximum = energy.max if method else lambda **kw: np.max(cast(Any, energy), **kw)
    with pytest.raises(TypeError, match="different kinds"):
        maximum(initial=ureg.Quantity(3, "J", kind="Torque"))
    with pytest.raises(TypeError, match="different kinds"):
        maximum(initial=ureg.Quantity(3, "J"))
    with pytest.raises(ValueError, match="different SemanticRegistry"):
        maximum(initial=SemanticRegistry().Quantity(3, "J", kind="Energy"))
    temperature = ureg.Quantity(np.array([1.0, 2.0]), "K", kind="Temperature")
    maximum = temperature.max if method else lambda **kw: np.max(cast(Any, temperature), **kw)
    with pytest.raises(TypeError, match="different affine roles"):
        maximum(initial=ureg.Quantity(3, "K"))


@pytest.mark.parametrize("method", [False, True])
def test_extrema_scalar_initial_is_an_in_unit_magnitude(method: bool) -> None:
    values = ureg.Quantity(np.array([1.0, 2.0]), m["tissue"], kind="Length")
    result = values.max(initial=3) if method else np.max(cast(Any, values), initial=3)
    assert result.magnitude == pytest.approx(3, rel=0.0, abs=0.0)
    assert result.kind == "Length"
    assert result.designation_signature == values.designation_signature


def test_nansum_positional_initial_converts_before_addition() -> None:
    values = ureg.Quantity(np.array([1.0, np.nan, 2.0]), "m")
    result = cast(Any, np.nansum)(values, None, None, None, False, 300 * ureg.cm)
    assert result.magnitude == pytest.approx(6, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("method", [False, True])
def test_extrema_initial_rejects_different_designations(method: bool) -> None:
    values = ureg.Quantity(np.array([1.0, 2.0]), m["tissue"])
    function = values.max if method else np.max
    args: list[Any] = [] if method else [values]
    with pytest.raises(TypeError, match="different factor designations"):
        function(*args, initial=ureg.Quantity(3, m["blood"]))


@pytest.mark.parametrize(("name", "expected"), [("amax", 3), ("amin", 1)])
def test_extrema_aliases_convert_positional_initial(name: str, expected: int) -> None:
    values = ureg.Quantity(np.array([1.0, 2.0]), "m", kind="Length")
    initial = ureg.Quantity(300, "cm", kind="Length")
    result = getattr(np, name)(values, None, None, False, initial)
    assert result.magnitude == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert result.kind == "Length"


def test_nanmean_rejects_positional_output_and_preserves_mean() -> None:
    values = ureg.Quantity(np.array([1.0, np.nan, 2.0]), "m", kind="Length")
    output = ureg.Quantity(np.array(7.0), "s")
    with pytest.raises(TypeError, match="output target"):
        cast(Any, np.nanmean)(values, None, None, output)
    assert output.magnitude == pytest.approx(7, rel=0.0, abs=0.0)
    result = cast(SemanticQuantity, np.nanmean(values))
    assert result.magnitude == pytest.approx(1.5, rel=1e-15, abs=0.0)
    assert result.kind == "Length"


@pytest.mark.parametrize("angle_dimension", [False, True])
@pytest.mark.parametrize("unit", ["degC", "kelvin"])
def test_arctan2_rejects_absolute_temperatures_in_any_scale(
    angle_dimension: bool, unit: str
) -> None:
    registry = SemanticRegistry(angle_dimension=angle_dimension)
    left = registry.Quantity(20, "degC")
    right = registry.Quantity(10, "degC")
    if unit == "kelvin":
        left, right = left.to(unit), right.to(unit)
    with pytest.raises(TypeError, match=r"arctan2.*absolute temperatures"):
        np.arctan2(left, right)
    angle = cast(
        SemanticQuantity,
        np.arctan2(left.to("K").as_unkinded(), right.to("K").as_unkinded()),
    )
    assert angle.m_as("radian") == pytest.approx(np.arctan2(293.15, 283.15), rel=1e-15, abs=0.0)
    assert angle.kind is None


@pytest.mark.parametrize("angle_dimension", [False, True])
@pytest.mark.parametrize(
    ("left_value", "left_unit", "right_value", "right_unit"),
    [
        (10, "delta_degC", 18, "delta_degF"),
        (1, "meter", 100, "centimeter"),
        (1, "kelvin", 1, "kelvin"),
    ],
)
def test_arctan2_preserves_delta_and_ordinary_numeric_behavior(
    angle_dimension: bool,
    left_value: int,
    left_unit: str,
    right_value: int,
    right_unit: str,
) -> None:
    registry = SemanticRegistry(angle_dimension=angle_dimension)
    left = registry.Quantity(left_value, left_unit)
    right = registry.Quantity(right_value, right_unit)
    angle = cast(SemanticQuantity, np.arctan2(left, right))
    assert angle.m_as("radian") == pytest.approx(np.pi / 4, rel=1e-15, abs=0.0)
    assert angle.kind is None
    assert angle.affine_role is None


@pytest.mark.parametrize("state", ["kind", "designation", "absolute"])
@pytest.mark.parametrize(
    "name,exponent", [("sqrt", 0.5), ("square", 2), ("cbrt", 1 / 3), ("reciprocal", -1)]
)
def test_unary_power_ufuncs_follow_semantic_powers(state: str, name: str, exponent: float) -> None:
    units = degC if state == "absolute" else m["sample"] if state == "designation" else m
    value = ureg.Quantity(
        np.array([1.0, 8.0], dtype=np.float64), units, kind="Length" if state == "kind" else None
    )
    operation = getattr(np, name)
    if state == "absolute":
        with pytest.raises(TypeError, match="absolute temperatures cannot be exponentiated"):
            operation(value)
        return
    result = operation(value)
    expected = value**exponent
    np.testing.assert_allclose(result.magnitude, operation(value.magnitude), rtol=1e-14, atol=0.0)
    assert result.units == expected.units
    assert result.designation_signature == expected.designation_signature
    assert result.kind is None


@pytest.mark.parametrize("state", ["kind", "designation", "absolute"])
@pytest.mark.parametrize("name", ["argmax", "argmin", "argsort", "nonzero"])
@pytest.mark.parametrize("method", [False, True])
def test_index_operations_return_plain_indices(state: str, name: str, method: bool) -> None:
    units = degC if state == "absolute" else m["sample"] if state == "designation" else m
    value = ureg.Quantity(
        np.array([3.0, 0.0, 2.0], dtype=np.float64),
        units,
        kind="Length" if state == "kind" else None,
    )
    result = getattr(value, name)() if method else getattr(np, name)(value)
    expected = getattr(np, name)(value.magnitude)
    assert not isinstance(result, SemanticQuantity)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("state", ["kind", "designation", "absolute"])
@pytest.mark.parametrize("name", ["std", "nanstd", "var", "nanvar", "cumsum", "nancumsum", "diff"])
def test_extended_numpy_operations_propagate_classification(state: str, name: str) -> None:
    units = degC if state == "absolute" else m["sample"] if state == "designation" else m
    value = ureg.Quantity(
        np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64),
        units,
        kind="Length" if state == "kind" else None,
    )
    operation = getattr(np, name)
    if state == "absolute" and name != "diff":
        with pytest.raises(TypeError, match=f"{name!r} is undefined for absolute temperatures"):
            operation(value, axis=1)
        return
    result = operation(value, axis=1)
    np.testing.assert_allclose(
        result.magnitude, operation(value.magnitude, axis=1), rtol=1e-14, atol=0.0
    )
    if name in ("var", "nanvar"):
        assert result.units == value.units**2
        assert result.kind is None
    elif state == "absolute":
        assert result.units == ureg.delta_degC
        assert result.affine_role is AffineRole.DELTA
        assert result.kind == "TemperatureDifference"
    else:
        assert result.units == value.units
        assert result.kind == value.kind
        assert result.designation_signature == value.designation_signature


@pytest.mark.parametrize("state", ["kind", "designation", "absolute"])
@pytest.mark.parametrize("name", ["std", "var", "cumsum"])
def test_extended_numpy_methods_match_functions(state: str, name: str) -> None:
    units = degC if state == "absolute" else m["sample"] if state == "designation" else m
    value = ureg.Quantity(
        np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float64),
        units,
        kind="Length" if state == "kind" else None,
    )
    if state == "absolute":
        with pytest.raises(TypeError, match=f"{name!r} is undefined for absolute temperatures"):
            getattr(value, name)(axis=1)
        return
    result = getattr(value, name)(axis=1)
    expected = getattr(np, name)(value, axis=1)
    np.testing.assert_allclose(result.magnitude, expected.magnitude, rtol=1e-14, atol=0.0)
    assert result.units == expected.units
    assert result.kind == expected.kind
    assert result.designation_signature == expected.designation_signature


@pytest.mark.parametrize("name", ["std", "nanstd", "var", "nanvar", "cumsum", "nancumsum"])
@pytest.mark.parametrize("positional", [False, True])
def test_extended_numpy_reductions_reject_out(name: str, positional: bool) -> None:
    value = ureg.Quantity(np.array([1.0, 2.0], dtype=np.float64), m, kind="Length")
    out = np.zeros(2, dtype=np.float64)
    with pytest.raises(TypeError, match="output target"):
        if positional:
            getattr(np, name)(value, None, None, out)
        else:
            getattr(np, name)(value, out=out)
    np.testing.assert_allclose(out, 0, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("name", ["std", "var", "cumsum"])
@pytest.mark.parametrize("positional", [False, True])
def test_extended_numpy_methods_reject_out(name: str, positional: bool) -> None:
    value = ureg.Quantity(np.array([1.0, 2.0], dtype=np.float64), m, kind="Length")
    out = np.zeros(2, dtype=np.float64)
    with pytest.raises(TypeError, match="output target"):
        if positional:
            getattr(value, name)(None, None, out)
        else:
            getattr(value, name)(out=out)
    np.testing.assert_allclose(out, 0, rtol=0.0, atol=0.0)


def test_cube_root_keeps_negative_real_domain_and_reciprocal_integer_dtype() -> None:
    values = ureg.Quantity(np.array([-8.0, 8.0], dtype=np.float64), m["sample"])
    rooted = cast(SemanticQuantity, np.cbrt(values))
    np.testing.assert_allclose(rooted.magnitude, [-2, 2], rtol=1e-15, atol=0.0)
    assert rooted.units == m["sample"] ** (1 / 3)
    integers = ureg.Quantity(np.array([1, 2], dtype=np.int64), m, kind="Length")
    reciprocal = cast(SemanticQuantity, np.reciprocal(integers))
    assert reciprocal.magnitude.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(reciprocal.magnitude, [1, 0])


@pytest.mark.parametrize("name", ["nanstd", "nanvar", "nancumsum"])
def test_nan_reductions_skip_nan_and_keep_semantics(name: str) -> None:
    values = ureg.Quantity(
        np.array([1.0, np.nan, 3.0], dtype=np.float64), m["sample"], kind="Length"
    )
    result = getattr(np, name)(values)
    np.testing.assert_allclose(
        result.magnitude, getattr(np, name)(values.magnitude), rtol=1e-15, atol=0.0
    )
    assert result.units == (values.units**2 if name == "nanvar" else values.units)
    assert result.kind == (None if name == "nanvar" else "Length")


@pytest.mark.parametrize("positional", [False, True])
def test_zero_order_diff_keeps_absolute_temperature(positional: bool) -> None:
    values = ureg.Quantity(np.array([20.0, 21.0], dtype=np.float64), degC)
    result = np.diff(values, 0) if positional else np.diff(values, n=0)
    assert result is values
    assert result.affine_role is AffineRole.ABSOLUTE


@pytest.mark.parametrize("edge", ["prepend", "append"])
def test_diff_requires_matching_semantics_at_edges(edge: str) -> None:
    values = ureg.Quantity(np.array([1.0, 3.0], dtype=np.float64), m["O2"])
    wrong_options: dict[str, Any] = {edge: ureg.Quantity(1, m["CO2"])}
    options: dict[str, Any] = {edge: ureg.Quantity(100, ureg.cm["O2"])}
    with pytest.raises(TypeError, match="different factor designations"):
        np.diff(values, **wrong_options)
    result = cast(SemanticQuantity, np.diff(values, **options))
    magnitude_options: dict[str, Any] = {edge: 1.0}
    expected = np.diff(values.magnitude, **magnitude_options)
    np.testing.assert_allclose(result.magnitude, expected, rtol=1e-15, atol=0.0)
    assert result.units == values.units


@pytest.mark.parametrize(
    "name,method",
    [
        ("std", False),
        ("nanstd", False),
        ("var", False),
        ("nanvar", False),
        ("std", True),
        ("var", True),
    ],
)
def test_statistical_mean_requires_matching_semantics(name: str, method: bool) -> None:
    values = ureg.Quantity(np.array([1.0, 3.0], dtype=np.float64), J, kind="Energy")
    wrong = ureg.Quantity(2.0, J, kind="Torque")
    operation = (
        getattr(values, name) if method else lambda **kwargs: getattr(np, name)(values, **kwargs)
    )
    with pytest.raises(TypeError, match="different kinds"):
        operation(mean=wrong)
    result = operation(mean=ureg.Quantity(0.002, "kJ", kind="Energy"))
    assert result.magnitude == pytest.approx(1, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("method", ["dot", "prod"])
def test_classified_product_methods_fail_closed(method: str) -> None:
    quantity = ureg.Quantity(np.array([1.0, 2.0], dtype=np.float64), ureg.mole["O2"])
    arguments = (quantity,) if method == "dot" else ()
    with pytest.raises(TypeError, match=f"{method!r} has no xpint semantic propagation rule"):
        getattr(quantity, method)(*arguments)
