"""Behavioral parity for quantities that do not use xpint semantics."""

from typing import Any, cast

import numpy as np
import pint
import pytest

from xpint import SemanticQuantity, SemanticRegistry
from xpint.units import m, ureg


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.array(1.5), id="zero-dimensional"),
        pytest.param(np.array([], dtype=np.float64), id="empty"),
        pytest.param(np.arange(6, dtype=np.int16).reshape(2, 3), id="matrix"),
    ],
)
def test_core_array_arithmetic_matches_pint_across_shapes(values: Any) -> None:
    plain_registry: pint.UnitRegistry[Any] = pint.UnitRegistry()
    xpint_quantity = ureg.Quantity(values.copy(), m)
    pint_quantity = plain_registry.Quantity(values.copy(), "meter")

    for xpint_result, pint_result in (
        (xpint_quantity + xpint_quantity, pint_quantity + pint_quantity),
        (xpint_quantity * 2, pint_quantity * 2),
    ):
        np.testing.assert_allclose(
            cast(Any, xpint_result.magnitude),
            cast(Any, pint_result.magnitude),
            rtol=0.0,
            atol=0.0,
        )
        assert xpint_result.shape == pint_result.shape
        assert xpint_result.magnitude.dtype == pint_result.magnitude.dtype
        assert str(xpint_result.units) == str(pint_result.units)


def test_reflected_ndarray_operations_match_pint() -> None:
    values = np.array([1.0, 2.0])
    xpint_quantity = ureg.Quantity(values, m)
    plain_registry: pint.UnitRegistry[Any] = pint.UnitRegistry()
    pint_quantity = plain_registry.Quantity(values, "meter")

    xpint_results = (
        values * m,
        values * xpint_quantity,
        values / xpint_quantity,
        np.multiply(values, xpint_quantity),
    )
    pint_results = (
        values * plain_registry.meter,
        values * pint_quantity,
        values / pint_quantity,
        np.multiply(values, pint_quantity),
    )

    for xpint_result, pint_result in zip(xpint_results, pint_results, strict=True):
        np.testing.assert_allclose(
            cast(Any, xpint_result.magnitude),
            cast(Any, pint_result.magnitude),
            rtol=0.0,
            atol=0.0,
        )
        assert xpint_result.dimensionality == pint_result.dimensionality


def test_reflected_ndarray_preserves_designated_unit() -> None:
    values = np.array([1.0, 2.0])

    result = values * m["tissue"]

    assert result.designation_signature == m["tissue"].designation_signature


def test_plain_ufunc_options_delegate_to_pint() -> None:
    values = np.array([1, 2], dtype=np.int16)
    xpint_quantity = ureg.Quantity(values, m)
    plain_registry: pint.UnitRegistry[Any] = pint.UnitRegistry()
    pint_quantity = plain_registry.Quantity(values, "meter")

    xpint_result = cast(
        SemanticQuantity,
        np.multiply(cast(Any, xpint_quantity), 2, dtype=np.float32),
    )
    pint_result = cast(
        pint.Quantity[Any],
        np.multiply(cast(Any, pint_quantity), 2, dtype=np.float32),
    )

    np.testing.assert_array_equal(
        cast(Any, xpint_result.magnitude),
        cast(Any, pint_result.magnitude),
    )
    assert xpint_result.magnitude.dtype == pint_result.magnitude.dtype
    assert xpint_result.dimensionality == pint_result.dimensionality


def test_plain_in_place_cast_error_matches_pint_and_preserves_alias() -> None:
    values = np.array([1, 2], dtype=np.int16)
    xpint_quantity = ureg.Quantity(values.copy(), m)
    plain_registry: pint.UnitRegistry[Any] = pint.UnitRegistry()
    pint_quantity = plain_registry.Quantity(values.copy(), "meter")
    xpint_alias = xpint_quantity.magnitude
    pint_alias = pint_quantity.magnitude

    with pytest.raises(TypeError):
        xpint_quantity.__iadd__(ureg.Quantity(np.array([0.5, 0.5]), m))
    with pytest.raises(TypeError):
        pint_quantity.__iadd__(plain_registry.Quantity(np.array([0.5, 0.5]), "meter"))

    assert xpint_alias is xpint_quantity.magnitude
    assert pint_alias is pint_quantity.magnitude
    np.testing.assert_array_equal(xpint_alias, pint_alias)


def test_ndarray_in_place_quantity_rejects_unit_stripping() -> None:
    values = np.zeros(2)
    quantity = ureg.Quantity(np.ones(2))

    with pytest.raises(TypeError, match="output target"):
        values += quantity

    np.testing.assert_array_equal(values, np.zeros(2))


@pytest.mark.parametrize("operation", ["power", "convert", "array_power"])
def test_in_place_array_power_and_conversion_match_pint(operation: str) -> None:
    plain_registry: pint.UnitRegistry[Any] = pint.UnitRegistry()
    units = "percent" if operation == "array_power" else "meter"
    xpint_quantity = ureg.Quantity(np.array([4.0, 9.0], dtype=np.float32), units)
    pint_quantity = plain_registry.Quantity(np.array([4.0, 9.0], dtype=np.float32), units)
    xpint_alias, pint_alias = xpint_quantity.magnitude, pint_quantity.magnitude
    if operation == "convert":
        xpint_quantity.ito("cm")
        pint_quantity.ito("cm")
    else:
        exponent: Any = np.array([2, 3]) if operation == "array_power" else 2
        xpint_quantity **= exponent
        pint_quantity **= exponent
    assert (xpint_quantity.magnitude is xpint_alias) == (pint_quantity.magnitude is pint_alias)
    assert xpint_quantity.magnitude.dtype == pint_quantity.magnitude.dtype
    assert str(xpint_quantity.units) == str(pint_quantity.units)
    # Both delegate the same numeric operation; allow one float32 rounding step.
    np.testing.assert_allclose(
        xpint_quantity.magnitude,
        pint_quantity.magnitude,
        rtol=float(np.finfo(np.float32).eps),
        atol=0.0,
    )


@pytest.mark.parametrize("operation", ["power", "convert"])
def test_plain_in_place_casting_failure_matches_pint(operation: str) -> None:
    plain_registry: pint.UnitRegistry[Any] = pint.UnitRegistry()
    for registry in (ureg, plain_registry):
        quantity = registry.Quantity(np.array([4, 9], dtype=np.int64), "meter")
        with pytest.raises(TypeError):
            if operation == "power":
                quantity **= 0.5
            else:
                quantity.ito("cm")


@pytest.mark.parametrize("operation", ["divide", "inplace", "explicit"])
def test_ordinary_unit_reduction_matches_pint(operation: str) -> None:
    plain_registry: pint.UnitRegistry[Any] = pint.UnitRegistry(auto_reduce_dimensions=True)
    semantic_registry = SemanticRegistry(auto_reduce_dimensions=True)
    results = []
    for registry in (semantic_registry, plain_registry):
        units = "meter/centimeter" if operation == "explicit" else "meter"
        quantity: Any = registry.Quantity(np.array([1.0, 2.0], dtype=np.float64), units)
        original = quantity.magnitude
        if operation == "divide":
            quantity = quantity / registry.Quantity(1, "centimeter")
        elif operation == "inplace":
            quantity /= registry.Quantity(1, "centimeter")
            assert quantity.magnitude is original
        else:
            quantity = quantity.to_reduced_units()
        results.append(quantity)
    xpint_result, pint_result = results
    assert str(xpint_result.units) == str(pint_result.units)
    assert xpint_result.magnitude.dtype == pint_result.magnitude.dtype
    np.testing.assert_allclose(xpint_result.magnitude, pint_result.magnitude, rtol=1e-15, atol=0.0)
