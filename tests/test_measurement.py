"""Measurement adapters must not discard semantic state."""

import operator
from collections.abc import Callable
from typing import Any

import numpy as np
import pint
import pytest
from uncertainties import ufloat  # type: ignore[import-untyped]

from xpint import SemanticRegistry


@pytest.mark.parametrize("classification", ["kind", "designation", "absolute", "delta"])
@pytest.mark.parametrize("entry", ["plus_minus", "value", "error"])
def test_measurement_rejects_classified_values(classification: str, entry: str) -> None:
    u = SemanticRegistry()
    quantities = {
        "kind": (5 * u.mol).as_kind("AmountOfSubstance"),
        "designation": 5 * u.mol["O2"],
        "absolute": (20 * u.degC).to("K"),
        "delta": 5 * u.delta_degC,
    }
    quantity = quantities[classification]
    with pytest.raises(TypeError, match=r"cannot represent classified.*explicitly remove"):
        if entry == "plus_minus":
            quantity.plus_minus(0.1)
        elif entry == "value":
            u.Measurement(quantity, 0.1)
        else:
            u.Measurement(5, quantity, quantity.units)


def test_plus_minus_rejects_classified_error_and_foreign_error() -> None:
    u = SemanticRegistry()
    with pytest.raises(TypeError, match="cannot represent classified"):
        (5 * u.mol).plus_minus(0.1 * u.mol["O2"])
    foreign = SemanticRegistry()
    with pytest.raises(ValueError, match="different Pint registry"):
        (5 * u.mol).plus_minus(0.1 * foreign.mol)


@pytest.mark.parametrize("unit_spec", ["degC", "delta_degC", "designated"])
def test_measurement_rejects_classified_unit_declarations(unit_spec: str) -> None:
    u = SemanticRegistry()
    units = u.mol["O2"] if unit_spec == "designated" else unit_spec
    with pytest.raises(TypeError, match="cannot represent classified"):
        u.Measurement(5, 0.1, units)
    with pytest.raises(TypeError, match="cannot represent classified"):
        u.Measurement(ufloat(5, 0.1), units)


@pytest.mark.parametrize("argument", ["value", "error", "units"])
def test_measurement_rejects_foreign_registry_arguments(argument: str) -> None:
    u = SemanticRegistry()
    foreign = SemanticRegistry()
    inputs: dict[str, Any] = {"value": 5, "error": 0.1, "units": u.mol}
    inputs[argument] = foreign.mol if argument == "units" else 1 * foreign.mol
    with pytest.raises(ValueError, match="different Pint registry"):
        u.Measurement(**inputs)


def test_plain_measurement_preserves_pint_behavior() -> None:
    u = SemanticRegistry()
    plain: pint.UnitRegistry[Any] = pint.UnitRegistry()
    actual = (5 * u.mol).plus_minus(2, relative=True)
    expected = (5 * plain.mol).plus_minus(2, relative=True)
    # Only scalar construction and conversion rounding is involved.
    assert actual.value.magnitude == pytest.approx(expected.value.magnitude, rel=1e-15, abs=0)
    assert actual.error.magnitude == pytest.approx(expected.error.magnitude, rel=1e-15, abs=0)
    assert actual.value.units == u.mol
    assert u.Measurement(5 * u.mol, 100 * u.mmol).error.magnitude == pytest.approx(
        0.1, rel=1e-15, abs=0
    )
    assert u.Measurement(ufloat(5, 0.1), "mol").value.magnitude == pytest.approx(
        5, rel=1e-15, abs=0
    )


@pytest.mark.parametrize("operation", [operator.add, operator.mul, operator.truediv, operator.iadd])
def test_measurement_arithmetic_rejects_classified_operand(
    operation: Callable[[Any, Any], Any],
) -> None:
    u = SemanticRegistry()
    measurement = (5 * u.J).plus_minus(0.1)
    for quantity in ((1 * u.J).as_kind("Energy"), 1 * u.mol["O2"]):
        with pytest.raises(TypeError, match="cannot represent classified"):
            operation(measurement, quantity)
    assert measurement.value.magnitude == pytest.approx(5, rel=1e-15, abs=0)


def test_measurement_conversion_and_comparison_reject_classified_targets() -> None:
    u = SemanticRegistry()
    measurement = (5 * u.mol).plus_minus(0.1)
    with pytest.raises(TypeError, match="cannot represent classified"):
        measurement.to(u.mol["O2"])
    with pytest.raises(TypeError, match="cannot represent classified"):
        measurement.ito(u.mol["O2"])
    with pytest.raises(TypeError, match="cannot represent classified"):
        _ = measurement < 5 * u.mol["O2"]
    assert measurement != 5 * u.mol["O2"]
    assert measurement != 5 * SemanticRegistry().mol
    assert measurement.to("mmol").value.magnitude == pytest.approx(5000, rel=1e-15, abs=0)


@pytest.mark.parametrize("operation", [operator.mul, operator.truediv, np.multiply, np.add])
@pytest.mark.parametrize("units", ["", "second"])
@pytest.mark.parametrize("classification", ["kind", "designation", "absolute", "delta"])
def test_classified_quantity_rejects_measurement_operands(
    operation: Callable[[Any, Any], Any], units: str, classification: str
) -> None:
    u = SemanticRegistry()
    quantity = {
        "kind": (5 * u.J).as_kind("Energy"),
        "designation": 5 * u.mol["O2"],
        "absolute": (20 * u.degC).to("K"),
        "delta": 5 * u.delta_degC,
    }[classification]
    measurement = u.Measurement(1, 0.1, units)
    with pytest.raises(TypeError, match=r"construct Quantity\(measurement.magnitude"):
        operation(quantity, measurement)


def test_explicit_uncertainty_quantity_uses_normal_kind_propagation() -> None:
    u = SemanticRegistry()
    measurement = u.Measurement(2, 0.1, "second")
    quantity = (5 * u.J).as_kind("Energy")
    result = quantity / u.Quantity(measurement.magnitude, measurement.units)
    assert result.kind is None
    assert result.units == u.J / u.second
    assert result.magnitude.nominal_value == pytest.approx(2.5, rel=1e-15, abs=0)


def test_plain_measurements_remain_dictionary_keys() -> None:
    u = SemanticRegistry()
    measurement = u.Measurement(5, 0.1, "mol")
    mapping = {measurement: "sample"}
    assert mapping[measurement] == "sample"


@pytest.mark.parametrize("operation", [operator.pow, operator.ipow])
def test_measurement_rejects_classified_exponents(operation: Callable[[Any, Any], Any]) -> None:
    u = SemanticRegistry()
    exponent = 2 * u.mL["blood"] / u.mL["tissue"]
    with pytest.raises(TypeError, match="cannot represent classified"):
        operation(u.Measurement(5, 0.1), exponent)
