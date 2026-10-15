"""Registry formatting must not silently omit factor designations."""

from typing import Any

import numpy as np
import pint
import pytest

from xpint import SemanticRegistry


@pytest.mark.parametrize("entry", ["direct", "babel", "delegate"])
@pytest.mark.parametrize("quantity", [False, True])
def test_registry_formatter_rejects_designated_objects(entry: str, quantity: bool) -> None:
    u = SemanticRegistry()
    unit = u.mol["O2"]
    value = 5 * unit if quantity else unit
    method = "format_quantity" if quantity else "format_unit"
    formatter = u.formatter.get_formatter("") if entry == "delegate" else u.formatter
    kwargs = {"locale": "en_US"} if entry == "babel" else {}
    if entry == "babel":
        method += "_babel"
    with pytest.raises(TypeError, match=r"designated.*str\(\).*format\(\)"):
        getattr(formatter, method)(value, **kwargs)


def test_deprecated_babel_helpers_reject_designations() -> None:
    u = SemanticRegistry()
    for value in (u.mol["O2"], 5 * u.mol["O2"]):
        with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="designated"):
            value.format_babel(locale="en_US")


def test_plain_formatter_retains_defaults_and_locale() -> None:
    u = SemanticRegistry(fmt_locale="en_US")
    plain: pint.UnitRegistry[Any] = pint.UnitRegistry(fmt_locale="en_US")
    for registry in (u, plain):
        registry.formatter.default_format = ".2f~"
    assert format(5 * u.mol) == format(5 * plain.mol)
    assert u.formatter.format_quantity_babel(5 * u.mol) == (
        plain.formatter.format_quantity_babel(5 * plain.mol)
    )
    actual_formatter: Any = u.formatter
    expected_formatter: Any = plain.formatter
    assert actual_formatter.format_unit(u.mol) == expected_formatter.format_unit(plain.mol)


def test_designated_formatting_survives_registry_guard() -> None:
    u = SemanticRegistry()
    quantity = u.Quantity(np.array([1.0, 2.0], dtype=np.float64), u.mol["O2"])
    assert format(quantity, ".2f~") == "[1.00 2.00] mol[O2]"
    assert str(u.mol["O2"]) == "mole[O2]"
    assert format(5 * u.mol["O2"], ".2f~") == "5.00 mol[O2]"
