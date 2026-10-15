"""Lazy unit symbols remain bound to the documented registry."""

import xpint.units
from xpint.units import kg, microgram, ureg


def test_lazy_symbols_use_registry_canonical_units() -> None:
    assert kg == ureg.kilogram
    assert microgram == ureg.microgram
    assert xpint.units.kg is kg
    assert "kilogram" not in vars(ureg)
    assert (1 * microgram).to("gram").magnitude > 0


def test_undefined_symbols_are_attributes_not_units() -> None:
    assert not hasattr(xpint.units, "bogus")
    names = dir(xpint.units)
    assert "kg" in names
    assert "kilogram" in names
    assert "bogus" not in names


def test_undefined_symbol_raises_actionable_attribute_error() -> None:
    import pytest

    with pytest.raises(AttributeError, match="has no unit 'bogus'"):
        _ = xpint.units.bogus
