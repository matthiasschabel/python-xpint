"""Public registry adapters reuse Pint while guarding metadata extraction."""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pint
import pytest

from xpint import AffineRole, SemanticRegistry


def test_wraps_preserves_pint_conversion_keywords_defaults_and_metadata() -> None:
    registry = SemanticRegistry()

    @registry.wraps("meter", ("meter", None))
    def scale(value: Any, factor: Any = 2) -> Any:
        return value * factor

    result = scale(value=registry.Quantity(100, "centimeter"))
    assert result.units == registry.meter
    assert result.magnitude == pytest.approx(2, rel=1e-15, abs=0.0)
    assert scale.__name__ == "scale"

    @registry.wraps(None, None)
    def identity(value: Any) -> Any:
        return value

    quantity = registry.Quantity(1, registry.meter["tissue"], kind="Length")
    assert identity(quantity) is quantity


def test_wraps_relative_units_and_multiple_returns_match_pint() -> None:
    registry = SemanticRegistry()

    @registry.wraps(("=A", "=A**2"), "=A")
    def powers(value: Any) -> Any:
        return value, value**2

    result, squared = powers(registry.Quantity(3, "meter"))
    assert result.units == registry.meter
    assert squared.units == registry.meter**2
    assert squared.magnitude == 9


@pytest.mark.parametrize("classification", ["kind", "designation", "affine", "string"])
def test_wraps_rejects_classified_converted_inputs_before_callback(classification: str) -> None:
    registry = SemanticRegistry()
    called = []

    @registry.wraps("meter", "meter")
    def identity(value: Any) -> Any:
        called.append(True)
        return value

    values = {
        "kind": registry.Quantity(1, "meter", kind="Length"),
        "designation": registry.Quantity(1, registry.meter["tissue"]),
        "affine": registry.Quantity(20, "degC"),
        "string": "20 degC",
    }
    with pytest.raises(TypeError, match=r"cannot represent classified|absolute temperatures"):
        identity(values[classification])
    assert not called


def test_wraps_rejects_classified_declarations_and_foreign_inputs() -> None:
    registry = SemanticRegistry()
    with pytest.raises(TypeError, match="cannot represent classified"):
        registry.wraps(registry.meter["tissue"], "meter")
    with pytest.raises(TypeError, match="cannot represent classified"):
        registry.wraps("meter", "degC")
    foreign = SemanticRegistry()
    with pytest.raises(ValueError, match="different Pint registry"):
        registry.wraps("meter", foreign.meter)
    identity = registry.wraps("meter", "meter")(lambda value: value)
    with pytest.raises(ValueError, match="different Pint registry"):
        identity(foreign.Quantity(1, "meter"))


def test_check_keeps_classified_inputs_unchanged_and_checks_dimensions() -> None:
    registry = SemanticRegistry()
    identity = registry.check("[length]")(lambda value: value)
    quantity = registry.Quantity(1, registry.meter["tissue"], kind="Length")
    assert identity(quantity) is quantity
    with pytest.raises(pint.DimensionalityError, match="Cannot convert"):
        identity(registry.Quantity(1, "second"))
    with pytest.raises(ValueError, match="different Pint registry"):
        identity(SemanticRegistry().Quantity(1, "meter"))
    with pytest.raises(ValueError, match="different Pint registry"):
        registry.check(SemanticRegistry().meter)


def test_pi_theorem_matches_pint_for_physical_inputs() -> None:
    registry = SemanticRegistry()
    plain: pint.UnitRegistry[Any] = pint.UnitRegistry()
    definitions = {"distance": "meter", "speed": "meter/second", "time": "second"}
    actual = registry.pi_theorem(definitions)
    expected = plain.pi_theorem(definitions)  # type: ignore[no-untyped-call]
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected, strict=True):
        assert result == pytest.approx(reference, rel=1e-15, abs=0.0)
    with pytest.raises(TypeError, match="cannot represent classified"):
        registry.pi_theorem({"x": registry.meter["tissue"]})
    with pytest.raises(TypeError, match="cannot represent classified"):
        registry.pi_theorem({"x": "degC"})
    with pytest.raises(ValueError, match="different Pint registry"):
        registry.pi_theorem({"x": SemanticRegistry().meter})


def test_matplotlib_registration_converts_plain_values_and_rejects_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    from matplotlib import units as mpl_units
    from matplotlib.figure import Figure

    registry = SemanticRegistry()
    other = SemanticRegistry()
    other.setup_matplotlib()
    try:
        registry.setup_matplotlib()
        try:
            axes = Figure().add_subplot()
            values = registry.Quantity(np.array([1.0, 2.0]), "meter")
            axes.plot(values)
            np.testing.assert_allclose(
                np.asarray(axes.lines[0].get_ydata(orig=False), dtype=np.float64),
                [1, 2],
                rtol=0.0,
                atol=0.0,
            )
            # Matplotlib's float annotations omit its registered unit-converter inputs.
            axes.axhline(registry.Quantity(1.0, "meter"))  # type: ignore[arg-type]
            axes.set_ylim(
                registry.Quantity(0, "meter"),  # type: ignore[arg-type]
                registry.Quantity(3, "meter"),  # type: ignore[arg-type]
            )
            np.testing.assert_allclose(axes.get_ylim(), [0, 3], rtol=0.0, atol=0.0)
            converter = mpl_units.registry[registry.Quantity]
            classified = registry.Quantity(np.array([1.0, 2.0]), "meter", kind="Length")
            with pytest.raises(mpl_units.ConversionError, match="Failed to convert"):
                axes.plot(classified)
            with pytest.raises(TypeError, match="cannot represent classified"):
                Figure().add_subplot().plot(
                    registry.Quantity(np.array([1.0, 2.0]), registry.meter["tissue"])
                )
            with pytest.raises(TypeError, match="cannot represent classified"):
                converter.convert(classified, registry.meter, axes.yaxis)
            with pytest.raises(TypeError, match="cannot represent classified"):
                converter.convert(iter([classified]), registry.meter, axes.yaxis)
            with pytest.raises(TypeError, match="cannot represent classified"):
                converter.convert(values, registry.meter["tissue"], axes.yaxis)
            with pytest.raises(ValueError, match="different Pint registry"):
                converter.convert(other.Quantity(1, "meter"), registry.meter, axes.yaxis)
        finally:
            registry.setup_matplotlib(False)
        assert registry.Quantity not in mpl_units.registry
        assert other.Quantity in mpl_units.registry
    finally:
        other.setup_matplotlib(False)


def test_setup_matplotlib_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SemanticRegistry()
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError, match="install matplotlib first"):
        registry.setup_matplotlib()


def test_wraps_keeps_pint_string_and_non_strict_numeric_inputs() -> None:
    registry = SemanticRegistry()
    strict = registry.wraps("meter", "meter")(lambda value: value)
    loose = registry.wraps("meter", "meter", strict=False)(lambda value: value)
    assert strict("100 centimeter").magnitude == pytest.approx(1, rel=1e-15, abs=0.0)
    assert loose(2.0).magnitude == pytest.approx(2, rel=1e-15, abs=0.0)
    with pytest.raises(TypeError, match="takes 1 parameters"):
        registry.wraps("meter", ("meter", "meter"))(lambda value: value)


def test_check_preserves_default_and_keyword_arguments() -> None:
    registry = SemanticRegistry()
    default = registry.Quantity(1, "meter", kind="Length")

    @registry.check("[length]", None)
    def identity(value: Any = default, unused: Any = None) -> Any:
        return value, unused

    quantity, marker = identity(unused="marker")
    assert quantity is default
    assert marker == "marker"


@pytest.mark.parametrize(
    "units, role", [("degC", AffineRole.ABSOLUTE), ("delta_degC", AffineRole.DELTA)]
)
def test_wraps_return_declarations_construct_temperature_roles(
    units: str, role: AffineRole
) -> None:
    registry = SemanticRegistry()
    identity = registry.wraps(units, None)(lambda value: value)
    result = identity(20)
    assert result.units == registry.Unit(units)
    assert result.affine_role is role
    with pytest.raises(TypeError, match="cannot represent classified"):
        registry.wraps(None, units)


def test_wraps_non_strict_container_passthrough_matches_pint() -> None:
    registry = SemanticRegistry()
    classified = registry.Quantity(1, "meter", kind="Length")
    values = [classified]
    identity = registry.wraps(None, "meter", strict=False)(lambda value: value)
    assert identity(values) is values

    @registry.wraps(None, None)
    def collect(*values: Any) -> Any:
        return values

    plain = registry.Quantity(100, "centimeter")
    first, second = collect(plain, classified)
    assert first is plain
    assert second is classified


@pytest.mark.parametrize("classification", ["kind", "designation", "affine", "string", "foreign"])
def test_wraps_rejects_later_variadic_conversion_before_callback(classification: str) -> None:
    registry = SemanticRegistry()
    called = []

    @registry.wraps(None, "meter")
    def collect(*values: Any) -> Any:
        called.append(True)
        return values

    values = {
        "kind": registry.Quantity(1, "meter", kind="Length"),
        "designation": registry.Quantity(1, registry.meter["tissue"]),
        "affine": registry.Quantity(20, "degC"),
        "string": "20 degC",
        "foreign": SemanticRegistry().Quantity(1, "meter"),
    }
    error = ValueError if classification == "foreign" else TypeError
    message = (
        "different Pint registry"
        if classification == "foreign"
        else "cannot represent classified|absolute temperatures"
    )
    with pytest.raises(error, match=message):
        collect(registry.Quantity(1, "meter"), values[classification])
    assert not called


def test_wraps_accepts_plain_variadic_values() -> None:
    registry = SemanticRegistry()

    @registry.wraps(None, "meter")
    def collect(*values: Any) -> Any:
        return values

    quantity = registry.Quantity(200, "centimeter")
    first, second = collect(quantity, quantity)
    assert first == pytest.approx(2, rel=1e-15, abs=0.0)
    # Older Pint converts only the first value; a corrected converter converts both.
    assert second is quantity or second == pytest.approx(2, rel=1e-15, abs=0.0)


def test_wraps_omitted_kwargs_specification_follows_pint() -> None:
    registry = SemanticRegistry()
    plain: pint.UnitRegistry[Any] = pint.UnitRegistry()

    def collect(*values: Any, **options: Any) -> Any:
        return values, options

    try:
        plain.wraps(None, "meter")(collect)
    except TypeError:
        with pytest.raises(TypeError, match="parameters"):
            registry.wraps(None, "meter")(collect)
        return

    wrapped = registry.wraps(None, "meter")(collect)
    values, options = wrapped(registry.Quantity(200, "centimeter"), label="x")
    assert values == pytest.approx((2,), rel=1e-15, abs=0.0)
    assert options == {"label": "x"}
    with pytest.raises(TypeError, match="cannot represent classified"):
        wrapped(registry.Quantity(1, "meter"), registry.Quantity(2, "meter", kind="Length"))


@pytest.mark.parametrize("unit", ["degC", "delta_degC"])
def test_pi_theorem_mapping_and_container_inputs_reject_implicit_roles(unit: str) -> None:
    registry = SemanticRegistry()
    for value in ({unit: 1}, registry.UnitsContainer({unit: 1})):
        with pytest.raises(TypeError, match="cannot represent classified"):
            registry.pi_theorem({"x": value})
    assert registry.pi_theorem({"x": {"[temperature]": 1}}) == []


def test_wraps_guards_pint_variadic_keyword_only_conversion() -> None:
    registry = SemanticRegistry()
    classified = registry.Quantity(1, "meter", kind="Length")
    called = []

    @registry.wraps(None, (None, "meter"))
    def collect(*values: Any, scale: Any) -> Any:
        called.append(True)
        return values, scale

    # Match the installed Pint converter's public pass-through behavior.
    plain: pint.UnitRegistry[Any] = pint.UnitRegistry()
    probe = plain.Quantity(1, "meter")

    @plain.wraps(None, (None, "meter"))
    def plain_collect(*values: Any, scale: Any) -> Any:
        return values, scale

    plain_values, _ = plain_collect(1, probe, scale=plain.Quantity(2, "meter"))
    if plain_values[1] is probe:
        values, scale = collect(1, classified, scale=registry.Quantity(200, "centimeter"))
        assert values[1] is classified
        assert scale == pytest.approx(2, rel=1e-15, abs=0.0)
        assert called == [True]
        called.clear()
    else:
        with pytest.raises(TypeError, match="cannot represent classified"):
            collect(1, classified, scale=registry.Quantity(2, "meter"))
    with pytest.raises(TypeError, match="cannot represent classified"):
        collect(1, registry.Quantity(200, "centimeter"), scale=classified)
    assert not called


def test_wraps_rejects_classified_keyword_only_default_after_variadic() -> None:
    registry = SemanticRegistry()
    classified = registry.Quantity(1, "meter", kind="Length")
    called = []

    @registry.wraps(None, (None, "meter"))
    def collect(*values: Any, scale: Any = classified) -> Any:
        called.append(True)
        return values, scale

    with pytest.raises(TypeError, match="cannot represent classified"):
        collect(1, registry.Quantity(200, "centimeter"))
    assert not called
