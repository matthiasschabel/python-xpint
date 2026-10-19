"""Semantic registry ownership and kind-catalog tests."""

import operator
from typing import Any

import numpy as np
import pint
import pytest
from pint.errors import UndefinedUnitError

from xpint import AffineRole, SemanticRegistry, get_default_registry
from xpint.units import J, m, s, ureg


@pytest.mark.parametrize("quantity", [False, True])
def test_compatible_kinds_rejects_foreign_registry(quantity: bool) -> None:
    foreign = SemanticRegistry()
    target = foreign.Quantity(1, "meter") if quantity else foreign.meter

    with pytest.raises(ValueError, match="target belongs to another registry; use this registry"):
        ureg.compatible_kinds(target)


@pytest.mark.parametrize("quantity", [False, True])
def test_compatible_kinds_rejects_ordinary_pint_objects(quantity: bool) -> None:
    plain: pint.UnitRegistry[Any] = pint.UnitRegistry()
    target = plain.Quantity(1, "meter") if quantity else plain.meter

    with pytest.raises(
        TypeError, match=r"target must be a unit expression string or.*this registry"
    ):
        ureg.compatible_kinds(target)  # type: ignore[arg-type]


def test_compatible_kinds_rejects_unsupported_type() -> None:
    with pytest.raises(
        TypeError, match=r"target must be a unit expression string or.*this registry"
    ):
        ureg.compatible_kinds(123)  # type: ignore[arg-type]


def test_compatible_kinds_propagates_undefined_unit() -> None:
    with pytest.raises(UndefinedUnitError, match="not_a_registered_unit"):
        ureg.compatible_kinds("not_a_registered_unit")


def test_default_symbols_share_one_registry() -> None:
    speed = 100 * m / (9.58 * s)

    assert get_default_registry() is ureg
    assert isinstance(speed, ureg.Quantity)
    assert speed.to("meter / second").magnitude == pytest.approx(
        100 / 9.58,
        rel=1e-15,
        abs=0.0,
    )


@pytest.mark.parametrize("constructor", ["from_sequence", "from_list"])
@pytest.mark.parametrize(
    ("unit", "kind", "target", "expected", "expected_kind"),
    [
        ("joule", "Energy", "kilojoule", [0.001, 0.002], "Energy"),
        ("kelvin", "Temperature", "kelvin", [1.0, 2.0], "Temperature"),
        ("degC", None, "kelvin", [274.15, 275.15], "Temperature"),
        ("delta_degC", None, "kelvin", [1.0, 2.0], "TemperatureDifference"),
        ("kelvin", "Temperature", None, [1.0, 2.0], "Temperature"),
    ],
)
def test_quantity_sequence_preserves_semantic_state(
    constructor: str,
    unit: str,
    kind: str | None,
    target: str | None,
    expected: list[float],
    expected_kind: str,
) -> None:
    sequence = [ureg.Quantity(value, unit, kind=kind) for value in (1, 2)]
    result = getattr(ureg.Quantity, constructor)(sequence, units=target)

    assert result.kind == expected_kind
    assert result.affine_role == sequence[0].affine_role
    assert result.shape == (2,)
    assert result.magnitude.dtype == np.dtype(np.float64)
    # Only unit conversion roundoff is expected.
    np.testing.assert_allclose(result.magnitude, expected, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("other_kind", ["Torque", None])
def test_quantity_sequence_rejects_mixed_kinds(other_kind: str | None) -> None:
    sequence = [ureg.Quantity(1, "J", kind="Energy"), ureg.Quantity(2, "J", kind=other_kind)]
    with pytest.raises(TypeError, match="different kinds"):
        ureg.Quantity.from_sequence(sequence)


def test_quantity_sequence_rejects_mixed_affine_roles() -> None:
    sequence = [ureg.Quantity(1, "K", kind="Temperature"), ureg.Quantity(2, "K")]
    with pytest.raises(TypeError, match="different affine roles"):
        ureg.Quantity.from_sequence(sequence)


def test_quantity_sequence_preserves_designation() -> None:
    sequence = [ureg.Quantity(value, ureg.millimole["O2"]) for value in (1, 2)]
    result = ureg.Quantity.from_sequence(sequence, ureg.mole["O2"])
    assert result.designation_signature == sequence[0].designation_signature
    np.testing.assert_allclose(result.magnitude, [0.001, 0.002], rtol=1e-15, atol=0.0)
    inferred = ureg.Quantity.from_sequence(sequence)
    assert inferred.kind is None
    assert inferred.affine_role is None
    assert inferred.units == ureg.millimole["O2"]
    assert inferred.designation_signature == sequence[0].designation_signature
    plain_target = ureg.Quantity.from_sequence(sequence, "mole")
    assert plain_target.units == ureg.mole["O2"]
    np.testing.assert_allclose(plain_target.magnitude, [0.001, 0.002], rtol=1e-15, atol=0.0)
    with pytest.raises(TypeError, match="different factor designations"):
        ureg.Quantity.from_sequence([sequence[0], ureg.Quantity(2, ureg.mole["CO2"])])


def test_quantity_sequence_validates_ownership_and_elements() -> None:
    foreign = SemanticRegistry()
    with pytest.raises(ValueError, match="different Pint registry"):
        ureg.Quantity.from_sequence([foreign.Quantity(1, "m")])
    with pytest.raises(ValueError, match="different Pint registry"):
        ureg.Quantity.from_sequence([], foreign.meter)
    with pytest.raises(TypeError, match="sequence elements must be quantities"):
        ureg.Quantity.from_sequence([1], "m")


def test_quantity_sequence_empty_and_ordinary_pint_behavior() -> None:
    empty = ureg.Quantity.from_sequence([], "m")
    assert empty.shape == (0,)
    assert empty.magnitude.dtype == np.dtype(np.float64)
    with pytest.raises(ValueError, match="empty sequence"):
        ureg.Quantity.from_sequence([])
    result = ureg.Quantity.from_sequence((1 * ureg.m, 2 * ureg.cm))
    assert result.kind is None
    np.testing.assert_allclose(result.magnitude, [1.0, 0.02], rtol=1e-15, atol=0.0)
    with pytest.raises(pint.DimensionalityError):
        ureg.Quantity.from_sequence([1 * ureg.m, 1 * ureg.s])


def test_conversion_accepts_pint_mapping_unit_syntax() -> None:
    converted = (1 * m).to({"centimeter": 1})

    assert converted.magnitude == pytest.approx(100, rel=1e-15, abs=0.0)


def test_constructing_registry_does_not_replace_application_registry() -> None:
    application_registry = pint.get_application_registry()  # type: ignore[no-untyped-call]

    SemanticRegistry()

    assert pint.get_application_registry() is application_registry  # type: ignore[no-untyped-call]


def test_registry_forwards_pint_configuration_options() -> None:
    registry = SemanticRegistry(case_sensitive=False)

    assert registry.Unit("METER") == registry.meter


@pytest.mark.parametrize(
    ("value", "type_name"),
    [("yes", "str"), (np.bool_(True), "numpy.bool")],
)
def test_registry_rejects_non_boolean_angle_dimension(value: object, type_name: str) -> None:
    with pytest.raises(
        TypeError,
        match=rf"angle_dimension must be a bool, got {type_name}",
    ):
        SemanticRegistry(angle_dimension=value)  # type: ignore[arg-type]


def test_cross_registry_quantity_construction_is_rejected() -> None:
    left_registry = SemanticRegistry()
    right_registry = SemanticRegistry()
    right = right_registry.Quantity(1, right_registry.meter)

    with pytest.raises(ValueError, match="different Pint registry"):
        left_registry.Quantity(right)


@pytest.mark.parametrize("source_unit", ["degC", "delta_degC"])
@pytest.mark.parametrize(
    ("magnitude", "absolute_kelvin"),
    [
        (20, 293.15),
        (
            np.array([20.0, 25.0], dtype=np.float64),
            np.array([293.15, 298.15], dtype=np.float64),
        ),
    ],
)
def test_quantity_reconstruction_preserves_converted_affine_role(
    source_unit: str, magnitude: Any, absolute_kelvin: Any
) -> None:
    source = ureg.Quantity(magnitude, source_unit)
    expected = source.to("kelvin")

    rebuilt = ureg.Quantity(source, "kelvin")

    assert rebuilt.affine_role is expected.affine_role
    assert rebuilt.kind == expected.kind
    # Both routes perform the same scale/offset conversion; allow float rounding.
    np.testing.assert_allclose(rebuilt.magnitude, expected.magnitude, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(
        rebuilt.magnitude,
        absolute_kelvin if source_unit == "degC" else magnitude,
        rtol=1e-15,
        atol=0.0,
    )
    if rebuilt.affine_role is AffineRole.ABSOLUTE:
        with pytest.raises(TypeError, match="absolute temperatures"):
            rebuilt * 2


@pytest.mark.parametrize(
    ("target_unit", "expected_magnitude"),
    [("millimole", [1, 2]), ("mole", [0.001, 0.002])],
)
def test_quantity_reconstruction_preserves_designated_target(
    target_unit: str, expected_magnitude: list[float]
) -> None:
    source = ureg.Quantity(
        np.array([1, 2], dtype=np.int64), ureg.millimole["O2"], kind="AmountOfSubstance"
    )
    target = ureg.Unit(target_unit)["O2"]
    expected = source.to(target)

    rebuilt = ureg.Quantity(source, target)

    assert rebuilt.units == target
    assert rebuilt.kind == expected.kind
    assert rebuilt.designation_signature == source.designation_signature
    assert rebuilt.shape == expected.shape
    assert rebuilt.magnitude.dtype == expected.magnitude.dtype
    # Identity conversion intentionally shares the source buffer, as Pint does.
    assert np.shares_memory(rebuilt.magnitude, source.magnitude) == np.shares_memory(
        expected.magnitude, source.magnitude
    )
    np.testing.assert_allclose(rebuilt.magnitude, expected.magnitude, rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(rebuilt.magnitude, expected_magnitude, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize(
    ("source_designation", "target_designation"),
    [(None, "O2"), ("O2", None), ("O2", "CO2")],
)
def test_quantity_reconstruction_rejects_designation_changes(
    source_designation: str | None, target_designation: str | None
) -> None:
    source_unit = ureg.mole if source_designation is None else ureg.mole[source_designation]
    target_unit = ureg.mole if target_designation is None else ureg.mole[target_designation]
    source = ureg.Quantity(2, source_unit)

    if target_designation is None:
        rebuilt = ureg.Quantity(source, target_unit)
        assert rebuilt.designation_signature == source.designation_signature
    else:
        with pytest.raises(TypeError, match="different factor designations"):
            ureg.Quantity(source, target_unit)


def test_existing_quantity_designation_attachment_is_explicit() -> None:
    assert ureg.Quantity(2, ureg.mole).with_designation("O2").units == ureg.mole["O2"]


def test_quantity_reconstruction_preserves_compound_designation() -> None:
    source = ureg.Quantity(2, ureg.millimole["O2"] / ureg.liter)
    target = ureg.mole["O2"] / ureg.meter**3

    rebuilt = ureg.Quantity(source, target)

    assert rebuilt.units == target
    assert rebuilt.magnitude == pytest.approx(2, rel=1e-15, abs=0.0)


def test_quantity_reconstruction_accepts_an_explicit_kind_override() -> None:
    source = ureg.Quantity(2000, "joule", kind="Energy")

    rebuilt = ureg.Quantity(source, "kilojoule", kind="Torque")

    assert rebuilt.kind == "Torque"
    assert rebuilt.magnitude == pytest.approx(2, rel=1e-15, abs=0.0)
    assert source.kind == "Energy"


def test_cross_registry_arithmetic_is_rejected() -> None:
    left_registry = SemanticRegistry()
    right_registry = SemanticRegistry()
    left = left_registry.Quantity(1, left_registry.meter)
    right = right_registry.Quantity(1, right_registry.meter)

    with pytest.raises(ValueError, match="different SemanticRegistry"):
        left + right


def test_ordinary_pint_quantity_is_rejected() -> None:
    ordinary_registry: pint.UnitRegistry[Any] = pint.UnitRegistry()
    semantic = 1 * J
    ordinary = ordinary_registry.Quantity(1, "joule")

    with pytest.raises(ValueError, match="ordinary Pint quantities"):
        semantic + ordinary


def test_quantity_equality_rejects_plain_pint_and_returns_false_for_foreign_xpint() -> None:
    ordinary_registry: pint.UnitRegistry[Any] = pint.UnitRegistry()
    other_registry = SemanticRegistry()
    semantic = 1 * J
    ordinary = ordinary_registry.Quantity(1, "joule")

    with pytest.raises(ValueError, match="cannot be compared with ordinary Pint quantities"):
        _ = semantic == ordinary
    # Pint owns the reverse dispatch and compares only physical value and units. This
    # remains an audited interop limitation until Pint checks registry ownership or
    # xpint replaces inheritance with composition.
    assert ordinary == semantic
    assert semantic != other_registry.Quantity(1, "joule")


def test_compatibility_returns_false_outside_the_owning_registry() -> None:
    ordinary_registry: pint.UnitRegistry[Any] = pint.UnitRegistry()
    other_registry = SemanticRegistry()
    semantic = 1 * J

    assert not semantic.is_compatible_with(ordinary_registry.Quantity(1, "joule"))
    assert not semantic.is_compatible_with(other_registry.Quantity(1, "joule"))
    assert not J.is_compatible_with(ordinary_registry.joule)
    assert not J.is_compatible_with(other_registry.joule)


def test_define_kind_validates_and_is_idempotent() -> None:
    registry = SemanticRegistry()

    first = registry.define_kind("Momentum", "kilogram * meter / second")
    second = registry.define_kind("Momentum", "newton * second")

    assert first is second
    assert registry.kinds["Momentum"] is first


def test_define_kind_accepts_dimensionless_reference() -> None:
    registry = SemanticRegistry()

    registry.define_kind("Probability", "dimensionless")
    probability = registry.Quantity(0.5).as_kind("Probability")

    assert probability.kind == "Probability"
    assert probability.dimensionless


def test_predefined_kind_catalog_uses_paired_names() -> None:
    expected = {
        "Energy",
        "Torque",
        "Temperature",
        "TemperatureDifference",
        "ElectricPotential",
        "MagneticFluxDensity",
        "CatalyticActivity",
    }

    assert expected <= ureg.kinds.keys()


@pytest.mark.parametrize(
    ("name", "reference", "error", "message"),
    [
        ("", "meter", ValueError, "must not be empty"),
        ("LengthLike", "", ValueError, "must not be empty"),
        (3, "meter", TypeError, "must both be strings"),
    ],
)
def test_define_kind_rejects_invalid_input(
    name: object,
    reference: str,
    error: type[Exception],
    message: str,
) -> None:
    registry = SemanticRegistry()

    with pytest.raises(error, match=message):
        registry.define_kind(name, reference)  # type: ignore[arg-type]


def test_define_kind_rejects_conflicting_dimension() -> None:
    registry = SemanticRegistry()
    registry.define_kind("Momentum", "kilogram * meter / second")

    with pytest.raises(ValueError, match="already constrained"):
        registry.define_kind("Momentum", "joule")


@pytest.mark.parametrize("angle_dimension", [False, True])
@pytest.mark.parametrize("reference", ["kelvin", "degC", "delta_degF"])
def test_define_kind_rejects_custom_temperature_kinds(
    angle_dimension: bool, reference: str
) -> None:
    registry = SemanticRegistry(angle_dimension=angle_dimension)

    with pytest.raises(ValueError, match="cannot be constrained to thermodynamic temperature"):
        registry.define_kind("SensorTemperature", reference)
    assert "SensorTemperature" not in registry.kinds


def test_define_kind_accepts_builtin_temperature_kinds_idempotently() -> None:
    registry = SemanticRegistry()

    assert registry.define_kind("Temperature", "kelvin") == registry.kinds["Temperature"]
    assert (
        registry.define_kind("TemperatureDifference", "delta_degC")
        == registry.kinds["TemperatureDifference"]
    )


@pytest.mark.parametrize("kind", [None, "Length"])
def test_in_place_power_preserves_array_alias_and_refreshes_dimensions(kind: str | None) -> None:
    quantity = ureg.Quantity(np.array([2.0, 3.0], dtype=np.float64), m["tissue"], kind=kind)
    alias = quantity.magnitude
    _ = quantity.dimensionality
    quantity **= 2
    assert quantity.magnitude is alias
    assert quantity.dimensionality == (m**2).dimensionality
    assert quantity.units == m["tissue"] ** 2
    assert quantity.kind is None
    np.testing.assert_allclose(alias, [4.0, 9.0], rtol=0.0, atol=0.0)


def test_in_place_scalar_power_follows_pint_replacement_behavior() -> None:
    quantity = ureg.Quantity(2, m)
    alias = quantity
    _ = quantity.dimensionality
    quantity **= 2
    assert quantity is not alias
    assert alias.units == m
    assert alias.magnitude == 2
    assert quantity.units == m**2
    assert quantity.dimensionality == (m**2).dimensionality
    assert quantity.magnitude == 4


@pytest.mark.parametrize("kind", [None, "Length"])
def test_in_place_power_cast_failure_preserves_complete_state(kind: str | None) -> None:
    quantity = ureg.Quantity(np.array([4, 9], dtype=np.int64), m["tissue"], kind=kind)
    alias = quantity.magnitude
    _ = quantity.dimensionality
    with pytest.raises(TypeError):
        quantity **= 0.5
    assert quantity.magnitude is alias
    assert quantity.units == m["tissue"]
    assert quantity.dimensionality == m.dimensionality
    assert quantity.kind == kind
    np.testing.assert_array_equal(alias, [4, 9])


@pytest.mark.parametrize("operation", ["ito", "ito_base_units", "ito_root_units"])
def test_in_place_conversion_preserves_alias_and_semantic_factors(operation: str) -> None:
    quantity = ureg.Quantity(
        np.array([100.0, 200.0], dtype=np.float64), ureg.centimeter["tissue"], kind="Length"
    )
    alias = quantity.magnitude
    _ = quantity.dimensionality
    if operation == "ito":
        quantity.ito(m["tissue"])
    else:
        getattr(quantity, operation)()
    assert quantity.magnitude is alias
    assert quantity.units == m["tissue"]
    assert quantity.kind == "Length"
    assert quantity.dimensionality == m.dimensionality
    np.testing.assert_allclose(alias, [1.0, 2.0], rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("operation", ["ito", "ito_base_units", "ito_root_units"])
def test_in_place_conversion_cast_failure_preserves_complete_state(operation: str) -> None:
    quantity = ureg.Quantity(
        np.array([100, 200], dtype=np.int64), ureg.centimeter["tissue"], kind="Length"
    )
    alias = quantity.magnitude
    _ = quantity.dimensionality
    with pytest.raises(TypeError):
        if operation == "ito":
            quantity.ito(m["tissue"])
        else:
            getattr(quantity, operation)()
    assert quantity.magnitude is alias
    assert quantity.units == ureg.centimeter["tissue"]
    assert quantity.kind == "Length"
    assert quantity.dimensionality == m.dimensionality
    np.testing.assert_array_equal(alias, [100, 200])


def test_in_place_context_conversion_refreshes_dimensionality() -> None:
    quantity = ureg.Quantity(500.0, "nm")
    _ = quantity.dimensionality
    quantity.ito("Hz", "sp")
    assert quantity.dimensionality == ureg.hertz.dimensionality
    assert quantity.magnitude == pytest.approx(299792458 / 500e-9, rel=1e-15, abs=0.0)


def test_in_place_context_conversion_rejects_kind_change_before_mutation() -> None:
    quantity = ureg.Quantity(np.array([500.0, 600.0]), "nm", kind="Length")
    alias = quantity.magnitude
    _ = quantity.dimensionality
    with pytest.raises(TypeError, match=r"as_unkinded\(\)"):
        quantity.ito("Hz", "sp")
    assert quantity.units == ureg.nanometer
    assert quantity.kind == "Length"
    assert quantity.magnitude is alias
    assert quantity.dimensionality == m.dimensionality
    np.testing.assert_allclose(alias, [500.0, 600.0], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("kind", [None, "Length"])
@pytest.mark.parametrize("divide", [False, True])
def test_in_place_multiplication_refreshes_dimensions_and_factors(
    kind: str | None, divide: bool
) -> None:
    quantity = ureg.Quantity(np.array([2.0, 3.0]), m, kind=kind)
    alias = quantity.magnitude
    _ = quantity.dimensionality
    if divide:
        quantity /= ureg.Quantity(2, "s")
        assert quantity.dimensionality == (m / ureg.second).dimensionality
        cancelled = quantity.units * ureg.second
        expected = [1.0, 1.5]
    else:
        quantity *= ureg.Quantity(2, "s")
        assert quantity.dimensionality == (m * ureg.second).dimensionality
        cancelled = quantity.units / ureg.second
        expected = [4.0, 6.0]
    assert str(cancelled["tissue"]) == "meter[tissue]"
    assert quantity.kind is None
    assert quantity.magnitude is alias
    np.testing.assert_allclose(alias, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("operation", ["power", "convert"])
@pytest.mark.parametrize("identity", [False, True])
def test_in_place_operations_respect_read_only_buffers(operation: str, identity: bool) -> None:
    magnitude = np.array([2.0, 3.0])
    magnitude.flags.writeable = False
    quantity = ureg.Quantity(magnitude, m, kind="Length")
    if identity:
        if operation == "power":
            quantity **= 1
        else:
            quantity.ito(m)
    else:
        with pytest.raises(ValueError, match="read-only"):
            if operation == "power":
                quantity **= 2
            else:
                quantity.ito("cm")
    assert quantity.magnitude is magnitude
    assert quantity.units == m
    assert quantity.kind == "Length"
    np.testing.assert_allclose(magnitude, [2.0, 3.0], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("operation", ["power", "convert"])
def test_in_place_numeric_failure_preserves_values_and_metadata(operation: str) -> None:
    quantity = ureg.Quantity(np.array([1e308], dtype=np.float64), m, kind="Length")
    alias = quantity.magnitude
    with np.errstate(over="raise"), pytest.raises(FloatingPointError, match="overflow"):
        if operation == "power":
            quantity **= 2
        else:
            quantity.ito("cm")
    assert quantity.magnitude is alias
    assert quantity.units == m
    assert quantity.kind == "Length"
    np.testing.assert_allclose(alias, [1e308], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("basis", ["base", "root"])
@pytest.mark.parametrize(
    ("unit", "kind"),
    [
        ("degC", None),
        ("delta_degC", None),
        ("degF", None),
        ("degR", None),
        ("kelvin", None),
        ("kelvin", "Temperature"),
        ("kelvin", "TemperatureDifference"),
        ("delta_degC / second", None),
    ],
)
def test_temperature_in_place_basis_conversion_matches_copy(
    basis: str, unit: str, kind: str | None
) -> None:
    quantity = ureg.Quantity(np.array([20.0, 21.0], dtype=np.float64), unit, kind=kind)
    expected = getattr(quantity, f"to_{basis}_units")()
    alias = quantity.magnitude
    getattr(quantity, f"ito_{basis}_units")()
    assert quantity.magnitude is alias
    assert quantity.units == expected.units
    assert quantity.kind == expected.kind
    assert quantity.affine_role == expected.affine_role
    assert quantity.designation_signature == expected.designation_signature
    np.testing.assert_allclose(quantity.magnitude, expected.magnitude, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("classification", ["kind", "designation", "affine"])
@pytest.mark.parametrize("method", ["view", "byteswap", "getfield", "tobytes", "__dlpack__"])
def test_classified_raw_magnitude_methods_fail_closed(method: str, classification: str) -> None:
    magnitude = np.array([1.0, 2.0], dtype=np.float64)
    quantity = ureg.Quantity(magnitude, ureg.meter, kind="Length")
    if classification == "designation":
        quantity = ureg.Quantity(magnitude, ureg.meter["tissue"])
    elif classification == "affine":
        quantity = ureg.Quantity(magnitude, "degC")
    kwargs = {"inplace": True} if method == "byteswap" else {}
    with pytest.raises(TypeError, match="magnitude explicitly"):
        getattr(quantity, method)(**kwargs)
    np.testing.assert_allclose(magnitude, [1, 2], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("attribute", ["data", "base", "ctypes"])
def test_classified_raw_storage_attributes_fail_closed(attribute: str) -> None:
    quantity = ureg.Quantity(np.array([1.0, 2.0]), ureg.meter, kind="Length")
    with pytest.raises(AttributeError, match="magnitude explicitly"):
        getattr(quantity, attribute)


def test_classified_array_metadata_and_wrapped_methods_remain_available() -> None:
    quantity = ureg.Quantity(np.array([[1, 2]], dtype=np.int64), ureg.meter, kind="Length")
    assert quantity.shape == (1, 2)
    assert quantity.dtype == np.dtype(np.int64)
    assert quantity.ndim == 2
    assert quantity.size == 2
    assert quantity.flatten().kind == "Length"
    assert quantity.astype(np.float64).kind == "Length"
    assert not hasattr(quantity, "missing_attribute")


def test_ordinary_raw_magnitude_forwarding_matches_pint() -> None:
    magnitude = np.array([1, 2], dtype=np.int64)
    quantity = ureg.Quantity(magnitude, ureg.meter)
    assert isinstance(quantity.view(), np.ndarray)
    assert quantity.ctypes.data == magnitude.ctypes.data


def test_classified_masked_array_metadata_is_readable() -> None:
    magnitude = np.ma.array([1, 2], mask=[False, True], dtype=np.int64)
    quantity = ureg.Quantity(magnitude, ureg.meter["tissue"])
    for name in ("strides", "nbytes", "itemsize", "hardmask", "sharedmask", "fill_value"):
        assert getattr(quantity, name) == getattr(magnitude, name)
    assert quantity.flags is magnitude.flags or quantity.flags == magnitude.flags
    np.testing.assert_array_equal(quantity.mask, magnitude.mask)
    np.testing.assert_array_equal(quantity.recordmask, magnitude.recordmask)


@pytest.mark.parametrize("kind", [None, "Length"])
@pytest.mark.parametrize("inplace", [False, True])
def test_power_rejects_foreign_exponent_before_mutation(kind: str | None, inplace: bool) -> None:
    registry = SemanticRegistry()
    foreign = SemanticRegistry().Quantity(2, "")
    quantity = registry.Quantity(np.array([2.0, 3.0]), "meter", kind=kind)
    with pytest.raises(ValueError, match="exponent belongs to a different Pint registry"):
        if inplace:
            quantity **= foreign
        else:
            quantity**foreign
    np.testing.assert_allclose(quantity.magnitude, [2, 3], rtol=0.0, atol=0.0)
    assert quantity.units == registry.meter


def test_registry_convert_validates_designations_and_retains_numeric_contract() -> None:
    registry = SemanticRegistry()
    with pytest.raises(TypeError, match="different factor designations"):
        registry.convert(1, registry.mol["O2"], registry.mol["CO2"])
    assert registry.convert(1, registry.mol["O2"], "mol") == 1
    with pytest.raises(TypeError, match="different factor designations"):
        registry.convert(1, "mol", registry.mol["O2"])
    magnitude = np.array([1.0, 2.0])
    result = registry.convert(magnitude, registry.mol["O2"], registry.mmol["O2"], inplace=True)
    assert result is magnitude
    np.testing.assert_allclose(result, [1000, 2000], rtol=1e-15, atol=0.0)
    assert registry.convert(1, "meter", "centimeter") == 100


@pytest.mark.parametrize("foreign_source", [False, True])
def test_registry_convert_rejects_foreign_unit_objects(foreign_source: bool) -> None:
    registry = SemanticRegistry()
    foreign = SemanticRegistry()
    src = foreign.meter if foreign_source else registry.meter
    dst = registry.centimeter if foreign_source else foreign.centimeter
    with pytest.raises(ValueError, match="different Pint registry"):
        registry.convert(1, src, dst)


def test_registry_convert_uses_quantity_target_units_without_reclassifying() -> None:
    registry = SemanticRegistry()
    energy = registry.Quantity(1, "joule", kind="Energy")
    torque = registry.Quantity(1, "joule", kind="Torque")
    assert registry.convert(2, energy, torque) == 2
    assert energy.to(torque).kind == "Energy"
    energy.ito(torque)
    assert energy.kind == "Energy"
    assert registry.convert(2, energy, "kilojoule") == pytest.approx(0.002, rel=1e-15, abs=0.0)


def test_registry_convert_rejects_foreign_quantity_magnitude() -> None:
    registry = SemanticRegistry()
    foreign = SemanticRegistry().Quantity(1, "second")
    with pytest.raises(ValueError, match="value belongs to a different Pint registry"):
        registry.convert(foreign, "meter", "centimeter")


def test_registry_convert_validates_affine_objects_and_keeps_physical_strings() -> None:
    registry = SemanticRegistry()
    with pytest.raises(TypeError, match="Generic Kelvin"):
        registry.convert(300, registry.kelvin, registry.degC)
    with pytest.raises(TypeError, match="Absolute temperature to a Delta"):
        registry.convert(20, registry.degC, registry.delta_degC)
    assert registry.convert(300, "kelvin", "degC") == pytest.approx(26.85, rel=0.0, abs=1e-12)


def test_registry_convert_context_kind_failure_preserves_input() -> None:
    registry = SemanticRegistry()
    magnitude = np.array([500.0, 600.0])
    source = registry.Quantity(1, "nanometer", kind="Length")
    with (
        registry.context("sp"),
        pytest.raises(ValueError, match="kind 'Length' requires dimensions"),
    ):
        registry.convert(magnitude, source, "hertz", inplace=True)
    np.testing.assert_allclose(magnitude, [500, 600], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("unit_objects", [False, True])
def test_registry_convert_rejects_quantity_as_magnitude(unit_objects: bool) -> None:
    registry = SemanticRegistry()
    source, target = (
        (registry.meter, registry.centimeter) if unit_objects else ("meter", "centimeter")
    )
    with pytest.raises(TypeError, match=r"magnitude; use quantity.to\(\)"):
        registry.convert(registry.Quantity(1, "kilometer"), source, target)


@pytest.mark.parametrize("inplace", [False, True])
def test_registry_convert_preserves_pint_list_behavior(inplace: bool) -> None:
    registry = SemanticRegistry()
    magnitude = [1.0, 2.0]
    for source, target in (("meter", "centimeter"), (registry.meter, registry.centimeter)):
        with pytest.raises(TypeError, match="multiply sequence"):
            registry.convert(magnitude, source, target, inplace=inplace)
    assert registry.convert(magnitude, registry.meter, registry.meter, inplace=inplace) is magnitude


def test_registry_convert_to_quantity_validates_and_ignores_target_magnitude() -> None:
    registry = SemanticRegistry()
    source = registry.Quantity(100, "joule", kind="Energy")
    target = registry.Quantity(999, "kilojoule", kind="Energy")
    assert registry.convert(2, source, target) == pytest.approx(0.002, rel=1e-15, abs=0.0)
    assert registry.convert(
        2, registry.mol["O2"], registry.Quantity(999, registry.mmol["O2"])
    ) == pytest.approx(2000, rel=1e-15, abs=0.0)


def test_classified_power_rejects_non_scalar_semantic_exponents() -> None:
    registry = SemanticRegistry()
    registry.define_kind("Probability", "dimensionless")
    quantity = registry.Quantity(2, "meter", kind="Length")
    with pytest.raises(TypeError, match="classified quantity cannot be used as an exponent"):
        quantity ** registry.Quantity(2, "", kind="Probability")
    with pytest.raises(TypeError, match="exponent quantity must be dimensionless"):
        quantity ** registry.Quantity(2, "meter")


def test_registry_convert_normalizes_mapping_source_with_semantic_target() -> None:
    registry = SemanticRegistry()
    assert registry.convert(1, {"meter": 1}, registry.centimeter) == pytest.approx(
        100, rel=1e-15, abs=0.0
    )
    assert registry.convert(1, registry.meter, {"centimeter": 1}) == pytest.approx(
        100, rel=1e-15, abs=0.0
    )
    with pytest.raises(TypeError, match="different factor designations"):
        registry.convert(1, {"meter": 1}, registry.centimeter["tissue"])


@pytest.mark.parametrize("operation", [operator.eq, operator.ne, np.equal, np.not_equal])
@pytest.mark.parametrize("reverse", [False, True])
def test_semantic_mismatch_equality_broadcasts_both_operands(operation: Any, reverse: bool) -> None:
    scalar = ureg.Quantity(1, "joule", kind="Energy")
    array = ureg.Quantity(np.array([[1.0, 2.0]]), "joule", kind="Torque")
    left, right = (array, scalar) if reverse else (scalar, array)
    result = operation(left, right)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 2)
    expected = operation in (operator.ne, np.not_equal)
    np.testing.assert_array_equal(result, np.full((1, 2), expected, dtype=np.bool_))


def test_foreign_registry_mismatch_broadcasts_scalar_and_array() -> None:
    foreign = SemanticRegistry()
    left = ureg.Quantity(1, "meter")
    right = foreign.Quantity(np.ones((2, 1)), "meter")
    assert np.shape(left == right) == (2, 1)
    np.testing.assert_array_equal(left == right, np.zeros((2, 1), dtype=np.bool_))
    np.testing.assert_array_equal(right == left, np.zeros((2, 1), dtype=np.bool_))


def test_unit_mismatch_comparison_preserves_array_shape() -> None:
    quantity = ureg.Quantity(np.ones((2, 3)), "joule", kind="Energy")
    assert np.shape(quantity == ureg.joule) == (2, 3)
    np.testing.assert_array_equal(quantity == ureg.joule, np.zeros((2, 3), dtype=np.bool_))


def test_semantic_mismatch_uses_numpy_broadcast_errors() -> None:
    left = ureg.Quantity(np.ones((2,)), "joule", kind="Energy")
    right = ureg.Quantity(np.ones((3,)), "joule", kind="Torque")
    with pytest.raises(ValueError, match="shape mismatch"):
        _ = left == right


@pytest.mark.parametrize("operation", [operator.add, operator.iadd])
def test_logarithmic_addition_requires_explicit_kind_removal(operation: Any) -> None:
    power = ureg.Quantity(10.0, "dBm", kind="Power")
    gain = ureg.Quantity(3.0, "dB")
    with pytest.raises(TypeError, match=r"quantity kinds 'Power' and None"):
        operation(power, gain)
    assert power.magnitude == pytest.approx(10.0, rel=0.0, abs=0.0)
    assert power.units == ureg.dBm
    assert power.kind == "Power"

    result = operation(power.as_unkinded(), gain)
    assert result.magnitude == pytest.approx(13.0, rel=1e-14, abs=0.0)
    assert result.units == ureg.dBm
    assert result.kind is None


@pytest.mark.parametrize("operation", [operator.add, operator.iadd])
def test_kinded_dimension_mismatch_precedes_kind_mismatch(operation: Any) -> None:
    length = ureg.Quantity(np.array([1, 2]), "meter", kind="Length")
    mass = ureg.Quantity(np.array([1, 2]), "kilogram", kind="Mass")
    with pytest.raises(pint.DimensionalityError, match="Cannot convert"):
        operation(length, mass)


@pytest.mark.parametrize("value", [1.0, np.array(1.0)])
def test_scalar_quantities_fail_iteration_eagerly(value: Any) -> None:
    quantity = ureg.Quantity(value, "meter", kind="Length")
    with pytest.raises(TypeError, match=r"not iterable|0-d array"):
        iter(quantity)
    assert not np.iterable(quantity)


def test_fortran_inplace_staging_preserves_aliases_and_failure_state() -> None:
    magnitude = np.asfortranarray([[1e308, 2.0], [3.0, 4.0]])
    original = magnitude.copy(order="K")
    quantity = ureg.Quantity(magnitude, "meter", kind="Length")
    with np.errstate(over="raise"), pytest.raises(FloatingPointError, match="overflow"):
        quantity *= 2
    assert quantity.magnitude is magnitude
    np.testing.assert_allclose(magnitude, original, rtol=0.0, atol=0.0)
    assert quantity.units == ureg.meter
    assert quantity.kind == "Length"
    quantity *= 0.5
    assert quantity.magnitude is magnitude
    assert magnitude.flags.f_contiguous
    np.testing.assert_allclose(magnitude, original * 0.5, rtol=1e-15, atol=0.0)


def test_registry_unit_conversion_preserves_plain_context_and_input_array() -> None:
    registry = SemanticRegistry()
    plain: pint.UnitRegistry[Any] = pint.UnitRegistry()
    values = np.array([500.0, 600.0], dtype=np.float32)
    original = values.copy()
    with registry.context("sp"), plain.context("sp"):
        actual = registry.convert(values, registry.nanometer, registry.hertz)
        expected = plain.convert(values, plain.nanometer, plain.hertz)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(values, original, rtol=0.0, atol=0.0)
    assert actual.shape == values.shape
    assert actual.dtype == expected.dtype


def test_registry_unit_conversion_uses_redefined_affine_role() -> None:
    registry = SemanticRegistry(on_redefinition="ignore")
    registry.define("custom_temperature = kelvin")
    source = registry.custom_temperature
    assert registry.convert(300, source, registry.kelvin) == pytest.approx(300, rel=1e-15)
    registry.define("custom_temperature = kelvin; offset: 273.15")
    with pytest.raises(TypeError, match="cannot convert a Absolute temperature to a Delta"):
        registry.convert(20, source, registry.delta_degC)


def test_registry_unit_conversion_excludes_redefined_affine_target() -> None:
    registry = SemanticRegistry(on_redefinition="ignore")
    registry.define("custom_temperature = kelvin")
    target = registry.custom_temperature
    assert registry.convert(300, registry.kelvin, target) == pytest.approx(300, rel=1e-15)
    registry.define("custom_temperature = kelvin; offset: 273.15")
    with pytest.raises(TypeError, match="Generic Kelvin"):
        registry.convert(300, registry.kelvin, target)


@pytest.mark.parametrize("target", ["mol/L", ureg.mole / ureg.liter])
def test_registry_convert_adopts_designations_with_plain_target(target: Any) -> None:
    source = ureg.millimole["O2"] / ureg.liter
    assert ureg.convert(5, source, target) == pytest.approx(0.005, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("constructor", ["__call__", "Quantity", "parse_expression"])
def test_absolute_temperature_string_parsing_explains_construction(constructor: str) -> None:
    with pytest.raises(
        TypeError, match=r'string parsing multiplies.*Quantity\(20, "degC"\).*20 \* degC'
    ) as caught:
        getattr(ureg, constructor)("20 degC")
    assert isinstance(caught.value.__cause__, TypeError)
    delta = getattr(ureg, constructor)("5 delta_degC")
    generic = getattr(ureg, constructor)("300 kelvin")
    assert delta.affine_role is AffineRole.DELTA
    assert delta.magnitude == 5
    assert generic.affine_role is AffineRole.GENERIC
    assert generic.magnitude == 300


@pytest.fixture
def restore_application_registries() -> Any:
    """Restore both pickle slots, selecting default mode last as at import."""

    from xpint import get_application_registry, set_application_registry

    previous_angle = get_application_registry(angle_dimension=True)
    previous_default = get_application_registry(angle_dimension=False)
    yield
    set_application_registry(previous_angle)
    set_application_registry(previous_default)


@pytest.mark.parametrize("angle_dimension", [False, True])
@pytest.mark.parametrize("quantity", [False, True])
@pytest.mark.usefixtures("restore_application_registries")
def test_application_registry_pickle_roundtrip(angle_dimension: bool, quantity: bool) -> None:
    import pickle

    from xpint import get_application_registry, set_application_registry

    pint_previous = pint.get_application_registry()  # type: ignore[no-untyped-call]
    registry = SemanticRegistry(angle_dimension=angle_dimension)
    registry.define("test_length = 2 * meter")
    registry.define_kind("TestLength", "meter")
    unit = registry.test_length["sample"]
    original = registry.Quantity(3.0, unit, kind="TestLength") if quantity else unit

    set_application_registry(registry)

    assert get_application_registry(angle_dimension=angle_dimension) is registry
    restored = pickle.loads(pickle.dumps(original))
    assert restored == original
    assert restored.designation_signature == original.designation_signature
    assert pint.get_application_registry() is pint_previous  # type: ignore[no-untyped-call]


@pytest.mark.parametrize("angle_first", [False, True], ids=["default-first", "angle-first"])
@pytest.mark.usefixtures("restore_application_registries")
def test_pickles_of_both_angle_modes_load_side_by_side(angle_first: bool) -> None:
    """Selecting an angle registry must not strand default-mode pickles, or vice versa."""

    import pickle

    from xpint import get_application_registry, set_application_registry

    default_target = SemanticRegistry()
    angle_target = SemanticRegistry(angle_dimension=True)
    for registry in (
        (angle_target, default_target) if angle_first else (default_target, angle_target)
    ):
        set_application_registry(registry)
    angle_source = SemanticRegistry(angle_dimension=True)
    payload = (
        ureg.Quantity(3.0, "m"),
        ureg.meter,
        angle_source.Quantity(1.0, "rad"),
        angle_source.radian,
    )

    default_quantity, default_unit, angle_quantity, angle_unit = pickle.loads(pickle.dumps(payload))

    assert get_application_registry(angle_dimension=False) is default_target
    assert get_application_registry(angle_dimension=True) is angle_target
    assert isinstance(default_quantity, default_target.Quantity)
    assert isinstance(default_unit, default_target.Unit)
    assert isinstance(angle_quantity, angle_target.Quantity)
    assert isinstance(angle_unit, angle_target.Unit)
    assert angle_quantity.m_as("radian") == pytest.approx(1.0, rel=1e-15)


def test_unselected_angle_slot_uses_a_shared_angle_registry() -> None:
    """Runs in a fresh interpreter: the first-use default cannot be observed otherwise."""

    script = """
import pickle
from xpint import SemanticRegistry, get_application_registry
payload = pickle.dumps(SemanticRegistry(angle_dimension=True).Quantity(2.0, "rad"))
restored = pickle.loads(payload)
shared = get_application_registry(angle_dimension=True)
assert shared is get_application_registry(angle_dimension=True)
assert shared.angle_dimension
assert isinstance(restored, shared.Quantity)
assert restored.m_as("radian") == 2.0
"""
    _run_fresh_interpreter(script)


@pytest.mark.parametrize(
    "getter",
    [
        "get_default_registry()",
        "get_application_registry(angle_dimension=False)",
        "get_application_registry(angle_dimension=True)",
    ],
)
def test_concurrent_first_use_shares_one_registry(getter: str) -> None:
    """Fresh interpreter, so the shared registry has not been built yet."""

    script = f"""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from xpint import get_application_registry, get_default_registry
barrier = Barrier(8)
def first_use(_):
    barrier.wait()
    return {getter}
with ThreadPoolExecutor(max_workers=8) as pool:
    registries = list(pool.map(first_use, range(8)))
assert len({{id(registry) for registry in registries}}) == 1
assert registries[0] is {getter}
from xpint.units import ureg
assert ureg is get_default_registry()
"""
    _run_fresh_interpreter(script)


# Interpreter start plus building two registries takes about a second; allow ample margin.
_FRESH_INTERPRETER_TIMEOUT_SECONDS = 60


def _run_fresh_interpreter(script: str) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        # A locking regression would otherwise hang the suite instead of failing.
        timeout=_FRESH_INTERPRETER_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.usefixtures("restore_application_registries")
def test_application_registry_getter_without_mode_is_deprecated_after_angle_selection() -> None:
    import warnings

    from xpint import get_application_registry, set_application_registry

    angle_registry = SemanticRegistry(angle_dimension=True)
    set_application_registry(angle_registry)
    with pytest.warns(DeprecationWarning, match=r"angle_dimension=True"):
        assert get_application_registry() is angle_registry

    set_application_registry(ureg)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert get_application_registry() is ureg


def test_application_registry_getter_rejects_non_bool_mode() -> None:
    from xpint import get_application_registry

    with pytest.raises(TypeError, match="angle_dimension must be a bool, got int"):
        get_application_registry(angle_dimension=1)  # type: ignore[arg-type]


@pytest.mark.usefixtures("restore_application_registries")
def test_pickle_requires_kind_definitions_in_the_selected_angle_registry() -> None:
    import pickle

    from xpint import set_application_registry

    set_application_registry(SemanticRegistry(angle_dimension=True))
    source = SemanticRegistry(angle_dimension=True)
    source.define_kind("UnregisteredTestAngle", "radian")

    with pytest.raises(KeyError, match=r"'UnregisteredTestAngle'.*set_application_registry"):
        pickle.loads(pickle.dumps(source.Quantity(1, "rad", kind="UnregisteredTestAngle")))


def test_quantity_and_unit_constructors_do_not_accept_factor_expressions() -> None:
    with pytest.raises(TypeError, match="_semantic_factors"):
        ureg.Quantity(1, "meter", _semantic_factors=())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="_semantic_factors"):
        ureg.Unit("meter", _semantic_factors=())  # type: ignore[call-arg]


def test_application_registry_setter_rejects_other_types() -> None:
    from xpint import get_application_registry, set_application_registry

    previous = get_application_registry()
    with pytest.raises(TypeError, match="must be a SemanticRegistry"):
        set_application_registry(pint.UnitRegistry())  # type: ignore[arg-type]
    assert get_application_registry() is previous


@pytest.mark.parametrize("quantity", [False, True])
def test_pickle_requires_application_registry_unit_definitions(quantity: bool) -> None:
    import pickle

    registry = SemanticRegistry()
    registry.define("unregistered_test_length = 2 * meter")
    value = (
        registry.Quantity(1, "unregistered_test_length")
        if quantity
        else registry.unregistered_test_length
    )
    with pytest.raises(pint.UndefinedUnitError, match="unregistered_test_length"):
        pickle.loads(pickle.dumps(value))


def test_pickle_requires_application_registry_kind_definitions() -> None:
    import pickle

    registry = SemanticRegistry()
    registry.define_kind("UnregisteredTestLength", "meter")
    with pytest.raises(KeyError, match=r"'UnregisteredTestLength'.*set_application_registry"):
        pickle.loads(pickle.dumps(registry.Quantity(1, "m", kind="UnregisteredTestLength")))


def test_parsing_preserves_unrelated_type_errors() -> None:
    with pytest.raises(TypeError, match="absolute temperatures cannot be exponentiated"):
        ureg.parse_expression("degC ** 2")


def test_compatibility_raises_for_undefined_units() -> None:
    with pytest.raises(UndefinedUnitError):
        (1 * m).is_compatible_with("not_a_unit_xyz")
    with pytest.raises(UndefinedUnitError):
        m.is_compatible_with("not_a_unit_xyz")


def test_compatible_units_is_a_physical_catalog() -> None:
    designated = 5 * ureg.mole["O2"]
    assert designated.compatible_units() == ureg.mole.compatible_units()  # type: ignore[no-untyped-call]
    assert all(not unit.has_designation for unit in designated.compatible_units())
    assert ureg.kelvin in (20 * ureg.degC).compatible_units()
    assert not ureg.kelvin.is_compatible_with(ureg.degC)


def test_plain_tuple_constructor_carries_only_physical_units() -> None:
    quantity = ureg.Quantity.from_tuple((5, (("meter", 1),)))  # type: ignore[no-untyped-call]
    assert quantity.magnitude == 5
    assert quantity.units == m
    assert quantity.kind is None
    assert not quantity.has_designation


@pytest.mark.parametrize(
    "method", ["to_unprefixed", "ito_unprefixed", "to_preferred", "ito_preferred"]
)
def test_inherited_conversions_preserve_classified_state(method: str) -> None:
    quantity = ureg.Quantity(5000, ureg.millimole["O2"], kind="AmountOfSubstance")
    arguments = ([ureg.mole],) if "preferred" in method else ()
    converted = getattr(quantity, method)(*arguments)
    result = quantity if method.startswith("ito") else converted
    assert result.kind == "AmountOfSubstance"
    assert result.designation_signature == quantity.designation_signature
    assert result.to("mol").magnitude == pytest.approx(5, rel=1e-15, abs=0)


def test_unit_conversion_adapters_preserve_or_reject_designations() -> None:
    oxygen: Any = ureg.mole["O2"]
    value = 5000 * ureg.millimole["O2"]
    assert oxygen.from_(value).designation_signature == value.designation_signature
    assert oxygen.m_from(value) == pytest.approx(5, rel=1e-15, abs=0)
    for method in (oxygen.from_, oxygen.m_from):
        with pytest.raises(TypeError, match="different factor designations"):
            method(5 * ureg.mole["CO2"])
    assert oxygen.from_(5, strict=False).designation_signature == value.designation_signature
    with pytest.raises(TypeError, match="different factor designations"):
        oxygen.compare(ureg.mole["CO2"], operator.lt)
