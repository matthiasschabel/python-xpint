"""Quantity-kind behavior tests."""

import copy
import datetime
import operator
import pickle
from collections import defaultdict
from collections.abc import Callable
from itertools import combinations
from typing import Any

import numpy as np
import pint
import pytest

from xpint import SemanticRegistry
from xpint.units import J, N, m, ureg


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("N * m", ("Energy", "Torque")),
        ("kelvin", ("Temperature", "TemperatureDifference")),
        ("degC", ("Temperature",)),
        ("delta_degC", ("TemperatureDifference",)),
        ("dimensionless", ("Dimensionless", "PlaneAngle", "SolidAngle")),
        ("meter ** 4", ()),
    ],
)
def test_compatible_kinds_matches_dimensions_and_affine_units(
    expression: str, expected: tuple[str, ...]
) -> None:
    unit = ureg.Unit(expression)
    quantity = ureg.Quantity(1, unit)

    assert ureg.compatible_kinds(expression) == expected
    assert ureg.compatible_kinds(unit) == unit.compatible_kinds() == expected
    assert ureg.compatible_kinds(quantity) == quantity.compatible_kinds() == expected


def test_compatible_kinds_ignores_designations() -> None:
    registry = SemanticRegistry()
    registry.define_kind("Concentration", "mole / liter")
    unit = registry.millimole["O2"] / registry.liter
    quantity = registry.Quantity(1, unit)
    expected = (registry.millimole / registry.liter).compatible_kinds()

    assert expected == ("Concentration",)
    assert unit.compatible_kinds() == quantity.compatible_kinds() == expected
    assert unit.has_designation
    assert quantity.has_designation


@pytest.mark.parametrize(
    ("expression", "kind", "expected"),
    [
        ("N * m", "Torque", ("Energy", "Torque")),
        ("kelvin", "Temperature", ("Temperature", "TemperatureDifference")),
        ("kelvin", "TemperatureDifference", ("Temperature", "TemperatureDifference")),
    ],
)
def test_compatible_kinds_ignores_current_kind(
    expression: str, kind: str, expected: tuple[str, ...]
) -> None:
    quantity = ureg.Quantity(np.array([1, 2], dtype=np.int16), expression, kind=kind)
    magnitude = quantity.magnitude

    assert quantity.compatible_kinds() == ureg.compatible_kinds(quantity) == expected
    assert quantity.kind == kind
    assert quantity.magnitude is magnitude
    assert quantity.units == ureg.Unit(expression)


def test_compatible_kinds_includes_new_kinds_once_in_alphabetical_order() -> None:
    registry = SemanticRegistry()
    unit = registry.joule
    assert unit.compatible_kinds() == ("Energy", "Torque")

    registry.define_kind("CustomEnergy", "kilojoule")
    registry.define_kind("CustomEnergy", "N * m")

    assert unit.compatible_kinds() == ("CustomEnergy", "Energy", "Torque")
    assert "CustomEnergy" not in ureg.compatible_kinds("joule")


def test_compatible_kinds_respects_angle_dimension() -> None:
    registry = SemanticRegistry(angle_dimension=True)

    assert registry.compatible_kinds("N * m") == ("Energy",)
    assert (registry.joule / registry.radian).compatible_kinds() == ("Torque",)


def test_as_kind_validates_dimension_and_can_be_removed() -> None:
    energy = (10 * J).as_kind("Energy")

    assert energy.kind == "Energy"
    assert energy.has_kind()
    assert energy.as_unkinded().kind is None
    assert not energy.as_unkinded().has_kind()


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown quantity kind"):
        (1 * J).as_kind("NotRegistered")


def test_kind_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires dimensions"):
        (1 * m).as_kind("Energy")


@pytest.mark.parametrize("operation", [operator.add, operator.sub])
def test_addition_and_subtraction_reject_unkinded_operand_in_both_orders(operation: Any) -> None:
    energy = (5 * J).as_kind("Energy")
    unkinded = 3 * J

    with pytest.raises(
        TypeError, match=r"quantity kinds 'Energy' and None.*as_kind\(\).*as_unkinded\(\)"
    ):
        operation(energy, unkinded)
    with pytest.raises(
        TypeError, match=r"quantity kinds None and 'Energy'.*as_kind\(\).*as_unkinded\(\)"
    ):
        operation(unkinded, energy)


def test_addition_rejects_distinct_kinds_with_same_dimensions() -> None:
    energy = (5 * J).as_kind("Energy")
    torque = (3 * N * m).as_kind("Torque")

    with pytest.raises(TypeError, match=r"'Energy'.*'Torque'"):
        energy + torque


def test_multiplication_only_preserves_kind_through_dimensionless_scaling() -> None:
    energy = (5 * J).as_kind("Energy")
    dimensionless = ureg.Quantity(2)

    assert (energy * 2).kind == "Energy"
    assert (2 * energy).kind == "Energy"
    assert (energy * dimensionless).kind == "Energy"
    assert (dimensionless * energy).kind == "Energy"
    assert (energy / dimensionless).kind == "Energy"
    assert (energy * m).kind is None
    assert (dimensionless / energy).kind is None


def test_classified_quantity_cannot_be_used_as_an_exponent() -> None:
    probability = ureg.Quantity(0.5).as_kind("Dimensionless")

    with pytest.raises(TypeError, match="classified quantity cannot be used as an exponent"):
        2**probability
    with pytest.raises(TypeError, match="classified quantity cannot be used as an exponent"):
        np.power(2.0, probability)


def test_unclassified_quantity_can_be_used_as_an_exponent() -> None:
    result = 2 ** ureg.Quantity(0.5)

    assert result == pytest.approx(2**0.5, rel=1e-15, abs=0.0)


def test_conversion_and_copy_preserve_kind() -> None:
    energy = (1500 * J).as_kind("Energy")

    converted = energy.to("kilojoule")
    shallow = copy.copy(energy)
    deep = copy.deepcopy(energy)

    assert converted.magnitude == pytest.approx(1.5, rel=1e-15, abs=0.0)
    assert converted.kind == "Energy"
    assert shallow.kind == "Energy"
    assert deep.kind == "Energy"


def test_context_conversion_rejects_kind_with_incompatible_result_dimension() -> None:
    wavelength = (500 * ureg.nanometer).as_kind("Length")

    with pytest.raises(TypeError, match=r"as_unkinded\(\)"):
        wavelength.to("terahertz", "spectroscopy")


def test_iteration_indexing_and_unary_operations_preserve_kind() -> None:
    energy = ureg.Quantity([1, 2, 3], J, kind="Energy")

    assert energy[1].kind == "Energy"
    assert next(iter(energy)).kind == "Energy"
    assert abs(-energy).kind == "Energy"


def test_in_place_operations_preserve_matching_kind() -> None:
    energy = ureg.Quantity([1.0, 2.0], J, kind="Energy")
    other = ureg.Quantity([3.0, 4.0], J, kind="Energy")

    energy += other
    energy *= 2

    assert energy.magnitude.tolist() == [8.0, 12.0]
    assert energy.kind == "Energy"


def test_in_place_array_operations_preserve_alias_and_pint_casting() -> None:
    energy = ureg.Quantity([1.0, 2.0], J, kind="Energy")
    alias = energy.magnitude

    energy += ureg.Quantity([0.5, 0.5], J, kind="Energy")

    assert alias is energy.magnitude
    assert alias.tolist() == [1.5, 2.5]
    integers = ureg.Quantity(np.array([1, 2], dtype="int16"), J)
    integer_alias = integers.magnitude
    with pytest.raises(TypeError):
        integers += ureg.Quantity(np.array([0.5, 0.5]), J)
    assert integer_alias is integers.magnitude
    assert integer_alias.tolist() == [1, 2]


@pytest.mark.parametrize("operation", [operator.iadd, operator.isub])
@pytest.mark.parametrize("kinds", [("Energy", None), (None, "Energy"), ("Energy", "Torque")])
def test_in_place_kind_mismatch_leaves_target_unchanged(
    operation: Any, kinds: tuple[str | None, str | None]
) -> None:
    magnitude = np.array([1, 2], dtype=np.int16)
    target = ureg.Quantity(magnitude, J, kind=kinds[0])
    other = ureg.Quantity(3, J, kind=kinds[1])

    with pytest.raises(TypeError, match="quantity kinds"):
        operation(target, other)

    assert target.magnitude is magnitude
    np.testing.assert_array_equal(magnitude, [1, 2])
    assert magnitude.dtype == np.int16
    assert target.units == J
    assert target.kind == kinds[0]


def test_in_place_conversion_synthesizes_affine_kind() -> None:
    temperature = ureg.Quantity([20.0], "degC")

    temperature.ito("kelvin")

    assert temperature.magnitude.tolist() == pytest.approx([293.15], rel=0.0, abs=1e-12)
    assert temperature.kind == "Temperature"


def test_pickle_round_trip_uses_default_registry_and_preserves_kind() -> None:
    energy = (5 * J).as_kind("Energy")

    restored = pickle.loads(pickle.dumps(energy))

    assert isinstance(restored, ureg.Quantity)
    assert restored == energy
    assert restored.kind == "Energy"


def test_moment_of_inertia_kind_is_available_without_angle_dimensions() -> None:
    """Default-mode code must classify and round-trip inertia, not only angle mode."""

    inertia = (2.0 * ureg.kilogram * m**2).as_kind("MomentOfInertia")

    assert inertia.kind == "MomentOfInertia"
    assert "MomentOfInertia" in inertia.compatible_kinds()
    restored = pickle.loads(pickle.dumps(inertia))
    assert restored.kind == "MomentOfInertia"
    assert restored == inertia


def test_timedelta_conversion_rejects_unrepresentable_kind() -> None:
    duration = (2 * ureg.hour).as_kind("Time")

    with pytest.raises(TypeError, match="cannot represent kind or designation metadata"):
        duration.to_timedelta()

    assert duration.as_unkinded().to_timedelta() == datetime.timedelta(hours=2)


@pytest.mark.parametrize("coercion", [float, int, complex])
def test_scalar_coercion_rejects_dimensionless_kind(
    coercion: Callable[[object], object],
) -> None:
    registry = SemanticRegistry()
    registry.define_kind("Probability", "dimensionless")
    probability = registry.Quantity(0.5).as_kind("Probability")

    with pytest.raises(TypeError, match="retained kind"):
        coercion(probability)


def test_plain_tuple_serialization_rejects_kind_loss() -> None:
    energy = (5 * J).as_kind("Energy")

    with pytest.raises(TypeError, match="cannot represent kind"):
        energy.to_tuple()


@pytest.mark.parametrize(
    "operation",
    [
        lambda value: value // value,
        lambda value: value % value,
        lambda value: divmod(value, value),
    ],
)
def test_discrete_arithmetic_on_kinds_fails_closed(
    operation: Callable[[Any], object],
) -> None:
    energy = (5 * J).as_kind("Energy")

    with pytest.raises(TypeError, match="no xpint semantic propagation rule"):
        operation(energy)


def test_equality_ordering_and_hashing_apply_strict_kind_policy() -> None:
    energy = (5 * J).as_kind("Energy")
    torque = (5 * J).as_kind("Torque")
    unkinded = 5 * J
    converted_energy = energy.to("kilojoule")

    assert energy == converted_energy
    assert energy != unkinded
    assert unkinded != torque
    assert energy != torque
    assert energy <= converted_energy
    with pytest.raises(TypeError, match=r"'Energy'.*None"):
        _ = energy <= unkinded
    with pytest.raises(TypeError, match=r"None.*'Energy'"):
        _ = unkinded <= energy
    with pytest.raises(TypeError, match=r"'Energy'.*'Torque'"):
        _ = energy < torque
    mapping = {energy: "energy", torque: "torque", unkinded: "plain"}
    assert mapping[converted_energy] == "energy"
    assert len(mapping) == 3
    assert hash(energy) == hash(converted_energy)


def test_compatibility_requires_exact_kind_state() -> None:
    energy = (5 * J).as_kind("Energy")
    same_kind = (0.005 * ureg.kilojoule).as_kind("Energy")
    torque = (5 * J).as_kind("Torque")
    unkinded = 5 * J

    assert energy.is_compatible_with(same_kind)
    assert not energy.is_compatible_with(torque)
    assert not energy.is_compatible_with(unkinded)
    assert energy.is_compatible_with(J)
    assert energy.is_compatible_with("joule")
    assert unkinded.is_compatible_with(J)


def test_classified_dimensionless_comparison_and_compatibility_reject_plain_scalars() -> None:
    probability = ureg.Quantity(0.5).as_kind("Dimensionless")

    assert probability != 0.5
    assert not probability.is_compatible_with(0.5)
    with pytest.raises(TypeError, match="classified quantity against an ordinary scalar"):
        _ = probability < 1.0


def test_dimensional_kind_ordering_against_scalar_uses_pint_error() -> None:
    energy = (5 * J).as_kind("Energy")

    with pytest.raises(ValueError, match="Cannot compare PlainQuantity"):
        _ = energy < 3


# Temperature and TemperatureDifference share a dimensionality but must still
# combine: an absolute temperature shifted by a difference is legitimate, and the
# affine mechanism rather than the kind policy governs it. See
# test_affine_temperature.py for that behavior.
AFFINE_TEMPERATURE_KINDS = frozenset({"Temperature", "TemperatureDifference"})


def _equal_dimension_kind_pairs() -> list[tuple[str, str]]:
    """Return every pair of built-in kinds that shares one dimensionality.

    Derived from the registry rather than enumerated by hand so that a newly
    registered kind is covered without editing this file. A hand-written list
    silently stops covering the pairs it was written before.
    """

    by_dimension: dict[Any, list[str]] = defaultdict(list)
    for name, definition in ureg.kinds.items():
        by_dimension[ureg.get_dimensionality(definition.reference_unit)].append(name)
    return [
        pair
        for names in by_dimension.values()
        for pair in combinations(sorted(names), 2)
        if frozenset(pair) != AFFINE_TEMPERATURE_KINDS
    ]


def test_equal_dimension_kind_derivation_finds_the_known_pairs() -> None:
    """Guard the parametrization below, which would pass vacuously if empty.

    An empty derivation produces zero test cases and a green suite, so assert the
    derivation actually found the pairs it is supposed to protect.
    """

    pairs = _equal_dimension_kind_pairs()

    assert len(pairs) >= 13
    assert ("Energy", "Torque") in pairs
    assert ("Entropy", "HeatCapacity") in pairs
    assert AFFINE_TEMPERATURE_KINDS not in {frozenset(pair) for pair in pairs}


@pytest.mark.parametrize(("left_kind", "right_kind"), _equal_dimension_kind_pairs())
def test_equal_dimension_kinds_reject_each_other(left_kind: str, right_kind: str) -> None:
    """Each pair exists to catch a mistake dimensional analysis cannot see.

    A pair is worthless unless both halves reduce to one dimensionality and the
    two still refuse to combine, so assert both rather than either alone.
    """

    left = ureg.Quantity(5.0, ureg.kinds[left_kind].reference_unit, kind=left_kind)
    right = ureg.Quantity(3.0, ureg.kinds[right_kind].reference_unit, kind=right_kind)

    assert left.dimensionality == right.dimensionality
    with pytest.raises(TypeError, match="quantity kinds"):
        left + right


def test_work_is_not_a_separate_kind_from_energy() -> None:
    """Work is energy transferred, not a distinct meaning, so it is not a kind."""

    assert "Work" not in ureg.kinds


def test_kinded_repr_extends_pint_float_rendering() -> None:
    assert repr((10 * J).as_kind("Energy")) == "<Quantity(10, 'joule', kind='Energy')>"
    value = (1.23456789123 * J).as_kind("Energy")
    plain: pint.UnitRegistry[Any] = pint.UnitRegistry()
    expected = repr(plain.Quantity(value.magnitude, "joule"))
    assert repr(value) == expected.removesuffix(")>") + ", kind='Energy')>"
    assert str(value) == "1.23456789123 joule"


def test_scalar_hashes_preserve_plain_number_equality_and_arrays_are_unhashable() -> None:
    value = ureg.Quantity(50, "percent")
    assert hash(value) == hash(0.5)
    mapping: dict[object, str] = {value: "half"}
    assert mapping[0.5] == "half"
    for kind in (None, "Energy"):
        array = ureg.Quantity(np.array([1.0, 2.0], dtype=np.float64), J, kind=kind)
        with pytest.raises(TypeError, match=r"unhashable type.*ndarray"):
            hash(array)


@pytest.mark.parametrize("inplace", [False, True])
def test_spectroscopy_conversion_requires_explicit_kind_removal(inplace: bool) -> None:
    source = (1 * ureg.nm).as_kind("Length")
    with pytest.raises(TypeError, match=r"as_unkinded\(\)"):
        if inplace:
            source.ito("eV", "spectroscopy")
        else:
            source.to("eV", "spectroscopy")
    assert source.kind == "Length"
    assert source.units == ureg.nm
    plain = source.as_unkinded()
    expected = plain.to("eV", "spectroscopy")
    plain.ito("eV", "spectroscopy")
    assert plain.magnitude == pytest.approx(expected.magnitude, rel=1e-15, abs=0.0)
    assert plain.kind is None
    assert not source.is_compatible_with("eV", "spectroscopy")
