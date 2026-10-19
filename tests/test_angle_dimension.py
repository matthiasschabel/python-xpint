"""Opt-in plane-angle base dimension, with solid angle derived as rad ** 2."""

import logging
import re
from typing import Any, cast

import numpy as np
import pytest
from pint.errors import DimensionalityError, RedefinitionError, UnitStrippedWarning

from xpint import AffineRole, SemanticQuantity, SemanticRegistry
from xpint.units import ureg as default_ureg


@pytest.fixture
def angle_registry() -> SemanticRegistry:
    return SemanticRegistry(angle_dimension=True)


# --- the default path must be untouched ---------------------------------------------


def test_default_registry_keeps_pint_angle_semantics() -> None:
    """The switch is opt-in, so an ordinary registry must behave exactly as before."""

    registry = SemanticRegistry()

    assert not registry.angle_dimension
    assert registry.get_dimensionality("radian") == registry.get_dimensionality("")
    assert registry.get_dimensionality("radian/second") == registry.get_dimensionality("hertz")
    assert registry.kinds["Torque"].reference_unit == "newton * meter"
    assert registry.kinds["MomentOfInertia"].reference_unit == "kilogram * meter ** 2"
    assert float(registry.Quantity(90.0, "degree")) == pytest.approx(np.pi / 2, rel=1e-15)


def test_building_an_angle_registry_does_not_disturb_other_registries() -> None:
    """A shared Pint or xpint global would make the switch leak into unrelated code."""

    before_module = default_ureg.get_dimensionality("radian")
    before_fresh = SemanticRegistry().get_dimensionality("radian")

    SemanticRegistry(angle_dimension=True)

    assert default_ureg.get_dimensionality("radian") == before_module
    assert SemanticRegistry().get_dimensionality("radian") == before_fresh
    assert default_ureg.kinds["Torque"].reference_unit == "newton * meter"


def test_angle_dimension_is_read_only() -> None:
    """Dimensionality is cached at construction, so a runtime toggle cannot be honest."""

    registry = SemanticRegistry(angle_dimension=True)

    with pytest.raises(AttributeError):
        registry.angle_dimension = False  # type: ignore[misc]


def test_construction_is_silent_and_preserves_the_redefinition_policy() -> None:
    """The overrides are the switch working, not a caller mistake, so they must not warn.

    Silencing must also be scoped: a caller who asked for on_redefinition="raise" still
    gets it for their own definitions, and the switch must not fail under that option.
    """

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    logger = logging.getLogger("pint")
    logger.addHandler(handler)
    try:
        registry = SemanticRegistry(angle_dimension=True, on_redefinition="raise")
    finally:
        logger.removeHandler(handler)

    assert not [r for r in records if "Redefining" in r.getMessage()]
    with pytest.raises(RedefinitionError):
        registry.define("meter = [length] = m")


# --- what the switch buys -----------------------------------------------------------


def test_plane_angle_is_a_base_dimension_and_solid_angle_derives_from_it(
    angle_registry: SemanticRegistry,
) -> None:
    """Solid angle is not independent: a small patch is a product of two plane angles."""

    registry = angle_registry

    assert registry.angle_dimension
    assert registry.get_dimensionality("radian") != registry.get_dimensionality("")
    assert registry.get_dimensionality("steradian") == registry.get_dimensionality("radian ** 2")
    assert registry.get_dimensionality("radian/second") != registry.get_dimensionality("hertz")


def test_the_derived_steradian_still_separates_solid_from_plane_angle(
    angle_registry: SemanticRegistry,
) -> None:
    """The objection to sr = rad**2 was that it reinstates a conflation. It does not.

    [angle] and [angle] ** 2 are different dimensions, so everything the switch exists
    to catch is still caught.
    """

    registry = angle_registry

    assert registry.get_dimensionality("steradian") != registry.get_dimensionality("radian")
    assert registry.get_dimensionality("steradian") != registry.get_dimensionality("")
    with pytest.raises(DimensionalityError):
        registry.Quantity(1.0, "steradian") + registry.Quantity(1.0, "radian")
    with pytest.raises(DimensionalityError):
        np.sin(registry.Quantity(1.0, "steradian"))


def test_angular_area_multiplies_to_a_solid_angle(angle_registry: SemanticRegistry) -> None:
    """The working relation in optics and antenna engineering, which independence rejects."""

    registry = angle_registry
    beam = registry.Quantity(1.0, "milliradian")

    assert (beam * beam).m_as("microsteradian") == pytest.approx(1.0, rel=1e-12)
    assert (registry.Quantity(1.0, "radian") * registry.Quantity(1.0, "radian")).m_as(
        "steradian"
    ) == pytest.approx(1.0, rel=1e-15)
    assert registry.Quantity(4 * np.pi, "radian ** 2").m_as("steradian") == pytest.approx(
        4 * np.pi, rel=1e-15
    )


def test_deriving_the_steradian_widens_a_dimensional_collision(
    angle_registry: SemanticRegistry,
) -> None:
    """The honest cost, recorded rather than hidden.

    Identifying sr with rad**2 is a quotient of the dimension group, so every distinction
    of the form [solid_angle] minus two [angle] disappears. That is an unbounded family,
    not a fixed list. Under an independent solid angle these pairs differed.
    """

    registry = angle_registry
    collides = [
        ("kilogram * meter ** 2 / radian ** 2", "kilogram * meter ** 2 / steradian"),
        ("joule / radian ** 2", "joule / steradian"),
        ("radian ** 2 / second", "steradian / second"),
    ]
    for left, right in collides:
        assert registry.get_dimensionality(left) == registry.get_dimensionality(right)

    # Kinds distinguish quantities whose dimensions collide.
    registry.define_kind("MassAreaPerSolidAngle", "kilogram * meter ** 2 / steradian")
    inertia = registry.Quantity(1.0, "kilogram * meter ** 2 / radian ** 2", kind="MomentOfInertia")
    per_solid_angle = registry.Quantity(
        1.0, "kilogram * meter ** 2 / steradian", kind="MassAreaPerSolidAngle"
    )

    assert inertia.dimensionality == per_solid_angle.dimensionality
    with pytest.raises(TypeError, match="quantity kinds"):
        inertia + per_solid_angle

    # Missing classification also requires an explicit choice by the caller.
    with pytest.raises(TypeError, match="quantity kinds"):
        inertia + registry.Quantity(1.0, "kilogram * meter ** 2 / steradian")


@pytest.mark.parametrize("unit", ["degree", "turn", "grade", "mil"])
def test_derived_angle_units_follow_the_redefinition(
    angle_registry: SemanticRegistry, unit: str
) -> None:
    """These are defined in terms of radian, so they must track it without extra work."""

    assert angle_registry.get_dimensionality(unit) == angle_registry.get_dimensionality("radian")


def test_named_luminous_dimensions_stay_consistent(angle_registry: SemanticRegistry) -> None:
    """Pint declares these separately from the units carrying them, so both need redefining."""

    assert angle_registry.Quantity(1.0, "lumen").check("[luminous_flux]")
    assert angle_registry.Quantity(1.0, "lux").check("[illuminance]")


def test_angle_and_solid_angle_do_not_combine(angle_registry: SemanticRegistry) -> None:
    with pytest.raises(DimensionalityError):
        angle_registry.Quantity(1.0, "radian") + angle_registry.Quantity(1.0, "steradian")


def test_trigonometry_rejects_a_solid_angle(angle_registry: SemanticRegistry) -> None:
    """The case stock Pint cannot catch: sin of a solid angle silently returns a number."""

    permissive = cast(SemanticQuantity, np.sin(default_ureg.Quantity(1.0, "steradian")))
    assert permissive.magnitude == pytest.approx(np.sin(1.0), rel=1e-15)
    with pytest.raises(DimensionalityError):
        np.sin(angle_registry.Quantity(1.0, "steradian"))


def test_every_builtin_kind_remains_constructible(angle_registry: SemanticRegistry) -> None:
    """Kinds freeze their dimensionality, so seeding order is easy to get wrong."""

    for name, definition in angle_registry.kinds.items():
        angle_registry.Quantity(1.0, definition.reference_unit, kind=name)


def test_conjugate_kinds_separate_torque_from_energy(angle_registry: SemanticRegistry) -> None:
    registry = angle_registry

    assert registry.kinds["Torque"].reference_unit == "joule / radian"
    assert registry.get_dimensionality("joule/radian") != registry.get_dimensionality("joule")
    assert registry.get_dimensionality("joule*second/radian") != registry.get_dimensionality(
        "joule*second"
    )


ROTATIONAL_IDENTITIES = [
    ("v = eta * omega * r", "(radian/second) * meter / radian", "meter/second"),
    ("L = I * omega", "(kilogram*meter**2/radian**2)*(radian/second)", "joule*second/radian"),
    ("tau = I * alpha", "(kilogram*meter**2/radian**2)*(radian/second**2)", "joule/radian"),
    ("tau = dL/dt", "(joule*second/radian)/second", "joule/radian"),
    ("KE = I * omega**2", "(kilogram*meter**2/radian**2)*(radian/second)**2", "joule"),
    ("P = tau * omega", "(joule/radian)*(radian/second)", "watt"),
    ("W = tau * theta", "(joule/radian)*radian", "joule"),
]


@pytest.mark.parametrize(("identity", "left", "right"), ROTATIONAL_IDENTITIES)
def test_rotational_algebra_closes(
    angle_registry: SemanticRegistry, identity: str, left: str, right: str
) -> None:
    """The conjugate spellings are only worth anything if the mechanics still balances."""

    assert angle_registry.get_dimensionality(left) == angle_registry.get_dimensionality(right), (
        identity
    )


# --- leaving the angle world --------------------------------------------------------


def test_float_names_the_escape_hatches(angle_registry: SemanticRegistry) -> None:
    """Pint's bare DimensionalityError would not tell a caller what to do instead."""

    with pytest.raises(DimensionalityError, match=r"m_as"):
        float(angle_registry.Quantity(90.0, "degree"))
    with pytest.raises(DimensionalityError, match=r"an angle is not a number"):
        int(angle_registry.Quantity(90.0, "degree"))
    with pytest.raises(DimensionalityError, match=r"an angle is not a number"):
        complex(angle_registry.Quantity(90.0, "degree"))


@pytest.mark.parametrize("convert", [np.asarray, np.array], ids=["asarray", "array"])
@pytest.mark.parametrize(
    ("magnitude", "unit"),
    [(90.0, "degree"), ([1.0, 2.0], "radian"), ([[1.0]], "radian * meter")],
    ids=["scalar", "vector", "mixed"],
)
def test_ndarray_conversion_refuses_an_angle_as_float_does(
    angle_registry: SemanticRegistry, convert: Any, magnitude: Any, unit: str
) -> None:
    """Pint would strip the angle with only a warning, so the guard must cover arrays too."""

    angle = angle_registry.Quantity(np.asarray(magnitude), unit)

    with pytest.raises(DimensionalityError, match=r"so numpy\.asarray\(\) cannot choose"):
        convert(angle)


def test_ndarray_conversion_of_a_non_angle_keeps_pints_stripping(
    angle_registry: SemanticRegistry,
) -> None:
    """Only the angle guard is new; ordinary units keep Pint's strip-with-warning."""

    with pytest.warns(UnitStrippedWarning):
        result = np.asarray(angle_registry.Quantity(np.array([1.0, 2.0]), "joule"))

    np.testing.assert_array_equal(result, [1.0, 2.0])


def test_ndarray_conversion_of_a_default_registry_angle_is_unchanged() -> None:
    """Without angle dimensions radian is dimensionless, as in Pint."""

    with pytest.warns(UnitStrippedWarning):
        result = np.asarray(default_ureg.Quantity(np.array([1.0, 2.0]), "radian"))

    np.testing.assert_array_equal(result, [1.0, 2.0])


def test_complex_coercion_of_an_unclassified_number_still_delegates_to_pint() -> None:
    assert complex(default_ureg.Quantity(1.0)) == complex(1.0)


# (unit, the angular content the message must name, the factor that clears it)
ANGULAR_COERCION_CASES = [
    ("radian", "radian", "radian ** -1"),
    ("degree", "radian", "radian ** -1"),
    ("steradian", "radian ** 2", "radian ** -2"),
    ("joule / radian", "radian ** -1", "radian"),
    ("kilogram * meter ** 2 / radian ** 2", "radian ** -2", "radian ** 2"),
    ("radian / second", "radian", "radian ** -1"),
    # powered and mixed pure-angular values: m_as needs the whole expression, not a factor
    ("radian ** 2", "radian ** 2", "radian ** -2"),
    ("radian ** 2 * steradian", "radian ** 4", "radian ** -4"),
    ("steradian ** 3", "radian ** 6", "radian ** -6"),
    # fractional exponents must round-trip exactly, not through limit_denominator
    ("radian ** 0.5", "radian ** 0.5", "radian ** -0.5"),
    ("radian ** 1.5", "radian ** 1.5", "radian ** -1.5"),
]


@pytest.mark.parametrize(("unit", "carried", "clearing"), ANGULAR_COERCION_CASES)
def test_coercion_message_reports_the_actual_angular_content(
    angle_registry: SemanticRegistry, unit: str, carried: str, clearing: str
) -> None:
    """A fixed suggestion naming one radian is wrong for a steradian and for a torque.

    The exponent has to come from the value, so assert the message names both what the
    quantity carries and the factor that removes it.
    """

    with pytest.raises(DimensionalityError) as excinfo:
        float(angle_registry.Quantity(1.0, unit))

    message = str(excinfo.value)
    assert f"carries {carried};" in message
    assert f"multiplying by {clearing}." in message


@pytest.mark.parametrize(("unit", "carried", "clearing"), ANGULAR_COERCION_CASES)
def test_every_suggested_remedy_actually_clears_the_angle(
    angle_registry: SemanticRegistry, unit: str, carried: str, clearing: str
) -> None:
    """Advice that does not work is worse than Pint's bare error, so execute it."""

    cleared = angle_registry.Quantity(1.0, unit) * angle_registry.Unit(clearing)

    assert not [name for name in cleared.dimensionality if "angle" in name]
    if cleared.dimensionless:
        float(cleared)  # a pure angle becomes a plain number; a torque becomes joules


@pytest.mark.parametrize(("unit", "carried", "clearing"), ANGULAR_COERCION_CASES)
def test_every_offered_m_as_target_actually_converts(
    angle_registry: SemanticRegistry, unit: str, carried: str, clearing: str
) -> None:
    """Whenever the message offers m_as, that exact target must convert.

    Driven from the same table as the other coercion tests so a new case cannot be
    added without its m_as suggestion being executed too.
    """

    quantity = angle_registry.Quantity(1.0, unit)

    with pytest.raises(DimensionalityError) as excinfo:
        float(quantity)

    target = re.search(r'\.m_as\("([^"]+)"\)', str(excinfo.value))
    if target is None:
        return  # compound quantities are deliberately offered no m_as; see below
    quantity.m_as(target.group(1))


def test_compound_quantities_are_not_offered_m_as(angle_registry: SemanticRegistry) -> None:
    """No unit shares a torque's dimensionality, so suggesting m_as would be false advice."""

    with pytest.raises(DimensionalityError) as excinfo:
        float(angle_registry.Quantity(1.0, "joule / radian"))

    assert "m_as" not in str(excinfo.value)


def test_reduced_planck_constant_is_action_and_eta_makes_it_angular_momentum(
    angle_registry: SemanticRegistry,
) -> None:
    """Pint's constants carry no angle, so the eta rule applies to them like any formula.

    Redefining hbar to J*s/rad was rejected: it propagates a half-integer angle power
    into planck_length, planck_mass and planck_time, and an inverse angle into the Bohr
    radius. hbar stays action, and eta * hbar is the angular-momentum quantum.
    """

    registry = angle_registry
    eta = 1 / registry.radian
    hbar = registry.Quantity(1.0, "hbar")
    angular_momentum = registry.get_dimensionality(registry.kinds["AngularMomentum"].reference_unit)

    assert hbar.dimensionality != angular_momentum
    assert (hbar * eta).dimensionality == angular_momentum
    (hbar * eta).to("joule * second / radian").as_kind("AngularMomentum")


def test_planck_derived_constants_keep_clean_dimensions(
    angle_registry: SemanticRegistry,
) -> None:
    """The regression guard for the rejected hbar redefinition."""

    registry = angle_registry

    assert registry.get_dimensionality("planck_length") == registry.get_dimensionality("meter")
    assert registry.get_dimensionality("planck_mass") == registry.get_dimensionality("kilogram")
    assert registry.get_dimensionality("planck_time") == registry.get_dimensionality("second")
    assert registry.get_dimensionality("bohr") == registry.get_dimensionality("meter")


def test_quantum_energy_relations_close_with_the_eta_rule(
    angle_registry: SemanticRegistry,
) -> None:
    """E = h nu needs no eta; E = hbar omega needs exactly one, like every r x F."""

    registry = angle_registry
    eta = 1 / registry.radian

    photon = registry.Quantity(1.0, "planck_constant") * registry.Quantity(1.0, "hertz")
    assert photon.to("joule").magnitude == pytest.approx(6.62607015e-34, rel=1e-12)

    omega = registry.Quantity(1.0, "radian / second")
    quantum = registry.Quantity(1.0, "hbar") * eta * omega
    assert quantum.to("joule").magnitude == pytest.approx(1.0545718176461565e-34, rel=1e-12)


def test_non_angle_coercion_keeps_pints_plain_error(angle_registry: SemanticRegistry) -> None:
    """The extra guidance must not attach itself to unrelated dimensional errors."""

    with pytest.raises(DimensionalityError) as excinfo:
        float(angle_registry.Quantity(1.0, "joule"))

    assert "an angle is not a number" not in str(excinfo.value)


def test_explicit_unit_extraction_is_the_escape_hatch(angle_registry: SemanticRegistry) -> None:
    """Both readings stay available; the caller names which one, rather than float() guessing."""

    theta = angle_registry.Quantity(90.0, "degree")

    assert theta.m_as("degree") == pytest.approx(90.0, rel=1e-15)
    assert theta.m_as("radian") == pytest.approx(np.pi / 2, rel=1e-15)


def test_eta_converts_angles_to_numbers_and_carries_the_kinematics(
    angle_registry: SemanticRegistry,
) -> None:
    """eta is 1/radian: the same constant the rotational algebra already needs."""

    registry = angle_registry
    eta = 1 / registry.radian
    theta = registry.Quantity(90.0, "degree")

    assert float((theta * eta).to("")) == pytest.approx(np.pi / 2, rel=1e-15)

    speed = eta * registry.Quantity(2.0, "radian/second") * registry.Quantity(3.0, "meter")
    assert speed.m_as("meter/second") == pytest.approx(6.0, rel=1e-15)


def test_a_bare_number_returns_to_the_angle_world_by_multiplication(
    angle_registry: SemanticRegistry,
) -> None:
    angle = (np.pi / 2) * angle_registry.radian

    assert cast(SemanticQuantity, np.sin(angle)).magnitude == pytest.approx(1.0, abs=1e-15)


def test_angle_and_ordinary_registries_do_not_mix(angle_registry: SemanticRegistry) -> None:
    """Cross-registry rejection is what stops the two conventions ever meeting."""

    with pytest.raises(ValueError, match="different SemanticRegistry"):
        angle_registry.Quantity(1.0, "meter") + default_ureg.Quantity(1.0, "meter")


# --- the switch must not disturb the other semantic features -------------------------


def test_designations_survive_the_angle_registry(angle_registry: SemanticRegistry) -> None:
    """Designations are orthogonal to dimensions, so the switch must leave them alone."""

    registry = angle_registry
    oxygen = 5.0 * registry.Unit("millimole")["O2"] / registry.Unit("liter")

    assert oxygen.has_designation
    with pytest.raises(TypeError, match="different factor designations"):
        5.0 * registry.Unit("millimole")["O2"] + 5.0 * registry.Unit("millimole")["NaCl"]


def test_affine_temperature_rules_survive_the_angle_registry(
    angle_registry: SemanticRegistry,
) -> None:
    """Affine roles are derived from temperature units, which the switch does not touch."""

    registry = angle_registry

    shifted = registry.Quantity(20.0, "degC") + registry.Quantity(5.0, "delta_degC")
    assert shifted.magnitude == pytest.approx(25.0, rel=1e-15)
    assert shifted.affine_role is AffineRole.ABSOLUTE
    with pytest.raises(TypeError, match="Generic Kelvin"):
        registry.Quantity(293.15, "kelvin").to("degC")


def test_kind_policy_survives_the_angle_registry(angle_registry: SemanticRegistry) -> None:
    """Missing and mismatched kinds reject with either angle policy."""

    registry = angle_registry

    with pytest.raises(TypeError, match="quantity kinds"):
        registry.Quantity(5.0, "joule", kind="Energy") + registry.Quantity(2.0, "joule")
    with pytest.raises(DimensionalityError, match="Cannot convert"):
        registry.Quantity(1.0, "joule / radian", kind="Torque") + registry.Quantity(
            1.0, "joule", kind="Energy"
        )
    with pytest.raises(TypeError, match="quantity kinds"):
        registry.Quantity(1.0, "joule / kelvin", kind="Entropy") + registry.Quantity(
            1.0, "joule / kelvin", kind="HeatCapacity"
        )


def test_array_semantics_survive_the_angle_registry(angle_registry: SemanticRegistry) -> None:
    """Kind propagation through the supported NumPy surface is unaffected by the switch."""

    registry = angle_registry
    energies = registry.Quantity(np.array([8.0, 12.0]), "joule", kind="Energy")

    assert cast(SemanticQuantity, energies.mean()).kind == "Energy"
    angles = registry.Quantity(np.array([1.0, 2.0]), "radian")
    assert cast(SemanticQuantity, np.sum(angles)).m_as("radian") == pytest.approx(3.0, rel=1e-15)


def test_trigonometry_still_separates_kinds_from_designations(
    angle_registry: SemanticRegistry,
) -> None:
    """A kinded angle passes; a designated ratio is still refused as not being an angle."""

    registry = angle_registry

    kinded = cast(SemanticQuantity, np.sin(registry.Quantity(1.0, "radian", kind="PlaneAngle")))
    assert kinded.magnitude == pytest.approx(np.sin(1.0), rel=1e-15)

    designated = 1.0 * registry.Unit("meter")["arc"] / registry.Unit("meter")["radius"]
    with pytest.raises(TypeError, match="retained factor designations"):
        np.sin(designated)
