"""Compare SI dimensionless angles with xpint's optional angle dimension.

Each section evaluates the same operation with the default registry and with a
registry built using ``angle_dimension=True``.
"""

from typing import Any

import numpy as np
from pint.errors import DimensionalityError

from xpint import SemanticRegistry

plain = SemanticRegistry()
angled = SemanticRegistry(angle_dimension=True)


def _outcome(description: str, action: object) -> None:
    print(f"    {description:<34} {action}")


def torque_is_not_energy() -> None:
    """A bolt's tightening torque and an energy budget share a dimension in SI."""

    print("1. Torque against energy")

    work = plain.Quantity(10.0, "joule")
    torque = plain.Quantity(10.0, "newton * meter")
    _outcome("plain: N*m + J", work + torque)

    torque = angled.Quantity(10.0, "joule / radian")
    try:
        angled.Quantity(10.0, "joule") + torque
    except DimensionalityError as error:
        _outcome("angled: J + J/rad", f"rejected ({str(error).split(';')[0][:44]}…)")
    _outcome("angled: torque * angle", (torque * angled.Quantity(2.0, "radian")).to("joule"))


def _larmor_frequency(registry: SemanticRegistry, gamma: Any, magnetic_flux_density: Any) -> Any:
    """Return cyclic Larmor frequency from an angular or cyclic ratio."""

    angular_units = registry.radian / registry.second / registry.tesla
    cyclic_units = registry.hertz / registry.tesla

    if gamma.is_compatible_with(angular_units):
        return (gamma * magnetic_flux_density / registry.turn).to(registry.megahertz)
    if gamma.is_compatible_with(cyclic_units):
        return (gamma * magnetic_flux_density).to(registry.megahertz)
    raise ValueError("gamma must have angular-rate-per-tesla or cyclic-frequency-per-tesla units")


def gyromagnetic_ratio_dispatch() -> None:
    """Normalize angular and cyclic gyromagnetic ratios at an API boundary."""

    print("\n2. Gyromagnetic-ratio input normalization")

    gamma_angular_value = 2.6752218708e8
    gamma_cyclic_value = 42.577478461

    plain_flux_density = 3 * plain.tesla
    plain_angular = gamma_angular_value * plain.radian / plain.second / plain.tesla
    plain_cyclic = gamma_cyclic_value * plain.megahertz / plain.tesla
    _outcome("plain: angular input", _larmor_frequency(plain, plain_angular, plain_flux_density))
    _outcome("plain: cyclic input", _larmor_frequency(plain, plain_cyclic, plain_flux_density))

    angled_flux_density = 3 * angled.tesla
    angled_angular = gamma_angular_value * angled.radian / angled.second / angled.tesla
    angled_cyclic = gamma_cyclic_value * angled.megahertz / angled.tesla
    _outcome(
        "angled: angular input",
        _larmor_frequency(angled, angled_angular, angled_flux_density),
    )
    _outcome(
        "angled: cyclic input",
        _larmor_frequency(angled, angled_cyclic, angled_flux_density),
    )


def solid_angle_is_not_an_angle() -> None:
    """Radian and steradian are both scale-1 dimensionless, so SI cannot separate them.

    Under the switch the steradian is not a second base dimension; it is rad**2, which
    is what a patch on a sphere works out to. That still separates it from a plane
    angle, because [angle] and [angle]**2 are different dimensions.
    """

    print("\n3. Solid angle against plane angle")

    _outcome("plain: sin(1 steradian)", np.sin(plain.Quantity(1.0, "steradian")))
    _outcome(
        "plain: 1 sr + 1 rad", plain.Quantity(1.0, "steradian") + plain.Quantity(1.0, "radian")
    )

    for label, call in (
        ("angled: sin(1 steradian)", lambda: np.sin(angled.Quantity(1.0, "steradian"))),
        (
            "angled: 1 sr + 1 rad",
            lambda: angled.Quantity(1.0, "steradian") + angled.Quantity(1.0, "radian"),
        ),
    ):
        try:
            call()
        except DimensionalityError:
            _outcome(label, "rejected")

    _outcome("angled: dimension of steradian", angled.get_dimensionality("steradian"))
    beam = angled.Quantity(1.0, "milliradian")
    _outcome("angled: 1 mrad * 1 mrad in usr", (beam * beam).m_as("microsteradian"))
    _outcome(
        "plain: 1 mrad * 1 mrad in usr",
        (plain.Quantity(1.0, "milliradian") ** 2).m_as("microsteradian"),
    )


def photometry_dimensions() -> None:
    """Lumen is candela times steradian, so luminous flux and intensity follow the switch."""

    print("\n4. Luminous flux against luminous intensity")

    _outcome(
        "plain: lm and cd share a dimension",
        plain.get_dimensionality("lumen") == plain.get_dimensionality("candela"),
    )
    _outcome(
        "angled: lm and cd share a dimension",
        angled.get_dimensionality("lumen") == angled.get_dimensionality("candela"),
    )
    _outcome("angled: dimension of lumen", angled.get_dimensionality("lumen"))


def rotational_mechanics_still_balances() -> None:
    """Verify standard rotational identities under the conjugate dimensions."""

    print("\n5. Rotational mechanics under the conjugate spellings")

    inertia = angled.Quantity(2.0, "kilogram * meter ** 2 / radian ** 2")
    omega = angled.Quantity(3.0, "radian / second")

    _outcome("L = I * omega", (inertia * omega).to("joule * second / radian"))
    _outcome("KE = 1/2 I omega^2", (0.5 * inertia * omega**2).to("joule"))
    _outcome("P = tau * omega", (angled.Quantity(4.0, "joule / radian") * omega).to("watt"))


def leaving_the_angle_world() -> None:
    """An angle is not a number here, so the caller names the convention rather than float()."""

    print("\n6. Getting a plain number back out")

    theta = angled.Quantity(90.0, "degree")
    _outcome("plain: float(90 degree)", float(plain.Quantity(90.0, "degree")))
    try:
        float(theta)
    except DimensionalityError as error:
        _outcome("angled: float(90 degree)", f"rejected: {str(error).split(';')[-1].strip()[:46]}…")

    _outcome("angled: theta.m_as('degree')", theta.m_as("degree"))
    _outcome("angled: theta.m_as('radian')", theta.m_as("radian"))

    eta = 1 / angled.radian
    _outcome("angled: eta = 1 / radian", float((theta * eta).to("")))
    speed = eta * angled.Quantity(2.0, "radian/second") * angled.Quantity(3.0, "meter")
    _outcome("angled: v = eta * omega * r", speed.to("meter / second"))


def main() -> None:
    torque_is_not_energy()
    gyromagnetic_ratio_dispatch()
    solid_angle_is_not_an_angle()
    photometry_dimensions()
    rotational_mechanics_still_balances()
    leaving_the_angle_world()

    print("\nCross-registry operations are rejected.")


if __name__ == "__main__":
    main()
