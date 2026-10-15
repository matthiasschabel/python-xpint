"""Distinguish absolute temperatures, temperature differences, and generic Kelvin."""

from xpint.units import K, degC, delta_degC


def main() -> None:
    room = 20 * degC
    warming = 5 * delta_degC
    warmer_room = room + warming
    measured_change = warmer_room - room

    print(f"room: {room} ({room.affine_role})")
    print(f"warming: {warming} ({warming.affine_role})")
    print(f"warmer room: {warmer_room} ({warmer_room.affine_role})")
    print(f"measured change: {measured_change} ({measured_change.affine_role})")
    print(f"absolute conversion: {room.to(K)}")
    print(f"delta conversion: {warming.to(K)}")

    generic_kelvin = 293.15 * K
    try:
        generic_kelvin.to(degC)
    except TypeError as error:
        print(f"generic Kelvin conversion rejected: {error}")

    absolute_kelvin = generic_kelvin.as_kind("Temperature")
    print(f"classified Kelvin conversion: {absolute_kelvin.to(degC)}")

    try:
        room + room
    except TypeError as error:
        print(f"absolute-temperature addition rejected: {error}")


if __name__ == "__main__":
    main()
