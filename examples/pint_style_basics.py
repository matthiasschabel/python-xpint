"""Follow Pint's familiar quantity, arithmetic, and conversion workflow."""

from math import isclose

from xpint.units import m, s, ureg


def main() -> None:
    distance = 24 * m
    duration = 8 * s
    speed = distance / duration
    converted = speed.to(ureg.inch / ureg.minute)

    print(f"distance: {distance}")
    print(f"duration: {duration}")
    print(f"speed: {speed}")
    print(f"converted: {converted}")

    assert isclose(converted.magnitude, 7086.614173228347, rel_tol=1e-15)


if __name__ == "__main__":
    main()
