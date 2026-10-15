"""Preserve xpint semantics through the supported NumPy array surface."""

from typing import cast

import numpy as np

from xpint import SemanticQuantity
from xpint.units import J, mM, ureg


def main() -> None:
    energies = ureg.Quantity(np.array([8.0, 12.0, 10.0]), J, kind="Energy")
    average_energy = energies.mean()
    total_energy = cast(SemanticQuantity, np.sum(energies))

    print(f"energies: {energies}")
    print(f"average: {average_energy} ({average_energy.kind})")
    print(f"total: {total_energy} ({total_energy.kind})")

    oxygen = ureg.Quantity(np.array([4.8, 5.1, 5.0]), mM["O2"])
    print(f"mean oxygen: {oxygen.mean()}")

    try:
        oxygen[0] = 4.8 * mM["NaCl"]
    except TypeError as error:
        print(f"mismatched array assignment rejected: {error}")

    try:
        np.asarray(energies)
    except TypeError as error:
        print(f"implicit ndarray downcast rejected: {error}")

    explicit_magnitudes = np.asarray(energies.magnitude)
    print(f"explicit magnitudes: {explicit_magnitudes}")


if __name__ == "__main__":
    main()
