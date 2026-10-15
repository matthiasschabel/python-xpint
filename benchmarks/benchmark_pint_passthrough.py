"""Compare xpint array arithmetic with Pint's baseline."""

from __future__ import annotations

import timeit
from collections.abc import Callable
from typing import Any

import numpy as np
import pint

from xpint import SemanticRegistry


def _best_microseconds(operation: Callable[[], object], *, number: int, repeat: int) -> float:
    return min(timeit.repeat(operation, number=number, repeat=repeat)) * 1e6 / number


def benchmark(
    *,
    size: int = 1_000,
    number: int = 20_000,
    repeat: int = 5,
) -> dict[str, float]:
    """Return best warm array timings and xpint-to-Pint ratios."""

    if size < 1:
        raise ValueError("size must be positive.")

    pint_registry: pint.UnitRegistry[Any] = pint.UnitRegistry()
    semantic_registry = SemanticRegistry()
    left = np.ones(size)
    right = np.full(size, 2.0)
    pint_left = pint_registry.Quantity(left, "meter")
    pint_right = pint_registry.Quantity(right, "meter")
    xpint_left = semantic_registry.Quantity(left, "meter")
    xpint_right = semantic_registry.Quantity(right, "meter")
    kind_left = semantic_registry.Quantity(left, "meter", kind="Length")
    kind_right = semantic_registry.Quantity(right, "meter", kind="Length")

    timings = {
        "pint_add_us": _best_microseconds(
            lambda: pint_left + pint_right,
            number=number,
            repeat=repeat,
        ),
        "xpint_plain_add_us": _best_microseconds(
            lambda: xpint_left + xpint_right,
            number=number,
            repeat=repeat,
        ),
        "xpint_kind_add_us": _best_microseconds(
            lambda: kind_left + kind_right,
            number=number,
            repeat=repeat,
        ),
        "pint_multiply_us": _best_microseconds(
            lambda: pint_left * 2,
            number=number,
            repeat=repeat,
        ),
        "xpint_plain_multiply_us": _best_microseconds(
            lambda: xpint_left * 2,
            number=number,
            repeat=repeat,
        ),
        "xpint_kind_multiply_us": _best_microseconds(
            lambda: kind_left * 2,
            number=number,
            repeat=repeat,
        ),
    }
    timings["plain_add_ratio"] = timings["xpint_plain_add_us"] / timings["pint_add_us"]
    timings["kind_add_ratio"] = timings["xpint_kind_add_us"] / timings["pint_add_us"]
    timings["plain_multiply_ratio"] = (
        timings["xpint_plain_multiply_us"] / timings["pint_multiply_us"]
    )
    timings["kind_multiply_ratio"] = timings["xpint_kind_multiply_us"] / timings["pint_multiply_us"]
    return timings


if __name__ == "__main__":
    for name, value in benchmark().items():
        print(f"{name}: {value:.3f}")
