"""Measure semantic safety overhead against Pint or an isolated xpint snapshot."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import operator
import platform
import statistics
import sys
import time
import timeit
import tracemalloc
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pint

_DEFAULT_SIZES = (0, 4, 1_000, 1_000_000)
_WARMUP_CALLS = 3
_CALIBRATION_CALLS = 10


def _registry_factory(
    backend: Literal["pint", "xpint"], source_root: Path | None
) -> tuple[Callable[..., Any], str]:
    if backend == "pint":
        if source_root is not None:
            raise ValueError("source_root applies only to the xpint backend")
        return pint.UnitRegistry, str(pint.__file__)
    if source_root is None:
        module = importlib.import_module("xpint")
    else:
        source_root = source_root.resolve()
        expected = source_root / "xpint" / "__init__.py"
        if not expected.is_file():
            raise ValueError(f"source_root must contain xpint/__init__.py: {source_root}")
        existing = sys.modules.get("xpint")
        if existing is not None and Path(str(existing.__file__)).resolve() != expected:
            raise RuntimeError("a different xpint source is already imported; use a fresh process")
        sys.path.insert(0, str(source_root))
        try:
            module = importlib.import_module("xpint")
        finally:
            sys.path.pop(0)
        if Path(str(module.__file__)).resolve() != expected:
            raise RuntimeError(f"xpint did not load from the requested source: {expected}")
    return module.SemanticRegistry, str(module.__file__)


def _operations(
    registry: Any,
    quantity: Any,
    other: Any,
    centimeter: Any,
    meter: Any,
    power_quantity: Any | None,
) -> dict[str, Callable[[], Any]]:
    def ito_pair() -> None:
        quantity.ito(centimeter)
        quantity.ito(meter)

    operations: dict[str, Callable[[], Any]] = {
        "add": lambda: quantity + other,
        "mul_scalar": lambda: quantity * 2,
        "to_unit": lambda: quantity.to(centimeter),
        "to_string": lambda: quantity.to("centimeter"),
        "ito_pair": ito_pair,
        "np_sum": lambda: np.sum(quantity),
        "registry_convert_unit": lambda: registry.convert(quantity.magnitude, meter, centimeter),
        "registry_convert_string": lambda: registry.convert(
            quantity.magnitude, "meter", "centimeter"
        ),
    }
    if power_quantity is not None:

        def ipow_pair() -> None:
            operand = power_quantity
            operand **= 2
            operand **= 0.5

        def imul_one() -> None:
            operand = quantity
            operand *= 1

        operations.update(
            {
                "method_sum": lambda: quantity.sum(),
                "ipow_pair": ipow_pair,
                "imul_one": imul_one,
                "shape": lambda: quantity.shape,
                "flatten": lambda: quantity.flatten(),
            }
        )
    return operations


def _cases(
    factory: Callable[..., Any], *, semantic: bool, sizes: tuple[int, ...]
) -> dict[str, Callable[[], Any]]:
    registry = factory()
    automatic = factory(auto_reduce_dimensions=True)

    def make_quantity(owner: Any, magnitude: Any, units: Any, *, kind: bool = False) -> Any:
        if semantic:
            return owner.Quantity(magnitude, units, kind="Length" if kind else None)
        return owner.Quantity(magnitude, units)

    def make_unit(owner: Any, name: str, designation: str | None = None) -> Any:
        units = owner.Unit(name)
        return units[designation] if semantic and designation is not None else units

    def auto_divide(numerator: Any, denominator: Any) -> Callable[[], Any]:
        return lambda: numerator / denominator

    cases: dict[str, Callable[[], Any]] = {}
    for size in sizes:
        shape = str(size) if size else "scalar"
        array_magnitude = np.ones(size, dtype=np.float64)
        magnitude = array_magnitude if size else 1.0
        for kind in (False, True):
            mode = "kind" if kind else "plain"
            quantity = make_quantity(
                registry, array_magnitude.copy() if size else magnitude, "meter", kind=kind
            )
            other = make_quantity(registry, magnitude, "meter", kind=kind)
            power_quantity = (
                make_quantity(
                    registry,
                    array_magnitude.copy(),
                    make_unit(registry, "meter", "a" if kind else None),
                )
                if size
                else None
            )
            for name, operation in _operations(
                registry, quantity, other, registry.centimeter, registry.meter, power_quantity
            ).items():
                # Squaring drops kinds; designations keep repeated powers classified.
                case_mode = "designated" if kind and name == "ipow_pair" else mode
                cases[f"{shape}/{case_mode}/{name}"] = operation
        numerator = make_quantity(automatic, magnitude, make_unit(automatic, "meter", "a"))
        denominator = make_quantity(automatic, 1.0, make_unit(automatic, "centimeter", "b"))
        cases[f"{shape}/designated/auto_divide"] = auto_divide(numerator, denominator)
        temperature = registry.Quantity(magnitude * 20, "degC")
        delta = registry.Quantity(magnitude, "delta_degC")
        cases[f"{shape}/absolute/to_kelvin"] = partial(temperature.to, "kelvin")
        cases[f"{shape}/absolute/add_delta"] = partial(operator.add, temperature, delta)
        cases[f"{shape}/absolute/subtract"] = partial(operator.sub, temperature, temperature)
        if size:
            left = registry.Quantity(1, "joule", **({"kind": "Energy"} if semantic else {}))
            right = registry.Quantity(
                magnitude, "joule", **({"kind": "Torque"} if semantic else {})
            )
            cases[f"{shape}/kind/scalar_mismatch_equal"] = partial(operator.eq, left, right)
    if 1_000_000 in sizes:
        matrix = np.ones((1_000, 1_000), dtype=np.float64)
        for layout, matrix_magnitude in (
            ("C", matrix),
            ("F", np.asfortranarray(matrix)),
            ("transposed", matrix.T),
        ):
            quantity = make_quantity(registry, matrix_magnitude, "meter", kind=True)
            cases[f"matrix_{layout}/kind/imul_one"] = _operations(
                registry, quantity, quantity, registry.centimeter, registry.meter, quantity
            )["imul_one"]
    return cases


def _measure(
    operation: Callable[[], Any], *, repeat: int, target_seconds: float, max_iterations: int
) -> dict[str, Any]:
    for _ in range(_WARMUP_CALLS):
        operation()
    timer = timeit.Timer(operation)
    estimate = timer.timeit(_CALIBRATION_CALLS) / _CALIBRATION_CALLS
    iterations = min(max_iterations, max(1, math.ceil(target_seconds / max(estimate, 1e-9))))
    samples = [elapsed * 1e6 / iterations for elapsed in timer.repeat(repeat, iterations)]
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        result = operation()
        peak_bytes = tracemalloc.get_traced_memory()[1] - before
        # Keep a returned array alive through the peak measurement.
        del result
    finally:
        tracemalloc.stop()
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "samples_us": samples,
        "iterations": iterations,
        "peak_extra_bytes": peak_bytes,
    }


def benchmark(
    *,
    backend: Literal["pint", "xpint"] = "xpint",
    source_root: Path | None = None,
    sizes: tuple[int, ...] = _DEFAULT_SIZES,
    repeat: int = 5,
    target_seconds: float = 0.025,
    max_iterations: int = 10_000,
) -> dict[str, Any]:
    """Return warm timings and traced allocation peaks for one backend.

    Operands and registries are constructed outside the timing region. Sizes
    select one-dimensional float64 arrays, with zero selecting scalar inputs.
    Including 1,000,000 also measures in-place multiplication on 1,000 x 1,000
    matrices in C, Fortran, and transposed layouts.
    Conversion and power cases perform two operations per sample so repeated
    calls retain bounded magnitudes and units. Pint's kind/designation cases
    measure equivalent physical arithmetic without semantic safety guarantees.

    Args:
        backend: Measure Pint or xpint in this process.
        source_root: Optional directory containing ``xpint/__init__.py``, such
            as an extracted snapshot's ``src`` directory. Requires a fresh
            process if a different xpint source has already been imported.
        sizes: Distinct nonnegative array lengths; zero means scalar.
        repeat: Number of timed batches for each case.
        target_seconds: Desired duration of each adaptively sized batch.
        max_iterations: Upper bound on operations per timed batch.

    Returns:
        JSON-serializable metadata, timing results, failure count, and elapsed
        seconds. Each failed case records its exception and does not stop the
        other cases. Allocation peaks exclude inputs and are traced bytes,
        not process RSS. ``timeit`` disables cyclic garbage collection during
        timing; allocation tracing runs separately afterward.

    Raises:
        ValueError: If options are invalid or the requested source is missing.
        RuntimeError: If an incompatible source is imported or allocation
            tracing is already active.
    """
    if backend not in {"pint", "xpint"}:
        raise ValueError("backend must be 'pint' or 'xpint'")
    if not sizes or any(size < 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("sizes must contain distinct nonnegative integers")
    if repeat < 1 or max_iterations < 1:
        raise ValueError("repeat and max_iterations must be positive")
    if not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError("target_seconds must be finite and positive")
    if tracemalloc.is_tracing():
        raise RuntimeError("stop existing allocation tracing before running the benchmark")

    started = time.perf_counter()
    factory, source = _registry_factory(backend, source_root)
    results: dict[str, Any] = {}
    failures = 0
    for name, operation in _cases(factory, semantic=backend == "xpint", sizes=sizes).items():
        try:
            results[name] = _measure(
                operation,
                repeat=repeat,
                target_seconds=target_seconds,
                max_iterations=max_iterations,
            )
        except Exception as error:
            # A benchmark must retain successful cases when one backend rejects an operation.
            results[name] = {"error": f"{type(error).__name__}: {error}"}
            failures += 1
    return {
        "metadata": {
            "backend": backend,
            "source": source,
            "python": sys.version,
            "pint": pint.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "sizes": sizes,
            "repeat": repeat,
            "target_seconds": target_seconds,
            "max_iterations": max_iterations,
        },
        "results": results,
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pint", "xpint"), default="xpint")
    parser.add_argument("--source-root", type=Path, help="directory containing xpint/__init__.py")
    parser.add_argument("--sizes", type=int, nargs="+", default=_DEFAULT_SIZES)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--target-seconds", type=float, default=0.025)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    options = parser.parse_args()
    try:
        report = benchmark(
            backend=options.backend,
            source_root=options.source_root,
            sizes=tuple(options.sizes),
            repeat=options.repeat,
            target_seconds=options.target_seconds,
            max_iterations=options.max_iterations,
        )
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    failures = report["failures"]
    count = len(report["results"])
    print(
        f"{options.backend}: {count - failures}/{count} cases succeeded; "
        f"{failures} failed; {report['elapsed_seconds']:.2f} seconds",
        file=sys.stderr,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
