"""Keep Pint measurements within the unclassified quantity surface."""

from __future__ import annotations

from typing import Any, Self, override

from pint.facets.measurement import Measurement
from pint.facets.measurement.objects import MISSING
from pint.util import UnitsContainer

from ._adapters import _require_plain


def _require_measurement_units(registry: Any, units: Any) -> None:
    if isinstance(units, (str, UnitsContainer)):
        units = registry.Unit(units)
    _require_plain(registry, units, "Measurement")


class _SemanticMeasurement(Measurement):
    """Reject semantic inputs before Pint extracts magnitudes and physical units."""

    __hash__ = Measurement.__hash__

    def __new__(cls, value: Any, error: Any = MISSING, units: Any = MISSING) -> Self:
        registry = cls._REGISTRY
        _require_plain(registry, value, "Measurement")
        _require_plain(registry, error, "Measurement")
        # Pint accepts Measurement(ufloat, units) as well as (value, error, units).
        if units is MISSING and hasattr(value, "nominal_value"):
            _require_measurement_units(registry, error)
        else:
            _require_measurement_units(registry, units)
        return super().__new__(cls, value, error, units)  # type: ignore[no-untyped-call,no-any-return]

    @override
    def _check(self, other: Any) -> bool:
        _require_plain(self._REGISTRY, other, "Measurement arithmetic")
        return super()._check(other)

    @override
    def __pow__(self, other: Any) -> Any:
        _require_plain(self._REGISTRY, other, "Measurement exponent")
        return super().__pow__(other)

    @override
    def __ipow__(self, other: Any) -> Any:
        _require_plain(self._REGISTRY, other, "Measurement exponent")
        return super().__ipow__(other)

    @override
    def to(self, other: Any = None, *contexts: Any, **ctx_kwargs: Any) -> Any:
        _require_measurement_units(self._REGISTRY, other)
        return super().to(other, *contexts, **ctx_kwargs)

    @override
    def ito(self, other: Any = None, *contexts: Any, **ctx_kwargs: Any) -> None:
        _require_measurement_units(self._REGISTRY, other)
        super().ito(other, *contexts, **ctx_kwargs)

    @override
    def compare(self, other: Any, op: Any) -> Any:
        _require_plain(self._REGISTRY, other, "Measurement comparison")
        return super().compare(other, op)

    @override
    def __eq__(self, other: Any) -> Any:
        registry: Any = self._REGISTRY
        if getattr(other, "_REGISTRY", registry) is not registry:
            return False
        if registry.Quantity._operands_have_semantics(other):
            return False
        return super().__eq__(other)
