"""Optional Pint plotting adapter that rejects semantic metadata loss."""

from __future__ import annotations

from typing import Any, override

from pint.matplotlib import PintConverter

from ._adapters import _require_plain, _require_plain_unit_spec


class _SemanticConverter(PintConverter):
    @override
    def convert(self, value: Any, unit: Any, axis: Any) -> Any:  # type: ignore[override]
        _require_plain(self._reg, value, "matplotlib")
        _require_plain_unit_spec(self._reg, unit, "matplotlib")
        return super().convert(value, unit, axis)  # type: ignore[no-untyped-call]

    @override
    def _convert_value(self, value: Any, unit: Any, axis: Any) -> Any:
        _require_plain(self._reg, value, "matplotlib")
        return super()._convert_value(value, unit, axis)  # type: ignore[no-untyped-call]

    @override
    def axisinfo(self, unit: Any, axis: Any) -> Any:  # type: ignore[override]
        _require_plain_unit_spec(self._reg, unit, "matplotlib")
        return super().axisinfo(unit, axis)  # type: ignore[no-untyped-call]
