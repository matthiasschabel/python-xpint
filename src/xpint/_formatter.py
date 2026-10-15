"""Guard registry formatter delegates that cannot represent designations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, override

from pint.delegates import Formatter


def _require_undesignated(value: Any) -> None:
    if getattr(value, "has_designation", False):
        raise TypeError(
            "Pint formatter APIs cannot represent designated objects; "
            "use str() or format() on the xpint quantity or unit instead"
        )


@dataclass
class _GuardedFormatter:
    delegate: Any

    def format_quantity(self, quantity: Any, *args: Any, **kwargs: Any) -> Any:
        _require_undesignated(quantity)
        return self.delegate.format_quantity(quantity, *args, **kwargs)

    def format_unit(self, unit: Any, *args: Any, **kwargs: Any) -> Any:
        _require_undesignated(unit)
        return self.delegate.format_unit(unit, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


class _SemanticFormatter(Formatter):
    @override
    def get_formatter(self, spec: str) -> Any:
        return _GuardedFormatter(super().get_formatter(spec))
