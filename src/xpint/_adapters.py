"""Guard Pint's numerical adapters before they strip semantic objects."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from inspect import Parameter, signature
from typing import Any

from pint import registry_helpers
from pint.compat import np  # type: ignore[attr-defined]
from pint.util import UnitsContainer, pi_theorem


def _require_plain(registry: Any, value: Any, operation: str) -> None:
    registry._require_owned(value, name=f"{operation} input")
    if isinstance(value, (registry.Quantity, registry.Unit)):
        if registry.Quantity._operands_have_semantics(value):
            raise TypeError(
                f"{operation} cannot represent classified quantity metadata; "
                "explicitly remove it or operate on magnitude first"
            )
    elif isinstance(value, dict):
        for item in value.values():
            _require_plain(registry, item, operation)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _require_plain(registry, item, operation)
    elif np is not None and isinstance(value, np.ndarray) and value.dtype == np.dtype(object):
        for item in value.flat:
            _require_plain(registry, item, operation)


def _require_plain_unit_spec(registry: Any, value: Any, operation: str) -> None:
    if value is None:
        return
    if isinstance(value, str) and "=" not in value and "[" not in value:
        value = registry.Unit(value)
    _require_plain(registry, value, operation)


def _require_plain_wraps_input(registry: Any, value: Any, strict: bool) -> None:
    if strict and isinstance(value, str):
        value = registry.parse_expression(value)
    registry._require_owned(value, name="wraps input")
    if isinstance(value, (registry.Quantity, registry.Unit)):
        _require_plain(registry, value, "wraps")


def _wraps_preserves_variadic_passthrough(registry: Any) -> bool:
    """Check Pint's binding behavior without calling the user's function."""
    probe = registry.Quantity(1, "")

    def preserves_probe(*values: Any, scale: Any) -> bool:
        return values[1] is probe

    wrapped = registry_helpers.wraps(registry, None, (None, ""))(preserves_probe)
    return bool(wrapped(None, probe, scale=probe))


def _wraps(
    registry: Any, ret: Any, args: Any, strict: bool
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    argument_specs = args if isinstance(args, (list, tuple)) else (args,)
    return_specs = ret if isinstance(ret, (list, tuple)) else (ret,)
    for spec in argument_specs:
        _require_plain_unit_spec(registry, spec, "wraps")
    for spec in return_specs:
        registry._require_owned(spec, name="wraps return unit")
        if isinstance(spec, registry.Unit) and spec.has_designation:
            _require_plain(registry, spec, "wraps")
    pint_decorator = registry_helpers.wraps(registry, ret, args, strict)

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        wrapped = pint_decorator(function)
        sig = signature(function)
        # Pint validates declaration counts; an omitted **kwargs specification
        # means pass-through in converters that support omitting it.
        declarations = tuple(zip(sig.parameters.values(), argument_specs, strict=False))
        # Every value bound to a declared unit is a conversion candidate: a converter
        # that binds by parameter name strips each *args element and each **kwargs
        # value under that parameter's declaration, not just the first one.
        declared = tuple(
            (parameter.name, parameter.kind) for parameter, spec in declarations if spec is not None
        )
        # Pint 0.25.3 addresses declarations by positional slot instead, so a declared
        # parameter after *args strips the variadic element occupying its index. Guard
        # those slots unless the declaration covers them or Pint binds them correctly.
        misbound_kinds = (Parameter.KEYWORD_ONLY, Parameter.VAR_KEYWORD)
        misbound_slots = tuple(
            index
            for index, (parameter, spec) in enumerate(declarations)
            if spec is not None and parameter.kind in misbound_kinds
        )
        if any(kind is Parameter.VAR_POSITIONAL for _name, kind in declared) or (
            misbound_slots
            and any(
                parameter.kind is Parameter.VAR_POSITIONAL for parameter in sig.parameters.values()
            )
            and _wraps_preserves_variadic_passthrough(registry)
        ):
            misbound_slots = ()

        @wraps(function)
        def guarded(*values: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*values, **kwargs)
            bound.apply_defaults()
            for name, kind in declared:
                value = bound.arguments[name]
                if kind is Parameter.VAR_POSITIONAL:
                    for item in value:
                        _require_plain_wraps_input(registry, item, strict)
                elif kind is Parameter.VAR_KEYWORD:
                    for item in value.values():
                        _require_plain_wraps_input(registry, item, strict)
                else:
                    _require_plain_wraps_input(registry, value, strict)
            for index in misbound_slots:
                if index < len(values):
                    _require_plain_wraps_input(registry, values[index], strict)
            return wrapped(*values, **kwargs)

        return guarded

    return decorate


def _check(
    registry: Any, dimensions: tuple[Any, ...]
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    for dimension in dimensions:
        registry._require_owned(dimension, name="check dimension")
    return registry_helpers.check(registry, *dimensions)


def _pi_theorem(registry: Any, quantities: dict[str, Any]) -> Any:
    for value in quantities.values():
        if isinstance(value, (dict, UnitsContainer)):
            value = registry.Unit(registry.UnitsContainer(value))
        _require_plain_unit_spec(registry, value, "pi_theorem")
    return pi_theorem(quantities, registry)


def _setup_matplotlib(registry: Any, enable: bool) -> None:
    try:
        import matplotlib.units
        from pint.matplotlib import setup_matplotlib_handlers

        from ._matplotlib import _SemanticConverter
    except ImportError as exc:
        raise ImportError("setup_matplotlib requires matplotlib; install matplotlib first") from exc
    setup_matplotlib_handlers(registry, enable)  # type: ignore[no-untyped-call]
    if enable:
        converters = matplotlib.units.registry
        converters[registry.Quantity] = _SemanticConverter(registry)  # type: ignore[no-untyped-call]
