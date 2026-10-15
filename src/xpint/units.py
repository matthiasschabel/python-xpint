"""xpint.units — concise unit symbols bound to one semantic registry.

Import symbols explicitly so scientific expressions remain readable while the
owning registry remains evident at the import site. Additional unit names and
prefixes resolve lazily through ``ureg`` and are cached in this module; the
registry itself is never mutated.
"""

from typing import cast

from pint.errors import UndefinedUnitError

from .registry import SemanticUnit, get_default_registry

ureg = get_default_registry()

J = ureg.joule
K = ureg.kelvin
L = ureg.liter
N = ureg.newton
degC = ureg.degC
degF = ureg.degF
degR = ureg.degR
delta_degC = ureg.delta_degC
delta_degF = ureg.delta_degF
delta_degR = ureg.delta_degR
h = ureg.hour
m = ureg.meter
mL = ureg.milliliter
mM = ureg.millimolar
mmol = ureg.millimole
mol = ureg.mole
s = ureg.second

__all__ = [
    "J",
    "K",
    "L",
    "N",
    "degC",
    "degF",
    "degR",
    "delta_degC",
    "delta_degF",
    "delta_degR",
    "h",
    "m",
    "mL",
    "mM",
    "mmol",
    "mol",
    "s",
    "ureg",
]


def __getattr__(name: str) -> SemanticUnit:
    """Resolve additional Pint unit names and prefixes through the shared registry.

    Raises:
        AttributeError: If name does not identify a unit.
    """
    try:
        canonical = ureg.get_name(name)
        unit = cast(SemanticUnit, getattr(ureg, canonical))
    except (UndefinedUnitError, AttributeError) as exc:
        raise AttributeError(f"module {__name__!r} has no unit {name!r}") from exc
    globals()[name] = unit
    return unit


def __dir__() -> list[str]:
    """List explicit symbols and unit names available through the registry."""
    return sorted(set(globals()) | set(dir(ureg)))
