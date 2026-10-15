"""xpint — extended semantic quantities for Pint."""

from importlib.metadata import version

from .registry import (
    AffineRole,
    DesignationSignature,
    DesignationTerm,
    KindDefinition,
    SemanticQuantity,
    SemanticRegistry,
    SemanticUnit,
    get_application_registry,
    get_default_registry,
    set_application_registry,
)

__version__ = version("xpint")

__all__ = [
    "AffineRole",
    "DesignationSignature",
    "DesignationTerm",
    "KindDefinition",
    "SemanticQuantity",
    "SemanticRegistry",
    "SemanticUnit",
    "__version__",
    "get_application_registry",
    "get_default_registry",
    "set_application_registry",
]
