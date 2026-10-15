"""Immutable semantic dispatch tables and built-in registry catalogs."""

from types import MappingProxyType

_NUMPY_METHOD_PRESERVING = frozenset(
    {
        "astype",
        "compress",
        "conj",
        "conjugate",
        "copy",
        "diagonal",
        "flatten",
        "item",
        "max",
        "mean",
        "min",
        "ravel",
        "repeat",
        "reshape",
        "round",
        "squeeze",
        "swapaxes",
        "take",
        "transpose",
    }
)
_NUMPY_METHOD_ADDITIVE = frozenset({"sum", "cumsum"})
# These methods share argument names with the corresponding NumPy functions.
_NUMPY_METHOD_REDUCTIONS = frozenset({"sum", "max", "min", "mean", "std", "var", "cumsum"})
_NUMPY_METHOD_SUPPORTED = (
    _NUMPY_METHOD_PRESERVING | _NUMPY_METHOD_ADDITIVE | frozenset({"std", "var"})
)

_NUMPY_UFUNC_BINARY_METHODS = MappingProxyType(
    {
        "add": ("__add__", "__radd__"),
        "subtract": ("__sub__", "__rsub__"),
        "multiply": ("__mul__", "__rmul__"),
        "divide": ("__truediv__", "__rtruediv__"),
        "true_divide": ("__truediv__", "__rtruediv__"),
        "power": ("__pow__", "__rpow__"),
        "equal": ("__eq__", "__eq__"),
        "not_equal": ("__ne__", "__ne__"),
        "less": ("__lt__", "__gt__"),
        "less_equal": ("__le__", "__ge__"),
        "greater": ("__gt__", "__lt__"),
        "greater_equal": ("__ge__", "__le__"),
    }
)
_NUMPY_UFUNC_MAGNITUDE_PREDICATES = frozenset({"isfinite", "isinf", "isnan", "signbit"})

# Trigonometry consumes or produces an angle, so an input kind never describes the
# output. Pint converts the argument to radian and rejects a non-angle; xpint only has
# to prevent its own metadata from crossing the boundary silently.
_ANGLE_UFUNCS = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "arctan2",
        "degrees",
        "radians",
        "deg2rad",
        "rad2deg",
    }
)

_NUMPY_FUNCTION_PRESERVING = frozenset({"mean", "nanmean", "max", "min", "amax", "amin"})
_NUMPY_FUNCTION_ADDITIVE = frozenset({"sum", "nansum", "cumsum", "nancumsum"})
_NUMPY_STATISTICS = frozenset({"std", "nanstd", "var", "nanvar"})
_NUMPY_VARIANCES = frozenset({"var", "nanvar"})
_NUMPY_MAGNITUDE_INDICES = frozenset({"argmax", "argmin", "argsort", "nonzero"})
_NUMPY_UFUNC_POWERS = MappingProxyType({"sqrt": 0.5, "square": 2, "cbrt": 1 / 3, "reciprocal": -1})
_NUMPY_FUNCTION_REDUCTIONS = (
    _NUMPY_FUNCTION_PRESERVING | _NUMPY_FUNCTION_ADDITIVE | _NUMPY_STATISTICS
)
_NUMPY_FUNCTION_LIKE_CONSTRUCTORS = frozenset({"zeros_like"})
_NUMPY_FUNCTION_COMPOSITION = frozenset({"concatenate", "stack", "where"})

# Canonical unit per angular dimension, used to tell a caller what a rejected value
# actually carries and which factor clears it.
_ANGULAR_UNITS = MappingProxyType({"[angle]": "radian"})

# Pint declares [luminous_flux] and [illuminance] independently of the units carrying
# them, so redefining steradian alone leaves those named dimensions inconsistent.
_ANGLE_DEFINITIONS = (
    "radian = [angle] = rad",
    # A small patch on a sphere is a rectangle whose sides are arcs, so its solid angle
    # is a product of two plane angles with the radius and every constant cancelling.
    # Leonard (Metrologia 58 052001, 2021) and Kalinin (Metrologia 56 065009, 2019) both
    # reach the same place: the steradian is a coherent derived unit, sr = rad**2.
    "steradian = radian ** 2 = sr",
    "[luminous_flux] = [luminosity] * [angle] ** 2",
    "[illuminance] = [luminous_flux] / [length] ** 2",
)

# Conjugate forms. Generalized force is energy per coordinate and conjugate momentum is
# action per coordinate, so once angle is a coordinate the dimension appears on its own;
# newton * meter carries no angle and would leave torque indistinguishable from energy.
_ANGLE_KIND_OVERRIDES = MappingProxyType(
    {
        "Dimensionless": "dimensionless",
        "Torque": "joule / radian",
        "AngularMomentum": "joule * second / radian",
        "MomentOfInertia": "kilogram * meter ** 2 / radian ** 2",
    }
)

_DEFAULT_KINDS = MappingProxyType(
    {
        "Dimensionless": "radian",
        "Energy": "joule",
        "Torque": "newton * meter",
        "PlaneAngle": "radian",
        "SolidAngle": "steradian",
        "Frequency": "hertz",
        "AngularVelocity": "radian / second",
        "Action": "joule * second",
        "AngularMomentum": "kilogram * meter ** 2 / second",
        "Temperature": "kelvin",
        "TemperatureDifference": "kelvin",
        "Length": "meter",
        "Time": "second",
        "Mass": "kilogram",
        "ElectricCurrent": "ampere",
        "AmountOfSubstance": "mole",
        "LuminousIntensity": "candela",
        "Force": "newton",
        "Pressure": "pascal",
        "EnergyDensity": "joule / meter ** 3",
        "Power": "watt",
        "ElectricCharge": "coulomb",
        "ElectricPotential": "volt",
        "Capacitance": "farad",
        "Resistance": "ohm",
        "Conductance": "siemens",
        "MagneticFlux": "weber",
        "MagneticFluxDensity": "tesla",
        "Inductance": "henry",
        "LuminousFlux": "lumen",
        "Illuminance": "lux",
        "Radioactivity": "becquerel",
        "AbsorbedDose": "gray",
        "EquivalentDose": "sievert",
        "CatalyticActivity": "katal",
        "Velocity": "meter / second",
        "Acceleration": "meter / second ** 2",
        "Area": "meter ** 2",
        "Volume": "meter ** 3",
        "Density": "kilogram / meter ** 3",
        "Entropy": "joule / kelvin",
        "HeatCapacity": "joule / kelvin",
        "KinematicViscosity": "meter ** 2 / second",
        "Diffusivity": "meter ** 2 / second",
    }
)
