"""Check an installed wheel with only its declared runtime dependencies."""

import importlib.util
import math
import sys
from importlib.resources import files
from pathlib import Path

import pint

import xpint
from xpint.units import J, degC, m, mmol, ureg


def main() -> None:
    """Verify scalar behavior and fail-closed adapters without optional packages."""
    if not __debug__:
        raise SystemExit("run this verification without -O or PYTHONOPTIMIZE")
    assert "site-packages" in Path(xpint.__file__).parts, xpint.__file__
    for package in ("numpy", "babel", "uncertainties", "matplotlib", "dask", "xarray"):
        assert importlib.util.find_spec(package) is None, f"unexpected optional package: {package}"
    assert files("xpint").joinpath("py.typed").is_file()
    assert math.isclose((2 * m).to("cm").magnitude, 200, rel_tol=1e-15, abs_tol=0)
    oxygen = (5 * mmol["O2"]).to("mol")
    assert oxygen.has_designation
    assert math.isclose(oxygen.magnitude, 0.005, rel_tol=1e-15, abs_tol=0)
    difference = 25 * degC - 20 * degC
    assert difference.affine_role is xpint.AffineRole.DELTA
    assert math.isclose(difference.magnitude, 5, rel_tol=1e-15, abs_tol=0)
    energy = (5 * J).as_kind("Energy")
    try:
        energy.plus_minus(0.1)
    except TypeError as exc:
        assert "cannot represent classified" in str(exc)
    else:
        raise AssertionError("classified plus_minus did not reject")
    try:
        (5 * J).plus_minus(0.1)
    except RuntimeError as exc:
        assert "uncertainties" in str(exc)
    else:
        raise AssertionError("ordinary plus_minus did not report the missing dependency")
    try:
        ureg.formatter.format_quantity(oxygen, length=None, locale=None, use_plural=False)
    except TypeError as exc:
        assert "designated" in str(exc)
    else:
        raise AssertionError("registry formatter silently accepted a designation")
    print(
        f"Minimal installed-wheel checks passed: xpint {xpint.__version__}, "
        f"Pint {pint.__version__}, Python {sys.version.split()[0]}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
