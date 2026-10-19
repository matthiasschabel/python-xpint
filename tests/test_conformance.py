"""Consume language-neutral semantic expectations through the public API."""

import json
import operator
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from xpint import SemanticQuantity, SemanticRegistry

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "semantic_conformance.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", _FIXTURE["parse_cases"], ids=lambda case: case["id"])
def test_designated_parse_conformance(case: dict[str, Any]) -> None:
    registry = SemanticRegistry()
    expected = registry.Unit("")
    for spec in case["factors"]:
        factor = registry.Unit(spec["unit"])
        if "designation" in spec:
            factor = factor.with_designation(spec["designation"])
        # Rational fixture exponents become floats, matching how Pint and xpint
        # parse "(1/3)"; an exact Fraction power would build a different unit.
        expected *= factor ** float(Fraction(spec["exponent"]))
    assert registry.Unit(case["input"]) == expected
    assert registry.Unit(str(expected)) == expected
    assert registry.Unit(format(expected, "~")) == expected


def _quantity(registry: SemanticRegistry, spec: dict[str, Any]) -> SemanticQuantity:
    unit = registry.Unit(spec["unit"])
    if "designation" in spec:
        unit = unit.with_designation(spec["designation"])
    return registry.Quantity(spec["magnitude"], unit, kind=spec.get("kind"))


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda case: case["id"])
def test_semantic_conformance(case: dict[str, Any]) -> None:
    assert _FIXTURE["schema_version"] == 1
    registry = SemanticRegistry()
    left = _quantity(registry, case["left"])
    right = _quantity(registry, case["right"])
    operation = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
    }[case["operation"]]
    if "expected_error" in case:
        assert case["expected_error"] == "semantic_mismatch"
        with pytest.raises(TypeError, match=r"quantity kinds|factor designations|absolute"):
            operation(left, right)
        return
    result = operation(left, right)
    expected = case["expected"]
    tolerance = _FIXTURE["numeric_tolerance"]
    reported = result.to(expected["unit"])
    assert reported.magnitude == pytest.approx(
        expected["magnitude"], rel=tolerance["relative"], abs=tolerance["absolute"]
    )
    assert result.kind == expected["kind"]
    assert result.affine_role == expected["role"]
    assert reported.kind == result.kind
    assert reported.affine_role == result.affine_role
    assert reported.designation_signature == result.designation_signature
    assert [
        [term.dimension.strip("[]"), term.designation, str(term.exponent)]
        for term in result.designation_signature
    ] == expected["designations"]
