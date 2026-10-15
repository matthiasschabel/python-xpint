"""Smoke tests for runnable user examples."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("filename", "expected_output"),
    [
        ("pint_style_basics.py", "converted: 7086.614173228347 inch / minute"),
        ("designated_solutions.py", "oxygen in sample: 0.01 millimole[O2]"),
        ("affine_temperature.py", "measured change: 5 delta_degree_Celsius (Delta)"),
        ("semantic_arrays.py", "average: 10.0 joule (Energy)"),
        ("angle_dimensions.py", "angled: cyclic input               127.732435383 megahertz"),
        ("custom_semantics.py", "mass_concentration[glucose]"),
    ],
)
def test_example_runs(
    filename: str, expected_output: str, capsys: pytest.CaptureFixture[str]
) -> None:
    example = Path(__file__).parents[1] / "examples" / filename

    runpy.run_path(str(example), run_name="__main__")

    assert expected_output in capsys.readouterr().out
