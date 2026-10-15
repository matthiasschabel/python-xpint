"""Public package metadata and typing-marker tests."""

import os
import tomllib
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import cast

import xpint


def test_package_version_matches_distribution_metadata() -> None:
    project_data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = cast(dict[str, object], project_data["project"])
    expected_version = project["version"]

    assert isinstance(expected_version, str)
    assert xpint.__version__ == expected_version
    assert version("xpint") == expected_version


def test_package_publishes_typing_marker() -> None:
    assert files("xpint").joinpath("py.typed").is_file()


def test_installed_wheel_provenance_when_required() -> None:
    if os.environ.get("XPINT_REQUIRE_WHEEL") == "1":
        assert "site-packages" in Path(xpint.__file__).parts, xpint.__file__
