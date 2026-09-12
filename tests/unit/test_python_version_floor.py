"""Python version-floor regression guard for v3.3.0's Python 3.11 -> 3.12 bump.

WHY THIS EXISTS. Python 3.11 has been security-fixes-only since April 2024 (EOL October
2027); 3.12 is the most-adopted 3.x version in the wild and SPEC 0's 3-year convention marks
it as its own drop-candidate starting October 2026 (see
docs/reports/v4-0-0-upgrade-research-2026-09-11.md). trelix's CI already matrix-tests
3.11/3.12/3.13/3.14 identically, and the Dockerfile is already on python:3.14-slim, so this
floor bump carries near-zero risk -- but four separate `pyproject.toml` files declare the
same floor independently (root + three sibling packages), and it's exactly the kind of
four-way-duplicated constant that drifts silently if only one copy gets bumped.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

_PYPROJECT_PATHS = (
    _ROOT / "pyproject.toml",
    _ROOT / "packages" / "trelix-mcp" / "pyproject.toml",
    _ROOT / "packages" / "trelix-langchain" / "pyproject.toml",
    _ROOT / "packages" / "trelix-llama-index" / "pyproject.toml",
)


def test_requires_python_floor_is_3_12_everywhere() -> None:
    """Every distribution's requires-python must read >=3.12, not just the root package."""
    stale = []
    for path in _PYPROJECT_PATHS:
        with path.open("rb") as fh:
            requires_python = tomllib.load(fh)["project"]["requires-python"]
        if requires_python != ">=3.12":
            stale.append((path.relative_to(_ROOT), requires_python))
    assert not stale, (
        "these pyproject.toml files declare a requires-python floor other than '>=3.12': "
        f"{stale} -- v3.3.0 bumps the floor everywhere, not just the root package"
    )


def test_root_mypy_and_ruff_target_python_3_12() -> None:
    """mypy/ruff's own target-version configs must track the same floor, or type/lint
    checking silently runs against the old language level even though packaging says 3.12."""
    with (_ROOT / "pyproject.toml").open("rb") as fh:
        data = tomllib.load(fh)
    assert data["tool"]["mypy"]["python_version"] == "3.12", (
        "[tool.mypy] python_version has not been bumped to match requires-python >=3.12"
    )
    assert data["tool"]["ruff"]["target-version"] == "py312", (
        "[tool.ruff] target-version has not been bumped to match requires-python >=3.12"
    )


def test_no_python_3_11_classifier() -> None:
    """A leftover 3.11 classifier misrepresents what the package supports post-bump."""
    with (_ROOT / "pyproject.toml").open("rb") as fh:
        classifiers = tomllib.load(fh)["project"]["classifiers"]
    assert "Programming Language :: Python :: 3.11" not in classifiers, (
        "the 3.11 classifier is still present after the floor moved to >=3.12 -- PyPI would "
        "advertise support for a version requires-python no longer allows installing on"
    )
    assert "Programming Language :: Python :: 3.12" in classifiers, (
        "the 3.12 classifier is missing even though it's now the floor"
    )
