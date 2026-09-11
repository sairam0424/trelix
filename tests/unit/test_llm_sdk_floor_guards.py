"""LLM SDK floor-regression guards.

WHY THIS EXISTS. openai>=3.0.0 and anthropic>=1.0.0 are not just "latest and greatest"
floors -- they're the exact versions this codebase's LLM backends were fixed to require:

  * anthropic>=1.0.0: AnthropicBackend no longer sends `temperature=` to
    Messages.create (removed in anthropic-sdk-python v1.0.0) and Bedrock's region check
    matches AnthropicBedrock's v1.0.0 enforcement. Lowering this floor would silently
    resurrect a fixed TypeError-on-first-call bug the moment pip resolves back below 1.0.0.
  * openai>=3.0.0: src/trelix/core/retry.py recognizes httpx2 exception shapes
    specifically because openai-python v3.0.0 made httpx2 the default transport.
    Lowering this floor doesn't break anything by itself, but re-couples the floor to a
    fact (the retry classifier's httpx2 support) that was added FOR this version.

These tests pin the current, deliberate floors so a future contributor loosening one
during an unrelated dependency bump gets a named, specific failure instead of silently
reopening a fixed bug.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load_pyproject() -> dict:
    with (_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _core_dependency_specifier(name: str) -> str:
    deps = _load_pyproject()["project"]["dependencies"]
    for dep in deps:
        if re.match(rf"^{re.escape(name)}\s*[><=!~]", dep):
            return dep
    raise AssertionError(f"{name!r} not found in [project] dependencies")


def _extra_dependency_specifier(extra: str, name: str) -> str:
    extras = _load_pyproject()["project"]["optional-dependencies"]
    assert extra in extras, f"optional-dependencies has no {extra!r} extra"
    for dep in extras[extra]:
        if re.match(rf"^{re.escape(name)}\s*[><=!~]", dep):
            return dep
    raise AssertionError(f"{name!r} not found in the {extra!r} extra")


def test_openai_floor_is_at_or_above_3_0_0() -> None:
    """Below 3.0.0, retry.py's httpx2 recognition is guarding against a
    transport switch that hasn't happened yet for the resolved version --
    harmless, but the floor should track the fact it was raised for."""
    spec = _core_dependency_specifier("openai")
    match = re.search(r">=\s*(\d+)\.(\d+)\.(\d+)", spec)
    assert match is not None, f"openai specifier {spec!r} has no >=X.Y.Z floor to check"
    floor = tuple(int(x) for x in match.groups())
    assert floor >= (3, 0, 0), (
        f"openai floor is {spec!r} -- below 3.0.0 this no longer needs retry.py's httpx2 "
        "recognition, but re-lowering it defeats the reason the floor was raised"
    )


def test_anthropic_floor_is_at_or_above_1_0_0() -> None:
    """Below 1.0.0, a fresh install can resolve an anthropic-sdk-python version
    that still accepts temperature=, silently un-fixing AnthropicBackend's
    complete()/stream() -- but also one where AnthropicBackend's dropped-
    temperature behavior and Bedrock's region requirement don't line up with
    what the installed SDK actually enforces."""
    spec = _extra_dependency_specifier("anthropic", "anthropic")
    match = re.search(r">=\s*(\d+)\.(\d+)\.(\d+)", spec)
    assert match is not None, f"anthropic specifier {spec!r} has no >=X.Y.Z floor to check"
    floor = tuple(int(x) for x in match.groups())
    assert floor >= (1, 0, 0), (
        f"anthropic floor is {spec!r} -- anthropic-sdk-python v1.0.0 removed "
        "temperature/top_p/top_k from Messages.create; AnthropicBackend.complete()/"
        "stream() were fixed to never send temperature, so a floor below 1.0.0 no "
        "longer matches what the code assumes the installed SDK enforces"
    )
