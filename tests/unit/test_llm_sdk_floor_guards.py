"""LLM SDK floor-regression guards.

WHY THIS EXISTS. anthropic>=1.0.0 is not just a "latest and greatest" floor -- it's the
exact version this codebase's LLM backend was fixed to require: AnthropicBackend no
longer sends `temperature=` to Messages.create (removed in anthropic-sdk-python v1.0.0)
and Bedrock's region check matches AnthropicBedrock's v1.0.0 enforcement. Lowering this
floor would silently resurrect a fixed TypeError-on-first-call bug the moment pip
resolves back below 1.0.0.

openai's floor tells a DIFFERENT story: it is >=2.20.0 with an explicit <3.0.0 ceiling
because litellm 1.104.0 (the latest when checked, 2026-10-05) requires
openai>=2.20.0,<3.0.0, and so does every litellm release since 1.84.0 -- an openai>=3.0.0
floor would make trelix[litellm] unresolvable (see pyproject.toml's own comment on the
openai dependency, and tests/unit/test_dependency_floor_guards.py's ceiling guard). The
retry layer is not the reason: src/trelix/core/retry.py has recognised the "httpx2"
transport since 3.3.0.

These tests pin the current, deliberate floors so a future contributor loosening one
during an unrelated dependency bump gets a named, specific failure instead of silently
reopening a fixed bug (anthropic) or making trelix[litellm] unresolvable (openai,
guarded separately in test_dependency_floor_guards.py).
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


def test_openai_floor_has_deliberate_sub_3_0_0_ceiling() -> None:
    """litellm (checked at 1.104.0) requires openai<3.0.0 -- this guard checks the core
    openai specifier keeps its <3.0.0 ceiling (see
    test_dependency_floor_guards.py::test_openai_ceiling_matches_litellm_requirement)
    rather than that the floor itself stayed at any particular value."""
    spec = _core_dependency_specifier("openai")
    assert "<3.0.0" in spec, (
        f"openai specifier is {spec!r} -- litellm 1.104.0 (and every release since 1.84.0) "
        "requires openai<3.0.0, so the core floor must keep an explicit <3.0.0 ceiling "
        "(not just a >=X floor) or trelix[litellm] becomes unresolvable"
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
