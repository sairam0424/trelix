"""Dependency-floor regression guards for CVE and premature-breaking-major exposure.

WHY THIS EXISTS. trelix's core dependency floors (`pyproject.toml`'s `[project] dependencies`
and `[project.optional-dependencies]`) are, by default, open-ended (`>=X`, no ceiling) — a
plain `pip install trelix` can silently resolve to whatever the latest release of a dependency
happens to be on install day, with no signal that anything changed.

Three concrete cases motivate the guards below:

  * CVE-2026-58203 (GHSA-4xgf-cpjx-pc3j): a symlink-traversal bug in pydantic-settings'
    `NestedSecretsSettingsSource` (versions 2.12.0-2.14.1, fixed in 2.14.2). trelix's floor
    was `pydantic-settings>=2.3.0` with no ceiling, so the vulnerable range sat inside what a
    fresh install could already resolve to.
  * `anthropic-sdk-python` v1.0.0 (2026-08-20) is an intentional breaking release: it drops
    `temperature`/`top_p`/`top_k` from every Messages method (which `AnthropicBackend` in
    `src/trelix/llm/providers/anthropic_backend.py` unconditionally passes today). The floor
    was unbounded (`anthropic>=0.40.0`), so that breaking major could resolve silently before
    the LLM abstraction layer is updated to handle it — see
    `docs/reports/v4-0-0-upgrade-research-2026-09-11.md`.
  * `openai>=3.0.0` (openai-python v3.0.0, 2026-08-12) cannot be installed together with
    `trelix[litellm]`: litellm 1.104.0 (the latest when checked, 2026-10-05) requires
    `openai>=2.20.0,<3.0.0`, and so does every litellm release since 1.84.0. The `openai`
    ceiling is that one requirement; the retry layer is not the reason, because
    `src/trelix/core/retry.py` has recognised the "httpx2" transport since 3.3.0.

These tests pin the current, deliberate floors/ceilings so a future contributor loosening one
(e.g. widening a version range during an unrelated dependency bump) gets a named, specific
failure instead of silent re-exposure. The last guard reads `packages/trelix-mcp/pyproject.toml`
instead: a floor that the code needs rather than one that avoids a CVE or a breaking major.
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
    """Return the exact specifier string for *name* in [project] dependencies."""
    deps = _load_pyproject()["project"]["dependencies"]
    for dep in deps:
        if re.match(rf"^{re.escape(name)}\s*[><=!~]", dep):
            return dep
    raise AssertionError(f"{name!r} not found in [project] dependencies")


def _extra_dependency_specifier(extra: str, name: str) -> str:
    """Return the exact specifier string for *name* inside
    [project.optional-dependencies][extra]."""
    extras = _load_pyproject()["project"]["optional-dependencies"]
    assert extra in extras, f"optional-dependencies has no {extra!r} extra"
    for dep in extras[extra]:
        if re.match(rf"^{re.escape(name)}\s*[><=!~]", dep):
            return dep
    raise AssertionError(f"{name!r} not found in the {extra!r} extra")


def test_pydantic_settings_floor_excludes_symlink_traversal_cve() -> None:
    """A floor below 2.14.2 re-opens CVE-2026-58203's symlink-traversal window."""
    spec = _core_dependency_specifier("pydantic-settings")
    match = re.search(r">=\s*(\d+)\.(\d+)\.(\d+)", spec)
    assert match is not None, f"pydantic-settings specifier {spec!r} has no >=X.Y.Z floor to check"
    floor = tuple(int(x) for x in match.groups())
    assert floor >= (2, 14, 2), (
        f"pydantic-settings floor is {spec!r} — versions 2.12.0-2.14.1 carry CVE-2026-58203 "
        "(GHSA-4xgf-cpjx-pc3j, symlink traversal in NestedSecretsSettingsSource); the floor "
        "must stay >=2.14.2"
    )


def test_openai_ceiling_matches_litellm_requirement() -> None:
    """litellm (checked at 1.104.0) requires openai<3.0.0, so trelix[litellm] needs the ceiling."""
    spec = _core_dependency_specifier("openai")
    assert "<3.0.0" in spec or re.search(r">=\s*3\.", spec), (
        f"openai specifier is {spec!r} — with no upper bound, a fresh install can resolve "
        "openai>=3.0.0, which trelix[litellm] cannot use: litellm 1.104.0 (the latest when "
        "checked, 2026-10-05) and every release since 1.84.0 require openai>=2.20.0,<3.0.0. "
        "Keep a <3.0.0 ceiling until a litellm release accepts openai 3 (re-check with "
        "`uv pip compile` on litellm>=1.90.2 plus openai>=3.0.0), then raise the floor to "
        ">=3.0.0"
    )


def test_anthropic_ceiling_excludes_removed_temperature_kwarg() -> None:
    """anthropic-sdk-python v1.0.0 removes temperature/top_p/top_k; AnthropicBackend
    passes temperature today."""
    spec = _extra_dependency_specifier("anthropic", "anthropic")
    assert "<1.0.0" in spec or re.search(r">=\s*1\.", spec), (
        f"anthropic specifier is {spec!r} — with no upper bound, a fresh install can resolve "
        "anthropic-sdk-python>=1.0.0, which removed temperature/top_p/top_k from every "
        "Messages method; src/trelix/llm/providers/anthropic_backend.py still passes "
        "temperature= unconditionally in complete()/stream() and would raise TypeError; pin "
        "a <1.0.0 ceiling until that's fixed, or bump to >=1.0.0 once it is"
    )


def test_trelix_mcp_fastmcp_floor_is_the_release_the_tool_metadata_was_run_against() -> None:
    """trelix-mcp passes cache_ttl, cache_scope and transforms to FastMCP(...) and calls
    server.disable(names=, components=); 4.0.10 is the release those were run against."""
    with (_ROOT / "packages" / "trelix-mcp" / "pyproject.toml").open("rb") as fh:
        deps = tomllib.load(fh)["project"]["dependencies"]
    spec = next((dep for dep in deps if re.match(r"^fastmcp\s*[><=!~]", dep)), None)
    assert spec is not None, "fastmcp not found in packages/trelix-mcp [project] dependencies"
    match = re.search(r">=\s*(\d+)\.(\d+)\.(\d+)", spec)
    assert match is not None, f"fastmcp specifier {spec!r} has no >=X.Y.Z floor to check"
    floor = tuple(int(x) for x in match.groups())
    assert floor >= (4, 0, 10), (
        f"trelix-mcp declares {spec!r}, but server.py and tool_metadata.py use FastMCP's "
        "cache_ttl, cache_scope and transforms arguments and server.disable(names=, "
        "components=), which were run only against fastmcp 4.0.10; 4.0.0 to 4.0.9 were not "
        "tested. Lower the floor only after running packages/trelix-mcp/tests on the older release"
    )
