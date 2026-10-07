"""Dependency-floor regression guards for CVE and premature-breaking-major exposure.

WHY THIS EXISTS. trelix's core dependency floors (`pyproject.toml`'s `[project] dependencies`
and `[project.optional-dependencies]`) are, by default, open-ended (`>=X`, no ceiling) — a
plain `pip install trelix` can silently resolve to whatever the latest release of a dependency
happens to be on install day, with no signal that anything changed.

Four concrete cases motivate the guards below:

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
  * `sqlite-vec` is the second guard of the fastmcp kind: a floor the code needs and a ceiling
    on untested releases, not a CVE or a breaking major: it is pinned `>=0.1.9,<0.1.10`; 0.1.7
    made `DELETE` reclaim space, which trelix's `DELETE`+`INSERT` upsert and `--prune` rely on;
    0.1.9 is the release the vec0 contract tests were verified against; the ceiling keeps the
    untested 0.1.10 pre-releases (ivf/diskann) out, and PEP 440 places every `0.1.10aN` under
    `<0.1.10`.

These tests pin the current, deliberate floors/ceilings so a future contributor loosening one
(e.g. widening a version range during an unrelated dependency bump) gets a named, specific
failure instead of silent re-exposure. The last guard reads `packages/trelix-mcp/pyproject.toml`
instead: a floor that the code needs rather than one that avoids a CVE or a breaking major.
The sqlite-vec pair is the only pair in this file that also checks the INSTALLED package,
deliberately: the requirement string says what pip may resolve, the installed check says what
this venv runs, and a venv on another release must fail with the reason rather than skip.
"""

from __future__ import annotations

import re
import sqlite3
import tomllib
from pathlib import Path

import sqlite_vec

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


def test_sqlite_vec_is_pinned_to_the_verified_0_1_9_line() -> None:
    """The floor is 0.1.7's DELETE space reclamation, placed at 0.1.9 (the release the vec0
    contract tests were verified against); the ceiling keeps the untested 0.1.10 pre-releases
    out. `_core_dependency_specifier` returns the full requirement string (critique B1).
    Mutations: `>=0.1.6` restored; ceiling dropped (`>=0.1.9`); ceiling loosened (`<0.1.11`).
    """
    spec = _core_dependency_specifier("sqlite-vec")
    assert spec == "sqlite-vec>=0.1.9,<0.1.10", (
        f"sqlite-vec specifier is {spec!r} — keep exactly >=0.1.9,<0.1.10: 0.1.7 made DELETE "
        "reclaim space (trelix upserts by DELETE+INSERT and --prune deletes), 0.1.9 is the "
        "release the vec0 contract tests were verified against, and <0.1.10 keeps the untested "
        "0.1.10 pre-releases (ivf/diskann) out under PEP 440; raise the ceiling only after "
        "re-running tests/unit/test_vector_store_contract.py and test_store.py on the new release"
    )


def test_installed_sqlite_vec_is_the_pinned_release() -> None:
    """Environment-coupled on purpose and without a skip: this is the guard, so a venv on
    another release fails with the reason. The package version and the loaded extension's
    `vec_version()` must both say 0.1.9. Mutations: `"0.1.9"` -> `"0.1.8"` (package
    assertion); `"v0.1.9"` -> `"v0.1.8"` (extension assertion).
    """
    assert sqlite_vec.__version__ == "0.1.9", (
        f"installed sqlite-vec is {sqlite_vec.__version__}; pyproject pins >=0.1.9,<0.1.10 and "
        "the vec0 contract tests were verified on 0.1.9 — reinstall (pip install -e .) before "
        "trusting this suite"
    )
    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        loaded = conn.execute("select vec_version()").fetchone()[0]
    finally:
        conn.close()
    assert loaded == "v0.1.9", (
        f"the loaded sqlite-vec extension reports {loaded!r} while the package says "
        f"{sqlite_vec.__version__}; the two must agree at v0.1.9"
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
