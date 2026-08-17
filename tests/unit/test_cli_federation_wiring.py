"""Regression tests: `trelix search-all` must honour TRELIX_FEDERATION_MAX_REPOS.

THE BUG. `RetrievalConfig.federation_max_repos` (`TRELIX_FEDERATION_MAX_REPOS`,
default 50) is declared in `core/config.py`, documented in `docs/CONFIGURATION.md`
as the cap that "prevents an unbounded `federation_add_repo` loop from making every
subsequent query scale linearly", and enforced by `packages/trelix-mcp`. The CLI
built `FederatedRetriever(registry)` with no arguments, so `max_repos` stayed at the
constructor default of `None` — unbounded — and `search-all` fanned out to every
registered repo no matter what the cap said. A cap that stops a runaway
`federation_add_repo` loop only on the MCP path is not a cap: the same
`~/.config/trelix/repos.json` the MCP tools write is what the CLI reads.

WHY "IT STILL RETURNS RESULTS" IS NOT THE ASSERTION. Both the capped and the
uncapped path exit 0 and print a table, so the only observable difference is the
constructor argument and how many repos were actually queried. These tests assert
the argument reaches the retriever, and that the truncation is *announced* rather
than applied silently — 47 of 50 repos being skipped with no output is the same
class of quiet failure as the cap not existing.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from trelix.cli.main import app
from trelix.core.models import Chunk, IndexedFile, Language, SearchResult, Symbol

runner = CliRunner()


def _result(rel_path: str = "src/app.py", symbol_name: str = "handler") -> SearchResult:
    """One SearchResult, enough to reach the rendering branches of search-all."""
    return SearchResult(
        chunk=Chunk(symbol_id=1, chunk_text="code", token_count=4, id=1),
        symbol=Symbol(
            file_id=1,
            name=symbol_name,
            qualified_name=symbol_name,
            kind="function",  # type: ignore[arg-type]
            line_start=1,
            line_end=2,
            signature=f"def {symbol_name}()",
            body="code",
            id=1,
        ),
        file=IndexedFile(
            path=f"/repo/{rel_path}",
            rel_path=rel_path,
            language=Language.PYTHON,
            hash="deadbeef",
            size_bytes=4,
        ),
        score=0.5,
        rank=1,
        source="backend:vector",
    )


def _patched_federation(registered: int, results: list[SearchResult]):
    """Patch RepoRegistry.load + FederatedRetriever so no repo is touched.

    Returns the patched class (to inspect its constructor call) and the two context
    managers to enter. `registered` fake entries are enough: the CLI only reads
    `len(registry.list())`, never an entry's fields, before the fan-out.
    """
    registry = MagicMock()
    registry.list.return_value = [MagicMock() for _ in range(registered)]

    fed_cls = MagicMock()
    fed_cls.return_value.retrieve.return_value = results

    return (
        fed_cls,
        patch("trelix.federation.registry.RepoRegistry.load", return_value=registry),
        patch("trelix.federation.retriever.FederatedRetriever", fed_cls),
    )


def test_search_all_passes_configured_max_repos_to_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declared cap must reach the constructor, not sit unread in the config."""
    monkeypatch.setenv("TRELIX_FEDERATION_MAX_REPOS", "3")

    fed_cls, patch_registry, patch_fed = _patched_federation(5, [_result()])
    with patch_registry, patch_fed:
        result = runner.invoke(app, ["search-all", "q"])

    assert result.exit_code == 0, result.output
    assert fed_cls.call_count == 1
    assert fed_cls.call_args.kwargs.get("max_repos") == 3, (
        "search-all built FederatedRetriever without max_repos, so "
        "TRELIX_FEDERATION_MAX_REPOS had no effect on the CLI path"
    )


def test_search_all_uses_default_cap_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env var still means capped at the documented default of 50, not None."""
    monkeypatch.delenv("TRELIX_FEDERATION_MAX_REPOS", raising=False)

    fed_cls, patch_registry, patch_fed = _patched_federation(2, [_result()])
    with patch_registry, patch_fed:
        result = runner.invoke(app, ["search-all", "q"])

    assert result.exit_code == 0, result.output
    assert fed_cls.call_args.kwargs.get("max_repos") == 50


def test_search_all_announces_repos_skipped_by_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncating the fan-out silently would hide an incomplete search."""
    monkeypatch.setenv("TRELIX_FEDERATION_MAX_REPOS", "2")

    fed_cls, patch_registry, patch_fed = _patched_federation(5, [_result()])
    with patch_registry, patch_fed:
        result = runner.invoke(app, ["search-all", "q"])

    assert result.exit_code == 0, result.output
    assert "TRELIX_FEDERATION_MAX_REPOS" in result.output, (
        "5 registered repos, cap 2 — 3 were skipped and nothing said so"
    )
    assert "2 of 5" in result.output


def test_search_all_says_nothing_extra_when_no_repo_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The notice is a warning about an incomplete search, not a banner."""
    monkeypatch.setenv("TRELIX_FEDERATION_MAX_REPOS", "50")

    fed_cls, patch_registry, patch_fed = _patched_federation(3, [_result()])
    with patch_registry, patch_fed:
        result = runner.invoke(app, ["search-all", "q"])

    assert result.exit_code == 0, result.output
    assert "TRELIX_FEDERATION_MAX_REPOS" not in result.output


def test_search_all_json_stdout_stays_parseable_when_the_cap_skips_repos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The skip notice is prose — on stdout it would break `search-all --json | jq`.

    Same contract the empty-registry branch above it already protects, and the same
    reason `_status_console()` exists.
    """
    monkeypatch.setenv("TRELIX_FEDERATION_MAX_REPOS", "1")

    try:
        runner_split = CliRunner(mix_stderr=False)
    except TypeError:  # click >= 8.2 dropped the kwarg and always splits the streams
        runner_split = CliRunner()

    fed_cls, patch_registry, patch_fed = _patched_federation(4, [_result()])
    with patch_registry, patch_fed:
        result = runner_split.invoke(app, ["search-all", "q", "--json"])

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload[0]["file"] == "src/app.py"
    assert "TRELIX_FEDERATION_MAX_REPOS" in result.stderr
