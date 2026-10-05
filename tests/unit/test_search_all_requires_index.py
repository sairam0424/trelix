"""Federated search skips a registered repo that has no index, and says so.

`FederatedRetriever` built a Retriever for every registered repo, and building one on a repo
that was never indexed creates its index (for a registered path that does not exist, even
the repo directory). `trelix search-all` then printed "No results found." for a repo it had
never searched, and the empty index it left behind made that repo look indexed. Now an
unindexed repo is skipped before anything is opened, `search-all` names it on stderr, and
only when nothing it would query has an index does it fail (exit 1, like every other read
command).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tests.fixtures.indexed import mark_indexed
from trelix.cli.main import app
from trelix.core.models import Chunk, IndexedFile, Language, SearchResult, Symbol
from trelix.federation.registry import RepoRegistry
from trelix.federation.retriever import FederatedRetriever

runner = CliRunner()


def _result(symbol_name: str = "handler") -> SearchResult:
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
            path="/repo/src/app.py",
            rel_path="src/app.py",
            language=Language.PYTHON,
            hash="deadbeef",
            size_bytes=4,
        ),
        score=0.5,
        rank=1,
        source="vector",
    )


def _unindexed(tmp_path: Path, name: str) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return repo


def _registry(tmp_path: Path, *repos: tuple[str, Path]) -> RepoRegistry:
    registry = RepoRegistry.load(str(tmp_path / "repos.json"))
    for alias, path in repos:
        registry.add(alias, str(path))
    registry.save()
    return registry


def _one_result_context() -> MagicMock:
    context = MagicMock()
    context.results = [_result()]
    return context


def _squash(text: str) -> str:
    """Rich folds a long path mid-word; compare with all whitespace removed."""
    return "".join(text.split())


# ---------------------------------------------------------------------------
# FederatedRetriever
# ---------------------------------------------------------------------------


def test_an_unindexed_repo_is_skipped_and_never_opened(tmp_path: Path) -> None:
    indexed = mark_indexed(_unindexed(tmp_path, "indexed"))
    unindexed = _unindexed(tmp_path, "unindexed")
    registry = _registry(tmp_path, ("a", indexed), ("b", unindexed))

    with patch("trelix.federation.retriever.Retriever") as retriever:
        retriever.return_value.retrieve.return_value = _one_result_context()
        results = FederatedRetriever(registry, max_workers=1, cache_ttl=0).retrieve("q", k=5)

    assert len(results) == 1
    assert retriever.call_count == 1
    assert retriever.call_args.args[0].repo_path == str(indexed)
    assert not (unindexed / ".trelix").exists()


def test_unindexed_repos_names_each_repo_with_the_cli_message(tmp_path: Path) -> None:
    indexed = mark_indexed(_unindexed(tmp_path, "indexed"))
    unindexed = _unindexed(tmp_path, "unindexed")
    registry = _registry(tmp_path, ("a", indexed), ("b", unindexed))

    missing = FederatedRetriever(registry).unindexed_repos()

    assert [entry.alias for entry, _ in missing] == ["b"]
    assert str(missing[0][1]) == (
        f"No index found at {unindexed}/.trelix/index.db. Run trelix index {unindexed} first."
    )
    assert not (unindexed / ".trelix").exists()


def test_when_nothing_is_indexed_retrieve_returns_nothing_and_opens_nothing(
    tmp_path: Path,
) -> None:
    first = _unindexed(tmp_path, "first")
    second = _unindexed(tmp_path, "second")
    registry = _registry(tmp_path, ("a", first), ("b", second))

    with patch("trelix.federation.retriever.Retriever") as retriever:
        results = FederatedRetriever(registry, cache_ttl=0).retrieve("q")

    assert results == []
    retriever.assert_not_called()
    assert not (first / ".trelix").exists()
    assert not (second / ".trelix").exists()


def test_a_registered_path_that_does_not_exist_is_not_created(tmp_path: Path) -> None:
    """The old fan-out built a Retriever even for this, which mkdir'd the whole path."""
    ghost = tmp_path / "ghost" / "repo"
    registry = _registry(tmp_path, ("ghost", ghost))

    with patch("trelix.federation.retriever.Retriever") as retriever:
        results = FederatedRetriever(registry, cache_ttl=0).retrieve("q")

    assert results == []
    retriever.assert_not_called()
    assert not (tmp_path / "ghost").exists()


def test_unindexed_repos_reports_a_registered_path_that_does_not_exist(tmp_path: Path) -> None:
    """`IndexConfig(repo_path=...)` rejects a missing directory, so the check must build its
    config without validation. `retrieve` swallows every error, but `search-all` and the MCP
    `federation_search_all` call `unindexed_repos` outside that, so a registry entry whose
    repo was moved or deleted would crash them."""
    ghost = tmp_path / "gone" / "repo"
    registry = _registry(tmp_path, ("ghost", ghost))

    missing = FederatedRetriever(registry).unindexed_repos()

    assert [entry.alias for entry, _ in missing] == ["ghost"]
    assert str(missing[0][1]) == (
        f"No index found at {ghost}/.trelix/index.db. Run trelix index {ghost} first."
    )
    assert not (tmp_path / "gone").exists()


def test_only_the_repos_inside_the_cap_are_checked(tmp_path: Path) -> None:
    """With max_repos=1 only the first entry is queried, so an unindexed second one is not
    reported (and not opened); without a cap it is."""
    first = mark_indexed(_unindexed(tmp_path, "first"))
    second = _unindexed(tmp_path, "second")
    registry = _registry(tmp_path, ("a", first), ("b", second))

    capped = FederatedRetriever(registry, max_repos=1)
    uncapped = FederatedRetriever(registry)

    assert capped.unindexed_repos() == []
    assert [entry.alias for entry, _ in uncapped.unindexed_repos()] == ["b"]


# ---------------------------------------------------------------------------
# trelix search-all
# ---------------------------------------------------------------------------


def test_search_all_fails_when_every_queried_repo_is_unindexed(tmp_path: Path) -> None:
    first = _unindexed(tmp_path, "first")
    second = _unindexed(tmp_path, "second")
    _registry(tmp_path, ("a", first), ("b", second))

    with patch("trelix.federation.retriever.Retriever") as retriever:
        result = runner.invoke(app, ["search-all", "q", "--config", str(tmp_path / "repos.json")])

    assert result.exit_code == 1, result.output
    squashed = _squash(result.output)
    assert _squash(f"Skipping a: No index found at {first}/.trelix/index.db.") in squashed
    assert _squash(f"Skipping b: No index found at {second}/.trelix/index.db.") in squashed
    assert _squash("None of the queried repos has an index.") in squashed
    retriever.assert_not_called()
    assert not (first / ".trelix").exists()
    assert not (second / ".trelix").exists()


def test_search_all_names_a_registered_path_that_does_not_exist_and_creates_nothing(
    tmp_path: Path,
) -> None:
    ghost = tmp_path / "gone" / "repo"
    _registry(tmp_path, ("ghost", ghost))

    with patch("trelix.federation.retriever.Retriever") as retriever:
        result = runner.invoke(app, ["search-all", "q", "--config", str(tmp_path / "repos.json")])

    assert result.exit_code == 1, result.output
    squashed = _squash(result.output)
    assert _squash(f"Skipping ghost: No index found at {ghost}/.trelix/index.db.") in squashed
    assert _squash("None of the queried repos has an index.") in squashed
    retriever.assert_not_called()
    assert not (tmp_path / "gone").exists()


def test_search_all_skips_the_unindexed_repo_and_searches_the_rest(tmp_path: Path) -> None:
    indexed = mark_indexed(_unindexed(tmp_path, "indexed"))
    unindexed = _unindexed(tmp_path, "unindexed")
    _registry(tmp_path, ("a", indexed), ("b", unindexed))

    with patch("trelix.federation.retriever.Retriever") as retriever:
        retriever.return_value.retrieve.return_value = _one_result_context()
        result = runner.invoke(app, ["search-all", "q", "--config", str(tmp_path / "repos.json")])

    assert result.exit_code == 0, result.output
    assert _squash(f"Skipping b: No index found at {unindexed}/.trelix/index.db.") in _squash(
        result.output
    )
    assert "handler" in result.stdout
    assert retriever.call_count == 1
    assert not (unindexed / ".trelix").exists()


def test_search_all_status_counts_only_the_repos_it_searches(tmp_path: Path) -> None:
    indexed = mark_indexed(_unindexed(tmp_path, "indexed"))
    unindexed = _unindexed(tmp_path, "unindexed")
    _registry(tmp_path, ("a", indexed), ("b", unindexed))

    with (
        patch("trelix.federation.retriever.Retriever") as retriever,
        patch("trelix.cli.main._status_console") as status_console,
    ):
        retriever.return_value.retrieve.return_value = _one_result_context()
        result = runner.invoke(app, ["search-all", "q", "--config", str(tmp_path / "repos.json")])

    assert result.exit_code == 0, result.output
    status_console.return_value.status.assert_called_once_with("Searching 1 repos...")


def test_search_all_json_keeps_stdout_clean_and_reports_on_stderr(tmp_path: Path) -> None:
    indexed = mark_indexed(_unindexed(tmp_path, "indexed"))
    unindexed = _unindexed(tmp_path, "unindexed")
    _registry(tmp_path, ("a", indexed), ("b", unindexed))

    with patch("trelix.federation.retriever.Retriever") as retriever:
        retriever.return_value.retrieve.return_value = _one_result_context()
        result = runner.invoke(
            app, ["search-all", "q", "--json", "--config", str(tmp_path / "repos.json")]
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert [row["symbol"] for row in payload] == ["handler"]
    assert "Skippingb:" in _squash(result.stderr)
    assert "Skipping" not in result.stdout


def test_search_all_json_on_an_unindexed_registry_fails_with_empty_stdout(
    tmp_path: Path,
) -> None:
    repo = _unindexed(tmp_path, "only")
    _registry(tmp_path, ("a", repo))

    result = runner.invoke(
        app, ["search-all", "q", "--json", "--config", str(tmp_path / "repos.json")]
    )

    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert not (repo / ".trelix").exists()


def test_search_all_fails_when_the_only_queried_repo_is_unindexed_even_if_a_later_one_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repos beyond TRELIX_FEDERATION_MAX_REPOS are not queried, so they cannot save the run."""
    monkeypatch.setenv("TRELIX_FEDERATION_MAX_REPOS", "1")
    first = _unindexed(tmp_path, "first")
    second = mark_indexed(_unindexed(tmp_path, "second"))
    _registry(tmp_path, ("a", first), ("b", second))

    with patch("trelix.federation.retriever.Retriever") as retriever:
        result = runner.invoke(app, ["search-all", "q", "--config", str(tmp_path / "repos.json")])

    assert result.exit_code == 1, result.output
    retriever.assert_not_called()
