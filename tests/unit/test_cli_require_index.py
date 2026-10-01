"""Read commands must refuse an unindexed repo instead of creating an empty index.

`search`, `ask`, `query`, `call-graph` and `graph` all resolve
`IndexConfig.db_path_absolute` and open `Database` / `SQLiteVectorStore` on it. Both
create things: the property makes `.trelix/` and writes `.trelix/.gitignore`, and the
stores create `index.db` with its schema and an empty `vec0` table. So pointing any of
them at a directory that was never indexed left an empty index behind, reported "No
results" (or, for `graph`, "Knowledge Graph built" with zero nodes) and exited 0. The
stray file then defeated the `db_path.exists()` guards in `stats`, `link-tickets`,
`link-artifacts` and `migrate-vectors`.

Everything here runs through `CliRunner` against a tmp fixture. Nothing indexes a real
repository, and `Retriever` is patched wherever a command is expected to reach it.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from trelix.cli.main import app
from trelix.core.models import RetrievedContext

runner = CliRunner()

_READ_COMMANDS = [
    pytest.param(["search", "{repo}", "q", "--provider", "local"], id="search"),
    pytest.param(["ask", "{repo}", "q", "--provider", "local"], id="ask"),
    pytest.param(["ask", "{repo}", "q", "--agentic", "--provider", "local"], id="ask-agentic"),
    pytest.param(["query", "{repo}", "q", "--provider", "local"], id="query"),
    pytest.param(["call-graph", "{repo}", "sym", "--provider", "local"], id="call-graph"),
    pytest.param(["graph", "{repo}"], id="graph"),
    pytest.param(["graph", "{repo}", "--json"], id="graph-json"),
]


def _flat(text: str) -> str:
    """Rich hard-wraps at the console width; collapse whitespace before matching."""
    return re.sub(r"\s+", " ", text)


def _argv(template: list[str], repo: Path) -> list[str]:
    return [str(repo) if part == "{repo}" else part for part in template]


@pytest.fixture
def unindexed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return repo


@pytest.fixture
def indexed_repo(unindexed_repo: Path) -> Path:
    """A repo whose `.trelix/index.db` exists (schema only, no rows)."""
    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

    config = IndexConfig(repo_path=str(unindexed_repo.resolve()))
    Database(config.db_path_absolute).close()
    return unindexed_repo


@pytest.mark.parametrize("template", _READ_COMMANDS)
def test_read_command_on_unindexed_repo_exits_1_and_creates_nothing(
    unindexed_repo: Path, template: list[str]
) -> None:
    result = runner.invoke(app, _argv(template, unindexed_repo))

    assert result.exit_code == 1, result.output
    combined = _flat(result.output)
    assert "No index found at" in combined
    assert "index.db" in combined
    assert "Run trelix index" in combined
    assert not (unindexed_repo / ".trelix").exists(), (
        "a read command must not create .trelix/ (or index.db) in an unindexed repo"
    )


@pytest.mark.parametrize("template", _READ_COMMANDS)
def test_read_command_refuses_before_opening_any_store(
    unindexed_repo: Path, template: list[str]
) -> None:
    """The guard must run before Retriever, GraphBuilder or AgentLoop is constructed."""
    with (
        patch("trelix.retrieval.retriever.Retriever") as retriever,
        patch("trelix.graph.builder.GraphBuilder") as builder,
        patch("trelix.agent.AgentLoop") as agent_loop,
    ):
        result = runner.invoke(app, _argv(template, unindexed_repo))

    assert result.exit_code == 1, result.output
    retriever.assert_not_called()
    builder.assert_not_called()
    agent_loop.assert_not_called()


def test_stats_on_unindexed_repo_leaves_no_trelix_directory(unindexed_repo: Path) -> None:
    """`stats` already had an exists() guard, but resolved the path with the creating
    property first, so it still left `.trelix/` and `.trelix/.gitignore` behind."""
    result = runner.invoke(app, ["stats", str(unindexed_repo)])

    assert result.exit_code == 1, result.output
    assert "No index found" in _flat(result.output)
    assert not (unindexed_repo / ".trelix").exists()


def test_resolved_db_path_is_free_of_side_effects(unindexed_repo: Path) -> None:
    from trelix.core.config import IndexConfig

    config = IndexConfig(repo_path=str(unindexed_repo.resolve()))

    resolved = config.db_path_resolved

    assert resolved == unindexed_repo.resolve() / ".trelix" / "index.db"
    assert not (unindexed_repo / ".trelix").exists()
    # Writers keep the old behaviour: the creating property still makes the directory.
    assert config.db_path_absolute == resolved
    assert (unindexed_repo / ".trelix" / ".gitignore").read_text(encoding="utf-8") == "*\n"


def test_search_on_indexed_repo_still_reaches_the_retriever(indexed_repo: Path) -> None:
    context = RetrievedContext(
        query="q", results=[], context_text="", total_tokens=0, elapsed_seconds=0.0
    )
    fake = MagicMock()
    fake.return_value.retrieve.return_value = context

    with patch("trelix.retrieval.retriever.Retriever", fake):
        result = runner.invoke(app, ["search", str(indexed_repo), "q", "--provider", "local"])

    assert result.exit_code == 0, result.output
    fake.assert_called_once()


def test_call_graph_on_indexed_repo_still_reaches_the_retriever(indexed_repo: Path) -> None:
    fake = MagicMock()
    fake.return_value.get_callers.return_value = []
    fake.return_value.get_callees.return_value = []
    fake.return_value.get_importers.return_value = []

    with patch("trelix.retrieval.retriever.Retriever", fake):
        result = runner.invoke(app, ["call-graph", str(indexed_repo), "sym", "--provider", "local"])

    assert result.exit_code == 0, result.output
    fake.assert_called_once()


def test_graph_on_indexed_repo_still_builds(indexed_repo: Path) -> None:
    result = runner.invoke(app, ["graph", str(indexed_repo)])

    assert result.exit_code == 0, result.output
    assert "Knowledge Graph built" in result.output
