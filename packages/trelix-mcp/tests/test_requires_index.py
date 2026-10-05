"""The MCP read tools refuse a repository that has no index, and create nothing.

Every read tool built a Retriever, GraphBuilder, AgentLoop or Database on `repo_path`, and
opening a missing index creates it. The tool then answered with an empty result and left a
repository that was never indexed looking indexed (a later `trelix search` said "no
results" instead of "not indexed"). The tools now return their normal error result (a
tool error, `isError: true`, the session carries on) with the CLI's words, before anything
is opened. `index_codebase`, which creates the index, is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from trelix_mcp.server import mcp

from trelix.core.models import Chunk, IndexedFile, Language, SearchResult, Symbol, SymbolKind

_READ_TOOLS = [
    pytest.param("search_code", {"query": "add numbers"}, id="search_code"),
    pytest.param("get_symbol", {"qualified_name": "add"}, id="get_symbol"),
    pytest.param("blast_radius", {"symbol_name": "add"}, id="blast_radius"),
    pytest.param("build_knowledge_graph", {}, id="build_knowledge_graph"),
    pytest.param("graph_search_mcp", {"query": "add"}, id="graph_search_mcp"),
    pytest.param("ask_agent", {"query": "what does add do"}, id="ask_agent"),
    pytest.param("agent_list_sessions", {}, id="agent_list_sessions"),
    pytest.param("agent_clear_session", {"session_id": "s1"}, id="agent_clear_session"),
]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return root


def _expected_message(repo: Path) -> str:
    """IndexConfig resolves repo_path, so the message names the resolved path."""
    resolved = repo.resolve()
    return f"No index found at {resolved}/.trelix/index.db. Run trelix index {resolved} first."


@pytest.mark.parametrize(("tool", "arguments"), _READ_TOOLS)
async def test_read_tool_on_an_unindexed_repo_returns_a_tool_error_and_creates_nothing(
    repo: Path, tool: str, arguments: dict[str, Any]
) -> None:
    async with Client(mcp, mode="legacy") as client:
        result = await client.call_tool(
            tool, {**arguments, "repo_path": str(repo)}, raise_on_error=False
        )

    assert result.is_error is True
    assert [block.text for block in result.content] == [_expected_message(repo)]
    assert not (repo / ".trelix").exists(), (
        "a read tool must not create .trelix/ (or index.db) in an unindexed repo"
    )


@pytest.mark.parametrize(("tool", "arguments"), _READ_TOOLS)
async def test_read_tool_refuses_before_opening_any_store(
    repo: Path, tool: str, arguments: dict[str, Any]
) -> None:
    with (
        patch("trelix_mcp.server.Retriever") as retriever,
        patch("trelix_mcp.server.Database") as database,
        patch("trelix_mcp.server.AgentLoop") as agent_loop,
        patch("trelix.graph.builder.GraphBuilder") as builder,
    ):
        async with Client(mcp, mode="legacy") as client:
            result = await client.call_tool(
                tool, {**arguments, "repo_path": str(repo)}, raise_on_error=False
            )

    assert result.is_error is True
    retriever.assert_not_called()
    database.assert_not_called()
    agent_loop.assert_not_called()
    builder.assert_not_called()


def test_a_refused_search_does_not_poison_the_retriever_cache(
    repo: Path, mark_indexed: Any
) -> None:
    """Once the repo is indexed, the next call must build its Retriever and work."""
    from trelix_mcp.server import search_code

    with pytest.raises(ToolError):
        search_code(query="add", repo_path=str(repo))

    mark_indexed(repo)
    context = MagicMock()
    context.results = []
    with patch("trelix_mcp.server.Retriever") as retriever:
        retriever.return_value.retrieve.return_value = context
        response = search_code(query="add", repo_path=str(repo))

    assert response == {"results": [], "next_cursor": None, "total_available": 0}
    retriever.assert_called_once()


def test_agent_list_sessions_on_an_indexed_repo_still_answers(
    repo: Path, mark_indexed: Any
) -> None:
    from trelix_mcp.server import agent_list_sessions

    mark_indexed(repo)

    assert agent_list_sessions(repo_path=str(repo)) == {"sessions": [], "count": 0}


def test_agent_clear_session_on_an_indexed_repo_still_answers(
    repo: Path, mark_indexed: Any
) -> None:
    from trelix_mcp.server import agent_clear_session

    mark_indexed(repo)

    assert agent_clear_session(repo_path=str(repo), session_id="s1") == {
        "cleared": False,
        "session_id": "s1",
    }


def test_get_symbol_and_blast_radius_on_an_indexed_repo_still_answer(
    repo: Path, mark_indexed: Any
) -> None:
    from trelix_mcp.server import blast_radius, get_symbol

    mark_indexed(repo)

    assert get_symbol(qualified_name="add", repo_path=str(repo)) is None
    assert blast_radius(symbol_name="add", repo_path=str(repo)) == []


def test_build_knowledge_graph_on_an_indexed_repo_still_builds(
    repo: Path, mark_indexed: Any
) -> None:
    from trelix_mcp.server import build_knowledge_graph

    mark_indexed(repo)

    assert build_knowledge_graph(repo_path=str(repo))["node_count"] == 0


# ---------------------------------------------------------------------------
# Resources: they never raise, they reply {"error": ...}
# ---------------------------------------------------------------------------


def test_resources_on_an_unindexed_repo_reply_with_the_error_and_create_nothing(
    repo: Path,
) -> None:
    from trelix_mcp.resources import get_index_stats, get_repo_manifest, get_symbol_source

    replies = [
        json.loads(get_index_stats(str(repo))),
        json.loads(get_repo_manifest(str(repo))),
        json.loads(get_symbol_source(str(repo), "add")),
    ]

    assert [reply["error"] for reply in replies] == [_expected_message(repo)] * 3
    assert not (repo / ".trelix").exists()


def test_resources_on_an_indexed_repo_still_answer(repo: Path, mark_indexed: Any) -> None:
    from trelix_mcp.resources import get_index_stats, get_repo_manifest

    mark_indexed(repo)

    assert json.loads(get_index_stats(str(repo))) == {
        "symbol_count": 0,
        "file_count": 0,
        "chunk_count": 0,
        "repo_path": str(repo),
    }
    assert json.loads(get_repo_manifest(str(repo)))["files"] == []


# ---------------------------------------------------------------------------
# federation_search_all: a registered repo with no index is skipped, not opened
# ---------------------------------------------------------------------------


def _federation_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *repos: Path) -> str:
    """A repos.json under <cwd>/.trelix/, the only place the tool reads a registry from."""
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / ".trelix"
    config_dir.mkdir()
    config = config_dir / "repos.json"
    entries = [{"alias": f"r{i}", "path": str(p), "weight": 1.0} for i, p in enumerate(repos)]
    config.write_text(json.dumps({"repos": entries}), encoding="utf-8")
    return str(config)


def _search_result() -> SearchResult:
    return SearchResult(
        chunk=Chunk(symbol_id=1, chunk_text="code", token_count=4, id=1),
        symbol=Symbol(
            file_id=1,
            name="handler",
            qualified_name="handler",
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=2,
            signature="def handler()",
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


def test_federation_search_all_with_nothing_indexed_says_so_and_creates_nothing(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trelix_mcp.server import federation_search_all

    config = _federation_config(tmp_path, monkeypatch, repo)

    with patch("trelix.federation.retriever.Retriever") as retriever:
        response = federation_search_all(query="add", config_path=config)

    assert response == {
        "results": [],
        "next_cursor": None,
        "total_available": 0,
        "repos_searched": 0,
        "repos_skipped": 0,
        "error": _expected_message_for(repo),
    }
    retriever.assert_not_called()
    assert not (repo / ".trelix").exists()


def _expected_message_for(repo: Path) -> str:
    """The registry keeps a path as registered, and federation does not resolve it."""
    return f"No index found at {repo}/.trelix/index.db. Run trelix index {repo} first."


def test_federation_search_all_names_a_registered_path_that_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool asks `unindexed_repos()` outside the fan-out's catch-all, so a registry entry
    whose directory is gone must come back as the error string, not as an exception."""
    from trelix_mcp.server import federation_search_all

    ghost = tmp_path / "gone" / "repo"
    config = _federation_config(tmp_path, monkeypatch, ghost)

    with patch("trelix.federation.retriever.Retriever") as retriever:
        response = federation_search_all(query="add", config_path=config)

    assert response == {
        "results": [],
        "next_cursor": None,
        "total_available": 0,
        "repos_searched": 0,
        "repos_skipped": 0,
        "error": _expected_message_for(ghost),
    }
    retriever.assert_not_called()
    assert not (tmp_path / "gone").exists()


def test_federation_search_all_skips_the_unindexed_repo_and_searches_the_rest(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mark_indexed: Any
) -> None:
    from trelix_mcp.server import federation_search_all

    indexed = mark_indexed(tmp_path / "indexed")
    config = _federation_config(tmp_path, monkeypatch, repo, indexed)

    with patch("trelix.federation.retriever.Retriever") as retriever:
        retriever.return_value.retrieve.return_value = MagicMock(results=[_search_result()])
        response = federation_search_all(query="add", config_path=config)

    assert response["error"] is None
    assert response["repos_searched"] == 1
    assert [row["symbol"] for row in response["results"]] == ["handler"]
    assert retriever.call_count == 1
    assert not (repo / ".trelix").exists()


def test_federation_search_all_names_every_repo_when_none_is_indexed(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trelix_mcp.server import federation_search_all

    second = tmp_path / "second"
    second.mkdir()
    config = _federation_config(tmp_path, monkeypatch, repo, second)

    with patch("trelix.federation.retriever.Retriever") as retriever:
        response = federation_search_all(query="add", config_path=config)

    assert response["error"] == f"{_expected_message_for(repo)} {_expected_message_for(second)}"
    assert response["repos_searched"] == 0
    assert response["repos_skipped"] == 0
    retriever.assert_not_called()


def test_federation_search_all_counts_the_repos_beyond_the_cap_when_none_is_indexed(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the first TRELIX_FEDERATION_MAX_REPOS repos are queried. The rest are skipped by the
    cap, not for lack of an index, so `repos_skipped` counts them and `error` does not."""
    from trelix_mcp.server import federation_search_all

    second = tmp_path / "second"
    second.mkdir()
    monkeypatch.setenv("TRELIX_FEDERATION_MAX_REPOS", "1")
    config = _federation_config(tmp_path, monkeypatch, repo, second)

    with patch("trelix.federation.retriever.Retriever") as retriever:
        response = federation_search_all(query="add", config_path=config)

    assert response == {
        "results": [],
        "next_cursor": None,
        "total_available": 0,
        "repos_searched": 0,
        "repos_skipped": 1,
        "error": _expected_message_for(repo),
    }
    retriever.assert_not_called()
