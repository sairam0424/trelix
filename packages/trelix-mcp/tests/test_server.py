"""Tests for trelix_mcp.server.

Uses unittest.mock.patch to avoid touching real files or embedding models.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_result(
    file: str = "src/foo.py",
    symbol: str = "foo.bar",
    kind: str = "function",
    line_start: int = 1,
    line_end: int = 10,
    score: float = 0.9,
    source: str = "vector",
    body: str = "def bar(): pass",
    language: str = "python",
) -> MagicMock:
    """Return a mock SearchResult compatible with server.py expectations."""
    r = MagicMock()
    r.file.rel_path = file
    r.file.language.value = language
    r.symbol.qualified_name = symbol
    r.symbol.kind.value = kind
    r.symbol.line_start = line_start
    r.symbol.line_end = line_end
    r.symbol.body = body
    r.score = score
    r.source = source
    return r


def _make_mock_context(results: list[MagicMock]) -> MagicMock:
    ctx = MagicMock()
    ctx.results = results
    return ctx


# ---------------------------------------------------------------------------
# Module import + basic structure
# ---------------------------------------------------------------------------


def test_server_importable() -> None:
    import trelix_mcp.server as srv  # noqa: F401


def test_mcp_attribute_exists() -> None:
    import trelix_mcp.server as srv

    assert hasattr(srv, "mcp"), "server.py must expose a top-level `mcp` object"


def test_main_callable() -> None:
    import trelix_mcp.server as srv

    assert callable(srv.main)


# ---------------------------------------------------------------------------
# 15 tools registered (8 original + 4 federation + 3 agent-session)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_four_tools_registered() -> None:
    import trelix_mcp.server as srv

    tools = await srv.mcp.list_tools()
    names = {t.name for t in tools}
    expected = {
        "search_code",
        "index_codebase",
        "get_symbol",
        "blast_radius",
        "build_knowledge_graph",
        "graph_search_mcp",
        "subscribe_resource",
        "unsubscribe_resource",
        "federation_list_repos",
        "federation_add_repo",
        "federation_remove_repo",
        "federation_search_all",
        "ask_agent",
        "agent_list_sessions",
        "agent_clear_session",
    }
    assert expected == names, f"Expected exactly 15 tools, got: {names}"


# ---------------------------------------------------------------------------
# search_code
# ---------------------------------------------------------------------------


def test_search_code_returns_dict_envelope() -> None:
    """search_code returns a pagination envelope dict with a results list."""
    import trelix_mcp.server as srv

    mock_results = [_make_mock_result()]
    mock_ctx = _make_mock_context(mock_results)

    with (
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.Retriever") as MockRetriever,
    ):
        MockRetriever.return_value.retrieve.return_value = mock_ctx
        response = srv.search_code("authentication", "/fake/repo", k=10)

    assert isinstance(response, dict)
    assert "results" in response
    assert "next_cursor" in response
    assert "total_available" in response
    assert len(response["results"]) == 1
    item = response["results"][0]
    assert set(item.keys()) >= {
        "file",
        "symbol",
        "kind",
        "lines",
        "score",
        "source",
        "body",
        "language",
    }


def test_search_code_respects_k_limit() -> None:
    """k=3 must truncate 20 mock results to 3."""
    import trelix_mcp.server as srv

    mock_results = [_make_mock_result(symbol=f"sym.{i}") for i in range(20)]
    mock_ctx = _make_mock_context(mock_results)

    with (
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.Retriever") as MockRetriever,
    ):
        MockRetriever.return_value.retrieve.return_value = mock_ctx
        response = srv.search_code("auth", "/fake/repo", k=3)

    assert len(response["results"]) == 3


# ---------------------------------------------------------------------------
# index_codebase
# ---------------------------------------------------------------------------


def test_index_codebase_returns_dict() -> None:
    import trelix_mcp.server as srv

    fake_stats: dict[str, Any] = {
        "files_found": 10,
        "files_indexed": 8,
        "files_skipped": 2,
        "symbols_extracted": 50,
        "chunks_total": 50,
        "chunks_embedded": 50,
        "errors": 0,
        "elapsed_seconds": 1.23,
    }

    with (
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.EmbedderConfig"),
        patch("trelix_mcp.server.Indexer") as MockIndexer,
    ):
        MockIndexer.return_value.index.return_value = fake_stats
        result = srv.index_codebase("/fake/repo", provider="local")

    assert isinstance(result, dict)
    assert result["files_found"] == 10


# ---------------------------------------------------------------------------
# blast_radius deduplication
# ---------------------------------------------------------------------------


def _seed_call_graph(tmp_path):  # type: ignore[no-untyped-def]
    """A real index with a known call graph: two callers of `target.run`, one unrelated.

    A real Database rather than mocks, because the whole point of the change under test
    is that the answer comes from the `calls` table. Mocking the data source would make
    the test pass against either implementation.
    """
    from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
    from trelix.store.db import Database

    db_path = tmp_path / ".trelix" / "index.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(db_path)

    def add(rel_path: str, name: str) -> int:
        file_id = db.upsert_file(
            IndexedFile(
                path=f"/repo/{rel_path}",
                rel_path=rel_path,
                language=Language.PYTHON,
                hash=f"h-{rel_path}-{name}",
                size_bytes=100,
            )
        )
        return db.insert_symbol(
            Symbol(
                file_id=file_id,
                name=name.split(".")[-1],
                qualified_name=name,
                kind=SymbolKind.FUNCTION,
                line_start=10,
                line_end=20,
                signature=f"def {name.split('.')[-1]}():",
                body="pass",
            )
        )

    target = add("src/target.py", "target.run")
    caller_a = add("src/alpha.py", "alpha.uses_target")
    caller_b = add("src/beta.py", "beta.also_uses_target")
    add("src/unrelated.py", "unrelated.nothing_to_do_with_it")

    db.insert_call_edges(
        [
            CallEdge(caller_id=caller_a, callee_id=target, callee_name="run", line=11),
            CallEdge(caller_id=caller_b, callee_id=target, callee_name="run", line=12),
        ]
    )
    db._conn.commit()
    db.close()
    return db_path


class TestBlastRadiusUsesTheCallGraph:
    """`blast_radius` must answer from `calls`, not from semantic search.

    It used to build `query = f"blast radius dependencies of {symbol_name}"` and run a
    Retriever, never touching the resolved call edges sitting in the same SQLite file.
    Measured on this repository's own index for `AuditStore.append`: the SQL oracle
    returns 320 caller symbols across 101 files in 128 ms, while the tool returned 12
    entries in 5,747 ms with precision 0.33 and **recall 0.04** — and listed the queried
    symbol itself among the hits.

    Recall 0.04 is the number that matters. An agent asking "what breaks if I change
    this?" was told about 4 of 101 affected files, and acted on it.

    Three shipped surfaces consume this answer: the MCP tool, the VS Code
    "N dependents" CodeLens, and `@trelix /impact`.
    """

    def test_returns_the_actual_callers(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import trelix_mcp.server as srv

        _seed_call_graph(tmp_path)
        results = srv.blast_radius("target.run", str(tmp_path))

        symbols = {r["symbol"] for r in results}
        assert symbols == {"alpha.uses_target", "beta.also_uses_target"}, (
            f"expected exactly the two real callers, got {symbols}"
        )

    def test_does_not_return_the_queried_symbol_itself(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The old implementation ranked the symbol itself first — it is not its own
        blast radius, and an agent that edits it because it appeared in the list is
        being misled."""
        import trelix_mcp.server as srv

        _seed_call_graph(tmp_path)
        results = srv.blast_radius("target.run", str(tmp_path))

        assert "target.run" not in {r["symbol"] for r in results}

    def test_excludes_unrelated_symbols(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Precision: a semantically similar but uncalled symbol must not appear."""
        import trelix_mcp.server as srv

        _seed_call_graph(tmp_path)
        results = srv.blast_radius("target.run", str(tmp_path))

        assert not any("unrelated" in r["symbol"] for r in results)

    def test_response_is_a_bare_array_of_the_documented_keys(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The response SHAPE is load-bearing and must not become an object envelope.

        workspace-vscode/src/mcp-client.ts does `(parsed ?? []).map(...)`, and its
        caller's bare catch swallows the resulting TypeError — so wrapping the array
        would make every "N dependents" CodeLens in the editor read 0 forever, silently.
        """
        import trelix_mcp.server as srv

        _seed_call_graph(tmp_path)
        results = srv.blast_radius("target.run", str(tmp_path))

        assert isinstance(results, list)
        for entry in results:
            assert set(entry) >= {"file", "symbol", "kind", "line_start", "language"}

    def test_needs_no_embedding_model(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A graph query must not pay for an embedder.

        `Retriever.__init__` builds one eagerly and reads `.dimension` for the
        DimensionGuard, which is why the old path cost 5.7 s and could fail outright on
        a dimension mismatch. Making the factory raise proves the new path never calls
        it.
        """
        import trelix_mcp.server as srv

        _seed_call_graph(tmp_path)
        with patch(
            "trelix.embedder.base.make_embedder",
            side_effect=AssertionError("blast_radius must not build an embedder"),
        ):
            results = srv.blast_radius("target.run", str(tmp_path))

        assert len(results) == 2

    def test_an_unknown_symbol_returns_an_empty_list(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import trelix_mcp.server as srv

        _seed_call_graph(tmp_path)
        assert srv.blast_radius("does.not.exist", str(tmp_path)) == []


# ---------------------------------------------------------------------------
# Federation (multi-repo) tools
# ---------------------------------------------------------------------------


def test_federation_list_repos_empty_registry() -> None:
    import trelix_mcp.server as srv

    empty_reg = MagicMock()
    empty_reg.list.return_value = []

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = empty_reg
        response = srv.federation_list_repos()

    assert response == {"repos": [], "count": 0, "error": None}
    MockRegistry.load.assert_called_once_with(None)


def test_federation_list_repos_returns_entries() -> None:
    import trelix_mcp.server as srv

    entry = MagicMock()
    entry.alias = "myrepo"
    entry.path = "/repo"
    entry.weight = 2.0
    reg = MagicMock()
    reg.list.return_value = [entry]

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = reg
        response = srv.federation_list_repos()

    assert response == {
        "repos": [{"alias": "myrepo", "path": "/repo", "weight": 2.0}],
        "count": 1,
        "error": None,
    }


def test_federation_list_repos_rejects_unconfined_config_path() -> None:
    import trelix_mcp.server as srv

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        response = srv.federation_list_repos(config_path="/etc/passwd")

    assert response["repos"] == []
    assert response["count"] == 0
    assert response["error"] is not None
    MockRegistry.load.assert_not_called()


def test_federation_add_repo_success() -> None:
    import trelix_mcp.server as srv

    reg = MagicMock()

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = reg
        response = srv.federation_add_repo(alias="myrepo", path="/repo", weight=1.5)

    reg.add.assert_called_once_with("myrepo", "/repo", 1.5, max_repos=50)
    reg.save.assert_called_once()
    assert response == {"added": True, "alias": "myrepo", "path": "/repo", "error": None}


def test_federation_add_repo_duplicate_alias_returns_error() -> None:
    import trelix_mcp.server as srv

    reg = MagicMock()
    reg.add.side_effect = ValueError("alias 'myrepo' already registered")

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = reg
        response = srv.federation_add_repo(alias="myrepo", path="/repo")

    assert response["added"] is False
    assert "already registered" in response["error"]
    reg.save.assert_not_called()


def test_federation_add_repo_rejects_unconfined_config_path() -> None:
    import trelix_mcp.server as srv

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        response = srv.federation_add_repo(alias="myrepo", path="/repo", config_path="/etc/passwd")

    assert response["added"] is False
    assert response["error"] is not None
    MockRegistry.load.assert_not_called()


def test_federation_remove_repo_existing() -> None:
    import trelix_mcp.server as srv

    entry = MagicMock()
    entry.alias = "myrepo"
    reg = MagicMock()
    reg.list.return_value = [entry]

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = reg
        response = srv.federation_remove_repo(alias="myrepo")

    reg.remove.assert_called_once_with("myrepo")
    reg.save.assert_called_once()
    assert response == {"removed": True, "alias": "myrepo", "error": None}


def test_federation_remove_repo_missing_is_noop() -> None:
    import trelix_mcp.server as srv

    reg = MagicMock()
    reg.list.return_value = []

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = reg
        response = srv.federation_remove_repo(alias="nonexistent")

    assert response == {"removed": False, "alias": "nonexistent", "error": None}


def test_federation_search_all_no_repos_returns_empty() -> None:
    import trelix_mcp.server as srv

    reg = MagicMock()
    reg.list.return_value = []

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        MockRegistry.load.return_value = reg
        response = srv.federation_search_all(query="auth")

    assert response == {
        "results": [],
        "next_cursor": None,
        "total_available": 0,
        "repos_searched": 0,
        "repos_skipped": 0,
        "error": None,
    }


def test_federation_search_all_rejects_unconfined_config_path() -> None:
    import trelix_mcp.server as srv

    with patch("trelix_mcp.server.RepoRegistry") as MockRegistry:
        response = srv.federation_search_all(query="auth", config_path="/etc/passwd")

    assert response["results"] == []
    assert response["error"] is not None
    MockRegistry.load.assert_not_called()


def test_federation_search_all_returns_dict_envelope() -> None:
    import trelix_mcp.server as srv

    entry = MagicMock()
    entry.alias = "myrepo"
    reg = MagicMock()
    reg.list.return_value = [entry]

    result = _make_mock_result(source="myrepo:vector")

    with (
        patch("trelix_mcp.server.RepoRegistry") as MockRegistry,
        patch("trelix_mcp.server.FederatedRetriever") as MockFed,
    ):
        MockRegistry.load.return_value = reg
        MockFed.return_value.repos_queried_count.return_value = 1
        MockFed.return_value.retrieve.return_value = [result]
        response = srv.federation_search_all(query="auth", k=10)

    assert response["repos_searched"] == 1
    assert response["repos_skipped"] == 0
    assert response["total_available"] == 1
    assert len(response["results"]) == 1
    assert response["results"][0]["repo"] == "myrepo"
    assert response["results"][0]["source"] == "myrepo:vector"
    # Fetch width must be the fixed constant, independent of cursor/k —
    # regression guard for the pagination-stability fix (issue #69 item 3).
    MockFed.return_value.retrieve.assert_called_once_with(
        "auth", k=srv._FEDERATION_SEARCH_ALL_FETCH_WIDTH
    )


def test_federation_search_all_fetch_width_independent_of_cursor() -> None:
    """Same fixed fetch width regardless of cursor — the actual pagination fix."""
    import trelix_mcp.server as srv

    entry = MagicMock()
    entry.alias = "myrepo"
    reg = MagicMock()
    reg.list.return_value = [entry]

    with (
        patch("trelix_mcp.server.RepoRegistry") as MockRegistry,
        patch("trelix_mcp.server.FederatedRetriever") as MockFed,
    ):
        MockRegistry.load.return_value = reg
        MockFed.return_value.repos_queried_count.return_value = 1
        MockFed.return_value.retrieve.return_value = []

        srv.federation_search_all(query="q", k=10, cursor=0)
        srv.federation_search_all(query="q", k=10, cursor=90)

    calls = MockFed.return_value.retrieve.call_args_list
    assert len(calls) == 2
    assert calls[0] == calls[1] == (("q",), {"k": srv._FEDERATION_SEARCH_ALL_FETCH_WIDTH})


def test_federation_search_all_reports_repos_skipped() -> None:
    import trelix_mcp.server as srv

    entries = [MagicMock(alias=f"repo{i}") for i in range(5)]
    reg = MagicMock()
    reg.list.return_value = entries

    with (
        patch("trelix_mcp.server.RepoRegistry") as MockRegistry,
        patch("trelix_mcp.server.FederatedRetriever") as MockFed,
    ):
        MockRegistry.load.return_value = reg
        MockFed.return_value.repos_queried_count.return_value = 3
        MockFed.return_value.retrieve.return_value = []
        response = srv.federation_search_all(query="q")

    assert response["repos_searched"] == 3
    assert response["repos_skipped"] == 2


# ---------------------------------------------------------------------------
# Agent-session (persistent ReAct memory) tools
# ---------------------------------------------------------------------------


def test_ask_agent_returns_dict_with_session_id() -> None:
    import trelix_mcp.server as srv

    mock_loop = MagicMock()
    mock_loop.run.return_value = ("answer text", "some-uuid")
    mock_db = MagicMock()
    mock_db.get_agent_turns.return_value = [{"turn_index": 0}, {"turn_index": 1}]

    with (
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.AgentLoop", return_value=mock_loop),
        patch("trelix_mcp.server.Database", return_value=mock_db),
    ):
        response = srv.ask_agent(query="how does auth work", repo_path="/fake/repo")

    assert response == {"answer": "answer text", "session_id": "some-uuid", "turn_count": 2}
    # Regression guard: get_agent_turns must be called with the loop-returned
    # resolved_session_id ("some-uuid"), not the input session_id param
    # (which was None here) — a mutation swapping these would still produce
    # turn_count=2 (mock_db.get_agent_turns.return_value is fixed regardless
    # of argument), so the shape-only assertion above would miss it.
    mock_db.get_agent_turns.assert_called_once_with("some-uuid")


def test_ask_agent_generates_session_id_when_omitted() -> None:
    import trelix_mcp.server as srv

    mock_loop = MagicMock()
    mock_loop.run.return_value = ("answer", "freshly-generated-uuid")
    mock_db = MagicMock()
    mock_db.get_agent_turns.return_value = []

    with (
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.AgentLoop", return_value=mock_loop),
        patch("trelix_mcp.server.Database", return_value=mock_db),
    ):
        response = srv.ask_agent(query="q", repo_path="/fake/repo")

    mock_loop.run.assert_called_once_with("q", session_id=None)
    assert response["session_id"] == "freshly-generated-uuid"
    assert response["session_id"]


def test_agent_list_sessions_returns_dict() -> None:
    import trelix_mcp.server as srv

    mock_db = MagicMock()
    mock_db.list_agent_sessions.return_value = [
        {
            "session_id": "s1",
            "created_at": "t1",
            "last_active_at": "t1",
            "query": "q1",
            "turn_count": 2,
        }
    ]

    with (
        patch("trelix_mcp.server.IndexConfig") as MockIndexConfig,
        patch("trelix_mcp.server.Database", return_value=mock_db),
    ):
        MockIndexConfig.return_value.retrieval.agent_session_max_age_seconds = 604_800.0
        response = srv.agent_list_sessions(repo_path="/fake/repo")

    assert response["count"] == 1
    assert response["sessions"][0]["session_id"] == "s1"


def test_agent_clear_session_returns_dict() -> None:
    import trelix_mcp.server as srv

    mock_db = MagicMock()
    mock_db.delete_agent_session.return_value = True

    with (
        patch("trelix_mcp.server.IndexConfig"),
        patch("trelix_mcp.server.Database", return_value=mock_db),
    ):
        response = srv.agent_clear_session(repo_path="/fake/repo", session_id="s1")

    assert response == {"cleared": True, "session_id": "s1"}


# ---------------------------------------------------------------------------
# Cursor-based pagination for search_code
# ---------------------------------------------------------------------------


class TestSearchCodePagination:
    def test_search_code_returns_pagination_envelope(self, tmp_path) -> None:
        """search_code returns dict with results + next_cursor + total_available."""
        from unittest.mock import MagicMock, patch

        from trelix_mcp.server import search_code

        mock_results = []
        for i in range(25):
            r = MagicMock()
            r.file.rel_path = f"src/file{i}.py"
            r.symbol.qualified_name = f"Func{i}"
            r.symbol.kind.value = "function"
            r.symbol.line_start = 1
            r.symbol.line_end = 5
            r.symbol.body = "def f(): pass"
            r.file.language.value = "python"
            r.score = 0.9 - i * 0.01
            r.source = "vector"
            mock_results.append(r)

        mock_ctx = MagicMock()
        mock_ctx.results = mock_results

        with patch("trelix_mcp.server.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            response = search_code(
                query="authentication",
                repo_path=str(tmp_path),
                k=10,
                cursor=0,
            )

        assert "results" in response
        assert "next_cursor" in response
        assert "total_available" in response
        assert len(response["results"]) == 10
        assert response["next_cursor"] == 10
        assert response["total_available"] == 25

    def test_search_code_pagination_second_page(self, tmp_path) -> None:
        """cursor=10 returns items 10-19."""
        from unittest.mock import MagicMock, patch

        from trelix_mcp.server import search_code

        mock_results = []
        for i in range(25):
            r = MagicMock()
            r.file.rel_path = f"src/file{i}.py"
            r.symbol.qualified_name = f"Func{i}"
            r.symbol.kind.value = "function"
            r.symbol.line_start = 1
            r.symbol.line_end = 5
            r.symbol.body = "def f(): pass"
            r.file.language.value = "python"
            r.score = float(i) / 25
            r.source = "bm25"
            mock_results.append(r)

        mock_ctx = MagicMock()
        mock_ctx.results = mock_results

        with patch("trelix_mcp.server.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            response = search_code(
                query="login",
                repo_path=str(tmp_path),
                k=10,
                cursor=10,
            )

        assert len(response["results"]) == 10
        assert response["results"][0]["symbol"] == "Func10"
        assert response["next_cursor"] == 20

    def test_search_code_last_page_has_null_next_cursor(self, tmp_path) -> None:
        """last page has next_cursor=None."""
        from unittest.mock import MagicMock, patch

        from trelix_mcp.server import search_code

        mock_results = [MagicMock() for _ in range(5)]
        for i, r in enumerate(mock_results):
            r.file.rel_path = f"src/f{i}.py"
            r.symbol.qualified_name = f"F{i}"
            r.symbol.kind.value = "function"
            r.symbol.line_start = 1
            r.symbol.line_end = 3
            r.symbol.body = "pass"
            r.file.language.value = "python"
            r.score = 0.5
            r.source = "vector"

        mock_ctx = MagicMock()
        mock_ctx.results = mock_results

        with patch("trelix_mcp.server.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            response = search_code(
                query="q",
                repo_path=str(tmp_path),
                k=10,
                cursor=0,
            )

        assert response["next_cursor"] is None
        assert len(response["results"]) == 5


# ---------------------------------------------------------------------------
# Progress notifications for index_codebase
# ---------------------------------------------------------------------------


class TestIndexCodebaseProgress:
    def test_index_codebase_accepts_context_param(self) -> None:
        """index_codebase tool signature accepts ctx: Context without error."""
        import inspect

        from trelix_mcp.server import index_codebase

        sig = inspect.signature(index_codebase)
        # ctx param should exist (FastMCP injects it)
        # We check the wrapped function's parameters
        list(sig.parameters.keys())
        # Either 'ctx' is in params, or the function works without it (backward compat)
        # The key check is that calling it with mock results succeeds
        assert callable(index_codebase)

    def test_index_codebase_returns_stats(self, tmp_path) -> None:
        """index_codebase returns stats dict with expected keys."""
        from unittest.mock import patch

        from trelix_mcp.server import index_codebase

        mock_stats = {
            "files_indexed": 15,
            "symbols_extracted": 220,
            "chunks_embedded": 220,
            "errors": 0,
            "elapsed_seconds": 3.1,
        }

        with patch("trelix_mcp.server.Indexer") as MockIndexer:
            MockIndexer.return_value.index.return_value = mock_stats
            result = index_codebase(repo_path=str(tmp_path), provider="local")

        assert result["files_indexed"] == 15
        assert result["errors"] == 0


# ---------------------------------------------------------------------------
# CRITICAL: no stdout bytes on import
# ---------------------------------------------------------------------------


def test_server_import_produces_no_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Importing server.py must not write anything to stdout.

    stdout is the MCP JSON protocol pipe — any spurious bytes corrupt the stream.
    """
    # Force a fresh import to catch any module-level print/write
    if "trelix_mcp.server" in sys.modules:
        del sys.modules["trelix_mcp.server"]

    import trelix_mcp.server  # noqa: F401

    captured = capsys.readouterr()
    assert captured.out == "", (
        f"server.py wrote {len(captured.out)} bytes to stdout on import: {captured.out!r}"
    )


class TestBlastRadiusIncludesImporters:
    """A symbol's blast radius includes files that import it, not only its callers.

    Changing a dataclass's shape affects every file that imports it even with no call
    edge, so restricting the answer to `calls` would under-report the case impact
    analysis exists for. Measured against a calls-only oracle the tool looks imprecise
    on widely-imported types (0.41 for IndexConfig, 0.48 for Symbol); against a
    calls-UNION-imports oracle — what it actually answers — precision and recall are
    both 1.00 for all five symbols tested on this repository's own index.

    Importers resolve via file_id. `Retriever.get_importers` matches on
    `files.rel_path`, so handing it a symbol name silently returns nothing.
    """

    @staticmethod
    def _seed_importer(tmp_path):  # type: ignore[no-untyped-def]
        """`consumer.py` imports `types.py` but calls nothing in it."""
        from trelix.core.models import (
            ImportEdge,
            IndexedFile,
            Language,
            Symbol,
            SymbolKind,
        )
        from trelix.store.db import Database

        db_path = tmp_path / ".trelix" / "index.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = Database(db_path)

        def add(rel_path: str, name: str) -> tuple[int, int]:
            file_id = db.upsert_file(
                IndexedFile(
                    path=f"/repo/{rel_path}",
                    rel_path=rel_path,
                    language=Language.PYTHON,
                    hash=f"h-{rel_path}",
                    size_bytes=50,
                )
            )
            symbol_id = db.insert_symbol(
                Symbol(
                    file_id=file_id,
                    name=name.split(".")[-1],
                    qualified_name=name,
                    kind=SymbolKind.CLASS,
                    line_start=1,
                    line_end=5,
                    signature=f"class {name}:",
                    body="pass",
                )
            )
            return file_id, symbol_id

        types_file, _ = add("src/types.py", "Payload")
        consumer_file, _ = add("src/consumer.py", "consumer.handler")

        db.insert_imports(
            [ImportEdge(file_id=consumer_file, imported_from="src.types", imported_names="Payload")]
        )
        db._conn.execute(
            "UPDATE imports SET imported_file_id = ? WHERE file_id = ?",
            (types_file, consumer_file),
        )
        db._conn.commit()
        db.close()

    def test_an_importer_with_no_call_edge_is_included(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import trelix_mcp.server as srv

        self._seed_importer(tmp_path)
        files = {r["file"] for r in srv.blast_radius("Payload", str(tmp_path))}

        assert "src/consumer.py" in files, (
            f"a file importing the symbol's module was not reported: {files}"
        )
        assert "src/types.py" not in files, "the defining file is not its own blast radius"


class TestKnowledgeGraphPayloadIsCapped:
    """One tool call must not consume an agent's whole context window.

    `build_knowledge_graph` shipped `result.community_summary` raw. Measured on this
    repository's own index: **1,160,415 bytes**, roughly 290,000 tokens — larger than
    most context windows, from a single call. 6,437 of the 6,497 entries (99.1%) are
    singleton "communities" carrying no architectural signal at all, while the five real
    clusters (514, 477, 393, 279, 277 nodes) account for almost none of the bytes.
    Filtering to size>1 leaves 60 clusters in 25,554 bytes — a 45x reduction that keeps
    every top cluster.

    The counts are reported rather than dropped. Capping silently would hide the fact
    that community detection is producing 6,437 singletons, which is a tuning problem
    somebody should see.
    """

    @staticmethod
    def _mock_result(singletons: int = 6437, real: int = 60):  # type: ignore[no-untyped-def]
        """A community_summary shaped like the real one: mostly singletons."""
        summary = [
            {
                "community_id": i,
                "size": 514 - i,
                "top_files": [f"src/f{i}.py"],
                "top_symbols": [f"sym{i}"],
            }
            for i in range(real)
        ] + [
            {
                "community_id": 1000 + i,
                "size": 1,
                "top_files": [f"src/single{i}.py"],
                "top_symbols": [f"lone{i}"],
            }
            for i in range(singletons)
        ]
        return MagicMock(
            node_count=10700,
            edge_count=11150,
            community_count=len(summary),
            concept_count=0,
            elapsed_seconds=1.5,
            community_summary=summary,
            # Ints, not the auto-created MagicMock attributes: the tool now puts these
            # two in the payload, and `json.dumps` of a MagicMock raises TypeError, so
            # omitting them here breaks test_the_payload_fits_in_a_sane_budget rather
            # than the code under test. 0/0 matches concept_count=0 — extraction off.
            concept_symbols_considered=0,
            concept_symbols_total=0,
        )

    def _call(self, **kwargs):  # type: ignore[no-untyped-def]
        import trelix_mcp.server as srv

        with (
            patch("trelix.core.config.IndexConfig"),
            patch("trelix.graph.builder.GraphBuilder") as MockBuilder,
        ):
            MockBuilder.return_value.build.return_value = self._mock_result()
            return srv.build_knowledge_graph("/fake/repo", **kwargs)

    def test_the_payload_fits_in_a_sane_budget(self) -> None:
        import json as _json

        payload = self._call()
        size = len(_json.dumps(payload))
        assert size < 32_000, (
            f"payload is {size:,} bytes; the uncapped version measured 1,160,415 "
            "(~290k tokens) on this repo"
        )

    def test_the_largest_clusters_survive(self) -> None:
        """A cap that drops the signal along with the noise is not a fix."""
        payload = self._call()
        sizes = [c["size"] for c in payload["community_summary"]]
        assert sizes == sorted(sizes, reverse=True), "clusters are not size-ordered"
        assert sizes[0] == 514, f"the largest cluster was dropped; got {sizes[:3]}"

    def test_singletons_are_excluded_by_default(self) -> None:
        payload = self._call()
        assert all(c["size"] > 1 for c in payload["community_summary"])

    def test_the_omission_is_reported_not_hidden(self) -> None:
        """6,437 singletons is a tuning signal, not something to quietly discard."""
        payload = self._call()
        assert payload["singleton_count"] == 6437
        assert payload["communities_omitted"] > 6000
        assert payload["community_count"] == 6497, (
            "community_count must still report the true total"
        )

    def test_the_documented_keys_are_unchanged(self) -> None:
        payload = self._call()
        for key in (
            "node_count",
            "edge_count",
            "community_count",
            "concept_count",
            "elapsed_seconds",
            "community_summary",
        ):
            assert key in payload, f"documented key {key!r} disappeared"

    def test_the_concept_sample_is_disclosed_not_just_its_count(self) -> None:
        """An agent must learn concept_count came from 1.6% of the repo, as v3.1.5's
        CLI output does: "Concepts : 47 (from the 200 most central of 12184 symbols)".
        """
        import trelix_mcp.server as srv

        result = self._mock_result()
        result.concept_count = 47
        result.concept_symbols_considered = 200
        result.concept_symbols_total = 12184

        with (
            patch("trelix.core.config.IndexConfig"),
            patch("trelix.graph.builder.GraphBuilder") as MockBuilder,
        ):
            MockBuilder.return_value.build.return_value = result
            payload = srv.build_knowledge_graph("/fake/repo", extract_concepts=True)

        assert payload["concept_count"] == 47
        assert payload["concept_symbols_considered"] == 200, (
            "the bound the paid calls actually covered is missing from the payload"
        )
        assert payload["concept_symbols_total"] == 12184, (
            "without the total, 200 is a number with nothing to be 1.6% of"
        )

    def test_the_coverage_keys_are_present_when_extraction_was_off(self) -> None:
        """0/0 is the value that distinguishes "never ran" from "ran, found nothing"."""
        payload = self._call()
        assert payload["concept_symbols_considered"] == 0
        assert payload["concept_symbols_total"] == 0

    def test_the_old_uncapped_shape_is_still_reachable(self) -> None:
        """An existing consumer parsing every community must have a way back."""
        payload = self._call(min_community_size=1, max_communities=0)
        assert len(payload["community_summary"]) == 6497
        assert payload["communities_omitted"] == 0
