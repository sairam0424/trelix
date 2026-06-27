"""
Integration tests for Retriever.get_callers / get_callees / get_importers.

Indexes the trelix repo itself using provider=local (no API keys).
The fixture is module-scoped so indexing runs only once for all tests.

Runtime: ~30-60 s on first run (depends on machine).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trelix.core.config import EmbedderConfig, IndexConfig
from trelix.indexing.indexer import Indexer
from trelix.retrieval.retriever import Retriever

# ---------------------------------------------------------------------------
# Module-scoped fixture: index trelix itself once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def retriever(tmp_path_factory: pytest.TempPathFactory) -> Retriever:
    """
    Index the trelix repo itself into a fresh temp dir and return a Retriever.

    Using a temp repo_path copy is expensive; instead we point repo_path at the
    worktree root and place the db in a fresh temp dir via StoreConfig so we
    always get a clean index (no "already up-to-date" short-circuit).
    """
    from trelix.core.config import StoreConfig

    repo = Path(__file__).parent.parent.parent  # trelix repo root
    db_dir = tmp_path_factory.mktemp("trelix_self_graph_index")
    config = IndexConfig(
        repo_path=str(repo),
        incremental=False,
        embedder=EmbedderConfig(provider="local"),
        store=StoreConfig(db_path=str(db_dir / "trelix.db")),
    )
    Indexer(config).index()
    return Retriever(config)


# ---------------------------------------------------------------------------
# get_callers
# ---------------------------------------------------------------------------


class TestGetCallersIntegration:
    def test_callers_of_retrieve_are_non_empty(self, retriever: Retriever) -> None:
        """
        Retriever.retrieve is called by cli/main.py::query and by integration
        tests — there must be at least one resolved internal caller.
        """
        results = retriever.get_callers("Retriever.retrieve")
        assert len(results) > 0, (
            "Expected at least one internal caller of Retriever.retrieve. "
            "Check that call edges were indexed (db.get_callers returned empty)."
        )

    def test_callers_source_is_graph_callers(self, retriever: Retriever) -> None:
        results = retriever.get_callers("Retriever.retrieve")
        assert all(r.source == "graph_callers" for r in results)

    def test_callers_score_is_one(self, retriever: Retriever) -> None:
        results = retriever.get_callers("Retriever.retrieve")
        assert all(r.score == 1.0 for r in results)

    def test_callers_rank_is_one_indexed_contiguous(self, retriever: Retriever) -> None:
        results = retriever.get_callers("Retriever.retrieve")
        assert results  # guard
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_callers_nonexistent_symbol_returns_empty(self, retriever: Retriever) -> None:
        results = retriever.get_callers("__this_symbol_does_not_exist_anywhere__")
        assert results == []

    def test_callers_sorted_by_file_then_line(self, retriever: Retriever) -> None:
        results = retriever.get_callers("Retriever.retrieve")
        if len(results) < 2:
            pytest.skip("Need at least 2 callers to check sort order")
        keys = [(r.file.rel_path, r.symbol.line_start) for r in results]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# get_callees
# ---------------------------------------------------------------------------


class TestGetCalleesIntegration:
    def test_callees_of_retrieve_are_non_empty(self, retriever: Retriever) -> None:
        """
        Retriever.retrieve calls multiple internal helpers (_run_vector, BM25,
        graph expansion, etc.) — at least one should resolve internally.
        """
        results = retriever.get_callees("Retriever.retrieve")
        assert len(results) > 0, (
            "Expected at least one resolved internal callee of Retriever.retrieve."
        )

    def test_callees_source_is_graph_callees(self, retriever: Retriever) -> None:
        results = retriever.get_callees("Retriever.retrieve")
        assert all(r.source == "graph_callees" for r in results)

    def test_callees_score_is_one(self, retriever: Retriever) -> None:
        results = retriever.get_callees("Retriever.retrieve")
        assert all(r.score == 1.0 for r in results)

    def test_callees_nonexistent_symbol_returns_empty(self, retriever: Retriever) -> None:
        results = retriever.get_callees("__this_symbol_does_not_exist_anywhere__")
        assert results == []


# ---------------------------------------------------------------------------
# get_importers
# ---------------------------------------------------------------------------


class TestGetImportersIntegration:
    def test_importers_of_retriever_module_non_empty(self, retriever: Retriever) -> None:
        """
        retrieval/retriever.py is imported by cli/main.py and by integration
        tests — there must be at least one importer.
        """
        results = retriever.get_importers("retrieval/retriever.py")
        assert len(results) > 0, (
            "Expected at least one file importing retrieval/retriever.py. "
            "Check that import edges were resolved during indexing."
        )

    def test_importers_source_is_graph_importers(self, retriever: Retriever) -> None:
        results = retriever.get_importers("retrieval/retriever.py")
        assert all(r.source == "graph_importers" for r in results)

    def test_importers_score_is_one(self, retriever: Retriever) -> None:
        results = retriever.get_importers("retrieval/retriever.py")
        assert all(r.score == 1.0 for r in results)

    def test_importers_unknown_module_returns_empty(self, retriever: Retriever) -> None:
        results = retriever.get_importers("__no_such_module_anywhere__")
        assert results == []

    def test_importers_sorted_by_rel_path(self, retriever: Retriever) -> None:
        results = retriever.get_importers("retrieval/retriever.py")
        if len(results) < 2:
            pytest.skip("Need at least 2 importers to check sort order")
        paths = [r.file.rel_path for r in results]
        assert paths == sorted(paths)

    def test_importers_rank_is_one_indexed_contiguous(self, retriever: Retriever) -> None:
        results = retriever.get_importers("retrieval/retriever.py")
        assert results  # guard
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))
