"""Unit tests for Retriever core paths.

Covers:
- __init__: builds successfully with valid IndexConfig
- retrieve(): returns RetrievedContext when plan is supplied externally
- Intent routing: FILE_OVERVIEW, PROJECT_OVERVIEW, CONFIG_LOOKUP, TIER_1_DIRECT
- _retrieve_standard: exercises the standard hybrid path
- Empty results fallback (project_overview with no DB symbols)
- _dedup: deduplication keeps highest score
- hydrate_symbol: returns None when DB returns None
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import IndexConfig
from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    RetrievedContext,
    SearchResult,
    Symbol,
    SymbolKind,
)
from trelix.retrieval.planner.models import (
    INTENT_STRATEGIES,
    IntentType,
    QueryPlan,
    RoutingTier,
    SubQuery,
)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_file(file_id: int = 1, rel_path: str = "src/foo.py") -> IndexedFile:
    return IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash=f"sha-{file_id}",
        size_bytes=1000,
        id=file_id,
        indexed_at=datetime(2024, 1, 1),
    )


def _make_symbol(sym_id: int = 1, file_id: int = 1, name: str = "my_func") -> Symbol:
    return Symbol(
        file_id=file_id,
        name=name,
        qualified_name=f"module.{name}",
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=10,
        signature=f"def {name}()",
        body=f"def {name}():\n    pass",
        id=sym_id,
    )


def _make_chunk(sym_id: int = 1, text: str = "def my_func(): pass") -> Chunk:
    return Chunk(
        symbol_id=sym_id,
        chunk_text=text,
        token_count=len(text.split()),
        id=sym_id,
    )


def _make_search_result(idx: int = 1, score: float = 0.9, source: str = "vector") -> SearchResult:
    file = _make_file(idx)
    sym = _make_symbol(idx, idx, name=f"func_{idx}")
    chunk = _make_chunk(idx, text=f"def func_{idx}(): pass")
    return SearchResult(chunk=chunk, symbol=sym, file=file, score=score, rank=idx, source=source)


def _make_plan(
    intent: IntentType = IntentType.FEATURE_FLOW,
    query: str = "how does authentication work?",
    routing_tier: RoutingTier = RoutingTier.TIER_2_SINGLE,
) -> QueryPlan:
    return QueryPlan(
        intent=intent,
        execution_mode="sequential",
        strategy=INTENT_STRATEGIES[intent],
        sub_queries=[
            SubQuery(
                semantic_query=query,
                hyde_snippet="",
                bm25_tokens=query.split(),
                grep_hints=[],
                file_hints=[],
            )
        ],
        raw_query=query,
        routing_tier=routing_tier,
    )


def _make_retrieved_context(
    query: str = "how does auth work?",
    num_results: int = 3,
) -> RetrievedContext:
    results = [_make_search_result(i) for i in range(1, num_results + 1)]
    return RetrievedContext(
        query=query,
        results=results,
        context_text="some context text",
        total_tokens=300,
        intent="feature_flow",
    )


def _make_retriever(tmp_path: str) -> object:
    """Build a Retriever with all heavy deps mocked out."""
    from trelix.retrieval.retriever import Retriever

    with (
        patch("trelix.retrieval.retriever.Database") as mock_db_cls,
        patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
        patch("trelix.retrieval.retriever.make_vector_store") as mock_make_vs,
        patch("trelix.retrieval.retriever.QueryPlanner") as mock_planner_cls,
        patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
    ):
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder.dimension = 1536
        mock_make_embedder.return_value = mock_embedder

        mock_vs = MagicMock()
        mock_make_vs.return_value = mock_vs

        mock_planner = MagicMock()
        mock_planner_cls.return_value = mock_planner

        config = IndexConfig(repo_path=tmp_path)
        retriever = Retriever(config)

    # Expose the mocks as attributes for test assertions
    retriever._db_mock = mock_db
    retriever._embedder_mock = mock_embedder
    retriever._vs_mock = mock_vs
    retriever._planner_mock = mock_planner

    return retriever


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetrieverInit:
    def test_init_sets_config_and_components(self, tmp_path: Path) -> None:  # type: ignore[name-defined]
        """Retriever.__init__ assigns config, db, embedder, vector_store, planner."""
        from trelix.retrieval.retriever import Retriever

        with (
            patch("trelix.retrieval.retriever.Database") as mock_db_cls,
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store") as mock_make_vs,
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder
            mock_make_vs.return_value = MagicMock()
            mock_db_cls.return_value = MagicMock()

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        assert retriever.config is config
        assert retriever.db is mock_db_cls.return_value
        assert retriever.embedder is mock_embedder
        assert retriever.vector_store is mock_make_vs.return_value

    def test_init_creates_debug_dir_path(self, tmp_path: Path) -> None:  # type: ignore[name-defined]
        """_debug_dir is set to <repo_path>/.trelix/debug."""
        from pathlib import Path

        from trelix.retrieval.retriever import Retriever

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        expected = Path(str(tmp_path)) / ".trelix" / "debug"
        assert retriever._debug_dir == expected


class TestRetrieveWithExternalPlan:
    def test_retrieve_returns_retrieved_context(self, tmp_path: Path) -> None:  # type: ignore[name-defined]
        """retrieve() with an external plan returns a RetrievedContext."""
        from trelix.retrieval.retriever import Retriever

        expected_ctx = _make_retrieved_context()

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        plan = _make_plan()

        with patch.object(retriever, "_execute_plan", return_value=expected_ctx) as mock_exec:
            result = retriever.retrieve("how does auth work?", plan=plan)

        mock_exec.assert_called_once_with(plan)
        assert isinstance(result, RetrievedContext)
        assert result.query == expected_ctx.query

    def test_retrieve_sets_elapsed_seconds(self, tmp_path: Path) -> None:  # type: ignore[name-defined]
        """retrieve() attaches elapsed_seconds to the returned context."""
        from trelix.retrieval.retriever import Retriever

        ctx = _make_retrieved_context()
        ctx.elapsed_seconds = 0.0  # will be overwritten

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        plan = _make_plan()
        with patch.object(retriever, "_execute_plan", return_value=ctx):
            result = retriever.retrieve("some query", plan=plan)

        assert result.elapsed_seconds >= 0.0

    def test_retrieve_calls_planner_when_no_plan_given(self, tmp_path: Path) -> None:  # type: ignore[name-defined]
        """retrieve() calls _planner.plan() when no external plan is supplied."""
        from trelix.retrieval.retriever import Retriever

        ctx = _make_retrieved_context()
        auto_plan = _make_plan()

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        retriever._planner.plan.return_value = auto_plan  # type: ignore[attr-defined]

        with patch.object(retriever, "_execute_plan", return_value=ctx):
            retriever.retrieve("how does auth work?")

        retriever._planner.plan.assert_called_once_with("how does auth work?")  # type: ignore[attr-defined]


class TestIntentRouting:
    def test_tier1_direct_routes_to_project_overview(self, tmp_path: Path) -> None:  # type: ignore[name-defined]
        """Tier 1 DIRECT routing tier calls _retrieve_project_overview."""
        from trelix.retrieval.retriever import Retriever

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        plan = _make_plan(routing_tier=RoutingTier.TIER_1_DIRECT)
        ctx = _make_retrieved_context()

        with patch.object(retriever, "_retrieve_project_overview", return_value=ctx) as mock_po:
            result = retriever._execute_plan(plan)

        mock_po.assert_called_once_with(plan)
        assert result is ctx

    def test_file_overview_intent_routes_correctly(self, tmp_path: Path) -> None:  # type: ignore[name-defined]
        """FILE_OVERVIEW intent calls _retrieve_file_overview."""
        from trelix.retrieval.retriever import Retriever

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        plan = _make_plan(intent=IntentType.FILE_OVERVIEW)
        ctx = _make_retrieved_context()

        with patch.object(retriever, "_retrieve_file_overview", return_value=ctx) as mock_fo:
            result = retriever._execute_plan(plan)

        mock_fo.assert_called_once_with(plan)
        assert result is ctx

    def test_project_overview_intent_routes_correctly(self, tmp_path: Path) -> None:  # type: ignore[name-defined]
        """PROJECT_OVERVIEW intent calls _retrieve_project_overview."""
        from trelix.retrieval.retriever import Retriever

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        plan = _make_plan(intent=IntentType.PROJECT_OVERVIEW)
        ctx = _make_retrieved_context()

        with patch.object(retriever, "_retrieve_project_overview", return_value=ctx) as mock_po:
            result = retriever._execute_plan(plan)

        mock_po.assert_called_once_with(plan)
        assert result is ctx

    def test_config_lookup_intent_routes_correctly(self, tmp_path: Path) -> None:  # type: ignore[name-defined]
        """CONFIG_LOOKUP intent calls _retrieve_config."""
        from trelix.retrieval.retriever import Retriever

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        plan = _make_plan(intent=IntentType.CONFIG_LOOKUP)
        ctx = _make_retrieved_context()

        with patch.object(retriever, "_retrieve_config", return_value=ctx) as mock_cfg:
            result = retriever._execute_plan(plan)

        mock_cfg.assert_called_once_with(plan)
        assert result is ctx

    def test_feature_flow_intent_routes_to_standard(self, tmp_path: Path) -> None:  # type: ignore[name-defined]
        """FEATURE_FLOW intent calls _retrieve_standard."""
        from trelix.retrieval.retriever import Retriever

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        plan = _make_plan(intent=IntentType.FEATURE_FLOW)
        ctx = _make_retrieved_context()

        with patch.object(retriever, "_retrieve_standard", return_value=ctx) as mock_std:
            result = retriever._execute_plan(plan)

        mock_std.assert_called_once_with(plan)
        assert result is ctx


class TestEmptyResults:
    def test_project_overview_falls_back_to_standard_when_empty(
        self,
        tmp_path: Path,  # type: ignore[name-defined]
    ) -> None:
        """_retrieve_project_overview falls back to _retrieve_standard when DB has no symbols."""
        from trelix.retrieval.retriever import Retriever

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        # DB returns no symbol IDs
        retriever.db.get_module_and_readme_symbols.return_value = []  # type: ignore[attr-defined]

        fallback_ctx = _make_retrieved_context()
        with patch.object(retriever, "_retrieve_standard", return_value=fallback_ctx) as mock_std:
            plan = _make_plan(intent=IntentType.PROJECT_OVERVIEW)
            result = retriever._retrieve_project_overview(plan)

        mock_std.assert_called_once()
        assert result is fallback_ctx

    def test_file_overview_falls_back_to_standard_when_no_file_matched(
        self,
        tmp_path: Path,  # type: ignore[name-defined]
    ) -> None:
        """_retrieve_file_overview falls back when no file matches the hint."""
        from trelix.retrieval.retriever import Retriever

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        retriever.db.find_file_by_path_fragment.return_value = []  # type: ignore[attr-defined]

        fallback_ctx = _make_retrieved_context()
        with patch.object(retriever, "_retrieve_standard", return_value=fallback_ctx) as mock_std:
            plan = QueryPlan(
                intent=IntentType.FILE_OVERVIEW,
                execution_mode="sequential",
                strategy=INTENT_STRATEGIES[IntentType.FILE_OVERVIEW],
                sub_queries=[
                    SubQuery(
                        semantic_query="tell me about auth.py",
                        hyde_snippet="",
                        bm25_tokens=["auth"],
                        grep_hints=[],
                        file_hints=["auth.py"],
                    )
                ],
                raw_query="tell me about auth.py",
            )
            result = retriever._retrieve_file_overview(plan)

        mock_std.assert_called_once()
        assert result is fallback_ctx


class TestDedup:
    def test_dedup_removes_duplicate_symbol_ids(self, tmp_path: Path) -> None:  # type: ignore[name-defined]
        """_dedup keeps highest-score result when symbol_id is repeated."""
        from trelix.retrieval.retriever import Retriever

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        # Two results with the SAME symbol_id (1) but different scores
        high_score = _make_search_result(idx=1, score=0.95, source="vector")
        low_score = _make_search_result(idx=1, score=0.50, source="bm25")
        other = _make_search_result(idx=2, score=0.80, source="grep")

        deduped = retriever._dedup([high_score, low_score, other])

        assert len(deduped) == 2
        sym_ids = {r.chunk.symbol_id for r in deduped}
        assert sym_ids == {1, 2}
        # The high-score duplicate was kept
        result_for_1 = next(r for r in deduped if r.chunk.symbol_id == 1)
        assert result_for_1.score == pytest.approx(0.95)

    def test_dedup_preserves_sort_order_by_score(self, tmp_path: Path) -> None:  # type: ignore[name-defined]
        """_dedup returns results sorted descending by score."""
        from trelix.retrieval.retriever import Retriever

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        r1 = _make_search_result(idx=1, score=0.30)
        r2 = _make_search_result(idx=2, score=0.90)
        r3 = _make_search_result(idx=3, score=0.60)

        deduped = retriever._dedup([r1, r2, r3])

        scores = [r.score for r in deduped]
        assert scores == sorted(scores, reverse=True)


class TestHydrateSymbol:
    def test_hydrate_symbol_returns_none_when_db_returns_none(
        self,
        tmp_path: Path,  # type: ignore[name-defined]
    ) -> None:
        """hydrate_symbol returns None when get_symbol_with_file returns None."""
        from trelix.retrieval.retriever import Retriever

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        retriever.db.get_symbol_with_file.return_value = None  # type: ignore[attr-defined]

        result = retriever.hydrate_symbol(symbol_id=99, score=0.9, rank=1, source="file_direct")
        assert result is None

    def test_hydrate_symbol_builds_search_result_from_db_row(
        self,
        tmp_path: Path,  # type: ignore[name-defined]
    ) -> None:
        """hydrate_symbol returns a SearchResult when the symbol exists in the DB."""
        from trelix.retrieval.retriever import Retriever

        with (
            patch("trelix.retrieval.retriever.Database"),
            patch("trelix.retrieval.retriever.make_embedder") as mock_make_embedder,
            patch("trelix.retrieval.retriever.make_vector_store"),
            patch("trelix.retrieval.retriever.QueryPlanner"),
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder-not-real"}),
        ):
            mock_embedder = MagicMock()
            mock_embedder.dimension = 1536
            mock_make_embedder.return_value = mock_embedder

            config = IndexConfig(repo_path=str(tmp_path))
            retriever = Retriever(config)

        sym = _make_symbol(sym_id=5, file_id=1)
        file = _make_file(file_id=1)
        chunk = _make_chunk(sym_id=5)

        retriever.db.get_symbol_with_file.return_value = (sym, file)  # type: ignore[attr-defined]
        retriever.db.get_first_chunk_for_symbol.return_value = chunk  # type: ignore[attr-defined]

        result = retriever.hydrate_symbol(symbol_id=5, score=0.75, rank=2, source="file_direct")

        assert isinstance(result, SearchResult)
        assert result.symbol is sym
        assert result.file is file
        assert result.chunk is chunk
        assert result.score == pytest.approx(0.75)
        assert result.rank == 2
        assert result.source == "file_direct"


# ---------------------------------------------------------------------------
# Helper: build a Retriever with mocked internals without entering context mgr
# ---------------------------------------------------------------------------


def _build_retriever(tmp_path):
    """Return a Retriever whose DB, embedder, vector_store, and planner are MagicMocks."""
    from trelix.retrieval.retriever import Retriever

    with (
        patch("trelix.retrieval.retriever.Database") as mock_db_cls,
        patch("trelix.retrieval.retriever.make_embedder") as mock_emb_cls,
        patch("trelix.retrieval.retriever.make_vector_store") as mock_vs_cls,
        patch("trelix.retrieval.retriever.QueryPlanner") as mock_planner_cls,
        patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder"}),
    ):
        mock_emb = MagicMock()
        mock_emb.dimension = 1536
        mock_emb_cls.return_value = mock_emb

        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs

        mock_planner = MagicMock()
        mock_planner_cls.return_value = mock_planner

        config = IndexConfig(repo_path=str(tmp_path))
        retriever = Retriever(config)

    # Stash mocks for assertions
    retriever._db_mock = mock_db
    retriever._emb_mock = mock_emb
    retriever._vs_mock = mock_vs
    retriever._planner_mock = mock_planner
    return retriever


# ---------------------------------------------------------------------------
# TestRetrieverIntentRouting — all 8 intent types + TIER_1_DIRECT
# ---------------------------------------------------------------------------


class TestRetrieverIntentRouting:
    """_execute_plan routes each intent to the right private method."""

    def _routing_test(self, tmp_path, intent: IntentType, expected_method: str):
        retriever = _build_retriever(str(tmp_path))
        plan = _make_plan(intent=intent)
        ctx = _make_retrieved_context()
        with patch.object(retriever, expected_method, return_value=ctx) as mock_method:
            result = retriever._execute_plan(plan)
        mock_method.assert_called_once_with(plan)
        assert result is ctx

    def test_symbol_lookup_routes_to_standard(self, tmp_path: Path) -> None:
        self._routing_test(tmp_path, IntentType.SYMBOL_LOOKUP, "_retrieve_standard")

    def test_comparison_routes_to_standard(self, tmp_path: Path) -> None:
        self._routing_test(tmp_path, IntentType.COMPARISON, "_retrieve_standard")

    def test_dependency_map_routes_to_standard(self, tmp_path: Path) -> None:
        self._routing_test(tmp_path, IntentType.DEPENDENCY_MAP, "_retrieve_standard")

    def test_blast_radius_routes_to_standard(self, tmp_path: Path) -> None:
        self._routing_test(tmp_path, IntentType.BLAST_RADIUS, "_retrieve_standard")

    def test_feature_flow_routes_to_standard(self, tmp_path: Path) -> None:
        self._routing_test(tmp_path, IntentType.FEATURE_FLOW, "_retrieve_standard")

    def test_file_overview_routes_to_file_overview(self, tmp_path: Path) -> None:
        self._routing_test(tmp_path, IntentType.FILE_OVERVIEW, "_retrieve_file_overview")

    def test_project_overview_routes_to_project_overview(self, tmp_path: Path) -> None:
        self._routing_test(tmp_path, IntentType.PROJECT_OVERVIEW, "_retrieve_project_overview")

    def test_config_lookup_routes_to_config(self, tmp_path: Path) -> None:
        self._routing_test(tmp_path, IntentType.CONFIG_LOOKUP, "_retrieve_config")

    def test_tier1_direct_routes_to_project_overview_override(self, tmp_path: Path) -> None:
        """Tier-1 plan with FEATURE_FLOW intent still routes to _retrieve_project_overview."""
        retriever = _build_retriever(str(tmp_path))
        # Use FEATURE_FLOW intent but TIER_1_DIRECT — tier check comes first
        plan = _make_plan(intent=IntentType.FEATURE_FLOW, routing_tier=RoutingTier.TIER_1_DIRECT)
        ctx = _make_retrieved_context()
        with patch.object(retriever, "_retrieve_project_overview", return_value=ctx) as mock_po:
            result = retriever._execute_plan(plan)
        mock_po.assert_called_once_with(plan)
        assert result is ctx


# ---------------------------------------------------------------------------
# TestRetrieverResultHydration — hydrate_symbol and _hydrate_chunk
# ---------------------------------------------------------------------------


class TestRetrieverResultHydration:
    """hydrate_symbol and _hydrate_chunk build correct SearchResult objects."""

    def test_hydrate_symbol_uses_synthetic_chunk_when_db_returns_none(self, tmp_path: Path) -> None:
        """When get_first_chunk_for_symbol returns None, a synthetic Chunk is built from body."""
        retriever = _build_retriever(str(tmp_path))
        sym = _make_symbol(sym_id=7, file_id=3, name="my_handler")
        file = _make_file(file_id=3)

        retriever.db.get_symbol_with_file.return_value = (sym, file)
        retriever.db.get_first_chunk_for_symbol.return_value = None  # force synthetic path

        result = retriever.hydrate_symbol(symbol_id=7, score=0.88, rank=1, source="file_direct")

        assert result is not None
        assert isinstance(result, SearchResult)
        assert result.chunk.symbol_id == 7
        assert result.chunk.chunk_text == sym.body[:2000]
        assert result.score == pytest.approx(0.88)

    def test_hydrate_chunk_returns_none_when_db_returns_none(self, tmp_path: Path) -> None:
        """_hydrate_chunk returns None when get_chunk_with_context returns None."""
        retriever = _build_retriever(str(tmp_path))
        retriever.db.get_chunk_with_context.return_value = None

        result = retriever._hydrate_chunk(chunk_id=42, score=0.5, rank=1, source="vector")
        assert result is None

    def test_hydrate_chunk_builds_search_result(self, tmp_path: Path) -> None:
        """_hydrate_chunk returns a fully populated SearchResult."""
        retriever = _build_retriever(str(tmp_path))
        sym = _make_symbol(sym_id=10, file_id=2)
        file = _make_file(file_id=2)
        chunk = _make_chunk(sym_id=10)

        retriever.db.get_chunk_with_context.return_value = (chunk, sym, file)

        result = retriever._hydrate_chunk(chunk_id=10, score=0.77, rank=3, source="vector")

        assert result is not None
        assert result.chunk is chunk
        assert result.symbol is sym
        assert result.file is file
        assert result.score == pytest.approx(0.77)
        assert result.rank == 3
        assert result.source == "vector"

    def test_retrieve_project_overview_returns_hydrated_results(self, tmp_path: Path) -> None:
        """_retrieve_project_overview hydrates symbol IDs from DB and assembles context."""
        retriever = _build_retriever(str(tmp_path))

        sym = _make_symbol(sym_id=1, file_id=1)
        file = _make_file(file_id=1)
        chunk = _make_chunk(sym_id=1)

        retriever.db.get_module_and_readme_symbols.return_value = [1, 2]
        retriever.db.get_symbol_with_file.return_value = (sym, file)
        retriever.db.get_first_chunk_for_symbol.return_value = chunk

        expected_ctx = _make_retrieved_context()
        with patch.object(retriever, "_assemble", return_value=expected_ctx) as mock_assemble:
            plan = _make_plan(intent=IntentType.PROJECT_OVERVIEW)
            result = retriever._retrieve_project_overview(plan)

        assert mock_assemble.called
        assert result is expected_ctx

    def test_retrieve_file_overview_returns_hydrated_results(self, tmp_path: Path) -> None:
        """_retrieve_file_overview finds file, hydrates all its symbols, and assembles."""
        retriever = _build_retriever(str(tmp_path))

        sym = _make_symbol(sym_id=1, file_id=1)
        file = _make_file(file_id=1)
        chunk = _make_chunk(sym_id=1)

        retriever.db.find_file_by_path_fragment.return_value = [1]
        retriever.db.get_all_symbols_for_file.return_value = [1, 2]
        retriever.db.get_symbol_with_file.return_value = (sym, file)
        retriever.db.get_first_chunk_for_symbol.return_value = chunk

        expected_ctx = _make_retrieved_context()
        with patch.object(retriever, "_assemble", return_value=expected_ctx) as mock_assemble:
            plan = QueryPlan(
                intent=IntentType.FILE_OVERVIEW,
                execution_mode="sequential",
                strategy=INTENT_STRATEGIES[IntentType.FILE_OVERVIEW],
                sub_queries=[
                    SubQuery(
                        semantic_query="show me auth.py",
                        hyde_snippet="",
                        bm25_tokens=["auth"],
                        grep_hints=[],
                        file_hints=["auth.py"],
                    )
                ],
                raw_query="show me auth.py",
            )
            result = retriever._retrieve_file_overview(plan)

        assert mock_assemble.called
        assert result is expected_ctx


# ---------------------------------------------------------------------------
# TestRetrieverCallGraphExpansion — _retrieve_standard triggers graph expansion
# ---------------------------------------------------------------------------


class TestRetrieverCallGraphExpansion:
    """_retrieve_standard runs graph expansion and merges results."""

    def _make_standard_plan(self, intent=IntentType.SYMBOL_LOOKUP, mode="sequential"):
        return QueryPlan(
            intent=intent,
            execution_mode=mode,
            strategy=INTENT_STRATEGIES[intent],
            sub_queries=[
                SubQuery(
                    semantic_query="find auth handler",
                    hyde_snippet="def authenticate(token): ...",
                    bm25_tokens=["auth", "handler"],
                    grep_hints=["authenticate"],
                    file_hints=[],
                )
            ],
            raw_query="find auth handler",
        )

    def test_retrieve_standard_runs_and_returns_context(self, tmp_path: Path) -> None:
        """_retrieve_standard completes without error and returns a RetrievedContext."""
        retriever = _build_retriever(str(tmp_path))

        sr = _make_search_result(idx=1, score=0.9)
        # Vector store returns one (chunk_id, distance) pair
        retriever.vector_store.search.return_value = [(1, 0.1)]
        retriever.db.get_chunk_with_context.return_value = (
            sr.chunk,
            sr.symbol,
            sr.file,
        )
        retriever.embedder.embed_query.return_value = [0.0] * 1536

        expected_ctx = _make_retrieved_context()

        with (
            patch("trelix.retrieval.retriever.bm25_search", return_value=[sr]),
            patch("trelix.retrieval.retriever.grep_search", return_value=[]),
            patch("trelix.retrieval.retriever.reciprocal_rank_fusion", return_value=[sr]),
            patch("trelix.retrieval.retriever.expand_with_call_graph", return_value=[]),
            patch("trelix.retrieval.retriever.expand_with_imports", return_value=[]),
            patch("trelix.retrieval.retriever.expand_with_type_edges", return_value=[]),
            patch.object(retriever, "_assemble", return_value=expected_ctx),
        ):
            plan = self._make_standard_plan()
            result = retriever._retrieve_standard(plan)

        assert result is expected_ctx

    def test_retrieve_standard_calls_expand_with_call_graph(self, tmp_path: Path) -> None:
        """_retrieve_standard always invokes expand_with_call_graph."""
        retriever = _build_retriever(str(tmp_path))

        sr = _make_search_result(idx=1, score=0.85)
        retriever.vector_store.search.return_value = [(1, 0.15)]
        retriever.db.get_chunk_with_context.return_value = (sr.chunk, sr.symbol, sr.file)
        retriever.embedder.embed_query.return_value = [0.0] * 1536

        expected_ctx = _make_retrieved_context()

        with (
            patch("trelix.retrieval.retriever.bm25_search", return_value=[]),
            patch("trelix.retrieval.retriever.grep_search", return_value=[]),
            patch("trelix.retrieval.retriever.reciprocal_rank_fusion", return_value=[sr]),
            patch("trelix.retrieval.retriever.expand_with_call_graph", return_value=[]) as mock_cg,
            patch("trelix.retrieval.retriever.expand_with_imports", return_value=[]),
            patch("trelix.retrieval.retriever.expand_with_type_edges", return_value=[]),
            patch.object(retriever, "_assemble", return_value=expected_ctx),
        ):
            plan = self._make_standard_plan()
            retriever._retrieve_standard(plan)

        mock_cg.assert_called_once()

    def test_retrieve_standard_parallel_mode(self, tmp_path: Path) -> None:
        """parallel execution_mode runs sub-queries via ThreadPoolExecutor."""
        retriever = _build_retriever(str(tmp_path))

        sr = _make_search_result(idx=1, score=0.8)
        retriever.vector_store.search.return_value = [(1, 0.2)]
        retriever.db.get_chunk_with_context.return_value = (sr.chunk, sr.symbol, sr.file)
        retriever.embedder.embed_query.return_value = [0.0] * 1536

        # Two sub-queries triggers parallel path
        plan = QueryPlan(
            intent=IntentType.FEATURE_FLOW,
            execution_mode="parallel",
            strategy=INTENT_STRATEGIES[IntentType.FEATURE_FLOW],
            sub_queries=[
                SubQuery(
                    semantic_query="auth flow",
                    hyde_snippet="",
                    bm25_tokens=["auth"],
                    grep_hints=[],
                    file_hints=[],
                ),
                SubQuery(
                    semantic_query="token validation",
                    hyde_snippet="",
                    bm25_tokens=["token"],
                    grep_hints=[],
                    file_hints=[],
                ),
            ],
            raw_query="auth and token flow",
        )

        expected_ctx = _make_retrieved_context()

        with (
            patch("trelix.retrieval.retriever.bm25_search", return_value=[sr]),
            patch("trelix.retrieval.retriever.grep_search", return_value=[]),
            patch("trelix.retrieval.retriever.reciprocal_rank_fusion", return_value=[sr]),
            patch("trelix.retrieval.retriever.expand_with_call_graph", return_value=[]),
            patch("trelix.retrieval.retriever.expand_with_imports", return_value=[]),
            patch("trelix.retrieval.retriever.expand_with_type_edges", return_value=[]),
            patch.object(retriever, "_assemble", return_value=expected_ctx),
        ):
            result = retriever._retrieve_standard(plan)

        assert result is expected_ctx

    def test_retrieve_standard_blast_radius_seeds_import_paths(self, tmp_path: Path) -> None:
        """BLAST_RADIUS with @ grep hints triggers seed_from_import_paths."""
        retriever = _build_retriever(str(tmp_path))

        sr = _make_search_result(idx=1, score=0.9)
        retriever.vector_store.search.return_value = [(1, 0.1)]
        retriever.db.get_chunk_with_context.return_value = (sr.chunk, sr.symbol, sr.file)
        retriever.embedder.embed_query.return_value = [0.0] * 1536

        plan = QueryPlan(
            intent=IntentType.BLAST_RADIUS,
            execution_mode="sequential",
            strategy=INTENT_STRATEGIES[IntentType.BLAST_RADIUS],
            sub_queries=[
                SubQuery(
                    semantic_query="what breaks if auth changes",
                    hyde_snippet="",
                    bm25_tokens=["auth"],
                    grep_hints=["@/lib/auth"],
                    file_hints=[],
                )
            ],
            raw_query="what breaks if auth changes",
        )

        expected_ctx = _make_retrieved_context()

        with (
            patch("trelix.retrieval.retriever.bm25_search", return_value=[]),
            patch("trelix.retrieval.retriever.grep_search", return_value=[sr]),
            patch("trelix.retrieval.retriever.reciprocal_rank_fusion", return_value=[sr]),
            patch("trelix.retrieval.retriever.expand_with_call_graph", return_value=[]),
            patch("trelix.retrieval.retriever.expand_with_imports", return_value=[]),
            patch("trelix.retrieval.retriever.expand_with_type_edges", return_value=[]),
            patch(
                "trelix.retrieval.retriever.seed_from_import_paths", return_value=[sr]
            ) as mock_seed,
            patch.object(retriever, "_assemble", return_value=expected_ctx),
        ):
            retriever._retrieve_standard(plan)

        mock_seed.assert_called_once()

    def test_retrieve_standard_skips_reranker_when_strategy_says_so(self, tmp_path: Path) -> None:
        """skip_reranker=True means the reranker module is never called."""
        from trelix.retrieval.planner.models import RetrievalStrategy

        retriever = _build_retriever(str(tmp_path))

        sr = _make_search_result(idx=1, score=0.9)
        retriever.vector_store.search.return_value = [(1, 0.1)]
        retriever.db.get_chunk_with_context.return_value = (sr.chunk, sr.symbol, sr.file)
        retriever.embedder.embed_query.return_value = [0.0] * 1536

        # Force rerank=True in config but skip_reranker=True in strategy
        retriever.config.retrieval.rerank = True

        skip_strategy = RetrievalStrategy(
            expand_depth=1,
            legs=["vector", "bm25"],
            skip_reranker=True,
            import_depth=1,
            import_max_extra=5,
            import_direction="both",
            assembly_mode="greedy",
            rerank_top_n=20,
        )

        plan = QueryPlan(
            intent=IntentType.SYMBOL_LOOKUP,
            execution_mode="sequential",
            strategy=skip_strategy,
            sub_queries=[
                SubQuery(
                    semantic_query="auth",
                    hyde_snippet="",
                    bm25_tokens=["auth"],
                    grep_hints=[],
                    file_hints=[],
                )
            ],
            raw_query="auth lookup",
        )

        expected_ctx = _make_retrieved_context()

        with (
            patch("trelix.retrieval.retriever.bm25_search", return_value=[sr]),
            patch("trelix.retrieval.retriever.grep_search", return_value=[]),
            patch("trelix.retrieval.retriever.reciprocal_rank_fusion", return_value=[sr]),
            patch("trelix.retrieval.retriever.expand_with_call_graph", return_value=[]),
            patch("trelix.retrieval.retriever.expand_with_imports", return_value=[]),
            patch("trelix.retrieval.retriever.expand_with_type_edges", return_value=[]),
            patch("trelix.retrieval.retriever.rerank") as mock_rerank,
            patch.object(retriever, "_assemble", return_value=expected_ctx),
        ):
            retriever._retrieve_standard(plan)

        mock_rerank.assert_not_called()

    def test_retrieve_standard_calls_reranker_when_enabled(self, tmp_path: Path) -> None:
        """rerank is called when cfg.rerank=True and skip_reranker=False."""
        from trelix.retrieval.planner.models import RetrievalStrategy

        retriever = _build_retriever(str(tmp_path))

        sr = _make_search_result(idx=1, score=0.9)
        retriever.vector_store.search.return_value = [(1, 0.1)]
        retriever.db.get_chunk_with_context.return_value = (sr.chunk, sr.symbol, sr.file)
        retriever.embedder.embed_query.return_value = [0.0] * 1536
        retriever.config.retrieval.rerank = True

        use_rerank_strategy = RetrievalStrategy(
            expand_depth=1,
            legs=["vector", "bm25"],
            skip_reranker=False,
            import_depth=1,
            import_max_extra=5,
            import_direction="both",
            assembly_mode="greedy",
            rerank_top_n=20,
        )

        plan = QueryPlan(
            intent=IntentType.FEATURE_FLOW,
            execution_mode="sequential",
            strategy=use_rerank_strategy,
            sub_queries=[
                SubQuery(
                    semantic_query="auth flow",
                    hyde_snippet="",
                    bm25_tokens=["auth"],
                    grep_hints=[],
                    file_hints=[],
                )
            ],
            raw_query="auth flow",
        )

        expected_ctx = _make_retrieved_context()

        with (
            patch("trelix.retrieval.retriever.bm25_search", return_value=[sr]),
            patch("trelix.retrieval.retriever.grep_search", return_value=[]),
            patch("trelix.retrieval.retriever.reciprocal_rank_fusion", return_value=[sr]),
            patch("trelix.retrieval.retriever.expand_with_call_graph", return_value=[]),
            patch("trelix.retrieval.retriever.expand_with_imports", return_value=[]),
            patch("trelix.retrieval.retriever.expand_with_type_edges", return_value=[]),
            patch("trelix.retrieval.retriever.rerank", return_value=[sr]) as mock_rerank,
            patch.object(retriever, "_assemble", return_value=expected_ctx),
        ):
            retriever._retrieve_standard(plan)

        mock_rerank.assert_called_once()


# ---------------------------------------------------------------------------
# TestRetrieverConfigLookup — _retrieve_config paths
# ---------------------------------------------------------------------------


class TestRetrieverConfigLookup:
    """_retrieve_config finds config files by hint and assembles results."""

    def test_config_lookup_with_matching_hint(self, tmp_path: Path) -> None:
        """_retrieve_config hydrates symbols when a config file is matched."""
        retriever = _build_retriever(str(tmp_path))

        sym = _make_symbol(sym_id=1, file_id=1)
        file = _make_file(file_id=1)
        chunk = _make_chunk(sym_id=1)

        retriever.db.find_file_by_path_fragment.return_value = [1]
        retriever.db.get_all_symbols_for_file.return_value = [1]
        retriever.db.get_symbol_with_file.return_value = (sym, file)
        retriever.db.get_first_chunk_for_symbol.return_value = chunk

        expected_ctx = _make_retrieved_context()

        plan = QueryPlan(
            intent=IntentType.CONFIG_LOOKUP,
            execution_mode="sequential",
            strategy=INTENT_STRATEGIES[IntentType.CONFIG_LOOKUP],
            sub_queries=[
                SubQuery(
                    semantic_query="jest configuration",
                    hyde_snippet="",
                    bm25_tokens=["jest"],
                    grep_hints=["jest.config.js"],
                    file_hints=["jest.config.js"],
                )
            ],
            raw_query="what is in jest.config",
        )

        with patch.object(retriever, "_assemble", return_value=expected_ctx) as mock_assemble:
            result = retriever._retrieve_config(plan)

        assert mock_assemble.called
        assert result is expected_ctx

    def test_config_lookup_falls_back_to_standard_when_no_match(self, tmp_path: Path) -> None:
        """_retrieve_config falls back to _retrieve_standard when no config file matched."""
        retriever = _build_retriever(str(tmp_path))
        retriever.db.find_file_by_path_fragment.return_value = []

        fallback_ctx = _make_retrieved_context()
        plan = QueryPlan(
            intent=IntentType.CONFIG_LOOKUP,
            execution_mode="sequential",
            strategy=INTENT_STRATEGIES[IntentType.CONFIG_LOOKUP],
            sub_queries=[
                SubQuery(
                    semantic_query="jest config",
                    hyde_snippet="",
                    bm25_tokens=["jest"],
                    grep_hints=["jest.config.js"],
                    file_hints=["jest.config.js"],
                )
            ],
            raw_query="what is in jest.config",
        )

        with patch.object(retriever, "_retrieve_standard", return_value=fallback_ctx) as mock_std:
            result = retriever._retrieve_config(plan)

        mock_std.assert_called_once()
        assert result is fallback_ctx

    def test_config_lookup_matches_hint_with_config_keyword(self, tmp_path: Path) -> None:
        """Hints containing 'config' (no extension) also trigger file lookup."""
        retriever = _build_retriever(str(tmp_path))

        sym = _make_symbol(sym_id=2, file_id=2)
        file = _make_file(file_id=2)
        chunk = _make_chunk(sym_id=2)

        retriever.db.find_file_by_path_fragment.return_value = [2]
        retriever.db.get_all_symbols_for_file.return_value = [2]
        retriever.db.get_symbol_with_file.return_value = (sym, file)
        retriever.db.get_first_chunk_for_symbol.return_value = chunk

        expected_ctx = _make_retrieved_context()

        plan = QueryPlan(
            intent=IntentType.CONFIG_LOOKUP,
            execution_mode="sequential",
            strategy=INTENT_STRATEGIES[IntentType.CONFIG_LOOKUP],
            sub_queries=[
                SubQuery(
                    semantic_query="app configuration",
                    hyde_snippet="",
                    bm25_tokens=["config"],
                    grep_hints=["app-config"],
                    file_hints=[],
                )
            ],
            raw_query="show me app config",
        )

        with patch.object(retriever, "_assemble", return_value=expected_ctx):
            result = retriever._retrieve_config(plan)

        assert result is expected_ctx


# ---------------------------------------------------------------------------
# TestRetrieverRunSubqueryLegs — _run_subquery_legs
# ---------------------------------------------------------------------------


class TestRetrieverRunSubqueryLegs:
    """_run_subquery_legs executes each enabled leg and collects results."""

    def test_vector_leg_runs_when_in_strategy(self, tmp_path: Path) -> None:
        """When 'vector' is in strategy.legs, vector_store.search is called."""
        retriever = _build_retriever(str(tmp_path))

        sr = _make_search_result(idx=1, score=0.9)
        retriever.vector_store.search.return_value = [(1, 0.1)]
        retriever.db.get_chunk_with_context.return_value = (sr.chunk, sr.symbol, sr.file)
        retriever.embedder.embed_query.return_value = [0.0] * 1536

        from trelix.retrieval.planner.models import RetrievalStrategy

        strategy = RetrievalStrategy(
            expand_depth=1,
            legs=["vector"],
            skip_reranker=False,
            import_depth=1,
            import_max_extra=5,
            import_direction="both",
            assembly_mode="greedy",
            rerank_top_n=20,
        )
        sq = SubQuery(
            semantic_query="auth handler",
            hyde_snippet="def authenticate(): ...",
            bm25_tokens=["auth"],
            grep_hints=[],
            file_hints=[],
        )

        out = retriever._run_subquery_legs(sq, strategy)

        retriever.embedder.embed_query.assert_called_once()
        assert len(out["vector"]) == 1

    def test_bm25_leg_runs_when_in_strategy(self, tmp_path: Path) -> None:
        """When 'bm25' is in strategy.legs, bm25_search is called."""
        retriever = _build_retriever(str(tmp_path))

        sr = _make_search_result(idx=2, score=0.8, source="bm25")

        from trelix.retrieval.planner.models import RetrievalStrategy

        strategy = RetrievalStrategy(
            expand_depth=0,
            legs=["bm25"],
            skip_reranker=False,
            import_depth=0,
            import_max_extra=0,
            import_direction="both",
            assembly_mode="greedy",
            rerank_top_n=10,
        )
        sq = SubQuery(
            semantic_query="token validation",
            hyde_snippet="",
            bm25_tokens=["token", "validate"],
            grep_hints=[],
            file_hints=[],
        )

        with patch("trelix.retrieval.retriever.bm25_search", return_value=[sr]) as mock_bm25:
            out = retriever._run_subquery_legs(sq, strategy)

        mock_bm25.assert_called_once()
        assert len(out["bm25"]) == 1

    def test_grep_leg_runs_when_in_strategy(self, tmp_path: Path) -> None:
        """When 'grep' is in strategy.legs, grep_search is called for each hint."""
        retriever = _build_retriever(str(tmp_path))

        sr = _make_search_result(idx=3, score=0.7, source="grep")

        from trelix.retrieval.planner.models import RetrievalStrategy

        strategy = RetrievalStrategy(
            expand_depth=0,
            legs=["grep"],
            skip_reranker=False,
            import_depth=0,
            import_max_extra=0,
            import_direction="both",
            assembly_mode="greedy",
            rerank_top_n=10,
        )
        sq = SubQuery(
            semantic_query="authenticate",
            hyde_snippet="",
            bm25_tokens=[],
            grep_hints=["authenticate", "login"],
            file_hints=[],
        )

        with patch("trelix.retrieval.retriever.grep_search", return_value=[sr]) as mock_grep:
            out = retriever._run_subquery_legs(sq, strategy)

        assert mock_grep.call_count == 2  # called once per hint
        assert len(out["grep"]) == 2

    def test_legs_not_in_strategy_are_skipped(self, tmp_path: Path) -> None:
        """Legs not listed in strategy.legs produce empty lists."""
        retriever = _build_retriever(str(tmp_path))

        from trelix.retrieval.planner.models import RetrievalStrategy

        # Only bm25 — no vector, no grep
        strategy = RetrievalStrategy(
            expand_depth=0,
            legs=["bm25"],
            skip_reranker=False,
            import_depth=0,
            import_max_extra=0,
            import_direction="both",
            assembly_mode="greedy",
            rerank_top_n=10,
        )
        sq = SubQuery(
            semantic_query="some query",
            hyde_snippet="",
            bm25_tokens=["some"],
            grep_hints=[],
            file_hints=[],
        )

        with patch("trelix.retrieval.retriever.bm25_search", return_value=[]):
            out = retriever._run_subquery_legs(sq, strategy)

        assert out["vector"] == []
        assert out["grep"] == []
        retriever.embedder.embed_query.assert_not_called()

    def test_hyde_snippet_used_as_embed_text(self, tmp_path: Path) -> None:
        """When hyde_snippet is non-empty, it is embedded rather than semantic_query."""
        retriever = _build_retriever(str(tmp_path))

        retriever.vector_store.search.return_value = []
        retriever.embedder.embed_query.return_value = [0.0] * 1536

        from trelix.retrieval.planner.models import RetrievalStrategy

        strategy = RetrievalStrategy(
            expand_depth=0,
            legs=["vector"],
            skip_reranker=False,
            import_depth=0,
            import_max_extra=0,
            import_direction="both",
            assembly_mode="greedy",
            rerank_top_n=10,
        )
        hyde = "def authenticate(token: str) -> bool: ..."
        sq = SubQuery(
            semantic_query="auth function",
            hyde_snippet=hyde,
            bm25_tokens=["auth"],
            grep_hints=[],
            file_hints=[],
        )

        retriever._run_subquery_legs(sq, strategy)

        retriever.embedder.embed_query.assert_called_once_with(hyde)

    def test_semantic_query_used_when_hyde_is_empty(self, tmp_path: Path) -> None:
        """When hyde_snippet is blank, semantic_query is embedded."""
        retriever = _build_retriever(str(tmp_path))

        retriever.vector_store.search.return_value = []
        retriever.embedder.embed_query.return_value = [0.0] * 1536

        from trelix.retrieval.planner.models import RetrievalStrategy

        strategy = RetrievalStrategy(
            expand_depth=0,
            legs=["vector"],
            skip_reranker=False,
            import_depth=0,
            import_max_extra=0,
            import_direction="both",
            assembly_mode="greedy",
            rerank_top_n=10,
        )
        sq = SubQuery(
            semantic_query="auth function",
            hyde_snippet="   ",  # whitespace only — treated as empty
            bm25_tokens=["auth"],
            grep_hints=[],
            file_hints=[],
        )

        retriever._run_subquery_legs(sq, strategy)

        retriever.embedder.embed_query.assert_called_once_with("auth function")


# ---------------------------------------------------------------------------
# TestRetrieverVectorSearch — _vector_search
# ---------------------------------------------------------------------------


class TestRetrieverVectorSearch:
    """_vector_search converts (chunk_id, distance) pairs to SearchResult objects."""

    def test_vector_search_returns_results(self, tmp_path: Path) -> None:
        """_vector_search returns a SearchResult for each hit from the vector store."""
        retriever = _build_retriever(str(tmp_path))

        sr = _make_search_result(idx=1, score=0.9)
        retriever.vector_store.search.return_value = [(1, 0.1)]
        retriever.db.get_chunk_with_context.return_value = (sr.chunk, sr.symbol, sr.file)

        results = retriever._vector_search([0.0] * 1536, k=5)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].score == pytest.approx(0.9)
        assert results[0].source == "vector"

    def test_vector_search_skips_missing_chunks(self, tmp_path: Path) -> None:
        """When get_chunk_with_context returns None, that result is skipped."""
        retriever = _build_retriever(str(tmp_path))

        retriever.vector_store.search.return_value = [(99, 0.2)]
        retriever.db.get_chunk_with_context.return_value = None

        results = retriever._vector_search([0.0] * 1536, k=5)
        assert results == []

    def test_vector_search_score_clamps_to_zero(self, tmp_path: Path) -> None:
        """Distance > 1 produces score clamped at 0.0, not negative."""
        retriever = _build_retriever(str(tmp_path))

        sr = _make_search_result(idx=1, score=0.0)
        retriever.vector_store.search.return_value = [(1, 1.5)]  # distance > 1 → score < 0
        retriever.db.get_chunk_with_context.return_value = (sr.chunk, sr.symbol, sr.file)

        results = retriever._vector_search([0.0] * 1536, k=5)
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestRetrieverFileOverviewGrepHintAsFilename — file hint from grep_hints
# ---------------------------------------------------------------------------


class TestRetrieverFileOverviewGrepHintAsFilename:
    """grep_hints that look like filenames (contain a dot) are treated as file hints."""

    def test_grep_hint_with_dot_treated_as_file_hint(self, tmp_path: Path) -> None:
        """A grep_hint like 'models.py' is used to find files in _retrieve_file_overview."""
        retriever = _build_retriever(str(tmp_path))

        sym = _make_symbol(sym_id=1, file_id=1)
        file = _make_file(file_id=1)
        chunk = _make_chunk(sym_id=1)

        retriever.db.find_file_by_path_fragment.return_value = [1]
        retriever.db.get_all_symbols_for_file.return_value = [1]
        retriever.db.get_symbol_with_file.return_value = (sym, file)
        retriever.db.get_first_chunk_for_symbol.return_value = chunk

        expected_ctx = _make_retrieved_context()

        plan = QueryPlan(
            intent=IntentType.FILE_OVERVIEW,
            execution_mode="sequential",
            strategy=INTENT_STRATEGIES[IntentType.FILE_OVERVIEW],
            sub_queries=[
                SubQuery(
                    semantic_query="show me models.py",
                    hyde_snippet="",
                    bm25_tokens=["models"],
                    grep_hints=["models.py"],  # dot-containing hint — treated as filename
                    file_hints=[],
                )
            ],
            raw_query="show me models.py",
        )

        with patch.object(retriever, "_assemble", return_value=expected_ctx):
            result = retriever._retrieve_file_overview(plan)

        retriever.db.find_file_by_path_fragment.assert_called()
        assert result is expected_ctx


# ---------------------------------------------------------------------------
# TestRetrieverTracing — _trace and _flush_trace
# ---------------------------------------------------------------------------


class TestRetrieverTracing:
    """_trace stores sections; _flush_trace writes a JSON file."""

    def test_trace_stores_section_data(self, tmp_path: Path) -> None:
        """_trace writes a named section into _trace_local.data."""
        from trelix.retrieval.retriever import _trace_local

        retriever = _build_retriever(str(tmp_path))
        _trace_local.data = {}

        retriever._trace("my_section", {"key": "value"})

        assert _trace_local.data["my_section"] == {"key": "value"}

    def test_trace_outside_retrieve_does_not_raise(self, tmp_path: Path) -> None:
        """_trace called without initialising _trace_local.data silently does nothing."""
        from trelix.retrieval.retriever import _trace_local

        retriever = _build_retriever(str(tmp_path))
        # Remove the attribute to simulate the AttributeError path
        if hasattr(_trace_local, "data"):
            del _trace_local.data

        retriever._trace("section", {"x": 1})  # must not raise

    def test_flush_trace_writes_json_file(self, tmp_path: Path) -> None:
        """_flush_trace creates a JSON file in _debug_dir."""
        from trelix.retrieval.retriever import _trace_local

        retriever = _build_retriever(str(tmp_path))
        _trace_local.data = {
            "query": "how does auth work",
            "ts": "2026-01-01T10:00:00",
        }

        retriever._flush_trace()

        debug_files = list(retriever._debug_dir.glob("*.json"))
        assert len(debug_files) == 1
        content = debug_files[0].read_text()
        assert "how does auth work" in content

    def test_flush_trace_silently_ignores_errors(self, tmp_path: Path) -> None:
        """_flush_trace never raises even if writing fails."""
        from trelix.retrieval.retriever import _trace_local

        retriever = _build_retriever(str(tmp_path))
        _trace_local.data = {
            "query": "test",
            "ts": "2026-01-01T10:00:00",
        }
        # Make _debug_dir a file so mkdir fails
        retriever._debug_dir.parent.mkdir(parents=True, exist_ok=True)
        retriever._debug_dir.touch()  # block directory creation

        retriever._flush_trace()  # must not raise
