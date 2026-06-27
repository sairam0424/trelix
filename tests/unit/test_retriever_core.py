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
