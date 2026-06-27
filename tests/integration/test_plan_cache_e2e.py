"""
Integration tests for the QueryPlan cache wired into Retriever.

No real LLM calls — QueryPlanner.plan() is replaced by a MagicMock with a
counter. Tests verify that Retriever.retrieve() calls the planner exactly
the right number of times depending on cache state.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

from trelix.core.config import IndexConfig, RetrievalConfig
from trelix.retrieval.planner.models import default_plan


def _make_config(tmp: str, plan_cache_size: int = 128) -> IndexConfig:
    return IndexConfig(
        repo_path=tmp,
        retrieval=RetrievalConfig(plan_cache_size=plan_cache_size),
    )


def _mock_retriever_deps(retriever: object) -> MagicMock:
    """Replace internal planner.plan() with a counter mock, fake vector/bm25."""
    from trelix.retrieval.plan_cache import CachingPlanner

    # Reach the raw planner (unwrap CachingPlanner if present)
    raw_planner = (
        retriever._planner._planner
        if isinstance(retriever._planner, CachingPlanner)
        else retriever._planner
    )
    mock_plan = MagicMock(return_value=default_plan("test"))
    raw_planner.plan = mock_plan

    # Stub out IO so retrieve() completes without DB/vector store
    retriever.vector_store.search = MagicMock(return_value=[])
    retriever.db.bm25_search = MagicMock(return_value=[])
    retriever.embedder.embed_query = MagicMock(return_value=[0.1] * 1536)

    return mock_plan


class TestPlanCacheE2E:
    def test_same_query_twice_calls_planner_once(self) -> None:
        """Same query twice -> planner called once when cache enabled."""
        from trelix.retrieval.retriever import Retriever

        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp, plan_cache_size=128)
            retriever = Retriever(config)
            mock_plan = _mock_retriever_deps(retriever)

            retriever.retrieve("how does authentication work")
            retriever.retrieve("how does authentication work")

            assert mock_plan.call_count == 1, (
                f"Expected 1 planner call (cache hit on second), got {mock_plan.call_count}"
            )

    def test_plan_cache_zero_disables(self) -> None:
        """Same query twice -> planner called twice when cache disabled."""
        from trelix.retrieval.retriever import Retriever

        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp, plan_cache_size=0)
            retriever = Retriever(config)
            mock_plan = _mock_retriever_deps(retriever)

            retriever.retrieve("how does authentication work")
            retriever.retrieve("how does authentication work")

            assert mock_plan.call_count == 2, (
                f"Expected 2 planner calls (cache disabled), got {mock_plan.call_count}"
            )

    def test_different_queries_each_call_planner(self) -> None:
        """Two different queries -> planner called twice even with cache enabled."""
        from trelix.retrieval.retriever import Retriever

        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp, plan_cache_size=128)
            retriever = Retriever(config)
            mock_plan = _mock_retriever_deps(retriever)

            retriever.retrieve("authentication")
            retriever.retrieve("database connection")

            assert mock_plan.call_count == 2, (
                f"Expected 2 planner calls (different queries), got {mock_plan.call_count}"
            )

    def test_external_plan_bypasses_cache(self) -> None:
        """Retriever.retrieve(query, plan=<external>) never touches the planner."""
        from trelix.retrieval.retriever import Retriever

        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp, plan_cache_size=128)
            retriever = Retriever(config)
            mock_plan = _mock_retriever_deps(retriever)

            external = default_plan("auth")
            retriever.retrieve("how does authentication work", plan=external)

            assert mock_plan.call_count == 0, (
                "External plan= path must not call the planner (cache or otherwise)"
            )
