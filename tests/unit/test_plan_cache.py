"""Unit tests for CachingPlanner.

Mirror tests/unit/test_embedder_cache.py exactly — same structure, same
coverage classes, adapted for QueryPlanner.plan() / QueryPlan objects.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from trelix.retrieval.plan_cache import CachingPlanner
from trelix.retrieval.planner.models import default_plan


def _make_mock_planner(query: str = "test query") -> MagicMock:
    """Return a mock QueryPlanner whose plan() returns a stable QueryPlan."""
    mock = MagicMock()
    mock.plan.return_value = default_plan(query)
    return mock


class TestCachingPlannerBasics:
    def test_second_call_returns_same_plan(self) -> None:
        mock = _make_mock_planner("how does auth work")
        cache = CachingPlanner(mock, max_size=128)

        p1 = cache.plan("how does auth work")
        p2 = cache.plan("how does auth work")

        assert p1 is p2  # same object returned from cache
        mock.plan.assert_called_once()  # underlying called only once

    def test_cache_key_normalisation(self) -> None:
        mock = _make_mock_planner()
        cache = CachingPlanner(mock, max_size=128)

        cache.plan("How Does Auth Work")
        cache.plan("  how does auth work  ")  # spaces + lower

        mock.plan.assert_called_once()  # same normalised key

    def test_different_queries_each_call_planner(self) -> None:
        mock = _make_mock_planner()
        cache = CachingPlanner(mock, max_size=128)

        cache.plan("query one")
        cache.plan("query two")

        assert mock.plan.call_count == 2


class TestCachingPlannerLRU:
    def test_lru_eviction_at_max_size(self) -> None:
        mock = _make_mock_planner()
        cache = CachingPlanner(mock, max_size=3)

        cache.plan("a")  # inserted: a         cache=[a]
        cache.plan("b")  # inserted: a, b       cache=[a,b]
        cache.plan("c")  # inserted: a, b, c    cache=[a,b,c] (full)
        cache.plan("d")  # a evicted (LRU)      cache=[b,c,d]

        assert mock.plan.call_count == 4

        # "a" was evicted — calling it again must hit the planner
        cache.plan("a")
        assert mock.plan.call_count == 5

        # "d" was not evicted (it was MRU when a,b,c,d were inserted)
        cache.plan("d")
        assert mock.plan.call_count == 5  # cache hit for d

    def test_zero_size_disables_cache(self) -> None:
        mock = _make_mock_planner()
        cache = CachingPlanner(mock, max_size=0)

        cache.plan("same query")
        cache.plan("same query")
        cache.plan("same query")

        assert mock.plan.call_count == 3  # every call goes to planner

    def test_negative_max_size_raises(self) -> None:
        mock = _make_mock_planner()
        with pytest.raises(ValueError, match="max_size must be >= 0"):
            CachingPlanner(mock, max_size=-1)


class TestCachingPlannerStats:
    def test_hit_miss_counts(self) -> None:
        mock = _make_mock_planner()
        cache = CachingPlanner(mock, max_size=128)

        cache.plan("first")  # miss
        cache.plan("second")  # miss
        cache.plan("first")  # hit
        cache.plan("first")  # hit

        assert cache.miss_count == 2
        assert cache.hit_count == 2

    def test_cache_size_property(self) -> None:
        mock = _make_mock_planner()
        cache = CachingPlanner(mock, max_size=128)

        assert cache.cache_size == 0
        cache.plan("one")
        assert cache.cache_size == 1
        cache.plan("two")
        assert cache.cache_size == 2

    def test_clear_resets_all(self) -> None:
        mock = _make_mock_planner()
        cache = CachingPlanner(mock, max_size=128)

        cache.plan("query")
        cache.plan("query")  # hit
        cache.clear()

        assert cache.cache_size == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0

        # After clear, the same query is a miss again
        cache.plan("query")
        assert cache.miss_count == 1
        assert mock.plan.call_count == 2  # called again after clear


class TestCachingPlannerThreadSafety:
    def test_concurrent_same_query_calls_planner_at_most_twice(self) -> None:
        """20 threads all plan("same") — planner called <= 2 times.

        We allow <= 2 because two threads may both see a miss before either
        stores the result (benign race: cache filled twice with same plan).
        What we forbid is 20 calls — that means the lock is broken.
        """
        mock = _make_mock_planner("same query")
        cache = CachingPlanner(mock, max_size=128)

        results: list = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                results.append(cache.plan("same query"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert not errors, f"Thread errors: {errors}"
        # At most 2 calls due to race on first insert; never 20
        assert mock.plan.call_count <= 2
