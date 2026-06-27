"""Unit tests for CachingEmbedder."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from trelix.embedder.cache import CachingEmbedder


def _make_mock_embedder(vector: list[float] | None = None) -> MagicMock:
    """Return a mock BaseEmbedder whose embed_query returns a stable vector."""
    mock = MagicMock()
    mock.dimension = 4
    mock.embed_query.return_value = vector or [0.1, 0.2, 0.3, 0.4]
    mock.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
    return mock


class TestCachingEmbedderBasics:
    def test_second_call_returns_same_vector(self) -> None:
        mock = _make_mock_embedder()
        cache = CachingEmbedder(mock, max_size=256)

        v1 = cache.embed_query("how does auth work")
        v2 = cache.embed_query("how does auth work")

        assert v1 == v2
        mock.embed_query.assert_called_once()  # underlying called only once

    def test_cache_key_normalisation(self) -> None:
        mock = _make_mock_embedder()
        cache = CachingEmbedder(mock, max_size=256)

        cache.embed_query("How Does Auth Work")
        cache.embed_query("  how does auth work  ")  # spaces + lower

        mock.embed_query.assert_called_once()  # same normalised key

    def test_different_queries_each_call_api(self) -> None:
        mock = _make_mock_embedder()
        cache = CachingEmbedder(mock, max_size=256)

        cache.embed_query("query one")
        cache.embed_query("query two")

        assert mock.embed_query.call_count == 2

    def test_passthrough_embed_not_cached(self) -> None:
        mock = _make_mock_embedder()
        cache = CachingEmbedder(mock, max_size=256)

        cache.embed(["hello"])
        cache.embed(["hello"])  # called twice — no cache

        assert mock.embed.call_count == 2

    def test_dimension_passthrough(self) -> None:
        mock = _make_mock_embedder()
        mock.dimension = 1536
        cache = CachingEmbedder(mock, max_size=256)

        assert cache.dimension == 1536

    @pytest.mark.asyncio
    async def test_passthrough_embed_async_not_cached(self) -> None:
        from unittest.mock import AsyncMock
        mock = _make_mock_embedder()
        mock.embed_async = AsyncMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
        cache = CachingEmbedder(mock, max_size=256)

        await cache.embed_async(["hello"])
        await cache.embed_async(["hello"])

        assert mock.embed_async.call_count == 2


class TestCachingEmbedderLRU:
    def test_lru_eviction_at_max_size(self) -> None:
        mock = _make_mock_embedder()
        cache = CachingEmbedder(mock, max_size=3)

        cache.embed_query("a")  # inserted: a          cache=[a]
        cache.embed_query("b")  # inserted: a, b        cache=[a,b]
        cache.embed_query("c")  # inserted: a, b, c     cache=[a,b,c] (full)
        cache.embed_query("d")  # a evicted (LRU)       cache=[b,c,d]

        assert mock.embed_query.call_count == 4

        # "a" was evicted — calling it again must hit the API
        # Evicts b (LRU of {b,c,d}). cache=[c,d,a]
        cache.embed_query("a")
        assert mock.embed_query.call_count == 5

        # "d" was not evicted — it was the MRU when a,b,c,d were inserted
        # After a was re-inserted, cache=[c,d,a]; d is still present
        cache.embed_query("d")
        assert mock.embed_query.call_count == 5  # still 5 — cache hit for d

    def test_zero_size_disables_cache(self) -> None:
        mock = _make_mock_embedder()
        cache = CachingEmbedder(mock, max_size=0)

        cache.embed_query("same query")
        cache.embed_query("same query")
        cache.embed_query("same query")

        assert mock.embed_query.call_count == 3  # every call goes to API


class TestCachingEmbedderStats:
    def test_hit_miss_counts(self) -> None:
        mock = _make_mock_embedder()
        cache = CachingEmbedder(mock, max_size=256)

        cache.embed_query("first")   # miss
        cache.embed_query("second")  # miss
        cache.embed_query("first")   # hit
        cache.embed_query("first")   # hit

        assert cache.miss_count == 2
        assert cache.hit_count == 2

    def test_cache_size_property(self) -> None:
        mock = _make_mock_embedder()
        cache = CachingEmbedder(mock, max_size=256)

        assert cache.cache_size == 0
        cache.embed_query("one")
        assert cache.cache_size == 1
        cache.embed_query("two")
        assert cache.cache_size == 2

    def test_clear_resets_all(self) -> None:
        mock = _make_mock_embedder()
        cache = CachingEmbedder(mock, max_size=256)

        cache.embed_query("query")
        cache.embed_query("query")  # hit
        cache.clear()

        assert cache.cache_size == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0

        # After clear, the same query is a miss again
        cache.embed_query("query")
        assert cache.miss_count == 1
        assert mock.embed_query.call_count == 2  # called again after clear


class TestCachingEmbedderThreadSafety:
    def test_concurrent_same_query_calls_api_once(self) -> None:
        """20 threads all embed_query("same") — underlying must be called ≤ 2 times.

        We allow ≤ 2 because two threads may both see a miss before either
        stores the result (benign race: cache filled twice with same value).
        What we forbid is 20 calls — that means the lock is broken.
        """
        mock = _make_mock_embedder()
        cache = CachingEmbedder(mock, max_size=256)

        errors: list[Exception] = []

        def worker() -> None:
            try:
                cache.embed_query("same query for all threads")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        # At most 2 calls due to race on first insert; never 20
        assert mock.embed_query.call_count <= 2
