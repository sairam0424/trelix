"""
Integration tests for the query embedding cache wired into Retriever.

These tests use mocked embedders — no real API calls. They verify that the
Retriever calls embed_query() exactly once for repeated queries when the cache
is enabled, and twice when it is disabled.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

from trelix.core.config import IndexConfig, RetrievalConfig


def _make_config(tmp: str, cache_size: int = 256) -> IndexConfig:
    return IndexConfig(
        repo_path=tmp,
        retrieval=RetrievalConfig(
            query_cache_size=cache_size,
            # Isolate from .env — prevent extra legs from inflating embed_query call counts
            file_summary_leg_enabled=False,
            hyde_fallback_enabled=False,
            flare_enabled=False,
            multi_query_enabled=False,
        ),
    )


def _mock_retriever_deps(retriever: object, vector: list[float]) -> MagicMock:
    """Patch the internal embedder's embed_query and fake vector search."""
    from trelix.embedder.cache import CachingEmbedder

    # If cache is enabled, the underlying embedder is inside CachingEmbedder
    raw = (
        retriever.embedder._embedder
        if isinstance(retriever.embedder, CachingEmbedder)
        else retriever.embedder
    )
    raw.embed_query = MagicMock(return_value=vector)
    # Patch vector store search to return empty (we only care about API call count)
    retriever.vector_store.search = MagicMock(return_value=[])
    retriever.db.bm25_search = MagicMock(return_value=[])
    return raw.embed_query


class TestQueryCacheE2E:
    def test_cache_reduces_api_calls(self) -> None:
        """Same query twice → embed_query called once when cache enabled."""
        from trelix.retrieval.retriever import Retriever

        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp, cache_size=256)
            retriever = Retriever(config)
            mock_embed = _mock_retriever_deps(retriever, [0.1] * 1536)

            retriever.retrieve("how does authentication work")
            retriever.retrieve("how does authentication work")

            assert mock_embed.call_count == 1, (
                f"Expected 1 API call (cache hit on second), got {mock_embed.call_count}"
            )

    def test_cache_size_zero_disables(self) -> None:
        """Same query twice → embed_query called twice when cache disabled."""
        from trelix.retrieval.retriever import Retriever

        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp, cache_size=0)
            retriever = Retriever(config)
            mock_embed = _mock_retriever_deps(retriever, [0.1] * 1536)

            retriever.retrieve("how does authentication work")
            retriever.retrieve("how does authentication work")

            assert mock_embed.call_count == 2, (
                f"Expected 2 API calls (cache disabled), got {mock_embed.call_count}"
            )

    def test_different_queries_both_call_api(self) -> None:
        """Two different queries → embed_query called twice even with cache enabled."""
        from trelix.retrieval.retriever import Retriever

        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp, cache_size=256)
            retriever = Retriever(config)
            mock_embed = _mock_retriever_deps(retriever, [0.1] * 1536)

            retriever.retrieve("authentication")
            retriever.retrieve("database connection")

            assert mock_embed.call_count == 2, (
                f"Expected 2 API calls (different queries), got {mock_embed.call_count}"
            )
