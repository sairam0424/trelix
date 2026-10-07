"""`CachedIndexEmbedder` from `src/trelix/indexing/embedding_cache.py`: the wrapper that answers
`embed`/`embed_async` from an `EmbeddingCache` first.

Split from `test_embedding_cache.py` (fingerprint, directory, file) to keep each test file under
500 lines; same harness: the cache file lives under `tmp_path` and the inner embedder is
`CountingEmbedder` from `tests/fixtures/fakes.py` (float32-exact vectors, so a cache hit compares
equal to a miss). The Indexer wiring is in `test_embedding_cache_indexer.py`.

Each test names the mutation(s) of the module that must fail it.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from tests.fixtures.fakes import CountingEmbedder
from trelix.indexing.embedding_cache import CachedIndexEmbedder, EmbeddingCache, prepare_cache_dir


def _key(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _open(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache.open(prepare_cache_dir(tmp_path / "cache") / "f.db", dimension=4)


def _wrapped(tmp_path: Path) -> tuple[CachedIndexEmbedder, CountingEmbedder, EmbeddingCache]:
    inner = CountingEmbedder(dimension=4)
    cache = _open(tmp_path)
    return CachedIndexEmbedder(inner, cache), inner, cache


class TestCachedIndexEmbedder:
    def test_embed_serves_hits_dedupes_misses_and_keeps_order(self, tmp_path: Path) -> None:
        """MUTATION: misses not de-duplicated (inner receives ["x", "x"]); merge order wrong;
        everything re-embedded (inner call count); hits counted per distinct text; the row
        key not the documented `sha256(chunk_text)` (`digest()[:16]`, say: the cache stays
        self-consistent, but SECURITY.md's key and any external lookup of it break)."""
        wrapped, inner, cache = _wrapped(tmp_path)
        reference = CountingEmbedder(dimension=4)

        assert wrapped.embed(["a", "b"]) == reference.embed(["a", "b"])
        assert inner.calls == [["a", "b"]]
        assert (cache.hits, cache.misses) == (0, 2)
        assert set(cache.get_many([_key("a"), _key("b")])) == {_key("a"), _key("b")}

        assert wrapped.embed(["b", "c", "a"]) == reference.embed(["b", "c", "a"])
        assert inner.calls == [["a", "b"], ["c"]]
        assert (cache.hits, cache.misses) == (2, 3)

        out = wrapped.embed(["x", "x"])
        assert inner.calls == [["a", "b"], ["c"], ["x"]]
        assert out[0] == out[1] == reference.embed(["x"])[0]
        assert (cache.hits, cache.misses) == (2, 4)

        assert wrapped.embed(["x", "x"]) == out
        assert inner.calls == [["a", "b"], ["c"], ["x"]]
        assert (cache.hits, cache.misses) == (4, 4)
        cache.close()

    def test_embed_async_behaves_the_same(self, tmp_path: Path) -> None:
        """MUTATION: embed_async bypasses the cache (calls the inner for everything)."""
        wrapped, inner, cache = _wrapped(tmp_path)
        reference = CountingEmbedder(dimension=4)

        assert asyncio.run(wrapped.embed_async(["a", "b"])) == reference.embed(["a", "b"])
        assert asyncio.run(wrapped.embed_async(["b", "c", "a"])) == reference.embed(["b", "c", "a"])
        out = asyncio.run(wrapped.embed_async(["x", "x"]))
        assert out[0] == out[1]
        assert inner.calls == [["a", "b"], ["c"], ["x"]]
        assert (cache.hits, cache.misses) == (2, 4)
        cache.close()

    def test_sync_and_async_share_one_cache(self, tmp_path: Path) -> None:
        """MUTATION: one of the two paths keyed differently (a sync miss is an async miss)."""
        wrapped, inner, cache = _wrapped(tmp_path)
        wrapped.embed(["a"])
        asyncio.run(wrapped.embed_async(["a"]))
        assert inner.calls == [["a"]]
        cache.close()

    def test_embed_query_passes_through_uncached(self, tmp_path: Path) -> None:
        """MUTATION: embed_query routed through the cache (row_count grows)."""
        wrapped, inner, cache = _wrapped(tmp_path)
        assert wrapped.embed_query("q") == inner.embed_query("q")
        assert cache.row_count() == 0
        assert inner.calls == []
        cache.close()

    def test_dimension_and_inner_are_forwarded(self, tmp_path: Path) -> None:
        """MUTATION: `dimension` returns the cache width or a constant."""
        wrapped, inner, cache = _wrapped(tmp_path)
        assert wrapped.dimension == 4
        assert wrapped.inner is inner
        assert wrapped.cache is cache
        cache.close()

    def test_a_short_result_raises_and_caches_nothing(self, tmp_path: Path) -> None:
        """MUTATION: `zip(strict=True)` -> plain zip (the short list is cached and the
        caller gets a wrong-length result)."""

        class ShortEmbedder(CountingEmbedder):
            def embed(self, texts: list[str]) -> list[list[float]]:
                return super().embed(texts)[:-1]

        cache = _open(tmp_path)
        wrapped = CachedIndexEmbedder(ShortEmbedder(dimension=4), cache)
        with pytest.raises(ValueError, match="returned 1 vector\\(s\\) for 2 text\\(s\\)"):
            wrapped.embed(["a", "b"])
        assert cache.get_many([_key("a"), _key("b")]) == {}
        assert (cache.hits, cache.misses) == (0, 0)
        cache.close()

    def test_every_lookup_group_is_consulted_past_500_keys(self, tmp_path: Path) -> None:
        """`get_many` asks SQLite in groups of 500 keys; a Phase-3 batch regularly has more
        (100000 tokens per batch at ~200 tokens a chunk). 1001 keys span three groups.
        MUTATION: the group loop bounded by `min(len(keys), 500)` (only the first group is
        consulted: 500 of 1001 found, and on a fully warm cache the other 501 are paid for
        again while `hits` under-reports)."""
        texts = [f"t{i}" for i in range(1001)]
        cache = _open(tmp_path)
        cache.put_many([(_key(t), [float(i), 0.0, 0.0, 0.0]) for i, t in enumerate(texts)])
        assert len(cache.get_many([_key(t) for t in texts])) == 1001
        cache.close()

        wrapped, inner, cache = _wrapped(tmp_path / "wrapped")
        wrapped.embed(texts)
        assert inner.calls == [texts]
        wrapped.embed(texts)
        assert inner.calls == [texts]
        assert (cache.hits, cache.misses) == (1001, 1001)
        cache.close()
