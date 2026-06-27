"""
LRU in-memory cache for embed_query() calls.

CachingEmbedder wraps any BaseEmbedder and caches query-time embeddings.
Document embeddings (embed/embed_async) pass through uncached — they run
at index time and are not repeated in interactive sessions.

Cache key: text.strip().lower() — "Auth Work" and "auth work" share one slot.
Eviction: LRU via OrderedDict. When full, least-recently-used entry removed.
Thread safety: single Lock guards all reads and writes.
Scope: per-CachingEmbedder instance (lives with the Retriever).
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

from trelix.embedder.base import BaseEmbedder

logger = logging.getLogger("trelix.embedder.cache")


class CachingEmbedder(BaseEmbedder):
    """
    Transparent LRU cache for embed_query().

    Usage::

        raw = make_embedder(config.embedder)
        cached = CachingEmbedder(raw, max_size=256)
        # Now use cached everywhere raw was used.
    """

    def __init__(self, embedder: BaseEmbedder, max_size: int = 256) -> None:
        self._embedder = embedder
        self._max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ── Cached path ──────────────────────────────────────────────────────────

    def embed_query(self, text: str) -> list[float]:
        """Return cached vector for text, or delegate and cache the result."""
        if self._max_size == 0:
            return self._embedder.embed_query(text)

        key = text.strip().lower()

        with self._lock:
            if key in self._cache:
                # Move to end (most-recently-used)
                self._cache.move_to_end(key)
                self._hits += 1
                logger.debug("embed_query cache_hit=True key=%r", key)
                return self._cache[key]

        # Cache miss — compute outside the lock to avoid blocking other threads
        t0 = time.perf_counter()
        vector = self._embedder.embed_query(text)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        with self._lock:
            # Re-check: another thread may have populated it while we computed
            if key not in self._cache:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)  # evict LRU (first item)
                self._cache[key] = vector
            self._misses += 1
            logger.debug(
                "embed_query cache_hit=False key=%r latency_ms=%s", key, elapsed_ms
            )

        return vector

    # ── Passthrough paths ────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Document embedding — always delegates, never cached."""
        return self._embedder.embed(texts)

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Async document embedding — always delegates, never cached."""
        return await self._embedder.embed_async(texts)

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def cache_size(self) -> int:
        """Current number of cached entries."""
        with self._lock:
            return len(self._cache)

    @property
    def hit_count(self) -> int:
        """Total cache hits since creation or last clear()."""
        with self._lock:
            return self._hits

    @property
    def miss_count(self) -> int:
        """Total cache misses since creation or last clear()."""
        with self._lock:
            return self._misses

    def clear(self) -> None:
        """Flush the cache and reset hit/miss counters."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
