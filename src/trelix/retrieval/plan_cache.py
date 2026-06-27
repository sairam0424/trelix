"""
LRU in-memory cache for QueryPlanner.plan() calls.

CachingPlanner wraps QueryPlanner and caches QueryPlan objects keyed on
the normalised raw query string.  A warm hit short-circuits the LLM call
entirely, dropping latency from ~3,000ms to <1ms.

Cache key:   query.strip().lower()
Eviction:    LRU via OrderedDict.
Thread safe: single Lock guards all reads and writes.
Scope:       per-CachingPlanner instance (lives with the Retriever).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from trelix.retrieval.planner.agent import QueryPlanner
from trelix.retrieval.planner.models import QueryPlan

logger = logging.getLogger("trelix.retrieval.plan_cache")

__all__ = ["CachingPlanner"]


class CachingPlanner:
    """Transparent LRU cache for QueryPlanner.plan()."""

    def __init__(self, planner: QueryPlanner, max_size: int = 128) -> None:
        if max_size < 0:
            raise ValueError(f"max_size must be >= 0, got {max_size}")
        self._planner = planner
        self._max_size = max_size
        self._cache: OrderedDict[str, QueryPlan] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def plan(
        self, query: str, project_context: dict[str, Any] | None = None
    ) -> QueryPlan:
        """Return cached QueryPlan for query, or delegate and cache the result."""
        if self._max_size == 0:
            return self._planner.plan(query, project_context)

        key = query.strip().lower()

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                logger.debug("plan cache_hit=True key=%r", key)
                return self._cache[key]

        # Cache miss — call LLM outside the lock to avoid blocking other threads
        t0 = time.perf_counter()
        result = self._planner.plan(query, project_context)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        with self._lock:
            if key not in self._cache:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = result
                self._misses += 1
                logger.debug(
                    "plan cache_hit=False key=%r latency_ms=%s", key, elapsed_ms
                )
            else:
                # Concurrent thread populated the cache while we were calling LLM
                self._hits += 1
                logger.debug(
                    "plan concurrent_hit=True key=%r latency_ms=%s", key, elapsed_ms
                )

        return result

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
