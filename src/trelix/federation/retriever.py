"""
FederatedRetriever — fan-out search across multiple independently-indexed repos.

Strategy: parallel query fan-out (one thread per repo) -> collect SearchResult
lists -> RRF merge with per-repo weight -> deduplicate by (file_path, symbol_id).

Cache: TTL-based in-memory cache keyed by SHA-256(query+sorted_repo_paths+k).
cache_ttl=0 disables caching. Thread-safe via threading.Lock.

Cross-repo symbol resolution (Plan A):
  make_scip_symbol_id() produces stable 16-char IDs per (package, version, symbol).
  FederatedRetriever maintains an in-memory SQLite `federation_symbols` table that
  can be queried via resolve_symbol() to find which repos define a given symbol.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from trelix.core.models import SearchResult
from trelix.federation.registry import RepoRegistry
from trelix.retrieval.fusion import reciprocal_rank_fusion
from trelix.retrieval.retriever import Retriever

logger = logging.getLogger("trelix.federation.retriever")


def make_scip_symbol_id(package: str, version: str, qualified_name: str) -> str:
    """
    Create a stable cross-repo symbol ID using SCIP-style concatenation.

    Format: sha256('{package}@{version}:{qualified_name}')[:16]
    Globally unique per (package, version, symbol) tuple.
    Same symbol in different packages -> different ID (version-aware routing).

    Reference: Sourcegraph SCIP cross-repo navigation
    (github.com/sourcegraph/scip-clang/blob/main/docs/CrossRepo.md)
    """
    raw = f"{package}@{version}:{qualified_name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class FederatedRetriever:
    """
    Fan-out retriever across multiple trelix-indexed repos with TTL query cache.

    Usage:
        registry = RepoRegistry.load()
        fed = FederatedRetriever(registry, max_workers=4, cache_ttl=120.0)
        results = fed.retrieve("how does authentication work", k=10)

    Args:
        cache_ttl: Seconds to cache identical query results. 0 disables cache.
    """

    def __init__(
        self,
        registry: RepoRegistry,
        max_workers: int = 4,
        cache_ttl: float = 120.0,
    ) -> None:
        self._registry = registry
        self._max_workers = max_workers
        self._cache_ttl = cache_ttl
        # {cache_key: (results, expiry_monotonic_time)}
        self._cache: dict[str, tuple[list[SearchResult], float]] = {}
        self._cache_lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        # Cross-repo symbol index (in-memory SQLite, rebuilt on record_exports call)
        self._fed_conn = sqlite3.connect(":memory:")
        self._fed_conn.execute(
            """CREATE TABLE IF NOT EXISTS federation_symbols (
                symbol_id      TEXT PRIMARY KEY,
                package        TEXT NOT NULL,
                version        TEXT NOT NULL DEFAULT '',
                qualified_name TEXT NOT NULL,
                repo_alias     TEXT NOT NULL,
                file_path      TEXT NOT NULL
            )"""
        )
        self._fed_conn.commit()

    def resolve_symbol(self, qualified_name: str) -> list[dict]:
        """
        Find all repos that define a symbol with the given qualified name.

        Returns list of {alias, file_path} dicts sorted by alias.
        Uses exact match OR suffix match so 'verify' matches 'AuthService.verify'.
        """
        rows = self._fed_conn.execute(
            """SELECT repo_alias, file_path FROM federation_symbols
               WHERE qualified_name = ? OR qualified_name LIKE ?
               ORDER BY repo_alias""",
            (qualified_name, f"%.{qualified_name}"),
        ).fetchall()
        return [{"alias": r[0], "file_path": r[1]} for r in rows]

    def _make_cache_key(self, query: str, k: int) -> str:
        """SHA-256 key over (query, sorted repo paths, k)."""
        entries = self._registry.list()
        sorted_paths = sorted(e.path for e in entries)
        raw = f"{query}|{sorted_paths}|{k}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> list[SearchResult] | None:
        """Return cached results if still valid and increment hit counter, else None."""
        if self._cache_ttl <= 0:
            return None
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            results, expiry = entry
            if time.monotonic() > expiry:
                del self._cache[key]
                return None
            self._hits += 1
            return results

    def _set_cached(self, key: str, results: list[SearchResult]) -> None:
        """Store results in cache with TTL expiry."""
        if self._cache_ttl <= 0:
            return
        expiry = time.monotonic() + self._cache_ttl
        with self._cache_lock:
            self._cache[key] = (results, expiry)

    def _query_repos(self, query: str, k: int = 10) -> list[SearchResult]:
        """Execute fan-out query to all registered repos. No caching."""
        entries = self._registry.list()
        if not entries:
            return []

        per_repo_results: list[list[SearchResult]] = []

        def _query_one(repo_path: str) -> list[SearchResult]:
            from trelix.core.config import IndexConfig

            config = IndexConfig.model_construct(repo_path=repo_path)
            retriever = Retriever(config)
            ctx = retriever.retrieve(query)
            return ctx.results[:k]

        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(entries))) as pool:
            future_to_entry = {pool.submit(_query_one, entry.path): entry for entry in entries}
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    results = future.result(timeout=30)
                    per_repo_results.append(results)
                except Exception as exc:
                    logger.warning("FederatedRetriever: repo %s failed: %s", entry.alias, exc)

        if not per_repo_results:
            return []

        merged = reciprocal_rank_fusion(per_repo_results)
        seen: set[str] = set()
        deduped: list[SearchResult] = []
        for r in merged:
            dedup_key = f"{r.file.rel_path}:{r.chunk.symbol_id}"
            if dedup_key not in seen:
                seen.add(dedup_key)
                deduped.append(r)
        return deduped[:k]

    def retrieve(self, query: str, k: int = 10) -> list[SearchResult]:
        """
        Fan-out query to all registered repos in parallel.
        Returns merged, deduplicated SearchResult list. Never raises.
        Caches results for cache_ttl seconds (0 = disabled).
        """
        cache_key = self._make_cache_key(query, k)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug("FederatedRetriever: cache HIT for query %r (k=%d)", query, k)
            return cached

        with self._cache_lock:
            self._misses += 1

        try:
            results = self._query_repos(query, k)
        except Exception as exc:
            logger.warning("FederatedRetriever.retrieve failed: %s", exc)
            results = []

        self._set_cached(cache_key, results)
        return results

    def cache_stats(self) -> dict[str, int]:
        """Return cache hit/miss/size stats for observability."""
        with self._cache_lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
            }

    def clear_cache(self) -> None:
        """Evict all cached entries (e.g., after a repo is re-indexed)."""
        with self._cache_lock:
            self._cache.clear()
        logger.debug("FederatedRetriever: cache cleared")
