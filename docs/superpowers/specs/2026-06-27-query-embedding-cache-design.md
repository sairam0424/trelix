# Query Embedding Cache — Design Spec

**Date:** 2026-06-27
**Status:** Approved
**Research basis:** Stress test audit — p50 query latency 4,867ms, root cause = live API call per query

---

## Problem

Every `Retriever.retrieve()` call executes `self.embedder.embed_query(text)`, which makes
a synchronous network call to the configured embedding provider (OpenAI/Azure/Bedrock).
This single call dominates total retrieval latency:

- trelix self-index (144 files): P50 = 4,867ms, P95 = 8,369ms
- Tombstone (325 files): P50 = 4,632ms, P95 = 5,787ms

In interactive use (IDE extension, MCP server, CLI), the same query is frequently repeated
within a session (e.g. "how does authentication work" asked twice, or sub-queries from
multi-step planning that share vocabulary). There is no mechanism to reuse already-computed
query vectors.

---

## Goal

Eliminate the API round-trip for repeat queries within a Retriever session.
Expected outcome: repeat queries drop from ~5,000ms → <10ms. First-time queries unchanged.

---

## Non-Goals

- Persistent cross-session cache (on-disk / SQLite) — adds invalidation complexity
- Caching document embeddings (`embed()`) — those run at index time, not query time
- Global process-level cache — creates staleness risk if embedder config changes
- Approximate deduplication (fuzzy query matching) — out of scope for Phase 1

---

## Architecture

### New file: `src/trelix/embedder/cache.py`

A `CachingEmbedder` class that wraps any `BaseEmbedder` and intercepts `embed_query()`.

```
Retriever
  └── self.embedder = CachingEmbedder(raw_embedder, max_size=256)
                          │
                          ├── embed_query(text)
                          │     ├── cache hit  → return cached vector  (< 1ms)
                          │     └── cache miss → delegate to raw_embedder.embed_query()
                          │                      → store in cache → return
                          │
                          └── embed() / embed_async() / dimension
                                → passthrough to raw_embedder (no caching)
```

**Cache key:** `text.strip().lower()` — normalises whitespace and case so
"How does Auth work" and "how does auth work" are treated as the same query.

**Eviction:** LRU using `collections.OrderedDict`. When `len(cache) >= max_size`,
evict the least-recently-used entry before inserting.

**Thread safety:** One `threading.Lock` guards all reads and writes.
Retriever may be called from `ThreadPoolExecutor` workers (concurrent search legs)
but `embed_query` is called before the fan-out, so lock contention is minimal.

**Scope:** Per-`Retriever` instance. The cache lives and dies with the Retriever.
This is intentional — if the user switches embedder provider, they create a new
Retriever with a new config, and the new CachingEmbedder starts cold.

### Integration point: `src/trelix/retrieval/retriever.py`

In `Retriever.__init__`, after `self.embedder = make_embedder(config.embedder)`:

```python
if config.retrieval.query_cache_size > 0:
    from trelix.embedder.cache import CachingEmbedder
    self.embedder = CachingEmbedder(self.embedder, max_size=config.retrieval.query_cache_size)
```

Zero other changes to retriever.py. The `_run_single_leg` method calls
`self.embedder.embed_query(embed_text)` at line 507 — this already goes through
`CachingEmbedder` transparently.

### Config: `src/trelix/core/config.py`

One new field on `RetrievalConfig`:

```python
query_cache_size: int = Field(
    default=256,
    alias="TRELIX_RETRIEVAL_QUERY_CACHE_SIZE",
)
```

Setting to `0` disables caching entirely (useful for benchmarking or when
the embedder is already local/fast).

---

## `CachingEmbedder` interface

```python
class CachingEmbedder(BaseEmbedder):
    def __init__(self, embedder: BaseEmbedder, max_size: int = 256) -> None: ...

    # CACHED — returns from OrderedDict if key present
    def embed_query(self, text: str) -> list[float]: ...

    # PASSTHROUGH — delegates to wrapped embedder unchanged
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_async(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...

    # Introspection (for tests and metrics)
    @property
    def cache_size(self) -> int: ...          # current entries
    @property
    def hit_count(self) -> int: ...           # total cache hits since creation
    @property
    def miss_count(self) -> int: ...          # total cache misses since creation
    def clear(self) -> None: ...              # flush (useful in tests)
```

Logging: cache hit logged at `DEBUG` with `cache_hit=True, key=<normalised>, latency_ms=<n>`.
Cache miss logged at `DEBUG` with `cache_hit=False, key=<normalised>`.

---

## Testing strategy

### Unit tests (`tests/unit/test_embedder_cache.py`)

- `test_second_call_returns_same_vector` — call `embed_query` twice, assert vectors equal,
  assert underlying embedder called exactly once
- `test_cache_key_normalisation` — "Auth Work" and "auth work" hit same cache entry
- `test_lru_eviction_at_max_size` — fill to max_size+1, assert oldest evicted
- `test_zero_size_disables_cache` — max_size=0, every call goes to underlying embedder
- `test_passthrough_embed_not_cached` — `embed(["a","b"])` always delegates
- `test_thread_safety` — 20 threads each call `embed_query("same")` concurrently,
  assert underlying called exactly once (or at most once per race window)
- `test_cache_stats` — hit_count / miss_count increment correctly
- `test_clear_resets_stats` — after clear(), hit_count=0, miss_count=0, next call is miss

### Integration test (`tests/integration/test_query_cache_e2e.py`)

- `test_cache_reduces_api_calls` — mock `embed_query` with a counter, call `Retriever.retrieve`
  twice with same query, assert counter == 1
- `test_cache_size_zero_disables` — set `query_cache_size=0`, call twice, assert counter == 2
- `test_different_queries_both_call_api` — two different queries, assert counter == 2

### Performance test (manual, not in CI)

Run `tests/perf/test_query_latency.py` against a live index:
- 20 queries, first pass (cold) → record P50/P95
- Same 20 queries, second pass (warm) → record P50/P95
- Assert warm P50 < 50ms (cache hit dominates)

---

## File structure

```
src/trelix/embedder/
  cache.py              ← NEW: CachingEmbedder
  base.py               ← unchanged
  __init__.py           ← add CachingEmbedder to __all__

src/trelix/retrieval/
  retriever.py          ← MODIFY: wrap embedder in __init__ if cache_size > 0

src/trelix/core/
  config.py             ← MODIFY: add query_cache_size to RetrievalConfig

tests/unit/
  test_embedder_cache.py   ← NEW: 8 unit tests

tests/integration/
  test_query_cache_e2e.py  ← NEW: 3 integration tests (no real API calls)
```

---

## Backward compatibility

- `query_cache_size` defaults to `256` — cache is ON by default for all users.
  No `.env` changes needed.
- Setting `TRELIX_RETRIEVAL_QUERY_CACHE_SIZE=0` in `.env` disables it.
- `CachingEmbedder` is transparent — any code holding a `BaseEmbedder` reference
  continues to work without changes.
- Public API (`from trelix import ...`) unchanged.

---

## Expected metrics after implementation

| Metric | Before | After (warm) | After (cold, first query) |
|--------|--------|--------------|---------------------------|
| P50 latency | 4,867ms | < 10ms | ~5,000ms (unchanged) |
| P95 latency | 8,369ms | < 50ms | ~8,500ms (unchanged) |
| API calls (20 same queries) | 20 | 1 | 1 |
| Memory overhead (256 entries × 1024 floats × 4 bytes) | 0 | ~1 MB | ~1 MB |
