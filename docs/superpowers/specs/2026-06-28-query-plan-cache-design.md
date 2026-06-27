# QueryPlan LLM Call Cache — Design Spec

**Phase:** 1b
**Date:** 2026-06-28
**Status:** Approved
**Builds on:** Phase 1a query-embedding-cache-design (2026-06-27)
**Research basis:** e2e audit — cold P50 = 4,548ms; LLM planner dominates after embed cache lands

---

## Problem

Phase 1a added `CachingEmbedder` which eliminates the ~350–1000ms embedding round-trip on
warm queries. After that fix, the dominant latency term becomes the LLM planner call.

`QueryPlanner.plan()` invokes `gpt-4o-mini` (or `gpt-4o` on Azure) to:
1. Classify intent (`IntentType`)
2. Generate HyDE snippets (`SubQuery.hyde_snippet`)
3. Produce BM25 tokens and grep hints

This call costs **2–4 seconds per query** at temperature=0. Even though temperature=0
should in principle be deterministic, OpenAI does not guarantee bit-identical outputs
across calls — so HyDE text varies between runs, the embedding of that HyDE text varies,
and the embedding cache rarely hits on a "warm" second pass for the same user query.

The result: after Phase 1a, a session re-asking the same question still pays the full
LLM round-trip:

| Pass | Planner (LLM) | Embed (API) | Total approx |
|------|--------------|-------------|--------------|
| Cold | ~3,000ms | ~500ms | ~4,500ms |
| Warm (Phase 1a only) | ~3,000ms | <1ms | ~3,000ms |
| Warm (Phase 1a + 1b) | <1ms | <1ms | <50ms |

**Root cause:** `QueryPlan` objects are discarded after each `retrieve()` call.
Identical raw queries always re-enter the LLM.

---

## Goal

Cache `QueryPlan` objects keyed on the normalised raw query string.
A warm hit short-circuits the LLM call entirely and returns the frozen plan directly.

Expected outcomes:
- Warm P50: ~4,500ms → **<50ms** (both plan and embed cached)
- Cold first query: unchanged (~4,500ms — LLM + embed must run once)
- Same-session repeat: **<50ms**

---

## Non-Goals

- Persistent cross-session plan cache (on-disk / SQLite)
- Fuzzy / semantic deduplication of similar-but-not-identical queries
- Global process-level cache (staleness risk if config changes)
- Caching plans produced by an externally-supplied `plan` argument
  (those bypass `self._planner` already; nothing to cache)
- Invalidating the plan cache when the index is rebuilt (out of scope; the
  Retriever instance is already replaced on re-index)

---

## Architecture

### New file: `src/trelix/retrieval/plan_cache.py`

`CachingPlanner` wraps `QueryPlanner` with the identical LRU pattern used by
`CachingEmbedder`.

```
Retriever
  └── self._planner = CachingPlanner(raw_planner, max_size=128)
                          │
                          └── plan(query)
                                ├── cache hit  → return cached QueryPlan  (<1ms)
                                └── cache miss → delegate to QueryPlanner.plan(query)
                                               → store in cache → return
```

**Cache key:** `query.strip().lower()` — matches the normalisation used by
`CachingEmbedder` so the two caches share the same query surface.

**Cache value:** The `QueryPlan` dataclass instance returned by `QueryPlanner.plan()`.
`QueryPlan` is a plain frozen-style dataclass (no mutable default fields beyond
`routing_tier`). Callers treat it as read-only — the retriever reads from it but never
mutates it — so returning the same object on cache hits is safe.

**Eviction:** LRU via `collections.OrderedDict`. Default `max_size=128` (query
diversity in a single session is lower than token diversity — 128 covers any realistic
interactive session without significant memory cost).

**Thread safety:** One `threading.Lock` guards all reads and writes (same pattern as
`CachingEmbedder`).

**Scope:** Per-`Retriever` instance. Cache lives with the Retriever and is destroyed
when the Retriever is garbage-collected.

**Bypass hatch:** `Retriever.retrieve(query, plan=<external>)` bypasses `self._planner`
entirely (line 127-128 in retriever.py). `CachingPlanner` is therefore never called in
that code path — no interaction, no risk of stale cached plan overriding an explicit one.

---

## Interface

```python
# src/trelix/retrieval/plan_cache.py

class CachingPlanner:
    def __init__(self, planner: QueryPlanner, max_size: int = 128) -> None: ...

    # CACHED — returns from OrderedDict if key present
    def plan(self, query: str, project_context: dict | None = None) -> QueryPlan: ...

    # Introspection (for tests and metrics)
    @property
    def cache_size(self) -> int: ...      # current entry count
    @property
    def hit_count(self) -> int: ...       # total hits since creation or last clear()
    @property
    def miss_count(self) -> int: ...      # total misses since creation or last clear()
    def clear(self) -> None: ...          # flush cache + reset counters
```

Note: `project_context` is forwarded to `QueryPlanner.plan()` on a miss but is **not**
part of the cache key. Rationale: in practice `project_context` is either always `None`
(current retriever usage) or stable within a session. If variable context becomes a
real use case in a future phase, the cache key can be extended to include a hash of
`project_context`. This is explicitly a deferred decision — YAGNI applies here.

---

## Implementation details

### `src/trelix/retrieval/plan_cache.py` — full skeleton

```python
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

        # Cache miss — call LLM outside the lock
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
        with self._lock:
            return len(self._cache)

    @property
    def hit_count(self) -> int:
        with self._lock:
            return self._hits

    @property
    def miss_count(self) -> int:
        with self._lock:
            return self._misses

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
```

### `src/trelix/retrieval/retriever.py` — `__init__` change only

After line 97 (`self._planner = QueryPlanner(config.embedder)`), add:

```python
# Wrap with LRU plan cache when enabled (default: 128 entries).
# plan() hits are returned in <1ms; cold misses delegate to the LLM unchanged.
if config.retrieval.plan_cache_size > 0:
    from trelix.retrieval.plan_cache import CachingPlanner
    self._planner = CachingPlanner(
        self._planner, max_size=config.retrieval.plan_cache_size
    )
```

Zero other changes to retriever.py. Line 128 (`plan = self._planner.plan(query)`)
already calls `self._planner.plan()` — `CachingPlanner` intercepts this transparently.

### `src/trelix/core/config.py` — one new field on `RetrievalConfig`

Add immediately after the existing `query_cache_size` block (line 312):

```python
# ── QueryPlan LLM call cache ──────────────────────────────────────────
# Caches QueryPlan objects in-memory (LRU, per-Retriever session).
# 0 = disabled. Default 128: query diversity in a session is lower than
# embedding diversity, so 128 covers all realistic interactive workloads.
plan_cache_size: int = Field(
    default=128,
    ge=0,
    alias="TRELIX_RETRIEVAL_PLAN_CACHE_SIZE",
)
```

---

## Correctness argument

**Is it safe to return the same `QueryPlan` object on every hit?**

`QueryPlan` is a `@dataclass` with all fields set at construction time.
The retriever reads from it (`plan.intent`, `plan.strategy`, `plan.sub_queries`,
`plan.raw_query`, `plan.routing_tier`) but never reassigns any field — checked by
grep across `retriever.py`. The only structural concern is `sub_queries: list[SubQuery]`
being a mutable list; however the retriever only iterates over it, never appends or
pops. Returning a shared reference is therefore safe.

If a future change were to mutate a `QueryPlan` in-place, the dataclass should be
converted to `@dataclass(frozen=True)` at that time. This is noted as a follow-up
concern but is **not** required for Phase 1b.

**Does `project_context` exclusion from the cache key create a correctness hazard?**

Current call site (retriever.py line 128): `self._planner.plan(query)` — no
`project_context` passed. The parameter defaults to `None` everywhere in the codebase.
Exclusion is therefore safe for all shipped code. Explicitly documented in the
interface section above.

---

## File structure

```
src/trelix/retrieval/
  plan_cache.py         ← NEW: CachingPlanner
  retriever.py          ← MODIFY: wrap self._planner in __init__ if plan_cache_size > 0
  planner/
    agent.py            ← unchanged
    models.py           ← unchanged

src/trelix/core/
  config.py             ← MODIFY: add plan_cache_size to RetrievalConfig

tests/unit/
  test_plan_cache.py    ← NEW (see testing section)

tests/integration/
  test_plan_cache_e2e.py  ← NEW (see testing section)
```

---

## Testing strategy

### Unit tests — `tests/unit/test_plan_cache.py`

Mirror `tests/unit/test_embedder_cache.py` exactly. Use a `MagicMock` for
`QueryPlanner` whose `plan()` returns a stable `default_plan("test query")` object.

| Test | What it verifies |
|------|-----------------|
| `test_second_call_returns_same_plan` | Two identical queries return same object, planner called once |
| `test_cache_key_normalisation` | "How Does Auth Work" and " how does auth work " share one slot |
| `test_different_queries_each_call_planner` | Two distinct queries each call underlying planner |
| `test_lru_eviction_at_max_size` | Fill cache to max_size+1, assert oldest evicted and re-called on next access |
| `test_zero_size_disables_cache` | `max_size=0`, every call delegates to planner |
| `test_negative_max_size_raises` | `ValueError` raised on construction |
| `test_hit_miss_counts` | `hit_count` / `miss_count` increment correctly across miss/hit sequence |
| `test_cache_size_property` | `cache_size` reflects actual entry count |
| `test_clear_resets_all` | After `clear()`, counts zero, same query is a miss again |
| `test_concurrent_same_query_calls_planner_at_most_twice` | 20 threads all `plan("same")` — planner called ≤ 2 times (benign race on first insert); all threads receive the same object |

Thread-safety test pattern (identical to `CachingEmbedder` test):

```python
def test_concurrent_same_query_calls_planner_at_most_twice() -> None:
    mock = MagicMock()
    mock.plan.return_value = default_plan("same query")
    cache = CachingPlanner(mock, max_size=128)

    results: list[QueryPlan] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            results.append(cache.plan("same query"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 20
    assert not errors
    assert mock.plan.call_count <= 2
```

### Integration tests — `tests/integration/test_plan_cache_e2e.py`

No real LLM calls — mock `QueryPlanner.plan` with a counter.

| Test | What it verifies |
|------|-----------------|
| `test_same_query_twice_calls_planner_once` | Call `Retriever.retrieve` twice with same query, assert planner called once |
| `test_plan_cache_zero_disables` | `plan_cache_size=0`, same query twice, assert planner called twice |
| `test_different_queries_each_call_planner` | Two different queries, assert planner called twice |
| `test_external_plan_bypasses_cache` | Pass `plan=<QueryPlan>` externally, assert `_planner.plan` never called |

---

## Backward compatibility

- `plan_cache_size` defaults to `128` — cache is **on by default** for all users.
  No `.env` changes needed.
- `TRELIX_RETRIEVAL_PLAN_CACHE_SIZE=0` in `.env` disables it (useful for benchmarks
  or when the planner is already local/fast via a future local-model integration).
- `CachingPlanner` exposes the same `plan()` signature as `QueryPlanner` — no other
  code needs changes.
- `Retriever.retrieve(query, plan=<external>)` path is unaffected: the external plan
  skips `self._planner` entirely (line 127 in retriever.py), so the cache is never
  consulted and never polluted by external plans.

---

## Expected metrics after Phase 1b

| Metric | Phase 0 (baseline) | Phase 1a (embed cache) | Phase 1b (+ plan cache) |
|--------|-------------------|----------------------|------------------------|
| Cold P50 | 4,548ms | 4,548ms | 4,548ms (LLM + embed, unchanged) |
| Warm P50 | 4,548ms | ~3,000ms | **<50ms** |
| Warm P95 | 8,300ms+ | ~3,200ms | **<100ms** |
| LLM calls (10 same queries) | 10 | 10 | **1** |
| Embed API calls (10 same queries) | 10 | 1 | 1 |
| Memory overhead (128 × ~1KB per plan) | 0 | 0 | ~128KB |

---

## Open questions (not blocking Phase 1b)

1. **`project_context` in cache key**: If variable `project_context` ever ships,
   extend key to `(query.strip().lower(), json.dumps(project_context, sort_keys=True))`.
2. **`frozen=True` on QueryPlan**: Convert the dataclass if any future code mutates
   plan fields in-place. Currently safe as-is.
3. **Cross-session persistence**: Could store plans in `.trelix/cache/plans.sqlite`
   for faster cold starts. Deferred — adds invalidation complexity.
4. **TTL / staleness on index rebuild**: If the user rebuilds the index between two
   calls to the same `Retriever` instance (unusual), the cached plan is still valid
   because plans describe query intent, not index content. No TTL needed.
