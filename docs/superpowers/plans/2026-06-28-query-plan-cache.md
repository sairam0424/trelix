# Phase 1b — QueryPlan Cache: TDD Implementation Plan

**Date:** 2026-06-28
**Spec:** `docs/superpowers/specs/2026-06-28-query-plan-cache-design.md`
**Builds on:** Phase 1a — `docs/superpowers/plans/2026-06-27-query-embedding-cache.md`
**Pattern model:** `CachingEmbedder` in `src/trelix/embedder/cache.py`

---

## Overview

Four sequential tasks, each TDD (red → green → refactor → commit):

| # | Task | Files touched | Tests |
|---|------|--------------|-------|
| 1 | `CachingPlanner` class | `src/trelix/retrieval/plan_cache.py` (new) | `tests/unit/test_plan_cache.py` (new) |
| 2 | `plan_cache_size` config field | `src/trelix/core/config.py` | `tests/unit/test_retriever_core.py` (extend) |
| 3 | Wire into `Retriever.__init__` | `src/trelix/retrieval/retriever.py` | `tests/unit/test_retriever_core.py` + `tests/integration/test_plan_cache_e2e.py` (new) |
| 4 | Full validation | — | full suite |

---

## Task 1 — `CachingPlanner` (new file + unit tests)

### TDD steps

**Red:** Create `tests/unit/test_plan_cache.py` with all tests importing
`from trelix.retrieval.plan_cache import CachingPlanner`. The module does not
exist yet — every test will fail with `ModuleNotFoundError`.

Run to confirm all red:
```bash
python -m pytest tests/unit/test_plan_cache.py -x 2>&1 | head -10
```

**Green:** Create `src/trelix/retrieval/plan_cache.py` (exact code in
§ "Implementation" below). Run the suite until all 10 tests pass.

**Refactor:** No structural changes needed — the spec code is already clean.
Add a one-line `__all__` export if desired:
```python
__all__ = ["CachingPlanner"]
```

**Commit:**
```
feat(retrieval): add CachingPlanner LRU cache for QueryPlanner.plan()

CachingPlanner wraps QueryPlanner with the same OrderedDict + Lock pattern
used by CachingEmbedder. Warm hits return in <1ms instead of ~3,000ms.

Cache key: query.strip().lower()
Default max_size: 128 (query diversity per session < embedding diversity)
```

---

### Test file: `tests/unit/test_plan_cache.py`

```python
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

        cache.plan("first")   # miss
        cache.plan("second")  # miss
        cache.plan("first")   # hit
        cache.plan("first")   # hit

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
```

---

### Implementation file: `src/trelix/retrieval/plan_cache.py`

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
```

---

## Task 2 — `plan_cache_size` config field

### TDD steps

**Red:** Add two tests to `tests/unit/test_retriever_core.py` in the
`TestRetrieverCacheWiring` class (which already exists at line 1614):

```python
def test_plan_cache_size_default(self) -> None:
    """plan_cache_size defaults to 128."""
    import tempfile
    from trelix.retrieval.retriever import Retriever

    with tempfile.TemporaryDirectory() as tmp:
        config = IndexConfig(repo_path=tmp)
        assert config.retrieval.plan_cache_size == 128

def test_plan_cache_size_zero_allowed(self) -> None:
    """plan_cache_size=0 is a valid (disabled) value."""
    from trelix.core.config import RetrievalConfig
    cfg = RetrievalConfig(query_cache_size=256, plan_cache_size=0)
    assert cfg.plan_cache_size == 0

def test_plan_cache_size_negative_raises(self) -> None:
    """plan_cache_size < 0 raises ValidationError."""
    from pydantic import ValidationError
    from trelix.core.config import RetrievalConfig
    with pytest.raises(ValidationError):
        RetrievalConfig(plan_cache_size=-1)
```

Run to confirm red:
```bash
python -m pytest tests/unit/test_retriever_core.py::TestRetrieverCacheWiring -x 2>&1 | tail -10
```

**Green:** In `src/trelix/core/config.py`, immediately after the closing of the
`query_cache_size` block (after line 312), add:

```python
    # ── QueryPlan LLM call cache ──────────────────────────────────────────────
    # Caches QueryPlan objects in-memory (LRU, per-Retriever session).
    # 0 = disabled. Default 128: query diversity in a session is lower than
    # embedding diversity, so 128 covers all realistic interactive workloads.
    plan_cache_size: int = Field(
        default=128,
        ge=0,
        alias="TRELIX_RETRIEVAL_PLAN_CACHE_SIZE",
    )
```

Run to confirm green:
```bash
python -m pytest tests/unit/test_retriever_core.py::TestRetrieverCacheWiring -x
```

**Refactor:** None needed.

**Commit:**
```
feat(config): add plan_cache_size to RetrievalConfig

Defaults to 128 (LRU, per-session). Set TRELIX_RETRIEVAL_PLAN_CACHE_SIZE=0
in .env to disable. Mirrors the existing query_cache_size field pattern.
```

---

## Task 3 — Wire `CachingPlanner` into `Retriever.__init__`

### TDD steps

**Red (unit — retriever wiring):** Add two tests to `TestRetrieverCacheWiring`
in `tests/unit/test_retriever_core.py`:

```python
def test_plan_cache_enabled_by_default(self) -> None:
    """Retriever wraps _planner with CachingPlanner when plan_cache_size > 0."""
    import tempfile
    from trelix.retrieval.plan_cache import CachingPlanner
    from trelix.retrieval.retriever import Retriever

    with tempfile.TemporaryDirectory() as tmp:
        config = IndexConfig(repo_path=tmp)
        assert config.retrieval.plan_cache_size == 128
        retriever = Retriever(config)
        assert isinstance(retriever._planner, CachingPlanner)

def test_plan_cache_disabled_when_size_zero(self) -> None:
    """When plan_cache_size=0, Retriever does NOT wrap _planner with CachingPlanner."""
    import tempfile
    from trelix.core.config import RetrievalConfig
    from trelix.retrieval.plan_cache import CachingPlanner
    from trelix.retrieval.retriever import Retriever

    with tempfile.TemporaryDirectory() as tmp:
        config = IndexConfig(
            repo_path=tmp,
            retrieval=RetrievalConfig(plan_cache_size=0),
        )
        retriever = Retriever(config)
        assert not isinstance(retriever._planner, CachingPlanner)
```

**Red (integration):** Create `tests/integration/test_plan_cache_e2e.py` with all
four tests (full content in § "Integration test file" below). Run to confirm
import error / AttributeError since wiring is not yet done:
```bash
python -m pytest tests/integration/test_plan_cache_e2e.py -x 2>&1 | tail -15
```

**Green:** In `src/trelix/retrieval/retriever.py`, immediately after line 97
(`self._planner = QueryPlanner(config.embedder)`), add:

```python
        # Wrap with LRU plan cache when enabled (default: 128 entries).
        # plan() hits are returned in <1ms; cold misses delegate to the LLM unchanged.
        if config.retrieval.plan_cache_size > 0:
            from trelix.retrieval.plan_cache import CachingPlanner
            self._planner = CachingPlanner(
                self._planner, max_size=config.retrieval.plan_cache_size
            )
```

No other changes to `retriever.py`. Line 128 (`plan = self._planner.plan(query)`)
already calls `self._planner.plan()` — `CachingPlanner` intercepts this
transparently.

Run both suites until green:
```bash
python -m pytest tests/unit/test_retriever_core.py::TestRetrieverCacheWiring tests/integration/test_plan_cache_e2e.py -v
```

**Refactor:** None — the inline import mirrors the existing `CachingEmbedder`
wiring block directly above it.

**Commit:**
```
feat(retrieval): wire CachingPlanner into Retriever.__init__

Same lazy-import pattern used by CachingEmbedder. When plan_cache_size > 0
(the default), self._planner is wrapped with CachingPlanner. The external
plan= bypass path (retriever.py line 128) is unaffected.

Warm repeat queries now cost <1ms instead of ~3,000ms (LLM round-trip).
```

---

### Integration test file: `tests/integration/test_plan_cache_e2e.py`

```python
"""
Integration tests for the QueryPlan cache wired into Retriever.

No real LLM calls — QueryPlanner.plan() is replaced by a MagicMock with a
counter. Tests verify that Retriever.retrieve() calls the planner exactly
the right number of times depending on cache state.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

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
```

---

## Task 4 — Full validation

Run the complete test suite to confirm no regressions:

```bash
python -m pytest tests/ -x --tb=short -q
```

Expected outcome: all existing tests pass, plus the new tests:
- `tests/unit/test_plan_cache.py` — 10 tests
- `tests/unit/test_retriever_core.py::TestRetrieverCacheWiring` — 2 new + 2 existing = 4 total
- `tests/integration/test_plan_cache_e2e.py` — 4 tests

Confirm new file is importable in isolation:
```bash
python -c "from trelix.retrieval.plan_cache import CachingPlanner; print('OK')"
```

Confirm config field is present and defaults correctly:
```bash
python -c "
from trelix.core.config import RetrievalConfig
cfg = RetrievalConfig()
print('plan_cache_size:', cfg.plan_cache_size)
assert cfg.plan_cache_size == 128
print('OK')
"
```

---

## File map: what changes, what does not

```
MODIFIED:
  src/trelix/core/config.py            — add plan_cache_size field to RetrievalConfig
  src/trelix/retrieval/retriever.py    — add CachingPlanner wiring after line 97
  tests/unit/test_retriever_core.py    — extend TestRetrieverCacheWiring (+5 tests)

CREATED:
  src/trelix/retrieval/plan_cache.py         — CachingPlanner class
  tests/unit/test_plan_cache.py              — 10 unit tests
  tests/integration/test_plan_cache_e2e.py   — 4 integration tests

UNCHANGED (by design):
  src/trelix/retrieval/planner/agent.py      — QueryPlanner not modified
  src/trelix/retrieval/planner/models.py     — QueryPlan dataclass not modified
  src/trelix/retrieval/retriever.py line 128 — plan = self._planner.plan(query) unchanged
```

---

## Commit sequence summary

```
Task 1:  feat(retrieval): add CachingPlanner LRU cache for QueryPlanner.plan()
Task 2:  feat(config): add plan_cache_size to RetrievalConfig
Task 3:  feat(retrieval): wire CachingPlanner into Retriever.__init__
Task 4:  (no commit — validation only; full suite green confirms Phase 1b done)
```

---

## Expected test count after Phase 1b

| Suite | Before | After |
|-------|--------|-------|
| `tests/unit/test_plan_cache.py` | 0 | 10 |
| `tests/unit/test_retriever_core.py` | existing | +5 |
| `tests/integration/test_plan_cache_e2e.py` | 0 | 4 |
| All other tests | unchanged | unchanged |

Total new tests: **19**. The existing 699 pass unchanged.
