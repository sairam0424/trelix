# Query Embedding Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LRU in-memory cache for `embed_query()` calls so repeat searches within a Retriever session skip the embedding API and return in <10ms instead of ~5,000ms.

**Architecture:** A new `CachingEmbedder` wrapper class (decorator pattern) intercepts `embed_query()` with an `OrderedDict`-based LRU cache keyed on normalised query text. `embed()` and `embed_async()` pass through unchanged. `Retriever.__init__` wraps its embedder automatically when `config.retrieval.query_cache_size > 0` (default 256). No changes to any call site beyond `retriever.py` and `config.py`.

**Tech Stack:** Python 3.11+, `collections.OrderedDict`, `threading.Lock`, `src/` layout, pytest, `.venv/bin/python`.

## Global Constraints

- Python ≥ 3.11; use `from __future__ import annotations` at top of every new file
- `src/` layout — all new source under `src/trelix/`; all tests under `tests/`
- Repo: `/Users/sairamugge/Desktop/Not-Humans-World/trelix`
- Venv: `.venv/bin/python`
- Run tests with: `cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && .venv/bin/python -m pytest <path> -v --tb=short`
- All existing 1127 unit tests must remain green throughout
- No new required runtime deps — `collections` and `threading` are stdlib
- `CachingEmbedder` must be a proper `BaseEmbedder` subclass (so any type-annotated `BaseEmbedder` slot accepts it)
- Cache key: `text.strip().lower()` — normalises whitespace and case
- Thread safety: one `threading.Lock` guards all cache reads and writes
- Default `query_cache_size = 256`; `0` disables caching entirely

---

### Task 1: `CachingEmbedder` — `src/trelix/embedder/cache.py`

**Files:**
- Create: `src/trelix/embedder/cache.py`
- Test: `tests/unit/test_embedder_cache.py`

**Interfaces:**
- Consumes: `BaseEmbedder` from `trelix.embedder.base`
- Produces:
  - `CachingEmbedder(embedder: BaseEmbedder, max_size: int = 256)` — the cache wrapper
  - `.embed_query(text: str) -> list[float]` — cached
  - `.embed(texts: list[str]) -> list[list[float]]` — passthrough
  - `async .embed_async(texts: list[str]) -> list[list[float]]` — passthrough
  - `.dimension: int` — property, passthrough
  - `.cache_size: int` — property, current number of cached entries
  - `.hit_count: int` — property, total cache hits since creation
  - `.miss_count: int` — property, total cache misses since creation
  - `.clear() -> None` — flush cache and reset stats

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_embedder_cache.py`:

```python
"""Unit tests for CachingEmbedder."""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, call

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


class TestCachingEmbedderLRU:
    def test_lru_eviction_at_max_size(self) -> None:
        mock = _make_mock_embedder()
        cache = CachingEmbedder(mock, max_size=3)

        cache.embed_query("a")  # inserted: a
        cache.embed_query("b")  # inserted: a, b
        cache.embed_query("c")  # inserted: a, b, c  (full)
        cache.embed_query("d")  # inserted: b, c, d  (a evicted — LRU)

        assert mock.embed_query.call_count == 4

        # "a" was evicted — calling it again must hit the API
        cache.embed_query("a")
        assert mock.embed_query.call_count == 5

        # "b" was NOT evicted (accessed after a was inserted)
        cache.embed_query("b")
        assert mock.embed_query.call_count == 5  # still 5 — cache hit

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
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/test_embedder_cache.py -v --tb=short 2>&1 | head -15
```
Expected: `ModuleNotFoundError: No module named 'trelix.embedder.cache'`

- [ ] **Step 3: Implement `src/trelix/embedder/cache.py`**

```python
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
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/test_embedder_cache.py -v --tb=short
```
Expected: all 12 tests PASS

- [ ] **Step 5: Run full unit suite — expect no regressions**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/ -q --tb=line 2>&1 | tail -3
```
Expected: 1127+ passed

- [ ] **Step 6: Commit**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
git add src/trelix/embedder/cache.py tests/unit/test_embedder_cache.py && \
git commit -m "feat(cache): CachingEmbedder — LRU query embedding cache

OrderedDict-based LRU wraps any BaseEmbedder. embed_query() cached,
embed/embed_async passthrough. Thread-safe. max_size=0 disables.
12 unit tests covering hits, misses, eviction, thread safety, stats."
```

---

### Task 2: Export `CachingEmbedder` + config field

**Files:**
- Modify: `src/trelix/embedder/__init__.py`
- Modify: `src/trelix/core/config.py` (line ~302, inside `RetrievalConfig`)

**Interfaces:**
- Consumes: `CachingEmbedder` from Task 1
- Produces:
  - `from trelix.embedder import CachingEmbedder` works
  - `RetrievalConfig.query_cache_size: int = 256` — readable via `config.retrieval.query_cache_size`
  - `TRELIX_RETRIEVAL_QUERY_CACHE_SIZE` env var overrides it

- [ ] **Step 1: Write failing test for config field**

Add to `tests/unit/test_config.py` (read the file first to find a good insertion point at the end):

```python
class TestRetrievalConfigQueryCache:
    def test_default_query_cache_size_is_256(self) -> None:
        from trelix.core.config import RetrievalConfig
        cfg = RetrievalConfig()
        assert cfg.query_cache_size == 256

    def test_zero_disables_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig
        monkeypatch.setenv("TRELIX_RETRIEVAL_QUERY_CACHE_SIZE", "0")
        cfg = RetrievalConfig()
        assert cfg.query_cache_size == 0

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig
        monkeypatch.setenv("TRELIX_RETRIEVAL_QUERY_CACHE_SIZE", "512")
        cfg = RetrievalConfig()
        assert cfg.query_cache_size == 512
```

- [ ] **Step 2: Run — expect failure**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/test_config.py::TestRetrievalConfigQueryCache -v --tb=short 2>&1 | tail -8
```
Expected: FAIL — `AttributeError: 'RetrievalConfig' object has no attribute 'query_cache_size'`

- [ ] **Step 3: Add `query_cache_size` to `RetrievalConfig`**

Read `src/trelix/core/config.py` first. Find `RetrievalConfig`. After the `graph_rag_threshold_results` line (~line 302), add:

```python
    # ── Query embedding cache ─────────────────────────────────────────────────
    # Caches embed_query() results in-memory (LRU, per-Retriever session).
    # 0 = disabled. Default 256 covers a typical interactive session.
    query_cache_size: int = Field(
        default=256,
        alias="TRELIX_RETRIEVAL_QUERY_CACHE_SIZE",
    )
```

- [ ] **Step 4: Update `embedder/__init__.py`**

Read `src/trelix/embedder/__init__.py`. Change it to:

```python
"""Embedder abstraction — public API."""

from trelix.embedder.base import (
    AzureOpenAIEmbedder,
    BaseEmbedder,
    BedrockCohereEmbedder,
    BedrockTitanEmbedder,
    LocalCodeEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    make_embedder,
)
from trelix.embedder.cache import CachingEmbedder

__all__ = [
    "BaseEmbedder",
    "AzureOpenAIEmbedder",
    "OpenAIEmbedder",
    "LocalEmbedder",
    "VoyageEmbedder",
    "LocalCodeEmbedder",
    "BedrockTitanEmbedder",
    "BedrockCohereEmbedder",
    "CachingEmbedder",
    "make_embedder",
]
```

- [ ] **Step 5: Verify imports**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -c "from trelix.embedder import CachingEmbedder; print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Run config tests — expect pass**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/test_config.py -v --tb=short 2>&1 | tail -10
```
Expected: all PASS including new 3 tests

- [ ] **Step 7: Run full suite**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/ -q --tb=line 2>&1 | tail -3
```
Expected: 1130+ passed

- [ ] **Step 8: Commit**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
git add src/trelix/embedder/__init__.py src/trelix/core/config.py tests/unit/test_config.py && \
git commit -m "feat(cache): export CachingEmbedder + add query_cache_size to RetrievalConfig

- CachingEmbedder added to trelix.embedder.__all__
- RetrievalConfig.query_cache_size: int = 256 (TRELIX_RETRIEVAL_QUERY_CACHE_SIZE)
- 3 new config unit tests"
```

---

### Task 3: Wire `CachingEmbedder` into `Retriever.__init__`

**Files:**
- Modify: `src/trelix/retrieval/retriever.py` (lines 77–90, `__init__`)
- Test: `tests/unit/test_retriever_core.py` (add new tests)
- Create: `tests/integration/test_query_cache_e2e.py`

**Interfaces:**
- Consumes: `CachingEmbedder` from Task 1; `query_cache_size` from Task 2
- Produces: `Retriever(config)` automatically wraps its embedder when `config.retrieval.query_cache_size > 0`

- [ ] **Step 1: Write failing unit tests**

Read `tests/unit/test_retriever_core.py`. Append these new test classes at the end:

```python
class TestRetrieverCacheWiring:
    def test_cache_enabled_by_default(self) -> None:
        """Retriever wraps embedder with CachingEmbedder when query_cache_size > 0."""
        from trelix.embedder.cache import CachingEmbedder

        with tempfile.TemporaryDirectory() as tmp:
            config = IndexConfig(repo_path=tmp)
            assert config.retrieval.query_cache_size == 256
            retriever = Retriever(config)
            assert isinstance(retriever.embedder, CachingEmbedder)

    def test_cache_disabled_when_size_zero(self) -> None:
        """When query_cache_size=0, Retriever does NOT wrap with CachingEmbedder."""
        from trelix.embedder.cache import CachingEmbedder
        from trelix.core.config import RetrievalConfig

        with tempfile.TemporaryDirectory() as tmp:
            config = IndexConfig(
                repo_path=tmp,
                retrieval=RetrievalConfig(query_cache_size=0),
            )
            retriever = Retriever(config)
            assert not isinstance(retriever.embedder, CachingEmbedder)
```

Note: `tempfile` is already imported in that test file. If not, add `import tempfile` at the top.

- [ ] **Step 2: Run — expect failure**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/test_retriever_core.py::TestRetrieverCacheWiring -v --tb=short 2>&1 | tail -10
```
Expected: FAIL — `AssertionError: assert False` (Retriever doesn't wrap yet)

- [ ] **Step 3: Modify `Retriever.__init__`**

Read `src/trelix/retrieval/retriever.py` lines 77–91. Change `__init__` to:

```python
    def __init__(self, config: IndexConfig) -> None:
        self.config = config
        self.db = Database(config.db_path_absolute)
        raw_embedder: BaseEmbedder = make_embedder(config.embedder)
        # Wrap with LRU query cache when enabled (default: 256 entries).
        # embed_query() hits are returned in <1ms; embed() passthrough unchanged.
        if config.retrieval.query_cache_size > 0:
            from trelix.embedder.cache import CachingEmbedder
            self.embedder: BaseEmbedder = CachingEmbedder(
                raw_embedder, max_size=config.retrieval.query_cache_size
            )
        else:
            self.embedder = raw_embedder
        self.vector_store: BaseVectorStore = make_vector_store(
            config=config,
            dimension=self.embedder.dimension,
        )
        # Instantiate the LLM query planner. Falls back gracefully to
        # default_plan() when no API key is set (provider=local).
        self._planner = QueryPlanner(config.embedder)

        # Debug output dir: <repo_root>/.trelix/debug/
        self._debug_dir = Path(config.repo_path) / ".trelix" / "debug"
```

- [ ] **Step 4: Run wiring tests — expect pass**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/test_retriever_core.py::TestRetrieverCacheWiring -v --tb=short
```
Expected: both PASS

- [ ] **Step 5: Write integration tests**

Create `tests/integration/test_query_cache_e2e.py`:

```python
"""
Integration tests for the query embedding cache wired into Retriever.

These tests use mocked embedders — no real API calls. They verify that the
Retriever calls embed_query() exactly once for repeated queries when the cache
is enabled, and twice when it is disabled.
"""
from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import EmbedderConfig, IndexConfig, RetrievalConfig


def _make_config(tmp: str, cache_size: int = 256) -> IndexConfig:
    return IndexConfig(
        repo_path=tmp,
        retrieval=RetrievalConfig(query_cache_size=cache_size),
    )


def _mock_retriever_deps(retriever: object, vector: list[float]) -> MagicMock:
    """Patch the internal embedder's embed_query and fake vector search."""
    from trelix.embedder.cache import CachingEmbedder

    # If cache is enabled, the underlying embedder is inside CachingEmbedder
    raw = retriever.embedder._embedder if isinstance(retriever.embedder, CachingEmbedder) else retriever.embedder
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
```

- [ ] **Step 6: Run integration tests**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/integration/test_query_cache_e2e.py -v --tb=short
```
Expected: all 3 PASS

- [ ] **Step 7: Run full unit suite**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/ -q --tb=line 2>&1 | tail -3
```
Expected: 1130+ passed

- [ ] **Step 8: Commit**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
git add src/trelix/retrieval/retriever.py \
        tests/unit/test_retriever_core.py \
        tests/integration/test_query_cache_e2e.py && \
git commit -m "feat(cache): wire CachingEmbedder into Retriever.__init__

Retriever wraps its embedder with CachingEmbedder when
config.retrieval.query_cache_size > 0 (default: 256).
2 unit tests for wiring; 3 integration tests verifying
repeat queries skip the embedding API."
```

---

### Task 4: Performance smoke test + `.env.example` + full validation

**Files:**
- Create: `tests/perf/test_query_latency.py` (manual, not in CI)
- Modify: `.env.example` (add `TRELIX_RETRIEVAL_QUERY_CACHE_SIZE` entry)

**Interfaces:**
- Consumes: everything from Tasks 1–3

- [ ] **Step 1: Create `tests/perf/test_query_latency.py`**

```python
"""
Manual performance test — NOT run in CI (no @pytest.mark, no conftest).

Usage:
    # Index a repo first:
    trelix index /path/to/repo

    # Then run (requires OPENAI_API_KEY or AZURE_API_KEY in .env):
    python tests/perf/test_query_latency.py /path/to/repo

Measures cold vs warm P50/P95 for 20 queries to validate the cache impact.
"""
from __future__ import annotations

import statistics
import sys
import time

from dotenv import load_dotenv

load_dotenv()

QUERIES = [
    "how does authentication work",
    "database connection pooling",
    "error handling patterns",
    "how is the index built",
    "what parsers are supported",
    "chunking algorithm",
    "vector search implementation",
    "BM25 scoring",
    "call graph expansion",
    "LLM synthesis",
    "how does the file watcher work",
    "GraphRAG map reduce",
    "embedding providers",
    "test coverage",
    "config validation",
    "how to add a new language parser",
    "retrieval pipeline",
    "reranking implementation",
    "incremental indexing",
    "SQLite schema",
]


def run_queries(retriever: object, label: str) -> list[float]:
    latencies = []
    for q in QUERIES:
        t0 = time.perf_counter()
        retriever.retrieve(q)  # type: ignore[attr-defined]
        latencies.append((time.perf_counter() - t0) * 1000)
    lat = sorted(latencies)
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95)]
    print(f"{label}: P50={p50:.0f}ms  P95={p95:.0f}ms  Max={max(lat):.0f}ms")
    return latencies


if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
    from trelix import IndexConfig, Retriever

    config = IndexConfig(repo_path=repo)
    retriever = Retriever(config)

    print("=== Query Embedding Cache Performance Test ===")
    print(f"Repo: {repo}")
    print(f"Cache size: {config.retrieval.query_cache_size}")
    print()

    cold = run_queries(retriever, "Cold (first pass)")
    warm = run_queries(retriever, "Warm (second pass, cached)")

    cold_p50 = sorted(cold)[len(cold) // 2]
    warm_p50 = sorted(warm)[len(warm) // 2]
    speedup = cold_p50 / max(warm_p50, 0.1)
    print(f"\nSpeedup: {speedup:.0f}x  (warm P50 {warm_p50:.0f}ms vs cold P50 {cold_p50:.0f}ms)")
    if warm_p50 < 50:
        print("✅ Cache working: warm P50 < 50ms")
    else:
        print(f"⚠️  Warm P50 {warm_p50:.0f}ms > 50ms — check if cache is enabled")
```

- [ ] **Step 2: Create `tests/perf/__init__.py`**

```bash
touch /Users/sairamugge/Desktop/Not-Humans-World/trelix/tests/perf/__init__.py
```

- [ ] **Step 3: Update `.env.example`**

Read `.env.example`. Find the `# Retrieval` section (or after the store section). Add:

```bash
# ---------------------------------------------------------------------------
# Query embedding cache
# ---------------------------------------------------------------------------
# LRU cache for embed_query() — eliminates repeat API calls within a session.
# Default: 256 (covers a typical interactive session). Set to 0 to disable.
# TRELIX_RETRIEVAL_QUERY_CACHE_SIZE=256
```

- [ ] **Step 4: Run ruff + format check**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/ruff check src/ tests/ --fix 2>&1 | tail -3 && \
.venv/bin/ruff format src/ tests/ 2>&1 | tail -3
```
Expected: `All checks passed!` and files reformatted/already formatted

- [ ] **Step 5: Run mypy**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/mypy src/trelix/ --ignore-missing-imports 2>&1 | tail -3
```
Expected: `Success: no issues found in 64 source files`

- [ ] **Step 6: Run full test suite with coverage**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/ tests/integration/test_query_cache_e2e.py \
    -q --tb=line --cov=trelix --cov-report=term-missing 2>&1 | tail -8
```
Expected: 1133+ passed, coverage ≥ 75%

- [ ] **Step 7: Final commit**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
git add tests/perf/ .env.example && \
git commit -m "docs(cache): perf smoke test + .env.example cache size entry

- tests/perf/test_query_latency.py: manual cold/warm latency comparison
- .env.example: TRELIX_RETRIEVAL_QUERY_CACHE_SIZE documented"
```
