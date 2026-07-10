# trelix Phase 2 — Cross-Repo Symbol Resolution, Semantic Diff Embeddings, Streaming Indexing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three independent scale & intelligence upgrades: (A) cross-repo symbol resolution so federated search resolves which repo defines a shared symbol, (B) semantic diff embeddings so `trelix review --pr` can match diffs against historically similar changes, (C) streaming indexing pipeline that eliminates the in-memory buffer for large repos.

**Architecture:** Plan A adds a `federation_symbols` SQLite table with SCIP-style symbol IDs (`package@version:qualified_name`) and a Bloom-filter pre-filter on `FederatedRetriever` to skip repos that can't possibly contain the symbol. Plan B adds a `diff_chunks` table and `DiffEmbedder` that encodes before/after code body pairs (CCRep-style) at index-time; `trelix review` gains a "similar past diffs" retrieval leg. Plan C refactors `Indexer._index_files` from a buffered list to a generator pipeline with a bounded producer-consumer queue; removes the O(n) memory spike on large repos.

**Tech Stack:** Python 3.11+, SQLite, NetworkX (already dep), hashlib (stdlib), pytest. Plan B optionally adds `diff_parser` (already in trelix). No new hard dependencies.

## Global Constraints

- Python ≥ 3.11
- **DO NOT bump any version numbers** in `pyproject.toml`, `__init__.py`, or CHANGELOG version fields
- All new DB tables use `CREATE TABLE IF NOT EXISTS` — safe on existing indexes
- All new features are opt-in via config flags defaulting to `False`
- Test run from trelix root: `python -m pytest tests/unit/<file>.py -v`
- Conventional commits: `feat(federation):`, `feat(review):`, `perf(indexing):`
- Plans A, B, C are fully independent — implement in any order

---

## Plan A — Cross-Repo Symbol Resolution

### Task A-1: `federation_symbols` table and SCIP-style symbol ID

**Files:**
- Modify: `src/trelix/store/db.py` — add `federation_symbols` table to schema
- Modify: `src/trelix/federation/retriever.py` — add `record_exports()` and `resolve_symbol()` methods
- Test: `tests/unit/test_federation.py` — add `TestCrossRepoSymbolResolution`

**Interfaces:**
- Produces: `Database.init_federation_schema()` — creates `federation_symbols(symbol_id TEXT PK, package TEXT, version TEXT, qualified_name TEXT, repo_alias TEXT, file_path TEXT)`
- Produces: `make_scip_symbol_id(package: str, version: str, qualified_name: str) -> str` — returns `sha256(f"{package}@{version}:{qualified_name}")[:16]`
- Produces: `FederatedRetriever.record_exports(alias: str, repo_path: str)` — indexes all exported symbols from the repo into `federation_symbols`
- Produces: `FederatedRetriever.resolve_symbol(qualified_name: str) -> list[dict]` — returns `[{alias, repo_path, file_path}]` for all repos that define the symbol

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_federation.py` (or create it):

```python
class TestCrossRepoSymbolResolution:
    def test_make_scip_symbol_id_is_deterministic(self):
        from trelix.federation.retriever import make_scip_symbol_id
        id1 = make_scip_symbol_id("myapp", "1.0.0", "AuthService.verify")
        id2 = make_scip_symbol_id("myapp", "1.0.0", "AuthService.verify")
        assert id1 == id2

    def test_make_scip_symbol_id_different_packages_differ(self):
        from trelix.federation.retriever import make_scip_symbol_id
        id1 = make_scip_symbol_id("app-a", "1.0.0", "login")
        id2 = make_scip_symbol_id("app-b", "1.0.0", "login")
        assert id1 != id2

    def test_resolve_symbol_returns_repo_that_defines_it(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from trelix.federation.retriever import FederatedRetriever, make_scip_symbol_id

        registry = MagicMock()
        registry.list.return_value = [
            MagicMock(alias="auth-service", path=str(tmp_path / "auth"))
        ]
        fed = FederatedRetriever(registry)

        # Insert a symbol directly
        fed._fed_db._conn.execute(
            "INSERT INTO federation_symbols VALUES (?, ?, ?, ?, ?, ?)",
            (make_scip_symbol_id("auth-service", "", "AuthService.verify"),
             "auth-service", "", "AuthService.verify", "auth-service", "src/auth.py")
        )
        fed._fed_db._conn.commit()

        results = fed.resolve_symbol("AuthService.verify")
        assert len(results) == 1
        assert results[0]["alias"] == "auth-service"
        assert results[0]["file_path"] == "src/auth.py"

    def test_resolve_symbol_empty_when_not_found(self, tmp_path):
        from unittest.mock import MagicMock
        from trelix.federation.retriever import FederatedRetriever

        registry = MagicMock()
        registry.list.return_value = []
        fed = FederatedRetriever(registry)
        results = fed.resolve_symbol("NonExistentClass.method")
        assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/unit/test_federation.py::TestCrossRepoSymbolResolution -v
```

Expected: `FAILED — ImportError: cannot import name 'make_scip_symbol_id'`

- [ ] **Step 3: Implement `make_scip_symbol_id` and federation schema**

Add to `src/trelix/federation/retriever.py` (at top, after imports):

```python
import hashlib


def make_scip_symbol_id(package: str, version: str, qualified_name: str) -> str:
    """
    Create a stable cross-repo symbol ID using SCIP-style concatenation.

    Format: sha256('{package}@{version}:{qualified_name}')[:16]
    Globally unique per (package, version, symbol) tuple.
    Same symbol in different packages → different ID (version-aware routing).

    Reference: Sourcegraph SCIP cross-repo navigation
    (github.com/sourcegraph/scip-clang/blob/main/docs/CrossRepo.md)
    """
    raw = f"{package}@{version}:{qualified_name}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

Add `_fed_db` initialization and `resolve_symbol` to `FederatedRetriever`:

```python
def __init__(self, registry: RepoRegistry, cache_ttl: float = 120.0) -> None:
    # ... existing init code ...
    # Cross-repo symbol index (in-memory SQLite, rebuilt on registry change)
    import sqlite3
    self._fed_conn = sqlite3.connect(":memory:")
    self._fed_conn.execute(
        """CREATE TABLE IF NOT EXISTS federation_symbols (
            symbol_id    TEXT PRIMARY KEY,
            package      TEXT NOT NULL,
            version      TEXT NOT NULL DEFAULT '',
            qualified_name TEXT NOT NULL,
            repo_alias   TEXT NOT NULL,
            file_path    TEXT NOT NULL
        )"""
    )
    self._fed_conn.commit()

def resolve_symbol(self, qualified_name: str) -> list[dict]:
    """
    Find all repos that define a symbol with the given qualified name.

    Returns list of {alias, repo_path, file_path} dicts sorted by alias.
    Uses case-insensitive suffix match so 'verify' matches 'AuthService.verify'.
    """
    rows = self._fed_conn.execute(
        """SELECT repo_alias, file_path FROM federation_symbols
           WHERE qualified_name = ? OR qualified_name LIKE ?
           ORDER BY repo_alias""",
        (qualified_name, f"%.{qualified_name}"),
    ).fetchall()
    return [{"alias": r[0], "file_path": r[1]} for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/unit/test_federation.py::TestCrossRepoSymbolResolution -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/trelix/federation/retriever.py tests/unit/test_federation.py
git commit -m "feat(federation): add cross-repo symbol resolution with SCIP-style IDs

Adds make_scip_symbol_id() (sha256-based, version-aware) and an in-memory
federation_symbols table on FederatedRetriever. resolve_symbol(name) returns
all repos defining a qualified name — enables 'which repo owns AuthService.verify?'
queries across a microservice federation."
```

---

## Plan B — Semantic Diff Embeddings

### Task B-1: `diff_chunks` table and `DiffEmbedder`

**Files:**
- Modify: `src/trelix/store/db.py` — add `diff_chunks` table
- Create: `src/trelix/review/diff_embedder.py` — `DiffEmbedder` class
- Test: `tests/unit/test_diff_embedder.py` — new file

**Interfaces:**
- Produces: `diff_chunks` table: `(id INTEGER PK, pr_ref TEXT, hunk_header TEXT, before_code TEXT, after_code TEXT, embedding BLOB, chunk_char_count INTEGER)`
- Produces: `DiffEmbedder(embedder: BaseEmbedder)` with `embed_hunk(before_code: str, after_code: str) -> list[float]`
- Produces: `DiffEmbedder.store_pr_diff(db: Database, pr_ref: str, hunks: list[DiffHunk])` — embeds and stores all hunks for a PR

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_diff_embedder.py`:

```python
"""Tests for semantic diff embeddings (CCRep-style before/after body pairs)."""
from __future__ import annotations


class TestDiffEmbedder:
    def test_embed_hunk_concatenates_before_and_after(self):
        from unittest.mock import MagicMock
        from trelix.review.diff_embedder import DiffEmbedder

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 384

        de = DiffEmbedder(mock_embedder)
        result = de.embed_hunk(
            before_code="def login(user, pw): return check(pw)",
            after_code="def login(user, pw): return bcrypt.check(pw)",
        )

        assert result == [0.1] * 384
        # Must call embed_query with concatenated before+after
        call_arg = mock_embedder.embed_query.call_args[0][0]
        assert "def login" in call_arg
        assert "bcrypt" in call_arg

    def test_embed_hunk_handles_empty_before(self):
        from unittest.mock import MagicMock
        from trelix.review.diff_embedder import DiffEmbedder

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.2] * 384

        de = DiffEmbedder(mock_embedder)
        result = de.embed_hunk(before_code="", after_code="def new_func(): pass")
        assert result == [0.2] * 384

    def test_embed_hunk_truncates_overlong_chunks(self):
        from unittest.mock import MagicMock
        from trelix.review.diff_embedder import DiffEmbedder

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.3] * 384

        de = DiffEmbedder(mock_embedder)
        # 3000-char code block — should be truncated before embedding
        long_code = "x = 1\n" * 500
        result = de.embed_hunk(before_code=long_code, after_code="y = 2")
        assert result is not None
        # Verify embed_query was called with something shorter than raw concat
        call_arg = mock_embedder.embed_query.call_args[0][0]
        assert len(call_arg) <= DiffEmbedder.MAX_EMBED_CHARS + 10  # small buffer

    def test_search_similar_diffs_returns_sorted_by_score(self, tmp_path):
        from unittest.mock import MagicMock
        from trelix.review.diff_embedder import DiffEmbedder
        from trelix.store.db import Database

        db = Database(str(tmp_path / "test.db"))
        db.init_schema()

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [1.0] * 4
        mock_embedder.embed.return_value = [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]]
        mock_embedder.dimension = 4

        de = DiffEmbedder(mock_embedder)
        # Insert two diff chunks manually
        db._conn.execute(
            "INSERT INTO diff_chunks (pr_ref, hunk_header, before_code, after_code, chunk_char_count)"
            " VALUES (?, ?, ?, ?, ?)",
            ("owner/repo#1", "@@ -1,3 +1,3 @@", "old", "new", 6),
        )
        db._conn.commit()

        # search_similar_diffs should return results (even if empty — just not crash)
        results = de.search_similar_diffs(db, query_before="old", query_after="new", k=5)
        assert isinstance(results, list)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/unit/test_diff_embedder.py -v
```

Expected: `FAILED — ModuleNotFoundError: No module named 'trelix.review.diff_embedder'`

- [ ] **Step 3: Create `diff_embedder.py`**

Create `src/trelix/review/diff_embedder.py`:

```python
"""
Semantic diff embeddings — CCRep-style before/after code body pairs.

Reference: CCRep (ICSE 2023, arXiv:2302.03924): encode a code change as the
concatenation of before-change and after-change code bodies, fed into a
pre-trained code model to produce contextual embeddings.

Enables 'historically similar diffs' retrieval in trelix review --pr:
  1. At review time, embed each PR hunk (before+after bodies)
  2. Search stored diff_chunks for similar past changes
  3. Surface: 'This change looks like the auth fix in PR #23'

Storage: diff_chunks SQLite table (added to db.py schema).
Chunking: hunk-granular with MAX_DIFF_CHUNKS=500 cap and MAX_EMBED_CHARS
truncation for SVG blobs and minified JS (validated: chunkhound PR #288).
"""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.embedder.base import BaseEmbedder
    from trelix.store.db import Database

logger = logging.getLogger("trelix.review.diff_embedder")

# Max chars to embed per hunk (before+after concatenated).
# Prevents pathological SVG/minified JS from dominating embedding budget.
MAX_EMBED_CHARS = 2000
MAX_DIFF_CHUNKS = 500


class DiffEmbedder:
    """Embed and store code diff hunks for similarity retrieval."""

    MAX_EMBED_CHARS = MAX_EMBED_CHARS

    def __init__(self, embedder: "BaseEmbedder") -> None:
        self._embedder = embedder

    def embed_hunk(self, before_code: str, after_code: str) -> list[float]:
        """
        Embed a code change as a before+after body pair (CCRep encoding).

        Concatenates before and after bodies with a separator, truncates to
        MAX_EMBED_CHARS, and embeds using the configured embedder.

        Returns the embedding vector. Never raises — returns [] on failure.
        """
        try:
            combined = f"{before_code}\n---\n{after_code}"
            if len(combined) > MAX_EMBED_CHARS:
                combined = combined[:MAX_EMBED_CHARS]
            return self._embedder.embed_query(combined)
        except Exception as exc:
            logger.debug("DiffEmbedder.embed_hunk failed: %s", exc)
            return []

    def store_pr_diff(
        self,
        db: "Database",
        pr_ref: str,
        hunks: list[dict],
    ) -> int:
        """
        Embed and store all hunks for a PR reference.

        Each hunk dict must have: {hunk_header, before_code, after_code}.
        Caps at MAX_DIFF_CHUNKS hunks per PR.

        Returns number of chunks stored.
        """
        stored = 0
        for hunk in hunks[:MAX_DIFF_CHUNKS]:
            before = hunk.get("before_code", "")
            after = hunk.get("after_code", "")
            header = hunk.get("hunk_header", "")
            char_count = len(before) + len(after)

            embedding = self.embed_hunk(before_code=before, after_code=after)
            if not embedding:
                continue

            try:
                packed = struct.pack(f"{len(embedding)}f", *embedding)
                db._conn.execute(
                    """INSERT INTO diff_chunks
                       (pr_ref, hunk_header, before_code, after_code,
                        embedding, chunk_char_count)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (pr_ref, header, before, after, packed, char_count),
                )
                stored += 1
            except Exception as exc:
                logger.debug("Failed to store diff chunk: %s", exc)

        if stored:
            db._conn.commit()
        return stored

    def search_similar_diffs(
        self,
        db: "Database",
        query_before: str,
        query_after: str,
        k: int = 5,
    ) -> list[dict]:
        """
        Find historically similar diffs using before+after embedding similarity.

        Returns list of {pr_ref, hunk_header, before_code, after_code, score}
        sorted descending by cosine similarity.
        """
        import math

        query_emb = self.embed_hunk(before_code=query_before, after_code=query_after)
        if not query_emb:
            return []

        rows = db._conn.execute(
            "SELECT pr_ref, hunk_header, before_code, after_code, embedding "
            "FROM diff_chunks WHERE embedding IS NOT NULL"
        ).fetchall()

        results = []
        q_norm = math.sqrt(sum(v * v for v in query_emb)) or 1.0

        for pr_ref, header, before, after, packed in rows:
            if not packed:
                continue
            try:
                n = len(packed) // 4
                stored_emb = list(struct.unpack(f"{n}f", packed))
                dot = sum(a * b for a, b in zip(query_emb, stored_emb))
                s_norm = math.sqrt(sum(v * v for v in stored_emb)) or 1.0
                score = dot / (q_norm * s_norm)
                results.append({
                    "pr_ref": pr_ref,
                    "hunk_header": header,
                    "before_code": before,
                    "after_code": after,
                    "score": score,
                })
            except Exception:
                continue

        return sorted(results, key=lambda x: x["score"], reverse=True)[:k]
```

Also add `diff_chunks` to `db.py` schema:

```python
CREATE TABLE IF NOT EXISTS diff_chunks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    pr_ref           TEXT    NOT NULL,
    hunk_header      TEXT    NOT NULL DEFAULT '',
    before_code      TEXT    NOT NULL DEFAULT '',
    after_code       TEXT    NOT NULL DEFAULT '',
    embedding        BLOB,
    chunk_char_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_diff_chunks_pr_ref ON diff_chunks(pr_ref);
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/unit/test_diff_embedder.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/trelix/review/diff_embedder.py src/trelix/store/db.py \
        tests/unit/test_diff_embedder.py
git commit -m "feat(review): add semantic diff embeddings with CCRep-style before/after encoding

DiffEmbedder encodes each PR hunk as before+after body pair concatenation
(arXiv:2302.03924). store_pr_diff() caps at 500 hunks, truncates at 2000 chars.
search_similar_diffs() finds historically similar changes by cosine similarity.
diff_chunks table added to SQLite schema with idx_diff_chunks_pr_ref index."
```

---

## Plan C — Streaming Indexing Pipeline

### Task C-1: Refactor `Indexer` to generator-based pipeline

**Files:**
- Modify: `src/trelix/indexing/indexer.py` — `index()` method, replace list buffer with generator + bounded queue
- Test: `tests/unit/test_indexer_core.py` — add `TestStreamingIndexing` class

**Interfaces:**
- Produces: `Indexer._iter_files(repo_path: str) -> Iterator[IndexedFile]` — generator yielding files
- The `index()` public API remains identical — callers see no change
- New config flag: `TRELIX_INDEXER_STREAMING=true` (default `False`) enables the generator path
- Streaming path uses `concurrent.futures.ThreadPoolExecutor` with `max_workers=4` and a bounded `queue.Queue(maxsize=64)` to prevent unbounded buffering

- [ ] **Step 1: Add `streaming_enabled` config flag**

In `src/trelix/core/config.py`, find `IndexerConfig` (or `IndexConfig`) and add:

```python
streaming_enabled: bool = Field(
    default=False,
    alias="TRELIX_INDEXER_STREAMING",
)
```

- [ ] **Step 2: Write the failing test**

Add to `tests/unit/test_indexer_core.py`:

```python
class TestStreamingIndexing:
    def test_streaming_mode_produces_same_result_as_batch(self, tmp_path):
        """Streaming pipeline must produce identical output to batch mode."""
        import os
        from trelix.core.config import IndexConfig
        from trelix.indexing.indexer import Indexer
        from unittest.mock import patch, MagicMock

        # Create a minimal repo with 3 Python files
        (tmp_path / "a.py").write_text("def foo(): pass")
        (tmp_path / "b.py").write_text("def bar(): pass")
        (tmp_path / "c.py").write_text("def baz(): pass")

        config_batch = IndexConfig(repo_path=str(tmp_path))
        config_stream = IndexConfig(repo_path=str(tmp_path))
        config_stream.indexer.streaming_enabled = True

        with patch("trelix.indexing.indexer.make_embedder") as mock_emb, \
             patch("trelix.indexing.indexer.make_vector_store"):
            mock_emb.return_value.embed.return_value = [[0.1] * 4]
            mock_emb.return_value.dimension = 4

            indexer_batch = Indexer(config_batch)
            indexer_stream = Indexer(config_stream)

            result_batch = indexer_batch.index(str(tmp_path))
            result_stream = indexer_stream.index(str(tmp_path))

        # Both must index the same number of files
        assert result_batch.get("files_processed") == result_stream.get("files_processed")

    def test_streaming_mode_does_not_buffer_all_files_in_memory(self, tmp_path):
        """Generator path must yield files one at a time, not collect all first."""
        from trelix.indexing.indexer import Indexer
        from trelix.core.config import IndexConfig
        from unittest.mock import patch

        config = IndexConfig(repo_path=str(tmp_path))
        config.indexer.streaming_enabled = True

        with patch("trelix.indexing.indexer.make_embedder"), \
             patch("trelix.indexing.indexer.make_vector_store"):
            indexer = Indexer(config)
            # _iter_files must be a generator (has __next__)
            gen = indexer._iter_files(str(tmp_path))
            assert hasattr(gen, "__next__"), "_iter_files must be a generator"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
python -m pytest tests/unit/test_indexer_core.py::TestStreamingIndexing -v
```

Expected: `FAILED` — `_iter_files` does not exist or `streaming_enabled` not on config.

- [ ] **Step 4: Implement `_iter_files` generator in `indexer.py`**

In `src/trelix/indexing/indexer.py`, find the `index()` method. Add `_iter_files`:

```python
def _iter_files(self, repo_path: str):
    """
    Generator yielding IndexedFile objects for the given repo path.

    Used by the streaming indexing pipeline to avoid buffering all files
    in memory before parsing begins. Yields files as they are discovered
    by the walker, allowing the parse/embed pipeline to start immediately.
    """
    from trelix.indexing.walker import FileWalker

    walker = FileWalker(self.config)
    yield from walker.walk(repo_path)
```

Then in `index()`, add a conditional path when `streaming_enabled=True`:

```python
def index(self, repo_path: str) -> dict:
    cfg = self.config
    if getattr(cfg.indexer, "streaming_enabled", False):
        return self._index_streaming(repo_path)
    return self._index_batch(repo_path)  # rename existing logic

def _index_streaming(self, repo_path: str) -> dict:
    """
    Streaming indexing pipeline — generator-based, bounded memory.

    Files are yielded one at a time from _iter_files() and processed
    through a ThreadPoolExecutor with a bounded queue. Memory usage is
    O(queue_size) rather than O(repo_size).
    """
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    QUEUE_SIZE = 64
    results = {"files_processed": 0, "symbols_indexed": 0, "errors": 0}
    file_queue: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
    SENTINEL = object()

    def producer():
        for f in self._iter_files(repo_path):
            file_queue.put(f)
        file_queue.put(SENTINEL)

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    while True:
        item = file_queue.get()
        if item is SENTINEL:
            break
        try:
            result = self.index_file(str(item.abs_path))
            if result.get("status") == "ok":
                results["files_processed"] += 1
                results["symbols_indexed"] += result.get("symbols_updated", 0)
        except Exception as exc:
            logger.debug("Streaming index error for %s: %s", item, exc)
            results["errors"] += 1

    producer_thread.join()
    return results
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/unit/test_indexer_core.py::TestStreamingIndexing -v
```

Expected: `2 passed`

- [ ] **Step 6: Run full indexer tests**

```bash
python -m pytest tests/unit/test_indexer_core.py -v -q 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/trelix/indexing/indexer.py src/trelix/core/config.py \
        tests/unit/test_indexer_core.py
git commit -m "feat(indexing): add streaming indexing pipeline with generator + bounded queue

TRELIX_INDEXER_STREAMING=true enables generator-based file processing that
eliminates the O(n) memory spike from buffering all files before parsing.
Files are yielded from _iter_files() into a bounded queue (size=64) and
consumed by the parse/embed/store pipeline immediately. Default off."
```

---

## Task P2-Final: Update ROADMAP

**Files:**
- Modify: `docs/ROADMAP.md`

- [ ] **Step 1: Mark Phase 2 items in progress / shipped**

Update the v2.6.x Remaining backlog section to reflect what shipped.

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest tests/unit/ -x -q 2>&1 | tail -5
```

- [ ] **Step 3: Lint**

```bash
ruff check src/ tests/ && ruff format --check src/ tests/ && echo "CLEAN"
```

- [ ] **Step 4: Commit**

```bash
git add docs/ROADMAP.md
git commit -m "docs: update ROADMAP for Phase 2 progress"
```
