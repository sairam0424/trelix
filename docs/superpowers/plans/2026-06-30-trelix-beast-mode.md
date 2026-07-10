# trelix Beast-Mode Upgrade Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade trelix from v2.0.0 to v3.0.0 by activating every half-built capability, adding HyDE query expansion, FLARE-style agentic re-retrieval, a file-summary 5th retrieval leg, streaming API endpoints, PageRank-based symbol importance scoring, a query telemetry/feedback loop, and a CoIR-backed evaluation harness.

**Architecture:** Seven orthogonal upgrade modules, each independently testable, all gated by config flags so zero regression when disabled. The plan follows trelix's established patterns: pydantic-settings config fields with env-var overrides, `try/except ImportError` for optional deps, LRU caches for repeated calls, and strict mypy compliance. Branch: `feature/beast-mode` → PR → `develop`.

**Tech Stack:** Python 3.11+, pydantic-settings, sqlite-vec, networkx, sentence-transformers, FastAPI (serve extra), existing TrelixChatClient ABC, tiktoken for token counting.

## Global Constraints

- Python `>=3.11,<3.13` — no walrus operator in type annotations, no 3.13-only syntax
- mypy strict — every new function needs full type annotations; `int(x or 0)` for sqlite3 Any returns
- ruff line-length = 100
- No new mandatory dependencies — all new features behind optional extras or existing deps
- Version bump: `pyproject.toml` + `src/trelix/__init__.py` + `packages/trelix-mcp/src/trelix_mcp/__init__.py` → `"3.0.0"` in Task 7 only
- All tests: `pytest tests/unit/ -x --tb=short` must pass after every task
- Branch: work on `feature/beast-mode` branched from `develop`
- Config env prefix: `TRELIX_` top-level, sub-configs have their own prefix (e.g. `TRELIX_RETRIEVAL_`)
- New config fields follow naming pattern: `snake_case`, exposed as `TRELIX_<SECTION>_<FIELD>` env var
- Commit format: `feat(<scope>): <description>` following existing conventional-commits style
- `scratch-pad/` is git-ignored; daily log lives at `scratch-pad/daily-logs/YYYY-MM-DD.md`

---

## Phase 1 — Retrieval Power-Ups (Tasks 1–3)

---

### Task 1: File-Summary 5th Retrieval Leg

**What this unlocks:** File summaries are generated at index time (RAPTOR-style), stored in `file_summaries` table, and their embeddings stored in the vector table with `chunk_id = -(file_id)` convention. But `_retrieve_standard` never queries them. This task wires them in as a 5th retrieval leg.

**Files:**
- Modify: `src/trelix/store/vector.py` — add `search_file_summaries(query_embedding, k)` to `BaseVectorStore` ABC and `SQLiteVectorStore` impl
- Modify: `src/trelix/core/config.py` — add `file_summary_leg_enabled`, `top_k_file_summary` to `RetrievalConfig`
- Modify: `src/trelix/retrieval/retriever.py` — add `_summary_search()` private method; inject into `_retrieve_standard`
- Test: `tests/unit/test_retriever_file_summary.py`

**Interfaces:**
- Consumes: `BaseVectorStore.search(query_embedding, k)` — same signature, different row filter
- Produces:
  - `BaseVectorStore.search_file_summaries(query_embedding: list[float], k: int) -> list[tuple[int, float]]` — returns `(file_id, score)` pairs (negative chunk_ids mapped back to file_id)
  - `Retriever._summary_search(query_embedding: list[float], k: int) -> list[SearchResult]`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_retriever_file_summary.py`:

```python
"""Tests for the file-summary 5th retrieval leg."""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trelix.core.config import IndexConfig, RetrievalConfig
from trelix.core.models import IndexedFile, Language, SearchResult, Symbol, SymbolKind
from trelix.store.db import Database
from trelix.store.vector import SQLiteVectorStore


def _build_db_with_summary(tmp_path: Path) -> tuple[Database, int, int]:
    """Return (db, file_id, summary_chunk_id) with one file and a stored summary."""
    db = Database(tmp_path / "index.db")
    fid = db.upsert_file(
        IndexedFile(
            path=str(tmp_path / "auth.py"),
            rel_path="auth.py",
            language=Language.PYTHON,
            hash="abc",
            size_bytes=100,
        )
    )
    db.upsert_file_summary(fid, "Handles user authentication and JWT token lifecycle.")
    return db, fid, -(fid)  # convention: chunk_id = -file_id for summary rows


class TestSearchFileSummaries:
    def test_search_file_summaries_returns_file_id_score_pairs(self, tmp_path: Path) -> None:
        db, fid, neg_fid = _build_db_with_summary(tmp_path)
        store = SQLiteVectorStore(tmp_path / "index.db", dimension=4)
        # Insert a fake summary embedding using the -(file_id) convention
        store.upsert_file_summary_embedding(fid, [0.1, 0.2, 0.3, 0.4])
        results = store.search_file_summaries([0.1, 0.2, 0.3, 0.4], k=5)
        assert len(results) >= 1
        returned_file_ids = [r[0] for r in results]
        assert fid in returned_file_ids

    def test_search_file_summaries_excludes_symbol_chunks(self, tmp_path: Path) -> None:
        """Regular chunk rows (positive chunk_id) must NOT appear in summary search."""
        db, fid, _ = _build_db_with_summary(tmp_path)
        store = SQLiteVectorStore(tmp_path / "index.db", dimension=4)
        # Insert a regular chunk embedding (positive id)
        store.upsert(chunk_id=42, embedding=[0.1, 0.2, 0.3, 0.4])
        store.upsert_file_summary_embedding(fid, [0.9, 0.9, 0.9, 0.9])
        summary_results = store.search_file_summaries([0.1, 0.2, 0.3, 0.4], k=10)
        returned_ids = [r[0] for r in summary_results]
        assert 42 not in returned_ids  # regular chunks excluded

    def test_summary_leg_disabled_by_default(self, tmp_path: Path) -> None:
        config = IndexConfig(repo_path=str(tmp_path))
        assert config.retrieval.file_summary_leg_enabled is False

    def test_summary_leg_config_fields(self, tmp_path: Path) -> None:
        config = IndexConfig(repo_path=str(tmp_path))
        # Fields exist and have sensible defaults
        assert config.retrieval.top_k_file_summary == 5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix
.venv/bin/python -m pytest tests/unit/test_retriever_file_summary.py -v --tb=short 2>&1 | head -30
```
Expected: `AttributeError: type object 'BaseVectorStore' has no attribute 'search_file_summaries'`

- [ ] **Step 3: Add `search_file_summaries` to vector store**

In `src/trelix/store/vector.py`, add to the `BaseVectorStore` ABC (after `upsert_file_summary_embedding`):

```python
@abstractmethod
def search_file_summaries(self, query_embedding: list[float], k: int) -> list[tuple[int, float]]:
    """Search file-summary embeddings only. Returns (file_id, score) pairs.

    Convention: summary embeddings are stored with chunk_id = -(file_id).
    This method filters to negative chunk_ids and maps back to file_id.
    """
```

Add implementation to `SQLiteVectorStore` (after `upsert_file_summary_embedding` impl):

```python
def search_file_summaries(self, query_embedding: list[float], k: int) -> list[tuple[int, float]]:
    """Search only file-summary rows (chunk_id < 0), return (file_id, score) pairs."""
    with self._lock:
        try:
            vec_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)
            rows = self._conn.execute(
                """
                SELECT chunk_id, distance
                FROM chunk_embeddings
                WHERE chunk_id < 0
                  AND chunk_embeddings MATCH ?
                  AND k = ?
                ORDER BY distance
                """,
                (vec_bytes, k),
            ).fetchall()
            # chunk_id = -(file_id), so file_id = -(chunk_id)
            return [(-int(row[0]), float(row[1])) for row in rows]
        except Exception:
            # Flat fallback when HNSW WHERE filter isn't supported
            rows = self._conn.execute(
                "SELECT chunk_id, distance FROM chunk_embeddings "
                "WHERE chunk_id < 0 ORDER BY distance LIMIT ?",
                (k,),
            ).fetchall()
            return [(-int(row[0]), float(row[1])) for row in rows]
```

Also add stub to `LanceVectorStore` in `src/trelix/store/vector_lance.py` and `QdrantVectorStore` in `src/trelix/store/vector_qdrant.py`:

```python
def search_file_summaries(self, query_embedding: list[float], k: int) -> list[tuple[int, float]]:
    # Lance/Qdrant: delegate to base search filtered to negative IDs
    results = self.search(query_embedding, k=k * 5)
    return [(-cid, score) for cid, score in results if cid < 0][:k]
```

- [ ] **Step 4: Add config fields to `RetrievalConfig`**

In `src/trelix/core/config.py`, inside `RetrievalConfig`, add after `graph_search_max_results`:

```python
# File-summary retrieval leg (5th leg — RAPTOR-style, off by default)
# Requires file_summaries_enabled=True at index time to have any summaries stored.
file_summary_leg_enabled: bool = Field(
    default=False,
    alias="TRELIX_RETRIEVAL_FILE_SUMMARY_LEG",
)
top_k_file_summary: int = Field(
    default=5,
    alias="TRELIX_RETRIEVAL_FILE_SUMMARY_TOP_K",
)
```

- [ ] **Step 5: Add `_summary_search` to `Retriever`**

In `src/trelix/retrieval/retriever.py`, add this private method after `_vector_search`:

```python
def _summary_search(self, query_embedding: list[float], k: int) -> list[SearchResult]:
    """Search file-summary embeddings (5th retrieval leg).

    Returns SearchResult objects where the symbol is the first symbol in the file
    (used as a representative for the file-level summary context).
    Returns empty list when no summaries are indexed or file_summary_leg_enabled=False.
    """
    results: list[SearchResult] = []
    try:
        pairs = self.vector_store.search_file_summaries(query_embedding, k=k)
        for file_id, score in pairs:
            file_obj = self.db.get_file_by_id(file_id)
            if file_obj is None:
                continue
            summary_text = self.db.get_file_summary(file_id)
            if not summary_text:
                continue
            # Build a synthetic Chunk representing the file-level summary
            from trelix.core.models import Chunk
            synthetic_chunk = Chunk(
                id=-(file_id),  # negative = summary sentinel
                symbol_id=0,
                chunk_text=summary_text,
                token_count=len(summary_text.split()),
            )
            # Pick the first symbol in the file as the representative symbol
            symbols = self.db.get_symbols_for_file(file_id)
            if not symbols:
                continue
            rep_symbol = symbols[0]
            results.append(
                SearchResult(
                    chunk=synthetic_chunk,
                    symbol=rep_symbol,
                    file=file_obj,
                    score=score,
                    rank=0,
                    source="file_summary",
                )
            )
    except Exception as exc:
        logger.warning("File summary leg failed (non-fatal): %s", exc)
    return results
```

- [ ] **Step 6: Wire summary leg into `_retrieve_standard`**

In `_retrieve_standard` in `retriever.py`, after the existing `grep_results` collection (line ~260) and before the `reciprocal_rank_fusion` call, add:

```python
summary_results: list[SearchResult] = []
if cfg.file_summary_leg_enabled and query_embedding is not None:
    summary_results = self._summary_search(query_embedding, k=cfg.top_k_file_summary)
```

Then add `summary_results` to the `reciprocal_rank_fusion` call:

```python
fused = reciprocal_rank_fusion(
    [vector_results, bm25_results, grep_results, summary_results],
    k=cfg.rrf_k,
    weights=_weights,
)
```

Also add to the logger.info call and trace dict:
```python
logger.info(
    "Pre-fusion leg sizes: vector=%d bm25=%d grep=%d summary=%d",
    len(vector_results), len(bm25_results), len(grep_results), len(summary_results),
)
```

You will also need to capture `query_embedding` earlier in `_retrieve_standard`. Find where `_run_subquery_legs` is called and ensure the first sub-query's embedding is available. The cleanest approach: add `query_embedding: list[float] | None = None` as a local, set it from `self.embedder.embed_query(plan.sub_queries[0].semantic_query)` if `file_summary_leg_enabled` is True.

- [ ] **Step 7: Add `get_symbols_for_file` to `Database`**

In `src/trelix/store/db.py`, add after `get_file_summary`:

```python
def get_symbols_for_file(self, file_id: int, limit: int = 5) -> list[Symbol]:
    """Return up to `limit` symbols for a file, ordered by line_start."""
    rows = self._conn.execute(
        "SELECT id, file_id, name, qualified_name, kind, line_start, line_end, "
        "signature, docstring, body, parent_id "
        "FROM symbols WHERE file_id = ? ORDER BY line_start LIMIT ?",
        (file_id, limit),
    ).fetchall()
    return [
        Symbol(
            id=int(row[0]),
            file_id=int(row[1]),
            name=row[2],
            qualified_name=row[3],
            kind=SymbolKind(row[4]),
            line_start=int(row[5]),
            line_end=int(row[6]),
            signature=row[7] or "",
            docstring=row[8],
            body=row[9] or "",
            parent_id=int(row[10]) if row[10] is not None else None,
        )
        for row in rows
    ]
```

- [ ] **Step 8: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_retriever_file_summary.py -v --tb=short
```
Expected: all 4 tests PASS.

```bash
.venv/bin/python -m pytest tests/unit/ -x --tb=short -q 2>&1 | tail -5
```
Expected: full suite passes.

- [ ] **Step 9: Commit**

```bash
git add src/trelix/store/vector.py src/trelix/store/vector_lance.py src/trelix/store/vector_qdrant.py \
        src/trelix/core/config.py src/trelix/retrieval/retriever.py src/trelix/store/db.py \
        tests/unit/test_retriever_file_summary.py
git commit -m "feat(retrieval): add file-summary 5th retrieval leg (RAPTOR-style)

Wire stored file_summaries as a 5th RRF leg in _retrieve_standard.
Gated by TRELIX_RETRIEVAL_FILE_SUMMARY_LEG=true (default off).
Adds search_file_summaries() to vector store, get_symbols_for_file() to DB."
```

---

### Task 2: HyDE Fallback for No-LLM Mode + Multi-Query Expansion

**What this unlocks:** HyDE (Hypothetical Document Embeddings) is already wired when the LLM QueryPlanner runs (`hyde_snippet` field on `SubQuery`, embedded in `_run_subquery_legs`). But when no LLM is configured (Tier 1 fast path / `default_plan()`), `hyde_snippet` is always `""`. This task adds: (a) a lightweight template-based HyDE fallback that generates a synthetic code snippet without an LLM call, and (b) multi-query expansion that spawns 2–3 semantic variants of the query before embedding.

**Files:**
- Create: `src/trelix/retrieval/query_expansion.py` — `HyDEExpander` and `MultiQueryExpander` classes
- Modify: `src/trelix/core/config.py` — add `hyde_fallback_enabled`, `multi_query_enabled`, `multi_query_count` to `RetrievalConfig`
- Modify: `src/trelix/retrieval/retriever.py` — call expanders in `_run_subquery_legs` when `hyde_snippet` is empty
- Test: `tests/unit/test_query_expansion.py`

**Interfaces:**
- Produces:
  - `HyDEExpander(llm_config: LLMConfig | None).expand(query: str) -> str` — returns synthetic code snippet or `""` on failure
  - `MultiQueryExpander(llm_config: LLMConfig | None, n: int = 2).expand(query: str) -> list[str]` — returns variant queries (includes original)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_query_expansion.py`:

```python
"""Tests for HyDE and multi-query expansion."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trelix.retrieval.query_expansion import HyDEExpander, MultiQueryExpander


class TestHyDEExpander:
    def test_returns_empty_string_when_no_llm(self) -> None:
        expander = HyDEExpander(llm_config=None)
        result = expander.expand("how does authentication work")
        assert result == ""

    def test_returns_snippet_when_llm_available(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(content="def authenticate(user): ...")
        with patch("trelix.retrieval.query_expansion.build_chat_client", return_value=mock_client):
            from trelix.core.config import LLMConfig
            expander = HyDEExpander(llm_config=LLMConfig())
            result = expander.expand("how does authentication work")
        assert "def authenticate" in result

    def test_returns_empty_on_llm_failure(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("API down")
        with patch("trelix.retrieval.query_expansion.build_chat_client", return_value=mock_client):
            from trelix.core.config import LLMConfig
            expander = HyDEExpander(llm_config=LLMConfig())
            result = expander.expand("how does X work")
        assert result == ""


class TestMultiQueryExpander:
    def test_returns_original_when_no_llm(self) -> None:
        expander = MultiQueryExpander(llm_config=None, n=2)
        result = expander.expand("what handles JWT tokens")
        assert result == ["what handles JWT tokens"]

    def test_returns_n_plus_original_when_llm_available(self) -> None:
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content="how is JWT verified\nwhere is token decoded"
        )
        with patch("trelix.retrieval.query_expansion.build_chat_client", return_value=mock_client):
            from trelix.core.config import LLMConfig
            expander = MultiQueryExpander(llm_config=LLMConfig(), n=2)
            result = expander.expand("what handles JWT tokens")
        # Original always included, plus up to n variants
        assert "what handles JWT tokens" in result
        assert len(result) >= 2

    def test_deduplicates_variants(self) -> None:
        mock_client = MagicMock()
        # LLM returns same query as original — should deduplicate
        mock_client.complete.return_value = MagicMock(
            content="what handles JWT tokens\nwhat handles JWT tokens"
        )
        with patch("trelix.retrieval.query_expansion.build_chat_client", return_value=mock_client):
            from trelix.core.config import LLMConfig
            expander = MultiQueryExpander(llm_config=LLMConfig(), n=2)
            result = expander.expand("what handles JWT tokens")
        assert len(result) == len(set(result))  # no duplicates

    def test_config_flags_default_off(self, tmp_path) -> None:
        from trelix.core.config import IndexConfig
        cfg = IndexConfig(repo_path=str(tmp_path))
        assert cfg.retrieval.hyde_fallback_enabled is False
        assert cfg.retrieval.multi_query_enabled is False
        assert cfg.retrieval.multi_query_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/test_query_expansion.py -v --tb=short 2>&1 | head -15
```
Expected: `ModuleNotFoundError: No module named 'trelix.retrieval.query_expansion'`

- [ ] **Step 3: Create `query_expansion.py`**

Create `src/trelix/retrieval/query_expansion.py`:

```python
"""
HyDE and multi-query expansion — zero-shot retrieval improvements.

HyDE (Hypothetical Document Embeddings, Gao et al. 2022, arXiv:2212.10496):
  Instead of embedding the user's NL query, ask an LLM to write a hypothetical
  code snippet that would answer the question, then embed that snippet.
  The encoder's bottleneck filters hallucinated details while preserving semantics.

Multi-query expansion:
  Ask an LLM to rephrase the query N ways. Run each variant as a separate
  sub-query, then RRF-merge. Increases recall on varied vocabulary.

Both are opt-in via RetrievalConfig flags. Both return empty/original on any
failure — the pipeline always has a fallback.
"""

from __future__ import annotations

import logging

from trelix.core.config import LLMConfig

logger = logging.getLogger("trelix.retrieval.query_expansion")

_HYDE_SYSTEM = (
    "You are a senior software engineer. Given a question about a codebase, "
    "write a SHORT hypothetical code snippet (3-8 lines) that would directly answer it. "
    "Output ONLY the code snippet, no explanation, no markdown fences."
)

_MULTI_QUERY_SYSTEM = (
    "You are a search query expert. Given a code search query, write {n} alternative "
    "phrasings that cover different vocabulary but have the same intent. "
    "Output one query per line, no numbering, no explanation."
)


class HyDEExpander:
    """Generate a hypothetical code snippet to use as the vector query (HyDE)."""

    def __init__(self, llm_config: LLMConfig | None) -> None:
        self._llm_config = llm_config
        self._client = None  # lazy-init

    def _get_client(self) -> object | None:
        if self._llm_config is None:
            return None
        if self._client is None:
            try:
                from trelix.llm.factory import build_chat_client
                self._client = build_chat_client(self._llm_config)
            except Exception as exc:
                logger.debug("HyDEExpander: could not build LLM client: %s", exc)
                return None
        return self._client

    def expand(self, query: str) -> str:
        """Return a synthetic code snippet, or '' on any failure."""
        client = self._get_client()
        if client is None:
            return ""
        try:
            from trelix.llm.client import ChatMessage
            resp = client.complete(  # type: ignore[union-attr]
                messages=[ChatMessage(role="user", content=query)],
                max_tokens=150,
                temperature=0.1,
                system=_HYDE_SYSTEM,
            )
            return resp.content.strip()
        except Exception as exc:
            logger.debug("HyDE expansion failed for query %r: %s", query, exc)
            return ""


class MultiQueryExpander:
    """Generate N rephrased variants of the query to increase retrieval recall."""

    def __init__(self, llm_config: LLMConfig | None, n: int = 2) -> None:
        self._llm_config = llm_config
        self._n = n
        self._client = None  # lazy-init

    def _get_client(self) -> object | None:
        if self._llm_config is None:
            return None
        if self._client is None:
            try:
                from trelix.llm.factory import build_chat_client
                self._client = build_chat_client(self._llm_config)
            except Exception as exc:
                logger.debug("MultiQueryExpander: could not build LLM client: %s", exc)
                return None
        return self._client

    def expand(self, query: str) -> list[str]:
        """Return [original] + up to N variants. Deduplicates. Never raises."""
        client = self._get_client()
        if client is None:
            return [query]
        try:
            from trelix.llm.client import ChatMessage
            system = _MULTI_QUERY_SYSTEM.format(n=self._n)
            resp = client.complete(  # type: ignore[union-attr]
                messages=[ChatMessage(role="user", content=query)],
                max_tokens=200,
                temperature=0.3,
                system=system,
            )
            variants = [line.strip() for line in resp.content.strip().splitlines() if line.strip()]
            # Deduplicate while preserving order; original always first
            seen: set[str] = {query}
            result = [query]
            for v in variants[: self._n]:
                if v not in seen:
                    seen.add(v)
                    result.append(v)
            return result
        except Exception as exc:
            logger.debug("Multi-query expansion failed for %r: %s", query, exc)
            return [query]
```

- [ ] **Step 4: Add config fields**

In `src/trelix/core/config.py`, inside `RetrievalConfig`, add after `top_k_file_summary`:

```python
# HyDE fallback — for no-LLM Tier 1 queries, generate a synthetic snippet
# using the LLM before embedding (requires LLM config).
# When the planner already set hyde_snippet, this is skipped (no double-call).
hyde_fallback_enabled: bool = Field(
    default=False,
    alias="TRELIX_RETRIEVAL_HYDE_FALLBACK",
)
# Multi-query expansion — generate N query variants, run each as a sub-query
multi_query_enabled: bool = Field(
    default=False,
    alias="TRELIX_RETRIEVAL_MULTI_QUERY",
)
multi_query_count: int = Field(
    default=2,
    ge=1,
    le=4,
    alias="TRELIX_RETRIEVAL_MULTI_QUERY_COUNT",
)
```

- [ ] **Step 5: Wire into `_run_subquery_legs` in `retriever.py`**

At the top of `_run_subquery_legs`, after the existing embedding logic, add:

```python
# HyDE fallback: if planner left hyde_snippet empty and fallback is enabled,
# generate a synthetic snippet now (single LLM call, result replaces semantic_query embed).
if cfg.hyde_fallback_enabled and not sq.hyde_snippet.strip():
    from trelix.retrieval.query_expansion import HyDEExpander
    snippet = HyDEExpander(self.config.llm).expand(sq.semantic_query)
    if snippet:
        # Override embed_text so the vector leg uses the snippet embedding
        out["_hyde_used"] = True
        embed_text = snippet  # the vector leg already reads embed_text from sq
```

Note: `embed_text` is computed at the start of `_run_subquery_legs` from `sq.hyde_snippet or sq.semantic_query`. You need to ensure the HyDE override happens BEFORE `embed_text` is used for `_vector_search`. Locate the `embed_text` assignment and add the HyDE override immediately after it:

```python
# (existing) embed_text already set from sq.hyde_snippet or sq.semantic_query
if cfg.hyde_fallback_enabled and not sq.hyde_snippet.strip():
    from trelix.retrieval.query_expansion import HyDEExpander
    snippet = HyDEExpander(self.config.llm).expand(sq.semantic_query)
    if snippet:
        embed_text = snippet
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_query_expansion.py -v --tb=short
```
Expected: all 7 tests PASS.

```bash
.venv/bin/python -m pytest tests/unit/ -x -q --tb=short 2>&1 | tail -5
```
Expected: full suite passes.

- [ ] **Step 7: Commit**

```bash
git add src/trelix/retrieval/query_expansion.py src/trelix/core/config.py \
        src/trelix/retrieval/retriever.py tests/unit/test_query_expansion.py
git commit -m "feat(retrieval): add HyDE fallback and multi-query expansion

HyDEExpander wraps LLM to generate synthetic code snippet before embedding.
MultiQueryExpander generates N rephrasings for recall improvement.
Both gated by config flags, crash-safe, no mandatory LLM dependency."
```

---

### Task 3: FLARE-Style Confidence-Gated Re-Retrieval

**What this unlocks:** Single-shot retrieval misses context needed mid-synthesis. FLARE (EMNLP 2023, arXiv:2305.06983) re-triggers retrieval when the LLM's generation entropy exceeds a threshold. This task implements a simplified version: after synthesis, if the answer contains uncertainty markers ("I don't know", "cannot find", "no information"), re-retrieve with a reformulated query and re-synthesize once.

**Files:**
- Create: `src/trelix/retrieval/flare.py` — `FLARELoop` class
- Modify: `src/trelix/core/config.py` — add `flare_enabled`, `flare_uncertainty_phrases`, `flare_max_iterations` to `RetrievalConfig`
- Modify: `src/trelix/cli/main.py` — `ask` command passes through FLARE when enabled
- Test: `tests/unit/test_flare.py`

**Interfaces:**
- Consumes: `Retriever.retrieve(query: str) -> RetrievedContext`, `Synthesizer.synthesize(context, config) -> str`
- Produces: `FLARELoop(retriever, synthesizer, config).run(query: str) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_flare.py`:

```python
"""Tests for FLARE-style confidence-gated re-retrieval."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from trelix.retrieval.flare import FLARELoop, _contains_uncertainty


class TestUncertaintyDetection:
    def test_detects_i_dont_know(self) -> None:
        assert _contains_uncertainty("I don't know how this works.") is True

    def test_detects_cannot_find(self) -> None:
        assert _contains_uncertainty("I cannot find any relevant code for this.") is True

    def test_detects_no_information(self) -> None:
        assert _contains_uncertainty("There is no information about JWT in the codebase.") is True

    def test_confident_answer_not_flagged(self) -> None:
        assert _contains_uncertainty("The authenticate() function in auth.py handles this.") is False

    def test_case_insensitive(self) -> None:
        assert _contains_uncertainty("NO INFORMATION available") is True


class TestFLARELoop:
    def _make_loop(self, first_answer: str, second_answer: str = "Found it.") -> tuple:
        mock_retriever = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.query = "how does auth work"
        mock_retriever.retrieve.return_value = mock_ctx

        mock_synthesizer = MagicMock()
        mock_synthesizer.synthesize.side_effect = [first_answer, second_answer]

        mock_config = MagicMock()
        mock_config.retrieval.flare_enabled = True
        mock_config.retrieval.flare_max_iterations = 1

        loop = FLARELoop(mock_retriever, mock_synthesizer, mock_config)
        return loop, mock_retriever, mock_synthesizer

    def test_no_retry_when_confident(self) -> None:
        loop, mock_retriever, mock_synthesizer = self._make_loop(
            "The auth.py file handles JWT in the login() function."
        )
        result = loop.run("how does auth work")
        assert mock_synthesizer.synthesize.call_count == 1
        assert "auth.py" in result

    def test_retries_once_on_uncertainty(self) -> None:
        loop, mock_retriever, mock_synthesizer = self._make_loop(
            first_answer="I cannot find any relevant code for this.",
            second_answer="The login() function in auth.py handles JWT.",
        )
        result = loop.run("how does auth work")
        assert mock_synthesizer.synthesize.call_count == 2
        assert "login()" in result

    def test_stops_after_max_iterations(self) -> None:
        mock_retriever = MagicMock()
        mock_ctx = MagicMock()
        mock_retriever.retrieve.return_value = mock_ctx
        mock_synthesizer = MagicMock()
        # All answers uncertain
        mock_synthesizer.synthesize.return_value = "I don't know."
        mock_config = MagicMock()
        mock_config.retrieval.flare_enabled = True
        mock_config.retrieval.flare_max_iterations = 2
        loop = FLARELoop(mock_retriever, mock_synthesizer, mock_config)
        result = loop.run("how does auth work")
        # max_iterations=2 means at most 2 synthesis calls (1 initial + 1 retry)
        assert mock_synthesizer.synthesize.call_count <= 2

    def test_config_defaults(self, tmp_path: Path) -> None:
        from trelix.core.config import IndexConfig
        cfg = IndexConfig(repo_path=str(tmp_path))
        assert cfg.retrieval.flare_enabled is False
        assert cfg.retrieval.flare_max_iterations == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_flare.py -v --tb=short 2>&1 | head -15
```
Expected: `ModuleNotFoundError: No module named 'trelix.retrieval.flare'`

- [ ] **Step 3: Create `flare.py`**

Create `src/trelix/retrieval/flare.py`:

```python
"""
FLARE-style confidence-gated re-retrieval loop.

Simplified from: FLARE — Forward-Looking Active REtrieval Augmented Generation
(Jiang et al., EMNLP 2023, arXiv:2305.06983).

Full FLARE monitors token-level log-probabilities. This implementation uses
a simpler but effective heuristic: detect uncertainty phrases in the generated
answer and re-retrieve once with an enriched query.

Usage:
    loop = FLARELoop(retriever, synthesizer, config)
    answer = loop.run("how does the authentication system work?")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.config import IndexConfig
    from trelix.retrieval.retriever import Retriever
    from trelix.retrieval.synthesizer import Synthesizer

logger = logging.getLogger("trelix.retrieval.flare")

# Phrases that signal the model lacks sufficient context — trigger a re-retrieval.
_DEFAULT_UNCERTAINTY_PHRASES: list[str] = [
    "i don't know",
    "i do not know",
    "cannot find",
    "no information",
    "not found in",
    "unable to locate",
    "no relevant code",
    "insufficient context",
    "not enough information",
    "couldn't find",
    "could not find",
]


def _contains_uncertainty(
    text: str,
    phrases: list[str] | None = None,
) -> bool:
    """Return True if text contains any uncertainty marker (case-insensitive)."""
    check = (phrases or _DEFAULT_UNCERTAINTY_PHRASES)
    lower = text.lower()
    return any(phrase in lower for phrase in check)


class FLARELoop:
    """
    Wraps a Retriever + Synthesizer with confidence-gated re-retrieval.

    When the initial synthesis contains uncertainty phrases, re-retrieves
    with a more specific query derived from the original + uncertainty context,
    then re-synthesizes. Runs at most `flare_max_iterations` retries.
    """

    def __init__(
        self,
        retriever: Retriever,
        synthesizer: Synthesizer,
        config: IndexConfig,
    ) -> None:
        self._retriever = retriever
        self._synthesizer = synthesizer
        self._config = config

    def run(self, query: str) -> str:
        """
        Execute the FLARE loop. Returns the final synthesized answer.

        If flare_enabled=False, behaves identically to a single retrieve+synthesize.
        """
        cfg = self._config.retrieval
        ctx = self._retriever.retrieve(query)
        answer = self._synthesizer.synthesize(ctx, self._config.embedder)

        if not cfg.flare_enabled:
            return answer

        iteration = 0
        while (
            _contains_uncertainty(answer)
            and iteration < cfg.flare_max_iterations
        ):
            iteration += 1
            logger.info(
                "FLARE re-retrieval iteration %d/%d for query: %r",
                iteration,
                cfg.flare_max_iterations,
                query[:80],
            )
            # Enrich the query with context about what was missing
            enriched_query = f"{query} (focus on implementation details and concrete code)"
            ctx = self._retriever.retrieve(enriched_query)
            answer = self._synthesizer.synthesize(ctx, self._config.embedder)

        return answer
```

- [ ] **Step 4: Add config fields**

In `src/trelix/core/config.py`, inside `RetrievalConfig`, add after `multi_query_count`:

```python
# FLARE-style confidence-gated re-retrieval
flare_enabled: bool = Field(
    default=False,
    alias="TRELIX_RETRIEVAL_FLARE",
)
flare_max_iterations: int = Field(
    default=1,
    ge=1,
    le=3,
    alias="TRELIX_RETRIEVAL_FLARE_MAX_ITER",
)
```

- [ ] **Step 5: Wire FLARE into `trelix ask` CLI command**

In `src/trelix/cli/main.py`, find the `ask` command implementation. The pattern is `retriever.retrieve(query)` → `synthesizer.synthesize(context, ...)`. Wrap it:

```python
# Before: direct retrieve + synthesize
# After: FLARE loop when enabled
if config.retrieval.flare_enabled:
    from trelix.retrieval.flare import FLARELoop
    from trelix.retrieval.synthesizer import Synthesizer as _Synth
    synth = _Synth(config)
    loop = FLARELoop(retriever, synth, config)
    answer = loop.run(query)
    console.print(answer)
else:
    context = retriever.retrieve(query)
    synth = Synthesizer(config)
    synth.synthesize(context, config.embedder)
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_flare.py -v --tb=short
```
Expected: all 6 tests PASS.

```bash
.venv/bin/python -m pytest tests/unit/ -x -q --tb=short 2>&1 | tail -5
```
Expected: full suite passes.

- [ ] **Step 7: Commit**

```bash
git add src/trelix/retrieval/flare.py src/trelix/core/config.py \
        src/trelix/cli/main.py tests/unit/test_flare.py
git commit -m "feat(retrieval): add FLARE-style confidence-gated re-retrieval

FLARELoop detects uncertainty phrases in synthesis output and re-retrieves
with an enriched query. Gated by TRELIX_RETRIEVAL_FLARE=true.
Based on Jiang et al. EMNLP 2023 (arXiv:2305.06983)."
```

---

## Phase 2 — Graph & Intelligence (Tasks 4–5)

---

### Task 4: PageRank Symbol Importance Scoring

**What this unlocks:** The CodeGraph has 4,599 nodes but all symbols have equal weight in retrieval. PageRank assigns importance scores based on how many other symbols call/import them — the "most central" symbols are typically the ones that matter most architecturally. This task computes PageRank over the CodeGraph, stores centrality in the existing `graph_metadata` table (`centrality` column already exists), and boosts those symbols' RRF scores via the existing file-type weights mechanism.

**Files:**
- Modify: `src/trelix/graph/community.py` — add `compute_pagerank(cg: CodeGraph, alpha: float = 0.85) -> dict[int, float]`
- Modify: `src/trelix/graph/builder.py` — call `compute_pagerank` in `build()`, store via `save_graph_metadata`
- Modify: `src/trelix/graph/persistence.py` — ensure `save_graph_metadata` writes centrality; add `get_top_central_symbols(db, top_n) -> list[int]`
- Modify: `src/trelix/core/config.py` — add `pagerank_boost_enabled`, `pagerank_boost_factor` to `RetrievalConfig`
- Modify: `src/trelix/retrieval/retriever.py` — apply PageRank boost post-rerank
- Test: `tests/unit/test_graph_pagerank.py`

**Interfaces:**
- Produces:
  - `compute_pagerank(cg: CodeGraph, alpha: float = 0.85) -> dict[int, float]` — node_id → normalized PageRank score (0.0–1.0)
  - `get_top_central_symbols(db: Database, top_n: int = 100) -> list[int]` — returns symbol_ids sorted by centrality DESC

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_graph_pagerank.py`:

```python
"""Tests for PageRank symbol importance scoring."""
from __future__ import annotations

from pathlib import Path

import pytest

from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import compute_pagerank
from trelix.graph.persistence import get_top_central_symbols, save_graph_metadata
from trelix.store.db import Database


def _build_star_graph(tmp_path: Path) -> tuple[Database, CodeGraph, int]:
    """Build a star graph: hub calls 3 leaves. Hub should have highest PageRank."""
    db = Database(tmp_path / "index.db")
    fid = db.upsert_file(
        IndexedFile(path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="x", size_bytes=10)
    )
    hub = db.insert_symbol(Symbol(file_id=fid, name="hub", qualified_name="hub",
        kind=SymbolKind.FUNCTION, line_start=1, line_end=5, signature="def hub()", body=""))
    leaf1 = db.insert_symbol(Symbol(file_id=fid, name="leaf1", qualified_name="leaf1",
        kind=SymbolKind.FUNCTION, line_start=10, line_end=14, signature="def leaf1()", body=""))
    leaf2 = db.insert_symbol(Symbol(file_id=fid, name="leaf2", qualified_name="leaf2",
        kind=SymbolKind.FUNCTION, line_start=20, line_end=24, signature="def leaf2()", body=""))
    leaf3 = db.insert_symbol(Symbol(file_id=fid, name="leaf3", qualified_name="leaf3",
        kind=SymbolKind.FUNCTION, line_start=30, line_end=34, signature="def leaf3()", body=""))
    # leaf1, leaf2, leaf3 all call hub (hub is the target, gets PageRank from incoming)
    db.insert_call_edges([
        CallEdge(caller_id=leaf1, callee_name="hub", callee_id=hub, line=11),
        CallEdge(caller_id=leaf2, callee_name="hub", callee_id=hub, line=21),
        CallEdge(caller_id=leaf3, callee_name="hub", callee_id=hub, line=31),
    ])
    cg = CodeGraph(db)
    return db, cg, hub


class TestComputePagerank:
    def test_returns_dict_of_node_scores(self, tmp_path: Path) -> None:
        _, cg, _ = _build_star_graph(tmp_path)
        scores = compute_pagerank(cg)
        assert isinstance(scores, dict)
        assert all(isinstance(v, float) for v in scores.values())

    def test_hub_has_higher_score_than_leaves(self, tmp_path: Path) -> None:
        _, cg, hub_id = _build_star_graph(tmp_path)
        scores = compute_pagerank(cg)
        hub_score = scores.get(hub_id, 0.0)
        leaf_scores = [v for k, v in scores.items() if k != hub_id]
        assert hub_score > max(leaf_scores, default=0.0)

    def test_scores_normalized_0_to_1(self, tmp_path: Path) -> None:
        _, cg, _ = _build_star_graph(tmp_path)
        scores = compute_pagerank(cg)
        assert max(scores.values()) <= 1.0 + 1e-9
        assert min(scores.values()) >= 0.0 - 1e-9

    def test_empty_graph_returns_empty_dict(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        cg = CodeGraph(db)
        assert compute_pagerank(cg) == {}


class TestGetTopCentralSymbols:
    def test_returns_sorted_by_centrality(self, tmp_path: Path) -> None:
        db, cg, hub_id = _build_star_graph(tmp_path)
        scores = compute_pagerank(cg)
        # Assign centrality scores to graph nodes
        for node_id, score in scores.items():
            cg.nx.nodes[node_id]["centrality"] = score
        save_graph_metadata(db, cg)
        top = get_top_central_symbols(db, top_n=1)
        assert len(top) == 1
        assert top[0] == hub_id
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_pagerank.py -v --tb=short 2>&1 | head -20
```
Expected: `ImportError: cannot import name 'compute_pagerank'`

- [ ] **Step 3: Implement `compute_pagerank`**

In `src/trelix/graph/community.py`, add after `get_community_summary`:

```python
def compute_pagerank(cg: "CodeGraph", alpha: float = 0.85) -> dict[int, float]:
    """
    Compute PageRank over the code graph. Returns node_id → normalized score.

    High-PageRank nodes are called/imported by many others — architecturally central.
    Scores are normalized to [0, 1] by dividing by the max score.

    Args:
        cg: CodeGraph instance (networkx MultiDiGraph under the hood)
        alpha: damping factor (default 0.85, standard PageRank value)

    Returns:
        dict[int, float] — empty dict if graph has no edges
    """
    import networkx as nx

    g = cg.nx
    if g.number_of_nodes() == 0:
        return {}

    try:
        raw: dict[int, float] = nx.pagerank(g, alpha=alpha, max_iter=100)
    except nx.PowerIterationFailedConvergence:
        raw = nx.pagerank(g, alpha=alpha, max_iter=500, tol=1e-4)

    # Normalize to [0, 1]
    max_score = max(raw.values()) if raw else 1.0
    if max_score == 0.0:
        return {k: 0.0 for k in raw}
    return {k: v / max_score for k, v in raw.items()}
```

Also update `src/trelix/graph/__init__.py` to export `compute_pagerank`:

```python
from trelix.graph.community import (
    assign_communities,
    compute_pagerank,          # add this
    detect_communities,
    get_community_summary,
)
```

- [ ] **Step 4: Add `get_top_central_symbols` to persistence**

In `src/trelix/graph/persistence.py`, add after `load_graph_metadata`:

```python
def get_top_central_symbols(db: Database, top_n: int = 100) -> list[int]:
    """Return symbol_ids sorted by centrality DESC from graph_metadata table."""
    rows = db._conn.execute(
        "SELECT symbol_id FROM graph_metadata ORDER BY centrality DESC LIMIT ?",
        (top_n,),
    ).fetchall()
    return [int(row[0]) for row in rows]
```

- [ ] **Step 5: Wire into `GraphBuilder.build()`**

In `src/trelix/graph/builder.py`, inside the `build()` method, after `save_graph_metadata(self._db, cg)`:

```python
# Compute and persist PageRank centrality scores
from trelix.graph.community import compute_pagerank
pr_scores = compute_pagerank(cg)
for node_id, score in pr_scores.items():
    if node_id in cg.nx.nodes:
        cg.nx.nodes[node_id]["centrality"] = score
# Re-save metadata now that centrality attrs are set
save_graph_metadata(self._db, cg)
```

- [ ] **Step 6: Add PageRank boost config + wire into retriever**

In `src/trelix/core/config.py`, inside `RetrievalConfig`, add:

```python
# PageRank-based symbol importance boost
pagerank_boost_enabled: bool = Field(
    default=False,
    alias="TRELIX_RETRIEVAL_PAGERANK_BOOST",
)
pagerank_boost_factor: float = Field(
    default=1.3,
    ge=1.0,
    le=3.0,
    alias="TRELIX_RETRIEVAL_PAGERANK_BOOST_FACTOR",
)
```

In `src/trelix/retrieval/retriever.py`, add `_apply_pagerank_boost` method:

```python
def _apply_pagerank_boost(self, results: list[SearchResult]) -> list[SearchResult]:
    """Boost RRF scores for high-centrality symbols (post-rerank, pre-assemble)."""
    cfg = self.config.retrieval
    if not cfg.pagerank_boost_enabled:
        return results
    try:
        from trelix.graph.persistence import get_top_central_symbols
        top_ids = set(get_top_central_symbols(self.db, top_n=200))
        boosted: list[SearchResult] = []
        for r in results:
            if r.symbol.id in top_ids:
                boosted.append(
                    SearchResult(
                        chunk=r.chunk,
                        symbol=r.symbol,
                        file=r.file,
                        score=r.score * cfg.pagerank_boost_factor,
                        rank=r.rank,
                        source=r.source,
                    )
                )
            else:
                boosted.append(r)
        return sorted(boosted, key=lambda x: x.score, reverse=True)
    except Exception as exc:
        logger.debug("PageRank boost failed (non-fatal): %s", exc)
        return results
```

Call it in `_retrieve_standard` after reranking and before `_assemble`:

```python
candidates = self._apply_pagerank_boost(candidates)
```

- [ ] **Step 7: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_pagerank.py -v --tb=short
```
Expected: all 5 tests PASS.

```bash
.venv/bin/python -m pytest tests/unit/ -x -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 8: Commit**

```bash
git add src/trelix/graph/community.py src/trelix/graph/__init__.py \
        src/trelix/graph/persistence.py src/trelix/graph/builder.py \
        src/trelix/core/config.py src/trelix/retrieval/retriever.py \
        tests/unit/test_graph_pagerank.py
git commit -m "feat(graph): add PageRank symbol importance scoring and retrieval boost

compute_pagerank() over CodeGraph (networkx), stored in graph_metadata.centrality.
TRELIX_RETRIEVAL_PAGERANK_BOOST=true boosts high-centrality symbols post-rerank."
```

---

### Task 5: Incremental Graph Updates (Watcher Integration)

**What this unlocks:** `trelix watch` monitors file changes via watchdog, but it only re-indexes symbols/chunks — the CodeGraph and graph_metadata are never updated incrementally. After re-indexing a file, graph nodes/edges for that file become stale. This task adds `GraphUpdater` that re-builds only the affected sub-graph after a file change.

**Files:**
- Create: `src/trelix/graph/updater.py` — `GraphUpdater` class
- Modify: `src/trelix/indexing/watcher.py` — call `GraphUpdater.update_file(file_path)` after re-indexing
- Test: `tests/unit/test_graph_updater.py`

**Interfaces:**
- Produces: `GraphUpdater(db: Database).update_file(rel_path: str) -> None` — rebuilds CodeGraph for one file and re-saves metadata

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_graph_updater.py`:

```python
"""Tests for incremental graph updates after file changes."""
from __future__ import annotations

from pathlib import Path

import pytest

from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.updater import GraphUpdater
from trelix.store.db import Database


def _make_db(tmp_path: Path) -> tuple[Database, int, int]:
    db = Database(tmp_path / "index.db")
    fid = db.upsert_file(
        IndexedFile(path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="h1", size_bytes=50)
    )
    sid = db.insert_symbol(Symbol(
        file_id=fid, name="my_func", qualified_name="my_func",
        kind=SymbolKind.FUNCTION, line_start=1, line_end=10,
        signature="def my_func()", body="def my_func(): pass"
    ))
    return db, fid, sid


class TestGraphUpdater:
    def test_update_file_does_not_raise_on_valid_file(self, tmp_path: Path) -> None:
        db, fid, sid = _make_db(tmp_path)
        updater = GraphUpdater(db)
        # Should complete without error
        updater.update_file("a.py")

    def test_update_file_on_unknown_file_is_noop(self, tmp_path: Path) -> None:
        db, _, _ = _make_db(tmp_path)
        updater = GraphUpdater(db)
        # Should not raise even if file not found
        updater.update_file("nonexistent.py")

    def test_graph_metadata_refreshed_after_update(self, tmp_path: Path) -> None:
        db, fid, sid = _make_db(tmp_path)
        updater = GraphUpdater(db)
        updater.update_file("a.py")
        # graph_metadata should have an entry for the symbol
        row = db._conn.execute(
            "SELECT symbol_id FROM graph_metadata WHERE symbol_id = ?", (sid,)
        ).fetchone()
        assert row is not None
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_updater.py -v --tb=short 2>&1 | head -15
```
Expected: `ModuleNotFoundError: No module named 'trelix.graph.updater'`

- [ ] **Step 3: Create `updater.py`**

Create `src/trelix/graph/updater.py`:

```python
"""
Incremental graph updater — rebuilds the CodeGraph sub-graph for a single file
after that file has been re-indexed by the watcher.

Only rebuilds nodes/edges touching the changed file, then re-saves graph_metadata
(community + centrality) for those nodes. This avoids rebuilding the full graph
on every file-save event.
"""
from __future__ import annotations

import logging

from trelix.store.db import Database

logger = logging.getLogger("trelix.graph.updater")


class GraphUpdater:
    """Lightweight incremental updater for the CodeGraph after file changes."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def update_file(self, rel_path: str) -> None:
        """
        Rebuild graph nodes/edges for `rel_path` and refresh graph_metadata.

        Safe to call even if the file isn't indexed — no-op in that case.
        Rebuilds the full CodeGraph (fast: NetworkX over SQLite reads, ~50ms
        for typical repos) and re-saves metadata. Full rebuild is simpler than
        partial updates and avoids stale-edge bugs when call targets change.
        """
        try:
            from trelix.graph.code_graph import CodeGraph
            from trelix.graph.community import assign_communities, compute_pagerank, detect_communities
            from trelix.graph.persistence import save_graph_metadata

            cg = CodeGraph(self._db)
            if cg.node_count == 0:
                return

            # Re-run community detection and PageRank
            communities = detect_communities(cg)
            assign_communities(cg, communities)
            pr_scores = compute_pagerank(cg)
            for node_id, score in pr_scores.items():
                if node_id in cg.nx.nodes:
                    cg.nx.nodes[node_id]["centrality"] = score

            save_graph_metadata(self._db, cg)
            logger.debug("Graph metadata refreshed after change to %s", rel_path)
        except Exception as exc:
            logger.warning("GraphUpdater.update_file(%r) failed (non-fatal): %s", rel_path, exc)
```

- [ ] **Step 4: Wire into watcher**

In `src/trelix/indexing/watcher.py`, find the method that handles file change events (after re-indexing a file). Add after the indexer call:

```python
# Refresh graph metadata incrementally
if self._config.file_summaries_enabled or True:  # always refresh graph if it exists
    try:
        from trelix.graph.updater import GraphUpdater
        GraphUpdater(self._indexer.db).update_file(rel_path)
    except Exception as exc:
        logger.debug("GraphUpdater watcher hook failed (non-fatal): %s", exc)
```

- [ ] **Step 5: Export from graph `__init__`**

In `src/trelix/graph/__init__.py`, add:
```python
from trelix.graph.updater import GraphUpdater
```

And add `"GraphUpdater"` to `__all__` if present.

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_graph_updater.py -v --tb=short
.venv/bin/python -m pytest tests/unit/ -x -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add src/trelix/graph/updater.py src/trelix/graph/__init__.py \
        src/trelix/indexing/watcher.py tests/unit/test_graph_updater.py
git commit -m "feat(graph): add incremental graph updater for watcher integration

GraphUpdater.update_file() rebuilds CodeGraph metadata after file changes.
Hooked into trelix watch so community/PageRank stays fresh on edit."
```

---

## Phase 3 — Observability & Evaluation (Tasks 6–7)

---

### Task 6: Query Telemetry + Feedback Loop

**What this unlocks:** Zero observability means no way to know which queries fail, which retrieval legs contribute, or how to improve. This task adds: (a) a persistent `query_telemetry` SQLite table that stores every query + retrieval stats + synthesis outcome; (b) a `trelix eval` CLI subcommand that runs a golden-set of queries against the live index and reports nDCG@10; (c) optional thumbs-up/thumbs-down feedback written back to the telemetry table for future analysis.

**Files:**
- Modify: `src/trelix/store/db.py` — add `query_telemetry` table schema + `insert_query_telemetry()`, `get_recent_telemetry()` methods
- Create: `src/trelix/retrieval/telemetry.py` — `TelemetryWriter` class
- Modify: `src/trelix/retrieval/retriever.py` — call `TelemetryWriter.record()` at end of `retrieve()`
- Modify: `src/trelix/cli/main.py` — add `trelix eval` subcommand
- Test: `tests/unit/test_telemetry.py`

**Interfaces:**
- Produces:
  - `TelemetryWriter(db: Database, enabled: bool = True).record(query, context, elapsed_ms) -> None`
  - `Database.insert_query_telemetry(query, intent, elapsed_ms, result_count, leg_sizes) -> int`
  - `Database.get_recent_telemetry(limit: int = 100) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_telemetry.py`:

```python
"""Tests for query telemetry recording."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trelix.retrieval.telemetry import TelemetryWriter
from trelix.store.db import Database


class TestTelemetryWriter:
    def test_record_inserts_row(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        writer = TelemetryWriter(db, enabled=True)
        mock_ctx = MagicMock()
        mock_ctx.query = "how does auth work"
        mock_ctx.intent = "code_search"
        mock_ctx.results = [MagicMock(), MagicMock()]
        writer.record(mock_ctx, elapsed_ms=42.5)
        rows = db.get_recent_telemetry(limit=10)
        assert len(rows) == 1
        assert rows[0]["query"] == "how does auth work"
        assert rows[0]["elapsed_ms"] == pytest.approx(42.5)
        assert rows[0]["result_count"] == 2

    def test_record_noop_when_disabled(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        writer = TelemetryWriter(db, enabled=False)
        mock_ctx = MagicMock()
        mock_ctx.query = "test"
        mock_ctx.results = []
        writer.record(mock_ctx, elapsed_ms=10.0)
        rows = db.get_recent_telemetry(limit=10)
        assert len(rows) == 0

    def test_record_never_raises(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        writer = TelemetryWriter(db, enabled=True)
        # Malformed context should not crash
        writer.record(MagicMock(spec=[]), elapsed_ms=0.0)

    def test_telemetry_disabled_by_default(self, tmp_path: Path) -> None:
        from trelix.core.config import IndexConfig
        cfg = IndexConfig(repo_path=str(tmp_path))
        assert cfg.telemetry_enabled is False
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_telemetry.py -v --tb=short 2>&1 | head -15
```
Expected: `ModuleNotFoundError: No module named 'trelix.retrieval.telemetry'`

- [ ] **Step 3: Add `query_telemetry` table to `Database`**

In `src/trelix/store/db.py`, in the `_setup_schema()` method, add after the `file_summaries` table creation:

```python
self._conn.execute("""
    CREATE TABLE IF NOT EXISTS query_telemetry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL DEFAULT (datetime('now')),
        query TEXT NOT NULL,
        intent TEXT DEFAULT '',
        elapsed_ms REAL DEFAULT 0.0,
        result_count INTEGER DEFAULT 0,
        leg_sizes TEXT DEFAULT '{}',
        thumbs_up INTEGER DEFAULT NULL
    )
""")
```

Add the methods to `Database`:

```python
def insert_query_telemetry(
    self,
    query: str,
    intent: str,
    elapsed_ms: float,
    result_count: int,
    leg_sizes: dict[str, int] | None = None,
) -> int:
    """Insert one telemetry row. Returns row id."""
    import json
    cur = self._conn.execute(
        "INSERT INTO query_telemetry (query, intent, elapsed_ms, result_count, leg_sizes) "
        "VALUES (?, ?, ?, ?, ?)",
        (query, intent, elapsed_ms, result_count, json.dumps(leg_sizes or {})),
    )
    self._conn.commit()
    return int(cur.lastrowid or 0)

def get_recent_telemetry(self, limit: int = 100) -> list[dict]:
    """Return most recent telemetry rows as list of dicts."""
    import json
    rows = self._conn.execute(
        "SELECT id, ts, query, intent, elapsed_ms, result_count, leg_sizes, thumbs_up "
        "FROM query_telemetry ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {
            "id": int(row[0]),
            "ts": row[1],
            "query": row[2],
            "intent": row[3],
            "elapsed_ms": float(row[4]),
            "result_count": int(row[5]),
            "leg_sizes": json.loads(row[6] or "{}"),
            "thumbs_up": row[7],
        }
        for row in rows
    ]
```

- [ ] **Step 4: Create `telemetry.py`**

Create `src/trelix/retrieval/telemetry.py`:

```python
"""
Query telemetry writer — records retrieve() calls to the query_telemetry table.

Off by default (telemetry_enabled=False). When enabled, every retrieve() call
appends one row: query text, intent, elapsed_ms, result count.
Used for: debugging slow queries, tracking improvement over time, computing
nDCG@10 against a golden set via `trelix eval`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.models import RetrievedContext
    from trelix.store.db import Database

logger = logging.getLogger("trelix.retrieval.telemetry")


class TelemetryWriter:
    """Write query telemetry to the query_telemetry table. Never raises."""

    def __init__(self, db: Database, enabled: bool = True) -> None:
        self._db = db
        self._enabled = enabled

    def record(self, context: RetrievedContext, elapsed_ms: float) -> None:
        """Record a single retrieve() call. No-op when disabled."""
        if not self._enabled:
            return
        try:
            query = str(getattr(context, "query", ""))
            intent = str(getattr(context, "intent", ""))
            results = getattr(context, "results", [])
            result_count = len(results) if hasattr(results, "__len__") else 0
            self._db.insert_query_telemetry(
                query=query,
                intent=intent,
                elapsed_ms=elapsed_ms,
                result_count=result_count,
            )
        except Exception as exc:
            logger.debug("Telemetry record failed (non-fatal): %s", exc)
```

- [ ] **Step 5: Add `telemetry_enabled` config field**

In `src/trelix/core/config.py`, inside `IndexConfig` (after `file_summaries_enabled`):

```python
telemetry_enabled: bool = Field(
    default=False,
    alias="TRELIX_TELEMETRY_ENABLED",
)
```

- [ ] **Step 6: Wire into `Retriever.retrieve()`**

In `src/trelix/retrieval/retriever.py`, at the end of the `retrieve()` method, before the `return` statement:

```python
# Telemetry — record timing and result count (no-op when disabled)
if self.config.telemetry_enabled:
    from trelix.retrieval.telemetry import TelemetryWriter
    elapsed_ms = (time.perf_counter() - _t0) * 1000
    TelemetryWriter(self.db, enabled=True).record(context, elapsed_ms=elapsed_ms)
```

You also need to capture `_t0 = time.perf_counter()` at the start of `retrieve()`.

- [ ] **Step 7: Add `trelix telemetry` CLI subcommand**

In `src/trelix/cli/main.py`, add a new `telemetry` command:

```python
@app.command()
def telemetry(
    repo: Annotated[str, typer.Argument(help="Path to the indexed repository.")] = ".",
    limit: Annotated[int, typer.Option("--limit", "-n", help="Rows to show")] = 20,
) -> None:
    """Show recent query telemetry (latency, result counts, intent breakdown)."""
    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

    config = IndexConfig(repo_path=repo)
    db = Database(config.db_path_absolute)
    rows = db.get_recent_telemetry(limit=limit)

    if not rows:
        console.print("[yellow]No telemetry recorded. Set TRELIX_TELEMETRY_ENABLED=true and run queries.[/yellow]")
        return

    from rich.table import Table
    table = Table(title=f"Recent Queries (last {len(rows)})")
    table.add_column("ts", style="dim")
    table.add_column("query", max_width=50)
    table.add_column("intent")
    table.add_column("ms", justify="right")
    table.add_column("results", justify="right")

    for row in rows:
        table.add_row(
            row["ts"],
            row["query"][:50],
            row["intent"],
            f"{row['elapsed_ms']:.0f}",
            str(row["result_count"]),
        )
    console.print(table)
```

- [ ] **Step 8: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_telemetry.py -v --tb=short
.venv/bin/python -m pytest tests/unit/ -x -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 9: Commit**

```bash
git add src/trelix/store/db.py src/trelix/retrieval/telemetry.py \
        src/trelix/retrieval/retriever.py src/trelix/core/config.py \
        src/trelix/cli/main.py tests/unit/test_telemetry.py
git commit -m "feat(telemetry): add query telemetry table and TelemetryWriter

Records every retrieve() call to query_telemetry SQLite table when
TRELIX_TELEMETRY_ENABLED=true. Adds 'trelix telemetry' CLI command.
Zero overhead when disabled (no-op path)."
```

---

### Task 7: CoIR Evaluation Harness + Version Bump to v3.0.0

**What this unlocks:** No evaluation harness = no way to measure whether upgrades actually improve retrieval. This task adds a `trelix eval` command that runs queries from a golden JSONL file and reports nDCG@10, Recall@10, and MRR. The golden format is compatible with the CoIR benchmark (arXiv:2407.02883, ACL 2025) query format.

**Files:**
- Create: `src/trelix/eval/__init__.py`
- Create: `src/trelix/eval/ndcg.py` — `ndcg_at_k()`, `recall_at_k()`, `mrr()` pure functions
- Create: `src/trelix/eval/harness.py` — `EvalHarness` class
- Modify: `src/trelix/cli/main.py` — add `trelix eval` subcommand
- Modify: `pyproject.toml` + `src/trelix/__init__.py` + `packages/trelix-mcp/src/trelix_mcp/__init__.py` — bump to `"3.0.0"`
- Test: `tests/unit/test_eval_harness.py`

**Interfaces:**
- Produces:
  - `ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int = 10) -> float`
  - `recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int = 10) -> float`
  - `mrr(ranked_ids: list[int], relevant_ids: set[int]) -> float`
  - `EvalHarness(config: IndexConfig).run(golden_path: str) -> dict[str, float]`

Golden JSONL format (one JSON object per line):
```json
{"query": "how does JWT authentication work", "relevant_files": ["src/auth.py", "src/jwt.py"]}
```

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_eval_harness.py`:

```python
"""Tests for CoIR-style evaluation harness and metrics."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trelix.eval.ndcg import mrr, ndcg_at_k, recall_at_k


class TestNdcgAtK:
    def test_perfect_ranking(self) -> None:
        ranked = [1, 2, 3, 4, 5]
        relevant = {1, 2}
        score = ndcg_at_k(ranked, relevant, k=5)
        assert score == pytest.approx(1.0)

    def test_no_relevant_in_top_k(self) -> None:
        ranked = [10, 11, 12]
        relevant = {99}
        assert ndcg_at_k(ranked, relevant, k=3) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        ranked = [1, 10, 2, 11, 12]
        relevant = {1, 2}
        score = ndcg_at_k(ranked, relevant, k=5)
        assert 0.0 < score < 1.0

    def test_empty_relevant(self) -> None:
        assert ndcg_at_k([1, 2, 3], set(), k=3) == pytest.approx(0.0)


class TestRecallAtK:
    def test_all_relevant_found(self) -> None:
        assert recall_at_k([1, 2, 3], {1, 2}, k=3) == pytest.approx(1.0)

    def test_none_found(self) -> None:
        assert recall_at_k([10, 11], {1}, k=2) == pytest.approx(0.0)

    def test_partial(self) -> None:
        assert recall_at_k([1, 10, 11], {1, 2}, k=3) == pytest.approx(0.5)


class TestMRR:
    def test_first_hit_at_rank_1(self) -> None:
        assert mrr([1, 2, 3], {1}) == pytest.approx(1.0)

    def test_first_hit_at_rank_2(self) -> None:
        assert mrr([10, 1, 2], {1}) == pytest.approx(0.5)

    def test_no_hit(self) -> None:
        assert mrr([10, 11, 12], {1}) == pytest.approx(0.0)


class TestEvalHarness:
    def test_run_returns_metrics_dict(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch
        from trelix.eval.harness import EvalHarness
        from trelix.core.config import IndexConfig

        golden = tmp_path / "golden.jsonl"
        golden.write_text(
            json.dumps({"query": "how does auth work", "relevant_files": ["auth.py"]}) + "\n"
        )

        mock_ctx = MagicMock()
        mock_result = MagicMock()
        mock_result.file.rel_path = "auth.py"
        mock_ctx.results = [mock_result]

        config = IndexConfig(repo_path=str(tmp_path))
        harness = EvalHarness(config)

        with patch.object(harness, "_retriever") as mock_r:
            mock_r.retrieve.return_value = mock_ctx
            metrics = harness.run(str(golden))

        assert "ndcg@10" in metrics
        assert "recall@10" in metrics
        assert "mrr" in metrics
        assert 0.0 <= metrics["ndcg@10"] <= 1.0
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/python -m pytest tests/unit/test_eval_harness.py -v --tb=short 2>&1 | head -15
```

- [ ] **Step 3: Create eval module**

Create `src/trelix/eval/__init__.py`:
```python
"""trelix evaluation harness — CoIR-compatible nDCG@10 + Recall@10 + MRR."""
```

Create `src/trelix/eval/ndcg.py`:

```python
"""
Pure metric functions for retrieval evaluation.

All functions are stateless, dependency-free, and O(k log k).
Compatible with CoIR benchmark format (arXiv:2407.02883).
"""
from __future__ import annotations

import math


def ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int = 10) -> float:
    """
    Compute nDCG@k.

    Args:
        ranked_ids: list of retrieved IDs in rank order (best first)
        relevant_ids: set of relevant (ground-truth) IDs
        k: cutoff

    Returns:
        nDCG@k score in [0, 1]
    """
    if not relevant_ids:
        return 0.0

    def dcg(ids: list[int], rel: set[int], k: int) -> float:
        return sum(
            1.0 / math.log2(rank + 2)
            for rank, doc_id in enumerate(ids[:k])
            if doc_id in rel
        )

    actual = dcg(ranked_ids, relevant_ids, k)
    # Ideal: all relevant docs at top positions
    ideal_ranked = list(relevant_ids)[:k]
    ideal = dcg(ideal_ranked, relevant_ids, k)
    return actual / ideal if ideal > 0 else 0.0


def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int = 10) -> float:
    """Fraction of relevant documents found in top-k."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mrr(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    """Mean Reciprocal Rank — reciprocal of the first relevant rank."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0
```

Create `src/trelix/eval/harness.py`:

```python
"""
EvalHarness — run a golden JSONL file through trelix retrieval and report metrics.

Golden file format (one JSON object per line):
    {"query": "how does JWT auth work", "relevant_files": ["src/auth.py"]}

Usage:
    harness = EvalHarness(config)
    metrics = harness.run("golden.jsonl")
    # -> {"ndcg@10": 0.74, "recall@10": 0.81, "mrr": 0.66, "n_queries": 12}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from trelix.core.config import IndexConfig
from trelix.eval.ndcg import mrr, ndcg_at_k, recall_at_k

logger = logging.getLogger("trelix.eval")


class EvalHarness:
    def __init__(self, config: IndexConfig) -> None:
        self._config = config
        from trelix.retrieval.retriever import Retriever
        self._retriever = Retriever(config)

    def run(self, golden_path: str) -> dict[str, float]:
        """
        Run all queries in the golden file and return aggregate metrics.

        Returns dict with keys: ndcg@10, recall@10, mrr, n_queries.
        """
        path = Path(golden_path)
        if not path.exists():
            raise FileNotFoundError(f"Golden file not found: {golden_path}")

        queries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not queries:
            return {"ndcg@10": 0.0, "recall@10": 0.0, "mrr": 0.0, "n_queries": 0}

        ndcg_scores: list[float] = []
        recall_scores: list[float] = []
        mrr_scores: list[float] = []

        for item in queries:
            query = item["query"]
            relevant_files: set[str] = set(item.get("relevant_files", []))
            if not relevant_files:
                continue

            try:
                ctx = self._retriever.retrieve(query)
            except Exception as exc:
                logger.warning("Query %r failed: %s", query[:60], exc)
                ndcg_scores.append(0.0)
                recall_scores.append(0.0)
                mrr_scores.append(0.0)
                continue

            # Use file rel_path as the ID for matching
            ranked_files = [r.file.rel_path for r in ctx.results]
            # Convert to integer IDs for metric functions (hash-based)
            file_to_id = {f: i for i, f in enumerate(set(ranked_files) | relevant_files)}
            ranked_ids = [file_to_id[f] for f in ranked_files]
            relevant_ids = {file_to_id[f] for f in relevant_files if f in file_to_id}

            ndcg_scores.append(ndcg_at_k(ranked_ids, relevant_ids, k=10))
            recall_scores.append(recall_at_k(ranked_ids, relevant_ids, k=10))
            mrr_scores.append(mrr(ranked_ids, relevant_ids))

        n = len(ndcg_scores)
        if n == 0:
            return {"ndcg@10": 0.0, "recall@10": 0.0, "mrr": 0.0, "n_queries": 0}

        return {
            "ndcg@10": sum(ndcg_scores) / n,
            "recall@10": sum(recall_scores) / n,
            "mrr": sum(mrr_scores) / n,
            "n_queries": float(n),
        }
```

- [ ] **Step 4: Add `trelix eval` CLI command**

In `src/trelix/cli/main.py`, add:

```python
@app.command()
def eval(
    repo: Annotated[str, typer.Argument(help="Path to the indexed repository.")] = ".",
    golden: Annotated[str, typer.Option("--golden", "-g", help="Path to golden JSONL file.")] = ".trelix/golden.jsonl",
) -> None:
    """Evaluate retrieval quality against a golden query set (nDCG@10, Recall@10, MRR)."""
    from trelix.core.config import IndexConfig
    from trelix.eval.harness import EvalHarness

    config = IndexConfig(repo_path=repo)
    harness = EvalHarness(config)
    try:
        metrics = harness.run(golden)
    except FileNotFoundError:
        console.print(f"[red]Golden file not found: {golden}[/red]")
        console.print("Create a golden.jsonl with lines like:")
        console.print('  {"query": "how does auth work", "relevant_files": ["src/auth.py"]}')
        raise typer.Exit(1)

    from rich.table import Table
    table = Table(title="Retrieval Evaluation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_row("nDCG@10", f"{metrics['ndcg@10']:.4f}")
    table.add_row("Recall@10", f"{metrics['recall@10']:.4f}")
    table.add_row("MRR", f"{metrics['mrr']:.4f}")
    table.add_row("Queries evaluated", str(int(metrics["n_queries"])))
    console.print(table)
```

- [ ] **Step 5: Bump version to 3.0.0**

Edit `pyproject.toml` line `version = "2.0.0"` → `version = "3.0.0"`

Edit `src/trelix/__init__.py` — change `__version__ = "2.0.0"` → `__version__ = "3.0.0"`

Edit `packages/trelix-mcp/src/trelix_mcp/__init__.py` — change `__version__ = "2.0.0"` → `__version__ = "3.0.0"`

Also update `CHANGELOG.md` — add `## [3.0.0] — 2026-06-30` section with all features from Tasks 1–7.

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_eval_harness.py -v --tb=short
.venv/bin/python -m pytest tests/unit/ -x -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 7: Run full lint + typecheck**

```bash
.venv/bin/python -m ruff check src/ tests/
.venv/bin/python -m ruff format src/ tests/
.venv/bin/python -m mypy src/trelix/ --ignore-missing-imports 2>&1 | tail -10
```

Fix any ruff/mypy issues before committing.

- [ ] **Step 8: Commit**

```bash
git add src/trelix/eval/ src/trelix/cli/main.py \
        pyproject.toml src/trelix/__init__.py \
        packages/trelix-mcp/src/trelix_mcp/__init__.py \
        CHANGELOG.md tests/unit/test_eval_harness.py
git commit -m "feat(eval): add CoIR-compatible evaluation harness + version bump to 3.0.0

EvalHarness runs golden JSONL queries and reports nDCG@10, Recall@10, MRR.
Pure metric functions in eval/ndcg.py (no pandas dependency).
'trelix eval' CLI command. Version bumped to 3.0.0."
```

---

## Verification (run after all tasks complete)

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix

# 1. Full test suite
.venv/bin/python -m pytest tests/unit/ -q --tb=short 2>&1 | tail -5
# Expected: all pass (no new failures)

# 2. New features smoke test
.venv/bin/python -c "
from trelix.retrieval.query_expansion import HyDEExpander, MultiQueryExpander
from trelix.retrieval.flare import FLARELoop, _contains_uncertainty
from trelix.retrieval.telemetry import TelemetryWriter
from trelix.graph.community import compute_pagerank
from trelix.graph.updater import GraphUpdater
from trelix.eval.ndcg import ndcg_at_k, recall_at_k, mrr
from trelix.eval.harness import EvalHarness
print('All new modules importable ✓')
"

# 3. Config flags all default to off
.venv/bin/python -c "
import tempfile, os
with tempfile.TemporaryDirectory() as d:
    from trelix.core.config import IndexConfig
    c = IndexConfig(repo_path=d)
    assert c.retrieval.file_summary_leg_enabled is False
    assert c.retrieval.hyde_fallback_enabled is False
    assert c.retrieval.multi_query_enabled is False
    assert c.retrieval.flare_enabled is False
    assert c.retrieval.pagerank_boost_enabled is False
    assert c.telemetry_enabled is False
    print('All flags default OFF ✓')
"

# 4. Version correct
.venv/bin/python -c "import trelix; assert trelix.__version__ == '3.0.0'; print('Version 3.0.0 ✓')"

# 5. CLI smoke
.venv/bin/python -m trelix --help | grep -E "eval|telemetry"
# Expected: eval and telemetry subcommands listed

# 6. nDCG metric correctness
.venv/bin/python -c "
from trelix.eval.ndcg import ndcg_at_k
# Perfect ranking: relevant docs at positions 1 and 2
score = ndcg_at_k([1, 2, 3, 4, 5], {1, 2}, k=5)
assert abs(score - 1.0) < 0.001, f'Expected 1.0, got {score}'
print('nDCG perfect ranking ✓')
"

# 7. Lint
.venv/bin/python -m ruff check src/ tests/ --quiet
```

---

## Daily Log Entry

Add to `scratch-pad/daily-logs/2026-06-30.md`:

```markdown
## trelix beast-mode plan written

- Plan covers 7 tasks across 3 phases
- All research-backed: HyDE (arXiv:2212.10496), FLARE (arXiv:2305.06983),
  CoIR harness (ACL 2025), PageRank centrality (NetworkX)
- All features gated by config flags — zero regression when disabled
- Version bumps to 3.0.0 in Task 7 only
```
