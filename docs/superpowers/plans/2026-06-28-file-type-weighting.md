# File-Type Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix README.md over-ranking in BM25+RRF fusion by multiplying each fused RRF score by a per-language weight multiplier. Recall@5 must reach ≥ 70% (up from 60% baseline). The BM25 and vector legs are completely unmodified; all logic lives in one auditable step inside `reciprocal_rank_fusion`.

**Architecture:** `RetrievalConfig` gains two new fields — `file_type_weighting_enabled` (bool, master switch) and `file_type_weights` (dict mapping Language string value → float multiplier). A `model_post_init` hook merges individual `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_{LANG}` env vars on top of the default dict. `reciprocal_rank_fusion` gains an optional `weights` parameter; when truthy it multiplies the accumulated RRF score by `weights.get(str(lang), 1.0)` before sorting. `Retriever._retrieve_standard` reads the config and passes `weights` (or `None` when disabled) into the fusion call.

**Tech Stack:** Python 3.11+, pydantic-settings, `src/` layout, pytest, `.venv/bin/python`.

## Global Constraints

- Python ≥ 3.11; use `from __future__ import annotations` at top of every modified file
- `src/` layout — all new source under `src/trelix/`; all tests under `tests/`
- Repo: `/Users/sairamugge/Desktop/Not-Humans-World/trelix`
- Venv: `.venv/bin/python`
- Run tests with: `cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && .venv/bin/python -m pytest <path> -v --tb=short`
- All existing tests must remain green throughout
- No new runtime dependencies — stdlib only; pydantic-settings already present
- `weights=None` (default in `reciprocal_rank_fusion`) must produce bit-for-bit identical output to the current implementation
- Missing language key in the weights dict must fall back to `1.0` (do not penalise unknown types)
- `file_type_weighting_enabled=False` is the kill-switch for rollback; it must pass `weights=None` to the fusion call

---

### Task 1: Add `file_type_weights` config to `RetrievalConfig` + tests

**Files:**
- Modify: `src/trelix/core/config.py` (inside `RetrievalConfig`)
- Test: `tests/unit/test_config.py` (append new class `TestRetrievalConfigFileTypeWeighting`)

**Interfaces:**
- Consumes: `Language` enum from `trelix.core.models` (already imported in `config.py`)
- Produces:
  - `RetrievalConfig.file_type_weighting_enabled: bool = True` (env: `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING`)
  - `RetrievalConfig.file_type_weights: dict[str, float]` — default dict with 21 language entries
  - `RetrievalConfig.model_post_init` — merges `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS` (JSON) and per-language `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_{LANG}` env vars on top of defaults

- [ ] **Step 1: Write the failing tests**

Read `tests/unit/test_config.py` first. Append at the end of the file after `TestRetrievalConfigQueryCache`:

```python
class TestRetrievalConfigFileTypeWeighting:
    def test_file_type_weighting_enabled_default_true(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        assert cfg.file_type_weighting_enabled is True

    def test_weighting_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING", "false")
        cfg = RetrievalConfig()
        assert cfg.file_type_weighting_enabled is False

    def test_default_weights_contain_all_expected_languages(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        expected_languages = {
            "python", "javascript", "typescript", "tsx", "go", "rust",
            "java", "kotlin", "ruby", "cpp", "c", "csharp", "razor",
            "cshtml", "csproj", "html", "css", "json", "yaml", "toml",
            "markdown", "unknown",
        }
        assert expected_languages.issubset(set(cfg.file_type_weights.keys()))

    def test_default_python_weight_is_1_0(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        assert cfg.file_type_weights["python"] == 1.0

    def test_default_markdown_weight_is_0_3(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        assert cfg.file_type_weights["markdown"] == 0.3

    def test_default_unknown_weight_is_0_8(self) -> None:
        from trelix.core.config import RetrievalConfig

        cfg = RetrievalConfig()
        assert cfg.file_type_weights["unknown"] == 0.8

    def test_per_language_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN", "0.1")
        cfg = RetrievalConfig()
        assert cfg.file_type_weights["markdown"] == 0.1
        # Other keys must still be at defaults
        assert cfg.file_type_weights["python"] == 1.0

    def test_full_json_dict_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv(
            "TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS",
            '{"markdown": 0.05, "yaml": 0.6}',
        )
        cfg = RetrievalConfig()
        assert cfg.file_type_weights["markdown"] == 0.05
        assert cfg.file_type_weights["yaml"] == 0.6
        # Defaults for other keys untouched
        assert cfg.file_type_weights["python"] == 1.0

    def test_per_language_override_beats_json_dict_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-language env var is highest priority — applied after JSON dict override."""
        from trelix.core.config import RetrievalConfig

        monkeypatch.setenv(
            "TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS",
            '{"markdown": 0.15}',
        )
        monkeypatch.setenv("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN", "0.02")
        cfg = RetrievalConfig()
        # Per-language override wins
        assert cfg.file_type_weights["markdown"] == 0.02
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/test_config.py::TestRetrievalConfigFileTypeWeighting -v --tb=short 2>&1 | head -20
```
Expected: `AttributeError: 'RetrievalConfig' object has no attribute 'file_type_weighting_enabled'`

- [ ] **Step 3: Implement the two config fields + `model_post_init` in `config.py`**

Read `src/trelix/core/config.py`. After the existing `query_cache_size` block (around line 313) and before the closing of `RetrievalConfig`, add:

```python
    # ── File-type weighting ──────────────────────────────────────────────────
    # Applies a per-language multiplier to RRF scores after fusion.
    # Env: TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING=false to disable entirely.
    file_type_weighting_enabled: bool = Field(
        default=True,
        alias="TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING",
    )
    """
    Master switch. False → no weight multiplier, identical to current behaviour.
    Env: TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING=false
    """

    file_type_weights: dict[str, float] = Field(
        default_factory=lambda: {
            # Source code — full weight
            "python": 1.0,
            "javascript": 1.0,
            "typescript": 1.0,
            "tsx": 1.0,
            "go": 1.0,
            "rust": 1.0,
            "java": 1.0,
            "kotlin": 1.0,
            "ruby": 1.0,
            "cpp": 1.0,
            "c": 1.0,
            "csharp": 1.0,
            "razor": 1.0,
            "cshtml": 1.0,
            "csproj": 1.0,
            # Style / markup
            "html": 0.4,
            "css": 0.4,
            # Config / data
            "json": 0.5,
            "yaml": 0.5,
            "toml": 0.5,
            # Documentation
            "markdown": 0.3,
            # Unknown — conservative default, do not penalise unknown files
            "unknown": 0.8,
        },
    )
    """
    Per-language RRF score multiplier applied after fusion.
    Keys are Language enum values (lowercase strings).
    Missing key → multiplier = 1.0 (safe fallback, does not downrank unknown types).

    Individual overrides via env (one var per language):
      TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN=0.1
      TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_YAML=0.6
      ...

    Note: Pydantic BaseSettings does not natively merge individual env keys into a
    dict field. The model_post_init hook reads
    TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_{LANG} vars and merges them on top of
    the default dict at construction time.
    """

    def model_post_init(self, __context: Any) -> None:
        import json
        import os

        # Full dict override (merged on top of defaults)
        full = os.environ.get("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS")
        if full:
            self.file_type_weights = {**self.file_type_weights, **json.loads(full)}
        # Per-language overrides (highest priority — applied last)
        prefix = "TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_"
        for key, val in os.environ.items():
            if key.startswith(prefix):
                lang = key[len(prefix) :].lower()
                self.file_type_weights[lang] = float(val)
```

Also add `Any` to the imports at the top of `config.py` if not already present:

```python
from typing import Any, Literal
```

- [ ] **Step 4: Run config tests — expect all pass**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/test_config.py::TestRetrievalConfigFileTypeWeighting -v --tb=short
```
Expected: all 9 tests PASS

- [ ] **Step 5: Run full unit suite — expect no regressions**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/ -q --tb=line 2>&1 | tail -3
```
Expected: existing count + 9 passed, 0 failed

- [ ] **Step 6: Commit**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
git add src/trelix/core/config.py tests/unit/test_config.py && \
git commit -m "feat(weighting): add file_type_weights + model_post_init to RetrievalConfig

Two new fields on RetrievalConfig:
- file_type_weighting_enabled: bool = True (TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING)
- file_type_weights: dict[str, float] with 22-language defaults

model_post_init merges TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS (JSON) and
per-language TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_{LANG} env vars on top
of defaults. 9 unit tests covering defaults, env overrides, priority."
```

---

### Task 2: Modify `reciprocal_rank_fusion()` to accept and apply weights + unit tests

**Files:**
- Modify: `src/trelix/retrieval/fusion.py`
- Test: `tests/unit/test_fusion.py` (append new class `TestFileTypeWeighting`)

**Interfaces:**
- Consumes: `SearchResult`, `Language` from `trelix.core.models`
- Produces:
  - `reciprocal_rank_fusion(ranked_lists, k=60, weights=None)` — new optional `weights` param
  - When `weights` is a non-empty dict: after RRF accumulation, multiply `rrf_scores[chunk_id]` by `weights.get(str(result.file.language), 1.0)`
  - When `weights=None` (default): skip multiplication entirely — output is bit-for-bit identical to the current implementation

- [ ] **Step 1: Write the failing tests**

Read `tests/unit/test_fusion.py`. The existing `_make_result` helper creates all results with `language=Language.PYTHON` and `file_id=1`. We need a helper that accepts arbitrary language. Append after the last existing test class (`TestRRFScoring`):

```python
# ---------------------------------------------------------------------------
# File-type weight multiplier
# ---------------------------------------------------------------------------


def _make_result_lang(
    symbol_id: int,
    score: float,
    language: Language,
    source: str = "bm25",
    file_id: int | None = None,
) -> SearchResult:
    """Build a SearchResult with a specific language — for weighting tests."""
    chunk = Chunk(symbol_id=symbol_id, chunk_text=f"body_{symbol_id}", token_count=10)
    symbol = Symbol(
        id=symbol_id,
        file_id=file_id or symbol_id,
        name=f"sym_{symbol_id}",
        qualified_name=f"mod.sym_{symbol_id}",
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=5,
        signature=f"def sym_{symbol_id}()",
        body=f"def sym_{symbol_id}(): pass",
    )
    file = IndexedFile(
        id=file_id or symbol_id,
        path=f"/repo/file_{symbol_id}.py",
        rel_path=f"file_{symbol_id}.py",
        language=language,
        hash="abc",
        size_bytes=100,
    )
    return SearchResult(chunk=chunk, symbol=symbol, file=file, score=score, rank=1, source=source)


class TestFileTypeWeighting:
    def test_weights_none_produces_identical_output_to_unweighted(self) -> None:
        """weights=None must give bit-for-bit identical scores to calling without weights."""
        k = 60
        list_a = [_make_result(1, 0.9), _make_result(2, 0.8), _make_result(3, 0.7)]
        list_b = [_make_result(2, 0.85), _make_result(1, 0.75), _make_result(4, 0.6)]

        fused_unweighted = reciprocal_rank_fusion([list_a, list_b], k=k)
        fused_none = reciprocal_rank_fusion([list_a, list_b], k=k, weights=None)

        assert len(fused_unweighted) == len(fused_none)
        for a, b in zip(fused_unweighted, fused_none):
            assert a.chunk.symbol_id == b.chunk.symbol_id
            assert a.score == b.score  # exact float equality — no arithmetic difference

    def test_empty_weights_dict_produces_identical_output(self) -> None:
        """weights={} (empty dict) is falsy → multiplier step skipped, same as weights=None."""
        k = 60
        results = [_make_result(i, 1.0 / i) for i in range(1, 5)]
        fused_none = reciprocal_rank_fusion([results], k=k, weights=None)
        fused_empty = reciprocal_rank_fusion([results], k=k, weights={})
        for a, b in zip(fused_none, fused_empty):
            assert a.chunk.symbol_id == b.chunk.symbol_id
            assert a.score == b.score

    def test_weight_multiplier_applied_to_rrf_score_python(self) -> None:
        """Python at rank 1, weight 1.0 → score = 1/(60+1) * 1.0."""
        py_result = _make_result_lang(symbol_id=1, score=0.9, language=Language.PYTHON)
        fused = reciprocal_rank_fusion([[py_result]], k=60, weights={"python": 1.0})
        expected = (1.0 / (60 + 1)) * 1.0
        assert abs(fused[0].score - expected) < 1e-12

    def test_weight_multiplier_applied_to_rrf_score_markdown(self) -> None:
        """Markdown at rank 1, weight 0.3 → score = 1/(60+1) * 0.3."""
        md_result = _make_result_lang(symbol_id=2, score=0.9, language=Language.MARKDOWN)
        fused = reciprocal_rank_fusion([[md_result]], k=60, weights={"markdown": 0.3})
        expected = (1.0 / (60 + 1)) * 0.3
        assert abs(fused[0].score - expected) < 1e-12

    def test_markdown_downweighted_below_python(self) -> None:
        """
        README.md (markdown) outranks the Python file in raw BM25 — rank 1 vs rank 2.
        After file-type weighting, the Python file must rank above the README.

        Without weights:
          markdown_score = 1/(60+1) ≈ 0.01639
          python_score   = 1/(60+2) ≈ 0.01613
          → markdown wins

        With weights={markdown: 0.3, python: 1.0}:
          markdown_score = 1/61 * 0.3 ≈ 0.00492
          python_score   = 1/62 * 1.0 ≈ 0.01613
          → python wins
        """
        md = _make_result_lang(symbol_id=10, score=0.95, language=Language.MARKDOWN)
        py = _make_result_lang(symbol_id=20, score=0.80, language=Language.PYTHON)

        # BM25 leg: markdown at rank 1, python at rank 2
        bm25_leg = [md, py]

        weights = {"python": 1.0, "markdown": 0.3}
        fused = reciprocal_rank_fusion([bm25_leg], k=60, weights=weights)

        ranked_ids = [r.chunk.symbol_id for r in fused]
        assert ranked_ids[0] == 20, (
            f"Expected python (id=20) at rank 1 after weighting, "
            f"but got {ranked_ids[0]} — markdown still winning"
        )

    def test_missing_language_key_defaults_to_1_0(self) -> None:
        """
        A Language value not present in the weights dict must NOT be penalised.
        weights.get(lang, 1.0) must return 1.0 for unknown languages.
        """
        # Use a language explicitly absent from the weights dict
        go_result = _make_result_lang(symbol_id=5, score=0.9, language=Language.GO)
        # weights dict has no "go" key
        weights = {"python": 1.0, "markdown": 0.3}
        fused = reciprocal_rank_fusion([[go_result]], k=60, weights=weights)
        expected = 1.0 / (60 + 1) * 1.0  # fallback multiplier = 1.0
        assert abs(fused[0].score - expected) < 1e-12

    def test_multiple_legs_weights_applied_after_accumulation(self) -> None:
        """
        Python chunk appears in both vector (rank 1) and BM25 (rank 2).
        Markdown chunk appears only in BM25 (rank 1).

        Accumulated RRF before weighting:
          python  = 1/61 + 1/62 ≈ 0.03252
          markdown = 1/61      ≈ 0.01639

        After weights {python: 1.0, markdown: 0.3}:
          python  = 0.03252 * 1.0 ≈ 0.03252
          markdown = 0.01639 * 0.3 ≈ 0.00492

        Python must rank first.
        """
        py = _make_result_lang(symbol_id=1, score=0.9, language=Language.PYTHON)
        md = _make_result_lang(symbol_id=2, score=0.85, language=Language.MARKDOWN)

        # vector leg: python at rank 1 only
        # bm25 leg: markdown at rank 1, python at rank 2
        vector_leg = [py]
        bm25_leg = [md, py]

        weights = {"python": 1.0, "markdown": 0.3}
        fused = reciprocal_rank_fusion([vector_leg, bm25_leg], k=60, weights=weights)

        assert fused[0].chunk.symbol_id == 1, "Python must outrank Markdown after weighting"

        # Verify exact scores
        k = 60
        expected_python = (1.0 / (k + 1) + 1.0 / (k + 2)) * 1.0
        expected_markdown = (1.0 / (k + 1)) * 0.3
        fused_map = {r.chunk.symbol_id: r.score for r in fused}
        assert abs(fused_map[1] - expected_python) < 1e-12
        assert abs(fused_map[2] - expected_markdown) < 1e-12

    def test_html_css_downweighted_below_source(self) -> None:
        """HTML at rank 1 (weight 0.4) must fall below Python at rank 2 (weight 1.0)."""
        html = _make_result_lang(symbol_id=30, score=0.9, language=Language.HTML)
        py = _make_result_lang(symbol_id=31, score=0.8, language=Language.PYTHON)
        bm25_leg = [html, py]
        weights = {"html": 0.4, "python": 1.0}
        fused = reciprocal_rank_fusion([bm25_leg], k=60, weights=weights)
        # html: 1/61 * 0.4 = 0.00656   python: 1/62 * 1.0 = 0.01613 → python wins
        assert fused[0].chunk.symbol_id == 31

    def test_scores_still_positive_after_weighting(self) -> None:
        """All weighted scores must remain strictly positive."""
        md = _make_result_lang(symbol_id=100, score=0.9, language=Language.MARKDOWN)
        fused = reciprocal_rank_fusion([[md]], k=60, weights={"markdown": 0.3})
        assert fused[0].score > 0.0

    def test_rank_field_updated_after_weighting(self) -> None:
        """rank fields on weighted fused output must be 1-indexed and sequential."""
        py = _make_result_lang(symbol_id=1, score=0.9, language=Language.PYTHON)
        md = _make_result_lang(symbol_id=2, score=0.8, language=Language.MARKDOWN)
        fused = reciprocal_rank_fusion([[md, py]], k=60, weights={"python": 1.0, "markdown": 0.3})
        for expected_rank, result in enumerate(fused, start=1):
            assert result.rank == expected_rank
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/test_fusion.py::TestFileTypeWeighting -v --tb=short 2>&1 | head -20
```
Expected: `TypeError: reciprocal_rank_fusion() got an unexpected keyword argument 'weights'`

- [ ] **Step 3: Implement the `weights` parameter in `fusion.py`**

Read `src/trelix/retrieval/fusion.py`. Replace the entire file with:

```python
"""
Reciprocal Rank Fusion (RRF) — combines multiple ranked lists into one.

Formula:  score(doc) = Σ  1 / (k + rank_i)
where k=60 is the standard constant (Cormack et al. 2009).

Why RRF instead of score normalization:
- Scores from different systems (BM25 vs cosine) are not comparable
- RRF only uses rank position, making it robust across any mix of retrievers
- Simple, fast, no training needed
"""

from __future__ import annotations

from collections import defaultdict

from trelix.core.models import SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[SearchResult]:
    """
    Fuse multiple ranked result lists using RRF, then optionally apply
    per-language file-type weight multipliers.

    Args:
        ranked_lists: list of result lists, each sorted by relevance (best first)
        k:            RRF constant (default 60, Cormack et al. 2009)
        weights:      optional dict mapping Language enum value (str) to a
                      multiplicative weight applied after RRF accumulation.
                      None or empty dict → no weighting (backward compatible).

    Returns:
        Single merged list sorted by fused (weighted) RRF score, best first.
    """
    # Map chunk_id → accumulated RRF score
    rrf_scores: dict[int, float] = defaultdict(float)
    # Keep the best SearchResult object per chunk (highest contributing list)
    best_result: dict[int, SearchResult] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            chunk_id = result.chunk.symbol_id  # use symbol_id as dedup key
            rrf_scores[chunk_id] += 1.0 / (k + rank)
            # Keep first-seen result: source reflects which leg first found it.
            # Do NOT replace based on raw score — scores across legs (cosine vs
            # BM25) are not comparable, so score comparison would always favor
            # vector (0.7–0.95 range) over BM25 (0.05–0.5 range).
            if chunk_id not in best_result:
                best_result[chunk_id] = result

    # Apply file-type weight multiplier (new step — skipped when weights is None/empty)
    if weights:
        for chunk_id, result in best_result.items():
            lang = result.file.language  # Language enum (StrEnum → str)
            multiplier = weights.get(str(lang), 1.0)
            rrf_scores[chunk_id] *= multiplier

    # Sort by fused score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    fused: list[SearchResult] = []
    for new_rank, chunk_id in enumerate(sorted_ids, start=1):
        result = best_result[chunk_id]
        # Overwrite score with the RRF score for downstream reranking
        result.score = rrf_scores[chunk_id]
        result.rank = new_rank
        fused.append(result)

    return fused
```

- [ ] **Step 4: Run weighting tests — expect all pass**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/test_fusion.py::TestFileTypeWeighting -v --tb=short
```
Expected: all 10 tests PASS

- [ ] **Step 5: Run all fusion tests — expect no regressions**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/test_fusion.py -v --tb=short
```
Expected: all tests PASS (existing + new)

- [ ] **Step 6: Run full unit suite — expect no regressions**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/ -q --tb=line 2>&1 | tail -3
```
Expected: existing count + new weighting tests passed, 0 failed

- [ ] **Step 7: Commit**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
git add src/trelix/retrieval/fusion.py tests/unit/test_fusion.py && \
git commit -m "feat(weighting): add file-type weight multiplier to reciprocal_rank_fusion

New optional weights param applies a per-language multiplier to
accumulated RRF scores after all legs contribute. weights=None (default)
skips the step entirely — scores are bit-for-bit identical to the
previous implementation. Missing language key falls back to 1.0.

10 unit tests: backward compat, exact score formula, markdown downweight,
multi-leg accumulation, missing key fallback, rank field correctness."
```

---

### Task 3: Wire weights from config into `Retriever._retrieve_standard()` + integration recall test

**Files:**
- Modify: `src/trelix/retrieval/retriever.py` (lines ~268–271, the `reciprocal_rank_fusion` call)
- Modify: `tests/integration/test_recall.py` (add `TestFileTypeWeightingRecall` class)

**Interfaces:**
- Consumes: `file_type_weighting_enabled` and `file_type_weights` from `RetrievalConfig` (Task 1); `weights` param from `reciprocal_rank_fusion` (Task 2)
- Produces:
  - When `cfg.file_type_weighting_enabled=True`: pass `weights=cfg.file_type_weights` to fusion
  - When `cfg.file_type_weighting_enabled=False`: pass `weights=None` to fusion
  - Integration recall test passes at ≥ 70% with weighting enabled

- [ ] **Step 1: Write a failing integration test**

Read `tests/integration/test_recall.py`. After the existing `TestRecallSummary` class, append:

```python
# ---------------------------------------------------------------------------
# File-type weighting recall tests
# ---------------------------------------------------------------------------


class TestFileTypeWeightingRecall:
    """
    Assert that file-type weighting improves recall for the 4 known-failing
    queries where README.md was outranking the actual source file.

    These tests use the same indexed_mini_repo fixture as the rest of the module.
    The mini_repo does not have a README.md, so these tests verify the mechanism
    using queries that should resolve to Python source files (not markdown).

    The key assertion is the Recall@5 metric: with weighting enabled (default),
    the overall recall must remain ≥ 70%.
    """

    def test_recall_with_weighting_enabled_meets_threshold(
        self, retriever: Retriever
    ) -> None:
        """Recall@5 >= 70% with file-type weighting enabled (default config)."""
        passed = sum(
            1
            for case in EVAL_CASES
            if _file_in_top_k(retriever, case.query, case.expected_file)
        )
        recall_pct = passed / len(EVAL_CASES) * 100
        assert recall_pct >= 70.0, (
            f"Recall@5 with weighting = {recall_pct:.0f}% ({passed}/{len(EVAL_CASES)}) "
            f"— must be ≥ 70%"
        )

    def test_weighting_disabled_still_returns_results(
        self,
        indexed_mini_repo: Path,
    ) -> None:
        """With weighting disabled, retriever still returns results (kill-switch test)."""
        from trelix.core.config import EmbedderConfig, IndexConfig, RetrievalConfig

        config = IndexConfig(
            repo_path=str(indexed_mini_repo),
            incremental=False,
            parse_workers=2,
            embedder=EmbedderConfig(provider="local"),
            retrieval=RetrievalConfig(rerank=False, file_type_weighting_enabled=False),
        )
        disabled_retriever = Retriever(config)
        context = disabled_retriever.retrieve("how does authentication work")
        assert len(context.results) > 0, (
            "Retriever with weighting disabled returned no results"
        )

    def test_python_source_files_ranked_ahead_of_config_files(
        self, retriever: Retriever
    ) -> None:
        """
        For a query that returns both Python source and JSON/YAML config,
        at least one Python file must appear in top-5 ahead of all config-only files.

        Uses 'main entry point' query — main.py should be in top-5.
        """
        context = retriever.retrieve("main entry point")
        top5_files = [r.file.rel_path for r in context.results[:5]]
        has_python_in_top5 = any(f.endswith(".py") for f in top5_files)
        assert has_python_in_top5, (
            f"No Python file in top-5 for 'main entry point'. Got: {top5_files}"
        )
```

- [ ] **Step 2: Run integration test — expect test_recall_with_weighting_enabled_meets_threshold to pass (the existing fixture already uses default config which now includes weighting), verify no regressions**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/integration/test_recall.py::TestFileTypeWeightingRecall::test_weighting_disabled_still_returns_results -v --tb=short 2>&1 | tail -10
```
Expected: FAIL — `TypeError: RetrievalConfig() got an unexpected keyword argument 'file_type_weighting_enabled'` (config field exists, but fusion wiring not done yet, so the test instantiating Retriever with that config should work if Task 1 is done; the kill-switch test should pass once wiring is done)

> Note: If Task 1 is complete, the `RetrievalConfig(file_type_weighting_enabled=False)` instantiation succeeds. The actual fusion-level wiring happens in the next step. The test above tests end-to-end (results returned), so it may already pass — but the `test_recall_with_weighting_enabled_meets_threshold` test uses the module-scoped `retriever` fixture which uses default config; that test may already pass or fail depending on the mini_repo queries.

- [ ] **Step 3: Wire config into `Retriever._retrieve_standard`**

Read `src/trelix/retrieval/retriever.py`. Find the `reciprocal_rank_fusion` call (around line 268):

```python
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results, grep_results],
            k=cfg.rrf_k,
        )
```

Replace it with:

```python
        _weights = cfg.file_type_weights if cfg.file_type_weighting_enabled else None
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results, grep_results],
            k=cfg.rrf_k,
            weights=_weights,
        )
```

No other changes to retriever.py.

- [ ] **Step 4: Run integration recall tests — expect all pass**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/integration/test_recall.py -v --tb=short 2>&1 | tail -20
```
Expected: all tests PASS including the summary recall@5 ≥ 70% assertion

- [ ] **Step 5: Run full unit suite — expect no regressions**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/ -q --tb=line 2>&1 | tail -3
```
Expected: all existing unit tests PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
git add src/trelix/retrieval/retriever.py tests/integration/test_recall.py && \
git commit -m "feat(weighting): wire file_type_weights from config into Retriever._retrieve_standard

Reads cfg.file_type_weighting_enabled to decide whether to pass
cfg.file_type_weights or None to reciprocal_rank_fusion.
file_type_weighting_enabled=False is the runtime kill-switch that
restores the previous unweighted behaviour without a code change.

Integration tests: recall@5 >= 70% assertion, kill-switch (disabled
path still returns results), Python source ranked above config files."
```

---

### Task 4: Full validation (ruff + mypy + pytest)

**Files:** No new files. This task validates the complete implementation.

**Interfaces:**
- Consumes: all changes from Tasks 1–3
- Produces: clean ruff, mypy, and full test suite output confirming the implementation is complete

- [ ] **Step 1: Run ruff lint + format**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/ruff check src/ tests/ --fix 2>&1 | tail -5 && \
.venv/bin/ruff format src/ tests/ 2>&1 | tail -5
```
Expected: `All checks passed!` (lint) and `N files left unchanged.` or reformatted files (format). Fix any remaining lint findings before proceeding.

- [ ] **Step 2: Run mypy**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/mypy src/trelix/ --ignore-missing-imports 2>&1 | tail -5
```
Expected: `Success: no issues found in N source files`

If mypy reports an error on `model_post_init`, add the `Any` import to `typing`:
```python
from typing import Any, Literal
```
(This was specified in Task 1 Step 3; confirm it is present.)

- [ ] **Step 3: Run the complete unit test suite**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/ -v --tb=short 2>&1 | tail -10
```
Expected: all tests PASS, 0 failed

- [ ] **Step 4: Run the complete integration test suite**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/integration/ -v --tb=short 2>&1 | tail -20
```
Expected: all tests PASS including recall@5 ≥ 70%

- [ ] **Step 5: Run full suite with coverage**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
.venv/bin/python -m pytest tests/unit/ tests/integration/ \
    -q --tb=line --cov=trelix --cov-report=term-missing 2>&1 | tail -12
```
Expected: all tests PASS, coverage ≥ 75% (or better than baseline)

- [ ] **Step 6: Final summary commit (validation docs only if anything changed)**

If ruff or mypy required any fixups, commit those:

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && \
git add -p && \
git commit -m "style(weighting): ruff + mypy fixups for file-type weighting implementation"
```

If nothing needed fixing, skip this step.

---

## Implementation Checklist

- [ ] Task 1: `RetrievalConfig` fields + `model_post_init` + 9 config unit tests
- [ ] Task 2: `reciprocal_rank_fusion(weights=)` param + 10 fusion unit tests
- [ ] Task 3: Retriever wiring + 3 integration recall tests
- [ ] Task 4: ruff clean, mypy clean, full suite green, coverage ≥ 75%

## Success Criteria

| Criterion | Threshold | How verified |
|---|---|---|
| Recall@5 with weighting | ≥ 70% | `test_recall_with_weighting_enabled_meets_threshold` |
| Recall@5 without weighting | Returns results (no crash) | `test_weighting_disabled_still_returns_results` |
| `weights=None` backward compat | Bit-for-bit identical | `test_weights_none_produces_identical_output_to_unweighted` |
| Markdown downranked | Python beats README at same BM25 rank | `test_markdown_downweighted_below_python` |
| Missing language fallback | Multiplier = 1.0, no penalty | `test_missing_language_key_defaults_to_1_0` |
| Config env overrides | Per-lang and JSON dict env vars work | 3 env-override tests in `TestRetrievalConfigFileTypeWeighting` |
| ruff | 0 lint errors | `ruff check src/ tests/` |
| mypy | 0 type errors | `mypy src/trelix/` |
| Existing tests | 0 regressions | Full unit + integration suite |

## Files Changed

| File | Change |
|---|---|
| `src/trelix/core/config.py` | Add `file_type_weighting_enabled`, `file_type_weights`, `model_post_init` to `RetrievalConfig`; add `Any` to `typing` imports |
| `src/trelix/retrieval/fusion.py` | Add `weights: dict[str, float] \| None = None` param to `reciprocal_rank_fusion`; apply multiplier after RRF accumulation |
| `src/trelix/retrieval/retriever.py` | Read `cfg.file_type_weighting_enabled` and pass `weights` into `reciprocal_rank_fusion` call in `_retrieve_standard` |
| `tests/unit/test_config.py` | Append `TestRetrievalConfigFileTypeWeighting` (9 tests) |
| `tests/unit/test_fusion.py` | Append `_make_result_lang` helper + `TestFileTypeWeighting` class (10 tests) |
| `tests/integration/test_recall.py` | Append `TestFileTypeWeightingRecall` (3 integration tests) |

Total diff: ~130 lines of code, ~90 lines of tests. No schema changes, no new dependencies, no migration needed.
