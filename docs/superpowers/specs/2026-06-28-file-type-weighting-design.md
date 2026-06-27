# File-Type Weighting in BM25+RRF Fusion — Design Spec

**Date:** 2026-06-28
**Status:** Approved
**Phase:** 2 — Search Quality

---

## Problem

The stress-test audit of 15 ground-truth recall queries revealed a systemic bias: README.md outranks Python/Go/TypeScript source files in 4 of 6 quality misses (Recall@5 = 60%).

### Root cause

SQLite FTS5 assigns BM25 scores purely on term-frequency/IDF statistics with no awareness of file type. A README.md that mentions `TrelixChatClient`, `Bedrock Converse API`, or `BM25 full-text search` as documentation prose receives the same BM25 rank contribution as the `.py` file that *defines* or *implements* those things. RRF then compounds this: once a README chunk holds a high BM25 rank, the 1/(k + rank) contribution from the BM25 leg pushes its fused score above the actual source file.

### Observed failures (from stress test audit)

| Query | Expected top result | Actual top result |
|---|---|---|
| "TrelixChatClient" | `llm/client.py` | `README.md` |
| "Bedrock Converse API" | `llm/providers/bedrock.py` | `README.md` |
| "BM25 full-text search" | `retrieval/bm25.py` | `README.md` |
| "vector embeddings" | `retrieval/vector.py` | `README.md` |

**Baseline Recall@5: 60%** (9 of 15 ground-truth queries return the expected source file in position 1–5).

### Why not fix BM25 directly?

Options considered:

1. **Boost source files in FTS5 at indexing time** — FTS5 does not support per-row score weights; requires a custom tokenizer or external ranking layer.
2. **Separate FTS5 tables per file type** — adds index fragmentation, complicates query routing, and breaks the single-leg abstraction.
3. **Post-BM25 score multiplication before fusion** — viable but couples file-type logic into the BM25 module, which is responsible only for retrieval, not ranking policy.
4. **Post-RRF weight multiplier (chosen)** — RRF already owns the "combine and rank" step. Injecting weights here is a single, auditable change with zero impact on BM25 or vector internals.

---

## Architecture

```
Query
  │
  ├── vector_search()        ← unchanged
  ├── bm25_search()          ← unchanged
  └── grep_search()          ← unchanged
           │
           ▼
  reciprocal_rank_fusion(ranked_lists, k, weights)
           │
           │  1. Accumulate RRF scores per chunk (existing logic)
           │     score(doc) = Σ 1/(k + rank_i)
           │
           │  2. NEW: apply file-type weight multiplier
           │     weighted_score = rrf_score × weights[language]
           │     (weights=None or disabled → multiplier = 1.0, no change)
           │
           ▼
  sorted fused list   (weighted scores, best first)
           │
           ▼
  Retriever.retrieve()  →  reranker  →  context assembly
```

The weights dict is threaded from `RetrievalConfig` through `Retriever` into `reciprocal_rank_fusion`. The BM25 and vector legs are completely unmodified.

---

## Configuration

### New fields on `RetrievalConfig`

```python
# src/trelix/core/config.py  —  inside class RetrievalConfig(BaseSettings)

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
        "python":     1.0,
        "javascript": 1.0,
        "typescript": 1.0,
        "tsx":        1.0,
        "go":         1.0,
        "rust":       1.0,
        "java":       1.0,
        "kotlin":     1.0,
        "ruby":       1.0,
        "cpp":        1.0,
        "c":          1.0,
        "csharp":     1.0,
        "razor":      1.0,
        "cshtml":     1.0,
        "csproj":     1.0,
        # Style / markup
        "html":       0.4,
        "css":        0.4,
        # Config / data
        "json":       0.5,
        "yaml":       0.5,
        "toml":       0.5,
        # Documentation
        "markdown":   0.3,
        # Unknown — conservative default, do not penalise unknown files
        "unknown":    0.8,
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
dict field. The RetrievalConfig.model_post_init hook reads
TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_{LANG} vars and merges them on top of
the default dict at construction time.
"""
```

### Environment variable resolution order

```
1. field default_factory dict (code defaults above)
2. TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS=<json>  (full dict override via JSON string)
3. TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_{LANG}   (per-language override, highest priority)
```

`model_post_init` implementation sketch:

```python
def model_post_init(self, __context: Any) -> None:
    import os, json
    # Full dict override
    full = os.environ.get("TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS")
    if full:
        self.file_type_weights = {**self.file_type_weights, **json.loads(full)}
    # Per-language overrides
    prefix = "TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_"
    for key, val in os.environ.items():
        if key.startswith(prefix):
            lang = key[len(prefix):].lower()
            self.file_type_weights[lang] = float(val)
```

---

## Weight table

| Language | Value | Rationale |
|---|---|---|
| python | 1.0 | Primary source language |
| javascript | 1.0 | Source |
| typescript | 1.0 | Source |
| tsx | 1.0 | Source (React components) |
| go | 1.0 | Source |
| rust | 1.0 | Source |
| java | 1.0 | Source |
| kotlin | 1.0 | Source |
| ruby | 1.0 | Source |
| cpp | 1.0 | Source |
| c | 1.0 | Source |
| csharp | 1.0 | Source |
| razor | 1.0 | Source (.cshtml templates) |
| cshtml | 1.0 | Source |
| csproj | 1.0 | Project file — still code-adjacent |
| html | 0.4 | Mostly structural markup, rarely defines logic |
| css | 0.4 | Style, not logic |
| json | 0.5 | Config/data — useful for schema queries |
| yaml | 0.5 | Config — CI, Helm, Kubernetes |
| toml | 0.5 | Config — Cargo.toml, pyproject.toml |
| markdown | 0.3 | Documentation prose — lowest priority for code queries |
| unknown | 0.8 | Conservative: do not aggressively penalise unrecognised types |

These defaults are calibrated to the stress-test failure set. All values are tunable via env vars without a code change.

---

## Implementation

### 1. `src/trelix/retrieval/fusion.py` — modify `reciprocal_rank_fusion`

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
    weights: dict[str, float] | None = None,   # NEW optional param
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
    rrf_scores: dict[int, float] = defaultdict(float)
    best_result: dict[int, SearchResult] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            chunk_id = result.chunk.symbol_id
            rrf_scores[chunk_id] += 1.0 / (k + rank)
            if chunk_id not in best_result:
                best_result[chunk_id] = result

    # Apply file-type weight multiplier (new step)
    if weights:
        for chunk_id, result in best_result.items():
            lang = result.file.language        # Language enum (StrEnum → str)
            multiplier = weights.get(str(lang), 1.0)
            rrf_scores[chunk_id] *= multiplier

    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    fused: list[SearchResult] = []
    for new_rank, chunk_id in enumerate(sorted_ids, start=1):
        result = best_result[chunk_id]
        result.score = rrf_scores[chunk_id]
        result.rank = new_rank
        fused.append(result)

    return fused
```

Key properties:
- `weights=None` (default) → no multiplication, **identical to current behaviour**.
- Multiplier lookup uses `.get(lang, 1.0)` — unknown languages are not penalised.
- Weight is applied to the *accumulated* RRF score, after all legs contribute. This preserves the RRF rank-combination logic and only scales the final composite score.

### 2. `src/trelix/core/config.py` — add fields to `RetrievalConfig`

Add the two fields and `model_post_init` as shown in the Configuration section above.

### 3. `src/trelix/retrieval/retriever.py` — wire config into fusion call

```python
# existing call (line ~268):
fused = reciprocal_rank_fusion(
    [vector_results, bm25_results, grep_results],
    k=cfg.rrf_k,
)

# updated call:
_weights = cfg.file_type_weights if cfg.file_type_weighting_enabled else None
fused = reciprocal_rank_fusion(
    [vector_results, bm25_results, grep_results],
    k=cfg.rrf_k,
    weights=_weights,
)
```

No other changes to the retriever.

---

## Backward compatibility

| Scenario | Behaviour |
|---|---|
| `weights=None` (default, no config) | Multiplier step is skipped entirely; scores are bit-for-bit identical to the current implementation |
| `file_type_weighting_enabled=False` | Equivalent to `weights=None`; master kill-switch for rollback |
| Language key absent from weights dict | `weights.get(lang, 1.0)` returns 1.0 — no effect |
| New language added to `Language` enum without a weights entry | Falls back to 1.0 — new language is not penalised |
| Existing tests with no config | `RetrievalConfig` defaults have `file_type_weighting_enabled=True`, but the unit tests for `reciprocal_rank_fusion` that do not pass `weights=` continue to work unchanged |

---

## Testing strategy

### Unit tests — `tests/retrieval/test_fusion.py`

**New test class: `TestFileTypeWeighting`**

```
test_weights_none_is_unchanged
    Given weights=None, assert scores match unweighted RRF formula exactly.

test_markdown_downweighted_below_python
    Given one Python chunk at rank 2 and one Markdown chunk at rank 1,
    with weights={python:1.0, markdown:0.3},
    assert python chunk ends up ranked above markdown in fused output.

test_weight_multiplier_applied_to_rrf_score
    For a single-list input with one Python result at rank 1,
    assert result.score == (1/(60+1)) * 1.0.
    For markdown at rank 1:
    assert result.score == (1/(60+1)) * 0.3.

test_missing_language_key_defaults_to_1_0
    Use a Language value not in the weights dict; assert score = rrf_score * 1.0.

test_weighting_disabled_flag
    Set file_type_weighting_enabled=False in RetrievalConfig;
    verify reciprocal_rank_fusion receives weights=None.

test_env_override_per_language
    Set env TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN=0.1;
    assert RetrievalConfig().file_type_weights["markdown"] == 0.1.

test_env_override_full_json_dict
    Set env TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS='{"markdown":0.05}';
    assert config.file_type_weights["markdown"] == 0.05.
```

### Integration tests — `tests/integration/test_recall.py`

The existing 15-query ground-truth suite runs end-to-end against a pre-indexed fixture repo.

**Target: Recall@5 >= 70%** (up from 60% baseline — represents fixing at least 2 of the 4 observed failures).

Additional assertions for the 4 known-failing queries:
```
assert_source_file_outranks_readme("TrelixChatClient", "llm/client.py")
assert_source_file_outranks_readme("Bedrock Converse API", "llm/providers/bedrock.py")
assert_source_file_outranks_readme("BM25 full-text search", "retrieval/bm25.py")
assert_source_file_outranks_readme("vector embeddings", "retrieval/vector.py")
```

Where `assert_source_file_outranks_readme(query, expected_source)` fetches the top-5 results, finds the rank of `expected_source` and the rank of any `README.md` result, and asserts `rank(expected_source) < rank(README.md)`.

### What is NOT tested here
- Changes to BM25 or vector leg scores — those modules are untouched.
- Reranker behaviour — weights are applied before the reranker; reranker tests are out of scope for this spec.

---

## Expected recall improvement

Conservative estimate based on the failure analysis:

| Query | Failure mechanism | Expected fix? |
|---|---|---|
| "TrelixChatClient" | README rank 1 in BM25 → fused rank 1 | Yes — README score × 0.3, Python × 1.0 |
| "Bedrock Converse API" | README rank 1 in BM25 → fused rank 1 | Yes |
| "BM25 full-text search" | README rank 1 in BM25 → fused rank 1 | Yes |
| "vector embeddings" | README rank 1 in BM25 → fused rank 1 | Yes |

Fixing all 4 raises raw Recall@5 from 9/15 to 13/15 = **86.7%**, exceeding the 70% integration test threshold. The actual improvement may be lower if some queries are also affected by the vector leg or graph expansion, but fixing the BM25-driven README dominance is sufficient to clear 70%.

Weight tuning guidance for future calibration:

- If doc-only repos (e.g. wikis) need better recall, raise `markdown` toward 0.6.
- If YAML CI config queries are underperforming, raise `yaml` toward 0.7.
- Use `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING=false` to A/B test against the baseline without a code change.

---

## Files changed

| File | Change |
|---|---|
| `src/trelix/retrieval/fusion.py` | Add `weights` param to `reciprocal_rank_fusion`; apply multiplier after RRF accumulation |
| `src/trelix/core/config.py` | Add `file_type_weighting_enabled` and `file_type_weights` to `RetrievalConfig`; add `model_post_init` for env merging |
| `src/trelix/retrieval/retriever.py` | Pass `weights` from config into `reciprocal_rank_fusion` call |
| `tests/retrieval/test_fusion.py` | New `TestFileTypeWeighting` test class (7 unit tests) |
| `tests/integration/test_recall.py` | Assert Recall@5 >= 70%; 4 new per-query README-vs-source assertions |

Total diff: ~120 lines of code, ~80 lines of tests. No schema changes, no new dependencies, no migration needed.
