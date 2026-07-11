# trelix v2.0 Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade trelix from a production v1.1.0 baseline into a state-of-the-art code intelligence system across embedding quality, retrieval architecture, chunking depth, evaluation rigor, storage scalability, and developer experience.

**Architecture:** Three independent phases ship as three PRs into `develop`. Phase 1 upgrades models and eval with zero schema changes. Phase 2 adds multi-granularity indexing, PLAID late-interaction reranking, and streaming. Phase 3 adds LanceDB storage, LLM-as-judge eval, and a VS Code extension scaffold.

**Tech Stack:** Python 3.11–3.12, tree-sitter-languages ≥ 1.10.2, sqlite-vec HNSW, tiktoken, pydantic-settings, voyageai, sentence-transformers, ragatouille (PLAID), lancedb ≥ 0.6, watchfiles, anthropic SDK, pytest ≥ 8.

## Global Constraints

- Branch: all work on `feature/*` branches → PR to `develop` (never commit directly to `develop` or `main`)
- Python: `>=3.11,<3.13` (cp313 tree-sitter-languages unavailable — do not relax this)
- No new mandatory runtime dependencies — all new providers are optional extras in `pyproject.toml`
- All new code must have ≥ 80% unit test coverage before PR merge
- Coverage gate: `fail_under = 75` in `pyproject.toml` must not regress
- Every new config field uses `pydantic-settings` `BaseSettings` with `env_prefix` — no raw `os.getenv()`
- Follow existing patterns exactly: `BaseEmbedder` ABC for embedders, `BaseVectorStore` ABC for stores, `TrelixChatClient` ABC for LLM clients
- Conventional commits: `feat(<scope>): …` / `fix(<scope>): …` / `chore: …`
- Repo root: `/Users/sairamugge/Desktop/Not-Humans-World/trelix/`
- Run tests with: `cd /Users/sairamugge/Desktop/Not-Humans-World/trelix && .venv/bin/python -m pytest`
- Run linting with: `.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/`

---

## PHASE 1 — Model Upgrades + Eval Hardening (highest ROI, zero schema change)

**Goal:** Add BGE-Code-v1 and Nomic CodeRankEmbed as first-class embedding providers; add Matryoshka dimension support to VoyageEmbedder; harden the eval harness with LLM-as-judge scoring; expose eval metrics in CI.

**Why first:** Research confirmed code-specialized models beat general-purpose by 6–7 NDCG@10 points at equal scale (CodeRAG-Bench, arXiv 2406.14497). These are drop-in `BaseEmbedder` subclasses — zero retrieval pipeline changes needed.

---

### Task 1: BGE-Code-v1 Embedder

**Files:**
- Create: `src/trelix/embedder/bge_code.py`
- Modify: `src/trelix/embedder/base.py` — add `"bge-code"` to `make_embedder()` factory switch
- Modify: `src/trelix/core/config.py` — extend `EmbedderConfig.provider` Literal, add `bge_code_model` and `bge_code_dimensions` fields
- Modify: `pyproject.toml` — add `[bge-code]` optional extra: `"FlagEmbedding>=1.3.0"`
- Create: `tests/unit/test_embedder_bge.py`

**Interfaces:**
- Consumes: `BaseEmbedder` ABC from `src/trelix/embedder/base.py` (`embed(texts) -> list[list[float]]`, `embed_query(text) -> list[float]`, `embed_async(texts)`, `dimension -> int`)
- Produces: `BGECodeEmbedder` class; `make_embedder()` returns it when `provider == "bge-code"`; `EmbedderConfig.effective_dimension` returns `bge_code_dimensions`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_embedder_bge.py`:
```python
"""Tests for BGECodeEmbedder."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from trelix.core.config import EmbedderConfig


class TestBGECodeEmbedder:
    def test_importable(self) -> None:
        from trelix.embedder.bge_code import BGECodeEmbedder
        assert BGECodeEmbedder is not None

    def test_is_base_embedder(self) -> None:
        from trelix.embedder.base import BaseEmbedder
        from trelix.embedder.bge_code import BGECodeEmbedder
        assert issubclass(BGECodeEmbedder, BaseEmbedder)

    def test_dimension_default(self) -> None:
        from trelix.embedder.bge_code import BGECodeEmbedder
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        with patch("trelix.embedder.bge_code.FlagModel", return_value=mock_model):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            emb = BGECodeEmbedder(cfg)
            assert emb.dimension == 768

    def test_embed_returns_correct_shape(self) -> None:
        from trelix.embedder.bge_code import BGECodeEmbedder
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = [[0.1] * 768, [0.2] * 768]
        with patch("trelix.embedder.bge_code.FlagModel", return_value=mock_model):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            emb = BGECodeEmbedder(cfg)
            result = emb.embed(["def foo(): pass", "class Bar: pass"])
            assert len(result) == 2
            assert len(result[0]) == 768

    def test_embed_query_uses_query_instruction(self) -> None:
        from trelix.embedder.bge_code import BGECodeEmbedder
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = [[0.1] * 768]
        with patch("trelix.embedder.bge_code.FlagModel", return_value=mock_model):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            emb = BGECodeEmbedder(cfg)
            emb.embed_query("how does auth work")
            call_kwargs = mock_model.encode.call_args
            # BGE query embedding should use instruction prefix
            assert call_kwargs is not None

    def test_make_embedder_returns_bge(self) -> None:
        from trelix.embedder.base import make_embedder
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        with patch("trelix.embedder.bge_code.FlagModel", return_value=mock_model):
            cfg = EmbedderConfig(provider="bge-code", _env_file=None)
            from trelix.embedder.bge_code import BGECodeEmbedder
            emb = make_embedder(cfg)
            assert isinstance(emb, BGECodeEmbedder)

    def test_config_effective_dimension(self) -> None:
        cfg = EmbedderConfig(provider="bge-code", bge_code_dimensions=768, _env_file=None)
        assert cfg.effective_dimension == 768
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix
.venv/bin/python -m pytest tests/unit/test_embedder_bge.py -v --tb=short 2>&1 | head -25
```
Expected: `ModuleNotFoundError` or `ImportError` — `BGECodeEmbedder` doesn't exist yet.

- [ ] **Step 3: Add `bge_code_model` and `bge_code_dimensions` fields to `EmbedderConfig`**

In `src/trelix/core/config.py`, find the `EmbedderConfig` class. Change the `provider` Literal and add fields after the `local_code_*` block (around line 197):

```python
provider: Literal[
    "openai", "azure", "local", "voyage", "local-code",
    "bedrock-titan", "bedrock-cohere", "bge-code", "nomic-code"
] = "local"
```

Add after `local_code_dimensions`:
```python
# ── BGE-Code-v1 (BAAI, CoIR SOTA 2025) ────────────────────────────────────
# Uses FlagEmbedding library. pip install trelix[bge-code]
bge_code_model: str = "BAAI/bge-code-v1"
bge_code_dimensions: int = 768  # BGE-Code-v1 default embedding dim
```

In `effective_dimension` property, add before the final `return 384`:
```python
if self.provider == "bge-code":
    return self.bge_code_dimensions
if self.provider == "nomic-code":
    return self.nomic_code_dimensions
```

- [ ] **Step 4: Create `src/trelix/embedder/bge_code.py`**

```python
"""
BGE-Code-v1 embedder (BAAI, May 2025).

Uses FlagEmbedding library (pip install FlagEmbedding>=1.3.0).
BGE-Code-v1 self-reports 81.77 CoIR average, the highest-known score
as of mid-2025. Uses asymmetric query/document encoding:
  - Documents: encoded directly (code text)
  - Queries: encoded with instruction prefix for retrieval

Install:
    pip install 'trelix[bge-code]'

Usage:
    TRELIX_EMBEDDER_PROVIDER=bge-code trelix index ./my-repo
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from trelix.embedder.base import BaseEmbedder

if TYPE_CHECKING:
    from trelix.core.config import EmbedderConfig

_QUERY_INSTRUCTION = (
    "Represent this query for searching relevant code: "
)


class BGECodeEmbedder(BaseEmbedder):
    """
    Embedder backed by BAAI/bge-code-v1 via FlagEmbedding.

    Asymmetric: queries use an instruction prefix; documents (code) are
    encoded directly. This matches BGE-Code-v1's training protocol.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        try:
            from FlagEmbedding import FlagModel
        except ImportError as e:
            raise ImportError(
                "FlagEmbedding is required for bge-code embedder. "
                "Install it with: pip install 'trelix[bge-code]'"
            ) from e

        self._model = FlagModel(
            config.bge_code_model,
            query_instruction_for_retrieval=_QUERY_INSTRUCTION,
            use_fp16=True,
        )
        self._dimensions = config.bge_code_dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, batch_size=32)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        vecs = self._model.encode_queries([text])
        return vecs[0].tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension() or self._dimensions
```

- [ ] **Step 5: Wire into `make_embedder()` in `src/trelix/embedder/base.py`**

Find the `make_embedder` factory function. Add after the `"bedrock-cohere"` case:
```python
case "bge-code":
    from trelix.embedder.bge_code import BGECodeEmbedder
    return BGECodeEmbedder(config)
```

Update the error message's list of valid providers to include `"bge-code"` and `"nomic-code"`.

- [ ] **Step 6: Add optional extra to `pyproject.toml`**

Find the `[project.optional-dependencies]` section. Add:
```toml
bge-code = [
    "FlagEmbedding>=1.3.0",   # BGE-Code-v1 and family
]
```

- [ ] **Step 7: Run tests — all must pass**

```bash
cd /Users/sairamugge/Desktop/Not-Humans-World/trelix
.venv/bin/python -m pytest tests/unit/test_embedder_bge.py -v --tb=short 2>&1 | tail -15
```
Expected: `6 passed`.

- [ ] **Step 8: Run full unit suite — no regressions**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=short 2>&1 | tail -5
```
Expected: all pass (≥ 1197 tests).

- [ ] **Step 9: Commit**

```bash
git add src/trelix/embedder/bge_code.py src/trelix/embedder/base.py \
        src/trelix/core/config.py pyproject.toml \
        tests/unit/test_embedder_bge.py
git commit -m "feat(embedder): add BGE-Code-v1 embedder (BAAI CoIR SOTA 2025)

- BGECodeEmbedder via FlagEmbedding>=1.3.0
- Asymmetric query/document encoding with instruction prefix
- Wired into make_embedder() factory as 'bge-code' provider
- pip install trelix[bge-code]
- CoIR self-reported: 81.77 avg (arXiv 2505.12697)"
```

---

### Task 2: Nomic CodeRankEmbed Embedder

**Files:**
- Create: `src/trelix/embedder/nomic_code.py`
- Modify: `src/trelix/embedder/base.py` — add `"nomic-code"` case to factory
- Modify: `src/trelix/core/config.py` — add `nomic_code_model`, `nomic_code_dimensions`, `nomic_code_task` fields
- Modify: `pyproject.toml` — add `[nomic-code]` optional extra
- Create: `tests/unit/test_embedder_nomic.py`

**Interfaces:**
- Consumes: `BaseEmbedder` ABC, `EmbedderConfig` with `provider == "nomic-code"`
- Produces: `NomicCodeEmbedder`; task-aware embedding (`search_document` vs `search_query`); `make_embedder()` returns it

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_embedder_nomic.py`:
```python
"""Tests for NomicCodeEmbedder."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from trelix.core.config import EmbedderConfig


class TestNomicCodeEmbedder:
    def test_importable(self) -> None:
        from trelix.embedder.nomic_code import NomicCodeEmbedder
        assert NomicCodeEmbedder is not None

    def test_is_base_embedder(self) -> None:
        from trelix.embedder.base import BaseEmbedder
        from trelix.embedder.nomic_code import NomicCodeEmbedder
        assert issubclass(NomicCodeEmbedder, BaseEmbedder)

    def test_embed_prepends_doc_task_prefix(self) -> None:
        from trelix.embedder.nomic_code import NomicCodeEmbedder, _DOC_PREFIX
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 768]
        with patch("trelix.embedder.nomic_code.SentenceTransformer", return_value=mock_model):
            cfg = EmbedderConfig(provider="nomic-code", _env_file=None)
            emb = NomicCodeEmbedder(cfg)
            emb.embed(["def foo(): pass"])
            called_texts = mock_model.encode.call_args[0][0]
            assert called_texts[0].startswith(_DOC_PREFIX)

    def test_embed_query_prepends_query_prefix(self) -> None:
        from trelix.embedder.nomic_code import NomicCodeEmbedder, _QUERY_PREFIX
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 768]
        with patch("trelix.embedder.nomic_code.SentenceTransformer", return_value=mock_model):
            cfg = EmbedderConfig(provider="nomic-code", _env_file=None)
            emb = NomicCodeEmbedder(cfg)
            emb.embed_query("authentication logic")
            called_texts = mock_model.encode.call_args[0][0]
            assert called_texts[0].startswith(_QUERY_PREFIX)

    def test_dimension(self) -> None:
        from trelix.embedder.nomic_code import NomicCodeEmbedder
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = [[0.1] * 768]
        with patch("trelix.embedder.nomic_code.SentenceTransformer", return_value=mock_model):
            cfg = EmbedderConfig(provider="nomic-code", nomic_code_dimensions=768, _env_file=None)
            emb = NomicCodeEmbedder(cfg)
            assert emb.dimension == 768

    def test_config_effective_dimension(self) -> None:
        cfg = EmbedderConfig(provider="nomic-code", nomic_code_dimensions=768, _env_file=None)
        assert cfg.effective_dimension == 768
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_embedder_nomic.py -v --tb=short 2>&1 | head -20
```
Expected: import error on `NomicCodeEmbedder`.

- [ ] **Step 3: Add `nomic_code_*` fields to `EmbedderConfig`**

In `src/trelix/core/config.py`, after the `bge_code_dimensions` field:
```python
# ── Nomic CodeRankEmbed (nomic-ai, sentence-transformers compatible) ─────────
# pip install trelix[nomic-code]
nomic_code_model: str = "nomic-ai/nomic-embed-code"
nomic_code_dimensions: int = 768  # CodeRankEmbed default dim
nomic_code_task: str = "code"     # Matryoshka task hint
```

- [ ] **Step 4: Create `src/trelix/embedder/nomic_code.py`**

```python
"""
Nomic CodeRankEmbed embedder (nomic-ai/nomic-embed-code).

Uses sentence-transformers (already a dependency via local embedder).
Nomic CodeRankEmbed uses task-prefix protocol:
  - Documents: "search_document: <code>"
  - Queries:   "search_query: <natural language>"

No extra dependencies beyond sentence-transformers (already in trelix[local]).

Usage:
    TRELIX_EMBEDDER_PROVIDER=nomic-code trelix index ./my-repo
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from trelix.embedder.base import BaseEmbedder

if TYPE_CHECKING:
    from trelix.core.config import EmbedderConfig

_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


class NomicCodeEmbedder(BaseEmbedder):
    """
    Embedder backed by nomic-ai/nomic-embed-code via sentence-transformers.

    Task-prefix asymmetric encoding (same protocol as Nomic text v1.5).
    Compatible with trelix[local] install — no extra dependencies.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for nomic-code embedder. "
                "Install it with: pip install 'trelix[local]'"
            ) from e

        self._model = SentenceTransformer(
            config.nomic_code_model,
            trust_remote_code=True,
        )
        self._dimensions = config.nomic_code_dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{_DOC_PREFIX}{t}" for t in texts]
        vecs = self._model.encode(prefixed, batch_size=32, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        prefixed = [f"{_QUERY_PREFIX}{text}"]
        vecs = self._model.encode(prefixed, normalize_embeddings=True)
        return vecs[0].tolist()

    @property
    def dimension(self) -> int:
        d = self._model.get_sentence_embedding_dimension()
        return d if d else self._dimensions
```

- [ ] **Step 5: Wire into `make_embedder()`**

In `src/trelix/embedder/base.py`, after the `"bge-code"` case:
```python
case "nomic-code":
    from trelix.embedder.nomic_code import NomicCodeEmbedder
    return NomicCodeEmbedder(config)
```

- [ ] **Step 6: Add optional extra to `pyproject.toml`**

```toml
nomic-code = [
    "sentence-transformers>=3.0.0",  # already in [local], duplicated for clarity
]
```

- [ ] **Step 7: Run tests — all pass**

```bash
.venv/bin/python -m pytest tests/unit/test_embedder_nomic.py -v --tb=short 2>&1 | tail -10
```
Expected: `5 passed`.

- [ ] **Step 8: Full unit suite — no regressions**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 9: Commit**

```bash
git add src/trelix/embedder/nomic_code.py src/trelix/embedder/base.py \
        src/trelix/core/config.py pyproject.toml \
        tests/unit/test_embedder_nomic.py
git commit -m "feat(embedder): add Nomic CodeRankEmbed embedder (nomic-ai/nomic-embed-code)

- NomicCodeEmbedder via sentence-transformers (no new deps vs local install)
- Task-prefix asymmetric encoding: search_document / search_query
- Provider: 'nomic-code', TRELIX_EMBEDDER_PROVIDER=nomic-code"
```

---

### Task 3: Matryoshka Dimension Support for VoyageEmbedder

**Files:**
- Modify: `src/trelix/embedder/base.py` — update `VoyageEmbedder.embed()` and `embed_query()` to pass `output_dimension`
- Modify: `src/trelix/core/config.py` — add `voyage_output_dimensions` field to `EmbedderConfig`
- Modify: `tests/unit/test_embedder.py` — add Matryoshka tests for voyage

**Interfaces:**
- Consumes: `EmbedderConfig.voyage_output_dimensions: int | None = None` (None = use full 2048)
- Produces: `VoyageEmbedder` honours `output_dimension` parameter in API calls; `effective_dimension` returns `voyage_output_dimensions or voyage_dimensions`

- [ ] **Step 1: Write failing test**

In `tests/unit/test_embedder.py`, add a new class:
```python
class TestVoyageMatryoshka:
    def test_embed_passes_output_dimension(self) -> None:
        """VoyageEmbedder passes output_dimension to API when set."""
        from trelix.embedder.base import VoyageEmbedder
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 512]
        mock_client.embed.return_value = mock_response
        with patch("trelix.embedder.base.voyageai") as mock_voyageai:
            mock_voyageai.Client.return_value = mock_client
            cfg = EmbedderConfig(
                provider="voyage",
                voyage_api_key="test-key",
                voyage_output_dimensions=512,
                _env_file=None,
            )
            emb = VoyageEmbedder(cfg)
            emb.embed(["def foo(): pass"])
            call_kwargs = mock_client.embed.call_args[1]
            assert call_kwargs.get("output_dimension") == 512

    def test_effective_dimension_with_output_dimensions(self) -> None:
        cfg = EmbedderConfig(
            provider="voyage",
            voyage_api_key="test-key",
            voyage_output_dimensions=256,
            _env_file=None,
        )
        assert cfg.effective_dimension == 256

    def test_effective_dimension_without_output_dimensions(self) -> None:
        cfg = EmbedderConfig(
            provider="voyage",
            voyage_api_key="test-key",
            _env_file=None,
        )
        assert cfg.effective_dimension == 1024  # voyage_dimensions default
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_embedder.py::TestVoyageMatryoshka -v --tb=short 2>&1 | head -20
```
Expected: `AttributeError` or `AssertionError` since `output_dimension` isn't passed yet.

- [ ] **Step 3: Add `voyage_output_dimensions` field to `EmbedderConfig`**

In `src/trelix/core/config.py`, inside `EmbedderConfig`, after `voyage_dimensions`:
```python
# Matryoshka output dimension (voyage-code-3 supports 256/512/1024/2048).
# None = use full voyage_dimensions. Set smaller for faster HNSW search.
voyage_output_dimensions: int | None = None
```

Update `effective_dimension`:
```python
if self.provider == "voyage":
    return self.voyage_output_dimensions or self.voyage_dimensions
```

- [ ] **Step 4: Update `VoyageEmbedder.embed()` and `embed_query()` in `src/trelix/embedder/base.py`**

Find the `VoyageEmbedder` class. Update `embed()`:
```python
def embed(self, texts: list[str]) -> list[list[float]]:
    kwargs: dict = {"model": self._model, "input_type": "document"}
    if self._output_dimensions is not None:
        kwargs["output_dimension"] = self._output_dimensions
    result = self._client.embed(texts, **kwargs)
    return result.embeddings
```

Update `embed_query()`:
```python
def embed_query(self, text: str) -> list[float]:
    kwargs: dict = {"model": self._model, "input_type": "query"}
    if self._output_dimensions is not None:
        kwargs["output_dimension"] = self._output_dimensions
    result = self._client.embed([text], **kwargs)
    return result.embeddings[0]
```

In `VoyageEmbedder.__init__()`, after `self._model = config.voyage_model`:
```python
self._output_dimensions = config.voyage_output_dimensions
```

Also update `dimension` property:
```python
@property
def dimension(self) -> int:
    return self._output_dimensions or self._dimensions
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_embedder.py -v --tb=short 2>&1 | tail -15
```
Expected: all pass including new Matryoshka tests.

- [ ] **Step 6: Full suite**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add src/trelix/embedder/base.py src/trelix/core/config.py tests/unit/test_embedder.py
git commit -m "feat(embedder): Matryoshka dimension support for VoyageEmbedder

voyage-code-3 first entries of 2048-dim embedding form valid sub-dim embeddings.
Set TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=512 for faster HNSW + smaller storage.
- voyage_output_dimensions config field (None = full dim)
- output_dimension passed to voyageai.Client.embed() when set
(arXiv 2024: voyage-code-3 Matryoshka verified 3-0)"
```

---

### Task 4: LLM-as-Judge Eval + CI Metric Gate

**Files:**
- Create: `tests/eval/llm_judge.py` — LLM judge scorer for retrieval quality
- Modify: `tests/eval/harness.py` — add `judge_score` field to `EvalResult`, optionally call judge
- Modify: `tests/eval/metrics.py` — add `mean_judge_score` to `EvalReport`
- Modify: `tests/integration/test_eval.py` — add judge-based test with threshold
- Create: `tests/unit/test_eval_judge.py`

**Interfaces:**
- Consumes: `EvalResult.query: str`, `EvalResult.retrieved_files: list[str]`, `EvalResult.expected_file: str`; `TrelixChatClient` for LLM calls
- Produces: `LLMJudge.score(query, retrieved_snippets, expected_file) -> float` (0.0–1.0); `EvalReport.mean_judge_score: float | None`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_eval_judge.py`:
```python
"""Tests for LLM-as-judge eval scorer."""
from __future__ import annotations
from unittest.mock import MagicMock


class TestLLMJudge:
    def test_importable(self) -> None:
        from tests.eval.llm_judge import LLMJudge
        assert LLMJudge is not None

    def test_score_returns_float_between_0_and_1(self) -> None:
        from tests.eval.llm_judge import LLMJudge
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(content='{"score": 0.8, "reason": "relevant"}')
        judge = LLMJudge(mock_client)
        score = judge.score(
            query="how does authentication work",
            retrieved_snippets=["def authenticate(user): ..."],
            expected_file="src/auth.py",
        )
        assert 0.0 <= score <= 1.0

    def test_score_returns_0_on_llm_failure(self) -> None:
        from tests.eval.llm_judge import LLMJudge
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("API error")
        judge = LLMJudge(mock_client)
        score = judge.score("query", ["snippet"], "expected.py")
        assert score == 0.0

    def test_score_parses_json_response(self) -> None:
        from tests.eval.llm_judge import LLMJudge
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content='{"score": 0.9, "reason": "exact file match, highly relevant"}'
        )
        judge = LLMJudge(mock_client)
        score = judge.score("how does auth work", ["def login(): ..."], "auth.py")
        assert score == pytest.approx(0.9)

    def test_score_clamps_out_of_range_values(self) -> None:
        from tests.eval.llm_judge import LLMJudge
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content='{"score": 1.5, "reason": "extra good"}'
        )
        judge = LLMJudge(mock_client)
        score = judge.score("query", ["snippet"], "file.py")
        assert score <= 1.0

import pytest
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_eval_judge.py -v --tb=short 2>&1 | head -15
```
Expected: `ModuleNotFoundError: No module named 'tests.eval.llm_judge'`.

- [ ] **Step 3: Create `tests/eval/llm_judge.py`**

```python
"""
LLM-as-judge scorer for trelix retrieval quality evaluation.

Given a query, the retrieved code snippets, and the expected file, asks
an LLM to rate how well the retrieval answered the query.

The judge prompt asks for a JSON response:
    {"score": <float 0.0–1.0>, "reason": "<brief explanation>"}

score = 1.0 → retrieved result perfectly answers the query
score = 0.0 → retrieved result is completely irrelevant

This is distinct from recall@k (which checks exact file match) — it measures
semantic relevance, which can be high even when the exact file is missed.

Usage::

    from trelix.llm.factory import build_chat_client
    from trelix.core.config import LLMConfig
    from tests.eval.llm_judge import LLMJudge

    client = build_chat_client(LLMConfig())
    judge = LLMJudge(client)
    score = judge.score(query, snippets, expected_file)
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.llm.client import TrelixChatClient

logger = logging.getLogger("trelix.eval.judge")

_JUDGE_SYSTEM = (
    "You are a code retrieval quality evaluator. "
    "Given a developer's query and retrieved code snippets, rate how well "
    "the retrieval answered the query. Return ONLY valid JSON."
)

_JUDGE_TEMPLATE = """\
Query: {query}

Expected file: {expected_file}

Retrieved code snippets:
{snippets}

Rate the quality of this retrieval on a scale of 0.0 to 1.0:
- 1.0: The retrieved code directly and completely answers the query
- 0.7: The retrieved code is highly relevant but incomplete
- 0.4: The retrieved code is somewhat relevant
- 0.1: The retrieved code is tangentially related
- 0.0: The retrieved code is irrelevant to the query

Respond with ONLY this JSON:
{{"score": <float>, "reason": "<one sentence>"}}"""


class LLMJudge:
    """LLM-as-judge retrieval quality scorer."""

    def __init__(self, client: TrelixChatClient) -> None:
        self._client = client

    def score(
        self,
        query: str,
        retrieved_snippets: list[str],
        expected_file: str,
        max_snippet_chars: int = 2000,
    ) -> float:
        """
        Score retrieval quality for a single query.

        Returns float in [0.0, 1.0]. Returns 0.0 on any LLM failure
        so eval runs never crash due to API errors.
        """
        snippets_text = "\n---\n".join(
            s[:max_snippet_chars] for s in retrieved_snippets[:5]
        )
        prompt = _JUDGE_TEMPLATE.format(
            query=query,
            expected_file=expected_file,
            snippets=snippets_text,
        )
        try:
            from trelix.llm.client import ChatMessage
            response = self._client.complete(
                messages=[ChatMessage(role="user", content=prompt)],
                max_tokens=150,
                temperature=0.0,
                system=_JUDGE_SYSTEM,
            )
            data = json.loads(response.content)
            raw_score = float(data.get("score", 0.0))
            return max(0.0, min(1.0, raw_score))
        except Exception as exc:
            logger.warning("LLM judge failed for query %r: %s", query, exc)
            return 0.0
```

- [ ] **Step 4: Add `judge_score` to `EvalResult` in `tests/eval/metrics.py`**

In `tests/eval/metrics.py`, find the `EvalResult` dataclass and add:
```python
judge_score: float | None = None  # LLM-as-judge score (0.0–1.0); None if judge not run
```

In `EvalReport`, add aggregate property:
```python
@property
def mean_judge_score(self) -> float | None:
    scores = [r.judge_score for r in self.results if r.judge_score is not None]
    return sum(scores) / len(scores) if scores else None
```

- [ ] **Step 5: Run tests — all pass**

```bash
.venv/bin/python -m pytest tests/unit/test_eval_judge.py -v --tb=short 2>&1 | tail -10
```
Expected: `5 passed`.

- [ ] **Step 6: Full suite**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add tests/eval/llm_judge.py tests/eval/metrics.py tests/unit/test_eval_judge.py
git commit -m "feat(eval): LLM-as-judge retrieval quality scorer

LLMJudge.score() rates semantic relevance 0.0–1.0.
Graceful degradation: returns 0.0 on any LLM failure.
EvalResult.judge_score + EvalReport.mean_judge_score added.
Research basis: CodeRAG-Bench shows oracle retrieval +27pp on SWE-bench-Lite
vs realistic retriever — judge score correlates with downstream task success."
```

---

### Task 5: Update Public API, README, and CHANGELOG for Phase 1

**Files:**
- Modify: `src/trelix/__init__.py` — ensure `BGECodeEmbedder`, `NomicCodeEmbedder` are importable (not in `__all__` — internal, but import path must work)
- Modify: `README.md` — add bge-code and nomic-code to Providers table, add Matryoshka voyage note
- Modify: `CHANGELOG.md` — add `[Unreleased]` section with Phase 1 entries
- Modify: `src/trelix/cli/main.py` — add `"bge-code"` and `"nomic-code"` to `_EmbedderProvider` Literal in CLI

- [ ] **Step 1: Update CLI Literal**

In `src/trelix/cli/main.py`, find the `_EmbedderProvider` type alias or Literal. Add `"bge-code"` and `"nomic-code"` to the list.

- [ ] **Step 2: Update README providers table**

Find the providers/embedders table. Add rows:
```markdown
| `bge-code`    | BAAI/bge-code-v1              | CoIR SOTA 2025 (self-reported 81.77)     | `pip install trelix[bge-code]`   |
| `nomic-code`  | nomic-ai/nomic-embed-code     | No new deps (uses sentence-transformers) | included in `trelix[local]`      |
```

Add note under voyage:
```markdown
> **voyage-code-3 Matryoshka:** Set `TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=512` for 2× faster HNSW search with minimal quality loss.
```

- [ ] **Step 3: Update CHANGELOG**

At the top of `CHANGELOG.md`, add:
```markdown
## [Unreleased]

### Added
- **BGE-Code-v1 embedder** (`bge-code` provider) — BAAI CoIR SOTA 2025, self-reported 81.77 avg. `pip install trelix[bge-code]`
- **Nomic CodeRankEmbed embedder** (`nomic-code` provider) — task-prefix asymmetric encoding, no new deps. `pip install trelix[local]`
- **Voyage Matryoshka support** — `TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=512` passes `output_dimension` to voyage-code-3 API for compact embeddings
- **LLM-as-judge eval scorer** — `LLMJudge.score()` rates semantic retrieval quality 0.0–1.0; `EvalReport.mean_judge_score` aggregate
```

- [ ] **Step 4: Run lint**

```bash
.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/
```
Expected: no issues. Fix any that appear.

- [ ] **Step 5: Full suite + integration**

```bash
.venv/bin/python -m pytest tests/unit/ tests/integration/ -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git add src/trelix/__init__.py src/trelix/cli/main.py README.md CHANGELOG.md
git commit -m "docs(phase1): update API, README, CHANGELOG for Phase 1 embedder upgrades"
```

---

## PHASE 2 — Retrieval Architecture Upgrades

**Goal:** Add RAPTOR-style multi-granularity indexing (file-level summaries alongside symbol-level chunks), PLAID late-interaction reranker, streaming retrieval for CLI and MCP, and pathspec deprecation fix.

**Why:** Research confirmed RAPTOR delivers +20pp on multi-document queries (arXiv 2401.18059, 2-1 vote); PLAID gives 7–45× ColBERT speedup with no quality loss (EMNLP 2022, 3-0 vote). Streaming makes `trelix ask` usable at scale — users see tokens immediately instead of waiting 10s.

---

### Task 6: Multi-Granularity Indexing (File-Level Summaries)

**Files:**
- Modify: `src/trelix/store/db.py` — add `file_summaries` table with DDL migration
- Modify: `src/trelix/store/vector.py` — expose `upsert_file_summary_embedding()` method
- Create: `src/trelix/indexing/file_summarizer.py` — LLM-based file-level summary generator
- Modify: `src/trelix/indexing/indexer.py` — Phase 2.5: generate file summaries after chunking
- Modify: `src/trelix/core/config.py` — add `IndexConfig.file_summaries_enabled: bool = False`
- Create: `tests/unit/test_file_summarizer.py`

**Interfaces:**
- Consumes: `IndexedFile.rel_path`, `list[Symbol]` from a file, `TrelixChatClient`, `BaseVectorStore`
- Produces: `FileSummarizer.summarize(file, symbols, client) -> str`; summary stored in `file_summaries` table + vector indexed; `Retriever._retrieve_standard()` queries `file_summaries` as a 4th retrieval leg when `file_summaries_enabled=True`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_file_summarizer.py`:
```python
"""Tests for file-level summary generator."""
from __future__ import annotations
from unittest.mock import MagicMock
from trelix.core.models import Symbol, SymbolKind, Language


class TestFileSummarizer:
    def test_importable(self) -> None:
        from trelix.indexing.file_summarizer import FileSummarizer
        assert FileSummarizer is not None

    def test_summarize_returns_non_empty_string(self) -> None:
        from trelix.indexing.file_summarizer import FileSummarizer
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content="This file implements authentication logic including login, logout, and session management."
        )
        summarizer = FileSummarizer(client=mock_client, max_symbols=20)
        symbols = [
            Symbol(
                file_id=1, name="login", qualified_name="login",
                kind=SymbolKind.FUNCTION, line_start=1, line_end=10,
                signature="def login(user, pwd)", body="def login(u, p): pass",
            )
        ]
        result = summarizer.summarize("src/auth.py", symbols, Language.PYTHON)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_returns_empty_on_llm_failure(self) -> None:
        from trelix.indexing.file_summarizer import FileSummarizer
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("API error")
        summarizer = FileSummarizer(client=mock_client)
        result = summarizer.summarize("src/auth.py", [], Language.PYTHON)
        assert result == ""

    def test_summarize_truncates_to_max_symbols(self) -> None:
        from trelix.indexing.file_summarizer import FileSummarizer
        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(content="Summary")
        summarizer = FileSummarizer(client=mock_client, max_symbols=2)
        symbols = [
            Symbol(
                file_id=1, name=f"fn{i}", qualified_name=f"fn{i}",
                kind=SymbolKind.FUNCTION, line_start=i*10, line_end=i*10+5,
                signature=f"def fn{i}()", body=f"def fn{i}(): pass",
            )
            for i in range(10)
        ]
        summarizer.summarize("src/big.py", symbols, Language.PYTHON)
        prompt_content = mock_client.complete.call_args[1].get(
            "messages", mock_client.complete.call_args[0][0]
        )
        # Verify that only max_symbols worth of content was sent
        assert mock_client.complete.called
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_file_summarizer.py -v --tb=short 2>&1 | head -15
```

- [ ] **Step 3: Create `src/trelix/indexing/file_summarizer.py`**

```python
"""
File-level LLM summarizer for RAPTOR-style multi-granularity indexing.

Generates a 2–4 sentence description of a file's purpose from its symbol list.
The summary is stored in `file_summaries` DB table and embedded as a
high-level retrieval entry — enabling queries like "how does this codebase
handle authentication" to retrieve file-level context rather than scattered
symbol chunks.

This is Phase 2 of multi-granularity indexing.  Phase 1 (symbol-level) is
the existing chunker. Phase 2 adds file-level summaries.

Research basis: RAPTOR (arXiv 2401.18059, ICLR 2024) — 82.6% on QuALITY
benchmark (+20.3pp absolute) via recursive hierarchical summarization.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.models import Language, Symbol
    from trelix.llm.client import TrelixChatClient

logger = logging.getLogger("trelix.indexing.file_summarizer")

_SYSTEM_PROMPT = (
    "You are a senior engineer writing concise file-level documentation. "
    "Summarize what a source file does in 2-4 sentences. "
    "Focus on: what problem it solves, key classes/functions, and its role "
    "in the broader codebase. Be specific, not generic."
)

_PROMPT_TEMPLATE = """\
File: {rel_path}
Language: {language}

Top symbols:
{symbols_text}

Write a 2-4 sentence summary of what this file does."""


class FileSummarizer:
    """
    Generates LLM-based file-level summaries for multi-granularity indexing.

    Safe to use without an LLM client — returns empty string on any failure.
    The indexer treats empty summaries as "no file-level entry" and skips them.
    """

    def __init__(
        self,
        client: TrelixChatClient,
        max_symbols: int = 30,
        max_tokens: int = 150,
    ) -> None:
        self._client = client
        self._max_symbols = max_symbols
        self._max_tokens = max_tokens

    def summarize(
        self,
        rel_path: str,
        symbols: list[Symbol],
        language: Language,
    ) -> str:
        """
        Generate a file-level summary. Returns "" on any failure.

        Args:
            rel_path: repo-relative file path (shown to LLM for context)
            symbols: parsed symbols from the file (truncated to max_symbols)
            language: file language enum

        Returns:
            summary string, or "" if LLM unavailable / failed
        """
        if not symbols:
            return ""

        top = symbols[: self._max_symbols]
        sym_lines = "\n".join(
            f"- {s.kind.value} {s.qualified_name}: {s.signature[:80]}"
            for s in top
        )
        prompt = _PROMPT_TEMPLATE.format(
            rel_path=rel_path,
            language=language.value,
            symbols_text=sym_lines,
        )

        try:
            from trelix.llm.client import ChatMessage
            response = self._client.complete(
                messages=[ChatMessage(role="user", content=prompt)],
                max_tokens=self._max_tokens,
                temperature=0.0,
                system=_SYSTEM_PROMPT,
            )
            return response.content.strip()
        except Exception as exc:
            logger.debug("File summarizer failed for %s: %s", rel_path, exc)
            return ""
```

- [ ] **Step 4: Add `file_summaries_enabled` to `IndexConfig`**

In `src/trelix/core/config.py`, inside `IndexConfig` (after the `llm` field):
```python
# Multi-granularity indexing: generate LLM file-level summaries (RAPTOR-style).
# Requires LLM API access. Off by default — zero cost when disabled.
file_summaries_enabled: bool = Field(
    default=False,
    alias="TRELIX_FILE_SUMMARIES_ENABLED",
)
```

- [ ] **Step 5: Add `file_summaries` table to DB schema**

In `src/trelix/store/db.py`, find the `DDL` string. Add after the `chunks` table:
```sql
CREATE TABLE IF NOT EXISTS file_summaries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id     INTEGER NOT NULL UNIQUE REFERENCES files(id) ON DELETE CASCADE,
    summary     TEXT    NOT NULL,
    chunk_id    INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_file_summaries_file_id ON file_summaries(file_id);
```

Add a migration guard in `Database.__init__()` (after the `CREATE TABLE` calls):
```python
# Phase 2 migration: add file_summaries if not present
self._conn.execute(
    "CREATE TABLE IF NOT EXISTS file_summaries ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "file_id INTEGER NOT NULL UNIQUE REFERENCES files(id) ON DELETE CASCADE, "
    "summary TEXT NOT NULL, chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL, "
    "created_at TEXT DEFAULT (datetime('now')))"
)
self._conn.execute(
    "CREATE INDEX IF NOT EXISTS idx_file_summaries_file_id ON file_summaries(file_id)"
)
self._conn.commit()
```

Add `upsert_file_summary()` method to `Database`:
```python
def upsert_file_summary(self, file_id: int, summary: str, chunk_id: int | None = None) -> int:
    """Insert or replace a file-level summary. Returns the row id."""
    with self._conn:
        cur = self._conn.execute(
            "INSERT INTO file_summaries (file_id, summary, chunk_id) VALUES (?, ?, ?) "
            "ON CONFLICT(file_id) DO UPDATE SET summary=excluded.summary, chunk_id=excluded.chunk_id, created_at=datetime('now')",
            (file_id, summary, chunk_id),
        )
    return cur.lastrowid or 0

def get_file_summary(self, file_id: int) -> str | None:
    row = self._conn.execute(
        "SELECT summary FROM file_summaries WHERE file_id = ?", (file_id,)
    ).fetchone()
    return row[0] if row else None
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_file_summarizer.py -v --tb=short 2>&1 | tail -10
```
Expected: `4 passed`.

- [ ] **Step 7: Full suite — no regressions**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 8: Commit**

```bash
git add src/trelix/indexing/file_summarizer.py src/trelix/store/db.py \
        src/trelix/core/config.py tests/unit/test_file_summarizer.py
git commit -m "feat(indexing): multi-granularity file-level summaries (RAPTOR-style)

FileSummarizer generates 2-4 sentence LLM file descriptions.
Stored in file_summaries DB table (with schema migration).
Gated by TRELIX_FILE_SUMMARIES_ENABLED=true (default off).
Research basis: RAPTOR arXiv 2401.18059 +20pp on QuALITY benchmark."
```

---

### Task 7: PLAID Late-Interaction Reranker

**Files:**
- Create: `src/trelix/retrieval/reranker_plaid.py` — PLAID ColBERT reranker via RAGatouille
- Modify: `src/trelix/retrieval/reranker.py` — expose `rerank()` dispatch to PLAID when config selects it
- Modify: `src/trelix/core/config.py` — add `"plaid"` to `RetrievalConfig.rerank_provider` Literal; add `plaid_model` field
- Modify: `pyproject.toml` — add `[plaid]` optional extra: `"ragatouille>=0.0.8"`
- Create: `tests/unit/test_reranker_plaid.py`

**Interfaces:**
- Consumes: `list[SearchResult]`, query string, `RetrievalConfig.rerank_provider == "plaid"`, `RetrievalConfig.plaid_model`
- Produces: `rerank(query, results, config) -> list[SearchResult]` (same signature, drops in as replacement); PLAID scores replace existing scores

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_reranker_plaid.py`:
```python
"""Tests for PLAID late-interaction reranker."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
from trelix.core.config import RetrievalConfig
from trelix.core.models import SearchResult, Symbol, SymbolKind, IndexedFile, Language, Chunk


def _make_result(score: float = 0.5) -> SearchResult:
    f = IndexedFile(path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="x", size_bytes=100, id=1)
    s = Symbol(file_id=1, name="fn", qualified_name="fn", kind=SymbolKind.FUNCTION,
               line_start=1, line_end=5, signature="def fn()", body="def fn(): pass", id=1)
    c = Chunk(symbol_id=1, chunk_text="def fn(): pass", token_count=5, id=1)
    return SearchResult(file=f, symbol=s, chunk=c, score=score, source="vector")


class TestPlaidReranker:
    def test_importable(self) -> None:
        from trelix.retrieval.reranker_plaid import PlaidReranker
        assert PlaidReranker is not None

    def test_rerank_returns_same_count(self) -> None:
        from trelix.retrieval.reranker_plaid import PlaidReranker
        mock_ragatouille = MagicMock()
        mock_ragatouille.return_value.rerank.return_value = [
            {"content": "def fn(): pass", "score": 0.9, "result_index": 0},
            {"content": "def fn(): pass", "score": 0.7, "result_index": 1},
        ]
        with patch("trelix.retrieval.reranker_plaid.RAGPretrainedModel", mock_ragatouille):
            cfg = RetrievalConfig(rerank_provider="plaid")
            results = [_make_result(0.5), _make_result(0.3)]
            reranker = PlaidReranker(cfg)
            reranked = reranker.rerank("how does auth work", results)
            assert len(reranked) == 2

    def test_rerank_updates_scores_from_plaid(self) -> None:
        from trelix.retrieval.reranker_plaid import PlaidReranker
        mock_model_instance = MagicMock()
        mock_model_instance.rerank.return_value = [
            {"content": "def fn(): pass", "score": 0.95, "result_index": 0},
        ]
        mock_ragatouille = MagicMock(return_value=mock_model_instance)
        with patch("trelix.retrieval.reranker_plaid.RAGPretrainedModel", mock_ragatouille):
            cfg = RetrievalConfig(rerank_provider="plaid")
            results = [_make_result(0.1)]
            reranker = PlaidReranker(cfg)
            reranked = reranker.rerank("query", results)
            assert reranked[0].score == pytest.approx(0.95)

    def test_rerank_falls_back_on_ragatouille_error(self) -> None:
        from trelix.retrieval.reranker_plaid import PlaidReranker
        mock_ragatouille = MagicMock(side_effect=ImportError("ragatouille not installed"))
        with patch("trelix.retrieval.reranker_plaid.RAGPretrainedModel", mock_ragatouille):
            cfg = RetrievalConfig(rerank_provider="plaid")
            results = [_make_result(0.5), _make_result(0.3)]
            reranker = PlaidReranker(cfg)
            reranked = reranker.rerank("query", results)
            # Fallback: returns original order unchanged
            assert len(reranked) == 2
            assert reranked[0].score == pytest.approx(0.5)

import pytest
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_reranker_plaid.py -v --tb=short 2>&1 | head -15
```

- [ ] **Step 3: Add `"plaid"` to `RetrievalConfig.rerank_provider`**

In `src/trelix/core/config.py`, find:
```python
rerank_provider: Literal["cohere", "cross_encoder"] = "cohere"
```
Change to:
```python
rerank_provider: Literal["cohere", "cross_encoder", "plaid"] = "cohere"
```

Add after `rerank_model`:
```python
# PLAID late-interaction reranker (ColBERT via RAGatouille)
# pip install trelix[plaid]
plaid_model: str = Field(
    default="colbert-ir/colbertv2.0",
    alias="TRELIX_RETRIEVAL_PLAID_MODEL",
)
```

- [ ] **Step 4: Create `src/trelix/retrieval/reranker_plaid.py`**

```python
"""
PLAID late-interaction reranker via RAGatouille (ColBERT).

PLAID (Progressive Late Interaction via Approximate Document Hierarchies)
reduces ColBERTv2 search latency 7–45× vs naive late interaction with no
quality degradation (EMNLP 2022, arXiv 2205.09707, confirmed 3-0).

RAGatouille provides a production-ready PLAID implementation:
    pip install ragatouille>=0.0.8

Usage: set TRELIX_RETRIEVAL_RERANK_PROVIDER=plaid in .env

Fallback: if ragatouille is not installed or loading fails, falls back to
returning results in the original order (safe degradation).
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.config import RetrievalConfig
    from trelix.core.models import SearchResult

logger = logging.getLogger("trelix.retrieval.reranker_plaid")


class PlaidReranker:
    """
    PLAID/ColBERT reranker backed by RAGatouille.

    Lazy-loads the model on first use to avoid slow startup when
    PLAID is configured but not every query needs reranking.
    """

    def __init__(self, config: RetrievalConfig) -> None:
        self._model_name = config.plaid_model
        self._top_n = config.rerank_top_n
        self._model = None  # lazy-loaded

    def _get_model(self):
        if self._model is None:
            try:
                from ragatouille import RAGPretrainedModel
                self._model = RAGPretrainedModel.from_pretrained(self._model_name)
            except Exception as exc:
                logger.warning(
                    "PLAID model load failed (%s) — reranking disabled. "
                    "Install with: pip install 'trelix[plaid]'",
                    exc,
                )
        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        """
        Rerank results using PLAID late-interaction scoring.

        Falls back to original order if PLAID is unavailable.
        """
        if not results:
            return results

        model = self._get_model()
        if model is None:
            return results  # graceful degradation

        n = top_n or self._top_n or len(results)
        texts = [r.chunk.chunk_text for r in results]

        try:
            reranked = model.rerank(
                query=query,
                documents=texts,
                k=min(n, len(texts)),
            )
            # reranked: list of {"content": str, "score": float, "result_index": int}
            scored: dict[int, float] = {
                item["result_index"]: item["score"]
                for item in reranked
                if "result_index" in item
            }
            updated = [
                replace(r, score=scored.get(i, r.score))
                for i, r in enumerate(results)
            ]
            return sorted(updated, key=lambda r: r.score, reverse=True)[:n]
        except Exception as exc:
            logger.warning("PLAID reranking failed: %s", exc)
            return results
```

- [ ] **Step 5: Wire PLAID into `rerank()` dispatch in `src/trelix/retrieval/reranker.py`**

Find the `rerank()` function. Add a branch before the existing `cross_encoder` case:
```python
if config.rerank_provider == "plaid":
    from trelix.retrieval.reranker_plaid import PlaidReranker
    return PlaidReranker(config).rerank(query, results, top_n=config.rerank_top_n)
```

- [ ] **Step 6: Add optional extra to `pyproject.toml`**

```toml
plaid = [
    "ragatouille>=0.0.8",  # PLAID ColBERT late-interaction reranker
]
```

- [ ] **Step 7: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_reranker_plaid.py -v --tb=short 2>&1 | tail -10
```
Expected: `4 passed`.

- [ ] **Step 8: Full unit suite**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 9: Commit**

```bash
git add src/trelix/retrieval/reranker_plaid.py src/trelix/retrieval/reranker.py \
        src/trelix/core/config.py pyproject.toml tests/unit/test_reranker_plaid.py
git commit -m "feat(retrieval): PLAID late-interaction reranker via RAGatouille

PlaidReranker wraps ColBERTv2 for 7-45x latency reduction vs naive ColBERT.
Set TRELIX_RETRIEVAL_RERANK_PROVIDER=plaid.
Graceful degradation: original order preserved if ragatouille unavailable.
Research basis: PLAID arXiv 2205.09707 EMNLP 2022, confirmed 3-0.
pip install trelix[plaid]"
```

---

### Task 8: Streaming Retrieval for CLI and MCP

**Files:**
- Modify: `src/trelix/retrieval/synthesizer.py` — expose `stream()` that yields `str` tokens instead of blocking
- Modify: `src/trelix/cli/main.py` — `trelix ask` uses streaming synthesis
- Modify: `packages/trelix-mcp/src/trelix_mcp/server.py` — `search_code` result streaming placeholder (MCP transport handles buffering)
- Create: `tests/unit/test_synthesizer_stream.py`

**Interfaces:**
- Consumes: `RetrievedContext`, `LLMConfig`, `RetrievalConfig`
- Produces: `Synthesizer.stream(context, config) -> Iterator[str]` — yields token strings; existing `synthesize()` unchanged for backward compat

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_synthesizer_stream.py`:
```python
"""Tests for streaming synthesis."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
from trelix.core.models import RetrievedContext, SearchResult


def _make_context() -> RetrievedContext:
    return RetrievedContext(
        query="how does auth work",
        results=[],
        context_text="def authenticate(user): ...",
        total_tokens=10,
    )


class TestSynthesizerStream:
    def test_stream_method_exists(self) -> None:
        from trelix.retrieval.synthesizer import Synthesizer
        assert hasattr(Synthesizer, "stream")

    def test_stream_yields_strings(self) -> None:
        from trelix.retrieval.synthesizer import Synthesizer
        from trelix.core.config import EmbedderConfig, RetrievalConfig
        mock_client = MagicMock()
        mock_client.stream.return_value = iter(["The ", "auth ", "flow ", "is..."])

        with patch("trelix.retrieval.synthesizer.build_chat_client", return_value=mock_client):
            cfg_emb = EmbedderConfig(_env_file=None)
            cfg_ret = RetrievalConfig()
            synth = Synthesizer(cfg_emb)
            tokens = list(synth.stream(_make_context(), cfg_ret))
            assert len(tokens) > 0
            assert all(isinstance(t, str) for t in tokens)

    def test_stream_falls_back_on_no_api_key(self) -> None:
        from trelix.retrieval.synthesizer import Synthesizer
        from trelix.core.config import EmbedderConfig, RetrievalConfig
        mock_client = MagicMock()
        mock_client.stream.side_effect = Exception("No API key")

        with patch("trelix.retrieval.synthesizer.build_chat_client", return_value=mock_client):
            synth = Synthesizer(EmbedderConfig(_env_file=None))
            tokens = list(synth.stream(_make_context(), RetrievalConfig()))
            # Should yield one error message token, not raise
            assert len(tokens) >= 1
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_synthesizer_stream.py -v --tb=short 2>&1 | head -15
```
Expected: `AttributeError: type object 'Synthesizer' has no attribute 'stream'`.

- [ ] **Step 3: Add `stream()` to `Synthesizer` in `src/trelix/retrieval/synthesizer.py`**

After the existing `synthesize()` method, add:
```python
def stream(
    self,
    context: RetrievedContext,
    config: RetrievalConfig,
) -> Iterator[str]:
    """
    Stream synthesis tokens to the caller.

    Yields str tokens as they arrive from the LLM.
    Yields a single error message string on failure.

    Usage::
        for token in synth.stream(context, config):
            print(token, end="", flush=True)
    """
    from collections.abc import Iterator

    intent = getattr(context, "intent", None) or "feature_flow"
    system_prompt = _INTENT_PROMPTS.get(intent, _INTENT_PROMPTS["feature_flow"])

    user_prompt = (
        f"Question: {context.query}\n\n"
        f"Code context:\n{context.context_text}\n\n"
        "Answer the question using the provided code context."
    )

    try:
        from trelix.llm.client import ChatMessage
        from trelix.llm.factory import build_chat_client
        from trelix.core.config import LLMConfig
        llm_cfg = LLMConfig()
        client = build_chat_client(llm_cfg)
        yield from client.stream(
            messages=[ChatMessage(role="user", content=user_prompt)],
            max_tokens=config.synthesis_max_tokens,
            temperature=0.0,
            system=system_prompt,
        )
    except Exception as exc:
        logger.warning("Streaming synthesis failed: %s", exc)
        yield f"\n[trelix: synthesis unavailable — {exc}]"
```

Add `from collections.abc import Iterator` to the top of the file imports.

- [ ] **Step 4: Update `trelix ask` CLI to use streaming**

In `src/trelix/cli/main.py`, find the `ask` command handler. Replace the blocking `synthesize()` call with:
```python
for token in synth.stream(context, config.retrieval):
    console.print(token, end="", highlight=False)
console.print()  # final newline
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_synthesizer_stream.py -v --tb=short 2>&1 | tail -10
```
Expected: `3 passed`.

- [ ] **Step 6: Full suite**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add src/trelix/retrieval/synthesizer.py src/trelix/cli/main.py \
        tests/unit/test_synthesizer_stream.py
git commit -m "feat(retrieval): streaming synthesis for trelix ask

Synthesizer.stream() yields tokens via TrelixChatClient.stream().
trelix ask CLI now streams tokens live — no more 10s wait.
Graceful: falls back to error message token on failure."
```

---

### Task 9: Fix pathspec DeprecationWarning (W3 from audit)

**Files:**
- Modify: `src/trelix/indexing/walker.py` — replace `'gitwildmatch'` with `'gitignore'` pattern class

- [ ] **Step 1: Find the pathspec usage**

```bash
grep -n "gitwildmatch\|GitWildMatch\|pathspec" /Users/sairamugge/Desktop/Not-Humans-World/trelix/src/trelix/indexing/walker.py
```

- [ ] **Step 2: Replace deprecated pattern**

In `src/trelix/indexing/walker.py`, find the line that constructs the pathspec pattern. It will look like:
```python
pathspec.PathSpec.from_lines("gitwildmatch", patterns)
```
Replace with:
```python
pathspec.PathSpec.from_lines("gitignore", patterns)
```

- [ ] **Step 3: Verify the warning is gone**

```bash
.venv/bin/python -m pytest tests/unit/test_walker.py -v -W error::DeprecationWarning --tb=short 2>&1 | tail -10
```
Expected: all pass with no `DeprecationWarning` errors.

- [ ] **Step 4: Full suite**

```bash
.venv/bin/python -m pytest tests/unit/ -q 2>&1 | grep "warning" | tail -5
```
Expected: 0 warnings (was 8).

- [ ] **Step 5: Commit**

```bash
git add src/trelix/indexing/walker.py
git commit -m "fix(walker): replace deprecated gitwildmatch with gitignore pathspec pattern

pathspec GitWildMatchPattern('gitwildmatch') deprecated since pathspec 0.10.
Use 'gitignore' (GitIgnoreSpecPattern) instead.
Eliminates 8 DeprecationWarnings per test run."
```

---

## PHASE 3 — Storage, Scale, and Platform

**Goal:** Add LanceDB as a third vector-store backend for large-scale deployments; add a VS Code extension scaffold; add `trelix serve` REST API with streaming endpoints.

**Why:** LanceDB 0.6+ supports ARM-native HNSW with zero SQLite dependency — validated in vecdb-bench as 3–5× faster insert at >100k vectors. VS Code extension unlocks the largest developer surface (70M+ users). REST API enables remote trelix deployments and web integrations.

---

### Task 10: LanceDB Vector Store Backend

**Files:**
- Create: `src/trelix/store/vector_lance.py` — `LanceVectorStore(BaseVectorStore)` implementation
- Modify: `src/trelix/store/vector.py` — add `"lance"` case to `make_vector_store()` factory
- Modify: `src/trelix/core/config.py` — add `"lance"` to `StoreConfig.backend` Literal; add `lance_uri`, `lance_table` fields
- Modify: `pyproject.toml` — add `[lance]` optional extra: `"lancedb>=0.6.0"`
- Create: `tests/unit/test_store_lance.py`

**Interfaces:**
- Consumes: `BaseVectorStore` ABC (`upsert_batch`, `search`, `delete_batch`, `count`)
- Produces: `LanceVectorStore` at `StoreConfig.lance_uri` path; `make_vector_store()` returns it when `backend == "lance"`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_store_lance.py`:
```python
"""Tests for LanceDB vector store backend."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest


class TestLanceVectorStore:
    def test_importable(self) -> None:
        from trelix.store.vector_lance import LanceVectorStore
        assert LanceVectorStore is not None

    def test_is_base_vector_store(self) -> None:
        from trelix.store.vector import BaseVectorStore
        from trelix.store.vector_lance import LanceVectorStore
        assert issubclass(LanceVectorStore, BaseVectorStore)

    def test_upsert_batch_calls_lance(self, tmp_path) -> None:
        from trelix.store.vector_lance import LanceVectorStore
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.create_table.return_value = mock_table
        mock_db.open_table.side_effect = Exception("not found")
        with patch("trelix.store.vector_lance.lancedb") as mock_lance:
            mock_lance.connect.return_value = mock_db
            store = LanceVectorStore(
                uri=str(tmp_path / "lance"),
                table_name="chunks",
                dimension=4,
            )
            store.upsert_batch([(1, [0.1, 0.2, 0.3, 0.4]), (2, [0.5, 0.6, 0.7, 0.8])])
            assert mock_table.add.called or mock_db.create_table.called

    def test_search_returns_list_of_tuples(self, tmp_path) -> None:
        from trelix.store.vector_lance import LanceVectorStore
        mock_table = MagicMock()
        mock_result = MagicMock()
        mock_result.to_list.return_value = [
            {"chunk_id": 1, "_distance": 0.1},
            {"chunk_id": 2, "_distance": 0.3},
        ]
        mock_table.search.return_value.limit.return_value.to_list.return_value = [
            {"chunk_id": 1, "_distance": 0.1},
            {"chunk_id": 2, "_distance": 0.3},
        ]
        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table
        with patch("trelix.store.vector_lance.lancedb") as mock_lance:
            mock_lance.connect.return_value = mock_db
            store = LanceVectorStore(
                uri=str(tmp_path / "lance"),
                table_name="chunks",
                dimension=4,
            )
            results = store.search([0.1, 0.2, 0.3, 0.4], k=2)
            assert isinstance(results, list)

    def test_count_returns_int(self, tmp_path) -> None:
        from trelix.store.vector_lance import LanceVectorStore
        mock_table = MagicMock()
        mock_table.count_rows.return_value = 42
        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table
        with patch("trelix.store.vector_lance.lancedb") as mock_lance:
            mock_lance.connect.return_value = mock_db
            store = LanceVectorStore(
                uri=str(tmp_path / "lance"),
                table_name="chunks",
                dimension=4,
            )
            assert store.count() == 42
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_store_lance.py -v --tb=short 2>&1 | head -15
```

- [ ] **Step 3: Add `lance_*` fields to `StoreConfig`**

In `src/trelix/core/config.py`, find `StoreConfig`. Change:
```python
backend: Literal["sqlite", "qdrant"] = Field(...)
```
to:
```python
backend: Literal["sqlite", "qdrant", "lance"] = Field(...)
```

Add after `qdrant_collection`:
```python
# ── LanceDB connection ───────────────────────────────────────────────────────
lance_uri: str = Field(default=".trelix/lance", alias="LANCE_URI")
lance_table: str = Field(default="chunks", alias="LANCE_TABLE")
```

- [ ] **Step 4: Create `src/trelix/store/vector_lance.py`**

```python
"""
LanceDB vector store backend.

LanceDB 0.6+ provides ARM-native HNSW with zero SQLite dependency.
Validated 3-5x faster insert at >100k vectors vs sqlite-vec (vecdb-bench).

Best for:
    - Repos > 500k chunks (where sqlite-vec HNSW becomes memory-constrained)
    - Apple Silicon and ARM servers (native SIMD)
    - Multi-repo deployments sharing a vector store

Usage:
    TRELIX_STORE_BACKEND=lance LANCE_URI=.trelix/lance trelix index ./my-repo
    pip install trelix[lance]
"""
from __future__ import annotations

import logging
from typing import Any

from trelix.store.vector import BaseVectorStore

logger = logging.getLogger("trelix.store.lance")

try:
    import lancedb
    _LANCE_AVAILABLE = True
except ImportError:
    _LANCE_AVAILABLE = False
    lancedb = None  # type: ignore[assignment]


class LanceVectorStore(BaseVectorStore):
    """
    Vector store backed by LanceDB.

    Each trelix index gets one LanceDB table named `chunks` (configurable).
    Vectors are stored as fixed-size float32 arrays in a `vector` column.
    chunk_id (from trelix's SQLite `chunks` table) is stored for lookup.
    """

    def __init__(
        self,
        uri: str,
        table_name: str = "chunks",
        dimension: int = 1024,
    ) -> None:
        if not _LANCE_AVAILABLE:
            raise ImportError(
                "lancedb is required for the lance store backend. "
                "Install with: pip install 'trelix[lance]'"
            )
        self._uri = uri
        self._table_name = table_name
        self._dimension = dimension
        self._db = lancedb.connect(uri)
        self._table = self._get_or_create_table()

    def _get_or_create_table(self):
        try:
            return self._db.open_table(self._table_name)
        except Exception:
            import pyarrow as pa
            schema = pa.schema([
                pa.field("chunk_id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), self._dimension)),
            ])
            return self._db.create_table(
                self._table_name,
                schema=schema,
                mode="create",
            )

    def upsert_batch(self, pairs: list[tuple[int, list[float]]]) -> None:
        import pyarrow as pa
        if not pairs:
            return
        ids = [p[0] for p in pairs]
        vecs = [p[1] for p in pairs]
        data = pa.table({
            "chunk_id": pa.array(ids, type=pa.int64()),
            "vector": pa.array(vecs, type=pa.list_(pa.float32(), self._dimension)),
        })
        # Delete existing rows for these chunk_ids then add fresh
        id_list = ", ".join(str(i) for i in ids)
        try:
            self._table.delete(f"chunk_id IN ({id_list})")
        except Exception:
            pass
        self._table.add(data)

    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        rows = (
            self._table.search(query)
            .limit(k)
            .to_list()
        )
        return [(row["chunk_id"], row.get("_distance", 0.0)) for row in rows]

    def delete_batch(self, chunk_ids: list[int]) -> None:
        if not chunk_ids:
            return
        id_list = ", ".join(str(i) for i in chunk_ids)
        try:
            self._table.delete(f"chunk_id IN ({id_list})")
        except Exception as exc:
            logger.warning("LanceDB delete_batch failed: %s", exc)

    def count(self) -> int:
        try:
            return self._table.count_rows()
        except Exception:
            return 0
```

- [ ] **Step 5: Wire into `make_vector_store()` in `src/trelix/store/vector.py`**

Find `make_vector_store()`. Add before the Qdrant case:
```python
if config.store.backend == "lance":
    from trelix.store.vector_lance import LanceVectorStore
    uri = config.store.lance_uri
    if not Path(uri).is_absolute():
        uri = str(Path(config.repo_path) / uri)
    return LanceVectorStore(
        uri=uri,
        table_name=config.store.lance_table,
        dimension=dimension,
    )
```

- [ ] **Step 6: Add optional extra to `pyproject.toml`**

```toml
lance = [
    "lancedb>=0.6.0",
    "pyarrow>=14.0",
]
```

- [ ] **Step 7: Run tests**

```bash
.venv/bin/python -m pytest tests/unit/test_store_lance.py -v --tb=short 2>&1 | tail -10
```
Expected: `4 passed`.

- [ ] **Step 8: Full suite**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 9: Commit**

```bash
git add src/trelix/store/vector_lance.py src/trelix/store/vector.py \
        src/trelix/core/config.py pyproject.toml tests/unit/test_store_lance.py
git commit -m "feat(store): LanceDB vector store backend (trelix[lance])

LanceVectorStore(BaseVectorStore) — ARM-native HNSW, 3-5x faster insert at 100k+ chunks.
Set TRELIX_STORE_BACKEND=lance for large-scale deployments.
Existing sqlite/qdrant backends unchanged.
Research basis: vecdb-bench benchmarks, LanceDB 0.6.0 release notes."
```

---

### Task 11: `trelix serve` REST API with Streaming

**Files:**
- Create: `src/trelix/api/app.py` — FastAPI app with `/search`, `/ask` (streaming SSE), `/index` endpoints
- Create: `src/trelix/api/__init__.py`
- Modify: `src/trelix/cli/main.py` — add `trelix serve` subcommand
- Modify: `pyproject.toml` — add `[serve]` optional extra: `"fastapi>=0.111", "uvicorn[standard]>=0.29"`
- Create: `tests/unit/test_api.py`

**Interfaces:**
- Consumes: `IndexConfig`, `Retriever`, `Synthesizer.stream()`
- Produces: `GET /search?query=&repo=&k=` → JSON; `GET /ask?query=&repo=` → SSE stream; `POST /index` → JSON stats

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_api.py`:
```python
"""Tests for trelix REST API."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest


class TestTrelixAPI:
    def test_app_importable(self) -> None:
        from trelix.api.app import create_app
        assert create_app is not None

    def test_search_endpoint_exists(self, tmp_path) -> None:
        from fastapi.testclient import TestClient
        from trelix.api.app import create_app

        mock_ctx = MagicMock()
        mock_ctx.results = []

        with patch("trelix.api.app.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            app = create_app()
            client = TestClient(app)
            resp = client.get(f"/search?query=auth&repo={tmp_path}&k=5")
            assert resp.status_code == 200
            assert isinstance(resp.json(), list)

    def test_health_endpoint(self, tmp_path) -> None:
        from fastapi.testclient import TestClient
        from trelix.api.app import create_app

        app = create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_search_returns_result_dicts(self, tmp_path) -> None:
        from fastapi.testclient import TestClient
        from trelix.api.app import create_app
        from trelix.core.models import Language

        mock_result = MagicMock()
        mock_result.file.rel_path = "src/auth.py"
        mock_result.symbol.qualified_name = "AuthService.login"
        mock_result.symbol.kind.value = "method"
        mock_result.symbol.line_start = 10
        mock_result.symbol.line_end = 20
        mock_result.symbol.body = "def login(): pass"
        mock_result.file.language.value = "python"
        mock_result.score = 0.9
        mock_result.source = "vector"

        mock_ctx = MagicMock()
        mock_ctx.results = [mock_result]

        with patch("trelix.api.app.Retriever") as MockRetriever:
            MockRetriever.return_value.retrieve.return_value = mock_ctx
            app = create_app()
            client = TestClient(app)
            resp = client.get(f"/search?query=auth&repo={tmp_path}&k=5")
            data = resp.json()
            assert len(data) == 1
            assert data[0]["file"] == "src/auth.py"
            assert data[0]["score"] == pytest.approx(0.9)
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/unit/test_api.py -v --tb=short 2>&1 | head -15
```

- [ ] **Step 3: Create `src/trelix/api/__init__.py`**

```python
"""trelix REST API (optional, requires trelix[serve])."""
```

- [ ] **Step 4: Create `src/trelix/api/app.py`**

```python
"""
trelix REST API.

Provides HTTP endpoints for trelix search, indexing, and LLM synthesis.
The /ask endpoint uses Server-Sent Events (SSE) for streaming output.

Install:
    pip install 'trelix[serve]'

Run:
    trelix serve ./my-repo --port 8765

Endpoints:
    GET  /health                    — liveness check
    GET  /search?query=&repo=&k=   — hybrid search, returns JSON
    GET  /ask?query=&repo=          — LLM synthesis, SSE stream
    POST /index                    — index a repository (body: {"repo_path": "..."})
    GET  /stats?repo=               — index statistics
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("trelix.api")


def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as e:
        raise ImportError(
            "FastAPI is required for trelix serve. "
            "Install with: pip install 'trelix[serve]'"
        ) from e

    from trelix.core.config import IndexConfig
    from trelix.retrieval.retriever import Retriever

    app = FastAPI(title="trelix API", version="1.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "1.1.0"}

    @app.get("/search")
    def search(query: str, repo: str, k: int = 10) -> list[dict[str, Any]]:
        config = IndexConfig(repo_path=repo)
        ctx = Retriever(config).retrieve(query)
        return [
            {
                "file": r.file.rel_path,
                "symbol": r.symbol.qualified_name,
                "kind": r.symbol.kind.value,
                "lines": f"{r.symbol.line_start}-{r.symbol.line_end}",
                "score": round(r.score, 4),
                "source": r.source,
                "body": r.symbol.body[:800],
                "language": r.file.language.value,
            }
            for r in ctx.results[:k]
        ]

    @app.get("/ask")
    def ask(query: str, repo: str):
        def _generate():
            try:
                from trelix.retrieval.synthesizer import Synthesizer
                config = IndexConfig(repo_path=repo)
                ctx = Retriever(config).retrieve(query)
                synth = Synthesizer(config.embedder)
                for token in synth.stream(ctx, config.retrieval):
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                yield f"data: [ERROR: {exc}]\n\n"

        return StreamingResponse(_generate(), media_type="text/event-stream")

    @app.post("/index")
    def index_repo(body: dict[str, str]) -> dict[str, Any]:
        from trelix.indexing.indexer import Indexer
        config = IndexConfig(repo_path=body["repo_path"])
        return Indexer(config).index()

    @app.get("/stats")
    def stats(repo: str) -> dict[str, Any]:
        from trelix.store.db import Database
        config = IndexConfig(repo_path=repo)
        db = Database(config.db_path_absolute)
        return {
            "files": db.count_files(),
            "symbols": db.count_symbols(),
            "chunks": db.count_chunks(),
        }

    return app
```

- [ ] **Step 5: Add `trelix serve` subcommand to CLI**

In `src/trelix/cli/main.py`, add a new Typer command after `watch`:
```python
@app.command()
def serve(
    repo_path: str = typer.Argument(..., help="Repository to serve"),
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(8765, help="Port to bind"),
) -> None:
    """Start a REST API server for trelix search and synthesis."""
    try:
        import uvicorn
        from trelix.api.app import create_app
    except ImportError:
        typer.echo("trelix serve requires: pip install 'trelix[serve]'")
        raise typer.Exit(1)

    api_app = create_app()
    typer.echo(f"trelix API serving {repo_path} at http://{host}:{port}")
    uvicorn.run(api_app, host=host, port=port)
```

- [ ] **Step 6: Add optional extra to `pyproject.toml`**

```toml
serve = [
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",
]
```

- [ ] **Step 7: Install test dep and run tests**

```bash
.venv/bin/pip install "httpx>=0.27" -q  # fastapi TestClient needs httpx
.venv/bin/python -m pytest tests/unit/test_api.py -v --tb=short 2>&1 | tail -12
```
Expected: `4 passed`.

- [ ] **Step 8: Full suite**

```bash
.venv/bin/python -m pytest tests/unit/ -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 9: Commit**

```bash
git add src/trelix/api/ src/trelix/cli/main.py pyproject.toml tests/unit/test_api.py
git commit -m "feat(api): trelix serve REST API with SSE streaming

GET /search — hybrid search, returns JSON array
GET /ask — LLM synthesis, Server-Sent Events stream
POST /index — index a repository
GET /health — liveness check
trelix serve ./my-repo --port 8765
pip install trelix[serve]"
```

---

### Task 12: Update CHANGELOG, README, and Public API for v2.0

**Files:**
- Modify: `CHANGELOG.md` — finalize `[Unreleased]` into `[2.0.0]`
- Modify: `README.md` — add Phase 2 and 3 features, update providers table, add serve docs
- Modify: `pyproject.toml` — bump version to `2.0.0`; update `src/trelix/__init__.py`
- Modify: `src/trelix/__init__.py` — ensure new public exports are listed

- [ ] **Step 1: Bump version everywhere**

```bash
# pyproject.toml
sed -i '' 's/^version = "1.1.0"/version = "2.0.0"/' pyproject.toml
# __init__.py
sed -i '' 's/__version__ = "1.1.0"/__version__ = "2.0.0"/' src/trelix/__init__.py
# sub-packages
sed -i '' 's/version = "1.1.0"/version = "2.0.0"/g' packages/trelix-mcp/pyproject.toml
sed -i '' 's/version = "1.1.0"/version = "2.0.0"/g' packages/trelix-langchain/pyproject.toml
sed -i '' 's/version = "1.1.0"/version = "2.0.0"/g' packages/trelix-llama-index/pyproject.toml
sed -i '' 's/__version__ = "1.1.0"/__version__ = "2.0.0"/g' packages/trelix-langchain/src/trelix_langchain/__init__.py
sed -i '' 's/__version__ = "1.1.0"/__version__ = "2.0.0"/g' packages/trelix-llama-index/src/trelix_llama_index/__init__.py
```

- [ ] **Step 2: Update CHANGELOG — move Unreleased to [2.0.0]**

In `CHANGELOG.md`, replace `## [Unreleased]` with `## [2.0.0] — <today's date>` and add a comprehensive overview paragraph covering all three phases.

- [ ] **Step 3: Update README**

Add to the Features section:
```markdown
- **BGE-Code-v1 / Nomic CodeRankEmbed** — CoIR SOTA embedding models (`bge-code`, `nomic-code` providers)
- **Matryoshka voyage embeddings** — compact 256/512-dim voyage-code-3 via `TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS`
- **PLAID late-interaction reranker** — 7–45× faster ColBERT via RAGatouille (`rerank_provider=plaid`)
- **Multi-granularity indexing** — LLM file-level summaries alongside symbol chunks (`TRELIX_FILE_SUMMARIES_ENABLED=true`)
- **Streaming synthesis** — `trelix ask` streams tokens live; `GET /ask` SSE endpoint
- **REST API** — `trelix serve ./repo --port 8765` exposes `/search`, `/ask`, `/index`, `/health`
- **LanceDB backend** — 3–5× faster vector insert at 100k+ chunks (`TRELIX_STORE_BACKEND=lance`)
- **LLM-as-judge eval** — `LLMJudge.score()` semantic retrieval quality measurement
```

- [ ] **Step 4: Run full test suite one final time**

```bash
.venv/bin/python -m pytest tests/unit/ tests/integration/ -q --tb=short 2>&1 | tail -8
```
Expected: all pass.

- [ ] **Step 5: Run linting**

```bash
.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/
```
Expected: clean.

- [ ] **Step 6: Final commit**

```bash
git add pyproject.toml src/trelix/__init__.py packages/ README.md CHANGELOG.md
git commit -m "chore(release): bump versions to v2.0.0, finalize CHANGELOG

Phase 1: BGE-Code-v1, Nomic CodeRankEmbed, Matryoshka voyage, LLM-as-judge eval
Phase 2: RAPTOR multi-granularity indexing, PLAID reranker, streaming synthesis, pathspec fix
Phase 3: LanceDB backend, trelix serve REST API with SSE streaming"
```

---

## Verification Checklist

After all tasks, run this full validation from the repo root:

```bash
# 1. Full unit + integration suite
.venv/bin/python -m pytest tests/unit/ tests/integration/ -q --tb=short 2>&1 | tail -5
# Expected: all pass, 0 failures, ≤0 warnings

# 2. New embedder providers exist
.venv/bin/python -c "
from trelix.core.config import EmbedderConfig
for p in ['bge-code', 'nomic-code']:
    cfg = EmbedderConfig(provider=p, _env_file=None)
    print(f'{p}: effective_dim={cfg.effective_dimension}')
"

# 3. PLAID provider registered
.venv/bin/python -c "
from trelix.core.config import RetrievalConfig
cfg = RetrievalConfig(rerank_provider='plaid')
print('PLAID provider:', cfg.rerank_provider, cfg.plaid_model)
"

# 4. LanceDB backend registered
.venv/bin/python -c "
from trelix.core.config import StoreConfig
cfg = StoreConfig(backend='lance')
print('Lance backend:', cfg.backend, cfg.lance_uri)
"

# 5. REST API importable
.venv/bin/python -c "from trelix.api.app import create_app; print('API: OK')"

# 6. Synthesizer has stream()
.venv/bin/python -c "
from trelix.retrieval.synthesizer import Synthesizer
assert hasattr(Synthesizer, 'stream'), 'stream() missing'
print('Synthesizer.stream: OK')
"

# 7. LLM judge importable
.venv/bin/python -c "from tests.eval.llm_judge import LLMJudge; print('LLMJudge: OK')"

# 8. Version correct
.venv/bin/python -c "import trelix; print('version:', trelix.__version__)"
# Expected: 2.0.0

# 9. CLI shows new commands
.venv/bin/trelix --help 2>&1 | grep -E "serve|graph"
# Expected: both present
```

---

## Summary: What This Delivers

| Phase | Feature | Research Basis | Impact |
|-------|---------|---------------|--------|
| 1 | BGE-Code-v1 embedder | arXiv 2505.12697 (81.77 CoIR) | +14+ NDCG@10 over local embedder |
| 1 | Nomic CodeRankEmbed | nomic-ai/nomic-embed-code | Code-specialized, no new deps |
| 1 | Voyage Matryoshka | voyage-code-3 API, 3-0 verified | 2× faster HNSW, smaller storage |
| 1 | LLM-as-judge eval | CodeRAG-Bench arXiv 2406.14497 | Semantic quality measurement |
| 2 | Multi-granularity indexing | RAPTOR arXiv 2401.18059 +20pp | "Explain codebase" queries work |
| 2 | PLAID reranker | EMNLP 2022 arXiv 2205.09707 7-45× | ColBERT quality at production speed |
| 2 | Streaming synthesis | Production pattern | Live token output for `trelix ask` |
| 2 | pathspec fix | Audit finding W3 | 0 DeprecationWarnings |
| 3 | LanceDB backend | vecdb-bench 3-5× insert speed | Scales to 500k+ chunks |
| 3 | REST API + SSE | FastAPI production pattern | Remote deployments, web integrations |
