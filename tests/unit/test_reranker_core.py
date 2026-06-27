"""
Unit tests for trelix.retrieval.reranker -- core paths.

The reranker module exposes a single `rerank()` function that dispatches on
config.rerank_provider:

  "cross_encoder"  -> local sentence-transformers CrossEncoder
  "cohere"         -> Cohere Rerank HTTP API (requires requests + api_key)
  anything else    -> identity pass-through (returns results[:top_n] unchanged)

Tests use only stdlib + pytest + unittest.mock -- no network, no GPU.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import RetrievalConfig
from trelix.core.models import Chunk, IndexedFile, Language, SearchResult, Symbol, SymbolKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER = "placeholder-for-tests"


def _make_symbol(name: str = "foo") -> Symbol:
    return Symbol(
        file_id=1,
        name=name,
        qualified_name=name,
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=5,
        signature=f"def {name}():",
        body=f"def {name}():\n    pass",
    )


def _make_file() -> IndexedFile:
    return IndexedFile(
        path="/repo/src/foo.py",
        rel_path="src/foo.py",
        language=Language.PYTHON,
        hash="deadbeef",
        size_bytes=128,
    )


def _make_result(text: str, score: float = 0.5, rank: int = 1) -> SearchResult:
    chunk = Chunk(
        symbol_id=1,
        chunk_text=text,
        token_count=len(text.split()),
    )
    return SearchResult(
        chunk=chunk,
        symbol=_make_symbol(),
        file=_make_file(),
        score=score,
        rank=rank,
        source="vector",
    )


def _cfg(**kwargs) -> RetrievalConfig:
    """Build a RetrievalConfig with env-loading disabled (model_construct skips validators)."""
    return RetrievalConfig.model_construct(**kwargs)


# ---------------------------------------------------------------------------
# Provider = None / unknown -> identity pass-through
# ---------------------------------------------------------------------------


class TestNoReranking:
    def test_unknown_provider_returns_unchanged(self) -> None:
        """An unrecognised provider returns results[:top_n] in original order."""
        from trelix.retrieval.reranker import rerank

        results = [_make_result(f"doc {i}", score=float(i)) for i in range(5)]
        cfg = _cfg(rerank_provider="none_provider")  # type: ignore[arg-type]
        out = rerank("query", results, cfg, top_n=3)
        assert out == results[:3]

    def test_unknown_provider_respects_top_n(self) -> None:
        from trelix.retrieval.reranker import rerank

        results = [_make_result(f"d{i}") for i in range(8)]
        cfg = _cfg(rerank_provider="unknown")  # type: ignore[arg-type]
        out = rerank("q", results, cfg, top_n=4)
        assert len(out) == 4
        assert out == results[:4]

    def test_empty_results_returns_empty_for_cohere(self) -> None:
        """Empty input -> empty output for cohere provider."""
        from trelix.retrieval.reranker import rerank

        cfg = _cfg(
            rerank_provider="cohere",
            cohere_api_key=_PLACEHOLDER,
            cohere_endpoint="http://x",
        )
        out = rerank("query", [], cfg, top_n=10)
        assert out == []

    def test_empty_results_cross_encoder_provider(self) -> None:
        """Empty input -> empty output for cross_encoder provider."""
        from trelix.retrieval.reranker import rerank

        cfg = _cfg(rerank_provider="cross_encoder", rerank_model="cross-encoder/x")
        out = rerank("query", [], cfg, top_n=5)
        assert out == []

    def test_empty_results_unknown_provider(self) -> None:
        """Empty input -> empty output regardless of provider."""
        from trelix.retrieval.reranker import rerank

        cfg = _cfg(rerank_provider="noop")  # type: ignore[arg-type]
        out = rerank("query", [], cfg, top_n=5)
        assert out == []


# ---------------------------------------------------------------------------
# Provider = cross_encoder -> _cross_encoder_rerank
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    def test_reorders_by_score(self) -> None:
        """CrossEncoder scores override original order; results come out highest-first."""
        from trelix.retrieval.reranker import rerank

        results = [
            _make_result("doc low",  score=0.9, rank=1),
            _make_result("doc high", score=0.1, rank=2),
            _make_result("doc mid",  score=0.5, rank=3),
        ]
        # predict returns scores positionally: low=0.1, high=0.9, mid=0.5
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]

        mock_ce_cls = MagicMock(return_value=mock_model)
        mock_st_module = MagicMock()
        mock_st_module.CrossEncoder = mock_ce_cls

        cfg = _cfg(rerank_provider="cross_encoder", rerank_model="cross-encoder/mock")

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            out = rerank("find high", results, cfg, top_n=3)

        assert len(out) == 3
        assert out[0].chunk.chunk_text == "doc high"
        assert out[1].chunk.chunk_text == "doc mid"
        assert out[2].chunk.chunk_text == "doc low"

    def test_ranks_are_reassigned(self) -> None:
        """rank field must be 1-indexed after reranking."""
        from trelix.retrieval.reranker import rerank

        results = [_make_result(f"doc {i}") for i in range(3)]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.3, 0.1, 0.9]
        mock_ce_cls = MagicMock(return_value=mock_model)
        mock_st_module = MagicMock()
        mock_st_module.CrossEncoder = mock_ce_cls

        cfg = _cfg(rerank_provider="cross_encoder", rerank_model="cross-encoder/mock")

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            out = rerank("q", results, cfg, top_n=3)

        assert [r.rank for r in out] == [1, 2, 3]

    def test_top_n_truncates(self) -> None:
        """Only top_n results are returned even when more are available."""
        from trelix.retrieval.reranker import rerank

        results = [_make_result(f"doc {i}") for i in range(5)]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.4, 0.3, 0.8, 0.1]
        mock_ce_cls = MagicMock(return_value=mock_model)
        mock_st_module = MagicMock()
        mock_st_module.CrossEncoder = mock_ce_cls

        cfg = _cfg(rerank_provider="cross_encoder", rerank_model="cross-encoder/mock")

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            out = rerank("q", results, cfg, top_n=2)

        assert len(out) == 2

    def test_import_error_falls_back_gracefully(self) -> None:
        """Missing sentence-transformers -> warning logged, returns results[:top_n] unchanged."""
        from trelix.retrieval.reranker import rerank

        results = [_make_result(f"doc {i}") for i in range(4)]
        cfg = _cfg(rerank_provider="cross_encoder", rerank_model="cross-encoder/missing")

        # Force ImportError inside the reranker by making the import fail
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            out = rerank("q", results, cfg, top_n=3)

        assert len(out) == 3
        assert out == results[:3]

    def test_does_not_mutate_originals(self) -> None:
        """Reranking produces new SearchResult objects, leaving originals untouched."""
        from trelix.retrieval.reranker import rerank

        results = [_make_result(f"doc {i}", score=0.5, rank=i + 1) for i in range(3)]
        original_scores = [r.score for r in results]
        original_ranks  = [r.rank  for r in results]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.1, 0.5]
        mock_ce_cls = MagicMock(return_value=mock_model)
        mock_st_module = MagicMock()
        mock_st_module.CrossEncoder = mock_ce_cls

        cfg = _cfg(rerank_provider="cross_encoder", rerank_model="cross-encoder/mock")

        with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            rerank("q", results, cfg, top_n=3)

        assert [r.score for r in results] == original_scores
        assert [r.rank  for r in results] == original_ranks


# ---------------------------------------------------------------------------
# Provider = cohere -> _cohere_rerank
# ---------------------------------------------------------------------------


class TestCohereReranker:
    # requests is imported *locally* inside _cohere_rerank, so we must inject it
    # via sys.modules, not as a module-level attribute of the reranker.

    def _cohere_response(self, order: list[int], scores: list[float]) -> MagicMock:
        """Build a mock requests.Response for the Cohere Rerank endpoint."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"index": idx, "relevance_score": score}
                for idx, score in zip(order, scores)
            ]
        }
        return mock_resp

    def _cohere_cfg(self, **extra: object) -> RetrievalConfig:
        return _cfg(
            rerank_provider="cohere",
            cohere_api_key=_PLACEHOLDER,
            cohere_endpoint="https://fake.endpoint/rerank",
            cohere_rerank_model="Cohere-rerank-v4.0-pro",
            **extra,
        )

    def _mock_requests_module(self, resp: MagicMock) -> MagicMock:
        """Build a mock 'requests' module whose .post() returns resp."""
        mock_requests = MagicMock()
        mock_requests.post.return_value = resp
        mock_requests.exceptions.SSLError = OSError
        mock_requests.exceptions.ConnectionError = OSError
        mock_requests.exceptions.Timeout = OSError
        return mock_requests

    def test_reorders_by_score(self) -> None:
        """Cohere API result order (by index) is respected."""
        from trelix.retrieval.reranker import rerank

        results = [
            _make_result("doc A", rank=1),
            _make_result("doc B", rank=2),
            _make_result("doc C", rank=3),
        ]
        # Cohere returns index 2 (doc C) first, then index 0 (doc A), then index 1 (doc B)
        mock_resp = self._cohere_response(order=[2, 0, 1], scores=[0.95, 0.80, 0.60])
        mock_req_mod = self._mock_requests_module(mock_resp)

        with patch.dict(sys.modules, {"requests": mock_req_mod}):
            out = rerank("query", results, self._cohere_cfg(), top_n=3)

        assert len(out) == 3
        assert out[0].chunk.chunk_text == "doc C"
        assert out[1].chunk.chunk_text == "doc A"
        assert out[2].chunk.chunk_text == "doc B"

    def test_relevance_scores_are_assigned(self) -> None:
        """Returned SearchResult.score equals the relevance_score from Cohere."""
        from trelix.retrieval.reranker import rerank

        results = [_make_result("doc X"), _make_result("doc Y")]
        mock_resp = self._cohere_response(order=[1, 0], scores=[0.99, 0.42])
        mock_req_mod = self._mock_requests_module(mock_resp)

        with patch.dict(sys.modules, {"requests": mock_req_mod}):
            out = rerank("q", results, self._cohere_cfg(), top_n=2)

        assert pytest.approx(out[0].score) == 0.99
        assert pytest.approx(out[1].score) == 0.42

    def test_missing_api_key_falls_back(self) -> None:
        """No api_key -> warning logged, results[:top_n] returned unchanged."""
        from trelix.retrieval.reranker import rerank

        results = [_make_result(f"d{i}") for i in range(5)]
        cfg = _cfg(
            rerank_provider="cohere",
            cohere_api_key=None,
            cohere_endpoint="http://x/rerank",
        )

        # requests present, but no api_key -> early return before any HTTP call
        mock_req_mod = MagicMock()
        with patch.dict(sys.modules, {"requests": mock_req_mod}):
            out = rerank("q", results, cfg, top_n=3)

        assert out == results[:3]

    def test_import_error_requests_falls_back(self) -> None:
        """Missing requests library -> warning logged, results[:top_n] returned unchanged."""
        from trelix.retrieval.reranker import rerank

        results = [_make_result(f"d{i}") for i in range(5)]
        cfg = self._cohere_cfg()

        with patch.dict(sys.modules, {"requests": None}):
            out = rerank("q", results, cfg, top_n=3)

        assert out == results[:3]

    def test_ranks_are_reassigned(self) -> None:
        """rank must be 1-indexed on returned results."""
        from trelix.retrieval.reranker import rerank

        results = [_make_result(f"doc {i}") for i in range(3)]
        mock_resp = self._cohere_response(order=[2, 0, 1], scores=[0.9, 0.8, 0.7])
        mock_req_mod = self._mock_requests_module(mock_resp)

        with patch.dict(sys.modules, {"requests": mock_req_mod}):
            out = rerank("q", results, self._cohere_cfg(), top_n=3)

        assert [r.rank for r in out] == [1, 2, 3]
