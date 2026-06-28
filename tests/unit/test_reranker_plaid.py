"""Tests for PLAID late-interaction reranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import RetrievalConfig
from trelix.core.models import Chunk, IndexedFile, Language, SearchResult, Symbol, SymbolKind


def _make_result(score: float = 0.5) -> SearchResult:
    f = IndexedFile(
        path="/r/a.py",
        rel_path="a.py",
        language=Language.PYTHON,
        hash="x",
        size_bytes=100,
        id=1,
    )
    s = Symbol(
        file_id=1,
        name="fn",
        qualified_name="fn",
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=5,
        signature="def fn()",
        body="def fn(): pass",
        id=1,
    )
    c = Chunk(symbol_id=1, chunk_text="def fn(): pass", token_count=5, id=1)
    return SearchResult(file=f, symbol=s, chunk=c, score=score, rank=1, source="vector")


class TestPlaidReranker:
    def test_importable(self) -> None:
        from trelix.retrieval.reranker_plaid import PlaidReranker

        assert PlaidReranker is not None

    def test_rerank_returns_same_count(self) -> None:
        from trelix.retrieval.reranker_plaid import PlaidReranker

        mock_model_instance = MagicMock()
        mock_model_instance.rerank.return_value = [
            {"content": "def fn(): pass", "score": 0.9, "result_index": 0},
            {"content": "def fn(): pass", "score": 0.7, "result_index": 1},
        ]
        mock_ragatouille = MagicMock(return_value=mock_model_instance)
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
