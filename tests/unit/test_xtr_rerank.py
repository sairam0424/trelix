"""
Unit tests for the XTR provider path in trelix.retrieval.reranker.

The XTR path (`rerank_provider="xtr"`) is a *degenerate* reranker: it feeds
xtr_score_documents() a query-token dict containing exactly one synthetic token
whose retrieved scores ARE the input scores, so the XTR average collapses to
sum([s])/1 == s. These tests pin that measured fact down in two directions:

  1. Characterisation -- the real (unmocked) scorer returns each document's own
     input score, so the output is a plain descending sort of the input. Anyone
     who later wires a token-level embedder should expect these to fail.
  2. Regression -- because the path is a no-op, it must SAY so (log.warning at
     selection time) rather than look like reranking is happening.

Plus two ordering/aliasing defects that only bite once the scorer is real:
doc_ids must be keyed by position (not by dataclass value equality), and the
caller's SearchResult objects must not be mutated in place.

Tests use only stdlib + pytest + unittest.mock -- no network, no GPU.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

from trelix.core.config import RetrievalConfig
from trelix.core.models import Chunk, IndexedFile, Language, SearchResult, Symbol, SymbolKind
from trelix.retrieval.reranker import rerank

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(text: str, score: float, rank: int) -> SearchResult:
    return SearchResult(
        chunk=Chunk(symbol_id=1, chunk_text=text, token_count=len(text.split())),
        symbol=Symbol(
            file_id=1,
            name="foo",
            qualified_name="foo",
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=5,
            signature="def foo():",
            body="def foo():\n    pass",
        ),
        file=IndexedFile(
            path="/repo/src/foo.py",
            rel_path="src/foo.py",
            language=Language.PYTHON,
            hash="deadbeef",
            size_bytes=128,
        ),
        score=score,
        rank=rank,
        source="vector",
    )


def _xtr_cfg() -> RetrievalConfig:
    """RetrievalConfig with env-loading disabled (model_construct skips validators)."""
    return RetrievalConfig.model_construct(rerank_provider="xtr", xtr_candidate_tokens=100)


def _score_by_doc_id(scores: dict[int, float]):
    """Stand-in for xtr_score_documents honouring the doc_id -> score contract."""

    def _scorer(
        query_token_scores: dict[int, list[tuple[int, float]]],
        candidate_doc_ids: list[int],
        k_impute: float,
    ) -> list[tuple[int, float]]:
        return sorted(
            ((cid, scores.get(cid, 0.0)) for cid in candidate_doc_ids),
            key=lambda pair: pair[1],
            reverse=True,
        )

    return _scorer


# ---------------------------------------------------------------------------
# Characterisation: the XTR path is a mathematical identity
# ---------------------------------------------------------------------------


class TestXTRIsDegenerate:
    def test_xtr_score_equals_input_score_for_every_document(self) -> None:
        """With the real scorer, every document's XTR score IS its input score.

        A single query token means the XTR average is sum([s])/1 == s, so the
        output score is bit-identical to the fused score that went in.
        """
        results = [
            _make_result("doc 0", score=0.30, rank=1),
            _make_result("doc 1", score=0.90, rank=2),
            _make_result("doc 2", score=0.50, rank=3),
        ]
        before = {r.chunk.chunk_text: r.score for r in results}

        out = rerank("a query that is never read by the xtr path", results, _xtr_cfg(), top_n=3)

        assert {r.chunk.chunk_text: r.score for r in out} == before

    def test_xtr_output_order_is_plain_descending_score_sort(self) -> None:
        """The XTR path reorders nothing the fused scores hadn't already ordered."""
        results = [
            _make_result("doc 0", score=0.30, rank=1),
            _make_result("doc 1", score=0.90, rank=2),
            _make_result("doc 2", score=0.50, rank=3),
        ]
        expected = [r.chunk.chunk_text for r in sorted(results, key=lambda r: -r.score)]

        out = rerank("query", results, _xtr_cfg(), top_n=3)

        assert [r.chunk.chunk_text for r in out] == expected

    def test_xtr_selection_warns_that_reranking_is_off(self, caplog) -> None:
        """Selecting xtr must log that it is a no-op, naming the ignored knob."""
        results = [_make_result(f"doc {i}", score=0.1 * i, rank=i + 1) for i in range(3)]

        with caplog.at_level(logging.WARNING, logger="trelix.retrieval.reranker"):
            rerank("query", results, _xtr_cfg(), top_n=3)

        warnings_text = " ".join(
            rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING
        )
        assert "TRELIX_RETRIEVAL_XTR_TOKENS" in warnings_text
        assert "single" in warnings_text.lower()
        assert "rerank_provider=plaid" in warnings_text


# ---------------------------------------------------------------------------
# Latent defects that only bite once xtr_score_documents is real
# ---------------------------------------------------------------------------


class TestXTRScoreKeying:
    def test_scores_are_keyed_by_position_not_value_equality(self) -> None:
        """Two value-equal results must get their own doc_id's score.

        SearchResult is a plain dataclass, so `results.index(r)` returns the
        FIRST value-equal element: every duplicate silently inherits the first
        one's XTR score and the true winner is buried.
        """
        results = [
            _make_result("dup", score=0.4, rank=1),
            _make_result("dup", score=0.4, rank=2),
            _make_result("unique", score=0.4, rank=3),
        ]
        # doc_id 1 -- the SECOND (value-equal) duplicate -- is the real winner.
        with (
            patch("trelix.retrieval.reranker_xtr.warn_experimental"),
            patch(
                "trelix.retrieval.reranker.xtr_score_documents",
                side_effect=_score_by_doc_id({0: 0.1, 1: 0.99, 2: 0.5}),
            ),
        ):
            out = rerank("query", results, _xtr_cfg(), top_n=3)

        assert [r.score for r in out] == [0.99, 0.5, 0.1]
        assert [r.chunk.chunk_text for r in out] == ["dup", "unique", "dup"]

    def test_output_carries_the_xtr_score_not_the_input_score(self) -> None:
        """A reranker that keeps the old score cannot be told apart from a no-op."""
        results = [
            _make_result("doc 0", score=0.30, rank=1),
            _make_result("doc 1", score=0.90, rank=2),
        ]
        with (
            patch("trelix.retrieval.reranker_xtr.warn_experimental"),
            patch(
                "trelix.retrieval.reranker.xtr_score_documents",
                side_effect=_score_by_doc_id({0: 0.75, 1: 0.25}),
            ),
        ):
            out = rerank("query", results, _xtr_cfg(), top_n=2)

        assert [(r.chunk.chunk_text, r.score, r.rank) for r in out] == [
            ("doc 0", 0.75, 1),
            ("doc 1", 0.25, 2),
        ]

    def test_xtr_does_not_mutate_the_caller_results(self) -> None:
        """Every other provider builds new SearchResults; xtr must not differ."""
        results = [
            _make_result("doc 0", score=0.30, rank=1),
            _make_result("doc 1", score=0.90, rank=2),
            _make_result("doc 2", score=0.50, rank=3),
        ]
        snapshot = [(r.chunk.chunk_text, r.score, r.rank) for r in results]

        rerank("query", results, _xtr_cfg(), top_n=3)

        assert [(r.chunk.chunk_text, r.score, r.rank) for r in results] == snapshot

    def test_ties_preserve_input_order(self) -> None:
        """Equal XTR scores must fall back to the incoming (fused) order."""
        results = [_make_result(f"doc {i}", score=0.5, rank=i + 1) for i in range(4)]

        with (
            patch("trelix.retrieval.reranker_xtr.warn_experimental"),
            patch(
                "trelix.retrieval.reranker.xtr_score_documents",
                side_effect=_score_by_doc_id({0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}),
            ),
        ):
            out = rerank("query", results, _xtr_cfg(), top_n=4)

        assert [r.chunk.chunk_text for r in out] == ["doc 0", "doc 1", "doc 2", "doc 3"]
