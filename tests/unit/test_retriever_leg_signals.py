"""A retrieval leg that loses results must SAY so.

Two separate silences in `retrieval/retriever.py`, both of which make an enabled
feature indistinguishable from a working one:

1. `_vector_search` fetches exactly `k` from the ANN index (the oversample only
   happens under `path_filter`) and `continue`s past every `_hydrate_chunk` that
   returns None. A None means the ANN index holds a `chunk_id` with no row left in
   the DB — a stale index — and each one silently costs a result slot, so the leg
   hands back fewer than `k` with no diagnostic anywhere. Backfilling the slot
   would make the leg quieter; naming the shortfall is what makes the staleness
   fixable (`trelix index <repo>`).

2. `_apply_pagerank_boost`'s empty-centrality WARNING carries a comment promising
   it is said "once". It was said on every single query, which turns the one
   actionable line into log noise operators filter out.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import IndexConfig, RetrievalConfig
from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    SearchResult,
    Symbol,
    SymbolKind,
)

# ---------------------------------------------------------------------------
# Helpers (mirrors test_retriever_core.py's fixtures — kept local so this file
# stands alone)
# ---------------------------------------------------------------------------


def _make_result(idx: int, rel_path: str = "src/foo.py") -> SearchResult:
    file = IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash=f"sha-{idx}",
        size_bytes=100,
        id=idx,
        indexed_at=datetime(2024, 1, 1),
    )
    symbol = Symbol(
        file_id=idx,
        name=f"func_{idx}",
        qualified_name=f"module.func_{idx}",
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=10,
        signature=f"def func_{idx}()",
        body=f"def func_{idx}():\n    pass",
        id=idx,
    )
    chunk = Chunk(symbol_id=idx, chunk_text=f"def func_{idx}(): pass", token_count=4, id=idx)
    return SearchResult(chunk=chunk, symbol=symbol, file=file, score=0.9, rank=idx, source="vector")


def _build_retriever(tmp_path: Path):
    """Retriever with DB / embedder / vector store / planner replaced by mocks."""
    from trelix.retrieval.retriever import Retriever

    with (
        patch("trelix.retrieval.retriever.Database") as mock_db_cls,
        patch("trelix.retrieval.retriever.make_embedder") as mock_emb_cls,
        patch("trelix.retrieval.retriever.make_vector_store") as mock_vs_cls,
        patch("trelix.retrieval.retriever.QueryPlanner") as mock_planner_cls,
        patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder"}),
    ):
        mock_emb = MagicMock()
        mock_emb.dimension = 1536
        mock_emb_cls.return_value = mock_emb
        mock_db_cls.return_value = MagicMock()
        mock_vs_cls.return_value = MagicMock()
        mock_planner_cls.return_value = MagicMock()
        return Retriever(
            IndexConfig(
                repo_path=str(tmp_path),
                retrieval=RetrievalConfig(query_cache_size=0),
            )
        )


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# (a) _vector_search shortfall
# ---------------------------------------------------------------------------


class TestVectorLegShortfallIsReported:
    """Hydration misses shrink the vector leg below k; that must be audible."""

    def test_hydration_misses_are_warned_with_counts(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """5 ANN hits, 2 dead chunk_ids → 3 results AND one WARNING naming 3/5."""
        retriever = _build_retriever(tmp_path)
        retriever.vector_store.search.return_value = [(i, 0.1) for i in range(1, 6)]
        rows: list[tuple | None] = []
        for i in range(1, 6):
            if i in (2, 4):  # chunk rows the ANN index still points at, but the DB lost
                rows.append(None)
            else:
                sr = _make_result(i)
                rows.append((sr.chunk, sr.symbol, sr.file))
        retriever.db.get_chunk_with_context.side_effect = rows

        with caplog.at_level(logging.DEBUG, logger="trelix.retrieval"):
            results = retriever._vector_search([0.0] * 1536, k=5)

        assert len(results) == 3, "precondition: the shortfall is real"
        warns = _warnings(caplog)
        assert len(warns) == 1, f"expected exactly one WARNING, got {[w.message for w in warns]}"
        msg = warns[0].getMessage()
        assert "2" in msg and "3" in msg and "5" in msg, msg

    def test_no_warning_when_every_hit_hydrates(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A healthy leg stays silent — the warning must mean something."""
        retriever = _build_retriever(tmp_path)
        retriever.vector_store.search.return_value = [(i, 0.1) for i in range(1, 4)]
        retriever.db.get_chunk_with_context.side_effect = [
            (r.chunk, r.symbol, r.file) for r in (_make_result(i) for i in range(1, 4))
        ]

        with caplog.at_level(logging.DEBUG, logger="trelix.retrieval"):
            results = retriever._vector_search([0.0] * 1536, k=3)

        assert len(results) == 3
        assert _warnings(caplog) == []

    def test_path_filter_drops_alone_do_not_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Prefix rejects are by design (that is what the oversample is for)."""
        retriever = _build_retriever(tmp_path)
        retriever.vector_store.search.return_value = [(1, 0.1), (2, 0.1)]
        inside = _make_result(1, rel_path="src/auth/login.py")
        outside = _make_result(2, rel_path="src/billing/invoice.py")
        retriever.db.get_chunk_with_context.side_effect = [
            (inside.chunk, inside.symbol, inside.file),
            (outside.chunk, outside.symbol, outside.file),
        ]

        with caplog.at_level(logging.DEBUG, logger="trelix.retrieval"):
            results = retriever._vector_search([0.0] * 1536, k=5, path_filter="src/auth")

        assert len(results) == 1
        assert _warnings(caplog) == []

    def test_ann_returning_fewer_than_k_does_not_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A small index legitimately has fewer than k vectors — not a defect."""
        retriever = _build_retriever(tmp_path)
        retriever.vector_store.search.return_value = [(1, 0.1)]
        sr = _make_result(1)
        retriever.db.get_chunk_with_context.return_value = (sr.chunk, sr.symbol, sr.file)

        with caplog.at_level(logging.DEBUG, logger="trelix.retrieval"):
            results = retriever._vector_search([0.0] * 1536, k=20)

        assert len(results) == 1
        assert _warnings(caplog) == []


# ---------------------------------------------------------------------------
# (b) PageRank empty-centrality warning fires once
# ---------------------------------------------------------------------------


class TestPageRankEmptyCentralityWarnsOnce:
    """The comment says "once at WARNING"; per-query repetition is log noise."""

    def _retriever_with_boost_on(self, tmp_path: Path):
        retriever = _build_retriever(tmp_path)
        retriever.config.retrieval.pagerank_boost_enabled = True
        return retriever

    def test_two_queries_produce_one_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        retriever = self._retriever_with_boost_on(tmp_path)
        results = [_make_result(1)]

        with (
            patch("trelix.graph.persistence.get_top_central_symbols", return_value=[]),
            caplog.at_level(logging.DEBUG, logger="trelix.retrieval"),
        ):
            retriever._apply_pagerank_boost(results)
            retriever._apply_pagerank_boost(results)

        warns = _warnings(caplog)
        assert len(warns) == 1, f"expected 1 warning across 2 queries, got {len(warns)}"
        assert "graph_metadata is empty" in warns[0].getMessage()

    def test_each_retriever_warns_for_itself(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Suppression is per-instance: a fresh Retriever re-reports the state."""
        first = self._retriever_with_boost_on(tmp_path)
        second = self._retriever_with_boost_on(tmp_path)
        results = [_make_result(1)]

        with (
            patch("trelix.graph.persistence.get_top_central_symbols", return_value=[]),
            caplog.at_level(logging.DEBUG, logger="trelix.retrieval"),
        ):
            first._apply_pagerank_boost(results)
            second._apply_pagerank_boost(results)

        assert len(_warnings(caplog)) == 2

    def test_populated_centrality_still_boosts_and_stays_quiet(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The once-flag must not touch the working path."""
        retriever = self._retriever_with_boost_on(tmp_path)
        results = [_make_result(1)]

        with (
            patch("trelix.graph.persistence.get_top_central_symbols", return_value=[1]),
            caplog.at_level(logging.DEBUG, logger="trelix.retrieval"),
        ):
            boosted = retriever._apply_pagerank_boost(results)

        assert boosted[0].score == pytest.approx(
            0.9 * retriever.config.retrieval.pagerank_boost_factor
        )
        assert _warnings(caplog) == []
