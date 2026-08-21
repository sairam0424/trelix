"""Tests for GraphBuilder — full graph construction pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from trelix.core.config import IndexConfig
from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.builder import GraphBuilder, GraphBuildResult
from trelix.store.db import Database


def _populated_repo(tmp_path: Path) -> Path:
    """Create a minimal indexed repo at tmp_path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".trelix").mkdir()
    db = Database(repo / ".trelix" / "index.db")

    fid = db.upsert_file(
        IndexedFile(
            path=str(repo / "auth.py"),
            rel_path="auth.py",
            language=Language.PYTHON,
            hash="x",
            size_bytes=100,
        )
    )
    sid1 = db.insert_symbol(
        Symbol(
            file_id=fid,
            name="login",
            qualified_name="login",
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=10,
            signature="def login()",
            body="def login(): pass",
        )
    )
    sid2 = db.insert_symbol(
        Symbol(
            file_id=fid,
            name="hash_password",
            qualified_name="hash_password",
            kind=SymbolKind.FUNCTION,
            line_start=12,
            line_end=20,
            signature="def hash_password()",
            body="def hash_password(): pass",
        )
    )
    db.insert_call_edges(
        [CallEdge(caller_id=sid1, callee_name="hash_password", callee_id=sid2, line=5)]
    )
    db._conn.commit()
    db.close()
    return repo


class TestGraphBuilder:
    def test_build_returns_result(self, tmp_path: Path) -> None:
        repo = _populated_repo(tmp_path)
        config = IndexConfig(repo_path=str(repo))
        builder = GraphBuilder(config)
        result = builder.build(extract_concepts=False)
        assert isinstance(result, GraphBuildResult)
        assert result.node_count >= 2
        assert result.edge_count >= 1
        assert result.community_count >= 1
        assert result.concept_count == 0  # no concept extraction

    def test_build_with_concepts_disabled_does_not_call_llm(self, tmp_path: Path) -> None:
        repo = _populated_repo(tmp_path)
        config = IndexConfig(repo_path=str(repo))
        builder = GraphBuilder(config)
        with patch("trelix.graph.builder.ConceptExtractor") as MockCE:
            result = builder.build(extract_concepts=False)
        MockCE.assert_not_called()
        assert result.concept_count == 0

    def test_build_assigns_communities(self, tmp_path: Path) -> None:
        repo = _populated_repo(tmp_path)
        config = IndexConfig(repo_path=str(repo))
        builder = GraphBuilder(config)
        result = builder.build(extract_concepts=False)
        # All nodes should have community set
        for _, attrs in result.code_graph.nx.nodes(data=True):
            assert attrs.get("community") is not None

    def test_elapsed_seconds_positive(self, tmp_path: Path) -> None:
        repo = _populated_repo(tmp_path)
        config = IndexConfig(repo_path=str(repo))
        result = GraphBuilder(config).build(extract_concepts=False)
        assert result.elapsed_seconds > 0


def _repo_with_many_symbols(tmp_path: Path, count: int, hub_callers: int = 60) -> Path:
    """An indexed repo with *count* symbols whose HUBS ARE THE LAST ones inserted.

    The hub placement is load-bearing and was got wrong once. PageRank rewards in-edges,
    so making `fn_0` the hub put the most central symbol at the LOWEST id — and since the
    DB returns rows in roughly id order, "top 200 by centrality" and "first 200 in DB
    order" then agreed on all 200 entries. Every ordering test below passed with the fix
    reverted, i.e. they were verifying nothing.

    Placing the hubs at the HIGHEST ids inverts the two orders: a cap of 200 over 250
    symbols excludes the hubs entirely under DB order and puts them first under
    centrality order, so the tests can only pass on the intended implementation.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".trelix").mkdir()
    db = Database(repo / ".trelix" / "index.db")

    fid = db.upsert_file(
        IndexedFile(
            path=str(repo / "big.py"),
            rel_path="big.py",
            language=Language.PYTHON,
            hash="x",
            size_bytes=100,
        )
    )
    ids: list[int] = []
    for i in range(count):
        ids.append(
            db.insert_symbol(
                Symbol(
                    file_id=fid,
                    name=f"fn_{i}",
                    qualified_name=f"fn_{i}",
                    kind=SymbolKind.FUNCTION,
                    line_start=i * 10 + 1,
                    line_end=i * 10 + 5,
                    signature=f"def fn_{i}()",
                    body=f"def fn_{i}(): pass",
                )
            )
        )
    # The final `_HUB_COUNT` symbols are the hubs, each with a different in-degree so the
    # centrality spread is a real ordering rather than a tie.
    hubs = ids[-_HUB_COUNT:]
    callers = ids[: max(hub_callers, _HUB_COUNT * 2)]
    edges: list[CallEdge] = []
    for rank, hub_id in enumerate(hubs):
        # Earlier hubs in this list get FEWER callers, so the last symbol is the most
        # central of all — the furthest possible position from where DB order would put it.
        in_degree = max(2, (rank + 1) * (len(callers) // (_HUB_COUNT + 1)))
        for caller_id in callers[:in_degree]:
            if caller_id == hub_id:
                continue
            edges.append(
                CallEdge(
                    caller_id=caller_id,
                    callee_name=f"fn_{ids.index(hub_id)}",
                    callee_id=hub_id,
                    line=2,
                )
            )
    db.insert_call_edges(edges)
    db._conn.commit()
    db.close()
    return repo


_HUB_COUNT = 5


def _centrality_by_symbol_id(repo: Path) -> dict[int, float]:
    db = Database(repo / ".trelix" / "index.db")
    try:
        rows = db._conn.execute("SELECT symbol_id, centrality FROM graph_metadata").fetchall()
        return {int(r[0]): float(r[1] or 0.0) for r in rows}
    finally:
        db.close()


class TestConceptExtractionCoverage:
    """`concept_count` described a capped, arbitrarily-selected sample and said so nowhere.

    Extraction stops at 200 symbols. It used to slice whatever order
    `iter_all_symbols_with_files()` returned, and that query has no ORDER BY. Measured on
    this repo's own 12,184-symbol index: the query plans as a plain `SCAN s`, so the
    "first 200" were the lowest symbol ids — 2..226, from 10 files, all of them `.github/`
    and `.devcontainer/` metadata. Every paid LLM call described an issue template; nothing
    in `src/` was ever reached. Lowest-id means earliest-indexed, which has no relationship
    to importance, and that is what ranking by centrality fixes.
    """

    CAP = 200

    def _captured_batches(self, repo: Path) -> tuple[list[Any], GraphBuildResult]:
        seen: list[Any] = []
        config = IndexConfig(repo_path=str(repo))
        with patch("trelix.graph.builder.ConceptExtractor") as MockCE:
            MockCE.return_value.extract_from_symbols.side_effect = lambda batch: (
                seen.extend(batch) or []
            )
            result = GraphBuilder(config).build(extract_concepts=True)
        return seen, result

    def test_coverage_is_reported_when_the_cap_truncates(self, tmp_path: Path) -> None:
        repo = _repo_with_many_symbols(tmp_path, count=250)

        seen, result = self._captured_batches(repo)

        assert result.concept_symbols_total == 250
        assert result.concept_symbols_considered == self.CAP
        assert len(seen) == self.CAP, f"extractor saw {len(seen)} symbols, expected {self.CAP}"

    def test_the_truncation_warns_at_WARNING_not_INFO(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The CLI's default level is WARNING; an INFO line here is invisible without -v.

        That is precisely how this truncation went unnoticed, and the same reasoning is
        already recorded for the degenerate-partition warning in this builder.
        """
        repo = _repo_with_many_symbols(tmp_path, count=250)

        with caplog.at_level(logging.WARNING, logger="trelix.graph.builder"):
            self._captured_batches(repo)

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("200 of 250" in m for m in warnings), warnings

    def test_no_warning_and_full_coverage_when_under_the_cap(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A repo smaller than the cap is fully covered, and must not be told otherwise."""
        repo = _repo_with_many_symbols(tmp_path, count=30, hub_callers=10)

        with caplog.at_level(logging.WARNING, logger="trelix.graph.builder"):
            seen, result = self._captured_batches(repo)

        assert result.concept_symbols_total == 30
        assert result.concept_symbols_considered == 30
        assert len(seen) == 30
        assert not [r for r in caplog.records if "of 30 symbols" in r.getMessage()]

    def test_the_selected_symbols_are_the_TOP_ones_by_centrality(self, tmp_path: Path) -> None:
        """The claim that makes the cap defensible instead of arbitrary."""
        repo = _repo_with_many_symbols(tmp_path, count=250)

        seen, _ = self._captured_batches(repo)
        centrality = _centrality_by_symbol_id(repo)
        chosen = [s.id for s in seen]
        expected = [
            sid for sid in sorted(centrality, key=lambda i: (-centrality[i], i))[: self.CAP]
        ]

        assert chosen == expected, (
            "concept extraction did not take the top symbols by centrality — the first "
            f"5 differ: got {chosen[:5]}, expected {expected[:5]}"
        )

    def test_the_hubs_are_covered_even_though_DB_ORDER_would_cut_them(self, tmp_path: Path) -> None:
        """The behavioural statement of the claim, independent of the sort expression.

        The five hubs are the last symbols inserted, so a cap of 200 over 250 in DB order
        excludes every one of them — they are exactly the symbols an operator most wants
        concepts for, and the old slice dropped them for no reason anybody chose.
        """
        repo = _repo_with_many_symbols(tmp_path, count=250)

        seen, _ = self._captured_batches(repo)
        centrality = _centrality_by_symbol_id(repo)
        top_hubs = sorted(centrality, key=lambda i: (-centrality[i], i))[:_HUB_COUNT]
        chosen_ids = {s.id for s in seen}

        # Precondition: the hubs must be beyond the cap in DB order, or this proves nothing.
        assert max(top_hubs) > self.CAP, (
            "fixture no longer discriminates — the hubs fall inside the first "
            f"{self.CAP} ids, so DB order would cover them too: {sorted(top_hubs)}"
        )
        missing = [h for h in top_hubs if h not in chosen_ids]
        assert not missing, f"the most central symbols were excluded from extraction: {missing}"
        assert seen[0].id == top_hubs[0], (
            f"the most central symbol was not ranked first: {seen[0].id} != {top_hubs[0]}"
        )

    def test_the_selection_is_stable_across_two_builds(self, tmp_path: Path) -> None:
        """Same index, same 200 — the reproducibility the old slice could not promise.

        The `(-centrality, id)` tiebreak is what makes this hold: equal-centrality symbols
        would otherwise be free to reshuffle between runs.
        """
        repo = _repo_with_many_symbols(tmp_path, count=250)

        first, _ = self._captured_batches(repo)
        second, _ = self._captured_batches(repo)

        assert [s.id for s in first] == [s.id for s in second]

    def test_coverage_stays_zero_when_extraction_is_off(self, tmp_path: Path) -> None:
        """0/0 is the honest answer for a run that extracted nothing."""
        repo = _repo_with_many_symbols(tmp_path, count=30, hub_callers=10)
        config = IndexConfig(repo_path=str(repo))

        result = GraphBuilder(config).build(extract_concepts=False)

        assert result.concept_symbols_considered == 0
        assert result.concept_symbols_total == 0

    def test_ties_break_on_symbol_id_regardless_of_the_order_the_DB_returns(
        self, tmp_path: Path
    ) -> None:
        """Why the `(-centrality, id)` tiebreak is load-bearing and not decoration.

        Most symbols in any repo sit at the same baseline PageRank score, so the cap's
        boundary is decided almost entirely by how ties are broken. `sorted` is stable,
        which means WITHOUT the id term the result simply inherits whatever order the DB
        produced — and that order has no ORDER BY behind it. Two builds in one process
        cannot show this, because they see the same DB order both times; feeding a
        reversed order is what makes the difference observable.

        With the tiebreak, tied symbols come out id-ascending whichever way they arrive.
        """
        from trelix.store.db import Database as _Db

        repo = _repo_with_many_symbols(tmp_path, count=250)
        real = _Db.iter_all_symbols_with_files

        def _reversed(self: Any) -> list[Any]:
            return list(reversed(real(self)))

        seen: list[Any] = []
        config = IndexConfig(repo_path=str(repo))
        with (
            patch.object(_Db, "iter_all_symbols_with_files", _reversed),
            patch("trelix.graph.builder.ConceptExtractor") as MockCE,
        ):
            MockCE.return_value.extract_from_symbols.side_effect = lambda batch: (
                seen.extend(batch) or []
            )
            GraphBuilder(config).build(extract_concepts=True)

        centrality = _centrality_by_symbol_id(repo)
        # Group the chosen symbols by their exact centrality and check each tied run is
        # id-ascending. Under a stable sort with no id term, a reversed input yields
        # id-DESCENDING runs, so this fails.
        runs: dict[float, list[int]] = {}
        for symbol in seen:
            runs.setdefault(centrality.get(symbol.id, 0.0), []).append(symbol.id)
        tied = {score: ids for score, ids in runs.items() if len(ids) > 1}

        assert tied, "fixture produced no ties, so this test would prove nothing"
        for score, ids in tied.items():
            assert ids == sorted(ids), (
                f"symbols tied at centrality {score} came back in DB order rather than "
                f"id order, so the cap boundary moves with the query plan: {ids[:8]}"
            )
