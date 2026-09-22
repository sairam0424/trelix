"""
Unit tests for trelix.retrieval.graph (call/import/type graph expansion + PageRank).

All tests use a real in-memory SQLite Database (tmp_path) seeded with minimal
fixtures — no external services required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trelix.analysis.defuse import DefUseEdge
from trelix.core.models import (
    CallEdge,
    Chunk,
    GenericEdge,
    ImportEdge,
    IndexedFile,
    Language,
    SearchResult,
    Symbol,
    SymbolKind,
    TypeEdge,
)
from trelix.retrieval.graph import (
    expand_with_call_graph,
    expand_with_dataflow,
    expand_with_imports,
    expand_with_type_edges,
    rank_by_pagerank,
    seed_from_import_paths,
)
from trelix.store.db import Database

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    """Fresh SQLite Database for each test."""
    return Database(tmp_path / "index.db")


def _insert_file(db: Database, rel_path: str = "mod.py") -> int:
    f = IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash="abc",
        size_bytes=100,
    )
    return db.upsert_file(f)


def _insert_symbol(
    db: Database,
    file_id: int,
    name: str,
    kind: SymbolKind = SymbolKind.FUNCTION,
) -> int:
    sym = Symbol(
        file_id=file_id,
        name=name,
        qualified_name=name,
        kind=kind,
        line_start=1,
        line_end=5,
        signature=f"def {name}()",
        body=f"def {name}(): pass",
    )
    sym_id = db.insert_symbol(sym)
    db._conn.commit()
    return sym_id


def _insert_chunk(db: Database, symbol_id: int) -> int:
    chunk = Chunk(symbol_id=symbol_id, chunk_text=f"body_{symbol_id}", token_count=5)
    chunk_id = db.insert_chunk(chunk)
    db._conn.commit()
    return chunk_id


def _make_search_result(db: Database, symbol_id: int, score: float = 0.9) -> SearchResult:
    """Build a SearchResult from a symbol already in the DB."""
    sym_file = db.get_symbol_with_file(symbol_id)
    assert sym_file is not None
    symbol, file = sym_file
    chunk = db.get_first_chunk_for_symbol(symbol_id)
    assert chunk is not None
    return SearchResult(
        chunk=chunk,
        symbol=symbol,
        file=file,
        score=score,
        rank=1,
        source="vector",
    )


# ---------------------------------------------------------------------------
# expand_with_call_graph
# ---------------------------------------------------------------------------


class TestExpandWithCallGraph:
    def test_empty_results_returns_empty(self, db: Database) -> None:
        extra = expand_with_call_graph(db, results=[])
        assert extra == []

    def test_returns_callee_symbols(self, db: Database) -> None:
        """
        caller → callee: expand_with_call_graph on [caller] should return callee.
        """
        fid = _insert_file(db)
        caller_id = _insert_symbol(db, fid, "caller")
        callee_id = _insert_symbol(db, fid, "callee")
        _insert_chunk(db, caller_id)
        _insert_chunk(db, callee_id)

        db.insert_call_edges(
            [CallEdge(caller_id=caller_id, callee_name="callee", line=2, callee_id=callee_id)]
        )
        db._conn.commit()

        result = _make_search_result(db, caller_id)
        extra = expand_with_call_graph(db, [result])

        ids = [r.chunk.symbol_id for r in extra]
        assert callee_id in ids

    def test_returns_caller_symbols(self, db: Database) -> None:
        """
        caller → callee: expand on [callee] should return caller (reverse edge).
        """
        fid = _insert_file(db)
        caller_id = _insert_symbol(db, fid, "callerA")
        callee_id = _insert_symbol(db, fid, "calleeA")
        _insert_chunk(db, caller_id)
        _insert_chunk(db, callee_id)

        db.insert_call_edges(
            [CallEdge(caller_id=caller_id, callee_name="calleeA", line=3, callee_id=callee_id)]
        )
        db._conn.commit()

        result = _make_search_result(db, callee_id)
        extra = expand_with_call_graph(db, [result])

        ids = [r.chunk.symbol_id for r in extra]
        assert caller_id in ids

    def test_source_is_graph_expansion(self, db: Database) -> None:
        fid = _insert_file(db)
        caller_id = _insert_symbol(db, fid, "fn_x")
        callee_id = _insert_symbol(db, fid, "fn_y")
        _insert_chunk(db, caller_id)
        _insert_chunk(db, callee_id)

        db.insert_call_edges(
            [CallEdge(caller_id=caller_id, callee_name="fn_y", line=1, callee_id=callee_id)]
        )
        db._conn.commit()

        result = _make_search_result(db, caller_id)
        extra = expand_with_call_graph(db, [result])

        assert all(r.source == "graph_expansion" for r in extra)

    def test_no_duplicate_symbols_in_expansion(self, db: Database) -> None:
        """Symbols already in the seed results must not appear in extra."""
        fid = _insert_file(db)
        a_id = _insert_symbol(db, fid, "fn_a")
        b_id = _insert_symbol(db, fid, "fn_b")
        _insert_chunk(db, a_id)
        _insert_chunk(db, b_id)

        # a calls b AND b calls a — no infinite loop and no duplicates
        db.insert_call_edges(
            [
                CallEdge(caller_id=a_id, callee_name="fn_b", line=1, callee_id=b_id),
                CallEdge(caller_id=b_id, callee_name="fn_a", line=2, callee_id=a_id),
            ]
        )
        db._conn.commit()

        result_a = _make_search_result(db, a_id)
        extra = expand_with_call_graph(db, [result_a])

        extra_ids = [r.chunk.symbol_id for r in extra]
        # a is already in seed; only b should appear in extra
        assert a_id not in extra_ids
        assert b_id in extra_ids

    def test_max_extra_is_respected(self, db: Database) -> None:
        fid = _insert_file(db)
        root_id = _insert_symbol(db, fid, "root")
        _insert_chunk(db, root_id)

        # Connect 10 callees
        callee_ids = []
        for i in range(10):
            cid = _insert_symbol(db, fid, f"callee_{i}")
            _insert_chunk(db, cid)
            callee_ids.append(cid)
            db.insert_call_edges(
                [CallEdge(caller_id=root_id, callee_name=f"callee_{i}", line=i, callee_id=cid)]
            )
        db._conn.commit()

        result = _make_search_result(db, root_id)
        extra = expand_with_call_graph(db, [result], max_extra=3)
        assert len(extra) <= 3

    def test_expansion_scores_are_discounted(self, db: Database) -> None:
        """Graph-expanded items must have lower scores than the seed result."""
        fid = _insert_file(db)
        caller_id = _insert_symbol(db, fid, "big_fn")
        callee_id = _insert_symbol(db, fid, "small_fn")
        _insert_chunk(db, caller_id)
        _insert_chunk(db, callee_id)

        db.insert_call_edges(
            [CallEdge(caller_id=caller_id, callee_name="small_fn", line=1, callee_id=callee_id)]
        )
        db._conn.commit()

        result = _make_search_result(db, caller_id, score=1.0)
        extra = expand_with_call_graph(db, [result])

        for r in extra:
            assert r.score < 1.0, "Expanded score should be discounted below seed score"

    def test_no_neighbors_returns_empty(self, db: Database) -> None:
        """Symbol with no call edges: expansion returns empty list."""
        fid = _insert_file(db)
        lone_id = _insert_symbol(db, fid, "lone_wolf")
        _insert_chunk(db, lone_id)

        result = _make_search_result(db, lone_id)
        extra = expand_with_call_graph(db, [result])
        assert extra == []

    # -----------------------------------------------------------------
    # graph_context — topology metadata riding along on expanded results
    # -----------------------------------------------------------------

    def test_graph_context_direction_differs_for_caller_vs_callee(self, db: Database) -> None:
        """
        hub calls callee_of_hub (get_callees(hub)); caller_of_hub calls hub
        (get_callers(hub)). Expanding on [hub] must tag each neighbor with a
        graph_context that reflects its OWN direction relative to hub, not a
        shared/undirected label — proving direction survives the BFS restructure.
        """
        fid = _insert_file(db)
        hub_id = _insert_symbol(db, fid, "hub")
        callee_id = _insert_symbol(db, fid, "callee_of_hub")
        caller_id = _insert_symbol(db, fid, "caller_of_hub")
        _insert_chunk(db, hub_id)
        _insert_chunk(db, callee_id)
        _insert_chunk(db, caller_id)

        db.insert_call_edges(
            [
                CallEdge(
                    caller_id=hub_id, callee_name="callee_of_hub", line=1, callee_id=callee_id
                ),
                CallEdge(caller_id=caller_id, callee_name="hub", line=2, callee_id=hub_id),
            ]
        )
        db._conn.commit()

        result = _make_search_result(db, hub_id)
        extra = expand_with_call_graph(db, [result])
        by_id = {r.chunk.symbol_id: r for r in extra}

        assert by_id[callee_id].graph_context is not None
        assert by_id[caller_id].graph_context is not None
        assert by_id[callee_id].graph_context != by_id[caller_id].graph_context
        # callee_of_hub is CALLED BY hub
        assert "called by" in by_id[callee_id].graph_context
        assert "hub" in by_id[callee_id].graph_context
        # caller_of_hub CALLS hub
        assert "calls" in by_id[caller_id].graph_context
        assert "hub" in by_id[caller_id].graph_context

    def test_graph_context_names_actual_via_parent_at_two_hops(self, db: Database) -> None:
        """
        seed -> mid -> leaf (leaf is 2 hops from seed, reached VIA mid).
        leaf's graph_context must name "mid" (the actual parent it was
        discovered through), not "seed", and must say 2 hops.
        """
        fid = _insert_file(db)
        seed_id = _insert_symbol(db, fid, "seed_fn")
        mid_id = _insert_symbol(db, fid, "mid_fn")
        leaf_id = _insert_symbol(db, fid, "leaf_fn")
        _insert_chunk(db, seed_id)
        _insert_chunk(db, mid_id)
        _insert_chunk(db, leaf_id)

        db.insert_call_edges(
            [
                CallEdge(caller_id=seed_id, callee_name="mid_fn", line=1, callee_id=mid_id),
                CallEdge(caller_id=mid_id, callee_name="leaf_fn", line=2, callee_id=leaf_id),
            ]
        )
        db._conn.commit()

        result = _make_search_result(db, seed_id)
        extra = expand_with_call_graph(db, [result], depth=2)
        by_id = {r.chunk.symbol_id: r for r in extra}

        assert leaf_id in by_id
        leaf_ctx = by_id[leaf_id].graph_context
        assert leaf_ctx is not None
        assert "mid_fn" in leaf_ctx
        assert "seed_fn" not in leaf_ctx
        assert "2 hops" in leaf_ctx

    def test_graph_context_is_none_when_no_networkx_reordering_needed(self, db: Database) -> None:
        """Sanity: a plain 1-hop callee still gets a non-empty graph_context —
        the field isn't only populated on some code paths (e.g. only when the
        PageRank re-sort branch runs)."""
        fid = _insert_file(db)
        caller_id = _insert_symbol(db, fid, "solo_caller")
        callee_id = _insert_symbol(db, fid, "solo_callee")
        _insert_chunk(db, caller_id)
        _insert_chunk(db, callee_id)

        db.insert_call_edges(
            [CallEdge(caller_id=caller_id, callee_name="solo_callee", line=1, callee_id=callee_id)]
        )
        db._conn.commit()

        result = _make_search_result(db, caller_id)
        extra = expand_with_call_graph(db, [result])

        assert len(extra) == 1
        assert extra[0].graph_context == "called by solo_caller (1 hop)"

    def test_regression_discovered_ids_and_hop_discount_unchanged(self, db: Database) -> None:
        """No regression in WHAT gets discovered: same symbol_ids at the same
        hop distances (verified indirectly through the hop-based score
        discount, base_score * 0.5**hop) as the pre-topology implementation."""
        fid = _insert_file(db)
        seed_id = _insert_symbol(db, fid, "seed2")
        mid_id = _insert_symbol(db, fid, "mid2")
        leaf_id = _insert_symbol(db, fid, "leaf2")
        _insert_chunk(db, seed_id)
        _insert_chunk(db, mid_id)
        _insert_chunk(db, leaf_id)

        db.insert_call_edges(
            [
                CallEdge(caller_id=seed_id, callee_name="mid2", line=1, callee_id=mid_id),
                CallEdge(caller_id=mid_id, callee_name="leaf2", line=2, callee_id=leaf_id),
            ]
        )
        db._conn.commit()

        result = _make_search_result(db, seed_id, score=1.0)
        extra = expand_with_call_graph(db, [result], depth=2)
        by_id = {r.chunk.symbol_id: r for r in extra}

        assert set(by_id) == {mid_id, leaf_id}
        assert by_id[mid_id].score == pytest.approx(1.0 * 0.5)
        assert by_id[leaf_id].score == pytest.approx(1.0 * 0.25)


# ---------------------------------------------------------------------------
# expand_with_dataflow
# ---------------------------------------------------------------------------


class TestExpandWithDataflow:
    """expand_with_dataflow correlates a seed symbol's def-use spans (from
    DataFlowExtractor / def_use_edges) against its resolved call sites
    (calls.line) — the new signal that distinguishes "every callee" (plain
    call-graph expansion) from "callees a specific tracked variable is live
    across". See docs/reports/post-v3.3.0-research-sweep-2026-09-16.md §6.2.
    """

    def test_empty_results_returns_empty(self, db: Database) -> None:
        extra = expand_with_dataflow(db, results=[])
        assert extra == []

    def test_callee_call_site_inside_def_use_span_is_included(self, db: Database) -> None:
        """
        caller defines `x` on line 2 and reads it on line 6. A call to
        `callee` at line 4 (inside [2, 6]) is data-flow-correlated and must
        be surfaced.
        """
        fid = _insert_file(db)
        caller_id = _insert_symbol(db, fid, "caller")
        callee_id = _insert_symbol(db, fid, "callee")
        _insert_chunk(db, caller_id)
        _insert_chunk(db, callee_id)

        db.insert_call_edges(
            [CallEdge(caller_id=caller_id, callee_name="callee", line=4, callee_id=callee_id)]
        )
        db.insert_def_use_edges(
            [DefUseEdge(symbol_id=caller_id, var_name="x", def_line=2, use_line=6, edge_type="use")]
        )
        db._conn.commit()

        result = _make_search_result(db, caller_id)
        extra = expand_with_dataflow(db, [result])

        ids = [r.chunk.symbol_id for r in extra]
        assert callee_id in ids

    def test_callee_call_site_outside_every_def_use_span_is_excluded(self, db: Database) -> None:
        """
        Core correctness property: a resolved callee whose call site line
        falls OUTSIDE every def-use span must NOT be surfaced, even though
        expand_with_call_graph would return it unconditionally.
        """
        fid = _insert_file(db)
        caller_id = _insert_symbol(db, fid, "caller2")
        callee_id = _insert_symbol(db, fid, "callee2")
        _insert_chunk(db, caller_id)
        _insert_chunk(db, callee_id)

        # Call site at line 20 — nowhere near the def-use span [2, 6].
        db.insert_call_edges(
            [CallEdge(caller_id=caller_id, callee_name="callee2", line=20, callee_id=callee_id)]
        )
        db.insert_def_use_edges(
            [DefUseEdge(symbol_id=caller_id, var_name="x", def_line=2, use_line=6, edge_type="use")]
        )
        db._conn.commit()

        result = _make_search_result(db, caller_id)
        extra = expand_with_dataflow(db, [result])

        ids = [r.chunk.symbol_id for r in extra]
        assert callee_id not in ids
        assert extra == []

    def test_no_def_use_data_returns_empty_gracefully(self, db: Database) -> None:
        """
        No def_use_edges rows for the symbol (dataflow_enabled was off at
        index time, or the symbol genuinely has no local variables) — must
        degrade to [] without raising, even when callees exist.
        """
        fid = _insert_file(db)
        caller_id = _insert_symbol(db, fid, "caller3")
        callee_id = _insert_symbol(db, fid, "callee3")
        _insert_chunk(db, caller_id)
        _insert_chunk(db, callee_id)

        db.insert_call_edges(
            [CallEdge(caller_id=caller_id, callee_name="callee3", line=3, callee_id=callee_id)]
        )
        db._conn.commit()
        # No insert_def_use_edges call at all.

        result = _make_search_result(db, caller_id)
        extra = expand_with_dataflow(db, [result])
        assert extra == []

    def test_max_extra_is_respected(self, db: Database) -> None:
        fid = _insert_file(db)
        root_id = _insert_symbol(db, fid, "root_df")
        _insert_chunk(db, root_id)

        for i in range(10):
            cid = _insert_symbol(db, fid, f"callee_df_{i}")
            _insert_chunk(db, cid)
            db.insert_call_edges(
                [
                    CallEdge(
                        caller_id=root_id, callee_name=f"callee_df_{i}", line=i + 1, callee_id=cid
                    )
                ]
            )
        db.insert_def_use_edges(
            [
                DefUseEdge(
                    symbol_id=root_id, var_name="shared", def_line=0, use_line=20, edge_type="use"
                )
            ]
        )
        db._conn.commit()

        result = _make_search_result(db, root_id)
        extra = expand_with_dataflow(db, [result], max_extra=3)
        assert len(extra) <= 3

    def test_already_seen_symbol_ids_are_not_duplicated(self, db: Database) -> None:
        """A callee that is ALSO one of the seed results must not be duplicated
        into `extra`, mirroring expand_with_call_graph's seen_ids pattern."""
        fid = _insert_file(db)
        caller_id = _insert_symbol(db, fid, "caller4")
        callee_id = _insert_symbol(db, fid, "callee4")
        _insert_chunk(db, caller_id)
        _insert_chunk(db, callee_id)

        db.insert_call_edges(
            [CallEdge(caller_id=caller_id, callee_name="callee4", line=3, callee_id=callee_id)]
        )
        db.insert_def_use_edges(
            [DefUseEdge(symbol_id=caller_id, var_name="y", def_line=1, use_line=5, edge_type="use")]
        )
        db._conn.commit()

        caller_result = _make_search_result(db, caller_id)
        callee_result = _make_search_result(db, callee_id)
        extra = expand_with_dataflow(db, [caller_result, callee_result])

        assert extra == []

    def test_max_extra_does_not_starve_later_seeds(self, db: Database) -> None:
        """A single early, higher-ranked seed with more correlated callees than
        max_extra must not exhaust the whole budget before later seeds are even
        examined -- every seed's candidates need to be collected before any
        truncation happens, mirroring expand_with_call_graph's collect-then-
        truncate shape."""
        fid = _insert_file(db)
        seed_a = _insert_symbol(db, fid, "seed_a")
        seed_b = _insert_symbol(db, fid, "seed_b")
        _insert_chunk(db, seed_a)
        _insert_chunk(db, seed_b)

        for i in range(5):
            cid = _insert_symbol(db, fid, f"seed_a_callee_{i}")
            _insert_chunk(db, cid)
            db.insert_call_edges(
                [
                    CallEdge(
                        caller_id=seed_a,
                        callee_name=f"seed_a_callee_{i}",
                        line=i + 1,
                        callee_id=cid,
                    )
                ]
            )
        db.insert_def_use_edges(
            [
                DefUseEdge(
                    symbol_id=seed_a, var_name="shared", def_line=0, use_line=10, edge_type="use"
                )
            ]
        )

        seed_b_callee = _insert_symbol(db, fid, "seed_b_callee")
        _insert_chunk(db, seed_b_callee)
        db.insert_call_edges(
            [
                CallEdge(
                    caller_id=seed_b, callee_name="seed_b_callee", line=3, callee_id=seed_b_callee
                )
            ]
        )
        db.insert_def_use_edges(
            [DefUseEdge(symbol_id=seed_b, var_name="y", def_line=1, use_line=5, edge_type="use")]
        )
        db._conn.commit()

        seed_a_result = _make_search_result(db, seed_a, score=1.0)
        seed_b_result = _make_search_result(db, seed_b, score=0.9)
        extra = expand_with_dataflow(db, [seed_a_result, seed_b_result], max_extra=3)

        ids = [r.chunk.symbol_id for r in extra]
        assert seed_b_callee in ids, (
            "seed_b's callee was starved because seed_a's 5 candidates alone "
            "filled max_extra before seed_b was ever examined"
        )

    def test_source_is_dataflow_expansion(self, db: Database) -> None:
        fid = _insert_file(db)
        caller_id = _insert_symbol(db, fid, "caller5")
        callee_id = _insert_symbol(db, fid, "callee5")
        _insert_chunk(db, caller_id)
        _insert_chunk(db, callee_id)

        db.insert_call_edges(
            [CallEdge(caller_id=caller_id, callee_name="callee5", line=3, callee_id=callee_id)]
        )
        db.insert_def_use_edges(
            [DefUseEdge(symbol_id=caller_id, var_name="z", def_line=1, use_line=5, edge_type="use")]
        )
        db._conn.commit()

        result = _make_search_result(db, caller_id)
        extra = expand_with_dataflow(db, [result])
        assert extra
        assert all(r.source == "dataflow_expansion" for r in extra)


# ---------------------------------------------------------------------------
# rank_by_pagerank
# ---------------------------------------------------------------------------


class TestRankByPagerank:
    def test_empty_list_returns_empty(self, db: Database) -> None:
        result = rank_by_pagerank([], db)
        assert result == []

    def test_returns_all_symbol_ids(self, db: Database) -> None:
        fid = _insert_file(db)
        ids = [_insert_symbol(db, fid, f"sym_{i}") for i in range(3)]
        for i in range(3):
            _insert_chunk(db, ids[i])

        pr = rank_by_pagerank(ids, db)
        returned_ids = [x[0] for x in pr]
        for sid in ids:
            assert sid in returned_ids

    def test_scores_are_non_negative(self, db: Database) -> None:
        """PageRank scores must always be >= 0."""
        fid = _insert_file(db)
        a = _insert_symbol(db, fid, "hub")
        b = _insert_symbol(db, fid, "spoke1")
        c = _insert_symbol(db, fid, "spoke2")
        d = _insert_symbol(db, fid, "spoke3")
        for sym_id in (a, b, c, d):
            _insert_chunk(db, sym_id)

        # hub calls all spokes
        db.insert_call_edges(
            [
                CallEdge(caller_id=a, callee_name="spoke1", line=1, callee_id=b),
                CallEdge(caller_id=a, callee_name="spoke2", line=2, callee_id=c),
                CallEdge(caller_id=a, callee_name="spoke3", line=3, callee_id=d),
            ]
        )
        db._conn.commit()

        pr = rank_by_pagerank([a, b, c, d], db)
        for _, score in pr:
            assert score >= 0.0

    def test_hub_node_has_higher_pagerank_than_leaves(self, db: Database) -> None:
        """
        A node called by many others (high in-degree) should have higher PageRank.
        Hub is called by spoke1, spoke2, spoke3.
        """
        fid = _insert_file(db)
        hub = _insert_symbol(db, fid, "hub_central")
        spokes = [_insert_symbol(db, fid, f"leaf_{i}") for i in range(3)]
        _insert_chunk(db, hub)
        for s in spokes:
            _insert_chunk(db, s)

        # All spokes call the hub
        db.insert_call_edges(
            [
                CallEdge(caller_id=s, callee_name="hub_central", line=1, callee_id=hub)
                for s in spokes
            ]
        )
        db._conn.commit()

        pr = dict(rank_by_pagerank([hub] + spokes, db))
        # hub is called by 3 spokes → higher PageRank than any spoke
        assert pr[hub] > pr[spokes[0]]

    def test_fallback_uniform_scores_when_no_edges(self, db: Database) -> None:
        """When graph has no edges, scores should still be returned (uniform or fallback)."""
        fid = _insert_file(db)
        ids = [_insert_symbol(db, fid, f"isolated_{i}") for i in range(3)]
        for sid in ids:
            _insert_chunk(db, sid)
        # No call edges inserted

        pr = rank_by_pagerank(ids, db)
        # Should return (symbol_id, score) pairs — either from networkx or fallback
        assert len(pr) > 0
        for _, score in pr:
            assert score >= 0.0

    def test_generic_edges_never_leak_into_returned_ranking(self, db: Database) -> None:
        """Synthetic artifact nodes (source_ref strings) participate in the
        PageRank computation but must never appear in the caller-facing
        (symbol_id: int, score) result list."""
        fid = _insert_file(db)
        sym = _insert_symbol(db, fid, "login")
        _insert_chunk(db, sym)
        db.insert_generic_edges(
            [
                GenericEdge(
                    from_symbol_id=sym, source_ref="ticket:PROJ-1", edge_kind="references_ticket"
                )
            ]
        )

        pr = rank_by_pagerank([sym], db)
        returned_ids = [x[0] for x in pr]
        assert all(isinstance(sid, int) for sid in returned_ids)
        assert "ticket:PROJ-1" not in returned_ids

    def test_symbol_referenced_by_ticket_ranks_higher_than_equivalent_without(
        self, db: Database
    ) -> None:
        """Two symbols with equal call-graph centrality (both call the same
        third symbol) — the one that ALSO has a cross-source ticket edge
        should rank higher, proving the generic edge genuinely contributes
        centrality rather than being silently dropped. (An isolated symbol
        with zero edges of any kind isn't added to G at all — a pre-existing
        property of this function, not specific to generic edges — so this
        test gives both symbols an equal call-edge baseline instead.)"""
        fid = _insert_file(db)
        referenced = _insert_symbol(db, fid, "referenced_fn")
        plain = _insert_symbol(db, fid, "plain_fn")
        shared_callee = _insert_symbol(db, fid, "shared_callee")
        for sid in (referenced, plain, shared_callee):
            _insert_chunk(db, sid)
        db.insert_call_edges(
            [
                CallEdge(
                    caller_id=referenced,
                    callee_name="shared_callee",
                    line=1,
                    callee_id=shared_callee,
                ),
                CallEdge(
                    caller_id=plain, callee_name="shared_callee", line=1, callee_id=shared_callee
                ),
            ]
        )
        db._conn.commit()
        db.insert_generic_edges(
            [
                GenericEdge(
                    from_symbol_id=referenced,
                    source_ref="ticket:PROJ-1",
                    edge_kind="references_ticket",
                )
            ]
        )

        pr = dict(rank_by_pagerank([referenced, plain, shared_callee], db))
        assert pr[referenced] > pr[plain]

    def test_personalization_disabled_is_byte_identical_to_default_call(self, db: Database) -> None:
        """personalization_enabled defaults to False — must reproduce
        today's exact plain-PageRank scores, not merely 'similar' ones."""
        fid = _insert_file(db)
        a = _insert_symbol(db, fid, "hub")
        b = _insert_symbol(db, fid, "spoke1")
        c = _insert_symbol(db, fid, "spoke2")
        for sid in (a, b, c):
            _insert_chunk(db, sid)
        db.insert_call_edges(
            [
                CallEdge(caller_id=b, callee_name="hub", line=1, callee_id=a),
                CallEdge(caller_id=c, callee_name="hub", line=2, callee_id=a),
            ]
        )
        db._conn.commit()

        default_call = dict(rank_by_pagerank([a, b, c], db))
        explicit_false = dict(rank_by_pagerank([a, b, c], db, personalization_enabled=False))
        assert default_call == explicit_false

    def test_personalization_with_no_cross_source_edges_falls_back_to_uniform(
        self, db: Database
    ) -> None:
        """Opting in on a subgraph with zero generic_edges must not error or
        change scores — there's no seed set to personalize toward, so this
        degrades to the exact same plain-PageRank call."""
        fid = _insert_file(db)
        a = _insert_symbol(db, fid, "hub")
        b = _insert_symbol(db, fid, "spoke")
        for sid in (a, b):
            _insert_chunk(db, sid)
        db.insert_call_edges([CallEdge(caller_id=b, callee_name="hub", line=1, callee_id=a)])
        db._conn.commit()

        without_personalization = dict(rank_by_pagerank([a, b], db))
        with_personalization = dict(rank_by_pagerank([a, b], db, personalization_enabled=True))
        assert without_personalization == with_personalization

    def test_personalization_shifts_rank_toward_cross_source_linked_symbol(
        self, db: Database
    ) -> None:
        """Proves personalization= is genuinely wired into the nx.pagerank
        call, not a no-op — a prior version of this test passed identically
        whether personalization was hardcoded to None or threaded through,
        because its fixture never gave call-graph centrality and ticket
        linkage a chance to disagree.

        This fixture makes them disagree: `hub` is called by 8 distinct
        callers (real call-graph centrality, no ticket link at all); `target`
        has zero callers but IS linked to a ticket. With enough callers,
        plain PageRank ranks `hub` above `target` on pure call centrality.
        Personalization must be strong enough to flip that ordering —
        anything that would pass with personalization=None (e.g. a fixture
        where the ticket-linked node already wins on call-graph structure
        alone) does not actually exercise the personalization= kwarg.
        """
        fid = _insert_file(db)
        hub = _insert_symbol(db, fid, "hub_fn")
        target = _insert_symbol(db, fid, "target_fn")
        callers = [_insert_symbol(db, fid, f"caller_fn_{i}") for i in range(8)]
        for sid in (hub, target, *callers):
            _insert_chunk(db, sid)
        db.insert_call_edges(
            [
                CallEdge(caller_id=caller, callee_name="hub_fn", line=1, callee_id=hub)
                for caller in callers
            ]
        )
        db._conn.commit()
        db.insert_generic_edges(
            [
                GenericEdge(
                    from_symbol_id=target,
                    source_ref="ticket:PROJ-1",
                    edge_kind="references_ticket",
                )
            ]
        )

        all_ids = [hub, target, *callers]
        without = dict(rank_by_pagerank(all_ids, db))
        # Sanity check the fixture's premise: on pure call-graph structure,
        # the heavily-called hub outranks the zero-callers ticket target.
        assert without[hub] > without[target]

        with_ppr = dict(rank_by_pagerank(all_ids, db, personalization_enabled=True))
        # Personalization must be strong enough to flip that ordering — a
        # no-op personalization (e.g. personalization=None reaching
        # nx.pagerank regardless of the flag) would leave hub > target
        # unchanged, since that ordering is exactly what plain PageRank
        # already produces from call-graph structure alone.
        assert with_ppr[target] > with_ppr[hub]


# ---------------------------------------------------------------------------
# expand_with_imports
# ---------------------------------------------------------------------------


class TestExpandWithImports:
    def test_empty_results_returns_empty(self, db: Database) -> None:
        extra = expand_with_imports(db, results=[])
        assert extra == []

    def test_follows_forward_import_edge(self, db: Database) -> None:
        """
        file_a imports file_b (resolved). Querying symbols in file_a should
        expand to include symbols from file_b.
        """
        fid_a = _insert_file(db, "a.py")
        fid_b = _insert_file(db, "b.py")

        sym_a = _insert_symbol(db, fid_a, "func_a")
        sym_b = _insert_symbol(db, fid_b, "func_b")
        _insert_chunk(db, sym_a)
        _insert_chunk(db, sym_b)

        # a imports b (resolved)
        db.insert_imports([ImportEdge(file_id=fid_a, imported_from="b", imported_names=["func_b"])])
        db._conn.execute(
            "UPDATE imports SET imported_file_id = ? WHERE file_id = ?", (fid_b, fid_a)
        )
        db._conn.commit()

        result = _make_search_result(db, sym_a)
        extra = expand_with_imports(db, [result])

        ids = [r.chunk.symbol_id for r in extra]
        assert sym_b in ids

    def test_source_is_import_expansion(self, db: Database) -> None:
        fid_a = _insert_file(db, "src_a.py")
        fid_b = _insert_file(db, "src_b.py")
        sym_a = _insert_symbol(db, fid_a, "main_fn")
        sym_b = _insert_symbol(db, fid_b, "helper_fn")
        _insert_chunk(db, sym_a)
        _insert_chunk(db, sym_b)

        db.insert_imports(
            [ImportEdge(file_id=fid_a, imported_from="src_b", imported_names=["helper_fn"])]
        )
        db._conn.execute(
            "UPDATE imports SET imported_file_id = ? WHERE file_id = ?", (fid_b, fid_a)
        )
        db._conn.commit()

        result = _make_search_result(db, sym_a)
        extra = expand_with_imports(db, [result])
        assert all(r.source == "import_expansion" for r in extra)

    def test_no_imports_returns_empty(self, db: Database) -> None:
        fid = _insert_file(db)
        sym_id = _insert_symbol(db, fid, "standalone")
        _insert_chunk(db, sym_id)

        result = _make_search_result(db, sym_id)
        extra = expand_with_imports(db, [result])
        assert extra == []

    def test_max_extra_zero_returns_empty(self, db: Database) -> None:
        fid_a = _insert_file(db, "p.py")
        fid_b = _insert_file(db, "q.py")
        sym_a = _insert_symbol(db, fid_a, "p_fn")
        sym_b = _insert_symbol(db, fid_b, "q_fn")
        _insert_chunk(db, sym_a)
        _insert_chunk(db, sym_b)

        db.insert_imports([ImportEdge(file_id=fid_a, imported_from="q", imported_names=["q_fn"])])
        db._conn.execute(
            "UPDATE imports SET imported_file_id = ? WHERE file_id = ?", (fid_b, fid_a)
        )
        db._conn.commit()

        result = _make_search_result(db, sym_a)
        extra = expand_with_imports(db, [result], max_extra=0)
        assert extra == []


# ---------------------------------------------------------------------------
# expand_with_type_edges
# ---------------------------------------------------------------------------


class TestExpandWithTypeEdges:
    def test_empty_results_returns_empty(self, db: Database) -> None:
        extra = expand_with_type_edges(db, results=[])
        assert extra == []

    def test_follows_parent_type(self, db: Database) -> None:
        """Child class → expand should surface the parent class."""
        fid = _insert_file(db)
        child_id = _insert_symbol(db, fid, "ChildClass", SymbolKind.CLASS)
        parent_id = _insert_symbol(db, fid, "ParentClass", SymbolKind.CLASS)
        _insert_chunk(db, child_id)
        _insert_chunk(db, parent_id)

        db.insert_type_edges(
            [
                TypeEdge(
                    from_symbol_id=child_id,
                    to_type_name="ParentClass",
                    edge_kind="extends",
                    to_symbol_id=parent_id,
                )
            ]
        )
        db._conn.commit()

        result = _make_search_result(db, child_id)
        extra = expand_with_type_edges(db, [result])

        ids = [r.chunk.symbol_id for r in extra]
        assert parent_id in ids

    def test_follows_child_type(self, db: Database) -> None:
        """Parent class → expand should surface implementing child classes."""
        fid = _insert_file(db)
        parent_id = _insert_symbol(db, fid, "BaseClass", SymbolKind.CLASS)
        child_id = _insert_symbol(db, fid, "SubClass", SymbolKind.CLASS)
        _insert_chunk(db, parent_id)
        _insert_chunk(db, child_id)

        db.insert_type_edges(
            [
                TypeEdge(
                    from_symbol_id=child_id,
                    to_type_name="BaseClass",
                    edge_kind="extends",
                    to_symbol_id=parent_id,
                )
            ]
        )
        db._conn.commit()

        result = _make_search_result(db, parent_id)
        extra = expand_with_type_edges(db, [result])

        ids = [r.chunk.symbol_id for r in extra]
        assert child_id in ids

    def test_source_is_type_expansion(self, db: Database) -> None:
        fid = _insert_file(db)
        parent_id = _insert_symbol(db, fid, "IFace", SymbolKind.INTERFACE)
        impl_id = _insert_symbol(db, fid, "ImplClass", SymbolKind.CLASS)
        _insert_chunk(db, parent_id)
        _insert_chunk(db, impl_id)

        db.insert_type_edges(
            [
                TypeEdge(
                    from_symbol_id=impl_id,
                    to_type_name="IFace",
                    edge_kind="implements",
                    to_symbol_id=parent_id,
                )
            ]
        )
        db._conn.commit()

        result = _make_search_result(db, impl_id)
        extra = expand_with_type_edges(db, [result])
        assert all(r.source == "type_expansion" for r in extra)

    def test_no_type_edges_returns_empty(self, db: Database) -> None:
        fid = _insert_file(db)
        sym_id = _insert_symbol(db, fid, "SoloClass", SymbolKind.CLASS)
        _insert_chunk(db, sym_id)

        result = _make_search_result(db, sym_id)
        extra = expand_with_type_edges(db, [result])
        assert extra == []


# ---------------------------------------------------------------------------
# seed_from_import_paths
# ---------------------------------------------------------------------------


class TestSeedFromImportPaths:
    def test_empty_patterns_returns_empty(self, db: Database) -> None:
        assert seed_from_import_paths(db, patterns=[]) == []

    def test_max_extra_zero_returns_empty(self, db: Database) -> None:
        fid = _insert_file(db)
        sym_id = _insert_symbol(db, fid, "anything")
        _insert_chunk(db, sym_id)
        db.insert_imports(
            [ImportEdge(file_id=fid, imported_from="@shared/utils", imported_names=["*"])]
        )
        db._conn.commit()

        result = seed_from_import_paths(db, patterns=["@shared"], max_extra=0)
        assert result == []

    def test_matches_import_path_pattern(self, db: Database) -> None:
        """
        File that imports '@shared/utils' should be seeded when pattern='@shared'.
        """
        fid = _insert_file(db, "consumer.py")
        sym_id = _insert_symbol(db, fid, "consumer_fn")
        _insert_chunk(db, sym_id)

        db.insert_imports(
            [ImportEdge(file_id=fid, imported_from="@shared/utils", imported_names=["helper"])]
        )
        db._conn.commit()

        results = seed_from_import_paths(db, patterns=["@shared"])
        ids = [r.chunk.symbol_id for r in results]
        assert sym_id in ids

    def test_no_matching_pattern_returns_empty(self, db: Database) -> None:
        fid = _insert_file(db, "other.py")
        sym_id = _insert_symbol(db, fid, "other_fn")
        _insert_chunk(db, sym_id)

        db.insert_imports(
            [ImportEdge(file_id=fid, imported_from="@core/internal", imported_names=["x"])]
        )
        db._conn.commit()

        results = seed_from_import_paths(db, patterns=["@shared"])
        assert results == []

    def test_source_is_import_path_seed(self, db: Database) -> None:
        fid = _insert_file(db, "service.py")
        sym_id = _insert_symbol(db, fid, "svc_fn")
        _insert_chunk(db, sym_id)

        db.insert_imports(
            [ImportEdge(file_id=fid, imported_from="@auth/jwt", imported_names=["verify"])]
        )
        db._conn.commit()

        results = seed_from_import_paths(db, patterns=["@auth"])
        assert all(r.source == "import_path_seed" for r in results)
