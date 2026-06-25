"""
Unit tests for trelix.retrieval.graph (call/import/type graph expansion + PageRank).

All tests use a real in-memory SQLite Database (tmp_path) seeded with minimal
fixtures — no external services required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trelix.core.models import (
    CallEdge,
    Chunk,
    ImportEdge,
    IndexedFile,
    Language,
    SearchResult,
    Symbol,
    SymbolKind,
    TypeEdge,
)
from trelix.store.db import Database
from trelix.retrieval.graph import (
    expand_with_call_graph,
    expand_with_imports,
    expand_with_type_edges,
    rank_by_pagerank,
    seed_from_import_paths,
)


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

        db.insert_call_edges([CallEdge(caller_id=caller_id, callee_name="callee", line=2, callee_id=callee_id)])
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

        db.insert_call_edges([CallEdge(caller_id=caller_id, callee_name="calleeA", line=3, callee_id=callee_id)])
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

        db.insert_call_edges([CallEdge(caller_id=caller_id, callee_name="fn_y", line=1, callee_id=callee_id)])
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
        db.insert_call_edges([
            CallEdge(caller_id=a_id, callee_name="fn_b", line=1, callee_id=b_id),
            CallEdge(caller_id=b_id, callee_name="fn_a", line=2, callee_id=a_id),
        ])
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
            db.insert_call_edges([CallEdge(caller_id=root_id, callee_name=f"callee_{i}", line=i, callee_id=cid)])
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

        db.insert_call_edges([CallEdge(caller_id=caller_id, callee_name="small_fn", line=1, callee_id=callee_id)])
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
        db.insert_call_edges([
            CallEdge(caller_id=a, callee_name="spoke1", line=1, callee_id=b),
            CallEdge(caller_id=a, callee_name="spoke2", line=2, callee_id=c),
            CallEdge(caller_id=a, callee_name="spoke3", line=3, callee_id=d),
        ])
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
        db.insert_call_edges([
            CallEdge(caller_id=s, callee_name="hub_central", line=1, callee_id=hub)
            for s in spokes
        ])
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
        db._conn.execute("UPDATE imports SET imported_file_id = ? WHERE file_id = ?", (fid_b, fid_a))
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

        db.insert_imports([ImportEdge(file_id=fid_a, imported_from="src_b", imported_names=["helper_fn"])])
        db._conn.execute("UPDATE imports SET imported_file_id = ? WHERE file_id = ?", (fid_b, fid_a))
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
        db._conn.execute("UPDATE imports SET imported_file_id = ? WHERE file_id = ?", (fid_b, fid_a))
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

        db.insert_type_edges([TypeEdge(from_symbol_id=child_id, to_type_name="ParentClass", edge_kind="extends", to_symbol_id=parent_id)])
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

        db.insert_type_edges([TypeEdge(from_symbol_id=child_id, to_type_name="BaseClass", edge_kind="extends", to_symbol_id=parent_id)])
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

        db.insert_type_edges([TypeEdge(from_symbol_id=impl_id, to_type_name="IFace", edge_kind="implements", to_symbol_id=parent_id)])
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
        db.insert_imports([ImportEdge(file_id=fid, imported_from="@shared/utils", imported_names=["*"])])
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

        db.insert_imports([ImportEdge(file_id=fid, imported_from="@shared/utils", imported_names=["helper"])])
        db._conn.commit()

        results = seed_from_import_paths(db, patterns=["@shared"])
        ids = [r.chunk.symbol_id for r in results]
        assert sym_id in ids

    def test_no_matching_pattern_returns_empty(self, db: Database) -> None:
        fid = _insert_file(db, "other.py")
        sym_id = _insert_symbol(db, fid, "other_fn")
        _insert_chunk(db, sym_id)

        db.insert_imports([ImportEdge(file_id=fid, imported_from="@core/internal", imported_names=["x"])])
        db._conn.commit()

        results = seed_from_import_paths(db, patterns=["@shared"])
        assert results == []

    def test_source_is_import_path_seed(self, db: Database) -> None:
        fid = _insert_file(db, "service.py")
        sym_id = _insert_symbol(db, fid, "svc_fn")
        _insert_chunk(db, sym_id)

        db.insert_imports([ImportEdge(file_id=fid, imported_from="@auth/jwt", imported_names=["verify"])])
        db._conn.commit()

        results = seed_from_import_paths(db, patterns=["@auth"])
        assert all(r.source == "import_path_seed" for r in results)
