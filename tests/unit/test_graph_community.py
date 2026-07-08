"""Tests for community detection on CodeGraph."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import assign_communities, detect_communities, get_community_summary
from trelix.store.db import Database


def _build_clustered_db(tmp_path: Path) -> tuple[Database, list[int]]:
    """Build a DB with two clearly separated clusters."""
    db = Database(tmp_path / "index.db")

    def _file(name: str) -> int:
        f = IndexedFile(
            path=f"/r/{name}", rel_path=name, language=Language.PYTHON, hash="x", size_bytes=10
        )
        return db.upsert_file(f)

    def _sym(fid: int, name: str) -> int:
        s = Symbol(
            file_id=fid,
            name=name,
            qualified_name=name,
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=5,
            signature=f"def {name}()",
            body="pass",
        )
        return db.insert_symbol(s)

    # Cluster A: auth module (3 symbols, densely connected)
    fid_a = _file("auth.py")
    a1 = _sym(fid_a, "login")
    a2 = _sym(fid_a, "logout")
    a3 = _sym(fid_a, "hash_password")

    # Cluster B: db module (3 symbols, densely connected)
    fid_b = _file("db.py")
    b1 = _sym(fid_b, "query")
    b2 = _sym(fid_b, "insert")
    b3 = _sym(fid_b, "connect")

    # Dense intra-cluster edges
    db.insert_call_edges(
        [
            CallEdge(caller_id=a1, callee_name="hash_password", callee_id=a3, line=2),
            CallEdge(caller_id=a2, callee_name="hash_password", callee_id=a3, line=3),
            CallEdge(caller_id=b1, callee_name="connect", callee_id=b3, line=2),
            CallEdge(caller_id=b2, callee_name="connect", callee_id=b3, line=3),
        ]
    )

    return db, [a1, a2, a3, b1, b2, b3]


class TestCommunityDetection:
    def test_returns_mapping_for_all_nodes(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assert isinstance(mapping, dict)
        for sid in sids:
            assert sid in mapping

    def test_community_ids_are_ints(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        for cid in mapping.values():
            assert isinstance(cid, int)

    def test_assign_sets_node_attrs(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assign_communities(cg, mapping)
        for sid in sids:
            assert cg.nx.nodes[sid]["community"] is not None

    def test_community_summary_structure(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assign_communities(cg, mapping)
        summary = get_community_summary(cg)
        assert isinstance(summary, list)
        assert len(summary) >= 1
        for item in summary:
            assert "community_id" in item
            assert "size" in item
            assert "top_files" in item
            assert "top_symbols" in item

    def test_empty_graph(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        assert mapping == {}
        summary = get_community_summary(cg)
        assert summary == []

    def test_clusters_get_distinct_community_ids(self, tmp_path: Path) -> None:
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg)
        # auth cluster (sids[0..2]) and db cluster (sids[3..5]) should be in different communities
        auth_community = mapping.get(sids[0])
        db_community = mapping.get(sids[3])
        # With dense intra-cluster and no inter-cluster edges, they must differ
        if auth_community is not None and db_community is not None:
            assert auth_community != db_community, (
                f"Expected auth cluster (community {auth_community}) != "
                f"db cluster (community {db_community})"
            )


class TestAffectedFrontier:
    def _make_graph(self):
        """Triangle A-B-C, separate node D."""
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (1, 3)])
        G.add_node(4)
        return G

    def test_seed_nodes_always_in_frontier(self):
        from trelix.graph.community import compute_affected_frontier

        G = self._make_graph()
        partition = {1: 0, 2: 0, 3: 0, 4: 1}
        frontier = compute_affected_frontier(G, seed_nodes={2}, partition=partition)
        assert 2 in frontier

    def test_neighbors_of_seed_in_frontier(self):
        from trelix.graph.community import compute_affected_frontier

        G = self._make_graph()
        partition = {1: 0, 2: 0, 3: 0, 4: 1}
        frontier = compute_affected_frontier(G, seed_nodes={2}, partition=partition)
        # node 2 neighbors are 1 and 3
        assert 1 in frontier
        assert 3 in frontier

    def test_same_community_nodes_in_frontier(self):
        from trelix.graph.community import compute_affected_frontier

        G = self._make_graph()
        # nodes 1,2,3 all in community 0; node 4 in community 1
        partition = {1: 0, 2: 0, 3: 0, 4: 1}
        frontier = compute_affected_frontier(G, seed_nodes={1}, partition=partition)
        # all of community 0 should be included
        assert {1, 2, 3}.issubset(frontier)

    def test_unrelated_node_not_in_frontier(self):
        from trelix.graph.community import compute_affected_frontier

        G = self._make_graph()
        partition = {1: 0, 2: 0, 3: 0, 4: 1}
        frontier = compute_affected_frontier(G, seed_nodes={1}, partition=partition)
        # node 4 is isolated, different community — should NOT be in frontier
        assert 4 not in frontier

    def test_empty_seed_returns_empty(self):
        from trelix.graph.community import compute_affected_frontier

        G = self._make_graph()
        partition = {1: 0, 2: 0, 3: 0, 4: 1}
        frontier = compute_affected_frontier(G, seed_nodes=set(), partition=partition)
        assert frontier == set()

    def test_empty_partition_returns_seed_plus_neighbors(self):
        from trelix.graph.community import compute_affected_frontier

        G = self._make_graph()
        partition = {}
        frontier = compute_affected_frontier(G, seed_nodes={2}, partition=partition)
        assert 2 in frontier
        assert 1 in frontier
        assert 3 in frontier


class TestIncrementalLouvain:
    def _make_cg_with_partition(self, tmp_path):
        """Build a CodeGraph with 6 nodes in 2 communities."""
        from trelix.graph.code_graph import CodeGraph

        cg = CodeGraph.__new__(CodeGraph)
        import networkx as nx

        cg._g = nx.MultiDiGraph()
        # Community 0: nodes 1,2,3 (triangle)
        cg._g.add_nodes_from([1, 2, 3, 4, 5, 6])
        cg._g.add_edges_from([(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6)])
        return cg

    def test_returns_complete_partition(self, tmp_path):
        from trelix.graph.community import detect_communities_incremental

        cg = self._make_cg_with_partition(tmp_path)
        prev = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
        result = detect_communities_incremental(cg, seed_nodes={2}, prev_partition=prev)
        # All 6 nodes must appear in result
        assert set(result.keys()) == {1, 2, 3, 4, 5, 6}

    def test_non_frontier_nodes_keep_prev_community(self, tmp_path):
        from trelix.graph.community import detect_communities_incremental

        cg = self._make_cg_with_partition(tmp_path)
        prev = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
        result = detect_communities_incremental(cg, seed_nodes={2}, prev_partition=prev)
        # nodes 4,5,6 are not in frontier (different community, not neighbors of 2)
        # they should keep community 1 from prev_partition
        assert result[4] == 1
        assert result[5] == 1
        assert result[6] == 1

    def test_empty_prev_falls_back_to_full(self, tmp_path):
        from trelix.graph.community import detect_communities_incremental

        cg = self._make_cg_with_partition(tmp_path)
        # empty prev_partition → full Louvain
        result = detect_communities_incremental(cg, seed_nodes={1}, prev_partition={})
        assert len(result) == 6

    def test_large_frontier_falls_back_to_full(self, tmp_path):
        from trelix.graph.community import detect_communities_incremental

        cg = self._make_cg_with_partition(tmp_path)
        prev = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
        # seed all nodes → frontier = 100% → full Louvain fallback
        result = detect_communities_incremental(
            cg, seed_nodes={1, 2, 3, 4, 5, 6}, prev_partition=prev
        )
        assert len(result) == 6
