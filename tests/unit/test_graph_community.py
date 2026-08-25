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


class TestAlgorithmSelection:
    """The `algorithm=` branches other than the "louvain" default.

    Round-11's own suspicion, confirmed by reading the mutmut survivors before
    writing anything here: every existing test calls detect_communities() with
    the implicit default, so "girvan_newman", "label_prop", the `else: raise
    ValueError`, and the `except Exception` fallback that catches it are ALL
    unexercised -- flipping `==` to `!=` on any of those branch conditions, or
    replacing the fallback's `mapping[int(node_id)] = 0` with `None` or `1`,
    changed nothing any test could see.
    """

    def test_invalid_algorithm_falls_back_to_all_nodes_in_community_zero(
        self, tmp_path: Path
    ) -> None:
        """MUTATION: `mapping[int(node_id)] = 0` (the except-Exception fallback)
        -> `= None` or `= 1`. An unknown algorithm name hits `else: raise
        ValueError(...)`, which the surrounding `except Exception` catches --
        the only path in this function that reaches that fallback at all."""
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg, algorithm="not-a-real-algorithm")
        assert set(mapping) == set(sids), (
            "the fallback must still map every node, same as the ValueError "
            f"path's caller-visible contract; got keys {sorted(mapping)}"
        )
        assert set(mapping.values()) == {0}, (
            f"every node must fall back to community 0 specifically, got "
            f"{sorted(set(mapping.values()))}"
        )

    def test_girvan_newman_algorithm_returns_mapping_for_all_nodes(self, tmp_path: Path) -> None:
        """MUTATION: `elif algorithm == "girvan_newman":` -> a typo'd string
        literal (e.g. `"XXgirvan_newmanXX"`), or `nx.community.girvan_newman
        (G_connected)` -> `(None)`, or `next(gen)` -> `None`.

        Asserting only "every node got SOME int community id" is too weak: a
        typo'd condition falls through to `else: raise ValueError`, which the
        `except Exception` catches and maps EVERY node to community 0 -- a
        dict that is still `{sid: <int>}` for every sid, so that shape alone
        cannot tell the real girvan_newman result from the exception
        fallback. This asserts more than one DISTINCT community id came back,
        which the all-0 fallback can never produce.
        """
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg, algorithm="girvan_newman")
        assert set(mapping) == set(sids)
        for community_id in mapping.values():
            assert isinstance(community_id, int)
        assert len(set(mapping.values())) > 1, (
            f"girvan_newman on two disconnected triangles must not collapse "
            f"into one community id -- got {mapping}, which is exactly the "
            "except-Exception fallback's signature (every node -> 0)"
        )

    def test_label_prop_algorithm_returns_mapping_for_all_nodes(self, tmp_path: Path) -> None:
        """MUTATION: `elif algorithm == "label_prop":` -> a typo'd string
        literal, or `label_propagation_communities(G_connected)` -> `(None)`.

        Same weak-assertion trap as girvan_newman above: label_prop on this
        fixture's two disconnected triangles deterministically separates them,
        so this pins the auth cluster (sids[0:3]) and the db cluster
        (sids[3:6]) landing in DIFFERENT communities specifically, which the
        all-0 exception fallback can never produce.
        """
        db, sids = _build_clustered_db(tmp_path)
        cg = CodeGraph(db)
        mapping = detect_communities(cg, algorithm="label_prop")
        assert set(mapping) == set(sids)
        for community_id in mapping.values():
            assert isinstance(community_id, int)
        assert mapping[sids[0]] != mapping[sids[3]], (
            f"the two disconnected triangles (auth: {sids[0:3]}, db: "
            f"{sids[3:6]}) must land in different communities; got {mapping}"
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

    def test_same_community_non_neighbor_still_reaches_frontier(self):
        """MUTATION: `if n in partition` -> `if n not in partition` inside the
        `seed_communities = {...}` comprehension, or `frontier.add(node_id)` ->
        `frontier.add(None)`.

        Every OTHER same-community test here (`_make_graph`'s triangle 1-2-3)
        has every same-community node reachable through the NEIGHBOR step
        alone, so the "same community" step's own contribution is never
        isolated -- flipping `in` to `not in` still passes them, because the
        neighbor step already put 2 and 3 in the frontier before the community
        step runs at all.

        PRECONDITION this test needs to discriminate: node 5 must be in the
        SAME community as the seed but have NO edge to it (or to anything the
        seed neighbors), or the neighbor step would smuggle it in too.
        """
        from trelix.graph.community import compute_affected_frontier

        G = nx.Graph()
        G.add_edge(1, 2)  # seed's only neighbor
        G.add_node(5)  # same community as seed, zero edges -- not a neighbor
        partition = {1: 0, 2: 0, 5: 0}
        assert not G.has_edge(1, 5) and 5 not in G.neighbors(1), (
            "precondition failed: node 5 must not be a neighbor of the seed, "
            "or this test cannot isolate the same-community step"
        )
        frontier = compute_affected_frontier(G, seed_nodes={1}, partition=partition)
        assert 5 in frontier, (
            "node 5 shares the seed's community but has no edge to it at all -- "
            "only the same-community step (not the neighbor step) can have put "
            f"it in the frontier; got {sorted(frontier)}"
        )
        assert None not in frontier


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

    def test_default_frontier_threshold_falls_back_to_full_at_100_percent(self, tmp_path):
        """MUTATION: the DEFAULT `frontier_threshold: float = 0.5` -> `1.5`.
        A frontier can never exceed 100% of the graph's nodes (ratio in
        [0, 1]), so a default >= 1.0 makes the fallback UNREACHABLE via the
        default for ANY input. `test_large_frontier_falls_back_to_full` above
        only asserts `len(result) == 6`, which is identical whichever path
        runs, so it cannot tell the fallback from the incremental path firing.

        This test instead distinguishes the two paths by their COMMUNITY-ID
        NUMBERING SCHEME: the fallback calls detect_communities(), whose ids
        start at 0 via a fresh enumerate(); the incremental path offsets new
        ids from `max(prev_partition.values()) + 1`. Giving `prev_partition`
        a deliberately large existing id (7) makes the two schemes land in
        non-overlapping ranges.
        """
        from trelix.graph.community import detect_communities_incremental

        cg = self._make_cg_with_partition(tmp_path)
        prev = {1: 7, 2: 7, 3: 7, 4: 7, 5: 7, 6: 7}
        result = detect_communities_incremental(
            cg, seed_nodes={1, 2, 3, 4, 5, 6}, prev_partition=prev
        )
        assert len(result) == 6
        assert all(cid < 7 for cid in result.values()), (
            "seeding every node makes the frontier 100% of the graph, which "
            "must cross the DEFAULT 0.5 threshold and fall back to fresh "
            "detect_communities() ids (small, enumerate()-based) -- got "
            f"{sorted(set(result.values()))}, which looks like the incremental "
            "path's max_existing(7)+1+... offset instead"
        )

    def test_new_frontier_communities_get_ids_offset_from_existing_max(self, tmp_path):
        """MUTATION: `global_id = max_existing + 1 + sub_id` -> any altered
        term. A seed touching only ONE of the two triangles keeps the frontier
        ratio at exactly 0.5 (NOT `> 0.5`), which stays on the INCREMENTAL
        path rather than falling back -- so the re-clustered frontier's new
        community id must be `max(prev_partition.values()) + 1`, colliding
        with neither 0 nor any existing prev id.
        """
        from trelix.graph.community import detect_communities_incremental

        cg = self._make_cg_with_partition(tmp_path)
        prev = {1: 0, 2: 0, 3: 0, 4: 4, 5: 4, 6: 4}
        result = detect_communities_incremental(cg, seed_nodes={1}, prev_partition=prev)
        assert len(result) == 6
        # frontier = {1} + neighbors{2,3} + same-community-as-1 {1,2,3} = {1,2,3};
        # ratio 3/6 == 0.5, NOT > 0.5 -- stays incremental, does not fall back.
        assert result[1] == result[2] == result[3] == 5, (
            "the frontier {1,2,3} triangle re-clusters as ONE community via "
            "Louvain (it is fully connected), so its new id must be "
            f"max_existing(4) + 1 + sub_id(0) == 5; got {result[1]!r}, "
            f"{result[2]!r}, {result[3]!r}"
        )
        assert result[4] == 4 and result[5] == 4 and result[6] == 4, (
            "nodes 4,5,6 are outside the frontier and must keep their "
            "previous community id 4 unchanged"
        )


class TestCommunitySummaryContent:
    """`get_community_summary`'s VALUES, not just its keys.

    test_community_summary_structure (above) only checks that each dict has
    the right KEYS -- a summary whose top_files/top_symbols silently hold None
    or an empty list, or whose file_counts dict was never populated, still has
    every key and passes that test. All three tests below build a CodeGraph's
    node attrs directly (bypassing detect_communities/Louvain, which would add
    nondeterminism this function itself does not own) and pin real content.
    """

    def _make_cg(self) -> CodeGraph:
        cg = CodeGraph.__new__(CodeGraph)
        cg._g = nx.MultiDiGraph()
        return cg

    def test_top_symbols_and_top_files_contain_real_values_not_placeholders(self) -> None:
        """MUTATION: `by_community[int(cid)].append(node_id)` -> `.append(None)`,
        or `f = attrs.get("file", "")` -> `f = None`, or the `attrs.get("name",
        "")` fallback key typo'd -- all leave a structurally-valid, content-
        empty summary."""
        cg = self._make_cg()
        cg._g.add_node(1, community=0, file="auth.py", qualified_name="Auth.login")
        # No qualified_name here -- exercises the `attrs.get("name", "")`
        # fallback specifically (`qualified_name or name`).
        cg._g.add_node(2, community=0, file="auth.py", name="logout")
        summary = get_community_summary(cg)
        assert len(summary) == 1
        item = summary[0]
        assert item["top_files"] == ["auth.py"]
        assert item["top_symbols"] == ["Auth.login", "logout"]

    def test_top_files_truncates_to_five_and_top_symbols_truncates_to_ten(self) -> None:
        """MUTATION: `file_counts.most_common(5)` -> `most_common(None)`
        (unlimited), or `symbol_names[:10]` -> `[:11]`. Both need MORE than
        the cap in the input to be observable at all -- a fixture with <= 5
        files or <= 10 symbols can never discriminate this, because both
        widths would agree by construction."""
        cg = self._make_cg()
        # 7 distinct files with strictly decreasing member counts, so
        # most_common(5) has an unambiguous, tie-free top 5. 7+6+5+4+3+2+1 = 28
        # symbols total, so symbol_names[:10] truncation is exercised too.
        node_id = 0
        expected_first_ten: list[str] = []
        for file_index, count in enumerate([7, 6, 5, 4, 3, 2, 1]):
            fname = f"f{file_index}.py"
            for i in range(count):
                name = f"{fname}:{i}"
                cg._g.add_node(node_id, community=0, file=fname, qualified_name=name)
                if len(expected_first_ten) < 10:
                    expected_first_ten.append(name)
                node_id += 1
        summary = get_community_summary(cg)
        assert len(summary) == 1
        item = summary[0]
        assert item["top_files"] == ["f0.py", "f1.py", "f2.py", "f3.py", "f4.py"], (
            f"most_common(5) must cap at the real top 5 by count, got {item['top_files']}"
        )
        assert item["top_symbols"] == expected_first_ten, (
            f"symbol_names[:10] must cap at exactly 10 entries in insertion "
            f"order, got {len(item['top_symbols'])}: {item['top_symbols']}"
        )

    def test_summary_sorts_by_size_then_id_even_when_id_order_disagrees(self) -> None:
        """MUTATION: `sorted(by_community.items(), key=lambda kv: (-len(kv[1]),
        kv[0]))` -> the key dropped entirely, sorting by community_id ascending
        only.

        PRECONDITION this test needs to discriminate: community_id order must
        actively DISAGREE with size-descending order (the small community gets
        the LOWER id), or dropping the key would coincidentally produce the
        same output and this test would pass for the wrong reason.
        """
        cg = self._make_cg()
        cg._g.add_node(100, community=1, file="a.py")  # community 1: size 1
        cg._g.add_node(200, community=9, file="b.py")  # community 9: size 3
        cg._g.add_node(201, community=9, file="b.py")
        cg._g.add_node(202, community=9, file="b.py")
        assert sorted([1, 9]) == [1, 9], (
            "precondition sanity: id-ascending order is [1, 9] (small community "
            "first) -- this must disagree with size-descending order below"
        )
        summary = get_community_summary(cg)
        ids_in_output = [item["community_id"] for item in summary]
        assert ids_in_output == [9, 1], (
            "expected size-descending order (community 9, size 3, first; then "
            f"community 1, size 1) but got {ids_in_output} -- dropping the sort "
            "key would instead give id-ascending [1, 9]"
        )
