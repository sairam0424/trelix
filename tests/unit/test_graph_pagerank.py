"""Tests for PageRank symbol importance scoring."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from trelix.core.models import CallEdge, GenericEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import compute_pagerank
from trelix.graph.persistence import get_top_central_symbols, save_graph_metadata
from trelix.store.db import Database


def _build_star_graph(tmp_path: Path) -> tuple[Database, CodeGraph, int]:
    """Build a star graph: hub calls 3 leaves. Hub should have highest PageRank."""
    db = Database(tmp_path / "index.db")
    fid = db.upsert_file(
        IndexedFile(
            path="/r/a.py",
            rel_path="a.py",
            language=Language.PYTHON,
            hash="x",
            size_bytes=10,
        )
    )
    hub = db.insert_symbol(
        Symbol(
            file_id=fid,
            name="hub",
            qualified_name="hub",
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=5,
            signature="def hub()",
            body="",
        )
    )
    leaf1 = db.insert_symbol(
        Symbol(
            file_id=fid,
            name="leaf1",
            qualified_name="leaf1",
            kind=SymbolKind.FUNCTION,
            line_start=10,
            line_end=14,
            signature="def leaf1()",
            body="",
        )
    )
    leaf2 = db.insert_symbol(
        Symbol(
            file_id=fid,
            name="leaf2",
            qualified_name="leaf2",
            kind=SymbolKind.FUNCTION,
            line_start=20,
            line_end=24,
            signature="def leaf2()",
            body="",
        )
    )
    leaf3 = db.insert_symbol(
        Symbol(
            file_id=fid,
            name="leaf3",
            qualified_name="leaf3",
            kind=SymbolKind.FUNCTION,
            line_start=30,
            line_end=34,
            signature="def leaf3()",
            body="",
        )
    )
    # leaf1, leaf2, leaf3 all call hub (hub is the target, gets PageRank from incoming)
    db.insert_call_edges(
        [
            CallEdge(caller_id=leaf1, callee_name="hub", callee_id=hub, line=11),
            CallEdge(caller_id=leaf2, callee_name="hub", callee_id=hub, line=21),
            CallEdge(caller_id=leaf3, callee_name="hub", callee_id=hub, line=31),
        ]
    )
    cg = CodeGraph(db)
    return db, cg, hub


class TestComputePagerank:
    def test_returns_dict_of_node_scores(self, tmp_path: Path) -> None:
        _, cg, _ = _build_star_graph(tmp_path)
        scores = compute_pagerank(cg)
        assert isinstance(scores, dict)
        assert all(isinstance(v, float) for v in scores.values())

    def test_hub_has_higher_score_than_leaves(self, tmp_path: Path) -> None:
        _, cg, hub_id = _build_star_graph(tmp_path)
        scores = compute_pagerank(cg)
        hub_score = scores.get(hub_id, 0.0)
        leaf_scores = [v for k, v in scores.items() if k != hub_id]
        assert hub_score > max(leaf_scores, default=0.0)

    def test_scores_normalized_0_to_1(self, tmp_path: Path) -> None:
        _, cg, _ = _build_star_graph(tmp_path)
        scores = compute_pagerank(cg)
        assert max(scores.values()) <= 1.0 + 1e-9
        assert min(scores.values()) >= 0.0 - 1e-9

    def test_empty_graph_returns_empty_dict(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        cg = CodeGraph(db)
        assert compute_pagerank(cg) == {}

    def test_alpha_parameter_is_actually_threaded_into_nx_pagerank(self, tmp_path: Path) -> None:
        """MUTATION: `nx.pagerank(g, alpha=alpha, max_iter=100, ...)` -> the
        `alpha=alpha` kwarg dropped, falling back to nx.pagerank's own default
        alpha (0.85) -- which IS compute_pagerank's own default too, so no
        test calling with the implicit default alpha can ever observe this.
        Calling with a distinctly non-default alpha is the only way; networkx
        itself (not compute_pagerank) is the independent oracle for what that
        alpha value should produce.
        """
        _, cg, hub_id = _build_star_graph(tmp_path)
        # A LEAF's score, not the hub's: the hub has the max raw PageRank in
        # this topology by construction, and normalization always rescales
        # the max score to exactly 1.0 regardless of alpha -- so the hub's own
        # (post-normalization) score is trivially 1.0 for every alpha and
        # cannot discriminate this mutation at all.
        default_scores = compute_pagerank(cg)
        leaf_id = next(k for k in default_scores if k != hub_id)
        low_alpha_scores = compute_pagerank(cg, alpha=0.2)
        assert default_scores[leaf_id] != pytest.approx(low_alpha_scores[leaf_id]), (
            "alpha=0.2 must move a leaf's score away from the alpha=0.85 "
            f"default; both were {default_scores[leaf_id]!r} -- alpha= is not "
            "reaching nx.pagerank at all"
        )
        oracle_raw = nx.pagerank(cg.nx, alpha=0.2, max_iter=100, personalization=None)
        oracle_max = max(oracle_raw.values())
        oracle_normalized = {k: v / oracle_max for k, v in oracle_raw.items()}
        assert low_alpha_scores[leaf_id] == pytest.approx(oracle_normalized[leaf_id]), (
            "compute_pagerank(alpha=0.2) must match nx.pagerank(alpha=0.2) "
            "directly (normalized the same way), not nx.pagerank's own "
            f"default-alpha result; got {low_alpha_scores[leaf_id]!r}, oracle "
            f"says {oracle_normalized[leaf_id]!r}"
        )

    def test_personalization_disabled_is_default_and_unchanged(self, tmp_path: Path) -> None:
        """personalization_enabled defaults to False — must reproduce
        today's exact scores, byte-identical to the pre-PPR behavior."""
        _, cg, _ = _build_star_graph(tmp_path)
        default_call = compute_pagerank(cg)
        explicit_false = compute_pagerank(cg, personalization_enabled=False)
        assert default_call == explicit_false

    def test_personalization_with_no_generic_edges_falls_back_to_uniform(
        self, tmp_path: Path
    ) -> None:
        """Opting in on a graph with zero generic_edges (no artifact nodes
        at all) must not error or change scores."""
        _, cg, _ = _build_star_graph(tmp_path)
        without = compute_pagerank(cg)
        with_ppr = compute_pagerank(cg, personalization_enabled=True)
        assert without == with_ppr

    def test_personalization_shifts_score_toward_ticket_linked_leaf(self, tmp_path: Path) -> None:
        """Proves personalization= is genuinely wired into compute_pagerank's
        nx.pagerank call, not a no-op — a prior version of this test passed
        identically whether personalization was hardcoded to None or
        threaded through, because the star-graph fixture's hub already
        outranked every leaf regardless of any ticket link (leaves have no
        way to beat the hub on call-graph structure alone in that
        topology), so the assertions never actually depended on
        personalization doing anything.

        This fixture instead builds a hub called by 8 distinct callers
        (real call-graph centrality, no ticket link) versus a target with
        zero callers but a ticket link — with enough callers, plain
        PageRank ranks hub above target on pure centrality. Personalization
        must be strong enough to flip that ordering; anything that would
        pass with personalization=None does not actually exercise the
        personalization= kwarg.
        """
        db = Database(tmp_path / "index.db")
        fid = db.upsert_file(
            IndexedFile(
                path="/r/a.py",
                rel_path="a.py",
                language=Language.PYTHON,
                hash="x",
                size_bytes=10,
            )
        )
        hub = db.insert_symbol(
            Symbol(
                file_id=fid,
                name="hub",
                qualified_name="hub",
                kind=SymbolKind.FUNCTION,
                line_start=1,
                line_end=5,
                signature="def hub()",
                body="",
            )
        )
        target = db.insert_symbol(
            Symbol(
                file_id=fid,
                name="target",
                qualified_name="target",
                kind=SymbolKind.FUNCTION,
                line_start=10,
                line_end=14,
                signature="def target()",
                body="",
            )
        )
        callers = [
            db.insert_symbol(
                Symbol(
                    file_id=fid,
                    name=f"caller{i}",
                    qualified_name=f"caller{i}",
                    kind=SymbolKind.FUNCTION,
                    line_start=20 + i,
                    line_end=20 + i,
                    signature=f"def caller{i}()",
                    body="",
                )
            )
            for i in range(8)
        ]
        db.insert_call_edges(
            [
                CallEdge(caller_id=caller, callee_name="hub", callee_id=hub, line=1)
                for caller in callers
            ]
        )
        db.insert_generic_edges(
            [
                GenericEdge(
                    from_symbol_id=target,
                    source_ref="ticket:PROJ-1",
                    edge_kind="references_ticket",
                )
            ]
        )
        cg = CodeGraph(db)

        without = compute_pagerank(cg)
        # Sanity check the fixture's premise: on pure call-graph structure,
        # the heavily-called hub outranks the zero-callers ticket target.
        assert without[hub] > without[target]

        with_ppr = compute_pagerank(cg, personalization_enabled=True)
        # Personalization must be strong enough to flip that ordering — a
        # no-op personalization would leave hub > target unchanged, since
        # that's exactly what plain PageRank already produces from
        # call-graph structure alone.
        assert with_ppr[target] > with_ppr[hub]


class TestGetTopCentralSymbols:
    def test_returns_sorted_by_centrality(self, tmp_path: Path) -> None:
        db, cg, hub_id = _build_star_graph(tmp_path)
        scores = compute_pagerank(cg)
        # Assign centrality scores to graph nodes
        for node_id, score in scores.items():
            cg.nx.nodes[node_id]["centrality"] = score
        save_graph_metadata(db, cg)
        top = get_top_central_symbols(db, top_n=1)
        assert len(top) == 1
        assert top[0] == hub_id
