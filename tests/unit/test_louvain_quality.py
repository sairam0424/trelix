"""
Louvain partition-quality diagnostics.

Background — all figures measured on trelix's own self-index at 01d148d
(.trelix/index.db, 10991 symbols / 467 files):

    simple undirected graph   10991 nodes, 9483 edges, mean degree 1.73
    isolated (degree 0)        6576 nodes = 59.83%
    degree <= 1                7921 nodes = 72.07%
    louvain(resolution=1.0)    6640 communities, 6579 singletons = 99.08%
    of those singletons        6576 are isolated nodes (99.95%)
    modularity                 0.7887   <-- looks healthy, is not informative
    top 5 community sizes      445, 382, 309, 281, 279
    resolution sweep           singletons = 6579 at res 0.2/0.5/1.0/2.0/5.0

The last line is the finding: resolution has NO effect on the singleton count,
because a degree-0 node contributes zero to modularity in every possible
community and Louvain therefore never moves it. On the 4415-node connected
subgraph the same algorithm yields 66 communities with 3 singletons (4.5%) at
Q=0.777 — a healthy partition. So the degeneracy is edge coverage, not tuning:
10320 of 20313 call edges are unresolved (stdlib/external, by design), only 130
type edges exist, and 3797 of the isolated nodes are markdown/JSON/YAML/TOML
symbols that cannot have call or import edges at all.

These tests lock in the diagnostic that reports it, and the resolution
invariance that rules out the tuning explanation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx
import pytest

from trelix.core.config import IndexConfig
from trelix.core.models import CallEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.builder import GraphBuilder
from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import (
    PartitionQuality,
    assess_partition_quality,
    detect_communities,
)
from trelix.graph.visualizer import GraphVisualizer
from trelix.store.db import Database


def _sparse_repo(tmp_path: Path, connected_pairs: int = 3, isolated: int = 30) -> Path:
    """
    An indexed repo with the measured shape of the real graph: a few small
    connected clusters plus a majority of edgeless symbols.

    Defaults give 36 nodes of which 30 (83%) are isolated — the real index is
    59.8% isolated, so this is the same failure, exaggerated to stay small.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".trelix").mkdir()
    db = Database(repo / ".trelix" / "index.db")

    fid = db.upsert_file(
        IndexedFile(
            path=str(repo / "mod.py"),
            rel_path="mod.py",
            language=Language.PYTHON,
            hash="x",
            size_bytes=100,
        )
    )

    def _sym(name: str) -> int:
        return db.insert_symbol(
            Symbol(
                file_id=fid,
                name=name,
                qualified_name=name,
                kind=SymbolKind.FUNCTION,
                line_start=1,
                line_end=2,
                signature=f"def {name}()",
                body="pass",
            )
        )

    edges = []
    for i in range(connected_pairs):
        caller = _sym(f"caller_{i}")
        callee = _sym(f"callee_{i}")
        edges.append(
            CallEdge(caller_id=caller, callee_name=f"callee_{i}", callee_id=callee, line=1)
        )
    db.insert_call_edges(edges)

    # The majority: symbols with no resolved call edge in either direction.
    for i in range(isolated):
        _sym(f"orphan_{i}")

    db._conn.commit()
    db.close()
    return repo


def _sparse_graph(tmp_path: Path, **kwargs: int) -> CodeGraph:
    repo = _sparse_repo(tmp_path, **kwargs)
    return CodeGraph(Database(repo / ".trelix" / "index.db"))


class TestAssessPartitionQuality:
    def test_reports_isolated_node_count(self, tmp_path: Path) -> None:
        cg = _sparse_graph(tmp_path, connected_pairs=3, isolated=30)
        quality = assess_partition_quality(cg, detect_communities(cg))
        assert quality.node_count == 36
        assert quality.isolated_nodes == 30
        assert quality.isolated_share > 0.8

    def test_reports_singleton_share(self, tmp_path: Path) -> None:
        cg = _sparse_graph(tmp_path, connected_pairs=3, isolated=30)
        quality = assess_partition_quality(cg, detect_communities(cg))
        # 3 pairs + 30 orphans = 33 communities, 30 of them singletons.
        assert quality.community_count == 33
        assert quality.singleton_communities == 30
        assert quality.singleton_share > 0.9

    def test_degeneracy_is_attributed_to_isolated_nodes(self, tmp_path: Path) -> None:
        """The distinction that matters: broken algorithm vs. edgeless graph."""
        cg = _sparse_graph(tmp_path, connected_pairs=3, isolated=30)
        quality = assess_partition_quality(cg, detect_communities(cg))
        assert quality.is_degenerate
        assert quality.is_edge_limited
        assert quality.isolated_singletons == quality.singleton_communities

    def test_describe_names_the_number_and_the_lever(self, tmp_path: Path) -> None:
        cg = _sparse_graph(tmp_path, connected_pairs=3, isolated=30)
        text = assess_partition_quality(cg, detect_communities(cg)).describe()
        assert "30/33" in text
        assert "no edge at all" in text
        assert "not Louvain tuning" in text

    def test_healthy_partition_is_not_degenerate(self, tmp_path: Path) -> None:
        cg = _sparse_graph(tmp_path, connected_pairs=8, isolated=0)
        quality = assess_partition_quality(cg, detect_communities(cg))
        assert quality.isolated_nodes == 0
        assert quality.singleton_communities == 0
        assert not quality.is_degenerate
        assert not quality.is_edge_limited

    def test_empty_graph_returns_zeros(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        quality = assess_partition_quality(CodeGraph(db), {})
        assert quality == PartitionQuality(
            node_count=0,
            edge_count=0,
            isolated_nodes=0,
            community_count=0,
            singleton_communities=0,
            isolated_singletons=0,
            modularity=0.0,
        )
        assert not quality.is_degenerate

    def test_edgeless_graph_is_reported_as_fully_degenerate(self, tmp_path: Path) -> None:
        cg = _sparse_graph(tmp_path, connected_pairs=0, isolated=6)
        quality = assess_partition_quality(cg, detect_communities(cg))
        assert quality.singleton_share == 1.0
        assert quality.isolated_share == 1.0
        assert quality.modularity == 0.0
        assert quality.is_degenerate and quality.is_edge_limited

    def test_stale_node_id_in_partition_does_not_crash(self, tmp_path: Path) -> None:
        """
        detect_communities_incremental() carries prev_partition forward, so a
        symbol deleted since the last pass can still appear as a key. Modularity
        would raise on a node the graph no longer has.
        """
        cg = _sparse_graph(tmp_path, connected_pairs=3, isolated=5)
        partition = detect_communities(cg)
        partition[999_999] = 4242
        quality = assess_partition_quality(cg, partition)
        assert quality.node_count == 11
        assert quality.modularity > 0.0

    def test_partial_coverage_is_reported_not_silently_averaged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A partition missing nodes yields a figure over a subset — say so."""
        cg = _sparse_graph(tmp_path, connected_pairs=3, isolated=5)
        partition = detect_communities(cg)
        for node_id in list(partition)[:4]:
            del partition[node_id]
        with caplog.at_level(logging.WARNING, logger="trelix.graph.community"):
            assess_partition_quality(cg, partition)
        assert any("partition covers" in r.getMessage() for r in caplog.records)

    def test_as_dict_is_json_serializable(self, tmp_path: Path) -> None:
        cg = _sparse_graph(tmp_path, connected_pairs=3, isolated=30)
        payload = assess_partition_quality(cg, detect_communities(cg)).as_dict()
        assert json.loads(json.dumps(payload))["singleton_communities"] == 30
        assert payload["edge_limited"] is True


class TestModularityDoesNotDetectThis:
    def test_modularity_stays_high_while_partition_is_all_singletons(self, tmp_path: Path) -> None:
        """
        Why the degeneracy survived several releases: the standard quality metric
        reports a good score. Isolated singletons contribute exactly zero to
        modularity, so adding thousands of them cannot lower it. Measured 0.7887
        on the real index alongside 99.08% singletons.
        """
        cg = _sparse_graph(tmp_path, connected_pairs=3, isolated=30)
        quality = assess_partition_quality(cg, detect_communities(cg))
        assert quality.singleton_share > 0.9
        assert quality.modularity > 0.5, (
            "modularity is expected to look healthy here — that is the trap this "
            "diagnostic exists to cover, not a bug in the assessment"
        )


class TestResolutionIsNotTheLever:
    def test_singleton_count_is_invariant_across_resolutions(self, tmp_path: Path) -> None:
        """
        Locks in the sweep that rules out tuning. On the real index the singleton
        count was 6579 at every resolution from 0.2 to 5.0; only the giant
        clusters resized. Same invariance must hold on the fixture, otherwise the
        premise of the diagnostic (and of leaving resolution alone) is wrong.
        """
        cg = _sparse_graph(tmp_path, connected_pairs=6, isolated=30)
        g_undirected = cg.nx.to_undirected()
        simple = nx.Graph((u, v) for u, v, _ in g_undirected.edges(data=True))
        for node in g_undirected.nodes():
            if node not in simple:
                simple.add_node(node)

        counts = {}
        for resolution in (0.2, 0.5, 1.0, 2.0, 5.0):
            communities = nx.community.louvain_communities(simple, seed=42, resolution=resolution)
            counts[resolution] = sum(1 for c in communities if len(c) == 1)

        assert len(set(counts.values())) == 1, (
            f"singleton count varied with resolution: {counts} — if this ever "
            "becomes true, tuning IS a lever and the diagnostic's claim is stale"
        )
        assert next(iter(counts.values())) == 30


class TestGraphBuildSurfacesIt:
    def test_build_warns_at_default_cli_log_level(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        The visible half. `trelix graph` calls _setup_logging(False) -> WARNING,
        so an INFO-level note would stay hidden without -v. That is exactly how
        the degeneracy went unreported, so this asserts the level too.
        """
        repo = _sparse_repo(tmp_path, connected_pairs=3, isolated=30)
        config = IndexConfig(repo_path=str(repo))
        with caplog.at_level(logging.WARNING, logger="trelix.graph.builder"):
            result = GraphBuilder(config).build(extract_concepts=False)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "a degenerate partition must not build silently"
        message = warnings[0].getMessage()
        assert "Degenerate community partition" in message
        assert "30/33" in message
        assert result.community_count == 33

    def test_build_attaches_partition_quality(self, tmp_path: Path) -> None:
        repo = _sparse_repo(tmp_path, connected_pairs=3, isolated=30)
        result = GraphBuilder(IndexConfig(repo_path=str(repo))).build(extract_concepts=False)
        assert result.partition_quality is not None
        assert result.partition_quality.isolated_nodes == 30
        assert result.partition_quality.is_edge_limited

    def test_healthy_build_does_not_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        repo = _sparse_repo(tmp_path, connected_pairs=8, isolated=0)
        config = IndexConfig(repo_path=str(repo))
        with caplog.at_level(logging.WARNING, logger="trelix.graph.builder"):
            GraphBuilder(config).build(extract_concepts=False)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestCommunityReportCarriesIt:
    def test_report_includes_partition_quality(self, tmp_path: Path) -> None:
        repo = _sparse_repo(tmp_path, connected_pairs=3, isolated=30)
        result = GraphBuilder(IndexConfig(repo_path=str(repo))).build(extract_concepts=False)
        out = tmp_path / "report.json"
        GraphVisualizer().export_community_report(result, str(out))
        report = json.loads(out.read_text())
        # community_count alone reads as healthy; the shape has to travel with it.
        assert report["community_count"] == 33
        assert report["partition_quality"]["singleton_communities"] == 30
        assert report["partition_quality"]["isolated_nodes"] == 30
        assert report["partition_quality"]["edge_limited"] is True


class TestSummaryIsSizeOrdered:
    """
    The CLI prints community_summary[:5] under the heading "Top Communities" and
    does no sorting of its own. Sorted by community_id that heading was a lie:
    measured on the real index it showed 2, 2, 279, 13, 3 nodes while the actual
    largest communities were 445, 382, 309, 281, 279.
    """

    def test_largest_communities_come_first(self, tmp_path: Path) -> None:
        repo = _sparse_repo(tmp_path, connected_pairs=4, isolated=20)
        result = GraphBuilder(IndexConfig(repo_path=str(repo))).build(extract_concepts=False)
        sizes = [c["size"] for c in result.community_summary]
        assert sizes == sorted(sizes, reverse=True), (
            f"community_summary is not size-ordered: {sizes[:8]}"
        )
        # 4 connected pairs of 2, then the 20 orphans.
        assert sizes[:5] == [2, 2, 2, 2, 1]

    def test_ties_break_on_community_id_for_stable_output(self, tmp_path: Path) -> None:
        repo = _sparse_repo(tmp_path, connected_pairs=4, isolated=0)
        result = GraphBuilder(IndexConfig(repo_path=str(repo))).build(extract_concepts=False)
        ids = [c["community_id"] for c in result.community_summary]
        assert ids == sorted(ids), f"equal-sized communities are not id-ordered: {ids}"
