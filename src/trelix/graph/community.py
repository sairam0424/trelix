"""Community detection for CodeGraph — Louvain (fast) or Girvan-Newman (quality)."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from trelix.graph.code_graph import CodeGraph

logger = logging.getLogger("trelix.graph.community")

# Above this share of one-node communities the partition carries almost no
# grouping information and callers should be told so. Measured on trelix's own
# index at 99.1% (6579 singletons of 6640 communities) — see
# assess_partition_quality() for why that is an edge-coverage fact, not a
# Louvain tuning fact.
_DEGENERATE_SINGLETON_SHARE = 0.5


def _simple_undirected(cg: CodeGraph) -> nx.Graph:
    """
    Collapse the MultiDiGraph into the simple undirected Graph the community
    algorithms actually run on: parallel edges dropped, direction dropped,
    isolated nodes preserved so every node is covered by the partition.

    Shared by detect_communities(), detect_communities_incremental() and
    assess_partition_quality() so the quality numbers describe the same graph
    the partition was computed from.
    """
    g_undirected = cg.nx.to_undirected()
    simple = nx.Graph((u, v) for u, v, _ in g_undirected.edges(data=True))
    for node in g_undirected.nodes():
        if node not in simple:
            simple.add_node(node)
    return simple


def detect_communities(
    cg: CodeGraph,
    algorithm: str = "louvain",
) -> dict[int, int]:
    """
    Detect communities and return {node_id: community_id}.

    algorithm:
        "louvain"       — fast, good quality, O(n log n). Preferred for >500 nodes.
        "girvan_newman" — betweenness-based, high quality, O(n³). Use for small graphs.
        "label_prop"    — very fast, approximate. Use for >10k nodes.
    """
    if cg.node_count == 0:
        return {}

    G_connected = _simple_undirected(cg)

    mapping: dict[int, int] = {}

    try:
        if algorithm == "louvain":
            communities = nx.community.louvain_communities(G_connected, seed=42)
        elif algorithm == "girvan_newman":
            gen = nx.community.girvan_newman(G_connected)
            last_communities = None
            try:
                for _ in range(3):
                    communities_tuple = next(gen)
                    last_communities = [set(c) for c in communities_tuple]
            except StopIteration:
                pass
            communities = (
                last_communities if last_communities is not None else [set(G_connected.nodes())]
            )
        elif algorithm == "label_prop":
            communities = list(nx.community.label_propagation_communities(G_connected))
        else:
            raise ValueError(f"Unknown algorithm: {algorithm!r}")

        for community_id, members in enumerate(communities):
            for node_id in members:
                mapping[int(node_id)] = community_id

    except Exception as exc:
        logger.warning("Community detection failed (%s), assigning all to 0: %s", algorithm, exc)
        for node_id in cg.nx.nodes():
            mapping[int(node_id)] = 0

    return mapping


def assign_communities(cg: CodeGraph, communities: dict[int, int]) -> None:
    """Write community IDs back into CodeGraph node attributes."""
    for node_id, community_id in communities.items():
        if node_id in cg.nx:
            cg.nx.nodes[node_id]["community"] = community_id


@dataclass(frozen=True)
class PartitionQuality:
    """
    Measured shape of a community partition, and how much of that shape the
    graph's edge set forced.

    Exists because `community_count` alone is actively misleading. On trelix's
    own index the graph command reported "Communities: 6640" for 10991 nodes and
    said nothing else; 6579 of those 6640 were one-node communities. See
    assess_partition_quality() for the attribution.
    """

    node_count: int
    edge_count: int
    isolated_nodes: int
    community_count: int
    singleton_communities: int
    isolated_singletons: int
    modularity: float
    largest_sizes: list[int] = field(default_factory=list)

    @property
    def singleton_share(self) -> float:
        """Fraction of communities that contain exactly one node."""
        if self.community_count == 0:
            return 0.0
        return self.singleton_communities / self.community_count

    @property
    def isolated_share(self) -> float:
        """Fraction of nodes with no edge of any kind."""
        if self.node_count == 0:
            return 0.0
        return self.isolated_nodes / self.node_count

    @property
    def is_degenerate(self) -> bool:
        """True when singletons dominate, i.e. the labels group almost nothing."""
        return self.singleton_share > _DEGENERATE_SINGLETON_SHARE

    @property
    def is_edge_limited(self) -> bool:
        """
        True when the singletons are explained by isolated nodes rather than by
        the algorithm. A degree-0 node contributes zero to modularity in every
        possible community, so Louvain leaves it alone at ANY resolution — no
        parameter value can merge it. When this is True, tuning is the wrong
        lever and edge extraction is the right one.
        """
        if self.singleton_communities == 0:
            return False
        return self.isolated_singletons / self.singleton_communities > 0.9

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable form for reports and API payloads."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "isolated_nodes": self.isolated_nodes,
            "isolated_share": round(self.isolated_share, 4),
            "community_count": self.community_count,
            "singleton_communities": self.singleton_communities,
            "singleton_share": round(self.singleton_share, 4),
            "isolated_singletons": self.isolated_singletons,
            "modularity": round(self.modularity, 4),
            "largest_sizes": list(self.largest_sizes),
            "degenerate": self.is_degenerate,
            "edge_limited": self.is_edge_limited,
        }

    def describe(self) -> str:
        """One-paragraph human summary naming the number and the consequence."""
        connected = self.node_count - self.isolated_nodes
        parts = [
            f"{self.singleton_communities}/{self.community_count} communities "
            f"({self.singleton_share:.1%}) contain a single node; "
            f"{self.isolated_nodes}/{self.node_count} nodes ({self.isolated_share:.1%}) "
            f"have no edge at all.",
        ]
        if self.is_edge_limited:
            parts.append(
                f"{self.isolated_singletons} of the singletons are those isolated nodes, "
                "so this is edge coverage, not Louvain tuning — a degree-0 node cannot be "
                "merged at any resolution. Community labels only carry information for the "
                f"{connected} connected nodes."
            )
        parts.append(
            f"Modularity {self.modularity:.3f} — note this stays high because isolated "
            "singletons contribute exactly zero to it, so it does not detect this."
        )
        return " ".join(parts)


def assess_partition_quality(
    cg: CodeGraph,
    communities: dict[int, int],
) -> PartitionQuality:
    """
    Measure the partition returned by `detect_communities` against the graph it
    came from, so a degenerate result is reportable instead of silent.

    Measured on trelix's own self-index (10991 nodes / 9483 simple edges):
    6579 of 6640 communities were singletons (99.1%), and 6576 of those 6579
    were nodes with degree 0 (59.8% of the graph). A resolution sweep over
    0.2/0.5/1.0/2.0/5.0 returned the SAME 6579 singletons every time — only the
    giant clusters resized. That is why this reports isolated-node counts rather
    than offering a tuning knob: the partition is limited by edge extraction
    (50.8% of call edges are unresolved by design for stdlib/external targets,
    and markdown/JSON/YAML/TOML symbols cannot have call or import edges at
    all), not by the algorithm.

    Args:
        cg:          CodeGraph the partition was computed over
        communities: {node_id: community_id} from detect_communities()

    Returns:
        PartitionQuality — all-zero when the graph or partition is empty
    """
    if cg.node_count == 0 or not communities:
        return PartitionQuality(
            node_count=cg.node_count,
            edge_count=0,
            isolated_nodes=0,
            community_count=0,
            singleton_communities=0,
            isolated_singletons=0,
            modularity=0.0,
        )

    G = _simple_undirected(cg)
    degrees = dict(G.degree())
    isolated = {node for node, deg in degrees.items() if deg == 0}

    members: dict[int, list[int]] = defaultdict(list)
    for node_id, community_id in communities.items():
        members[community_id].append(node_id)

    singletons = [group[0] for group in members.values() if len(group) == 1]
    sizes = sorted((len(group) for group in members.values()), reverse=True)

    # Modularity of the partition as produced. Nodes the partition does not
    # mention would raise NotAPartition, so only covered nodes are passed and a
    # coverage gap is reported rather than crashing the graph build.
    covered = [set(group) & set(G.nodes()) for group in members.values()]
    covered = [group for group in covered if group]
    modularity = 0.0
    if G.number_of_edges() > 0 and covered:
        assigned = set().union(*covered)
        leftover = set(G.nodes()) - assigned
        if leftover:
            logger.warning(
                "Community partition covers %d/%d nodes — %d unassigned, "
                "excluded from the modularity figure",
                len(assigned),
                G.number_of_nodes(),
                len(leftover),
            )
        try:
            modularity = float(nx.community.modularity(G.subgraph(assigned), covered))
        except Exception as exc:  # ZeroDivisionError on an edgeless subgraph, etc.
            logger.warning("Modularity computation failed, reporting 0.0: %s", exc)

    return PartitionQuality(
        node_count=G.number_of_nodes(),
        edge_count=G.number_of_edges(),
        isolated_nodes=len(isolated),
        community_count=len(members),
        singleton_communities=len(singletons),
        isolated_singletons=sum(1 for node in singletons if node in isolated),
        modularity=modularity,
        largest_sizes=sizes[:5],
    )


def compute_affected_frontier(
    G: nx.Graph,
    seed_nodes: set[int],
    partition: dict[int, int],
) -> set[int]:
    """
    Compute the DF Louvain approximate affected-vertex frontier.

    Returns the set of nodes that should be re-evaluated in an incremental
    Louvain pass after a batch of graph changes touching `seed_nodes`.

    Frontier = seed_nodes
               + all neighbors of seed_nodes (in G)
               + all nodes sharing a community with any seed_node (from partition)

    This is the DF Louvain heuristic from arXiv:2404.19634 — approximate
    (may miss some affected vertices) but 179x faster than full re-run.
    The trade-off is acceptable for trelix's community labels which are
    search-quality metadata, not security decisions.

    Args:
        G:          The undirected graph (same G_connected used in detect_communities)
        seed_nodes: Nodes directly affected by the file change
        partition:  Previous community assignment {node_id: community_id}

    Returns:
        set[int] of node IDs to include in the next Louvain pass
    """
    if not seed_nodes:
        return set()

    frontier: set[int] = set(seed_nodes)

    # Add direct neighbors of all seed nodes
    for node in seed_nodes:
        if G.has_node(node):
            frontier.update(G.neighbors(node))

    # Add all nodes in the same community as any seed node
    if partition:
        seed_communities = {partition[n] for n in seed_nodes if n in partition}
        for node_id, community_id in partition.items():
            if community_id in seed_communities:
                frontier.add(node_id)

    return frontier


def detect_communities_incremental(
    cg: CodeGraph,
    seed_nodes: set[int],
    prev_partition: dict[int, int],
    frontier_threshold: float = 0.5,
) -> dict[int, int]:
    """
    Incremental Louvain community detection using the DF Louvain frontier heuristic.

    Only re-runs Louvain on the affected frontier (seed nodes + neighbors +
    same-community nodes). Non-frontier nodes inherit their previous community
    assignment from `prev_partition`. Falls back to full `detect_communities()`
    when frontier exceeds `frontier_threshold` fraction of all nodes or when
    `prev_partition` is empty.

    Args:
        cg:                 CodeGraph to update
        seed_nodes:         Nodes directly touched by the file change (symbol IDs)
        prev_partition:     Previous {node_id: community_id} assignment
        frontier_threshold: If frontier / total_nodes > this, fall back to full Louvain

    Returns:
        dict[int, int] — full updated {node_id: community_id} for all nodes
    """
    if cg.node_count == 0:
        return {}

    # Fall back to full detection when no prior state
    if not prev_partition:
        return detect_communities(cg)

    G_connected = _simple_undirected(cg)

    total_nodes = G_connected.number_of_nodes()
    if total_nodes == 0:
        return {}

    # Compute the approximate DF Louvain frontier
    frontier = compute_affected_frontier(G_connected, seed_nodes, prev_partition)

    # Fall back to full detection when frontier is too large (cost threshold)
    if total_nodes > 0 and len(frontier) / total_nodes > frontier_threshold:
        logger.debug(
            "Incremental Louvain frontier (%d/%d nodes) exceeds threshold %.0f%% — "
            "falling back to full Louvain",
            len(frontier),
            total_nodes,
            frontier_threshold * 100,
        )
        return detect_communities(cg)

    # Run Louvain only on the frontier subgraph
    frontier_subgraph = G_connected.subgraph(frontier).copy()
    new_partition: dict[int, int] = dict(prev_partition)  # start from previous state

    if frontier_subgraph.number_of_nodes() > 0:
        try:
            sub_communities = nx.community.louvain_communities(frontier_subgraph, seed=42)
            # Map sub-community IDs to globally unique IDs
            # Use large offset to avoid colliding with existing community IDs
            max_existing = max(prev_partition.values(), default=-1)
            for sub_id, members in enumerate(sub_communities):
                global_id = max_existing + 1 + sub_id
                for node_id in members:
                    new_partition[int(node_id)] = global_id
        except Exception as exc:
            logger.warning("Incremental Louvain on frontier failed — falling back to full: %s", exc)
            return detect_communities(cg)

    # Ensure all nodes are covered (add any missing nodes from prev_partition)
    for node_id in G_connected.nodes():
        if int(node_id) not in new_partition:
            new_partition[int(node_id)] = prev_partition.get(int(node_id), 0)

    return new_partition


def compute_pagerank(
    cg: CodeGraph,
    alpha: float = 0.85,
    personalization_enabled: bool = False,
) -> dict[int, float]:
    """
    Compute PageRank over the code graph. Returns node_id → normalized score.

    High-PageRank nodes are called/imported by many others — architecturally central.
    Scores are normalized to [0, 1] by dividing by the max score.

    Args:
        cg: CodeGraph instance (networkx MultiDiGraph under the hood)
        alpha: damping factor (default 0.85, standard PageRank value)
        personalization_enabled: when True (default False — zero behavior
            change unless a caller opts in via
            RetrievalConfig.pagerank_personalization_enabled), replaces the
            uniform teleport vector with a Personalized PageRank vector
            concentrated on symbol nodes with a cross-source generic_edge
            (i.e. every symbol adjacent to a `type="artifact"` node — see
            CodeGraph._build()'s bidirectional GENERIC edge loop). Same
            construction as retrieval/graph.py's rank_by_pagerank(), applied
            here to the full persistent graph rather than a per-query
            subgraph.

    Returns:
        dict[int, float] — empty dict if graph has no edges
    """
    g = cg.nx
    if g.number_of_nodes() == 0:
        return {}

    personalization = None
    if personalization_enabled:
        artifact_nodes = [n for n, attrs in g.nodes(data=True) if attrs.get("type") == "artifact"]
        cross_source_nodes: set[int] = set()
        for artifact_node in artifact_nodes:
            for neighbor in g.successors(artifact_node):
                if isinstance(neighbor, int):
                    cross_source_nodes.add(neighbor)
            for neighbor in g.predecessors(artifact_node):
                if isinstance(neighbor, int):
                    cross_source_nodes.add(neighbor)
        if cross_source_nodes:
            mass = 1.0 / len(cross_source_nodes)
            personalization = {node: mass for node in cross_source_nodes}

    try:
        raw: dict[int, float] = nx.pagerank(
            g, alpha=alpha, max_iter=100, personalization=personalization
        )
    except nx.PowerIterationFailedConvergence:
        raw = nx.pagerank(g, alpha=alpha, max_iter=500, tol=1e-4, personalization=personalization)

    # Normalize to [0, 1]
    max_score = max(raw.values()) if raw else 1.0
    if max_score == 0.0:
        return {k: 0.0 for k in raw}
    return {k: v / max_score for k, v in raw.items()}


def get_community_summary(cg: CodeGraph) -> list[dict[str, Any]]:
    """
    Return summary info per detected community, largest first.

    Ordered by size because the only consumer that trusts this order is the CLI,
    which prints the first five under the heading "Top Communities". Sorted by
    community_id it printed the five LOWEST IDs — measured on this repo as
    "2, 2, 279, 13, 3 nodes" while the actual largest were 445/382/309/281/279,
    so the giant clusters the heading promises were invisible. The API's
    /graph/communities and the MCP build_knowledge_graph tool both re-sort by
    size themselves, so this is a no-op for them. Ties break on community_id to
    keep the output stable across runs.
    """
    if cg.node_count == 0:
        return []

    by_community: dict[int, list[int]] = defaultdict(list)
    for node_id, attrs in cg.nx.nodes(data=True):
        cid = attrs.get("community")
        if cid is not None:
            by_community[int(cid)].append(node_id)

    if not by_community:
        return []

    summaries = []
    for cid, members in sorted(by_community.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        # Top files by member count
        file_counts: Counter[str] = Counter()
        symbol_names: list[str] = []
        for mid in members:
            attrs = cg.nx.nodes.get(mid, {})
            f = attrs.get("file", "")
            if f:
                file_counts[f] += 1
            name = attrs.get("qualified_name") or attrs.get("name", "")
            if name:
                symbol_names.append(name)

        summaries.append(
            {
                "community_id": cid,
                "size": len(members),
                "top_files": [f for f, _ in file_counts.most_common(5)],
                "top_symbols": symbol_names[:10],
                "label": f"community_{cid}",
            }
        )

    return summaries
