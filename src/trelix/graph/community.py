"""Community detection for CodeGraph — Louvain (fast) or Girvan-Newman (quality)."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import networkx as nx

from trelix.graph.code_graph import CodeGraph

logger = logging.getLogger("trelix.graph.community")


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

    # Work on undirected version for community detection
    G_undirected = cg.nx.to_undirected()

    # Build a simple Graph from edges (drops parallel edges for community algorithms)
    G_connected = nx.Graph((u, v) for u, v, _ in G_undirected.edges(data=True))
    # Re-add isolated nodes so every node is covered
    for node in G_undirected.nodes():
        if node not in G_connected:
            G_connected.add_node(node)

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

    G_undirected = cg.nx.to_undirected()
    G_connected = nx.Graph((u, v) for u, v, _ in G_undirected.edges(data=True))
    for node in G_undirected.nodes():
        if node not in G_connected:
            G_connected.add_node(node)

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


def compute_pagerank(cg: CodeGraph, alpha: float = 0.85) -> dict[int, float]:
    """
    Compute PageRank over the code graph. Returns node_id → normalized score.

    High-PageRank nodes are called/imported by many others — architecturally central.
    Scores are normalized to [0, 1] by dividing by the max score.

    Args:
        cg: CodeGraph instance (networkx MultiDiGraph under the hood)
        alpha: damping factor (default 0.85, standard PageRank value)

    Returns:
        dict[int, float] — empty dict if graph has no edges
    """
    g = cg.nx
    if g.number_of_nodes() == 0:
        return {}

    try:
        raw: dict[int, float] = nx.pagerank(g, alpha=alpha, max_iter=100)
    except nx.PowerIterationFailedConvergence:
        raw = nx.pagerank(g, alpha=alpha, max_iter=500, tol=1e-4)

    # Normalize to [0, 1]
    max_score = max(raw.values()) if raw else 1.0
    if max_score == 0.0:
        return {k: 0.0 for k in raw}
    return {k: v / max_score for k, v in raw.items()}


def get_community_summary(cg: CodeGraph) -> list[dict[str, Any]]:
    """Return summary info per detected community."""
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
    for cid, members in sorted(by_community.items()):
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
