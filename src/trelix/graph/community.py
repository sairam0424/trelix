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


def compute_pagerank(cg: "CodeGraph", alpha: float = 0.85) -> dict[int, float]:
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
