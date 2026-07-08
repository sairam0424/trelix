"""
Incremental graph updater — rebuilds CodeGraph metadata for a changed file
using DF Louvain incremental community detection.

Maintains _prev_partition (the last known community assignment) so that only
the affected-vertex frontier is reprocessed on each file-change event rather
than the full graph. Falls back to full Louvain when no prior state exists or
when the frontier exceeds 50% of nodes.

Reference: DF Louvain (arXiv:2404.19634) — 179x speedup over static Louvain.
"""

from __future__ import annotations

import logging

from trelix.graph.code_graph import CodeGraph
from trelix.graph.community import (
    assign_communities,
    compute_pagerank,
    detect_communities_incremental,
)
from trelix.graph.persistence import save_graph_metadata
from trelix.store.db import Database

logger = logging.getLogger("trelix.graph.updater")


class GraphUpdater:
    """Lightweight incremental updater for the CodeGraph after file changes."""

    def __init__(self, db: Database) -> None:
        self._db = db
        # Stores the last known {node_id: community_id} partition.
        # Empty on first run — triggers full Louvain; populated on every
        # successful update so subsequent changes use the fast incremental path.
        self._prev_partition: dict[int, int] = {}

    def update_file(self, rel_path: str) -> None:
        """
        Rebuild graph nodes/edges for `rel_path` and refresh graph_metadata.

        Uses incremental Louvain (DF Louvain frontier heuristic) when a prior
        partition exists, falling back to full Louvain on first run or when
        the frontier is too large.

        Safe to call even if the file isn't indexed — no-op in that case.
        Never raises — all failures are logged as warnings.
        """
        try:
            cg = CodeGraph(self._db)
            if cg.node_count == 0:
                return

            # Identify which symbol IDs belong to the changed file
            # (used as the initial frontier for DF Louvain)
            try:
                seed_symbol_ids = set(self._db.get_symbol_ids_for_file(rel_path))
            except Exception:
                seed_symbol_ids = set()

            # Run incremental Louvain
            communities = detect_communities_incremental(
                cg,
                seed_nodes=seed_symbol_ids,
                prev_partition=self._prev_partition,
            )

            # Update in-memory graph with new community assignments
            assign_communities(cg, communities)

            # Update PageRank scores
            pr_scores = compute_pagerank(cg)
            for node_id, score in pr_scores.items():
                if node_id in cg.nx.nodes:
                    cg.nx.nodes[node_id]["centrality"] = score

            save_graph_metadata(self._db, cg)

            # Persist partition for next incremental update
            self._prev_partition = communities

            logger.debug(
                "Graph metadata refreshed (incremental) after change to %s "
                "(frontier from %d seed symbols, total %d nodes)",
                rel_path, len(seed_symbol_ids), cg.node_count,
            )
        except Exception as exc:
            logger.warning("GraphUpdater.update_file(%r) failed (non-fatal): %s", rel_path, exc)
