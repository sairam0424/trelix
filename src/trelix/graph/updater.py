"""
Incremental graph updater — rebuilds the CodeGraph sub-graph for a single file
after that file has been re-indexed by the watcher.

Only rebuilds nodes/edges touching the changed file, then re-saves graph_metadata
(community + centrality) for those nodes. This avoids rebuilding the full graph
on every file-save event.
"""
from __future__ import annotations

import logging

from trelix.store.db import Database

logger = logging.getLogger("trelix.graph.updater")


class GraphUpdater:
    """Lightweight incremental updater for the CodeGraph after file changes."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def update_file(self, rel_path: str) -> None:
        """
        Rebuild graph nodes/edges for `rel_path` and refresh graph_metadata.

        Safe to call even if the file isn't indexed — no-op in that case.
        Rebuilds the full CodeGraph (fast: NetworkX over SQLite reads, ~50ms
        for typical repos) and re-saves metadata. Full rebuild is simpler than
        partial updates and avoids stale-edge bugs when call targets change.
        """
        try:
            from trelix.graph.code_graph import CodeGraph
            from trelix.graph.community import (
                assign_communities,
                compute_pagerank,
                detect_communities,
            )
            from trelix.graph.persistence import save_graph_metadata

            cg = CodeGraph(self._db)
            if cg.node_count == 0:
                return

            # Re-run community detection and PageRank
            communities = detect_communities(cg)
            assign_communities(cg, communities)
            pr_scores = compute_pagerank(cg)
            for node_id, score in pr_scores.items():
                if node_id in cg.nx.nodes:
                    cg.nx.nodes[node_id]["centrality"] = score

            save_graph_metadata(self._db, cg)
            logger.debug("Graph metadata refreshed after change to %s", rel_path)
        except Exception as exc:
            logger.warning("GraphUpdater.update_file(%r) failed (non-fatal): %s", rel_path, exc)
