"""Persist CodeGraph metadata (community, centrality) back to SQLite."""

from __future__ import annotations

import logging

import networkx as nx

from trelix.graph.code_graph import CodeGraph
from trelix.store.db import Database

logger = logging.getLogger("trelix.graph.persistence")

_DDL = """
CREATE TABLE IF NOT EXISTS graph_metadata (
    symbol_id INTEGER PRIMARY KEY,
    community INTEGER,
    centrality REAL DEFAULT 0.0,
    node_type TEXT DEFAULT 'symbol'
);
"""


def _ensure_table(db: Database) -> None:
    db._conn.execute(_DDL)
    db._conn.commit()


def save_graph_metadata(db: Database, cg: CodeGraph) -> None:
    """Write community and centrality for all nodes into graph_metadata table.

    Centrality is read from node attrs (``cg.nx.nodes[id]["centrality"]``) when
    available — this allows the caller to pre-compute PageRank and store it by
    setting the node attribute before calling this function.  Falls back to
    ``nx.degree_centrality`` when the attr is absent.
    """
    _ensure_table(db)

    # Fall back to degree_centrality for any node that has no "centrality" attr.
    degree_centrality: dict[int, float] = {}
    try:
        degree_centrality = nx.degree_centrality(cg.nx)
    except Exception:
        pass

    rows = [
        (
            node_id,
            attrs.get("community"),
            attrs.get("centrality", degree_centrality.get(node_id, 0.0)),
            attrs.get("type", "symbol"),
        )
        for node_id, attrs in cg.nx.nodes(data=True)
    ]
    db._conn.executemany(
        """
        INSERT INTO graph_metadata (symbol_id, community, centrality, node_type)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(symbol_id) DO UPDATE SET
            community = excluded.community,
            centrality = excluded.centrality,
            node_type = excluded.node_type
        """,
        rows,
    )
    db._conn.commit()
    logger.debug("Saved graph metadata for %d nodes", len(rows))


def load_graph_metadata(db: Database, cg: CodeGraph) -> None:
    """Read community and centrality from graph_metadata and set node attrs."""
    _ensure_table(db)
    rows = db._conn.execute(
        "SELECT symbol_id, community, centrality FROM graph_metadata"
    ).fetchall()
    for symbol_id, community, centrality in rows:
        if symbol_id in cg.nx:
            cg.nx.nodes[symbol_id]["community"] = community
            cg.nx.nodes[symbol_id]["centrality"] = centrality


def get_top_central_symbols(db: Database, top_n: int = 100) -> list[int]:
    """Return symbol_ids sorted by centrality DESC from graph_metadata table."""
    rows = db._conn.execute(
        "SELECT symbol_id FROM graph_metadata ORDER BY centrality DESC LIMIT ?",
        (top_n,),
    ).fetchall()
    return [int(row[0]) for row in rows]
