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
    """Write community and centrality for all nodes into graph_metadata table."""
    _ensure_table(db)

    centrality: dict[int, float] = {}
    try:
        centrality = nx.degree_centrality(cg.nx)
    except Exception:
        pass

    rows = [
        (
            node_id,
            attrs.get("community"),
            centrality.get(node_id, 0.0),
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
