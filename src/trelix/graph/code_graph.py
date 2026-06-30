"""Unified Code Property Graph over trelix's SQLite edge tables."""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
from networkx import MultiDiGraph

from trelix.store.db import Database

logger = logging.getLogger("trelix.graph.code_graph")

_EDGE_KINDS_TO_LABEL: dict[str, str] = {
    "extends": "EXTENDS",
    "implements": "IMPLEMENTS",
    "trait_impl": "TRAIT_IMPL",
    "embedded": "EMBEDDED",
    "angular_selector": "ANGULAR_SELECTOR",
}


class CodeGraph:
    """
    Unified MultiDiGraph over trelix's SQLite edge tables.

    Nodes  = symbol IDs (int), with attrs: name, qualified_name, kind, file, language, community
    Edges  = directed, labeled: CALLS | IMPORTS | EXTENDS | IMPLEMENTS | TRAIT_IMPL | EMBEDDED
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._g: MultiDiGraph = MultiDiGraph()
        self._build()

    def _build(self) -> None:
        """Load all symbols as nodes and all edges from the DB."""
        # --- Nodes: all symbols ---
        for symbol, file in self._db.iter_all_symbols_with_files():
            self._g.add_node(
                symbol.id,
                type="symbol",
                name=symbol.name,
                qualified_name=symbol.qualified_name,
                kind=symbol.kind.value,
                file=file.rel_path,
                language=file.language.value,
                community=None,
            )

        # --- CALLS edges ---
        for caller_id, callee_id in self._db.iter_resolved_calls():
            if caller_id in self._g and callee_id in self._g:
                self._g.add_edge(caller_id, callee_id, label="CALLS")

        # --- IMPORTS edges (file-level → map to representative file-module node) ---
        for file_id, imported_file_id in self._db.iter_resolved_imports():
            # Add file nodes if not present as symbols (files may have no symbols)
            for nid in (file_id, imported_file_id):
                if nid not in self._g:
                    fi = self._db.get_file_by_id(nid)
                    if fi is not None:
                        self._g.add_node(
                            nid,
                            type="file",
                            name=fi.rel_path,
                            qualified_name=fi.rel_path,
                            kind="file",
                            file=fi.rel_path,
                            language=fi.language.value,
                            community=None,
                        )
            if file_id in self._g and imported_file_id in self._g:
                self._g.add_edge(file_id, imported_file_id, label="IMPORTS")

        # --- TYPE edges (EXTENDS / IMPLEMENTS / TRAIT_IMPL / EMBEDDED) ---
        for from_id, edge_kind, to_id in self._db.iter_resolved_type_edges():
            if edge_kind not in _EDGE_KINDS_TO_LABEL:
                logger.debug("Unknown edge_kind %r, using TYPE_REL fallback", edge_kind)
            label = _EDGE_KINDS_TO_LABEL.get(edge_kind, "TYPE_REL")
            if from_id in self._g and to_id in self._g:
                self._g.add_edge(from_id, to_id, label=label)

        logger.debug(
            "CodeGraph built: %d nodes, %d edges",
            self._g.number_of_nodes(),
            self._g.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def nx(self) -> MultiDiGraph:
        return self._g

    @property
    def node_count(self) -> int:
        return int(self._g.number_of_nodes())

    @property
    def edge_count(self) -> int:
        return int(self._g.number_of_edges())

    def neighbors(self, symbol_id: int) -> list[int]:
        """Return all adjacent node IDs (successors + predecessors)."""
        if symbol_id not in self._g:
            return []
        succs = list(self._g.successors(symbol_id))
        preds = list(self._g.predecessors(symbol_id))
        return list(dict.fromkeys(succs + preds))

    def shortest_path(self, src: int, dst: int) -> list[int] | None:
        """Return node-ID list of shortest undirected path, or None."""
        try:
            return list(nx.shortest_path(self._g.to_undirected(), src, dst))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def subgraph(self, symbol_ids: list[int]) -> MultiDiGraph:
        """Return induced subgraph over the given node IDs."""
        return self._g.subgraph(symbol_ids).copy()

    def get_node_attrs(self, symbol_id: int) -> dict[str, Any]:
        """Return node attribute dict, or empty dict if not found."""
        return dict(self._g.nodes.get(symbol_id, {}))
