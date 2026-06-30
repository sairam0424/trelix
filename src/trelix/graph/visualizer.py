"""Pyvis-based interactive visualization for CodeGraph."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from trelix.graph.builder import GraphBuildResult
from trelix.graph.code_graph import CodeGraph

logger = logging.getLogger("trelix.graph.visualizer")

# Community color palette (pastel fills)
_PALETTE = [
    "#a5d8ff",
    "#d0bfff",
    "#b2f2bb",
    "#ffd8a8",
    "#c3fae8",
    "#ffc9c9",
    "#ffe8cc",
    "#e5dbff",
    "#d3f9d8",
    "#fff3bf",
]

_EDGE_COLORS: dict[str, str] = {
    "CALLS": "#4a9eed",
    "IMPORTS": "#8b5cf6",
    "EXTENDS": "#22c55e",
    "IMPLEMENTS": "#06b6d4",
    "TRAIT_IMPL": "#f59e0b",
    "EMBEDDED": "#ef4444",
}


class GraphVisualizer:
    """Export CodeGraph to interactive Pyvis HTML or JSON community report."""

    def export_html(
        self,
        cg: CodeGraph,
        output_path: str,
        max_nodes: int = 500,
    ) -> str:
        """
        Generate an interactive Pyvis HTML visualization.

        Nodes are colored by community. Edges are colored by type.
        If the graph has more than max_nodes nodes, sample the highest-degree nodes.
        Returns the absolute path of the written file.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError(
                "pyvis is required for graph visualization. "
                "Install with: pip install 'trelix[graph-viz]'"
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Sample if too large
        g = cg.nx
        if g.number_of_nodes() > max_nodes:
            top_nodes = sorted(g.nodes(), key=lambda n: g.degree(n), reverse=True)[:max_nodes]
            g = cg.subgraph(top_nodes)

        net = Network(
            notebook=False,
            cdn_resources="remote",
            height="900px",
            width="100%",
            select_menu=True,
            filter_menu=False,
            bgcolor="#1a1a2e",
            font_color="#e0e0e0",
        )

        # Add nodes
        for node_id, attrs in g.nodes(data=True):
            community = attrs.get("community") or 0
            color = _PALETTE[int(community) % len(_PALETTE)]
            degree = g.degree(node_id)
            size = max(10, min(50, 10 + degree * 3))
            label = attrs.get("name", str(node_id))
            title = (
                f"<b>{attrs.get('qualified_name', label)}</b><br>"
                f"Kind: {attrs.get('kind', '?')}<br>"
                f"File: {attrs.get('file', '?')}<br>"
                f"Community: {community}"
            )
            net.add_node(
                node_id,
                label=label[:25],
                title=title,
                color=color,
                size=size,
            )

        # Add edges
        for src, dst, edge_attrs in g.edges(data=True):
            label = edge_attrs.get("label", "")
            color = _EDGE_COLORS.get(label, "#666666")
            net.add_edge(src, dst, title=label, color=color, width=1.5)

        net.force_atlas_2based(central_gravity=0.015, gravity=-31)
        net.save_graph(output_path)
        logger.info("Graph HTML written to %s (%d nodes)", output_path, g.number_of_nodes())
        return str(Path(output_path).resolve())

    def export_community_report(
        self,
        result: GraphBuildResult,
        output_path: str,
    ) -> str:
        """Write a JSON community report. Returns absolute path."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {
            "node_count": result.node_count,
            "edge_count": result.edge_count,
            "community_count": result.community_count,
            "concept_count": result.concept_count,
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "communities": result.community_summary,
        }
        Path(output_path).write_text(json.dumps(report, indent=2))
        logger.info("Community report written to %s", output_path)
        return str(Path(output_path).resolve())
