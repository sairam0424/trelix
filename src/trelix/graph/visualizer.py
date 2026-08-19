"""Pyvis-based interactive visualization for CodeGraph."""

from __future__ import annotations

import html
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


def _html_safe(value: object) -> str:
    """Escape one repo-controlled value on its way into the generated page.

    Node attrs are indexed content, not trusted strings: a markdown heading becomes a
    `qualified_name`, a filename becomes `file`. pyvis builds the page with a Jinja
    `Environment` that has NO autoescape, so whatever is handed to it is what the
    browser parses. `html.escape` rather than a hand-rolled `.replace()` chain because
    the escaping has to be right for two contexts at once — element content and a
    double-quoted attribute value — which is what `quote=True` covers.
    """
    return html.escape(str(value), quote=True)


def _html_safe_node_id(node_id: Any) -> Any:
    """Return a node id that cannot break out of pyvis's select menu.

    The tooltip is not the only sink. With `select_menu=True` pyvis emits
    `<option value="{{node.id}}">{{node.id}}</option>` — raw Jinja, no JSON encoder in
    front of it, no hover required, and completely untouched by escaping the tooltip.
    Symbol ids are ints, but `generic_edges.source_ref` becomes a STRING node id that
    the Jira/Linear connectors copy verbatim out of an HTTP response, so a `">` in a
    ticket key closes the attribute and then the element.

    Ints pass through unchanged, so the export of an ordinary code graph is unaffected;
    only the string ids are escaped. Applied to edge endpoints too — escape one side
    only and every edge silently detaches from its node.
    """
    return node_id if isinstance(node_id, int) else _html_safe(node_id)


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
            # `<b>` and `<br>` below are the only markup this exporter authors; every
            # hole in the f-string is repo text. pyvis switches its whole template to a
            # custom popup that does `popup.innerHTML = nodeData[0].title` as soon as ANY
            # node title contains the substring "href" (network.py:449-460) — a rendering
            # branch selected by attacker bytes, which a payload supplies for itself. So
            # an unescaped `<` here is a script tag in the developer's browser, not text
            # in a tooltip. Escaped once per value on the way in.
            title = (
                f"<b>{_html_safe(attrs.get('qualified_name', label))}</b><br>"
                f"Kind: {_html_safe(attrs.get('kind', '?'))}<br>"
                f"File: {_html_safe(attrs.get('file', '?'))}<br>"
                f"Community: {_html_safe(community)}"
            )
            net.add_node(
                _html_safe_node_id(node_id),
                # `label` is deliberately NOT escaped: vis-network draws it with canvas
                # text and never puts it in the DOM, so it is not an HTML sink, and
                # escaping it would render a legitimate `Vec<T>` as `Vec&lt;T&gt;` on the
                # graph. That stops being true the day this exporter sets
                # `font: {multi: "html"}`, which turns labels into a DOM sink.
                label=label[:25],
                title=title,
                color=color,
                size=size,
            )

        # Add edges
        for src, dst, edge_attrs in g.edges(data=True):
            label = edge_attrs.get("label", "")
            color = _EDGE_COLORS.get(label, "#666666")
            net.add_edge(
                _html_safe_node_id(src),
                _html_safe_node_id(dst),
                title=label,
                color=color,
                width=1.5,
            )

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
        # Additive. `community_count` on its own reads as a healthy number when it
        # is mostly singletons (6640 communities / 6579 singletons on trelix's own
        # index), so the report carries the shape alongside the count. Omitted
        # rather than faked when the result predates the assessment.
        if result.partition_quality is not None:
            report["partition_quality"] = result.partition_quality.as_dict()
        Path(output_path).write_text(json.dumps(report, indent=2))
        logger.info("Community report written to %s", output_path)
        return str(Path(output_path).resolve())
