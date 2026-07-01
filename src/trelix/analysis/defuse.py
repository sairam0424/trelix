"""
Def-use chain extraction — local data-flow within function bodies.

A DefUseEdge records where a variable is defined (assigned) and used (read)
within a single symbol (function). This is intra-procedural only — no
cross-function tracking. For cross-function taint, see taint.py.

Research basis: CodeQL data-flow documentation (3-0 adversarial vote):
"nodes represent values at runtime, edges represent value movement."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from trelix.core.models import Symbol

logger = logging.getLogger("trelix.analysis.defuse")

# Tree-sitter node types for assignments and identifiers (Python-focused)
_ASSIGNMENT_TYPES: frozenset[str] = frozenset(
    [
        "assignment",  # x = 1
        "augmented_assignment",  # x += 1
        "named_expression",  # walrus :=
        "for_statement",  # for x in ...
        "with_statement",  # with open() as x
        "import_statement",  # import x
        "import_from_statement",
        "parameters",  # function parameters
    ]
)


@dataclass
class DefUseEdge:
    """
    A single def->use relationship for a variable within a symbol.

    edge_type:
      "def" -- the variable is assigned at def_line
      "use" -- the variable is read at use_line
    """

    symbol_id: int
    var_name: str
    def_line: int  # line of the assignment
    use_line: int  # line of the read (same as def_line for pure defs)
    edge_type: str  # "def" | "use"


class DataFlowExtractor:
    """
    Extract intra-procedural def-use edges from a symbol's body.

    Uses tree-sitter to parse the body and walk the AST for assignment
    and identifier nodes. Language-agnostic at the node-type level;
    falls back to regex on parse failure.

    This is a best-effort extractor -- it catches common patterns but not
    all language-specific assignment forms. Returns [] on any failure.
    """

    def extract(self, symbol: Symbol) -> list[DefUseEdge]:
        """Extract def-use edges from a symbol body. Never raises."""
        if not symbol.body or not symbol.id:
            return []
        try:
            return self._extract_tree_sitter(symbol)
        except Exception as exc:
            logger.debug("DataFlowExtractor fallback for %s: %s", symbol.qualified_name, exc)
            return self._extract_regex_fallback(symbol)

    def _extract_tree_sitter(self, symbol: Symbol) -> list[DefUseEdge]:
        """Walk tree-sitter AST to find assignment and identifier nodes."""
        try:
            from tree_sitter_languages import get_parser as get_ts_parser
        except ImportError:
            return []

        # Detect language from first token heuristic
        lang_name = "python"  # default; extendable
        body = symbol.body.encode("utf-8")

        try:
            parser = get_ts_parser(lang_name)
            tree = parser.parse(body)
        except Exception:
            return []

        edges: list[DefUseEdge] = []
        defined_vars: dict[str, int] = {}  # var_name -> def_line

        # Narrow symbol.id to int once so the nested closure can reference it safely.
        assert symbol.id is not None
        symbol_id: int = symbol.id

        def walk(node: object) -> None:
            node_type = getattr(node, "type", "")
            if node_type in _ASSIGNMENT_TYPES:
                # Left side of assignment = definition
                children = getattr(node, "children", [])
                for child in children:
                    if getattr(child, "type", "") == "identifier":
                        start_byte: int = getattr(child, "start_byte", 0)
                        end_byte: int = getattr(child, "end_byte", 0)
                        start_point: tuple[int, int] = getattr(child, "start_point", (0, 0))
                        var_name = body[start_byte:end_byte].decode("utf-8", errors="ignore")
                        line = start_point[0] + symbol.line_start
                        defined_vars[var_name] = line
                        edges.append(
                            DefUseEdge(
                                symbol_id=symbol_id,
                                var_name=var_name,
                                def_line=line,
                                use_line=line,
                                edge_type="def",
                            )
                        )
            elif node_type == "identifier":
                start_byte_n: int = getattr(node, "start_byte", 0)
                end_byte_n: int = getattr(node, "end_byte", 0)
                start_point_n: tuple[int, int] = getattr(node, "start_point", (0, 0))
                var_name = body[start_byte_n:end_byte_n].decode("utf-8", errors="ignore")
                if var_name in defined_vars:
                    use_line = start_point_n[0] + symbol.line_start
                    if use_line != defined_vars[var_name]:  # skip self-references
                        edges.append(
                            DefUseEdge(
                                symbol_id=symbol_id,
                                var_name=var_name,
                                def_line=defined_vars[var_name],
                                use_line=use_line,
                                edge_type="use",
                            )
                        )
            for child in getattr(node, "children", []):
                walk(child)

        walk(tree.root_node)
        return edges

    def _extract_regex_fallback(self, symbol: Symbol) -> list[DefUseEdge]:
        """Simple regex fallback when tree-sitter fails."""
        import re

        if not symbol.body or not symbol.id:
            return []
        edges: list[DefUseEdge] = []
        defined: dict[str, int] = {}
        for i, line in enumerate(symbol.body.splitlines(), start=symbol.line_start):
            m = re.match(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=", line)
            if m:
                var = m.group(1)
                if var not in ("def", "class", "return", "import"):
                    defined[var] = i
                    edges.append(DefUseEdge(int(symbol.id), var, i, i, "def"))
        return edges
