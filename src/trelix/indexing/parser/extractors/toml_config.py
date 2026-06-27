"""TOML config file parser using tree-sitter-toml.

Extracts configuration keys and their values as indexed symbols:
  - File-level MODULE symbol summarising all top-level table names
  - [section] headers → SymbolKind.SECTION with their key-value pairs as body
  - [[array-of-tables]] → SECTION with [[key]] signature, disambiguated by "name" field
  - Key-value pairs inside sections → CONSTANT or SECTION (for inline tables)
  - Dotted section headers ([tool.pytest.ini_options]) → qualified_name preserved
  - # comment immediately before a table → docstring on its SECTION symbol
  - Dependency sections → ImportEdge per package (Cargo.toml, pip-style pyproject.toml)

Covers:
  - pyproject.toml  ([tool.pytest], [tool.ruff], [build-system], [project])
  - Cargo.toml      ([package], [dependencies], [features], [[bin]])
  - Any other .toml config file

Uses tree-sitter-toml (already bundled in tree-sitter-languages) for exact
line_start / line_end on every symbol — no text-search approximation.
"""

from __future__ import annotations

import re
from typing import Any

from trelix.core.models import ImportEdge, Symbol, SymbolKind
from trelix.indexing.parser.base import BaseParser, ParseResult

# TS node types that carry key text
_KEY_TYPES = {"bare_key", "quoted_key", "dotted_key"}

# Table section names whose keys are treated as package dependency ImportEdges
_DEP_SECTIONS = frozenset(
    {
        "dependencies",
        "dev-dependencies",
        "build-dependencies",
        "dev_dependencies",
        "build_dependencies",
        # pyproject.toml optional-dependencies sub-table is nested, handled via prefix check
    }
)


class TomlParser(BaseParser):
    """
    Parser for TOML configuration files.

    Extracts key-path symbols so config values become searchable.
    Caps total symbols at 80 to avoid flooding from large config files.
    """

    MAX_SYMBOLS = 80
    MAX_DEPTH = 3
    MAX_BODY_LEN = 800

    @property
    def language_name(self) -> str:
        return "toml"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def parse(self, source: str, file_id: int) -> ParseResult:
        try:
            from trelix.indexing.parser._grammar import make_parser

            parser = make_parser("toml")
        except Exception:
            return ParseResult(
                symbols=[], call_edges=[], import_edges=[], parse_errors=1, type_edges=[]
            )

        src_bytes = source.encode("utf-8", errors="replace")
        tree = parser.parse(src_bytes)
        parse_errors = 1 if tree.root_node.has_error else 0

        symbols: list[Symbol] = []
        import_edges: list[ImportEdge] = []

        # MODULE symbol: collect top-level table names for a file summary
        top_sections: list[str] = []
        for child in tree.root_node.children:
            if child.type in ("table", "table_array_element"):
                for c in child.children:
                    if c.type in _KEY_TYPES:
                        top_sections.append(self._key_text(c, src_bytes))
                        break

        if top_sections:
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name="config",
                    qualified_name="config",
                    kind=SymbolKind.MODULE,
                    line_start=1,
                    line_end=tree.root_node.end_point[0] + 1,
                    signature="[" + "] [".join(top_sections[:10]) + "]",
                    body=self._render_node(tree.root_node, src_bytes),
                    is_public=True,
                )
            )

        for child in tree.root_node.children:
            if len(symbols) >= self.MAX_SYMBOLS:
                break

            if child.type == "pair":
                # Top-level key-value (before any section header)
                self._handle_pair(
                    child, "", file_id, symbols, import_edges, src_bytes, depth=0, section_name=""
                )

            elif child.type in ("table", "table_array_element"):
                self._handle_table(child, file_id, symbols, import_edges, src_bytes)

        return ParseResult(
            symbols=symbols,
            call_edges=[],
            import_edges=import_edges,
            parse_errors=parse_errors,
            type_edges=[],
        )

    # ------------------------------------------------------------------
    # Table sections: [section] and [[array-of-tables]]
    # ------------------------------------------------------------------

    def _handle_table(
        self,
        table_node: Any,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        src_bytes: bytes,
    ) -> None:
        is_array = table_node.type == "table_array_element"

        # Extract header key
        header_key = ""
        for child in table_node.children:
            if child.type in _KEY_TYPES:
                header_key = self._key_text(child, src_bytes)
                break

        if not header_key:
            return

        # For [[array-of-tables]], try to use the "name" pair value as disambiguator
        qualified_name = header_key
        if is_array:
            name_val = self._find_pair_value(table_node, "name", src_bytes)
            if name_val:
                qualified_name = f"{header_key}[{name_val}]"

        line_start = table_node.start_point[0] + 1
        line_end = table_node.end_point[0] + 1
        body = self._render_node(table_node, src_bytes)
        last_part = qualified_name.split(".")[-1]
        sig_bracket = f"[[{header_key}]]" if is_array else f"[{header_key}]"
        docstring = self._get_preceding_comment(table_node, src_bytes)

        if len(symbols) < self.MAX_SYMBOLS:
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=last_part,
                    qualified_name=qualified_name,
                    kind=SymbolKind.SECTION,
                    line_start=line_start,
                    line_end=line_end,
                    signature=sig_bracket,
                    body=body,
                    docstring=docstring,
                    is_public=True,
                )
            )

        # Pairs inside the table
        # The leaf section name (e.g. "dependencies" from "[dependencies]")
        section_leaf = header_key.split(".")[-1]
        for child in table_node.children:
            if child.type == "pair" and len(symbols) < self.MAX_SYMBOLS:
                self._handle_pair(
                    child,
                    qualified_name,
                    file_id,
                    symbols,
                    import_edges,
                    src_bytes,
                    depth=1,
                    section_name=section_leaf,
                )

        # Emit ImportEdges for dependency sections
        if section_leaf in _DEP_SECTIONS or header_key in _DEP_SECTIONS:
            self._emit_dep_edges(table_node, file_id, import_edges, src_bytes)

    # ------------------------------------------------------------------
    # Key-value pairs
    # ------------------------------------------------------------------

    def _handle_pair(
        self,
        pair_node: Any,
        prefix: str,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        src_bytes: bytes,
        depth: int,
        section_name: str,
    ) -> None:
        key_node = None
        val_node = None
        for child in pair_node.children:
            if child.type in _KEY_TYPES and key_node is None:
                key_node = child
            elif child.type != "=" and key_node is not None:
                val_node = child
                break

        if key_node is None or val_node is None:
            return

        key = self._key_text(key_node, src_bytes)
        path = f"{prefix}.{key}" if prefix else key
        line_start = key_node.start_point[0] + 1
        line_end = val_node.end_point[0] + 1
        body = self._render_node(val_node, src_bytes)
        docstring = self._get_preceding_comment(pair_node, src_bytes)

        if val_node.type == "inline_table":
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=key,
                    qualified_name=path,
                    kind=SymbolKind.SECTION,
                    line_start=line_start,
                    line_end=line_end,
                    signature=f"{path} = {{...}}",
                    body=body,
                    docstring=docstring,
                    is_public=True,
                )
            )
            if depth < self.MAX_DEPTH:
                for child in val_node.children:
                    if child.type == "pair" and len(symbols) < self.MAX_SYMBOLS:
                        self._handle_pair(
                            child,
                            path,
                            file_id,
                            symbols,
                            import_edges,
                            src_bytes,
                            depth + 1,
                            section_name=key,
                        )

        elif val_node.type == "array":
            item_count = sum(
                1 for c in val_node.children if c.type not in (",", "[", "]", "comment")
            )
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=key,
                    qualified_name=path,
                    kind=SymbolKind.SECTION,
                    line_start=line_start,
                    line_end=line_end,
                    signature=f"{path} = [{item_count} items]",
                    body=body,
                    docstring=docstring,
                    is_public=True,
                )
            )

        else:
            # Scalar: string, integer, float, boolean, datetime
            val_text = src_bytes[val_node.start_byte : val_node.end_byte].decode(
                "utf-8", errors="replace"
            )
            body = f"{path} = {val_text}"
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=key,
                    qualified_name=path,
                    kind=SymbolKind.CONSTANT,
                    line_start=line_start,
                    line_end=line_end,
                    signature=body[:200],
                    body=body,
                    docstring=docstring,
                    is_public=True,
                )
            )

    # ------------------------------------------------------------------
    # Dependency ImportEdges
    # ------------------------------------------------------------------

    def _emit_dep_edges(
        self,
        table_node: Any,
        file_id: int,
        import_edges: list[ImportEdge],
        src_bytes: bytes,
    ) -> None:
        """Emit one ImportEdge per package key in a dependency table."""
        for child in table_node.children:
            if child.type != "pair":
                continue
            key_node = next((c for c in child.children if c.type in _KEY_TYPES), None)
            if key_node is None:
                continue
            pkg = self._key_text(key_node, src_bytes)
            if pkg:
                import_edges.append(
                    ImportEdge(
                        file_id=file_id,
                        imported_from=pkg,
                        imported_names=[],
                    )
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_pair_value(self, table_node: Any, key: str, src_bytes: bytes) -> str | None:
        """Return the string value of the first pair matching `key` in table_node."""
        for child in table_node.children:
            if child.type != "pair":
                continue
            key_node = next((c for c in child.children if c.type in _KEY_TYPES), None)
            if key_node and self._key_text(key_node, src_bytes) == key:
                val_node = next((c for c in child.children if c.type == "string"), None)
                if val_node:
                    raw = src_bytes[val_node.start_byte : val_node.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    return raw.strip("\"'")
        return None

    def _get_preceding_comment(self, node: Any, src_bytes: bytes) -> str | None:
        """Return the # comment immediately before this node, if any."""
        prev = node.prev_named_sibling
        if prev is not None and prev.type == "comment":
            if prev.end_point[0] + 1 >= node.start_point[0]:
                raw = src_bytes[prev.start_byte : prev.end_byte].decode("utf-8", errors="replace")
                return re.sub(r"^#+\s*", "", raw.strip())
        return None

    def _key_text(self, node: Any, src_bytes: bytes) -> str:
        """Extract key string from bare_key, quoted_key, or dotted_key node."""
        if node.type == "bare_key":
            return src_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        if node.type == "quoted_key":
            text = src_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
            return text.strip("\"'")
        if node.type == "dotted_key":
            parts = [
                self._key_text(c, src_bytes)
                for c in node.children
                if c.type in {"bare_key", "quoted_key", "dotted_key"}
            ]
            return ".".join(parts)
        return src_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _render_node(self, node: Any, src_bytes: bytes) -> str:
        """Extract raw source text for a node, capped at MAX_BODY_LEN."""
        text = src_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        if len(text) > self.MAX_BODY_LEN:
            text = text[: self.MAX_BODY_LEN] + "\n  ..."
        return text
