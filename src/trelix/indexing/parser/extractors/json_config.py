"""JSON config file parser using tree-sitter-json.

Extracts configuration keys and their values as indexed symbols:
  - Top-level keys with object/array values → SymbolKind.SECTION
  - Top-level keys with scalar values → SymbolKind.CONSTANT
  - Nested keys up to max_depth → SECTION or CONSTANT
  - File-level MODULE symbol summarising top-level keys
  - ImportEdge for every package name found in dependency sections

Covers:
  - package.json  (scripts, jest, dependencies, devDependencies)
  - angular.json  (projects, budgets, build options)
  - tsconfig.json, .eslintrc.json, .babelrc.json (including JSONC with // comments)
  - Any other .json config file; root may be an object OR an array

Uses tree-sitter-json (already bundled) for:
  - Exact line_start / line_end per key-value pair
  - JSONC support — // and /* */ comments parse correctly (grammar built-in)
  - Comment nodes used as docstrings for the key that follows them
  - Error recovery — partial extraction from malformed JSON

See toml_config.py and yaml_config.py for TOML and YAML parsers.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from trelix.core.models import ImportEdge, Symbol, SymbolKind
from trelix.indexing.parser.base import BaseParser, ParseResult

# Top-level keys whose string values are treated as npm package ImportEdges
_DEP_KEYS = frozenset({
    "dependencies", "devDependencies", "peerDependencies",
    "optionalDependencies", "bundledDependencies", "bundleDependencies",
})


class JsonParser(BaseParser):
    """
    Parser for JSON and JSONC configuration files.

    Extracts key-path symbols so config values become searchable.
    Caps total symbols at 80 to avoid flooding from large config files.
    """

    MAX_SYMBOLS = 80
    MAX_DEPTH = 3
    MAX_BODY_LEN = 800

    @property
    def language_name(self) -> str:
        return "json"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def parse(self, source: str, file_id: int) -> ParseResult:
        return self._parse_json_treesitter(source, file_id)

    # ------------------------------------------------------------------
    # JSON via tree-sitter (exact lines, JSONC support)
    # ------------------------------------------------------------------

    def _parse_json_treesitter(self, source: str, file_id: int) -> ParseResult:
        try:
            import tree_sitter_languages
            from tree_sitter import Parser as TSParser
            lang = tree_sitter_languages.get_language("json")
            parser = TSParser()
            parser.set_language(lang)
        except Exception:
            return ParseResult(symbols=[], call_edges=[], import_edges=[],
                               parse_errors=1, type_edges=[])

        src_bytes = source.encode("utf-8", errors="replace")
        tree = parser.parse(src_bytes)
        parse_errors = 1 if tree.root_node.has_error else 0

        symbols: list[Symbol] = []
        import_edges: list[ImportEdge] = []

        # document → find first object or array root value
        root_value = next(
            (c for c in tree.root_node.children if c.type in ("object", "array")),
            None,
        )
        if root_value is None:
            return ParseResult(symbols=symbols, call_edges=[], import_edges=import_edges,
                               parse_errors=parse_errors, type_edges=[])

        # Build MODULE symbol from top-level key names (object root only)
        if root_value.type == "object":
            top_keys = self._collect_top_keys(root_value, src_bytes)
            if top_keys:
                symbols.append(Symbol(
                    file_id=file_id,
                    name="config",
                    qualified_name="config",
                    kind=SymbolKind.MODULE,
                    line_start=root_value.start_point[0] + 1,
                    line_end=root_value.end_point[0] + 1,
                    signature="{ " + ", ".join(top_keys[:12]) + " }",
                    body=self._render_node(root_value, src_bytes),
                    is_public=True,
                ))
            self._walk_object(root_value, "", file_id, symbols, import_edges,
                              src_bytes, depth=0)
        else:
            # Root is an array (e.g. some Babel/PostCSS configs)
            self._walk_array_root(root_value, file_id, symbols, import_edges,
                                  src_bytes)

        return ParseResult(symbols=symbols, call_edges=[], import_edges=import_edges,
                           parse_errors=parse_errors, type_edges=[])

    # ------------------------------------------------------------------
    # Object walker
    # ------------------------------------------------------------------

    def _walk_object(
        self,
        obj_node: Any,
        prefix: str,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        src_bytes: bytes,
        depth: int,
    ) -> None:
        for child in obj_node.children:
            if child.type != "pair" or len(symbols) >= self.MAX_SYMBOLS:
                continue

            key_node = child.child_by_field_name("key")
            val_node = child.child_by_field_name("value")
            if key_node is None or val_node is None:
                continue

            key_text = self._decode_string(key_node, src_bytes)
            path = f"{prefix}.{key_text}" if prefix else key_text
            line_start = key_node.start_point[0] + 1
            line_end = val_node.end_point[0] + 1
            docstring = self._get_preceding_comment(child, src_bytes)

            if val_node.type == "object":
                body = self._render_node(val_node, src_bytes)
                symbols.append(Symbol(
                    file_id=file_id,
                    name=key_text,
                    qualified_name=path,
                    kind=SymbolKind.SECTION,
                    line_start=line_start,
                    line_end=line_end,
                    signature=f'"{path}": {{...}}',
                    body=body,
                    docstring=docstring,
                    is_public=True,
                ))
                if depth < self.MAX_DEPTH:
                    self._walk_object(val_node, path, file_id, symbols, import_edges,
                                      src_bytes, depth + 1)

            elif val_node.type == "array":
                item_count = sum(1 for c in val_node.children if c.type not in (",", "[", "]"))
                body = self._render_node(val_node, src_bytes)
                symbols.append(Symbol(
                    file_id=file_id,
                    name=key_text,
                    qualified_name=path,
                    kind=SymbolKind.SECTION,
                    line_start=line_start,
                    line_end=line_end,
                    signature=f'"{path}": [{item_count} items]',
                    body=body,
                    docstring=docstring,
                    is_public=True,
                ))
                if depth < self.MAX_DEPTH:
                    self._walk_array_items(val_node, path, file_id, symbols,
                                           import_edges, src_bytes, depth + 1)

            else:
                # Scalar: string, number, true, false, null
                val_text = src_bytes[val_node.start_byte:val_node.end_byte].decode(
                    "utf-8", errors="replace"
                )
                body = f'"{path}": {val_text}'
                symbols.append(Symbol(
                    file_id=file_id,
                    name=key_text,
                    qualified_name=path,
                    kind=SymbolKind.CONSTANT,
                    line_start=line_start,
                    line_end=line_end,
                    signature=body[:200],
                    body=body,
                    docstring=docstring,
                    is_public=True,
                ))

            # Emit ImportEdges for dependency sections (depth 0 only)
            if depth == 0 and key_text in _DEP_KEYS and val_node.type == "object":
                self._emit_dependency_edges(val_node, file_id, import_edges, src_bytes)

    # ------------------------------------------------------------------
    # Array walkers
    # ------------------------------------------------------------------

    def _walk_array_items(
        self,
        arr_node: Any,
        prefix: str,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        src_bytes: bytes,
        depth: int,
    ) -> None:
        """Recurse into object items inside an array, using 'name' field as discriminator."""
        idx = 0
        for item in arr_node.children:
            if item.type == "object" and len(symbols) < self.MAX_SYMBOLS:
                # Use "name", "id", or "type" field value as sub-path if present
                label = self._find_name_field(item, src_bytes) or str(idx)
                item_path = f"{prefix}[{label}]"
                self._walk_object(item, item_path, file_id, symbols, import_edges,
                                  src_bytes, depth)
                idx += 1

    def _walk_array_root(
        self,
        arr_node: Any,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        src_bytes: bytes,
    ) -> None:
        """Handle a JSON file whose root is an array."""
        item_count = sum(1 for c in arr_node.children if c.type not in (",", "[", "]"))
        symbols.append(Symbol(
            file_id=file_id,
            name="config",
            qualified_name="config",
            kind=SymbolKind.MODULE,
            line_start=arr_node.start_point[0] + 1,
            line_end=arr_node.end_point[0] + 1,
            signature=f"[{item_count} items]",
            body=self._render_node(arr_node, src_bytes),
            is_public=True,
        ))
        self._walk_array_items(arr_node, "", file_id, symbols, import_edges,
                               src_bytes, depth=0)

    # ------------------------------------------------------------------
    # Dependency ImportEdges
    # ------------------------------------------------------------------

    def _emit_dependency_edges(
        self,
        obj_node: Any,
        file_id: int,
        import_edges: list[ImportEdge],
        src_bytes: bytes,
    ) -> None:
        """Emit one ImportEdge per package in a dependency object."""
        for child in obj_node.children:
            if child.type != "pair":
                continue
            key_node = child.child_by_field_name("key")
            if key_node is None:
                continue
            pkg_name = self._decode_string(key_node, src_bytes)
            if pkg_name:
                import_edges.append(ImportEdge(
                    file_id=file_id,
                    imported_from=pkg_name,
                    imported_names=[],
                ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_string(self, node: Any, src_bytes: bytes) -> str:
        """Decode a tree-sitter string node, stripping surrounding quotes."""
        raw = src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
            return raw[1:-1]
        return raw

    def _collect_top_keys(self, obj_node: Any, src_bytes: bytes) -> list[str]:
        """Return the list of top-level key names in an object node."""
        keys: list[str] = []
        for child in obj_node.children:
            if child.type == "pair":
                key_node = child.child_by_field_name("key")
                if key_node:
                    keys.append(self._decode_string(key_node, src_bytes))
        return keys

    def _find_name_field(self, obj_node: Any, src_bytes: bytes) -> Optional[str]:
        """Return value of 'name', 'id', or 'type' key if present in object."""
        for child in obj_node.children:
            if child.type != "pair":
                continue
            key_node = child.child_by_field_name("key")
            val_node = child.child_by_field_name("value")
            if key_node is None or val_node is None:
                continue
            key = self._decode_string(key_node, src_bytes)
            if key in ("name", "id", "type") and val_node.type == "string":
                return self._decode_string(val_node, src_bytes)
        return None

    def _get_preceding_comment(self, node: Any, src_bytes: bytes) -> Optional[str]:
        """Return JSONC comment immediately before this node, if any."""
        prev = node.prev_named_sibling
        if prev is not None and prev.type == "comment":
            # Only use if comment ends on the line immediately before node starts
            if prev.end_point[0] + 1 >= node.start_point[0]:
                raw = src_bytes[prev.start_byte:prev.end_byte].decode("utf-8", errors="replace")
                return self._clean_comment(raw)
        return None

    @staticmethod
    def _clean_comment(raw: str) -> str:
        """Strip // or /* */ delimiters from a JSONC comment."""
        raw = raw.strip()
        if raw.startswith("//"):
            return raw[2:].strip()
        # Block comment: /* ... */
        raw = re.sub(r'^/\*+\s*', '', raw)
        raw = re.sub(r'\s*\*+/$', '', raw)
        raw = re.sub(r'^\s*\*\s?', '', raw, flags=re.MULTILINE)
        return raw.strip()

    def _render_node(self, node: Any, src_bytes: bytes) -> str:
        """Extract raw source text for a node, capped at MAX_BODY_LEN."""
        text = src_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        if len(text) > self.MAX_BODY_LEN:
            text = text[: self.MAX_BODY_LEN] + "\n  ..."
        return text
