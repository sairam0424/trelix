"""CSS parser — tree-sitter AST traversal for CSS/SCSS/SASS/LESS files.

Extracts:
  - Class selectors (kind=VARIABLE) — .my-class { ... }
  - ID selectors (kind=VARIABLE) — #my-id { ... }
  - CSS custom properties (kind=CONSTANT) — --color-primary: #fff
  - @keyframes animations (kind=FUNCTION) — @keyframes fade-in { ... }
  - @media query blocks (kind=SECTION) — @media (max-width: 768px) { ... }
  - @font-face blocks (kind=SECTION)
  - @import / @use / @forward → ImportEdge

SCSS/SASS/LESS note:
  These files are parsed with the CSS grammar (best-effort). Standard CSS
  within SCSS files (selectors, custom properties, @media, @keyframes) is
  extracted correctly. SCSS-specific syntax ($variables, @mixin, @include,
  nesting with &) will produce parse errors but won't crash — valid CSS
  portions are still indexed.

Tree-sitter note:
  CSS grammar has NO child_by_field_name fields — all access is via
  child type iteration, same pattern as the HTML parser.
"""

from __future__ import annotations

import re

from tree_sitter import Node

from trelix.core.models import ImportEdge, Symbol, SymbolKind
from trelix.indexing.parser._grammar import load_language, make_parser
from trelix.indexing.parser.base import BaseParser, ParseResult

# Cap to prevent massive generated CSS files from flooding the symbol table
MAX_SELECTORS = 300  # total class/id selector symbols per file
MAX_CSS_VARS = 100  # total CSS custom property symbols per file


class CssParser(BaseParser):
    """Tree-sitter based CSS parser. Also handles SCSS/SASS/LESS (best-effort)."""

    def __init__(self) -> None:
        self._ts_language = load_language("css")
        self._parser = make_parser("css")

    @property
    def language_name(self) -> str:
        return "css"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def parse(self, source: str, file_id: int) -> ParseResult:
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        symbols: list[Symbol] = []
        import_edges: list[ImportEdge] = []
        seen_selectors: set[str] = set()  # deduplicate class/id symbols per file
        seen_vars: set[str] = set()  # deduplicate CSS custom properties

        self._walk_stylesheet(
            root,
            source_bytes,
            file_id,
            symbols,
            import_edges,
            seen_selectors,
            seen_vars,
        )

        return ParseResult(
            symbols=symbols,
            call_edges=[],
            import_edges=import_edges,
            parse_errors=self._count_errors(root),
        )

    # ------------------------------------------------------------------
    # Stylesheet-level walk
    # ------------------------------------------------------------------

    def _walk_stylesheet(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        seen_selectors: set[str],
        seen_vars: set[str],
    ) -> None:
        for child in node.children:
            ntype = child.type

            if ntype == "rule_set":
                self._handle_rule_set(
                    child,
                    src,
                    file_id,
                    symbols,
                    import_edges,
                    seen_selectors,
                    seen_vars,
                )

            elif ntype == "media_statement":
                self._handle_media(
                    child,
                    src,
                    file_id,
                    symbols,
                    import_edges,
                    seen_selectors,
                    seen_vars,
                )

            elif ntype == "keyframes_statement":
                self._handle_keyframes(child, src, file_id, symbols)

            elif ntype == "import_statement":
                edge = self._handle_import(child, src, file_id)
                if edge:
                    import_edges.append(edge)

            elif ntype == "supports_statement":
                # @supports (display: grid) { ... }
                # Grammar uses a dedicated supports_statement node, not at_rule.
                block_node = self._get_child_by_type(child, "block")
                if block_node:
                    self._walk_stylesheet(
                        block_node,
                        src,
                        file_id,
                        symbols,
                        import_edges,
                        seen_selectors,
                        seen_vars,
                    )

            elif ntype == "scope_statement":
                # @scope (.card) { .title { } } — CSS Scoping Level 1
                block_node = self._get_child_by_type(child, "block")
                if block_node:
                    self._walk_stylesheet(
                        block_node,
                        src,
                        file_id,
                        symbols,
                        import_edges,
                        seen_selectors,
                        seen_vars,
                    )

            elif ntype == "at_rule":
                # Generic @-rule: @font-face, @layer, @container, etc.
                self._handle_at_rule(
                    child,
                    src,
                    file_id,
                    symbols,
                    import_edges,
                    seen_selectors,
                    seen_vars,
                )

    # ------------------------------------------------------------------
    # Rule set
    # ------------------------------------------------------------------

    def _handle_rule_set(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        seen_selectors: set[str],
        seen_vars: set[str],
    ) -> None:
        """Handle: .my-class { ... }  /  #my-id { ... }  /  tag { ... }"""
        selectors_node = self._get_child_by_type(node, "selectors")
        block_node = self._get_child_by_type(node, "block")

        selector_names: list[tuple[str, str]] = []  # (name, "class"|"id")
        if selectors_node:
            self._collect_selectors(selectors_node, src, selector_names)

        body_text = self._txt(node, src)

        # Emit one symbol per unique class/id selector found in this rule
        for sel_name, sel_kind in selector_names:
            if len(seen_selectors) >= MAX_SELECTORS:
                break
            if sel_name in seen_selectors:
                continue
            seen_selectors.add(sel_name)

            docstring = self._get_preceding_comment(node, src)
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=sel_name,
                    qualified_name=sel_name,
                    kind=SymbolKind.VARIABLE,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=f"{sel_name} {{ ... }}",
                    body=body_text[:800],
                    docstring=docstring,
                    is_public=True,
                )
            )

        # Extract CSS custom properties + nested rules from the block
        if block_node:
            self._extract_css_vars(block_node, src, file_id, symbols, seen_vars)
            # CSS Nesting (modern CSS): .parent { .child { } }
            for nested in block_node.children:
                if nested.type == "rule_set":
                    self._handle_rule_set(
                        nested,
                        src,
                        file_id,
                        symbols,
                        import_edges,
                        seen_selectors,
                        seen_vars,
                    )

    def _collect_selectors(
        self,
        node: Node,
        src: bytes,
        results: list[tuple[str, str]],
    ) -> None:
        """Recursively collect (name, kind) for class and id selectors."""
        for child in node.children:
            if child.type == "class_selector":
                name_node = self._get_child_by_type(child, "class_name")
                if name_node:
                    results.append(("." + self._txt(name_node, src), "class"))
            elif child.type == "id_selector":
                name_node = self._get_child_by_type(child, "id_name")
                if name_node:
                    results.append(("#" + self._txt(name_node, src), "id"))
            elif child.type not in ("tag_name", "class_name", "id_name", ",", "{", "}"):
                # Recurse into compound selectors, descendant selectors, etc.
                self._collect_selectors(child, src, results)

    # ------------------------------------------------------------------
    # CSS custom properties
    # ------------------------------------------------------------------

    def _extract_css_vars(
        self,
        block_node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        seen_vars: set[str],
    ) -> None:
        """Extract --custom-property declarations from a block node."""
        for child in block_node.children:
            if child.type != "declaration":
                continue
            if len(seen_vars) >= MAX_CSS_VARS:
                break
            prop_node = self._get_child_by_type(child, "property_name")
            if not prop_node:
                continue
            prop = self._txt(prop_node, src)
            if not prop.startswith("--"):
                continue
            if prop in seen_vars:
                continue
            seen_vars.add(prop)

            # Get the value — everything after the property_name and ":"
            value = self._get_declaration_value(child, src, prop)
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=prop,
                    qualified_name=prop,
                    kind=SymbolKind.CONSTANT,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    signature=f"{prop}: {value}",
                    body=self._txt(child, src),
                    is_public=True,
                )
            )

    def _get_declaration_value(self, decl_node: Node, src: bytes, prop: str) -> str:
        """Extract the value portion of a declaration node."""
        full = self._txt(decl_node, src)
        # Strip the property name and colon: "--color-primary: #fff" → "#fff"
        after_colon = full[len(prop) :].lstrip().lstrip(":").strip().rstrip(";").strip()
        return after_colon[:120] if after_colon else ""

    # ------------------------------------------------------------------
    # @media
    # ------------------------------------------------------------------

    def _handle_media(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        seen_selectors: set[str],
        seen_vars: set[str],
    ) -> None:
        """Handle: @media (max-width: 768px) { ... }"""
        # Build query description: everything between @media and the opening {
        full_text = self._txt(node, src)
        # Extract the query part: @media <query> {
        query_match = re.match(r"@media\s+(.+?)\s*\{", full_text, re.DOTALL)
        query_str = query_match.group(1).strip() if query_match else "query"
        # Collapse whitespace
        query_str = re.sub(r"\s+", " ", query_str)[:120]

        name = f"@media {query_str}"
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.SECTION,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"@media {query_str} {{ ... }}",
                body=full_text[:800],
                is_public=True,
            )
        )

        # Also recurse into rule_sets inside the media block
        block_node = self._get_child_by_type(node, "block")
        if block_node:
            self._walk_stylesheet(
                block_node,
                src,
                file_id,
                symbols,
                import_edges,
                seen_selectors,
                seen_vars,
            )

    # ------------------------------------------------------------------
    # @keyframes
    # ------------------------------------------------------------------

    def _handle_keyframes(
        self, node: Node, src: bytes, file_id: int, symbols: list[Symbol]
    ) -> None:
        """Handle: @keyframes fade-in { from { ... } to { ... } }"""
        name_node = self._get_child_by_type(node, "keyframes_name")
        if not name_node:
            return
        name = self._txt(name_node, src).strip()
        if not name:
            return

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.FUNCTION,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"@keyframes {name}",
                body=self._txt(node, src)[:800],
                docstring=self._get_preceding_comment(node, src),
                is_public=True,
            )
        )

    # ------------------------------------------------------------------
    # @import / @use / @forward (SCSS)
    # ------------------------------------------------------------------

    def _handle_import(self, node: Node, src: bytes, file_id: int) -> ImportEdge | None:
        """Handle: @import "file.css" / @import url("file") / @use "module" (SCSS)"""
        full = self._txt(node, src)
        # Extract path from quoted strings or url()
        path_match = re.search(r'["\']([^"\']+)["\']', full)
        if not path_match:
            path_match = re.search(r'url\(["\']?([^)"\']+)["\']?\)', full)
        if not path_match:
            return None
        path = path_match.group(1).strip()
        if not path:
            return None
        return ImportEdge(
            file_id=file_id,
            imported_from=path,
            imported_names=[],
        )

    # ------------------------------------------------------------------
    # Generic @-rule (@font-face, @supports, @layer, etc.)
    # ------------------------------------------------------------------

    def _handle_at_rule(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        seen_selectors: set[str],
        seen_vars: set[str],
    ) -> None:
        """Handle generic at-rules: @font-face, @supports, @layer, @use, @forward."""
        at_keyword_node = self._get_child_by_type(node, "at_keyword")
        keyword = self._txt(at_keyword_node, src).lstrip("@") if at_keyword_node else ""

        # @use / @forward are SCSS import-like statements
        if keyword in ("use", "forward"):
            edge = self._handle_import(node, src, file_id)
            if edge:
                import_edges.append(edge)
            return

        # @font-face → emit as a SECTION symbol
        if keyword == "font-face":
            block_node = self._get_child_by_type(node, "block")
            family = ""
            if block_node:
                for child in block_node.children:
                    if child.type == "declaration":
                        prop = self._get_child_by_type(child, "property_name")
                        if prop and self._txt(prop, src) == "font-family":
                            family = re.sub(
                                r'["\';]',
                                "",
                                self._get_declaration_value(child, src, "font-family"),
                            ).strip()
                            break
            name = f"@font-face {family}".strip()
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.SECTION,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=f"@font-face {{ font-family: {family} }}",
                    body=self._txt(node, src)[:500],
                )
            )
            return

        # @layer and @container — recurse into their blocks
        if keyword in ("layer", "container"):
            block_node = self._get_child_by_type(node, "block")
            if block_node:
                self._walk_stylesheet(
                    block_node,
                    src,
                    file_id,
                    symbols,
                    import_edges,
                    seen_selectors,
                    seen_vars,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_preceding_comment(self, node: Node, src: bytes) -> str | None:
        """Collect CSS block comment immediately before this node."""
        prev = node.prev_named_sibling
        if prev is not None and prev.type == "comment":
            if prev.end_point[0] + 1 >= node.start_point[0]:
                raw = self._txt(prev, src)
                return self._clean_comment(raw)
        return None

    @staticmethod
    def _clean_comment(raw: str) -> str:
        """Strip /* ... */ delimiters and leading * from CSS comments."""
        raw = re.sub(r"^/\*+\s*", "", raw.strip())
        raw = re.sub(r"\s*\*+/$", "", raw)
        raw = re.sub(r"^\s*\*\s?", "", raw, flags=re.MULTILINE)
        return raw.strip()

    def _count_errors(self, node: Node) -> int:
        count = 1 if node.type == "ERROR" else 0
        for child in node.children:
            count += self._count_errors(child)
        return count

    def _txt(self, node: Node, src: bytes) -> str:
        return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _get_child_by_type(self, node: Node, type_name: str) -> Node | None:
        for child in node.children:
            if child.type == type_name:
                return child
        return None
