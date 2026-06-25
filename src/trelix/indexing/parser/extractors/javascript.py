"""JavaScript parser — direct AST traversal using Tree-sitter.

Extracts:
  - Classes (kind=CLASS) — including extends type edges
  - Class property fields (kind=VARIABLE, parent_id=class) — public fields only (#private skipped)
  - Methods inside classes (kind=METHOD, parent_id=class)
  - Top-level function declarations (kind=FUNCTION) — regular and generator functions
  - Arrow functions assigned to const/let (kind=FUNCTION)
  - Module-level constants: ALL_CAPS names + all exported consts (kind=CONSTANT)
  - CommonJS module.exports = {...} (kind=CONSTANT)
  - Import statements (ES6 + CommonJS require) → ImportEdge
  - Re-exports → ImportEdge
  - Call sites inside functions/methods → CallEdge
  - Module docstring — leading JSDoc comment at top of file (kind=MODULE)

Parent linkage:
  parent_id in Symbol is set to the LOCAL INDEX in the symbols list during
  parsing. The Indexer remaps this to the actual DB id after insertion.
"""

from __future__ import annotations

import re

import tree_sitter_languages
from tree_sitter import Node, Parser

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge

from ..base import BaseParser, ParseResult


class JavaScriptParser(BaseParser):
    """Tree-sitter based JavaScript parser using direct AST traversal."""

    # Cap on class field symbols extracted per class (avoids symbol flood).
    MAX_CLASS_FIELDS: int = 30

    def __init__(self) -> None:
        self._ts_language = tree_sitter_languages.get_language("javascript")
        self._parser = Parser()
        self._parser.set_language(self._ts_language)

    @property
    def language_name(self) -> str:
        return "javascript"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def parse(self, source: str, file_id: int) -> ParseResult:
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        symbols: list[Symbol] = []
        import_edges: list[ImportEdge] = []
        type_edges: list[TypeEdge] = []
        # raw_calls: (caller_local_idx, callee_name, line) — remapped by Indexer
        raw_calls: list[tuple[int, str, int]] = []

        # Module docstring — leading JSDoc/block comment before first declaration
        module_doc = self._get_module_docstring(root, source_bytes)
        if module_doc:
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name="<module>",
                    qualified_name="<module>",
                    kind=SymbolKind.MODULE,
                    line_start=1,
                    line_end=root.end_point[0] + 1,
                    signature=module_doc.split("\n")[0][:120],
                    body=module_doc,
                    docstring=module_doc,
                    is_public=True,
                )
            )

        self._walk_top_level(
            root, source_bytes, file_id, symbols, import_edges, type_edges, raw_calls
        )

        call_edges: list[CallEdge] = [
            CallEdge(caller_id=caller_idx, callee_name=name, line=line)
            for caller_idx, name, line in raw_calls
        ]

        return ParseResult(
            symbols=symbols,
            call_edges=call_edges,
            import_edges=import_edges,
            parse_errors=self._count_errors(root),
            type_edges=type_edges,
        )

    # ------------------------------------------------------------------
    # Top-level walk
    # ------------------------------------------------------------------

    def _walk_top_level(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        for child in node.children:
            ntype = child.type
            if ntype == "import_statement":
                import_edges.extend(self._extract_import(child, src, file_id))
            elif ntype == "export_statement":
                self._handle_export(
                    child, src, file_id, symbols, import_edges, type_edges, raw_calls
                )
            elif ntype == "class_declaration":
                self._handle_class(child, src, file_id, symbols, type_edges, raw_calls)
            elif ntype in ("function_declaration", "generator_function_declaration"):
                self._handle_function(child, src, file_id, symbols, raw_calls)
            elif ntype in ("lexical_declaration", "variable_declaration"):
                self._handle_var_decl(child, src, file_id, symbols, import_edges, raw_calls)
            elif ntype == "expression_statement":
                # CommonJS: require(...), module.exports = ...
                self._handle_expression_stmt(child, src, file_id, symbols, import_edges)

    def _handle_export(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        # Re-export: export { A, B } from './module' — record as import edge
        export_clause = self._get_child_by_type(node, "export_clause")
        source_str = self._get_child_by_type(node, "string")
        if export_clause and source_str:
            module_name = self._txt(source_str, src).strip("'\"")
            names = []
            for spec in export_clause.children:
                if spec.type == "export_specifier":
                    id_node = self._get_child_by_type(spec, "identifier")
                    if id_node:
                        names.append(self._txt(id_node, src))
            import_edges.append(
                ImportEdge(
                    file_id=file_id,
                    imported_from=module_name,
                    imported_names=names,
                )
            )
            return

        # Collect decorators from the export_statement node (rare in JS but possible)
        decs = [
            self._extract_decorator_text(c, src) for c in node.children if c.type == "decorator"
        ]
        for child in node.children:
            ntype = child.type
            if ntype == "class_declaration":
                self._handle_class(
                    child,
                    src,
                    file_id,
                    symbols,
                    type_edges,
                    raw_calls,
                    exported=True,
                    decorators=decs,
                )
            elif ntype in ("function_declaration", "generator_function_declaration"):
                self._handle_function(
                    child, src, file_id, symbols, raw_calls, exported=True, decorators=decs
                )
            elif ntype in ("lexical_declaration", "variable_declaration"):
                self._handle_var_decl(
                    child, src, file_id, symbols, import_edges, raw_calls, exported=True
                )

    # ------------------------------------------------------------------
    # Class handling
    # ------------------------------------------------------------------

    def _handle_class(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        type_edges: list[TypeEdge],
        raw_calls: list[tuple[int, str, int]],
        exported: bool = False,
        decorators: list[str] | None = None,
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        own_decs = [
            self._extract_decorator_text(c, src) for c in node.children if c.type == "decorator"
        ]
        all_decs = (decorators or []) + own_decs

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=SymbolKind.CLASS,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=self._class_signature(node, src),
            body=self._txt(node, src),
            docstring=self._get_preceding_comment(node, src),
            decorators=all_decs,
            is_public=exported,
        )
        class_local_idx = len(symbols)
        symbols.append(sym)

        # Extract inheritance type edges (JS extends only, no implements).
        # In the JS grammar class_heritage has direct children:
        #   extends_clause { extends identifier } — tree-sitter >=0.21 wraps it
        # OR (older grammar / v0.21 JavaScript):
        #   extends keyword + identifier as siblings directly under class_heritage.
        # We handle both forms.
        heritage_node = self._get_child_by_type(node, "class_heritage")
        if heritage_node:
            found_extends = False
            for clause in heritage_node.children:
                if clause.type == "extends_clause":
                    # Wrapped form: tree-sitter >= some version
                    for c in clause.children:
                        if c.type == "identifier":
                            type_edges.append(
                                TypeEdge(
                                    from_symbol_id=class_local_idx,
                                    to_type_name=self._txt(c, src),
                                    edge_kind="extends",
                                )
                            )
                    found_extends = True
            if not found_extends:
                # Flat form: class_heritage → extends + identifier as direct siblings
                saw_extends_kw = False
                for c in heritage_node.children:
                    if c.type == "extends":
                        saw_extends_kw = True
                    elif saw_extends_kw and c.type == "identifier":
                        type_edges.append(
                            TypeEdge(
                                from_symbol_id=class_local_idx,
                                to_type_name=self._txt(c, src),
                                edge_kind="extends",
                            )
                        )

        # Walk class body: methods + public field definitions
        body_node = self._get_child_by_type(node, "class_body")
        if body_node:
            field_count = 0
            for child in body_node.children:
                if child.type == "method_definition":
                    self._handle_method(
                        child, src, file_id, symbols, class_local_idx, name, raw_calls
                    )
                elif child.type == "field_definition" and field_count < self.MAX_CLASS_FIELDS:
                    if self._handle_class_field(
                        child, src, file_id, symbols, class_local_idx, name
                    ):
                        field_count += 1

    def _handle_method(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        class_local_idx: int,
        class_name: str,
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        name_node = self._get_child_by_type(node, "property_identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        func_local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=f"{class_name}.{name}",
                kind=SymbolKind.METHOD,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=self._method_signature(node, src, class_name),
                body=self._txt(node, src),
                parent_id=class_local_idx,
                docstring=self._get_preceding_comment(node, src),
                is_public=not name.startswith("_"),
            )
        )

        # Walk method body for call edges
        body_node = self._get_child_by_type(node, "statement_block")
        if body_node:
            self._walk_body(body_node, src, func_local_idx, raw_calls)

    def _handle_class_field(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        class_local_idx: int,
        class_name: str,
    ) -> bool:
        """Extract a JS class field declaration as a VARIABLE symbol.

        Returns True if a symbol was created (so caller can track the count).
        Skips private fields using # syntax (private_property_identifier).

        Handles:
          - State: state = { count: 0 }
          - Arrow function handlers: handleClick = () => { ... }
          - Static fields: static DEFAULT_TIMEOUT = 5000
          - Regular typed fields: title = ''
        """
        # Public field: property_identifier; Private field: private_property_identifier (skip)
        name_node = self._get_child_by_type(node, "property_identifier")
        if not name_node:
            return False  # private field (#name) or computed — skip
        name = self._txt(name_node, src)

        # Skip single-underscore convention private fields
        if name.startswith("_"):
            return False

        body = self._txt(node, src)
        if len(body) > 300:
            body = body[:300] + "..."

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=f"{class_name}.{name}",
                kind=SymbolKind.VARIABLE,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=body.split("\n")[0][:200],
                body=body,
                parent_id=class_local_idx,
                is_public=True,
            )
        )
        return True

    # ------------------------------------------------------------------
    # Function handling
    # ------------------------------------------------------------------

    def _handle_function(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int, str, int]],
        exported: bool = False,
        decorators: list[str] | None = None,
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        func_local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.FUNCTION,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=self._func_signature(node, src),
                body=self._txt(node, src),
                docstring=self._get_preceding_comment(node, src),
                decorators=decorators or [],
                is_public=exported,
            )
        )

        # Walk body for call edges
        body_node = self._get_child_by_type(node, "statement_block")
        if body_node:
            self._walk_body(body_node, src, func_local_idx, raw_calls)

    def _handle_var_decl(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        raw_calls: list[tuple[int, str, int]],
        exported: bool = False,
    ) -> None:
        """Handle const/let declarations — arrow functions AND module-level constants."""
        for child in node.children:
            if child.type != "variable_declarator":
                continue

            name_node = self._get_child_by_type(child, "identifier")

            # Find function-like value and detect require()
            fn_node = None
            require_path: str | None = None
            for c in child.children:
                if c.type in ("arrow_function", "function", "function_expression"):
                    fn_node = c
                    break
                elif c.type == "call_expression":
                    # Destructured or direct: const x = require('./mod')
                    fn_r = c.child_by_field_name("function")
                    if fn_r and fn_r.type == "identifier" and self._txt(fn_r, src) == "require":
                        args = c.child_by_field_name("arguments")
                        if args:
                            for arg in args.children:
                                if arg.type == "string":
                                    require_path = self._txt(arg, src).strip("'\"")
                                    break

            if require_path:
                import_edges.append(
                    ImportEdge(
                        file_id=file_id,
                        imported_from=require_path,
                        imported_names=[],
                    )
                )

            if name_node and fn_node:
                name = self._txt(name_node, src)
                func_local_idx = len(symbols)
                symbols.append(
                    Symbol(
                        file_id=file_id,
                        name=name,
                        qualified_name=name,
                        kind=SymbolKind.FUNCTION,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        signature=self._arrow_signature(name_node, fn_node, src),
                        body=self._txt(child, src),
                        docstring=self._get_preceding_comment(node, src),
                        is_public=exported,
                    )
                )
                # Walk arrow/function body for call edges
                body_node = fn_node.child_by_field_name("body")
                if body_node and body_node.type == "statement_block":
                    self._walk_body(body_node, src, func_local_idx, raw_calls)
            elif name_node:
                name = self._txt(name_node, src)
                if exported or self._is_constant_name(name):
                    body = self._txt(child, src)
                    if len(body) > 500:
                        body = body[:500] + "..."
                    symbols.append(
                        Symbol(
                            file_id=file_id,
                            name=name,
                            qualified_name=name,
                            kind=SymbolKind.CONSTANT,
                            line_start=child.start_point[0] + 1,
                            line_end=child.end_point[0] + 1,
                            signature=body.split("\n")[0][:200],
                            body=body,
                            is_public=exported,
                        )
                    )

    def _handle_expression_stmt(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
    ) -> None:
        """Detect CommonJS require() calls and module.exports assignments."""
        for child in node.children:
            if child.type == "call_expression":
                # CommonJS: require('./module')
                fn_node = child.child_by_field_name("function")
                if (
                    fn_node
                    and fn_node.type == "identifier"
                    and self._txt(fn_node, src) == "require"
                ):
                    args_node = child.child_by_field_name("arguments")
                    if args_node:
                        for arg in args_node.children:
                            if arg.type == "string":
                                module_name = self._txt(arg, src).strip("'\"")
                                import_edges.append(
                                    ImportEdge(
                                        file_id=file_id,
                                        imported_from=module_name,
                                        imported_names=[],
                                    )
                                )

            elif child.type == "assignment_expression":
                # CommonJS: module.exports = { ... } or module.exports = SomeClass
                left = child.child_by_field_name("left")
                right = child.child_by_field_name("right")
                if not left or not right:
                    continue
                if left.type != "member_expression":
                    continue
                obj = left.child_by_field_name("object")
                prop = left.child_by_field_name("property")
                if not obj or not prop:
                    continue
                if self._txt(obj, src) != "module" or self._txt(prop, src) != "exports":
                    continue
                # module.exports = <rhs>
                body = self._txt(right, src)
                if len(body) > 500:
                    body = body[:500] + "..."
                symbols.append(
                    Symbol(
                        file_id=file_id,
                        name="module.exports",
                        qualified_name="module.exports",
                        kind=SymbolKind.CONSTANT,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        signature="module.exports = ...",
                        body=body,
                        is_public=True,
                    )
                )

    # ------------------------------------------------------------------
    # Call extraction
    # ------------------------------------------------------------------

    def _walk_body(
        self,
        node: Node,
        src: bytes,
        current_func_local_idx: int,
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        """Recursively walk a function/method body collecting call_expression nodes.

        Does NOT recurse into nested function/class definitions — they have their
        own scope and their calls should be attributed to their own symbol.
        """
        for child in node.children:
            if child.type == "call_expression":
                fn_node = child.child_by_field_name("function")
                if fn_node:
                    if fn_node.type == "identifier":
                        raw_calls.append(
                            (
                                current_func_local_idx,
                                self._txt(fn_node, src),
                                child.start_point[0] + 1,
                            )
                        )
                    elif fn_node.type == "member_expression":
                        prop = fn_node.child_by_field_name("property")
                        if prop:
                            raw_calls.append(
                                (
                                    current_func_local_idx,
                                    self._txt(prop, src),
                                    child.start_point[0] + 1,
                                )
                            )
                # Recurse into call args for nested calls (foo(bar()))
                self._walk_body(child, src, current_func_local_idx, raw_calls)
            elif child.type == "new_expression":
                # new SomeClass(...) or new pkg.Class(...)
                ctor_node = child.child_by_field_name("constructor")
                if ctor_node:
                    if ctor_node.type == "identifier":
                        raw_calls.append(
                            (
                                current_func_local_idx,
                                self._txt(ctor_node, src),
                                child.start_point[0] + 1,
                            )
                        )
                    elif ctor_node.type == "member_expression":
                        prop = ctor_node.child_by_field_name("property")
                        if prop:
                            raw_calls.append(
                                (
                                    current_func_local_idx,
                                    self._txt(prop, src),
                                    child.start_point[0] + 1,
                                )
                            )
                self._walk_body(child, src, current_func_local_idx, raw_calls)
            elif child.type not in (
                "function_declaration",
                "generator_function_declaration",
                "arrow_function",
                "function_expression",
                "generator_function",
                "class_declaration",
                "class_expression",
            ):
                self._walk_body(child, src, current_func_local_idx, raw_calls)

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_import(self, node: Node, src: bytes, file_id: int) -> list[ImportEdge]:
        """Handle ES6 import_statement → ImportEdge."""
        source_node: Node | None = None
        imported_names: list[str] = []

        for child in node.children:
            if child.type == "string":
                source_node = child
            elif child.type == "import_clause":
                imported_names = self._extract_import_clause(child, src)

        if source_node is None:
            return []

        module_name = self._txt(source_node, src).strip("'\"")
        return [
            ImportEdge(
                file_id=file_id,
                imported_from=module_name,
                imported_names=imported_names,
            )
        ]

    def _extract_import_clause(self, node: Node, src: bytes) -> list[str]:
        names: list[str] = []
        for child in node.children:
            if child.type == "identifier":
                names.append(self._txt(child, src))
            elif child.type == "named_imports":
                for spec in child.children:
                    if spec.type == "import_specifier":
                        id_node = self._get_child_by_type(spec, "identifier")
                        if id_node:
                            names.append(self._txt(id_node, src))
            elif child.type == "namespace_import":
                id_node = self._get_child_by_type(child, "identifier")
                if id_node:
                    names.append(self._txt(id_node, src))
        return names

    # ------------------------------------------------------------------
    # Signature builders
    # ------------------------------------------------------------------

    def _func_signature(self, node: Node, src: bytes) -> str:
        name_node = self._get_child_by_type(node, "identifier")
        params_node = node.child_by_field_name("parameters")

        name = self._txt(name_node, src) if name_node else "?"
        params = self._txt(params_node, src) if params_node else "()"
        prefix = "function* " if node.type == "generator_function_declaration" else "function "
        return f"{prefix}{name}{params}"

    def _method_signature(self, node: Node, src: bytes, class_name: str) -> str:
        name_node = self._get_child_by_type(node, "property_identifier")
        params_node = node.child_by_field_name("parameters")

        name = self._txt(name_node, src) if name_node else "?"
        params = self._txt(params_node, src) if params_node else "()"
        return f"{class_name}.{name}{params}"

    def _arrow_signature(self, name_node: Node, fn_node: Node, src: bytes) -> str:
        name = self._txt(name_node, src)
        params_node = fn_node.child_by_field_name("parameters") or self._get_child_by_type(
            fn_node, "formal_parameters"
        )
        params = self._txt(params_node, src) if params_node else "(...)"
        return f"const {name} = {params} => ..."

    def _class_signature(self, node: Node, src: bytes) -> str:
        name_node = self._get_child_by_type(node, "identifier")
        heritage_node = self._get_child_by_type(node, "class_heritage")

        name = self._txt(name_node, src) if name_node else "?"
        heritage = f" {self._txt(heritage_node, src)}" if heritage_node else ""
        return f"class {name}{heritage}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_decorator_text(self, node: Node, src: bytes) -> str:
        """Extract human-readable decorator text from a decorator node."""
        for child in node.children:
            if child.type in ("identifier", "call_expression", "member_expression"):
                text = self._txt(child, src)
                return text[:200] if len(text) <= 200 else text[:200] + "..."
        text = self._txt(node, src).lstrip("@").strip()
        return text[:200] if len(text) <= 200 else text[:200] + "..."

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

    @staticmethod
    def _is_constant_name(name: str) -> bool:
        """True for ALL_CAPS names and __dunder__ names."""
        if re.match(r"^_*[A-Z][A-Z0-9_]*$", name):
            return True
        if name.startswith("__") and name.endswith("__") and len(name) > 4:
            return True
        return False

    def _get_module_docstring(self, root: Node, src: bytes) -> str | None:
        """Return the leading file-level JSDoc/block comment if present at top of file."""
        for child in root.children:
            if child.type == "comment":
                text = self._txt(child, src)
                # Only block comments (/** ... */ or /* ... */) at line 0
                if text.startswith("/*") and child.start_point[0] == 0:
                    cleaned = self._clean_comment(text)
                    return cleaned if cleaned else None
            else:
                break  # Stop at first non-comment node
        return None

    def _get_preceding_comment(self, node: Node, src: bytes) -> str | None:
        """Find doc comment immediately before this node (or its export wrapper)."""
        start = node
        if start.parent is not None and start.parent.type == "export_statement":
            start = start.parent
        lines: list[str] = []
        prev = start.prev_named_sibling
        next_start_line = start.start_point[0]
        while prev is not None and prev.type == "comment":
            if prev.end_point[0] + 1 < next_start_line:
                break
            lines.insert(0, self._txt(prev, src))
            next_start_line = prev.start_point[0]
            prev = prev.prev_named_sibling
        return self._clean_comment("\n".join(lines)) if lines else None

    @staticmethod
    def _clean_comment(raw: str) -> str:
        """Strip comment delimiters: /** */, //, ///, * line prefixes."""
        raw = re.sub(r"^/\*+\s*", "", raw.strip())
        raw = re.sub(r"\s*\*+/$", "", raw)
        raw = re.sub(r"^\s*\*\s?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"^///?\s?", "", raw, flags=re.MULTILINE)
        return raw.strip()
