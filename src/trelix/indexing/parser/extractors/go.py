"""Go parser — direct AST traversal using Tree-sitter.

Extracts:
  - Functions (kind=FUNCTION) — top-level func declarations (with return types, generics)
  - Methods (kind=METHOD, parent_id=struct local idx) — func with receiver
  - Structs (kind=CLASS) — type Foo struct {}
  - Interfaces (kind=INTERFACE) — type Foo interface {}
  - Type aliases / named types (kind=INTERFACE) — type Foo = Bar, type MyInt int
  - Exported struct fields (kind=VARIABLE) — inside struct bodies
  - Interface method specs (kind=FUNCTION) — inside interface bodies
  - Exported package-level constants and vars (kind=CONSTANT)
  - Module docstring — package comment → MODULE symbol
  - Imports → ImportEdge
  - Call edges → CallEdge (function and method calls)

Parent linkage:
  Methods reference their receiver struct by qualified_name matching (best-effort).
  parent_id is set to the local index of the struct symbol if found, else None.
  The Indexer remaps local indices to real DB ids after insertion.
"""

from __future__ import annotations

import re

from tree_sitter import Node

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge
from trelix.indexing.parser._grammar import load_language, make_parser
from trelix.indexing.parser.base import BaseParser, ParseResult

MAX_STRUCT_FIELDS = 30
MAX_INTERFACE_METHODS = 20


class GoParser(BaseParser):
    """Tree-sitter based Go parser using direct AST traversal."""

    def __init__(self) -> None:
        self._ts_language = load_language("go")
        self._parser = make_parser("go")

    @property
    def language_name(self) -> str:
        return "go"

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
        raw_calls: list[tuple[int, str, int]] = []  # (caller_local_idx, callee_name, line)

        # Module docstring from package doc comment
        module_sym = self._get_module_symbol(root, source_bytes, file_id)
        if module_sym:
            symbols.append(module_sym)

        # Build name→local_idx map so methods can reference their receiver struct
        struct_idx: dict[str, int] = {}

        self._walk_top_level(
            root, source_bytes, file_id, symbols, import_edges, struct_idx, raw_calls, type_edges
        )

        call_edges = [
            CallEdge(caller_id=caller, callee_name=callee, line=line)
            for caller, callee, line in raw_calls
        ]

        return ParseResult(
            symbols=symbols,
            call_edges=call_edges,
            import_edges=import_edges,
            parse_errors=self._count_errors(root),
            type_edges=type_edges,
        )

    # ------------------------------------------------------------------
    # Module symbol
    # ------------------------------------------------------------------

    def _get_module_symbol(self, root: Node, src: bytes, file_id: int) -> Symbol | None:
        """Extract package-level doc comment as a MODULE symbol."""
        pkg_node = None
        for child in root.children:
            if child.type == "package_clause":
                pkg_node = child
                break
        if not pkg_node:
            return None

        pkg_name_node = pkg_node.child_by_field_name("name")
        pkg_name = self._txt(pkg_name_node, src) if pkg_name_node else ""

        docstring = self._get_preceding_comment(pkg_node, src)
        if not docstring:
            return None

        return Symbol(
            file_id=file_id,
            name=pkg_name or "package",
            qualified_name=pkg_name or "package",
            kind=SymbolKind.MODULE,
            line_start=1,
            line_end=root.end_point[0] + 1,
            signature=f"package {pkg_name}",
            body=docstring,
            docstring=docstring,
            is_public=True,
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
        struct_idx: dict[str, int],
        raw_calls: list[tuple[int, str, int]],
        type_edges: list[TypeEdge],
    ) -> None:
        for child in node.children:
            ntype = child.type
            if ntype == "import_declaration":
                import_edges.extend(self._extract_import(child, src, file_id))
            elif ntype == "function_declaration":
                self._handle_function(child, src, file_id, symbols, raw_calls)
            elif ntype == "method_declaration":
                self._handle_method(child, src, file_id, symbols, struct_idx, raw_calls)
            elif ntype == "type_declaration":
                self._handle_type_decl(child, src, file_id, symbols, struct_idx, type_edges)
            elif ntype == "const_declaration":
                self._handle_const_decl(child, src, file_id, symbols)
            elif ntype == "var_declaration":
                self._handle_var_decl(child, src, file_id, symbols)

    # ------------------------------------------------------------------
    # Type declarations (struct / interface / alias / named type)
    # ------------------------------------------------------------------

    def _handle_type_decl(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        struct_idx: dict[str, int],
        type_edges: list[TypeEdge],
    ) -> None:
        """Handle: type Foo struct{} / type Foo interface{} / type Foo = Bar / type MyInt int"""
        for spec in node.children:
            if spec.type == "type_spec":
                self._handle_type_spec(
                    spec, node, src, file_id, symbols, struct_idx, type_edges, is_alias=False
                )
            elif spec.type == "type_alias":
                self._handle_type_spec(
                    spec, node, src, file_id, symbols, struct_idx, type_edges, is_alias=True
                )

    def _handle_type_spec(
        self,
        spec: Node,
        decl_node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        struct_idx: dict[str, int],
        type_edges: list[TypeEdge],
        is_alias: bool,
    ) -> None:
        name_node = spec.child_by_field_name("name")
        if not name_node:
            return
        name = self._txt(name_node, src)
        if not name:
            return

        type_node = spec.child_by_field_name("type")
        if not type_node:
            return

        is_public = bool(name) and name[0].isupper()
        docstring = self._get_preceding_comment(decl_node, src)

        if type_node.type == "struct_type":
            local_idx = len(symbols)
            sym = Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.CLASS,
                line_start=decl_node.start_point[0] + 1,
                line_end=decl_node.end_point[0] + 1,
                signature=self._type_signature(spec, src, "struct"),
                body=self._txt(decl_node, src),
                docstring=docstring,
                is_public=is_public,
            )
            symbols.append(sym)
            struct_idx[name] = local_idx
            self._extract_struct_fields(type_node, src, file_id, symbols, local_idx, type_edges)

        elif type_node.type == "interface_type":
            local_idx = len(symbols)
            sym = Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.INTERFACE,
                line_start=decl_node.start_point[0] + 1,
                line_end=decl_node.end_point[0] + 1,
                signature=self._type_signature(spec, src, "interface"),
                body=self._txt(decl_node, src),
                docstring=docstring,
                is_public=is_public,
            )
            symbols.append(sym)
            struct_idx[name] = local_idx
            self._extract_interface_methods(type_node, src, file_id, symbols, local_idx, type_edges)

        else:
            # Type alias (type Foo = Bar) or named type (type MyInt int)
            eq = " = " if is_alias else " "
            sym = Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.INTERFACE,
                line_start=decl_node.start_point[0] + 1,
                line_end=decl_node.end_point[0] + 1,
                signature=f"type {name}{eq}{self._txt(type_node, src)}",
                body=self._txt(decl_node, src),
                docstring=docstring,
                is_public=is_public,
            )
            symbols.append(sym)

    def _extract_struct_fields(
        self,
        struct_node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        parent_local_idx: int,
        type_edges: list[TypeEdge],
    ) -> None:
        """Extract exported fields from a struct_type node."""
        field_list = None
        for child in struct_node.children:
            if child.type == "field_declaration_list":
                field_list = child
                break
        if not field_list:
            return

        field_count = 0
        for child in field_list.children:
            if child.type != "field_declaration" or field_count >= MAX_STRUCT_FIELDS:
                continue

            type_node = child.child_by_field_name("type")
            type_str = self._txt(type_node, src) if type_node else ""

            # All field_identifier children are field names
            name_nodes = [c for c in child.children if c.type == "field_identifier"]
            if not name_nodes:
                # Embedded field (anonymous field) — e.g. sync.Mutex or http.Handler
                # The field name is the unqualified type name; emit a TypeEdge for graph expansion
                if type_node:
                    embedded_type = self._unqualified_type_name(type_node, src)
                    if embedded_type:
                        type_edges.append(
                            TypeEdge(
                                from_symbol_id=parent_local_idx,
                                to_type_name=embedded_type,
                                edge_kind="embedded",
                            )
                        )
                continue

            for name_node in name_nodes:
                name = self._txt(name_node, src)
                if not name or not name[0].isupper():
                    continue  # skip unexported fields
                symbols.append(
                    Symbol(
                        file_id=file_id,
                        name=name,
                        qualified_name=name,
                        kind=SymbolKind.VARIABLE,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        signature=f"{name} {type_str}",
                        body=self._txt(child, src),
                        parent_id=parent_local_idx,
                        is_public=True,
                    )
                )
                field_count += 1

    def _extract_interface_methods(
        self,
        iface_node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        parent_local_idx: int,
        type_edges: list[TypeEdge],
    ) -> None:
        """Extract method specs and embedded types from an interface_type node."""
        method_count = 0
        for child in iface_node.children:
            if method_count >= MAX_INTERFACE_METHODS:
                break
            if child.type == "method_elem":
                # Interface method: name is a field_identifier child
                name_node = self._get_child_by_type(child, "field_identifier")
                if not name_node:
                    continue
                name = self._txt(name_node, src)
                # parameter_list children: first is params, second (if any) is result
                param_lists = [c for c in child.children if c.type == "parameter_list"]
                params_str = self._txt(param_lists[0], src) if param_lists else "()"
                result_str = (" " + self._txt(param_lists[1], src)) if len(param_lists) > 1 else ""
                symbols.append(
                    Symbol(
                        file_id=file_id,
                        name=name,
                        qualified_name=name,
                        kind=SymbolKind.FUNCTION,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        signature=f"{name}{params_str}{result_str}",
                        body=self._txt(child, src),
                        parent_id=parent_local_idx,
                        is_public=name[0].isupper() if name else False,
                    )
                )
                method_count += 1
            elif child.type in ("constraint_elem", "type_elem"):
                # Embedded interface or type constraint: io.Reader, int | string, ~int
                # Emit TypeEdges for simple embedded interface names only
                type_text = self._txt(child, src).strip()
                if "|" not in type_text and not type_text.startswith("~"):
                    embedded = self._unqualified_name(type_text)
                    if embedded:
                        type_edges.append(
                            TypeEdge(
                                from_symbol_id=parent_local_idx,
                                to_type_name=embedded,
                                edge_kind="embedded",
                            )
                        )

    # ------------------------------------------------------------------
    # Function / method handling
    # ------------------------------------------------------------------

    def _handle_function(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        name_node = node.child_by_field_name("name")
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
                is_public=name[0].isupper() if name else False,
            )
        )

        body_node = node.child_by_field_name("body")
        if body_node:
            self._walk_body(body_node, func_local_idx, raw_calls, src)

    def _handle_method(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        struct_idx: dict[str, int],
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        """Handle: func (s *Server) MethodName(...) {}"""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = self._txt(name_node, src)

        receiver_type = ""
        receiver_node = node.child_by_field_name("receiver")
        if receiver_node:
            receiver_type = self._get_receiver_type(receiver_node, src)

        qualified_name = f"{receiver_type}.{name}" if receiver_type else name
        parent_id = struct_idx.get(receiver_type)

        func_local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=qualified_name,
                kind=SymbolKind.METHOD,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=self._method_signature(node, src, receiver_type),
                body=self._txt(node, src),
                parent_id=parent_id,
                docstring=self._get_preceding_comment(node, src),
                is_public=name[0].isupper() if name else False,
            )
        )

        body_node = node.child_by_field_name("body")
        if body_node:
            self._walk_body(body_node, func_local_idx, raw_calls, src)

    def _get_receiver_type(self, receiver_list: Node, src: bytes) -> str:
        """Extract the type name from a Go receiver parameter_list like (s *Server)."""
        for child in receiver_list.children:
            if child.type == "parameter_declaration":
                for sub in child.children:
                    if sub.type == "type_identifier":
                        return self._txt(sub, src)
                    elif sub.type == "pointer_type":
                        ti = self._get_child_by_type(sub, "type_identifier")
                        if ti:
                            return self._txt(ti, src)
        return ""

    # ------------------------------------------------------------------
    # Call graph
    # ------------------------------------------------------------------

    def _walk_body(
        self,
        node: Node,
        caller_local_idx: int,
        raw_calls: list[tuple[int, str, int]],
        src: bytes,
    ) -> None:
        """Collect call_expression nodes within a function/method body.

        Skips func_literal (anonymous function) scopes — they are not attributed
        to the enclosing named function.
        """
        for child in node.children:
            if child.type == "func_literal":
                continue  # anonymous func has its own scope
            if child.type == "call_expression":
                func_node = child.child_by_field_name("function")
                if func_node:
                    if func_node.type == "identifier":
                        raw_calls.append(
                            (
                                caller_local_idx,
                                self._txt(func_node, src),
                                child.start_point[0] + 1,
                            )
                        )
                    elif func_node.type == "selector_expression":
                        # e.g., s.Method() or pkg.Func()
                        field_node = func_node.child_by_field_name("field")
                        if field_node:
                            raw_calls.append(
                                (
                                    caller_local_idx,
                                    self._txt(field_node, src),
                                    child.start_point[0] + 1,
                                )
                            )
            self._walk_body(child, caller_local_idx, raw_calls, src)

    # ------------------------------------------------------------------
    # Constant / variable extraction
    # ------------------------------------------------------------------

    def _handle_const_decl(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
    ) -> None:
        """Handle: const X = 1  or  const ( X = 1; Y = 2 )"""
        for child in node.children:
            if child.type == "const_spec":
                self._extract_go_spec(child, src, file_id, symbols, keyword="const")

    def _handle_var_decl(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
    ) -> None:
        """Handle: var X = 1  or  var ( X = 1; Y = 2 )"""
        for child in node.children:
            if child.type == "var_spec":
                self._extract_go_spec(child, src, file_id, symbols, keyword="var")
            elif child.type == "var_spec_list":
                # Grouped var ( A = 1; B = 2 ) — specs wrapped in var_spec_list
                for spec in child.children:
                    if spec.type == "var_spec":
                        self._extract_go_spec(spec, src, file_id, symbols, keyword="var")

    def _extract_go_spec(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        keyword: str,
    ) -> None:
        """Shared extractor for const_spec and var_spec nodes. Handles multi-name specs."""
        # Collect all names: `const X, Y = 1, 2` → multiple identifier children directly
        # `var A, B int` → same pattern; identifier_list only appears in := statements
        id_list = self._get_child_by_type(node, "identifier_list")
        if id_list:
            name_nodes = [c for c in id_list.children if c.type == "identifier"]
        else:
            name_nodes = [c for c in node.children if c.type == "identifier"]

        body = self._txt(node, src)
        if len(body) > 500:
            body = body[:500] + "..."
        sig_line = f"{keyword} {body.split(chr(10))[0][:200]}"

        for name_node in name_nodes:
            name = self._txt(name_node, src)
            if not name or not name[0].isupper():
                continue  # only exported package-level symbols
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.CONSTANT,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=sig_line,
                    body=body,
                )
            )

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_import(self, node: Node, src: bytes, file_id: int) -> list[ImportEdge]:
        """Handle both single and grouped import declarations."""
        edges: list[ImportEdge] = []

        for child in node.children:
            if child.type == "import_spec_list":
                for spec in child.children:
                    if spec.type == "import_spec":
                        path = self._get_import_path(spec, src)
                        if path:
                            edges.append(
                                ImportEdge(
                                    file_id=file_id,
                                    imported_from=path,
                                    imported_names=[],
                                )
                            )
            elif child.type == "import_spec":
                path = self._get_import_path(child, src)
                if path:
                    edges.append(
                        ImportEdge(
                            file_id=file_id,
                            imported_from=path,
                            imported_names=[],
                        )
                    )

        return edges

    def _get_import_path(self, spec: Node, src: bytes) -> str:
        """Extract the string path from an import_spec node."""
        for child in spec.children:
            if child.type == "interpreted_string_literal":
                return self._txt(child, src).strip('"')
        return ""

    # ------------------------------------------------------------------
    # Signature builders
    # ------------------------------------------------------------------

    def _type_signature(self, spec: Node, src: bytes, kind_str: str) -> str:
        """Build: type Foo[T any] struct  or  type Bar interface"""
        name_node = spec.child_by_field_name("name")
        name = self._txt(name_node, src) if name_node else "?"
        tp_node = spec.child_by_field_name("type_parameters")
        tp_str = self._txt(tp_node, src) if tp_node else ""
        return f"type {name}{tp_str} {kind_str}"

    def _func_signature(self, node: Node, src: bytes) -> str:
        """Build: func Name[T any](params) result"""
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        result_node = node.child_by_field_name("result")
        tp_node = node.child_by_field_name("type_parameters")

        name = self._txt(name_node, src) if name_node else "?"
        tp_str = self._txt(tp_node, src) if tp_node else ""
        params = self._txt(params_node, src) if params_node else "()"
        result = (" " + self._txt(result_node, src)) if result_node else ""
        return f"func {name}{tp_str}{params}{result}"

    def _method_signature(self, node: Node, src: bytes, receiver_type: str) -> str:
        """Build: func (ReceiverType) Name(params) result"""
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        result_node = node.child_by_field_name("result")

        name = self._txt(name_node, src) if name_node else "?"
        params = self._txt(params_node, src) if params_node else "()"
        result = (" " + self._txt(result_node, src)) if result_node else ""
        recv = f"({receiver_type}) " if receiver_type else ""
        return f"func {recv}{name}{params}{result}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _unqualified_type_name(self, type_node: Node, src: bytes) -> str:
        """Get the unqualified name from a type node (strips pointer and package prefix)."""
        # Unwrap pointer: *sync.Mutex → sync.Mutex
        inner = type_node
        if inner.type == "pointer_type":
            for c in inner.children:
                if c.type not in ("*",):
                    inner = c
                    break
        return self._unqualified_name(self._txt(inner, src).strip())

    @staticmethod
    def _unqualified_name(type_str: str) -> str:
        """Strip package prefix: 'sync.Mutex' → 'Mutex', 'io.Reader' → 'Reader'."""
        # Take last dot-separated component, strip pointer/brackets
        name = type_str.lstrip("*").split("[")[0].strip()
        return name.split(".")[-1].strip() if name else ""

    def _get_preceding_comment(self, node: Node, src: bytes) -> str | None:
        """Collect consecutive Go doc comment lines immediately before this node."""
        lines: list[str] = []
        prev = node.prev_named_sibling
        next_start_line = node.start_point[0]
        while prev is not None and prev.type == "comment":
            if prev.end_point[0] + 1 < next_start_line:
                break
            lines.insert(0, self._txt(prev, src))
            next_start_line = prev.start_point[0]
            prev = prev.prev_named_sibling
        return self._clean_comment("\n".join(lines)) if lines else None

    @staticmethod
    def _clean_comment(raw: str) -> str:
        """Strip comment delimiters: /* */, //, /// prefixes."""
        raw = re.sub(r"^/\*+\s*", "", raw.strip())
        raw = re.sub(r"\s*\*+/$", "", raw)
        raw = re.sub(r"^\s*\*\s?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"^///?\s?", "", raw, flags=re.MULTILINE)
        return raw.strip()
