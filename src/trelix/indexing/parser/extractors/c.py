"""C parser — direct AST traversal using Tree-sitter.

Handles .c and .h files. Extracts:
  - Top-level functions (kind=FUNCTION)
  - Function declarations (treated as FUNCTION)
  - Structs (kind=STRUCT) — with member fields
  - Unions (kind=STRUCT) — treated like structs
  - Enums (kind=ENUM) + enum members as CONSTANT
  - Typedefs (kind=CLASS when complex typedef'd structs/unions)
  - Global variables (kind=VARIABLE) — only top-level, named constants
  - #include directives → ImportEdge (both <system> and "local")
  - #define directives → CONSTANT symbols
  - Function calls → CallEdge
  - Struct inheritance (typedef'd structs) → TypeEdge

Design notes:
  - C structs use field_declaration_list for members (similar to C++)
  - Enums are handled like C++ enums with enumerator_list
  - #define directives are treated as constants
  - Call graph built from call_expression nodes
  - Parent linkage uses local indices during parse, remapped by Indexer
  - Function declarations vs definitions both extracted as FUNCTION
  - typedef handling preserves the original type information

Parent linkage:
  parent_id in Symbol is set to the LOCAL INDEX in the symbols list during
  parsing. The Indexer remaps this to the actual DB id after insertion.
"""

from __future__ import annotations

import tree_sitter_languages
from tree_sitter import Node, Parser

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge
from trelix.indexing.parser.base import BaseParser, ParseResult


class CParser(BaseParser):
    """Tree-sitter based C parser using direct AST traversal."""

    # Cap on member fields extracted per struct (prevents symbol flood)
    MAX_STRUCT_FIELDS: int = 30

    def __init__(self) -> None:
        self._ts_lang = tree_sitter_languages.get_language("c")
        self._parser = Parser()
        self._parser.set_language(self._ts_lang)

    @property
    def language_name(self) -> str:
        return "c"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def parse(self, source: str, file_id: int) -> ParseResult:
        """Parse C source code and extract symbols, relationships."""
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        symbols: list[Symbol] = []
        raw_calls: list[tuple[int | None, str, int]] = []
        import_edges: list[ImportEdge] = []
        type_edges: list[TypeEdge] = []

        # Walk the translation unit (root)
        self._walk(
            node=root,
            src=source_bytes,
            file_id=file_id,
            symbols=symbols,
            raw_calls=raw_calls,
            import_edges=import_edges,
            type_edges=type_edges,
            parent_struct_local_idx=None,
            current_func_local_idx=None,
            depth=0,
        )

        # Build CallEdge list — caller_id is local index, remapped by Indexer
        call_edges: list[CallEdge] = [
            CallEdge(caller_id=caller_idx, callee_name=name, line=line)
            for caller_idx, name, line in raw_calls
            if caller_idx is not None
        ]

        return ParseResult(
            symbols=symbols,
            call_edges=call_edges,
            import_edges=import_edges,
            parse_errors=self._count_errors(root),
            type_edges=type_edges,
        )

    # ------------------------------------------------------------------
    # AST walk
    # ------------------------------------------------------------------

    def _walk(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        parent_struct_local_idx: int | None,
        current_func_local_idx: int | None,
        depth: int,
    ) -> None:
        """Recursive depth-first walk of C AST."""
        if depth > 20:
            return

        for child in node.children:
            ntype = child.type

            # ---- Preprocessor #include directives → ImportEdge ----
            if ntype == "preproc_include":
                import_edges.extend(self._extract_include(child, src, file_id))

            # ---- Preprocessor #define directives → CONSTANT symbols ----
            elif ntype == "preproc_def":
                self._handle_preproc_define(child, src, file_id, symbols)

            # ---- Struct declaration ----
            elif ntype == "struct_specifier" and parent_struct_local_idx is None:
                self._handle_struct(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    depth,
                    kind=SymbolKind.STRUCT,
                )

            # ---- Union declaration ----
            elif ntype == "union_specifier" and parent_struct_local_idx is None:
                self._handle_struct(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    depth,
                    kind=SymbolKind.STRUCT,
                )

            # ---- Enum declaration ----
            elif ntype == "enum_specifier" and parent_struct_local_idx is None:
                self._handle_enum(child, src, file_id, symbols, type_edges)

            # ---- Function definition ----
            elif ntype == "function_definition" and parent_struct_local_idx is None:
                self._handle_function(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    depth,
                )

            # ---- Declaration (function declarations, variable declarations) ----
            elif ntype == "declaration" and parent_struct_local_idx is None:
                self._handle_declaration(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    depth,
                )

            # ---- Typedef (type_definition node in tree-sitter-c 0.24+) ----
            elif ntype == "type_definition" and parent_struct_local_idx is None:
                self._handle_typedef(child, src, file_id, symbols)

            # ---- Call sites (track for call graph) ----
            elif ntype == "call_expression":
                self._handle_call(child, src, raw_calls, current_func_local_idx)
                self._walk(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_struct_local_idx,
                    current_func_local_idx,
                    depth + 1,
                )

            # ---- Recurse into statement / expression containers ----
            elif ntype in (
                "compound_statement",
                "if_statement",
                "else_clause",
                "for_statement",
                "while_statement",
                "do_statement",
                "return_statement",
                "expression_statement",
                "switch_statement",
                "case_statement",
                "default_statement",
                # expressions that may contain calls
                "binary_expression",
                "unary_expression",
                "call_expression",
                "conditional_expression",
                "assignment_expression",
                "comma_expression",
                "parenthesized_expression",
                "subscript_expression",
                "field_expression",
                "pointer_expression",
                "initializer_list",
                "cast_expression",
                "sizeof_expression",
                # array/struct initializers
                "initializer_pair",
                # declaration containers
                "declaration",
                "block_item",
            ):
                self._walk(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_struct_local_idx,
                    current_func_local_idx,
                    depth + 1,
                )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_struct(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        depth: int,
        kind: SymbolKind = SymbolKind.STRUCT,
    ) -> None:
        """Extract struct or union declarations."""
        name_node = self._get_child_by_type(node, "type_identifier")
        if not name_node:
            name_node = node.child_by_field_name("name")
        if not name_node:
            # Anonymous struct/union — skip
            return

        name = self._txt(name_node, src)
        if not name:
            return

        struct_local_idx = len(symbols)
        signature = f"{kind.value} {name}"
        docstring = self._get_preceding_comment(node, src)

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=kind,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=signature[:300],
                body=self._txt(node, src)[:2000],
                docstring=docstring,
                is_public=True,
            )
        )

        # Walk struct body for members
        body_node = self._get_child_by_type(node, "field_declaration_list")
        if body_node:
            self._walk_struct_body(
                body_node,
                src,
                file_id,
                symbols,
                raw_calls,
                import_edges,
                type_edges,
                struct_local_idx,
                name,
                depth,
            )

    def _walk_struct_body(
        self,
        body_node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        struct_local_idx: int,
        struct_name: str,
        depth: int,
    ) -> None:
        """Walk struct body and extract members."""
        field_count = 0
        for child in body_node.children:
            ntype = child.type

            if ntype == "field_declaration" and field_count < self.MAX_STRUCT_FIELDS:
                before = len(symbols)
                self._handle_field_declaration(
                    child,
                    src,
                    file_id,
                    symbols,
                    struct_local_idx,
                    struct_name,
                )
                field_count += len(symbols) - before

            elif ntype == "struct_specifier":
                self._handle_struct(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    depth + 1,
                    kind=SymbolKind.STRUCT,
                )

            elif ntype == "union_specifier":
                self._handle_struct(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    depth + 1,
                    kind=SymbolKind.STRUCT,
                )

            elif ntype == "enum_specifier":
                self._handle_enum(child, src, file_id, symbols, type_edges)

    def _handle_field_declaration(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        struct_local_idx: int,
        struct_name: str,
    ) -> None:
        """Extract struct field/member variable declarations."""
        for child in node.children:
            if child.type == "declarator":
                name = self._extract_declarator_name(child, src)
                if not name or name.startswith("_"):
                    continue

                field_sig = self._txt(node, src).split(";")[0][:200]

                symbols.append(
                    Symbol(
                        file_id=file_id,
                        name=name,
                        qualified_name=f"{struct_name}.{name}",
                        kind=SymbolKind.VARIABLE,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        signature=field_sig,
                        body=self._txt(node, src)[:500],
                        parent_id=struct_local_idx,
                        is_public=True,
                    )
                )

    def _handle_enum(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        type_edges: list[TypeEdge],
    ) -> None:
        """Extract enum declarations."""
        name_node = self._get_child_by_type(node, "type_identifier")
        if not name_node:
            name_node = node.child_by_field_name("name")
        if not name_node:
            return

        name = self._txt(name_node, src)
        if not name:
            return

        enum_local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.ENUM,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"enum {name}",
                body=self._txt(node, src)[:1000],
                is_public=True,
                docstring=self._get_preceding_comment(node, src),
            )
        )

        enumerator_list = self._get_child_by_type(node, "enumerator_list")
        if enumerator_list:
            for child in enumerator_list.children:
                if child.type == "enumerator":
                    member_node = self._get_child_by_type(child, "identifier")
                    if member_node:
                        member_name = self._txt(member_node, src)
                        symbols.append(
                            Symbol(
                                file_id=file_id,
                                name=member_name,
                                qualified_name=f"{name}::{member_name}",
                                kind=SymbolKind.CONSTANT,
                                line_start=child.start_point[0] + 1,
                                line_end=child.end_point[0] + 1,
                                signature=self._txt(child, src)[:100],
                                body=self._txt(child, src),
                                parent_id=enum_local_idx,
                                is_public=True,
                            )
                        )

    def _handle_function(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        depth: int,
    ) -> None:
        """Extract top-level function."""
        decl_node = node.child_by_field_name("declarator")
        if not decl_node:
            return

        name = self._extract_declarator_name(decl_node, src)
        if not name:
            return

        func_local_idx = len(symbols)
        signature = self._extract_function_signature(node, src)

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.FUNCTION,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=signature[:300],
                body=self._txt(node, src)[:2000],
                is_public=True,
                docstring=self._get_preceding_comment(node, src),
            )
        )

        body_node = node.child_by_field_name("body")
        if body_node:
            self._walk_for_calls(body_node, src, func_local_idx, raw_calls, depth + 1)

    def _handle_declaration(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        depth: int,
    ) -> None:
        """Handle declarations (function declarations, variable declarations, typedefs)."""
        declarator_node = node.child_by_field_name("declarator")

        is_typedef = False
        for child in node.children:
            if child.type == "typedef":
                is_typedef = True
                break

        if is_typedef and declarator_node:
            name = self._extract_declarator_name(declarator_node, src)
            if name:
                orig_type = ""
                for child in node.children:
                    if child.type in ("struct_specifier", "union_specifier", "enum_specifier"):
                        orig_type = self._txt(child, src).split("{")[0].strip()
                        break

                symbols.append(
                    Symbol(
                        file_id=file_id,
                        name=name,
                        qualified_name=name,
                        kind=SymbolKind.CLASS,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        signature=f"typedef {orig_type} {name}",
                        body=self._txt(node, src)[:500],
                        docstring=self._get_preceding_comment(node, src),
                        is_public=True,
                    )
                )

        elif declarator_node and not is_typedef:
            # Function declaration (no body)
            has_params = False
            for child in declarator_node.children:
                if child.type == "parameter_list":
                    has_params = True
                    break

            if has_params:
                name = self._extract_declarator_name(declarator_node, src)
                if name:
                    signature = self._extract_function_signature_from_declaration(node, src)

                    symbols.append(
                        Symbol(
                            file_id=file_id,
                            name=name,
                            qualified_name=name,
                            kind=SymbolKind.FUNCTION,
                            line_start=node.start_point[0] + 1,
                            line_end=node.end_point[0] + 1,
                            signature=signature[:300],
                            body=self._txt(node, src)[:500],
                            is_public=True,
                            docstring=self._get_preceding_comment(node, src),
                        )
                    )

    def _handle_typedef(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
    ) -> None:
        """
        Handle type_definition node (typedef in tree-sitter-c 0.24+).
        Structure: typedef <type_specifier> <type_identifier> ;
        Example: typedef struct { int x; } Point;
        """
        # The alias name is the type_identifier child immediately before the semicolon
        name_node = None
        for c in node.children:
            if c.type == "type_identifier":
                name_node = c  # keep updating; we want the last one (alias name)
        if not name_node:
            return
        name = self._txt(name_node, src)
        if not name:
            return

        # Extract original type description (struct/union/enum specifier or plain type)
        orig_type = ""
        for c in node.children:
            if c.type in (
                "struct_specifier",
                "union_specifier",
                "enum_specifier",
                "primitive_type",
                "sized_type_specifier",
            ):
                orig_type = self._txt(c, src).split("{")[0].strip()
                break

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.CLASS,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"typedef {orig_type} {name}".strip(),
                body=self._txt(node, src)[:500],
                docstring=self._get_preceding_comment(node, src),
                is_public=True,
            )
        )

    def _walk_for_calls(
        self,
        node: Node,
        src: bytes,
        func_local_idx: int,
        raw_calls: list[tuple[int | None, str, int]],
        depth: int,
    ) -> None:
        """Recursively walk function body looking for calls."""
        if depth > 15:
            return

        for child in node.children:
            if child.type == "call_expression":
                self._handle_call(child, src, raw_calls, func_local_idx)

            self._walk_for_calls(child, src, func_local_idx, raw_calls, depth + 1)

    def _handle_call(
        self,
        node: Node,
        src: bytes,
        raw_calls: list[tuple[int | None, str, int]],
        current_func_local_idx: int | None,
    ) -> None:
        """Extract function call."""
        func_node = node.child_by_field_name("function")
        if not func_node:
            return

        callee_name = ""
        if func_node.type == "identifier":
            callee_name = self._txt(func_node, src)
        elif func_node.type == "call_expression":
            actual_func = func_node.child_by_field_name("function")
            if actual_func and actual_func.type == "identifier":
                callee_name = self._txt(actual_func, src)
        elif func_node.type == "field_expression":
            access_node = func_node.child_by_field_name("field")
            if access_node:
                callee_name = self._txt(access_node, src)
        elif func_node.type == "pointer_expression":
            for child in func_node.children:
                if child.type == "identifier":
                    callee_name = self._txt(child, src)
                    break

        if callee_name:
            raw_calls.append(
                (
                    current_func_local_idx,
                    callee_name,
                    node.start_point[0] + 1,
                )
            )

    # ------------------------------------------------------------------
    # Import extraction (#include directives)
    # ------------------------------------------------------------------

    def _extract_include(self, node: Node, src: bytes, file_id: int) -> list[ImportEdge]:
        """Extract #include directive as ImportEdge."""
        edges: list[ImportEdge] = []

        path_node = self._get_child_by_type(node, "string_literal")
        if not path_node:
            path_node = self._get_child_by_type(node, "system_lib_string")
        if not path_node:
            return edges

        path = self._txt(path_node, src)
        path = path.strip('<>"')

        if path:
            edges.append(
                ImportEdge(
                    file_id=file_id,
                    imported_from=path,
                    imported_names=[],
                )
            )

        return edges

    def _handle_preproc_define(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
    ) -> None:
        """Extract #define preprocessor directives as CONSTANT symbols."""
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return

        name = self._txt(name_node, src)
        body = self._txt(node, src)

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.CONSTANT,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"#define {name}",
                body=body[:500],
                is_public=True,
            )
        )

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _extract_declarator_name(self, decl_node: Node, src: bytes) -> str | None:
        """Extract function/variable name from declarator node."""
        current = decl_node
        while current:
            if current.type == "identifier":
                return self._txt(current, src)
            elif current.type == "pointer_declarator":
                current = current.child_by_field_name("declarator")
            elif current.type == "function_declarator":
                current = current.child_by_field_name("declarator")
            elif current.type == "parenthesized_declarator":
                ident_child = self._get_child_by_type(current, "identifier")
                if ident_child:
                    return self._txt(ident_child, src)
                current = current.child_by_field_name("declarator")
            elif current.type == "array_declarator":
                current = current.child_by_field_name("declarator")
            else:
                ident_child = self._get_child_by_type(current, "identifier")
                if ident_child:
                    return self._txt(ident_child, src)
                break

        return None

    def _extract_function_signature(self, func_def_node: Node, src: bytes) -> str:
        """Extract function signature from function_definition node."""
        decl_node = func_def_node.child_by_field_name("declarator")
        if not decl_node:
            return "function"

        return_type = ""
        for child in func_def_node.children:
            if child.type in ("primitive_type", "type_identifier", "sized_type_specifier"):
                return_type = self._txt(child, src)
                break

        func_name = self._extract_declarator_name(decl_node, src)
        if not func_name:
            func_name = "?"

        params_node = None
        for child in decl_node.children:
            if child.type == "parameter_list":
                params_node = child
                break

        params = self._txt(params_node, src) if params_node else "()"
        return f"{return_type} {func_name}{params}".strip()

    def _extract_function_signature_from_declaration(self, decl_node: Node, src: bytes) -> str:
        """Extract function signature from declaration node (no body)."""
        declarator_node = decl_node.child_by_field_name("declarator")
        if not declarator_node:
            return "function"

        return_type = ""
        for child in decl_node.children:
            if child.type in ("primitive_type", "type_identifier", "sized_type_specifier"):
                return_type = self._txt(child, src)
                break

        func_name = self._extract_declarator_name(declarator_node, src)
        if not func_name:
            func_name = "?"

        params_node = None
        for child in declarator_node.children:
            if child.type == "parameter_list":
                params_node = child
                break

        params = self._txt(params_node, src) if params_node else "()"
        return f"{return_type} {func_name}{params}".strip()

    def _get_preceding_comment(self, node: Node, src: bytes) -> str | None:
        """Extract preceding comment above a node."""
        if node.start_point[0] == 0:
            return None

        start_line = node.start_point[0]
        src_lines = src.decode("utf-8", errors="replace").split("\n")

        for i in range(start_line - 1, max(0, start_line - 10), -1):
            line = src_lines[i].strip()
            if line.startswith("//"):
                return line[2:].strip()
            elif line.startswith("/*"):
                return line.replace("/*", "").replace("*/", "").strip()
            elif line and not line.startswith("*") and not line.endswith("*/"):
                break

        return None

    def _count_errors(self, node: Node) -> int:
        """Count ERROR nodes in tree (parse quality metric)."""
        count = 1 if node.type == "ERROR" else 0
        for child in node.children:
            count += self._count_errors(child)
        return count

    def _txt(self, node: Node, src: bytes) -> str:
        """Extract text from node."""
        return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _get_child_by_type(self, node: Node, type_name: str) -> Node | None:
        """Find first child node of given type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None
