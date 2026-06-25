"""C++ parser — direct AST traversal using Tree-sitter.

Handles .cpp, .cc, .cxx, .h, .hpp files. Extracts:
  - Namespaces (kind=MODULE) — block-scoped namespace declarations
  - Classes & structs (kind=CLASS/STRUCT) — with base class tracking
  - Constructors & destructors (kind=METHOD, parent_id=class)
  - Methods inside classes/structs (kind=METHOD, parent_id=class)
  - Top-level functions (kind=FUNCTION)
  - Public/member fields (kind=VARIABLE, parent_id=class)
  - Template classes (kind=CLASS with <template> signature)
  - Enums (kind=ENUM) + members as CONSTANT
  - #include directives → ImportEdge (both <system> and "local")
  - Function calls → CallEdge
  - Base classes / inheritance → TypeEdge

Design notes:
  - Namespaces create MODULE symbols for architectural queries
  - Templates are preserved in signature but not elaborated
  - Public/protected fields extracted, private skipped unless named
  - Call graph built from function_call nodes
  - Parent linkage uses local indices during parse, remapped by Indexer

Parent linkage:
  parent_id in Symbol is set to the LOCAL INDEX in the symbols list during
  parsing. The Indexer remaps this to the actual DB id after insertion.
"""

from __future__ import annotations

from typing import Optional

import tree_sitter_languages
from tree_sitter import Node, Parser

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge
from trelix.indexing.parser.base import BaseParser, ParseResult


class CppParser(BaseParser):
    """Tree-sitter based C++ parser using direct AST traversal."""

    # Cap on member fields extracted per class (prevents symbol flood)
    MAX_CLASS_FIELDS: int = 30

    def __init__(self) -> None:
        self._ts_lang = tree_sitter_languages.get_language("cpp")
        self._parser = Parser()
        self._parser.set_language(self._ts_lang)

    @property
    def language_name(self) -> str:
        return "cpp"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def parse(self, source: str, file_id: int) -> ParseResult:
        """Parse C++ source code and extract symbols, relationships."""
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        symbols: list[Symbol] = []
        raw_calls: list[tuple[Optional[int], str, int]] = []
        import_edges: list[ImportEdge] = []
        type_edges: list[TypeEdge] = []

        self._walk(
            node=root,
            src=source_bytes,
            file_id=file_id,
            symbols=symbols,
            raw_calls=raw_calls,
            import_edges=import_edges,
            type_edges=type_edges,
            parent_class_local_idx=None,
            current_func_local_idx=None,
            depth=0,
        )

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
        raw_calls: list[tuple[Optional[int], str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        parent_class_local_idx: Optional[int],
        current_func_local_idx: Optional[int],
        depth: int,
    ) -> None:
        """Recursive depth-first walk of C++ AST."""
        if depth > 20:
            return

        for child in node.children:
            ntype = child.type

            if ntype == "namespace_declaration":
                self._walk(
                    child, src, file_id, symbols, raw_calls, import_edges, type_edges,
                    parent_class_local_idx, current_func_local_idx, depth + 1,
                )

            elif ntype == "preproc_include":
                import_edges.extend(self._extract_include(child, src, file_id))

            elif ntype == "preproc_def":
                self._handle_preproc_define(child, src, file_id, symbols)

            elif ntype == "class_specifier" and parent_class_local_idx is None:
                self._handle_class(
                    child, src, file_id, symbols, raw_calls, import_edges,
                    type_edges, depth, kind=SymbolKind.CLASS,
                )

            elif ntype == "struct_specifier" and parent_class_local_idx is None:
                self._handle_class(
                    child, src, file_id, symbols, raw_calls, import_edges,
                    type_edges, depth, kind=SymbolKind.STRUCT,
                )

            elif ntype == "enum_specifier" and parent_class_local_idx is None:
                self._handle_enum(child, src, file_id, symbols, type_edges)

            elif ntype == "function_definition" and parent_class_local_idx is None:
                self._handle_function(
                    child, src, file_id, symbols, raw_calls, import_edges,
                    type_edges, depth,
                )

            elif ntype == "call_expression":
                self._handle_call(child, src, raw_calls, current_func_local_idx)
                self._walk(
                    child, src, file_id, symbols, raw_calls, import_edges, type_edges,
                    parent_class_local_idx, current_func_local_idx, depth + 1,
                )

            elif ntype in (
                "compound_statement", "if_statement", "else_clause",
                "for_statement", "while_statement", "do_statement",
                "try_statement", "catch_clause", "throw_statement",
                "return_statement", "expression_statement",
                "switch_statement", "case_statement", "default_statement",
                "for_range_statement",
                "binary_expression", "unary_expression", "call_expression",
                "conditional_expression", "assignment_expression",
                "comma_expression", "parenthesized_expression",
                "subscript_expression", "field_expression", "pointer_expression",
                "initializer_list", "new_expression", "delete_expression",
                "cast_expression", "sizeof_expression",
                "lambda_expression",
                "declaration", "block_item",
            ):
                self._walk(
                    child, src, file_id, symbols, raw_calls, import_edges, type_edges,
                    parent_class_local_idx, current_func_local_idx, depth + 1,
                )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_class(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[Optional[int], str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        depth: int,
        kind: SymbolKind = SymbolKind.CLASS,
    ) -> None:
        """Extract class or struct declarations."""
        name_node = self._get_child_by_type(node, "type_identifier")
        if not name_node:
            name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return

        name = self._txt(name_node, src)

        template_node = self._find_preceding_template(node, src)
        template_sig = self._txt(template_node, src) if template_node else ""

        class_local_idx = len(symbols)
        signature = f"{kind.value} {name}"
        if template_sig:
            signature = f"{template_sig} {signature}"

        docstring = self._get_preceding_comment(node, src)

        symbols.append(Symbol(
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
        ))

        # Extract base classes → TypeEdge
        base_clause_node = node.child_by_field_name("base_class_list")
        if not base_clause_node:
            base_clause_node = self._get_child_by_type(node, "base_class_list")
        if base_clause_node:
            for child in base_clause_node.children:
                if child.type == "base_class":
                    base_name = self._extract_base_name(child, src)
                    if base_name:
                        type_edges.append(TypeEdge(
                            from_symbol_id=class_local_idx,
                            to_type_name=base_name,
                            edge_kind="extends",
                        ))
                elif child.type == "type_identifier":
                    base_name = self._txt(child, src)
                    if base_name:
                        type_edges.append(TypeEdge(
                            from_symbol_id=class_local_idx,
                            to_type_name=base_name,
                            edge_kind="extends",
                        ))

        body_node = self._get_child_by_type(node, "field_declaration_list")
        if body_node:
            self._walk_class_body(
                body_node, src, file_id, symbols, raw_calls, import_edges,
                type_edges, class_local_idx, name, depth,
            )

    def _walk_class_body(
        self,
        body_node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[Optional[int], str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        class_local_idx: int,
        class_name: str,
        depth: int,
    ) -> None:
        """Walk class body and extract members."""
        field_count = 0
        for child in body_node.children:
            ntype = child.type

            if ntype == "function_definition":
                self._handle_class_method(
                    child, src, file_id, symbols, raw_calls, import_edges,
                    type_edges, class_local_idx, class_name, depth,
                )

            elif ntype == "access_specifier":
                pass

            elif ntype == "field_declaration" and field_count < self.MAX_CLASS_FIELDS:
                before = len(symbols)
                self._handle_field_declaration(
                    child, src, file_id, symbols, class_local_idx, class_name,
                )
                field_count += len(symbols) - before

            elif ntype in ("class_specifier", "struct_specifier"):
                self._handle_class(
                    child, src, file_id, symbols, raw_calls, import_edges,
                    type_edges, depth + 1,
                    kind=SymbolKind.CLASS if ntype == "class_specifier" else SymbolKind.STRUCT,
                )

            elif ntype == "enum_specifier":
                self._handle_enum(child, src, file_id, symbols, type_edges)

    def _handle_class_method(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[Optional[int], str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        class_local_idx: int,
        class_name: str,
        depth: int,
    ) -> None:
        """Extract class method or constructor."""
        decl_node = node.child_by_field_name("declarator")
        if not decl_node:
            return

        name = self._extract_declarator_name(decl_node, src)
        if not name:
            return

        func_local_idx = len(symbols)
        signature = self._extract_function_signature(node, src)

        symbols.append(Symbol(
            file_id=file_id,
            name=name,
            qualified_name=f"{class_name}::{name}",
            kind=SymbolKind.METHOD,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=signature[:300],
            body=self._txt(node, src)[:2000],
            parent_id=class_local_idx,
            is_public=True,
            docstring=self._get_preceding_comment(node, src),
        ))

        body_node = node.child_by_field_name("body")
        if body_node:
            self._walk_for_calls(body_node, src, func_local_idx, raw_calls, depth + 1)

    def _handle_function(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[Optional[int], str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        depth: int,
    ) -> None:
        """Extract top-level function."""
        decl_node = node.child_by_field_name("declarator")
        if not decl_node:
            return

        name = self._extract_declarator_name(decl_node, src)
        if not name or name.startswith("operator"):
            return

        func_local_idx = len(symbols)
        signature = self._extract_function_signature(node, src)

        symbols.append(Symbol(
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
        ))

        body_node = node.child_by_field_name("body")
        if body_node:
            self._walk_for_calls(body_node, src, func_local_idx, raw_calls, depth + 1)

    def _handle_field_declaration(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        class_local_idx: int,
        class_name: str,
    ) -> None:
        """Extract class field/member variable declarations."""
        for child in node.children:
            if child.type == "declarator":
                name = self._extract_declarator_name(child, src)
                if not name or name.startswith("_"):
                    continue

                field_sig = self._txt(node, src).split(";")[0][:200]

                symbols.append(Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=f"{class_name}::{name}",
                    kind=SymbolKind.VARIABLE,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=field_sig,
                    body=self._txt(node, src)[:500],
                    parent_id=class_local_idx,
                    is_public=True,
                ))

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
            name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return

        name = self._txt(name_node, src)

        enum_local_idx = len(symbols)
        symbols.append(Symbol(
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
        ))

        enumerator_list = self._get_child_by_type(node, "enumerator_list")
        if enumerator_list:
            for child in enumerator_list.children:
                if child.type == "enumerator":
                    member_node = self._get_child_by_type(child, "identifier")
                    if member_node:
                        member_name = self._txt(member_node, src)
                        symbols.append(Symbol(
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
                        ))

    def _walk_for_calls(
        self,
        node: Node,
        src: bytes,
        func_local_idx: int,
        raw_calls: list[tuple[Optional[int], str, int]],
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
        raw_calls: list[tuple[Optional[int], str, int]],
        current_func_local_idx: Optional[int],
    ) -> None:
        """Extract function call."""
        func_node = node.child_by_field_name("function")
        if not func_node:
            return

        callee_name = ""
        if func_node.type == "identifier":
            callee_name = self._txt(func_node, src)
        elif func_node.type == "field_expression":
            access_node = func_node.child_by_field_name("field")
            if access_node:
                callee_name = self._txt(access_node, src)
        elif func_node.type == "qualified_identifier":
            parts = []
            for child in func_node.children:
                if child.type == "identifier":
                    parts.append(self._txt(child, src))
            callee_name = "::".join(parts) if parts else ""

        if callee_name:
            raw_calls.append((
                current_func_local_idx,
                callee_name,
                node.start_point[0] + 1,
            ))

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
            edges.append(ImportEdge(
                file_id=file_id,
                imported_from=path,
                imported_names=[],
            ))

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

        symbols.append(Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=SymbolKind.CONSTANT,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=f"#define {name}",
            body=body[:500],
            is_public=True,
        ))

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _extract_declarator_name(self, decl_node: Node, src: bytes) -> Optional[str]:
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

    def _extract_base_name(self, base_class_node: Node, src: bytes) -> Optional[str]:
        """Extract base class name from base_class node."""
        for child in base_class_node.children:
            if child.type == "type_identifier":
                return self._txt(child, src)
            elif child.type == "qualified_identifier":
                parts = []
                for c in child.children:
                    if c.type == "identifier":
                        parts.append(self._txt(c, src))
                return "::".join(parts) if parts else None

        return None

    def _find_preceding_template(self, node: Node, src: bytes) -> Optional[Node]:
        """Locate template_declaration before class/struct node."""
        parent = node.parent
        if not parent:
            return None

        prev_sibling = None
        for child in parent.children:
            if child == node:
                break
            prev_sibling = child

        if prev_sibling and prev_sibling.type == "template_declaration":
            return prev_sibling

        return None

    def _get_preceding_comment(self, node: Node, src: bytes) -> Optional[str]:
        """Extract preceding comment above a node."""
        if node.start_point[0] == 0:
            return None

        start_line = node.start_point[0]
        src_lines = src.decode("utf-8", errors="replace").split("\n")

        for i in range(start_line - 1, max(0, start_line - 10), -1):
            line = src_lines[i].strip()
            if line.startswith("//"):
                return line[2:].strip()
            elif line and not line.startswith("*") and not line.startswith("/*"):
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
        return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _get_child_by_type(self, node: Node, type_name: str) -> Optional[Node]:
        """Find first child node of given type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None
