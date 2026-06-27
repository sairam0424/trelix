"""Rust parser — direct AST traversal using Tree-sitter.

Extracts:
  - Structs (kind=CLASS) — struct Foo {}
  - Enums (kind=ENUM) — enum Foo {}
  - Traits (kind=INTERFACE) — trait Foo {}
  - Type aliases (kind=INTERFACE) — type Foo = Bar
  - Functions (kind=FUNCTION) — top-level fn declarations
  - Methods inside impl blocks (kind=METHOD, parent_id=struct local idx)
  - Trait method signatures (kind=METHOD) — fn declarations inside trait bodies
  - Struct fields (kind=VARIABLE) — pub fields inside named structs
  - Enum variants (kind=CONSTANT) — variants inside enum bodies
  - Constants and non-mutable statics (kind=CONSTANT) — const/static items
  - Module docstring — //! inner doc comments → MODULE symbol
  - Use declarations → ImportEdge
  - Call edges → CallEdge (function calls and method calls)

Parent linkage:
  impl blocks link methods to their struct by matching the impl type name
  to a previously seen struct/enum/trait. parent_id is set to the local
  index of that symbol. The Indexer remaps local indices to real DB ids.
"""

from __future__ import annotations

import re

from tree_sitter import Node

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge
from trelix.indexing.parser._grammar import load_language, make_parser
from trelix.indexing.parser.base import BaseParser, ParseResult

MAX_STRUCT_FIELDS = 30
MAX_ENUM_VARIANTS = 50


class RustParser(BaseParser):
    """Tree-sitter based Rust parser using direct AST traversal."""

    def __init__(self) -> None:
        self._ts_language = load_language("rust")
        self._parser = make_parser("rust")

    @property
    def language_name(self) -> str:
        return "rust"

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

        # Module docstring from //! inner doc comments
        module_sym = self._get_module_symbol(root, source_bytes, file_id)
        if module_sym:
            symbols.append(module_sym)

        # name → local_idx map for linking impl methods to their struct/enum/trait
        type_idx: dict[str, int] = {}

        self._walk_top_level(
            root, source_bytes, file_id, symbols, import_edges, type_edges, type_idx, raw_calls
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
        """Extract //! inner doc comments at the top of the file as a MODULE symbol."""
        inner_doc_lines: list[str] = []
        for child in root.children:
            if child.type == "line_comment":
                text = self._txt(child, src)
                if text.startswith("//!"):
                    inner_doc_lines.append(text)
                    continue
            elif child.type == "block_comment":
                text = self._txt(child, src)
                if text.startswith("/*!"):
                    inner_doc_lines.append(text)
                    continue
            # Stop at first non-comment non-shebang content
            if child.type not in ("shebang", "attribute_item", "inner_attribute_item"):
                break

        if not inner_doc_lines:
            return None

        docstring = self._clean_comment("\n".join(inner_doc_lines))
        if not docstring:
            return None

        # Try to find the crate name from mod_item or the file path (unavailable here)
        # Use "crate" as a fallback name
        return Symbol(
            file_id=file_id,
            name="crate",
            qualified_name="crate",
            kind=SymbolKind.MODULE,
            line_start=1,
            line_end=root.end_point[0] + 1,
            signature="crate",
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
        type_edges: list[TypeEdge],
        type_idx: dict[str, int],
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        for child in node.children:
            ntype = child.type
            if ntype == "use_declaration":
                import_edges.extend(self._extract_import(child, src, file_id))
            elif ntype == "extern_crate_declaration":
                edge = self._handle_extern_crate(child, src, file_id)
                if edge:
                    import_edges.append(edge)
            elif ntype == "function_item":
                self._handle_function(child, src, file_id, symbols, raw_calls)
            elif ntype == "struct_item":
                self._handle_struct(child, src, file_id, symbols, type_idx)
            elif ntype == "union_item":
                self._handle_union(child, src, file_id, symbols, type_idx)
            elif ntype == "enum_item":
                self._handle_enum(child, src, file_id, symbols, type_idx)
            elif ntype == "trait_item":
                self._handle_trait(child, src, file_id, symbols, type_idx, type_edges, raw_calls)
            elif ntype == "type_item":
                self._handle_type_alias(child, src, file_id, symbols)
            elif ntype == "impl_item":
                self._handle_impl(child, src, file_id, symbols, type_idx, type_edges, raw_calls)
            elif ntype == "const_item":
                self._handle_const_item(child, src, file_id, symbols, parent_local_idx=None)
            elif ntype == "static_item":
                self._handle_static_item(child, src, file_id, symbols)
            elif ntype == "macro_definition":
                self._handle_macro_def(child, src, file_id, symbols)
            elif ntype == "mod_item":
                # Recurse into inline modules
                body = self._get_child_by_type(child, "declaration_list")
                if body:
                    self._walk_top_level(
                        body, src, file_id, symbols, import_edges, type_edges, type_idx, raw_calls
                    )

    # ------------------------------------------------------------------
    # Struct
    # ------------------------------------------------------------------

    def _handle_struct(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        type_idx: dict[str, int],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = self._txt(name_node, src)

        tp_node = node.child_by_field_name("type_parameters")
        tp_str = self._txt(tp_node, src) if tp_node else ""

        attrs = self._get_rust_attributes(node, src)
        is_pub = (
            node.child_by_field_name("visibility") is not None
            or self._get_child_by_type(node, "visibility_modifier") is not None
        )

        local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.CLASS,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"struct {name}{tp_str}",
                body=self._txt(node, src),
                docstring=self._get_preceding_comment(node, src),
                decorators=attrs,
                is_public=is_pub,
            )
        )
        type_idx[name] = local_idx

        # Extract pub struct fields from named struct bodies
        body_node = self._get_child_by_type(node, "field_declaration_list")
        if body_node:
            self._extract_struct_fields(body_node, src, file_id, symbols, local_idx)

    def _extract_struct_fields(
        self,
        body: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        parent_local_idx: int,
    ) -> None:
        """Extract pub field_declaration children from a field_declaration_list."""
        field_count = 0
        for child in body.children:
            if child.type != "field_declaration" or field_count >= MAX_STRUCT_FIELDS:
                continue
            # Only extract pub fields (is_public check)
            is_pub = self._get_child_by_type(child, "visibility_modifier") is not None
            if not is_pub:
                continue
            name_node = child.child_by_field_name("name")
            type_node = child.child_by_field_name("type")
            if not name_node:
                continue
            name = self._txt(name_node, src)
            type_str = self._txt(type_node, src) if type_node else ""
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.VARIABLE,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    signature=f"{name}: {type_str}",
                    body=self._txt(child, src),
                    parent_id=parent_local_idx,
                    is_public=True,
                )
            )
            field_count += 1

    # ------------------------------------------------------------------
    # Union
    # ------------------------------------------------------------------

    def _handle_union(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        type_idx: dict[str, int],
    ) -> None:
        """Handle: union RawData { i: i32, f: f32 }"""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = self._txt(name_node, src)
        tp_node = node.child_by_field_name("type_parameters")
        tp_str = self._txt(tp_node, src) if tp_node else ""
        is_pub = self._get_child_by_type(node, "visibility_modifier") is not None
        local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.CLASS,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"union {name}{tp_str}",
                body=self._txt(node, src),
                docstring=self._get_preceding_comment(node, src),
                decorators=self._get_rust_attributes(node, src),
                is_public=is_pub,
            )
        )
        type_idx[name] = local_idx
        body_node = self._get_child_by_type(node, "field_declaration_list")
        if body_node:
            self._extract_struct_fields(body_node, src, file_id, symbols, local_idx)

    # ------------------------------------------------------------------
    # Macro definition
    # ------------------------------------------------------------------

    def _handle_macro_def(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
    ) -> None:
        """Handle: macro_rules! my_macro { ... }"""
        name_node = node.child_by_field_name("name") or self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)
        body = self._txt(node, src)
        if len(body) > 800:
            body = body[:800] + "..."
        is_pub = self._get_child_by_type(node, "visibility_modifier") is not None
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.FUNCTION,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"macro_rules! {name}",
                body=body,
                docstring=self._get_preceding_comment(node, src),
                decorators=self._get_rust_attributes(node, src),
                is_public=is_pub,
            )
        )

    # ------------------------------------------------------------------
    # Enum
    # ------------------------------------------------------------------

    def _handle_enum(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        type_idx: dict[str, int],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = self._txt(name_node, src)

        tp_node = node.child_by_field_name("type_parameters")
        tp_str = self._txt(tp_node, src) if tp_node else ""

        attrs = self._get_rust_attributes(node, src)
        is_pub = self._get_child_by_type(node, "visibility_modifier") is not None

        local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.ENUM,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"enum {name}{tp_str}",
                body=self._txt(node, src),
                docstring=self._get_preceding_comment(node, src),
                decorators=attrs,
                is_public=is_pub,
            )
        )
        type_idx[name] = local_idx

        # Extract enum variants
        body_node = self._get_child_by_type(node, "enum_variant_list")
        if body_node:
            self._extract_enum_variants(body_node, src, file_id, symbols, local_idx, name)

    def _extract_enum_variants(
        self,
        body: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        parent_local_idx: int,
        enum_name: str,
    ) -> None:
        """Extract enum_variant children as CONSTANT symbols."""
        variant_count = 0
        for child in body.children:
            if child.type != "enum_variant" or variant_count >= MAX_ENUM_VARIANTS:
                continue
            name_node = child.child_by_field_name("name")
            if not name_node:
                continue
            name = self._txt(name_node, src)
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=f"{enum_name}::{name}",
                    kind=SymbolKind.CONSTANT,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    signature=f"{enum_name}::{name}",
                    body=self._txt(child, src),
                    parent_id=parent_local_idx,
                    is_public=True,
                )
            )
            variant_count += 1

    # ------------------------------------------------------------------
    # Trait
    # ------------------------------------------------------------------

    def _handle_trait(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        type_idx: dict[str, int],
        type_edges: list[TypeEdge],
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = self._txt(name_node, src)

        tp_node = node.child_by_field_name("type_parameters")
        tp_str = self._txt(tp_node, src) if tp_node else ""

        attrs = self._get_rust_attributes(node, src)
        is_pub = self._get_child_by_type(node, "visibility_modifier") is not None

        local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.INTERFACE,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"trait {name}{tp_str}",
                body=self._txt(node, src),
                docstring=self._get_preceding_comment(node, src),
                decorators=attrs,
                is_public=is_pub,
            )
        )
        type_idx[name] = local_idx

        # Supertraits: trait Foo: Clone + Debug → TypeEdge(extends)
        bounds_node = node.child_by_field_name("bounds")
        if bounds_node:
            for tc in bounds_node.children:
                if tc.type == "type_identifier":
                    type_edges.append(
                        TypeEdge(
                            from_symbol_id=local_idx,
                            to_type_name=self._txt(tc, src),
                            edge_kind="extends",
                        )
                    )
                elif tc.type == "generic_type":
                    ti = self._get_child_by_type(tc, "type_identifier")
                    if ti:
                        type_edges.append(
                            TypeEdge(
                                from_symbol_id=local_idx,
                                to_type_name=self._txt(ti, src),
                                edge_kind="extends",
                            )
                        )

        body = self._get_child_by_type(node, "declaration_list")
        if body:
            for child in body.children:
                if child.type in ("function_item", "function_signature_item"):
                    self._handle_trait_fn(child, src, file_id, symbols, local_idx, name, raw_calls)
                elif child.type == "associated_type":
                    # type Output; inside trait body
                    aname_node = child.child_by_field_name("name") or self._get_child_by_type(
                        child, "type_identifier"
                    )
                    if aname_node:
                        aname = self._txt(aname_node, src)
                        symbols.append(
                            Symbol(
                                file_id=file_id,
                                name=aname,
                                qualified_name=f"{name}::{aname}",
                                kind=SymbolKind.INTERFACE,
                                line_start=child.start_point[0] + 1,
                                line_end=child.end_point[0] + 1,
                                signature=self._txt(child, src).rstrip(";"),
                                body=self._txt(child, src),
                                parent_id=local_idx,
                                is_public=True,
                            )
                        )

    def _handle_trait_fn(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        trait_local_idx: int,
        trait_name: str,
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        name_node = node.child_by_field_name("name") or self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        attrs = self._get_rust_attributes(node, src)
        is_pub = self._get_child_by_type(node, "visibility_modifier") is not None

        func_local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=f"{trait_name}::{name}",
                kind=SymbolKind.METHOD,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=self._fn_signature(node, src, trait_name),
                body=self._txt(node, src),
                parent_id=trait_local_idx,
                docstring=self._get_preceding_comment(node, src),
                decorators=attrs,
                is_public=is_pub,
            )
        )

        body_node = node.child_by_field_name("body")
        if body_node:
            self._walk_body(body_node, func_local_idx, raw_calls, src)

    # ------------------------------------------------------------------
    # Type alias
    # ------------------------------------------------------------------

    def _handle_type_alias(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
    ) -> None:
        """Handle: type Foo = Bar;  type MyVec<T> = Vec<T>;"""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = self._txt(name_node, src)

        tp_node = node.child_by_field_name("type_parameters")
        tp_str = self._txt(tp_node, src) if tp_node else ""
        type_node = node.child_by_field_name("type")
        type_str = self._txt(type_node, src) if type_node else ""

        is_pub = self._get_child_by_type(node, "visibility_modifier") is not None

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.INTERFACE,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"type {name}{tp_str} = {type_str}",
                body=self._txt(node, src),
                docstring=self._get_preceding_comment(node, src),
                is_public=is_pub,
            )
        )

    # ------------------------------------------------------------------
    # Impl block (methods)
    # ------------------------------------------------------------------

    def _handle_impl(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        type_idx: dict[str, int],
        type_edges: list[TypeEdge],
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        """Handle: impl Foo { fn method(...) {...} }  and  impl Trait for Foo { ... }"""
        # Grammar: impl_item has "type" field (the implementing type) and optional "trait" field
        type_node = node.child_by_field_name("type")
        trait_node = node.child_by_field_name("trait")

        if type_node:
            # type_node may be generic_type (e.g. Vec<T>) — extract base type_identifier
            type_name = self._extract_type_name(type_node, src)
        else:
            # Fallback: scan for type_identifier manually
            ti = self._get_child_by_type(node, "type_identifier")
            type_name = self._txt(ti, src) if ti else ""

        if not type_name:
            return

        trait_name = ""
        if trait_node:
            trait_name = self._extract_type_name(trait_node, src)

        parent_local_idx = type_idx.get(type_name)

        # Record trait_impl type edge
        if trait_name and parent_local_idx is not None:
            type_edges.append(
                TypeEdge(
                    from_symbol_id=parent_local_idx,
                    to_type_name=trait_name,
                    edge_kind="trait_impl",
                )
            )

        body = self._get_child_by_type(node, "declaration_list")
        if not body:
            return

        for child in body.children:
            if child.type == "function_item":
                self._handle_impl_fn(
                    child, src, file_id, symbols, parent_local_idx, type_name, raw_calls
                )
            elif child.type == "const_item":
                self._handle_const_item(
                    child, src, file_id, symbols, parent_local_idx=parent_local_idx
                )
            elif child.type == "type_item":
                # type Alias = ... inside impl block
                self._handle_type_alias(child, src, file_id, symbols)

    def _extract_type_name(self, node: Node, src: bytes) -> str:
        """Extract the base type identifier from a type node (handles generic_type)."""
        if node.type == "type_identifier":
            return self._txt(node, src)
        # generic_type: Foo<T> → get the type_identifier child
        ti = self._get_child_by_type(node, "type_identifier")
        if ti:
            return self._txt(ti, src)
        # scoped_type_identifier: crate::Foo → last segment
        text = self._txt(node, src)
        return text.split("::")[-1] if "::" in text else text

    def _handle_impl_fn(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        parent_local_idx: int | None,
        type_name: str,
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        name_node = node.child_by_field_name("name") or self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        is_method = self._has_self_param(node)
        kind = SymbolKind.METHOD if is_method else SymbolKind.FUNCTION

        attrs = self._get_rust_attributes(node, src)
        is_pub = self._get_child_by_type(node, "visibility_modifier") is not None

        func_local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=f"{type_name}::{name}",
                kind=kind,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=self._fn_signature(node, src, type_name),
                body=self._txt(node, src),
                parent_id=parent_local_idx,
                docstring=self._get_preceding_comment(node, src),
                decorators=attrs,
                is_public=is_pub,
            )
        )

        body_node = node.child_by_field_name("body")
        if body_node:
            self._walk_body(body_node, func_local_idx, raw_calls, src)

    def _has_self_param(self, fn_node: Node) -> bool:
        """Return True if the fn has a self/&self/&mut self parameter."""
        params = self._get_child_by_type(fn_node, "parameters")
        if not params:
            return False
        for child in params.children:
            if child.type == "self_parameter":
                return True
        return False

    # ------------------------------------------------------------------
    # Top-level functions
    # ------------------------------------------------------------------

    def _handle_function(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        name_node = node.child_by_field_name("name") or self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        attrs = self._get_rust_attributes(node, src)
        is_pub = self._get_child_by_type(node, "visibility_modifier") is not None

        func_local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.FUNCTION,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=self._fn_signature(node, src, None),
                body=self._txt(node, src),
                docstring=self._get_preceding_comment(node, src),
                decorators=attrs,
                is_public=is_pub,
            )
        )

        body_node = node.child_by_field_name("body")
        if body_node:
            self._walk_body(body_node, func_local_idx, raw_calls, src)

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
        """Collect call_expression and method_call_expression nodes in a function body.

        Skips closure_expression and nested function_item scopes.
        """
        for child in node.children:
            if child.type == "function_item":
                continue  # nested fn item — has its own symbol, skip

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
                    elif func_node.type == "scoped_identifier":
                        # std::mem::swap → last segment
                        name_node = func_node.child_by_field_name(
                            "name"
                        ) or self._get_child_by_type(func_node, "identifier")
                        if name_node:
                            raw_calls.append(
                                (
                                    caller_local_idx,
                                    self._txt(name_node, src),
                                    child.start_point[0] + 1,
                                )
                            )
                    elif func_node.type == "field_expression":
                        # items.iter() — method name is in "field" (not "name")
                        field_node = func_node.child_by_field_name("field")
                        if field_node:
                            raw_calls.append(
                                (
                                    caller_local_idx,
                                    self._txt(field_node, src),
                                    child.start_point[0] + 1,
                                )
                            )

            elif child.type == "method_call_expression":
                # foo.bar(args) — "method" field is the method name identifier
                method_node = child.child_by_field_name("method")
                if method_node:
                    raw_calls.append(
                        (
                            caller_local_idx,
                            self._txt(method_node, src),
                            child.start_point[0] + 1,
                        )
                    )

            self._walk_body(child, caller_local_idx, raw_calls, src)

    # ------------------------------------------------------------------
    # Constant / static extraction
    # ------------------------------------------------------------------

    def _handle_const_item(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        parent_local_idx: int | None = None,
    ) -> None:
        """Handle: const MAX_SIZE: usize = 100;  pub const VERSION: &str = "1.0";"""
        name_node = node.child_by_field_name("name") or self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)
        body = self._txt(node, src)
        if len(body) > 500:
            body = body[:500] + "..."
        is_pub = self._get_child_by_type(node, "visibility_modifier") is not None
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.CONSTANT,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=body.split("\n")[0][:200],
                body=body,
                is_public=is_pub,
                parent_id=parent_local_idx,
            )
        )

    def _handle_static_item(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
    ) -> None:
        """Handle: static LANGUAGES: &[&str] = &[...];  (skips mutable statics)"""
        if any(c.type == "mutable_specifier" for c in node.children):
            return  # static mut is unsafe — skip
        name_node = node.child_by_field_name("name") or self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)
        body = self._txt(node, src)
        if len(body) > 500:
            body = body[:500] + "..."
        is_pub = self._get_child_by_type(node, "visibility_modifier") is not None
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.CONSTANT,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=body.split("\n")[0][:200],
                body=body,
                is_public=is_pub,
            )
        )

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _handle_extern_crate(self, node: Node, src: bytes, file_id: int) -> ImportEdge | None:
        """Handle: extern crate serde;  extern crate serde as s;"""
        # Children: 'extern', 'crate', identifier, (optional 'as' + identifier), ';'
        for child in node.children:
            if child.type == "identifier":
                return ImportEdge(
                    file_id=file_id,
                    imported_from=self._txt(child, src),
                    imported_names=[],
                )
        return None

    def _extract_import(self, node: Node, src: bytes, file_id: int) -> list[ImportEdge]:
        """Handle: use std::collections::HashMap; use crate::foo::{A, B};"""
        edges: list[ImportEdge] = []
        for child in node.children:
            if child.type not in ("use", ";"):
                path_text = self._txt(child, src)
                self._flatten_use_tree(child, src, file_id, path_text, edges)
                break
        return edges

    def _flatten_use_tree(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        path_so_far: str,
        edges: list[ImportEdge],
    ) -> None:
        """Recursively expand use trees like foo::{A, B, C}."""
        if node.type == "use_tree_list":
            for child in node.children:
                if child.type == "use_tree":
                    self._flatten_use_tree(child, src, file_id, path_so_far, edges)
        elif node.type == "use_tree":
            full = self._txt(node, src)
            edges.append(
                ImportEdge(
                    file_id=file_id,
                    imported_from=full,
                    imported_names=[],
                )
            )
        else:
            full = self._txt(node, src)
            parts = full.replace("::", ".").split(".")
            name = parts[-1] if parts else full
            module = ".".join(parts[:-1]) if len(parts) > 1 else full
            edges.append(
                ImportEdge(
                    file_id=file_id,
                    imported_from=module,
                    imported_names=[name],
                )
            )

    # ------------------------------------------------------------------
    # Signature builders
    # ------------------------------------------------------------------

    def _fn_signature(self, node: Node, src: bytes, type_name: str | None) -> str:
        """Build: fn Type::name<T>(params) -> ReturnType"""
        name_node = node.child_by_field_name("name") or self._get_child_by_type(node, "identifier")
        params_node = node.child_by_field_name("parameters") or self._get_child_by_type(
            node, "parameters"
        )
        return_type_node = node.child_by_field_name("return_type")
        tp_node = node.child_by_field_name("type_parameters")

        name = self._txt(name_node, src) if name_node else "?"
        tp_str = self._txt(tp_node, src) if tp_node else ""
        params = self._txt(params_node, src) if params_node else "()"
        ret = (" -> " + self._txt(return_type_node, src)) if return_type_node else ""
        prefix = f"{type_name}::" if type_name else ""
        return f"fn {prefix}{name}{tp_str}{params}{ret}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_rust_attributes(self, node: Node, src: bytes) -> list[str]:
        """Collect #[attr] attribute_item siblings immediately before this node."""
        result: list[str] = []
        prev = node.prev_named_sibling
        while prev is not None and prev.type == "attribute_item":
            text = self._txt(prev, src).strip()
            result.insert(0, text[:200] if len(text) <= 200 else text[:200] + "...")
            prev = prev.prev_named_sibling
        return result

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

    def _get_preceding_comment(self, node: Node, src: bytes) -> str | None:
        """Collect consecutive Rust doc comment lines (///) immediately before this node."""
        lines: list[str] = []
        prev = node.prev_named_sibling
        next_start_line = node.start_point[0]
        while prev is not None and prev.type in ("line_comment", "block_comment"):
            if prev.end_point[0] + 1 < next_start_line:
                break
            lines.insert(0, self._txt(prev, src))
            next_start_line = prev.start_point[0]
            prev = prev.prev_named_sibling
        return self._clean_comment("\n".join(lines)) if lines else None

    @staticmethod
    def _clean_comment(raw: str) -> str:
        """Strip comment delimiters: /* */, //, ///, //! prefixes."""
        raw = re.sub(r"^/\*[*!]?\s*", "", raw.strip())
        raw = re.sub(r"\s*\*+/$", "", raw)
        raw = re.sub(r"^\s*\*\s?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"^//[/!]?\s?", "", raw, flags=re.MULTILINE)
        return raw.strip()
