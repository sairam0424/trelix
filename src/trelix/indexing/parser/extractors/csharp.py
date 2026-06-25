"""C# parser — direct AST traversal using Tree-sitter.

Handles .cs files. Extracts:
  - Namespaces (kind=MODULE) — block-scoped and file-scoped (C# 10+)
  - Classes, structs, records (kind=CLASS)
  - Interfaces (kind=INTERFACE)
  - Enums (kind=ENUM) + members as CONSTANT
  - Methods inside classes (kind=METHOD, parent_id=class)
  - Constructors (kind=METHOD, parent_id=class)
  - Top-level / static methods (kind=FUNCTION)
  - Properties (kind=VARIABLE, parent_id=class) — public only
  - Fields: const/static-readonly (kind=CONSTANT), public instance (kind=VARIABLE)
  - XML doc comments (/// <summary>...) extracted as docstrings
  - Attributes ([HttpPost], [ApiController]) extracted as decorators
  - using directives → ImportEdge
  - Call sites (invocation_expression) → CallEdge (caller_id = local idx)
  - Base classes / interfaces in base_list → TypeEdge

Parent linkage:
  parent_id in Symbol is set to the LOCAL INDEX in the symbols list during
  parsing. The Indexer remaps this to the actual DB id after insertion.
"""

from __future__ import annotations

import re

import tree_sitter_languages
from tree_sitter import Node, Parser

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge
from trelix.indexing.parser.base import BaseParser, ParseResult


class CSharpParser(BaseParser):
    """Tree-sitter based C# parser using direct AST traversal."""

    # Cap on class field/property symbols extracted per class.
    MAX_CLASS_FIELDS: int = 30
    # Cap on interface member symbols extracted per interface.
    MAX_INTERFACE_MEMBERS: int = 20

    def __init__(self) -> None:
        self._ts_lang = tree_sitter_languages.get_language("c_sharp")
        self._parser = Parser()
        self._parser.set_language(self._ts_lang)

    @property
    def language_name(self) -> str:
        return "c_sharp"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def parse(self, source: str, file_id: int) -> ParseResult:
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        symbols: list[Symbol] = []
        raw_calls: list[tuple[int | None, str, int]] = []
        import_edges: list[ImportEdge] = []
        type_edges: list[TypeEdge] = []

        # File-level XML doc comment (/// before any declaration)
        module_doc = self._get_file_doc(root, source_bytes)
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

        # Append top-level signatures to module symbol body if present
        if module_doc and symbols and symbols[0].kind == SymbolKind.MODULE:
            top_sigs = [
                s.signature
                for s in symbols[1:]
                if s.parent_id is None and s.kind not in (SymbolKind.CONSTANT,)
            ][:20]
            if top_sigs:
                symbols[0].body = module_doc + "\n\n# Symbols:\n" + "\n".join(top_sigs)

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
        parent_class_local_idx: int | None,
        current_func_local_idx: int | None,
        depth: int,
    ) -> None:
        """Recursive depth-first walk."""
        if depth > 20:
            return

        children = list(node.children)
        for i, child in enumerate(children):
            ntype = child.type

            # ---- Namespace containers (block-scoped and file-scoped) ----
            if ntype in ("namespace_declaration", "file_scoped_namespace_declaration"):
                self._walk(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_class_local_idx,
                    current_func_local_idx,
                    depth + 1,
                )

            # ---- Declaration list (class body / namespace body) ----------
            elif ntype == "declaration_list":
                self._walk(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_class_local_idx,
                    current_func_local_idx,
                    depth + 1,
                )

            # ---- Using directives → ImportEdge ---------------------------
            elif ntype == "using_directive":
                edge = self._extract_using(child, src, file_id)
                if edge:
                    import_edges.append(edge)

            # ---- Class / struct / record ---------------------------------
            elif ntype in ("class_declaration", "struct_declaration", "record_declaration"):
                doc = self._get_preceding_xml_doc(children, i, src)
                self._handle_class(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    depth,
                    docstring=doc,
                )

            # ---- Delegate declaration → INTERFACE symbol ----------------
            elif ntype == "delegate_declaration":
                doc = self._get_preceding_xml_doc(children, i, src)
                self._handle_delegate(
                    child, src, file_id, symbols, parent_class_local_idx, docstring=doc
                )

            # ---- Event field → VARIABLE symbol --------------------------
            elif ntype == "event_field_declaration" and parent_class_local_idx is not None:
                self._handle_event_field(child, src, file_id, symbols, parent_class_local_idx)

            # ---- Interface ----------------------------------------------
            elif ntype == "interface_declaration":
                doc = self._get_preceding_xml_doc(children, i, src)
                self._handle_interface(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    depth,
                    docstring=doc,
                )

            # ---- Enum ---------------------------------------------------
            elif ntype == "enum_declaration":
                doc = self._get_preceding_xml_doc(children, i, src)
                self._handle_enum(
                    child,
                    src,
                    file_id,
                    symbols,
                    type_edges,
                    docstring=doc,
                )

            # ---- Method -------------------------------------------------
            elif ntype == "method_declaration":
                doc = self._get_preceding_xml_doc(children, i, src)
                self._handle_method(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_class_local_idx,
                    current_func_local_idx,
                    depth,
                    docstring=doc,
                )

            # ---- Constructor --------------------------------------------
            elif ntype == "constructor_declaration":
                doc = self._get_preceding_xml_doc(children, i, src)
                self._handle_constructor(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_class_local_idx,
                    current_func_local_idx,
                    depth,
                    docstring=doc,
                )

            # ---- Property (class member only) ---------------------------
            elif ntype == "property_declaration" and parent_class_local_idx is not None:
                field_count = sum(
                    1
                    for s in symbols
                    if s.parent_id == parent_class_local_idx
                    and s.kind in (SymbolKind.VARIABLE, SymbolKind.CONSTANT)
                )
                if field_count < self.MAX_CLASS_FIELDS:
                    doc = self._get_preceding_xml_doc(children, i, src)
                    self._handle_property(
                        child,
                        src,
                        file_id,
                        symbols,
                        parent_class_local_idx,
                        docstring=doc,
                    )

            # ---- Field (class member only) ------------------------------
            elif ntype == "field_declaration" and parent_class_local_idx is not None:
                field_count = sum(
                    1
                    for s in symbols
                    if s.parent_id == parent_class_local_idx
                    and s.kind in (SymbolKind.VARIABLE, SymbolKind.CONSTANT)
                )
                if field_count < self.MAX_CLASS_FIELDS:
                    self._handle_field(
                        child,
                        src,
                        file_id,
                        symbols,
                        parent_class_local_idx,
                    )

            # ---- Call sites (track for call graph) ----------------------
            elif ntype == "invocation_expression":
                self._handle_call(child, src, raw_calls, current_func_local_idx)
                self._walk(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_class_local_idx,
                    current_func_local_idx,
                    depth + 1,
                )

            # ---- Recurse into statement / expression containers ---------
            elif ntype in (
                "block",
                "if_statement",
                "else_clause",
                "for_statement",
                "foreach_statement",
                "while_statement",
                "do_statement",
                "try_statement",
                "catch_clause",
                "finally_clause",
                "switch_statement",
                "switch_section",
                "using_statement",
                "lock_statement",
                "expression_statement",
                "return_statement",
                "local_declaration_statement",
                "local_function_statement",
                "checked_statement",
                "unchecked_statement",
                "fixed_statement",
                "unsafe_statement",
                "yield_statement",
                "throw_statement",
                "assignment_expression",
                "conditional_expression",
                "await_expression",
                "object_creation_expression",
                "equals_value_clause",
                "argument",
                "argument_list",
                "variable_declaration",
                "variable_declarator",
                "lambda_expression",
                "anonymous_method_expression",
                "element_access_expression",
                "element_binding_expression",
                "postfix_unary_expression",
                "prefix_unary_expression",
                "binary_expression",
                "parenthesized_expression",
                "initializer_expression",
            ):
                self._walk(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_class_local_idx,
                    current_func_local_idx,
                    depth + 1,
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
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        depth: int,
        docstring: str | None = None,
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        modifiers = self._get_modifiers(node, src)
        is_public = "public" in modifiers or "internal" in modifiers
        decorators = self._get_attribute_texts(node, src)
        bases = self._get_base_list(node, src)

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=SymbolKind.CLASS,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=self._class_signature(node, src),
            body=self._txt(node, src),
            docstring=docstring,
            decorators=decorators,
            is_public=is_public,
        )
        class_local_idx = len(symbols)
        symbols.append(sym)

        for base in bases:
            type_edges.append(
                TypeEdge(
                    from_symbol_id=class_local_idx,
                    to_type_name=base,
                    edge_kind="extends",
                )
            )

        body_node = self._get_child_by_type(node, "declaration_list")
        if body_node:
            self._walk(
                body_node,
                src,
                file_id,
                symbols,
                raw_calls,
                import_edges,
                type_edges,
                parent_class_local_idx=class_local_idx,
                current_func_local_idx=None,
                depth=depth + 1,
            )

    def _handle_interface(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        depth: int,
        docstring: str | None = None,
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        modifiers = self._get_modifiers(node, src)
        is_public = "public" in modifiers or "internal" in modifiers
        decorators = self._get_attribute_texts(node, src)
        bases = self._get_base_list(node, src)

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=SymbolKind.INTERFACE,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=self._interface_signature(node, src),
            body=self._txt(node, src),
            docstring=docstring,
            decorators=decorators,
            is_public=is_public,
        )
        iface_local_idx = len(symbols)
        symbols.append(sym)

        for base in bases:
            type_edges.append(
                TypeEdge(
                    from_symbol_id=iface_local_idx,
                    to_type_name=base,
                    edge_kind="extends",
                )
            )

        body_node = self._get_child_by_type(node, "declaration_list")
        if body_node:
            children = list(body_node.children)
            member_count = 0
            for j, member in enumerate(children):
                if member_count >= self.MAX_INTERFACE_MEMBERS:
                    break
                if member.type in ("method_declaration", "property_declaration"):
                    mdoc = self._get_preceding_xml_doc(children, j, src)
                    member_name_node = (
                        self._get_method_name_node(member)
                        if member.type == "method_declaration"
                        else self._get_child_by_type(member, "identifier")
                    )
                    if not member_name_node:
                        continue
                    member_name = self._txt(member_name_node, src)
                    sig = self._txt(member, src).split("{")[0].split(";")[0].strip()
                    if len(sig) > 200:
                        sig = sig[:200] + "..."
                    symbols.append(
                        Symbol(
                            file_id=file_id,
                            name=member_name,
                            qualified_name=f"{name}.{member_name}",
                            kind=(
                                SymbolKind.METHOD
                                if member.type == "method_declaration"
                                else SymbolKind.VARIABLE
                            ),
                            line_start=member.start_point[0] + 1,
                            line_end=member.end_point[0] + 1,
                            signature=sig,
                            body=self._txt(member, src),
                            docstring=mdoc,
                            parent_id=iface_local_idx,
                            is_public=True,
                        )
                    )
                    member_count += 1

    def _handle_enum(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        type_edges: list[TypeEdge],
        docstring: str | None = None,
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        modifiers = self._get_modifiers(node, src)
        is_public = "public" in modifiers or "internal" in modifiers
        decorators = self._get_attribute_texts(node, src)

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=SymbolKind.ENUM,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=self._enum_signature(node, src),
            body=self._txt(node, src),
            docstring=docstring,
            decorators=decorators,
            is_public=is_public,
        )
        enum_local_idx = len(symbols)
        symbols.append(sym)

        member_list = self._get_child_by_type(node, "enum_member_declaration_list")
        if member_list:
            for member in member_list.children:
                if member.type != "enum_member_declaration":
                    continue
                member_name_node = self._get_child_by_type(member, "identifier")
                if not member_name_node:
                    continue
                member_name = self._txt(member_name_node, src)
                body = self._txt(member, src)
                symbols.append(
                    Symbol(
                        file_id=file_id,
                        name=member_name,
                        qualified_name=f"{name}.{member_name}",
                        kind=SymbolKind.CONSTANT,
                        line_start=member.start_point[0] + 1,
                        line_end=member.end_point[0] + 1,
                        signature=body[:200],
                        body=body,
                        parent_id=enum_local_idx,
                        is_public=is_public,
                    )
                )

    def _handle_method(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        parent_class_local_idx: int | None,
        current_func_local_idx: int | None,
        depth: int,
        docstring: str | None = None,
    ) -> None:
        name_node = self._get_method_name_node(node)
        if not name_node:
            return
        name = self._txt(name_node, src)

        modifiers = self._get_modifiers(node, src)
        is_public = "public" in modifiers or "internal" in modifiers or "protected" in modifiers
        decorators = self._get_attribute_texts(node, src)
        is_method = parent_class_local_idx is not None
        kind = SymbolKind.METHOD if is_method else SymbolKind.FUNCTION

        if is_method:
            class_name = symbols[parent_class_local_idx].name  # type: ignore[index]
            qualified_name = f"{class_name}.{name}"
        else:
            qualified_name = name

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=qualified_name,
            kind=kind,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=self._method_signature(node, src),
            body=self._txt(node, src),
            docstring=docstring,
            decorators=decorators,
            is_public=is_public,
            parent_id=parent_class_local_idx,
        )
        method_local_idx = len(symbols)
        symbols.append(sym)

        body_node = self._get_child_by_type(node, "block")
        if body_node:
            self._walk(
                body_node,
                src,
                file_id,
                symbols,
                raw_calls,
                import_edges,
                type_edges,
                parent_class_local_idx=parent_class_local_idx,
                current_func_local_idx=method_local_idx,
                depth=depth + 1,
            )

    def _handle_constructor(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        parent_class_local_idx: int | None,
        current_func_local_idx: int | None,
        depth: int,
        docstring: str | None = None,
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        modifiers = self._get_modifiers(node, src)
        is_public = "public" in modifiers
        decorators = self._get_attribute_texts(node, src)

        class_name = (
            symbols[parent_class_local_idx].name  # type: ignore[index]
            if parent_class_local_idx is not None
            else name
        )
        qualified_name = f"{class_name}.{name}"

        params_node = node.child_by_field_name("parameters") or self._get_child_by_type(
            node, "parameter_list"
        )
        params_text = self._txt(params_node, src) if params_node else "()"
        sig = f"{name}{params_text}"

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=qualified_name,
            kind=SymbolKind.METHOD,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=sig,
            body=self._txt(node, src),
            docstring=docstring,
            decorators=decorators,
            is_public=is_public,
            parent_id=parent_class_local_idx,
        )
        ctor_local_idx = len(symbols)
        symbols.append(sym)

        body_node = self._get_child_by_type(node, "block")
        if body_node:
            self._walk(
                body_node,
                src,
                file_id,
                symbols,
                raw_calls,
                import_edges,
                type_edges,
                parent_class_local_idx=parent_class_local_idx,
                current_func_local_idx=ctor_local_idx,
                depth=depth + 1,
            )

    def _handle_property(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        parent_class_local_idx: int,
        docstring: str | None = None,
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        modifiers = self._get_modifiers(node, src)
        is_public = "public" in modifiers or "internal" in modifiers or "protected" in modifiers
        if not is_public:
            return

        class_name = symbols[parent_class_local_idx].name  # type: ignore[index]
        body = self._txt(node, src)
        if len(body) > 500:
            body = body[:500] + "..."
        sig = self._property_signature(node, src)

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=f"{class_name}.{name}",
                kind=SymbolKind.VARIABLE,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=sig,
                body=body,
                docstring=docstring,
                parent_id=parent_class_local_idx,
                is_public=is_public,
            )
        )

    def _handle_field(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        parent_class_local_idx: int,
    ) -> None:
        """Extract field_declaration members."""
        modifiers = self._get_modifiers(node, src)
        is_const = "const" in modifiers
        is_static_readonly = "static" in modifiers and "readonly" in modifiers
        is_public = "public" in modifiers or "internal" in modifiers
        is_protected = "protected" in modifiers

        if not (is_const or is_static_readonly or is_public or is_protected):
            return

        var_decl = self._get_child_by_type(node, "variable_declaration")
        if not var_decl:
            return

        class_name = symbols[parent_class_local_idx].name  # type: ignore[index]
        body = self._txt(node, src).rstrip(";").strip()
        if len(body) > 500:
            body = body[:500] + "..."

        for child in var_decl.children:
            if child.type != "variable_declarator":
                continue
            name_node = self._get_child_by_type(child, "identifier")
            if not name_node:
                continue
            name = self._txt(name_node, src)

            kind = SymbolKind.CONSTANT if (is_const or is_static_readonly) else SymbolKind.VARIABLE
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=f"{class_name}.{name}",
                    kind=kind,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=body.split("\n")[0][:200],
                    body=body,
                    parent_id=parent_class_local_idx,
                    is_public=is_public,
                )
            )

    def _get_method_name_node(self, node: Node) -> Node | None:
        """Return the identifier node that is the method name."""
        children = list(node.children)
        for i, child in enumerate(children):
            if child.type == "parameter_list":
                j = i - 1
                while j >= 0:
                    if children[j].type == "type_parameter_list":
                        j -= 1
                        continue
                    if children[j].type == "identifier":
                        return children[j]
                    break
                break
        return self._get_child_by_type(node, "identifier")

    def _handle_delegate(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        parent_class_local_idx: int | None,
        docstring: str | None = None,
    ) -> None:
        """Extract delegate_declaration as SymbolKind.INTERFACE."""
        name_node = self._get_method_name_node(node) or self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)
        modifiers = self._get_modifiers(node, src)
        is_public = "public" in modifiers or "internal" in modifiers
        body = self._txt(node, src).rstrip(";").strip()
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.INTERFACE,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=body[:200],
                body=body,
                docstring=docstring,
                parent_id=parent_class_local_idx,
                is_public=is_public,
            )
        )

    def _handle_event_field(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        parent_class_local_idx: int,
    ) -> None:
        """Extract event_field_declaration as SymbolKind.VARIABLE."""
        modifiers = self._get_modifiers(node, src)
        is_public = "public" in modifiers or "internal" in modifiers or "protected" in modifiers
        if not is_public:
            return

        class_name = symbols[parent_class_local_idx].name  # type: ignore[index]
        body = self._txt(node, src).rstrip(";").strip()

        var_decl = self._get_child_by_type(node, "variable_declaration")
        if not var_decl:
            return
        for child in var_decl.children:
            if child.type != "variable_declarator":
                continue
            name_node = self._get_child_by_type(child, "identifier")
            if not name_node:
                continue
            name = self._txt(name_node, src)
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=f"{class_name}.{name}",
                    kind=SymbolKind.VARIABLE,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=body[:200],
                    body=body,
                    parent_id=parent_class_local_idx,
                    is_public=is_public,
                )
            )

    def _handle_call(
        self,
        node: Node,
        src: bytes,
        raw_calls: list[tuple[int | None, str, int]],
        current_func_local_idx: int | None,
    ) -> None:
        """Extract callee name from an invocation_expression node."""
        if not node.children:
            return
        func_node = node.children[0]
        name = ""
        if func_node.type == "identifier":
            name = self._txt(func_node, src)
        elif func_node.type == "member_access_expression":
            member = self._get_last_identifier(func_node, src)
            name = member
        elif func_node.type == "generic_name":
            ident = self._get_child_by_type(func_node, "identifier")
            name = self._txt(ident, src) if ident else ""

        if name:
            raw_calls.append((current_func_local_idx, name, node.start_point[0] + 1))

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_using(self, node: Node, src: bytes, file_id: int) -> ImportEdge | None:
        """Extract a using directive."""
        has_semi = any(c.type == ";" for c in node.children)
        if not has_semi:
            return None

        # Three forms:
        #  using System;                      → qualified_name / identifier
        #  using static System.Math;          → qualified_name / identifier
        #  using MyAlias = System.Text;       → name_equals child + qualified_name sibling
        has_alias = any(c.type == "name_equals" for c in node.children)
        if has_alias:
            # Skip the name_equals node; the RHS qualified_name is a sibling
            for child in node.children:
                if child.type in ("qualified_name", "identifier") and child.is_named:
                    return ImportEdge(
                        file_id=file_id,
                        imported_from=self._txt(child, src),
                        imported_names=[],
                    )
            return None

        for child in node.children:
            if child.type in ("qualified_name", "identifier") and child.is_named:
                return ImportEdge(
                    file_id=file_id,
                    imported_from=self._txt(child, src),
                    imported_names=[],
                )
        return None

    # ------------------------------------------------------------------
    # Signature builders
    # ------------------------------------------------------------------

    def _class_signature(self, node: Node, src: bytes) -> str:
        """Build 'public class ClassName : Base1, IFoo' signature."""
        modifiers = " ".join(self._get_modifiers(node, src))
        name_node = self._get_child_by_type(node, "identifier")
        name = self._txt(name_node, src) if name_node else "?"
        keyword = "class"
        for c in node.children:
            if c.type in ("class", "struct", "record"):
                keyword = self._txt(c, src)
                break

        type_params = node.child_by_field_name("type_parameters") or self._get_child_by_type(
            node, "type_parameter_list"
        )
        type_params_text = self._txt(type_params, src) if type_params else ""

        base_list = self._get_child_by_type(node, "base_list")
        bases_text = f" {self._txt(base_list, src)}" if base_list else ""

        prefix = f"{modifiers} " if modifiers else ""
        return f"{prefix}{keyword} {name}{type_params_text}{bases_text}"

    def _interface_signature(self, node: Node, src: bytes) -> str:
        """Build 'public interface IFoo : IBar' signature."""
        modifiers = " ".join(self._get_modifiers(node, src))
        name_node = self._get_child_by_type(node, "identifier")
        name = self._txt(name_node, src) if name_node else "?"
        type_params = self._get_child_by_type(node, "type_parameter_list")
        type_params_text = self._txt(type_params, src) if type_params else ""
        base_list = self._get_child_by_type(node, "base_list")
        bases_text = f" {self._txt(base_list, src)}" if base_list else ""
        prefix = f"{modifiers} " if modifiers else ""
        return f"{prefix}interface {name}{type_params_text}{bases_text}"

    def _enum_signature(self, node: Node, src: bytes) -> str:
        """Build 'public enum Status' signature."""
        modifiers = " ".join(self._get_modifiers(node, src))
        name_node = self._get_child_by_type(node, "identifier")
        name = self._txt(name_node, src) if name_node else "?"
        prefix = f"{modifiers} " if modifiers else ""
        return f"{prefix}enum {name}"

    def _method_signature(self, node: Node, src: bytes) -> str:
        """Build 'public async Task<IActionResult> MethodName(params)' signature."""
        modifiers = " ".join(self._get_modifiers(node, src))
        name_node = self._get_child_by_type(node, "identifier")
        name = self._txt(name_node, src) if name_node else "?"
        params_node = self._get_child_by_type(node, "parameter_list")
        params = self._txt(params_node, src) if params_node else "()"
        return_type = self._get_return_type_text(node, src, name)
        prefix = f"{modifiers} " if modifiers else ""
        return f"{prefix}{return_type} {name}{params}".strip()

    def _property_signature(self, node: Node, src: bytes) -> str:
        """Build 'public string Name { get; set; }' signature (first line)."""
        modifiers = " ".join(self._get_modifiers(node, src))
        name_node = self._get_child_by_type(node, "identifier")
        name = self._txt(name_node, src) if name_node else "?"
        type_text = self._get_return_type_text(node, src, name)
        accessor = "{ get; set; }"
        accessor_node = self._get_child_by_type(node, "accessor_list")
        if accessor_node:
            accessor = self._txt(accessor_node, src).replace("\n", " ")
            if len(accessor) > 60:
                accessor = "{ ... }"
        prefix = f"{modifiers} " if modifiers else ""
        return f"{prefix}{type_text} {name} {accessor}".strip()

    # ------------------------------------------------------------------
    # XML doc extraction
    # ------------------------------------------------------------------

    def _get_preceding_xml_doc(
        self, siblings: list[Node], target_idx: int, src: bytes
    ) -> str | None:
        """Collect consecutive /// XML doc comment lines immediately before target."""
        lines: list[str] = []
        i = target_idx - 1
        while i >= 0:
            sib = siblings[i]
            if sib.type == "comment":
                text = self._txt(sib, src).strip()
                if text.startswith("///"):
                    lines.append(text[3:].strip())
                    i -= 1
                else:
                    break
            else:
                break
        if not lines:
            return None
        lines.reverse()
        combined = " ".join(lines)
        combined = re.sub(r"<[^>]+>", "", combined).strip()
        return combined if combined else None

    def _get_file_doc(self, root: Node, src: bytes) -> str | None:
        """Extract leading /// XML doc from the top of the file."""
        children = list(root.children)
        lines: list[str] = []
        for child in children:
            if child.type == "comment":
                text = self._txt(child, src).strip()
                if text.startswith("///"):
                    lines.append(text[3:].strip())
                elif text.startswith("//"):
                    break
                else:
                    break
            elif child.type in (
                "using_directive",
                "file_scoped_namespace_declaration",
                "namespace_declaration",
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
            ):
                break
        if not lines:
            return None
        combined = " ".join(lines)
        combined = re.sub(r"<[^>]+>", "", combined).strip()
        return combined if combined else None

    # ------------------------------------------------------------------
    # Decorator/attribute extraction
    # ------------------------------------------------------------------

    def _get_attribute_texts(self, node: Node, src: bytes) -> list[str]:
        """Extract [AttributeName] decorator texts from a declaration node."""
        result: list[str] = []
        for child in node.children:
            if child.type == "attribute_list":
                text = self._txt(child, src).strip()
                if len(text) > 200:
                    text = text[:200] + "..."
                result.append(text)
        return result

    # ------------------------------------------------------------------
    # Modifier / base-list helpers
    # ------------------------------------------------------------------

    def _get_modifiers(self, node: Node, src: bytes) -> list[str]:
        """Return list of modifier keywords (public, static, async, etc.)."""
        modifiers: list[str] = []
        for child in node.children:
            if child.type == "modifier":
                modifiers.append(self._txt(child, src).strip())
        return modifiers

    def _get_base_list(self, node: Node, src: bytes) -> list[str]:
        """Return the names from the base_list (base class + interfaces)."""
        base_list_node = self._get_child_by_type(node, "base_list")
        if not base_list_node:
            return []
        bases: list[str] = []
        for child in base_list_node.children:
            if child.type in ("identifier", "generic_name"):
                bases.append(self._txt(self._get_child_by_type(child, "identifier") or child, src))
            elif child.type == "qualified_name":
                ident = self._get_last_identifier(child, src)
                if ident:
                    bases.append(ident)
        return bases

    def _get_return_type_text(self, node: Node, src: bytes, method_name: str) -> str:
        """Extract the return type text from a method or property declaration."""
        found_name = False
        for child in reversed(node.children):
            if child.type == "identifier" and self._txt(child, src) == method_name:
                found_name = True
                continue
            if found_name and child.type not in ("modifier", "attribute_list", "comment", ";"):
                return self._txt(child, src)
        return ""

    # ------------------------------------------------------------------
    # Error counting
    # ------------------------------------------------------------------

    def _count_errors(self, node: Node) -> int:
        count = 1 if node.type == "ERROR" else 0
        for child in node.children:
            count += self._count_errors(child)
        return count

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _txt(self, node: Node, src: bytes) -> str:
        return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _get_child_by_type(self, node: Node, type_name: str) -> Node | None:
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    def _get_last_identifier(self, node: Node, src: bytes) -> str:
        """Return the text of the last identifier in a node."""
        last: Node | None = None
        for child in node.children:
            if child.type == "identifier":
                last = child
        return self._txt(last, src) if last else ""
