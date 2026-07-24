"""TypeScript / TSX parser — direct AST traversal using Tree-sitter.

Handles .ts and .tsx files. Extracts:
  - Classes (kind=CLASS) — regular and abstract, Angular @Component, @Injectable, etc.
  - Class property fields (kind=VARIABLE, parent_id=class) — @Input/@Output, typed fields
  - Methods inside classes (kind=METHOD, parent_id=class)
  - Top-level functions and exported functions (kind=FUNCTION)
  - Arrow functions assigned to const/let (kind=FUNCTION)
  - Interfaces (kind=INTERFACE) + property/method member signatures
  - Type aliases (kind=INTERFACE)
  - Enums (kind=ENUM) + members as CONSTANT
  - Namespaces / internal modules (kind=MODULE) — recursively walks body
  - Module-level constants: ALL_CAPS names + all exported consts (kind=CONSTANT)
  - Import statements → ImportEdge
  - Re-exports → ImportEdge
  - Call sites inside functions/methods → CallEdge

Parent linkage:
  parent_id in Symbol is set to the LOCAL INDEX in the symbols list during
  parsing. The Indexer remaps this to the actual DB id after insertion.
"""

from __future__ import annotations

import re

from tree_sitter import Node

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge

from .._grammar import load_language, make_parser
from ..base import BaseParser, ParseResult


class TypeScriptParser(BaseParser):
    """Tree-sitter based TypeScript/TSX parser using direct AST traversal."""

    # Cap on class field symbols extracted per class (avoids symbol flood).
    MAX_CLASS_FIELDS: int = 30
    # Cap on interface member symbols extracted per interface.
    MAX_INTERFACE_MEMBERS: int = 20

    def __init__(self, tsx: bool = False) -> None:
        lang_name = "tsx" if tsx else "typescript"
        self._ts_language = load_language(lang_name)
        self._parser = make_parser(lang_name)
        self._tsx = tsx

    @property
    def language_name(self) -> str:
        return "tsx" if self._tsx else "typescript"

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
        # raw_calls: (caller_local_idx, callee_name, line, callee_type_hint | None)
        raw_calls: list[tuple[int, str, int, str | None]] = []

        # Module docstring — leading JSDoc/block comment before first import/declaration
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
            CallEdge(
                caller_id=caller_idx,
                callee_name=name,
                line=line,
                callee_type_hint=type_hint,
            )
            for caller_idx, name, line, type_hint in raw_calls
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
        raw_calls: list[tuple[int, str, int, str | None]],
    ) -> None:
        for child in node.children:
            ntype = child.type
            if ntype == "import_statement":
                import_edges.extend(self._extract_import(child, src, file_id))
            elif ntype == "export_statement":
                self._handle_export(
                    child, src, file_id, symbols, import_edges, type_edges, raw_calls
                )
            elif ntype in ("class_declaration", "abstract_class_declaration"):
                self._handle_class(
                    child, src, file_id, symbols, import_edges, type_edges, raw_calls
                )
            elif ntype in ("function_declaration", "generator_function_declaration"):
                self._handle_function(child, src, file_id, symbols, raw_calls)
            elif ntype == "function_signature":
                self._handle_function_signature(child, src, file_id, symbols)
            elif ntype in ("lexical_declaration", "variable_declaration"):
                self._handle_var_decl(child, src, file_id, symbols, raw_calls=raw_calls)
            elif ntype == "interface_declaration":
                self._handle_interface(child, src, file_id, symbols, type_edges)
            elif ntype == "type_alias_declaration":
                self._handle_type_alias(child, src, file_id, symbols)
            elif ntype == "enum_declaration":
                self._handle_enum(child, src, file_id, symbols)
            elif ntype in ("internal_module", "module"):
                self._handle_namespace(
                    child, src, file_id, symbols, import_edges, type_edges, raw_calls
                )
            elif ntype == "ambient_declaration":
                self._handle_ambient(
                    child, src, file_id, symbols, import_edges, type_edges, raw_calls
                )

    def _handle_export(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        raw_calls: list[tuple[int, str, int, str | None]],
    ) -> None:
        """Unwrap export statement (including decorated exports) and dispatch."""
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

        # Collect decorators from the export_statement node
        decs = [
            self._extract_decorator_text(c, src) for c in node.children if c.type == "decorator"
        ]
        for child in node.children:
            ntype = child.type
            if ntype in ("class_declaration", "abstract_class_declaration"):
                self._handle_class(
                    child,
                    src,
                    file_id,
                    symbols,
                    import_edges,
                    type_edges,
                    raw_calls,
                    exported=True,
                    decorators=decs,
                )
            elif ntype in ("function_declaration", "generator_function_declaration"):
                self._handle_function(
                    child, src, file_id, symbols, raw_calls, exported=True, decorators=decs
                )
            elif ntype == "function_signature":
                self._handle_function_signature(child, src, file_id, symbols, exported=True)
            elif ntype in ("lexical_declaration", "variable_declaration"):
                self._handle_var_decl(
                    child, src, file_id, symbols, exported=True, raw_calls=raw_calls
                )
            elif ntype == "interface_declaration":
                self._handle_interface(child, src, file_id, symbols, type_edges, exported=True)
            elif ntype == "type_alias_declaration":
                self._handle_type_alias(child, src, file_id, symbols, exported=True)
            elif ntype == "enum_declaration":
                self._handle_enum(child, src, file_id, symbols, exported=True)
            elif ntype in ("internal_module", "module"):
                self._handle_namespace(
                    child, src, file_id, symbols, import_edges, type_edges, raw_calls, exported=True
                )
            elif ntype == "ambient_declaration":
                # export declare class Foo / export declare function foo()
                self._handle_ambient(
                    child, src, file_id, symbols, import_edges, type_edges, raw_calls
                )

    # ------------------------------------------------------------------
    # Class handling
    # ------------------------------------------------------------------

    # Regex to extract templateUrl and styleUrls paths from @Component decorator text
    _TEMPLATE_URL_RE = re.compile(r'templateUrl\s*:\s*[\'"]([^\'"]+)[\'"]')
    _STYLE_URLS_RE = re.compile(
        r'styleUrl[s]?\s*:\s*\[([^\]]*)\]|styleUrl[s]?\s*:\s*[\'"]([^\'"]+)[\'"]'
    )
    _STYLE_URL_PATH_RE = re.compile(r'[\'"]([^\'"]+)[\'"]')

    def _handle_class(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        raw_calls: list[tuple[int, str, int, str | None]],
        exported: bool = False,
        decorators: list[str] | None = None,
    ) -> None:
        # TypeScript class names use type_identifier; JS uses identifier
        name_node = self._get_child_by_type(node, "type_identifier") or self._get_child_by_type(
            node, "identifier"
        )
        if not name_node:
            return
        name = self._txt(name_node, src)

        # Collect decorators from the class_declaration itself (non-exported decorated classes)
        own_decs = [
            self._extract_decorator_text(c, src) for c in node.children if c.type == "decorator"
        ]
        all_decs = (decorators or []) + own_decs

        # Extract templateUrl / styleUrls from @Component/@NgModule decorators → ImportEdge
        # This links home.component.ts ↔ home.component.html and home.component.scss
        # so that import graph expansion pulls them together during retrieval.
        for dec_text in all_decs:
            m = self._TEMPLATE_URL_RE.search(dec_text)
            if m:
                import_edges.append(
                    ImportEdge(
                        file_id=file_id,
                        imported_from=m.group(1),
                        imported_names=[],
                    )
                )
            for sm in self._STYLE_URLS_RE.finditer(dec_text):
                # Group 1 = array content, group 2 = bare string value
                arr_content = sm.group(1) or sm.group(2) or ""
                for pm in self._STYLE_URL_PATH_RE.finditer(arr_content):
                    import_edges.append(
                        ImportEdge(
                            file_id=file_id,
                            imported_from=pm.group(1),
                            imported_names=[],
                        )
                    )

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

        # Extract inheritance / implements type edges
        heritage_node = self._get_child_by_type(node, "class_heritage")
        if heritage_node:
            for clause in heritage_node.children:
                if clause.type == "extends_clause":
                    for c in clause.children:
                        if c.type in ("identifier", "type_identifier"):
                            type_edges.append(
                                TypeEdge(
                                    from_symbol_id=class_local_idx,
                                    to_type_name=self._txt(c, src),
                                    edge_kind="extends",
                                )
                            )
                        elif c.type == "generic_type":
                            tn = self._get_child_by_type(c, "type_identifier")
                            if tn:
                                type_edges.append(
                                    TypeEdge(
                                        from_symbol_id=class_local_idx,
                                        to_type_name=self._txt(tn, src),
                                        edge_kind="extends",
                                    )
                                )
                elif clause.type == "implements_clause":
                    for c in clause.children:
                        if c.type in ("identifier", "type_identifier"):
                            type_edges.append(
                                TypeEdge(
                                    from_symbol_id=class_local_idx,
                                    to_type_name=self._txt(c, src),
                                    edge_kind="implements",
                                )
                            )
                        elif c.type == "generic_type":
                            tn = self._get_child_by_type(c, "type_identifier")
                            if tn:
                                type_edges.append(
                                    TypeEdge(
                                        from_symbol_id=class_local_idx,
                                        to_type_name=self._txt(tn, src),
                                        edge_kind="implements",
                                    )
                                )

        # Walk class body to extract methods and property fields
        body_node = self._get_child_by_type(node, "class_body")
        if body_node:
            field_count = 0
            for child in body_node.children:
                if child.type == "method_definition":
                    self._handle_method(
                        child, src, file_id, symbols, class_local_idx, name, raw_calls
                    )
                elif child.type in ("abstract_method_signature", "method_signature"):
                    # abstract or overload declaration — no body, record as METHOD for lookup
                    self._handle_abstract_method(
                        child, src, file_id, symbols, class_local_idx, name
                    )
                elif child.type == "class_static_block":
                    # static { ... } initializer — walk body for calls, attributed to the class
                    self._walk_body(child, src, class_local_idx, raw_calls)
                elif (
                    child.type == "public_field_definition" and field_count < self.MAX_CLASS_FIELDS
                ):
                    self._handle_class_field(child, src, file_id, symbols, class_local_idx, name)
                    field_count += 1

    def _handle_method(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        class_local_idx: int,
        class_name: str,
        raw_calls: list[tuple[int, str, int, str | None]],
    ) -> None:
        name_node = self._get_child_by_type(node, "property_identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        # TS accessibility_modifier: "public" | "private" | "protected"
        vis_node = self._get_child_by_type(node, "accessibility_modifier")
        if vis_node:
            is_public = self._txt(vis_node, src) == "public"
        else:
            is_public = True  # TS default is public

        method_decs = [
            self._extract_decorator_text(c, src) for c in node.children if c.type == "decorator"
        ]

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
                decorators=method_decs,
                is_public=is_public,
            )
        )

        # Walk method body for call edges
        body_node = self._get_child_by_type(node, "statement_block")
        if body_node:
            params_node = self._get_child_by_type(node, "formal_parameters")
            method_param_types: dict[str, str] = (
                self._extract_param_types(params_node, src) if params_node else {}
            )
            self._walk_body(body_node, src, func_local_idx, raw_calls, method_param_types)

    def _handle_class_field(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        class_local_idx: int,
        class_name: str,
    ) -> None:
        """Extract a TypeScript class property declaration as a VARIABLE symbol.

        Handles: typed fields (name: string), @Input()/@Output() decorated fields,
        readonly properties (readonly id: number), optional fields (title?: string).
        Skips single-underscore private fields to reduce noise.
        """
        name_node = self._get_child_by_type(node, "property_identifier") or self._get_child_by_type(
            node, "identifier"
        )
        if not name_node:
            return
        name = self._txt(name_node, src)

        # Skip single-underscore private fields; keep dunder (__slots__ etc.)
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            return

        vis_node = self._get_child_by_type(node, "accessibility_modifier")
        is_public = True if not vis_node else self._txt(vis_node, src) == "public"

        # Decorators on the field: @Input(), @Output(), @InjectRepository(), etc.
        field_decs = [
            self._extract_decorator_text(c, src) for c in node.children if c.type == "decorator"
        ]

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
                decorators=field_decs,
                is_public=is_public,
            )
        )

    # ------------------------------------------------------------------
    # Interface / type alias / enum / namespace handling
    # ------------------------------------------------------------------

    def _handle_interface(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        type_edges: list[TypeEdge],
        exported: bool = False,
    ) -> None:
        name_node = self._get_child_by_type(node, "type_identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        interface_local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.INTERFACE,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=self._interface_signature(node, src),
                body=self._txt(node, src),
                docstring=self._get_preceding_comment(node, src),
                is_public=exported,
            )
        )

        # Extract extends type edges
        for child in node.children:
            if child.type == "extends_type_clause":
                for c in child.children:
                    if c.type == "type_identifier":
                        type_edges.append(
                            TypeEdge(
                                from_symbol_id=interface_local_idx,
                                to_type_name=self._txt(c, src),
                                edge_kind="extends",
                            )
                        )

        # Extract method_signature and property_signature members
        body_node = self._get_child_by_type(node, "interface_body")
        if body_node:
            self._extract_interface_members(
                body_node, src, file_id, symbols, interface_local_idx, name, exported
            )

    def _extract_interface_members(
        self,
        body_node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        interface_local_idx: int,
        interface_name: str,
        exported: bool,
    ) -> None:
        """Extract method_signature and property_signature children from an interface body."""
        count = 0
        for child in body_node.children:
            if count >= self.MAX_INTERFACE_MEMBERS:
                break
            if child.type in ("method_signature", "property_signature"):
                name_node = self._get_child_by_type(
                    child, "property_identifier"
                ) or self._get_child_by_type(child, "identifier")
                if not name_node:
                    continue
                member_name = self._txt(name_node, src)
                body = self._txt(child, src)
                kind = (
                    SymbolKind.FUNCTION if child.type == "method_signature" else SymbolKind.VARIABLE
                )
                symbols.append(
                    Symbol(
                        file_id=file_id,
                        name=member_name,
                        qualified_name=f"{interface_name}.{member_name}",
                        kind=kind,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        signature=body[:200],
                        body=body,
                        parent_id=interface_local_idx,
                        is_public=exported,
                    )
                )
                count += 1

    def _handle_type_alias(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        exported: bool = False,
    ) -> None:
        name_node = self._get_child_by_type(node, "type_identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        body = self._txt(node, src)
        if len(body) > 500:
            body = body[:500] + "..."

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.INTERFACE,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=body.split("\n")[0][:200],
                body=body,
                docstring=self._get_preceding_comment(node, src),
                is_public=exported,
            )
        )

    def _handle_enum(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        exported: bool = False,
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

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
                body=self._txt(node, src),
                docstring=self._get_preceding_comment(node, src),
                is_public=exported,
            )
        )

        # Extract enum members as CONSTANT symbols
        body_node = self._get_child_by_type(node, "enum_body")
        if body_node:
            for child in body_node.children:
                if child.type == "enum_assignment":
                    member_name_node = self._get_child_by_type(
                        child, "property_identifier"
                    ) or self._get_child_by_type(child, "identifier")
                    if member_name_node:
                        member_name = self._txt(member_name_node, src)
                        symbols.append(
                            Symbol(
                                file_id=file_id,
                                name=member_name,
                                qualified_name=f"{name}.{member_name}",
                                kind=SymbolKind.CONSTANT,
                                line_start=child.start_point[0] + 1,
                                line_end=child.end_point[0] + 1,
                                signature=f"{name}.{member_name}",
                                body=self._txt(child, src),
                                parent_id=enum_local_idx,
                                is_public=exported,
                            )
                        )
                elif child.type == "property_identifier":
                    # Simple enum member without value: enum Dir { Up, Down }
                    member_name = self._txt(child, src)
                    symbols.append(
                        Symbol(
                            file_id=file_id,
                            name=member_name,
                            qualified_name=f"{name}.{member_name}",
                            kind=SymbolKind.CONSTANT,
                            line_start=child.start_point[0] + 1,
                            line_end=child.end_point[0] + 1,
                            signature=f"{name}.{member_name}",
                            body=member_name,
                            parent_id=enum_local_idx,
                            is_public=exported,
                        )
                    )

    def _handle_namespace(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        raw_calls: list[tuple[int, str, int, str | None]],
        exported: bool = False,
    ) -> None:
        """Handle TypeScript namespace/module blocks: namespace Foo { } / module "foo" { }"""
        # namespace/internal_module use identifier; declare module "foo" uses string
        name_node = self._get_child_by_type(node, "identifier") or self._get_child_by_type(
            node, "string"
        )
        if not name_node:
            return
        name = self._txt(name_node, src).strip("'\"")  # strip quotes for module "foo"

        body_text = self._txt(node, src)
        if len(body_text) > 500:
            body_text = body_text[:500] + "..."

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.MODULE,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"namespace {name}",
                body=body_text,
                docstring=self._get_preceding_comment(node, src),
                is_public=exported,
            )
        )

        # Recurse into namespace body — declarations inside are full first-class symbols
        body_node = self._get_child_by_type(node, "statement_block")
        if body_node:
            self._walk_top_level(
                body_node, src, file_id, symbols, import_edges, type_edges, raw_calls
            )

    # ------------------------------------------------------------------
    # Function handling
    # ------------------------------------------------------------------

    def _handle_function(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int, str, int, str | None]],
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
            params_node = self._get_child_by_type(node, "formal_parameters")
            func_param_types: dict[str, str] = (
                self._extract_param_types(params_node, src) if params_node else {}
            )
            self._walk_body(body_node, src, func_local_idx, raw_calls, func_param_types)

    def _handle_var_decl(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        exported: bool = False,
        raw_calls: list[tuple[int, str, int, str | None]] | None = None,
    ) -> None:
        """Handle const/let declarations — arrow functions AND module-level constants."""
        for child in node.children:
            if child.type != "variable_declarator":
                continue
            name_node = self._get_child_by_type(child, "identifier")
            fn_node = None
            for c in child.children:
                if c.type in ("arrow_function", "function"):
                    fn_node = c
                    break
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
                        is_public=exported,
                    )
                )
                # Walk arrow function body for call edges
                if raw_calls is not None:
                    body_node = fn_node.child_by_field_name("body")
                    if body_node:
                        # body may be statement_block or bare expression (x => foo(x))
                        self._walk_node(body_node, src, func_local_idx, raw_calls)
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
                        )
                    )

    # ------------------------------------------------------------------
    # Call extraction
    # ------------------------------------------------------------------

    # Node types that start a new declaration scope — we do NOT recurse into them
    _SCOPE_BOUNDARY = frozenset(
        {
            "function_declaration",
            "generator_function_declaration",
            "function_expression",
            "generator_function",
            "class_declaration",
            "abstract_class_declaration",
            "class_expression",
        }
    )

    def _walk_body(
        self,
        node: Node,
        src: bytes,
        current_func_local_idx: int,
        raw_calls: list[tuple[int, str, int, str | None]],
        param_types: dict[str, str] | None = None,
    ) -> None:
        """Iterate over `node`'s children and dispatch each through `_walk_node`."""
        pt = param_types or {}
        for child in node.children:
            self._walk_node(child, src, current_func_local_idx, raw_calls, pt)

    def _walk_node(
        self,
        child: Node,
        src: bytes,
        current_func_local_idx: int,
        raw_calls: list[tuple[int, str, int, str | None]],
        param_types: dict[str, str] | None = None,
    ) -> None:
        """Process a single AST node: extract calls and recurse into sub-expressions.

        Arrow function expression bodies (e.g. ``x => foo(x)``) pass the body node
        directly here so it is handled correctly whether it is a statement_block or a
        bare expression.

        ``param_types`` maps parameter names to their TypeScript type annotation names
        (e.g. ``{"authService": "AuthService", "db": "Database"}``).  When a method
        call ``receiver.method()`` is found and the receiver name is in ``param_types``,
        the type is stored as ``callee_type_hint`` on the CallEdge so resolution can
        use the type-hint priority-2 path.
        """
        pt = param_types or {}
        ntype = child.type
        if ntype == "call_expression":
            fn_node = child.child_by_field_name("function")
            if fn_node:
                if fn_node.type == "identifier":
                    raw_calls.append(
                        (
                            current_func_local_idx,
                            self._txt(fn_node, src),
                            child.start_point[0] + 1,
                            None,
                        )
                    )
                elif fn_node.type in ("member_expression", "optional_chain"):
                    prop = fn_node.child_by_field_name("property")
                    if prop:
                        # Try to resolve receiver type hint
                        type_hint: str | None = None
                        obj_node = fn_node.child_by_field_name("object")
                        if obj_node and obj_node.type == "identifier":
                            receiver_name = self._txt(obj_node, src)
                            type_hint = pt.get(receiver_name)
                        elif obj_node and obj_node.type == "this":
                            pass  # this.method() — type hint not applicable
                        raw_calls.append(
                            (
                                current_func_local_idx,
                                self._txt(prop, src),
                                child.start_point[0] + 1,
                                type_hint,
                            )
                        )
            # Recurse into args for nested calls: foo(bar())
            self._walk_body(child, src, current_func_local_idx, raw_calls, pt)
        elif ntype == "new_expression":
            # new SomeClass() / new pkg.Class() — constructor call
            ctor = child.child_by_field_name("constructor")
            if ctor:
                if ctor.type == "identifier":
                    raw_calls.append(
                        (
                            current_func_local_idx,
                            self._txt(ctor, src),
                            child.start_point[0] + 1,
                            None,
                        )
                    )
                elif ctor.type in ("member_expression", "nested_identifier"):
                    prop = ctor.child_by_field_name("property")
                    if prop:
                        raw_calls.append(
                            (
                                current_func_local_idx,
                                self._txt(prop, src),
                                child.start_point[0] + 1,
                                None,
                            )
                        )
            self._walk_body(child, src, current_func_local_idx, raw_calls, pt)
        elif ntype == "arrow_function":
            # Attribute calls in nested arrow functions to the enclosing symbol.
            # body may be a statement_block OR a bare expression (e.g. x => foo(x)).
            body_node = child.child_by_field_name("body")
            if body_node:
                self._walk_node(body_node, src, current_func_local_idx, raw_calls, pt)
        elif ntype not in self._SCOPE_BOUNDARY:
            # Generic container (if, for, block, etc.) — recurse
            self._walk_body(child, src, current_func_local_idx, raw_calls, pt)

    def _handle_ambient(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        raw_calls: list[tuple[int, str, int, str | None]],
    ) -> None:
        """Handle `declare ...` ambient declarations (common in .d.ts files).

        Unwraps to the inner declaration and dispatches normally, so all
        declared classes, functions, consts, interfaces, and modules are indexed.
        """
        for child in node.children:
            ntype = child.type
            if ntype in ("class_declaration", "abstract_class_declaration"):
                self._handle_class(
                    child, src, file_id, symbols, import_edges, type_edges, raw_calls, exported=True
                )
            elif ntype in ("function_declaration", "generator_function_declaration"):
                self._handle_function(child, src, file_id, symbols, raw_calls, exported=True)
            elif ntype == "function_signature":
                self._handle_function_signature(child, src, file_id, symbols, exported=True)
            elif ntype in ("lexical_declaration", "variable_declaration"):
                self._handle_var_decl(
                    child, src, file_id, symbols, exported=True, raw_calls=raw_calls
                )
            elif ntype == "interface_declaration":
                self._handle_interface(child, src, file_id, symbols, type_edges, exported=True)
            elif ntype == "type_alias_declaration":
                self._handle_type_alias(child, src, file_id, symbols, exported=True)
            elif ntype == "enum_declaration":
                self._handle_enum(child, src, file_id, symbols, exported=True)
            elif ntype in ("internal_module", "module"):
                self._handle_namespace(
                    child, src, file_id, symbols, import_edges, type_edges, raw_calls, exported=True
                )

    def _handle_function_signature(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        exported: bool = False,
    ) -> None:
        """Handle function overload / ambient function declaration (no body)."""
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
                kind=SymbolKind.FUNCTION,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=body[:200],
                body=body,
                docstring=self._get_preceding_comment(node, src),
                is_public=exported,
            )
        )

    def _handle_abstract_method(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        class_local_idx: int,
        class_name: str,
    ) -> None:
        """Extract abstract method declaration as a METHOD symbol (no body to walk)."""
        name_node = self._get_child_by_type(node, "property_identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)
        vis_node = self._get_child_by_type(node, "accessibility_modifier")
        is_public = True if not vis_node else self._txt(vis_node, src) == "public"
        body = self._txt(node, src)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=f"{class_name}.{name}",
                kind=SymbolKind.METHOD,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=body[:200],
                body=body,
                parent_id=class_local_idx,
                docstring=self._get_preceding_comment(node, src),
                is_public=is_public,
            )
        )

    # ------------------------------------------------------------------
    # Parameter type extraction (for callee_type_hint)
    # ------------------------------------------------------------------

    def _extract_param_types(self, params_node: Node, src: bytes) -> dict[str, str]:
        """
        Return ``{parameter_name: type_name}`` for a TypeScript function's
        ``formal_parameters`` node.  Only simple identifier type annotations are
        captured (e.g. ``authService: AuthService``); generic types (``Array<Foo>``)
        and union types are skipped because they cannot be reliably matched against
        a single qualified_name prefix.

        Examples:
          ``(authService: AuthService, db: Database)``
          → ``{"authService": "AuthService", "db": "Database"}``

          ``(items: string[], count: number)``
          → ``{}``  (array/primitive types don't correspond to indexed symbols)
        """
        result: dict[str, str] = {}
        for param in params_node.children:
            if param.type not in ("required_parameter", "optional_parameter"):
                continue
            name_node = self._get_child_by_type(param, "identifier") or self._get_child_by_type(
                param, "this"
            )
            type_node = param.child_by_field_name("type")
            if not name_node or not type_node:
                continue
            param_name = self._txt(name_node, src)
            if param_name == "this":
                continue  # TS `this` parameter — skip
            # Drill into the type_annotation wrapper
            inner = self._get_child_by_type(
                type_node, "type_identifier"
            ) or self._get_child_by_type(type_node, "identifier")
            if inner:
                result[param_name] = self._txt(inner, src)
        return result

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_import(self, node: Node, src: bytes, file_id: int) -> list[ImportEdge]:
        """Handle ES6 import_statement → ImportEdge.

        Also handles: import Foo = require("./mod")  which tree-sitter parses as
        import_statement > import_require_clause > string  (NOT import_alias).
        """
        source_node: Node | None = None
        imported_names: list[str] = []

        for child in node.children:
            if child.type == "string":
                source_node = child
            elif child.type == "import_clause":
                imported_names = self._extract_import_clause(child, src)
            elif child.type == "import_require_clause":
                # import Foo = require("./module")
                str_node = self._get_child_by_type(child, "string")
                if str_node:
                    module_name = self._txt(str_node, src).strip("'\"")
                    return [
                        ImportEdge(
                            file_id=file_id,
                            imported_from=module_name,
                            imported_names=[],
                        )
                    ]

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
                # Default import: import Foo from '...'
                names.append(self._txt(child, src))
            elif child.type == "named_imports":
                for spec in child.children:
                    if spec.type == "import_specifier":
                        id_node = self._get_child_by_type(spec, "identifier")
                        if id_node:
                            names.append(self._txt(id_node, src))
            elif child.type == "namespace_import":
                # import * as Foo from '...'
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
        return_node = self._get_child_by_type(node, "type_annotation")

        name = self._txt(name_node, src) if name_node else "?"
        params = self._txt(params_node, src) if params_node else "()"
        ret = self._txt(return_node, src) if return_node else ""
        return f"function {name}{params}{ret}"

    def _method_signature(self, node: Node, src: bytes, class_name: str) -> str:
        name_node = self._get_child_by_type(node, "property_identifier")
        params_node = node.child_by_field_name("parameters")
        return_node = self._get_child_by_type(node, "type_annotation")

        name = self._txt(name_node, src) if name_node else "?"
        params = self._txt(params_node, src) if params_node else "()"
        ret = self._txt(return_node, src) if return_node else ""
        return f"{class_name}.{name}{params}{ret}"

    def _arrow_signature(self, name_node: Node, fn_node: Node, src: bytes) -> str:
        name = self._txt(name_node, src)
        params_node = fn_node.child_by_field_name("parameters") or self._get_child_by_type(
            fn_node, "formal_parameters"
        )
        params = self._txt(params_node, src) if params_node else "(...)"
        return f"const {name} = {params} => ..."

    def _interface_signature(self, node: Node, src: bytes) -> str:
        name_node = self._get_child_by_type(node, "type_identifier")
        name = self._txt(name_node, src) if name_node else "?"
        return f"interface {name}"

    def _class_signature(self, node: Node, src: bytes) -> str:
        name_node = self._get_child_by_type(node, "type_identifier") or self._get_child_by_type(
            node, "identifier"
        )
        heritage_node = self._get_child_by_type(node, "class_heritage")

        name = self._txt(name_node, src) if name_node else "?"
        heritage = f" {self._txt(heritage_node, src)}" if heritage_node else ""
        prefix = "abstract class" if node.type == "abstract_class_declaration" else "class"
        return f"{prefix} {name}{heritage}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_decorator_text(self, node: Node, src: bytes) -> str:
        """Extract human-readable decorator text from a decorator node, e.g. 'Injectable()'."""
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
