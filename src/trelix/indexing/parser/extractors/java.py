"""Java parser — direct AST traversal using Tree-sitter.

Targets JDK 16+ (records) and JDK 17+ (sealed classes). Extracts:
  - Classes (kind=CLASS) — regular, abstract, sealed
  - Records (kind=CLASS) — Java 16+ records with component fields
  - Interfaces (kind=INTERFACE) — with default/abstract method members
  - Enums (kind=ENUM) + enum constants as CONSTANT
  - Annotation types / @interface (kind=INTERFACE)
  - Methods inside classes/interfaces/records (kind=METHOD, parent_id=class)
  - Constructors (kind=METHOD, parent_id=class)
  - Static final fields (kind=CONSTANT, parent_id=class)
  - Annotated/public instance fields (kind=VARIABLE, parent_id=class)
    — surfaces @Autowired, @Column, @Id, @Value dependency injection fields
  - Record components (kind=VARIABLE, parent_id=record)
  - Call sites inside methods/constructors → CallEdge
  - Import declarations → ImportEdge

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


class JavaParser(BaseParser):
    """Tree-sitter based Java parser using direct AST traversal."""

    # Cap on field symbols extracted per class/record (avoids symbol flood).
    MAX_FIELDS: int = 30

    def __init__(self) -> None:
        self._ts_language = tree_sitter_languages.get_language("java")
        self._parser = Parser()
        self._parser.set_language(self._ts_language)

    @property
    def language_name(self) -> str:
        return "java"

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
            if ntype == "import_declaration":
                import_edges.extend(self._extract_import(child, src, file_id))
            elif ntype == "class_declaration":
                self._handle_class(child, src, file_id, symbols, type_edges, raw_calls)
            elif ntype == "interface_declaration":
                self._handle_interface(child, src, file_id, symbols, type_edges, raw_calls)
            elif ntype == "enum_declaration":
                self._handle_enum(child, src, file_id, symbols, type_edges, raw_calls)
            elif ntype == "record_declaration":
                self._handle_record(child, src, file_id, symbols, type_edges, raw_calls)
            elif ntype == "annotation_type_declaration":
                self._handle_annotation_type(child, src, file_id, symbols)

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
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        modifiers_node = self._get_child_by_type(node, "modifiers")
        is_public = "public" in self._txt(modifiers_node, src) if modifiers_node else False
        annotations = self._get_annotations(modifiers_node, src) if modifiers_node else []

        class_local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.CLASS,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=self._class_signature(node, src),
                body=self._txt(node, src),
                docstring=self._get_preceding_comment(node, src),
                decorators=annotations,
                is_public=is_public,
            )
        )

        # Extract superclass (extends) and interfaces (implements) type edges
        superclass_node = node.child_by_field_name("superclass")
        if superclass_node:
            self._extract_type_list_edges(
                superclass_node, class_local_idx, "extends", type_edges, src
            )
        interfaces_node = node.child_by_field_name("interfaces")
        if interfaces_node:
            self._extract_type_list_edges(
                interfaces_node, class_local_idx, "implements", type_edges, src
            )

        body_node = self._get_child_by_type(node, "class_body")
        if body_node:
            field_count = 0
            for child in body_node.children:
                if child.type == "method_declaration":
                    self._handle_method(
                        child, src, file_id, symbols, class_local_idx, name, raw_calls
                    )
                elif child.type == "constructor_declaration":
                    self._handle_constructor(
                        child, src, file_id, symbols, class_local_idx, name, raw_calls
                    )
                elif child.type == "field_declaration" and field_count < self.MAX_FIELDS:
                    before = len(symbols)
                    self._handle_field_decl(child, src, file_id, symbols, class_local_idx, name)
                    field_count += len(symbols) - before
                elif child.type == "class_declaration":
                    self._handle_class(child, src, file_id, symbols, type_edges, raw_calls)
                elif child.type == "interface_declaration":
                    self._handle_interface(child, src, file_id, symbols, type_edges, raw_calls)
                elif child.type == "enum_declaration":
                    self._handle_enum(child, src, file_id, symbols, type_edges, raw_calls)
                elif child.type == "record_declaration":
                    self._handle_record(child, src, file_id, symbols, type_edges, raw_calls)
                elif child.type == "annotation_type_declaration":
                    self._handle_annotation_type(child, src, file_id, symbols)

    def _handle_interface(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        type_edges: list[TypeEdge],
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        modifiers_node = self._get_child_by_type(node, "modifiers")
        is_public = "public" in self._txt(modifiers_node, src) if modifiers_node else False
        annotations = self._get_annotations(modifiers_node, src) if modifiers_node else []

        interface_local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.INTERFACE,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"interface {name}",
                body=self._txt(node, src),
                docstring=self._get_preceding_comment(node, src),
                decorators=annotations,
                is_public=is_public,
            )
        )

        # interface Foo extends Bar, Baz → TypeEdges
        extends_node = node.child_by_field_name("extends_interfaces")
        if extends_node:
            self._extract_type_list_edges(
                extends_node, interface_local_idx, "extends", type_edges, src
            )

        body_node = self._get_child_by_type(node, "interface_body")
        if body_node:
            field_count = 0
            for child in body_node.children:
                if child.type == "method_declaration":
                    self._handle_method(
                        child, src, file_id, symbols, interface_local_idx, name, raw_calls
                    )
                elif child.type == "constant_declaration" and field_count < self.MAX_FIELDS:
                    # Interface constants: always public static final
                    before = len(symbols)
                    self._handle_interface_constant(
                        child, src, file_id, symbols, interface_local_idx, name
                    )
                    field_count += len(symbols) - before

    def _handle_interface_constant(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        interface_local_idx: int,
        interface_name: str,
    ) -> None:
        """Interface constants are implicitly public static final."""
        for child in node.children:
            if child.type != "variable_declarator":
                continue
            name_node = self._get_child_by_type(child, "identifier")
            if not name_node:
                continue
            name = self._txt(name_node, src)
            body = self._txt(node, src)
            if len(body) > 500:
                body = body[:500] + "..."
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=f"{interface_name}.{name}",
                    kind=SymbolKind.CONSTANT,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=body.split("\n")[0][:200].strip(),
                    body=body,
                    parent_id=interface_local_idx,
                    is_public=True,
                )
            )

    # ------------------------------------------------------------------
    # Record handling (Java 16+)
    # ------------------------------------------------------------------

    def _handle_record(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        type_edges: list[TypeEdge],
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        """Handle Java 16+ record declarations.

        record Point(int x, int y) implements Serializable { ... }
        Components (x, y) are extracted as VARIABLE symbols — they are both
        fields and constructor parameters, making them the primary query surface.
        """
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        modifiers_node = self._get_child_by_type(node, "modifiers")
        is_public = "public" in self._txt(modifiers_node, src) if modifiers_node else False
        annotations = self._get_annotations(modifiers_node, src) if modifiers_node else []

        record_local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.CLASS,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=self._record_signature(node, src),
                body=self._txt(node, src),
                docstring=self._get_preceding_comment(node, src),
                decorators=annotations,
                is_public=is_public,
            )
        )

        # implements type edges
        interfaces_node = node.child_by_field_name("interfaces")
        if interfaces_node:
            for c in interfaces_node.children:
                if c.type == "type_identifier":
                    type_edges.append(
                        TypeEdge(
                            from_symbol_id=record_local_idx,
                            to_type_name=self._txt(c, src),
                            edge_kind="implements",
                        )
                    )

        # Record components → VARIABLE symbols (always public, immutable fields)
        params_node = self._get_child_by_type(node, "record_parameters")
        if params_node:
            for comp in params_node.children:
                if comp.type == "record_component":
                    comp_name_node = self._get_child_by_type(comp, "identifier")
                    if comp_name_node:
                        comp_name = self._txt(comp_name_node, src)
                        symbols.append(
                            Symbol(
                                file_id=file_id,
                                name=comp_name,
                                qualified_name=f"{name}.{comp_name}",
                                kind=SymbolKind.VARIABLE,
                                line_start=comp.start_point[0] + 1,
                                line_end=comp.end_point[0] + 1,
                                signature=self._txt(comp, src)[:200],
                                body=self._txt(comp, src),
                                parent_id=record_local_idx,
                                is_public=True,
                            )
                        )

        # Walk body for methods and compact constructor
        body_node = self._get_child_by_type(node, "class_body")
        if body_node:
            for child in body_node.children:
                if child.type in ("method_declaration", "compact_constructor_declaration"):
                    self._handle_method(
                        child, src, file_id, symbols, record_local_idx, name, raw_calls
                    )

    # ------------------------------------------------------------------
    # Annotation type (@interface)
    # ------------------------------------------------------------------

    def _handle_annotation_type(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
    ) -> None:
        """Handle @interface annotation type declarations."""
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        modifiers_node = self._get_child_by_type(node, "modifiers")
        is_public = "public" in self._txt(modifiers_node, src) if modifiers_node else False

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.INTERFACE,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"@interface {name}",
                body=self._txt(node, src),
                docstring=self._get_preceding_comment(node, src),
                is_public=is_public,
            )
        )

    # ------------------------------------------------------------------
    # Method / constructor handling
    # ------------------------------------------------------------------

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
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        mod_node = self._get_child_by_type(node, "modifiers")
        method_annotations = self._get_annotations(mod_node, src) if mod_node else []
        method_is_public = "public" in self._txt(mod_node, src) if mod_node else False

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
                decorators=method_annotations,
                is_public=method_is_public,
            )
        )

        # Walk method body for call edges
        body_node = self._get_child_by_type(node, "block")
        if body_node:
            self._walk_body(body_node, src, func_local_idx, raw_calls)

    def _handle_constructor(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        class_local_idx: int,
        class_name: str,
        raw_calls: list[tuple[int, str, int]],
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)
        params_node = self._get_child_by_type(node, "formal_parameters")
        params = self._txt(params_node, src) if params_node else "()"

        ctor_mod = self._get_child_by_type(node, "modifiers")
        ctor_is_public = "public" in self._txt(ctor_mod, src) if ctor_mod else False

        func_local_idx = len(symbols)
        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=f"{class_name}.{name}",
                kind=SymbolKind.METHOD,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=f"{class_name}{params}",
                body=self._txt(node, src),
                parent_id=class_local_idx,
                docstring=self._get_preceding_comment(node, src),
                is_public=ctor_is_public,
            )
        )

        # Walk constructor body for call edges
        body_node = self._get_child_by_type(node, "constructor_body")
        if body_node:
            self._walk_body(body_node, src, func_local_idx, raw_calls)

    # ------------------------------------------------------------------
    # Field handling
    # ------------------------------------------------------------------

    def _handle_field_decl(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        class_local_idx: int,
        class_name: str,
    ) -> None:
        """Extract field declarations:
          - static final → CONSTANT (public static final String VERSION = "1.0")
          - annotated → VARIABLE (@Autowired, @Column, @Id, @Value fields)
          - public/protected non-static → VARIABLE (visible API fields)
        Skips unannotated private/package-private non-static-final fields (noise).
        """
        modifiers_node = self._get_child_by_type(node, "modifiers")
        mods_text = self._txt(modifiers_node, src) if modifiers_node else ""
        annotations = self._get_annotations(modifiers_node, src) if modifiers_node else []

        is_static_final = "static" in mods_text and "final" in mods_text
        has_annotations = bool(annotations)
        is_public = "public" in mods_text or "protected" in mods_text

        # Skip unremarkable private non-annotated instance fields
        if not is_static_final and not has_annotations and not is_public:
            return

        kind = SymbolKind.CONSTANT if is_static_final else SymbolKind.VARIABLE

        for child in node.children:
            if child.type != "variable_declarator":
                continue
            name_node = self._get_child_by_type(child, "identifier")
            if not name_node:
                continue
            name = self._txt(name_node, src)
            body = self._txt(node, src)
            if len(body) > 500:
                body = body[:500] + "..."
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=f"{class_name}.{name}",
                    kind=kind,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=body.split("\n")[0][:200].strip(),
                    body=body,
                    parent_id=class_local_idx,
                    decorators=annotations,
                    is_public=is_public,
                )
            )

    def _handle_enum(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        type_edges: list[TypeEdge] | None = None,
        raw_calls: list[tuple[int, str, int]] | None = None,
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        modifiers_node = self._get_child_by_type(node, "modifiers")
        is_public = "public" in self._txt(modifiers_node, src) if modifiers_node else False
        annotations = self._get_annotations(modifiers_node, src) if modifiers_node else []

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
                decorators=annotations,
                is_public=is_public,
            )
        )

        # enum Status implements Serializable → TypeEdges
        interfaces_node = node.child_by_field_name("interfaces")
        if interfaces_node and type_edges is not None:
            self._extract_type_list_edges(
                interfaces_node, enum_local_idx, "implements", type_edges, src
            )

        # Extract enum constants and body methods
        body_node = self._get_child_by_type(node, "enum_body")
        if body_node:
            for child in body_node.children:
                if child.type == "enum_constant":
                    cname_node = self._get_child_by_type(child, "identifier")
                    if cname_node:
                        cname = self._txt(cname_node, src)
                        symbols.append(
                            Symbol(
                                file_id=file_id,
                                name=cname,
                                qualified_name=f"{name}.{cname}",
                                kind=SymbolKind.CONSTANT,
                                line_start=child.start_point[0] + 1,
                                line_end=child.end_point[0] + 1,
                                signature=f"{name}.{cname}",
                                body=self._txt(child, src),
                                parent_id=enum_local_idx,
                                is_public=is_public,
                            )
                        )
                elif child.type == "enum_body_declarations":
                    # Methods declared inside the enum body
                    for decl in child.children:
                        if decl.type == "method_declaration" and raw_calls is not None:
                            self._handle_method(
                                decl, src, file_id, symbols, enum_local_idx, name, raw_calls
                            )
                        elif decl.type == "constructor_declaration" and raw_calls is not None:
                            self._handle_constructor(
                                decl, src, file_id, symbols, enum_local_idx, name, raw_calls
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
        """Recursively walk a method/constructor body collecting method_invocation nodes.

        Does NOT recurse into nested class/lambda definitions (own scope).
        """
        for child in node.children:
            if child.type == "method_invocation":
                name_node = child.child_by_field_name("name")
                if name_node:
                    raw_calls.append(
                        (
                            current_func_local_idx,
                            self._txt(name_node, src),
                            child.start_point[0] + 1,
                        )
                    )
                # Recurse into arguments for chained/nested calls
                self._walk_body(child, src, current_func_local_idx, raw_calls)
            elif child.type == "object_creation_expression":
                # new SomeService(...) — the type being instantiated is a callee
                type_node = child.child_by_field_name("type")
                if type_node:
                    type_name = self._txt(type_node, src).split("<")[0].strip()
                    if type_name:
                        raw_calls.append(
                            (
                                current_func_local_idx,
                                type_name,
                                child.start_point[0] + 1,
                            )
                        )
                self._walk_body(child, src, current_func_local_idx, raw_calls)
            elif child.type not in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
                "lambda_expression",
            ):
                self._walk_body(child, src, current_func_local_idx, raw_calls)

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_import(self, node: Node, src: bytes, file_id: int) -> list[ImportEdge]:
        """Handle: import java.util.List; import java.util.*;"""
        for child in node.children:
            if child.type in ("scoped_identifier", "identifier"):
                path = self._txt(child, src)
                parts = path.split(".")
                imported_name = parts[-1] if parts[-1] != "*" else "*"
                module = ".".join(parts[:-1]) if len(parts) > 1 else path
                return [
                    ImportEdge(
                        file_id=file_id,
                        imported_from=module,
                        imported_names=[imported_name],
                    )
                ]
        return []

    # ------------------------------------------------------------------
    # Signature builders
    # ------------------------------------------------------------------

    def _class_signature(self, node: Node, src: bytes) -> str:
        name_node = self._get_child_by_type(node, "identifier")
        modifiers_node = self._get_child_by_type(node, "modifiers")
        superclass_node = node.child_by_field_name("superclass")
        interfaces_node = node.child_by_field_name("interfaces")
        permits_node = node.child_by_field_name("permits")

        mods = self._txt(modifiers_node, src) + " " if modifiers_node else ""
        name = self._txt(name_node, src) if name_node else "?"
        extends = f" extends {self._txt(superclass_node, src)}" if superclass_node else ""
        implements = f" implements {self._txt(interfaces_node, src)}" if interfaces_node else ""
        permits = f" permits {self._txt(permits_node, src)}" if permits_node else ""
        return f"{mods}class {name}{extends}{implements}{permits}"

    def _record_signature(self, node: Node, src: bytes) -> str:
        name_node = self._get_child_by_type(node, "identifier")
        params_node = self._get_child_by_type(node, "record_parameters")
        modifiers_node = self._get_child_by_type(node, "modifiers")

        mods = self._txt(modifiers_node, src) + " " if modifiers_node else ""
        name = self._txt(name_node, src) if name_node else "?"
        params = self._txt(params_node, src) if params_node else "()"
        return f"{mods}record {name}{params}"

    def _method_signature(self, node: Node, src: bytes, class_name: str) -> str:
        name_node = self._get_child_by_type(node, "identifier")
        params_node = self._get_child_by_type(node, "formal_parameters")
        modifiers_node = self._get_child_by_type(node, "modifiers")
        return_type = self._get_return_type(node, src)

        mods = self._txt(modifiers_node, src) + " " if modifiers_node else ""
        name = self._txt(name_node, src) if name_node else "?"
        params = self._txt(params_node, src) if params_node else "()"
        ret = f"{return_type} " if return_type else ""
        return f"{mods}{ret}{class_name}.{name}{params}"

    def _get_return_type(self, method_node: Node, src: bytes) -> str:
        """Extract return type — the type node that precedes the method name."""
        type_types = {
            "type_identifier",
            "void_type",
            "generic_type",
            "array_type",
            "integral_type",
            "floating_point_type",
            "boolean_type",
            "primitive_type",
        }
        for child in method_node.children:
            if child.type in type_types:
                return self._txt(child, src)
        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_type_list_edges(
        self,
        node: Node,
        from_local_idx: int,
        edge_kind: str,
        type_edges: list[TypeEdge],
        src: bytes,
    ) -> None:
        """Extract TypeEdges from an extends_interfaces / interfaces / type_list node."""
        for c in node.children:
            if c.type == "type_identifier":
                type_edges.append(
                    TypeEdge(
                        from_symbol_id=from_local_idx,
                        to_type_name=self._txt(c, src),
                        edge_kind=edge_kind,
                    )
                )
            elif c.type == "type_list":
                for tc in c.children:
                    if tc.type == "type_identifier":
                        type_edges.append(
                            TypeEdge(
                                from_symbol_id=from_local_idx,
                                to_type_name=self._txt(tc, src),
                                edge_kind=edge_kind,
                            )
                        )
            elif c.type == "generic_type":
                # Foo<T> — extract base type name only
                ti = self._get_child_by_type(c, "type_identifier")
                if ti:
                    type_edges.append(
                        TypeEdge(
                            from_symbol_id=from_local_idx,
                            to_type_name=self._txt(ti, src),
                            edge_kind=edge_kind,
                        )
                    )

    def _get_annotations(self, modifiers_node: Node, src: bytes) -> list[str]:
        """Extract Java annotation names from a modifiers node."""
        result: list[str] = []
        for child in modifiers_node.children:
            if child.type in ("marker_annotation", "annotation"):
                name_node = self._get_child_by_type(child, "identifier")
                if name_node:
                    result.append(f"@{self._txt(name_node, src)}")
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
        """Find Javadoc/line comment immediately before this node."""
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
        """Strip comment delimiters: /** */, //, * line prefixes."""
        raw = re.sub(r"^/\*+\s*", "", raw.strip())
        raw = re.sub(r"\s*\*+/$", "", raw)
        raw = re.sub(r"^\s*\*\s?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"^//\s?", "", raw, flags=re.MULTILINE)
        return raw.strip()
