"""Kotlin parser: direct AST traversal using Tree-sitter.

Design: same pattern as Python/TypeScript/Java parsers — direct tree-sitter walk,
no .scm query files at runtime.

Extracts:
  - Classes (kind=CLASS) — regular, data, sealed, abstract, open
  - Interfaces (kind=INTERFACE) — `interface Foo`
  - Enums (kind=ENUM) — `enum class Color`, members → CONSTANT
  - Object declarations (kind=CLASS) — `object AppConfig` singletons
  - Companion objects — methods/constants attributed to parent class
  - Nested classes and inner objects
  - Top-level functions (kind=FUNCTION) — including extension functions
  - Methods inside classes (kind=METHOD, parent_id=class local idx)
  - Type aliases (kind=INTERFACE) — `typealias Names = List<String>`
  - Top-level `const val` and ALL_CAPS vals (kind=CONSTANT)
  - Class property declarations (kind=VARIABLE, parent_id=class local idx)
  - Import statements → ImportEdge
  - Call sites inside functions → CallEdge (caller_id = local idx, remapped by Indexer)
  - Supertype delegation specifiers → TypeEdge (extends / implements)
  - Kotlin annotations (@Component, @Inject, etc.) → stored as decorators

Parent linkage:
  parent_id in Symbol is set to the LOCAL INDEX in the symbols list during
  parsing. The Indexer remaps this to the actual DB id after insertion.

Extension functions:
  `fun String.extended(): Int` → name="extended", qualified_name="String.extended",
  kind=FUNCTION, is_public based on visibility modifier.
"""

from __future__ import annotations

import re

import tree_sitter_languages
from tree_sitter import Node, Parser

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge
from trelix.indexing.parser.base import BaseParser, ParseResult


class KotlinParser(BaseParser):
    """Tree-sitter based Kotlin parser using direct AST traversal."""

    MAX_CLASS_FIELDS: int = 30

    def __init__(self) -> None:
        self._ts_lang = tree_sitter_languages.get_language("kotlin")
        self._parser = Parser()
        self._parser.set_language(self._ts_lang)

    @property
    def language_name(self) -> str:
        return "kotlin"

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

        self._walk_top_level(
            root, source_bytes, file_id, symbols, raw_calls, import_edges, type_edges
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
    # Top-level walk
    # ------------------------------------------------------------------

    def _walk_top_level(
        self,
        root: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
    ) -> None:
        for child in root.children:
            ntype = child.type
            if ntype == "import_list":
                for imp in child.children:
                    if imp.type == "import_header":
                        import_edges.extend(self._extract_import(imp, src, file_id))
            elif ntype == "class_declaration":
                self._handle_class(
                    child, src, file_id, symbols, raw_calls, type_edges, parent_class_local_idx=None
                )
            elif ntype == "object_declaration":
                self._handle_object(
                    child, src, file_id, symbols, raw_calls, type_edges, parent_class_local_idx=None
                )
            elif ntype == "function_declaration":
                self._handle_function(
                    child, src, file_id, symbols, raw_calls, parent_class_local_idx=None
                )
            elif ntype == "property_declaration":
                self._handle_top_property(child, src, file_id, symbols)
            elif ntype == "type_alias":
                self._handle_type_alias(child, src, file_id, symbols)

    # ------------------------------------------------------------------
    # Class / interface / enum
    # ------------------------------------------------------------------

    def _handle_class(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        type_edges: list[TypeEdge],
        parent_class_local_idx: int | None,
    ) -> None:
        # Name is always a type_identifier child
        name_node = self._field(node, "name")
        if not name_node:
            return
        name = self._txt(name_node, src)
        if not name:
            return

        # Determine symbol kind from the 'kind' field (keyword token: 'interface', 'enum', 'class')
        kind_node = self._field(node, "kind")
        kind_text = kind_node.type if kind_node else "class"

        if kind_text == "interface":
            sym_kind = SymbolKind.INTERFACE
        elif kind_text == "enum":
            sym_kind = SymbolKind.ENUM
        else:
            sym_kind = SymbolKind.CLASS

        # Decorators from annotations in modifiers
        decorators = self._extract_annotations(node, src)

        # Visibility — default in Kotlin is public
        is_public = self._is_public(node, src)

        # Qualified name (nested class gets parent prefix)
        if parent_class_local_idx is not None:
            parent_name = symbols[parent_class_local_idx].name  # type: ignore[index]
            qualified_name = f"{parent_name}.{name}"
        else:
            qualified_name = name

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=qualified_name,
            kind=sym_kind,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=self._class_signature(node, src),
            body=self._txt(node, src),
            decorators=decorators,
            is_public=is_public,
            parent_id=parent_class_local_idx,
        )
        class_local_idx = len(symbols)
        symbols.append(sym)

        # Type edges from delegation specifiers (: Base(), Interface)
        self._extract_type_edges(node, src, class_local_idx, type_edges)

        # Walk body
        body_node = self._field(node, "body")
        if body_node:
            if body_node.type == "enum_class_body":
                self._walk_enum_body(
                    body_node, src, file_id, symbols, raw_calls, type_edges, class_local_idx
                )
            else:
                self._walk_class_body(
                    body_node, src, file_id, symbols, raw_calls, type_edges, class_local_idx
                )

    def _walk_class_body(
        self,
        body: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        type_edges: list[TypeEdge],
        class_local_idx: int,
    ) -> None:
        field_count = 0
        for child in body.children:
            ntype = child.type
            if ntype == "function_declaration":
                self._handle_function(
                    child, src, file_id, symbols, raw_calls, parent_class_local_idx=class_local_idx
                )
            elif ntype == "property_declaration":
                if field_count < self.MAX_CLASS_FIELDS:
                    self._handle_class_property(child, src, file_id, symbols, class_local_idx)
                    field_count += 1
            elif ntype == "companion_object":
                # Companion object members are attributed to the parent class
                comp_body = self._field(child, "body")
                if comp_body:
                    self._walk_class_body(
                        comp_body, src, file_id, symbols, raw_calls, type_edges, class_local_idx
                    )
            elif ntype == "class_declaration":
                self._handle_class(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    type_edges,
                    parent_class_local_idx=class_local_idx,
                )
            elif ntype == "object_declaration":
                self._handle_object(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    type_edges,
                    parent_class_local_idx=class_local_idx,
                )

    def _walk_enum_body(
        self,
        body: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        type_edges: list[TypeEdge],
        class_local_idx: int,
    ) -> None:
        parent_name = symbols[class_local_idx].name  # type: ignore[index]
        for child in body.children:
            if child.type == "enum_entry":
                name_node = child.children[0] if child.children else None
                # enum_entry: first named child is simple_identifier
                for c in child.children:
                    if c.is_named and c.type == "simple_identifier":
                        name_node = c
                        break
                if name_node:
                    entry_name = self._txt(name_node, src)
                    symbols.append(
                        Symbol(
                            file_id=file_id,
                            name=entry_name,
                            qualified_name=f"{parent_name}.{entry_name}",
                            kind=SymbolKind.CONSTANT,
                            line_start=child.start_point[0] + 1,
                            line_end=child.end_point[0] + 1,
                            signature=f"{parent_name}.{entry_name}",
                            body=self._txt(child, src),
                            parent_id=class_local_idx,
                            is_public=True,
                        )
                    )
            elif child.type == "function_declaration":
                self._handle_function(
                    child, src, file_id, symbols, raw_calls, parent_class_local_idx=class_local_idx
                )

    # ------------------------------------------------------------------
    # Object declarations (singletons, companion objects)
    # ------------------------------------------------------------------

    def _handle_object(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        type_edges: list[TypeEdge],
        parent_class_local_idx: int | None,
    ) -> None:
        name_node = self._field(node, "name")
        if not name_node:
            return
        name = self._txt(name_node, src)

        decorators = self._extract_annotations(node, src)
        is_public = self._is_public(node, src)

        if parent_class_local_idx is not None:
            parent_name = symbols[parent_class_local_idx].name  # type: ignore[index]
            qualified_name = f"{parent_name}.{name}"
        else:
            qualified_name = name

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=qualified_name,
            kind=SymbolKind.CLASS,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=f"object {name}",
            body=self._txt(node, src),
            decorators=decorators,
            is_public=is_public,
            parent_id=parent_class_local_idx,
        )
        obj_local_idx = len(symbols)
        symbols.append(sym)

        self._extract_type_edges(node, src, obj_local_idx, type_edges)

        body_node = self._field(node, "body")
        if body_node:
            self._walk_class_body(
                body_node, src, file_id, symbols, raw_calls, type_edges, obj_local_idx
            )

    # ------------------------------------------------------------------
    # Functions / methods
    # ------------------------------------------------------------------

    def _handle_function(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        parent_class_local_idx: int | None,
    ) -> None:
        name_node = self._field(node, "name")
        if not name_node:
            return
        name = self._txt(name_node, src)
        if not name:
            return

        # Extension function receiver: `fun String.upper()` → receiver = user_type before name
        receiver_node = self._field(node, "receiver")
        receiver_name = self._extract_type_name(receiver_node, src) if receiver_node else None

        is_method = parent_class_local_idx is not None
        if is_method:
            parent_name = symbols[parent_class_local_idx].name  # type: ignore[index]
            qualified_name = f"{parent_name}.{name}"
            sym_kind = SymbolKind.METHOD
        elif receiver_name:
            qualified_name = f"{receiver_name}.{name}"
            sym_kind = SymbolKind.FUNCTION
        else:
            qualified_name = name
            sym_kind = SymbolKind.FUNCTION

        decorators = self._extract_annotations(node, src)
        is_public = self._is_public(node, src)

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=qualified_name,
            kind=sym_kind,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=self._func_signature(node, src),
            body=self._txt(node, src),
            decorators=decorators,
            is_public=is_public,
            parent_id=parent_class_local_idx,
        )
        func_local_idx = len(symbols)
        symbols.append(sym)

        # Walk function body for call edges
        body_node = self._field(node, "body")
        if body_node:
            self._walk_body_for_calls(body_node, src, raw_calls, func_local_idx)

    # ------------------------------------------------------------------
    # Properties / constants
    # ------------------------------------------------------------------

    def _handle_top_property(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
    ) -> None:
        """Top-level property: extract if `const val` or ALL_CAPS name."""
        var_node = self._field(node, "variable")
        if not var_node:
            return
        name_node = self._get_child_by_type(var_node, "simple_identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        is_const = self._has_const_modifier(node, src)
        if not (is_const or self._is_constant_name(name)):
            return

        body = self._txt(node, src)
        if len(body) > 500:
            body = body[:500] + "..."

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
                is_public=self._is_public(node, src),
            )
        )

    def _handle_class_property(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        class_local_idx: int,
    ) -> None:
        """Class-level property: const → CONSTANT, typed field → VARIABLE."""
        var_node = self._field(node, "variable")
        if not var_node:
            return
        name_node = self._get_child_by_type(var_node, "simple_identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        parent_name = symbols[class_local_idx].name  # type: ignore[index]
        body = self._txt(node, src)
        if len(body) > 500:
            body = body[:500] + "..."

        is_const = self._has_const_modifier(node, src)

        if is_const or self._is_constant_name(name):
            sym_kind = SymbolKind.CONSTANT
        else:
            # Only emit typed fields (has type annotation on variable_declaration)
            has_type = (
                self._get_child_by_type(var_node, "user_type") is not None
                or self._get_child_by_type(var_node, "nullable_type") is not None
            )
            if not has_type:
                return
            sym_kind = SymbolKind.VARIABLE

        # Skip private single-underscore fields
        if name.startswith("_") and not name.startswith("__"):
            return

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=f"{parent_name}.{name}",
                kind=sym_kind,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=body.split("\n")[0][:200],
                body=body,
                parent_id=class_local_idx,
                is_public=self._is_public(node, src),
            )
        )

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
        name_node = self._field(node, "name")
        if not name_node:
            return
        name = self._txt(name_node, src)
        body = self._txt(node, src)
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
                is_public=self._is_public(node, src),
            )
        )

    # ------------------------------------------------------------------
    # Call edge extraction (walk function body)
    # ------------------------------------------------------------------

    def _walk_body_for_calls(
        self,
        node: Node,
        src: bytes,
        raw_calls: list[tuple[int | None, str, int]],
        func_local_idx: int,
        depth: int = 0,
    ) -> None:
        if depth > 15:
            return
        for child in node.children:
            if not child.is_named:
                continue
            if child.type == "call_expression":
                self._extract_call(child, src, raw_calls, func_local_idx)
                # Recurse into call arguments for nested calls
                self._walk_body_for_calls(child, src, raw_calls, func_local_idx, depth + 1)
            elif child.type not in (
                "function_declaration",
                "class_declaration",
                "object_declaration",
                "lambda_literal",  # separate scope — skip
            ):
                self._walk_body_for_calls(child, src, raw_calls, func_local_idx, depth + 1)

    def _extract_call(
        self,
        node: Node,
        src: bytes,
        raw_calls: list[tuple[int | None, str, int]],
        func_local_idx: int,
    ) -> None:
        """Extract callee name from a call_expression node."""
        if not node.children:
            return
        func_node = node.children[0] if node.children[0].is_named else None
        if not func_node:
            return

        if func_node.type == "simple_identifier":
            name = self._txt(func_node, src)
        elif func_node.type == "navigation_expression":
            # obj.method() — last navigation_suffix simple_identifier is the method
            name = ""
            for c in func_node.children:
                if c.type == "navigation_suffix":
                    for sc in c.children:
                        if sc.is_named and sc.type == "simple_identifier":
                            name = self._txt(sc, src)
        else:
            return

        if name:
            raw_calls.append((func_local_idx, name, node.start_point[0] + 1))

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_import(self, node: Node, src: bytes, file_id: int) -> list[ImportEdge]:
        """
        import com.example.util.Logger  → imported_from="com.example.util.Logger"
        import kotlin.io.*              → imported_from="kotlin.io", imported_names=["*"]
        """
        ident_node = self._get_child_by_type(node, "identifier")
        if not ident_node:
            return []

        parts = [
            self._txt(c, src)
            for c in ident_node.children
            if c.is_named and c.type == "simple_identifier"
        ]
        if not parts:
            return []

        module_path = ".".join(parts)

        # Check for wildcard: last raw child of import_header is "*"
        imported_names: list[str] = []
        for c in node.children:
            if not c.is_named and src[c.start_byte : c.end_byte] == b"*":
                imported_names = ["*"]
                break

        if not imported_names and parts:
            imported_names = [parts[-1]]

        return [
            ImportEdge(
                file_id=file_id,
                imported_from=module_path,
                imported_names=imported_names,
            )
        ]

    # ------------------------------------------------------------------
    # Type edge extraction
    # ------------------------------------------------------------------

    def _extract_type_edges(
        self,
        node: Node,
        src: bytes,
        from_idx: int,
        type_edges: list[TypeEdge],
    ) -> None:
        """Extract TypeEdges from delegation_specifier children."""
        for child in node.children:
            if child.type == "delegation_specifier":
                # delegation_specifier > user_type > type_identifier
                # delegation_specifier > constructor_invocation > user_type > type_identifier
                type_name = self._extract_delegation_type(child, src)
                if type_name:
                    type_edges.append(
                        TypeEdge(
                            from_symbol_id=from_idx,
                            to_type_name=type_name,
                            edge_kind="extends",
                        )
                    )

    def _extract_delegation_type(self, node: Node, src: bytes) -> str | None:
        """Get the type name from a delegation_specifier node."""
        for child in node.children:
            if child.type == "user_type":
                return self._extract_type_name(child, src)
            elif child.type == "constructor_invocation":
                ut = self._get_child_by_type(child, "user_type")
                if ut:
                    return self._extract_type_name(ut, src)
        return None

    def _extract_type_name(self, user_type_node: Node, src: bytes) -> str | None:
        """Get the base type name (without generics) from a user_type node."""
        ti = self._get_child_by_type(user_type_node, "type_identifier")
        return self._txt(ti, src) if ti else None

    # ------------------------------------------------------------------
    # Signature builders
    # ------------------------------------------------------------------

    def _class_signature(self, node: Node, src: bytes) -> str:
        """Build `[modifiers] kind Name[(params)] [: Supertypes]`."""
        kind_node = self._field(node, "kind")
        kind_text = kind_node.type if kind_node else "class"

        # Class modifier prefix (data, sealed, abstract, open)
        mod_prefix = self._class_mod_prefix(node, src)

        name_node = self._field(node, "name")
        name = self._txt(name_node, src) if name_node else "?"

        # Primary constructor params (condensed)
        ctor_node = self._field(node, "primary_constructor")
        ctor_text = ""
        if ctor_node:
            params_node = (
                self._field(ctor_node, "parameters")
                or self._get_child_by_type(ctor_node, "function_value_parameters")
                or self._get_child_by_type(ctor_node, "class_parameters")
            )
            if params_node:
                ctor_text = self._txt(params_node, src)
                if len(ctor_text) > 80:
                    ctor_text = ctor_text[:77] + "..."

        # Supertypes
        supers: list[str] = []
        for child in node.children:
            if child.type == "delegation_specifier":
                t = self._extract_delegation_type(child, src)
                if t:
                    supers.append(t)
        super_text = f" : {', '.join(supers)}" if supers else ""

        return f"{mod_prefix}{kind_text} {name}{ctor_text}{super_text}"

    def _func_signature(self, node: Node, src: bytes) -> str:
        """Build `fun [Receiver.]name(params)[: ReturnType]`."""
        receiver_node = self._field(node, "receiver")
        name_node = self._field(node, "name")
        params_node = self._field(node, "parameters")
        return_node = self._field(node, "return_type")

        receiver_text = f"{self._extract_type_name(receiver_node, src)}." if receiver_node else ""
        name = self._txt(name_node, src) if name_node else "?"
        params = self._txt(params_node, src) if params_node else "()"
        if len(params) > 100:
            params = params[:97] + "..."
        ret = f": {self._txt(return_node, src)}" if return_node else ""
        return f"fun {receiver_text}{name}{params}{ret}"

    def _class_mod_prefix(self, node: Node, src: bytes) -> str:
        """Extract data/sealed/abstract/open prefix from modifiers."""
        mod_node = self._get_child_by_type(node, "modifiers")
        if not mod_node:
            return ""
        parts: list[str] = []
        for c in mod_node.children:
            if c.type == "class_modifier":
                parts.append(self._txt(c, src))
            elif c.type == "inheritance_modifier":
                parts.append(self._txt(c, src))
        return (" ".join(parts) + " ") if parts else ""

    # ------------------------------------------------------------------
    # Modifier helpers
    # ------------------------------------------------------------------

    def _extract_annotations(self, node: Node, src: bytes) -> list[str]:
        """Extract @Annotation strings from the node's modifiers."""
        mod_node = self._get_child_by_type(node, "modifiers")
        if not mod_node:
            return []
        result: list[str] = []
        for c in mod_node.children:
            if c.type == "annotation":
                text = "@" + self._txt(c, src).lstrip("@")
                if len(text) > 200:
                    text = text[:200] + "..."
                result.append(text)
        return result

    def _is_public(self, node: Node, src: bytes) -> bool:
        """Kotlin default is public unless explicitly private/protected."""
        mod_node = self._get_child_by_type(node, "modifiers")
        if not mod_node:
            return True
        for c in mod_node.children:
            if c.type == "visibility_modifier":
                v = self._txt(c, src)
                if v in ("private", "protected"):
                    return False
        return True

    def _has_const_modifier(self, node: Node, src: bytes) -> bool:
        """True if the property has a `const` modifier."""
        mod_node = self._get_child_by_type(node, "modifiers")
        if not mod_node:
            return False
        for c in mod_node.children:
            if c.type == "property_modifier" and self._txt(c, src) == "const":
                return True
        return False

    @staticmethod
    def _is_constant_name(name: str) -> bool:
        """True for ALL_CAPS names (Kotlin constant naming convention)."""
        return bool(re.match(r"^[A-Z][A-Z0-9_]*$", name))

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

    def _field(self, node: Node, field_name: str) -> Node | None:
        """Return child node at a named field position."""
        return node.child_by_field_name(field_name)

    def _get_child_by_type(self, node: Node, type_name: str) -> Node | None:
        for child in node.children:
            if child.type == type_name:
                return child
        return None
