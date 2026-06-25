"""Kotlin parser: direct AST traversal using Tree-sitter 0.25+.

Targets tree-sitter-kotlin >=1.1 (individual package, tree-sitter 0.25 API).

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
  - Call sites inside functions → CallEdge (caller_id = local idx)
  - Supertype delegation specifiers → TypeEdge (extends)
  - Kotlin annotations (@Component, @Inject, etc.) → stored as decorators

Grammar notes (tree-sitter-kotlin 1.1 / tree-sitter 0.25):
  - Imports: `import` node with `qualified_identifier` child (not import_list)
  - Class body: `class_body` child (no `body` field in child_by_field_name)
  - Class kind: keyword child type is 'class'/'interface'/'enum' (no `kind` field)
  - Function params: `function_value_parameters` child node
  - Delegation specifiers: `delegation_specifiers` → `delegation_specifier` children
  - Annotations: inside `modifiers` as `annotation` children
  - Companion object body: `class_body` child

Parent linkage:
  parent_id in Symbol is set to the LOCAL INDEX in the symbols list during
  parsing. The Indexer remaps this to the actual DB id after insertion.
"""

from __future__ import annotations

import re
from typing import Optional

import tree_sitter_kotlin as _ts_kotlin
from tree_sitter import Language, Node, Parser

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge
from trelix.indexing.parser.base import BaseParser, ParseResult


class KotlinParser(BaseParser):
    """Tree-sitter based Kotlin parser (tree-sitter-kotlin 1.1 / ts 0.25)."""

    MAX_CLASS_FIELDS: int = 30

    def __init__(self) -> None:
        self._parser = Parser(Language(_ts_kotlin.language()))

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
        raw_calls: list[tuple[Optional[int], str, int]] = []
        import_edges: list[ImportEdge] = []
        type_edges: list[TypeEdge] = []

        self._walk_top_level(root, source_bytes, file_id, symbols, raw_calls, import_edges, type_edges)

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
        raw_calls: list[tuple[Optional[int], str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
    ) -> None:
        for child in root.children:
            ntype = child.type
            if ntype == "import":
                import_edges.extend(self._extract_import(child, src, file_id))
            elif ntype == "class_declaration":
                self._handle_class(child, src, file_id, symbols, raw_calls, type_edges,
                                   parent_class_local_idx=None)
            elif ntype == "object_declaration":
                self._handle_object(child, src, file_id, symbols, raw_calls, type_edges,
                                    parent_class_local_idx=None)
            elif ntype == "function_declaration":
                self._handle_function(child, src, file_id, symbols, raw_calls,
                                      parent_class_local_idx=None)
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
        raw_calls: list[tuple[Optional[int], str, int]],
        type_edges: list[TypeEdge],
        parent_class_local_idx: Optional[int],
    ) -> None:
        # Name node is the `identifier` child
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)
        if not name:
            return

        # Determine kind from keyword children: 'class', 'interface', 'fun' (fun interface)
        # enum is indicated by modifiers containing 'enum' class_modifier
        sym_kind = SymbolKind.CLASS
        for c in node.children:
            if c.type == "interface":
                sym_kind = SymbolKind.INTERFACE
                break
        # Check for enum modifier
        mod_node = self._get_child_by_type(node, "modifiers")
        if mod_node:
            for c in mod_node.children:
                if c.type == "class_modifier" and self._txt(c, src) == "enum":
                    sym_kind = SymbolKind.ENUM
                    break

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

        # Type edges from delegation_specifiers
        self._extract_type_edges(node, src, class_local_idx, type_edges)

        # Walk body
        body_node = self._get_child_by_type(node, "enum_class_body")
        if body_node:
            self._walk_enum_body(body_node, src, file_id, symbols, raw_calls, type_edges,
                                 class_local_idx)
        else:
            body_node = self._get_child_by_type(node, "class_body")
            if body_node:
                self._walk_class_body(body_node, src, file_id, symbols, raw_calls, type_edges,
                                      class_local_idx)

    def _walk_class_body(
        self,
        body: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[Optional[int], str, int]],
        type_edges: list[TypeEdge],
        class_local_idx: int,
    ) -> None:
        field_count = 0
        for child in body.children:
            ntype = child.type
            if ntype == "function_declaration":
                self._handle_function(child, src, file_id, symbols, raw_calls,
                                      parent_class_local_idx=class_local_idx)
            elif ntype == "property_declaration":
                if field_count < self.MAX_CLASS_FIELDS:
                    self._handle_class_property(child, src, file_id, symbols, class_local_idx)
                    field_count += 1
            elif ntype == "companion_object":
                # Companion object body is a class_body child
                comp_body = self._get_child_by_type(child, "class_body")
                if comp_body:
                    self._walk_class_body(comp_body, src, file_id, symbols, raw_calls,
                                          type_edges, class_local_idx)
            elif ntype == "class_declaration":
                self._handle_class(child, src, file_id, symbols, raw_calls, type_edges,
                                   parent_class_local_idx=class_local_idx)
            elif ntype == "object_declaration":
                self._handle_object(child, src, file_id, symbols, raw_calls, type_edges,
                                    parent_class_local_idx=class_local_idx)

    def _walk_enum_body(
        self,
        body: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[Optional[int], str, int]],
        type_edges: list[TypeEdge],
        class_local_idx: int,
    ) -> None:
        parent_name = symbols[class_local_idx].name  # type: ignore[index]
        for child in body.children:
            if child.type == "enum_entry":
                # First named identifier child is the entry name
                name_node = self._get_child_by_type(child, "identifier")
                if name_node:
                    entry_name = self._txt(name_node, src)
                    symbols.append(Symbol(
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
                    ))
            elif child.type == "function_declaration":
                self._handle_function(child, src, file_id, symbols, raw_calls,
                                      parent_class_local_idx=class_local_idx)

    # ------------------------------------------------------------------
    # Object declarations (singletons, companion objects)
    # ------------------------------------------------------------------

    def _handle_object(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[Optional[int], str, int]],
        type_edges: list[TypeEdge],
        parent_class_local_idx: Optional[int],
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
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

        body_node = self._get_child_by_type(node, "class_body")
        if body_node:
            self._walk_class_body(body_node, src, file_id, symbols, raw_calls, type_edges,
                                  obj_local_idx)

    # ------------------------------------------------------------------
    # Functions / methods
    # ------------------------------------------------------------------

    def _handle_function(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[Optional[int], str, int]],
        parent_class_local_idx: Optional[int],
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)
        if not name:
            return

        # Extension function: `fun String.reverse()` — look for a type before the identifier
        # In the grammar, receiver type appears before the identifier when field named 'receiver'
        receiver_node = node.child_by_field_name("receiver")
        receiver_name: Optional[str] = None
        if receiver_node:
            id_node = self._get_child_by_type(receiver_node, "identifier")
            if id_node:
                receiver_name = self._txt(id_node, src)
            else:
                # Try user_type
                ut = self._get_child_by_type(receiver_node, "user_type")
                if ut:
                    id2 = self._get_child_by_type(ut, "identifier")
                    if id2:
                        receiver_name = self._txt(id2, src)

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

        # Walk body for call edges
        body_node = self._get_child_by_type(node, "function_body")
        if body_node:
            self._walk_body_for_calls(body_node, src, raw_calls, func_local_idx)
        # Also check block directly
        block_node = self._get_child_by_type(node, "block")
        if block_node:
            self._walk_body_for_calls(block_node, src, raw_calls, func_local_idx)

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
        var_decl = self._get_child_by_type(node, "variable_declaration")
        if not var_decl:
            return
        name_node = self._get_child_by_type(var_decl, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)

        is_const = self._has_const_modifier(node, src)
        if not (is_const or self._is_constant_name(name)):
            return

        body = self._txt(node, src)
        if len(body) > 500:
            body = body[:500] + "..."

        symbols.append(Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=SymbolKind.CONSTANT,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=body.split("\n")[0][:200],
            body=body,
            is_public=self._is_public(node, src),
        ))

    def _handle_class_property(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        class_local_idx: int,
    ) -> None:
        """Class-level property: const → CONSTANT, typed field → VARIABLE."""
        var_decl = self._get_child_by_type(node, "variable_declaration")
        if not var_decl:
            return
        name_node = self._get_child_by_type(var_decl, "identifier")
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
            # Only emit typed fields
            has_type = (
                self._get_child_by_type(var_decl, "user_type") is not None
                or self._get_child_by_type(var_decl, "nullable_type") is not None
            )
            if not has_type:
                return
            sym_kind = SymbolKind.VARIABLE

        if name.startswith("_") and not name.startswith("__"):
            return

        symbols.append(Symbol(
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
        ))

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
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)
        body = self._txt(node, src)
        symbols.append(Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=SymbolKind.INTERFACE,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=body[:200],
            body=body,
            is_public=self._is_public(node, src),
        ))

    # ------------------------------------------------------------------
    # Call edge extraction (walk function body)
    # ------------------------------------------------------------------

    def _walk_body_for_calls(
        self,
        node: Node,
        src: bytes,
        raw_calls: list[tuple[Optional[int], str, int]],
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
                self._walk_body_for_calls(child, src, raw_calls, func_local_idx, depth + 1)
            elif child.type not in (
                "function_declaration", "class_declaration", "object_declaration",
                "lambda_literal",
            ):
                self._walk_body_for_calls(child, src, raw_calls, func_local_idx, depth + 1)

    def _extract_call(
        self,
        node: Node,
        src: bytes,
        raw_calls: list[tuple[Optional[int], str, int]],
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
        elif func_node.type == "identifier":
            name = self._txt(func_node, src)
        elif func_node.type == "navigation_expression":
            name = ""
            for c in func_node.children:
                if c.type == "navigation_suffix":
                    for sc in c.children:
                        if sc.is_named and sc.type in ("simple_identifier", "identifier"):
                            name = self._txt(sc, src)
        else:
            return

        if name:
            raw_calls.append((func_local_idx, name, node.start_point[0] + 1))

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_import(
        self, node: Node, src: bytes, file_id: int
    ) -> list[ImportEdge]:
        """
        import com.example.util.Logger  → imported_from="com.example.util.Logger"
        import kotlin.io.*              → imported_from="kotlin.io", imported_names=["*"]
        """
        # In tree-sitter-kotlin 1.1, import node has a `qualified_identifier` child
        qi = self._get_child_by_type(node, "qualified_identifier")
        if not qi:
            return []

        # Collect identifier parts (separated by '.' non-named children)
        parts: list[str] = []
        for c in qi.children:
            if c.is_named and c.type == "identifier":
                parts.append(self._txt(c, src))

        if not parts:
            return []

        module_path = ".".join(parts)

        # Check for wildcard: '*' as a non-named child of the import node itself
        imported_names: list[str] = []
        for c in node.children:
            if not c.is_named and self._txt(c, src) == "*":
                imported_names = ["*"]
                break

        if not imported_names and parts:
            imported_names = [parts[-1]]

        return [ImportEdge(
            file_id=file_id,
            imported_from=module_path,
            imported_names=imported_names,
        )]

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
        """Extract TypeEdges from delegation_specifiers children."""
        del_specs = self._get_child_by_type(node, "delegation_specifiers")
        if not del_specs:
            return
        for child in del_specs.children:
            if child.type == "delegation_specifier":
                type_name = self._extract_delegation_type(child, src)
                if type_name:
                    type_edges.append(TypeEdge(
                        from_symbol_id=from_idx,
                        to_type_name=type_name,
                        edge_kind="extends",
                    ))

    def _extract_delegation_type(self, node: Node, src: bytes) -> Optional[str]:
        """Get the type name from a delegation_specifier node."""
        for child in node.children:
            if child.type == "user_type":
                id_node = self._get_child_by_type(child, "identifier")
                return self._txt(id_node, src) if id_node else None
            elif child.type == "constructor_invocation":
                ut = self._get_child_by_type(child, "user_type")
                if ut:
                    id_node = self._get_child_by_type(ut, "identifier")
                    return self._txt(id_node, src) if id_node else None
        return None

    # ------------------------------------------------------------------
    # Signature builders
    # ------------------------------------------------------------------

    def _class_signature(self, node: Node, src: bytes) -> str:
        """Build `[modifiers] class/interface/enum Name[params] [: Supertypes]`."""
        # Determine keyword
        keyword = "class"
        for c in node.children:
            if c.type in ("class", "interface", "fun"):
                keyword = c.type
                break

        mod_prefix = self._class_mod_prefix(node, src)

        name_node = self._get_child_by_type(node, "identifier")
        name = self._txt(name_node, src) if name_node else "?"

        # Primary constructor params
        ctor_text = ""
        params_node = self._get_child_by_type(node, "primary_constructor")
        if params_node:
            fp = self._get_child_by_type(params_node, "class_parameters")
            if fp:
                ctor_text = self._txt(fp, src)
                if len(ctor_text) > 80:
                    ctor_text = ctor_text[:77] + "..."

        # Supertypes
        supers: list[str] = []
        del_specs = self._get_child_by_type(node, "delegation_specifiers")
        if del_specs:
            for child in del_specs.children:
                if child.type == "delegation_specifier":
                    t = self._extract_delegation_type(child, src)
                    if t:
                        supers.append(t)
        super_text = f" : {', '.join(supers)}" if supers else ""

        return f"{mod_prefix}{keyword} {name}{ctor_text}{super_text}"

    def _func_signature(self, node: Node, src: bytes) -> str:
        """Build `fun [Receiver.]name(params)[: ReturnType]`."""
        receiver_node = node.child_by_field_name("receiver")
        name_node = self._get_child_by_type(node, "identifier")
        params_node = self._get_child_by_type(node, "function_value_parameters")
        # Return type: find 'user_type' or 'nullable_type' after the params
        return_type = ""
        found_params = False
        for c in node.children:
            if c == params_node:
                found_params = True
                continue
            if found_params and c.type in ("user_type", "nullable_type", "function_type"):
                return_type = self._txt(c, src)
                break

        receiver_text = ""
        if receiver_node:
            id_node = self._get_child_by_type(receiver_node, "identifier")
            if not id_node:
                ut = self._get_child_by_type(receiver_node, "user_type")
                if ut:
                    id_node = self._get_child_by_type(ut, "identifier")
            if id_node:
                receiver_text = f"{self._txt(id_node, src)}."

        name = self._txt(name_node, src) if name_node else "?"
        params = self._txt(params_node, src) if params_node else "()"
        if len(params) > 100:
            params = params[:97] + "..."
        ret = f": {return_type}" if return_type else ""
        return f"fun {receiver_text}{name}{params}{ret}"

    def _class_mod_prefix(self, node: Node, src: bytes) -> str:
        """Extract data/sealed/abstract/open/enum prefix from modifiers."""
        mod_node = self._get_child_by_type(node, "modifiers")
        if not mod_node:
            return ""
        parts: list[str] = []
        for c in mod_node.children:
            if c.type in ("class_modifier", "inheritance_modifier", "member_modifier"):
                text = self._txt(c, src)
                if text not in parts:
                    parts.append(text)
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
                text = self._txt(c, src)
                if not text.startswith("@"):
                    text = "@" + text
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
        return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _get_child_by_type(self, node: Node, type_name: str) -> Optional[Node]:
        for child in node.children:
            if child.type == type_name:
                return child
        return None
