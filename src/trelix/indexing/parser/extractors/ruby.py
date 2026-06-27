"""
Ruby parser: direct AST traversal using Tree-sitter.

Design: tree-sitter parses source into an AST; all extraction logic is ours.
We walk the AST directly for full control over parent-child linkage and to
handle Ruby's dynamic visibility modifiers (private/protected/public).

Extracts:
  - Module definitions (kind=MODULE) — "module Foo"
  - Class definitions (kind=CLASS) — "class Foo < Bar"
  - Singleton classes (kind=CLASS) — "class << self"
  - Method definitions (kind=METHOD inside class, FUNCTION at top level)
  - Singleton methods (kind=METHOD) — "def self.method_name"
  - Constants (kind=CONSTANT) — ALL_CAPS assignments, CamelCase top-level
  - require/require_relative → ImportEdge
  - Method calls → CallEdge (caller_id = local idx, remapped by Indexer)
  - Class inheritance → TypeEdge (edge_kind="extends")
  - include/prepend/extend → TypeEdge (edge_kind="implements")

Parent linkage:
  parent_id in Symbol is set to the LOCAL INDEX in the symbols list during
  parsing. The Indexer remaps this to the actual DB id after insertion.

Qualified names:
  - Instance methods: "ClassName#method_name"
  - Singleton/class methods: "ClassName.method_name"
  - Module-qualified: "ModuleName::ClassName" (not reconstructed here; only
    the immediate parent class name is used for qualified names)

Visibility:
  Ruby methods are public by default. After a bare "private" or "protected"
  identifier node in the class body, subsequent methods are non-public.
  After "public", they revert to public. The grammar emits these as plain
  `identifier` nodes inside the class body_statement.
"""

from __future__ import annotations

from tree_sitter import Node

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge
from trelix.indexing.parser._grammar import load_language, make_parser
from trelix.indexing.parser.base import BaseParser, ParseResult


class RubyParser(BaseParser):
    """Tree-sitter based Ruby parser using direct AST traversal."""

    # tree-sitter grammar name
    _GRAMMAR = "ruby"

    # Visibility modifiers that appear as bare identifiers in the class body
    _PRIVATE_MODIFIERS: frozenset[str] = frozenset({"private", "protected"})
    _PUBLIC_MODIFIERS: frozenset[str] = frozenset({"public"})

    # require/require_relative calls that produce ImportEdge
    _REQUIRE_METHODS: frozenset[str] = frozenset({"require", "require_relative"})

    # include/prepend/extend calls that produce TypeEdge(implements)
    _MIXIN_METHODS: frozenset[str] = frozenset({"include", "prepend", "extend"})

    def __init__(self) -> None:
        self._ts_lang = load_language(self._GRAMMAR)
        self._parser = make_parser(self._GRAMMAR)

    @property
    def language_name(self) -> str:
        return "ruby"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def parse(self, source: str, file_id: int) -> ParseResult:
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        symbols: list[Symbol] = []
        # (caller_local_idx | None, callee_name, line)
        raw_calls: list[tuple[int | None, str, int]] = []
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
            parent_module_name=None,
            current_func_local_idx=None,
            current_visibility_is_public=True,
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
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        parent_class_local_idx: int | None,
        parent_module_name: str | None,
        current_func_local_idx: int | None,
        current_visibility_is_public: bool,
        depth: int,
    ) -> None:
        """Recursive depth-first walk. Returns updated visibility state."""
        if depth > 25:
            return

        visibility_public = current_visibility_is_public

        for child in node.children:
            ntype = child.type

            # ---- Module definition ------------------------------------------
            if ntype == "module":
                self._handle_module(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_module_name,
                    depth,
                )

            # ---- Class definition -------------------------------------------
            elif ntype == "class":
                self._handle_class(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_class_local_idx,
                    parent_module_name,
                    depth,
                )

            # ---- Singleton class: class << self --------------------------------
            elif ntype == "singleton_class":
                self._handle_singleton_class(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_class_local_idx,
                    parent_module_name,
                    depth,
                )

            # ---- Instance method definition -----------------------------------
            elif ntype == "method":
                self._handle_method(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_class_local_idx,
                    parent_module_name,
                    current_func_local_idx,
                    is_singleton=False,
                    is_public=visibility_public,
                    depth=depth,
                )

            # ---- Singleton method: def self.foo --------------------------------
            elif ntype == "singleton_method":
                self._handle_method(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_class_local_idx,
                    parent_module_name,
                    current_func_local_idx,
                    is_singleton=True,
                    is_public=True,  # singleton methods are always public
                    depth=depth,
                )

            # ---- Top-level constant assignment --------------------------------
            # "assignment" with a "constant" left-hand side at top level or
            # inside a class/module body.
            elif ntype == "assignment":
                self._handle_assignment(
                    child,
                    src,
                    file_id,
                    symbols,
                    parent_class_local_idx,
                    parent_module_name,
                    current_func_local_idx,
                )

            # ---- Call node: require / include / extend / regular calls ---------
            elif ntype == "call":
                self._handle_call(
                    child,
                    src,
                    file_id,
                    import_edges,
                    type_edges,
                    raw_calls,
                    parent_class_local_idx,
                    current_func_local_idx,
                )
                # Recurse into call arguments (nested calls)
                self._walk(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_class_local_idx,
                    parent_module_name,
                    current_func_local_idx,
                    visibility_public,
                    depth + 1,
                )

            # ---- Visibility modifiers: bare "private" / "protected" / "public" --
            # These appear as plain identifier children inside body_statement.
            # We update the running visibility flag for subsequent siblings.
            elif ntype == "identifier" and parent_class_local_idx is not None:
                word = self._txt(child, src)
                if word in self._PRIVATE_MODIFIERS:
                    visibility_public = False
                elif word in self._PUBLIC_MODIFIERS:
                    visibility_public = True

            # ---- Recurse into container nodes ---------------------------------
            elif ntype in (
                "program",
                "body_statement",
                "if",
                "unless",
                "while",
                "until",
                "for",
                "begin",
                "rescue",
                "ensure",
                "do_block",
                "block",
                "lambda",
                "then",
                "else",
                "elsif",
                "case",
                "when",
                "in",
                "return",
                "yield",
                "parenthesized_statements",
                "binary",
                "unary",
                "array",
                "hash",
                "pair",
                "argument_list",
                "method_parameters",
                "assignment",
                "operator_assignment",
                "conditional",
                "string_interpolation",
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
                    parent_module_name,
                    current_func_local_idx,
                    visibility_public,
                    depth + 1,
                )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_module(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        parent_module_name: str | None,
        depth: int,
    ) -> None:
        """Handle `module Foo ... end`."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = self._txt(name_node, src)

        qualified = f"{parent_module_name}::{name}" if parent_module_name else name

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=qualified,
            kind=SymbolKind.MODULE,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=f"module {qualified}",
            body=self._txt(node, src),
            is_public=True,
        )
        module_local_idx = len(symbols)
        symbols.append(sym)

        body_node = node.child_by_field_name("body")
        if body_node:
            self._walk(
                body_node,
                src,
                file_id,
                symbols,
                raw_calls,
                import_edges,
                type_edges,
                parent_class_local_idx=module_local_idx,
                parent_module_name=qualified,
                current_func_local_idx=None,
                current_visibility_is_public=True,
                depth=depth + 1,
            )

    def _handle_class(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        parent_class_local_idx: int | None,
        parent_module_name: str | None,
        depth: int,
    ) -> None:
        """Handle `class Foo < Bar ... end`."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = self._txt(name_node, src)

        qualified = f"{parent_module_name}::{name}" if parent_module_name else name

        # Extract superclass
        superclass_node = node.child_by_field_name("superclass")
        superclass_name: str | None = None
        if superclass_node:
            # superclass node: `< ConstantName`
            for sc in superclass_node.children:
                if sc.type in ("constant", "scope_resolution"):
                    superclass_name = self._txt(sc, src)
                    break

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=qualified,
            kind=SymbolKind.CLASS,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=self._class_signature(node, src, qualified, superclass_name),
            body=self._txt(node, src),
            is_public=True,
        )
        class_local_idx = len(symbols)
        symbols.append(sym)

        if superclass_name:
            type_edges.append(
                TypeEdge(
                    from_symbol_id=class_local_idx,
                    to_type_name=superclass_name,
                    edge_kind="extends",
                )
            )

        body_node = node.child_by_field_name("body")
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
                parent_module_name=parent_module_name,
                current_func_local_idx=None,
                current_visibility_is_public=True,
                depth=depth + 1,
            )

    def _handle_singleton_class(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        parent_class_local_idx: int | None,
        parent_module_name: str | None,
        depth: int,
    ) -> None:
        """Handle `class << self ... end`."""
        # Use the parent class name if available, otherwise "<singleton>"
        if parent_class_local_idx is not None and symbols:
            parent_name = symbols[parent_class_local_idx].name
            name = f"{parent_name}::<singleton>"
        else:
            name = "<singleton>"

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=SymbolKind.CLASS,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature="class << self",
            body=self._txt(node, src),
            is_public=True,
        )
        singleton_local_idx = len(symbols)
        symbols.append(sym)

        # Methods inside singleton_class belong to it
        body_node = None
        for child in node.children:
            if child.type == "body_statement":
                body_node = child
                break

        if body_node:
            self._walk(
                body_node,
                src,
                file_id,
                symbols,
                raw_calls,
                import_edges,
                type_edges,
                parent_class_local_idx=singleton_local_idx,
                parent_module_name=parent_module_name,
                current_func_local_idx=None,
                current_visibility_is_public=True,
                depth=depth + 1,
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
        parent_module_name: str | None,
        current_func_local_idx: int | None,
        is_singleton: bool,
        is_public: bool,
        depth: int,
    ) -> None:
        """Handle both `def method` and `def self.method` nodes."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
        name = self._txt(name_node, src)

        is_inside_class = parent_class_local_idx is not None
        kind = SymbolKind.METHOD if is_inside_class else SymbolKind.FUNCTION

        if is_inside_class:
            assert parent_class_local_idx is not None  # guaranteed by is_inside_class check
            parent_sym = symbols[parent_class_local_idx]
            parent_name = parent_sym.name
            # "ClassName.method" for singleton/class methods, "ClassName#method" for instance
            separator = "." if is_singleton else "#"
            qualified_name = f"{parent_name}{separator}{name}"
        else:
            qualified_name = name

        params_node = node.child_by_field_name("parameters")
        sig = self._method_signature(name, is_singleton, params_node, src)

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=qualified_name,
            kind=kind,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=sig,
            body=self._txt(node, src),
            is_public=is_public,
            parent_id=parent_class_local_idx,
        )
        func_local_idx = len(symbols)
        symbols.append(sym)

        body_node = node.child_by_field_name("body")
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
                parent_module_name=parent_module_name,
                current_func_local_idx=func_local_idx,
                current_visibility_is_public=True,
                depth=depth + 1,
            )

    def _handle_assignment(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        parent_class_local_idx: int | None,
        parent_module_name: str | None,
        current_func_local_idx: int | None,
    ) -> None:
        """
        Handle constant assignments: UPPER_CASE = ... or CamelCase = ...

        Only emitted at module/class scope (not inside methods).
        Ruby constant names start with an uppercase letter.
        """
        # Skip assignments inside method bodies
        if current_func_local_idx is not None:
            return

        # Left-hand side must be a constant node (starts with uppercase in Ruby grammar)
        lhs = None
        for child in node.children:
            if child.type == "constant":
                lhs = child
                break
        if not lhs:
            return

        name = self._txt(lhs, src)

        # Qualify name if inside a class/module
        if parent_class_local_idx is not None and symbols:
            parent_name = symbols[parent_class_local_idx].name
            qualified_name = f"{parent_name}::{name}"
        else:
            qualified_name = f"{parent_module_name}::{name}" if parent_module_name else name

        body = self._txt(node, src)
        if len(body) > 500:
            body = body[:500] + "..."

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=qualified_name,
                kind=SymbolKind.CONSTANT,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=body.split("\n")[0][:200],
                body=body,
                is_public=True,
                parent_id=parent_class_local_idx,
            )
        )

    def _handle_call(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        raw_calls: list[tuple[int | None, str, int]],
        parent_class_local_idx: int | None,
        current_func_local_idx: int | None,
    ) -> None:
        """
        Handle call nodes:
          - require/require_relative → ImportEdge
          - include/prepend/extend → TypeEdge (implements)
          - everything else → CallEdge
        """
        # Get the method name
        method_node = node.child_by_field_name("method")
        if method_node is None:
            # Bare call without receiver: `bar()` — method is first child
            for child in node.children:
                if child.type == "identifier":
                    method_node = child
                    break
        if method_node is None:
            return

        method_name = self._txt(method_node, src)
        line = node.start_point[0] + 1

        # --- require / require_relative → ImportEdge -------------------------
        if method_name in self._REQUIRE_METHODS:
            args_node = node.child_by_field_name("arguments")
            if args_node:
                for arg in args_node.children:
                    if arg.type == "string":
                        path = self._extract_string_content(arg, src)
                        if path:
                            import_edges.append(
                                ImportEdge(
                                    file_id=file_id,
                                    imported_from=path,
                                    imported_names=[],
                                )
                            )
            return

        # --- include / prepend / extend → TypeEdge(implements) ---------------
        if method_name in self._MIXIN_METHODS and parent_class_local_idx is not None:
            args_node = node.child_by_field_name("arguments")
            if args_node:
                for arg in args_node.children:
                    if arg.type in ("constant", "scope_resolution"):
                        mixin_name = self._txt(arg, src)
                        type_edges.append(
                            TypeEdge(
                                from_symbol_id=parent_class_local_idx,
                                to_type_name=mixin_name,
                                edge_kind="implements",
                            )
                        )
            return

        # --- Regular call → CallEdge -----------------------------------------
        raw_calls.append((current_func_local_idx, method_name, line))

    # ------------------------------------------------------------------
    # Signature builders
    # ------------------------------------------------------------------

    def _class_signature(
        self,
        node: Node,
        src: bytes,
        qualified_name: str,
        superclass_name: str | None,
    ) -> str:
        if superclass_name:
            return f"class {qualified_name} < {superclass_name}"
        return f"class {qualified_name}"

    def _method_signature(
        self,
        name: str,
        is_singleton: bool,
        params_node: Node | None,
        src: bytes,
    ) -> str:
        prefix = "def self." if is_singleton else "def "
        params = self._txt(params_node, src) if params_node else ""
        return f"{prefix}{name}{params}"

    # ------------------------------------------------------------------
    # String content extraction
    # ------------------------------------------------------------------

    def _extract_string_content(self, string_node: Node, src: bytes) -> str:
        """Extract the content of a Ruby string literal node."""
        for child in string_node.children:
            if child.type == "string_content":
                return self._txt(child, src)
        # Fallback: strip surrounding quotes
        raw = self._txt(string_node, src)
        for quote in ('"""', "'''", '"', "'"):
            if raw.startswith(quote) and raw.endswith(quote) and len(raw) >= 2 * len(quote):
                return raw[len(quote) : -len(quote)]
        return raw.strip("'\"")

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
        return node.child_by_field_name(field_name)
