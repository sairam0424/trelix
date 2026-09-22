"""
Python parser: direct AST traversal using Tree-sitter.

Design: tree-sitter parses source into an AST; all extraction logic is ours.
We walk the AST directly (not .scm queries) for full control over parent-child
linkage and to handle Python's complex syntax (decorators, nested classes, etc.).

NOTE: annotated assignments (x: str = "y") are plain `assignment` nodes with
a 'type' child — same node type as unannotated assignments. There is no
separate 'annotated_assignment' node in this grammar. There is also no
`expression_statement` wrapper — statement-position expressions (assignments,
calls, bare strings/docstrings) appear as direct children of their block.

Extracts:
  - Classes (kind=CLASS)
  - Enum subclasses — class Status(Enum) → kind=ENUM, all members → CONSTANT
  - Protocol subclasses — class Foo(Protocol) → kind=INTERFACE
  - Top-level functions (kind=FUNCTION)
  - Methods inside classes (kind=METHOD, parent_id=class local idx)
  - Module-level constants (kind=CONSTANT) — ALL_CAPS, __dunder__, and typed ALL_CAPS
  - Python 3.12 type aliases — type Vector = list[float] → kind=INTERFACE
  - Class field annotations (kind=VARIABLE, parent_id=class local idx):
      - Pydantic fields:    provider: Literal["sqlite", "postgres"] = "sqlite"
      - Dataclass fields:   name: str  /  age: int = 0
      - TypedDict fields:   name: str
      - General typed vars: timeout: int = 30
  - Class-level ALL_CAPS constants (kind=CONSTANT, parent_id=class local idx)
  - __all__ value parsed into human-readable "Exports: X, Y, Z" body prefix
  - Import statements → ImportEdge
  - Call sites inside functions → CallEdge (caller_id = local idx, remapped by indexer)

Parent linkage:
  parent_id in Symbol is set to the LOCAL INDEX in the symbols list during
  parsing. The Indexer remaps this to the actual DB id after insertion.

  The Indexer resolves these indices against the FINAL symbols list, so the
  list must never be reordered or have anything inserted into the middle of it
  after the walk has recorded an index. See the comment in parse() about the
  reserved symbols[0] slot for the synthetic "<module>" symbol.
"""

from __future__ import annotations

import re

from tree_sitter import Node

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge

from .._grammar import load_language, make_parser
from ..base import BaseParser, ParseResult


class PythonParser(BaseParser):
    """Tree-sitter based Python parser using direct AST traversal."""

    # Cap on annotated field symbols extracted per class.
    # Prevents symbol flood from very large Pydantic/SQLAlchemy models.
    MAX_CLASS_FIELDS: int = 30

    # Base class names that make a class an Enum (all members → CONSTANT)
    _ENUM_BASES: frozenset[str] = frozenset({"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"})
    # Base class names that make a class a Protocol (→ INTERFACE)
    _PROTOCOL_BASES: frozenset[str] = frozenset({"Protocol"})

    # Builtin constructors that would produce a useless/misleading
    # callee_type_hint if captured (e.g. `self.items = list()` should NOT
    # make later `self.items.append(...)` calls look like `list.append`
    # method calls on a user-defined "list" type). Mirrors the same
    # conservative philosophy as _extract_param_types, which skips
    # ambiguous/generic annotations rather than guess.
    _BUILTIN_CONSTRUCTORS: frozenset[str] = frozenset(
        {
            "list",
            "dict",
            "str",
            "int",
            "set",
            "tuple",
            "frozenset",
            "bool",
            "float",
            "bytes",
            "bytearray",
            "object",
        }
    )

    def __init__(self) -> None:
        self._ts_lang = load_language("python")
        self._parser = make_parser("python")

    @property
    def language_name(self) -> str:
        return "python"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def parse(self, source: str, file_id: int) -> ParseResult:
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        symbols: list[Symbol] = []
        # raw calls: (caller_local_idx | None, callee_name, line, callee_type_hint | None)
        raw_calls: list[tuple[int | None, str, int, str | None]] = []
        import_edges: list[ImportEdge] = []
        type_edges: list[TypeEdge] = []

        # Module-level docstring → file-level summary symbol so architectural
        # queries ("how does X work end-to-end", "what is the Y architecture")
        # can find the right file via its module description.
        #
        # The docstring is probed BEFORE the walk so that symbols[0] can be
        # RESERVED for the synthetic "<module>" symbol up front.
        #
        # WHY the slot is reserved instead of inserted afterwards: the walk
        # records LOCAL INDICES into `symbols` as it goes, and three separate
        # families of index point into this list —
        #     Symbol.parent_id        (method/field -> enclosing class)
        #     raw_calls[i][0]         (call site -> enclosing function)
        #     TypeEdge.from_symbol_id (subclass -> its base)
        # Inserting at index 0 *after* the walk shifts every symbol by one and
        # silently invalidates all three at once: a method's parent_id names
        # whatever class was declared before its real parent, a call is
        # attributed to the previous function (fabricating self-recursion), and
        # a class appears to inherit from itself. Reserving the slot first
        # means every index recorded during the walk is already correct for the
        # list's final layout, so no remapping is needed anywhere.
        #
        # Do not turn this back into symbols.insert(0, ...).
        module_doc = self._get_docstring(root, source_bytes)
        if module_doc:
            symbols.append(self._module_symbol(file_id, root, module_doc))

        # Map every function/method NAME in the file to its declared return
        # type annotation, computed ONCE up front (independent of the walk's
        # scoping) so that `x = some_func(); x.method()` can resolve
        # `x`'s type from `some_func`'s `-> ReturnType` no matter where
        # `some_func` is defined relative to the call site.
        func_return_types = self._collect_function_return_types(root, source_bytes)

        # Walk the module-level children to extract top-level constructs
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
            param_types={},
            class_attr_types={},
            func_return_types=func_return_types,
            depth=0,
        )

        # The top-level signature index can only be built once the walk has
        # run, so rebuild the reserved symbol now that its body is knowable.
        # symbols[1:] skips the reserved slot itself; replacing the slot (rather
        # than mutating the Symbol in place) keeps Symbol values immutable.
        if module_doc:
            top_sigs = [
                s.signature
                for s in symbols[1:]
                if s.parent_id is None and s.kind not in (SymbolKind.CONSTANT,)
            ][:20]
            symbols[0] = self._module_symbol(file_id, root, module_doc, top_sigs)

        # Build CallEdge list — caller_id is a local index here, remapped by Indexer
        call_edges: list[CallEdge] = [
            CallEdge(
                caller_id=caller_idx,
                callee_name=name,
                line=line,
                callee_type_hint=type_hint,
            )
            for caller_idx, name, line, type_hint in raw_calls
            if caller_idx is not None
        ]

        return ParseResult(
            symbols=symbols,
            call_edges=call_edges,
            import_edges=import_edges,
            parse_errors=self._count_errors(root),
            type_edges=type_edges,
        )

    def _module_symbol(
        self,
        file_id: int,
        root: Node,
        module_doc: str,
        top_sigs: list[str] | None = None,
    ) -> Symbol:
        """
        Build the synthetic file-level "<module>" symbol.

        Called twice: once to reserve symbols[0] before the walk (without
        `top_sigs`, which are not known yet) and once afterwards to replace
        that slot with the same symbol carrying the top-level signature index.
        """
        body = module_doc
        if top_sigs:
            body += "\n\n# Symbols:\n" + "\n".join(top_sigs)
        return Symbol(
            file_id=file_id,
            name="<module>",
            qualified_name="<module>",
            kind=SymbolKind.MODULE,
            line_start=1,
            line_end=root.end_point[0] + 1,
            signature="module",
            body=body,
            docstring=module_doc,
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
        raw_calls: list[tuple[int | None, str, int, str | None]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        parent_class_local_idx: int | None,
        current_func_local_idx: int | None,
        param_types: dict[str, str],
        class_attr_types: dict[str, str],
        func_return_types: dict[str, str],
        depth: int,
    ) -> None:
        """Recursive depth-first walk. depth guards against absurdly nested code.

        `param_types` is mutated in place as local-variable assignments are
        encountered in source order (see `_handle_local_var_assignment`) —
        everything else threaded through this walk is read-only for the
        duration of the walk. `class_attr_types` is the enclosing class's
        `self.attr` -> type map (built once per class before any of its
        methods are walked; empty at module scope). `func_return_types` is
        the whole-file function/method-name -> declared-return-type map,
        computed once in `parse()`.
        """
        if depth > 20:
            return

        for child in node.children:
            ntype = child.type

            # ---- Class definition ----------------------------------------
            if ntype == "class_definition":
                self._handle_class(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    func_return_types,
                    depth,
                )

            # ---- Function / method definition ----------------------------
            elif ntype == "function_definition":
                self._handle_function(
                    child,
                    src,
                    file_id,
                    symbols,
                    raw_calls,
                    import_edges,
                    type_edges,
                    parent_class_local_idx,
                    current_func_local_idx,
                    class_attr_types,
                    func_return_types,
                    depth,
                )

            # ---- Decorated definition: @dec\ndef foo or @dec\nclass Foo ---
            elif ntype == "decorated_definition":
                decs = self._extract_decorators(child, src)
                inner = self._get_child_by_type(child, "function_definition")
                if inner:
                    self._handle_function(
                        inner,
                        src,
                        file_id,
                        symbols,
                        raw_calls,
                        import_edges,
                        type_edges,
                        parent_class_local_idx,
                        current_func_local_idx,
                        class_attr_types,
                        func_return_types,
                        depth,
                        decorators=decs,
                    )
                else:
                    inner_cls = self._get_child_by_type(child, "class_definition")
                    if inner_cls:
                        self._handle_class(
                            inner_cls,
                            src,
                            file_id,
                            symbols,
                            raw_calls,
                            import_edges,
                            type_edges,
                            func_return_types,
                            depth,
                            decorators=decs,
                        )

            # ---- Python 3.12 type alias: type Vector = list[float] ----------
            elif (
                ntype == "type_alias_statement"
                and parent_class_local_idx is None
                and current_func_local_idx is None
            ):
                self._handle_type_alias(child, src, file_id, symbols)

            # ---- Import statements ----------------------------------------
            elif ntype in ("import_statement", "import_from_statement"):
                import_edges.extend(self._extract_import(child, src, file_id))

            # ---- Module-level constants: ALL_CAPS and __dunder__ names ----
            # Only at module scope (not inside a function or class).
            # Annotated assignments (BASE_URL: str = "x") are ALSO plain
            # `assignment` nodes — the 'type' child distinguishes annotated
            # from plain. _handle_module_assignment handles both.
            elif (
                ntype == "assignment"
                and parent_class_local_idx is None
                and current_func_local_idx is None
            ):
                self._handle_module_assignment(child, src, file_id, symbols)

            # ---- Class-level assignments: field annotations and constants -----
            # Covers Pydantic fields, dataclass fields, TypedDict fields, typed class vars.
            # 'provider: str = "x"' is a plain `assignment` node with a 'type'
            # child — same node type as unannotated assignments.
            elif (
                ntype == "assignment"
                and parent_class_local_idx is not None
                and current_func_local_idx is None
            ):
                # Enforce per-class field cap to avoid symbol flood on large models
                existing_fields = sum(
                    1
                    for s in symbols
                    if s.parent_id == parent_class_local_idx and s.kind == SymbolKind.VARIABLE
                )
                if existing_fields < self.MAX_CLASS_FIELDS:
                    self._handle_class_expression(
                        child, src, file_id, symbols, parent_class_local_idx
                    )

            # ---- Local variable assignment inside a function/method body -----
            # `x = some_func()` — widens callee_type_hint resolution to local
            # variables assigned from a call to a function/method with a
            # declared return type annotation (see _handle_local_var_assignment
            # for the "no retroactive application" / reassignment semantics).
            # Fires for ANY assignment inside a function body (current_func_local_idx
            # is set), which is everything the two branches above do NOT already
            # claim (those both require current_func_local_idx is None).
            elif ntype == "assignment" and current_func_local_idx is not None:
                self._handle_local_var_assignment(child, src, param_types, func_return_types)
                # Still recurse into the assignment for nested calls in the RHS
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
                    param_types,
                    class_attr_types,
                    func_return_types,
                    depth + 1,
                )

            # ---- Call sites (track for call graph) -----------------------
            elif ntype == "call":
                self._handle_call(
                    child, src, raw_calls, current_func_local_idx, param_types, class_attr_types
                )
                # Still recurse into call arguments for nested calls
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
                    param_types,
                    class_attr_types,
                    func_return_types,
                    depth + 1,
                )

            # ---- Recurse into statement / expression wrappers ------------
            # assignment, augmented_assignment, etc. all wrap `call` nodes —
            # we must recurse into them to find call sites (this only fires
            # for assignments NOT already handled by the module-/class-level
            # branches above, e.g. nested inside a function body).
            # Also covers: return foo(), assert foo(), raise Foo(), del foo()
            elif ntype in (
                "block",
                "module",
                "if_statement",
                "for_statement",
                "while_statement",
                "with_statement",
                "try_statement",
                "match_statement",
                "case_clause",
                # if/try/with sub-clauses (contain blocks with calls)
                "elif_clause",
                "else_clause",
                "except_clause",
                "finally_clause",
                "with_clause",
                "with_item",
                # expression containers (calls live inside these)
                # NOTE: "assignment" is deliberately absent here — every
                # (parent_class_local_idx, current_func_local_idx) combination
                # for an `assignment` node is already fully covered by the
                # three dedicated `assignment` branches above (module-const,
                # class-field, local-var), so it can never reach this generic
                # fallback. augmented_assignment (x += 1) is NOT one of those
                # three branches and still needs this generic recursion.
                "augmented_assignment",
                "return_statement",
                "assert_statement",
                "raise_statement",
                "delete_statement",
                "await",
                "yield",
                "yield_from",
                # walrus / paren / subscript / lambda / f-string
                "named_expression",
                "parenthesized_expression",
                "expression_list",
                "subscript",
                "lambda",
                # f-string: recurse into string to reach interpolation children
                "string",
                "interpolation",
                # with X as f / except E as e — call is inside as_pattern
                "as_pattern",
                # boolean / comparison / arithmetic expressions
                "boolean_operator",
                "comparison_operator",
                "not_operator",
                "binary_operator",
                "unary_operator",
                "conditional_expression",
                # containers
                "list",
                "tuple",
                "set",
                "dictionary",
                "list_comprehension",
                "set_comprehension",
                "dictionary_comprehension",
                "generator_expression",
                "argument_list",
                "parameters",
                "keyword_argument",
                "pair",
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
                    param_types,
                    class_attr_types,
                    func_return_types,
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
        raw_calls: list[tuple[int | None, str, int, str | None]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        func_return_types: dict[str, str],
        depth: int,
        decorators: list[str] | None = None,
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)
        body_node = self._get_child_by_type(node, "block")
        docstring = self._get_docstring(body_node, src) if body_node else None

        # Extract base classes for type edges + detect Enum/Protocol subclasses
        bases: list[str] = []
        args_node = node.child_by_field_name("superclasses")
        if args_node:
            for c in args_node.children:
                if c.type == "identifier":
                    bases.append(self._txt(c, src))
                elif c.type == "attribute":
                    attr = c.child_by_field_name("attribute")
                    if attr:
                        bases.append(self._txt(attr, src))

        base_set = set(bases)
        is_enum = bool(base_set & self._ENUM_BASES)
        is_protocol = bool(base_set & self._PROTOCOL_BASES)

        if is_protocol:
            kind = SymbolKind.INTERFACE
        elif is_enum:
            kind = SymbolKind.ENUM
        else:
            kind = SymbolKind.CLASS

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=kind,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=self._class_signature(node, src),
            body=self._txt(node, src),
            docstring=docstring,
            decorators=decorators or [],
            is_public=not name.startswith("_"),
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

        # Walk the class body: extract methods (pass is_enum so all members get extracted)
        if body_node:
            # Two-pass: collect every self.attr assignment ACROSS ALL METHODS of
            # this class (e.g. set in __init__, used in process()) BEFORE walking
            # any method's calls — a call in one method can reference an
            # attribute set in a different method, so the map must exist in
            # full before _handle_call ever consults it.
            self_attr_types = self._collect_self_attr_types(body_node, src, func_return_types)
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
                param_types={},
                class_attr_types=self_attr_types,
                func_return_types=func_return_types,
                depth=depth + 1,
            )

    def _handle_function(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        raw_calls: list[tuple[int | None, str, int, str | None]],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        parent_class_local_idx: int | None,
        current_func_local_idx: int | None,
        class_attr_types: dict[str, str],
        func_return_types: dict[str, str],
        depth: int,
        decorators: list[str] | None = None,
    ) -> None:
        name_node = self._get_child_by_type(node, "identifier")
        if not name_node:
            return
        name = self._txt(name_node, src)
        body_node = self._get_child_by_type(node, "block")
        docstring = self._get_docstring(body_node, src) if body_node else None

        is_method = parent_class_local_idx is not None
        kind = SymbolKind.METHOD if is_method else SymbolKind.FUNCTION

        if is_method:
            assert parent_class_local_idx is not None  # guaranteed by is_method check
            class_name = symbols[parent_class_local_idx].name
            qualified_name = f"{class_name}.{name}"
        else:
            qualified_name = name

        # Dunder methods (__init__, __str__) are public despite underscore convention
        is_dunder = name.startswith("__") and name.endswith("__")
        is_public = is_dunder or not name.startswith("_")

        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=qualified_name,
            kind=kind,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            signature=self._func_signature(node, src),
            body=self._txt(node, src),
            docstring=docstring,
            decorators=decorators or [],
            is_public=is_public,
            parent_id=parent_class_local_idx,  # local idx — remapped by Indexer
        )
        func_local_idx = len(symbols)
        symbols.append(sym)

        # Build param_types: {param_name: type_name} for typed parameters.
        # Used in _handle_call to infer callee_type_hint for method calls.
        # This dict is also MUTATED as the body walk encounters
        # `x = some_func()` local-variable assignments (see
        # _handle_local_var_assignment) — it is fresh per function, so that
        # mutation never leaks into a sibling or enclosing function's scope.
        params_node = node.child_by_field_name("parameters")
        func_param_types: dict[str, str] = (
            self._extract_param_types(params_node, src) if params_node else {}
        )

        # Walk the function body to find nested calls and imports
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
                current_func_local_idx=func_local_idx,
                param_types=func_param_types,
                class_attr_types=class_attr_types,
                func_return_types=func_return_types,
                depth=depth + 1,
            )

    # Class factory calls that produce a new class at module level.
    # These appear as plain assignments but should be CLASS symbols.
    _CLASS_FACTORIES: frozenset[str] = frozenset(
        {
            "namedtuple",
            "NamedTuple",
            "TypedDict",
            "make_dataclass",
        }
    )

    def _handle_module_assignment(
        self,
        assign_node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
    ) -> None:
        """
        Extract module-level ALL_CAPS constants, __dunder__ package metadata,
        and class factory calls (namedtuple, TypedDict, make_dataclass).

        Examples:
          LANGUAGES = {...}         → CONSTANT
          __version__ = "1.2.3"    → CONSTANT
          Point = namedtuple(...)   → CLASS
          Config = TypedDict(...)   → CLASS
        """
        stmt_node = assign_node
        left = assign_node.child_by_field_name("left")
        if not left or left.type != "identifier":
            return  # skip tuple unpacking, subscript assignments, etc.

        name = self._txt(left, src)

        # --- Class factory detection: Point = namedtuple('Point', [...]) ----
        right = assign_node.child_by_field_name("right")
        if right and right.type == "call":
            func_node = right.child_by_field_name("function")
            if func_node:
                # Handle both `namedtuple(...)` and `collections.namedtuple(...)`
                func_name = ""
                if func_node.type == "identifier":
                    func_name = self._txt(func_node, src)
                elif func_node.type == "attribute":
                    attr = func_node.child_by_field_name("attribute")
                    func_name = self._txt(attr, src) if attr else ""
                if func_name in self._CLASS_FACTORIES:
                    body = self._txt(stmt_node, src)
                    symbols.append(
                        Symbol(
                            file_id=file_id,
                            name=name,
                            qualified_name=name,
                            kind=SymbolKind.CLASS,
                            line_start=stmt_node.start_point[0] + 1,
                            line_end=stmt_node.end_point[0] + 1,
                            signature=f"class {name}  # {func_name}",
                            body=body[:500],
                            is_public=not name.startswith("_"),
                        )
                    )
                    return

        if not self._is_constant_name(name):
            return

        body = self._txt(stmt_node, src)
        if len(body) > 800:
            body = body[:800] + "..."

        # For __all__, prepend a human-readable exports line so the exported
        # names are prominently surfaced for both BM25 and vector search.
        # e.g. "__all__ = ['Foo', 'Bar']" → "Exports: Foo, Bar\n__all__ = ..."
        if name == "__all__":
            right = assign_node.child_by_field_name("right")
            exported = self._parse_string_list(right, src) if right else []
            if exported:
                body = f"Exports: {', '.join(exported)}\n{body}"

        symbols.append(
            Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.CONSTANT,
                line_start=stmt_node.start_point[0] + 1,
                line_end=stmt_node.end_point[0] + 1,
                signature=body.split("\n")[0][:200],
                body=body,
            )
        )

    def _handle_type_alias(
        self,
        node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
    ) -> None:
        """
        Handle Python 3.12+ type alias statements: type Vector = list[float]
        The tree-sitter node is type_alias_statement with a nested 'type' child
        that contains an identifier for the alias name.
        Extracted as kind=INTERFACE (consistent with TypeScript type aliases).
        """
        # Structure: type_alias_statement → 'type' keyword (leaf), 'type' node (name),
        # '=', 'type' node (value)
        # The keyword 'type' is also a 'type' leaf node — skip it and find the name node,
        # which is the first 'type' node that contains an identifier child.
        name_node: Node | None = None
        for child in node.children:
            if child.type == "type" and child.child_count > 0:
                ident = self._get_child_by_type(child, "identifier")
                if ident:
                    name_node = ident
                    break
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
                is_public=not name.startswith("_"),
            )
        )

    def _handle_class_expression(
        self,
        assign_node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        class_local_idx: int,
    ) -> None:
        """
        Extract annotated class field declarations as VARIABLE symbols, and
        ALL_CAPS class-level constants as CONSTANT symbols.

        Both plain and annotated assignments inside a class body are plain
        `assignment` nodes. A 'type' child on the assignment node indicates a
        type annotation (provider: str = "sqlite").

        Handles:
          - Pydantic fields:     provider: Literal["sqlite", "postgres"] = "sqlite"
          - Dataclass fields:    name: str  /  age: int = 0
          - TypedDict fields:    name: str
          - Class-level consts:  MAX_SIZE: ClassVar[int] = 100

        Skips single-underscore private fields to reduce noise.
        """
        stmt_node = assign_node
        left = assign_node.child_by_field_name("left")
        if not left or left.type != "identifier":
            return  # skip self.x = ..., tuple targets, subscript assignments

        name = self._txt(left, src)

        # Detect whether this is a type-annotated field (has a 'type' child on assignment)
        has_annotation = any(c.type == "type" for c in assign_node.children)

        parent_sym = symbols[class_local_idx]
        class_name = parent_sym.name
        is_enum_class = parent_sym.kind == SymbolKind.ENUM
        body = self._txt(stmt_node, src)
        if len(body) > 500:
            body = body[:500] + "..."

        if is_enum_class and not name.startswith("_"):
            # Inside an Enum subclass every non-private assignment is a member constant,
            # regardless of casing (PENDING = 1, active = 2, ERROR = "err").
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=f"{class_name}.{name}",
                    kind=SymbolKind.CONSTANT,
                    line_start=stmt_node.start_point[0] + 1,
                    line_end=stmt_node.end_point[0] + 1,
                    signature=body.split("\n")[0][:200],
                    body=body,
                    parent_id=class_local_idx,
                    is_public=True,
                )
            )
        elif self._is_constant_name(name):
            # ALL_CAPS or __dunder__ class constant — annotated or not:
            #   MAX_SIZE: ClassVar[int] = 100  /  MAX_SIZE = 100
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=f"{class_name}.{name}",
                    kind=SymbolKind.CONSTANT,
                    line_start=stmt_node.start_point[0] + 1,
                    line_end=stmt_node.end_point[0] + 1,
                    signature=body.split("\n")[0][:200],
                    body=body,
                    parent_id=class_local_idx,
                    is_public=not name.startswith("_"),
                )
            )
        elif has_annotation:
            # Type-annotated instance field: provider: str = "sqlite"
            # Skip single-underscore private fields but keep dunder (__slots__, etc.)
            if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
                return
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=f"{class_name}.{name}",
                    kind=SymbolKind.VARIABLE,
                    line_start=stmt_node.start_point[0] + 1,
                    line_end=stmt_node.end_point[0] + 1,
                    signature=body.split("\n")[0][:200],
                    body=body,
                    parent_id=class_local_idx,
                    is_public=not name.startswith("_"),
                )
            )

    def _parse_string_list(self, node: Node, src: bytes) -> list[str]:
        """
        Parse a Python list/tuple of string literals and return the string values.
        Used to extract names from __all__ = ["Foo", "Bar", ...].
        Returns empty list if the node is not a list/tuple of strings.
        """
        if node.type not in ("list", "tuple"):
            return []
        names: list[str] = []
        for child in node.children:
            if child.type == "string":
                raw = self._txt(child, src)
                # Strip surrounding quotes: "foo" → foo, 'foo' → foo
                for quote in ('"""', "'''", '"', "'"):
                    if (
                        raw.startswith(quote)
                        and raw.endswith(quote)
                        and len(raw) > 2 * len(quote) - 1
                    ):
                        names.append(raw[len(quote) : -len(quote)])
                        break
        return names

    @staticmethod
    def _is_constant_name(name: str) -> bool:
        """True for ALL_CAPS constants and __dunder__ package metadata names."""
        # ALL_CAPS with optional leading underscores: LANGUAGES, _GRAMMAR_LOADERS, MAX_WORKERS
        if re.match(r"^_*[A-Z][A-Z0-9_]*$", name):
            return True
        # __dunder__ names: __version__, __all__, __author__
        if name.startswith("__") and name.endswith("__") and len(name) > 4:
            return True
        return False

    def _handle_call(
        self,
        node: Node,
        src: bytes,
        raw_calls: list[tuple[int | None, str, int, str | None]],
        current_func_local_idx: int | None,
        param_types: dict[str, str],
        class_attr_types: dict[str, str],
    ) -> None:
        """Extract the callee name from a call node.

        For method calls of the form ``receiver.method()``, attempts to resolve
        the static type of ``receiver`` two ways:

          1. ``receiver`` is a plain identifier — resolved from the enclosing
             function's annotated parameter list OR a local variable assigned
             from a call to a function with a declared return type (both live
             in ``param_types``; see ``_handle_local_var_assignment``).
          2. ``receiver`` is the literal attribute expression ``self.attr`` —
             resolved from the enclosing class's self-attribute type map
             (``class_attr_types``; see ``_collect_self_attr_types``). Any
             OTHER attribute chain (``other_obj.attr.method()``,
             ``self.attr.nested.method()``) is deliberately left unresolved —
             widening beyond the literal ``self.attr`` pattern risks a wrong
             hint, which is worse than no hint.

        When found, the type name is stored as ``callee_type_hint`` on the
        resulting CallEdge so that ``resolve_cross_file_calls()`` can use
        priority-2 type-hint resolution instead of falling back to ambiguous
        name-only matching.
        """
        func_node = node.child_by_field_name("function")
        if not func_node:
            return

        type_hint: str | None = None

        if func_node.type == "identifier":
            # Simple call: foo()
            name = self._txt(func_node, src)
        elif func_node.type == "attribute":
            # Method call: obj.foo()
            attr = func_node.child_by_field_name("attribute")
            name = self._txt(attr, src) if attr else ""
            obj_node = func_node.child_by_field_name("object")
            if obj_node and obj_node.type == "identifier":
                # receiver.foo() — try param annotation / local-var return type
                receiver_name = self._txt(obj_node, src)
                type_hint = param_types.get(receiver_name)
            elif obj_node and obj_node.type == "attribute":
                # self.attr.foo() — try the class-scoped self-attr map, but
                # ONLY for the literal `self.attr` shape (object is exactly
                # the identifier "self"), never a longer or non-self chain.
                inner_obj = obj_node.child_by_field_name("object")
                inner_attr = obj_node.child_by_field_name("attribute")
                if (
                    inner_obj is not None
                    and inner_obj.type == "identifier"
                    and self._txt(inner_obj, src) == "self"
                    and inner_attr is not None
                ):
                    attr_name = self._txt(inner_attr, src)
                    type_hint = class_attr_types.get(attr_name)
        else:
            return

        if name:
            raw_calls.append(
                (
                    current_func_local_idx,
                    name,
                    node.start_point[0] + 1,
                    type_hint,
                )
            )

    # ------------------------------------------------------------------
    # Parameter type extraction (for callee_type_hint)
    # ------------------------------------------------------------------

    def _extract_param_types(self, params_node: Node, src: bytes) -> dict[str, str]:
        """
        Return a mapping of {parameter_name: type_annotation} for a function's
        parameter list.  Only simple identifier type annotations are captured
        (e.g. ``user_service: UserService``); generic types (``List[Foo]``) and
        union types are skipped because they cannot be reliably matched against
        a single qualified_name prefix.

        Example: ``(self, user_service: UserService, db: Database)``
        → ``{"user_service": "UserService", "db": "Database"}``
        """
        result: dict[str, str] = {}
        for param in params_node.children:
            # tree-sitter 0.21: typed parameters are `typed_parameter` nodes.
            # tree-sitter 0.22+: they may be `identifier` nodes with a `type` child
            # depending on grammar version.  Handle both.
            if param.type == "typed_parameter":
                name_node = self._get_child_by_type(param, "identifier")
                type_node = param.child_by_field_name("type")
                if name_node and type_node and type_node.type == "type":
                    # Drill into the 'type' wrapper to find the inner identifier
                    inner = self._get_child_by_type(type_node, "identifier")
                    if inner:
                        param_name = self._txt(name_node, src)
                        type_name = self._txt(inner, src)
                        if param_name != "self" and param_name != "cls":
                            result[param_name] = type_name
            elif param.type == "identifier":
                # plain untyped parameter — no hint available, skip
                pass
        return result

    # ------------------------------------------------------------------
    # Local-variable-from-return-type propagation (for callee_type_hint)
    # ------------------------------------------------------------------

    def _collect_function_return_types(self, root: Node, src: bytes) -> dict[str, str]:
        """
        Map every function/method NAME (bare, unqualified) in the file to its
        declared return type annotation, when that annotation is a single
        plain identifier (e.g. ``-> ServiceType:``).  Generic/union return
        types (``-> Optional[Foo]``, ``-> list[Bar]``) are skipped for the
        same reason ``_extract_param_types`` skips them: they cannot be
        reliably matched against a single qualified_name prefix.

        Computed ONCE for the whole file (module functions AND methods,
        keyed by bare name — collisions between same-named methods on
        different classes are an accepted, rare imprecision; this mirrors
        the same "simple, not a whole-program analysis" scope as the
        local-variable propagation that consumes this map).
        """
        result: dict[str, str] = {}
        self._scan_function_return_types(root, src, result)
        return result

    def _scan_function_return_types(self, node: Node, src: bytes, result: dict[str, str]) -> None:
        for child in node.children:
            target = child
            if target.type == "decorated_definition":
                target = self._get_child_by_type(target, "function_definition") or target
            if target.type == "function_definition":
                name_node = self._get_child_by_type(target, "identifier")
                return_node = target.child_by_field_name("return_type")
                if name_node and return_node is not None and return_node.type == "type":
                    inner = self._get_child_by_type(return_node, "identifier")
                    if inner:
                        result[self._txt(name_node, src)] = self._txt(inner, src)
            self._scan_function_return_types(child, src, result)

    def _handle_local_var_assignment(
        self,
        assign_node: Node,
        src: bytes,
        param_types: dict[str, str],
        func_return_types: dict[str, str],
    ) -> None:
        """
        Track ``x = some_func()`` inside a function/method body so that a
        LATER ``x.method()`` call in the SAME body can resolve
        ``callee_type_hint`` from ``some_func``'s declared return type.

        Mutates ``param_types`` IN PLACE — the same dict ``_handle_call``
        reads from for receiver resolution — so the effect is visible only
        for statements that come AFTER this assignment in source order.
        ``_walk`` iterates a block's children in source order, so a call on
        `x` BEFORE this assignment is visited (and resolved, or not) before
        this mutation ever happens: no retroactive application is possible.

        Only a plain ``x = ...`` (bare identifier target) is handled — tuple
        unpacking, attribute targets (``self.x = ...``, tracked separately
        per-class in ``_collect_self_attr_types``), and subscript targets are
        left untouched.

        Deliberately simple "last assignment wins" semantics, not a full
        data-flow analysis: reassigning ``x`` to something that does NOT
        resolve to a known return type clears any previous mapping for that
        name, so a stale type is never carried past a reassignment.
        """
        left = assign_node.child_by_field_name("left")
        if not left or left.type != "identifier":
            return
        var_name = self._txt(left, src)

        resolved_type: str | None = None
        right = assign_node.child_by_field_name("right")
        if right is not None and right.type == "call":
            func_node = right.child_by_field_name("function")
            if func_node is not None and func_node.type == "identifier":
                callee_name = self._txt(func_node, src)
                resolved_type = func_return_types.get(callee_name)

        if resolved_type:
            param_types[var_name] = resolved_type
        else:
            param_types.pop(var_name, None)

    # ------------------------------------------------------------------
    # self.attr type tracking, per-class (for callee_type_hint)
    # ------------------------------------------------------------------

    def _collect_self_attr_types(
        self, class_body: Node, src: bytes, func_return_types: dict[str, str]
    ) -> dict[str, str]:
        """
        Scan every method in a class body for ``self.attr = ...`` assignments
        and build a name -> type map BEFORE any method's calls are resolved,
        since an attribute set in one method (typically ``__init__``) is
        commonly used in another (e.g. ``process()``). Two forms are captured
        (see ``_record_self_attr_assignment`` for the exact rules):

          - Annotated:        ``self.attr: SomeType = ...``
          - Constructor-call:  ``self.attr = SomeClass(...)``

        ``func_return_types`` (the same whole-file function/method-name ->
        declared-return-type map used by ``_handle_local_var_assignment``) is
        threaded through so the constructor-call form can tell a factory
        FUNCTION call (``self.attr = get_service()``) apart from an actual
        class constructor call — see ``_record_self_attr_assignment``.

        Nested class bodies are skipped entirely — ``self`` inside a nested
        class's own methods refers to instances of THAT class, not this one.
        """
        attr_types: dict[str, str] = {}
        for child in class_body.children:
            target = child
            if target.type == "decorated_definition":
                target = self._get_child_by_type(target, "function_definition") or target
            if target.type != "function_definition":
                continue
            method_body = target.child_by_field_name("body")
            if method_body is not None:
                self._scan_self_attr_assignments(method_body, src, attr_types, func_return_types)
        return attr_types

    def _scan_self_attr_assignments(
        self,
        node: Node,
        src: bytes,
        attr_types: dict[str, str],
        func_return_types: dict[str, str],
    ) -> None:
        for child in node.children:
            if child.type == "class_definition":
                continue  # nested class has its own `self` — do not descend
            if child.type == "assignment":
                self._record_self_attr_assignment(child, src, attr_types, func_return_types)
            self._scan_self_attr_assignments(child, src, attr_types, func_return_types)

    def _record_self_attr_assignment(
        self,
        assign_node: Node,
        src: bytes,
        attr_types: dict[str, str],
        func_return_types: dict[str, str],
    ) -> None:
        left = assign_node.child_by_field_name("left")
        if not left or left.type != "attribute":
            return  # not `self.attr = ...` — plain-name / tuple / subscript target

        obj_node = left.child_by_field_name("object")
        attr_node = left.child_by_field_name("attribute")
        if (
            obj_node is None
            or obj_node.type != "identifier"
            or self._txt(obj_node, src) != "self"
            or attr_node is None
        ):
            return
        attr_name = self._txt(attr_node, src)

        # Annotated form takes priority: self.attr: SomeType = ...
        type_node = assign_node.child_by_field_name("type")
        if type_node is not None:
            inner = self._get_child_by_type(type_node, "identifier")
            if inner:
                attr_types[attr_name] = self._txt(inner, src)
            return

        # Constructor-call inference: self.attr = SomeClass(...) — only when
        # the callee is a plain identifier (not `module.Class()`, which is a
        # multi-level chain out of scope here).
        right = assign_node.child_by_field_name("right")
        if right is None or right.type != "call":
            return
        func_node = right.child_by_field_name("function")
        if func_node is None or func_node.type != "identifier":
            return
        callee_name = self._txt(func_node, src)

        # 1) Factory-FUNCTION call: `self.attr = get_service()` where
        #    `get_service` has a declared `-> ReturnType` annotation
        #    elsewhere in the file (mirrors `_handle_local_var_assignment`,
        #    which resolves the exact same shape for plain local variables).
        #    The factory function's own bare name (e.g. "get_service") must
        #    NEVER be recorded as the hint — that would either fail
        #    resolve_cross_file_calls' priority-2 lookup outright, or, worse,
        #    silently match an unrelated symbol elsewhere in the whole-project
        #    index that happens to share that name as a qualified_name prefix.
        if callee_name in func_return_types:
            resolved_type = func_return_types[callee_name]
            if resolved_type:
                attr_types[attr_name] = resolved_type
            return

        # 2) Constructor call: `self.attr = SomeClass(...)`. `callee_name` is
        #    NOT a known function/method (checked above), so it cannot be
        #    confirmed as a factory — but nothing here confirms it IS a class
        #    either (classes may be imported from other files and are
        #    invisible to this whole-file walk). Require it to at least LOOK
        #    like a class per PEP 8 convention (PascalCase) and not be one of
        #    the builtin type constructors that would produce a
        #    useless/misleading hint (self.attr = list()). This rejects the
        #    common idiomatic factory patterns that are NOT classes
        #    (get_x()/create_x()/load_x()/build_x()/make_x(), or bare
        #    builtins like sorted()/open()/zip()) — all lowercase — while
        #    still accepting genuine constructor calls, including on classes
        #    defined in other files. A wrong hint is worse than no hint.
        if callee_name[0].isupper() and callee_name not in self._BUILTIN_CONSTRUCTORS:
            attr_types[attr_name] = callee_name

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_import(self, node: Node, src: bytes, file_id: int) -> list[ImportEdge]:
        edges: list[ImportEdge] = []

        if node.type == "import_statement":
            # import foo           → ImportEdge(imported_from="foo", imported_names=[])
            # import foo.bar       → ImportEdge(imported_from="foo.bar", imported_names=[])
            # import foo as f      → ImportEdge(imported_from="foo", imported_names=[])
            for child in node.children:
                if child.type == "dotted_name":
                    edges.append(
                        ImportEdge(
                            file_id=file_id,
                            imported_from=self._txt(child, src),
                            imported_names=[],
                        )
                    )
                elif child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        edges.append(
                            ImportEdge(
                                file_id=file_id,
                                imported_from=self._txt(name_node, src),
                                imported_names=[],
                            )
                        )

        elif node.type == "import_from_statement":
            # from foo import bar, baz   → from=foo, names=[bar, baz]
            # from . import something    → from=., names=[something]
            # from foo import *          → from=foo, names=[*]
            #
            # Key: use field access for module_name (first dotted_name after "from"),
            # then collect ALL remaining dotted_name children as imported names.
            # This avoids the "last dotted_name wins" bug from iterating children blindly.

            module_node = node.child_by_field_name("module_name")
            if module_node is None:
                # relative import with no explicit module: "from . import X"
                module_node = self._get_child_by_type(node, "relative_import")

            if module_node is None:
                return edges

            module_name = self._txt(module_node, src)
            imported_names: list[str] = []

            # Everything after the "import" keyword is an imported name
            past_import_kw = False
            for child in node.children:
                if child.type == "import":
                    past_import_kw = True
                    continue
                if not past_import_kw:
                    continue
                if child.type == "dotted_name":
                    imported_names.append(self._txt(child, src))
                elif child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        imported_names.append(self._txt(name_node, src))
                elif child.type == "wildcard_import":
                    imported_names = ["*"]

            if module_name:
                edges.append(
                    ImportEdge(
                        file_id=file_id,
                        imported_from=module_name,
                        imported_names=imported_names,
                    )
                )

        return edges

    # ------------------------------------------------------------------
    # Signature builders
    # ------------------------------------------------------------------

    def _func_signature(self, node: Node, src: bytes) -> str:
        """Reconstruct 'def name(params) -> return_type' from AST nodes."""
        name_node = self._get_child_by_type(node, "identifier")
        params_node = node.child_by_field_name("parameters")
        return_node = node.child_by_field_name("return_type")

        name = self._txt(name_node, src) if name_node else "?"
        params = self._txt(params_node, src) if params_node else "()"
        ret = f" -> {self._txt(return_node, src)}" if return_node else ""
        return f"def {name}{params}{ret}"

    def _class_signature(self, node: Node, src: bytes) -> str:
        """Reconstruct 'class Name(Base1, Base2)' from AST nodes."""
        name_node = self._get_child_by_type(node, "identifier")
        args_node = node.child_by_field_name("superclasses")

        name = self._txt(name_node, src) if name_node else "?"
        bases = f"({self._txt(args_node, src)})" if args_node else ""
        return f"class {name}{bases}"

    # ------------------------------------------------------------------
    # Decorator extraction
    # ------------------------------------------------------------------

    def _extract_decorators(self, decorated_node: Node, src: bytes) -> list[str]:
        """Extract @decorator texts from a decorated_definition node."""
        result: list[str] = []
        for child in decorated_node.children:
            if child.type == "decorator":
                text = self._txt(child, src).strip()
                if len(text) > 200:
                    text = text[:200] + "..."
                result.append(text)
        return result

    # ------------------------------------------------------------------
    # Docstring extraction
    # ------------------------------------------------------------------

    def _get_docstring(self, block_node: Node, src: bytes) -> str | None:
        """Return the docstring of a function/class body block, or None."""
        for child in block_node.children:
            if child.type in ("string", "concatenated_string"):
                return self._clean_docstring(self._txt(child, src))
            # Docstring must be the FIRST real statement — stop after first
            if child.type not in ("comment", "\n"):
                break
        return None

    @staticmethod
    def _clean_docstring(raw: str) -> str:
        """Strip triple quotes and leading/trailing whitespace."""
        for quote in ('"""', "'''", '"', "'"):
            if raw.startswith(quote) and raw.endswith(quote) and len(raw) >= 2 * len(quote):
                return raw[len(quote) : -len(quote)].strip()
        return raw.strip()

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
