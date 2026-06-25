"""
Python parser: direct AST traversal using Tree-sitter.

Design: tree-sitter parses source into an AST; all extraction logic is ours.
We walk the AST directly (not .scm queries) for full control over parent-child
linkage and to handle Python's complex syntax (decorators, nested classes, etc.).

NOTE on tree-sitter 0.21: annotated assignments (x: str = "y") are
expression_statement > assignment WITH a 'type' child — same node type as plain
assignments, just with an annotation. There is no separate 'annotated_assignment'
node in this grammar version.

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
"""

from __future__ import annotations

import re

from tree_sitter import Language, Node, Parser

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge

from ..base import BaseParser, ParseResult


def _get_python_language() -> Language:
    """
    Load the Python grammar, trying tree_sitter_languages first (0.21-era
    bundled grammars), then falling back to the modern tree-sitter-python
    package (tree-sitter >=0.22).
    """
    try:
        import tree_sitter_languages  # type: ignore[import]

        return tree_sitter_languages.get_language("python")  # type: ignore[no-any-return]
    except ImportError:
        import tree_sitter_python  # type: ignore[import]

        return Language(tree_sitter_python.language())


def _make_parser(language: Language) -> Parser:
    """
    Construct a tree-sitter Parser, compatible with both the 0.21 API
    (parser.set_language(lang)) and the 0.22+ API (Parser(lang)).

    On tree-sitter 0.21, Parser(lang) does not raise TypeError but silently
    creates a parser with no language set (parse() then raises ValueError).
    We detect this by checking for the set_language attribute, which exists
    only on the 0.21 Parser API, and prefer that path when available.
    """
    try:
        # tree-sitter 0.21 — Parser has set_language method
        p = Parser()
        p.set_language(language)  # type: ignore[attr-defined]
        return p
    except (TypeError, AttributeError):
        # tree-sitter >=0.22 — Language is passed to the constructor
        return Parser(language)


class PythonParser(BaseParser):
    """Tree-sitter based Python parser using direct AST traversal."""

    # Cap on annotated field symbols extracted per class.
    # Prevents symbol flood from very large Pydantic/SQLAlchemy models.
    MAX_CLASS_FIELDS: int = 30

    # Base class names that make a class an Enum (all members → CONSTANT)
    _ENUM_BASES: frozenset[str] = frozenset({"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"})
    # Base class names that make a class a Protocol (→ INTERFACE)
    _PROTOCOL_BASES: frozenset[str] = frozenset({"Protocol"})

    def __init__(self) -> None:
        self._ts_lang = _get_python_language()
        self._parser = _make_parser(self._ts_lang)

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
            depth=0,
        )

        # Module-level docstring → file-level summary symbol so architectural
        # queries ("how does X work end-to-end", "what is the Y architecture")
        # can find the right file via its module description.
        module_doc = self._get_docstring(root, source_bytes)
        if module_doc:
            top_sigs = [
                s.signature
                for s in symbols
                if s.parent_id is None and s.kind not in (SymbolKind.CONSTANT,)
            ][:20]
            body = module_doc
            if top_sigs:
                body += "\n\n# Symbols:\n" + "\n".join(top_sigs)
            symbols.insert(
                0,
                Symbol(
                    file_id=file_id,
                    name="<module>",
                    qualified_name="<module>",
                    kind=SymbolKind.MODULE,
                    line_start=1,
                    line_end=root.end_point[0] + 1,
                    signature="module",
                    body=body,
                    docstring=module_doc,
                ),
            )

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
        depth: int,
    ) -> None:
        """Recursive depth-first walk. depth guards against absurdly nested code."""
        if depth > 20:
            return

        for child in node.children:
            ntype = child.type

            # ---- Class definition ----------------------------------------
            if ntype == "class_definition":
                self._handle_class(
                    child, src, file_id, symbols, raw_calls, import_edges, type_edges, depth
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
            # Note: in tree-sitter 0.21, annotated assignments (BASE_URL: str = "x")
            # are ALSO expression_statement > assignment nodes — the 'type' child
            # distinguishes annotated from plain. _handle_module_assignment handles both.
            elif (
                ntype == "expression_statement"
                and parent_class_local_idx is None
                and current_func_local_idx is None
            ):
                self._handle_module_assignment(child, src, file_id, symbols)

            # ---- Class-level expression statements: field annotations and constants ----
            # Covers Pydantic fields, dataclass fields, TypedDict fields, typed class vars.
            # In tree-sitter 0.21, 'provider: str = "x"' is expression_statement > assignment
            # with a 'type' child — same node type as plain assignments, just with annotation.
            elif (
                ntype == "expression_statement"
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

            # ---- Call sites (track for call graph) -----------------------
            elif ntype == "call":
                self._handle_call(child, src, raw_calls, current_func_local_idx, param_types)
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
                    depth + 1,
                )

            # ---- Recurse into statement / expression wrappers ------------
            # expression_statement, assignment, augmented_assignment, etc. all
            # wrap `call` nodes — we must recurse into them to find call sites.
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
                "expression_statement",
                "assignment",
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
            class_name = symbols[parent_class_local_idx].name  # type: ignore[index]
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
        stmt_node: Node,
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
        assign_node = self._get_child_by_type(stmt_node, "assignment")
        if not assign_node:
            return

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
        stmt_node: Node,
        src: bytes,
        file_id: int,
        symbols: list[Symbol],
        class_local_idx: int,
    ) -> None:
        """
        Extract annotated class field declarations as VARIABLE symbols, and
        ALL_CAPS class-level constants as CONSTANT symbols.

        In tree-sitter 0.21, both plain and annotated assignments inside a class
        body appear as expression_statement > assignment. A 'type' child on the
        assignment node indicates a type annotation (provider: str = "sqlite").

        Handles:
          - Pydantic fields:     provider: Literal["sqlite", "postgres"] = "sqlite"
          - Dataclass fields:    name: str  /  age: int = 0
          - TypedDict fields:    name: str
          - Class-level consts:  MAX_SIZE: ClassVar[int] = 100

        Skips single-underscore private fields to reduce noise.
        """
        assign_node = self._get_child_by_type(stmt_node, "assignment")
        if not assign_node:
            return

        left = assign_node.child_by_field_name("left")
        if not left or left.type != "identifier":
            return  # skip self.x = ..., tuple targets, subscript assignments

        name = self._txt(left, src)

        # Detect whether this is a type-annotated field (has a 'type' child on assignment)
        has_annotation = any(c.type == "type" for c in assign_node.children)

        parent_sym = symbols[class_local_idx]  # type: ignore[index]
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
    ) -> None:
        """Extract the callee name from a call node.

        For method calls of the form ``receiver.method()``, attempts to resolve
        the static type of ``receiver`` from the enclosing function's annotated
        parameter list (stored in ``param_types``).  When found, the type name
        is stored as ``callee_type_hint`` on the resulting CallEdge so that
        ``resolve_cross_file_calls()`` can use priority-2 type-hint resolution
        instead of falling back to ambiguous name-only matching.
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
            # Try to resolve receiver type hint from param annotations
            obj_node = func_node.child_by_field_name("object")
            if obj_node and obj_node.type == "identifier":
                receiver_name = self._txt(obj_node, src)
                type_hint = param_types.get(receiver_name)
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
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type in ("string", "concatenated_string"):
                        raw = self._txt(sub, src)
                        return self._clean_docstring(raw)
            # Docstring must be the FIRST real statement — stop after first
            if child.type not in ("comment", "\n", "expression_statement"):
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
