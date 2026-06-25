"""Razor Component parser (.razor / Blazor files).

No tree-sitter grammar exists for Razor — we use regex + brace-counting to
extract the embedded C# code and directives, then pipe through CSharpParser.

Extracts:
  - <component> MODULE symbol — name from @page route, signature includes route
  - @using directives → ImportEdge
  - @inject directives → ImportEdge (service type as imported_from)
  - @inherits BaseType → TypeEdge on the component symbol
  - @attribute [Attr] → recorded as decorator on component symbol
  - @code { } block → full C# parsing via CSharpParser
      - Methods (kind=METHOD, parent_id=component)
      - Properties / Parameters (kind=VARIABLE, parent_id=component)
      - Constants (kind=CONSTANT, parent_id=component)
      - Call sites inside methods → CallEdge
  - Inline @{ } statement blocks → call site extraction only (no member symbols)

Line numbers:
  All symbols extracted from @code are adjusted to their actual line in the
  .razor file (not the wrapped C# text line numbers).

Design:
  _RazorBase contains all extraction logic shared with CshtmlParser.
  RazorParser subclasses it for .razor-specific behaviour (strict @code only).
"""

from __future__ import annotations

import re
from typing import Optional

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge
from trelix.indexing.parser.base import BaseParser, ParseResult

# ---------------------------------------------------------------------------
# Directive regexes
# ---------------------------------------------------------------------------

# @page "/route" or @page "/route/{id:int}"
_PAGE_RE = re.compile(r'^\s*@page\s+"([^"]*)"', re.MULTILINE)
# @using Namespace.Path
_USING_RE = re.compile(r'^\s*@using\s+([\w.]+)', re.MULTILINE)
# @namespace MyApp.Components
_NAMESPACE_RE = re.compile(r'^\s*@namespace\s+([\w.]+)', re.MULTILINE)
# @inject ServiceType VarName  (ServiceType may be generic: ILogger<T>)
_INJECT_RE = re.compile(r'^\s*@inject\s+([\w][\w.<>, \[\]?]*?)\s+(\w+)\s*$', re.MULTILINE)
# @inherits BaseComponent<T>
_INHERITS_RE = re.compile(r'^\s*@inherits\s+([\w][\w.<>, \[\]?]*)', re.MULTILINE)
# @implements IDisposable / @implements IAsyncDisposable<T>
_IMPLEMENTS_RE = re.compile(r'^\s*@implements\s+([\w][\w.<>, \[\]?]*)', re.MULTILINE)
# @layout MainLayout
_LAYOUT_RE = re.compile(r'^\s*@layout\s+([\w.]+)', re.MULTILINE)
# @typeparam T  or  @typeparam T where T : IBase
_TYPEPARAM_RE = re.compile(r'^\s*@typeparam\s+(.+)', re.MULTILINE)
# @rendermode InteractiveServer / InteractiveWebAssembly / InteractiveAuto
_RENDERMODE_RE = re.compile(r'^\s*@rendermode\s+(\S+)', re.MULTILINE)
# @attribute [Authorize] or @attribute [Authorize(Roles="admin")]
_ATTRIBUTE_RE = re.compile(r'^\s*@attribute\s+(\[.+?\])', re.MULTILINE)
# @code { or @functions {  (the opening brace may be on next line)
_CODE_BLOCK_RE = re.compile(r'@(?:code|functions)\s*\{', re.MULTILINE)
# Inline @{ statement block
_STMT_BLOCK_RE = re.compile(r'(?<!\w)@\{', re.MULTILINE)
# PascalCase HTML tags = Blazor component references: <MyComponent or <Namespace.Component
_COMPONENT_REF_RE = re.compile(r'<([A-Z][A-Za-z0-9]*(?:\.[A-Z][A-Za-z0-9]*)*)[\s/>]')


def _extract_brace_block(src: str, open_brace_pos: int) -> tuple[str, int]:
    """
    Extract text between balanced braces starting at open_brace_pos (the '{').
    Returns (inner_content, position_after_closing_brace).
    Handles nested braces and braces inside strings/comments approximately.
    """
    depth = 0
    i = open_brace_pos
    n = len(src)
    in_string: Optional[str] = None
    in_verbatim = False
    while i < n:
        ch = src[i]
        if in_string:
            if ch == '\\' and not in_verbatim:
                i += 2
                continue
            if ch == in_string:
                in_string = None
        elif ch == '@' and i + 1 < n and src[i + 1] == '"':
            in_verbatim = True
            in_string = '"'
            i += 2
            continue
        elif ch in ('"', "'"):
            in_string = ch
            in_verbatim = False
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return src[open_brace_pos + 1:i], i + 1
        i += 1
    return src[open_brace_pos + 1:], n


def _count_lines_to(src: str, pos: int) -> int:
    """Return 1-based line number of character at pos in src."""
    return src[:pos].count('\n') + 1


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _RazorBase(BaseParser):
    """
    Shared extraction logic for Razor (.razor) and CSHTML (.cshtml) files.
    Subclasses override `language_name` and may adjust parsing rules.
    """

    def __init__(self) -> None:
        from trelix.indexing.parser.extractors.csharp import CSharpParser
        self._cs_parser = CSharpParser()

    @property
    def language_name(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self, source: str, file_id: int) -> ParseResult:
        symbols: list[Symbol] = []
        import_edges: list[ImportEdge] = []
        type_edges: list[TypeEdge] = []
        raw_calls: list[tuple[Optional[int], str, int]] = []

        component_local_idx = self._make_module_symbol(
            source, file_id, symbols, import_edges, type_edges
        )

        for m in _USING_RE.finditer(source):
            import_edges.append(ImportEdge(
                file_id=file_id,
                imported_from=m.group(1),
                imported_names=[],
            ))

        for m in _INJECT_RE.finditer(source):
            service_type = m.group(1).strip()
            base_type = re.sub(r'<.*>', '', service_type).strip()
            import_edges.append(ImportEdge(
                file_id=file_id,
                imported_from=base_type,
                imported_names=[m.group(2).strip()],
            ))

        for code_match in _CODE_BLOCK_RE.finditer(source):
            brace_pos = code_match.end() - 1
            block_line = _count_lines_to(source, brace_pos)
            content, _ = _extract_brace_block(source, brace_pos)
            self._parse_code_block(
                content, file_id, symbols, import_edges, type_edges, raw_calls,
                component_local_idx, line_offset=block_line,
            )

        for stmt_match in _STMT_BLOCK_RE.finditer(source):
            brace_pos = stmt_match.end() - 1
            content, _ = _extract_brace_block(source, brace_pos)
            block_line = _count_lines_to(source, brace_pos)
            self._extract_calls_from_snippet(
                content, raw_calls,
                caller_idx=component_local_idx,
                line_offset=block_line,
            )

        seen_refs: set[str] = set()
        for ref_match in _COMPONENT_REF_RE.finditer(source):
            tag_name = ref_match.group(1)
            if tag_name not in seen_refs:
                seen_refs.add(tag_name)
                raw_calls.append((
                    component_local_idx,
                    tag_name,
                    _count_lines_to(source, ref_match.start()),
                ))

        call_edges: list[CallEdge] = [
            CallEdge(caller_id=caller_idx, callee_name=name, line=line)
            for caller_idx, name, line in raw_calls
            if caller_idx is not None
        ]

        return ParseResult(
            symbols=symbols,
            call_edges=call_edges,
            import_edges=import_edges,
            parse_errors=0,
            type_edges=type_edges,
        )

    # ------------------------------------------------------------------
    # Module symbol builder (overridden by CshtmlParser)
    # ------------------------------------------------------------------

    def _make_module_symbol(
        self,
        source: str,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
    ) -> int:
        """Create the top-level MODULE symbol. Returns local index."""
        page_match = _PAGE_RE.search(source)
        route = page_match.group(1) if page_match else ""
        ns_match = _NAMESPACE_RE.search(source)
        namespace = ns_match.group(1) if ns_match else ""
        layout_match = _LAYOUT_RE.search(source)
        layout = layout_match.group(1) if layout_match else ""
        rendermode_match = _RENDERMODE_RE.search(source)
        rendermode = rendermode_match.group(1).strip() if rendermode_match else ""
        typeparams = [m.group(1).strip() for m in _TYPEPARAM_RE.finditer(source)]

        decorators: list[str] = [m.group(1) for m in _ATTRIBUTE_RE.finditer(source)]
        if rendermode:
            decorators.append(f"@rendermode {rendermode}")

        sig_parts = ["@component"]
        if route:
            sig_parts.append(f'@page "{route}"')
        if namespace:
            sig_parts.append(f"@namespace {namespace}")
        if typeparams:
            sig_parts.append("@typeparam " + ", ".join(typeparams))
        if layout:
            sig_parts.append(f"@layout {layout}")
        signature = "  ".join(sig_parts)

        body_lines: list[str] = []
        if route:
            body_lines.append(f'@page "{route}"')
        if layout:
            body_lines.append(f"@layout {layout}")
        for tp in typeparams:
            body_lines.append(f"@typeparam {tp}")
        for m in _USING_RE.finditer(source):
            body_lines.append(f"@using {m.group(1)}")
        for m in _INJECT_RE.finditer(source):
            body_lines.append(f"@inject {m.group(1).strip()} {m.group(2).strip()}")
        for dec in decorators:
            body_lines.append(f"@attribute {dec}" if dec.startswith("[") else dec)
        body = "\n".join(body_lines) if body_lines else signature

        component_local_idx = len(symbols)
        symbols.append(Symbol(
            file_id=file_id,
            name="<component>",
            qualified_name="<component>",
            kind=SymbolKind.MODULE,
            line_start=1,
            line_end=source.count('\n') + 1,
            signature=signature,
            body=body,
            decorators=decorators,
            is_public=True,
        ))

        inh_match = _INHERITS_RE.search(source)
        if inh_match:
            base = re.sub(r'<.*>', '', inh_match.group(1)).strip()
            type_edges.append(TypeEdge(
                from_symbol_id=component_local_idx,
                to_type_name=base,
                edge_kind="extends",
            ))

        for impl_match in _IMPLEMENTS_RE.finditer(source):
            iface = re.sub(r'<.*>', '', impl_match.group(1)).strip()
            type_edges.append(TypeEdge(
                from_symbol_id=component_local_idx,
                to_type_name=iface,
                edge_kind="implements",
            ))

        if layout:
            import_edges.append(ImportEdge(
                file_id=file_id,
                imported_from=layout,
                imported_names=[],
            ))

        return component_local_idx

    # ------------------------------------------------------------------
    # C# code block parsing
    # ------------------------------------------------------------------

    def _parse_code_block(
        self,
        content: str,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
        raw_calls: list[tuple[Optional[int], str, int]],
        component_local_idx: int,
        line_offset: int,
    ) -> None:
        """Parse a @code { } block by wrapping it in a C# class shell."""
        wrapped = f"public class _RazorComponent_ {{\n{content}\n}}"
        try:
            cs_result = self._cs_parser.parse(wrapped, file_id)
        except Exception:
            return

        wrapper_class_idx = 0
        for sym in cs_result.symbols[1:]:
            adjusted_start = max(1, sym.line_start - 1 + line_offset)
            adjusted_end = max(adjusted_start, sym.line_end - 1 + line_offset)

            new_parent: Optional[int]
            if sym.parent_id == wrapper_class_idx:
                new_parent = component_local_idx
            elif sym.parent_id is not None:
                new_parent = len(symbols) + sym.parent_id - 1
            else:
                new_parent = None

            symbols.append(Symbol(
                file_id=file_id,
                name=sym.name,
                qualified_name=sym.qualified_name,
                kind=sym.kind,
                line_start=adjusted_start,
                line_end=adjusted_end,
                signature=sym.signature,
                body=sym.body,
                docstring=sym.docstring,
                decorators=sym.decorators,
                is_public=sym.is_public,
                parent_id=new_parent,
            ))

        symbols_before = len(symbols) - len(cs_result.symbols) + 1
        for edge in cs_result.call_edges:
            if edge.caller_id is None:
                continue
            if edge.caller_id == wrapper_class_idx:
                remapped = component_local_idx
            else:
                remapped = symbols_before + edge.caller_id - 1
            adjusted_line = max(1, edge.line - 1 + line_offset)
            raw_calls.append((remapped, edge.callee_name, adjusted_line))

        import_edges.extend(cs_result.import_edges)
        for te in cs_result.type_edges:
            if te.from_symbol_id == wrapper_class_idx:
                type_edges.append(TypeEdge(
                    from_symbol_id=component_local_idx,
                    to_type_name=te.to_type_name,
                    edge_kind=te.edge_kind,
                ))

    # ------------------------------------------------------------------
    # Call extraction from statement snippets
    # ------------------------------------------------------------------

    def _extract_calls_from_snippet(
        self,
        content: str,
        raw_calls: list[tuple[Optional[int], str, int]],
        caller_idx: Optional[int],
        line_offset: int,
    ) -> None:
        """Extract method call names from a statement block using regex."""
        for m in re.finditer(r'\b([A-Za-z_]\w+)\s*\(', content):
            name = m.group(1)
            if name in _CS_KEYWORDS:
                continue
            call_line = _count_lines_to(content, m.start()) + line_offset
            raw_calls.append((caller_idx, name, call_line))

    # ------------------------------------------------------------------
    # Error counting (no tree-sitter, always 0)
    # ------------------------------------------------------------------

    def _count_errors(self, _node: object) -> int:
        return 0


# C# keywords to skip when extracting call names from statement snippets
_CS_KEYWORDS: frozenset[str] = frozenset({
    "if", "else", "for", "foreach", "while", "do", "switch", "case",
    "return", "new", "var", "int", "string", "bool", "void", "async",
    "await", "try", "catch", "finally", "throw", "using", "lock",
    "typeof", "nameof", "sizeof", "stackalloc", "checked", "unchecked",
    "true", "false", "null", "this", "base", "static", "readonly",
    "const", "public", "private", "protected", "internal", "override",
    "virtual", "abstract", "sealed", "partial", "class", "interface",
    "struct", "enum", "delegate", "event", "get", "set", "init",
    "where", "select", "from", "in", "out", "ref", "params",
})


# ---------------------------------------------------------------------------
# RazorParser — .razor (Blazor components)
# ---------------------------------------------------------------------------

class RazorParser(_RazorBase):
    """
    Parser for Blazor Razor Components (.razor).

    Extracts the @code block as C# class members, @using/@inject as imports,
    and @page/@inherits/@attribute as component metadata.
    """

    @property
    def language_name(self) -> str:
        return "razor"
