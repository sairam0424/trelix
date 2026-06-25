"""Razor MVC View parser (.cshtml files).

Handles ASP.NET MVC / Razor Pages .cshtml files. Shares all extraction
infrastructure with RazorParser but adds .cshtml-specific directives:
  - @model TypeName  → ImportEdge + recorded in view MODULE symbol
  - @section Name { } → lightweight SECTION symbol (no C# parsing)
  - @functions { }  → same as @code (legacy Razor member block)
  - @{ }            → inline statement blocks (call extraction)
  - @using / @inject / @inherits / @namespace / @attribute  (inherited)

Extracts:
  - <view> MODULE symbol — signature includes @model type
  - @using  → ImportEdge
  - @inject → ImportEdge
  - @model  → ImportEdge (model type as imported_from)
  - @functions { } block → C# method/property extraction via CSharpParser
  - Call sites in @{ } and @functions → CallEdge
  - @inherits → TypeEdge on view symbol
"""

from __future__ import annotations

import re
from typing import Optional

from trelix.core.models import CallEdge, ImportEdge, Symbol, SymbolKind, TypeEdge
from trelix.indexing.parser.base import ParseResult
from trelix.indexing.parser.extractors.razor import (
    _RazorBase,
    _USING_RE,
    _INJECT_RE,
    _INHERITS_RE,
    _NAMESPACE_RE,
    _ATTRIBUTE_RE,
    _STMT_BLOCK_RE,
    _CODE_BLOCK_RE,
    _count_lines_to,
    _extract_brace_block,
)

# @model TypeName (MVC / Razor Pages)
_MODEL_RE = re.compile(r'^\s*@model\s+([\w][\w.<>, \[\]?]*)', re.MULTILINE)
# @section SectionName {
_SECTION_RE = re.compile(r'@section\s+(\w+)\s*\{', re.MULTILINE)
# @page (Razor Pages — no route required)
_PAGE_RE = re.compile(r'^\s*@page(?:\s+"([^"]*)")?', re.MULTILINE)


class CshtmlParser(_RazorBase):
    """
    Parser for ASP.NET MVC / Razor Pages views (.cshtml).

    Reuses _RazorBase extraction infrastructure and overrides the module symbol
    builder to include @model and handle Razor Pages @page directive.
    """

    @property
    def language_name(self) -> str:
        return "cshtml"

    # ------------------------------------------------------------------
    # Override: parse() — same flow but adds @model handling
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

        model_match = _MODEL_RE.search(source)
        if model_match:
            model_type = re.sub(r'<.*>', '', model_match.group(1)).strip()
            import_edges.append(ImportEdge(
                file_id=file_id,
                imported_from=model_type,
                imported_names=[],
            ))

        for m in _INJECT_RE.finditer(source):
            service_type = re.sub(r'<.*>', '', m.group(1)).strip()
            import_edges.append(ImportEdge(
                file_id=file_id,
                imported_from=service_type,
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

        for sec_match in _SECTION_RE.finditer(source):
            name = sec_match.group(1)
            brace_pos = sec_match.end() - 1
            _, end_pos = _extract_brace_block(source, brace_pos)
            sec_end_line = _count_lines_to(source, end_pos)
            sec_start_line = _count_lines_to(source, sec_match.start())
            symbols.append(Symbol(
                file_id=file_id,
                name=name,
                qualified_name=name,
                kind=SymbolKind.SECTION,
                line_start=sec_start_line,
                line_end=sec_end_line,
                signature=f"@section {name}",
                body=f"@section {name} {{ ... }}",
                parent_id=component_local_idx,
                is_public=True,
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
    # Override: module symbol — adds @model to signature/body
    # ------------------------------------------------------------------

    def _make_module_symbol(
        self,
        source: str,
        file_id: int,
        symbols: list[Symbol],
        import_edges: list[ImportEdge],
        type_edges: list[TypeEdge],
    ) -> int:
        page_match = _PAGE_RE.search(source)
        is_razor_page = page_match is not None
        route = (page_match.group(1) or "") if page_match else ""

        model_match = _MODEL_RE.search(source)
        model_type = model_match.group(1).strip() if model_match else ""

        ns_match = _NAMESPACE_RE.search(source)
        namespace = ns_match.group(1) if ns_match else ""

        decorators: list[str] = [m.group(1) for m in _ATTRIBUTE_RE.finditer(source)]

        sig_parts = ["@razor-page" if is_razor_page else "@view"]
        if model_type:
            sig_parts.append(f"@model {model_type}")
        if route:
            sig_parts.append(f'@page "{route}"')
        signature = "  ".join(sig_parts)

        body_lines: list[str] = []
        if model_type:
            body_lines.append(f"@model {model_type}")
        if route:
            body_lines.append(f'@page "{route}"')
        if namespace:
            body_lines.append(f"@namespace {namespace}")
        for m in _USING_RE.finditer(source):
            body_lines.append(f"@using {m.group(1)}")
        for m in _INJECT_RE.finditer(source):
            body_lines.append(f"@inject {m.group(1).strip()} {m.group(2).strip()}")
        for dec in decorators:
            body_lines.append(f"@attribute {dec}")
        body = "\n".join(body_lines) if body_lines else signature

        component_local_idx = len(symbols)
        symbols.append(Symbol(
            file_id=file_id,
            name="<view>",
            qualified_name="<view>",
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

        return component_local_idx
