"""MSBuild project file parser (.csproj, .fsproj, .vbproj).

Uses defusedxml.ElementTree (XXE/billion-laughs safe) — no tree-sitter needed.

Extracts:
  - PROJECT MODULE symbol — project name (from filename via qualified_name) +
    TargetFramework(s) + OutputType in signature/body
  - <PackageReference Include="Pkg" Version="x.y"> → CONSTANT symbol + ImportEdge
    (makes "what NuGet packages does this project use?" queries answerable)
  - <ProjectReference Include="path/Other.csproj"> → ImportEdge
    (cross-project dependency graph)
  - <Sdk Name="..."> or Sdk="..." attribute → recorded in MODULE body
  - <RootNamespace> / <AssemblyName> → recorded in MODULE body

Design:
  .csproj is standard XML. We strip the MSBuild XML namespace so ElementTree
  can navigate with simple tag names. Malformed XML falls back to a minimal
  MODULE symbol so the file is still indexed.

Symbol kinds used:
  MODULE   — the project itself (one per file)
  CONSTANT — each NuGet PackageReference (name + version = a constant fact)
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as _stdlib_ET

import defusedxml
import defusedxml.ElementTree as ET  # XXE / billion-laughs safe replacement

from trelix.core.models import ImportEdge, Symbol, SymbolKind
from trelix.indexing.parser.base import BaseParser, ParseResult

# Strip MSBuild XML namespace so we can match tag names without prefixes
_NS_RE = re.compile(r'\s+xmlns(?::[^=]*)?\s*=\s*"[^"]*"')


class CsprojParser(BaseParser):
    """
    Parser for MSBuild project files (.csproj / .fsproj / .vbproj).
    Extracts NuGet package references, project references, and project metadata.
    """

    @property
    def language_name(self) -> str:
        return "csproj"

    def parse(self, source: str, file_id: int) -> ParseResult:
        symbols: list[Symbol] = []
        import_edges: list[ImportEdge] = []

        cleaned = _NS_RE.sub("", source)

        try:
            root = ET.fromstring(cleaned)
        except (_stdlib_ET.ParseError, defusedxml.DefusedXmlException):
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name="<project>",
                    qualified_name="<project>",
                    kind=SymbolKind.MODULE,
                    line_start=1,
                    line_end=source.count("\n") + 1,
                    signature="<project> (parse error)",
                    body=source[:500],
                    is_public=True,
                )
            )
            return ParseResult(
                symbols=symbols,
                call_edges=[],
                import_edges=import_edges,
                parse_errors=1,
            )

        # ------------------------------------------------------------------
        # Collect project-level metadata
        # ------------------------------------------------------------------
        sdk = root.get("Sdk", "")
        target_fw = (
            self._find_text(root, "TargetFramework")
            or self._find_text(root, "TargetFrameworks")
            or ""
        )
        output_type = self._find_text(root, "OutputType") or ""
        root_ns = self._find_text(root, "RootNamespace") or ""
        asm_name = self._find_text(root, "AssemblyName") or ""
        nullable = self._find_text(root, "Nullable") or ""
        lang_ver = self._find_text(root, "LangVersion") or ""

        body_parts: list[str] = []
        if sdk:
            body_parts.append(f"Sdk: {sdk}")
        if target_fw:
            body_parts.append(f"TargetFramework: {target_fw}")
        if output_type:
            body_parts.append(f"OutputType: {output_type}")
        if root_ns:
            body_parts.append(f"RootNamespace: {root_ns}")
        if asm_name:
            body_parts.append(f"AssemblyName: {asm_name}")
        if nullable:
            body_parts.append(f"Nullable: {nullable}")
        if lang_ver:
            body_parts.append(f"LangVersion: {lang_ver}")

        sig_parts = ["<project>"]
        if target_fw:
            sig_parts.append(target_fw)
        if output_type:
            sig_parts.append(output_type)
        signature = "  ".join(sig_parts)

        symbols.append(
            Symbol(
                file_id=file_id,
                name="<project>",
                qualified_name="<project>",
                kind=SymbolKind.MODULE,
                line_start=1,
                line_end=source.count("\n") + 1,
                signature=signature,
                body="\n".join(body_parts) if body_parts else signature,
                is_public=True,
            )
        )
        project_local_idx = 0

        # ------------------------------------------------------------------
        # PackageReference → CONSTANT symbol + ImportEdge
        # ------------------------------------------------------------------
        for pkg in root.iter("PackageReference"):
            name = pkg.get("Include") or pkg.get("Update") or ""
            if not name:
                continue
            version = pkg.get("Version") or self._find_text(pkg, "Version") or ""
            line_no = self._approx_line(source, name)
            body = f"{name} {version}".strip()
            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.CONSTANT,
                    line_start=line_no,
                    line_end=line_no,
                    signature=f'<PackageReference Include="{name}" Version="{version}" />',
                    body=body,
                    parent_id=project_local_idx,
                    is_public=True,
                )
            )
            import_edges.append(
                ImportEdge(
                    file_id=file_id,
                    imported_from=name,
                    imported_names=[version] if version else [],
                )
            )

        # ------------------------------------------------------------------
        # ProjectReference → ImportEdge only
        # ------------------------------------------------------------------
        for proj_ref in root.iter("ProjectReference"):
            path = proj_ref.get("Include") or ""
            if not path:
                continue
            normalised = path.replace("\\", "/").lstrip("./")
            import_edges.append(
                ImportEdge(
                    file_id=file_id,
                    imported_from=normalised,
                    imported_names=[],
                )
            )

        return ParseResult(
            symbols=symbols,
            call_edges=[],
            import_edges=import_edges,
            parse_errors=0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_text(self, root: _stdlib_ET.Element, tag: str) -> str | None:
        """Find first element with given tag anywhere in the tree, return its text."""
        el = root.find(f".//{tag}")
        if el is not None and el.text:
            return el.text.strip()
        return None

    def _approx_line(self, source: str, name: str) -> int:
        """Approximate line number for a package name in the source text."""
        idx = source.find(name)
        if idx == -1:
            return 1
        return source[:idx].count("\n") + 1
