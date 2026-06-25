"""
MarkdownParser: splits Markdown files into heading-based sections.

No tree-sitter needed — regex + line scanning covers everything we need:

  - ATX headings (#, ##, ..., ######)
  - Setext headings  (text\n=====  and  text\n-----)
  - YAML / TOML front matter  (--- ... --- / +++ ... +++) extracted as a
    MODULE symbol whose docstring is the title/description/tags

Qualified name encodes the nesting path, e.g.:
    # Installation           → "Installation"
    ## Quick Start           → "Installation > Quick Start"
    ### With Docker          → "Installation > Quick Start > With Docker"

Files with no headings (plain prose) are indexed as a single
"Document" section so they're still retrievable.
"""

from __future__ import annotations

import re

from trelix.core.models import Symbol, SymbolKind
from trelix.indexing.parser.base import BaseParser, ParseResult

# ATX heading: # to ######
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)")

# Setext underlines (must be 2+ chars, all the same)
_SETEXT_H1_RE = re.compile(r"^=+$")
_SETEXT_H2_RE = re.compile(r"^-+$")

# Front matter delimiters
_YAML_FM_START = "---"
_TOML_FM_START = "+++"


class MarkdownParser(BaseParser):
    @property
    def language_name(self) -> str:
        return "markdown"

    def parse(self, source: str, file_id: int) -> ParseResult:
        lines = source.splitlines()
        n_lines = len(lines)

        # ── 1. Extract front matter ─────────────────────────────────────────
        fm_symbol, fm_end_line = self._extract_frontmatter(lines, file_id)
        # fm_end_line: first line index AFTER the front matter block (0-based)

        # ── 2. Find all headings (ATX + setext) ─────────────────────────────
        headings: list[tuple[int, int, str]] = []  # (line_0, level, text)
        # Track which lines are setext underlines so we skip them as heading text
        setext_underline_lines: set[int] = set()

        for i in range(fm_end_line, n_lines):
            line = lines[i]

            # ATX
            m = _HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                text = _strip_inline_md(m.group(2).strip())
                headings.append((i, level, text))
                continue

            # Setext: check if the NEXT line is an underline
            if i + 1 < n_lines:
                underline = lines[i + 1].strip()
                if underline and len(underline) >= 2 and i + 1 not in setext_underline_lines:
                    text = line.strip()
                    if text and not _HEADING_RE.match(line):
                        if _SETEXT_H1_RE.match(underline):
                            headings.append((i, 1, _strip_inline_md(text)))
                            setext_underline_lines.add(i + 1)
                        elif _SETEXT_H2_RE.match(underline):
                            headings.append((i, 2, _strip_inline_md(text)))
                            setext_underline_lines.add(i + 1)

        # ── 3. No headings → single "Document" section ──────────────────────
        content_after_fm = "\n".join(lines[fm_end_line:])
        symbols: list[Symbol] = []

        if fm_symbol:
            symbols.append(fm_symbol)

        if not headings and content_after_fm.strip():
            if not fm_symbol:
                symbols.append(
                    Symbol(
                        file_id=file_id,
                        name="Document",
                        qualified_name="Document",
                        kind=SymbolKind.SECTION,
                        line_start=1,
                        line_end=n_lines,
                        body=source,
                        signature="(document)",
                    )
                )
            return ParseResult(
                symbols=symbols,
                call_edges=[],
                import_edges=[],
                parse_errors=0,
                type_edges=[],
            )

        # ── 4. Build one symbol per heading ────────────────────────────────
        for idx, (line_0, level, name) in enumerate(headings):
            # Section ends at line before next same-or-higher heading
            line_end_0 = n_lines - 1
            for j in range(idx + 1, len(headings)):
                next_line_0, next_level, _ = headings[j]
                if next_level <= level:
                    line_end_0 = next_line_0 - 1
                    break

            # Qualified name: walk back to find parent headings
            parts = [name]
            current_level = level
            for k in range(idx - 1, -1, -1):
                _, prev_level, prev_name = headings[k]
                if prev_level < current_level:
                    parts.insert(0, prev_name)
                    current_level = prev_level
                    if current_level == 1:
                        break

            qualified_name = " > ".join(parts)
            signature = "#" * level + " " + name
            body = "\n".join(lines[line_0 : line_end_0 + 1])

            symbols.append(
                Symbol(
                    file_id=file_id,
                    name=name,
                    qualified_name=qualified_name,
                    kind=SymbolKind.SECTION,
                    line_start=line_0 + 1,  # 1-indexed
                    line_end=line_end_0 + 1,
                    body=body,
                    signature=signature,
                )
            )

        return ParseResult(
            symbols=symbols,
            call_edges=[],
            import_edges=[],
            parse_errors=0,
            type_edges=[],
        )

    # ------------------------------------------------------------------
    # Front matter
    # ------------------------------------------------------------------

    def _extract_frontmatter(self, lines: list[str], file_id: int) -> tuple[Symbol | None, int]:
        """
        Detect and parse YAML (---) or TOML (+++) front matter at start of file.
        Returns (MODULE symbol or None, first line index after front matter).
        """
        if not lines:
            return None, 0

        first = lines[0].strip()
        if first not in (_YAML_FM_START, _TOML_FM_START):
            return None, 0

        end_delimiter = first  # "---" or "+++"
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == end_delimiter:
                end_idx = i
                break

        if end_idx is None:
            return None, 0  # unclosed front matter → ignore

        fm_lines = lines[1:end_idx]
        fm_text = "\n".join(fm_lines)

        title = _fm_field(fm_text, "title")
        description = _fm_field(fm_text, "description")
        tags = _fm_list_field(fm_text, "tags") or _fm_list_field(fm_text, "keywords")

        name = title or "Document"
        parts: list[str] = []
        if description:
            parts.append(description)
        if tags:
            parts.append("tags: " + ", ".join(tags))
        docstring = "\n".join(parts) if parts else None

        sig_parts = [f"title: {title}"] if title else []
        if tags:
            sig_parts.append(f"tags: {', '.join(tags)}")
        signature = " | ".join(sig_parts) if sig_parts else "(front matter)"

        symbol = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=SymbolKind.MODULE,
            line_start=1,
            line_end=end_idx + 1,  # 1-indexed, inclusive of closing delimiter
            body=fm_text,
            signature=signature,
            docstring=docstring,
            is_public=True,
        )
        return symbol, end_idx + 1  # skip past the closing delimiter line


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _strip_inline_md(text: str) -> str:
    """Remove **bold**, *italic*, `code`, [text](url) from heading text."""
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


def _fm_field(fm_text: str, key: str) -> str | None:
    """Extract a scalar YAML/TOML front matter field: key: value (or key = value)."""
    m = re.search(
        rf'^{re.escape(key)}\s*[:=]\s*["\']?(.+?)["\']?\s*$',
        fm_text,
        re.MULTILINE | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().strip("'\"")
    return None


def _fm_list_field(fm_text: str, key: str) -> list[str]:
    """
    Extract a YAML list field, both inline and block styles:
      tags: [a, b, c]
      tags:
        - a
        - b
    """
    # Inline YAML array: tags: [a, b, c]
    m = re.search(
        rf"^{re.escape(key)}\s*:\s*\[([^\]]*)\]",
        fm_text,
        re.MULTILINE | re.IGNORECASE,
    )
    if m:
        return [t.strip().strip("'\"") for t in m.group(1).split(",") if t.strip()]

    # Block YAML list: collect "- item" lines after "key:"
    m2 = re.search(
        rf"^{re.escape(key)}\s*:",
        fm_text,
        re.MULTILINE | re.IGNORECASE,
    )
    if m2:
        rest = fm_text[m2.end() :]
        items: list[str] = []
        for line in rest.splitlines():
            li = re.match(r"^\s*-\s+(.+)", line)
            if li:
                items.append(li.group(1).strip().strip("'\""))
            elif line.strip() and not line.startswith(" "):
                break  # new top-level key
        return items

    return []
