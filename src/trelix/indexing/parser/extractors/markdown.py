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
        # Parent chain as a stack of (level, name) with strictly rising levels. Popping the
        # entries at or below the heading's level leaves the nearest earlier heading of each
        # lower level, in one pass (walking back through all earlier headings is quadratic).
        ancestors: list[tuple[int, str]] = []
        for idx, (line_0, level, name) in enumerate(headings):
            # Section ends at line before next same-or-higher heading
            line_end_0 = n_lines - 1
            for j in range(idx + 1, len(headings)):
                next_line_0, next_level, _ = headings[j]
                if next_level <= level:
                    line_end_0 = next_line_0 - 1
                    break

            # Qualified name: the parent headings, then this one
            while ancestors and ancestors[-1][0] >= level:
                ancestors.pop()
            qualified_name = " > ".join([parent for _, parent in ancestors] + [name])
            ancestors.append((level, name))
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
    text = _strip_md_links(text)
    return text.strip()


def _strip_md_links(text: str) -> str:
    """
    Replace each inline link `[label](url)` with its label.

    A `[` is a link when the first `]` after it holds at least one character, is followed by
    `(`, and the first `)` after that has at least one character before it. A regex for this
    retries from every `[` of a long run of them (quadratic). Here a `[` that is not a link is
    skipped together with every `[` before the same `]`, because they fail the same way, and the
    scan ends when the text has no `]` or no `)` left.
    """
    parts: list[str] = []
    copied_to = 0
    open_at = text.find("[")
    while open_at != -1:
        close_at = text.find("]", open_at + 1)
        if close_at == -1:
            break
        if close_at > open_at + 1 and text.startswith("(", close_at + 1):
            url_end = text.find(")", close_at + 2)
            if url_end == -1:
                break
            if url_end > close_at + 2:
                parts.append(text[copied_to:open_at])
                parts.append(text[open_at + 1 : close_at])
                copied_to = url_end + 1
                open_at = text.find("[", copied_to)
                continue
        open_at = text.find("[", close_at + 1)
    parts.append(text[copied_to:])
    return "".join(parts)


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
