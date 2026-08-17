"""LineWindowParser: makes a file retrievable when nothing can parse its structure.

Chunks in this index hang off `symbol_id`, so a file that yields no symbols yields no
chunks and is unreachable by every retrieval leg — vector, BM25, grep, summary and
sub-chunk alike. It is in `files`, it is not in the index in any useful sense.

Two populations end up there:

* Artifacts with no extractor. Shell scripts, Dockerfiles and Makefiles have tree-sitter
  grammars available but no structural extractor, and writing five of those is 200-500
  lines each — while one fallback makes all of them reachable at once.
* Files whose extractor legitimately finds nothing: Go-templated helm manifests the YAML
  extractor cannot parse, `.mjs` build configs, and a few test files. This population is
  a point-in-time count that moves with the tree, not a constant — when the fallback was
  written it was 12 non-empty files on this repository totalling 17 KB; re-run against
  the v3.1.2 self-index it is 11 files / 10 KB. Measure it, do not quote it:

      sqlite3 .trelix/index.db "SELECT count(*), sum(size_bytes) FROM files
        WHERE id NOT IN (SELECT file_id FROM symbols) AND size_bytes > 0"

  A nonzero result after this fallback shipped is expected rather than a regression:
  both `trelix index` and `trelix update-index` skip an unchanged file on its content
  hash, so a file indexed before the fallback existed keeps its zero-symbol row until a
  forced re-parse (`TRELIX_INCREMENTAL=false`).

The fallback emits fixed line windows as SECTION symbols. Windows rather than one symbol
per file because `Chunker` TRUNCATES a chunk that exceeds its token budget rather than
splitting it, so a single 200-line symbol would be silently cut off at the budget and the
tail would be as unreachable as before.

This is deliberately not a parser. It extracts no calls, no imports and no names — only
retrievable text with honest line numbers. A structural extractor for any of these
languages is strictly better and should replace it for that language when written.
"""

from __future__ import annotations

from trelix.core.models import Symbol, SymbolKind
from trelix.indexing.parser.base import BaseParser, ParseResult

# Roughly four characters per token, and a window should fit inside the chunker's budget
# with room for the header it prepends. 60 lines of shell or YAML lands well under a
# 512-token chunk in practice; the ceiling below is what actually bounds it.
_DEFAULT_WINDOW_LINES = 60
_MAX_WINDOW_CHARS = 1_600


class LineWindowParser(BaseParser):
    """Emit fixed line windows as SECTION symbols so a file becomes retrievable."""

    def __init__(
        self,
        window_lines: int = _DEFAULT_WINDOW_LINES,
        max_window_chars: int = _MAX_WINDOW_CHARS,
    ) -> None:
        self._window_lines = max(1, window_lines)
        self._max_window_chars = max(80, max_window_chars)

    @property
    def language_name(self) -> str:
        """No tree-sitter grammar is loaded — this parser reads no syntax at all.

        The name is reported so a caller inspecting a parser can tell that the file's
        symbols came from a line split rather than a real parse.
        """
        return "line-window"

    def parse(self, source: str, file_id: int) -> ParseResult:
        """Split `source` into windows, one SECTION symbol each.

        Blank-only windows are dropped: they would cost a chunk and an embedding each
        while matching nothing. A file that is entirely whitespace therefore still yields
        no symbols, which is correct — there is nothing in it to retrieve.
        """
        symbols: list[Symbol] = []
        lines = source.splitlines()
        if not lines:
            return ParseResult(symbols=[], call_edges=[], import_edges=[], parse_errors=0)

        window_number = 0
        start = 0
        while start < len(lines):
            end = min(start + self._window_lines, len(lines))
            body = "\n".join(lines[start:end])

            # A long-line file (minified JSON, a single-line script) can blow the char
            # ceiling well inside the line budget. Shrink the window rather than emit a
            # body the chunker would truncate.
            while len(body) > self._max_window_chars and end - start > 1:
                end -= 1
                body = "\n".join(lines[start:end])

            if body.strip():
                window_number += 1
                name = f"section_{window_number}"
                symbols.append(
                    Symbol(
                        file_id=file_id,
                        name=name,
                        qualified_name=name,
                        kind=SymbolKind.SECTION,
                        # 1-indexed and inclusive, matching every other extractor, so a
                        # citation points where a reader would look.
                        line_start=start + 1,
                        line_end=end,
                        signature=lines[start].strip()[:200],
                        body=body,
                    )
                )
            start = end

        return ParseResult(symbols=symbols, call_edges=[], import_edges=[], parse_errors=0)
