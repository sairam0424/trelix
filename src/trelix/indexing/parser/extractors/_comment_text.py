"""Linear-time comment-text helpers shared by the language extractors.

The Go, Java, JavaScript, TypeScript, Rust and JSON extractors stripped a block comment's
closing ``*/`` and its leading ``*`` columns with two regular expressions, and the C#
extractor stripped the XML tags of a ``///`` doc comment with a third. All three backtrack
quadratically on a comment holding a long run of whitespace, of blank lines or of ``<``, so
one crafted source file could stall the indexer for minutes. The helpers below return exactly
what those expressions returned, in time linear in the text length.
"""

from __future__ import annotations

import re

# Whitespace at a line start, then optionally a ``*`` and one more whitespace character.
# The star group is optional, so the pattern never backtracks out of the whitespace run:
# a run that does not end in ``*`` is consumed once instead of being re-read from every
# later line start inside it (the quadratic case for a run of blank lines).
_LINE_STAR_PREFIX = re.compile(r"^\s*(\*\s?)?", re.MULTILINE)


def strip_block_comment_end(raw: str) -> str:
    """Drop a trailing ``*/`` together with the stars and whitespace before it.

    Same result as ``re.sub(r"\\s*\\*+/$", "", raw)``. That ``$`` also matches just before
    one final newline, so a final newline is set aside and put back.
    """
    body, newline = (raw[:-1], "\n") if raw.endswith("\n") else (raw, "")
    if not body.endswith("*/"):
        return raw
    return body[:-1].rstrip("*").rstrip() + newline


def strip_star_line_prefixes(raw: str) -> str:
    """Drop the whitespace, ``*`` and one following whitespace character at each line start.

    Same result as ``re.sub(r"^\\s*\\*\\s?", "", raw, flags=re.MULTILINE)``: the leading
    whitespace may span blank lines, and a line whose first non-blank character is not a
    ``*`` is left alone.
    """
    return _LINE_STAR_PREFIX.sub(lambda match: "" if match.group(1) else match.group(0), raw)


def strip_xml_tags(text: str) -> str:
    """Remove every ``<...>`` span.

    Same result as ``re.sub(r"<[^>]+>", "", text)``, which starts an attempt at each ``<``
    and scans to the next ``>`` or to the end, so a long run of ``<`` with no ``>`` after it
    costs quadratic time. A span runs from a ``<`` to the first ``>`` after it and needs at
    least one character in between, so ``<>`` stays.
    """
    kept: list[str] = []
    copied_to = 0
    search_from = 0
    while (start := text.find("<", search_from)) != -1:
        end = text.find(">", start + 1)
        if end == -1:
            break
        if end == start + 1:
            search_from = end
            continue
        kept.append(text[copied_to:start])
        copied_to = search_from = end + 1
    kept.append(text[copied_to:])
    return "".join(kept)
