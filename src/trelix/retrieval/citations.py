"""
Citation tags on retrieved context: ``[C1]``, ``[C2]``, ...

Behind ``TRELIX_RETRIEVAL_CITATIONS`` (default off). When the flag is on, the
assembler prefixes every block header it renders with a tag and records what each
tag refers to as a :class:`CitationSource` on ``RetrievedContext.citation_sources``;
the synthesizer then tells the model to cite those tags. The tag number is the only
thing the model ever writes about a source — path, lines and symbol stay with the
index, so nothing the model emits is read as a path.

Tag numbering lives in ``ContextAssembler._format_context`` only: tags follow the
rendered order of the context (files in first-seen order, symbols by line within a
file), not the retrieval score order, and one symbol keeps one tag however many
blocks render it (a compressed body renders one block per kept span).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CitationSource:
    """What one ``[C<tag>]`` tag of a rendered context refers to."""

    tag: int  # 1-based, rendered order
    symbol_id: int  # the de-duplication key; GraphRAG looks the tag up by it
    path: str  # IndexedFile.rel_path
    line_start: int  # Symbol.line_start, 1-indexed inclusive
    line_end: int  # Symbol.line_end, 1-indexed inclusive
    symbol: str  # Symbol.qualified_name


def cite_tag(tag: int) -> str:
    """The prefix a tagged block header carries: ``cite_tag(3) == "[C3] "``."""
    return f"[C{tag}] "
