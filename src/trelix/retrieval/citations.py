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

Abstention shares this module because it is the other half of the same protocol: the
instruction that asks for tags also asks the model to reply with one line starting
``INSUFFICIENT_EVIDENCE:`` when the context does not answer the question, and
:func:`is_abstention` is how every caller (CLI, REST, FLARE, eval) recognises that line.
:data:`NO_RESULTS_MESSAGE` is the notice the Synthesizer answers with, instead of calling
the model, when retrieval found nothing at all.

Verification closes the loop: :func:`verify_citations` reads the markers back out of the
model's answer and checks each one against the ``CitationSource`` it names and against the
file on disk. The only thing taken from the model's text is the digits of a ``[C<n>]``
marker, matched by :data:`MARKER_RE`; the path, lines and symbol of every
:class:`Citation` come from the index. The verifier opens at most one file per distinct
cited path, under ``repo_root / rel_path`` only, and reads it for its newline count alone.

``trelix ask`` shows the verdicts two ways, both built here so the two agree:
:func:`footer_lines` is the ``Sources:`` footer printed after a streamed answer and
:func:`citation_as_json` is one element of the ``citations`` list of ``trelix ask --json``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# The Synthesizer's answer when retrieval returned no results: there is nothing to ground an
# answer in, so no model is called. One literal, shared by the CLI, REST /ask and the eval.
NO_RESULTS_MESSAGE = "[trelix] No relevant code found — cannot synthesize an answer."

# The first line of a model answer that declines to answer (design: "reply with exactly one
# line starting with INSUFFICIENT_EVIDENCE: followed by what is missing, and nothing else").
ABSTAIN_PREFIX = "INSUFFICIENT_EVIDENCE:"

# Why the Synthesizer's last answer was an abstention: retrieval found nothing, or the model
# said the retrieved context does not contain what the question needs.
AbstainReason = Literal["no_results", "insufficient_evidence"]

# A marker as the model writes it: [C1] .. [C999]. [C0], [C007], [c1], [C 1] and [C1000]
# are text. The pattern has no notion of context, so a marker inside a code span or a
# quotation counts like any other.
MARKER_RE = re.compile(r"\[C([1-9][0-9]{0,2})\]")

# What verification found for one marker. `unknown`: no retrieved chunk carries that tag
# (the model invented it, or cited past the context). `file_missing`: the tag's path is no
# longer a file under the repository root. `line_out_of_range`: the cited chunk ends past
# the file's current line count. Both stale cases mean the index is behind the tree.
CitationStatus = Literal["valid", "file_missing", "line_out_of_range", "unknown"]

# count_lines reads in pieces of this size, so a file that has grown since it was indexed
# costs time, never memory.
_READ_CHUNK = 1 << 20


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


def is_abstention(text: str) -> bool:
    """True when ``text`` is a model answer that abstains: its first non-blank characters are
    ``INSUFFICIENT_EVIDENCE:``. Case-sensitive, and the prefix must open the answer: a
    sentence that merely mentions the marker mid-text is an answer, not an abstention."""
    return text.lstrip().startswith(ABSTAIN_PREFIX)


@dataclass(frozen=True)
class Citation:
    """One distinct ``[C<tag>]`` marker of an answer, verified.

    ``path``, ``line_start``, ``line_end`` and ``symbol`` are copied from the
    :class:`CitationSource` the tag names and are ``None`` when the status is ``unknown``;
    ``detail`` is one line for a human and ``""`` when the status is ``valid``.
    """

    marker: str  # "[C2]", as written
    tag: int
    status: CitationStatus
    path: str | None
    line_start: int | None
    line_end: int | None
    symbol: str | None
    detail: str


def find_markers(answer: str) -> list[tuple[str, int]]:
    """The distinct ``[C<n>]`` markers of ``answer`` as ``(marker, tag)`` pairs, in order of
    first appearance: ``"[C2] .. [C1] .. [C2]"`` gives ``[("[C2]", 2), ("[C1]", 1)]``."""
    seen: set[int] = set()
    found: list[tuple[str, int]] = []
    for match in MARKER_RE.finditer(answer):
        tag = int(match.group(1))
        if tag in seen:
            continue
        seen.add(tag)
        found.append((match.group(0), tag))
    return found


def count_lines(path: Path) -> int:
    """The number of lines of ``path`` as the extractors count them: newlines plus one.

    Tree-sitter's root node ends on the row after the final newline (``python.py``,
    ``go.py``, ``csharp.py`` use ``root.end_point[0] + 1``) and the XML and Razor extractors
    use ``source.count("\\n") + 1``, so a whole-file symbol of a newline-terminated 7-line
    file spans ``1-8`` and that file counts 8 here; a function or class node ends on its
    last token's row and always fits. An empty file counts 1. The file is read in 1 MiB
    pieces for its newline count only; no content is kept.
    """
    newlines = 0
    with path.open("rb") as handle:
        while piece := handle.read(_READ_CHUNK):
            newlines += piece.count(b"\n")
    return newlines + 1


def verify_citations(
    answer: str, sources: Sequence[CitationSource], repo_root: Path
) -> list[Citation]:
    """Classify every distinct marker of ``answer`` against ``sources`` and the files on disk.

    Per marker, in this order: a tag no source carries is ``unknown``; a source whose
    ``repo_root / path`` is not a file is ``file_missing``; a source whose ``line_end`` is
    past :func:`count_lines` of that file is ``line_out_of_range``; otherwise ``valid``.
    Results come back in the markers' first-appearance order, one per distinct tag. A
    file-summary source (``TRELIX_RETRIEVAL_FILE_SUMMARY_LEG``, a synthetic ``symbol_id``
    below zero with the representative symbol's range) is checked like any other: its path
    and lines are real index data. Each distinct path is counted once per call. Pure given
    the answer, the sources and the filesystem. A path that raises an ``OSError`` while it
    is checked (a permission denied on the file or a directory above it) is reported as
    ``file_missing`` with the exception's name in the detail, so a stale or unreadable tree
    never makes the caller crash after the answer has already been shown.
    """
    by_tag = {source.tag: source for source in sources}
    line_counts: dict[str, int] = {}
    return [
        _classify(marker, tag, by_tag.get(tag), repo_root, line_counts)
        for marker, tag in find_markers(answer)
    ]


def _classify(
    marker: str,
    tag: int,
    source: CitationSource | None,
    repo_root: Path,
    line_counts: dict[str, int],
) -> Citation:
    if source is None:
        return Citation(
            marker, tag, "unknown", None, None, None, None, "no retrieved chunk has this tag"
        )
    file = repo_root / source.path
    try:
        if not file.is_file():
            return _cited(
                marker, source, "file_missing", f"{source.path} is not in the repository; re-index"
            )
        if source.path not in line_counts:
            line_counts[source.path] = count_lines(file)
    except OSError as exc:
        detail = f"{source.path} could not be read ({type(exc).__name__}); re-index"
        return _cited(marker, source, "file_missing", detail)
    lines = line_counts[source.path]
    if source.line_end > lines:
        detail = (
            f"{source.path} has {lines} lines, "
            f"the cited chunk ends at line {source.line_end}; re-index"
        )
        return _cited(marker, source, "line_out_of_range", detail)
    return _cited(marker, source, "valid", "")


def _cited(marker: str, source: CitationSource, status: CitationStatus, detail: str) -> Citation:
    return Citation(
        marker,
        source.tag,
        status,
        source.path,
        source.line_start,
        source.line_end,
        source.symbol,
        detail,
    )


def footer_lines(citations: Sequence[Citation]) -> list[str]:
    """The ``Sources:`` footer ``trelix ask`` prints after an answer, one string per line.

    A valid row is ``  [C2] src/auth/middleware.py:70-80 AuthMiddleware.bearer``; any other
    status is ``  [C3] unverified (line_out_of_range): <detail>``. An answer with no markers
    gives the single line ``Sources: none cited.``. Rows keep the verifier's order.
    """
    if not citations:
        return ["Sources: none cited."]
    lines = ["Sources:"]
    for citation in citations:
        if citation.status == "valid":
            lines.append(
                f"  {citation.marker} {citation.path}:{citation.line_start}-"
                f"{citation.line_end} {citation.symbol}"
            )
        else:
            lines.append(f"  {citation.marker} unverified ({citation.status}): {citation.detail}")
    return lines


def citation_as_json(citation: Citation) -> dict[str, object]:
    """One element of ``trelix ask --json``'s ``citations`` list.

    Exactly the keys ``marker, status, path, lines, symbol, detail``; ``lines`` is the
    ``"a-b"`` string the other JSON surfaces use (``trelix search --json``) and, like
    ``path`` and ``symbol``, ``None`` for an ``unknown`` marker.
    """
    lines = (
        f"{citation.line_start}-{citation.line_end}"
        if citation.line_start is not None and citation.line_end is not None
        else None
    )
    return {
        "marker": citation.marker,
        "status": citation.status,
        "path": citation.path,
        "lines": lines,
        "symbol": citation.symbol,
        "detail": citation.detail,
    }
