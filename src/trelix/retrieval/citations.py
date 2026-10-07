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
"""

from __future__ import annotations

from dataclasses import dataclass
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
