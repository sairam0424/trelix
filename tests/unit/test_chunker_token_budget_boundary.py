"""Exact-boundary tests for the chunker token budget.

Both Chunker.build_chunks and ContextualChunker.build_chunks gate splitting on

    if token_count > self.config.max_tokens_per_chunk:

Flipping that ``>`` to ``>=`` splits a chunk that exactly fills the budget into
an unnecessary extra piece -- wasteful (an extra row, an extra "# Split:"
marker) but not, since the fix in this file's sibling
tests/unit/test_chunker_split_on_overflow.py, content-lossy the way the old
truncate-and-discard behavior was. The pre-existing suite of 84 chunker tests
still would not notice this mutation on its own, because none of them builds a
chunk whose token count is exactly max_tokens_per_chunk.

Every test below states the mutation it must fail under.

The token counter used for the assertions is tiktoken's ``cl100k_base``
encoding obtained here, independently of the module under test, so the
expected numbers are not taken from the code being tested.
"""

from __future__ import annotations

import re

import tiktoken

from trelix.core.config import ChunkerConfig
from trelix.core.models import Chunk, Symbol, SymbolKind
from trelix.indexing.chunker import Chunker, ContextualChunker

# The literal marker _split_chunk_text inserts into every split piece. Hard-
# coded here on purpose: it is the observable sign that a chunk was split, and
# importing it from the module under test would make the assertion vacuous.
_SPLIT_MARKER_RE = re.compile(r"# Split: chunk \d+ of \d+\n\n")

# The literal suffix the OLD `_truncate_chunk` used to append. That method no
# longer exists; every assertion below that this string is absent is a
# regression guard against reintroducing truncate-and-discard.
_TRUNCATION_MARKER = "# ... (truncated)"

_ENCODING = tiktoken.get_encoding("cl100k_base")

# "zz" encodes to exactly one cl100k_base token, and " zz" also encodes to
# exactly one, so a space-joined run of N of them is exactly N tokens.
_FILLER = "zz"

_REL_PATH = "src/pkg/mod.py"
_LANGUAGE = "python"

# A budget small enough to build exactly, large enough to hold the header.
_LIMIT = 64

# Non-splitting budget used only to MEASURE a candidate body's token count.
# Nothing at this size can reach the boundary, so the measurement is identical
# under both `>` and `>=`.
_NO_SPLIT_LIMIT = 100_000


def _count(text: str) -> int:
    """Token count via tiktoken directly -- never via the chunker."""
    return len(_ENCODING.encode(text))


def _filler_body(token_count: int) -> str:
    """Body consisting of exactly `token_count` one-token words."""
    return " ".join([_FILLER] * token_count)


def _symbol(body: str, *, id: int = 1) -> Symbol:
    return Symbol(
        id=id,
        file_id=1,
        name="my_func",
        qualified_name="my_func",
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=1,
        signature="def my_func()",
        body=body,
        docstring=None,
        parent_id=None,
    )


def _build_one(chunker: Chunker, body: str) -> Chunk:
    """For bodies that must NOT split -- asserts exactly one chunk comes back."""
    chunks = chunker.build_chunks([_symbol(body)], [], _REL_PATH, _LANGUAGE)
    assert len(chunks) == 1
    return chunks[0]


def _build_all(chunker: Chunker, body: str) -> list[Chunk]:
    """For bodies that MAY split -- no length assertion."""
    return chunker.build_chunks([_symbol(body)], [], _REL_PATH, _LANGUAGE)


def _recombine_body(chunks: list[Chunk]) -> str:
    """Strip each chunk's header+marker prefix and concatenate the remainder.

    Fails loudly (via the regex split's own length check) if a chunk is
    missing its "# Split: chunk i of n" marker, rather than silently
    returning a wrong reconstruction.
    """
    pieces: list[str] = []
    for chunk in chunks:
        parts = _SPLIT_MARKER_RE.split(chunk.chunk_text, maxsplit=1)
        assert len(parts) == 2, (
            f"expected exactly one split marker in chunk_text, found "
            f"{len(parts) - 1}: {chunk.chunk_text!r}"
        )
        pieces.append(parts[1])
    return "".join(pieces)


def _filler_for_exact_total(chunker: Chunker, target_total: int) -> str:
    """Return a body whose (unsplit) chunk_text is EXACTLY `target_total` tokens.

    The header is fixed-size, so one filler token adds one chunk token. The
    result is verified with tiktoken (not with the chunker's own count) and the
    caller must treat a mismatch as a broken fixture, not a passing test.
    """
    overhead = _count(_build_one(chunker, "").chunk_text)
    assert target_total > overhead, (
        f"budget {target_total} is smaller than the {overhead}-token header; pick a larger _LIMIT"
    )
    body = _filler_body(target_total - overhead)
    produced = _count(_build_one(chunker, body).chunk_text)
    assert produced == target_total, (
        "FIXTURE NO LONGER DISCRIMINATES: asked for a chunk of exactly "
        f"{target_total} tokens, built one of {produced} "
        f"(header overhead measured as {overhead}). The at-the-limit case is "
        "the only one that separates `>` from `>=`, so an off-by-one fixture "
        "would make these tests pass under the mutation."
    )
    return body


class _StubCompletions:
    """Hand-written stand-in for openai's `client.chat.completions`.

    Deliberately NOT a MagicMock: a mock answers to any attribute, so it cannot
    be wrong about the interface it is imitating.
    """

    def __init__(self, summary: str) -> None:
        self._summary = summary
        self.call_count = 0

    def create(self, **kwargs: object) -> _StubResponse:
        self.call_count += 1
        return _StubResponse(self._summary)


class _StubChat:
    def __init__(self, completions: _StubCompletions) -> None:
        self.completions = completions


class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = _StubMessage(content)


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_StubChoice(content)]


class _StubLLMClient:
    def __init__(self, summary: str) -> None:
        self.completions = _StubCompletions(summary)
        self.chat = _StubChat(self.completions)


_SUMMARY = "Adds two integers and returns the sum."


def _contextual_chunker(max_tokens: int) -> tuple[ContextualChunker, _StubLLMClient]:
    client = _StubLLMClient(_SUMMARY)
    chunker = ContextualChunker(
        ChunkerConfig(max_tokens_per_chunk=max_tokens, contextual=True),
        llm_client=client,
    )
    return chunker, client


class TestChunkerTokenBudgetBoundary:
    """Chunker.build_chunks: exactly at / one below / one above the budget."""

    def test_exactly_at_budget_is_not_split(self) -> None:
        """Fails under: chunker.py Chunker.build_chunks `token_count >
        self.config.max_tokens_per_chunk` -> `token_count >=
        self.config.max_tokens_per_chunk`.
        """
        measurer = Chunker(ChunkerConfig(max_tokens_per_chunk=_NO_SPLIT_LIMIT))
        body = _filler_for_exact_total(measurer, _LIMIT)

        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=_LIMIT))
        chunk = _build_one(chunker, body)

        assert "# Split:" not in chunk.chunk_text
        assert _TRUNCATION_MARKER not in chunk.chunk_text
        assert _count(chunk.chunk_text) == 64
        assert chunk.token_count == 64

    def test_one_token_below_budget_is_not_split(self) -> None:
        """Fails under: `>` -> `<` or `>` -> `!=` in Chunker.build_chunks."""
        measurer = Chunker(ChunkerConfig(max_tokens_per_chunk=_NO_SPLIT_LIMIT))
        body = _filler_for_exact_total(measurer, _LIMIT - 1)

        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=_LIMIT))
        chunk = _build_one(chunker, body)

        assert "# Split:" not in chunk.chunk_text
        assert _TRUNCATION_MARKER not in chunk.chunk_text
        assert _count(chunk.chunk_text) == 63
        assert chunk.token_count == 63

    def test_one_token_above_budget_is_split_into_two_chunks(self) -> None:
        """Fails under: `>` -> `<`, or deleting the split branch in
        Chunker.build_chunks, or reverting `_split_chunk_text` to the old
        truncate-and-discard `_truncate_chunk`.
        """
        measurer = Chunker(ChunkerConfig(max_tokens_per_chunk=_NO_SPLIT_LIMIT))
        body = _filler_for_exact_total(measurer, _LIMIT + 1)

        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=_LIMIT))
        chunks = _build_all(chunker, body)

        assert len(chunks) == 2
        assert all(_TRUNCATION_MARKER not in c.chunk_text for c in chunks)
        assert all(c.chunk_text.startswith(f"# File: {_REL_PATH}") for c in chunks)
        # No content lost: stitching the two pieces back together reproduces
        # the exact 1-token-over-budget body this fixture built.
        assert _recombine_body(chunks) == body

    def test_split_pieces_token_counts_match_their_own_text(self) -> None:
        """token_count is recounted per piece from its OWN final text (header
        + split marker + body slice), not derived from the pre-split budget.
        """
        measurer = Chunker(ChunkerConfig(max_tokens_per_chunk=_NO_SPLIT_LIMIT))
        body = _filler_for_exact_total(measurer, _LIMIT + 1)

        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=_LIMIT))
        chunks = _build_all(chunker, body)

        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk.token_count == _count(chunk.chunk_text)


class TestContextualChunkerTokenBudgetBoundary:
    """ContextualChunker.build_chunks has its own copy of the comparison."""

    def test_exactly_at_budget_is_not_split(self) -> None:
        """Fails under: chunker.py ContextualChunker.build_chunks `token_count >
        self.config.max_tokens_per_chunk` -> `token_count >=
        self.config.max_tokens_per_chunk`.
        """
        measurer, _ = _contextual_chunker(_NO_SPLIT_LIMIT)
        body = _filler_for_exact_total(measurer, _LIMIT)

        chunker, client = _contextual_chunker(_LIMIT)
        chunk = _build_one(chunker, body)

        # Precondition: the contextual path really ran (summary prepended),
        # otherwise this would silently re-test the base Chunker.
        assert client.completions.call_count == 1
        assert chunk.chunk_text.startswith(_SUMMARY + "\n\n")

        assert "# Split:" not in chunk.chunk_text
        assert _TRUNCATION_MARKER not in chunk.chunk_text
        assert _count(chunk.chunk_text) == 64
        assert chunk.token_count == 64

    def test_one_token_below_budget_is_not_split(self) -> None:
        """Fails under: `>` -> `<` or `>` -> `!=` in
        ContextualChunker.build_chunks.
        """
        measurer, _ = _contextual_chunker(_NO_SPLIT_LIMIT)
        body = _filler_for_exact_total(measurer, _LIMIT - 1)

        chunker, client = _contextual_chunker(_LIMIT)
        chunk = _build_one(chunker, body)

        assert client.completions.call_count == 1
        assert chunk.chunk_text.startswith(_SUMMARY + "\n\n")
        assert "# Split:" not in chunk.chunk_text
        assert _TRUNCATION_MARKER not in chunk.chunk_text
        assert _count(chunk.chunk_text) == 63
        assert chunk.token_count == 63

    def test_one_token_above_budget_is_split_into_two_chunks(self) -> None:
        """Fails under: `>` -> `<`, or deleting the split branch in
        ContextualChunker.build_chunks.
        """
        measurer, _ = _contextual_chunker(_NO_SPLIT_LIMIT)
        body = _filler_for_exact_total(measurer, _LIMIT + 1)

        chunker, client = _contextual_chunker(_LIMIT)
        chunks = _build_all(chunker, body)

        assert client.completions.call_count == 1
        assert len(chunks) == 2
        assert all(_TRUNCATION_MARKER not in c.chunk_text for c in chunks)
        # The summary describes the whole symbol once -- it belongs on the
        # first piece only, never duplicated onto the second.
        assert chunks[0].chunk_text.startswith(_SUMMARY + "\n\n")
        assert _SUMMARY not in chunks[1].chunk_text
        # No content lost, contextual summary aside: stitching the two
        # pieces' bodies back together reproduces the original body exactly.
        assert _recombine_body(chunks) == body

    def test_split_pieces_token_counts_match_their_own_text(self) -> None:
        measurer, _ = _contextual_chunker(_NO_SPLIT_LIMIT)
        body = _filler_for_exact_total(measurer, _LIMIT + 1)

        chunker, _client = _contextual_chunker(_LIMIT)
        chunks = _build_all(chunker, body)

        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk.token_count == _count(chunk.chunk_text)
