"""Exact-boundary tests for the chunker token budget.

Both Chunker.build_chunks and ContextualChunker.build_chunks gate truncation on

    if token_count > self.config.max_tokens_per_chunk:

Flipping that ``>`` to ``>=`` truncates a chunk that exactly fills the budget --
silent content loss on every at-the-limit chunk -- and the pre-existing suite of
84 chunker tests does not notice, because none of them builds a chunk whose token
count is exactly max_tokens_per_chunk.

Every test below states the mutation it must fail under.

The token counter used for the assertions is tiktoken's ``cl100k_base``
encoding obtained here, independently of the module under test, so the
expected numbers are not taken from the code being tested.
"""

from __future__ import annotations

import pytest
import tiktoken

from trelix.core.config import ChunkerConfig
from trelix.core.models import Chunk, Symbol, SymbolKind
from trelix.indexing.chunker import Chunker, ContextualChunker

# The literal suffix _truncate_chunk appends. Hard-coded here on purpose: it is
# the observable marker of "this chunk was truncated", and importing it from the
# module under test would make the assertion vacuous.
_TRUNCATION_MARKER = "# ... (truncated)"

# 7 = len(cl100k_base.encode("\n# ... (truncated)")), pinned below.
_TRUNCATION_MARKER_TOKENS = 7

_ENCODING = tiktoken.get_encoding("cl100k_base")

# "zz" encodes to exactly one cl100k_base token, and " zz" also encodes to
# exactly one, so a space-joined run of N of them is exactly N tokens.
_FILLER = "zz"

_REL_PATH = "src/pkg/mod.py"
_LANGUAGE = "python"

# A budget small enough to build exactly, large enough to hold the header.
_LIMIT = 64

# Non-truncating budget used only to MEASURE a candidate body's token count.
# Nothing at this size can reach the boundary, so the measurement is identical
# under both `>` and `>=`.
_NO_TRUNCATION_LIMIT = 100_000


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
    chunks = chunker.build_chunks([_symbol(body)], [], _REL_PATH, _LANGUAGE)
    assert len(chunks) == 1
    return chunks[0]


def _filler_for_exact_total(chunker: Chunker, target_total: int) -> str:
    """Return a body whose chunk_text is EXACTLY `target_total` tokens.

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


class TestTruncationMarkerCost:
    def test_truncation_suffix_is_seven_tokens(self) -> None:
        """Pins the constant the +7 accounting bug below depends on.

        Fails under: changing the appended suffix in Chunker._truncate_chunk
        without updating _TRUNCATION_MARKER_TOKENS.
        """
        assert _count("\n" + _TRUNCATION_MARKER) == _TRUNCATION_MARKER_TOKENS


class TestChunkerTokenBudgetBoundary:
    """Chunker.build_chunks: exactly at / one below / one above the budget."""

    def test_exactly_at_budget_is_not_truncated(self) -> None:
        """Fails under: chunker.py Chunker.build_chunks `token_count >
        self.config.max_tokens_per_chunk` -> `token_count >=
        self.config.max_tokens_per_chunk`.
        """
        measurer = Chunker(ChunkerConfig(max_tokens_per_chunk=_NO_TRUNCATION_LIMIT))
        body = _filler_for_exact_total(measurer, _LIMIT)

        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=_LIMIT))
        chunk = _build_one(chunker, body)

        assert _TRUNCATION_MARKER not in chunk.chunk_text
        assert _count(chunk.chunk_text) == 64
        assert chunk.token_count == 64

    def test_one_token_below_budget_is_not_truncated(self) -> None:
        """Fails under: `>` -> `<` or `>` -> `!=` in Chunker.build_chunks."""
        measurer = Chunker(ChunkerConfig(max_tokens_per_chunk=_NO_TRUNCATION_LIMIT))
        body = _filler_for_exact_total(measurer, _LIMIT - 1)

        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=_LIMIT))
        chunk = _build_one(chunker, body)

        assert _TRUNCATION_MARKER not in chunk.chunk_text
        assert _count(chunk.chunk_text) == 63
        assert chunk.token_count == 63

    def test_one_token_above_budget_is_truncated(self) -> None:
        """Fails under: `>` -> `<`, or deleting the truncation branch in
        Chunker.build_chunks.
        """
        measurer = Chunker(ChunkerConfig(max_tokens_per_chunk=_NO_TRUNCATION_LIMIT))
        body = _filler_for_exact_total(measurer, _LIMIT + 1)

        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=_LIMIT))
        chunk = _build_one(chunker, body)

        assert _TRUNCATION_MARKER in chunk.chunk_text
        assert chunk.token_count == 64
        # DEFECT (documented, not asserted as correct): the recorded
        # token_count is the budget, while the text actually carries the
        # 7-token truncation suffix on top of the 64 kept tokens.
        assert _count(chunk.chunk_text) == 64 + _TRUNCATION_MARKER_TOKENS


class TestTruncatedChunkTokenCountAccounting:
    """Second, independent defect: the recorded token_count of a truncated
    chunk excludes the truncation suffix, so every over-budget chunk is sent to
    the embedder carrying more tokens than the row claims.

    Chunker._truncate_chunk returns decode(tokens[:max_tokens]) + the suffix,
    while build_chunks sets token_count = max_tokens. Measured on this tree with
    max_tokens_per_chunk=512 and a 400-line padded body: token_count == 512 but
    tiktoken counts 518 tokens in chunk_text (delta 6 -- the 7-token suffix,
    less one token that merges with the trailing kept token; with a filler body
    that does not merge, the delta is the full 7, as pinned above).
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known defect: token_count of a truncated chunk omits the "
            "truncation suffix. Measured 512 recorded vs 518 actual at "
            "max_tokens_per_chunk=512."
        ),
    )
    def test_truncated_chunk_token_count_matches_its_text(self) -> None:
        """Fails today (xfail strict). Once _truncate_chunk reserves room for
        its suffix, or build_chunks recounts after truncating, this passes and
        strict xfail turns the stale expectation into a failure.
        """
        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=512))
        body = "def f():\n" + "    # pad pad pad\n" * 400
        chunk = _build_one(chunker, body)

        # Precondition: truncation actually happened, so this is not a
        # vacuously-true assertion about a short chunk. Load-bearing -- keep it.
        assert _TRUNCATION_MARKER in chunk.chunk_text

        # NOTE: deliberately NO `assert chunk.token_count == 512` here.
        #
        # This file names two acceptable fixes for the defect: reserve room for the
        # suffix before truncating (giving 512 == 512), or recount after truncating
        # (giving 518 == 518). Pinning the CURRENT wrong value of 512 would make the
        # second fix fail on that line FIRST, so the test would keep XFAILing and
        # `strict=True` would never convert to a failure -- the stale expectation would
        # rot silently, which is the exact thing strict xfail exists to prevent.
        #
        # With only the equality below, both candidate fixes XPASS and strict turns that
        # into a build failure that says "remove this marker".
        assert chunk.token_count == _count(chunk.chunk_text)


class TestContextualChunkerTokenBudgetBoundary:
    """ContextualChunker.build_chunks has its own copy of the comparison."""

    def test_exactly_at_budget_is_not_truncated(self) -> None:
        """Fails under: chunker.py ContextualChunker.build_chunks `token_count >
        self.config.max_tokens_per_chunk` -> `token_count >=
        self.config.max_tokens_per_chunk`.
        """
        measurer, _ = _contextual_chunker(_NO_TRUNCATION_LIMIT)
        body = _filler_for_exact_total(measurer, _LIMIT)

        chunker, client = _contextual_chunker(_LIMIT)
        chunk = _build_one(chunker, body)

        # Precondition: the contextual path really ran (summary prepended),
        # otherwise this would silently re-test the base Chunker.
        assert client.completions.call_count == 1
        assert chunk.chunk_text.startswith(_SUMMARY + "\n\n")

        assert _TRUNCATION_MARKER not in chunk.chunk_text
        assert _count(chunk.chunk_text) == 64
        assert chunk.token_count == 64

    def test_one_token_below_budget_is_not_truncated(self) -> None:
        """Fails under: `>` -> `<` or `>` -> `!=` in
        ContextualChunker.build_chunks.
        """
        measurer, _ = _contextual_chunker(_NO_TRUNCATION_LIMIT)
        body = _filler_for_exact_total(measurer, _LIMIT - 1)

        chunker, client = _contextual_chunker(_LIMIT)
        chunk = _build_one(chunker, body)

        assert client.completions.call_count == 1
        assert chunk.chunk_text.startswith(_SUMMARY + "\n\n")
        assert _TRUNCATION_MARKER not in chunk.chunk_text
        assert _count(chunk.chunk_text) == 63
        assert chunk.token_count == 63

    def test_one_token_above_budget_is_truncated(self) -> None:
        """Fails under: `>` -> `<`, or deleting the truncation branch in
        ContextualChunker.build_chunks.
        """
        measurer, _ = _contextual_chunker(_NO_TRUNCATION_LIMIT)
        body = _filler_for_exact_total(measurer, _LIMIT + 1)

        chunker, client = _contextual_chunker(_LIMIT)
        chunk = _build_one(chunker, body)

        assert client.completions.call_count == 1
        assert _TRUNCATION_MARKER in chunk.chunk_text
        assert chunk.token_count == 64
        assert _count(chunk.chunk_text) == 64 + _TRUNCATION_MARKER_TOKENS
