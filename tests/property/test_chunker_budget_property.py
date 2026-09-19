"""Property: Chunker's token-budget boundary holds across ARBITRARY budgets and
ARBITRARY header content, not just the one (budget=64, fixed rel_path/language)
case hand-written in tests/unit/test_chunker_token_budget_boundary.py.

That file already exhaustively covers budget=64 with a fixed `_REL_PATH` /
`_LANGUAGE` header. What it does NOT cover -- and what a Hypothesis property can
cheaply sweep -- is whether the SAME `>` boundary in
`Chunker.build_chunks` (chunker.py:92) holds once the context header's own token
cost varies (different `file_rel_path` / `language` strings change how many
tokens the header itself consumes before any filler body is added). A defect
that only appears when the header is unusually long or short (e.g. an
off-by-one in how the header is joined into `chunk_text`) would pass every
fixed-header hand-written case and still be wrong.

FALSIFYING INPUT CONFIRMED BY HAND (see round notes / PROOF PROTOCOL below):
mutating chunker.py:92 `token_count > self.config.max_tokens_per_chunk` to
`>=` breaks the "exactly at budget" case for EVERY (rel_path, language, budget)
triple tried by hand (budget in {20, 30, 50, 100, 200}, default rel_path), not
just budget=64. Verified: unmutated -- not split; mutated -- split.

Over-budget bodies used to be truncated (and the tail permanently discarded);
they are now split across multiple chunks instead, so the over-budget property
below asserts content is fully preserved across however many pieces the split
produces, rather than pinning an exact split count -- the number of pieces for
a given (header, budget) combination depends on header-overhead arithmetic
that is an implementation detail, not part of the contract under test.

DERANDOMIZED: both `@settings` below pin `derandomize=True`, so the example
sequence Hypothesis explores is a fixed hash of the test function rather
than a fresh `random.Random()` seed per run -- the same falsifying input (if
any exists) is found on every run, on every machine, not just on whichever
run got unlucky. `max_examples=30` is unchanged (not raised): re-running the
mutation above under the pinned seed still catches it on every one of three
consecutive runs (see round report), so 30 is already sufficient here.
"""

from __future__ import annotations

import re
import string

import tiktoken
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from trelix.core.config import ChunkerConfig
from trelix.core.models import Chunk, Symbol, SymbolKind
from trelix.indexing.chunker import Chunker

_ENCODING = tiktoken.get_encoding("cl100k_base")
_TRUNCATION_MARKER = "# ... (truncated)"
_SPLIT_MARKER_RE = re.compile(r"# Split: chunk \d+ of \d+\n\n")

# Path/language alphabet kept deliberately boring (ASCII letters, digits, and
# the punctuation that appears in real repo-relative paths) -- the property
# is about header LENGTH varying, not about exercising tiktoken's handling of
# exotic Unicode, which is a different concern from the budget boundary.
_PATH_ALPHABET = string.ascii_lowercase + string.digits + "/_.-"


def _count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _symbol(body: str) -> Symbol:
    return Symbol(
        id=1,
        file_id=1,
        name="f",
        qualified_name="f",
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=1,
        signature="def f()",
        body=body,
        docstring=None,
        parent_id=None,
    )


def _build_one(chunker: Chunker, body: str, rel_path: str, language: str):
    """For bodies that must NOT split -- asserts exactly one chunk comes back."""
    chunks = chunker.build_chunks([_symbol(body)], [], rel_path, language)
    assert len(chunks) == 1
    return chunks[0]


def _build_all(chunker: Chunker, body: str, rel_path: str, language: str) -> list[Chunk]:
    """For bodies that MAY split -- no length assertion."""
    return chunker.build_chunks([_symbol(body)], [], rel_path, language)


def _recombine_body(chunks: list[Chunk]) -> str:
    """Strip each chunk's header+marker prefix and concatenate the remainder.

    Fails loudly if a chunk is missing its "# Split: chunk i of n" marker,
    rather than silently returning a wrong reconstruction.
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


def _filler_body(token_count: int) -> str:
    """`token_count` copies of the one-cl100k-token word "zz", space-joined."""
    return " ".join(["zz"] * token_count)


def _body_for_exact_total(
    measurer: Chunker, rel_path: str, language: str, target_total: int
) -> tuple[str, int]:
    """Filler body whose chunk_text is exactly `target_total` tokens for this header.

    Returns (body, overhead) so the caller can skip totals the header alone
    already exceeds. `measurer` must use a budget large enough that nothing
    truncates while probing the header's own cost.
    """
    overhead = _count(_build_one(measurer, "", rel_path, language).chunk_text)
    if target_total <= overhead:
        return "", overhead
    body = _filler_body(target_total - overhead)
    produced = _count(_build_one(measurer, body, rel_path, language).chunk_text)
    assert produced == target_total, (
        "FIXTURE NO LONGER DISCRIMINATES: asked for a chunk of exactly "
        f"{target_total} tokens for header (rel_path={rel_path!r}, "
        f"language={language!r}), built one of {produced} (overhead={overhead}). "
    )
    return body, overhead


# Keep well under the 4.0s "slow" threshold (tests/conftest.py SLOW_FILES): each
# example builds two tiny Chunker.build_chunks() calls with no I/O, so a bound of
# 30 examples is generous headroom, not a tight squeeze. Measured locally at a
# small fraction of a second for the whole file.
@settings(
    derandomize=True,
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@example(rel_path="src/pkg/mod.py", language="python", budget=64)  # the hand-written case
@example(rel_path="a", language="x", budget=21)  # minimal header, near-minimal budget
@given(
    rel_path=st.text(alphabet=_PATH_ALPHABET, min_size=1, max_size=40),
    language=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=12),
    budget=st.integers(min_value=20, max_value=300),
)
def test_exactly_at_budget_never_splits_for_any_header(
    rel_path: str, language: str, budget: int
) -> None:
    """Fails under: chunker.py Chunker.build_chunks `token_count >
    self.config.max_tokens_per_chunk` -> `>=`, for ANY header/budget, not just
    the hand-written budget=64 case.
    """
    measurer = Chunker(ChunkerConfig(max_tokens_per_chunk=100_000))
    body, overhead = _body_for_exact_total(measurer, rel_path, language, budget)
    if body == "" and overhead >= budget:
        return  # header alone already meets/exceeds this budget; nothing to build

    chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=budget))
    chunk = _build_one(chunker, body, rel_path, language)

    assert "# Split:" not in chunk.chunk_text
    assert _TRUNCATION_MARKER not in chunk.chunk_text
    assert chunk.token_count == budget


@settings(
    derandomize=True,
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@example(rel_path="src/pkg/mod.py", language="python", budget=64)
@given(
    rel_path=st.text(alphabet=_PATH_ALPHABET, min_size=1, max_size=40),
    language=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=12),
    budget=st.integers(min_value=20, max_value=300),
)
def test_one_token_above_budget_always_splits_for_any_header(
    rel_path: str, language: str, budget: int
) -> None:
    """Fails under: `>` -> `<` in Chunker.build_chunks, or the split branch
    being deleted (or reverted to the old truncate-and-discard behavior), for
    ANY header/budget.
    """
    measurer = Chunker(ChunkerConfig(max_tokens_per_chunk=100_000))
    body, overhead = _body_for_exact_total(measurer, rel_path, language, budget + 1)
    if body == "" and overhead >= budget + 1:
        # Regression guard for a real bug found in review: the header alone
        # (e.g. a very long docstring/import list) already exceeds budget+1
        # with an empty body, so there is no body content to spread across
        # further pieces. `_split_chunk_text` used to iterate
        # `while idx < len(body_tokens):`, which never runs when body_tokens
        # is empty, so the symbol silently got ZERO chunks -- unsearchable,
        # strictly worse than the truncate-and-discard behavior this split
        # path replaced. It must always produce exactly one (over-budget)
        # chunk here instead.
        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=budget))
        chunks = _build_all(chunker, body, rel_path, language)
        assert len(chunks) == 1
        assert _TRUNCATION_MARKER not in chunks[0].chunk_text
        assert chunks[0].token_count == _count(chunks[0].chunk_text)
        # Confirms this really is the over-budget regime the fixture asked
        # for, not a vacuously-true check on a chunk that happened to fit.
        assert chunks[0].token_count > budget
        assert _recombine_body(chunks) == body
        return

    chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=budget))
    chunks = _build_all(chunker, body, rel_path, language)

    # At least one further piece is required to hold the overflow -- the
    # exact count depends on header-overhead arithmetic that varies with
    # `rel_path`/`language`, so it is not pinned to a specific number here.
    assert len(chunks) >= 2
    assert all(_TRUNCATION_MARKER not in c.chunk_text for c in chunks)
    # token_count is recounted per piece from its own final text (header +
    # split marker + body slice), not derived from the pre-split budget.
    for c in chunks:
        assert c.token_count == _count(c.chunk_text)
    # No content lost: stitching every piece's body back together in order
    # reproduces the exact 1-token-over-budget body this fixture built.
    assert _recombine_body(chunks) == body
