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
just budget=64. Verified: unmutated -- not truncated; mutated -- truncated.
"""

from __future__ import annotations

import string

import tiktoken
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from trelix.core.config import ChunkerConfig
from trelix.core.models import Symbol, SymbolKind
from trelix.indexing.chunker import Chunker

_ENCODING = tiktoken.get_encoding("cl100k_base")
_TRUNCATION_MARKER = "# ... (truncated)"

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
    chunks = chunker.build_chunks([_symbol(body)], [], rel_path, language)
    assert len(chunks) == 1
    return chunks[0]


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
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@example(rel_path="src/pkg/mod.py", language="python", budget=64)  # the hand-written case
@example(rel_path="a", language="x", budget=21)  # minimal header, near-minimal budget
@given(
    rel_path=st.text(alphabet=_PATH_ALPHABET, min_size=1, max_size=40),
    language=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=12),
    budget=st.integers(min_value=20, max_value=300),
)
def test_exactly_at_budget_never_truncates_for_any_header(
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

    assert _TRUNCATION_MARKER not in chunk.chunk_text
    assert chunk.token_count == budget


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@example(rel_path="src/pkg/mod.py", language="python", budget=64)
@given(
    rel_path=st.text(alphabet=_PATH_ALPHABET, min_size=1, max_size=40),
    language=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=12),
    budget=st.integers(min_value=20, max_value=300),
)
def test_one_token_above_budget_always_truncates_for_any_header(
    rel_path: str, language: str, budget: int
) -> None:
    """Fails under: `>` -> `<` in Chunker.build_chunks, or the truncation branch
    being deleted, for ANY header/budget.
    """
    measurer = Chunker(ChunkerConfig(max_tokens_per_chunk=100_000))
    body, overhead = _body_for_exact_total(measurer, rel_path, language, budget + 1)
    if body == "" and overhead >= budget + 1:
        return  # header alone already exceeds budget+1; the "+1 over" shape doesn't apply

    chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=budget))
    chunk = _build_one(chunker, body, rel_path, language)

    assert _TRUNCATION_MARKER in chunk.chunk_text
    assert chunk.token_count == budget
