"""
Mutation-killing tests for extractive context compression.

Each test names, in its docstring, the source mutation it must fail under. These
cover the arithmetic that decides WHICH retrieved lines survive into the LLM
prompt — span merging, segment splitting, the lexical score, cosine scoring, the
signature floor, and the min-token floor. A wrong answer here degrades answers
with no error anywhere, so every oracle below is an exact value or an exact set,
never a "len > 0" proxy.

Headless by construction: the lexical path takes no sub_chunks table and no
query embedding, so nothing here can make an embedding, API, or network call.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import tiktoken

from trelix.compression import CompressionUnit, ExtractiveCompressor
from trelix.compression.extractive import _cosine, _merge_spans
from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    SearchResult,
    Symbol,
    SymbolKind,
)
from trelix.retrieval.context_compression import pack_compressed

_ENC = tiktoken.get_encoding("cl100k_base")


class _NoSubChunkDB:
    """No sub-chunk substrate -> forces the zero-inference lexical path.

    A plain class, not a Mock: a Mock would happily invent whatever method the
    compressor asked for and the test would pass against an interface that does
    not exist.
    """

    _conn = None

    def get_sub_chunks_for_symbol(self, symbol_id: int, granularity: str | None = None) -> list:
        return []


def _compressor() -> ExtractiveCompressor:
    return ExtractiveCompressor(db=_NoSubChunkDB())


def _is_subsequence(small: list[str], big: list[str]) -> bool:
    """True when every element of `small` appears in `big` in the same order."""
    it = iter(big)
    return all(any(candidate == want for candidate in it) for want in small)


# ---------------------------------------------------------------------------
# _merge_spans — the ordering / never-invent guarantee
# ---------------------------------------------------------------------------


def test_merge_spans_sorts_unordered_input() -> None:
    """Kills: `ordered = sorted(spans)` -> `ordered = list(spans)` in _merge_spans.

    Without the sort, kept spans come back in insertion order, so the renderer
    emits body lines out of order and the compressed text stops being a
    subsequence of the source -- a citation that points at reordered code.
    """
    unordered = [(5, 6), (0, 1)]
    # Precondition: the fixture must actually be out of order, otherwise this
    # test cannot discriminate a missing sort. Names the fixture per the rule.
    assert unordered != sorted(unordered), (
        "fixture `unordered` is already sorted; it can no longer detect a missing sort"
    )
    assert _merge_spans(unordered) == [(0, 1), (5, 6)]


def test_merge_spans_merges_adjacent_spans() -> None:
    """Kills: `if s <= le + 1:` -> `if s <= le:` in _merge_spans.

    (0,2) and (3,5) touch with no gap. Left unmerged they render identical text
    but report two kept_spans instead of one, so the citation claims a break in
    the source where none exists.
    """
    assert _merge_spans([(0, 2), (3, 5)]) == [(0, 5)]


def test_merge_spans_keeps_the_wider_end_when_nested() -> None:
    """Kills: `out[-1] = (ls, max(le, e))` -> `out[-1] = (ls, e)` in _merge_spans.

    Merging the enclosed (2,3) into (0,10) must not shrink the end to 3 -- that
    silently DROPS kept lines 4-10 from the citation while the text still
    contains them.
    """
    assert _merge_spans([(0, 10), (2, 3)]) == [(0, 10)]
    # Disjoint spans must stay separate -- guards against "merge everything".
    assert _merge_spans([(0, 1), (5, 6)]) == [(0, 1), (5, 6)]


# ---------------------------------------------------------------------------
# _cosine — the sub-chunk scoring path
# ---------------------------------------------------------------------------


def test_cosine_normalises_by_vector_norm() -> None:
    """Kills: `float(np.dot(q, vec) / (q_norm * v_norm))` -> `float(np.dot(q, vec))`.

    An unnormalised dot product ranks by vector MAGNITUDE, so a long, badly
    aligned sub-chunk outranks a short, perfectly aligned one and the wrong
    lines are kept.
    """
    q = np.asarray([1.0, 0.0], dtype=np.float64)
    q_norm = 1.0
    aligned_short = np.asarray([0.6, 0.0], dtype=np.float64)  # cosine 1.0, dot 0.6
    skewed_long = np.asarray([3.0, 3.0], dtype=np.float64)  # cosine 1/sqrt(2), dot 3.0

    # Precondition: the raw dot products must rank these two the OTHER way
    # round, otherwise dropping the normalisation would be undetectable here.
    assert float(np.dot(q, aligned_short)) < float(np.dot(q, skewed_long)), (
        "fixtures `aligned_short`/`skewed_long` no longer invert under raw dot "
        "product; they can no longer detect a missing cosine normalisation"
    )

    assert _cosine(q, q_norm, aligned_short) == pytest.approx(1.0, rel=1e-12)
    assert _cosine(q, q_norm, skewed_long) == pytest.approx(1 / 2**0.5, rel=1e-12)
    assert _cosine(q, q_norm, aligned_short) > _cosine(q, q_norm, skewed_long)


def test_cosine_returns_zero_for_dimension_mismatch() -> None:
    """Kills: `if vec.shape != q.shape: return 0.0` -> `return 1.0`.

    A stored vector of the wrong dimension is stale/foreign. Scoring it 1.0
    would make the WORST candidate the top-ranked span; it must score 0.0 so it
    loses to every well-formed vector.
    """
    q = np.asarray([1.0, 0.0], dtype=np.float64)
    assert _cosine(q, 1.0, np.asarray([1.0, 0.0, 0.0], dtype=np.float64)) == 0.0
    # A zero vector has no direction -- also 0.0, not a divide-by-zero.
    assert _cosine(q, 1.0, np.asarray([0.0, 0.0], dtype=np.float64)) == 0.0


# ---------------------------------------------------------------------------
# _segments — line splitting
# ---------------------------------------------------------------------------


def test_segments_keeps_the_final_unterminated_segment() -> None:
    """Kills: dropping `raw.append((start, len(body_lines) - 1))` after the loop.

    Most real bodies do NOT end in a blank line, so without this the LAST
    segment of almost every symbol is never scored and therefore can never be
    kept -- the tail of every compressed body would silently vanish.
    """
    assert _compressor()._segments(["a", "b", "", "c", "d"]) == [(0, 1), (3, 4)]


def test_segments_treats_a_whitespace_only_line_as_a_separator() -> None:
    """Kills: `if line.strip() == "":` -> `if line == "":` in _segments.

    Indented code routinely leaves trailing spaces on "blank" lines. Comparing
    raw equality fuses two logical stanzas into one segment, so they can only be
    kept or dropped together.
    """
    compressor = _compressor()
    whitespace_only = ["a", "   ", "b"]
    # Precondition: the separator must be whitespace-but-not-empty, else this
    # test degenerates into the plain-empty-line case and stops discriminating.
    assert whitespace_only[1] != "" and whitespace_only[1].strip() == "", (
        "fixture `whitespace_only` middle line is no longer whitespace-only"
    )
    assert compressor._segments(whitespace_only) == [(0, 0), (2, 2)]
    assert compressor._segments(["a", "", "b"]) == [(0, 0), (2, 2)]


def test_brace_balance_counts_open_minus_close() -> None:
    """Kills: `text.count("{") - text.count("}")` -> `text.count("}") - text.count("{")`.

    The sign drives the brace-merge loop, which only runs while the balance is
    positive. Inverted, an open block never pulls in its continuation.
    """
    compressor = _compressor()
    assert compressor._brace_balance(["function f() {"], 0, 0) == 1
    assert compressor._brace_balance(["}"], 0, 0) == -1
    assert compressor._brace_balance(["if (a) { b(); }"], 0, 0) == 0


def test_segments_merges_blocks_until_braces_balance() -> None:
    """Kills the inverted `_brace_balance` sign (see above) at the _segments level.

    A brace-language body whose opening line is separated from its closing brace
    by blank lines must come back as ONE segment; inverted, it comes back as
    three, so the closing brace can be elided away from its opener.
    """
    brace_body = ["function f() {", "", "  body;", "", "}"]
    assert _compressor()._segments(brace_body) == [(0, 4)]


def test_segments_stops_merging_when_the_balance_is_negative() -> None:
    """Kills: `while balance > 0 and ...` -> `while balance != 0 and ...`.

    An extracted body can begin with an UNMATCHED closing brace (balance -1).
    `> 0` leaves it alone; `!= 0` would keep swallowing later stanzas into it,
    merging unrelated code into a single all-or-nothing segment.
    """
    unmatched_close = ["}", "", "x = 1"]
    compressor = _compressor()
    # Precondition: the first segment's balance must be NEGATIVE, which is the
    # only state where `> 0` and `!= 0` disagree.
    assert compressor._brace_balance(unmatched_close, 0, 0) < 0, (
        "fixture `unmatched_close` no longer has a negative brace balance; it "
        "can no longer distinguish `balance > 0` from `balance != 0`"
    )
    assert compressor._segments(unmatched_close) == [(0, 0), (2, 2)]


# ---------------------------------------------------------------------------
# _signature_end_idx — the unconditional must-keep floor
# ---------------------------------------------------------------------------


def test_signature_end_idx_spans_a_multiline_python_header() -> None:
    """Kills: `body_lines[:8]` -> `body_lines[:1]` in _signature_end_idx.

    A signature broken across lines must be kept WHOLE. Scanning only the first
    line reports index 0, so the must-keep span covers `def f(` and the
    compressor may elide the parameters and the closing `):`.
    """
    multiline_header = ["def f(", "    a,", "    b,", "):", "    pass"]
    # Precondition: the header must genuinely span past line 0, else a 1-line
    # scan window would give the same answer and the test proves nothing.
    assert multiline_header[0].rstrip().endswith("("), (
        "fixture `multiline_header` first line now terminates the signature; a "
        "1-line scan window would agree and the test stops discriminating"
    )
    assert _compressor()._signature_end_idx(multiline_header) == 3
    assert _compressor()._signature_end_idx(["def f(a):", "    pass"]) == 0


def test_signature_end_idx_recognises_a_brace_language_header() -> None:
    """Kills: `endswith((":", "{"))` -> `endswith((":",))` in _signature_end_idx.

    Dropping "{" makes every brace-language body (JS/Go/Java/C) fall through to
    the `return 0` default, so only the first line of a multi-line signature is
    protected -- the module advertises "any language".
    """
    js_header = ["function f(", "  a", ") {", "  return a;", "}"]
    # Precondition: no line of this fixture may end in ":", or the colon-only
    # mutant would still find a header and the test would not discriminate.
    assert not any(line.rstrip().endswith(":") for line in js_header), (
        "fixture `js_header` now contains a colon-terminated line; the "
        "colon-only mutant would still pass"
    )
    assert _compressor()._signature_end_idx(js_header) == 2


# ---------------------------------------------------------------------------
# _score_lexical — the scoring formula
# ---------------------------------------------------------------------------


def test_lexical_score_is_density_normalised() -> None:
    """Kills all three scoring-formula mutants in _score_lexical:
      `overlap / (len(seg_tokens) + 1) ** 0.5` -> `overlap`            (no normalisation)
                                              -> `... + 2) ** 0.5`     (wrong constant)
                                              -> `... + 1) ** 1.0`     (wrong exponent)

    The exact values are hand-derived from the documented rule
    (overlap / sqrt(unique_seg_tokens + 1)), not read out of the module.
    """
    lines = ["authenticate", "", "def login user token extra"]
    query = "authenticate login user"
    # Hand-counted: query tokens = {authenticate, login, user}
    #   seg (0,0) "authenticate"               -> 1 unique token,  overlap 1
    #   seg (2,2) "def login user token extra" -> 5 unique tokens, overlap 2
    scored = _compressor()._score_lexical(query, lines)

    # Precondition: the two segments must differ in unique-token count, which is
    # the only thing the denominator can act on.
    assert len(scored) == 2, "fixture `lines` must yield exactly two segments"
    assert scored[0][0] == (0, 0) and scored[1][0] == (2, 2)

    assert scored[0][1] == pytest.approx(1 / 2**0.5, rel=1e-12)
    assert scored[1][1] == pytest.approx(2 / 6**0.5, rel=1e-12)


def test_lexical_matching_is_case_insensitive_on_both_sides() -> None:
    """Kills BOTH lowercase drops in _score_lexical:
      `{t.lower() for t in _WORD_RE.findall(query or "")}` -> `{t for t in ...}`
      `{t.lower() for t in _WORD_RE.findall(seg_text)}`    -> `{t for t in ...}`

    Either one makes matching case-sensitive, so an upper-case query stops
    matching a capitalised identifier and the on-topic segment scores 0 --
    silently keeping the wrong lines.
    """
    lines = ["def Authenticate(self):", "", "    return 1"]
    query = "AUTHENTICATE"
    # Precondition: query and body must differ in case for the shared token,
    # otherwise neither lowercase call is observable.
    assert "Authenticate" in lines[0] and "AUTHENTICATE" not in lines[0], (
        "fixture `lines`/`query` no longer differ in case; dropping either "
        "`.lower()` would be undetectable"
    )
    scored = dict(_compressor()._score_lexical(query, lines))
    # Hand-counted: seg (0,0) = {def, authenticate, self} -> 3 unique, overlap 1
    assert scored[(0, 0)] == pytest.approx(1 / 4**0.5, rel=1e-12)
    assert scored[(2, 2)] == 0.0


# ---------------------------------------------------------------------------
# End-to-end: output is a subsequence and the elision counts add up
# ---------------------------------------------------------------------------

_HANDLER_BODY = "\n".join(
    [
        "def handler(request):",  # rel 0
        '    """Handle the auth request."""',  # rel 1
        "",  # rel 2
        "    validate_auth_token(request)",  # rel 3
        "",  # rel 4
        "    log.debug('one')",  # rel 5
        "    log.debug('two')",  # rel 6
        "    log.debug('three')",  # rel 7
        "    log.debug('four')",  # rel 8
        "    log.debug('five')",  # rel 9
        "",  # rel 10
        "    return 1",  # rel 11
    ]
)


def test_compressed_text_is_a_subsequence_and_elisions_account_for_every_line() -> None:
    """Kills: `ordered = sorted(spans)` -> `list(spans)`, and any renderer
    mutation that miscounts an elided run (e.g. `gap = i0 - prev_end - 1`).

    Two invariants at once, both stated in the module docstring:
      * every non-marker output line is a body line, in body order (subsequence)
      * kept lines + elided lines == the whole body, so no line is lost or
        double-counted in a citation.
    """
    unit = CompressionUnit(
        symbol_id=1,
        body=_HANDLER_BODY,
        signature="def handler(request)",
        docstring="Handle the auth request.",
        line_start=100,
        line_end=111,
        qualified_name="handler",
    )
    result = _compressor().compress("validate auth token", unit, target_ratio=0.5)

    body_lines = _HANDLER_BODY.splitlines()
    out_lines = result.text.splitlines()
    markers = [ln for ln in out_lines if ln.startswith("# ...")]
    kept_lines = [ln for ln in out_lines if not ln.startswith("# ...")]

    # Precondition: something must actually have been elided, or a passthrough
    # would satisfy both invariants trivially.
    assert markers, (
        "fixture `_HANDLER_BODY` at ratio 0.5 produced no elision; the "
        "subsequence and accounting invariants become vacuous"
    )

    assert _is_subsequence(kept_lines, body_lines), (
        f"output is not a subsequence of the body: {kept_lines}"
    )
    elided = sum(int(m.split("# ... ")[1].split(" lines")[0]) for m in markers)
    assert len(kept_lines) + elided == len(body_lines)
    # The exact rendering for this fixture, pinned so a silent re-slice fails.
    assert result.kept_spans == [(100, 101), (103, 103), (111, 111)]
    assert markers == ["# ... 1 lines elided ...", "# ... 7 lines elided ..."]
    # token_count is always recomputed, never inherited from the original.
    assert result.token_count == len(_ENC.encode(result.text))
    assert result.token_count < result.original_token_count


# ---------------------------------------------------------------------------
# min-token floor (assembler-side packing)
# ---------------------------------------------------------------------------

_FILE = IndexedFile(
    path="/repo/src/pack/service.py",
    rel_path="src/pack/service.py",
    language=Language.PYTHON,
    hash="sha-1",
    size_bytes=4096,
    id=1,
    indexed_at=datetime(2024, 1, 1),
)


def _packing_body(name: str, blocks: int = 6) -> str:
    lines = [f"def {name}(request):", f'    """Handle {name}."""']
    for b in range(blocks):
        lines += [
            "",
            f"    # step {b} dispatch payload",
            f"    payload_{b} = build(request, {b})",
            f"    out_{b} = dispatch(payload_{b})",
        ]
    lines += ["", "    return out_0"]
    return "\n".join(lines)


def _result_from_body(
    sym_id: int, name: str, line_start: int, body: str, score: float, rank: int
) -> SearchResult:
    """A SearchResult whose chunk_text IS its body, so token accounting is exact."""
    symbol = Symbol(
        file_id=1,
        name=name,
        qualified_name=f"Service.{name}",
        kind=SymbolKind.FUNCTION,
        line_start=line_start,
        line_end=line_start + len(body.splitlines()) - 1,
        signature=f"def {name}(request)",
        body=body,
        docstring=f"Handle {name}.",
        id=sym_id,
    )
    return SearchResult(
        chunk=Chunk(
            symbol_id=sym_id, chunk_text=body, token_count=len(_ENC.encode(body)), id=sym_id
        ),
        symbol=symbol,
        file=_FILE,
        score=score,
        rank=rank,
        source="vector",
    )


def _packing_result(
    sym_id: int, name: str, line_start: int, score: float, rank: int
) -> SearchResult:
    """A multi-stanza body big enough that compressing it genuinely shrinks it."""
    return _result_from_body(sym_id, name, line_start, _packing_body(name), score, rank)


def test_min_tokens_floor_is_exclusive_at_the_boundary() -> None:
    """Kills: `if original <= 0 or original < min_tokens:` -> `... <= min_tokens:`.

    A body of EXACTLY min_tokens is at the floor, not below it, so it must still
    be compressed and included. Making the comparison inclusive drops the
    boundary body from the pack -- a silently lost retrieval result.
    """
    kept = _packing_result(1, "alpha", 10, 0.9, 1)
    candidate = _packing_result(2, "beta", 200, 0.8, 2)
    exact = candidate.chunk.token_count

    # Precondition: min_tokens must equal the candidate's size EXACTLY -- that
    # is the only value at which `<` and `<=` disagree.
    assert exact > 0, "fixture `candidate` has no tokens; the boundary is untestable"

    _selected, _compressed, stats = pack_compressed(
        query="dispatch payload build",
        eligible=[kept, candidate],
        wave1=[kept],
        token_budget=kept.chunk.token_count + exact // 2,
        compressor=_compressor(),
        target_ratio=0.45,
        min_tokens=exact,
        query_embedding=None,
    )
    assert stats["below_min_tokens"] == 0, "a body AT the floor must not be skipped"
    assert stats["wave2_added"] == 1

    # One token smaller than the floor -> genuinely below it, and skipped.
    _s2, _c2, below = pack_compressed(
        query="dispatch payload build",
        eligible=[kept, candidate],
        wave1=[kept],
        token_budget=kept.chunk.token_count + exact // 2,
        compressor=_compressor(),
        target_ratio=0.45,
        min_tokens=exact + 1,
        query_embedding=None,
    )
    assert below["below_min_tokens"] == 1
    assert below["wave2_added"] == 0


# ---------------------------------------------------------------------------
# Ratio arithmetic — the two off-by-ones that silently drop or keep a whole body
# ---------------------------------------------------------------------------


def test_a_span_costing_exactly_the_budget_is_kept() -> None:
    """Kills: `if self._span_tokens(...) <= target_tokens:` -> `< target_tokens`.

    Fixture arithmetic (hand-derived, tiktoken cl100k_base):
      body = ['def g():', '', '    pass', '']  -> 6 original tokens
      target_tokens = int(0.9 * 6) = 5
      signature span + segment (2,2) costs exactly 5 tokens

    So the only candidate lands EXACTLY on the budget. With `<` it is rejected
    and the body collapses to the signature alone -- `pass` is dropped even
    though it fits.
    """
    body = "def g():\n\n    pass\n\n"
    unit = CompressionUnit(
        symbol_id=1,
        body=body,
        signature="def g()",
        docstring=None,
        line_start=1,
        line_end=4,
        qualified_name="g",
    )
    compressor = _compressor()
    result = compressor.compress("log build x return", unit, target_ratio=0.9)

    # Preconditions: both the token total and the exact-fit property must hold,
    # or this fixture is no longer sitting on the boundary it was built for.
    assert result.original_token_count == 6, (
        "fixture `body` no longer encodes to 6 tokens; target_tokens is no "
        "longer 5 and the exact-budget boundary is untested"
    )
    body_lines = body.splitlines()
    exact_cost = compressor._span_tokens(body_lines, [(0, 0), (2, 2)])
    assert exact_cost == 5, (
        f"fixture `body` candidate now costs {exact_cost}, not the budget of 5; "
        "`<=` and `<` would agree and the test stops discriminating"
    )

    assert result.kept_spans == [(1, 1), (3, 3)]
    assert result.text == "def g():\n# ... 1 lines elided ...\n    pass\n# ... 1 lines elided ..."


def test_target_tokens_truncates_rather_than_rounds() -> None:
    """Kills: `int(target_ratio * original_tokens)` -> `round(...)`.

    Fixture arithmetic (hand-derived, tiktoken cl100k_base):
      body is ONE brace-merged segment costing 17 tokens; original = 17
      int(0.99 * 17)   = 16  -> the segment does NOT fit
      round(0.99 * 17) = 17  -> the segment DOES fit

    Under `round` the compressor returns the body verbatim, i.e. it silently
    performs NO compression when asked for 99%. Truncation is what keeps the
    output strictly inside the requested ratio.

    NOTE (pins current behaviour, deliberately): because the whole body is a
    single indivisible brace-balanced segment that is one token over budget, the
    documented greedy rule ("skip oversized, no break") collapses it to the
    signature line. That is sharp but by design and still result-lossless.
    """
    body = "\n".join(["function f(a) {", "", "  const x = build(a);", "", "  return x;", "}"])
    unit = CompressionUnit(
        symbol_id=1,
        body=body,
        signature="function f(a)",
        docstring=None,
        line_start=1,
        line_end=6,
        qualified_name="f",
    )
    result = _compressor().compress("log build x return", unit, target_ratio=0.99)

    # Precondition: int() and round() must actually disagree for this fixture.
    assert result.original_token_count == 17, (
        "fixture `body` no longer encodes to 17 tokens; int() and round() may "
        "now agree at ratio 0.99 and the test stops discriminating"
    )
    assert int(0.99 * 17) == 16 and round(0.99 * 17) == 17

    assert result.kept_spans == [(1, 1)]
    assert result.text == "function f(a) {\n# ... 5 lines elided ..."
    assert result.text != body, "asking for 99% must still compress, not passthrough"


_TIE_BODY = "\n".join(
    [
        "def handler(request):",  # rel 0  signature
        '    """Doc."""',  # rel 1  docstring
        "",  # rel 2
        "    validate(request)",  # rel 3  top-scoring segment
        "",  # rel 4
        "    log('a')",  # rel 5  tied segment A (earlier)
        "    log('b')",  # rel 6
        "    log('c')",  # rel 7
        "",  # rel 8
        "    return 1",  # rel 9  tied segment B (later)
    ]
)


def test_equal_scoring_spans_are_broken_by_earliest_source_position() -> None:
    """Kills: `key=lambda s: (-s[1], s[0][0])` -> `(-s[1], -s[0][0])` in _compress.

    Segments (5,7) and (9,9) both score EXACTLY 0.0 against this query, and the
    budget only has room for one of them. The documented tie-break is source
    order, so the EARLIER span wins; flipping it keeps `return 1` instead of the
    log block and makes selection depend on an unstable ordering rather than on
    the source.
    """
    unit = CompressionUnit(
        symbol_id=1,
        body=_TIE_BODY,
        signature="def handler(request)",
        docstring="Doc.",
        line_start=1,
        line_end=10,
        qualified_name="handler",
    )
    compressor = _compressor()
    query = "validate request"

    # Precondition: the two competing segments must score EXACTLY equal, or the
    # score (not the tie-break) is what is being measured and the mutant hides.
    scores = dict(compressor._score_lexical(query, _TIE_BODY.splitlines()))
    assert scores[(5, 7)] == scores[(9, 9)] == 0.0, (
        f"fixture `_TIE_BODY` segments no longer tie ({scores[(5, 7)]} vs "
        f"{scores[(9, 9)]}); the tie-break is no longer what this test measures"
    )

    result = compressor.compress(query, unit, target_ratio=0.9)
    # Absolute spans (line_start=1): rel (5,7) -> (6,8); rel (9,9) -> (10,10).
    assert result.kept_spans == [(1, 2), (4, 4), (6, 8)]
    assert "    log('a')" in result.text
    assert "    return 1" not in result.text


# ---------------------------------------------------------------------------
# The two packing-layer guards that keep "compression" from making things worse
# ---------------------------------------------------------------------------

# A short multi-stanza body. Compressing it INFLATES the token count, because
# three "# ... N lines elided ..." markers cost more than the elided lines saved.
_INFLATING_BODY = "\n".join(
    [
        "def handler(request):",
        '    """Doc."""',
        "",
        "    validate(request)",
        "",
        "    log('a')",
        "    log('b')",
        "    log('c')",
        "",
        "    return 1",
    ]
)


def test_a_compression_that_would_inflate_the_body_is_rejected() -> None:
    """Kills: dropping `and candidate.token_count < original` in _compress_to_fit.

    ExtractiveCompressor.compress() is allowed to return MORE tokens than it was
    given -- on a short body the elision markers cost more than the lines they
    replace. The packing layer is the only thing standing between that and a
    prompt that grows when we asked it to shrink, so the guard must hold even
    when the leftover budget is large enough to accept the inflated version.
    """
    anchor = _result_from_body(1, "alpha", 10, "def alpha(request):\n    return 1", 0.9, 1)
    candidate = _result_from_body(2, "handler", 200, _INFLATING_BODY, 0.8, 2)

    # Precondition AND the finding it rests on: the first ladder rung really
    # does inflate. If this stops being true the guard is unobservable here.
    unit = CompressionUnit(
        symbol_id=2,
        body=_INFLATING_BODY,
        signature="def handler(request)",
        docstring="Handle handler.",
        line_start=200,
        line_end=209,
        qualified_name="Service.handler",
    )
    rung_one = _compressor().compress("validate request", unit, target_ratio=0.9)
    assert rung_one.token_count > rung_one.original_token_count, (
        "fixture `_INFLATING_BODY` no longer inflates at ratio 0.9 "
        f"({rung_one.token_count} vs {rung_one.original_token_count}); this test "
        "can no longer detect a missing anti-inflation guard"
    )

    _selected, compressed, stats = pack_compressed(
        query="validate request",
        eligible=[anchor, candidate],
        wave1=[anchor],
        token_budget=anchor.chunk.token_count + 500,  # deliberately roomy
        compressor=_compressor(),
        target_ratio=0.9,
        min_tokens=1,
        query_embedding=None,
    )

    # The inflating rung must be refused; the ladder falls through to the floor.
    assert stats["compressed"] == 0, "the inflating rung must not be accepted"
    assert stats["floor"] == 1
    for cresult in compressed.values():
        assert cresult.token_count < cresult.original_token_count, (
            "a packed compression must never be larger than the body it replaced"
        )


def test_a_compression_that_would_overflow_the_budget_is_skipped() -> None:
    """Kills: dropping `candidate.token_count <= remaining` in _compress_to_fit.

    With only 5 tokens left, neither ladder rung can fit -- the unconditional
    must-keep spans alone cost more than that. The candidate must be SKIPPED and
    recorded, not packed. Without the budget half of the guard it is accepted
    purely because it is smaller than the original, and the assembled context
    silently overruns the hard token budget it promised to respect.
    """
    anchor = _result_from_body(1, "alpha", 10, "def alpha(request):\n    return 1", 0.9, 1)
    candidate = _packing_result(2, "beta", 200, 0.8, 2)
    remaining = 5
    budget = anchor.chunk.token_count + remaining

    # Precondition: even the smallest rung (signature + docstring) must exceed
    # the leftover budget, or the guard would never be exercised.
    unit = CompressionUnit(
        symbol_id=2,
        body=candidate.symbol.body or "",
        signature=candidate.symbol.signature or "",
        docstring=candidate.symbol.docstring,
        line_start=candidate.symbol.line_start,
        line_end=candidate.symbol.line_end,
        qualified_name=candidate.symbol.qualified_name,
    )
    floor_rung = _compressor().compress("dispatch payload build", unit, target_ratio=0.01)
    assert floor_rung.token_count > remaining, (
        f"fixture `candidate` floor rung now costs {floor_rung.token_count}, which "
        f"fits in the {remaining} leftover tokens; the budget guard is unobservable"
    )
    assert floor_rung.token_count < candidate.chunk.token_count, (
        "floor rung must still be smaller than the original, or the OTHER half "
        "of the guard would be what rejects it"
    )

    _selected, _compressed, stats = pack_compressed(
        query="dispatch payload build",
        eligible=[anchor, candidate],
        wave1=[anchor],
        token_budget=budget,
        compressor=_compressor(),
        target_ratio=0.45,
        min_tokens=1,
        query_embedding=None,
    )
    assert stats["skipped"] == 1
    assert stats["wave2_added"] == 0
    assert stats["tokens_used"] == anchor.chunk.token_count
    assert stats["tokens_used"] <= budget, "the hard token budget must never be exceeded"
