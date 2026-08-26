"""
Mutation-killing tests for ExtractiveCompressor.compress()/_compress()'s path
selection, its two internal `_passthrough` call sites, and the `provider`
field on the returned `CompressionResult` -- the surviving mutants left after
``test_compression_extractive_mutants.py``'s existing coverage (round-11
measurement).

Split into its own file purely to keep each file within a readable size --
both this file and ``test_compression_extractive_internal_helpers_mutants.py``
exercise the same ``src/trelix/compression/extractive.py`` module.

Every fixture below was checked against mutmut's own generated mutant variants
before being written down: each docstring's claimed mutation was reproduced
against the actual mutant code and shown to diverge from the unmutated
baseline.

EQUIVALENT MUTANTS in `_compress`/`_passthrough`, deliberately NOT tested
(rule 7) -- each verified empirically against mutmut's generated variant, not
guessed:
  mutmut_5, 7, 8          The three OTHER mutations of the `n == 0` branch's
                          `return self._passthrough(unit, original_tokens)`
                          call (passing `None` as `unit`, dropping
                          `original_tokens`, or a bare trailing-comma call)
                          all raise inside `_passthrough` (`AttributeError`
                          or `TypeError: missing argument`). Every one of
                          those exceptions propagates straight out of
                          `_compress` into `compress()`'s own
                          `except Exception:` handler, which then calls
                          `self._passthrough(unit, original_tokens)` --
                          the EXACT SAME call, with the EXACT SAME (correct)
                          arguments, that the unmutated code would have made
                          at that point anyway. The crash is invisible: the
                          final `CompressionResult` is byte-identical.
  mutmut_54               The greedy tie-break key `(-score, s[0][0])` ->
                          `(-score, s[0][1])` (sort by span END instead of
                          START on a score tie). Proven equivalent by case
                          exhaustion over every possible pair of 1-D
                          intervals: DISJOINT spans have start-order and
                          end-order that always agree (so the sort order is
                          identical either way), and OVERLAPPING/nested spans
                          make the final admitted coverage order-independent
                          anyway (the wider span's own admission check only
                          depends on its own merged cost, never on whether a
                          narrower nested span was tried first). No other
                          interval relationship exists, so no fixture can
                          discriminate this key for any input this algorithm
                          could ever be given.
  mutmut_66, 67, 68       The `if not merged: merged = [...]` fallback and its
                          three mutated bodies. `selected` starts as
                          `_must_keep_spans(...)`, which always returns at
                          least one span, and every later reassignment only
                          MERGES more spans in -- `_merge_spans` of a
                          non-empty list is never empty. `merged` can
                          therefore never be falsy for any input, so this
                          whole branch (and every mutation confined to it) is
                          dead code. Checked across four different body
                          shapes (empty, single-char, multi-blank-line,
                          all-whitespace) with no divergence in any of them.
  mutmut_73, 75, 76       The three OTHER mutations of the "nothing elided"
                          branch's `return self._passthrough(unit,
                          original_tokens)` call -- same masking mechanism as
                          mutmut_5/7/8 above, at a different call site.
  mutmut_98               `provider="extractive"` kwarg dropped from
                          `_compress`'s own `CompressionResult(...)` call.
                          Same dataclass-default reasoning as
                          `_passthrough`'s mutmut_20 (see the sibling test
                          file) -- the field defaults to the literal
                          `"extractive"` already.
"""

from __future__ import annotations

import logging
import sqlite3
import struct

from trelix.compression import CompressionUnit, ExtractiveCompressor


class _NoSubChunkDB:
    """No sub-chunk substrate -- plain class, not a Mock (rule 3)."""

    _conn = None

    def get_sub_chunks_for_symbol(self, symbol_id: int, granularity: str | None = None) -> list:
        return []


def _compressor() -> ExtractiveCompressor:
    return ExtractiveCompressor(db=_NoSubChunkDB())


# ---------------------------------------------------------------------------
# n == 0 (empty body) branch
# ---------------------------------------------------------------------------


def test_compress_reports_last_path_passthrough_for_a_zero_line_body() -> None:
    """Kills: `if n == 0:` -> `if n == 1:` in `_compress`.

    An empty body (0 lines) must take the passthrough branch and record
    ``last_path == "passthrough"``. Under the mutant, `0 == 1` is false, so
    execution falls through into the full scoring machinery instead, which
    -- for an empty body -- ends up setting `last_path = "lexical"` even
    though the returned text and token counts happen to coincide with the
    correct passthrough result. `last_path` is the field this mutant actually
    moves.
    """
    unit = CompressionUnit(
        symbol_id=1,
        body="",
        signature="s",
        docstring=None,
        line_start=1,
        line_end=1,
        qualified_name="q",
    )
    compressor = _compressor()
    compressor.compress("anything", unit, target_ratio=0.5)
    assert compressor.last_path == "passthrough"


def test_compress_zero_line_body_reports_a_real_zero_token_count_not_none() -> None:
    """Kills: `return self._passthrough(unit, original_tokens)` ->
    `return self._passthrough(unit, None)` in the `n == 0` branch.

    `token_count`/`original_token_count` must be the real computed value (0
    for an empty body), never `None`.
    """
    unit = CompressionUnit(
        symbol_id=1,
        body="",
        signature="s",
        docstring=None,
        line_start=1,
        line_end=1,
        qualified_name="q",
    )
    result = _compressor().compress("anything", unit, target_ratio=0.5)
    assert result.token_count == 0
    assert result.original_token_count == 0


# ---------------------------------------------------------------------------
# target_tokens floor
# ---------------------------------------------------------------------------


def test_compress_target_tokens_floor_is_one_not_two() -> None:
    """Kills: `target_tokens = max(1, int(target_ratio * original_tokens))` ->
    `max(2, ...)`.

    Fixture (hand-derived, tiktoken cl100k_base): body = "\\nX" encodes to 2
    tokens; a blank FIRST line makes `_segments` skip it entirely, so the
    unconditional must-keep span is just that blank line (0 tokens) and the
    only scoreable span is line 1 ("X"). At `target_ratio=0.5`,
    `int(0.5 * 2) == 1`. Merging the must-keep span with the one scored span
    costs exactly 2 tokens (the body's own newline separator costs a token):
    with the real floor of 1, 2 > 1 -- REJECTED, leaving an elision marker.
    With a floor of 2, 2 <= 2 -- ACCEPTED, collapsing to a byte-identical
    passthrough with no elision marker at all.
    """
    unit = CompressionUnit(
        symbol_id=1,
        body="\nX",
        signature="s",
        docstring=None,
        line_start=1,
        line_end=2,
        qualified_name="q",
    )
    result = _compressor().compress("X", unit, target_ratio=0.5)
    # Precondition: the fixture must not have collapsed to a full passthrough,
    # or the floor being tested no longer has anything to reject.
    assert result.original_token_count == 2, "fixture body no longer encodes to 2 tokens"
    assert "lines elided" in result.text, (
        "fixture no longer produces an elision marker at the real floor of 1; "
        "the max(1,...) vs max(2,...) boundary is no longer observable here"
    )
    assert result.text == "\n# ... 1 lines elided ..."


# ---------------------------------------------------------------------------
# sub-chunk vs lexical path selection
# ---------------------------------------------------------------------------


class _DBSubChunksTableMissing:
    """`chunk_embeddings` exists (so a sub-chunk COULD have a real vector),
    but `sub_chunks` does not -- `_has_sub_chunks()` genuinely returns False,
    independent of what `get_sub_chunks_for_symbol` (a plain Python method)
    happens to return."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute(
            "CREATE TABLE chunk_embeddings (chunk_id INTEGER PRIMARY KEY, embedding BLOB)"
        )
        blob = struct.pack("2f", 1.0, 0.0)
        self._conn.execute("INSERT INTO chunk_embeddings VALUES (?, ?)", (10_000_001, blob))
        self._conn.commit()

    def get_sub_chunks_for_symbol(self, symbol_id: int, granularity: str | None = None) -> list:
        class Sub:
            id = 1
            line_start = 1
            line_end = 1

        return [Sub()]


def test_compress_stays_lexical_when_the_sub_chunks_table_is_absent() -> None:
    """Kills: `if query_embedding is not None and self._has_sub_chunks():` ->
    `... or self._has_sub_chunks():`.

    `or` short-circuits on `query_embedding is not None` alone, skipping the
    `_has_sub_chunks()` check entirely and unconditionally trying the
    sub-chunk path whenever an embedding is supplied -- even when the index
    genuinely has no `sub_chunks` table. The real `and` must stay on the
    lexical path in that case.
    """
    unit = CompressionUnit(
        symbol_id=1,
        body="a\nb",
        signature="s",
        docstring=None,
        line_start=1,
        line_end=2,
        qualified_name="q",
    )
    compressor = ExtractiveCompressor(db=_DBSubChunksTableMissing())
    compressor.compress("a", unit, target_ratio=0.5, query_embedding=[1.0, 0.0])
    assert compressor.last_path == "lexical"


class _TwoSubsAsymmetricDB:
    """SubA (early in source, orthogonal to the query -> real cosine 0);
    SubB (later in source, aligned with the query -> real cosine 1)."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("CREATE TABLE sub_chunks (id INTEGER PRIMARY KEY)")
        self._conn.execute("INSERT INTO sub_chunks (id) VALUES (1)")
        self._conn.execute("INSERT INTO sub_chunks (id) VALUES (2)")
        self._conn.execute(
            "CREATE TABLE chunk_embeddings (chunk_id INTEGER PRIMARY KEY, embedding BLOB)"
        )
        blob_a = struct.pack("2f", 0.0, 1.0)
        blob_b = struct.pack("2f", 1.0, 0.0)
        self._conn.execute("INSERT INTO chunk_embeddings VALUES (?, ?)", (10_000_001, blob_a))
        self._conn.execute("INSERT INTO chunk_embeddings VALUES (?, ?)", (10_000_002, blob_b))
        self._conn.commit()

    def get_sub_chunks_for_symbol(self, symbol_id: int, granularity: str | None = None) -> list:
        class SubA:
            id = 1
            line_start = 2  # body-relative index 1: "early_span_line"
            line_end = 2

        class SubB:
            id = 2
            line_start = 4  # body-relative index 3: "late_span_line"
            line_end = 4

        return [SubA(), SubB()]


def test_compress_passes_the_real_query_embedding_into_sub_chunk_scoring() -> None:
    """Kills: `scored = self._score_sub_chunks(unit, body_lines, query_embedding)`
    -> `self._score_sub_chunks(unit, body_lines, None)`.

    With the real embedding, the ALIGNED later sub-chunk (SubB, real cosine 1)
    outranks the orthogonal earlier one (SubA, real cosine 0) and gets kept.
    Passing `None` instead collapses both to a tied cosine of 0.0 (`np.asarray
    (None, ...)` is a shapeless scalar, so `_cosine`'s shape guard scores
    everything 0.0), and the tie-break then falls back to EARLIEST source
    position -- silently keeping the wrong (orthogonal) span instead.
    """
    unit = CompressionUnit(
        symbol_id=1,
        body="sig_line\nearly_span_line\nmid\nlate_span_line\ntail",
        signature="s",
        docstring=None,
        line_start=1,
        line_end=5,
        qualified_name="q",
    )
    compressor = ExtractiveCompressor(db=_TwoSubsAsymmetricDB())
    result = compressor.compress("q", unit, target_ratio=0.6, query_embedding=[1.0, 0.0])
    assert compressor.last_path == "sub_chunk"
    assert "late_span_line" in result.text
    assert "early_span_line" not in result.text


def test_compress_logs_the_actual_selected_path_name(caplog) -> None:
    """Kills: `self._log_path_once(path)` -> `self._log_path_once(None)`.

    The log line must name the REAL selected path ("lexical" here), not the
    literal string "None".
    """
    unit = CompressionUnit(
        symbol_id=1,
        body="a\nb\n\nc\nd",
        signature="s",
        docstring=None,
        line_start=1,
        line_end=5,
        qualified_name="q",
    )
    compressor = _compressor()
    with caplog.at_level(logging.INFO, logger="trelix.compression.extractive"):
        compressor.compress("c", unit, target_ratio=0.4)
    assert any("lexical" in r.getMessage() for r in caplog.records)
    assert not any("None" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# "nothing elided -> byte-identical passthrough" branch
# ---------------------------------------------------------------------------


def test_compress_full_body_coverage_returns_a_byte_identical_passthrough() -> None:
    """Kills four mutants at once, all guarding the SAME branch:
      `if merged == [(0, n - 1)]:` -> `[(0, n + 1)]` (mutmut_71) or
                                       `[(0, n - 2)]` (mutmut_72) -- both
        conditions can never match a real `merged`, so the branch is skipped
        and the compressor falls through to `_render` instead of returning
        the exact original body.
      `return self._passthrough(unit, original_tokens)` ->
        `self._passthrough(unit, None)` (mutmut_74) -- reaches the branch but
        stores `token_count = None` instead of the real count.

    Fixture: body = "def g():\\n    pass\\n" (a TRAILING newline). At
    `target_ratio=0.9` the greedy loop legitimately re-covers the whole body
    (must-keep + the one lexical segment fits the budget), so `merged` really
    does equal `(0, n - 1)`. `"\\n".join(body_lines)` (what `_render` would
    produce instead) DROPS that trailing newline -- so skipping the "nothing
    elided" branch is directly observable as a text mismatch, not just a
    token-count mismatch.
    """
    unit = CompressionUnit(
        symbol_id=1,
        body="def g():\n    pass\n",
        signature="def g()",
        docstring=None,
        line_start=1,
        line_end=2,
        qualified_name="g",
    )
    result = _compressor().compress("anything", unit, target_ratio=0.9)
    # Precondition: the fixture must genuinely reach full coverage, not just
    # happen to render the same text by coincidence.
    assert result.text == unit.body, "fixture no longer reaches the 'nothing elided' branch"
    assert result.token_count == 6


# ---------------------------------------------------------------------------
# provider field
# ---------------------------------------------------------------------------


def test_compress_and_passthrough_provider_field_is_pinned_to_extractive() -> None:
    """Kills six `provider=` mutants at once, three in `_compress` and three
    in `_passthrough`:
      `provider="extractive"` -> `provider=None` (mutmut_93/_15)
                              -> `provider="XXextractiveXX"` (mutmut_100/_25)
                              -> `provider="EXTRACTIVE"` (mutmut_101/_26)

    Two calls, one that reaches `_compress`'s own `CompressionResult(...)`
    (ratio < 1.0, no full-body coverage) and one that reaches `_passthrough`
    directly (ratio >= 1.0) -- both must report the exact literal
    ``"extractive"``.
    """
    lexical_unit = CompressionUnit(
        symbol_id=1,
        body="a\nb\n\nc\nd",
        signature="s",
        docstring=None,
        line_start=1,
        line_end=5,
        qualified_name="q",
    )
    compress_result = _compressor().compress("c", lexical_unit, target_ratio=0.4)
    assert compress_result.provider == "extractive"

    passthrough_unit = CompressionUnit(
        symbol_id=1,
        body="a\nb",
        signature="s",
        docstring=None,
        line_start=1,
        line_end=2,
        qualified_name="q",
    )
    passthrough_result = _compressor().compress("x", passthrough_unit, target_ratio=1.0)
    assert passthrough_result.provider == "extractive"
