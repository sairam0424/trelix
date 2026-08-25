"""
Mutation-killing tests for ExtractiveCompressor's smaller internal helpers --
``__init__``, ``_log_path_once``, ``_brace_balance``, ``_signature_end_idx``,
``_span_tokens``, ``_sub_to_span``, ``_segments``, ``_has_sub_chunks``,
``_fetch_sub_chunk_vectors``, ``_score_sub_chunks``, ``_score_lexical``, and the
module-level ``_locate_text``.

Split out from ``test_compression_extractive_mutants.py`` (which already covers
``compress``/``_compress``-level behaviour) purely to keep each file within a
readable size -- both files exercise the same module.

Every fixture below was checked against mutmut's own generated mutant variants
for ``src/trelix/compression/extractive.py`` (round-11 measurement) before being
written down: each docstring's claimed mutation was reproduced against the
actual mutant code and shown to diverge from the unmutated baseline.

EQUIVALENT MUTANTS covered by this file's methods, deliberately NOT tested
(rule 7) -- each verified empirically against mutmut's generated variant, not
guessed:
  __init__ mutmut_2       `self._embedder = embedder` -> `= None`. Grepped the
                          whole src tree: no code anywhere ever reads
                          `ExtractiveCompressor._embedder` (it is stored "for
                          reserved future use / trace" per its own comment).
                          No test can observe a difference without adding a
                          getter that does not exist.
  _brace_balance mutmut_3 `"\n".join(...)` -> `"XX\nXX".join(...)`. The
                          function only counts `"{"`/`"}"` characters in the
                          joined text; the separator string contains neither,
                          so its own content can never change the count for
                          ANY input.
  _clamp_spans mutmut_3   `hi = line_end if line_end >= lo else lo` -> `>`.
                          The only point where `>=` and `>` disagree is
                          `line_end == lo`, and BOTH branches evaluate to
                          `lo` there -- same value either way, for every input.
  _fetch_sub_chunk_vectors
    mutmut_9              `if conn is None or not sub_chunk_ids: return {}`
                          -> `and`. Checked exhaustively over all four
                          conn/ids truth combinations: whenever the mutant's
                          `and` fails to return `{}` early, the loop body
                          itself produces `{}` anyway (empty `sub_chunk_ids`
                          never iterates; `conn is None` raises inside the
                          per-item `try/except Exception: return out`,
                          which is the SAME empty accumulator).
    mutmut_19, 20, 31, 33, 40, 42
                          SQL keyword/identifier casing (`SELECT`->`select`,
                          `CHUNK_ID`->`chunk_id` uppercased) and dropped
                          `dtype=np.float64` kwargs on `np.asarray(...)`.
                          Verified against a real in-memory sqlite connection:
                          identifiers are matched case-insensitively, and
                          `np.asarray` infers `float64` from a list/tuple of
                          Python floats even without an explicit `dtype=`.
  _has_sub_chunks mutmut_16, 17
                          Same SQL-case reasoning as above, verified against a
                          real `sub_chunks` table WITH a row present (so the
                          query's true/false answer, not just "did it error",
                          is what is being compared).
  _passthrough mutmut_9   `n_lines > 0` -> `>= 0`. The two branches disagree
                          only at `n_lines == 0`, where the IF-branch gives
                          `line_start - 1` and the ELSE-branch gives
                          `line_start` -- but `_clamp_spans` (called on the
                          result) clamps any value below `line_start` back UP
                          to `line_start`, so the two feed identical
                          `kept_spans` regardless.
  _passthrough mutmut_10  `n_lines > 0` -> `> 1`. The two branches disagree
                          only at `n_lines == 1`, where BOTH formulas
                          (`line_start + n_lines - 1` and `line_start`)
                          evaluate to the same `line_start` by construction.
  _passthrough mutmut_20  `provider="extractive"` kwarg dropped entirely.
                          `CompressionResult.provider` defaults to the
                          literal `"extractive"` at the dataclass level, so
                          omitting the kwarg reaches the same value.
  _render mutmut_8, 9, 10 `if first_start > 0: ...` and its body. `_render`
                          is called ONLY from `_compress`, always with a
                          `merged` list whose first span always starts at
                          literal index 0 (the unconditional must-keep span
                          is always `(0, ...)`, and `_merge_spans` cannot move
                          that start upward) -- so `first_start > 0` is
                          always False and this whole branch is dead code
                          for every reachable input.
  _render mutmut_16       `if gap > 0: ...`. `_merge_spans` only leaves TWO
                          spans adjacent in its output when they could not be
                          merged, which by its own merge rule (`s <= le + 1`)
                          means the gap between them is always >= 1 -- `gap`
                          can never be 0 for any `merged` list `_render`
                          actually receives.
  _score_lexical mutmut_12
                          `body_lines[i0 : i1 + 1]` -> `[i0 : i1 + 2]`. Every
                          `(i0, i1)` `_score_lexical` is ever called with
                          comes from `self._segments(...)`, whose own
                          construction guarantees the line right after `i1`
                          is either a blank separator (contributes no tokens)
                          or past the end of the body (Python slicing
                          tolerates it) -- the one extra line the mutant
                          reads can never carry a real token.
  _score_sub_chunks mutmut_27, 29
                          Dropped/`None`ed `dtype=np.float64` on
                          `np.asarray(query_embedding, ...)`. Same numpy
                          dtype-inference reasoning as the fetch cluster
                          above -- verified with a real query embedding.
"""

from __future__ import annotations

import logging
import sqlite3
import struct

import pytest

from trelix.compression import CompressionUnit, ExtractiveCompressor
from trelix.compression.extractive import _locate_text


class _NoSubChunkDB:
    """No sub-chunk substrate -- plain class, not a Mock (rule 3)."""

    _conn = None

    def get_sub_chunks_for_symbol(self, symbol_id: int, granularity: str | None = None) -> list:
        return []


def _compressor() -> ExtractiveCompressor:
    return ExtractiveCompressor(db=_NoSubChunkDB())


# ---------------------------------------------------------------------------
# module-level _locate_text
# ---------------------------------------------------------------------------


def test_locate_text_skips_a_blank_line_without_stopping_the_scan() -> None:
    """Kills: `if not stripped: continue` -> `break` in `_locate_text`.

    A blank line in the middle of the body must be SKIPPED, not treated as the
    end of the search -- otherwise every docstring line after the first blank
    line in the body becomes unmatchable.
    """
    body_lines = ["auth", "", "login user"]
    # Precondition: there must be a blank line strictly between two wanted
    # lines, or `break` and `continue` would agree trivially.
    assert body_lines[1] == "", (
        "fixture must have a blank separator to discriminate break vs continue"
    )
    hit = _locate_text(body_lines, "auth\nlogin user")
    assert hit == (0, 2)


def test_locate_text_matches_a_body_line_that_is_a_strict_substring_of_the_wanted_text() -> None:
    """Kills: `stripped == w or stripped in w or w in stripped` ->
    `(stripped == w and stripped in w) or w in stripped`.

    `stripped == w and stripped in w` collapses to just `stripped == w` (equality
    implies containment), silently dropping the "stripped is a substring of a
    longer wanted line" case. A body line reading "authenticate" must still
    match a wanted docstring line "authenticate the user".
    """
    body_lines = ["authenticate", "unrelated"]
    wanted = "authenticate the user"
    # Precondition: the body line is a strict substring of `wanted`, not equal
    # to it and not the other way round -- the only case the mutant drops.
    assert body_lines[0] != wanted and body_lines[0] in wanted and wanted not in body_lines[0], (
        "fixture no longer isolates the dropped 'stripped in w' disjunct"
    )
    hit = _locate_text(body_lines, wanted)
    assert hit == (0, 0)


# ---------------------------------------------------------------------------
# __init__ / _log_path_once
# ---------------------------------------------------------------------------


def test_compressor_starts_with_logged_path_false_and_last_path_none() -> None:
    """Kills three __init__ mutants at once, all via strict identity (`is`):
      `self._logged_path = False` -> `= None` (mutmut_8) or `= True` (mutmut_9)
      `self.last_path: str | None = None` -> `= ""` (mutmut_10)

    `is False`/`is None` are used deliberately: `None` and `False` are both
    falsy under `not self._logged_path`, so a plain truthiness assertion would
    not distinguish mutmut_8 from the real value.
    """
    compressor = _compressor()
    assert compressor._logged_path is False
    assert compressor.last_path is None


def test_log_path_once_logs_exactly_once_across_repeated_calls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Kills three `_log_path_once` mutants:
      `if not self._logged_path:` -> `if self._logged_path:` (mutmut_1)
      `self._logged_path = True` -> `= None` (mutmut_2) or `= False` (mutmut_3)

    mutmut_1 never logs (the guard is inverted); mutmut_2/3 leave the flag
    falsy after the first call, so a SECOND call logs again. The real
    contract is exactly one INFO line no matter how many times the path is
    resolved for the same compressor instance.
    """
    compressor = _compressor()
    with caplog.at_level(logging.INFO, logger="trelix.compression.extractive"):
        compressor._log_path_once("lexical")
        compressor._log_path_once("lexical")
    assert len(caplog.records) == 1


# ---------------------------------------------------------------------------
# _brace_balance
# ---------------------------------------------------------------------------


def test_brace_balance_counts_over_the_declared_line_range_only() -> None:
    """Kills: `body_lines[i0 : i1 + 1]` -> `body_lines[i0 : i1 + 2]`.

    Fixture: an open brace at index 0 with a lone closing brace at index 1,
    OUTSIDE the declared range (0, 0). The declared range alone is unbalanced
    (+1); pulling in one extra line wrongly balances it to 0.
    """
    lines = ["{", "}"]
    compressor = _compressor()
    assert compressor._brace_balance(lines, 0, 0) == 1


# ---------------------------------------------------------------------------
# _signature_end_idx
# ---------------------------------------------------------------------------


def test_signature_end_idx_only_scans_the_first_eight_lines() -> None:
    """Kills: `body_lines[:8]` -> `body_lines[:9]`.

    A header-terminating line placed at index 8 (the NINTH line) must be
    invisible to the scan window; the function must fall back to the default
    0, not report index 8.
    """
    lines = ["1", "2", "3", "4", "5", "6", "7", "8", "9:"]
    # Precondition: the ONLY line ending in ":" or "{" sits at index 8.
    assert lines[8].endswith(":") and not any(ln.endswith((":", "{")) for ln in lines[:8]), (
        "fixture's header-terminating line must sit at index 8 exclusively"
    )
    assert ExtractiveCompressor._signature_end_idx(lines) == 0


def test_signature_end_idx_uses_rstrip_not_lstrip() -> None:
    """Kills: `line.rstrip().endswith(...)` -> `line.lstrip().endswith(...)`.

    A header line with TRAILING whitespace must still be recognised. Placing
    it at index 1 (not 0) means a wrongly-lstripped scan that fails to match
    it falls through to the unrelated default (0), not accidentally landing
    on the right answer by coincidence.
    """
    lines = ["comment line", "def f():   ", "pass"]
    # Precondition: the header line has trailing whitespace after the colon,
    # and it is NOT the first line -- both needed to discriminate lstrip from
    # rstrip (see module docstring for the coincidence lstrip/rstrip share at
    # index 0).
    assert lines[1].endswith(" ") and lines[1].rstrip().endswith(":"), (
        "fixture's header line must have trailing whitespace after ':'"
    )
    assert ExtractiveCompressor._signature_end_idx(lines) == 1


def test_signature_end_idx_defaults_to_zero_without_a_header_line() -> None:
    """Kills: `return 0` -> `return 1` (the no-match fallback)."""
    lines = ["a", "b", "c"]
    assert not any(ln.endswith((":", "{")) for ln in lines), (
        "fixture must contain no header terminator"
    )
    assert ExtractiveCompressor._signature_end_idx(lines) == 0


# ---------------------------------------------------------------------------
# _span_tokens
# ---------------------------------------------------------------------------


def test_span_tokens_of_no_spans_is_zero() -> None:
    """Kills: `if not spans: return 0` -> `return 1`."""
    compressor = _compressor()
    assert compressor._span_tokens(["a"], []) == 0


# ---------------------------------------------------------------------------
# _sub_to_span
# ---------------------------------------------------------------------------


class _Sub:
    def __init__(self, line_start: int, line_end: int) -> None:
        self.line_start = line_start
        self.line_end = line_end


def _unit_for_sub_to_span() -> CompressionUnit:
    return CompressionUnit(
        symbol_id=1,
        body="x",
        signature="x",
        docstring=None,
        line_start=10,
        line_end=15,
        qualified_name="q",
    )


def test_sub_to_span_floors_at_zero_not_one() -> None:
    """Kills: `i0 = max(0, ...)` -> `max(1, ...)`.

    A sub-chunk whose declared start equals the unit's own `line_start`
    (diff == 0) must map to body-relative index 0, not 1 -- `max(1, ...)`
    would silently drop the first line of the sub-chunk's own span.
    """
    compressor = _compressor()
    unit = _unit_for_sub_to_span()
    span = compressor._sub_to_span(_Sub(line_start=10, line_end=12), unit, n=5)
    assert span == (0, 2)


def test_sub_to_span_ceiling_is_n_minus_one_not_n_plus_one() -> None:
    """Kills: `i1 = min(n - 1, ...)` -> `min(n + 1, ...)`.

    diff == n (one past the last valid body index) must clamp DOWN to n - 1;
    `n + 1` would let an out-of-range index escape uncapped.
    """
    compressor = _compressor()
    unit = _unit_for_sub_to_span()
    n = 5
    span = compressor._sub_to_span(_Sub(line_start=10, line_end=15), unit, n)  # diff = 5 = n
    assert span == (0, 4)


def test_sub_to_span_ceiling_is_n_minus_one_not_n_minus_two() -> None:
    """Kills: `i1 = min(n - 1, ...)` -> `min(n - 2, ...)`.

    diff == n - 1 (exactly the last valid body index) must be kept whole, not
    shortened by one extra line.
    """
    compressor = _compressor()
    unit = _unit_for_sub_to_span()
    n = 5
    span = compressor._sub_to_span(_Sub(line_start=10, line_end=14), unit, n)  # diff = 4 = n - 1
    assert span == (0, 4)


def test_sub_to_span_accepts_a_single_line_span() -> None:
    """Kills: `if i1 < i0:` -> `if i1 <= i0:`.

    A single-line sub-chunk (i0 == i1) is VALID, not malformed -- `<=` would
    wrongly reject every single-line sub-chunk as `None`.
    """
    compressor = _compressor()
    unit = _unit_for_sub_to_span()
    span = compressor._sub_to_span(_Sub(line_start=12, line_end=12), unit, n=5)
    assert span == (2, 2)


# ---------------------------------------------------------------------------
# _segments
# ---------------------------------------------------------------------------


def test_segments_brace_balance_scans_only_the_current_raw_segments_own_lines() -> None:
    """Kills three `_segments` mutants at once, all in the OUTER per-segment
    balance computation / bookkeeping:
      `balance = self._brace_balance(body_lines, s, e)` -> `(body_lines, None, e)` (mutmut_24)
      `while balance > 0 and j + 1 < len(raw):` -> `j - 1 < len(raw)` (mutmut_33,
        raises IndexError once `j` runs past `len(raw)`)
      `while balance > 0 and j + 1 < len(raw):` -> `j + 1 <= len(raw)` (mutmut_35,
        same IndexError failure mode)
      `j += 1` -> `j += 2` (mutmut_38, skips a raw segment and also IndexErrors
        once `j` runs past `len(raw)`)

    Fixture: three blank-line-delimited segments -- a lone unmatched `}` (index
    0), a lone unmatched `{` (index 2), and a harmless line (index 4). The
    real algorithm keeps segment 0 standalone (its own balance is 0) and
    merges segments 1+2 (segment 1's unmatched `{` never closes). Slicing
    from `None` for a LATER segment's start silently reintroduces segment 0's
    stray `}`, cancelling out segment 1's real imbalance.
    """
    compressor = _compressor()
    lines = ["}", "", "{", "", "x"]
    assert compressor._segments(lines) == [(0, 0), (2, 4)]


def test_segments_inner_accumulation_starts_from_each_raw_segments_own_bounds() -> None:
    """Kills: `balance += self._brace_balance(body_lines, raw[j][0], raw[j][1])`
    -> `(body_lines, None, raw[j][1])` (mutmut_44).

    Fixture: segment 0 = lone `{` (unmatched, balance +1), segment 1 = lone
    `}` (closes it, balance -1), segment 2 = harmless `z`. The real merge
    stops after segment 1 (balance returns to 0), leaving segment 2 standalone.
    Slicing the INNER accumulation from `None` (i.e. from line 0) re-adds
    segment 0's own `{` into segment 1's contribution, so the running balance
    never returns to 0 and segment 2 gets wrongly swallowed into the merge.
    """
    compressor = _compressor()
    lines = ["{", "", "}", "", "z"]
    assert compressor._segments(lines) == [(0, 2), (4, 4)]


def test_segments_inner_accumulation_uses_the_raw_segments_full_range_not_just_its_last_line() -> (
    None
):
    """Kills: `balance += self._brace_balance(body_lines, raw[j][0], raw[j][1])`
    -> `(body_lines, raw[j][1], raw[j][1])` (mutmut_49).

    Fixture: segment 0 = lone `{` (unmatched, +1), segment 1 = `{` then `}`
    on two lines (net 0 over its own full range, but its LAST line alone is a
    lone `}`, i.e. -1), segment 2 = harmless `z`. The real algorithm counts
    segment 1's FULL range (net 0), so the running balance stays at +1 and
    segment 2 gets pulled into the merge too. Counting only segment 1's last
    line (-1) makes the running balance hit 0 early and segment 2 stays
    standalone.
    """
    compressor = _compressor()
    lines = ["{", "", "{", "}", "", "z"]
    assert compressor._segments(lines) == [(0, 5)]


# ---------------------------------------------------------------------------
# _has_sub_chunks
# ---------------------------------------------------------------------------


def test_has_sub_chunks_initial_value_is_exactly_false_when_conn_is_none() -> None:
    """Kills: `available = False` -> `= None` (mutmut_10) or `= True` (mutmut_11).

    With `conn is None`, the `if conn is not None:` guard is skipped entirely,
    so the function returns whatever `available` was initialised to. `is False`
    (not truthiness) is required to catch `None`, which is equally falsy.
    """
    compressor = _compressor()
    result = compressor._has_sub_chunks()
    assert result is False


class _RaisingConn:
    def execute(self, sql: str) -> None:
        raise sqlite3.OperationalError("boom")


class _DBWithRaisingConn:
    _conn = _RaisingConn()

    def get_sub_chunks_for_symbol(self, symbol_id: int, granularity: str | None = None) -> list:
        return []


def test_has_sub_chunks_exception_path_reports_exactly_false() -> None:
    """Kills: `except Exception: available = False` -> `= None` (mutmut_19)
    or `= True` (mutmut_20).

    A connection whose query genuinely raises must be reported as "no
    sub-chunk substrate" (`False`), not `None` or `True`.
    """
    compressor = ExtractiveCompressor(db=_DBWithRaisingConn())
    result = compressor._has_sub_chunks()
    assert result is False


def test_has_sub_chunks_caches_the_computed_boolean() -> None:
    """Kills: `self._sub_chunks_available = available` -> `= None`.

    Forcing the cache to `None` (mutmut_21) makes it indistinguishable from
    "never computed", defeating the whole point of caching: every subsequent
    call would re-run the query.
    """
    compressor = _compressor()
    compressor._has_sub_chunks()
    assert compressor._sub_chunks_available is False


# ---------------------------------------------------------------------------
# _fetch_sub_chunk_vectors
# ---------------------------------------------------------------------------


class _CompletelyMinimalDB:
    """Defines neither `_conn` nor `get_sub_chunks_for_symbol` at all -- the
    leanest possible db-shaped object, distinguishing a dropped `getattr`
    default (which raises on a missing attribute) from the real default
    (which degrades gracefully)."""


def test_internal_getattr_defaults_tolerate_a_db_missing_the_attribute_entirely() -> None:
    """Kills three dropped-default `getattr` mutants at once:
      `getattr(self._db, "_conn", None)` in `_has_sub_chunks` (mutmut_7)
      `getattr(self._db, "_conn", None)` in `_fetch_sub_chunk_vectors` (mutmut_6)
      `getattr(self._db, "get_sub_chunks_for_symbol", None)` in
        `_score_sub_chunks` (mutmut_6)

    Each dropped default turns a graceful `getattr(obj, name, default)` into a
    plain 2-arg lookup that RAISES `AttributeError` on an object that simply
    does not define the attribute -- a real shape for a minimal db backend.
    """
    db = _CompletelyMinimalDB()
    compressor = ExtractiveCompressor(db=db)
    assert compressor._has_sub_chunks() is False
    assert compressor._fetch_sub_chunk_vectors([1, 2]) == {}
    unit = CompressionUnit(
        symbol_id=1,
        body="x",
        signature="x",
        docstring=None,
        line_start=1,
        line_end=1,
        qualified_name="q",
    )
    assert compressor._score_sub_chunks(unit, ["x"], [1.0]) is None


class _RowMissingCursor:
    def __init__(self, row: tuple | None) -> None:
        self._row = row

    def fetchone(self) -> tuple | None:
        return self._row


class _RowLookupConn:
    def __init__(self, rows: dict[int, tuple]) -> None:
        self._rows = rows

    def execute(self, sql: str, params: tuple) -> _RowMissingCursor:
        (chunk_id,) = params
        return _RowMissingCursor(self._rows.get(chunk_id))


class _DBWithRowLookupConn:
    def __init__(self, conn: _RowLookupConn) -> None:
        self._conn = conn


def test_fetch_sub_chunk_vectors_skips_a_missing_row_without_stopping_the_scan() -> None:
    """Kills: `if row is None: continue` -> `break`.

    id 2 has no stored row (a genuinely missing sub-chunk vector); id 3, which
    comes AFTER it, must still be fetched. `break` would silently drop id 3
    too, just because id 2 happened to come first.
    """
    conn = _RowLookupConn(
        {
            10_000_001: (struct.pack("2f", 1.0, 2.0),),
            10_000_003: (struct.pack("2f", 5.0, 6.0),),
        }
    )
    compressor = ExtractiveCompressor(db=_DBWithRowLookupConn(conn))
    vectors = compressor._fetch_sub_chunk_vectors([1, 2, 3])
    assert set(vectors.keys()) == {1, 3}
    assert vectors[1].tolist() == [1.0, 2.0]
    assert vectors[3].tolist() == [5.0, 6.0]


class _RealSqliteVectorDB:
    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute(
            "CREATE TABLE chunk_embeddings (chunk_id INTEGER PRIMARY KEY, embedding BLOB)"
        )
        blob = struct.pack("3f", 7.0, 8.0, 9.0)
        self._conn.execute("INSERT INTO chunk_embeddings VALUES (?, ?)", (10_000_002, blob))
        self._conn.commit()


def test_fetch_sub_chunk_vectors_unpacks_the_real_bytes_not_a_placeholder() -> None:
    """Kills: `vec = np.asarray(struct.unpack(f"{count}f", blob), dtype=np.float64)`
    -> `np.asarray(None, dtype=np.float64)`.

    The stored BLOB's real floats (7.0, 8.0, 9.0) must come back verbatim, not
    a scalar NaN placeholder.
    """
    compressor = ExtractiveCompressor(db=_RealSqliteVectorDB())
    vectors = compressor._fetch_sub_chunk_vectors([2])
    assert vectors[2].tolist() == [7.0, 8.0, 9.0]


class _ListBlobCursor:
    def __init__(self, row: tuple | None) -> None:
        self._row = row

    def fetchone(self) -> tuple | None:
        return self._row


class _ListBlobConn:
    """A connection whose stored 'blob' is a plain list, not bytes -- the
    non-bytes branch of `_fetch_sub_chunk_vectors` (`else: vec =
    np.asarray(list(blob), ...)`), which real sqlite BLOB storage never
    reaches (sqlite3 always returns `bytes` for BLOB columns) but the method
    still guards against."""

    def __init__(self, rows: dict[int, tuple]) -> None:
        self._rows = rows

    def execute(self, sql: str, params: tuple) -> _ListBlobCursor:
        (chunk_id,) = params
        return _ListBlobCursor(self._rows.get(chunk_id))


class _DBWithListBlobConn:
    def __init__(self, conn: _ListBlobConn) -> None:
        self._conn = conn


def test_fetch_sub_chunk_vectors_else_branch_handles_a_non_bytes_blob() -> None:
    """Kills four mutants in the non-bytes ("else") branch at once:
      `vec = np.asarray(list(blob), dtype=np.float64)`
        -> `vec = None`                                (mutmut_38)
        -> `vec = np.asarray(None, dtype=np.float64)`   (mutmut_39)
        -> `vec = np.asarray(dtype=np.float64)`         (mutmut_41, TypeError:
           missing required positional argument)
        -> `vec = np.asarray(list(None), dtype=np.float64)` (mutmut_43,
           TypeError: 'NoneType' object is not iterable)

    Any of the four either crashes (killed by an uncaught exception) or
    silently replaces the real stored values.
    """
    conn = _ListBlobConn({10_000_007: ([1.0, 2.0, 3.0],)})
    compressor = ExtractiveCompressor(db=_DBWithListBlobConn(conn))
    vectors = compressor._fetch_sub_chunk_vectors([7])
    assert vectors[7].tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# _score_sub_chunks
# ---------------------------------------------------------------------------


class _SubWithNoIdAttribute:
    """Duck-typed sub-chunk object with no `id` attribute at all."""

    line_start = 2
    line_end = 2


class _MixedSubsDB:
    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute(
            "CREATE TABLE chunk_embeddings (chunk_id INTEGER PRIMARY KEY, embedding BLOB)"
        )
        blob = struct.pack("2f", 1.0, 0.0)
        self._conn.execute("INSERT INTO chunk_embeddings VALUES (?, ?)", (10_000_010, blob))
        self._conn.commit()

    def get_sub_chunks_for_symbol(self, symbol_id: int, granularity: str | None = None) -> list:
        class GoodSub:
            id = 10
            line_start = 2
            line_end = 2

        return [GoodSub(), _SubWithNoIdAttribute()]


def test_score_sub_chunks_getattr_id_default_tolerates_a_sub_missing_the_attribute() -> None:
    """Kills two dropped-default `getattr` mutants for the "id" attribute:
      `ids = [s.id for s in subs if getattr(s, "id", None) is not None]` (mutmut_18)
      `sub_id = getattr(sub, "id", None)` in the scoring loop (mutmut_42)

    A duck-typed sub-chunk object entirely missing `id` must be gracefully
    filtered out (not crash), while a well-formed sibling is still scored.
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
    compressor = ExtractiveCompressor(db=_MixedSubsDB())
    scored = compressor._score_sub_chunks(unit, unit.body.splitlines(), [1.0, 0.0])
    assert scored == [((1, 1), 1.0)]


class _FiveItemSubsDB:
    """Good1, then three DIFFERENT failure modes, then Good2 -- so that a
    `continue`-turned-`break` at any ONE failure point is caught by Good2
    never getting scored."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute(
            "CREATE TABLE chunk_embeddings (chunk_id INTEGER PRIMARY KEY, embedding BLOB)"
        )
        blob = struct.pack("2f", 1.0, 0.0)
        for sid in (1, 4, 5):
            self._conn.execute(
                "INSERT INTO chunk_embeddings VALUES (?, ?)", (sid + 10_000_000, blob)
            )
        self._conn.commit()

    def get_sub_chunks_for_symbol(self, symbol_id: int, granularity: str | None = None) -> list:
        class Good1:
            id = 1
            line_start = 1
            line_end = 1

        class BadSubIdNone:
            id = None  # triggers `if sub_id is None:`
            line_start = 2
            line_end = 2

        class BadVecMissing:
            id = 3  # no stored vector -> triggers `if vec is None:`
            line_start = 3
            line_end = 3

        class BadSpanNone:
            id = 4  # has a vector, but an invalid line range -> `if span is None:`
            line_start = 100
            line_end = 50

        class Good2:
            id = 5
            line_start = 4
            line_end = 4

        return [Good1(), BadSubIdNone(), BadVecMissing(), BadSpanNone(), Good2()]


def test_score_sub_chunks_skips_bad_items_without_stopping_the_scan() -> None:
    """Kills three `continue`-turned-`break` mutants in `_score_sub_chunks`'s
    main scoring loop:
      `if sub_id is None: continue` -> `break` (mutmut_46)
      `if vec is None: continue` -> `break` (mutmut_51)
      `if span is None: continue` -> `break` (mutmut_60)

    Five sub-chunks: a good one, then three that each fail a DIFFERENT one of
    the three checks above, then a final good one. Any one `break` stops the
    scan before the trailing good item is ever reached; `continue` reaches it.
    """
    unit = CompressionUnit(
        symbol_id=1,
        body="l0\nl1\nl2\nl3\nl4",
        signature="s",
        docstring=None,
        line_start=0,
        line_end=4,
        qualified_name="q",
    )
    compressor = ExtractiveCompressor(db=_FiveItemSubsDB())
    scored = compressor._score_sub_chunks(unit, unit.body.splitlines(), [1.0, 0.0])
    assert scored == [((1, 1), 1.0), ((4, 4), 1.0)]


class _OneSubDB:
    _conn = None

    def get_sub_chunks_for_symbol(self, symbol_id: int, granularity: str | None = None) -> list:
        class Sub:
            id = 1
            line_start = 2
            line_end = 2

        return [Sub()]


def test_score_sub_chunks_uses_the_real_query_embedding_not_a_placeholder() -> None:
    """Kills: `q = np.asarray(query_embedding, dtype=np.float64)` ->
    `np.asarray(None, dtype=np.float64)`.

    `np.asarray(None, dtype=np.float64)` is a 0-D `nan` scalar, whose `.shape`
    (`()`) never matches a real stored vector's shape -- `_cosine`'s shape
    guard then scores EVERY sub-chunk 0.0, regardless of true alignment. A
    perfectly-aligned vector must score 1.0, not 0.0.
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
    compressor = ExtractiveCompressor(db=_OneSubDB())
    compressor._fetch_sub_chunk_vectors = lambda ids: {1: __import__("numpy").array([1.0, 0.0])}
    scored = compressor._score_sub_chunks(unit, unit.body.splitlines(), [1.0, 0.0])
    assert scored == [((1, 1), 1.0)]


# ---------------------------------------------------------------------------
# _score_lexical
# ---------------------------------------------------------------------------


def test_score_lexical_falls_back_to_an_empty_query_token_set_not_a_placeholder() -> None:
    """Kills: `query_tokens = {t.lower() for t in _WORD_RE.findall(query or "")}`
    -> `findall(query or "XXXX")`.

    With a falsy `query` (empty string), the token set must stay EMPTY. The
    body deliberately contains the literal word "xxxx" so the placeholder
    fallback would spuriously create a matching token and score above zero.
    """
    compressor = _compressor()
    lines = ["hello", "xxxx"]
    scored = compressor._score_lexical("", lines)
    assert scored == [((0, 1), 0.0)]


def test_score_lexical_segment_text_join_separator_is_a_bare_newline() -> None:
    """Kills: `seg_text = "\\n".join(body_lines[i0 : i1 + 1])` ->
    `"XX\\nXX".join(...)`.

    A two-line segment joined with `"XX\\nXX"` glues "XX" onto the adjacent
    words with no whitespace between them (e.g. "foo" + "XX\\nXX" + "bar" ->
    "fooXX\\nXXbar"), so the word-boundary regex tokenizes "fooXX"/"XXbar"
    instead of "foo"/"bar" -- silently changing which query tokens match.
    """
    compressor = _compressor()
    lines = ["foo", "bar"]
    scored = compressor._score_lexical("foo bar", lines)
    # Precondition + expected value, hand-derived from the documented rule
    # (overlap / sqrt(unique_seg_tokens + 1)): with the real "\n" separator,
    # {foo, bar} both match the query -> overlap 2, 2 unique tokens.
    assert scored == [((0, 1), pytest.approx(2 / 3**0.5, rel=1e-12))]
