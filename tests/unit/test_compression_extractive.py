"""
Unit tests for ExtractiveCompressor (headless — no network, no inference).

Covers both scoring paths and the load-bearing invariants:
  * sub-chunk cosine path keeps the top-scored spans + always signature/docstring
  * lexical fallback path (NO sub_chunks) still compresses by query-token overlap
  * result-lossless: can't-fit degrades to signature-only, never empty/dropped
  * every kept span is a subrange of the unit's original line span
  * token_count is recomputed via tiktoken (never inherited)
  * target_ratio >= 1.0 is a byte-identical passthrough
"""

from __future__ import annotations

import sqlite3
import struct

import tiktoken

from trelix.compression import (
    CompressionResult,
    CompressionUnit,
    ExtractiveCompressor,
    make_compressor,
)
from trelix.indexing.multi_granularity import Granularity, SubSymbolChunk

_ENC = tiktoken.get_encoding("cl100k_base")


def _tok(text: str) -> int:
    return len(_ENC.encode(text))


def _assert_subranges(result: CompressionResult, line_start: int, line_end: int) -> None:
    """Every kept span must be an inclusive subrange of [line_start, line_end]."""
    assert result.kept_spans, "kept_spans must never be empty (result-lossless)"
    for a, b in result.kept_spans:
        assert line_start <= a <= b <= line_end, (
            f"span ({a},{b}) escapes original [{line_start},{line_end}]"
        )


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeDBWithSubChunks:
    """In-memory DB exposing the same surface the compressor reads."""

    def __init__(self, subs: list[SubSymbolChunk], vectors: dict[int, list[float]]) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("CREATE TABLE sub_chunks (id INTEGER PRIMARY KEY, parent_symbol_id INT)")
        self._conn.execute(
            "CREATE TABLE chunk_embeddings (chunk_id INTEGER PRIMARY KEY, embedding BLOB)"
        )
        for s in subs:
            self._conn.execute(
                "INSERT INTO sub_chunks (id, parent_symbol_id) VALUES (?, ?)",
                (s.id, s.parent_symbol_id),
            )
        # Vectors live at chunk_id = sub_chunk_id + _SUB_CHUNK_OFFSET (10_000_000).
        for sub_id, vec in vectors.items():
            blob = struct.pack(f"{len(vec)}f", *vec)
            self._conn.execute(
                "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                (sub_id + 10_000_000, blob),
            )
        self._conn.commit()
        self._subs = subs

    def get_sub_chunks_for_symbol(
        self, symbol_id: int, granularity: str | None = None
    ) -> list[SubSymbolChunk]:
        return [s for s in self._subs if s.parent_symbol_id == symbol_id]


class _NoSubChunkDB:
    """DB with no sub-chunk substrate — forces the lexical fallback path."""

    _conn = None

    def get_sub_chunks_for_symbol(
        self, symbol_id: int, granularity: str | None = None
    ) -> list[SubSymbolChunk]:
        return []


# ---------------------------------------------------------------------------
# Sub-chunk cosine path
# ---------------------------------------------------------------------------

_SUBCHUNK_BODY = (
    "def process(self, items):\n"  # rel 0 (abs 10) signature
    '    """Process items and return results."""\n'  # rel 1 (abs 11) docstring
    "    for item in items:\n"  # rel 2 (abs 12) block A ("good")
    "        total += item.value\n"  # rel 3 (abs 13) block A
    "    log.debug('a')\n"  # rel 4 (abs 14) block B ("bad", large)
    "    log.debug('b')\n"  # rel 5 (abs 15)
    "    log.debug('c')\n"  # rel 6 (abs 16)
    "    log.debug('d')\n"  # rel 7 (abs 17)
    "    log.debug('e')\n"  # rel 8 (abs 18)
    "    log.debug('f')\n"  # rel 9 (abs 19)
    "    return total"  # rel 10 (abs 20)
)


def _subchunk_unit() -> CompressionUnit:
    return CompressionUnit(
        symbol_id=42,
        body=_SUBCHUNK_BODY,
        signature="def process(self, items)",
        docstring="Process items and return results.",
        line_start=10,
        line_end=20,
        qualified_name="Processor.process",
    )


def test_sub_chunk_path_keeps_top_span_and_signature_and_docstring() -> None:
    unit = _subchunk_unit()
    subs = [
        # block A: absolute lines 12-13, vector aligned with the query -> cosine 1
        SubSymbolChunk(
            id=1,
            parent_symbol_id=42,
            granularity=Granularity.BLOCK,
            chunk_text="for item in items:\n    total += item.value",
            line_start=12,
            line_end=13,
            token_count=8,
        ),
        # block B: absolute lines 14-19, orthogonal vector -> cosine 0, and large
        SubSymbolChunk(
            id=2,
            parent_symbol_id=42,
            granularity=Granularity.BLOCK,
            chunk_text="log.debug('a')\n...\nlog.debug('f')",
            line_start=14,
            line_end=19,
            token_count=24,
        ),
    ]
    vectors = {1: [1.0, 0.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0, 0.0]}
    db = _FakeDBWithSubChunks(subs, vectors)
    compressor = ExtractiveCompressor(db=db, embedder=None)

    result = compressor.compress(
        "process the items loop",
        unit,
        target_ratio=0.5,
        query_embedding=[1.0, 0.0, 0.0, 0.0],
    )

    assert compressor.last_path == "sub_chunk"
    # Always-kept guarantees
    assert "def process" in result.text  # signature line
    assert "Process items and return results" in result.text  # docstring
    # Top-scored span (block A) kept; low-cosine large block B elided
    assert "for item in items" in result.text
    assert "log.debug" not in result.text
    assert "lines elided" in result.text
    # Compressed + recomputed token count
    assert result.token_count < result.original_token_count
    assert result.token_count == _tok(result.text)
    assert result.original_token_count == _tok(_SUBCHUNK_BODY)
    _assert_subranges(result, 10, 20)


# ---------------------------------------------------------------------------
# Lexical fallback path (no sub_chunks)
# ---------------------------------------------------------------------------

_LEXICAL_BODY = (
    "def handler(request):\n"  # rel 0  signature
    '    """Handle the incoming request."""\n'  # rel 1  docstring
    "\n"  # rel 2
    "    user = authenticate(request)\n"  # rel 3  auth segment
    "    session = create_session(user)\n"  # rel 4
    "\n"  # rel 5
    "    payment = process_payment(request)\n"  # rel 6  payment segment (target)
    "    receipt = generate_receipt(payment)\n"  # rel 7
    "\n"  # rel 8
    "    audit = write_audit_log(request)\n"  # rel 9  audit segment
    "    metrics = record_metrics(request)\n"  # rel 10
    "\n"  # rel 11
    "    return build_response(session)"  # rel 12
)


def _lexical_unit() -> CompressionUnit:
    return CompressionUnit(
        symbol_id=7,
        body=_LEXICAL_BODY,
        signature="def handler(request)",
        docstring="Handle the incoming request.",
        line_start=1,
        line_end=13,
        qualified_name="handler",
    )


def test_lexical_fallback_path_compresses_by_overlap() -> None:
    db = _NoSubChunkDB()
    compressor = ExtractiveCompressor(db=db, embedder=None)
    unit = _lexical_unit()

    # No query_embedding + no sub_chunks -> lexical path. Query targets payment.
    result = compressor.compress(
        "payment processing receipt",
        unit,
        target_ratio=0.5,
        query_embedding=None,
    )

    assert compressor.last_path == "lexical"
    # Signature + docstring always kept
    assert "def handler" in result.text
    assert "Handle the incoming request" in result.text
    # Highest query-token-overlap segment kept
    assert "process_payment" in result.text or "generate_receipt" in result.text
    # A lower-overlap segment was dropped -> compression happened
    assert "lines elided" in result.text
    assert result.token_count < result.original_token_count
    assert result.token_count == _tok(result.text)
    _assert_subranges(result, 1, 13)


def test_lexical_path_also_used_when_query_embedding_present_but_no_sub_chunks() -> None:
    # A query embedding is available but the index has NO sub_chunks -> still lexical.
    db = _NoSubChunkDB()
    compressor = ExtractiveCompressor(db=db, embedder=None)
    result = compressor.compress(
        "payment processing receipt",
        _lexical_unit(),
        target_ratio=0.5,
        query_embedding=[0.1, 0.2, 0.3, 0.4],
    )
    assert compressor.last_path == "lexical"
    assert result.token_count == _tok(result.text)


# ---------------------------------------------------------------------------
# Result-lossless: can't-fit -> signature-only, never empty/dropped
# ---------------------------------------------------------------------------


def test_result_lossless_signature_only_when_budget_tiny() -> None:
    unit = CompressionUnit(
        symbol_id=99,
        body=(
            "def tiny_budget_symbol(a, b, c):\n"  # rel 0 signature (ends ':')
            "    x = compute(a)\n"  # rel 1
            "    y = compute(b)\n"  # rel 2
            "    z = compute(c)\n"  # rel 3
            "    return x + y + z"  # rel 4
        ),
        signature="def tiny_budget_symbol(a, b, c)",
        docstring=None,  # no docstring -> must-keep is the signature line only
        line_start=100,
        line_end=104,
        qualified_name="tiny_budget_symbol",
    )
    db = _NoSubChunkDB()
    compressor = ExtractiveCompressor(db=db, embedder=None)

    result = compressor.compress("anything", unit, target_ratio=0.01, query_embedding=None)

    assert result is not None
    assert result.text.strip() != ""  # never empty
    assert "def tiny_budget_symbol" in result.text  # signature preserved
    # Degraded to the signature line only
    assert result.kept_spans == [(100, 100)]
    assert result.token_count == _tok(result.text)
    assert result.token_count > 0
    _assert_subranges(result, 100, 104)


# ---------------------------------------------------------------------------
# target_ratio >= 1.0 passthrough (byte-identical)
# ---------------------------------------------------------------------------


def test_target_ratio_one_is_identical_passthrough() -> None:
    unit = _lexical_unit()
    db = _NoSubChunkDB()
    compressor = ExtractiveCompressor(db=db, embedder=None)

    result = compressor.compress("payment", unit, target_ratio=1.0, query_embedding=None)

    assert compressor.last_path == "passthrough"
    assert result.text == unit.body  # byte-identical
    assert result.token_count == result.original_token_count == _tok(unit.body)
    assert result.kept_spans == [(1, 13)]


def test_target_ratio_above_one_is_identical_passthrough() -> None:
    unit = _subchunk_unit()
    db = _NoSubChunkDB()
    compressor = ExtractiveCompressor(db=db, embedder=None)
    result = compressor.compress("x", unit, target_ratio=2.0, query_embedding=None)
    assert result.text == unit.body
    assert result.token_count == result.original_token_count


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_make_compressor_defaults_to_extractive() -> None:
    db = _NoSubChunkDB()
    comp = make_compressor(config=None, db=db, embedder=None)
    assert isinstance(comp, ExtractiveCompressor)


def test_make_compressor_rejects_unknown_provider() -> None:
    class _Cfg:
        class compression:  # noqa: N801 — mimic pydantic sub-config attribute access
            provider = "abstractive"

    raised = False
    try:
        make_compressor(config=_Cfg(), db=_NoSubChunkDB(), embedder=None)
    except NotImplementedError:
        raised = True
    assert raised, "unknown provider must raise NotImplementedError (reserved v3.4)"
