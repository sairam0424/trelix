"""Four more `retrieval/retriever.py` survivors, RE-DERIVED by mutation on this tree.

WHY RE-DERIVED RATHER THAN INHERITED. Round 4 handed down a list of 14 retriever
findings and round 5 closed 8 of them, but two entries on that list turned out not to
be survivors at all (M23 was never one; M43's twin was already dead). A handed-down
survivor list is a PROXY for a mutation run, so every mutation below was applied ALONE
to `src/trelix/retrieval/retriever.py`, with the exact anchor text verified to occur
EXACTLY ONCE, and reproduced as a survivor against the 665-test retriever-adjacent
suite (test_assembler{,_compression,_backcompat_golden} / test_config{,_weight_parsing}
/ test_fusion / test_graph_{pagerank,persistence,search} / test_model_aware_budget /
test_rerank_outcome / test_reranker_core / test_retrieval_breadth_floor /
test_retriever_{budget_and_ranking_knobs,core,file_summary,leg_gates_and_dedup,
leg_signals}) BEFORE any test here existed.

    C2  `sid = r.chunk.symbol_id` -> `sid = r.chunk.id`            in `_dedup`
    L   `symbol_id=-(file_id)`    -> `symbol_id=file_id`           in `_summary_search`
    I   `min(symbols, key=...)`   -> `max(symbols, key=...)`       in `_summary_search`
    F   `cfg.leg_weights.get("grep", 1.0)` -> `1.0`                in `_standard_candidates`

WHY C2 AND L ARE ONE FAMILY. Both are the ROW-IDENTITY key `_dedup` reduces on. `_dedup`
is the last thing standing between the six fused legs plus four graph-expansion tails
and the assembler's token budget, and its promise is one row PER SYMBOL. C2 changes what
"one row" means; L makes two different things claim the same identity. A wrong identity
here does not raise: it silently either burns budget on a duplicate or deletes a correct
row, which is the failure mode fusion.py's own docstring calls out ("any further pass can
only delete correct rows, which is exactly how a whole repo went missing").

WHY C2 SURVIVED, NAMED PRECISELY -- this is the interesting part. `test_retriever_core.py`
DOES have `TestDedup::test_dedup_removes_duplicate_symbol_ids`. It cannot kill C2 because
its own helper builds chunks as `Chunk(symbol_id=sym_id, ..., id=sym_id)` -- chunk.id and
symbol_id are EQUAL BY CONSTRUCTION, so keying on either passes.
`test_retriever_leg_gates_and_dedup.py`'s `_result()` helper does the same
(`Chunk(symbol_id=sid, ..., id=sid)`). Every test below therefore builds its rows the way
the PRODUCTION code builds them, where the two ids live in DIFFERENT tables and are never
equal: `_sub_chunk_search` sets `chunk.id` from the `sub_chunks` rowid and
`chunk.symbol_id` from `sub_chunks.parent_symbol_id`. The fixture is not "a pair of
numbers I chose to differ" -- it is the real leg's real output.

MECHANICS
* No Mock anywhere for anything the Retriever talks to: the DB, vector store, embedder
  and planner are plain recording classes, so a call that should not happen is
  observable rather than absorbed.
* Every expected value is a written LITERAL. Nothing is imported or recomputed from
  `retriever.py`, `fusion.py` or `config.py`.
* Each test's docstring names the ONE mutation that must make it fail, and each
  precondition names the FIXTURE property it depends on, so a fixture that stops
  discriminating fails loudly instead of passing vacuously.
* No environment claim: nothing here needs an optional extra, a credential, a network
  socket or a model weight, so there is no leaner-CI install that could make an
  assertion below stronger than the environment guarantees.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from trelix.core.config import IndexConfig, RetrievalConfig
from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    SearchResult,
    Symbol,
    SymbolKind,
)
from trelix.indexing.multi_granularity import Granularity, SubSymbolChunk
from trelix.retrieval.planner.models import (
    IntentType,
    QueryPlan,
    RetrievalStrategy,
    SubQuery,
)

# ---------------------------------------------------------------------------
# Fixture id spaces, written down once and kept DELIBERATELY DISJOINT.
#
# The whole point of this file is that `chunks.id` / `sub_chunks.id` and
# `symbols.id` are different tables, so a number is never both. Overlapping
# these ranges would make C2 unkillable again, exactly as it is unkillable
# against the two pre-existing helpers named in the module docstring.
# ---------------------------------------------------------------------------
_SUB_CHUNK_ID_LO = 41
_SUB_CHUNK_ID_HI = 42
_SHARED_PARENT_SYMBOL_ID = 500

_SUMMARY_FILE_ID = 7
_COLLIDING_REAL_SYMBOL_ID = 7  # equals _SUMMARY_FILE_ID on purpose -- see the test

_CHUNK_ID_A = 9001
_CHUNK_ID_B = 9002
_SYMBOL_ID_A = 11
_SYMBOL_ID_B = 12


# ---------------------------------------------------------------------------
# Plain doubles (no Mock)
# ---------------------------------------------------------------------------


def _file(fid: int, rel_path: str | None = None) -> IndexedFile:
    rp = rel_path or f"src/m{fid}.py"
    return IndexedFile(
        path=f"/repo/{rp}",
        rel_path=rp,
        language=Language.PYTHON,
        hash=f"sha-{fid}",
        size_bytes=100,
        id=fid,
        indexed_at=datetime(2024, 1, 1),
    )


def _symbol(sid: int, fid: int = 1, name: str | None = None, line_start: int = 1) -> Symbol:
    nm = name or f"func_{sid}"
    return Symbol(
        file_id=fid,
        name=nm,
        qualified_name=f"m.{nm}",
        kind=SymbolKind.FUNCTION,
        line_start=line_start,
        line_end=line_start + 8,
        signature=f"def {nm}()",
        body=f"def {nm}():\n    pass",
        id=sid,
    )


def _row(chunk_id: int, symbol_id: int, file_id: int, score: float, source: str) -> SearchResult:
    """A SearchResult whose chunk.id and chunk.symbol_id are DIFFERENT numbers.

    Mirrors what `_sub_chunk_search` / `_hydrate_chunk` actually build. The two
    pre-existing helpers in test_retriever_core.py and
    test_retriever_leg_gates_and_dedup.py set them equal, which is why C2 survived.
    """
    return SearchResult(
        chunk=Chunk(
            id=chunk_id,
            symbol_id=symbol_id,
            chunk_text=f"body {symbol_id}",
            token_count=2,
        ),
        symbol=_symbol(symbol_id, fid=file_id),
        file=_file(file_id),
        score=score,
        rank=1,
        source=source,
    )


class _SubChunkVectorStore:
    """Vector store answering ONLY the sub-chunk ANN entry point.

    Two hits, both belonging to ONE parent symbol -- which is exactly what MGS3
    produces: `multi_granularity` splits a single symbol into several
    block/statement sub-chunks, each with its own `sub_chunks` rowid and its own
    embedding, all pointing at the same `parent_symbol_id`.
    """

    def __init__(self) -> None:
        self.sub_chunk_hits: list[tuple[int, float]] = [
            (_SUB_CHUNK_ID_LO, 0.42),
            (_SUB_CHUNK_ID_HI, 0.91),
        ]

    def search_sub_chunks(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        return self.sub_chunk_hits[:k]

    def search(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        return []

    def search_file_summaries(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        return []


class _SubChunkDatabase:
    """Both sub-chunk rowids resolve to the SAME parent symbol."""

    def get_sub_chunk_by_id(self, sub_chunk_id: int) -> SubSymbolChunk:
        return SubSymbolChunk(
            parent_symbol_id=_SHARED_PARENT_SYMBOL_ID,
            granularity=Granularity.BLOCK,
            chunk_text=f"sub {sub_chunk_id}",
            line_start=1,
            line_end=2,
            token_count=2,
            id=sub_chunk_id,
        )

    def get_symbol_with_file(self, symbol_id: int) -> tuple[Symbol, IndexedFile]:
        return (_symbol(symbol_id), _file(1))


class _SummaryVectorStore:
    """Vector store answering ONLY the file-summary ANN entry point."""

    def search_file_summaries(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        return [(_SUMMARY_FILE_ID, 0.77)][:k]

    def search(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        return []

    def search_sub_chunks(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        return []


class _SummaryDatabase:
    """One file with a stored summary and three symbols, NOT in line order.

    The out-of-order list is load-bearing: with a pre-sorted list,
    `min(symbols, key=line_start)` and `symbols[0]` are indistinguishable and the
    representative-symbol test below would pass for the wrong reason.
    """

    #: (name, line_start), in the order the DB double hands them back.
    SYMBOLS: tuple[tuple[str, int], ...] = (
        ("declared_first_but_lowest_in_the_file", 30),
        ("declared_second_but_first_in_the_file", 5),
        ("declared_third_in_the_middle", 17),
    )

    def get_file_by_id(self, file_id: int) -> IndexedFile:
        return _file(file_id)

    def get_file_summary(self, file_id: int) -> str:
        return f"summary of file {file_id}"

    def get_symbols_for_file(self, file_id: int) -> list[Symbol]:
        return [
            _symbol(600 + i, fid=file_id, name=name, line_start=line)
            for i, (name, line) in enumerate(self.SYMBOLS)
        ]


class _LegWeightVectorStore:
    """ANN hits in a FIXED order: chunk A first, chunk B second."""

    def search(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        return [(_CHUNK_ID_A, 0.10), (_CHUNK_ID_B, 0.20)][:k]

    def search_file_summaries(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        return []

    def search_sub_chunks(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        return []


class _LegWeightDatabase:
    """Chunk ids hydrate to two DISTINCT symbols in two DISTINCT files.

    Distinct files because fusion dedups on (absolute file path, symbol_id); two
    rows sharing a path would make the identity key do work this test is not
    about.
    """

    _BY_CHUNK: dict[int, tuple[int, int]] = {
        _CHUNK_ID_A: (_SYMBOL_ID_A, 1),
        _CHUNK_ID_B: (_SYMBOL_ID_B, 2),
    }

    def get_chunk_with_context(self, chunk_id: int):
        symbol_id, file_id = self._BY_CHUNK[chunk_id]
        row = _row(chunk_id, symbol_id, file_id, 0.0, "vector")
        return (row.chunk, row.symbol, row.file)


class _RecordingEmbedder:
    dimension = 8

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.0] * 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]


class _StubPlanner:
    def plan(self, query: str):
        raise AssertionError("planner must not be consulted: every test supplies a plan")


def _build(retrieval: RetrievalConfig, repo_path: Path, db: object, vs: object) -> object:
    from trelix.retrieval.retriever import Retriever

    with (
        patch("trelix.retrieval.retriever.Database", return_value=db),
        patch("trelix.retrieval.retriever.make_embedder", return_value=_RecordingEmbedder()),
        patch("trelix.retrieval.retriever.make_vector_store", return_value=vs),
        patch("trelix.retrieval.retriever.QueryPlanner", return_value=_StubPlanner()),
        patch.dict(os.environ, {}, clear=False),
    ):
        return Retriever(IndexConfig(repo_path=str(repo_path), retrieval=retrieval))


def _plan(legs: list[str]) -> QueryPlan:
    return QueryPlan(
        intent=IntentType.FEATURE_FLOW,
        execution_mode="sequential",
        strategy=RetrievalStrategy(
            expand_depth=0,
            legs=legs,
            skip_reranker=True,
            import_depth=0,
            import_max_extra=0,
            import_direction="both",
            assembly_mode="greedy",
            rerank_top_n=10,
        ),
        sub_queries=[
            SubQuery(
                semantic_query="how does auth work",
                hyde_snippet="",
                bm25_tokens=["auth"],
                grep_hints=["authenticate"],
                file_hints=[],
            )
        ],
        raw_query="how does auth work",
    )


# ---------------------------------------------------------------------------
# C2 -- _dedup's identity key
# ---------------------------------------------------------------------------


class TestDedupIsKeyedOnTheSymbolNotTheChunk:
    """`_dedup` promises "Remove duplicate symbols, keeping highest score".

    MUTATION that must make this fail: `sid = r.chunk.symbol_id` -> `sid = r.chunk.id`
    in `Retriever._dedup`.

    CONSEQUENCE if it survives: `_dedup` is the ONLY reduction between the six fused
    legs plus four graph-expansion tails and the assembler's token budget. Keyed on
    `chunk.id` it stops being a per-symbol reduction: MGS3's sub-chunk leg emits one
    row per `sub_chunks` rowid, so a symbol split into N sub-chunks contributes N rows
    and the assembler packs the SAME symbol N times -- N-1 copies of the budget spent
    on text the caller already has. The mirror-image failure is worse: `hydrate_symbol`
    and `_hydrate_symbol_id` fabricate `Chunk(symbol_id=..., ...)` with NO id at all
    when a symbol has no stored chunk row, so under the mutation every such row keys on
    `None` and all but one DISTINCT symbol reached by graph expansion is deleted.
    """

    def test_two_sub_chunks_of_one_symbol_collapse_to_a_single_row(self, tmp_path: Path) -> None:
        retriever = _build(
            RetrievalConfig(sub_chunk_search_enabled=True),
            tmp_path,
            _SubChunkDatabase(),
            _SubChunkVectorStore(),
        )

        # The REAL leg builds the rows -- not a hand-written pair.
        rows = retriever._sub_chunk_search([0.0] * 8, k=2)

        # PRECONDITIONS naming _SubChunkVectorStore / _SubChunkDatabase. If either
        # stops producing two rows that share a parent symbol while carrying distinct
        # sub-chunk rowids, the assertion below is true by construction and this test
        # has stopped discriminating.
        assert len(rows) == 2, "_SubChunkVectorStore must yield exactly two sub-chunk hits"
        assert {r.chunk.symbol_id for r in rows} == {500}, (
            "_SubChunkDatabase must map BOTH sub-chunk rowids to one parent symbol"
        )
        assert sorted(r.chunk.id for r in rows) == [41, 42], (
            "the two rows must carry DISTINCT chunk ids -- equal ids is exactly the "
            "by-construction fixture that lets the mutation survive elsewhere"
        )

        deduped = retriever._dedup(rows)

        assert len(deduped) == 1
        actual_symbol_ids = {r.chunk.symbol_id for r in deduped}
        assert actual_symbol_ids == {500}
        assert {500} == actual_symbol_ids
        # Highest score wins, and it is the sub-chunk that scored 0.91.
        assert deduped[0].score == pytest.approx(0.91)
        assert deduped[0].chunk.id == 42


# ---------------------------------------------------------------------------
# L -- the file-summary sentinel must not claim a real symbol's identity
# ---------------------------------------------------------------------------


class TestFileSummarySentinelCannotCollideWithARealSymbolId:
    """`_summary_search`'s synthetic chunk carries `symbol_id=-(file_id)`.

    MUTATION that must make this fail: `symbol_id=-(file_id)` -> `symbol_id=file_id`
    in `Retriever._summary_search` (the code comment on that very line already states
    the requirement: "unique per file so dedup keeps all summaries").

    CONSEQUENCE if it survives: `symbols.id` and `files.id` are independent AUTOINCREMENT
    sequences in the same database, so small values overlap constantly -- file 7 and
    symbol 7 both exist in any real index. A positive sentinel makes the file-7 summary
    row and the symbol-7 row claim ONE `_dedup` identity, and `_dedup` keeps whichever
    scored higher. Either the whole RAPTOR file summary vanishes, or a genuine symbol hit
    does. Nothing raises; the leg simply appears to under-return.
    """

    def test_a_summary_row_and_a_real_symbol_row_with_the_same_number_both_survive(
        self, tmp_path: Path
    ) -> None:
        retriever = _build(
            RetrievalConfig(file_summary_leg_enabled=True),
            tmp_path,
            _SummaryDatabase(),
            _SummaryVectorStore(),
        )

        summary_rows = retriever._summary_search([0.0] * 8, k=1)
        assert len(summary_rows) == 1, "_SummaryVectorStore must yield exactly one summary hit"

        # A perfectly ordinary vector-leg row whose SYMBOL id equals the summary's
        # FILE id. PRECONDITION naming the constants: if these two ever stop being
        # the same number there is no collision left to detect.
        assert _COLLIDING_REAL_SYMBOL_ID == _SUMMARY_FILE_ID, (
            "the fixture's real-symbol id must equal the summarised file's id, or "
            "this test cannot observe the collision it exists to observe"
        )
        real_row = _row(_CHUNK_ID_A, _COLLIDING_REAL_SYMBOL_ID, 3, 0.99, "vector")

        deduped = retriever._dedup([*summary_rows, real_row])

        actual = {r.chunk.symbol_id for r in deduped}
        expected = {-7, 7}
        assert actual == expected
        assert expected == actual
        assert len(deduped) == 2
        assert {r.source for r in deduped} == {"file_summary", "vector"}


# ---------------------------------------------------------------------------
# I -- which symbol represents the file-level summary
# ---------------------------------------------------------------------------


class TestSummaryRepresentativeSymbolIsTheEarliestInTheFile:
    """`_summary_search`'s docstring: "the symbol is the FIRST symbol in the file".

    MUTATION that must make this fail: `rep_symbol = min(symbols, key=lambda s: s.line_start)`
    -> `max(...)` in `Retriever._summary_search`.

    CONSEQUENCE if it survives: the representative symbol is the only per-symbol
    metadata a file-summary row carries, and it is what every downstream consumer
    reports the summary AS -- `_trace("retrieval_legs", top_summary=[{"name": r.symbol.name
    ...}])`, the assembler's per-source attribution, and the `file:line` a caller is told
    to open. `max` names the LAST declaration in the file, so a summary of `auth.py`
    is attributed to whatever helper happens to sit at the bottom of it. The summary
    text itself is unchanged, which is why nothing fails loudly: the citation is simply
    wrong.
    """

    def test_the_representative_symbol_is_the_one_with_the_lowest_line_start(
        self, tmp_path: Path
    ) -> None:
        db = _SummaryDatabase()
        retriever = _build(
            RetrievalConfig(file_summary_leg_enabled=True),
            tmp_path,
            db,
            _SummaryVectorStore(),
        )

        # PRECONDITION naming _SummaryDatabase.SYMBOLS: the list must NOT already be
        # in line order, or `min(...)`, `max(...)` and `symbols[0]` become
        # indistinguishable and this test stops discriminating between them.
        declared_order = [line for _name, line in _SummaryDatabase.SYMBOLS]
        assert declared_order == [30, 5, 17]
        assert declared_order != sorted(declared_order)
        assert declared_order != sorted(declared_order, reverse=True)

        rows = retriever._summary_search([0.0] * 8, k=1)

        assert len(rows) == 1
        assert rows[0].symbol.name == "declared_second_but_first_in_the_file"
        assert rows[0].symbol.line_start == 5


# ---------------------------------------------------------------------------
# F -- per-leg RRF weights
# ---------------------------------------------------------------------------


class TestPerLegRRFWeightIsReadPerLeg:
    """`RetrievalConfig.leg_weights` must be honoured for the grep leg specifically.

    MUTATION that must make this fail: `cfg.leg_weights.get("grep", 1.0)` -> `1.0`
    inside `_standard_candidates`'s `_list_weights`.

    NOT AN EQUIVALENT MUTANT: the shipped default IS 1.0 for every leg (config.py
    documents "All-1.0 (the default) is a no-op"), so at defaults the mutation is
    invisible -- which is precisely why it survived 665 tests. It becomes live the
    moment an operator sets `TRELIX_RETRIEVAL_LEG_WEIGHT_GREP`, the one documented way
    to say "this repo's grep leg is noise, down-rank it". Under the mutation that
    setting is silently ignored for grep alone while the other five legs still honour
    theirs, so the operator sees the knob work for five legs and not the sixth.

    ARITHMETIC, written out so the expected order is not a guess. RRF contribution is
    `list_weight / (k + rank)` with the shipped `rrf_k = 60`, and `_list_weights[2]` is
    the grep slot:
        A appears at vector rank 1 only        -> 1.0/61                = 0.016393
        B appears at vector rank 2 and grep 1  -> 1.0/62 + w_grep/61
      w_grep = 0.0 (this test's config): B = 0.016129            -> A outranks B
      w_grep = 1.0 (the mutation):      B = 0.016129 + 0.016393  -> B outranks A
    No tie in either direction, so the expected order does not depend on sort stability.
    """

    def test_a_zero_weighted_grep_leg_cannot_lift_a_row_above_a_vector_only_row(
        self, tmp_path: Path
    ) -> None:
        cfg = RetrievalConfig(
            leg_weights={"vector": 1.0, "bm25": 1.0, "grep": 0.0},
            rerank=False,
            graph_expansion_max_symbols=10,
        )

        # PRECONDITIONS naming the config: the arithmetic above is only the arithmetic
        # this test runs if these three hold.
        assert cfg.rrf_k == 60
        assert cfg.leg_weights["grep"] == 0.0
        assert cfg.leg_weights["vector"] == 1.0

        retriever = _build(cfg, tmp_path, _LegWeightDatabase(), _LegWeightVectorStore())

        # The grep leg returns ONLY B, so grep's weight is the only thing that can
        # move B relative to A.
        grep_rows = [_row(_CHUNK_ID_B, _SYMBOL_ID_B, 2, 0.5, "grep")]
        with ExitStack() as stack:
            p = stack.enter_context
            p(patch("trelix.retrieval.retriever.grep_search", return_value=grep_rows))
            p(patch("trelix.retrieval.retriever.expand_with_call_graph", return_value=[]))
            p(patch("trelix.retrieval.retriever.expand_with_imports", return_value=[]))
            p(patch("trelix.retrieval.retriever.expand_with_type_edges", return_value=[]))
            candidates = retriever._standard_candidates(_plan(["vector", "grep"]))

        # PRECONDITION naming _LegWeightVectorStore / _LegWeightDatabase: both rows
        # must have survived to the candidate set, or "A before B" is vacuous.
        actual_symbol_ids = {r.chunk.symbol_id for r in candidates}
        assert actual_symbol_ids == {11, 12}
        assert {11, 12} == actual_symbol_ids

        assert [r.chunk.symbol_id for r in candidates] == [11, 12]
