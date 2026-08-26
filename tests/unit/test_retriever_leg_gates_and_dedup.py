"""Four unpinned decisions in `retrieval/retriever.py`, each found by mutation.

Every test below names, in its docstring, the exact source mutation that must make
it fail. All four mutants SURVIVED the pre-existing suite
(test_retriever_core.py / test_retriever_leg_signals.py /
test_retrieval_breadth_floor.py / test_fusion.py):

1. `if cfg.graph_search_enabled:` -> `if True:`
   The CodeGraph BFS leg is documented MEASURED HARMFUL and ships default-off
   (`RetrievalConfig.graph_search_enabled = False`). Flipping that default ON put
   BFS rows into the candidate set of every standard query and nothing failed.
   Same for the other three default-off legs (`file_summary_leg_enabled`,
   `sub_chunk_search_enabled`, `sparse_enabled`).

2. `if "bm25" in strategy.legs:` -> `if True:` and the same for `"grep"`.
   `RetrievalStrategy.legs` is the per-intent contract for which legs run.
   Deleting the bm25 and grep gates ran both legs on an intent that asked for
   neither, and the suite stayed green: the one existing test
   (`test_legs_not_in_strategy_are_skipped`) checks only that the OUTPUT lists
   are empty, which they still are when the search function is called and
   happens to find nothing.

3. `if sid not in seen or r.score > seen[sid].score:` -> `if sid not in seen:`
   `_dedup`'s docstring promises "keeping highest score". Dropping the score
   comparison keeps whichever duplicate arrived FIRST. `_dedup` runs on
   `fused + call_expanded + import_expanded + type_expanded + ...`, where the
   graph-expansion tails append at fixed score 1.0 AFTER the fused rows, so the
   comparison is load-bearing for real orderings.

4. `if ratio >= 1.0:` -> `if ratio > 1.0:` in `_make_compressor`.
   `compression_ratio_for_intent` returns exactly 1.0 for symbol_lookup,
   file_overview, project_overview and config_lookup, and its docstring says
   "A returned 1.0 means 'do not compress this intent' and callers MUST skip
   compression entirely". `> 1.0` builds a compressor for all four.

No MagicMock is used for anything Retriever talks to on these paths: the DB,
vector store, embedder and optional-leg collaborators are all plain recording
classes, so a call that should not happen is observable rather than absorbed.
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
# Plain (non-Mock) doubles
# ---------------------------------------------------------------------------


def _file(fid: int, rel_path: str | None = None) -> IndexedFile:
    return IndexedFile(
        path=f"/repo/{rel_path or f'src/m{fid}.py'}",
        rel_path=rel_path or f"src/m{fid}.py",
        language=Language.PYTHON,
        hash=f"sha-{fid}",
        size_bytes=100,
        id=fid,
        indexed_at=datetime(2024, 1, 1),
    )


def _symbol(sid: int, fid: int = 1, name: str | None = None) -> Symbol:
    nm = name or f"func_{sid}"
    return Symbol(
        file_id=fid,
        name=nm,
        qualified_name=f"m.{nm}",
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=9,
        signature=f"def {nm}()",
        body=f"def {nm}():\n    pass",
        id=sid,
    )


def _result(sid: int, score: float, source: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(symbol_id=sid, chunk_text=f"body {sid}", token_count=2, id=sid),
        symbol=_symbol(sid),
        file=_file(sid),
        score=score,
        rank=1,
        source=source,
    )


class _RecordingVectorStore:
    """Answers all three ANN entry points and records which ones were reached."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        # Non-empty on purpose: a leg that runs WILL produce rows, so "no rows"
        # cannot be mistaken for "leg correctly skipped".
        self.ann_hits: list[tuple[int, float]] = [(1, 0.1)]
        self.summary_hits: list[tuple[int, float]] = [(7, 0.9)]
        self.sub_chunk_hits: list[tuple[int, float]] = [(8, 0.9)]

    def search(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        self.calls.append("search")
        return self.ann_hits[:k]

    def search_file_summaries(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        self.calls.append("search_file_summaries")
        return self.summary_hits[:k]

    def search_sub_chunks(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        self.calls.append("search_sub_chunks")
        return self.sub_chunk_hits[:k]


class _RecordingDatabase:
    """Enough of Database for the standard path, with every lookup satisfiable."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    # vector leg hydration
    def get_chunk_with_context(self, chunk_id: int):
        r = _result(chunk_id, 0.9, "vector")
        return (r.chunk, r.symbol, r.file)

    # file-summary leg
    def get_file_by_id(self, file_id: int) -> IndexedFile:
        return _file(file_id)

    def get_file_summary(self, file_id: int) -> str:
        return f"summary of file {file_id}"

    def get_symbols_for_file(self, file_id: int) -> list[Symbol]:
        return [_symbol(100 + file_id, fid=file_id)]

    # sub-chunk leg
    def get_sub_chunk_by_id(self, sub_chunk_id: int) -> SubSymbolChunk:
        return SubSymbolChunk(
            parent_symbol_id=200 + sub_chunk_id,
            granularity=Granularity.BLOCK,
            chunk_text=f"sub {sub_chunk_id}",
            line_start=1,
            line_end=2,
            token_count=2,
            id=sub_chunk_id,
        )

    def get_symbol_with_file(self, symbol_id: int) -> tuple[Symbol, IndexedFile]:
        return (_symbol(symbol_id), _file(symbol_id))

    def get_first_chunk_for_symbol(self, symbol_id: int) -> None:
        return None


class _RecordingEmbedder:
    dimension = 8

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.0] * 8

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - unused
        return [[0.0] * 8 for _ in texts]


class _StubPlanner:
    def plan(self, query: str):  # pragma: no cover - external plans only here
        raise AssertionError("planner must not be consulted: every test supplies a plan")


def _build(
    retrieval: RetrievalConfig, repo_path: Path
) -> tuple[object, _RecordingDatabase, _RecordingVectorStore]:
    from trelix.retrieval.retriever import Retriever

    db = _RecordingDatabase()
    vs = _RecordingVectorStore()
    emb = _RecordingEmbedder()
    with (
        patch("trelix.retrieval.retriever.Database", return_value=db),
        patch("trelix.retrieval.retriever.make_embedder", return_value=emb),
        patch("trelix.retrieval.retriever.make_vector_store", return_value=vs),
        patch("trelix.retrieval.retriever.QueryPlanner", return_value=_StubPlanner()),
        patch.dict(os.environ, {}, clear=False),
    ):
        r = Retriever(IndexConfig(repo_path=str(repo_path), retrieval=retrieval))
    return r, db, vs


# ---------------------------------------------------------------------------
# Optional-leg collaborator stubs (plain classes / functions)
# ---------------------------------------------------------------------------


class _StubCodeGraph:
    def __init__(self, db: object) -> None:
        self.db = db


class _StubSparseEmbedder:
    instances = 0

    def __init__(self, model_name: str, top_k: int) -> None:
        type(self).instances += 1

    def embed_query(self, text: str) -> dict[int, float]:
        return {1: 1.0}


class _StubSparseStore:
    def __init__(self, path: object) -> None:
        pass


def _plan(intent: IntentType, legs: list[str]) -> QueryPlan:
    return QueryPlan(
        intent=intent,
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


def _candidate_sources(retriever: object, plan: QueryPlan) -> set[str]:
    """Run _standard_candidates with every optional leg's collaborator stubbed in.

    The stubs all SUCCEED, so a leg that runs contributes an identifiable
    `source` string. Only the retriever's own gates decide what appears.
    """
    _StubSparseEmbedder.instances = 0
    graph_rows = [_result(900, 1.0, "graph_bfs")]
    sparse_rows = [_result(901, 1.0, "sparse")]
    with ExitStack() as stack:
        p = stack.enter_context
        p(patch("trelix.retrieval.retriever.bm25_search", return_value=[]))
        p(patch("trelix.retrieval.retriever.grep_search", return_value=[]))
        p(patch("trelix.retrieval.retriever.expand_with_call_graph", return_value=[]))
        p(patch("trelix.retrieval.retriever.expand_with_imports", return_value=[]))
        p(patch("trelix.retrieval.retriever.expand_with_type_edges", return_value=[]))
        p(patch("trelix.graph.code_graph.CodeGraph", _StubCodeGraph))
        p(patch("trelix.graph.search.graph_search", lambda **kw: graph_rows))
        p(patch("trelix.embedder.sparse.SparseEmbedder", _StubSparseEmbedder))
        p(patch("trelix.store.sparse_store.SparseStore", _StubSparseStore))
        p(patch("trelix.retrieval.sparse_search.sparse_search", lambda *a, **k: sparse_rows))
        candidates = retriever._standard_candidates(plan)
    return {r.source for r in candidates}


# ---------------------------------------------------------------------------
# 1. Default-off legs stay off
# ---------------------------------------------------------------------------


class TestDefaultOffLegsStayOff:
    """The four optional legs ship OFF; only an explicit opt-in may turn them on."""

    def test_all_four_optional_legs_are_off_in_a_default_config(self) -> None:
        """Pins the shipped defaults themselves.

        MUTATION THAT MUST FAIL THIS: flipping any of
        `graph_search_enabled` / `file_summary_leg_enabled` /
        `sub_chunk_search_enabled` / `sparse_enabled` to True in
        core/config.py:RetrievalConfig.
        """
        cfg = RetrievalConfig()
        assert cfg.graph_search_enabled is False
        assert cfg.file_summary_leg_enabled is False
        assert cfg.sub_chunk_search_enabled is False
        assert cfg.sparse_enabled is False

    def test_default_config_contributes_only_the_vector_leg(self, tmp_path: Path) -> None:
        """Only "vector" reaches the candidate set under a default RetrievalConfig.

        MUTATION THAT MUST FAIL THIS: `if cfg.graph_search_enabled:` -> `if True:`
        (adds "graph_bfs"); `if cfg.file_summary_leg_enabled and plan.sub_queries:`
        -> `if plan.sub_queries:` (adds "file_summary");
        `if cfg.sub_chunk_search_enabled and plan.sub_queries:` ->
        `if plan.sub_queries:` (adds "sub_chunk"); `if cfg.sparse_enabled:` ->
        `if True:` (adds "sparse").
        """
        retriever, _db, vs = _build(RetrievalConfig(), tmp_path)
        sources = _candidate_sources(retriever, _plan(IntentType.FEATURE_FLOW, ["vector"]))

        assert sources == {"vector"}
        # Explicit table, both directions, no iteration of the pinned collection.
        assert "graph_bfs" not in sources
        assert "file_summary" not in sources
        assert "sub_chunk" not in sources
        assert "sparse" not in sources
        # _RecordingVectorStore precondition: the two summary/sub-chunk entry
        # points were never reached, and they WOULD have answered if they had
        # been (see test_opt_in_turns_every_optional_leg_on).
        assert "search_file_summaries" not in vs.calls
        assert "search_sub_chunks" not in vs.calls
        assert _StubSparseEmbedder.instances == 0

    def test_opt_in_turns_every_optional_leg_on(self, tmp_path: Path) -> None:
        """Discrimination guard for the test above, naming the fixtures it relies on.

        If `_RecordingVectorStore` / `_StubSparseEmbedder` / the `graph_search`
        stub ever stop being able to produce rows, this test fails and the
        default-off assertion above stops meaning anything.

        MUTATION THAT MUST FAIL THIS: hard-wiring any optional leg OFF, e.g.
        `if cfg.graph_search_enabled:` -> `if False:`.
        """
        retriever, _db, vs = _build(
            RetrievalConfig(
                graph_search_enabled=True,
                file_summary_leg_enabled=True,
                sub_chunk_search_enabled=True,
                sparse_enabled=True,
            ),
            tmp_path,
        )
        sources = _candidate_sources(retriever, _plan(IntentType.FEATURE_FLOW, ["vector"]))

        assert sources == {"vector", "file_summary", "sub_chunk", "sparse", "graph_bfs"}
        assert "search_file_summaries" in vs.calls
        assert "search_sub_chunks" in vs.calls
        assert _StubSparseEmbedder.instances == 1


# ---------------------------------------------------------------------------
# 2. strategy.legs actually gates each leg
# ---------------------------------------------------------------------------


class TestStrategyLegsGateEachLeg:
    """A leg absent from `strategy.legs` must not be CALLED, not merely return [].

    The pre-existing `test_legs_not_in_strategy_are_skipped` asserts empty output
    lists, which a called-but-fruitless search also satisfies. These pin the call.
    """

    @staticmethod
    def _run(legs: list[str], repo_path: Path) -> set[str]:
        retriever, _db, _vs = _build(RetrievalConfig(), repo_path)
        called: set[str] = set()

        def _bm25(*a: object, **k: object) -> list[SearchResult]:
            called.add("bm25")
            return []

        def _grep(*a: object, **k: object) -> list[SearchResult]:
            called.add("grep")
            return []

        sq = SubQuery(
            semantic_query="how does auth work",
            hyde_snippet="",
            bm25_tokens=["auth"],
            grep_hints=["authenticate"],
            file_hints=[],
        )
        strategy = RetrievalStrategy(
            expand_depth=0,
            legs=legs,
            skip_reranker=True,
            import_depth=0,
            import_max_extra=0,
            import_direction="both",
            assembly_mode="greedy",
            rerank_top_n=10,
        )
        with (
            patch("trelix.retrieval.retriever.bm25_search", _bm25),
            patch("trelix.retrieval.retriever.grep_search", _grep),
        ):
            out = retriever._run_subquery_legs(sq, strategy)
        if out["vector"]:
            called.add("vector")
        return called

    @pytest.mark.parametrize(
        ("legs", "expected_called"),
        [
            (["vector"], {"vector"}),
            (["bm25"], {"bm25"}),
            (["grep"], {"grep"}),
            (["vector", "bm25"], {"vector", "bm25"}),
            (["grep", "bm25", "vector"], {"vector", "bm25", "grep"}),
            ([], set()),
        ],
    )
    def test_exactly_the_requested_legs_run(
        self, legs: list[str], expected_called: set[str], tmp_path: Path
    ) -> None:
        """Explicit table; set equality in both directions.

        MUTATION THAT MUST FAIL THIS: `if "bm25" in strategy.legs:` -> `if True:`
        (row ["vector"] then calls bm25); `if "grep" in strategy.legs:` ->
        `if True:` (row ["vector"] then calls grep); `if "vector" in
        strategy.legs and not skip_vector:` -> `if not skip_vector:` (row ["bm25"]
        then runs the ANN search).
        """
        assert self._run(legs, tmp_path) == expected_called


# ---------------------------------------------------------------------------
# 3. _dedup keeps the highest score, not the first arrival
# ---------------------------------------------------------------------------


class TestDedupKeepsHighestScore:
    """`_dedup`'s contract is "keeping highest score", independent of arrival order."""

    def test_a_later_higher_score_replaces_an_earlier_lower_one(self, tmp_path: Path) -> None:
        """Same symbol_id twice, WORSE first: the better row must win.

        This is the real shape in `_standard_candidates` - `_dedup` is handed
        `fused + call_expanded + ...` and the expansion tails append later.

        MUTATION THAT MUST FAIL THIS:
        `if sid not in seen or r.score > seen[sid].score:` -> `if sid not in seen:`
        """
        retriever, _db, _vs = _build(RetrievalConfig(), tmp_path)
        worse = _result(42, 0.10, "call_graph")
        better = _result(42, 0.90, "vector")

        deduped = retriever._dedup([worse, better])

        assert len(deduped) == 1
        assert deduped[0].score == pytest.approx(0.90)
        assert deduped[0].source == "vector"

    def test_an_earlier_higher_score_is_not_replaced(self, tmp_path: Path) -> None:
        """The mirror case, so the assertion is not satisfiable by "always keep last".

        MUTATION THAT MUST FAIL THIS: `if sid not in seen or r.score >
        seen[sid].score:` -> `if True:` (unconditional overwrite).
        """
        retriever, _db, _vs = _build(RetrievalConfig(), tmp_path)
        better = _result(42, 0.90, "vector")
        worse = _result(42, 0.10, "call_graph")

        deduped = retriever._dedup([better, worse])

        assert len(deduped) == 1
        assert deduped[0].score == pytest.approx(0.90)
        assert deduped[0].source == "vector"

    def test_an_exact_score_tie_keeps_the_first_arrival(self, tmp_path: Path) -> None:
        """Equal scores: the strict `>` means the incumbent stays.

        Pins CURRENT behaviour deliberately - with fusion ahead of the expansion
        tails, "first" is the higher-provenance row, so keeping it is right.

        MUTATION THAT MUST FAIL THIS:
        `if sid not in seen or r.score > seen[sid].score:` ->
        `if sid not in seen or r.score >= seen[sid].score:`
        """
        retriever, _db, _vs = _build(RetrievalConfig(), tmp_path)
        first = _result(42, 0.50, "vector")
        second = _result(42, 0.50, "call_graph")

        deduped = retriever._dedup([first, second])

        assert len(deduped) == 1
        assert deduped[0].source == "vector"


# ---------------------------------------------------------------------------
# 4. Compression intent opt-out at ratio exactly 1.0
# ---------------------------------------------------------------------------


class TestCompressionIntentOptOut:
    """ratio == 1.0 means "do not compress"; no compressor may be constructed."""

    _OPT_OUT_INTENTS = ["symbol_lookup", "file_overview", "project_overview", "config_lookup"]

    @pytest.mark.parametrize("intent", _OPT_OUT_INTENTS)
    def test_ratio_one_intents_build_no_compressor(self, intent: str, tmp_path: Path) -> None:
        """MUTATION THAT MUST FAIL THIS: `if ratio >= 1.0:` -> `if ratio > 1.0:`
        in `_make_compressor` - the four ratio-1.0 intents then construct a
        compressor and `_assemble` runs it over bodies that ARE the answer.
        """
        retriever, _db, _vs = _build(RetrievalConfig(compression_enabled=True), tmp_path)
        built: list[str] = []

        def _make(*a: object, **k: object) -> object:
            built.append("called")
            raise AssertionError("make_compressor must not be reached for a ratio-1.0 intent")

        with patch("trelix.compression.make_compressor", _make):
            compressor, ratio = retriever._make_compressor(intent)

        assert compressor is None
        assert ratio == pytest.approx(1.0)
        assert built == []

    def test_a_compressing_intent_still_builds_one(self, tmp_path: Path) -> None:
        """Discrimination guard: proves the patched `trelix.compression.make_compressor`
        seam is the one `_make_compressor` uses, so the four assertions above are
        not passing because the patch target is wrong.

        MUTATION THAT MUST FAIL THIS: `if ratio >= 1.0:` -> `if ratio >= 0.0:`
        (every intent then opts out).
        """
        retriever, _db, _vs = _build(RetrievalConfig(compression_enabled=True), tmp_path)
        sentinel = object()

        with patch("trelix.compression.make_compressor", lambda *a, **k: sentinel):
            compressor, ratio = retriever._make_compressor("feature_flow")

        assert compressor is sentinel
        assert ratio == pytest.approx(0.45)
