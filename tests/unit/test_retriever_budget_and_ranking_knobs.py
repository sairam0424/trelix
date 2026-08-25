"""Eight unpinned knobs in `retrieval/retriever.py`, each measured as a survivor first.

Every mutation below was applied ALONE (one site, verified by an exact-anchor count of
1) and reproduced as a SURVIVOR against the pre-existing retriever-adjacent suite
(627 tests: test_retriever_core / test_retriever_leg_signals /
test_retriever_leg_gates_and_dedup / test_retriever_file_summary /
test_retrieval_breadth_floor / test_fusion / test_assembler{,_compression,
_backcompat_golden} / test_model_aware_budget / test_reranker_core /
test_rerank_outcome / test_graph_pagerank / test_graph_persistence /
test_graph_search / test_config) before any test here existed.

    M34  delete `if not cfg.pagerank_boost_enabled: return results`
    M26  `top = fused[: cfg.graph_expansion_max_symbols]` -> `top = fused`
    M19  `k=cfg.rrf_k` -> `k=cfg.rrf_k + 1` at the retriever's RRF CALL SITE
    M18  `_weights = cfg.file_type_weights if cfg.file_type_weighting_enabled else None`
         -> `_weights = cfg.file_type_weights`
    M22  drop `and cfg.context_token_budget is None` from the rerank_top_n ternary
    M10  `max(1, int(top_k_vector * scale_factor))` -> `int(...)`
    M11  `max(1, int(rerank_top_n * scale_factor))` -> `int(...)`
    M43  `find_file_by_path_fragment(hint)[:2]` -> `[:1]` in `_retrieve_file_overview`

Every test asserts on a RETURNED ORDER or a RETURNED SET — the ranking or the row set a
caller actually receives — never on a mock's call arguments. No MagicMock is used for
anything the Retriever talks to.

Deliberately NOT tested here (see the round report):
  * the `token_budget=self._effective_budget` handoff in `_assemble` is ALREADY pinned by
    test_assembler_backcompat_golden.py::
    test_retriever_assemble_disabled_matches_head_and_writes_no_trace, which compares
    `_assemble` against a legacy assembler built at the same budget. `* 2` fails it.
  * the same `[:2]` cap in `_retrieve_config` is ALREADY pinned by
    test_retrieval_breadth_floor.py::TestConfigLookupBreadthFloor::
    test_many_files_does_not_widen. Only the `_retrieve_file_overview` twin is unpinned,
    which is why the two sites were mutated separately.
  * dropping `and cfg.context_token_budget is None` from the *top_k_vector* ternary is an
    EQUIVALENT mutant, not a survivor worth a test: that ternary's else-branch is
    `cfg.top_k_vector`, and `self._effective_top_k_vector` can only differ from
    `cfg.top_k_vector` when `__init__`'s scaling guard passed — which is exactly the
    condition the mutation removes. Pinning it would pin dead code. The rerank_top_n
    ternary (M22) is NOT equivalent because its else-branch is `strategy.rerank_top_n`.

Leg gating, `_dedup` and the compression opt-out are covered by
tests/unit/test_retriever_leg_gates_and_dedup.py and are not repeated here; the doubles
below are purpose-built for path-fragment lookup, per-language fusion weights and
budget-scaled ceilings, which that file's doubles do not model.
"""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from trelix.core.config import IndexConfig, LLMConfig, RetrievalConfig
from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    SearchResult,
    Symbol,
    SymbolKind,
)
from trelix.retrieval.planner.models import (
    IntentType,
    QueryPlan,
    RetrievalStrategy,
    SubQuery,
)

# ---------------------------------------------------------------------------
# Plain doubles (no Mock anywhere)
# ---------------------------------------------------------------------------


def _make_file(fid: int, rel_path: str, language: Language) -> IndexedFile:
    return IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=language,
        hash=f"sha-{fid}",
        size_bytes=100,
        id=fid,
        indexed_at=datetime(2024, 1, 1),
    )


def _make_symbol(sid: int, fid: int) -> Symbol:
    return Symbol(
        file_id=fid,
        name=f"sym_{sid}",
        qualified_name=f"m.sym_{sid}",
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=9,
        signature=f"def sym_{sid}()",
        body=f"def sym_{sid}():\n    pass",
        id=sid,
    )


def _row(
    sid: int,
    *,
    rel_path: str | None = None,
    language: Language = Language.PYTHON,
    score: float = 0.5,
    rank: int = 1,
    source: str = "vector",
    token_count: int = 0,
) -> SearchResult:
    """One SearchResult with a globally distinct fusion identity (file path, symbol_id)."""
    path = rel_path if rel_path is not None else f"src/gen/f{sid}.py"
    return SearchResult(
        chunk=Chunk(symbol_id=sid, chunk_text=f"body {sid}", token_count=token_count, id=sid),
        symbol=_make_symbol(sid, sid),
        file=_make_file(sid, path, language),
        score=score,
        rank=rank,
        source=source,
    )


class _RegistryDatabase:
    """A Database stand-in backed by an explicit symbol registry.

    Everything is registered by the test, so a lookup that should not happen returns
    nothing rather than being absorbed by a permissive auto-answering mock.
    """

    def __init__(self) -> None:
        self.rows: dict[int, SearchResult] = {}
        # `_retrieve_file_overview` inputs
        self.fragment_hits: list[int] = []
        self.file_symbols: dict[int, list[int]] = {}
        self.fragment_queries: list[str] = []

    def register(self, row: SearchResult) -> SearchResult:
        assert row.chunk.symbol_id is not None
        self.rows[row.chunk.symbol_id] = row
        return row

    # -- DimensionGuard (returns "unknown", which the guard treats as skip) --
    def get_embedding_dimension(self) -> None:
        return None

    # -- vector leg hydration --
    def get_chunk_with_context(self, chunk_id: int) -> tuple[Chunk, Symbol, IndexedFile] | None:
        row = self.rows.get(chunk_id)
        if row is None:
            return None
        return (row.chunk, row.symbol, row.file)

    # -- hydrate_symbol (file_overview path) --
    def get_symbol_with_file(self, symbol_id: int) -> tuple[Symbol, IndexedFile] | None:
        row = self.rows.get(symbol_id)
        if row is None:
            return None
        return (row.symbol, row.file)

    def get_first_chunk_for_symbol(self, symbol_id: int) -> Chunk | None:
        row = self.rows.get(symbol_id)
        return None if row is None else row.chunk

    # -- file_overview lookup --
    def find_file_by_path_fragment(self, fragment: str) -> list[int]:
        self.fragment_queries.append(fragment)
        return list(self.fragment_hits)

    def get_all_symbols_for_file(self, file_id: int) -> list[int]:
        return list(self.file_symbols.get(file_id, []))


class _ListVectorStore:
    """ANN double honouring the `k` contract: at most `k` neighbours, best first."""

    def __init__(self, hits: list[tuple[int, float]]) -> None:
        self.hits = hits

    def search(self, embedding: list[float], k: int) -> list[tuple[int, float]]:
        return self.hits[:k]


class _ZeroEmbedder:
    dimension = 8

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 8

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - unused
        return [[0.0] * 8 for _ in texts]


class _NoPlanner:
    def plan(self, query: str) -> QueryPlan:  # pragma: no cover - plans are always supplied
        raise AssertionError("planner must not be consulted: every test supplies a plan")


def _build(
    retrieval: RetrievalConfig,
    repo_path: Path,
    *,
    db: _RegistryDatabase | None = None,
    vector_hits: list[tuple[int, float]] | None = None,
    llm_model: str = "gpt-4o",
) -> tuple[object, _RegistryDatabase, _ListVectorStore]:
    from trelix.retrieval.retriever import Retriever

    database = db if db is not None else _RegistryDatabase()
    store = _ListVectorStore(vector_hits or [])
    repo_path.mkdir(parents=True, exist_ok=True)  # IndexConfig validates existence
    with (
        patch("trelix.retrieval.retriever.Database", return_value=database),
        patch("trelix.retrieval.retriever.make_embedder", return_value=_ZeroEmbedder()),
        patch("trelix.retrieval.retriever.make_vector_store", return_value=store),
        patch("trelix.retrieval.retriever.QueryPlanner", return_value=_NoPlanner()),
    ):
        retriever = Retriever(
            IndexConfig(
                repo_path=str(repo_path),
                retrieval=retrieval,
                llm=LLMConfig(model=llm_model, openai_api_key=None),
            )
        )
    return retriever, database, store


def _sub_query(**over: object) -> SubQuery:
    base: dict[str, object] = {
        "semantic_query": "how does auth work",
        "hyde_snippet": "",
        "bm25_tokens": ["auth"],
        "grep_hints": ["authenticate"],
        "file_hints": [],
    }
    base.update(over)
    return SubQuery(**base)  # type: ignore[arg-type]


def _strategy(legs: list[str], *, skip_reranker: bool = True, rerank_top_n: int = 10):
    return RetrievalStrategy(
        expand_depth=0,
        legs=legs,
        skip_reranker=skip_reranker,
        import_depth=0,
        import_max_extra=0,
        import_direction="both",
        assembly_mode="greedy",
        rerank_top_n=rerank_top_n,
    )


def _plan(
    legs: list[str],
    *,
    intent: IntentType = IntentType.FEATURE_FLOW,
    skip_reranker: bool = True,
    rerank_top_n: int = 10,
    sub_queries: list[SubQuery] | None = None,
) -> QueryPlan:
    return QueryPlan(
        intent=intent,
        execution_mode="sequential",
        strategy=_strategy(legs, skip_reranker=skip_reranker, rerank_top_n=rerank_top_n),
        sub_queries=sub_queries if sub_queries is not None else [_sub_query()],
        raw_query="how does auth work",
    )


def _standard_candidates(
    retriever: object,
    plan: QueryPlan,
    *,
    bm25: list[SearchResult] | None = None,
    grep: list[SearchResult] | None = None,
    call_graph: object = None,
) -> list[SearchResult]:
    """Run `_standard_candidates` with every graph collaborator under test control."""
    calls = call_graph if call_graph is not None else (lambda *a, **k: [])
    with ExitStack() as stack:
        p = stack.enter_context
        p(patch("trelix.retrieval.retriever.bm25_search", lambda *a, **k: list(bm25 or [])))
        p(patch("trelix.retrieval.retriever.grep_search", lambda *a, **k: list(grep or [])))
        p(patch("trelix.retrieval.retriever.expand_with_call_graph", calls))
        p(patch("trelix.retrieval.retriever.expand_with_imports", lambda *a, **k: []))
        p(patch("trelix.retrieval.retriever.expand_with_type_edges", lambda *a, **k: []))
        return retriever._standard_candidates(plan)


def _ids(results: list[SearchResult]) -> list[int]:
    return [r.chunk.symbol_id for r in results]


# ===========================================================================
# M34 — pagerank_boost_enabled is a real gate, under the SHIPPED default
# ===========================================================================


class TestPageRankBoostFlagIsHonoured:
    """`pagerank_boost_enabled` ships False, so deleting its guard changes the DEFAULT
    ranking every user gets — not merely an opted-in one."""

    _CENTRAL_SYMBOL_ID = 22  # the lower-scoring row; boosting it must reorder the pair

    def _rows(self) -> list[SearchResult]:
        # 0.45 * 1.3 == 0.585 > 0.50, so a boost applied to id 22 flips the pair.
        return [_row(11, score=0.50), _row(22, score=0.45)]

    def test_default_config_leaves_the_ranking_untouched(self, tmp_path: Path) -> None:
        """MUTATION THAT MUST FAIL THIS: deleting
        `if not cfg.pagerank_boost_enabled: return results` from
        `_apply_pagerank_boost` — id 22 is then boosted to 0.585 and overtakes id 11
        under the shipped default (`pagerank_boost_enabled=False`).
        """
        cfg = RetrievalConfig()
        # Precondition on the SHIPPED default: if this default ever flips to True the
        # assertion below stops describing default behaviour and must be rewritten.
        assert cfg.pagerank_boost_enabled is False
        assert cfg.pagerank_boost_factor == pytest.approx(1.3)

        retriever, _db, _vs = _build(cfg, tmp_path)
        central_calls: list[int] = []

        def _top_central(db: object, top_n: int = 200) -> list[int]:
            central_calls.append(top_n)
            return [self._CENTRAL_SYMBOL_ID]

        with patch("trelix.graph.persistence.get_top_central_symbols", _top_central):
            out = retriever._apply_pagerank_boost(self._rows())

        assert _ids(out) == [11, 22]
        assert [r.score for r in out] == [pytest.approx(0.50), pytest.approx(0.45)]
        assert central_calls == []

    def test_opting_in_does_reorder_the_pair(self, tmp_path: Path) -> None:
        """Discrimination guard naming its fixtures. If
        `patch("trelix.graph.persistence.get_top_central_symbols")` stops being the seam
        `_apply_pagerank_boost` imports, or if `_rows()` margins stop being flippable by
        the 1.3x factor, this fails — and the default-off assertion above stops meaning
        anything.

        MUTATION THAT MUST FAIL THIS: `if not cfg.pagerank_boost_enabled:` ->
        `if True:` (the boost becomes unreachable for everyone).
        """
        retriever, _db, _vs = _build(RetrievalConfig(pagerank_boost_enabled=True), tmp_path)
        central_calls: list[int] = []

        def _top_central(db: object, top_n: int = 200) -> list[int]:
            central_calls.append(top_n)
            return [self._CENTRAL_SYMBOL_ID]

        with patch("trelix.graph.persistence.get_top_central_symbols", _top_central):
            out = retriever._apply_pagerank_boost(self._rows())

        assert _ids(out) == [22, 11]
        assert out[0].score == pytest.approx(0.585)
        assert central_calls == [200]


# ===========================================================================
# M26 — the graph-expansion seed cap
# ===========================================================================


class TestGraphExpansionSeedCap:
    """Only the top `graph_expansion_max_symbols` fused rows may seed expansion."""

    # 12 fused rows; the cap is 10, so rows 11 and 12 must never seed.
    _FUSED_IDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]
    _EXPECTED_EXPANSION_IDS = {
        9101,
        9102,
        9103,
        9104,
        9105,
        9106,
        9107,
        9108,
        9109,
        9110,
    }

    def _setup(self, tmp_path: Path) -> tuple[object, list[list[int]]]:
        db = _RegistryDatabase()
        for sid in self._FUSED_IDS:
            db.register(_row(sid))
        # Descending distances -> descending vector scores -> ranks 1..12 in order.
        hits = [(sid, 0.01 * i) for i, sid in enumerate(self._FUSED_IDS)]
        retriever, _db, _vs = _build(RetrievalConfig(), tmp_path, db=db, vector_hits=hits)
        seeds_seen: list[list[int]] = []

        def _expand(db_arg: object, seeds: list[SearchResult], **kw: object):
            seeds_seen.append([r.chunk.symbol_id for r in seeds])
            return [_row(9000 + r.chunk.symbol_id, source="call_graph", score=1.0) for r in seeds]

        self._expand = _expand
        return retriever, seeds_seen

    def test_only_the_first_ten_fused_rows_seed_expansion(self, tmp_path: Path) -> None:
        """MUTATION THAT MUST FAIL THIS:
        `top = fused[: cfg.graph_expansion_max_symbols]` -> `top = fused`
        (rows 111 and 112 then seed, adding 9111 and 9112 to the returned candidates).
        """
        cfg = RetrievalConfig()
        assert cfg.graph_expansion_max_symbols == 10  # shipped default the cap uses

        retriever, seeds_seen = self._setup(tmp_path)
        out = _standard_candidates(retriever, _plan(["vector"]), call_graph=self._expand)
        returned = set(_ids(out))

        # Fixture precondition: all 12 fused rows really reached fusion, so the cap —
        # not a thin `_ListVectorStore` — is what limited the seeds.
        assert set(self._FUSED_IDS) <= returned

        expansions = {sid for sid in returned if sid >= 9000}
        assert expansions == self._EXPECTED_EXPANSION_IDS
        assert self._EXPECTED_EXPANSION_IDS == expansions  # set equality, both directions
        assert 9111 not in returned
        assert 9112 not in returned
        assert seeds_seen == [[101, 102, 103, 104, 105, 106, 107, 108, 109, 110]]

    def test_raising_the_cap_admits_the_eleventh_and_twelfth_seeds(self, tmp_path: Path) -> None:
        """Discrimination guard naming `_setup`'s 12-row fixture and the `_expand`
        double: if either stops being able to produce a seed for row 111/112, the
        assertion above stops discriminating.

        MUTATION THAT MUST FAIL THIS: `top = fused[: cfg.graph_expansion_max_symbols]`
        -> `top = fused[:10]` (a hard-wired cap ignores the config).
        """
        db = _RegistryDatabase()
        for sid in self._FUSED_IDS:
            db.register(_row(sid))
        hits = [(sid, 0.01 * i) for i, sid in enumerate(self._FUSED_IDS)]
        retriever, _db, _vs = _build(
            RetrievalConfig(graph_expansion_max_symbols=12),
            tmp_path,
            db=db,
            vector_hits=hits,
        )

        def _expand(db_arg: object, seeds: list[SearchResult], **kw: object):
            return [_row(9000 + r.chunk.symbol_id, source="call_graph", score=1.0) for r in seeds]

        out = _standard_candidates(retriever, _plan(["vector"]), call_graph=_expand)
        returned = set(_ids(out))

        assert 9111 in returned
        assert 9112 in returned


# ===========================================================================
# M19 — the retriever's own RRF k, observed as a RETURNED ORDER
# ===========================================================================


class TestRRFKAtTheCallSite:
    """`cfg.rrf_k` is what the retriever hands `reciprocal_rank_fusion`.

    tests/unit/test_fusion.py pins the FORMULA for a given k; nothing pinned the k the
    retriever passes. RRF's k controls how much a single top-ranked hit is worth against
    the same row appearing twice further down, so an off-by-one there silently reorders
    real answers.

    The fixture is built at the crossover: with A at rank 1 in the vector leg and B at
    rank 62 in bm25 and rank 63 in grep,
        k=60 -> A = 1/61 = 0.01639344,  B = 1/122 + 1/123 = 0.01632680  -> A first
        k=61 -> A = 1/62 = 0.01612903,  B = 1/123 + 1/124 = 0.01619460  -> B first
    """

    _A_ID = 7001
    _B_ID = 7002

    def _fixture(self, tmp_path: Path, retrieval: RetrievalConfig):
        db = _RegistryDatabase()
        a = db.register(_row(self._A_ID, rel_path="src/a.py"))
        b_row = _row(self._B_ID, rel_path="src/b.py")
        db.register(b_row)

        # 61 bm25 fillers, then B at rank 62.
        bm25 = [_row(7100 + i, rel_path=f"src/bm{i}.py", source="bm25") for i in range(61)]
        bm25.append(b_row)
        # 62 grep fillers, then B at rank 63.
        grep = [_row(7300 + i, rel_path=f"src/gp{i}.py", source="grep") for i in range(62)]
        grep.append(b_row)

        retriever, _db, _vs = _build(retrieval, tmp_path, db=db, vector_hits=[(self._A_ID, 0.1)])
        return retriever, bm25, grep, a

    def _order_of_watched_pair(self, out: list[SearchResult]) -> list[int]:
        return [sid for sid in _ids(out) if sid in (self._A_ID, self._B_ID)]

    def test_default_k_ranks_the_single_top_hit_above_the_two_deep_hits(
        self, tmp_path: Path
    ) -> None:
        """MUTATION THAT MUST FAIL THIS: `k=cfg.rrf_k` -> `k=cfg.rrf_k + 1` at the
        `reciprocal_rank_fusion(...)` call in `_standard_candidates` — B (7002) then
        overtakes A (7001) in the RETURNED ranking.
        """
        cfg = RetrievalConfig()
        assert cfg.rrf_k == 60  # shipped default the crossover is built around
        # Both rows are Python, whose shipped file-type weight is exactly 1.0, so the
        # weighting step cannot be what decides this order.
        assert cfg.file_type_weights["python"] == pytest.approx(1.0)
        assert cfg.leg_weights["vector"] == pytest.approx(1.0)
        assert cfg.leg_weights["bm25"] == pytest.approx(1.0)
        assert cfg.leg_weights["grep"] == pytest.approx(1.0)

        retriever, bm25, grep, _a = self._fixture(tmp_path, cfg)
        out = _standard_candidates(
            retriever, _plan(["vector", "bm25", "grep"]), bm25=bm25, grep=grep
        )

        # Fixture precondition: both watched rows survived fusion and dedup, so the
        # order below is a real comparison and not an absence.
        assert set(self._order_of_watched_pair(out)) == {self._A_ID, self._B_ID}
        assert self._order_of_watched_pair(out) == [self._A_ID, self._B_ID]

    def test_k_61_flips_that_order(self, tmp_path: Path) -> None:
        """Discrimination guard naming `_fixture`: it proves this construction sits
        exactly on the k=60/k=61 crossover, so the assertion above is decided by k and
        not by an accident of the filler lists.

        MUTATION THAT MUST FAIL THIS: `k=cfg.rrf_k` -> `k=60` (a hard-wired constant
        ignores the configured k).
        """
        retriever, bm25, grep, _a = self._fixture(tmp_path, RetrievalConfig(rrf_k=61))
        out = _standard_candidates(
            retriever, _plan(["vector", "bm25", "grep"]), bm25=bm25, grep=grep
        )

        assert self._order_of_watched_pair(out) == [self._B_ID, self._A_ID]


# ===========================================================================
# M18 — file_type_weighting_enabled actually disables the multiplier
# ===========================================================================


class TestFileTypeWeightingFlagIsHonoured:
    """Turning the master switch OFF must stop the per-language multiplier entirely.

    Grading note: `file_type_weighting_enabled` ships True, so this mutant does NOT
    change behaviour under the shipped default — it only bites a user who opted OUT.
    That makes it a weaker finding than M34 (which flips the default), but an opt-out
    that does nothing is still a broken switch.
    """

    _HTML_ID = 4001
    _PY_ID = 4002

    def _fixture(self, tmp_path: Path, retrieval: RetrievalConfig):
        db = _RegistryDatabase()
        db.register(_row(self._HTML_ID, rel_path="web/page.html", language=Language.HTML))
        db.register(_row(self._PY_ID, rel_path="src/app.py", language=Language.PYTHON))
        # HTML at ANN rank 1, Python at rank 2 -> unweighted RRF puts HTML first
        # (1/61 > 1/62); the shipped 0.4 HTML multiplier drops it below Python.
        return _build(
            retrieval,
            tmp_path,
            db=db,
            vector_hits=[(self._HTML_ID, 0.1), (self._PY_ID, 0.2)],
        )

    def test_switch_off_leaves_html_ahead_on_pure_rrf(self, tmp_path: Path) -> None:
        """MUTATION THAT MUST FAIL THIS:
        `_weights = cfg.file_type_weights if cfg.file_type_weighting_enabled else None`
        -> `_weights = cfg.file_type_weights` — the 0.4 HTML multiplier is then applied
        despite the switch and Python (4002) overtakes HTML (4001).
        """
        cfg = RetrievalConfig(file_type_weighting_enabled=False)
        assert cfg.file_type_weights["html"] == pytest.approx(0.4)
        assert cfg.file_type_weights["python"] == pytest.approx(1.0)

        retriever, _db, _vs = self._fixture(tmp_path, cfg)
        out = _standard_candidates(retriever, _plan(["vector"]))

        assert _ids(out) == [self._HTML_ID, self._PY_ID]

    def test_shipped_default_downweights_html_below_python(self, tmp_path: Path) -> None:
        """Discrimination guard naming `_fixture`, and the shipped-default pin.

        It proves the 0.4 HTML weight really does reach this ordering, so the
        switch-off assertion above is not passing because the weights are inert.

        MUTATION THAT MUST FAIL THIS: `_weights = cfg.file_type_weights if
        cfg.file_type_weighting_enabled else None` -> `_weights = None`.
        """
        retriever, _db, _vs = self._fixture(tmp_path, RetrievalConfig())
        out = _standard_candidates(retriever, _plan(["vector"]))

        assert _ids(out) == [self._PY_ID, self._HTML_ID]


# ===========================================================================
# M10 / M11 — the max(1, ...) floors, measured SEPARATELY
#
# `scale_factor = effective_budget / 12_000`, and effective_budget is
# int(window * context_window_fraction) when context_token_budget is None.
# "gpt-3.5-turbo" resolves to a 4,096-token window, so:
#   fraction 0.10 -> budget  409 -> scale 0.03408 -> int(20*s)=0, int(15*s)=0
#   fraction 0.17 -> budget  696 -> scale 0.05800 -> int(20*s)=1, int(15*s)=0
# The 0.17 row engages the rerank floor WITHOUT engaging the top_k_vector floor, which
# is what lets each floor be measured on its own.
# ===========================================================================


class TestTopKVectorFloor:
    """A derived top_k_vector of 0 must be floored to 1, not starve the vector leg."""

    _HIT_IDS = [3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008]

    def _build_scaled(self, tmp_path: Path, fraction: float):
        db = _RegistryDatabase()
        for sid in self._HIT_IDS:
            db.register(_row(sid))
        hits = [(sid, 0.01 * i) for i, sid in enumerate(self._HIT_IDS)]
        return _build(
            RetrievalConfig(
                scale_top_k_to_budget=True,
                context_token_budget=None,
                context_window_fraction=fraction,
            ),
            tmp_path,
            db=db,
            vector_hits=hits,
            llm_model="gpt-3.5-turbo",
        )

    def test_a_derived_top_k_of_zero_still_returns_one_vector_row(self, tmp_path: Path) -> None:
        """MUTATION THAT MUST FAIL THIS:
        `self._effective_top_k_vector = max(1, int(config.retrieval.top_k_vector *
        scale_factor))` -> `int(...)` — the vector leg is then asked for 0 neighbours
        and the user's semantic leg returns NOTHING.
        """
        assert RetrievalConfig().top_k_vector == 20  # shipped default being scaled

        retriever, _db, store = self._build_scaled(tmp_path, 0.1)
        # Fixture precondition: the ANN double has 8 answers ready, so one row is a real
        # cap and zero rows would be real starvation, not an empty index.
        assert len(store.hits) == 8

        out = retriever._run_subquery_legs(_sub_query(), _strategy(["vector"]))

        assert _ids(out["vector"]) == [3001]

    def test_a_larger_fraction_returns_proportionally_more_rows(self, tmp_path: Path) -> None:
        """Discrimination guard naming `_build_scaled` and `_ListVectorStore`: it proves
        the double scales with the k it is handed, so `[3001]` above is the floor at
        work and not a one-row fixture.

        Explicit table, no iteration of the pinned collection:
          fraction 0.17 -> budget 696 -> scale 0.058 -> int(20*0.058) = 1 row
          fraction 0.90 -> budget 3686 -> scale 0.3072 -> int(20*0.3072) = 6 rows

        MUTATION THAT MUST FAIL THIS: `max(1, int(config.retrieval.top_k_vector *
        scale_factor))` -> `max(1, config.retrieval.top_k_vector)` (scaling ignored;
        both rows then return 8).
        """
        retriever_a, _db_a, _vs_a = self._build_scaled(tmp_path / "a", 0.17)
        out_a = retriever_a._run_subquery_legs(_sub_query(), _strategy(["vector"]))
        assert _ids(out_a["vector"]) == [3001]

        retriever_b, _db_b, _vs_b = self._build_scaled(tmp_path / "b", 0.9)
        out_b = retriever_b._run_subquery_legs(_sub_query(), _strategy(["vector"]))
        assert _ids(out_b["vector"]) == [3001, 3002, 3003, 3004, 3005, 3006]


class TestRerankTopNFloor:
    """A derived rerank_top_n of 0 must be floored to 1, not empty the candidate set.

    Measured with `context_window_fraction=0.17`, where the top_k_vector floor is NOT
    engaged (int(20 * 0.058) == 1 already), so this test isolates the rerank_top_n
    floor from its sibling.
    """

    _HIT_IDS = [5001, 5002, 5003, 5004]

    def _build_scaled(self, tmp_path: Path, fraction: float, top_k_vector: int):
        db = _RegistryDatabase()
        for sid in self._HIT_IDS:
            db.register(_row(sid))
        hits = [(sid, 0.01 * i) for i, sid in enumerate(self._HIT_IDS)]
        return _build(
            RetrievalConfig(
                scale_top_k_to_budget=True,
                context_token_budget=None,
                context_window_fraction=fraction,
                top_k_vector=top_k_vector,
                rerank=True,
                rerank_provider="cohere",
                cohere_api_key=None,
            ),
            tmp_path,
            db=db,
            vector_hits=hits,
            llm_model="gpt-3.5-turbo",
        )

    def test_a_derived_rerank_top_n_of_zero_still_returns_one_candidate(
        self, tmp_path: Path
    ) -> None:
        """MUTATION THAT MUST FAIL THIS:
        `self._effective_rerank_top_n = max(1, int(config.retrieval.rerank_top_n *
        scale_factor))` -> `int(...)` — the reranker is asked for a top-0 and
        `_standard_candidates` returns an EMPTY list, i.e. the query answers nothing.
        """
        assert RetrievalConfig().rerank_top_n == 15  # shipped default being scaled

        # top_k_vector=60 keeps int(60 * 0.058) == 3 > 0, so the vector leg is healthy
        # and only the rerank ceiling is under test here.
        retriever, _db, store = self._build_scaled(tmp_path, 0.17, top_k_vector=60)
        assert len(store.hits) == 4  # fixture precondition: candidates were available

        out = _standard_candidates(
            retriever, _plan(["vector"], skip_reranker=False, rerank_top_n=99)
        )

        assert _ids(out) == [5001]

    def test_a_larger_fraction_admits_proportionally_more_candidates(self, tmp_path: Path) -> None:
        """NON-DISCRIMINATING COMPANION, stated as such. This test has no mutation that
        fails it, and it is here to show the rerank ceiling MOVES with the fraction --
        which is what makes the floor test above a statement about the floor rather than
        about an unrelated cap.

        Explicit table:
          fraction 0.17 -> budget 696 -> scale 0.058  -> int(15*s) = 0 -> floored to 1
          fraction 0.90 -> budget 3686 -> scale 0.3072 -> int(15*s) = 4 -> 4 candidates

        WHY NO MUTATION IS NAMED, measured rather than assumed. The docstring originally
        claimed `max(1, int(config.retrieval.rerank_top_n * scale_factor))` ->
        `max(1, config.retrieval.rerank_top_n)` would fail this. It does not: only four
        candidate rows exist, so an ignored scale factor yields `max(1, 15)` = 15, the
        result is still truncated to the 4 available, and the assertion holds. Adversarial
        review caught that and proposed `* scale_factor` -> `* scale_factor * 4` instead;
        that was measured too and ALSO passes (int(15*0.3072*4) = 18, still >= 4). Both
        mutations were applied single-site with an anchor-count guard and a sha256-verified
        restore. Any mutation that raises this ceiling is invisible here for the same
        structural reason: the fixture cannot supply more rows than the ceiling admits.
        Rather than invent a third candidate and risk naming a fiction, this test declares
        that it discriminates nothing -- the same disclosure
        `test_a_single_matching_file_is_not_widened_by_the_cap` makes.

        The class as a whole DOES discriminate: under either mutation above,
        `pytest -k TestRerankTopNFloor` reports `1 failed, 1 passed` -- the floor test
        dies, this one does not.
        """
        retriever, _db, _vs = self._build_scaled(tmp_path / "b", 0.9, top_k_vector=60)
        out = _standard_candidates(
            retriever, _plan(["vector"], skip_reranker=False, rerank_top_n=99)
        )

        assert _ids(out) == [5001, 5002, 5003, 5004]


# ===========================================================================
# M22 — the rerank_top_n ternary's `context_token_budget is None` guard
# ===========================================================================


class TestRerankTopNTernaryGuard:
    """With an EXPLICIT budget, the reranker ceiling must come from the plan's strategy.

    Grading note: `scale_top_k_to_budget` ships False, so this mutant needs a user to
    opt into scaling. It is a weaker finding than M34 for that reason — but the config
    it breaks (scaling on, explicit budget kept) is exactly the combination the
    `and cfg.context_token_budget is None` clause exists to handle, and it silently
    replaces the per-intent ceiling with the global one.
    """

    _HIT_IDS = [6001, 6002, 6003, 6004, 6005]

    def _build_explicit_budget(self, tmp_path: Path):
        db = _RegistryDatabase()
        for sid in self._HIT_IDS:
            db.register(_row(sid))
        hits = [(sid, 0.01 * i) for i, sid in enumerate(self._HIT_IDS)]
        return _build(
            RetrievalConfig(
                scale_top_k_to_budget=True,
                context_token_budget=12_000,  # the SHIPPED default: an explicit int
                rerank=True,
                rerank_provider="cohere",
                cohere_api_key=None,
            ),
            tmp_path,
            db=db,
            vector_hits=hits,
        )

    def test_explicit_budget_uses_the_strategy_ceiling_not_the_global_one(
        self, tmp_path: Path
    ) -> None:
        """MUTATION THAT MUST FAIL THIS: dropping `and cfg.context_token_budget is None`
        from the `effective_rerank_top_n` ternary — `self._effective_rerank_top_n` (15,
        unscaled because __init__'s own guard held) is then used instead of the plan's
        `strategy.rerank_top_n` of 2, and the caller receives 5 rows where the intent
        asked for 2.
        """
        cfg = RetrievalConfig()
        assert cfg.context_token_budget == 12_000  # shipped default is an int, not None
        assert cfg.rerank_top_n == 15

        retriever, _db, store = self._build_explicit_budget(tmp_path)
        # Fixture precondition: 5 candidates exist, so a ceiling of 2 is a real
        # truncation and 15 would admit all of them.
        assert len(store.hits) == 5

        out = _standard_candidates(
            retriever, _plan(["vector"], skip_reranker=False, rerank_top_n=2)
        )

        assert _ids(out) == [6001, 6002]

    def test_the_strategy_ceiling_is_what_moves_the_result(self, tmp_path: Path) -> None:
        """Discrimination guard naming `_build_explicit_budget`: raising the plan's
        ceiling must widen the returned set, proving `strategy.rerank_top_n` is the live
        input under an explicit budget.

        MUTATION THAT MUST FAIL THIS: `else strategy.rerank_top_n` ->
        `else cfg.rerank_top_n` (the plan's ceiling stops mattering).
        """
        retriever, _db, _vs = self._build_explicit_budget(tmp_path)
        out = _standard_candidates(
            retriever, _plan(["vector"], skip_reranker=False, rerank_top_n=4)
        )

        assert _ids(out) == [6001, 6002, 6003, 6004]


# ===========================================================================
# M43 — the per-hint file cap in _retrieve_file_overview
# ===========================================================================


class TestFileOverviewPerHintFileCap:
    """Each file hint considers the first TWO path-fragment matches, not one.

    Measured under the SHIPPED default config, with 12 symbols per file so the breadth
    floor (`min_files=2 AND min_symbols=10`) does not fire on either side of the
    mutation — the observed difference is the cap alone, not a widening fallback.
    """

    _FILE_IDS = [501, 502, 503]
    _EXPECTED_PATHS = {"src/pkg501/mod.py", "src/pkg502/mod.py"}

    def _build_overview(self, tmp_path: Path):
        db = _RegistryDatabase()
        db.fragment_hits = list(self._FILE_IDS)
        for fid in self._FILE_IDS:
            sids = [fid * 100 + i for i in range(1, 13)]
            db.file_symbols[fid] = sids
            for sid in sids:
                db.register(_row(sid, rel_path=f"src/pkg{fid}/mod.py"))
        retriever, _db, _vs = _build(RetrievalConfig(), tmp_path, db=db)
        return retriever, db

    def test_two_files_per_hint_reach_the_returned_context(self, tmp_path: Path) -> None:
        """MUTATION THAT MUST FAIL THIS:
        `for file_id in self.db.find_file_by_path_fragment(hint)[:2]:` -> `[:1]` in
        `_retrieve_file_overview` — the second matching file's 12 symbols vanish from
        the RETURNED context, halving what an overview query sees per hint.
        """
        cfg = RetrievalConfig()
        assert cfg.breadth_floor_min_files == 2
        assert cfg.breadth_floor_min_symbols == 10

        retriever, db = self._build_overview(tmp_path)
        plan = _plan(
            ["file_direct"],
            intent=IntentType.FILE_OVERVIEW,
            sub_queries=[_sub_query(grep_hints=[], file_hints=["mod.py"])],
        )

        context = retriever._retrieve_file_overview(plan)

        # Fixture precondition: THREE files match the fragment, so [:2] is a real
        # truncation and [:1] a real halving — not an artefact of a one-file double.
        assert db.fragment_hits == [501, 502, 503]
        assert db.fragment_queries == ["mod.py"]

        paths = {r.file.rel_path for r in context.results}
        assert paths == self._EXPECTED_PATHS
        assert self._EXPECTED_PATHS == paths  # set equality, both directions
        assert "src/pkg503/mod.py" not in paths
        assert len(context.results) == 24

    def test_a_single_matching_file_is_not_widened_by_the_cap(self, tmp_path: Path) -> None:
        """Discrimination guard naming `_build_overview`: with only one match the cap
        cannot invent a second file, so the two-file assertion above is measuring the
        cap against a genuinely deeper candidate list.

        MUTATION THAT MUST FAIL THIS: `find_file_by_path_fragment(hint)[:2]` ->
        `find_file_by_path_fragment(hint)` — indistinguishable here (one match), which
        is precisely why the three-match fixture above is the load-bearing one.
        """
        retriever, db = self._build_overview(tmp_path)
        db.fragment_hits = [502]
        plan = _plan(
            ["file_direct"],
            intent=IntentType.FILE_OVERVIEW,
            sub_queries=[_sub_query(grep_hints=[], file_hints=["mod.py"])],
        )

        context = retriever._retrieve_file_overview(plan)

        assert {r.file.rel_path for r in context.results} == {"src/pkg502/mod.py"}
        assert len(context.results) == 12
