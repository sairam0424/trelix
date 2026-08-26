"""GraphBuilder: what the concept cap batches, what it reports, and whether the
PageRank it computes is the centrality that actually reaches the database.

These cover survivors that `tests/unit/test_graph_builder.py` leaves alive. That file
already pins the shipped defect (the 200-symbol cap taking symbols in DB id order rather
than by centrality) from five angles, so none of that is repeated here.

No Mock/MagicMock anywhere: the ConceptExtractor stand-in is a plain class exposing only
`extract_from_symbols`, so a builder that starts calling some other method raises
AttributeError instead of silently passing against an invented attribute.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from trelix.core.config import IndexConfig, RetrievalConfig
from trelix.core.models import CallEdge, GenericEdge, IndexedFile, Language, Symbol, SymbolKind
from trelix.graph.builder import GraphBuilder
from trelix.graph.concepts import SemanticConcept
from trelix.graph.persistence import get_top_central_symbols
from trelix.store.db import Database

# The builder's own values, written as literals on purpose. Importing
# _MAX_CONCEPT_SYMBOLS / _CONCEPT_BATCH_SIZE from the module under test would make every
# assertion below true for whatever value the module happened to hold.
CAP = 200
BATCH = 20


# ── fixtures ─────────────────────────────────────────────────────────────────────


def _new_db(tmp_path: Path) -> tuple[Path, Database, int]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / ".trelix").mkdir()
    db = Database(repo / ".trelix" / "index.db")
    fid = db.upsert_file(
        IndexedFile(
            path=str(repo / "a.py"),
            rel_path="a.py",
            language=Language.PYTHON,
            hash="x",
            size_bytes=10,
        )
    )
    return repo, db, fid


def _add(db: Database, fid: int, name: str, line: int) -> int:
    return db.insert_symbol(
        Symbol(
            file_id=fid,
            name=name,
            qualified_name=name,
            kind=SymbolKind.FUNCTION,
            line_start=line,
            line_end=line,
            signature=f"def {name}()",
            body="",
        )
    )


def _flat_repo(tmp_path: Path, count: int) -> Path:
    """`count` symbols in a shallow call chain — enough structure for a real PageRank."""
    repo, db, fid = _new_db(tmp_path)
    ids = [_add(db, fid, f"fn_{i}", i * 10 + 1) for i in range(count)]
    db.insert_call_edges(
        [
            CallEdge(caller_id=ids[i], callee_name=f"fn_{i + 1}", callee_id=ids[i + 1], line=2)
            for i in range(min(10, count - 1))
        ]
    )
    db._conn.commit()
    db.close()
    return repo


def _deep_vs_shallow_repo(tmp_path: Path) -> tuple[Path, int, int, int]:
    """A graph whose PageRank order is the REVERSE of any purely local degree measure.

    `deep` has a single caller, but that caller is a 20-times-called hub, so PageRank
    flows most of the hub's mass into it. `shallow` has three callers and every one is a
    dead end. Any degree-based score therefore ranks shallow above deep while PageRank
    ranks deep above shallow — which is the only reason this fixture can tell the two
    apart. Measured: PageRank deep 0.906 > shallow 0.197; degree deep 0.04 < shallow 0.12.
    """
    repo, db, fid = _new_db(tmp_path)
    hub = _add(db, fid, "hub", 1)
    deep = _add(db, fid, "deep", 2)
    shallow = _add(db, fid, "shallow", 3)
    hub_callers = [_add(db, fid, f"hc{i}", 10 + i) for i in range(20)]
    shallow_callers = [_add(db, fid, f"sc{i}", 50 + i) for i in range(3)]

    edges = [CallEdge(caller_id=c, callee_name="hub", callee_id=hub, line=1) for c in hub_callers]
    edges.append(CallEdge(caller_id=hub, callee_name="deep", callee_id=deep, line=1))
    edges += [
        CallEdge(caller_id=c, callee_name="shallow", callee_id=shallow, line=1)
        for c in shallow_callers
    ]
    db.insert_call_edges(edges)
    db._conn.commit()
    db.close()
    return repo, hub, deep, shallow


def _ticket_linked_repo(tmp_path: Path) -> tuple[Path, int, int]:
    """`hub` wins on call-graph structure; `target` wins only under personalization.

    Personalization concentrates the teleport vector on symbols adjacent to an artifact
    node, and `target` is the only one with a ticket edge. Plain PageRank must rank the
    8-times-called hub first, so the two settings produce opposite orders.
    """
    repo, db, fid = _new_db(tmp_path)
    hub = _add(db, fid, "hub", 1)
    target = _add(db, fid, "target", 2)
    callers = [_add(db, fid, f"caller{i}", 10 + i) for i in range(8)]
    db.insert_call_edges(
        [CallEdge(caller_id=c, callee_name="hub", callee_id=hub, line=1) for c in callers]
    )
    db.insert_generic_edges(
        [
            GenericEdge(
                from_symbol_id=target,
                source_ref="ticket:PROJ-1",
                edge_kind="references_ticket",
            )
        ]
    )
    db._conn.commit()
    db.close()
    return repo, hub, target


def _stub_extractor(batches: list[list[Symbol]], concepts_per_batch: int = 0) -> type:
    """A plain ConceptExtractor stand-in that records each batch it is handed."""

    class _StubExtractor:
        def __init__(self, llm_config: Any) -> None:
            self._llm_config = llm_config

        def extract_from_symbols(
            self, symbols: list[Symbol], max_symbols: int = 20
        ) -> list[SemanticConcept]:
            batches.append(list(symbols))
            call_index = len(batches)
            return [
                SemanticConcept(
                    name=f"concept_{call_index}_{k}",
                    category="domain",
                    importance=5,
                    source_symbol_ids=[s.id for s in symbols if s.id is not None],
                )
                for k in range(concepts_per_batch)
            ]

    return _StubExtractor


def _centrality_column(repo: Path) -> dict[int, float]:
    db = Database(repo / ".trelix" / "index.db")
    try:
        rows = db._conn.execute("SELECT symbol_id, centrality FROM graph_metadata").fetchall()
        return {int(r[0]): float(r[1] or 0.0) for r in rows}
    finally:
        db.close()


def _build(repo: Path, batches: list[list[Symbol]], concepts_per_batch: int = 0) -> Any:
    config = IndexConfig(repo_path=str(repo))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "trelix.graph.builder.ConceptExtractor",
            _stub_extractor(batches, concepts_per_batch),
        )
        return GraphBuilder(config).build(extract_concepts=True)


# ── the concept cap's batching ───────────────────────────────────────────────────


class TestConceptBatching:
    def test_the_cap_is_spent_as_ten_batches_of_twenty(self, tmp_path: Path) -> None:
        """MUTATION: _CONCEPT_BATCH_SIZE = 20 -> 1, 200, or any other divisor of the cap.

        Nothing pinned the batch size, and every divisor of 200 kept the suite green
        because the only thing asserted was that 200 symbols were seen in total. The batch
        size is not cosmetic: it is one paid LLM call per batch, so 20 -> 1 turns a single
        extraction into 200 requests, and 20 -> 200 puts every symbol into one prompt.

        Non-divisors are caught here too, and for a second reason. `ranked` holds all 250
        symbols while the loop bounds are the capped 200, so a batch size that does not
        divide the cap makes the final slice overrun it: at 21 the extractor is handed 210
        symbols, i.e. 10 more than the cap claims to allow. The total assertion below is
        what states the cap as a fact about what the extractor actually receives.
        """
        repo = _flat_repo(tmp_path, count=250)
        batches: list[list[Symbol]] = []

        result = _build(repo, batches)

        # Precondition: the cap must actually be truncating, or the batching is trivial.
        assert result.concept_symbols_total == 250, (
            "_flat_repo no longer produces 250 symbols, so the cap does not bind and this "
            f"test proves nothing: {result.concept_symbols_total}"
        )

        sizes = [len(b) for b in batches]
        assert sizes == [BATCH] * 10, f"expected ten batches of {BATCH}, got {sizes}"

        flat = [s.id for b in batches for s in b]
        assert len(flat) == CAP, (
            f"the extractor received {len(flat)} symbols but the cap is {CAP} — a batch "
            "size that does not divide the cap overruns the final slice"
        )
        assert len(set(flat)) == CAP, "a symbol was handed to the extractor twice"

    def test_concept_count_counts_concepts_not_the_symbols_they_came_from(
        self, tmp_path: Path
    ) -> None:
        """MUTATION: concept_count = len(concepts) -> len(ranked) (or len(symbols)).

        This line was unreachable in the existing tests: their extractor returns [], so
        `if concepts:` is always false and the assignment never runs. With an extractor
        that returns something, `concept_count` must describe concepts. Reporting 250
        would restate the repository's symbol count under a field name that promises
        extracted concepts.
        """
        repo = _flat_repo(tmp_path, count=250)
        batches: list[list[Symbol]] = []

        # One concept per batch, ten batches -> ten concepts. Distinct from both the cap
        # (200) and the symbol total (250), so no off-by-collection can pass by accident.
        result = _build(repo, batches, concepts_per_batch=1)

        assert len(batches) == 10, f"fixture handed the extractor {len(batches)} batches"
        assert result.concept_count == 10, (
            f"concept_count is {result.concept_count}; 10 concepts were returned, while "
            f"{CAP} symbols were considered and 250 exist"
        )


# ── what the truncation warning says ─────────────────────────────────────────────


class TestTruncationReporting:
    def test_no_truncation_warning_when_the_repo_is_exactly_the_cap(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MUTATION: `if concept_symbols_total > _MAX_CONCEPT_SYMBOLS` -> `>=`.

        At exactly 200 symbols nothing is truncated and coverage is 100%, so warning that
        `concept_count` "describes that sample, not the repository" would be false. The
        existing tests bracket this boundary at 250 (above) and 30 (below) and never land
        on it, so the off-by-one survived.
        """
        repo = _flat_repo(tmp_path, count=CAP)
        batches: list[list[Symbol]] = []

        with caplog.at_level(logging.WARNING, logger="trelix.graph.builder"):
            result = _build(repo, batches)

        # Precondition: the fixture must sit exactly ON the boundary, not near it.
        assert result.concept_symbols_total == CAP, (
            f"_flat_repo produced {result.concept_symbols_total} symbols, not exactly the "
            f"cap {CAP} — this test only discriminates `>` from `>=` at the boundary"
        )
        assert result.concept_symbols_considered == CAP

        truncation = [r.getMessage() for r in caplog.records if "of 200 symbols" in r.getMessage()]
        assert truncation == [], (
            f"a full-coverage build claimed its concepts were a truncated sample: {truncation}"
        )

    def test_the_coverage_percentage_is_the_fraction_actually_covered(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MUTATION: `100.0 * considered / total` -> `total / considered`, or dropping 100.0.

        The existing test only matches the substring "200 of 250", so the percentage in
        the same sentence was free to read 125.0% or 0.8%. This message exists to be
        honest about coverage; an inverted ratio makes it claim more coverage than exists.
        200 of 250 is 80.0% — written as a literal, not recomputed from the module.
        """
        repo = _flat_repo(tmp_path, count=250)
        batches: list[list[Symbol]] = []

        with caplog.at_level(logging.WARNING, logger="trelix.graph.builder"):
            result = _build(repo, batches)

        assert (result.concept_symbols_considered, result.concept_symbols_total) == (CAP, 250), (
            "fixture drifted; 80.0% below is only the right answer for 200 of 250"
        )
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("200 of 250 symbols (80.0%)" in m for m in messages), messages


# ── the centrality that reaches the database ─────────────────────────────────────


class TestPersistedCentrality:
    def test_the_persisted_centrality_is_pagerank_not_the_degree_fallback(
        self, tmp_path: Path
    ) -> None:
        """MUTATIONS: dropping the second `save_graph_metadata`, or negating the
        `if node_id in cg.nx.nodes` guard, or writing 0.0 instead of `score`.

        All three survived, and for one shared reason: `save_graph_metadata` silently
        falls back to `nx.degree_centrality` for any node with no "centrality" attr. So
        when step 3b's scores never reach the nodes, or are written after the last save,
        the column still fills with plausible-looking numbers — a different metric wearing
        the same name. `get_top_central_symbols` orders by that column and feeds retrieval,
        so the substitution is invisible at every existing assertion.

        The existing builder tests cannot catch it because their fixture's hubs rank top
        under both metrics; separating PageRank from degree needs a graph where the two
        disagree, which is what `_deep_vs_shallow_repo` is for.
        """
        repo, hub, deep, shallow = _deep_vs_shallow_repo(tmp_path)
        batches: list[list[Symbol]] = []

        _build(repo, batches)

        # Precondition, read back from the DB rather than asserted about the source: any
        # purely local degree measure must rank shallow ABOVE deep, or the degree fallback
        # would agree with PageRank here and the assertion below would hold either way.
        db = Database(repo / ".trelix" / "index.db")
        try:
            calls = list(db.iter_resolved_calls())
        finally:
            db.close()
        in_degree = {deep: 0, shallow: 0}
        for _caller, callee in calls:
            if callee in in_degree:
                in_degree[callee] += 1
        assert (in_degree[deep], in_degree[shallow]) == (1, 3), (
            "_deep_vs_shallow_repo no longer discriminates PageRank from degree: deep must "
            f"have exactly 1 caller and shallow exactly 3, got {in_degree}"
        )

        centrality = _centrality_column(repo)
        assert centrality, "_deep_vs_shallow_repo persisted no graph_metadata rows at all"

        assert centrality[deep] > centrality[shallow], (
            "the persisted centrality ranks the 3-caller dead end above the symbol fed by "
            f"the 20-times-called hub — that is degree, not PageRank: deep="
            f"{centrality[deep]:.5f} shallow={centrality[shallow]:.5f}"
        )

        db = Database(repo / ".trelix" / "index.db")
        try:
            assert get_top_central_symbols(db, top_n=3) == [hub, deep, shallow], (
                "get_top_central_symbols — the consumer retrieval actually uses — does not "
                "return the PageRank order"
            )
        finally:
            db.close()

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason=(
            "PRE-EXISTING DEFECT, not a property of this test: GraphBuilder.build() raises "
            "ValueError on any index containing a cross-source generic edge. "
            "detect_communities() does int(node_id) over every node, but CodeGraph keys "
            "artifact nodes by the source_ref STRING ('ticket:PROJ-1'), and its except "
            "handler repeats the same unguarded cast — so the fallback raises too and the "
            "error escapes build(). Verified on pristine source: the identical repo builds "
            "fine with call edges only and raises after adding exactly one generic edge. "
            "This is strict+raises=ValueError deliberately: when the crash is fixed this "
            "test XPASSes, strict turns that into a failure, and whoever fixes it removes "
            "the marker and inherits a real guard for the config flag. Until then the "
            "hardcode-True / hardcode-False mutants of "
            "pagerank_personalization_enabled CANNOT be killed through build(), because "
            "the flag only changes anything when artifact nodes exist and artifact nodes "
            "are exactly what crashes the build."
        ),
    )
    def test_the_builder_honours_the_personalization_config_flag(self, tmp_path: Path) -> None:
        """MUTATION: `personalization_enabled=self._config.retrieval.
        pagerank_personalization_enabled` -> hardcoded True or hardcoded False.

        Both constants passed the whole suite, i.e. nothing observed that the builder
        reads the flag at all. Asserting only one direction would kill only one constant,
        so both settings are exercised on the same graph and must produce opposite orders.

        This test does NOT kill those mutants today — see the xfail reason. It is here
        because it is the assertion that will kill them the moment build() stops crashing.
        """
        off_repo, off_hub, off_target = _ticket_linked_repo(tmp_path / "off")
        on_repo, on_hub, on_target = _ticket_linked_repo(tmp_path / "on")

        GraphBuilder(IndexConfig(repo_path=str(off_repo))).build(extract_concepts=False)
        GraphBuilder(
            IndexConfig(
                repo_path=str(on_repo),
                retrieval=RetrievalConfig(pagerank_personalization_enabled=True),
            )
        ).build(extract_concepts=False)

        off = _centrality_column(off_repo)
        on = _centrality_column(on_repo)

        # Precondition: with the flag off, plain call-graph structure must put the
        # 8-times-called hub above the ticket-linked leaf. If _ticket_linked_repo ever
        # stops satisfying this, the "on" assertion below becomes unfalsifiable.
        assert off[off_hub] > off[off_target], (
            "_ticket_linked_repo no longer discriminates — with personalization disabled "
            f"the hub must outrank the ticket target: hub={off[off_hub]:.5f} "
            f"target={off[off_target]:.5f}"
        )
        assert on[on_target] > on[on_hub], (
            "enabling pagerank_personalization_enabled did not change the builder's "
            f"output, so the config flag is not reaching compute_pagerank: "
            f"target={on[on_target]:.5f} hub={on[on_hub]:.5f}"
        )
