"""Shared helpers for the synthesis eval v2 tests (not a test module; imported as
``tests.unit.synthesis_eval_fixtures`` by test_synthesis_eval_v2.py, test_synthesis_eval_records.py
and test_synthesis_eval_metrics.py; the next change in this series extends it, never duplicates it).
The constants here are INPUTS; every value a test compares against is a literal in the test."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tests.fixtures.citation_context import BEARER_BLOCK, VERIFY_BLOCK, bearer_result, verify_result
from trelix.core.config import IndexConfig
from trelix.core.models import RetrievedContext
from trelix.eval.synthesis import SynthesisEvalHarness
from trelix.eval.synthesis_records import SynthesisRecord
from trelix.retrieval.citations import CitationSource

# The one place the aggregate's key set is written in the tests.
AGGREGATE_KEYS = frozenset(
    {"hallucination_rate", "completeness", "faithfulness", "overall"}
    | {"n_queries", "unscoreable", "n_unanswerable", "n_abstained"}
    | {"abstention_precision", "abstention_recall", "citation_validity", "n_cited_entries"}
    | {"citation_recall", "n_expected_citation_entries", "citations_total", "citations_valid"}
)
# A v1 line (no v2 field) for the query substituted with `%`.
V1 = '{"query": "%s", "expected_answer_fragments": ["jwt"], "expected_symbols": ["AuthMiddleware.verify"]}'  # noqa: E501
# The patched Synthesizer's answer: a scored V1 line gives hallucination 0.0 (the expected
# symbol was retrieved) and completeness 1.0 (`jwt` is present).
ANSWER = "The jwt is checked by AuthMiddleware.verify."
# The tagged retrieval's [C1]..[C4] over F2 on disk (`write_f2`): tags 1 and 2 are valid, tag 3
# ends past jwt.py's 25 lines (line_out_of_range), tag 4's file is absent (file_missing).
TAGGED_SOURCES = (
    CitationSource(1, 11, "src/auth/middleware.py", 42, 67, "AuthMiddleware.verify"),
    CitationSource(2, 33, "src/auth/middleware.py", 70, 80, "AuthMiddleware.bearer"),
    CitationSource(3, 22, "src/auth/jwt.py", 10, 30, "decode_token"),
    CitationSource(4, 44, "src/db/pool.py", 1, 10, "Pool.acquire"),
)
# Golden G6 of the design's R5b example (ids e1..e6) and the answers its Synthesizer gives.
G6_LINES: tuple[str, ...] = (
    '{"id": "e1", "query": "g1", "expected_answer_fragments": ["jwt"], "expected_symbols": ["AuthMiddleware.verify"], "expected_citations": ["src/auth/middleware.py"]}',  # noqa: E501
    '{"id": "e2", "query": "g2", "expected_answer_fragments": ["pool"], "expected_citations": ["src/db/pool.py"]}',  # noqa: E501
    '{"id": "e3", "query": "g3", "answerable": false}',
    '{"id": "e4", "query": "g4", "answerable": false}',
    '{"id": "e5", "query": "g5", "expected_answer_fragments": ["retry"]}',
    '{"id": "e6", "query": "g6", "answerable": false}',
)
G6_ANSWERS: dict[str, str] = {
    "g1": "The jwt is checked by AuthMiddleware.verify [C1].",
    "g2": "Connections come from the pool [C4].",
    "g3": "\nINSUFFICIENT_EVIDENCE: no retrieved chunk mentions rate limiting.",
    "g4": "Eviction happens in bearer [C2] [C9].",
    "g5": "INSUFFICIENT_EVIDENCE: nothing about retries was retrieved.",
    "g6": "Sessions are written by verify [C1] and bearer [C2].",
}


def write_golden(tmp_path: Path, lines: Sequence[str]) -> str:
    """Write `lines` verbatim, each followed by a newline; return the path as a string."""
    path = tmp_path / "golden_synthesis.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return str(path)


def tagged_context(query: str) -> RetrievedContext:
    """The verify and bearer blocks tagged [C1] and [C2], with `TAGGED_SOURCES` as their sources."""
    return RetrievedContext(
        query=query,
        results=[verify_result(), bearer_result()],
        context_text="[C1] " + VERIFY_BLOCK + "\n[C2] " + BEARER_BLOCK,
        total_tokens=40,
        citation_sources=TAGGED_SOURCES,
    )


class StubRetriever:
    """The verify context for every query (`tagged_context` when `tagged`); raises for the named."""

    def __init__(self, raise_for: frozenset[str] = frozenset(), tagged: bool = False) -> None:
        self.raise_for = raise_for
        self.tagged = tagged
        self.calls = 0

    def retrieve(self, query: str) -> RetrievedContext:
        self.calls += 1
        if query in self.raise_for:
            raise RuntimeError("index is unreadable")
        if self.tagged:
            return tagged_context(query)
        return RetrievedContext(
            query=query, results=[verify_result()], context_text=VERIFY_BLOCK, total_tokens=20
        )


def write_f2(tmp_path: Path) -> Path:
    """F2 on disk under `tmp_path` (the harness's repo_root), as test_citation_verifier.py's `repo`
    writes it: middleware.py 80 newline-terminated lines (counts 81), jwt.py 25 lines with no
    trailing newline (counts 25), no src/db/pool.py."""
    auth = tmp_path / "src" / "auth"
    auth.mkdir(parents=True)
    (auth / "middleware.py").write_bytes("".join(f"line {i}\n" for i in range(1, 81)).encode())
    (auth / "jwt.py").write_bytes("\n".join(f"line {i}" for i in range(1, 26)).encode())
    return tmp_path


def answers_from(table: Mapping[str, str]) -> Callable[[RetrievedContext], str]:
    """A `synthesize` side effect that answers each context's query from `table`."""
    return lambda context: table[context.query]


def make_harness(tmp_path: Path, retriever: StubRetriever) -> SynthesisEvalHarness:
    """A harness over `retriever`, built as tests/unit/test_synthesis_eval.py builds one."""
    harness = SynthesisEvalHarness.__new__(SynthesisEvalHarness)
    harness._config = IndexConfig(repo_path=str(tmp_path))
    harness._retriever = retriever
    return harness


def make_record(**overrides: Any) -> SynthesisRecord:
    """A complete record with every field defaulted: the one way the tests build one."""
    fields: dict[str, Any] = {"id": "q0001", "query": "q", "answerable": True, "error": None}
    fields |= {"abstained": False, "abstain_reason": None}
    fields |= {"hallucination": 0.0, "completeness": 1.0, "faithfulness": 1.0, "overall": 1.0}
    fields |= {"citations_total": 0, "citations_valid": 0, "citation_statuses": {}}
    fields |= {"expected_citations_hit": 0, "expected_citations_total": 0}
    return SynthesisRecord(**{**fields, **overrides})


def _cited(
    total: int, valid: int, statuses: dict[str, int], hit: int = 0, expected: int = 0
) -> dict[str, Any]:
    return {
        "citations_total": total,
        "citations_valid": valid,
        "citation_statuses": statuses,
        "expected_citations_hit": hit,
        "expected_citations_total": expected,
    }


def metrics_records() -> list[SynthesisRecord]:
    """The six hand-built records of the R5b example (`M`, ids e1..e6), written here ONCE: the
    metrics test aggregates them, the records test writes them and the CLI test prints them."""
    abstained = {"abstained": True, "abstain_reason": "insufficient_evidence"}
    zero_scores = {"hallucination": 0.0, "completeness": 0.0, "faithfulness": 0.0, "overall": 0.0}
    # `valid` before `unknown`, as the verifier counts e4's `[C2] [C9]`; the file writer sorts.
    valid_then_unknown = {"valid": 1, "unknown": 1}
    return [
        make_record(id="e1", query="g1", **_cited(1, 1, {"valid": 1}, hit=1, expected=1)),
        make_record(id="e2", query="g2", **_cited(1, 0, {"file_missing": 1}, expected=1)),
        make_record(id="e3", query="g3", answerable=False, **abstained),
        make_record(id="e4", query="g4", answerable=False, **_cited(2, 1, valid_then_unknown)),
        make_record(id="e5", query="g5", **abstained, **zero_scores),
        make_record(id="e6", query="g6", answerable=False, **_cited(2, 2, {"valid": 2})),
    ]
