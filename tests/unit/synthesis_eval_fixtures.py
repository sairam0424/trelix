"""Shared helpers for the synthesis eval v2 tests (not a test module; imported as
``tests.unit.synthesis_eval_fixtures`` by test_synthesis_eval_v2.py and
test_synthesis_eval_records.py; the next changes in this series extend it, never duplicate it).
The constants here are INPUTS; every value a test compares against is a literal in the test."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tests.fixtures.citation_context import VERIFY_BLOCK, verify_result
from trelix.core.config import IndexConfig
from trelix.core.models import RetrievedContext
from trelix.eval.synthesis import SynthesisEvalHarness
from trelix.eval.synthesis_records import SynthesisRecord

# The one place the aggregate's key set is written in the tests.
AGGREGATE_KEYS = frozenset(
    {"hallucination_rate", "completeness", "faithfulness", "overall"}
    | {"n_queries", "unscoreable", "n_unanswerable"}
)
# A v1 line (no v2 field) for the query substituted with `%`.
V1 = '{"query": "%s", "expected_answer_fragments": ["jwt"], "expected_symbols": ["AuthMiddleware.verify"]}'  # noqa: E501
# The patched Synthesizer's answer: a scored V1 line gives hallucination 0.0 (the expected
# symbol was retrieved) and completeness 1.0 (`jwt` is present).
ANSWER = "The jwt is checked by AuthMiddleware.verify."


def write_golden(tmp_path: Path, lines: Sequence[str]) -> str:
    """Write `lines` verbatim, each followed by a newline; return the path as a string."""
    path = tmp_path / "golden_synthesis.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return str(path)


class StubRetriever:
    """The `AuthMiddleware.verify` context for every query; raises for the named ones."""

    def __init__(self, raise_for: frozenset[str] = frozenset()) -> None:
        self.raise_for = raise_for
        self.calls = 0

    def retrieve(self, query: str) -> RetrievedContext:
        self.calls += 1
        if query in self.raise_for:
            raise RuntimeError("index is unreadable")
        return RetrievedContext(
            query=query, results=[verify_result()], context_text=VERIFY_BLOCK, total_tokens=20
        )


def make_harness(tmp_path: Path, retriever: StubRetriever) -> SynthesisEvalHarness:
    """A harness over `retriever`, built as tests/unit/test_synthesis_eval.py builds one."""
    harness = SynthesisEvalHarness.__new__(SynthesisEvalHarness)
    harness._config = IndexConfig(repo_path=str(tmp_path))
    harness._retriever = retriever
    return harness


def make_record(**overrides: Any) -> SynthesisRecord:
    """A complete record with every field defaulted: the one way the tests build one."""
    fields: dict[str, Any] = {"id": "q0001", "query": "q", "answerable": True, "error": None}
    fields |= {"hallucination": 0.0, "completeness": 1.0, "faithfulness": 1.0, "overall": 1.0}
    return SynthesisRecord(**{**fields, **overrides})
