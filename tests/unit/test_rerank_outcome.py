"""Pins that a reported retrieval score can say which pipeline produced it.

`rerank` defaults to True and `rerank_provider` to "cohere". With no COHERE_API_KEY the
reranker logs a warning and hands back the unranked head, so `trelix eval` printed a
score that a reader would attribute to a Cohere-reranked pipeline when no reranking had
happened. Measured on the 10-query mini_repo fixture, that difference is nDCG@10 0.9631
unranked against 0.9131 reranked — larger than most changes anyone measures with eval,
and invisible in the output.

The warning was already logged. What was missing was a record attached to the number,
which is what these tests pin: every provider reports whether it actually applied, the
outcome rides back on RetrievedContext, and eval prints it.

Deliberately covers the two cases that a naive implementation gets wrong:
  * `xtr` on its SUCCESS path must report applied=False, because that path re-sorts by
    the score each document already had and never reads the query.
  * A run where some queries reranked and others fell back must report MIXED rather than
    picking one, because such a score is a blend of two pipelines.
"""

from __future__ import annotations

from typing import Any

import pytest

from trelix.core.config import RetrievalConfig
from trelix.core.models import RerankOutcome
from trelix.retrieval.reranker import rerank, rerank_with_outcome


def _results(n: int = 3) -> list[Any]:
    """Minimal SearchResults, built the same way tests/unit/test_reranker_core.py does.

    Only `.chunk.chunk_text` and `.score` are read by the code under test.
    """
    from trelix.core.models import Chunk, IndexedFile, Language, SearchResult, Symbol, SymbolKind

    out: list[Any] = []
    for i in range(n):
        out.append(
            SearchResult(
                chunk=Chunk(symbol_id=i, chunk_text=f"body {i}", token_count=2),
                symbol=Symbol(
                    file_id=1,
                    name=f"s{i}",
                    qualified_name=f"s{i}",
                    kind=SymbolKind.FUNCTION,
                    line_start=1,
                    line_end=5,
                    signature=f"def s{i}():",
                    body=f"def s{i}():\n    pass",
                ),
                file=IndexedFile(
                    path=f"/repo/src/f{i}.py",
                    rel_path=f"src/f{i}.py",
                    language=Language.PYTHON,
                    hash="deadbeef",
                    size_bytes=128,
                ),
                score=1.0 - i * 0.1,
                rank=i + 1,
                source="vector",
            )
        )
    return out


# ---------------------------------------------------------------------------
# The outcome type
# ---------------------------------------------------------------------------


def test_describe_names_the_reason_when_it_did_not_apply() -> None:
    assert RerankOutcome("cohere", applied=True).describe() == "cohere"
    assert (
        RerankOutcome("cohere", applied=False, skipped_because="no API key").describe()
        == "cohere (skipped: no API key)"
    )


def test_the_outcome_is_frozen_so_a_verdict_cannot_be_edited_after_the_fact() -> None:
    """A record that a later stage can rewrite is not a record."""
    outcome = RerankOutcome("cohere", applied=False, skipped_because="no API key")

    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError, any raise is the point
        outcome.applied = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Providers report their own outcome
# ---------------------------------------------------------------------------


def test_cohere_without_a_key_reports_skipped_not_applied() -> None:
    """The exact case that made eval's numbers depend on ambient credentials."""
    config = RetrievalConfig(rerank=True, rerank_provider="cohere", cohere_api_key=None)

    results, outcome = rerank_with_outcome("q", _results(), config, top_n=2)

    assert outcome.provider == "cohere"
    assert outcome.applied is False
    assert outcome.skipped_because == "no API key"
    assert len(results) == 2, "must still return the unranked head — a rerank miss is not a failure"


def test_xtr_reports_not_applied_even_on_its_SUCCESS_path() -> None:
    """xtr's own docstring says it reranks NOTHING; the outcome must agree.

    Its single-token approximation makes each score equal the input score, so the
    success path re-sorts by the score a document already had and never reads the
    query. Reporting applied=True here would be the misreport this type exists to stop.
    """
    config = RetrievalConfig(rerank=True, rerank_provider="xtr")

    _, outcome = rerank_with_outcome("q", _results(), config, top_n=3)

    assert outcome.applied is False
    assert "identity function" in (outcome.skipped_because or "")


def test_an_unknown_provider_reports_itself_rather_than_passing_silently() -> None:
    config = RetrievalConfig(rerank=True, rerank_provider="cross_encoder")
    object.__setattr__(config, "rerank_provider", "nonexistent")  # bypass Literal validation

    results, outcome = rerank_with_outcome("q", _results(), config, top_n=2)

    assert outcome.applied is False
    assert "nonexistent" in (outcome.skipped_because or "")
    assert len(results) == 2


def test_no_candidates_reports_why_rather_than_claiming_a_rerank() -> None:
    config = RetrievalConfig(rerank=True, rerank_provider="cohere")

    results, outcome = rerank_with_outcome("q", [], config)

    assert results == []
    assert outcome.applied is False
    assert outcome.skipped_because == "no candidates"


def test_the_original_rerank_contract_is_unchanged() -> None:
    """33 pre-existing call sites want only the results; they must not have to care.

    Counted as module-level `rerank(` calls excluding method calls and the def itself: 34 at
    v3.1.4 (33 in tests, 1 in retriever.py), and the production one is precisely the caller
    that moved to `rerank_with_outcome`. Stated exactly because an earlier draft of this
    docstring said "22", which was a guess.
    """
    config = RetrievalConfig(rerank=True, rerank_provider="cohere", cohere_api_key=None)

    out = rerank("q", _results(), config, top_n=2)

    assert isinstance(out, list), "rerank() must still return a bare list, not a tuple"
    assert len(out) == 2


# ---------------------------------------------------------------------------
# eval reports it
# ---------------------------------------------------------------------------


def _harness_reporting(outcomes: tuple[RerankOutcome | None, ...]) -> str:
    """Summary for a given set of outcomes, without needing a real index.

    EvalHarness.__init__ builds a Retriever, which needs an index on disk; the summary
    reads nothing but the recorded outcomes, so it is exercised directly.
    """
    from trelix.eval.harness import EvalHarness

    harness = object.__new__(EvalHarness)
    harness._rerank_outcomes = outcomes  # type: ignore[attr-defined]
    return harness.rerank_summary()


def test_summary_says_disabled_when_the_stage_was_never_entered() -> None:
    assert _harness_reporting((None, None)) == "disabled"


def test_summary_names_the_provider_when_every_query_reranked() -> None:
    applied = RerankOutcome("cross_encoder", applied=True)

    assert _harness_reporting((applied, applied, applied)) == "cross_encoder"


def test_summary_carries_the_skip_reason_so_the_score_is_self_describing() -> None:
    skipped = RerankOutcome("cohere", applied=False, skipped_because="no API key")

    assert _harness_reporting((skipped,) * 10) == "cohere (skipped: no API key)"


def test_summary_refuses_to_hide_a_run_that_used_TWO_pipelines() -> None:
    """The case that most needs saying, and the one an average would erase.

    Three queries reranked and seven fell back on a transient API failure: the reported
    mean is a blend, and a summary reading "cohere" would conceal precisely what makes
    such a number untrustworthy.
    """
    applied = RerankOutcome("cohere", applied=True)
    failed = RerankOutcome(
        "cohere", applied=False, skipped_because="API call failed after 3 attempt(s)"
    )

    summary = _harness_reporting((applied,) * 3 + (failed,) * 7)

    assert "MIXED" in summary
    assert "3/10 applied" in summary
    assert "API call failed" in summary


def test_summary_counts_queries_that_produced_no_verdict() -> None:
    """A failed query has no rerank verdict; the count must not vanish."""
    applied = RerankOutcome("cross_encoder", applied=True)

    summary = _harness_reporting((applied, applied, None))

    assert summary.startswith("cross_encoder")
    assert "+1 query(s) with no verdict" in summary


# ---------------------------------------------------------------------------
# It survives the trip back through RetrievedContext
# ---------------------------------------------------------------------------


def test_retrieved_context_defaults_to_no_verdict() -> None:
    """Default None keeps every existing construction site valid."""
    from trelix.core.models import RetrievedContext

    ctx = RetrievedContext(query="q", results=[], context_text="", total_tokens=0)

    assert ctx.rerank is None
