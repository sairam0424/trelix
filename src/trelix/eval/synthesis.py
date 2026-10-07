"""
Synthesis quality evaluation harness for trelix GraphRAG output.

Implements a GroUSE-inspired (arXiv:2409.06595, COLING 2025) failure-mode
checker for code-specific RAG synthesis. Covers 7 generator failure modes
with code-specific extensions:

NL failure modes (from GroUSE):
  1. Hallucination — answer mentions symbols not in retrieved context
  2. Partial answer — expected fragments missing from answer
  3. Faithful answer — answer grounded in retrieved context
  4. Irrelevant context — answer ignores retrieved symbols entirely
  5. Insufficient context — answer says "I don't know" when context exists
  6. Correct answer — all fragments present, no hallucinations
  7. Answer with caveats — correct but hedged

Code-specific extensions (trelix):
  8. Symbol hallucination — function/class names not in codebase index
  9. Stale line reference — a cited chunk whose line range no longer fits the file on disk (counted as line_out_of_range in citation_statuses; lines are counted as the extractors count them, count("\\n") + 1)
"""  # noqa: E501

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trelix.eval.synthesis_records import (
    SynthesisRecord,
    aggregate_synthesis_metrics,
    failed_record,
    scored_record,
)
from trelix.retrieval.synthesizer import Synthesizer

if TYPE_CHECKING:
    from trelix.core.config import IndexConfig

logger = logging.getLogger("trelix.eval.synthesis")


@dataclass
class SynthesisResult:
    """Result of evaluating one synthesized answer against a golden QA entry."""

    query: str
    answer: str
    retrieved_symbols: list[str]
    expected_symbols: list[str]
    expected_fragments: list[str]
    hallucinated_symbols: list[str]
    missing_fragments: list[str]
    scores: dict[str, float] = field(default_factory=dict)


def score_hallucination(
    answer: str,
    retrieved_symbols: list[str],
    expected_symbols: list[str],
) -> float:
    """
    Measure symbol hallucination in the synthesized answer.

    A symbol is hallucinated when it appears in ``expected_symbols`` but NOT
    in ``retrieved_symbols`` — the answer mentions it without retrieval support.

    Returns:
        0.0 — no hallucinations (all expected symbols were retrieved)
        1.0 — all expected symbols are hallucinated
        0.5 — half of expected symbols are hallucinated
    """
    if not expected_symbols:
        return 0.0

    retrieved_lower = {s.lower() for s in retrieved_symbols}
    answer_lower = answer.lower()

    hallucinated = [
        sym
        for sym in expected_symbols
        if sym.lower() in answer_lower and sym.lower() not in retrieved_lower
    ]
    return len(hallucinated) / len(expected_symbols)


def score_completeness(
    answer: str,
    expected_fragments: list[str],
) -> float:
    """
    Measure answer completeness — fraction of expected fragments present.

    Fragments are case-insensitive substring matches. An empty fragment
    list means "no completeness requirement" and returns 1.0.

    Returns:
        1.0 — all expected fragments found in answer
        0.0 — no expected fragments found
        0.5 — half of expected fragments found
    """
    if not expected_fragments:
        return 1.0

    answer_lower = answer.lower()
    found = sum(1 for frag in expected_fragments if frag.lower() in answer_lower)
    return found / len(expected_fragments)


def score_faithfulness(
    answer: str,
    retrieved_context: str,
) -> float:
    """
    Estimate how faithfully the answer is grounded in retrieved context.

    Heuristic: fraction of non-trivial answer tokens (len>=4) that appear
    in the retrieved context. This is a lexical approximation — not semantic.
    For semantic faithfulness, use an LLM judge with GroUSE criteria.

    Returns:
        0.0 — answer shares no vocabulary with retrieved context
        1.0 — all significant answer tokens appear in retrieved context
    """
    if not answer.strip():
        return 0.0
    if not retrieved_context.strip():
        return 0.0

    context_lower = retrieved_context.lower()
    answer_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{3,}", answer)
    if not answer_tokens:
        return 0.0

    grounded = sum(1 for t in answer_tokens if t.lower() in context_lower)
    return grounded / len(answer_tokens)


def evaluate_synthesis(
    query: str,
    answer: str,
    retrieved_context: str,
    retrieved_symbols: list[str],
    expected_symbols: list[str],
    expected_fragments: list[str],
) -> SynthesisResult:
    """
    Evaluate a synthesized answer against golden expectations.

    Args:
        query:               The original user query
        answer:              The synthesized answer from trelix
        retrieved_context:   The full retrieved context text passed to synthesis
        retrieved_symbols:   Qualified names of all retrieved symbols
        expected_symbols:    Symbol names that should appear (from golden file)
        expected_fragments:  Text fragments that must appear (from golden file)

    Returns:
        SynthesisResult with all scores populated
    """
    hallucination_score = score_hallucination(answer, retrieved_symbols, expected_symbols)
    completeness_score = score_completeness(answer, expected_fragments)
    faithfulness_score = score_faithfulness(answer, retrieved_context)

    hallucinated = [
        sym
        for sym in expected_symbols
        if sym.lower() in answer.lower()
        and sym.lower() not in {s.lower() for s in retrieved_symbols}
    ]
    missing = [frag for frag in expected_fragments if frag.lower() not in answer.lower()]

    return SynthesisResult(
        query=query,
        answer=answer,
        retrieved_symbols=retrieved_symbols,
        expected_symbols=expected_symbols,
        expected_fragments=expected_fragments,
        hallucinated_symbols=hallucinated,
        missing_fragments=missing,
        scores={
            "hallucination": hallucination_score,
            "completeness": completeness_score,
            "faithfulness": faithfulness_score,
            "overall": (
                (1 - hallucination_score) * 0.4
                + completeness_score * 0.4
                + faithfulness_score * 0.2
            ),
        },
    )


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_text_list(value: object) -> bool:
    return isinstance(value, list) and all(_is_text(item) for item in value)


# The synthesis golden's v2 fields: (name, accepts, what a refusal says). Only a field that is
# present is checked, and `null` counts as present, as `trelix.eval.golden.validate_entry` does.
_V2_FIELDS: tuple[tuple[str, Callable[[object], bool], str], ...] = (
    ("answerable", lambda value: isinstance(value, bool), "must be true or false"),
    ("gold_answer", _is_text, "must be a non-empty string"),
    ("expected_citations", _is_text_list, "must be a list of non-empty strings"),
)


def validate_synthesis_entry(item: Mapping[str, object]) -> list[str]:
    """Describe what is wrong with the v2 fields `item` carries; `[]` when they are fine.

    `answerable` must be a JSON boolean, `gold_answer` a non-empty string and
    `expected_citations` a list of non-empty strings (an empty list is fine). The fields the
    harness has always read (`query`, `expected_answer_fragments`, `expected_symbols`) are not
    looked at here: the loader stays as lenient about them as it always was.
    """
    return [
        f'"{name}" {message}'
        for name, accepts, message in _V2_FIELDS
        if name in item and not accepts(item[name])
    ]


def _load_entries(path: Path) -> list[dict[str, Any]]:
    """Every JSON object in the file, in order; raise once with every v2 problem found.

    Today's leniency is kept: a blank line or one that is not JSON is skipped silently. A line
    that is JSON but not an object is refused (it used to reach `entry.get` and crash the run),
    and so is an object whose v2 field has the wrong type, each with its 1-based line number
    over the FILE's lines.
    """
    entries: list[dict[str, Any]] = []
    problems: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                kind = type(item).__name__
                problems.append(f"line {line_no}: expected a JSON object, got {kind}")
                continue
            problems.extend(f"line {line_no}: {p}" for p in validate_synthesis_entry(item))
            entries.append(item)
    if problems:
        count = len(problems)
        raise ValueError(
            f"{path}: {count} unusable golden entr{'y' if count == 1 else 'ies'}:\n  "
            + "\n  ".join(problems)
        )
    return entries


class SynthesisEvalHarness:
    """
    Run a synthesis quality evaluation against a golden QA file.

    Golden file format (JSONL, superset of EvalHarness format):
        {
          "query": "how does JWT validation work?",
          "relevant_files": ["src/auth/middleware.py"],
          "expected_answer_fragments": ["decode", "secret", "bearer"],
          "expected_symbols": ["AuthMiddleware.verify", "jwt.decode"],
          "answerable": true,
          "gold_answer": "...",
          "expected_citations": ["src/auth/middleware.py"]
        }

    Fields ``expected_answer_fragments`` and ``expected_symbols`` are optional —
    queries without them contribute only to n_queries count with score 1.0.
    ``answerable`` (absent means true) says whether the repository answers the query; an
    unanswerable query is counted and left out of the four means. ``gold_answer`` is
    validated and stored, never scored. ``expected_citations`` (repo-relative paths) is
    validated and stored; the next change in this series counts it.
    """

    def __init__(self, config: IndexConfig) -> None:
        # Annotated concretely rather than as `Any`: the `Any` that used to be here is
        # what stopped mypy --strict from noticing that this IndexConfig was being handed
        # to Synthesizer, which takes an EmbedderConfig.
        self._config = config
        from trelix.retrieval.retriever import Retriever

        self._retriever = Retriever(config)

    def run(self, golden_path: str) -> dict[str, float]:
        """
        Evaluate synthesis quality across all queries in the golden file.

        Returns `aggregate_synthesis_metrics` over `run_detailed`'s records:
            hallucination_rate: mean hallucination score over answerable queries (lower = better)
            completeness:       mean completeness score over answerable queries (higher = better)
            faithfulness:       mean faithfulness score over answerable queries (higher = better)
            overall:            mean overall score over answerable queries
            n_queries:          number of queries evaluated
            unscoreable:        queries that raised; their four scores are placeholders
            n_unanswerable:     queries marked unanswerable that did not raise
        """
        return aggregate_synthesis_metrics(self.run_detailed(golden_path))

    def run_detailed(self, golden_path: str) -> list[SynthesisRecord]:
        """One record per golden line, in file order; `[]` for a file with no entries."""
        path = Path(golden_path)
        if not path.exists():
            # Refuse rather than report 0.0 — a missing file and a genuinely bad
            # synthesis result must not render as the same table. Mirrors
            # EvalHarness.run()'s identical FileNotFoundError contract in
            # harness.py, which the CLI's eval-synthesis command already
            # catches with an actionable message.
            raise FileNotFoundError(f"Golden file not found: {golden_path}")

        entries = _load_entries(path)
        if not entries:
            return []

        # Built once, outside the loop. Constructing it per entry re-initialised an LLM
        # client for every query in the golden file.
        #
        # The argument shape matters and was wrong: Synthesizer takes an EmbedderConfig
        # first, and with llm_config=None it falls into a shim that reads
        # `config.provider`. Passing the IndexConfig raised AttributeError on every call,
        # which the swallow below turned into an empty answer — so every query scored
        # against `""` and `overall` was a constant. This mirrors the construction
        # `trelix ask` uses.
        synthesizer = Synthesizer(
            self._config.embedder,
            retrieval_config=self._config.retrieval,
            llm_config=self._config.llm,
        )
        return [
            self._record(entry, position, synthesizer)
            for position, entry in enumerate(entries, start=1)
        ]

    def _record(
        self, entry: dict[str, Any], position: int, synthesizer: Synthesizer
    ) -> SynthesisRecord:
        """Retrieve, synthesize and score one line; a query that raises gives a failed record."""
        query = entry.get("query", "")
        expected_fragments = entry.get("expected_answer_fragments", [])
        expected_symbols = entry.get("expected_symbols", [])
        try:
            context = self._retriever.retrieve(query)
            retrieved_symbols = [
                r.symbol.qualified_name
                for r in context.results
                if hasattr(r, "symbol") and r.symbol
            ]
            try:
                answer = synthesizer.synthesize(context)
            except Exception as exc:
                # Still non-fatal — one flaky LLM call should not abort the run — but no longer
                # silent. An empty answer scores completeness and faithfulness at 0.0, which is
                # indistinguishable from a genuinely bad answer, so the reader has to be told.
                logger.warning(
                    "Synthesis failed for query %r, scoring an empty answer: %s", query[:80], exc
                )
                answer = ""
            result = evaluate_synthesis(
                query=query,
                answer=answer,
                retrieved_context=getattr(context, "context_text", ""),
                retrieved_symbols=retrieved_symbols,
                expected_symbols=expected_symbols,
                expected_fragments=expected_fragments,
            )
            return scored_record(entry, position, result.scores)
        except Exception as exc:
            # Reached when retrieval or scoring fails, not when the MODEL does — and the
            # record's scores are placeholders, not measurements: hallucination=1.0 would make
            # a broken index read as "the model hallucinated everything". They are kept so a
            # partial run still aggregates like a whole one, but the cause is logged and the
            # record carries `error`, so a reader can tell how much was measured at all.
            logger.warning(
                "Could not score query %r (%s) — recording it as unscoreable; the "
                "hallucination/completeness/faithfulness figures for it are "
                "placeholders, not measurements",
                query[:80],
                exc,
            )
            return failed_record(entry, position, exc)
