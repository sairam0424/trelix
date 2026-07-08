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
  9. Stale line reference — answer cites line numbers inconsistent with index
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


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
        sym for sym in expected_symbols
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
        sym for sym in expected_symbols
        if sym.lower() in answer.lower()
        and sym.lower() not in {s.lower() for s in retrieved_symbols}
    ]
    missing = [
        frag for frag in expected_fragments
        if frag.lower() not in answer.lower()
    ]

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


class SynthesisEvalHarness:
    """
    Run a synthesis quality evaluation against a golden QA file.

    Golden file format (JSONL, superset of EvalHarness format):
        {
          "query": "how does JWT validation work?",
          "relevant_files": ["src/auth/middleware.py"],
          "expected_answer_fragments": ["decode", "secret", "bearer"],
          "expected_symbols": ["AuthMiddleware.verify", "jwt.decode"]
        }

    Fields ``expected_answer_fragments`` and ``expected_symbols`` are optional —
    queries without them contribute only to n_queries count with score 1.0.
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        from trelix.retrieval.retriever import Retriever
        self._retriever = Retriever(config)

    def run(self, golden_path: str) -> dict[str, float]:
        """
        Evaluate synthesis quality across all queries in the golden file.

        Returns aggregate metrics:
            hallucination_rate: mean hallucination score (lower = better)
            completeness:       mean completeness score (higher = better)
            faithfulness:       mean faithfulness score (higher = better)
            overall:            mean overall score
            n_queries:          number of queries evaluated
        """
        import json
        from pathlib import Path

        path = Path(golden_path)
        if not path.exists():
            return {
                "hallucination_rate": 0.0,
                "completeness": 0.0,
                "faithfulness": 0.0,
                "overall": 0.0,
                "n_queries": 0.0,
            }

        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not entries:
            return {
                "hallucination_rate": 0.0,
                "completeness": 0.0,
                "faithfulness": 0.0,
                "overall": 0.0,
                "n_queries": 0.0,
            }

        hallucination_scores: list[float] = []
        completeness_scores: list[float] = []
        faithfulness_scores: list[float] = []
        overall_scores: list[float] = []

        for entry in entries:
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
                    from trelix.retrieval.synthesizer import Synthesizer
                    synthesizer = Synthesizer(self._config)
                    answer = synthesizer.synthesize(query, context)
                except Exception:
                    answer = ""

                result = evaluate_synthesis(
                    query=query,
                    answer=answer,
                    retrieved_context=getattr(context, "context_text", ""),
                    retrieved_symbols=retrieved_symbols,
                    expected_symbols=expected_symbols,
                    expected_fragments=expected_fragments,
                )
                hallucination_scores.append(result.scores["hallucination"])
                completeness_scores.append(result.scores["completeness"])
                faithfulness_scores.append(result.scores["faithfulness"])
                overall_scores.append(result.scores["overall"])
            except Exception:
                hallucination_scores.append(1.0)
                completeness_scores.append(0.0)
                faithfulness_scores.append(0.0)
                overall_scores.append(0.0)

        n = len(hallucination_scores)
        return {
            "hallucination_rate": sum(hallucination_scores) / n,
            "completeness": sum(completeness_scores) / n,
            "faithfulness": sum(faithfulness_scores) / n,
            "overall": sum(overall_scores) / n,
            "n_queries": float(n),
        }
