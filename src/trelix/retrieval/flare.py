"""
FLARE-style confidence-gated re-retrieval loop.

Simplified from: FLARE — Forward-Looking Active REtrieval Augmented Generation
(Jiang et al., EMNLP 2023, arXiv:2305.06983).

Full FLARE monitors token-level log-probabilities. This implementation uses
a simpler but effective heuristic: detect uncertainty phrases in the generated
answer and re-retrieve once with an enriched query.

Usage:
    loop = FLARELoop(retriever, synthesizer, config)
    answer = loop.run("how does the authentication system work?")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trelix.core.config import IndexConfig
    from trelix.retrieval.retriever import Retriever
    from trelix.retrieval.synthesizer import Synthesizer

logger = logging.getLogger("trelix.retrieval.flare")

# Phrases that signal the model lacks sufficient context — trigger a re-retrieval.
_DEFAULT_UNCERTAINTY_PHRASES: list[str] = [
    "i don't know",
    "i do not know",
    "cannot find",
    "no information",
    "not found in",
    "unable to locate",
    "no relevant code",
    "insufficient context",
    "not enough information",
    "couldn't find",
    "could not find",
]


def _contains_uncertainty(
    text: str,
    phrases: list[str] | None = None,
) -> bool:
    """Return True if text contains any uncertainty marker (case-insensitive)."""
    check = phrases or _DEFAULT_UNCERTAINTY_PHRASES
    lower = text.lower()
    return any(phrase in lower for phrase in check)


class FLARELoop:
    """
    Wraps a Retriever + Synthesizer with confidence-gated re-retrieval.

    When the initial synthesis contains uncertainty phrases, re-retrieves
    with a more specific query derived from the original + uncertainty context,
    then re-synthesizes. Runs at most ``flare_max_retries`` additional
    re-retrieval rounds after the initial synthesis.
    """

    def __init__(
        self,
        retriever: Retriever,
        synthesizer: Synthesizer,
        config: IndexConfig,
    ) -> None:
        self._retriever = retriever
        self._synthesizer = synthesizer
        self._config = config

    def run(self, query: str) -> str:
        """
        Execute the FLARE loop. Returns the final synthesized answer.

        If flare_enabled=False, behaves identically to a single retrieve+synthesize.
        """
        cfg = self._config.retrieval
        ctx = self._retriever.retrieve(query)
        answer = self._synthesizer.synthesize(ctx, self._config.embedder)

        if not cfg.flare_enabled:
            return answer

        # ``flare_max_retries`` is the total synthesis call budget (initial + retries).
        # The minimum budget is 2 (initial + 1 retry) because enabling FLARE with no
        # retry is meaningless. Values 1 and 2 both allow exactly 1 retry; 3 allows 2.
        max_retries = max(1, cfg.flare_max_retries - 1)

        iteration = 0
        while _contains_uncertainty(answer) and iteration < max_retries:
            logger.info(
                "FLARE re-retrieval round %d/%d for query: %r",
                iteration + 1,
                max_retries,
                query[:80],
            )
            # Enrich the query with context about what was missing
            enriched_query = f"{query} (focus on implementation details and concrete code)"
            ctx = self._retriever.retrieve(enriched_query)
            answer = self._synthesizer.synthesize(ctx, self._config.embedder)
            iteration += 1

        return answer
