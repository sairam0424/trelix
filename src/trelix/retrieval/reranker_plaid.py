"""
PLAID late-interaction reranker via RAGatouille (ColBERT).

PLAID (Progressive Late Interaction via Approximate Document Hierarchies)
reduces ColBERTv2 search latency 7-45x vs naive late interaction with no
quality degradation (EMNLP 2022, arXiv 2205.09707, confirmed 3-0).

RAGatouille provides a production-ready PLAID implementation:
    pip install ragatouille>=0.0.8

Usage: set TRELIX_RETRIEVAL_RERANK_PROVIDER=plaid in .env

Fallback: if ragatouille is not installed or loading fails, falls back to
returning results in the original order (safe degradation).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trelix.core.config import RetrievalConfig
    from trelix.core.models import SearchResult

logger = logging.getLogger("trelix.retrieval.reranker_plaid")

# Top-level name for patch target in tests.
# When ragatouille is not installed this will be None; the class guards against
# that in _get_model().
_RAGPretrainedModel: Any | None
try:
    from ragatouille import RAGPretrainedModel as _RAGPretrainedModel_cls

    _RAGPretrainedModel = _RAGPretrainedModel_cls
except ImportError:
    _RAGPretrainedModel = None

RAGPretrainedModel = _RAGPretrainedModel


class PlaidReranker:
    """
    PLAID/ColBERT reranker backed by RAGatouille.

    Lazy-loads the model on first use to avoid slow startup when
    PLAID is configured but not every query needs reranking.
    """

    def __init__(self, config: RetrievalConfig) -> None:
        self._model_name = config.plaid_model
        self._top_n = config.rerank_top_n
        self._model = None  # lazy-loaded

    def _get_model(self) -> Any:
        if self._model is None:
            if RAGPretrainedModel is None:
                logger.warning(
                    "ragatouille is not installed — PLAID reranking disabled. "
                    "Install with: pip install 'trelix[plaid]'",
                )
                return None
            try:
                self._model = RAGPretrainedModel.from_pretrained(self._model_name)
            except Exception as exc:
                logger.warning(
                    "PLAID model load failed (%s) — reranking disabled. "
                    "Install with: pip install 'trelix[plaid]'",
                    exc,
                )
        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        """
        Rerank results using PLAID late-interaction scoring.

        Falls back to original order if PLAID is unavailable.
        """
        if not results:
            return results

        model = self._get_model()
        if model is None:
            return results  # graceful degradation

        n = top_n or self._top_n or len(results)
        texts = [r.chunk.chunk_text for r in results]

        try:
            reranked = model.rerank(
                query=query,
                documents=texts,
                k=min(n, len(texts)),
            )
            # reranked: list of {"content": str, "score": float, "result_index": int}
            scored: dict[int, float] = {
                item["result_index"]: item["score"] for item in reranked if "result_index" in item
            }
            updated = [replace(r, score=scored.get(i, r.score)) for i, r in enumerate(results)]
            return sorted(updated, key=lambda r: r.score, reverse=True)[:n]
        except Exception as exc:
            logger.warning("PLAID reranking failed: %s", exc)
            return results
