"""
Reranker: re-scores candidates using a cross-encoder (more accurate than bi-encoder).

Cross-encoders jointly encode [query, document] pairs and produce a single
relevance score — much more accurate than vector cosine similarity, but slower
(can't pre-compute). Used on the top-K candidates after fusion.

Supported providers:
  - cross_encoder: local sentence-transformers cross-encoder (free, fast)
                   requires: pip install trelix[local]
  - cohere: Cohere Rerank API (best quality, requires API key)
            requires: pip install trelix[rerank]

When neither cohere nor sentence-transformers is installed: logs a warning and
returns results unchanged rather than raising.
"""

from __future__ import annotations

import logging

from trelix.core.config import RetrievalConfig
from trelix.core.models import SearchResult
from trelix.core.retry import with_retry
from trelix.retrieval.reranker_xtr import (
    xtr_score_documents,  # noqa: F401 — imported for mock patching
)

log = logging.getLogger(__name__)


def rerank(
    query: str,
    results: list[SearchResult],
    config: RetrievalConfig,
    top_n: int = 10,
) -> list[SearchResult]:
    """
    Rerank results using the configured reranker. Returns top_n results.

    Falls back gracefully (warning, no raise) when the required library is
    not installed for the configured provider.
    """
    if not results:
        return []

    match config.rerank_provider:
        case "plaid":
            from trelix.retrieval.reranker_plaid import PlaidReranker

            return PlaidReranker(config).rerank(query, results, top_n=top_n)
        case "cross_encoder":
            return _cross_encoder_rerank(query, results, config.rerank_model, top_n)
        case "cohere":
            return _cohere_rerank(
                query,
                results,
                top_n,
                api_key=config.cohere_api_key,
                endpoint=config.cohere_endpoint,
                model=config.cohere_rerank_model,
            )
        case "xtr":
            return _xtr_rerank(query, results, top_n)
        case _:
            return results[:top_n]


def _cross_encoder_rerank(
    query: str,
    results: list[SearchResult],
    model_name: str,
    top_n: int,
) -> list[SearchResult]:
    """Local cross-encoder reranking (no API key needed).

    Requires: pip install trelix[local]
    When sentence-transformers is not installed, logs a warning and returns
    the original top-N results unchanged.
    """
    try:
        import contextlib
        import io
        import os

        from sentence_transformers import CrossEncoder
    except ImportError:
        log.warning(
            "sentence-transformers is not installed; skipping cross-encoder reranking. "
            "Install it with: pip install trelix[local]"
        )
        return results[:top_n]

    # Suppress noisy model-loading output (progress bars, weight load reports)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_TQDM"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    for name in ("sentence_transformers", "transformers", "safetensors", "tqdm"):
        logging.getLogger(name).setLevel(logging.ERROR)

    # Redirect stderr to swallow tqdm & safetensors LOAD REPORT prints
    with contextlib.redirect_stderr(io.StringIO()):
        model = CrossEncoder(model_name, trust_remote_code=False)

    pairs = [(query, r.chunk.chunk_text) for r in results]
    scores = model.predict(pairs)  # returns list of float

    # Build new SearchResult objects so we don't mutate the originals
    reranked_results: list[SearchResult] = []
    for result, score in zip(results, scores):
        reranked_results.append(
            SearchResult(
                chunk=result.chunk,
                symbol=result.symbol,
                file=result.file,
                score=float(score),
                rank=result.rank,
                source=result.source,
            )
        )

    reranked = sorted(reranked_results, key=lambda x: x.score, reverse=True)
    for i, r in enumerate(reranked, start=1):
        r.rank = i

    return reranked[:top_n]


def _cohere_rerank(
    query: str,
    results: list[SearchResult],
    top_n: int,
    api_key: str | None = None,
    endpoint: str | None = None,
    model: str = "Cohere-rerank-v4.0-pro",
    max_retries: int = 3,
) -> list[SearchResult]:
    """Cohere Rerank via HTTP endpoint.

    Requires: pip install trelix[rerank]
    When requests is not installed or the API key is missing, logs a warning
    and returns the original top-N results unchanged.

    Retries via the shared retry contract (trelix.core.retry.with_retry) —
    replaces a prior ad-hoc loop that only caught SSLError/ConnectionError/
    Timeout and therefore never actually retried a 429/5xx response (the
    exact case a rerank API is most likely to return under load), despite
    calling raise_for_status() to raise on one.
    Falls back to returning the original top-N results (unmodified) if all
    retries are exhausted so the query pipeline still produces an answer.
    """
    try:
        import requests
    except ImportError:
        log.warning(
            "requests is not installed; skipping Cohere reranking. "
            "Install it with: pip install trelix[rerank]"
        )
        return results[:top_n]

    if not api_key:
        log.warning(
            "COHERE_API_KEY is not set; skipping Cohere reranking. "
            "Set it with: export COHERE_API_KEY=<your-key>"
        )
        return results[:top_n]

    url = endpoint  # full URL including path (e.g. .../providers/cohere/v2/rerank)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "query": query,
        "documents": [r.chunk.chunk_text for r in results],
        "top_n": top_n,
        "return_documents": False,
    }

    @with_retry(max_attempts=max_retries)
    def _post() -> dict:  # type: ignore[type-arg]
        resp = requests.post(url, json=payload, headers=headers, timeout=30)  # type: ignore[arg-type]
        resp.raise_for_status()
        return dict(resp.json())

    try:
        data = _post()
    except Exception as exc:  # noqa: BLE001 — any exhausted-retry failure falls back
        log.error(
            "Cohere rerank failed after %d attempt(s): %s. Falling back to un-reranked results.",
            max_retries,
            exc,
        )
        fallback = results[:top_n]
        for i, r in enumerate(fallback, start=1):
            r.rank = i
        return fallback

    reranked: list[SearchResult] = []
    for item in data["results"]:
        result = results[item["index"]]
        reranked.append(
            SearchResult(
                chunk=result.chunk,
                symbol=result.symbol,
                file=result.file,
                score=item["relevance_score"],
                rank=len(reranked) + 1,
                source=result.source,
            )
        )
    return reranked


def _xtr_rerank(
    query: str,
    results: list[SearchResult],
    top_n: int,
) -> list[SearchResult]:
    """XTR late-interaction reranking (experimental, arXiv:2304.01982).

    Uses a single-vector approximation: each result is treated as a single
    retrieved token, keyed by its position index. True token-level XTR requires
    a ColBERT-style multi-vector token embedder; this approximation reuses the
    already-retrieved per-result scores.

    Emits a UserWarning on every call (experimental status).
    Falls back gracefully (warning, no raise) if xtr_score_documents errors.
    """
    from trelix.retrieval.reranker_xtr import warn_experimental

    warn_experimental()
    try:
        # Build query_token_scores: treat the single query as one token (index 0),
        # with each result's position as its doc_id and its current score as the
        # retrieved token score.
        query_token_scores = {0: [(idx, r.score) for idx, r in enumerate(results)]}
        candidate_ids = list(range(len(results)))
        k_impute = min(r.score for r in results) if results else 0.0

        xtr_pairs = xtr_score_documents(
            query_token_scores=query_token_scores,
            candidate_doc_ids=candidate_ids,
            k_impute=k_impute,
        )
        idx_to_xtr_score = {idx: score for idx, score in xtr_pairs}
        reranked = sorted(
            results,
            key=lambda r: idx_to_xtr_score.get(results.index(r), 0.0),
            reverse=True,
        )
        for i, r in enumerate(reranked, start=1):
            r.rank = i
        return reranked[:top_n]
    except Exception as exc:
        log.warning("XTR reranker failed, returning unranked results: %s", exc)
        return results[:top_n]
