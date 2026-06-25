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
        case "cross_encoder":
            return _cross_encoder_rerank(query, results, config.rerank_model, top_n)
        case "cohere":
            return _cohere_rerank(
                query, results, top_n,
                api_key=config.cohere_api_key,
                endpoint=config.cohere_endpoint,
                model=config.cohere_rerank_model,
            )
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

        from sentence_transformers import CrossEncoder  # type: ignore[import]
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

    Retries on transient network/SSL errors with exponential backoff.
    Falls back to returning the original top-N results (unmodified) if all
    retries are exhausted so the query pipeline still produces an answer.
    """
    try:
        import requests  # type: ignore[import]
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

    import time

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

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

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
        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as exc:
            last_error = exc
            if attempt < max_retries:
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s …
                log.warning(
                    "Cohere rerank attempt %d/%d failed (%s), retrying in %ds …",
                    attempt, max_retries, type(exc).__name__, wait,
                )
                time.sleep(wait)

    # All retries exhausted — fall back to the original ordering so the
    # query pipeline can still return results (just without reranking).
    log.error(
        "Cohere rerank failed after %d attempts: %s. "
        "Falling back to un-reranked results.",
        max_retries,
        last_error,
    )
    fallback = results[:top_n]
    for i, r in enumerate(fallback, start=1):
        r.rank = i
    return fallback
