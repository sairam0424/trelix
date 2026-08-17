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
  - plaid:  ColBERT late interaction via RAGatouille
            requires: pip install trelix[plaid]
  - xtr:    DEGENERATE — reranks nothing. Its single-token approximation makes
            each XTR score equal the input score, so it only re-sorts by the
            fused score and never reads the query. It logs a warning on every
            call; use "plaid" if you want real late interaction. See
            _xtr_rerank() for why (no token-level index exists to fix it with).

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

    DEGENERATE: this provider currently reranks NOTHING. It builds a
    query_token_scores dict with exactly one synthetic token whose retrieved
    scores are the incoming fused scores, so the XTR average over query tokens
    is sum([s])/1 == s for every document. Measured on a 3-result fixture, each
    output score is bit-identical to its input score and the output order is a
    plain descending sort of the input — the query string is never read. Real
    token-level XTR needs a ColBERT-style multi-vector token embedder, which
    trelix does not index, so the honest behaviour is to say so on every call
    (see the log.warning below) rather than let callers believe reranking ran.
    `xtr_candidate_tokens` / TRELIX_RETRIEVAL_XTR_TOKENS exists for that future
    embedder and has no effect today.

    Emits a UserWarning on every call (experimental status).
    Falls back gracefully (warning, no raise) if xtr_score_documents errors.
    """
    from trelix.retrieval.reranker_xtr import warn_experimental

    warn_experimental()
    # Logged at selection time, every call: warn_experimental() only says
    # "unbenchmarked", which understates a provider that is an identity
    # function. Silence here reads as "xtr reranked your results".
    log.warning(
        "XTR reranker is a no-op: the single-token approximation makes every "
        "document's XTR score equal its input score, so results are only "
        "re-sorted by the score they already had and the query text is ignored. "
        "TRELIX_RETRIEVAL_XTR_TOKENS has no effect (token-level indexing is not "
        "implemented). Use rerank_provider=plaid for real late interaction."
    )
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
        idx_to_xtr_score = dict(xtr_pairs)

        # Key by POSITION, not by object: SearchResult is a plain dataclass, so
        # results.index(r) returns the first value-equal element — duplicates all
        # inherited doc_id 0's score, burying the real winner, and the lookup was
        # O(n^2) besides. sorted() is stable, so equal scores keep fused order.
        order = sorted(
            range(len(results)),
            key=lambda idx: idx_to_xtr_score.get(idx, 0.0),
            reverse=True,
        )
        # New SearchResults carrying the XTR score, matching _cross_encoder_rerank:
        # the old code mutated the caller's objects' .rank in place and left .score
        # at the pre-rerank value, which hid whether scoring had happened at all.
        return [
            SearchResult(
                chunk=results[idx].chunk,
                symbol=results[idx].symbol,
                file=results[idx].file,
                score=idx_to_xtr_score.get(idx, 0.0),
                rank=i,
                source=results[idx].source,
            )
            for i, idx in enumerate(order[:top_n], start=1)
        ]
    except Exception as exc:
        log.warning("XTR reranker failed, returning unranked results: %s", exc)
        return results[:top_n]
