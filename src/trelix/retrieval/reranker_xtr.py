"""
XTR (eXpanded Token Retrieval) late-interaction scorer.

Reference: "From Distillation to Hard Negative Sampling: Making Sparse Neural
IR Models More Effective" — NeurIPS 2023, Google DeepMind (arXiv:2304.01982).

XTR eliminates ColBERT's document-token gather stage: scoring uses only tokens
already retrieved during the token-retrieval pass, with scalar imputation for
query tokens that have no retrieved match in a document.

XTR scoring formula (Algorithm 1 in paper):
    score(q, d) = (1/|q|) * sum_{t in q} max(
        max_{(d', s) in R_t, d'=d} s,   # best retrieved score for this doc
        k_impute                          # imputation for unmatched tokens
    )

where R_t is the set of (doc_id, score) pairs retrieved for query token t,
and k_impute = s_{k'} (the k'-th retrieval score, used as imputation floor).

Status: EXPERIMENTAL — not yet benchmarked on code-specific retrieval tasks.
        Enable with TRELIX_RETRIEVAL_RERANK_PROVIDER=xtr.
        Code-domain evaluation pending (CoIR / CoREB benchmark run needed).
"""

from __future__ import annotations

import warnings


def xtr_score_documents(
    query_token_scores: dict[int, list[tuple[int, float]]],
    candidate_doc_ids: list[int],
    k_impute: float,
) -> list[tuple[int, float]]:
    """
    Score candidate documents using the XTR formula.

    Args:
        query_token_scores: For each query token index (int), a list of
            (doc_id, score) pairs — the top-k token neighbors retrieved
            from the vector index for that query token.
        candidate_doc_ids:  All candidate document IDs to score.
        k_impute:           Imputation score for query tokens that have
                            no retrieved match in a document. Typically
                            set to the score of the k'-th retrieved token
                            (the retrieval threshold score).

    Returns:
        List of (doc_id, xtr_score) sorted descending by score.
    """
    if not candidate_doc_ids:
        return []

    n_query_tokens = len(query_token_scores)
    if n_query_tokens == 0:
        return [(doc_id, 0.0) for doc_id in candidate_doc_ids]

    # For each (query_token, doc_id), find the best retrieved score.
    # Structure: best_scores[query_token_idx][doc_id] = max_score
    best_scores: dict[int, dict[int, float]] = {
        qt: {} for qt in query_token_scores
    }
    for qt_idx, retrievals in query_token_scores.items():
        for doc_id, score in retrievals:
            if doc_id in best_scores[qt_idx]:
                best_scores[qt_idx][doc_id] = max(best_scores[qt_idx][doc_id], score)
            else:
                best_scores[qt_idx][doc_id] = score

    # Score each candidate document.
    results: list[tuple[int, float]] = []
    for doc_id in candidate_doc_ids:
        token_contributions: list[float] = []
        for qt_idx in query_token_scores:
            # Use retrieved score if available, else impute.
            doc_score = best_scores[qt_idx].get(doc_id, k_impute)
            token_contributions.append(doc_score)

        # XTR score = average over query tokens.
        xtr_score = sum(token_contributions) / n_query_tokens
        results.append((doc_id, xtr_score))

    return sorted(results, key=lambda x: x[1], reverse=True)


def warn_experimental() -> None:
    """Emit a UserWarning when XTR is first activated."""
    warnings.warn(
        "XTR reranker is experimental in trelix v2.6.0. "
        "It has not been benchmarked on code-specific retrieval tasks (CoIR/CoREB). "
        "Performance on code queries vs PLAID is unverified. "
        "Set TRELIX_RETRIEVAL_RERANK_PROVIDER=plaid for the production-validated option.",
        UserWarning,
        stacklevel=3,
    )
