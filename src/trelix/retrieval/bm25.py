"""
BM25 keyword search via SQLite FTS5 — zero extra dependencies.

FTS5's built-in BM25 ranking is good enough for code identifier search.
Returns hydrated SearchResult objects ready for RRF fusion.
"""

from __future__ import annotations

from trelix.core.models import SearchResult
from trelix.store.db import Database


def bm25_search(db: Database, query: str, k: int = 20) -> list[SearchResult]:
    """
    Run FTS5 BM25 search over the symbols table.
    Returns SearchResult list with source="bm25", ready for RRF fusion.
    """
    # Escape FTS5 special chars to avoid query parse errors on raw identifiers
    safe_query = _escape_fts5(query)
    raw = db.bm25_search(safe_query, limit=k)

    results: list[SearchResult] = []
    for rank, (symbol_id, bm25_rank) in enumerate(raw, start=1):
        # FTS5 rank is negative BM25 — closer to 0 = better match
        # Convert to positive score in (0, 1]
        score = 1.0 / (1.0 + abs(bm25_rank))

        sym_file = db.get_symbol_with_file(symbol_id)
        if sym_file is None:
            continue
        symbol, file = sym_file

        chunk = db.get_first_chunk_for_symbol(symbol_id)
        if chunk is None:
            from trelix.core.models import Chunk

            chunk = Chunk(
                symbol_id=symbol_id,
                chunk_text=symbol.body[:2000],
                token_count=0,
            )

        results.append(
            SearchResult(
                chunk=chunk,
                symbol=symbol,
                file=file,
                score=score,
                rank=rank,
                source="bm25",
            )
        )

    return results


_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "it",
        "in",
        "of",
        "to",
        "do",
        "be",
        "as",
        "at",
        "by",
        "or",
        "and",
        "for",
        "on",
        "are",
        "was",
        "has",
        "had",
        "can",
        "not",
        "but",
        "this",
        "that",
        "with",
        "from",
        "what",
        "how",
        "why",
        "who",
        "when",
        "where",
        "which",
        "show",
        "me",
        "tell",
        "give",
        "get",
        "all",
        "about",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "have",
        "been",
        "into",
        "than",
        "then",
        "them",
        "they",
        "their",
        "there",
        "use",
        "used",
        "using",
        "any",
        "our",
        "each",
        "more",
        "also",
        "just",
        "some",
        "such",
        "like",
        "over",
        "after",
        "before",
        "between",
        "out",
        "make",
        "know",
        "find",
        "list",
        "return",
        "returns",
        "code",
        "file",
        "function",
        "method",
        "class",
        "type",
        "value",
        "call",
        "calls",
        "new",
        "my",
        "you",
        "your",
        "so",
        "no",
        "if",
        "am",
        "we",
        "he",
        "she",
        "his",
        "her",
    }
)


def count_meaningful_tokens(query: str) -> int:
    """
    Count non-stop-word tokens of length ≥ 3 in `query`.

    Uses the same tokenization as `_preprocess_query` — splits on
    word-character boundaries, drops stop words, drops tokens < 3 chars.
    This is the signal for short-query detection.
    """
    import re

    raw_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query)
    return sum(1 for t in raw_tokens if t.lower() not in _STOP_WORDS and len(t) >= 3)


def is_short_query(query: str, threshold: int = 5) -> bool:
    """
    Return True when `query` contains ≤ threshold meaningful tokens.

    Short queries (≤5 meaningful tokens after stop-word removal) perform
    near-zero with embedding models across all code retrieval benchmarks
    (CoREB arXiv:2605.04615). Routing them to BM25+grep lexical search
    significantly improves recall for developer keyword-style queries.

    Args:
        query:     Raw user query string
        threshold: Maximum meaningful-token count to classify as "short"
                   (default 5, configurable via TRELIX_RETRIEVAL_SHORT_QUERY_TOKENS)
    """
    return count_meaningful_tokens(query) <= threshold


def _escape_fts5(query: str) -> str:
    """
    Build an FTS5 MATCH expression from a natural-language or identifier query.

    Strategy:
      1. Single identifier (no spaces) → prefix search "name"*
      2. Multi-word → strip English stop words, keep only meaningful code tokens,
         AND-match them. This prevents stop words ("what", "is", "show") from
         zeroing out BM25 results for natural-language queries.
    """
    import re

    # Single identifier (no spaces) — use prefix search
    if " " not in query.strip() and all(c.isalnum() or c in "_." for c in query.strip()):
        safe = query.strip().replace('"', '""')
        return f'"{safe}"*'

    # Multi-word / natural language: extract tokens, drop stop words + short tokens.
    raw_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query)
    tokens = [t for t in raw_tokens if t.lower() not in _STOP_WORDS and len(t) > 2]

    # Fall back to all non-trivial tokens if stop-word filtering removed everything
    if not tokens:
        tokens = [t for t in raw_tokens if len(t) > 2]
    if not tokens:
        return '""'  # matches nothing

    return " ".join(tokens)
