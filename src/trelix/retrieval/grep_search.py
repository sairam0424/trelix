"""
Grep search — the third retrieval leg, inspired by grep.app.

When a user types an exact identifier (function name, class name, variable),
exact-match search is faster and more precise than vector or BM25 search.

Two modes:
  1. Exact name lookup  — hits the DB index (O(log n), instant)
  2. Regex/substring    — scans symbol bodies in memory

Results are hydrated into SearchResult objects and fed into RRF fusion.
"""

from __future__ import annotations

import re

from trelix.core.models import Chunk, SearchResult
from trelix.store.db import Database


def grep_search(
    db: Database,
    query: str,
    k: int = 10,
    path_filter: str | None = None,
    use_regex: bool = False,
) -> list[SearchResult]:
    """
    Exact or regex search. Returns SearchResult list with source="grep".

    Score: 1.0 for exact name match, 0.8 for body/docstring match.
    """
    results: list[SearchResult] = []
    seen: set[int] = set()

    # --- 1. Exact symbol name match (fastest, hits DB index) ---
    for symbol_id, score in _name_search(db, query, path_filter, k):
        if symbol_id in seen:
            continue
        seen.add(symbol_id)
        r = _hydrate(db, symbol_id, score, len(results) + 1, "grep")
        if r:
            results.append(r)

    # --- 2. Body/regex match (if we still have budget) ---
    remaining = k - len(results)
    if remaining > 0:
        for symbol_id, score in _body_search(db, query, path_filter, use_regex, remaining):
            if symbol_id in seen:
                continue
            seen.add(symbol_id)
            r = _hydrate(db, symbol_id, score, len(results) + 1, "grep")
            if r:
                results.append(r)

    return results[:k]


# ------------------------------------------------------------------
# Search helpers
# ------------------------------------------------------------------


def _name_search(
    db: Database,
    name: str,
    path_filter: str | None,
    limit: int,
) -> list[tuple[int, float]]:
    """Exact + prefix match on symbol.name — uses DB index."""
    conn = db._conn
    if path_filter:
        rows = conn.execute(
            """
            SELECT s.id FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE (s.name = ? OR s.qualified_name = ? OR s.name LIKE ?)
              AND f.rel_path LIKE ?
            LIMIT ?
            """,
            (name, name, f"{name}%", f"{path_filter}%", limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id FROM symbols
            WHERE name = ? OR qualified_name = ? OR name LIKE ?
            LIMIT ?
            """,
            (name, name, f"{name}%", limit),
        ).fetchall()

    return [(r[0], 1.0) for r in rows]


def _body_search(
    db: Database,
    pattern: str,
    path_filter: str | None,
    use_regex: bool,
    limit: int,
) -> list[tuple[int, float]]:
    """Regex or substring search over symbol bodies.

    Strategy (bounded — never fetches the full table unbounded):
    1. Try FTS5 first: fast index lookup, capped at 500 rows.
    2. If FTS5 returns nothing (e.g. regex/partial token not in index),
       fall back to a LIMIT-2000 scan so memory exposure is bounded.
    """
    conn = db._conn
    _FTS_LIMIT = 500
    _SCAN_LIMIT = 2000

    # Build the match function once (used against whichever candidate set wins)
    if use_regex:
        try:
            compiled = re.compile(pattern, re.MULTILINE)
            match_fn = lambda body: bool(compiled.search(body))  # noqa: E731
        except re.error:
            match_fn = lambda body: pattern in (body or "")  # noqa: E731
    else:
        match_fn = lambda body: pattern in (body or "")  # noqa: E731

    # --- 1. FTS5 path (bounded) ---
    # FTS5 MATCH uses its own tokenizer so it won't match all regex patterns,
    # but it's a great pre-filter for plain-text queries.
    try:
        if path_filter:
            rows = conn.execute(
                """
                SELECT s.id, s.body FROM symbols s
                JOIN symbols_fts f ON s.id = f.rowid
                JOIN files fi ON s.file_id = fi.id
                WHERE symbols_fts MATCH ?
                  AND fi.rel_path LIKE ?
                LIMIT ?
                """,
                (pattern, f"{path_filter}%", _FTS_LIMIT),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT s.id, s.body FROM symbols s
                JOIN symbols_fts f ON s.id = f.rowid
                WHERE symbols_fts MATCH ?
                LIMIT ?
                """,
                (pattern, _FTS_LIMIT),
            ).fetchall()
    except Exception:
        # FTS5 MATCH will raise if the query string is syntactically invalid
        # (e.g. bare regex operators). Treat as no FTS5 results.
        rows = []

    # --- 2. Bounded fallback scan if FTS5 found nothing ---
    if not rows:
        if path_filter:
            rows = conn.execute(
                """
                SELECT s.id, s.body FROM symbols s
                JOIN files f ON s.file_id = f.id
                WHERE f.rel_path LIKE ?
                LIMIT ?
                """,
                (f"{path_filter}%", _SCAN_LIMIT),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, body FROM symbols LIMIT ?",
                (_SCAN_LIMIT,),
            ).fetchall()

    matched: list[tuple[int, float]] = []
    for symbol_id, body in rows:
        if match_fn(body):  # type: ignore[no-untyped-call]
            matched.append((symbol_id, 0.8))
            if len(matched) >= limit:
                break

    return matched


# ------------------------------------------------------------------
# Hydration
# ------------------------------------------------------------------


def _hydrate(
    db: Database,
    symbol_id: int,
    score: float,
    rank: int,
    source: str,
) -> SearchResult | None:
    sym_file = db.get_symbol_with_file(symbol_id)
    if sym_file is None:
        return None
    symbol, file = sym_file

    chunk = db.get_first_chunk_for_symbol(symbol_id)
    if chunk is None:
        chunk = Chunk(
            symbol_id=symbol_id,
            chunk_text=symbol.body[:2000],
            token_count=0,
        )

    return SearchResult(
        chunk=chunk,
        symbol=symbol,
        file=file,
        score=score,
        rank=rank,
        source=source,
    )
