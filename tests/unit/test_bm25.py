"""
Unit tests for trelix.retrieval.bm25.bm25_search.

Uses a real (file-based tmp_path) SQLite DB with FTS5 triggers — no mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    Symbol,
    SymbolKind,
)
from trelix.retrieval.bm25 import _escape_fts5, bm25_search
from trelix.store.db import Database

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file(db: Database, rel_path: str = "src/auth/login.py") -> int:
    f = IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash="deadbeef",
        size_bytes=512,
    )
    return db.upsert_file(f)


def _insert_symbol(
    db: Database,
    file_id: int,
    name: str,
    body: str,
    docstring: str | None = None,
) -> int:
    sym = Symbol(
        file_id=file_id,
        name=name,
        qualified_name=name,
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=10,
        signature=f"def {name}():",
        body=body,
        docstring=docstring,
    )
    sym_id = db.insert_symbol(sym)
    db._conn.commit()
    return sym_id


def _insert_chunk(db: Database, symbol_id: int, text: str) -> int:
    chunk = Chunk(symbol_id=symbol_id, chunk_text=text, token_count=len(text.split()))
    chunk_id = db.insert_chunk(chunk)
    db._conn.commit()
    return chunk_id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    """Fresh SQLite DB per test (real file so FTS5 triggers fire correctly)."""
    return Database(tmp_path / "index.db")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBm25Search:
    def test_returns_empty_for_empty_query(self, db: Database) -> None:
        """An empty query should return an empty list without error."""
        file_id = _make_file(db)
        sym_id = _insert_symbol(db, file_id, "authenticate_user", "def authenticate_user(): pass")
        _insert_chunk(db, sym_id, "authenticate_user")

        results = bm25_search(db, "", k=10)
        assert results == []

    def test_finds_matching_symbol(self, db: Database) -> None:
        """bm25_search finds a symbol whose name matches the query."""
        file_id = _make_file(db)
        sym_id = _insert_symbol(
            db,
            file_id,
            "authenticate_user",
            "def authenticate_user(username, password):\n"
            "    return check_password(username, password)",
            docstring="Authenticate a user by username and password.",
        )
        _insert_chunk(db, sym_id, "def authenticate_user(): ...")

        results = bm25_search(db, "authenticate_user", k=10)
        assert len(results) >= 1
        assert results[0].symbol.name == "authenticate_user"
        assert results[0].source == "bm25"

    def test_relevance_ordering(self, db: Database) -> None:
        """
        Symbol with stronger BM25 signal (query term in name + body + docstring)
        should rank above a symbol with weaker signal (term only in body).
        """
        file_id = _make_file(db)

        # Strong match: "tokenize" appears in name, body, and docstring
        strong_id = _insert_symbol(
            db,
            file_id,
            "tokenize_source",
            "def tokenize_source(text):\n    return tokenize(text)",
            docstring="Tokenize a source string into tokens.",
        )
        _insert_chunk(db, strong_id, "tokenize source tokens")

        # Weak match: "tokenize" appears only once in body
        weak_id = _insert_symbol(
            db,
            file_id,
            "parse_config",
            "def parse_config(path):\n    # tokenize before parsing\n    return {}",
        )
        _insert_chunk(db, weak_id, "parse config")

        results = bm25_search(db, "tokenize", k=10)
        assert len(results) >= 2

        names = [r.symbol.name for r in results]
        assert "tokenize_source" in names
        assert "parse_config" in names
        # Strong match must appear before weak match
        assert names.index("tokenize_source") < names.index("parse_config")

    def test_score_positive_and_in_range(self, db: Database) -> None:
        """SearchResult scores must be > 0 and <= 1."""
        file_id = _make_file(db)
        sym_id = _insert_symbol(
            db,
            file_id,
            "process_request",
            "def process_request(req): return req",
        )
        _insert_chunk(db, sym_id, "process request handler")

        results = bm25_search(db, "process_request", k=5)
        assert results, "Expected at least one result"
        for r in results:
            assert 0 < r.score <= 1.0

    def test_result_has_correct_source(self, db: Database) -> None:
        """source field must always be 'bm25'."""
        file_id = _make_file(db)
        sym_id = _insert_symbol(db, file_id, "handle_login", "def handle_login(): pass")
        _insert_chunk(db, sym_id, "handle login")

        results = bm25_search(db, "handle_login", k=5)
        for r in results:
            assert r.source == "bm25"

    def test_respects_k_limit(self, db: Database) -> None:
        """bm25_search must return at most k results."""
        file_id = _make_file(db)
        for i in range(10):
            sym_id = _insert_symbol(
                db,
                file_id,
                f"validate_field_{i}",
                f"def validate_field_{i}(v): return validate(v)",
            )
            _insert_chunk(db, sym_id, f"validate field {i}")

        results = bm25_search(db, "validate", k=3)
        assert len(results) <= 3

    def test_fallback_chunk_created_when_no_chunk_stored(self, db: Database) -> None:
        """
        When a symbol has no chunk row, bm25_search creates a synthetic Chunk
        from the symbol body so the SearchResult is still complete.
        """
        file_id = _make_file(db)
        body = "def orphan_function():\n    pass"
        # Insert symbol but NO chunk
        _insert_symbol(db, file_id, "orphan_function", body)

        results = bm25_search(db, "orphan_function", k=5)
        assert len(results) >= 1
        r = results[0]
        assert r.chunk is not None
        # Synthetic chunk is sliced from body
        assert body[:2000] in r.chunk.chunk_text or r.chunk.chunk_text in body[:2000]


# ---------------------------------------------------------------------------
# _escape_fts5 unit tests
# ---------------------------------------------------------------------------


class TestEscapeFts5:
    def test_single_identifier_becomes_prefix_search(self) -> None:
        result = _escape_fts5("authenticate_user")
        assert result == '"authenticate_user"*'

    def test_multi_word_strips_stop_words(self) -> None:
        result = _escape_fts5("what is the authenticate function")
        # Stop words: what, is, the, function → only "authenticate" should remain.
        # Check whole-word presence via token split rather than substring (avoids
        # "the" falsely matching inside "authenticate").
        tokens = result.split()
        assert "authenticate" in tokens
        assert "what" not in tokens
        assert "is" not in tokens
        # "the" would be a standalone token if not stripped; must not appear alone
        assert "the" not in tokens

    def test_empty_query_returns_empty_matcher(self) -> None:
        result = _escape_fts5("")
        # Empty/whitespace → single identifier branch fires → '""*' (no-match prefix)
        # OR multi-word branch returns '""'. Either is acceptable as long as it
        # produces no real FTS5 matches.
        assert result in ('""', '""*')


# ---------------------------------------------------------------------------
# is_short_query unit tests
# ---------------------------------------------------------------------------


class TestIsShortQuery:
    def test_single_word_is_short(self) -> None:
        from trelix.retrieval.bm25 import is_short_query

        assert is_short_query("login") is True

    def test_five_meaningful_words_at_threshold(self) -> None:
        from trelix.retrieval.bm25 import is_short_query

        # exactly 5 meaningful tokens → short at default threshold=5
        assert is_short_query("JWT token validation auth middleware") is True

    def test_six_meaningful_words_not_short(self) -> None:
        from trelix.retrieval.bm25 import is_short_query

        assert is_short_query("JWT token validation auth middleware handler") is False

    def test_stop_words_not_counted(self) -> None:
        from trelix.retrieval.bm25 import is_short_query

        # "how does the auth work" — only "auth" and "work" are meaningful (len>2, not stop)
        assert is_short_query("how does the auth work") is True

    def test_custom_threshold(self) -> None:
        from trelix.retrieval.bm25 import is_short_query

        assert is_short_query("auth token user session", threshold=3) is False
        assert is_short_query("auth token", threshold=3) is True

    def test_count_meaningful_tokens(self) -> None:
        from trelix.retrieval.bm25 import count_meaningful_tokens

        assert count_meaningful_tokens("JWT auth middleware") == 3
        assert count_meaningful_tokens("how does it work") == 1  # only "work" passes

    def test_empty_query_is_short(self) -> None:
        from trelix.retrieval.bm25 import is_short_query

        assert is_short_query("") is True


class TestBM25ReadPoolOptIn:
    """Opt-in read-only connection pool for parallel bm25_search() calls
    (v2.6.x scale backlog, Plan C Task C-1). Default (pool disabled) must
    be byte-for-byte identical to pre-existing behavior."""

    def test_bm25_search_unaffected_when_pool_not_enabled(self, tmp_path):
        """Default behavior (no enable_bm25_read_pool call) must produce
        identical results to before this feature existed."""
        db = Database(tmp_path / "test.db")
        file_id = _make_file(db, rel_path="foo.py")
        _insert_symbol(db, file_id, "authenticate_user", "def authenticate_user(): ...")

        results = bm25_search(db, "authenticate", k=10)
        assert len(results) == 1
        assert db._bm25_read_pool is None

    def test_bm25_search_works_identically_with_pool_enabled(self, tmp_path):
        db_path = tmp_path / "test.db"
        db = Database(db_path)
        file_id = _make_file(db, rel_path="foo.py")
        _insert_symbol(db, file_id, "authenticate_user", "def authenticate_user(): ...")
        db.enable_bm25_read_pool(pool_size=2)

        results = bm25_search(db, "authenticate", k=10)
        assert len(results) == 1

        db._bm25_read_pool.close_all()

    def test_concurrent_bm25_search_with_pool_enabled(self, tmp_path):
        """The whole point of this feature: N threads querying bm25_search
        concurrently must all succeed without 'database is locked' errors,
        when the read pool is enabled."""
        import threading

        db_path = tmp_path / "test.db"
        db = Database(db_path)
        file_id = _make_file(db, rel_path="foo.py")
        for i in range(20):
            _insert_symbol(db, file_id, f"fn_{i}", f"def fn_{i}(): return {i}")
        db.enable_bm25_read_pool(pool_size=4)

        errors = []

        def worker():
            try:
                bm25_search(db, "fn", k=10)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent bm25_search under the read pool raised: {errors}"
        db._bm25_read_pool.close_all()
