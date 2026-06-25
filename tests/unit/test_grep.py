"""
Unit tests for trelix.retrieval.grep_search.grep_search.

Uses a real (file-based tmp_path) SQLite DB — no mocks.
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
from trelix.retrieval.grep_search import grep_search
from trelix.store.db import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file(db: Database, rel_path: str = "src/auth/login.py") -> int:
    f = IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash="cafebabe",
        size_bytes=256,
    )
    return db.upsert_file(f)


def _insert_symbol(
    db: Database,
    file_id: int,
    name: str,
    body: str,
    qualified_name: str | None = None,
) -> int:
    sym = Symbol(
        file_id=file_id,
        name=name,
        qualified_name=qualified_name or name,
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=10,
        signature=f"def {name}():",
        body=body,
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
    """Fresh SQLite DB per test."""
    return Database(tmp_path / "index.db")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGrepSearch:
    def test_exact_symbol_name_match(self, db: Database) -> None:
        """Searching the exact function name returns that symbol with score 1.0."""
        file_id = _make_file(db)
        sym_id = _insert_symbol(
            db, file_id, "authenticate_user",
            "def authenticate_user(username, password): pass",
        )
        _insert_chunk(db, sym_id, "authenticate user impl")

        results = grep_search(db, "authenticate_user", k=10)

        assert len(results) >= 1
        names = [r.symbol.name for r in results]
        assert "authenticate_user" in names

        # The exact match should have score 1.0
        exact = next(r for r in results if r.symbol.name == "authenticate_user")
        assert exact.score == 1.0
        assert exact.source == "grep"

    def test_partial_name_match_prefix(self, db: Database) -> None:
        """Prefix of a symbol name is matched via LIKE query."""
        file_id = _make_file(db)
        sym_id = _insert_symbol(
            db, file_id, "authenticate_user",
            "def authenticate_user(username, password): pass",
        )
        _insert_chunk(db, sym_id, "authenticate impl")

        # "authenticate" is a prefix of "authenticate_user"
        results = grep_search(db, "authenticate", k=10)

        assert len(results) >= 1
        names = [r.symbol.name for r in results]
        assert "authenticate_user" in names

    def test_body_substring_match(self, db: Database) -> None:
        """A pattern that appears in the body (but not the name) is found at score 0.8."""
        file_id = _make_file(db)
        sym_id = _insert_symbol(
            db, file_id, "process_request",
            "def process_request(req):\n    result = call_downstream_service(req)\n    return result",
        )
        _insert_chunk(db, sym_id, "process request")

        results = grep_search(db, "call_downstream_service", k=10)

        assert len(results) >= 1
        body_match = next(
            (r for r in results if r.symbol.name == "process_request"), None
        )
        assert body_match is not None
        assert body_match.score == 0.8

    def test_no_match_returns_empty(self, db: Database) -> None:
        """Query that matches nothing returns an empty list."""
        file_id = _make_file(db)
        _insert_symbol(db, file_id, "hello_world", "def hello_world(): pass")

        results = grep_search(db, "zzz_nonexistent_xyz", k=10)
        assert results == []

    def test_source_is_grep(self, db: Database) -> None:
        """All results must carry source='grep'."""
        file_id = _make_file(db)
        sym_id = _insert_symbol(
            db, file_id, "compute_hash",
            "def compute_hash(data): return hash(data)",
        )
        _insert_chunk(db, sym_id, "compute hash")

        results = grep_search(db, "compute_hash", k=5)
        assert results
        for r in results:
            assert r.source == "grep"

    def test_respects_k_limit(self, db: Database) -> None:
        """grep_search returns at most k results."""
        file_id = _make_file(db)
        for i in range(15):
            sym_id = _insert_symbol(
                db, file_id, f"validate_item_{i}",
                f"def validate_item_{i}(v): return validate(v)",
            )
            _insert_chunk(db, sym_id, f"validate item {i}")

        results = grep_search(db, "validate_item", k=5)
        assert len(results) <= 5

    def test_deduplication_exact_and_body_match(self, db: Database) -> None:
        """
        A symbol that matches both by name AND body is only returned once.
        """
        file_id = _make_file(db)
        sym_id = _insert_symbol(
            db, file_id, "check_token",
            "def check_token(tok):\n    return check_token_validity(tok)",
        )
        _insert_chunk(db, sym_id, "check token validation")

        results = grep_search(db, "check_token", k=10)
        symbol_ids = [r.symbol.id for r in results]
        # No duplicates
        assert len(symbol_ids) == len(set(symbol_ids))

    def test_regex_mode_matches_pattern(self, db: Database) -> None:
        """use_regex=True applies a compiled regex to symbol bodies."""
        file_id = _make_file(db)
        sym_id = _insert_symbol(
            db, file_id, "send_email",
            "def send_email(to):\n    smtp.sendmail(FROM, to, message)",
        )
        _insert_chunk(db, sym_id, "send email impl")

        # Regex matching "smtp\.\w+" should find "smtp.sendmail"
        results = grep_search(db, r"smtp\.\w+", k=10, use_regex=True)
        names = [r.symbol.name for r in results]
        assert "send_email" in names

    def test_fallback_chunk_when_no_chunk_stored(self, db: Database) -> None:
        """
        When a symbol has no stored chunk, grep_search creates a synthetic Chunk
        so SearchResult is complete.
        """
        file_id = _make_file(db)
        body = "def orphan_grep(): pass"
        # Insert symbol but NO chunk
        _insert_symbol(db, file_id, "orphan_grep", body)

        results = grep_search(db, "orphan_grep", k=5)
        assert len(results) >= 1
        r = results[0]
        assert r.chunk is not None
        # Fallback chunk uses symbol body
        assert r.chunk.chunk_text in body or body[:2000] in r.chunk.chunk_text

    def test_path_filter_excludes_other_files(self, db: Database) -> None:
        """path_filter restricts results to symbols in matching file paths."""
        file_id_auth = _make_file(db, rel_path="src/auth/login.py")
        file_id_api = _make_file(db, rel_path="src/api/views.py")

        sym_auth = _insert_symbol(
            db, file_id_auth, "login_handler",
            "def login_handler(): pass",
        )
        sym_api = _insert_symbol(
            db, file_id_api, "login_handler",
            "def login_handler(): pass",
        )
        _insert_chunk(db, sym_auth, "login auth")
        _insert_chunk(db, sym_api, "login api")

        results = grep_search(db, "login_handler", k=10, path_filter="src/auth")

        file_ids = {r.file.rel_path for r in results}
        assert all("auth" in p for p in file_ids), (
            f"Expected only auth files, got: {file_ids}"
        )
