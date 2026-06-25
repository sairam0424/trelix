"""
Unit tests for trelix.store.db (Database) and trelix.store.vector (VectorStore).

All tests use tmp_path SQLite databases — no external services required.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from trelix.core.models import (
    Chunk,
    ImportEdge,
    IndexedFile,
    Language,
    Symbol,
    SymbolKind,
)
from trelix.store.db import Database
from trelix.store.vector import VectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> Database:
    """Fresh in-memory-equivalent SQLite Database for each test."""
    return Database(tmp_path / "index.db")


@pytest.fixture()
def sample_file() -> IndexedFile:
    return IndexedFile(
        path="/repo/src/auth/login.py",
        rel_path="src/auth/login.py",
        language=Language.PYTHON,
        hash="abc123",
        size_bytes=1024,
    )


@pytest.fixture()
def sample_symbol(db: Database, sample_file: IndexedFile) -> Symbol:
    """Insert a file and return a Symbol ready to be inserted."""
    file_id = db.upsert_file(sample_file)
    return Symbol(
        file_id=file_id,
        name="authenticate_user",
        qualified_name="LoginView.authenticate_user",
        kind=SymbolKind.METHOD,
        line_start=10,
        line_end=30,
        signature="def authenticate_user(self, username: str) -> User",
        body="def authenticate_user(self, username):\n    ...",
        docstring="Authenticate a user by username.",
        decorators=["@login_required"],
        is_public=True,
    )


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

class TestSchemaCreation:
    def test_tables_exist(self, db: Database) -> None:
        """All expected tables should be created on Database init."""
        conn = db._conn
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for expected in ("files", "symbols", "calls", "imports", "chunks"):
            assert expected in tables, f"Table '{expected}' not found"

    def test_fts5_virtual_table_exists(self, db: Database) -> None:
        """FTS5 virtual table symbols_fts must be created."""
        conn = db._conn
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='symbols_fts'"
        ).fetchone()
        assert row is not None, "symbols_fts virtual table not found"

    def test_wal_mode(self, db: Database) -> None:
        """WAL journal mode should be active."""
        row = db._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"

    def test_foreign_keys_on(self, db: Database) -> None:
        """Foreign keys pragma should be enabled."""
        row = db._conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1

    def test_imported_file_id_migration_column(self, db: Database) -> None:
        """imports.imported_file_id migration column must exist after init."""
        cols = {r[1] for r in db._conn.execute("PRAGMA table_info(imports)").fetchall()}
        assert "imported_file_id" in cols


# ---------------------------------------------------------------------------
# upsert_file
# ---------------------------------------------------------------------------

class TestUpsertFile:
    def test_insert_returns_id(self, db: Database, sample_file: IndexedFile) -> None:
        file_id = db.upsert_file(sample_file)
        assert isinstance(file_id, int)
        assert file_id > 0

    def test_upsert_same_path_returns_same_id(
        self, db: Database, sample_file: IndexedFile
    ) -> None:
        id1 = db.upsert_file(sample_file)
        id2 = db.upsert_file(sample_file)
        assert id1 == id2

    def test_upsert_updates_hash(self, db: Database, sample_file: IndexedFile) -> None:
        db.upsert_file(sample_file)
        updated = IndexedFile(
            path=sample_file.path,
            rel_path=sample_file.rel_path,
            language=sample_file.language,
            hash="newHash999",
            size_bytes=sample_file.size_bytes,
        )
        db.upsert_file(updated)
        stored_hash = db.get_file_hash(sample_file.rel_path)
        assert stored_hash == "newHash999"

    def test_get_file_hash_returns_none_for_unknown(self, db: Database) -> None:
        assert db.get_file_hash("does/not/exist.py") is None

    def test_get_file_hash_returns_stored_hash(
        self, db: Database, sample_file: IndexedFile
    ) -> None:
        db.upsert_file(sample_file)
        stored = db.get_file_hash(sample_file.rel_path)
        assert stored == sample_file.hash


# ---------------------------------------------------------------------------
# insert_symbol
# ---------------------------------------------------------------------------

class TestInsertSymbol:
    def test_insert_returns_id(self, db: Database, sample_symbol: Symbol) -> None:
        symbol_id = db.insert_symbol(sample_symbol)
        assert isinstance(symbol_id, int)
        assert symbol_id > 0

    def test_get_symbol_by_name(self, db: Database, sample_symbol: Symbol) -> None:
        db.insert_symbol(sample_symbol)
        results = db.get_symbol_by_name("authenticate_user")
        assert len(results) == 1
        sym = results[0]
        assert sym.name == "authenticate_user"
        assert sym.kind == SymbolKind.METHOD
        assert sym.qualified_name == "LoginView.authenticate_user"

    def test_get_symbols_for_file(self, db: Database, sample_symbol: Symbol) -> None:
        db.insert_symbol(sample_symbol)
        symbols = db.get_symbols_for_file(sample_symbol.file_id)
        assert len(symbols) == 1
        assert symbols[0].name == "authenticate_user"

    def test_decorators_roundtrip(self, db: Database, sample_symbol: Symbol) -> None:
        db.insert_symbol(sample_symbol)
        retrieved = db.get_symbol_by_name("authenticate_user")[0]
        assert retrieved.decorators == ["@login_required"]

    def test_is_public_roundtrip(self, db: Database, sample_symbol: Symbol) -> None:
        db.insert_symbol(sample_symbol)
        retrieved = db.get_symbol_by_name("authenticate_user")[0]
        assert retrieved.is_public is True

    def test_docstring_roundtrip(self, db: Database, sample_symbol: Symbol) -> None:
        db.insert_symbol(sample_symbol)
        retrieved = db.get_symbol_by_name("authenticate_user")[0]
        assert retrieved.docstring == "Authenticate a user by username."

    def test_multiple_symbols_different_files(self, db: Database, sample_file: IndexedFile) -> None:
        file1 = sample_file
        file2 = IndexedFile(
            path="/repo/src/api/views.py",
            rel_path="src/api/views.py",
            language=Language.PYTHON,
            hash="def456",
            size_bytes=512,
        )
        fid1 = db.upsert_file(file1)
        fid2 = db.upsert_file(file2)

        sym1 = Symbol(
            file_id=fid1, name="login", qualified_name="login",
            kind=SymbolKind.FUNCTION, line_start=1, line_end=5,
            signature="def login()", body="def login(): pass",
        )
        sym2 = Symbol(
            file_id=fid2, name="logout", qualified_name="logout",
            kind=SymbolKind.FUNCTION, line_start=1, line_end=5,
            signature="def logout()", body="def logout(): pass",
        )
        db.insert_symbol(sym1)
        db.insert_symbol(sym2)

        assert len(db.get_symbols_for_file(fid1)) == 1
        assert len(db.get_symbols_for_file(fid2)) == 1


# ---------------------------------------------------------------------------
# delete_file_symbols
# ---------------------------------------------------------------------------

class TestDeleteFileSymbols:
    def test_delete_removes_symbols(self, db: Database, sample_symbol: Symbol) -> None:
        db.insert_symbol(sample_symbol)
        db.delete_file_symbols(sample_symbol.file_id)
        assert db.get_symbols_for_file(sample_symbol.file_id) == []

    def test_delete_removes_imports(self, db: Database, sample_file: IndexedFile) -> None:
        file_id = db.upsert_file(sample_file)
        edge = ImportEdge(file_id=file_id, imported_from="django.contrib.auth", imported_names=["authenticate"])
        db.insert_imports([edge])
        db.delete_file_symbols(file_id)
        assert db.get_imports_for_file(file_id) == []


# ---------------------------------------------------------------------------
# FTS5 search (BM25)
# ---------------------------------------------------------------------------

class TestFTS5Search:
    def _insert_symbol(
        self,
        db: Database,
        file_id: int,
        name: str,
        body: str,
        docstring: str = "",
    ) -> int:
        sym = Symbol(
            file_id=file_id,
            name=name,
            qualified_name=name,
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=10,
            signature=f"def {name}()",
            body=body,
            docstring=docstring,
        )
        return db.insert_symbol(sym)

    def test_bm25_search_finds_match(self, db: Database, sample_file: IndexedFile) -> None:
        file_id = db.upsert_file(sample_file)
        self._insert_symbol(
            db, file_id, "authenticate_user",
            body="def authenticate_user(username, password):\n    return check_password(username, password)",
            docstring="Authenticate a user by checking their password.",
        )
        results = db.bm25_search("authenticate password")
        assert len(results) > 0

    def test_bm25_search_returns_symbol_id(self, db: Database, sample_file: IndexedFile) -> None:
        file_id = db.upsert_file(sample_file)
        sym_id = self._insert_symbol(
            db, file_id, "compute_hash",
            body="def compute_hash(data): return hashlib.sha256(data)",
            docstring="Compute SHA-256 hash of data.",
        )
        results = db.bm25_search("hash sha256")
        ids = [r[0] for r in results]
        assert sym_id in ids

    def test_bm25_search_no_results_for_unrelated_query(
        self, db: Database, sample_file: IndexedFile
    ) -> None:
        file_id = db.upsert_file(sample_file)
        self._insert_symbol(
            db, file_id, "send_email",
            body="def send_email(to, subject): smtp.send(to, subject)",
        )
        results = db.bm25_search("quantum_teleportation_algorithm_xyz")
        assert results == []

    def test_bm25_search_rank_is_float(self, db: Database, sample_file: IndexedFile) -> None:
        file_id = db.upsert_file(sample_file)
        self._insert_symbol(
            db, file_id, "process_payment",
            body="def process_payment(amount): stripe.charge(amount)",
            docstring="Process a Stripe payment.",
        )
        results = db.bm25_search("payment stripe")
        assert len(results) > 0
        sym_id, rank = results[0]
        assert isinstance(rank, float)

    def test_bm25_fts5_trigger_fires_on_insert(
        self, db: Database, sample_file: IndexedFile
    ) -> None:
        """FTS5 trigger should automatically index newly inserted symbols."""
        file_id = db.upsert_file(sample_file)
        self._insert_symbol(
            db, file_id, "validate_token",
            body="def validate_token(token): return jwt.decode(token)",
            docstring="Validate a JWT token.",
        )
        results = db.bm25_search("validate jwt token")
        assert len(results) > 0

    def test_bm25_limit_respected(self, db: Database, sample_file: IndexedFile) -> None:
        file_id = db.upsert_file(sample_file)
        for i in range(10):
            self._insert_symbol(
                db, file_id, f"func_{i}",
                body=f"def func_{i}(): return process_data_{i}()",
                docstring=f"Process data variant {i}.",
            )
        results = db.bm25_search("process data", limit=3)
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

class TestChunks:
    def test_insert_chunk_returns_id(self, db: Database, sample_symbol: Symbol) -> None:
        sym_id = db.insert_symbol(sample_symbol)
        chunk = Chunk(symbol_id=sym_id, chunk_text="def foo(): pass", token_count=10)
        chunk_id = db.insert_chunk(chunk)
        assert isinstance(chunk_id, int)
        assert chunk_id > 0

    def test_get_chunk_ids_for_file(self, db: Database, sample_symbol: Symbol) -> None:
        sym_id = db.insert_symbol(sample_symbol)
        chunk = Chunk(symbol_id=sym_id, chunk_text="def foo(): pass", token_count=10)
        chunk_id = db.insert_chunk(chunk)
        ids = db.get_chunk_ids_for_file(sample_symbol.file_id)
        assert chunk_id in ids

    def test_get_first_chunk_for_symbol(self, db: Database, sample_symbol: Symbol) -> None:
        sym_id = db.insert_symbol(sample_symbol)
        chunk = Chunk(symbol_id=sym_id, chunk_text="body text", token_count=5)
        chunk_id = db.insert_chunk(chunk)
        result = db.get_first_chunk_for_symbol(sym_id)
        assert result is not None
        assert result.id == chunk_id
        assert result.chunk_text == "body text"

    def test_get_first_chunk_returns_none_when_no_chunk(
        self, db: Database, sample_symbol: Symbol
    ) -> None:
        sym_id = db.insert_symbol(sample_symbol)
        assert db.get_first_chunk_for_symbol(sym_id) is None


# ---------------------------------------------------------------------------
# Hydration
# ---------------------------------------------------------------------------

class TestHydration:
    def test_get_chunk_with_context(self, db: Database, sample_symbol: Symbol) -> None:
        sym_id = db.insert_symbol(sample_symbol)
        chunk = Chunk(symbol_id=sym_id, chunk_text="some chunk text", token_count=8)
        chunk_id = db.insert_chunk(chunk)

        result = db.get_chunk_with_context(chunk_id)
        assert result is not None
        c, s, f = result
        assert c.id == chunk_id
        assert s.id == sym_id
        assert s.name == "authenticate_user"
        assert f.rel_path == "src/auth/login.py"

    def test_get_chunk_with_context_returns_none_for_missing(self, db: Database) -> None:
        assert db.get_chunk_with_context(99999) is None

    def test_get_symbol_with_file(self, db: Database, sample_symbol: Symbol) -> None:
        sym_id = db.insert_symbol(sample_symbol)
        result = db.get_symbol_with_file(sym_id)
        assert result is not None
        s, f = result
        assert s.id == sym_id
        assert f.rel_path == "src/auth/login.py"

    def test_get_symbol_with_file_returns_none_for_missing(self, db: Database) -> None:
        assert db.get_symbol_with_file(99999) is None


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class TestVectorStore:
    DIM = 4  # tiny dimension for fast tests

    @pytest.fixture()
    def vs(self, tmp_path: Path) -> VectorStore:
        return VectorStore(tmp_path / "vectors.db", dimension=self.DIM)

    def _unit_vec(self, *components: float) -> list[float]:
        """Normalize a vector to unit length."""
        mag = math.sqrt(sum(x * x for x in components))
        return [x / mag for x in components]

    def test_upsert_and_search_returns_results(self, vs: VectorStore) -> None:
        emb = [1.0, 0.0, 0.0, 0.0]
        vs.upsert(chunk_id=1, embedding=emb)
        results = vs.search(emb, k=5)
        assert len(results) == 1
        chunk_id, dist = results[0]
        assert chunk_id == 1

    def test_search_ordering_closest_first(self, vs: VectorStore) -> None:
        """The most similar vector should come first (lowest L2 distance)."""
        vs.upsert(chunk_id=1, embedding=[1.0, 0.0, 0.0, 0.0])
        vs.upsert(chunk_id=2, embedding=[0.0, 1.0, 0.0, 0.0])
        vs.upsert(chunk_id=3, embedding=[0.0, 0.0, 1.0, 0.0])

        query = [1.0, 0.0, 0.0, 0.0]
        results = vs.search(query, k=3)
        assert results[0][0] == 1  # chunk_id=1 is identical → distance 0

    def test_upsert_replaces_existing(self, vs: VectorStore) -> None:
        vs.upsert(chunk_id=42, embedding=[1.0, 0.0, 0.0, 0.0])
        vs.upsert(chunk_id=42, embedding=[0.0, 1.0, 0.0, 0.0])
        results = vs.search([0.0, 1.0, 0.0, 0.0], k=5)
        assert len(results) == 1
        assert results[0][0] == 42

    def test_upsert_batch(self, vs: VectorStore) -> None:
        items = [
            (10, [1.0, 0.0, 0.0, 0.0]),
            (11, [0.0, 1.0, 0.0, 0.0]),
            (12, [0.0, 0.0, 1.0, 0.0]),
        ]
        vs.upsert_batch(items)
        results = vs.search([1.0, 0.0, 0.0, 0.0], k=10)
        ids = {r[0] for r in results}
        assert {10, 11, 12}.issubset(ids)

    def test_delete_removes_vector(self, vs: VectorStore) -> None:
        vs.upsert(chunk_id=5, embedding=[1.0, 0.0, 0.0, 0.0])
        vs.delete(chunk_id=5)
        results = vs.search([1.0, 0.0, 0.0, 0.0], k=5)
        assert all(r[0] != 5 for r in results)

    def test_delete_batch(self, vs: VectorStore) -> None:
        vs.upsert_batch([
            (20, [1.0, 0.0, 0.0, 0.0]),
            (21, [0.0, 1.0, 0.0, 0.0]),
            (22, [0.0, 0.0, 1.0, 0.0]),
        ])
        vs.delete_batch([20, 21])
        results = vs.search([1.0, 0.0, 0.0, 0.0], k=10)
        ids = {r[0] for r in results}
        assert 20 not in ids
        assert 21 not in ids
        assert 22 in ids

    def test_delete_batch_empty_list_is_noop(self, vs: VectorStore) -> None:
        vs.upsert(chunk_id=99, embedding=[1.0, 0.0, 0.0, 0.0])
        vs.delete_batch([])  # should not raise
        results = vs.search([1.0, 0.0, 0.0, 0.0], k=5)
        assert any(r[0] == 99 for r in results)

    def test_search_k_limit_respected(self, vs: VectorStore) -> None:
        for i in range(10):
            vs.upsert(chunk_id=i, embedding=[float(i % 4 == j) for j in range(4)])
        results = vs.search([1.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) <= 3

    def test_distance_is_non_negative(self, vs: VectorStore) -> None:
        vs.upsert(chunk_id=1, embedding=[0.5, 0.5, 0.5, 0.5])
        vs.upsert(chunk_id=2, embedding=[0.0, 0.0, 0.0, 1.0])
        results = vs.search([0.5, 0.5, 0.5, 0.5], k=5)
        for _, dist in results:
            assert dist >= 0.0
