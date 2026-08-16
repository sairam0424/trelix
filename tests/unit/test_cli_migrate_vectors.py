"""Regression tests for the migrate-vectors CLI command's coverage of all
embedding sources (primary chunks, multi-granularity sub-chunks, and
RAPTOR-style file-summary embeddings).

Investigation note (trelix v2.6.x scale backlog, Plan A / Task A-2):
The backlog item this test file was written for describes a bug where
migrate-vectors only reads chunk_embeddings and silently drops sub-chunk /
file-summary vectors stored in separate tables. That premise does not match
the current codebase: sub-chunk and file-summary embeddings are NOT stored
in separate tables. They live in the *same* sqlite-vec chunk_embeddings
vec0 virtual table as regular chunks (see src/trelix/store/vector.py),
distinguished only by an id-sentinel convention:
  - regular chunks:      chunk_id = chunk_id (positive, small)
  - file summaries:      chunk_id = -(file_id)                (negative)
  - sub-chunks:          chunk_id = sub_chunk_id + 10_000_000  (large positive)

migrate_vectors's existing query (``SELECT chunk_id, embedding FROM
chunk_embeddings LIMIT ? OFFSET ?``) has no WHERE clause restricting the id
range, so it already migrates every row regardless of which sentinel space
it falls into. This was confirmed by direct execution against a real
sqlite-vec-backed store before writing this test (not just by reading code).

These tests exist as a regression guard: if migrate_vectors is ever
rewritten to filter by an id range (e.g. "chunk_id > 0" to skip sentinels),
these tests will catch the resulting silent data loss.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Fake qdrant_client module — mirrors the helper in tests/unit/test_store_qdrant.py
# so migrate-vectors can run against a fake Qdrant without a live instance or
# the optional qdrant-client dependency installed.
# ---------------------------------------------------------------------------


def _build_fake_qdrant_module() -> tuple[types.ModuleType, MagicMock]:
    mock_client_instance = MagicMock()
    mock_client_instance.get_collections.return_value = MagicMock(collections=[])
    mock_client_instance.create_collection.return_value = None
    mock_client_instance.upsert.return_value = None
    mock_client_instance.search.return_value = []

    class PointStruct:
        def __init__(self, id: int, vector: list, payload: dict) -> None:
            self.id = id
            self.vector = vector
            self.payload = payload

    class PointIdsList:
        def __init__(self, points: list[int]) -> None:
            self.points = points

    class Distance:
        COSINE = "Cosine"

    class HnswConfigDiff:
        def __init__(self, m: int, ef_construct: int) -> None:
            self.m = m
            self.ef_construct = ef_construct

    class VectorParams:
        def __init__(self, size: int, distance: str, hnsw_config: Any = None) -> None:
            self.size = size
            self.distance = distance
            self.hnsw_config = hnsw_config

    fake_models = types.ModuleType("qdrant_client.models")
    fake_models.PointStruct = PointStruct  # type: ignore[attr-defined]
    fake_models.PointIdsList = PointIdsList  # type: ignore[attr-defined]
    fake_models.Distance = Distance  # type: ignore[attr-defined]
    fake_models.HnswConfigDiff = HnswConfigDiff  # type: ignore[attr-defined]
    fake_models.VectorParams = VectorParams  # type: ignore[attr-defined]

    fake_pkg = types.ModuleType("qdrant_client")
    fake_pkg.QdrantClient = MagicMock(return_value=mock_client_instance)  # type: ignore[attr-defined]
    fake_pkg.models = fake_models  # type: ignore[attr-defined]

    return fake_pkg, mock_client_instance


def _inject_fake_qdrant() -> MagicMock:
    for key in list(sys.modules):
        if key.startswith("trelix.store.vector_qdrant"):
            del sys.modules[key]

    fake_pkg, mock_instance = _build_fake_qdrant_module()
    sys.modules["qdrant_client"] = fake_pkg
    sys.modules["qdrant_client.models"] = fake_pkg.models
    return mock_instance


def _remove_fake_qdrant() -> None:
    sys.modules.pop("qdrant_client", None)
    sys.modules.pop("qdrant_client.models", None)
    for key in list(sys.modules):
        if key.startswith("trelix.store.vector_qdrant"):
            del sys.modules[key]


class TestMigrateVectorsCoversAllEmbeddingSources:
    """Confirms migrate-vectors migrates chunk, sub-chunk, and file-summary
    embeddings — not just primary chunks."""

    def setup_method(self) -> None:
        self.mock_client = _inject_fake_qdrant()

    def teardown_method(self) -> None:
        _remove_fake_qdrant()

    def _seed_index(self, repo: Path) -> None:
        """Build a real sqlite-vec-backed index containing all three
        embedding sources, at the exact path IndexConfig/migrate_vectors
        expects to find it."""
        from trelix.core.config import IndexConfig
        from trelix.store.vector import SQLiteVectorStore

        config = IndexConfig(repo_path=str(repo))
        db_path = config.db_path_absolute

        store = SQLiteVectorStore(db_path=db_path, dimension=4)
        # 3 regular chunk embeddings
        store.upsert_batch([(i, [0.1, 0.2, 0.3, 0.4]) for i in range(1, 4)])
        # 2 sub-chunk embeddings (multi-granularity)
        store.upsert_sub_chunk_embedding(1, [0.5, 0.6, 0.7, 0.8])
        store.upsert_sub_chunk_embedding(2, [0.55, 0.65, 0.75, 0.85])
        # 2 file-summary embeddings (RAPTOR-style)
        store.upsert_file_summary_embedding(1, [0.9, 0.1, 0.2, 0.3])
        store.upsert_file_summary_embedding(2, [0.91, 0.11, 0.21, 0.31])
        store.close()

    def test_migrate_vectors_carries_sub_chunk_and_file_summary_embeddings(
        self, tmp_path: Path
    ) -> None:
        """All 7 embeddings (3 regular + 2 sub-chunk + 2 file-summary) must be
        upserted into Qdrant — none silently dropped."""
        from typer.testing import CliRunner

        from trelix.cli.main import app

        repo = tmp_path / "repo"
        repo.mkdir()
        self._seed_index(repo)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "migrate-vectors",
                str(repo),
                "--to",
                "qdrant",
                "--url",
                "http://localhost:6333",
            ],
        )

        assert result.exit_code == 0, f"output={result.output!r}"

        migrated_ids: set[int] = set()
        for call in self.mock_client.upsert.call_args_list:
            points = call.kwargs["points"]
            for point in points:
                migrated_ids.add(point.id)

        expected_ids = {
            1,
            2,
            3,  # regular chunks
            1 + 10_000_000,
            2 + 10_000_000,  # sub-chunks
            -1,
            -2,  # file summaries
        }
        assert migrated_ids == expected_ids, (
            "migrate-vectors must migrate every embedding source (regular "
            "chunks, sub-chunks, and file summaries) since they share the "
            f"same chunk_embeddings table. Got: {sorted(migrated_ids)}, "
            f"expected: {sorted(expected_ids)}"
        )

    def test_migrate_vectors_reports_total_count_including_all_sources(
        self, tmp_path: Path
    ) -> None:
        """The 'Migrating N embeddings' summary must count sub-chunk and
        file-summary rows too, not just regular chunks."""
        from typer.testing import CliRunner

        from trelix.cli.main import app

        repo = tmp_path / "repo"
        repo.mkdir()
        self._seed_index(repo)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "migrate-vectors",
                str(repo),
                "--to",
                "qdrant",
                "--url",
                "http://localhost:6333",
            ],
        )

        assert result.exit_code == 0, f"output={result.output!r}"
        assert "Migrating 7 embeddings" in result.output
        assert "7 embeddings written to Qdrant" in result.output


class TestResetActuallyResets:
    """`--reset` must leave the index in a state a re-index can repair.

    It previously did none of the three things that requires, and reported success
    anyway — printing "Embeddings and dimension metadata cleared" while clearing no
    embeddings at all:

    1. `db.clear_all_embeddings()` ran `DELETE FROM chunk_embeddings` on Database's own
       connection, which never loads the sqlite-vec extension. That raises
       `OperationalError: no such module: vec0`, and the method caught it with
       `except Exception: pass` — a guaranteed no-op on every install.
    2. Even a successful delete could not help: a vec0 table's vector width is fixed by
       its CREATE statement, and both creation paths use IF NOT EXISTS, so the table
       keeps rejecting vectors of the new dimension. Verified: after deleting every row
       and re-issuing the CREATE at FLOAT[8], the schema still said FLOAT[4].
    3. File and symbol hashes survived, so the re-index the message told you to run
       either reported "Nothing to index — all files up to date" or re-parsed
       everything and still embedded nothing, because `_insert_one` leaves symbols with
       a matching content_hash completely untouched.

    Measured end-to-end after the fix: index at 4 dims (3 vectors / 3 chunks) ->
    --reset for 8 dims (table FLOAT[8], 0 vectors) -> re-index (chunks_embedded=3,
    3 vectors / 3 chunks).
    """

    def test_clear_all_embeddings_refuses_without_a_vector_store(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The silent no-op is now impossible: no store means an explicit error."""
        import pytest

        from trelix.store.db import Database

        db = Database(tmp_path / "index.db")
        with pytest.raises(ValueError, match="requires the vector_store"):
            db.clear_all_embeddings()

    def test_clear_all_embeddings_rebuilds_through_the_store(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from unittest.mock import MagicMock

        from trelix.store.db import Database

        db = Database(tmp_path / "index.db")
        store = MagicMock()
        db.clear_all_embeddings(vector_store=store)

        (
            store.recreate.assert_called_once(),
            ("clearing must go through the store, which owns the virtual table"),
        )

    def test_invalidate_file_hashes_reports_what_it_changed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from trelix.core.models import IndexedFile, Language
        from trelix.store.db import Database

        db = Database(tmp_path / "index.db")
        for name in ("a.py", "b.py"):
            db.upsert_file(
                IndexedFile(
                    path=f"/repo/{name}",
                    rel_path=name,
                    language=Language.PYTHON,
                    hash="deadbeef",
                    size_bytes=10,
                )
            )
        db._conn.commit()

        assert db.invalidate_all_file_hashes() == 2
        assert db.get_file_hash("a.py") == "", "the hash was not actually blanked"

    def test_invalidate_symbol_hashes_reports_what_it_changed(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Without this, a re-index re-parses every file and embeds nothing."""
        from trelix.core.models import IndexedFile, Language, Symbol, SymbolKind
        from trelix.store.db import Database

        db = Database(tmp_path / "index.db")
        file_id = db.upsert_file(
            IndexedFile(
                path="/repo/a.py",
                rel_path="a.py",
                language=Language.PYTHON,
                hash="deadbeef",
                size_bytes=10,
            )
        )
        db.insert_symbol(
            Symbol(
                file_id=file_id,
                name="f",
                qualified_name="f",
                kind=SymbolKind.FUNCTION,
                line_start=1,
                line_end=2,
                signature="def f():",
                body="def f(): pass",
            )
        )
        db._conn.commit()

        assert db.invalidate_all_symbol_hashes() == 1
        stored = db._conn.execute("SELECT content_hash FROM symbols").fetchone()[0]
        assert stored == "", "the symbol content hash was not blanked"

    def test_the_error_message_names_the_provider_flag(self) -> None:
        """The remedy the guard prints must be the one that works.

        --provider is what decides the rebuilt table's width, so a message omitting it
        sends the reader to rebuild at the dimension they are trying to leave.
        """
        from trelix.store.dimension_guard import DimensionMismatchError

        message = str(DimensionMismatchError(stored=384, current=3072, provider="azure"))
        assert "--reset" in message
        assert "--provider azure" in message

    def test_reset_refuses_a_backend_it_cannot_rebuild(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Qdrant and LanceDB pin their dimension at creation too, and --reset has no
        per-backend dispatch — so without this check it would report success having
        touched nothing at all for those users."""
        from unittest.mock import patch

        from typer.testing import CliRunner

        from trelix.cli.main import app

        with patch.dict("os.environ", {"TRELIX_STORE_BACKEND": "qdrant"}):
            res = CliRunner().invoke(app, ["migrate-vectors", str(tmp_path), "--reset"])

        assert res.exit_code == 1, res.output
        assert "does not support" in res.output and "qdrant" in res.output
