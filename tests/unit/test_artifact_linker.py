"""
Unit tests for ArtifactLinker — scans artifacts (real DB fixtures, no mocks
for the regex path) for symbol-name mentions and asserts GenericEdges land
correctly. Mirrors test_git_linker.py's conventions.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from trelix.core.config import ArtifactLinkerConfig
from trelix.core.models import Artifact, IndexedFile, Language, Symbol, SymbolKind
from trelix.indexing.artifact_linker import ArtifactLinker
from trelix.store.db import Database

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file(db: Database, rel_path: str) -> int:
    f = IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash="deadbeef",
        size_bytes=512,
    )
    return db.upsert_file(f)


def _insert_symbol(db: Database, file_id: int, name: str, qualified_name: str | None = None) -> int:
    sym = Symbol(
        file_id=file_id,
        name=name,
        qualified_name=qualified_name or name,
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=10,
        signature=f"def {name}():",
        body=f"def {name}(): pass",
    )
    sym_id = db.insert_symbol(sym)
    db._conn.commit()
    return sym_id


def _make_artifact(source_ref: str, title: str = "", body: str = "") -> Artifact:
    return Artifact(source_ref=source_ref, artifact_kind="ticket", title=title, body=body)


# ---------------------------------------------------------------------------
# Tests — regex reference-extraction (real DB, no mocks)
# ---------------------------------------------------------------------------


class TestArtifactLinkerRegexMatch:
    def test_links_symbol_mentioned_by_name_in_title(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login")
        db.upsert_artifact(_make_artifact("ticket:PROJ-1", title="login is broken"))

        count = ArtifactLinker(db, ArtifactLinkerConfig()).link()

        assert count == 1
        assert db.get_generic_edge_targets(sym_id) == ["ticket:PROJ-1"]

    def test_links_symbol_mentioned_by_qualified_name_in_body(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login", qualified_name="auth.login")
        db.upsert_artifact(
            _make_artifact("ticket:PROJ-2", body="Users cannot call auth.login successfully")
        )

        count = ArtifactLinker(db, ArtifactLinkerConfig()).link()

        assert count == 1
        assert db.get_generic_edge_targets(sym_id) == ["ticket:PROJ-2"]

    def test_no_symbol_mentioned_links_nothing_when_fallback_disabled(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        _insert_symbol(db, file_id, "login")
        db.upsert_artifact(_make_artifact("ticket:PROJ-3", title="something unrelated"))

        count = ArtifactLinker(db, ArtifactLinkerConfig(embedding_fallback_enabled=False)).link()

        assert count == 0

    def test_multiple_symbols_mentioned_in_one_artifact(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        login_id = _insert_symbol(db, file_id, "login")
        logout_id = _insert_symbol(db, file_id, "logout")
        db.upsert_artifact(_make_artifact("ticket:PROJ-4", title="login and logout both fail"))

        count = ArtifactLinker(db, ArtifactLinkerConfig()).link()

        assert count == 2
        assert db.get_generic_edge_targets(login_id) == ["ticket:PROJ-4"]
        assert db.get_generic_edge_targets(logout_id) == ["ticket:PROJ-4"]

    def test_no_artifacts_returns_zero(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        count = ArtifactLinker(db, ArtifactLinkerConfig()).link()
        assert count == 0

    def test_rerunning_link_on_same_artifacts_does_not_duplicate_edges(
        self, tmp_path: Path
    ) -> None:
        """Re-running (e.g. `trelix link-artifacts` twice) must not duplicate
        generic_edges rows — same DB-level guard (idx_generic_edges_dedup +
        INSERT OR IGNORE) proof as test_git_linker.py's equivalent test."""
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login")
        db.upsert_artifact(_make_artifact("ticket:PROJ-5", title="login is broken"))

        first_count = ArtifactLinker(db, ArtifactLinkerConfig()).link()
        second_count = ArtifactLinker(db, ArtifactLinkerConfig()).link()

        assert first_count == 1
        assert second_count == 1  # link() itself still reports what it tried to insert
        assert db.get_generic_edge_targets(sym_id) == ["ticket:PROJ-5"]
        row_count = db._conn.execute("SELECT COUNT(*) FROM generic_edges").fetchone()[0]
        assert row_count == 1

    def test_link_one_links_a_single_artifact_by_source_ref(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login")
        db.upsert_artifact(_make_artifact("ticket:PROJ-6", title="login is broken"))

        count = ArtifactLinker(db, ArtifactLinkerConfig()).link_one("ticket:PROJ-6")

        assert count == 1
        assert db.get_generic_edge_targets(sym_id) == ["ticket:PROJ-6"]

    def test_link_one_unknown_source_ref_returns_zero(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        count = ArtifactLinker(db, ArtifactLinkerConfig()).link_one("ticket:does-not-exist")
        assert count == 0


# ---------------------------------------------------------------------------
# Tests — embedding-similarity fallback (mocked embedder/vector store only)
# ---------------------------------------------------------------------------


class TestArtifactLinkerEmbeddingFallback:
    def test_falls_back_to_embedding_when_regex_finds_nothing(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login")
        chunk_id = db.insert_chunk_for_symbol(sym_id, "def login(): pass", token_count=5)
        db.upsert_artifact(_make_artifact("ticket:PROJ-7", title="the sign-in flow is broken"))

        from trelix.core.config import IndexConfig

        index_config = IndexConfig(repo_path=str(tmp_path))
        linker = ArtifactLinker(
            db,
            ArtifactLinkerConfig(embedding_fallback_enabled=True, similarity_threshold=0.5),
            index_config=index_config,
        )

        mock_embedder = MagicMock()
        mock_embedder.dimension = 4
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [(chunk_id, 0.1)]  # distance 0.1 -> similarity 0.9

        with (
            patch("trelix.embedder.base.make_embedder", return_value=mock_embedder),
            patch("trelix.store.vector.make_vector_store", return_value=mock_vector_store),
        ):
            count = linker.link()

        assert count == 1
        assert db.get_generic_edge_targets(sym_id) == ["ticket:PROJ-7"]
        # Embedding-fallback matches are lower-confidence than a regex hit.
        row = db._conn.execute(
            "SELECT weight FROM generic_edges WHERE source_ref = ?", ("ticket:PROJ-7",)
        ).fetchone()
        assert row[0] == 0.5

    def test_below_similarity_threshold_is_not_linked(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login")
        chunk_id = db.insert_chunk_for_symbol(sym_id, "def login(): pass", token_count=5)
        db.upsert_artifact(_make_artifact("ticket:PROJ-8", title="unrelated content"))

        from trelix.core.config import IndexConfig

        index_config = IndexConfig(repo_path=str(tmp_path))
        linker = ArtifactLinker(
            db,
            ArtifactLinkerConfig(embedding_fallback_enabled=True, similarity_threshold=0.9),
            index_config=index_config,
        )

        mock_embedder = MagicMock()
        mock_embedder.dimension = 4
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        mock_vector_store = MagicMock()
        # distance 0.5 -> similarity 0.5, below the 0.9 threshold
        mock_vector_store.search.return_value = [(chunk_id, 0.5)]

        with (
            patch("trelix.embedder.base.make_embedder", return_value=mock_embedder),
            patch("trelix.store.vector.make_vector_store", return_value=mock_vector_store),
        ):
            count = linker.link()

        assert count == 0

    def test_embedding_fallback_disabled_by_default(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        _insert_symbol(db, file_id, "login")
        db.upsert_artifact(_make_artifact("ticket:PROJ-9", title="unrelated content"))

        # No index_config, no mocking — must not attempt any embedder/vector
        # store call and must simply return 0.
        count = ArtifactLinker(db, ArtifactLinkerConfig()).link()

        assert count == 0

    def test_embedding_failure_is_non_fatal(self, tmp_path: Path) -> None:
        """An embedder/vector-store construction failure degrades to
        'no matches' for that artifact, never raises."""
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        _insert_symbol(db, file_id, "login")
        db.upsert_artifact(_make_artifact("ticket:PROJ-10", title="unrelated content"))

        from trelix.core.config import IndexConfig

        index_config = IndexConfig(repo_path=str(tmp_path))
        linker = ArtifactLinker(
            db,
            ArtifactLinkerConfig(embedding_fallback_enabled=True),
            index_config=index_config,
        )

        with patch(
            "trelix.embedder.base.make_embedder", side_effect=RuntimeError("no provider configured")
        ):
            count = linker.link()

        assert count == 0


# ---------------------------------------------------------------------------
# Tests — connector sync auto-link integration
# ---------------------------------------------------------------------------


class TestConnectorSyncAutoLink:
    def test_sync_with_linker_produces_generic_edge_end_to_end(self, tmp_path: Path) -> None:
        """ArtifactSource.sync(db, linker=...) — a fetched artifact is
        reachable from generic_edges the moment sync() returns, without a
        separate link pass."""
        from trelix.indexing.connectors.base import ArtifactSource

        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login")

        class _FakeSource(ArtifactSource):
            def validate_config(self) -> None:
                return None

            def fetch(self) -> list[Artifact]:
                return [_make_artifact("ticket:PROJ-11", title="login is broken")]

        linker = ArtifactLinker(db, ArtifactLinkerConfig())
        result = _FakeSource().sync(db, linker=linker)

        assert result.artifacts_fetched == 1
        assert result.artifacts_written == 1
        assert result.errors == 0
        assert result.edges_linked == 1
        assert db.get_generic_edge_targets(sym_id) == ["ticket:PROJ-11"]

    def test_sync_without_linker_writes_artifact_but_creates_no_edge(self, tmp_path: Path) -> None:
        from trelix.indexing.connectors.base import ArtifactSource

        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "auth.py")
        sym_id = _insert_symbol(db, file_id, "login")

        class _FakeSource(ArtifactSource):
            def validate_config(self) -> None:
                return None

            def fetch(self) -> list[Artifact]:
                return [_make_artifact("ticket:PROJ-12", title="login is broken")]

        result = _FakeSource().sync(db)  # linker omitted — default None

        assert result.artifacts_written == 1
        assert result.edges_linked == 0
        assert db.get_generic_edge_targets(sym_id) == []
