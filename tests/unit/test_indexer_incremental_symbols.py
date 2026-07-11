"""Tests confirming Indexer skips re-embedding symbols whose content is unchanged
on a partial re-index (Item 5 of the v2.6.0 scale backlog).

Strategy mirrors tests/unit/test_indexer_core.py: mock make_embedder and
make_vector_store so no ML models are loaded, use a real SQLite Database and
the real tree-sitter Python parser so symbol content_hash values are genuine.
"""

from __future__ import annotations

import pathlib
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

from trelix.core.config import EmbedderConfig, IndexConfig, StoreConfig

_DIM = 4


class _FakeEmbedder:
    """Records every text passed to embed() so tests can assert on exactly
    what was (or wasn't) sent for re-embedding."""

    def __init__(self) -> None:
        self.embed_call_texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_call_texts.extend(texts)
        return [[0.1] * _DIM for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * _DIM

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    @property
    def dimension(self) -> int:
        return _DIM


class _FakeVectorStore:
    """In-memory vector store stub — stores nothing, raises nothing."""

    def upsert_batch(self, pairs: list[tuple[int, list[float]]]) -> None:
        pass

    def delete_batch(self, ids: list[int]) -> None:
        pass

    def search(self, vector: list[float], k: int) -> list[Any]:
        return []


@contextmanager
def _patch_rich_progress():
    """Suppress rich terminal output during tests."""
    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task = MagicMock(return_value=0)
    mock_progress.advance = MagicMock()
    with patch("trelix.indexing.indexer.Progress", return_value=mock_progress):
        yield mock_progress


def _make_indexer(tmp_dir: str, fake_embedder: _FakeEmbedder) -> Any:
    from trelix.indexing.indexer import Indexer

    cfg = IndexConfig(
        repo_path=tmp_dir,
        incremental=False,
        store=StoreConfig(db_path=str(pathlib.Path(tmp_dir) / ".trelix" / "index.db")),
        embedder=EmbedderConfig.model_construct(provider="local"),
    )

    with (
        patch("trelix.indexing.indexer.make_embedder", return_value=fake_embedder),
        patch("trelix.indexing.indexer.make_vector_store", return_value=_FakeVectorStore()),
    ):
        indexer = Indexer(cfg, quiet=True)

    return indexer


class TestIncrementalSymbolReEmbedding:
    def test_unchanged_symbol_is_not_re_embedded_on_second_index_pass(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Index a file, then re-index it with ONE function body changed and
        ONE unchanged. Only the changed function's chunk must be re-embedded."""
        repo = tmp_path
        source_file = repo / "mod.py"
        source_file.write_text(
            "def unchanged_fn():\n    return 1\n\n\ndef changed_fn():\n    return 2\n"
        )

        fake_embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), fake_embedder)

        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        fake_embedder.embed_call_texts.clear()

        # Second pass: change only changed_fn's body.
        source_file.write_text(
            "def unchanged_fn():\n    return 1\n\n\ndef changed_fn():\n    return 999\n"
        )
        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        embed_call_texts = fake_embedder.embed_call_texts
        changed_fn_reembedded = any("999" in text for text in embed_call_texts)
        unchanged_fn_reembedded = any(
            "return 1" in text and "999" not in text for text in embed_call_texts
        )
        assert changed_fn_reembedded, "changed_fn's new body must be re-embedded"
        assert not unchanged_fn_reembedded, (
            "unchanged_fn must NOT be re-embedded — its content_hash didn't change. "
            f"Embed was called with: {embed_call_texts}"
        )

    def test_removed_symbol_is_deleted_from_db(self, tmp_path: pathlib.Path) -> None:
        """A symbol present in the first pass but absent from the second parse
        (function deleted from the file) must be removed from the symbols table."""
        repo = tmp_path
        source_file = repo / "mod.py"
        source_file.write_text("def stays():\n    return 1\n\n\ndef goes():\n    return 2\n")

        fake_embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), fake_embedder)

        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        source_file.write_text("def stays():\n    return 1\n")
        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        rows = indexer.db._conn.execute("SELECT qualified_name FROM symbols").fetchall()
        names = {r[0] for r in rows}
        assert "stays" in names
        assert "goes" not in names

    def test_unchanged_symbol_keeps_its_row_id(self, tmp_path: pathlib.Path) -> None:
        """Unchanged symbols must not be deleted+re-inserted — their DB row
        (and hence chunk_id/embedding) is left untouched, so the symbol id
        must be stable across the second pass."""
        repo = tmp_path
        source_file = repo / "mod.py"
        source_file.write_text(
            "def unchanged_fn():\n    return 1\n\n\ndef changed_fn():\n    return 2\n"
        )

        fake_embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), fake_embedder)

        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        row = indexer.db._conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'unchanged_fn'"
        ).fetchone()
        first_pass_id = row[0]

        source_file.write_text(
            "def unchanged_fn():\n    return 1\n\n\ndef changed_fn():\n    return 999\n"
        )
        with _patch_rich_progress():
            indexer.index_file(str(source_file))

        row = indexer.db._conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'unchanged_fn'"
        ).fetchone()
        second_pass_id = row[0]

        assert first_pass_id == second_pass_id, (
            "unchanged_fn's symbol row must be preserved (same id), not deleted and re-inserted"
        )
