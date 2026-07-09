"""
Unit tests for trelix.indexing.indexer -- core paths.

Strategy:
  - Mock make_embedder and make_vector_store in Indexer.__init__ so tests
    run without sentence-transformers, OpenAI, or sqlite-vec installed.
  - Use real tempfile directories and actual SQLite Database instances (no
    mock on the DB layer) so the stat counters are driven by real code paths.
  - Patch rich.progress.Progress to prevent terminal rendering in CI.

Covered:
  - Indexer.__init__ wires up db, embedder, vector_store, chunker, walker.
  - index() returns a stats dict with the expected keys.
  - index() on an empty directory returns files_indexed=0.
  - index() on a directory containing one Python file returns files_indexed >= 1.
  - index_file() on an existing file updates the symbol table (incremental update).
"""

from __future__ import annotations

import pathlib
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

from trelix.core.config import EmbedderConfig, IndexConfig, StoreConfig
from trelix.core.models import Symbol, SymbolKind
from trelix.indexing.parser.base import BaseParser, ParseResult
from trelix.store.db import Database

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_DIM = 4  # tiny embedding dimension — keeps the sqlite-vec index small


class _FakeEmbedder:
    """Minimal embedder that returns zero vectors without touching any model."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * _DIM for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * _DIM

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


class _FakeParser(BaseParser):
    """
    Minimal parser stub that returns one Symbol for any source file.

    Used when tree_sitter_languages is not installed so the real Python
    parser cannot be instantiated.  Lets TestIndexSingleFile exercise the
    files_indexed counter without any native grammar wheels.
    """

    @property
    def language_name(self) -> str:
        return "python"

    def parse(self, source: str, file_id: int) -> ParseResult:
        sym = Symbol(
            file_id=file_id,
            name="__stub__",
            qualified_name="__stub__",
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=max(1, len(source.splitlines())),
            signature="def __stub__()",
            body=source,
        )
        return ParseResult(
            symbols=[sym],
            call_edges=[],
            import_edges=[],
            parse_errors=0,
        )


def _fake_get_parser(language: Any) -> _FakeParser:  # noqa: ANN401
    """Drop-in replacement for get_parser that always returns _FakeParser."""
    return _FakeParser()


@contextmanager
def _patch_rich_progress(*, fake_parser: bool = False):
    """
    Suppress rich terminal output during tests.

    When ``fake_parser=True`` also patches ``get_parser`` so that Python
    files are processed by ``_FakeParser`` rather than the real tree-sitter
    extractor.  Use this for tests that need *files_indexed >= 1* but run in
    environments where ``tree_sitter_languages`` is not installed.
    """
    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task = MagicMock(return_value=0)
    mock_progress.advance = MagicMock()
    if fake_parser:
        with (
            patch("trelix.indexing.indexer.Progress", return_value=mock_progress),
            patch("trelix.indexing.indexer.get_parser", side_effect=_fake_get_parser),
        ):
            yield mock_progress
    else:
        with patch("trelix.indexing.indexer.Progress", return_value=mock_progress):
            yield mock_progress


def _make_indexer(tmp_dir: str) -> Indexer:  # noqa: F821
    """
    Build an Indexer with fake embedder + vector store so no ML models are
    loaded.  Uses a real SQLite Database so stat counters are exercised.

    Also patches get_parser so the indexer can process Python files even when
    tree_sitter_languages (the native grammar wheels) is not installed.
    """
    from trelix.indexing.indexer import Indexer

    cfg = IndexConfig(
        repo_path=tmp_dir,
        incremental=False,
        store=StoreConfig(db_path=str(pathlib.Path(tmp_dir) / ".trelix" / "index.db")),
        embedder=EmbedderConfig.model_construct(provider="local"),
    )

    with (
        patch("trelix.indexing.indexer.make_embedder", return_value=_FakeEmbedder()),
        patch("trelix.indexing.indexer.make_vector_store", return_value=_FakeVectorStore()),
        patch("trelix.indexing.indexer.get_parser", side_effect=_fake_get_parser),
    ):
        indexer = Indexer(cfg, quiet=True)

    return indexer


# ---------------------------------------------------------------------------
# Indexer.__init__ tests
# ---------------------------------------------------------------------------


class TestIndexerInit:
    def test_db_is_database_instance(self, tmp_path: pathlib.Path) -> None:
        """Indexer.db must be a Database (SQLite-backed) after construction."""
        indexer = _make_indexer(str(tmp_path))
        assert isinstance(indexer.db, Database)

    def test_embedder_is_set(self, tmp_path: pathlib.Path) -> None:
        """Indexer.embedder must be the object returned by make_embedder."""
        indexer = _make_indexer(str(tmp_path))
        assert isinstance(indexer.embedder, _FakeEmbedder)

    def test_vector_store_is_set(self, tmp_path: pathlib.Path) -> None:
        indexer = _make_indexer(str(tmp_path))
        assert isinstance(indexer.vector_store, _FakeVectorStore)

    def test_chunker_is_created(self, tmp_path: pathlib.Path) -> None:
        from trelix.indexing.chunker import Chunker

        indexer = _make_indexer(str(tmp_path))
        assert isinstance(indexer.chunker, Chunker)

    def test_walker_is_created(self, tmp_path: pathlib.Path) -> None:
        from trelix.indexing.walker import FileWalker

        indexer = _make_indexer(str(tmp_path))
        assert isinstance(indexer.walker, FileWalker)


# ---------------------------------------------------------------------------
# index() stats shape
# ---------------------------------------------------------------------------


class TestIndexReturnShape:
    _EXPECTED_KEYS = {
        "files_found",
        "files_indexed",
        "files_skipped",
        "symbols_extracted",
        "chunks_total",
        "chunks_embedded",
        "errors",
        "elapsed_seconds",
    }

    def test_stats_has_all_required_keys(self, tmp_path: pathlib.Path) -> None:
        """index() must return a dict with all expected stat keys."""
        indexer = _make_indexer(str(tmp_path))
        with _patch_rich_progress():
            stats = indexer.index()
        assert self._EXPECTED_KEYS.issubset(stats.keys()), (
            f"Missing keys: {self._EXPECTED_KEYS - stats.keys()}"
        )

    def test_elapsed_seconds_is_positive_float(self, tmp_path: pathlib.Path) -> None:
        indexer = _make_indexer(str(tmp_path))
        with _patch_rich_progress():
            stats = indexer.index()
        assert isinstance(stats["elapsed_seconds"], float)
        assert stats["elapsed_seconds"] >= 0.0

    def test_error_count_is_int(self, tmp_path: pathlib.Path) -> None:
        indexer = _make_indexer(str(tmp_path))
        with _patch_rich_progress():
            stats = indexer.index()
        assert isinstance(stats["errors"], int)


# ---------------------------------------------------------------------------
# index() on empty directory
# ---------------------------------------------------------------------------


class TestIndexEmptyDirectory:
    def test_files_indexed_is_zero(self, tmp_path: pathlib.Path) -> None:
        """An empty repo produces files_indexed=0."""
        indexer = _make_indexer(str(tmp_path))
        with _patch_rich_progress():
            stats = indexer.index()
        assert stats["files_indexed"] == 0

    def test_errors_is_zero(self, tmp_path: pathlib.Path) -> None:
        indexer = _make_indexer(str(tmp_path))
        with _patch_rich_progress():
            stats = indexer.index()
        assert stats["errors"] == 0

    def test_symbols_extracted_is_zero(self, tmp_path: pathlib.Path) -> None:
        indexer = _make_indexer(str(tmp_path))
        with _patch_rich_progress():
            stats = indexer.index()
        assert stats["symbols_extracted"] == 0


# ---------------------------------------------------------------------------
# index() on directory with one Python file
# ---------------------------------------------------------------------------


class TestIndexSingleFile:
    def _write_py(self, directory: pathlib.Path, name: str = "sample.py") -> pathlib.Path:
        p = directory / name
        p.write_text(
            "def hello():\n"
            "    '''Say hello.'''\n"
            "    return 'hello'\n"
            "\n"
            "class Greeter:\n"
            "    def greet(self, name: str) -> str:\n"
            "        return f'Hello, {name}'\n",
            encoding="utf-8",
        )
        return p

    def test_files_indexed_at_least_one(self, tmp_path: pathlib.Path) -> None:
        """A directory with one Python file should index at least 1 file."""
        self._write_py(tmp_path)
        indexer = _make_indexer(str(tmp_path))
        with _patch_rich_progress(fake_parser=True):
            stats = indexer.index()
        assert stats["files_indexed"] >= 1

    def test_files_found_at_least_one(self, tmp_path: pathlib.Path) -> None:
        self._write_py(tmp_path)
        indexer = _make_indexer(str(tmp_path))
        with _patch_rich_progress(fake_parser=True):
            stats = indexer.index()
        assert stats["files_found"] >= 1

    def test_symbols_extracted_at_least_one(self, tmp_path: pathlib.Path) -> None:
        """The two functions/method in the sample file should yield at least one symbol."""
        self._write_py(tmp_path)
        indexer = _make_indexer(str(tmp_path))
        with _patch_rich_progress(fake_parser=True):
            stats = indexer.index()
        assert stats["symbols_extracted"] >= 1

    def test_no_errors(self, tmp_path: pathlib.Path) -> None:
        self._write_py(tmp_path)
        indexer = _make_indexer(str(tmp_path))
        with _patch_rich_progress(fake_parser=True):
            stats = indexer.index()
        assert stats["errors"] == 0

    def test_non_python_file_not_counted(self, tmp_path: pathlib.Path) -> None:
        """A lone .txt file should produce files_indexed=0 (no supported parser)."""
        (tmp_path / "notes.txt").write_text("just notes\n", encoding="utf-8")
        indexer = _make_indexer(str(tmp_path))
        with _patch_rich_progress(fake_parser=True):
            stats = indexer.index()
        # .txt has no parser -> skipped; files_indexed should stay 0
        assert stats["files_indexed"] == 0


# ---------------------------------------------------------------------------
# index_file() incremental update
# ---------------------------------------------------------------------------


class TestIndexFileIncremental:
    def _make_py(self, directory: pathlib.Path, content: str, name: str = "mod.py") -> pathlib.Path:
        p = directory / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_index_file_returns_ok_status(self, tmp_path: pathlib.Path) -> None:
        """index_file() on a valid Python file returns status='ok'."""
        py_file = self._make_py(tmp_path, "def alpha(): pass\n")
        indexer = _make_indexer(str(tmp_path))

        with _patch_rich_progress():
            result = indexer.index_file(str(py_file))

        assert result["status"] == "ok"

    def test_index_file_second_call_with_same_content_is_skipped(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Re-indexing an unchanged file should be detected as skipped."""
        py_file = self._make_py(tmp_path, "def beta(): pass\n")
        indexer = _make_indexer(str(tmp_path))

        with _patch_rich_progress():
            indexer.index_file(str(py_file))  # first pass: index it
            result = indexer.index_file(str(py_file))  # second pass: same hash

        assert result.get("skipped") is True or result["symbols_updated"] == 0

    def test_index_file_after_content_change_updates(self, tmp_path: pathlib.Path) -> None:
        """After file content changes, index_file() re-indexes and reports symbols_updated."""
        py_file = self._make_py(tmp_path, "def gamma(): pass\n")
        indexer = _make_indexer(str(tmp_path))

        with _patch_rich_progress():
            indexer.index_file(str(py_file))

        # Modify the file
        py_file.write_text("def gamma(): pass\ndef delta(): pass\n", encoding="utf-8")

        with _patch_rich_progress():
            result = indexer.index_file(str(py_file))

        assert result["status"] == "ok"
        # After the update the file should have been processed (not skipped)
        assert not result.get("skipped", False)

    def test_index_file_symbol_in_db_after_indexing(self, tmp_path: pathlib.Path) -> None:
        """After index_file(), the DB should contain at least one symbol for the file."""
        py_file = self._make_py(
            tmp_path,
            "def my_func():\n    '''A function.'''\n    return 42\n",
        )
        indexer = _make_indexer(str(tmp_path))

        with _patch_rich_progress(fake_parser=True):
            indexer.index_file(str(py_file))

        # Verify via the DB directly
        rel = py_file.relative_to(tmp_path)
        conn = indexer.db._conn
        rows = conn.execute(
            "SELECT f.id FROM files f WHERE f.rel_path = ?",
            (str(rel),),
        ).fetchall()
        assert rows, "Expected at least one file row in the DB after index_file()"

        file_id = rows[0][0]
        sym_rows = conn.execute(
            "SELECT COUNT(*) FROM symbols WHERE file_id = ?",
            (file_id,),
        ).fetchone()
        assert sym_rows[0] >= 1, "Expected at least one symbol inserted for the file"


# ---------------------------------------------------------------------------
# Streaming indexing pipeline (Plan C)
# ---------------------------------------------------------------------------


class TestStreamingIndexing:
    """
    Tests for the streaming indexing pipeline (TRELIX_INDEXER_STREAMING=true).

    Streaming mode uses a generator + bounded queue to avoid buffering all
    files in memory before parsing begins.  The public index() API is unchanged.
    """

    def _make_streaming_indexer(self, tmp_dir: str) -> Indexer:  # noqa: F821
        """
        Build an Indexer with streaming_enabled=True and fake embedder / vector store.
        """
        import pathlib

        from trelix.core.config import EmbedderConfig, IndexConfig, IndexerConfig, StoreConfig
        from trelix.indexing.indexer import Indexer

        cfg = IndexConfig(
            repo_path=tmp_dir,
            incremental=False,
            store=StoreConfig(db_path=str(pathlib.Path(tmp_dir) / ".trelix" / "index.db")),
            embedder=EmbedderConfig.model_construct(provider="local"),
            indexer=IndexerConfig(streaming_enabled=True),
        )

        with (
            patch("trelix.indexing.indexer.make_embedder", return_value=_FakeEmbedder()),
            patch("trelix.indexing.indexer.make_vector_store", return_value=_FakeVectorStore()),
            patch("trelix.indexing.indexer.get_parser", side_effect=_fake_get_parser),
        ):
            indexer = Indexer(cfg, quiet=True)

        return indexer

    def test_streaming_mode_produces_same_result_as_batch(self, tmp_path: pathlib.Path) -> None:
        """Streaming pipeline must index the same number of files as batch mode."""
        # Use SEPARATE directories so the two indexers do not share a DB
        batch_dir = tmp_path / "batch_repo"
        batch_dir.mkdir()
        stream_dir = tmp_path / "stream_repo"
        stream_dir.mkdir()

        for d in (batch_dir, stream_dir):
            (d / "a.py").write_text("def foo(): pass", encoding="utf-8")
            (d / "b.py").write_text("def bar(): pass", encoding="utf-8")
            (d / "c.py").write_text("def baz(): pass", encoding="utf-8")

        indexer_batch = _make_indexer(str(batch_dir))
        indexer_stream = self._make_streaming_indexer(str(stream_dir))

        with _patch_rich_progress(fake_parser=True):
            result_batch = indexer_batch.index()

        with _patch_rich_progress(fake_parser=True):
            result_stream = indexer_stream.index()

        batch_count = result_batch.get("files_processed", result_batch.get("files_indexed", -1))
        stream_count = result_stream.get("files_processed", result_stream.get("files_indexed", -2))
        assert batch_count == stream_count, (
            f"Batch files: {batch_count}, Stream files: {stream_count}\n"
            f"Batch: {result_batch}\nStream: {result_stream}"
        )

    def test_streaming_mode_does_not_buffer_all_files_in_memory(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Generator path must yield files one at a time, not collect all first."""
        indexer = self._make_streaming_indexer(str(tmp_path))

        # _iter_files must be a generator (has __next__)
        gen = indexer._iter_files(str(tmp_path))
        assert hasattr(gen, "__next__"), "_iter_files must be a generator"

    def test_streaming_iter_files_is_true_generator_not_list_iter(
        self, tmp_path: pathlib.Path
    ) -> None:
        """_iter_files must be a generator function (GeneratorType), not iter(list(...))."""
        import types

        indexer = self._make_streaming_indexer(str(tmp_path))
        gen = indexer._iter_files(str(tmp_path))
        assert isinstance(gen, types.GeneratorType), (
            "_iter_files must return a GeneratorType so files are yielded lazily. "
            f"Got: {type(gen).__name__}"
        )

    def test_streaming_producer_exception_does_not_hang(self, tmp_path: pathlib.Path) -> None:
        """If _iter_files raises, _index_streaming must complete — not hang forever."""
        import threading
        from unittest.mock import patch

        indexer = self._make_streaming_indexer(str(tmp_path))

        def bad_iter_files(repo_path: str):
            raise RuntimeError("walker exploded")
            yield  # make it a generator

        completed = threading.Event()
        result = {}

        def run():
            with patch.object(indexer, "_iter_files", bad_iter_files):
                result["out"] = indexer._index_streaming(str(tmp_path))
            completed.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        # Must complete within 3 seconds — not hang
        finished = completed.wait(timeout=3.0)
        assert finished, (
            "_index_streaming hung after _iter_files raised. "
            "Producer must use try/finally to guarantee sentinel is enqueued."
        )
        assert result["out"]["errors"] >= 0  # completed with some error count
