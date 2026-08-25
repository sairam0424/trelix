"""What an index run REPORTS about itself, pinned against the run it actually performed.

Every test here was written against a surviving mutant of
`src/trelix/indexing/indexer.py`, i.e. a one-line change to the write path that the
existing 241-test indexer suite accepted in silence. They fall into three families, and
all three are the same failure: **a number that describes a run nobody verified**.

  * **Provenance** — `index()` calls `_capture_provenance()` before the walk and
    `_record_provenance()` at the end, so `trelix stats` can answer "does this index
    reflect my worktree". Deleting either call, in either pipeline, changed nothing that
    any test read. `tests/unit/test_provenance.py` exercises capture/write/read as a
    module; nothing tied the indexer to it.

  * **Accounting** — `chunks_total`, `chunks_embedded` (via `index_file`'s
    `chunks_updated`), `symbols_extracted` on the all-symbols-unchanged path,
    `files_skipped`, `files_unreadable`, `files_found`, and `chunks_reconciled` could each
    be pinned to 0 (or to a larger wrong number) with the work still being done — or, for
    `chunks_reconciled`, with work merely claimed. `files_unreadable` is the sharpest of
    these: indexer.py's own comment says anything that DELETES rows for files the walk did
    not yield must consult it first, because a truncated walk is indistinguishable from
    deletion.

  * **Fail-safe knobs** — `_env_int` refusing a negative value, `_vector_store_is_empty`
    answering False when the store cannot be counted, and `_report_progress` clamping its
    fraction. Each has a docstring stating the invariant; none had a test.

Conventions, each from a shipped defect in this repo:
  * Expected values are written as literals, never recomputed from indexer.py. Where a
    literal encodes a module constant (`_FULL_RESOLVE_THRESHOLD`, `_PHASE_WEIGHTS`), a
    precondition asserts the constant so a change breaks the test loudly instead of
    silently making it vacuous.
  * Counts are cross-checked against the DB / vector store read with a plain sqlite3
    connection, not against another indexer-reported number.
  * No Mock stands in for the Indexer, the vector store under test, or the walker.
    `rich.Progress` is suppressed with a MagicMock exactly as the rest of the indexer
    suite does — it is a display sink, not a subject.
"""

from __future__ import annotations

import os
import pathlib
import sqlite3
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import sqlite_vec

from trelix.core.config import EmbedderConfig, IndexConfig, StoreConfig
from trelix.core.models import Language

_DIM = 4

# `stored_chunk_ids()` treats ids at or above this as sub-chunk sentinels; the raw SQL
# below filters the same way so a summary/sub-chunk row cannot be mistaken for a chunk
# vector. Written out rather than imported: the point is to read the store the way a
# human would, independently of the constant the indexer uses.
_SUB_CHUNK_OFFSET = 10_000_000


class _RecordingEmbedder:
    """Records every embed() call as its own list, so batch BOUNDARIES are observable.

    A flat list of texts cannot distinguish one call of two texts from two calls of one,
    which is exactly the difference the `_store_summaries` batching tests turn on.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def texts(self) -> list[str]:
        return [t for call in self.calls for t in call]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1] * _DIM for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * _DIM

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    @property
    def dimension(self) -> int:
        return _DIM


class _UncountableStore:
    """A vector store whose `count()` fails — the "unknown, not empty" case.

    A plain class rather than a Mock: the method under test asks this object one
    question, and a Mock would answer it by accident.
    """

    def __init__(self) -> None:
        self.count_calls = 0

    def count(self) -> int:
        self.count_calls += 1
        raise sqlite3.OperationalError("no such table: chunk_embeddings")


@contextmanager
def _quiet_progress():  # type: ignore[no-untyped-def]
    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task = MagicMock(return_value=0)
    mock_progress.advance = MagicMock()
    with patch("trelix.indexing.indexer.Progress", return_value=mock_progress):
        yield mock_progress


def _make_indexer(
    tmp_dir: pathlib.Path,
    embedder: _RecordingEmbedder,
    *,
    streaming: bool = False,
) -> Any:
    """A real Indexer over `tmp_dir`: real walker, real parser, real sqlite-vec store.

    Only the embedder is substituted — everything else is the shipped object, because the
    subject of every test here is what the shipped write path reports.
    """
    from trelix.indexing.indexer import Indexer
    from trelix.store.vector import VectorStore

    cfg = IndexConfig(
        repo_path=str(tmp_dir),
        incremental=True,
        store=StoreConfig(db_path=str(tmp_dir / ".trelix" / "index.db")),
        embedder=EmbedderConfig.model_construct(provider="local"),
    )
    if streaming:
        cfg.indexer.streaming_enabled = True

    real_store = VectorStore(cfg.db_path_absolute, dimension=_DIM)
    with (
        patch("trelix.indexing.indexer.make_embedder", return_value=embedder),
        patch("trelix.indexing.indexer.make_vector_store", return_value=real_store),
    ):
        return Indexer(cfg, quiet=True)


def _conn(db_path: str | pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _chunk_row_count(db_path: str | pathlib.Path) -> int:
    conn = _conn(db_path)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    finally:
        conn.close()


def _chunk_vector_count(db_path: str | pathlib.Path) -> int:
    conn = _conn(db_path)
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM chunk_embeddings WHERE chunk_id > 0 AND chunk_id < ?",
                (_SUB_CHUNK_OFFSET,),
            ).fetchone()[0]
        )
    finally:
        conn.close()


def _chunk_ids_for_symbol(db_path: str | pathlib.Path, qualified_name: str) -> list[int]:
    conn = _conn(db_path)
    try:
        return [
            int(r[0])
            for r in conn.execute(
                "SELECT c.id FROM chunks c JOIN symbols s ON c.symbol_id = s.id "
                "WHERE s.qualified_name = ?",
                (qualified_name,),
            )
        ]
    finally:
        conn.close()


def _delete_vectors(db_path: str | pathlib.Path, chunk_ids: list[int]) -> None:
    """What a SIGKILL between "chunk row committed" and "vector written" leaves behind."""
    conn = _conn(db_path)
    try:
        for chunk_id in chunk_ids:
            conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
        conn.commit()
    finally:
        conn.close()


def _first_file_id(db_path: str | pathlib.Path) -> int:
    conn = _conn(db_path)
    try:
        return int(conn.execute("SELECT id FROM files ORDER BY id LIMIT 1").fetchone()[0])
    finally:
        conn.close()


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """One file, two top-level functions — two symbols, two chunks."""
    (tmp_path / "alpha.py").write_text(
        "def a_one():\n    return 1\n\n\ndef a_two():\n    return 2\n"
    )
    return tmp_path


@pytest.fixture
def repo_with_unreadable_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Two readable files plus one directory the walk cannot enter.

    An unreadable directory drops its WHOLE SUBTREE, which is the failure indexer.py's
    `files_unreadable` comment is about. Restored in teardown so pytest's tmp_path
    cleanup can remove it.
    """
    (tmp_path / "beta.py").write_text("def b_one():\n    return 3\n")
    (tmp_path / "gamma.py").write_text("def g_one():\n    return 4\n")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden.py").write_text("def hidden():\n    return 5\n")
    os.chmod(locked, 0o000)
    try:
        yield tmp_path
    finally:
        os.chmod(locked, 0o755)  # noqa: S103 - restoring 0o755 after a deliberate chmod 0o000 fixture


# ──────────────────────────────────────────────────────────────────────────────
# Provenance
# ──────────────────────────────────────────────────────────────────────────────


class TestProvenanceIsRecordedByEveryPipeline:
    """`trelix stats` cannot report drift against a record no run wrote.

    Four separate one-line mutations all produced an index with empty provenance and a
    fully green suite: dropping `self._record_provenance()` from `index()`, dropping it
    from `_index_streaming()`, making `_record_provenance` read its snapshot as None, and
    dropping `self._capture_provenance()` from the top of `index()`.
    """

    def test_the_batch_pipeline_writes_a_provenance_row(self, repo: pathlib.Path) -> None:
        """Fails when `index()` stops calling `_record_provenance` / `_capture_provenance`."""
        from trelix.store.provenance import read_provenance

        indexer = _make_indexer(repo, _RecordingEmbedder())
        with _quiet_progress():
            stats = indexer.index()

        # Precondition, naming the `repo` fixture: a run that indexed nothing would make
        # the assertion below uninteresting.
        assert stats["files_indexed"] == 1, (
            "the `repo` fixture no longer produces exactly one indexed file; "
            f"got {stats['files_indexed']}"
        )

        provenance = read_provenance(indexer.db)
        assert not provenance.is_empty, "index() left no provenance record"
        assert provenance.indexed_at is not None, "provenance recorded no timestamp"
        # Literal, not `indexer.config.embedder.provider`: the value the fixture asked for.
        assert provenance.embedder_provider == "local"

    def test_the_streaming_pipeline_writes_a_provenance_row(self, repo: pathlib.Path) -> None:
        """Fails when `_index_streaming()` stops calling `_record_provenance`."""
        from trelix.store.provenance import read_provenance

        indexer = _make_indexer(repo, _RecordingEmbedder(), streaming=True)
        with _quiet_progress():
            results = indexer.index()

        # Precondition: prove the streaming pipeline really ran. The batch pipeline
        # reports `file_summaries_generated`; the streaming one never does.
        assert "file_summaries_generated" not in results, (
            "this run took the batch pipeline, so it proves nothing about streaming"
        )
        assert results["files_indexed"] == 1

        provenance = read_provenance(indexer.db)
        assert not provenance.is_empty, "the streaming pipeline left no provenance record"
        assert provenance.embedder_provider == "local"

    def test_provenance_is_captured_before_the_walk(self, repo: pathlib.Path) -> None:
        """Fails when `_capture_provenance()` moves after (or off) the walk.

        The ordering is the whole point of the docstring on `_capture_provenance`: file
        hashes are computed during the walk, so an end-of-run capture would pair a NEWER
        commit with older content and report the index as more current than it is. Only
        the direction is testable without a git fixture, and the direction is what
        matters.
        """
        order: list[str] = []
        indexer = _make_indexer(repo, _RecordingEmbedder())

        real_capture = indexer._capture_provenance
        real_walk = indexer.walker.walk

        def recording_capture() -> None:
            order.append("capture")
            real_capture()

        def recording_walk():  # type: ignore[no-untyped-def]
            order.append("walk")
            yield from real_walk()

        indexer._capture_provenance = recording_capture  # type: ignore[method-assign]
        indexer.walker.walk = recording_walk  # type: ignore[method-assign]

        with _quiet_progress():
            indexer.index()

        assert order[:2] == ["capture", "walk"], (
            f"provenance was not captured before the walk began: {order}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Accounting
# ──────────────────────────────────────────────────────────────────────────────


class TestTheReportedCountsMatchWhatLanded:
    """Each stat is checked against the store, never against another reported stat."""

    def test_chunks_total_counts_the_chunk_rows_a_fresh_index_created(
        self, repo: pathlib.Path
    ) -> None:
        """Fails when `stats["chunks_total"] = len(pending)` is pinned to 0 (or dropped)."""
        indexer = _make_indexer(repo, _RecordingEmbedder())
        with _quiet_progress():
            stats = indexer.index()

        rows = _chunk_row_count(indexer.config.db_path_absolute)
        # Precondition naming the `repo` fixture: on an empty repo every count is 0 and
        # the equality below holds by construction.
        assert rows == 2, f"the `repo` fixture no longer yields exactly 2 chunk rows: {rows}"
        assert stats["chunks_total"] == 2

    def test_index_file_reports_the_chunks_it_actually_embedded(self, repo: pathlib.Path) -> None:
        """Fails when the SYNC Phase 3 stops accumulating `stats["chunks_embedded"]`.

        `index_file()` returns that counter as `chunks_updated`, and `_index_streaming`
        folds `chunks_updated` into its own `chunks_embedded` / `chunks_total`. So a
        sync-path counter pinned to 0 makes both the watch path and the streaming pipeline
        report "indexed nothing" over vectors they paid for and wrote.
        """
        indexer = _make_indexer(repo, _RecordingEmbedder())
        with _quiet_progress():
            result = indexer.index_file(str(repo / "alpha.py"))

        assert result["status"] == "ok"
        vectors = _chunk_vector_count(indexer.config.db_path_absolute)
        # Precondition: without vectors on disk, `chunks_updated == 0` would be correct.
        assert vectors == 2, f"the `repo` fixture no longer lands exactly 2 vectors: {vectors}"
        assert result["chunks_updated"] == 2

    def test_an_all_symbols_unchanged_reindex_still_reports_the_files_symbols(
        self, repo: pathlib.Path
    ) -> None:
        """Fails when the `not changed_or_new_symbols` early return stops adding
        `len(all_symbols)` to `symbols_extracted`.

        The state is reachable and ordinary: appending a trailing newline changes the
        file's content hash — so the incremental pre-filter does NOT skip it — while every
        symbol's `signature + body` is byte-identical, so `_insert_one` returns before the
        chunker. The file is genuinely indexed and genuinely has two symbols; reporting 0
        makes a re-index look like it lost them.
        """
        embedder_first = _RecordingEmbedder()
        with _quiet_progress():
            _make_indexer(repo, embedder_first).index()

        (repo / "alpha.py").write_text(
            "def a_one():\n    return 1\n\n\ndef a_two():\n    return 2\n\n"
        )

        embedder_second = _RecordingEmbedder()
        indexer = _make_indexer(repo, embedder_second)
        with _quiet_progress():
            stats = indexer.index()

        # Preconditions that pin the path taken, naming the fixture edit above. If a
        # trailing newline ever starts changing a symbol body, or starts leaving the file
        # hash alone, these fail instead of the test silently testing the main path.
        assert stats["files_skipped"] == 0, (
            "the trailing-newline edit no longer changes the file hash, so the "
            "pre-filter skipped the file and the unchanged-symbol path never ran"
        )
        assert embedder_second.texts == [], (
            "something was re-embedded, so this run did not take the "
            "all-symbols-unchanged early return"
        )
        assert stats["chunks_total"] == 0

        assert stats["files_indexed"] == 1
        assert stats["symbols_extracted"] == 2

    def test_a_file_with_no_parser_is_counted_as_skipped(self, tmp_path: pathlib.Path) -> None:
        """Fails when Phase 2's `stats["files_skipped"] += 1` stops accumulating.

        Two files are walked and one is indexable, so `files_found - files_indexed` is 1.
        A `files_skipped` of 0 beside that is the "reported success while doing nothing"
        shape: it claims the whole walk was serviced.
        """
        (tmp_path / "a.py").write_text("def one():\n    return 1\n")
        (tmp_path / "opaque.xyz").write_text("nothing structural here\n")

        indexer = _make_indexer(tmp_path, _RecordingEmbedder())
        # UNKNOWN is not in the shipped `walker.languages`, so the walk would drop the
        # .xyz file before any parser was consulted. Allowing it is what puts a
        # parser-less file into Phase 1.
        indexer.config.walker.languages = [Language.PYTHON, Language.UNKNOWN]

        with _quiet_progress():
            stats = indexer.index()

        # Precondition naming the fixture: the .xyz file must actually reach the walk.
        assert stats["files_found"] == 2, (
            f"the walk did not yield both fixture files: {stats['files_found']}"
        )
        assert stats["files_indexed"] == 1
        assert stats["errors"] == 0, "the parser-less file must be skipped, not errored"
        assert stats["files_skipped"] == 1


class TestTheWalkTruncationIsDisclosedByBothPipelines:
    """An unreadable directory drops its subtree, and both pipelines must say so.

    indexer.py: "Anything that later DELETES rows for files the walk did not yield must
    consult this first — a truncated walk is indistinguishable from files having been
    removed, and acting on it destroys paid-for embeddings." Pinning either pipeline's
    `files_unreadable` to 0 left the suite green.
    """

    def test_the_batch_pipeline_reports_the_unreadable_path(
        self, repo_with_unreadable_dir: pathlib.Path
    ) -> None:
        """Fails when `index()` stops deriving `files_unreadable` from the walker."""
        indexer = _make_indexer(repo_with_unreadable_dir, _RecordingEmbedder())
        with _quiet_progress():
            stats = indexer.index()

        # Precondition naming the `repo_with_unreadable_dir` fixture: if chmod 0o000 ever
        # stops blocking the walk (e.g. the suite is run as root), the walker records
        # nothing and the assertion below would pass for the wrong reason.
        assert indexer.walker.incomplete_paths == ["locked"], (
            "the chmod 0o000 fixture no longer truncates the walk: "
            f"{indexer.walker.incomplete_paths}"
        )
        assert stats["files_unreadable"] == 1
        assert stats["files_found"] == 2

    def test_the_streaming_pipeline_reports_the_unreadable_path(
        self, repo_with_unreadable_dir: pathlib.Path
    ) -> None:
        """Fails when `_index_streaming()` stops deriving `files_unreadable`.

        Sharper here than in the batch pipeline: streaming never materialises a file
        list, so this stat is the only observable evidence that the walk was truncated.
        """
        indexer = _make_indexer(repo_with_unreadable_dir, _RecordingEmbedder(), streaming=True)
        with _quiet_progress():
            results = indexer.index()

        assert indexer.walker.incomplete_paths == ["locked"], (
            "the chmod 0o000 fixture no longer truncates the walk: "
            f"{indexer.walker.incomplete_paths}"
        )
        assert results["files_unreadable"] == 1

    def test_the_streaming_pipeline_counts_every_file_it_walked(
        self, repo_with_unreadable_dir: pathlib.Path
    ) -> None:
        """Fails when the streaming consumer stops accumulating `results["files_found"]`."""
        indexer = _make_indexer(repo_with_unreadable_dir, _RecordingEmbedder(), streaming=True)
        with _quiet_progress():
            results = indexer.index()

        # Explicit table, both directions, rather than iterating what the run produced.
        assert {p.name for p in repo_with_unreadable_dir.glob("*.py")} == {
            "beta.py",
            "gamma.py",
        }, "the fixture's readable file set changed"
        assert results["files_found"] == 2
        assert results["files_indexed"] == 2

    def test_the_streaming_pipeline_counts_the_unchanged_files_it_skipped(
        self, repo_with_unreadable_dir: pathlib.Path
    ) -> None:
        """Fails when the streaming incremental pre-filter stops accumulating
        `results["files_skipped"]`.

        A second streaming run over an untouched tree must report 2 skipped, not 0 —
        `files_found == 2` with `files_indexed == 0` and `files_skipped == 0` reads as
        "two files vanished", which is the disclosure this stat exists to give.
        """
        with _quiet_progress():
            _make_indexer(repo_with_unreadable_dir, _RecordingEmbedder(), streaming=True).index()

        second = _make_indexer(repo_with_unreadable_dir, _RecordingEmbedder(), streaming=True)
        with _quiet_progress():
            results = second.index()

        assert results["files_indexed"] == 0, (
            "the second run re-indexed something, so nothing was skipped to count"
        )
        assert results["files_found"] == 2
        assert results["files_skipped"] == 2


class TestTheRepairCountsOnlyWhatItRepaired:
    """`chunks_reconciled` is a claim about spending, in both directions."""

    def test_chunks_reconciled_excludes_a_repair_id_phase_2_deleted(
        self, repo: pathlib.Path
    ) -> None:
        """Fails when `chunks_reconciled` is taken from `repair_ids` instead of `repaired`.

        Hole `a_one`'s chunk, then change `a_one`'s body. Phase 2 deletes and re-inserts
        that chunk row, so by service time the collected id is dead and
        `get_chunk_text_and_tokens` returns nothing for it — nothing is repaired, and the
        chunk is covered as ordinary new work instead. Counting `repair_ids` here reports
        a repair that did not happen, on the one run where a user is checking whether
        their partial index healed.
        """
        indexer = _make_indexer(repo, _RecordingEmbedder())
        with _quiet_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        holed = _chunk_ids_for_symbol(db_path, "a_one")
        assert holed == [1], (
            f"the `repo` fixture no longer puts a_one's only chunk at id 1: {holed}"
        )
        _delete_vectors(db_path, holed)

        (repo / "alpha.py").write_text(
            "def a_one():\n    return 999\n\n\ndef a_two():\n    return 2\n"
        )

        second = _make_indexer(repo, _RecordingEmbedder())
        with _quiet_progress():
            stats = second.index()

        # Precondition: the scan must still SEE the hole, otherwise `chunks_reconciled`
        # would be 0 for the trivial reason.
        assert stats["chunks_missing_vectors"] == 1, (
            "the fixture no longer presents exactly one hole to the reconcile: "
            f"{stats['chunks_missing_vectors']}"
        )
        assert stats["chunks_reconciled"] == 0

    def test_the_streaming_chunks_total_includes_the_repaired_chunks(
        self, repo: pathlib.Path
    ) -> None:
        """Fails when `_index_streaming` stops adding `len(repaired)` to `chunks_total`.

        A repair-only streaming run walks two unchanged files and embeds one chunk. A
        `chunks_total` of 0 next to `chunks_embedded` of 1 is not merely inconsistent —
        it is the pipeline reporting that it did no work on the run whose entire purpose
        was to undo a partial index.
        """
        indexer = _make_indexer(repo, _RecordingEmbedder(), streaming=True)
        with _quiet_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        holed = _chunk_ids_for_symbol(db_path, "a_one")
        assert holed == [1], (
            f"the `repo` fixture no longer puts a_one's only chunk at id 1: {holed}"
        )
        _delete_vectors(db_path, holed)

        embedder = _RecordingEmbedder()
        second = _make_indexer(repo, embedder, streaming=True)
        with _quiet_progress():
            results = second.index()

        # Preconditions: exactly one chunk was re-embedded, and no file was re-indexed,
        # so every unit of `chunks_total` below must have come from the repair.
        assert len(embedder.texts) == 1, f"expected one repair embed, got {embedder.texts}"
        assert results["files_indexed"] == 0
        assert results["chunks_reconciled"] == 1
        assert results["chunks_total"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# Batch-size handling on the summary embed path
# ──────────────────────────────────────────────────────────────────────────────


class TestSummaryEmbedBatchBoundary:
    """`_store_summaries` groups summaries under `embed_max_tokens_per_batch`.

    Two mutations survived: flipping the boundary to `>=` (splits a batch that exactly
    fits, one extra request per boundary) and never flushing early at all (one unbounded
    request, i.e. the request-size error the batching exists to prevent). Neither changed
    any existing assertion, because the only test of this path asserts "fewer calls than
    files" — true of one giant call too.

    `count_tokens` is replaced with `len(str.split())` so the token counts are stated by
    the test rather than derived from a tokenizer. Both cases below use the SAME two
    summaries, 3 and 4 tokens; only the limit moves.
    """

    @staticmethod
    def _run(repo: pathlib.Path, max_tokens: int) -> _RecordingEmbedder:
        from trelix.indexing.indexer import _PendingSummary

        embedder = _RecordingEmbedder()
        indexer = _make_indexer(repo, embedder)
        with _quiet_progress():
            indexer.index()
        file_id = _first_file_id(indexer.config.db_path_absolute)

        indexer.chunker.count_tokens = lambda text: len(text.split())  # type: ignore[method-assign]
        indexer.config.embedder.embed_max_tokens_per_batch = max_tokens

        embedder.calls.clear()
        indexer._store_summaries(
            [
                (_PendingSummary(file_id=file_id, rel_path="alpha.py"), "one two three"),
                (_PendingSummary(file_id=file_id, rel_path="alpha.py"), "four five six seven"),
            ],
            {},
        )
        return embedder

    def test_two_summaries_summing_exactly_to_the_limit_go_in_one_call(
        self, repo: pathlib.Path
    ) -> None:
        """Fails when the flush boundary becomes `>=` — 3 + 4 == 7 must still fit in 7."""
        embedder = self._run(repo, 7)

        assert embedder.calls == [["one two three", "four five six seven"]]

    def test_two_summaries_over_the_limit_are_split(self, repo: pathlib.Path) -> None:
        """Fails when the early flush is removed — 3 + 4 == 7 exceeds 6 and must split."""
        embedder = self._run(repo, 6)

        assert embedder.calls == [["one two three"], ["four five six seven"]]


# ──────────────────────────────────────────────────────────────────────────────
# Fail-safe knobs
# ──────────────────────────────────────────────────────────────────────────────


class TestPerformanceKnobsFailSafe:
    """`_env_int`'s docstring: a typo "must not silently read as 0 (= unlimited)".

    Deleting the negative check left every test passing, and a negative
    `TRELIX_FILE_SUMMARY_RPM` disarms `_RpmRateLimiter` outright (`rpm_limit <= 0` →
    no waiting), i.e. turns a mistyped throttle into an un-throttled fan-out at five
    chat backends whose RPM ceilings trelix does not know.
    """

    def test_a_negative_value_falls_back_to_the_supplied_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fails when `_env_int`'s `if value < 0` guard is removed."""
        from trelix.indexing.indexer import _env_int

        monkeypatch.setenv("TRELIX_TEST_KNOB", "-5")

        # 7 is this test's own number, not the module's default for anything.
        assert _env_int("TRELIX_TEST_KNOB", 7) == 7

    def test_an_unparseable_value_falls_back_to_the_supplied_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fails when the ValueError fallback stops returning `default`."""
        from trelix.indexing.indexer import _env_int

        monkeypatch.setenv("TRELIX_TEST_KNOB", "four")

        assert _env_int("TRELIX_TEST_KNOB", 7) == 7

    def test_a_negative_rpm_env_does_not_leave_the_chat_limiter_unlimited(
        self, repo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The consequence, stated without pinning the shipped default's value.

        Fails when the negative guard is removed: `_summary_rpm` becomes -5, and
        `_RpmRateLimiter.acquire` returns immediately for any limit <= 0.
        """
        monkeypatch.setenv("TRELIX_FILE_SUMMARY_RPM", "-5")

        indexer = _make_indexer(repo, _RecordingEmbedder())

        assert indexer._summary_rpm > 0, (
            "a negative TRELIX_FILE_SUMMARY_RPM was accepted, which makes the Phase 2.5 "
            "chat limiter unlimited"
        )


class TestUnknownStoreStateIsNotTreatedAsEmpty:
    def test_a_store_that_cannot_be_counted_is_not_empty(self, repo: pathlib.Path) -> None:
        """Fails when `_vector_store_is_empty`'s except branch returns True.

        Its docstring: "A store that cannot be counted answers False, not True.
        Everything gated on this becomes more cautious when the answer is unknown;
        guessing 'empty' would stamp a dimension over an index that may already hold
        vectors of another width." The only consumer is Phase 3's decision to record the
        embedding dimension BEFORE the embed, which over a populated store is what locks
        the correct provider out of its own index.
        """
        indexer = _make_indexer(repo, _RecordingEmbedder())
        broken = _UncountableStore()
        indexer.vector_store = broken

        assert indexer._vector_store_is_empty() is False
        # Precondition: the answer must come from a failed count, not from a short-circuit
        # that never asked.
        assert broken.count_calls == 1


class TestTheResolveThresholdIsInclusive:
    def test_a_batch_exactly_at_the_threshold_runs_the_resolve(self, repo: pathlib.Path) -> None:
        """Fails when `files_in_batch >= threshold` becomes `>`.

        At exactly the threshold the four O(N) resolve passes must fire — that boundary
        is the documented contract of the `files_in_batch` argument, and a watch batch
        landing exactly on it would otherwise silently keep stale cross-file edges.
        """
        from trelix.indexing.indexer import Indexer

        # Precondition, so the literals below cannot go stale silently.
        assert Indexer._FULL_RESOLVE_THRESHOLD == 5, (
            "the resolve threshold moved; update the literals in this test"
        )

        indexer = _make_indexer(repo, _RecordingEmbedder())
        ran: list[str] = []
        for name in (
            "resolve_cross_file_calls",
            "resolve_import_file_ids",
            "resolve_cross_file_type_edges",
            "resolve_angular_selectors",
        ):
            real = getattr(indexer.db, name)

            def recorder(_real=real, _name=name):  # type: ignore[no-untyped-def]
                ran.append(_name)
                return _real()

            setattr(indexer.db, name, recorder)

        with _quiet_progress():
            at_threshold = indexer.index_file(str(repo / "alpha.py"), files_in_batch=5)

        assert at_threshold["status"] == "ok"
        assert set(ran) == {
            "resolve_cross_file_calls",
            "resolve_import_file_ids",
            "resolve_cross_file_type_edges",
            "resolve_angular_selectors",
        }

    def test_a_batch_below_the_threshold_skips_the_resolve(self, repo: pathlib.Path) -> None:
        """The other side of the boundary, so the test above cannot pass vacuously."""
        indexer = _make_indexer(repo, _RecordingEmbedder())
        ran: list[str] = []
        real = indexer.db.resolve_cross_file_calls

        def recorder():  # type: ignore[no-untyped-def]
            ran.append("resolve_cross_file_calls")
            return real()

        indexer.db.resolve_cross_file_calls = recorder  # type: ignore[method-assign]

        with _quiet_progress():
            indexer.index_file(str(repo / "alpha.py"), files_in_batch=4)

        assert ran == []


class TestProgressFractionIsClamped:
    """`_report_progress` maps a per-phase fraction onto a global 0→1 bar.

    Dropping the `min(max(...))` clamp survived: no test read the reported value at all,
    let alone out of range. A phase that reports 1.5 would then hand a caller a progress
    value past its phase ceiling — and past 1.0 in the last phase.
    """

    @staticmethod
    def _reported(phase: int, fraction: float) -> float:
        from trelix.indexing.indexer import Indexer

        seen: list[dict[str, Any]] = []
        indexer = Indexer.__new__(Indexer)  # no DB, no embedder — this method needs neither
        indexer._progress_cb = seen.append  # type: ignore[attr-defined]
        indexer._report_progress(phase, "label", fraction, {})
        assert len(seen) == 1
        return float(seen[0]["progress"])

    def test_an_over_unity_fraction_stops_at_the_phase_ceiling(self) -> None:
        """Fails when the clamp is removed: 0.50 + 0.45 * 1.5 = 1.175."""
        from trelix.indexing.indexer import Indexer

        assert Indexer._PHASE_WEIGHTS[3] == (0.50, 0.95), (
            "the phase-3 weight window moved; update the literals in this test"
        )

        assert self._reported(3, 1.5) == 0.95

    def test_a_negative_fraction_stops_at_the_phase_floor(self) -> None:
        """Fails when the clamp is removed: 0.50 + 0.45 * -2.0 = -0.4."""
        assert self._reported(3, -2.0) == 0.5
