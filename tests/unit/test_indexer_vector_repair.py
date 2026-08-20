"""A run killed mid-Phase-3 must heal on the next plain `trelix index`.

The bug, reproduced on a real index: 68,880 chunk rows, 61,652 vectors. 7,228 chunks
(10.5%) had no vector and could never be retrieved; `index_metadata` had 0 rows so
`DimensionGuard.check` silently disarmed too; and re-running the index was a NO-OP —
"Files walked 3,136 / Unchanged since last index 3,136 / Files to embed 0".

The cause is ordering: `_insert_one` commits a file's content hash and its chunk rows in
Phase 2, BEFORE Phase 3 embeds them. Any interruption in between — SIGKILL, laptop sleep,
CI timeout, OOM, quota exhaustion — leaves permanent holes that the incremental skip then
refuses to fill. `PartialIndexError` covers the case where Phase 3 RAISES and is correct;
it cannot cover the case where nothing runs afterwards to raise.

Interruption is simulated by deleting vector rows and `index_metadata` from a fixture
index — no SIGKILL and no paid embeddings. Every test uses a real tree-sitter parser (so
`content_hash` values are genuine), a real sqlite-vec store in the same `index.db`, and a
fake embedder that records what it was asked to embed.

`tests/unit/test_indexer_incremental_symbols.py` is the specification this must not
break: repair is driven by OBSERVED missing vectors, never by widening the
change-detection rule.
"""

from __future__ import annotations

import logging
import pathlib
import sqlite3
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
import sqlite_vec

from trelix.core.config import EmbedderConfig, IndexConfig, StoreConfig

_DIM = 4


class _FakeEmbedder:
    """Records every text passed to embed() so a test can assert on exactly what was —
    and was not — sent. A repair that re-embeds a healthy chunk costs real money."""

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


@contextmanager
def _patch_rich_progress():  # type: ignore[no-untyped-def]
    """Suppress rich progress output during tests."""
    from unittest.mock import MagicMock

    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task = MagicMock(return_value=0)
    mock_progress.advance = MagicMock()
    with patch("trelix.indexing.indexer.Progress", return_value=mock_progress):
        yield mock_progress


def _make_indexer(tmp_dir: str, fake_embedder: _FakeEmbedder, *, streaming: bool = False) -> Any:
    """`test_indexer_incremental_symbols._make_indexer` with exactly two changes.

    1. `incremental=True` — that fixture forces False, which disables the very skip
       under test here.
    2. A REAL `VectorStore` instead of the fake, pointed at the same `index.db`, so
       `chunk_embeddings` lands beside `chunks` and both are countable.

    The real parser is kept (no `get_parser` patch): genuine `content_hash` values are
    the point, since the whole failure mode is a hash that says "already done".
    """
    from trelix.indexing.indexer import Indexer
    from trelix.store.vector import VectorStore

    cfg = IndexConfig(
        repo_path=tmp_dir,
        incremental=True,
        store=StoreConfig(db_path=str(pathlib.Path(tmp_dir) / ".trelix" / "index.db")),
        embedder=EmbedderConfig.model_construct(provider="local"),
    )
    if streaming:
        cfg.indexer.streaming_enabled = True

    real_store = VectorStore(cfg.db_path_absolute, dimension=_DIM)
    with (
        patch("trelix.indexing.indexer.make_embedder", return_value=fake_embedder),
        patch("trelix.indexing.indexer.make_vector_store", return_value=real_store),
    ):
        return Indexer(cfg, quiet=True)


def _raw_conn(db_path: pathlib.Path) -> sqlite3.Connection:
    """A plain connection with sqlite_vec loaded — vec0 is unreachable without it."""
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _simulate_kill_during_embedding(
    db_path: pathlib.Path, chunk_ids: list[int], *, drop_metadata: bool = True
) -> None:
    """Reproduce both halves of the real failure without a SIGKILL and without spending.

    Deleting the vector rows is what a kill between "chunk row committed" and "vector
    written" leaves. Emptying `index_metadata` is what the real index showed (0 rows), and
    it matters independently: `DimensionGuard.check` early-returns when the stored value is
    not an int, so that index had no protection against mixed-width vectors either.
    """
    conn = _raw_conn(db_path)
    try:
        for chunk_id in chunk_ids:
            conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
        if drop_metadata:
            conn.execute("DELETE FROM index_metadata")
        conn.commit()
    finally:
        conn.close()


def _seed_chunk_at_id(db_path: pathlib.Path, chunk_id: int, *, with_vector: bool) -> None:
    """Add one chunk row at an exact id, optionally with its vector already stored.

    Written as raw SQL because `Database.insert_chunk` cannot choose an id — the column is
    `INTEGER PRIMARY KEY AUTOINCREMENT`. Reaching 10,000,000 for real needs ~145 full
    re-chunks of a 68,880-chunk index (Phase 2 deletes and re-inserts the chunk rows of
    every changed symbol, so the counter only climbs), which is a fixture nobody can afford
    to build honestly. Seeding the id is the same state, arrived at cheaply.

    `with_vector=True` is the state that reads as HEALTHY on disk — the chunk row and its
    vector both present — and still counts as a hole to any caller that diffs a flat chunk-id
    set against `stored_chunk_ids()`, because every backend excludes ids at or above the
    offset as sub-chunk sentinels.
    """
    conn = _raw_conn(db_path)
    try:
        symbol_id = int(conn.execute("SELECT id FROM symbols LIMIT 1").fetchone()[0])
        conn.execute(
            "INSERT INTO chunks (id, symbol_id, chunk_text, token_count) VALUES (?, ?, ?, ?)",
            (chunk_id, symbol_id, "def past_the_offset(): return 0", 7),
        )
        if with_vector:
            conn.execute(
                "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, sqlite_vec.serialize_float32([0.1] * _DIM)),
            )
        conn.commit()
    finally:
        conn.close()


def _coverage(db_path: pathlib.Path) -> tuple[set[int], set[int]]:
    """(chunk row ids, real vector ids) read straight from disk, not through the indexer."""
    conn = _raw_conn(db_path)
    try:
        chunks = {int(r[0]) for r in conn.execute("SELECT id FROM chunks")}
        vectors = {
            int(r[0])
            for r in conn.execute(
                "SELECT chunk_id FROM chunk_embeddings WHERE chunk_id > 0 AND chunk_id < 10000000"
            )
        }
    finally:
        conn.close()
    return chunks, vectors


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """Two files, four top-level functions — enough to hole one file and leave another."""
    (tmp_path / "alpha.py").write_text(
        "def a_one():\n    return 1\n\n\ndef a_two():\n    return 2\n"
    )
    (tmp_path / "beta.py").write_text(
        "def b_one():\n    return 3\n\n\ndef b_two():\n    return 4\n"
    )
    return tmp_path


class TestTheKilledRunHeals:
    def test_a_second_plain_index_refills_exactly_the_missing_vectors(
        self, repo: pathlib.Path
    ) -> None:
        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)
        with _patch_rich_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        chunks, vectors = _coverage(db_path)
        assert chunks == vectors and len(chunks) >= 4, "first run did not fully embed"

        holed = sorted(chunks)[:3]
        _simulate_kill_during_embedding(db_path, holed)
        assert _coverage(db_path)[1] == chunks - set(holed)

        embedder2 = _FakeEmbedder()
        indexer2 = _make_indexer(str(repo), embedder2)
        with _patch_rich_progress():
            stats = indexer2.index()

        chunks_after, vectors_after = _coverage(db_path)
        assert vectors_after == chunks_after, "the holes were not refilled"
        assert stats["chunks_reconciled"] == 3
        assert stats["chunks_missing_vectors"] == 3

    def test_it_re_embeds_only_the_holes(self, repo: pathlib.Path) -> None:
        """The cost guarantee. Re-embedding every chunk of the affected files — the
        obvious "force the file back through the parser" fix — costs strictly more."""
        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)
        with _patch_rich_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        chunks, _ = _coverage(db_path)
        assert len(chunks) > 3, "fixture must have more chunks than holes"
        holed = sorted(chunks)[:3]
        _simulate_kill_during_embedding(db_path, holed)

        embedder2 = _FakeEmbedder()
        with _patch_rich_progress():
            _make_indexer(str(repo), embedder2).index()

        assert len(embedder2.embed_call_texts) == 3

    def test_the_all_up_to_date_claim_is_withheld(self, repo: pathlib.Path) -> None:
        """The exact line the real bug printed over a 10.5%-blind index."""
        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)
        with _patch_rich_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        chunks, _ = _coverage(db_path)
        _simulate_kill_during_embedding(db_path, sorted(chunks)[:2])

        embedder2 = _FakeEmbedder()
        indexer2 = _make_indexer(str(repo), embedder2)
        printed: list[str] = []
        indexer2._console.print = lambda *args, **_kw: printed.append(
            " ".join(str(a) for a in args)
        )  # type: ignore[method-assign]
        with _patch_rich_progress():
            stats = indexer2.index()

        assert not any("all files up to date" in line for line in printed), printed
        assert stats["chunks_missing_vectors"] == 2

    def test_the_dimension_record_is_restored(self, repo: pathlib.Path) -> None:
        """The second half of the real damage: with no recorded dimension,
        `DimensionGuard.check` early-returns and the index accepts any vector width."""
        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)
        with _patch_rich_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        chunks, _ = _coverage(db_path)
        _simulate_kill_during_embedding(db_path, sorted(chunks)[:2], drop_metadata=True)

        indexer_mid = _make_indexer(str(repo), _FakeEmbedder())
        assert indexer_mid.db.get_embedding_dimension() is None, "fixture did not disarm the guard"

        embedder2 = _FakeEmbedder()
        indexer2 = _make_indexer(str(repo), embedder2)
        with _patch_rich_progress():
            indexer2.index()

        assert indexer2.db.get_embedding_dimension() == _DIM

    def test_a_repair_only_run_completes_with_no_files_to_parse(self, repo: pathlib.Path) -> None:
        """`to_parse` is empty, so Phase 1 runs over nothing — the path that used to
        `return stats` early, skipping provenance and never setting elapsed_seconds."""
        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)
        with _patch_rich_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        chunks, _ = _coverage(db_path)
        _simulate_kill_during_embedding(db_path, sorted(chunks)[:3])

        embedder2 = _FakeEmbedder()
        indexer2 = _make_indexer(str(repo), embedder2)
        with _patch_rich_progress():
            stats = indexer2.index()

        assert stats["files_indexed"] == 0
        assert stats["files_skipped"] == 2
        assert stats["chunks_embedded"] == 3
        assert stats["errors"] == 0
        assert stats["elapsed_seconds"] >= 0.0

    def test_a_healthy_index_is_still_a_no_op(self, repo: pathlib.Path) -> None:
        """The regression guard against reconciliation becoming a paid re-embed loop.
        Three consecutive runs, nothing holed: only the first may spend."""
        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)
        with _patch_rich_progress():
            indexer.index()
        assert embedder.embed_call_texts, "first run embedded nothing — fixture is broken"

        for _ in range(2):
            again = _FakeEmbedder()
            with _patch_rich_progress():
                stats = _make_indexer(str(repo), again).index()
            assert again.embed_call_texts == []
            assert stats["chunks_reconciled"] == 0
            assert stats["chunks_missing_vectors"] == 0

    def test_a_changed_file_and_a_hole_are_each_serviced_exactly_once(
        self, repo: pathlib.Path
    ) -> None:
        """Both work sources in one run, with no double payment for either."""
        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)
        with _patch_rich_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        conn = _raw_conn(db_path)
        try:
            beta_chunks = [
                int(r[0])
                for r in conn.execute(
                    "SELECT c.id FROM chunks c JOIN symbols s ON c.symbol_id = s.id "
                    "JOIN files f ON s.file_id = f.id WHERE f.rel_path = 'beta.py'"
                )
            ]
        finally:
            conn.close()
        assert beta_chunks

        # Hole one of beta.py's chunks, and separately change a function in alpha.py.
        _simulate_kill_during_embedding(db_path, beta_chunks[:1])
        (repo / "alpha.py").write_text(
            "def a_one():\n    return 111\n\n\ndef a_two():\n    return 2\n"
        )

        embedder2 = _FakeEmbedder()
        indexer2 = _make_indexer(str(repo), embedder2)
        with _patch_rich_progress():
            stats = indexer2.index()

        chunks_after, vectors_after = _coverage(db_path)
        assert vectors_after == chunks_after
        assert stats["chunks_reconciled"] == 1
        # The changed symbol plus the one hole, each embedded once — no duplicates.
        assert len(embedder2.embed_call_texts) == 2
        assert len(set(embedder2.embed_call_texts)) == 2
        assert any("111" in text for text in embedder2.embed_call_texts)

    def test_a_repair_id_whose_chunk_row_phase_2_deleted_is_dropped(
        self, repo: pathlib.Path
    ) -> None:
        """The reason the ids are re-read at service time rather than used directly.

        Hole a chunk, then change the very symbol that owns it. Phase 2 deletes and
        re-inserts that chunk row, so the collected id is dead by service time. Embedding
        it would write a vector with no chunk row — a fresh orphan, the opposite direction
        of the bug being fixed.

        Note this is the ONE test here that also passes against the pre-change code, and
        that is correct rather than a gap: it guards a failure mode the REPAIR could
        introduce, not the original bug. Before the fix nothing is repaired, so no orphan
        can be written. It fails if the re-read filter is ever dropped.
        """
        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)
        with _patch_rich_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        conn = _raw_conn(db_path)
        try:
            a_one_chunks = [
                int(r[0])
                for r in conn.execute(
                    "SELECT c.id FROM chunks c JOIN symbols s ON c.symbol_id = s.id "
                    "WHERE s.qualified_name = 'a_one'"
                )
            ]
        finally:
            conn.close()
        assert a_one_chunks

        _simulate_kill_during_embedding(db_path, a_one_chunks)
        (repo / "alpha.py").write_text(
            "def a_one():\n    return 999\n\n\ndef a_two():\n    return 2\n"
        )

        embedder2 = _FakeEmbedder()
        with _patch_rich_progress():
            _make_indexer(str(repo), embedder2).index()

        chunks_after, vectors_after = _coverage(db_path)
        assert vectors_after - chunks_after == set(), "a vector was written for a dead chunk row"
        assert vectors_after == chunks_after
        for dead_id in a_one_chunks:
            assert dead_id not in vectors_after


class TestChunkIdsPastTheSentinelOffset:
    """A chunk row whose id reached `_SUB_CHUNK_OFFSET` must not be billed as a hole.

    `stored_chunk_ids()` excludes ids at or above the offset on all three backends, because
    that is where sub-chunk vectors live. Diffing a FLAT set of chunk ids against it therefore
    reports every such row as missing a vector forever, and the repair above then re-embeds
    them on every single `trelix index` — for money, on a tree with nothing to do. That is a
    recurring charge, not a one-off, and it is worse than the state it replaced: before the
    repair existed these chunks were merely unretrievable.

    Re-embedding cannot help them either way: `SQLiteVectorStore.search()` filters results
    through `_is_chunk_id`, so a vector at such an id is dropped from every search no matter
    how often it is paid for. They need a re-key, not a re-embed.
    """

    def test_a_chunk_past_the_offset_is_never_billed_across_two_runs(
        self, repo: pathlib.Path
    ) -> None:
        """The exact symptom: run after run of `chunks_reconciled=2` on a healthy tree.

        Two consecutive runs, because one is not enough to distinguish a one-off charge from
        a recurring one — the reproduction showed run3 and run4 each paying for the same two
        chunks.
        """
        from trelix.store.vector import BaseVectorStore

        offset = BaseVectorStore._SUB_CHUNK_OFFSET
        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)
        with _patch_rich_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        chunks, vectors = _coverage(db_path)
        assert chunks == vectors, "first run did not fully embed — fixture is broken"

        # Both rows have their vector on disk, so nothing here is actually damaged.
        _seed_chunk_at_id(db_path, offset, with_vector=True)
        _seed_chunk_at_id(db_path, offset + 1, with_vector=True)

        for run in (1, 2):
            again = _FakeEmbedder()
            with _patch_rich_progress():
                stats = _make_indexer(str(repo), again).index()
            assert again.embed_call_texts == [], f"run {run} paid to re-embed"
            assert stats["chunks_reconciled"] == 0, f"run {run} reconciled"
            assert stats["chunks_missing_vectors"] == 0, f"run {run} saw a hole"

    def test_the_exhaustion_is_reported_at_error_level_naming_the_offset(
        self, repo: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Excluded from the repair, but never silent: it is permanent and needs a re-key."""
        from trelix.store.vector import BaseVectorStore

        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)
        with _patch_rich_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        _seed_chunk_at_id(db_path, BaseVectorStore._SUB_CHUNK_OFFSET, with_vector=True)

        caplog.clear()
        with _patch_rich_progress(), caplog.at_level(logging.ERROR, logger="trelix.indexing"):
            _make_indexer(str(repo), _FakeEmbedder()).index()

        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, [r.getMessage() for r in caplog.records]
        assert any("re-key" in message for message in errors), errors
        assert any(str(BaseVectorStore._SUB_CHUNK_OFFSET) in message for message in errors), errors

    def test_a_real_hole_below_the_offset_is_still_repaired_alongside_one(
        self, repo: pathlib.Path
    ) -> None:
        """The partition must not become a blanket excuse to skip the repair."""
        from trelix.store.vector import BaseVectorStore

        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)
        with _patch_rich_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        chunks, _ = _coverage(db_path)
        holed = sorted(chunks)[:2]
        _simulate_kill_during_embedding(db_path, holed)
        _seed_chunk_at_id(db_path, BaseVectorStore._SUB_CHUNK_OFFSET, with_vector=False)

        embedder2 = _FakeEmbedder()
        with _patch_rich_progress():
            stats = _make_indexer(str(repo), embedder2).index()

        assert stats["chunks_reconciled"] == 2
        assert stats["chunks_missing_vectors"] == 2
        assert len(embedder2.embed_call_texts) == 2
        assert not any("past_the_offset" in text for text in embedder2.embed_call_texts)


class TestTheDimensionStampFollowsTheStore:
    """`index_metadata` must never claim a width the vec0 table on disk contradicts.

    Recording BEFORE Phase 3 is what re-arms the guard on a fresh index whose embed dies, and
    that is worth keeping. Doing it unconditionally is not: `CREATE VIRTUAL TABLE IF NOT
    EXISTS … FLOAT[dim]` cannot widen or narrow a PRE-EXISTING table, so on the crash-recovery
    case this whole file is about, an early stamp records a width the table does not have. The
    next run with the CORRECT provider then dies with `DimensionMismatchError`, whose own
    remedy is `migrate-vectors --reset` — i.e. discard every paid-for embedding.
    """

    def test_a_failed_phase_3_does_not_restamp_a_populated_store(self, repo: pathlib.Path) -> None:
        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)
        with _patch_rich_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        chunks, _ = _coverage(db_path)
        # The state a kill leaves: holes to repair AND no recorded dimension.
        _simulate_kill_during_embedding(db_path, sorted(chunks)[:2], drop_metadata=True)

        class _BrokenEmbedder(_FakeEmbedder):
            def embed(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("401 Unauthorized")

        indexer2 = _make_indexer(str(repo), _BrokenEmbedder())
        assert indexer2.db.get_embedding_dimension() is None, "fixture did not disarm the guard"
        with _patch_rich_progress(), pytest.raises(Exception):
            indexer2.index()

        # Still None: the store already held 61,652-vectors-worth of one width in the real
        # case, and nothing in this failed run learned anything new about it.
        assert indexer2.db.get_embedding_dimension() is None

    def test_a_fresh_store_is_still_stamped_before_the_embed(self, repo: pathlib.Path) -> None:
        """The property the early record exists for, kept: an index that has never stored a
        vector arms the guard even when Phase 3 dies, because there is no width on disk for
        the stamp to contradict and `--reset` would discard nothing."""

        class _BrokenEmbedder(_FakeEmbedder):
            def embed(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("401 Unauthorized")

        indexer = _make_indexer(str(repo), _BrokenEmbedder())
        assert indexer.vector_store.count() == 0, "fixture store is not fresh"
        with _patch_rich_progress(), pytest.raises(Exception):
            indexer.index()

        assert indexer.db.get_embedding_dimension() == _DIM

    def test_a_successful_repair_run_still_re_arms_the_guard(self, repo: pathlib.Path) -> None:
        """The late record covers the case the early one no longer does. This is the real
        Ag-Bash shape: a populated store, no recorded dimension, correct provider."""
        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)
        with _patch_rich_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        chunks, _ = _coverage(db_path)
        _simulate_kill_during_embedding(db_path, sorted(chunks)[:2], drop_metadata=True)

        indexer2 = _make_indexer(str(repo), _FakeEmbedder())
        with _patch_rich_progress():
            indexer2.index()

        assert indexer2.db.get_embedding_dimension() == _DIM


class TestFailureToCheckDoesNotBlockTheRun:
    def test_a_store_that_cannot_enumerate_still_indexes_and_says_so(
        self, repo: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A failed coverage check must never block an index — and must never claim
        health. `chunks_missing_vectors` is None, not 0."""
        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder)

        def _boom() -> set[int]:
            raise NotImplementedError("this backend cannot enumerate")

        indexer.vector_store.stored_chunk_ids = _boom  # type: ignore[method-assign]

        with _patch_rich_progress(), caplog.at_level(logging.WARNING, logger="trelix.indexing"):
            stats = indexer.index()

        assert stats["chunks_missing_vectors"] is None
        assert stats["chunks_reconciled"] == 0
        assert stats["files_indexed"] == 2
        assert stats["chunks_embedded"] >= 4
        assert any(
            "missing vectors" in record.message or "missing vectors" in record.getMessage()
            for record in caplog.records
        ), [r.getMessage() for r in caplog.records]


class TestStreamingPipeline:
    def test_the_streaming_pipeline_heals_too(self, repo: pathlib.Path) -> None:
        """Reconciled once per run at the pipeline level, not once per file: a check
        inside index_file() would run a full store scan per file (64 per streaming batch)
        for an answer that changes once."""
        embedder = _FakeEmbedder()
        indexer = _make_indexer(str(repo), embedder, streaming=True)
        with _patch_rich_progress():
            indexer.index()

        db_path = indexer.config.db_path_absolute
        chunks, vectors = _coverage(db_path)
        assert chunks == vectors and len(chunks) >= 4, "streaming first run did not fully embed"

        holed = sorted(chunks)[:2]
        _simulate_kill_during_embedding(db_path, holed)

        embedder2 = _FakeEmbedder()
        indexer2 = _make_indexer(str(repo), embedder2, streaming=True)
        with _patch_rich_progress():
            stats = indexer2.index()

        chunks_after, vectors_after = _coverage(db_path)
        assert vectors_after == chunks_after
        assert stats["chunks_reconciled"] == 2
        assert stats["chunks_missing_vectors"] == 2
        assert len(embedder2.embed_call_texts) == 2

    def test_a_run_that_had_errors_does_not_repair_what_it_just_failed_to_embed(
        self, repo: pathlib.Path
    ) -> None:
        """This pipeline repairs AFTER the walk, so a file whose own embed just failed is
        sitting in the store as a fresh hole — `index_file` swallowed the failure into
        `status=error`. Repairing it here re-sends the identical text to the provider that
        just refused it: paid twice, failed twice. The holes stay; the next run heals them.
        """
        embedder = _FakeEmbedder()
        calls: list[int] = []

        def _fail_after_first(texts: list[str]) -> list[list[float]]:
            calls.append(len(texts))
            if len(calls) > 1:
                raise RuntimeError("429 rate limited")
            return [[0.1] * _DIM for _ in texts]

        indexer = _make_indexer(str(repo), embedder, streaming=True)
        indexer.embedder.embed = _fail_after_first  # type: ignore[method-assign]

        with _patch_rich_progress():
            stats = indexer.index()

        assert stats["errors"] >= 1, stats
        # The failed file's chunks are holes now, and they are REPORTED …
        assert stats["chunks_missing_vectors"], stats
        # … but nothing was paid to repair them in the same run that just failed on them.
        assert stats["chunks_reconciled"] == 0, stats
        assert len(calls) == 2, calls
