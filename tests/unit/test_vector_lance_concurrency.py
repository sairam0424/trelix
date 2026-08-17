"""
Concurrency regression tests for LanceVectorStore.upsert_batch.

upsert_batch replaces rows with a non-atomic delete-then-add pair. LanceDB has
no chunk_id uniqueness constraint, so two writers that interleave as
delete(5) / delete(5) / add(5) / add(5) both succeed and leave TWO rows for
chunk_id 5 — no exception anywhere. Measured on lancedb 0.33.0 with 4 threads
doing 60 batches of 32 upserts over 50 chunk_ids: 50-53 rows where 50 were
expected, and instrumenting table.delete confirmed zero exceptions. That is a
different failure from the swallowed-delete bug covered by
test_vector_lance_upsert.py: here every call reports success.

A second, separate defect compounds it: a table handle pins the version it
opened at, so a handle opened before another handle wrote sees none of that
write. Its delete then matches nothing and its add appends a duplicate — with no
concurrency at all. TestStaleSnapshot covers that half and needs no threads.

Scope, measured rather than assumed: indexing/indexer.py does drive upsert_batch
through a ThreadPoolExecutor(max_workers=2) (indexer.py:1273/1300), but its
chunk_ids come from SQLite autoincrement and _make_token_batches partitions
them, so concurrent batches never name the same chunk_id — 2560 ids over 2
threads produced zero duplicates and zero lost rows pre-fix, six runs
(TestIndexerBatchShape). What is exposed is any two writers that do share a
chunk_id: two handles in one process, or two processes on one URI, which is the
"multi-repo deployments sharing a vector store" case in the store's own
docstring.

Duplicates are not permanent — a later clean upsert of the same chunk_id
collapses them (verified: 2 rows -> 1). But nothing re-upserts a chunk within a
run, so they outlive it: search() then spends N of its k slots on one chunk and
count() drifts above the SQLite chunks table until a reindex.

The interleave is timing-dependent (0-2 duplicate rows per natural run), so
these tests force it: an injected barrier or sleep between delete and add makes
the pre-fix duplication deterministic rather than a flaky ~1-in-3.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter

import pytest

lancedb = pytest.importorskip("lancedb", reason="lance extra not installed")

VEC = [0.1, 0.2, 0.3, 0.4]
# Long enough that a serialising fix always trips it, short enough that the
# suite does not pay for it twice.
BARRIER_TIMEOUT_S = 0.75


def _store(uri, dimension: int = 4):
    from trelix.store.vector_lance import LanceVectorStore

    return LanceVectorStore(uri=str(uri), table_name="chunks", dimension=dimension)


def _all_rows(store) -> list[int]:
    """Every chunk_id in the table, duplicates included (empty query = full scan)."""
    return [row["chunk_id"] for row in store._table.search().limit(1_000_000).to_list()]


def _duplicate_rows(store) -> int:
    counts = Counter(_all_rows(store))
    return sum(n - 1 for n in counts.values() if n > 1)


def _stagger_delete(store, gate) -> None:
    """
    Hold each writer between its delete and its add.

    `gate` is called after the real delete lands. With a Barrier this parks
    writer A until writer B has also deleted, which is the exact interleave that
    duplicates a row. Once writes are serialised the barrier can never fill, so
    it times out — BrokenBarrierError here is the fix working, not a failure.
    """
    original = store._table.delete

    def delete_then_wait(predicate: str) -> None:
        original(predicate)
        try:
            gate()
        except threading.BrokenBarrierError:
            pass

    store._table.delete = delete_then_wait


def _run(threads: list[threading.Thread]) -> None:
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not [t for t in threads if t.is_alive()], "writer thread hung — deadlocked lock?"


class TestConcurrentUpsertSharedHandle:
    """One store handle, two threads — the executor's shape, same chunk_id."""

    def test_forced_interleave_does_not_duplicate_a_chunk_id(self, tmp_path) -> None:
        store = _store(tmp_path / "lance")
        store.upsert_batch([(5, VEC)])
        assert store.count() == 1

        barrier = threading.Barrier(2, timeout=BARRIER_TIMEOUT_S)
        _stagger_delete(store, barrier.wait)

        errors: list[BaseException] = []

        def writer(salt: float) -> None:
            try:
                store.upsert_batch([(5, [salt] * 4)])
            except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
                errors.append(exc)

        _run([threading.Thread(target=writer, args=(s,)) for s in (0.5, 0.9)])

        assert not errors, f"upsert raised under concurrency: {errors}"
        # Pre-fix: 2. Both deletes ran against a table that still had one row,
        # both adds then landed, and neither call reported anything wrong.
        assert _all_rows(store) == [5]

    def test_stress_never_opens_two_write_windows_at_once(self, tmp_path) -> None:
        """
        Natural contention, amplified by a 2 ms gap in the delete->add window.

        Asserted on window overlap rather than the final row count, because the
        final count hides the bug: a duplicate is erased by any later clean
        upsert of the same chunk_id (verified: 2 rows -> 1 after one), so
        repeated rounds over the same ids self-heal and only the last interleave
        survives. What ships is worse, not better — the indexer embeds each
        chunk once, so nothing comes along to repair it.
        """
        store = _store(tmp_path / "lance")
        ids = list(range(8))
        store.upsert_batch([(i, VEC) for i in ids])

        open_windows = 0
        peak_windows = 0
        guard = threading.Lock()
        real_delete, real_add = store._table.delete, store._table.add

        def delete(predicate: str) -> None:
            nonlocal open_windows, peak_windows
            real_delete(predicate)
            with guard:
                open_windows += 1
                peak_windows = max(peak_windows, open_windows)
            time.sleep(0.002)

        def add(data) -> None:
            nonlocal open_windows
            real_add(data)
            with guard:
                open_windows -= 1

        store._table.delete, store._table.add = delete, add

        def writer(worker: int) -> None:
            for round_ in range(6):
                store.upsert_batch([(i, [0.1 * worker, 0.2, 0.3, 0.1 * round_]) for i in ids])

        _run([threading.Thread(target=writer, args=(w,)) for w in range(4)])

        assert peak_windows == 1, f"{peak_windows} writers were mid-upsert at once"
        assert _duplicate_rows(store) == 0
        assert sorted(set(_all_rows(store))) == ids
        assert store.count() == len(ids)

    def test_delete_and_add_are_never_interleaved(self, tmp_path) -> None:
        """
        Direct check of the invariant, not just its symptom.

        A duplicate needs two deletes before an add, so the write log must read
        del,add,del,add — never del,del,add,add.
        """
        store = _store(tmp_path / "lance")
        log: list[str] = []
        log_guard = threading.Lock()
        real_delete, real_add = store._table.delete, store._table.add
        barrier = threading.Barrier(2, timeout=BARRIER_TIMEOUT_S)

        def delete(predicate: str) -> None:
            real_delete(predicate)
            with log_guard:
                log.append("del")
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass

        def add(data) -> None:
            real_add(data)
            with log_guard:
                log.append("add")

        store._table.delete, store._table.add = delete, add

        _run([threading.Thread(target=store.upsert_batch, args=([(9, VEC)],)) for _ in range(2)])

        assert log == ["del", "add", "del", "add"], f"writes interleaved: {log}"


class TestIndexerBatchShape:
    """
    The shape indexing/indexer.py really produces.

    Its chunk_ids come from SQLite autoincrement and _make_token_batches
    partitions the list, so two concurrent batches never share a chunk_id.
    Measured pre-fix over 2560 ids, 2 threads, 80 batches of 32, three runs each
    on a fresh and on a pre-populated table: zero duplicates and zero lost rows.
    So the executor path was not the one duplicating; this test pins that, and
    fails if batching ever starts repeating a chunk_id across batches.
    """

    def test_disjoint_batches_lose_nothing_and_duplicate_nothing(self, tmp_path) -> None:
        store = _store(tmp_path / "lance")
        per_thread, batch, workers = 8, 16, 2
        total = per_thread * batch * workers
        store.upsert_batch([(i, VEC) for i in range(total)])  # reindex, not first pass

        def writer(worker: int) -> None:
            for b in range(per_thread):
                start = (worker * per_thread + b) * batch
                store.upsert_batch([(start + i, [0.5] * 4) for i in range(batch)])

        _run([threading.Thread(target=writer, args=(w,)) for w in range(workers)])

        rows = _all_rows(store)
        assert sorted(rows) == list(range(total))


class TestConcurrentUpsertSeparateHandles:
    """Two LanceVectorStore objects on one URI — the MCP-server-plus-CLI shape."""

    def test_forced_interleave_across_handles_does_not_duplicate(self, tmp_path) -> None:
        uri = tmp_path / "lance"
        stores = [_store(uri), _store(uri)]
        stores[0].upsert_batch([(3, VEC)])

        barrier = threading.Barrier(len(stores), timeout=BARRIER_TIMEOUT_S)
        for store in stores:
            _stagger_delete(store, barrier.wait)

        _run(
            [
                threading.Thread(target=s.upsert_batch, args=([(3, [0.1 * i] * 4)],))
                for i, s in enumerate(stores)
            ]
        )

        assert _all_rows(_store(uri)) == [3]


class TestStaleSnapshot:
    """
    The second, independent half of the bug — and it needs no threads.

    A handle pins the table version it opened at and never advances on another
    handle's writes. Measured on lancedb 0.33.0: after handle B upserts one row,
    A.count() returns 0 against a true 1, and A.delete("chunk_id IN (3)")
    commits a new version while removing nothing. So A's next upsert of that
    chunk_id deletes nothing and adds a second row — serialising the writers
    does not help, because there is no race here at all.
    """

    def test_upsert_from_an_older_handle_replaces_rather_than_appends(self, tmp_path) -> None:
        uri = tmp_path / "lance"
        older = _store(uri)  # opened before the write below
        _store(uri).upsert_batch([(3, VEC)])

        older.upsert_batch([(3, [0.9] * 4)])

        assert _all_rows(_store(uri)) == [3]  # pre-fix: [3, 3]

    def test_count_sees_rows_written_by_another_handle(self, tmp_path) -> None:
        uri = tmp_path / "lance"
        older = _store(uri)
        _store(uri).upsert_batch([(1, VEC), (2, VEC)])

        assert older.count() == 2  # pre-fix: 0

    def test_search_sees_rows_written_by_another_handle(self, tmp_path) -> None:
        uri = tmp_path / "lance"
        older = _store(uri)
        _store(uri).upsert_batch([(1, VEC)])

        assert older.search(VEC, k=5) and older.search(VEC, k=5)[0][0] == 1  # pre-fix: []

    def test_delete_batch_from_an_older_handle_removes_the_row(self, tmp_path) -> None:
        uri = tmp_path / "lance"
        older = _store(uri)
        _store(uri).upsert_batch([(1, VEC)])

        older.delete_batch([1])

        assert _store(uri).count() == 0  # pre-fix: 1, and no error


class TestSerialisationScope:
    def test_distinct_uris_do_not_block_each_other(self, tmp_path) -> None:
        """
        Guard against over-correcting into one global write lock.

        Independent tables must still write in parallel, so a barrier spanning
        two URIs has to fill. If it times out, the lock is too coarse.
        """
        stores = [_store(tmp_path / "a"), _store(tmp_path / "b")]
        barrier = threading.Barrier(len(stores), timeout=5.0)
        for store in stores:
            _stagger_delete(store, barrier.wait)

        _run([threading.Thread(target=s.upsert_batch, args=([(1, VEC)],)) for s in stores])

        assert barrier.n_waiting == 0 and not barrier.broken, (
            "writers to different URIs were serialised against each other"
        )
        assert all(store.count() == 1 for store in stores)


class TestForeignWriterIsReported:
    """
    A process-local lock cannot cover a second OS process on the same URI.

    Rather than leave that gap silent — the exact failure mode this store keeps
    being bitten by — upsert_batch counts the rows it just wrote and logs when a
    foreign writer has left a duplicate behind.
    """

    def test_duplicate_left_by_a_foreign_writer_is_logged(self, tmp_path, caplog) -> None:
        store = _store(tmp_path / "lance")
        real_add = store._table.add

        def add_twice(data) -> None:
            real_add(data)
            real_add(data)  # stands in for another process's concurrent add

        store._table.add = add_twice

        with caplog.at_level(logging.ERROR, logger="trelix.store.lance"):
            store.upsert_batch([(4, VEC)])

        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert messages, "duplicate rows after upsert produced no ERROR log"
        assert any("duplicate" in m for m in messages)

    def test_clean_upsert_logs_nothing(self, tmp_path, caplog) -> None:
        store = _store(tmp_path / "lance")
        with caplog.at_level(logging.WARNING, logger="trelix.store.lance"):
            store.upsert_batch([(1, VEC), (2, VEC)])
            store.upsert_batch([(1, [0.9] * 4)])
        assert [r.getMessage() for r in caplog.records] == []
        assert store.count() == 2

    def test_concurrent_delete_of_the_same_ids_is_not_reported(self, tmp_path) -> None:
        """
        Fewer rows than written is a legitimate concurrent delete_batch, not
        corruption a reindex is needed for. Only the duplicate direction is an
        error, so this must stay quiet.
        """
        store = _store(tmp_path / "lance")
        real_add = store._table.add

        def add_then_drop(data) -> None:
            real_add(data)
            store._table.delete("chunk_id IN (1)")

        store._table.add = add_then_drop

        store.upsert_batch([(1, VEC)])
        assert store.count() == 0
