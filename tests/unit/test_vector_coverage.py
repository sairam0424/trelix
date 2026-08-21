"""`stored_chunk_ids()` — the primitive that makes a half-embedded index visible.

A run killed mid-Phase-3 leaves chunk rows whose vectors were never written, and the
incremental pre-filter then skips those files forever because their content hash was
committed in Phase 2. Reproduced on a real index: 68,880 chunk rows, 61,652 vectors,
7,228 chunks (10.5%) permanently unretrievable, with `trelix index` reporting
"Files to embed 0".

Finding them needs an ID SET, not a count. `count()` is sentinel-inclusive, so
`count_chunks() - count()` returns holes-minus-sentinels: on the fixture below it reports
1 against a true 2, and it goes negative once sentinels outnumber holes. Both facts are
pinned here.

The planner cliff behind the Python set difference is deliberately NOT tested. A
`LEFT JOIN` / `NOT EXISTS` anti-join measures 6.2 ms at 4 dims and 28.4 s at 3072 dims —
the fixture is FASTER on the wrong implementation, so a timing assertion at this scale
would be worse than none. It is guarded by design (`Database.all_chunk_ids` +
set difference, no anti-join SQL) and by the comment there recording the measurement.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trelix.store.vector import BaseVectorStore, VectorStore

_DIM = 4
_VEC = [0.1, 0.2, 0.3, 0.4]
_OFFSET = BaseVectorStore._SUB_CHUNK_OFFSET


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    """A real sqlite-vec store. No importorskip: sqlite-vec is a core dependency."""
    return VectorStore(tmp_path / "vectors.db", dimension=_DIM)


class TestSqliteStoredChunkIds:
    def test_returns_exactly_the_stored_real_ids(self, store: VectorStore) -> None:
        for chunk_id in (1, 2, 3):
            store.upsert(chunk_id=chunk_id, embedding=_VEC)
        assert store.stored_chunk_ids() == {1, 2, 3}

    def test_empty_store_returns_an_empty_set_not_an_error(self, store: VectorStore) -> None:
        assert store.stored_chunk_ids() == set()

    def test_file_summary_sentinels_are_excluded(self, store: VectorStore) -> None:
        """Summary rows live at -(file_id) in the same table and are not chunks."""
        store.upsert(chunk_id=1, embedding=_VEC)
        store.upsert_file_summary_embedding(7, _VEC)
        assert store.stored_chunk_ids() == {1}

    def test_sub_chunk_sentinels_are_excluded(self, store: VectorStore) -> None:
        """Sub-chunk rows live at sub_chunk_id + _SUB_CHUNK_OFFSET and are not chunks."""
        store.upsert(chunk_id=1, embedding=_VEC)
        store.upsert_sub_chunk_embedding(5, _VEC)
        assert _OFFSET + 5 not in store.stored_chunk_ids()
        assert store.stored_chunk_ids() == {1}

    def test_holes_are_the_set_difference(self, store: VectorStore) -> None:
        store.upsert(chunk_id=1, embedding=_VEC)
        store.upsert(chunk_id=3, embedding=_VEC)
        assert {1, 2, 3} - store.stored_chunk_ids() == {2}

    def test_orphaned_vectors_are_the_reverse_difference(self, store: VectorStore) -> None:
        """The direction that catches a swallowed delete. Reported, never acted on."""
        for chunk_id in (1, 2, 3):
            store.upsert(chunk_id=chunk_id, embedding=_VEC)
        assert store.stored_chunk_ids() - {1, 2} == {3}

    def test_a_count_comparison_cannot_find_the_holes(self, store: VectorStore) -> None:
        """Why this method returns ids. Pins the arithmetic that made the bug invisible.

        3 chunk rows, 1 vector, 2 sentinels: the naive `chunks - count()` says 0 holes
        while the true answer is 2. Add one more sentinel and it goes negative.
        """
        store.upsert(chunk_id=1, embedding=_VEC)
        store.upsert_file_summary_embedding(7, _VEC)
        store.upsert_sub_chunk_embedding(5, _VEC)

        chunk_rows = {1, 2, 3}
        assert len(chunk_rows) - store.count() == 0, "fixture drifted"
        assert len(chunk_rows - store.stored_chunk_ids()) == 2

    def test_a_missing_vec0_table_raises_rather_than_reporting_total_blindness(
        self, tmp_path: Path
    ) -> None:
        """Swallowing this into set() makes a broken store read as "every chunk is a
        hole", which the indexer would act on by re-embedding the whole repo for money."""
        store = VectorStore(tmp_path / "vectors.db", dimension=_DIM)
        store._conn.execute("DROP TABLE chunk_embeddings")
        store._conn.commit()

        with pytest.raises(sqlite3.OperationalError):
            store.stored_chunk_ids()


class TestBaseDefault:
    def test_a_backend_that_cannot_enumerate_says_so(self) -> None:
        """Not an @abstractmethod on purpose: that would make a backend missing this
        capability fail at construction, i.e. trelix would not start."""

        class _Minimal(BaseVectorStore):
            def upsert_batch(self, pairs: list[tuple[int, list[float]]]) -> None: ...
            def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
                return []

            def delete_batch(self, chunk_ids: list[int]) -> None: ...
            def count(self) -> int:
                return 0

            def upsert_file_summary_embedding(
                self, file_id: int, embedding: list[float]
            ) -> None: ...
            def search_file_summaries(
                self, query_embedding: list[float], k: int
            ) -> list[tuple[int, float]]:
                return []

            def upsert_sub_chunk_embedding(
                self, sub_chunk_id: int, embedding: list[float]
            ) -> None: ...
            def search_sub_chunks(
                self, query_embedding: list[float], k: int
            ) -> list[tuple[int, float]]:
                return []

        store = _Minimal()  # constructing must work — that is the whole point
        with pytest.raises(NotImplementedError):
            store.stored_chunk_ids()


class TestSentinelOffsetIsSharedNotCopied:
    def test_every_backend_reads_the_one_definition(self) -> None:
        """Three shadowing copies of _SUB_CHUNK_OFFSET were deleted. All three backends
        now filter sentinels off the base constant, so a re-added copy would drift."""
        from trelix.store.vector import SQLiteVectorStore

        assert "_SUB_CHUNK_OFFSET" not in vars(SQLiteVectorStore)
        assert SQLiteVectorStore._SUB_CHUNK_OFFSET == _OFFSET

        lance = pytest.importorskip("trelix.store.vector_lance")
        assert "_SUB_CHUNK_OFFSET" not in vars(lance.LanceVectorStore)

        qdrant = pytest.importorskip("trelix.store.vector_qdrant")
        assert "_SUB_CHUNK_OFFSET" not in vars(qdrant.QdrantVectorStore)


class TestLanceStoredChunkIds:
    """Same assertions on the lance backend. Skipped when the extra is absent."""

    @pytest.fixture
    def lance_store(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        pytest.importorskip("lancedb", reason="lance extra not installed")
        from trelix.store.vector_lance import LanceVectorStore

        return LanceVectorStore(uri=str(tmp_path / "lance"), table_name="chunks", dimension=_DIM)

    def test_returns_exactly_the_stored_real_ids(self, lance_store) -> None:  # type: ignore[no-untyped-def]
        lance_store.upsert_batch([(1, _VEC), (2, _VEC), (3, _VEC)])
        assert lance_store.stored_chunk_ids() == {1, 2, 3}

    def test_empty_table_returns_an_empty_set(self, lance_store) -> None:  # type: ignore[no-untyped-def]
        assert lance_store.stored_chunk_ids() == set()

    def test_sentinels_are_excluded(self, lance_store) -> None:  # type: ignore[no-untyped-def]
        lance_store.upsert_batch([(1, _VEC)])
        lance_store.upsert_file_summary_embedding(7, _VEC)
        lance_store.upsert_sub_chunk_embedding(5, _VEC)
        assert lance_store.stored_chunk_ids() == {1}

    def test_holes_are_the_set_difference(self, lance_store) -> None:  # type: ignore[no-untyped-def]
        lance_store.upsert_batch([(1, _VEC), (3, _VEC)])
        assert {1, 2, 3} - lance_store.stored_chunk_ids() == {2}

    def test_a_second_handles_write_is_visible(self, tmp_path: Path) -> None:
        """The _checkout_latest() guard. Without it a handle answers from the version it
        opened at: reproduced 4,500 ids against a true 4,501 after another handle's write.

        Under-reporting is the expensive direction — every id missed reads as a hole and
        gets re-embedded for money.
        """
        pytest.importorskip("lancedb", reason="lance extra not installed")
        from trelix.store.vector_lance import LanceVectorStore

        uri = str(tmp_path / "lance")
        handle_a = LanceVectorStore(uri=uri, table_name="chunks", dimension=_DIM)
        handle_a.upsert_batch([(1, _VEC)])

        handle_b = LanceVectorStore(uri=uri, table_name="chunks", dimension=_DIM)
        handle_b.upsert_batch([(2, _VEC)])

        assert handle_a.stored_chunk_ids() == {1, 2}


class TestQdrantStoredChunkIds:
    """The qdrant backend against qdrant-client's local mode.

    Local mode runs the real client, the real scroll and the real pagination cursor with
    no server — so this is genuine coverage of the loop, not a MagicMock asserting that
    the call written is the call written. It is NOT coverage of a Qdrant *server*: notably
    the server requires unsigned 64-bit point ids while local mode accepts the negative
    file-summary sentinels, so whether those land at all on a real server is untested and
    unrelated to this method.
    """

    @pytest.fixture
    def qdrant_store(self, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
        pytest.importorskip("qdrant_client", reason="qdrant extra not installed")
        import qdrant_client

        from trelix.core.config import IndexConfig, StoreConfig
        from trelix.store.vector_qdrant import QdrantVectorStore

        local = qdrant_client.QdrantClient(location=":memory:")
        monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **_kwargs: local)

        config = IndexConfig(
            repo_path=".", store=StoreConfig(backend="qdrant", qdrant_collection="t")
        )
        return QdrantVectorStore(config, dimension=_DIM)

    def test_returns_exactly_the_stored_real_ids(self, qdrant_store) -> None:  # type: ignore[no-untyped-def]
        qdrant_store.upsert_batch([(1, _VEC), (2, _VEC), (3, _VEC)])
        assert qdrant_store.stored_chunk_ids() == {1, 2, 3}

    def test_empty_collection_returns_an_empty_set(self, qdrant_store) -> None:  # type: ignore[no-untyped-def]
        assert qdrant_store.stored_chunk_ids() == set()

    def test_sentinels_are_excluded(self, qdrant_store) -> None:  # type: ignore[no-untyped-def]
        qdrant_store.upsert_batch([(1, _VEC)])
        qdrant_store.upsert_file_summary_embedding(7, _VEC)
        qdrant_store.upsert_sub_chunk_embedding(5, _VEC)
        assert qdrant_store.stored_chunk_ids() == {1}

    def test_it_pages_past_one_scroll_page(self, qdrant_store, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """The cursor loop, not just the first page. Paged at 3 ids so the fixture stays
        small while still forcing several round trips."""
        import trelix.store.vector_qdrant as mod

        monkeypatch.setattr(mod, "_SCROLL_PAGE_SIZE", 3)
        expected = set(range(1, 21))
        qdrant_store.upsert_batch([(i, _VEC) for i in sorted(expected)])
        assert qdrant_store.stored_chunk_ids() == expected
