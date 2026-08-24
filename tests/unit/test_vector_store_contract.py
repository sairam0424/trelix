"""VectorStoreContract: behaviour every real `BaseVectorStore` backend must share.

One parametrized `store` fixture instantiates a REAL backend per param id
("sqlite", "lance") — no `Mock`/`MagicMock` standing in for the interface under
test (rule 3). The same test bodies run against both, so a change that keeps one
backend's contract and silently breaks the other's is caught structurally,
instead of by two hand-maintained files that can drift apart from each other.

Backend coverage, and why Qdrant is NOT a third param
-------------------------------------------------------
sqlite  -- always available (sqlite-vec is a core dependency, not an extra).
lance   -- real LanceVectorStore when `lancedb` is importable in this venv
           (it is, 0.33.0 measured); `pytest.importorskip` inside the fixture
           skips only the lance-parametrized instance, per-test, when it is
           not, following the existing per-test pattern in
           tests/unit/test_vector_coverage.py (module-scope importorskip would
           take the sqlite half down with it, which is the wrong blast radius,
           and would additionally require registering this file in
           tests/conftest.py's REQUIRES_EXTRA_FILES table).
qdrant  -- deliberately excluded from this fixture. `QdrantClient(':memory:')`
           was VERIFIED on today's qdrant-client (1.18.0, this venv) to accept
           `PointStruct(id=-5, ...)` without error — the exact fake-blindness
           the original plan warned about, re-confirmed here rather than
           trusted from the plan. Real Qdrant point ids are unsigned 64-bit
           integers or UUIDs; `vector_qdrant.py`'s own
           `upsert_file_summary_embedding` writes `id=-(file_id)`, and its
           `stored_chunk_ids` docstring already documents that a real server
           is expected to reject that write, untested. A contract fixture built
           on `:memory:` would therefore pass by exercising the fake's
           blindness, not the real backend's contract — exactly backwards for
           an instrument whose point is to catch backend divergence. See
           `test_qdrant_leg_is_out_of_scope_here` below for the full,
           runnable spec this leg needs (testcontainers; Docker is not
           installed in this execution environment — verified: `docker info`
           reports "command not found").

Contract scope
--------------
Every method below is on `BaseVectorStore` and is exercised through the public
`upsert_batch` / `search` / `delete_batch` / `count` / `stored_chunk_ids` /
`upsert_file_summary_embedding` / `search_file_summaries` /
`upsert_sub_chunk_embedding` / `search_sub_chunks` surface — never through a
backend-private attribute, so the contract stays honest about what a NEW
backend would actually have to implement.

Deliberately NOT asserted here: that `search()` itself excludes file-summary /
sub-chunk sentinel rows. `tests/unit/test_store.py`'s
`TestVectorSearchExcludesSentinelRows.test_sub_chunk_search_still_sees_sub_chunks`
documents that this is SQLite-specific by design — Lance and Qdrant build their
sentinel-only searches on top of the plain `search()` with a k*5 oversample and
do NOT filter sentinels out of `search()` itself. Asserting it here would
encode a backend-specific behaviour as a cross-backend contract and fail on
Lance for a reason that has nothing to do with a real defect.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from trelix.store.vector import SQLiteVectorStore

_DIM = 4
_VEC = [0.1, 0.2, 0.3, 0.4]

# Explicit table, not something derived by iterating the params list (rule 2):
# the exception TYPE a wrong-width insert raises is genuinely backend-specific
# (there is no shared `DimensionMismatchError` at the vector-store-engine layer
# -- that class lives one layer up, in trelix.store.dimension_guard, and
# compares a STORED dimension against the CURRENT provider's, not a vector's
# actual length against the table's declared width). Verified directly against
# this tree, not assumed from the plan:
#   sqlite (sqlite-vec 0.1.x): sqlite3.OperationalError
#     "Dimension mismatch for inserted vector for the "embedding" column.
#      Expected 4 dimensions but received 8."
#   lance (lancedb 0.33.0):    pyarrow.lib.ArrowInvalid
#     "Length of item not correct: expected 4 but got array of size 8"
#
# `pyarrow` itself lives in the `lance` EXTRA (pyproject.toml: `lance = [
# "lancedb>=0.6.0", "pyarrow>=14.0"]`), not in trelix's core dependencies, so
# it is imported lazily here rather than at module top-level -- an
# unconditional `import pyarrow` would take the sqlite-only leg of this file
# down with it on a lean install that has neither extra, which is exactly the
# "wrong blast radius" this module's docstring warns against for the lancedb
# import itself.
try:
    import pyarrow

    _ARROW_INVALID: type[Exception] | None = pyarrow.ArrowInvalid
except ImportError:
    _ARROW_INVALID = None

_DIMENSION_MISMATCH_EXCEPTION: dict[str, type[Exception]] = {"sqlite": sqlite3.OperationalError}
if _ARROW_INVALID is not None:
    _DIMENSION_MISMATCH_EXCEPTION["lance"] = _ARROW_INVALID


def _make_sqlite_store(tmp_path: Path) -> SQLiteVectorStore:
    return SQLiteVectorStore(tmp_path / "vectors.db", dimension=_DIM)


def _make_lance_store(tmp_path: Path) -> Any:
    pytest.importorskip("lancedb", reason="lance extra not installed")
    from trelix.store.vector_lance import LanceVectorStore

    return LanceVectorStore(uri=str(tmp_path / "lance"), table_name="chunks", dimension=_DIM)


_BACKEND_FACTORIES = {
    "sqlite": _make_sqlite_store,
    "lance": _make_lance_store,
}


@pytest.fixture(params=sorted(_BACKEND_FACTORIES), ids=sorted(_BACKEND_FACTORIES))
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    """One real backend instance per param id. See module docstring for scope."""
    return _BACKEND_FACTORIES[request.param](tmp_path)


@pytest.fixture
def backend_name(request: pytest.FixtureRequest, store: Any) -> str:
    """The param id `store` was built from, for the exception-type lookup table."""
    # store's own fixture id is the parametrization key; re-derive it from the
    # callspec rather than threading a second parameter through every test.
    return request.node.callspec.params["store"]


class TestVectorStoreContract:
    """Shared behaviour, run once per backend via the `store` fixture above."""

    # -- insert-then-query round trip -----------------------------------

    def test_insert_then_query_round_trip(self, store: Any) -> None:
        """FALSIFIED BY: `upsert_batch` that stores nothing, or `search` that
        never finds a vector identical to the query. Real defect shape this
        catches: a backend whose write path silently no-ops."""
        store.upsert_batch([(1, _VEC)])
        results = store.search(_VEC, k=5)
        assert len(results) == 1
        chunk_id, _score = results[0]
        assert chunk_id == 1

    def test_search_nearest_is_returned_first(self, store: Any) -> None:
        """FALSIFIED BY: a backend that returns ANN results in insertion order
        rather than similarity order (both measured L2/L2-equivalent here)."""
        store.upsert_batch(
            [
                (1, [1.0, 0.0, 0.0, 0.0]),
                (2, [0.0, 1.0, 0.0, 0.0]),
                (3, [0.0, 0.0, 1.0, 0.0]),
            ]
        )
        results = store.search([1.0, 0.0, 0.0, 0.0], k=3)
        assert results[0][0] == 1, f"expected the identical vector first, got {results}"

    def test_search_on_empty_store_returns_empty_list(self, store: Any) -> None:
        """FALSIFIED BY: a backend that raises on an empty index instead of
        returning `[]` -- a first-ever `trelix index` run must not crash here."""
        assert store.search(_VEC, k=5) == []

    def test_search_k_limit_is_respected(self, store: Any) -> None:
        """FALSIFIED BY: a backend ignoring `k` and returning every stored row."""
        store.upsert_batch([(i, [float(i)] * _DIM) for i in range(1, 11)])
        results = store.search([1.0] * _DIM, k=3)
        assert len(results) == 3
        ids = {cid for cid, _ in results}
        assert ids.issubset(set(range(1, 11)))

    # -- upsert semantics --------------------------------------------------

    def test_upsert_replaces_rather_than_duplicates(self, store: Any) -> None:
        """FALSIFIED BY: a backend whose upsert appends a second row for the
        same chunk_id instead of replacing -- the LanceDB delete-then-add bug
        this repo already fixed once (see vector_lance.py's own docstring)."""
        store.upsert_batch([(42, [1.0, 0.0, 0.0, 0.0])])
        store.upsert_batch([(42, [0.0, 1.0, 0.0, 0.0])])
        assert store.count() == 1
        results = store.search([0.0, 1.0, 0.0, 0.0], k=5)
        assert [cid for cid, _ in results] == [42]

    def test_upsert_batch_multiple_ids(self, store: Any) -> None:
        store.upsert_batch(
            [
                (10, [1.0, 0.0, 0.0, 0.0]),
                (11, [0.0, 1.0, 0.0, 0.0]),
                (12, [0.0, 0.0, 1.0, 0.0]),
            ]
        )
        assert store.count() == 3
        assert store.stored_chunk_ids() == {10, 11, 12}

    # -- delete semantics --------------------------------------------------

    def test_delete_batch_removes_the_named_ids_only(self, store: Any) -> None:
        store.upsert_batch(
            [
                (20, [1.0, 0.0, 0.0, 0.0]),
                (21, [0.0, 1.0, 0.0, 0.0]),
                (22, [0.0, 0.0, 1.0, 0.0]),
            ]
        )
        store.delete_batch([20, 21])
        assert store.count() == 1
        assert store.stored_chunk_ids() == {22}

    def test_delete_batch_empty_list_is_a_noop(self, store: Any) -> None:
        """FALSIFIED BY: a backend that raises on `delete_batch([])`, or one
        that treats an empty selector as "delete everything" (a real footgun
        shape for a SQL/predicate-based backend)."""
        store.upsert_batch([(99, [1.0, 0.0, 0.0, 0.0])])
        before = store.count()
        store.delete_batch([])
        assert store.count() == before
        assert store.stored_chunk_ids() == {99}

    # -- dimension mismatch -------------------------------------------------

    def test_dimension_mismatch_raises(self, store: Any, backend_name: str) -> None:
        """FALSIFIED BY: a backend that silently truncates, pads, or corrupts a
        wrong-width vector instead of raising. The exact exception type is
        backend-specific (see `_DIMENSION_MISMATCH_EXCEPTION`); the SHARED
        contract is only "raises, does not corrupt", which is what the bare
        `pytest.raises(Exception)` on the base fixture would check -- this
        also pins the concrete type per backend so a silent change of error
        type (e.g. sqlite-vec swapping its OperationalError for a bare
        ValueError) is visible instead of masked by a generic catch.
        """
        expected = _DIMENSION_MISMATCH_EXCEPTION.get(backend_name)
        if expected is None:
            pytest.skip(f"pyarrow (lance extra) not importable, cannot type-check {backend_name}")
        wrong_width = [0.1] * (_DIM * 2)
        with pytest.raises(expected):
            store.upsert_batch([(1, wrong_width)])

    def test_dimension_mismatch_in_a_batch_leaves_no_partial_write(
        self, store: Any, backend_name: str
    ) -> None:
        """FALSIFIED BY: a backend that lands the valid rows of a batch before
        hitting the invalid one. Verified directly on both backends before
        writing this test: sqlite's per-row loop rolls back its transaction on
        exception (`except Exception: self._conn.rollback(); raise`), and
        lance's pyarrow.table() construction fails before any LanceDB commit
        is issued -- both land `count() == 0`, not `count() == 1`.
        """
        expected = _DIMENSION_MISMATCH_EXCEPTION.get(backend_name)
        if expected is None:
            pytest.skip(f"pyarrow (lance extra) not importable, cannot type-check {backend_name}")
        wrong_width = [0.1] * (_DIM * 2)
        with pytest.raises(expected):
            store.upsert_batch([(1, _VEC), (2, wrong_width)])
        assert store.count() == 0, (
            "a failed batch left a partial write behind -- the valid row landed "
            "even though the batch as a whole raised"
        )

    # -- file-summary / sub-chunk sentinel round trip -----------------------

    def test_file_summary_embedding_round_trips(self, store: Any) -> None:
        store.upsert_file_summary_embedding(file_id=7, embedding=[9.0] * _DIM)
        results = store.search_file_summaries([9.0] * _DIM, k=5)
        assert [file_id for file_id, _ in results] == [7]

    def test_sub_chunk_embedding_round_trips(self, store: Any) -> None:
        store.upsert_sub_chunk_embedding(sub_chunk_id=3, embedding=[8.0] * _DIM)
        results = store.search_sub_chunks([8.0] * _DIM, k=5)
        assert [sub_chunk_id for sub_chunk_id, _ in results] == [3]

    def test_stored_chunk_ids_excludes_both_sentinel_kinds(self, store: Any) -> None:
        """`stored_chunk_ids()` (unlike plain `search()`, see module docstring)
        IS a shared contract: both backends override the base's
        `NotImplementedError` and both must exclude file-summary and sub-chunk
        sentinels from the real-chunk id set they return."""
        store.upsert_batch([(1, _VEC)])
        store.upsert_file_summary_embedding(file_id=7, embedding=_VEC)
        store.upsert_sub_chunk_embedding(sub_chunk_id=5, embedding=_VEC)
        assert store.stored_chunk_ids() == {1}


# ---------------------------------------------------------------------------
# Qdrant: fully specified, not run here (no Docker in this environment).
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Qdrant leg intentionally not run against QdrantClient(':memory:') -- "
        "verified on qdrant-client 1.18.0 (this venv, 2026-08-24) that :memory: "
        "accepts PointStruct(id=-5, ...), which a real server rejects (point "
        "ids are unsigned 64-bit ints or UUIDs). Docker is not installed in "
        "this execution environment (`docker info` -> 'command not found'), "
        "so the real-backend leg cannot run here. See the docstring below for "
        "the exact command and assertions to run it wherever Docker exists."
    )
)
def test_qdrant_leg_is_out_of_scope_here() -> None:
    """SPEC for the Qdrant contract leg, to run against a real container.

    Prerequisite (verify first, do not assume): Docker must be running.
        docker info

    Bring up a real Qdrant server (matches qdrant-client 1.18.0's wire
    protocol; pin the image tag to whatever CI's compose file uses if this
    repo has one -- `docker-compose.yml` at the repo root does not currently
    define a qdrant service):
        docker run -d --rm -p 6333:6333 --name trelix-contract-qdrant qdrant/qdrant:v1.11.0

    Point QdrantVectorStore at it instead of ':memory:' (this is the whole
    fix -- do not construct `QdrantClient(location=':memory:')` anywhere in
    the replacement fixture):
        from trelix.core.config import IndexConfig, StoreConfig
        from trelix.store.vector_qdrant import QdrantVectorStore

        config = IndexConfig(
            repo_path=".",
            store=StoreConfig(
                backend="qdrant",
                qdrant_url="http://localhost:6333",
                qdrant_collection="trelix_contract_test",
            ),
        )
        store = QdrantVectorStore(config, dimension=4)

    Then run every test in `TestVectorStoreContract` above against that
    `store` (add "qdrant" to `_BACKEND_FACTORIES`/`_DIMENSION_MISMATCH_EXCEPTION`
    with the real-server factory and exception type measured against the
    container, not assumed).

    Two assertions this leg exists specifically to make, that ':memory:'
    CANNOT make honestly:
      1. `upsert_file_summary_embedding(file_id=7, ...)` either succeeds and
         `search_file_summaries` finds it (current in-process behaviour), OR
         raises on the negative id -- and vector_qdrant.py's own
         `stored_chunk_ids` docstring already predicts the latter against a
         real server. Whichever it is, PIN it here; today it is UNTESTED
         against a real server, only asserted against the fake that is known
         to be more permissive.
      2. `test_dimension_mismatch_raises` against a real server: qdrant-client
         validates vector length against the collection's configured `size`
         before sending the request in recent versions, but this must be
         MEASURED against qdrant-client 1.18.0 + a real server, not assumed
         from :memory:'s behaviour -- :memory: is exactly the component whose
         fidelity is in question here.

    Teardown:
        docker rm -f trelix-contract-qdrant
    """
