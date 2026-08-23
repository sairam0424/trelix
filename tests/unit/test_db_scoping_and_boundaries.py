"""Per-file scoping, transaction boundaries and upsert-conflict handling in the store.

Every class here was written against a mutant of `src/trelix/store/db.py` that the
3,868-test unit suite passed in full. Each docstring names the mutation it exists to
kill; the four scoping classes all kill the same SHAPE of defect, which is why they
are together: a read helper that takes a `file_id` and then does not actually filter
on it.

Why that shape is expensive rather than merely wrong. `symbols.qualified_name` is
`ClassName.method_name` (see `Symbol` in core/models.py) — it is NOT file-qualified,
so `main`, `__init__`, `setUp`, `handler` and `Config.__init__` collide across files
in every real repo. `Indexer._insert_one` decides what to re-embed with

    existing_hashes = self.db.get_symbol_hashes_for_file(file_id)
    ...
    if existing_hashes.get(symbol.qualified_name) == new_hash:
        unchanged_qualified_names.add(symbol.qualified_name)

so the moment that map leaks another file's rows, a symbol in the file being indexed
is declared unchanged because a DIFFERENT file happens to hold the same
qualified_name with the same body. It is then never inserted, never chunked and
never embedded — silently unsearchable, with `trelix stats` still counting it. That
is the "skip everything" half of the incremental path, and nothing in the suite
looked at it.

The scoping tests deliberately give the two files IDENTICAL symbol names and
identical bodies. That is not a contrived collision; it is the common case
(`def main(): pass` in two entry points) and it is the only fixture in which an
unfiltered query is distinguishable from a filtered one.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trelix.core.models import Chunk, IndexedFile, Language, Symbol, SymbolKind
from trelix.store import db as db_module
from trelix.store.db import Database

# The batch width `get_chunk_text_and_tokens` pages over. Written as a literal
# rather than imported, so a change to the module constant cannot silently
# redefine what this file claims. TestChunkTextBatchBoundary asserts the
# relationship between the two instead.
_BATCH_WIDTH_LITERAL = 500


def _make_file(db: Database, rel_path: str, *, file_hash: str = "h") -> int:
    return db.upsert_file(
        IndexedFile(
            path=f"/repo/{rel_path}",
            rel_path=rel_path,
            language=Language.PYTHON,
            hash=file_hash,
            size_bytes=10,
        )
    )


def _make_symbol(db: Database, file_id: int, qualified_name: str, *, body: str) -> int:
    return db.insert_symbol(
        Symbol(
            file_id=file_id,
            name=qualified_name.split(".")[-1],
            qualified_name=qualified_name,
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=5,
            signature=f"def {qualified_name}()",
            body=body,
        )
    )


class TestSymbolHashMapIsScopedToItsFile:
    """get_symbol_hashes_for_file() must return ONLY the given file's symbols.

    Mutation that must make these fail: dropping the WHERE filter in
    `get_symbol_hashes_for_file`, e.g.

        SELECT qualified_name, content_hash FROM symbols WHERE file_id = ? OR 1=1

    Consequence if it survives: SKIP EVERYTHING. Every symbol in a newly-added
    file whose `qualified_name` + body already exists anywhere in the index is
    classified unchanged by `Indexer._insert_one` and never embedded.
    """

    def test_another_files_symbol_is_absent_from_this_files_hash_map(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        shared_body = "def main(): return 1"
        file_a = _make_file(db, "a.py")
        file_b = _make_file(db, "b.py")
        _make_symbol(db, file_a, "main", body=shared_body)
        db._conn.commit()

        # Precondition on the FIXTURE, not on the assertion: file A really does
        # hold a row under this qualified_name, so an empty answer for file B
        # below is scoping and not an empty table. If this ever fails, the
        # _make_file/_make_symbol fixture stopped inserting and the real
        # assertion has become vacuous.
        assert set(db.get_symbol_hashes_for_file(file_a)) == {"main"}

        assert db.get_symbol_hashes_for_file(file_b) == {}

    def test_the_map_holds_exactly_this_files_qualified_names(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_a = _make_file(db, "a.py")
        file_b = _make_file(db, "b.py")
        _make_symbol(db, file_a, "main", body="def main(): return 1")
        _make_symbol(db, file_a, "Config.load", body="def load(self): return 2")
        _make_symbol(db, file_b, "main", body="def main(): return 1")
        _make_symbol(db, file_b, "helper", body="def helper(): return 3")
        db._conn.commit()

        # Explicit tables, written out rather than derived from the inserts above,
        # and compared with set equality in BOTH directions.
        expected_a = {"main", "Config.load"}
        expected_b = {"main", "helper"}

        actual_a = set(db.get_symbol_hashes_for_file(file_a))
        actual_b = set(db.get_symbol_hashes_for_file(file_b))

        assert actual_a == expected_a
        assert expected_a == actual_a
        assert actual_b == expected_b
        assert expected_b == actual_b

    def test_the_hash_under_a_colliding_name_is_this_files_own(self, tmp_path: Path) -> None:
        """When two files hold the same qualified_name the leak is invisible in the
        KEY SET — the dict comprehension collapses both rows onto one key — so the
        `OR 1=1` mutant is only observable in the VALUE, where the later row wins.

        The expected hash is read back with raw SQL scoped to file A rather than
        recomputed from `insert_symbol`'s signature+body formula, so this pins the
        stored row and not db.py's own arithmetic.
        """
        db = Database(tmp_path / "index.db")
        file_a = _make_file(db, "a.py")
        file_b = _make_file(db, "b.py")
        _make_symbol(db, file_a, "main", body="def main(): return 1")
        _make_symbol(db, file_b, "main", body="def main(): return 999999")
        db._conn.commit()

        hash_a = db._conn.execute(
            "SELECT content_hash FROM symbols WHERE file_id = ?", (file_a,)
        ).fetchone()[0]
        hash_b = db._conn.execute(
            "SELECT content_hash FROM symbols WHERE file_id = ?", (file_b,)
        ).fetchone()[0]

        # Preconditions naming the fixture: two distinct rows with DIFFERENT
        # hashes. If the bodies ever stop differing, or _make_symbol stops
        # storing a hash, the assertion below becomes true by construction.
        assert hash_a != hash_b
        assert hash_a

        assert db.get_symbol_hashes_for_file(file_a)["main"] == hash_a


class TestChunkIdsForSymbolsIsScopedToItsFile:
    """get_chunk_ids_for_symbols() must not return another file's chunk ids.

    Mutation that must make this fail: neutralising the `s.file_id = ?` term in
    `get_chunk_ids_for_symbols`, e.g. `WHERE (s.file_id = ? OR 1=1) AND ...`.

    Consequence if it survives: the incremental path feeds this list straight to
    `vector_store.delete_batch()`. Re-indexing a.py after a one-character edit
    would delete the vectors of every same-named symbol in every OTHER file,
    which are then never re-embedded because those files' hashes are unchanged.
    Silent, permanent search holes in files nobody touched.
    """

    def test_only_this_files_chunk_id_comes_back(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_a = _make_file(db, "a.py")
        file_b = _make_file(db, "b.py")
        sym_a = _make_symbol(db, file_a, "main", body="def main(): return 1")
        sym_b = _make_symbol(db, file_b, "main", body="def main(): return 1")
        chunk_a = db.insert_chunk(Chunk(symbol_id=sym_a, chunk_text="a body", token_count=3))
        chunk_b = db.insert_chunk(Chunk(symbol_id=sym_b, chunk_text="b body", token_count=3))
        db._conn.commit()

        # Precondition naming the fixture: the two chunks are distinct rows, so
        # a single-element answer is scoping rather than a collapsed insert.
        assert chunk_a != chunk_b

        assert set(db.get_chunk_ids_for_symbols(file_a, ["main"])) == {chunk_a}
        assert set(db.get_chunk_ids_for_symbols(file_b, ["main"])) == {chunk_b}


class TestSymbolIdsForFileIdIsScopedToItsFile:
    """get_symbol_ids_for_file_id() must not return another file's symbol ids.

    Mutation that must make this fail: `SELECT id FROM symbols WHERE file_id = ?
    OR 1=1` in `get_symbol_ids_for_file_id`.

    Consequence if it survives: the import graph resolves a file to every symbol
    in the repo, so graph expansion from one imported file pulls in the whole
    index.
    """

    def test_the_id_set_is_exactly_this_files_symbols(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_a = _make_file(db, "a.py")
        file_b = _make_file(db, "b.py")
        sym_a = _make_symbol(db, file_a, "main", body="def main(): return 1")
        sym_b = _make_symbol(db, file_b, "main", body="def main(): return 1")
        db._conn.commit()

        # Precondition naming the fixture: distinct rows in two distinct files.
        assert sym_a != sym_b

        actual = set(db.get_symbol_ids_for_file_id(file_a))
        expected = {sym_a}
        assert actual == expected
        assert expected == actual
        assert sym_b not in actual


class TestTransactionRollsBackOnFailure:
    """Database.transaction() must roll back the partial write when the body raises.

    Mutation that must make this fail: deleting `self._conn.rollback()` from the
    `except Exception:` arm of `transaction()`, keeping the bare `raise`.

    Consequence if it survives: `insert_symbol` does NOT commit on its own — it
    relies on this context manager — and `Indexer._insert_one` inserts a whole
    file's symbols inside one `with self.db.transaction()`. Without the rollback
    the half-written symbol rows stay in the open transaction, and the next
    unrelated `commit()` on the same connection (every `upsert_file`,
    `insert_query_telemetry`, `set_index_metadata`, ...) flushes them to disk. A
    file that FAILED to index is then indistinguishable from one that succeeded,
    except that it is missing the symbols the failure cut short.
    """

    def test_a_raising_body_leaves_no_rows_behind(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "a.py")
        db._conn.commit()

        with pytest.raises(RuntimeError, match="phase 2 blew up"):
            with db.transaction():
                _make_symbol(db, file_id, "main", body="def main(): return 1")
                raise RuntimeError("phase 2 blew up")

        # A later, unrelated commit on the same connection is what promotes an
        # un-rolled-back write to durable. Do it, so the assertion below is
        # about rollback and not about "nothing has been committed yet".
        db.set_index_metadata("unrelated", "1")

        surviving = db._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        assert int(surviving) == 0

    def test_a_clean_body_still_commits(self, tmp_path: Path) -> None:
        """Discriminating counterpart: if this fails, the class above proves
        nothing, because a transaction() that never persists anything would pass
        the rollback test for the wrong reason."""
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "a.py")

        with db.transaction():
            _make_symbol(db, file_id, "main", body="def main(): return 1")

        persisted = sqlite3.connect(str(tmp_path / "index.db"))
        try:
            count = persisted.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        finally:
            persisted.close()
        assert int(count) == 1


class TestDeleteFileByPathMatchesEitherKey:
    """delete_file_by_path() looks up on abs_path OR rel_path — never both at once.

    Mutation that must make these fail: `WHERE path = ? OR rel_path = ?` becoming
    `WHERE path = ? AND rel_path = ?`.

    Consequence if it survives: the watcher's delete path returns False and
    removes nothing whenever the caller's two keys do not BOTH match the stored
    row — which is exactly the case the `rel_path` fallback exists for (an index
    written from a different absolute root, e.g. moved checkout, container
    mount, or symlinked path). Deleted files stay searchable forever and their
    vectors are never reclaimed.
    """

    def test_a_matching_rel_path_alone_is_enough(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        _make_file(db, "a.py")

        # Precondition naming the fixture: the row is really there under the
        # rel_path we are about to look up by.
        assert db.get_file_hash("a.py") == "h"

        assert db.delete_file_by_path("/some/other/root/a.py", "a.py") is True
        assert db.get_file_hash("a.py") is None

    def test_a_matching_abs_path_alone_is_enough(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        _make_file(db, "a.py")

        assert db.get_file_hash("a.py") == "h"

        assert db.delete_file_by_path("/repo/a.py", "moved/elsewhere/a.py") is True
        assert db.get_file_hash("a.py") is None

    def test_neither_key_matching_still_deletes_nothing(self, tmp_path: Path) -> None:
        """The OR must not become a tautology either."""
        db = Database(tmp_path / "index.db")
        _make_file(db, "a.py")

        assert db.delete_file_by_path("/nope/z.py", "z.py") is False
        assert db.get_file_hash("a.py") == "h"


class TestFileSummaryUpsertOverwrites:
    """upsert_file_summary() must UPDATE on conflict, not ignore the new value.

    Mutation that must make these fail: `ON CONFLICT(file_id) DO UPDATE SET
    summary=excluded.summary, chunk_id=excluded.chunk_id, created_at=...`
    becoming `ON CONFLICT(file_id) DO NOTHING`.

    Consequence if it survives: a file's summary is frozen at whatever the first
    index run produced. Re-indexing an edited file leaves the stale summary in
    place — and the summary is embedded under the `chunk_id = -(file_id)`
    sentinel, so the file-level search leg answers from text that no longer
    exists in the repo.
    """

    def test_a_second_upsert_replaces_the_summary(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "a.py")

        db.upsert_file_summary(file_id, "first summary")
        # Precondition naming the fixture: the first write landed, so a stale
        # read below is DO NOTHING and not a write that never happened.
        assert db.get_file_summary(file_id) == "first summary"

        db.upsert_file_summary(file_id, "second summary")
        assert db.get_file_summary(file_id) == "second summary"

    def test_the_second_upsert_does_not_add_a_row(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "a.py")

        db.upsert_file_summary(file_id, "first summary")
        db.upsert_file_summary(file_id, "second summary")

        rows = db._conn.execute(
            "SELECT COUNT(*) FROM file_summaries WHERE file_id = ?", (file_id,)
        ).fetchone()[0]
        assert int(rows) == 1


class TestChunkTextBatchBoundary:
    """get_chunk_text_and_tokens() must not drop an id at a batch boundary.

    Mutation that must make this fail: the slice `ids[start : start +
    _MAX_SQL_PARAMS]` becoming `ids[start : start + _MAX_SQL_PARAMS - 1]` while
    `start` still advances by the full width — one id per batch is silently
    never queried.

    Consequence if it survives: this is the vector-repair read path. Chunks
    whose id lands on a boundary are dropped from the repair work item, so the
    hole they represent is reported on every subsequent run and never healed —
    `trelix index` warns about vectors it will never actually restore.
    """

    def test_every_requested_id_comes_back_across_a_batch_boundary(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        file_id = _make_file(db, "a.py")
        symbol_id = _make_symbol(db, file_id, "main", body="def main(): return 1")

        # Precondition naming the fixture: 500 rows only span a batch boundary
        # while the module's batch width is <= 500. Reading the constant here
        # guards the FIXTURE's discriminating power; the expected value below is
        # still the literal 500, never derived from the module.
        assert db_module._MAX_SQL_PARAMS <= _BATCH_WIDTH_LITERAL, (
            "the 500-row fixture no longer spans a batch boundary — raise it "
            "above _MAX_SQL_PARAMS or this test stops discriminating"
        )

        chunk_ids = [
            db.insert_chunk(Chunk(symbol_id=symbol_id, chunk_text=f"c{i}", token_count=1))
            for i in range(500)
        ]
        db._conn.commit()
        assert len(chunk_ids) == 500

        returned = {row[0] for row in db.get_chunk_text_and_tokens(chunk_ids)}
        assert len(returned) == 500
        assert returned == set(chunk_ids)
        assert set(chunk_ids) == returned


class TestAgentTurnIndexCollisionRaises:
    """A duplicate (session_id, turn_index) must raise, not persist silently.

    Mutation that must make this fail: `CREATE UNIQUE INDEX IF NOT EXISTS
    idx_agent_turns_session` losing the UNIQUE keyword.

    Consequence if it survives: `insert_agent_turn`'s docstring names this index
    as the defense-in-depth that turns a residual race between two Database
    connections into an IntegrityError "which the caller catches and logs, rather
    than silently persisting a duplicate/colliding row". Drop UNIQUE and that
    documented backstop is gone: two turns share a turn_index, and
    `get_agent_turns` ORDER BY turn_index replays the ReAct history in a
    nondeterministic order.
    """

    def test_a_colliding_turn_index_is_rejected_by_the_schema(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "index.db")
        db.upsert_agent_session("s1", "why is login slow")

        insert = (
            "INSERT INTO agent_turns "
            "(session_id, turn_index, thought, action_type, action_arguments, "
            " observation_content, observation_source, observation_success) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        row = ("s1", 0, "thinking", "search", "{}", "found it", "vector", 1)
        db._conn.execute(insert, row)
        db._conn.commit()

        # Precondition naming the fixture: the first turn really is stored under
        # turn_index 0, so the raise below is the UNIQUE index and not a
        # different constraint firing on an empty table.
        assert [t["turn_index"] for t in db.get_agent_turns("s1")] == [0]

        with pytest.raises(sqlite3.IntegrityError):
            db._conn.execute(insert, row)
