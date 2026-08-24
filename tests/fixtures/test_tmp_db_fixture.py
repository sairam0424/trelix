"""Standalone proof that ``tmp_db`` (tests/fixtures/db.py) works in isolation,
BEFORE any call site migrates to it. Uses ONLY the ``tmp_db`` fixture --
no other fixture, no helper, so a future regression in ``tmp_db`` itself is
caught here first rather than surfacing as a confusing failure three files
away.

FALSIFIED BY: ``tmp_db`` failing to construct (bad schema init), or returning
a ``Database`` whose on-disk file does not exist, or whose connection is
already closed.
"""

from __future__ import annotations

from pathlib import Path

from tests.fixtures.db import tmp_db as tmp_db  # re-exported explicitly (ruff F401)
from trelix.store.db import Database


def test_tmp_db_is_a_real_open_database_backed_by_a_file_on_disk(tmp_db: Database) -> None:
    assert isinstance(tmp_db, Database)
    assert Path(tmp_db._db_path).exists(), "tmp_db must have created its sqlite file on disk"
    # A closed connection raises sqlite3.ProgrammingError on execute(); a live
    # PRAGMA query proves the connection is open and the schema was initialized.
    assert tmp_db.schema_version() >= 0


def test_a_row_inserted_here_must_not_survive_into_the_next_test(tmp_db: Database) -> None:
    """Insert a row via the real Database API. This test runs BEFORE
    test_tmp_db_is_a_fresh_instance_per_test_not_reused below (default pytest
    file-order execution) -- that next test's "must be empty" assertion is
    only a meaningful discriminator (rule 4) because THIS test guarantees
    there is a row for it to catch if `tmp_db` stops being function-scoped."""
    from trelix.core.models import IndexedFile, Language

    tmp_db.upsert_file(
        IndexedFile(
            path="/r/a.py", rel_path="a.py", language=Language.PYTHON, hash="x", size_bytes=1
        )
    )
    assert tmp_db._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1


def test_tmp_db_is_a_fresh_instance_per_test_not_reused(tmp_db: Database) -> None:
    """FALSIFIED BY: tmp_db being session/module-scoped and carrying state
    (e.g. the row test_a_row_inserted_here_must_not_survive_into_the_next_test
    just inserted) left behind by the previous test."""
    assert tmp_db._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0, (
        "a fresh tmp_db must start with an empty files table -- "
        "if this is nonzero, tmp_db is being reused across tests"
    )
