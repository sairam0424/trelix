"""``tmp_db``: a real sqlite ``Database`` (src/trelix/store/db.py) rooted in a
fresh ``tmp_path``, function-scoped so every test gets its own on-disk file
and its own connection -- never shared, never reused across tests.

Deliberately NOT a fake or a Mock: ``Database.__init__`` opens a real sqlite3
connection and runs the real ``init_schema()`` DDL/migrations, so a test using
this fixture exercises the exact schema production code writes against.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from trelix.store.db import Database


@pytest.fixture
def tmp_db(tmp_path: Path) -> Iterator[Database]:
    """A real, freshly-initialized ``Database`` at ``tmp_path/index.db``.

    Function-scoped: each test gets its own file and its own sqlite3
    connection. Closed on teardown via the real ``Database.close()`` --
    not skipped, so a test that leaves the writer connection in a bad
    state (uncommitted transaction, locked file) surfaces as a teardown
    error rather than silently leaking into the next test.
    """
    db = Database(tmp_path / "index.db")
    try:
        yield db
    finally:
        db.close()
