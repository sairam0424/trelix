"""Pins BOTH halves of the on-disk schema generation: the literal value and the
behaviour that value gates.

Why this file exists next to `tests/unit/test_db_structural.py`: every assertion
there is written in terms of `SCHEMA_VERSION` imported from the module under test
(`SCHEMA_VERSION + 1`, `== SCHEMA_VERSION`), so changing the constant to any other
integer moves the expectation with it and 13 tests still pass. A mutation of
`store/db.py:SCHEMA_VERSION` from `1` to `999` — which would make every index this
build writes unreadable by every released trelix, and would make an index stamped
`2` (a genuinely newer, unreadable generation) open silently and get *downgraded*
to `999` — survived the whole suite.

So `SCHEMA_VERSION` is NOT imported here. The expected value is the literal `1`,
written beside each assertion, and the stamp is read back off disk with a raw
`sqlite3` connection rather than through `Database.schema_version()` so that the
module's own accessor is not both the subject and the oracle.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from trelix.core.models import IndexedFile, Language
from trelix.store.db import Database, SchemaVersionError

# The current on-disk schema generation, hard-coded. Bump this ONLY together with
# store/db.py's SCHEMA_VERSION, and only for a change an older reader would
# MISREAD (see the "When to bump" note on that constant). A diff that changes one
# and not the other is the bug this file is here to catch.
CURRENT_SCHEMA_VERSION = 1
# The next generation up: what a newer trelix would stamp, which this build must
# refuse instead of misreading.
NEXT_SCHEMA_VERSION = 2


def _raw_user_version(db_path: Path) -> int:
    """Read `pragma user_version` off disk without going through Database."""
    raw = sqlite3.connect(str(db_path))
    try:
        return int(raw.execute("PRAGMA user_version").fetchone()[0])
    finally:
        raw.close()


def _raw_stamp(db_path: Path, version: int) -> None:
    """Stamp `pragma user_version` the way another trelix build would have."""
    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute(f"PRAGMA user_version = {int(version)}")
        raw.commit()
    finally:
        raw.close()


def _make_file(db: Database, rel_path: str = "a.py") -> int:
    return db.upsert_file(
        IndexedFile(
            path=f"/r/{rel_path}",
            rel_path=rel_path,
            language=Language.PYTHON,
            hash="h",
            size_bytes=10,
        )
    )


class TestSchemaVersionValueAndBehaviour:
    def test_fresh_index_is_stamped_with_the_literal_current_version(self, tmp_path: Path) -> None:
        """MUTATION THIS MUST FAIL ON: store/db.py `SCHEMA_VERSION = 1` -> any other
        integer (e.g. 999). Reads the stamp off disk with raw sqlite3 and compares it
        to the literal 1, so the expectation cannot travel with the constant.
        """
        db_path = tmp_path / "index.db"
        db = Database(db_path)
        db.close()

        assert _raw_user_version(db_path) == CURRENT_SCHEMA_VERSION

    def test_index_stamped_one_generation_ahead_is_refused_untouched(self, tmp_path: Path) -> None:
        """MUTATIONS THIS MUST FAIL ON:
        - store/db.py `SCHEMA_VERSION = 1` -> 999: version 2 then looks OLDER than the
          code's, so the open is allowed and `_upgrade_to_current()` restamps the file.
        - `_guard_schema_version`: `if on_disk > SCHEMA_VERSION` -> `<`/`!=`/removed.
        - `_guard_schema_version` raising with swapped found/supported.
        """
        db_path = tmp_path / "index.db"
        Database(db_path).close()
        _raw_stamp(db_path, NEXT_SCHEMA_VERSION)
        # Precondition: the fixture must actually present a NEWER index, otherwise
        # "no refusal" would be the correct behaviour and this test would be vacuous.
        assert _raw_user_version(db_path) == NEXT_SCHEMA_VERSION
        assert NEXT_SCHEMA_VERSION > CURRENT_SCHEMA_VERSION

        with pytest.raises(SchemaVersionError) as exc:
            Database(db_path)

        assert exc.value.found == NEXT_SCHEMA_VERSION
        assert exc.value.supported == CURRENT_SCHEMA_VERSION
        # Refused before touching it: the stamp on disk is unchanged, so a later
        # upgrade of trelix still sees its own generation.
        assert _raw_user_version(db_path) == NEXT_SCHEMA_VERSION

    def test_index_stamped_at_the_current_version_reopens(self, tmp_path: Path) -> None:
        """MUTATIONS THIS MUST FAIL ON:
        - `_guard_schema_version`: `if on_disk > SCHEMA_VERSION` -> `>=` (an index this
          build just wrote would then be refused by this build).
        - store/db.py `SCHEMA_VERSION = 1` -> 999 (the stamp assertion below).
        """
        db_path = tmp_path / "index.db"
        db = Database(db_path)
        _make_file(db)
        db.close()
        # Precondition: the file under test is stamped at the current generation --
        # not 0, which would take the pre-versioning upgrade path instead.
        assert _raw_user_version(db_path) == CURRENT_SCHEMA_VERSION

        reopened = Database(db_path)
        try:
            assert reopened.get_file_hash("a.py") == "h"
        finally:
            reopened.close()

        assert _raw_user_version(db_path) == CURRENT_SCHEMA_VERSION

    def test_unversioned_index_is_upgraded_to_the_literal_current_version(
        self, tmp_path: Path
    ) -> None:
        """MUTATIONS THIS MUST FAIL ON:
        - store/db.py `SCHEMA_VERSION = 1` -> any other integer (the stamp written by
          `_upgrade_to_current()` is compared to the literal 1).
        - `init_schema`: `if on_disk < SCHEMA_VERSION` -> `>`/`==`/removed, which
          leaves a pre-versioning index unstamped forever.
        """
        db_path = tmp_path / "index.db"
        db = Database(db_path)
        _make_file(db)
        db.close()
        _raw_stamp(db_path, 0)
        # Precondition: 0 means "written before versioning existed". If the fixture
        # stopped producing 0 the upgrade path would not be exercised at all.
        assert _raw_user_version(db_path) == 0

        reopened = Database(db_path)
        try:
            assert reopened.get_file_hash("a.py") == "h"
        finally:
            reopened.close()

        assert _raw_user_version(db_path) == CURRENT_SCHEMA_VERSION
