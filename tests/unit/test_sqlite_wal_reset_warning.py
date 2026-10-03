"""Pins the WAL-reset warning: which SQLite versions it covers, and when it is logged.

SQLite 3.7.0 through 3.51.2 can lose committed WAL writes when two or more connections,
in different threads or processes, write or checkpoint at the same instant. The fix is in
3.51.3 and was backported to 3.50.7 and 3.44.6. `Database` logs one WARNING per process
when it opens as a writer on an affected build.

The boundary table is written out as literals on purpose. Deriving the expected answer from
the module's own numbers would pass for any boundary at all, which is the defect
tests/unit/test_no_imported_expected_values_in_tests.py exists to keep out.

The once-per-process flag is re-armed with `monkeypatch.setattr(db_module, "_wal_reset_warned",
False)`, the seam documented beside the flag in store/db.py. monkeypatch puts the previous
value back afterwards, so a test here cannot change what a later test observes.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

import trelix.store.db as db_module
from trelix.store.db import Database, SchemaVersionError, wal_reset_risk

_LOGGER_NAME = "trelix.store.db"

# upstream version -> is it in the WAL-reset range
_VERSION_TABLE = {
    "3.6.23": False,  # WAL does not exist yet
    "3.7.0": True,  # first WAL release
    "3.44.5": True,
    "3.44.6": False,  # backported fix
    "3.44.9": False,
    "3.45.0": True,
    "3.46.1": True,  # Debian trixie's upstream number
    "3.50.4": True,
    "3.50.6": True,
    "3.50.7": False,  # backported fix
    "3.50.9": False,
    "3.51.0": True,
    "3.51.2": True,
    "3.51.3": False,  # the upstream fix
    "3.52.0": False,
    "3.53.4": False,
}


def _as_tuple(dotted: str) -> tuple[int, ...]:
    return tuple(int(part) for part in dotted.split("."))


def _pretend_linked_sqlite(monkeypatch: pytest.MonkeyPatch, dotted: str) -> None:
    monkeypatch.setattr(sqlite3, "sqlite_version_info", _as_tuple(dotted))
    monkeypatch.setattr(sqlite3, "sqlite_version", dotted)


def _rearm_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db_module, "_wal_reset_warned", False)


def _open_and_close(path: Path, *, read_only: bool = False) -> None:
    Database(path, read_only=read_only).close()


def _wal_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == _LOGGER_NAME and record.levelno == logging.WARNING
    ]


@pytest.fixture(scope="module", autouse=True)
def _warning_unspent_before_the_conftest_fixture_runs() -> Iterator[None]:
    """Leave the once-flag unspent before tests/unit/conftest.py's fixture sets it.

    pytest sets up higher-scoped fixtures first, so this runs ahead of the function-scoped
    `_wal_reset_warning_already_emitted`. Without it, a True flag at the start of a test could
    come from that fixture or from an earlier test that really opened a writer on an affected
    SQLite, and the test below could not tell the two apart. The previous value is put back
    when this module finishes.
    """
    with pytest.MonkeyPatch.context() as module_patch:
        module_patch.setattr(db_module, "_wal_reset_warned", False)
        yield


@pytest.mark.parametrize(("dotted", "expected"), list(_VERSION_TABLE.items()))
def test_wal_reset_risk_boundaries(dotted: str, expected: bool) -> None:
    assert wal_reset_risk(_as_tuple(dotted)) is expected


def test_unit_suite_starts_each_test_with_the_warning_already_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """tests/unit/conftest.py marks the process-wide warning spent, so no other test sees it."""
    # The conftest fixture's effect, read before this test touches the flag. The module fixture
    # above left it unspent, so True here can only have come from that fixture.
    assert db_module._wal_reset_warned is True

    _pretend_linked_sqlite(monkeypatch, "3.46.1")  # affected, and deliberately NOT re-armed

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _open_and_close(tmp_path / "index.db")

    assert _wal_warnings(caplog) == []


def test_writer_on_an_affected_build_logs_one_warning_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _pretend_linked_sqlite(monkeypatch, "3.46.1")
    _rearm_warning(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _open_and_close(tmp_path / "first" / "index.db")
        first = _wal_warnings(caplog)
        caplog.clear()
        _open_and_close(tmp_path / "second" / "index.db")
        second = _wal_warnings(caplog)

    assert len(first) == 1
    assert second == []
    message = first[0].getMessage()
    assert "linked SQLite is 3.46.1" in message
    assert "3.7.0 through 3.51.2" in message
    assert "fixed in 3.51.3" in message
    assert "3.50.7" in message
    assert "3.44.6" in message
    assert "distro build may carry the fix" in message
    assert "not a confirmed defect" in message
    assert "only one process that writes to the index at a time" in message
    assert "upgrade Python or SQLite" in message


@pytest.mark.parametrize("fixed", ["3.44.6", "3.50.7", "3.51.3", "3.53.4"])
def test_writer_on_a_fixed_build_is_silent(
    fixed: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _pretend_linked_sqlite(monkeypatch, fixed)
    _rearm_warning(monkeypatch)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _open_and_close(tmp_path / "index.db")

    assert _wal_warnings(caplog) == []


def test_read_only_open_is_silent_and_does_not_spend_the_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    db_path = tmp_path / "index.db"
    _pretend_linked_sqlite(monkeypatch, "3.53.4")
    _open_and_close(db_path)  # a writer on a fixed build creates the file, silently

    _pretend_linked_sqlite(monkeypatch, "3.46.1")
    _rearm_warning(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        _open_and_close(db_path, read_only=True)
        after_read_only = _wal_warnings(caplog)
        _open_and_close(db_path)
        after_writer = _wal_warnings(caplog)

    assert after_read_only == []
    assert len(after_writer) == 1


def test_an_open_that_is_refused_does_not_spend_the_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    newer = tmp_path / "newer.db"
    raw = sqlite3.connect(str(newer))
    try:
        raw.execute("PRAGMA user_version = 2")
        raw.commit()
    finally:
        raw.close()

    _pretend_linked_sqlite(monkeypatch, "3.46.1")
    _rearm_warning(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        with pytest.raises(SchemaVersionError):
            Database(newer)
        after_refusal = _wal_warnings(caplog)
        _open_and_close(tmp_path / "fresh.db")
        after_good_open = _wal_warnings(caplog)

    assert after_refusal == []
    assert len(after_good_open) == 1
