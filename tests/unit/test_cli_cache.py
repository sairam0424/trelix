"""`trelix cache gc` and `trelix cache clear`. (The two places `trelix index` shows the
on-disk embedding cache, the Index Summary row and the `--dry-run` preview after a real
cached run, are in test_cli_cache_index_rows.py.)

The commands read `TRELIX_EMBEDDING_CACHE_*` alone: no repository argument, no index
opened, nothing created. Every file here is built under `tmp_path` with the real
`EmbeddingCache`, the CLI runs through `CliRunner`, and no model is loaded.

Each test names the mutation(s) that must fail it.
"""

from __future__ import annotations

import os
import pathlib
import re
import sqlite3
from typing import Any

import pytest
from typer.testing import CliRunner

from trelix.indexing.embedding_cache import EmbeddingCache, text_key

runner = CliRunner()

_A = "a" * 32 + ".db"
_B = "b" * 32 + ".db"
_MIB = 1024 * 1024
_POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX file modes and symlinks")


def _invoke(*args: str) -> Any:
    from trelix.cli.main import app

    return runner.invoke(app, list(args))


def _squash(output: str) -> str:
    """Rich wraps at 80 columns off a terminal; collapse whitespace so a line under test
    is matched by content, not width."""
    return " ".join(output.split())


def _unwrapped(text: str) -> str:
    """For a whole-line comparison that contains a path: Rich folds a long path mid-word,
    so a space can land inside it. Dropping every space compares the content only."""
    return "".join(text.split())


def _cache_file(
    cache_dir: pathlib.Path, name: str, *, dimension: int, rows: int, batches: int = 1
) -> pathlib.Path:
    """A real cache file of `rows` vectors, written in `batches` groups at clocks 1..batches
    (so the oldest group is the first evicted and ties inside a group break by key)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    now = [0.0]
    cache = EmbeddingCache.open(cache_dir / name, dimension=dimension, clock=lambda: now[0])
    per_batch = rows // batches
    for batch in range(1, batches + 1):
        now[0] = float(batch)
        first = (batch - 1) * per_batch + 1
        cache.put_many(
            [
                (text_key(f"k_{i}"), [float(i % 7)] * dimension)
                for i in range(first, first + per_batch)
            ]
        )
    cache.close()
    return cache_dir / name


def _line_for(output: str, name: str) -> re.Match[str]:
    match = re.search(rf"{name}: (\d+) -> (\d+) rows, (\d+) -> (\d+) bytes", output)
    assert match, f"no gc line for {name} in:\n{output}"
    return match


# ---------------------------------------------------------------------------
# trelix cache gc
# ---------------------------------------------------------------------------


class TestCacheGc:
    def test_trims_the_file_over_the_cap_leaves_the_other_and_prints_both(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: `enforce_size_cap` not called (rows unchanged); the MB -> bytes
        conversion wrong (the 1.2 MB file is not trimmed at --max-mb 1, or the 20 KB one
        is); `--max-mb` ignored.

        File B is 5 SQLite pages of 4096 bytes whatever the platform: the literal is exact.
        """
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        _cache_file(cache_dir, _A, dimension=1000, rows=300)  # 300 x 4000 B > 1 MiB
        _cache_file(cache_dir, _B, dimension=4, rows=3)

        result = _invoke("cache", "gc", "--max-mb", "1")

        assert result.exit_code == 0, result.output
        output = _squash(result.output)
        a = _line_for(output, _A)
        assert int(a.group(1)) == 300
        assert 0 < int(a.group(2)) < 300
        assert int(a.group(3)) > _MIB
        assert int(a.group(4)) < int(a.group(3))
        assert f"{_B}: 3 -> 3 rows, 20480 -> 20480 bytes" in output

    def test_prints_the_files_in_name_order_whatever_the_directory_listing_order(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: `sorted()` dropped from the file listing (B prints before A here,
        because the listing is forced to reverse name order)."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        _cache_file(cache_dir, _A, dimension=4, rows=1)
        _cache_file(cache_dir, _B, dimension=4, rows=1)
        real_iterdir = pathlib.Path.iterdir
        monkeypatch.setattr(
            pathlib.Path, "iterdir", lambda self: iter(sorted(real_iterdir(self), reverse=True))
        )

        result = _invoke("cache", "gc")

        assert result.exit_code == 0, result.output
        output = _squash(result.output)
        assert output.index(f"{_A}:") < output.index(f"{_B}:"), output

    def test_the_default_cap_is_the_configured_max_mb(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: `--max-mb` unset falls back to a constant instead of
        `TRELIX_EMBEDDING_CACHE_MAX_MB` (nothing is trimmed at the 4096 MB default)."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_MAX_MB", "1")
        _cache_file(cache_dir, _A, dimension=1000, rows=300)

        result = _invoke("cache", "gc")

        assert result.exit_code == 0, result.output
        a = _line_for(_squash(result.output), _A)
        assert int(a.group(1)) == 300
        assert int(a.group(2)) < 300

    def test_a_max_mb_below_one_is_a_usage_error(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: `min=1` dropped (a cap of 0 MB would evict every row)."""
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(tmp_path / "cache"))
        _cache_file(tmp_path / "cache", _A, dimension=4, rows=1)

        result = _invoke("cache", "gc", "--max-mb", "0")

        assert result.exit_code == 2, result.output
        assert EmbeddingCache.open(tmp_path / "cache" / _A, dimension=4).row_count() == 1

    def test_an_unreadable_file_is_reported_the_rest_is_trimmed_and_the_exit_code_is_1(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: `raise typer.Exit(1)` after the loop dropped (exit 0); the loop stops at
        the bad file (no line for B); the error swallowed (no `Embedding cache unreadable`)."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        cache_dir.mkdir()
        (cache_dir / _A).write_bytes(b"not a database")
        _cache_file(cache_dir, _B, dimension=4, rows=3)

        result = _invoke("cache", "gc")

        assert result.exit_code == 1, result.output
        output = _squash(result.output)
        assert "Embedding cache unreadable:" in output
        assert "is not a trelix embedding cache" in output
        assert f"{_B}: 3 -> 3 rows" in output

    @_POSIX_ONLY
    @pytest.mark.skipif(os.geteuid() == 0, reason="root lists a directory with no permissions")
    def test_an_unlistable_cache_directory_is_reported_and_the_exit_code_is_1(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cache directory owned by another user (SECURITY.md's shared host): the exit-code
        table promises `Embedding cache unreadable`, exit 1, not a traceback.
        MUTATION: the `OSError` handler around the listing dropped (`PermissionError`
        escapes `CliRunner` instead of `typer.Exit`); its exit code not 1."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        _cache_file(cache_dir, _A, dimension=4, rows=1)
        cache_dir.chmod(0)
        try:
            result = _invoke("cache", "gc")
        finally:
            cache_dir.chmod(0o700)

        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit), result.exception
        output = _squash(result.output)
        assert "Embedding cache unreadable:" in output
        assert "Permission denied" in output
        assert f"{_A}:" not in output

    def test_only_fingerprint_named_files_are_opened(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only regular files whose whole name is a fingerprint are opened: not a `-journal`
        sidecar (a concurrent `trelix index` holds one), not a directory carrying a cache
        file's name (`clear` leaves it alone too), not a symlink to a directory (`clear`
        unlinks that one). MUTATION: the name filter loosened to `.*\\.db` (`notes.db` is
        opened, found foreign, exit 1); `fullmatch` -> `match` (the journal is opened the
        same way); the directory skip in `_cache_files` dropped; the `is_dir()` skip in
        `cache_gc` dropped (`os.open` on the link hits EISDIR: `unreadable`, exit 1)."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        _cache_file(cache_dir, _A, dimension=4, rows=1)
        (cache_dir / (_A + "-journal")).write_bytes(b"\x00" * 100)
        (cache_dir / _B).mkdir()
        if os.name != "nt":  # a symlink to a directory, kept by `_cache_files` for `clear`
            (cache_dir / ("c" * 32 + ".db")).symlink_to(cache_dir / _B, target_is_directory=True)
        (cache_dir / "notes.db").write_bytes(b"not a database")
        (cache_dir / ("c" * 31 + ".db")).write_bytes(b"not a database")  # 31 hex chars

        result = _invoke("cache", "gc")

        assert result.exit_code == 0, result.output
        assert f"{_A}: 1 -> 1 rows" in _squash(result.output)
        assert "notes.db" not in result.output and "journal" not in result.output
        assert "unreadable" not in result.output

    def test_gc_uses_the_recorded_dimension(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 4-wide file is trimmed without any width being passed: `gc` opens it with
        `dimension=None` and trusts `meta.dimension` (C6), so the width check cannot fire.
        Eviction is least recently used first, as the module's own cap test shows: the
        oldest batch (clock 1) goes entirely, the newest (clock 15) survives entirely, and
        the survivors still read back as 4-float vectors.

        MUTATION: `gc` opens with a fixed width (the width check refuses the file: exit 1);
        `enforce_size_cap` not called (30000 rows stay)."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        path = _cache_file(cache_dir, _A, dimension=4, rows=30_000, batches=15)
        size_before = path.stat().st_size
        assert size_before > 2 * _MIB, size_before  # so --max-mb 1 must evict

        result = _invoke("cache", "gc", "--max-mb", "1")

        assert result.exit_code == 0, result.output
        line = _line_for(_squash(result.output), _A)
        rows_after = int(line.group(2))
        assert int(line.group(1)) == 30_000
        assert 0 < rows_after < 30_000
        assert int(line.group(4)) < size_before // 2
        cache = EmbeddingCache.open(path, dimension=4)
        try:
            assert cache.row_count() == rows_after
            oldest = [text_key(f"k_{i}") for i in range(1, 2001)]
            newest = [text_key(f"k_{i}") for i in range(28_001, 30_001)]
            assert cache.get_many(oldest) == {}
            served = cache.get_many(newest)
            assert len(served) == 2000
            assert served[text_key("k_28001")] == [float(28_001 % 7)] * 4
        finally:
            cache.close()

    @pytest.mark.parametrize("removed", [False, pytest.param(True, marks=_POSIX_ONLY)])
    def test_a_failure_during_the_trim_is_reported_the_rest_is_trimmed_and_the_exit_code_is_1(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, removed: bool
    ) -> None:
        """Between `gc`'s `open()` and its trim, a concurrent `trelix index` can take the write
        lock (`enforce_size_cap` raises SQLite's own `OperationalError`) and a concurrent
        `trelix cache clear` can remove the file (`enforce_size_cap`'s own `stat` raises
        `FileNotFoundError`); the exit-code table promises `Embedding cache unreadable` for a
        file that could not be trimmed. Both races are staged at the trim: a real lock would
        wait out `open()`'s 30 s timeout and surface there. MUTATION: `sqlite3.Error` (locked) or
        `OSError` (removed) dropped from `gc`'s except tuple (a traceback, no `SystemExit`)."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        _cache_file(cache_dir, _A, dimension=4, rows=3)
        _cache_file(cache_dir, _B, dimension=4, rows=3)
        real_enforce = EmbeddingCache.enforce_size_cap

        def _failing_for_a(cache: EmbeddingCache, max_bytes: int) -> int:
            if cache.path.name == _A and not removed:
                raise sqlite3.OperationalError("database is locked")
            if cache.path.name == _A:
                cache.path.unlink()  # `clear` won the race; the real trim's `stat` raises
            return real_enforce(cache, max_bytes)

        monkeypatch.setattr(EmbeddingCache, "enforce_size_cap", _failing_for_a)

        result = _invoke("cache", "gc")

        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit), result.exception
        output = _squash(result.output)
        detail = "No such file or directory" if removed else "database is locked"
        assert "Embedding cache unreadable:" in output and detail in output, output
        assert f"{_A}:" not in output
        assert f"{_B}: 3 -> 3 rows" in output

    @_POSIX_ONLY
    def test_a_dangling_symlink_named_like_a_cache_file_creates_nothing_and_is_reported(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`gc` opens with `dimension=None`, and a link whose target is missing has nothing
        recorded to trust: it is refused before SQLite touches it, so no file appears at the
        target (outside the cache directory) and the directory cannot reach outside itself
        through `gc` any more than through `clear`.
        MUTATION: the refusal before `os.open` in `EmbeddingCache.open` dropped (a 0-byte
        file appears at the target; the exit code and the lines are unchanged)."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        _cache_file(cache_dir, _B, dimension=4, rows=3)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (cache_dir / _A).symlink_to(elsewhere / "target.db")

        result = _invoke("cache", "gc")

        assert result.exit_code == 1, result.output
        output = _squash(result.output)
        assert "Embedding cache unreadable:" in output
        assert "is not a trelix embedding cache" in output
        assert f"{_B}: 3 -> 3 rows" in output
        assert sorted(p.name for p in elsewhere.iterdir()) == []

    def test_help_for_the_group_and_both_commands(self) -> None:
        for args in (["cache"], ["cache", "gc"], ["cache", "clear"]):
            result = _invoke(*args, "--help")
            assert result.exit_code == 0, result.output
        assert "TRELIX_EMBEDDING_CACHE_MAX_MB" in _invoke("cache", "gc", "--help").output


# ---------------------------------------------------------------------------
# trelix cache clear
# ---------------------------------------------------------------------------


class TestCacheClear:
    def test_removes_cache_files_and_journals_only_and_prints_the_total(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the name filter widened to `*` (`notes.txt` goes) or loosened to
        `.*\\.db(-journal)?` (`notes.db` and the 31-hex `.db` go); the listing recurses
        (`sub/<fingerprint>.db` goes); the journal pattern dropped (2 -> 1); bytes not summed."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        a = _cache_file(cache_dir, _A, dimension=4, rows=3)
        journal = cache_dir / (_B + "-journal")
        journal.write_bytes(b"\x00" * 100)
        foreign = {"notes.txt": "keep me", "notes.db": "not a database", "c" * 31 + ".db": ""}
        for name, text in foreign.items():
            (cache_dir / name).write_text(text, encoding="utf-8")
        nested = cache_dir / "sub"
        nested.mkdir()
        _cache_file(nested, _A, dimension=4, rows=1)
        expected_bytes = a.stat().st_size + 100

        result = _invoke("cache", "clear")

        assert result.exit_code == 0, result.output
        assert _unwrapped(result.output) == _unwrapped(
            f"Removed 2 file(s), {expected_bytes} bytes, from {cache_dir}"
        )
        assert sorted(p.name for p in cache_dir.iterdir()) == sorted([*foreign, "sub"])
        assert (nested / _A).is_file()

    @_POSIX_ONLY
    def test_a_symlink_named_like_a_cache_file_is_unlinked_not_followed(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the target is deleted (`resolve()` before `unlink`, or `stat` for the
        size and a follow-through); a symlink is skipped as "not a file"; the directory skip
        applied to a symlink TO a directory (`and not path.is_symlink()` dropped: the link
        to `elsewhere/` is left behind and the count is 1)."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        target = tmp_path / "elsewhere.db"
        target.write_bytes(b"\x00" * 4096)
        link = cache_dir / _A
        link.symlink_to(target)
        target_dir = tmp_path / "elsewhere"
        target_dir.mkdir()
        (target_dir / "keep.db").write_bytes(b"\x00" * 10)
        dir_link = cache_dir / _B
        dir_link.symlink_to(target_dir, target_is_directory=True)
        link_bytes = link.lstat().st_size + dir_link.lstat().st_size

        result = _invoke("cache", "clear")

        assert result.exit_code == 0, result.output
        assert _unwrapped(result.output) == _unwrapped(
            f"Removed 2 file(s), {link_bytes} bytes, from {cache_dir}"
        )
        assert not link.is_symlink() and not link.exists()
        assert not dir_link.is_symlink() and not dir_link.exists()
        assert target.is_file() and target.stat().st_size == 4096
        assert (target_dir / "keep.db").stat().st_size == 10

    def test_a_directory_named_like_a_cache_file_is_left_alone(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the directory check dropped (`unlink` on a directory raises and the
        command exits 1)."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        (cache_dir / _A).mkdir(parents=True)
        _cache_file(cache_dir, _B, dimension=4, rows=1)

        result = _invoke("cache", "clear")

        assert result.exit_code == 0, result.output
        assert "Removed 1 file(s)," in result.output
        assert (cache_dir / _A).is_dir()

    @_POSIX_ONLY
    @pytest.mark.skipif(os.geteuid() == 0, reason="root unlinks regardless of directory mode")
    def test_a_file_that_cannot_be_removed_is_reported_and_the_exit_code_is_1(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the `OSError` handler dropped (a traceback instead of the line); the
        failure not reflected in the exit code."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        _cache_file(cache_dir, _A, dimension=4, rows=1)
        cache_dir.chmod(0o500)  # no write permission on the directory: unlink is refused
        try:
            result = _invoke("cache", "clear")
        finally:
            cache_dir.chmod(0o700)

        assert result.exit_code == 1, result.output
        output = _squash(result.output)
        assert "Could not remove:" in output
        assert "Removed 0 file(s), 0 bytes," in output
        assert (cache_dir / _A).is_file()

    @_POSIX_ONLY
    @pytest.mark.skipif(os.geteuid() == 0, reason="root lists a directory with no permissions")
    def test_an_unlistable_cache_directory_is_reported_and_the_exit_code_is_1(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """gc's shared-host case for `clear`: `Embedding cache unreadable`, exit 1, nothing
        removed and no `Removed N` line (the listing failed before any file was seen).
        MUTATION: the `OSError` handler around the listing dropped; its exit code not 1."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))
        _cache_file(cache_dir, _A, dimension=4, rows=1)
        cache_dir.chmod(0)
        try:
            result = _invoke("cache", "clear")
        finally:
            cache_dir.chmod(0o700)

        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit), result.exception
        output = _squash(result.output)
        assert "Embedding cache unreadable:" in output
        assert "Permission denied" in output
        assert "Removed" not in output
        assert (cache_dir / _A).is_file()

    def test_a_relative_cache_dir_is_a_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the pydantic `ValidationError` handler dropped (a traceback, exit 1 with
        no `Configuration error` line)."""
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", "relative/cache")

        result = _invoke("cache", "clear")

        assert result.exit_code == 1, result.output
        output = _squash(result.output)
        assert "Configuration error:" in output
        assert "must be an absolute path" in output


@pytest.mark.parametrize("command", ["gc", "clear"])
@pytest.mark.parametrize("occupant", ["nothing", "a regular file"])
def test_without_a_cache_directory_both_say_so_and_exit_0(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, command: str, occupant: str
) -> None:
    """Nothing at the configured path, or a regular file there (a misconfiguration the real
    run names; these two create nothing, so there is nothing to do). MUTATION: a missing
    directory exits 1, or is created; `is_dir()` -> `exists()` in either command (the file
    is listed: `Embedding cache unreadable: [Errno 20] Not a directory`, exit 1)."""
    path = tmp_path / "absent"
    if occupant == "a regular file":
        path.write_bytes(b"")
    monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(path))

    result = _invoke("cache", command)

    assert result.exit_code == 0, result.output
    assert _unwrapped(result.output) == _unwrapped(f"No embedding cache at {path}.")
    assert not path.is_dir() and path.exists() == (occupant == "a regular file")
