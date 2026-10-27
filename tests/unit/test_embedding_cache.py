"""`src/trelix/indexing/embedding_cache.py` as a module: fingerprint, directory, file.

Every expected value is a literal (the three fingerprints were computed by hand from the
canonical identity in the module docstring and must NOT be recomputed here) and the file
lives under `tmp_path`. The wrapper (`CachedIndexEmbedder`) is in
`test_embedding_cache_wrapper.py`, the Indexer wiring in `test_embedding_cache_indexer.py`.

Each test names the mutation(s) of the module that must fail it.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
from pathlib import Path

import pytest

from trelix.core.config import EmbedderConfig, EmbeddingCacheConfig
from trelix.indexing.embedding_cache import (
    _KNOB_FIELDS,
    EMBED_MODEL_FIELDS,
    EmbeddingCache,
    EmbeddingCacheError,
    embedder_fingerprint,
    prepare_cache_dir,
    resolve_cache_dir,
)

_POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")

_MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def _key(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _open(tmp_path: Path, dimension: int = 4, **kwargs: object) -> EmbeddingCache:
    cache_dir = prepare_cache_dir(tmp_path / "cache")
    return EmbeddingCache.open(cache_dir / "f.db", dimension=dimension, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestEmbedderFingerprint:
    def test_the_three_hand_computed_literals(self) -> None:
        """MUTATION: model omitted from the identity; the openai `dimensions` knob omitted;
        `sort_keys` removed; the schema number changed. Each changes every literal."""
        local = EmbedderConfig.model_construct(provider="local", local_model=_MINILM)
        assert embedder_fingerprint(local) == "9837970b2d56d811957572da7bae6138"
        wide = EmbedderConfig.model_construct(
            provider="openai", openai_model="text-embedding-3-large", openai_dimensions=3072
        )
        assert embedder_fingerprint(wide) == "d351f06f3c67eb27e31d81d5df871e0e"
        narrow = EmbedderConfig.model_construct(
            provider="openai", openai_model="text-embedding-3-large", openai_dimensions=1024
        )
        assert embedder_fingerprint(narrow) == "4aceb61bd433736da0c1c220eac81428"

    def test_stable_for_the_same_config_and_sensitive_to_the_model(self) -> None:
        """MUTATION: the model field dropped from the identity (both configs collide)."""
        a = EmbedderConfig.model_construct(provider="local", local_model=_MINILM)
        b = EmbedderConfig.model_construct(provider="local", local_model=_MINILM)
        other = EmbedderConfig.model_construct(provider="local", local_model="x")
        assert embedder_fingerprint(a) == embedder_fingerprint(b)
        assert embedder_fingerprint(other) != embedder_fingerprint(a)
        assert len(embedder_fingerprint(a)) == 32

    def test_azure_endpoint_changes_the_hash_but_never_appears_in_it(self) -> None:
        """MUTATION: `azure_endpoint` dropped from the azure knobs."""
        one = EmbedderConfig.model_construct(
            provider="azure", azure_endpoint="https://one.example/", azure_dimensions=3072
        )
        two = EmbedderConfig.model_construct(
            provider="azure", azure_endpoint="https://two.example/", azure_dimensions=3072
        )
        assert embedder_fingerprint(one) != embedder_fingerprint(two)
        assert "example" not in embedder_fingerprint(one)

    def test_every_provider_has_a_model_field(self) -> None:
        """MUTATION: a provider removed from EMBED_MODEL_FIELDS (its fingerprint would
        raise KeyError, or two models of that provider would share a file)."""
        assert set(EMBED_MODEL_FIELDS) == {
            "openai",
            "azure",
            "voyage",
            "local",
            "local-code",
            "bge-code",
            "nomic-code",
            "bedrock-titan",
            "bedrock-cohere",
            "cohere",
        }
        for provider, field in EMBED_MODEL_FIELDS.items():
            assert field in EmbedderConfig.model_fields, (provider, field)

    def test_the_width_knobs_are_real_fields_of_the_named_providers(self) -> None:
        """MUTATION: a knob tuple emptied (two widths of one model share a file); a knob
        misspelt (`getattr` raises at Indexer construction, only for that provider and
        only with the cache on)."""
        assert _KNOB_FIELDS == {
            "openai": ("openai_dimensions",),
            "azure": ("azure_dimensions", "azure_endpoint"),
            "voyage": ("voyage_output_dimensions",),
            "bedrock-titan": ("bedrock_titan_dimensions", "bedrock_titan_normalize"),
        }
        for provider, knobs in _KNOB_FIELDS.items():
            for name in knobs:
                assert name in EmbedderConfig.model_fields, (provider, name)


# ---------------------------------------------------------------------------
# Directory
# ---------------------------------------------------------------------------


class TestResolveCacheDir:
    def test_explicit_dir_wins(self) -> None:
        """MUTATION: `cfg.dir` ignored."""
        cfg = EmbeddingCacheConfig(dir=Path("/explicit"), _env_file=None)
        assert resolve_cache_dir(cfg, environ={"XDG_CACHE_HOME": "/x"}) == Path("/explicit")

    def test_xdg_cache_home_then_home_dot_cache(self) -> None:
        """MUTATION: the `embeddings` segment dropped; XDG_CACHE_HOME ignored; blank XDG
        treated as set."""
        cfg = EmbeddingCacheConfig(_env_file=None)
        assert resolve_cache_dir(cfg, environ={"XDG_CACHE_HOME": "/x"}) == Path(
            "/x/trelix/embeddings"
        )
        assert resolve_cache_dir(cfg, environ={}, home=Path("/h")) == Path(
            "/h/.cache/trelix/embeddings"
        )
        assert resolve_cache_dir(cfg, environ={"XDG_CACHE_HOME": " "}, home=Path("/h")) == Path(
            "/h/.cache/trelix/embeddings"
        )

    def test_a_relative_xdg_cache_home_is_ignored(self) -> None:
        """The XDG spec says a relative value is invalid and must be ignored; honoured, it
        would put the cache under the process cwd, i.e. inside the repository being indexed
        (`XDG_CACHE_HOME=.cache` is a common dotfile mistake). MUTATION: the `is_absolute()`
        check removed (returns `.cache/trelix/embeddings`)."""
        cfg = EmbeddingCacheConfig(_env_file=None)
        assert resolve_cache_dir(
            cfg, environ={"XDG_CACHE_HOME": ".cache"}, home=Path("/h")
        ) == Path("/h/.cache/trelix/embeddings")


class TestPrepareCacheDir:
    @_POSIX_ONLY
    def test_creates_the_leaf_0o700_and_is_idempotent(self, tmp_path: Path) -> None:
        """MUTATION: `mode=0o700` and the chmod removed (umask 022 gives 0o755)."""
        target = tmp_path / "deep" / "cache"
        assert prepare_cache_dir(target) == target
        assert target.is_dir()
        assert _mode(target) == 0o700
        os.chmod(target, 0o500)
        prepare_cache_dir(target)
        assert _mode(target) == 0o700

    def test_a_regular_file_at_the_path_is_refused(self, tmp_path: Path) -> None:
        """MUTATION: the OSError handler removed (FileExistsError escapes raw); the
        ValueError base of EmbeddingCacheError dropped (the Batch API refusal beside it in
        `Indexer._prepare_embedding_cache_dir` is a ValueError; one handler covers both)."""
        target = tmp_path / "cache"
        target.write_text("not a directory", encoding="utf-8")
        with pytest.raises(EmbeddingCacheError) as excinfo:
            prepare_cache_dir(target)
        assert isinstance(excinfo.value, ValueError)
        message = str(excinfo.value)
        assert message.startswith("embedding cache directory cannot be used: ")
        assert str(target) in message
        assert "TRELIX_EMBEDDING_CACHE_ENABLED=false" in message


# ---------------------------------------------------------------------------
# Cache file
# ---------------------------------------------------------------------------


class TestEmbeddingCacheFile:
    @_POSIX_ONLY
    def test_new_file_is_0o600(self, tmp_path: Path) -> None:
        """MUTATION: the chmod after os.open removed (umask 022 leaves 0o644 only when the
        O_CREAT mode is also widened; both together are the mutant)."""
        cache = _open(tmp_path)
        assert _mode(cache.path) == 0o600
        cache.close()

    def test_open_never_creates_the_directory(self, tmp_path: Path) -> None:
        """MUTATION: `path.parent.mkdir(...)` added to `open` (prepare_cache_dir is the
        only creator, so a typo'd TRELIX_EMBEDDING_CACHE_DIR cannot sprout directories)."""
        missing = tmp_path / "missing" / "f.db"
        with pytest.raises(EmbeddingCacheError) as excinfo:
            EmbeddingCache.open(missing, dimension=4)
        assert str(excinfo.value).startswith("embedding cache directory cannot be used: ")
        assert not missing.parent.exists()

    def test_put_then_get_returns_float32_exact_values_and_omits_absent_keys(
        self, tmp_path: Path
    ) -> None:
        """MUTATION: vectors stored as float16 or truncated; `get_many` returns None for a
        miss instead of leaving it out."""
        cache = _open(tmp_path)
        cache.put_many([(_key("a"), [0.5, 0.25, -1.0, 2.0])])
        got = cache.get_many([_key("a"), _key("b")])
        assert got == {_key("a"): [0.5, 0.25, -1.0, 2.0]}
        assert _key("b") not in got
        assert cache.row_count() == 1
        cache.close()

    def test_a_row_of_the_wrong_width_is_a_miss_and_is_overwritten(self, tmp_path: Path) -> None:
        """MUTATION: the blob-length check removed; `INSERT OR REPLACE` -> `INSERT OR IGNORE`."""
        cache = _open(tmp_path)
        cache.close()
        conn = sqlite3.connect(str(tmp_path / "cache" / "f.db"))
        conn.execute(
            "INSERT INTO embeddings(text_sha256, vector, last_used_at) VALUES (?, ?, 0)",
            (_key("c"), struct.pack("<3f", 1.0, 2.0, 3.0)),
        )
        conn.commit()
        conn.close()
        cache = _open(tmp_path)
        assert cache.get_many([_key("c")]) == {}
        cache.put_many([(_key("c"), [1.0, 2.0, 3.0, 4.0])])
        assert cache.get_many([_key("c")]) == {_key("c"): [1.0, 2.0, 3.0, 4.0]}
        cache.close()

    def test_put_many_refuses_a_vector_of_the_wrong_width(self, tmp_path: Path) -> None:
        """MUTATION: the length check in put_many removed (struct.error escapes instead)."""
        cache = _open(tmp_path)
        with pytest.raises(ValueError, match="expects 4-dimensional vectors"):
            cache.put_many([(_key("a"), [1.0, 2.0, 3.0])])
        assert cache.row_count() == 0
        cache.close()

    def test_reopening_at_another_width_names_both_widths(self, tmp_path: Path) -> None:
        """MUTATION: the meta.dimension check removed."""
        _open(tmp_path, dimension=4).close()
        path = tmp_path / "cache" / "f.db"
        with pytest.raises(EmbeddingCacheError) as excinfo:
            EmbeddingCache.open(path, dimension=8)
        assert str(excinfo.value) == (
            f"embedding cache {path} holds 4-dimensional vectors but the current embedder "
            "produces 8: the model behind this fingerprint changed. Delete that file and "
            "index again."
        )

    def test_a_sqlite_file_without_a_meta_table_is_foreign(self, tmp_path: Path) -> None:
        """MUTATION: the sqlite_master inspection replaced by CREATE TABLE IF NOT EXISTS
        (which would quietly adopt the foreign file)."""
        cache_dir = prepare_cache_dir(tmp_path / "cache")
        path = cache_dir / "f.db"
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        with pytest.raises(EmbeddingCacheError) as excinfo:
            EmbeddingCache.open(path, dimension=4)
        assert str(excinfo.value) == (
            f"{path} is not a trelix embedding cache (no meta table). Move it away."
        )

    def test_a_file_that_is_not_sqlite_is_foreign_too(self, tmp_path: Path) -> None:
        """MUTATION: the sqlite3.DatabaseError handler in `open` removed."""
        cache_dir = prepare_cache_dir(tmp_path / "cache")
        path = cache_dir / "f.db"
        path.write_bytes(b"not a database, and long enough to not look empty to sqlite")
        with pytest.raises(EmbeddingCacheError, match="is not a trelix embedding cache"):
            EmbeddingCache.open(path, dimension=4)

    def test_a_zero_byte_file_is_initialised(self, tmp_path: Path) -> None:
        """MUTATION: initialisation keyed on `not path.exists()` alone."""
        cache_dir = prepare_cache_dir(tmp_path / "cache")
        path = cache_dir / "f.db"
        path.touch()
        cache = EmbeddingCache.open(path, dimension=4)
        assert cache.row_count() == 0
        cache.close()
        conn = sqlite3.connect(str(path))
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
        conn.close()
        assert meta == {"schema_version": "1", "dimension": "4"}

    @_POSIX_ONLY
    def test_a_zero_byte_file_is_also_made_private(self, tmp_path: Path) -> None:
        """MUTATION: `is_new` keyed on `not path.exists()` alone (a pre-existing 0o644
        zero-byte file is initialised but keeps its mode)."""
        cache_dir = prepare_cache_dir(tmp_path / "cache")
        path = cache_dir / "f.db"
        path.touch(mode=0o644)
        assert _mode(path) == 0o644
        EmbeddingCache.open(path, dimension=4).close()
        assert _mode(path) == 0o600

    def test_another_schema_version_is_refused(self, tmp_path: Path) -> None:
        """MUTATION: the schema_version comparison removed."""
        _open(tmp_path).close()
        path = tmp_path / "cache" / "f.db"
        conn = sqlite3.connect(str(path))
        conn.execute("UPDATE meta SET value = '2' WHERE key = 'schema_version'")
        conn.commit()
        conn.close()
        with pytest.raises(EmbeddingCacheError) as excinfo:
            EmbeddingCache.open(path, dimension=4)
        assert str(excinfo.value) == (
            f"{path} is a trelix embedding cache with schema 2; this trelix reads schema 1. "
            "Delete that file and index again."
        )

    def test_size_cap_evicts_the_least_recently_used_half(self, tmp_path: Path) -> None:
        """MUTATION: `ORDER BY last_used_at` reversed (evicts the newest); the hit-time
        UPDATE skipped (k_1..k_10 evicted); VACUUM removed (size unchanged).

        `evicted == 50` is deterministic: the file is 120 pages of 4096 bytes, the cap 60,
        and the ten refreshes leave at most one index page on the freelist, so the live
        size is 119 or 120 pages and `60 * 100 // live_pages` is 50 either way.
        """
        now = [0.0]
        cache = _open(tmp_path, dimension=1000, clock=lambda: now[0])
        keys = {f"k_{i}": _key(f"k_{i}") for i in range(1, 101)}
        for i in range(1, 101):
            now[0] = float(i)
            cache.put_many([(keys[f"k_{i}"], [float(i)] * 1000)])
        now[0] = 200.0
        refreshed = cache.get_many([keys[f"k_{i}"] for i in range(1, 11)])
        assert len(refreshed) == 10
        size_before = cache.path.stat().st_size
        cap = size_before // 2

        evicted = cache.enforce_size_cap(cap)

        assert evicted == 50
        survivors = {name for name, key in keys.items() if cache.get_many([key])}
        assert survivors == {f"k_{i}" for i in range(1, 11)} | {f"k_{i}" for i in range(61, 101)}
        assert cache.path.stat().st_size < size_before
        cache.close()

    def test_a_hits_only_run_persists_the_refreshed_last_used_at(self, tmp_path: Path) -> None:
        """The production warm run never writes a row: every lookup hits, so `put_many`
        returns before its commit, `enforce_size_cap` returns at the `st_size` check before
        any statement, and the Indexer never calls `close()`. The hit-time UPDATE must
        therefore commit on its own, or the connection's end rolls it back and the next
        trim evicts the rows that were just used. MUTATION: the commit after the hit-time
        UPDATE in get_many removed (the stamp reads 10, not 500; every other LRU test
        commits the refresh as a side effect of a later `put_many` or DELETE)."""
        now = [10.0]
        cache = _open(tmp_path, clock=lambda: now[0])
        cache.put_many([(_key("a"), [1.0, 2.0, 3.0, 4.0])])
        cache.close()
        path = tmp_path / "cache" / "f.db"

        now[0] = 500.0
        cache = EmbeddingCache.open(path, dimension=4, clock=lambda: now[0])
        assert cache.get_many([_key("a")]) == {_key("a"): [1.0, 2.0, 3.0, 4.0]}
        assert cache.enforce_size_cap(path.stat().st_size * 10) == 0
        cache.close()

        conn = sqlite3.connect(str(path))
        (stamp,) = conn.execute("SELECT last_used_at FROM embeddings").fetchone()
        conn.close()
        assert stamp == 500

    def test_size_cap_within_the_cap_evicts_nothing(self, tmp_path: Path) -> None:
        """MUTATION: the `st_size` return removed (a file within the cap is still VACUUMed,
        and its pages counted, after every run: no statement may run). The `live <=
        max_bytes` return behind it is only reachable when the file is over the cap and its
        live data under it; `test_free_pages_left_by_a_failed_vacuum_are_not_evicted_again`
        covers that one. (`<=` -> `<` is not caught here: equal sizes are page-aligned
        coincidences.)"""
        cache = _open(tmp_path)
        cache.put_many([(_key("a"), [1.0, 2.0, 3.0, 4.0])])
        statements: list[str] = []
        cache._conn.set_trace_callback(statements.append)
        assert cache.enforce_size_cap(cache.path.stat().st_size * 10) == 0
        assert statements == []
        assert cache.row_count() == 1
        cache.close()

    def test_free_pages_left_by_a_failed_vacuum_are_not_evicted_again(self, tmp_path: Path) -> None:
        """A trim commits its DELETE before its VACUUM. When that VACUUM failed (another
        indexer's lock, which the Indexer logs) the file keeps the freed pages, so its size
        still exceeds the cap although its live data is under it. Here: 100 rows of 4000
        bytes, 50 deleted and not vacuumed, leaves 59 of 121 pages free (live 253952 of
        495616 bytes); the cap, three quarters of the file, lies between the two.
        MUTATION: the cap measured against `st_size` instead of the live pages (13 of the
        50 remaining rows are evicted, and again on every later run while the VACUUM keeps
        failing); the `live <= max_bytes` return removed (`evicted = 50 - 371712 * 50 //
        253952 = -23`, and a negative LIMIT means "no limit" in SQLite: every row goes and
        -23 comes back); the VACUUM skipped when nothing is evicted (the file stays above
        the cap).
        """
        cache = _open(tmp_path, dimension=1000)
        cache.put_many([(_key(f"k_{i}"), [1.0] * 1000) for i in range(1, 101)])
        cache.close()
        path = tmp_path / "cache" / "f.db"
        conn = sqlite3.connect(str(path))
        conn.execute(
            "DELETE FROM embeddings WHERE text_sha256 IN ("
            "SELECT text_sha256 FROM embeddings ORDER BY text_sha256 LIMIT 50)"
        )
        conn.commit()
        assert conn.execute("PRAGMA freelist_count").fetchone()[0] == 59
        conn.close()
        bloated = path.stat().st_size
        assert bloated == 495616
        cap = bloated * 3 // 4

        cache = EmbeddingCache.open(path, dimension=1000)
        assert cache.enforce_size_cap(cap) == 0
        assert cache.row_count() == 50
        assert path.stat().st_size == 249856
        cache.close()

    def test_two_first_openers_of_one_new_file_both_succeed(self, tmp_path: Path) -> None:
        """Two indexers that both inspected the same empty file before either wrote it (a
        first index on two CI runners, or in two worktrees, at the same moment). MUTATION:
        `INSERT OR IGNORE INTO meta` -> `INSERT INTO meta` (the second initialiser raises
        IntegrityError, which `open` would report as "not a trelix embedding cache")."""
        cache_dir = prepare_cache_dir(tmp_path / "cache")
        path = cache_dir / "f.db"
        path.touch()
        first = sqlite3.connect(str(path), timeout=30)
        second = sqlite3.connect(str(path), timeout=30)
        for conn in (first, second):
            assert conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0
        for conn in (first, second):
            EmbeddingCache._initialise(conn, 4)
            conn.commit()
            conn.close()
        cache = EmbeddingCache.open(path, dimension=4)
        assert cache.dimension == 4
        assert cache.row_count() == 0
        cache.close()

    def test_the_schema_and_the_meta_rows_land_in_one_transaction(self, tmp_path: Path) -> None:
        """A concurrent opener can never see the tables without `meta` (it would report
        "is a trelix embedding cache with schema None"). An observer connection counts the
        tables it can see at the moment each meta row is inserted. MUTATION: the
        `BEGIN IMMEDIATE` in `_verify_or_initialise` removed (each CREATE then commits on
        its own and the observer sees 2 tables)."""
        cache_dir = prepare_cache_dir(tmp_path / "cache")
        path = cache_dir / "f.db"
        path.touch()
        tables_seen_at_insert: list[int] = []

        def observe(statement: str) -> None:
            if statement.lstrip().upper().startswith("INSERT"):
                other = sqlite3.connect(str(path), timeout=30)
                tables_seen_at_insert.append(
                    other.execute(
                        "SELECT count(*) FROM sqlite_master WHERE type='table'"
                    ).fetchone()[0]
                )
                other.close()

        conn = sqlite3.connect(str(path), timeout=30)
        conn.set_trace_callback(observe)
        EmbeddingCache._verify_or_initialise(conn, path, 4)
        conn.close()
        assert tables_seen_at_insert == [0, 0]  # one INSERT per meta row, nothing visible yet
        cache = EmbeddingCache.open(path, dimension=4)
        assert cache.row_count() == 0
        cache.close()

    @_POSIX_ONLY
    def test_a_file_sqlite_cannot_write_reports_sqlite_s_reason(self, tmp_path: Path) -> None:
        """A directory SQLite cannot create the `-journal` in (as a lock held past the
        timeout would be) is not a foreign file, and the advice must not be "move it away".
        MUTATION: the `sqlite3.OperationalError` handler in `open` removed (the error falls
        through to the DatabaseError handler and names a perfectly good file as foreign)."""
        cache_dir = prepare_cache_dir(tmp_path / "cache")
        path = cache_dir / "f.db"
        path.touch()
        os.chmod(cache_dir, 0o500)
        try:
            with pytest.raises(EmbeddingCacheError) as excinfo:
                EmbeddingCache.open(path, dimension=4)
        finally:
            os.chmod(cache_dir, 0o700)
        assert str(excinfo.value) == (
            f"embedding cache {path} cannot be opened (attempt to write a readonly database). "
            "If another indexer holds it, retry; otherwise fix the directory or set "
            "TRELIX_EMBEDDING_CACHE_ENABLED=false."
        )


# ---------------------------------------------------------------------------
# The two openers without an embedder: `open(dimension=None)` (gc) and
# `open_readonly` (--dry-run)
# ---------------------------------------------------------------------------


def _last_used_at(path: Path, key: bytes) -> int:
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute(
            "SELECT last_used_at FROM embeddings WHERE text_sha256 = ?", (key,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return int(row[0])


class TestOpeningAtTheRecordedWidth:
    def test_dimension_none_trusts_the_recorded_width(self, tmp_path: Path) -> None:
        """MUTATION: `None` compared as a width (the file is refused as `4` vs `None`);
        the recorded width not returned (the cache reads 4-float rows at another width and
        reports a miss)."""
        cache = _open(tmp_path, dimension=4)
        cache.put_many([(_key("a"), [0.5, 0.25, -1.0, 2.0])])
        cache.close()

        reopened = EmbeddingCache.open(cache.path, dimension=None)
        try:
            assert reopened.dimension == 4
            assert reopened.get_many([_key("a")]) == {_key("a"): [0.5, 0.25, -1.0, 2.0]}
        finally:
            reopened.close()

    def test_dimension_none_cannot_initialise_an_empty_file(self, tmp_path: Path) -> None:
        """Nothing is recorded in a 0-byte file, so there is no width to trust; `gc`
        reports it rather than inventing one. MUTATION: both refusals dropped, the one
        before `os.open` and the no-tables one (the file is initialised with a
        `meta.dimension` of `None`); the first alone is caught by test_cli_cache.py's
        dangling-symlink test, the second alone by the no-tables test below."""
        cache_dir = prepare_cache_dir(tmp_path / "cache")
        path = cache_dir / "f.db"
        path.touch()
        with pytest.raises(EmbeddingCacheError) as excinfo:
            EmbeddingCache.open(path, dimension=None)
        assert str(excinfo.value) == (
            f"{path} is not a trelix embedding cache (no meta table). Move it away."
        )
        assert path.stat().st_size == 0

    def test_dimension_none_cannot_initialise_a_file_with_no_tables(self, tmp_path: Path) -> None:
        """A SQLite file with a header but no tables (`PRAGMA user_version` on a new file
        writes page 1) is not empty, so the refusal before `os.open` does not fire; the
        no-tables branch must refuse too, before it writes a `meta.dimension` of `None`
        into the file. MUTATION: that branch initialises with `None` (the file gains the
        tables and a meta row; the error text alone cannot tell, so the tables are checked)."""
        cache_dir = prepare_cache_dir(tmp_path / "cache")
        path = cache_dir / "f.db"
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()
        assert path.stat().st_size > 0
        with pytest.raises(EmbeddingCacheError) as excinfo:
            EmbeddingCache.open(path, dimension=None)
        assert str(excinfo.value) == (
            f"{path} is not a trelix embedding cache (no meta table). Move it away."
        )
        conn = sqlite3.connect(str(path))
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
        assert tables == []

    def test_open_readonly_is_none_for_a_missing_file(self, tmp_path: Path) -> None:
        """MUTATION: the missing file created (`open` semantics) or raised on; `is_file()`
        loosened to `exists()` (a directory at the path is opened and raises)."""
        cache_dir = prepare_cache_dir(tmp_path / "cache")
        assert EmbeddingCache.open_readonly(cache_dir / "missing.db") is None
        (cache_dir / "d.db").mkdir()
        assert EmbeddingCache.open_readonly(cache_dir / "d.db") is None
        assert sorted(p.name for p in cache_dir.iterdir()) == ["d.db"]

    def test_open_readonly_serves_hits_at_the_recorded_width_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        """MUTATION: `mode=ro` dropped (the `put_many` below succeeds); the read-only flag
        ignored (`get_many` tries the `last_used_at` refresh and raises on the read-only
        connection); the width taken from anywhere but `meta.dimension`."""
        now = [10.0]
        cache = _open(tmp_path, dimension=4, clock=lambda: now[0])
        cache.put_many([(_key("a"), [0.5, 0.25, -1.0, 2.0])])
        cache.close()
        assert _last_used_at(cache.path, _key("a")) == 10

        readonly = EmbeddingCache.open_readonly(cache.path)
        assert readonly is not None
        try:
            assert readonly.dimension == 4
            assert readonly.get_many([_key("a"), _key("b")]) == {_key("a"): [0.5, 0.25, -1.0, 2.0]}
            assert _last_used_at(cache.path, _key("a")) == 10
            with pytest.raises(sqlite3.OperationalError, match="readonly database"):
                readonly.put_many([(_key("b"), [1.0, 1.0, 1.0, 1.0])])
        finally:
            readonly.close()
        assert _last_used_at(cache.path, _key("a")) == 10

    def test_open_readonly_refuses_a_foreign_file_by_name(self, tmp_path: Path) -> None:
        """MUTATION: the `meta` check dropped (an `OperationalError: no such table` escapes
        instead of `EmbeddingCacheError`); the `DatabaseError` handler dropped."""
        cache_dir = prepare_cache_dir(tmp_path / "cache")
        not_sqlite = cache_dir / "a.db"
        not_sqlite.write_bytes(b"not a database")
        with pytest.raises(EmbeddingCacheError) as excinfo:
            EmbeddingCache.open_readonly(not_sqlite)
        assert str(excinfo.value) == (
            f"{not_sqlite} is not a trelix embedding cache (no meta table). Move it away."
        )

        no_meta = cache_dir / "b.db"
        conn = sqlite3.connect(str(no_meta))
        conn.execute("CREATE TABLE other (x INTEGER)")
        conn.commit()
        conn.close()
        with pytest.raises(EmbeddingCacheError) as excinfo:
            EmbeddingCache.open_readonly(no_meta)
        assert str(excinfo.value) == (
            f"{no_meta} is not a trelix embedding cache (no meta table). Move it away."
        )

    def test_open_readonly_encodes_the_path_into_the_uri(self, tmp_path: Path) -> None:
        """A `?` or `#` in the directory name would otherwise end the URI's query string and
        drop `mode=ro`. MUTATION: the raw path interpolated (the file at `dir?x/f.db` is not
        found, or is opened read-write)."""
        cache_dir = prepare_cache_dir(tmp_path / "odd?name#here")
        cache = EmbeddingCache.open(cache_dir / "f.db", dimension=4)
        cache.put_many([(_key("a"), [1.0, 2.0, 3.0, 4.0])])
        cache.close()

        readonly = EmbeddingCache.open_readonly(cache.path)
        assert readonly is not None
        try:
            assert readonly.get_many([_key("a")]) == {_key("a"): [1.0, 2.0, 3.0, 4.0]}
            with pytest.raises(sqlite3.OperationalError, match="readonly database"):
                readonly.put_many([(_key("b"), [1.0, 1.0, 1.0, 1.0])])
        finally:
            readonly.close()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file with no permissions")
    @pytest.mark.parametrize("unreadable", ["file", "directory"])
    def test_open_readonly_reports_a_file_it_cannot_read_as_unusable_not_foreign(
        self, tmp_path: Path, unreadable: str
    ) -> None:
        """SECURITY.md's shared-host case: a valid cache file, or its directory, owned by
        another user. The preview must get `EmbeddingCacheError` carrying SQLite's or the
        OS's own words and the retry/fix advice — not a raw `OperationalError` from
        `connect` or `PermissionError` from `is_file()`, and not "Move it away" (the file
        may well be a valid cache).

        MUTATION: `is_file()`/`connect()` outside the `try` (the raw exception escapes);
        the handler mapped to `_NOT_A_CACHE`."""
        cache = _open(tmp_path, dimension=4)
        cache.put_many([(_key("a"), [0.5, 0.25, -1.0, 2.0])])
        cache.close()
        locked = cache.path if unreadable == "file" else cache.path.parent
        locked.chmod(0)
        try:
            with pytest.raises(EmbeddingCacheError) as excinfo:
                EmbeddingCache.open_readonly(cache.path)
        finally:
            locked.chmod(0o600 if unreadable == "file" else 0o700)
        message = str(excinfo.value)
        assert message.startswith(f"embedding cache {cache.path} cannot be opened (")
        assert message.endswith(
            "). If another indexer holds it, retry; otherwise fix the directory or set "
            "TRELIX_EMBEDDING_CACHE_ENABLED=false."
        )
        assert "Move it away" not in message

    def test_open_readonly_reports_a_locked_file_as_unusable_not_foreign(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`database is locked` past SQLite's timeout is an `OperationalError` from the first
        query, as it is in `open`; the advice is retry, never "Move it away". The lock is
        simulated (a real one costs the connection's 5s timeout per test).

        MUTATION: the `OperationalError` branch dropped (falls into `DatabaseError`)."""
        cache = _open(tmp_path, dimension=4)
        cache.close()

        def _locked(_conn: sqlite3.Connection) -> set[str]:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(EmbeddingCache, "_table_names", staticmethod(_locked))
        with pytest.raises(EmbeddingCacheError) as excinfo:
            EmbeddingCache.open_readonly(cache.path)
        assert str(excinfo.value) == (
            f"embedding cache {cache.path} cannot be opened (database is locked). If another "
            "indexer holds it, retry; otherwise fix the directory or set "
            "TRELIX_EMBEDDING_CACHE_ENABLED=false."
        )

    def test_a_meta_dimension_that_is_not_a_number_is_refused_not_a_traceback(
        self, tmp_path: Path
    ) -> None:
        """`open(dimension=None)` and `open_readonly` trust `meta.dimension`; a row that is
        not a number, or no row at all, must be the foreign-file line `gc` and `--dry-run`
        handle, not the `ValueError`/`TypeError` of `int()`, which neither handler catches.
        MUTATION: the `isdecimal` guard dropped (`int('abc')` escapes); loosened to `isdigit`
        (SUPERSCRIPT TWO is a digit to `str` but not to `int()`: `int('²')` escapes); only its
        `stored is None` half dropped (`None.isdecimal()` escapes)."""
        _open(tmp_path).close()
        path = tmp_path / "cache" / "f.db"
        expected = f"{path} is not a trelix embedding cache (no meta table). Move it away."
        for statement in (
            "UPDATE meta SET value = 'abc' WHERE key = 'dimension'",
            "UPDATE meta SET value = '²' WHERE key = 'dimension'",
            "DELETE FROM meta WHERE key = 'dimension'",
        ):
            conn = sqlite3.connect(str(path))
            conn.execute(statement)
            conn.commit()
            conn.close()
            with pytest.raises(EmbeddingCacheError) as excinfo:
                EmbeddingCache.open(path, dimension=None)
            assert str(excinfo.value) == expected, statement
            with pytest.raises(EmbeddingCacheError) as excinfo:
                EmbeddingCache.open_readonly(path)
            assert str(excinfo.value) == expected, statement
