"""The embedding cache wired into a REAL `Indexer`: real walker, parser, sqlite-vec store.

Only the embedder is substituted (`CountingEmbedder`, tests/fixtures/fakes.py), because
the subject here is what the shipped write path does with the cache: which calls reach
the provider, what bytes land in the store, what the run reports, and what is refused
before any model is loaded. The module itself is covered in `test_embedding_cache.py`.

Harness per `test_indexer_accounting_and_provenance._make_indexer`. The cache is turned on
through the environment (`TRELIX_EMBEDDING_CACHE_ENABLED`/`_DIR`) via monkeypatch, which
is also how the hosted GitHub App's child process would receive it (it forwards every
`TRELIX_*` host variable), so the "App-style" tests below are the same mechanism.

Each test names the mutation(s) that must fail it.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sqlite3
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import sqlite_vec
from typer.testing import CliRunner

from tests._env_isolation import BEAST_MODE_DEFAULTS
from tests.fixtures.fakes import CountingEmbedder
from trelix.core.config import EmbedderConfig, IndexConfig, StoreConfig
from trelix.indexing.embedding_cache import CachedIndexEmbedder, EmbeddingCache, EmbeddingCacheError

_DIM = 4
_POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")

_FILES = {
    "alpha.py": 'def alpha(x: int) -> int:\n    """Add one."""\n    return x + 1\n',
    "beta.py": 'def beta(y: int) -> int:\n    """Double."""\n    return y * 2\n',
    "gamma.py": (
        'class Gamma:\n    """Holds a value."""\n\n    def get(self) -> int:\n        return 3\n'
    ),
}

_E2 = (
    "--use-batch-api / TRELIX_USE_BATCH_API cannot be combined with "
    "TRELIX_EMBEDDING_CACHE_ENABLED=true: the Batch API path submits texts to OpenAI "
    "without consulting the cache. Disable one of the two."
)


def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for name, body in _FILES.items():
        (repo / name).write_text(body, encoding="utf-8")
    return repo


def _enable_cache(monkeypatch: pytest.MonkeyPatch, cache_dir: pathlib.Path) -> None:
    monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_ENABLED", "true")
    monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))


def _config(
    repo: pathlib.Path,
    *,
    provider: str = "local",
    local_model: str | None = None,
    streaming: bool = False,
) -> IndexConfig:
    fields: dict[str, Any] = {"provider": provider}
    if local_model is not None:
        fields["local_model"] = local_model
    cfg = IndexConfig(
        repo_path=str(repo),
        incremental=True,
        store=StoreConfig(db_path=str(repo / ".trelix" / "index.db")),
        embedder=EmbedderConfig.model_construct(**fields),
    )
    if streaming:
        cfg.indexer.streaming_enabled = True
    return cfg


def _indexer(cfg: IndexConfig, embedder: CountingEmbedder) -> Any:
    from trelix.indexing.indexer import Indexer
    from trelix.store.vector import VectorStore

    real_store = VectorStore(cfg.db_path_absolute, dimension=_DIM)
    with (
        patch("trelix.indexing.indexer.make_embedder", return_value=embedder),
        patch("trelix.indexing.indexer.make_vector_store", return_value=real_store),
    ):
        return Indexer(cfg, quiet=True)


@contextmanager
def _quiet_progress():  # type: ignore[no-untyped-def]
    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task = MagicMock(return_value=0)
    mock_progress.advance = MagicMock()
    with patch("trelix.cli.progress.Progress", return_value=mock_progress):
        yield mock_progress


def _run(indexer: Any) -> dict[str, Any]:
    with _quiet_progress():
        stats: dict[str, Any] = indexer.index()
    return stats


def _fresh_db(repo: pathlib.Path) -> None:
    for path in (repo / ".trelix").glob("index.db*"):
        path.unlink()


def _blobs(repo: pathlib.Path) -> dict[str, bytes]:
    """{chunk_text: stored float32 bytes}, read with a plain sqlite3 connection."""
    conn = sqlite3.connect(str(repo / ".trelix" / "index.db"))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    try:
        rows = conn.execute(
            "SELECT c.chunk_text, e.embedding FROM chunks c "
            "JOIN chunk_embeddings e ON e.chunk_id = c.id"
        ).fetchall()
    finally:
        conn.close()
    return {text: bytes(blob) for text, blob in rows}


def _mode(path: pathlib.Path) -> int:
    return path.stat().st_mode & 0o777


# ---------------------------------------------------------------------------
# Hits, misses and bytes
# ---------------------------------------------------------------------------


class TestCacheHitsAndBytes:
    def test_second_fresh_index_makes_no_embed_calls_and_stores_identical_bytes(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: wrapper not installed (`two.calls` non-empty); `chunks_from_cache` not
        computed (stays 0 on run 2); vectors cached at reduced precision (blobs differ);
        the cache consulted when disabled (run "off" creates the directory).
        """
        repo = _repo(tmp_path)
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(cache_dir))

        off = CountingEmbedder(_DIM)
        stats_off = _run(_indexer(_config(repo), off))
        assert stats_off["chunks_from_cache"] == 0
        assert len(off.texts) == stats_off["chunks_embedded"] > 0
        assert not cache_dir.exists()
        blobs_off = _blobs(repo)

        _fresh_db(repo)
        _enable_cache(monkeypatch, cache_dir)
        one = CountingEmbedder(_DIM)
        stats_one = _run(_indexer(_config(repo), one))
        assert stats_one["chunks_from_cache"] == 0
        assert len(one.texts) == stats_one["chunks_embedded"] == stats_off["chunks_embedded"]
        assert _blobs(repo) == blobs_off

        _fresh_db(repo)
        two = CountingEmbedder(_DIM)
        stats_two = _run(_indexer(_config(repo), two))
        assert two.calls == []
        assert stats_two["chunks_embedded"] == stats_one["chunks_embedded"]
        assert stats_two["chunks_from_cache"] == stats_one["chunks_embedded"]
        assert _blobs(repo) == blobs_off
        assert len(list(cache_dir.glob("*.db"))) == 1

    def test_a_different_model_is_a_different_file_with_all_misses(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the fingerprint ignores the model (one file, zero calls on run 2)."""
        repo = _repo(tmp_path)
        cache_dir = tmp_path / "cache"
        _enable_cache(monkeypatch, cache_dir)

        _run(_indexer(_config(repo), CountingEmbedder(_DIM)))
        _fresh_db(repo)
        other = CountingEmbedder(_DIM)
        stats = _run(_indexer(_config(repo, local_model="other/model"), other))

        assert len(other.texts) == stats["chunks_embedded"] > 0
        assert stats["chunks_from_cache"] == 0
        assert len(list(cache_dir.glob("*.db"))) == 2

    def test_index_file_reports_its_own_cache_hits(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: `index_file`'s result lacks the key, or snapshots hits after the embed.

        The watch-mode path: one file through `index_file()` on a fresh DB after a cached
        run is all hits.
        """
        repo = _repo(tmp_path)
        _enable_cache(monkeypatch, tmp_path / "cache")
        _run(_indexer(_config(repo), CountingEmbedder(_DIM)))
        _fresh_db(repo)

        again = CountingEmbedder(_DIM)
        with _quiet_progress():
            result = _indexer(_config(repo), again).index_file(str(repo / "alpha.py"))

        assert result["status"] == "ok"
        assert result["chunks_updated"] > 0
        assert result["chunks_from_cache"] == result["chunks_updated"]
        assert again.calls == []

    def test_streaming_pipeline_reports_chunks_from_cache_the_same_way(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: `_index_streaming` does not sum `index_file`'s `chunks_from_cache`."""
        repo = _repo(tmp_path)
        _enable_cache(monkeypatch, tmp_path / "cache")
        first = _run(_indexer(_config(repo, streaming=True), CountingEmbedder(_DIM)))
        assert first["chunks_from_cache"] == 0
        assert first["chunks_embedded"] > 0

        _fresh_db(repo)
        again = CountingEmbedder(_DIM)
        second = _run(_indexer(_config(repo, streaming=True), again))
        assert again.calls == []
        assert second["chunks_embedded"] == first["chunks_embedded"]
        assert second["chunks_from_cache"] == first["chunks_embedded"]


# ---------------------------------------------------------------------------
# Default off, and the environment the hosted App would forward
# ---------------------------------------------------------------------------


class TestDefaultOffAndEnvironment:
    def test_env_isolation_pins_the_flag_off(self) -> None:
        """MUTATION: the entry removed from tests/_env_isolation.BEAST_MODE_DEFAULTS."""
        assert BEAST_MODE_DEFAULTS["TRELIX_EMBEDDING_CACHE_ENABLED"] == "false"

    def test_default_off_creates_nothing_and_leaves_the_embedder_unwrapped(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: directory prepared or file opened when disabled; wrapper always on."""
        repo = _repo(tmp_path)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        embedder = CountingEmbedder(_DIM)
        indexer = _indexer(_config(repo), embedder)

        assert indexer.embedder is embedder
        stats = _run(indexer)
        assert stats["chunks_from_cache"] == 0
        assert not (tmp_path / "xdg").exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["repo"]

    def test_a_relative_dir_in_the_environment_fails_closed_before_any_indexer(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the absolute-path validator removed (the run would resolve the cache
        against the cwd — inside whatever repository is being indexed)."""
        from pydantic import ValidationError

        repo = _repo(tmp_path)
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_ENABLED", "true")
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", "relative/cache")
        with pytest.raises(ValidationError, match="must be an absolute path"):
            _config(repo)

    @_POSIX_ONLY
    def test_home_only_environment_writes_under_home_dot_cache_with_private_modes(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: default directory not under `~/.cache/trelix/embeddings`; 0o700/0o600
        not applied; the file written somewhere other than the resolved directory.

        The hosted App forwards HOME and every TRELIX_* variable; this is that shape.
        """
        repo = _repo(tmp_path)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_ENABLED", "true")
        monkeypatch.delenv("TRELIX_EMBEDDING_CACHE_DIR", raising=False)

        _run(_indexer(_config(repo), CountingEmbedder(_DIM)))

        cache_dir = home / ".cache" / "trelix" / "embeddings"
        assert cache_dir.is_dir()
        assert _mode(cache_dir) == 0o700
        files = list(cache_dir.iterdir())
        assert len(files) == 1 and files[0].suffix == ".db"
        assert _mode(files[0]) == 0o600
        assert sorted(p.name for p in tmp_path.iterdir()) == ["home", "repo"]

    def test_xdg_cache_home_is_honoured(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: XDG_CACHE_HOME ignored in `resolve_cache_dir`."""
        repo = _repo(tmp_path)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_ENABLED", "true")
        monkeypatch.delenv("TRELIX_EMBEDDING_CACHE_DIR", raising=False)

        _run(_indexer(_config(repo), CountingEmbedder(_DIM)))

        assert len(list((tmp_path / "xdg" / "trelix" / "embeddings").glob("*.db"))) == 1


# ---------------------------------------------------------------------------
# Refusals before any model load
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_batch_api_with_openai_is_refused_before_any_model_load(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the check removed; the check placed after `make_embedder`."""
        from trelix.indexing.indexer import Indexer

        repo = _repo(tmp_path)
        _enable_cache(monkeypatch, tmp_path / "cache")
        cfg = _config(repo, provider="openai")
        cfg.use_batch_api = True
        with (
            patch(
                "trelix.indexing.indexer.make_embedder",
                side_effect=AssertionError("a model was loaded"),
            ),
            pytest.raises(ValueError) as excinfo,
        ):
            Indexer(cfg, quiet=True)
        assert str(excinfo.value) == _E2

    def test_batch_api_with_a_non_openai_provider_is_not_refused(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the provider condition dropped (`local` + the flag raises, where today
        the indexer only warns and embeds synchronously)."""
        repo = _repo(tmp_path)
        _enable_cache(monkeypatch, tmp_path / "cache")
        cfg = _config(repo, provider="local")
        cfg.use_batch_api = True
        indexer = _indexer(cfg, CountingEmbedder(_DIM))
        assert isinstance(indexer.embedder, CachedIndexEmbedder)

    def test_an_unusable_cache_dir_stops_before_any_model_load(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the directory pre-flight moved after `make_embedder`."""
        from trelix.indexing.indexer import Indexer

        repo = _repo(tmp_path)
        blocker = tmp_path / "cachefile"
        blocker.write_text("a file, not a directory", encoding="utf-8")
        _enable_cache(monkeypatch, blocker)
        with (
            patch(
                "trelix.indexing.indexer.make_embedder",
                side_effect=AssertionError("a model was loaded"),
            ),
            pytest.raises(EmbeddingCacheError) as excinfo,
        ):
            Indexer(_config(repo), quiet=True)
        assert str(excinfo.value).startswith("embedding cache directory cannot be used: ")

    def test_resume_batch_is_refused_when_the_cache_is_on(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the `--resume-batch` check in cli/main.py removed (the run reaches
        `Indexer()` and, in production, dies in the Batch API method's type guard).

        `Indexer.__init__` is patched to raise, as test_dry_run.py does, so the second
        invocation proves the check is the only thing that kept the first from it.
        """
        from trelix.cli.main import app
        from trelix.indexing.indexer import Indexer

        repo = _repo(tmp_path)

        def _explode(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("an Indexer was constructed")

        monkeypatch.setattr(Indexer, "__init__", _explode)
        runner = CliRunner()

        _enable_cache(monkeypatch, tmp_path / "cache")
        refused = runner.invoke(app, ["index", str(repo), "--resume-batch"])
        assert refused.exit_code == 1
        output = " ".join(refused.output.split())
        # The whole sentence, as docs/CLI_REFERENCE.md quotes it under "Exit codes".
        assert (
            "Cannot resume a Batch API job: TRELIX_EMBEDDING_CACHE_ENABLED=true is set, and "
            "the Batch API poll path bypasses the cache. Re-run with "
            "TRELIX_EMBEDDING_CACHE_ENABLED=false to collect the job, then re-enable it."
        ) in output
        assert "an Indexer was constructed" not in output

        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_ENABLED", "false")
        reached = runner.invoke(app, ["index", str(repo), "--resume-batch"])
        assert reached.exit_code == 1
        assert "Batch API resume failed" in reached.output
        assert "Cannot resume a Batch API job" not in reached.output


# ---------------------------------------------------------------------------
# Size cap after a run
# ---------------------------------------------------------------------------


class TestSizeCap:
    @pytest.mark.parametrize("streaming", [False, True])
    def test_cap_is_applied_in_bytes_from_max_mb(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
    ) -> None:
        """MUTATION: the MB -> bytes conversion wrong; `_enforce_cache_cap` not called at
        the end of `index()` (streaming=False) or of `_index_streaming` (streaming=True)."""
        repo = _repo(tmp_path)
        _enable_cache(monkeypatch, tmp_path / "cache")
        seen: list[int] = []

        def _record(self: EmbeddingCache, max_bytes: int) -> int:
            seen.append(max_bytes)
            return 0

        monkeypatch.setattr(EmbeddingCache, "enforce_size_cap", _record)
        _run(_indexer(_config(repo, streaming=streaming), CountingEmbedder(_DIM)))
        assert seen == [4294967296]

        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_MAX_MB", "2")
        _fresh_db(repo)
        _run(_indexer(_config(repo, streaming=streaming), CountingEmbedder(_DIM)))
        assert seen == [4294967296, 2097152]

    def test_a_failed_trim_is_logged_and_the_run_still_succeeds(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """MUTATION: the `sqlite3.Error` handler removed (a locked VACUUM fails the run)."""
        repo = _repo(tmp_path)
        _enable_cache(monkeypatch, tmp_path / "cache")
        first = _run(_indexer(_config(repo), CountingEmbedder(_DIM)))

        def _locked(self: EmbeddingCache, max_bytes: int) -> int:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(EmbeddingCache, "enforce_size_cap", _locked)
        _fresh_db(repo)
        with caplog.at_level(logging.WARNING, logger="trelix.indexing"):
            second = _run(_indexer(_config(repo), CountingEmbedder(_DIM)))

        assert second["chunks_from_cache"] == first["chunks_embedded"]
        assert any(
            "size cap not enforced this run (database is locked)" in record.getMessage()
            for record in caplog.records
        )

    @_POSIX_ONLY
    def test_a_deleted_cache_file_is_logged_and_the_run_still_succeeds(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The operator follows SECURITY.md ("delete the `<fingerprint>.db` files") while a
        run is in Phase 3: the REAL `enforce_size_cap` starts with `stat`, which raises
        `FileNotFoundError` -- an `OSError`, not a `sqlite3.Error`.
        MUTATION: `OSError` removed from the handler tuple (the run fails after every chunk
        was embedded and stored)."""
        repo = _repo(tmp_path)
        _enable_cache(monkeypatch, tmp_path / "cache")
        first = _run(_indexer(_config(repo), CountingEmbedder(_DIM)))

        real_enforce = EmbeddingCache.enforce_size_cap

        def _deleted_underneath(self: EmbeddingCache, max_bytes: int) -> int:
            self.path.unlink()
            return real_enforce(self, max_bytes)

        monkeypatch.setattr(EmbeddingCache, "enforce_size_cap", _deleted_underneath)
        _fresh_db(repo)
        with caplog.at_level(logging.WARNING, logger="trelix.indexing"):
            second = _run(_indexer(_config(repo), CountingEmbedder(_DIM)))

        assert second["chunks_from_cache"] == first["chunks_embedded"]
        assert any(
            "size cap not enforced this run ([Errno 2] No such file or directory: "
            in record.getMessage()
            for record in caplog.records
        )
