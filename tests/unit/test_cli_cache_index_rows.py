"""The two places `trelix index` shows the on-disk embedding cache: the Index Summary row
(`Chunks from cache`) and the `--dry-run` cost preview after a REAL cached run.

Split from test_cli_cache.py (which keeps `trelix cache gc` and `trelix cache clear`) so
each file stays under the 500-line limit. The two preview tests run a real `Indexer` with
`CountingEmbedder` patched in, as `test_embedding_cache_indexer.py` does; that is what puts
THIS file in conftest's `SLOW_FILES`. The summary-row tests use a fake `Indexer` and ride
along because they share the subject and the helpers.

Each test names the mutation(s) that must fail it.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tests.fixtures.fakes import CountingEmbedder
from trelix.core.config import EmbedderConfig, IndexConfig, StoreConfig

runner = CliRunner()

_FILES = {
    "alpha.py": 'def alpha(x: int) -> int:\n    """Add one."""\n    return x + 1\n',
    "beta.py": 'def beta(y: int) -> int:\n    """Double."""\n    return y * 2\n',
}


def _invoke(*args: str) -> Any:
    from trelix.cli.main import app

    return runner.invoke(app, list(args))


def _squash(output: str) -> str:
    """Rich wraps at 80 columns off a terminal; collapse whitespace so a line under test
    is matched by content, not width."""
    return " ".join(output.split())


# ---------------------------------------------------------------------------
# trelix index: the Index Summary row
# ---------------------------------------------------------------------------


def _fake_indexer(monkeypatch: pytest.MonkeyPatch, stats: dict[str, int]) -> None:
    """`Indexer` that builds nothing and reports `stats`, as test_cli_batch_api_flags does."""
    import trelix.store.provenance as provenance_mod
    from trelix.indexing.indexer import Indexer

    def _fake_init(self: Any, *_args: object, **_kwargs: object) -> None:
        self.db = MagicMock()

    monkeypatch.setattr(Indexer, "__init__", _fake_init)
    monkeypatch.setattr(Indexer, "index", lambda self: dict(stats))
    monkeypatch.setattr(provenance_mod, "read_provenance", lambda db: None)


_STATS = {
    "files_found": 2,
    "files_indexed": 2,
    "files_skipped": 0,
    "symbols_extracted": 4,
    "chunks_embedded": 3,
}


class TestIndexSummaryRow:
    def test_chunks_from_cache_is_a_row_when_nonzero(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the row not added; the row reads another key."""
        _fake_indexer(monkeypatch, {**_STATS, "chunks_from_cache": 3})

        result = _invoke("index", str(tmp_path))

        assert result.exit_code == 0, result.output
        output = _squash(result.output)
        assert re.search(r"Chunks from cache\D*?3\b", output), output
        assert re.search(r"Chunks embedded\D*?3\b", output), output

    def test_no_row_at_zero(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """MUTATION: the row printed unconditionally (`Chunks from cache 0`)."""
        _fake_indexer(monkeypatch, {**_STATS, "chunks_from_cache": 0})

        result = _invoke("index", str(tmp_path))

        assert result.exit_code == 0, result.output
        assert "Chunks from cache" not in result.output
        assert "Chunks embedded" in result.output


# ---------------------------------------------------------------------------
# trelix index --dry-run after a real cached run (end to end)
# ---------------------------------------------------------------------------


def _repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for name, body in _FILES.items():
        (repo / name).write_text(body, encoding="utf-8")
    return repo


def _index_once(repo: pathlib.Path, *, provider: str, dimension: int) -> dict[str, Any]:
    """One real `Indexer` run with `CountingEmbedder(dimension)` in place of the provider
    (`test_embedding_cache_indexer.py`'s harness), then the index deleted so the preview
    counts every file as new. The cache file records `dimension`, not the configured width."""
    from trelix.indexing.indexer import Indexer
    from trelix.store.vector import VectorStore

    cfg = IndexConfig(
        repo_path=str(repo),
        incremental=True,
        store=StoreConfig(db_path=str(repo / ".trelix" / "index.db")),
        embedder=EmbedderConfig.model_construct(provider=provider),
    )
    store = VectorStore(cfg.db_path_absolute, dimension=dimension)
    progress = MagicMock()
    progress.__enter__ = MagicMock(return_value=progress)
    progress.__exit__ = MagicMock(return_value=False)
    with (
        patch("trelix.indexing.indexer.make_embedder", return_value=CountingEmbedder(dimension)),
        patch("trelix.indexing.indexer.make_vector_store", return_value=store),
        patch("trelix.cli.progress.Progress", return_value=progress),
    ):
        stats: dict[str, Any] = Indexer(cfg, quiet=True).index()
    for path in (repo / ".trelix").glob("index.db*"):
        path.unlink()
    return stats


def _number(output: str, label: str) -> int:
    match = re.search(rf"{label}\D*?([\d,]+)", output)
    assert match, f"no {label!r} row in:\n{output}"
    return int(match.group(1).replace(",", ""))


class TestDryRunAfterACachedRun:
    def test_every_chunk_of_the_default_local_config_is_already_cached(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """C7: the file records width 8 while `effective_dimension` says 384 for every
        `local` model, and the preview must still see every chunk as cached.

        MUTATION: the preview opens the file at `config.embedder.effective_dimension`
        (refused or zero hits); the preview keys on anything but `chunk_text`."""
        repo = _repo(tmp_path)
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_ENABLED", "true")
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(tmp_path / "cache"))
        stats = _index_once(repo, provider="local", dimension=8)
        assert stats["chunks_embedded"] >= 2

        result = _invoke("index", str(repo), "--dry-run")

        assert result.exit_code == 0, result.output
        output = _squash(result.output)
        assert _number(output, "Chunks to embed") == stats["chunks_embedded"]
        assert _number(output, "Chunks already cached") == stats["chunks_embedded"]
        assert _number(output, "Tokens already cached") == _number(output, "Embedding tokens") > 0

    def test_the_priced_total_is_reduced_by_the_cached_tokens(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MUTATION: the subtraction dropped from `_print_cost_estimate`'s argument (the
        priced count equals `Embedding tokens` instead of 0)."""
        repo = _repo(tmp_path)
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_ENABLED", "true")
        monkeypatch.setenv("TRELIX_EMBEDDING_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setenv("TRELIX_EMBEDDER_PROVIDER", "openai")
        monkeypatch.setenv("TRELIX_EMBEDDER_OPENAI_MODEL", "text-embedding-3-large")
        stats = _index_once(repo, provider="openai", dimension=8)

        result = _invoke("index", str(repo), "--dry-run")

        assert result.exit_code == 0, result.output
        output = _squash(result.output)
        assert _number(output, "Chunks already cached") == stats["chunks_embedded"] >= 2
        assert _number(output, "Embedding tokens") > 0
        priced = re.search(r"Estimated cost.*?([\d,]+) tokens at", output)
        assert priced, output
        assert int(priced.group(1).replace(",", "")) == 0
