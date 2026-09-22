"""`trelix index --use-batch-api` / `--resume-batch` CLI flags.

Two separate concerns:

1. `--use-batch-api` must reach `IndexConfig.use_batch_api` so `Indexer.index()`'s
   own dispatch guard (`isinstance(self.embedder, OpenAIEmbedder)`) sees it.
2. `--resume-batch` must skip the walk/parse/chunk pipeline entirely -- it only
   makes sense for a job already submitted in a prior run -- and go straight to
   `Indexer._batch_embed_and_store_via_batch_api([], stats)`, which already
   handles "check for a pending job and resolve/poll it" on its own regardless
   of what `pending` chunks are passed.

Follows tests/unit/test_dry_run.py's convention: patch `Indexer.__init__` on the
real class rather than the module namespace, since `trelix.cli.main.index()`
imports `Indexer` locally inside the function body.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from trelix.indexing.indexer import Indexer

runner = CliRunner()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "sample.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def _no_real_provenance_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """index() reads provenance before running -- irrelevant to these tests, and
    it expects a real Database connection our fake Indexer never sets up."""
    from trelix.store import provenance as provenance_mod

    monkeypatch.setattr(provenance_mod, "read_provenance", lambda db: None)


def test_use_batch_api_flag_reaches_index_config(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    """--use-batch-api must set IndexConfig.use_batch_api=True."""
    from trelix.cli import main as cli_main

    captured_configs = []

    def _fake_init(self, config, *args, **kwargs):
        captured_configs.append(config)
        self.db = MagicMock()

    monkeypatch.setattr(Indexer, "__init__", _fake_init)
    monkeypatch.setattr(Indexer, "index", lambda self: {"files_found": 0, "files_indexed": 0})

    result = runner.invoke(cli_main.app, ["index", str(repo), "--use-batch-api"])

    assert result.exit_code == 0, result.output
    assert len(captured_configs) == 1
    assert captured_configs[0].use_batch_api is True


def test_without_use_batch_api_flag_defaults_false(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    """Omitting the flag must leave IndexConfig.use_batch_api at its False default."""
    from trelix.cli import main as cli_main

    captured_configs = []

    def _fake_init(self, config, *args, **kwargs):
        captured_configs.append(config)
        self.db = MagicMock()

    monkeypatch.setattr(Indexer, "__init__", _fake_init)
    monkeypatch.setattr(Indexer, "index", lambda self: {"files_found": 0, "files_indexed": 0})

    result = runner.invoke(cli_main.app, ["index", str(repo)])

    assert result.exit_code == 0, result.output
    assert captured_configs[0].use_batch_api is False


def test_resume_batch_skips_the_walk_and_resolves_the_pending_job(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    """--resume-batch must call _batch_embed_and_store_via_batch_api directly,
    never Indexer.index() (which would re-walk/re-parse/re-chunk the repo)."""
    from trelix.cli import main as cli_main
    from trelix.cli.main import _build_embedder_config

    mock_db = MagicMock()
    mock_db.get_pending_batch_job.return_value = {
        "id": 1,
        "job_id": "batch_123",
        "pending_chunk_ids": [1, 2],
        # Whatever provider actually resolves with no --provider passed on
        # THIS machine (an operator env file can override the "local"
        # default) -- must match so the provider-mismatch guard this test
        # isn't exercising doesn't trip.
        "provider": _build_embedder_config(None).provider,
    }

    def _fake_init(self, config, *args, **kwargs):
        self.config = config
        self.db = mock_db

    mock_index = MagicMock()
    mock_batch_method = MagicMock()
    monkeypatch.setattr(Indexer, "__init__", _fake_init)
    monkeypatch.setattr(Indexer, "index", mock_index)
    monkeypatch.setattr(Indexer, "_batch_embed_and_store_via_batch_api", mock_batch_method)

    result = runner.invoke(cli_main.app, ["index", str(repo), "--resume-batch"])

    assert result.exit_code == 0, result.output
    mock_index.assert_not_called()
    mock_batch_method.assert_called_once()
    pending_arg = mock_batch_method.call_args[0][0]
    assert pending_arg == []  # nothing walked


def test_resume_batch_with_no_pending_job_reports_clearly(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    """--resume-batch with no pending job at all must say so, not silently no-op."""
    from trelix.cli import main as cli_main

    mock_db = MagicMock()
    mock_db.get_pending_batch_job.return_value = None

    def _fake_init(self, config, *args, **kwargs):
        self.db = mock_db

    monkeypatch.setattr(Indexer, "__init__", _fake_init)
    monkeypatch.setattr(Indexer, "index", MagicMock())

    result = runner.invoke(cli_main.app, ["index", str(repo), "--resume-batch"])

    assert result.exit_code != 0
    assert "no pending" in result.output.lower()


def test_resume_batch_with_provider_mismatch_gives_actionable_error(
    monkeypatch: pytest.MonkeyPatch, repo: Path
) -> None:
    """A job submitted with --provider openai, resumed without repeating that
    flag (so the embedder resolves to the "local" default), must fail with
    an actionable message telling the user which --provider to pass -- not
    the internal isinstance-guard TypeError from deep inside
    _batch_embed_and_store_via_batch_api."""
    from trelix.cli import main as cli_main

    mock_db = MagicMock()
    mock_db.get_pending_batch_job.return_value = {
        "id": 1,
        "job_id": "batch_123",
        "pending_chunk_ids": [1, 2],
        "provider": "openai",
    }

    def _fake_init(self, config, *args, **kwargs):
        self.config = config
        self.db = mock_db

    mock_batch_method = MagicMock()
    monkeypatch.setattr(Indexer, "__init__", _fake_init)
    monkeypatch.setattr(Indexer, "_batch_embed_and_store_via_batch_api", mock_batch_method)

    result = runner.invoke(cli_main.app, ["index", str(repo), "--resume-batch"])

    assert result.exit_code != 0
    assert "openai" in result.output.lower()
    assert "--provider" in result.output
    mock_batch_method.assert_not_called()
