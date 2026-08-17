"""
Phase 3 embed/store failures must abort as a named, explained PartialIndexError.

Covers what a bare exception escaping `asyncio.gather` did not:
  - the raised error names the failing batch count, the landed chunk count and the
    recovery path (a partial index does NOT heal on the next run);
  - batches that have not started embedding when the first failure lands are
    skipped, so a store that is rejecting writes does not keep buying embeddings
    it cannot store;
  - the upsert thread executor is shut down on the failure path too.

`LanceVectorStore.upsert_batch` re-raising on a failed delete (store/vector_lance.py)
is the caller-visible failure these tests stand in for; the store is mocked here so
the test needs neither lancedb nor a table on disk.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trelix.indexing.indexer import Indexer, PartialIndexError, _PendingChunk


def _make_chunk(chunk_id: int, tokens: int = 10) -> _PendingChunk:
    return _PendingChunk(chunk_id=chunk_id, chunk_text=f"chunk text {chunk_id}", token_count=tokens)


@contextmanager
def _patch_rich_progress():
    """Silence rich.progress — no terminal, no live-render timer during tests."""
    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task = MagicMock(return_value=0)
    with patch("trelix.indexing.indexer.Progress", return_value=mock_progress):
        yield mock_progress


def _build_indexer(max_tokens_per_batch: int = 10):
    """A bare Indexer with only the Phase 3 collaborators wired (see test_indexer_async)."""
    embedder_cfg = MagicMock()
    embedder_cfg.tpm_limit = 0
    embedder_cfg.embed_max_tokens_per_batch = max_tokens_per_batch

    config = MagicMock()
    config.embedder = embedder_cfg
    config.store.backend = "lance"
    config.store.lance_uri = ".trelix/lance"
    config.db_path_absolute = "/tmp/repo/.trelix/index.db"

    indexer = object.__new__(Indexer)
    indexer.config = config
    indexer.embedder = MagicMock()
    indexer.vector_store = MagicMock()
    indexer._console = MagicMock()
    indexer._progress_cb = None
    return indexer


async def _ok_embed(texts: list[str]) -> list[list[float]]:
    return [[0.0, 1.0] for _ in texts]


class TestAsyncPartialAbort:
    def test_store_failure_raises_partial_index_error(self) -> None:
        """A failing upsert aborts the run as PartialIndexError, not a bare store error."""
        pending = [_make_chunk(i) for i in range(4)]
        indexer = _build_indexer()
        indexer.embedder.embed_async = _ok_embed
        indexer.vector_store.upsert_batch.side_effect = RuntimeError("delete failed")
        stats: dict = {"chunks_embedded": 0}

        with _patch_rich_progress():
            with pytest.raises(PartialIndexError) as excinfo:
                asyncio.run(indexer._batch_embed_and_store_async(pending, stats))

        message = str(excinfo.value)
        # Every fact a user needs to act, in the exception text: the CLI prints
        # str(exc) as its only output (cli/main.py _print_error).
        assert "0 of 4 chunk(s)" in message  # nothing landed
        assert "delete failed" in message  # the underlying store error
        assert "does not heal on the next run" in message
        assert ".trelix/lance" in message  # what to delete to rebuild
        assert "/tmp/repo/.trelix/index.db" in message
        assert stats["chunks_embedded"] == 0
        assert stats["errors"] >= 1

    def test_a_failure_stops_later_batches_from_embedding(self) -> None:
        """Once a batch fails, un-started batches skip the paid embed call.

        The failure is on the embed side here, and that is what makes the assertion
        exact rather than racy: a raise with no await ahead of it never suspends, so
        the flag is set before gather schedules batch 2. A store-side failure
        suspends at run_in_executor, and how many batches embed before it surfaces
        is genuinely timing-dependent — see the assertion in the test below.
        """
        pending = [_make_chunk(i) for i in range(5)]
        indexer = _build_indexer()
        embed_calls: list[list[str]] = []

        async def failing_embed(texts: list[str]) -> list[list[float]]:
            embed_calls.append(texts)
            raise RuntimeError("429 rate limited")

        indexer.embedder.embed_async = failing_embed
        stats: dict = {"chunks_embedded": 0}

        with _patch_rich_progress():
            with pytest.raises(PartialIndexError) as excinfo:
                asyncio.run(indexer._batch_embed_and_store_async(pending, stats))

        assert len(embed_calls) == 1, f"kept embedding after the abort: {embed_calls}"
        assert indexer.vector_store.upsert_batch.call_count == 0
        message = str(excinfo.value)
        assert "1 failed, 4 batch(es) were skipped" in message
        assert "429 rate limited" in message

    def test_batch_accounting_always_adds_up(self) -> None:
        """failed + skipped + written == total, whatever the interleaving was."""
        pending = [_make_chunk(i) for i in range(8)]
        indexer = _build_indexer()
        indexer.embedder.embed_async = _ok_embed

        calls = {"n": 0}

        def upsert(pairs) -> None:
            calls["n"] += 1
            if calls["n"] > 1:  # first batch lands, everything after fails
                raise RuntimeError("io error")

        indexer.vector_store.upsert_batch.side_effect = upsert
        stats: dict = {"chunks_embedded": 0}

        with _patch_rich_progress():
            with pytest.raises(PartialIndexError) as excinfo:
                asyncio.run(indexer._batch_embed_and_store_async(pending, stats))

        message = str(excinfo.value)
        # One chunk per batch, so the landed chunk count is the written batch count.
        assert stats["chunks_embedded"] == 1
        assert "1 of 8 chunk(s)" in message
        failed, skipped, written = (int(part) for part in _parse_batch_counts(message)[:3])
        assert failed + skipped + written == 8
        assert written == 1

    def test_upsert_executor_is_shut_down_on_the_failure_path(self) -> None:
        """The failure path must not leak the trelix-upsert threads."""
        pending = [_make_chunk(i) for i in range(3)]
        indexer = _build_indexer()
        indexer.embedder.embed_async = _ok_embed
        indexer.vector_store.upsert_batch.side_effect = RuntimeError("delete failed")

        with _patch_rich_progress():
            with pytest.raises(PartialIndexError):
                asyncio.run(indexer._batch_embed_and_store_async(pending, {"chunks_embedded": 0}))

        alive = [t.name for t in threading.enumerate() if t.name.startswith("trelix-upsert")]
        assert alive == [], f"upsert executor threads outlived the run: {alive}"

    def test_success_path_raises_nothing(self) -> None:
        """No failures → no abort, and every batch still lands (regression guard)."""
        pending = [_make_chunk(i) for i in range(4)]
        indexer = _build_indexer()
        indexer.embedder.embed_async = _ok_embed
        stats: dict = {"chunks_embedded": 0}

        with _patch_rich_progress():
            asyncio.run(indexer._batch_embed_and_store_async(pending, stats))

        assert stats["chunks_embedded"] == 4
        assert indexer.vector_store.upsert_batch.call_count == 4
        assert "errors" not in stats


def _parse_batch_counts(message: str) -> tuple[str, str, str, str]:
    """Pull (failed, skipped, written, total) out of the message's `batches:` line."""
    line = next(ln for ln in message.splitlines() if ln.strip().startswith("batches:"))
    numbers = [tok for tok in line.replace(",", " ").split() if tok.isdigit()]
    return numbers[0], numbers[1], numbers[2], numbers[3]


class TestSyncPartialAbort:
    """`index_file()` (watch mode) uses the sync Phase 3 — same abort contract."""

    def test_store_failure_raises_partial_index_error(self) -> None:
        pending = [_make_chunk(i) for i in range(3)]
        indexer = _build_indexer()
        indexer.embedder.embed.side_effect = lambda texts: [[0.0, 1.0] for _ in texts]
        indexer.vector_store.upsert_batch.side_effect = RuntimeError("delete failed")
        stats: dict = {"chunks_embedded": 0}

        with _patch_rich_progress():
            with pytest.raises(PartialIndexError) as excinfo:
                indexer._batch_embed_and_store(pending, stats)

        message = str(excinfo.value)
        # Serial by construction: the first failure stops the loop, so the two
        # remaining batches are never embedded.
        assert "1 failed, 2 batch(es) were skipped" in message
        assert "0 of 3 chunk(s)" in message
        assert indexer.embedder.embed.call_count == 1

    def test_index_file_surfaces_the_message(self) -> None:
        """index_file() converts the abort to status=error carrying the same guidance."""
        indexer = _build_indexer()
        indexer.embedder.embed.side_effect = lambda texts: [[0.0, 1.0] for _ in texts]
        indexer.vector_store.upsert_batch.side_effect = RuntimeError("delete failed")
        indexer.db = MagicMock()
        indexer.db.get_file_hash.return_value = "not-the-current-hash"
        indexer._file_summarizer = None
        # index_file() hashes and rel_paths a real file, so this one stands in for
        # the changed source file; only the parse and insert steps are stubbed.
        indexer.config.repo_path = str(Path(__file__).resolve().parent)

        parsed = MagicMock(skipped=False)
        with (
            _patch_rich_progress(),
            patch.object(Indexer, "_parse_one", return_value=parsed),
            patch.object(Indexer, "_insert_one", return_value=([_make_chunk(0)], None)),
            patch("trelix.indexing.indexer.detect_language", MagicMock()),
        ):
            result = indexer.index_file(__file__)

        assert result["status"] == "error"
        assert "does not heal on the next run" in result["error"]
