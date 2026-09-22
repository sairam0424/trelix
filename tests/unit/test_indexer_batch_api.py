"""
Unit tests for the OpenAI Batch API opt-in indexer strategy
(Indexer._batch_embed_and_store_via_batch_api).

Batch API jobs can take up to 24h to complete, so this strategy is a
two-invocation protocol rather than a single blocking call:

  First invocation (no pending job recorded for this repo_path):
    - `pending` is grouped into token-aware batches (same _make_token_batches
      grouping the sync/async strategies use).
    - Each batch is submitted via `embedder.submit_batch()` and persisted via
      `db.insert_batch_job()`.
    - The call returns WITHOUT blocking — nothing is embedded in this run.

  Resume invocation (a pending job already exists for this repo_path):
    - `embedder.poll_batch(job_id)` is called.
    - `None` → still processing: no upsert, no status change, returns cleanly.
    - Vectors → zipped back onto the job's persisted `pending_chunk_ids` and
      upserted via the existing `vector_store.upsert_batch` path; the job row
      is marked 'completed' via `db.update_batch_job_status`.

Follows tests/unit/test_indexer_async.py's mock-injection convention:
object.__new__(Indexer) + direct attribute assignment, no real DB/filesystem/
Progress dependency.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from trelix.embedder.base import BatchJobTerminalError, OpenAIEmbedder
from trelix.indexing.indexer import Indexer, _PendingChunk


def _make_chunk(chunk_id: int, tokens: int = 10) -> _PendingChunk:
    return _PendingChunk(
        chunk_id=chunk_id,
        chunk_text=f"chunk text {chunk_id}",
        token_count=tokens,
    )


def _build_indexer_mock(max_tokens_per_batch: int = 1000) -> Indexer:
    """
    Build a mock Indexer with:
      - self.embedder.submit_batch / poll_batch  →  set by caller
      - self.db.get_pending_batch_job / insert_batch_job / update_batch_job_status
      - self.vector_store.upsert_batch  →  MagicMock (sync call)
      - self.config.embedder / config.repo_path / config.use_batch_api
      - self._console, self._progress_cb  →  no-ops
    """
    embedder_cfg = MagicMock()
    embedder_cfg.embed_max_tokens_per_batch = max_tokens_per_batch
    embedder_cfg.provider = "openai"

    config = MagicMock()
    config.embedder = embedder_cfg
    config.repo_path = "/fake/repo"
    config.use_batch_api = True

    indexer = object.__new__(Indexer)
    indexer.config = config
    indexer.embedder = MagicMock(spec=OpenAIEmbedder)
    indexer.vector_store = MagicMock()
    indexer.db = MagicMock()
    indexer._console = MagicMock()
    indexer._progress_cb = None

    return indexer


_PENDING_JOB_ROW = {
    "id": 7,
    "repo_path": "/fake/repo",
    "provider": "openai",
    "job_id": "batch_123",
    "status": "in_progress",
    "submitted_at": "2026-01-01 00:00:00",
    "pending_chunk_ids": [10, 20],
    "created_at": "2026-01-01 00:00:00",
}


class TestBatchEmbedAndStoreViaBatchApi:
    def test_first_invocation_submits_job_without_blocking(self) -> None:
        """No pending job yet: submit_batch() is called and the job persisted
        via insert_batch_job(); no poll/upsert happens, and the call returns
        without embedding anything in this run."""
        pending = [_make_chunk(i) for i in range(3)]
        indexer = _build_indexer_mock(max_tokens_per_batch=1000)
        indexer.db.get_pending_batch_job.return_value = None
        indexer.embedder.submit_batch.return_value = "batch_123"
        stats: dict = {"chunks_embedded": 0}

        indexer._batch_embed_and_store_via_batch_api(pending, stats)

        indexer.embedder.submit_batch.assert_called_once_with([p.chunk_text for p in pending])
        indexer.db.insert_batch_job.assert_called_once_with(
            "/fake/repo", "openai", "batch_123", [p.chunk_id for p in pending]
        )
        indexer.embedder.poll_batch.assert_not_called()
        indexer.vector_store.upsert_batch.assert_not_called()
        indexer.db.update_batch_job_status.assert_not_called()
        assert stats["chunks_embedded"] == 0

    def test_first_invocation_submits_one_job_per_token_batch(self) -> None:
        """Multiple token-batches → one submit_batch/insert_batch_job call per batch."""
        # 4 chunks x 10 tokens, max_tokens_per_batch=10 → 4 separate batches.
        pending = [_make_chunk(i, tokens=10) for i in range(4)]
        indexer = _build_indexer_mock(max_tokens_per_batch=10)
        indexer.db.get_pending_batch_job.return_value = None
        indexer.embedder.submit_batch.side_effect = ["job-0", "job-1", "job-2", "job-3"]
        stats: dict = {"chunks_embedded": 0}

        indexer._batch_embed_and_store_via_batch_api(pending, stats)

        assert indexer.embedder.submit_batch.call_count == 4
        assert indexer.db.insert_batch_job.call_count == 4
        indexer.db.insert_batch_job.assert_any_call("/fake/repo", "openai", "job-0", [0])
        indexer.db.insert_batch_job.assert_any_call("/fake/repo", "openai", "job-3", [3])

    def test_resume_completed_job_upserts_and_marks_completed(self) -> None:
        """A pending job already exists: poll_batch() returns real vectors,
        which get upserted via vector_store.upsert_batch and the job row is
        marked completed."""
        indexer = _build_indexer_mock()
        indexer.db.get_pending_batch_job.return_value = dict(_PENDING_JOB_ROW)
        vectors = [[1.0, 1.0], [2.0, 2.0]]
        indexer.embedder.poll_batch.return_value = vectors
        stats: dict = {"chunks_embedded": 0}

        indexer._batch_embed_and_store_via_batch_api([], stats)

        indexer.embedder.poll_batch.assert_called_once_with("batch_123", expected_count=2)
        indexer.vector_store.upsert_batch.assert_called_once_with(
            [(10, [1.0, 1.0]), (20, [2.0, 2.0])]
        )
        indexer.db.update_batch_job_status.assert_called_once_with(7, "completed")
        indexer.embedder.submit_batch.assert_not_called()
        assert stats["chunks_embedded"] == 2

    def test_resume_still_processing_returns_without_upsert(self) -> None:
        """poll_batch() returning None means the job isn't done — no upsert,
        no status change, method returns cleanly without blocking."""
        indexer = _build_indexer_mock()
        indexer.db.get_pending_batch_job.return_value = dict(_PENDING_JOB_ROW)
        indexer.embedder.poll_batch.return_value = None
        stats: dict = {"chunks_embedded": 0}

        indexer._batch_embed_and_store_via_batch_api([], stats)

        indexer.vector_store.upsert_batch.assert_not_called()
        indexer.db.update_batch_job_status.assert_not_called()
        indexer.embedder.submit_batch.assert_not_called()
        assert stats["chunks_embedded"] == 0

    def test_resume_terminal_failure_marks_job_failed(self) -> None:
        """poll_batch() raising BatchJobTerminalError (OpenAI itself reported
        the job dead) marks the job row 'failed' rather than leaving it stuck
        as in-flight forever, and the error still propagates to the caller."""
        indexer = _build_indexer_mock()
        indexer.db.get_pending_batch_job.return_value = dict(_PENDING_JOB_ROW)
        indexer.embedder.poll_batch.side_effect = BatchJobTerminalError("batch job failed")
        stats: dict = {"chunks_embedded": 0}

        try:
            indexer._batch_embed_and_store_via_batch_api([], stats)
        except BatchJobTerminalError:
            pass
        else:
            raise AssertionError("expected BatchJobTerminalError to propagate")

        indexer.db.update_batch_job_status.assert_called_once_with(7, "failed")
        indexer.vector_store.upsert_batch.assert_not_called()

    def test_resume_transient_poll_error_does_not_mark_job_failed(self) -> None:
        """A non-terminal exception from poll_batch (network error surviving
        retries, BatchJobIncompleteError from a partial batch failure, a
        parsing bug -- anything that ISN'T BatchJobTerminalError) must NOT
        mark the job 'failed'. The OpenAI-side job may still be alive; marking
        it dead here would both lose track of it and cause a costly duplicate
        resubmission on the next run. The error must still propagate so the
        caller (index()) reports it."""
        indexer = _build_indexer_mock()
        indexer.db.get_pending_batch_job.return_value = dict(_PENDING_JOB_ROW)
        indexer.embedder.poll_batch.side_effect = ConnectionError("network blip")
        stats: dict = {"chunks_embedded": 0}

        try:
            indexer._batch_embed_and_store_via_batch_api([], stats)
        except ConnectionError:
            pass
        else:
            raise AssertionError("expected ConnectionError to propagate")

        indexer.db.update_batch_job_status.assert_not_called()
        indexer.vector_store.upsert_batch.assert_not_called()
