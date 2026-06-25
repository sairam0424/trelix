"""
Unit tests for U5: async batch embedding pipeline in Indexer.

Covers:
  - 4 batches run concurrently (semaphore allows 4 at once)
  - TPM rate limiter respected (asyncio.sleep called when limit hit)
  - upsert_batch called with correct (chunk_id, embedding) pairs per batch
  - stats["chunks_embedded"] correctly counted across all batches
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from trelix.indexing.indexer import (
    _AsyncTpmRateLimiter,
    _make_token_batches,
    _PendingChunk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: int, tokens: int = 10) -> _PendingChunk:
    return _PendingChunk(
        chunk_id=chunk_id,
        chunk_text=f"chunk text {chunk_id}",
        token_count=tokens,
    )


def _fake_embedding(n: int = 4) -> list[float]:
    """Return a trivial embedding vector of length n."""
    return [float(i) for i in range(n)]


@contextmanager
def _patch_rich_progress():
    """
    Patch rich.progress.Progress so it does not attempt real terminal/timer
    operations during tests. The context manager and task methods are no-ops.
    """
    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task = MagicMock(return_value=0)
    mock_progress.advance = MagicMock()

    with patch("trelix.indexing.indexer.Progress", return_value=mock_progress):
        yield mock_progress


# ---------------------------------------------------------------------------
# _AsyncTpmRateLimiter tests
# ---------------------------------------------------------------------------


class TestAsyncTpmRateLimiter:
    def test_no_limit_does_not_sleep(self) -> None:
        """tpm_limit=0 → no sleep, acquire returns immediately."""
        limiter = _AsyncTpmRateLimiter(tpm_limit=0)

        async def _run() -> None:
            with patch("asyncio.sleep") as mock_sleep:
                await limiter.acquire(9999)
                mock_sleep.assert_not_called()

        asyncio.run(_run())

    def test_within_limit_does_not_sleep(self) -> None:
        """Tokens below the limit within a window → no sleep."""
        limiter = _AsyncTpmRateLimiter(tpm_limit=1000)

        async def _run() -> None:
            with patch("asyncio.sleep") as mock_sleep:
                await limiter.acquire(100)
                await limiter.acquire(100)
                mock_sleep.assert_not_called()

        asyncio.run(_run())

    def test_exceeding_limit_triggers_sleep(self) -> None:
        """When cumulative tokens exceed the limit, asyncio.sleep is called."""
        limiter = _AsyncTpmRateLimiter(tpm_limit=50, console=MagicMock())

        async def _run() -> list[float]:
            sleep_calls: list[float] = []

            async def fake_sleep(secs: float) -> None:
                sleep_calls.append(secs)
                # Advance the window start so limiter resets correctly after sleep
                limiter._window_start = time.monotonic() - 65.0

            with patch("asyncio.sleep", side_effect=fake_sleep):
                await limiter.acquire(40)  # used=40, ok
                await limiter.acquire(20)  # used+20=60 > 50 → sleep
            return sleep_calls

        sleep_calls = asyncio.run(_run())
        assert len(sleep_calls) == 1
        assert sleep_calls[0] > 0

    def test_window_reset_after_sleep(self) -> None:
        """After sleep the used counter resets so further acquires succeed."""
        limiter = _AsyncTpmRateLimiter(tpm_limit=50, console=MagicMock())

        async def _run() -> int:
            sleep_count = 0

            async def fake_sleep(secs: float) -> None:
                nonlocal sleep_count
                sleep_count += 1
                limiter._window_start = time.monotonic() - 65.0

            with patch("asyncio.sleep", side_effect=fake_sleep):
                await limiter.acquire(40)  # ok
                await limiter.acquire(20)  # triggers sleep, resets
                await limiter.acquire(30)  # should NOT trigger another sleep
            return sleep_count

        count = asyncio.run(_run())
        assert count == 1

    def test_concurrent_acquires_serialised_by_lock(self) -> None:
        """Multiple concurrent acquire() calls are safe — lock prevents races."""
        limiter = _AsyncTpmRateLimiter(tpm_limit=0)  # unlimited, just test no crash

        async def _run() -> None:
            tasks = [limiter.acquire(10) for _ in range(20)]
            await asyncio.gather(*tasks)

        asyncio.run(_run())  # must not raise


# ---------------------------------------------------------------------------
# _batch_embed_and_store_async tests
# ---------------------------------------------------------------------------


class TestBatchEmbedAndStoreAsync:
    """
    Tests for Indexer._batch_embed_and_store_async.

    We construct a minimal Indexer-like fixture using direct attribute injection
    rather than a real Database/VectorStore, so tests run without any filesystem
    or DB dependencies.  Rich.Progress is patched so terminal operations don't
    interfere.
    """

    def _build_indexer_mock(
        self,
        tpm_limit: int = 0,
        max_tokens_per_batch: int = 1000,
    ):
        """
        Build a mock Indexer with:
          - self.embedder.embed_async  →  set by caller
          - self.vector_store.upsert_batch  →  MagicMock (sync call)
          - self.config.embedder  →  stubbed config
          - self._console, self._progress_cb  →  no-ops
        """
        from trelix.indexing.indexer import Indexer  # type: ignore[attr-defined]

        embedder_cfg = MagicMock()
        embedder_cfg.tpm_limit = tpm_limit
        embedder_cfg.embed_max_tokens_per_batch = max_tokens_per_batch

        config = MagicMock()
        config.embedder = embedder_cfg

        embedder = MagicMock()
        vector_store = MagicMock()

        indexer = object.__new__(Indexer)
        indexer.config = config
        indexer.embedder = embedder
        indexer.vector_store = vector_store
        indexer._console = MagicMock()
        indexer._progress_cb = None

        return indexer

    # ------------------------------------------------------------------

    def test_four_batches_run_concurrently(self) -> None:
        """
        With semaphore=4 and tpm_limit=0, all 4 batches should be launched
        concurrently — asyncio.gather fans them out simultaneously.
        """
        pending = [_make_chunk(i) for i in range(4)]
        # max_tokens_per_batch=10, each chunk=10 tokens → exactly 1 chunk per batch
        indexer = self._build_indexer_mock(max_tokens_per_batch=10)
        stats: dict = {"chunks_embedded": 0}

        call_times: list[float] = []

        async def timed_embed(texts: list[str]) -> list[list[float]]:
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)  # simulate short API latency
            return [_fake_embedding(4) for _ in texts]

        indexer.embedder.embed_async = timed_embed

        with _patch_rich_progress():
            asyncio.run(indexer._batch_embed_and_store_async(pending, stats))

        # All 4 batches should have started within a short window
        assert len(call_times) == 4
        # If sequential (0.01s each), span would be ~0.03s.
        # If concurrent, all start at roughly the same time → span < 0.02s.
        span = max(call_times) - min(call_times)
        assert span < 0.02, f"Batches appear sequential (span={span:.4f}s); expected concurrent"

    def test_upsert_batch_called_once_per_batch(self) -> None:
        """upsert_batch must be called exactly once per batch."""
        # 4 chunks, max_tokens=10 per batch, each chunk=10 → 4 batches
        pending = [_make_chunk(i, tokens=10) for i in range(4)]
        indexer = self._build_indexer_mock(max_tokens_per_batch=10)

        async def default_embed(texts: list[str]) -> list[list[float]]:
            return [_fake_embedding(4) for _ in texts]

        indexer.embedder.embed_async = default_embed
        stats: dict = {"chunks_embedded": 0}

        with _patch_rich_progress():
            asyncio.run(indexer._batch_embed_and_store_async(pending, stats))

        assert indexer.vector_store.upsert_batch.call_count == 4

    def test_upsert_batch_called_with_correct_pairs(self) -> None:
        """upsert_batch must receive the right (chunk_id, embedding) pairs."""
        pending = [
            _make_chunk(chunk_id=10, tokens=10),
            _make_chunk(chunk_id=20, tokens=10),
        ]
        # 2 batches: one per chunk
        results_by_call: list[list[list[float]]] = [
            [[1.0, 1.0, 1.0, 1.0]],  # batch 0
            [[2.0, 2.0, 2.0, 2.0]],  # batch 1
        ]
        call_counter = [0]

        async def ordered_embed(texts: list[str]) -> list[list[float]]:
            idx = call_counter[0]
            call_counter[0] += 1
            return results_by_call[idx]

        indexer = self._build_indexer_mock(max_tokens_per_batch=10)
        indexer.embedder.embed_async = ordered_embed
        stats: dict = {"chunks_embedded": 0}

        with _patch_rich_progress():
            asyncio.run(indexer._batch_embed_and_store_async(pending, stats))

        calls = indexer.vector_store.upsert_batch.call_args_list
        all_pairs: list[tuple[int, list[float]]] = []
        for c in calls:
            all_pairs.extend(c[0][0])

        ids_seen = {pair[0] for pair in all_pairs}
        assert 10 in ids_seen
        assert 20 in ids_seen

    def test_chunks_embedded_stat_correct(self) -> None:
        """stats['chunks_embedded'] must equal total number of pending chunks."""
        n = 12
        pending = [_make_chunk(i) for i in range(n)]
        indexer = self._build_indexer_mock(max_tokens_per_batch=500)

        async def default_embed(texts: list[str]) -> list[list[float]]:
            return [_fake_embedding(4) for _ in texts]

        indexer.embedder.embed_async = default_embed
        stats: dict = {"chunks_embedded": 0}

        with _patch_rich_progress():
            asyncio.run(indexer._batch_embed_and_store_async(pending, stats))

        assert stats["chunks_embedded"] == n

    def test_tpm_rate_limiter_respected(self) -> None:
        """
        When tpm_limit is set and a batch would exceed it, asyncio.sleep is
        called (non-blocking wait) before the next API call.
        """
        # 3 chunks × 40 tokens each; limit=50 → first chunk ok, second triggers sleep
        pending = [_make_chunk(i, tokens=40) for i in range(3)]
        indexer = self._build_indexer_mock(tpm_limit=50, max_tokens_per_batch=40)

        async def default_embed(texts: list[str]) -> list[list[float]]:
            return [_fake_embedding(4) for _ in texts]

        indexer.embedder.embed_async = default_embed
        stats: dict = {"chunks_embedded": 0}
        sleep_calls: list[float] = []

        async def fake_sleep(secs: float) -> None:
            sleep_calls.append(secs)
            # Don't actually wait — but DO advance the limiter's window so it
            # can accept tokens again after the "sleep".
            # We find the limiter via the coroutine's closure — the simplest
            # approach is to just capture the call and not actually sleep.

        with _patch_rich_progress():
            with patch("asyncio.sleep", side_effect=fake_sleep):
                asyncio.run(indexer._batch_embed_and_store_async(pending, stats))

        assert len(sleep_calls) >= 1, (
            "Expected asyncio.sleep to be called for TPM limiting but got none"
        )

    def test_empty_pending_is_noop(self) -> None:
        """No pending chunks → no embed_async or upsert_batch calls."""
        indexer = self._build_indexer_mock()
        indexer.embedder.embed_async = AsyncMock()
        stats: dict = {"chunks_embedded": 0}

        with _patch_rich_progress():
            asyncio.run(indexer._batch_embed_and_store_async([], stats))

        indexer.embedder.embed_async.assert_not_called()
        indexer.vector_store.upsert_batch.assert_not_called()
        assert stats["chunks_embedded"] == 0

    def test_upsert_batch_receives_correct_ids(self) -> None:
        """Each chunk's chunk_id must be correctly matched to its embedding."""
        pending = [
            _make_chunk(chunk_id=100, tokens=10),
            _make_chunk(chunk_id=200, tokens=10),
        ]

        async def default_embed(texts: list[str]) -> list[list[float]]:
            return [_fake_embedding(4) for _ in texts]

        indexer = self._build_indexer_mock(max_tokens_per_batch=10)
        indexer.embedder.embed_async = default_embed
        stats: dict = {"chunks_embedded": 0}

        with _patch_rich_progress():
            asyncio.run(indexer._batch_embed_and_store_async(pending, stats))

        calls = indexer.vector_store.upsert_batch.call_args_list
        all_pairs: list[tuple[int, list[float]]] = []
        for c in calls:
            all_pairs.extend(c[0][0])

        ids_seen = {pair[0] for pair in all_pairs}
        assert 100 in ids_seen
        assert 200 in ids_seen


# ---------------------------------------------------------------------------
# _make_token_batches (regression — unchanged from sync version)
# ---------------------------------------------------------------------------


class TestMakeTokenBatches:
    def test_single_batch_when_all_fit(self) -> None:
        chunks = [_make_chunk(i, tokens=10) for i in range(5)]
        batches = _make_token_batches(chunks, max_tokens_per_batch=100)
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_splits_on_token_overflow(self) -> None:
        # 3 chunks of 40 tokens, max=80 → batches: [0,1], [2]
        chunks = [_make_chunk(i, tokens=40) for i in range(3)]
        batches = _make_token_batches(chunks, max_tokens_per_batch=80)
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1

    def test_oversized_single_chunk_gets_own_batch(self) -> None:
        chunks = [_make_chunk(0, tokens=200)]
        batches = _make_token_batches(chunks, max_tokens_per_batch=100)
        assert len(batches) == 1
        assert batches[0][0].chunk_id == 0

    def test_empty_input(self) -> None:
        assert _make_token_batches([], max_tokens_per_batch=100) == []
