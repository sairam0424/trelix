"""
Phase 2.5 (LLM file summaries): failure visibility, then concurrency.

Two problems, in this order, because the second multiplies the first.

VISIBILITY.  `FileSummarizer.summarize()` returns "" on any LLM failure and the
indexer's `if summary:` had no `else` and no counter.  0 summaries out of 467
therefore looked exactly like 467 out of 467 from the stats dict, the console, and
the default (WARNING) log level alike.  The live index had 442/467 — 25 files had
already lost their file-level retrieval entry and nothing anywhere said so.

CONCURRENCY.  Phase 2.5 was inline in `_insert_one`, i.e. fully sequential: a
measured 428 summaries spanning 1034.2 s of an 1153.7 s run (89.6%), median
inter-file gap 2.386 s.  Fanning the chat calls out needs a chat-side rate limiter,
which did not exist anywhere in trelix — `EmbedderConfig.tpm_limit` is consumed
only by the Phase 3 embed path.

The DB write and the summary embed deliberately stay on the calling thread:
`Database.upsert_file_summary()` writes through the single shared
`sqlite3.Connection` without taking `_conn_lock`, and db.py's own comment records
that the connection "is not safe for concurrent statement execution from multiple
threads even with check_same_thread=False".
"""

from __future__ import annotations

import logging
import pathlib
import threading
import time
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import EmbedderConfig, IndexConfig, StoreConfig
from trelix.core.models import Symbol, SymbolKind
from trelix.indexing.parser.base import BaseParser, ParseResult

_DIM = 4


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.0] * _DIM for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * _DIM

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    @property
    def dimension(self) -> int:
        return _DIM


class _SummaryEmbedExplodes(_FakeEmbedder):
    """Fails only on texts that look like summaries, so chunk embedding survives."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        if texts and all(t.startswith("SUMMARY") for t in texts):
            raise RuntimeError("azure: 429 rate limit on summary embed")
        return super().embed(texts)


class _FakeVectorStore:
    def __init__(self) -> None:
        self.summary_embeddings: list[int] = []
        self.sub_chunk_embeddings: list[int] = []

    def upsert_batch(self, pairs: list[tuple[int, list[float]]]) -> None:
        pass

    def upsert_file_summary_embedding(self, file_id: int, embedding: list[float]) -> None:
        self.summary_embeddings.append(file_id)

    def upsert_sub_chunk_embedding(self, sub_chunk_id: int, embedding: list[float]) -> None:
        self.sub_chunk_embeddings.append(sub_chunk_id)

    def delete_batch(self, ids: list[int]) -> None:
        pass

    def search(self, vector: list[float], k: int) -> list[Any]:
        return []


class _FakeParser(BaseParser):
    @property
    def language_name(self) -> str:
        return "python"

    def parse(self, source: str, file_id: int) -> ParseResult:
        sym = Symbol(
            file_id=file_id,
            name="__stub__",
            qualified_name="__stub__",
            kind=SymbolKind.FUNCTION,
            line_start=1,
            line_end=max(1, len(source.splitlines())),
            signature="def __stub__()",
            body=source,
        )
        return ParseResult(symbols=[sym], call_edges=[], import_edges=[], parse_errors=0)


def _fake_get_parser(language: Any) -> _FakeParser:  # noqa: ANN401
    return _FakeParser()


class _FakeClock:
    """Injectable clock+sleep so window waits are asserted, not spent."""

    def __init__(self) -> None:
        self._t = 1000.0
        self.slept: list[float] = []
        self._lock = threading.Lock()

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.slept.append(seconds)
            self._t += seconds


class _RecordingSummarizer:
    """Stands in for FileSummarizer; records concurrency and thread identity."""

    def __init__(self, latency: float = 0.0, fail_paths: set[str] | None = None) -> None:
        self._latency = latency
        self._fail_paths = fail_paths or set()
        self._lock = threading.Lock()
        self.in_flight = 0
        self.peak_in_flight = 0
        self.calls: list[str] = []
        self.threads: set[int] = set()

    def summarize(self, rel_path: str, symbols: list[Symbol], language: Any) -> str:  # noqa: ANN401
        with self._lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.calls.append(rel_path)
            self.threads.add(threading.get_ident())
        try:
            if self._latency:
                time.sleep(self._latency)
            if rel_path in self._fail_paths:
                # Exactly what the real summarizer does on an LLM error.
                return ""
            return f"SUMMARY of {rel_path}"
        finally:
            with self._lock:
                self.in_flight -= 1


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@contextmanager
def _quiet_progress():  # type: ignore[no-untyped-def]
    mock_progress = MagicMock()
    mock_progress.__enter__ = MagicMock(return_value=mock_progress)
    mock_progress.__exit__ = MagicMock(return_value=False)
    mock_progress.add_task = MagicMock(return_value=0)
    mock_progress.advance = MagicMock()
    with (
        patch("trelix.indexing.indexer.Progress", return_value=mock_progress),
        patch("trelix.indexing.indexer.get_parser", side_effect=_fake_get_parser),
    ):
        yield


def _make_indexer(tmp_dir: str, embedder: Any = None) -> Any:  # noqa: ANN401
    from trelix.indexing.indexer import Indexer

    cfg = IndexConfig(
        repo_path=tmp_dir,
        incremental=False,
        file_summaries_enabled=True,
        store=StoreConfig(db_path=str(pathlib.Path(tmp_dir) / ".trelix" / "index.db")),
        embedder=EmbedderConfig.model_construct(provider="local", tpm_limit=0),
        _env_file=None,
    )
    with (
        patch("trelix.indexing.indexer.make_embedder", return_value=embedder or _FakeEmbedder()),
        patch("trelix.indexing.indexer.make_vector_store", return_value=_FakeVectorStore()),
        patch("trelix.indexing.indexer.get_parser", side_effect=_fake_get_parser),
        # build_chat_client would need real credentials; the summarizer is replaced below.
        patch("trelix.llm.factory.build_chat_client", return_value=MagicMock()),
    ):
        indexer = Indexer(cfg, quiet=True)
    return indexer


def _write_files(tmp_path: pathlib.Path, count: int) -> list[str]:
    names = []
    for i in range(count):
        name = f"mod{i}.py"
        (tmp_path / name).write_text(f"def fn{i}():\n    return {i}\n", encoding="utf-8")
        names.append(name)
    return names


def _run(
    tmp_path: pathlib.Path,
    summarizer: Any,  # noqa: ANN401
    *,
    files: int = 4,
    workers: int | None = None,
    embedder: Any = None,  # noqa: ANN401
) -> tuple[Any, dict[str, Any]]:
    _write_files(tmp_path, files)
    indexer = _make_indexer(str(tmp_path), embedder=embedder)
    indexer._file_summarizer = summarizer
    if workers is not None:
        indexer._summary_workers = workers
    with _quiet_progress():
        stats = indexer.index()
    return indexer, stats


# ---------------------------------------------------------------------------
# 1. Failure visibility  (the prerequisite)
# ---------------------------------------------------------------------------


class TestSummaryOutcomeIsCounted:
    """Success and total failure must not produce the same stats dict."""

    def test_generated_count_is_reported(self, tmp_path: pathlib.Path) -> None:
        _, stats = _run(tmp_path, _RecordingSummarizer(), files=4)
        assert stats["file_summaries_generated"] == 4, stats

    def test_total_failure_is_reported(self, tmp_path: pathlib.Path) -> None:
        names = [f"mod{i}.py" for i in range(4)]
        _, stats = _run(tmp_path, _RecordingSummarizer(fail_paths=set(names)), files=4)
        assert stats["file_summaries_generated"] == 0, stats
        assert stats["file_summaries_failed"] == 4, stats

    def test_success_and_total_failure_differ(self, tmp_path: pathlib.Path) -> None:
        """The regression that mattered: 0/467 read identically to 467/467."""
        good = tmp_path / "good"
        bad = tmp_path / "bad"
        good.mkdir()
        bad.mkdir()
        _, ok_stats = _run(good, _RecordingSummarizer(), files=3)
        _, bad_stats = _run(
            bad, _RecordingSummarizer(fail_paths={f"mod{i}.py" for i in range(3)}), files=3
        )
        assert ok_stats["file_summaries_generated"] != bad_stats["file_summaries_generated"]
        assert bad_stats["file_summaries_failed"] == 3

    def test_partial_failure_splits_the_two_counters(self, tmp_path: pathlib.Path) -> None:
        _, stats = _run(tmp_path, _RecordingSummarizer(fail_paths={"mod1.py"}), files=4)
        assert stats["file_summaries_generated"] == 3, stats
        assert stats["file_summaries_failed"] == 1, stats

    def test_embedded_count_separates_stored_from_retrievable(self, tmp_path: pathlib.Path) -> None:
        """A summary row with no vector is invisible to the 4th retrieval leg."""
        _, stats = _run(tmp_path, _RecordingSummarizer(), files=3, embedder=_SummaryEmbedExplodes())
        assert stats["file_summaries_generated"] == 3, stats
        assert stats["file_summaries_embedded"] == 0, stats

    def test_a_failed_summary_embed_does_not_cost_the_chunks(self, tmp_path: pathlib.Path) -> None:
        _, stats = _run(tmp_path, _RecordingSummarizer(), files=3, embedder=_SummaryEmbedExplodes())
        assert stats["chunks_embedded"] > 0, stats
        assert stats["errors"] == 0, stats

    def test_failure_is_logged_at_warning_by_the_indexer(
        self, tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="trelix.indexing"):
            _run(tmp_path, _RecordingSummarizer(fail_paths={"mod0.py"}), files=2)
        assert any("mod0.py" in r.getMessage() for r in caplog.records), caplog.text


class TestSummarizerSurfacesItsOwnFailure:
    """The reason for the "" must reach the default log level, not only DEBUG."""

    @staticmethod
    def _symbols() -> list[Symbol]:
        return [
            Symbol(
                file_id=1,
                name="login",
                qualified_name="login",
                kind=SymbolKind.FUNCTION,
                line_start=1,
                line_end=10,
                signature="def login(u, p)",
                body="def login(u, p): pass",
            )
        ]

    def test_llm_error_logs_at_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        from trelix.core.models import Language
        from trelix.indexing.file_summarizer import FileSummarizer

        client = MagicMock()
        client.complete.side_effect = RuntimeError("429 too many requests")
        summarizer = FileSummarizer(client=client)
        with caplog.at_level(logging.WARNING, logger="trelix.indexing.file_summarizer"):
            assert summarizer.summarize("src/auth.py", self._symbols(), Language.PYTHON) == ""
        assert "src/auth.py" in caplog.text
        assert "429" in caplog.text

    def test_blank_llm_response_logs_at_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A model that answers with whitespace is a failure, not a summary."""
        from trelix.core.models import Language
        from trelix.indexing.file_summarizer import FileSummarizer

        client = MagicMock()
        client.complete.return_value = MagicMock(content="   \n ")
        summarizer = FileSummarizer(client=client)
        with caplog.at_level(logging.WARNING, logger="trelix.indexing.file_summarizer"):
            assert summarizer.summarize("src/auth.py", self._symbols(), Language.PYTHON) == ""
        assert "src/auth.py" in caplog.text


# ---------------------------------------------------------------------------
# 2. Concurrency
# ---------------------------------------------------------------------------


class TestSummariesRunConcurrently:
    def test_calls_overlap(self, tmp_path: pathlib.Path) -> None:
        summarizer = _RecordingSummarizer(latency=0.15)
        _run(tmp_path, summarizer, files=8, workers=4)
        assert summarizer.peak_in_flight > 1, (
            f"Phase 2.5 is still sequential: peak in-flight={summarizer.peak_in_flight}"
        )

    def test_concurrency_is_bounded_by_worker_count(self, tmp_path: pathlib.Path) -> None:
        summarizer = _RecordingSummarizer(latency=0.15)
        _run(tmp_path, summarizer, files=12, workers=3)
        assert summarizer.peak_in_flight <= 3, summarizer.peak_in_flight

    def test_every_file_is_still_summarized(self, tmp_path: pathlib.Path) -> None:
        summarizer = _RecordingSummarizer(latency=0.01)
        _, stats = _run(tmp_path, summarizer, files=8, workers=4)
        assert sorted(summarizer.calls) == sorted(f"mod{i}.py" for i in range(8))
        assert stats["file_summaries_generated"] == 8, stats

    def test_the_llm_calls_leave_the_main_thread(self, tmp_path: pathlib.Path) -> None:
        summarizer = _RecordingSummarizer(latency=0.05)
        _run(tmp_path, summarizer, files=8, workers=4)
        assert threading.get_ident() not in summarizer.threads, (
            "summaries ran on the main thread — the pool is not doing any work"
        )

    def test_default_worker_count_is_conservative(self, tmp_path: pathlib.Path) -> None:
        """Not 8-way by default: there is no per-provider RPM knowledge to lean on."""
        indexer = _make_indexer(str(tmp_path))
        assert 2 <= indexer._summary_workers <= 4, indexer._summary_workers


class TestDbWritesStayOnTheCallingThread:
    """`upsert_file_summary` writes the shared sqlite3.Connection without _conn_lock."""

    def test_summary_rows_are_written_from_one_thread(self, tmp_path: pathlib.Path) -> None:
        _write_files(tmp_path, 8)
        indexer = _make_indexer(str(tmp_path))
        indexer._file_summarizer = _RecordingSummarizer(latency=0.05)
        indexer._summary_workers = 4

        writer_threads: set[int] = set()
        real_upsert = indexer.db.upsert_file_summary

        def spy(file_id: int, summary: str, chunk_id: int | None = None) -> int:
            writer_threads.add(threading.get_ident())
            return real_upsert(file_id, summary, chunk_id)

        indexer.db.upsert_file_summary = spy  # type: ignore[method-assign]
        with _quiet_progress():
            indexer.index()

        assert writer_threads == {threading.get_ident()}, (
            f"file_summaries was written from {len(writer_threads)} threads through an "
            "unlocked sqlite3.Connection"
        )

    def test_the_rows_are_actually_there(self, tmp_path: pathlib.Path) -> None:
        indexer, _ = _run(tmp_path, _RecordingSummarizer(), files=6)
        rows = indexer.db._conn.execute("SELECT COUNT(*) FROM file_summaries").fetchone()[0]
        assert rows == 6, rows


# ---------------------------------------------------------------------------
# 3. Chat-side rate limiter
# ---------------------------------------------------------------------------


class TestRpmRateLimiter:
    def test_zero_limit_never_sleeps(self) -> None:
        from trelix.indexing.indexer import _RpmRateLimiter

        limiter = _RpmRateLimiter(0)
        t0 = time.monotonic()
        for _ in range(50):
            limiter.acquire()
        assert time.monotonic() - t0 < 0.5

    def test_under_limit_never_sleeps(self) -> None:
        from trelix.indexing.indexer import _RpmRateLimiter

        limiter = _RpmRateLimiter(60)
        t0 = time.monotonic()
        for _ in range(60):
            limiter.acquire()
        assert time.monotonic() - t0 < 0.5

    def test_over_limit_waits_for_the_window(self) -> None:
        from trelix.indexing.indexer import _RpmRateLimiter

        clock = _FakeClock()
        limiter = _RpmRateLimiter(2, sleep=clock.sleep, monotonic=clock.now)
        for _ in range(3):
            limiter.acquire()
        assert clock.slept, "the 3rd request in a 2/min window did not wait"
        assert 59.0 < clock.slept[0] <= 61.0, clock.slept

    def test_the_window_slides_rather_than_resetting(self) -> None:
        """A fixed window admits 2x the limit across a boundary; a sliding one does not."""
        from trelix.indexing.indexer import _RpmRateLimiter

        clock = _FakeClock()
        limiter = _RpmRateLimiter(2, sleep=clock.sleep, monotonic=clock.now)
        limiter.acquire()
        limiter.acquire()
        clock.advance(59.0)  # both admissions still inside the trailing 60 s
        limiter.acquire()
        assert limiter.waits == 1, "the 3rd request slipped through a reset window"

    def test_limiter_is_thread_safe(self) -> None:
        """Exactly `limit` of 8 racing threads may pass without waiting.

        Without the lock, all 8 read the same under-limit state and none waits — the
        burst that a 429 is made of. `sleep` raises here so a thread that reaches the
        wait path is counted rather than served.
        """
        from trelix.indexing.indexer import _RpmRateLimiter

        class _Throttled(Exception):
            pass

        def refuse_to_wait(_seconds: float) -> None:
            raise _Throttled

        limiter = _RpmRateLimiter(4, sleep=refuse_to_wait)
        barrier = threading.Barrier(8)
        admitted: list[int] = []
        throttled: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            try:
                limiter.acquire()
            except _Throttled:
                with lock:
                    throttled.append(1)
            else:
                with lock:
                    admitted.append(1)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(admitted) == 4, f"admitted {len(admitted)} of 8 into a 4/min window"
        assert len(throttled) == 4, len(throttled)


class TestChatCallsGoThroughTheLimiter:
    def test_every_summary_acquires_the_limiter(self, tmp_path: pathlib.Path) -> None:
        from trelix.indexing import indexer as indexer_mod

        acquisitions: list[int] = []
        real_acquire = indexer_mod._RpmRateLimiter.acquire

        def spy(self: Any) -> None:  # noqa: ANN401
            acquisitions.append(1)
            real_acquire(self)

        with patch.object(indexer_mod._RpmRateLimiter, "acquire", spy):
            _run(tmp_path, _RecordingSummarizer(), files=5, workers=2)
        assert len(acquisitions) == 5, acquisitions

    def test_rpm_default_is_set_and_finite(self, tmp_path: pathlib.Path) -> None:
        indexer = _make_indexer(str(tmp_path))
        assert indexer._summary_rpm > 0, (
            "an unlimited chat-side default is what retry.py's backoff exists to paper over"
        )


# ---------------------------------------------------------------------------
# 4. Side embeds honour the embedder TPM ceiling
# ---------------------------------------------------------------------------


class TestSummaryEmbedsAreThrottledAndBatched:
    def test_summary_embed_acquires_the_tpm_limiter(self, tmp_path: pathlib.Path) -> None:
        from trelix.indexing import indexer as indexer_mod

        acquired: list[int] = []
        real_acquire = indexer_mod._TpmRateLimiter.acquire

        def spy(self: Any, tokens: int) -> None:  # noqa: ANN401
            acquired.append(tokens)
            real_acquire(self, tokens)

        with patch.object(indexer_mod._TpmRateLimiter, "acquire", spy):
            _run(tmp_path, _RecordingSummarizer(), files=4)
        assert acquired, "the summary embed still bypasses the TPM limiter"
        assert all(t > 0 for t in acquired), acquired

    def test_summaries_are_embedded_in_batches_not_one_call_per_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        embedder = _FakeEmbedder()
        _run(tmp_path, _RecordingSummarizer(), files=6, embedder=embedder)
        summary_calls = [c for c in embedder.calls if c and c[0].startswith("SUMMARY")]
        assert summary_calls, "no summary embed happened at all"
        assert len(summary_calls) < 6, (
            f"one embed API call per file is still happening: {len(summary_calls)} calls"
        )

    def test_multi_granularity_embed_acquires_the_tpm_limiter(self, tmp_path: pathlib.Path) -> None:
        """Phase 2.6's per-symbol embed bypassed the limiter too, in the same way.

        MultiGranularityChunker is stubbed because the real one needs tree-sitter, which
        this suite deliberately runs without (see _FakeParser).
        """
        from trelix.indexing import indexer as indexer_mod
        from trelix.indexing.multi_granularity import Granularity, SubSymbolChunk

        _write_files(tmp_path, 3)
        indexer = _make_indexer(str(tmp_path))
        indexer._file_summarizer = None  # isolate Phase 2.6 from Phase 2.5
        indexer.config.chunker.multi_granularity_enabled = True

        def one_sub_chunk(sym: Any, granularities: Any) -> list[SubSymbolChunk]:  # noqa: ANN401
            return [
                SubSymbolChunk(
                    parent_symbol_id=sym.id,
                    granularity=Granularity.BLOCK,
                    chunk_text="return 0",
                    line_start=1,
                    line_end=1,
                    token_count=3,
                )
            ]

        stub_chunker = MagicMock()
        stub_chunker.extract_sub_chunks.side_effect = one_sub_chunk

        acquired: list[int] = []
        real_acquire = indexer_mod._TpmRateLimiter.acquire

        def spy(self: Any, tokens: int) -> None:  # noqa: ANN401
            acquired.append(tokens)
            real_acquire(self, tokens)

        with (
            patch(
                "trelix.indexing.multi_granularity.MultiGranularityChunker",
                return_value=stub_chunker,
            ),
            patch.object(indexer_mod._TpmRateLimiter, "acquire", spy),
            _quiet_progress(),
        ):
            indexer.index()
        assert stub_chunker.extract_sub_chunks.called, "Phase 2.6 did not run"
        assert indexer.vector_store.sub_chunk_embeddings, (
            "the sub-chunk embed did not complete, so the acquire below proves less"
        )
        assert acquired, "the multi-granularity sub-chunk embed still bypasses the TPM limiter"
