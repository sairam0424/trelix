"""
Indexer: orchestrates the full indexing pipeline.

Three-phase design for large-repo performance:

  Phase 1 — parallel parse
    N threads read + parse files concurrently (file I/O + tree-sitter both
    release the GIL, so threading gives real speedup).

  Phase 2 — sequential DB write + chunk
    Symbols and chunks are inserted in the main thread to keep parent_id
    remapping consistent (local parse indices → real DB row ids).

  Phase 2.5 — concurrent LLM file summaries  (opt-in)
    Runs between Phase 2 and Phase 3, not inside Phase 2's serial loop where it
    used to sit: measured at 428 summaries spanning 1034.2 s of an 1153.7 s run
    (89.6% of it), median inter-file gap 2.386 s. `_summarize_files()` fans the
    chat calls out over `_summary_workers` threads (all five chat backends are
    sync `complete()` + `@with_retry`, so threads, not asyncio) behind a
    `_RpmRateLimiter` — the only chat-side rate limit in trelix; `tpm_limit` is
    embedder-only. DB writes and summary embeds stay on the main thread because
    `Database.upsert_file_summary` does not hold `_conn_lock`.

  Phase 3 — async concurrent batch embed  (U5)
    Up to 4 API calls run concurrently via asyncio.gather + Semaphore(4).
    _make_token_batches() groups chunks so each batch stays under
    embed_max_tokens_per_batch tokens (prevents request-size errors).
    _AsyncTpmRateLimiter uses asyncio.sleep (non-blocking) to stay within
    the configured Azure TPM ceiling — we never exceed the quota.
    vector_store.upsert_batch() is sync → called in a thread executor.
    A batch that cannot be embedded or stored aborts the run with
    `PartialIndexError` rather than letting a store-level exception escape the
    gather: see that class and `Indexer._partial_index_error`.

  Phase 4 — cross-file resolution
    Call-edge targets and import file_ids are resolved after every file
    has been inserted (same second-pass as before).

parent_id / caller_id convention:
  During parsing, parent_id and caller_id are LOCAL INDICES (0-based) into
  the per-file symbol list.  Phase 2 remaps them to real DB ids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from trelix.indexing.file_summarizer import FileSummarizer

import asyncio
import hashlib
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from trelix.core.config import IndexConfig
from trelix.core.console_safety import safe_text as _safe_text
from trelix.core.models import IndexedFile, Language, Symbol
from trelix.embedder.base import BaseEmbedder, make_embedder
from trelix.indexing.chunker import Chunker, ContextualChunker
from trelix.indexing.parser.base import ParseResult
from trelix.indexing.parser.registry import get_parser
from trelix.indexing.walker import FileWalker, detect_language
from trelix.store.db import Database
from trelix.store.vector import BaseVectorStore, make_vector_store

logger = logging.getLogger("trelix.indexing")


# ---------------------------------------------------------------------------
# Phase 3 abort
# ---------------------------------------------------------------------------


class PartialIndexError(RuntimeError):
    """Phase 3 could not write every embedding batch, so the index is incomplete.

    Exists because the vector stores now REFUSE to paper over a failed write
    rather than adding anyway (see `LanceVectorStore.upsert_batch`: a failed
    delete makes the following add an append, so one chunk_id gains a row per
    re-index). That was the right call, but it gave Phase 3 an exception nobody
    handled: it escaped `asyncio.gather` as a bare store-level message
    ("LanceDB upsert_batch aborted: refresh/delete of 1 chunk_id(s) failed"),
    which `cli/main.py` prints as its single line of output. True, and useless —
    it names neither the scope of the damage nor the way out.

    Aborting is still the right behaviour; it just has to be a deliberate one.
    `str(self)` therefore carries the whole report, because that one CLI line
    and `index_file()`'s `{"status": "error", "error": ...}` are the only places
    it surfaces.
    """


# ---------------------------------------------------------------------------
# Internal data-transfer objects (not part of the public API)
# ---------------------------------------------------------------------------


@dataclass
class _ParsedFile:
    """Carries the result of Phase 1 for a single file."""

    file: IndexedFile
    parse_result: ParseResult | None  # ParseResult | None
    skipped: bool = False  # True  → hash unchanged, nothing to do
    error: str | None = None  # non-None → parse failed


@dataclass
class _PendingChunk:
    """A chunk that has been inserted into the DB and is waiting to be embedded."""

    chunk_id: int
    chunk_text: str
    token_count: int


@dataclass
class _PendingSummary:
    """One file's Phase 2.5 summary request, deferred out of the serial Phase 2.

    Carries the parsed symbols rather than a file path so the worker thread needs no
    disk or DB access — the chat call is the only thing that leaves the main thread.
    """

    file_id: int
    rel_path: str
    symbols: list[Symbol] = field(default_factory=list)
    language: Language = Language.UNKNOWN


# ---------------------------------------------------------------------------
# Phase 2.5 concurrency defaults
#
# NOT config fields: IndexConfig has no chat-side concurrency or rate-limit
# knobs (`tpm_limit` exists once, on EmbedderConfig, and is consumed only by the
# Phase 3 embed path). Read from the environment here so this file can carry the
# fix on its own; the lead should promote both to IndexConfig fields
# (TRELIX_FILE_SUMMARY_WORKERS / TRELIX_FILE_SUMMARY_RPM) so they appear in
# `trelix config` alongside every other tunable.
#
# Why 4 and 60, and not the 8 the measurement suggests:
#   - Measured sequential rate was 428 summaries / 1034.2 s = 24.8 requests/min,
#     median inter-file gap 2.386 s.
#   - 4 workers at that latency would issue ~100 requests/min if nothing capped
#     them. Capping at 60 keeps the LIMITER the binding constraint rather than the
#     pool size, so raising workers cannot silently raise the request rate.
#   - 60/min is 2.4x the measured rate: ~19 min of Phase 2.5 becomes ~8 min. Less
#     than the ~4 min an 8-way fan-out would give, and deliberately so — trelix
#     supports five chat backends and knows none of their per-model RPM ceilings,
#     so the only honest default is one that stays under the lowest plausible one.
#     Anyone who knows their own quota raises it.
# ---------------------------------------------------------------------------

_DEFAULT_SUMMARY_WORKERS = 4
_DEFAULT_SUMMARY_RPM = 60


def _env_int(name: str, default: int) -> int:
    """Read a non-negative int from the environment, falling back on anything unparseable.

    Falls back loudly rather than raising: a typo in a performance knob must not stop
    an index run, but it must not silently read as 0 (= unlimited) either.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer — using default %d", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s=%d is negative — using default %d", name, value, default)
        return default
    return value


# ---------------------------------------------------------------------------
# Rate limiters
# ---------------------------------------------------------------------------


class _RpmRateLimiter:
    """
    Sliding 60-second requests-per-minute guard (sync, thread-safe).

    Exists because trelix had NO chat-side rate limiter of any kind. Without one,
    fanning Phase 2.5's chat calls out across threads leans entirely on
    `core/retry.py`'s backoff, which converts a rate-limit into latency and billed
    retry attempts instead of preventing it.

    Requests, not tokens: a chat completion's output length is unknown until it
    returns, so there is nothing to reserve up front the way _TpmRateLimiter can for
    an embed batch whose token count is already computed.

    A true sliding window (deque of the last `limit` admission times), not the fixed
    window _TpmRateLimiter uses. A fixed window lets 2x the limit through across a
    boundary, which under an 8-way fan-out is exactly the burst that trips a 429.

    rpm_limit <= 0  →  unlimited (no waiting).

    `sleep`/`monotonic` are injectable together so tests can assert the wait without
    spending 60 seconds of wall clock on it.
    """

    def __init__(
        self,
        rpm_limit: int,
        console: Console | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = rpm_limit
        self._console = console or Console()
        self._sleep = sleep
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._admitted: deque[float] = deque()
        self._waits = 0

    @property
    def waits(self) -> int:
        """How many acquisitions had to wait — reported so throttling is visible."""
        return self._waits

    def acquire(self) -> None:
        """Block until one more request fits inside the trailing 60-second window."""
        if self._limit <= 0:
            return
        # The sleep happens INSIDE the lock on purpose: the whole point is to stop the
        # other workers issuing requests during the wait. Holding it costs nothing but
        # the wait itself, which every worker would otherwise have to take anyway.
        with self._lock:
            while True:
                now = self._monotonic()
                while self._admitted and now - self._admitted[0] >= 60.0:
                    self._admitted.popleft()
                if len(self._admitted) < self._limit:
                    self._admitted.append(now)
                    return
                wait = 60.0 - (now - self._admitted[0]) + 0.05
                self._waits += 1
                self._sleep(max(0.0, wait))


# ---------------------------------------------------------------------------
# TPM rate limiters
# ---------------------------------------------------------------------------


class _TpmRateLimiter:
    """
    Sliding 60-second window TPM guard (sync).

    Call .acquire(tokens) before every embedding API call.  If adding
    `tokens` to the running total would exceed tpm_limit within the current
    window, the method sleeps until the window resets.

    tpm_limit = 0  →  unlimited (no sleeping, used for local embedder).
    """

    def __init__(self, tpm_limit: int, console: Console | None = None) -> None:
        self._limit = tpm_limit
        self._used = 0
        self._window_start = time.monotonic()
        self._console = console or Console()

    def acquire(self, tokens: int) -> None:
        if self._limit <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._window_start
        if elapsed >= 60.0:
            # Previous window expired — reset
            self._used = 0
            self._window_start = now
            elapsed = 0.0
        if self._used + tokens > self._limit:
            wait = 61.0 - elapsed  # +1 s safety buffer
            self._console.print(
                f"[yellow]⏸  TPM limit ({self._limit:,}/min) reached — "
                f"waiting {wait:.1f} s[/yellow]"
            )
            time.sleep(max(0.0, wait))
            self._used = 0
            self._window_start = time.monotonic()
        self._used += tokens


class _AsyncTpmRateLimiter:
    """
    Async sliding 60-second window TPM guard (U5).

    Identical logic to _TpmRateLimiter but uses asyncio.sleep (non-blocking)
    and an asyncio.Lock to prevent multiple concurrent coroutines from all
    seeing the same under-limit state at the same instant.

    tpm_limit = 0  →  unlimited (no sleeping).
    """

    def __init__(self, tpm_limit: int, console: Console | None = None) -> None:
        self._limit = tpm_limit
        self._used = 0
        self._window_start = time.monotonic()
        self._console = console or Console()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> None:
        if self._limit <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._window_start
            if elapsed >= 60.0:
                self._used = 0
                self._window_start = now
                elapsed = 0.0
            if self._used + tokens > self._limit:
                wait = 61.0 - elapsed  # +1 s safety buffer
                self._console.print(
                    f"[yellow]⏸  TPM limit ({self._limit:,}/min) reached — "
                    f"waiting {wait:.1f} s[/yellow]"
                )
                await asyncio.sleep(max(0.0, wait))
                self._used = 0
                self._window_start = time.monotonic()
            self._used += tokens


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


class Indexer:
    """
    Top-level indexer.  Call `index()` to build or update the index.

    Usage:
        config = IndexConfig(repo_path="/path/to/repo")
        stats  = Indexer(config).index()
    """

    # Phase weight allocation for overall progress (must sum to 1.0)
    _PHASE_WEIGHTS = {
        0: (0.00, 0.05),  # discovery
        1: (0.05, 0.30),  # parse
        2: (0.30, 0.50),  # insert / chunk
        3: (0.50, 0.95),  # embed
        4: (0.95, 1.00),  # resolve
    }

    # Minimum number of files changed in a single index_file() call-site batch
    # before the O(total_calls + total_imports) global resolve passes run.
    # For single-file watch events this is 1, so resolve is skipped; the next
    # full index() run (or a batch >= this threshold) will catch any new edges.
    _FULL_RESOLVE_THRESHOLD = 5

    def __init__(
        self,
        config: IndexConfig,
        quiet: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self._console = Console(quiet=quiet)
        self._progress_cb = progress_callback
        db_path = config.db_path_absolute
        self.db = Database(db_path)
        # Load embedder first so we can query its actual dimension
        self.embedder: BaseEmbedder = make_embedder(config.embedder)
        self.vector_store: BaseVectorStore = make_vector_store(
            config=config,
            dimension=self.embedder.dimension,
        )
        self.chunker = self._build_chunker(config)
        self.walker = FileWalker(config)
        self._file_summarizer = self._build_file_summarizer(config)

        # Phase 2.5 fan-out. See _DEFAULT_SUMMARY_WORKERS for why these numbers.
        self._summary_workers = max(
            1, _env_int("TRELIX_FILE_SUMMARY_WORKERS", _DEFAULT_SUMMARY_WORKERS)
        )
        self._summary_rpm = _env_int("TRELIX_FILE_SUMMARY_RPM", _DEFAULT_SUMMARY_RPM)

        # Shared limiter for the embed calls OUTSIDE Phase 3: the file-summary embeds
        # and the multi-granularity sub-chunk embeds, both of which called
        # self.embedder.embed() directly and so spent the same Azure TPM quota the rest
        # of the indexer carefully rations. Separate instance from Phase 3's limiter
        # because Phase 3's is async; safe only because these phases never overlap in
        # time (2.5 and 2.6 both complete before Phase 3 starts).
        self._side_embed_limiter = _TpmRateLimiter(config.embedder.tpm_limit, console=self._console)

        # Dimension guard: detect provider switch mismatches at startup
        try:
            from trelix.store.dimension_guard import DimensionGuard, DimensionMismatchError

            DimensionGuard.check(
                self.db,
                current_dimension=self.embedder.dimension,
                provider=config.embedder.provider,
            )
        except DimensionMismatchError:
            raise  # Re-raise with the clear user-facing message
        except Exception as exc:
            logger.debug("DimensionGuard.check failed (non-fatal): %s", exc)

    def _build_chunker(self, config: IndexConfig) -> Chunker:
        """
        Return a ContextualChunker if contextual=True in ChunkerConfig,
        otherwise a plain Chunker.  The LLM client is built here so it is
        created once and reused across all files.
        """
        if not config.chunker.contextual:
            return Chunker(config.chunker)
        try:
            from trelix.llm.factory import build_chat_client

            llm_client = build_chat_client(config.llm)
            logger.info(
                "ContextualChunker: using %s provider, model=%s",
                config.llm.provider,
                config.llm.model,
            )
            return ContextualChunker(config.chunker, llm_client=llm_client)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ContextualChunker: could not build LLM client (%s) — falling back to base Chunker",
                exc,
            )
            return Chunker(config.chunker)

    def _build_file_summarizer(self, config: IndexConfig) -> object | None:
        """
        Return a FileSummarizer when file_summaries_enabled=True, else None.

        The LLM client is shared with the contextual chunker where possible —
        if that client was already built it is recreated here (cheap; clients
        are thin wrappers). Returns None on any build failure so that the
        indexer degrades gracefully without crashing.
        """
        if not config.file_summaries_enabled:
            return None
        try:
            from trelix.indexing.file_summarizer import FileSummarizer
            from trelix.llm.factory import build_chat_client

            llm_client = build_chat_client(config.llm)
            logger.info(
                "FileSummarizer: enabled, using %s provider, model=%s",
                config.llm.provider,
                config.llm.model,
            )
            return FileSummarizer(client=llm_client)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FileSummarizer: could not build LLM client (%s) — file summaries disabled",
                exc,
            )
            return None

    def _report_progress(
        self,
        phase: int,
        phase_label: str,
        phase_fraction: float,
        stats: dict[str, Any],
    ) -> None:
        """Call the progress callback with an overall progress value 0→1."""
        if self._progress_cb is None:
            return
        lo, hi = self._PHASE_WEIGHTS[phase]
        overall = lo + (hi - lo) * min(max(phase_fraction, 0.0), 1.0)
        self._progress_cb(
            {
                "phase": phase,
                "phase_label": phase_label,
                "progress": round(overall, 4),
                "stats": dict(stats),
            }
        )

    # ──────────────────────────────────────────────────────────────────────
    # Streaming indexing pipeline (Plan C — TRELIX_INDEXER_STREAMING=true)
    # ──────────────────────────────────────────────────────────────────────

    def _iter_files(self, repo_path: str) -> Iterator[IndexedFile]:
        """
        Generator yielding IndexedFile objects for the given repo path.

        Used by the streaming indexing pipeline to avoid buffering all files
        in memory before parsing begins.  Yields files as they are discovered
        by the walker, allowing the parse/embed pipeline to start immediately.

        Memory cost: O(1) — one file object in memory at a time vs O(n) for
        the existing list(walker.walk()) pattern.
        """
        yield from self.walker.walk()

    def _index_streaming(self, repo_path: str) -> dict[str, Any]:
        """
        Streaming indexing pipeline — generator-based, bounded memory.

        Files are yielded one at a time from _iter_files() via a producer
        thread and consumed by the main thread through a bounded Queue(64).
        Memory usage is O(queue_size) rather than O(repo_size).

        Processes each file via index_file() which handles parse, insert,
        chunk, and embed in a single call (same code path as watch-mode).
        Skips the full-repo cross-file resolution pass used by batch index()
        and instead runs a single resolve pass at the end.
        """
        import queue
        import threading

        QUEUE_SIZE = 64
        results: dict[str, Any] = {
            "files_found": 0,
            "files_unreadable": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "symbols_extracted": 0,
            "chunks_total": 0,
            "chunks_embedded": 0,
            # Same two keys, same meaning, as index()'s dict — a stat that exists on only
            # one of the two pipelines is a stat the CLI cannot render honestly.
            "chunks_missing_vectors": None,
            "chunks_reconciled": 0,
            "errors": 0,
            "elapsed_seconds": 0.0,
        }
        t_start = time.perf_counter()

        file_queue: queue.Queue[IndexedFile | object] = queue.Queue(maxsize=QUEUE_SIZE)
        sentinel = object()

        def producer() -> None:
            """Walk the repo and put each IndexedFile onto the queue."""
            try:
                for indexed_file in self._iter_files(repo_path):
                    file_queue.put(indexed_file)
            except Exception as exc:
                logger.warning("Streaming indexer producer failed: %s", exc)
            finally:
                # Always enqueue sentinel so consumer never hangs.
                file_queue.put(sentinel)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        while True:
            item = file_queue.get()
            if item is sentinel:
                break

            assert isinstance(item, IndexedFile)  # sentinel excluded above

            results["files_found"] += 1

            # Skip unchanged files when incremental mode is enabled
            if self.config.incremental and self.db.get_file_hash(item.rel_path) == item.hash:
                results["files_skipped"] += 1
                continue

            try:
                call_result = self.index_file(str(item.path))
                if call_result.get("status") == "ok":
                    if not call_result.get("skipped"):
                        results["files_indexed"] += 1
                        results["symbols_extracted"] += call_result.get("symbols_updated", 0)
                        results["chunks_embedded"] += call_result.get("chunks_updated", 0)
                        results["chunks_total"] += call_result.get("chunks_updated", 0)
                    else:
                        results["files_skipped"] += 1
                else:
                    results["errors"] += 1
                    logger.debug(
                        "Streaming index: error for %s: %s",
                        item.path,
                        call_result.get("error"),
                    )
            except Exception as exc:
                logger.debug("Streaming index: unhandled error for %s: %s", item.path, exc)
                results["errors"] += 1

        producer_thread.join()

        # Crash recovery, once per run at the pipeline level rather than per file.
        #
        # Deliberately NOT inside index_file(), which is what every file here funnels
        # through: a check there would run a full store scan per file — 64 per streaming
        # batch — for a repo-wide answer that changes once. The same reasoning keeps the
        # WATCHING phase out of it: a save event on one unrelated file is the wrong trigger
        # for a repo-wide repair, and an unbounded store scan per filesystem event is the
        # wrong cost profile for a long-lived process.
        #
        # That is a fact about the watching phase, and an earlier version of this comment
        # generalised it to `trelix watch` as a whole. It is false of watch's STARTUP pass:
        # `cli/main.py`'s watch command calls `indexer.index()` before it starts watching,
        # and `index()` routes into this pipeline whenever TRELIX_INDEXER_STREAMING is set —
        # into its own reconcile, decided right after the incremental pre-filter, when it is
        # not — so the repair below is exactly what that pass runs on a streaming setup: 2
        # holed chunks reconciled and exactly 2 texts embedded, in
        # tests/unit/test_indexer_vector_repair.py::TestStreamingPipeline. What never
        # reconciles is the watching phase that follows: `FileWatcher` calls `index_file()`
        # per changed file, which has no repo-wide diff, so a hole opened while watch is
        # already running survives until the next startup or `trelix index`. `trelix index`
        # stays the repair path to name to a user — `trelix stats` names it too — because it
        # is the reliable route, not because watch cannot repair. Same split as
        # `_partial_index_error`'s docstring below; keep the two in step.
        #
        # No re-read filter needed here, unlike index(): the walk loop above has finished,
        # so nothing further deletes chunk rows before these ids are embedded. The rows are
        # read for their text regardless. Embeds through the sync path because this pipeline
        # has no async phase of its own.
        #
        # Skipped entirely when any file errored IN THIS RUN, and that guard is a cost guard
        # rather than a tidiness one. This repair runs AFTER the walk, unlike index()'s which
        # runs before it, so a file whose own Phase 3 just failed — embedding already paid
        # for, `index_file` having swallowed the failure into `status=error` — is now sitting
        # in the store as a hole. Repairing it here re-sends the identical text to the same
        # provider that just refused it: paid twice, failed twice. The holes do not go
        # anywhere; the next `trelix index` repairs them once the cause is fixed, which is
        # the same deal the batch pipeline offers.
        repair_ids, results["chunks_missing_vectors"] = self._chunks_missing_vectors()
        if repair_ids and results["errors"]:
            logger.warning(
                "%d chunk(s) have no vector, but %d file(s) failed in this run — not "
                "repairing now, because a chunk this run just failed to embed would be "
                "re-sent to the provider that refused it and paid for twice. Fix the "
                "failures and re-run `trelix index` to repair.",
                len(repair_ids),
                results["errors"],
            )
        elif repair_ids:
            repaired = [
                _PendingChunk(chunk_id=cid, chunk_text=text, token_count=tokens)
                for cid, text, tokens in self.db.get_chunk_text_and_tokens(repair_ids)
            ]
            if repaired:
                self._log_repair_intent(len(repaired))
                try:
                    self._batch_embed_and_store(repaired, results)
                    results["chunks_reconciled"] = len(repaired)
                    results["chunks_total"] += len(repaired)
                except Exception as exc:
                    # Contained rather than raised, matching every other failure in this
                    # loop: the files that DID index are committed, and aborting here would
                    # lose the resolution pass below for a repair that was already broken
                    # before this run started.
                    logger.warning("Repairing vector-less chunks failed: %s", exc)
                    results["errors"] += 1

        # Assigned here, after the producer has finished walking. Declaring the key in
        # the results dict without ever setting it made this path report 0 unconditionally
        # — a silent "the walk was complete" for the pipeline that is hardest to observe,
        # since it streams rather than materialising a file list.
        results["files_unreadable"] = len(self.walker.incomplete_paths)
        if not self.walker.walk_was_complete:
            skipped = self.walker.incomplete_paths
            self._console.print(
                f"[yellow]  {len(skipped)} path(s) could not be read and are missing "
                f"from this index: {_safe_text(', '.join(skipped[:5]))}"
                f"{' …' if len(skipped) > 5 else ''}[/yellow]"
            )

        # Cross-file resolution — single pass after all files are processed
        try:
            self.db.resolve_cross_file_calls()
            self.db.resolve_import_file_ids()
            self.db.resolve_cross_file_type_edges()
            self.db.resolve_angular_selectors()
        except Exception as exc:
            logger.debug("Streaming index: resolution pass failed (non-fatal): %s", exc)

        results["elapsed_seconds"] = round(time.perf_counter() - t_start, 2)
        self._record_provenance()
        return results

    def _capture_provenance(self) -> None:
        """Snapshot git and embedder state BEFORE any file is walked.

        Deliberately not captured at the end. File hashes are computed during the walk,
        so if a caller commits mid-run, end-of-run capture would pair the *new* commit
        with content from the old one — recording an index as more current than it is,
        which is the direction that causes silent wrong answers. Start-of-run capture
        errs the safe way: the index may be attributed to an older commit than the tree
        now at HEAD, which `trelix stats` reports as drift rather than hiding.
        """
        from trelix.store.provenance import capture_provenance

        self._provenance = capture_provenance(self.config)

    def _record_provenance(self) -> None:
        """Persist the provenance captured at the start of this run.

        Written at the end so a run that crashed partway through does not leave a record
        claiming a complete index. Read back by `trelix stats`, which previously had no
        way to answer "does this index reflect my worktree" — `index_metadata` held only
        the embedding dimension.
        """
        from trelix.store.provenance import write_provenance

        provenance = getattr(self, "_provenance", None)
        if provenance is None:
            logger.debug("No provenance captured for this run; nothing to record")
            return
        write_provenance(self.db, provenance)

    # ──────────────────────────────────────────────────────────────────────
    # Crash recovery: chunk rows a killed run left with no vector
    # ──────────────────────────────────────────────────────────────────────

    def _vector_store_is_empty(self) -> bool:
        """True only when the store demonstrably holds nothing — see the Phase 3 caller.

        `count()` is sentinel-inclusive, which is what this wants: a summary or sub-chunk
        row is still a row somebody paid for, so the store is not fresh.

        A store that cannot be counted answers False, not True. Everything gated on this
        becomes more cautious when the answer is unknown; guessing "empty" would stamp a
        dimension over an index that may already hold vectors of another width.
        """
        try:
            return self.vector_store.count() == 0
        except Exception as exc:
            logger.debug(
                "Could not count the vector store (%s: %s) — treating it as non-empty",
                type(exc).__name__,
                exc,
            )
            return False

    def _record_embedding_dimension(self) -> None:
        """Stamp `index_metadata` with the embedder's advertised width. Never fatal.

        Non-fatal because it protects future runs rather than this one: a run that embedded
        successfully must not be reported as failed because a metadata write did not land.
        """
        try:
            from trelix.store.dimension_guard import DimensionGuard

            DimensionGuard.record(
                self.db,
                dimension=self.embedder.dimension,
                provider=self.config.embedder.provider,
            )
        except Exception as exc:
            logger.debug("DimensionGuard.record failed (non-fatal): %s", exc)

    def _chunks_missing_vectors(self) -> tuple[list[int], int | None]:
        """Chunk rows this index holds no vector for. Returns (ids to repair, count).

        The count is `None` when the scan could not run at all — a store that cannot
        enumerate its ids must not read as zero holes, which is the same silent-health
        claim this whole path exists to remove.

        This is the crash-recovery counterpart to `PartialIndexError`, not a duplicate of
        it. That class covers Phase 3 RAISING: the run aborts deliberately, with a full
        report, and it is correct. It structurally cannot cover Phase 3 never getting to
        raise — SIGKILL, laptop sleep, a CI timeout, OOM — because nothing runs afterwards
        to report anything. The damage is identical and equally permanent: `_insert_one`
        commits `upsert_file` (hence `files.hash`) and the chunk rows in Phase 2, BEFORE
        this phase, so `index()`'s pre-filter skips those files forever and `_insert_one`'s
        per-symbol hash diff finds nothing to re-chunk even with `incremental=False`.

        Reproduced on a real 68,880-chunk index left holding 61,652 vectors by a
        mid-Phase-3 SIGKILL: 7,228 chunks (10.5%) permanently unretrievable, `index_metadata`
        emptied so `DimensionGuard.check` silently disarmed too, and a re-index reporting
        "Files walked 3,136 / Unchanged since last index 3,136 / Files to embed 0".

        Driven by OBSERVED missing vectors, deliberately, rather than by widening the
        change-detection rule. Forcing the affected files back through the parser does not
        work: `_insert_one` diffs `sha256(signature + body)` per symbol and returns early
        when nothing differs, so a recovery run over byte-identical files re-parses
        everything and embeds nothing (this is exactly why
        `db.invalidate_all_symbol_hashes` had to exist). It would also cost strictly more,
        re-embedding every chunk of every affected file instead of only the holes.
        """
        try:
            stored = self.vector_store.stored_chunk_ids()
            partition = self.db.all_chunk_ids()
            missing = sorted(partition.repairable - stored)
        except Exception as exc:
            # WARNING, not DEBUG, following the sparse phase below: the CLI runs at
            # WARNING, and a coverage check that silently produced nothing is how the
            # original bug stayed invisible. The run continues either way — a failed
            # check must never block an index, and must never claim health.
            logger.warning(
                "Could not check which chunks are missing vectors (%s: %s) — this run "
                "will index normally but will not repair any gap left by an interrupted "
                "run. `trelix stats` reports coverage.",
                type(exc).__name__,
                exc,
            )
            return [], None
        if partition.id_space_exhausted:
            # ERROR, not WARNING: unlike a hole, this one does not clear. Every backend's
            # `stored_chunk_ids()` treats ids at or above the offset as sub-chunk sentinels
            # and `search()` filters them out through `_is_chunk_id`, so a vector written
            # at such an id is unreachable no matter how often it is paid for. Reported and
            # then EXCLUDED from `missing` deliberately: counting these as holes is what
            # turns a permanently-unretrievable chunk into an unbounded recurring embedding
            # bill, one full re-embed of them per `trelix index` on a tree with nothing to
            # do. The fix is a re-key (renumber `chunks`, or raise the offset), not a
            # re-embed, so this run does not offer to spend on them.
            logger.error(
                "%d chunk(s) have an id at or above the vector store's sub-chunk offset "
                "(%d) — the chunk id space is exhausted. Their vectors would be filtered "
                "out of every search as sub-chunk sentinels, so re-embedding them cannot "
                "help and this run will not: they need a re-key, not a re-embed.",
                len(partition.id_space_exhausted),
                BaseVectorStore._SUB_CHUNK_OFFSET,
            )
        # Deliberately silent about the holes themselves: this function observes, it does
        # not decide. It used to announce "Re-embedding them in this run." here, which the
        # streaming caller then contradicted three lines later with "not repairing now" —
        # two WARNINGs back to back, the first one false, and the first one is what a user
        # reading top-to-bottom believes. Each caller now states its own outcome; see
        # `_log_repair_intent` and the skip branch in `_index_streaming`.
        return missing, len(missing)

    def _log_repair_intent(self, count: int) -> None:
        """Announce a repair that is about to happen, from the caller that will do it.

        Shared by both pipelines so the sentence cannot drift between them. Only ever
        called where the chunks really are about to be re-embedded, and only from a point
        where every hole predates this run: `index()` reconciles before Phase 1, and
        `_index_streaming` reconciles after its walk but skips the repair outright if any
        file failed in the same run.
        """
        logger.warning(
            "%d chunk(s) have no vector — an earlier run was interrupted after their "
            "chunk rows and file hashes were committed but before they were embedded. "
            "Re-embedding them in this run.",
            count,
        )

    def index(self) -> dict[str, Any]:
        # Captured before the routing below so both pipelines record it, and before any
        # file is walked so the commit matches the content that gets hashed.
        self._capture_provenance()

        # Route to streaming pipeline when enabled
        if getattr(self.config, "indexer", None) and self.config.indexer.streaming_enabled:
            return self._index_streaming(self.config.repo_path)

        t_start = time.perf_counter()
        stats: dict[str, Any] = {
            "files_found": 0,
            "files_unreadable": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "symbols_extracted": 0,
            "chunks_total": 0,
            "chunks_embedded": 0,
            # Phase 2.5 outcome, split three ways so "no summaries" cannot masquerade as
            # "all summaries". Always present, including when file_summaries_enabled is
            # False — three zeros beside a disabled flag is unambiguous, a missing key
            # is not.
            "file_summaries_generated": 0,
            "file_summaries_failed": 0,
            "file_summaries_embedded": 0,
            # Crash-recovery outcome, kept out of files_skipped on purpose — that key
            # already carries two meanings (hash unchanged, and no parser for the
            # language). `chunks_missing_vectors` is None when the check could not run,
            # which is NOT the same as zero holes; same principle as the three summary
            # keys above.
            "chunks_missing_vectors": None,
            "chunks_reconciled": 0,
            "errors": 0,
            "elapsed_seconds": 0.0,
        }

        logger.info("Starting indexing: repo=%s", self.config.repo_path)
        self._report_progress(0, "Discovering files…", 0.0, stats)
        files = list(self.walker.walk())
        stats["files_found"] = len(files)
        # Surfaced as a stat, not only as a log line: an unreadable directory drops its
        # whole subtree, so "files_found" alone cannot be read as the contents of the
        # repository. Anything that later DELETES rows for files the walk did not yield
        # must consult this first — a truncated walk is indistinguishable from files
        # having been removed, and acting on it destroys paid-for embeddings.
        stats["files_unreadable"] = len(self.walker.incomplete_paths)
        if not self.walker.walk_was_complete:
            skipped = self.walker.incomplete_paths
            self._console.print(
                f"[yellow]  {len(skipped)} path(s) could not be read and are missing "
                f"from this index: {_safe_text(', '.join(skipped[:5]))}"
                f"{' …' if len(skipped) > 5 else ''}[/yellow]"
            )
        self._report_progress(0, "Discovering files…", 1.0, stats)

        # Pre-filter: skip files whose hash hasn't changed (sequential, read-only DB)
        if self.config.incremental:
            to_parse = [f for f in files if self.db.get_file_hash(f.rel_path) != f.hash]
            stats["files_skipped"] = len(files) - len(to_parse)
        else:
            to_parse = files

        # Decided here, immediately after the pre-filter, because THIS is the line that
        # printed "Nothing to index — all files up to date." over an index missing 10.5%
        # of its vectors. Serviced further down, after Phase 2 — see there for why the
        # ids are re-read rather than used directly.
        repair_ids, stats["chunks_missing_vectors"] = self._chunks_missing_vectors()

        if not to_parse and not repair_ids:
            self._console.print("[green]Nothing to index — all files up to date.[/green]")
            return stats
        if repair_ids:
            self._console.print(
                f"[yellow]  Repairing {len(repair_ids)} chunk(s) left without a vector by "
                f"an interrupted run.[/yellow]"
            )

        # ── Phase 1: parallel parse ─────────────────────────────────────────
        self._console.print(
            f"[dim]  Phase 1/3: parsing {len(to_parse)} files "
            f"({self.config.parse_workers} workers)…[/dim]"
        )
        self._report_progress(1, "Parsing files…", 0.0, stats)
        parsed = self._parse_all(to_parse, stats)

        # ── Phase 2: sequential DB write + chunk ────────────────────────────
        self._console.print("[dim]  Phase 2/3: inserting symbols & building chunks…[/dim]")
        self._report_progress(2, "Building symbols & chunks…", 0.0, stats)
        pending, summary_requests = self._insert_and_chunk_all(parsed, stats)

        # ── Phase 2.5: concurrent file-level summaries ──────────────────────
        # Between Phase 2 and Phase 3 rather than inside Phase 2's serial loop. Kept
        # ahead of Phase 3 so the two never share a TPM window (see
        # self._side_embed_limiter) and so a summary failure still cannot cost a file
        # its chunk vectors.
        self._summarize_files(summary_requests, stats)

        # ── Crash recovery: fold the vector-less chunks into Phase 3's work ──
        #
        # This is the only correct point for it, and the re-read is load-bearing rather
        # than defensive. Phase 2 above DELETES the chunk rows of every changed symbol and
        # re-inserts replacements, so an id from `repair_ids` can be dead by now; embedding
        # it would write a vector with no chunk row, a fresh orphan. Re-reading through
        # `get_chunk_text_and_tokens` makes dead ids simply not come back, and is also
        # where `chunk_text` / `token_count` come from — no parse, no chunker call.
        #
        # Subtracting what Phase 2 already queued prevents paying twice for a chunk it just
        # re-created. `stats["chunks_total"]` is computed below, AFTER this, so repaired
        # chunks are counted, embedded by the existing Phase 3, and covered by the existing
        # `_partial_index_error` machinery — no second embed path.
        if repair_ids:
            already_queued = {p.chunk_id for p in pending}
            repaired = [
                _PendingChunk(chunk_id=cid, chunk_text=text, token_count=tokens)
                for cid, text, tokens in self.db.get_chunk_text_and_tokens(repair_ids)
                if cid not in already_queued
            ]
            if repaired:
                self._log_repair_intent(len(repaired))
            pending.extend(repaired)
            stats["chunks_reconciled"] = len(repaired)

        # ── Phase 3: async concurrent batch embed ───────────────────────────
        stats["chunks_total"] = len(pending)

        # Recorded BEFORE the embed when — and ONLY when — the store holds nothing yet.
        # Recording early is what re-arms the guard on an index whose store is still empty
        # when Phase 3 dies: `DimensionGuard.check` early-returns on a non-int stored value,
        # so an index with no recorded dimension has no protection against mixed-width
        # vectors at all. Over an EMPTY store the early stamp costs nothing to be wrong
        # about — no stored vector carries the other width, and the worst case is one loud
        # `DimensionMismatchError` whose `migrate-vectors --reset` remedy discards zero
        # embeddings.
        #
        # "Empty" is `_vector_store_is_empty()`, i.e. a sentinel-INCLUSIVE `count()`, and
        # that narrows this claim in one reachable case: Phase 2.5 above writes file-summary
        # vectors at `-file_id` BEFORE this phase, so on a fresh index with
        # `file_summaries_enabled` (opt-in, default False) a single summary makes the store
        # non-empty and the early stamp does not fire — that run's dying Phase 3 leaves the
        # guard disarmed. Left as is deliberately: the count is sentinel-inclusive because a
        # row somebody paid for is a row, and loosening it to "no chunk vectors" would stamp
        # over a store that already holds paid-for embeddings of another width, which is the
        # more expensive of the two mistakes.
        #
        # Over a NON-EMPTY store it is not free, and the `if pending` version of this was a
        # trap: a repair run against a pre-existing index would stamp the new provider's
        # width, Phase 3 would then raise `PartialIndexError`, and the correct provider was
        # locked out of its own index on the next run — with `migrate-vectors --reset`, i.e.
        # discard every paid-for embedding, offered as the way out. The comment that used to
        # justify it claimed `SQLiteVectorStore.__init__` had already created the vec0 table
        # at this width; that is true only for a table it actually created, and
        # `CREATE VIRTUAL TABLE IF NOT EXISTS` can neither widen nor narrow a pre-existing
        # one — which is exactly the crash-recovery case this path exists for. So a
        # populated store records LATE, after the embed, and a failed Phase 3 leaves the
        # stored width alone. The re-arming property survives where it is needed: on a
        # 61,652-vector index with the correct provider the embed succeeds and the late
        # record fires.
        #
        # Note `record()` stores `self.embedder.dimension` — the ADVERTISED width, not an
        # observed vector length. Late recording has the identical property, so neither
        # branch verifies it against what actually landed.
        record_early = bool(pending) and self._vector_store_is_empty()
        if record_early:
            self._record_embedding_dimension()

        if pending:
            total_tokens = sum(p.token_count for p in pending)
            self._console.print(
                f"[dim]  Phase 3/3: embedding {len(pending)} chunks "
                f"({total_tokens:,} tokens, up to 4 concurrent API calls)…[/dim]"
            )
            self._report_progress(3, "Embedding chunks…", 0.0, stats)
            asyncio.run(self._batch_embed_and_store_async(pending, stats))

        # The late record, for every store that already held vectors. Unreachable when the
        # embed above raised, which is the point.
        if pending and not record_early:
            self._record_embedding_dimension()

        # ── Sparse embedding phase (SPLADE-Code) — runs when sparse_enabled=True ──
        if self.config.retrieval.sparse_enabled and pending:
            try:
                from trelix.embedder.sparse import SparseEmbedder
                from trelix.store.sparse_store import SparseStore

                sparse_emb = SparseEmbedder(
                    model_name=self.config.sparse.model,
                    top_k=self.config.sparse.top_k_tokens,
                    # Previously not passed at all, and SparseEmbedder did not accept it:
                    # `SparseConfig.batch_size` was referenced nowhere in src/ while
                    # docs/architecture.md documented it as part of the signature. Every
                    # chunk went through a single forward pass, which for this repo's
                    # 10,700 chunks is a 668 GB logits tensor.
                    batch_size=self.config.sparse.batch_size,
                )
                sparse_store = SparseStore(self.config.db_path_absolute)
                texts = [pc.chunk_text for pc in pending]
                sparse_vecs = sparse_emb.embed(texts)
                pairs = [(int(pc.chunk_id), vec) for pc, vec in zip(pending, sparse_vecs) if vec]
                if pairs:
                    sparse_store.upsert_batch(pairs)
                    logger.info("Sparse embedding: indexed %d chunks", len(pairs))
                else:
                    # SparseEmbedder.embed() returns a dict per text and an EMPTY dict
                    # for every one of them when the model fails to load, so `pairs`
                    # filters to nothing and the phase used to finish silently. That is
                    # how sparse_embeddings stayed at 0 rows with the flag switched on
                    # and no indication anywhere that the leg was dead.
                    logger.warning(
                        "Sparse embedding produced no vectors for %d chunks — "
                        "sparse_embeddings will be empty and the sparse retrieval leg "
                        "inert. Check that TRELIX_SPARSE_MODEL (%s) is a loadable "
                        "BERT-family SPLADE model and that trelix[sparse] is installed.",
                        len(pending),
                        self.config.sparse.model,
                    )
            except Exception as exc:
                # WARNING, not DEBUG: this is an explicitly enabled feature producing
                # nothing, and the CLI runs at WARNING.
                logger.warning("Sparse embedding phase failed (non-fatal): %s", exc)

        # ── Phase 4: cross-file resolution ──────────────────────────────────
        self._report_progress(4, "Resolving cross-file references…", 0.0, stats)
        resolved_calls = self.db.resolve_cross_file_calls()
        resolved_imports = self.db.resolve_import_file_ids()
        resolved_types = self.db.resolve_cross_file_type_edges()
        resolved_angular = self.db.resolve_angular_selectors()
        if resolved_calls or resolved_imports or resolved_types or resolved_angular:
            self._console.print(
                f"[dim]  Resolution: {resolved_calls} call edges, "
                f"{resolved_imports} import paths, "
                f"{resolved_types} type edges, "
                f"{resolved_angular} Angular selector edges[/dim]"
            )
        self._report_progress(4, "Done", 1.0, stats)

        stats["elapsed_seconds"] = round(time.perf_counter() - t_start, 2)
        logger.info(
            "Indexing complete: files_indexed=%d files_skipped=%d symbols=%d "
            "chunks=%d repaired=%d summaries=%d/%d embedded=%d errors=%d elapsed=%.2fs",
            stats["files_indexed"],
            stats["files_skipped"],
            stats["symbols_extracted"],
            stats["chunks_embedded"],
            stats["chunks_reconciled"],
            stats["file_summaries_generated"],
            stats["file_summaries_generated"] + stats["file_summaries_failed"],
            stats["file_summaries_embedded"],
            stats["errors"],
            stats["elapsed_seconds"],
        )
        self._record_provenance()
        self._console.print(f"\n[green]Done.[/green] {stats}")
        return stats

    # ──────────────────────────────────────────────────────────────────────
    # Phase 1: parallel parse
    # ──────────────────────────────────────────────────────────────────────

    def _parse_all(self, files: list[IndexedFile], stats: dict[str, int]) -> list[_ParsedFile]:
        """Submit all files to a thread pool; collect _ParsedFile results."""
        results: list[_ParsedFile] = []
        total = len(files)
        done_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self._console,
        ) as progress:
            task = progress.add_task("Parsing…", total=len(files))

            with ThreadPoolExecutor(max_workers=self.config.parse_workers) as pool:
                future_to_file = {pool.submit(self._parse_one, f): f for f in files}
                for future in as_completed(future_to_file):
                    progress.advance(task)
                    done_count += 1
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        orig = future_to_file[future]
                        logger.error("Parse error %s: %s", orig.rel_path, exc)
                        # escape() both values, and note the asymmetry with the
                        # logger.error above: %s-style args go to the logging
                        # framework, which never interprets markup, so that line was
                        # always safe. This Console has markup ON, so an unmatched
                        # "[/…]" in EITHER value raised MarkupError from inside the
                        # error handler — turning one skipped file into an aborted
                        # index run that discarded every other file's work, while
                        # hiding the real cause. Neither value is tame: a directory
                        # named "deep[" gives rel_path a literal "[/", and a parser
                        # exception routinely quotes the offending source, so a
                        # bracket in `exc` needs no hostile filename at all.
                        self._console.print(
                            f"[red]Parse error[/red] {_safe_text(orig.rel_path)}: "
                            f"{_safe_text(str(exc))}"
                        )
                        stats["errors"] += 1
                    self._report_progress(1, "Parsing files…", done_count / total, stats)

        return results

    def _parse_one(self, file: IndexedFile) -> _ParsedFile:
        """
        Parse a single file (worker thread).  No DB access here — all DB
        interaction happens in Phase 2 on the main thread.
        """
        parser = get_parser(file.language)
        if parser is None:
            return _ParsedFile(file=file, parse_result=None, skipped=True)

        source = Path(file.path).read_text(encoding="utf-8", errors="replace")
        # file_id=0 is a placeholder; the real DB id is set in _insert_one (Phase 2)
        parse_result = parser.parse(source, file_id=0)

        # A file whose extractor found nothing is unreachable, not merely sparse: chunks
        # hang off symbol_id, so no symbols means no chunks and every retrieval leg is
        # blind to it. On this repository that was 12 non-empty files totalling 17 KB —
        # Go-templated helm manifests the YAML extractor cannot parse, .mjs build configs,
        # and a few test files.
        #
        # Falling back here rather than in _insert_one because _ParsedFile carries no
        # source text: the serial main thread would have to re-read the file from disk,
        # while this worker already has `source` in hand.
        if not parse_result.symbols and source.strip():
            from trelix.indexing.parser.extractors.line_window import LineWindowParser

            logger.debug(
                "No symbols extracted from %s (%s) — falling back to line windows",
                file.rel_path,
                file.language,
            )
            parse_result = LineWindowParser().parse(source, file_id=0)

        return _ParsedFile(file=file, parse_result=parse_result)

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2: sequential DB write + chunk
    # ──────────────────────────────────────────────────────────────────────

    def _insert_and_chunk_all(
        self, parsed: list[_ParsedFile], stats: dict[str, int]
    ) -> tuple[list[_PendingChunk], list[_PendingSummary]]:
        """Insert symbols + chunks for every parsed file; collect embed + summary queues.

        Returns the Phase 2.5 summary requests instead of servicing them inline. This
        loop is single-threaded by design (parent_id remapping depends on it), and the
        summary call it used to make is a network round trip: measured at 428 summaries
        spanning 1034.2 s of an 1153.7 s run, 89.6% of the whole index, with nothing
        else logging in the 2.386 s median gaps between files.
        """
        all_pending: list[_PendingChunk] = []
        all_summaries: list[_PendingSummary] = []
        total = len(parsed)
        done_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self._console,
        ) as progress:
            task = progress.add_task("Writing symbols…", total=len(parsed))

            for pf in parsed:
                progress.advance(task)
                done_count += 1
                if pf.skipped or pf.parse_result is None:
                    stats["files_skipped"] += 1
                    self._report_progress(
                        2, "Building symbols & chunks…", done_count / total, stats
                    )
                    continue
                try:
                    pending, summary_request = self._insert_one(pf, stats)
                    all_pending.extend(pending)
                    if summary_request is not None:
                        all_summaries.append(summary_request)
                except Exception as exc:
                    logger.error("DB error %s: %s", pf.file.rel_path, exc)
                    # Same sink, same reasoning as the Phase 1 handler above.
                    self._console.print(
                        f"[red]DB error[/red] {_safe_text(pf.file.rel_path)}: "
                        f"{_safe_text(str(exc))}"
                    )
                    stats["errors"] += 1
                self._report_progress(2, "Building symbols & chunks…", done_count / total, stats)

        return all_pending, all_summaries

    def _insert_one(
        self, pf: _ParsedFile, stats: dict[str, int]
    ) -> tuple[list[_PendingChunk], _PendingSummary | None]:
        """
        Insert file + symbols + chunks for one parsed file.
        Returns (_PendingChunk list, Phase 2.5 summary request or None) — the chunk_ids
        are known and their embeddings still missing; the summary has not been requested
        yet, so the LLM round trip can be fanned out by _summarize_files().

        Symbols whose qualified_name + content_hash exactly match what's
        already stored are left untouched (no delete, no re-insert, no
        re-embed) — only new/changed symbols flow through the rest of this
        method's insert + chunk + embed pipeline. Symbols removed from the
        file since the last index are deleted.
        """
        file = pf.file
        parse_result = pf.parse_result
        assert parse_result is not None  # guaranteed by _insert_and_chunk_all's None check

        # Upsert file record → get real file_id
        file_id = self.db.upsert_file(file)
        file.id = file_id

        # Fix file_id on all symbols + import edges (was placeholder 0 from parallel parse)
        all_symbols = parse_result.symbols
        for symbol in all_symbols:
            symbol.file_id = file_id
        for edge in parse_result.import_edges:
            edge.file_id = file_id

        # ── Diff newly-parsed symbols against what's already stored ──────
        # Unchanged symbols (same qualified_name + content hash) skip
        # delete+re-insert+re-embed entirely — only new/changed symbols are
        # cleaned up and rebuilt below. Symbols that existed before but are
        # no longer present in the new parse (removed functions/classes)
        # are deleted too. `changed_local_indices` holds the LOCAL indices
        # (0-based, into `all_symbols`) of every symbol that needs
        # (re)inserting — this local-index space is also what
        # parent_id / caller_id / from_symbol_id are expressed in during
        # parsing (see module docstring), so it doubles as the filter for
        # call_edges / type_edges below.
        existing_hashes = self.db.get_symbol_hashes_for_file(file_id)
        existing_ids_by_qn: dict[str, int] = {
            row[0]: row[1]
            for row in self.db._conn.execute(
                "SELECT qualified_name, id FROM symbols WHERE file_id = ?", (file_id,)
            ).fetchall()
        }
        changed_local_indices: set[int] = set()
        unchanged_qualified_names: set[str] = set()
        for local_idx, symbol in enumerate(all_symbols):
            new_hash = hashlib.sha256((symbol.signature + symbol.body).encode("utf-8")).hexdigest()
            if existing_hashes.get(symbol.qualified_name) == new_hash:
                unchanged_qualified_names.add(symbol.qualified_name)
            else:
                changed_local_indices.add(local_idx)

        # Every previously-stored symbol EXCEPT the unchanged ones must be
        # deleted — this covers both "content changed" and "removed" cases.
        qualified_names_to_delete = [
            qn for qn in existing_hashes if qn not in unchanged_qualified_names
        ]

        # symbols.parent_id / calls.callee_id / type_edges.to_symbol_id are
        # all ON DELETE SET NULL — deleting a changed/removed symbol's old
        # row below silently NULLs these on any OTHER row that pointed at
        # it, including unchanged rows this pass never touches. Snapshot
        # who's pointing at what BEFORE the delete fires, so it can be
        # re-pointed at the symbol's new row (or correctly left NULL if the
        # symbol was actually removed, not just changed) once new ids are
        # known below.
        old_id_to_qn: dict[int, str] = {
            existing_ids_by_qn[qn]: qn
            for qn in qualified_names_to_delete
            if qn in existing_ids_by_qn
        }
        stale_parent_links = self.db.get_children_with_stale_parent(list(old_id_to_qn))
        stale_callee_links = self.db.get_calls_referencing_symbols(list(old_id_to_qn))
        stale_type_links = self.db.get_type_edges_referencing_symbols(list(old_id_to_qn))

        if qualified_names_to_delete:
            old_chunk_ids = self.db.get_chunk_ids_for_symbols(file_id, qualified_names_to_delete)
            if old_chunk_ids:
                self.vector_store.delete_batch(old_chunk_ids)
            # vector_store passed explicitly: sub_chunks has no FK to symbols, and a
            # cascade could not reach a vector store anyway. Omitting it would delete
            # the rows while orphaning their vectors permanently — the row id is the
            # only handle on its vector.
            self.db.delete_symbols_by_qualified_names(
                file_id, qualified_names_to_delete, vector_store=self.vector_store
            )

        # Import edges are file-scoped (not per-symbol), so they are always
        # fully replaced on re-index — same as the pre-existing behavior of
        # delete_file_symbols(), which unconditionally cleared imports.
        self.db._conn.execute("DELETE FROM imports WHERE file_id = ?", (file_id,))
        self.db._conn.commit()

        if not all_symbols:
            stats["files_indexed"] += 1
            return [], None

        # ── Insert changed/new symbols with parent_id remapping ──────────
        # Unchanged symbols are NOT re-inserted — they keep their existing
        # DB row (and hence chunk_id/embedding) untouched. Their existing
        # DB id is looked up so parent_id / caller_id / from_symbol_id
        # references FROM changed symbols TO an unchanged symbol still
        # resolve correctly.
        local_to_db: dict[int, int] = {}
        changed_or_new_symbols: list[Symbol] = []
        with self.db.transaction():
            for local_idx, symbol in enumerate(all_symbols):
                if local_idx not in changed_local_indices:
                    existing_id = existing_ids_by_qn[symbol.qualified_name]
                    symbol.id = existing_id
                    local_to_db[local_idx] = existing_id
                    continue
                if symbol.parent_id is not None:
                    symbol.parent_id = local_to_db.get(symbol.parent_id)
                db_id = self.db.insert_symbol(symbol)
                symbol.id = db_id
                local_to_db[local_idx] = db_id
                changed_or_new_symbols.append(symbol)

            if parse_result.import_edges:
                self.db.insert_imports(parse_result.import_edges)

        # Repair FK links captured before the delete above nulled them.
        # A symbol whose qualified_name is in old_id_to_qn but was
        # content-changed (not removed) has just been re-inserted with a
        # new id in changed_or_new_symbols — repoint stale references at
        # that new id. If the qualified_name has no match there, the
        # symbol was genuinely removed and the NULL from the cascade is
        # correct as-is.
        if old_id_to_qn:
            new_id_by_qn: dict[str, int] = {
                sym.qualified_name: sym.id for sym in changed_or_new_symbols if sym.id is not None
            }

            parent_repairs = {
                child_id: new_id_by_qn[old_id_to_qn[old_parent_id]]
                for child_id, old_parent_id in stale_parent_links
                if old_id_to_qn.get(old_parent_id) in new_id_by_qn
            }
            self.db.repoint_parent_ids(parent_repairs)

            callee_repairs = {
                call_id: new_id_by_qn[old_id_to_qn[old_callee_id]]
                for call_id, old_callee_id in stale_callee_links
                if old_id_to_qn.get(old_callee_id) in new_id_by_qn
            }
            self.db.repoint_call_callee_ids(callee_repairs)

            type_edge_repairs = {
                edge_id: new_id_by_qn[old_id_to_qn[old_target_id]]
                for edge_id, old_target_id in stale_type_links
                if old_id_to_qn.get(old_target_id) in new_id_by_qn
            }
            self.db.repoint_type_edge_targets(type_edge_repairs)

        # Resolve + store call edges — only for changed/new callers; edges
        # from unchanged callers are already correctly stored from a prior
        # pass and must not be duplicated.
        if parse_result.call_edges:
            new_call_edges = [
                e for e in parse_result.call_edges if e.caller_id in changed_local_indices
            ]
            if new_call_edges:
                self._store_call_edges(new_call_edges, local_to_db)

        # Remap + store type edges — same changed-only filtering as call edges.
        if parse_result.type_edges:
            new_type_edges = [
                e for e in parse_result.type_edges if e.from_symbol_id in changed_local_indices
            ]
            if new_type_edges:
                self._store_type_edges(new_type_edges, local_to_db)

        # ── Data-flow extraction (def-use chains) ─────────────────────
        # Optional, zero cost when disabled. Runs after symbols are committed.
        # Only for changed/new symbols — unchanged symbols' def_use_edges
        # from a prior pass remain valid and must not be duplicated.
        if self.config.parser.dataflow_enabled:
            try:
                from trelix.analysis.defuse import DataFlowExtractor

                extractor = DataFlowExtractor()
                for sym in changed_or_new_symbols:
                    if sym.id is not None:
                        edges = extractor.extract(sym)
                        if edges:
                            self.db.insert_def_use_edges(edges)
            except Exception as exc:
                logger.debug(
                    "DataFlowExtractor failed for %s (non-fatal): %s", pf.file.rel_path, exc
                )

        if not changed_or_new_symbols:
            # Every symbol in the file was unchanged — nothing new to chunk
            # or embed, but the file's total symbol count is still reported.
            stats["files_indexed"] += 1
            stats["symbols_extracted"] += len(all_symbols)
            return [], None

        # ── Chunk ────────────────────────────────────────────────────────
        # Only changed/new symbols get new chunks; unchanged symbols keep
        # their existing chunk rows untouched. parent_symbols spans ALL
        # symbols (including unchanged ones) so a changed child's chunk
        # header can still reference an unchanged parent class.
        imports = self.db.get_imports_for_file(file_id)
        parent_map = {s.id: s for s in all_symbols if s.id is not None}
        chunks = self.chunker.build_chunks(
            symbols=changed_or_new_symbols,
            imports=imports,
            file_rel_path=file.rel_path,
            language=file.language.value,
            parent_symbols=parent_map,
        )

        stats["files_indexed"] += 1
        stats["symbols_extracted"] += len(all_symbols)

        # Persist context_summary back to DB if ContextualChunker populated it
        symbols_with_summary = [s for s in changed_or_new_symbols if s.context_summary and s.id]
        if symbols_with_summary:
            with self.db.transaction():
                for sym in symbols_with_summary:
                    self.db._conn.execute(
                        "UPDATE symbols SET context_summary = ? WHERE id = ?",
                        (sym.context_summary, sym.id),
                    )

        if not chunks:
            return [], None

        # Insert chunks into DB to get chunk_ids; embedding deferred to Phase 3
        pending: list[_PendingChunk] = []
        with self.db.transaction():
            for chunk in chunks:
                chunk_id = self.db.insert_chunk(chunk)
                pending.append(
                    _PendingChunk(
                        chunk_id=chunk_id,
                        chunk_text=chunk.chunk_text,
                        token_count=chunk.token_count,
                    )
                )

        # ── Phase 2.5 request (serviced later, concurrently) ─────────────────
        # The LLM round trip used to happen right here, inside the strictly serial
        # Phase 2 loop, and it dominated the run: 428 summaries over 1034.2 s of an
        # 1153.7 s index. Deferred to _summarize_files() so the calls can overlap.
        #
        # The DB write and the summary embed stay OFF the worker threads there for a
        # reason that outlives this change: `Database.upsert_file_summary` writes the
        # single shared sqlite3.Connection without taking `_conn_lock`, and db.py's own
        # comment records that the connection "is not safe for concurrent statement
        # execution from multiple threads even with check_same_thread=False".
        summary_request = (
            _PendingSummary(
                file_id=file_id,
                rel_path=file.rel_path,
                symbols=all_symbols,
                language=file.language,
            )
            if self._file_summarizer is not None
            else None
        )

        # ── Phase 2.6: multi-granularity sub-chunk extraction (MGS3) ──────────
        # Runs only when multi_granularity_enabled=True. Failures are non-fatal —
        # a crash inside MultiGranularityChunker returns [] and does not abort indexing.
        if self.config.chunker.multi_granularity_enabled:
            try:
                from trelix.indexing.multi_granularity import (
                    Granularity,
                    MultiGranularityChunker,
                )

                mg_chunker = MultiGranularityChunker()
                levels = [Granularity(lvl) for lvl in self.config.chunker.multi_granularity_levels]
                for sym in changed_or_new_symbols:
                    if sym.id is None:
                        continue
                    sub_chunks = mg_chunker.extract_sub_chunks(sym, granularities=levels)
                    if not sub_chunks:
                        continue
                    ids = self.db.insert_sub_chunks(sub_chunks)
                    texts = [sc.chunk_text for sc in sub_chunks]
                    # One un-throttled embed call per symbol used to go straight out,
                    # spending the same Azure TPM quota Phase 3 rations batch by batch —
                    # so with multi_granularity_enabled the indexer could 429 itself.
                    self._side_embed_limiter.acquire(
                        sum(self.chunker.count_tokens(t) for t in texts)
                    )
                    embeddings = self.embedder.embed(texts)
                    for sc_id, emb in zip(ids, embeddings):
                        if emb:
                            self.vector_store.upsert_sub_chunk_embedding(sc_id, emb)
            except Exception as exc:
                logger.debug("Multi-granularity indexing failed (non-fatal): %s", exc)

        return pending, summary_request

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2.5: concurrent file summaries
    # ──────────────────────────────────────────────────────────────────────

    def _summarize_files(self, work: list[_PendingSummary], stats: dict[str, Any]) -> None:
        """
        Generate, store and embed file-level summaries for `work`.

        Shape: the chat calls fan out across `self._summary_workers` threads, gated by a
        `_RpmRateLimiter`; the DB writes and the summary embeds happen on this thread.
        Threads rather than asyncio because all five chat backends expose a sync
        `complete()` wrapped in `@with_retry` — there is no async path to await.

        Three counters land in `stats`, and they are the point of this method as much as
        the concurrency is:

            file_summaries_generated  — the LLM returned text
            file_summaries_failed     — it did not (429, bad credentials, blank reply)
            file_summaries_embedded   — the text also reached the vector index

        Before them, `if summary:` had no `else` and no counter, so 0 of 467 summaries
        read exactly like 467 of 467. The live index was at 442/467 and nobody knew.
        `generated` without `embedded` matters too: a `file_summaries` row with no vector
        is invisible to the retriever's 4th leg, which is the only consumer.
        """
        if not work:
            return

        if self._file_summarizer is None:  # pragma: no cover — callers gate on this already
            return
        # _build_file_summarizer is annotated `object | None` (it returns None on any
        # build failure), so the concrete type has to be reasserted here.
        summarizer: FileSummarizer = self._file_summarizer  # type: ignore[assignment]

        limiter = _RpmRateLimiter(self._summary_rpm, console=self._console)
        total = len(work)
        generated: list[tuple[_PendingSummary, str]] = []

        self._console.print(
            f"[dim]  Phase 2.5: summarizing {total} files "
            f"({self._summary_workers} workers, "
            f"{'unlimited' if self._summary_rpm <= 0 else f'≤{self._summary_rpm}'} req/min)…[/dim]"
        )
        self._report_progress(2, "Summarizing files…", 0.0, stats)

        def request(item: _PendingSummary) -> str:
            limiter.acquire()
            return summarizer.summarize(
                rel_path=item.rel_path,
                symbols=item.symbols,
                language=item.language,
            )

        done = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self._console,
        ) as progress:
            task = progress.add_task("Summarizing…", total=total)
            with ThreadPoolExecutor(
                max_workers=self._summary_workers, thread_name_prefix="trelix-summary"
            ) as pool:
                future_to_item = {pool.submit(request, item): item for item in work}
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    progress.advance(task)
                    done += 1
                    try:
                        summary = future.result()
                    except Exception as exc:
                        # summarize() swallows LLM errors itself, so reaching here means
                        # something structural (a bad summarizer, a MemoryError). Still
                        # per-file and still counted — never fatal to the run.
                        summary = ""
                        # No _safe_text: %s-style args go to the logging framework, which
                        # never interprets markup — escaping here would mangle the path.
                        # The Console prints below carry no untrusted values at all.
                        logger.warning("File summary raised for %s: %s", item.rel_path, exc)
                    if summary:
                        generated.append((item, summary))
                        stats["file_summaries_generated"] = (
                            stats.get("file_summaries_generated", 0) + 1
                        )
                    else:
                        stats["file_summaries_failed"] = stats.get("file_summaries_failed", 0) + 1
                        # WARNING and named: the reason (if there is one) was logged by
                        # FileSummarizer; this line is what makes the missing file
                        # visible at the indexer level without a log-level change.
                        logger.warning(
                            "No file summary for %s — it will have no file-level retrieval entry",
                            item.rel_path,
                        )
                    self._report_progress(2, "Summarizing files…", done / total, stats)

        self._store_summaries(generated, stats)

        failed = stats.get("file_summaries_failed", 0)
        embedded = stats.get("file_summaries_embedded", 0)
        self._console.print(
            f"[dim]  Phase 2.5: {len(generated)}/{total} summaries generated, "
            f"{embedded} embedded[/dim]"
        )
        if failed:
            self._console.print(
                f"[yellow]  {failed} file summary(ies) failed and are missing from the "
                f"file-level retrieval leg[/yellow]"
            )
        if generated and not embedded:
            self._console.print(
                "[yellow]  file_summaries rows were written but none were embedded — "
                "the file-level retrieval leg is inert for this run[/yellow]"
            )
        if limiter.waits:
            logger.info(
                "Phase 2.5: waited on the %d req/min chat limiter %d time(s)",
                self._summary_rpm,
                limiter.waits,
            )

    def _store_summaries(
        self, generated: list[tuple[_PendingSummary, str]], stats: dict[str, Any]
    ) -> None:
        """Write summary rows, then embed them in token-aware batches under the TPM limiter.

        Main thread only — see `_PendingSummary` and db.py:292 for why.

        Batched rather than one call per file: the old code issued
        `self.embedder.embed([summary])` per file, which for this repo is 428 separate
        embed requests that also bypassed the limiter Phase 3 honours.

        A failing embed batch costs the batch's vectors, not the run and not the rows —
        the caller's `file_summaries_embedded` counter is what reports the difference.
        """
        if not generated:
            return

        for item, summary in generated:
            try:
                self.db.upsert_file_summary(item.file_id, summary)
            except Exception as exc:
                logger.warning("Storing the file summary for %s failed: %s", item.rel_path, exc)

        max_tokens = self.config.embedder.embed_max_tokens_per_batch
        # (item, summary, token_count) — counted once here rather than again per batch.
        sized = [(item, summary, self.chunker.count_tokens(summary)) for item, summary in generated]
        batch: list[tuple[_PendingSummary, str, int]] = []
        batch_tokens = 0

        def flush() -> None:
            nonlocal batch, batch_tokens
            if not batch:
                return
            current, batch, batch_tokens = batch, [], 0
            try:
                self._side_embed_limiter.acquire(sum(t for _, _, t in current))
                embeddings = self.embedder.embed([s for _, s, _ in current])
                for (entry, _, _), embedding in zip(current, embeddings):
                    self.vector_store.upsert_file_summary_embedding(entry.file_id, embedding)
                    stats["file_summaries_embedded"] = stats.get("file_summaries_embedded", 0) + 1
            except Exception as exc:
                # Contained here, not raised: by now every chunk row and every file hash
                # in this batch is committed, so an escaping error would leave those
                # files' chunks with no vectors AND make every later run skip them as up
                # to date. A summary is optional; the files' embeddings are not.
                logger.warning(
                    "Embedding %d file summary(ies) failed — those files keep their "
                    "summary row but stay out of the file-level retrieval leg: %s",
                    len(current),
                    exc,
                )

        for entry in sized:
            if batch and batch_tokens + entry[2] > max_tokens:
                flush()
            batch.append(entry)
            batch_tokens += entry[2]
        flush()

    # ──────────────────────────────────────────────────────────────────────
    # Phase 3: token-aware batch embed + store
    # ──────────────────────────────────────────────────────────────────────

    def _partial_index_error(
        self,
        *,
        failures: list[tuple[int, BaseException]],
        skipped_batches: int,
        total_batches: int,
        chunks_landed: int,
        chunks_total: int,
    ) -> PartialIndexError:
        """Build the abort both Phase 3 paths raise, with every fact needed to act on it.

        Written as one self-contained block because of where it lands: `cli/main.py`
        prints `str(exc)` and exits 1, and `index_file()` returns it as
        `{"status": "error", "error": str(exc)}`. There is no second, more detailed
        channel.

        Three claims in the text are load-bearing and all three are properties of this
        file:

          * "PARTIAL right now" — `_insert_one` commits `upsert_file` (which writes
            files.hash) and the chunk rows in Phase 2, BEFORE this phase, and nothing
            here rolls either back. Re-parsing alone would not find the damage: the
            incremental pre-filter in `index()` skips the file on a matching hash, and
            with `incremental=False` the file is re-parsed but `_insert_one` only chunks
            `changed_or_new_symbols`, i.e. symbols whose signature+body hash differs.
          * "run `trelix index` again … re-embeds exactly the chunks left without a
            vector" — `_chunks_missing_vectors` diffs the store against the `chunks`
            table once per run and both pipelines fold the result into this same Phase 3.
            The text names `trelix index` because that is the reliable route. `trelix
            watch` also reconciles, but only in its startup pass — `cli/main.py`'s watch
            command calls `indexer.index()` before it starts watching, and that call
            reaches the reconcile above like any other. What never reconciles is the
            watching phase itself: `FileWatcher` calls `index_file()` per changed file,
            which has no repo-wide diff, so a hole opened while watch is already running
            survives until the next startup. Saying "watch never reconciles" was wrong
            about the startup pass and is why the docs told users it would not help.
            The text also says to fix the failure FIRST: the
            streaming pipeline repairs after its walk and skips the repair entirely while
            any file is still failing, so a re-run that fails the same way repairs
            nothing. It replaced advice to DELETE the index database, which this branch
            made both unnecessary and expensive — that throws away every embedding this
            run did land (61,652 of them on the index this was reproduced against) to
            recover holes the reconcile refills for the price of the holes alone.
          * "counted as indexed" — `trelix stats` reads `SELECT COUNT(*) FROM chunks`;
            the chunk rows exist. It now also reports coverage against the vector store,
            which is what makes "keeps counting them as indexed" a pointer rather than a
            dead end.
        """
        failed_batches = len(failures)
        written_batches = total_batches - failed_batches - skipped_batches
        first_index, first_exc = failures[0]

        backend = str(getattr(self.config.store, "backend", "unknown"))
        locations = [f"the index database ({self.config.db_path_absolute})"]
        if backend == "lance":
            locations.append(f"the LanceDB directory ({self.config.store.lance_uri})")
        elif backend == "qdrant":
            locations.append(
                f"the Qdrant collection {self.config.store.qdrant_collection} "
                f"at {self.config.store.qdrant_url}"
            )

        # Only claimed when it happened: with a handful of batches the first failure
        # can surface after the last one has already been sent.
        skip_note = (
            f"The {skipped_batches} skipped batch(es) were not embedded at all: a store "
            f"that refuses one write is rarely done refusing, and every batch is a paid "
            f"embedding call.\n"
            if skipped_batches
            else ""
        )

        return PartialIndexError(
            f"Indexing aborted — Phase 3 could not embed and store every batch "
            f"(vector store backend: {backend}).\n"
            f"  batches: {failed_batches} failed, {skipped_batches} batch(es) were "
            f"skipped un-embedded, {written_batches} written, {total_batches} total\n"
            f"  chunks:  {chunks_landed} of {chunks_total} chunk(s) from this run have "
            f"vectors\n"
            f"  first failure (batch {first_index + 1}): "
            f"{type(first_exc).__name__}: {first_exc}\n"
            f"{skip_note}"
            f"This index is PARTIAL right now — the chunk rows and the files' content "
            f"hashes were committed before this phase, so the chunks this run failed to "
            f"embed stay unsearchable while `trelix stats` keeps counting them as "
            f"indexed.\n"
            f"To recover, fix the failure above and run `trelix index` again: it "
            f"reconciles the vector store against the `chunks` table once per run and "
            f"re-embeds exactly the chunks left without a vector, so nothing that already "
            f"landed is paid for twice. `trelix watch` will not do it — only `trelix "
            f"index` reconciles. The partial state is in {' and '.join(locations)}."
        )

    def _batch_embed_and_store(self, pending: list[_PendingChunk], stats: dict[str, int]) -> None:
        """
        Embed all pending chunks in token-aware batches, then write vectors.

        Batching strategy:
          - Group chunks so that each batch's total token count ≤
            embed_max_tokens_per_batch (prevents API request-size errors).
          - _TpmRateLimiter sleeps before a batch if sending it would push
            the rolling 60-second token total above tpm_limit.

        A batch that fails to embed or to store stops the loop and raises
        `PartialIndexError` — see `_partial_index_error`. This is the path
        `index_file()` takes, and its blanket `except` turns the abort into
        `{"status": "error", "error": <the message>}` plus one ERROR log line for
        that one file, so watch mode and the streaming pipeline keep going with the
        rest of the repo rather than dying on one file.
        """
        cfg = self.config.embedder
        limiter = _TpmRateLimiter(cfg.tpm_limit, console=self._console)
        batches = _make_token_batches(pending, cfg.embed_max_tokens_per_batch)
        total_chunks = len(pending)
        embedded_so_far = 0
        failures: list[tuple[int, BaseException]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self._console,
        ) as progress:
            task = progress.add_task("Embedding…", total=len(pending))

            for batch_index, batch in enumerate(batches):
                batch_tokens = sum(p.token_count for p in batch)
                limiter.acquire(batch_tokens)  # may sleep to respect TPM limit

                try:
                    embeddings = self.embedder.embed([p.chunk_text for p in batch])
                    self.vector_store.upsert_batch(
                        [(p.chunk_id, emb) for p, emb in zip(batch, embeddings)]
                    )
                except Exception as exc:
                    failures.append((batch_index, exc))
                    break
                stats["chunks_embedded"] += len(batch)
                embedded_so_far += len(batch)
                progress.advance(task, advance=len(batch))
                self._report_progress(
                    3,
                    "Embedding chunks…",
                    embedded_so_far / total_chunks if total_chunks else 1.0,
                    stats,
                )

        if failures:
            stats["errors"] = stats.get("errors", 0) + len(failures)
            raise self._partial_index_error(
                failures=failures,
                skipped_batches=len(batches) - failures[0][0] - 1,
                total_batches=len(batches),
                chunks_landed=embedded_so_far,
                chunks_total=total_chunks,
            )

    async def _batch_embed_and_store_async(
        self, pending: list[_PendingChunk], stats: dict[str, int]
    ) -> None:
        """
        Async Phase 3: embed all pending chunks with up to 4 concurrent API calls.

        Concurrency model:
          - asyncio.Semaphore(4) caps simultaneous embed_async() calls at 4.
          - asyncio.gather() fans out all batches at once; the semaphore ensures
            at most 4 are in-flight to the embedding API at any given time.
          - _AsyncTpmRateLimiter uses asyncio.sleep (non-blocking) to honour
            the rolling TPM ceiling.
          - vector_store.upsert_batch() is sync → run in a thread executor so
            it does not block the event loop.

        Progress tracking uses a lock-protected shared counter so concurrent
        coroutines can safely increment stats["chunks_embedded"].

        Failure handling, which the concurrency makes non-obvious:
          - Each batch catches its own embed/store exception instead of letting it
            escape the gather. A bare `gather` (no return_exceptions) re-raises the
            first exception while its siblings keep running unwatched: measured with
            8 batches and a store that always raises, all 8 upserts still ran, the
            caller saw only the store's own one-line message, and
            `upsert_executor.shutdown()` — which used to sit after this block — never
            ran, leaking both trelix-upsert threads.
          - The first failure sets `abort`, and batches that have not started
            embedding yet return without calling the API. A store rejecting one
            write is rarely done rejecting, and each skipped batch is a paid
            embedding call not spent. Batches already past `embed_async` still
            attempt their upsert: the embedding is bought either way, and landing it
            is strictly better.
          - The abort itself is raised after the executor is shut down, as a
            `PartialIndexError` naming the damage and the recovery.
        """
        cfg = self.config.embedder
        limiter = _AsyncTpmRateLimiter(cfg.tpm_limit, console=self._console)
        semaphore = asyncio.Semaphore(4)
        batches = _make_token_batches(pending, cfg.embed_max_tokens_per_batch)
        total_chunks = len(pending)

        # Thread executor for the sync upsert_batch call
        loop = asyncio.get_event_loop()
        upsert_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="trelix-upsert")

        # Shared mutable counter guarded by a lock
        counter_lock = asyncio.Lock()
        embedded_so_far = 0
        failures: list[tuple[int, BaseException]] = []
        skipped_batches = 0
        abort = False

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self._console,
        ) as progress:
            task = progress.add_task("Embedding…", total=total_chunks)

            async def embed_one_batch(batch_index: int, batch: list[_PendingChunk]) -> None:
                nonlocal embedded_so_far, skipped_batches, abort
                if abort:
                    skipped_batches += 1
                    return
                batch_tokens = sum(p.token_count for p in batch)
                # Respect TPM before acquiring semaphore to avoid holding it
                # during a potentially long sleep.
                await limiter.acquire(batch_tokens)
                try:
                    async with semaphore:
                        # Re-checked here, not only on entry: with 4 permits a batch can
                        # wait out an earlier batch's whole embed+store before it gets
                        # this far, and the point of the flag is to not pay for it.
                        if abort:
                            skipped_batches += 1
                            return
                        embeddings = await self.embedder.embed_async([p.chunk_text for p in batch])
                    # upsert_batch is sync — run in executor to not block event loop
                    pairs = [(p.chunk_id, emb) for p, emb in zip(batch, embeddings)]
                    await loop.run_in_executor(
                        upsert_executor, self.vector_store.upsert_batch, pairs
                    )
                except Exception as exc:
                    abort = True
                    failures.append((batch_index, exc))
                    return
                # Update shared counters safely
                async with counter_lock:
                    embedded_so_far += len(batch)
                    stats["chunks_embedded"] += len(batch)
                    progress.advance(task, advance=len(batch))
                    self._report_progress(
                        3,
                        "Embedding chunks…",
                        embedded_so_far / total_chunks if total_chunks else 1.0,
                        stats,
                    )

            try:
                await asyncio.gather(*[embed_one_batch(i, b) for i, b in enumerate(batches)])
            finally:
                # In the finally so a KeyboardInterrupt or a CancelledError mid-gather
                # cannot leak the pool either.
                upsert_executor.shutdown(wait=True)

        if failures:
            stats["errors"] = stats.get("errors", 0) + len(failures)
            # Logged as well as raised: a library caller embedding Indexer can swallow
            # the exception, and this line is what remains in the log if it does.
            logger.error(
                "Phase 3 aborted: %d of %d embedding batch(es) failed to embed or store, "
                "%d skipped; %d of %d chunk(s) have vectors. First failure: %s",
                len(failures),
                len(batches),
                skipped_batches,
                embedded_so_far,
                total_chunks,
                failures[0][1],
            )
            raise self._partial_index_error(
                failures=failures,
                skipped_batches=skipped_batches,
                total_batches=len(batches),
                chunks_landed=embedded_so_far,
                chunks_total=total_chunks,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_symbol_match(
        matches: list[Symbol],
        name: str,
        type_hint: str | None = None,
    ) -> int | None:
        """
        Pick the correct symbol id out of `matches` (all rows returned by
        `Database.get_symbol_by_name(name)`) using the same priority order as
        `Database.resolve_cross_file_calls()`'s SQL cascade — kept in sync
        deliberately, since this runs at insert time (every index() and
        index_file() call) while that method only runs for batches >=
        `_FULL_RESOLVE_THRESHOLD` files. Taking `matches[0]` unconditionally
        (the old behavior) silently picked whichever symbol SQLite happened
        to return first whenever two symbols shared a bare name — e.g. two
        classes each defining a same-named method — wiring every call site
        to the wrong one. "Leave unresolved" is correct here, not a fallback:
        a wrong edge is worse than a missing edge (see docs/architecture.md
        Key Design Invariants #5).

        Every priority below requires the match to be UNIQUE at that
        priority level, not merely present — for a top-level class/function,
        qualified_name equals the bare name, so two unrelated top-level
        symbols sharing that name (e.g. two files each defining `class
        Base`) both satisfy "qualified_name exact match" without a
        uniqueness check; that would silently reintroduce the same
        wrong-edge risk this method exists to close. Note this makes
        priorities 1-2 here slightly stricter than `resolve_cross_file_calls`'s
        SQL (`... LIMIT 1`, no uniqueness check on those two passes) — a
        pre-existing, narrower gap in the batch-mode cascade that's out of
        scope for this fix.

        1. Exact qualified_name match against `name`, if exactly one match has it.
        2. If `type_hint` is set, a match whose qualified_name starts with
           "<type_hint>.", if exactly one match has it.
        3. The single match, only if `matches` has exactly one entry overall.
        4. Otherwise None — ambiguous, leave unresolved for the batch-mode
           cascade (or forever, in watch mode — still correct).
        """
        if not matches:
            return None

        qualified = [m for m in matches if m.qualified_name == name]
        if len(qualified) == 1:
            return qualified[0].id

        if type_hint:
            prefix = f"{type_hint}."
            hinted = [m for m in matches if m.qualified_name.startswith(prefix)]
            if len(hinted) == 1:
                return hinted[0].id

        if len(matches) == 1:
            return matches[0].id
        return None

    def _store_call_edges(
        self,
        edges: list[Any],
        local_to_db: dict[int, int],
    ) -> None:
        """
        Remap caller local_idx → DB id and resolve callee_name → callee DB id.
        Unresolved callees (external libs, stdlib, or ambiguous bare-name
        matches) get callee_id = None — fine, see _resolve_symbol_match().
        """
        for edge in edges:
            db_caller_id = local_to_db.get(edge.caller_id)
            if db_caller_id is None:
                continue
            edge.caller_id = db_caller_id
            matches = self.db.get_symbol_by_name(edge.callee_name)
            edge.callee_id = self._resolve_symbol_match(
                matches, edge.callee_name, edge.callee_type_hint
            )

        valid = [e for e in edges if e.caller_id in local_to_db.values()]
        if valid:
            with self.db.transaction():
                self.db.insert_call_edges(valid)

    def _store_type_edges(
        self,
        edges: list[Any],
        local_to_db: dict[int, int],
    ) -> None:
        """
        Remap from_symbol_id local_idx → DB id and best-effort resolve to_symbol_id.
        Unresolvable types (external libs, or ambiguous bare-name matches)
        remain with to_symbol_id = None — see _resolve_symbol_match().
        """
        from trelix.core.models import TypeEdge

        valid: list[TypeEdge] = []
        for edge in edges:
            db_from_id = local_to_db.get(edge.from_symbol_id)
            if db_from_id is None:
                continue
            edge.from_symbol_id = db_from_id
            # Best-effort intra-file resolution
            matches = self.db.get_symbol_by_name(edge.to_type_name)
            edge.to_symbol_id = self._resolve_symbol_match(matches, edge.to_type_name)
            valid.append(edge)

        if valid:
            with self.db.transaction():
                self.db.insert_type_edges(valid)

    # ──────────────────────────────────────────────────────────────────────
    # Single-file update (called by `trelix update-index`)
    # ──────────────────────────────────────────────────────────────────────

    def index_file(
        self,
        file_path: str,
        *,
        files_in_batch: int = 1,
    ) -> dict[str, Any]:
        """
        Re-index a single file.  Faster than a full `--incremental` run because
        it skips the repo walk entirely.

        Args:
            file_path: absolute path to the file, or path relative to repo root.
            files_in_batch: total number of files being updated in this watch
                event batch.  When the batch size is below
                ``_FULL_RESOLVE_THRESHOLD`` the four O(N) global resolve passes
                are skipped — they are already correct from the last full index()
                run and the benefit of re-running them for a single changed file
                is marginal compared to the cost.  Callers processing a large
                burst (e.g. a branch checkout) should pass the actual count so
                the resolve still fires when it matters.

        Returns:
            {"status": "ok", "symbols_updated": N, "chunks_updated": N, "ms": N}
            {"status": "error", "error": "<message>"}
        """

        t0 = time.perf_counter()

        try:
            abs_path = Path(file_path)
            if not abs_path.is_absolute():
                abs_path = Path(self.config.repo_path) / abs_path
            abs_path = abs_path.resolve()

            repo_root = Path(self.config.repo_path).resolve()
            rel_path = str(abs_path.relative_to(repo_root))

            language = detect_language(abs_path)
            file_hash = hashlib.sha256(abs_path.read_bytes()).hexdigest()
            size_bytes = abs_path.stat().st_size

            # Skip if file content hasn't changed (same as incremental logic in index())
            if self.db.get_file_hash(rel_path) == file_hash:
                logger.debug("index_file: no change detected for %s — skipping", rel_path)
                return {
                    "status": "ok",
                    "symbols_updated": 0,
                    "chunks_updated": 0,
                    "ms": round((time.perf_counter() - t0) * 1000),
                    "skipped": True,
                }

            file = IndexedFile(
                path=str(abs_path),
                rel_path=rel_path,
                language=language,
                hash=file_hash,
                size_bytes=size_bytes,
            )

            pf = self._parse_one(file)

            inner_stats: dict[str, Any] = {
                "files_indexed": 0,
                "symbols_extracted": 0,
                "chunks_total": 0,
                "chunks_embedded": 0,
                "file_summaries_generated": 0,
                "file_summaries_failed": 0,
                "file_summaries_embedded": 0,
                "errors": 0,
            }

            if pf.skipped or pf.parse_result is None:
                # Language not supported — clear stale data and return
                existing_file_id = self._get_file_id(rel_path)
                if existing_file_id is not None:
                    old_chunk_ids = self.db.get_chunk_ids_for_file(existing_file_id)
                    if old_chunk_ids:
                        self.vector_store.delete_batch(old_chunk_ids)
                    self.db.delete_file_symbols(existing_file_id, vector_store=self.vector_store)
                return {
                    "status": "ok",
                    "symbols_updated": 0,
                    "chunks_updated": 0,
                    "ms": round((time.perf_counter() - t0) * 1000),
                }

            pending, summary_request = self._insert_one(pf, inner_stats)

            # Same phase order as index(), with a one-item work list: the fan-out
            # degenerates to a single call rather than diverging into a second code path.
            if summary_request is not None:
                self._summarize_files([summary_request], inner_stats)

            if pending:
                self._batch_embed_and_store(pending, inner_stats)

            # Cross-file resolution: all four passes are O(total_calls + total_imports)
            # regardless of how many files changed, so skipping them for small watch
            # events is a significant win.  The resolve state from the last full
            # index() run remains valid — a single changed file rarely adds edges
            # that were previously unresolvable across the whole codebase.
            # We only pay the full cost when the batch is large enough that new
            # symbols are likely to unlock previously unresolved edges.
            if files_in_batch >= self._FULL_RESOLVE_THRESHOLD:
                self.db.resolve_cross_file_calls()
                self.db.resolve_import_file_ids()
                self.db.resolve_cross_file_type_edges()
                self.db.resolve_angular_selectors()
                logger.debug(
                    "index_file: ran full cross-file resolve (batch=%d >= threshold=%d)",
                    files_in_batch,
                    self._FULL_RESOLVE_THRESHOLD,
                )
            else:
                logger.debug(
                    "index_file: skipped cross-file resolve (batch=%d < threshold=%d); "
                    "next full index() will reconcile any new edges",
                    files_in_batch,
                    self._FULL_RESOLVE_THRESHOLD,
                )

            return {
                "status": "ok",
                "symbols_updated": inner_stats["symbols_extracted"],
                "chunks_updated": inner_stats["chunks_embedded"],
                "ms": round((time.perf_counter() - t0) * 1000),
            }

        except Exception as exc:
            logger.error("index_file failed for %s: %s", file_path, exc)
            return {"status": "error", "error": str(exc)}

    def _get_file_id(self, rel_path: str) -> int | None:
        """Return the DB id for a file by rel_path, or None if not indexed."""
        row = self.db._conn.execute(
            "SELECT id FROM files WHERE rel_path = ?", (rel_path,)
        ).fetchone()
        return row[0] if row else None


# ---------------------------------------------------------------------------
# Utility: token-aware batch builder
# ---------------------------------------------------------------------------


def _make_token_batches(
    chunks: list[_PendingChunk],
    max_tokens_per_batch: int,
) -> list[list[_PendingChunk]]:
    """
    Greedily group chunks into batches where the sum of token_count per batch
    does not exceed max_tokens_per_batch.

    A single chunk that exceeds the limit on its own is placed in its own
    batch (the Chunker already caps individual chunks via max_tokens_per_chunk,
    so this only happens if max_tokens_per_batch is misconfigured very small).
    """
    batches: list[list[_PendingChunk]] = []
    current: list[_PendingChunk] = []
    current_tokens = 0

    for chunk in chunks:
        t = chunk.token_count
        if current and current_tokens + t > max_tokens_per_batch:
            batches.append(current)
            current = [chunk]
            current_tokens = t
        else:
            current.append(chunk)
            current_tokens += t

    if current:
        batches.append(current)

    return batches
