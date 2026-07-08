"""
Indexer: orchestrates the full indexing pipeline.

Three-phase design for large-repo performance:

  Phase 1 — parallel parse
    N threads read + parse files concurrently (file I/O + tree-sitter both
    release the GIL, so threading gives real speedup).

  Phase 2 — sequential DB write + chunk
    Symbols and chunks are inserted in the main thread to keep parent_id
    remapping consistent (local parse indices → real DB row ids).

  Phase 3 — async concurrent batch embed  (U5)
    Up to 4 API calls run concurrently via asyncio.gather + Semaphore(4).
    _make_token_batches() groups chunks so each batch stays under
    embed_max_tokens_per_batch tokens (prevents request-size errors).
    _AsyncTpmRateLimiter uses asyncio.sleep (non-blocking) to stay within
    the configured Azure TPM ceiling — we never exceed the quota.
    vector_store.upsert_batch() is sync → called in a thread executor.

  Phase 4 — cross-file resolution
    Call-edge targets and import file_ids are resolved after every file
    has been inserted (same second-pass as before).

parent_id / caller_id convention:
  During parsing, parent_id and caller_id are LOCAL INDICES (0-based) into
  the per-file symbol list.  Phase 2 remaps them to real DB ids.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from trelix.core.config import IndexConfig
from trelix.core.models import IndexedFile
from trelix.embedder.base import BaseEmbedder, make_embedder
from trelix.indexing.chunker import Chunker, ContextualChunker
from trelix.indexing.parser.base import ParseResult
from trelix.indexing.parser.registry import get_parser
from trelix.indexing.walker import FileWalker
from trelix.store.db import Database
from trelix.store.vector import BaseVectorStore, make_vector_store

logger = logging.getLogger("trelix.indexing")


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

    def _iter_files(self, repo_path: str):
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
            "files_indexed": 0,
            "files_skipped": 0,
            "symbols_extracted": 0,
            "chunks_total": 0,
            "chunks_embedded": 0,
            "errors": 0,
            "elapsed_seconds": 0.0,
        }
        t_start = time.perf_counter()

        file_queue: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        sentinel = object()

        def producer() -> None:
            """Walk the repo and put each IndexedFile onto the queue."""
            for indexed_file in self._iter_files(repo_path):
                file_queue.put(indexed_file)
            file_queue.put(sentinel)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        while True:
            item = file_queue.get()
            if item is sentinel:
                break

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

        # Cross-file resolution — single pass after all files are processed
        try:
            self.db.resolve_cross_file_calls()
            self.db.resolve_import_file_ids()
            self.db.resolve_cross_file_type_edges()
            self.db.resolve_angular_selectors()
        except Exception as exc:
            logger.debug("Streaming index: resolution pass failed (non-fatal): %s", exc)

        results["elapsed_seconds"] = round(time.perf_counter() - t_start, 2)
        return results

    def index(self) -> dict[str, Any]:
        # Route to streaming pipeline when enabled
        if getattr(self.config, "indexer", None) and self.config.indexer.streaming_enabled:
            return self._index_streaming(self.config.repo_path)

        t_start = time.perf_counter()
        stats: dict[str, Any] = {
            "files_found": 0,
            "files_indexed": 0,
            "files_skipped": 0,
            "symbols_extracted": 0,
            "chunks_total": 0,
            "chunks_embedded": 0,
            "errors": 0,
            "elapsed_seconds": 0.0,
        }

        logger.info("Starting indexing: repo=%s", self.config.repo_path)
        self._report_progress(0, "Discovering files…", 0.0, stats)
        files = list(self.walker.walk())
        stats["files_found"] = len(files)
        self._report_progress(0, "Discovering files…", 1.0, stats)

        # Pre-filter: skip files whose hash hasn't changed (sequential, read-only DB)
        if self.config.incremental:
            to_parse = [f for f in files if self.db.get_file_hash(f.rel_path) != f.hash]
            stats["files_skipped"] = len(files) - len(to_parse)
        else:
            to_parse = files

        if not to_parse:
            self._console.print("[green]Nothing to index — all files up to date.[/green]")
            return stats

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
        pending = self._insert_and_chunk_all(parsed, stats)

        # ── Phase 3: async concurrent batch embed ───────────────────────────
        stats["chunks_total"] = len(pending)
        if pending:
            total_tokens = sum(p.token_count for p in pending)
            self._console.print(
                f"[dim]  Phase 3/3: embedding {len(pending)} chunks "
                f"({total_tokens:,} tokens, up to 4 concurrent API calls)…[/dim]"
            )
            self._report_progress(3, "Embedding chunks…", 0.0, stats)
            asyncio.run(self._batch_embed_and_store_async(pending, stats))

        # Record dimension after successful embedding phase
        if pending:
            try:
                from trelix.store.dimension_guard import DimensionGuard

                DimensionGuard.record(
                    self.db,
                    dimension=self.embedder.dimension,
                    provider=self.config.embedder.provider,
                )
            except Exception as exc:
                logger.debug("DimensionGuard.record failed (non-fatal): %s", exc)

        # ── Sparse embedding phase (SPLADE-Code) — runs when sparse_enabled=True ──
        if self.config.retrieval.sparse_enabled and pending:
            try:
                from trelix.embedder.sparse import SparseEmbedder
                from trelix.store.sparse_store import SparseStore

                sparse_emb = SparseEmbedder(
                    model_name=self.config.sparse.model,
                    top_k=self.config.sparse.top_k_tokens,
                )
                sparse_store = SparseStore(self.config.db_path_absolute)
                texts = [pc.chunk_text for pc in pending]
                sparse_vecs = sparse_emb.embed(texts)
                pairs = [(int(pc.chunk_id), vec) for pc, vec in zip(pending, sparse_vecs) if vec]
                if pairs:
                    sparse_store.upsert_batch(pairs)
                    logger.info("Sparse embedding: indexed %d chunks", len(pairs))
            except Exception as exc:
                logger.debug("Sparse embedding phase failed (non-fatal): %s", exc)

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
            "chunks=%d errors=%d elapsed=%.2fs",
            stats["files_indexed"],
            stats["files_skipped"],
            stats["symbols_extracted"],
            stats["chunks_embedded"],
            stats["errors"],
            stats["elapsed_seconds"],
        )
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
                        self._console.print(f"[red]Parse error[/red] {orig.rel_path}: {exc}")
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
        return _ParsedFile(file=file, parse_result=parse_result)

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2: sequential DB write + chunk
    # ──────────────────────────────────────────────────────────────────────

    def _insert_and_chunk_all(
        self, parsed: list[_ParsedFile], stats: dict[str, int]
    ) -> list[_PendingChunk]:
        """Insert symbols + chunks for every parsed file; collect embed queue."""
        all_pending: list[_PendingChunk] = []
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
                    pending = self._insert_one(pf, stats)
                    all_pending.extend(pending)
                except Exception as exc:
                    logger.error("DB error %s: %s", pf.file.rel_path, exc)
                    self._console.print(f"[red]DB error[/red] {pf.file.rel_path}: {exc}")
                    stats["errors"] += 1
                self._report_progress(2, "Building symbols & chunks…", done_count / total, stats)

        return all_pending

    def _insert_one(self, pf: _ParsedFile, stats: dict[str, int]) -> list[_PendingChunk]:
        """
        Insert file + symbols + chunks for one parsed file.
        Returns _PendingChunk list (chunk_id known, embedding still missing).
        """
        file = pf.file
        parse_result = pf.parse_result
        assert parse_result is not None  # guaranteed by _insert_and_chunk_all's None check

        # Upsert file record → get real file_id
        file_id = self.db.upsert_file(file)
        file.id = file_id

        # Fix file_id on all symbols + import edges (was placeholder 0 from parallel parse)
        for symbol in parse_result.symbols:
            symbol.file_id = file_id
        for edge in parse_result.import_edges:
            edge.file_id = file_id

        # Clean stale vectors + symbols before re-indexing
        old_chunk_ids = self.db.get_chunk_ids_for_file(file_id)
        if old_chunk_ids:
            self.vector_store.delete_batch(old_chunk_ids)
        self.db.delete_file_symbols(file_id)

        if not parse_result.symbols:
            stats["files_indexed"] += 1
            return []

        # ── Insert symbols with parent_id remapping ──────────────────────
        local_to_db: dict[int, int] = {}
        with self.db.transaction():
            for local_idx, symbol in enumerate(parse_result.symbols):
                if symbol.parent_id is not None:
                    symbol.parent_id = local_to_db.get(symbol.parent_id)
                db_id = self.db.insert_symbol(symbol)
                symbol.id = db_id
                local_to_db[local_idx] = db_id

            if parse_result.import_edges:
                self.db.insert_imports(parse_result.import_edges)

        # Resolve + store call edges
        if parse_result.call_edges:
            self._store_call_edges(parse_result.call_edges, local_to_db)

        # Remap + store type edges
        if parse_result.type_edges:
            self._store_type_edges(parse_result.type_edges, local_to_db)

        # ── Data-flow extraction (def-use chains) ─────────────────────
        # Optional, zero cost when disabled. Runs after symbols are committed.
        if self.config.parser.dataflow_enabled:
            try:
                from trelix.analysis.defuse import DataFlowExtractor

                extractor = DataFlowExtractor()
                for sym in parse_result.symbols:
                    if sym.id is not None:
                        edges = extractor.extract(sym)
                        if edges:
                            self.db.insert_def_use_edges(edges)
            except Exception as exc:
                logger.debug(
                    "DataFlowExtractor failed for %s (non-fatal): %s", pf.file.rel_path, exc
                )

        # ── Chunk ────────────────────────────────────────────────────────
        imports = self.db.get_imports_for_file(file_id)
        parent_map = {s.id: s for s in parse_result.symbols if s.id is not None}
        chunks = self.chunker.build_chunks(
            symbols=parse_result.symbols,
            imports=imports,
            file_rel_path=file.rel_path,
            language=file.language.value,
            parent_symbols=parent_map,
        )

        stats["files_indexed"] += 1
        stats["symbols_extracted"] += len(parse_result.symbols)

        # Persist context_summary back to DB if ContextualChunker populated it
        symbols_with_summary = [s for s in parse_result.symbols if s.context_summary and s.id]
        if symbols_with_summary:
            with self.db.transaction():
                for sym in symbols_with_summary:
                    self.db._conn.execute(
                        "UPDATE symbols SET context_summary = ? WHERE id = ?",
                        (sym.context_summary, sym.id),
                    )

        if not chunks:
            return []

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

        # ── Phase 2.5: generate file-level summary (RAPTOR-style) ────────────
        # Runs only when file_summaries_enabled=True and an LLM client is
        # available.  Failures are swallowed inside FileSummarizer.summarize()
        # so a single flaky LLM call never aborts the whole indexing pass.
        if self._file_summarizer is not None:
            from trelix.indexing.file_summarizer import FileSummarizer

            summarizer: FileSummarizer = self._file_summarizer  # type: ignore[assignment]
            summary = summarizer.summarize(
                rel_path=file.rel_path,
                symbols=parse_result.symbols,
                language=file.language,
            )
            if summary:
                summary_row_id = self.db.upsert_file_summary(file_id, summary)
                # Embed the summary and store in the vector index so the
                # retriever can surface file-level context via the 4th leg.
                summary_embedding = self.embedder.embed([summary])[0]
                self.vector_store.upsert_file_summary_embedding(file_id, summary_embedding)
                logger.debug(
                    "File summary stored for %s (row_id=%d, len=%d chars)",
                    file.rel_path,
                    summary_row_id,
                    len(summary),
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
                for sym in parse_result.symbols:
                    if sym.id is None:
                        continue
                    sub_chunks = mg_chunker.extract_sub_chunks(sym, granularities=levels)
                    if not sub_chunks:
                        continue
                    ids = self.db.insert_sub_chunks(sub_chunks)
                    texts = [sc.chunk_text for sc in sub_chunks]
                    embeddings = self.embedder.embed(texts)
                    for sc_id, emb in zip(ids, embeddings):
                        if emb:
                            self.vector_store.upsert_sub_chunk_embedding(sc_id, emb)
            except Exception as exc:
                logger.debug("Multi-granularity indexing failed (non-fatal): %s", exc)

        return pending

    # ──────────────────────────────────────────────────────────────────────
    # Phase 3: token-aware batch embed + store
    # ──────────────────────────────────────────────────────────────────────

    def _batch_embed_and_store(self, pending: list[_PendingChunk], stats: dict[str, int]) -> None:
        """
        Embed all pending chunks in token-aware batches, then write vectors.

        Batching strategy:
          - Group chunks so that each batch's total token count ≤
            embed_max_tokens_per_batch (prevents API request-size errors).
          - _TpmRateLimiter sleeps before a batch if sending it would push
            the rolling 60-second token total above tpm_limit.
        """
        cfg = self.config.embedder
        limiter = _TpmRateLimiter(cfg.tpm_limit, console=self._console)
        batches = _make_token_batches(pending, cfg.embed_max_tokens_per_batch)
        total_chunks = len(pending)
        embedded_so_far = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self._console,
        ) as progress:
            task = progress.add_task("Embedding…", total=len(pending))

            for batch in batches:
                batch_tokens = sum(p.token_count for p in batch)
                limiter.acquire(batch_tokens)  # may sleep to respect TPM limit

                embeddings = self.embedder.embed([p.chunk_text for p in batch])
                self.vector_store.upsert_batch(
                    [(p.chunk_id, emb) for p, emb in zip(batch, embeddings)]
                )
                stats["chunks_embedded"] += len(batch)
                embedded_so_far += len(batch)
                progress.advance(task, advance=len(batch))
                self._report_progress(
                    3,
                    "Embedding chunks…",
                    embedded_so_far / total_chunks if total_chunks else 1.0,
                    stats,
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

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self._console,
        ) as progress:
            task = progress.add_task("Embedding…", total=total_chunks)

            async def embed_one_batch(batch: list[_PendingChunk]) -> None:
                nonlocal embedded_so_far
                batch_tokens = sum(p.token_count for p in batch)
                # Respect TPM before acquiring semaphore to avoid holding it
                # during a potentially long sleep.
                await limiter.acquire(batch_tokens)
                async with semaphore:
                    embeddings = await self.embedder.embed_async([p.chunk_text for p in batch])
                # upsert_batch is sync — run in executor to not block event loop
                pairs = [(p.chunk_id, emb) for p, emb in zip(batch, embeddings)]
                await loop.run_in_executor(upsert_executor, self.vector_store.upsert_batch, pairs)
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

            await asyncio.gather(*[embed_one_batch(b) for b in batches])

        upsert_executor.shutdown(wait=True)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _store_call_edges(
        self,
        edges: list[Any],
        local_to_db: dict[int, int],
    ) -> None:
        """
        Remap caller local_idx → DB id and resolve callee_name → callee DB id.
        Unresolved callees (external libs, stdlib) get callee_id = None — fine.
        """
        for edge in edges:
            db_caller_id = local_to_db.get(edge.caller_id)
            if db_caller_id is None:
                continue
            edge.caller_id = db_caller_id
            matches = self.db.get_symbol_by_name(edge.callee_name)
            if matches:
                edge.callee_id = matches[0].id

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
        Unresolvable types (external libs) remain with to_symbol_id = None.
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
            if matches:
                edge.to_symbol_id = matches[0].id
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
        import hashlib

        from trelix.core.models import Language
        from trelix.indexing.walker import EXTENSION_MAP

        t0 = time.perf_counter()

        try:
            abs_path = Path(file_path)
            if not abs_path.is_absolute():
                abs_path = Path(self.config.repo_path) / abs_path
            abs_path = abs_path.resolve()

            repo_root = Path(self.config.repo_path).resolve()
            rel_path = str(abs_path.relative_to(repo_root))

            language = EXTENSION_MAP.get(abs_path.suffix.lower(), Language.UNKNOWN)
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
                "errors": 0,
            }

            if pf.skipped or pf.parse_result is None:
                # Language not supported — clear stale data and return
                existing_file_id = self._get_file_id(rel_path)
                if existing_file_id is not None:
                    old_chunk_ids = self.db.get_chunk_ids_for_file(existing_file_id)
                    if old_chunk_ids:
                        self.vector_store.delete_batch(old_chunk_ids)
                    self.db.delete_file_symbols(existing_file_id)
                return {
                    "status": "ok",
                    "symbols_updated": 0,
                    "chunks_updated": 0,
                    "ms": round((time.perf_counter() - t0) * 1000),
                }

            pending = self._insert_one(pf, inner_stats)

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
