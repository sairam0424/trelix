"""
Context Assembler: packs reranked results into a token-budget-aware context string.

Key insight stolen from Aider's repo-map:
- Use a greedy algorithm: add highest-ranked results until token budget is full
- Format matters: include file path + line range so LLM can cite sources
- Group chunks from the same file together (reads more naturally)
- Always include the query's best match first (most relevant at the top)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import tiktoken

from trelix.core.models import RetrievedContext, SearchResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trelix.compression.base import CompressionResult, Compressor

logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Assembles a list of SearchResults into a formatted context string
    that fits within a token budget.

    Usage:
        assembler = ContextAssembler(token_budget=8000)
        context = assembler.assemble(query="...", results=[...])
    """

    def __init__(
        self,
        token_budget: int = 8_000,
        per_source_budget: bool = False,
        *,
        compressor: Compressor | None = None,
        compression_ratio: float = 1.0,
        compression_min_tokens: int = 120,
    ) -> None:
        self.token_budget = token_budget
        # Split the budget proportionally to each source leg's result count
        # (not a fixed weight table — self-tuning, no extra config surface)
        # instead of one shared pool a single noisy leg could crowd out.
        # False (default) reproduces the exact prior single-pool behavior.
        self.per_source_budget = per_source_budget
        # Compression is opt-in and additive: with compressor=None (the default)
        # or ratio >= 1.0, NOTHING below changes and the assembled context is
        # byte-identical to the pre-compression implementation.
        self.compressor = compressor
        self.compression_ratio = compression_ratio
        self.compression_min_tokens = compression_min_tokens
        #: Trace-friendly summary of the last assemble() compression pass
        #: (None when compression was inactive). Mirrors the compressor's own
        #: ``last_path`` attribute so the retriever can record both.
        self.last_compression_stats: dict[str, object] | None = None
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def _compression_active(self) -> bool:
        """True only when a compressor was supplied AND the ratio asks to shrink."""
        return self.compressor is not None and 0.0 < self.compression_ratio < 1.0

    def assemble(
        self,
        query: str,
        results: list[SearchResult],
        intent: str | None = None,
        assembly_mode: str = "greedy",
        *,
        query_embedding: list[float] | None = None,
    ) -> RetrievedContext:
        """
        Pack results into context within the token budget.

        assembly_mode="greedy"       — take results in score order until budget full.
                                       Best for focused queries (symbol_lookup, feature_flow).
        assembly_mode="breadth_first"— limit to 2 symbols per file, prioritise covering
                                       many files. Best for dependency_map / blast_radius
                                       where breadth matters more than depth.

        `intent` adds a structured preamble so the LLM understands the answer shape.

        `query_embedding` is an embedding the caller ALREADY computed. It is only
        forwarded to the compressor (which uses it to score already-stored
        sub-chunk vectors); assembly never computes one itself, so no code path
        here can trigger an embedding or network call.
        """
        self.last_compression_stats = None
        if not results:
            return RetrievedContext(
                query=query,
                results=[],
                context_text="No relevant code found.",
                total_tokens=0,
            )

        # `eligible` is the ordered candidate pool the pack considered — for
        # breadth-first that is the max_per_file-truncated list, so compression
        # inherits that cap instead of quietly reopening it.
        if assembly_mode == "breadth_first":
            eligible = self._breadth_first_candidates(results)
            selected = self._pack_greedy(eligible)
        elif self.per_source_budget:
            eligible = results
            selected = self._pack_proportional(results)
        else:
            eligible = results
            selected = self._pack_greedy(results)

        compressed: dict[int, CompressionResult] = {}
        if self._compression_active:
            selected, compressed = self._pack_compressed(query, eligible, selected, query_embedding)

        source_counts: dict[str, int] = defaultdict(int)
        tokens_used = 0
        for r in selected:
            source_counts[r.source] += 1
            tokens_used += r.chunk.token_count

        context_text = self._format_context(selected, intent=intent, compressed=compressed)

        return RetrievedContext(
            query=query,
            results=selected,
            context_text=context_text,
            total_tokens=tokens_used,
            intent=intent or "",
            retrieval_sources=dict(source_counts),
        )

    def _pack_greedy(self, results: list[SearchResult]) -> list[SearchResult]:
        """Take results in score order until the token budget is exhausted."""
        selected: list[SearchResult] = []
        tokens_used = 0
        for result in results:
            if tokens_used + result.chunk.token_count <= self.token_budget:
                selected.append(result)
                tokens_used += result.chunk.token_count
        return selected

    def _pack_proportional(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Split the token budget across source legs (vector/bm25/grep/...)
        proportionally to each leg's result count, then greedy-pack within
        each leg's slice — so one noisy leg with many low-value results
        can't crowd out a smaller, higher-precision leg the way a single
        shared budget pool would.

        `results` is assumed already score-sorted (the fused/reranked order
        every caller passes in) — that ordering is preserved both within
        each leg's slice and in the final merged output.

        Any leg that doesn't spend its full slice (because it simply has
        fewer candidates than its budget covers, not because the next
        candidate didn't fit) returns the unused tokens to a shared pool,
        which a second pass spends on the highest-scoring not-yet-selected
        results overall, regardless of leg.
        """
        total = len(results)
        source_counts: dict[str, int] = defaultdict(int)
        for r in results:
            source_counts[r.source] += 1

        budget_by_source = {
            source: round(self.token_budget * count / total)
            for source, count in source_counts.items()
        }

        selected: list[SearchResult] = []
        selected_ids: set[int] = set()
        leftover_budget = 0
        for source, sub_budget in budget_by_source.items():
            leg_results = [r for r in results if r.source == source]
            leg_tokens_used = 0
            leg_selected: list[SearchResult] = []
            for result in leg_results:
                if leg_tokens_used + result.chunk.token_count <= sub_budget:
                    leg_selected.append(result)
                    leg_tokens_used += result.chunk.token_count
            selected.extend(leg_selected)
            selected_ids.update(id(r) for r in leg_selected)
            leftover_budget += sub_budget - leg_tokens_used

        if leftover_budget > 0:
            for result in results:
                if id(result) in selected_ids:
                    continue
                if result.chunk.token_count <= leftover_budget:
                    selected.append(result)
                    selected_ids.add(id(result))
                    leftover_budget -= result.chunk.token_count

        # Preserve the original score-descending order in the merged output.
        order = {id(r): i for i, r in enumerate(results)}
        selected.sort(key=lambda r: order[id(r)])
        return selected

    def _pack_breadth_first(
        self,
        results: list[SearchResult],
        max_per_file: int = 2,
    ) -> list[SearchResult]:
        """
        Prefer breadth (many files) over depth (many symbols per file).

        Greedy-pack over the breadth-first candidate ordering — identical to the
        prior inline implementation, which was itself an accumulate-if-fits scan
        over exactly that ordering.
        """
        return self._pack_greedy(self._breadth_first_candidates(results, max_per_file))

    def _breadth_first_candidates(
        self,
        results: list[SearchResult],
        max_per_file: int = 2,
    ) -> list[SearchResult]:
        """
        The breadth-first candidate ordering, budget-independent.

        Groups results by file, orders files by their best symbol score, then
        takes up to max_per_file symbols from each file. Ensures dependency_map
        and blast_radius queries surface at least one representative symbol from
        every relevant file rather than exhausting the budget on a single file.

        Returned separately from packing so compression can reuse the SAME
        truncated pool — max_per_file stays enforced through wave 2.
        """
        # Group by file, preserve best-score ordering across files
        file_groups: dict[str, list[SearchResult]] = defaultdict(list)
        for r in results:
            file_groups[r.file.rel_path].append(r)

        # Sort files by their best symbol score (highest first)
        sorted_files = sorted(
            file_groups.items(),
            key=lambda kv: max(r.score for r in kv[1]),
            reverse=True,
        )

        candidates: list[SearchResult] = []
        for _file_path, file_results in sorted_files:
            candidates.extend(
                sorted(file_results, key=lambda r: r.score, reverse=True)[:max_per_file]
            )
        return candidates

    def _pack_compressed(
        self,
        query: str,
        eligible: list[SearchResult],
        wave1: list[SearchResult],
        query_embedding: list[float] | None,
    ) -> tuple[list[SearchResult], dict[int, CompressionResult]]:
        """
        Wave 2: re-offer the candidates wave 1 could not fit, compressed.

        Result-lossless — the output is always a superset of `wave1`, so no
        result the uncompressed pack would have kept is ever displaced. Any
        failure degrades to the uncompressed selection with a warning and a
        trace note (never raises, never silently worsens).
        """
        assert self.compressor is not None  # guarded by _compression_active
        from trelix.retrieval.context_compression import pack_compressed

        try:
            selected, compressed, stats = pack_compressed(
                query=query,
                eligible=eligible,
                wave1=wave1,
                token_budget=self.token_budget,
                compressor=self.compressor,
                target_ratio=self.compression_ratio,
                min_tokens=self.compression_min_tokens,
                query_embedding=query_embedding,
            )
        except Exception as exc:  # noqa: BLE001 — graceful degradation contract
            logger.warning("Context compression failed (%s); packing uncompressed", exc)
            self.last_compression_stats = {"error": str(exc), "wave1_kept": len(wave1)}
            return wave1, {}
        stats["ratio"] = self.compression_ratio
        stats["min_tokens"] = self.compression_min_tokens
        self.last_compression_stats = stats
        return selected, compressed

    def _format_context(
        self,
        results: list[SearchResult],
        intent: str | None = None,
        compressed: dict[int, CompressionResult] | None = None,
    ) -> str:
        """
        Format results into a clean, LLM-readable context block.

        Output format (base):
            === src/auth/login.py ===

            [Lines 42-67] LoginView.authenticate_user
            def authenticate_user(...):
                ...

        With intent preamble prepended for structured query types.

        A result in `compressed` was only partially kept, so it is rendered as
        one truthful line-range block PER kept span instead of a single header
        that would claim lines the text no longer contains.
        """
        preamble = self._make_preamble(results, intent)

        # Group by file
        by_file: dict[str, list[SearchResult]] = defaultdict(list)
        for r in results:
            by_file[r.file.rel_path].append(r)

        blocks: list[str] = []

        for file_path, file_results in by_file.items():
            blocks.append(f"=== {file_path} ===\n")
            for r in sorted(file_results, key=lambda x: x.symbol.line_start):
                cresult = compressed.get(r.chunk.symbol_id) if compressed else None
                if cresult is not None:
                    from trelix.retrieval.context_compression import format_compressed_blocks

                    blocks.append(f"{format_compressed_blocks(r, cresult)}\n")
                    continue
                header = (
                    f"[Lines {r.symbol.line_start}-{r.symbol.line_end}] {r.symbol.qualified_name}"
                )
                blocks.append(f"{header}\n{r.chunk.chunk_text}\n")

        body = "\n".join(blocks)
        return f"{preamble}\n{body}" if preamble else body

    def _make_preamble(self, results: list[SearchResult], intent: str | None) -> str:
        """
        Return an intent-specific preamble that orients the LLM before it reads code.

        file_overview    -> table of contents listing every symbol in the file
        project_overview -> "Architecture Overview" label with source list
        comparison       -> "Comparison" label
        symbol_lookup    -> names the primary symbol being examined
        others           -> empty string (no preamble needed)
        """
        if not intent or not results:
            return ""

        if intent == "file_overview":
            files = sorted({r.file.rel_path for r in results})
            lines = [f"# File Overview: {', '.join(files)}"]
            lines.append("# Contents:")
            # Show top-level symbols only (classes + functions, skip methods/constants)
            TOP_LEVEL_KINDS = {"module", "class", "interface", "function", "enum"}
            for r in results:
                if r.symbol.kind in TOP_LEVEL_KINDS:
                    lines.append(
                        f"#   {r.symbol.kind:<12} {r.symbol.name:<40} "
                        f"[lines {r.symbol.line_start}-{r.symbol.line_end}]"
                    )
            return "\n".join(lines)

        if intent == "project_overview":
            sources = sorted({r.file.rel_path for r in results})
            lines = ["# Project Architecture Overview"]
            lines.append(f"# Sources ({len(sources)} files): " + ", ".join(sources[:8]))
            if len(sources) > 8:
                lines.append(f"#   ... and {len(sources) - 8} more")
            return "\n".join(lines)

        if intent == "comparison":
            return "# Comparison"

        if intent == "symbol_lookup" and results:
            sym = results[0].symbol
            file_path = results[0].file.rel_path if sym.file_id else ""
            return f"# Symbol: {sym.qualified_name} ({sym.kind}) — {file_path}"

        return ""

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))
