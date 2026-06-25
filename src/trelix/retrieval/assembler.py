"""
Context Assembler: packs reranked results into a token-budget-aware context string.

Key insight stolen from Aider's repo-map:
- Use a greedy algorithm: add highest-ranked results until token budget is full
- Format matters: include file path + line range so LLM can cite sources
- Group chunks from the same file together (reads more naturally)
- Always include the query's best match first (most relevant at the top)
"""

from __future__ import annotations

from collections import defaultdict

import tiktoken

from trelix.core.models import RetrievedContext, SearchResult


class ContextAssembler:
    """
    Assembles a list of SearchResults into a formatted context string
    that fits within a token budget.

    Usage:
        assembler = ContextAssembler(token_budget=8000)
        context = assembler.assemble(query="...", results=[...])
    """

    def __init__(self, token_budget: int = 8_000) -> None:
        self.token_budget = token_budget
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def assemble(
        self,
        query: str,
        results: list[SearchResult],
        intent: str | None = None,
        assembly_mode: str = "greedy",
    ) -> RetrievedContext:
        """
        Pack results into context within the token budget.

        assembly_mode="greedy"       — take results in score order until budget full.
                                       Best for focused queries (symbol_lookup, feature_flow).
        assembly_mode="breadth_first"— limit to 2 symbols per file, prioritise covering
                                       many files. Best for dependency_map / blast_radius
                                       where breadth matters more than depth.

        `intent` adds a structured preamble so the LLM understands the answer shape.
        """
        if not results:
            return RetrievedContext(
                query=query,
                results=[],
                context_text="No relevant code found.",
                total_tokens=0,
            )

        if assembly_mode == "breadth_first":
            selected = self._pack_breadth_first(results)
        else:
            selected = self._pack_greedy(results)

        source_counts: dict[str, int] = defaultdict(int)
        tokens_used = 0
        for r in selected:
            source_counts[r.source] += 1
            tokens_used += r.chunk.token_count

        context_text = self._format_context(selected, intent=intent)

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

    def _pack_breadth_first(
        self,
        results: list[SearchResult],
        max_per_file: int = 2,
    ) -> list[SearchResult]:
        """
        Prefer breadth (many files) over depth (many symbols per file).

        Groups results by file, orders files by their best symbol score,
        then takes up to max_per_file symbols from each file within the budget.
        Ensures dependency_map and blast_radius queries surface at least one
        representative symbol from every relevant file rather than exhausting
        the token budget on a single file.
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

        selected: list[SearchResult] = []
        tokens_used = 0
        for _file_path, file_results in sorted_files:
            top_for_file = sorted(file_results, key=lambda r: r.score, reverse=True)[:max_per_file]
            for result in top_for_file:
                if tokens_used + result.chunk.token_count <= self.token_budget:
                    selected.append(result)
                    tokens_used += result.chunk.token_count
        return selected

    def _format_context(self, results: list[SearchResult], intent: str | None = None) -> str:
        """
        Format results into a clean, LLM-readable context block.

        Output format (base):
            === src/auth/login.py ===

            [Lines 42-67] LoginView.authenticate_user
            def authenticate_user(...):
                ...

        With intent preamble prepended for structured query types.
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
