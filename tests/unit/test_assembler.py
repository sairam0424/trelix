"""Unit tests for ContextAssembler (Phase 11)."""

from __future__ import annotations

from datetime import datetime

import pytest

from trelix.core.models import (
    Chunk,
    IndexedFile,
    Language,
    RetrievedContext,
    SearchResult,
    Symbol,
    SymbolKind,
)
from trelix.retrieval.assembler import ContextAssembler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file(file_id: int, rel_path: str) -> IndexedFile:
    return IndexedFile(
        path=f"/repo/{rel_path}",
        rel_path=rel_path,
        language=Language.PYTHON,
        hash=f"sha-{file_id}",
        size_bytes=1000,
        id=file_id,
        indexed_at=datetime(2024, 1, 1),
    )


def _make_symbol(
    sym_id: int,
    file_id: int,
    name: str,
    qualified_name: str,
    line_start: int,
    line_end: int,
    kind: SymbolKind = SymbolKind.FUNCTION,
) -> Symbol:
    body_lines = [f"def {name}():", "    pass"]
    return Symbol(
        file_id=file_id,
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        line_start=line_start,
        line_end=line_end,
        signature=f"def {name}()",
        body="\n".join(body_lines),
        id=sym_id,
    )


def _make_chunk(chunk_id: int, sym_id: int, text: str, token_count: int) -> Chunk:
    return Chunk(
        symbol_id=sym_id,
        chunk_text=text,
        token_count=token_count,
        id=chunk_id,
    )


def _make_result(
    chunk: Chunk,
    symbol: Symbol,
    file: IndexedFile,
    score: float,
    rank: int,
    source: str = "vector",
) -> SearchResult:
    return SearchResult(
        chunk=chunk,
        symbol=symbol,
        file=file,
        score=score,
        rank=rank,
        source=source,
    )


# ---------------------------------------------------------------------------
# Fixture: 10 mock SearchResult objects from 3 different files
#
# File A (file_id=1, rel_path="src/auth/login.py")     — 4 results
# File B (file_id=2, rel_path="src/auth/models.py")    — 3 results
# File C (file_id=3, rel_path="src/utils/helpers.py")  — 3 results
# ---------------------------------------------------------------------------

FILE_A = _make_file(1, "src/auth/login.py")
FILE_B = _make_file(2, "src/auth/models.py")
FILE_C = _make_file(3, "src/utils/helpers.py")

# File A symbols
SYM_A1 = _make_symbol(1, 1, "authenticate_user", "LoginView.authenticate_user", 10, 30)
SYM_A2 = _make_symbol(2, 1, "logout_user",       "LoginView.logout_user",       32, 45)
SYM_A3 = _make_symbol(3, 1, "check_token",       "LoginView.check_token",       47, 60)
SYM_A4 = _make_symbol(4, 1, "refresh_session",   "LoginView.refresh_session",   62, 80)

# File B symbols
SYM_B1 = _make_symbol(5, 2, "User",         "User",         5,  40, SymbolKind.CLASS)
SYM_B2 = _make_symbol(6, 2, "Session",      "Session",      42, 70, SymbolKind.CLASS)
SYM_B3 = _make_symbol(7, 2, "get_user",     "get_user",     72, 85)

# File C symbols
SYM_C1 = _make_symbol(8,  3, "hash_password", "hash_password",  1, 15)
SYM_C2 = _make_symbol(9,  3, "verify_token",  "verify_token",  17, 30)
SYM_C3 = _make_symbol(10, 3, "encode_jwt",    "encode_jwt",    32, 50)

# Chunks (token_count chosen so we can control budget effects)
CHUNK_A1 = _make_chunk(1,  1, "# File: src/auth/login.py\ndef authenticate_user(): pass",  100)
CHUNK_A2 = _make_chunk(2,  2, "# File: src/auth/login.py\ndef logout_user(): pass",          80)
CHUNK_A3 = _make_chunk(3,  3, "# File: src/auth/login.py\ndef check_token(): pass",          90)
CHUNK_A4 = _make_chunk(4,  4, "# File: src/auth/login.py\ndef refresh_session(): pass",     110)
CHUNK_B1 = _make_chunk(5,  5, "# File: src/auth/models.py\nclass User: pass",               200)
CHUNK_B2 = _make_chunk(6,  6, "# File: src/auth/models.py\nclass Session: pass",            150)
CHUNK_B3 = _make_chunk(7,  7, "# File: src/auth/models.py\ndef get_user(): pass",            70)
CHUNK_C1 = _make_chunk(8,  8, "# File: src/utils/helpers.py\ndef hash_password(): pass",    60)
CHUNK_C2 = _make_chunk(9,  9, "# File: src/utils/helpers.py\ndef verify_token(): pass",     65)
CHUNK_C3 = _make_chunk(10, 10, "# File: src/utils/helpers.py\ndef encode_jwt(): pass",      55)

# Results — scores spread intentionally so we can assert ordering
ALL_RESULTS: list[SearchResult] = [
    _make_result(CHUNK_A1, SYM_A1, FILE_A, score=0.95, rank=1),   # highest score
    _make_result(CHUNK_B1, SYM_B1, FILE_B, score=0.88, rank=2),
    _make_result(CHUNK_C1, SYM_C1, FILE_C, score=0.82, rank=3),
    _make_result(CHUNK_A2, SYM_A2, FILE_A, score=0.78, rank=4),
    _make_result(CHUNK_B2, SYM_B2, FILE_B, score=0.74, rank=5),
    _make_result(CHUNK_C2, SYM_C2, FILE_C, score=0.70, rank=6),
    _make_result(CHUNK_A3, SYM_A3, FILE_A, score=0.65, rank=7),
    _make_result(CHUNK_B3, SYM_B3, FILE_B, score=0.60, rank=8),
    _make_result(CHUNK_C3, SYM_C3, FILE_C, score=0.55, rank=9),
    _make_result(CHUNK_A4, SYM_A4, FILE_A, score=0.50, rank=10),  # lowest score
]

QUERY = "how does user authentication work?"


# ---------------------------------------------------------------------------
# Tests: greedy mode
# ---------------------------------------------------------------------------

class TestGreedyMode:
    def test_greedy_results_ordered_by_score_descending(self) -> None:
        """Greedy mode must preserve the input score ordering (highest first)."""
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="greedy")

        scores = [r.score for r in ctx.results]
        assert scores == sorted(scores, reverse=True), (
            "Greedy results must be ordered by score descending"
        )

    def test_greedy_first_result_has_highest_score(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="greedy")

        assert ctx.results[0].score == 0.95

    def test_greedy_last_result_has_lowest_included_score(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="greedy")

        # All 10 results fit inside 10_000 tokens (total = 980), so last = 0.50
        assert ctx.results[-1].score == 0.50

    def test_greedy_returns_retrieved_context(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="greedy")

        assert isinstance(ctx, RetrievedContext)

    def test_greedy_default_mode_is_greedy(self) -> None:
        """Calling assemble without assembly_mode must default to greedy."""
        assembler = ContextAssembler(token_budget=10_000)
        ctx_default = assembler.assemble(QUERY, ALL_RESULTS)
        ctx_greedy  = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="greedy")

        assert [r.score for r in ctx_default.results] == [r.score for r in ctx_greedy.results]


# ---------------------------------------------------------------------------
# Tests: breadth_first mode
# ---------------------------------------------------------------------------

class TestBreadthFirstMode:
    def test_breadth_first_multiple_files_represented(self) -> None:
        """Breadth-first must surface results from more than one file."""
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="breadth_first")

        file_paths = {r.file.rel_path for r in ctx.results}
        assert len(file_paths) >= 2, (
            "breadth_first mode must include results from at least 2 different files"
        )

    def test_breadth_first_all_three_files_represented(self) -> None:
        """With a large budget, all 3 files must appear."""
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="breadth_first")

        file_paths = {r.file.rel_path for r in ctx.results}
        assert "src/auth/login.py"     in file_paths
        assert "src/auth/models.py"    in file_paths
        assert "src/utils/helpers.py"  in file_paths

    def test_breadth_first_max_two_per_file(self) -> None:
        """Breadth-first must not include more than 2 results from the same file."""
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="breadth_first")

        from collections import Counter
        counts = Counter(r.file.rel_path for r in ctx.results)
        for path, count in counts.items():
            assert count <= 2, (
                f"breadth_first mode must not include >2 results per file, "
                f"but {path!r} has {count}"
            )

    def test_breadth_first_higher_scored_files_come_first(self) -> None:
        """Files with higher best-scores must appear before lower-scored files."""
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="breadth_first")

        # First result must come from FILE_A (best score 0.95)
        assert ctx.results[0].file.rel_path == "src/auth/login.py"


# ---------------------------------------------------------------------------
# Tests: token budget
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_total_tokens_within_budget_greedy(self) -> None:
        budget = 300
        assembler = ContextAssembler(token_budget=budget)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="greedy")

        assert ctx.total_tokens <= budget, (
            f"total_tokens {ctx.total_tokens} exceeds budget {budget}"
        )

    def test_total_tokens_within_budget_breadth_first(self) -> None:
        budget = 300
        assembler = ContextAssembler(token_budget=budget)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="breadth_first")

        assert ctx.total_tokens <= budget, (
            f"total_tokens {ctx.total_tokens} exceeds budget {budget}"
        )

    def test_greedy_stops_adding_when_budget_exceeded(self) -> None:
        """With a very tight budget (100 tokens) only the first result (100 tokens) fits."""
        budget = 100
        assembler = ContextAssembler(token_budget=budget)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="greedy")

        # CHUNK_A1 = 100 tokens exactly; CHUNK_A1 + CHUNK_A2 = 180 > 100
        assert len(ctx.results) == 1
        assert ctx.results[0].score == 0.95
        assert ctx.total_tokens == 100

    def test_empty_results_returns_zero_tokens(self) -> None:
        assembler = ContextAssembler(token_budget=8_000)
        ctx = assembler.assemble(QUERY, [], assembly_mode="greedy")

        assert ctx.total_tokens == 0

    def test_total_tokens_equals_sum_of_chunk_token_counts(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="greedy")

        expected = sum(r.chunk.token_count for r in ctx.results)
        assert ctx.total_tokens == expected


# ---------------------------------------------------------------------------
# Tests: context_text content
# ---------------------------------------------------------------------------

class TestContextText:
    def test_context_text_contains_file_path(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="greedy")

        assert "src/auth/login.py" in ctx.context_text

    def test_context_text_contains_symbol_qualified_name(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="greedy")

        # The top result is LoginView.authenticate_user
        assert "LoginView.authenticate_user" in ctx.context_text

    def test_context_text_file_section_header_format(self) -> None:
        """File sections must use the === path === format."""
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="greedy")

        assert "=== src/auth/login.py ===" in ctx.context_text

    def test_context_text_contains_lines_range(self) -> None:
        """Each symbol block must include [Lines start-end] header."""
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="greedy")

        # SYM_A1 starts at line 10 and ends at line 30
        assert "[Lines 10-30]" in ctx.context_text

    def test_empty_results_context_text(self) -> None:
        assembler = ContextAssembler(token_budget=8_000)
        ctx = assembler.assemble(QUERY, [], assembly_mode="greedy")

        assert ctx.context_text == "No relevant code found."

    def test_context_text_with_symbol_lookup_intent_preamble(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(
            QUERY, ALL_RESULTS, intent="symbol_lookup", assembly_mode="greedy"
        )

        assert "# Symbol:" in ctx.context_text
        assert "LoginView.authenticate_user" in ctx.context_text

    def test_context_text_with_project_overview_intent_preamble(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(
            QUERY, ALL_RESULTS, intent="project_overview", assembly_mode="greedy"
        )

        assert "# Project Architecture Overview" in ctx.context_text

    def test_context_text_with_comparison_intent_preamble(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(
            QUERY, ALL_RESULTS, intent="comparison", assembly_mode="greedy"
        )

        assert "# Comparison" in ctx.context_text

    def test_no_preamble_without_intent(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, assembly_mode="greedy")

        # Without intent there should be no "# " preamble line
        assert not ctx.context_text.startswith("#")


# ---------------------------------------------------------------------------
# Tests: RetrievedContext fields
# ---------------------------------------------------------------------------

class TestRetrievedContextFields:
    def test_query_preserved_in_context(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS)

        assert ctx.query == QUERY

    def test_intent_preserved_in_context(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, intent="symbol_lookup")

        assert ctx.intent == "symbol_lookup"

    def test_empty_intent_stored_as_empty_string(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS, intent=None)

        assert ctx.intent == ""

    def test_retrieval_sources_counted(self) -> None:
        assembler = ContextAssembler(token_budget=10_000)
        ctx = assembler.assemble(QUERY, ALL_RESULTS)

        # All 10 results use source="vector"
        assert "vector" in ctx.retrieval_sources
        assert ctx.retrieval_sources["vector"] >= 1
