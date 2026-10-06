"""Citation tags on retrieved context ([C1], [C2], ...), behind TRELIX_RETRIEVAL_CITATIONS.

First of six changes toward `trelix ask` answers that cite the code they rest on. This one
tags the assembled context and instructs the model; nothing reads the model's tags yet.
This file covers the flag, the assembler and the Retriever wiring; compressed bodies are in
test_citation_tags_compressed.py and the synthesis prompts in test_citation_prompts.py.

Two contracts, each pinned with the literal text it produces:

* FLAG OFF (the default): the assembled context is byte-identical to what it was before
  the flag existed (``test_assembler_backcompat_golden.py`` holds the frozen digests).
* FLAG ON: every block header carries ``[C<n>] `` in RENDERED order (files first-seen,
  symbols by line), one number per symbol, the intent preambles excluded;
  ``RetrievedContext.citation_sources`` lists what each tag refers to.

Every expected value is a literal written in this file (rule 1), never imported from the
module under test. Each test names the mutation that must make it fail.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures.citation_context import (
    BEARER_BLOCK,
    DECODE_BLOCK,
    QUERY,
    VERIFY_BLOCK,
    bearer_result,
    f1_results,
    sources_of,
    verify_result,
)
from trelix.core.config import RetrievalConfig
from trelix.retrieval.assembler import ContextAssembler
from trelix.retrieval.citations import cite_tag

# What the assembler rendered for F1 before this change (the design's observed value).
UNTAGGED_CONTEXT = (
    "=== src/auth/middleware.py ===\n\n" + VERIFY_BLOCK + "\n" + BEARER_BLOCK + "\n"
    "=== src/auth/jwt.py ===\n\n" + DECODE_BLOCK
)
TAGGED_CONTEXT = (
    "=== src/auth/middleware.py ===\n\n"
    + "[C1] "
    + VERIFY_BLOCK
    + "\n"
    + "[C2] "
    + BEARER_BLOCK
    + "\n"
    + "=== src/auth/jwt.py ===\n\n"
    + "[C3] "
    + DECODE_BLOCK
)
F1_SOURCES = [
    (1, 11, "src/auth/middleware.py", 42, 67, "AuthMiddleware.verify"),
    (2, 33, "src/auth/middleware.py", 70, 80, "AuthMiddleware.bearer"),
    (3, 22, "src/auth/jwt.py", 10, 30, "decode_token"),
]


# ---------------------------------------------------------------------------
# Config flag
# ---------------------------------------------------------------------------


class TestConfigFlag:
    def test_default_is_off(self) -> None:
        """MUTATION: default=True on `citations_enabled` fails this (and the digests)."""
        assert RetrievalConfig.model_fields["citations_enabled"].default is False
        assert RetrievalConfig().citations_enabled is False

    def test_env_alias_turns_it_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MUTATION: rename the alias and the env var stops being read."""
        monkeypatch.setenv("TRELIX_RETRIEVAL_CITATIONS", "true")
        assert RetrievalConfig().citations_enabled is True


# ---------------------------------------------------------------------------
# Assembler: tags on block headers
# ---------------------------------------------------------------------------


class TestAssemblerTags:
    def test_flag_off_renders_todays_context_byte_for_byte(self) -> None:
        """MUTATION: `cite_tags: bool = True`, or a prefix rendered when the flag is off."""
        for assembler in (
            ContextAssembler(token_budget=8_000),
            ContextAssembler(token_budget=8_000, cite_tags=False),
        ):
            context = assembler.assemble(QUERY, f1_results())
            assert context.context_text == UNTAGGED_CONTEXT
            assert context.citation_sources == ()
            assert context.total_tokens == 60

    def test_flag_on_prefixes_each_header_in_rendered_order(self) -> None:
        """MUTATION: number tags in `selected` (score) order and bearer becomes [C3]."""
        context = ContextAssembler(token_budget=8_000, cite_tags=True).assemble(QUERY, f1_results())

        assert context.context_text == TAGGED_CONTEXT
        assert sources_of(context) == F1_SOURCES
        assert context.total_tokens == 60  # tags never count toward the budget

    def test_tags_follow_rendered_order_not_score_order(self) -> None:
        """The premise of the test above, stated on its own: bearer scored below
        decode_token (results keep score order) yet renders first, so it is tag 2."""
        context = ContextAssembler(token_budget=8_000, cite_tags=True).assemble(QUERY, f1_results())

        assert [r.symbol.qualified_name for r in context.results] == [
            "AuthMiddleware.verify",
            "decode_token",
            "AuthMiddleware.bearer",
        ]
        assert [s.symbol for s in context.citation_sources] == [
            "AuthMiddleware.verify",
            "AuthMiddleware.bearer",
            "decode_token",
        ]

    def test_one_symbol_keeps_one_tag_across_two_results(self) -> None:
        """MUTATION: drop the symbol_id de-duplication and the second verify becomes [C2]."""
        duplicate = verify_result(score=0.6, rank=4, source="bm25")
        context = ContextAssembler(token_budget=8_000, cite_tags=True).assemble(
            QUERY, [verify_result(), duplicate, bearer_result()]
        )

        assert context.context_text == (
            "=== src/auth/middleware.py ===\n\n"
            + "[C1] "
            + VERIFY_BLOCK
            + "\n"
            + "[C1] "
            + VERIFY_BLOCK
            + "\n"
            + "[C2] "
            + BEARER_BLOCK
        )
        assert sources_of(context) == F1_SOURCES[:2]

    def test_intent_preambles_are_not_tagged(self) -> None:
        """MUTATION: tag the preamble's `[lines a-b]` entries, or the `# Symbol:` line."""
        assembler = ContextAssembler(token_budget=8_000, cite_tags=True)

        overview = assembler.assemble(QUERY, f1_results(), intent="file_overview")
        assert overview.context_text == (
            "# File Overview: src/auth/jwt.py, src/auth/middleware.py\n"
            "# Contents:\n"
            "#   function     decode_token                             [lines 10-30]\n"
            + TAGGED_CONTEXT
        )
        assert sources_of(overview) == F1_SOURCES

        lookup = assembler.assemble(QUERY, f1_results(), intent="symbol_lookup")
        assert lookup.context_text == (
            "# Symbol: AuthMiddleware.verify (method) — src/auth/middleware.py\n" + TAGGED_CONTEXT
        )

    def test_empty_results_carry_no_sources(self) -> None:
        context = ContextAssembler(token_budget=8_000, cite_tags=True).assemble(QUERY, [])

        assert context.context_text == "No relevant code found."
        assert context.citation_sources == ()

    def test_sources_are_reset_between_calls(self) -> None:
        """MUTATION: drop the per-call reset and the second call reports six sources."""
        assembler = ContextAssembler(token_budget=8_000, cite_tags=True)
        assembler.assemble(QUERY, f1_results())

        second = assembler.assemble(QUERY, f1_results())

        assert sources_of(second) == F1_SOURCES
        assert second.context_text == TAGGED_CONTEXT


# ---------------------------------------------------------------------------
# Retriever wiring: the flag reaches the assembler
# ---------------------------------------------------------------------------


class TestRetrieverWiring:
    @staticmethod
    def _stub(**retrieval_kwargs: object) -> Any:
        from trelix.retrieval import retriever as retriever_mod
        from trelix.retrieval.retriever import Retriever

        class _Cfg:
            def __init__(self) -> None:
                self.retrieval = RetrievalConfig(**retrieval_kwargs)  # type: ignore[arg-type]

        class _NoSubChunkDB:
            _conn = None

        class _Stub:
            _assemble = Retriever._assemble
            _make_compressor = Retriever._make_compressor
            _cached_query_embedding = Retriever._cached_query_embedding
            _trace = Retriever._trace

            def __init__(self) -> None:
                self.config = _Cfg()
                self.db = _NoSubChunkDB()
                self.embedder = None
                self._effective_budget = 8_000

        retriever_mod._trace_local.data = {}
        return _Stub()

    def test_flag_on_reaches_the_assembler(self) -> None:
        """MUTATION: drop `cite_tags=cfg.citations_enabled` from Retriever._assemble."""
        context = self._stub(citations_enabled=True)._assemble(QUERY, f1_results())

        assert context.context_text == TAGGED_CONTEXT
        assert sources_of(context) == F1_SOURCES

    def test_flag_off_leaves_the_assembler_output_unchanged(self) -> None:
        context = self._stub()._assemble(QUERY, f1_results())

        assert context.context_text == UNTAGGED_CONTEXT
        assert context.citation_sources == ()


def test_cite_tag_is_the_header_prefix() -> None:
    assert cite_tag(1) == "[C1] "
    assert cite_tag(12) == "[C12] "
