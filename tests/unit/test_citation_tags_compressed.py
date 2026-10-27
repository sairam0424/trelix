"""Citation tags on COMPRESSED bodies: one tag on every kept-span header, never on text.

Companion to test_citation_tags.py. A partially-kept body renders one ``[Lines a-b]``
header per kept span (``format_compressed_blocks``); with tags on, every one of those
headers, the fallback header included, carries the symbol's ``[C<n>] `` prefix, and the
prefix is built INTO the header rather than applied to the rendered text, so a source line
that starts with ``[Lines `` is left alone. ``citation_sources`` records the symbol's full
range, not the span envelope.

Every expected value is a literal written in this file. Each test names the mutation
that must make it fail.
"""

from __future__ import annotations

import tiktoken

from tests.fixtures.citation_context import (
    BEARER_BLOCK,
    MIDDLEWARE,
    QUERY,
    bearer_result,
    make_result,
    sources_of,
    verify_result,
)
from trelix.compression import CompressionResult, ExtractiveCompressor
from trelix.core.models import SearchResult, SymbolKind
from trelix.retrieval.assembler import ContextAssembler
from trelix.retrieval.context_compression import format_compressed_blocks


def _verify_with_26_line_body() -> SearchResult:
    """verify's symbol spans lines 42-67; its body has one line per line number."""
    result = verify_result()
    body = "\n".join(f"L{n}" for n in range(42, 68))
    result.symbol.body = body
    result.chunk.chunk_text = body
    return result


def _two_kept_spans() -> CompressionResult:
    return CompressionResult(
        text="L42\nL43\nL44\nL45\n# ... 9 lines elided ...\nL55\nL56\n# ... 11 lines elided ...",
        token_count=9,
        original_token_count=20,
        kept_spans=[(42, 45), (55, 56)],
    )


COMPRESSED_PLAIN = (
    "[Lines 42-45] AuthMiddleware.verify\nL42\nL43\nL44\nL45\n"
    "# ... 9 lines elided ...\n"
    "[Lines 55-56] AuthMiddleware.verify\nL55\nL56\n"
    "# ... 11 lines elided ..."
)
COMPRESSED_TAGGED = (
    "[C1] [Lines 42-45] AuthMiddleware.verify\nL42\nL43\nL44\nL45\n"
    "# ... 9 lines elided ...\n"
    "[C1] [Lines 55-56] AuthMiddleware.verify\nL55\nL56\n"
    "# ... 11 lines elided ..."
)


class _NoSubChunkDB:
    """An index with no sub_chunks table: the extractive compressor's lexical path."""

    _conn = None

    def get_sub_chunks_for_symbol(self, symbol_id: int) -> list[object]:  # noqa: ARG002
        return []


class TestCompressedBlocks:
    def test_every_kept_span_header_carries_the_tag(self) -> None:
        """MUTATION: build the span headers without `header_prefix`."""
        rendered = format_compressed_blocks(
            _verify_with_26_line_body(), _two_kept_spans(), header_prefix="[C1] "
        )
        assert rendered == COMPRESSED_TAGGED

    def test_default_prefix_renders_byte_for_byte_as_before(self) -> None:
        assert format_compressed_blocks(_verify_with_26_line_body(), _two_kept_spans()) == (
            COMPRESSED_PLAIN
        )

    def test_fallback_header_carries_the_tag_too(self) -> None:
        """MUTATION: prefix the span headers but not `_fallback_block`'s."""
        unmappable = CompressionResult(
            text="def verify(...): ...",
            token_count=7,
            original_token_count=20,
            kept_spans=[(9_000, 9_001)],
        )
        result = _verify_with_26_line_body()

        assert format_compressed_blocks(result, unmappable, header_prefix="[C1] ") == (
            "[C1] [Lines 42-42] AuthMiddleware.verify\ndef verify(...): ..."
        )
        assert format_compressed_blocks(result, unmappable) == (
            "[Lines 42-42] AuthMiddleware.verify\ndef verify(...): ..."
        )

    def test_source_text_that_looks_like_a_header_is_untouched(self) -> None:
        """MUTATION: tag by rewriting `^\\[Lines ` over the rendered block and the
        source line gains a prefix too (it is a Markdown chunk, not a header)."""
        body = "[Lines 1-2] not a header, source text\nreal line\nmore\nend"
        doc = make_result(
            symbol_id=77,
            rel_path="docs/x.md",
            file_id=3,
            name="doc",
            qualified_name="doc",
            kind=SymbolKind.SECTION,
            line_start=1,
            line_end=4,
            body=body,
            score=0.5,
            rank=1,
        )
        doc.chunk.chunk_text = body
        kept = CompressionResult(
            text="[Lines 1-2] not a header, source text\nreal line",
            token_count=4,
            original_token_count=5,
            kept_spans=[(1, 2)],
        )

        assert format_compressed_blocks(doc, kept, header_prefix="[C1] ") == (
            "[C1] [Lines 1-2] doc\n[Lines 1-2] not a header, source text\nreal line\n"
            "# ... 2 lines elided ..."
        )
        plain = ContextAssembler(token_budget=8_000, cite_tags=True).assemble(QUERY, [doc])
        assert plain.context_text == (
            "=== docs/x.md ===\n\n[C1] [Lines 1-4] doc\n"
            "[Lines 1-2] not a header, source text\nreal line\nmore\nend\n"
        )

    def test_assembler_tags_compressed_and_plain_blocks_alike(self) -> None:
        """One symbol, one tag, one source with the symbol's FULL range (42-67, not 42-56).

        MUTATION: render the compressed block without the prefix; record the span
        envelope as the source's range."""
        assembler = ContextAssembler(token_budget=8_000, cite_tags=True)

        text = assembler._format_context(
            [_verify_with_26_line_body(), bearer_result()], compressed={11: _two_kept_spans()}
        )

        assert text == (
            "=== src/auth/middleware.py ===\n\n" + COMPRESSED_TAGGED + "\n\n[C2] " + BEARER_BLOCK
        )
        assert [(s.tag, s.line_start, s.line_end) for s in assembler._citation_sources] == [
            (1, 42, 67),
            (2, 70, 80),
        ]

    def test_live_compression_tags_every_header_by_symbol(self) -> None:
        """End to end through the extractive compressor's lexical path (no inference).

        The body, query, ratio and budget are those of test_assembler_compression.py's
        citation-fidelity gate, which proves this fixture keeps `beta` as several spans."""
        encoding = tiktoken.get_encoding("cl100k_base")

        def _long(name: str, line_start: int, symbol_id: int, score: float) -> SearchResult:
            lines = [f"def {name}(request, session, retries):", f'    """Handle {name}."""']
            for step in range(8):
                lines += [
                    "",
                    f"    # step {step}: prepare the payload for stage {step}",
                    f"    payload_{step} = build_payload(request, stage={step})",
                    f"    validated_{step} = validate(payload_{step}, session, retries)",
                    f"    dispatched_{step} = dispatch(validated_{step}, timeout={step + 1})",
                ]
            lines += ["", "    return aggregate(dispatched_0)"]
            body = "\n".join(lines)
            result = make_result(
                symbol_id=symbol_id,
                rel_path=MIDDLEWARE,
                file_id=1,
                name=name,
                qualified_name=f"Service.{name}",
                kind=SymbolKind.FUNCTION,
                line_start=line_start,
                line_end=line_start + len(lines) - 1,
                body=body,
                score=score,
                rank=1,
            )
            result.chunk.chunk_text = body  # the chunk IS the body, so token accounting is exact
            result.chunk.token_count = len(encoding.encode(body))
            return result

        alpha, beta = _long("alpha", 10, 1, 0.9), _long("beta", 60, 2, 0.8)
        context = ContextAssembler(
            token_budget=alpha.chunk.token_count + beta.chunk.token_count // 2,
            compressor=ExtractiveCompressor(db=_NoSubChunkDB(), embedder=None),
            compression_ratio=0.45,
            compression_min_tokens=10,
            cite_tags=True,
        ).assemble("how does dispatch build the payload for stage 3", [alpha, beta])

        headers = [line for line in context.context_text.splitlines() if "[Lines " in line]
        alpha_headers = [h for h in headers if h.endswith("Service.alpha")]
        beta_headers = [h for h in headers if h.endswith("Service.beta")]
        assert len(beta_headers) >= 2, "premise: beta was kept compressed, one header per span"
        assert alpha_headers == ["[C1] [Lines 10-53] Service.alpha"]
        assert all(h.startswith("[C2] [Lines ") for h in beta_headers)
        assert len(headers) == len(alpha_headers) + len(beta_headers)
        assert sources_of(context) == [
            (1, 1, "src/auth/middleware.py", 10, 53, "Service.alpha"),
            (2, 2, "src/auth/middleware.py", 60, 103, "Service.beta"),
        ]
