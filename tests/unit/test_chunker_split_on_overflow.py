"""Regression tests: an over-budget symbol must be SPLIT across multiple
chunks, never truncated-and-discarded.

Before this fix, `Chunker._truncate_chunk` cut a chunk to `max_tokens_per_chunk`
and appended "# ... (truncated)" — the excess body content was gone forever;
neither the vector index nor BM25 nor any retrieval leg could ever surface it,
because no row anywhere stored it. `Chunk.symbol_id` is a plain `int`, and the
`chunks` table has no unique constraint on `symbol_id` (only an index + a plain
FK to `symbols.id`), so multiple chunk rows legally pointing at one symbol was
already schema-safe before this change — the indexer's insert loop
(`indexer.py::_insert_and_chunk_all`) inserts rows one at a time keyed by the
per-row `chunk_id` it gets back from `Database.insert_chunk`, not by
`symbol_id`, so it never assumed exactly one chunk per symbol either.

`test_large_symbol_tail_is_not_silently_discarded` is the TDD anchor for this
file: written to fail against the old truncate-and-discard `_truncate_chunk`
(a sentinel word placed at the very tail of an over-budget body would never
appear in `chunks[0].chunk_text`, and there was never a `chunks[1]`), and to
pass once `Chunker.build_chunks` splits instead of truncating.
"""

from __future__ import annotations

import re

import tiktoken

from trelix.core.config import ChunkerConfig
from trelix.core.models import Symbol, SymbolKind
from trelix.indexing.chunker import Chunker, ContextualChunker

_ENCODING = tiktoken.get_encoding("cl100k_base")
_TRUNCATION_MARKER = "# ... (truncated)"
_SPLIT_MARKER_RE = re.compile(r"# Split: chunk \d+ of \d+\n\n")

_REL_PATH = "src/pkg/mod.py"
_LANGUAGE = "python"


def _count(text: str) -> int:
    return len(_ENCODING.encode(text))


def _filler_body(token_count: int) -> str:
    """`token_count` copies of the one-cl100k-token word "zz", space-joined.

    Same fixture convention as test_chunker_token_budget_boundary.py.
    """
    return " ".join(["zz"] * token_count)


def _symbol(body: str, *, id: int = 1, docstring: str | None = None) -> Symbol:
    return Symbol(
        id=id,
        file_id=1,
        name="my_func",
        qualified_name="my_func",
        kind=SymbolKind.FUNCTION,
        line_start=1,
        line_end=1,
        signature="def my_func()",
        body=body,
        docstring=docstring,
        parent_id=None,
    )


def _split_off_marker(chunk_text: str) -> str:
    """Strip the leading header through the "# Split: chunk i of n" marker,
    returning only the piece's body content. Fails loudly if the marker is
    missing or appears more than once, rather than silently returning the
    whole text.
    """
    parts = _SPLIT_MARKER_RE.split(chunk_text, maxsplit=1)
    assert len(parts) == 2, (
        f"expected exactly one '# Split: chunk i of n' marker in chunk_text, "
        f"found {len(parts) - 1}: {chunk_text!r}"
    )
    return parts[1]


class TestLargeSymbolIsNotSilentlyDiscarded:
    """The TDD anchor: fails under the old truncate-and-discard behavior."""

    def test_large_symbol_tail_is_not_silently_discarded(self) -> None:
        """A sentinel placed at the very tail of an over-budget body must
        survive SOMEWHERE across the concatenated, in-order piece bodies —
        not necessarily whole within a single `chunk_text`. The sentinel is
        not one cl100k token, so a chunk boundary can legitimately fall
        mid-sentinel (split across `chunks[i]`'s tail and `chunks[i+1]`'s
        head) with zero bytes lost — checking a single `chunk_text` for the
        literal substring would wrongly fail on that legitimate case, so this
        recombines piece bodies first, exactly as `_recombine_body`-style
        reconstruction tests elsewhere in this file do.

        Fails under: reverting to `_truncate_chunk` (single chunk, tail cut
        off and replaced with "# ... (truncated)") — the sentinel is absent
        from the recombined text entirely, not just split across a boundary.
        """
        max_tokens = 40
        sentinel = "UNIQUE_TAIL_SENTINEL_MARKER"
        body = _filler_body(120) + " " + sentinel

        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=max_tokens))
        symbol = _symbol(body)
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        # Precondition: this body really does overflow a single chunk, so the
        # assertion below is not vacuously true for a body that never needed
        # splitting in the first place.
        single_chunk_tokens = _count(
            chunker._build_chunk_text(
                symbol=symbol,
                file_rel_path=_REL_PATH,
                language=_LANGUAGE,
                import_header="",
                parent_symbols={},
            )
        )
        assert single_chunk_tokens > max_tokens, (
            "FIXTURE NO LONGER DISCRIMINATES: body fits in one chunk, so "
            "splitting/truncation is never exercised."
        )

        recombined = "".join(_split_off_marker(c.chunk_text) for c in chunks)
        assert sentinel in recombined, (
            "tail sentinel is missing from the recombined piece bodies — "
            "content was silently discarded instead of split into a further chunk"
        )


class TestSplitProducesMultipleChunks:
    def test_over_budget_symbol_yields_more_than_one_chunk(self) -> None:
        max_tokens = 40
        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=max_tokens))
        symbol = _symbol(_filler_body(120))
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(chunks) > 1

    def test_all_split_chunks_share_the_parent_symbol_id(self) -> None:
        max_tokens = 40
        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=max_tokens))
        symbol = _symbol(_filler_body(120), id=42)
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(chunks) > 1
        assert all(c.symbol_id == 42 for c in chunks)

    def test_no_truncation_marker_appears_in_any_split_chunk(self) -> None:
        max_tokens = 40
        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=max_tokens))
        symbol = _symbol(_filler_body(120))
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(chunks) > 1
        assert all(_TRUNCATION_MARKER not in c.chunk_text for c in chunks)

    def test_every_split_chunk_repeats_the_structural_header(self) -> None:
        """Each piece must stay independently embeddable — the module's own
        stated design goal for the header — so every piece, not just the
        first, must carry the "# File:" line.
        """
        max_tokens = 40
        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=max_tokens))
        symbol = _symbol(_filler_body(120))
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(chunks) > 1
        assert all(c.chunk_text.startswith(f"# File: {_REL_PATH}") for c in chunks)

    def test_every_split_chunk_token_count_matches_its_own_text(self) -> None:
        """Same recount invariant the old truncation path had, extended to
        every piece, not just the (sole) old chunk.
        """
        max_tokens = 40
        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=max_tokens))
        symbol = _symbol(_filler_body(120))
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count == _count(c.chunk_text)

    def test_every_split_chunk_token_count_stays_within_budget(self) -> None:
        """Regression guard: `_split_chunk_text`'s per-piece budget used to be
        computed by summing independently-encoded header/marker token counts
        (`header_tokens + marker_tokens`), which does not equal the token
        count of the actual joined skeleton text -- tiktoken's BPE can merge
        tokens across concatenation boundaries, so separately-encoded lengths
        don't reliably add up to the length of the joined string. Live repro
        confirmed this produced pieces measuring exactly one token OVER
        budget every time: budget=40 -> 41-token pieces, budget=60 ->
        61-token pieces, budget=100 -> 101-token pieces -- despite the
        per-piece math "adding up" on paper.
        """
        for max_tokens in (40, 60, 100):
            chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=max_tokens))
            symbol = _symbol(_filler_body(max_tokens * 3))
            chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

            assert len(chunks) > 1, "FIXTURE NO LONGER DISCRIMINATES: body fits in one chunk."
            for c in chunks:
                assert c.token_count <= max_tokens, (
                    f"piece token_count {c.token_count} exceeds max_tokens_per_chunk={max_tokens}"
                )


class TestSplitChunksReconstructFullBodyContent:
    def test_concatenated_split_bodies_equal_original_body_exactly(self) -> None:
        """The core bug fix, proven directly: splitting the body across N
        chunks and stitching the pieces back together (by their declared
        order) must reproduce the ORIGINAL body byte-for-byte -- no gap, no
        duplication, no reordering.
        """
        max_tokens = 40
        body = _filler_body(153)  # deliberately not a multiple of any budget
        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=max_tokens))
        symbol = _symbol(body)
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(chunks) > 1, "FIXTURE NO LONGER DISCRIMINATES: body fits in one chunk."

        recombined = "".join(_split_off_marker(c.chunk_text) for c in chunks)
        assert recombined == body

    def test_reconstruction_holds_for_a_realistic_multiline_function_body(self) -> None:
        """Same invariant, but with real code (newlines, indentation) instead
        of a synthetic space-joined filler, so the reconstruction isn't only
        proven for a body shape that never appears in a real repo.
        """
        max_tokens = 60
        body = "def big_function():\n" + "    x = 1  # padding line\n" * 60
        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=max_tokens))
        symbol = _symbol(body)
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(chunks) > 1, "FIXTURE NO LONGER DISCRIMINATES: body fits in one chunk."

        recombined = "".join(_split_off_marker(c.chunk_text) for c in chunks)
        assert recombined == body


class TestContextualChunkerSplitSummaryPlacement:
    """Design decision under test: an LLM-generated context summary describes
    the WHOLE symbol once, so it belongs on the FIRST split piece only --
    repeating it on every later piece would spend that piece's scarce
    body-token budget re-embedding a summary the reader already saw.
    """

    def test_summary_appears_only_on_the_first_split_chunk(self) -> None:
        class _StubClient:
            def chat(self) -> None:  # pragma: no cover - not used via this path
                raise NotImplementedError

        class _Completions:
            def create(self, **kwargs: object) -> object:
                class _Msg:
                    content = "Handles a large amount of padding."

                class _Choice:
                    message = _Msg()

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        max_tokens = 40
        summary = "Handles a large amount of padding."
        chunker = ContextualChunker(
            ChunkerConfig(max_tokens_per_chunk=max_tokens, contextual=True),
            llm_client=_Client(),
        )
        symbol = _symbol(_filler_body(120))
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(chunks) > 1, "FIXTURE NO LONGER DISCRIMINATES: body fits in one chunk."
        assert chunks[0].chunk_text.startswith(summary + "\n\n")
        assert all(summary not in c.chunk_text for c in chunks[1:])

    def test_split_content_preserved_with_contextual_summary_active(self) -> None:
        class _Completions:
            def create(self, **kwargs: object) -> object:
                class _Msg:
                    content = "A summary."

                class _Choice:
                    message = _Msg()

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        max_tokens = 40
        body = _filler_body(150)
        chunker = ContextualChunker(
            ChunkerConfig(max_tokens_per_chunk=max_tokens, contextual=True),
            llm_client=_Client(),
        )
        symbol = _symbol(body)
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(chunks) > 1, "FIXTURE NO LONGER DISCRIMINATES: body fits in one chunk."
        recombined = "".join(_split_off_marker(c.chunk_text) for c in chunks)
        assert recombined == body


class TestSplitDoesNotRegressUnsplitBehavior:
    """A symbol that already fits within budget must be completely unaffected
    -- still exactly one chunk, with no split marker at all.
    """

    def test_small_symbol_still_produces_exactly_one_chunk_with_no_marker(self) -> None:
        chunker = Chunker(ChunkerConfig())
        symbol = _symbol("def f():\n    return 1")
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(chunks) == 1
        assert "# Split:" not in chunks[0].chunk_text
        assert _TRUNCATION_MARKER not in chunks[0].chunk_text


class TestOversizedHeaderWithEmptyBodyNeverVanishes:
    """Regression guard for a real bug found in review: `_split_chunk_text`'s
    body-splitting loop was `while idx < len(body_tokens):`, which never runs
    at all when `body_tokens` is empty -- so a symbol whose BODY is empty (or
    tiny) but whose HEADER alone (e.g. a very long docstring, or many
    imports) already exceeds `max_tokens_per_chunk` got ZERO chunks back from
    `build_chunks`, not one. Zero chunks means the symbol is completely
    absent from every index (vector, BM25, everything) with no record it
    ever existed -- strictly worse than the old truncate-and-discard
    behavior this split path replaced, which at least produced one lossy
    chunk. A single over-budget "header-only" chunk is the correct fallback.
    """

    def test_empty_body_with_oversized_docstring_header_yields_one_chunk(self) -> None:
        max_tokens = 40
        # A docstring alone, with nothing else, pushes the header well past
        # max_tokens even though symbol.body is empty.
        symbol = _symbol("", docstring=_filler_body(200))
        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=max_tokens))
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(chunks) == 1, (
            f"expected exactly one (over-budget) chunk for an empty-body "
            f"oversized-header symbol, got {len(chunks)} -- zero means the "
            f"symbol silently vanished from the index"
        )
        assert chunks[0].token_count == _count(chunks[0].chunk_text)
        assert chunks[0].token_count > max_tokens
        assert _TRUNCATION_MARKER not in chunks[0].chunk_text

    def test_empty_body_with_oversized_docstring_header_via_contextual_chunker(self) -> None:
        """Same guarantee holds through `ContextualChunker.build_chunks`,
        which has its own copy of the split-branch call.
        """

        class _Completions:
            def create(self, **kwargs: object) -> object:
                class _Msg:
                    content = "A summary."

                class _Choice:
                    message = _Msg()

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        max_tokens = 40
        symbol = _symbol("", docstring=_filler_body(200))
        chunker = ContextualChunker(
            ChunkerConfig(max_tokens_per_chunk=max_tokens, contextual=True),
            llm_client=_Client(),
        )
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(chunks) == 1, (
            f"expected exactly one (over-budget) chunk, got {len(chunks)} -- "
            f"zero means the symbol silently vanished from the index"
        )
        assert chunks[0].token_count == _count(chunks[0].chunk_text)
        assert chunks[0].token_count > max_tokens

    def test_tiny_nonempty_body_with_oversized_header_still_yields_at_least_one_chunk(
        self,
    ) -> None:
        """Same guarantee for a body that has SOME tokens, just not enough to
        ever reach the loop's original entry condition in a meaningfully
        different way than the empty-body case -- content must never be
        dropped either.
        """
        max_tokens = 40
        symbol = _symbol("x", docstring=_filler_body(200))
        chunker = Chunker(ChunkerConfig(max_tokens_per_chunk=max_tokens))
        chunks = chunker.build_chunks([symbol], [], _REL_PATH, _LANGUAGE)

        assert len(chunks) >= 1
        recombined = "".join(_split_off_marker(c.chunk_text) for c in chunks)
        assert recombined == "x"
