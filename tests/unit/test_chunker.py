"""Unit tests for the Chunker (Phase 7) and ContextualChunker (U1)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import ChunkerConfig
from trelix.core.models import Chunk, ImportEdge, Symbol, SymbolKind
from trelix.indexing.chunker import Chunker, ContextualChunker


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

def _make_symbol(
    *,
    id: int = 1,
    file_id: int = 1,
    name: str = "my_func",
    qualified_name: str = "my_func",
    kind: SymbolKind = SymbolKind.FUNCTION,
    body: str = "def my_func():\n    return 42",
    docstring: str | None = None,
    parent_id: int | None = None,
    line_start: int = 1,
    line_end: int = 2,
    signature: str = "def my_func()",
) -> Symbol:
    return Symbol(
        id=id,
        file_id=file_id,
        name=name,
        qualified_name=qualified_name,
        kind=kind,
        line_start=line_start,
        line_end=line_end,
        signature=signature,
        body=body,
        docstring=docstring,
        parent_id=parent_id,
    )


def _make_import(
    *,
    file_id: int = 1,
    imported_from: str = "os",
    imported_names: list[str] | None = None,
) -> ImportEdge:
    return ImportEdge(
        file_id=file_id,
        imported_from=imported_from,
        imported_names=imported_names or [],
    )


def _make_chunker(**kwargs) -> Chunker:
    config = ChunkerConfig(**kwargs)
    return Chunker(config)


# ---------------------------------------------------------------------------
# Context header format
# ---------------------------------------------------------------------------

class TestContextHeaderFormat:
    def test_header_contains_file_path(self) -> None:
        chunker = _make_chunker()
        symbol = _make_symbol()
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        assert "src/foo.py" in chunks[0].chunk_text

    def test_header_contains_language(self) -> None:
        chunker = _make_chunker()
        symbol = _make_symbol()
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        assert "Python" in chunks[0].chunk_text

    def test_header_format_exact_pattern(self) -> None:
        """Header must be: # File: <path> | Language: <Language>"""
        chunker = _make_chunker()
        symbol = _make_symbol()
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        first_line = chunks[0].chunk_text.splitlines()[0]
        assert first_line == "# File: src/foo.py | Language: Python"

    def test_language_is_capitalized(self) -> None:
        chunker = _make_chunker()
        symbol = _make_symbol()
        chunks = chunker.build_chunks([symbol], [], "src/bar.ts", "typescript")
        first_line = chunks[0].chunk_text.splitlines()[0]
        assert "Typescript" in first_line

    def test_one_chunk_per_symbol(self) -> None:
        chunker = _make_chunker()
        symbols = [_make_symbol(id=1, name="fn_a"), _make_symbol(id=2, name="fn_b")]
        chunks = chunker.build_chunks(symbols, [], "src/foo.py", "python")
        assert len(chunks) == 2

    def test_empty_symbols_returns_empty_list(self) -> None:
        chunker = _make_chunker()
        chunks = chunker.build_chunks([], [], "src/foo.py", "python")
        assert chunks == []


# ---------------------------------------------------------------------------
# Parent class signature in header
# ---------------------------------------------------------------------------

class TestParentClassInHeader:
    def test_parent_class_name_in_header_when_enabled(self) -> None:
        chunker = _make_chunker(include_parent_signature=True)
        parent = _make_symbol(id=10, name="MyClass", kind=SymbolKind.CLASS)
        method = _make_symbol(id=11, name="do_thing", parent_id=10)
        parent_symbols = {10: parent}
        chunks = chunker.build_chunks([method], [], "src/foo.py", "python", parent_symbols)
        assert "MyClass" in chunks[0].chunk_text

    def test_parent_header_line_format(self) -> None:
        """Parent line must be: # Class: <ClassName>"""
        chunker = _make_chunker(include_parent_signature=True)
        parent = _make_symbol(id=10, name="LoginView", kind=SymbolKind.CLASS)
        method = _make_symbol(id=11, name="authenticate", parent_id=10)
        parent_symbols = {10: parent}
        chunks = chunker.build_chunks([method], [], "src/auth.py", "python", parent_symbols)
        text = chunks[0].chunk_text
        assert any(line == "# Class: LoginView" for line in text.splitlines())

    def test_parent_class_absent_when_disabled(self) -> None:
        chunker = _make_chunker(include_parent_signature=False)
        parent = _make_symbol(id=10, name="ShouldNotAppear", kind=SymbolKind.CLASS)
        method = _make_symbol(id=11, name="method", parent_id=10)
        parent_symbols = {10: parent}
        chunks = chunker.build_chunks([method], [], "src/foo.py", "python", parent_symbols)
        assert "ShouldNotAppear" not in chunks[0].chunk_text

    def test_no_parent_entry_when_parent_id_missing(self) -> None:
        """Symbol with no parent_id must not produce a # Class: line."""
        chunker = _make_chunker(include_parent_signature=True)
        symbol = _make_symbol(parent_id=None)
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        text = chunks[0].chunk_text
        assert not any(line.startswith("# Class:") for line in text.splitlines())

    def test_parent_lookup_uses_parent_id(self) -> None:
        """parent_symbols dict key is parent_id — wrong key must not appear."""
        chunker = _make_chunker(include_parent_signature=True)
        parent = _make_symbol(id=99, name="CorrectParent", kind=SymbolKind.CLASS)
        method = _make_symbol(id=11, name="method", parent_id=99)
        parent_symbols = {99: parent}
        chunks = chunker.build_chunks([method], [], "src/foo.py", "python", parent_symbols)
        assert "CorrectParent" in chunks[0].chunk_text


# ---------------------------------------------------------------------------
# Import list in header
# ---------------------------------------------------------------------------

class TestImportListInHeader:
    def test_imports_present_when_enabled(self) -> None:
        chunker = _make_chunker(include_imports_in_header=True)
        symbol = _make_symbol()
        imp = _make_import(imported_from="django.contrib.auth", imported_names=["authenticate"])
        chunks = chunker.build_chunks([symbol], [imp], "src/foo.py", "python")
        assert "django.contrib.auth" in chunks[0].chunk_text

    def test_imports_absent_when_disabled(self) -> None:
        chunker = _make_chunker(include_imports_in_header=False)
        symbol = _make_symbol()
        imp = _make_import(imported_from="should.not.appear", imported_names=["fn"])
        chunks = chunker.build_chunks([symbol], [imp], "src/foo.py", "python")
        assert "should.not.appear" not in chunks[0].chunk_text

    def test_imports_line_starts_with_hash_imports(self) -> None:
        chunker = _make_chunker(include_imports_in_header=True)
        symbol = _make_symbol()
        imp = _make_import(imported_from="os", imported_names=["path"])
        chunks = chunker.build_chunks([symbol], [imp], "src/foo.py", "python")
        lines = chunks[0].chunk_text.splitlines()
        import_lines = [ln for ln in lines if ln.startswith("# Imports:")]
        assert len(import_lines) == 1

    def test_no_import_line_when_no_imports(self) -> None:
        chunker = _make_chunker(include_imports_in_header=True)
        symbol = _make_symbol()
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        lines = chunks[0].chunk_text.splitlines()
        assert not any(ln.startswith("# Imports:") for ln in lines)

    def test_max_imports_respected(self) -> None:
        chunker = _make_chunker(include_imports_in_header=True, max_imports_in_header=2)
        symbol = _make_symbol()
        imports = [
            _make_import(imported_from=f"module_{i}", imported_names=["x"])
            for i in range(5)
        ]
        chunks = chunker.build_chunks([symbol], imports, "src/foo.py", "python")
        # Only module_0 and module_1 should appear (max=2)
        assert "module_2" not in chunks[0].chunk_text
        assert "module_0" in chunks[0].chunk_text

    def test_wildcard_import_shows_module_only(self) -> None:
        """imported_names=["*"] must show just the module path, no braces."""
        chunker = _make_chunker(include_imports_in_header=True)
        symbol = _make_symbol()
        imp = _make_import(imported_from="some.module", imported_names=["*"])
        chunks = chunker.build_chunks([symbol], [imp], "src/foo.py", "python")
        text = chunks[0].chunk_text
        assert "some.module" in text
        assert "{" not in text


# ---------------------------------------------------------------------------
# token_count computed via tiktoken
# ---------------------------------------------------------------------------

class TestTokenCount:
    def test_token_count_is_positive(self) -> None:
        chunker = _make_chunker()
        symbol = _make_symbol(body="def hello():\n    pass")
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        assert chunks[0].token_count > 0

    def test_token_count_is_integer(self) -> None:
        chunker = _make_chunker()
        symbol = _make_symbol()
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        assert isinstance(chunks[0].token_count, int)

    def test_token_count_matches_tiktoken(self) -> None:
        import tiktoken
        chunker = _make_chunker()
        symbol = _make_symbol(body="def greet():\n    return 'hello'")
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        enc = tiktoken.get_encoding("cl100k_base")
        expected = len(enc.encode(chunks[0].chunk_text))
        assert chunks[0].token_count == expected

    def test_longer_body_has_more_tokens(self) -> None:
        chunker = _make_chunker(max_tokens_per_chunk=10_000)
        short = _make_symbol(id=1, body="def f(): pass")
        long_body = "def f():\n" + "    x = 1\n" * 50
        long = _make_symbol(id=2, body=long_body)
        chunks = chunker.build_chunks([short, long], [], "src/foo.py", "python")
        assert chunks[1].token_count > chunks[0].token_count

    def test_truncation_caps_token_count(self) -> None:
        max_tokens = 20
        chunker = _make_chunker(max_tokens_per_chunk=max_tokens)
        large_body = "def f():\n" + "    # comment line\n" * 200
        symbol = _make_symbol(body=large_body)
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        # After truncation token_count is capped at max_tokens
        assert chunks[0].token_count <= max_tokens


# ---------------------------------------------------------------------------
# chunk_text contains symbol body
# ---------------------------------------------------------------------------

class TestChunkTextContainsBody:
    def test_chunk_text_contains_body(self) -> None:
        chunker = _make_chunker()
        body = "def compute(x: int) -> int:\n    return x * 2"
        symbol = _make_symbol(body=body)
        chunks = chunker.build_chunks([symbol], [], "src/math.py", "python")
        assert body in chunks[0].chunk_text

    def test_chunk_text_starts_with_header_not_body(self) -> None:
        chunker = _make_chunker()
        symbol = _make_symbol(body="def fn(): pass")
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        assert chunks[0].chunk_text.startswith("# File:")

    def test_body_appears_after_header(self) -> None:
        chunker = _make_chunker()
        body = "def fn(): pass"
        symbol = _make_symbol(body=body)
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        text = chunks[0].chunk_text
        header_end = text.index("\n\n")  # blank line separates header from body
        body_section = text[header_end:]
        assert body in body_section

    def test_chunk_returns_chunk_dataclass(self) -> None:
        chunker = _make_chunker()
        symbol = _make_symbol()
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        assert isinstance(chunks[0], Chunk)

    def test_chunk_symbol_id_matches(self) -> None:
        chunker = _make_chunker()
        symbol = _make_symbol(id=42)
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        assert chunks[0].symbol_id == 42

    def test_docstring_surfaced_before_body_when_not_in_body(self) -> None:
        """When the body does NOT start with a string literal, the docstring
        should be prepended as a # Doc: comment."""
        chunker = _make_chunker()
        body = "def fn():\n    x = 1"
        symbol = _make_symbol(body=body, docstring="Does something useful.")
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        text = chunks[0].chunk_text
        doc_idx = text.find("# Doc:")
        body_idx = text.find(body)
        assert doc_idx != -1
        assert doc_idx < body_idx

    def test_docstring_not_doubled_when_body_starts_with_triple_quote(self) -> None:
        """When the body already starts with a docstring literal, # Doc: must NOT appear."""
        chunker = _make_chunker()
        body = '"""Does something useful."""\ndef fn(): pass'
        symbol = _make_symbol(body=body, docstring="Does something useful.")
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        assert "# Doc:" not in chunks[0].chunk_text


# ---------------------------------------------------------------------------
# ContextualChunker (U1)
# ---------------------------------------------------------------------------

def _make_mock_llm_client(summary: str = "This function computes a value.") -> MagicMock:
    """Build a mock OpenAI client whose .chat.completions.create() returns `summary`."""
    mock_message = MagicMock()
    mock_message.content = summary
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def _make_contextual_chunker(
    contextual: bool = True,
    llm_client: object = None,
    **kwargs: object,
) -> ContextualChunker:
    config = ChunkerConfig(contextual=contextual, **kwargs)
    return ContextualChunker(config, llm_client=llm_client)


class TestContextualChunker:
    def test_contextual_true_prepends_summary_to_chunk_text(self) -> None:
        """With contextual=True and a mock client, summary is prepended to chunk_text."""
        summary = "This function authenticates a user."
        mock_client = _make_mock_llm_client(summary)
        chunker = _make_contextual_chunker(contextual=True, llm_client=mock_client)
        symbol = _make_symbol()
        chunks = chunker.build_chunks([symbol], [], "src/auth.py", "python")
        assert chunks[0].chunk_text.startswith(summary)

    def test_contextual_true_summary_separated_by_blank_line(self) -> None:
        """Summary and base chunk_text must be separated by a blank line."""
        summary = "Handles login logic."
        mock_client = _make_mock_llm_client(summary)
        chunker = _make_contextual_chunker(contextual=True, llm_client=mock_client)
        symbol = _make_symbol()
        chunks = chunker.build_chunks([symbol], [], "src/auth.py", "python")
        # The blank line separator means summary\n\n appears in the text
        assert f"{summary}\n\n" in chunks[0].chunk_text

    def test_contextual_true_stores_summary_on_symbol(self) -> None:
        """context_summary must be stored on the Symbol object after build_chunks."""
        summary = "Returns the square of x."
        mock_client = _make_mock_llm_client(summary)
        chunker = _make_contextual_chunker(contextual=True, llm_client=mock_client)
        symbol = _make_symbol()
        chunker.build_chunks([symbol], [], "src/math.py", "python")
        assert symbol.context_summary == summary

    def test_contextual_true_token_count_includes_summary(self) -> None:
        """token_count must reflect summary + base chunk_text combined."""
        import tiktoken
        summary = "Computes a result."
        mock_client = _make_mock_llm_client(summary)
        chunker = _make_contextual_chunker(contextual=True, llm_client=mock_client)
        symbol = _make_symbol()
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")

        enc = tiktoken.get_encoding("cl100k_base")
        expected = len(enc.encode(chunks[0].chunk_text))
        assert chunks[0].token_count == expected

    def test_contextual_false_identical_to_base_chunker(self) -> None:
        """With contextual=False the output must be byte-for-byte identical to Chunker."""
        symbol = _make_symbol()
        base_chunker = Chunker(ChunkerConfig())
        ctx_chunker = _make_contextual_chunker(contextual=False, llm_client=None)

        base_chunks = base_chunker.build_chunks([symbol], [], "src/foo.py", "python")
        ctx_chunks = ctx_chunker.build_chunks([symbol], [], "src/foo.py", "python")

        assert len(base_chunks) == len(ctx_chunks)
        assert base_chunks[0].chunk_text == ctx_chunks[0].chunk_text
        assert base_chunks[0].token_count == ctx_chunks[0].token_count

    def test_contextual_true_no_client_falls_back_to_base_output(self) -> None:
        """contextual=True but llm_client=None must behave identically to base Chunker."""
        symbol = _make_symbol()
        base_chunker = Chunker(ChunkerConfig())
        ctx_chunker = _make_contextual_chunker(contextual=True, llm_client=None)

        base_chunks = base_chunker.build_chunks([symbol], [], "src/foo.py", "python")
        ctx_chunks = ctx_chunker.build_chunks([symbol], [], "src/foo.py", "python")

        assert base_chunks[0].chunk_text == ctx_chunks[0].chunk_text

    def test_contextual_true_llm_failure_falls_back_gracefully(self) -> None:
        """If the LLM call raises an exception, fall back to base chunk_text silently."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        symbol = _make_symbol()
        base_chunker = Chunker(ChunkerConfig())
        ctx_chunker = _make_contextual_chunker(contextual=True, llm_client=mock_client)

        base_chunks = base_chunker.build_chunks([symbol], [], "src/foo.py", "python")
        ctx_chunks = ctx_chunker.build_chunks([symbol], [], "src/foo.py", "python")

        # No summary prepended — identical to base
        assert base_chunks[0].chunk_text == ctx_chunks[0].chunk_text
        # context_summary is None when the call failed
        assert symbol.context_summary is None

    def test_contextual_symbol_id_preserved(self) -> None:
        """symbol_id must be correctly set on the returned Chunk."""
        mock_client = _make_mock_llm_client("Does stuff.")
        chunker = _make_contextual_chunker(contextual=True, llm_client=mock_client)
        symbol = _make_symbol(id=77)
        chunks = chunker.build_chunks([symbol], [], "src/foo.py", "python")
        assert chunks[0].symbol_id == 77

    def test_llm_called_with_correct_arguments(self) -> None:
        """Verify the LLM is invoked with model, messages, max_tokens and temperature."""
        mock_client = _make_mock_llm_client("A summary.")
        config = ChunkerConfig(
            contextual=True,
            contextual_model="gpt-4o-mini",
            contextual_max_tokens=100,
        )
        chunker = ContextualChunker(config, llm_client=mock_client)
        symbol = _make_symbol(body="def fn(): pass")
        chunker.build_chunks([symbol], [], "src/foo.py", "python")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs.kwargs["max_tokens"] == 100
        assert call_kwargs.kwargs["temperature"] == 0
