"""Tests for file-level summary generator."""

from __future__ import annotations

from unittest.mock import MagicMock

from trelix.core.models import Language, Symbol, SymbolKind


class TestFileSummarizer:
    def test_importable(self) -> None:
        from trelix.indexing.file_summarizer import FileSummarizer

        assert FileSummarizer is not None

    def test_summarize_returns_non_empty_string(self) -> None:
        from trelix.indexing.file_summarizer import FileSummarizer

        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content=(
                "This file implements authentication logic including login,"
                " logout, and session management."
            )
        )
        summarizer = FileSummarizer(client=mock_client, max_symbols=20)
        symbols = [
            Symbol(
                file_id=1,
                name="login",
                qualified_name="login",
                kind=SymbolKind.FUNCTION,
                line_start=1,
                line_end=10,
                signature="def login(user, pwd)",
                body="def login(u, p): pass",
            )
        ]
        result = summarizer.summarize("src/auth.py", symbols, Language.PYTHON)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_returns_empty_on_llm_failure(self) -> None:
        from trelix.indexing.file_summarizer import FileSummarizer

        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("API error")
        summarizer = FileSummarizer(client=mock_client)
        # Pass at least one symbol so the guard `if not symbols` is bypassed
        # and the LLM call is actually attempted — this verifies the crash-safe
        # contract: summarize() returns "" on any LLM failure, never raises.
        symbols = [
            Symbol(
                file_id=1,
                name="login",
                qualified_name="login",
                kind=SymbolKind.FUNCTION,
                line_start=1,
                line_end=10,
                signature="def login(user, pwd)",
                body="def login(u, p): pass",
            )
        ]
        result = summarizer.summarize("src/auth.py", symbols, Language.PYTHON)
        assert result == ""
        assert mock_client.complete.called

    def test_summarize_truncates_to_max_symbols(self) -> None:
        from trelix.indexing.file_summarizer import FileSummarizer

        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(content="Summary")
        summarizer = FileSummarizer(client=mock_client, max_symbols=2)
        symbols = [
            Symbol(
                file_id=1,
                name=f"fn{i}",
                qualified_name=f"fn{i}",
                kind=SymbolKind.FUNCTION,
                line_start=i * 10,
                line_end=i * 10 + 5,
                signature=f"def fn{i}()",
                body=f"def fn{i}(): pass",
            )
            for i in range(10)
        ]
        summarizer.summarize("src/big.py", symbols, Language.PYTHON)
        assert mock_client.complete.called
        # Inspect the prompt sent to complete() and verify that symbols beyond
        # max_symbols=2 are absent.  A regression that removes the
        # `symbols[:self._max_symbols]` slice would include 'fn2'..'fn9' and
        # fail this assertion.
        call_kwargs = mock_client.complete.call_args[1]
        messages = call_kwargs.get("messages") or mock_client.complete.call_args[0][0]
        # messages is a list of ChatMessage objects; grab the user message content
        prompt_text = messages[0].content
        assert "fn0" in prompt_text, "first symbol should be in truncated prompt"
        assert "fn1" in prompt_text, "second symbol should be in truncated prompt"
        assert "fn2" not in prompt_text, "symbol beyond max_symbols=2 must not be in prompt"
        assert "fn9" not in prompt_text, "symbol beyond max_symbols=2 must not be in prompt"
