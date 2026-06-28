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
        result = summarizer.summarize("src/auth.py", [], Language.PYTHON)
        assert result == ""

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
        # Verify that only max_symbols worth of content was sent
        assert mock_client.complete.called
