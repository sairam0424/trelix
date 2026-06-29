"""Tests for LLM-as-judge eval scorer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestLLMJudge:
    def test_importable(self) -> None:
        from tests.eval.llm_judge import LLMJudge

        assert LLMJudge is not None

    def test_score_returns_float_between_0_and_1(self) -> None:
        from tests.eval.llm_judge import LLMJudge

        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content='{"score": 0.8, "reason": "relevant"}'
        )
        judge = LLMJudge(mock_client)
        score = judge.score(
            query="how does authentication work",
            retrieved_snippets=["def authenticate(user): ..."],
            expected_file="src/auth.py",
        )
        assert 0.0 <= score <= 1.0

    def test_score_returns_0_on_llm_failure(self) -> None:
        from tests.eval.llm_judge import LLMJudge

        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("API error")
        judge = LLMJudge(mock_client)
        score = judge.score("query", ["snippet"], "expected.py")
        assert score == 0.0

    def test_score_parses_json_response(self) -> None:
        from tests.eval.llm_judge import LLMJudge

        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content='{"score": 0.9, "reason": "exact file match, highly relevant"}'
        )
        judge = LLMJudge(mock_client)
        score = judge.score("how does auth work", ["def login(): ..."], "auth.py")
        assert score == pytest.approx(0.9)

    def test_score_clamps_out_of_range_values(self) -> None:
        from tests.eval.llm_judge import LLMJudge

        mock_client = MagicMock()
        mock_client.complete.return_value = MagicMock(
            content='{"score": 1.5, "reason": "extra good"}'
        )
        judge = LLMJudge(mock_client)
        score = judge.score("query", ["snippet"], "file.py")
        assert score <= 1.0
