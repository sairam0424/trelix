"""Tests for FLARE-style confidence-gated re-retrieval."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from trelix.retrieval.flare import FLARELoop, _contains_uncertainty


class TestUncertaintyDetection:
    def test_detects_i_dont_know(self) -> None:
        assert _contains_uncertainty("I don't know how this works.") is True

    def test_detects_cannot_find(self) -> None:
        assert _contains_uncertainty("I cannot find any relevant code for this.") is True

    def test_detects_no_information(self) -> None:
        assert _contains_uncertainty("There is no information about JWT in the codebase.") is True

    def test_confident_answer_not_flagged(self) -> None:
        assert (
            _contains_uncertainty("The authenticate() function in auth.py handles this.") is False
        )

    def test_case_insensitive(self) -> None:
        assert _contains_uncertainty("NO INFORMATION available") is True


class TestFLARELoop:
    def _make_loop(self, first_answer: str, second_answer: str = "Found it.") -> tuple:
        mock_retriever = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.query = "how does auth work"
        mock_retriever.retrieve.return_value = mock_ctx

        mock_synthesizer = MagicMock()
        mock_synthesizer.synthesize.side_effect = [first_answer, second_answer]

        mock_config = MagicMock()
        mock_config.retrieval.flare_enabled = True
        mock_config.retrieval.flare_max_iterations = 1

        loop = FLARELoop(mock_retriever, mock_synthesizer, mock_config)
        return loop, mock_retriever, mock_synthesizer

    def test_no_retry_when_confident(self) -> None:
        loop, mock_retriever, mock_synthesizer = self._make_loop(
            "The auth.py file handles JWT in the login() function."
        )
        result = loop.run("how does auth work")
        assert mock_synthesizer.synthesize.call_count == 1
        assert "auth.py" in result

    def test_retries_once_on_uncertainty(self) -> None:
        loop, mock_retriever, mock_synthesizer = self._make_loop(
            first_answer="I cannot find any relevant code for this.",
            second_answer="The login() function in auth.py handles JWT.",
        )
        result = loop.run("how does auth work")
        assert mock_synthesizer.synthesize.call_count == 2
        assert "login()" in result

    def test_stops_after_max_iterations(self) -> None:
        mock_retriever = MagicMock()
        mock_ctx = MagicMock()
        mock_retriever.retrieve.return_value = mock_ctx
        mock_synthesizer = MagicMock()
        # All answers uncertain
        mock_synthesizer.synthesize.return_value = "I don't know."
        mock_config = MagicMock()
        mock_config.retrieval.flare_enabled = True
        mock_config.retrieval.flare_max_iterations = 2
        loop = FLARELoop(mock_retriever, mock_synthesizer, mock_config)
        loop.run("how does auth work")
        # max_iterations=2 means at most 2 synthesis calls (1 initial + 1 retry)
        assert mock_synthesizer.synthesize.call_count <= 2

    def test_config_defaults(self, tmp_path: Path) -> None:
        from trelix.core.config import IndexConfig

        cfg = IndexConfig(repo_path=str(tmp_path))
        assert cfg.retrieval.flare_enabled is False
        assert cfg.retrieval.flare_max_iterations == 1
