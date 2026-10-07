"""`TRELIX_LLM_LOCAL_CONTEXT_TOKENS` is the context window the auto-derived budget uses.

With `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null` the Retriever sizes its context budget as
`int(window * TRELIX_RETRIEVAL_CONTEXT_WINDOW_FRACTION)`, where the window comes from
`context_windows.resolve_window(model)`. That table knows no local model tags (`qwen2.5-coder:7b`
carries a `:` no prefix matches), so every local model fell back to 12,000. The new field
supplies the window instead; the fraction and the fallback are unchanged. It is the window for
any tag, including one the table knows (docs/CONFIGURATION.md, the window table).

Built with the `Retriever(` shape of test_retriever_budget_and_ranking_knobs.py: every
collaborator the constructor touches is a plain double, the budget is read from
`_effective_budget` (the value the assembler receives), and every expected value is a literal.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from trelix.core.config import IndexConfig, LLMConfig, RetrievalConfig

_LOCAL_URL = "http://127.0.0.1:11434/v1"
_LOCAL_TAG = "qwen2.5-coder:7b"
_LOGGER = "trelix.retrieval"


class _Database:
    def get_embedding_dimension(self) -> None:
        return None


class _Embedder:
    dimension = 8


class _NoPlanner:
    def plan(self, query: str) -> None:  # pragma: no cover - never consulted here
        raise AssertionError("planner must not be consulted: only the budget is read")


def _build(
    repo_path: Path,
    *,
    local_context_tokens: int | None,
    fraction: float = 0.5,
    budget: int | None = None,
    model: str = _LOCAL_TAG,
) -> object:
    from trelix.retrieval.retriever import Retriever

    repo_path.mkdir(parents=True, exist_ok=True)  # IndexConfig validates existence
    with (
        patch("trelix.retrieval.retriever.Database", return_value=_Database()),
        patch("trelix.retrieval.retriever.make_embedder", return_value=_Embedder()),
        patch("trelix.retrieval.retriever.make_vector_store", return_value=object()),
        patch("trelix.retrieval.retriever.QueryPlanner", return_value=_NoPlanner()),
    ):
        return Retriever(
            IndexConfig(
                repo_path=str(repo_path),
                retrieval=RetrievalConfig(
                    context_token_budget=budget, context_window_fraction=fraction
                ),
                llm=LLMConfig(
                    model=model,
                    openai_api_key=None,
                    base_url=_LOCAL_URL,
                    local_context_tokens=local_context_tokens,
                ),
            )
        )


class TestTheLocalWindowSizesTheBudget:
    def test_32768_at_fraction_half_is_16384(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MUTATION: ignore `llm.local_context_tokens` in `_resolve_effective_budget` (the
        unknown tag falls back to 12,000); use the value without the fraction (32,768)."""
        caplog.set_level(logging.INFO, logger=_LOGGER)

        retriever = _build(tmp_path / "repo", local_context_tokens=32768)

        assert retriever._effective_budget == 16384  # type: ignore[attr-defined]
        assert "Context window 32768 from TRELIX_LLM_LOCAL_CONTEXT_TOKENS" in caplog.text
        assert "falling back to 12,000" not in caplog.text

    def test_8192_at_fraction_half_is_4096(self, tmp_path: Path) -> None:
        retriever = _build(tmp_path / "repo", local_context_tokens=8192)

        assert retriever._effective_budget == 4096  # type: ignore[attr-defined]

    def test_the_fraction_is_the_configured_one(self, tmp_path: Path) -> None:
        """MUTATION: multiply by a literal 0.5 instead of `cfg.context_window_fraction`."""
        retriever = _build(tmp_path / "repo", local_context_tokens=32768, fraction=0.25)

        assert retriever._effective_budget == 8192  # type: ignore[attr-defined]

    def test_the_variable_wins_for_a_tag_the_table_knows(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """docs/CONFIGURATION.md: `any tag with TRELIX_LLM_LOCAL_CONTEXT_TOKENS=32768` resolves to
        32,768 "from the variable". A gateway may expose a hosted model's name for a server that
        runs a smaller window, so the table's 128,000 for `gpt-4o` must not override the value.

        MUTATION: `window = resolve_window(model) or local` (the table wins: 64,000)."""
        caplog.set_level(logging.INFO, logger=_LOGGER)

        retriever = _build(tmp_path / "repo", local_context_tokens=32768, model="gpt-4o")

        assert retriever._effective_budget == 16384  # type: ignore[attr-defined]
        assert "Context window 32768 from TRELIX_LLM_LOCAL_CONTEXT_TOKENS" in caplog.text


class TestTheRestOfTheResolutionIsUnchanged:
    def test_unset_falls_back_to_12000_for_a_local_tag(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MUTATION: treat `None` as a window (TypeError, caught, still 12,000 but with the
        'Failed to resolve' line); replace the fallback with the fraction of something."""
        caplog.set_level(logging.INFO, logger=_LOGGER)

        retriever = _build(tmp_path / "repo", local_context_tokens=None)

        assert retriever._effective_budget == 12000  # type: ignore[attr-defined]
        assert "not recognized by context_windows" in caplog.text
        assert "TRELIX_LLM_LOCAL_CONTEXT_TOKENS" not in caplog.text

    def test_an_explicit_budget_still_wins(self, tmp_path: Path) -> None:
        """MUTATION: read the local window before the explicit-budget return."""
        retriever = _build(tmp_path / "repo", local_context_tokens=32768, budget=20000)

        assert retriever._effective_budget == 20000  # type: ignore[attr-defined]
