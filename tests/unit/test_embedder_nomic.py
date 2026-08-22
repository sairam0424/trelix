"""Tests for NomicCodeEmbedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trelix.core.config import EmbedderConfig
from trelix.embedder.base import REMOTE_MODEL_CODE_ENV_VAR


class TestNomicCodeEmbedder:
    @pytest.fixture(autouse=True)
    def _remote_model_code_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Grant SEC-03c's opt-in for this class only.

        nomic-code loads CodeRankEmbed with remote code trusted, which the gate in
        embedder/base.py refuses unless the operator set the variable in the real
        process environment. Scoped to this class deliberately: in conftest.py it
        would blind the whole suite to the gate regressing. The refusal side lives
        in tests/unit/test_remote_model_code_gate.py.
        """
        monkeypatch.setenv(REMOTE_MODEL_CODE_ENV_VAR, "1")

    def test_importable(self) -> None:
        from trelix.embedder.nomic_code import NomicCodeEmbedder

        assert NomicCodeEmbedder is not None

    def test_is_base_embedder(self) -> None:
        from trelix.embedder.base import BaseEmbedder
        from trelix.embedder.nomic_code import NomicCodeEmbedder

        assert issubclass(NomicCodeEmbedder, BaseEmbedder)

    # ── The published protocol ────────────────────────────────────────────────────────
    #
    # These replace two tests that asserted `called_texts[0].startswith(_DOC_PREFIX)` and
    # the query equivalent. Both derived their expected value FROM the constant they were
    # pinning, so they passed for ANY prefix value — including the
    # nomic-embed-text-v1.5 pair that CodeRankEmbed was never trained on, which is exactly
    # what they were guarding while it shipped. A test whose oracle is the code under test
    # cannot detect a wrong protocol; these assert against what the model PUBLISHES.

    def test_documents_are_encoded_with_no_prompt_at_all(self) -> None:
        """CodeRankEmbed publishes a query prompt only; documents take nothing.

        MUTATION: prepend any string to the texts in `embed`, or pass prompt/prompt_name
        there, and this fails.
        """
        from trelix.embedder.nomic_code import NomicCodeEmbedder

        seen: dict[str, object] = {}

        class _Recorder:
            """Plain class, not MagicMock: a mock would answer to any kwarg name asked of
            it and could not tell an absent prompt from a misspelled one."""

            def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
                seen["texts"] = list(texts)
                seen["kwargs"] = dict(kwargs)
                return [[0.1] * 768 for _ in texts]

        with patch("trelix.embedder.nomic_code.SentenceTransformer", return_value=_Recorder()):
            cfg = EmbedderConfig(provider="nomic-code", _env_file=None)
            NomicCodeEmbedder(cfg).embed(["def foo(): pass"])

        assert seen["texts"] == ["def foo(): pass"], (
            f"the document text was modified before encoding: {seen['texts']!r}. "
            "CodeRankEmbed publishes no document prefix."
        )
        kwargs = seen["kwargs"]
        assert isinstance(kwargs, dict)
        assert "prompt" not in kwargs and "prompt_name" not in kwargs, (
            f"a prompt was supplied on the DOCUMENT path: {kwargs!r}"
        )

    def test_queries_use_the_prompt_name_the_model_publishes(self) -> None:
        """The instruction must be looked up in the model's config, not hardcoded here.

        Asserting `prompt_name` rather than a literal string is the point: it is what makes
        the protocol travel with the weights, so a model swap cannot leave this provider
        asserting a prompt its model does not declare.

        MUTATION: drop `prompt_name=` from `embed_query`, or replace it with a literal
        `prompt=<text>`, and this fails.
        """
        from trelix.embedder.nomic_code import _QUERY_PROMPT_NAME, NomicCodeEmbedder

        seen: dict[str, object] = {}

        class _Recorder:
            def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
                seen["texts"] = list(texts)
                seen["kwargs"] = dict(kwargs)
                return [[0.1] * 768 for _ in texts]

        with patch("trelix.embedder.nomic_code.SentenceTransformer", return_value=_Recorder()):
            cfg = EmbedderConfig(provider="nomic-code", _env_file=None)
            NomicCodeEmbedder(cfg).embed_query("authentication logic")

        kwargs = seen["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs.get("prompt_name") == _QUERY_PROMPT_NAME, (
            f"the query path did not ask for the published prompt by name: {kwargs!r}"
        )
        assert seen["texts"] == ["authentication logic"], (
            f"the query text was modified before encoding: {seen['texts']!r}. The prompt "
            "must come from the model's config via prompt_name, not from a manual prefix."
        )

    def test_the_encode_path_does_not_hardcode_the_prompt_text(self) -> None:
        """Structural guard: the prompt string must not reappear in the call path.

        The recorded expectation is allowed to exist as a snapshot for drift detection; what
        must not happen is the encode path reading it. This is the guard that would have
        caught the original defect, where a prompt constant was applied by a provider whose
        model never published it — and it is the same defect currently live in bge_code.py.
        """
        import inspect

        from trelix.embedder import nomic_code

        expected = nomic_code._EXPECTED_PUBLISHED_QUERY_PROMPT
        for fn in (nomic_code.NomicCodeEmbedder.embed, nomic_code.NomicCodeEmbedder.embed_query):
            body = inspect.getsource(fn)
            assert expected not in body, (
                f"{fn.__qualname__} hardcodes the published prompt text. Pass "
                "prompt_name= and let sentence-transformers read the model's own config."
            )

    def test_the_recorded_published_prompt_is_codrankembeds(self) -> None:
        """Snapshot of nomic-ai/CodeRankEmbed's config_sentence_transformers.json.

        `prompts: {"query": "Represent this query for searching relevant code: "}`, fetched
        2026-08-22. Recorded so that a careless edit to the constant fails here rather than
        silently changing what a future reader believes the model publishes. Note the
        trailing space — it is part of the published string.
        """
        from trelix.embedder.nomic_code import _EXPECTED_PUBLISHED_QUERY_PROMPT, _QUERY_PROMPT_NAME

        assert _QUERY_PROMPT_NAME == "query"
        assert (
            _EXPECTED_PUBLISHED_QUERY_PROMPT == "Represent this query for searching relevant code: "
        )

    def test_batch_size_comes_from_config_not_a_hardcoded_32(self) -> None:
        """TRELIX_EMBEDDER_BATCH_SIZE was silently ignored by this provider.

        MUTATION: restore `batch_size=32` in `embed` and this fails.
        """
        from trelix.embedder.nomic_code import NomicCodeEmbedder

        seen: dict[str, object] = {}

        class _Recorder:
            def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
                seen["kwargs"] = dict(kwargs)
                return [[0.1] * 768 for _ in texts]

        with patch("trelix.embedder.nomic_code.SentenceTransformer", return_value=_Recorder()):
            cfg = EmbedderConfig(provider="nomic-code", batch_size=7, _env_file=None)
            NomicCodeEmbedder(cfg).embed(["a", "b"])

        kwargs = seen["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs.get("batch_size") == 7, (
            f"embed() ignored the configured batch size: {kwargs!r}"
        )

    def test_dimension(self) -> None:
        from trelix.embedder.nomic_code import NomicCodeEmbedder

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = [[0.1] * 768]
        with patch("trelix.embedder.nomic_code.SentenceTransformer", return_value=mock_model):
            cfg = EmbedderConfig(provider="nomic-code", nomic_code_dimensions=768, _env_file=None)
            emb = NomicCodeEmbedder(cfg)
            assert emb.dimension == 768

    def test_config_effective_dimension(self) -> None:
        cfg = EmbedderConfig(provider="nomic-code", nomic_code_dimensions=768, _env_file=None)
        assert cfg.effective_dimension == 768
