"""`TRELIX_LLM_BASE_URL` with a provider other than `openai` is ignored with one warning, never
an error.

Three retrieval shims (`QueryPlanner`, `GraphRAGSynthesizer`, `Synthesizer`) build an
`LLMConfig(provider=<the embedder's provider>, _env_file=None)` when no `llm_config` is passed,
and that config still reads `TRELIX_LLM_BASE_URL` from the environment. For an Azure-embedder
operator whose `TRELIX_LLM_PROVIDER=openai` + `TRELIX_LLM_BASE_URL` review configuration is
valid, a cross-field error in LLMConfig would have raised inside Retriever/Synthesizer
construction and broken `search`, `ask`, the REST `/ask` stream and `eval-synthesis`. So the
URL is read only by the plain openai branch of `OpenAIBackend`, and `build_chat_client` warns.
"""

from __future__ import annotations

import logging

import pytest
from openai import AzureOpenAI

from trelix.core.config import EmbedderConfig, LLMConfig, RetrievalConfig
from trelix.llm.factory import build_chat_client
from trelix.llm.providers.openai_backend import OpenAIBackend
from trelix.retrieval.graph_rag import GraphRAGSynthesizer
from trelix.retrieval.planner.agent import QueryPlanner
from trelix.retrieval.synthesizer import Synthesizer

_LOCAL_URL = "http://127.0.0.1:11434/v1"
_FAKE_KEY = "test-k"  # short enough not to trigger the secret scanner; never sent anywhere real
_AZURE_ENDPOINT = "https://test.openai.azure.com/"
_FACTORY_LOGGER = "trelix.llm.factory"
_AZURE_WARNING = "TRELIX_LLM_BASE_URL is set but TRELIX_LLM_PROVIDER=azure does not use it"


def _factory_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _FACTORY_LOGGER]


def _azure_llm(**extra: object) -> LLMConfig:
    return LLMConfig(  # type: ignore[call-arg]
        provider="azure",
        azure_api_key=_FAKE_KEY,
        azure_endpoint=_AZURE_ENDPOINT,
        _env_file=None,
        **extra,
    )


def _assert_azure_backend_ignoring_the_url(client: object) -> None:
    assert isinstance(client, OpenAIBackend)
    assert client._local_server is False
    assert isinstance(client._client, AzureOpenAI)
    assert "127.0.0.1" not in str(client._client.base_url)
    # The Azure request shape is untouched: the deployment name is not a legacy model name.
    assert client._token_limit(100) == {"max_completion_tokens": 100}


class TestFactoryWarning:
    def test_azure_with_the_url_builds_an_azure_client_and_warns_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MUTATION: honour `base_url` in the Azure branch (`_local_server` is True); drop the
        factory warning (no record)."""
        with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
            client = build_chat_client(_azure_llm(base_url=_LOCAL_URL))

        _assert_azure_backend_ignoring_the_url(client)
        assert _factory_warnings(caplog) == [_AZURE_WARNING]

    def test_anthropic_with_the_url_warns_with_its_own_provider_name(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """MUTATION: compare the provider to `"azure"` instead of `!= "openai"`."""
        pytest.importorskip("anthropic")
        config = LLMConfig(  # type: ignore[call-arg]
            provider="anthropic", anthropic_api_key=_FAKE_KEY, base_url=_LOCAL_URL, _env_file=None
        )

        with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
            build_chat_client(config)

        assert _factory_warnings(caplog) == [
            "TRELIX_LLM_BASE_URL is set but TRELIX_LLM_PROVIDER=anthropic does not use it"
        ]

    def test_openai_with_the_url_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        config = LLMConfig(  # type: ignore[call-arg]
            provider="openai", base_url=_LOCAL_URL, model="gpt-oss:20b", _env_file=None
        )

        with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
            client = build_chat_client(config)

        assert isinstance(client, OpenAIBackend)
        assert client._local_server is True
        assert _factory_warnings(caplog) == []

    def test_azure_without_the_url_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
            client = build_chat_client(_azure_llm())

        assert isinstance(client, OpenAIBackend)
        assert _factory_warnings(caplog) == []


class TestTheThreeShims:
    """Each shim constructs with the URL in the environment and an Azure embedder."""

    @pytest.fixture
    def azure_embedder(self, monkeypatch: pytest.MonkeyPatch) -> EmbedderConfig:
        monkeypatch.setenv("TRELIX_LLM_BASE_URL", _LOCAL_URL)
        embedder = EmbedderConfig(provider="azure", _env_file=None)  # type: ignore[call-arg]
        return embedder.model_copy(
            update={"azure_api_key": _FAKE_KEY, "azure_endpoint": _AZURE_ENDPOINT}
        )

    def test_query_planner(
        self, azure_embedder: EmbedderConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
            planner = QueryPlanner(azure_embedder)

        _assert_azure_backend_ignoring_the_url(planner._llm_client)
        assert _factory_warnings(caplog) == [_AZURE_WARNING]

    def test_graph_rag_synthesizer(
        self, azure_embedder: EmbedderConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
            synthesizer = GraphRAGSynthesizer(azure_embedder, RetrievalConfig())

        _assert_azure_backend_ignoring_the_url(synthesizer._llm_client)
        assert _factory_warnings(caplog) == [_AZURE_WARNING]

    def test_synthesizer(
        self, azure_embedder: EmbedderConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_FACTORY_LOGGER):
            synthesizer = Synthesizer(azure_embedder)

        # The shim did read the URL from the environment; the backend just does not use it.
        assert synthesizer._llm_config.base_url == _LOCAL_URL
        _assert_azure_backend_ignoring_the_url(synthesizer._llm_client)
        assert _factory_warnings(caplog) == [_AZURE_WARNING]
