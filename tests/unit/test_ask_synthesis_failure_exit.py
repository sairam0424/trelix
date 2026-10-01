"""
`trelix ask` must not exit 0 when no answer was produced.

`Synthesizer.stream()` never raises: on failure it yields a banner token
("[trelix: synthesis unavailable - ...]"), and a key-less backend yields a
"not configured" placeholder token. `ask` printed both to stdout as if they were the
answer and exited 0, so `trelix ask . q > answer.md` stored an error banner and CI
saw success. The Synthesizer now exposes `last_error` / `is_configured` while its
token stream stays exactly as it was (REST /ask iterates the same generator).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from trelix.cli.main import app
from trelix.core.config import EmbedderConfig, LLMConfig, RetrievalConfig
from trelix.core.models import RetrievedContext
from trelix.llm.client import ChatMessage, ChatResponse, TrelixChatClient
from trelix.llm.providers.openai_backend import OpenAIBackend
from trelix.retrieval.synthesizer import Synthesizer

runner = CliRunner()

_BANNER = "synthesis unavailable"
_NOT_CONFIGURED = "LLM not configured"


def _context(*, total_tokens: int = 42) -> RetrievedContext:
    return RetrievedContext(
        query="how does add work",
        results=[MagicMock()],
        context_text="def add(a, b): return a + b",
        total_tokens=total_tokens,
        elapsed_seconds=0.01,
    )


def _keyless_backend() -> OpenAIBackend:
    keyless = LLMConfig(provider="openai", _env_file=None)  # type: ignore[call-arg]
    return OpenAIBackend(keyless.model_copy(update={"openai_api_key": None}))


def _failing_client() -> MagicMock:
    def _boom(**_kwargs: Any) -> Iterator[str]:
        yield "partial "
        raise RuntimeError("401 invalid api key")

    client = MagicMock()
    client.stream.side_effect = _boom
    return client


class _RaisingChatClient(TrelixChatClient):
    """A real TrelixChatClient (so Synthesizer.synthesize takes its normal path)."""

    def __init__(self, error: Exception) -> None:
        self._client = object()  # non-None: this backend "has credentials"
        self._error = error

    def complete(self, messages: list[ChatMessage], **_kwargs: Any) -> ChatResponse:
        raise self._error

    def stream(self, messages: list[ChatMessage], **_kwargs: Any) -> Iterator[str]:
        raise self._error

    def tool_call(self, *args: Any, **kwargs: Any) -> Any:
        raise self._error


def _ok_client() -> MagicMock:
    client = MagicMock()
    client.stream.side_effect = lambda **_kwargs: iter(["The ", "answer."])
    return client


# Above graph_rag_threshold_tokens (8000), so synthesize() delegates to GraphRAG map-reduce.
_LARGE_CONTEXT_TOKENS = 9000


def _invoke_ask(
    tmp_path: Path, client: Any, *, flare: bool = False, context: RetrievedContext | None = None
) -> Any:
    # `ask` refuses a repo with no index; Retriever is mocked, the index file is not.
    (tmp_path / ".trelix").mkdir(exist_ok=True)
    (tmp_path / ".trelix" / "index.db").touch()
    env = {"TRELIX_RETRIEVAL_FLARE": "true"} if flare else {}
    with (
        patch("trelix.retrieval.retriever.Retriever") as mock_retriever,
        patch("trelix.retrieval.synthesizer.build_chat_client", return_value=client),
        # GraphRAGSynthesizer builds its own chat client through the factory.
        patch("trelix.llm.factory.build_chat_client", return_value=client),
    ):
        mock_retriever.return_value.retrieve.return_value = context or _context()
        return runner.invoke(
            app, ["ask", str(tmp_path), "how does add work", "--provider", "openai"], env=env
        )


class TestAskExitsNonZeroOnFailure:
    def test_stream_failure_exits_1_with_stderr_message(self, tmp_path: Path) -> None:
        result = _invoke_ask(tmp_path, _failing_client())

        assert result.exit_code == 1, result.stdout
        assert "401 invalid api key" in result.stderr
        assert _BANNER not in result.stdout  # the banner is not an answer

    def test_unconfigured_llm_exits_1_before_printing_a_fake_answer(self, tmp_path: Path) -> None:
        result = _invoke_ask(tmp_path, _keyless_backend())

        assert result.exit_code == 1, result.stdout
        assert _NOT_CONFIGURED in result.stderr
        assert _NOT_CONFIGURED not in result.stdout

    def test_flare_synthesis_failure_exits_1(self, tmp_path: Path) -> None:
        result = _invoke_ask(
            tmp_path, _RaisingChatClient(RuntimeError("503 upstream down")), flare=True
        )

        assert result.exit_code == 1, result.stdout
        assert "503 upstream down" in result.stderr

    def test_flare_unconfigured_llm_exits_1(self, tmp_path: Path) -> None:
        result = _invoke_ask(tmp_path, _keyless_backend(), flare=True)

        assert result.exit_code == 1, result.stdout
        assert _NOT_CONFIGURED in result.stderr

    def test_flare_graphrag_synthesis_failure_exits_1(self, tmp_path: Path) -> None:
        """GraphRAG map-reduce swallows every LLM error and returns "": still a failure."""
        result = _invoke_ask(
            tmp_path,
            _RaisingChatClient(RuntimeError("401 invalid api key")),
            flare=True,
            context=_context(total_tokens=_LARGE_CONTEXT_TOKENS),
        )

        assert result.exit_code == 1, result.stdout
        assert "GraphRAG synthesis produced no answer" in result.stderr

    def test_successful_answer_is_unchanged(self, tmp_path: Path) -> None:
        result = _invoke_ask(tmp_path, _ok_client())

        assert result.exit_code == 0, result.stderr
        assert result.stdout.strip() == "The answer."
        assert result.stderr.strip() == ""


class TestSynthesizerStatusKeepsTheTokenContract:
    def _synth(self, client: Any) -> Synthesizer:
        with patch("trelix.retrieval.synthesizer.build_chat_client", return_value=client):
            return Synthesizer(EmbedderConfig(_env_file=None))  # type: ignore[call-arg]

    def test_failure_still_yields_banner_and_never_raises(self) -> None:
        synth = self._synth(_failing_client())

        tokens = list(synth.stream(_context(), RetrievalConfig()))

        assert tokens[0] == "partial "
        assert _BANNER in tokens[-1]
        assert synth.last_error is not None and "401 invalid api key" in synth.last_error

    def test_error_is_visible_before_the_banner_token_is_delivered(self) -> None:
        """The CLI relies on this to keep the banner off stdout."""
        synth = self._synth(_failing_client())
        seen: list[tuple[str, str | None]] = []

        for token in synth.stream(_context(), RetrievalConfig()):
            seen.append((token, synth.last_error))

        assert seen[0][1] is None
        assert _BANNER in seen[-1][0]
        assert seen[-1][1] is not None

    def test_success_leaves_no_error_and_resets_between_calls(self) -> None:
        synth = self._synth(_failing_client())
        list(synth.stream(_context(), RetrievalConfig()))
        assert synth.last_error is not None

        synth._llm_client = _ok_client()
        tokens = list(synth.stream(_context(), RetrievalConfig()))

        assert tokens == ["The ", "answer."]
        assert synth.last_error is None

    def test_keyless_backend_is_not_configured_and_still_streams_its_placeholder(self) -> None:
        synth = self._synth(_keyless_backend())

        assert synth.is_configured is False
        tokens = list(synth.stream(_context(), RetrievalConfig()))

        assert any(_NOT_CONFIGURED in t for t in tokens)
        assert synth.last_error is not None

    def test_configured_client_reports_configured(self) -> None:
        assert self._synth(_ok_client()).is_configured is True

    def test_graphrag_empty_answer_is_recorded_as_a_failure(self) -> None:
        synth = self._synth(_ok_client())

        with (
            patch("trelix.retrieval.graph_rag.GraphRAGSynthesizer.should_use", return_value=True),
            patch("trelix.retrieval.graph_rag.GraphRAGSynthesizer.synthesize", return_value=""),
        ):
            answer = synth.synthesize(_context(total_tokens=_LARGE_CONTEXT_TOKENS))

        assert answer == ""
        assert synth.last_error is not None

    def test_graphrag_answer_is_returned_without_error(self) -> None:
        synth = self._synth(_ok_client())

        with (
            patch("trelix.retrieval.graph_rag.GraphRAGSynthesizer.should_use", return_value=True),
            patch(
                "trelix.retrieval.graph_rag.GraphRAGSynthesizer.synthesize", return_value="mapped"
            ),
        ):
            answer = synth.synthesize(_context(total_tokens=_LARGE_CONTEXT_TOKENS))

        assert answer == "mapped"
        assert synth.last_error is None

    def test_synthesize_records_failure_too(self) -> None:
        synth = self._synth(_keyless_backend())

        assert synth.synthesize(_context()) == ""
        assert synth.last_error is not None
