"""
`trelix ask` must not exit 0 when no answer was produced.

`Synthesizer.stream()` never raises: on failure it yields a banner token
("[trelix: synthesis unavailable - ...]"), and a key-less backend yields a
"not configured" placeholder token. `ask` printed both to stdout as if they were the
answer and exited 0, so `trelix ask . q > answer.md` stored an error banner and CI
saw success. The Synthesizer now exposes `last_error` / `is_configured` while its
token stream stays exactly as it was (REST /ask iterates the same generator).

The second half of the file pins the failure that raises nothing: an endpoint that answers
HTTP 200 with no answer in it (an HTML page behind a wrong OPENAI_BASE_URL, a proxy that
ignores stream=True, a model that streams only whitespace). That is recorded in `last_error`
as well, `ask` exits 1 on it, and REST /ask stays unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from typer.testing import CliRunner

from tests.fixtures.indexed import mark_indexed
from trelix.api.app import create_app
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


# ---------------------------------------------------------------------------
# An endpoint that answers 200 but carries no answer
# ---------------------------------------------------------------------------
#
# The failures above all raise or return an error. These do not: a wrong
# OPENAI_BASE_URL (an HTML page behind a 200), a proxy that ignores stream=True, or a
# model that streams only empty content. The openai SDK hands back a Stream that yields
# nothing, nothing was recorded, and `trelix ask` exited 0 with an empty stdout.

_EMPTY_ANSWER = (
    "The LLM returned no answer: the endpoint may not be an OpenAI-compatible chat API, "
    "OPENAI_BASE_URL (or a proxy in front of it) may be wrong, or the model returned nothing."
)
_EMPTY_ANSWER_HEAD = "The LLM returned no answer"

_SSE_CHUNK = (
    b'data: {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m", '
    b'"choices": [{"index": 0, "delta": {"content": "%s"}, "finish_reason": null}]}\n\n'
)
_SSE_END = b"data: [DONE]\n\n"

# (id, HTTP body, content type) for replies that are not an answer. A JSON body is what a
# proxy returns when it ignores stream=True; the SDK finds no SSE events in it.
_NO_ANSWER_REPLIES = [
    pytest.param(b"<html>not json at all", "text/html", id="html-page"),
    pytest.param(b'{"choices": []}', "application/json", id="json-ignoring-stream"),
    pytest.param(_SSE_CHUNK % b"" + _SSE_END, "text/event-stream", id="empty-sse-content"),
    pytest.param(_SSE_CHUNK % b"  " + _SSE_END, "text/event-stream", id="blank-sse-content"),
    pytest.param(b"", "text/event-stream", id="empty-body"),
]


class _ScriptedChatClient(TrelixChatClient):
    """A real TrelixChatClient (so Synthesizer.synthesize takes its normal path)."""

    def __init__(self, tokens: tuple[str, ...], *, configured: bool = True) -> None:
        self._client = object() if configured else None
        self._tokens = tokens

    def complete(self, messages: list[ChatMessage], **_kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    def stream(self, messages: list[ChatMessage], **_kwargs: Any) -> Iterator[str]:
        return iter(self._tokens)

    def tool_call(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _sdk_backend(body: bytes, content_type: str) -> OpenAIBackend:
    """A real OpenAIBackend over the real openai SDK; only the HTTP transport is faked.

    This is what makes the garbage-body tests meaningful: the SDK itself decides what
    an HTML page behind a 200 turns into, and a hand-made fake stream would only
    restate our assumption about it.
    """

    def _respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": content_type}, content=body)

    config = LLMConfig(provider="openai", _env_file=None)  # type: ignore[call-arg]
    backend = OpenAIBackend(config.model_copy(update={"openai_api_key": "test-k"}))
    backend._client = OpenAI(
        api_key="test-k",
        max_retries=0,
        base_url="http://llm.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_respond)),
    )
    return backend


def _one_line(text: str) -> str:
    """Undo Rich's wrapping so a long message can be matched as one sentence."""
    return " ".join(text.split())


class TestAskExitsNonZeroOnAnEmptyAnswer:
    @pytest.mark.parametrize("flare", [False, True], ids=["plain", "flare"])
    @pytest.mark.parametrize(("body", "content_type"), _NO_ANSWER_REPLIES)
    def test_reply_without_an_answer_exits_1_with_the_reason(
        self, tmp_path: Path, body: bytes, content_type: str, flare: bool
    ) -> None:
        result = _invoke_ask(tmp_path, _sdk_backend(body, content_type), flare=flare)

        assert result.exit_code == 1, result.stdout
        assert f"Synthesis failed: {_EMPTY_ANSWER}" in _one_line(result.stderr)
        assert result.stdout.strip() == ""  # the message is not an answer

    @pytest.mark.parametrize("flare", [False, True], ids=["plain", "flare"])
    @pytest.mark.parametrize(
        "tokens", [(), ("",), ("  ", "\n")], ids=["no-tokens", "empty-token", "whitespace"]
    )
    def test_stream_without_text_exits_1_and_keeps_the_message_off_stdout(
        self, tmp_path: Path, tokens: tuple[str, ...], flare: bool
    ) -> None:
        result = _invoke_ask(tmp_path, _ScriptedChatClient(tokens), flare=flare)

        assert result.exit_code == 1, result.stdout
        assert f"Synthesis failed: {_EMPTY_ANSWER}" in _one_line(result.stderr)
        assert _EMPTY_ANSWER_HEAD not in result.stdout
        assert result.stdout.strip() == ""

    @pytest.mark.parametrize("flare", [False, True], ids=["plain", "flare"])
    def test_a_real_sse_answer_still_exits_0_and_prints_it(
        self, tmp_path: Path, flare: bool
    ) -> None:
        """Control: the SDK-backed harness above can succeed, so the failures are real."""
        body = _SSE_CHUNK % b"The " + _SSE_CHUNK % b"answer." + _SSE_END
        result = _invoke_ask(tmp_path, _sdk_backend(body, "text/event-stream"), flare=flare)

        assert result.exit_code == 0, result.stderr
        assert result.stdout.strip() == "The answer."
        assert _EMPTY_ANSWER_HEAD not in result.stderr

    @pytest.mark.parametrize("flare", [False, True], ids=["plain", "flare"])
    @pytest.mark.parametrize(
        "tokens",
        [("\n", "The answer."), ("The answer.", "\n"), ("The ", "\n", "answer.", "  ")],
        ids=["leading-blank", "trailing-blank", "blank-in-the-middle"],
    )
    def test_an_answer_with_blank_tokens_around_text_is_still_an_answer(
        self, tmp_path: Path, tokens: tuple[str, ...], flare: bool
    ) -> None:
        result = _invoke_ask(tmp_path, _ScriptedChatClient(tokens), flare=flare)

        assert result.exit_code == 0, result.stderr
        assert result.stdout.split() == "".join(tokens).split()
        assert result.stderr.strip() == ""


class TestSynthesizerRecordsAnEmptyAnswer:
    def _synth(self, client: Any) -> Synthesizer:
        with patch("trelix.retrieval.synthesizer.build_chat_client", return_value=client):
            return Synthesizer(EmbedderConfig(_env_file=None))  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "tokens", [(), ("",), ("  ", "\n")], ids=["no-tokens", "empty-token", "whitespace"]
    )
    def test_stream_records_the_failure_and_never_raises(self, tokens: tuple[str, ...]) -> None:
        synth = self._synth(_ScriptedChatClient(tokens))

        yielded = list(synth.stream(_context(), RetrievalConfig()))

        assert yielded == list(tokens)  # no banner token: the caller reads last_error
        assert synth.last_error == _EMPTY_ANSWER

    @pytest.mark.parametrize(
        "tokens",
        [("\n", "ok"), ("ok", "\n"), ("o", "\n", "k", "  ")],
        ids=["leading-blank", "trailing-blank", "blank-in-the-middle"],
    )
    def test_stream_text_next_to_blank_tokens_is_an_answer(self, tokens: tuple[str, ...]) -> None:
        synth = self._synth(_ScriptedChatClient(tokens))

        assert list(synth.stream(_context(), RetrievalConfig())) == list(tokens)
        assert synth.last_error is None

    def test_stream_through_the_real_sdk_records_an_html_reply(self) -> None:
        synth = self._synth(_sdk_backend(b"<html>not json at all", "text/html"))

        assert list(synth.stream(_context(), RetrievalConfig())) == []
        assert synth.last_error == _EMPTY_ANSWER

    def test_an_unconfigured_backend_keeps_its_own_more_specific_message(self) -> None:
        synth = self._synth(_ScriptedChatClient((), configured=False))

        assert list(synth.stream(_context(), RetrievalConfig())) == []
        assert synth.last_error is not None
        assert _NOT_CONFIGURED in synth.last_error
        assert _EMPTY_ANSWER_HEAD not in synth.last_error

    def test_a_raised_error_is_not_replaced_by_the_empty_answer_message(self) -> None:
        synth = self._synth(_failing_client())

        list(synth.stream(_context(), RetrievalConfig()))

        assert synth.last_error == "401 invalid api key"

    @pytest.mark.parametrize(
        ("tokens", "expected"), [((), ""), (("  ", "\n"), "  \n")], ids=["none", "whitespace"]
    )
    def test_synthesize_returns_the_empty_answer_and_records_the_failure(
        self, tokens: tuple[str, ...], expected: str
    ) -> None:
        synth = self._synth(_ScriptedChatClient(tokens))

        assert synth.synthesize(_context()) == expected
        assert synth.last_error == _EMPTY_ANSWER

    def test_synthesize_answer_leaves_no_error(self) -> None:
        synth = self._synth(_ScriptedChatClient(("The ", "answer.")))

        assert synth.synthesize(_context()) == "The answer."
        assert synth.last_error is None

    def test_synthesize_clears_a_failure_from_the_previous_call(self) -> None:
        client = _ScriptedChatClient(())
        synth = self._synth(client)
        synth.synthesize(_context())
        assert synth.last_error == _EMPTY_ANSWER

        client._tokens = ("ok",)

        assert synth.synthesize(_context()) == "ok"
        assert synth.last_error is None

    def test_an_empty_retrieval_is_not_an_llm_failure(self) -> None:
        """No results means the LLM is never called: that is not an empty LLM answer."""
        synth = self._synth(_ScriptedChatClient(()))
        no_results = RetrievedContext(
            query="how does add work",
            results=[],
            context_text="",
            total_tokens=0,
            elapsed_seconds=0.01,
        )

        answer = synth.synthesize(no_results)

        assert "No relevant code found" in answer
        assert synth.last_error is None

    def test_an_empty_retrieval_is_not_an_llm_failure_for_stream_either(self) -> None:
        """The `stream()` twin: the notice is the whole stream, and it is not an empty answer
        (the client is never called, so there is no LLM answer to be empty)."""
        synth = self._synth(_ScriptedChatClient(()))
        no_results = RetrievedContext(
            query="how does add work",
            results=[],
            context_text="",
            total_tokens=0,
            elapsed_seconds=0.01,
        )

        tokens = list(synth.stream(no_results, RetrievalConfig()))

        assert len(tokens) == 1 and "No relevant code found" in tokens[0]
        assert synth.last_error is None


class TestRestAskWithAnEmptyAnswer:
    def test_the_sse_stream_still_ends_with_done_and_does_not_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Consumer audit pin: GET /ask iterates Synthesizer.stream() and never reads
        last_error (for any failure, not only this one), so an empty answer stays a bare
        `[DONE]`. Whether /ask should answer with an `[ERROR: ...]` frame is a separate
        decision; this only proves the new recording does not break the stream."""
        monkeypatch.setenv("TRELIX_ALLOWED_REPO_ROOTS", str(tmp_path))
        mark_indexed(tmp_path)
        with (
            patch("trelix.api.app.Retriever") as mock_retriever,
            patch(
                "trelix.retrieval.synthesizer.build_chat_client",
                return_value=_ScriptedChatClient(()),
            ),
        ):
            mock_retriever.return_value.retrieve.return_value = _context()
            resp = TestClient(create_app()).get(f"/ask?query=how+does+add+work&repo={tmp_path}")

        assert resp.status_code == 200
        assert resp.text == "data: [DONE]\n\n"
