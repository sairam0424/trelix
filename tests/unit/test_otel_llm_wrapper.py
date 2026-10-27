"""
TracedChatClient's contracts that need no OpenTelemetry install (roadmap C-8 PR 3).

- R6: the off path returns the bare backend, type-identical, without importing opentelemetry;
  OTel on with the library unimportable returns the bare backend and warns ONCE per process
- R8 (critique C1): `_client` is the backend's, by identity, and nothing else is forwarded;
  through the synthesizer, the TrelixChatClient path is taken and `is_configured` follows the
  backend's client
- C9: `request_model` reads the backend's `_model` (the Azure deployment, the Anthropic model)
- R4(d): `seed_kwargs()` returns {} for every wrapper method
- The provider table (D7), and the span lifecycle and the trelix content gate as applied to the
  invocation, against a recording fake handler (critique C2: what trelix ASSIGNED, not what the
  library later chose to emit)

Real spans are in test_otel_llm_spans.py; the shared fakes in tests/unit/otel_llm_fakes.py. The
few tests that need util-genai's types import them lazily behind the 1.2b0 floor; everything
else runs in the default CI job.
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.unit.otel_llm_fakes import (
    FakeBackend,
    RecordingHandler,
    cfg,
    reply,
    require_util_genai_at_the_floor,
)
from trelix.llm.client import ChatMessage, TrelixChatClient

_UNAVAILABLE_PREFIX = "TRELIX_OTEL_ENABLED is set but OpenTelemetry GenAI spans are unavailable ("
_UNAVAILABLE_SUFFIX = ") — LLM calls will NOT be traced. Install: pip install 'trelix[otel]'"


def _wrapper(backend: TrelixChatClient, handler: RecordingHandler, config: Any = None) -> Any:
    from trelix.llm.otel import TracedChatClient

    return TracedChatClient(backend, config if config is not None else cfg(), handler)


def _otel_tracing() -> Any:
    return importlib.import_module("trelix.retrieval.otel_tracing")


@pytest.fixture()
def capture_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRELIX_OTEL_CAPTURE_CONTENT", "false")
    monkeypatch.setattr(_otel_tracing(), "_env_otel_settings", None)
    monkeypatch.setattr(_otel_tracing(), "_capture_inert_warned", False)


@pytest.fixture()
def capture_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRELIX_OTEL_CAPTURE_CONTENT", "true")
    monkeypatch.setattr(_otel_tracing(), "_env_otel_settings", None)
    monkeypatch.setattr(_otel_tracing(), "_capture_inert_warned", False)


# ---------------------------------------------------------------------------
# R6 -- the off path and the library-missing path of the factory
# ---------------------------------------------------------------------------


class TestFactoryPaths:
    def test_flag_off_returns_the_bare_backend_without_importing_opentelemetry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R6. MUTATION: wrap unconditionally -> type name TracedChatClient."""
        from trelix.llm.factory import build_chat_client

        monkeypatch.setenv("TRELIX_OTEL_ENABLED", "false")
        monkeypatch.setattr(_otel_tracing(), "_env_otel_settings", None)
        purged = {k: v for k, v in sys.modules.items() if k.startswith("opentelemetry")}
        for key in purged:
            del sys.modules[key]
        try:
            client = build_chat_client(cfg(provider="openai", model="gpt-4o"))
            assert type(client).__name__ == "OpenAIBackend"
            assert not any(k.startswith("opentelemetry") for k in sys.modules)
        finally:
            sys.modules.update(purged)

    def test_importing_the_wrapper_module_does_not_import_opentelemetry(self) -> None:
        """MUTATION: import util-genai at the module top of trelix/llm/otel.py."""
        import trelix.llm as llm_package

        purged = {k: v for k, v in sys.modules.items() if k.startswith("opentelemetry")}
        original = sys.modules.pop("trelix.llm.otel", None)
        for key in purged:
            del sys.modules[key]
        try:
            importlib.import_module("trelix.llm.otel")
            assert not any(k.startswith("opentelemetry") for k in sys.modules)
        finally:
            sys.modules.update(purged)
            # Put the ORIGINAL module object back in both places a later test can reach it from
            # (sys.modules and the package attribute), the lesson of test_otel_tracing.py's
            # retriever-import test: a half-restored module splits the state under test.
            if original is not None:
                sys.modules["trelix.llm.otel"] = original
                llm_package.otel = original  # type: ignore[attr-defined]

    def test_flag_on_without_opentelemetry_returns_the_bare_backend_and_warns_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """R6. MUTATIONS: log at DEBUG (0 records); warn per call (2 records); never return
        the backend (an exception escapes)."""
        from trelix.llm.factory import build_chat_client

        monkeypatch.setenv("TRELIX_OTEL_ENABLED", "true")
        monkeypatch.setattr(_otel_tracing(), "_env_otel_settings", None)
        monkeypatch.setattr(_otel_tracing(), "_handler", None)
        monkeypatch.setattr(_otel_tracing(), "_handler_service_name", None)
        llm_otel = importlib.import_module("trelix.llm.otel")
        monkeypatch.setattr(llm_otel, "_unavailable_warned", False)
        absent = dict.fromkeys(
            [k for k in sys.modules if k.startswith("opentelemetry")]
            + ["opentelemetry", "opentelemetry.util.genai.types", "opentelemetry.trace"]
        )
        with (
            patch.dict(sys.modules, absent),
            caplog.at_level(logging.WARNING, logger="trelix.llm.otel"),
        ):
            first = build_chat_client(cfg(provider="openai", model="gpt-4o"))
            second = build_chat_client(cfg(provider="openai", model="gpt-4o"))

        assert type(first).__name__ == "OpenAIBackend"
        assert type(second).__name__ == "OpenAIBackend"
        records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(records) == 1, "should warn exactly once, not once per call"
        message = records[0].getMessage()
        assert message.startswith(_UNAVAILABLE_PREFIX)
        assert message.endswith(_UNAVAILABLE_SUFFIX)
        assert "trelix[otel]" in message
        # The parenthesised reason is the ImportError the operator acts on (it names the
        # module), not the handler's own "could not be built" text, which is a different fault.
        reason = message[len(_UNAVAILABLE_PREFIX) : -len(_UNAVAILABLE_SUFFIX)]
        assert "opentelemetry.util.genai.types" in reason, reason

    def test_a_handler_that_cannot_be_built_also_means_the_bare_backend(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The other failure: the library imports but TelemetryHandler() failed (debug log)."""
        require_util_genai_at_the_floor(sdk=False)
        from trelix.llm.otel import traced

        llm_otel = importlib.import_module("trelix.llm.otel")
        monkeypatch.setattr(llm_otel, "_unavailable_warned", False)
        monkeypatch.setattr(_otel_tracing(), "handler_from_env", lambda: None)
        backend = FakeBackend()
        with caplog.at_level(logging.WARNING, logger="trelix.llm.otel"):
            assert traced(backend, cfg()) is backend

        (record,) = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert "TelemetryHandler could not be built" in record.getMessage()

    @pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
    def test_is_enabled_from_env_follows_trelix_otel_enabled(
        self, monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
    ) -> None:
        """MUTATIONS: return a constant; read `capture_content` instead of `enabled`."""
        monkeypatch.setenv("TRELIX_OTEL_ENABLED", value)
        monkeypatch.setenv("TRELIX_OTEL_CAPTURE_CONTENT", "true" if value == "false" else "false")
        monkeypatch.setattr(_otel_tracing(), "_env_otel_settings", None)

        assert _otel_tracing().is_enabled_from_env() is expected


# ---------------------------------------------------------------------------
# R8 -- the `_client` property (critique C1)
# ---------------------------------------------------------------------------


def _make_context() -> Any:
    from trelix.core.models import RetrievedContext

    return RetrievedContext(
        query="how does auth work",
        results=[MagicMock()],
        context_text="def authenticate(user): ...",
        total_tokens=10,
    )


def _synthesizer_over(wrapper: Any) -> Any:
    from trelix.core.config import EmbedderConfig
    from trelix.retrieval.synthesizer import Synthesizer

    with patch("trelix.retrieval.synthesizer.build_chat_client", return_value=wrapper):
        return Synthesizer(
            EmbedderConfig(_env_file=None, provider="openai"),  # type: ignore[call-arg]
            llm_config=cfg(provider="openai", model="gpt-4o"),
        )


class TestClientProperty:
    def test_client_is_the_backends_by_identity(self) -> None:
        """MUTATIONS: delete the property; return `self` from it."""
        backend = FakeBackend()
        backend._client = object()  # type: ignore[attr-defined]

        assert _wrapper(backend, RecordingHandler())._client is backend._client

    def test_hasattr_is_false_when_the_backend_has_no_client(self) -> None:
        """LiteLLMBackend has no `_client`; `hasattr` on the wrapper must stay False."""
        wrapper = _wrapper(FakeBackend(), RecordingHandler())

        assert hasattr(wrapper, "_client") is False
        assert getattr(wrapper, "_client", None) is None

    def test_nothing_else_of_the_backend_is_forwarded(self) -> None:
        backend = FakeBackend()
        backend.extra = "x"  # type: ignore[attr-defined]

        assert hasattr(_wrapper(backend, RecordingHandler()), "extra") is False

    def test_the_synthesizer_takes_the_trelix_chat_client_path_through_the_wrapper(
        self, capture_off: None, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """synthesizer.py reads `_llm_client._client` and compares it by identity to choose the
        TrelixChatClient path over the legacy raw-OpenAI path. MUTATION: delete the property ->
        the legacy path calls `.chat.completions.create` on the wrapper and synthesize() returns
        ''."""
        backend = FakeBackend(chunks=("x",))
        backend._client = object()  # type: ignore[attr-defined]
        handler = RecordingHandler()
        synth = _synthesizer_over(
            _wrapper(backend, handler, cfg(provider="openai", model="gpt-4o"))
        )

        assert synth.is_configured is True
        assert synth.synthesize(_make_context()) == "x"
        assert synth.last_error is None
        assert [inv.lifecycle for inv in handler.invocations] == [["suspend", "stop"]]

    def test_the_synthesizer_is_not_configured_when_the_backends_client_is_none(
        self, capture_off: None
    ) -> None:
        """MUTATION: return `self` from the property -> is_configured True for an unconfigured
        backend."""
        backend = FakeBackend()
        backend._client = None  # type: ignore[attr-defined]
        synth = _synthesizer_over(
            _wrapper(backend, RecordingHandler(), cfg(provider="openai", model="gpt-4o"))
        )

        assert synth.is_configured is False


# ---------------------------------------------------------------------------
# C9 -- request_model; D7 -- provider_name; R4(d) -- seed_kwargs
# ---------------------------------------------------------------------------


class TestRequestModelAndProviderName:
    def test_azure_reports_the_deployment_name(self) -> None:
        """MUTATION: read config.model -> "gpt-4o"."""
        from trelix.core.config import LLMConfig
        from trelix.llm.otel import request_model
        from trelix.llm.providers.openai_backend import OpenAIBackend

        config = LLMConfig(
            _env_file=None,  # type: ignore[call-arg]
            provider="azure",
            azure_chat_deployment="my-deploy",
        )
        assert request_model(OpenAIBackend(config), config) == "my-deploy"

    def test_anthropic_reports_its_model(self) -> None:
        pytest.importorskip("anthropic")
        from trelix.core.config import LLMConfig
        from trelix.llm.otel import request_model
        from trelix.llm.providers.anthropic_backend import AnthropicBackend

        config = LLMConfig(
            _env_file=None,  # type: ignore[call-arg]
            provider="anthropic",
            model="claude-sonnet-4-6",
        )
        assert request_model(AnthropicBackend(config), config) == "claude-sonnet-4-6"

    def test_a_missing_empty_or_non_string_model_falls_back_to_the_config(self) -> None:
        """Critique C9: '' and a non-str are "absent", not reported."""
        from trelix.llm.otel import request_model

        config = cfg(provider="openai", model="gpt-4o")
        assert request_model(SimpleNamespace(), config) == "gpt-4o"
        assert request_model(SimpleNamespace(_model=""), config) == "gpt-4o"
        assert request_model(SimpleNamespace(_model=None), config) == "gpt-4o"
        assert request_model(SimpleNamespace(_model="m"), config) == "m"

    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            ({"provider": "openai", "model": "gpt-4o"}, "openai"),
            ({"provider": "azure", "model": "gpt-4o"}, "azure.ai.openai"),
            ({"provider": "anthropic"}, "anthropic"),
            ({"provider": "bedrock", "aws_region": "us-east-1"}, "aws.bedrock"),
            ({"provider": "litellm"}, "litellm"),
            ({"provider": "vertex", "google_api_key": "test-k"}, "gcp.gemini"),
            ({"provider": "vertex", "google_project_id": "p"}, "gcp.vertex_ai"),
            (
                {"provider": "vertex", "google_api_key": "test-k", "google_project_id": "p"},
                "gcp.gemini",
            ),
        ],
    )
    def test_provider_name_rows(self, fields: dict[str, Any], expected: str) -> None:
        """D7 and R2(d). MUTATION: swap any row; invert the vertex rule."""
        from trelix.llm.otel import provider_name

        assert provider_name(cfg(**fields)) == expected

    def test_seed_kwargs_is_empty_for_every_wrapper_method(self) -> None:
        """R4(d). MUTATION: add **kwargs to a wrapper method -> {"seed": 7}."""
        from trelix.llm.client import seed_kwargs

        wrapper = _wrapper(FakeBackend(), RecordingHandler())
        assert seed_kwargs(wrapper.complete, 7) == {}
        assert seed_kwargs(wrapper.stream, 7) == {}
        assert seed_kwargs(wrapper.tool_call, 7) == {}


# ---------------------------------------------------------------------------
# What trelix assigns on the invocation, against the recording handler
# ---------------------------------------------------------------------------


class _Interrupt(BaseException):
    """Not an Exception: what KeyboardInterrupt and SystemExit are to the wrapper."""


class TestWhatTrelixAssigns:
    def test_complete_records_the_request_and_the_usage_and_ends_with_stop(
        self, capture_off: None
    ) -> None:
        """SDK-free twin of test_otel_llm_spans R2. MUTATIONS: input_tokens without the cache
        sum; finish_reasons from finish_reason; max_tokens from the argument only; provider row
        swapped; response model from cfg.model; stop() missing."""
        handler = RecordingHandler()
        response = reply()
        result = _wrapper(FakeBackend(response=response), handler).complete(
            [ChatMessage("user", "CANARY-IN-7f3a")], system="CANARY-SYS-9c1e"
        )

        assert result is response
        (inv,) = handler.invocations
        assert (inv.provider, inv.request_model) == ("anthropic", "fake-model-1")
        assert inv.max_tokens == 2048
        assert inv.response_model_name == "claude-sonnet-4-6-20260101"
        assert (inv.input_tokens, inv.output_tokens) == (19, 5)
        assert (inv.cache_read_input_tokens, inv.cache_write_input_tokens) == (7, 2)
        assert inv.finish_reasons == ["max_tokens"]
        assert inv.attributes == {"trelix.finish_reason": "length"}
        assert inv.lifecycle == ["stop"]
        for name in ("input_messages", "system_instruction", "output_messages"):
            assert not hasattr(inv, name), name

    def test_stream_suspends_at_start_and_stops_at_the_end(self, capture_off: None) -> None:
        """MUTATION: remove suspend()."""
        handler = RecordingHandler()
        it = _wrapper(FakeBackend(), handler).stream([ChatMessage("user", "q")], max_tokens=64)
        assert handler.invocations == [], "the span must start at the first next(), not the call"

        assert list(it) == ["a", "b"]
        (inv,) = handler.invocations
        assert inv.max_tokens == 64
        assert inv.lifecycle == ["suspend", "stop"]
        for name in ("input_tokens", "output_tokens", "response_model_name", "finish_reasons"):
            assert not hasattr(inv, name), name

    def test_a_stream_closed_early_is_still_stopped(self, capture_off: None) -> None:
        handler = RecordingHandler()
        it = _wrapper(FakeBackend(chunks=("a", "b", "c")), handler).stream(
            [ChatMessage("user", "q")]
        )
        assert next(it) == "a"
        it.close()

        (inv,) = handler.invocations
        assert inv.lifecycle == ["suspend", "stop"]

    def test_tool_call_records_the_request_only_and_stops(self, capture_off: None) -> None:
        """MUTATION: tool_call() passing None for its own max_tokens -> 2048 (the cfg cap)."""
        handler = RecordingHandler()
        result = _wrapper(FakeBackend(), handler).tool_call(
            [ChatMessage("user", "q")], tools=[{"type": "function"}], max_tokens=16
        )

        assert result.tool_name == "t"
        (inv,) = handler.invocations
        assert (inv.provider, inv.request_model, inv.max_tokens) == (
            "anthropic",
            "fake-model-1",
            16,
        )
        assert inv.lifecycle == ["stop"]
        assert not hasattr(inv, "input_tokens")

    def test_a_backend_error_fails_the_span_with_the_class_name_only(
        self, capture_off: None
    ) -> None:
        """Critique C3 / D10. MUTATIONS: stop() instead of fail(); fail(exc) -> message "boom";
        swallow the exception."""
        require_util_genai_at_the_floor(sdk=False)
        handler = RecordingHandler()
        wrapper = _wrapper(FakeBackend(error=RuntimeError("boom")), handler)
        with pytest.raises(RuntimeError, match="boom"):
            wrapper.complete([ChatMessage("user", "q")])

        (inv,) = handler.invocations
        (error,) = inv.lifecycle
        assert (error.type, error.message) == ("RuntimeError", "RuntimeError")

    @pytest.mark.parametrize(
        "call",
        [
            lambda w: w.complete([ChatMessage("user", "q")]),
            lambda w: list(w.stream([ChatMessage("user", "q")])),
            lambda w: w.tool_call([ChatMessage("user", "q")], tools=[]),
        ],
        ids=["complete", "stream", "tool_call"],
    )
    def test_a_base_exception_also_fails_the_span_and_propagates(
        self, capture_off: None, call: Callable[[Any], Any]
    ) -> None:
        """`except BaseException`, not `except Exception`: a KeyboardInterrupt or SystemExit
        inside the backend still ends the span with ERROR instead of leaving it open.
        MUTATION: `except Exception` in any one of the three methods -> no error recorded."""
        require_util_genai_at_the_floor(sdk=False)
        handler = RecordingHandler()
        interrupt = _Interrupt("stop")
        wrapper = _wrapper(FakeBackend(error=interrupt, chunks=("a", interrupt)), handler)
        with pytest.raises(_Interrupt):
            call(wrapper)

        (inv,) = handler.invocations
        error = inv.lifecycle[-1]
        assert error.type.endswith("._Interrupt")
        assert error.message == error.type

    def test_text_is_handed_over_only_when_the_trelix_flag_is_on(self, capture_on: None) -> None:
        """The gate on the invocation itself (critique C2). MUTATION: invert or drop the gate."""
        require_util_genai_at_the_floor(sdk=False)
        handler = RecordingHandler(captures=True)
        _wrapper(FakeBackend(response=reply()), handler).complete(
            [ChatMessage("system", "CANARY-SYS-9c1e"), ChatMessage("user", "CANARY-IN-7f3a")]
        )

        (inv,) = handler.invocations
        assert [(m.role, m.parts[0].content) for m in inv.input_messages] == [
            ("user", "CANARY-IN-7f3a")
        ]
        assert [p.content for p in inv.system_instruction] == ["CANARY-SYS-9c1e"]
        assert [(m.role, m.parts[0].content) for m in inv.output_messages] == [
            ("assistant", "CANARY-OUT-5b2d")
        ]

    def test_the_system_argument_wins_over_a_system_role_message(self, capture_on: None) -> None:
        """Critique C12 with BOTH sources present: the backends send `system` and drop the
        system-role message (openai_backend `_build_messages`, anthropic_backend
        `_extract_system`), so the span must carry the argument. MUTATION (review M16, which
        survived before this test): `system or <first system message>` ->
        `<first system message> or system`."""
        require_util_genai_at_the_floor(sdk=False)
        handler = RecordingHandler(captures=True)
        _wrapper(FakeBackend(response=reply()), handler).complete(
            [ChatMessage("system", "CANARY-MSG-SYS-4d8e"), ChatMessage("user", "CANARY-IN-7f3a")],
            system="CANARY-SYS-9c1e",
        )

        (inv,) = handler.invocations
        assert [p.content for p in inv.system_instruction] == ["CANARY-SYS-9c1e"]

    def test_inert_capture_warns_once_across_two_calls(
        self, capture_on: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """R5(d) against the fake: it says it will not capture. MUTATIONS: remove the call;
        warn per call."""
        require_util_genai_at_the_floor(sdk=False)
        wrapper = _wrapper(FakeBackend(response=reply()), RecordingHandler(captures=False))
        with caplog.at_level(logging.WARNING, logger="trelix.retrieval.otel"):
            for _ in range(2):
                wrapper.complete([ChatMessage("user", "q")])

        records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(records) == 1
        assert "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT" in records[0].getMessage()
