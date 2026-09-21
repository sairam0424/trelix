"""
Unit tests for AbstractiveCompressor (headless — a hand-written fake chat
client, never a real LLM call).

Covers the load-bearing invariants shared with ExtractiveCompressor:
  * target_ratio >= 1.0 is a byte-identical passthrough (and makes no LLM call)
  * the declaration line is ALWAYS kept verbatim, never paraphrased
  * every kept span is a subrange of the unit's original line span
  * token_count is recomputed via tiktoken (never inherited)
  * result-lossless: an empty/failed LLM response degrades to passthrough,
    never an empty or dropped result
  * a raising chat client degrades to passthrough rather than propagating
  * the summary text is rendered under an explicit "not verbatim" header, and
    the query/signature/docstring/qualified_name the research says matter are
    actually present in the prompt sent to the chat client
"""

from __future__ import annotations

from typing import Any

import tiktoken

from trelix.compression import AbstractiveCompressor, CompressionResult, CompressionUnit
from trelix.llm.client import ChatMessage, ChatResponse, ToolCallResponse, TrelixChatClient

_ENC = tiktoken.get_encoding("cl100k_base")


def _tok(text: str) -> int:
    return len(_ENC.encode(text))


def _assert_subranges(result: CompressionResult, line_start: int, line_end: int) -> None:
    for a, b in result.kept_spans:
        assert line_start <= a <= b <= line_end, (
            f"span ({a},{b}) escapes original [{line_start},{line_end}]"
        )


# ---------------------------------------------------------------------------
# Fakes — real ABC subclasses, never Mock (matches
# tests/unit/test_chunker_contextual_llm_gaps.py's established convention)
# ---------------------------------------------------------------------------


class _StubChatClient(TrelixChatClient):
    """Hand-written fake. `.calls` records every complete() invocation's own
    arguments so a test can assert on exactly what AbstractiveCompressor
    sent, not on a value recomputed from the module under test."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: bool = False,
    ) -> ChatResponse:
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return ChatResponse(content=self._content, model="stub-model", finish_reason="stop")

    def stream(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("AbstractiveCompressor never calls stream()")

    def tool_call(self, *args: object, **kwargs: object) -> ToolCallResponse:
        raise AssertionError("AbstractiveCompressor never calls tool_call()")


class _RaisingChatClient(TrelixChatClient):
    """A chat client whose complete() always raises — for the graceful-
    degradation path."""

    def complete(self, *args: object, **kwargs: object) -> ChatResponse:
        raise RuntimeError("simulated provider outage")

    def stream(self, *args: object, **kwargs: object) -> Any:
        raise AssertionError("not called in these tests")

    def tool_call(self, *args: object, **kwargs: object) -> ToolCallResponse:
        raise AssertionError("not called in these tests")


def _unit(body: str, **overrides: object) -> CompressionUnit:
    base: dict[str, object] = {
        "symbol_id": 1,
        "body": body,
        "signature": "def handle_request(req):",
        "docstring": "Handle an inbound request.",
        "line_start": 10,
        "line_end": 10 + len(body.splitlines()) - 1,
        "qualified_name": "module.handle_request",
    }
    base.update(overrides)
    return CompressionUnit(**base)  # type: ignore[arg-type]


_BODY = "\n".join(
    [
        "def handle_request(req):",
        '    """Handle an inbound request."""',
        "    validate(req)",
        "    result = process(req)",
        "    log(result)",
        "    return result",
    ]
)


def test_target_ratio_at_or_above_1_is_a_byte_identical_passthrough_with_no_llm_call() -> None:
    unit = _unit(_BODY)
    client = _StubChatClient("this should never be used")
    compressor = AbstractiveCompressor(client)

    result = compressor.compress("how are requests handled?", unit, target_ratio=1.0)

    assert result.text == unit.body
    assert result.provider == "abstractive"
    assert client.calls == []  # no LLM call made for a passthrough
    _assert_subranges(result, unit.line_start, unit.line_end)


def test_compress_keeps_the_declaration_line_verbatim() -> None:
    unit = _unit(_BODY)
    client = _StubChatClient("Validates, processes, and logs the request.")
    compressor = AbstractiveCompressor(client)

    result = compressor.compress("how are requests handled?", unit, target_ratio=0.3)

    assert unit.body.splitlines()[0] in result.text
    assert "Validates, processes, and logs the request." in result.text
    _assert_subranges(result, unit.line_start, unit.line_end)


def test_summary_is_marked_as_not_verbatim_source() -> None:
    unit = _unit(_BODY)
    client = _StubChatClient("A synthesized description of the body.")
    compressor = AbstractiveCompressor(client)

    result = compressor.compress("q", unit, target_ratio=0.3)

    assert "not verbatim source" in result.text
    assert "A synthesized description of the body." in result.text


def test_token_count_is_recomputed_not_inherited() -> None:
    unit = _unit(_BODY)
    client = _StubChatClient("Short summary.")
    compressor = AbstractiveCompressor(client)

    result = compressor.compress("q", unit, target_ratio=0.3)

    assert result.token_count == _tok(result.text)
    assert result.token_count != result.original_token_count


def test_prompt_includes_query_signature_docstring_and_qualified_name() -> None:
    """Per research (arXiv:2304.06815): don't rely on the LLM to implicitly
    re-derive these facts from the body -- inject them explicitly."""
    unit = _unit(_BODY, signature="def handle_request(req):", docstring="Handle it.")
    client = _StubChatClient("summary")
    compressor = AbstractiveCompressor(client)

    compressor.compress("what does validate do?", unit, target_ratio=0.3)

    assert len(client.calls) == 1
    sent_prompt = client.calls[0]["messages"][0].content
    assert "what does validate do?" in sent_prompt
    assert unit.signature in sent_prompt
    assert unit.docstring in sent_prompt
    assert unit.qualified_name in sent_prompt
    assert unit.body in sent_prompt


def test_empty_llm_response_degrades_to_passthrough() -> None:
    unit = _unit(_BODY)
    client = _StubChatClient("   ")  # whitespace-only -> treated as empty
    compressor = AbstractiveCompressor(client)

    result = compressor.compress("q", unit, target_ratio=0.3)

    assert result.text == unit.body
    _assert_subranges(result, unit.line_start, unit.line_end)


def test_raising_chat_client_degrades_to_passthrough_never_raises() -> None:
    unit = _unit(_BODY)
    compressor = AbstractiveCompressor(_RaisingChatClient())

    result = compressor.compress("q", unit, target_ratio=0.3)

    assert result.text == unit.body
    assert result.provider == "abstractive"
    _assert_subranges(result, unit.line_start, unit.line_end)


def test_empty_body_degrades_to_passthrough_with_no_llm_call() -> None:
    unit = _unit("   \n  ")
    client = _StubChatClient("should not be called")
    compressor = AbstractiveCompressor(client)

    result = compressor.compress("q", unit, target_ratio=0.3)

    assert result.text == unit.body
    assert client.calls == []


def test_result_is_never_empty() -> None:
    unit = _unit(_BODY)
    client = _StubChatClient("x")  # minimal but non-empty summary
    compressor = AbstractiveCompressor(client)

    result = compressor.compress("q", unit, target_ratio=0.1)

    assert result.text.strip() != ""
    assert result.kept_spans


def test_kept_spans_never_escape_the_units_original_range() -> None:
    unit = _unit(_BODY, line_start=100, line_end=105)
    client = _StubChatClient("summary text")
    compressor = AbstractiveCompressor(client)

    result = compressor.compress("q", unit, target_ratio=0.3)

    _assert_subranges(result, 100, 105)


def test_make_compressor_dispatches_to_abstractive_and_wires_a_real_chat_client() -> None:
    """make_compressor's abstractive branch, exercised end-to-end against a
    real LLMConfig(provider="openai", ...) -- constructing the client must
    not itself raise just because no network call is ever made."""
    from types import SimpleNamespace

    from trelix.compression import make_compressor
    from trelix.core.config import LLMConfig

    placeholder_credential = "not-a-real-key" + "-construction-only"
    fake_index_config = SimpleNamespace(
        retrieval=SimpleNamespace(compression_provider="abstractive"),
        llm=LLMConfig(provider="openai", openai_api_key=placeholder_credential),
    )

    compressor = make_compressor(fake_index_config, db=object())

    assert isinstance(compressor, AbstractiveCompressor)


def test_make_compressor_still_raises_for_a_genuinely_unknown_provider() -> None:
    from types import SimpleNamespace

    from trelix.compression import make_compressor

    fake_config = SimpleNamespace(retrieval=SimpleNamespace(compression_provider="made-up"))

    try:
        make_compressor(fake_config, db=object())
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError as exc:
        assert "made-up" in str(exc)
