"""The citation instruction reaches every synthesis prompt, and only for a tagged context.

Companion to test_citation_tags.py. Three prompt paths exist: ``Synthesizer.stream()``
(plain `trelix ask` and REST /ask), ``synthesize()`` through a TrelixChatClient (FLARE,
eval-synthesis) and ``synthesize()`` through a raw injected client. All three go through
``_system_prompt(intent, cited=)`` now, each keeping the base prompt it used before:
``stream()`` falls back to the ``feature_flow`` prompt for a missing intent, the other two
pass ``context.intent`` as is (so ``""`` resolves to the default prompt). GraphRAG
map-reduce prefixes its group headers with the tag looked up by symbol_id and appends one
line to each of its two templates.

With ``citation_sources`` empty (the flag off) every prompt is byte-identical to today's.
Every expected value is a literal written in this file. Each test names the mutation that
must make it fail.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.fixtures.citation_context import MIDDLEWARE, QUERY, bearer_result, verify_result
from trelix.core.config import EmbedderConfig, RetrievalConfig
from trelix.core.models import RetrievedContext
from trelix.llm.client import ChatMessage, ChatResponse, TrelixChatClient
from trelix.retrieval.citations import CitationSource
from trelix.retrieval.graph_rag import GraphRAGSynthesizer
from trelix.retrieval.synthesizer import Synthesizer

FEATURE_FLOW_PROMPT = (
    "You are a senior engineer explaining a feature's end-to-end implementation. "
    "Trace the flow from entry point to final output, naming the key functions and "
    "data transformations at each step. Show the call chain clearly."
)
DEFAULT_PROMPT = (
    "You are an expert software engineer answering questions about a codebase. "
    "Base your answer strictly on the provided code context. "
    "Be precise, cite the relevant file and function names, and avoid speculation."
)
CITATION_INSTRUCTION = (
    "\n\nEvery block of the code context starts with a tag such as [C3]. After each sentence "
    "that relies on a block, write that block's tag, for example "
    "`validate_token checks the signature [C3].` Cite only tags that appear in the context."
)

# Tags deliberately NOT in rank order: verify (rank 1, symbol 11) is [C2], bearer is [C1].
SWAPPED_SOURCES = (
    CitationSource(1, 33, MIDDLEWARE, 70, 80, "AuthMiddleware.bearer"),
    CitationSource(2, 11, MIDDLEWARE, 42, 67, "AuthMiddleware.verify"),
)


def _context(*, tagged: bool, intent: str = "") -> RetrievedContext:
    return RetrievedContext(
        query=QUERY,
        results=[verify_result(), bearer_result(rank=2)],
        context_text="ctx",
        total_tokens=40,
        intent=intent,
        citation_sources=SWAPPED_SOURCES if tagged else (),
    )


# ---------------------------------------------------------------------------
# Synthesizer: the instruction reaches all three prompt paths, only when tagged
# ---------------------------------------------------------------------------


class _RecordingChatClient(TrelixChatClient):
    """A real TrelixChatClient, so `_stream_response` takes its normal path."""

    def __init__(self) -> None:
        self._client = object()  # non-None: this backend "has credentials"
        self.systems: list[str] = []

    def complete(self, messages: list[ChatMessage], **_kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    def stream(self, messages: list[ChatMessage], **kwargs: Any) -> Iterator[str]:
        self.systems.append(kwargs["system"])
        yield "The "
        yield "answer."

    def tool_call(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _synthesizer(client: Any) -> Synthesizer:
    with patch("trelix.retrieval.synthesizer.build_chat_client", return_value=client):
        return Synthesizer(EmbedderConfig(_env_file=None))


class TestSynthesizerPrompt:
    """MUTATIONS, each named by the case it breaks: always append -> the untagged cases;
    never append -> the tagged cases; drop stream()'s `or "feature_flow"` -> both stream
    cases (they would get DEFAULT_PROMPT); route `_stream_response` around the helper ->
    the synthesize cases."""

    @pytest.mark.parametrize(
        ("tagged", "expected"),
        [(True, FEATURE_FLOW_PROMPT + CITATION_INSTRUCTION), (False, FEATURE_FLOW_PROMPT)],
    )
    def test_stream_keeps_its_feature_flow_fallback(self, tagged: bool, expected: str) -> None:
        client = _RecordingChatClient()
        synth = _synthesizer(client)

        tokens = list(synth.stream(_context(tagged=tagged), RetrievalConfig()))

        assert tokens == ["The ", "answer."]
        assert client.systems == [expected]

    @pytest.mark.parametrize(
        ("tagged", "expected"),
        [(True, DEFAULT_PROMPT + CITATION_INSTRUCTION), (False, DEFAULT_PROMPT)],
    )
    def test_synthesize_passes_the_intent_as_is(self, tagged: bool, expected: str) -> None:
        client = _RecordingChatClient()
        synth = _synthesizer(client)

        answer = synth.synthesize(_context(tagged=tagged))

        assert answer == "The answer."
        assert synth.last_error is None
        assert client.systems == [expected]

    @pytest.mark.parametrize(
        ("tagged", "expected"),
        [(True, DEFAULT_PROMPT + CITATION_INSTRUCTION), (False, DEFAULT_PROMPT)],
    )
    def test_raw_client_path_sends_the_same_system_message(
        self, tagged: bool, expected: str
    ) -> None:
        raw = MagicMock()
        synth = _synthesizer(raw)

        synth.synthesize(_context(tagged=tagged))

        create = raw._client.chat.completions.create
        assert create.call_count == 1
        assert create.call_args.kwargs["messages"][0] == {"role": "system", "content": expected}

    def test_a_named_intent_keeps_its_own_prompt_under_the_instruction(self) -> None:
        client = _RecordingChatClient()
        synth = _synthesizer(client)

        list(synth.stream(_context(tagged=True, intent="feature_flow"), RetrievalConfig()))
        synth.synthesize(_context(tagged=True, intent="feature_flow"))

        assert client.systems == [FEATURE_FLOW_PROMPT + CITATION_INSTRUCTION] * 2


# ---------------------------------------------------------------------------
# GraphRAG: group headers tagged by symbol_id, one cite line per prompt
# ---------------------------------------------------------------------------

MAP_PROMPT = (
    "Partially answer the following question using ONLY the code context provided.\n"
    "Be concise. Focus on what this specific code reveals about the question.\n"
    "\n"
    "Question: how is a jwt verified\n"
    "\n"
    "Code context:\n"
    "{verify_header}# src/auth/middleware.py — verify (method)\n"
    "# File: src/auth/middleware.py | Language: Python\n"
    "\n"
    "def verify(self, token):\n"
    "    return decode_token(token)\n"
    "\n"
    "{bearer_header}# src/auth/middleware.py — bearer (method)\n"
    "# File: src/auth/middleware.py | Language: Python\n"
    "\n"
    "def bearer(self, header):\n"
    "    return header.split()[1]\n"
    "\n"
    "Partial answer:"
)
MAP_CITE_LINE = (
    "\nEach block starts with a tag such as [C3]; write the tag after each statement it supports."
)
REDUCE_PROMPT = (
    "You are synthesizing partial answers about a codebase query into a single coherent "
    "response.\n"
    "Each partial answer was derived from a different subset of the relevant code.\n"
    "\n"
    "Question: how is a jwt verified\n"
    "\n"
    "Partial answers:\n"
    "[Partial 1]\n"
    "partial answer\n"
    "\n"
    "Provide a complete, synthesized answer that integrates all relevant information above.\n"
    "Cite specific file and function names where possible. Be precise and technical."
)
REDUCE_CITE_LINE = (
    "\nKeep every [C#] tag from the partial answers beside the statement it supports; "
    "do not invent tags."
)


def _graph_rag() -> tuple[GraphRAGSynthesizer, MagicMock]:
    """A GraphRAGSynthesizer over a keyless openai config with a recording raw client."""
    message = MagicMock()
    message.content = "partial answer"
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    raw = MagicMock()
    raw.chat.completions.create.return_value = response

    synth = GraphRAGSynthesizer(
        EmbedderConfig(provider="openai", _env_file=None),  # type: ignore[call-arg]
        RetrievalConfig(),
    )
    synth._client = raw
    return synth, raw


def _prompts(raw: MagicMock) -> list[str]:
    return [c.kwargs["messages"][1]["content"] for c in raw.chat.completions.create.call_args_list]


class TestGraphRagPrompts:
    def test_tagged_context_tags_group_headers_by_symbol_id(self) -> None:
        """MUTATION: look the tag up by rank (verify would be [C1]); drop either cite line."""
        synth, raw = _graph_rag()

        answer = synth.synthesize(QUERY, _context(tagged=True), "feature_flow")

        assert answer == "partial answer"
        assert _prompts(raw) == [
            MAP_PROMPT.format(verify_header="[C2] ", bearer_header="[C1] ") + MAP_CITE_LINE,
            REDUCE_PROMPT + REDUCE_CITE_LINE,
        ]

    def test_untagged_context_sends_todays_prompts_byte_for_byte(self) -> None:
        """MUTATION: append a cite line unconditionally."""
        synth, raw = _graph_rag()

        synth.synthesize(QUERY, _context(tagged=False), "feature_flow")

        assert _prompts(raw) == [
            MAP_PROMPT.format(verify_header="", bearer_header=""),
            REDUCE_PROMPT,
        ]
