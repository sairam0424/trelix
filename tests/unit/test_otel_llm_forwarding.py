"""
TracedChatClient forwards every argument to the backend unchanged (roadmap C-8 PR 3, review
round 3).

Forwarding is the wrapper's primary job once TRELIX_OTEL_ENABLED is on, and nothing pinned it:
FakeBackend ignored its arguments, so a wrapper that passed `force_tool=None`, `temperature=None`,
`system=None` or `thinking=False` to the backend survived every test. The planner calls
`tool_call(..., force_tool="produce_query_plan")`: a dropped `force_tool` lets the model answer
freely and every plan falls back to `default_plan()` while the span still records the request and
the suite stays green. SDK-free, like test_otel_llm_wrapper.py (which is at its size limit):
`FakeBackend.calls` records what each method RECEIVED by parameter name, so these tests pin the
values that reached the backend, not whether they travelled positionally or by keyword.
"""

from __future__ import annotations

from typing import Any

from tests.unit.otel_llm_fakes import FakeBackend, RecordingHandler, cfg, reply
from trelix.llm.client import ChatMessage


def _wrapper(backend: FakeBackend) -> Any:
    from trelix.llm.otel import TracedChatClient

    return TracedChatClient(backend, cfg(), RecordingHandler())


class TestForwarding:
    """Every argument is a non-default value, so a wrapper that drops one (the backend's
    default applies) and a wrapper that replaces one are both visible. MUTATION: any one of
    the forwarded keywords replaced by `None` / `False` in any of the three methods."""

    def test_complete_forwards_every_argument(self) -> None:
        backend = FakeBackend(response=reply())

        _wrapper(backend).complete(
            [ChatMessage("user", "q")], max_tokens=5, temperature=0.2, system="s", thinking=True
        )

        assert backend.calls == [
            (
                "complete",
                {
                    "messages": [ChatMessage("user", "q")],
                    "max_tokens": 5,
                    "temperature": 0.2,
                    "system": "s",
                    "thinking": True,
                },
            )
        ]

    def test_stream_forwards_every_argument(self) -> None:
        backend = FakeBackend(chunks=("a",))

        chunks = _wrapper(backend).stream(
            [ChatMessage("user", "q")], max_tokens=5, temperature=0.2, system="s", thinking=True
        )

        assert list(chunks) == ["a"]
        assert backend.calls == [
            (
                "stream",
                {
                    "messages": [ChatMessage("user", "q")],
                    "max_tokens": 5,
                    "temperature": 0.2,
                    "system": "s",
                    "thinking": True,
                },
            )
        ]

    def test_tool_call_forwards_every_argument(self) -> None:
        backend = FakeBackend()

        _wrapper(backend).tool_call(
            [ChatMessage("user", "q")],
            [{"type": "function", "name": "produce_query_plan"}],
            force_tool="produce_query_plan",
            max_tokens=16,
        )

        assert backend.calls == [
            (
                "tool_call",
                {
                    "messages": [ChatMessage("user", "q")],
                    "tools": [{"type": "function", "name": "produce_query_plan"}],
                    "force_tool": "produce_query_plan",
                    "max_tokens": 16,
                },
            )
        ]
