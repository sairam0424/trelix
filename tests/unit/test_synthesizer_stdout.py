"""`Synthesizer(stream_to_stdout=False)` and `Synthesizer.last_context` (C-5 design, 3.4 PR4).

`trelix ask --json` owns stdout, so the Synthesizer it builds must keep every stdout write in:
the streamed tokens and closing newline of `_stream_response`, and the three notices of
`synthesize()` (not configured, no results, synthesis failed), which go through `_notice`.
`stream()` never writes either way. `last_context` is the context the last call answered
from, so the CLI can verify markers after FLARE, whose final context is private to its loop;
in `stream()` it is set inside the generator, on the first `next()`. Fixtures:
tests/unit/ask_cli_harness.py.

Every expected value is a literal in this file or the harness. MUTATIONS, each named by the
test it breaks:
`stream_to_stdout: bool = False` -> test_synthesize_streams_to_stdout_by_default;
leave any one of the three notices unrouted -> test_synthesize_notices_* (its row);
leave a `sys.stdout.write` in `_stream_response` unguarded ->
    test_synthesize_with_stream_to_stdout_off_prints_nothing;
set `last_context` when `stream()` is called instead of on the first `next()` ->
    test_stream_sets_last_context_when_iteration_starts;
set `last_context` after the no-results guard in `synthesize()` ->
    test_synthesize_sets_last_context_even_when_it_abstains;
set `last_context` after the no-results guard in `stream()` ->
    test_stream_sets_last_context_even_when_it_abstains.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest

from tests.unit.ask_cli_harness import (
    NO_RESULTS_NOTICE,
    RaisingChatClient,
    ScriptedChatClient,
    make_context,
    no_results_context,
)
from trelix.core.config import EmbedderConfig, RetrievalConfig
from trelix.core.models import RetrievedContext
from trelix.retrieval.synthesizer import Synthesizer


def _synth(client: Any, **kwargs: Any) -> Synthesizer:
    with patch("trelix.retrieval.synthesizer.build_chat_client", return_value=client):
        return Synthesizer(EmbedderConfig(_env_file=None), **kwargs)  # type: ignore[call-arg]


class TestSynthesizerStdout:
    def test_synthesize_streams_to_stdout_by_default(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        synth = _synth(ScriptedChatClient(("The ", "answer.")))

        assert synth.synthesize(make_context()) == "The answer."
        assert capsys.readouterr().out == "The answer.\n"

    def test_synthesize_with_stream_to_stdout_off_prints_nothing(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        synth = _synth(ScriptedChatClient(("The ", "answer.")), stream_to_stdout=False)

        assert synth.synthesize(make_context()) == "The answer."
        assert capsys.readouterr().out == ""
        assert synth.last_error is None

    @pytest.mark.parametrize(
        ("make_client", "make_ctx", "printed"),
        [
            pytest.param(
                lambda: ScriptedChatClient(("x",), configured=False),
                make_context,
                "[trelix] No LLM API key configured — skipping synthesis. Set OPENAI_API_KEY "
                "(or AZURE_API_KEY + AZURE_ENDPOINT) to enable answers.\n",
                id="not-configured",
            ),
            pytest.param(
                lambda: ScriptedChatClient(("x",)),
                no_results_context,
                NO_RESULTS_NOTICE + "\n",
                id="no-results",
            ),
            pytest.param(
                lambda: RaisingChatClient(RuntimeError("503 upstream down")),
                make_context,
                "\n[trelix] Synthesis failed: 503 upstream down\n",
                id="raises",
            ),
        ],
    )
    def test_synthesize_notices_print_by_default_and_not_when_silenced(
        self,
        capsys: pytest.CaptureFixture[str],
        make_client: Callable[[], Any],
        make_ctx: Callable[[], RetrievedContext],
        printed: str,
    ) -> None:
        # Factories, not instances: a scripted client is consumed by one call, and mutmut's
        # stats and clean passes run the suite twice in one process over the same module.
        _synth(make_client()).synthesize(make_ctx())
        assert capsys.readouterr().out == printed

        _synth(make_client(), stream_to_stdout=False).synthesize(make_ctx())
        assert capsys.readouterr().out == ""

    def test_stream_never_writes_to_stdout_either_way(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        for synth in (
            _synth(ScriptedChatClient(("The ", "answer."))),
            _synth(ScriptedChatClient(("The ", "answer.")), stream_to_stdout=False),
        ):
            assert list(synth.stream(make_context(), RetrievalConfig())) == ["The ", "answer."]
            assert capsys.readouterr().out == ""


class TestLastContext:
    def test_stream_sets_last_context_when_iteration_starts(self) -> None:
        synth = _synth(ScriptedChatClient(("The ", "answer.")))
        context = make_context()
        assert synth.last_context is None

        tokens = synth.stream(context, RetrievalConfig())
        assert synth.last_context is None  # a generator: nothing ran yet

        assert next(tokens) == "The "
        assert synth.last_context is context

    def test_synthesize_sets_last_context_even_when_it_abstains(self) -> None:
        synth = _synth(ScriptedChatClient(("x",)))
        context = no_results_context()

        synth.synthesize(context)

        assert synth.last_context is context
        assert synth.last_abstain_reason == "no_results"

    def test_stream_sets_last_context_even_when_it_abstains(self) -> None:
        """Design 3.4: `last_context` is set at the top of `stream()`, before the no-results
        guard, so the CLI can read the context whichever way the stream ended."""
        synth = _synth(ScriptedChatClient(("x",)))
        context = no_results_context()

        assert list(synth.stream(context, RetrievalConfig())) == [NO_RESULTS_NOTICE]

        assert synth.last_context is context
        assert synth.last_abstain_reason == "no_results"
