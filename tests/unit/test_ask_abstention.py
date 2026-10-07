"""`trelix ask` abstains instead of answering from nothing (C-5 design, R2 to R2d).

Two abstentions exist. Retrieval found nothing: `Synthesizer.stream()` and `synthesize()`
answer with one notice literal and never call the model (`stream()`, behind plain `trelix ask`
and REST `/ask`, used to call it with `No relevant code found.` as the whole code context and
got an answer from the model's own knowledge). The model found nothing in what was retrieved:
the citation instruction asks for one line starting `INSUFFICIENT_EVIDENCE:`, which streams
as any answer does, is recorded in `last_abstain_reason` once the whole answer is in, exits 0
and makes FLARE re-retrieve. Neither sets `last_error`: an abstention is an answer, not a
failure.

Every expected value is a literal in this file. MUTATIONS, each named by the test it breaks:
remove the `stream()` empty-results guard -> test_stream_on_no_results_* (the call count);
set the reason only in `synthesize()` -> test_stream_on_no_results_* (the reason);
drop `lstrip()` in `is_abstention` -> the leading-newline case of test_is_abstention;
`in` instead of `startswith` -> its mid-text case;
detect inside the token loop instead of at stream end -> test_a_stream_closed_early_*;
drop `"insufficient_evidence:"` from FLARE's phrases -> test_flare_re_retrieves_* (one call);
treat the abstention as `last_error` -> test_cli_streams_the_abstention_* (exit 1);
forget to reset the reason -> test_the_reason_resets_*;
skip the GraphRAG answer -> test_synthesize_records_a_graphrag_abstention;
look at the retrieval result before the LLM key in `stream()` -> test_a_missing_llm_key_*;
drop the colon from `ABSTAIN_PREFIX` -> the no-colon case of test_is_abstention;
drop the colon from FLARE's phrase -> test_the_marker_without_its_colon_*.
The `last_error is None` guard before `_record_abstention` in `stream()` has no killing test: with
`last_error` set there the collected text is blank, so dropping it is an equivalent mutant.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from trelix.cli.main import app
from trelix.core.config import EmbedderConfig, RetrievalConfig
from trelix.core.models import RetrievedContext
from trelix.llm.client import ChatMessage, ChatResponse, TrelixChatClient
from trelix.retrieval.citations import is_abstention
from trelix.retrieval.flare import FLARELoop, _contains_uncertainty
from trelix.retrieval.synthesizer import Synthesizer

runner = CliRunner()

QUERY = "how are retries done"
NO_RESULTS_NOTICE = "[trelix] No relevant code found — cannot synthesize an answer."
ABSTENTION = "INSUFFICIENT_EVIDENCE: nothing about retries was retrieved."
# The marker split across tokens, after a blank one: only the whole answer shows it.
ABSTENTION_TOKENS = ("\n", "INSUFFICIENT_EVIDENCE: ", "nothing about retries")
ANSWER_TOKENS = ("The ", "answer.")
# Above graph_rag_threshold_tokens (8000), so synthesize() delegates to GraphRAG map-reduce.
LARGE_CONTEXT_TOKENS = 9000


class _ScriptedChatClient(TrelixChatClient):
    """A real TrelixChatClient (so `synthesize()` takes its normal path) that counts its
    calls and streams one scripted answer per call, in order."""

    def __init__(self, *answers: tuple[str, ...], configured: bool = True) -> None:
        # non-None: this backend "has credentials"; None: it streams its own placeholder.
        self._client = object() if configured else None
        self._answers = list(answers)
        self.calls = 0

    def complete(self, messages: list[ChatMessage], **_kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    def stream(self, messages: list[ChatMessage], **_kwargs: Any) -> Iterator[str]:
        self.calls += 1
        return iter(self._answers.pop(0))

    def tool_call(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _synth(client: Any) -> Synthesizer:
    with patch("trelix.retrieval.synthesizer.build_chat_client", return_value=client):
        return Synthesizer(EmbedderConfig(_env_file=None))  # type: ignore[call-arg]


def _context(*, total_tokens: int = 42) -> RetrievedContext:
    return RetrievedContext(
        query=QUERY,
        results=[MagicMock()],
        context_text="def retry(call): ...",
        total_tokens=total_tokens,
    )


def _no_results() -> RetrievedContext:
    """What the assembler returns when retrieval found nothing."""
    return RetrievedContext(
        query=QUERY, results=[], context_text="No relevant code found.", total_tokens=0
    )


# ---------------------------------------------------------------------------
# R2: an empty retrieval never reaches the model
# ---------------------------------------------------------------------------


class TestNoResults:
    def test_stream_on_no_results_yields_the_notice_and_never_calls_the_model(self) -> None:
        client = _ScriptedChatClient(ANSWER_TOKENS)
        synth = _synth(client)

        tokens = list(synth.stream(_no_results(), RetrievalConfig()))

        assert tokens == [NO_RESULTS_NOTICE]
        assert client.calls == 0
        assert synth.last_abstain_reason == "no_results"
        assert synth.last_error is None

    def test_synthesize_on_no_results_returns_the_same_notice_without_a_call(self) -> None:
        client = _ScriptedChatClient(ANSWER_TOKENS)
        synth = _synth(client)

        answer = synth.synthesize(_no_results())

        assert answer == NO_RESULTS_NOTICE
        assert client.calls == 0
        assert synth.last_abstain_reason == "no_results"
        assert synth.last_error is None

    def test_no_results_clears_the_previous_calls_failure(self) -> None:
        """An empty stream is a recorded failure; the next, empty retrieval is not one."""
        synth = _synth(_ScriptedChatClient(()))
        list(synth.stream(_context(), RetrievalConfig()))
        assert synth.last_error is not None

        list(synth.stream(_no_results(), RetrievalConfig()))

        assert synth.last_error is None
        assert synth.last_abstain_reason == "no_results"

    def test_a_missing_llm_key_is_reported_before_the_empty_retrieval(self) -> None:
        """Configuration first, as in `synthesize()`: a misconfiguration is not hidden behind the
        notice until a question retrieves code. The stream is the backend's own placeholder
        (the backend has no client, so there is no API call) with `last_error` set."""
        client = _ScriptedChatClient(("[trelix] LLM not configured.",), configured=False)
        synth = _synth(client)

        tokens = list(synth.stream(_no_results(), RetrievalConfig()))

        assert tokens == ["[trelix] LLM not configured."]
        assert client.calls == 1
        assert synth.last_error == (
            "LLM not configured — set OPENAI_API_KEY (or AZURE_API_KEY + AZURE_ENDPOINT), "
            "or choose another provider with TRELIX_LLM_PROVIDER."
        )
        assert synth.last_abstain_reason is None


# ---------------------------------------------------------------------------
# R2b: INSUFFICIENT_EVIDENCE detection, on the whole answer, at stream end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("INSUFFICIENT_EVIDENCE: no retrieved chunk mentions rate limiting", True),
        ("\n  INSUFFICIENT_EVIDENCE: x", True),
        ("Insufficient_evidence: x", False),
        ("INSUFFICIENT_EVIDENCE — x", False),
        ("The answer. INSUFFICIENT_EVIDENCE:", False),
        ("", False),
    ],
    ids=["plain", "leading-blank", "wrong-case", "no-colon", "mid-text", "empty"],
)
def test_is_abstention(text: str, expected: bool) -> None:
    assert is_abstention(text) is expected


class TestAbstentionIsRecorded:
    def test_stream_records_an_abstention_once_the_stream_ends(self) -> None:
        synth = _synth(_ScriptedChatClient(ABSTENTION_TOKENS))

        tokens = list(synth.stream(_context(), RetrievalConfig()))

        assert tokens == ["\n", "INSUFFICIENT_EVIDENCE: ", "nothing about retries"]
        assert synth.last_abstain_reason == "insufficient_evidence"
        assert synth.last_error is None

    def test_stream_of_an_answer_records_no_abstention(self) -> None:
        synth = _synth(_ScriptedChatClient(ANSWER_TOKENS))

        assert list(synth.stream(_context(), RetrievalConfig())) == ["The ", "answer."]
        assert synth.last_abstain_reason is None
        assert synth.last_error is None

    def test_a_stream_closed_early_records_nothing(self) -> None:
        """The reader has not seen the end of the answer, so nothing is known about it."""
        synth = _synth(_ScriptedChatClient(("INSUFFICIENT_EVIDENCE: ", "x")))
        tokens = synth.stream(_context(), RetrievalConfig())

        assert next(tokens) == "INSUFFICIENT_EVIDENCE: "
        tokens.close()

        assert synth.last_abstain_reason is None
        assert synth.last_error is None

    def test_synthesize_records_an_abstention_from_the_returned_answer(self) -> None:
        synth = _synth(_ScriptedChatClient(("INSUFFICIENT_EVIDENCE: ", "nothing about retries")))

        answer = synth.synthesize(_context())

        assert answer == "INSUFFICIENT_EVIDENCE: nothing about retries"
        assert synth.last_abstain_reason == "insufficient_evidence"
        assert synth.last_error is None

    def test_synthesize_records_a_graphrag_abstention(self) -> None:
        synth = _synth(_ScriptedChatClient(ANSWER_TOKENS))

        with (
            patch("trelix.retrieval.graph_rag.GraphRAGSynthesizer.should_use", return_value=True),
            patch(
                "trelix.retrieval.graph_rag.GraphRAGSynthesizer.synthesize",
                return_value=ABSTENTION,
            ),
        ):
            answer = synth.synthesize(_context(total_tokens=LARGE_CONTEXT_TOKENS))

        assert answer == "INSUFFICIENT_EVIDENCE: nothing about retries was retrieved."
        assert synth.last_abstain_reason == "insufficient_evidence"
        assert synth.last_error is None

    def test_the_reason_resets_on_the_next_call(self) -> None:
        client = _ScriptedChatClient(ABSTENTION_TOKENS, ANSWER_TOKENS, ANSWER_TOKENS)
        synth = _synth(client)

        list(synth.stream(_context(), RetrievalConfig()))
        assert synth.last_abstain_reason == "insufficient_evidence"
        list(synth.stream(_context(), RetrievalConfig()))
        assert synth.last_abstain_reason is None

        synth.synthesize(_no_results())
        assert synth.last_abstain_reason == "no_results"
        synth.synthesize(_context())
        assert synth.last_abstain_reason is None
        assert client.calls == 3


# ---------------------------------------------------------------------------
# R2c: FLARE treats the abstention as the uncertainty it exists for
# ---------------------------------------------------------------------------


def _flare_config(max_retries: int) -> MagicMock:
    config = MagicMock()
    config.retrieval.flare_enabled = True
    config.retrieval.flare_max_retries = max_retries
    config.embedder = EmbedderConfig(_env_file=None)  # type: ignore[call-arg]
    return config


class TestFlare:
    def test_the_abstention_line_is_an_uncertainty_phrase(self) -> None:
        assert _contains_uncertainty("INSUFFICIENT_EVIDENCE: x") is True

    def test_the_marker_without_its_colon_is_not_one(self) -> None:
        assert _contains_uncertainty("insufficient_evidence was never printed") is False

    def test_flare_re_retrieves_after_an_abstention(self) -> None:
        retriever = MagicMock()
        retriever.retrieve.return_value = _context()
        client = _ScriptedChatClient(
            ("INSUFFICIENT_EVIDENCE: ", "nothing about retries"), ANSWER_TOKENS
        )
        synth = _synth(client)

        answer = FLARELoop(retriever, synth, _flare_config(max_retries=2)).run(QUERY)

        assert answer == "The answer."
        assert retriever.retrieve.call_count == 2
        assert client.calls == 2
        assert synth.last_abstain_reason is None  # the final answer was not an abstention

    def test_flare_on_an_empty_retrieval_prints_the_notice_once_per_round(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The notice contains `no relevant code`, an uncertainty phrase, so FLARE re-retrieves
        (once at the default budget of 1) and every round answers with the notice; the model is
        never called."""
        retriever = MagicMock()
        retriever.retrieve.return_value = _no_results()
        client = _ScriptedChatClient(ANSWER_TOKENS)
        synth = _synth(client)

        answer = FLARELoop(retriever, synth, _flare_config(max_retries=1)).run(QUERY)

        assert capsys.readouterr().out == NO_RESULTS_NOTICE + "\n" + NO_RESULTS_NOTICE + "\n"
        assert answer == NO_RESULTS_NOTICE
        assert retriever.retrieve.call_count == 2
        assert client.calls == 0
        assert synth.last_abstain_reason == "no_results"


# ---------------------------------------------------------------------------
# R2d: the CLI exits 0 on either abstention, with nothing on stderr
# ---------------------------------------------------------------------------


def _invoke_ask(tmp_path: Path, client: Any, context: RetrievedContext) -> Any:
    # `ask` refuses a repo with no index; Retriever is mocked, the index file is not.
    (tmp_path / ".trelix").mkdir(exist_ok=True)
    (tmp_path / ".trelix" / "index.db").touch()
    with (
        patch("trelix.retrieval.retriever.Retriever") as mock_retriever,
        patch("trelix.retrieval.synthesizer.build_chat_client", return_value=client),
    ):
        mock_retriever.return_value.retrieve.return_value = context
        return runner.invoke(app, ["ask", str(tmp_path), QUERY, "--provider", "openai"])


class TestCliExit:
    def test_cli_streams_the_abstention_and_exits_0(self, tmp_path: Path) -> None:
        client = _ScriptedChatClient(
            ("INSUFFICIENT_EVIDENCE: ", "nothing about retries was retrieved.")
        )

        result = _invoke_ask(tmp_path, client, _context())

        assert result.exit_code == 0, result.stderr
        assert result.stdout == "INSUFFICIENT_EVIDENCE: nothing about retries was retrieved.\n"
        assert result.stderr == ""

    def test_cli_on_an_empty_retrieval_prints_the_notice_without_an_llm_call(
        self, tmp_path: Path
    ) -> None:
        client = _ScriptedChatClient(ANSWER_TOKENS)

        result = _invoke_ask(tmp_path, client, _no_results())

        assert result.exit_code == 0, result.stderr
        assert result.stdout == NO_RESULTS_NOTICE + "\n"
        assert result.stderr == ""
        assert client.calls == 0

    def test_cli_on_an_empty_retrieval_without_an_llm_key_still_exits_1(
        self, tmp_path: Path
    ) -> None:
        """The missing key is reported first (stderr, exit 1), not hidden behind the notice."""
        client = _ScriptedChatClient(("[trelix] LLM not configured.",), configured=False)

        result = _invoke_ask(tmp_path, client, _no_results())

        assert result.exit_code == 1, result.stdout
        assert "LLM not configured" in result.stderr
        assert result.stdout == "\n"
