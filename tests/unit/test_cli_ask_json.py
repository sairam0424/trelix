"""`trelix ask --json` prints one object and nothing else on stdout (C-5 design, R4, D3, D4).

Fourth of six changes toward `trelix ask` answers that cite the code they rest on. `--json`
prints one object `{query, answer, abstained, abstain_reason, citations}` and nothing else on
stdout; a failure leaves stdout empty and puts the reason on stderr. The verifier runs only
when the context carried `citation_sources`: with the flag off a `[C1]` in an answer is text
(B5(b)). Fixtures: tests/unit/ask_cli_harness.py. The human-mode footer is in
tests/unit/test_cli_ask_footer.py, the Synthesizer's stdout switch in
tests/unit/test_synthesizer_stdout.py.

Every expected value is a literal in this file or the harness. MUTATIONS, each named by the
test it breaks:
print tokens as they stream in json mode -> test_json_two_marker_answer_* (stdout is not one
    JSON object);
forget `stream_to_stdout=False` under FLARE -> test_flare_json_* (the streamed answer precedes
    the object on stdout);
treat an abstention as exit 1 -> test_json_abstention_*;
verify the markers of an abstention line -> test_json_abstention_with_a_marker_* (a `[C1]` row
    appears where the footer, gated the same way, prints nothing);
emit `lines` as two ints -> test_json_two_marker_answer_* (literal comparison);
leave the third `synthesize()` write unrouted -> test_flare_json_stream_failure_* (stdout not
    empty);
verify citations when `citation_sources` is empty -> test_json_with_the_flag_off_* (a `[C1]`
    row appears);
refuse `--json` after retrieval instead of before -> test_json_context_only_* (Retriever is
    constructed);
drop the `--agentic`/`--session` conflict check -> test_json_with_an_agent_flag_* (exit 0);
drop the environment-agentic refusal -> test_json_with_agentic_mode_from_the_environment_*;
drop `--json` from the option -> test_ask_help_lists_json.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tests.unit.ask_cli_harness import (
    ANSWER,
    ANSWER_TOKENS,
    JWT,
    MIDDLEWARE,
    NO_RESULTS_NOTICE,
    QUERY,
    RaisingChatClient,
    ScriptedChatClient,
    app,
    invoke_ask,
    make_context,
    make_repo,
    no_results_context,
    one_line,
    runner,
)

JSON_KEYS = {"query", "answer", "abstained", "abstain_reason", "citations"}
TWO_MARKER_OBJECT = {
    "query": QUERY,
    "answer": ANSWER,
    "abstained": False,
    "abstain_reason": None,
    "citations": [
        {
            "marker": "[C2]",
            "status": "valid",
            "path": MIDDLEWARE,
            "lines": "70-80",
            "symbol": "AuthMiddleware.bearer",
            "detail": "",
        },
        {
            "marker": "[C1]",
            "status": "valid",
            "path": MIDDLEWARE,
            "lines": "42-67",
            "symbol": "AuthMiddleware.verify",
            "detail": "",
        },
    ],
}
NO_RESULTS_OBJECT = {
    "query": QUERY,
    "answer": NO_RESULTS_NOTICE,
    "abstained": True,
    "abstain_reason": "no_results",
    "citations": [],
}
CONTEXT_ONLY_REFUSAL = (
    "Error: --json needs LLM synthesis; with the local embedder and FLARE off, trelix ask prints "
    "the retrieved context only. Use trelix search --json for machine-readable retrieval, or a "
    "non-local --provider."
)
CONFLICT_REFUSAL = "--json cannot be combined with --agentic or --session."


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path)


# ---------------------------------------------------------------------------
# R4: --json prints one object and nothing else
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_json_two_marker_answer_is_the_design_object(self, repo: Path) -> None:
        result = invoke_ask(repo, ScriptedChatClient(ANSWER_TOKENS), make_context(), "--json")

        assert result.exit_code == 0, result.stderr
        assert result.stderr == ""
        assert json.loads(result.stdout) == TWO_MARKER_OBJECT

    def test_json_abstention_exits_0_with_abstained_true(self, repo: Path) -> None:
        client = ScriptedChatClient(
            ("INSUFFICIENT_EVIDENCE: ", "nothing about retries was retrieved.")
        )

        result = invoke_ask(repo, client, make_context(), "--json")

        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert set(payload) == JSON_KEYS
        assert payload["answer"] == "INSUFFICIENT_EVIDENCE: nothing about retries was retrieved."
        assert payload["abstained"] is True
        assert payload["abstain_reason"] == "insufficient_evidence"
        assert payload["citations"] == []

    def test_json_abstention_with_a_marker_has_no_citations(self, repo: Path) -> None:
        """An abstention names what is missing, not sources: its markers are not verified, so
        `citations` is `[]` exactly where human mode prints no footer."""
        client = ScriptedChatClient(
            ("INSUFFICIENT_EVIDENCE: the retrieved [C1] covers verification, not retries.",)
        )

        result = invoke_ask(repo, client, make_context(), "--json")

        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["abstained"] is True
        assert payload["abstain_reason"] == "insufficient_evidence"
        assert payload["citations"] == []

    def test_json_empty_retrieval_is_the_notice_with_no_results(self, repo: Path) -> None:
        client = ScriptedChatClient(ANSWER_TOKENS)

        result = invoke_ask(repo, client, no_results_context(), "--json")

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == NO_RESULTS_OBJECT
        assert client.calls == 0

    def test_json_with_the_flag_off_treats_a_marker_as_text(self, repo: Path) -> None:
        """B5(b): no `citation_sources`, so a `[C1]` in the answer is not verified."""
        client = ScriptedChatClient(("see [C1] here",))

        result = invoke_ask(repo, client, make_context(sources=()), "--json")

        assert result.exit_code == 0, result.stderr
        payload = json.loads(result.stdout)
        assert set(payload) == JSON_KEYS
        assert payload["answer"] == "see [C1] here"
        assert payload["citations"] == []

    def test_json_reports_stale_and_unknown_markers_with_null_fields(self, repo: Path) -> None:
        client = ScriptedChatClient(("Decoding happens in [C3]; nothing supports [C9].",))

        result = invoke_ask(repo, client, make_context(), "--json")

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout)["citations"] == [
            {
                "marker": "[C3]",
                "status": "line_out_of_range",
                "path": JWT,
                "lines": "10-30",
                "symbol": "decode_token",
                "detail": "src/auth/jwt.py has 25 lines, the cited chunk ends at line 30; re-index",
            },
            {
                "marker": "[C9]",
                "status": "unknown",
                "path": None,
                "lines": None,
                "symbol": None,
                "detail": "no retrieved chunk has this tag",
            },
        ]

    @pytest.mark.parametrize(
        ("make_client", "reason"),
        [
            pytest.param(
                lambda: RaisingChatClient(RuntimeError("401 invalid api key")),
                "401 invalid api key",
                id="raises",
            ),
            pytest.param(
                lambda: ScriptedChatClient(("[trelix] LLM not configured.",), configured=False),
                "LLM not configured",
                id="keyless",
            ),
        ],
    )
    def test_json_stream_failure_exits_1_with_an_empty_stdout(
        self, repo: Path, make_client: Callable[[], Any], reason: str
    ) -> None:
        # A factory, not an instance: a scripted client is consumed by one run, and mutmut's
        # stats and clean passes run the suite twice in one process over the same module.
        result = invoke_ask(repo, make_client(), make_context(), "--json")

        assert result.exit_code == 1, result.stdout
        assert result.stdout == ""
        assert reason in one_line(result.stderr)

    def test_flare_json_is_the_same_object_from_the_loops_answer(self, repo: Path) -> None:
        """`synthesize()` streams through `_stream_response`; `stream_to_stdout=False` keeps that
        off stdout, and the object is built from `loop.run()`'s return and `last_context`."""
        result = invoke_ask(
            repo, ScriptedChatClient(ANSWER_TOKENS), make_context(), "--json", flare=True
        )

        assert result.exit_code == 0, result.stderr
        assert result.stderr == ""
        assert json.loads(result.stdout) == TWO_MARKER_OBJECT

    def test_flare_json_empty_retrieval_prints_no_notice(self, repo: Path) -> None:
        """FLARE re-retrieves once on the notice; neither round's notice reaches stdout.
        `json.loads` over the whole of stdout is the proof: a notice line before or after the
        object is not JSON (`json.dumps` writes the em dash as `\\u2014`, so a substring count
        of the notice would see nothing either way)."""
        client = ScriptedChatClient(ANSWER_TOKENS)

        result = invoke_ask(repo, client, no_results_context(), "--json", flare=True)

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == NO_RESULTS_OBJECT
        assert client.calls == 0

    def test_flare_json_stream_failure_keeps_stdout_empty(self, repo: Path) -> None:
        """B2: `synthesize()`'s `[trelix] Synthesis failed:` line goes through `_notice`."""
        client = RaisingChatClient(RuntimeError("503 upstream down"))

        result = invoke_ask(repo, client, make_context(), "--json", flare=True)

        assert result.exit_code == 1, result.stdout
        assert result.stdout == ""
        assert "Synthesis failed: 503 upstream down" in one_line(result.stderr)


# ---------------------------------------------------------------------------
# D3, D4: the refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    @pytest.mark.parametrize(
        "flags", [["--agentic"], ["--session", "x"]], ids=["agentic", "session"]
    )
    def test_json_with_an_agent_flag_is_a_usage_error(self, repo: Path, flags: list[str]) -> None:
        with patch("trelix.agent.AgentLoop") as agent_loop:
            result = runner.invoke(app, ["ask", str(repo), QUERY, "--json", *flags])

        assert result.exit_code == 2, result.output
        assert CONFLICT_REFUSAL in one_line(result.output)
        agent_loop.assert_not_called()

    def test_json_with_agentic_mode_from_the_environment_exits_1(self, repo: Path) -> None:
        """The agent loop prints its own result shape, however it was switched on."""
        with patch("trelix.agent.AgentLoop") as agent_loop:
            result = runner.invoke(
                app,
                ["ask", str(repo), QUERY, "--json", "--provider", "openai"],
                env={"TRELIX_RETRIEVAL_AGENTIC": "true"},
            )

        assert result.exit_code == 1, result.output
        assert result.stdout == ""
        assert "--json is not available in agentic mode" in one_line(result.stderr)
        agent_loop.assert_not_called()

    def test_json_context_only_mode_is_refused_before_retrieval(self, repo: Path) -> None:
        """B7: both conditions are known from config, so the index is never opened."""
        with patch(
            "trelix.retrieval.retriever.Retriever",
            side_effect=AssertionError("Retriever must not be constructed"),
        ) as retriever:
            result = runner.invoke(app, ["ask", str(repo), QUERY, "--json", "--provider", "local"])

        assert result.exit_code == 1, result.output
        assert result.stdout == ""
        assert CONTEXT_ONLY_REFUSAL in one_line(result.stderr)
        retriever.assert_not_called()

    def test_json_with_the_local_embedder_and_flare_on_is_allowed(self, repo: Path) -> None:
        """FLARE synthesises whatever the embedder is, so there is an answer to return."""
        with (
            patch("trelix.retrieval.retriever.Retriever") as mock_retriever,
            patch(
                "trelix.retrieval.synthesizer.build_chat_client",
                return_value=ScriptedChatClient(ANSWER_TOKENS),
            ),
        ):
            mock_retriever.return_value.retrieve.return_value = make_context()
            result = runner.invoke(
                app,
                ["ask", str(repo), QUERY, "--json", "--provider", "local"],
                env={"TRELIX_RETRIEVAL_FLARE": "true"},
            )

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == TWO_MARKER_OBJECT


def test_ask_help_lists_json() -> None:
    result = runner.invoke(app, ["ask", "--help"])

    assert result.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--json" in plain
