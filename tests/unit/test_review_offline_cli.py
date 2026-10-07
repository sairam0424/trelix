"""`trelix review --json` against a local OpenAI-compatible server that cut one prompt (R-C4-02,
the CLI end to end, in process).

The real openai SDK runs over a scripted `httpx.MockTransport`: the constructor is patched at
`trelix.llm.providers.openai_backend.OpenAI` with a wrapper that forwards every kwarg and adds
`http_client=`, so the bearer, the path and the request bodies are what the SDK sends. The
server answers a five-file diff with, in order: a finding; a reply cut off at the limit and the
same again for the retry; a `[]` whose `prompt_tokens` says the prompt was cut; a 200 body that
is not JSON; prose with no array. One hunk reviewed of five: exit 4, the finding on stdout, the
four unreviewed hunks in the outcome file, exit 0 when the fraction allows it.

Exactly six requests because the repository has no index, so no planner call is made: without
an index `DiffReviewer._get_retriever` returns None and no `Retriever` (hence no `QueryPlanner`
and no planner `tool_call`) is ever built. The size-floor warning for `qwen2.5-coder:7b` is
asserted on `result.stderr`, not caplog: the CLI rebinds the root handler.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import tiktoken
from openai import OpenAI
from typer.testing import CliRunner

from trelix.cli.main import app

# Loaded at collection, the shape of test_chunker_token_budget_boundary.py: the process cache is
# warm before pytest-socket's per-test ban, so the guard's estimate never needs the network.
_ENC = tiktoken.get_encoding("cl100k_base")

runner = CliRunner()

_FILES = ("a.py", "b.py", "c.py", "d.py", "e.py")
_DIFF = "".join(
    f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n@@ -1,1 +1,2 @@\n old\n+added\n"
    for name in _FILES
)
_FINDING = '[{"line_start": 1, "line_end": 1, "severity": "WARN", "comment": "smell here"}]'
_FINDING_JSON = [{"file": "a.py", "lines": "1-1", "severity": "WARN", "comment": "smell here"}]
_ENV = {
    "TRELIX_LLM_PROVIDER": "openai",
    "TRELIX_LLM_BASE_URL": "http://127.0.0.1:1/v1",
    "TRELIX_LLM_MODEL": "qwen2.5-coder:7b",
}
_OUTCOME = {
    "schema_version": 1,
    "hunks_total": 5,
    "hunks_reviewed": 1,
    "hunks_unreviewed": 4,
    "exit_code": 4,
    "hunks": [
        {"file": "b.py", "line": 1, "status": "truncated", "detail": "length_after_retry"},
        {"file": "c.py", "line": 1, "status": "truncated", "detail": "prompt_truncated"},
        {"file": "d.py", "line": 1, "status": "error", "detail": "exception:JSONDecodeError"},
        {"file": "e.py", "line": 1, "status": "parse_failed", "detail": "no_review_array"},
    ],
    "hunks_omitted": 0,
}


def _completion(content: str, *, prompt_tokens: int, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 1,
            "model": "qwen2.5-coder:7b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,
                "total_tokens": prompt_tokens + 1,
            },
        },
    )


def _script() -> Iterator[httpx.Response]:
    """The six replies, in request order (fresh objects per run)."""
    yield _completion(_FINDING, prompt_tokens=10_000)  # a.py: reviewed
    yield _completion("[", prompt_tokens=10_000, finish_reason="length")  # b.py: cut off
    yield _completion("[", prompt_tokens=10_000, finish_reason="length")  # b.py: retry, cut off
    yield _completion("[]", prompt_tokens=1)  # c.py: the server cut the prompt
    yield httpx.Response(200, headers={"content-type": "application/json"}, content=b"{not json")
    yield _completion("I see no problems here.", prompt_tokens=10_000)  # e.py: no array


class _Server:
    def __init__(self) -> None:
        self._replies = _script()
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return next(self._replies)

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


def _review(tmp_path: Path, extra_env: dict[str, str] | None = None) -> tuple[Any, _Server, Path]:
    """Run `trelix review <tmp_path> --diff ... --json`; the repository has no index."""
    server = _Server()
    diff_file = tmp_path / "changes.diff"
    diff_file.write_text(_DIFF)
    outcome = tmp_path / "outcome.json"
    env = {**_ENV, "TRELIX_REVIEW_OUTCOME_FILE": str(outcome), **(extra_env or {})}

    def construct(**kwargs: Any) -> OpenAI:
        return OpenAI(http_client=httpx.Client(transport=httpx.MockTransport(server)), **kwargs)

    with patch("trelix.llm.providers.openai_backend.OpenAI", new=construct):
        result = runner.invoke(
            app, ["review", str(tmp_path), "--diff", str(diff_file), "--json"], env=env
        )
    return result, server, outcome


class TestTheSixReplyScript:
    def test_exit_4_with_the_one_finding_on_stdout(self, tmp_path: Path) -> None:
        result, _, _ = _review(tmp_path)

        assert result.exit_code == 4, result.stderr
        assert json.loads(result.stdout) == _FINDING_JSON

    def test_six_requests_in_order_with_the_placeholder_bearer(self, tmp_path: Path) -> None:
        """MUTATION: retry the prompt-truncated hunk too (a seventh request); send the retry at
        the first limit again (no 16384)."""
        _, server, _ = _review(tmp_path)

        bodies = server.bodies()
        assert len(bodies) == 6
        assert [b["messages"][1]["content"].split(" ")[1] for b in bodies] == [
            "a.py",
            "b.py",
            "b.py",
            "c.py",
            "d.py",
            "e.py",
        ]
        assert [b["max_tokens"] for b in bodies] == [4096, 4096, 16384, 4096, 4096, 4096]
        assert all("max_completion_tokens" not in b for b in bodies)
        assert [b["model"] for b in bodies] == ["qwen2.5-coder:7b"] * 6
        assert [r.headers["authorization"] for r in server.requests] == ["Bearer trelix-local"] * 6
        assert {str(r.url) for r in server.requests} == {"http://127.0.0.1:1/v1/chat/completions"}

    def test_the_outcome_file_is_the_literal_record(self, tmp_path: Path) -> None:
        """MUTATION: salvage from the prompt-truncated reply (c.py drops out of the list);
        map the signal to `error` or after the finish-reason branches."""
        _, _, outcome = _review(tmp_path)

        assert json.loads(outcome.read_text(encoding="utf-8")) == _OUTCOME

    def test_stderr_names_the_size_floor_and_the_cut_prompt(self, tmp_path: Path) -> None:
        result, _, _ = _review(tmp_path)

        text = " ".join(result.stderr.split())
        assert "under the 20B floor" in text
        assert "Local server truncated the prompt: it reports 1 prompt tokens" in text
        assert "c.py:1 was not reviewed (truncated: prompt_truncated)" in text
        assert "4 of 5 hunks could not be fully reviewed" in text

    def test_the_fraction_at_one_accepts_the_partial_review(self, tmp_path: Path) -> None:
        result, server, outcome = _review(tmp_path, {"TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION": "1"})

        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == _FINDING_JSON
        assert len(server.requests) == 6
        assert json.loads(outcome.read_text(encoding="utf-8")) == {**_OUTCOME, "exit_code": 0}
