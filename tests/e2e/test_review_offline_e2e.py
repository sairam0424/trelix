"""The installed `trelix` console script reviews a diff against `FakeOpenAIServer` (R-C4-04).

`trelix review --json` runs as a real OS subprocess and reaches `tests/e2e/fake_openai_server.py`
over a real TCP socket on `127.0.0.1`, so the exit code, the stdout array, the outcome record, the
request count and order, the `Authorization: Bearer trelix-local` header, the `max_tokens` field
and the `prompt_truncated` detail are asserted at the process boundary that `CliRunner` and
`httpx.MockTransport` cannot see. The server answers a five-file diff with, in order: a finding;
a reply cut off at the limit and the same again for the retry; a `[]` whose `prompt_tokens` says
the prompt was cut; a 200 body that is not JSON; prose with no array. One hunk reviewed of five:
exit 4, the finding on stdout, the four unreviewed hunks in the outcome file, exit 0 when the
fraction allows it.

Exactly six requests because the repository has no index, so no planner call is made: without
an index `DiffReviewer._get_retriever` returns None and no `Retriever` (hence no `QueryPlanner`
and no planner `tool_call`) is ever built.

`requires_network` is a directory label applied by `tests/e2e/conftest.py`; this test needs no
model and no network beyond tiktoken's one-time `cl100k_base` download at collection. tiktoken's `cl100k_base` is loaded at collection (module level, before
pytest-socket's per-test ban) so the on-disk cache the child process reads is warm; the child
inherits `TIKTOKEN_CACHE_DIR`, `DATA_GYM_CACHE_DIR` and `TMPDIR`, none of which is a `TRELIX_*`
or provider name, so it never downloads.

Assertions on the child's stderr normalise whitespace: Rich wraps at 80 columns in a pipe; the
child runs with `NO_COLOR=1` and without `FORCE_COLOR`, so no ANSI lands in the text. The
`trelix` on PATH is what runs: `release.yml`'s smoke job runs this file against the built wheel.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import tiktoken

from tests._env_isolation import CONFIG_NON_PREFIXED_ENV, INSTALLED_SDK_PROVIDER_ENV
from tests.e2e.fake_openai_server import FakeOpenAIServer, Reply, completion, garbage_json
from trelix.core.config import LLMConfig
from trelix.llm.client import ChatMessage
from trelix.llm.providers.openai_backend import OpenAIBackend

# Loaded at collection, the shape of test_chunker_token_budget_boundary.py: the on-disk cache is
# warm before pytest-socket's per-test ban, so the child's estimate never needs the network.
_ENC = tiktoken.get_encoding("cl100k_base")

_TRELIX = shutil.which("trelix")

_FILES = ("a.py", "b.py", "c.py", "d.py", "e.py")
_DIFF = "".join(
    f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n@@ -1,1 +1,2 @@\n old\n+added\n"
    for name in _FILES
)
_FINDING = '[{"line_start": 1, "line_end": 1, "severity": "WARN", "comment": "smell here"}]'
_FINDING_JSON = [{"file": "a.py", "lines": "1-1", "severity": "WARN", "comment": "smell here"}]
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
_USERINFO_ERROR = "TRELIX_LLM_BASE_URL must not carry a user name or password"
# Rich treats a pipe as a terminal when any of these is set; the child must get none of them.
_RICH_FORCE_NAMES = ("FORCE_COLOR", "TTY_COMPATIBLE", "TTY_INTERACTIVE")


def _script() -> list[Reply]:
    """The six replies, in request order."""
    return [
        completion(_FINDING, prompt_tokens=10_000),  # a.py: reviewed
        completion("[", prompt_tokens=10_000, finish_reason="length"),  # b.py: cut off
        completion("[", prompt_tokens=10_000, finish_reason="length"),  # b.py: retry, cut off
        completion("[]", prompt_tokens=1),  # c.py: the server cut the prompt
        garbage_json(),  # d.py: not JSON
        completion("I see no problems here.", prompt_tokens=10_000),  # e.py: no array
    ]


def _require_trelix() -> str:
    if _TRELIX is None:
        pytest.fail("trelix console script not on PATH: pip install -e .")
    return _TRELIX


def _child_env(base_url: str, home: Path, outcome: Path) -> dict[str, str]:
    """`os.environ` minus every TRELIX_, provider and Rich-forcing name, plus the fixed names.

    `TRELIX_CONFIG_FILE` points at an existing empty file so the operator's own env file is
    never read; `HOME` is a fresh directory for the same reason.
    """
    kept = {key: os.environ[key] for key in os.environ if not key.upper().startswith("TRELIX_")}
    dropped = (
        {name.upper() for name in CONFIG_NON_PREFIXED_ENV}
        | {name.upper() for name in INSTALLED_SDK_PROVIDER_ENV}
        | set(_RICH_FORCE_NAMES)
    )
    home.mkdir()
    config_file = home / "trelix-env"
    config_file.write_text("", encoding="utf-8")
    return {
        **{key: value for key, value in kept.items() if key.upper() not in dropped},
        "HOME": str(home),
        "TRELIX_CONFIG_FILE": str(config_file),
        "TRELIX_LLM_PROVIDER": "openai",
        "TRELIX_LLM_BASE_URL": base_url,
        "TRELIX_LLM_MODEL": "qwen2.5-coder:7b",
        "TRELIX_REVIEW_OUTCOME_FILE": str(outcome),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NO_COLOR": "1",
    }


def _run_review(
    tmp_path: Path, server: FakeOpenAIServer, extra_env: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run `trelix review <repo> --diff ... --json`; the repository is empty (no index)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    diff = tmp_path / "changes.diff"
    diff.write_text(_DIFF, encoding="utf-8")
    outcome = tmp_path / "outcome.json"
    env = _child_env(server.base_url, tmp_path / "home", outcome) | (extra_env or {})
    proc = subprocess.run(
        [_require_trelix(), "review", str(repo), "--diff", str(diff), "--json"],
        env=env,
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=100,
        check=False,
    )
    return proc, outcome


@pytest.mark.timeout(120)
class TestReviewAgainstTheFakeServer:
    def test_exit_4_with_the_one_finding_on_stdout(self, tmp_path: Path) -> None:
        """MUTATION: `_local_server = False` in `OpenAIBackend.__init__` (no client, exit 3,
        no requests); this one pins the exit code and the stdout array."""
        with FakeOpenAIServer(_script()) as server:
            proc, _ = _run_review(tmp_path, server)

        assert proc.returncode == 4, proc.stderr
        assert json.loads(proc.stdout) == _FINDING_JSON

    def test_six_requests_in_order_over_the_real_socket(self, tmp_path: Path) -> None:
        """MUTATION: `_RETRY_CEILING` 16384 -> 8192 (the third `max_tokens`); the placeholder
        bearer -> `x`; `max_completion_tokens` on the local branch."""
        with FakeOpenAIServer(_script()) as server:
            _run_review(tmp_path, server)

        assert len(server.requests) == 6
        assert [r.path for r in server.requests] == ["/v1/chat/completions"] * 6
        assert [r.headers["authorization"] for r in server.requests] == ["Bearer trelix-local"] * 6
        assert all(
            r.headers["content-type"].startswith("application/json") for r in server.requests
        )
        bodies = server.bodies()
        assert [b["messages"][1]["content"].split(" ")[1] for b in bodies] == [
            "a.py",
            "b.py",
            "b.py",
            "c.py",
            "d.py",
            "e.py",
        ]
        assert [b["messages"][0]["role"] for b in bodies] == ["system"] * 6
        assert [len(b["messages"]) for b in bodies] == [2] * 6
        assert [b["max_tokens"] for b in bodies] == [4096, 4096, 16384, 4096, 4096, 4096]
        assert all("max_completion_tokens" not in b for b in bodies)
        assert [b["model"] for b in bodies] == ["qwen2.5-coder:7b"] * 6
        assert all(b.get("stream") is not True for b in bodies)

    def test_the_outcome_file_is_the_literal_record(self, tmp_path: Path) -> None:
        """MUTATION: delete the `_prompt_truncated` branch of `_classify_reply` (c.py is then
        REVIEWED with `[]`); `TRUNCATION_FLOOR` 0.85 -> 0.0; a `completion()` with no `usage`."""
        with FakeOpenAIServer(_script()) as server:
            _, outcome = _run_review(tmp_path, server)

        assert json.loads(outcome.read_text(encoding="utf-8")) == _OUTCOME

    def test_stderr_carries_the_warnings_of_a_real_process(self, tmp_path: Path) -> None:
        """MUTATION: the same two as the outcome test, seen on the child's stderr instead."""
        with FakeOpenAIServer(_script()) as server:
            proc, _ = _run_review(tmp_path, server)

        text = " ".join(proc.stderr.split())
        assert "under the 20B floor" in text
        assert "Local server truncated the prompt: it reports 1 prompt tokens" in text
        assert "c.py:1 was not reviewed (truncated: prompt_truncated)" in text
        assert "4 of 5 hunks could not be fully reviewed" in text
        assert "Traceback" not in proc.stderr
        assert "\x1b[" not in proc.stderr

    def test_the_fraction_at_one_accepts_the_partial_review(self, tmp_path: Path) -> None:
        """MUTATION: ignore `TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION` in the exit-code rule."""
        with FakeOpenAIServer(_script()) as server:
            proc, outcome = _run_review(
                tmp_path, server, extra_env={"TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION": "1"}
            )

        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout) == _FINDING_JSON
        assert len(server.requests) == 6
        assert json.loads(outcome.read_text(encoding="utf-8")) == {**_OUTCOME, "exit_code": 0}


@pytest.mark.timeout(120)
class TestACredentialInTheUrlNeverReachesAProcessStderr:
    def test_configuration_error_without_the_value_or_a_traceback(self, tmp_path: Path) -> None:
        """MUTATION: delete the `except _PydanticValidationError` arm in `review` (the
        `ValueError` arm then prints `Error: 1 validation error ...`, no `Configuration error`);
        delete both arms (Typer's traceback lands on stderr)."""
        with FakeOpenAIServer(_script()) as server:
            proc, _ = _run_review(
                tmp_path,
                server,
                extra_env={"TRELIX_LLM_BASE_URL": f"http://user:pw@127.0.0.1:{server.port}/v1"},
            )

        assert proc.returncode == 1, proc.stderr
        text = " ".join(proc.stderr.split())
        assert "Configuration error" in text
        assert _USERINFO_ERROR in text
        assert "pw@" not in proc.stderr
        assert "input_value" not in proc.stderr
        assert "Traceback" not in proc.stderr
        assert proc.stdout == ""
        assert server.requests == []


class TestStreamingOverARealSocket:
    def test_stream_reads_the_sse_reply_over_tcp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MUTATION: make the handler ignore `stream` and always answer JSON (`stream()` yields
        nothing); `max_completion_tokens` on the local branch; the placeholder bearer -> `x`.

        The e2e conftest never scrubs provider names; a developer's exported `OPENAI_API_KEY`
        would otherwise be sent as the bearer: red locally, green in CI.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        with FakeOpenAIServer([completion("x", prompt_tokens=3)]) as server:
            # A tag at the 20B floor, so construction logs nothing.
            backend = OpenAIBackend(
                LLMConfig(
                    provider="openai", base_url=server.base_url, model="gpt-oss:20b", _env_file=None
                )
            )
            chunks = list(backend.stream([ChatMessage(role="user", content="hi")], max_tokens=256))

        assert chunks == ["x"]
        assert len(server.requests) == 1
        assert server.requests[0].headers["authorization"] == "Bearer trelix-local"
        body = server.bodies()[0]
        assert body["stream"] is True
        assert body["max_tokens"] == 256
        assert "max_completion_tokens" not in body
        assert body["model"] == "gpt-oss:20b"
