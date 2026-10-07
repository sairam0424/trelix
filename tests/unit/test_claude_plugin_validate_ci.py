"""CI runs Claude Code's own validator over the marketplace and the plugin, exactly pinned.

WHY THIS EXISTS. The offline tests in test_claude_plugin_{manifest,skill,session_start}.py pin
the shapes this repository chose for `.claude-plugin/marketplace.json` and `plugins/trelix/`;
only `claude plugin validate --strict` pins what Claude Code accepts (an unquoted
`${CLAUDE_PLUGIN_ROOT}` in hooks.json, an unrecognised manifest field and a missing `version`
are warnings that `--strict` turns into exit 1). ci.yml's `typescript-sdk` job runs it through
`npx -y @anthropic-ai/claude-code@X.Y.Z` from the repository root (the job's default working
directory is the SDK package, where `.` would be the wrong target). This file is the second
statement of that step, written as literals: its name, its working directory, its env, its
three command lines and the Claude Code version, and the rule that every `npx` in every
workflow carries an exact `@X.Y.Z` (the npm counterpart of release.yml's `build==X.Y.Z`). The
version literal lives here and in ci.yml only; the docs use placeholders, so a validator bump
is one reviewed edit of two files and never a plugin version bump.

Each rule is a pure function over the parsed workflow, and `TestTheCheckersBite` feeds it
documents that break it, so a parser that stopped matching fails there instead of leaving
the real-file assertions green for the wrong reason.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
1. Delete the whole step -> the first test fails naming the step.
2. Drop `working-directory`, or set it to `.` -> fails on `${{ github.workspace }}`.
3. Drop the `env` block -> fails on CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC.
4. One line's `@2.1.292` -> `@latest` -> the three-lines test and the repo-wide rule fail.
5. `@2.1.292` -> `@2.1.293` on all three lines -> the three-lines and pinned-pairs tests fail.
6. Drop `--strict` from one line; append ` || true` to one line; add `continue-on-error: true`
   -> the three-lines test fails.
7. Move the step above the `setup-node` step -> the order test fails.
8. Add `npx eslint .` to any other workflow's `run:` -> the repo-wide rule fails for that file.
9. Drop the `(?:-y|--yes) ` group from `_PINNED_NPX` -> the missing-`-y` fixture stops being
   flagged and `TestTheCheckersBite` fails.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

_WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
_CI = "ci.yml"
_JOB = "typescript-sdk"
_JOB_NAME = "TypeScript SDK"
_SDK_DIR = "packages/trelix-typescript"
_STEP_NAME = "Validate the Claude Code marketplace and plugin"
_PACKAGE = "@anthropic-ai/claude-code"
EXPECTED_CLAUDE_CODE_VERSION = "2.1.292"
# One literal, on purpose: a test that built it from the version constant could not notice a
# line that lost the version, the `--strict` or the second target.
_EXPECTED_RUN = (
    "npx -y @anthropic-ai/claude-code@2.1.292 --version\n"
    "npx -y @anthropic-ai/claude-code@2.1.292 plugin validate --strict .\n"
    "npx -y @anthropic-ai/claude-code@2.1.292 plugin validate --strict plugins/trelix"
)
_EXPECTED_ENV = {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
_WORKSPACE = "${{ github.workspace }}"
_MASKING_FRAGMENTS = ("|| true", "|| :", "|| exit 0", "; true", "&& true", "set +e")
# `npx -y <package>@X.Y.Z ...` as the whole line: `-y` (or `--yes`) so npx never prompts, a
# package that may be scoped, and exactly three integer components (no range, no tag, no
# fourth component). `cd x && npx ...` and `echo npx ...` do not match: write npx on its own line.
_PINNED_NPX = re.compile(r"^npx (?:-y|--yes) (@?[^@\s]+)@(\d+\.\d+\.\d+)(?:\s|$)")
# A parser that finds nothing reports "every npx is pinned" as loudly as a clean tree; today
# ci.yml has three npx lines, so the walk must see at least that many.
_MIN_NPX_LINES = 3


def _jobs(name: str) -> dict[str, Any]:
    spec = yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))
    return dict(spec["jobs"])


def _validate_steps(job: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    return [(i, step) for i, step in enumerate(job["steps"]) if step.get("name") == _STEP_NAME]


def _setup_node_index(job: dict[str, Any]) -> int:
    return next(
        i
        for i, step in enumerate(job["steps"])
        if str(step.get("uses", "")).startswith("actions/setup-node@")
    )


def npx_problems(workflow: dict[str, Any]) -> tuple[list[str], list[tuple[str, str]]]:
    """Every stripped non-empty, non-comment `run:` line that contains the token `npx` must
    match `_PINNED_NPX` as a whole-line prefix (a `#` line is shell commentary, not a command).
    Returns the problems and the (package, version) pairs found, in document order."""
    problems: list[str] = []
    pins: list[tuple[str, str]] = []
    for job_id, job in (workflow.get("jobs") or {}).items():
        for index, step in enumerate(job.get("steps") or []):
            for raw in str(step.get("run") or "").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or not re.search(r"\bnpx\b", line):
                    continue
                match = _PINNED_NPX.match(line)
                if match is None:
                    problems.append(
                        f"{job_id} step {index}: {line!r} is not `npx -y <package>@X.Y.Z ...` "
                        "(write npx on its own line, with an exact version)"
                    )
                    continue
                pins.append((match.group(1), match.group(2)))
    return problems, pins


def _workflow_files() -> list[Path]:
    return sorted([*_WORKFLOWS.glob("*.yml"), *_WORKFLOWS.glob("*.yaml")], key=lambda p: p.name)


def _load(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


# ---------------------------------------------------------------------------
# The real file
# ---------------------------------------------------------------------------
def test_the_typescript_sdk_job_has_exactly_one_validator_step_from_the_repository_root() -> None:
    job = _jobs(_CI)[_JOB]
    assert job["name"] == _JOB_NAME, "the job name is a check name and must not move"
    assert job["defaults"]["run"]["working-directory"] == _SDK_DIR, (
        "the step's comment explains the override; re-read it if the default moved"
    )
    found = _validate_steps(job)
    assert len(found) == 1, f"expected exactly one step named {_STEP_NAME!r}, found {found!r}"
    _index, step = found[0]
    assert step.get("working-directory") == _WORKSPACE, (
        "the validator must run from the repository root, where `.` is the marketplace"
    )
    assert step.get("env") == _EXPECTED_ENV, (
        "the validator must stay off the network beyond the package download"
    )
    assert "if" not in step, "the validator runs on every push and pull request"


def test_the_validator_step_runs_exactly_these_three_lines() -> None:
    job = _jobs(_CI)[_JOB]
    [(_index, step)] = _validate_steps(job)
    assert textwrap.dedent(step["run"]).strip() == _EXPECTED_RUN
    assert "continue-on-error" not in step and "continue-on-error" not in job
    assert "shell" not in step
    for fragment in _MASKING_FRAGMENTS:
        assert fragment not in step["run"], fragment


def test_the_validator_runs_after_node_is_set_up() -> None:
    # `node-version: "24"` on that step is test_github_app_node_runtime_pin.py's to assert.
    job = _jobs(_CI)[_JOB]
    [(index, _step)] = _validate_steps(job)
    assert _setup_node_index(job) < index


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_every_npx_in_every_workflow_is_pinned_to_an_exact_version(path: Path) -> None:
    assert npx_problems(_load(path))[0] == []


def test_the_only_npx_package_in_ci_is_claude_code_at_the_pinned_version() -> None:
    pins = [pin for path in _workflow_files() for pin in npx_problems(_load(path))[1]]
    assert len(pins) >= _MIN_NPX_LINES, "the parser found no npx lines; today ci.yml has three"
    assert set(pins) == {(_PACKAGE, EXPECTED_CLAUDE_CODE_VERSION)}


def test_the_pinned_version_is_exactly_three_integers() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", EXPECTED_CLAUDE_CODE_VERSION)
    assert _EXPECTED_RUN.count("@" + EXPECTED_CLAUDE_CODE_VERSION + " ") == 3


# ---------------------------------------------------------------------------
# The checkers bite: inline documents that break each rule
# ---------------------------------------------------------------------------
def _run_step(script: str) -> dict[str, Any]:
    return {"jobs": {"build": {"steps": [{"run": script}]}}}


class TestTheCheckersBite:
    @pytest.mark.parametrize(
        ("script", "pins"),
        [
            (
                "npx -y @anthropic-ai/claude-code@2.1.292 --version",
                [("@anthropic-ai/claude-code", "2.1.292")],
            ),
            ("npx --yes typescript@5.9.0 --version", [("typescript", "5.9.0")]),
            (
                "npx -y typescript@5.9.0 --version\n"
                "npx -y @anthropic-ai/claude-code@2.1.292 --version\n",
                [("typescript", "5.9.0"), ("@anthropic-ai/claude-code", "2.1.292")],
            ),
        ],
        ids=["scoped-package", "long-yes-flag", "two-lines-in-order"],
    )
    def test_exactly_pinned_lines_are_clean_and_their_pins_come_back_in_order(
        self, script: str, pins: list[tuple[str, str]]
    ) -> None:
        assert npx_problems(_run_step(script)) == ([], pins)

    @pytest.mark.parametrize(
        "script",
        [
            "npx -y @anthropic-ai/claude-code plugin validate --strict .",
            "npx -y @anthropic-ai/claude-code@latest plugin validate --strict .",
            "npx -y @anthropic-ai/claude-code@2.1 plugin validate --strict .",
            "npx -y @anthropic-ai/claude-code@^2.1.292 plugin validate --strict .",
            "npx -y @anthropic-ai/claude-code@2.1.292.1 plugin validate --strict .",
            "npx @anthropic-ai/claude-code@2.1.292 plugin validate --strict .",
            "cd packages && npx -y typescript@5.9.0 --version",
            "echo npx -y x@1.2.3",
        ],
        ids=[
            "no-version",
            "latest-tag",
            "two-components",
            "caret-range",
            "four-components",
            "missing-yes-flag",
            "compound-line",
            "not-the-first-token",
        ],
    )
    def test_an_unpinned_or_buried_npx_is_exactly_one_problem_and_no_pin(self, script: str) -> None:
        problems, pins = npx_problems(_run_step(script))
        assert len(problems) == 1, problems
        assert pins == []

    def test_a_step_without_run_and_a_job_without_steps_are_clean(self) -> None:
        checkout_only = {"jobs": {"build": {"steps": [{"uses": "actions/checkout@" + "0" * 40}]}}}
        reusable = {"jobs": {"call": {"uses": "org/repo/.github/workflows/x.yml@" + "0" * 40}}}
        assert npx_problems(checkout_only) == ([], [])
        assert npx_problems(reusable) == ([], [])

    def test_a_shell_comment_that_mentions_npx_is_not_a_command(self) -> None:
        """MUTATION that must make this fail: drop the `line.startswith("#")` clause."""
        script = "# npx is deliberately not used here\nnpm ci\n"
        assert npx_problems(_run_step(script)) == ([], [])
