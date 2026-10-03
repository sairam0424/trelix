"""Runs the publishing script of ``.github/workflows/trelix-review.yml`` under node.

The script is inline JavaScript for ``actions/github-script``, so there is nothing to import:
this reads it out of the workflow, points its two fixed ``/tmp`` paths at a test directory and
runs it in an ``AsyncFunction`` with a recording fake ``github`` and a ``context`` shaped like a
``pull_request`` event (``context.sha`` is the merge commit, the PR page shows ``head.sha``).

It is shared by ``test_review_not_run_exit.py`` (the shared case table, the exit codes) and
``test_review_publish_incomplete.py`` (the record of unreviewed hunks: hostile file names,
missing and corrupt records, time bounds). Not collected: no ``test_`` prefix.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/trelix-review.yml"
# The table the GitHub App's tests run too (infra/github-app/tests/review-outcome.test.ts).
CASES_FILE = REPO_ROOT / "infra/github-app/tests/fixtures/review-conclusion-cases.json"

PUBLISH_STEP = "Post review as Check annotations"
REVIEW_FILE_PATH = "/tmp/trelix-review.json"
OUTCOME_FILE_PATH = "/tmp/trelix-review-outcome.json"
NODE = shutil.which("node")

# Prints the recorded `checks.create` calls and the CPU time the script itself used, so that a
# time bound is not fooled by node's start-up or by a busy machine.
_HARNESS = """
const fs = require('fs');
const [scriptFile] = process.argv.slice(2);
const script = fs.readFileSync(scriptFile, 'utf8');
const calls = [];
const github = { rest: { checks: { create: async (args) => { calls.push(args); } } } };
// sha is the merge commit of a pull_request event; the PR page shows head.sha.
const context = {
  repo: { owner: 'o', repo: 'r' },
  sha: 'merge000',
  payload: { pull_request: { head: { sha: 'head111' } } },
};
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const before = process.cpuUsage();
new AsyncFunction('github', 'context', 'require', 'process', script)(
  github, context, require, process
).then(() => {
  const used = process.cpuUsage(before);
  console.log(JSON.stringify({ calls, cpuMs: (used.user + used.system) / 1000 }));
});
"""


def load_cases() -> dict[str, Any]:
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


def workflow_steps() -> list[dict[str, Any]]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["trelix-review"]["steps"]


def workflow_step(step_id_or_name: str) -> dict[str, Any]:
    for step in workflow_steps():
        if step.get("id") == step_id_or_name or step.get("name") == step_id_or_name:
            return step
    raise AssertionError(f"no step {step_id_or_name!r} in {WORKFLOW}")


def publish_script() -> str:
    return str(workflow_step(PUBLISH_STEP)["with"]["script"])


@dataclass(frozen=True)
class Published:
    """The one Check the script created, and the CPU time it took."""

    call: dict[str, Any]
    cpu_ms: float
    stdout: str


def run_publish_script(
    tmp_path: Path,
    exit_code: str | None,
    review_output: str | None,
    *,
    outcome: str | None = None,
    script: str | None = None,
    before_run: Callable[[Path], None] | None = None,
) -> Published:
    """Run the publishing script as the workflow would after the review step.

    `review_output` is the content of the review step's stdout file, or None for no file at
    all; `outcome` is the content of the outcome record, or None for no file at all. `script`
    substitutes a different script (the time-bound mutation check uses it); the default is the
    workflow's own. `before_run` is called with the record's path once the files are in place,
    to make it something other than a plain file (a directory, a symlink).
    """
    review_file = tmp_path / "trelix-review.json"
    if review_output is not None:
        review_file.write_text(review_output, encoding="utf-8")
    outcome_file = tmp_path / "trelix-review-outcome.json"
    if outcome is not None:
        outcome_file.write_text(outcome, encoding="utf-8")
    if before_run is not None:
        before_run(outcome_file)
    source = script if script is not None else publish_script()
    for fixed, actual in (
        (REVIEW_FILE_PATH, review_file),
        (OUTCOME_FILE_PATH, outcome_file),
    ):
        assert fixed in source, f"the script no longer reads the fixed path {fixed}"
        source = source.replace(fixed, str(actual))
    script_file = tmp_path / "script.js"
    script_file.write_text(source, encoding="utf-8")
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "TRELIX_PR_NUMBER": "7"}
    if exit_code is not None:
        env["TRELIX_REVIEW_EXIT_CODE"] = exit_code
    assert NODE is not None
    proc = subprocess.run(  # noqa: S603
        [NODE, str(harness), str(script_file)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert len(result["calls"]) == 1, result["calls"]
    return Published(call=result["calls"][0], cpu_ms=result["cpuMs"], stdout=proc.stdout)


def findings_json(severities: list[str]) -> str:
    """The review step's stdout for findings with these severities, one file each."""
    return json.dumps(
        [
            {"file": f"src/f{i}.py", "lines": f"{i + 1}-{i + 1}", "severity": s, "comment": "c"}
            for i, s in enumerate(severities)
        ]
    )


def stdout_text(case: dict[str, Any]) -> str:
    """The review step's stdout for a table row: `stdout_text` when the row has one (output that
    is not a findings array), else the findings its `severities` describe."""
    if "stdout_text" in case:
        return str(case["stdout_text"])
    return findings_json(case["severities"])


def outcome_text(kind: str | None, cases: dict[str, Any]) -> str | None:
    """The outcome record for a table row: the valid one, none, or one that is not JSON."""
    if kind is None or kind == "missing":
        return None
    if kind == "valid":
        return json.dumps(cases["valid_outcome"])
    if kind == "corrupt":
        return str(cases["corrupt_outcome_text"])
    raise AssertionError(f"unknown outcome kind {kind!r} in {CASES_FILE}")
