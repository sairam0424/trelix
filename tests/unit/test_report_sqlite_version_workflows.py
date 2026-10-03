"""Pins the CI steps that run scripts/report_sqlite_version.py.

The report is informational, so what these tests protect is the shape of its call sites:

  * every interpreter and image the plan names is measured: the unit matrix, the slim and
    `-local` Docker images, the GitHub App image and the binary build interpreters;
  * the steps hide nothing: no `continue-on-error`, no `|| true`. The script exits 0 on
    every path (tests/unit/test_report_sqlite_version.py), so masking could only ever hide
    a real break in the script itself;
  * the job display names, which are the required check names, do not change, and no step
    sits in a job that overrides the working directory (the commands are repo-root
    relative).

The expected commands are written out as literals on purpose: a second statement of what
the workflows must say, so an edit to either side fails here first.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

_WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"

_SLIM = "docker run --rm -i --entrypoint python trelix:ci-test - < scripts/report_sqlite_version.py"
_LOCAL = (
    "docker run --rm -i --entrypoint python trelix:ci-test-local - "
    "< scripts/report_sqlite_version.py"
)
_APP = (
    "docker run --rm -i --entrypoint python trelix-github-app:ci - "
    "< scripts/report_sqlite_version.py"
)
_LOCAL_IF = "steps.docker-relevant-changes.outputs.changed == 'true'"

# (workflow, job id) -> (job display name = the check name, [(run command, step `if`)])
_EXPECTED: dict[tuple[str, str], tuple[str, list[tuple[str, str | None]]]] = {
    ("ci.yml", "test"): (
        "Test (Python ${{ matrix.python-version }})",
        [("python scripts/report_sqlite_version.py", None)],
    ),
    ("ci.yml", "docker-build"): (
        "Docker build smoke test",
        [(_SLIM, None), (_LOCAL, _LOCAL_IF)],
    ),
    ("github-app-ci.yml", "docker-build"): ("Docker build", [(_APP, None)]),
    ("build-binaries.yml", "build"): (
        "Build trelix (${{ matrix.os }})",
        [
            (
                "source .venv-build/bin/activate\npython scripts/report_sqlite_version.py",
                "runner.os == 'macOS'",
            ),
            ("python scripts/report_sqlite_version.py", "runner.os != 'macOS'"),
        ],
    ),
}

# The step each report must come after: the one that installs or builds what it measures.
_PROVIDER: dict[tuple[str, str], Callable[[dict[str, Any]], bool]] = {
    ("ci.yml", "test"): lambda step: (
        "pip install -e" in (step.get("run") or "") and "nomic-code" in step["run"]
    ),
    ("github-app-ci.yml", "docker-build"): lambda step: (
        "-t trelix-github-app:ci" in (step.get("run") or "")
    ),
    ("build-binaries.yml", "build"): lambda step: (
        step.get("name") == "Install dependencies (Windows/Linux)"
    ),
}

_MASKING_FRAGMENTS = ("|| true", "|| :", "|| exit 0", "; true", "&& true", "set +e")


def _jobs(workflow: str) -> dict[str, Any]:
    spec = yaml.safe_load((_WORKFLOWS / workflow).read_text(encoding="utf-8"))
    return dict(spec["jobs"])


def _report_steps(job: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, step)
        for index, step in enumerate(job["steps"])
        if "report_sqlite_version.py" in (step.get("run") or "")
    ]


@pytest.mark.parametrize(("workflow", "job_id"), list(_EXPECTED))
def test_each_listed_job_runs_exactly_the_expected_report_steps(workflow: str, job_id: str) -> None:
    _name, expected = _EXPECTED[(workflow, job_id)]
    found = [
        (textwrap.dedent(step["run"]).strip(), step.get("if"))
        for _index, step in _report_steps(_jobs(workflow)[job_id])
    ]
    assert found == expected


@pytest.mark.parametrize(("workflow", "job_id"), list(_EXPECTED))
def test_job_display_names_are_unchanged_because_they_are_check_names(
    workflow: str, job_id: str
) -> None:
    job = _jobs(workflow)[job_id]
    assert job["name"] == _EXPECTED[(workflow, job_id)][0]
    assert "defaults" not in job  # the commands use repo-root-relative paths


@pytest.mark.parametrize(("workflow", "job_id"), list(_PROVIDER))
def test_reports_come_after_the_step_that_provides_what_they_measure(
    workflow: str, job_id: str
) -> None:
    job = _jobs(workflow)[job_id]
    provider = next(i for i, step in enumerate(job["steps"]) if _PROVIDER[(workflow, job_id)](step))
    for index, _step in _report_steps(job):
        assert index > provider


def test_slim_and_local_image_reports_come_after_their_own_builds() -> None:
    job = _jobs("ci.yml")["docker-build"]
    built = {
        step["with"]["tags"]: index
        for index, step in enumerate(job["steps"])
        if str(step.get("uses", "")).startswith("docker/build-push-action")
    }
    assert set(built) == {"trelix:ci-test", "trelix:ci-test-local"}
    reported = {step["run"].strip(): index for index, step in _report_steps(job)}
    assert reported[_SLIM] > built["trelix:ci-test"]
    assert reported[_LOCAL] > built["trelix:ci-test-local"]


def test_no_workflow_hides_a_failure_of_the_report_script() -> None:
    checked = 0
    for workflow_file in sorted(_WORKFLOWS.glob("*.yml")):
        for job_id, job in _jobs(workflow_file.name).items():
            for _index, step in _report_steps(job):
                checked += 1
                where = f"{workflow_file.name}:{job_id}"
                assert "continue-on-error" not in step, where
                assert "continue-on-error" not in job, where
                assert "shell" not in step, where
                for fragment in _MASKING_FRAGMENTS:
                    assert fragment not in step["run"], f"{where}: {fragment}"
    # No reporting step exists outside the table above, so a new call site has to be added
    # to it (and so to this review) rather than appearing unpinned.
    assert checked == sum(len(steps) for _name, steps in _EXPECTED.values())
