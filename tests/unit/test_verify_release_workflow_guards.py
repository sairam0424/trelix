"""verify-release.yml keeps the guards that make its `workflow_run` trigger acceptable.

SCANS (nothing more): the parsed YAML of .github/workflows/verify-release.yml. The job `if:` is
compared as whitespace-normalised text, not evaluated, and nothing runs zizmor or GitHub Actions.
The cache, trigger and `# zizmor: ignore` pins are in test_release_workflow_zizmor_fixes.py.

WHY. A `workflow_run` workflow runs with the base repository's token even when a fork started the
run it follows. This one is kept (zizmor: dangerous-triggers, accepted) only while its job `if:`
lets through just a `workflow_dispatch` or a successful `push` run of THIS repository whose
head_branch starts with `v`, the token is `contents: read` and `actions: read`, the checkout names
no `ref` or `repository` (default branch, never a fork), and event, input and step-output data
reach a `run:` script only through `env:`. The ignore on `workflow_run:` silences the whole `on:`
block, so the trigger set and the job set are pinned too.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit.workflow_yaml_helpers import action, load_workflow, steps, triggers

_DISPATCH = "github.event_name == 'workflow_dispatch'"
_RUN_CLAUSES = (
    "github.event_name == 'workflow_run'",
    "github.event.workflow_run.conclusion == 'success'",
    "github.event.workflow_run.event == 'push'",
    "github.event.workflow_run.repository.full_name == github.repository",
    "github.event.workflow_run.head_repository.full_name == github.repository",
    "startsWith(github.event.workflow_run.head_branch, 'v')",
)
_EXPECTED_IF = f"{_DISPATCH} || ( {' && '.join(_RUN_CLAUSES)} )"
_PERMISSIONS = {"contents": "read", "actions": "read"}


def set_problems(workflow: dict[str, Any]) -> list[str]:
    found = (sorted(triggers(workflow)), sorted(workflow["jobs"]))
    return [] if found == (["workflow_dispatch", "workflow_run"], ["verify"]) else [f"{found}"]


def if_problems(workflow: dict[str, Any]) -> list[str]:
    text = " ".join(str(workflow["jobs"]["verify"].get("if", "")).split())
    problems = [f"missing: {clause}" for clause in (_DISPATCH, *_RUN_CLAUSES) if clause not in text]
    if not problems and text != _EXPECTED_IF:
        problems.append(f"every clause is there but combined differently: {text}")
    return problems


def permission_problems(workflow: dict[str, Any]) -> list[str]:
    found = workflow["jobs"]["verify"].get("permissions")
    return [] if found == _PERMISSIONS else [f"job permissions are {found}"]


def checkout_problems(workflow: dict[str, Any]) -> list[str]:
    return [
        f"checkout names '{key}'"
        for step in steps(workflow)
        if action(step) == "actions/checkout"
        for key in ("ref", "repository")
        if key in (step.get("with") or {})
    ]


def run_problems(workflow: dict[str, Any]) -> list[str]:
    """Any `${{ }}` in a `run:` script: every value must arrive through `env:` instead."""
    return [
        f"run interpolates: {s['run']!r}" for s in steps(workflow) if "${{" in str(s.get("run"))
    ]


_CHECKS = [set_problems, if_problems, permission_problems, checkout_problems, run_problems]


@pytest.mark.parametrize("check", _CHECKS, ids=lambda check: check.__name__)
def test_verify_release_breaks_no_guard(check: Any) -> None:
    assert check(load_workflow("verify-release.yml")) == []


def _workflow(
    *,
    on: dict[str, Any] | None = None,
    if_: str = _EXPECTED_IF,
    permissions: dict[str, str] | None = _PERMISSIONS,
    checkout_with: dict[str, Any] | None = None,
    run: str = 'echo "$INPUT_VERSION"',
    other_jobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verify = {
        "if": if_,
        "permissions": permissions,
        "steps": [
            {"uses": "actions/checkout@v4", "with": checkout_with or {"fetch-depth": 0}},
            {"run": run},
        ],
    }
    return {
        "on": on or {"workflow_run": {}, "workflow_dispatch": {}},
        "jobs": {"verify": verify, **(other_jobs or {})},
    }


_BOTH = {"workflow_run": {}, "workflow_dispatch": {}}
_BITES = {
    "trigger-removed": (set_problems, {"on": {"workflow_run": {}}}),
    "trigger-added": (set_problems, {"on": {**_BOTH, "pull_request_target": {}}}),
    "second-job": (set_problems, {"other_jobs": {"other": {"steps": []}}}),
    **{
        f"if-clause-{n}-dropped": (if_problems, {"if_": _EXPECTED_IF.replace(clause, "true")})
        for n, clause in enumerate((_DISPATCH, *_RUN_CLAUSES))
    },
    "if-and-became-or": (if_problems, {"if_": _EXPECTED_IF.replace(" && ", " || ")}),
    "if-extra-alternative": (
        if_problems,
        {"if_": f"{_EXPECTED_IF} || github.event_name == 'push'"},
    ),
    "if-empty": (if_problems, {"if_": ""}),
    "permission-write": (
        permission_problems,
        {"permissions": {**_PERMISSIONS, "contents": "write"}},
    ),
    "permission-extra": (
        permission_problems,
        {"permissions": {**_PERMISSIONS, "packages": "read"}},
    ),
    "permission-missing": (permission_problems, {"permissions": None}),
    "checkout-ref": (checkout_problems, {"checkout_with": {"ref": "abc"}}),
    "checkout-repository": (checkout_problems, {"checkout_with": {"repository": "fork/x"}}),
    "run-event-input": (run_problems, {"run": "echo ${{ github.event.inputs.version }}"}),
    "run-input": (run_problems, {"run": "echo ${{ inputs.version }}"}),
    "run-bracket-spelling": (run_problems, {"run": "echo ${{ github['event'] }}"}),
    "run-step-output": (run_problems, {"run": "echo ${{ steps.resolve.outputs.version }}"}),
}


@pytest.mark.parametrize("check", _CHECKS, ids=lambda check: check.__name__)
def test_the_fixture_is_clean(check: Any) -> None:
    assert check(_workflow()) == []


@pytest.mark.parametrize(("check", "overrides"), _BITES.values(), ids=_BITES.keys())
def test_a_departure_is_flagged(check: Any, overrides: dict[str, Any]) -> None:
    assert check(_workflow(**overrides)) != []
