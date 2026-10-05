"""verify-release.yml keeps the guards that make its `workflow_run` trigger acceptable.

SCANS (nothing more): the parsed YAML of .github/workflows/verify-release.yml. The job `if:` is
compared as whitespace-normalised text, not evaluated, and nothing runs zizmor or GitHub Actions.
The release.yml cache, trigger and `# zizmor: ignore` pins are in
test_release_workflow_zizmor_fixes.py.

WHY. A `workflow_run` workflow runs with the base repository's token even when a fork started the
run it follows. This one is kept (zizmor: dangerous-triggers, accepted) only while its job `if:`
lets through just a `workflow_dispatch` or a successful `push` run of THIS repository whose
head_branch starts with `v`, the token is `contents: read` and `actions: read`, the checkout names
no `ref` or `repository` (default branch, never a fork), and event, input and step-output data
reach a `run:` script only through `env:`. The one credential is `github.token`: the `secrets`
context appears nowhere, and no step saves or restores a cache (a `workflow_run` job that read
a cache could run what another run wrote). The ignore on `workflow_run:` silences the whole `on:`
block, so the trigger set, the exact list of workflows it watches and the job set are pinned too.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from tests.unit.workflow_yaml_helpers import action, load_workflow, saves_a_cache, steps, triggers

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
_WATCHED = ["Release", "Docker Publish"]
_SECRETS_CONTEXT = re.compile(r"\$\{\{(?:(?!\}\}).)*?\bsecrets\b", re.IGNORECASE)


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


def watch_problems(workflow: dict[str, Any]) -> list[str]:
    """The `workflows:` list of the `workflow_run` trigger: exactly the two publish workflows."""
    found = (triggers(workflow).get("workflow_run") or {}).get("workflows")
    return [] if found == _WATCHED else [f"workflow_run watches {found}"]


def secret_problems(workflow: dict[str, Any]) -> list[str]:
    """Any `${{ }}` that reads the `secrets` context, in a step or in a job or workflow `env:`."""
    match = _SECRETS_CONTEXT.search(json.dumps(workflow, default=str))
    return [f"uses the secrets context: {match.group(0)}"] if match else []


def cache_problems(workflow: dict[str, Any]) -> list[str]:
    """No `actions/cache` step of any kind and no setup action with its cache switched on."""
    return [
        f"{step.get('name') or step.get('uses')}: {action(step)} uses a cache"
        for step in steps(workflow)
        if action(step).startswith("actions/cache") or saves_a_cache(step)
    ]


_CHECKS = [
    set_problems,
    if_problems,
    permission_problems,
    checkout_problems,
    run_problems,
    watch_problems,
    secret_problems,
    cache_problems,
]


@pytest.mark.parametrize("check", _CHECKS, ids=lambda check: check.__name__)
def test_verify_release_breaks_no_guard(check: Any) -> None:
    assert check(load_workflow("verify-release.yml")) == []


def test_the_watched_names_are_the_names_of_the_two_publish_workflows() -> None:
    """A rename of either workflow would stop verify-release from running after it."""
    names = [load_workflow(file)["name"] for file in ("release.yml", "docker-publish.yml")]
    assert names == _WATCHED


def _workflow(
    *,
    on: dict[str, Any] | None = None,
    if_: str = _EXPECTED_IF,
    permissions: dict[str, str] | None = _PERMISSIONS,
    checkout_with: dict[str, Any] | None = None,
    run: str = 'echo "$INPUT_VERSION"',
    other_jobs: dict[str, Any] | None = None,
    extra_steps: tuple[dict[str, Any], ...] = (),
    job_env: dict[str, str] | None = None,
    workflow_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    verify = {
        "if": if_,
        "permissions": permissions,
        **({"env": job_env} if job_env else {}),
        "steps": [
            {"uses": "actions/checkout@v4", "with": checkout_with or {"fetch-depth": 0}},
            {"run": run},
            *extra_steps,
        ],
    }
    return {
        "on": on or _BOTH,
        **({"env": workflow_env} if workflow_env else {}),
        "jobs": {"verify": verify, **(other_jobs or {})},
    }


_BOTH = {"workflow_run": {"workflows": _WATCHED}, "workflow_dispatch": {}}
_FROM_SECRETS = "${{ secrets.NAME }}"


def _watching(workflows: list[str]) -> dict[str, Any]:
    return {**_BOTH, "workflow_run": {"workflows": workflows}}


def _with_step(step: dict[str, Any]) -> dict[str, Any]:
    return {"extra_steps": (step,)}


_BITES = {
    "trigger-removed": (set_problems, {"on": {"workflow_run": {"workflows": _WATCHED}}}),
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
    "watch-list-shrunk": (watch_problems, {"on": _watching(["Release"])}),
    "watch-list-grown": (watch_problems, {"on": _watching([*_WATCHED, "CI"])}),
    "watch-list-other-name": (watch_problems, {"on": _watching(["Release", "Docker"])}),
    "watch-list-missing": (watch_problems, {"on": {**_BOTH, "workflow_run": {}}}),
    "secret-in-step-env": (secret_problems, _with_step({"env": {"V": _FROM_SECRETS}, "run": "x"})),
    "secret-in-step-with": (
        secret_problems,
        _with_step({"uses": "x/y@v1", "with": {"input": _FROM_SECRETS}}),
    ),
    "secret-in-run": (secret_problems, _with_step({"run": f"echo {_FROM_SECRETS}"})),
    "secret-bracket-spelling": (
        secret_problems,
        _with_step({"env": {"V": "${{ secrets['NAME'] }}"}, "run": "x"}),
    ),
    "secret-upper-case": (
        secret_problems,
        _with_step({"env": {"V": "${{ SECRETS.NAME }}"}, "run": "x"}),
    ),
    "secret-mixed-case": (
        secret_problems,
        _with_step({"env": {"V": "${{ Secrets.NAME }}"}, "run": "x"}),
    ),
    "secret-whole-context": (
        secret_problems,
        _with_step({"env": {"V": "${{ toJSON(secrets) }}"}, "run": "x"}),
    ),
    "secret-after-a-brace": (
        secret_problems,
        _with_step({"env": {"V": "${{ format('{0}', secrets.NAME) }}"}, "run": "x"}),
    ),
    "secret-in-job-env": (secret_problems, {"job_env": {"V": _FROM_SECRETS}}),
    "secret-in-workflow-env": (secret_problems, {"workflow_env": {"V": _FROM_SECRETS}}),
    "cache-action": (cache_problems, _with_step({"uses": "actions/cache@v4"})),
    "cache-mixed-case": (cache_problems, _with_step({"uses": "Actions/Cache@v4"})),
    "cache-save": (cache_problems, _with_step({"uses": "actions/cache/save@v4"})),
    "cache-restore": (cache_problems, _with_step({"uses": "actions/cache/restore@v4"})),
    "setup-python-cache": (
        cache_problems,
        _with_step({"uses": "actions/setup-python@v5", "with": {"cache": "pip"}}),
    ),
}


_LOOK_ALIKES = {
    "token-expression-and-the-word-secrets": (
        secret_problems,
        _with_step({"env": {"TOKEN": "${{ github.token }}"}, "run": "echo no secrets here"}),
    ),
}


@pytest.mark.parametrize("check", _CHECKS, ids=lambda check: check.__name__)
def test_the_fixture_is_clean(check: Any) -> None:
    assert check(_workflow()) == []


@pytest.mark.parametrize(("check", "overrides"), _LOOK_ALIKES.values(), ids=_LOOK_ALIKES.keys())
def test_a_look_alike_is_not_flagged(check: Any, overrides: dict[str, Any]) -> None:
    assert check(_workflow(**overrides)) == []


@pytest.mark.parametrize(("check", "overrides"), _BITES.values(), ids=_BITES.keys())
def test_a_departure_is_flagged(check: Any, overrides: dict[str, Any]) -> None:
    assert check(_workflow(**overrides)) != []
