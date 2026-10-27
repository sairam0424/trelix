"""Editing a workflow that builds and runs the GitHub App must start the App CI.

WHY THIS EXISTS. `github-app-ci.yml` runs only when a path it lists changes. The
redelivery workflow (`redeliver-failed-webhooks.yml`) runs `npm ci`, `npm run build` and
a script from `infra/github-app`, so a broken edit to it was not built or tested by the
App CI until the next scheduled run. Both the `push` and the `pull_request` lists name
the App's own workflow files, and a file added to one list and not the other is the kind
of drift this catches. The expected names are written out below.
"""

from __future__ import annotations

import pytest

from tests.unit.workflow_yaml_helpers import load_workflow, triggers

_APP_WORKFLOW_PATHS = [
    ".github/workflows/github-app-ci.yml",
    ".github/workflows/redeliver-failed-webhooks.yml",
]


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_the_app_ci_runs_when_an_app_workflow_changes(event: str) -> None:
    paths = triggers(load_workflow("github-app-ci.yml"))[event]["paths"]

    missing = [path for path in _APP_WORKFLOW_PATHS if path not in paths]

    assert missing == []
