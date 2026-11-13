"""Every workflow declares a workflow-level `permissions:` block that is `{}` or `contents: read`.

WHY THIS EXISTS. OSSF Scorecard's Token-Permissions check takes half a point off for each
workflow that has no top-level `permissions:` key, even when every job declares its own (9 of 10
with the two files named below missing it): a job added later without one would inherit the
repository's default token scope, which this tree cannot see.
docker-publish.yml and trelix-review.yml were the two files without the key. A job that needs
more than read access re-declares the extra scope at job level (docker-publish's `publish` job
asks for `packages: write`), so this file pins only the default.

The files are read as YAML, not mocked: the failure being prevented is a missing key.
"""

from __future__ import annotations

import pytest

from tests.unit.workflow_yaml_helpers import WORKFLOWS, load_workflow

_NAMES = sorted(path.name for path in [*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])

# `{}` grants no scope at all; `contents: read` is the least a checkout needs.
_READ_ONLY_DEFAULTS = ({}, {"contents": "read"})

# A broken glob reports "nothing wrong" just as loudly as a clean tree; there are 15 workflows
# today, and the floor sits below that so that consolidating them does not trip it.
_MIN_WORKFLOWS = 10


def test_the_walk_reached_the_workflows() -> None:
    """PRECONDITION: an empty walk would make every case below vanish."""
    assert len(_NAMES) >= _MIN_WORKFLOWS, f"only {len(_NAMES)} workflow files found"


@pytest.mark.parametrize("name", _NAMES)
def test_workflow_declares_a_read_only_default_token(name: str) -> None:
    workflow = load_workflow(name)
    assert "permissions" in workflow, (
        f"{name} has no workflow-level `permissions:`; add `permissions: contents: read` "
        "and grant any extra scope on the job that needs it"
    )
    assert workflow["permissions"] in _READ_ONLY_DEFAULTS, (
        f"{name}: workflow-level permissions are {workflow['permissions']!r}; they must be "
        "`{}` or `contents: read`, with other scopes declared on the job that needs them"
    )
