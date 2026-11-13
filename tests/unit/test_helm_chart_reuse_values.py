"""The chart refuses the values of a chart 0.2.x release, and says what to do instead.

`helm upgrade --reuse-values` does not merge the old release's values into the new chart's
defaults: it REPLACES the defaults with them. A key the new chart added is then simply absent, and
chart 0.3.0's `networkpolicy.yaml` read `.Values.networkPolicy.enabled` through it and failed with
`nil pointer evaluating interface {}.enabled`. A render that had succeeded would have been worse:
it would keep the old release's `image.tag` and leave out the seccomp profile, the dropped
capabilities and the resource requests that 0.3.0 turns on. So the chart fails closed, in
`trelix.requireCurrentValues`, when the `networkPolicy` key is missing.

`helm template` has no `--reuse-values`, so these tests copy the chart without its `values.yaml`
and feed `tests/fixtures/helm/values-chart-0.2.8.yaml`, the frozen defaults of the 0.2.8 chart, as
the only values. That is the input Helm builds (`chart.Values = oldVals`, pkg/action/upgrade.go).
`--reset-then-reuse-values` keeps the new chart's defaults and puts the old values on top, which
is the real chart rendered with `-f` the same file.

These skip without a `helm` binary, like the rendered tests in test_helm_chart_hardening.py; the
`lint` job in helm-lint.yml renders the same input with the pinned helm.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART = REPO_ROOT / "helm" / "trelix"
OLD_VALUES = REPO_ROOT / "tests" / "fixtures" / "helm" / "values-chart-0.2.8.yaml"
HELM = shutil.which("helm")
_needs_helm = pytest.mark.skipif(HELM is None, reason="helm is required to render the chart")

_BACKENDS = ("sqlite", "lance", "qdrant")
_QDRANT_KEY = ("--set", "store.qdrant.apiKey=test-k")


def _template(chart: Path, *args: str) -> subprocess.CompletedProcess[str]:
    assert HELM is not None
    return subprocess.run(
        [HELM, "template", "test", str(chart), *_QDRANT_KEY, *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _render(chart: Path, *args: str) -> list[dict[str, Any]]:
    done = _template(chart, *args)
    assert done.returncode == 0, done.stderr
    return [doc for doc in yaml.safe_load_all(done.stdout) if doc]


def _kinds(docs: list[dict[str, Any]]) -> list[str]:
    return sorted(doc["kind"] for doc in docs)


def _pod(docs: list[dict[str, Any]]) -> dict[str, Any]:
    deployment = next(doc for doc in docs if doc["kind"] == "Deployment")
    return deployment["spec"]["template"]["spec"]


@pytest.fixture
def reused_chart(tmp_path: Path) -> Path:
    """The chart with its own defaults removed, so only the old release's values are left."""
    chart = tmp_path / "trelix"
    shutil.copytree(CHART, chart)
    (chart / "values.yaml").unlink()
    return chart


def test_the_fixture_is_the_frozen_defaults_of_chart_0_2_8() -> None:
    values = yaml.safe_load(OLD_VALUES.read_text(encoding="utf-8"))
    assert "networkPolicy" not in values  # the key chart 0.3.0 added: what the refusal looks for
    assert values["image"]["tag"] == "3.5.0"
    assert values["mcp"]["enabled"] is False  # 0.2.7 added mcp: its absence is not what fails here


@_needs_helm
@pytest.mark.parametrize("backend", _BACKENDS)
def test_old_release_values_are_refused_with_the_cause_and_the_fix(
    reused_chart: Path, backend: str
) -> None:
    done = _template(reused_chart, "-f", str(OLD_VALUES), "--set", f"store.backend={backend}")
    assert done.returncode != 0
    assert "nil pointer" not in done.stderr
    flat = " ".join(done.stderr.split())
    assert "The values predate chart 0.3.0 (they have no networkPolicy key)" in flat
    assert "helm upgrade --reuse-values replaces the chart's defaults" in flat
    assert "helm upgrade --reset-then-reuse-values (Helm 3.14 or newer)" in flat
    assert "or with --reset-values plus your overrides" in flat


@_needs_helm
@pytest.mark.parametrize("backend", _BACKENDS)
def test_the_same_values_with_the_network_policy_key_render(
    reused_chart: Path, backend: str
) -> None:
    """The refusal is the missing key and nothing else: add it and the old values render."""
    docs = _render(
        reused_chart,
        "-f",
        str(OLD_VALUES),
        "--set",
        f"store.backend={backend}",
        "--set",
        "networkPolicy.enabled=false",
    )
    kinds = _kinds(docs)
    for kind in ("Deployment", "PersistentVolumeClaim", "Service", "ServiceAccount"):
        assert kind in kinds, kind
    assert "NetworkPolicy" not in kinds


@_needs_helm
def test_reset_then_reuse_values_gives_the_new_defaults_under_the_old_values() -> None:
    """What `helm upgrade --reset-then-reuse-values` builds: the chart's own values.yaml first,
    the old release's overrides on top (the whole old values.yaml here, which is more than a
    release has as overrides). It renders, and the pod has the 0.3.0 hardening."""
    docs = _render(CHART, "-f", str(OLD_VALUES))
    pod = _pod(docs)
    assert pod["securityContext"] == {"fsGroup": 1000, "seccompProfile": {"type": "RuntimeDefault"}}
    assert pod["automountServiceAccountToken"] is False
    assert pod["containers"][0]["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
    }
    assert "NetworkPolicy" not in _kinds(docs)


@_needs_helm
@pytest.mark.parametrize(
    ("args", "policies"),
    [
        ((), 0),
        (("--set", "networkPolicy.enabled=false"), 0),
        (("--set", "networkPolicy.enabled=true"), 1),
    ],
)
def test_a_values_set_that_has_the_network_policy_key_is_not_refused(
    args: tuple[str, ...], policies: int
) -> None:
    docs = _render(CHART, *args)
    assert _kinds(docs).count("NetworkPolicy") == policies


@_needs_helm
def test_a_deleted_network_policy_key_is_refused_like_a_missing_one() -> None:
    """`null` deletes a chart default, which is the other way the key goes missing; it failed with
    the same nil pointer before the check."""
    done = _template(CHART, "--set", "networkPolicy=null")
    assert done.returncode != 0
    assert "nil pointer" not in done.stderr
    assert "The values predate chart 0.3.0" in " ".join(done.stderr.split())


def test_readme_says_what_reuse_values_does() -> None:
    flat = " ".join((CHART / "README.md").read_text(encoding="utf-8").split())
    assert "`helm upgrade --reuse-values` is refused with a message that names the fix" in flat
    assert "`--reset-then-reuse-values`, which needs Helm 3.14 or newer" in flat
    assert "would keep the old `image.tag` and leave out the hardening" in flat


def test_helm_lint_workflow_renders_the_old_release_values() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "helm-lint.yml").read_text(encoding="utf-8")
    )
    runs = [s["run"] for s in workflow["jobs"]["lint"]["steps"] if "run" in s]
    step = next(r for r in runs if "values-chart-0.2.8.yaml" in r)
    assert "cp -R helm/trelix/Chart.yaml helm/trelix/templates" in step  # no values.yaml
    assert "predate chart 0.3.0" in step
    # The step fails when the old values render, when the refusal lacks the fix, when it is a
    # nil pointer error again, and when the old values on top of the defaults do not render.
    assert 'rendered"; exit 1' in step
    assert "grep -qF 'helm upgrade --reset-then-reuse-values' <<< \"$ERR\"" in step
    assert 'nil pointer\' <<< "$ERR"' in step
    assert 'nil pointer error again"; exit 1' in step
    assert (
        "helm template test helm/trelix -f tests/fixtures/helm/values-chart-0.2.8.yaml > /dev/null"
        in step
    )
