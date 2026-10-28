"""The Helm chart's hardened defaults and its opt-in security settings.

Chart 0.3.0 turns on what cannot break a working install (seccomp RuntimeDefault, no privilege
escalation, all capabilities dropped, resource requests, no service-account token) and puts what
can behind one value each (read-only root, non-root ids, NetworkPolicy). This file pins both
halves, in the two ways the chart can be read:

* as text and YAML, which needs no tool: the values, `Chart.yaml`'s `kubeVersion`, and what
  `helm-lint.yml` says about Checkov (the skipped rules, each with a reason; `HELM_NAMESPACE`);
* rendered by `helm template`, in the default state and in each opt-in state. These skip without
  a `helm` binary, as `test_review_check_counts_every_finding.py` skips without `node`; the
  `lint` job in helm-lint.yml renders the same states with the pinned helm.

`test_mcp_deployment_contracts.py` pins the trelix-mcp side and the checksum annotation; the
render tests here also pin that none of the new output mentions mcp while it is off, since
helm-lint.yml fails any render that does.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART = REPO_ROOT / "helm" / "trelix"
HELM = shutil.which("helm")
_needs_helm = pytest.mark.skipif(HELM is None, reason="helm is required to render the chart")

_DEFAULT_SECURITY_CONTEXT = {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}}
_MCP_ARGS = ("--set", "mcp.enabled=true", "--set", "mcp.auth.token=test-k")
_RO_ARGS = ("--set", "securityContext.readOnlyRootFilesystem=true")


def _values() -> dict[str, Any]:
    return yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))


def _run_helm(*args: str, command: str = "template") -> subprocess.CompletedProcess[str]:
    assert HELM is not None
    return subprocess.run(  # noqa: S603
        [HELM, command, "test", str(CHART), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _render(*args: str) -> list[dict[str, Any]]:
    done = _run_helm(*args)
    assert done.returncode == 0, done.stderr
    return [doc for doc in yaml.safe_load_all(done.stdout) if doc]


def _notes(*args: str) -> str:
    """The notes `helm install` prints; `helm template` never renders NOTES.txt."""
    done = _run_helm("-n", "ns-x", "--dry-run=client", *args, command="install")
    assert done.returncode == 0, done.stderr
    return done.stdout.split("\nNOTES:\n", 1)[1]


def _render_error(*args: str) -> str:
    done = _run_helm(*args)
    assert done.returncode != 0, "the chart rendered, and should have refused"
    return done.stderr


def _of_kind(docs: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [doc for doc in docs if doc["kind"] == kind]


def _pod(docs: list[dict[str, Any]]) -> dict[str, Any]:
    return _of_kind(docs, "Deployment")[0]["spec"]["template"]["spec"]


def _containers(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in _pod(docs)["containers"]}


def _mounts(container: dict[str, Any]) -> dict[str, str]:
    return {m["name"]: m["mountPath"] for m in container.get("volumeMounts", [])}


def _env(container: dict[str, Any]) -> dict[str, str]:
    return {e["name"]: e.get("value", "") for e in container["env"]}


# ── values, Chart.yaml, docs: no helm needed ──────────────────────────────────────────


def test_values_ship_the_safe_defaults() -> None:
    values = _values()
    assert values["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
    }
    assert values["podSecurityContext"] == {
        "fsGroup": 1000,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    assert values["serviceAccount"]["automount"] is False
    assert values["resources"] == {"requests": {"cpu": "250m", "memory": "512Mi"}}


def test_values_leave_the_opt_in_settings_and_all_limits_off() -> None:
    values = _values()
    assert values["networkPolicy"] == {"enabled": False, "ingressFrom": []}
    for key in ("readOnlyRootFilesystem", "runAsNonRoot", "runAsUser", "runAsGroup"):
        assert key not in values["securityContext"], key
    assert "limits" not in values["resources"]


def test_chart_requires_kubernetes_1_19_and_accepts_suffixed_versions() -> None:
    chart = yaml.safe_load((CHART / "Chart.yaml").read_text(encoding="utf-8"))
    assert chart["kubeVersion"] == ">=1.19.0-0"


def _checkov_step_text() -> str:
    text = (REPO_ROOT / ".github" / "workflows" / "helm-lint.yml").read_text(encoding="utf-8")
    start = text.index("      - name: Checkov\n")
    return text[start : text.index("      - if: always()", start)]


def _checkov_step() -> dict[str, Any]:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "helm-lint.yml").read_text(encoding="utf-8")
    )
    return next(s for s in workflow["jobs"]["security"]["steps"] if s.get("name") == "Checkov")


def test_checkov_skips_exactly_the_rules_that_stay_open_by_design() -> None:
    """Eight rules, none of them one the chart fixes. A rule that is fixed must not be skipped:
    the skip would hide a regression of the fix."""
    skipped = _checkov_step()["with"]["skip_check"].split(",")
    assert skipped == [
        "CKV_K8S_11",
        "CKV_K8S_13",
        "CKV_K8S_15",
        "CKV_K8S_22",
        "CKV_K8S_23",
        "CKV_K8S_40",
        "CKV_K8S_43",
        "CKV2_K8S_6",
    ]


def test_every_skipped_checkov_rule_has_its_own_reason_comment() -> None:
    """`skip_check` is one comma-separated scalar, which cannot hold a comment, so the reasons sit
    above it, one `# <rule> (...): ...` line each, and each names the value that closes the rule."""
    step = _checkov_step_text()
    for rule in _checkov_step()["with"]["skip_check"].split(","):
        # A reason continues on lines indented five spaces past the `#`; the next rule's own
        # line is indented three, so it is not swallowed.
        reasons = re.findall(rf"^ +# +{rule} \([^)]+\): (.+(?:\n +# {{5,}}\S.*)*)", step, re.M)
        assert len(reasons) == 1, rule
        assert re.search(r"\bSet\b", reasons[0]), f"{rule}: the reason names no value to set"


def test_checkov_runs_with_a_helm_namespace_and_the_lint_job_does_not() -> None:
    """CKV_K8S_21 passes on a non-default HELM_NAMESPACE, so the chart needs no hard-coded
    namespace. The `lint` job must not set it: its allow-list test expects `.default`."""
    assert _checkov_step()["env"] == {"HELM_NAMESPACE": "trelix"}
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "helm-lint.yml").read_text(encoding="utf-8")
    )
    lint = workflow["jobs"]["lint"]
    assert "env" not in lint
    assert all("env" not in step for step in lint["steps"])
    assert "HELM_NAMESPACE" not in yaml.safe_dump(lint)


def test_helm_lint_workflow_renders_the_opt_in_settings() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "helm-lint.yml").read_text(encoding="utf-8")
    )
    runs = [s["run"] for s in workflow["jobs"]["lint"]["steps"] if "run" in s]
    step = next(r for r in runs if "--set networkPolicy.enabled=true" in r)
    for needle in (
        "--set securityContext.readOnlyRootFilesystem=true",
        "--set securityContext.runAsNonRoot=true",
        "--set securityContext.runAsUser=999",
        "--set securityContext.runAsGroup=999",
        "kind: NetworkPolicy",
        "mountPath: /tmp",
        "grep -qiF mcp",  # off: nothing of trelix-mcp may appear
        "--set mcp.enabled=true",
        "needs networkPolicy.ingressFrom",
        "'- port: 8766'",  # the mcp port is in the policy
        """grep -cF 'mountPath: /tmp' <<< "$OUT") == 2 ]]""",  # both containers get the mounts
    ):
        assert needle in step, needle


def test_readme_documents_the_hardening_and_installs_into_its_own_namespace() -> None:
    readme = (CHART / "README.md").read_text(encoding="utf-8")
    assert "helm install my-trelix helm/trelix -n trelix --create-namespace\n" in readme
    assert "\nhelm install my-trelix helm/trelix\n" not in readme
    for key in (
        "serviceAccount.automount",
        "podSecurityContext",
        "securityContext",
        "resources",
        "networkPolicy.enabled",
        "networkPolicy.ingressFrom",
    ):
        assert re.search(rf"^\| `{re.escape(key)}`", readme, re.M), key
    for heading in ("## Hardening", "### On by default", "### Sizing", "### Opt-in hardening"):
        assert f"\n{heading}\n" in readme, heading
    # The restore recipe uses null: Helm merges an empty map into the defaults and removes none.
    flat = " ".join(readme.split())
    assert (
        "`securityContext: null`, `resources: null` and `podSecurityContext.seccompProfile: null`"
    ) in flat
    assert "`--set securityContext=null`" in flat
    assert "an empty map (`securityContext: {}`) does not" in flat
    # With a read-only root the audit trail has no writable default path: the README must say so.
    assert "`TRELIX_AUDIT_DB_PATH` to a file on the PVC, for example `/data/audit.db`" in flat
    # Two ways the new defaults can make an upgrade fail, each with its way out.
    assert "set `resources.requests` at or below your limits" in flat
    assert "`--set podSecurityContext.seccompProfile=null` removes it" in flat


# ── rendered with helm: the default state ─────────────────────────────────────────────


@_needs_helm
def test_default_render_is_hardened_and_has_none_of_the_opt_in_features() -> None:
    docs = _render()
    pod = _pod(docs)
    container = _containers(docs)["trelix"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"] == {
        "fsGroup": 1000,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    assert container["securityContext"] == _DEFAULT_SECURITY_CONTEXT
    assert container["resources"] == {"requests": {"cpu": "250m", "memory": "512Mi"}}
    assert sorted(doc["kind"] for doc in docs) == [
        "Deployment",
        "PersistentVolumeClaim",
        "Service",
        "ServiceAccount",
    ]
    assert [v["name"] for v in pod["volumes"]] == ["data"]
    assert _mounts(container) == {"data": "/data"}
    assert "XDG_CACHE_HOME" not in _env(container)


@_needs_helm
def test_default_render_mentions_mcp_nowhere() -> None:
    done = _run_helm("--set", "mcp.auth.token=test-k")
    assert done.returncode == 0, done.stderr
    assert "mcp" not in done.stdout.lower()


@_needs_helm
def test_install_notes_name_the_release_namespace_in_every_kubectl_command() -> None:
    """Installed with `-n ns-x`, a bare `kubectl` command would look in the default namespace.
    Rendered in the two states that between them show every command of NOTES.txt."""
    token = ("--set", "mcp.auth.token=test-k", "--set", "mcp.enabled=true")
    first = _notes("--set", "apiAuth.existingSecretName=s", *token)
    assert len(re.findall(r"kubectl\b", first)) == 7
    assert re.findall(r"kubectl\b(?! -n ns-x )", first) == []
    assert "kubectl -n ns-x port-forward svc/test-trelix 8766:8766" in first
    assert "kubectl -n ns-x port-forward svc/test-trelix 8765:8765" in first
    second = _notes("--set", "service.type=LoadBalancer")
    assert len(re.findall(r"kubectl\b", second)) == 3
    assert re.findall(r"kubectl\b(?! -n ns-x )", second) == []
    assert "kubectl -n ns-x get svc -w test-trelix" in second


@_needs_helm
def test_automount_is_rendered_through_the_value_in_both_states() -> None:
    assert _pod(_render("--set", "serviceAccount.automount=true"))["automountServiceAccountToken"]
    assert _pod(_render())["automountServiceAccountToken"] is False


@_needs_helm
def test_the_hardened_defaults_reach_the_mcp_container_but_its_resources_stay_unset() -> None:
    containers = _containers(_render(*_MCP_ARGS))
    assert containers["trelix-mcp"]["securityContext"] == _DEFAULT_SECURITY_CONTEXT
    assert containers["trelix"]["securityContext"] == _DEFAULT_SECURITY_CONTEXT
    assert "resources" not in containers["trelix-mcp"]
    assert "resources" in containers["trelix"]


@_needs_helm
def test_a_limit_is_added_to_the_requests_not_in_place_of_them() -> None:
    resources = _containers(_render("--set", "resources.limits.memory=4Gi"))["trelix"]["resources"]
    assert resources == {
        "requests": {"cpu": "250m", "memory": "512Mi"},
        "limits": {"memory": "4Gi"},
    }


@_needs_helm
def test_the_documented_restore_recipe_removes_every_new_default(tmp_path: Path) -> None:
    """`null` deletes a default key; the README and the CHANGELOG tell operators to use it."""
    restore = tmp_path / "restore.yaml"
    restore.write_text(
        "serviceAccount:\n  automount: true\n"
        "securityContext: null\nresources: null\n"
        "podSecurityContext:\n  seccompProfile: null\n",
        encoding="utf-8",
    )
    docs = _render("-f", str(restore))
    pod = _pod(docs)
    container = _containers(docs)["trelix"]
    assert pod["automountServiceAccountToken"] is True
    assert pod["securityContext"] == {"fsGroup": 1000}
    assert "securityContext" not in container
    assert "resources" not in container


@_needs_helm
def test_the_chart_refuses_a_cluster_older_than_1_19_and_takes_suffixed_versions() -> None:
    assert "1.19" in _render_error("--kube-version", "v1.18.0")
    assert _render("--kube-version", "v1.29.1-gke.100")


# ── rendered with helm: read-only root filesystem ─────────────────────────────────────


@_needs_helm
def test_read_only_root_mounts_writable_scratch_dirs_in_every_container() -> None:
    docs = _render(*_RO_ARGS, *_MCP_ARGS)
    pod = _pod(docs)
    assert len(pod["containers"]) == 2
    for container in pod["containers"]:
        assert container["securityContext"]["readOnlyRootFilesystem"] is True
        assert container["securityContext"]["allowPrivilegeEscalation"] is False
        assert _mounts(container) == {"data": "/data", "tmp": "/tmp", "cache": "/cache"}
        assert _env(container)["XDG_CACHE_HOME"] == "/cache"
    volumes = {v["name"]: v for v in pod["volumes"]}
    assert list(volumes) == ["data", "tmp", "cache"]
    assert volumes["tmp"]["emptyDir"] == {}
    assert volumes["cache"]["emptyDir"] == {}


@_needs_helm
def test_read_only_root_without_a_volume_still_gets_its_scratch_dirs() -> None:
    docs = _render(*_RO_ARGS, "--set", "persistence.enabled=false")
    pod = _pod(docs)
    assert [v["name"] for v in pod["volumes"]] == ["tmp", "cache"]
    assert _mounts(_containers(docs)["trelix"]) == {"tmp": "/tmp", "cache": "/cache"}


@_needs_helm
@pytest.mark.parametrize(
    "setting",
    [
        "securityContext.readOnlyRootFilesystem=false",
        "securityContext=null",
        "securityContext={}",
    ],
)
def test_no_scratch_dirs_unless_read_only_root_is_on(setting: str) -> None:
    """`null` leaves securityContext nil and `={}` makes it a list; neither may fail the render."""
    docs = _render("--set", setting)
    container = _containers(docs)["trelix"]
    assert [v["name"] for v in _pod(docs)["volumes"]] == ["data"]
    assert _mounts(container) == {"data": "/data"}
    assert "XDG_CACHE_HOME" not in _env(container)


@_needs_helm
def test_read_only_root_leaves_no_mention_of_mcp_while_mcp_is_off() -> None:
    done = _run_helm(*_RO_ARGS, "--set", "mcp.auth.token=test-k")
    assert done.returncode == 0, done.stderr
    assert "readOnlyRootFilesystem: true" in done.stdout
    assert "mcp" not in done.stdout.lower()


# ── rendered with helm: non-root ids ──────────────────────────────────────────────────


@_needs_helm
def test_non_root_ids_reach_every_container_next_to_the_safe_defaults() -> None:
    args = (
        "--set",
        "securityContext.runAsNonRoot=true",
        "--set",
        "securityContext.runAsUser=999",
        "--set",
        "securityContext.runAsGroup=999",
    )
    for container in _pod(_render(*args, *_MCP_ARGS))["containers"]:
        assert container["securityContext"] == {
            **_DEFAULT_SECURITY_CONTEXT,
            "runAsNonRoot": True,
            "runAsUser": 999,
            "runAsGroup": 999,
        }


# ── rendered with helm: NetworkPolicy ─────────────────────────────────────────────────


@_needs_helm
def test_network_policy_is_a_single_ingress_policy_on_the_pods_labels() -> None:
    docs = _render("--set", "networkPolicy.enabled=true")
    policy = _of_kind(docs, "NetworkPolicy")
    assert len(policy) == 1
    # A name that ignores the release would make two releases in one namespace collide, and
    # labels that ignore it would hide the policy from `kubectl get -l`.
    assert policy[0]["metadata"]["name"] == "test-trelix"
    assert policy[0]["metadata"]["labels"]["app.kubernetes.io/name"] == "trelix"
    assert policy[0]["metadata"]["labels"]["app.kubernetes.io/instance"] == "test"
    spec = policy[0]["spec"]
    selector = _of_kind(docs, "Deployment")[0]["spec"]["selector"]["matchLabels"]
    assert selector == {"app.kubernetes.io/name": "trelix", "app.kubernetes.io/instance": "test"}
    assert spec["podSelector"] == {"matchLabels": selector}
    assert spec["policyTypes"] == ["Ingress"]
    assert "egress" not in spec
    assert spec["ingress"] == [
        {"from": [{"podSelector": {}}], "ports": [{"port": 8765, "protocol": "TCP"}]}
    ]


@_needs_helm
def test_network_policy_follows_the_service_port_and_adds_the_mcp_port_only_when_on() -> None:
    docs = _render("--set", "networkPolicy.enabled=true", "--set", "service.port=9000")
    assert _of_kind(docs, "NetworkPolicy")[0]["spec"]["ingress"][0]["ports"] == [
        {"port": 9000, "protocol": "TCP"}
    ]
    docs = _render("--set", "networkPolicy.enabled=true", *_MCP_ARGS)
    ports = _of_kind(docs, "NetworkPolicy")[0]["spec"]["ingress"][0]["ports"]
    assert ports == [{"port": 8765, "protocol": "TCP"}, {"port": 8766, "protocol": "TCP"}]


@_needs_helm
def test_network_policy_admits_the_listed_peers_after_the_release_namespace(
    tmp_path: Path,
) -> None:
    values = tmp_path / "peers.yaml"
    values.write_text(
        "networkPolicy:\n  enabled: true\n  ingressFrom:\n"
        "    - namespaceSelector:\n        matchLabels:\n"
        "          kubernetes.io/metadata.name: ingress-nginx\n"
        "    - ipBlock:\n        cidr: 10.0.0.0/8\n"
        "ingress:\n  enabled: true\n",
        encoding="utf-8",
    )
    docs = _render("-f", str(values))
    assert _of_kind(docs, "NetworkPolicy")[0]["spec"]["ingress"][0]["from"] == [
        {"podSelector": {}},
        {"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "ingress-nginx"}}},
        {"ipBlock": {"cidr": "10.0.0.0/8"}},
    ]
    assert len(_of_kind(docs, "Ingress")) == 1


@_needs_helm
@pytest.mark.parametrize(
    "exposure",
    ["ingress.enabled=true", "service.type=LoadBalancer", "service.type=NodePort"],
)
def test_network_policy_refuses_to_cut_off_an_exposed_service_without_peers(
    exposure: str,
) -> None:
    error = _render_error("--set", "networkPolicy.enabled=true", "--set", exposure)
    assert "needs networkPolicy.ingressFrom" in error


@_needs_helm
def test_an_ingress_alone_renders_no_network_policy() -> None:
    docs = _render("--set", "ingress.enabled=true")
    assert _of_kind(docs, "NetworkPolicy") == []
    assert len(_of_kind(docs, "Ingress")) == 1


@_needs_helm
def test_every_opt_in_on_together_still_leaves_no_mention_of_mcp() -> None:
    done = _run_helm(
        *_RO_ARGS,
        "--set",
        "securityContext.runAsNonRoot=true",
        "--set",
        "securityContext.runAsUser=999",
        "--set",
        "networkPolicy.enabled=true",
        "--set",
        "mcp.auth.token=test-k",
    )
    assert done.returncode == 0, done.stderr
    assert "kind: NetworkPolicy" in done.stdout
    assert "mcp" not in done.stdout.lower()
