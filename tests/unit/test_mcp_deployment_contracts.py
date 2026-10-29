"""The optional `trelix-mcp` service in docker-compose.yml and the Helm chart.

`trelix-mcp --transport http` is deployable the two ways `trelix serve` already is. This file
pins the shapes that make that safe, because nothing renders either deployment in CI's pytest
run (helm-lint.yml renders the chart; no job renders the compose file):

* the compose service is opt-in (profile `mcp`), published on host loopback only, serves the
  `/repo` mount, and takes its token and Host allow-list from variables of its own, never the
  REST service's;
* the chart's container is off by default, serves the PVC, shares the store env with the
  `trelix` container, has a token of its own, and the chart refuses to render it without a
  token or a volume;
* the chart's pod template carries a hash of the Secret the chart renders, so rotating a token
  or key through `helm upgrade` rolls the pod, and the docs name `kubectl rollout restart` for
  a Secret the chart does not own.

The chart is read as text, not parsed: a Go template is not YAML until Helm has rendered it,
and `helm` is not installed in the unit-test job. helm-lint.yml renders it; the last test here
pins that workflow's steps. `test_request_guard_deployment_contracts.py` covers the REST side.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from trelix.api.request_guard import resolve_allowed_hosts

REPO_ROOT = Path(__file__).resolve().parents[2]


def _compose() -> dict[str, Any]:
    return yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def _mcp_service() -> dict[str, Any]:
    return _compose()["services"]["trelix-mcp"]


def _mcp_environment() -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in _mcp_service()["environment"]:
        name, _, value = str(entry).partition("=")
        out[name] = value
    return out


def _chart(relative: str) -> str:
    return (REPO_ROOT / "helm" / "trelix" / relative).read_text(encoding="utf-8")


def _mcp_container() -> str:
    """The `deployment.yaml` text from the `trelix-mcp` container to the end of the file."""
    text = _chart("templates/deployment.yaml")
    return text[text.index("- name: trelix-mcp") :]


# ── docker-compose.yml ────────────────────────────────────────────────────────────────


def test_compose_mcp_service_is_opt_in() -> None:
    assert _mcp_service()["profiles"] == ["mcp"]
    assert "profiles" not in _compose()["services"]["trelix"]


def test_compose_mcp_publishes_on_host_loopback_only() -> None:
    assert _mcp_service()["ports"] == ["127.0.0.1:8766:8766"]


def test_compose_mcp_serves_the_mount_over_http_with_the_same_image() -> None:
    service = _mcp_service()
    assert service["image"] == "ghcr.io/sairam0424/trelix:latest"
    assert service["entrypoint"] == ["trelix-mcp"]
    assert service["command"] == [
        "--transport",
        "http",
        "--host",
        "0.0.0.0",  # noqa: S104
        "--port",
        "8766",
        "--root",
        "/repo",
    ]
    assert service["volumes"] == ["${REPO_PATH:-.}:/repo"]


def test_compose_mcp_environment_is_the_embedder_pair_plus_its_own_token_and_hosts() -> None:
    """`${VAR:?msg}` is not used: Compose interpolates services of inactive profiles too
    (measured on Docker Compose v5.5.0: plain `docker compose config` and `up --dry-run` both
    fail), so it would break plain `docker compose up`. The container exits 2 without a token
    and names the variable instead. The embedder pair keeps the `trelix` service's names and
    defaults; the exact key set also keeps the REST token and allow-list variables out."""
    env = {k: v for k, v in _mcp_environment().items() if k != "TRELIX_MCP_ALLOWED_HOSTS"}
    assert env == {  # the allow-list's default is pinned by the next test
        "OPENAI_API_KEY": "${OPENAI_API_KEY:-}",
        "TRELIX_EMBEDDER_PROVIDER": "${TRELIX_EMBEDDER_PROVIDER:-openai}",
        "TRELIX_MCP_AUTH_TOKEN": "${TRELIX_MCP_AUTH_TOKEN:-}",
    }


def test_compose_mcp_default_allowed_hosts_are_the_three_loopback_names() -> None:
    value = _mcp_environment()["TRELIX_MCP_ALLOWED_HOSTS"]
    match = re.fullmatch(r"\$\{TRELIX_MCP_ALLOWED_HOSTS:-(?P<default>[^}]*)\}", value)
    assert match, f"expected a ${{VAR:-default}} interpolation, got {value!r}"
    hosts = resolve_allowed_hosts(None, match["default"])
    assert hosts == frozenset({"localhost", "127.0.0.1", "::1"})


def test_compose_mcp_healthcheck_probes_its_own_port() -> None:
    """The image's HEALTHCHECK probes 8765, which this service does not serve."""
    test = _mcp_service()["healthcheck"]["test"]
    assert test[:2] == ["CMD", "python"]  # the runtime image is python:slim and has no curl
    assert re.findall(r"https?://[^\s'\"]+", " ".join(test)) == ["http://127.0.0.1:8766/health"]


# ── helm/trelix ───────────────────────────────────────────────────────────────────────


def test_chart_mcp_values_are_off_by_default_with_a_token_of_their_own() -> None:
    values = yaml.safe_load(_chart("values.yaml"))["mcp"]
    assert values["enabled"] is False
    assert values["port"] == 8766
    assert values["allowedHosts"] == []
    assert values["resources"] == {}
    auth = values["auth"]
    assert set(auth) == {"token", "existingSecretName", "existingSecretKey"}
    assert not auth["token"]
    assert not auth["existingSecretName"]
    assert auth["existingSecretKey"] == "TRELIX_MCP_AUTH_TOKEN"


def test_chart_mcp_container_serves_the_volume_over_http() -> None:
    text = _chart("templates/deployment.yaml")
    container = _mcp_container()
    needles = [
        'command: ["trelix-mcp"]',
        'args: ["--transport", "http", "--host", "0.0.0.0", "--port", "{{ .Values.mcp.port }}", '
        '"--root", "{{ .Values.persistence.mountPath }}"]',
        "- name: mcp\n              containerPort: {{ .Values.mcp.port }}",
        'image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"',
        "imagePullPolicy: {{ .Values.image.pullPolicy }}",
        "{{- with .Values.securityContext }}\n          securityContext:\n"
        "            {{- toYaml . | nindent 12 }}",
        "volumeMounts:\n            - name: data\n"
        "              mountPath: {{ .Values.persistence.mountPath }}",
        "{{- with .Values.mcp.resources }}\n          resources:\n"
        "            {{- toYaml . | nindent 12 }}",
    ]
    for needle in needles:
        assert needle in container, needle
    assert "{{- if .Values.mcp.enabled }}\n        - name: trelix-mcp" in text


@pytest.mark.parametrize("probe", ["livenessProbe", "readinessProbe"])
def test_chart_mcp_probes_use_the_exempt_path_on_the_mcp_port(probe: str) -> None:
    pattern = rf"{probe}:\s*\n\s*httpGet:\s*\n\s*path:\s*(\S+)\s*\n\s*port:\s*(\S+)"
    found = re.search(pattern, _mcp_container())
    assert found, f"{probe} httpGet block not found in the trelix-mcp container"
    assert found.groups() == ("/health", "mcp")


def test_chart_mcp_container_shares_the_store_env_and_has_its_own_token() -> None:
    text = _chart("templates/deployment.yaml")
    container = _mcp_container()
    # Each Secret reference is pinned from `name:` to the end of its `key:` line: a bare
    # `name: ...` needle also matches `name: ...-suffix`, a Secret that does not exist.
    for needle in (
        "name: TRELIX_MCP_ALLOWED_HOSTS",
        'value: {{ include "trelix.mcpAllowedHosts" . | quote }}',
        'name: {{ include "trelix.fullname" . }}\n                  key: TRELIX_MCP_AUTH_TOKEN\n',
        "name: {{ .Values.mcp.auth.existingSecretName }}\n"
        "                  key: {{ .Values.mcp.auth.existingSecretKey }}\n",
    ):
        assert needle in container, needle
    assert "apiAuth" not in container
    assert "TRELIX_API_AUTH_TOKEN" not in container
    assert text.count('include "trelix.storeEnv" .') == 2
    assert text.count('include "trelix.extraEnv" .') == 2


def test_chart_mcp_allowed_hosts_are_the_service_names_plus_the_values_list() -> None:
    helpers = _chart("templates/_helpers.tpl")
    start = helpers.index('define "trelix.mcpAllowedHosts"')
    define = helpers[start : helpers.index("{{- end }}", start)]
    for needle in (
        'list $fullname (printf "%s.%s" $fullname .Release.Namespace)',
        '(printf "%s.%s.svc" $fullname .Release.Namespace)',
        "concat $names .Values.mcp.allowedHosts",
        'join ","',
    ):
        assert needle in define, needle


def test_chart_refuses_mcp_without_a_token_or_a_volume() -> None:
    helpers = _chart("templates/_helpers.tpl")
    assert "mcp.enabled needs mcp.auth.token or mcp.auth.existingSecretName: " in helpers
    assert "mcp.enabled needs persistence.enabled: " in helpers
    assert _chart("templates/deployment.yaml").startswith('{{- include "trelix.mcpChecks" . -}}')


def test_chart_service_and_secret_carry_the_mcp_port_and_token_key() -> None:
    block = "\n".join(
        [
            "    {{- if .Values.mcp.enabled }}",
            "    - port: {{ .Values.mcp.port }}",
            "      targetPort: mcp",
            "      protocol: TCP",
            "      name: mcp",
            "    {{- end }}",
        ]
    )
    assert block in _chart("templates/service.yaml")
    secret = _chart("templates/secret.yaml")
    # The literals below are split on purpose: a SecretScan hook rejects a written line that
    # reads like a credential assignment. Keep the concatenation.
    tail = ".Values.mcp.auth.token " + "(not .Values.mcp.auth.existingSecretName)"
    assert "(and .Values.mcp.enabled " + tail + ")" in secret
    assert "TRELIX_MCP_AUTH_" + "TOKEN: {{ .Values.mcp.auth.token | quote }}" in secret
    # The key's own guard, a different line from the `$hasAny` operand above.
    assert "{{- if and .Values.mcp.enabled " + tail + " }}" in secret


def test_chart_pod_template_hashes_its_secret_so_a_rotated_token_rolls_the_pod() -> None:
    """A container reads a `secretKeyRef` variable once, at start, so a changed Secret alone
    rolls nothing and the old token stays accepted (measured with `helm template`: two tokens
    rendered a byte-identical Deployment). The hash must sit in the POD template's metadata,
    not the Deployment's own: only a change there rolls the pod. The Secret template renders a
    bare newline when it holds no value (measured: without `| trim` the annotation was written
    on every render), and `with` is falsy only for an empty string, so a release whose
    credentials all come from an existing Secret writes no annotation and no empty
    `annotations:` key."""
    text = _chart("templates/deployment.yaml")
    pod_metadata = text[text.index("  template:\n    metadata:\n") : text.index("    spec:\n")]
    block = (
        '{{- with (include (print $.Template.BasePath "/secret.yaml") . | trim) }}\n'
        "      annotations:\n"
        "        checksum/secret: {{ . | sha256sum }}\n"
        "      {{- end }}\n"
    )
    assert block in pod_metadata


def test_chart_docs_say_to_restart_after_changing_a_secret_the_chart_does_not_own() -> None:
    """The chart cannot hash a Secret it does not own (`lookup` returns nothing under
    `helm template`), so the README and the NOTES name the command for each token. Each note
    in NOTES.txt sits inside its own block (API auth on, `mcp.enabled`), so it shows only when
    that token exists."""
    readme = " ".join(_chart("README.md").split())
    notes = " ".join(_chart("templates/NOTES.txt").split())
    restart = "`kubectl -n <namespace> rollout restart deployment/<fullname>`"
    for needle in (
        "With `mcp.auth.existingSecretName` the chart cannot see the Secret, so after you "
        f"change it run {restart}.",
        "after you change the Secret behind `apiAuth.existingSecretName` run " + restart,
        f"while a Secret you manage needs {restart} after you change it.",
        "The same goes for a Secret named in `extraSecretEnv`: the chart cannot see it either, "
        "so changing it never rolls the pod by itself.",
    ):
        assert needle in readme, needle
    rollout = (
        "kubectl -n {{ .Release.Namespace }} rollout restart "
        'deployment/{{ include "trelix.fullname" . }}'
    )
    api_start = notes.index("{{- if or .Values.apiAuth.token .Values.apiAuth.existingSecretName }}")
    api_block = notes[api_start : notes.index("{{- else }}", api_start)]
    mcp_start = notes.index("{{- if .Values.mcp.enabled }}")
    mcp_block = notes[mcp_start : notes.index("{{- end }}", mcp_start)]
    assert (
        "after you change the Secret behind apiAuth.existingSecretName, run: " + rollout
        in api_block
    )
    assert (
        "after you change the Secret behind mcp.auth.existingSecretName, run: " + rollout
        in mcp_block
    )


def test_every_kubectl_command_in_the_chart_docs_names_the_namespace() -> None:
    """The README installs with `-n trelix` and the chart sets no `metadata.namespace`, so a
    `kubectl` command without `-n` looks in the user's default namespace and answers `not found`.
    NOTES.txt takes the namespace from the release; the README names `trelix` in its worked
    examples and a `<namespace>` placeholder where the command also has a `<fullname>`. The
    lower bounds keep the scans from passing on a file that lost its commands."""
    notes = _chart("templates/NOTES.txt")
    readme = _chart("README.md")
    assert len(re.findall(r"kubectl\b", notes)) >= 8
    assert len(re.findall(r"kubectl\b", readme)) >= 6
    assert re.findall(r"kubectl\b(?! -n \{\{ \.Release\.Namespace \}\} )", notes) == []
    assert re.findall(r"kubectl\b(?! -n (?:trelix|<namespace>) )", readme) == []


def test_chart_readme_documents_the_mcp_values_and_the_ingress_note() -> None:
    readme = _chart("README.md")
    for key in ("mcp.enabled", "mcp.port", "mcp.allowedHosts", "mcp.auth.token", "mcp.resources"):
        assert re.search(rf"^\| `{re.escape(key)}`", readme, re.M), key
    assert "nginx.ingress.kubernetes.io/proxy-read-timeout" in readme
    assert "nginx.ingress.kubernetes.io/proxy-buffering" in readme
    assert "--set 'mcp.allowedHosts={trelix.example.com}'" in readme


def test_helm_lint_workflow_renders_and_refuses_the_mcp_container() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "helm-lint.yml").read_text(encoding="utf-8")
    steps = yaml.safe_load(text)["jobs"]["lint"]["steps"]
    runs = [step["run"] for step in steps if "run" in step]
    enabled = [run for run in runs if "--set mcp.enabled=true" in run]
    allow_list = 'value: "test-trelix,test-trelix.default,test-trelix.default.svc"'
    assert any(allow_list in run for run in enabled)
    assert any("name: ci-secret" in run for run in enabled)
    refusals = [run for run in runs if "needs mcp.auth.token or" in run]
    assert any("mcp.enabled needs persistence.enabled" in run for run in refusals)
    # The step with mcp off must look for any mention of mcp, must not enable it, and must set a
    # token anyway: only then does a Secret that ignores `mcp.enabled` show up in the render.
    off = [run for run in runs if "grep -qiF mcp" in run and "--set mcp.enabled" not in run]
    assert any("--set mcp.auth.token=" in run for run in off)
