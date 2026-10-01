"""Shipped deployment shapes the Host/Origin guard must not break (and the one it must set).

* The Dockerfile HEALTHCHECK requests `http://127.0.0.1:8765/health` and the Helm chart's
  kubelet probes send `Host: <pod-ip>:<port>` to `/health`. The guard exempts exactly
  `GET|HEAD /health`, so these targets have to stay on that exact path.
* `docker-compose.yml` runs `serve /repo --host 0.0.0.0`, which is a non-loopback bind, so
  the serve-level rule leaves the guard off. The compose file therefore opts in through
  `TRELIX_API_ALLOWED_HOSTS` for the loopback-published port.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from trelix.api.request_guard import resolve_allowed_hosts

REPO_ROOT = Path(__file__).resolve().parents[2]
_EXEMPT_PATH = "/health"


def _compose_environment() -> dict[str, str]:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    entries = compose["services"]["trelix"]["environment"]
    if isinstance(entries, dict):
        return {k: "" if v is None else str(v) for k, v in entries.items()}
    out: dict[str, str] = {}
    for entry in entries:
        name, _, value = str(entry).partition("=")
        out[name] = value
    return out


def test_compose_sets_the_allowed_hosts_variable_for_the_trelix_service() -> None:
    env = _compose_environment()
    assert "TRELIX_API_ALLOWED_HOSTS" in env


def test_compose_default_allows_the_three_loopback_names() -> None:
    value = _compose_environment()["TRELIX_API_ALLOWED_HOSTS"]
    match = re.fullmatch(r"\$\{TRELIX_API_ALLOWED_HOSTS:-(?P<default>[^}]*)\}", value)
    assert match, f"expected a ${{VAR:-default}} interpolation, got {value!r}"
    hosts = resolve_allowed_hosts(None, match["default"])
    assert hosts == frozenset({"localhost", "127.0.0.1", "::1"})


def test_compose_still_binds_all_interfaces_inside_the_container() -> None:
    """If this ever changes to a loopback bind the env var becomes redundant; say so loudly."""
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    command = compose["services"]["trelix"]["command"]
    assert command[command.index("--host") + 1] == "0.0.0.0"  # noqa: S104


def test_compose_publishes_on_host_loopback_only() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    ports = compose["services"]["trelix"]["ports"]
    assert ports == ["127.0.0.1:8765:8765"]


def test_dockerfile_healthcheck_targets_the_exempt_path() -> None:
    text = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    healthcheck = text[text.index("HEALTHCHECK") :].split("ENTRYPOINT")[0]
    urls = re.findall(r"https?://[^\s'\"]+", healthcheck)
    assert urls == [f"http://127.0.0.1:8765{_EXEMPT_PATH}"]


@pytest.mark.parametrize("probe", ["livenessProbe", "readinessProbe"])
def test_helm_probes_use_the_exempt_path(probe: str) -> None:
    text = (REPO_ROOT / "helm/trelix/templates/deployment.yaml").read_text(encoding="utf-8")
    block = re.search(rf"{probe}:\s*\n\s*httpGet:\s*\n\s*path:\s*(\S+)", text)
    assert block, f"{probe} httpGet block not found"
    assert block.group(1) == _EXEMPT_PATH


@pytest.mark.parametrize("doc", ["docs/INSTALLATION_GUIDE.md", "docs/FAQ.md"])
def test_every_documented_docker_run_serve_example_turns_the_guard_on(doc: str) -> None:
    """A container binds 0.0.0.0, where the guard is off unless the variable is set.

    The two docs showed the shape that bypassed the guard until this test existed, so every
    `docker run ... serve ... --host 0.0.0.0` block must pass the variable.
    """
    text = (REPO_ROOT / doc).read_text(encoding="utf-8")
    blocks = re.findall(r"docker run[^`]*?serve[^`]*?--host 0\.0\.0\.0[^`]*", text)
    assert blocks, f"no docker run serve example found in {doc}"
    for block in blocks:
        assert "TRELIX_API_ALLOWED_HOSTS=" in block, f"{doc}: example lacks the variable:\n{block}"


@pytest.mark.parametrize("doc", ["docs/INSTALLATION_GUIDE.md", "docs/FAQ.md", "SECURITY.md"])
def test_the_documented_allowed_hosts_value_with_brackets_is_quoted(doc: str) -> None:
    """zsh (the macOS default shell) treats an unquoted `[::1]` as a glob and refuses the
    command with "no matches found" before docker starts, so the value must be quoted."""
    text = (REPO_ROOT / doc).read_text(encoding="utf-8")
    examples = re.findall(r"-e\s+(\S*TRELIX_API_ALLOWED_HOSTS=\S*\[\S*)", text)
    assert examples, f"no bracketed -e TRELIX_API_ALLOWED_HOSTS example found in {doc}"
    for example in examples:
        assert example.startswith(("'", '"')), f"{doc}: unquoted value {example!r}"


def test_helm_chart_does_not_set_the_allowed_hosts_variable() -> None:
    """Documented only: a ClusterIP service is not browser-reachable, kubelet probes are exempt."""
    for path in (REPO_ROOT / "helm/trelix").rglob("*"):
        if path.is_file():
            assert "TRELIX_API_ALLOWED_HOSTS" not in path.read_text(encoding="utf-8")
