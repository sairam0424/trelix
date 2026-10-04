"""The tag-triggered release workflows must be least-privilege, bounded and SHA-pinned.

WHY THIS EXISTS. `release.yml` and `docker-publish.yml` both fire on `push: tags: v*` and
hold the most sensitive credentials in the repo (PyPI trusted publishing, GHCR write). A
static triage found three gaps that no other test covers:

* `release.yml` had no workflow-level `permissions:` block, so the jobs that run
  third-party code on a tag (test, build-binaries, smoke-test) inherited whatever the
  repository default token scope is. If that default is read/write, a malicious dependency
  executing in one of them would hold a `contents: write` token.
* Neither file set `timeout-minutes`, so a hung job runs for the 360 minute default.
* The docker matrix lacked `fail-fast: false`, so one failing image variant cancelled the
  other mid-push and left the registry with half a release.

A fourth gap was found later: the job that builds what is uploaded to PyPI ran
`pip install build twine`, so a release took whatever version of either tool was newest on
the day of the tag. They are pinned to exact versions now, and `TestBuildToolsArePinned`
fails when either goes back to a range or to no version at all.

These tests read the workflow YAML rather than mocking Actions: the failure being
prevented is a *missing* key, and only the file can show whether it is there. The pinning
check re-asserts a property that already holds today so that it stays true.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_RELEASE = _WORKFLOWS / "release.yml"
_DOCKER = _WORKFLOWS / "docker-publish.yml"

_SHA_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
_BUILD_TOOLS = ("build", "twine")
_PIP_INSTALL_LINE = re.compile(r"\bpip3?\s+install\b.*")
_PUBLISH_WRITE_SCOPES = {"id-token", "contents"}


def _workflow(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def _jobs(path: Path) -> dict[str, dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = _workflow(path)["jobs"]
    return jobs


def _uses_refs(path: Path) -> list[tuple[str, str]]:
    """Every (location, ref) `uses:` in a workflow: job-level and step-level."""
    refs: list[tuple[str, str]] = []
    for job_name, job in _jobs(path).items():
        if "uses" in job:
            refs.append((f"{job_name} (job)", job["uses"]))
        for index, step in enumerate(job.get("steps") or []):
            if "uses" in step:
                refs.append((f"{job_name} step {index}", step["uses"]))
    return refs


class TestReleasePermissionsAreLeastPrivilege:
    def test_top_level_permissions_block_exists(self) -> None:
        assert "permissions" in _workflow(_RELEASE), (
            "release.yml needs a workflow-level `permissions:` block so jobs that do not "
            "declare their own do not inherit the repository default token scope"
        )

    def test_top_level_default_is_read_only(self) -> None:
        assert _workflow(_RELEASE)["permissions"] == {"contents": "read"}

    def test_only_publish_holds_write_or_oidc_scopes(self) -> None:
        """id-token (PyPI trusted publishing) and contents: write (GitHub Release) are the
        two scopes that must stay confined to the one job that uses them. Any other write
        scope, on any job, would be new and should be a deliberate, reviewed change."""
        for name, job in _jobs(_RELEASE).items():
            granted = job.get("permissions") or {}
            writes = {scope for scope, level in granted.items() if level == "write"}
            if name == "publish":
                assert writes == _PUBLISH_WRITE_SCOPES
            else:
                assert not writes, f"job '{name}' holds write scopes {sorted(writes)}"

    def test_docker_publish_job_keeps_its_explicit_scopes(self) -> None:
        permissions = _jobs(_DOCKER)["publish"]["permissions"]
        assert permissions == {"contents": "read", "packages": "write"}


class TestEveryJobIsBounded:
    @pytest.mark.parametrize("path", [_RELEASE, _DOCKER], ids=lambda p: p.name)
    def test_every_job_has_a_timeout(self, path: Path) -> None:
        for name, job in _jobs(path).items():
            timeout = job.get("timeout-minutes")
            assert isinstance(timeout, int) and timeout > 0, (
                f"{path.name}: job '{name}' has no timeout-minutes (GitHub's default is 360)"
            )


class TestDockerMatrixDoesNotCancelSiblings:
    def test_fail_fast_is_false(self) -> None:
        strategy = _jobs(_DOCKER)["publish"]["strategy"]
        assert strategy.get("fail-fast") is False, (
            "one failing image variant must not cancel the other mid-push"
        )


class TestEveryActionIsPinnedToACommit:
    @pytest.mark.parametrize("path", [_RELEASE, _DOCKER], ids=lambda p: p.name)
    def test_uses_refs_are_40_hex_shas_or_local(self, path: Path) -> None:
        refs = _uses_refs(path)
        assert refs, f"{path.name}: found no `uses:` references; the parser is wrong"
        for location, ref in refs:
            if ref.startswith("./"):
                continue
            assert _SHA_PIN.match(ref), f"{path.name}: {location} uses mutable ref '{ref}'"


def build_tool_problems(workflow: dict[str, Any]) -> list[str]:
    """`build` and `twine` must each carry an exact `==X.Y.Z` pin on a `pip install` line of a
    `run:` script. A removed install line is flagged the same way: a pin named on a line that
    does not install (an `echo`) does not count."""
    scripts = "\n".join(
        str(step.get("run", ""))
        for job in workflow["jobs"].values()
        for step in job.get("steps") or []
    )
    installs = " ".join(_PIP_INSTALL_LINE.findall(scripts))
    return [
        f"{tool} is not installed with an exact ==X.Y.Z pin"
        for tool in _BUILD_TOOLS
        if not re.search(rf"{tool}==\d+(\.\d+)*(?![\w.*])", installs)
    ]


def _run_step(script: str) -> dict[str, Any]:
    return {"jobs": {"build": {"steps": [{"run": script}]}}}


class TestBuildToolsArePinned:
    def test_release_installs_build_and_twine_at_exact_versions(self) -> None:
        assert build_tool_problems(_workflow(_RELEASE)) == []

    @pytest.mark.parametrize(
        "script",
        [
            "pip install build==1.6.1 twine==7.0.0",
            'python -m pip install --upgrade "build==1.6.1" "twine==7.0.0"',
        ],
    )
    def test_exact_pins_are_clean(self, script: str) -> None:
        assert build_tool_problems(_run_step(script)) == []

    @pytest.mark.parametrize(
        "script",
        [
            "pip install build twine",
            "pip install build==1.6.1 twine",
            'pip install "build>=1.6" twine==7.0.0',
            "pip install build==1.* twine==7.0.0",
            "echo build==1.6.1 twine==7.0.0",
        ],
        ids=["unpinned", "half-pinned", "range", "wildcard", "removed"],
    )
    def test_a_range_or_no_version_or_no_install_is_flagged(self, script: str) -> None:
        assert build_tool_problems(_run_step(script)) != []
