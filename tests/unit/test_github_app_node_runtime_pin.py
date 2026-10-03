"""Every place the repo pins Node for the GitHub App's toolchain must say 24.

WHY THIS EXISTS. Node 20 reached end of life on 2026-04-30 and the App's image used to
install it from a NodeSource `curl | bash` script. The move to Node 24 (LTS, end of life
2028-04-30) touches five workflows, the Dockerfile, `engines`, `@types/node` and the
lockfile, and each is one stale line away from quietly running something else while every
test stays green: a workflow left on 20 still builds, and an `@types/node` of the wrong
major still type-checks. This reads those files, so the pin is the test's expected value
rather than something the test derives from them.

Each rule is a pure function over the file's content, and `TestTheCheckersBite` feeds every
one of them a document that breaks it, so a checker that stopped matching fails there
instead of leaving the real-file assertions green for the wrong reason.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
1. Any workflow's `node-version: "24"` -> `"20"`, or the line deleted
   -> TestEverySetupNodeStepIsNode24 fails naming the file and job.
2. Dockerfile: a `FROM node:24-bookworm-slim` -> `node:20-...`
   -> TestTheDockerfileRunsNode24 fails.
3. Dockerfile: put a NodeSource `curl | bash` step back, or drop the
   `COPY --from=node-builder /usr/local/bin/node` line
   -> TestTheDockerfileRunsNode24 fails.
4. `engines.node` back to `>=20`, `@types/node` to another major, or a lockfile that
   disagrees with package.json
   -> TestThePackageDeclaresNode24 fails.
5. github-app-ci.yml: drop the `node --version` step or the image's `v24.` assertion
   -> TestTheAppCiProvesTheVersion fails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_APP = _ROOT / "infra" / "github-app"
_DOCKERFILE = _APP / "Dockerfile"
_PACKAGE_JSON = _APP / "package.json"
_LOCKFILE = _APP / "package-lock.json"
_APP_CI = _WORKFLOWS / "github-app-ci.yml"

_SETUP_NODE_PREFIX = "actions/setup-node@"
_FROM_NODE = re.compile(r"^FROM\s+node:\S+", re.MULTILINE | re.IGNORECASE)
_PINNED_FROM_NODE = re.compile(r"^FROM node:24-bookworm-slim AS \w[\w-]*$", re.MULTILINE)
_NODE_BINARY_COPY = re.compile(
    r"^COPY --from=node-builder /usr/local/bin/node /usr/local/bin/node$", re.MULTILINE
)
_NODESOURCE = re.compile(r"nodesource|setup_\d+\.x", re.IGNORECASE)
_CURL_PIPED_TO_SHELL = re.compile(r"curl[^\n]*\|\s*(?:sudo\s+)?(?:ba|z)?sh\b", re.IGNORECASE)
_TYPES_NODE_SPEC = re.compile(r"^\^24\.\d+\.\d+$")

# A glob or parser that found nothing reports "no stale pin" as loudly as a clean tree.
# Five workflows use setup-node today (ci.yml has one such step); the floor sits below that.
_MIN_SETUP_NODE_STEPS = 5


# ---------------------------------------------------------------------------
# The rules: pure functions over a parsed document or file text, each returning
# its violations
# ---------------------------------------------------------------------------
def _steps(workflow: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    for job_name, job in (workflow.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            found.append((job_name, step))
    return found


def setup_node_steps(workflow: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (job, step)
        for job, step in _steps(workflow)
        if str(step.get("uses", "")).startswith(_SETUP_NODE_PREFIX)
    ]


def stale_node_pins(workflow: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for job, step in setup_node_steps(workflow):
        version = (step.get("with") or {}).get("node-version")
        if version is None:
            problems.append(f"job '{job}': setup-node has no node-version")
        elif str(version) != "24":
            problems.append(f"job '{job}': node-version is {version!r}, expected '24'")
    return problems


def _without_comments(dockerfile: str) -> str:
    """Instructions only: a comment may say why NodeSource and `curl | bash` are gone."""
    return "\n".join(line for line in dockerfile.splitlines() if not line.lstrip().startswith("#"))


def dockerfile_violations(dockerfile: str) -> list[str]:
    text = _without_comments(dockerfile)
    problems: list[str] = []
    from_lines = _FROM_NODE.findall(text)
    pinned = _PINNED_FROM_NODE.findall(text)
    if len(from_lines) < 2:
        problems.append(f"expected the two node build stages, found {len(from_lines)}")
    if len(pinned) != len(from_lines):
        problems.append(
            f"{len(from_lines) - len(pinned)} node stage(s) are not `FROM node:24-bookworm-slim`"
        )
    if not _NODE_BINARY_COPY.search(text):
        problems.append("the runtime stage does not COPY the node binary out of node-builder")
    if _NODESOURCE.search(text):
        problems.append("the Dockerfile still references NodeSource")
    if _CURL_PIPED_TO_SHELL.search(text):
        problems.append("the Dockerfile pipes curl into a shell")
    return problems


def package_violations(package: dict[str, Any], lock: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    engines = (package.get("engines") or {}).get("node")
    if engines != ">=24":
        problems.append(f"package.json engines.node is {engines!r}, expected '>=24'")
    types_spec = (package.get("devDependencies") or {}).get("@types/node", "")
    if not _TYPES_NODE_SPEC.match(types_spec):
        problems.append(f"package.json @types/node is {types_spec!r}, expected ^24.x.y")
    packages = lock.get("packages") or {}
    root = packages.get("") or {}
    if (root.get("engines") or {}).get("node") != engines:
        problems.append("package-lock.json root engines.node disagrees with package.json")
    if (root.get("devDependencies") or {}).get("@types/node") != types_spec:
        problems.append("package-lock.json root @types/node disagrees with package.json")
    locked = (packages.get("node_modules/@types/node") or {}).get("version", "")
    if not locked.startswith("24."):
        problems.append(f"package-lock.json locks @types/node {locked!r}, expected 24.x.y")
    return problems


def app_ci_violations(workflow: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    jobs = workflow.get("jobs") or {}
    test_runs = [str(s.get("run", "")) for s in (jobs.get("test") or {}).get("steps") or []]
    if not any("node --version" in run for run in test_runs):
        problems.append("the 'test' job does not print `node --version`")
    docker_runs = [
        str(s.get("run", "")) for s in (jobs.get("docker-build") or {}).get("steps") or []
    ]
    if not any("--entrypoint node" in run and "v24." in run for run in docker_runs):
        problems.append("the 'docker-build' job does not assert the image's node is v24.x")
    return problems


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def _workflow_files() -> list[Path]:
    return sorted([*_WORKFLOWS.glob("*.yml"), *_WORKFLOWS.glob("*.yaml")])


# ---------------------------------------------------------------------------
# The real files
# ---------------------------------------------------------------------------
class TestEverySetupNodeStepIsNode24:
    def test_the_walk_reached_the_setup_node_steps(self) -> None:
        found = sum(len(setup_node_steps(_load_yaml(p))) for p in _workflow_files())

        assert found >= _MIN_SETUP_NODE_STEPS

    @pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
    def test_no_setup_node_step_is_stale(self, path: Path) -> None:
        assert stale_node_pins(_load_yaml(path)) == []


class TestTheDockerfileRunsNode24:
    def test_the_pins_hold(self) -> None:
        assert dockerfile_violations(_DOCKERFILE.read_text(encoding="utf-8")) == []


class TestThePackageDeclaresNode24:
    def test_package_json_and_the_lockfile_agree_on_24(self) -> None:
        package = json.loads(_PACKAGE_JSON.read_text(encoding="utf-8"))
        lock = json.loads(_LOCKFILE.read_text(encoding="utf-8"))

        assert package_violations(package, lock) == []


class TestTheAppCiProvesTheVersion:
    def test_the_workflow_prints_and_asserts_it(self) -> None:
        assert app_ci_violations(_load_yaml(_APP_CI)) == []


# ---------------------------------------------------------------------------
# The checkers bite: a document that breaks each rule is flagged by that rule
# ---------------------------------------------------------------------------
_GOOD_DOCKERFILE = """\
FROM node:24-bookworm-slim AS node-deps
RUN npm ci --omit=dev
FROM node:24-bookworm-slim AS node-builder
RUN npm run build
FROM python:3.14-slim AS runtime
COPY --from=node-builder /usr/local/bin/node /usr/local/bin/node
"""

_GOOD_PACKAGE: dict[str, Any] = {
    "engines": {"node": ">=24"},
    "devDependencies": {"@types/node": "^24.19.1"},
}
_GOOD_LOCK: dict[str, Any] = {
    "packages": {
        "": {"engines": {"node": ">=24"}, "devDependencies": {"@types/node": "^24.19.1"}},
        "node_modules/@types/node": {"version": "24.19.1"},
    }
}

_GOOD_IMAGE_CHECK = 'v="$(docker run --entrypoint node img --version)"; case "$v" in v24.*) ;; esac'

_GOOD_APP_CI: dict[str, Any] = {
    "jobs": {
        "test": {"steps": [{"run": "node --version"}]},
        "docker-build": {"steps": [{"run": _GOOD_IMAGE_CHECK}]},
    }
}


def _workflow(version: Any) -> dict[str, Any]:
    with_block = {} if version is None else {"node-version": version}
    return {
        "jobs": {
            "build": {"steps": [{"uses": "actions/setup-node@" + "a" * 40, "with": with_block}]}
        }
    }


class TestTheCheckersBite:
    def test_the_good_fixtures_are_clean(self) -> None:
        assert stale_node_pins(_workflow("24")) == []
        assert dockerfile_violations(_GOOD_DOCKERFILE) == []
        assert package_violations(_GOOD_PACKAGE, _GOOD_LOCK) == []
        assert app_ci_violations(_GOOD_APP_CI) == []

    @pytest.mark.parametrize(
        "version",
        [
            pytest.param("20", id="quoted-20"),
            pytest.param("22", id="quoted-22"),
            pytest.param(20, id="unquoted-20"),
            pytest.param(26, id="unquoted-26"),
            pytest.param("lts/*", id="lts-alias"),
            pytest.param(None, id="no-node-version"),
        ],
    )
    def test_a_setup_node_step_not_on_24_is_flagged(self, version: Any) -> None:
        assert len(stale_node_pins(_workflow(version))) == 1

    def test_an_unquoted_24_is_still_24(self) -> None:
        assert stale_node_pins(_workflow(24)) == []

    @pytest.mark.parametrize(
        ("old", "new", "needle"),
        [
            pytest.param(
                "FROM node:24-bookworm-slim AS node-builder",
                "FROM node:20-bookworm-slim AS node-builder",
                "not `FROM node:24-bookworm-slim`",
                id="builder-on-20",
            ),
            pytest.param(
                "FROM node:24-bookworm-slim AS node-deps",
                "FROM node:24-slim AS node-deps",
                "not `FROM node:24-bookworm-slim`",
                id="wrong-variant",
            ),
            pytest.param(
                "COPY --from=node-builder /usr/local/bin/node /usr/local/bin/node\n",
                "",
                "does not COPY the node binary",
                id="copy-dropped",
            ),
            pytest.param(
                "COPY --from=node-builder /usr/local/bin/node /usr/local/bin/node\n",
                "RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash -\n",
                "NodeSource",
                id="nodesource-back",
            ),
            pytest.param(
                "COPY --from=node-builder /usr/local/bin/node /usr/local/bin/node\n",
                "RUN curl -fsSL https://example.invalid/install.sh | sh\n",
                "pipes curl into a shell",
                id="curl-pipe-shell",
            ),
        ],
    )
    def test_each_dockerfile_departure_is_flagged(self, old: str, new: str, needle: str) -> None:
        mutated = _GOOD_DOCKERFILE.replace(old, new)

        assert mutated != _GOOD_DOCKERFILE
        assert any(needle in problem for problem in dockerfile_violations(mutated))

    def test_a_comment_may_explain_the_removed_install(self) -> None:
        commented = (
            "# Replaces a NodeSource `curl -fsSL .../setup_20.x | bash -`.\n"
            "# FROM node:20-bookworm-slim AS old\n" + _GOOD_DOCKERFILE
        )

        assert dockerfile_violations(commented) == []

    def test_a_missing_build_stage_is_flagged(self) -> None:
        one_stage = (
            "FROM node:24-bookworm-slim AS node-builder\n" + _GOOD_DOCKERFILE.split("\n")[-2]
        )

        assert any("two node build stages" in p for p in dockerfile_violations(one_stage))

    @pytest.mark.parametrize(
        ("package", "lock", "needle"),
        [
            pytest.param(
                {**_GOOD_PACKAGE, "engines": {"node": ">=20"}},
                _GOOD_LOCK,
                "engines.node is",
                id="engines-20",
            ),
            pytest.param(
                {**_GOOD_PACKAGE, "devDependencies": {"@types/node": "^26.6.2"}},
                _GOOD_LOCK,
                "@types/node is",
                id="types-26",
            ),
            pytest.param(
                _GOOD_PACKAGE,
                {
                    "packages": {
                        **_GOOD_LOCK["packages"],
                        "node_modules/@types/node": {"version": "26.6.2"},
                    }
                },
                "locks @types/node",
                id="lock-on-26",
            ),
            pytest.param(
                _GOOD_PACKAGE,
                {
                    "packages": {
                        **_GOOD_LOCK["packages"],
                        "": {
                            "engines": {"node": ">=20"},
                            "devDependencies": {"@types/node": "^24.19.1"},
                        },
                    }
                },
                "root engines.node disagrees",
                id="lock-root-engines",
            ),
            pytest.param(
                _GOOD_PACKAGE,
                {
                    "packages": {
                        **_GOOD_LOCK["packages"],
                        "": {
                            "engines": {"node": ">=24"},
                            "devDependencies": {"@types/node": "^26.6.2"},
                        },
                    }
                },
                "root @types/node disagrees",
                id="lock-root-types",
            ),
        ],
    )
    def test_each_package_departure_is_flagged(
        self, package: dict[str, Any], lock: dict[str, Any], needle: str
    ) -> None:
        assert any(needle in problem for problem in package_violations(package, lock))

    def test_a_missing_print_step_is_flagged(self) -> None:
        workflow = {
            **_GOOD_APP_CI,
            "jobs": {**_GOOD_APP_CI["jobs"], "test": {"steps": [{"run": "npm ci"}]}},
        }

        assert any("print `node --version`" in p for p in app_ci_violations(workflow))

    def test_an_image_check_that_does_not_assert_24_is_flagged(self) -> None:
        workflow = {
            "jobs": {
                **_GOOD_APP_CI["jobs"],
                "docker-build": {"steps": [{"run": "docker run --entrypoint node img --version"}]},
            }
        }

        assert any("assert the image's node is v24.x" in p for p in app_ci_violations(workflow))
