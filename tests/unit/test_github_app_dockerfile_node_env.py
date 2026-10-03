"""The GitHub App image must run with NODE_ENV=production, and only in its runtime stage.

Express reads NODE_ENV when the app is built: left unset it means "development", and its
default error handler then answers an unhandled error with the stack trace. The App has its
own final error handler (`infra/github-app/src/error-handler.ts`), so the variable is the
second layer, but it still has to be there.

It must not be set earlier in the file. The `node-builder` stage runs `npm ci` and then `tsc`,
and `tsc` is a devDependency: with NODE_ENV=production in force, `npm ci` skips it and the
build fails only in the Docker build, which no other test runs. Neither mistake shows in the
TypeScript suite, so this reads the Dockerfile itself, the same way
`test_github_app_env_contract.py` does for the walker flag.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

_DOCKERFILE = Path(__file__).resolve().parents[2] / "infra" / "github-app" / "Dockerfile"

# `FROM [--platform=...] image [AS stage]`: the flags are optional and there may be several.
_FROM = re.compile(r"^FROM\s+(?:--\S+\s+)*\S+(?:\s+AS\s+(?P<stage>\S+))?\s*$", re.IGNORECASE)
_INSTRUCTION = re.compile(r"^(?P<kind>ENV|ARG)\s+(?P<args>.+)$", re.IGNORECASE)

_VARIABLE = "NODE_ENV"


def _logical_lines(text: str) -> list[str]:
    """Dockerfile lines with comments dropped and backslash continuations joined."""
    lines: list[str] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not pending and stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        lines.append((pending + stripped).strip())
        pending = ""
    if pending:
        lines.append(pending.strip())
    return lines


def _assignments(args: str) -> list[tuple[str, str]]:
    """`k=v k2=v2`, or the legacy `k v` form, as (name, value) pairs."""
    first = args.split(None, 1)[0]
    if "=" not in first:
        name, _, value = args.partition(" ")
        return [(name, value.strip().strip("\"'"))]
    pairs: list[tuple[str, str]] = []
    for token in shlex.split(args):
        name, separator, value = token.partition("=")
        if separator:
            pairs.append((name, value))
    return pairs


def node_env_settings(dockerfile: str) -> list[tuple[str, str, str]]:
    """Every (instruction, stage, value) that sets NODE_ENV, by ENV or by ARG."""
    found: list[tuple[str, str, str]] = []
    stage = ""
    for line in _logical_lines(dockerfile):
        from_match = _FROM.match(line)
        if from_match:
            stage = from_match.group("stage") or ""
            continue
        instruction = _INSTRUCTION.match(line)
        if not instruction:
            continue
        kind = instruction.group("kind").upper()
        for name, value in _assignments(instruction.group("args")):
            if name == _VARIABLE:
                found.append((kind, stage, value))
    return found


def stage_names(dockerfile: str) -> list[str]:
    names: list[str] = []
    for line in _logical_lines(dockerfile):
        from_match = _FROM.match(line)
        if from_match:
            names.append(from_match.group("stage") or "")
    return names


def test_node_env_is_production_in_the_runtime_stage_only() -> None:
    settings = node_env_settings(_DOCKERFILE.read_text(encoding="utf-8"))

    assert settings == [("ENV", "runtime", "production")], (
        "infra/github-app/Dockerfile must set `ENV NODE_ENV=production` once, in the "
        f"`runtime` stage, and nowhere else (not in the stages that run npm ci); found {settings}"
    )


def test_the_runtime_stage_is_the_one_that_ships() -> None:
    """The last stage is the image that is built and run, so NODE_ENV must be set in it."""
    names = stage_names(_DOCKERFILE.read_text(encoding="utf-8"))

    assert names[-1] == "runtime"


class TestTheParser:
    """Control: the checks above read a Dockerfile correctly, so they can fail."""

    def test_a_build_stage_setting_is_reported_against_that_stage(self) -> None:
        text = "FROM node:20 AS node-builder\nENV NODE_ENV=production\nRUN npm ci\n"

        assert node_env_settings(text) == [("ENV", "node-builder", "production")]

    def test_the_legacy_form_without_an_equals_sign_is_read(self) -> None:
        text = "FROM node:20 AS runtime\nENV NODE_ENV production\n"

        assert node_env_settings(text) == [("ENV", "runtime", "production")]

    def test_one_line_with_several_variables_is_read(self) -> None:
        text = "FROM node:20 AS runtime\nENV PORT=3000 NODE_ENV=development\n"

        assert node_env_settings(text) == [("ENV", "runtime", "development")]

    def test_an_arg_is_reported_too(self) -> None:
        text = "FROM node:20 AS node-deps\nARG NODE_ENV=production\n"

        assert node_env_settings(text) == [("ARG", "node-deps", "production")]

    def test_a_continued_line_is_joined(self) -> None:
        text = "FROM node:20 AS runtime\nENV PORT=3000 \\\n    NODE_ENV=production\n"

        assert node_env_settings(text) == [("ENV", "runtime", "production")]

    def test_a_comment_is_not_a_setting(self) -> None:
        text = "FROM node:20 AS runtime\n# ENV NODE_ENV=production\n"

        assert node_env_settings(text) == []

    def test_another_variable_is_not_a_setting(self) -> None:
        text = "FROM node:20 AS runtime\nENV NODE_ENV_FILE=production\n"

        assert node_env_settings(text) == []

    def test_a_from_line_with_a_platform_flag_starts_its_stage(self) -> None:
        text = (
            "FROM node:20 AS node-builder\n"
            "RUN npm ci\n"
            "FROM --platform=$BUILDPLATFORM node:20 AS runtime\n"
            "ENV NODE_ENV=production\n"
        )

        assert node_env_settings(text) == [("ENV", "runtime", "production")]
        assert stage_names(text) == ["node-builder", "runtime"]

    def test_several_flags_and_an_unnamed_stage_are_read(self) -> None:
        text = (
            "FROM node:20 AS node-builder\n"
            "FROM --platform=linux/amd64 --other=1 node:20\n"
            "ENV NODE_ENV=production\n"
        )

        assert node_env_settings(text) == [("ENV", "", "production")]
        assert stage_names(text) == ["node-builder", ""]
