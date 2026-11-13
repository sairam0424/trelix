"""Every base image a Dockerfile in this repo builds FROM is pinned to a content digest.

WHY THIS EXISTS. OSSF Scorecard's Pinned-Dependencies check flags a `FROM image:tag` that has
no `@sha256:` (six alerts before the pins landed), and a floating tag also means a rebuild of
one commit can ship a different base. Dependabot moves the digest when the tag is re-pushed
(.github/dependabot.yml has a docker entry for each Dockerfile), so the pin does not freeze
the image: it makes each change of it a reviewable diff. The tag stays next to the digest
because Dependabot follows the tag and a reader needs to see which release the digest is.

The Dockerfiles are DISCOVERED, not listed: every file named `Dockerfile`, `Dockerfile.*` or
`*.Dockerfile` under the repository root, except `.git`, `node_modules` and virtualenvs. A new
Dockerfile is therefore checked the day it is added, with nothing to remember.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
1. Drop `@sha256:...` from any `FROM` in any Dockerfile
   -> test_every_base_image_is_pinned_to_a_digest fails naming the file and the image.
2. Replace a digest with a truncated or a too-long one, or drop the tag in front of it
   -> the same test fails (a short hash is not a pin, and `image@sha256:` loses the tag).
3. Stop skipping `scratch` or a reference to an earlier stage (in any letter case), or stop
   reading `FROM` lines -> TestTheCheckerBites fails.
4. Make the search miss a Dockerfile (a name pattern that drops `Dockerfile.dev` or
   `x.Dockerfile`, a pruned directory that hides a real one) -> TestDiscovery fails, and so
   does test_the_known_dockerfiles_are_discovered.
5. Stop pruning `.git`, `node_modules` or a virtualenv -> TestDiscovery fails.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

# `Dockerfile`, `Dockerfile.dev`, `dev.Dockerfile`; Docker also accepts the lower-case spelling.
_DOCKERFILE_NAME = re.compile(r"^(?:dockerfile(?:\..+)?|.+\.dockerfile)$", re.IGNORECASE)
# Trees that hold somebody else's files, never this repository's Dockerfiles. A virtualenv has
# no fixed name, so it is recognised by the `pyvenv.cfg` at its top instead.
_PRUNED_DIRS = frozenset({".git", "node_modules"})

# `FROM [--platform=...] image [AS stage]`: the flags are optional and there may be several.
_FROM = re.compile(
    r"^FROM\s+(?:--\S+\s+)*(?P<image>\S+)(?:\s+AS\s+(?P<stage>\S+))?\s*$", re.IGNORECASE
)
# name:tag@sha256:<64 hex>. The tag is required, so `image@sha256:...` does not pass.
_PINNED = re.compile(r"^[^@\s]+:[^:@/\s]+@sha256:[0-9a-f]{64}$")


def external_base_images(dockerfile: str) -> list[str]:
    """Images named by a FROM, without `scratch` and without references to an earlier stage."""
    stages: set[str] = set()
    images: list[str] = []
    for line in dockerfile.splitlines():
        match = _FROM.match(line.strip())
        if not match:
            continue
        image = match.group("image")
        if image.lower() != "scratch" and image.lower() not in stages:
            images.append(image)
        if match.group("stage"):
            stages.add(match.group("stage").lower())
    return images


def unpinned_base_images(dockerfile: str) -> list[str]:
    return [image for image in external_base_images(dockerfile) if not _PINNED.match(image)]


def find_dockerfiles(root: Path) -> list[Path]:
    """Every Dockerfile under `root`, without the vendored trees."""
    found: list[Path] = []
    for directory, subdirs, files in os.walk(root):
        # os.walk prunes by having this list edited in place.
        subdirs[:] = [
            name
            for name in subdirs
            if name not in _PRUNED_DIRS and not (Path(directory) / name / "pyvenv.cfg").exists()
        ]
        found.extend(Path(directory) / name for name in files if _DOCKERFILE_NAME.match(name))
    return found


def floating_base_images(root: Path) -> dict[str, list[str]]:
    """Relative path -> unpinned base images, for each Dockerfile under `root` that has any."""
    report: dict[str, list[str]] = {}
    for path in find_dockerfiles(root):
        unpinned = unpinned_base_images(path.read_text(encoding="utf-8"))
        if unpinned:
            report[path.relative_to(root).as_posix()] = unpinned
    return report


def test_the_known_dockerfiles_are_discovered() -> None:
    found = {path.relative_to(_ROOT).as_posix() for path in find_dockerfiles(_ROOT)}

    assert {"Dockerfile", "infra/github-app/Dockerfile"} <= found


def test_every_base_image_is_pinned_to_a_digest() -> None:
    assert floating_base_images(_ROOT) == {}


class TestDiscovery:
    _FLOATING = "FROM python:3.14-slim\n"

    def _write(self, root: Path, name: str, text: str | None = None) -> None:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._FLOATING if text is None else text, encoding="utf-8")

    @pytest.mark.parametrize(
        "name",
        [
            "Dockerfile",
            "packages/x/Dockerfile",
            "packages/x/Dockerfile.dev",
            "tools/app.Dockerfile",
            "tools/dockerfile",
        ],
    )
    def test_a_floating_dockerfile_anywhere_is_reported(self, tmp_path: Path, name: str) -> None:
        self._write(tmp_path, name)

        assert floating_base_images(tmp_path) == {name: ["python:3.14-slim"]}

    @pytest.mark.parametrize("name", ["README.md", "docs/dockerfile-notes.md", "mydockerfile"])
    def test_a_file_that_is_not_named_like_a_dockerfile_is_not_read(
        self, tmp_path: Path, name: str
    ) -> None:
        self._write(tmp_path, name)

        assert floating_base_images(tmp_path) == {}

    @pytest.mark.parametrize(
        "vendored",
        [
            "node_modules/pkg/Dockerfile",
            "packages/x/node_modules/pkg/Dockerfile",
            ".git/Dockerfile",
        ],
    )
    def test_a_vendored_tree_is_not_scanned(self, tmp_path: Path, vendored: str) -> None:
        self._write(tmp_path, "Dockerfile")
        self._write(tmp_path, vendored)

        assert floating_base_images(tmp_path) == {"Dockerfile": ["python:3.14-slim"]}

    def test_a_virtualenv_is_not_scanned(self, tmp_path: Path) -> None:
        self._write(tmp_path, "Dockerfile")
        self._write(tmp_path, "envs/py/pyvenv.cfg", "home = /usr/bin\n")
        self._write(tmp_path, "envs/py/lib/site-packages/pkg/Dockerfile")

        assert floating_base_images(tmp_path) == {"Dockerfile": ["python:3.14-slim"]}


class TestTheCheckerBites:
    _DIGEST = "a" * 64

    def test_a_pinned_image_is_clean(self) -> None:
        text = f"FROM python:3.14-slim@sha256:{self._DIGEST} AS builder\n"

        assert unpinned_base_images(text) == []

    @pytest.mark.parametrize(
        "line",
        [
            "FROM python:3.14-slim AS builder",
            "from python:3.14-slim AS builder",
            "    FROM python:3.14-slim AS builder",
            "FROM --platform=linux/amd64 python:3.14-slim AS builder",
        ],
        ids=["floating-tag", "lowercase-instruction", "indented", "platform-flag"],
    )
    def test_a_floating_tag_is_flagged(self, line: str) -> None:
        assert unpinned_base_images(line + "\n") == ["python:3.14-slim"]

    @pytest.mark.parametrize("digest", ["abc123", "a" * 65], ids=["truncated", "too-long"])
    def test_a_malformed_digest_is_flagged(self, digest: str) -> None:
        image = f"python:3.14-slim@sha256:{digest}"

        assert unpinned_base_images(f"FROM {image} AS b\n") == [image]

    def test_a_digest_without_a_tag_is_flagged(self) -> None:
        text = f"FROM python@sha256:{self._DIGEST} AS b\n"

        assert unpinned_base_images(text) == [f"python@sha256:{self._DIGEST}"]

    def test_a_reference_to_an_earlier_stage_and_scratch_are_exempt(self) -> None:
        text = (
            f"FROM node:24-bookworm-slim@sha256:{self._DIGEST} AS build\n"
            "FROM build AS test\n"
            "FROM scratch\n"
        )

        assert external_base_images(text) == [f"node:24-bookworm-slim@sha256:{self._DIGEST}"]
        assert unpinned_base_images(text) == []

    def test_a_stage_reference_ignores_letter_case(self) -> None:
        text = f"FROM node:24-bookworm-slim@sha256:{self._DIGEST} AS Build\nFROM BUILD AS test\n"

        assert external_base_images(text) == [f"node:24-bookworm-slim@sha256:{self._DIGEST}"]

    def test_a_commented_out_from_is_ignored(self) -> None:
        assert external_base_images("# FROM python:3.11-slim\n") == []
