"""The GitHub App image's walker flag must be a name and value trelix really honours.

The App (infra/github-app) is TypeScript; the setting it depends on lives in Python
(`WalkerConfig`, `env_prefix="TRELIX_WALKER_"`). Nothing in either language's own test
suite ties the two together, so a typo in the Dockerfile's `ENV` line (a wrong name, or
a value pydantic reads as true) would ship a container that follows symlinks out of an
untrusted PR checkout while every test stays green. This reads the Dockerfile's actual
line and proves the config layer resolves it to `follow_symlinks=False`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from trelix.core.config import WalkerConfig

_DOCKERFILE = Path(__file__).resolve().parents[2] / "infra" / "github-app" / "Dockerfile"
_ENV_LINE = re.compile(r"^ENV\s+(TRELIX_WALKER_FOLLOW_SYMLINKS)=(\S+)\s*$", re.MULTILINE)


def _dockerfile_walker_env() -> tuple[str, str]:
    matches = _ENV_LINE.findall(_DOCKERFILE.read_text(encoding="utf-8"))
    assert len(matches) == 1, (
        "infra/github-app/Dockerfile must set TRELIX_WALKER_FOLLOW_SYMLINKS exactly once "
        f"with `ENV NAME=value`; found {len(matches)}"
    )
    name, value = matches[0]
    return name, value.strip("\"'")


def test_dockerfile_env_line_disables_symlink_following(monkeypatch: pytest.MonkeyPatch) -> None:
    name, value = _dockerfile_walker_env()

    monkeypatch.setenv(name, value)

    assert WalkerConfig().follow_symlinks is False


def test_the_default_still_follows_symlinks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: without the variable the flag is on, so the test above proves the ENV line."""
    monkeypatch.delenv("TRELIX_WALKER_FOLLOW_SYMLINKS", raising=False)

    assert WalkerConfig().follow_symlinks is True
