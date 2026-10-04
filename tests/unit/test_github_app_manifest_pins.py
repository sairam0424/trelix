"""The GitHub App manifest asks GitHub for exactly these permissions and events.

WHY THIS EXISTS. `infra/github-app/manifest.yml` is what a new copy of the App is
registered from, and it is the written record of what the App holds. The App keeps
`pull_requests` at read on purpose (it never comments on or edits a pull request) and
mints its three per-review tokens from these three permissions. Nothing else ties the
manifest to that design: a later change could widen a permission, or add one, and every
other test would stay green. The expected values are written out below, not read from
the manifest, so changing the manifest fails this file until a person edits it on purpose.

A manifest is read only when an App is created; an already registered App's permissions
are changed by its owner in GitHub's settings. This pins the file, not the live App.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_MANIFEST = Path(__file__).resolve().parents[2] / "infra" / "github-app" / "manifest.yml"


def _manifest() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    return loaded


def test_the_manifest_asks_for_exactly_these_permissions() -> None:
    assert _manifest()["default_permissions"] == {
        "pull_requests": "read",
        "checks": "write",
        "contents": "read",
    }


def test_the_manifest_subscribes_to_exactly_the_pull_request_event() -> None:
    assert _manifest()["default_events"] == ["pull_request"]
