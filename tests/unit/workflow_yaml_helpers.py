"""Reading GitHub workflow files for the two release-workflow tests.

Not a test module (no `test_` prefix), so pytest does not collect it. Used by
test_release_workflow_zizmor_fixes.py and test_verify_release_workflow_guards.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
GITHUB = ROOT / ".github"
WORKFLOWS = GITHUB / "workflows"


def load_workflow(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    return loaded


def triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """The `on:` mapping. PyYAML reads the bare key `on` as the boolean True."""
    found: dict[str, Any] = workflow.get("on", workflow.get(True))
    return found


def steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Every step of every job (a job that only `uses:` a reusable workflow has none)."""
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


def action(step: dict[str, Any]) -> str:
    """`owner/repo[/path]` of a step's `uses:`, lower-cased (GitHub reads it without case)."""
    return str(step.get("uses", "")).split("@")[0].lower()
