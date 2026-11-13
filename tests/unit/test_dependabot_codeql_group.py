"""Dependabot must put every github/codeql-action/* step in ONE group, so the pair moves together.

WHY THIS EXISTS. codeql.yml runs `github/codeql-action/init` and `/analyze`, and the two have
to stay on the same CodeQL Action release. Dependabot opens one PR per action by default (#578
moved only init, #579 only analyze), so merging either alone splits the pair. A group in the
github-actions entry of dependabot.yml fixes that, and it is one deleted block, one mistyped
pattern or one split into two groups away from not doing so, with nothing failing.

MUTATIONS THAT MUST MAKE THIS FILE FAIL (each applied to a scratch copy of dependabot.yml)
1. Delete the group from the github-actions entry, or move it to another entry.
2. Narrow the pattern to `github/codeql-action/init`, or mistype it as
   `github/codeql-actions/*`.
3. Split the pattern into two groups, one for init and one for analyze.
4. Give the group `applies-to: security-updates`, so it no longer groups version updates.
5. Put `github/codeql-action/analyze` (or init) in the group's `exclude-patterns`.
Each fails test_one_group_matches_every_codeql_action_step, naming the steps and the groups.
6. Move `analyze` in codeql.yml to another SHA than `init`: the CodeQL action then stops with
   "Loaded a configuration file for version X, but running version Y", which is what merging
   #578 or #579 alone did -> test_codeql_init_and_analyze_run_the_same_release.
Renaming the group is not a mutation: Dependabot does not care what a group is called.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_DEPENDABOT = _ROOT / ".github" / "dependabot.yml"
_WORKFLOWS = _ROOT / ".github" / "workflows"
_CODEQL_STEP = re.compile(r"uses:\s*(github/codeql-action/[\w-]+)@")


def _github_actions_groups() -> dict[str, dict[str, Any]]:
    config: dict[str, Any] = yaml.safe_load(_DEPENDABOT.read_text(encoding="utf-8"))
    entries = [e for e in config["updates"] if e.get("package-ecosystem") == "github-actions"]
    assert len(entries) == 1, f"expected one github-actions entry, found {len(entries)}"
    return dict(entries[0].get("groups") or {})


def _codeql_steps_in_workflows() -> set[str]:
    steps: set[str] = set()
    for workflow in sorted(_WORKFLOWS.glob("*.yml")):
        steps.update(_CODEQL_STEP.findall(workflow.read_text(encoding="utf-8")))
    return steps


def _groups_every_version_update(group: dict[str, Any], steps: set[str]) -> bool:
    if group.get("applies-to", "version-updates") != "version-updates":
        return False
    patterns = list(group.get("patterns") or [])
    excluded = list(group.get("exclude-patterns") or [])
    return all(
        any(fnmatch.fnmatchcase(s, p) for p in patterns)
        and not any(fnmatch.fnmatchcase(s, p) for p in excluded)
        for s in steps
    )


def test_one_group_matches_every_codeql_action_step() -> None:
    steps = _codeql_steps_in_workflows()
    # A walk that finds nothing would pass the check below for the wrong reason.
    assert {"github/codeql-action/init", "github/codeql-action/analyze"} <= steps
    groups = _github_actions_groups()
    covering = [
        name for name, group in groups.items() if _groups_every_version_update(group, steps)
    ]
    assert covering, f"no one group matches all of {sorted(steps)}; groups are {groups}"


def test_codeql_init_and_analyze_run_the_same_release() -> None:
    text = (_WORKFLOWS / "codeql.yml").read_text(encoding="utf-8")
    pinned = {
        step: set(re.findall(rf"uses:\s*github/codeql-action/{step}@([0-9a-f]{{40}})", text))
        for step in ("init", "analyze")
    }
    assert all(len(shas) == 1 for shas in pinned.values()), f"each step needs one SHA: {pinned}"
    assert pinned["init"] == pinned["analyze"], f"init and analyze differ: {pinned}"
