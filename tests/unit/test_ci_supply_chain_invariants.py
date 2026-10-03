"""The supply-chain settings under .github/ must keep the shape that makes them protective.

WHY THIS EXISTS. Three properties shrink the attack surface of this repo's CI, and each is
one deleted line away from silently not being there, with nothing failing:

* `dependabot.yml`: every `updates` entry sets `cooldown: default-days: 7`, so a version is
  proposed a week after it is published instead of Dependabot's implicit three days. A new
  ecosystem entry copied from an old template leaves one manifest on the short window.
* `.github/workflows/*.yml`: every third-party `uses:` is pinned to a 40 hex commit SHA, so
  a hijacked action tag cannot reach this repo's CI. test_release_workflow_hardening.py
  checks this for the two tag-triggered publish workflows only; this file widens it to all
  of them.
* `zizmor.yml`: the audit is ADVISORY. It holds `contents: read` only, checks out without
  persisting the token, cannot fail a PR, and is time-bounded. Turning it into a gate
  (dropping `continue-on-error`) is meant to be a deliberate edit to this file's
  expectations, not an accident.

These tests read the YAML rather than mocking Actions: what is being prevented is a
*missing* key, and only the file shows whether it is there. Each rule is a pure function
over the parsed document, and `TestTheCheckersBite` feeds every one of them a document that
breaks it, so a checker that stopped matching fails there instead of turning the real-file
assertions green for the wrong reason.

MUTATIONS THAT MUST MAKE THIS FILE FAIL (each applied to a scratch copy of .github/)
1. `default-days: 7` -> `3` on any dependabot entry, or delete a `cooldown:` block
   -> TestDependabotCooldown fails naming the entry.
2. Any workflow's `uses: x@<sha>` -> `uses: x@v4`
   -> TestEveryActionIsPinned fails for that file.
3. zizmor.yml: add a write scope, set `persist-credentials: true`, drop
   `continue-on-error`, or drop `timeout-minutes`
   -> TestZizmorStaysAdvisory fails with the specific rule.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_DEPENDABOT = _ROOT / ".github" / "dependabot.yml"
_ZIZMOR = _WORKFLOWS / "zizmor.yml"

# zizmor's `dependabot-cooldown` audit recommends seven days; Dependabot's own default is three.
_MIN_COOLDOWN_DAYS = 7
_SHA_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
_ADVISORY_PERMISSIONS = {"contents": "read"}
_CHECKOUT = "actions/checkout"

# A broken glob or parser reports "nothing unpinned" just as loudly as a clean tree. Today
# there are 15 workflows and 106 `uses:`; the floors sit well below so that consolidating
# workflows does not trip them, while an empty walk does.
_MIN_WORKFLOWS = 10
_MIN_USES_REFS = 50


def _load(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def _workflow_files() -> list[Path]:
    return sorted([*_WORKFLOWS.glob("*.yml"), *_WORKFLOWS.glob("*.yaml")])


# ---------------------------------------------------------------------------
# The rules: pure functions over a parsed document, each returning its violations
# ---------------------------------------------------------------------------
def cooldown_violations(config: dict[str, Any]) -> list[str]:
    """One message per `updates` entry whose cooldown is missing or shorter than a week."""
    updates = config.get("updates") or []
    if not updates:
        return ["no `updates` entries: the parser is wrong"]
    problems: list[str] = []
    for entry in updates:
        cooldown = entry.get("cooldown")
        days = cooldown.get("default-days") if isinstance(cooldown, dict) else None
        if not isinstance(days, int) or days < _MIN_COOLDOWN_DAYS:
            problems.append(
                f"{entry.get('package-ecosystem')} {entry.get('directory')}: "
                f"cooldown.default-days is {days!r}, need >= {_MIN_COOLDOWN_DAYS}"
            )
    return problems


def uses_refs(workflow: dict[str, Any]) -> list[tuple[str, str]]:
    """Every (location, ref) `uses:` in a workflow: job-level and step-level."""
    refs: list[tuple[str, str]] = []
    for job_name, job in (workflow.get("jobs") or {}).items():
        if "uses" in job:
            refs.append((f"{job_name} (job)", job["uses"]))
        for index, step in enumerate(job.get("steps") or []):
            if "uses" in step:
                refs.append((f"{job_name} step {index}", step["uses"]))
    return refs


def unpinned_uses(workflow: dict[str, Any]) -> list[str]:
    """`uses:` refs that are neither local, a docker:// image, nor a 40 hex commit SHA."""
    return [
        f"{location} uses mutable ref '{ref}'"
        for location, ref in uses_refs(workflow)
        if not ref.startswith(("./", "docker://")) and not _SHA_PIN.match(ref)
    ]


def _within_advisory_permissions(permissions: Any) -> bool:
    return isinstance(permissions, dict) and set(permissions.items()) <= set(
        _ADVISORY_PERMISSIONS.items()
    )


def zizmor_shape_violations(workflow: dict[str, Any]) -> list[str]:
    """Departures from the advisory shape: read-only token, no persisted credentials,
    cannot fail a PR, bounded run time."""
    problems: list[str] = []
    if workflow.get("permissions") != _ADVISORY_PERMISSIONS:
        problems.append(
            f"workflow permissions are {workflow.get('permissions')!r}, "
            f"want {_ADVISORY_PERMISSIONS!r}"
        )
    jobs = workflow.get("jobs") or {}
    if not jobs:
        problems.append("no jobs: the parser is wrong")
    checkouts = 0
    for name, job in jobs.items():
        if "permissions" in job and not _within_advisory_permissions(job["permissions"]):
            problems.append(f"job '{name}' widens permissions to {job['permissions']!r}")
        if job.get("continue-on-error") is not True:
            problems.append(f"job '{name}' is not `continue-on-error: true`")
        timeout = job.get("timeout-minutes")
        if not isinstance(timeout, int) or timeout <= 0:
            problems.append(f"job '{name}' has no timeout-minutes")
        for index, step in enumerate(job.get("steps") or []):
            if str(step.get("uses", "")).split("@")[0] != _CHECKOUT:
                continue
            checkouts += 1
            if (step.get("with") or {}).get("persist-credentials") is not False:
                problems.append(
                    f"job '{name}' step {index}: {_CHECKOUT} without `persist-credentials: false`"
                )
    if not checkouts:
        problems.append(f"no {_CHECKOUT} step: the persist-credentials rule would be vacuous")
    return problems


# ---------------------------------------------------------------------------
# The real files
# ---------------------------------------------------------------------------
class TestDependabotCooldown:
    def test_every_entry_waits_at_least_a_week(self) -> None:
        problems = cooldown_violations(_load(_DEPENDABOT))
        assert not problems, "dependabot.yml entries with a short cooldown:\n" + "\n".join(problems)


class TestEveryActionIsPinned:
    def test_the_walk_reached_the_workflows(self) -> None:
        """PRECONDITION: an empty walk reports a clean tree just as loudly."""
        files = _workflow_files()
        assert len(files) >= _MIN_WORKFLOWS, f"only {len(files)} workflow files found"
        refs = sum(len(uses_refs(_load(path))) for path in files)
        assert refs >= _MIN_USES_REFS, f"only {refs} `uses:` references found; the parser is wrong"

    @pytest.mark.parametrize("path", _workflow_files(), ids=lambda path: path.name)
    def test_uses_refs_are_40_hex_shas_or_local(self, path: Path) -> None:
        problems = unpinned_uses(_load(path))
        assert not problems, f"{path.name}:\n" + "\n".join(problems)


class TestZizmorStaysAdvisory:
    def test_shape(self) -> None:
        problems = zizmor_shape_violations(_load(_ZIZMOR))
        assert not problems, "zizmor.yml left its advisory shape:\n" + "\n".join(problems)


# ---------------------------------------------------------------------------
# The checkers themselves: each rule must flag a document that breaks it
# ---------------------------------------------------------------------------
_SHA = "0123456789abcdef0123456789abcdef01234567"


def _entry(cooldown: Any = None, *, directory: str = "/") -> dict[str, Any]:
    entry: dict[str, Any] = {"package-ecosystem": "pip", "directory": directory}
    if cooldown is not None:
        entry["cooldown"] = cooldown
    return entry


_ADVISORY_YAML = f"""\
name: zizmor
permissions:
  contents: read
jobs:
  zizmor:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    continue-on-error: true
    steps:
      - uses: actions/checkout@{_SHA}
        with:
          persist-credentials: false
      - uses: zizmorcore/zizmor-action@{_SHA}
"""

_CHECKOUT_STEP = f"""\
      - uses: actions/checkout@{_SHA}
        with:
          persist-credentials: false
"""


class TestTheCheckersBite:
    @pytest.mark.parametrize("days", [7, 8, 30])
    def test_a_cooldown_of_at_least_a_week_is_clean(self, days: int) -> None:
        assert cooldown_violations({"updates": [_entry({"default-days": days})]}) == []

    @pytest.mark.parametrize(
        "cooldown",
        [
            pytest.param(None, id="no-cooldown-block"),
            pytest.param({"default-days": 6}, id="six-days"),
            pytest.param({"default-days": 3}, id="dependabots-implicit-three"),
            pytest.param({"semver-major-days": 14}, id="only-a-semver-key"),
            pytest.param(7, id="not-a-mapping"),
        ],
    )
    def test_a_short_or_missing_cooldown_is_flagged(self, cooldown: Any) -> None:
        problems = cooldown_violations({"updates": [_entry(cooldown)]})
        assert len(problems) == 1
        assert problems[0].startswith("pip /: cooldown.default-days is ")

    def test_only_the_offending_entry_is_named(self) -> None:
        updates = [_entry({"default-days": 7}), _entry(None, directory="/packages/x")]
        assert cooldown_violations({"updates": updates}) == [
            "pip /packages/x: cooldown.default-days is None, need >= 7"
        ]

    def test_an_empty_updates_list_is_flagged(self) -> None:
        assert len(cooldown_violations({"updates": []})) == 1

    @pytest.mark.parametrize(
        "ref",
        [
            pytest.param(f"actions/checkout@{_SHA}", id="full-sha"),
            pytest.param(f"org/repo/sub/path@{_SHA}", id="full-sha-with-subpath"),
            pytest.param("./.github/actions/local", id="local-action"),
            pytest.param("docker://alpine:3.20", id="docker-image"),
        ],
    )
    def test_a_pinned_or_local_ref_is_clean(self, ref: str) -> None:
        assert unpinned_uses({"jobs": {"build": {"steps": [{"uses": ref}]}}}) == []

    @pytest.mark.parametrize(
        "ref",
        [
            pytest.param("actions/checkout@v4", id="tag"),
            pytest.param("actions/checkout@main", id="branch"),
            pytest.param("actions/checkout", id="no-ref"),
            pytest.param(f"actions/checkout@{_SHA[:-1]}", id="39-hex"),
            pytest.param(f"actions/checkout@{_SHA}8", id="41-hex"),
            pytest.param(f"actions/checkout@{_SHA.upper()}", id="uppercase-hex"),
        ],
    )
    def test_a_mutable_step_ref_is_flagged(self, ref: str) -> None:
        workflow = {"jobs": {"build": {"steps": [{"name": "no uses"}, {"uses": ref}]}}}
        assert unpinned_uses(workflow) == [f"build step 1 uses mutable ref '{ref}'"]

    def test_a_mutable_reusable_workflow_ref_is_flagged(self) -> None:
        workflow = {"jobs": {"call": {"uses": "org/repo/.github/workflows/x.yml@v1"}}}
        assert unpinned_uses(workflow) == [
            "call (job) uses mutable ref 'org/repo/.github/workflows/x.yml@v1'"
        ]

    def test_the_advisory_fixture_is_clean(self) -> None:
        """CONTROL for the cases below: they break one line of this and nothing else."""
        assert zizmor_shape_violations(yaml.safe_load(_ADVISORY_YAML)) == []

    @pytest.mark.parametrize(
        ("old", "new", "expected"),
        [
            pytest.param(
                "contents: read",
                "contents: write",
                "workflow permissions are",
                id="workflow-write-scope",
            ),
            pytest.param(
                "  contents: read\n",
                "  contents: read\n  security-events: write\n",
                "workflow permissions are",
                id="workflow-extra-scope",
            ),
            pytest.param(
                "    timeout-minutes: 10\n",
                "    timeout-minutes: 10\n    permissions:\n      id-token: write\n",
                "widens permissions",
                id="job-widens-permissions",
            ),
            pytest.param(
                "persist-credentials: false",
                "persist-credentials: true",
                "without `persist-credentials: false`",
                id="persisted-credentials",
            ),
            pytest.param(
                "        with:\n          persist-credentials: false\n",
                "",
                "without `persist-credentials: false`",
                id="checkout-with-block-missing",
            ),
            pytest.param(
                "    continue-on-error: true\n",
                "",
                "is not `continue-on-error: true`",
                id="continue-on-error-dropped",
            ),
            pytest.param(
                "continue-on-error: true",
                "continue-on-error: false",
                "is not `continue-on-error: true`",
                id="continue-on-error-false",
            ),
            pytest.param(
                "    timeout-minutes: 10\n",
                "",
                "has no timeout-minutes",
                id="timeout-dropped",
            ),
            pytest.param(
                _CHECKOUT_STEP,
                "",
                "the persist-credentials rule would be vacuous",
                id="checkout-removed",
            ),
        ],
    )
    def test_each_departure_from_the_advisory_shape_is_flagged(
        self, old: str, new: str, expected: str
    ) -> None:
        mutated = _ADVISORY_YAML.replace(old, new)
        assert mutated != _ADVISORY_YAML, (
            "the case did not change the fixture, so it proves nothing"
        )
        problems = zizmor_shape_violations(yaml.safe_load(mutated))
        assert len(problems) == 1, problems
        assert expected in problems[0]
