"""The supply-chain settings under .github/ must keep the shape that makes them protective.

WHY THIS EXISTS. Five properties shrink the attack surface of this repo's CI, and each is
one deleted line away from silently not being there, with nothing failing:

* `dependabot.yml`: every `updates` entry sets `cooldown: default-days: 7`, so a version is
  proposed a week after it is published instead of Dependabot's implicit three days. A new
  ecosystem entry copied from an old template leaves one manifest on the short window.
* `.github/workflows/*.yml`: every third-party `uses:` is pinned to a 40 hex commit SHA, so
  a hijacked action tag cannot reach this repo's CI. test_release_workflow_hardening.py
  checks this for the two tag-triggered publish workflows only; this file widens it to all
  of them.
* every `actions/checkout` step sets `persist-credentials: false`, so the job token is not
  left in `.git/config`, readable by every later step (third-party actions included) and by
  anything that archives the workspace. Only a job whose later steps really authenticate git
  may keep it, and it must be named in `_CHECKOUT_CREDENTIAL_EXCEPTIONS` with the reason.
  Today none does: no workflow pushes, and the one `git fetch` after a checkout
  (`scripts/verify_release.py`, a tag fallback) reads this public repository anonymously.
* a `# vX.Y.Z` comment on a SHA pin names the tag that points at that SHA (zizmor's
  `ref-version-mismatch`), for the pins whose tag was checked against the upstream repo.
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
4. Delete `persist-credentials: false` from any one checkout, or add a checkout without it
   -> TestEveryCheckoutDropsItsCredentials fails naming the file, job and step.
5. Add an entry to `_CHECKOUT_CREDENTIAL_EXCEPTIONS` for a job that does not exist, has no
   checkout, already sets `persist-credentials: false`, or has a blank reason
   -> the same class fails on the stale-entry test.
6. Change a `docker/build-push-action` pin's comment back to `# v7` (the floating major tag,
   which points at a different commit than the pinned v7.4.0)
   -> TestVersionCommentsNameTheirTag fails for that file.
"""

from __future__ import annotations

import re
import time
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
_USES_WITH_VERSION_COMMENT = re.compile(r"^\s*(?:-\s+)?uses:\s*(\S+)\s+#\s*(\S+)")

# (workflow file name, job id) -> why that job's checkout has to keep the job token in
# .git/config because a LATER step in the same job authenticates git with it (a push, or a
# fetch of a ref the anonymous caller cannot read).
#
# EMPTY ON PURPOSE. Every one of the 30 checkouts was read against the steps after it: no
# workflow pushes or commits, `gh` and `GITHUB_TOKEN` steps take the token from `env`, not
# from .git/config, and the only `git fetch` after a checkout is the tag fallback in
# scripts/verify_release.py, which reads this public repository anonymously. An entry is
# added here by a reviewed edit, with a reason; one whose job no longer needs it fails
# `test_the_exception_list_has_no_stale_or_unexplained_entries`.
_CHECKOUT_CREDENTIAL_EXCEPTIONS: dict[tuple[str, str], str] = {}

# `uses:` pins whose version comment was resolved against the upstream repository, so the
# comment can be checked offline: `owner/repo@<sha>` -> the tag that points at that commit.
# docker/build-push-action@c3c9e263... is v7.4.0 (`gh api
# repos/docker/build-push-action/git/ref/tags/v7.4.0`); the pins once cited the floating `v7`
# tag, which points at a different commit whenever a release lands: zizmor's
# `ref-version-mismatch`. An entry goes inert when Dependabot moves the pin to a new SHA, and
# the guard stops biting silently: in the same PR, replace the entry with the new SHA and tag.
_VERIFIED_PIN_TAGS: dict[str, str] = {
    "docker/build-push-action@c3c9e263c25d99ce0380d002d59b67737d91b0dc": "v7.4.0",
}

# A broken glob or parser reports "nothing unpinned" just as loudly as a clean tree. Today
# there are 15 workflows, 106 `uses:` and 30 checkouts; the floors sit well below so that
# consolidating workflows does not trip them, while an empty walk does.
_MIN_WORKFLOWS = 10
_MIN_USES_REFS = 50
_MIN_CHECKOUTS = 20


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


def is_checkout_step(step: dict[str, Any]) -> bool:
    return str(step.get("uses", "")).split("@")[0] == _CHECKOUT


def persists_credentials(step: dict[str, Any]) -> bool:
    """True unless the step sets the YAML boolean `persist-credentials: false`.

    A string, an expression and a missing key all leave the token in .git/config as far as
    this check can prove, so all of them count as persisting.
    """
    return (step.get("with") or {}).get("persist-credentials") is not False


def checkout_steps(workflow: dict[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
    """Every (job, step index, step) that is an `actions/checkout` step."""
    return [
        (job_name, index, step)
        for job_name, job in (workflow.get("jobs") or {}).items()
        for index, step in enumerate(job.get("steps") or [])
        if is_checkout_step(step)
    ]


def credential_persistence_violations(
    file_name: str,
    workflow: dict[str, Any],
    exceptions: dict[tuple[str, str], str],
) -> list[str]:
    """Checkouts that leave the token in .git/config and are not an allow-listed job."""
    return [
        f"{file_name} job '{job}' step {index}: {_CHECKOUT} without `persist-credentials: false` "
        "(set it, or allow-list the job with the reason a later step needs git credentials)"
        for job, index, step in checkout_steps(workflow)
        if persists_credentials(step) and (file_name, job) not in exceptions
    ]


def exception_problems(
    workflows: dict[str, dict[str, Any]], exceptions: dict[tuple[str, str], str]
) -> list[str]:
    """Allow-list entries that excuse nothing, or that do not say why."""
    problems: list[str] = []
    for (file_name, job), reason in sorted(exceptions.items()):
        where = f"{file_name} job '{job}'"
        if not reason.strip():
            problems.append(f"{where}: the exception has no reason")
        if file_name not in workflows:
            problems.append(f"{where}: no workflow file with that name")
            continue
        steps = [step for name, _, step in checkout_steps(workflows[file_name]) if name == job]
        if not steps:
            problems.append(f"{where}: the job has no {_CHECKOUT} step")
        elif not any(persists_credentials(step) for step in steps):
            problems.append(f"{where}: its checkout already sets `persist-credentials: false`")
    return problems


def version_comment_violations(text: str, verified: dict[str, str]) -> list[str]:
    """`uses: <pin> # <comment>` lines whose comment is not the verified tag for that pin."""
    problems: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = _USES_WITH_VERSION_COMMENT.match(line)
        if not match:
            continue
        pin, comment = match.groups()
        expected = verified.get(pin)
        if expected is not None and comment != expected:
            problems.append(f"line {lineno}: {pin} is commented '{comment}', its tag is {expected}")
    return problems


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
            if not is_checkout_step(step):
                continue
            checkouts += 1
            if persists_credentials(step):
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


class TestEveryCheckoutDropsItsCredentials:
    def test_the_walk_reached_the_checkouts(self) -> None:
        """PRECONDITION: a parser that finds no checkout reports every one of them clean."""
        found = sum(len(checkout_steps(_load(path))) for path in _workflow_files())
        assert found >= _MIN_CHECKOUTS, f"only {found} checkout steps found; the parser is wrong"

    @pytest.mark.parametrize("path", _workflow_files(), ids=lambda path: path.name)
    def test_checkout_sets_persist_credentials_false_unless_allow_listed(self, path: Path) -> None:
        problems = credential_persistence_violations(
            path.name, _load(path), _CHECKOUT_CREDENTIAL_EXCEPTIONS
        )
        assert not problems, "\n".join(problems)

    def test_the_exception_list_has_no_stale_or_unexplained_entries(self) -> None:
        workflows = {path.name: _load(path) for path in _workflow_files()}
        problems = exception_problems(workflows, _CHECKOUT_CREDENTIAL_EXCEPTIONS)
        assert not problems, "_CHECKOUT_CREDENTIAL_EXCEPTIONS:\n" + "\n".join(problems)


class TestVersionCommentsNameTheirTag:
    @pytest.mark.parametrize("path", _workflow_files(), ids=lambda path: path.name)
    def test_a_verified_pin_carries_the_tag_that_points_at_it(self, path: Path) -> None:
        problems = version_comment_violations(path.read_text(encoding="utf-8"), _VERIFIED_PIN_TAGS)
        assert not problems, f"{path.name}:\n" + "\n".join(problems)


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


# Two jobs, both clean. `build` has a non-checkout step first (so the checkout is step 1) and
# a `with:` block holding another key; `release` has the bare one-key form. Each case below
# breaks one line of this and nothing else.
_CHECKOUT_JOBS_YAML = f"""\
jobs:
  build:
    steps:
      - uses: actions/setup-python@{_SHA}
      - uses: actions/checkout@{_SHA}
        with:
          fetch-depth: 0
          persist-credentials: false
  release:
    steps:
      - uses: actions/checkout@{_SHA}
        with:
          persist-credentials: false
"""
_BUILD_LINE = "          fetch-depth: 0\n          persist-credentials: false\n"
_RELEASE_CHECKOUT = (
    f"      - uses: actions/checkout@{_SHA}\n        with:\n          persist-credentials: false\n"
)


def _mutated(old: str, new: str) -> dict[str, Any]:
    mutated = _CHECKOUT_JOBS_YAML.replace(old, new)
    assert mutated != _CHECKOUT_JOBS_YAML, "the case did not change the fixture"
    loaded: dict[str, Any] = yaml.safe_load(mutated)
    return loaded


class TestTheCheckoutCheckersBite:
    def test_the_fixture_is_clean(self) -> None:
        """CONTROL for the cases below."""
        workflow = yaml.safe_load(_CHECKOUT_JOBS_YAML)
        assert credential_persistence_violations("wf.yml", workflow, {}) == []
        assert [(job, index) for job, index, _ in checkout_steps(workflow)] == [
            ("build", 1),
            ("release", 0),
        ]

    @pytest.mark.parametrize(
        ("old", "new", "job", "index"),
        [
            pytest.param(_BUILD_LINE, "          fetch-depth: 0\n", "build", 1, id="key-missing"),
            pytest.param(
                _BUILD_LINE,
                "          fetch-depth: 0\n          persist-credentials: true\n",
                "build",
                1,
                id="true",
            ),
            pytest.param(
                _BUILD_LINE,
                '          fetch-depth: 0\n          persist-credentials: "false"\n',
                "build",
                1,
                id="quoted-string",
            ),
            pytest.param(
                _BUILD_LINE,
                "          fetch-depth: 0\n          persist-credentials: ${{ inputs.keep }}\n",
                "build",
                1,
                id="expression",
            ),
            pytest.param(
                _RELEASE_CHECKOUT,
                f"      - uses: actions/checkout@{_SHA}\n",
                "release",
                0,
                id="with-block-missing",
            ),
            pytest.param(
                _RELEASE_CHECKOUT,
                _RELEASE_CHECKOUT + f"      - uses: actions/checkout@{_SHA}\n",
                "release",
                1,
                id="second-checkout-added-without-it",
            ),
        ],
    )
    def test_a_checkout_that_keeps_the_token_is_flagged(
        self, old: str, new: str, job: str, index: int
    ) -> None:
        problems = credential_persistence_violations("wf.yml", _mutated(old, new), {})
        assert len(problems) == 1, problems
        assert problems[0].startswith(f"wf.yml job '{job}' step {index}: actions/checkout")

    def test_a_new_job_with_a_bare_checkout_is_flagged(self) -> None:
        workflow = _mutated(
            "  release:\n",
            f"  added:\n    steps:\n      - uses: actions/checkout@{_SHA}\n  release:\n",
        )
        problems = credential_persistence_violations("wf.yml", workflow, {})
        assert [problem.split(":")[0] for problem in problems] == ["wf.yml job 'added' step 0"]

    @pytest.mark.parametrize(
        ("exceptions", "flagged"),
        [
            pytest.param({("wf.yml", "release"): "pushes a tag"}, False, id="exact-file-and-job"),
            pytest.param({("other.yml", "release"): "pushes a tag"}, True, id="other-file"),
            pytest.param({("wf.yml", "build"): "pushes a tag"}, True, id="other-job"),
        ],
    )
    def test_an_exception_excuses_exactly_its_file_and_job(
        self, exceptions: dict[tuple[str, str], str], flagged: bool
    ) -> None:
        workflow = _mutated(_RELEASE_CHECKOUT, f"      - uses: actions/checkout@{_SHA}\n")
        problems = credential_persistence_violations("wf.yml", workflow, exceptions)
        assert bool(problems) is flagged, problems

    def test_a_job_without_steps_is_not_a_checkout(self) -> None:
        reusable = {"jobs": {"call": {"uses": f"org/repo/.github/workflows/x.yml@{_SHA}"}}}
        assert checkout_steps(reusable) == []

    @pytest.mark.parametrize(
        ("exceptions", "expected"),
        [
            pytest.param({("wf.yml", "release"): "pushes a tag"}, None, id="needed-and-explained"),
            pytest.param({("gone.yml", "release"): "pushes a tag"}, "no workflow file", id="file"),
            pytest.param({("wf.yml", "lint"): "pushes a tag"}, "has no actions/checkout", id="job"),
            pytest.param(
                {("wf.yml", "build"): "pushes a tag"},
                "already sets `persist-credentials: false`",
                id="dead",
            ),
            pytest.param({("wf.yml", "release"): "  "}, "the exception has no reason", id="blank"),
        ],
    )
    def test_a_stale_or_unexplained_exception_is_flagged(
        self, exceptions: dict[tuple[str, str], str], expected: str | None
    ) -> None:
        # `release` keeps the token so an entry for it is live; `lint` has no checkout.
        base = _mutated(_RELEASE_CHECKOUT, f"      - uses: actions/checkout@{_SHA}\n")
        base["jobs"]["lint"] = {"steps": [{"uses": f"actions/setup-python@{_SHA}"}]}
        problems = exception_problems({"wf.yml": base}, exceptions)
        if expected is None:
            assert problems == []
            return
        assert len(problems) == 1, problems
        assert expected in problems[0]

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            pytest.param(f"uses: org/act@{_SHA}  # v1.2.3", [], id="right-tag-two-spaces"),
            pytest.param(f"- uses: org/act@{_SHA} # v1.2.3", [], id="right-tag-dash-form"),
            pytest.param(f"uses: org/act@{_SHA}", [], id="no-comment"),
            pytest.param(f"uses: org/act@{_SHA[:-1]}0 # v9", [], id="another-pin-unchecked"),
            pytest.param(f"uses: org/other@{_SHA} # v9", [], id="another-action-unchecked"),
            pytest.param(
                f"uses: org/act@{_SHA} # v1",
                [f"line 1: org/act@{_SHA} is commented 'v1', its tag is v1.2.3"],
                id="floating-major",
            ),
            pytest.param(
                f"uses: org/act@{_SHA}  # v1.2.4",
                [f"line 1: org/act@{_SHA} is commented 'v1.2.4', its tag is v1.2.3"],
                id="wrong-patch",
            ),
        ],
    )
    def test_a_verified_pin_with_the_wrong_comment_is_flagged(
        self, line: str, expected: list[str]
    ) -> None:
        verified = {f"org/act@{_SHA}": "v1.2.3"}
        assert version_comment_violations(line + "\n", verified) == expected

    def test_the_line_number_follows_the_offending_line(self) -> None:
        text = f"name: x\njobs:\n  a:\n    steps:\n      - uses: org/act@{_SHA} # v1\n"
        problems = version_comment_violations(text, {f"org/act@{_SHA}": "v1.2.3"})
        assert [problem.split(":")[0] for problem in problems] == ["line 5"]


_HOSTILE_LENGTH = 50_000
_MAX_SCAN_SECONDS = 2.0
_HOSTILE_LINES = {
    "spaces-then-uses": " " * _HOSTILE_LENGTH + "uses: a",
    "dash-runs": "- " * (_HOSTILE_LENGTH // 2) + "uses: a",
    "one-long-token": "uses: " + "a" * _HOSTILE_LENGTH,
    "trailing-spaces": "uses: a" + " " * _HOSTILE_LENGTH,
    "space-hash-runs": "uses: a" + " #" * (_HOSTILE_LENGTH // 2),
    "hash-run": "uses: a " + "#" * _HOSTILE_LENGTH,
    "newline-runs": "\n" * _HOSTILE_LENGTH,
    "space-newline-runs": " \n" * (_HOSTILE_LENGTH // 2),
    "many-short-lines": "uses: a # v1\n" * (_HOSTILE_LENGTH // 10),
    "backticks": "`" * _HOSTILE_LENGTH,
    "tildes": "~" * _HOSTILE_LENGTH,
    "nested-brackets": "[" * _HOSTILE_LENGTH,
    "unterminated-comment-openers": "<!--" * (_HOSTILE_LENGTH // 4),
    "combining-marks": "uses: a" + "\u0301" * _HOSTILE_LENGTH,
    "zero-width-characters": "uses: a #" + "\u200b" * _HOSTILE_LENGTH,
}


class TestTheCommentScanIsLinear:
    """The scan reads workflow files a fork's pull request can rewrite, so it must not be
    something a crafted line can slow down."""

    @pytest.mark.parametrize("hostile", _HOSTILE_LINES.values(), ids=_HOSTILE_LINES.keys())
    def test_a_hostile_file_is_scanned_in_well_under_two_seconds(self, hostile: str) -> None:
        assert len(hostile) >= _HOSTILE_LENGTH
        started = time.perf_counter()
        problems = version_comment_violations(hostile, {f"org/act@{_SHA}": "v1.2.3"})
        elapsed = time.perf_counter() - started
        assert problems == []
        assert elapsed < _MAX_SCAN_SECONDS, f"{elapsed:.2f}s for {len(hostile)} characters"
