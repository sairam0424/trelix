"""CI installs its security tools from hash-locked requirements, not by name.

SCANS (nothing more): four `run:` steps (security-scan.yml: semgrep and pip-audit;
verify-release.yml: pip-audit; schema-drift.yml: the `[serve]` runtime), the files under
.github/requirements/ that they name, and dependabot.yml. Nothing is installed or fetched here;
pip's own `--require-hashes` is what rejects a wrong file at CI time.

WHY. `pip install semgrep==1.173.0` pins a version, not the bytes: OSSF Scorecard counts it as
unpinned, and a changed upload on the index would run in a job that holds a token or gates a
merge. Each step is one edited line away from going back to a bare install, and nothing else
would fail: Scorecard only rescans after a merge. The `.in` file holds the version a human chose,
the `.txt` is its hashed closure, and Dependabot's `pip` entry for the directory is what keeps
both fresh, so each of the three has a rule here.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
1. A step loses `--require-hashes`, installs a bare package, reads another file, or gains any other
   `pip install` line (`pip`, `pip3`, `pipx`, with `install` right after the program name; a global
   option between them, as in `pip --quiet install x`, is not seen), before or after the lock
   install -> test_the_step_installs_from_its_hashed_lock.
2. schema-drift.yml drops `--no-deps` from the editable install, points it at another path, drops
   its `pip check` or runs it before the editable install (it would verify nothing)
   -> test_the_step_installs_from_its_hashed_lock.
3. A requirement in a lock loses its `--hash` lines, or is no longer an exact pin
   -> test_every_lock_entry_is_an_exact_pin_with_hashes.
4. A `.in` pin and its lock disagree -> test_each_in_file_pin_is_what_its_lock_resolved.
5. The dependabot `pip` entry for /.github/requirements is removed or loses target-branch develop
   -> test_dependabot_proposes_bumps_for_the_locks.
6. The `.in` file of the semgrep or pip-audit lock is deleted
   -> test_a_compiled_lock_keeps_the_in_file_that_holds_its_chosen_version.

NOT COVERED. verify-release.yml's lock is the runner-level install only: scripts/verify_release.py
builds its own venv and installs pip-audit there, unpinned and unhashed. This file cannot see that.
A pip call must also sit on one line: a trailing backslash makes `shlex` raise ValueError, which
fails the step's test (it does not pass it).
"""

from __future__ import annotations

import re
import shlex
from typing import Any

import pytest
import yaml

from tests.unit.workflow_yaml_helpers import GITHUB, load_workflow, steps

_LOCK_DIR = ".github/requirements"
_REQUIREMENTS = GITHUB / "requirements"
_LOCKS = sorted(_REQUIREMENTS.glob("requirements-*.txt"))
# (workflow, step name, lock file, whether the step also installs trelix itself and checks it)
_INSTALLS = (
    ("security-scan.yml", "Install semgrep", "requirements-semgrep.txt", False),
    ("security-scan.yml", "Install pip-audit", "requirements-pip-audit.txt", False),
    (
        "verify-release.yml",
        "Install pip-audit on the runner (verify_release.py installs its own, unhashed)",
        "requirements-pip-audit.txt",
        False,
    ),
    ("schema-drift.yml", "Install trelix + serve extra", "requirements-schema-drift.txt", True),
)
_QUIET_FLAGS = ("-q", "--quiet")
_PIP_WORD = re.compile(r"pip3?(?:\.\d+)?|pipx")
# `name==1.2.3 \` or `name==1.2.3 ; python_full_version < '3.13' \`: how uv writes a pin.
_LOCK_ENTRY = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>\S+)(?: ; .+)? \\$")


def pip_calls(script: str) -> list[tuple[str, list[str]]]:
    """`(install | check, the words after it)` for each `pip install` / `pip check` in a script,
    whether spelled `pip`, `pip3`, `pip3.12` or `pipx`."""
    calls: list[tuple[str, list[str]]] = []
    for line in script.splitlines():
        words = shlex.split(line, comments=True)
        for index in range(len(words) - 1):
            if _PIP_WORD.fullmatch(words[index]) and words[index + 1] in ("install", "check"):
                calls.append((words[index + 1], words[index + 2 :]))
    return calls


def _is_lock_install(tail: list[str], wanted: str) -> bool:
    if "--require-hashes" not in tail or "-r" not in tail:
        return False
    after_flag = tail[tail.index("-r") + 1 : tail.index("-r") + 2]
    bare_words = [word for word in tail if not word.startswith("-") and word != wanted]
    return after_flag == [wanted] and not bare_words


def _is_project_install(tail: list[str]) -> bool:
    return [word for word in tail if word not in _QUIET_FLAGS] == ["--no-deps", "-e", "."]


def _role(verb: str, tail: list[str], wanted: str) -> str:
    """What a pip call is in a step: `lock`, `project`, `check`, or its own text if unexpected."""
    if verb == "check" and not tail:
        return "check"
    if verb == "install" and _is_lock_install(tail, wanted):
        return "lock"
    if verb == "install" and _is_project_install(tail):
        return "project"
    return " ".join(["pip", verb, *tail])


def pip_problems(script: str, lock: str, *, builds_project: bool) -> list[str]:
    """Empty when the step's pip calls are exactly `--require-hashes -r <lock>` and, for a step
    that builds trelix, then `--no-deps -e .`, then `pip check`. Any other `pip install` is
    flagged wherever it sits, and so is a `pip check` that does not come after the project
    install (before it, it verifies nothing; after it, it fails a lock that fell behind a floor in
    pyproject's `dependencies`, though not one in the `serve` extra, which pip check ignores)."""
    wanted = f"{_LOCK_DIR}/{lock}"
    expected = ["lock", "project", "check"] if builds_project else ["lock"]
    found = [_role(verb, tail, wanted) for verb, tail in pip_calls(script)]
    if found == expected:
        return []
    return [f"expected pip calls {expected}, found {found}"]


def lock_problems(text: str) -> list[str]:
    """Every requirement is an exact pin whose next line is a sha256 hash."""
    lines = text.splitlines()
    entries = [(number, line) for number, line in enumerate(lines) if line and line[0] not in "# "]
    if not entries:
        return ["no requirements"]
    problems: list[str] = []
    for number, line in entries:
        following = lines[number + 1].lstrip() if number + 1 < len(lines) else ""
        if not _LOCK_ENTRY.match(line):
            problems.append(f"line {number + 1} is not an exact pin: {line!r}")
        elif not following.startswith("--hash=sha256:"):
            problems.append(f"line {number + 1} has no hash: {line!r}")
    return problems


def _normalised(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pin_problems(in_text: str, lock_text: str) -> list[str]:
    """Every `name==version` in an .in file is the version its lock resolved."""
    locked = {
        _normalised(match["name"]): match["version"]
        for match in map(_LOCK_ENTRY.match, lock_text.splitlines())
        if match
    }
    problems: list[str] = []
    for line in in_text.splitlines():
        pin = line.strip()
        if not pin or pin.startswith("#"):
            continue
        name, _, version = pin.partition("==")
        if locked.get(_normalised(name)) != version:
            problems.append(f"{pin}: the lock has {locked.get(_normalised(name))}")
    return problems


def dependabot_problems(config: dict[str, Any]) -> list[str]:
    entries = [
        entry
        for entry in config.get("updates") or []
        if entry.get("package-ecosystem") == "pip" and entry.get("directory") == f"/{_LOCK_DIR}"
    ]
    if len(entries) != 1:
        return [f"expected one pip entry for /{_LOCK_DIR}, found {len(entries)}"]
    if entries[0].get("target-branch") != "develop":
        return [f"target-branch is {entries[0].get('target-branch')!r}, need 'develop'"]
    return []


def step_script(all_steps: list[dict[str, Any]], name: str) -> str:
    """The `run:` text of the one step called `name`. A name that is missing or used twice is
    refused: with two, only one of them would be checked."""
    found = [step for step in all_steps if step.get("name") == name]
    assert len(found) == 1, f"expected one step named {name!r}, found {len(found)}"
    return str(found[0]["run"])


# ---------------------------------------------------------------------------
# The real files
# ---------------------------------------------------------------------------
def test_the_walk_reached_the_locks() -> None:
    assert len(_LOCKS) >= 3


@pytest.mark.parametrize(("workflow", "step_name", "lock", "builds_project"), _INSTALLS)
def test_the_step_installs_from_its_hashed_lock(
    workflow: str, step_name: str, lock: str, builds_project: bool
) -> None:
    script = step_script(steps(load_workflow(workflow)), step_name)
    assert pip_problems(script, lock, builds_project=builds_project) == []
    assert (_REQUIREMENTS / lock).is_file()


@pytest.mark.parametrize("lock", _LOCKS, ids=lambda path: path.name)
def test_every_lock_entry_is_an_exact_pin_with_hashes(lock: Any) -> None:
    assert lock_problems(lock.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("in_file", sorted(_REQUIREMENTS.glob("*.in")), ids=lambda p: p.name)
def test_each_in_file_pin_is_what_its_lock_resolved(in_file: Any) -> None:
    lock = in_file.with_suffix(".txt")
    assert pin_problems(in_file.read_text("utf-8"), lock.read_text("utf-8")) == []


@pytest.mark.parametrize("lock", ["requirements-semgrep.txt", "requirements-pip-audit.txt"])
def test_a_compiled_lock_keeps_the_in_file_that_holds_its_chosen_version(lock: str) -> None:
    # The parametrized pin test above is built from a glob, so deleting an .in file would
    # delete its test instead of failing it.
    assert (_REQUIREMENTS / lock).with_suffix(".in").is_file()


def test_dependabot_proposes_bumps_for_the_locks() -> None:
    config = yaml.safe_load((GITHUB / "dependabot.yml").read_text(encoding="utf-8"))
    assert dependabot_problems(config) == []


# ---------------------------------------------------------------------------
# The checkers bite: each one is fed a document that breaks its rule
# ---------------------------------------------------------------------------
_LOCK = f"{_LOCK_DIR}/requirements-semgrep.txt"
_LOCK_LINE = f"pip install --quiet --require-hashes -r {_LOCK}"
_PROJECT_LINE = "pip install --quiet --no-deps -e ."


@pytest.mark.parametrize(
    "script",
    [
        'pip install --quiet "semgrep==1.173.0"',
        f"pip install --quiet -r {_LOCK}",
        "pip install --quiet --require-hashes -r .github/requirements/requirements-other.txt",
        f"pip install --quiet --require-hashes -r {_LOCK} semgrep",
        f"pip install --quiet --require-hashes {_LOCK}",
        "echo nothing to install",
        f"{_LOCK_LINE}\npip install --quiet --upgrade fastapi",
        f"pip install --quiet --upgrade fastapi\n{_LOCK_LINE}",
        f"{_LOCK_LINE}\n/tmp/auditor/bin/python -m pip install --quiet pip-audit",
        f"{_LOCK_LINE}\n{_PROJECT_LINE}",
        f"{_LOCK_LINE}\npip3 install --quiet semgrep",
        f"{_LOCK_LINE}\npip3.12 install --quiet semgrep",
        f"{_LOCK_LINE}\npipx install semgrep",
    ],
    ids=[
        "bare-pin",
        "no-require-hashes",
        "other-file",
        "extra-package",
        "no-r-flag",
        "no-install",
        "bare-install-after",
        "bare-install-before",
        "bare-install-by-interpreter-path",
        "project-install-in-a-lock-only-step",
        "pip3-install-after",
        "pip3-minor-install-after",
        "pipx-install-after",
    ],
)
def test_a_step_that_is_not_only_a_hashed_lock_install_is_flagged(script: str) -> None:
    assert len(pip_problems(script, "requirements-semgrep.txt", builds_project=False)) == 1


@pytest.mark.parametrize(
    "script",
    [
        _LOCK_LINE,
        f"python -m pip install --quiet --require-hashes -r {_LOCK}",
        f"set -euo pipefail\n/tmp/auditor/bin/python -m pip install --require-hashes -r {_LOCK}",
        f"# pip install semgrep\n{_LOCK_LINE}  # pip install other",
        f"pip3 install --quiet --require-hashes -r {_LOCK}",
    ],
)
def test_a_hashed_lock_install_is_clean(script: str) -> None:
    assert pip_problems(script, "requirements-semgrep.txt", builds_project=False) == []


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        (f"{_LOCK_LINE}\n{_PROJECT_LINE}\npip check", 0),
        (f"{_LOCK_LINE}\npip install --no-deps -e .\npython -m pip check", 0),
        (f"{_LOCK_LINE}\npip install --quiet -e .\npip check", 1),
        (f"{_LOCK_LINE}\n{_PROJECT_LINE}", 1),
        (f"{_LOCK_LINE}\npip install --quiet -e .", 1),
        (f"{_LOCK_LINE}\npip check\n{_PROJECT_LINE}", 1),
        (f"{_LOCK_LINE}\n{_PROJECT_LINE}\npip install --quiet --upgrade fastapi\npip check", 1),
        (f"pip install --quiet --upgrade fastapi\n{_LOCK_LINE}\n{_PROJECT_LINE}\npip check", 1),
        (f"{_LOCK_LINE}\npip install --quiet --no-deps -e packages/trelix-mcp\npip check", 1),
        (f"{_LOCK_LINE}\n{_PROJECT_LINE}\npip install --no-deps -e packages/x\npip check", 1),
        (f"{_LOCK_LINE}\n{_PROJECT_LINE}\npip check || true", 1),
        (f"{_PROJECT_LINE}\npip check", 1),
    ],
    ids=[
        "clean",
        "clean-without-quiet-through-the-interpreter",
        "no-no-deps",
        "no-pip-check",
        "no-no-deps-and-no-pip-check",
        "pip-check-before-the-project-install",
        "bare-install-between",
        "bare-install-before-the-lock",
        "other-editable-path",
        "second-editable-install",
        "pip-check-cannot-fail-the-step",
        "no-lock-install",
    ],
)
def test_a_step_that_builds_trelix_flags_each_departure(script: str, expected: int) -> None:
    assert len(pip_problems(script, "requirements-semgrep.txt", builds_project=True)) == expected


def test_the_step_is_found_by_its_name() -> None:
    all_steps = [{"name": "Other", "run": "echo a"}, {"name": "Install", "run": "pip install x"}]
    assert step_script(all_steps, "Install") == "pip install x"


@pytest.mark.parametrize("copies", [0, 2], ids=["missing", "duplicated"])
def test_a_step_name_that_is_missing_or_used_twice_is_refused(copies: int) -> None:
    all_steps = [{"name": "Install", "run": "pip install x"}] * copies
    with pytest.raises(AssertionError, match="expected one step named 'Install'"):
        step_script(all_steps, "Install")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a==1.0 \\\n    --hash=sha256:aa\n    # via b\n", 0),
        ("a==1.0 ; python_full_version < '3.13' \\\n    --hash=sha256:aa\n", 0),
        ("a==1.0 \\\n    # via b\n", 1),
        ("a==1.0\n", 1),
        ("a==1.0\n    --hash=sha256:aa\n", 1),
        ("a>=1.0 \\\n    --hash=sha256:aa\n", 1),
        ("a @ https://example.invalid/a.whl\n", 1),
        ("# only a comment\n", 1),
        ("a==1.0 \\\n    --hash=sha1:aa\n", 1),
    ],
    ids=[
        "clean",
        "marker",
        "no-hash",
        "no-continuation",
        "no-backslash",
        "range",
        "url",
        "empty",
        "weak-digest",
    ],
)
def test_the_lock_rule_flags_each_departure(text: str, expected: int) -> None:
    assert len(lock_problems(text)) == expected


@pytest.mark.parametrize(
    ("in_text", "lock_text", "expected"),
    [
        ("semgrep==1.173.0\n", "semgrep==1.173.0 \\\n    --hash=sha256:aa\n", 0),
        ("Pip_Audit==2.10.1\n", "pip-audit==2.10.1 \\\n    --hash=sha256:aa\n", 0),
        ("semgrep==1.173.0\n", "semgrep==1.174.0 \\\n    --hash=sha256:aa\n", 1),
        ("semgrep==1.173.0\n", "other==1.173.0 \\\n    --hash=sha256:aa\n", 1),
    ],
    ids=["same", "normalised-name", "other-version", "missing"],
)
def test_the_pin_rule_flags_each_disagreement(in_text: str, lock_text: str, expected: int) -> None:
    assert len(pin_problems(in_text, lock_text)) == expected


_DEPENDABOT_LOCKS_ENTRY = {
    "package-ecosystem": "pip",
    "directory": "/.github/requirements",
    "target-branch": "develop",
}


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ([_DEPENDABOT_LOCKS_ENTRY], 0),
        ([{"package-ecosystem": "pip", "directory": "/", "target-branch": "develop"}], 1),
        ([{"package-ecosystem": "npm", "directory": "/.github/requirements"}], 1),
        ([{"package-ecosystem": "pip", "directory": "/.github/requirements"}], 1),
        (
            [
                {
                    "package-ecosystem": "pip",
                    "directory": "/.github/requirements",
                    "target-branch": "main",
                }
            ],
            1,
        ),
        ([], 1),
        ([_DEPENDABOT_LOCKS_ENTRY, _DEPENDABOT_LOCKS_ENTRY], 1),
    ],
    ids=[
        "clean",
        "other-directory",
        "other-ecosystem",
        "no-branch",
        "branch-main",
        "no-entries",
        "two-entries-for-the-directory",
    ],
)
def test_the_dependabot_rule_flags_each_departure(
    updates: list[dict[str, str]], expected: int
) -> None:
    assert len(dependabot_problems({"updates": updates})) == expected
