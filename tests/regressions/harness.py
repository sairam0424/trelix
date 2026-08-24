"""The patch-based revert harness: the instrument mutation testing structurally cannot be.

WHY THIS EXISTS
---------------
cargo-mutants' own documentation names the blind spot: "the diff is only matched against
the code under test, not the test code." Every mutation tool in this class shares it --
mutmut, cosmic-ray and cargo-mutants all mutate PRODUCTION source and ask whether the
suite notices. None of them can answer the inverse question, which is the one that matters
after five rounds of test work on this repo: *does a TEST still detect the defect it was
written for?*

Every vacuity fix landed here has been a TEST-SIDE change -- a fixture that started
padding for real, a precondition that fails when it stops discriminating, an oracle moved
off the constant under test. A mutation run over ``src/`` says nothing about any of them.
Weaken ``assert _PHANTOM_ID not in batched`` to ``assert True`` and every mutation score in
this repository stays exactly where it was.

This harness closes that hole by running the experiment in the only direction that
measures it: RESTORE THE DEFECT, then demand that the named tests fail. If a test has been
weakened, deleted, renamed, made vacuous, or silently deselected, the defect comes back
green and this harness fails.

THE THREE ASSERTIONS, and why all three are required
----------------------------------------------------
1. THE PATCH APPLIED CLEANLY. A stale patch is a LOUD ERROR, never a skip. A harness that
   skips when its patch no longer applies is precisely the green-when-vacuous shape this
   repo keeps finding: the file still exists, the suite still passes, and nothing has been
   measured since the day the source moved.
2. EVERY ``must_fail`` NODE ID FAILED. This is the kill.
3. THE REST OF THAT FILE STILL PASSED. Without this, a patch that breaks the module at
   import time would "kill" every test in it and read as a perfect result while proving
   nothing about the specific defect. The remainder is also required to contain at least
   one genuine PASS, so an all-skipped remainder cannot satisfy it.

ANTI-DARK
---------
``test_regressions.py`` additionally asserts that every mapped node id still COLLECTS in
the live tree. A regression suite going dark is invisible otherwise -- and going dark is
exactly how this repo lost the ``integration`` marker for a whole release: registered,
documented as *the* credential-free run, carried by nobody, deselecting nothing.

MECHANICS, and the trap each one closes
---------------------------------------
* Patches are applied in a THROWAWAY ``git worktree``, never in the live tree. Four agents
  once collided by mutating ``src/`` in place.
* The throwaway worktree is checked out at ``HEAD`` and then OVERLAID with the live
  working-tree copies of everything a mapped test reads (see ``OVERLAY_PATHS``). The
  overlay is the whole point: without it the child would run the COMMITTED tests, and a
  weakened test in the working tree -- the one thing this instrument exists to catch --
  would be invisible.
* The child records ``os.getcwd()`` and where ``trelix`` actually resolved from, and
  :func:`run_under_patch` asserts both live inside the throwaway worktree. That is a
  positive control on the measurement itself, not a proxy for it: if the child had run
  against the editable install in the developer's venv, or against the live ``src/``,
  these assertions fail instead of quietly reporting an unpatched run as a survivor.
* Per-node outcomes come from a ``pytest_runtest_logreport`` hook writing JSON, not from
  parsing ``-q`` output or a return code. An exit code cannot distinguish "the two tests I
  named failed" from "collection errored".
* A patch may not touch anything under ``tests/``. An anti-fix that edits the test it is
  checked against is circular, and the check is cheap.
* ``PYTHONDONTWRITEBYTECODE=1`` in the child. A same-length edit was once masked by a
  stale ``.pyc``.

WHAT THIS FILE IS NOT
---------------------
It is not a test file and collects nothing (no ``test_`` prefix). It doubles as the
child-side reporting plugin: when ``REGRESSION_HARNESS_REPORT`` is set in the child's
environment the hooks at the bottom activate, and they are inert otherwise. That variable
name is NOT ``TRELIX_``-prefixed, and the comment on ``_REPORT_PATH`` records the measured
reason why.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGRESSIONS_DIR = Path(__file__).resolve().parent
TESTS_DIR = REGRESSIONS_DIR.parent
REPO_ROOT = TESTS_DIR.parent
PATCHES_DIR = REGRESSIONS_DIR / "patches"
MANIFEST_PATH = REGRESSIONS_DIR / "REGRESSIONS.toml"

# Copied from the LIVE tree over the throwaway worktree's own checkout, which is at HEAD.
#
# ``tests`` is non-negotiable: it is the single thing that makes a working-tree edit to a
# mapped test visible here, and catching such an edit is the reason this file exists.
# ``src`` and ``pyproject.toml`` are what the anti-fix patches touch. ``README.md`` and
# ``packages`` are here because a mapped file reads them -- test_readme_install_commands.py
# parses the root README, every ``packages/*/README.md`` and every ``packages/*/pyproject.toml``
# -- and if they came from HEAD instead, an uncommitted README edit would surface as
# "the patch also broke tests it does not claim to".
#
# STATED LIMITATION: anything NOT in this tuple (docs/, CHANGELOG.md, helm/, scripts/) comes
# from HEAD, so an uncommitted change to one of those is invisible to this harness. That is
# acceptable only because no mapped test reads them today; a new defect entry whose test
# does must add the path here in the same change.
OVERLAY_PATHS = ("src", "tests", "pyproject.toml", "README.md", "packages")

# node_modules is excluded because packages/trelix-typescript has one in any checkout where
# the TS SDK has been built; it is not read by any mapped test and copying it would dominate
# the cost of the whole harness.
_OVERLAY_IGNORE = ("__pycache__", "*.pyc", ".pytest_cache", "node_modules", ".vscode-test")

# Set in the child so it reports outcomes; also the recursion guard read by
# test_regressions.py, which refuses to run inside a child of itself.
REPORT_ENV_VAR = "REGRESSION_HARNESS_REPORT"

# READ ONCE, AT IMPORT, and never re-read from ``os.environ`` afterwards. This is not
# style; the first version of this file was named ``TRELIX_REGRESSION_REPORT`` and
# consulted ``os.environ`` inside the hook, and it MEASURED NOTHING while looking healthy:
# the report file was written on every run with ``"outcomes": {}`` and every defect read as
# a survivor. The cause is tests/_env_isolation.py::SCRUB_PREFIXES == ("TRELIX_",) --
# an autouse fixture monkeypatch-deletes every TRELIX_-prefixed variable for the duration
# of each test, so the hook saw ``None`` during the setup and call phases and the variable
# only reappeared at teardown, the one phase that records nothing. Renaming off the scrubbed
# prefix fixes it once; reading at import time fixes it for any future scrub table, because
# a plugin module is imported at pre-parse, before any fixture can touch the environment.
_REPORT_PATH: Path | None = (
    Path(os.environ[REPORT_ENV_VAR]) if os.environ.get(REPORT_ENV_VAR) else None
)

_OUTCOME_FAILED = "failed"
_OUTCOME_PASSED = "passed"
_OUTCOME_SKIPPED = "skipped"
_OUTCOME_XFAILED = "xfailed"
_OUTCOME_XPASSED = "xpassed"
_OUTCOME_ERROR = "error"

# Outcomes acceptable for the non-mapped remainder of a mapped file. ``xpassed`` is
# deliberately absent: under `strict = true` an xfail that starts passing is already a hard
# failure, so seeing one here means something other than this patch changed.
TOLERATED_REMAINDER_OUTCOMES = frozenset({_OUTCOME_PASSED, _OUTCOME_SKIPPED, _OUTCOME_XFAILED})

_REQUIRED_KEYS = frozenset({"id", "changelog", "summary", "patch", "must_fail"})
_OPTIONAL_KEYS = frozenset({"requires", "deselect", "deselect_reason", "commit"})


class HarnessUnavailable(RuntimeError):
    """The harness cannot run here at all (no git, or not a git checkout).

    Raised rather than swallowed so the caller decides between skipping loudly and
    failing; it must never be turned into a silent pass.
    """


class PatchDidNotApply(AssertionError):
    """``git apply`` refused the patch. A LOUD error, deliberately never a skip.

    A stale patch means the source moved and nobody re-derived the anti-fix. Skipping
    there would leave a defect unguarded while the suite stayed green -- the exact failure
    mode this whole directory was built to remove.
    """


@dataclass(frozen=True)
class Defect:
    """One shipped fix, keyed on PATCH TEXT rather than on any mutation tool's name.

    Mutmut/cargo-mutants identifiers are positional: they move when a line moves, they
    differ between tool versions, and they mean nothing to a reader. A patch file is the
    defect, stated in the only form that can be re-applied and re-verified.
    """

    id: str
    changelog: str
    summary: str
    patch: str
    must_fail: tuple[str, ...]
    requires: tuple[str, ...] = ()
    deselect: tuple[str, ...] = ()
    deselect_reason: str = ""
    commit: str = ""

    @property
    def patch_path(self) -> Path:
        """``patch`` is relative to this directory, so the manifest reads ``patches/x.patch``."""
        return REGRESSIONS_DIR / self.patch

    @property
    def mapped_files(self) -> tuple[str, ...]:
        """The test files named by ``must_fail``, deduplicated, in first-seen order."""
        seen: list[str] = []
        for node_id in self.must_fail:
            path = node_id.split("::", 1)[0]
            if path not in seen:
                seen.append(path)
        return tuple(seen)


@dataclass(frozen=True)
class RunResult:
    """Everything the child reported, plus the controls proving what it measured."""

    outcomes: dict[str, str]
    child_cwd: str
    trelix_origin: str | None
    files_changed: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def _fail_manifest(message: str) -> None:
    raise AssertionError(f"{MANIFEST_PATH.name}: {message}")


def load_defects() -> tuple[Defect, ...]:
    """Parse and VALIDATE the manifest. An unknown key is an error, not a shrug.

    Strict both ways on purpose. A typo'd key name (``must_faill``) would otherwise
    produce a defect entry with an empty kill set, which every assertion below would
    happily satisfy.
    """
    with MANIFEST_PATH.open("rb") as handle:
        raw = tomllib.load(handle)

    entries = raw.get("defect")
    if not isinstance(entries, list) or not entries:
        _fail_manifest("no [[defect]] entries found")
    unexpected_top = set(raw) - {"defect"}
    if unexpected_top:
        _fail_manifest(f"unexpected top-level keys {sorted(unexpected_top)}")

    defects: list[Defect] = []
    for index, entry in enumerate(entries):
        where = f"[[defect]] #{index + 1}"
        keys = set(entry)
        missing = _REQUIRED_KEYS - keys
        if missing:
            _fail_manifest(f"{where} is missing required keys {sorted(missing)}")
        unknown = keys - _REQUIRED_KEYS - _OPTIONAL_KEYS
        if unknown:
            _fail_manifest(
                f"{where} has unknown keys {sorted(unknown)}; a typo'd key would leave "
                f"this defect with an empty kill set and pass vacuously"
            )
        must_fail = tuple(entry["must_fail"])
        if not must_fail:
            _fail_manifest(f"{where} has an empty must_fail; it would assert nothing")
        for node_id in must_fail:
            if "::" not in node_id or not node_id.startswith("tests/"):
                _fail_manifest(f"{where} must_fail entry {node_id!r} is not a tests/... node id")
        deselect = tuple(entry.get("deselect", ()))
        if deselect and not entry.get("deselect_reason", "").strip():
            _fail_manifest(
                f"{where} deselects node ids without a deselect_reason. Every node "
                f"excluded from the measurement has to say why in writing."
            )
        overlap = set(deselect) & set(must_fail)
        if overlap:
            _fail_manifest(
                f"{where} both requires and deselects {sorted(overlap)}, which cannot both hold"
            )
        defects.append(
            Defect(
                id=str(entry["id"]),
                changelog=str(entry["changelog"]),
                summary=str(entry["summary"]),
                patch=str(entry["patch"]),
                must_fail=must_fail,
                requires=tuple(entry.get("requires", ())),
                deselect=deselect,
                deselect_reason=str(entry.get("deselect_reason", "")),
                commit=str(entry.get("commit", "")),
            )
        )

    ids = [defect.id for defect in defects]
    duplicates = sorted({name for name in ids if ids.count(name) > 1})
    if duplicates:
        _fail_manifest(f"duplicate defect ids {duplicates}")
    return tuple(defects)


def patch_target_files(defect: Defect) -> tuple[str, ...]:
    """The paths a patch claims to touch, per ``git apply --numstat`` -- no hand parsing.

    ``--numstat`` also validates that the patch PARSES, which a regex over ``+++`` lines
    does not. A malformed patch therefore fails here rather than later, where an empty
    file list would read as "this patch changes nothing" and pass.
    """
    if not defect.patch_path.is_file():
        raise PatchDidNotApply(f"{defect.id}: patch file {defect.patch_path} does not exist")
    proc = subprocess.run(
        ["git", "apply", "--numstat", "-z", str(defect.patch_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise PatchDidNotApply(
            f"{defect.id}: `git apply --numstat` could not parse "
            f"{defect.patch}:\n{proc.stderr.strip()}"
        )
    # `--numstat -z` emits one NUL-terminated record per file, "<added>\t<removed>\t<path>".
    # Only the path is NUL-terminated; the counts are tab-separated inside the record. The
    # -z form is used so a path containing a space or a quote cannot be mis-split.
    records = [record for record in proc.stdout.split("\0") if record]
    paths = tuple(record.split("\t", 2)[2] for record in records if record.count("\t") >= 2)
    if not paths:
        raise PatchDidNotApply(f"{defect.id}: {defect.patch} names no files at all")
    return paths


def unavailable_reason() -> str | None:
    """Why the harness cannot run here, or ``None`` if it can.

    Public so a test can skip LOUDLY naming the missing capability rather than passing on
    an environment where nothing was measured.
    """
    try:
        _require_git_checkout()
    except HarnessUnavailable as exc:
        return str(exc)
    return None


def _require_git_checkout() -> None:
    if shutil.which("git") is None:
        raise HarnessUnavailable(
            "the `git` executable is not on PATH, so no throwaway worktree can be "
            "created; this harness cannot measure anything here"
        )
    proc = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        raise HarnessUnavailable(
            f"{REPO_ROOT} is not inside a git work tree (an installed sdist, for "
            f"example), so `git worktree add` is unavailable"
        )


def _overlay_live_tree(worktree: Path) -> None:
    """Replace the throwaway checkout's src/tests/pyproject with the LIVE ones.

    Without this the child would measure the COMMITTED tests. The whole reason this
    instrument exists is to notice a weakened test in the WORKING TREE, so measuring the
    committed copy would make it structurally blind to its own subject.
    """
    ignore = shutil.ignore_patterns(*_OVERLAY_IGNORE)
    for name in OVERLAY_PATHS:
        source = REPO_ROOT / name
        target = worktree / name
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target, ignore=ignore, symlinks=True)
        elif source.is_file():
            shutil.copy2(source, target)
        else:  # pragma: no cover - a missing src/ or tests/ is not a reachable state
            raise HarnessUnavailable(f"{source} is neither a file nor a directory")


def _content_snapshot(worktree: Path) -> dict[str, str]:
    """relpath -> sha256 for every file in the throwaway worktree.

    Taken immediately after the overlay and again after ``git apply``, so the difference
    is exactly what the patch did. ``git diff`` cannot serve here: the overlay
    deliberately leaves the worktree dirty relative to HEAD, so git's diff conflates the
    two. Hashing also catches a patch that "applies" while changing nothing.
    """
    snapshot: dict[str, str] = {}
    for path in worktree.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        parts = path.relative_to(worktree).parts
        if ".git" in parts or "__pycache__" in parts or ".pytest_cache" in parts:
            continue
        snapshot["/".join(parts)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _child_env(worktree: Path, report_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env[REPORT_ENV_VAR] = str(report_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # The throwaway root ONLY, so `-p tests.regressions.harness` is importable at
    # pre-parse time. Deliberately NOT `<worktree>/src`: `trelix` must resolve through
    # pyproject's own `pythonpath = ["src", "."]` relative to the rootdir under test, and
    # inheriting an outer PYTHONPATH pointing at another src/ is exactly how a "mutated"
    # run silently measures the unmutated tree.
    env["PYTHONPATH"] = str(worktree)
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    return env


def run_under_patch(defect: Defect) -> RunResult:
    """Apply ``defect``'s patch in a throwaway worktree and run its mapped test files.

    Raises :class:`PatchDidNotApply` if the patch is stale, :class:`HarnessUnavailable`
    if no worktree can be made. Never returns a "nothing measured" result: the controls
    below assert the child ran inside the throwaway tree against the patched source.
    """
    _require_git_checkout()
    expected_changes = patch_target_files(defect)

    parent = Path(tempfile.mkdtemp(prefix="trelix-regress-"))
    worktree = parent / "wt"
    added = False
    try:
        add = subprocess.run(
            ["git", "worktree", "add", "--detach", "--quiet", str(worktree), "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if add.returncode != 0:
            raise HarnessUnavailable(
                f"`git worktree add` failed: {add.stderr.strip() or add.stdout.strip()}"
            )
        added = True
        _overlay_live_tree(worktree)
        before = _content_snapshot(worktree)

        applied = subprocess.run(
            ["git", "apply", "--verbose", str(defect.patch_path)],
            cwd=worktree,
            capture_output=True,
            text=True,
            check=False,
        )
        if applied.returncode != 0:
            raise PatchDidNotApply(
                f"{defect.id}: {defect.patch} no longer applies to today's tree.\n"
                f"This is a hard error and must not be skipped: the source moved and the "
                f"anti-fix was never re-derived, so the defect it names is UNGUARDED.\n"
                f"Re-derive it from {defect.commit or 'the shipped fix'} and re-verify "
                f"that {list(defect.must_fail)} still fail under it.\n"
                f"git apply said:\n{applied.stderr.strip()}"
            )

        # Content hashes, not `git diff --name-only`: the overlay above already leaves the
        # worktree dirty relative to HEAD (that is its job), so git's own diff cannot
        # separate the overlay from the patch. Comparing before/after hashes measures
        # exactly one thing -- what THIS patch changed.
        after = _content_snapshot(worktree)
        files_changed = tuple(
            sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
        )
        if files_changed != tuple(sorted(expected_changes)):
            raise PatchDidNotApply(
                f"{defect.id}: `git apply` reported success but the files whose CONTENT "
                f"actually changed were {list(files_changed)}, not the "
                f"{sorted(expected_changes)} the patch names. A patch that applies to "
                f"nothing is not a restored defect."
            )

        report_path = parent / "outcomes.json"
        argv = [
            sys.executable,
            "-m",
            "pytest",
            *defect.mapped_files,
            "-p",
            "no:cacheprovider",
            "-p",
            "tests.regressions.harness",
            "--no-header",
            "-q",
            "--tb=no",
        ]
        for node_id in defect.deselect:
            argv += ["--deselect", node_id]
        proc = subprocess.run(
            argv,
            cwd=worktree,
            capture_output=True,
            text=True,
            env=_child_env(worktree, report_path),
            check=False,
            timeout=600,
        )
        if not report_path.is_file():
            raise AssertionError(
                f"{defect.id}: the child pytest wrote no outcome report, so nothing was "
                f"measured. exit={proc.returncode}\nstdout:\n{proc.stdout[-4000:]}\n"
                f"stderr:\n{proc.stderr[-2000:]}"
            )
        payload: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        result = RunResult(
            outcomes=dict(payload["outcomes"]),
            child_cwd=str(payload["cwd"]),
            trelix_origin=payload["trelix_origin"],
            files_changed=files_changed,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    finally:
        if added:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        shutil.rmtree(parent, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    _assert_child_measured_the_patched_tree(defect, worktree, result)
    return result


def _assert_child_measured_the_patched_tree(
    defect: Defect, worktree: Path, result: RunResult
) -> None:
    """The harness's own positive control, asserted rather than assumed.

    A child that ran in the wrong directory, or that imported ``trelix`` from the
    developer's editable install instead of the patched ``src/``, would report every
    mapped test as PASSING -- indistinguishable from "the test no longer detects the
    defect" unless something checks. This is that something.
    """
    expected = str(worktree)
    assert os.path.realpath(result.child_cwd) == os.path.realpath(expected), (
        f"{defect.id}: the child pytest ran in {result.child_cwd!r}, not in the "
        f"throwaway worktree {expected!r}; whatever it measured, it was not the patch"
    )
    assert result.trelix_origin is not None, (
        f"{defect.id}: the child never resolved the `trelix` package at all, so it "
        f"cannot have exercised the patched source"
    )
    origin = Path(result.trelix_origin).resolve()
    assert origin.is_relative_to(Path(expected).resolve()), (
        f"{defect.id}: the child imported trelix from {origin}, which is OUTSIDE the "
        f"throwaway worktree {expected}. The patched source was not what ran -- most "
        f"likely an editable install or an inherited PYTHONPATH won the import."
    )


def evaluate(defect: Defect, result: RunResult) -> None:
    """The three assertions. Raises ``AssertionError`` on the first that does not hold.

    Factored out of the test so it can itself be shown to FAIL: ``test_regressions.py``
    re-evaluates a real, already-collected :class:`RunResult` against a deliberately wrong
    ``must_fail`` list and requires this function to object. Without that control, a green
    run here would only mean "``evaluate`` returned", which is what a function whose
    assertions had been weakened also does.
    """
    outcomes = result.outcomes
    assert outcomes, (
        f"{defect.id}: the child reported no outcomes at all, so nothing was measured "
        f"(child exit {result.returncode})"
    )

    # (1) Every mapped node ran. A node id that no longer collects would otherwise
    # silently drop out of the kill set -- the dark-suite failure, seen from inside the
    # patched run rather than from the collection probe.
    absent = [node_id for node_id in defect.must_fail if node_id not in outcomes]
    assert not absent, (
        f"{defect.id}: mapped node id(s) did not run under the patch: {absent}. Either "
        f"they were renamed/deleted (update REGRESSIONS.toml in the same change) or "
        f"something deselected them. A kill set that shrinks silently is a regression "
        f"suite going dark."
    )

    # (2) Every mapped node FAILED. This is the kill.
    survivors = {
        node_id: outcomes[node_id]
        for node_id in defect.must_fail
        if outcomes[node_id] != _OUTCOME_FAILED
    }
    assert not survivors, (
        f"{defect.id}: the defect was restored and these tests DID NOT FAIL: "
        f"{survivors}.\n{defect.summary}\n"
        f"An outcome of {_OUTCOME_PASSED!r} means the test no longer detects the defect it "
        f"was written for -- it has been weakened, made vacuous, or its subject moved. "
        f"An outcome of {_OUTCOME_ERROR!r} or {_OUTCOME_SKIPPED!r} means it never got to "
        f"express an opinion, which is not a kill either."
    )

    # (3) The REST of each mapped file still passes, and the remainder is not empty of
    # real passes. Without this, a patch that breaks the module at import time would
    # "kill" everything in the file and read as a flawless result while proving nothing
    # about this particular defect.
    remainder = set(outcomes) - set(defect.must_fail) - set(defect.deselect)
    collateral = {
        node_id: outcomes[node_id]
        for node_id in sorted(remainder)
        if outcomes[node_id] not in TOLERATED_REMAINDER_OUTCOMES
    }
    assert not collateral, (
        f"{defect.id}: the patch also broke tests it does not claim to: {collateral}. "
        f"Either the anti-fix is wider than the defect (make it one site), or those node "
        f"ids belong in must_fail because they genuinely detect it too."
    )
    still_passing = sorted(node_id for node_id in remainder if outcomes[node_id] == _OUTCOME_PASSED)
    assert still_passing, (
        f"{defect.id}: not one test outside must_fail actually PASSED under the patch "
        f"(remainder outcomes: { {n: outcomes[n] for n in sorted(remainder)} }). A "
        f"remainder that is entirely skipped cannot show that the patch is narrow."
    )


def collect_node_ids(paths: tuple[str, ...]) -> frozenset[str]:
    """Node ids a fresh pytest collects for ``paths`` in the LIVE tree.

    A child process, not this session's ``session.items``: under ``-k``, ``-m`` or a
    single-file invocation ``session.items`` holds only what THIS run selected, so an
    assertion built on it reports green while measuring nothing.
    """
    argv = [
        sys.executable,
        "-m",
        "pytest",
        *paths,
        "--collect-only",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop(REPORT_ENV_VAR, None)
    proc = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True, env=env, timeout=300)
    assert proc.returncode in (0, 5), (
        f"`pytest --collect-only {list(paths)}` exited {proc.returncode}; this probe is "
        f"broken, not measuring.\nstdout tail:\n{proc.stdout[-3000:]}\n"
        f"stderr tail:\n{proc.stderr[-2000:]}"
    )
    return frozenset(
        line.strip()
        for line in proc.stdout.splitlines()
        if "::" in line and line.startswith("tests/")
    )


# ---------------------------------------------------------------------------
# Child-side reporting plugin. Inert unless REGRESSION_HARNESS_REPORT was set in the
# environment at import, so importing this module in the parent session costs nothing.
#
# Per-node outcomes, not an exit code: exit 1 cannot distinguish "the two tests I named
# failed" from "the module errored on import and took the whole file with it", and those
# two readings are the difference between a kill and a meaningless result.
# ---------------------------------------------------------------------------
_OUTCOMES: dict[str, str] = {}


def running_as_child() -> bool:
    """True inside a child pytest spawned by :func:`run_under_patch`.

    ``test_regressions.py`` refuses to run when this holds, so a stray collection of
    ``tests/regressions`` inside a child cannot recurse.
    """
    return _REPORT_PATH is not None


def pytest_runtest_logreport(report: Any) -> None:  # pragma: no cover - child only
    if _REPORT_PATH is None:
        return
    node_id = report.nodeid
    if report.failed:
        # A setup/teardown failure is an ERROR, and an error is NOT a kill: the test
        # never got as far as expressing an opinion about the defect. Kept distinct so
        # `must_fail` cannot be satisfied by a collection or fixture explosion.
        if _OUTCOMES.get(node_id) != _OUTCOME_FAILED:
            _OUTCOMES[node_id] = _OUTCOME_FAILED if report.when == "call" else _OUTCOME_ERROR
    elif report.skipped:
        _OUTCOMES.setdefault(
            node_id,
            _OUTCOME_XFAILED if hasattr(report, "wasxfail") else _OUTCOME_SKIPPED,
        )
    elif report.when == "call":
        _OUTCOMES[node_id] = _OUTCOME_XPASSED if hasattr(report, "wasxfail") else _OUTCOME_PASSED


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:  # pragma: no cover
    path = _REPORT_PATH
    if path is None:
        return
    origin: str | None = None
    module = sys.modules.get("trelix")
    if module is not None and getattr(module, "__file__", None):
        origin = module.__file__
    else:
        try:
            spec = importlib.util.find_spec("trelix")
        except (ImportError, ValueError):  # pragma: no cover - defensive
            spec = None
        if spec is not None and spec.origin:
            origin = spec.origin
    path.write_text(
        json.dumps(
            {
                "outcomes": _OUTCOMES,
                "cwd": os.getcwd(),
                "trelix_origin": origin,
                "exitstatus": int(exitstatus),
            }
        ),
        encoding="utf-8",
    )
