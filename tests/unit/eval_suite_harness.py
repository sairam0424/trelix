"""Shared helpers for the `trelix eval-suite` tests.

Not a test module (no ``test_`` prefix); imported as ``tests.unit.eval_suite_harness``.

``make_remote`` builds the "remote" the suite clones: a real two-commit git repository in a
temp directory, so a clone, a checkout at a pin and a moved branch are all the real thing and
nothing touches the network. Commit 1 holds ``src/app.py`` and ``README.md`` (the gold files)
and three files that run code if anything ever runs them (``conftest.py``, ``setup.py``,
``sitecustomize.py``, each writing an ``EXECUTED`` marker beside itself). Commit 2 changes
``src/app.py`` and adds ``src/extra.py``, so "the pin" and "the branch head" give opposite
answers and a checkout of HEAD cannot pass for a checkout of the pin.

The repository is cloned by path, which production refuses (``repo.url`` must be an https URL
on ``ALLOWED_REPO_HOSTS``); ``load_local`` and ``clone_local`` pass the explicit ``allow_local``
keyword that only the Python API has. Every git command here runs with the operator's git
configuration switched off, like the code under test.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from trelix.eval.suite import SuiteSpec, load_suite
from trelix.eval.suite_git import ensure_clone

CANARY = "import pathlib\npathlib.Path(__file__).with_name('EXECUTED').write_text('x')\n"
FIRST_FILES: Mapping[str, str] = {
    "src/app.py": "def login(user):\n    return user\n",
    "README.md": "demo\n",
    "conftest.py": CANARY,
    "setup.py": CANARY,
    "sitecustomize.py": CANARY,
}
SECOND_FILES: Mapping[str, str] = {
    "src/app.py": "def login(user):\n    return user.name\n",
    "src/extra.py": "X = 1\n",
}
GOLDEN: tuple[dict[str, Any], ...] = (
    {"query": "how does login work", "relevant_files": ["src/app.py"]},
    {"query": "what is in the readme", "relevant_files": ["README.md"]},
)
# The sha256 of the two lines of GOLDEN as `write_suite` writes them (`shasum -a 256` of the
# two lines typed out, not of anything this module produced).
GOLDEN_SHA256 = "8fea0361cdb46af03f5587f5faa2177f59d63df3f9e97c5e6228bb874a1ba9f7"


@dataclass(frozen=True)
class Remote:
    """A git repository with two commits: `first` and `second` are their ids."""

    path: Path
    first: str
    second: str


def git_env() -> dict[str, str]:
    """A git environment with no operator configuration and no inherited GIT_* variable."""
    return {
        "PATH": os.environ.get("PATH", ""),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }


def git(cwd: Path, *args: str) -> str:
    """Run `git *args` in `cwd` with a committer identity and no operator configuration."""
    result = subprocess.run(  # noqa: S603
        [
            "git",
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "user.name=t",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "protocol.file.allow=always",
            *args,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=git_env(),
    )
    return result.stdout.strip()


def write_files(root: Path, files: Mapping[str, str]) -> None:
    for name, text in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def commit_all(root: Path, message: str) -> str:
    git(root, "add", "-A", "-f")
    git(root, "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD")


def make_remote(
    parent: Path,
    first_files: Mapping[str, str] = FIRST_FILES,
    second_files: Mapping[str, str] = SECOND_FILES,
    name: str = "remote",
    links: Mapping[str, Path] | None = None,
) -> Remote:
    """Two commits in `parent/name`: `first_files`, then `second_files` written over them.

    `links` maps a path to the absolute target of a symlink committed with the first files.
    """
    root = parent / name
    root.mkdir(parents=True)
    git(root, "init", "-q", "-b", "main")
    write_files(root, first_files)
    for link, target in (links or {}).items():
        (root / link).parent.mkdir(parents=True, exist_ok=True)
        (root / link).symlink_to(target)
    first = commit_all(root, "first")
    write_files(root, second_files)
    return Remote(root, first, commit_all(root, "second"))


def plan_record(query: str) -> dict[str, Any]:
    """One line of a plans file: the shape `_FrozenPlanCache` writes and reads."""
    return {
        "plan": {
            "execution_mode": "parallel",
            "intent": "feature_flow",
            "raw_query": query,
            "routing_tier": 2,
            "sub_queries": [
                {
                    "bm25_tokens": ["login"],
                    "file_hints": [],
                    "grep_hints": [],
                    "hyde_snippet": "",
                    "semantic_query": query,
                }
            ],
        },
        "project_context": None,
        "query": query,
    }


def jsonl(records: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode("utf-8")


def write_suite(
    directory: Path,
    url: str,
    sha: str,
    *,
    name: str = "demo",
    golden: list[dict[str, Any]] | None = None,
    plans: list[str] | None = None,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Write `suite.json` with its golden and plans files; return the path of `suite.json`.

    `plans` are the queries a plan is recorded for (the golden queries by default). `mutate`
    edits the document last, after the hashes are computed, to plant a fault.
    """
    directory.mkdir(parents=True, exist_ok=True)
    entries = list(GOLDEN) if golden is None else golden
    golden_bytes = jsonl(entries)
    plan_queries = [e["query"] for e in entries] if plans is None else plans
    plans_bytes = jsonl([plan_record(q) for q in plan_queries])
    (directory / "golden.jsonl").write_bytes(golden_bytes)
    (directory / "plans.jsonl").write_bytes(plans_bytes)
    doc: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "golden_version": "v1",
        "repo": {"url": url, "sha": sha, "license": "MIT"},
        "golden": {"path": "golden.jsonl", "sha256": hashlib.sha256(golden_bytes).hexdigest()},
        "plans": {"path": "plans.jsonl", "sha256": hashlib.sha256(plans_bytes).hexdigest()},
    }
    if mutate is not None:
        mutate(doc)
    suite_json = directory / "suite.json"
    suite_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return suite_json


@dataclass(frozen=True)
class GitCall:
    """One `subprocess.run` of the code under test: what it ran, with which environment."""

    argv: list[str]
    env: dict[str, str]
    timeout: float | None
    stdin: int | None


class _SubprocessSpy:
    """Stands in for the `subprocess` module of `trelix.eval.suite_git`: records, then runs."""

    def __init__(self, calls: list[GitCall]) -> None:
        self._calls = calls

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self._calls.append(
            GitCall(
                list(argv),
                dict(kwargs.get("env") or {}),
                kwargs.get("timeout"),
                kwargs.get("stdin"),
            )
        )
        return subprocess.run(argv, **kwargs)  # noqa: S603

    def __getattr__(self, name: str) -> Any:
        return getattr(subprocess, name)


def record_git(monkeypatch: pytest.MonkeyPatch) -> list[GitCall]:
    """Record every git command `trelix.eval.suite_git` runs during the test; they still run."""
    calls: list[GitCall] = []
    monkeypatch.setattr("trelix.eval.suite_git.subprocess", _SubprocessSpy(calls))
    return calls


def load_local(path: Path | str) -> SuiteSpec:
    """`load_suite` as only the tests may call it: `repo.url` may be a local path."""
    return load_suite(path, allow_local=True)


def clone_local(spec: SuiteSpec, cache_root: Path) -> Path:
    """`ensure_clone` as only the tests may call it: git may use its `file` transport."""
    return ensure_clone(spec, cache_root, allow_local=True)


@pytest.fixture(scope="module")
def remote(tmp_path_factory: pytest.TempPathFactory) -> Remote:
    """The demo repository, built once per test module; a test must not change it."""
    return make_remote(tmp_path_factory.mktemp("eval-suite-remote"))
