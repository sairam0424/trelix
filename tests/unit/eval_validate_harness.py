"""Shared helpers for the `trelix eval-validate` tests.

Not a test module (no ``test_`` prefix); imported as ``tests.unit.eval_validate_harness``.
The ``repo`` fixture is a real two-commit git repository in which ``src/old.py`` is renamed to
``src/new.py``, so "the path exists at HEAD" and "at the older revision" give opposite answers
and a check that ignores ``--rev`` cannot pass both.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from trelix.cli.main import app

NOTE_NO_REPO = "note: --repo was not given, so relevant_files paths were not checked"
NOTE_V1 = (
    "note: no entry has a kind or a gold_status, so this file is checked as v1: only the "
    "schema, duplicate and (with --repo) path checks ran"
)


def invoke_eval_validate(*args: str) -> Result:
    """Run `trelix eval-validate` with stdout and stderr kept apart on any click version."""
    try:
        runner = CliRunner(mix_stderr=False)  # type: ignore[call-arg]
    except TypeError:  # click >= 8.2 has no mix_stderr; its result always separates them
        runner = CliRunner()
    return runner.invoke(app, ["eval-validate", *args])


def write_golden(tmp_path: Path, entries: list[dict[str, Any]], name: str = "golden.jsonl") -> str:
    path = tmp_path / name
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    return str(path)


def golden_entry(query: str, *files: str, **extra: Any) -> dict[str, Any]:
    return {"query": query, "relevant_files": list(files) or ["src/new.py"], **extra}


def run_git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603
        [
            "git",
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "user.name=t",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Two commits: `src/old.py` and `README.md`, then `src/old.py` renamed to `src/new.py`."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    run_git(root, "init", "-q")
    (root / "src" / "old.py").write_text("x = 1\n", encoding="utf-8")
    (root / "README.md").write_text("readme\n", encoding="utf-8")
    run_git(root, "add", "src/old.py", "README.md")
    run_git(root, "commit", "-q", "-m", "first")
    run_git(root, "mv", "src/old.py", "src/new.py")
    run_git(root, "commit", "-q", "-m", "rename")
    return root


def v2_entries(kind: str, count: int, status: str = "validated") -> list[dict[str, Any]]:
    """`count` v2 entries of one kind, each with a distinct query and id."""
    return [
        golden_entry(f"{kind} question {i}", id=f"{kind}-{i}", kind=kind, gold_status=status)
        for i in range(count)
    ]
