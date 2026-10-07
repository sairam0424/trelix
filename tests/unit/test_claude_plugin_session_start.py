"""The plugin's SessionStart hook tells Claude whether the repository has a trelix index.

WHY. The pinned `trelix-mcp==3.4.3` sends no server instructions and answers a search on an
unindexed repository with `results: []` while leaving an empty index behind, so the one line
`plugins/trelix/scripts/session_start.py` prints is the only place the model learns whether
`search_code` can answer, how stale the index is, and which absolute `repo_path` to pass. The hook
runs on every session start with the user's permissions, so these tests pin the whole surface: one
hook event and command, standard-library-only imports and no network call, a read-only open that
creates nothing under the project, the three line shapes, the 1,000 character cap, and exit 0 with
an empty stdout on every bad input. Every expected value is a literal (the commit hashes are
observations of a git repository the test builds). The script runs as a child process with a
minimal environment, so it sees only what Claude Code would give it (the AST allow-list test is what
forbids a `trelix` import; the venv interpreter could otherwise find it); `describe`,
`cap` and `read_only_uri` are also loaded through `importlib` for the cases that need no process.
"""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from trelix.core.config import IndexConfig
from trelix.store.db import Database
from trelix.store.provenance import IndexProvenance, write_provenance

_ROOT = Path(__file__).resolve().parents[2]
PLUGIN = _ROOT / "plugins" / "trelix"
HOOKS_JSON = PLUGIN / "hooks" / "hooks.json"
SCRIPT = PLUGIN / "scripts" / "session_start.py"

# The whole hooks file: one event, one matcher, one shell-form command with the plugin root quoted
# (`claude plugin validate` warns otherwise), a 15 s timeout (default 600 s). Other events are B-5.
EXPECTED_HOOKS_JSON = {
    "description": (
        "trelix: tell Claude at session start whether this repository has a trelix index"
    ),
    "hooks": {
        "SessionStart": [
            {
                "matcher": "startup|resume|clear",
                "hooks": [
                    {
                        "type": "command",
                        "command": (
                            "uv run --no-project --script "
                            '"${CLAUDE_PLUGIN_ROOT}/scripts/session_start.py"'
                        ),
                        "timeout": 15,
                    }
                ],
            }
        ]
    },
}
# `uv run --no-project --script` runs the script in a bare interpreter: standard library only.
ALLOWED_IMPORT_ROOTS = frozenset(
    "__future__ collections dataclasses json os pathlib sqlite3 subprocess sys urllib".split()
)
# The PEP 723 header `uv run --script` reads, byte for byte.
PEP723_HEADER = ["# /// script", '# requires-python = ">=3.12"', "# dependencies = []", "# ///"]
INDEXED_AT = "2026-10-01T00:00:00+00:00"
# A well-formed commit hash no test repository contains.
FOREIGN_COMMIT = "0123456789abcdef0123456789abcdef01234567"
NO_PROJECT = "trelix session_start: skipped (no project directory)\n"


def _git(repo: Path, home: Path, *args: str) -> str:
    """Run git in `repo` with a fixed identity and no user or system config."""
    result = subprocess.run(  # noqa: S603
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(home),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
        },
    )
    return result.stdout.strip()


def _commit(repo: Path, home: Path, name: str) -> str:
    """Add one file, commit it, return the new HEAD hash."""
    (repo / name).write_text(name + "\n", encoding="utf-8")
    _git(repo, home, "add", name)
    _git(repo, home, "commit", "-q", "-m", name)
    return _git(repo, home, "rev-parse", "HEAD")


def _make_index(
    repo: Path, *, files: int = 0, symbols: int = 0, provenance: IndexProvenance | None = None
) -> Path:
    """A real index where trelix puts it (`IndexConfig(...).db_path_absolute`): `files` rows in
    `files`, `symbols` rows in `symbols`, and the provenance an index run would have recorded."""
    db_path = IndexConfig(repo_path=str(repo)).db_path_absolute
    db = Database(db_path)
    if provenance is not None:
        write_provenance(db, provenance)
    db.close()
    conn = sqlite3.connect(db_path)
    with conn:
        for i in range(files):
            conn.execute(
                "INSERT INTO files (path, rel_path, language, hash, size_bytes) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"{repo}/f{i}.py", f"f{i}.py", "python", f"h{i}", 10),
            )
        for i in range(symbols):
            conn.execute(
                "INSERT INTO symbols (file_id, name, qualified_name, kind, line_start, line_end) "
                "VALUES (1, ?, ?, 'function', 1, 2)",
                (f"s{i}", f"f0.s{i}"),
            )
    conn.close()
    return db_path


def _hook_stdin(cwd: Path) -> str:
    """What Claude Code writes to a SessionStart hook's stdin."""
    return json.dumps(
        {
            "session_id": "s1",
            "cwd": str(cwd),
            "hook_event_name": "SessionStart",
            "source": "startup",
        }
    )


def _run(stdin_text: str, env: dict[str, str], *, home: Path) -> subprocess.CompletedProcess[str]:
    """The script as Claude Code runs it: a child process, stdin JSON, a minimal environment."""
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT)],
        input=stdin_text,
        env={"PATH": os.environ["PATH"], "HOME": str(home), **env},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


class _SourceOnlyLoader(importlib.machinery.SourceFileLoader):
    """A loader that never writes `__pycache__`: a `.pyc` under plugins/trelix/scripts/ would be
    a seventh plugin file to the manifest test's file list and contents hash."""

    def set_data(self, path: str, data: bytes, *, _mode: int = 0o666) -> None:
        return None


def _load_script() -> ModuleType:
    name = "trelix_plugin_session_start"
    spec = importlib.util.spec_from_file_location(
        name, SCRIPT, loader=_SourceOnlyLoader(name, str(SCRIPT))
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The script's frozen dataclass resolves its `from __future__` annotations through
    # sys.modules while the class is built (the importlib recipe registers before exec).
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _import_roots(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None and node.level == 0, "no relative imports in a script"
            roots.add(node.module.split(".")[0])
    return roots


def test_hooks_json_runs_only_the_session_start_script() -> None:
    """Whole-object equality with the hooks file, and the script it names exists. MUTATIONS that
    must fail: add a `PreToolUse` group; unquote `${CLAUDE_PLUGIN_ROOT}`; change the timeout; add
    a `compact` matcher."""
    data = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
    assert data == EXPECTED_HOOKS_JSON
    assert set(data["hooks"]) == {"SessionStart"}, "hooks other than SessionStart are B-5"
    assert SCRIPT.is_file()


def test_session_start_imports_only_the_standard_library() -> None:
    """Every import root is in the literal allow-list, `trelix` and `trelix_mcp` are absent, the
    PEP 723 header is exactly the four lines `uv run --script` reads, and the name `urlopen`
    appears nowhere (`urllib` is imported for `pathname2url` only). MUTATIONS that must fail:
    `import trelix`; `import re`; `dependencies = ["rich"]`; write `urlopen` anywhere."""
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.splitlines()[:4] == PEP723_HEADER

    roots = _import_roots(ast.parse(source))
    assert roots <= ALLOWED_IMPORT_ROOTS, f"imports outside the standard library: {sorted(roots)}"
    assert "trelix" not in roots and "trelix_mcp" not in roots
    assert "urlopen" not in source, "the SessionStart hook must not be able to reach the network"
    # The read-only open is the only connect in the script (a plain `sqlite3.connect(db_path)` is an
    # equivalent mutant behaviourally today, so the property is pinned at the source level).
    assert source.count("sqlite3.connect(") == 1
    assert "sqlite3.connect(read_only_uri(db_path), uri=True)" in source


def test_indexed_repository_facts_line(tmp_path: Path) -> None:
    """Counts, build time, the 12-character commit, the distance and the embedder provider (never
    the model name), then the absolute `repo_path`; at most 1,000 characters. MUTATIONS that must
    fail: swap the two COUNT queries; `sha[:12]` to `sha[:7]`; reverse the rev-list range
    (`HEAD..<commit>` is 0, so `(= HEAD)`); print `embedder_model` instead of the provider."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, tmp_path, "init", "-q")
    sha = _commit(repo, tmp_path, "a.txt")
    _commit(repo, tmp_path, "b.txt")
    _commit(repo, tmp_path, "c.txt")
    _make_index(
        repo,
        files=3,
        symbols=7,
        provenance=IndexProvenance(
            git_commit=sha,
            indexed_at=INDEXED_AT,
            embedder_provider="local",
            embedder_model="all-MiniLM-L6-v2",
        ),
    )

    result = _run(_hook_stdin(repo), {"CLAUDE_PROJECT_DIR": str(repo)}, home=tmp_path)

    assert result.returncode == 0
    assert result.stdout == (
        f"trelix: {repo} is indexed: 3 files, 7 symbols; built 2026-10-01T00:00:00+00:00 from "
        f"commit {sha[:12]} (HEAD is 2 commits ahead); embedder local. "
        f'Pass repo_path="{repo}" to every trelix tool.\n'
    )
    assert len(result.stdout) <= 1000
    assert result.stderr == ""


@pytest.mark.parametrize("trelix_dir_exists", [False, True], ids=["bare", "dot-trelix-only"])
def test_unindexed_repository_is_reported_and_nothing_is_created(
    tmp_path: Path, trelix_dir_exists: bool
) -> None:
    """The exact "no index" line, exit 0, and no `.trelix/` (or no `index.db` inside an existing
    `.trelix/`, the state `trelix index --dry-run` or a deleted index leaves) afterwards: on the
    pinned release a search creates an empty index; the hook must not. MUTATIONS that must fail:
    remove the `is_file()` check in `main` (the read-only open of a missing file raises, so stdout
    is empty); build the path with `IndexConfig(...).db_path_absolute` (creates `.trelix/`); write
    the "is indexed" line for a missing index."""
    repo = tmp_path / "repo"
    repo.mkdir()
    if trelix_dir_exists:
        (repo / ".trelix").mkdir()
        (repo / ".trelix" / ".gitignore").write_text("*\n", encoding="utf-8")

    result = _run(_hook_stdin(repo), {"CLAUDE_PROJECT_DIR": str(repo)}, home=tmp_path)

    assert result.returncode == 0
    assert result.stdout == (
        f"trelix: no index at {repo}/.trelix/index.db. "
        f'Call index_codebase(repo_path="{repo}") before search_code; until then, use grep.\n'
    )
    assert result.stderr == ""
    if trelix_dir_exists:
        assert sorted(p.name for p in (repo / ".trelix").iterdir()) == [".gitignore"]
    else:
        assert not (repo / ".trelix").exists()
        assert list(repo.iterdir()) == []


def test_empty_index_is_reported_as_empty(tmp_path: Path) -> None:
    """An index with the schema and no rows (what a 3.4.3 `search_code` on an unindexed repository
    leaves behind) is "empty (0 files)", not "indexed: 0 files, 0 symbols". MUTATION that must
    fail: drop the `files == 0` branch of `describe`."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_index(repo)

    result = _run(_hook_stdin(repo), {"CLAUDE_PROJECT_DIR": str(repo)}, home=tmp_path)

    assert result.returncode == 0
    assert result.stdout == (
        f"trelix: the index at {repo}/.trelix/index.db is empty (0 files). "
        f'Call index_codebase(repo_path="{repo}") before search_code; until then, use grep.\n'
    )


def _assert_skipped(result: subprocess.CompletedProcess[str], stderr: str, case: str) -> None:
    assert result.returncode == 0, f"{case}: exit {result.returncode}, stderr {result.stderr!r}"
    assert result.stdout == "", f"{case}: stdout {result.stdout!r}"
    assert result.stderr == stderr, f"{case}: stderr {result.stderr!r}"


def test_bad_input_exits_zero_with_no_context(tmp_path: Path) -> None:
    """Exit 0, empty stdout and one `trelix session_start: skipped (<reason>)` line on stderr for:
    stdin that is not JSON, or JSON but not an object (`[1]`), with no `CLAUDE_PROJECT_DIR`; a
    `cwd` that does not exist; a `CLAUDE_PROJECT_DIR` naming a file; an index that is not a
    database; a 0-byte index; a provenance query that fails for a reason other than a missing table.

    MUTATIONS that must fail: remove the top-level `except Exception`; let `project_root` raise on
    non-JSON stdin; catch only `JSONDecodeError` there (`[1]` becomes `skipped (AttributeError)`);
    tolerate every `OperationalError` in `read_index_facts` (the 0-byte index reads as indexed).
    """
    not_a_dir = tmp_path / "a-file"
    not_a_dir.write_text("", encoding="utf-8")

    result = _run("not json", {}, home=tmp_path)
    _assert_skipped(result, NO_PROJECT, "non-JSON stdin, no CLAUDE_PROJECT_DIR")

    result = _run("[1]", {}, home=tmp_path)
    _assert_skipped(result, NO_PROJECT, "JSON non-object stdin, no CLAUDE_PROJECT_DIR")

    result = _run('{"cwd": "/nonexistent/dir/for/trelix"}', {}, home=tmp_path)
    _assert_skipped(result, NO_PROJECT, "cwd does not exist")

    result = _run("{}", {"CLAUDE_PROJECT_DIR": str(not_a_dir)}, home=tmp_path)
    _assert_skipped(result, NO_PROJECT, "CLAUDE_PROJECT_DIR names a file")

    repo = tmp_path / "repo"
    (repo / ".trelix").mkdir(parents=True)
    index = repo / ".trelix" / "index.db"
    env = {"CLAUDE_PROJECT_DIR": str(repo)}

    index.write_bytes(b"not a database")
    result = _run(_hook_stdin(repo), env, home=tmp_path)
    _assert_skipped(result, "trelix session_start: skipped (DatabaseError)\n", "not a database")

    index.write_bytes(b"")
    result = _run(_hook_stdin(repo), env, home=tmp_path)
    _assert_skipped(result, "trelix session_start: skipped (OperationalError)\n", "0-byte index")
    assert index.read_bytes() == b"", "the read-only open must not write a header into the file"

    # Counts readable, but the provenance query fails for a reason other than a missing table
    # (here: no `value` column). Only "no such table: index_metadata" is tolerated; this one
    # propagates and the hook says nothing rather than guess.
    index.unlink()
    _bare_index(index, metadata_table="CREATE TABLE index_metadata (key TEXT PRIMARY KEY)")
    result = _run(_hook_stdin(repo), env, home=tmp_path)
    _assert_skipped(
        result, "trelix session_start: skipped (OperationalError)\n", "unreadable provenance"
    )


def _bare_index(index: Path, *, metadata_table: str | None) -> None:
    """A hand-made index with one `files` row, no symbols, and the given `index_metadata` DDL
    (None for an index written before trelix recorded provenance)."""
    conn = sqlite3.connect(index)
    with conn:
        conn.execute("CREATE TABLE files (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO files DEFAULT VALUES")
        conn.execute("CREATE TABLE symbols (id INTEGER PRIMARY KEY)")
        if metadata_table is not None:
            conn.execute(metadata_table)
    conn.close()


def test_index_without_metadata_table_reads_as_unknown_provenance(tmp_path: Path) -> None:
    """An index written before trelix recorded provenance has no `index_metadata` table at all:
    that one `OperationalError` is tolerated and the line says so; `files > 0, symbols == 0` is
    still "is indexed", with `0 symbols`. MUTATIONS that must fail: re-raise every
    `OperationalError` in the provenance query (stdout empty); report `symbols == 0` as "empty"."""
    repo = tmp_path / "repo"
    (repo / ".trelix").mkdir(parents=True)
    _bare_index(repo / ".trelix" / "index.db", metadata_table=None)

    result = _run(_hook_stdin(repo), {"CLAUDE_PROJECT_DIR": str(repo)}, home=tmp_path)

    assert result.returncode == 0
    assert result.stdout == (
        f"trelix: {repo} is indexed: 1 files, 0 symbols; built at an unknown time (re-index to "
        f'record provenance). Pass repo_path="{repo}" to every trelix tool.\n'
    )
    assert result.stderr == ""


def test_output_is_capped_at_one_thousand_characters() -> None:
    """A 1,400-character project path makes the line 2,938 characters; `describe` returns exactly
    1,000 ending in `…`. A 40-character path (218 characters) is untouched, as is `cap` on text at
    the limit. MUTATIONS that must fail: delete the `cap()` call in `describe`; cap at 1,001;
    truncate to `limit` characters before adding the ellipsis."""
    module = _load_script()
    facts = module.IndexFacts(files=1, symbols=1, git_commit=None, indexed_at=None, embedder=None)

    text = module.describe(Path("/r" * 700), facts, None)
    assert len(text) == 1000
    assert text.endswith("…")
    assert text.startswith("trelix: /r/r")

    short = "/" + "a" * 39
    assert module.describe(Path(short), facts, None) == (
        f"trelix: {short} is indexed: 1 files, 1 symbols; built at an unknown time (re-index to "
        f'record provenance). Pass repo_path="{short}" to every trelix tool.'
    )
    assert "…" not in module.describe(Path(short), facts, None)
    assert module.cap("a" * 1000) == "a" * 1000
    assert module.cap("a" * 1001) == "a" * 999 + "…"


@pytest.mark.parametrize(
    ("extra_commits", "commit", "expected_build"),
    [
        (0, FOREIGN_COMMIT, f"{INDEXED_AT} from commit 0123456789ab (distance from HEAD unknown)"),
        (0, "HEAD", f"{INDEXED_AT} from commit HEAD (distance from HEAD unknown)"),
        (0, "{sha}", f"{INDEXED_AT} from commit {{sha12}} (= HEAD)"),
        (1, "{sha}", f"{INDEXED_AT} from commit {{sha12}} (HEAD is 1 commit ahead)"),
        (0, None, "at an unknown time (re-index to record provenance)"),
    ],
    ids=["unknown-commit", "not-a-hash", "same-head", "one-ahead", "no-provenance"],
)
def test_distance_and_provenance_variants(
    tmp_path: Path, extra_commits: int, commit: str | None, expected_build: str
) -> None:
    """A commit git does not know, and a `HEAD` row (not a hash, so it never enters git's argv,
    where `HEAD..HEAD` would count 0): `distance from HEAD unknown`; the current HEAD: `= HEAD`;
    one newer commit: `HEAD is 1 commit ahead` (singular); no provenance rows: the "unknown time"
    wording and no `from commit` fragment. MUTATIONS that must fail: `_looks_like_commit` returning
    True (`(= HEAD)` for the `HEAD` row); map a `None` distance to 0 (`(= HEAD)` for the foreign
    commit); write `HEAD is 1 commits ahead`; print `from commit None` if provenance is absent."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, tmp_path, "init", "-q")
    sha = _commit(repo, tmp_path, "a.txt")
    for i in range(extra_commits):
        _commit(repo, tmp_path, f"more{i}.txt")
    provenance = None
    if commit is not None:
        provenance = IndexProvenance(git_commit=commit.format(sha=sha), indexed_at=INDEXED_AT)
    _make_index(repo, files=3, symbols=7, provenance=provenance)

    result = _run(_hook_stdin(repo), {"CLAUDE_PROJECT_DIR": str(repo)}, home=tmp_path)

    assert result.returncode == 0
    build = expected_build.format(sha12=sha[:12])
    assert result.stdout == (
        f"trelix: {repo} is indexed: 3 files, 7 symbols; built {build}. "
        f'Pass repo_path="{repo}" to every trelix tool.\n'
    )


def test_project_dir_env_wins_over_stdin_cwd(tmp_path: Path) -> None:
    """`CLAUDE_PROJECT_DIR` (which stays put) beats stdin's `cwd` (which follows Claude into
    worktrees) when both name directories; the line names the env var's repository only. MUTATION
    that must fail: read `cwd` before the env var (the `no index at .../repo-b/...` line)."""
    repo_a = tmp_path / "repo-a"
    repo_a.mkdir()
    _make_index(repo_a, files=3, symbols=7)
    repo_b = tmp_path / "repo-b"
    repo_b.mkdir()

    result = _run(_hook_stdin(repo_b), {"CLAUDE_PROJECT_DIR": str(repo_a)}, home=tmp_path)

    assert result.returncode == 0
    assert result.stdout == (
        f"trelix: {repo_a} is indexed: 3 files, 7 symbols; built at an unknown time (re-index to "
        f'record provenance). Pass repo_path="{repo_a}" to every trelix tool.\n'
    )
    assert str(repo_b) not in result.stdout


def test_unusable_project_dir_falls_through_to_stdin_cwd(tmp_path: Path) -> None:
    """A set but unusable `CLAUDE_PROJECT_DIR` is ignored and stdin's `cwd` is used. MUTATION that
    must fail: return None as soon as the env var is set."""
    repo = tmp_path / "repo"
    repo.mkdir()

    result = _run(
        _hook_stdin(repo), {"CLAUDE_PROJECT_DIR": "/nonexistent/dir/for/trelix"}, home=tmp_path
    )

    assert result.returncode == 0
    assert result.stdout == (
        f"trelix: no index at {repo}/.trelix/index.db. "
        f'Call index_codebase(repo_path="{repo}") before search_code; until then, use grep.\n'
    )


def test_read_only_uri_encodes_the_path_and_asks_for_mode_ro() -> None:
    """The index is opened through `file:<encoded path>?mode=ro`: a `#` or `?` in the path is
    percent-encoded so it cannot end the URI early and silently drop `mode=ro` (the defect
    `trelix.store.db.read_only_uri` documents); Python 3.14 prefixes the path with `//`, so only
    the tail is checked. MUTATIONS that must fail: drop `?mode=ro`; interpolate the raw path."""
    module = _load_script()
    uri = module.read_only_uri(Path("/tmp/a b#c?d/index.db"))  # noqa: S108 - a path shape, not a file
    assert uri.startswith("file:")
    assert uri.endswith("/tmp/a%20b%23c%3Fd/index.db?mode=ro")
    assert uri.count("?") == 1 and "#" not in uri
