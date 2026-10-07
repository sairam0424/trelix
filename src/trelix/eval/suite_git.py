"""The git half of `trelix eval-suite`: clone a pinned commit, verify it, never repair it.

The repository a suite names is untrusted input: it is read, hashed and parsed, and nothing in
it is ever executed, imported, built or installed. For git itself that takes four things.

* The environment is built, not inherited. Every git child gets only the variables in
  `_INHERITED_ENV` (a path to find git, a proxy and CA bundle for the network) plus a fixed set
  that points git's global and system configuration at the null device and allows only the
  protocols a suite may use. An ambient `GIT_DIR`, `GIT_CONFIG_*`, `GIT_SSL_NO_VERIFY`,
  `core.hooksPath`, `url.*.insteadOf`, `init.templateDir`, `filter.lfs.*` or
  `submodule.recurse` therefore cannot shape a clone, and `core.hooksPath` is also given on the
  command line. The cost: no credentials, so only public repositories.
* Every command is `git -C <directory> ...` on a directory this command created, and the URL
  follows `--`. No submodule is fetched and no file is smudged by LFS.
* The clone is made in `<name>.partial`, checked out at the pin, verified, and only then moved
  into place with one rename. A `.partial` left by a killed run is refused, never deleted.
* A clone that already exists is re-verified and otherwise left alone: HEAD is the pin, the
  worktree is pristine (ignored files included, because the walker would index them), the
  origin URL is the suite's, and the clone is neither shallow nor partial (git could complete
  those from the network while the index is built). It is never repaired and never deleted.

Layout: `<cache>/clones/<repo sha>/<suite name>/`. A suite name has no `.`, so `<name>.partial`
can never be another suite's directory.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from trelix.eval._loading import clip
from trelix.eval.suite import SuiteError, SuiteSpec

logger = logging.getLogger("trelix.eval.suite")

CLONE_TIMEOUT_SECONDS = 1800
GIT_TIMEOUT_SECONDS = 120

_STATUS_LISTED = 5
_GIT_MESSAGE_CHARS = 200

# The only variables a git child inherits. Nothing named GIT_* comes through: that would let
# the operator's shell choose the repository, the configuration or the certificate checks.
_INHERITED_ENV = (
    "PATH",
    "SYSTEMROOT",
    "TMPDIR",
    "TMP",
    "TEMP",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "CURL_CA_BUNDLE",
)

# What every git child gets, and what the whole process gets while a suite is indexed and its
# queries run (`isolated_git`): the Indexer's own `git rev-parse` and `git status` in
# `trelix.store.provenance` inherit the process environment, so the operator's configuration
# must be switched off there too. `GIT_ALLOW_PROTOCOL` is `https` here; only the clone of a
# local test repository adds `file` (`git_env(allow_local=True)`).
ISOLATION_ENV: dict[str, str] = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_ALLOW_PROTOCOL": "https",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_LFS_SKIP_SMUDGE": "1",
    "GIT_OPTIONAL_LOCKS": "0",
}
# What `isolated_git` removes from the process for the block: each of these makes git answer
# for a repository other than the directory it runs in (a hook or a shell with `GIT_DIR`
# exported), and `trelix.store.provenance` runs `git` with `cwd=` and no `-C`, so an index's
# provenance rows would otherwise name another repository's commit. A git child of this module
# never sees them: `git_env` builds its environment from `_INHERITED_ENV` alone.
REPOSITORY_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
)


def git_env(*, allow_local: bool = False) -> dict[str, str]:
    """The complete environment of one git child; `allow_local` adds git's `file` transport."""
    env = {name: os.environ[name] for name in _INHERITED_ENV if name in os.environ}
    env.update(ISOLATION_ENV)
    if allow_local:
        # `file` covers a local path and a file:// URL; only the tests may use either.
        env["GIT_ALLOW_PROTOCOL"] = "https:file"
    return env


@contextmanager
def isolated_git() -> Iterator[None]:
    """Set `ISOLATION_ENV` and remove `REPOSITORY_ENV` in this process for the block, and put
    back what was there before.

    For the index build and the query run of a suite: every git child trelix itself spawns in
    the clone then runs with the operator's global and system configuration switched off, like
    the clone did, and answers for the clone and not for an exported `GIT_DIR`. A variable that
    was absent is removed again; one that was set gets its old value, also when the block raises.
    """
    previous = {name: os.environ.get(name) for name in (*ISOLATION_ENV, *REPOSITORY_ENV)}
    for name in REPOSITORY_ENV:
        os.environ.pop(name, None)
    os.environ.update(ISOLATION_ENV)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_git(
    directory: Path, *args: str, timeout: int = GIT_TIMEOUT_SECONDS, allow_local: bool = False
) -> str:
    """The stdout of `git -C directory *args` under `git_env()`; SuiteError when it fails.

    A missing `git`, a timeout, any other OSError and a non-zero exit are each one SuiteError.
    """
    argv = ["git", "-C", str(directory), "-c", f"core.hooksPath={os.devnull}", *args]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            env=git_env(allow_local=allow_local),
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SuiteError(
            ["git was not found on PATH: eval-suite needs it to clone the suite's repository"]
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SuiteError([f"git {args[0]} timed out after {timeout} seconds"]) from exc
    except OSError as exc:
        raise SuiteError([f"cannot run git: {clip(str(exc), _GIT_MESSAGE_CHARS)}"]) from exc
    if result.returncode != 0:
        reason = next(iter(result.stderr.strip().splitlines()), "no error output")
        raise SuiteError(
            [f"git {args[0]} failed (exit {result.returncode}): {clip(reason, _GIT_MESSAGE_CHARS)}"]
        )
    return result.stdout


def default_cache_root() -> Path:
    """`$XDG_CACHE_HOME/trelix/eval-suites`, or `~/.cache/trelix/eval-suites` without it.

    A relative `XDG_CACHE_HOME` is invalid by the XDG rules and is ignored.
    """
    configured = os.environ.get("XDG_CACHE_HOME", "")
    if configured and os.path.isabs(configured):
        return Path(configured) / "trelix" / "eval-suites"
    try:
        return Path.home() / ".cache" / "trelix" / "eval-suites"
    except RuntimeError as exc:
        raise SuiteError(
            ["cannot find a home directory for the suite cache: pass --cache-dir"]
        ) from exc


def clone_path(cache_root: Path, spec: SuiteSpec) -> Path:
    """Where the clone of `spec` lives under `cache_root`."""
    return cache_root / "clones" / spec.repo_sha / spec.name


def tracked_files(clone: Path, sha: str) -> frozenset[str]:
    """The paths of every file of commit `sha`, relative to the repository root."""
    listing = run_git(clone, "ls-tree", "-r", "--name-only", "-z", sha)
    return frozenset(name for name in listing.split("\0") if name)


def check_clone(path: Path, spec: SuiteSpec) -> list[str]:
    """Why the clone at `path` is not the pinned commit of `spec`, untouched; `[]` when it is."""
    where = f"clone {path}"
    git_dir = path / ".git"
    # Without a `.git` of its own git would climb to an enclosing repository and answer for that.
    if path.is_symlink() or git_dir.is_symlink() or not git_dir.is_dir():
        return [f"{where} is not a git repository of its own (it has no real .git directory)"]
    try:
        problems = _identity_problems(path, spec) + _worktree_problems(path)
    except SuiteError as exc:
        problems = list(exc.problems)
    return [f"{where}: {problem}" for problem in problems]


def _identity_problems(path: Path, spec: SuiteSpec) -> list[str]:
    """HEAD, the origin URL, and whether the clone is complete."""
    problems: list[str] = []
    head, _, shallow = run_git(path, "rev-parse", "HEAD", "--is-shallow-repository").partition("\n")
    if head != spec.repo_sha:
        problems.append(f"HEAD is {head}, not the pinned {spec.repo_sha}")
    if shallow.strip() != "false":
        problems.append("it is a shallow clone, which git may complete from the network")
    entries = [
        line.partition("=") for line in run_git(path, "config", "--local", "--list").splitlines()
    ]
    urls = [value for key, _, value in entries if key == "remote.origin.url"]
    if urls != [spec.repo_url]:
        problems.append(f"the origin URL is {clip(repr(urls))}, not {clip(repr(spec.repo_url))}")
    if any(_is_partial_clone_key(key) for key, _, _ in entries):
        problems.append("it is a partial clone, which git may complete from the network")
    return problems


def _is_partial_clone_key(key: str) -> bool:
    if key == "extensions.partialclone":
        return True
    return key.startswith("remote.") and key.endswith((".promisor", ".partialclonefilter"))


def _worktree_problems(path: Path) -> list[str]:
    """Any modified, untracked or ignored file: the walker would index it."""
    entries = run_git(
        path, "status", "--porcelain", "--untracked-files=all", "--ignored"
    ).splitlines()
    if not entries:
        return []
    listed = "; ".join(clip(entry) for entry in entries[:_STATUS_LISTED])
    more = len(entries) - _STATUS_LISTED
    tail = f"; and {more} more" if more > 0 else ""
    return [f"the worktree is not pristine ({len(entries)} entries: {listed}{tail})"]


def ensure_clone(spec: SuiteSpec, cache_root: Path, *, allow_local: bool = False) -> Path:
    """The verified clone of `spec` under `cache_root`, made when it does not exist yet.

    An existing clone is re-verified and returned, or refused with every reason; it is never
    repaired or deleted. A new one is cloned into `<name>.partial`, checked out at the pin and
    verified before the rename that puts it in place, and removed again if any step fails or the
    run is interrupted (Ctrl-C).

    The host of `spec.repo_url` was checked by `load_suite`; here git itself is held to https,
    and `allow_local` (the tests only, never the command line) adds its `file` transport.
    """
    final = clone_path(cache_root, spec)
    partial = final.with_name(f"{spec.name}.partial")
    if final.exists() or final.is_symlink():
        problems = check_clone(final, spec)
        if problems:
            raise SuiteError(
                [
                    *problems,
                    f"eval-suite never repairs or deletes a clone: remove {final} and run again",
                ]
            )
        # A `.partial` beside a verified clone is a killed or concurrent run as well: the same
        # refusal as when no clone exists yet, so the operator sees it on every run until it is
        # removed, not only on the first.
        if partial.exists() or partial.is_symlink():
            raise SuiteError([_in_the_way(partial)])
        return final
    _claim(partial)
    try:
        _clone_into(partial, spec, allow_local)
        os.replace(partial, final)
    except (SuiteError, KeyboardInterrupt):
        _discard(partial)
        raise
    except OSError as exc:
        _discard(partial)
        raise SuiteError([f"cannot move the verified clone into place at {final}: {exc}"]) from exc
    return final


def _claim(partial: Path) -> None:
    """Create the empty `.partial` directory; refuse one that is already there.

    Only the `.partial` itself is blamed on a killed or concurrent run. A parent that cannot be
    made (a file where `<cache>/clones/<sha>` should be) is reported as what it is.
    """
    try:
        partial.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SuiteError([f"cannot create {partial}: {exc}"]) from exc
    try:
        partial.mkdir()
    except FileExistsError as exc:
        raise SuiteError([_in_the_way(partial)]) from exc
    except OSError as exc:
        raise SuiteError([f"cannot create {partial}: {exc}"]) from exc


def _in_the_way(partial: Path) -> str:
    return (
        f"{partial} is in the way: a run that was killed left it, or another eval-suite "
        "is cloning now. eval-suite never deletes it; remove it and run again"
    )


def _clone_into(partial: Path, spec: SuiteSpec, allow_local: bool) -> None:
    """A full clone (no depth, no filter), detached at the pin, and verified."""
    run_git(
        partial,
        "clone",
        "--quiet",
        "--no-checkout",
        "--",
        spec.repo_url,
        ".",
        timeout=CLONE_TIMEOUT_SECONDS,
        allow_local=allow_local,
    )
    run_git(partial, "checkout", "--quiet", "--detach", spec.repo_sha, "--")
    problems = check_clone(partial, spec)
    if problems:
        raise SuiteError(problems)


def _discard(partial: Path) -> None:
    """Remove the `.partial` directory this run created and failed to finish."""
    try:
        shutil.rmtree(partial)
    except OSError as exc:
        logger.warning("could not remove the failed partial clone %s: %s", partial, exc)
