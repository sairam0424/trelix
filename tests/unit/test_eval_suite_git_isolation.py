"""Git is isolated from the operator and from the repository: the environment of every git
child, hostile configuration that must not act, an option-shaped URL, and the canary files that
must never run.

Every hostile case has a CONTROL: the same hostile setting is first shown to act on a plain
`git` run, so the assertion that the code under test is immune cannot pass for lack of effect.

MUTATIONS THAT MUST MAKE THIS FILE FAIL
---------------------------------------
1. one of the seven variables left out of `git_env()` (one parametrized case each)
                                              (test_every_git_child_gets_the_isolating_variable)
2. one of the fourteen inherited names dropped (one parametrized case each), or any other
   variable inherited, a `GIT_*` one or `HOME` included
                                      (test_the_variable_is_passed_on_..., test_nothing_else_...)
3. `-c core.hooksPath=` left off a command, or `-C` dropped   (TestWhatEveryGitChildGets)
4. the `--` before the URL removed                             (test_the_url_follows_a_double_dash)
5. `--depth`, `--filter` or `--recurse-submodules` added to the clone (same test)
6. the timeouts changed from 1800 and 120 seconds              (test_timeouts_...)
7. `GIT_ALLOW_PROTOCOL` allowing `file` without `allow_local`, or on a command other than the
   clone, or `allow_local` not reaching the clone            (TestTheProtocolList)
8. the environment inherited whole instead of built            (TestHostileConfiguration)
9. any of the hostile cases (hooks, template, env config, system config, insteadOf, LFS,
   GIT_DIR) acting on the clone                                (TestHostileConfiguration)
"""

from __future__ import annotations

import dataclasses
import os
import runpy
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.unit.eval_suite_harness import (
    FIRST_FILES,
    SECOND_FILES,
    GitCall,
    Remote,
    clone_local,
    git,
    load_local,
    make_remote,
    record_git,
    write_suite,
)
from tests.unit.eval_suite_harness import remote as remote
from trelix.eval.suite import SuiteError, SuiteSpec
from trelix.eval.suite_git import check_clone, ensure_clone, git_env, tracked_files
from trelix.eval.suite_prepare import prepare_suite

_NULL = os.devnull


def _spec(tmp_path: Path, remote: Remote, url: str | None = None) -> SuiteSpec:
    return load_local(write_suite(tmp_path / "suite", url or str(remote.path), remote.first))


@pytest.fixture(scope="module")
def recorded(
    tmp_path_factory: pytest.TempPathFactory, remote: Remote
) -> tuple[list[GitCall], Path]:
    """Every git command of a fresh clone, a re-verification and a tree listing, run once."""
    root = tmp_path_factory.mktemp("eval-suite-calls")
    with pytest.MonkeyPatch.context() as patch:
        calls = record_git(patch)
        spec = _spec(root, remote)
        clone = clone_local(spec, root / "cache")
        clone_local(spec, root / "cache")
        tracked_files(clone, remote.first)
    return calls, root / "cache"


class TestWhatEveryGitChildGets:
    @pytest.fixture
    def calls(self, recorded: tuple[list[GitCall], Path]) -> list[GitCall]:
        return recorded[0]

    def test_the_commands_are_the_expected_ones_in_order(self, calls: list[GitCall]) -> None:
        assert [call.argv[5] for call in calls] == [
            "clone",
            "checkout",
            "rev-parse",
            "config",
            "status",
            "rev-parse",
            "config",
            "status",
            "ls-tree",
        ]

    def test_every_command_runs_in_a_directory_the_run_created(
        self, recorded: tuple[list[GitCall], Path]
    ) -> None:
        calls, cache = recorded
        for call in calls:
            assert call.argv[:2] == ["git", "-C"]
            assert call.argv[2].startswith(str(cache))

    def test_the_hooks_path_is_the_null_device_on_every_command(self, calls: list[GitCall]) -> None:
        for call in calls:
            assert call.argv[3:5] == ["-c", f"core.hooksPath={_NULL}"]

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("GIT_CONFIG_GLOBAL", _NULL),
            ("GIT_CONFIG_SYSTEM", _NULL),
            ("GIT_CONFIG_NOSYSTEM", "1"),
            ("GIT_TERMINAL_PROMPT", "0"),
            ("GIT_LFS_SKIP_SMUDGE", "1"),
            ("GIT_OPTIONAL_LOCKS", "0"),
        ],
    )
    def test_every_git_child_gets_the_isolating_variable(
        self, calls: list[GitCall], name: str, value: str
    ) -> None:
        assert len(calls) == 9
        for call in calls:
            assert call.env[name] == value

    def test_the_url_follows_a_double_dash_and_the_clone_is_full_and_flat(
        self, calls: list[GitCall], tmp_path: Path, remote: Remote
    ) -> None:
        clone_call = calls[0]
        assert clone_call.argv[5:] == [
            "clone",
            "--quiet",
            "--no-checkout",
            "--",
            str(remote.path),
            ".",
        ]

    def test_the_checkout_is_detached_at_the_pin_and_names_no_path(
        self, calls: list[GitCall], remote: Remote
    ) -> None:
        assert calls[1].argv[5:] == ["checkout", "--quiet", "--detach", remote.first, "--"]

    def test_timeouts_are_1800_seconds_for_the_clone_and_120_for_the_rest(
        self, calls: list[GitCall]
    ) -> None:
        assert [call.timeout for call in calls] == [1800, 120, 120, 120, 120, 120, 120, 120, 120]

    def test_no_git_child_can_read_the_terminal(self, calls: list[GitCall]) -> None:
        assert [call.stdin for call in calls] == [subprocess.DEVNULL] * 9


class TestTheProtocolList:
    """Only https in production; `file` only for the clone, and only when `allow_local` is set."""

    def test_the_default_is_https_only(self) -> None:
        assert git_env()["GIT_ALLOW_PROTOCOL"] == "https"

    def test_allow_local_adds_the_file_transport(self) -> None:
        assert git_env(allow_local=True)["GIT_ALLOW_PROTOCOL"] == "https:file"

    def test_only_the_clone_of_a_local_run_gets_the_file_transport(
        self, recorded: tuple[list[GitCall], Path]
    ) -> None:
        calls = recorded[0]
        assert calls[0].argv[5] == "clone"
        assert [call.env["GIT_ALLOW_PROTOCOL"] for call in calls] == ["https:file"] + ["https"] * 8

    def test_without_allow_local_git_refuses_a_local_repository(
        self, tmp_path: Path, remote: Remote
    ) -> None:
        """A spec that reached the clone with a local url still cannot use the `file` transport."""
        spec = _spec(tmp_path, remote)
        with pytest.raises(SuiteError) as caught:
            ensure_clone(spec, tmp_path / "cache")
        [problem] = caught.value.problems
        assert problem.startswith("git clone failed (exit 128): ")
        assert "transport 'file' not allowed" in problem
        assert not (tmp_path / "cache" / "clones" / remote.first / "demo.partial").exists()
        assert not (tmp_path / "cache" / "clones" / remote.first / "demo").exists()


# The fourteen variables a git child inherits, typed out: one dropped from the allowlist or one
# added to it has to change this file as well.
_PASSED_ON = (
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


class TestTheEnvironmentIsBuilt:
    @pytest.mark.parametrize("name", _PASSED_ON)
    def test_the_variable_is_passed_on_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        """The path to find git, the temporary directory, and the proxy and CA settings a network
        clone needs: the README promises these come through."""
        monkeypatch.setenv(name, f"canary-{name}")
        assert git_env()[name] == f"canary-{name}"

    def test_nothing_else_is_inherited(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        for name in _PASSED_ON:
            monkeypatch.setenv(name, "canary")
        for name, value in {
            "GIT_DIR": str(tmp_path),
            "GIT_WORK_TREE": str(tmp_path),
            "GIT_SSL_NO_VERIFY": "1",
            "GIT_SSH_COMMAND": "evil",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_VALUE_0": str(tmp_path),
            "GIT_EXEC_PATH": str(tmp_path),
            "GIT_ASKPASS": "evil",
            "HOME": str(tmp_path),
            "LD_PRELOAD": "evil.so",
            "PYTHONPATH": str(tmp_path),
            "SSH_AUTH_SOCK": str(tmp_path),
        }.items():
            monkeypatch.setenv(name, value)
        assert set(git_env()) == {
            *_PASSED_ON,
            "GIT_CONFIG_GLOBAL",
            "GIT_CONFIG_SYSTEM",
            "GIT_CONFIG_NOSYSTEM",
            "GIT_ALLOW_PROTOCOL",
            "GIT_TERMINAL_PROMPT",
            "GIT_LFS_SKIP_SMUDGE",
            "GIT_OPTIONAL_LOCKS",
        }


@dataclass(frozen=True)
class Hostile:
    """A hostile setting: the environment that carries it, the remote to clone, and a control.

    `control()` runs plain `git` under that environment and says whether the setting acted.
    `marker` is a file a planted hook writes if it runs.
    """

    env: dict[str, str]
    remote: Remote
    control: Callable[[], bool]
    marker: Path | None = None


def _planted_hook(directory: Path, marker: Path) -> Path:
    directory.mkdir(parents=True)
    hook = directory / "post-checkout"
    hook.write_text(f"#!/bin/sh\necho ran > '{marker}'\n", encoding="utf-8")
    hook.chmod(0o755)
    return directory


def _plain(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Plain git with the process environment, hostile settings included."""
    return subprocess.run(  # noqa: S603, S607
        ["git", *args], cwd=cwd, env=dict(os.environ), capture_output=True, text=True, check=False
    )


def _hook_fires(remote: Remote, marker: Path, scratch: Path) -> Callable[[], bool]:
    def control() -> bool:
        _plain(["clone", "-q", str(remote.path), str(scratch)], scratch.parent)
        return marker.exists()

    return control


def _config_file(tmp_path: Path, text: str) -> str:
    path = tmp_path / "hostile.gitconfig"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _hooks_path(tmp_path: Path, remote: Remote) -> Hostile:
    marker = tmp_path / "hooks-ran"
    hooks = _planted_hook(tmp_path / "hooks", marker)
    env = {"GIT_CONFIG_GLOBAL": _config_file(tmp_path, f"[core]\n\thooksPath = {hooks}\n")}
    return Hostile(env, remote, _hook_fires(remote, marker, tmp_path / "plain"), marker)


def _template_dir(tmp_path: Path, remote: Remote) -> Hostile:
    marker = tmp_path / "template-ran"
    _planted_hook(tmp_path / "template" / "hooks", marker)
    config = f"[init]\n\ttemplateDir = {tmp_path / 'template'}\n"
    env = {"GIT_CONFIG_GLOBAL": _config_file(tmp_path, config)}
    return Hostile(env, remote, _hook_fires(remote, marker, tmp_path / "plain"), marker)


def _env_config(tmp_path: Path, remote: Remote) -> Hostile:
    marker = tmp_path / "env-config-ran"
    hooks = _planted_hook(tmp_path / "hooks", marker)
    env = {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": str(hooks),
    }
    return Hostile(env, remote, _hook_fires(remote, marker, tmp_path / "plain"), marker)


def _system_config(tmp_path: Path, remote: Remote) -> Hostile:
    marker = tmp_path / "system-ran"
    hooks = _planted_hook(tmp_path / "hooks", marker)
    env = {"GIT_CONFIG_SYSTEM": _config_file(tmp_path, f"[core]\n\thooksPath = {hooks}\n")}
    return Hostile(env, remote, _hook_fires(remote, marker, tmp_path / "plain"), marker)


def _insteadof(tmp_path: Path, remote: Remote) -> Hostile:
    config = f'[url "file:///no/such/place/"]\n\tinsteadOf = {remote.path}\n'
    env = {"GIT_CONFIG_GLOBAL": _config_file(tmp_path, config)}

    def control() -> bool:
        return (
            _plain(["clone", "-q", str(remote.path), str(tmp_path / "plain")], tmp_path).returncode
            != 0
        )

    return Hostile(env, remote, control)


def _lfs(tmp_path: Path, remote: Remote) -> Hostile:
    lfs_remote = make_remote(
        tmp_path,
        {**FIRST_FILES, ".gitattributes": "*.bin filter=lfs\n", "assets/blob.bin": "pointer\n"},
        SECOND_FILES,
        name="lfs-remote",
    )
    config = '[filter "lfs"]\n\trequired = true\n\tsmudge = false\n\tclean = cat\n'
    env = {"GIT_CONFIG_GLOBAL": _config_file(tmp_path, config)}

    def control() -> bool:
        plain = _plain(["clone", "-q", str(lfs_remote.path), str(tmp_path / "plain")], tmp_path)
        return plain.returncode != 0

    return Hostile(env, lfs_remote, control)


def _git_dir(tmp_path: Path, remote: Remote) -> Hostile:
    other = make_remote(tmp_path, name="other-repository")
    env = {"GIT_DIR": str(other.path / ".git")}

    def control() -> bool:
        return _plain(["rev-parse", "HEAD"], tmp_path).stdout.strip() == other.second

    return Hostile(env, remote, control)


_HOSTILE: dict[str, Callable[[Path, Remote], Hostile]] = {
    "hooks path": _hooks_path,
    "template directory": _template_dir,
    "environment config": _env_config,
    "system config": _system_config,
    "url rewriting": _insteadof,
    "required LFS filter": _lfs,
    "GIT_DIR": _git_dir,
}


class TestHostileConfiguration:
    @pytest.mark.parametrize("case", list(_HOSTILE))
    def test_the_clone_and_its_verification_are_immune(
        self, tmp_path: Path, remote: Remote, monkeypatch: pytest.MonkeyPatch, case: str
    ) -> None:
        hostile = _HOSTILE[case](tmp_path, remote)
        for name, value in hostile.env.items():
            monkeypatch.setenv(name, value)

        assert hostile.control(), f"{case}: the hostile setting had no effect on plain git"
        if hostile.marker is not None:
            hostile.marker.unlink()

        spec = load_local(
            write_suite(
                tmp_path / "suite", str(hostile.remote.path), hostile.remote.first, name="demo"
            )
        )
        clone = clone_local(spec, tmp_path / "cache")
        assert check_clone(clone, spec) == []
        assert git(clone, "rev-parse", "HEAD") == hostile.remote.first
        assert not (clone / ".git" / "hooks" / "post-checkout").exists()
        if hostile.marker is not None:
            assert not hostile.marker.exists()


class TestAnOptionShapedUrl:
    def test_an_option_as_the_url_is_a_repository_name_and_runs_nothing(
        self, tmp_path: Path, remote: Remote, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        marker = tmp_path / "upload-pack-ran"
        option = f"--upload-pack=touch {marker}"
        # The control: written as an option, ahead of the URL, git runs the command.
        _plain(
            ["clone", "-q", option, f"file://{remote.path}", str(tmp_path / "control")], tmp_path
        )
        assert marker.exists(), "control: git did not treat the text as an option"
        marker.unlink()

        spec = dataclasses.replace(_spec(tmp_path, remote), repo_url=option)
        calls = record_git(monkeypatch)
        with pytest.raises(SuiteError) as caught:
            clone_local(spec, tmp_path / "cache")
        assert not marker.exists()
        assert "git clone failed" in caught.value.problems[0]
        assert calls[0].argv[-3:] == ["--", option, "."]


class TestNothingFromTheRepositoryRuns:
    def test_the_planted_files_never_run_during_a_full_prepare(
        self, tmp_path: Path, remote: Remote
    ) -> None:
        """`conftest.py`, `setup.py` and `sitecustomize.py` each write `EXECUTED` beside them."""
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        planted = scratch / "setup.py"
        planted.write_text((FIRST_FILES["setup.py"]), encoding="utf-8")
        runpy.run_path(str(planted))
        assert (scratch / "EXECUTED").exists(), "control: the canary did not fire when run"
        (scratch / "EXECUTED").unlink()

        suite_json = write_suite(tmp_path / "suite", str(remote.path), remote.first)
        prepared = prepare_suite(suite_json, tmp_path / "cache", allow_local=True)

        assert (prepared.clone / "setup.py").read_text() == FIRST_FILES["setup.py"]
        assert not (prepared.clone / "EXECUTED").exists()
        assert check_clone(prepared.clone, prepared.spec) == []
