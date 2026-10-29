"""``POST /parse`` answers 400, not 500, for a ``file_path`` the filesystem cannot resolve or look
up. (``GET /graph/visualize`` is in test_api_unresolvable_output.py.)

A repository is untrusted content and git tracks symlinks, so a caller can name ``a`` where
``a -> b`` and ``b -> a``; a name longer than NAME_MAX is just as easy to send. What ``pathlib``
does with them depends on the interpreter:

* ``Path.resolve()`` raises RuntimeError for a symlink loop on Python 3.12 and stopped doing so in
  3.13, where the loop surfaces on the next stat or write instead;
* ``Path.exists()`` and ``is_file()`` raise OSError for a too-long name or a directory that may
  not be entered on 3.12 and 3.13, and answer False on 3.14;
* ``Path.read_text()`` raises for a file that may not be read on every version.

Each of those used to end in an unhandled exception and an HTTP 500. The contract asserted here
holds on every supported version: status 400, one of the route's own fixed messages, and no
exception text.

Two more ends of the same class. A JSON body can carry a lone surrogate (``"\\udcff"``, the byte
0xff of a file name that is not UTF-8): the 400 for a missing file echoes the path, Starlette
cannot encode it, and that answer was itself a 500. The echo now writes the surrogate out as an
escape, and a file that does exist under such a name (a Linux filesystem takes any bytes) is still
parsed. And only a refused permission or a name over NAME_MAX is the caller's mistake when the file
is looked up or read: a failing device or a stale handle is the server's, and stays a 500. Every
expected value is a literal.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from trelix.api.app import create_app

# Root, and Windows, ignore permission bits: the denial the tests below need cannot be made there.
_CANNOT_DENY_ACCESS = sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0

_NOT_A_VALID_PATH = "file_path is not a valid path"
_EXCEPTION_TEXT = ("Errno", "Symlink loop", "Too many levels", "File name too long", "Permission")

# Where the interpreter itself starts to report a name it cannot look up as missing, so that the
# route's own "file_path not found: " is just as good a 400. Before that version resolve() or
# exists() raises, and only the fixed answer above is right. Measured on 3.12.12, 3.13.12 and
# 3.14.7: a symlink loop is reported missing from 3.13 (ELOOP is one of the errors exists()
# ignores); a name over NAME_MAX and a directory that may not be entered from 3.14.
_LOOP_REPORTED_MISSING_FROM = (3, 13)
_UNEXAMINABLE_REPORTED_MISSING_FROM = (3, 14)


def _assert_400(resp: Any, reported_missing_from: tuple[int, int]) -> None:  # noqa: ANN401
    answers = (_NOT_A_VALID_PATH,)
    if sys.version_info[:2] >= reported_missing_from:
        answers += ("file_path not found: ",)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"].startswith(answers), resp.text
    for fragment in _EXCEPTION_TEXT:
        assert fragment not in resp.text, resp.text


def _answering(method: str, name: str, answer: object) -> AbstractContextManager[object]:
    """Patch ``Path.<method>`` so the one path called ``name`` gets ``answer`` (raised when it is
    an exception) and every other path gets the real lookup, which leaves the rest of the request
    (the repository checks, the store) alone. ``exists`` is patched rather than ``stat``: from
    3.14 it no longer calls ``stat``."""
    real = getattr(Path, method)

    def lookup(self: Path, *args: object, **kwargs: object) -> object:
        if self.name != name:
            return real(self, *args, **kwargs)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return patch.object(Path, method, lookup)


# ---------------------------------------------------------------------------
# POST /parse
# ---------------------------------------------------------------------------


class TestParseFilePathTheFilesystemRejects:
    @pytest.fixture
    def world(self, tmp_path: Path) -> tuple[Path, TestClient]:
        root = tmp_path / "root"
        root.mkdir()
        (root / "a").symlink_to("b")
        (root / "b").symlink_to("a")
        return root, TestClient(create_app(served_root=root), raise_server_exceptions=False)

    @pytest.mark.parametrize("shape", ["relative", "absolute"])
    @pytest.mark.parametrize(
        ("name", "missing_from"),
        [
            pytest.param("a", _LOOP_REPORTED_MISSING_FROM, id="symlink-loop"),
            pytest.param("a/c.py", _LOOP_REPORTED_MISSING_FROM, id="below-a-symlink-loop"),
            pytest.param(
                "x" * 300 + ".py", _UNEXAMINABLE_REPORTED_MISSING_FROM, id="name-over-name-max"
            ),
            pytest.param(
                "x" * 300 + "/c.py",
                _UNEXAMINABLE_REPORTED_MISSING_FROM,
                id="directory-over-name-max",
            ),
        ],
    )
    def test_is_400_not_500(
        self, world: tuple[Path, TestClient], name: str, missing_from: tuple[int, int], shape: str
    ) -> None:
        root, client = world
        file_path = name if shape == "relative" else str(root / name)

        resp = client.post("/parse", json={"repo_path": str(root), "file_path": file_path})

        _assert_400(resp, missing_from)

    def test_a_good_file_next_to_the_loop_still_parses(
        self, world: tuple[Path, TestClient]
    ) -> None:
        root, client = world
        (root / "ok.py").write_text("def fine():\n    return 1\n", encoding="utf-8")

        resp = client.post("/parse", json={"repo_path": str(root), "file_path": "ok.py"})

        assert resp.status_code == 200, resp.text
        assert [s["name"] for s in resp.json()["symbols"]] == ["fine"]


@pytest.mark.skipif(_CANNOT_DENY_ACCESS, reason="root and Windows ignore permission bits")
class TestParseFilePathTheServerMayNotRead:
    @pytest.fixture
    def world(self, tmp_path: Path) -> tuple[Path, TestClient]:
        root = tmp_path / "root"
        root.mkdir()
        return root, TestClient(create_app(served_root=root), raise_server_exceptions=False)

    def test_a_directory_it_may_not_enter_is_400_not_500(
        self, world: tuple[Path, TestClient]
    ) -> None:
        root, client = world
        locked = root / "locked"
        locked.mkdir()
        (locked / "f.py").write_text("x = 1\n", encoding="utf-8")
        locked.chmod(0o000)
        try:
            resp = client.post("/parse", json={"repo_path": str(root), "file_path": "locked/f.py"})
        finally:
            locked.chmod(0o700)  # tmp_path cleanup must be able to remove it

        _assert_400(resp, _UNEXAMINABLE_REPORTED_MISSING_FROM)

    def test_a_file_it_may_not_read_is_400_not_500_on_every_version(
        self, world: tuple[Path, TestClient], caplog: pytest.LogCaptureFixture
    ) -> None:
        """The file exists and is a file, so only the read can fail, on every interpreter.
        The response omits the cause; the server log carries it."""
        root, client = world
        secret = root / "secret.py"
        secret.write_text("x = 1\n", encoding="utf-8")
        secret.chmod(0o000)
        caplog.set_level(logging.WARNING, logger="trelix.api")
        try:
            resp = client.post("/parse", json={"repo_path": str(root), "file_path": "secret.py"})
        finally:
            secret.chmod(0o600)

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": _NOT_A_VALID_PATH}
        logged = [r.getMessage() for r in caplog.records if r.name == "trelix.api"]
        assert len(logged) == 1
        assert logged[0].startswith("Refused request: file_path='secret.py' cannot be used: ")
        assert "[Errno 13]" in logged[0]


_ESCAPED_NAME = "\\udcff.py"  # the JSON text of a name holding a lone surrogate
_SURROGATE_NAME = "\udcff.py"  # what the server sees


def _post_escaped(client: TestClient, root: Path, escaped_name: str, shape: str) -> httpx.Response:
    """``POST /parse`` with ``file_path`` set to the JSON text ``escaped_name``, relative or
    absolute. The ``json=`` helper cannot encode a lone surrogate: the body is written by hand."""
    file_path = escaped_name
    if shape == "absolute":
        file_path = f"{json.dumps(str(root))[1:-1]}/{escaped_name}"
    body = f'{{"repo_path": {json.dumps(str(root))}, "file_path": "{file_path}"}}'
    return client.post(
        "/parse", content=body.encode("utf-8"), headers={"content-type": "application/json"}
    )


class TestParseFilePathWithALoneSurrogate:
    """``"\\udcff"`` is what the byte 0xff of a file name that is not UTF-8 decodes to
    (surrogateescape), and a Linux repository can hold such a file. The 400s that name the path
    echo it, which Starlette cannot encode, so they used to end in a 500 on every version.
    ``"\\ud800"`` is no byte of a file name, so ``resolve()`` rejects it."""

    @pytest.fixture
    def world(self, tmp_path: Path) -> tuple[Path, TestClient]:
        root = tmp_path / "root"
        root.mkdir()
        return root, TestClient(create_app(served_root=root), raise_server_exceptions=False)

    def test_a_surrogate_that_is_no_file_name_byte_is_400_and_logged(
        self, world: tuple[Path, TestClient], caplog: pytest.LogCaptureFixture
    ) -> None:
        root, client = world
        caplog.set_level(logging.WARNING, logger="trelix.api")

        resp = _post_escaped(client, root, "\\ud800.py", "relative")

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": _NOT_A_VALID_PATH}
        logged = [r.getMessage() for r in caplog.records if r.name == "trelix.api"]
        assert len(logged) == 1
        assert logged[0].startswith("Refused request: file_path='\\ud800.py' cannot be used: ")

    @pytest.mark.parametrize("shape", ["relative", "absolute"])
    def test_a_missing_file_is_400_with_the_name_escaped(
        self, world: tuple[Path, TestClient], shape: str
    ) -> None:
        root, client = world

        resp = _post_escaped(client, root, _ESCAPED_NAME, shape)

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": f"file_path not found: {root.resolve()}{os.sep}\\udcff.py"}

    def test_a_directory_is_400_with_the_name_escaped(self, world: tuple[Path, TestClient]) -> None:
        """macOS cannot create a file named like this (EILSEQ), so the lookups answer as a Linux
        filesystem would."""
        root, client = world

        with (
            _answering("exists", _SURROGATE_NAME, True),
            _answering("is_file", _SURROGATE_NAME, False),
        ):
            resp = _post_escaped(client, root, _ESCAPED_NAME, "relative")

        assert resp.status_code == 400, resp.text
        assert resp.json() == {
            "detail": f"file_path is not a file: {root.resolve()}{os.sep}\\udcff.py"
        }

    def test_a_file_that_exists_under_such_a_name_still_parses(
        self, world: tuple[Path, TestClient]
    ) -> None:
        """Nothing is echoed on success, so refusing the name would only break a working call.
        The lookups answer as a Linux filesystem would, as above."""
        root, client = world

        with (
            _answering("exists", _SURROGATE_NAME, True),
            _answering("is_file", _SURROGATE_NAME, True),
            _answering("read_text", _SURROGATE_NAME, "def fine():\n    return 1\n"),
        ):
            resp = _post_escaped(client, root, _ESCAPED_NAME, "relative")

        assert resp.status_code == 200, resp.text
        assert [s["name"] for s in resp.json()["symbols"]] == ["fine"]

    def test_a_name_that_is_not_ascii_is_echoed_as_written(
        self, world: tuple[Path, TestClient]
    ) -> None:
        """Only text UTF-8 cannot encode is escaped, not every name outside ASCII."""
        root, client = world

        resp = client.post("/parse", json={"repo_path": str(root), "file_path": "naïve.py"})

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": f"file_path not found: {root.resolve()}{os.sep}naïve.py"}


class TestParseLookupAndReadFailure:
    """Only what is about the name is the caller's mistake: a refused permission, and for the
    lookup a name over NAME_MAX. A failing device or a stale network handle is the server's, and
    must not be blamed on the path. The calls are patched, so none of this depends on the
    interpreter (3.14 answers False for any failed lookup) or on running as a user whose
    permission bits count."""

    @pytest.fixture
    def world(self, tmp_path: Path) -> tuple[Path, TestClient]:
        root = tmp_path / "root"
        root.mkdir()
        (root / "ok.py").write_text("def fine():\n    return 1\n", encoding="utf-8")
        return root, TestClient(create_app(served_root=root), raise_server_exceptions=False)

    @pytest.mark.parametrize(
        ("method", "code"),
        [
            pytest.param("exists", errno.ENAMETOOLONG, id="exists-name-too-long"),
            pytest.param("is_file", errno.EACCES, id="is_file-permission-denied"),
        ],
    )
    def test_a_refused_lookup_is_400_and_logged(
        self,
        world: tuple[Path, TestClient],
        caplog: pytest.LogCaptureFixture,
        method: str,
        code: int,
    ) -> None:
        root, client = world
        caplog.set_level(logging.WARNING, logger="trelix.api")

        with _answering(method, "ok.py", OSError(code, "injected")):
            resp = client.post("/parse", json={"repo_path": str(root), "file_path": "ok.py"})

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": _NOT_A_VALID_PATH}
        logged = [r.getMessage() for r in caplog.records if r.name == "trelix.api"]
        assert len(logged) == 1
        assert logged[0].startswith("Refused request: file_path='ok.py' cannot be used: ")

    @pytest.mark.parametrize(
        ("method", "code"),
        [
            pytest.param("exists", errno.EIO, id="exists-failing-device"),
            pytest.param("is_file", errno.ESTALE, id="is_file-stale-handle"),
        ],
    )
    def test_any_other_lookup_failure_stays_a_500(
        self, world: tuple[Path, TestClient], method: str, code: int
    ) -> None:
        root, client = world

        with _answering(method, "ok.py", OSError(code, "injected")):
            resp = client.post("/parse", json={"repo_path": str(root), "file_path": "ok.py"})

        assert resp.status_code == 500, resp.text

    @pytest.mark.parametrize(
        "code",
        [
            pytest.param(errno.EACCES, id="permission-denied"),
            pytest.param(errno.EPERM, id="operation-not-permitted"),
        ],
    )
    def test_a_refused_read_is_400(self, world: tuple[Path, TestClient], code: int) -> None:
        root, client = world

        with patch.object(Path, "read_text", side_effect=OSError(code, "injected")):
            resp = client.post("/parse", json={"repo_path": str(root), "file_path": "ok.py"})

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": _NOT_A_VALID_PATH}

    @pytest.mark.parametrize(
        "code",
        [
            pytest.param(errno.EIO, id="failing-device"),
            pytest.param(errno.EMFILE, id="no-descriptors-left"),
        ],
    )
    def test_any_other_read_failure_stays_a_500(
        self, world: tuple[Path, TestClient], code: int
    ) -> None:
        root, client = world

        with patch.object(Path, "read_text", side_effect=OSError(code, "injected")):
            resp = client.post("/parse", json={"repo_path": str(root), "file_path": "ok.py"})

        assert resp.status_code == 500, resp.text
