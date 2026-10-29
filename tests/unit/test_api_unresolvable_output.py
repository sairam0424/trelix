"""``GET /graph/visualize`` answers 400, not 500, for an ``output`` the filesystem cannot use.

A repository is untrusted content and git tracks symlinks, so the name a caller gives can be a
symlink loop, longer than NAME_MAX, below a file, or the name of a directory.
``Path.resolve()`` raises RuntimeError for a loop on Python 3.12 and stopped doing so in 3.13,
where the loop surfaces at the write instead, and ``export_html`` creates the directory and the
file, so the write can fail on the name too. Each used to end in an unhandled exception and an
HTTP 500. The contract asserted here holds on every supported version: status 400, one of the
route's own fixed messages, and no exception text. A failure that is not about the name (a full
disk, a failing device, the default file) stays a 500. Every expected value is a literal.

The ``POST /parse`` half of the same fix is in test_api_unresolvable_paths.py.
"""

from __future__ import annotations

import errno
import logging
import sys
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.api_confinement import pyvis_without_asserts
from tests.fixtures.indexed import mark_indexed
from trelix.api.app import create_app


class _VizWorld(NamedTuple):
    repo: Path
    store: Path  # <repo>/.trelix, the only place `output` may name
    client: TestClient


@pytest.fixture
def viz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _VizWorld:
    pyvis_without_asserts(monkeypatch)
    root = tmp_path / "root"
    repo = mark_indexed(root / "repo")
    client = TestClient(create_app(served_root=root), raise_server_exceptions=False)
    return _VizWorld(repo, repo / ".trelix", client)


class TestVisualizeOutputTheFilesystemRejects:
    @pytest.mark.parametrize("shape", ["relative", "absolute"])
    def test_a_symlink_loop_is_400_not_500(
        self, viz: _VizWorld, monkeypatch: pytest.MonkeyPatch, shape: str
    ) -> None:
        """3.12 stops at ``resolve()``; 3.13 and later get as far as the write."""
        sub = viz.store / "sub"
        sub.mkdir()
        (sub / "a.html").symlink_to("b.html")
        (sub / "b.html").symlink_to("a.html")
        monkeypatch.chdir(sub)
        output = "a.html" if shape == "relative" else str(sub / "a.html")

        resp = viz.client.get("/graph/visualize", params={"repo": str(viz.repo), "output": output})

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": "output is not a valid path"}

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param("x" * 300 + ".html", id="file-name-longer-than-name-max"),
            pytest.param("x" * 300 + "/g.html", id="directory-longer-than-name-max"),
            pytest.param("index.db/g.html", id="parent-is-a-file"),
            pytest.param("index.db/sub/g.html", id="below-a-file"),
            pytest.param("dir.html", id="names-a-directory"),
        ],
    )
    def test_a_name_that_cannot_be_written_is_400_not_500(self, viz: _VizWorld, name: str) -> None:
        (viz.store / "dir.html").mkdir()
        index_before = (viz.store / "index.db").read_bytes()

        resp = viz.client.get(
            "/graph/visualize",
            params={"repo": str(viz.repo), "output": str(viz.store / name)},
        )

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": "output is not a valid path"}
        assert (viz.store / "index.db").read_bytes() == index_before

    @pytest.mark.skipif(sys.platform == "win32", reason="a Windows file name cannot hold a newline")
    def test_the_log_carries_the_cause_and_the_quoted_value(
        self, viz: _VizWorld, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The value is the caller's, so it is logged quoted: a newline in it cannot forge a
        second record."""
        target = viz.store / "d\nir.html"
        target.mkdir()
        caplog.set_level(logging.WARNING, logger="trelix.api")

        resp = viz.client.get(
            "/graph/visualize", params={"repo": str(viz.repo), "output": str(target)}
        )

        assert resp.status_code == 400, resp.text
        assert "[Errno" not in resp.text
        logged = [r.getMessage() for r in caplog.records if r.name == "trelix.api"]
        assert len(logged) == 1
        assert len(logged[0].splitlines()) == 1
        assert logged[0].startswith("Refused request: output='")
        assert r"d\nir.html' cannot be used: [Errno 21]" in logged[0]

    @pytest.mark.parametrize(
        ("named", "failure"),
        [
            pytest.param(True, OSError(errno.ENOSPC, "no space"), id="disk-full"),
            pytest.param(True, OSError(errno.EIO, "device error"), id="failing-device"),
            pytest.param(True, OSError(errno.EACCES, "denied"), id="store-not-writable"),
            pytest.param(False, OSError(errno.ENAMETOOLONG, "too long"), id="not-the-callers-name"),
        ],
    )
    def test_a_failure_that_is_not_the_callers_name_stays_a_500(
        self, viz: _VizWorld, named: bool, failure: OSError
    ) -> None:
        """A full disk is the server's problem, and so is any failure of the default file."""
        params = {"repo": str(viz.repo)}
        if named:
            params["output"] = str(viz.store / "custom.html")

        with patch("trelix.graph.visualizer.GraphVisualizer.export_html", side_effect=failure):
            resp = viz.client.get("/graph/visualize", params=params)

        assert resp.status_code == 500, resp.text

    def test_a_good_output_is_still_written(self, viz: _VizWorld) -> None:
        target = viz.store / "custom.html"

        resp = viz.client.get(
            "/graph/visualize", params={"repo": str(viz.repo), "output": str(target)}
        )

        assert resp.status_code == 200, resp.text
        assert target.is_file()


class TestVisualizeRelativeOutput:
    """A relative ``output`` is resolved against the SERVER's working directory, not joined to the
    repository: it is accepted only when that lands inside ``<repo>/.trelix``."""

    def test_is_not_joined_to_the_repository_root(
        self, viz: _VizWorld, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        resp = viz.client.get(
            "/graph/visualize",
            params={"repo": str(viz.repo), "output": ".trelix/custom.html"},
        )

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": "output path must be inside <repo>/.trelix/"}
        assert not (viz.store / "custom.html").exists()
        assert not (elsewhere / ".trelix").exists()

    def test_works_when_the_working_directory_is_inside_the_store(
        self, viz: _VizWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(viz.store)

        resp = viz.client.get(
            "/graph/visualize", params={"repo": str(viz.repo), "output": "custom.html"}
        )

        assert resp.status_code == 200, resp.text
        assert (viz.store / "custom.html").is_file()
