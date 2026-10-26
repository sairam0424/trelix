"""The federation tools stay inside what the server allows: `config_path` and the registry entries.

* `config_path` is refused with one of two fixed texts, on every federation tool the server lists:
  `config_path is not inside an allowed config directory` (outside ~/.config/trelix and
  <cwd>/.trelix) and `config_path is not a repository registry` (a name not ending in `.json`, or
  an existing non-empty file that is not a JSON object with a `repos` list). The resolved value
  and the allowed directories go to the operator log, never to the client.
* A file that is not a registry is never written: `RepoRegistry.load` reads it as an empty registry
  and the tools would otherwise `save()` over it (`.trelix/index.db`, the operator env file).
  `federation_remove_repo` does not save either when it removed nothing.
* Under --root or TRELIX_ALLOWED_REPO_ROOTS, `federation_list_repos` shows only the entries inside
  the roots and counts the others as `repos_outside_roots`, and `federation_remove_repo` cannot
  drop an entry outside them.

Every expected value is a literal written here, and the files are real ones in tmp_path. This
package's tests are governed by packages/trelix-mcp/pyproject.toml (no socket ban, no timeout), so
the markers below supply both.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import trelix_mcp.server as srv
from fastmcp import Client

pytestmark = [pytest.mark.disable_socket, pytest.mark.timeout(60)]

DIRECTORY_REFUSAL = "config_path is not inside an allowed config directory"
REGISTRY_REFUSAL = "config_path is not a repository registry"

# Every tool whose input schema has config_path; the test below fails if the server lists another.
FEDERATION_TOOLS = [
    "federation_list_repos",
    "federation_add_repo",
    "federation_remove_repo",
    "federation_search_all",
]


def _arguments(tool: str, repo: Path) -> dict[str, Any]:
    """What each tool needs besides config_path, so that the path is the only thing that varies."""
    return {
        "federation_list_repos": {},
        "federation_add_repo": {"alias": "svc", "path": str(repo)},
        "federation_remove_repo": {"alias": "svc"},
        "federation_search_all": {"query": "q"},
    }[tool]


def _refused(tool: str, repo: Path, error: str) -> dict[str, Any]:
    """The whole response of each tool to a refused config_path."""
    return {
        "federation_list_repos": {"repos": [], "count": 0, "error": error},
        "federation_add_repo": {"added": False, "alias": "svc", "path": str(repo), "error": error},
        "federation_remove_repo": {"removed": False, "alias": "svc", "error": error},
        "federation_search_all": {
            "results": [],
            "next_cursor": None,
            "total_available": 0,
            "repos_searched": 0,
            "repos_skipped": 0,
            "error": error,
        },
    }[tool]


def _call(tool: str, repo: Path, config_path: Path | str) -> dict[str, Any]:
    result: dict[str, Any] = getattr(srv, tool)(
        **_arguments(tool, repo), config_path=str(config_path)
    )
    return result


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """cwd is tmp_path/work, so tmp_path/work/.trelix is an allowed config directory.

    `repo` is a real directory, which `federation_add_repo` requires for its `path`.
    """
    work = tmp_path / "work"
    config = work / ".trelix"
    config.mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(work)
    return SimpleNamespace(work=work, config=config, repo=repo)


@pytest.mark.parametrize("name", FEDERATION_TOOLS)
async def test_every_federation_tool_refuses_a_config_path_with_a_fixed_text(
    workdir: SimpleNamespace, name: str
) -> None:
    async with Client(srv.mcp) as client:
        tools = await client.list_tools()
        # The tools found in the schemas, not a list kept here: a tool that gains a config_path
        # fails this line until it is added to FEDERATION_TOOLS and so covered by this test.
        assert sorted(t.name for t in tools if "config_path" in t.input_schema["properties"]) == (
            sorted(FEDERATION_TOOLS)
        )

        with patch("trelix_mcp.server.RepoRegistry") as registry:
            result = await client.call_tool(
                name,
                {**_arguments(name, workdir.repo), "config_path": "/etc/passwd"},
                raise_on_error=False,
            )

    assert result.is_error is False
    assert result.structured_content == _refused(name, workdir.repo, DIRECTORY_REFUSAL)
    text = json.dumps(result.structured_content)
    for leaked in ("passwd", ".trelix", ".config", str(workdir.work)):
        assert leaked not in text, leaked
    registry.load.assert_not_called()


def test_the_refusal_detail_goes_to_the_log_and_not_to_the_client(
    workdir: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    index_db = workdir.config / "index.db"
    index_db.write_bytes(b"SQLite format 3\x00")

    with caplog.at_level(logging.WARNING, logger="trelix_mcp"):
        outside = srv.federation_list_repos(config_path="/etc/passwd")
        not_registry = srv.federation_list_repos(config_path=str(index_db))

    assert outside["error"] == DIRECTORY_REFUSAL
    assert not_registry["error"] == REGISTRY_REFUSAL
    outside_log, not_registry_log = [r.getMessage() for r in caplog.records]
    assert "passwd" in outside_log
    assert str(workdir.config) in outside_log  # the allowed directories, for the operator
    assert "outside the allowed config directories" in outside_log
    assert "not a .json repository registry" in not_registry_log


def test_a_hostile_config_path_is_logged_escaped_and_cut(
    workdir: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    hostile = "/etc/passwd\nWARNING forged line " + "x" * 300

    with caplog.at_level(logging.WARNING, logger="trelix_mcp"):
        response = srv.federation_list_repos(config_path=hostile)

    assert response["error"] == DIRECTORY_REFUSAL
    (record,) = caplog.records
    message = record.getMessage()
    assert "\n" not in message
    assert "\\nWARNING forged line xxxxx" in message
    assert "x" * 101 not in message


def test_a_config_path_that_cannot_be_resolved_gets_a_fixed_refusal_not_an_exception(
    workdir: SimpleNamespace,
) -> None:
    """A NUL byte makes Path.resolve() raise ValueError, and a symlink loop makes it raise
    RuntimeError on Python 3.12 (3.13 and later return the path instead, which the registry check
    then refuses): the client gets one of the two fixed texts either way, never the exception."""
    loop_a = workdir.config / "loop-a.json"
    loop_b = workdir.config / "loop-b.json"
    loop_a.symlink_to(loop_b)
    loop_b.symlink_to(loop_a)

    nul = srv.federation_list_repos(config_path="a\x00b")
    loop = srv.federation_list_repos(config_path=str(loop_a))

    assert nul["error"] == DIRECTORY_REFUSAL
    assert loop["error"] in (DIRECTORY_REFUSAL, REGISTRY_REFUSAL)
    assert nul["repos"] == [] and loop["repos"] == []


def test_add_and_remove_never_overwrite_a_non_registry_file(workdir: SimpleNamespace) -> None:
    """The SQLite index stand-in sits in the allowed directory, as `<cwd>/.trelix/index.db` does."""
    index_db = workdir.config / "index.db"
    index_db.write_bytes(b"SQLite format 3\x00" + b"\x01" * 64)
    before = _digest(index_db)

    add = _call("federation_add_repo", workdir.repo, index_db)
    remove = _call("federation_remove_repo", workdir.repo, index_db)

    assert add == _refused("federation_add_repo", workdir.repo, REGISTRY_REFUSAL)
    assert remove == _refused("federation_remove_repo", workdir.repo, REGISTRY_REFUSAL)
    assert _digest(index_db) == before


@pytest.mark.parametrize(
    ("name", "content"),
    [
        # Valid registry JSON under the wrong name: only the name rule refuses it.
        ("repos.txt", b'{"repos": []}'),
        ("env.json", b"TRELIX_PROVIDER=none\n"),
        ("array.json", b"[]"),
        ("no-repos-key.json", b'{"other": []}'),
        ("repos-not-a-list.json", b'{"repos": "x"}'),
        ("whitespace.json", b"  \n"),
        ("not-text.json", b"\xff\xfe\x00\x01"),
    ],
)
@pytest.mark.parametrize("tool", FEDERATION_TOOLS)
def test_a_file_that_is_not_a_registry_is_refused_by_every_tool_and_left_alone(
    workdir: SimpleNamespace, tool: str, name: str, content: bytes
) -> None:
    target = workdir.config / name
    target.write_bytes(content)

    response = _call(tool, workdir.repo, target)

    assert response == _refused(tool, workdir.repo, REGISTRY_REFUSAL)
    assert target.read_bytes() == content


@pytest.mark.parametrize("tool", FEDERATION_TOOLS)
@pytest.mark.parametrize("directory", ["elsewhere", "work/.trelixevil", "work"])
def test_a_valid_registry_outside_the_config_directories_is_refused_and_left_alone(
    workdir: SimpleNamespace, tmp_path: Path, tool: str, directory: str
) -> None:
    """Only the directory rule refuses these: the name and the content are those of a registry.

    `work/.trelixevil` is the sibling a string-prefix check would let through, and `work` is the
    working directory itself, which is not its `.trelix`.
    """
    target = tmp_path / directory / "repos.json"
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(b'{"repos": []}')

    response = _call(tool, workdir.repo, target)

    assert response == _refused(tool, workdir.repo, DIRECTORY_REFUSAL)
    assert target.read_bytes() == b'{"repos": []}'


def test_a_json_symlink_is_judged_by_what_it_points_to(
    workdir: SimpleNamespace, tmp_path: Path
) -> None:
    index_db = workdir.config / "index.db"
    index_db.write_bytes(b"SQLite format 3\x00" + b"\x01" * 64)
    onto_index = workdir.config / "link.json"
    onto_index.symlink_to(index_db)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    registry = elsewhere / "repos.json"
    registry.write_bytes(b'{"repos": []}')
    out_of_the_directories = workdir.config / "out.json"
    out_of_the_directories.symlink_to(registry)
    before = _digest(index_db)

    onto_index_response = _call("federation_add_repo", workdir.repo, onto_index)
    out_response = _call("federation_add_repo", workdir.repo, out_of_the_directories)

    assert onto_index_response == _refused("federation_add_repo", workdir.repo, REGISTRY_REFUSAL)
    assert out_response == _refused("federation_add_repo", workdir.repo, DIRECTORY_REFUSAL)
    assert _digest(index_db) == before
    assert registry.read_bytes() == b'{"repos": []}'


def test_a_directory_is_not_a_registry(workdir: SimpleNamespace) -> None:
    folder = workdir.config / "folder.json"
    folder.mkdir()

    response = _call("federation_add_repo", workdir.repo, folder)

    assert response == _refused("federation_add_repo", workdir.repo, REGISTRY_REFUSAL)
    assert folder.is_dir()


def test_an_existing_registry_still_works(workdir: SimpleNamespace) -> None:
    registry = workdir.config / "team.json"
    existing = {"alias": "a", "path": "/srv/a", "weight": 2.0}
    registry.write_text(json.dumps({"repos": [existing]}))

    listed = srv.federation_list_repos(config_path=str(registry))
    added = srv.federation_add_repo(
        alias="svc", path=str(workdir.repo), weight=0.5, config_path=str(registry)
    )

    assert listed == {"repos": [existing], "count": 1, "error": None}
    assert added == {"added": True, "alias": "svc", "path": str(workdir.repo), "error": None}
    assert json.loads(registry.read_text()) == {
        "repos": [existing, {"alias": "svc", "path": str(workdir.repo), "weight": 0.5}]
    }

    removed = srv.federation_remove_repo(alias="a", config_path=str(registry))

    assert removed == {"removed": True, "alias": "a", "error": None}
    assert json.loads(registry.read_text()) == {
        "repos": [{"alias": "svc", "path": str(workdir.repo), "weight": 0.5}]
    }


def test_a_missing_file_is_created_by_the_first_add(workdir: SimpleNamespace) -> None:
    registry = workdir.config / "new" / "repos.json"

    added = srv.federation_add_repo(alias="svc", path=str(workdir.repo), config_path=str(registry))

    assert added == {"added": True, "alias": "svc", "path": str(workdir.repo), "error": None}
    assert json.loads(registry.read_text()) == {
        "repos": [{"alias": "svc", "path": str(workdir.repo), "weight": 1.0}]
    }


def test_an_empty_file_is_accepted_by_add(workdir: SimpleNamespace) -> None:
    registry = workdir.config / "empty.json"
    registry.write_bytes(b"")

    added = srv.federation_add_repo(alias="svc", path=str(workdir.repo), config_path=str(registry))

    assert added == {"added": True, "alias": "svc", "path": str(workdir.repo), "error": None}
    assert json.loads(registry.read_text()) == {
        "repos": [{"alias": "svc", "path": str(workdir.repo), "weight": 1.0}]
    }


def test_removing_an_unregistered_alias_writes_nothing(workdir: SimpleNamespace) -> None:
    registry = workdir.config / "repos.json"
    # One line, which save() would rewrite indented: any write shows in the bytes.
    registry.write_text('{"repos": [{"alias": "a", "path": "/srv/a", "weight": 1.0}]}')
    before = (registry.read_bytes(), registry.stat().st_mtime_ns)
    missing = workdir.config / "missing.json"

    response = srv.federation_remove_repo(alias="nope", config_path=str(registry))
    on_missing = srv.federation_remove_repo(alias="nope", config_path=str(missing))

    assert response == {"removed": False, "alias": "nope", "error": None}
    assert on_missing == {"removed": False, "alias": "nope", "error": None}
    assert (registry.read_bytes(), registry.stat().st_mtime_ns) == before
    assert not missing.exists()


@pytest.fixture
def served(workdir: SimpleNamespace, tmp_path: Path) -> SimpleNamespace:
    """A registry with one entry inside `root` and one outside it; no roots are configured yet."""
    root = tmp_path / "served"
    inside = root / "app"
    inside.mkdir(parents=True)
    outside = tmp_path / "private"
    outside.mkdir()
    registry = workdir.config / "repos.json"
    registry.write_text(
        json.dumps(
            {
                "repos": [
                    {"alias": "in", "path": str(inside), "weight": 1.0},
                    {"alias": "out", "path": str(outside), "weight": 0.5},
                ]
            },
            indent=2,
        )
    )
    return SimpleNamespace(root=root, inside=inside, outside=outside, registry=registry)


def test_list_shows_only_the_entries_inside_the_roots_and_counts_the_rest(
    served: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(srv, "_allowed_roots", (served.root.resolve(),))

    listed = srv.federation_list_repos(config_path=str(served.registry))
    refused = srv.federation_list_repos(config_path="/etc/passwd")

    assert listed == {
        "repos": [{"alias": "in", "path": str(served.inside), "weight": 1.0}],
        "count": 1,
        "error": None,
        "repos_outside_roots": 1,
    }
    assert str(served.outside) not in json.dumps(listed)
    assert refused == {
        "repos": [],
        "count": 0,
        "error": DIRECTORY_REFUSAL,
        "repos_outside_roots": 0,
    }


def test_list_is_unfiltered_when_no_roots_are_configured(served: SimpleNamespace) -> None:
    listed = srv.federation_list_repos(config_path=str(served.registry))

    assert listed == {
        "repos": [
            {"alias": "in", "path": str(served.inside), "weight": 1.0},
            {"alias": "out", "path": str(served.outside), "weight": 0.5},
        ],
        "count": 2,
        "error": None,
    }


def test_remove_cannot_drop_an_entry_outside_the_roots(
    served: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(srv, "_allowed_roots", (served.root.resolve(),))
    before = _digest(served.registry)

    hidden = srv.federation_remove_repo(alias="out", config_path=str(served.registry))
    unknown = srv.federation_remove_repo(alias="nope", config_path=str(served.registry))

    assert hidden == {"removed": False, "alias": "out", "error": None}
    assert unknown == {"removed": False, "alias": "nope", "error": None}
    assert _digest(served.registry) == before

    visible = srv.federation_remove_repo(alias="in", config_path=str(served.registry))

    assert visible == {"removed": True, "alias": "in", "error": None}
    # The entry outside the roots is still in the file: the roots hide it, they do not delete it.
    assert json.loads(served.registry.read_text()) == {
        "repos": [{"alias": "out", "path": str(served.outside), "weight": 0.5}]
    }


def test_remove_can_drop_any_entry_when_no_roots_are_configured(served: SimpleNamespace) -> None:
    removed = srv.federation_remove_repo(alias="out", config_path=str(served.registry))

    assert removed == {"removed": True, "alias": "out", "error": None}
    assert json.loads(served.registry.read_text()) == {
        "repos": [{"alias": "in", "path": str(served.inside), "weight": 1.0}]
    }
