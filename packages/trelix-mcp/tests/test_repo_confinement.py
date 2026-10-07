"""`trelix-mcp --root PATH` and TRELIX_ALLOWED_REPO_ROOTS confine what a caller may name.

* Every tool with a `repo_path` argument, and `federation_add_repo`'s `path`, is refused before
  the tool runs when the value is blank or resolves outside every root: `isError: true` with the
  text `repo_path is not inside an allowed repository root` (`path ...` for federation_add_repo).
* The trelix://repo/... resources are refused the same way, where FastMCP has parsed the URI.
* federation_search_all searches only the registry entries inside the roots and says how many it
  left out (`repos_outside_roots`) in every shape it returns.
* Stdio with no --root and no variable installs nothing, so every other test runs unconfined.

Every expected value is a literal written here, and tool and resource calls go through an
in-process `fastmcp.Client` so the middleware runs as it does for a real client. This package's
tests are governed by packages/trelix-mcp/pyproject.toml (no socket ban, no timeout), so the
markers below supply both.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import trelix_mcp.server as srv
from fastmcp import Client
from mcp.shared.exceptions import MCPError
from trelix_mcp.confinement import RepoConfinementMiddleware, entries_inside_roots

from trelix.federation.registry import RepoEntry

pytestmark = [pytest.mark.disable_socket, pytest.mark.timeout(60)]

REFUSAL = "repo_path is not inside an allowed repository root"
PATH_REFUSAL = "path is not inside an allowed repository root"

# Every tool whose input schema has repo_path, in the server's listing order.
CONFINED_TOOLS = [
    "index_codebase",
    "search_code",
    "get_symbol",
    "blast_radius",
    "build_knowledge_graph",
    "graph_search_mcp",
    "ask_agent",
    "agent_list_sessions",
    "agent_clear_session",
]
# What each tool needs besides repo_path, so that the path is the only thing that varies.
OTHER_ARGUMENTS: dict[str, dict[str, str]] = {
    "search_code": {"query": "q"},
    "get_symbol": {"qualified_name": "Cls.m"},
    "blast_radius": {"symbol_name": "Cls.m"},
    "graph_search_mcp": {"query": "q"},
    "ask_agent": {"query": "q"},
    "agent_clear_session": {"session_id": "s1"},
}


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """One allowed root (with a real `sub` directory) and the values that must be refused.

    cwd is the root, so a blank or whitespace repo_path, which Path("") turns into the cwd, would
    resolve INSIDE it: only the blank check refuses it. The sibling and the `..` spelling need not
    exist (the middleware refuses them before any directory check); `outside` exists because `link`
    points at it. The NUL byte is not a path at all: `Path.resolve()` raises on it under every
    Python, and it must be refused rather than answered with an internal error. `loop` is a symlink
    to itself, which only Python 3.12's `resolve()` raises on (see the gated test below).
    """
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "link"
    link.symlink_to(outside, target_is_directory=True)
    loop = root / "loop"
    loop.symlink_to(loop)
    monkeypatch.chdir(root)
    srv._install_confinement((root.resolve(),))
    return SimpleNamespace(
        root=root,
        outside=outside,
        loop=loop,
        refused=[
            str(outside),
            str(tmp_path / "root-evil"),
            str(link),
            f"{root}/../outside",
            "",
            "   ",
            "\x00",
        ],
    )


def _confinement_middlewares() -> list[RepoConfinementMiddleware]:
    return [m for m in srv.mcp.middleware if isinstance(m, RepoConfinementMiddleware)]


async def test_every_repo_path_tool_is_confined(
    roots: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with Client(srv.mcp) as client:
        tools = await client.list_tools()
        assert [t.name for t in tools if "repo_path" in t.input_schema["properties"]] == (
            CONFINED_TOOLS
        )

        for name in CONFINED_TOOLS:
            for value in roots.refused:
                arguments = {**OTHER_ARGUMENTS.get(name, {}), "repo_path": value}
                result = await client.call_tool(name, arguments, raise_on_error=False)
                assert result.is_error is True, (name, value)
                assert result.content[0].text == REFUSAL, (name, value)

        # Positive control: the root itself and a directory under it reach the tool body, which
        # answers that this (never indexed) repository has no index.
        for name in CONFINED_TOOLS:
            if name == "index_codebase":
                continue
            for inside in (str(roots.root), str(roots.root / "sub")):
                arguments = {**OTHER_ARGUMENTS.get(name, {}), "repo_path": inside}
                result = await client.call_tool(name, arguments, raise_on_error=False)
                assert result.is_error is True, (name, inside)
                assert result.content[0].text.startswith("No index found at "), (name, inside)

        # index_codebase has no index to miss; its body is the Indexer, patched to prove it ran.
        indexer = MagicMock()
        indexer.return_value.index.return_value = {"files_indexed": 0}
        monkeypatch.setattr(srv, "Indexer", indexer)
        sub = str(roots.root / "sub")
        result = await client.call_tool("index_codebase", {"repo_path": sub}, raise_on_error=False)
        assert result.is_error is False
        assert indexer.call_args.args[0].repo_path == str((roots.root / "sub").resolve())

        # A missing repo_path is FastMCP's own validation error, as it was before.
        result = await client.call_tool("search_code", {"query": "q"}, raise_on_error=False)
        assert result.is_error is True
        assert result.content[0].text.startswith("1 validation error")

        # config_path points into tmp_path so that, should the refusal ever regress, the tool body
        # writes the alias there and not into the developer's real ~/.config/trelix/repos.json.
        result = await client.call_tool(
            "federation_add_repo",
            {
                "alias": "x",
                "path": str(roots.outside),
                "config_path": str(roots.root / ".trelix" / "repos.json"),
            },
            raise_on_error=False,
        )
        assert result.is_error is True
        assert result.content[0].text == PATH_REFUSAL

        # A relative repo_path whose working directory no longer exists cannot be resolved either
        # (Path.resolve() raises FileNotFoundError): the refusal, not an internal error.
        gone = roots.root / "gone"
        gone.mkdir()
        monkeypatch.chdir(gone)
        gone.rmdir()
        result = await client.call_tool(
            "search_code", {"query": "q", "repo_path": "rel"}, raise_on_error=False
        )
        assert result.is_error is True
        assert result.content[0].text == REFUSAL


async def test_resource_uri_is_confined(
    roots: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo segment is resolved against the cwd, which is tmp_path here: `root` is the root."""
    monkeypatch.chdir(tmp_path)
    async with Client(srv.mcp) as client:
        stats = json.loads((await client.read_resource("trelix://repo/root/stats"))[0].text)
        assert "symbol_count" in stats or "error" in stats
        manifest = json.loads((await client.read_resource("trelix://repo/root/manifest"))[0].text)
        assert "file_count" in manifest or "error" in manifest
        symbol_uri = "trelix://repo/root/symbols/Cls.m"
        symbol = json.loads((await client.read_resource(symbol_uri))[0].text)
        assert "qualified_name" in symbol or "error" in symbol

        for uri in (
            "trelix://repo/outside/stats",
            "trelix://repo/outside/manifest",
            "trelix://repo/outside/symbols/Cls.m",
        ):
            with pytest.raises(MCPError) as exc_info:
                await client.read_resource(uri)
            assert str(exc_info.value) == REFUSAL, uri

        # A blank segment is refused before the handler runs, with the cwd AT the root: there a
        # missing blank check would resolve `root/ ` inside the root and the handler would answer
        # `{"error": "repo_path is not a directory"}` JSON instead of the refusal.
        monkeypatch.chdir(roots.root)
        with pytest.raises(MCPError) as exc_info:
            await client.read_resource("trelix://repo/%20/stats")
        assert str(exc_info.value) == REFUSAL

        # trelix://index/stats names no repository and is not confined.
        hint = json.loads((await client.read_resource("trelix://index/stats"))[0].text)
        assert hint["hint"].startswith("Use trelix://repo/")


@pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason="3.13+ resolves a symlink loop to itself (inside the root, so the tool's own directory "
    "check answers); only 3.12's Path.resolve() raises RuntimeError on it",
)
async def test_symlink_loop_is_refused_where_resolve_raises(
    roots: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path `resolve()` raises on is refused like one outside, on every confined surface.

    Without the guard the RuntimeError escapes: the tool call answers `Internal server error`, the
    resource read echoes the path in the handler's error, and federation_search_all raises.
    """
    loop = str(roots.loop)
    async with Client(srv.mcp) as client:
        result = await client.call_tool(
            "search_code", {"query": "q", "repo_path": loop}, raise_on_error=False
        )
        assert result.is_error is True
        assert result.content[0].text == REFUSAL

        # cwd is the root, so the one-segment `loop` of the URI names root/loop.
        with pytest.raises(MCPError) as exc_info:
            await client.read_resource("trelix://repo/loop/stats")
        assert str(exc_info.value) == REFUSAL

    assert entries_inside_roots([RepoEntry("loop", loop)], (roots.root.resolve(),)) == ([], 1)


def test_federation_entries_outside_roots_are_not_searched(
    backends: SimpleNamespace, roots: SimpleNamespace
) -> None:
    inside = RepoEntry("in", str(roots.root / "a"))
    outside = RepoEntry("out", str(roots.outside))
    # A registry entry that cannot be resolved at all is left out like one outside, not raised
    # through federation_search_all.
    unresolvable = RepoEntry("nul", "\x00")
    srv.RepoRegistry.load.return_value.list.return_value = [inside, outside, unresolvable]

    result = srv.federation_search_all(query="q")

    # The registry handed to FederatedRetriever holds only the entry inside the root.
    config_path, entries = srv.RepoRegistry.call_args.args
    assert Path(config_path) == Path.home() / ".config" / "trelix" / "repos.json"
    assert entries == [RepoEntry("in", str(roots.root / "a"))]
    assert srv.FederatedRetriever.call_args.args[0] is srv.RepoRegistry.return_value
    assert result["repos_outside_roots"] == 2
    assert result["repos_searched"] == 1
    assert result["results"] == []


def test_federation_every_shape_carries_repos_outside_roots(
    backends: SimpleNamespace, roots: SimpleNamespace
) -> None:
    inside = RepoEntry("in", str(roots.root / "a"))
    registry = srv.RepoRegistry.load.return_value

    # Both entries outside: the empty-registry shape, with the count.
    registry.list.return_value = [
        RepoEntry("out1", str(roots.outside)),
        RepoEntry("out2", str(roots.outside / "b")),
    ]
    assert srv.federation_search_all(query="q") == {
        "results": [],
        "next_cursor": None,
        "total_available": 0,
        "repos_searched": 0,
        "repos_skipped": 0,
        "error": None,
        "repos_outside_roots": 2,
    }
    srv.FederatedRetriever.assert_not_called()

    # The config-path error: nothing was read, so the count is 0.
    rejected = srv.federation_search_all(query="q", config_path="/etc/passwd")
    assert rejected["error"] is not None
    assert rejected["repos_outside_roots"] == 0

    # Every kept repo unindexed: the error shape, with the count.
    registry.list.return_value = [inside, RepoEntry("out", str(roots.outside))]
    missing = "No index found at /r/a/.trelix/index.db. Run trelix index /r/a first."
    srv.FederatedRetriever.return_value.unindexed_repos.return_value = [(inside, missing)]
    unindexed = srv.federation_search_all(query="q")
    assert unindexed["error"] == missing
    assert unindexed["repos_searched"] == 0
    assert unindexed["repos_outside_roots"] == 1


def test_federation_search_all_is_unchanged_without_roots(backends: SimpleNamespace) -> None:
    """The fixture's entries have no `path`: with no roots, no entry is read and no key is added."""
    result = srv.federation_search_all(query="q")

    assert "repos_outside_roots" not in result
    assert result["repos_searched"] == 1
    srv.RepoRegistry.assert_not_called()


def test_stdio_default_unconfined(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(srv.mcp, "run", MagicMock())
    monkeypatch.setattr("sys.argv", ["trelix-mcp"])

    srv.main()

    assert _confinement_middlewares() == []
    assert srv._allowed_roots == ()

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr("sys.argv", ["trelix-mcp", "--root", str(repo)])

    srv.main()

    assert len(_confinement_middlewares()) == 1
    assert srv._allowed_roots == (repo.resolve(),)
    assert srv.mcp.run.call_count == 2


def test_env_roots_confine_stdio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The VS Code extension launches the server with its own environment: an exported
    TRELIX_ALLOWED_REPO_ROOTS confines the stdio server although no --root was given."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("TRELIX_ALLOWED_REPO_ROOTS", str(repo))
    monkeypatch.setattr(srv.mcp, "run", MagicMock())
    monkeypatch.setattr("sys.argv", ["trelix-mcp"])

    srv.main()

    assert len(_confinement_middlewares()) == 1
    assert srv._allowed_roots == (repo.resolve(),)
