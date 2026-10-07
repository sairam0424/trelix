"""`trelix-mcp --transport http`, driven through the production wiring.

Every app here comes from `build_app(argv, environ)`: it runs `trelix_mcp.server.main()` with the
argv and environment a real start would see and builds the ASGI app from the kwargs `mcp.run`
received, so `_reject_http_only_flags`, the confinement, the error mask and the middleware list are
the ones an operator gets. Requests go over raw ASGI or an in-process httpx2 transport, never a
socket. Every expected value is a literal written here (the 401 and 403 bodies are copied in, never
imported); the one import used as an expected value is `trelix_mcp.__version__`, the package's
identity. This package's tests are governed by packages/trelix-mcp/pyproject.toml (no socket ban,
no timeout), so the markers below supply both.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import trelix_mcp
import trelix_mcp.server as srv
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from http_support import INIT_BODY, asgi_factory, build_app, raw, serving

pytestmark = [pytest.mark.disable_socket, pytest.mark.timeout(60)]

TOKEN_ENV = {"TRELIX_MCP_AUTH_TOKEN": "t0k"}
AUTH = ("authorization", "Bearer t0k")
UNAUTHORIZED = (
    "Invalid or missing bearer token. Send Authorization: Bearer <token> or "
    "X-Trelix-Api-Key: <token> matching the server's TRELIX_MCP_AUTH_TOKEN "
    "(or TRELIX_API_AUTH_TOKEN)."
)
REJECTION = (
    "Request rejected: the Host or Origin header is not allowed for this trelix server. "
    "If the hostname is legitimate, add it to TRELIX_API_ALLOWED_HOSTS (comma-separated). "
    "Set TRELIX_API_ALLOWED_HOSTS=* to disable this check."
)
CROSS_SITE = (
    "Request rejected: this trelix server refuses cross-site browser requests. Call it "
    "from a page served from the same host as the API, or set "
    "TRELIX_API_ALLOWED_HOSTS=* to disable this check."
)
MASKED = "Error calling tool 'search_code'"
REFUSAL = "repo_path is not inside an allowed repository root"
# Every tool, in the server's listing order, on both protocol eras.
ALL_TOOLS = [
    "index_codebase",
    "search_code",
    "get_symbol",
    "blast_radius",
    "build_knowledge_graph",
    "graph_search_mcp",
    "ask_agent",
    "agent_list_sessions",
    "agent_clear_session",
    "federation_list_repos",
    "federation_add_repo",
    "federation_remove_repo",
    "federation_search_all",
    "subscribe_resource",
    "unsubscribe_resource",
]


def _headers(*extra: tuple[str, str], host: str = "127.0.0.1:8766") -> list[tuple[str, str]]:
    return [
        ("host", host),
        ("content-type", "application/json"),
        ("accept", "application/json, text/event-stream"),
        *extra,
    ]


def _http_argv(root: Path, *extra: str) -> list[str]:
    return ["--transport", "http", "--root", str(root), *extra]


def _client(app: Any, mode: str) -> Client:
    transport = StreamableHttpTransport(
        "http://127.0.0.1:8766/mcp",
        headers={"Authorization": "Bearer t0k"},
        httpx_client_factory=asgi_factory(app),
    )
    return Client(transport, mode=mode)


@pytest.fixture
def root(tmp_path: Path, mark_indexed: Callable[[Path], Path]) -> Path:
    """An indexed root with an unindexed child directory (`get_symbol`'s positive control)."""
    root = mark_indexed(tmp_path / "root")
    (root / "sub").mkdir()
    return root


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    outside = tmp_path / "outside"
    outside.mkdir()
    return outside


@pytest.fixture
def app(root: Path) -> Any:
    """`trelix-mcp --transport http --root <root>` with a token: the shipped loopback defaults."""
    return build_app(_http_argv(root), TOKEN_ENV)


@pytest.mark.parametrize("mode", ["legacy", "2026-07-28"])
async def test_http_serves_legacy_and_modern_eras(app: Any, mode: str) -> None:
    """One endpoint negotiates both protocol eras and lists the same fifteen tools (the stateful
    app keeps its `GET /mcp` row in `test_stateless_and_stateful_get`).

    MUTATION: hard-code `apply_tool_profile(server.mcp, "core")` in `main()` (the list shrinks).
    Registering the health route at `/mcp` is NOT killed here: FastMCP appends custom routes after
    the MCP route, so both eras still negotiate; `test_stateless_and_stateful_get` kills that slip.
    """
    async with serving(app), _client(app, mode) as client:
        assert client.protocol_version == {"legacy": "2025-11-25", "2026-07-28": "2026-07-28"}[mode]
        assert [tool.name for tool in await client.list_tools()] == ALL_TOOLS


@pytest.fixture
def private_tool_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--tools core` hides tools through the singleton server's transform list and rewrites its
    instructions; give the test copies that monkeypatch puts back (test_tool_profile.py does the
    same), or `core` would leak into every later test."""
    monkeypatch.setattr(srv.mcp, "_transforms", list(srv.mcp.transforms))
    monkeypatch.setattr(srv.mcp, "instructions", srv.mcp.instructions)


@pytest.mark.usefixtures("private_tool_profile")
async def test_http_serves_only_the_tools_of_the_profile(root: Path) -> None:
    """`--transport http --tools core` lists the seven core tools, not all fifteen.

    MUTATION: wrap `apply_tool_profile(server.mcp, args.tools)` in `main()` in
    `if args.transport == "stdio":` (the HTTP server would serve every tool).
    """
    app = build_app(_http_argv(root, "--tools", "core"), TOKEN_ENV)
    async with serving(app), _client(app, "legacy") as client:
        assert [tool.name for tool in await client.list_tools()] == [
            "index_codebase",
            "search_code",
            "get_symbol",
            "blast_radius",
            "build_knowledge_graph",
            "graph_search_mcp",
            "ask_agent",
        ]


@pytest.mark.parametrize(
    ("headers", "detail"),
    [
        (_headers(AUTH, host="evil.example"), REJECTION),
        (_headers(AUTH, ("origin", "null")), REJECTION),
        (_headers(AUTH, ("sec-fetch-site", "cross-site")), CROSS_SITE),
    ],
    ids=["foreign-host", "origin-null", "cross-site"],
)
async def test_guard_rejects_foreign_host_origin_null_crosssite(
    app: Any, headers: list[tuple[str, str]], detail: str
) -> None:
    """The REST guard is wired outermost: a credential does not get a foreign Host through.

    MUTATION: `server.mcp.run(transport="http", host=..., port=..., path=...)` without the rest of
    `run_kwargs` (this test, the bearer rows and the stateless rows fail together).
    """
    async with serving(app):
        status, _, text = await raw(app, "POST", "/mcp", headers=headers, body=INIT_BODY.encode())
    assert (status, json.loads(text)) == (403, {"detail": detail})
    assert "evil.example" not in text


async def test_admitted_nonloopback_host_reaches_mcp(root: Path) -> None:
    """`host_origin_protection=False` leaves the REST guard as the one allow-list: a Host it admits
    (with or without a port) is not refused by a second FastMCP or SDK check further in."""
    app = build_app(_http_argv(root, "--allowed-host", "mcp.internal"), TOKEN_ENV)
    async with serving(app):
        for host in ("mcp.internal", "mcp.internal:8766"):
            headers = _headers(AUTH, host=host)
            status, out, text = await raw(
                app, "POST", "/mcp", headers=headers, body=INIT_BODY.encode()
            )
            assert status == 200, host
            assert out["content-type"].startswith("text/event-stream"), host
            assert '"protocolVersion"' in text, host


async def test_bearer_through_the_wired_app(app: Any) -> None:
    """The token from TRELIX_MCP_AUTH_TOKEN guards the wired app (the exhaustive rows are
    test_http_bearer.py's). Killed with the guard test's `run_kwargs` mutation."""
    async with serving(app):
        status, out, text = await raw(
            app, "POST", "/mcp", headers=_headers(), body=INIT_BODY.encode()
        )
        assert (status, out["www-authenticate"]) == (401, "Bearer")
        assert json.loads(text) == {"detail": UNAUTHORIZED}
        status, _, _ = await raw(
            app, "POST", "/mcp", headers=_headers(AUTH), body=INIT_BODY.encode()
        )
        assert status == 200
        api_key = _headers(("x-trelix-api-key", "t0k"))
        status, _, _ = await raw(app, "POST", "/mcp", headers=api_key, body=INIT_BODY.encode())
        assert status == 200


async def test_health_exempt(app: Any) -> None:
    """`GET`/`HEAD /health` answer without a credential and under a foreign Host.

    MUTATION: delete `register_health_route(mcp)` in server.py (404).
    """
    body = {"status": "ok", "version": trelix_mcp.__version__}
    async with serving(app):
        status, _, text = await raw(app, "GET", "/health", headers=[("host", "evil.example")])
        assert (status, json.loads(text)) == (200, body)
        # Starlette writes the JSON body for HEAD too; stripping it is the ASGI server's job.
        status, _, text = await raw(app, "HEAD", "/health", headers=[("host", "evil.example")])
        assert (status, json.loads(text)) == (200, body)


async def test_stateless_and_stateful_get(root: Path, app: Any) -> None:
    """Stateless (the default): the MCP route is POST and DELETE only. `--stateful`: a GET stream
    needs a session.

    MUTATION: register the health route at `/mcp` instead of `/health` (the stateless row answers
    200 with the health body: Starlette takes the later full match).
    """
    async with serving(app):
        status, _, _ = await raw(app, "GET", "/mcp", headers=_headers(AUTH))
    assert status == 405
    stateful = build_app(_http_argv(root, "--stateful"), TOKEN_ENV)
    async with serving(stateful):
        status, _, text = await raw(stateful, "GET", "/mcp", headers=_headers(AUTH))
    assert status == 400
    assert "Missing session ID" in text


async def test_http_errors_do_not_leak_paths(
    app: Any, root: Path, outside: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-4 over the wired app: an exception the tool did not expect is masked, a ToolError (the
    no-index text, the confinement refusal) is not.

    MUTATION: delete `server.mcp._mask_error_details = True` in `_start_http` (the first and the
    `boom` assertions fail).
    """
    assert srv.mcp._mask_error_details is True

    def _raise(repo_path: str) -> Any:
        raise RuntimeError(f"boom {outside}")

    monkeypatch.setattr(srv, "_get_retriever", _raise)
    async with serving(app), _client(app, "auto") as client:
        result = await client.call_tool(
            "search_code", {"query": "q", "repo_path": str(root)}, raise_on_error=False
        )
        assert result.is_error is True
        assert result.content[0].text == MASKED
        assert str(outside) not in result.content[0].text
        result = await client.call_tool(
            "get_symbol",
            {"qualified_name": "x", "repo_path": str(root / "sub")},
            raise_on_error=False,
        )
        assert result.is_error is True
        assert result.content[0].text.startswith("No index found at")
        result = await client.call_tool(
            "search_code", {"query": "q", "repo_path": str(outside)}, raise_on_error=False
        )
        assert result.is_error is True
        assert result.content[0].text == REFUSAL


async def test_a_tilde_path_is_expanded_by_the_confinement_but_not_by_the_tool(
    app: Any, root: Path, outside: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 19's last tool-call sentence. With HOME at the root, `~/sub` names the existing
    `<root>/sub`, so the confinement lets it through, and the tool answers its own unmasked
    `does not exist` text for the literal `~/sub`. With HOME outside every root the confinement
    answers the refusal instead.

    MUTATIONS: `Path(repo_path).expanduser()` in `arguments._check_existing_dir` (the first call
    then gets past the directory check and fails inside `IndexConfig` as a protocol error, which
    the client raises); `.expanduser()` removed from `is_within_allowed_roots` (the first call
    answers the refusal).
    """
    tilde_call = {"query": "q", "repo_path": "~/sub"}
    async with serving(app), _client(app, "auto") as client:
        monkeypatch.setenv("HOME", str(root))
        result = await client.call_tool("search_code", tilde_call, raise_on_error=False)
        assert result.is_error is True
        assert result.content[0].text == (
            "repo_path does not exist: '~/sub'; pass the absolute path of the repository root."
        )
        monkeypatch.setenv("HOME", str(outside))
        result = await client.call_tool("search_code", tilde_call, raise_on_error=False)
        assert result.is_error is True
        assert result.content[0].text == REFUSAL


def test_fastmcp_mask_attribute_canary() -> None:
    """`_mask_error_details` is FastMCP's private name for the constructor's `mask_error_details`;
    `_start_http` sets it. A rename fails here before a release does."""
    assert FastMCP("x", mask_error_details=True)._mask_error_details is True
    assert FastMCP("x")._mask_error_details is False
