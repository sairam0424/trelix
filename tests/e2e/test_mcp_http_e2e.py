"""Real-socket end-to-end test of `trelix-mcp --transport http`.

The in-process tests (`packages/trelix-mcp/tests/test_http_transport.py`) serve the app over an
ASGI transport with sockets disabled, so they cannot see what exists only at the process boundary:
the console-script entry point, what uvicorn does with the flags, the Host header a TCP client
really sends (port included), SSE framing on a live connection and the SIGTERM exit code. Here the
installed `trelix-mcp` runs as a child process on a free 127.0.0.1 port.

Two clients: raw `httpx` for what an MCP client cannot express (`/health`, the 401 and its header,
a forged Host, the child's log) and `fastmcp.Client`, imported lazily, for initialize, both
protocol eras and tool calls. The `trelix-mcp` on PATH is what runs, and without one every test
skips; `release.yml`'s smoke job runs this file against the built wheel. The child gets
`os.environ` minus every `TRELIX_*` name, a fresh HOME and config file, offline flags and no
update check; it needs no model and no network beyond 127.0.0.1.
"""

import contextlib
import os
import re
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

# Placeholders, never real credentials: the server's configured value and a value it must refuse.
_SHARED_VALUE = "NOT-A-REAL-CREDENTIAL-e2e-placeholder-do-not-use"
_WRONG_VALUE = "NOT-THE-SERVER-VALUE-e2e-wrong-placeholder"

_TRELIX_MCP = shutil.which("trelix-mcp")
_READY_SECONDS = 30.0
_EXIT_SECONDS = 10.0
_LOG_TAIL_CHARS = 3000

_INIT_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "clientInfo": {"name": "e2e", "version": "0"},
    },
}
_MCP_HEADERS = {"content-type": "application/json", "accept": "application/json, text/event-stream"}

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


@dataclass(frozen=True)
class _Server:
    url: str
    root: Path
    process: subprocess.Popen[bytes]
    log: Path

    def tail(self) -> str:
        """The end of the child's combined output (the Rich banner is not ASCII)."""
        text = self.log.read_text(encoding="utf-8", errors="replace")
        return text[-_LOG_TAIL_CHARS:]


def _free_port() -> int:
    """A port nothing holds right now. Between this close and the child's bind lies its import
    time (1-2.5 s measured); a collision is an early exit with `address already in use` in the
    failure message and a re-run clears it, so there is deliberately no retry."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _child_env(home: Path) -> dict[str, str]:
    """`os.environ` minus every TRELIX_ name, plus the names this test fixes.

    `TRELIX_CONFIG_FILE` is an existing empty file and `HOME` a fresh directory, so the
    operator's own configuration is never read. `FASTMCP_CHECK_FOR_UPDATES=off` stops the start-up
    banner from asking pypi.org whether a newer FastMCP exists.
    """
    kept = {
        key: value for key, value in os.environ.items() if not key.upper().startswith("TRELIX_")
    }
    config_file = home / "trelix-env"
    config_file.write_text("", encoding="utf-8")
    return {
        **kept,
        "HOME": str(home),
        "TRELIX_CONFIG_FILE": str(config_file),
        "TRELIX_MCP_AUTH_TOKEN": _SHARED_VALUE,
        "FASTMCP_CHECK_FOR_UPDATES": "off",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NO_COLOR": "1",
    }


def _wait_ready(server: _Server) -> None:
    """Poll `GET /health` until 200; fail at once if the child exits, or after `_READY_SECONDS`."""
    deadline = time.monotonic() + _READY_SECONDS
    with httpx.Client(trust_env=False, timeout=2.0) as probe:
        while time.monotonic() < deadline:
            code = server.process.poll()
            if code is not None:
                pytest.fail(
                    f"trelix-mcp exited with {code} before /health answered:\n{server.tail()}"
                )
            with contextlib.suppress(httpx.TransportError):
                if probe.get(f"{server.url}/health").status_code == 200:
                    return
            time.sleep(0.1)
    pytest.fail(f"/health did not answer 200 in {_READY_SECONDS}s:\n{server.tail()}")


def _reap(process: subprocess.Popen[bytes]) -> None:
    """Stop the one child this test started: SIGTERM, then SIGKILL if it ignores that."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_EXIT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Server]:
    """`trelix-mcp --transport http` on a free loopback port, ready, and reaped afterwards."""
    if _TRELIX_MCP is None:
        pytest.skip("the trelix-mcp console script is not on PATH")
    # fastmcp.Client trusts proxy variables; a proxy must never sit between it and loopback.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    root = tmp_path / "root"
    home = tmp_path / "home"
    root.mkdir()
    home.mkdir()
    port = _free_port()
    command = [_TRELIX_MCP, "--transport", "http", "--host", "127.0.0.1", "--port", str(port)]
    log = tmp_path / "server.log"
    # A file, never a pipe: a chatty child cannot block on a full buffer.
    with log.open("wb") as sink:
        process = subprocess.Popen(
            [*command, "--root", str(root)],
            env=_child_env(home),
            cwd=tmp_path,
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=subprocess.STDOUT,
        )
    started = _Server(url=f"http://127.0.0.1:{port}", root=root, process=process, log=log)
    try:
        _wait_ready(started)
        yield started
    finally:
        _reap(process)


def _bearer(value: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


def _http(server: _Server) -> httpx.Client:
    return httpx.Client(base_url=server.url, trust_env=False, timeout=10.0)


def _mcp_client(server: _Server, mode: str) -> Any:
    """A `fastmcp.Client` for the server's `/mcp`, in the given protocol era."""
    fastmcp = pytest.importorskip("fastmcp")
    transports = pytest.importorskip("fastmcp.client.transports")
    transport = transports.StreamableHttpTransport(
        f"{server.url}/mcp", headers=_bearer(_SHARED_VALUE)
    )
    return fastmcp.Client(transport, mode=mode)


@pytest.mark.timeout(120)
class TestTrelixMcpOverARealSocket:
    """One server per test: the spawn costs about 1.5 s and keeps every test independent."""

    def test_health_answers_without_a_credential(self, server: _Server) -> None:
        with _http(server) as http:
            reply = http.get("/health")
        assert reply.status_code == 200
        body = reply.json()
        assert set(body) == {"status", "version"}
        assert body["status"] == "ok"
        assert re.match(r"\d+\.\d+\.\d+", body["version"])

    def test_a_missing_or_wrong_token_is_401_and_the_right_one_initializes(
        self, server: _Server
    ) -> None:
        with _http(server) as http:
            for headers in (_MCP_HEADERS, {**_MCP_HEADERS, **_bearer(_WRONG_VALUE)}):
                refused = http.post("/mcp", json=_INIT_BODY, headers=headers)
                assert refused.status_code == 401
                assert refused.headers["www-authenticate"] == "Bearer"
                assert refused.json()["detail"].startswith("Invalid or missing bearer token.")
            # The control: a real `Host: 127.0.0.1:<port>` passes the guard with the right value.
            accepted = http.post(
                "/mcp", json=_INIT_BODY, headers={**_MCP_HEADERS, **_bearer(_SHARED_VALUE)}
            )
        assert accepted.status_code == 200
        assert accepted.headers["content-type"].startswith("text/event-stream")
        assert '"protocolVersion"' in accepted.text
        assert re.search(r'"name":\s*"trelix"', accepted.text)
        log = server.log.read_text(encoding="utf-8", errors="replace")
        assert "Rejected POST /mcp: missing or invalid bearer token" in log
        assert _WRONG_VALUE not in log

    def test_a_foreign_host_is_403_even_with_the_token(self, server: _Server) -> None:
        headers = {**_MCP_HEADERS, **_bearer(_SHARED_VALUE), "Host": "evil.example"}
        with _http(server) as http:
            reply = http.post("/mcp", json=_INIT_BODY, headers=headers)
        assert reply.status_code == 403
        assert reply.json()["detail"].startswith(
            "Request rejected: the Host or Origin header is not allowed"
        )
        assert "evil.example" not in reply.text

    @pytest.mark.parametrize(
        ("mode", "version"), [("legacy", "2025-11-25"), ("2026-07-28", "2026-07-28")]
    )
    async def test_initialize_and_tools_list_in_both_protocol_eras(
        self, server: _Server, mode: str, version: str
    ) -> None:
        async with _mcp_client(server, mode) as client:
            assert client.protocol_version == version
            assert [tool.name for tool in await client.list_tools()] == ALL_TOOLS

    async def test_a_path_outside_the_root_is_refused_and_one_inside_is_not(
        self, server: _Server
    ) -> None:
        async with _mcp_client(server, "auto") as client:
            for outside in ("/", str(server.root.parent)):
                refused = await client.call_tool(
                    "search_code", {"query": "q", "repo_path": outside}, raise_on_error=False
                )
                assert refused.is_error is True
                assert (
                    refused.content[0].text == "repo_path is not inside an allowed repository root"
                )
            # The control: a refusal that fires for every path passes the loop above alone.
            reached = await client.call_tool(
                "get_symbol",
                {"qualified_name": "x", "repo_path": str(server.root)},
                raise_on_error=False,
            )
        assert reached.is_error is True
        assert reached.content[0].text.startswith("No index found at")

    def test_sigterm_stops_the_server_with_exit_code_0(self, server: _Server) -> None:
        server.process.terminate()
        code = server.process.wait(timeout=_EXIT_SECONDS)
        assert code == 0, server.tail()
