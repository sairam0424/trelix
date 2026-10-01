"""Host / Origin / fetch-metadata guard for the REST API (`api/request_guard.py`).

The defect these tests exist for: `trelix serve` with no credential is open by design,
and nothing looked at the `Host` or `Origin` header. A page on an attacker's domain that
is re-resolved to 127.0.0.1 (DNS rebinding) is then same-origin with the server and can
READ `/search`, `/ask` and `/parse`; without any rebinding a web page can still fire
no-cors GETs that spend LLM money. A deployment with a token is not exposed (the page has
no key), which is why the guard is a loopback-and-open-mode measure, not a replacement
for `TRELIX_API_AUTH_TOKEN`.

Written from the attacker's position: the hostile header is constructed and the response
is asserted to be a refusal that does not echo the hostile value.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from trelix.api.app import create_app
from trelix.api.request_guard import (
    CROSS_SITE_DETAIL,
    REJECTION_DETAIL,
    RequestGuardMiddleware,
    _route_path,
    is_allowed_host,
    is_allowed_origin,
    loopback_bind_allowed_hosts,
    normalise_host,
    resolve_allowed_hosts,
)

LOOPBACK = ("localhost", "127.0.0.1", "::1")
EVIL = "evil-rebind-canary.example"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("localhost", "localhost"),
        ("localhost:8765", "localhost"),
        ("LocalHost:8765", "localhost"),
        ("localhost.", "localhost"),
        ("localhost.:8765", "localhost"),
        ("127.0.0.1", "127.0.0.1"),
        ("127.0.0.1:1", "127.0.0.1"),
        ("[::1]", "::1"),
        ("[::1]:8765", "::1"),
        ("[0:0:0:0:0:0:0:1]:80", "::1"),
        ("[0:0:0:0:0:0:0:1]", "::1"),
        ("::1", "::1"),
        ("My-Host.Example.COM", "my-host.example.com"),
        ("host_name", "host_name"),
    ],
)
def test_normalise_host_accepts_and_canonicalises(raw: str, expected: str) -> None:
    assert normalise_host(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        " ",
        "localhost@evil.com",
        "user:pw@localhost",
        "evil.com#localhost",
        "evil.com?localhost",
        "localhost/evil",
        "localhost\\evil",
        "[::1]evil",
        "[::1",
        "::1]",
        "[::1]:",
        "[::1]:port",
        "[not-an-ip]",
        "[127.0.0.1]",
        "localhost:",
        "localhost:80:80x",
        "localhost:abc",
        "localhost\t",
        "local host",
        " localhost",
        "localhost ",
        "local\x00host",
        "local\nhost",
        "local\rhost",
        "loc\x7fal",
        "localhost..",
        "..",
        ".",
        "a..b",
        ".localhost",
        "*",
        "*.example.com",
        "lócalhost",
        "localhost%00",
    ],
)
def test_normalise_host_rejects_junk(raw: str | None) -> None:
    assert normalise_host(raw) is None


def test_lookalike_hosts_normalise_but_are_not_allowed() -> None:
    allowed = frozenset(LOOPBACK)
    for lookalike in ("localhost.evil.com", "127.0.0.1.evil.com", "evil.com", "127.0.0.2"):
        assert normalise_host(lookalike) == lookalike
        assert not is_allowed_host(lookalike, allowed)


@pytest.mark.parametrize(
    "value", ["localhost", "localhost:8765", "LOCALHOST", "localhost.", "[::1]:8765", "127.0.0.1"]
)
def test_is_allowed_host_matches_loopback_variants(value: str) -> None:
    assert is_allowed_host(value, frozenset(LOOPBACK))


@pytest.mark.parametrize("value", [None, "", "localhost@evil.com", "[::1]evil", "10.0.0.5"])
def test_is_allowed_host_rejects_everything_else(value: str | None) -> None:
    assert not is_allowed_host(value, frozenset(LOOPBACK))


def test_an_empty_allow_list_allows_nothing() -> None:
    assert not is_allowed_host("localhost", frozenset())


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost",
        "http://localhost:3000",
        "https://localhost:8443",
        "HTTP://LOCALHOST:3000",
        "http://127.0.0.1:8765",
        "http://[::1]:8765",
        "https://localhost.",
    ],
)
def test_is_allowed_origin_accepts_allowed_hostnames_on_any_port_and_scheme(origin: str) -> None:
    assert is_allowed_origin(origin, frozenset(LOOPBACK))


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "",
        "null",
        "NULL",
        "http://evil.com",
        "https://localhost.evil.com",
        "http://localhost@evil.com",
        "http://evil.com@localhost",
        "http://localhost/path",
        "http://localhost/",
        "http://localhost?x=1",
        "http://localhost#x",
        "ftp://localhost",
        "file://localhost",
        "chrome-extension://localhost",
        "localhost",
        "//localhost",
        "http://",
        "http:// localhost",
        "http://localhost\t",
        "http://local\x00host",
    ],
)
def test_is_allowed_origin_rejects(origin: str | None) -> None:
    assert not is_allowed_origin(origin, frozenset(LOOPBACK))


class TestResolveAllowedHosts:
    """The decision matrix behind `create_app(allowed_hosts=...)` and the env var."""

    def test_nothing_configured_means_no_guard(self) -> None:
        assert resolve_allowed_hosts(None, None) is None
        assert resolve_allowed_hosts(None, "") is None
        assert resolve_allowed_hosts(None, "  ") is None
        assert resolve_allowed_hosts(None, " , ,, ") is None

    def test_env_alone_turns_the_guard_on_with_loopback_names(self) -> None:
        assert resolve_allowed_hosts(None, "trelix.internal") == frozenset(
            {*LOOPBACK, "trelix.internal"}
        )

    def test_explicit_hosts_turn_the_guard_on_without_adding_loopback(self) -> None:
        assert resolve_allowed_hosts(("localhost",), None) == frozenset({"localhost"})

    def test_explicit_hosts_are_joined_by_env_hosts(self) -> None:
        assert resolve_allowed_hosts(("localhost",), "a.example,b.example") == frozenset(
            {"localhost", "a.example", "b.example"}
        )

    def test_an_explicit_empty_tuple_is_a_guard_that_allows_nothing(self) -> None:
        assert resolve_allowed_hosts((), None) == frozenset()

    @pytest.mark.parametrize("explicit", [None, ("localhost",), ()])
    @pytest.mark.parametrize("star", ["*", " * ", "\t*\n"])
    def test_a_lone_star_is_the_opt_out_in_every_case(
        self, explicit: tuple[str, ...] | None, star: str
    ) -> None:
        assert resolve_allowed_hosts(explicit, star) is None

    def test_env_entries_are_normalised_like_header_values(self) -> None:
        got = resolve_allowed_hosts(
            None, " Trelix.Internal:8765 , [::1]:80 ,, LOCALHOST. ,  , 10.0.0.5 "
        )
        assert got == frozenset({*LOOPBACK, "trelix.internal", "10.0.0.5"})

    def test_junk_entries_are_dropped_not_promoted_and_do_not_disable_the_guard(self) -> None:
        got = resolve_allowed_hosts(None, "evil.com@x,*.example.com,good.example,*")
        assert got == frozenset({*LOOPBACK, "good.example"})
        assert got is not None

    def test_only_junk_in_env_still_means_the_guard_is_on(self) -> None:
        assert resolve_allowed_hosts(None, "*.example.com") == frozenset(LOOPBACK)

    def test_explicit_entries_that_are_junk_are_dropped(self) -> None:
        assert resolve_allowed_hosts(("localhost", "bad host", "*"), None) == frozenset(
            {"localhost"}
        )

    def test_a_dropped_entry_is_logged_without_echoing_a_huge_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            resolve_allowed_hosts(None, "x" * 500 + "@y")
        assert len(caplog.records) == 1
        assert len(caplog.records[0].getMessage()) < 400

    def test_the_dropped_entry_warning_names_the_accepted_form(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An operator who pasted a URL or a wildcard must learn what to write instead."""
        with caplog.at_level(logging.WARNING):
            resolve_allowed_hosts(None, "http://dev.example")
        message = caplog.records[0].getMessage()
        assert "bare hostname" in message
        assert "scheme" in message
        assert "wildcard" in message
        assert len(message) < 400


class TestLoopbackBindAllowedHosts:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST", "::1", "[::1]"])
    def test_canonical_loopback_binds_get_exactly_the_loopback_names(self, host: str) -> None:
        assert loopback_bind_allowed_hosts(host) == LOOPBACK

    def test_a_specific_loopback_address_is_added(self) -> None:
        assert loopback_bind_allowed_hosts("127.0.0.2") == (*LOOPBACK, "127.0.0.2")

    def test_the_whole_127_slash_8_counts_as_loopback(self) -> None:
        assert loopback_bind_allowed_hosts("127.255.255.254") is not None

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "::", "10.0.0.7", "192.168.1.5", "example.internal", "", "bad host"],  # noqa: S104 - the literal IS the subject
    )
    def test_everything_else_is_not_a_loopback_bind(self, host: str) -> None:
        assert loopback_bind_allowed_hosts(host) is None


# ---------------------------------------------------------------------------
# Middleware over the real app
# ---------------------------------------------------------------------------


@pytest.fixture
def guarded_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """The real app with the guard ON and `tmp_path` as the only served root."""
    monkeypatch.setenv("TRELIX_ALLOWED_REPO_ROOTS", str(tmp_path))
    mock_ctx = MagicMock()
    mock_ctx.results = []
    with patch("trelix.api.app.Retriever") as retriever:
        retriever.return_value.retrieve.return_value = mock_ctx
        yield create_app(allowed_hosts=LOOPBACK)


@pytest.fixture
def client(guarded_app: Any) -> TestClient:
    return TestClient(guarded_app, base_url="http://localhost:8765")


def _search(client: TestClient, tmp_path: Path, headers: dict[str, str] | None = None) -> Any:
    return client.get(f"/search?query=auth&repo={tmp_path}", headers=headers)


def _gated_routes(app: Any) -> list[tuple[str, str]]:
    """(method, path) for every route except /health, incl. /docs and /openapi.json."""
    found = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or path == "/health":
            continue
        found.extend((method, path) for method in sorted(methods) if method != "HEAD")
    return found


def test_the_route_table_has_the_routes_we_think_it_has(guarded_app: Any) -> None:
    """Guards the enumeration below from silently covering nothing."""
    paths = {path for _, path in _gated_routes(guarded_app)}
    assert {"/search", "/ask", "/index", "/parse", "/graph/visualize", "/openapi.json"} <= paths
    assert len(_gated_routes(guarded_app)) >= 9


def test_an_evil_host_is_refused_on_every_gated_route(guarded_app: Any) -> None:
    client = TestClient(guarded_app, base_url="http://localhost:8765")
    for method, path in _gated_routes(guarded_app):
        resp = client.request(method, path, headers={"Host": f"{EVIL}:8765"})
        assert resp.status_code == 403, f"{method} {path} answered {resp.status_code}"
        assert EVIL not in resp.text


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "localhost:8765",
        "LOCALHOST:8765",
        "localhost.:8765",
        "127.0.0.1:8765",
        "127.0.0.1",
        "[::1]:8765",
        "[::1]",
    ],
)
def test_allowed_host_variants_pass(client: TestClient, tmp_path: Path, host: str) -> None:
    resp = _search(client, tmp_path, {"Host": host})
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize(
    "host",
    [
        EVIL,
        "localhost.evil.com",
        "127.0.0.1.evil.com",
        "localhost@evil.com",
        "evil.com#localhost",
        "evil.com/localhost",
        "[::1]evil",
        "10.0.0.5:8765",
        "localhost evil",
        "localhost\t",
    ],
)
def test_hostile_or_lookalike_host_values_are_refused(
    client: TestClient, tmp_path: Path, host: str
) -> None:
    resp = _search(client, tmp_path, {"Host": host})
    assert resp.status_code == 403
    assert resp.json() == {"detail": REJECTION_DETAIL}


def test_the_default_testclient_host_is_refused_when_the_guard_is_on(guarded_app: Any) -> None:
    """`testserver` is the TestClient default; a guarded app must not accept it."""
    resp = TestClient(guarded_app).get("/stats")
    assert resp.status_code == 403


@pytest.mark.parametrize(
    "origin",
    ["http://localhost:3000", "https://localhost", "http://127.0.0.1:8765", "http://[::1]:8765"],
)
def test_an_allowed_origin_passes(client: TestClient, tmp_path: Path, origin: str) -> None:
    assert _search(client, tmp_path, {"Origin": origin}).status_code == 200


@pytest.mark.parametrize(
    "origin",
    [
        f"http://{EVIL}",
        f"https://{EVIL}:8765",
        "null",
        "http://localhost.evil.com",
        "http://localhost@evil.com",
        "chrome-extension://abc",
        "localhost",
        "",
    ],
)
def test_a_foreign_or_malformed_origin_is_refused(
    client: TestClient, tmp_path: Path, origin: str
) -> None:
    resp = _search(client, tmp_path, {"Origin": origin})
    assert resp.status_code == 403
    assert resp.json() == {"detail": REJECTION_DETAIL}


@pytest.mark.parametrize("value", ["same-origin", "same-site", "none", "SAME-ORIGIN"])
def test_benign_fetch_metadata_passes(client: TestClient, tmp_path: Path, value: str) -> None:
    assert _search(client, tmp_path, {"Sec-Fetch-Site": value}).status_code == 200


@pytest.mark.parametrize("value", ["cross-site", "Cross-Site", "CROSS-SITE", " cross-site "])
def test_cross_site_fetch_metadata_is_refused_even_with_a_good_host(
    client: TestClient, tmp_path: Path, value: str
) -> None:
    resp = _search(client, tmp_path, {"Sec-Fetch-Site": value})
    assert resp.status_code == 403
    assert resp.json() == {"detail": CROSS_SITE_DETAIL}


def test_each_rejection_text_names_the_env_var_and_the_two_differ() -> None:
    """Adding a hostname cannot fix a cross-site fetch, so that text must not promise it."""
    assert "TRELIX_API_ALLOWED_HOSTS" in REJECTION_DETAIL
    assert "TRELIX_API_ALLOWED_HOSTS" in CROSS_SITE_DETAIL
    assert "TRELIX_API_ALLOWED_HOSTS=*" in CROSS_SITE_DETAIL
    assert "cross-site" in CROSS_SITE_DETAIL
    assert CROSS_SITE_DETAIL != REJECTION_DETAIL


def test_a_request_with_neither_origin_nor_fetch_metadata_is_unaffected(
    client: TestClient, tmp_path: Path
) -> None:
    """curl, Python clients and the trelix clients send neither header."""
    assert _search(client, tmp_path).status_code == 200


def test_the_guard_adds_no_cors_headers(client: TestClient, tmp_path: Path) -> None:
    resp = _search(client, tmp_path, {"Origin": "http://localhost:3000"})
    assert resp.status_code == 200
    assert not any(name.lower().startswith("access-control-") for name in resp.headers)
    pre = client.options(
        "/search",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert not any(name.lower().startswith("access-control-") for name in pre.headers)


# -- raw ASGI: things an HTTP client refuses to send -------------------------


def _scope(
    headers: list[tuple[bytes, bytes]], *, path: str = "/stats", method: str = "GET"
) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8765),
    }


def _drive(app: Any, scope: dict[str, Any]) -> tuple[int, bytes]:
    """Run one ASGI http request to completion; return (status, body)."""
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], body


def test_a_duplicate_host_header_is_refused(guarded_app: Any) -> None:
    status, body = _drive(guarded_app, _scope([(b"host", b"localhost"), (b"host", b"localhost")]))
    assert status == 403
    assert b"localhost" not in body


def test_a_duplicate_host_header_is_refused_even_when_the_second_is_evil(
    guarded_app: Any,
) -> None:
    status, _ = _drive(guarded_app, _scope([(b"host", b"localhost"), (b"host", EVIL.encode())]))
    assert status == 403


def test_a_missing_host_header_is_refused(guarded_app: Any) -> None:
    status, _ = _drive(guarded_app, _scope([(b"accept", b"*/*")]))
    assert status == 403


def test_more_than_one_origin_header_is_refused(guarded_app: Any) -> None:
    headers = [
        (b"host", b"localhost"),
        (b"origin", b"http://localhost"),
        (b"origin", b"http://localhost"),
    ]
    status, _ = _drive(guarded_app, _scope(headers))
    assert status == 403


def test_a_nul_byte_in_host_is_refused(guarded_app: Any) -> None:
    status, _ = _drive(guarded_app, _scope([(b"host", b"local\x00host")]))
    assert status == 403


def test_non_http_scopes_pass_through_untouched() -> None:
    seen: list[str] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        seen.append(scope["type"])

    guard = RequestGuardMiddleware(inner, allowed_hosts=frozenset(LOOPBACK))
    asyncio.run(guard({"type": "lifespan"}, None, None))
    assert seen == ["lifespan"]


def test_a_websocket_with_a_foreign_host_is_closed_before_the_app_sees_it() -> None:
    reached: list[bool] = []
    sent: list[dict[str, Any]] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        reached.append(True)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    guard = RequestGuardMiddleware(inner, allowed_hosts=frozenset(LOOPBACK))
    scope = {"type": "websocket", "path": "/ws", "headers": [(b"host", EVIL.encode())]}
    asyncio.run(guard(scope, None, send))
    assert reached == []
    assert sent and sent[0]["type"] == "websocket.close"


# -- /health exemption --------------------------------------------------------


def test_health_is_exempt_from_every_check(guarded_app: Any) -> None:
    client = TestClient(guarded_app, base_url="http://localhost:8765")
    hostile = {
        "Host": "10.42.0.7:8765",
        "Origin": f"http://{EVIL}",
        "Sec-Fetch-Site": "cross-site",
    }
    assert client.get("/health", headers=hostile).status_code == 200
    # HEAD passes the guard; the router (GET-only route) then answers 405, never 403.
    assert client.head("/health", headers=hostile).status_code == 405


def test_health_is_exempt_even_with_no_host_or_a_duplicate_host(guarded_app: Any) -> None:
    for headers in ([], [(b"host", b"a"), (b"host", b"b")]):
        status, _ = _drive(guarded_app, _scope(headers, path="/health"))
        assert status == 200


@pytest.mark.parametrize("path", ["/health/", "/HEALTH", "//health", "/health;x", "/healthz"])
def test_lookalike_health_paths_are_not_exempt(guarded_app: Any, path: str) -> None:
    status, body = _drive(guarded_app, _scope([(b"host", EVIL.encode())], path=path))
    assert status == 403
    assert EVIL.encode() not in body


def _rooted_scope(path: str, root_path: str, host: str) -> dict[str, Any]:
    scope = _scope([(b"host", host.encode())], path=path)
    scope["root_path"] = root_path
    return scope


def test_health_exemption_follows_root_path(guarded_app: Any) -> None:
    """uvicorn --root-path /x delivers the probe as path=/x/health; the router strips the prefix."""
    status, _ = _drive(guarded_app, _rooted_scope("/x/health", "/x", "10.42.0.7:8765"))
    assert status == 200


@pytest.mark.parametrize(
    ("path", "root_path"),
    [
        ("/x/health/", "/x"),
        ("/x/HEALTH", "/x"),
        ("/xhealth", "/x"),
        ("/y/health", "/x"),
        ("/x", "/x"),
        ("/health", "/health"),
    ],
)
def test_root_path_does_not_widen_the_health_exemption(
    guarded_app: Any, path: str, root_path: str
) -> None:
    status, body = _drive(guarded_app, _rooted_scope(path, root_path, EVIL))
    assert status == 403
    assert EVIL.encode() not in body


@pytest.mark.parametrize(
    ("path", "root_path", "expected"),
    [
        ("/health", "", "/health"),
        ("/x/health", "/x", "/health"),
        ("/x", "/x", ""),
        ("/xhealth", "/x", "/xhealth"),
        ("/y/health", "/x", "/y/health"),
        ("/health", "/h", "/health"),
        ("/x/health", "/x/", "/x/health"),
        ("/health", "/health", ""),
    ],
)
def test_route_path_strips_root_path_only_on_a_whole_segment(
    path: str, root_path: str, expected: str
) -> None:
    """Starlette's rule: strip on a whole-segment prefix, otherwise leave the path alone."""
    assert _route_path({"path": path, "root_path": root_path}) == expected


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "OPTIONS"])
def test_only_get_and_head_are_exempt_on_health(guarded_app: Any, method: str) -> None:
    status, _ = _drive(
        guarded_app, _scope([(b"host", EVIL.encode())], path="/health", method=method)
    )
    assert status == 403


# -- rejection body and log ---------------------------------------------------


def test_the_403_body_tells_the_operator_the_fix_and_never_the_value(client: TestClient) -> None:
    resp = client.get("/stats", headers={"Host": EVIL})
    assert resp.status_code == 403
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == {"detail": REJECTION_DETAIL}
    assert "TRELIX_API_ALLOWED_HOSTS" in REJECTION_DETAIL
    assert "TRELIX_API_ALLOWED_HOSTS=*" in REJECTION_DETAIL
    assert EVIL not in resp.text


def test_each_rejection_logs_exactly_one_truncated_repr_warning(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    hostile = "evil.example\nFAKE LOG LINE level=INFO " + "A" * 500
    with caplog.at_level(logging.WARNING, logger="trelix.api"):
        resp = client.get("/stats", headers={"Host": hostile.replace("\n", " ")})
    assert resp.status_code == 403
    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "\n" not in message
    assert "A" * 200 not in message
    assert "'evil.example" in message  # repr() quoting is present


def test_a_raw_newline_in_a_header_cannot_forge_a_log_line(
    guarded_app: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="trelix.api"):
        status, _ = _drive(guarded_app, _scope([(b"host", b"evil\nlevel=CRITICAL fake")]))
    assert status == 403
    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(records) == 1
    assert "\n" not in records[0].getMessage()
    assert "\\n" in records[0].getMessage()


def test_an_allowed_request_logs_nothing(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="trelix.api"):
        client.get("/health")
        client.get("/stats", headers={"Host": "localhost"})
    assert [r for r in caplog.records if "ejected" in r.getMessage()] == []


# ---------------------------------------------------------------------------
# create_app switch
# ---------------------------------------------------------------------------


def _get_stats_status(app: Any, host: str) -> int:
    return (
        TestClient(app, base_url="http://localhost")
        .get("/stats", headers={"Host": host})
        .status_code
    )


def test_create_app_without_the_parameter_or_env_accepts_any_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Today's behavior: ASGI-factory callers and every existing test are untouched."""
    monkeypatch.delenv("TRELIX_API_ALLOWED_HOSTS", raising=False)
    app = create_app()
    assert _get_stats_status(app, "testserver") != 403
    assert _get_stats_status(app, EVIL) != 403


def test_env_alone_turns_the_guard_on_for_a_factory_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRELIX_API_ALLOWED_HOSTS", "trelix.internal:8765")
    app = create_app()
    assert _get_stats_status(app, "localhost:8765") != 403
    assert _get_stats_status(app, "trelix.internal") != 403
    assert _get_stats_status(app, EVIL) == 403
    assert _get_stats_status(app, "testserver") == 403


def test_env_star_wins_over_an_explicit_allow_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRELIX_API_ALLOWED_HOSTS", "*")
    app = create_app(allowed_hosts=LOOPBACK)
    assert _get_stats_status(app, EVIL) != 403


def test_explicit_hosts_are_joined_by_env_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRELIX_API_ALLOWED_HOSTS", "extra.example")
    app = create_app(allowed_hosts=("localhost",))
    assert _get_stats_status(app, "extra.example") != 403
    assert _get_stats_status(app, "localhost") != 403
    assert _get_stats_status(app, "127.0.0.1") == 403


def test_allowed_hosts_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        create_app(None, LOOPBACK)  # type: ignore[misc]


def test_enabling_the_guard_logs_one_info_line_naming_the_opt_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="trelix.api"):
        create_app(allowed_hosts=LOOPBACK)
    lines = [
        r.getMessage() for r in caplog.records if "TRELIX_API_ALLOWED_HOSTS=*" in r.getMessage()
    ]
    assert len(lines) == 1


def test_a_disabled_guard_logs_no_guard_line(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRELIX_API_ALLOWED_HOSTS", raising=False)
    with caplog.at_level(logging.INFO, logger="trelix.api"):
        create_app()
    assert [r for r in caplog.records if "TRELIX_API_ALLOWED_HOSTS" in r.getMessage()] == []


# ---------------------------------------------------------------------------
# Compatibility with the audit middleware
# ---------------------------------------------------------------------------


def test_the_audit_log_records_a_guard_rejection_as_403(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audit middleware must be OUTSIDE the guard, or a rebinding probe leaves no trace.

    Starlette's `add_middleware` inserts at the FRONT of the list and the stack is built
    from it in order, so the LAST middleware added is the OUTERMOST. The guard is
    therefore registered before the audit middleware in `create_app`.
    """
    from trelix.audit.store import AuditStore

    db_path = tmp_path / "audit.db"
    monkeypatch.setenv("TRELIX_AUDIT_ENABLED", "true")
    monkeypatch.setenv("TRELIX_AUDIT_DB_PATH", str(db_path))

    app = create_app(allowed_hosts=LOOPBACK)
    resp = TestClient(app, base_url="http://localhost").get("/stats", headers={"Host": EVIL})
    assert resp.status_code == 403

    store = AuditStore(db_path)
    try:
        events = list(store.iter_for_export())
    finally:
        store.close()
    assert len(events) == 1
    assert events[0]["status_code"] == 403
    assert events[0]["outcome"] == "denied"
    assert events[0]["resource"] == "/stats"
    for value in events[0].values():
        assert EVIL not in ("" if value is None else str(value))
