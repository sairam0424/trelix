"""The five documentation sentences that list every flag `trelix-mcp` accepts, pinned to the parser.

docs/MCP_GUIDE.md (twice), docs/TROUBLESHOOTING.md, docs/USER_GUIDE.md and
docs/integrations/vscode-plugin.md each tell a reader that `trelix-mcp` "accepts only" a fixed list
of flags, so that nobody hunts for a `--cache-dir` or a log-level option that does not exist. When
`--root PATH` arrived, all five kept naming `--help`, `--version` and `--tools core|full`, and
MCP_GUIDE.md contradicted its own section 8, which documents `--root`. The first pin holds each
sentence to the complete list, in that document's own spelling (code spans, a table cell with an
escaped pipe, a shell comment wrapped onto three lines). The second holds the list to the flags
`cli.py` declares, read as text because this suite's environment has no fastmcp to import the
module with: a new flag fails here until every sentence names it. The Streamable HTTP transport
added seven flags (`--transport` and six `--transport http` only ones) and every sentence gained
them in one change, which is the case the pin exists for.

The remaining pins hold section 19 to what the code does. The logging paragraph says which check
logs what: only the bearer check never logs the credential, while the REST Host/Origin guard logs
the refused value at WARNING (`Rejected http '/mcp': Host header not allowed: 'evil.example'`),
which an operator shipping stderr to a shared sink must know. Uvicorn's access lines reach stdout,
not the stderr stream section 18's wrapper captures; `0.0.0.0` and `::` are refused as a `Host`
only until listed; and the idle timeout has an environment variable but a default that depends on
the `fastmcp` and `mcp` versions, while the session cap is `mcp` 2.2.0's. Of the two 403 bodies,
only the Host/Origin one names `TRELIX_MCP_ALLOWED_HOSTS` and `--allowed-host`; the cross-site one
names neither, and SECURITY.md and the changelog must say so instead of crediting both.

The last pin holds the two documents that explain the pre-approved `search_code`, `get_symbol` and
`blast_radius` grant (SECURITY.md and the plugin README) to the one file `search_code` writes
besides the index: a JSON trace of the query under `.trelix/debug/`, on every call. Both once said
"nothing else is written", and the README kept saying it after SECURITY.md was corrected.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CLI = _ROOT / "packages" / "trelix-mcp" / "src" / "trelix_mcp" / "cli.py"
_SENTENCES = [
    pytest.param(
        "docs/MCP_GUIDE.md",
        "`trelix-mcp` accepts only `--help`, `--version`, `--tools core|full`, `--root PATH` "
        "(section 8) and\n"
        "the Streamable HTTP flags `--transport stdio|http`, `--host`, `--port`, `--path`, "
        "`--allowed-host`,\n"
        "`--stateful` and `--allow-unauthenticated` (section 19)",
        id="mcp-guide-install",
    ),
    pytest.param(
        "docs/MCP_GUIDE.md",
        "`trelix-mcp` accepts only `--help`, `--version`, `--tools core|full`, `--root PATH` and "
        "the HTTP\n"
        "flags `--transport stdio|http`, `--host`, `--port`, `--path`, `--allowed-host`, "
        "`--stateful` and\n"
        "`--allow-unauthenticated` (section 19), and has no",
        id="mcp-guide-logging",
    ),
    pytest.param(
        "docs/TROUBLESHOOTING.md",
        "# Verify the binary is on PATH. trelix-mcp accepts only --help, --version, "
        "--tools core|full,\n"
        "# --root PATH and the HTTP flags --transport stdio|http, --host, --port, --path, "
        "--allowed-host,\n"
        "# --stateful and --allow-unauthenticated",
        id="troubleshooting",
    ),
    pytest.param(
        "docs/USER_GUIDE.md",
        "nothing — `trelix-mcp` accepts only `--help`, `--version`, `--tools core\\|full`, "
        "`--root PATH` and the HTTP flags `--transport stdio\\|http`, `--host`, `--port`, "
        "`--path`, `--allowed-host`, `--stateful` and `--allow-unauthenticated` "
        "(MCP_GUIDE section 19) |",
        id="user-guide",
    ),
    pytest.param(
        "docs/integrations/vscode-plugin.md",
        "`trelix-mcp` accepts only `--help`, `--version`, `--tools core|full`, `--root PATH` and "
        "the HTTP\n"
        "flags `--transport stdio|http`, `--host`, `--port`, `--path`, `--allowed-host`, "
        "`--stateful` and\n"
        "`--allow-unauthenticated`; invoking",
        id="vscode-plugin",
    ),
]
# A declared flag is every `--name` written inside a `parser.add_argument(...)` call, wherever it
# stands (`add_argument("-t", "--transport", ...)` counts); `-h`/`--help` is argparse's own.
_ADD_ARGUMENT_CALL = re.compile(r"add_argument\(([^)]*)")
_LONG_FLAG = re.compile(r'"(--[\w-]+)"')


def _declared_flags(source: str) -> set[str]:
    return {
        flag for call in _ADD_ARGUMENT_CALL.findall(source) for flag in _LONG_FLAG.findall(call)
    }


@pytest.mark.parametrize(("document", "sentence"), _SENTENCES)
def test_each_document_names_every_flag(document: str, sentence: str) -> None:
    assert sentence in (_ROOT / document).read_text(encoding="utf-8")


def test_the_documented_list_is_the_parser_flag_list() -> None:
    """`--version`, `--tools`, `--root`, `--transport`, `--host`, `--port`, `--path`,
    `--allowed-host`, `--stateful` and `--allow-unauthenticated`, and nothing else: an eleventh
    flag must reach all five documents."""
    declared = _declared_flags(_CLI.read_text(encoding="utf-8"))
    assert declared == {
        "--version",
        "--tools",
        "--root",
        "--transport",
        "--host",
        "--port",
        "--path",
        "--allowed-host",
        "--stateful",
        "--allow-unauthenticated",
    }


def test_the_logging_paragraph_scopes_the_no_header_value_claim_to_the_bearer_check() -> None:
    """Only the bearer check never logs what the client sent; the Host/Origin guard logs the
    refused value, so section 19 must not generalise the first rule to both. Whitespace is
    collapsed, so re-wrapping the paragraph does not fail the pin."""
    guide = " ".join((_ROOT / "docs/MCP_GUIDE.md").read_text(encoding="utf-8").split())
    assert (
        "one WARNING per rejected request — the bearer check names only the method and path "
        "(never the credential); the Host/Origin guard names the refused `Host` or `Origin` "
        "value —"
    ) in guide


def test_the_logging_paragraph_sends_uvicorn_access_lines_to_stdout() -> None:
    """`mcp.run` hands uvicorn no `log_config`, and uvicorn's default one writes the access logger
    to stdout and everything else to stderr; the access lines therefore miss the stderr stream
    section 18's wrapper captures, so section 19 must name both streams."""
    guide = (_ROOT / "docs/MCP_GUIDE.md").read_text(encoding="utf-8")
    assert "uvicorn writes each access line to **stdout** by default" in guide
    assert "capture both streams (`>>/tmp/trelix-mcp.log 2>&1`). The rest goes to stderr" in guide


@pytest.mark.parametrize("document", ["docs/MCP_GUIDE.md", "SECURITY.md"])
def test_the_wildcard_binds_are_refused_as_host_only_until_listed(document: str) -> None:
    """Only the default allow-list leaves `0.0.0.0` and `::` out: `--allowed-host 0.0.0.0` (or the
    variable) admits that `Host`, so neither document may say "never"."""
    text = (_ROOT / document).read_text(encoding="utf-8")
    assert "(`0.0.0.0` and `::` are accepted only when listed)" in text


@pytest.mark.parametrize(
    ("document", "sentence"),
    [
        pytest.param(
            "SECURITY.md",
            "Its Host/Origin 403 names `TRELIX_MCP_ALLOWED_HOSTS` and `--allowed-host` (not the "
            "REST API's variable), its cross-site 403 does not name the REST variable either, and "
            "neither suggests `*`.",
            id="security",
        ),
        pytest.param(
            "CHANGELOG.md",
            "cross-site 403 does not name the REST variable either",
            id="changelog",
        ),
    ],
)
def test_only_the_host_origin_403_is_credited_with_the_mcp_allow_list_hint(
    document: str, sentence: str
) -> None:
    """`HOST_REJECTION_DETAIL` names the variable and the flag; `CROSS_SITE_DETAIL` is a fixed text
    that names neither (docs/MCP_GUIDE.md quotes both). A sentence saying that both 403 bodies name
    them over-claims, so an operator who gets the cross-site refusal would look for a hint that is
    not there. Whitespace is collapsed, so re-wrapping the paragraph does not fail the pin."""
    text = " ".join((_ROOT / document).read_text(encoding="utf-8").split())
    assert sentence in text


def test_the_sessions_paragraph_scopes_the_idle_timeout_to_the_fastmcp_version() -> None:
    """No flag or `mcp.run` argument sets the idle timeout, but FastMCP's
    `FASTMCP_HTTP_SESSION_IDLE_TIMEOUT` reaches the SDK's session manager. What it defaults to
    depends on the `fastmcp` and `mcp` the declared ranges (`fastmcp>=4.0.10,<5`,
    `mcp>=2.0.0,<3.0`) resolve: 4.0.x never expires an idle session, 4.1.x falls back to the SDK's
    default, which is 1800 s from `mcp` 2.2.0 and none before it, and only 4.1.x parses `none`
    (4.0.x fails to import on it). The 10,000-session cap is `mcp` 2.2.0's, absent at the declared
    2.0.0 floor, and below 2.2.0 a set variable stops the default stateless server from starting
    (the SDK's session manager raises `RuntimeError`; run on `mcp` 2.0.0 with fastmcp 4.0.10, and
    read in the 2.1.1 and 2.2.0 sources), so the guide says to set it with `--stateful`.
    Whitespace is collapsed, so re-wrapping the paragraph does not fail the pin."""
    guide = " ".join((_ROOT / "docs/MCP_GUIDE.md").read_text(encoding="utf-8").split())
    variable = "`FASTMCP_HTTP_SESSION_IDLE_TIMEOUT=<seconds>`"
    assert (
        "`fastmcp` 4.0.x never ends an idle one and 4.1.x ends it after 1800 s with `mcp` 2.2.0 "
        "or newer, the `mcp` SDK's default from that release (earlier `mcp` releases have no "
        "such default; recent `fastmcp` 4.1.x releases require `mcp` 2.2.0 or newer); "
        f"{variable} in the environment sets that timeout "
        "(`none` disables it on 4.1.x; 4.0.x fails to start on `none`), "
        "and no flag or `mcp.run` argument reaches it."
    ) in guide
    assert (
        "the SDK caps them at 10,000 from `mcp` 2.2.0 "
        "(earlier releases, the 2.0.0 floor included, have no cap)"
    ) in guide
    assert (
        "Set it only together with `--stateful`: on `mcp` before 2.2.0 (2.0.x and 2.1.x) the "
        "default stateless server will not start with it "
        "(`session_idle_timeout is not supported in stateless mode`)."
    ) in guide


def test_the_logging_paragraph_brackets_an_ipv6_bind_in_the_start_line_and_the_warning() -> None:
    """`cli._start_http` and the no-token WARNING both log the bind through `url_host`, so the guide
    must not say the bracketing is the start line's alone."""
    guide = " ".join((_ROOT / "docs/MCP_GUIDE.md").read_text(encoding="utf-8").split())
    assert (
        "(an IPv6 bind is bracketed, in the `--allow-unauthenticated` WARNING too: "
        "`[::1]:8766/mcp`)"
    ) in guide


@pytest.mark.parametrize(
    ("document", "sentences", "stale"),
    [
        pytest.param(
            "SECURITY.md",
            (
                "`search_code` also writes one JSON trace of the query per call "
                "(`Retriever._flush_trace`) into a `debug/` directory beside the index",
                "`get_symbol` and `blast_radius` write no such file.",
                "grows by at most one file per `search_code` call",
            ),
            ("and answers empty; nothing else is written",),
            id="security",
        ),
        pytest.param(
            "plugins/trelix/README.md",
            (
                "`search_code` also writes one JSON trace of the query per call, named "
                "`.trelix/debug/<timestamp>_<query>.json`",
                "`get_symbol` and `blast_radius` write nothing else.",
                "every `search_code` call also writes a trace file (see above)",
            ),
            (
                "`get_symbol` with `null`); nothing else is written",
                "(see above); nothing else is written",
            ),
            id="plugin-readme",
        ),
    ],
)
def test_the_pre_approved_read_tools_are_not_credited_with_writing_only_the_index(
    document: str, sentences: tuple[str, ...], stale: tuple[str, ...]
) -> None:
    """`Retriever._flush_trace` writes `<index dir>/debug/<timestamp>_<query>.json` for every
    `search_code` call, with no switch, cap or rotation (`test_tool_readonly.py` measures the
    file); `get_symbol` and `blast_radius` write none. A sentence saying that nothing but the
    empty index is written contradicts the code, so neither document may keep it. Whitespace is
    collapsed, so re-wrapping the paragraph does not fail the pin."""
    text = " ".join((_ROOT / document).read_text(encoding="utf-8").split())
    for sentence in sentences:
        assert sentence in text
    for sentence in stale:
        assert sentence not in text
