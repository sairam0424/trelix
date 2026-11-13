"""A request that holds text UTF-8 cannot encode is answered, not a 500.

A JSON body can carry a lone surrogate (``"\\ud800"``): ``json.loads`` accepts it, a JavaScript
string can hold half of a pair, and a Python client that reads a file with
``errors="surrogateescape"`` sends the undecodable byte 0xff as ``"\\udcff"``. UTF-8 cannot encode
it, and three things downstream tried:

* ``POST /parse`` with inline ``content``: most extractors call ``source.encode("utf-8")``
  (UnicodeEncodeError), and the others (shell, SQL, Makefile, Dockerfile, protobuf, Markdown)
  copy text out of the source into a symbol name or signature that the response writer cannot
  encode (PydanticSerializationError). Both ended in ``500``.
* A ``422`` echoes each error's ``input``, which for a model error is the whole body, and
  Starlette's ``JSONResponse`` writes UTF-8. Any body that fails validation and holds a surrogate
  anywhere, an unknown key included, ended in ``500`` on ``POST /parse`` and ``POST /index``.
* ``X-Trelix-Api-Key: é`` reached ``hmac.compare_digest`` on two ``str`` and raised ``TypeError``
  (the same 500, from a header, with a token configured).

The contract asserted here: inline ``content`` parses exactly like the same text with U+FFFD in
place of each surrogate (what the disk-backed branch and the indexer make of bytes UTF-8 cannot
decode); a ``422`` writes a surrogate as its JSON escape and everything else as before; a
header that is not ASCII is a ``401``, and the token check still lets in exactly the callers it let
in before. A surrogate in a PATH keeps its own answer (400, see test_api_unresolvable_paths.py).
Every expected value is a literal.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from trelix.api.app import create_app

# One snippet per extractor, "@@" standing for the text under test, placed where the extractor
# copies it into a symbol name or signature (the first line for the line-window ones, the @page
# route for Razor). With a surrogate in its place most answered 500. JSON, YAML and TOML answered
# 200 but read another text: a "?" in JSON and TOML, a parse error and no symbols for YAML.
_SNIPPETS: dict[str, str] = {
    "a.py": 'def f():\n    return "@@"\n',
    "a.js": 'function f() { return "@@"; }\n',
    "a.ts": 'function f(): string { return "@@"; }\n',
    "a.tsx": 'function f(): string { return "@@"; }\n',
    "a.go": 'package p\n\nfunc F() string { return "@@" }\n',
    "a.rs": 'fn f() -> &\'static str { "@@" }\n',
    "a.java": 'class C { String f() { return "@@"; } }\n',
    "a.kt": 'fun f(): String = "@@"\n',
    "a.cpp": 'const char* f() { return "@@"; }\n',
    "a.c": 'const char* f(void) { return "@@"; }\n',
    "a.cs": 'class C { string F() => "@@"; }\n',
    "a.csproj": "<Project><PropertyGroup><X>@@</X></PropertyGroup></Project>\n",
    "a.html": "<html><body><p>@@</p></body></html>\n",
    "a.css": '.a::after { content: "@@"; }\n',
    "a.rb": 'def f\n  "@@"\nend\n',
    "a.sh": 'echo "@@"\nf() { :; }\n',
    "Dockerfile": "# @@\nFROM scratch\nRUN true\n",
    "Makefile": "# @@\nall:\n\ttrue\n",
    "a.sql": "select '@@';\n",
    "a.proto": "// @@\nmessage M {}\n",
    "a.md": "# @@\n\ntext\n",
    "a.razor": '@page "/@@"\n@code { string F() => "@@"; }\n',
    "a.cshtml": '@page "/@@"\n@{ var x = "@@"; }\n',
    "a.json": '{"k": "@@"}\n',
    "a.yaml": "k: '@@'\n",
    "a.toml": 'k = "@@"\n',
}

_SURROGATES = [
    pytest.param("\ud800", id="high"),
    pytest.param("\udfff", id="low"),
    pytest.param("\udcff", id="byte-0xff-as-surrogateescape"),
]


@pytest.fixture
def world(tmp_path: Path) -> tuple[Path, TestClient]:
    root = tmp_path / "root"
    root.mkdir()
    return root, TestClient(create_app(served_root=root), raise_server_exceptions=False)


def _post(client: TestClient, route: str, body: object, **headers: bytes | str) -> httpx.Response:
    """``json.dumps`` writes a surrogate as the escape ``\\ud800``, which is what a client sends;
    the ``json=`` helper of the test client writes UTF-8 instead and cannot."""
    return client.post(
        route,
        content=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
    )


def _parse(client: TestClient, root: Path, name: str, text: str) -> httpx.Response:
    return _post(
        client,
        "/parse",
        {"repo_path": str(root), "content": text, "file_name": name},
    )


class TestInlineContentWithALoneSurrogate:
    @pytest.mark.parametrize("surrogate", _SURROGATES)
    @pytest.mark.parametrize("name", list(_SNIPPETS))
    def test_parses_like_the_same_text_with_a_replacement_character(
        self, world: tuple[Path, TestClient], name: str, surrogate: str
    ) -> None:
        root, client = world
        snippet = _SNIPPETS[name]

        got = _parse(client, root, name, snippet.replace("@@", f"a{surrogate}b"))
        want = _parse(client, root, name, snippet.replace("@@", "a\ufffdb"))

        assert want.status_code == 200, want.text
        assert got.status_code == 200, got.text
        assert got.json() == want.json()

    def test_a_python_function_next_to_the_surrogate_is_still_found(
        self, world: tuple[Path, TestClient]
    ) -> None:
        root, client = world

        resp = _parse(client, root, "a.py", '\n\ndef g():\n    return "a\ud800b"\n')

        assert resp.status_code == 200, resp.text
        symbols = resp.json()["symbols"]
        assert [(s["name"], s["line_start"], s["line_end"]) for s in symbols] == [("g", 3, 4)]
        assert resp.json()["parse_errors"] == 0

    def test_the_text_reaches_the_response_as_a_replacement_character(
        self, world: tuple[Path, TestClient]
    ) -> None:
        root, client = world

        shell = _parse(client, root, "a.sh", 'echo "a\ud800b"\nf() { :; }\n')
        markdown = _parse(client, root, "a.md", "# T\ud800T\n\ntext\n")

        assert shell.status_code == 200, shell.text
        assert [s["signature"] for s in shell.json()["symbols"]] == ['echo "a\ufffdb"']
        assert markdown.status_code == 200, markdown.text
        assert [s["name"] for s in markdown.json()["symbols"]] == ["T\ufffdT"]

    def test_one_surrogate_is_one_replacement_character(
        self, world: tuple[Path, TestClient]
    ) -> None:
        """A reversed pair is two lone surrogates, so two characters, and nothing else moves."""
        root, client = world

        resp = _parse(client, root, "a.sh", 'echo "a\ude00\ud83db"\nf() { :; }\n')

        assert resp.status_code == 200, resp.text
        assert [s["signature"] for s in resp.json()["symbols"]] == ['echo "a\ufffd\ufffdb"']

    def test_a_valid_pair_is_kept(self, world: tuple[Path, TestClient]) -> None:
        """``"\\ud83d\\ude00"`` is one character (U+1F600) once ``json.loads`` has joined it."""
        root, client = world

        resp = _parse(client, root, "a.sh", 'echo "\U0001f600"\nf() { :; }\n')

        assert resp.status_code == 200, resp.text
        assert [s["signature"] for s in resp.json()["symbols"]] == ['echo "\U0001f600"']

    def test_a_surrogate_in_the_file_name_is_not_a_problem(
        self, world: tuple[Path, TestClient]
    ) -> None:
        """``file_name`` only picks the language and is echoed through ``repr``."""
        root, client = world

        resp = _parse(client, root, "\ud800.unknownext", "x = 1\n")

        assert resp.status_code == 200, resp.text
        assert resp.json()["symbols"] == []
        assert "'\\ud800.unknownext'" in resp.json()["note"]


class TestAValidationErrorThatEchoesALoneSurrogate:
    """FastAPI's 422 carries ``input``; a model error carries the whole body."""

    def test_parse_without_file_name_is_422_with_the_surrogate_escaped(
        self, world: tuple[Path, TestClient]
    ) -> None:
        root, client = world

        resp = _post(client, "/parse", {"repo_path": str(root), "content": "x\ud800"})

        assert resp.status_code == 422, resp.text
        assert "\\ud800" in resp.text
        error = resp.json()["detail"][0]
        assert error["type"] == "value_error"
        assert error["msg"] == "Value error, file_name is required when content is provided"
        assert error["input"]["content"] == "x\ud800"

    def test_parse_with_both_sources_is_422_with_the_surrogate_escaped(
        self, world: tuple[Path, TestClient]
    ) -> None:
        root, client = world

        resp = _post(
            client,
            "/parse",
            {
                "repo_path": str(root),
                "file_path": "a.py",
                "content": "x\udcff",
                "file_name": "a.py",
            },
        )

        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"][0]["input"]["content"] == "x\udcff"

    def test_an_unknown_key_cannot_turn_a_422_into_a_500(
        self, world: tuple[Path, TestClient]
    ) -> None:
        _, client = world

        resp = _post(client, "/index", {"zzz": "\ud800"})

        assert resp.status_code == 422, resp.text
        assert resp.json()["detail"] == [
            {"type": "missing", "loc": ["body", "repo_path"], "msg": "Field required"}
            | {"input": {"zzz": "\ud800"}}
        ]

    def test_a_json_string_body_is_422_too(self, world: tuple[Path, TestClient]) -> None:
        _, client = world

        resp = client.post(
            "/index", content=b'"\\ud800"', headers={"content-type": "application/json"}
        )

        assert resp.status_code == 422, resp.text

    def test_every_other_character_is_written_as_before(
        self, world: tuple[Path, TestClient]
    ) -> None:
        """Only a surrogate becomes an escape: text outside ASCII is still plain UTF-8, and the
        body is the compact JSON FastAPI's own handler writes."""
        _, client = world

        resp = client.post(
            "/index",
            content=json.dumps({"zzz": "naïve"}, ensure_ascii=False).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

        assert resp.status_code == 422, resp.text
        assert resp.headers["content-type"] == "application/json"
        assert (
            resp.content
            == (
                '{"detail":[{"type":"missing","loc":["body","repo_path"],"msg":"Field required",'
                '"input":{"zzz":"naïve"}}]}'
            ).encode()
        )

    def test_a_surrogate_in_a_path_keeps_its_own_400(self, world: tuple[Path, TestClient]) -> None:
        _, client = world

        resp = _post(client, "/index", {"repo_path": "\ud800"})

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": "repo_path is not a valid path"}


def _raw_post(
    app: object, headers: list[tuple[bytes, bytes]], body: dict[str, str]
) -> httpx.Response:
    """``POST /parse`` with the header bytes exactly as given.

    Starlette's ``TestClient`` decodes every header value and sends it on re-encoded as UTF-8, so
    a byte that is not UTF-8 (a Latin-1 "é", 0xff) never reaches the app through it. An ASGI
    transport passes the raw bytes, which is what a socket delivers.
    """

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/parse",
                content=json.dumps(body).encode("utf-8"),
                headers=[(b"content-type", b"application/json"), *headers],
            )

    return asyncio.run(send())


def _app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, token: str | None
) -> tuple[object, dict[str, str]]:
    """An app with ``token`` as its API token (none when ``None``), and a valid parse body."""
    root = tmp_path / "root"
    root.mkdir()
    if token is not None:
        monkeypatch.setenv("TRELIX_API_AUTH_TOKEN", token)
    body = {"repo_path": str(root), "content": "def f():\n    return 1\n", "file_name": "a.py"}
    return create_app(served_root=root), body


_KEY = b"x-trelix-api-key"
_AUTHORIZATION = b"authorization"
_UNAUTHORIZED = {"detail": "Invalid or missing API key"}


@pytest.fixture
def secured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[object, dict[str, str]]:
    return _app(tmp_path, monkeypatch, "test-k")


class TestAHeaderThatIsNotAscii:
    """With a token configured, ``authenticate`` compared two ``str`` with ``hmac.compare_digest``,
    which raises ``TypeError`` when either holds a character outside ASCII."""

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            pytest.param(_KEY, b"\xe9", id="api-key-latin-1-e-acute"),
            pytest.param(_KEY, b"test-k\xff", id="api-key-token-plus-0xff"),
            pytest.param(_KEY, b"\xc3\xa9", id="api-key-e-acute-as-utf-8"),
            pytest.param(_AUTHORIZATION, b"Bearer \xe9", id="bearer-latin-1-e-acute"),
            pytest.param(_AUTHORIZATION, b"Bearer test-k\xff", id="bearer-token-plus-0xff"),
        ],
    )
    def test_is_401_not_500(
        self, secured: tuple[object, dict[str, str]], name: bytes, value: bytes
    ) -> None:
        app, body = secured

        resp = _raw_post(app, [(name, value)], body)

        assert resp.status_code == 401, resp.text
        assert resp.json() == _UNAUTHORIZED

    def test_the_answer_is_the_one_a_wrong_ascii_token_gets(
        self, secured: tuple[object, dict[str, str]]
    ) -> None:
        app, body = secured

        wrong = _raw_post(app, [(_KEY, b"test-j")], body)
        non_ascii = _raw_post(app, [(_KEY, b"test-\xe9")], body)

        assert wrong.status_code == non_ascii.status_code == 401
        assert wrong.content == non_ascii.content
        assert dict(wrong.headers) == dict(non_ascii.headers)

    def test_the_comparison_is_constant_time_over_bytes(
        self, secured: tuple[object, dict[str, str]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both sides reach ``hmac.compare_digest`` as ``bytes``: the wire bytes of the header and
        the token as UTF-8. (The real function still answers; this only records its arguments.)"""
        app, body = secured
        calls: list[tuple[bytes, bytes]] = []
        real = hmac.compare_digest

        def spy(a: bytes, b: bytes) -> bool:
            calls.append((a, b))
            return real(a, b)

        monkeypatch.setattr(hmac, "compare_digest", spy)

        resp = _raw_post(app, [(_KEY, b"test-\xe9")], body)

        assert resp.status_code == 401, resp.text
        assert calls == [(b"test-\xe9", b"test-k")]


class TestTheTokenCheckLetsInExactlyTheCallersItDidBefore:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            pytest.param(_KEY, b"test-k", id="api-key"),
            pytest.param(_AUTHORIZATION, b"Bearer test-k", id="bearer"),
            pytest.param(_AUTHORIZATION, b"bearer test-k", id="bearer-lowercase-scheme"),
        ],
    )
    def test_the_right_token_authenticates(
        self, secured: tuple[object, dict[str, str]], name: bytes, value: bytes
    ) -> None:
        app, body = secured

        resp = _raw_post(app, [(name, value)], body)

        assert resp.status_code == 200, resp.text
        assert [s["name"] for s in resp.json()["symbols"]] == ["f"]

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            pytest.param(_KEY, b"test-j", id="api-key-wrong"),
            pytest.param(_KEY, b"test-", id="api-key-prefix-of-the-token"),
            pytest.param(_KEY, b"test-kk", id="api-key-token-plus-one"),
            pytest.param(_KEY, b"", id="api-key-empty"),
            pytest.param(_AUTHORIZATION, b"Bearer test-j", id="bearer-wrong"),
            pytest.param(_AUTHORIZATION, b"Bearer ", id="bearer-empty"),
            pytest.param(_AUTHORIZATION, b"Basic test-k", id="other-scheme-right-token"),
        ],
    )
    def test_anything_else_is_401(
        self, secured: tuple[object, dict[str, str]], name: bytes, value: bytes
    ) -> None:
        app, body = secured

        resp = _raw_post(app, [(name, value)], body)

        assert resp.status_code == 401, resp.text
        assert resp.json() == _UNAUTHORIZED

    def test_a_request_with_no_credential_is_401(
        self, secured: tuple[object, dict[str, str]]
    ) -> None:
        app, body = secured

        resp = _raw_post(app, [], body)

        assert resp.status_code == 401, resp.text
        assert resp.json() == _UNAUTHORIZED

    def test_with_no_token_configured_every_header_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Open stays open: nothing is compared, so a header that is not ASCII is not a problem."""
        app, body = _app(tmp_path, monkeypatch, None)

        statuses = [
            _raw_post(app, headers, body).status_code
            for headers in (
                [],
                [(_KEY, b"\xe9")],
                [(_KEY, b"")],
                [(_AUTHORIZATION, b"Bearer \xe9")],
            )
        ]

        assert statuses == [200, 200, 200, 200]

    def test_a_blank_token_is_still_no_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, body = _app(tmp_path, monkeypatch, "  ")

        resp = _raw_post(app, [(_KEY, b"")], body)

        assert resp.status_code == 200, resp.text


class TestATokenThatIsNotAscii:
    """The header is compared by its wire bytes with the token as UTF-8. Before, a token with one
    such character made every authenticated request a 500."""

    def test_its_utf8_bytes_authenticate_and_the_ascii_spelling_does_not(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, body = _app(tmp_path, monkeypatch, "tökén")

        right = _raw_post(app, [(_KEY, "tökén".encode())], body)
        bearer = _raw_post(app, [(_AUTHORIZATION, b"Bearer " + "tökén".encode())], body)
        ascii_only = _raw_post(app, [(_KEY, b"token")], body)

        assert right.status_code == 200, right.text
        assert bearer.status_code == 200, bearer.text
        assert ascii_only.status_code == 401, ascii_only.text

    def test_the_same_text_as_latin_1_bytes_is_another_secret(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app, body = _app(tmp_path, monkeypatch, "tökén")

        resp = _raw_post(app, [(_KEY, "tökén".encode("latin-1"))], body)

        assert resp.status_code == 401, resp.text

    @pytest.mark.skipif(sys.platform == "win32", reason="the environment holds text, not bytes")
    def test_a_token_with_a_byte_that_is_not_utf8_matches_that_byte(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An environment variable holding the byte 0xff reaches Python as ``"\\udcff"``
        (``surrogateescape``); the client that sends the byte 0xff must be let in."""
        app, body = _app(tmp_path, monkeypatch, "tok\udcff")

        right = _raw_post(app, [(_KEY, b"tok\xff")], body)
        without = _raw_post(app, [(_KEY, b"tok")], body)
        as_utf8 = _raw_post(app, [(_KEY, "tok\xff".encode())], body)

        assert right.status_code == 200, right.text
        assert without.status_code == 401, without.text
        assert as_utf8.status_code == 401, as_utf8.text
