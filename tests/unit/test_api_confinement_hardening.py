"""REST follow-ups to the allow-list: ``~``, unusable paths, the ``.html`` limit and the fixed
``/ask`` error frame.

``test_api_containment.py`` proves a request cannot NAME a path outside the allowed roots. These
tests cover what the allow-list does not see:

* ``confine_repo`` expands a leading ``~`` before it compares, so a request for ``repo="~"`` is
  approved when HOME is inside a root. The routes then re-derived the path from the raw string,
  where ``~`` is a literal directory name in the server's working directory. The guard and the
  route must agree on one path.
* A NUL byte, an empty ``file_path`` and a directory ``file_path`` answered 500.
* ``GET /graph/visualize?output=`` leaned on pyvis's ``assert`` for the ``.html`` limit; under
  ``python -O`` the assert is gone and ``<repo>/.trelix/index.db`` was overwritten with HTML.
* ``GET /ask`` streamed ``str(exc)`` to the caller. It now streams one fixed line, for a failure
  inside the model call too (``Synthesizer.stream()`` caught that one itself and yielded the
  exception text as an ordinary token).

``test_api_symlinked_store.py`` covers the symlinked ``.trelix`` store. Every expected value here
is a literal.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tests.fixtures.api_confinement import pyvis_without_asserts, tree
from tests.fixtures.indexed import mark_indexed
from trelix.api.app import create_app
from trelix.core.config import EmbedderConfig, IndexConfig, LLMConfig, RetrievalConfig
from trelix.core.models import RetrievedContext
from trelix.llm.providers.openai_backend import OpenAIBackend
from trelix.retrieval.synthesizer import Synthesizer

ROOT_SYMBOL = "symbol_from_the_real_home"
TILDE_SYMBOL = "symbol_from_the_literal_tilde_dir"


# ---------------------------------------------------------------------------
# '~': one path for the guard and for the route
# ---------------------------------------------------------------------------


class _TildeWorld(NamedTuple):
    root: Path  # the allowed root, and HOME
    literal: Path  # a directory literally named "~" in the server's cwd (outside the root)
    client: TestClient


@pytest.fixture
def tilde_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _TildeWorld:
    root = tmp_path / "root"
    root.mkdir()
    (root / "x.py").write_text(f"def {ROOT_SYMBOL}():\n    return 1\n", encoding="utf-8")
    cwd = tmp_path / "cwd"
    literal = cwd / "~"
    literal.mkdir(parents=True)
    (literal / "x.py").write_text(f"def {TILDE_SYMBOL}():\n    return 1\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(root))
    monkeypatch.chdir(cwd)
    client = TestClient(create_app(served_root=root), raise_server_exceptions=False)
    return _TildeWorld(root.resolve(), literal.resolve(), client)


class TestTildeIsOnePath:
    def test_index_config_expands_a_leading_tilde(self, tilde_world: _TildeWorld) -> None:
        config = IndexConfig(repo_path="~")

        assert config.repo_path == str(tilde_world.root)

    def test_index_config_reports_an_unknown_user_as_a_validation_error(self) -> None:
        """``~user`` for a user that does not exist makes ``expanduser`` raise RuntimeError."""
        with pytest.raises(ValidationError, match="repo_path does not exist: ~no_such_user_zz"):
            IndexConfig(repo_path="~no_such_user_zz")

    def test_parse_reads_home_not_the_literal_tilde_directory(
        self, tilde_world: _TildeWorld
    ) -> None:
        resp = tilde_world.client.post("/parse", json={"repo_path": "~", "file_path": "x.py"})

        assert resp.status_code == 200, resp.text
        assert [s["name"] for s in resp.json()["symbols"]] == [ROOT_SYMBOL]
        assert TILDE_SYMBOL not in resp.text

    def test_a_read_route_answers_for_home_not_the_literal_tilde_directory(
        self, tilde_world: _TildeWorld
    ) -> None:
        mark_indexed(tilde_world.literal)
        before = tree(tilde_world.literal)

        resp = tilde_world.client.get("/stats", params={"repo": "~"})

        assert resp.status_code == 400, resp.text
        assert resp.json() == {
            "detail": (
                f"No index found at {tilde_world.root}/.trelix/index.db. "
                f"Run trelix index {tilde_world.root} first."
            )
        }
        assert tree(tilde_world.literal) == before

    def test_a_read_route_reads_the_index_under_home(self, tilde_world: _TildeWorld) -> None:
        mark_indexed(tilde_world.root)

        resp = tilde_world.client.get("/stats", params={"repo": "~"})

        assert resp.status_code == 200, resp.text
        assert resp.json() == {"files": 0, "symbols": 0, "chunks": 0}

    def test_graph_visualize_writes_under_home_not_the_literal_tilde_directory(
        self, tilde_world: _TildeWorld, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pyvis_without_asserts(monkeypatch)
        mark_indexed(tilde_world.root)
        mark_indexed(tilde_world.literal)

        resp = tilde_world.client.get("/graph/visualize", params={"repo": "~"})

        assert resp.status_code == 200, resp.text
        assert resp.json()["path"] == str(tilde_world.root / ".trelix" / "graph.html")
        assert not (tilde_world.literal / ".trelix" / "graph.html").exists()


# ---------------------------------------------------------------------------
# /parse and the guard answer 400, not 500, for a path that cannot name a file
# ---------------------------------------------------------------------------


class TestUnusablePathsAre400:
    @pytest.fixture
    def parse_client(self, tmp_path: Path) -> tuple[Path, TestClient]:
        root = tmp_path / "root"
        (root / "adir").mkdir(parents=True)
        client = TestClient(create_app(served_root=root), raise_server_exceptions=False)
        return root, client

    @pytest.mark.parametrize(
        ("file_path", "detail"),
        [
            pytest.param("a\x00b.py", "file_path is not a valid path", id="nul-byte"),
            pytest.param("", "file_path is not a file", id="empty"),
            pytest.param("adir", "file_path is not a file", id="directory"),
        ],
    )
    def test_parse_file_path(
        self, parse_client: tuple[Path, TestClient], file_path: str, detail: str
    ) -> None:
        root, client = parse_client

        resp = client.post("/parse", json={"repo_path": str(root), "file_path": file_path})

        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"].startswith(detail), resp.text

    def test_a_missing_file_keeps_its_own_message(
        self, parse_client: tuple[Path, TestClient]
    ) -> None:
        root, client = parse_client

        resp = client.post("/parse", json={"repo_path": str(root), "file_path": "nope.py"})

        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"].startswith("file_path not found: ")

    def test_a_nul_byte_in_a_repo_field_is_400_on_every_shape(
        self, parse_client: tuple[Path, TestClient]
    ) -> None:
        _, client = parse_client
        nul = "a\x00b"

        assert client.get("/stats", params={"repo": nul}).status_code == 400
        assert client.post("/index", json={"repo_path": nul}).status_code == 400
        resp = client.post("/parse", json={"repo_path": nul, "file_path": "x.py"})
        assert resp.status_code == 400
        assert resp.json() == {"detail": "repo_path is not a valid path"}

    def test_a_tilde_user_that_does_not_exist_is_400_on_every_shape(
        self, parse_client: tuple[Path, TestClient]
    ) -> None:
        """``expanduser`` raises RuntimeError (not ValueError) for ``~user`` with no such user."""
        _, client = parse_client
        unknown = "~no_such_user_zz"

        assert client.get("/stats", params={"repo": unknown}).status_code == 400
        assert client.post("/index", json={"repo_path": unknown}).status_code == 400
        resp = client.post("/parse", json={"repo_path": unknown, "file_path": "x.py"})
        assert resp.status_code == 400
        assert resp.json() == {"detail": "repo_path is not a valid path"}


# ---------------------------------------------------------------------------
# /graph/visualize: an explicit .html limit
# ---------------------------------------------------------------------------


class TestVisualizeOutputMustBeHtml:
    @pytest.fixture
    def viz(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, TestClient]:
        pyvis_without_asserts(monkeypatch)
        root = tmp_path / "root"
        repo = mark_indexed(root / "repo")
        client = TestClient(create_app(served_root=root), raise_server_exceptions=False)
        return repo, client

    def test_the_index_database_cannot_be_overwritten(self, viz: tuple[Path, TestClient]) -> None:
        repo, client = viz
        index = repo / ".trelix" / "index.db"
        before = index.read_bytes()

        resp = client.get("/graph/visualize", params={"repo": str(repo), "output": str(index)})

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": "output must be an .html file"}
        assert index.read_bytes() == before

    @pytest.mark.parametrize("name", ["graph.txt", "graph", "graph.html.bak"])
    def test_another_suffix_creates_nothing(self, viz: tuple[Path, TestClient], name: str) -> None:
        repo, client = viz
        target = repo / ".trelix" / name

        resp = client.get("/graph/visualize", params={"repo": str(repo), "output": str(target)})

        assert resp.status_code == 400, resp.text
        assert not target.exists()

    def test_an_html_output_inside_trelix_is_still_written(
        self, viz: tuple[Path, TestClient]
    ) -> None:
        repo, client = viz
        target = repo / ".trelix" / "custom.html"

        resp = client.get("/graph/visualize", params={"repo": str(repo), "output": str(target)})

        assert resp.status_code == 200, resp.text
        assert resp.json()["path"] == str(target.resolve())
        assert target.is_file()

    @pytest.mark.parametrize(
        "shape",
        [pytest.param("relative", id="relative"), pytest.param("absolute", id="absolute")],
    )
    def test_a_nul_byte_in_output_is_400_not_500(
        self, viz: tuple[Path, TestClient], shape: str
    ) -> None:
        """A relative ``output`` skips the guard's per-field check (the route resolves it
        against the working directory and re-checks it), so the route has to answer a NUL byte
        itself."""
        repo, client = viz
        output = "a\x00b.html" if shape == "relative" else f"{repo}/.trelix/a\x00b.html"

        resp = client.get("/graph/visualize", params={"repo": str(repo), "output": output})

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": "output is not a valid path"}

    def test_the_location_check_still_answers_first(self, viz: tuple[Path, TestClient]) -> None:
        repo, client = viz
        target = repo / "elsewhere.txt"

        resp = client.get("/graph/visualize", params={"repo": str(repo), "output": str(target)})

        assert resp.status_code == 400, resp.text
        assert resp.json() == {"detail": "output path must be inside <repo>/.trelix/"}


# ---------------------------------------------------------------------------
# /ask: a fixed error frame, the detail in the server log
# ---------------------------------------------------------------------------


class TestAskStreamsAFixedErrorFrame:
    @pytest.fixture
    def ask_client(self, tmp_path: Path) -> tuple[Path, TestClient]:
        root = tmp_path / "root"
        repo = mark_indexed(root / "repo")
        return repo, TestClient(create_app(served_root=root), raise_server_exceptions=False)

    def test_a_retrieval_failure_streams_the_fixed_frame_and_logs_the_exception(
        self, ask_client: tuple[Path, TestClient], caplog: pytest.LogCaptureFixture
    ) -> None:
        repo, client = ask_client
        caplog.set_level(logging.ERROR, logger="trelix.api")

        with patch("trelix.api.app.Retriever") as retriever:
            retriever.return_value.retrieve.side_effect = RuntimeError(
                "provider failed at host api.internal.example\nsecond line"
            )
            resp = client.get("/ask", params={"repo": str(repo), "query": "q"})

        assert resp.status_code == 200
        assert resp.text == "data: [ERROR: synthesis failed]\n\n"
        logged = [r for r in caplog.records if r.name == "trelix.api" and r.exc_info]
        assert len(logged) == 1
        assert logged[0].exc_info is not None
        assert (
            str(logged[0].exc_info[1])
            == "provider failed at host api.internal.example\nsecond line"
        )

    def test_a_failure_after_some_tokens_ends_with_the_same_frame(
        self, ask_client: tuple[Path, TestClient]
    ) -> None:
        repo, client = ask_client

        def _stream(*_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
            yield "tok"
            raise ValueError("detail the caller must not see")

        with (
            patch("trelix.api.app.Retriever"),
            patch("trelix.retrieval.synthesizer.Synthesizer") as synthesizer,
        ):
            synthesizer.return_value.stream.side_effect = _stream
            resp = client.get("/ask", params={"repo": str(repo), "query": "q"})

        assert resp.text == "data: tok\n\ndata: [ERROR: synthesis failed]\n\n"
        assert "detail the caller must not see" not in resp.text


# ---------------------------------------------------------------------------
# /ask through the REAL Synthesizer: a failure inside the model call
# ---------------------------------------------------------------------------

PROVIDER_DETAIL = "provider failed at host api.internal.example request-id req_123"


class _ModelCall:
    """The chat client behind the real Synthesizer: yields ``tokens``, then raises ``error``."""

    def __init__(self, tokens: tuple[str, ...], error: Exception | None) -> None:
        self._tokens = tokens
        self._error = error

    def stream(self, **_kwargs: Any) -> Iterator[str]:  # noqa: ANN401 - mirrors the client
        yield from self._tokens
        if self._error is not None:
            raise self._error


def _failing_model_call(*tokens: str) -> _ModelCall:
    return _ModelCall(tokens, RuntimeError(PROVIDER_DETAIL))


def _retrieved() -> RetrievedContext:
    return RetrievedContext(
        query="q", results=[MagicMock()], context_text="def f(): ...", total_tokens=5
    )


def _synthesizer(chat_client: Any) -> Synthesizer:  # noqa: ANN401 - any client double
    with patch("trelix.retrieval.synthesizer.build_chat_client", return_value=chat_client):
        return Synthesizer(EmbedderConfig(_env_file=None))  # type: ignore[call-arg]


class TestAskFailureInsideTheModelCall:
    """``Synthesizer.stream()`` catches a model-call failure itself and, by default, yields its
    text as a token. ``/ask`` must not forward that: these tests drive the real Synthesizer
    (only the chat client and the retriever are doubles)."""

    @pytest.fixture
    def ask_client(self, tmp_path: Path) -> tuple[Path, TestClient]:
        root = tmp_path / "root"
        repo = mark_indexed(root / "repo")
        return repo, TestClient(create_app(served_root=root), raise_server_exceptions=False)

    def _ask(self, repo: Path, client: TestClient, chat_client: Any) -> Any:  # noqa: ANN401
        with (
            patch("trelix.api.app.Retriever") as retriever,
            patch("trelix.retrieval.synthesizer.build_chat_client", return_value=chat_client),
        ):
            retriever.return_value.retrieve.return_value = _retrieved()
            return client.get("/ask", params={"repo": str(repo), "query": "q"})

    def test_a_model_call_failure_streams_only_the_fixed_frame(
        self, ask_client: tuple[Path, TestClient], caplog: pytest.LogCaptureFixture
    ) -> None:
        repo, client = ask_client
        caplog.set_level(logging.WARNING)

        resp = self._ask(repo, client, _failing_model_call())

        assert resp.status_code == 200
        assert resp.text == "data: [ERROR: synthesis failed]\n\n"
        assert "api.internal.example" not in resp.text
        assert "req_123" not in resp.text
        assert "synthesis unavailable" not in resp.text
        # The detail is in the server log, and at ERROR level only once.
        assert any(PROVIDER_DETAIL in record.getMessage() for record in caplog.records)
        assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1

    def test_a_failure_after_some_tokens_keeps_them_then_the_fixed_frame(
        self, ask_client: tuple[Path, TestClient]
    ) -> None:
        repo, client = ask_client

        resp = self._ask(repo, client, _failing_model_call("tok1", "tok2"))

        assert resp.text == "data: tok1\n\ndata: tok2\n\ndata: [ERROR: synthesis failed]\n\n"
        assert "api.internal.example" not in resp.text

    def test_an_unconfigured_llm_still_streams_the_backends_fixed_placeholder(
        self, ask_client: tuple[Path, TestClient]
    ) -> None:
        """Not an exception: the backend yields its own fixed text (no secret in it) and the
        Synthesizer records ``last_error``. /ask must keep streaming it, then ``[DONE]``."""
        repo, client = ask_client
        keyless = LLMConfig(provider="openai", _env_file=None)  # type: ignore[call-arg]
        backend = OpenAIBackend(keyless.model_copy(update={"openai_api_key": None}))

        resp = self._ask(repo, client, backend)

        assert resp.text == (
            "data: [trelix] LLM not configured — set OPENAI_API_KEY or AZURE_API_KEY.\n\n"
            "data: [DONE]\n\n"
        )

    def test_a_successful_answer_is_unchanged(self, ask_client: tuple[Path, TestClient]) -> None:
        repo, client = ask_client

        resp = self._ask(repo, client, _ModelCall(("The ", "answer."), None))

        assert resp.text == "data: The \n\ndata: answer.\n\ndata: [DONE]\n\n"


class TestSynthesizerStreamErrorOptIn:
    """The default of ``Synthesizer.stream()`` is what ``trelix ask`` relies on (its operator
    wants the banner); ``raise_on_error=True`` is the opt-in the REST route uses."""

    def test_by_default_a_failure_still_yields_the_banner_token_and_never_raises(self) -> None:
        synth = _synthesizer(_failing_model_call("partial "))

        tokens = list(synth.stream(_retrieved(), RetrievalConfig()))

        assert tokens == [
            "partial ",
            "\n[trelix: synthesis unavailable — provider failed at host api.internal.example"
            " request-id req_123]",
        ]
        assert synth.last_error == PROVIDER_DETAIL

    def test_raise_on_error_re_raises_after_recording_the_failure(self) -> None:
        synth = _synthesizer(_failing_model_call("partial "))
        seen: list[str] = []

        with pytest.raises(RuntimeError, match="request-id req_123"):
            for token in synth.stream(_retrieved(), RetrievalConfig(), raise_on_error=True):
                seen.append(token)

        assert seen == ["partial "]
        assert synth.last_error == PROVIDER_DETAIL

    def test_raise_on_error_leaves_a_stream_that_did_not_fail_alone(self) -> None:
        synth = _synthesizer(_ModelCall(("The ", "answer."), None))

        tokens = list(synth.stream(_retrieved(), RetrievalConfig(), raise_on_error=True))

        assert tokens == ["The ", "answer."]
        assert synth.last_error is None
