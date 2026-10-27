"""Shared fixtures for the `trelix ask --json` and `Sources:` footer tests.

Not a test module (no ``test_`` prefix); imported as ``tests.unit.ask_cli_harness`` by
tests/unit/test_cli_ask_json.py, tests/unit/test_cli_ask_footer.py and
tests/unit/test_synthesizer_stdout.py.

The fixture is the C-5 design's: the sources S over a repository F2 built in ``tmp_path``
(``src/auth/middleware.py`` = 80 newline-terminated lines, ``src/auth/jwt.py`` = 25 lines with
no trailing newline), a scripted chat client that is a real ``TrelixChatClient`` (so
``synthesize()`` takes its normal path), and ``Retriever`` patched to return a context carrying
the sources. No network. Every value a test compares against is a literal, here or in the test.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from trelix.cli.main import app
from trelix.core.models import RetrievedContext
from trelix.llm.client import ChatMessage, ChatResponse, TrelixChatClient
from trelix.retrieval.citations import CitationSource

runner = CliRunner()

QUERY = "how is a jwt verified"
MIDDLEWARE = "src/auth/middleware.py"
JWT = "src/auth/jwt.py"
NO_RESULTS_NOTICE = "[trelix] No relevant code found — cannot synthesize an answer."
ABSTENTION = "INSUFFICIENT_EVIDENCE: nothing about retries was retrieved."
ABSTENTION_TOKENS = ("INSUFFICIENT_EVIDENCE: ", "nothing about retries was retrieved.")
ANSWER = "The token is read from the header [C2] and verified by `verify` [C1]."
ANSWER_TOKENS = ("The token is read from the header [C2] ", "and verified by `verify` [C1].")
SOURCES = (
    CitationSource(1, 11, MIDDLEWARE, 42, 67, "AuthMiddleware.verify"),
    CitationSource(2, 33, MIDDLEWARE, 70, 80, "AuthMiddleware.bearer"),
    CitationSource(3, 22, JWT, 10, 30, "decode_token"),
)


class ScriptedChatClient(TrelixChatClient):
    """A real TrelixChatClient that streams one scripted answer per call; ``configured=False``
    is a backend without credentials."""

    def __init__(self, *answers: tuple[str, ...], configured: bool = True) -> None:
        self._client = object() if configured else None
        self._answers = list(answers)
        self.calls = 0

    def complete(self, messages: list[ChatMessage], **_kwargs: Any) -> ChatResponse:
        raise NotImplementedError

    def stream(self, messages: list[ChatMessage], **_kwargs: Any) -> Iterator[str]:
        self.calls += 1
        return iter(self._answers.pop(0))

    def tool_call(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class RaisingChatClient(TrelixChatClient):
    """A configured client whose every call raises ``error``."""

    def __init__(self, error: Exception) -> None:
        self._client = object()
        self._error = error

    def complete(self, messages: list[ChatMessage], **_kwargs: Any) -> ChatResponse:
        raise self._error

    def stream(self, messages: list[ChatMessage], **_kwargs: Any) -> Iterator[str]:
        raise self._error

    def tool_call(self, *args: Any, **kwargs: Any) -> Any:
        raise self._error


def make_context(
    *, sources: tuple[CitationSource, ...] = SOURCES, results: list[Any] | None = None
) -> RetrievedContext:
    """A retrieved context with one result and the given citation sources."""
    return RetrievedContext(
        query=QUERY,
        results=[MagicMock()] if results is None else results,
        context_text="[C1] [Lines 42-67] AuthMiddleware.verify\n...",
        total_tokens=42,
        citation_sources=sources,
    )


def no_results_context() -> RetrievedContext:
    """What the assembler returns when retrieval found nothing."""
    return RetrievedContext(
        query=QUERY, results=[], context_text="No relevant code found.", total_tokens=0
    )


def make_repo(tmp_path: Path) -> Path:
    """F2 on disk, marked indexed (`ask` refuses a repo with no index; Retriever is patched)."""
    auth = tmp_path / "src" / "auth"
    auth.mkdir(parents=True)
    (auth / "middleware.py").write_bytes("".join(f"line {i}\n" for i in range(1, 81)).encode())
    (auth / "jwt.py").write_bytes("\n".join(f"line {i}" for i in range(1, 26)).encode())
    (tmp_path / ".trelix").mkdir()
    (tmp_path / ".trelix" / "index.db").touch()
    return tmp_path


def one_line(text: str) -> str:
    """Undo Rich's wrapping so a long stderr message can be matched as one sentence."""
    return " ".join(text.split())


def invoke_ask(
    repo: Path,
    client: Any,
    context: RetrievedContext,
    *args: str,
    flare: bool = False,
) -> Any:
    """Run ``trelix ask REPO QUERY --provider openai *args`` against ``client`` and ``context``."""
    env = {"TRELIX_RETRIEVAL_FLARE": "true"} if flare else {}
    with (
        patch("trelix.retrieval.retriever.Retriever") as mock_retriever,
        patch("trelix.retrieval.synthesizer.build_chat_client", return_value=client),
        # GraphRAGSynthesizer (consulted on the FLARE path) builds its client via the factory.
        patch("trelix.llm.factory.build_chat_client", return_value=client),
    ):
        mock_retriever.return_value.retrieve.return_value = context
        return runner.invoke(app, ["ask", str(repo), QUERY, "--provider", "openai", *args], env=env)
