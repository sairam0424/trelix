"""Regression test: `ask`'s FLARE branch printed the synthesized answer twice.

THE BUG. `Synthesizer.synthesize()` -> `_stream_response()` writes every
streamed token directly to `sys.stdout` as it arrives (that IS the point —
the terminal shows tokens live), then also returns the fully-assembled
string. `FLARELoop.run()` returns that same string unchanged. `ask`'s FLARE
branch then did `console.print(_safe_text(answer))` on the return value,
printing the whole answer a second time — found live during the v4.0.0
pre-promotion dry run (`TRELIX_RETRIEVAL_FLARE=true`, this repo's own
documented default-recommended setting).

The non-FLARE `synth.stream()` path is unaffected: it is a pure generator
with no internal stdout writes, and `ask` prints each token itself exactly
once (see the token loop lower in the same command).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from trelix.cli.main import app
from trelix.core.models import RetrievedContext

runner = CliRunner()


def _context() -> RetrievedContext:
    return RetrievedContext(
        query="how does add work",
        results=[MagicMock()],
        context_text="def add(a, b): return a + b",
        total_tokens=42,
        elapsed_seconds=0.01,
    )


def test_ask_flare_branch_prints_the_answer_exactly_once(tmp_path: Path) -> None:
    answer = "`add` returns the sum of its two arguments."

    def _fake_run(query: str) -> str:
        # Mirrors the real bug: Synthesizer._stream_response() already wrote
        # this to stdout as it streamed, before FLARELoop.run() returns it.
        print(answer, end="")
        return answer

    fake_loop = MagicMock()
    fake_loop.run.side_effect = _fake_run

    with (
        patch("trelix.retrieval.retriever.Retriever") as MockRetriever,
        patch("trelix.retrieval.synthesizer.Synthesizer"),
        patch("trelix.retrieval.flare.FLARELoop", return_value=fake_loop),
    ):
        MockRetriever.return_value.retrieve.return_value = _context()
        result = runner.invoke(
            app,
            ["ask", str(tmp_path), "how does add work", "--provider", "local"],
            env={"TRELIX_RETRIEVAL_FLARE": "true"},
        )

    assert result.exit_code == 0, result.stdout
    assert result.stdout.count(answer) == 1, (
        f"expected the answer exactly once, found {result.stdout.count(answer)}: {result.stdout!r}"
    )
