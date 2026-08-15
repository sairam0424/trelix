"""Regression tests: indexed-repository text must survive Rich's markup parser.

THE BUG. ``trelix.cli.main`` builds a module-level ``Console()``, so markup is
ON and every string reaching ``console.print`` / ``Table.add_row`` / a Panel or
Table title is parsed for "[tag]" console markup. The CLI then renders values
that came out of the indexed repository — file paths, symbol names, retrieved
source code — and LLM answers that quote that code. A bracket-shaped token in
any of them either

  * raises ``rich.errors.MarkupError`` when it looks like an unmatched closing
    tag, so the command prints NOTHING and exits nonzero, or
  * is silently swallowed when it looks like an opening tag, so the value
    renders WRONG.

Neither needs an attacker. trelix's own source trips it: line 1035 of
``src/trelix/indexing/parser/extractors/rust.py`` is

    raw = re.sub(r"^//[/!]?\\s?", "", raw, flags=re.MULTILINE)

and its ``[/!]`` is an unmatched closing tag, so ``trelix ask`` / ``trelix
search`` against any repo containing a Rust comment-stripping regex — trelix
included — died instead of showing results. That exact line is read off disk
below rather than pasted, so these tests keep pinning the real thing.

WHY "NO EXCEPTION" IS NOT THE ASSERTION. A command that renders nothing also
raises nothing, and a swallowed opening tag exits 0 while losing text. So every
test here asserts the payload's LITERAL characters appear in the output. The
``audit list`` tests in tests/unit/test_cli_audit.py pin the same defect class
for attacker-controlled audit rows; this file covers the retrieval commands.

The JSON test at the bottom is the guard rail in the other direction: escaping
a machine-readable payload would be a new bug, so ``search --json`` must emit
the value byte-for-byte.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from trelix.cli.main import app
from trelix.core.models import Chunk, IndexedFile, Language, RetrievedContext, SearchResult, Symbol

runner = CliRunner()

# The verbatim trigger, read from trelix's own source so the test cannot drift
# away from the line it is pinning.
_RUST_EXTRACTOR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "trelix"
    / "indexing"
    / "parser"
    / "extractors"
    / "rust.py"
)


def _rust_comment_stripper_line() -> str:
    """Return the real ``re.sub(r"^//[/!]?...")`` line from the Rust extractor."""
    for line in _RUST_EXTRACTOR.read_text(encoding="utf-8").splitlines():
        if 'r"^//[/!]?' in line:
            return line.strip()
    raise AssertionError(
        f"the [/!] comment-stripping regex is gone from {_RUST_EXTRACTOR}; "
        "re-point this test at whatever markup-shaped source line replaced it"
    )


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin console width so cells are never ellipsized.

    Rich falls back to 80 columns when stdout is not a terminal (as under
    CliRunner), and an 80-column table truncates the File/Symbol cells to
    "src/we…" — which would fail the literal-payload assertions for a reason
    unrelated to escaping. Rich reads COLUMNS live from os.environ, so setenv
    works even though the Console was built at import time.
    """
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setenv("TERM", "dumb")


@pytest.fixture(autouse=True)
def _no_leaked_log_handler() -> Iterator[None]:
    """Undo the root-logger handler each invoked command installs.

    Every real command here calls ``_setup_logging()``, which builds a bare
    ``logging.StreamHandler()``. That binds to whatever ``sys.stderr`` is at
    construction time — under CliRunner, a capture buffer that is closed when
    the invocation ends. The handler outlives it on the root logger, so the
    NEXT test in the session that logs anything writes to a closed file and
    dumps a "--- logging error ---" traceback into ITS output. That reliably
    broke tests/unit/test_cli_audit.py when this module ran before it. Snapshot
    and restore instead of asserting on the leak, so ordering stays irrelevant.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        yield
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def _result(rel_path: str, symbol_name: str, *, body: str = "code") -> SearchResult:
    """One SearchResult with fully caller-controlled path and symbol name."""
    return SearchResult(
        chunk=Chunk(symbol_id=1, chunk_text=body, token_count=4, id=1),
        symbol=Symbol(
            file_id=1,
            name=symbol_name,
            qualified_name=symbol_name,
            kind="function",  # type: ignore[arg-type]
            line_start=1,
            line_end=2,
            signature=f"def {symbol_name}()",
            body=body,
            id=1,
        ),
        file=IndexedFile(
            path=f"/repo/{rel_path}",
            rel_path=rel_path,
            language=Language.PYTHON,
            hash="deadbeef",
            size_bytes=len(body),
        ),
        score=0.9876,
        rank=1,
        source="vector",
    )


def _context(results: list[SearchResult], context_text: str = "") -> RetrievedContext:
    return RetrievedContext(
        query="q",
        results=results,
        context_text=context_text,
        total_tokens=42,
        elapsed_seconds=0.01,
    )


def _patched_retriever(context: RetrievedContext):
    """Patch the Retriever class so no index, DB, or embedder is touched.

    ``search``/``ask``/``query`` import Retriever from the module at call time,
    so patching the module attribute is enough.
    """
    fake = MagicMock()
    fake.return_value.retrieve.return_value = context
    return patch("trelix.retrieval.retriever.Retriever", fake)


def _combined(result) -> str:  # type: ignore[no-untyped-def]
    out = result.stdout
    err = getattr(result, "stderr", "") or ""
    return out + err


# ---------------------------------------------------------------------------
# 1. The demonstrated crash: trelix's own rust.py line through `ask`'s context
# ---------------------------------------------------------------------------


def test_ask_renders_real_rust_extractor_line_without_markup_error(tmp_path: Path) -> None:
    """THE reproducer. ``ask`` with the local provider prints context_text — the
    retrieved source, verbatim. Feeding it trelix's own
    ``re.sub(r"^//[/!]?\\s?", ...)`` raised MarkupError: "closing tag '[/!]'
    doesn't match any open tag", so the command printed nothing and exited 1.
    """
    from rich.errors import MarkupError

    line = _rust_comment_stripper_line()
    assert "[/!]" in line, line  # the payload really is an unmatched closing tag

    # --provider local is load-bearing, not decoration: `ask` only prints
    # context_text on the local (no-API-key) path, and a developer's .env
    # setting TRELIX_EMBEDDER_PROVIDER=azure would otherwise route this test
    # down the streaming branch and quietly stop testing the reported bug.
    with _patched_retriever(_context([_result("src/rust.py", "strip")], context_text=line)):
        result = runner.invoke(
            app, ["ask", str(tmp_path), "how are comments stripped", "--provider", "local"]
        )

    assert not isinstance(result.exception, MarkupError), result.exception
    assert result.exit_code == 0, _combined(result)
    # Rendered, not merely survived: the regex must come back out intact.
    assert 'r"^//[/!]?' in _combined(result)


def test_search_renders_real_rust_extractor_line_as_a_symbol_name(tmp_path: Path) -> None:
    """The same payload arriving as a symbol name instead of as context text.

    Symbol names are genuinely arbitrary: the HTML extractor derives one from an
    attribute value via ``script_src.split("/")[-1]``, and the YAML extractor
    decodes escapes out of double-quoted keys — so a symbol can hold anything a
    file can hold, newlines included.
    """
    line = _rust_comment_stripper_line()

    with _patched_retriever(_context([_result("src/rust.py", line)])):
        result = runner.invoke(
            app, ["search", str(tmp_path), "strip comments", "--provider", "local"]
        )

    assert result.exception is None, result.exception
    assert result.exit_code == 0, _combined(result)
    assert "[/!]" in _combined(result)


# ---------------------------------------------------------------------------
# 2. A file path containing "[/red]"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("search", lambda repo: ["search", repo, "q", "--provider", "local"]),
        ("query", lambda repo: ["query", repo, "q", "--provider", "local"]),
    ],
)
def test_result_table_renders_closing_tag_in_file_path_literally(
    tmp_path: Path, command: str, args
) -> None:
    """A lone closing tag is unparseable markup — the minimal MarkupError.

    ``[/red]`` in a path is legal on every filesystem trelix indexes, and both
    result tables used to abort on it having rendered zero rows.
    """
    from rich.errors import MarkupError

    hostile = "src/w[/red]/handler.py"
    with _patched_retriever(_context([_result(hostile, "handle"), _result("src/ok.py", "ok")])):
        result = runner.invoke(app, args(str(tmp_path)))

    assert not isinstance(result.exception, MarkupError), f"{command}: {result.exception!r}"
    assert result.exit_code == 0, _combined(result)
    combined = _combined(result)
    assert "[/red]" in combined, f"{command} swallowed the literal closing tag"
    # The ordinary row rendered too — MarkupError killed the entire table.
    assert "src/ok.py" in combined


def test_call_graph_renders_closing_tag_in_file_path_literally(tmp_path: Path) -> None:
    """`call-graph` builds its own table from get_callers/get_callees/get_importers."""
    from rich.errors import MarkupError

    fake = MagicMock()
    fake.return_value.get_callers.return_value = [_result("a[/red]b.py", "caller_fn")]
    fake.return_value.get_callees.return_value = []
    fake.return_value.get_importers.return_value = []

    with patch("trelix.retrieval.retriever.Retriever", fake):
        result = runner.invoke(app, ["call-graph", str(tmp_path), "target", "--provider", "local"])

    assert not isinstance(result.exception, MarkupError), result.exception
    assert result.exit_code == 0, _combined(result)
    combined = _combined(result)
    assert "a[/red]b.py" in combined
    assert "caller_fn" in combined


# ---------------------------------------------------------------------------
# 3. A symbol name containing "[bold]x[/bold]" — the SILENT failure mode
# ---------------------------------------------------------------------------


def test_search_renders_balanced_tags_in_symbol_name_literally(tmp_path: Path) -> None:
    """A *balanced* tag pair raises nothing and still corrupts the output.

    "[bold]x[/bold]" is valid markup, so Rich renders a bold "x" and drops ten
    characters of the real symbol name. The command exits 0 while lying about
    what is in the repository, which is why this file asserts on literal text
    rather than on the absence of an exception.
    """
    with _patched_retriever(_context([_result("src/app.py", "[bold]x[/bold]")])):
        result = runner.invoke(app, ["search", str(tmp_path), "q", "--provider", "local"])

    assert result.exit_code == 0, _combined(result)
    combined = _combined(result)
    assert "[bold]x[/bold]" in combined, (
        "Rich interpreted the symbol name as markup and rendered a bare 'x' — "
        "the value must survive verbatim"
    )


def test_search_all_renders_balanced_tags_in_qualified_name_literally(tmp_path: Path) -> None:
    """Federated search widens the blast radius: one hostile symbol name in one
    registered repo must not blank the whole cross-repo table."""
    hostile = _result("src/app.py", "[bold]Klass.method[/bold]")
    hostile.source = "backend:vector"
    benign = _result("src/ok.py", "fine")
    benign.source = "frontend:vector"

    registry = MagicMock()
    registry.list.return_value = [MagicMock()]
    fed = MagicMock()
    fed.return_value.retrieve.return_value = [hostile, benign]

    with (
        patch("trelix.federation.registry.RepoRegistry.load", return_value=registry),
        patch("trelix.federation.retriever.FederatedRetriever", fed),
    ):
        result = runner.invoke(app, ["search-all", "q"])

    assert result.exit_code == 0, _combined(result)
    combined = _combined(result)
    assert "[bold]Klass.method[/bold]" in combined
    assert "fine" in combined


# ---------------------------------------------------------------------------
# 4. An LLM answer containing an unmatched closing tag
# ---------------------------------------------------------------------------


def test_ask_agentic_renders_llm_answer_with_unmatched_closing_tag(tmp_path: Path) -> None:
    """An agent answer quotes the code it retrieved, so it inherits every
    bracket in that code. One "[/!]" used to discard the whole answer."""
    from rich.errors import MarkupError

    answer = f"The Rust extractor strips comments with:\n\n    {_rust_comment_stripper_line()}\n"
    loop = MagicMock()
    loop.return_value.run.return_value = (answer, "sess-123")

    with patch("trelix.agent.AgentLoop", loop):
        result = runner.invoke(app, ["ask", str(tmp_path), "q", "--agentic", "--provider", "local"])

    assert not isinstance(result.exception, MarkupError), result.exception
    assert result.exit_code == 0, _combined(result)
    combined = _combined(result)
    assert 'r"^//[/!]?' in combined
    assert "sess-123" in combined


def test_agent_sessions_show_renders_observation_with_unmatched_closing_tag(
    tmp_path: Path,
) -> None:
    """`agent sessions show` replays LLM thoughts and tool observations, and an
    observation IS retrieved repository source. Replaying a session that once
    read a Rust comment-stripping regex aborted on the first "[/!]"."""
    from rich.errors import MarkupError

    line = _rust_comment_stripper_line()
    turns = [
        {
            "turn_index": 0,
            "thought": "I should grep for [/dim] in the extractors",
            "action_type": "search",
            "action_arguments": '{"q": "[/!]"}',
            "observation_success": True,
            "observation_content": line,
        }
    ]
    db = MagicMock()
    db.return_value.get_agent_turns.return_value = turns

    with patch("trelix.store.db.Database", db):
        result = runner.invoke(app, ["agent", "sessions", "show", str(tmp_path), "sess-1"])

    assert not isinstance(result.exception, MarkupError), result.exception
    assert result.exit_code == 0, _combined(result)
    combined = _combined(result)
    assert "[/dim]" in combined, "the agent's own thought text was swallowed"
    assert "[/!]" in combined, "the tool observation (retrieved source) was swallowed"


def test_review_renders_llm_comment_with_unmatched_closing_tag(tmp_path: Path) -> None:
    """Review comments are LLM prose quoting the reviewed diff, and `severity`
    is model output nested inside trelix's own "[red]…[/red]" colour markup."""
    from rich.errors import MarkupError

    from trelix.review.reviewer import ReviewComment

    comment = ReviewComment(
        file_path="src/r[/red].rs",
        line_start=1,
        line_end=1,
        severity="ERROR",
        comment=f"This regex is wrong: {_rust_comment_stripper_line()}",
    )
    hunks = [MagicMock(file_path="src/r[/red].rs")]

    with (
        patch("trelix.review.diff_parser.DiffParser.from_git", return_value=hunks),
        patch("trelix.review.reviewer.DiffReviewer.review", return_value=[comment]),
    ):
        result = runner.invoke(app, ["review", str(tmp_path)])

    assert not isinstance(result.exception, MarkupError), result.exception
    assert result.exit_code == 0, _combined(result)
    combined = _combined(result)
    assert "[/!]" in combined, "the LLM comment body was swallowed"
    assert "src/r[/red].rs" in combined
    # trelix's own severity colouring still works — the literal "[red]" tag
    # around the value must NOT have been escaped along with it.
    assert "ERROR" in combined


# ---------------------------------------------------------------------------
# 5. The other direction: JSON output must NOT be escaped
# ---------------------------------------------------------------------------


def test_search_json_output_is_not_escaped(tmp_path: Path) -> None:
    """`--json` goes through builtin print(), which never parses markup.

    Escaping here would be a NEW bug: consumers would parse stray backslashes
    into their file paths and symbol names. The payload must round-trip through
    json.loads byte-for-byte.
    """
    hostile_path = "src/w[/red]/handler.py"
    hostile_symbol = "[bold]x[/bold]"

    with _patched_retriever(_context([_result(hostile_path, hostile_symbol)])):
        result = runner.invoke(app, ["search", str(tmp_path), "q", "--json", "--provider", "local"])

    assert result.exit_code == 0, _combined(result)
    payload = json.loads(result.stdout)
    assert payload["results"][0]["file"] == hostile_path
    assert payload["results"][0]["symbol"] == hostile_symbol
    assert "\\[" not in result.stdout, "escape() leaked into the machine-readable payload"


# ---------------------------------------------------------------------------
# 6. stdout purity for --json: the payload is only half the problem
# ---------------------------------------------------------------------------


def test_no_json_command_spins_on_the_stdout_console() -> None:
    """Structural invariant, checked over the whole module rather than per command.

    `_print_json()` renders the payload safely, but a `console.status()` spinner
    wrapping the work writes its animation frames and prose to the SAME stream.
    Rich suppresses those when stdout is not a terminal, so it is invisible in a
    pipe during development — but `FORCE_COLOR=1` is routinely set in CI, and then
    `trelix graph --json | jq` gets

        \x1b[?25l\x1b[32m⠋\x1b[0m Building knowledge graph...

    ahead of the array, and fails with **exit 0**.

    This is an AST check, not four per-command tests, so a NEW `--json` command
    that spins is caught the day it is added rather than the day someone pipes it.
    """
    import ast
    import pathlib

    source = pathlib.Path(
        str(pathlib.Path(__file__).resolve().parents[2] / "src" / "trelix" / "cli" / "main.py")
    ).read_text(encoding="utf-8")

    offenders: list[str] = []
    for fn in [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)]:
        if not any(a.arg == "json_output" for a in fn.args.args):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "status"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "console"
            ):
                offenders.append(f"{fn.name}() at line {node.lineno}")

    assert not offenders, (
        "these --json commands spin on the stdout console, so their JSON is "
        f"preceded by ANSI spinner output under FORCE_COLOR=1: {offenders}. "
        "Use _status_console(json_output) instead."
    )


def test_status_console_routes_to_stderr_in_json_mode() -> None:
    """The routing decision itself, tested directly.

    There is deliberately no in-process end-to-end companion to this. Rich only
    animates a `status()` spinner when the console `is_interactive`, and that is
    fixed relative to when the Console was built: `cli.main`'s console is
    constructed at import time, so setting FORCE_COLOR afterwards — which is all
    `CliRunner(env=...)` can do — flips `is_terminal` to True but leaves
    `is_interactive` False. The spinner therefore never activates under
    CliRunner, and such a test passes against the unrouted code, proving nothing.
    Observing the real contamination needs a subprocess with FORCE_COLOR already
    in its environment; the AST invariant above is what actually guards the
    call sites.
    """
    from trelix.cli.main import _status_console, console, err_console

    assert _status_console(json_output=True) is err_console, (
        "spinner output would land on stdout and precede the JSON payload"
    )
    assert _status_console(json_output=False) is console, (
        "non-JSON mode must keep its progress output on stdout as before"
    )


# ---------------------------------------------------------------------------
# 7. --json on the EMPTY path, and the pip hint Rich used to eat
# ---------------------------------------------------------------------------


def test_search_all_json_emits_an_array_when_no_repos_are_registered(tmp_path: Path) -> None:
    """The empty path is the first thing a new consumer hits, and it broke.

    `search-all` returned before reaching its `if json_output:` branch, so
    `--json` printed the human "No repos registered..." line to stdout and
    `json.loads` failed. An empty registry is the default state, so a consumer
    piping to jq hit this immediately — and it exited 0 while doing it.
    """
    empty_registry = tmp_path / "repos.json"
    empty_registry.write_text('{"repos": []}', encoding="utf-8")

    result = runner.invoke(app, ["search-all", "q", "--json", "--config", str(empty_registry)])

    assert result.exit_code == 0, _combined(result)
    assert json.loads(result.stdout) == [], f"not parseable JSON: {result.stdout!r}"


def test_search_all_non_json_still_prints_the_human_message(tmp_path: Path) -> None:
    """Control: the fix must be --json-gated, not a blanket behaviour change."""
    empty_registry = tmp_path / "repos.json"
    empty_registry.write_text('{"repos": []}', encoding="utf-8")

    result = runner.invoke(app, ["search-all", "q", "--config", str(empty_registry)])

    assert result.exit_code == 0, _combined(result)
    assert "No repos registered" in _combined(result)


def test_taint_pip_hint_is_escaped_at_the_call_site() -> None:
    """`pip install trelix[taint]` rendered as `pip install 'trelix'`.

    Rich read the unescaped `[taint]` as an opening tag and swallowed it, so the
    remediation instruction named the package the reader already had installed —
    the identical swallow `_print_error()`'s docstring documents for
    `trelix[local]`, at a site that fix did not reach.

    Asserted against the SOURCE rather than by rendering, deliberately. Reaching
    that branch through the CLI needs semgrep absent AND a real index, and a test
    that applies escape() to its own payload and then checks Rich preserved it is
    a tautology — it verifies Rich, not that trelix calls it. This fails if the
    escape() is ever removed from the call site, which is the actual regression.
    """
    source = (Path(__file__).resolve().parents[2] / "src" / "trelix" / "cli" / "main.py").read_text(
        encoding="utf-8"
    )

    # Scoped to the RENDERED line, not the whole file. The taint command's own
    # docstring also reads "Requires: pip install trelix[taint]", and a docstring
    # never reaches Rich — a whole-file assertion flags that harmless line and
    # fails against correct code.
    sink_lines = [ln for ln in source.splitlines() if "Ensure semgrep is installed" in ln]
    assert len(sink_lines) == 1, f"expected one rendered hint, found {len(sink_lines)}"
    sink = sink_lines[0]

    assert "pip install trelix[taint]" not in sink, (
        "the taint hint interpolates trelix[taint] unescaped again — Rich will "
        "swallow [taint] and tell the user to install plain 'trelix'"
    )
    assert "escape(" in sink, "the taint hint no longer escapes its bracketed payload"
