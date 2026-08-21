"""
trelix CLI — Phase 14 full implementation.

Commands:
    trelix index  <repo> [--provider local|openai|azure|voyage|local-code
                          |bedrock-titan|bedrock-cohere] [-v]
    trelix search <repo> <query> [--provider ...] [--json]
    trelix ask    <repo> <query> [--provider ...]
    trelix query  <repo> <query> [--provider ...]
    trelix stats  <repo>
    trelix update-index <repo> <file>
"""

from __future__ import annotations

import errno
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from trelix.core.console_safety import safe_text
from trelix.federation.registry import RepoRegistry

# Imported eagerly, unlike every other store import in this module, because it is the
# default value of `index --prune-max-percent` and typer reads that at decoration time.
# provenance.py itself pulls in nothing heavier than hashlib/subprocess.
from trelix.store.provenance import _PRUNE_MAX_FRACTION_DEFAULT

if TYPE_CHECKING:
    import sqlite3

    from trelix.core.config import EmbedderConfig, IndexConfig
    from trelix.core.models import IndexedFile
    from trelix.indexing.indexer import Indexer
    from trelix.store.db import Database
    from trelix.store.provenance import DriftReport, IndexProvenance, PrunePlan

# Windows' legacy console codepage (cp1252 etc.) can't encode the Unicode
# braille glyphs Rich's default spinner renders (e.g. U+280B), crashing with
# "'charmap' codec can't encode character ...". Reconfiguring to UTF-8 with
# errors="replace" before any Console is constructed makes every command
# degrade gracefully instead of crashing — must run before Console()
# below, and reconfigure() is a no-op if stdout/stderr are already UTF-8
# (macOS/Linux terminals, or Windows Terminal with UTF-8 already active).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*huggingface.*")

# ---------------------------------------------------------------------------
# A vanished reader is not an error — READ THIS BEFORE SIMPLIFYING IT AWAY
# ---------------------------------------------------------------------------
#
# `trelix stats <repo> | grep -q "Symbols"` is the binary smoke test in
# build-binaries.yml. `grep -q` exits the instant it matches — before stats has
# written the provenance block that now follows the table — so every later write
# lands on a pipe with no reader. Three layers each call that a failure, and none
# of them agrees with the other two:
#
#   * rich's Console.on_broken_pipe raises SystemExit(1);
#   * click's Command.main exits 1, but only when errno is EPIPE;
#   * nothing handles errno EINVAL, and EINVAL is what Windows reports for a
#     reader-less pipe (WriteFile -> ERROR_NO_DATA), not BrokenPipeError.
#
# So the step failed two different ways. On POSIX, silently with exit 1 — GitHub
# Actions runs `shell: bash` as `bash -e -o pipefail`, which fails the pipeline on
# the LEFT side's status even though grep matched. On Windows, the OSError escaped
# entirely: a rich traceback plus PyInstaller's "Failed to execute script main".
# Both were filed as CI flakiness, twice, across two days — the specific cost of a
# red that does not look like the bug it is.
#
# What status to report when the reader walks away is the application's policy, not
# rich's and not click's, so it is decided here once for every command instead of
# inside stats. A fix inside stats would have left search/telemetry/graph — which
# share the consoles below — failing exactly the same way.

# EINVAL is in this set because of Windows and only because of Windows. Keeping the
# set to these two is the point: ENOSPC or EACCES on stdout is a real failure a
# caller must still be able to detect, and a blanket `except OSError` would hide it.
_CLOSED_CONSUMER_ERRNOS = frozenset({errno.EPIPE, errno.EINVAL})


def _is_closed_consumer_error(exc: OSError) -> bool:
    """True when `exc` means "nothing is reading our output any more"."""
    return exc.errno in _CLOSED_CONSUMER_ERRNOS


class _PacifiedStream:
    """Hold a broken stream so its *final* flush cannot raise.

    CPython flushes sys.stdout/sys.stderr during interpreter shutdown, long after
    any handler has returned. If the stream is still broken, that flush raises with
    nowhere to report it: Python prints "Exception ignored in: ..." and the process
    exits **120**. That is a second, different red replacing the one just handled,
    which is why silencing the stream is part of the fix and not tidy-up.

    Same job as click's PacifyFlushWrapper, kept local so a click internal is not
    load-bearing. Wrapping (rather than replacing) also keeps the original stream
    referenced, so it is not finalised — and cannot raise from __del__ — early.
    """

    def __init__(self, wrapped: Any) -> None:  # noqa: ANN401 - any text stream
        self._wrapped = wrapped

    def flush(self) -> None:
        try:
            self._wrapped.flush()
        except OSError:
            pass

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - passthrough
        return getattr(self._wrapped, name)


def _exit_quietly_after_closed_consumer(stream: Any = None) -> NoReturn:  # noqa: ANN401
    """Stop writing to `stream`, leave no traceback, exit 0."""
    target = sys.stdout if stream is None else stream

    # Point the file descriptor at the null device first. This is the half that
    # works when the bytes are already buffered inside a real TextIOWrapper: the
    # shutdown flush still happens, it just lands in /dev/null (NUL on Windows).
    try:
        fd = target.fileno()
    except (AttributeError, OSError, ValueError):
        fd = None
    if fd is not None:
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull, fd)
            finally:
                os.close(devnull)
        except OSError:
            pass

    # Then pacify the Python object, for the streams where the descriptor swap is
    # not enough or not available: a frozen build with no console, a stream replaced
    # by a test harness, or a wrapper whose flush() fails independently of the fd.
    if target is sys.stdout:
        sys.stdout = cast("Any", _PacifiedStream(sys.stdout))
    elif target is sys.stderr:
        sys.stderr = cast("Any", _PacifiedStream(sys.stderr))

    raise SystemExit(0)


class _PipeSafeConsole(Console):
    """A rich Console whose broken-pipe policy is exit 0, not rich's exit 1.

    Applied to `console`/`err_console` below, so the EPIPE half is fixed once for
    every command that renders through them — stats, telemetry, graph, review,
    search's table output — rather than per command.

    NOT covered, because they bypass these consoles entirely: the three
    `print(json.dumps(...))` sites (`search --json`, `update-index`,
    `audit export`). Those go to click's own EPIPE handler, which exits 1 silently
    with no traceback. Their *crash* exposure — Windows errno EINVAL — is covered by
    _PipeSafeTyper below; only the POSIX exit code differs, and no CI step pipes them.
    """

    def on_broken_pipe(self) -> None:
        _exit_quietly_after_closed_consumer(self.file)


class _PipeSafeTyper(typer.Typer):
    """The entry point, which is where the process exit status is actually decided.

    Catches the case neither rich nor click recognises: OSError(EINVAL), i.e. every
    Windows closed-pipe write, raised by *any* writer in the process — including
    typer's own help renderer and Indexer's private Console, neither of which goes
    through the consoles below. This is the layer that turns "Failed to execute
    script main" into a clean exit, and it covers both entry paths: the console
    script (`trelix.cli.main:app`) and the frozen binary, whose PyInstaller spec
    runs this module as __main__.

    Known residual, deliberately not papered over: on POSIX, `--help`/`--version`
    still exit 1 on EPIPE, because rich/click convert that errno to SystemExit(1)
    inside code this class cannot see the errno from. It is silent and
    traceback-free, and no CI step pipes those. Catching SystemExit(1) here to
    "fix" it would swallow every genuine exit 1 in the CLI.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - typer's own
        try:
            return super().__call__(*args, **kwargs)
        except OSError as exc:
            if not _is_closed_consumer_error(exc):
                raise
            _exit_quietly_after_closed_consumer()


app = _PipeSafeTyper(
    name="trelix",
    help="Fast, reliable code indexing and retrieval.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = _PipeSafeConsole()
err_console = _PipeSafeConsole(stderr=True)

logger = logging.getLogger("trelix.cli")


def _print_error(label: str, detail: object) -> None:
    """Print a "[red]<label>:[/red] <detail>" error line, safely.

    `detail` is almost always str(exc) — arbitrary text that may contain
    literal square brackets (e.g. "pip install 'trelix[local]'"). Rich
    interprets bracketed text as markup, so interpolating it directly into
    an f-string silently strips/mangles it — the exact bug that made a real
    "pip install 'trelix[local]'" fix instruction render as the already-run
    "pip install 'trelix'". escape() neutralizes detail's brackets while
    leaving the surrounding [red]/[/red] markup intact.
    """

    err_console.print(f"[red]{label}:[/red] {_safe_text(str(detail))}")


def _print_json(payload: object, *, indent: int | None = 2) -> None:
    """Emit `payload` on stdout as JSON a consumer can parse.

    This is the OPPOSITE fix to escape(). A display sink wants the value
    transformed so Rich cannot misread it; a machine-readable sink wants the
    value byte-identical and the renderer's features switched off instead —
    escaping here would write stray backslashes into consumers' parsed strings.

    Rich broke these payloads two ways, and neither needed hostile input:

      * markup=False — `console.print(json.dumps(...))` parses the JSON text
        for "[tag]" markup, so a reviewed file path or LLM comment containing
        "[/red]" raised MarkupError and `trelix review --json` emitted nothing.
      * soft_wrap=True — Rich hard-wraps at console width, and a wrap landing
        inside a JSON *string* injects a literal newline, which json.loads
        rejects as "Invalid control character". A single long unbroken token in
        an LLM review comment was enough; whitespace *between* JSON tokens is
        semantically free, which is why short payloads hid this for so long.

    highlight=False drops Rich's JSON syntax highlighter, whose ANSI codes
    would corrupt stdout whenever it is a terminal rather than a pipe.
    """
    console.print(
        json.dumps(payload, indent=indent),
        markup=False,
        highlight=False,
        soft_wrap=True,
    )


# Relocated to trelix.core.console_safety so indexing/indexer.py can share it — it is
# the only other module in src/ that builds a markup Console, and it cannot import from
# the CLI without a cycle. Aliased to the private name so the ~86 call sites below are
# unchanged; see that module for why the strip-then-escape order is load-bearing.

_safe_text = safe_text


def _status_console(json_output: bool) -> Console:
    """Console for spinners and progress messages, given the caller's `--json` flag.

    `_print_json()` above makes the payload itself clean, but that is only half of
    stdout purity: a `console.status()` spinner wrapping the work writes its
    animation frames and prose to the SAME stream. Rich suppresses those when
    stdout is not a terminal, so it is invisible in a pipe during development —
    but `FORCE_COLOR=1` is routinely set in CI, and any pty-allocating wrapper has
    the same effect. Then stdout opens with

        \\x1b[?25l\\x1b[32m⠋\\x1b[0m Building knowledge graph...

    and `trelix graph --json | jq` yields garbage with **exit 0** — the worst
    failure mode for a machine-readable contract, because nothing signals it.

    Routing status to stderr in `--json` mode is the fix `review --pr` already
    used; this makes it the single way every command does it.
    """
    return err_console if json_output else console


# ---------------------------------------------------------------------------
# Rich markup safety — READ THIS BEFORE REMOVING ANY escape() BELOW
# ---------------------------------------------------------------------------
#
# The module-level Console() above has markup ENABLED, so every string handed
# to console.print(), err_console.print(), Table.add_row(), a Table title, or a
# Panel body is parsed for "[tag]" console markup. Two failure modes follow:
#
#   1. An unmatched closing tag raises rich.errors.MarkupError, so the command
#      renders NOTHING and exits nonzero.
#   2. A well-formed-looking opening tag is silently swallowed, so the value
#      renders WRONG (this is the bug _print_error() above documents: a real
#      "pip install 'trelix[local]'" hint rendered as "pip install 'trelix'").
#
# Most values these commands render are arbitrary text, not tame identifiers:
# indexed file paths and symbol names, retrieved source code, LLM answers and
# agent observations that quote that code, Semgrep findings, GitHub PR
# filenames, and persisted queries. Neither failure mode needs an attacker —
# trelix's own source trips it. src/trelix/indexing/parser/extractors/rust.py
# contains
#
#       raw = re.sub(r"^//[/!]?\s?", "", raw, flags=re.MULTILINE)
#
# whose "[/!]" is an unmatched closing tag, so `trelix ask` / `trelix search`
# against any repo holding a Rust comment-stripping regex — including trelix
# itself — used to die with MarkupError instead of showing results. Symbol
# names are equally arbitrary: the HTML extractor derives one from an attribute
# value via script_src.split("/")[-1], and HTML attributes may legally contain
# a newline.
#
# THE INVARIANT: interpolate no dynamic value into a markup-interpreting sink
# unescaped. rich.markup.escape() renders the value identically (its only
# artifact is doubling a trailing lone backslash) while neutralizing brackets.
# Escape the VALUE only — trelix's own literal "[red]…[/red]" markup around it
# must stay unescaped to keep working. Sites that emit machine-readable JSON are
# deliberately NOT escaped — they go through _print_json() above, which turns
# Rich's markup parser and line wrapping OFF instead of altering the value.
#
# `trelix audit list` was fixed this way first (see the comment there).


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        import trelix

        typer.echo(f"trelix {trelix.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """trelix — fast, reliable code indexing and retrieval."""


_EmbedderProvider = Literal[
    "openai",
    "azure",
    "local",
    "voyage",
    "local-code",
    "bge-code",
    "nomic-code",
    "bedrock-titan",
    "bedrock-cohere",
]

_PROVIDER_HELP = (
    "Embedding provider: local | openai | azure | voyage"
    " | local-code | bge-code | nomic-code | bedrock-titan | bedrock-cohere"
    " (default: TRELIX_EMBEDDER_PROVIDER env var, or 'local' if unset)"
)


def _build_embedder_config(provider: str | None) -> EmbedderConfig:
    """Build an EmbedderConfig, honoring TRELIX_EMBEDDER_PROVIDER when --provider wasn't passed.

    EmbedderConfig is a pydantic-settings model, so an explicit constructor
    kwarg always wins over the env var — passing provider="local" unconditionally
    (the old behavior) silently overrode TRELIX_EMBEDDER_PROVIDER on every call,
    even when the user never touched --provider. Omitting the kwarg entirely
    when unset lets pydantic-settings fall through to the env var as documented.
    """
    from trelix.core.config import EmbedderConfig

    if provider is None:
        return EmbedderConfig()
    return EmbedderConfig(provider=cast(_EmbedderProvider, provider))


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    """Configure the trelix logger. Call once at CLI entry."""
    from trelix.core.logging_setup import setup_console_logging

    level = logging.DEBUG if verbose else logging.WARNING
    setup_console_logging(level)


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


@app.command()
def index(
    repo: str = typer.Argument(..., help="Path to the repository to index"),
    provider: str | None = typer.Option(None, help=_PROVIDER_HELP),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Cost preview only. Walks, chunks and counts tokens with the real tokenizer, "
            "then reports the embedding spend a real run would incur. Embeds nothing."
        ),
    ),
    prune: bool = typer.Option(
        False,
        "--prune",
        help=(
            "After indexing, remove index rows and embeddings for files no longer in the "
            "repository. Previews only unless --yes is given, and refuses unless the walk "
            "can be shown to be trustworthy."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Actually delete what --prune found. Meaningless without --prune.",
    ),
    prune_max_percent: float = typer.Option(
        _PRUNE_MAX_FRACTION_DEFAULT * 100,
        "--prune-max-percent",
        help=(
            "Refuse a prune that would remove more than this percentage of the index. "
            "Raise it only after reading the previewed list."
        ),
    ),
) -> None:
    """Index a repository — builds the search index at <repo>/.trelix/index.db"""
    _setup_logging(verbose)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import IndexConfig
    from trelix.indexing.indexer import Indexer

    # Refused rather than resolved by precedence. Either combination means the user has a
    # different command in mind than the one they typed, and picking one for them is how a
    # `--yes` that was meant to authorise a prune ends up authorising nothing at all.
    if dry_run and yes:
        _print_error(
            "Contradictory flags",
            "--dry-run changes nothing, so --yes has nothing to authorise. Drop one.",
        )
        raise typer.Exit(1)
    if yes and not prune:
        _print_error(
            "Nothing to authorise",
            "--yes only applies to --prune. Add --prune, or drop --yes.",
        )
        raise typer.Exit(1)

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=_build_embedder_config(provider),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        _print_error("Configuration error", detail)
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    # _safe_text(repo): a directory legitimately named e.g. "Project [old]" would
    # otherwise render with "[old]" swallowed as a markup tag.
    if dry_run:
        console.print(
            Panel(
                f"[bold cyan]Dry run — cost preview only[/bold cyan] {_safe_text(repo)}",
                expand=False,
            )
        )
        _print_cost_preview(config)
        if prune:
            # Previewed against the index as it stands, which is the point: a prune
            # planned before a re-index reports the refusal the user would hit.
            _prune_from_index(config, confirmed=False, max_fraction=prune_max_percent / 100.0)
        return

    console.print(Panel(f"[bold cyan]Indexing[/bold cyan] {_safe_text(repo)}", expand=False))

    t0 = time.perf_counter()
    try:
        indexer = Indexer(config)

        # Read BEFORE index() runs. index() writes provenance at the end of the run, so a
        # record read after it describes the run that just finished — and a prune guard fed
        # that record compares this walk config against a copy of itself. Captured here even
        # when --prune was not passed: it is one cheap key-value read, and making it
        # conditional invites the next caller to reach for the post-run value.
        from trelix.store.provenance import read_provenance

        provenance_before = read_provenance(indexer.db)

        stats = indexer.index()
    except KeyboardInterrupt:
        err_console.print("[yellow]Indexing cancelled.[/yellow]")
        raise typer.Exit(1)
    except Exception as exc:
        _print_error("Indexing failed", exc)
        raise typer.Exit(1) from exc

    elapsed = time.perf_counter() - t0

    table = Table(title="Index Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_row("Files found", str(stats.get("files_found", 0)))
    table.add_row("Files indexed", str(stats.get("files_indexed", 0)))
    table.add_row("Files skipped", str(stats.get("files_skipped", 0)))
    table.add_row("Symbols extracted", str(stats.get("symbols_extracted", 0)))
    table.add_row("Chunks embedded", str(stats.get("chunks_embedded", 0)))
    # Only when it happened. A permanent "Chunks repaired 0" row invites reading its
    # absence as "not checked", which is a different thing that the stats command reports.
    if stats.get("chunks_reconciled"):
        table.add_row("Chunks repaired", str(stats["chunks_reconciled"]))
    table.add_row("Elapsed", f"{elapsed:.1f}s")
    if stats.get("errors"):
        table.add_row("[red]Errors[/red]", f"[red]{stats['errors']}[/red]")
    console.print(table)

    if prune:
        # The DELETIONS run after the index — a prune wants the freshest possible view of
        # what is on disk, and the drift check pays for a second walk, which is why --prune
        # is opt-in rather than part of every run.
        #
        # But the GUARD is evaluated against `provenance_before`, captured above before
        # `index()` ran. index() writes provenance at the end of the run, so reading it here
        # would compare this run's walk config against a copy of itself and disarm three of
        # the six conditions. See _run_prune's docstring for the reproduction.
        _run_prune(
            config,
            indexer.db,
            indexer.vector_store,
            confirmed=yes,
            max_fraction=prune_max_percent / 100.0,
            provenance=provenance_before,
        )


# ---------------------------------------------------------------------------
# index --prune
# ---------------------------------------------------------------------------

# How many candidate paths to print before summarising the rest. A prune is authorised by
# a human reading this list, so it has to be readable; the count and the percentage below
# it carry the part that got elided.
_PRUNE_PREVIEW_LIMIT = 25


def _run_prune(
    config: IndexConfig,
    db: Database,
    vector_store: object | None,
    *,
    confirmed: bool,
    max_fraction: float,
    provenance: IndexProvenance,
) -> None:
    """Plan a prune, print it, and delete only when every guard licenses it.

    Exits nonzero on a refusal that BLOCKED something. A refused prune that exited 0 would
    tell a script that the deletions happened, which is the failure mode this whole feature
    was withheld over; but a refusal with an empty candidate list blocked nothing, and
    exiting 1 there made the first `--prune` on every index — the case with no provenance
    yet and nothing to delete — look like a failure.

    `provenance` MUST be the record as it stood BEFORE the index run that precedes this
    call, and is therefore passed in rather than read here. Reading it here is what the
    first version did, and it silently disarmed three of the six guards:

    `Indexer.index()` writes provenance at the END of the run, and this prune executes
    after that — so a locally-read record describes the run that just finished, and
    `walk_config_comparable`, `walk_config_changed` and the `trelix_version` check all
    compare the current walk config against a copy of itself. They could only ever pass.

    Reproduced end to end with a real Indexer over a temp repo: index three files, then
    re-run with one file edited and `vendored/` newly ignored. The two `vendored/` files
    are still on disk, and the plan came back with 2 candidates and **zero refusals**.
    On this repository that is the 35 files under `packages/`.

    It looked correct under test because `index()` returns early when nothing changed
    (`if not to_parse`), never reaching the provenance write — so with an unmodified tree
    the stale record is still there and the guard fires. One edited file removes that
    accident. The question the guard asks — "was the index I am about to prune built with
    the walk config I am pruning it with?" — is about the state before the run, so the
    pre-run snapshot is not a workaround, it is the correct input.
    """
    from trelix.store.provenance import compute_drift, plan_prune

    # The snapshot goes to BOTH. Two of the six guards (`walk_config_comparable`,
    # `walk_config_changed`) are read off the report, not off this argument, so passing it
    # only to plan_prune left them comparing the post-run record against itself — measured,
    # and it is what made the first version of this feature delete live files.
    report = compute_drift(config, db, provenance=provenance)
    plan = plan_prune(
        report,
        provenance,
        len(db.get_all_file_rel_paths()),
        max_fraction=max_fraction,
    )

    _print_prune_plan(plan, will_act=confirmed and not plan.is_refused)

    # An empty candidate list is checked BEFORE the refusals, and the order is the fix.
    # A refusal blocks a proposed deletion; with nothing proposed, nothing was blocked, and
    # exiting nonzero reports a failure for the ordinary outcome of the FIRST `--prune` on
    # any index — which records no provenance yet, so two guards have something to say about
    # a deletion that does not exist. A CI job reading the exit code fails on a healthy repo.
    if not plan.candidates:
        return
    if plan.is_refused:
        raise typer.Exit(1)
    if not confirmed:
        return

    repo_root = Path(config.repo_path)
    deleted = 0
    unmatched: list[str] = []
    for rel_path in plan.candidates:
        # The absolute path is reconstructed rather than read back out of `files.path`
        # because delete_file_by_path matches `path = ? OR rel_path = ?` — so a checkout
        # that has moved since it was indexed still deletes on the rel_path arm.
        if db.delete_file_by_path(str(repo_root / rel_path), rel_path, vector_store):
            deleted += 1
        else:
            unmatched.append(rel_path)

    console.print(
        f"[green]Pruned {deleted} file(s)[/green] — rows, symbols, chunks and embeddings."
    )
    if unmatched:
        # Loud, and nonzero. The plan came from `files.rel_path` seconds earlier, so a row
        # that cannot be found by the same key now means the delete and the plan disagree
        # about identity, and the remaining embeddings are orphaned rather than removed.
        err_console.print(
            f"[red]{len(unmatched)} planned deletion(s) matched no index row[/red] "
            f"({_safe_text(', '.join(unmatched[:5]))}"
            f"{' …' if len(unmatched) > 5 else ''}) — the plan and the index disagree. "
            "Run `trelix stats --drift` before trusting this index."
        )
        raise typer.Exit(1)


def _prune_from_index(config: IndexConfig, *, confirmed: bool, max_fraction: float) -> None:
    """Plan a prune against the index as it already stands, without indexing first.

    Only reachable from `--dry-run`, so nothing is deleted and no vector store is needed.

    Opened `read_only=True`, which is load-bearing rather than tidy. A normal `Database(...)`
    runs `init_schema()` — DDL, migrations, and a `PRAGMA user_version` stamp — so on any
    pre-3.1.2 index this path used to commit a `DELETE` and a schema write while telling the
    user it was a dry run. The cost-preview path in this same file already opened
    `file:…?mode=ro` for exactly this reason; the two disagreed. SQLite now enforces it: a
    stray write raises instead of succeeding.

    The provenance record is read here rather than snapshotted, and that is correct on this
    path: no index run precedes it, so what is in the database IS the pre-run state.
    """
    from trelix.store.db import Database as _Database
    from trelix.store.provenance import read_provenance

    db_path = config.db_path_absolute
    if not db_path.exists():
        err_console.print(
            f"[yellow]No index at {_safe_text(str(db_path))}[/yellow] — nothing to prune. "
            "Run `trelix index` first."
        )
        return

    with _Database(db_path, read_only=True) as db:
        _run_prune(
            config,
            db,
            None,
            confirmed=confirmed,
            max_fraction=max_fraction,
            provenance=read_provenance(db),
        )


def _print_prune_plan(plan: PrunePlan, *, will_act: bool) -> None:
    """Render what a prune would remove, then every reason it may not.

    The list comes first and the refusals last, deliberately: the refusals name what to fix
    and the fix depends on what is in the list (35 phantom deletions all under `packages/`
    is a different problem from one file under an unreadable directory).
    """
    if not plan.candidates:
        console.print(
            "\n[green]Nothing to prune:[/green] every indexed file was yielded by this walk."
        )
    else:
        table = Table(
            title="Indexed but no longer in the repository",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Path")
        for rel_path in plan.candidates[:_PRUNE_PREVIEW_LIMIT]:
            # Repository-controlled text: a directory named "old[1]" renders wrong, and one
            # named "old[/1]" raises MarkupError, taking the whole listing with it.
            table.add_row(_safe_text(rel_path))
        console.print(table)
        if len(plan.candidates) > _PRUNE_PREVIEW_LIMIT:
            console.print(
                f"[dim]… and {len(plan.candidates) - _PRUNE_PREVIEW_LIMIT} more not shown.[/dim]"
            )
        console.print(
            f"[bold]{len(plan.candidates)} of {plan.indexed_count} indexed file(s)[/bold] "
            f"({plan.fraction_of_index:.1%} of the index)."
        )

    # Only when something was proposed. Every refusal is a sentence about deleting files, so
    # printing them under "Nothing to prune" describes a deletion that was never on the
    # table — and "Prune refused" on a healthy index is how a user learns to stop reading it.
    if plan.candidates:
        for refusal in plan.refusals:
            # Escaped as a whole: these sentences carry repo-controlled paths and setting
            # values interpolated by plan_prune.
            err_console.print(f"[red]Prune refused:[/red] {_safe_text(refusal)}")

    if plan.candidates and not plan.refusals and not will_act:
        console.print(
            "[yellow]Nothing was deleted[/yellow] — `--prune` previews by default. "
            "Re-run with `--prune --yes` to remove the file(s) above and their embeddings."
        )


# ---------------------------------------------------------------------------
# index --dry-run  (cost preview)
# ---------------------------------------------------------------------------

# USD per 1,000,000 input tokens at each provider's published list price, checked
# 2026-08. A provider/model pair absent from here is reported as "unknown" rather than
# defaulted or interpolated: this table exists so someone can decide whether to spend the
# money, and a stale-but-plausible number is indistinguishable from a checked one.
#
# Azure is deliberately absent and unpriceable from config: it bills per deployment and
# tier, and `azure_embeddings_deployment` is a name chosen by whoever created the
# deployment, so it carries no rate at all.
_EMBED_PRICE_PER_MTOK_USD: dict[tuple[str, str], float] = {
    ("openai", "text-embedding-3-large"): 0.13,
    ("openai", "text-embedding-3-small"): 0.02,
    ("openai", "text-embedding-ada-002"): 0.10,
    ("voyage", "voyage-code-3"): 0.18,
}

# Providers that never make a billed call. The cost is real — CPU/GPU time and disk — but
# it is not a token bill, and reporting $0.00 for them would read as "free".
_LOCAL_EMBED_PROVIDERS = frozenset({"local", "local-code", "bge-code", "nomic-code"})

# Which EmbedderConfig field holds the model name for each provider. There is no single
# `model` attribute, so this is the mapping the price lookup keys on.
_EMBED_MODEL_FIELDS = {
    "openai": "openai_model",
    "azure": "azure_embeddings_deployment",
    "voyage": "voyage_model",
    "local": "local_model",
    "local-code": "local_code_model",
    "bge-code": "bge_code_model",
    "nomic-code": "nomic_code_model",
    "bedrock-titan": "bedrock_titan_model",
    "bedrock-cohere": "bedrock_cohere_model",
}


def _embed_model_name(embedder: EmbedderConfig) -> str:
    field = _EMBED_MODEL_FIELDS.get(str(embedder.provider))
    return str(getattr(embedder, field, "") or "") if field else "unknown"


def _print_cost_preview(config: IndexConfig) -> None:
    """Report what a real run would embed and what it would cost, embedding nothing.

    `scripts/self-index.sh --dry-run` already answers "which files would be indexed" and
    where they live, so this deliberately does not repeat that breakdown. What it adds is
    the number that decides whether to start: tokens, counted with the chunker's own
    tiktoken encoder over the chunk text the chunker actually builds, priced where the rate
    is known and reported as unknown where it is not.

    The count is an UPPER BOUND for files that are already indexed and have changed: the
    real run re-embeds only the symbols whose content hash moved, and reproducing that
    needs the symbol rows this preview declines to write. For a first index of a repo the
    two agree.
    """
    from trelix.indexing.chunker import Chunker
    from trelix.indexing.indexer import Indexer
    from trelix.indexing.walker import FileWalker

    walker = FileWalker(config)
    files = list(walker.walk())

    if not walker.walk_was_complete:
        # Same disclosure the real indexer makes: an unreadable directory drops its whole
        # subtree, so an estimate built on this walk under-counts by an unknown amount.
        skipped = walker.incomplete_paths
        err_console.print(
            f"[yellow]{len(skipped)} path(s) could not be read and are missing from this "
            f"estimate: {_safe_text(', '.join(skipped[:5]))}"
            f"{' …' if len(skipped) > 5 else ''}[/yellow]"
        )

    to_embed, index_read_error = _files_needing_embedding(config, files)
    repair_count, repair_tokens, repair_exhausted, repair_note = _repair_cost(config)

    chunker = Chunker(config.chunker)
    chunk_count = 0
    token_count = 0
    no_symbols = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Chunking (no embedding)…", total=len(to_embed))
        for file in to_embed:
            progress.advance(task)
            try:
                # Called unbound on purpose. `_parse_one` touches no instance state, and
                # constructing an Indexer would load an embedding model — the one thing a
                # cost preview must not do. Borrowing it also means the preview cannot
                # drift from the parse whose cost it is estimating; a private copy here
                # would silently mis-count the line-window fallback.
                parsed = Indexer._parse_one(cast("Indexer", None), file)
            except Exception as exc:
                no_symbols += 1
                logging.getLogger("trelix.cli").debug(
                    "Dry run could not parse %s: %s", file.rel_path, exc
                )
                continue
            if parsed.skipped or parsed.parse_result is None:
                no_symbols += 1
                continue
            symbols = parsed.parse_result.symbols
            chunks = chunker.build_chunks(
                symbols=symbols,
                imports=parsed.parse_result.import_edges,
                file_rel_path=file.rel_path,
                language=file.language.value,
                # `parent_id` is still a local index at this point — Phase 2 is what
                # remaps it to DB ids — so the parent map is keyed the same way. Keyed by
                # DB id here instead, every method would lose its "# Class:" header line
                # and the token count with it.
                parent_symbols=dict(enumerate(symbols)),
            )
            chunk_count += len(chunks)
            token_count += sum(c.token_count for c in chunks)

    table = Table(
        title="Cost preview (nothing embedded)", show_header=True, header_style="bold cyan"
    )
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_row("Files walked", f"{len(files):,}")
    if not walker.walk_was_complete:
        table.add_row("[yellow]Paths unreadable[/yellow]", f"{len(walker.incomplete_paths):,}")
    table.add_row("Unchanged since last index", f"{len(files) - len(to_embed):,}")
    table.add_row("Files to embed", f"{len(to_embed):,}")
    if no_symbols:
        table.add_row("Files yielding no symbols", f"{no_symbols:,}")
    # Repair work is reported on its own two rows and NOT folded into "Chunks to embed" /
    # "Embedding tokens", which keep their from-scratch meaning. It reaches the dollar
    # figure below though: a preview that under-quotes the run it is pricing is the same
    # class of dishonesty as an index that reports itself complete while 10.5% blind.
    if repair_note is not None:
        table.add_row("[yellow]Chunks missing vectors[/yellow]", f"[yellow]{repair_note}[/yellow]")
    else:
        table.add_row("Chunks missing vectors", f"{repair_count:,}")
        table.add_row("Repair tokens", f"{repair_tokens:,}")
    table.add_row("Chunks to embed", f"{chunk_count:,}")
    table.add_row("Embedding tokens", f"{token_count:,}")
    console.print(table)

    if index_read_error is not None:
        console.print(
            f"[yellow]The existing index could not be read[/yellow] "
            f"({_safe_text(index_read_error)}), so every walked file is counted as new — "
            "this is an over-estimate, not an under-estimate."
        )

    if repair_note is not None:
        # Named because the direction matters and it is the unsafe one: the repair is priced
        # at 0 here, and a real run will still perform it. Cross-backend repair pricing would
        # mean opening a Lance/Qdrant handle, which creates on open — so the honest move is to
        # label the figure, not to make the preview a writer.
        console.print(
            "[yellow]The repair could not be priced[/yellow] "
            f"({_safe_text(repair_note)}), so the estimate below is a LOWER BOUND — a real "
            "run may also re-embed chunks left without a vector by an interrupted one."
        )

    if repair_exhausted:
        # A sentence rather than a table row, because there is nothing to price: no real run
        # will embed these, so a priced row would quote a spend that cannot happen. Silence
        # was the wrong answer though — the rows above then read as a clean bill of health
        # for chunks `trelix stats` reports in red, and the two commands answer the same
        # question.
        console.print(
            f"[red]{repair_exhausted:,} chunk(s) are past the id-space limit[/red] and are "
            "NOT in the figures above — their `chunks.id` reached the vector store's "
            "sub-chunk sentinel offset, so any vector stored for them is filtered out of "
            "every search. No real run will embed them either: they need a re-key, not a "
            "re-embed. Run [bold]trelix stats[/bold] for the count and the remedy."
        )

    _print_cost_estimate(config, token_count + repair_tokens)
    _print_cost_caveats(config)


def _files_needing_embedding(
    config: IndexConfig, files: list[IndexedFile]
) -> tuple[list[IndexedFile], str | None]:
    """Split off the files a real run would skip on an unchanged content hash.

    Read through a read-only sqlite3 connection rather than `Database`, whose constructor
    runs schema migrations: a cost preview must not be able to modify the index it prices.
    Returns the files to embed plus the reason the index could not be consulted, if any —
    reported rather than swallowed, because "no index readable" and "index fully current"
    are opposite answers that would otherwise look the same.
    """
    import sqlite3

    if not config.incremental:
        return list(files), None

    db_path = config.db_path_absolute
    if not db_path.exists():
        return list(files), None

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT rel_path, hash FROM files")
            stored = {str(r[0]): str(r[1]) for r in rows}
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return list(files), str(exc)

    return [f for f in files if stored.get(f.rel_path) != f.hash], None


def _readonly_vec_conn(db_path: Path) -> sqlite3.Connection:
    """A connection to `db_path` that CANNOT write it, with sqlite-vec loaded.

    Answers "which chunks have a vector" without a width, which is what makes it usable on
    the state a killed run leaves: `index_metadata` empty, so no dimension to open a store
    at. Every part of it is load-bearing:

      * `mode=ro` + `PRAGMA query_only=ON` rather than `Database` or `SQLiteVectorStore`.
        `Database.__init__` runs schema migrations and `SQLiteVectorStore.__init__` issues
        `CREATE VIRTUAL TABLE IF NOT EXISTS … FLOAT[{dim}]` — so opening a store is the one
        thing that can create a permanently-wrong table where none exists, since a vec0
        table's width is fixed at creation (see `SQLiteVectorStore.recreate`).
      * `pathname2url` before interpolating the path, copied from `scripts/verify-index.sh`
        along with the reason: raw, a `#` in the path ends the URI as a fragment and a `?`
        opens a second query section, so `mode=ro` is silently dropped and SQLite creates
        an empty database at the truncated path — every count then reads 0 and a
        10.5%-blind index reads as healthy.
      * sqlite-vec must be loaded even to project a column: `chunk_embeddings` is a vec0
        virtual table and is unreachable without the extension.

    Raises rather than returning None — every caller has to render "not checked" with the
    exception type anyway, and a None here would read as "no vectors".
    """
    import sqlite3
    import urllib.request

    import sqlite_vec

    uri = f"file:{urllib.request.pathname2url(str(db_path))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception:
        conn.close()
        raise
    return conn


def _stored_chunk_ids_sql(conn: sqlite3.Connection) -> set[int]:
    """`SQLiteVectorStore.stored_chunk_ids()` against a raw connection, sentinels excluded.

    Duplicated as SQL rather than reached through the store class on purpose: the store
    cannot be constructed without a width, and this exists precisely for the case where
    there is none. The sentinel bound comes from the class attribute so it cannot drift.
    """
    # Imported for the sentinel offset only — reading a class attribute constructs no
    # store, and the alternative is a magic 10_000_000 that silently drifts from the one
    # every backend now filters on.
    from trelix.store.vector import BaseVectorStore

    return {
        int(r[0])
        for r in conn.execute(
            "SELECT chunk_id FROM chunk_embeddings WHERE chunk_id > 0 AND chunk_id < ?",
            (BaseVectorStore._SUB_CHUNK_OFFSET,),
        )
    }


def _repair_cost(config: IndexConfig) -> tuple[int, int, int, str | None]:
    """Price the chunks a real run would re-embed to repair an interrupted one.

    Returns (chunk count, token count, id-space-exhausted count, reason it could not be
    checked). The reason is non-None exactly when the numbers are meaningless, so the
    caller renders a sentence instead of a zero — "no holes" and "never looked" must not
    print the same.

    The exhausted count is returned SEPARATELY rather than dropped, which is what this used
    to do: those ids are excluded from the bill (see below) and were then excluded from the
    output too, so the preview printed "Chunks missing vectors 0 / Repair tokens 0" — a
    clean bill of health — for an index `trelix stats` reports in red. It is not a priced
    row because there is nothing to price; the caller states it under the table.

    STRICTLY READ-ONLY via `_readonly_vec_conn` — a cost preview must not be able to modify
    the index it prices. lance / qdrant are reported as not checked rather than opened:
    `LanceVectorStore._get_or_create_table` creates the table when the open fails and
    `QdrantVectorStore._ensure_collection` creates a collection; both are writes.

    Chunk ids at or above the sub-chunk offset are excluded from the bill for the same
    reason `Indexer._chunks_missing_vectors` excludes them from the repair: no real run will
    embed them, so pricing them would quote a spend that cannot happen.

    Tokens come from `chunks.token_count`, already stored, so this needs no parse and no
    chunker call — and it is the same number Phase 3 will send.
    """
    from trelix.store.vector import BaseVectorStore

    backend = str(getattr(config.store, "backend", "sqlite"))
    if backend != "sqlite":
        return 0, 0, 0, f"not checked ({backend} backend)"

    db_path = config.db_path_absolute
    if not db_path.exists():
        return 0, 0, 0, None  # No index at all: nothing indexed, so nothing half-indexed.

    try:
        conn = _readonly_vec_conn(db_path)
        try:
            stored = _stored_chunk_ids_sql(conn)
            rows = conn.execute("SELECT id, token_count FROM chunks").fetchall()
        finally:
            conn.close()
    except Exception as exc:
        return 0, 0, 0, f"not checked ({type(exc).__name__})"

    offset = BaseVectorStore._SUB_CHUNK_OFFSET
    missing = [
        (int(r[0]), int(r[1])) for r in rows if int(r[0]) < offset and int(r[0]) not in stored
    ]
    exhausted = sum(1 for r in rows if int(r[0]) >= offset)
    return len(missing), sum(t for _, t in missing), exhausted, None


def _print_cost_estimate(config: IndexConfig, token_count: int) -> None:
    """Price the token count, or say plainly that this provider cannot be priced."""
    provider = str(config.embedder.provider)
    model = _embed_model_name(config.embedder)
    label = _safe_text(f"{provider} / {model}")

    if provider in _LOCAL_EMBED_PROVIDERS:
        console.print(
            f"[bold]Estimated cost[/bold] none — {label} runs locally, so there is no API "
            "cost. The spend is CPU/GPU time and disk, which this does not estimate."
        )
    elif (price := _EMBED_PRICE_PER_MTOK_USD.get((provider, model))) is not None:
        console.print(
            f"[bold]Estimated cost[/bold] ${token_count / 1_000_000 * price:.4f} — "
            f"{token_count:,} tokens at ${price:.2f} per 1M tokens ({label})."
        )
    else:
        console.print(
            f"[bold]Estimated cost[/bold] unknown — trelix has no checked list price for "
            f"{label}, and a guessed rate is worse than none. {token_count:,} tokens would "
            "be sent; price them against that provider's own published rate."
        )

    # Stated for every provider, not only the ones it is wrong for. cl100k_base is the
    # chunker's encoder, so it is exactly what a real run's Phase 3 reports — but only
    # OpenAI and Azure bill on it, and Voyage / Titan / Cohere tokenize differently.
    console.print(
        "[dim]Tokens counted with cl100k_base, the encoder the chunker uses. Providers "
        "outside OpenAI/Azure bill on their own tokenizer, so their invoice will differ "
        "from this count.[/dim]"
    )


def _print_cost_caveats(config: IndexConfig) -> None:
    """Name the spend this preview does NOT include, one line per enabled feature."""
    console.print(
        "[dim]Files already indexed and changed are counted whole: the real run re-embeds "
        "only the symbols whose content hash moved, so the figure above is an upper bound "
        "for them.[/dim]"
    )
    if config.chunker.contextual:
        console.print(
            "[yellow]Contextual chunking is enabled[/yellow] — every chunk also costs one "
            f"LLM call ({_safe_text(config.chunker.contextual_model)}), which is not priced "
            "above."
        )
    if config.file_summaries_enabled:
        console.print(
            "[yellow]File summaries are enabled[/yellow] — one LLM call per file on top of "
            "the embeddings above, not priced here."
        )
    if config.chunker.multi_granularity_enabled:
        console.print(
            "[yellow]Multi-granularity indexing is enabled[/yellow] — it embeds once per "
            "sub-symbol as well, bypassing the token batching, and those embeddings are "
            "NOT in the count above."
        )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@app.command()
def search(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    query: str = typer.Argument(..., help="Natural language query"),
    provider: str | None = typer.Option(None, help=_PROVIDER_HELP),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Search for code — returns ranked results as a table or JSON"""
    _setup_logging(False)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import IndexConfig, RetrievalConfig
    from trelix.retrieval.retriever import Retriever

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=_build_embedder_config(provider),
            retrieval=RetrievalConfig(rerank=False),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        _print_error("Configuration error", detail)
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    try:
        retriever = Retriever(config)
        context = retriever.retrieve(query)
    except Exception as exc:
        _print_error("Search failed", exc)
        raise typer.Exit(1) from exc

    if json_output:
        # NOT escaped, deliberately: these values go into a JSON document
        # emitted with builtin print(), which never interprets markup. Escaping
        # here would corrupt the machine-readable contract by writing stray
        # backslashes into consumers' parsed strings.
        results_json = []
        for r in context.results:
            results_json.append(
                {
                    "file": r.file.rel_path,
                    "symbol": r.symbol.name,
                    "lines": f"{r.symbol.line_start}-{r.symbol.line_end}",
                    "score": round(r.score, 4),
                }
            )
        print(json.dumps({"status": "ok", "results": results_json}))
        return

    if not context.results:
        console.print("[yellow]No results found.[/yellow]")
        return

    # The Rich path is the opposite case: title and cells are markup-parsed, and
    # rel_path / symbol.name are indexed-repository text — a "[/!]" anywhere in
    # them raised MarkupError and rendered zero rows. See the markup-safety note
    # near the top of this file. `query` is escaped too: searching for the very
    # regex that triggers this bug ("trelix search . '[/!]'") must not crash.
    table = Table(title=f"Search: {_safe_text(query)}", show_header=True, header_style="bold cyan")
    table.add_column("File", style="dim", max_width=40)
    table.add_column("Symbol", style="bold")
    table.add_column("Lines", justify="right")
    table.add_column("Score", justify="right")

    for r in context.results:
        table.add_row(
            _safe_text(r.file.rel_path),
            _safe_text(r.symbol.name),
            f"{r.symbol.line_start}-{r.symbol.line_end}",
            f"{r.score:.4f}",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


@app.command()
def ask(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    query: str = typer.Argument(..., help="Question to answer about the codebase"),
    provider: str | None = typer.Option(None, help=_PROVIDER_HELP),
    agentic: Annotated[
        bool, typer.Option("--agentic", help="Enable multi-turn agentic ReAct loop.")
    ] = False,
    session: Annotated[
        str | None,
        typer.Option(
            "--session", help="Resume a persisted agent session by ID (implies --agentic)."
        ),
    ] = None,
) -> None:
    """Ask a question — retrieval + LLM synthesis (requires OPENAI_API_KEY for full synthesis)"""
    _setup_logging(False)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import IndexConfig, RetrievalConfig
    from trelix.retrieval.retriever import Retriever
    from trelix.retrieval.synthesizer import Synthesizer

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=_build_embedder_config(provider),
            retrieval=RetrievalConfig(rerank=False),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        _print_error("Configuration error", detail)
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    # --agentic flag overrides the config field; --session implies --agentic
    if session is not None or agentic:
        config.retrieval.agentic_enabled = True

    try:
        if config.retrieval.agentic_enabled:
            from trelix.agent import AgentLoop

            agent_loop = AgentLoop(config)
            answer, resolved_session = agent_loop.run(query, session_id=session)
            # Every answer/context blob printed in this command is escaped: an
            # LLM answer quotes the code it retrieved, and context_text below IS
            # that code verbatim. Rendering trelix's own rust.py line
            # `re.sub(r"^//[/!]?\s?", ...)` raised MarkupError, so `trelix ask`
            # printed nothing and exited nonzero. escape() is display-only —
            # nothing here is trelix-authored markup meant to be interpreted.
            console.print(_safe_text(answer))
            err_console.print(f"[dim]Session: {_safe_text(resolved_session)}[/dim]")
            return

        retriever = Retriever(config)
    except Exception as exc:
        _print_error("Retrieval failed", exc)
        raise typer.Exit(1) from exc

    try:
        synth = Synthesizer(config.embedder, llm_config=config.llm)
        if config.retrieval.flare_enabled:
            from trelix.retrieval.flare import FLARELoop

            loop = FLARELoop(retriever, synth, config)
            answer = loop.run(query)
            console.print(_safe_text(answer))
        else:
            context = retriever.retrieve(query)
            # If provider=local (no API key), print the context text directly.
            # Read the resolved provider off config, not the raw CLI arg — the
            # effective value may come from TRELIX_EMBEDDER_PROVIDER when
            # --provider wasn't passed.
            if config.embedder.provider == "local":
                console.print(
                    Panel(f"[bold cyan]Context for:[/bold cyan] {_safe_text(query)}", expand=False)
                )
                if context.context_text:
                    console.print(_safe_text(context.context_text))
                else:
                    console.print("[yellow]No relevant code found.[/yellow]")
                return
            # Escaped per token, not per answer: each print() renders in
            # isolation, so a bracket pair split across two tokens can never
            # form a tag, and a complete "[/!]" inside one token can no longer
            # abort the stream mid-answer.
            for token in synth.stream(context, config.retrieval):
                console.print(_safe_text(token), end="", highlight=False)
            console.print()  # final newline
    except Exception as exc:
        _print_error("Synthesis failed", exc)
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# query (human-readable, always Rich, no --json flag)
# ---------------------------------------------------------------------------


@app.command()
def query(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    query_str: str = typer.Argument(..., metavar="QUERY", help="Natural language query"),
    provider: str | None = typer.Option(None, help=_PROVIDER_HELP),
) -> None:
    """Query a repository — human-readable Rich terminal output (no LLM synthesis)"""
    _setup_logging(False)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import IndexConfig, RetrievalConfig
    from trelix.retrieval.retriever import Retriever

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=_build_embedder_config(provider),
            retrieval=RetrievalConfig(rerank=False),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        _print_error("Configuration error", detail)
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {_safe_text(query_str)}", expand=False))

    try:
        retriever = Retriever(config)
        context = retriever.retrieve(query_str)
    except Exception as exc:
        _print_error("Query failed", exc)
        raise typer.Exit(1) from exc

    console.print(
        f"\n[dim]Retrieved {len(context.results)} results "
        f"({context.total_tokens} tokens) in {context.elapsed_seconds:.3f}s[/dim]\n"
    )

    if not context.results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("File", style="dim", max_width=40)
    table.add_column("Symbol", style="bold")
    table.add_column("Lines", justify="right")
    table.add_column("Score", justify="right")

    # Indexed-repository text into markup-parsed cells — same sink as search().
    for r in context.results:
        table.add_row(
            _safe_text(r.file.rel_path),
            _safe_text(r.symbol.name),
            f"{r.symbol.line_start}-{r.symbol.line_end}",
            f"{r.score:.4f}",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------


@app.command("call-graph")
def call_graph(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    symbol: str = typer.Argument(..., help="Symbol name or module path to inspect"),
    provider: str | None = typer.Option(None, help=_PROVIDER_HELP),
    direction: str = typer.Option(
        "all",
        "--direction",
        "-d",
        help="callers | callees | importers | all",
    ),
) -> None:
    """Show call graph and import edges for a symbol or module."""
    _setup_logging(False)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import IndexConfig, RetrievalConfig
    from trelix.core.models import SearchResult as _SearchResult
    from trelix.retrieval.retriever import Retriever

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=_build_embedder_config(provider),
            retrieval=RetrievalConfig(rerank=False),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        _print_error("Configuration error", detail)
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    try:
        retriever = Retriever(config)
    except Exception as exc:
        _print_error("Failed to open index", exc)
        raise typer.Exit(1) from exc

    console.print(f"\n[bold] Graph:[/bold] {_safe_text(symbol)}\n")

    def _render_table(title: str, results: list[_SearchResult]) -> None:
        tbl = Table(show_header=True, header_style="bold cyan", title=title)
        tbl.add_column("File", style="dim", max_width=45)
        tbl.add_column("Symbol", style="bold")
        tbl.add_column("Lines", justify="right")
        tbl.add_column("Kind")
        if results:
            # File/symbol are indexed-repository text and must be escaped;
            # `kind` is a closed SymbolKind enum validated at hydration, so it
            # is left alone. The "(none)" row below is trelix's own markup.
            for r in results:
                tbl.add_row(
                    _safe_text(r.file.rel_path),
                    _safe_text(r.symbol.qualified_name or r.symbol.name),
                    f"{r.symbol.line_start}-{r.symbol.line_end}",
                    r.symbol.kind.value if hasattr(r.symbol.kind, "value") else str(r.symbol.kind),
                )
        else:
            tbl.add_row("[dim](none)[/dim]", "", "", "")
        console.print(tbl)

    valid_directions = {"callers", "callees", "importers", "all"}
    if direction not in valid_directions:
        err_console.print(
            f"[red]Invalid direction[/red] {_safe_text(repr(direction))}. "
            f"Choose from: {', '.join(sorted(valid_directions))}"
        )
        raise typer.Exit(1)

    if direction in ("callers", "all"):
        callers = retriever.get_callers(symbol)
        _render_table(f"Callers ({len(callers)})", callers)

    if direction in ("callees", "all"):
        callees = retriever.get_callees(symbol)
        _render_table(f"Callees ({len(callees)})", callees)

    if direction in ("importers", "all"):
        importers = retriever.get_importers(symbol)
        _render_table(f'Importers of "{_safe_text(symbol)}" ({len(importers)})', importers)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@app.command()
def stats(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    drift: bool = typer.Option(
        False,
        "--drift",
        help=(
            "Also compare every indexable file on disk against its stored hash. "
            "Costs a full walk plus a SHA-256 per file, so it is off by default."
        ),
    ),
) -> None:
    """Show index statistics (files, symbols, chunks, DB size)"""
    _setup_logging(False)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

    try:
        config = IndexConfig(repo_path=str(Path(repo).resolve()))
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        _print_error("Configuration error", detail)
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    db_path = config.db_path_absolute
    if not db_path.exists():
        err_console.print(
            f"[red]No index found at {_safe_text(str(db_path))}[/red] —"
            f" run `trelix index {_safe_text(repo)}` first."
        )
        raise typer.Exit(1)

    from trelix.store.provenance import commits_since, compute_drift, read_provenance

    try:
        with Database(db_path) as db:
            conn = db._conn
            file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            symbol_count = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            db_size_bytes = db_path.stat().st_size
            provenance = read_provenance(db)
            coverage = _vector_coverage(config, db)
            # Computed inside the `with` because it reads per-file hashes back out of
            # this same connection.
            drift_report = compute_drift(config, db) if drift else None
    except Exception as exc:
        _print_error("Failed to read index", exc)
        raise typer.Exit(1) from exc

    db_size_kb = db_size_bytes / 1024

    console.print(Panel(f"[bold cyan]Index Stats:[/bold cyan] {_safe_text(repo)}", expand=False))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_row("Files indexed", str(file_count))
    table.add_row("Symbols", str(symbol_count))
    table.add_row("Chunks", str(chunk_count))
    _add_coverage_rows(table, coverage)
    table.add_row("DB size", f"{db_size_kb:.1f} KB")
    console.print(table)
    _print_coverage_remedy(coverage, repo, config)

    _print_provenance(provenance, commits_since(config, provenance.git_commit))
    if drift_report is not None:
        _print_drift(drift_report)


@dataclass
class _VectorCoverage:
    """What `trelix stats` can say about whether every chunk actually has a vector.

    `missing` / `orphaned` are None when the store could not be consulted, which is a
    different answer from 0 and is rendered as such. `dimension` is None when
    `index_metadata` holds no `embedding_dimension` — the state a killed run leaves, and
    the state in which `DimensionGuard.check` silently disarms itself. Note that a None
    `dimension` no longer implies unknown coverage: on the sqlite backend the two are
    answered independently (see `_vector_coverage`).

    `id_space_exhausted` counts chunk rows whose id reached the vector store's sub-chunk
    offset. Held apart from `missing` because no re-index can fix them — it is the one
    coverage number whose remedy is not `trelix index`.
    """

    missing: int | None = None
    orphaned: int | None = None
    with_vectors: int | None = None
    dimension: int | None = None
    id_space_exhausted: int | None = None
    unavailable_reason: str | None = None


def _vector_coverage(config: IndexConfig, db: Database) -> _VectorCoverage:
    """Count chunks with and without a vector, in both directions.

    A recorded dimension is NOT required, and that is the whole shape of this function.
    Requiring one made `stats` blind on exactly the index this reporting exists for: a
    killed run leaves `index_metadata` with 0 rows, so the command that is supposed to
    reveal the holes reported "not checked" instead, and the remedy line never printed.

    What the old gate was right about is narrower than it looked: never GUESS a width, and
    never call `make_vector_store` without one, because `SQLiteVectorStore.__init__` issues
    `CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings … FLOAT[{dim}]` and a vec0 table's
    width is fixed at creation (see `SQLiteVectorStore.recreate`), so a guess would leave a
    permanently-wrong table where none existed. A read-only projection of `chunk_id` needs
    no width at all and cannot create anything — `--dry-run` has priced the identical
    question that way since this feature landed — so for the sqlite backend the answer is
    read that way when there is no dimension to open a store with. lance / qdrant keep
    reporting "not checked": both create on open, and neither can be projected from SQL.

    `stats` already constructs `Database`, which runs schema migrations, so it is a writer
    already; the read-only path here is about not creating a vec0 table, not about the
    `--dry-run` contract.
    """
    from trelix.store.vector import make_vector_store

    partition = db.all_chunk_ids()
    exhausted = len(partition.id_space_exhausted)
    dimension = db.get_embedding_dimension()

    try:
        if isinstance(dimension, int):
            stored = make_vector_store(config, dimension).stored_chunk_ids()
        elif str(getattr(config.store, "backend", "sqlite")) != "sqlite":
            return _VectorCoverage(
                id_space_exhausted=exhausted,
                unavailable_reason="no embedding dimension recorded",
            )
        else:
            conn = _readonly_vec_conn(config.db_path_absolute)
            try:
                stored = _stored_chunk_ids_sql(conn)
            finally:
                conn.close()
    except Exception as exc:
        return _VectorCoverage(
            dimension=dimension if isinstance(dimension, int) else None,
            id_space_exhausted=exhausted,
            unavailable_reason=f"{type(exc).__name__}",
        )

    return _VectorCoverage(
        # Both directions, never netted: netting is what makes a single count lie, and the
        # reverse direction is the only signal for LanceDB's swallowed delete_batch.
        # `partition.repairable`, not every chunk row: an id past the sub-chunk offset is
        # unretrievable but is NOT a hole `trelix index` can fill, and counting it here is
        # what printed a remedy that could never clear.
        missing=len(partition.repairable - stored),
        orphaned=len(stored - partition.repairable),
        with_vectors=len(stored),
        dimension=dimension if isinstance(dimension, int) else None,
        id_space_exhausted=exhausted,
    )


def _add_coverage_rows(table: Table, coverage: _VectorCoverage) -> None:
    """Add the vector-coverage rows to the stats table."""
    if coverage.missing is None:
        why = _safe_text(coverage.unavailable_reason or "unknown")
        table.add_row(
            "[yellow]Chunks with vectors[/yellow]",
            f"[yellow]not checked — {why}[/yellow]",
        )
    else:
        table.add_row("Chunks with vectors", str(coverage.with_vectors))
        missing_style = "red" if coverage.missing else "green"
        table.add_row(
            f"[{missing_style}]Chunks missing vectors[/{missing_style}]",
            f"[{missing_style}]{coverage.missing}[/{missing_style}]",
        )
        orphan_style = "yellow" if coverage.orphaned else "green"
        table.add_row(
            f"[{orphan_style}]Vectors with no chunk row[/{orphan_style}]",
            f"[{orphan_style}]{coverage.orphaned}[/{orphan_style}]",
        )

    # Only when non-zero, unlike the rows above. Those three are reported at 0 because "0
    # holes" is the reassurance the command exists to give; this one is a condition no
    # normal index is in, and a permanent `0` row invites it to be read as a hole count.
    if coverage.id_space_exhausted:
        table.add_row(
            "[red]Chunks past the id-space limit[/red]",
            f"[red]{coverage.id_space_exhausted}[/red]",
        )

    if coverage.dimension is None:
        # The single row that makes the reproduced bug loud before any hole is counted:
        # `DimensionGuard.check` early-returns when the stored value is not an int, so this
        # index has no protection against mixed-width vectors either.
        table.add_row(
            "[yellow]Embedding dimension[/yellow]",
            "[yellow]not recorded — dimension guard disabled[/yellow]",
        )
    else:
        table.add_row("Embedding dimension", str(coverage.dimension))


def _vector_store_locations(config: IndexConfig) -> list[str]:
    """Every place a from-scratch rebuild has to clear, named per backend.

    `Indexer._partial_index_error` names the same set for the same reason: on lance and
    qdrant the vectors do not live in the index file, so "remove the index file" renumbers
    `chunks` from 1 and leaves every old vector behind. That does clear the id-space row —
    a fresh file has no `sqlite_sequence`, so AUTOINCREMENT restarts at 1 — but it strands
    every vector the old ids carried as a `Vectors with no chunk row` orphan that nothing
    reclaims. Reproduced on lancedb 0.33.0: 0 -> 4 orphans on a four-chunk fixture whose
    ids had climbed, which on a real index is the entire former vector set.
    """
    backend = str(getattr(config.store, "backend", "sqlite"))
    locations = [f"the index database ({config.db_path_absolute})"]
    if backend == "lance":
        locations.append(f"the LanceDB directory ({config.store.lance_uri})")
    elif backend == "qdrant":
        locations.append(
            f"the Qdrant collection {config.store.qdrant_collection} at {config.store.qdrant_url}"
        )
    return locations


def _print_coverage_remedy(coverage: _VectorCoverage, repo: str, config: IndexConfig) -> None:
    """Name the way out, but only when there is something to fix.

    Deliberately does not change the exit code. A non-zero status here would break every
    script that pipes `stats` — `tests/unit/test_cli_closed_stdout.py` exists because a
    non-zero exit on this command broke CI twice.

    Two remedies, never merged, because they are not the same instruction: holes are healed
    by `trelix index`, and id-space exhaustion is not healed by anything a user can type
    today. Printing one sentence for both would send someone to spend money on chunks no
    embedding can rescue.
    """
    if coverage.missing:
        console.print(
            f"\n[red]{coverage.missing} chunk(s) can never be retrieved[/red] — they have a "
            f"chunk row but no vector, which is what an interrupted index run leaves behind. "
            f"Run [bold]trelix index {_safe_text(repo)}[/bold] to re-embed those chunks "
            # "exactly those chunks and nothing else" was false: a plain index also embeds
            # every file whose content hash moved since the last run, which is its job.
            f"(on top of anything that changed since the last run), or "
            f"[bold]trelix index {_safe_text(repo)} --dry-run[/bold] to price it first."
        )
        # Stated because the alternative is discovering it by waiting. `trelix watch` never
        # reconciles — a save event on one file is the wrong trigger for a repo-wide store
        # scan, and an unbounded scan per filesystem event is the wrong cost profile for a
        # long-lived process — so someone who leaves watch running expects these to close
        # and they never do.
        console.print(
            "[dim]  `trelix watch` will not repair them: it only re-indexes files as they "
            "change, and never scans the store for holes.[/dim]"
        )

    if coverage.id_space_exhausted:
        from trelix.store.vector import BaseVectorStore

        console.print(
            f"\n[red]{coverage.id_space_exhausted} chunk(s) are past the id-space limit"
            f"[/red] — their `chunks.id` reached the vector store's sub-chunk sentinel "
            f"offset ({BaseVectorStore._SUB_CHUNK_OFFSET:,}), so any vector stored for them "
            f"is filtered out of every search as a sentinel. Re-indexing CANNOT fix this and "
            f"will not try: they need a re-key, not a re-embed. Rebuild the index from "
            f"scratch — remove {' and '.join(_vector_store_locations(config))}, then "
            f"[bold]trelix index {_safe_text(repo)}[/bold] — to renumber `chunks` from 1."
        )


def _print_provenance(provenance: IndexProvenance, behind: int | None) -> None:
    """Render what the index was built from, or say plainly that it is unknown."""
    if provenance.is_empty:
        console.print(
            "\n[yellow]Provenance:[/yellow] not recorded — this index predates "
            "provenance tracking. Re-index to capture it."
        )
        return

    table = Table(show_header=True, header_style="bold cyan", title="Built from")
    table.add_column("Field", style="dim")
    table.add_column("Value", justify="right")

    if provenance.git_commit:
        commit = provenance.git_commit[:12]
        # "unknown" rather than "up to date" when the count could not be computed: the
        # recorded commit may have been rebased away or garbage collected, and rendering
        # that as 0 would assert the index is current when nothing checked.
        if behind is None:
            suffix = " [dim](distance from HEAD unknown)[/dim]"
        elif behind == 0:
            suffix = " [green](= HEAD)[/green]"
        else:
            suffix = f" [yellow](HEAD is {behind} commit(s) ahead)[/yellow]"
        table.add_row("Commit", _safe_text(commit) + suffix)
    if provenance.git_branch:
        table.add_row("Branch", _safe_text(provenance.git_branch))
    if provenance.git_dirty is not None:
        table.add_row(
            "Worktree at index time",
            "[yellow]had uncommitted changes[/yellow]" if provenance.git_dirty else "clean",
        )
    if provenance.indexed_at:
        table.add_row("Indexed at", _safe_text(provenance.indexed_at))
    if provenance.trelix_version:
        table.add_row("trelix version", _safe_text(provenance.trelix_version))
    if provenance.embedder_provider:
        model = f" / {provenance.embedder_model}" if provenance.embedder_model else ""
        table.add_row("Embedder", _safe_text(provenance.embedder_provider + model))

    console.print(table)


def _print_drift(report: DriftReport) -> None:
    """Render file-level drift, and refuse to present `missing` as actionable.

    A truncated walk yields the same `missing` list as a set of genuinely deleted files.
    Anyone acting on that count would delete embeddings that cost money to recompute, so
    an incomplete walk is stated before the numbers rather than after them — and, when
    `missing_is_trustworthy` is False, the headline total (`actionable_drifted_count`) and
    the remediation sentence are both gated on it as well. Printing a warning and then a
    number that ignores it is worse than printing neither: it reads as a rounding
    disagreement rather than as a limit on what was measured.
    """
    if not report.walk_was_complete:
        shown = ", ".join(report.incomplete_paths[:5])
        console.print(
            f"\n[yellow]Warning:[/yellow] {len(report.incomplete_paths)} path(s) could not "
            f"be read during this walk ({_safe_text(shown)}"
            f"{' …' if len(report.incomplete_paths) > 5 else ''}). Files under them are "
            "counted as [bold]missing[/bold] below even though they may be present — do "
            "not prune on this result."
        )

    if report.walk_config_changed:
        console.print(
            "\n[yellow]Warning:[/yellow] this walk used different settings than the index "
            "was built with, so [bold]missing[/bold] below includes files the walk simply "
            "no longer reaches — not deletions."
        )
        diff_table = Table(show_header=True, header_style="bold yellow")
        diff_table.add_column("Setting", style="dim")
        diff_table.add_column("At index time")
        diff_table.add_column("Now")
        for name, recorded, current in report.walk_config_diff:
            diff_table.add_row(
                _safe_text(name), _safe_text(recorded[:60]), _safe_text(current[:60])
            )
        console.print(diff_table)
        console.print(
            "[dim]TRELIX_WALKER_* is read from the process environment only, never from "
            ".env, and EXTRA_IGNORE_DIRS replaces the default list rather than extending "
            "it. Re-run with the same environment that built the index — "
            "`scripts/self-index.sh` is the reference.[/dim]"
        )
    elif not report.walk_config_comparable:
        console.print(
            "\n[dim]This index predates walk-config recording, so whether the walk used "
            "the same ignore rules could not be checked. Treat [bold]missing[/bold] as "
            "unverified.[/dim]"
        )

    if report.is_clean:
        console.print(
            f"\n[green]No drift:[/green] all {report.unchanged_count} indexable files "
            "match their stored hashes."
        )
        return

    table = Table(show_header=True, header_style="bold cyan", title="Worktree drift")
    table.add_column("State", style="dim")
    table.add_column("Files", justify="right")
    table.add_column("Examples")

    # The row label carries the caveat too. A bare "Indexed but not found  35" beside a
    # total of 64 reads as an arithmetic bug rather than a deliberate exclusion.
    missing_label = (
        "Indexed but not found"
        if report.missing_is_trustworthy
        else "Indexed but not found (unverified)"
    )
    for label, paths in (
        ("Changed since indexing", report.stale),
        ("Not indexed yet", report.new),
        (missing_label, report.missing),
    ):
        if paths:
            examples = ", ".join(paths[:3]) + (" …" if len(paths) > 3 else "")
            table.add_row(label, str(len(paths)), _safe_text(examples))
    table.add_row("Unchanged", str(report.unchanged_count), "")
    console.print(table)

    if report.missing_is_trustworthy:
        console.print(
            f"[yellow]{report.drifted_count} file(s) have drifted.[/yellow] "
            "Run `trelix index` to rebuild, or `trelix update-index <file>` per file."
        )
        return

    # Neither the total nor the remedy may contradict the warning printed above. Measured
    # on this repository: "99 file(s) have drifted. Run `trelix index` to rebuild" where
    # 35 of the 99 were every file under `packages/`, all present on disk — `packages` is
    # in the default `extra_ignore_dirs` for .NET NuGet output, so that walk could not
    # reach them. A bare `trelix index` repeats the same walk and so restores none of
    # them; the number was inflated and the advice was void.
    console.print(
        f"[yellow]{report.actionable_drifted_count} file(s) have drifted.[/yellow] "
        "Run `trelix index` to rebuild, or `trelix update-index <file>` per file."
    )
    if report.missing:
        console.print(
            f"[yellow]The {len(report.missing)} 'not found' file(s) are excluded from "
            "that total[/yellow] — see the warning above; they may all be present. "
            "`trelix index` will [bold]NOT[/bold] restore them: it repeats the same walk, "
            "which cannot reach them, so it has nothing to re-index. Fix the walk first "
            "(same environment and ignore rules as the index), then re-run --drift to "
            "find out whether any of them were really deleted."
        )


# ---------------------------------------------------------------------------
# link-tickets
# ---------------------------------------------------------------------------


@app.command("link-tickets")
def link_tickets(
    repo: str = typer.Argument(..., help="Path to the indexed repository (must be a git repo)"),
    # All three default to None so an omitted flag can be told apart from an explicit
    # one. Passing a typer default unconditionally into GitLinkerConfig() made the
    # kwarg win over pydantic-settings, so TRELIX_GIT_LINKER_TICKET_PATTERN,
    # _MAX_COMMITS and _SINCE were inert on the only path that invokes this — while
    # docs/CONFIGURATION.md documents all three as working. `_build_embedder_config`
    # above already solves this exact bug class the same way: omit the kwarg and let
    # the settings loader fall through to the environment.
    max_commits: int | None = typer.Option(
        None, help="Max commits to walk (bounds cost on large repos) [default: 5000]"
    ),
    since: str | None = typer.Option(
        None,
        help='Only walk commits after this date, e.g. "90 days ago" (passed to git log --since)',
    ),
    ticket_pattern: str | None = typer.Option(
        None,
        help='Regex for ticket IDs in commit messages (default: Jira-style "PROJ-123", '
        "excluding technical constants like UTF-8 and CVE-2021-44228)",
    ),
) -> None:
    """
    Walk git history to link code symbols to ticket references found in
    commit messages (e.g. "PROJ-123") — feeds cross-source PageRank.

    A separate, slower pass from `trelix index` — run it after indexing, on
    an already-indexed repo. Requires *repo* to be a real git checkout;
    non-git directories or shallow clones degrade gracefully to 0 links.
    """
    _setup_logging(False)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import GitLinkerConfig, IndexConfig
    from trelix.indexing.git_linker import GitLinker
    from trelix.store.db import Database

    try:
        config = IndexConfig(repo_path=str(Path(repo).resolve()))
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        _print_error("Configuration error", detail)
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    db_path = config.db_path_absolute
    if not db_path.exists():
        err_console.print(
            f"[red]No index found at {_safe_text(str(db_path))}[/red] —"
            f" run `trelix index {_safe_text(repo)}` first."
        )
        raise typer.Exit(1)

    # Only what the user actually passed, so an unset flag leaves the field to the
    # settings loader (env var, .env, then the field default).
    linker_overrides: dict[str, object] = {"enabled": True}
    if max_commits is not None:
        linker_overrides["max_commits"] = max_commits
    if since is not None:
        linker_overrides["since"] = since
    if ticket_pattern is not None:
        linker_overrides["ticket_pattern"] = ticket_pattern
    linker_config = GitLinkerConfig(**linker_overrides)  # type: ignore[arg-type]

    try:
        with Database(db_path) as db:
            with console.status("[bold cyan]Walking git history…[/bold cyan]"):
                count = GitLinker(db, linker_config).link(config.repo_path)
    except Exception as exc:
        _print_error("Failed to link tickets", exc)
        raise typer.Exit(1) from exc

    if count == 0:
        console.print(
            "[yellow]No ticket references linked.[/yellow] Either this isn't a git "
            "repo, no commits matched the ticket pattern, or no touched files are "
            "indexed yet."
        )
    else:
        console.print(f"[green]Linked {count} symbol-ticket edge(s).[/green]")


# ---------------------------------------------------------------------------
# link-artifacts
# ---------------------------------------------------------------------------


@app.command("link-artifacts")
def link_artifacts(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    embedding_fallback: bool = typer.Option(
        False,
        help="For artifacts with no regex match, fall back to embedding "
        "similarity against indexed chunks (costs one embed call per "
        "unmatched artifact)",
    ),
    similarity_threshold: float = typer.Option(
        0.75, help="Minimum similarity for an embedding-fallback match (0.0-1.0)"
    ),
) -> None:
    """
    Scan connector-fetched artifacts (Jira tickets, TestRail cases, ...) for
    mentions of indexed symbols — feeds cross-source PageRank the same way
    `trelix link-tickets` does for git commit messages.

    `trelix connector sync` only writes to the artifacts table; it never
    creates generic_edges on its own. Run this after syncing to make synced
    artifacts reachable from the code graph.
    """
    from trelix.core.config import ArtifactLinkerConfig, IndexConfig
    from trelix.indexing.artifact_linker import ArtifactLinker
    from trelix.store.db import Database

    try:
        config = IndexConfig(repo_path=str(Path(repo).resolve()))
    except (ValueError, FileNotFoundError) as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    db_path = config.db_path_absolute
    if not db_path.exists():
        err_console.print(
            f"[red]No index found at {_safe_text(str(db_path))}[/red] —"
            f" run `trelix index {_safe_text(repo)}` first."
        )
        raise typer.Exit(1)

    linker_config = ArtifactLinkerConfig(
        embedding_fallback_enabled=embedding_fallback,
        similarity_threshold=similarity_threshold,
    )

    try:
        with Database(db_path) as db:
            with console.status("[bold cyan]Scanning artifacts…[/bold cyan]"):
                count = ArtifactLinker(db, linker_config, index_config=config).link()
    except Exception as exc:
        _print_error("Failed to link artifacts", exc)
        raise typer.Exit(1) from exc

    if count == 0:
        console.print(
            "[yellow]No artifact references linked.[/yellow] Either no artifacts "
            "have been synced yet (run `trelix connector sync`), or none mention "
            "an indexed symbol by name."
        )
    else:
        console.print(f"[green]Linked {count} symbol-artifact edge(s).[/green]")


# ---------------------------------------------------------------------------
# update-index
# ---------------------------------------------------------------------------


@app.command("update-index")
def update_index(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    file: str = typer.Argument(..., help="File to re-index (absolute or relative to repo)"),
    provider: str | None = typer.Option(None, help=_PROVIDER_HELP),
) -> None:
    """Re-index a single file after editing"""
    _setup_logging(False)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import IndexConfig
    from trelix.indexing.indexer import Indexer

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=_build_embedder_config(provider),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        _print_error("Configuration error", detail)
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    try:
        indexer = Indexer(config)
        result = indexer.index_file(file)
    except Exception as exc:
        _print_error("update-index failed", exc)
        raise typer.Exit(1) from exc

    print(json.dumps(result))


# ---------------------------------------------------------------------------
# migrate-vectors
# ---------------------------------------------------------------------------


@app.command("migrate-vectors")
def migrate_vectors(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    to: str = typer.Option("qdrant", help="Target backend: qdrant"),
    url: str = typer.Option("http://localhost:6333", help="Qdrant URL"),
    collection: str = typer.Option("trelix", help="Qdrant collection name"),
    api_key: str = typer.Option("", help="Qdrant API key (optional)"),
    reset: Annotated[
        bool,
        typer.Option(
            "--reset",
            help=(
                "Rebuild the vector store at the current embedder's dimension and "
                "invalidate every file hash, so `trelix index` re-embeds from scratch. "
                "Use after switching embedding providers."
            ),
        ),
    ] = False,
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            help=(
                "With --reset: the embedding provider to rebuild FOR. The vector store's "
                "width is fixed at creation, so this decides it. Defaults to "
                "TRELIX_EMBEDDER_PROVIDER."
            ),
        ),
    ] = "",
) -> None:
    """Migrate embeddings from SQLite to Qdrant (or another backend).

    Migrates all embeddings stored in the SQLite chunk_embeddings table:
    primary chunk embeddings, multi-granularity sub-chunk embeddings, and
    RAPTOR-style file-summary embeddings. All three share the same
    sqlite-vec vec0 table (distinguished only by an id-sentinel convention —
    see SQLiteVectorStore), so a single unfiltered read-and-upsert loop
    already carries every source.
    """
    _setup_logging(False)

    import sqlite3
    import struct

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import IndexConfig, StoreConfig
    from trelix.store.vector_qdrant import QdrantVectorStore

    if reset:
        from trelix.core.config import IndexConfig as _IndexConfig
        from trelix.embedder import make_embedder
        from trelix.store.db import Database as _Database
        from trelix.store.dimension_guard import DimensionGuard as _DimensionGuard
        from trelix.store.vector import make_vector_store

        cfg = _IndexConfig(repo_path=str(Path(repo).resolve()))
        if provider:
            cfg.embedder = _build_embedder_config(provider)

        # Refuse early on a backend this cannot rebuild. Qdrant and LanceDB pin their
        # dimension at collection/table creation exactly as vec0 does, and --reset has
        # no per-backend dispatch, so it would otherwise print success having touched
        # nothing. A loud refusal naming the backend beats both a false success and the
        # generic NotImplementedError the base store would raise a few lines later.
        if cfg.store.backend != "sqlite":
            _print_error(
                f"--reset does not support the '{cfg.store.backend}' backend",
                "only the sqlite backend can be rebuilt in place; delete the collection "
                "or table and re-index instead",
            )
            raise typer.Exit(1)

        db = _Database(cfg.db_path_absolute)

        # Three things have to happen for a re-index to actually work afterwards, and
        # the previous implementation did none of them.
        #
        # 1. The vec0 table must be REBUILT, not emptied. Its vector width is fixed by
        #    its CREATE statement, so a row delete leaves a table that still rejects
        #    vectors of the new dimension. Worse, the delete never happened: it ran on
        #    Database's connection, which does not load the sqlite-vec extension, so it
        #    raised "no such module: vec0" into a bare `except: pass`.
        # 2. Every file hash must be invalidated. `index()` skips files whose hash is
        #    unchanged and does not check whether their vectors still exist, so
        #    otherwise the next run prints "Nothing to index — all files up to date"
        #    over an index with no vectors at all.
        # 3. The recorded dimension is cleared LAST. Clearing it first — as this did —
        #    leaves DimensionGuard with nothing to compare against, so a mismatch is no
        #    longer caught up front and the next run pays for a full embedding pass
        #    before failing on its first insert.
        try:
            dimension = make_embedder(cfg.embedder).dimension
        except Exception as exc:
            _print_error("Cannot determine the embedder's dimension", exc)
            raise typer.Exit(1) from exc

        vector_store = make_vector_store(cfg, dimension=dimension)
        try:
            db.clear_all_embeddings(vector_store=vector_store)
        except Exception as exc:
            _print_error("Could not rebuild the vector store", exc)
            raise typer.Exit(1) from exc

        # Both levels, because they gate different stages. File hashes gate re-PARSING;
        # symbol content hashes gate re-EMBEDDING, and `_insert_one` leaves a symbol
        # whose hash still matches completely untouched. Invalidating only the file
        # hashes re-parsed every file and still embedded nothing.
        files_invalidated = db.invalidate_all_file_hashes()
        symbols_invalidated = db.invalidate_all_symbol_hashes()
        _DimensionGuard.reset(db)

        console.print(
            f"[green]Vector store rebuilt at {dimension} dimensions.[/green]\n"
            f"Invalidated {files_invalidated} file and {symbols_invalidated} symbol "
            "hashes so every chunk is re-embedded.\n"
            "Run [bold]trelix index .[/bold] to re-embed with the new provider."
        )
        return

    if to != "qdrant":
        err_console.print(
            f"[red]Unsupported target backend:[/red] {_safe_text(repr(to))}."
            " Only 'qdrant' is supported."
        )
        raise typer.Exit(1)

    try:
        # Build config pointing at the existing SQLite index
        config = IndexConfig(repo_path=str(Path(repo).resolve()))
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        _print_error("Configuration error", detail)
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    db_path = config.db_path_absolute
    if not db_path.exists():
        err_console.print(
            f"[red]No index found at {_safe_text(str(db_path))}[/red] —"
            f" run `trelix index {_safe_text(repo)}` first."
        )
        raise typer.Exit(1)

    # Connect to the SQLite vector store directly to read raw embeddings
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        conn.enable_load_extension(True)
        import sqlite_vec

        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:
        _print_error("Failed to load sqlite-vec", exc)
        raise typer.Exit(1) from exc

    # Detect embedding dimension from the sqlite-vec virtual table metadata
    try:
        row = conn.execute("SELECT embedding FROM chunk_embeddings LIMIT 1").fetchone()
    except Exception as exc:
        _print_error("Failed to read chunk_embeddings", exc)
        raise typer.Exit(1) from exc

    if row is None:
        console.print(
            "[yellow]No embeddings found in the SQLite store — nothing to migrate.[/yellow]"
        )
        return

    raw_bytes: bytes = row[0]
    dimension = len(raw_bytes) // 4  # float32 = 4 bytes

    # Build a temporary StoreConfig pointing at Qdrant
    qdrant_config = IndexConfig(
        repo_path=config.repo_path,
        store=StoreConfig(  # type: ignore[call-arg]
            db_path=config.store.db_path,
            qdrant_url=url,
            qdrant_api_key=api_key or None,
            qdrant_collection=collection,
        ),
    )
    qdrant_store = QdrantVectorStore(qdrant_config, dimension)

    # Stream all rows from sqlite-vec in batches
    total_row = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()
    total = total_row[0] if total_row else 0
    console.print(
        f"[cyan]Migrating {total:,} embeddings (dim={dimension}) → Qdrant {_safe_text(url)}[/cyan]"
    )

    BATCH = 500
    offset = 0
    migrated = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Migrating…", total=total)

        while True:
            rows = conn.execute(
                "SELECT chunk_id, embedding FROM chunk_embeddings LIMIT ? OFFSET ?",
                (BATCH, offset),
            ).fetchall()
            if not rows:
                break

            pairs: list[tuple[int, list[float]]] = []
            for chunk_id, raw in rows:
                n = len(raw) // 4
                emb = list(struct.unpack(f"{n}f", raw))
                pairs.append((chunk_id, emb))

            qdrant_store.upsert_batch(pairs)
            migrated += len(pairs)
            offset += BATCH
            progress.advance(task, advance=len(pairs))

    conn.close()
    console.print(f"[green]Migration complete:[/green] {migrated:,} embeddings written to Qdrant.")


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------


@app.command()
def watch(
    repo: str = typer.Argument(..., help="Path to the repository to watch"),
    provider: str | None = typer.Option(None, help=_PROVIDER_HELP),
) -> None:
    """Watch repo for changes and auto-update index. Ctrl+C to stop."""
    _setup_logging(False)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import IndexConfig
    from trelix.indexing.indexer import Indexer
    from trelix.indexing.watcher import FileWatcher

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=_build_embedder_config(provider),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        _print_error("Configuration error", detail)
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    try:
        indexer = Indexer(config)
    except Exception as exc:
        _print_error("Failed to initialize indexer", exc)
        raise typer.Exit(1) from exc

    # Run initial full index so the watcher starts from a known-good state
    console.print(Panel(f"[bold cyan]Initial index[/bold cyan] {_safe_text(repo)}", expand=False))
    try:
        indexer.index()
    except Exception as exc:
        _print_error("Initial indexing failed", exc)
        raise typer.Exit(1) from exc

    # Start the file watcher
    try:
        watcher = FileWatcher(indexer, indexer.walker)
    except ImportError as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    watcher.start()
    console.print("[green]Watching for changes. Press Ctrl+C to stop.[/green]")

    try:
        import time as _time

        while True:
            _time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
        console.print("\n[dim]Watch stopped.[/dim]")


# ---------------------------------------------------------------------------
# watch-all
# ---------------------------------------------------------------------------


@app.command("watch-all")
def watch_all(
    config: str | None = typer.Option(
        None,
        "--config",
        help="Path to federation registry JSON. Defaults to ~/.config/trelix/repos.json",
    ),
) -> None:
    """Watch all federated repos for changes and auto-update their indexes. Ctrl+C to stop."""
    _setup_logging(False)

    registry = RepoRegistry.load(config_path=config)
    entries = registry.list()

    if not entries:
        console.print(
            "[yellow]No repos registered. Use: trelix federation add <alias> <path>[/yellow]"
        )
        raise typer.Exit(0)

    console.print(
        Panel(
            f"[bold cyan]watch-all[/bold cyan] — watching {len(entries)} repo(s):\n"
            # Alias/path come from the federation registry JSON on disk, not
            # from argv — bracket-shaped values there must not eat the Panel.
            + "\n".join(
                f"  [green]{_safe_text(e.alias)}[/green]  {_safe_text(e.path)}" for e in entries
            ),
            expand=False,
        )
    )

    try:
        from trelix.indexing.multi_watcher import MultiRepoWatcher
    except ImportError as exc:
        _print_error("Error", exc)
        raise typer.Exit(1)

    watcher = MultiRepoWatcher(registry)

    import asyncio
    import signal

    async def _watch_until_signalled() -> None:
        # Handlers MUST be installed from in here, on the loop asyncio.run
        # built. Registering them against asyncio.get_event_loop() out in the
        # sync body put them on a different loop object that never runs an
        # iteration (measured: differing id(), and _signal_handlers[SIGTERM]
        # is None inside this coroutine), so a `docker stop` / `kubectl
        # delete` SIGTERM was swallowed for the whole grace period and the
        # process got hard-killed before the stats block below ever printed.
        # add_signal_handler also repoints process-level SIGINT at asyncio's
        # no-op handler, so the old KeyboardInterrupt fallback was dead too —
        # Ctrl+C did literally nothing.
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except (NotImplementedError, RuntimeError) as exc:
                # Windows has no add_signal_handler. Say so instead of
                # pretending shutdown is graceful: on this platform Ctrl+C
                # unwinds via KeyboardInterrupt and SIGTERM just kills us,
                # losing the summary.
                console.print(
                    f"[yellow]Warning:[/yellow] cannot install a {sig.name} handler "
                    f"({_safe_text(str(exc))}) — shutdown on {sig.name} will not be graceful."
                )

        await watcher.run(stop_event)

    console.print("[green]Watching for changes. Press Ctrl+C to stop.[/green]")

    try:
        asyncio.run(_watch_until_signalled())
    except KeyboardInterrupt:
        pass

    stats = watcher.stats()
    # files_skipped_ignored existed in MultiRepoWatcher.stats() and was invisible here, so
    # the one counter that says "an ignore rule is hiding files you think are watched" never
    # reached a user. `.get` rather than `[...]`: a stats() without the key means a watcher
    # too old to count it, and "n/a" is the honest rendering of that — 0 would assert
    # nothing was ignored.
    ignored = stats.get("files_skipped_ignored")
    console.print(
        f"\n[dim]Watch stopped. "
        f"Re-indexed: {stats['files_reindexed']} files | "
        f"Skipped (unchanged): {stats['files_skipped_unchanged']} files | "
        f"Skipped (ignored): {'n/a' if ignored is None else ignored} files.[/dim]"
    )


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    repo_path: str = typer.Argument(..., help="Repository to serve"),
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(8765, help="Port to bind"),
) -> None:
    """Start a REST API server for trelix search and synthesis."""
    try:
        import uvicorn

        from trelix.api.app import create_app
    except ImportError:
        typer.echo("trelix serve requires: pip install 'trelix[serve]'")
        raise typer.Exit(1)

    from trelix.core.logging_setup import setup_json_logging, uvicorn_log_config

    setup_json_logging()

    # repo_path is the scope, not a label. It used to reach only the banner
    # below, so `serve /one/repo` left every route willing to read any absolute
    # path the caller named; passing it here makes it the containment allow-list
    # root (see api/app.py's "Containment" section).
    api_app = create_app(served_root=repo_path)

    _warn_if_exposed_without_auth(host)

    typer.echo(f"trelix API serving {repo_path} at http://{host}:{port}")
    uvicorn.run(api_app, host=host, port=port, log_config=uvicorn_log_config())


# Loopback forms. Anything else is reachable from another machine.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


def _warn_if_exposed_without_auth(host: str) -> None:
    """Say so, loudly, when the API is reachable off-box with no credential required.

    `api/app.py`'s `authenticate()` is open by design when neither a static token nor
    OIDC is configured — see the "3./4." branch there. That is a reasonable default for
    the documented local use (`--host` defaults to `127.0.0.1`), and this does NOT change
    it: failing closed would break that UX and is a bigger decision than a warning.

    What makes it dangerous is that the shipped container overrides the host.
    `Dockerfile:56` and `docker-compose.yml` both run `serve /repo --host 0.0.0.0`, and
    compose publishes `8765` while bind-mounting the user's repository — so
    `docker compose up` served open, unauthenticated code search over the mounted repo.
    `.github/SECURITY.md` told readers serve binds to 127.0.0.1, which is true of the CLI
    and inverted by the image.

    Warning at the bind boundary rather than in compose alone because it also catches
    `trelix serve --host 0.0.0.0` run directly, which no compose fix can reach. This
    release exists largely because things were switched on and saying nothing; an open
    port is the last place to keep that habit.
    """
    if host in _LOOPBACK_HOSTS:
        return

    # Reuses the exact objects `create_app()` gates on, so this cannot drift from the
    # real decision. `_ApiAuthSettings` lives in api/app.py rather than core/config
    # because auth is process-wide while IndexConfig is per-repo.
    #
    # OIDC alone is sufficient protection: once a verifier exists, `authenticate()`
    # raises 401 for a missing credential, so an SSO-only deployment is not open. Asking
    # `_build_oidc_verifier` rather than reading an env var means an SSO config that is
    # enabled but unusable (bad issuer, unreachable JWKS) still triggers the warning —
    # which is the case where a reader would most wrongly assume they were covered.
    from trelix.api.app import _ApiAuthSettings, _build_oidc_verifier
    from trelix.core.config import SSOConfig

    try:
        if _ApiAuthSettings().api_auth_token is not None:
            return
        if _build_oidc_verifier(SSOConfig()) is not None:
            return
    except Exception as exc:  # pragma: no cover - config shape is validated elsewhere
        logger.debug("Could not read auth settings for the exposure check: %s", exc)
        return

    err_console.print(
        f"[bold yellow]WARNING:[/bold yellow] serving on [bold]{_safe_text(host)}[/bold] "
        "with no authentication configured — every endpoint is open to anyone who can "
        "reach this port.\n"
        "[dim]Set TRELIX_API_AUTH_TOKEN=<secret> to require an X-Trelix-Api-Key header, "
        "or bind to 127.0.0.1 instead. See docs/SSO.md for the OIDC path.[/dim]"
    )


# ---------------------------------------------------------------------------
# graph (knowledge graph build)
# ---------------------------------------------------------------------------


@app.command(
    help=(
        "Build the knowledge graph for an indexed repository.\n\n"
        "NOTE: The old call-graph display command has been renamed to 'trelix call-graph'. "
        "See 'trelix call-graph --help'."
    )
)
def graph(
    repo_path: str = typer.Argument(..., help="Path to indexed repository"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Export Pyvis HTML"),
    output: str = typer.Option(
        "", "--output", "-o", help="Output path for HTML (default: .trelix/graph.html)"
    ),
    concepts: bool = typer.Option(
        False,
        "--concepts",
        "-c",
        # The bound is real (200 symbols in batches of 20, so at most 10 calls) but was
        # invisible at the point of decision, which is the only place it helps.
        help=(
            "Extract LLM semantic concepts (paid: up to 10 LLM calls over the 200 "
            "most central symbols; requires an LLM API key)"
        ),
    ),
    json_output: bool = typer.Option(False, "--json", help="Output stats as JSON"),
) -> None:
    """Build the knowledge graph for an indexed repository.

    NOTE: The old 'trelix graph <repo> <symbol>' command for call-graph display has been
    renamed to 'trelix call-graph'. See 'trelix call-graph --help'.
    """
    _setup_logging(False)

    from pathlib import Path as _Path

    from trelix.core.config import IndexConfig
    from trelix.graph.builder import GraphBuilder

    config = IndexConfig(repo_path=str(_Path(repo_path).resolve()))
    builder = GraphBuilder(config)

    with _status_console(json_output).status("Building knowledge graph..."):
        result = builder.build(extract_concepts=concepts)

    # Exported BEFORE the --json early return below. It used to sit after it, so
    # `--visualize --json` accepted the flag, wrote no file, and said nothing —
    # measured on this repo as a stale 31-byte graph.html surviving the run while the
    # same command without --json wrote 326 KB.
    viz_path: str | None = None
    viz_error: str | None = None
    if visualize:
        from trelix.graph.visualizer import GraphVisualizer

        out = output or str(_Path(repo_path) / ".trelix" / "graph.html")
        try:
            viz_path = str(GraphVisualizer().export_html(result.code_graph, out))
        except Exception as exc:
            # Guarded because moving the export BEFORE the --json return also moved it
            # in front of the payload. Unguarded, a pyvis failure or an unwritable
            # output path would take the graph statistics down with it — the command's
            # primary result, which was already computed successfully. The visualization
            # is the optional half, so it fails on its own.
            viz_error = str(exc)
            logger.warning("Graph visualization failed: %s", exc)

    if json_output:
        data: dict[str, object] = {
            "node_count": result.node_count,
            "edge_count": result.edge_count,
            "community_count": result.community_count,
            "concept_count": result.concept_count,
        }
        # Same additive rule as the visualization keys below: a machine consumer that
        # reads concept_count cannot otherwise tell it describes a capped sample rather
        # than the repository, and the cap is 200 against 12,184 symbols on this repo.
        if concepts:
            data["concept_symbols_considered"] = result.concept_symbols_considered
            data["concept_symbols_total"] = result.concept_symbols_total
        # Additive only, and only when the flag was passed: the four documented keys
        # keep their names and types, so an existing consumer is unaffected, while one
        # that asked for a visualization learns where it landed.
        if viz_path is not None:
            data["visualization_path"] = viz_path
        elif viz_error is not None:
            # Reported in-band rather than dropped: a consumer that asked for a
            # visualization needs to learn it did not get one.
            data["visualization_error"] = viz_error
        # indent=None keeps this payload compact, as it has always been —
        # _print_json's default would reformat a machine-readable contract.
        _print_json(data, indent=None)
        return

    console.print("[green]Knowledge Graph built[/green]")
    console.print(f"  Nodes      : {result.node_count}")
    console.print(f"  Edges      : {result.edge_count}")
    console.print(f"  Communities: {result.community_count}")
    if concepts:
        # The count alone reads as a property of the repository when it is really a
        # property of a capped sample — on a 12,184-symbol index, 200 of them. Saying
        # so here is the difference between "this repo has 47 concepts" and "47 concepts
        # were found in the 200 most central symbols of 12,184".
        coverage = ""
        if result.concept_symbols_total > result.concept_symbols_considered > 0:
            pct = 100.0 * result.concept_symbols_considered / result.concept_symbols_total
            coverage = (
                f" [yellow](from the {result.concept_symbols_considered} most central of "
                f"{result.concept_symbols_total} symbols — {pct:.1f}%)[/yellow]"
            )
        console.print(f"  Concepts   : {result.concept_count}{coverage}")
    console.print(f"  Time       : {result.elapsed_seconds:.2f}s")

    if result.community_summary:
        console.print("\n[bold]Top Communities:[/bold]")
        # top_files are indexed-repository paths. community_id is an int, and
        # "[3]" is not tag-shaped to Rich (a tag must start [a-z#/@]), so the
        # surrounding brackets are left as the literal formatting they are.
        for c in result.community_summary[:5]:
            files = _safe_text(", ".join(c["top_files"][:3]))
            console.print(f"  [{c['community_id']}] {c['size']} nodes — {files}")

    if viz_path is not None:
        console.print(f"\n[blue]Graph visualization:[/blue] {_safe_text(viz_path)}")
    elif viz_error is not None:
        err_console.print(
            f"\n[yellow]Graph visualization failed (statistics above are "
            f"unaffected):[/yellow] {_safe_text(viz_error)}"
        )


# ---------------------------------------------------------------------------
# telemetry
# ---------------------------------------------------------------------------


@app.command()
def telemetry(
    repo: Annotated[str, typer.Argument(help="Path to the indexed repository.")] = ".",
    limit: Annotated[int, typer.Option("--limit", "-n", help="Rows to show")] = 20,
) -> None:
    """Show recent query telemetry (latency, result counts, intent breakdown)."""
    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

    config = IndexConfig(repo_path=str(Path(repo).resolve()))
    db = Database(config.db_path_absolute)
    rows = db.get_recent_telemetry(limit=limit)

    if not rows:
        console.print(
            "[yellow]No telemetry recorded. "
            "Set TRELIX_TELEMETRY_ENABLED=true and run queries.[/yellow]"
        )
        return

    table = Table(title=f"Recent Queries (last {len(rows)})")
    table.add_column("ts", style="dim")
    table.add_column("query", max_width=50)
    table.add_column("intent")
    table.add_column("ms", justify="right")
    table.add_column("results", justify="right")

    # `query` is replayed text recorded by whoever issued the search (CLI user,
    # REST API caller, MCP client, agent loop), so it is arbitrary. Truncate
    # first and escape after: escaping first could let [:50] slice through an
    # inserted backslash, and a truncated half-tag renders literally anyway.
    for row in rows:
        table.add_row(
            _safe_text(str(row["ts"])),
            _safe_text(row["query"][:50]),
            _safe_text(str(row["intent"])),
            f"{row['elapsed_ms']:.0f}",
            str(row["result_count"]),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@app.command()
def eval(
    repo: Annotated[str, typer.Argument(help="Path to the indexed repository.")] = ".",
    golden: Annotated[
        str, typer.Option("--golden", "-g", help="Path to golden JSONL file.")
    ] = ".trelix/golden.jsonl",
    plan_cache_file: Annotated[
        str | None,
        typer.Option(
            "--plan-cache-file",
            help=(
                "JSONL file of frozen query plans. Written on the first run, replayed "
                "byte-for-byte afterwards. Without it the LLM planner re-plans every "
                "query and nDCG@10 moves by sd 0.022-0.029 between identical runs."
            ),
        ),
    ] = None,
) -> None:
    """Evaluate retrieval quality against a golden query set (nDCG@10, Recall@10, MRR)."""
    from trelix.core.config import IndexConfig
    from trelix.eval.harness import EvalHarness

    config = IndexConfig(repo_path=repo)
    if plan_cache_file is not None:
        # Rebuild rather than mutate, so the flag cannot half-apply if validation
        # rejects it. Equivalent to TRELIX_RETRIEVAL_PLAN_CACHE_FILE; the flag exists
        # because a measurement people are asked to repeat should be one argument away.
        config = config.model_copy(
            update={
                "retrieval": config.retrieval.model_copy(
                    update={"plan_cache_file": Path(plan_cache_file)}
                )
            }
        )
    harness = EvalHarness(config)
    try:
        metrics = harness.run(golden)
    except FileNotFoundError:
        console.print(f"[red]Golden file not found: {_safe_text(golden)}[/red]")
        console.print("Create a golden.jsonl with lines like:")
        console.print('  {"query": "how does auth work", "relevant_files": ["src/auth.py"]}')
        raise typer.Exit(1)

    table = Table(title="Retrieval Evaluation Results")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_row("nDCG@10", f"{metrics['ndcg@10']:.4f}")
    table.add_row("Recall@10", f"{metrics['recall@10']:.4f}")
    table.add_row("MRR", f"{metrics['mrr']:.4f}")
    table.add_row("Queries evaluated", str(int(metrics["n_queries"])))
    console.print(table)


@app.command("eval-synthesis")
def eval_synthesis(
    repo: Annotated[str, typer.Argument(help="Path to the indexed repository.")] = ".",
    golden: Annotated[
        str, typer.Option("--golden", "-g", help="Path to golden JSONL file.")
    ] = ".trelix/golden_synthesis.jsonl",
) -> None:
    """Evaluate synthesis quality against a golden QA file (GroUSE-style)."""
    from trelix.core.config import IndexConfig
    from trelix.eval.synthesis import SynthesisEvalHarness

    config = IndexConfig(repo_path=repo)
    harness = SynthesisEvalHarness(config)
    try:
        metrics = harness.run(golden)
    except FileNotFoundError:
        console.print(f"[red]Golden file not found: {_safe_text(golden)}[/red]")
        console.print("Create a golden_synthesis.jsonl with lines like:")
        console.print(
            '  {"query": "how does auth work", "relevant_files": ["src/auth.py"],'
            ' "expected_answer_fragments": ["jwt"], "expected_symbols": ["AuthMiddleware.verify"]}'
        )
        raise typer.Exit(1)

    table = Table(title="Synthesis Quality Results (GroUSE-style)")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Direction", style="dim")
    table.add_row("Hallucination rate", f"{metrics['hallucination_rate']:.4f}", "lower = better")
    table.add_row("Completeness", f"{metrics['completeness']:.4f}", "higher = better")
    table.add_row("Faithfulness", f"{metrics['faithfulness']:.4f}", "higher = better")
    table.add_row("Overall", f"{metrics['overall']:.4f}", "higher = better")
    table.add_row("Queries evaluated", str(int(metrics["n_queries"])), "")
    console.print(table)


# ---------------------------------------------------------------------------
# taint
# ---------------------------------------------------------------------------


@app.command()
def taint(
    repo: Annotated[str, typer.Argument(help="Path to the indexed repository.")] = ".",
    tier: Annotated[
        str, typer.Option("--tier", "-t", help="Taint tier: default|intrafile|interfile")
    ] = "default",
    severity: Annotated[
        str, typer.Option("--severity", "-s", help="Filter: ERROR|WARNING|INFO")
    ] = "",
    rules: Annotated[
        str,
        typer.Option(
            "--rules",
            "-r",
            help="Path to a semgrep rules file or directory. "
            "Defaults to the 'p/default' registry pack, which requires network access.",
        ),
    ] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Output raw JSON.")] = False,
) -> None:
    """Run Semgrep taint analysis and show source->sink flows.

    Requires: pip install trelix[taint]
    """

    from trelix.analysis.taint import TaintAnalyzer
    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

    # TaintAnalyzer.run() has always taken a rules_path; the CLI just never passed one,
    # so every invocation was pinned to the `p/default` registry pack. That needs
    # outbound network access and can change between runs, which makes a taint result
    # neither reproducible nor pinnable in CI. `--rules config/semgrep-taint.yaml` runs
    # against rules that live in the repository and change only when a commit changes
    # them.
    rules_path = str(Path(rules).resolve()) if rules else None
    if rules_path and not Path(rules_path).exists():
        _print_error("Rules not found", rules_path)
        raise typer.Exit(code=1)

    config = IndexConfig(repo_path=str(Path(repo).resolve()))
    analyzer = TaintAnalyzer(repo_path=str(Path(repo).resolve()), tier=tier)
    with _status_console(json_output).status("Running Semgrep taint analysis..."):
        # scan() rather than run(): run() returns [] for a clean scan, a missing binary
        # AND a failed one, so it cannot distinguish "no vulnerabilities" from "no
        # analysis". Reporting the former when the latter happened is the worst thing
        # this command can do.
        scan_result = analyzer.scan(rules_path=rules_path)
    flows = scan_result.flows

    from trelix.analysis.taint import ScanOutcome

    # Whether the scan can be BELIEVED is decided here, above the flow count — not
    # inside `if not flows:`, where every outcome branch used to live. `scan()` carries
    # flows through on the failed path on purpose, so a semgrep run that errored fatally
    # and still emitted one finding matched no branch at all: no diagnostic, no non-zero
    # exit, and `--json` stdout that parses exactly like a healthy scan. A CI step gating
    # on this command read "no vulnerabilities" when what happened was "no analysis".
    # `ScanResult.is_trustworthy` was written for this decision and had no caller.
    #
    # Ordering is the whole fix: an untrustworthy result also never reaches the DB write
    # below, because partial output from a broken scan is not an index of the repository.
    if not scan_result.is_trustworthy:
        # Three earlier defects in these messages, recorded because each is a mistake a
        # later edit can make again. (1) The no-flows message printed prose to STDOUT even
        # under --json, so `trelix taint --json | jq` failed on the most common path in CI
        # — the one a no-semgrep install always takes; that message now sits in the
        # `if not flows:` block below and stays `[]` under --json. (2) "trelix[taint]" was
        # unescaped, so Rich read "[taint]" as an opening tag and swallowed it, rendering
        # the instruction as "pip install 'trelix'" — telling the reader to install the
        # package they already have. That is the same swallow _print_error() documents.
        # (3) A CLEAN SCAN was reported as a possible missing install, because run()
        # collapses four outcomes into an empty list; an earlier attempt at fixing that
        # branched on `shutil.which` alone, which made one case worse — a scan that FAILED
        # (--tier intrafile without the Semgrep Pro Engine exits 2 with empty stdout) then
        # printed a confident "semgrep ran and reported nothing" at exit 0.
        if json_output:
            # The payload stays `[]` so `| jq` keeps working, but the EXIT CODE must
            # still distinguish a clean scan from one that never ran. Printing [] and
            # exiting 0 here left the very defect this branch exists to fix in place on
            # the JSON path — which is the path CI uses, and therefore the one where a
            # false all-clear does the most damage. Diagnostics go to stderr so stdout
            # remains byte-compatible.
            _print_json([])
            err_console.print(
                f"[red]Taint analysis did not complete ({scan_result.outcome}):[/red] "
                f"{_safe_text(scan_result.detail or 'no detail')}"
            )
        elif scan_result.outcome is ScanOutcome.SEMGREP_MISSING:
            console.print(
                "[yellow]semgrep is not installed, so no taint analysis ran: "
                f"pip install {_safe_text('trelix[taint]')}[/yellow]"
            )
        elif scan_result.outcome is ScanOutcome.SEMGREP_FAILED:
            _print_error(
                "Taint analysis failed — this is NOT a clean result",
                scan_result.detail or "semgrep exited with an error",
            )
            if tier != "default":
                err_console.print(
                    f"[dim]--tier {_safe_text(tier)} requires the Semgrep Pro Engine, which "
                    f"pip install {_safe_text('trelix[taint]')} does not include.[/dim]"
                )
        elif scan_result.outcome is ScanOutcome.SCANNED_NOTHING:
            _print_error(
                "Taint analysis examined 0 files — this is NOT a clean result",
                scan_result.detail or "check the target path and the rules' languages",
            )
        else:
            # Reached only if a new ScanOutcome is added without a branch here. Naming
            # the outcome beats the old fallthrough, which was the green "reported
            # nothing" message.
            _print_error(
                f"Taint analysis did not complete ({scan_result.outcome})",
                scan_result.detail or "no detail",
            )
        # The one carve-out, and it is not a scan failure: an optional dependency that
        # was never installed must not turn every CI run into a red one.
        if scan_result.outcome is not ScanOutcome.SEMGREP_MISSING:
            raise typer.Exit(1)
        return

    # Reached only for a trustworthy scan, so this is now the one place that may say the
    # word "clean". Under --json the payload stays `[]` rather than prose: see note (1).
    if not flows:
        if json_output:
            _print_json([])
        else:
            console.print(
                f"[green]No taint flows found.[/green] semgrep examined "
                f"{scan_result.files_scanned} file(s) and reported nothing.\n"
                "[dim]If you expected findings, check the ruleset — the default "
                "'p/default' registry pack needs network access. Pass --rules "
                "<path> to scan against rules you control.[/dim]"
            )
        return

    # Persist to DB
    db = Database(config.db_path_absolute)
    db.insert_taint_flows(flows)

    filtered = [f for f in flows if not severity or f.severity == severity.upper()]

    if json_output:
        # Semgrep rule ids and scanned file paths are third-party text; a
        # bracket in either used to abort this with MarkupError. See _print_json.
        _print_json(
            [
                {
                    "rule": f.rule_id,
                    "severity": f.severity,
                    "source": f"{f.source_file}:{f.source_line}",
                    "sink": f"{f.sink_file}:{f.sink_line}",
                }
                for f in filtered
            ]
        )
        return

    table = Table(title=f"Taint Flows ({len(filtered)} found)")
    table.add_column("Severity", style="bold red")
    table.add_column("Rule")
    table.add_column("Source")
    table.add_column("Sink")
    # Severity/rule id come from Semgrep's rule pack and the file paths from the
    # scanned repository — all third-party text landing in markup-parsed cells.
    for f in filtered[:50]:
        table.add_row(
            _safe_text(f.severity),
            _safe_text(f.rule_id),
            f"{_safe_text(f.source_file)}:{f.source_line}",
            f"{_safe_text(f.sink_file)}:{f.sink_line}",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


@app.command()
def review(
    repo: Annotated[str, typer.Argument(help="Path to the indexed repository.")] = ".",
    diff: Annotated[
        str | None,
        typer.Option("--diff", "-d", help="Path to .diff file. If omitted, runs git diff."),
    ] = None,
    base: Annotated[str, typer.Option("--base", help="Base git ref for diff.")] = "HEAD~1",
    head: Annotated[str, typer.Option("--head", help="Head git ref for diff.")] = "HEAD",
    json_output: Annotated[bool, typer.Option("--json", help="Output raw JSON.")] = False,
    max_files: Annotated[int, typer.Option("--max-files", help="Max files to review.")] = 10,
    pr: str | None = typer.Option(
        None,
        "--pr",
        help=(
            "GitHub PR ref (owner/repo#number). Fetches diff from GitHub API. "
            "Requires GITHUB_TOKEN env var."
        ),
    ),
    post_comments: bool = typer.Option(
        False,
        "--post-comments",
        help=(
            "Post review comments back to GitHub PR (requires GITHUB_TOKEN + pull_requests:write)."
        ),
    ),
) -> None:
    """Review a git diff using trelix retrieval-augmented analysis."""
    _setup_logging(False)

    from trelix.core.config import IndexConfig
    from trelix.review.diff_parser import DiffParser
    from trelix.review.reviewer import DiffReviewer

    config = IndexConfig(repo_path=str(Path(repo).resolve()))

    # ------------------------------------------------------------------
    # GitHub PR path
    # ------------------------------------------------------------------
    if pr is not None:
        import os

        from trelix.review.github import GitHubPRClient, parse_pr_ref

        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            err_console.print(
                "[red]Error:[/red] GITHUB_TOKEN environment variable is required for --pr."
            )
            raise typer.Exit(1)

        try:
            owner, repo_name, pr_number = parse_pr_ref(pr)
        except ValueError as exc:
            _print_error("Error", exc)
            raise typer.Exit(1)

        # In --json mode, stdout must carry ONLY the final JSON array — every
        # progress/status message below goes to err_console (stderr) instead,
        # since callers (e.g. the PR-review CI workflow) redirect stdout to a
        # file and parse it as JSON.
        status_console = _status_console(json_output)

        status_console.print(f"[cyan]Fetching PR diff from GitHub:[/cyan] {_safe_text(pr)}")
        gh_client = GitHubPRClient(token=token)

        try:
            pr_files = gh_client.get_pr_files(owner, repo_name, pr_number)
        except Exception as exc:
            _print_error("GitHub API error", exc)
            raise typer.Exit(1)

        # Build a unified diff string from PR files
        diff_lines: list[str] = []
        for f in pr_files:
            if f.patch is None:
                status_console.print(
                    f"[dim]Skipping binary/oversized file: {_safe_text(f.filename)}[/dim]"
                )
                continue
            diff_lines.append(f"diff --git a/{f.filename} b/{f.filename}")
            diff_lines.append(f"--- a/{f.previous_filename or f.filename}")
            diff_lines.append(f"+++ b/{f.filename}")
            diff_lines.append(f.patch)
        pr_diff_str = "\n".join(diff_lines)

        if not pr_diff_str.strip():
            if json_output:
                _print_json([])
            else:
                console.print(
                    "[yellow]No textual changes found in PR (all binary files?).[/yellow]"
                )
            raise typer.Exit(0)

        reviewer = DiffReviewer(config)
        with status_console.status("Retrieving context and generating review..."):
            comments = reviewer.review(diff_text=pr_diff_str)

        if not comments:
            if json_output:
                _print_json([])
            else:
                console.print("[green]No issues found.[/green]")
            raise typer.Exit(0)

        if json_output:
            # `comment` is LLM prose and `file_path` comes from the PR diff. A
            # bracket in either raised MarkupError, and a long comment wrapped
            # mid-string into unparseable JSON. See _print_json.
            _print_json(
                [
                    {
                        "file": c.file_path,
                        "lines": f"{c.line_start}-{c.line_end}",
                        "severity": c.severity,
                        "comment": c.comment,
                    }
                    for c in comments
                ]
            )
        else:
            from rich.table import Table as _Table

            table = _Table(title=f"Review Results ({len(comments)} comments)")
            table.add_column("File", style="dim")
            table.add_column("Lines")
            table.add_column("Severity", style="bold")
            table.add_column("Comment", max_width=80)
            # `comment` is LLM prose that quotes the reviewed diff, and
            # file_path comes from that diff — both arbitrary text. `color` is
            # trelix's own whitelisted markup and stays unescaped; the severity
            # *inside* it is model output, so an "[/x]" severity would otherwise
            # make "[white][/x][/white]" unparseable.
            for c in comments:
                color = {"ERROR": "red", "WARN": "yellow", "INFO": "blue"}.get(c.severity, "white")
                table.add_row(
                    _safe_text(c.file_path),
                    f"{c.line_start}-{c.line_end}",
                    f"[{color}]{_safe_text(c.severity)}[/{color}]",
                    _safe_text(c.comment),
                )
            console.print(table)

        if post_comments:
            from trelix.review.github import ReviewComment as _GHReviewComment

            try:
                head_sha = gh_client.get_pr_head_sha(owner, repo_name, pr_number)
                inline_comments = [
                    _GHReviewComment(
                        path=c.file_path,
                        line=c.line_start,
                        body=c.comment,
                    )
                    for c in comments
                    if c.line_start
                ]
                gh_client.post_review(
                    owner=owner,
                    repo=repo_name,
                    pr_number=pr_number,
                    commit_sha=head_sha,
                    body=f"trelix review: {len(inline_comments)} inline comment(s) found.",
                    comments=inline_comments,
                )
                console.print(
                    f"[green]Posted review with {len(inline_comments)} inline comments.[/green]"
                )
            except Exception as exc:
                err_console.print(
                    f"[yellow]Warning: failed to post comments: {_safe_text(str(exc))}[/yellow]"
                )

        raise typer.Exit(0)

    # ------------------------------------------------------------------
    # Local git diff path
    # ------------------------------------------------------------------
    parser = DiffParser()

    with _status_console(json_output).status("Parsing diff..."):
        if diff:
            diff_text = Path(diff).read_text()
            hunks = parser.parse(diff_text)
        else:
            hunks = parser.from_git(str(Path(repo).resolve()), base=base, head=head)

    if not hunks:
        console.print("[yellow]No changes found in diff.[/yellow]")
        return

    # Limit to max_files unique files
    seen_files: set[str] = set()
    filtered = []
    for h in hunks:
        if h.file_path not in seen_files:
            seen_files.add(h.file_path)
        if len(seen_files) <= max_files:
            filtered.append(h)

    console.print(f"Reviewing {len(filtered)} hunks across {len(seen_files)} files...")

    reviewer = DiffReviewer(config)
    with _status_console(json_output).status("Retrieving context and generating review..."):
        comments = reviewer.review(filtered)

    if not comments:
        console.print("[green]No issues found.[/green]")
        return

    if json_output:
        # Same untrusted pair as the --pr branch above: LLM comment text and a
        # file path out of the reviewed diff. See _print_json.
        _print_json(
            [
                {
                    "file": c.file_path,
                    "lines": f"{c.line_start}-{c.line_end}",
                    "severity": c.severity,
                    "comment": c.comment,
                }
                for c in comments
            ]
        )
        return

    from rich.table import Table

    table = Table(title=f"Review Results ({len(comments)} comments)")
    table.add_column("File", style="dim")
    table.add_column("Lines")
    table.add_column("Severity", style="bold")
    table.add_column("Comment", max_width=80)
    # Same untrusted trio as the --pr table above: LLM comment text, diff file
    # path, model-supplied severity nested in trelix's own colour markup.
    for c in comments:
        color = {"ERROR": "red", "WARN": "yellow", "INFO": "blue"}.get(c.severity, "white")
        table.add_row(
            _safe_text(c.file_path),
            f"{c.line_start}-{c.line_end}",
            f"[{color}]{_safe_text(c.severity)}[/{color}]",
            _safe_text(c.comment),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# search-all (federated search)
# ---------------------------------------------------------------------------


@app.command(name="search-all")
def search_all(
    query: Annotated[str, typer.Argument(help="Search query.")],
    config_file: Annotated[
        str | None, typer.Option("--config", help="Path to federation.json")
    ] = None,
    k: Annotated[int, typer.Option("--k", help="Results per repo.")] = 10,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Search across all registered repos (federated search)."""

    from trelix.core.config import RetrievalConfig
    from trelix.federation.registry import RepoRegistry
    from trelix.federation.retriever import FederatedRetriever

    registry = RepoRegistry.load(config_file)
    if not registry.list():
        # Same --json contract break as taint's empty path: this returned before
        # the `if json_output:` branch below, so `search-all --json` printed prose
        # to stdout and failed json.loads. An empty registry is the default state,
        # so this was the first thing a new consumer hit.
        if json_output:
            _print_json([])
        else:
            console.print(
                "[yellow]No repos registered. Use: trelix federation add <alias> <path>[/yellow]"
            )
        return

    # federation_max_repos (TRELIX_FEDERATION_MAX_REPOS, default 50) was declared in
    # RetrievalConfig, documented in docs/CONFIGURATION.md and enforced by
    # packages/trelix-mcp — while this line read `FederatedRetriever(registry)`, so the
    # constructor default of max_repos=None applied and search-all fanned out to every
    # registered repo. The cap's stated purpose is to stop a runaway federation_add_repo
    # loop from making every later query scale linearly with the repo count — and both
    # sides read the same RepoRegistry (default ~/.config/trelix/repos.json), so a
    # registry the MCP tools grew was uncapped the moment it was queried from the CLI.
    max_repos = RetrievalConfig().federation_max_repos
    fed = FederatedRetriever(registry, max_repos=max_repos)

    total_registered = len(registry.list())
    # Mirrors the truncation the retriever actually performs — `entries[:max_repos]` in
    # FederatedRetriever._query_repos, i.e. the first N in registry order.
    repos_queried = min(total_registered, max_repos)
    if repos_queried < total_registered:
        # Truncating the fan-out in silence would report a partial search as a complete
        # one: the table looks identical, exits 0, and omits whatever the skipped repos
        # held. Named on stderr — the same stream _status_console() routes to under
        # --json — so `search-all --json | jq` keeps a clean stdout.
        err_console.print(
            f"[yellow]Querying {repos_queried} of {total_registered} registered repos: "
            f"{total_registered - repos_queried} skipped by "
            f"TRELIX_FEDERATION_MAX_REPOS={max_repos}.[/yellow]"
        )
    with _status_console(json_output).status(f"Searching {repos_queried} repos..."):
        results = fed.retrieve(query, k=k)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    if json_output:
        # rel_path/qualified_name are indexed text from every federated repo.
        # See _print_json.
        _print_json(
            [
                {
                    "file": r.file.rel_path,
                    "symbol": r.symbol.qualified_name,
                    "score": round(r.score, 4),
                    "source": r.source,
                }
                for r in results
            ]
        )
        return

    from rich.table import Table

    table = Table(title=f"Federated Search: '{_safe_text(query)}' ({len(results)} results)")
    table.add_column("Repo", style="dim")
    table.add_column("File")
    table.add_column("Symbol")
    table.add_column("Score", justify="right")
    # rel_path/qualified_name are indexed text from every federated repo, and
    # repo_tag is the registry alias — see the note in search(). Federation
    # widens the blast radius: one hostile symbol name in one registered repo
    # would otherwise blank the whole cross-repo result table.
    for r in results[:20]:
        repo_tag = r.source.split(":")[0] if ":" in r.source else ""
        table.add_row(
            _safe_text(repo_tag),
            _safe_text(r.file.rel_path),
            _safe_text(r.symbol.qualified_name),
            f"{r.score:.4f}",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# federation sub-app
# ---------------------------------------------------------------------------

federation_app = typer.Typer(help="Manage federated repo registry.")
app.add_typer(federation_app, name="federation")


@federation_app.command("add")
def federation_add(
    alias: Annotated[str, typer.Argument(help="Short alias for the repo.")],
    path: Annotated[str, typer.Argument(help="Absolute path to the repo root.")],
    weight: Annotated[float, typer.Option("--weight", help="RRF weight (default 1.0).")] = 1.0,
    config_file: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Register a repo for federated search."""
    from trelix.federation.registry import RepoRegistry

    registry = RepoRegistry.load(config_file)
    try:
        registry.add(alias, path, weight)
        registry.save()
        console.print(f"[green]Registered '{_safe_text(alias)}' -> {_safe_text(path)}[/green]")
    except ValueError as exc:
        console.print(f"[red]{_safe_text(str(exc))}[/red]")
        raise typer.Exit(1)


@federation_app.command("list")
def federation_list(
    config_file: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """List all registered repos."""
    from trelix.federation.registry import RepoRegistry

    registry = RepoRegistry.load(config_file)
    entries = registry.list()
    if not entries:
        console.print("[yellow]No repos registered.[/yellow]")
        return
    table = Table(title="Registered Repos")
    table.add_column("Alias")
    table.add_column("Path")
    table.add_column("Weight", justify="right")
    for e in entries:
        table.add_row(_safe_text(e.alias), _safe_text(e.path), str(e.weight))
    console.print(table)


@federation_app.command("remove")
def federation_remove(
    alias: Annotated[str, typer.Argument(help="Alias of the repo to remove.")],
    config_file: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Unregister a repo from federated search."""
    from trelix.federation.registry import RepoRegistry

    registry = RepoRegistry.load(config_file)
    existed = any(e.alias == alias for e in registry.list())
    registry.remove(alias)
    registry.save()
    if existed:
        console.print(f"[green]Removed '{_safe_text(alias)}'[/green]")
    else:
        console.print(f"[yellow]No repo registered with alias '{_safe_text(alias)}'[/yellow]")


# ---------------------------------------------------------------------------
# agent sub-app (persisted ReAct session management)
# ---------------------------------------------------------------------------

agent_app = typer.Typer(help="Manage persisted agentic (ReAct) sessions.")
app.add_typer(agent_app, name="agent")

sessions_app = typer.Typer(help="List/show/clear persisted agent sessions.")
agent_app.add_typer(sessions_app, name="sessions")


@sessions_app.command("list")
def agent_sessions_list(
    repo: Annotated[str, typer.Argument(help="Path to the indexed repository.")],
    limit: Annotated[int, typer.Option("--limit", help="Max sessions to show.")] = 50,
) -> None:
    """List persisted agent sessions for a repo, most recently active first."""
    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

    config = IndexConfig(repo_path=str(Path(repo).resolve()))
    db = Database(config.db_path_absolute)
    try:
        max_age = config.retrieval.agent_session_max_age_seconds
        if max_age > 0:
            db.evict_stale_agent_sessions(max_age)
        sessions = db.list_agent_sessions(limit=limit)
    finally:
        db.close()

    if not sessions:
        console.print("[yellow]No persisted agent sessions.[/yellow]")
        return

    table = Table(title="Agent Sessions")
    table.add_column("Session ID")
    table.add_column("Query")
    table.add_column("Turns", justify="right")
    table.add_column("Last Active")
    # The recorded `query` is arbitrary text (same sink as telemetry()).
    for s in sessions:
        table.add_row(
            _safe_text(str(s["session_id"])),
            _safe_text(s["query"][:60]),
            str(s["turn_count"]),
            _safe_text(str(s["last_active_at"])),
        )
    console.print(table)


@sessions_app.command("show")
def agent_sessions_show(
    repo: Annotated[str, typer.Argument(help="Path to the indexed repository.")],
    session_id: Annotated[str, typer.Argument(help="Session ID to show.")],
) -> None:
    """Show the full turn-by-turn history for a persisted agent session."""
    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

    config = IndexConfig(repo_path=str(Path(repo).resolve()))
    db = Database(config.db_path_absolute)
    try:
        turns = db.get_agent_turns(session_id)
    finally:
        db.close()

    if not turns:
        console.print(f"[yellow]No turns found for session '{_safe_text(session_id)}'.[/yellow]")
        return

    for t in turns:
        # The worst offender in this file: `thought` is raw LLM prose and
        # `observation_content` is a tool result — i.e. retrieved repository
        # source, verbatim. Replaying a session that ever read a Rust
        # comment-stripping regex used to abort on the first "[/!]". Only the
        # [bold] labels are trelix's own markup.
        console.print(
            Panel(
                f"[bold]Thought:[/bold] {_safe_text(t['thought'])}\n"
                f"[bold]Action:[/bold] {_safe_text(str(t['action_type']))} "
                f"{_safe_text(str(t['action_arguments']))}\n"
                f"[bold]Observation ({'ok' if t['observation_success'] else 'err'}):[/bold] "
                f"{_safe_text(t['observation_content'][:500])}",
                title=f"Turn {t['turn_index'] + 1}",
            )
        )


@sessions_app.command("clear")
def agent_sessions_clear(
    repo: Annotated[str, typer.Argument(help="Path to the indexed repository.")],
    session_id: Annotated[str, typer.Argument(help="Session ID to delete.")],
) -> None:
    """Delete a persisted agent session and all its turns."""
    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

    config = IndexConfig(repo_path=str(Path(repo).resolve()))
    db = Database(config.db_path_absolute)
    try:
        existed = db.delete_agent_session(session_id)
    finally:
        db.close()

    if existed:
        console.print(f"[green]Cleared session '{_safe_text(session_id)}'[/green]")
    else:
        console.print(f"[yellow]No session found with ID '{_safe_text(session_id)}'[/yellow]")


# ---------------------------------------------------------------------------
# connector sub-app (Jira/TestRail/Xray/Linear source-connector sync)
# ---------------------------------------------------------------------------

connector_app = typer.Typer(
    help="Sync external artefacts (Jira tickets, TestRail cases, Xray tests, Linear issues)."
)
app.add_typer(connector_app, name="connector")


@connector_app.command("sync")
def connector_sync(
    repo: Annotated[str, typer.Argument(help="Path to the indexed repository.")],
    name: Annotated[
        str, typer.Argument(help="Connector to sync: 'jira', 'testrail', 'xray', or 'linear'.")
    ],
    link: Annotated[
        bool,
        typer.Option(help="Auto-link each synced artefact into generic_edges via ArtifactLinker"),
    ] = True,
) -> None:
    """
    Fetch artefacts from an external system, persist them via the
    connector's ArtifactSource.sync(), and (by default) immediately link
    each synced artefact into generic_edges via ArtifactLinker so it's
    reachable from the code graph without a separate `trelix link-artifacts`
    pass. Pass --no-link to skip linking (e.g. to sync many artefacts
    quickly, then run `trelix link-artifacts` once as a batch afterward).
    """
    _setup_logging(False)

    from trelix.core.config import ArtifactLinkerConfig, IndexConfig
    from trelix.indexing.artifact_linker import ArtifactLinker
    from trelix.indexing.connectors.registry import get_artifact_source
    from trelix.store.db import Database

    try:
        config = IndexConfig(repo_path=str(Path(repo).resolve()))
    except (ValueError, FileNotFoundError) as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    db_path = config.db_path_absolute
    if not db_path.exists():
        err_console.print(
            f"[red]No index found at {_safe_text(str(db_path))}[/red] —"
            f" run `trelix index {_safe_text(repo)}` first."
        )
        raise typer.Exit(1)

    try:
        source = get_artifact_source(name)  # type: ignore[arg-type]
    except ValueError as exc:
        _print_error("Error", exc)
        raise typer.Exit(1) from exc

    try:
        source.validate_config()
    except ValueError as exc:
        _print_error(f"{name} connector is misconfigured", exc)
        raise typer.Exit(1) from exc

    db = Database(db_path)
    try:
        linker = ArtifactLinker(db, ArtifactLinkerConfig(), index_config=config) if link else None
        with console.status(f"[bold cyan]Syncing {name}…[/bold cyan]"):
            result = source.sync(db, linker=linker)
    except Exception as exc:
        _print_error(f"Failed to sync {name}", exc)
        raise typer.Exit(1) from exc
    finally:
        db.close()

    console.print(
        f"[green]Synced {name}:[/green] fetched {result.artifacts_fetched}, "
        f"wrote {result.artifacts_written}, errors {result.errors}, "
        f"linked {result.edges_linked} edge(s)"
    )
    if result.errors:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# audit sub-app (inspect the tamper-evident audit log)
# ---------------------------------------------------------------------------

audit_app = typer.Typer(help="Inspect the tamper-evident audit log (audit.db).")
app.add_typer(audit_app, name="audit")


def _open_audit_store(db: str | None):  # type: ignore[no-untyped-def]
    """Open the AuditStore at *db*, or AuditConfig's resolved default path.

    Exits nonzero when the database could not be opened. AuditStore's init is
    intentionally non-raising, so without this guard every command silently
    operated on an unopened store — `audit verify` would print "chain intact"
    and exit 0 for a path it never read, which is a false integrity assurance.

    The existence check has to come BEFORE the AuditStore() call, and an earlier
    revision of this guard got that wrong. `AuditStore(path)` CREATES the file
    and its schema when the path is absent, so `is_open` was always True and the
    guard never fired: `audit verify --db /typo/path` printed "Audit chain
    intact.", exited 0, and left a fresh 28 KB SQLite file behind — a false
    all-clear from a read-only command, and exactly the failure the paragraph
    above says this function exists to prevent. A CI integrity gate pointed at a
    not-yet-created or misspelled path passed green.

    Opened ``read_only=True``, which is what makes "these commands never write the
    database they read" true rather than intended. The existence check above stopped
    an *absent* path from being created; it said nothing about a path that exists and
    is some other SQLite file. Pointed at, say, a customers database, the old
    read-write open ran the writer's DDL and grew an 8 KB one-table file into a
    32 KB five-table one, then reported "Audit chain intact." — a green integrity
    verdict on a database that is not an audit log. A read-only handle cannot do
    that, and the ``missing_tables`` gate turns the verdict itself into exit 2.
    """
    from trelix.audit.store import AuditStore
    from trelix.core.config import AuditConfig

    path = Path(db) if db else AuditConfig().resolved_db_path

    # Absent path: report and stop BEFORE constructing the store, which would
    # otherwise create it. A path that exists but is not a usable database (a
    # directory, an unreadable file) still falls through to the is_open check
    # below — sqlite cannot open those, so that guard does fire for them.
    if not path.exists():
        err_console.print(
            f"[red]Audit database does not exist[/red] at {_safe_text(str(path))} — "
            "nothing was read, and no database was created. Check the path, or "
            "enable auditing (TRELIX_AUDIT_ENABLED=true) to start a chain."
        )
        raise typer.Exit(2)  # 2 = could not check; 1 is reserved for detected tamper

    store = AuditStore(path, read_only=True)
    # Reported BEFORE the generic is_open check, and with its own wording, because
    # "this file is not an audit log" and "this file would not open" send an
    # operator to different places. Nothing here may resemble a verdict about a
    # chain: there is no chain to have a verdict about.
    if store.missing_tables:
        err_console.print(
            f"[red]Not an audit database[/red] at {_safe_text(str(path))} — no "
            f"{_safe_text(', '.join(store.missing_tables))} table. Nothing was read and "
            "nothing was created, and no verdict about a chain is possible here. Check "
            "the path, or enable auditing (TRELIX_AUDIT_ENABLED=true) to start a chain."
        )
        raise typer.Exit(2)  # 2 = could not check; 1 is reserved for detected tamper
    if not store.is_open:
        if getattr(store, "needs_journal_recovery", False):
            # Named separately because the generic message below is actively wrong
            # here: the path IS a readable file, and telling an operator to check
            # that sends them looking for a problem that does not exist. This is
            # also the moment the command matters most — a writer was just killed
            # and the question is whether entries were lost.
            err_console.print(
                f"[red]Audit database needs journal recovery[/red] at "
                f"{_safe_text(str(path))} — a writer was interrupted and left a "
                "rollback journal beside it. The read commands open read-only and "
                "cannot replay one, so nothing was read and no verdict is possible. "
                "One read-write open recovers it: restart the audited service, or "
                f"run `sqlite3 {_safe_text(str(path))} .tables` once, then re-run "
                "this command. The file is intact; it has not been modified here."
            )
            raise typer.Exit(2)  # 2 = could not check; 1 is reserved for detected tamper
        err_console.print(
            f"[red]Could not open audit database[/red] at {_safe_text(str(path))} — "
            "nothing was read. Check the path (it must be a file, not a directory) "
            "and that it is readable."
        )
        raise typer.Exit(2)  # 2 = could not check; 1 is reserved for detected tamper
    return store, path


def _audit_db_unreadable(path: Path, exc: Exception) -> NoReturn:
    """Report a damaged audit database as *could not check*, never as tamper.

    A corrupted b-tree page — 256 bytes of garbage over page 2 is enough — leaves
    the file openable and makes the first row read raise ``sqlite3.DatabaseError``.
    That surfaced as a traceback and exit 1, which is the tamper exit code: a
    damaged disk was being reported as an attack, and the tokens an alert matches on
    ("TAMPERED", a reason code) were absent, so the alert saw only a nonzero exit.
    Restoring from backup and hunting an intruder are different first moves.
    """
    err_console.print(
        f"[red]Could not read audit database[/red] at {_safe_text(str(path))} — "
        f"SQLite reports: {_safe_text(str(exc))}. That is a damaged or truncated "
        "file, not a tamper verdict; nothing about the chain was determined."
    )
    raise typer.Exit(2)  # 2 = could not check; 1 is reserved for detected tamper


@audit_app.command("list")
def audit_list(
    db: Annotated[str | None, typer.Option("--db", help="Path to audit.db")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Rows to show")] = 50,
) -> None:
    """Show the most recent audit entries (newest first)."""
    import sqlite3

    store, path = _open_audit_store(db)
    try:
        rows = store.recent(limit)
    except sqlite3.DatabaseError as exc:
        _audit_db_unreadable(path, exc)
    if not rows:
        console.print("[yellow]No audit entries.[/yellow]")
        return

    # The title was the one sink this command's original fix missed: a Table
    # title is markup-parsed like any cell, so `--db '/tmp/a[/x].db'` still
    # raised MarkupError after every row had been made safe.
    table = Table(
        title=f"Audit Log ({_safe_text(str(path))})", show_header=True, header_style="bold cyan"
    )
    table.add_column("id", justify="right", style="dim")
    table.add_column("ts", style="dim")
    table.add_column("principal")
    table.add_column("action")
    table.add_column("resource")
    table.add_column("outcome")
    table.add_column("status", justify="right")
    # Every cell is escaped: audit rows record attacker-controlled data (a request
    # path, a JWT `sub`), and Rich would otherwise parse "[...]" as console markup.
    # An unauthenticated `GET /%5B/red%5D` used to store "/[/red]" and make this
    # command die with MarkupError — a request-side DoS of the audit tooling, i.e.
    # exactly the log a responder needs during an incident.
    for r in rows:
        table.add_row(
            _safe_text(str(r.get("id", ""))),
            _safe_text(str(r.get("ts", ""))),
            _safe_text(str(r.get("principal", ""))),
            _safe_text(str(r.get("action", ""))),
            _safe_text(str(r.get("resource", "") or "")),
            _safe_text(str(r.get("outcome", ""))),
            _safe_text(str(r.get("status_code", "") if r.get("status_code") is not None else "")),
        )
    console.print(table)


@audit_app.command("verify")
def audit_verify(
    db: Annotated[str | None, typer.Option("--db", help="Path to audit.db")] = None,
) -> None:
    """Verify the hash chain. Exits nonzero and names the first divergent id on tamper."""
    import sqlite3

    from trelix.audit.store import UNPROVABLE_ID_REASONS

    store, path = _open_audit_store(db)
    try:
        result = store.verify()
    except sqlite3.DatabaseError as exc:
        _audit_db_unreadable(path, exc)
    if result.tampered_id is None:
        console.print("[green]Audit chain intact.[/green]")
        return
    # Two sentences because these are two different facts, and the id alone does
    # not distinguish them. A row fault names a row that is present and wrong. An
    # anchor or count fault names the first id whose EXISTENCE can no longer be
    # proven — every surviving row verified — so calling that "the first divergent
    # entry" would send a responder to inspect a row that is not in the table.
    # The reason code is printed for both so an alert can match on it.
    if result.reason in UNPROVABLE_ID_REASONS:
        err_console.print(
            f"[red]Audit chain TAMPERED[/red] ({result.reason}) — no surviving entry diverged; "
            f"first unprovable entry id: [bold]{result.tampered_id}[/bold]"
        )
    else:
        err_console.print(
            f"[red]Audit chain TAMPERED[/red] ({result.reason}) — "
            f"first divergent entry id: [bold]{result.tampered_id}[/bold]"
        )
    raise typer.Exit(1)


@audit_app.command("export")
def audit_export(
    db: Annotated[str | None, typer.Option("--db", help="Path to audit.db")] = None,
    export_format: Annotated[
        str, typer.Option("--format", help="Export format (ndjson)")
    ] = "ndjson",
) -> None:
    """Export every audit entry (oldest first) to stdout as NDJSON."""
    import sqlite3

    if export_format != "ndjson":
        _print_error("Unsupported format", f"{export_format!r} (only 'ndjson' is supported)")
        raise typer.Exit(1)
    store, path = _open_audit_store(db)
    # `default=str`, mirroring `_canonical_hash`, because a column's declared type
    # does not constrain what is stored: SQLite accepts a BLOB in any of them, and
    # `json.dumps` raised "Object of type bytes is not JSON serializable" — export
    # was the only command a single planted BLOB could kill outright, which is the
    # command an offline verifier and every SIEM shipper depend on.
    try:
        for row in store.iter_for_export():
            print(json.dumps(row, default=str))
    except sqlite3.DatabaseError as exc:
        _audit_db_unreadable(path, exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
