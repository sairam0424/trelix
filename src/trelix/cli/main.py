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

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from trelix.federation.registry import RepoRegistry

warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*huggingface.*")

app = typer.Typer(
    name="trelix",
    help="Fast, reliable code indexing and retrieval.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()
err_console = Console(stderr=True)


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
)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    """Configure the trelix logger. Call once at CLI entry."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    for lib in ("httpx", "httpcore", "openai", "sentence_transformers", "transformers"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


@app.command()
def index(
    repo: str = typer.Argument(..., help="Path to the repository to index"),
    provider: str = typer.Option("local", help=_PROVIDER_HELP),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
) -> None:
    """Index a repository — builds the search index at <repo>/.trelix/index.db"""
    _setup_logging(verbose)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import EmbedderConfig, IndexConfig
    from trelix.indexing.indexer import Indexer

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=EmbedderConfig(provider=cast(_EmbedderProvider, provider)),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        err_console.print(f"[red]Configuration error[/red]: {detail}")
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(Panel(f"[bold cyan]Indexing[/bold cyan] {repo}", expand=False))

    t0 = time.perf_counter()
    try:
        indexer = Indexer(config)
        stats = indexer.index()
    except KeyboardInterrupt:
        err_console.print("[yellow]Indexing cancelled.[/yellow]")
        raise typer.Exit(1)
    except Exception as exc:
        err_console.print(f"[red]Indexing failed:[/red] {exc}")
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
    table.add_row("Elapsed", f"{elapsed:.1f}s")
    if stats.get("errors"):
        table.add_row("[red]Errors[/red]", f"[red]{stats['errors']}[/red]")
    console.print(table)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@app.command()
def search(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    query: str = typer.Argument(..., help="Natural language query"),
    provider: str = typer.Option("local", help=_PROVIDER_HELP),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Search for code — returns ranked results as a table or JSON"""
    _setup_logging(False)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import EmbedderConfig, IndexConfig, RetrievalConfig
    from trelix.retrieval.retriever import Retriever

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=EmbedderConfig(provider=cast(_EmbedderProvider, provider)),
            retrieval=RetrievalConfig(rerank=False),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        err_console.print(f"[red]Configuration error[/red]: {detail}")
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        retriever = Retriever(config)
        context = retriever.retrieve(query)
    except Exception as exc:
        err_console.print(f"[red]Search failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    if json_output:
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

    table = Table(title=f"Search: {query}", show_header=True, header_style="bold cyan")
    table.add_column("File", style="dim", max_width=40)
    table.add_column("Symbol", style="bold")
    table.add_column("Lines", justify="right")
    table.add_column("Score", justify="right")

    for r in context.results:
        table.add_row(
            r.file.rel_path,
            r.symbol.name,
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
    provider: str = typer.Option("local", help=_PROVIDER_HELP),
    agentic: Annotated[
        bool, typer.Option("--agentic", help="Enable multi-turn agentic ReAct loop.")
    ] = False,
) -> None:
    """Ask a question — retrieval + LLM synthesis (requires OPENAI_API_KEY for full synthesis)"""
    _setup_logging(False)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import EmbedderConfig, IndexConfig, RetrievalConfig
    from trelix.retrieval.retriever import Retriever
    from trelix.retrieval.synthesizer import Synthesizer

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=EmbedderConfig(provider=cast(_EmbedderProvider, provider)),
            retrieval=RetrievalConfig(rerank=False),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        err_console.print(f"[red]Configuration error[/red]: {detail}")
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    # --agentic flag overrides the config field
    if agentic:
        config.retrieval.agentic_enabled = True

    try:
        if config.retrieval.agentic_enabled:
            from trelix.agent import AgentLoop

            agent_loop = AgentLoop(config)
            answer = agent_loop.run(query)
            console.print(answer)
            return

        retriever = Retriever(config)
    except Exception as exc:
        err_console.print(f"[red]Retrieval failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        synth = Synthesizer(config.embedder, llm_config=config.llm)
        if config.retrieval.flare_enabled:
            from trelix.retrieval.flare import FLARELoop

            loop = FLARELoop(retriever, synth, config)
            answer = loop.run(query)
            console.print(answer)
        else:
            context = retriever.retrieve(query)
            # If provider=local (no API key), print the context text directly
            if provider == "local":
                console.print(Panel(f"[bold cyan]Context for:[/bold cyan] {query}", expand=False))
                if context.context_text:
                    console.print(context.context_text)
                else:
                    console.print("[yellow]No relevant code found.[/yellow]")
                return
            for token in synth.stream(context, config.retrieval):
                console.print(token, end="", highlight=False)
            console.print()  # final newline
    except Exception as exc:
        err_console.print(f"[red]Synthesis failed:[/red] {exc}")
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# query (human-readable, always Rich, no --json flag)
# ---------------------------------------------------------------------------


@app.command()
def query(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    query_str: str = typer.Argument(..., metavar="QUERY", help="Natural language query"),
    provider: str = typer.Option("local", help=_PROVIDER_HELP),
) -> None:
    """Query a repository — human-readable Rich terminal output (no LLM synthesis)"""
    _setup_logging(False)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import EmbedderConfig, IndexConfig, RetrievalConfig
    from trelix.retrieval.retriever import Retriever

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=EmbedderConfig(provider=cast(_EmbedderProvider, provider)),
            retrieval=RetrievalConfig(rerank=False),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        err_console.print(f"[red]Configuration error[/red]: {detail}")
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query_str}", expand=False))

    try:
        retriever = Retriever(config)
        context = retriever.retrieve(query_str)
    except Exception as exc:
        err_console.print(f"[red]Query failed:[/red] {exc}")
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

    for r in context.results:
        table.add_row(
            r.file.rel_path,
            r.symbol.name,
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
    provider: str = typer.Option("local", help=_PROVIDER_HELP),
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

    from trelix.core.config import EmbedderConfig, IndexConfig, RetrievalConfig
    from trelix.core.models import SearchResult as _SearchResult
    from trelix.retrieval.retriever import Retriever

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=EmbedderConfig(provider=cast(_EmbedderProvider, provider)),
            retrieval=RetrievalConfig(rerank=False),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        err_console.print(f"[red]Configuration error[/red]: {detail}")
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        retriever = Retriever(config)
    except Exception as exc:
        err_console.print(f"[red]Failed to open index:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"\n[bold] Graph:[/bold] {symbol}\n")

    def _render_table(title: str, results: list[_SearchResult]) -> None:
        tbl = Table(show_header=True, header_style="bold cyan", title=title)
        tbl.add_column("File", style="dim", max_width=45)
        tbl.add_column("Symbol", style="bold")
        tbl.add_column("Lines", justify="right")
        tbl.add_column("Kind")
        if results:
            for r in results:
                tbl.add_row(
                    r.file.rel_path,
                    r.symbol.qualified_name or r.symbol.name,
                    f"{r.symbol.line_start}-{r.symbol.line_end}",
                    r.symbol.kind.value if hasattr(r.symbol.kind, "value") else str(r.symbol.kind),
                )
        else:
            tbl.add_row("[dim](none)[/dim]", "", "", "")
        console.print(tbl)

    valid_directions = {"callers", "callees", "importers", "all"}
    if direction not in valid_directions:
        err_console.print(
            f"[red]Invalid direction[/red] {direction!r}. "
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
        _render_table(f'Importers of "{symbol}" ({len(importers)})', importers)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@app.command()
def stats(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
) -> None:
    """Show index statistics (files, symbols, chunks, DB size)"""
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
        err_console.print(f"[red]Configuration error[/red]: {detail}")
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    db_path = config.db_path_absolute
    if not db_path.exists():
        err_console.print(
            f"[red]No index found at {db_path}[/red] — run `trelix index {repo}` first."
        )
        raise typer.Exit(1)

    try:
        with Database(db_path) as db:
            conn = db._conn
            file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            symbol_count = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            db_size_bytes = db_path.stat().st_size
    except Exception as exc:
        err_console.print(f"[red]Failed to read index:[/red] {exc}")
        raise typer.Exit(1) from exc

    db_size_kb = db_size_bytes / 1024

    console.print(Panel(f"[bold cyan]Index Stats:[/bold cyan] {repo}", expand=False))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    table.add_row("Files indexed", str(file_count))
    table.add_row("Symbols", str(symbol_count))
    table.add_row("Chunks", str(chunk_count))
    table.add_row("DB size", f"{db_size_kb:.1f} KB")
    console.print(table)


# ---------------------------------------------------------------------------
# update-index
# ---------------------------------------------------------------------------


@app.command("update-index")
def update_index(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    file: str = typer.Argument(..., help="File to re-index (absolute or relative to repo)"),
    provider: str = typer.Option("local", help=_PROVIDER_HELP),
) -> None:
    """Re-index a single file after editing"""
    _setup_logging(False)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import EmbedderConfig, IndexConfig
    from trelix.indexing.indexer import Indexer

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=EmbedderConfig(provider=cast(_EmbedderProvider, provider)),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        err_console.print(f"[red]Configuration error[/red]: {detail}")
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        indexer = Indexer(config)
        result = indexer.index_file(file)
    except Exception as exc:
        err_console.print(f"[red]update-index failed:[/red] {exc}")
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
                "Clear all stored embeddings and dimension metadata so trelix index starts fresh. "
                "Use after switching embedding providers."
            ),
        ),
    ] = False,
) -> None:
    """Migrate embeddings from SQLite to Qdrant (or another backend)."""
    _setup_logging(False)

    import sqlite3
    import struct

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import IndexConfig, StoreConfig
    from trelix.store.vector_qdrant import QdrantVectorStore

    if reset:
        from trelix.core.config import IndexConfig as _IndexConfig
        from trelix.store.db import Database as _Database
        from trelix.store.dimension_guard import DimensionGuard as _DimensionGuard

        cfg = _IndexConfig(repo_path=str(Path(repo).resolve()))
        db = _Database(cfg.db_path_absolute)
        _DimensionGuard.reset(db)
        db.clear_all_embeddings()
        console.print(
            "[green]Embeddings and dimension metadata cleared.[/green]\n"
            "Run [bold]trelix index .[/bold] to re-embed with the new provider."
        )
        return

    if to != "qdrant":
        err_console.print(
            f"[red]Unsupported target backend:[/red] {to!r}. Only 'qdrant' is supported."
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
        err_console.print(f"[red]Configuration error[/red]: {detail}")
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    db_path = config.db_path_absolute
    if not db_path.exists():
        err_console.print(
            f"[red]No index found at {db_path}[/red] — run `trelix index {repo}` first."
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
        err_console.print(f"[red]Failed to load sqlite-vec:[/red] {exc}")
        raise typer.Exit(1) from exc

    # Detect embedding dimension from the sqlite-vec virtual table metadata
    try:
        row = conn.execute("SELECT embedding FROM chunk_embeddings LIMIT 1").fetchone()
    except Exception as exc:
        err_console.print(f"[red]Failed to read chunk_embeddings:[/red] {exc}")
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
    console.print(f"[cyan]Migrating {total:,} embeddings (dim={dimension}) → Qdrant {url}[/cyan]")

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
    provider: str = typer.Option("local", help=_PROVIDER_HELP),
) -> None:
    """Watch repo for changes and auto-update index. Ctrl+C to stop."""
    _setup_logging(False)

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import EmbedderConfig, IndexConfig
    from trelix.indexing.indexer import Indexer
    from trelix.indexing.watcher import FileWatcher

    try:
        config = IndexConfig(
            repo_path=str(Path(repo).resolve()),
            embedder=EmbedderConfig(provider=cast(_EmbedderProvider, provider)),
        )
    except _PydanticValidationError as exc:
        first_err = exc.errors()[0]
        msg = first_err.get("msg", str(exc))
        field = " -> ".join(str(x) for x in first_err.get("loc", []))
        detail = f"{field}: {msg}" if field else msg
        err_console.print(f"[red]Configuration error[/red]: {detail}")
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        indexer = Indexer(config)
    except Exception as exc:
        err_console.print(f"[red]Failed to initialize indexer:[/red] {exc}")
        raise typer.Exit(1) from exc

    # Run initial full index so the watcher starts from a known-good state
    console.print(Panel(f"[bold cyan]Initial index[/bold cyan] {repo}", expand=False))
    try:
        indexer.index()
    except Exception as exc:
        err_console.print(f"[red]Initial indexing failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    # Start the file watcher
    try:
        watcher = FileWatcher(indexer, indexer.walker)
    except ImportError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
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
            + "\n".join(f"  [green]{e.alias}[/green]  {e.path}" for e in entries),
            expand=False,
        )
    )

    try:
        from trelix.indexing.multi_watcher import MultiRepoWatcher
    except ImportError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    watcher = MultiRepoWatcher(registry)

    import asyncio
    import signal

    stop_event = asyncio.Event()

    def _on_signal(*_: object) -> None:
        stop_event.set()

    try:
        asyncio.get_event_loop().add_signal_handler(signal.SIGINT, _on_signal)
        asyncio.get_event_loop().add_signal_handler(signal.SIGTERM, _on_signal)
    except (NotImplementedError, RuntimeError):
        # Windows / no event loop yet — fall through to KeyboardInterrupt
        pass

    console.print("[green]Watching for changes. Press Ctrl+C to stop.[/green]")

    try:
        asyncio.run(watcher.run(stop_event))
    except KeyboardInterrupt:
        pass

    stats = watcher.stats()
    console.print(
        f"\n[dim]Watch stopped. "
        f"Re-indexed: {stats['files_reindexed']} files | "
        f"Skipped (unchanged): {stats['files_skipped_unchanged']} files.[/dim]"
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

    api_app = create_app()
    typer.echo(f"trelix API serving {repo_path} at http://{host}:{port}")
    uvicorn.run(api_app, host=host, port=port)


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
    concepts: bool = typer.Option(False, "--concepts", "-c", help="Extract LLM semantic concepts"),
    json_output: bool = typer.Option(False, "--json", help="Output stats as JSON"),
) -> None:
    """Build the knowledge graph for an indexed repository.

    NOTE: The old 'trelix graph <repo> <symbol>' command for call-graph display has been
    renamed to 'trelix call-graph'. See 'trelix call-graph --help'.
    """
    from pathlib import Path as _Path

    from trelix.core.config import IndexConfig
    from trelix.graph.builder import GraphBuilder

    config = IndexConfig(repo_path=str(_Path(repo_path).resolve()))
    builder = GraphBuilder(config)

    with console.status("Building knowledge graph..."):
        result = builder.build(extract_concepts=concepts)

    if json_output:
        import json as _json

        data = {
            "node_count": result.node_count,
            "edge_count": result.edge_count,
            "community_count": result.community_count,
            "concept_count": result.concept_count,
        }
        console.print(_json.dumps(data))
        return

    console.print("[green]Knowledge Graph built[/green]")
    console.print(f"  Nodes      : {result.node_count}")
    console.print(f"  Edges      : {result.edge_count}")
    console.print(f"  Communities: {result.community_count}")
    if concepts:
        console.print(f"  Concepts   : {result.concept_count}")
    console.print(f"  Time       : {result.elapsed_seconds:.2f}s")

    if result.community_summary:
        console.print("\n[bold]Top Communities:[/bold]")
        for c in result.community_summary[:5]:
            files = ", ".join(c["top_files"][:3])
            console.print(f"  [{c['community_id']}] {c['size']} nodes — {files}")

    if visualize:
        from trelix.graph.visualizer import GraphVisualizer

        out = output or str(_Path(repo_path) / ".trelix" / "graph.html")
        viz = GraphVisualizer()
        path = viz.export_html(result.code_graph, out)
        console.print(f"\n[blue]Graph visualization:[/blue] {path}")


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

    for row in rows:
        table.add_row(
            row["ts"],
            row["query"][:50],
            row["intent"],
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
) -> None:
    """Evaluate retrieval quality against a golden query set (nDCG@10, Recall@10, MRR)."""
    from trelix.core.config import IndexConfig
    from trelix.eval.harness import EvalHarness

    config = IndexConfig(repo_path=repo)
    harness = EvalHarness(config)
    try:
        metrics = harness.run(golden)
    except FileNotFoundError:
        console.print(f"[red]Golden file not found: {golden}[/red]")
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
        console.print(f"[red]Golden file not found: {golden}[/red]")
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
    json_output: Annotated[bool, typer.Option("--json", help="Output raw JSON.")] = False,
) -> None:
    """Run Semgrep taint analysis and show source->sink flows.

    Requires: pip install trelix[taint]
    """
    import json as _json

    from trelix.analysis.taint import TaintAnalyzer
    from trelix.core.config import IndexConfig
    from trelix.store.db import Database

    config = IndexConfig(repo_path=str(Path(repo).resolve()))
    analyzer = TaintAnalyzer(repo_path=str(Path(repo).resolve()), tier=tier)
    with console.status("Running Semgrep taint analysis..."):
        flows = analyzer.run()

    if not flows:
        console.print(
            "[yellow]No taint flows found. "
            "Ensure semgrep is installed: pip install trelix[taint][/yellow]"
        )
        return

    # Persist to DB
    db = Database(config.db_path_absolute)
    db.insert_taint_flows(flows)

    filtered = [f for f in flows if not severity or f.severity == severity.upper()]

    if json_output:
        console.print(
            _json.dumps(
                [
                    {
                        "rule": f.rule_id,
                        "severity": f.severity,
                        "source": f"{f.source_file}:{f.source_line}",
                        "sink": f"{f.sink_file}:{f.sink_line}",
                    }
                    for f in filtered
                ],
                indent=2,
            )
        )
        return

    table = Table(title=f"Taint Flows ({len(filtered)} found)")
    table.add_column("Severity", style="bold red")
    table.add_column("Rule")
    table.add_column("Source")
    table.add_column("Sink")
    for f in filtered[:50]:
        table.add_row(
            f.severity,
            f.rule_id,
            f"{f.source_file}:{f.source_line}",
            f"{f.sink_file}:{f.sink_line}",
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
    import json as _json

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
            err_console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)

        console.print(f"[cyan]Fetching PR diff from GitHub:[/cyan] {pr}")
        gh_client = GitHubPRClient(token=token)

        try:
            pr_files = gh_client.get_pr_files(owner, repo_name, pr_number)
        except Exception as exc:
            err_console.print(f"[red]GitHub API error:[/red] {exc}")
            raise typer.Exit(1)

        # Build a unified diff string from PR files
        diff_lines: list[str] = []
        for f in pr_files:
            if f.patch is None:
                console.print(f"[dim]Skipping binary/oversized file: {f.filename}[/dim]")
                continue
            diff_lines.append(f"diff --git a/{f.filename} b/{f.filename}")
            diff_lines.append(f"--- a/{f.previous_filename or f.filename}")
            diff_lines.append(f"+++ b/{f.filename}")
            diff_lines.append(f.patch)
        pr_diff_str = "\n".join(diff_lines)

        if not pr_diff_str.strip():
            console.print("[yellow]No textual changes found in PR (all binary files?).[/yellow]")
            raise typer.Exit(0)

        reviewer = DiffReviewer(config)
        with console.status("Retrieving context and generating review..."):
            comments = reviewer.review(diff_text=pr_diff_str)

        if not comments:
            console.print("[green]No issues found.[/green]")
            raise typer.Exit(0)

        if json_output:
            console.print(
                _json.dumps(
                    [
                        {
                            "file": c.file_path,
                            "lines": f"{c.line_start}-{c.line_end}",
                            "severity": c.severity,
                            "comment": c.comment,
                        }
                        for c in comments
                    ],
                    indent=2,
                )
            )
        else:
            from rich.table import Table as _Table

            table = _Table(title=f"Review Results ({len(comments)} comments)")
            table.add_column("File", style="dim")
            table.add_column("Lines")
            table.add_column("Severity", style="bold")
            table.add_column("Comment", max_width=80)
            for c in comments:
                color = {"ERROR": "red", "WARN": "yellow", "INFO": "blue"}.get(c.severity, "white")
                table.add_row(
                    c.file_path,
                    f"{c.line_start}-{c.line_end}",
                    f"[{color}]{c.severity}[/{color}]",
                    c.comment,
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
                err_console.print(f"[yellow]Warning: failed to post comments: {exc}[/yellow]")

        raise typer.Exit(0)

    # ------------------------------------------------------------------
    # Local git diff path
    # ------------------------------------------------------------------
    parser = DiffParser()

    with console.status("Parsing diff..."):
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
    with console.status("Retrieving context and generating review..."):
        comments = reviewer.review(filtered)

    if not comments:
        console.print("[green]No issues found.[/green]")
        return

    if json_output:
        import json as _json

        console.print(
            _json.dumps(
                [
                    {
                        "file": c.file_path,
                        "lines": f"{c.line_start}-{c.line_end}",
                        "severity": c.severity,
                        "comment": c.comment,
                    }
                    for c in comments
                ],
                indent=2,
            )
        )
        return

    from rich.table import Table

    table = Table(title=f"Review Results ({len(comments)} comments)")
    table.add_column("File", style="dim")
    table.add_column("Lines")
    table.add_column("Severity", style="bold")
    table.add_column("Comment", max_width=80)
    for c in comments:
        color = {"ERROR": "red", "WARN": "yellow", "INFO": "blue"}.get(c.severity, "white")
        table.add_row(
            c.file_path,
            f"{c.line_start}-{c.line_end}",
            f"[{color}]{c.severity}[/{color}]",
            c.comment,
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
    import json as _json

    from trelix.federation.registry import RepoRegistry
    from trelix.federation.retriever import FederatedRetriever

    registry = RepoRegistry.load(config_file)
    if not registry.list():
        console.print(
            "[yellow]No repos registered. Use: trelix federation add <alias> <path>[/yellow]"
        )
        return

    fed = FederatedRetriever(registry)
    with console.status(f"Searching {len(registry.list())} repos..."):
        results = fed.retrieve(query, k=k)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    if json_output:
        console.print(
            _json.dumps(
                [
                    {
                        "file": r.file.rel_path,
                        "symbol": r.symbol.qualified_name,
                        "score": round(r.score, 4),
                        "source": r.source,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
        return

    from rich.table import Table

    table = Table(title=f"Federated Search: '{query}' ({len(results)} results)")
    table.add_column("Repo", style="dim")
    table.add_column("File")
    table.add_column("Symbol")
    table.add_column("Score", justify="right")
    for r in results[:20]:
        repo_tag = r.source.split(":")[0] if ":" in r.source else ""
        table.add_row(repo_tag, r.file.rel_path, r.symbol.qualified_name, f"{r.score:.4f}")
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
        console.print(f"[green]Registered '{alias}' -> {path}[/green]")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
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
        table.add_row(e.alias, e.path, str(e.weight))
    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
