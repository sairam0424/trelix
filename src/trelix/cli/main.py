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
from typing import Literal, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

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

    try:
        retriever = Retriever(config)
        context = retriever.retrieve(query)
    except Exception as exc:
        err_console.print(f"[red]Retrieval failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    # If provider=local (no API key), print the context text directly
    if provider == "local":
        console.print(Panel(f"[bold cyan]Context for:[/bold cyan] {query}", expand=False))
        if context.context_text:
            console.print(context.context_text)
        else:
            console.print("[yellow]No relevant code found.[/yellow]")
        return

    try:
        synth = Synthesizer(config.embedder, llm_config=config.llm)
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
) -> None:
    """Migrate embeddings from SQLite to Qdrant (or another backend)."""
    _setup_logging(False)

    import sqlite3
    import struct

    from pydantic import ValidationError as _PydanticValidationError

    from trelix.core.config import IndexConfig, StoreConfig
    from trelix.store.vector_qdrant import QdrantVectorStore

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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
