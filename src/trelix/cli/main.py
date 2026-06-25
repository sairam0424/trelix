"""
trelix CLI — placeholder until Phase 14 is implemented.

All commands are stubbed so `trelix --help` works immediately after install.
Full implementations land in feature/phase-14-cli.
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="trelix",
    help="Fast, reliable code indexing and retrieval.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()

_NOT_IMPLEMENTED = "[yellow]This command will be implemented in Phase 14.[/yellow]"


@app.command()
def index(
    repo: str = typer.Argument(..., help="Path to the repository to index"),
    provider: str = typer.Option("local", help="Embedding provider: local | openai | azure"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
) -> None:
    """Index a repository — builds the search index at <repo>/.trelix/index.db"""
    console.print(_NOT_IMPLEMENTED)


@app.command()
def search(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    query: str = typer.Argument(..., help="Natural language query"),
    provider: str = typer.Option("local", help="Embedding provider: local | openai | azure"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Search for code — returns ranked results as JSON"""
    console.print(_NOT_IMPLEMENTED)


@app.command()
def ask(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    query: str = typer.Argument(..., help="Question to answer about the codebase"),
    provider: str = typer.Option("local", help="Embedding provider: local | openai | azure"),
) -> None:
    """Ask a question — retrieval + LLM synthesis (requires OPENAI_API_KEY)"""
    console.print(_NOT_IMPLEMENTED)


@app.command()
def query(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    query_str: str = typer.Argument(..., metavar="QUERY", help="Natural language query"),
    provider: str = typer.Option("local", help="Embedding provider: local | openai | azure"),
) -> None:
    """Query a repository — human-readable terminal output (no LLM synthesis)"""
    console.print(_NOT_IMPLEMENTED)


@app.command()
def stats(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
) -> None:
    """Show index statistics (files, symbols, chunks, DB size)"""
    console.print(_NOT_IMPLEMENTED)


@app.command("update-index")
def update_index(
    repo: str = typer.Argument(..., help="Path to the indexed repository"),
    file: str = typer.Argument(..., help="File to re-index (absolute or relative to repo)"),
) -> None:
    """Re-index a single file after editing"""
    console.print(_NOT_IMPLEMENTED)
