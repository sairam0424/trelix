import logging
import sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[trelix-mcp] %(levelname)s %(message)s",
)

import signal
from typing import Any, Literal

from fastmcp import FastMCP

from trelix.core.config import EmbedderConfig, IndexConfig
from trelix.indexing.indexer import Indexer
from trelix.retrieval.retriever import Retriever
from trelix.store.db import Database

mcp = FastMCP("trelix")
_log = logging.getLogger("trelix_mcp")


@mcp.tool()
def search_code(query: str, repo_path: str, k: int = 10) -> list[dict[str, Any]]:
    """Search a codebase for symbols semantically relevant to *query*.

    Args:
        query: Natural-language or keyword search query.
        repo_path: Absolute path to the repository root (must already be indexed).
        k: Maximum number of results to return (default 10).

    Returns:
        List of result dicts with keys: file, symbol, kind, lines, score, source,
        body, language.
    """
    _log.info("search_code query=%r repo_path=%r k=%d", query, repo_path, k)
    config = IndexConfig(repo_path=repo_path)
    retriever = Retriever(config)
    context = retriever.retrieve(query)
    results = context.results[:k]
    return [
        {
            "file": r.file.rel_path,
            "symbol": r.symbol.qualified_name,
            "kind": r.symbol.kind,
            "lines": [r.symbol.line_start, r.symbol.line_end],
            "score": round(r.score, 4),
            "source": r.source,
            "body": r.symbol.body,
            "language": r.file.language,
        }
        for r in results
    ]


@mcp.tool()
def index_codebase(
    repo_path: str,
    provider: Literal["local", "openai", "azure", "voyage", "local-code"] = "local",
) -> dict[str, Any]:
    """Index a codebase so it can be searched with search_code.

    Args:
        repo_path: Absolute path to the repository root.
        provider: Embedding provider — "local" requires no API key.

    Returns:
        Indexing statistics dict with keys: files_found, files_indexed,
        files_skipped, symbols_extracted, chunks_total, chunks_embedded,
        errors, elapsed_seconds.
    """
    _log.info("index_codebase repo_path=%r provider=%r", repo_path, provider)
    embedder_config = EmbedderConfig(provider=provider)  # type: ignore[call-arg]
    config = IndexConfig(repo_path=repo_path, embedder=embedder_config)
    stats = Indexer(config, quiet=True).index()
    return stats


@mcp.tool()
def get_symbol(qualified_name: str, repo_path: str) -> dict[str, Any] | None:
    """Look up a symbol by its fully-qualified name.

    Args:
        qualified_name: e.g. "MyClass.my_method" or "my_function".
        repo_path: Absolute path to the repository root.

    Returns:
        Symbol dict or None if not found.  Keys: name, qualified_name, kind,
        file, line_start, line_end, signature, docstring, body, language.
    """
    _log.info("get_symbol qualified_name=%r repo_path=%r", qualified_name, repo_path)
    config = IndexConfig(repo_path=repo_path)
    db = Database(config.db_path_absolute)
    rows = db.get_symbol_by_name(qualified_name)
    if not rows:
        # Fall back to bare name lookup
        name_only = qualified_name.split(".")[-1]
        rows = db.get_symbol_by_name(name_only)
        # Filter to exact qualified_name match when possible
        exact = [s for s in rows if s.qualified_name == qualified_name]
        if exact:
            rows = exact

    if not rows:
        return None

    sym = rows[0]
    sym_file = db.get_symbol_with_file(sym.id)  # type: ignore[arg-type]
    if sym_file is None:
        return None
    symbol, file = sym_file
    return {
        "name": symbol.name,
        "qualified_name": symbol.qualified_name,
        "kind": symbol.kind,
        "file": file.rel_path,
        "line_start": symbol.line_start,
        "line_end": symbol.line_end,
        "signature": symbol.signature,
        "docstring": symbol.docstring,
        "body": symbol.body,
        "language": file.language,
    }


@mcp.tool()
def blast_radius(symbol_name: str, repo_path: str) -> list[dict[str, Any]]:
    """Find all symbols that depend on (call or import) a given symbol.

    Useful for impact analysis: "if I change X, what else might break?"

    Args:
        symbol_name: Name or qualified name of the symbol to analyse.
        repo_path: Absolute path to the repository root.

    Returns:
        Deduplicated list of dependent-symbol dicts with keys: file, symbol,
        kind, line_start, language.
    """
    _log.info("blast_radius symbol_name=%r repo_path=%r", symbol_name, repo_path)
    query = f"blast radius dependencies of {symbol_name}"
    config = IndexConfig(repo_path=repo_path)
    retriever = Retriever(config)
    context = retriever.retrieve(query)

    seen_files: set[str] = set()
    output: list[dict[str, Any]] = []
    for r in context.results:
        file_key = r.file.rel_path
        if file_key in seen_files:
            continue
        seen_files.add(file_key)
        output.append(
            {
                "file": r.file.rel_path,
                "symbol": r.symbol.qualified_name,
                "kind": r.symbol.kind,
                "line_start": r.symbol.line_start,
                "language": r.file.language,
            }
        )
    return output


def main() -> None:
    """Entry point for the trelix-mcp server (stdio transport)."""

    def _handle_sigterm(signum: int, frame: Any) -> None:
        _log.info("Received SIGTERM — shutting down")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    _log.info("trelix-mcp starting (transport=stdio)")
    mcp.run(transport="stdio")
