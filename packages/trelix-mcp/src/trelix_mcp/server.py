import json
import logging
import sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[trelix-mcp] %(levelname)s %(message)s",
)

import signal  # noqa: E402
from typing import Any, Literal  # noqa: E402

from fastmcp import Context, FastMCP  # noqa: E402

from trelix.core.config import EmbedderConfig, IndexConfig  # noqa: E402
from trelix.indexing.indexer import Indexer  # noqa: E402
from trelix.retrieval.retriever import Retriever  # noqa: E402
from trelix.store.db import Database  # noqa: E402

mcp = FastMCP("trelix")
_log = logging.getLogger("trelix_mcp")


# ---------------------------------------------------------------------------
# MCP Resources (application-controlled URI-addressable data)
# MCP spec: Resources are passive data; Tools are callable functions.
# trelix:// is a custom URI scheme — fully permitted by the MCP spec.
# ---------------------------------------------------------------------------


@mcp.resource("trelix://index/stats")
def resource_index_stats() -> str:
    """Aggregate statistics for the active trelix index.

    Returns JSON with a usage hint — use the manifest template for repo-specific
    stats since direct resources cannot receive parameters.
    """
    return json.dumps(
        {"hint": "Use trelix://repo/{repo_path}/manifest for repo-specific stats"}
    )


@mcp.resource("trelix://repo/{repo_path}/manifest")
def resource_repo_manifest(repo_path: str) -> str:
    """List all indexed files in the repository at *repo_path*.

    Returns JSON with ``file_count`` and ``files[]`` list.
    Example URI: ``trelix://repo//Users/you/myrepo/manifest``
    """
    from trelix_mcp.resources import get_repo_manifest

    return get_repo_manifest(repo_path)


@mcp.resource("trelix://repo/{repo_path}/symbols/{qualified_name}")
def resource_symbol_source(repo_path: str, qualified_name: str) -> str:
    """Get full source code of a symbol by its qualified name.

    Returns JSON with ``qualified_name``, ``kind``, ``signature``, ``body``.
    Example URI: ``trelix://repo//Users/you/myrepo/symbols/AuthService.login``
    """
    from trelix_mcp.resources import get_symbol_source

    return get_symbol_source(repo_path, qualified_name)


# ---------------------------------------------------------------------------
# MCP Prompts (user-controlled reusable LLM interaction templates)
# ---------------------------------------------------------------------------


@mcp.prompt("trelix-search")
def prompt_search(query: str, repo_path: str) -> list[dict[str, str]]:
    """Structured prompt for semantic code search using trelix.

    Args:
        query: Natural-language or keyword search query.
        repo_path: Absolute path to the repository root.
    """
    from trelix_mcp.prompts import build_search_prompt

    return build_search_prompt(query=query, repo_path=repo_path)


@mcp.prompt("trelix-explain")
def prompt_explain(qualified_name: str, repo_path: str) -> list[dict[str, str]]:
    """Structured prompt for explaining a specific code symbol.

    Args:
        qualified_name: Fully-qualified symbol name, e.g. ``AuthService.login``.
        repo_path: Absolute path to the repository root.
    """
    from trelix_mcp.prompts import build_explain_prompt

    return build_explain_prompt(qualified_name=qualified_name, repo_path=repo_path)


@mcp.prompt("trelix-blast-radius")
def prompt_blast_radius(symbol_name: str, repo_path: str) -> list[dict[str, str]]:
    """Structured prompt for impact analysis before refactoring a symbol.

    Args:
        symbol_name: Name or qualified name of the symbol to analyse.
        repo_path: Absolute path to the repository root.
    """
    from trelix_mcp.prompts import build_blast_radius_prompt

    return build_blast_radius_prompt(symbol_name=symbol_name, repo_path=repo_path)


@mcp.tool()
def search_code(
    query: str,
    repo_path: str,
    k: int = 10,
    cursor: int = 0,
) -> dict:
    """
    Search the indexed codebase using natural language queries.

    ⚠️ IMPORTANT:
    - repo_path must be an ABSOLUTE path to an already-indexed repository.
    - Run index_codebase first if you receive an error about a missing index.

    🎯 When to Use:
    - Find specific functions, classes, or implementations
    - Understand architecture before making changes
    - Locate all callers of a function before refactoring
    - Find similar patterns to follow when adding code

    📄 Pagination:
    - Use cursor=0 for first page (default).
    - If next_cursor is not null, pass it as cursor for the next page.
    - k controls page size.

    Returns:
        {"results": [...], "next_cursor": int|null, "total_available": int}
    """
    _log.info("search_code query=%r repo=%s k=%d cursor=%d", query, repo_path, k, cursor)
    config = IndexConfig(repo_path=repo_path)
    ctx = Retriever(config).retrieve(query)
    all_results = ctx.results

    page = all_results[cursor : cursor + k]
    next_cursor = cursor + k if cursor + k < len(all_results) else None

    return {
        "results": [
            {
                "file": r.file.rel_path,
                "symbol": r.symbol.qualified_name,
                "kind": r.symbol.kind.value,
                "lines": f"{r.symbol.line_start}-{r.symbol.line_end}",
                "score": round(r.score, 4),
                "source": r.source,
                "body": r.symbol.body[:800],
                "language": r.file.language.value,
            }
            for r in page
        ],
        "next_cursor": next_cursor,
        "total_available": len(all_results),
    }


@mcp.tool()
def index_codebase(
    repo_path: str,
    provider: Literal["local", "openai", "azure", "voyage", "local-code"] = "local",
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    Index a repository for code search. Run once before calling search_code.

    ⚠️ IMPORTANT:
    - Stores the index in <repo_path>/.trelix/index.db (zero external infra).
    - Re-run to refresh after large code changes; incremental update is fast.

    ✨ Providers:
    - local   — no API key, CPU-only, fast for small repos
    - openai  — requires OPENAI_API_KEY, best quality
    - azure   — requires AZURE_API_KEY + AZURE_ENDPOINT
    - voyage  — requires VOYAGE_API_KEY, best code-specific quality

    Progress notifications are sent if the MCP client supports them.
    """
    _log.info("index_codebase repo=%s provider=%s", repo_path, provider)

    embedder_config = EmbedderConfig(provider=provider)  # type: ignore[call-arg]
    config = IndexConfig(repo_path=repo_path, embedder=embedder_config)

    # Send progress notifications if client supports it (best-effort, never blocks)
    if ctx is not None:
        try:
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                ctx.report_progress(0, 3)  # stage 0/3: starting
            )
        except Exception:
            pass

    stats = Indexer(config, quiet=True).index()

    if ctx is not None:
        try:
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                ctx.report_progress(3, 3)  # stage 3/3: done
            )
        except Exception:
            pass

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


@mcp.tool()
def build_knowledge_graph(repo_path: str, extract_concepts: bool = False) -> dict:
    """
    Build a knowledge graph for an indexed codebase.

    ⚠️ IMPORTANT: Run index_codebase first. repo_path must be absolute.

    🎯 What this builds:
    - Unified code property graph (calls + imports + type hierarchy)
    - Community detection: clusters modules into architectural groups
    - Optional LLM concept extraction (set extract_concepts=True, requires LLM config)

    ✨ Returns:
    - node_count: number of symbols in the graph
    - edge_count: number of structural relationships
    - community_count: detected architectural clusters
    - community_summary: top files + symbols per cluster
    """
    from trelix.core.config import IndexConfig
    from trelix.graph.builder import GraphBuilder

    _log.info("build_knowledge_graph repo=%s concepts=%s", repo_path, extract_concepts)
    config = IndexConfig(repo_path=repo_path)
    result = GraphBuilder(config).build(extract_concepts=extract_concepts)
    return {
        "node_count": result.node_count,
        "edge_count": result.edge_count,
        "community_count": result.community_count,
        "concept_count": result.concept_count,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "community_summary": result.community_summary,
    }


@mcp.tool()
def graph_search_mcp(query: str, repo_path: str, k: int = 10) -> list[dict]:
    """
    Graph-traversal search: find structurally related symbols by starting
    from semantically similar seeds and following code relationships.

    ⚠️ IMPORTANT: Run index_codebase and optionally build_knowledge_graph first.

    🎯 When to use:
    - "What other code is connected to X?" — follow call/import/type edges
    - "Find the blast radius of a class" — who calls or imports it?
    - "What lives in the same architectural cluster as X?"
    """
    from trelix.core.config import IndexConfig
    from trelix.graph.builder import GraphBuilder
    from trelix.graph.search import graph_search
    from trelix.retrieval.retriever import Retriever

    _log.info("graph_search_mcp query=%r repo=%s k=%d", query, repo_path, k)
    config = IndexConfig(repo_path=repo_path)

    # First find seed symbols via standard retrieval
    ctx = Retriever(config).retrieve(query)
    seed_ids = [r.chunk.symbol_id for r in ctx.results[:5]]

    if not seed_ids:
        return []

    # Then expand via graph — reuse the DB already opened by GraphBuilder/CodeGraph
    build_result = GraphBuilder(config).build(extract_concepts=False)
    db = build_result.code_graph._db
    graph_results = graph_search(db, build_result.code_graph, seed_ids, depth=2, max_results=k)

    return [
        {
            "file": r.file.rel_path,
            "symbol": r.symbol.qualified_name,
            "kind": r.symbol.kind.value,
            "score": round(r.score, 4),
            "source": r.source,
            "body": r.symbol.body[:600],
        }
        for r in graph_results[:k]
    ]


def main() -> None:
    """Entry point for the trelix-mcp server (stdio transport)."""

    def _handle_sigterm(signum: int, frame: Any) -> None:
        _log.info("Received SIGTERM — shutting down")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    _log.info("trelix-mcp starting (transport=stdio)")
    mcp.run(transport="stdio")
