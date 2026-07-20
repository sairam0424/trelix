import json
import logging
import os
import sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[trelix-mcp] %(levelname)s %(message)s",
)

import signal  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Literal  # noqa: E402

from fastmcp import Context, FastMCP  # noqa: E402

from trelix.agent.loop import AgentLoop  # noqa: E402
from trelix.core.config import EmbedderConfig, IndexConfig, RetrievalConfig  # noqa: E402
from trelix.federation.registry import RepoRegistry  # noqa: E402
from trelix.federation.retriever import FederatedRetriever  # noqa: E402
from trelix.indexing.indexer import Indexer  # noqa: E402
from trelix.retrieval.retriever import Retriever  # noqa: E402
from trelix.store.db import Database  # noqa: E402
from trelix_mcp.subscriptions import SubscriptionLimitExceeded, SubscriptionRegistry  # noqa: E402

mcp = FastMCP("trelix")
_log = logging.getLogger("trelix_mcp")

# Global subscription registry — tracks which MCP clients are watching which
# trelix:// resource URIs.  notify_file_changed() fires notifications to all
# active subscribers when trelix watch detects a file change.
_subscription_registry = SubscriptionRegistry(
    max_subscribers=int(os.environ.get("TRELIX_MCP_MAX_SUBSCRIBERS", "1000")),
    ttl_seconds=float(os.environ.get("TRELIX_MCP_SUBSCRIPTION_TTL_SECONDS", "3600")),
)

# ---------------------------------------------------------------------------
# MCP spec 2024-11-05 §Resources — declare resources.subscribe=True so that
# MCP clients (Claude Code, Cursor, VS Code Copilot) know they may send
# resources/subscribe requests.  FastMCP's low-level SDK hardcodes
# subscribe=False when it builds ServerCapabilities, so we patch the
# get_capabilities method on the server instance after construction.
# subscribe and listChanged are independent optional fields; we only
# opt-in to subscribe here — listChanged is handled separately by FastMCP's
# notification_options.
# ---------------------------------------------------------------------------

_orig_get_capabilities = mcp._mcp_server.get_capabilities


def _get_capabilities_with_subscribe(notification_options, experimental_capabilities):
    """Wrap get_capabilities to advertise resources.subscribe=True."""
    caps = _orig_get_capabilities(notification_options, experimental_capabilities)
    if caps.resources is not None:
        caps = caps.model_copy(
            update={"resources": caps.resources.model_copy(update={"subscribe": True})}
        )
    return caps


mcp._mcp_server.get_capabilities = _get_capabilities_with_subscribe  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# MCP Resource Subscription handlers (MCP spec 2024-11-05 §Resources)
# Wire protocol: resources/subscribe → notifications/resources/updated → resources/read
# The subscribe/unsubscribe tools register/deregister URIs in _subscription_registry.
# Callers (trelix watch) call notify_file_changed() to fire the push notifications.
# ---------------------------------------------------------------------------


@mcp.tool()
def subscribe_resource(uri: str, subscription_id: str) -> dict:
    """Register a subscription for a trelix:// resource URI.

    MCP clients call this after receiving resources.subscribe=True in capabilities.
    When the resource changes (e.g. trelix watch detects a file edit), the server
    pushes notifications/resources/updated carrying only the URI — no content.
    The client then calls resources/read to fetch the updated content.

    Args:
        uri: The trelix:// resource URI to watch (e.g. trelix://repo//path/manifest).
        subscription_id: Client-chosen correlation ID included in _meta of notifications.
    """
    try:
        _subscription_registry.subscribe(uri, subscription_id)
    except SubscriptionLimitExceeded as exc:
        _log.warning(
            "Subscription rejected (at capacity): uri=%s subscription_id=%s",
            uri,
            subscription_id,
        )
        return {
            "subscribed": False,
            "uri": uri,
            "subscription_id": subscription_id,
            "error": str(exc),
        }
    _log.info("Subscribed: uri=%s subscription_id=%s", uri, subscription_id)
    return {"subscribed": True, "uri": uri, "subscription_id": subscription_id}


@mcp.tool()
def unsubscribe_resource(subscription_id: str) -> dict:
    """Deregister a resource subscription by its subscription ID.

    Args:
        subscription_id: The ID returned when subscribe_resource was called.
    """
    uri = _subscription_registry.get_uri(subscription_id)
    _subscription_registry.unsubscribe(subscription_id)
    _log.info("Unsubscribed: subscription_id=%s uri=%s", subscription_id, uri)
    return {"unsubscribed": True, "subscription_id": subscription_id}


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
    return json.dumps({"hint": "Use trelix://repo/{repo_path}/manifest for repo-specific stats"})


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

    def _send_progress(current: int, total: int) -> None:
        """Send MCP progress notification — best-effort, never raises."""
        if ctx is None:
            return
        try:
            import asyncio

            loop = asyncio.get_running_loop()
            loop.create_task(ctx.report_progress(current, total))
        except RuntimeError:
            # No running loop (sync context) — skip progress notification silently
            pass
        except Exception:
            pass

    _send_progress(0, 3)
    stats = Indexer(config, quiet=True).index()
    _send_progress(3, 3)

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


# Fixed, cursor-independent per-repo fan-out width for federation_search_all.
# Wide enough to cover any realistic single-page request without letting the
# per-repo candidate pool (and therefore the RRF fusion input) change shape
# as `cursor` grows — see federation_search_all's docstring.
_FEDERATION_SEARCH_ALL_FETCH_WIDTH = 100


class ConfigPathNotAllowedError(ValueError):
    """Raised when a caller-supplied federation config_path resolves outside
    every allowlisted root."""


def _confine_federation_config_path(config_path: str | None) -> str | None:
    """Resolve and confine a caller-supplied federation config_path.

    Mirrors the path-confinement pattern documented in SECURITY.md for
    GET /graph/visualize (src/trelix/api/app.py) — canonicalize with
    Path.resolve(), then require the result live under an allowlisted root.
    Uses Path.is_relative_to() rather than a naive string startswith() check
    (startswith("/repo/.trelix") would incorrectly also match a sibling
    directory named "/repo/.trelixevil").

    Allowlisted roots:
    - ~/.config/trelix/ (RepoRegistry's default config directory)
    - <cwd>/.trelix/ (a repo-local override, when the MCP server process is
      launched from within a repo — these 4 tools have no repo_path param
      of their own to derive a repo root from, so the process cwd is the
      closest available analog to "the repo-local .trelix/" the docstring
      already promises)

    Returns None unchanged (the RepoRegistry default). Raises
    ConfigPathNotAllowedError if config_path resolves outside both roots.
    """
    if config_path is None:
        return None

    from trelix.federation.registry import _DEFAULT_CONFIG

    resolved = Path(config_path).resolve()
    allowed_roots = [_DEFAULT_CONFIG.parent, Path.cwd() / ".trelix"]
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
        raise ConfigPathNotAllowedError(
            f"config_path must resolve inside one of {[str(r) for r in allowed_roots]}, "
            f"got {resolved}"
        )
    return str(resolved)


@mcp.tool()
def federation_list_repos(config_path: str | None = None) -> dict:
    """List all repos registered for federated (multi-repo) search.

    Args:
        config_path: Optional path to a custom repos.json. Must resolve
            inside ~/.config/trelix/ or <cwd>/.trelix/. Defaults to
            ~/.config/trelix/repos.json.

    Returns:
        {"repos": [{"alias": str, "path": str, "weight": float}, ...],
         "count": int, "error": str|None}
    """
    _log.info("federation_list_repos config_path=%r", config_path)
    try:
        confined_path = _confine_federation_config_path(config_path)
    except ConfigPathNotAllowedError as exc:
        return {"repos": [], "count": 0, "error": str(exc)}
    registry = RepoRegistry.load(confined_path)
    entries = registry.list()
    return {
        "repos": [{"alias": e.alias, "path": e.path, "weight": e.weight} for e in entries],
        "count": len(entries),
        "error": None,
    }


@mcp.tool()
def federation_add_repo(
    alias: str,
    path: str,
    weight: float = 1.0,
    config_path: str | None = None,
) -> dict:
    """Register a repo for federated search across MCP tool calls.

    ⚠️ IMPORTANT:
    - path must be an ABSOLUTE path.
    - Run index_codebase on it separately before federation_search_all can
      return results from it — registering a repo does not index it.
    - The registry is capped at TRELIX_FEDERATION_MAX_REPOS entries
      (default 50) to prevent unbounded growth from a scripted client.

    Args:
        alias: Short unique name for the repo (e.g. "auth-service").
        path: Absolute path to the repo root.
        weight: RRF weight multiplier (default 1.0; higher = ranked higher
            in federation_search_all's fused results).
        config_path: Optional path to a custom repos.json. Must resolve
            inside ~/.config/trelix/ or <cwd>/.trelix/.

    Returns:
        {"added": bool, "alias": str, "path": str, "error": str|None}
    """
    _log.info("federation_add_repo alias=%r path=%r weight=%s", alias, path, weight)
    try:
        confined_path = _confine_federation_config_path(config_path)
    except ConfigPathNotAllowedError as exc:
        return {"added": False, "alias": alias, "path": path, "error": str(exc)}
    registry = RepoRegistry.load(confined_path)
    max_repos = RetrievalConfig().federation_max_repos
    try:
        registry.add(alias, path, weight, max_repos=max_repos)
        registry.save()
        return {"added": True, "alias": alias, "path": path, "error": None}
    except ValueError as exc:
        return {"added": False, "alias": alias, "path": path, "error": str(exc)}


@mcp.tool()
def federation_remove_repo(alias: str, config_path: str | None = None) -> dict:
    """Unregister a repo from federated search by alias.

    Args:
        alias: The alias to remove. No-op (removed=False) if not registered.
        config_path: Optional path to a custom repos.json. Must resolve
            inside ~/.config/trelix/ or <cwd>/.trelix/.

    Returns:
        {"removed": bool, "alias": str, "error": str|None}
    """
    _log.info("federation_remove_repo alias=%r", alias)
    try:
        confined_path = _confine_federation_config_path(config_path)
    except ConfigPathNotAllowedError as exc:
        return {"removed": False, "alias": alias, "error": str(exc)}
    registry = RepoRegistry.load(confined_path)
    existed = any(e.alias == alias for e in registry.list())
    registry.remove(alias)
    registry.save()
    return {"removed": existed, "alias": alias, "error": None}


@mcp.tool()
def federation_search_all(
    query: str,
    k: int = 10,
    cursor: int = 0,
    config_path: str | None = None,
) -> dict:
    """Search across ALL registered repos simultaneously (federated search).

    ⚠️ IMPORTANT:
    - Requires repos to already be registered via federation_add_repo AND
      already indexed (run index_codebase on each repo path beforehand).
    - Results are merged via Reciprocal Rank Fusion weighted by each repo's
      registered weight, then deduplicated.
    - Only the first TRELIX_FEDERATION_MAX_REPOS registered repos (default
      50) are actually queried; repos_skipped reports how many were
      omitted.

    🎯 When to Use:
    - Cross-service / cross-repo questions ("where is auth handled across
      our microservices?")
    - You don't know which of several registered repos contains the answer.

    📄 Pagination: same cursor/k contract as search_code — pages are sliced
    from one fixed-width fetch, independent of cursor, so page contents are
    stable across calls (results don't shift/duplicate/vanish between
    pages the way a cursor-scaled fetch width would cause).

    Returns:
        {"results": [...], "next_cursor": int|None, "total_available": int,
         "repos_searched": int, "repos_skipped": int, "error": str|None}
    """
    _log.info("federation_search_all query=%r k=%d cursor=%d", query, k, cursor)
    try:
        confined_path = _confine_federation_config_path(config_path)
    except ConfigPathNotAllowedError as exc:
        return {
            "results": [],
            "next_cursor": None,
            "total_available": 0,
            "repos_searched": 0,
            "repos_skipped": 0,
            "error": str(exc),
        }
    registry = RepoRegistry.load(confined_path)
    entries = registry.list()
    if not entries:
        return {
            "results": [],
            "next_cursor": None,
            "total_available": 0,
            "repos_searched": 0,
            "repos_skipped": 0,
            "error": None,
        }

    max_repos = RetrievalConfig().federation_max_repos
    fed = FederatedRetriever(registry, max_repos=max_repos)
    repos_searched = fed.repos_queried_count(len(entries))
    repos_skipped = len(entries) - repos_searched

    all_results = fed.retrieve(query, k=_FEDERATION_SEARCH_ALL_FETCH_WIDTH)

    page = all_results[cursor : cursor + k]
    next_cursor = cursor + k if cursor + k < len(all_results) else None

    return {
        "results": [
            {
                "repo": r.source.split(":")[0] if ":" in r.source else "",
                "file": r.file.rel_path,
                "symbol": r.symbol.qualified_name,
                "kind": r.symbol.kind.value,
                "score": round(r.score, 4),
                "source": r.source,
                "body": r.symbol.body[:800],
                "language": r.file.language.value,
            }
            for r in page
        ],
        "next_cursor": next_cursor,
        "total_available": len(all_results),
        "repos_searched": repos_searched,
        "repos_skipped": repos_skipped,
        "error": None,
    }


@mcp.tool()
def ask_agent(
    query: str,
    repo_path: str,
    session_id: str | None = None,
) -> dict:
    """Ask a question using the multi-turn ReAct agentic loop, with persistent memory.

    ⚠️ IMPORTANT:
    - repo_path must be an ABSOLUTE path to an already-indexed repository.
    - Session history is scoped to (repo_path, session_id) — a session_id
      created against one repo is invisible when querying a different repo_path.
    - Requires LLM configuration (e.g. OPENAI_API_KEY) — this tool always
      uses the agentic loop, unlike search_code which is retrieval-only.

    🎯 When to Use:
    - Multi-step questions needing iterative retrieve/grep/get_symbol drilling.
    - Follow-up questions in the same conversation — pass back the session_id
      returned from the previous call to preserve context across calls.

    Session lifecycle:
    - Omit session_id on the first call — a new one is generated and returned.
    - Pass that session_id on subsequent related calls to resume with full
      turn history loaded from persistent storage.
    - Sessions are automatically evicted after
      TRELIX_RETRIEVAL_AGENT_SESSION_MAX_AGE_SECONDS of inactivity (default
      7 days). Use agent_clear_session to delete one explicitly.

    Returns:
        {"answer": str, "session_id": str, "turn_count": int}
    """
    _log.info("ask_agent query=%r repo=%s session_id=%r", query, repo_path, session_id)
    config = IndexConfig(repo_path=repo_path)
    config.retrieval.agentic_enabled = True
    loop = AgentLoop(config)
    answer, resolved_session_id = loop.run(query, session_id=session_id)

    db = Database(config.db_path_absolute)
    try:
        turns = db.get_agent_turns(resolved_session_id)
    finally:
        db.close()

    return {"answer": answer, "session_id": resolved_session_id, "turn_count": len(turns)}


@mcp.tool()
def agent_list_sessions(repo_path: str, limit: int = 50) -> dict:
    """List recent agent sessions for a repo, most recently active first.

    Args:
        repo_path: Absolute path to the repository root.
        limit: Max sessions to return (default 50).

    Returns:
        {"sessions": [{"session_id", "created_at", "last_active_at", "query",
         "turn_count"}, ...], "count": int}
    """
    _log.info("agent_list_sessions repo=%s limit=%d", repo_path, limit)
    config = IndexConfig(repo_path=repo_path)
    db = Database(config.db_path_absolute)
    try:
        max_age = config.retrieval.agent_session_max_age_seconds
        if max_age > 0:
            db.evict_stale_agent_sessions(max_age)
        sessions = db.list_agent_sessions(limit=limit)
    finally:
        db.close()
    return {"sessions": sessions, "count": len(sessions)}


@mcp.tool()
def agent_clear_session(repo_path: str, session_id: str) -> dict:
    """Delete a persisted agent session and all its turn history.

    Args:
        repo_path: Absolute path to the repository root.
        session_id: The session to delete.

    Returns:
        {"cleared": bool, "session_id": str}
    """
    _log.info("agent_clear_session repo=%s session_id=%r", repo_path, session_id)
    config = IndexConfig(repo_path=repo_path)
    db = Database(config.db_path_absolute)
    try:
        existed = db.delete_agent_session(session_id)
    finally:
        db.close()
    return {"cleared": existed, "session_id": session_id}


def main() -> None:
    """Entry point for the trelix-mcp server (stdio transport)."""

    def _handle_sigterm(signum: int, frame: Any) -> None:
        _log.info("Received SIGTERM — shutting down")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    _log.info("trelix-mcp starting (transport=stdio)")
    mcp.run(transport="stdio")
