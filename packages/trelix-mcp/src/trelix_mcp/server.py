import argparse
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
import threading  # noqa: E402
from collections import OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Literal  # noqa: E402

from fastmcp import Context, FastMCP  # noqa: E402
from fastmcp.exceptions import ResourceError, ToolError  # noqa: E402
from fastmcp.prompts import Message  # noqa: E402
from fastmcp.tools.base import InputRequiredToolResult  # noqa: E402
from mcp.types import (  # noqa: E402
    ElicitRequest,
    ElicitRequestFormParams,
    ElicitResult,
    InputRequiredResult,
)

from trelix.agent.loop import AgentLoop  # noqa: E402
from trelix.core.config import EmbedderConfig, IndexConfig, RetrievalConfig  # noqa: E402
from trelix.core.confinement import resolve_allowed_roots  # noqa: E402
from trelix.core.index_check import IndexNotFoundError, require_index  # noqa: E402
from trelix.core.models import IndexedFile, Symbol  # noqa: E402
from trelix.federation.registry import _DEFAULT_CONFIG, RepoRegistry  # noqa: E402
from trelix.federation.retriever import FederatedRetriever  # noqa: E402
from trelix.indexing.indexer import Indexer  # noqa: E402
from trelix.retrieval.retriever import Retriever  # noqa: E402
from trelix.store.db import Database  # noqa: E402
from trelix_mcp import __version__  # noqa: E402
from trelix_mcp.arguments import (  # noqa: E402
    check_absolute_repo_dir,
    check_positive_weight,
    check_repo_dir,
    check_session_id,
    check_text,
)
from trelix_mcp.budget import (  # noqa: E402
    DEFAULT_BLAST_LIMIT,
    DEFAULT_MAX_BODY_CHARS,
    MAX_BLAST_LIMIT,
    MAX_RESULT_CHARS_ENV,
    BudgetConfigError,
    Detail,
    Limits,
    bare_array_result,
    blast_radius_remedy,
    body_or_signature,
    cap_session_query,
    check_body_limit,
    check_cursor,
    clamp_page_size,
    empty_bare_array,
    fit_page,
    fit_rows,
    limits_from_env,
    null_result,
    truncate_body,
)
from trelix_mcp.confinement import (  # noqa: E402
    REFUSAL,
    RepoConfinementMiddleware,
    entries_inside_roots,
    inside_roots,
)
from trelix_mcp.subscriptions import SubscriptionLimitExceeded, SubscriptionRegistry  # noqa: E402
from trelix_mcp.tool_metadata import (  # noqa: E402
    LISTING_CACHE_SCOPE,
    LISTING_CACHE_TTL_SECONDS,
    SERVER_INSTRUCTIONS,
    TOOL_PROFILES,
    ToolMetadata,
    apply_tool_profile,
)

mcp = FastMCP(
    "trelix",
    version=__version__,
    instructions=SERVER_INSTRUCTIONS,
    cache_ttl=LISTING_CACHE_TTL_SECONDS,
    cache_scope=LISTING_CACHE_SCOPE,
    transforms=[ToolMetadata()],
)
_log = logging.getLogger("trelix_mcp")

# Retriever construction is the expensive part of every tool call that uses
# one -- for the `local` embedder specifically, make_embedder() loads a
# SentenceTransformer model from disk, several seconds every time. An MCP
# server is a long-lived process serving many tool calls, unlike a CLI
# command's one-shot invocation, so it's worth reusing the same Retriever
# across calls against the same repo instead of rebuilding it from scratch
# on every search_code/graph_search_mcp call. Keyed on the resolved absolute
# path so relative/absolute spellings of the same repo share one entry.
# Invalidated by index_codebase (see there) -- a re-index can switch
# embedder providers, which changes what Retriever.__init__ needs to build.
# Bounded: an LRU of TRELIX_MCP_RETRIEVER_CACHE_SIZE entries (default 8), since
# each one may hold an embedding model and a client can name any number of
# distinct repositories. Insertion order is recency: a hit moves its entry to
# the end, and _get_retriever drops entries from the front past the bound.
_retriever_cache: OrderedDict[str, Retriever] = OrderedDict()
_retriever_cache_lock = threading.Lock()

# The repository roots every repo_path, federation path and trelix://repo/... URI must resolve
# inside, set once by main() from --root and TRELIX_ALLOWED_REPO_ROOTS. Empty (the stdio default
# with neither) confines nothing, which is what every client got before the flag existed.
_allowed_roots: tuple[Path, ...] = ()


def _install_confinement(roots: tuple[Path, ...]) -> None:
    """Confine tool arguments and resource URIs to `roots`; with no roots, install nothing."""
    global _allowed_roots
    _allowed_roots = roots
    if not roots:
        return
    _log.info("Confining repo paths to %s", [str(root) for root in roots])
    mcp.add_middleware(RepoConfinementMiddleware(roots))


def _confine_resource_repo(repo_path: str) -> None:
    """Refuse a trelix://repo/... read whose repository lies outside the allowed roots.

    Called first in each repo resource handler, where FastMCP has already parsed the URI into
    `repo_path`, so the same value is checked and then used (a second URI parser in a middleware
    could disagree with FastMCP's). A `ResourceError` reaches the client with this text verbatim.
    Nothing to do when no roots are configured.
    """
    if not _allowed_roots:
        return
    if repo_path.strip() and inside_roots(repo_path, _allowed_roots):
        return
    raise ResourceError(REFUSAL.format(field="repo_path"))


def _require_index(config: IndexConfig) -> None:
    """Raise the tool error a read tool answers with when `config`'s repo has no index.

    Every read tool calls this BEFORE it builds a Retriever, GraphBuilder, AgentLoop or
    Database: opening a missing index creates one, and an empty index left behind makes a
    repository that was never indexed look indexed. A `ToolError` is this server's normal
    error result (`isError: true` with the message as text, the session carries on, and
    clients such as the VS Code extension already show or swallow it). The words are the
    CLI's: "No index found at <path>. Run trelix index <repo> first."
    """
    try:
        require_index(config)
    except IndexNotFoundError as exc:
        raise ToolError(str(exc)) from exc


def _indexed_config(repo_path: str) -> IndexConfig:
    """The IndexConfig of an indexed repository directory, or the tool error saying what is wrong.

    `check_repo_dir` answers a blank path, a path that does not exist and a file before
    `IndexConfig` sees the value. Its own validator raises a pydantic `ValidationError`, which
    FastMCP forwards as a JSON-RPC "Invalid request parameters" error instead of a tool result (a
    client such as the VS Code extension then sees an exception that names no argument), a blank
    path resolves to the server's working directory, so the no-index message named that directory,
    and a file was told to run `trelix index <file>`.
    """
    check_repo_dir(repo_path)
    config = IndexConfig(repo_path=repo_path)
    _require_index(config)
    return config


def _limits() -> Limits:
    """The output limits in force for this call (see budget.py), as a tool error if unusable."""
    try:
        return limits_from_env()
    except BudgetConfigError as exc:
        raise ToolError(str(exc)) from exc


def _get_retriever(repo_path: str) -> Retriever:
    """Return a cached Retriever for repo_path, constructing one if needed.

    The directory check comes before the cache lookup: a blank repo_path resolves to the server's
    working directory, which may well be a cached repo. The cache keeps the
    TRELIX_MCP_RETRIEVER_CACHE_SIZE most recently used entries (a hit counts as a use).
    """
    check_repo_dir(repo_path)
    key = str(Path(repo_path).resolve())
    with _retriever_cache_lock:
        cached = _retriever_cache.get(key)
        if cached is not None:
            _retriever_cache.move_to_end(key)
            return cached
        size = _limits().retriever_cache_size
        config = IndexConfig(repo_path=repo_path)
        _require_index(config)
        retriever = Retriever(config)
        _retriever_cache[key] = retriever
        # Evicted, not closed: a tool call in another worker thread may still hold the entry
        # (tools run in FastMCP's thread pool), and index_codebase already pops without closing.
        # Its SQLite connection closes with the last reference.
        while len(_retriever_cache) > size:
            _retriever_cache.popitem(last=False)
        return retriever


# Global subscription registry — records which subscription IDs were registered
# (via the subscribe_resource tool) for which trelix:// resource URIs.  It lives in this
# server process only; the `trelix watch` process has its own, empty one.
_subscription_registry = SubscriptionRegistry(
    max_subscribers=int(os.environ.get("TRELIX_MCP_MAX_SUBSCRIBERS", "1000")),
    ttl_seconds=float(os.environ.get("TRELIX_MCP_SUBSCRIPTION_TTL_SECONDS", "3600")),
)

# ---------------------------------------------------------------------------
# Resource subscription TOOLS. This server does not serve the MCP
# resources/subscribe request, so it does not advertise resources.subscribe (the
# capability is derived from the registered handlers, not forced).
# The tools below only register/deregister URIs in _subscription_registry.
# notify_file_changed() can push notifications/resources/updated for them, but only
# when it runs inside this stdio server process, and nothing here starts a watcher.
# ---------------------------------------------------------------------------


@mcp.tool()
def subscribe_resource(uri: str, subscription_id: str) -> dict[str, Any]:
    """Register a subscription for a trelix:// resource URI.

    Registers the URI in this server process's in-memory subscription registry. The
    server does not advertise resources.subscribe and does not serve resources/subscribe,
    so this tool is the only way to register. A notifications/resources/updated (URI
    only, no content) is sent for a registered URI only if a file watcher runs inside
    this server process; nothing starts one today, so none is delivered yet.

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
def unsubscribe_resource(subscription_id: str) -> dict[str, Any]:
    """Deregister a resource subscription by its subscription ID.

    Args:
        subscription_id: The ID returned when subscribe_resource was called.
    """
    uri = _subscription_registry.get_uri(subscription_id)
    _subscription_registry.unsubscribe(subscription_id)
    _log.info("Unsubscribed: subscription_id=%s uri=%s", subscription_id, uri)
    return {"unsubscribed": True, "subscription_id": subscription_id, "uri": uri}


# ---------------------------------------------------------------------------
# MCP Resources (application-controlled URI-addressable data)
# MCP spec: Resources are passive data; Tools are callable functions.
# trelix:// is a custom URI scheme — fully permitted by the MCP spec.
# ---------------------------------------------------------------------------


@mcp.resource("trelix://index/stats")
def resource_index_stats() -> str:
    """Aggregate statistics for the active trelix index.

    A parameterless resource cannot know which repository is meant, so this points at
    the repo-scoped template below rather than guessing. That template is new: this
    resource previously returned only this hint while `resources.get_index_stats` — a
    complete implementation with symbol/file/chunk counts and error handling — was
    reachable from nothing but its own tests.
    """
    return json.dumps(
        {
            "hint": "Use trelix://repo/{repo_path}/stats for index statistics, "
            "or trelix://repo/{repo_path}/manifest for the file listing"
        }
    )


@mcp.resource("trelix://repo/{repo_path}/stats")
def resource_repo_index_stats(repo_path: str) -> str:
    """Symbol, file and chunk counts for the index at `repo_path`.

    Serves `resources.get_index_stats`, which was fully written and wired to nothing.
    Cheap by design — three COUNT(*) queries, no embedder and no graph build — so an
    agent can check whether a repo is indexed at all before committing to a search.
    """
    _confine_resource_repo(repo_path)
    from trelix_mcp.resources import get_index_stats

    _log.info("resource_repo_index_stats repo_path=%r", repo_path)
    return get_index_stats(repo_path=repo_path)


@mcp.resource("trelix://repo/{repo_path}/manifest")
def resource_repo_manifest(repo_path: str) -> str:
    """List all indexed files in the repository at *repo_path*.

    Returns JSON with ``file_count`` and ``files[]`` list.
    Example URI: ``trelix://repo//Users/you/myrepo/manifest``
    """
    _confine_resource_repo(repo_path)
    from trelix_mcp.resources import get_repo_manifest

    return get_repo_manifest(repo_path)


@mcp.resource("trelix://repo/{repo_path}/symbols/{qualified_name}")
def resource_symbol_source(repo_path: str, qualified_name: str) -> str:
    """Get full source code of a symbol by its qualified name.

    Returns JSON with ``qualified_name``, ``kind``, ``signature``, ``body``.
    Example URI: ``trelix://repo//Users/you/myrepo/symbols/AuthService.login``
    """
    _confine_resource_repo(repo_path)
    from trelix_mcp.resources import get_symbol_source

    return get_symbol_source(repo_path, qualified_name)


# ---------------------------------------------------------------------------
# MCP Prompts (user-controlled reusable LLM interaction templates)
# ---------------------------------------------------------------------------


def _as_prompt_role(role: str) -> Literal["user", "assistant"]:
    """Narrow a builder's ``role`` string to the two roles the MCP spec allows.

    The builders are typed ``list[dict[str, str]]``, so their ``role`` is an
    arbitrary ``str`` to a type checker, while fastmcp's ``Message`` accepts only
    ``Literal["user", "assistant"]``. Branching rather than casting keeps the
    narrowing honest instead of asserted, and an unexpected role then fails here
    naming the offending value, rather than surfacing from inside fastmcp's own
    validation with no indication of which message was wrong.

    All three builders currently emit ``"user"``, so the raise is defensive.
    """
    if role == "user":
        return "user"
    if role == "assistant":
        return "assistant"
    raise ValueError(f"unsupported MCP prompt role {role!r}: expected 'user' or 'assistant'")


def _as_prompt_messages(messages: list[dict[str, str]]) -> list[Message]:
    """Convert the pure builders' message dicts into fastmcp ``Message`` objects.

    This conversion exists at the transport boundary on purpose. ``prompts.py``
    documents its builders as returning plain dicts and stays free of any fastmcp
    import, matching how the tool bodies are transport-agnostic pure functions —
    so the fastmcp-specific type is applied here, not there. Its existing tests
    call the builders directly and are unaffected.

    Without this, every ``prompts/get`` failed under fastmcp >= 3.4, which
    validates the return value and accepts only ``Message`` or ``str``:

        TypeError: messages[0] must be Message or str, got dict.

    ``prompts/list`` was unaffected, so all three prompts advertised themselves
    and then errored on use. The package pins ``fastmcp>=3.4.0``; the pin is now
    bounded below 4 so the next major cannot silently change this contract again.
    """
    return [Message(m["content"], role=_as_prompt_role(m.get("role", "user"))) for m in messages]


@mcp.prompt("trelix-search")
def prompt_search(query: str, repo_path: str) -> list[Message]:
    """Structured prompt for semantic code search using trelix.

    Args:
        query: Natural-language or keyword search query.
        repo_path: Absolute path to the repository root.
    """
    from trelix_mcp.prompts import build_search_prompt

    return _as_prompt_messages(build_search_prompt(query=query, repo_path=repo_path))


@mcp.prompt("trelix-explain")
def prompt_explain(qualified_name: str, repo_path: str) -> list[Message]:
    """Structured prompt for explaining a specific code symbol.

    Args:
        qualified_name: Fully-qualified symbol name, e.g. ``AuthService.login``.
        repo_path: Absolute path to the repository root.
    """
    from trelix_mcp.prompts import build_explain_prompt

    return _as_prompt_messages(
        build_explain_prompt(qualified_name=qualified_name, repo_path=repo_path)
    )


@mcp.prompt("trelix-blast-radius")
def prompt_blast_radius(symbol_name: str, repo_path: str) -> list[Message]:
    """Structured prompt for impact analysis before refactoring a symbol.

    Args:
        symbol_name: Name or qualified name of the symbol to analyse.
        repo_path: Absolute path to the repository root.
    """
    from trelix_mcp.prompts import build_blast_radius_prompt

    return _as_prompt_messages(
        build_blast_radius_prompt(symbol_name=symbol_name, repo_path=repo_path)
    )


@mcp.tool()
def search_code(
    query: str,
    repo_path: str,
    k: int = 10,
    cursor: int = 0,
    intent_hint: str | None = None,
    hyde_snippet_hint: str | None = None,
    detail: Detail = "detailed",
) -> dict[str, Any]:
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
    - Use cursor=0 for first page (default). A negative cursor is an error.
    - If next_cursor is not null, pass it as cursor for the next page.
    - k controls page size, from 1 up to TRELIX_MCP_MAX_K (default 50); page_size
      in the response is the k the server used.
    - The response text is capped at TRELIX_MCP_MAX_RESULT_CHARS (default
      15000). When results are dropped to fit, truncated is true, omitted counts
      them and next_cursor points at the first dropped result.

    🪶 Detail:
    - detail="detailed" (default) returns each result's body (first 800 chars).
    - detail="concise" drops body and returns a one-line signature instead.

    🧭 Intent hint (optional):
    - If the calling agent already knows the query's intent, pass it as
      intent_hint (one of: symbol_lookup, file_overview, feature_flow,
      project_overview, comparison, config_lookup, dependency_map,
      blast_radius) to skip trelix's own internal LLM classification and
      route directly. An unrecognized value is never rejected — it falls
      through to normal classification.
    - hyde_snippet_hint (a short hypothetical code snippet) is only used
      when intent_hint is also valid.

    Returns:
        {"results": [...], "next_cursor": int|null, "total_available": int,
         "page_size": int, "truncated": bool, "omitted": int}
    """
    limits = _limits()
    check_cursor(cursor)
    check_text(query, "query", "the text to search for")
    k = clamp_page_size(k, limits.max_k)
    _log.info("search_code query=%r repo=%s k=%d cursor=%d", query, repo_path, k, cursor)
    from trelix.retrieval.planner.models import plan_from_intent_hint

    plan = (
        plan_from_intent_hint(query, intent_hint, hyde_snippet_hint)
        if intent_hint is not None
        else None
    )
    ctx = _get_retriever(repo_path).retrieve(query, plan=plan)
    all_results = ctx.results

    rows = [
        {
            "file": r.file.rel_path,
            "symbol": r.symbol.qualified_name,
            "kind": r.symbol.kind.value,
            "lines": f"{r.symbol.line_start}-{r.symbol.line_end}",
            "score": round(r.score, 4),
            "source": r.source,
            **body_or_signature(r.symbol, detail, 800),
            "language": r.file.language.value,
        }
        for r in all_results[cursor : cursor + k]
    ]
    return fit_page(
        rows,
        cursor=cursor,
        page_size=k,
        total=len(all_results),
        max_chars=limits.max_result_chars,
    )


@mcp.tool()
def index_codebase(
    repo_path: str,
    provider: Literal["local", "openai", "azure", "voyage", "local-code"] | None = None,
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

    Defaults to TRELIX_EMBEDDER_PROVIDER (or "local" if that's unset too).

    Progress notifications are sent if the MCP client supports them.
    """
    _log.info("index_codebase repo=%s provider=%s", repo_path, provider)
    check_repo_dir(repo_path)

    # Omit the kwarg when unset so pydantic-settings falls through to
    # TRELIX_EMBEDDER_PROVIDER — passing provider="local" unconditionally
    # would silently override the env var on every call.
    embedder_config = EmbedderConfig() if provider is None else EmbedderConfig(provider=provider)
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

    # Drop any cached Retriever for this repo -- a re-index can switch
    # embedder providers (the `provider` argument above), which changes what
    # the next search_code/graph_search_mcp call needs to construct. The next
    # call rebuilds fresh and gets cached again from there.
    with _retriever_cache_lock:
        _retriever_cache.pop(str(Path(repo_path).resolve()), None)

    return stats


def _find_symbol(db: Database, qualified_name: str) -> tuple[Symbol, IndexedFile] | None:
    """The symbol named `qualified_name` and its file, or None when the index has no match."""
    rows = db.get_symbol_by_name(qualified_name)
    if not rows:
        # "Class.method" may be stored under another prefix: fall back to the last segment.
        rows = db.get_symbol_by_name(qualified_name.split(".")[-1])
    if not rows:
        return None
    return db.get_symbol_with_file(rows[0].id)  # type: ignore[arg-type]


@mcp.tool()
def get_symbol(
    qualified_name: str, repo_path: str, max_body_chars: int = DEFAULT_MAX_BODY_CHARS
) -> dict[str, Any] | None:
    """Look up a symbol by its fully-qualified name.

    Args:
        qualified_name: e.g. "MyClass.my_method" or "my_function".
        repo_path: Absolute path to the repository root.
        max_body_chars: Longest body to return, in characters (default 20000;
            0 means no limit). A longer body is cut and body_truncated is true.

    Returns:
        Symbol dict or None if not found.  Keys: name, qualified_name, kind,
        file, line_start, line_end, signature, docstring, body, language,
        body_truncated.
    """
    _log.info("get_symbol qualified_name=%r repo_path=%r", qualified_name, repo_path)
    check_body_limit(max_body_chars)
    check_text(
        qualified_name, "qualified_name", "the symbol's qualified name, e.g. MyClass.my_method"
    )
    config = _indexed_config(repo_path)
    db = Database(config.db_path_absolute)
    try:
        found = _find_symbol(db, qualified_name)
    finally:
        db.close()
    if found is None:
        return null_result()
    symbol, file = found
    body, body_truncated = truncate_body(symbol.body, max_body_chars)
    return {
        "name": symbol.name,
        "qualified_name": symbol.qualified_name,
        "kind": symbol.kind,
        "file": file.rel_path,
        "line_start": symbol.line_start,
        "line_end": symbol.line_end,
        "signature": symbol.signature,
        "docstring": symbol.docstring,
        "body": body,
        "language": file.language,
        "body_truncated": body_truncated,
    }


@mcp.tool()
def blast_radius(
    symbol_name: str, repo_path: str, limit: int = DEFAULT_BLAST_LIMIT
) -> list[dict[str, Any]]:
    """Find all symbols that depend on (call or import) a given symbol.

    Useful for impact analysis: "if I change X, what else might break?"

    Args:
        symbol_name: Name or qualified name of the symbol to analyse.
        repo_path: Absolute path to the repository root.
        limit: Most dependents to return, from 1 to 500 (default 100). The list is also
            capped at TRELIX_MCP_MAX_RESULT_CHARS characters (default 15000). When
            dependents are left out, the result carries a second text block saying so and
            `_meta.trelix` has total_available and omitted.

    Answered from the resolved call edges in the index, not by semantic search.

    This used to build `f"blast radius dependencies of {symbol_name}"` and run it
    through the Retriever, which never touched the `calls` table. Measured on trelix's
    own index for `AuditStore.append`: a SQL join over `calls` returns 320 caller
    symbols across 101 files in 128 ms, while the semantic version returned 12 entries
    in 5,747 ms — precision 0.33, **recall 0.04** — and ranked the queried symbol itself
    among the results. An agent asking "what breaks if I change this?" was being told
    about 4 of 101 affected files and acting on it.

    Two deliberate constraints:

    * The response stays a BARE ARRAY. workspace-vscode/src/mcp-client.ts does
      `(parsed ?? []).map(...)` and its caller swallows the TypeError, so wrapping this
      in an object envelope would make every "N dependents" CodeLens read 0 forever,
      silently. Counts and diagnostics belong in a separate tool.
    * Hydration goes through Database directly rather than Retriever, because
      `Retriever.__init__` eagerly constructs an embedder and reads its `.dimension`
      for the DimensionGuard. A graph query should not need an embedding model, and on
      a dimension mismatch it would not merely be slow — it would raise.

    Returns:
        Deduplicated list of dependent-symbol dicts with keys: file, symbol,
        kind, line_start, language. Empty when the symbol is unknown to the index.
    """
    max_chars = _limits().max_result_chars
    limit = max(1, min(limit, MAX_BLAST_LIMIT))
    _log.info("blast_radius symbol_name=%r repo_path=%r limit=%d", symbol_name, repo_path, limit)
    check_text(symbol_name, "symbol_name", "the name or qualified name of the symbol to analyse")
    config = _indexed_config(repo_path)
    db = Database(config.db_path_absolute)
    try:
        # A bare name can match several symbols (same method name in different
        # classes). Union their callers rather than guessing which was meant — for
        # impact analysis, over-reporting is the safer direction.
        targets = db.get_symbol_by_name(symbol_name)
        target_ids = {s.id for s in targets if s.id is not None}
        if not target_ids:
            return empty_bare_array()

        caller_ids: set[int] = set()
        for target_id in target_ids:
            caller_ids.update(db.get_callers(target_id))

        # Files importing the target's file are affected too, even with no direct call
        # edge. Resolved via file_id: `Retriever.get_importers` matches on
        # `files.rel_path`, so passing it a SYMBOL name returns nothing.
        importer_file_ids: set[int] = set()
        for symbol in targets:
            if symbol.file_id is not None:
                importer_file_ids.update(db.get_files_importing(symbol.file_id))

        for file_id in importer_file_ids:
            caller_ids.update(db.get_symbol_ids_for_file_id(file_id))

        # The symbol is not part of its own blast radius. Self-recursion and
        # same-name matches would otherwise put the queried symbol in the answer,
        # which is what the semantic version did as its top hit.
        caller_ids -= target_ids

        seen_files: set[str] = set()
        output: list[dict[str, Any]] = []
        for caller_id in sorted(caller_ids):
            pair = db.get_symbol_with_file(caller_id)
            if pair is None:
                continue
            symbol, file = pair
            if file.rel_path in seen_files:
                continue
            seen_files.add(file.rel_path)
            output.append(
                {
                    "file": file.rel_path,
                    "symbol": symbol.qualified_name,
                    "kind": symbol.kind.value
                    if hasattr(symbol.kind, "value")
                    else str(symbol.kind),
                    "line_start": symbol.line_start,
                    "language": file.language.value
                    if hasattr(file.language, "value")
                    else str(file.language),
                }
            )
        return bare_array_result(
            output[:limit],
            total_available=len(output),
            max_chars=max_chars,
            noun="dependents",
            remedy=lambda kept: blast_radius_remedy(kept, limit=limit, total=len(output)),
        )
    finally:
        db.close()


@mcp.tool()
def build_knowledge_graph(
    repo_path: str,
    extract_concepts: bool = False,
    min_community_size: int = 2,
    max_communities: int = 50,
) -> dict[str, Any]:
    """
    Build a knowledge graph for an indexed codebase.

    ⚠️ IMPORTANT: Run index_codebase first. repo_path must be absolute.

    🎯 What this builds:
    - Unified code property graph (calls + imports + type hierarchy)
    - Community detection: clusters modules into architectural groups
    - Optional LLM concept extraction. extract_concepts=True is PAID: up to 10 LLM
      calls over the 200 most central symbols (batches of 20), and it requires an LLM
      API key. It does NOT cover the repository — on trelix's own 12,184-symbol index
      those 200 symbols are 1.6% of it — so concept_count describes that sample.
      Selection is by PageRank centrality descending, symbol id ascending.

    ✨ Returns:
    - node_count: number of symbols in the graph
    - edge_count: number of structural relationships
    - community_count: detected architectural clusters
    - community_summary: the largest clusters, size-ordered (see the caps below)
    - concept_count: concepts found in that sample, 0 unless extract_concepts=True
    - concept_symbols_considered / concept_symbols_total: the coverage those concepts
      came from — 200 of 12,184 on this repo. Both are 0 when extract_concepts=False,
      which is what distinguishes "extraction never ran" from "it ran over 200 symbols
      and found nothing"; concept_count=0 alone cannot tell those apart.

    community_summary used to ship every detected community, unsorted. Measured on
    trelix's own index that was **1,160,415 bytes — roughly 290,000 tokens** from a
    single tool call, larger than most context windows. 6,437 of the 6,497 entries
    (99.1%) were singleton communities carrying no architectural signal, while the five
    real clusters (514, 477, 393, 279, 277 nodes) accounted for almost none of the bytes.

    min_community_size=2 and max_communities=50 cut that to under 32 KB while keeping
    every significant cluster. Pass min_community_size=1 and max_communities=0 to
    restore the old uncapped array.

    singleton_count and communities_omitted are always reported: 6,437 singletons is a
    community-detection tuning problem, and capping the payload without saying so would
    hide it.

    NOTE this does not reduce the endpoint's real COST. Both this tool and
    GET /graph/communities call GraphBuilder(config).build() on every request — a full
    Louvain pass, a PageRank rebuild and two metadata saves — and that is untouched here.
    """
    from trelix.graph.builder import GraphBuilder

    _log.info("build_knowledge_graph repo=%s concepts=%s", repo_path, extract_concepts)
    config = _indexed_config(repo_path)
    result = GraphBuilder(config).build(extract_concepts=extract_concepts)

    summary = list(result.community_summary or [])
    singleton_count = sum(1 for c in summary if int(c.get("size", 0)) <= 1)

    trimmed = [c for c in summary if int(c.get("size", 0)) >= max(1, min_community_size)]
    trimmed.sort(key=lambda c: int(c.get("size", 0)), reverse=True)
    if max_communities > 0:
        trimmed = trimmed[:max_communities]

    return {
        "node_count": result.node_count,
        "edge_count": result.edge_count,
        # The TRUE total, not the number returned — an agent deciding whether to look
        # closer needs to know how much it is not seeing.
        "community_count": result.community_count,
        "concept_count": result.concept_count,
        # concept_count alone reads as a property of the REPOSITORY when it is really a
        # property of a capped sample: extraction stops at the 200 most central symbols,
        # which on this repo's 12,184-symbol index is 1.6%. v3.1.5 gave that coverage to
        # humans in `trelix graph --concepts` and to `--json`; an agent calling this tool
        # still got the bare count, i.e. exactly the misreport that release fixed.
        # Reported unconditionally, unlike the CLI's `if concepts:` — 0/0 alongside
        # concept_count=0 is what tells an agent extraction never ran, and an added key
        # is compatible under the Stable "MCP tool signatures" row of
        # docs/BACKWARDS_COMPATIBILITY.md, which forbids removals, not additions.
        "concept_symbols_considered": result.concept_symbols_considered,
        "concept_symbols_total": result.concept_symbols_total,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "community_summary": trimmed,
        "singleton_count": singleton_count,
        "communities_omitted": len(summary) - len(trimmed),
    }


@mcp.tool()
def graph_search_mcp(
    query: str, repo_path: str, k: int = 10, detail: Detail = "detailed"
) -> list[dict[str, Any]]:
    """
    Graph-traversal search: find structurally related symbols by starting
    from semantically similar seeds and following code relationships.

    ⚠️ IMPORTANT: Run index_codebase and optionally build_knowledge_graph first.

    🎯 When to use:
    - "What other code is connected to X?" — follow call/import/type edges
    - "Find the blast radius of a class" — who calls or imports it?
    - "What lives in the same architectural cluster as X?"

    📏 Size:
    - k is the most results, from 1 up to TRELIX_MCP_MAX_K (default 50).
    - detail="detailed" (default) returns each body (first 600 chars); detail="concise"
      returns a one-line signature instead.
    - The list is capped at TRELIX_MCP_MAX_RESULT_CHARS characters (default 15000). When
      results are left out, the result carries a second text block saying so and
      `_meta.trelix` has total_available and omitted.
    """
    from trelix.graph.builder import GraphBuilder
    from trelix.graph.search import graph_search

    limits = _limits()
    k = clamp_page_size(k, limits.max_k)
    check_text(query, "query", "the text to search for")
    _log.info("graph_search_mcp query=%r repo=%s k=%d", query, repo_path, k)

    # First find seed symbols via standard retrieval. _get_retriever refuses a repo_path that is
    # not a directory or has no index, so GraphBuilder below never sees one.
    retriever = _get_retriever(repo_path)
    ctx = retriever.retrieve(query)
    seed_ids = [r.chunk.symbol_id for r in ctx.results[:5]]

    if not seed_ids:
        return empty_bare_array()

    # Then expand via graph, on the retriever's own (checked) config — reuse the DB already
    # opened by GraphBuilder/CodeGraph
    build_result = GraphBuilder(retriever.config).build(extract_concepts=False)
    db = build_result.code_graph._db
    graph_results = graph_search(db, build_result.code_graph, seed_ids, depth=2, max_results=k)

    rows = [
        {
            "file": r.file.rel_path,
            "symbol": r.symbol.qualified_name,
            "kind": r.symbol.kind.value,
            "score": round(r.score, 4),
            "source": r.source,
            **body_or_signature(r.symbol, detail, 600),
        }
        for r in graph_results[:k]
    ]
    return bare_array_result(
        rows,
        total_available=len(rows),
        max_chars=limits.max_result_chars,
        noun="results",
        remedy=(
            f"The list is bounded by {MAX_RESULT_CHARS_ENV}, not by k; "
            'use detail="concise" for shorter results, or raise it, to see more.'
        ),
    )


# Fixed, cursor-independent per-repo fan-out width for federation_search_all.
# Wide enough to cover any realistic single-page request without letting the
# per-repo candidate pool (and therefore the RRF fusion input) change shape
# as `cursor` grows — see federation_search_all's docstring.
_FEDERATION_SEARCH_ALL_FETCH_WIDTH = 100


def _outside_roots_key(outside: int) -> dict[str, int]:
    """`repos_outside_roots` for every federation_search_all response while roots are configured.

    Absent when nothing is confined, so the unconfined responses keep the exact keys they had.
    """
    return {"repos_outside_roots": outside} if _allowed_roots else {}


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
    A blank config_path is a ToolError like every other blank argument:
    Path("").resolve() is the process cwd, and the refusal would otherwise
    name that directory and the allowed roots, none of which the caller passed.
    """
    if config_path is None:
        return None
    check_text(
        config_path,
        "config_path",
        "a path inside ~/.config/trelix or <cwd>/.trelix, or omit it for the default registry",
    )
    resolved = Path(config_path).resolve()
    allowed_roots = [_DEFAULT_CONFIG.parent, Path.cwd() / ".trelix"]
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
        raise ConfigPathNotAllowedError(
            f"config_path must resolve inside one of {[str(r) for r in allowed_roots]}, "
            f"got {resolved}"
        )
    return str(resolved)


@mcp.tool()
def federation_list_repos(config_path: str | None = None) -> dict[str, Any]:
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
) -> dict[str, Any]:
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
    check_text(alias, "alias", "a short unique name for the repo, e.g. auth-service")
    check_absolute_repo_dir(path, "path")
    check_positive_weight(weight)
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
def federation_remove_repo(alias: str, config_path: str | None = None) -> dict[str, Any]:
    """Unregister a repo from federated search by alias.

    Args:
        alias: The alias to remove. No-op (removed=False) if not registered.
        config_path: Optional path to a custom repos.json. Must resolve
            inside ~/.config/trelix/ or <cwd>/.trelix/.

    Returns:
        {"removed": bool, "alias": str, "error": str|None}
    """
    _log.info("federation_remove_repo alias=%r", alias)
    check_text(alias, "alias", "the alias to unregister (see federation_list_repos)")
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
    detail: Detail = "detailed",
) -> dict[str, Any]:
    """Search across ALL registered repos simultaneously (federated search).

    ⚠️ IMPORTANT:
    - Requires repos to already be registered via federation_add_repo AND
      already indexed (run index_codebase on each repo path beforehand). A
      registered repo with no index is skipped, not searched, and not counted
      in repos_searched; repos_unindexed names its alias, and if none of the
      queried repos is indexed, error says so.
    - Results are merged via Reciprocal Rank Fusion weighted by each repo's
      registered weight, then deduplicated.
    - Only the first TRELIX_FEDERATION_MAX_REPOS registered repos (default
      50) are actually queried; repos_skipped reports how many were
      omitted.
    - When the server runs with --root or TRELIX_ALLOWED_REPO_ROOTS, a
      registered repo outside those roots is not searched, and every
      response carries repos_outside_roots (how many were left out).

    🎯 When to Use:
    - Cross-service / cross-repo questions ("where is auth handled across
      our microservices?")
    - You don't know which of several registered repos contains the answer.

    📄 Pagination: same cursor/k contract as search_code — pages are sliced
    from one fixed-width fetch, independent of cursor, so page contents are
    stable across calls (results don't shift/duplicate/vanish between
    pages the way a cursor-scaled fetch width would cause). k is clamped to
    1..TRELIX_MCP_MAX_K, a negative cursor is an error, and the text is capped
    at TRELIX_MCP_MAX_RESULT_CHARS (see search_code).

    🪶 Detail: detail="concise" drops each body and returns a one-line signature.

    Returns:
        {"results": [...], "next_cursor": int|None, "total_available": int,
         "page_size": int, "truncated": bool, "omitted": int,
         "repos_searched": int, "repos_skipped": int,
         "repos_unindexed": [alias, ...], "error": str|None}
        The error and empty-registry responses keep their shorter shape (no
        page_size, truncated, omitted or repos_unindexed). With --root or
        TRELIX_ALLOWED_REPO_ROOTS set, every response also has
        "repos_outside_roots": int.
    """
    limits = _limits()
    check_cursor(cursor)
    check_text(query, "query", "the text to search for")
    k = clamp_page_size(k, limits.max_k)
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
            **_outside_roots_key(0),
        }
    registry = RepoRegistry.load(confined_path)
    entries, outside = entries_inside_roots(registry.list(), _allowed_roots)
    if outside:
        # Search only what the roots allow: a registry holding the kept entries, at the same path.
        registry = RepoRegistry(confined_path or str(_DEFAULT_CONFIG), entries)
    if not entries:
        return {
            "results": [],
            "next_cursor": None,
            "total_available": 0,
            "repos_searched": 0,
            "repos_skipped": 0,
            "error": None,
            **_outside_roots_key(outside),
        }

    max_repos = RetrievalConfig().federation_max_repos
    fed = FederatedRetriever(registry, max_repos=max_repos)
    repos_queried = fed.repos_queried_count(len(entries))
    repos_skipped = len(entries) - repos_queried

    # A registered repo with no index is not searched (searching it would create the empty
    # index that makes it look indexed), so it is not counted in repos_searched. When that
    # leaves nothing to search, say why instead of returning an empty page that reads as
    # "no matches".
    unindexed = fed.unindexed_repos()
    repos_searched = repos_queried - len(unindexed)
    if unindexed and repos_searched == 0:
        return {
            "results": [],
            "next_cursor": None,
            "total_available": 0,
            "repos_searched": 0,
            "repos_skipped": repos_skipped,
            "error": " ".join(str(missing) for _, missing in unindexed),
            **_outside_roots_key(outside),
        }

    all_results = fed.retrieve(query, k=_FEDERATION_SEARCH_ALL_FETCH_WIDTH)

    rows = [
        {
            "repo": r.source.split(":")[0] if ":" in r.source else "",
            "file": r.file.rel_path,
            "symbol": r.symbol.qualified_name,
            "kind": r.symbol.kind.value,
            "score": round(r.score, 4),
            "source": r.source,
            **body_or_signature(r.symbol, detail, 800),
            "language": r.file.language.value,
        }
        for r in all_results[cursor : cursor + k]
    ]
    return fit_page(
        rows,
        cursor=cursor,
        page_size=k,
        total=len(all_results),
        max_chars=limits.max_result_chars,
        extra={
            "repos_searched": repos_searched,
            "repos_skipped": repos_skipped,
            "repos_unindexed": [entry.alias for entry, _ in unindexed],
            "error": None,
            **_outside_roots_key(outside),
        },
    )


def _extract_elicit_answer(ctx: Context | None) -> str | None:
    """Read the client's answer to a prior ask_agent clarify request, if any.

    None on the initial round (nothing asked yet), or if the client declined
    or cancelled — both mean "no usable answer" and are handled distinctly
    by the caller (a decline must not re-trigger the same clarify question).

    The "clarification" key can only resolve to an ElicitResult in practice —
    ask_agent is the only thing that ever mints that key's InputRequest, and
    it always mints an ElicitRequest — but InputResponses' declared type is a
    union across every SEP-2322 request kind, so this narrows explicitly
    rather than assuming the attribute exists.
    """
    if ctx is None or ctx.input_responses is None:
        return None
    elicit_result = ctx.input_responses.get("clarification")
    if not isinstance(elicit_result, ElicitResult):
        return None
    if elicit_result.action != "accept" or not elicit_result.content:
        return None
    answer = elicit_result.content.get("answer")
    return answer if isinstance(answer, str) else None


@mcp.tool()
def ask_agent(
    query: str,
    repo_path: str,
    session_id: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any] | InputRequiredToolResult:
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

    Clarifying questions (SEP-2322, v3.3.0+):
    - When the agent's question is genuinely ambiguous, this tool returns an
      `InputRequiredToolResult` instead of the usual dict — an MCP client
      that supports SEP-2322 elicitation surfaces it as a form and resends
      this SAME call (same query/repo_path/session_id) with the answer
      attached as `inputResponses`; do not pass the answer as a new `query`.
    - A client that declines/cancels gets a normal dict answer back
      explaining the question could not be resolved — this tool never
      re-asks the same clarify question on a decline.

    Returns:
        {"answer": str, "session_id": str, "turn_count": int} normally, or
        an InputRequiredToolResult when the agent needs a clarifying answer
        before it can continue.
    """
    check_text(query, "query", "the question to answer")
    if session_id is not None:
        check_session_id(
            session_id, "the session_id a previous answer returned, or omit it for a new session"
        )
    request_state = ctx.request_state if ctx is not None else None
    resolved_session_id = request_state or session_id
    client_answer = _extract_elicit_answer(ctx)

    if ctx is not None and ctx.input_responses is not None and client_answer is None:
        # The client responded, but declined or cancelled — there is no
        # answer to resume with.
        return {
            "answer": "Cannot continue without an answer to the clarifying question.",
            "session_id": resolved_session_id or "",
            "turn_count": 0,
        }

    effective_query = client_answer or query
    _log.info(
        "ask_agent query=%r repo=%s session_id=%r", effective_query, repo_path, resolved_session_id
    )
    config = _indexed_config(repo_path)
    config.retrieval.agentic_enabled = True
    loop = AgentLoop(config)
    result = loop.run(effective_query, session_id=resolved_session_id)

    if result.needs_input:
        return InputRequiredToolResult(
            InputRequiredResult(
                input_requests={
                    "clarification": ElicitRequest(
                        params=ElicitRequestFormParams(
                            message=result.content,
                            requested_schema={
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                        "description": (
                                            "Your answer to trelix's clarifying question."
                                        ),
                                    }
                                },
                                "required": ["answer"],
                            },
                        )
                    )
                },
                request_state=result.session_id,
            )
        )

    db = Database(config.db_path_absolute)
    try:
        turns = db.get_agent_turns(result.session_id)
    finally:
        db.close()

    return {"answer": result.content, "session_id": result.session_id, "turn_count": len(turns)}


@mcp.tool()
def agent_list_sessions(repo_path: str, limit: int = 50) -> dict[str, Any]:
    """List recent agent sessions for a repo, most recently active first.

    Args:
        repo_path: Absolute path to the repository root.
        limit: Max sessions to return, from 1 up to TRELIX_MCP_MAX_K (default 50).
            The text is also capped at TRELIX_MCP_MAX_RESULT_CHARS characters; when
            the oldest sessions are left out, truncated is true and omitted counts them.

    Returns:
        {"sessions": [{"session_id", "created_at", "last_active_at", "query",
         "turn_count"}, ...], "count": int, "page_size": int, "truncated": bool,
         "omitted": int}. query is the session's most recent prompt, cut to its first 300
        characters, and then the session also has query_truncated: true.
    """
    limits = _limits()
    limit = clamp_page_size(limit, limits.max_k)
    _log.info("agent_list_sessions repo=%s limit=%d", repo_path, limit)
    config = _indexed_config(repo_path)
    db = Database(config.db_path_absolute)
    try:
        max_age = config.retrieval.agent_session_max_age_seconds
        if max_age > 0:
            db.evict_stale_agent_sessions(max_age)
        sessions = [cap_session_query(row) for row in db.list_agent_sessions(limit=limit)]
    finally:
        db.close()

    def build(kept: list[dict[str, Any]], omitted: int) -> dict[str, Any]:
        return {
            "sessions": kept,
            "count": len(kept),
            "page_size": limit,
            "truncated": omitted > 0,
            "omitted": omitted,
        }

    return fit_rows(sessions, build, limits.max_result_chars)


@mcp.tool()
def agent_clear_session(repo_path: str, session_id: str) -> dict[str, Any]:
    """Delete a persisted agent session and all its turn history.

    Args:
        repo_path: Absolute path to the repository root.
        session_id: The session to delete.

    Returns:
        {"cleared": bool, "session_id": str}
    """
    _log.info("agent_clear_session repo=%s session_id=%r", repo_path, session_id)
    check_session_id(session_id, "the session_id to delete (see agent_list_sessions)")
    config = _indexed_config(repo_path)
    db = Database(config.db_path_absolute)
    try:
        existed = db.delete_agent_session(session_id)
    finally:
        db.close()
    return {"cleared": existed, "session_id": session_id}


def main() -> None:
    """Entry point for the trelix-mcp server (stdio transport).

    Parses argv for --help/--version/--tools/--root and rejects unknown flags — the normal path
    (no args, launched by an MCP client's server config) falls straight through to running
    the server with every tool and no confinement, unchanged from before this parser existed.
    """
    parser = argparse.ArgumentParser(
        prog="trelix-mcp",
        description="MCP server for trelix — semantic code search over stdio.",
    )
    parser.add_argument("--version", action="version", version=f"trelix-mcp {__version__}")
    parser.add_argument(
        "--tools",
        choices=TOOL_PROFILES,
        default="full",
        help="tool profile: 'full' (default) lists every tool, 'core' lists only the everyday "
        "search and indexing tools and hides the rest",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        metavar="PATH",
        help="a repository root every repo_path, federation path and trelix://repo/... URI must "
        "lie inside; repeatable, and TRELIX_ALLOWED_REPO_ROOTS (os.pathsep-separated) adds more. "
        "With neither, nothing is confined",
    )
    args = parser.parse_args()
    if any(not root.strip() for root in args.root):
        parser.error("--root must not be blank")
    try:
        limits_from_env()
    except BudgetConfigError as exc:
        parser.error(str(exc))
    apply_tool_profile(mcp, args.tools)
    _install_confinement(resolve_allowed_roots(*args.root))

    def _handle_sigterm(signum: int, frame: Any) -> None:
        _log.info("Received SIGTERM — shutting down")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    _log.info("trelix-mcp starting (transport=stdio)")
    mcp.run(transport="stdio")
