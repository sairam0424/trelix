# Trelix MCP Server Guide

Complete guide for using trelix as an MCP (Model Context Protocol) server in Claude Code, Cursor, Windsurf, and Continue.dev.

---

## 1. What is MCP?

Model Context Protocol (MCP) is an open standard that lets AI assistants connect to external tools and data sources through a unified interface. MCP servers expose tools, resources, and prompts that the AI can invoke directly during a conversation. Trelix implements MCP so that any compatible IDE or agent can query your codebase with hybrid search, symbol lookup, and graph analysis without writing any integration code.

---

## 2. Install trelix-mcp

```bash
pip install trelix-mcp==2.8.1
```

Verify the binary is on your PATH:

```bash
trelix-mcp --version
# trelix-mcp 2.8.1
```

> **Note:** Python 3.10+ is required. Use a virtual environment if you manage multiple projects.

---

## 3. Setup in Claude Code

Register trelix as a persistent MCP server with one command:

```bash
claude mcp add trelix -- trelix-mcp
```

Confirm it registered correctly:

```bash
claude mcp list
# trelix   trelix-mcp   (stdio)
```

The server starts automatically whenever Claude Code launches a session. No further configuration is needed.

---

## 4. Setup in Cursor

Edit (or create) `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "trelix": {
      "command": "trelix-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

Restart Cursor after saving. Trelix tools will appear in the MCP tool palette under **trelix**.

---

## 5. Setup in Windsurf

Edit (or create) `~/.windsurf/mcp_settings.json`:

```json
{
  "servers": [
    {
      "name": "trelix",
      "transport": "stdio",
      "command": "trelix-mcp",
      "args": []
    }
  ]
}
```

Restart Windsurf. The trelix tools appear in the agent sidebar under **Tools > trelix**.

---

## 6. Setup in Continue.dev

Edit `~/.continue/config.json` and add trelix to the `mcpServers` array:

```json
{
  "mcpServers": [
    {
      "name": "trelix",
      "command": "trelix-mcp",
      "args": []
    }
  ]
}
```

Reload Continue.dev (Cmd+Shift+P → **Continue: Reload**). The tools are available in any chat session.

---

## 7. The 15 MCP Tools

Trelix-mcp exposes 15 MCP tools organized into four functional groups:

1. **Core search & indexing** (4 tools): `search_code`, `index_codebase`, `get_symbol`, `blast_radius`
2. **Graph analysis** (2 tools): `build_knowledge_graph`, `graph_search_mcp`
3. **Resource subscriptions** (2 tools): `subscribe_resource`, `unsubscribe_resource`
4. **Multi-repo federation** (4 tools): `federation_list_repos`, `federation_add_repo`, `federation_remove_repo`, `federation_search_all`
5. **Persistent agent sessions** (3 tools): `ask_agent`, `agent_list_sessions`, `agent_clear_session`

### Core Search & Indexing

### `search_code`

```
search_code(query, repo_path, k=10, cursor=0) → {results, next_cursor, total_available}
```

**What it does:** Runs trelix hybrid search (dense + sparse) over an indexed codebase. Returns ranked code snippets with file path, line range, symbol context, and relevance score.

**When to use:**
- Finding all usages of an API or pattern
- Locating where a concept is implemented
- Exploring unfamiliar codebases before making changes

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Natural-language or keyword query |
| `repo_path` | str | required | Absolute path to the indexed repository |
| `k` | int | 10 | Number of results per page |
| `cursor` | int | 0 | Pagination offset (see Section 11) |

**Response shape:**
```json
{
  "results": [
    {
      "file": "src/auth/service.py",
      "start_line": 42,
      "end_line": 67,
      "symbol": "AuthService.login",
      "score": 0.91,
      "snippet": "def login(self, username: str, password: str) -> Token:\n    ..."
    }
  ],
  "next_cursor": 10,
  "total_available": 47
}
```

**Example queries:**
```
"JWT token validation middleware"
"database connection pool exhaustion"
"retry logic with exponential backoff"
```

**Pagination example:**
```python
cursor = 0
while cursor is not None:
    page = search_code("authentication handler", "/path/to/repo", k=10, cursor=cursor)
    process(page["results"])
    cursor = page["next_cursor"] if page["next_cursor"] < page["total_available"] else None
```

---

### `index_codebase`

```
index_codebase(repo_path, provider="local") → stats dict
```

**What it does:** Parses, embeds, and indexes all source files in the repository. This must be run before any search or symbol tool can work. The server emits progress notifications as it processes files so you can track long indexing jobs.

**When to use:** Run once after cloning a new repo, and re-run after large commits or branch switches.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_path` | str | required | Absolute path to the repository root |
| `provider` | str | `"local"` | Embedding provider: `"local"` (sentence-transformers) or `"openai"` |

**Response shape:**
```json
{
  "files_indexed": 312,
  "symbols_extracted": 1847,
  "chunks_stored": 4203,
  "elapsed_seconds": 18.4,
  "index_version": "2.8.1"
}
```

> **Tip:** Large repos (10 000+ files) can take a few minutes. The MCP client will receive streaming progress events — watch your IDE's MCP output panel.

---

### `get_symbol`

```
get_symbol(qualified_name, repo_path) → symbol dict
```

**What it does:** Returns the full source, docstring, file location, and metadata for a specific symbol identified by its qualified name.

**When to use:** Inspecting a specific function, class, or method before modifying it.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `qualified_name` | str | required | Dot-separated symbol path |
| `repo_path` | str | required | Absolute path to the indexed repository |

**Response shape:**
```json
{
  "qualified_name": "AuthService.login",
  "kind": "method",
  "file": "src/auth/service.py",
  "start_line": 42,
  "end_line": 67,
  "docstring": "Authenticate a user and return a signed JWT.",
  "source": "def login(self, username: str, password: str) -> Token:\n    ..."
}
```

**Example:**
```
get_symbol("AuthService.login", "/path/to/repo")
get_symbol("config.settings.DatabaseConfig", "/path/to/repo")
get_symbol("utils.retry.exponential_backoff", "/path/to/repo")
```

---

### `blast_radius`

```
blast_radius(symbol_name, repo_path) → list of dependent files
```

**What it does:** Traverses the call graph and import graph to find every file that transitively depends on the given symbol. Helps you understand the full impact of a change before you make it.

**When to use:** Always run this before refactoring a function, renaming a class, or changing a public API signature.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol_name` | str | required | Qualified name of the symbol to analyze |
| `repo_path` | str | required | Absolute path to the indexed repository |

**Response shape:**
```json
[
  {
    "file": "src/api/routes/users.py",
    "dependency_type": "direct_call",
    "depth": 1
  },
  {
    "file": "tests/integration/test_auth.py",
    "dependency_type": "transitive_import",
    "depth": 2
  }
]
```

**Workflow pattern:**
```
1. blast_radius("PaymentService.charge", "/repo")   ← know what breaks
2. get_symbol("PaymentService.charge", "/repo")     ← read the current code
3. Make the change
4. blast_radius again to confirm scope hasn't grown
```

---

### `subscribe_resource`

```
subscribe_resource(uri, subscription_id) → {status}
```

**What it does:** Registers a subscription for a `trelix://` resource URI. Once subscribed, the MCP client receives `notifications/resources/updated` whenever trelix detects a file change that affects that resource (e.g. a manifest change after a re-index). Requires the client to support MCP resource subscriptions (`resources.subscribe=True`).

**When to use:** Use in MCP clients (Claude Code, Cursor) that support resource subscriptions when you want the AI to be notified automatically whenever the index changes, without polling.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `uri` | str | required | The `trelix://` resource URI to subscribe to, e.g. `trelix://repo//path/to/repo/manifest` |
| `subscription_id` | str | required | A client-chosen identifier used to correlate `notifications/resources/updated` payloads and to cancel the subscription |

**Response shape:**
```json
{
  "status": "subscribed",
  "uri": "trelix://repo//Users/you/projects/myapp/manifest",
  "subscription_id": "my-sub-001"
}
```

**New in v2.5.0.**

---

### `unsubscribe_resource`

```
unsubscribe_resource(subscription_id) → {status}
```

**What it does:** Removes a previously registered resource subscription by its `subscription_id`. No further `notifications/resources/updated` messages will be sent for the associated URI. Safe to call even if the subscription_id is unknown (returns `status: "not_found"` without error).

**When to use:** Call when the client no longer needs live updates for a resource, or when tearing down a session.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subscription_id` | str | required | The subscription identifier returned by (or passed to) `subscribe_resource` |

**Response shape:**
```json
{
  "status": "unsubscribed",
  "subscription_id": "my-sub-001"
}
```

**New in v2.5.0.**

---

### `build_knowledge_graph`

```
build_knowledge_graph(repo_path) → graph stats
```

**What it does:** Constructs a NetworkX-based directed graph combining call relationships, import dependencies, and type hierarchies across the entire codebase. The graph is cached on disk and used by `graph_search_mcp`.

**When to use:** Run once after indexing, or after significant structural changes. Required before `graph_search_mcp` will return graph-aware results.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_path` | str | required | Absolute path to the indexed repository |

**Response shape:**
```json
{
  "nodes": 1847,
  "edges": 5312,
  "connected_components": 3,
  "max_depth": 12,
  "build_seconds": 4.2
}
```

---

### `graph_search_mcp`

```
graph_search_mcp(query, repo_path, k=10) → list of results
```

**What it does:** Combines the knowledge graph topology with semantic search to surface results that are structurally central — symbols that many other symbols depend on, or that are highly connected in the call graph.

**When to use:** When you want to find the most architecturally significant code related to a concept, not just the textually closest matches.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Natural-language or keyword query |
| `repo_path` | str | required | Absolute path to the indexed repository |
| `k` | int | 10 | Number of results to return |

**Response shape:**
```json
[
  {
    "symbol": "DatabasePool.acquire",
    "file": "src/db/pool.py",
    "graph_centrality": 0.87,
    "semantic_score": 0.79,
    "combined_score": 0.83,
    "in_degree": 24,
    "out_degree": 3
  }
]
```

---

### Multi-Repo Federation

#### `federation_list_repos`

```
federation_list_repos(config_path=None) → {repos, count, error}
```

**What it does:** Lists all repos registered for federated (multi-repo) search.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | str\|None | None | Optional path to a custom repos.json. Must resolve inside `~/.config/trelix/` or `<cwd>/.trelix/`. Defaults to `~/.config/trelix/repos.json`. |

**Response shape:**
```json
{
  "repos": [
    {"alias": "auth-service", "path": "/Users/you/auth", "weight": 1.0},
    {"alias": "payment-api", "path": "/Users/you/payment", "weight": 0.8}
  ],
  "count": 2,
  "error": null
}
```

**New in v2.8.0.**

---

#### `federation_add_repo`

```
federation_add_repo(alias, path, weight=1.0, config_path=None) → {added, alias, path, error}
```

**What it does:** Registers a repo for federated search across MCP tool calls.

**Important:**
- `path` must be an **ABSOLUTE** path
- Run `index_codebase` on the repo separately — registering does not index
- The registry is capped at `TRELIX_FEDERATION_MAX_REPOS` entries (default 50) to prevent unbounded growth

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alias` | str | required | Short unique name for the repo (e.g. "auth-service") |
| `path` | str | required | Absolute path to the repo root |
| `weight` | float | 1.0 | RRF weight multiplier — higher values rank this repo's results higher in `federation_search_all` |
| `config_path` | str\|None | None | Optional path to a custom repos.json |

**Response shape:**
```json
{
  "added": true,
  "alias": "auth-service",
  "path": "/Users/you/auth",
  "error": null
}
```

**Workflow pattern:**
```
1. federation_add_repo("auth", "/path/to/auth")
2. index_codebase("/path/to/auth")               ← index it
3. federation_search_all("JWT validation")       ← now searchable
```

**New in v2.8.0.**

---

#### `federation_remove_repo`

```
federation_remove_repo(alias, config_path=None) → {removed, alias, error}
```

**What it does:** Unregisters a repo from federated search by alias. No-op if the alias is not registered (returns `removed: false`).

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alias` | str | required | The alias to remove |
| `config_path` | str\|None | None | Optional path to a custom repos.json |

**Response shape:**
```json
{
  "removed": true,
  "alias": "auth-service",
  "error": null
}
```

**New in v2.8.0.**

---

#### `federation_search_all`

```
federation_search_all(query, k=10, cursor=0, config_path=None) → {results, next_cursor, total_available, repos_searched, repos_skipped, error}
```

**What it does:** Searches across ALL registered repos simultaneously using Reciprocal Rank Fusion to merge results, weighted by each repo's registered `weight`.

**Important:**
- Requires repos to already be registered via `federation_add_repo` AND already indexed
- Results are deduplicated by `(file_path, symbol_id)`
- Only the first `TRELIX_FEDERATION_MAX_REPOS` registered repos (default 50) are actually queried — `repos_skipped` reports the omitted count
- Pagination uses a stable fixed-width fetch (100 results per repo) sliced by `cursor`/`k`, so page contents don't shift between calls

**When to use:**
- Cross-service / cross-repo questions ("where is auth handled across our microservices?")
- You don't know which of several registered repos contains the answer

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Natural-language or keyword query |
| `k` | int | 10 | Number of results per page |
| `cursor` | int | 0 | Pagination offset |
| `config_path` | str\|None | None | Optional path to a custom repos.json |

**Response shape:**
```json
{
  "results": [
    {
      "repo": "auth-service",
      "file": "src/jwt/validator.py",
      "symbol": "JWTValidator.verify",
      "kind": "method",
      "score": 0.91,
      "source": "auth-service:jwt_verification",
      "body": "def verify(self, token: str) -> Claims:\n    ...",
      "language": "python"
    }
  ],
  "next_cursor": 10,
  "total_available": 47,
  "repos_searched": 2,
  "repos_skipped": 0,
  "error": null
}
```

**New in v2.8.0.**

---

### Persistent Agent Sessions

#### `ask_agent`

```
ask_agent(query, repo_path, session_id=None) → {answer, session_id, turn_count}
```

**What it does:** Asks a question using the multi-turn ReAct agentic loop with persistent memory. The agent can iteratively retrieve, grep, and inspect symbols to answer complex questions.

**Important:**
- `repo_path` must be an **ABSOLUTE** path to an already-indexed repository
- Session history is scoped to `(repo_path, session_id)` — a session created against one repo is invisible when querying a different repo
- Requires LLM configuration (e.g. `OPENAI_API_KEY`) — always uses the agentic loop, unlike `search_code` which is retrieval-only

**When to use:**
- Multi-step questions needing iterative retrieve/grep/get_symbol drilling
- Follow-up questions in the same conversation — pass back the `session_id` to preserve context

**Session lifecycle:**
- Omit `session_id` on the first call — a new UUID4 is generated and returned
- Pass that `session_id` on subsequent calls to resume with full turn history
- Sessions auto-evict after `TRELIX_RETRIEVAL_AGENT_SESSION_MAX_AGE_SECONDS` of inactivity (default 7 days)
- Use `agent_clear_session` to delete one explicitly

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Natural-language question |
| `repo_path` | str | required | Absolute path to the indexed repository |
| `session_id` | str\|None | None | Session ID to resume (omit for new session) |

**Response shape:**
```json
{
  "answer": "The JWT validation is handled by the JWTValidator.verify method in src/jwt/validator.py. It checks signature, expiry, and issuer claims.",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "turn_count": 3
}
```

**Example conversation:**
```python
# First question
r1 = ask_agent("Where is JWT validation implemented?", "/path/to/repo")
# r1["session_id"] = "550e8400-..."

# Follow-up in the same session
r2 = ask_agent(
    "What are the dependencies of that validator?",
    "/path/to/repo",
    session_id=r1["session_id"]
)
# Agent remembers the JWT validator from turn 1
```

**New in v2.8.0.**

---

#### `agent_list_sessions`

```
agent_list_sessions(repo_path, limit=50) → {sessions, count}
```

**What it does:** Lists recent agent sessions for a repo, most recently active first. Runs stale-session eviction first if `TRELIX_RETRIEVAL_AGENT_SESSION_MAX_AGE_SECONDS > 0`.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_path` | str | required | Absolute path to the repository root |
| `limit` | int | 50 | Max sessions to return |

**Response shape:**
```json
{
  "sessions": [
    {
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "created_at": "2026-07-15T10:30:00",
      "last_active_at": "2026-07-15T10:45:00",
      "query": "Where is JWT validation implemented?",
      "turn_count": 3
    }
  ],
  "count": 1
}
```

**New in v2.8.0.**

---

#### `agent_clear_session`

```
agent_clear_session(repo_path, session_id) → {cleared, session_id}
```

**What it does:** Deletes a persisted agent session and all its turn history (cascade delete).

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_path` | str | required | Absolute path to the repository root |
| `session_id` | str | required | The session to delete |

**Response shape:**
```json
{
  "cleared": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**New in v2.8.0.**

---

## 8. Federation Security & Configuration (v2.8.1)

### Path Confinement for `config_path`

All four federation MCP tools (`federation_list_repos`, `federation_add_repo`, `federation_remove_repo`, `federation_search_all`) accept an optional `config_path` parameter to override the default registry location. In v2.8.1, this parameter is **confined** to two allowlisted roots:

1. `~/.config/trelix/` (the default federation config directory)
2. `<mcp-server-cwd>/.trelix/` (a repo-local override when the MCP server process is launched from within a repo)

Any `config_path` that resolves outside both roots will be rejected with a `ConfigPathNotAllowedError` returned as `{"error": str}` in the tool response. This prevents an MCP client (or a prompt-injected agent) from pointing registry I/O at an arbitrary filesystem path.

**Why this matters:** Before v2.8.1, a caller-supplied `config_path` was passed straight into file I/O operations with no validation. This fix uses `Path.is_relative_to()` (not a naive string prefix check) to ensure the resolved path lives under an allowlisted root.

### Registry Capacity Cap

The federation registry is capped at `TRELIX_FEDERATION_MAX_REPOS` entries (default **50**, configurable via environment variable). When `federation_add_repo` is called and the registry is at capacity, it returns:

```json
{
  "added": false,
  "alias": "...",
  "path": "...",
  "error": "Registry is at capacity (50 repos) — remove a repo before adding another"
}
```

This prevents a runaway or adversarial `federation_add_repo` loop from making every subsequent `federation_search_all` call scale linearly with an unbounded repo count.

Additionally, `federation_search_all` only actually queries the **first N repos** in registry order (where N = `min(registered_count, TRELIX_FEDERATION_MAX_REPOS)`). The response includes:

- `repos_searched` — how many repos were actually queried
- `repos_skipped` — how many registered repos were omitted due to the cap

**Environment variable:**
```bash
export TRELIX_FEDERATION_MAX_REPOS=100  # raise the cap to 100
```

**New in v2.8.1.**

---

## 9. The 3 MCP Resources

MCP resources are read-only data endpoints that the AI can fetch without executing a tool. Use them to give the model static context about the indexed codebase.

### `trelix://index/stats`

Returns aggregate statistics for all indexed repositories managed by the running trelix-mcp server.

```json
{
  "repos_indexed": 2,
  "total_files": 654,
  "total_symbols": 3891,
  "total_chunks": 8702,
  "server_version": "2.8.1"
}
```

### `trelix://repo/{repo_path}/manifest`

Returns the full list of indexed files for a specific repository, including file size, language, and symbol count.

```
trelix://repo//Users/you/projects/myapp/manifest
```

```json
{
  "repo_path": "/Users/you/projects/myapp",
  "files": [
    {"path": "src/auth/service.py", "language": "python", "symbols": 12, "size_bytes": 4210},
    {"path": "src/db/pool.py", "language": "python", "symbols": 8, "size_bytes": 2108}
  ]
}
```

### `trelix://repo/{repo_path}/symbols/{qualified_name}`

Returns the raw source of a symbol without invoking the `get_symbol` tool. Useful for embedding static symbol definitions in prompts.

```
trelix://repo//Users/you/projects/myapp/symbols/AuthService.login
```

---

## 10. The 3 MCP Prompts

MCP prompts are pre-built instruction templates that the client can inject into a conversation. They configure the model to perform a specific trelix-powered workflow.

### `trelix-search`

Prompts the model to search the codebase using `search_code` and summarize the most relevant results with file references. Pass the search query as the prompt argument.

**Usage in Claude Code:** `/trelix-search "error handling in the payment module"`

### `trelix-explain`

Prompts the model to retrieve a symbol with `get_symbol`, fetch its blast radius, and produce a structured explanation of what the symbol does and what depends on it.

**Usage in Claude Code:** `/trelix-explain AuthService.login`

### `trelix-blast-radius`

Prompts the model to run `blast_radius`, group the results by dependency depth, and generate a risk-ordered refactoring plan.

**Usage in Claude Code:** `/trelix-blast-radius PaymentService.charge`

---

## 11. Watch Bridge (v2.7.0)

The `trelix watch` command now fires `notifications/resources/updated` events to all subscribed MCP clients after every file re-index. This enables real-time codebase awareness in Claude Code and other agents without polling.

**How it works:**
1. Run `trelix watch` in your project directory
2. trelix-mcp listens for file system changes
3. After re-indexing completes, MCP clients receive a notification via `notifications/resources/updated`
4. The client can refresh cached code context or trigger workflows based on changed files

This is useful for:
- Keeping codebase context fresh during active development
- Triggering automated analysis pipelines when code changes
- Multi-agent coordination where file changes need propagation

---

## 12. v2.4.0 Breaking Change — `search_code` Pagination

In v2.3.x and earlier, `search_code` accepted an `offset` integer parameter and returned a flat list:

```python
# BEFORE (v2.3.x) — flat list, offset parameter
results = search_code(
    query="authentication handler",
    repo_path="/path/to/repo",
    k=10,
    offset=20          # old parameter name
)
# → [{"file": ..., "snippet": ...}, ...]
```

In v2.4.0, `offset` was renamed to `cursor` and the return type changed to a paginated envelope:

```python
# AFTER (v2.4.0) — paginated envelope, cursor parameter
response = search_code(
    query="authentication handler",
    repo_path="/path/to/repo",
    k=10,
    cursor=20          # new parameter name
)
# → {"results": [...], "next_cursor": 30, "total_available": 47}
results = response["results"]
```

**Migration checklist:**
- Rename `offset=` to `cursor=` at all call sites
- Update result extraction from `response` to `response["results"]`
- Use `response["next_cursor"]` and `response["total_available"]` for pagination logic
- If `next_cursor == total_available`, you have reached the last page

---

## 13. Pagination Example (Full Paging Loop)

```python
def fetch_all_results(query: str, repo_path: str, page_size: int = 10) -> list:
    """Retrieve every result for a query by paging through all results."""
    all_results = []
    cursor = 0

    while True:
        response = search_code(
            query=query,
            repo_path=repo_path,
            k=page_size,
            cursor=cursor,
        )

        batch = response["results"]
        all_results.extend(batch)

        next_cursor = response["next_cursor"]
        total = response["total_available"]

        # Stop when we have consumed all available results
        if next_cursor >= total or not batch:
            break

        cursor = next_cursor

    return all_results
```

---

## 14. IDE Integrations

### VS Code Extension

The `workspace-vscode/` extension provides two command shortcuts for rapid trelix access:

- **`trelix.search`** — Search the workspace codebase with trelix hybrid search
- **`trelix.ask`** — Ask a natural-language question about the code

Install from the `workspace-vscode/` directory:

```bash
cd workspace-vscode && npm install && code --install-extension .
```

Then use in the command palette (Cmd+Shift+P):
- `Trelix: Search` — Opens search input, runs hybrid query
- `Trelix: Ask` — Opens question input, streams conversational response

---

## 15. Example Claude Code Session

The following shows three realistic prompts you might use once trelix-mcp is registered.

**Prompt 1 — Index and orient yourself**

```
Index /Users/me/projects/myapi with trelix, then tell me which files have
the most symbols and what the top-level architecture looks like.
```

Claude will call `index_codebase`, then fetch `trelix://repo/.../manifest` and summarize the module structure.

**Prompt 2 — Find an implementation and understand its impact**

```
I need to change how sessions expire. Use trelix to find all session-related
code, then show me the blast radius of SessionManager.refresh before I touch it.
```

Claude will call `search_code("session expiry")`, then `get_symbol("SessionManager.refresh", ...)`, then `blast_radius("SessionManager.refresh", ...)` and produce a risk-annotated summary.

**Prompt 3 — Graph-aware refactoring plan**

```
Build the knowledge graph for /Users/me/projects/myapi, then use graph search
to find the most connected database layer symbols. I want to replace the ORM
with raw SQL and need to know the full blast radius.
```

Claude will call `build_knowledge_graph`, then `graph_search_mcp("database ORM query layer", ...)`, then `blast_radius` on each high-centrality symbol and output a dependency-ordered migration plan.

---

## 16. Resource Subscriptions (v2.5.0)

trelix-mcp v2.5.0 implements the MCP resource subscription protocol
([MCP spec §Resources](https://modelcontextprotocol.io/specification/2024-11-05/server/resources)).

### How it works

1. trelix-mcp advertises `resources.subscribe=True` in server capabilities
2. MCP clients (Claude Code, Cursor) can subscribe to a resource URI
3. When trelix watch detects a file change, it fires `notifications/resources/updated`
4. The client then calls `resources/read` to fetch the updated index content

### Subscription tools

**`subscribe_resource(uri, subscription_id)`**
Register a subscription for a trelix:// resource URI.

```
uri:             trelix://repo//path/to/repo/manifest
subscription_id: any string — used to correlate notifications
```

**`unsubscribe_resource(subscription_id)`**
Remove a subscription by its ID.

### Wire protocol

```
Client → Server:  resources/subscribe  { uri }
Server → Client:  notifications/resources/updated  { uri, _meta: { subscriptionId } }
Client → Server:  resources/read  { uri }
```

---

## 17. Troubleshooting MCP Issues

### `trelix-mcp: command not found`

The binary is not on your PATH. Fix:

```bash
# Check where pip installed it
python -m site --user-base
# e.g. /Users/you/Library/Python/3.11

# Add to PATH in ~/.zshrc or ~/.bashrc
export PATH="$HOME/Library/Python/3.11/bin:$PATH"
source ~/.zshrc

# Verify
trelix-mcp --version
```

### Claude Code does not list the trelix server

```bash
claude mcp list          # check if it appears
claude mcp remove trelix # remove stale entry
claude mcp add trelix -- trelix-mcp   # re-add
```

Restart Claude Code after re-registering.

### `index_codebase` fails or returns 0 files

- Confirm `repo_path` is an **absolute** path (not `~/...` — expand the tilde).
- Trelix skips files matched by `.gitignore` and `.trelixignore`. Check those files if expected sources are missing.
- Large repos may hit default memory limits. Set `TRELIX_MAX_WORKERS=2` to lower parallelism:

```bash
TRELIX_MAX_WORKERS=2 trelix-mcp
```

### `search_code` returns empty results

The codebase must be indexed first. Run `index_codebase` and check that it returned `files_indexed > 0` before querying.

### `graph_search_mcp` returns no results or errors

The knowledge graph must be built separately from the index. Call `build_knowledge_graph(repo_path)` after indexing.

### MCP server crashes silently in Cursor / Windsurf

Enable debug logging by setting the environment variable in your MCP config:

```json
{
  "mcpServers": {
    "trelix": {
      "command": "trelix-mcp",
      "args": ["--log-level", "debug"],
      "env": {
        "TRELIX_LOG_FILE": "/tmp/trelix-mcp.log"
      }
    }
  }
}
```

Then inspect `/tmp/trelix-mcp.log` for the error.

### Version mismatch between trelix-mcp and a pinned index

If you downgrade trelix-mcp after indexing, the cached index may be incompatible. Delete the index cache and re-index:

```bash
rm -rf ~/.trelix/cache/<repo_hash>
```

The cache location is printed by `trelix-mcp --cache-dir`.

### Pagination returns duplicate results

Duplicates indicate that the index changed between pages (a background re-index ran). Re-run the full query from `cursor=0` to get a consistent snapshot.
