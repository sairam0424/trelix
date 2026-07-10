# Trelix MCP Server Guide

Complete guide for using trelix as an MCP (Model Context Protocol) server in Claude Code, Cursor, Windsurf, and Continue.dev.

---

## 1. What is MCP?

Model Context Protocol (MCP) is an open standard that lets AI assistants connect to external tools and data sources through a unified interface. MCP servers expose tools, resources, and prompts that the AI can invoke directly during a conversation. Trelix implements MCP so that any compatible IDE or agent can query your codebase with hybrid search, symbol lookup, and graph analysis without writing any integration code.

---

## 2. Install trelix-mcp

```bash
pip install trelix-mcp==2.7.0
```

Verify the binary is on your PATH:

```bash
trelix-mcp --version
# trelix-mcp 2.7.0
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

## 7. The 6 MCP Tools

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
  "index_version": "2.7.0"
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

## 8. The 3 MCP Resources

MCP resources are read-only data endpoints that the AI can fetch without executing a tool. Use them to give the model static context about the indexed codebase.

### `trelix://index/stats`

Returns aggregate statistics for all indexed repositories managed by the running trelix-mcp server.

```json
{
  "repos_indexed": 2,
  "total_files": 654,
  "total_symbols": 3891,
  "total_chunks": 8702,
  "server_version": "2.7.0"
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

## 9. The 3 MCP Prompts

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

## 10. Watch Bridge (v2.7.0)

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

## 11. v2.4.0 Breaking Change — `search_code` Pagination

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

## 12. Pagination Example (Full Paging Loop)

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

## 13. IDE Integrations

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

## 14. Example Claude Code Session

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

## 15. Troubleshooting MCP Issues

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
