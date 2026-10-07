# Trelix MCP Server Guide

Complete guide for using trelix as an MCP (Model Context Protocol) server in Claude Code, Cursor, Windsurf, Continue.dev, and JetBrains IDEs.

---

## 1. What is MCP?

Model Context Protocol (MCP) is an open standard that lets AI assistants connect to external tools and data sources through a unified interface. MCP servers expose tools, resources, and prompts that the AI can invoke directly during a conversation. Trelix implements MCP so that any compatible IDE or agent can query your codebase with hybrid search, symbol lookup, and graph analysis without writing any integration code.

---

## 2. Install trelix-mcp

```bash
pip install trelix-mcp
```

Verify the binary is on your PATH:

```bash
which trelix-mcp
```

`trelix-mcp` accepts only `--help`, `--version` and `--tools core|full` (section 8) —
running it with no arguments starts the stdio MCP server. `trelix-mcp --version` prints the
installed version, and the package exposes it too:

```bash
python -c "import trelix_mcp; print(trelix_mcp.__version__)"
# 3.3.7
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

## 7. Setup in JetBrains IDEs (2025.2+)

JetBrains IDEs (IntelliJ IDEA, PyCharm, WebStorm, GoLand, RustRover, etc.), version 2025.2 and later, ship a first-party built-in MCP client — no plugin install required. Configure it via **Settings → Tools → AI Assistant → Model Context Protocol (MCP)** and add a new server:

```json
{
  "mcpServers": {
    "trelix": {
      "command": "trelix-mcp",
      "args": []
    }
  }
}
```

Or, via the IDE's MCP settings UI: set **Command** to `trelix-mcp` and leave **Arguments** empty.

Restart the IDE after saving. Trelix's tools then appear alongside JetBrains' built-in AI Assistant tools in any chat session.

> **Note:** This is JetBrains' first-party built-in MCP client, not a trelix-published JetBrains plugin — there is currently no dedicated trelix plugin for the JetBrains Marketplace.

---

## 8. The 15 MCP Tools

Trelix-mcp exposes 15 MCP tools organized into four functional groups:

1. **Core search & indexing** (4 tools): `search_code`, `index_codebase`, `get_symbol`, `blast_radius`
2. **Graph analysis** (2 tools): `build_knowledge_graph`, `graph_search_mcp`
3. **Resource subscriptions** (2 tools): `subscribe_resource`, `unsubscribe_resource`
4. **Multi-repo federation** (4 tools): `federation_list_repos`, `federation_add_repo`, `federation_remove_repo`, `federation_search_all`
5. **Persistent agent sessions** (3 tools): `ask_agent`, `agent_list_sessions`, `agent_clear_session`

### What `tools/list` tells a client

Every tool carries the four MCP annotation hints, taken from one table in
`packages/trelix-mcp/src/trelix_mcp/tool_metadata.py`. `openWorldHint` is `false` for all of
them: the tools act on a local index and registry, and the network calls they can make go to a
closed list of destinations. `ask_agent`'s LLM calls go to the provider the operator configured.
`index_codebase`'s embedding calls go to the provider its `provider` argument names (`local`,
`openai`, `azure`, `voyage` or `local-code`), so the model picks among those; the hosted ones
need credentials the operator has set.

| Tool | `readOnlyHint` | `destructiveHint` | `idempotentHint` |
|------|:--:|:--:|:--:|
| `index_codebase` | false | false | true |
| `search_code` | **true** | false | false |
| `get_symbol` | **true** | false | false |
| `blast_radius` | **true** | false | false |
| `build_knowledge_graph` | false | false | false |
| `graph_search_mcp` | false | false | false |
| `ask_agent` | false | false | false |
| `agent_list_sessions` | false | false | false |
| `agent_clear_session` | false | **true** | false |
| `federation_list_repos` | false | false | false |
| `federation_add_repo` | false | false | false |
| `federation_remove_repo` | false | **true** | false |
| `federation_search_all` | false | false | false |
| `subscribe_resource` | false | false | false |
| `unsubscribe_resource` | false | false | false |

- `readOnlyHint` is `true` only for the three tools whose calls are measured to leave
  `.trelix/index.db` (and its `-wal` and `-shm` files) byte-identical:
  `packages/trelix-mcp/tests/test_tool_readonly.py` indexes a real temporary repository, runs each
  tool and hashes the files before and after, with every database connection closed at both
  points (a running server keeps one open, and while one is open SQLite leaves an empty `-wal`
  file and a `-shm` file beside the database). That holds for an index already at the current
  schema with telemetry off (the default). It is a claim about the index database, not about
  every file. Three things write, and the same test file records each: with
  `TRELIX_TELEMETRY_ENABLED=true` each `search_code` adds a row to `query_telemetry`; the first
  open of an index written by an older trelix migrates that index; and every `search_code`
  writes a small JSON trace of the query to `.trelix/debug/` (one new file per call, the same
  trace `docs/OBSERVABILITY.md` describes; the directory has no ignore file of its own, it is
  covered by `.trelix/.gitignore` one level up, which ignores everything under `.trelix/`), which
  leaves the database as it was. `get_symbol` and `blast_radius` write no file. A client that
  reads `readOnlyHint` as "does not touch the repository directory" is therefore wrong for
  `search_code`.
- `build_knowledge_graph` and `graph_search_mcp` are not read-only because both rebuild and save
  the graph metadata. `agent_list_sessions` evicts sessions past the configured maximum age before
  it lists.
- `federation_list_repos` and `federation_search_all` only read, but that test does not cover
  them yet, so they are left as `readOnlyHint: false`.

`tools/list` returns the tools in a fixed order, the same on every run: the indexing and search
workflow first (`index_codebase`, `search_code`, `get_symbol`, `blast_radius`,
`build_knowledge_graph`, `graph_search_mcp`), then the agent-session tools, the federation
tools, and last the two subscription tools, which deliver nothing today (section 17).

The server also sends `instructions` (under 2,000 characters) when a client connects: what the
tools are for and the order to use them in (index first, then `search_code`, then `get_symbol`
or `blast_radius` for a symbol you know). They say that the subscription tools send no
notifications. With `--tools core` the instructions leave out the tools that profile hides.

On the 2026-07-28 protocol, list results carry `ttlMs: 300000` and `cacheScope: "private"`, so a
client that opts in to caching may reuse the tool list for five minutes. FastMCP has one
server-wide setting for this hint, so it is sent on the other list results and on
`resources/read` as well: a caching client can show `trelix://` resource content up to five
minutes old.

#### `--tools core|full`

```bash
claude mcp add trelix -- trelix-mcp --tools core
```

`full` is the default and lists all 15 tools. `core` lists only `index_codebase`,
`search_code`, `get_symbol`, `blast_radius`, `build_knowledge_graph`, `graph_search_mcp` and
`ask_agent`, in that order; the other eight are hidden, not removed, and a call to one is
answered as an unknown tool. (`repo_map` and `exact_search` do not exist in this server, so
`core` does not list them.) Any other value is a usage error: exit code 2 and no server.

### Output size and limits

A tool result costs a client's context twice: FastMCP sends a dict result as a text block and again as `structuredContent`, and the text block holds the JSON as a string, so every quote in it is escaped: the wire is a little more than double the text. Measured through an in-process `fastmcp.Client` with 100,000-character bodies, before these limits: `search_code` with `k=10` was about 9,700 characters of text and 20,000 on the wire, and with `k=100` about 97,000 and 197,000 (over Claude Code's 25,000-token cap for one tool result). `graph_search_mcp` had no upper bound on `k`, and `blast_radius` costs about 127 characters per dependent file with short paths (more with long ones) and had no bound either.

| Setting | Default | What it does |
|---------|---------|--------------|
| `TRELIX_MCP_MAX_K` | `50` | `k` (and `limit` on `agent_list_sessions`) is clamped to 1..this value, and `page_size` in the response says what was used. Blank means the default. A value that is not an integer of at least 1 stops `trelix-mcp` at start-up with exit code 2 |
| `TRELIX_MCP_MAX_RESULT_CHARS` | `15000` | The budget for the text of a list result (`search_code`, `federation_search_all`, `graph_search_mcp`, `blast_radius`, `agent_list_sessions`); the tail is dropped to fit it. The budget counts both copies a client is sent (the text block, with each quote escaped, and `structuredContent`), so the whole response is at most twice the budget (30,000 characters by default) and its text under the budget. That makes it a limit on what is sent and not on the text alone: at the default the text can be about 14,800 characters when it holds no quotes or backslashes and about 10,000 when it is mostly quotes. `0` turns the cut off. Blank means the default; a negative or non-integer value stops the server at start-up |
| `TRELIX_MCP_RETRIEVER_CACHE_SIZE` | `8` | Most Retrievers the server keeps across calls, one per repository (`search_code` and `graph_search_mcp` reuse them, and each may hold an embedding model with the `local` provider). Past the bound the least recently used one is dropped, not closed, and a later call for that repository builds it again. Blank means the default. A value that is not an integer of at least 1 stops `trelix-mcp` at start-up with exit code 2 and makes a tool call return an error |
| `detail="concise"` | `"detailed"` | On `search_code`, `graph_search_mcp` and `federation_search_all`: drops each result's `body` and adds a one-line `signature` (its first line, at most 200 characters; the first line of the body when the signature is blank) |

What a client sees when results are left out:

- **Cursor-paged results** (`search_code`, `federation_search_all`): `truncated` is `true`, `omitted` counts the dropped results, and `next_cursor` points at the first dropped result, so the next page repeats nothing and skips nothing. The text block is the response, so the keys `results`, `next_cursor` and `total_available` are where they always were.
- **Bare arrays** (`blast_radius`, `graph_search_mcp`): the first text block is still the JSON array, a second text block says what was left out, and `_meta.trelix` is `{"total_available": M, "omitted": K}`. A result that fits is unchanged: one text block, no `_meta.trelix`. `graph_search_mcp` has no cursor, and a larger `k` cannot help once the character budget is what cut it, so its note points at `detail="concise"`. `blast_radius`'s note ends with what can still be raised, decided from what cut the list so that a raise it names always shows more: `TRELIX_MCP_MAX_RESULT_CHARS` when the budget cut it (and `limit` next, when `limit` is under 500 and would cut too), `limit` when the limit alone cut it, or, at `limit=500` when the limit alone cut it, that nothing can be raised and the remaining dependents cannot be fetched with this tool.
- `agent_list_sessions`: the oldest sessions are left out and `truncated` and `omitted` say so (it has no cursor). A session's `query` (its most recent prompt, which has no bound of its own) is cut to its first 300 characters and that session gets `query_truncated: true`, so one long prompt cannot push a response past the ceilings.
- `blast_radius` has no offset: `limit=500` returns at most the first 500 dependents whatever the budget, and the rest cannot be fetched; its note says so once the limit is what cut the list.
- One result is always kept, even under a tiny budget, so paging always advances; a single result longer than the budget is returned whole.
- A negative `cursor` is an error result (`isError: true`) saying to use 0 or the previous `next_cursor`.
- A cut result is sent within the 30,000-character ceiling however many quotes it holds (a test uses bodies full of quotes, 72-character paths and 200-character queries, at default arguments and at the maximum, and another a 100,000-character most recent prompt). An empty bare array (`blast_radius` for a symbol with no dependents or one the index does not know, `graph_search_mcp` with no hits) is sent as a `[]` text block with `structuredContent` of `{"result": []}`, and `get_symbol` for an unknown symbol as a `null` text block with `{"result": null}`. FastMCP alone sends neither text block, only the structured content, so a client that reads only the first text block had nothing to parse (the VS Code extension defaults to `null` and `[]` for that case); every result now has a text block that parses to its structured content.

`build_knowledge_graph` and `federation_list_repos` are not cut: the first already caps its own community list (`min_community_size`, `max_communities`) and the second lists a registry that `TRELIX_FEDERATION_MAX_REPOS` caps when repos are added.

### Errors

An invalid input is answered with a tool error: `isError: true`, one text block holding the message, no `structuredContent`, and the session carries on (the VS Code extension shows the text; a model reads it and corrects the call). The message names the argument, shows what was given (cut to 200 characters) and says what would be valid. It is never a Python traceback, and the only path in it is the one the caller passed. The two subscription tools, `subscribe_resource` and `unsubscribe_resource`, accept any string and are outside this table.

| Input | Text |
|-------|------|
| `repo_path` (or `federation_add_repo`'s `path`) empty or only whitespace | `repo_path must not be empty or whitespace (got '  '); pass the absolute path of the repository root.` |
| `repo_path` that does not exist | `repo_path does not exist: '/x/y'; pass the absolute path of the repository root.` |
| `repo_path` that is a file | `repo_path is not a directory: '/x/y/a.py'; pass the repository root, not a file in it.` |
| `repo_path` with no index (every tool but `index_codebase`, which creates it) | `No index found at /x/y/.trelix/index.db. Run trelix index /x/y first.` (unchanged) |
| `query` (`search_code`, `graph_search_mcp`, `federation_search_all`, `ask_agent`), `qualified_name`, `symbol_name`, `alias` (`federation_add_repo`, `federation_remove_repo`) empty or only whitespace | `query must not be empty or whitespace (got ''); pass the text to search for.`, and the same shape for the others |
| `session_id` given but blank (`ask_agent`, `agent_clear_session`) | `session_id must not be empty or whitespace (got ''); pass the session_id a previous answer returned, or omit it for a new session.` (`ask_agent`); `session_id must not be empty or whitespace (got ''); pass the session_id to delete (see agent_list_sessions).` (`agent_clear_session`) |
| `config_path` given but blank (`federation_list_repos`, `federation_add_repo`, `federation_remove_repo`, `federation_search_all`) | `config_path must not be empty or whitespace (got ''); pass a path inside ~/.config/trelix or <cwd>/.trelix, or omit it for the default registry.` |
| `federation_add_repo` `path` not absolute | `path must be an absolute path (got 'services/auth'); pass the absolute path of the repository root.` |
| `federation_add_repo` `weight` 0 or less, or not finite | `weight must be a positive number (got -1.5); 1.0 is the default, and a higher value ranks that repo's results higher.` |
| negative `cursor`, negative `max_body_chars` | `cursor must be 0 or greater, got -1. Use 0 for the first page, then pass the next_cursor value from the previous response.`, `max_body_chars must be 0 (no limit) or greater, got -1. Use a positive number to cut the body.` (unchanged) |
| an argument of the wrong type (`k="abc"`, `cursor=1.5`) or outside its choices (`detail="verbose"`, an unknown `provider`) | rejected by FastMCP before the tool runs, also as `isError: true`; the text is pydantic's and names the argument and the valid values (`Input should be a valid integer`, `Input should be 'concise' or 'detailed'`) |

Not errors, on purpose: a relative `repo_path` is not rejected but resolved against the server's working directory, which the caller does not control, so pass an absolute path (`federation_add_repo`'s `path` must be absolute because a registry entry outlives the working directory); `k` and `limit` outside 1..`TRELIX_MCP_MAX_K` are clamped and the envelope tools (`search_code`, `federation_search_all`, `agent_list_sessions`) report the value used as `page_size`, while `blast_radius`'s `limit` is clamped to 1..500 and the clamped value shows only in its truncation note; an unknown `session_id` is not an error either (`agent_clear_session` answers `cleared: false`; `ask_agent` starts a session under that id); `federation_remove_repo` of an alias that is not registered is a no-op (`removed: false`); `get_symbol` of an unknown symbol is `null` and `blast_radius` of one is `[]`; and the federation tools keep their `error` key, not `isError`, for a duplicate alias, a full registry, a non-blank `config_path` outside the allowed roots (section 9) and a registry in which no queried repo is indexed (a client contract; moving those to `isError` is an open owner decision).

Before these checks, a `repo_path` that did not exist (or held only whitespace) was not a tool result at all but a JSON-RPC "Invalid request parameters" error: `IndexConfig` rejects the path with a pydantic `ValidationError`, which FastMCP forwards as a protocol error instead of masking it into a result, so the VS Code extension saw an exception naming no argument. A blank `repo_path` meant the server's working directory: the no-index message named that directory and `index_codebase` indexed it. A file was told to run `trelix index <file>`. A blank `query` was searched (and `ask_agent` persisted a session for it), and `federation_add_repo` registered a blank alias, a blank, relative, missing or file path, and a weight of 0 or less. A blank `config_path` on the four federation tools resolved to the server's working directory (`Path("").resolve()`) and was refused with a 200 `error` dict that named the allowed roots and that directory, none of which the caller had passed.

### Core Search & Indexing

### `search_code`

```
search_code(query, repo_path, k=10, cursor=0, intent_hint=None, hyde_snippet_hint=None, detail="detailed") → {results, next_cursor, total_available, page_size, truncated, omitted}
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
| `k` | int | 10 | Results per page, clamped to 1..`TRELIX_MCP_MAX_K` (default 50); `page_size` in the response is the value used |
| `cursor` | int | 0 | Pagination offset (see [Output size and limits](#output-size-and-limits)); a negative value is an error |
| `intent_hint` | str \| None | None | One of the 8 `IntentType` values — see below |
| `hyde_snippet_hint` | str \| None | None | Short hypothetical code snippet (HyDE); only used when `intent_hint` is also valid |
| `detail` | `"detailed"` \| `"concise"` | `"detailed"` | `concise` drops `body` from each result and adds a one-line `signature` |

**Caller-supplied intent routing:** If the calling agent has already classified the query's intent, pass `intent_hint` — one of `symbol_lookup`, `file_overview`, `feature_flow`, `project_overview`, `comparison`, `config_lookup`, `dependency_map`, or `blast_radius` — to skip trelix's own internal LLM intent classification and route directly to that intent's retrieval strategy. This is useful when an orchestrating agent already knows, from its own reasoning, what kind of question it's asking, and wants to avoid the extra classification round-trip. An unrecognized or invalid `intent_hint` value is never rejected: the call silently falls through to trelix's normal internal classification, as if `intent_hint` had not been passed. `hyde_snippet_hint` is only honored when `intent_hint` is also valid — pass a short snippet of what the target code might look like to steer HyDE-style query expansion toward that shape.

**Response shape:**
```json
{
  "results": [
    {
      "file": "src/auth/service.py",
      "symbol": "AuthService.login",
      "kind": "method",
      "lines": "42-67",
      "score": 0.91,
      "source": "vector+bm25",
      "body": "def login(self, username: str, password: str) -> Token:\n    ...",
      "language": "python"
    }
  ],
  "next_cursor": 10,
  "total_available": 47,
  "page_size": 10,
  "truncated": false,
  "omitted": 0
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
    cursor = page["next_cursor"]  # null after the last page
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
  "index_version": "3.3.7"
}
```

> **Tip:** Large repos (10 000+ files) can take a few minutes. The MCP client will receive streaming progress events — watch your IDE's MCP output panel.

---

### `get_symbol`

```
get_symbol(qualified_name, repo_path, max_body_chars=20000) → symbol dict
```

**What it does:** Returns the full source, docstring, file location, and metadata for a specific symbol identified by its qualified name.

**When to use:** Inspecting a specific function, class, or method before modifying it.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `qualified_name` | str | required | Dot-separated symbol path |
| `repo_path` | str | required | Absolute path to the indexed repository |
| `max_body_chars` | int | 20000 | Longest `body` to return; `0` means no limit, a negative value is an error |

**Response shape:**
```json
{
  "name": "login",
  "qualified_name": "AuthService.login",
  "kind": "method",
  "file": "src/auth/service.py",
  "line_start": 42,
  "line_end": 67,
  "signature": "def login(self, username: str, password: str) -> Token:",
  "docstring": "Authenticate a user and return a signed JWT.",
  "body": "def login(self, username: str, password: str) -> Token:\n    ...",
  "language": "python",
  "body_truncated": false
}
```

A body longer than `max_body_chars` is cut and `body_truncated` is `true`. The result is `null` when no symbol matches.

**Example:**
```
get_symbol("AuthService.login", "/path/to/repo")
get_symbol("config.settings.DatabaseConfig", "/path/to/repo")
get_symbol("utils.retry.exponential_backoff", "/path/to/repo")
```

---

### `blast_radius`

```
blast_radius(symbol_name, repo_path, limit=100) → list of dependent symbols, one per file
```

**What it does:** Queries the resolved call edges and import edges in the index for everything that **directly** depends on the given symbol — its callers, plus every file importing the module that defines it. One hop, not a transitive closure: on this repository a single hop from `AuditStore.append` is already 104 files, and a transitive walk reaches most of the codebase, which is not an actionable answer.

Answered from SQLite, so it needs no embedding model and costs 56-117 ms. Before v3.1.2 it ran a semantic search for the phrase "blast radius dependencies of X" and never read the call graph at all — measured against a SQL oracle on this repo's index, that returned **4% of the affected files** in 5.7 s, and ranked the queried symbol itself among the results.

**When to use:** Always run this before refactoring a function, renaming a class, or changing a public API signature.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol_name` | str | required | Qualified name of the symbol to analyze |
| `repo_path` | str | required | Absolute path to the indexed repository |
| `limit` | int | 100 | Most dependents to return, clamped to 1..500 |

**Response shape:** a bare JSON array (the VS Code extension reads it as one), one entry per dependent file:
```json
[
  {
    "file": "src/api/routes/users.py",
    "symbol": "users.create_user",
    "kind": "function",
    "line_start": 31,
    "language": "python"
  }
]
```

Each dependent costs about 130 characters, so the list is bounded by `limit` and by `TRELIX_MCP_MAX_RESULT_CHARS`. When dependents are left out, the first text block is still the JSON array, a second text block says `Truncated: N of M dependents returned, K omitted.` and ends with what can still be raised (`TRELIX_MCP_MAX_RESULT_CHARS` when the budget cut the list, `limit` when the limit did, or, at `limit=500` when the limit did, that nothing can), and `_meta.trelix` is `{"total_available": M, "omitted": K}`. A result that fits carries one text block and no `_meta.trelix`; a symbol with no dependents, or one the index does not know, gives `[]` as the text block and `{"result": []}` as the structured content. A blank `symbol_name`, or a `repo_path` that is not an indexed directory, is an error result (see [Errors](#errors)).

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

**What it does:** Records a subscription for a `trelix://` resource URI in the server's in-memory registry. This is a plain tool: the server does not advertise `resources.subscribe` and does not serve the `resources/subscribe` request. No `notifications/resources/updated` is delivered yet, because nothing starts a file watcher inside the server process (see [section 17](#17-resource-subscriptions-v250)).

**When to use:** Not useful for live updates yet, since no notification is delivered today (see [section 17](#17-resource-subscriptions-v250)). The registration itself works.

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
| `extract_concepts` | bool | `false` | Run LLM semantic-concept extraction. **Paid**: at most 10 LLM calls over the 200 most central symbols (batches of 20); requires an LLM API key. Covers 1.6% of a 12,184-symbol index — see the coverage note below |
| `min_community_size` | int | `2` | Drop communities smaller than this. `1` returns singletons too |
| `max_communities` | int | `50` | Cap on returned communities, largest first. `0` disables the cap |

**Response shape** (verified against the live tool — the key names previously documented
here, `nodes`/`edges`/`connected_components`/`max_depth`/`build_seconds`, do not exist):

```json
{
  "node_count": 10700,
  "edge_count": 11150,
  "community_count": 6497,
  "concept_count": 0,
  "concept_symbols_considered": 0,
  "concept_symbols_total": 0,
  "elapsed_seconds": 1.52,
  "community_summary": [{"community_id": 12, "size": 514, "top_files": ["..."], "top_symbols": ["..."]}],
  "singleton_count": 6437,
  "communities_omitted": 6447
}
```

**Concept-extraction coverage.** `extract_concepts=True` does not cover the repository.
It processes the **200 most central symbols** by the PageRank centrality computed during
the build, in batches of 20 — at most 10 paid LLM calls. On a 12,184-symbol index that is
**1.6% of symbols**, so `concept_count` describes that sample, not the repo.
`concept_symbols_considered` / `concept_symbols_total` report that coverage in every
response, and are `0`/`0` when extraction was off — which is what distinguishes
"extraction never ran" from "it ran over 200 symbols and found nothing". Same two numbers
as `trelix graph --concepts --json` ([CLI_REFERENCE](CLI_REFERENCE.md#trelix-graph)).

**Why the caps exist.** `community_summary` used to ship every detected community,
unsorted. On trelix's own index that is **1,160,517 bytes — roughly 290,000 tokens from
one tool call**, larger than most context windows, and 6,437 of the 6,497 entries (99.1%)
are singletons carrying no architectural signal. The defaults return 23,788 bytes (~5,900
tokens) while keeping every significant cluster. `community_count` still reports the true
total, and `singleton_count` / `communities_omitted` say exactly what was left out.

**What the caps do NOT fix:** every call still runs a full `GraphBuilder.build()` — a
Louvain pass, a PageRank rebuild and two metadata saves. The payload is smaller; the
server-side cost is unchanged.

---

### `graph_search_mcp`

```
graph_search_mcp(query, repo_path, k=10, detail="detailed") → list of results
```

**What it does:** Combines the knowledge graph topology with semantic search to surface results that are structurally central — symbols that many other symbols depend on, or that are highly connected in the call graph.

**When to use:** When you want to find the most architecturally significant code related to a concept, not just the textually closest matches.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Natural-language or keyword query |
| `repo_path` | str | required | Absolute path to the indexed repository |
| `k` | int | 10 | Number of results to return, clamped to 1..`TRELIX_MCP_MAX_K` (default 50) |
| `detail` | `"detailed"` \| `"concise"` | `"detailed"` | `concise` drops `body` and adds a one-line `signature` |

**Response shape:** a bare JSON array:
```json
[
  {
    "file": "src/db/pool.py",
    "symbol": "DatabasePool.acquire",
    "kind": "method",
    "score": 0.83,
    "source": "graph_search",
    "body": "def acquire(self) -> Connection:\n    ..."
  }
]
```

Like `blast_radius`, a list cut to `TRELIX_MCP_MAX_RESULT_CHARS` keeps the array in the first text block and adds a note block and `_meta.trelix` (see [Output size and limits](#output-size-and-limits)). With no hits the text block is `[]`; a blank `query` is an error result (see [Errors](#errors)).

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
federation_search_all(query, k=10, cursor=0, config_path=None, detail="detailed") → {results, next_cursor, total_available, page_size, truncated, omitted, repos_searched, repos_skipped, repos_unindexed, error}
```

**What it does:** Searches across ALL registered repos simultaneously using Reciprocal Rank Fusion to merge results, weighted by each repo's registered `weight`.

**Important:**
- Requires repos to already be registered via `federation_add_repo` AND already indexed. A registered repo with no index is skipped (not opened, not counted in `repos_searched`) and `repos_unindexed` names its alias; if none of the queried repos is indexed, `error` carries the `No index found at ...` message and the response keeps its shorter shape
- A blank `query` or a negative `cursor` is an error result (see [Errors](#errors))
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
| `k` | int | 10 | Results per page, clamped to 1..`TRELIX_MCP_MAX_K` (default 50) |
| `cursor` | int | 0 | Pagination offset; a negative value is an error |
| `config_path` | str\|None | None | Optional path to a custom repos.json |
| `detail` | `"detailed"` \| `"concise"` | `"detailed"` | `concise` drops `body` and adds a one-line `signature` |

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
  "page_size": 10,
  "truncated": false,
  "omitted": 0,
  "repos_searched": 2,
  "repos_skipped": 0,
  "repos_unindexed": [],
  "error": null
}
```

`repos_unindexed` lists the aliases of the queried repos that were skipped for having no index (`[]` when none was); `repos_skipped` counts those beyond the `TRELIX_FEDERATION_MAX_REPOS` cap. The error and empty-registry responses keep the shorter shape they always had (no `page_size`, `truncated`, `omitted` or `repos_unindexed`).

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
agent_list_sessions(repo_path, limit=50) → {sessions, count, page_size, truncated, omitted}
```

**What it does:** Lists recent agent sessions for a repo, most recently active first. Runs stale-session eviction first if `TRELIX_RETRIEVAL_AGENT_SESSION_MAX_AGE_SECONDS > 0`.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `repo_path` | str | required | Absolute path to the repository root |
| `limit` | int | 50 | Max sessions to return, clamped to 1..`TRELIX_MCP_MAX_K` (default 50) |

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
  "count": 1,
  "page_size": 50,
  "truncated": false,
  "omitted": 0
}
```

If the response would pass the budget (`TRELIX_MCP_MAX_RESULT_CHARS`), the oldest sessions are left out: `truncated` is `true` and `omitted` counts them. `query` is the session's most recent prompt, cut to its first 300 characters; a session whose prompt was cut also has `query_truncated: true` (the key is absent otherwise).

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

## 9. Federation Security & Configuration (v2.8.1)

### Path Confinement for `config_path`

All four federation MCP tools (`federation_list_repos`, `federation_add_repo`, `federation_remove_repo`, `federation_search_all`) accept an optional `config_path` parameter to override the default registry location. In v2.8.1, this parameter is **confined** to two allowlisted roots:

1. `~/.config/trelix/` (the default federation config directory)
2. `<mcp-server-cwd>/.trelix/` (a repo-local override when the MCP server process is launched from within a repo)

Any `config_path` that resolves outside both roots will be rejected with a `ConfigPathNotAllowedError` returned as `{"error": str}` in the tool response. This prevents an MCP client (or a prompt-injected agent) from pointing registry I/O at an arbitrary filesystem path. A blank `config_path` is a tool error (`isError: true`) instead, like every other blank argument (section 8, [Errors](#errors)): omit the argument for the default registry.

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

## 10. The 3 MCP Resources

MCP resources are read-only data endpoints that the AI can fetch without executing a tool. Use them to give the model static context about the indexed codebase.

### `trelix://index/stats`

A parameterless resource cannot know which repository is meant, so this returns a pointer
to the repo-scoped resources below. The aggregate payload previously documented here
(`repos_indexed`, `total_files`, `total_symbols`, `total_chunks`, `server_version`) was
never produced by any code — the resource returned only a hint string.

```json
{"hint": "Use trelix://repo/{repo_path}/stats for index statistics, or trelix://repo/{repo_path}/manifest for the file listing"}
```

### `trelix://repo/{repo_path}/stats`

Symbol, file and chunk counts for one indexed repository. Three `COUNT(*)` queries — no
embedder, no graph build — so an agent can cheaply check whether a repo is indexed at all
before committing to a search.

```json
{"symbol_count": 10700, "file_count": 459, "chunk_count": 10700, "repo_path": "/path/to/repo"}
```

### `trelix://repo/{repo_path}/manifest`

Returns indexed files for a specific repository with language and symbol count — the
**first 500 by path**, not "the full list" as previously documented.

```
trelix://repo//Users/you/projects/myapp/manifest
```

```json
{
  "repo_path": "/Users/you/projects/myapp",
  "file_count": 500,
  "total_file_count": 620,
  "files_truncated": true,
  "files": [
    {"path": "src/auth/service.py", "language": "python", "symbols": 12, "size_bytes": 4210},
    {"path": "src/db/pool.py", "language": "python", "symbols": 8, "size_bytes": 2108}
  ]
}
```

`file_count` is the size of the returned PAGE and keeps that meaning for existing
consumers. It was previously the only count in the response, so a repository with more
than 500 indexed files reported exactly 500 as its file count, with nothing to indicate a
limit had been applied. `total_file_count` is the real number and `files_truncated` says
whether you are looking at a page.

### `trelix://repo/{repo_path}/symbols/{qualified_name}`

Returns the raw source of a symbol without invoking the `get_symbol` tool. Useful for embedding static symbol definitions in prompts.

```
trelix://repo//Users/you/projects/myapp/symbols/AuthService.login
```

---

## 11. The 3 MCP Prompts

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

## 12. Watch Bridge (v2.7.0)

**Status: no MCP client receives `notifications/resources/updated` today.** After a file re-index that was not skipped, `trelix watch` calls `notify_file_changed()`, which looks up the subscriptions that `subscribe_resource` registered for the repo's manifest URI. `trelix watch` runs in its own process, where that registry is empty (subscriptions live in the `trelix-mcp` stdio server process), and nothing in trelix-mcp starts a file watcher. The notification can only reach a client when the watcher runs inside the stdio server process. See [section 17](#17-resource-subscriptions-v250).

Once delivery works, this is intended for:
- Keeping codebase context fresh during active development
- Triggering automated analysis pipelines when code changes
- Multi-agent coordination where file changes need propagation

---

## 13. v2.4.0 Breaking Change — `search_code` Pagination

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
- If `next_cursor` is null, you have reached the last page

---

## 14. Pagination Example (Full Paging Loop)

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

        # next_cursor is null after the last page. A page cut to fit the output budget
        # (response["truncated"]) sets next_cursor to the first result it dropped, so the
        # same loop continues without skipping or repeating a result.
        if next_cursor is None or not batch:
            break

        cursor = next_cursor

    return all_results
```

---

## 15. IDE Integrations

### VS Code Extension

The `workspace-vscode/` extension surfaces trelix inside the editor through palette commands, a hover provider, actionable code lenses, and a `@trelix` chat participant. It spawns `trelix-mcp` over stdio on first use — the same server this guide configures for every other client — so nothing extra needs to be running.

- **`trelix.search`** — Search the workspace codebase with trelix hybrid search
- **`trelix.ask`** — Ask a natural-language question about the code
- **Hover** — hovering over any identifier looks it up via `get_symbol` and shows its signature, docstring, and file/line location. Results are cached per word+repo for the session (no TTL). **Known limitation:** `get_symbol` falls back to an ambiguous bare-name lookup when the exact qualified name doesn't resolve — if multiple symbols share a name, hover may show the wrong one.

Install from the `workspace-vscode/` directory:

```bash
cd workspace-vscode && npm install && code --install-extension .
```

Then use in the command palette (Cmd+Shift+P):
- `trelix: Search Codebase` — Opens search input, runs hybrid query
- `trelix: Ask about Code` — Opens question input, shows the synthesized answer in a side panel

`trelix: Find Similar Code` (`trelix.findSimilar`) and `trelix: Show Blast Radius` (`trelix.blastRadius`) are also registered, but deliberately hidden from the palette — they are what the code lenses below invoke.

**Fixed in v3.0.0:** `trelix.ask` used to call `getPrompt("trelix-search")` and render the interpolated *prompt template* as if it were the answer. It now calls the `ask_agent` tool and renders the real `answer` (the tool also returns `session_id` and `turn_count`).

#### Actionable code lenses

Two lenses appear above every symbol reported by VS Code's own document-symbol provider:

| Lens | Calls | What you get |
|------|-------|--------------|
| `Find similar` | `search_code`, seeded with the symbol name | A QuickPick of semantically similar code; pick one to jump to it |
| `N dependents` | `blast_radius` on that symbol | VS Code's native Peek References popup listing the symbols that call/import it; pick one to jump to `file:line` |

- **Setting:** `trelix.codeLens.enabled` (boolean, default `true`) — "Show trelix code lenses (Find similar, blast radius) above symbols". Flipping it re-queries lenses immediately; no window reload needed.
- Symbols come from `vscode.executeDocumentSymbolProvider`, i.e. whichever language extension you already have installed. A file with no symbol provider gets no lenses (and no error).
- At most 200 symbols per document are annotated, and nesting is followed to depth 2 — top-level symbols, their children, and their grandchildren — so a deeply nested file does not produce an unreadable wall of lenses.
- **Lens resolution is lazy, so typing never triggers MCP traffic.** `provideCodeLenses` makes zero MCP calls: it derives lens ranges locally and returns the count-bearing lens *unresolved*. The single `blast_radius` call happens in `resolveCodeLens`, which VS Code invokes only for lenses it actually paints. Each result is cached per `uri@version::symbol`, so scrolling back over the same revision is free, while an edit bumps the document version and correctly invalidates the count.
- Lens failures are silent by design — never an error dialog. A `blast_radius` call that errors resolves the lens to `0 dependents` for that document revision, so treat a surprising zero as "check the server" rather than "nothing depends on this".
- **Long dependent lists.** A `trelix-mcp` that cuts a long `blast_radius` list (the release that ships the output budget) sends the real count in `_meta.trelix.total_available`, and the extension reads it. The lens then shows the real count and how many the popup lists (`150 dependents (showing 100)`), and `@trelix /impact` says `has 150 dependent(s), showing the first 100`. When nothing was cut, or the server sends no `_meta` (an older release), the count is the number of entries received.

#### `@trelix` chat participant

In the Chat view, type `@trelix` followed by a question. The participant is sticky, so follow-ups stay addressed to trelix without retyping the mention.

| Invocation | Calls | Behavior |
|------------|-------|----------|
| `@trelix <question>` | `ask_agent` | Runs the agentic ReAct loop and renders the answer as markdown (a progress note shows while retrieval runs; the answer itself arrives in one piece, not token-by-token) |
| `@trelix /search <query>` | `search_code` | Up to 10 results, each listed as `` `symbol` — file:lines (kind) `` plus a clickable reference to that line range |
| `@trelix /explain [question]` | `ask_agent` | Explains the active editor's selection (and/or the text you type) in the context of the codebase |
| `@trelix /impact <symbol>` | `blast_radius` | Lists the symbols that depend on it, with a clickable reference each; if the server cut a long list it reports the real count and says `showing the first N`. Falls back to the editor selection if you pass no symbol name |

Invoking `@trelix` with no prompt prints a usage hint instead of calling the server, and any client error is rendered as markdown in the chat rather than thrown.

**Availability.** The extension declares `engines.vscode: ^1.90.0`, but the chat participant registers only when `vscode.chat.createChatParticipant` exists. On builds that do not ship the chat API, activation succeeds normally and the participant is simply absent — you get the palette commands, hover, and code lenses, with no error and no failed activation.

**Session limitation (honest).** The chat API exposes no thread/conversation id, so trelix sessions are keyed by a single constant: **two chat threads open at the same time share one trelix agent session.** The only per-conversation signal available is the chat history — an empty history starts a fresh session, a non-empty one resumes the stored `session_id`. Note also that only plain `@trelix` asks participate in the session; `/search`, `/explain`, and `/impact` are stateless one-shot calls.

---

## 16. Example Claude Code Session

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

## 17. Resource Subscriptions (v2.5.0)

trelix-mcp v2.5.0 added two tools modelled on the MCP resource subscription flow
([MCP spec §Resources](https://modelcontextprotocol.io/specification/2024-11-05/server/resources)).
It does not implement the protocol's `resources/subscribe` request.

### What works today

1. The server does not advertise `resources.subscribe` (it reports `false`) and does not serve `resources/subscribe`; a client that sends it gets "Method not found"
2. Clients register a URI with the `subscribe_resource` tool instead; the registration lives in memory inside the `trelix-mcp` stdio server process
3. `notifications/resources/updated` is sent only if `notify_file_changed()` runs inside that same process, and nothing in trelix-mcp starts a file watcher, so no client receives one yet
4. Once a notification arrives, the client would call `resources/read` to fetch the updated index content

### Subscription tools

**`subscribe_resource(uri, subscription_id)`**
Register a subscription for a trelix:// resource URI.

```
uri:             trelix://repo//path/to/repo/manifest
subscription_id: any string — used to correlate notifications
```

**`unsubscribe_resource(subscription_id)`**
Remove a subscription by its ID.

### Wire protocol (target flow; only the first step works today)

```
Client → Server:  tools/call subscribe_resource  { uri, subscription_id }   (works today)
Server → Client:  notifications/resources/updated  { uri, _meta: { subscriptionId } }   (not sent today)
Client → Server:  resources/read  { uri }
```

---

## 18. Troubleshooting MCP Issues

### `trelix-mcp: command not found`

The binary is not on your PATH. Fix:

```bash
# Check where pip installed it
python -m site --user-base
# e.g. /Users/you/Library/Python/3.12

# Add to PATH in ~/.zshrc or ~/.bashrc
export PATH="$HOME/Library/Python/3.12/bin:$PATH"
source ~/.zshrc

# Verify (trelix-mcp with no arguments starts the stdio server; --version prints the version instead)
which trelix-mcp
python -c "import trelix_mcp; print(trelix_mcp.__version__)"
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
- Trelix skips files matched by `.gitignore` — the repo-root file *and*, as of v3.1.2, nested `.gitignore` files in subdirectories (the one closest to a path wins). There is no `.trelixignore`. If expected sources are missing, check both the root `.gitignore` and any `.gitignore` in the directories above them, or set `TRELIX_WALKER_RESPECT_GITIGNORE=false` to index them anyway. Note that variable is read from the **process environment only** — putting it in `.env` has no effect (see `docs/CONFIGURATION.md`).
- Large repos may hit memory limits. There is no worker/parallelism env var; the levers are the walker's file-size ceiling and the embedder batch size:

```bash
TRELIX_WALKER_MAX_FILE_SIZE_BYTES=200000 TRELIX_EMBEDDER_BATCH_SIZE=16 trelix-mcp
```

> `TRELIX_WALKER_*` variables are read from the process environment only — they ignore `.env`. See [CONFIGURATION.md](CONFIGURATION.md#file-walker-which-files-get-indexed).

### `search_code` returns an error: `No index found at ...`

The repository has no index. Run `index_codebase` (or `trelix index <repo>`) and check that it returned `files_indexed > 0` before querying. Every tool that reads an index (`search_code`, `get_symbol`, `blast_radius`, `build_knowledge_graph`, `graph_search_mcp`, `ask_agent`, `agent_list_sessions`, `agent_clear_session`) answers this way, as a normal tool error (`isError: true`) that leaves the session running, and none of them creates `.trelix/` on the way. Earlier releases returned an empty result and left an empty `index.db` behind, so a repository that was never indexed looked indexed.

### A tool returns an error naming an argument

The call had an argument the tool cannot use: the text names it, shows what was given and says what is valid (`repo_path is not a directory: ...`, `query must not be empty or whitespace ...`; the table is under [Errors](#errors) in section 8). Correct the argument and call again; nothing was created or changed.

### `search_code` returns empty results

The repository is indexed but nothing matched, or the index holds no files: check that `index_codebase` returned `files_indexed > 0`.

### `graph_search_mcp` returns no results or errors

The knowledge graph must be built separately from the index. Call `build_knowledge_graph(repo_path)` after indexing.

### MCP server crashes silently in Cursor / Windsurf

`trelix-mcp` accepts only `--help`, `--version` and `--tools core|full`, and has no log-file or log-level setting — it logs
unconditionally to **stderr** at INFO via a hardcoded `logging.basicConfig`. To capture that
stream, point the MCP host at a tiny wrapper that redirects stderr to a file:

```bash
cat > /usr/local/bin/trelix-mcp-logged <<'EOF'
#!/bin/sh
exec trelix-mcp 2>>/tmp/trelix-mcp.log
EOF
chmod +x /usr/local/bin/trelix-mcp-logged
```

```json
{
  "mcpServers": {
    "trelix": {
      "command": "/usr/local/bin/trelix-mcp-logged",
      "args": []
    }
  }
}
```

Then inspect `/tmp/trelix-mcp.log` for the error. Do **not** redirect stdout — stdout is the
MCP stdio transport, and writing to it corrupts the protocol stream.

### Version mismatch between trelix-mcp and an existing index

There is no separate MCP cache directory — `trelix-mcp` reads the same index the CLI
writes, at `<repo>/.trelix/index.db`. If you downgrade `trelix-mcp` (or switch
embedding providers) and the existing index is no longer compatible, delete the index
and re-index the repo:

```bash
rm -rf ./my-repo/.trelix/index.db
trelix index ./my-repo
```

If the failure is specifically an embedding-dimension mismatch, clear just the stored
vectors instead of the whole DB:

```bash
trelix migrate-vectors ./my-repo --reset --provider <new-provider>
trelix index ./my-repo
```

### Pagination returns duplicate results

Duplicates indicate that the index changed between pages (a background re-index ran). Re-run the full query from `cursor=0` to get a consistent snapshot.
