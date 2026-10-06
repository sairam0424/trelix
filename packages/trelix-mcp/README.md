# trelix-mcp

<!-- mcp-name: io.github.sairam0424/trelix -->

MCP server for [trelix](https://github.com/sairam0424/trelix) v3.3.7 — semantic code search with streaming /ask endpoint and REST API integration for Claude Code, Cursor, Windsurf, and Continue.dev.

## ⚠️ Breaking Change in v2.4.0

`search_code` now returns a **pagination envelope** instead of a bare list:

```python
# v2.3.x (old)
results = search_code(query="auth", repo_path="/repo")
for r in results:  # results was list[dict]
    print(r["symbol"])

# v2.4.0 (new)
response = search_code(query="auth", repo_path="/repo")
for r in response["results"]:  # now dict with pagination
    print(r["symbol"])
# Paginate: pass response["next_cursor"] as cursor= for next page
```

## Install

```bash
pip install trelix-mcp
```

Unpinned on purpose. `trelix-mcp` carries the core version stamp and is published only by a
core `v*` tag, so the newest `trelix-mcp` is always the one built alongside the newest
`trelix`. To pin in your own `requirements.txt`, pin all four distributions to the same
version — see [the FAQ](../../docs/FAQ.md#is-trelix-suitable-for-production-use).

To use Bedrock embeddings or synthesis (no extra API key beyond AWS credentials):

```bash
pip install trelix-mcp "trelix[bedrock]"
```

Other optional LLM provider extras:

```bash
pip install trelix-mcp "trelix[anthropic]"   # Anthropic Claude direct
pip install trelix-mcp "trelix[vertex]"      # Google Vertex AI / Gemini
pip install trelix-mcp "trelix[litellm]"     # 100+ providers via LiteLLM
pip install trelix-mcp "trelix[llm-all]"     # all LLM providers
```

## Usage

### Claude Code

```bash
claude mcp add trelix -- trelix-mcp
```

To list only the everyday search and indexing tools, add `--tools core` (the default is `--tools full`, every tool):

```bash
claude mcp add trelix -- trelix-mcp --tools core
```

### Cursor (`~/.cursor/mcp.json`)

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

### Continue.dev (`.continue/config.json`)

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

## Configuration

Set environment variables before starting the MCP server. All variables are optional — defaults work out of the box with the `local` embedding provider and `openai` chat provider.

### Embedding provider

```bash
# Local sentence-transformers — no API key (default)
TRELIX_EMBEDDER_PROVIDER=local

# Local BGE Code — EXPERIMENTAL: pooling unverified, no quality claim, no API key
TRELIX_EMBEDDER_PROVIDER=bge-code

# Local Nomic Code — competitive code embeddings, no API key
TRELIX_EMBEDDER_PROVIDER=nomic-code

# Azure OpenAI embeddings
TRELIX_EMBEDDER_PROVIDER=azure
AZURE_API_KEY=...
AZURE_ENDPOINT=https://<resource>.openai.azure.com/

# Voyage AI — best API-based code embeddings (CoIR 56.26)
TRELIX_EMBEDDER_PROVIDER=voyage
VOYAGE_API_KEY=...

# AWS Bedrock Cohere — strong code retrieval, no extra key beyond AWS creds
TRELIX_EMBEDDER_PROVIDER=bedrock-cohere
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# AWS Bedrock Titan v2 — configurable 256/512/1024 dims
TRELIX_EMBEDDER_PROVIDER=bedrock-titan
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

### Chat / synthesis provider (used by `index_codebase` contextual chunking and synthesis)

```bash
# OpenAI (default)
TRELIX_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Azure GPT-4o
TRELIX_LLM_PROVIDER=azure
AZURE_API_KEY=...
AZURE_ENDPOINT=https://<resource>.openai.azure.com/

# AWS Bedrock — Claude Sonnet 4.6 default with auto-fallback to Haiku
TRELIX_LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
# Optional overrides:
TRELIX_LLM_BEDROCK_PRIMARY_MODEL=us.anthropic.claude-sonnet-4-6
TRELIX_LLM_BEDROCK_FALLBACK_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0

# Anthropic direct
TRELIX_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Vertex AI / Gemini
TRELIX_LLM_PROVIDER=vertex
GOOGLE_CLOUD_PROJECT=my-project
GOOGLE_CLOUD_LOCATION=us-central1

# LiteLLM — 100+ providers
TRELIX_LLM_PROVIDER=litellm
TRELIX_LLM_MODEL=bedrock/claude-3-5-sonnet
```

### Resource subscription limits

```bash
# Max concurrent resource subscriptions across all clients — caps unbounded
# growth from a client that subscribes repeatedly without ever unsubscribing
TRELIX_MCP_MAX_SUBSCRIBERS=1000

# Auto-evict a subscription after this many seconds of inactivity
TRELIX_MCP_SUBSCRIPTION_TTL_SECONDS=3600
```

## Tools

`tools/list` returns the tools in a fixed order and marks each one with MCP annotation hints; the server also sends short `instructions` that tell a model to index first and which search tool to use next. Only `search_code`, `get_symbol` and `blast_radius` are marked `readOnlyHint: true` (each is tested to leave the index database byte-identical once the server's database connections are closed, with telemetry off and the index already at the current schema; `TRELIX_TELEMETRY_ENABLED=true` adds a `query_telemetry` row per `search_code`, the first open of an index written by an older trelix migrates it, and every `search_code` writes a small JSON trace of the query to `.trelix/debug/`, so the hint means the index database is left alone, not the whole repository directory); `agent_clear_session` and `federation_remove_repo` are marked `destructiveHint: true`; every tool has `openWorldHint: false`. The full table is in [docs/MCP_GUIDE.md](https://github.com/sairam0424/trelix/blob/main/docs/MCP_GUIDE.md), section 8.

`--tools full` (default) lists all 15 tools. `--tools core` lists seven: `index_codebase`, `search_code`, `get_symbol`, `blast_radius`, `build_knowledge_graph`, `graph_search_mcp` and `ask_agent`. The other eight are hidden, not removed. (`repo_map` and `exact_search` do not exist in this server, so `core` does not list them.)

| Tool | Description |
|------|-------------|
| `search_code(query, repo_path, k=10, cursor=0)` | Hybrid semantic+BM25 search with cursor pagination |
| `index_codebase(repo_path, provider="local")` | Index a repo (run once); emits progress notifications |
| `get_symbol(qualified_name, repo_path)` | Get full source of a symbol by qualified name |
| `blast_radius(symbol_name, repo_path)` | Direct callers + importers of a symbol, from the call/import graph (no embedding model, ~60-120 ms) |
| `build_knowledge_graph(repo_path)` | Build code property graph |
| `graph_search_mcp(query, repo_path)` | Search via knowledge graph |
| `subscribe_resource(uri, subscription_id)` | Register a subscription for a trelix:// resource URI (v2.5.0+); no notification is delivered yet, see Resource Subscriptions |
| `unsubscribe_resource(subscription_id)` | Cancel a resource subscription (v2.5.0+) |

### Multi-Repo Federation Tools (v2.8.0)

| Tool | Description |
|------|-------------|
| `federation_list_repos(config_path=None)` | List all repos registered for federated search |
| `federation_add_repo(alias, path, weight=1.0, config_path=None)` | Register a repo for federated search (absolute path required) |
| `federation_remove_repo(alias, config_path=None)` | Unregister a repo by alias |
| `federation_search_all(query, k=10, cursor=0, config_path=None)` | Search across all registered repos with RRF-weighted fusion |

### Persistent Agent Session Tools (v2.8.0)

| Tool | Description |
|------|-------------|
| `ask_agent(query, repo_path, session_id=None)` | Multi-turn ReAct Q&A with persistent memory (pass session_id to resume) |
| `agent_list_sessions(repo_path, limit=50)` | List recent agent sessions for a repo |
| `agent_clear_session(repo_path, session_id)` | Delete a persisted agent session and all its turn history |

## Resource Subscriptions (v2.5.0)

trelix-mcp has tools for registering interest in a resource, but it does not push index change notifications yet: see the status note below. Register a URI with the `subscribe_resource` tool.

```python
# Subscribe to a repo manifest
subscribe_resource(
    uri="trelix://repo//path/to/repo/manifest",
    subscription_id="my-sub-001"
)
# → the URI is registered; no notification is delivered yet (see below)

# Cancel the subscription
unsubscribe_resource(subscription_id="my-sub-001")
```

The server does not advertise `resources.subscribe` (it reports `false`) and does not serve the `resources/subscribe` request: `subscribe_resource` and `unsubscribe_resource` are ordinary tools that record URIs in an in-memory registry inside the `trelix-mcp` process. URIs follow the scheme `trelix://repo/{repo_path}/manifest`. The `notify_file_changed()` hook (wired into `FileWatcher._do_reindex` since v2.7.0) would fire per-URI notifications with the `subscriptionId` in `params._meta`, but it runs in the `trelix watch` process, where this registry is empty, and nothing starts a watcher inside `trelix-mcp` — so no notification is delivered today.

Subscriptions are capped and TTL'd by default — see [Resource subscription limits](#resource-subscription-limits) (`TRELIX_MCP_MAX_SUBSCRIBERS`, default `1000`; `TRELIX_MCP_SUBSCRIPTION_TTL_SECONDS`, default `3600`). Subscriptions older than the TTL are swept on the next registry access. If the registry is at capacity, `subscribe_resource` does not raise — it returns `{"subscribed": false, "uri": ..., "subscription_id": ..., "error": "..."}` so callers can handle rejection gracefully.

## Pagination

`search_code` supports cursor-based pagination for large codebases:

```python
# Fetch page 1
page1 = search_code(query="authentication", repo_path="/repo", k=10)
print(page1["total_available"])  # total results
print(page1["results"])          # this page's results

# Fetch page 2 if more results exist
if page1["next_cursor"] is not None:
    page2 = search_code(query="authentication", repo_path="/repo", k=10, cursor=page1["next_cursor"])
```

## Knowledge Graph Tools

Two tools expose the knowledge graph layer to AI agents:

### build_knowledge_graph

Builds a Code Property Graph over an indexed repo. Returns node/edge counts, community count, and a summary of top architectural clusters.

```
build_knowledge_graph(repo_path="/path/to/repo")
→ {node_count: 4599, edge_count: 4945, community_count: 2409, community_summary: [...]}
```

Use this before `graph_search_mcp` for best results — or let `graph_search_mcp` call it automatically.

### graph_search_mcp

Hybrid search: first retrieves semantic seeds, then expands via BFS over call/import/type edges.

```
graph_search_mcp(query="how does auth relate to the user model?", repo_path="/path/to/repo", k=10)
→ [{file, symbol, kind, score, source, body}, ...]
```

**When to use `graph_search_mcp` instead of `search_code`:**
- "What does X depend on?"
- "What would break if I change Y?"
- "How does module A connect to module B?"
- Architecture understanding queries where structural relationships matter

Install the knowledge graph extra for full functionality:

```bash
pip install trelix-mcp 'trelix[knowledge-graph]'
```

## Watch Bridge (v2.7.0)

After each file re-index, `trelix watch` calls `notify_file_changed()`, which writes `notifications/resources/updated` for the URIs registered with `subscribe_resource`. **No MCP client receives these notifications today.** `trelix watch` is a separate process from `trelix-mcp` and its subscription registry is empty, and nothing in trelix-mcp starts a file watcher, so running `trelix-mcp` in one terminal and `trelix watch /path/to/repo` in another does not connect them. The notification can only reach a client when the watcher runs inside the stdio server process.

Registering a subscription to a repository's manifest URI works; `subscribe_resource` takes two
required strings. There is no glob support:

```
subscribe_resource(uri="trelix://repo//path/to/repo/manifest", subscription_id="my-sub-001")
```

Once delivery works, a notification will carry only
`{"uri": ..., "_meta": {"subscriptionId": ...}}` — no file paths and no stats. Call
`resources/read` on that URI to see what changed.
