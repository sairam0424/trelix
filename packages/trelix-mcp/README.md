# trelix-mcp

<!-- mcp-name: io.github.sairam0424/trelix -->

MCP server for [trelix](https://github.com/sairam0424/trelix) v2.7.1 — semantic code search with streaming /ask endpoint, watch bridge notifications, and REST API integration for Claude Code, Cursor, Windsurf, and Continue.dev.

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
pip install trelix-mcp==2.7.1
```

To use Bedrock embeddings or synthesis (no extra API key beyond AWS credentials):

```bash
pip install "trelix-mcp==2.7.1" "trelix[bedrock]"
```

Other optional LLM provider extras:

```bash
pip install "trelix-mcp==2.7.1" "trelix[anthropic]"   # Anthropic Claude direct
pip install "trelix-mcp==2.7.1" "trelix[vertex]"       # Google Vertex AI / Gemini
pip install "trelix-mcp==2.7.1" "trelix[litellm]"      # 100+ providers via LiteLLM
pip install "trelix-mcp==2.7.1" "trelix[llm-all]"      # all LLM providers
```

## Usage

### Claude Code

```bash
claude mcp add trelix -- trelix-mcp
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

# Local BGE Code (v1.5) — superior code retrieval, no API key
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

## Tools

| Tool | Description |
|------|-------------|
| `search_code(query, repo_path, k=10, cursor=0)` | Hybrid semantic+BM25 search with cursor pagination |
| `index_codebase(repo_path, provider="local")` | Index a repo (run once); emits progress notifications |
| `get_symbol(qualified_name, repo_path)` | Get full source of a symbol by qualified name |
| `blast_radius(symbol_name, repo_path)` | Find what depends on a symbol |
| `ask` | Streaming chat endpoint for conversational code exploration (v2.0.0+) |
| `build_knowledge_graph(repo_path)` | Build code property graph |
| `graph_search_mcp(query, repo_path)` | Search via knowledge graph |
| `subscribe_resource(uri, subscription_id)` | Subscribe to change notifications for a trelix:// resource URI (v2.5.0+) |
| `unsubscribe_resource(subscription_id)` | Cancel a resource subscription (v2.5.0+) |

## Resource Subscriptions (v2.5.0)

trelix-mcp supports live index change notifications. When `trelix watch` detects a file change, connected MCP clients receive a `notifications/resources/updated` push — then call `resources/read` to fetch the updated index. Subscribe with the `subscribe_resource` tool.

```python
# Subscribe to a repo manifest
subscribe_resource(
    uri="trelix://repo//path/to/repo/manifest",
    subscription_id="my-sub-001"
)
# → client receives notifications/resources/updated when trelix watch fires
# → call resources/read on the URI to get the refreshed index

# Cancel the subscription
unsubscribe_resource(subscription_id="my-sub-001")
```

The `resources.subscribe` capability is advertised in server capabilities. URIs follow the scheme `trelix://repo/{repo_path}/manifest`. The `notify_file_changed()` hook (wired into `FileWatcher._do_reindex` since v2.7.0) fires per-URI notifications with the `subscriptionId` in `params._meta`.

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
pip install 'trelix-mcp==2.7.1' 'trelix[knowledge-graph]'
```

## Watch Bridge (v2.7.0)

The `trelix watch` command now fires MCP notifications after every file re-index, allowing real-time codebase awareness across all subscribed clients:

```bash
# Terminal 1: Start trelix-mcp
trelix-mcp

# Terminal 2: Enable file watching
trelix watch /path/to/repo
```

Clients can subscribe to file changes via `subscribe_resource` with glob patterns:

```
subscribe_resource(["src/**/*.ts", "tests/**/*.test.ts"])
```

After each re-index, all clients receive `notifications/resources/updated` containing changed file paths and re-index stats.
