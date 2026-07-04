# trelix-mcp

<!-- mcp-name: io.github.sairam0424/trelix -->

MCP server for [trelix](https://github.com/sairam0424/trelix) v2.4.0 — semantic code search with streaming /ask endpoint and REST API integration for Claude Code, Cursor, Windsurf, and Continue.dev.

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
pip install trelix-mcp==2.4.0
```

To use Bedrock embeddings or synthesis (no extra API key beyond AWS credentials):

```bash
pip install "trelix-mcp==2.4.0" "trelix[bedrock]"
```

Other optional LLM provider extras:

```bash
pip install "trelix-mcp==2.4.0" "trelix[anthropic]"   # Anthropic Claude direct
pip install "trelix-mcp==2.4.0" "trelix[vertex]"       # Google Vertex AI / Gemini
pip install "trelix-mcp==2.4.0" "trelix[litellm]"      # 100+ providers via LiteLLM
pip install "trelix-mcp==2.4.0" "trelix[llm-all]"      # all LLM providers
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
pip install 'trelix-mcp==2.4.0' 'trelix[knowledge-graph]'
```

## Beast-Mode Configuration (v2.1.0)

All beast-mode features are opt-in via environment variables. Set them before starting trelix-mcp to enable advanced retrieval optimization and codebase intelligence:

### Environment Variables

```bash
# File Summaries — generate semantic summaries during indexing
TRELIX_FILE_SUMMARIES_ENABLED=true

# 5-Leg Retrieval — use file summaries as an additional retrieval dimension
TRELIX_RETRIEVAL_FILE_SUMMARY_LEG=true

# HyDE Fallback — synthesize hypothetical snippets when semantic results are weak
TRELIX_RETRIEVAL_HYDE_FALLBACK=true

# FLARE Re-retrieval — confidence-gated follow-up retrieval for ambiguous queries
TRELIX_RETRIEVAL_FLARE=true

# PageRank Boosting — amplify results for architecturally central symbols
TRELIX_RETRIEVAL_PAGERANK_BOOST=true

# Telemetry — record search queries and performance metrics (privacy: local only)
TRELIX_TELEMETRY_ENABLED=true
```

### Activation Workflow

For first-time setup with beast-mode enabled:

```bash
# 1. Index the repository with file summaries (one-time, requires LLM API)
#    Adds ~30-50% to index time but produces richer context
trelix index /path/to/repo

# 2. Build knowledge graph and PageRank centrality scores
#    Enables structural graph analysis and architecture-aware boosting
trelix graph /path/to/repo

# 3. MCP server now automatically uses all 5 retrieval legs
#    Start the server and beast-mode is active
trelix-mcp
```

### Five-Leg Retrieval Architecture (v2.1.0)

`search_code` now combines five independent retrieval paths:

| Leg | Enabled by | When it helps |
|-----|-----------|---------------|
| **Base** | Always | Exact semantic matches |
| **File Summaries** | `TRELIX_RETRIEVAL_FILE_SUMMARY_LEG=true` | Broad topic queries ("where's auth?") |
| **HyDE Synthetic** | `TRELIX_RETRIEVAL_HYDE_FALLBACK=true` | Weak embedding match or novel vocabulary |
| **FLARE Re-retrieval** | `TRELIX_RETRIEVAL_FLARE=true` | Ambiguous multi-intent queries |
| **PageRank Boost** | `TRELIX_RETRIEVAL_PAGERANK_BOOST=true` | Architecture-aware results (central modules first) |

All five legs run in parallel and results are merged by relevance score.

### Performance Notes

- **Index time**: +30–50% with `TRELIX_FILE_SUMMARIES_ENABLED=true` (LLM API calls)
- **Query latency**: +5–15ms per query with all legs enabled (negligible for interactive use)
- **Storage**: ~15–20% larger index with summaries
- **Cost**: Proportional to file count during indexing; queries have minimal ongoing cost

Disable individual legs via env vars to find your latency/quality sweet spot.

## Beast-Mode Configuration

Enable advanced retrieval and indexing features via environment variables. All are optional — defaults use single-leg semantic search with local embeddings.

```bash
# v2.0.0+ features
TRELIX_RETRIEVAL_HYBRID=true         # BM25 + semantic (Reciprocal Rank Fusion)
TRELIX_RETRIEVAL_RERANK=true         # Cross-encoder reranking for top-k
TRELIX_GRAPH_ENABLED=true            # Graph search alongside vector search

# v2.2.0 additions
TRELIX_RETRIEVAL_AGENTIC=true        # multi-turn ReAct loop for uncertain queries
TRELIX_PARSER_DATAFLOW=true          # def-use chain extraction at index time
TRELIX_RETRIEVAL_SPARSE=true         # SPLADE-Code 6th RRF leg (pip install trelix[sparse])
TRELIX_CHUNKER_MULTI_GRANULARITY=true  # block+statement indexing
TRELIX_RETRIEVAL_SUB_CHUNK=true      # use sub-symbol chunks as 7th RRF leg
```
