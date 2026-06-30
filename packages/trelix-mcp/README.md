# trelix-mcp

<!-- mcp-name: io.github.sairam0424/trelix -->

MCP server for [trelix](https://github.com/sairam0424/trelix) v2.1.0 — semantic code search with streaming /ask endpoint, REST API integration, and beast-mode retrieval optimization for Claude Code, Cursor, Windsurf, and Continue.dev.

## Install

```bash
pip install trelix-mcp==2.1.0
```

To use Bedrock embeddings or synthesis (no extra API key beyond AWS credentials):

```bash
pip install "trelix-mcp==2.0.0" "trelix[bedrock]"
```

Other optional LLM provider extras:

```bash
pip install "trelix-mcp==2.0.0" "trelix[anthropic]"   # Anthropic Claude direct
pip install "trelix-mcp==2.0.0" "trelix[vertex]"       # Google Vertex AI / Gemini
pip install "trelix-mcp==2.0.0" "trelix[litellm]"      # 100+ providers via LiteLLM
pip install "trelix-mcp==2.0.0" "trelix[llm-all]"      # all LLM providers
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

| Tool | Version | Description |
|------|---------|-------------|
| `search_code` | v2.1.0 | Semantic hybrid search over an indexed codebase with 5-leg retrieval (base, summaries, HyDE, FLARE, PageRank boost) |
| `index_codebase` | v2.1.0 | Index a repository with optional file summaries and knowledge graph |
| `get_symbol` | v2.1.0 | Look up a symbol by qualified name |
| `blast_radius` | v2.1.0 | Find all files that depend on a symbol |
| `ask` | v2.1.0 | Streaming chat endpoint for conversational code exploration |
| `build_knowledge_graph` | v2.1.0 | Build Code Property Graph, detect communities, return stats + community summary |
| `graph_search_mcp` | v2.1.0 | Vector-seeded graph BFS — finds structurally related code by following call/import/type edges |

## Knowledge Graph Tools

Two new tools in v2.0.0 expose the knowledge graph layer to AI agents:

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
pip install 'trelix-mcp' 'trelix[knowledge-graph]'
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
