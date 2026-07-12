# Trelix Configuration Reference — v2.7.1

Complete reference for all configuration options available in trelix.

---

## Configuration Methods

Settings are resolved in priority order (highest wins):

1. **Environment variables** — set in the shell or via CI/CD secrets
2. **.env file** — `<repo-root>/.env`, auto-loaded on startup
3. **Defaults** — built-in fallbacks documented in the tables below

---

## Environment Variables

### Embedder

| Variable | Default | Description |
|---|---|---|
| `TRELIX_EMBEDDER_PROVIDER` | `local` | Embedding provider. One of: `local`, `openai`, `azure`, `voyage`, `bge-code`, `nomic-code` |
| `TRELIX_EMBEDDER_OPENAI_MODEL` | `text-embedding-3-small` | OpenAI embedding model name |
| `TRELIX_EMBEDDER_AZURE_DEPLOYMENT` | `text-embedding-3-small` | Azure deployment name for embeddings |
| `TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS` | _(none)_ | Matryoshka output dimension for Voyage models. Accepted values: `256`, `512`, `1024`, `2048` |
| `OPENAI_API_KEY` | _(none)_ | API key — required when `TRELIX_EMBEDDER_PROVIDER=openai` |
| `AZURE_API_KEY` | _(none)_ | API key — required when `TRELIX_EMBEDDER_PROVIDER=azure` |
| `AZURE_ENDPOINT` | _(none)_ | Full Azure endpoint URL (e.g. `https://<name>.openai.azure.com/`) |
| `VOYAGE_API_KEY` | _(none)_ | API key — required when `TRELIX_EMBEDDER_PROVIDER=voyage` |

### Retrieval

| Variable | Default | Description |
|---|---|---|
| `TRELIX_RETRIEVAL_MULTI_QUERY` | `false` | Enable multi-query expansion — generates additional query variants to improve recall |
| `TRELIX_RETRIEVAL_MULTI_QUERY_COUNT` | `2` | Number of query variants to generate when multi-query is enabled |
| `TRELIX_RETRIEVAL_SHORT_QUERY_LEXICAL` | `false` | Route short queries (≤threshold tokens) to BM25+grep only, skipping vector ANN |
| `TRELIX_RETRIEVAL_SHORT_QUERY_TOKENS` | `5` | Meaningful-token threshold for short-query classification (1–10) |
| `TRELIX_INDEXER_STREAMING` | `false` | Enable generator-based streaming indexing pipeline (bounded Queue, lazy file iteration). Default off — zero behavior change when unset. |
| `TRELIX_RETRIEVAL_RERANK_PROVIDER` | _(none)_ | Reranker to apply after fusion. One of: `cross_encoder`, `cohere`, `plaid`, `xtr` (**experimental**) |
| `TRELIX_RETRIEVAL_XTR_TOKENS` | `100` | Candidate token count for XTR reranker (10–1000). Only applies when `TRELIX_RETRIEVAL_RERANK_PROVIDER=xtr` |
| `TRELIX_RETRIEVAL_FLARE` | `false` | Enable FLARE re-retrieval — iteratively retrieves more context when confidence is low |
| `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` | `1` | Maximum FLARE iterations per query (min: 1, max: 3) |
| `TRELIX_RETRIEVAL_HYDE_FALLBACK` | `false` | Enable HyDE (Hypothetical Document Embeddings) fallback when standard retrieval returns weak results |
| `TRELIX_RETRIEVAL_FILE_SUMMARY_LEG` | `false` | Enable the file-summary retrieval leg — retrieves against LLM-generated file summaries in addition to raw chunks |
| `TRELIX_RETRIEVAL_PAGERANK_BOOST` | `false` | Enable PageRank-based symbol boosting — surfaces frequently referenced symbols higher in results |
| `TRELIX_RETRIEVAL_GRAPH_SEARCH` | `false` | Enable knowledge graph search leg — queries the code graph in addition to vector search |
| `TRELIX_RETRIEVAL_TELEMETRY` | `false` | Emit per-query telemetry (latency, hit counts, scores) to the configured telemetry sink |
| `TRELIX_FILE_SUMMARIES_ENABLED` | `false` | Generate LLM-powered file summaries at index time (requires a configured LLM provider) |

### LLM / Synthesis

| Variable | Default | Description |
|---|---|---|
| `TRELIX_LLM_PROVIDER` | `openai` | LLM provider used for answer synthesis. One of: `openai`, `azure`, `anthropic` |
| `TRELIX_LLM_OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model for synthesis |
| `TRELIX_LLM_AZURE_MODEL` | `gpt-4o` | Azure chat model deployment name |
| `ANTHROPIC_API_KEY` | _(none)_ | Anthropic API key — required when `TRELIX_LLM_PROVIDER=anthropic` |
| `TRELIX_RETRIEVAL_AGENTIC` | `false` | Enable the agentic ReAct loop — the LLM iteratively issues retrieval calls before producing a final answer |

### Storage

| Variable | Default | Description |
|---|---|---|
| `TRELIX_STORE_BACKEND` | `sqlite-vec` | Vector store backend. One of: `sqlite-vec`, `qdrant`, `lance` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL — required when backend is `qdrant` |
| `QDRANT_API_KEY` | _(none)_ | Qdrant API key — required for authenticated Qdrant Cloud instances |
| `QDRANT_COLLECTION` | `trelix` | Qdrant collection name |
| `QDRANT_PREFER_GRPC` | `false` | Use Qdrant's gRPC port (6334) instead of REST (6333) — lower latency, recommended for Qdrant Cloud |
| `QDRANT_TIMEOUT` | `10.0` | Client request timeout in seconds — raise for Cloud deployments with higher network latency |
| `TRELIX_STORE_BM25_READ_POOL_SIZE` | `0` | Number of read-only SQLite connections to pool for parallel `bm25_search()` calls. `0` disables pooling (default — identical to the pre-existing single-connection behavior). When set, `Retriever` automatically calls `Database.enable_bm25_read_pool()` at construction time. |

### Federation

| Variable | Default | Description |
|---|---|---|
| `TRELIX_FEDERATION_ENABLED` | `false` | Enable federated search across multiple indexed repositories |
| `TRELIX_FEDERATION_MAX_WORKERS` | `4` | Maximum number of parallel workers when querying federated repos |
| `TRELIX_FEDERATION_CONFIG` | `~/.config/trelix/repos.json` | Path to the federation registry JSON file |

### MCP Server

| Variable | Default | Description |
|---|---|---|
| `TRELIX_MCP_MAX_SUBSCRIBERS` | `1000` | Maximum number of concurrent resource subscriptions `trelix-mcp` will accept. Re-subscribing an existing `subscription_id` never counts as growth. Once at capacity, `subscribe_resource` returns a soft error (`{"subscribed": false, ...}`) instead of raising. |
| `TRELIX_MCP_SUBSCRIPTION_TTL_SECONDS` | `3600` | Time-to-live (seconds) for an inactive resource subscription before it is evicted from the `SubscriptionRegistry`. Expired subscriptions are swept lazily on the next registry access. |

---

## .env File Example

Copy this to `<repo-root>/.env` and fill in the values relevant to your setup. Lines beginning with `#` are comments and are ignored.

```dotenv
# =============================================================================
# Trelix v2.7.1 — complete .env example
# Copy to .env and fill in values. Never commit this file.
# =============================================================================

# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

# Provider: local | openai | azure | voyage | bge-code | nomic-code
TRELIX_EMBEDDER_PROVIDER=local

# OpenAI embeddings
# TRELIX_EMBEDDER_OPENAI_MODEL=text-embedding-3-small
# OPENAI_API_KEY=sk-...

# Azure embeddings
# TRELIX_EMBEDDER_AZURE_DEPLOYMENT=text-embedding-3-small
# AZURE_API_KEY=...
# AZURE_ENDPOINT=https://<your-resource>.openai.azure.com/

# Voyage embeddings (Matryoshka dimension: 256 | 512 | 1024 | 2048)
# TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=1024
# VOYAGE_API_KEY=...

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Multi-query expansion
TRELIX_RETRIEVAL_MULTI_QUERY=false
TRELIX_RETRIEVAL_MULTI_QUERY_COUNT=2

# Streaming indexing
TRELIX_INDEXER_STREAMING=false

# FLARE iterative re-retrieval (max retries: 1-3)
TRELIX_RETRIEVAL_FLARE=false
TRELIX_RETRIEVAL_FLARE_MAX_RETRIES=1

# HyDE fallback
TRELIX_RETRIEVAL_HYDE_FALLBACK=false

# Extra retrieval legs
TRELIX_RETRIEVAL_FILE_SUMMARY_LEG=false
TRELIX_RETRIEVAL_PAGERANK_BOOST=false
TRELIX_RETRIEVAL_GRAPH_SEARCH=false

# Telemetry
TRELIX_RETRIEVAL_TELEMETRY=false

# Generate LLM file summaries at index time (requires LLM provider)
TRELIX_FILE_SUMMARIES_ENABLED=false

# ---------------------------------------------------------------------------
# LLM / Synthesis
# ---------------------------------------------------------------------------

# Provider: openai | azure | anthropic
TRELIX_LLM_PROVIDER=openai

# OpenAI chat
TRELIX_LLM_OPENAI_MODEL=gpt-4o-mini
# OPENAI_API_KEY=sk-...  (shared with embedder if both use OpenAI)

# Azure chat
# TRELIX_LLM_AZURE_MODEL=gpt-4o

# Anthropic
# ANTHROPIC_API_KEY=sk-ant-...

# Agentic ReAct loop
TRELIX_RETRIEVAL_AGENTIC=false

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

# Backend: sqlite-vec | qdrant | lance
TRELIX_STORE_BACKEND=sqlite-vec

# Qdrant (required when backend=qdrant)
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=...
# QDRANT_COLLECTION=trelix
# QDRANT_PREFER_GRPC=false
# QDRANT_TIMEOUT=10.0

# Parallel read-only BM25 connections (0 = disabled, default)
# TRELIX_STORE_BM25_READ_POOL_SIZE=4

# ---------------------------------------------------------------------------
# Federation
# ---------------------------------------------------------------------------

TRELIX_FEDERATION_ENABLED=false
TRELIX_FEDERATION_MAX_WORKERS=4
# TRELIX_FEDERATION_CONFIG=~/.config/trelix/repos.json

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

# TRELIX_MCP_MAX_SUBSCRIBERS=1000
# TRELIX_MCP_SUBSCRIPTION_TTL_SECONDS=3600
```

---

## Per-Project Configuration

Trelix supports a per-project TOML config file at `.trelix/config.toml` inside any indexed repository. Settings in this file override the global defaults for that project only. Environment variables still take precedence over per-project config.

### Supported keys

```toml
# .trelix/config.toml

[embedder]
provider = "openai"                    # override global TRELIX_EMBEDDER_PROVIDER
openai_model = "text-embedding-3-large"

[retrieval]
multi_query = true
multi_query_count = 3
flare = false
flare_max_retries = 1
hyde_fallback = true
file_summary_leg = true
pagerank_boost = true
graph_search = false
telemetry = false
file_summaries_enabled = true
agentic = true

[llm]
provider = "openai"
openai_model = "gpt-4o"
azure_model = "gpt-4o"

[store]
backend = "sqlite-vec"
qdrant_url = "http://localhost:6333"
```

### Resolution order (most specific wins)

```
Environment variable
  > .trelix/config.toml (per-project)
    > .env (repo root)
      > built-in defaults
```

### Creating the file

```bash
mkdir -p .trelix
touch .trelix/config.toml
```

Add `.trelix/config.toml` to version control so all contributors share the same project-level defaults. Do **not** put secrets in this file — use environment variables or `.env` (which should be git-ignored) for those.

---

## MCP Server

trelix ships a Model Context Protocol server (`trelix-mcp`) that exposes indexed repositories as MCP resources and tools, allowing MCP-compatible clients (e.g. Claude Desktop) to query trelix directly.

### Resource subscriptions (v2.5.0)

trelix-mcp v2.5.0 advertises `resources.subscribe = true` in its server capabilities and exposes two new tools:

| Tool | Parameters | Description |
|---|---|---|
| `subscribe_resource` | `uri`, `subscription_id` | Subscribe to change notifications for a `trelix://` resource URI |
| `unsubscribe_resource` | `subscription_id` | Cancel an active subscription |

**URI scheme:** `trelix://repo/{repo_path}/manifest`

**Wire protocol:**
1. Client calls `subscribe_resource(uri, subscription_id)` — the server registers the subscription.
2. When a watched file changes, trelix-mcp emits a `notifications/resources/updated` notification (URI only, with `subscriptionId` in `params._meta`).
3. Client calls `resources/read` to fetch the updated content.

Subscriptions are held in-memory (not persisted across server restarts). The subscription registry is thread-safe.
