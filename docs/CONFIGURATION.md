# Trelix Configuration Reference — v2.11.1

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
| `TRELIX_RETRIEVAL_PATH_FILTER_OVERSAMPLE` | `3` (min: `1`) | **Advanced/internal tuning knob.** When a `SubQuery.path_filter` is set, the vector leg over-fetches by this factor before post-filtering results by path prefix and truncating back to `k` — protects recall against the filter discarding raw ANN hits. There is currently no CLI, REST, or MCP parameter to set `path_filter` itself; it is only set programmatically. |
| `TRELIX_RETRIEVAL_CONTEXT_BUDGET_PER_SOURCE` | `false` | When enabled, the context-assembly token budget is split proportionally across each retrieval leg's result count instead of one shared pool, so a single noisy leg cannot crowd out the others. `false` reproduces the pre-existing single-pool greedy-pack behavior byte-for-byte. |
| `TRELIX_INDEXER_STREAMING` | `false` | Enable generator-based streaming indexing pipeline (bounded Queue, lazy file iteration). Default off — zero behavior change when unset. |
| `TRELIX_RETRIEVAL_RERANK_PROVIDER` | _(none)_ | Reranker to apply after fusion. One of: `cross_encoder`, `cohere`, `plaid`, `xtr` (**experimental**) |
| `TRELIX_RETRIEVAL_XTR_TOKENS` | `100` | Candidate token count for XTR reranker (10–1000). Only applies when `TRELIX_RETRIEVAL_RERANK_PROVIDER=xtr` |
| `TRELIX_RETRIEVAL_FLARE` | `false` | Enable FLARE re-retrieval — iteratively retrieves more context when confidence is low |
| `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` | `1` | Maximum FLARE iterations per query (min: 1, max: 3) |
| `TRELIX_RETRIEVAL_HYDE_FALLBACK` | `false` | Enable HyDE (Hypothetical Document Embeddings) fallback when standard retrieval returns weak results |
| `TRELIX_RETRIEVAL_FILE_SUMMARY_LEG` | `false` | Enable the file-summary retrieval leg — retrieves against LLM-generated file summaries in addition to raw chunks |
| `TRELIX_RETRIEVAL_PAGERANK_BOOST` | `false` | Enable PageRank-based symbol boosting — surfaces frequently referenced symbols higher in results |
| `TRELIX_RETRIEVAL_PAGERANK_PERSONALIZATION` | `false` | Enable Personalized PageRank: teleport mass is weighted toward symbols with a `generic_edges` connector-artifact/ticket relationship (uniform `1/\|T\|` over that seed set) instead of uniform teleportation across every node. Applies to both `rank_by_pagerank()` (query-time) and `compute_pagerank()` (index-time). Falls back to plain uniform-teleportation PageRank when disabled or when the seed set is empty — zero behavior change unless opted in. Interaction risk with `TRELIX_RETRIEVAL_PAGERANK_BOOST`: on a repo where only a few symbols have ever been referenced by a ticket/artifact, enabling both together can invert `get_top_central_symbols()`'s ranking — if boost results look off with both enabled, try disabling personalization first to isolate which flag is driving the change. |
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

### Agentic ReAct Loop — Persistent Sessions

The agentic loop (`trelix ask --agentic` / `TRELIX_RETRIEVAL_AGENTIC=true`) persists its turn history to `agent_sessions`/`agent_turns` tables in the repo's `.trelix/index.db`, keyed by a client-supplied or auto-generated `session_id`. Resume a session with `trelix ask --session <id>` (implies `--agentic`) or the MCP `ask_agent` tool's `session_id` argument.

| Variable | Default | Description |
|---|---|---|
| `TRELIX_RETRIEVAL_AGENT_MAX_TURNS` | `8` | Maximum ReAct turns per `ask_agent`/`--agentic` call (1–20) |
| `TRELIX_RETRIEVAL_AGENT_TOKEN_BUDGET` | `6000` | Token budget for `HistoryCompressor` when trimming turn history (minimum 1000) |
| `TRELIX_RETRIEVAL_AGENT_SESSION_MAX_AGE_SECONDS` | `604800` (7 days) | Idle time before a persisted agent session is auto-evicted. `0` disables eviction entirely. Use `trelix agent sessions clear <id>` (or the `agent_clear_session` MCP tool) to remove a session explicitly |

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
| `TRELIX_FEDERATION_MAX_WORKERS` | `4` | Maximum number of parallel workers when querying federated repos (1–16) |
| `TRELIX_FEDERATION_MAX_REPOS` | `50` | Maximum number of registered repos actually queried per federated search call (1–500). Registered repos beyond this cap are skipped (reported via `repos_skipped` in the MCP `federation_search_all` response); prevents an unbounded `federation_add_repo` loop from making every subsequent query scale linearly |

There is no environment variable for the federation registry file path. The registry JSON file location defaults to `~/.config/trelix/repos.json` and can be overridden per-call via the `--config` CLI option (`trelix search-all --config`, `trelix federation add/list/remove --config`) or the `config_path` argument on the corresponding MCP tools. For security, MCP callers may only point `config_path` at `~/.config/trelix/` or `<mcp-server-cwd>/.trelix/` — paths outside those roots are rejected.

### Git ticket linking

Configuration for [`trelix link-tickets`](CLI_REFERENCE.md#trelix-link-tickets), which walks git history to link code symbols to external ticket references found in commit messages. Off by default — requires the repo to actually be a git checkout, and is a separate, slower pass from the main index pipeline (invoked only via `trelix link-tickets`, never automatically from indexing).

| Variable | Default | Description |
|---|---|---|
| `TRELIX_GIT_LINKER_ENABLED` | `false` | Enable git-history ticket linking. Set to `true` automatically by `trelix link-tickets`; not something you typically set directly. |
| `TRELIX_GIT_LINKER_TICKET_PATTERN` | `[A-Z]+-\d+` | Regex for matching ticket IDs in commit messages. The default matches Jira-style tickets (`PROJ-123`); override for other conventions (GitHub `#123`, Linear `ENG-123`). |
| `TRELIX_GIT_LINKER_MAX_COMMITS` | `5000` (min: `1`) | Maximum number of commits to walk. Bounds cost on repos with 100k+ commit histories. |
| `TRELIX_GIT_LINKER_SINCE` | _(none)_ | Only walk commits after this date, e.g. `"90 days ago"`. Passed straight through to `git log --since`. |

### Connector credentials (Jira / TestRail / Xray / Linear)

Configuration for [`trelix connector sync`](CLI_REFERENCE.md#trelix-connector-sync), which fetches artifacts from an external system and writes them to trelix's `artifacts` table. Jira and TestRail use HTTP Basic auth; Xray Cloud exchanges a client_id/client_secret for a short-lived bearer JWT; Linear uses a personal API key sent directly in the `Authorization` header with no `Bearer` prefix. All required variables per connector must be set — missing any of them fails config validation before any HTTP call is made.

| Variable | Default | Description |
|---|---|---|
| `TRELIX_JIRA_BASE_URL` | _(none, required)_ | Base URL of the Jira Cloud instance, e.g. `https://acme.atlassian.net` |
| `TRELIX_JIRA_EMAIL` | _(none, required)_ | Email address used for HTTP Basic auth against the Jira REST API |
| `TRELIX_JIRA_API_TOKEN` | _(none, required)_ | Jira API token (paired with `TRELIX_JIRA_EMAIL` for Basic auth) |
| `TRELIX_JIRA_PROJECT_KEY` | _(none, required)_ | Jira project key to sync tickets from |
| `TRELIX_JIRA_PAGE_SIZE` | `100` (max `100`) | Page size for Jira API pagination |
| `TRELIX_TESTRAIL_BASE_URL` | _(none, required)_ | Base URL of the TestRail instance, e.g. `https://acme.testrail.io` |
| `TRELIX_TESTRAIL_USERNAME` | _(none, required)_ | Username used for HTTP Basic auth against the TestRail REST API |
| `TRELIX_TESTRAIL_API_KEY` | _(none, required)_ | TestRail API key (paired with `TRELIX_TESTRAIL_USERNAME` for Basic auth) |
| `TRELIX_TESTRAIL_PROJECT_ID` | _(none, required)_ | TestRail project ID to sync test cases from |
| `TRELIX_TESTRAIL_PAGE_SIZE` | `250` (max `250` — TestRail's own API ceiling) | Page size for TestRail API pagination |
| `TRELIX_XRAY_CLIENT_ID` | _(none, required)_ | Xray Cloud client ID, issued by a Jira admin in Xray's global settings (distinct from a user's own Jira API token) |
| `TRELIX_XRAY_CLIENT_SECRET` | _(none, required)_ | Xray Cloud client secret, paired with `TRELIX_XRAY_CLIENT_ID` and exchanged for a short-lived bearer JWT |
| `TRELIX_XRAY_PROJECT_KEY` | _(none, required)_ | Jira project key whose tests to sync (Xray Cloud tests are Jira issues under the hood) |
| `TRELIX_XRAY_JIRA_BASE_URL` | _(none, required)_ | Base URL of the Jira Cloud instance backing this Xray project, e.g. `https://acme.atlassian.net` |
| `TRELIX_XRAY_PAGE_SIZE` | `100` (max `100`) | Page size for Xray's GraphQL `getTests` pagination |
| `TRELIX_LINEAR_API_KEY` | _(none, required)_ | Linear personal API key — sent verbatim as `Authorization: <key>` (no `Bearer` prefix) |
| `TRELIX_LINEAR_TEAM_KEY` | _(none, required)_ | Linear team key to scope issue sync to, e.g. `ENG` |
| `TRELIX_LINEAR_PAGE_SIZE` | `100` (max `100`) | Page size for Linear's cursor-paginated `issues` query — not a confirmed Linear platform ceiling, chosen to stay well under its GraphQL query-complexity cap |

### REST API

| Variable | Default | Description |
|---|---|---|
| `TRELIX_API_AUTH_TOKEN` | _(none)_ | Shared secret for `trelix serve`'s REST API. Opt-in: unset (the default) leaves every route open, matching the same "off by default" pattern as `TRELIX_OTEL_ENABLED` and `TRELIX_TELEMETRY_ENABLED`. When set, every route except `GET /health` requires a matching `X-Trelix-Api-Key` header (checked with a constant-time comparison) — see [USER_GUIDE.md § API Quick Reference](USER_GUIDE.md#14-api-quick-reference). |

### MCP Server

| Variable | Default | Description |
|---|---|---|
| `TRELIX_MCP_MAX_SUBSCRIBERS` | `1000` | Maximum number of concurrent resource subscriptions `trelix-mcp` will accept. Re-subscribing an existing `subscription_id` never counts as growth. Once at capacity, `subscribe_resource` returns a soft error (`{"subscribed": false, ...}`) instead of raising. |
| `TRELIX_MCP_SUBSCRIPTION_TTL_SECONDS` | `3600` | Time-to-live (seconds) for an inactive resource subscription before it is evicted from the `SubscriptionRegistry`. Expired subscriptions are swept lazily on the next registry access. |

### Observability (OpenTelemetry)

Requires `pip install trelix[otel]`. See [OBSERVABILITY.md](OBSERVABILITY.md) for the full span/attribute reference.

| Variable | Default | Description |
|---|---|---|
| `TRELIX_OTEL_ENABLED` | `false` | Emit one OpenTelemetry span per retrieval leg (vector/BM25/grep/sparse/sub-chunk/file-summary) plus pipeline-stage spans (planner/fusion/expansion/rerank/pagerank/assembly). Zero import cost and zero behavior change when disabled. |
| `OTEL_SERVICE_NAME` | `trelix` | Service name attached to the installed `TracerProvider`'s resource attributes. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | _(none)_ | OTLP collector endpoint. If unset, spans are still created but have nowhere to export to unless a host application configures its own exporter/processor before trelix runs. |

---

## .env File Example

Copy this to `<repo-root>/.env` and fill in the values relevant to your setup. Lines beginning with `#` are comments and are ignored.

```dotenv
# =============================================================================
# Trelix v2.11.1 — complete .env example
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
TRELIX_RETRIEVAL_PAGERANK_PERSONALIZATION=false
TRELIX_RETRIEVAL_GRAPH_SEARCH=false

# Telemetry
TRELIX_RETRIEVAL_TELEMETRY=false

# Generate LLM file summaries at index time (requires LLM provider)
TRELIX_FILE_SUMMARIES_ENABLED=false

# Advanced retrieval tuning (internal knobs — no CLI/REST/MCP param yet)
# TRELIX_RETRIEVAL_PATH_FILTER_OVERSAMPLE=3
# TRELIX_RETRIEVAL_CONTEXT_BUDGET_PER_SOURCE=false

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

# Agentic ReAct loop — persistent session tuning
# TRELIX_RETRIEVAL_AGENT_MAX_TURNS=8
# TRELIX_RETRIEVAL_AGENT_TOKEN_BUDGET=6000
# TRELIX_RETRIEVAL_AGENT_SESSION_MAX_AGE_SECONDS=604800

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
# TRELIX_FEDERATION_MAX_REPOS=50

# Federation registry file path has no env var override — use --config (CLI)
# or config_path (MCP tools) instead. Defaults to ~/.config/trelix/repos.json.

# ---------------------------------------------------------------------------
# Git ticket linking (trelix link-tickets)
# ---------------------------------------------------------------------------

# TRELIX_GIT_LINKER_ENABLED=false
# TRELIX_GIT_LINKER_TICKET_PATTERN=[A-Z]+-\d+
# TRELIX_GIT_LINKER_MAX_COMMITS=5000
# TRELIX_GIT_LINKER_SINCE=90 days ago

# ---------------------------------------------------------------------------
# Connector credentials (trelix connector sync)
# ---------------------------------------------------------------------------

# Jira Cloud (base_url/email/api_token/project_key all required to sync)
# TRELIX_JIRA_BASE_URL=https://acme.atlassian.net
# TRELIX_JIRA_EMAIL=bot@acme.com
# TRELIX_JIRA_API_TOKEN=...
# TRELIX_JIRA_PROJECT_KEY=PROJ
# TRELIX_JIRA_PAGE_SIZE=100

# TestRail (base_url/username/api_key/project_id all required to sync)
# TRELIX_TESTRAIL_BASE_URL=https://acme.testrail.io
# TRELIX_TESTRAIL_USERNAME=bot@acme.com
# TRELIX_TESTRAIL_API_KEY=...
# TRELIX_TESTRAIL_PROJECT_ID=7
# TRELIX_TESTRAIL_PAGE_SIZE=250

# Xray Cloud (client_id/client_secret/project_key/jira_base_url all required to sync)
# TRELIX_XRAY_CLIENT_ID=...
# TRELIX_XRAY_CLIENT_SECRET=...
# TRELIX_XRAY_PROJECT_KEY=PROJ
# TRELIX_XRAY_JIRA_BASE_URL=https://acme.atlassian.net
# TRELIX_XRAY_PAGE_SIZE=100

# Linear (api_key/team_key both required to sync)
# TRELIX_LINEAR_API_KEY=...
# TRELIX_LINEAR_TEAM_KEY=ENG
# TRELIX_LINEAR_PAGE_SIZE=100

# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

# Opt-in auth for `trelix serve` — unset means every route stays open
# TRELIX_API_AUTH_TOKEN=...

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
