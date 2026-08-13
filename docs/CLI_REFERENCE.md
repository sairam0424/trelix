# trelix CLI Reference

**Version:** 3.0.0  
**Last updated:** 2026-08-03

trelix is a fast, hybrid code-search and synthesis tool. The CLI wraps every
capability of the library — indexing, retrieval, analysis, federation, watching
and more — with Rich terminal output by default and machine-parseable `--json`
on most commands.

---

## Table of Contents

1. [Global flags](#global-flags)
2. [Environment variables](#environment-variables)
3. [Exit codes](#exit-codes)
4. [Embedding providers](#embedding-providers)
5. [Commands](#commands)
   - [index](#trelix-index)
   - [search](#trelix-search)
   - [ask](#trelix-ask)
   - [query](#trelix-query)
   - [call-graph](#trelix-call-graph)
   - [stats](#trelix-stats)
   - [update-index](#trelix-update-index)
   - [migrate-vectors](#trelix-migrate-vectors)
   - [watch](#trelix-watch)
   - [watch-all](#trelix-watch-all)
   - [serve](#trelix-serve)
   - [graph](#trelix-graph)
   - [telemetry](#trelix-telemetry)
   - [eval](#trelix-eval)
   - [eval-synthesis](#trelix-eval-synthesis)
   - [taint](#trelix-taint)
   - [review](#trelix-review)
   - [link-tickets](#trelix-link-tickets)
   - [connector sync](#trelix-connector-sync)
   - [link-artifacts](#trelix-link-artifacts)
   - [search-all](#trelix-search-all)
   - [federation add](#trelix-federation-add)
   - [federation list](#trelix-federation-list)
   - [federation remove](#trelix-federation-remove)
   - [agent sessions list](#trelix-agent-sessions-list)
   - [agent sessions show](#trelix-agent-sessions-show)
   - [agent sessions clear](#trelix-agent-sessions-clear)
   - [audit list](#trelix-audit-list)
   - [audit verify](#trelix-audit-verify)
   - [audit export](#trelix-audit-export)

---

## Global flags

These flags are processed before any subcommand.

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--version` | `-V` | flag | Print the installed trelix version and exit. |
| `--help` | | flag | Show help text for the command or subcommand and exit. |

**Examples**

```bash
trelix --version        # trelix 3.0.0
trelix --help           # top-level help
trelix index --help     # help for the index command
```

---

## Environment variables

trelix uses [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
throughout. Every config value can be set via environment variable or a `.env`
file in the current working directory. The most important variables are listed
below; less common ones follow the same `TRELIX_<SECTION>_<FIELD>` pattern.

### API keys

| Variable | Required for |
|----------|-------------|
| `OPENAI_API_KEY` | `--provider openai`; LLM synthesis with `trelix ask` |
| `AZURE_API_KEY` | `--provider azure` |
| `AZURE_ENDPOINT` | `--provider azure` |
| `AZURE_API_VERSION` | Azure API version (default: `2025-04-01-preview`) |
| `AZURE_EMBEDDINGS_MODEL` | Azure embeddings deployment name |
| `AZURE_CHAT_MODEL` | Azure chat deployment name (default: `gpt-4o`) |
| `VOYAGE_API_KEY` | `--provider voyage` |
| `ANTHROPIC_API_KEY` | `TRELIX_LLM_PROVIDER=anthropic` |
| `AWS_ACCESS_KEY_ID` | `--provider bedrock-titan` or `bedrock-cohere` |
| `AWS_SECRET_ACCESS_KEY` | AWS Bedrock providers |
| `AWS_REGION` | AWS region (default: `us-east-1`) |
| `AWS_PROFILE` | AWS named profile (alternative to key/secret) |
| `COHERE_API_KEY` | Cohere reranker |
| `GITHUB_TOKEN` | `trelix review --pr` and `--post-comments` |
| `QDRANT_API_KEY` | Qdrant cloud instances |

### Embedding and retrieval tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `TRELIX_EMBEDDER_PROVIDER` | `local` | Default embedding provider (overridden per command by `--provider`) |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI chat model for synthesis |
| `TRELIX_LLM_PROVIDER` | `openai` | LLM backend for synthesis: `openai` \| `azure` \| `anthropic` \| `bedrock` \| `vertex` \| `litellm` |
| `TRELIX_RETRIEVAL_RERANK_PROVIDER` | `cohere` | Reranker: `cohere` \| `cross_encoder` \| `plaid` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `QDRANT_COLLECTION` | `trelix` | Qdrant collection name |
| `TRELIX_STORE_BACKEND` | `sqlite` | Vector store backend: `sqlite` \| `qdrant` \| `lance` |
| `TRELIX_STORE_HNSW` | `true` | Enable HNSW index (sqlite-vec ≥ 0.1.6) |
| `TRELIX_RETRIEVAL_MULTI_QUERY` | `false` | Multi-query expansion before retrieval |
| `TRELIX_RETRIEVAL_MULTI_QUERY_COUNT` | `2` | Query variant count for multi-query expansion (1–4) |
| `TRELIX_RETRIEVAL_SPARSE` | `false` | Enable SPLADE-Code sparse retrieval leg |
| `TRELIX_RETRIEVAL_AGENTIC` | `false` | Enable multi-turn ReAct loop (also set by `--agentic`) |
| `TRELIX_RETRIEVAL_FLARE` | `false` | Enable FLARE confidence-gated re-retrieval |
| `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` | `1` | Max FLARE retries (range: 1–3). Replaces the old `TRELIX_RETRIEVAL_FLARE_MAX_ITER`, which is **deprecated but still honoured** as of v3.0.0 — it remains in the field's `AliasChoices` and setting it logs a deprecation warning rather than being ignored. Values > 3 raise `ValidationError` at startup. |
| `TRELIX_RETRIEVAL_PAGERANK_BOOST` | `false` | Boost results by PageRank symbol importance |
| `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING` | `true` | Apply per-language RRF score multipliers |
| `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_<LANG>` | varies | Per-language override, e.g. `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN=0.1` |
| `TRELIX_RETRIEVAL_LEG_WEIGHT_<LEG>` | `1.0` | Per-leg RRF score multiplier applied during fusion. `<LEG>` is one of `VECTOR`, `BM25`, `GREP`, `SUMMARY`, `SUB_CHUNK`, `SPARSE`, e.g. `TRELIX_RETRIEVAL_LEG_WEIGHT_BM25=0.7` |
| `TRELIX_RETRIEVAL_DECLARATION_BOOST` | `false` | Boost BM25 matches on a symbol's `name`/`qualified_name` over incidental `docstring`/`body` mentions |
| `TRELIX_RETRIEVAL_DECLARATION_BOOST_WEIGHT` | `1.0` | Declaration-boost multiplier (range: 1.0–10.0) |

### Indexing and chunking

| Variable | Default | Description |
|----------|---------|-------------|
| `TRELIX_PARSE_WORKERS` | `4` | Parallel parse workers during `trelix index` |
| `TRELIX_CHUNKER_MULTI_GRANULARITY` | `false` | Index sub-symbol blocks and statements (MGS3) |
| `TRELIX_PARSER_DATAFLOW` | `false` | Extract def-use chains during parsing |
| `TRELIX_PARSER_TAINT` | `false` | Enable taint-flow tracking during parsing |
| `TRELIX_FILE_SUMMARIES_ENABLED` | `false` | Generate LLM file-level summaries at index time (RAPTOR-style) |
| `TRELIX_TELEMETRY_ENABLED` | `false` | Record every `retrieve()` call to `query_telemetry` table |

### Federation

| Variable | Default | Description |
|----------|---------|-------------|
| `TRELIX_FEDERATION_ENABLED` | `false` | Enable multi-repo federated retrieval |
| `TRELIX_FEDERATION_MAX_WORKERS` | `4` | Max parallel workers for federated search (range: 1–16) |
| `TRELIX_FEDERATION_MAX_REPOS` | `50` | Max repos actually queried per `federation_search_all` MCP call, and max repos `federation_add_repo` will accept (range: 1–500) |

**MCP security note:** The four federation MCP tools (`federation_list_repos`, `federation_add_repo`, `federation_remove_repo`, `federation_search_all`) confine any caller-supplied `config_path` argument to `~/.config/trelix/` or `<mcp-server-cwd>/.trelix/`, rejecting paths outside both roots. Prevents prompt-injected or adversarial clients from pointing registry I/O at arbitrary filesystem locations.

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error — configuration invalid, index not found, I/O failure, API error, or user cancelled with Ctrl+C |

---

## Embedding providers

The `--provider` flag accepted by most commands controls which embedding model
is used. The same value must be consistent between `index` and all retrieval
commands on the same repository.

| Value | Model | API key required |
|-------|-------|-----------------|
| `local` | `sentence-transformers/all-MiniLM-L6-v2` | No |
| `local-code` | `Salesforce/SFR-Embedding-Code-2B_R` | No (large download) |
| `bge-code` | `BAAI/bge-code-v1` | No (`pip install trelix[bge-code]`) |
| `nomic-code` | `nomic-ai/CodeRankEmbed` | No (`pip install trelix[local]`) |
| `openai` | `text-embedding-3-large` (3072-dim) | `OPENAI_API_KEY` |
| `azure` | deployment via `AZURE_EMBEDDINGS_MODEL` | `AZURE_API_KEY` + `AZURE_ENDPOINT` |
| `voyage` | `voyage-code-3` (1024-dim) | `VOYAGE_API_KEY` |
| `bedrock-titan` | `amazon.titan-embed-text-v2:0` (1024-dim) | AWS credentials |
| `bedrock-cohere` | `cohere.embed-english-v3` (1024-dim) | AWS credentials |

**Important:** Switching provider after indexing changes the embedding dimension.
Run `trelix migrate-vectors <repo> --reset` and then re-index.

---

## Commands

---

### `trelix index`

#### Synopsis

```
trelix index <repo_path> [--provider PROVIDER] [--verbose]
```

#### Description

Scans `<repo_path>`, parses source files with tree-sitter, chunks and embeds
every symbol, and stores the result in `<repo_path>/.trelix/index.db`. On
subsequent runs only changed files are re-indexed (incremental mode). Prints
a summary table with file counts, symbol count, chunk count, and elapsed time.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--provider` | string | `local` | Embedding provider. See [Embedding providers](#embedding-providers). |
| `--verbose`, `-v` | flag | `false` | Show DEBUG-level log output from the indexer and embedder. |

#### Examples

```bash
# Index with the default local embedder
trelix index .

# Index with OpenAI (requires OPENAI_API_KEY)
trelix index /path/to/myrepo --provider openai

# Debug a slow index run
trelix index . --verbose
```

#### Notes

- The `.trelix/` directory created inside the repo is auto-gitignored
  (trelix writes a `.trelix/.gitignore` containing `*`).
- Parallel parse workers default to 4. Override with
  `TRELIX_PARSE_WORKERS=<n>`.
- Languages indexed by default: Python, JavaScript, TypeScript, TSX, Go, Rust,
  Java, Kotlin, Ruby, C/C++, C#, Razor, Markdown, JSON, YAML, TOML, HTML, CSS.
  Files larger than 500 KB are skipped.

---

### `trelix search`

#### Synopsis

```
trelix search <repo_path> <query> [--provider PROVIDER] [--json]
```

#### Description

Runs a hybrid (vector + BM25 + grep) search over the indexed repository and
displays ranked results. Output is a Rich table by default or a JSON object
with `--json`.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--provider` | string | `local` | Embedding provider. Must match the one used at index time. |
| `--json` | flag | `false` | Print results as JSON instead of a Rich table. |

#### Examples

```bash
# Basic search
trelix search . "authentication middleware"

# JSON output for scripting
trelix search . "rate limiter" --json

# With OpenAI embeddings
trelix search /my/repo "database connection pool" --provider openai
```

#### JSON output schema

```json
{
  "status": "ok",
  "results": [
    {
      "file": "src/auth.py",
      "symbol": "check_token",
      "lines": "42-67",
      "score": 0.9231
    }
  ]
}
```

#### Notes

- `trelix search` disables the reranker for its own invocation (it constructs
  `RetrievalConfig(rerank=False)`), so `TRELIX_RETRIEVAL_RERANK*` affects `ask`, the MCP
  tools, the REST API, and the Python API rather than this command. Elsewhere reranking
  is on by default; turn it off with `TRELIX_RETRIEVAL_RERANK=false` and choose the
  backend with `TRELIX_RETRIEVAL_RERANK_PROVIDER=cohere|cross_encoder|plaid|xtr`.
- The `--provider` flag affects the query embedding, not index scanning.
  Always use the same provider that was used for `trelix index`.

---

### `trelix ask`

#### Synopsis

```
trelix ask <repo_path> <question> [--provider PROVIDER] [--agentic] [--session ID]
```

#### Description

Retrieves relevant code context and synthesizes a natural-language answer
using an LLM. With `--provider local` (no LLM key), trelix prints the
retrieved context text instead of a synthesized answer. Streaming output is
used when an LLM is available.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--provider` | string | `local` | Embedding provider for retrieval. |
| `--agentic` | flag | `false` | Enable a multi-turn ReAct loop: the agent can issue multiple sub-queries, observe results, and refine its answer. Requires an LLM API key. |
| `--session` | string | — | Resume a persisted agent session by ID. Implies `--agentic`. See [`trelix agent sessions list`](#trelix-agent-sessions-list). |

#### Examples

```bash
# Context-only (no LLM)
trelix ask . "how does the token refresh flow work"

# Full LLM synthesis
OPENAI_API_KEY=sk-... trelix ask . "explain the caching strategy"

# Agentic mode for complex questions
OPENAI_API_KEY=sk-... trelix ask . "trace the data flow from API request to database" --agentic
```

#### Notes

- `--agentic` sets `TRELIX_RETRIEVAL_AGENTIC=true` for this invocation only.
- With `--provider local` and no LLM key, the command prints the assembled
  context text, which is useful for debugging retrieval quality.
- FLARE iterative retrieval can be enabled globally with
  `TRELIX_RETRIEVAL_FLARE=true`.

---

### `trelix query`

#### Synopsis

```
trelix query <repo_path> <question> [--provider PROVIDER]
```

#### Description

Performs retrieval and prints a human-readable Rich terminal table of matching
symbols. Unlike `trelix ask`, this command performs no LLM synthesis and has
no `--json` flag — it is designed for interactive terminal exploration.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--provider` | string | `local` | Embedding provider for retrieval. |

#### Examples

```bash
# Interactive symbol lookup
trelix query . "connection pool initialization"

# With a specific provider
trelix query /my/repo "error handling patterns" --provider voyage
```

#### Notes

- For machine-readable output use `trelix search ... --json` instead.
- The header line shows the number of results, total tokens, and elapsed time.

---

### `trelix call-graph`

#### Synopsis

```
trelix call-graph <repo_path> <symbol> [--direction DIRECTION] [--provider PROVIDER]
```

#### Description

Displays the call graph and import edges for a given symbol or module path.
Prints three tables: callers (who calls this symbol), callees (what this
symbol calls), and importers (who imports this module).

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--direction` | `-d` | string | `all` | Limit output: `callers` \| `callees` \| `importers` \| `all` |
| `--provider` | | string | `local` | Embedding provider. |

#### Examples

```bash
# Show full call graph for a function
trelix call-graph . "authenticate_user"

# Show only callers of a symbol
trelix call-graph . "send_email" --direction callers

# Show only callees
trelix call-graph . "process_payment" --direction callees

# Show only importers of a module
trelix call-graph . "trelix.retrieval.retriever" --direction importers
```

#### Notes

- The `symbol` argument can be a simple function name or a qualified module
  path (e.g., `pkg.module.ClassName`).
- Graph edges are built during indexing. Re-index if the graph looks stale.

---

### `trelix stats`

#### Synopsis

```
trelix stats <repo_path>
```

#### Description

Reads the SQLite index at `<repo_path>/.trelix/index.db` and prints a summary
table showing the number of indexed files, symbols, chunks, and database size
on disk.

#### Options

None.

#### Examples

```bash
trelix stats .
trelix stats /path/to/large-repo
```

#### Notes

- Exits with code 1 if no index exists. Run `trelix index <repo_path>` first.

---

### `trelix update-index`

#### Synopsis

```
trelix update-index <repo_path> <file_path> [--provider PROVIDER]
```

#### Description

Re-indexes a single file without re-scanning the entire repository. Useful
after editing one file during a watch-less workflow. Prints JSON with the
result of the incremental update.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--provider` | string | `local` | Embedding provider. Must match the one used at index time. |

#### Examples

```bash
# Re-index a single file
trelix update-index . src/auth/middleware.py

# With a specific provider
trelix update-index /my/repo src/core/db.go --provider openai
```

#### Notes

- `<file_path>` may be absolute or relative to `<repo_path>`.
- Always outputs JSON to stdout (not controlled by a flag). Redirect stderr
  for error messages.

---

### `trelix migrate-vectors`

#### Synopsis

```
trelix migrate-vectors <repo_path> [--to TARGET] [--url URL] [--collection NAME] [--api-key KEY] [--reset]
```

#### Description

Either migrates all embeddings from the local SQLite store to Qdrant
(`--to qdrant`), or clears the local embedding store and dimension metadata
so the next `trelix index` run starts fresh (`--reset`).

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--to` | string | `qdrant` | Target backend. Only `qdrant` is supported. |
| `--url` | string | `http://localhost:6333` | Qdrant server URL. |
| `--collection` | string | `trelix` | Qdrant collection name. |
| `--api-key` | string | `""` | Qdrant API key (for Qdrant Cloud). |
| `--reset` | flag | `false` | Clear all stored embeddings and dimension metadata from the SQLite index. Use this when switching embedding providers. Does NOT migrate to Qdrant — it resets the local store only. |

#### Examples

```bash
# Migrate to a local Qdrant instance
trelix migrate-vectors . --to qdrant

# Migrate to Qdrant Cloud
trelix migrate-vectors . \
  --to qdrant \
  --url https://abc.qdrant.io \
  --api-key $QDRANT_API_KEY \
  --collection myproject

# Reset after switching from openai to local provider
trelix migrate-vectors . --reset
trelix index . --provider local
```

#### Notes

- `--reset` and `--to qdrant` are mutually exclusive. Use `--reset` alone
  when switching embedding providers on the same repo.
- After `--reset`, run `trelix index <repo_path>` to rebuild the embeddings.
- `--to qdrant` requires `sqlite-vec` to be installed. The command exits with
  an error if the extension cannot be loaded.

---

### `trelix watch`

#### Synopsis

```
trelix watch <repo_path> [--provider PROVIDER]
```

#### Description

Performs an initial full index and then watches the repository for file changes
using `watchfiles`. Changed files are re-indexed automatically. Press Ctrl+C
to stop. Useful during active development on a single repository.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--provider` | string | `local` | Embedding provider. |

#### Examples

```bash
# Watch the current directory
trelix watch .

# Watch with a specific provider
trelix watch /my/repo --provider openai
```

#### Notes

- The command exits with code 1 if the initial index fails.
- For watching multiple repositories simultaneously, use
  [`trelix watch-all`](#trelix-watch-all).
- Requires `watchfiles` package. Install with `pip install trelix[watch]` or
  `pip install watchfiles`.

---

### `trelix watch-all`

**New in v2.4.0**

#### Synopsis

```
trelix watch-all [--config PATH]
```

#### Description

Watches all repositories registered in the federation registry simultaneously.
Uses a single `watchfiles.awatch()` call across all repo paths. A hash guard
prevents re-index cascade loops. Deleted files are removed from the SQLite
index and vector store. Prints per-repo stats on exit.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | string | `~/.config/trelix/repos.json` | Path to the federation registry JSON file. |

#### Examples

```bash
# Watch all registered repos (uses default registry)
trelix watch-all

# Watch repos from a custom registry file
trelix watch-all --config /projects/.trelix/repos.json
```

#### Notes

- Register repos first with `trelix federation add`.
- If no repos are registered, the command exits with code 0 and prints a hint.
- Graceful shutdown on Ctrl+C or SIGTERM. Exit summary shows total files
  re-indexed and files skipped (unchanged).

---

### `trelix serve`

#### Synopsis

```
trelix serve <repo_path> [--host HOST] [--port PORT]
```

#### Description

Starts a FastAPI REST server exposing trelix search and synthesis endpoints
over HTTP. Useful for integrating trelix into IDEs, notebooks, or external
tooling without using the MCP server.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--host` | string | `127.0.0.1` | Host interface to bind. Use `0.0.0.0` to expose on all interfaces. |
| `--port` | integer | `8765` | TCP port to listen on. |

#### Examples

```bash
# Start the API server on the default port
trelix serve .

# Bind on all interfaces, custom port
trelix serve /my/repo --host 0.0.0.0 --port 9000
```

#### Notes

- Requires the `serve` extra: `pip install 'trelix[serve]'` (installs FastAPI
  and uvicorn).
- The API is undocumented in this reference. Point a browser at
  `http://127.0.0.1:8765/docs` after starting for the auto-generated OpenAPI
  docs.
- Logs are always structured JSON lines (one object per log entry, keys
  include `event`/`level`/`timestamp`/`logger`), including uvicorn's own
  access/error logs — not just the app's own `trelix.*` loggers. This
  differs from every other command, which logs human-readable console text
  via the same underlying `logging.*` call sites. When `TRELIX_OTEL_ENABLED=true`,
  each JSON line emitted from inside an active span also carries `trace_id`/
  `span_id`, correlating logs with the OpenTelemetry traces described under
  [Environment variables](#environment-variables).

---

### `trelix graph`

#### Synopsis

```
trelix graph <repo_path> [--concepts] [--visualize] [--output PATH] [--json]
```

#### Description

Builds a code knowledge graph (nodes = symbols, edges = call/import
relationships) over the indexed repository. Optionally extracts semantic
concepts via LLM and exports an interactive Pyvis HTML visualization.

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--concepts` | `-c` | flag | `false` | Use LLM to extract semantic concepts for each community. Requires LLM API key. |
| `--visualize` | `-v` | flag | `false` | Export an interactive HTML visualization via Pyvis. |
| `--output` | `-o` | string | `<repo>/.trelix/graph.html` | Output path for the HTML file (only used with `--visualize`). |
| `--json` | | flag | `false` | Print graph statistics as JSON instead of Rich output. |

#### Examples

```bash
# Build the graph and print stats
trelix graph .

# Build and export a visual HTML file
trelix graph . --visualize

# Extract concepts and export
trelix graph . --concepts --visualize --output /tmp/graph.html

# JSON stats for CI or monitoring
trelix graph . --json
```

#### JSON output schema

```json
{
  "node_count": 1840,
  "edge_count": 5320,
  "community_count": 12,
  "concept_count": 0
}
```

#### Notes

- `trelix graph` builds the knowledge graph. The old command for displaying
  call/import edges for a single symbol has been renamed to
  `trelix call-graph`.
- Top 5 communities (by node count) are shown in Rich output, with up to 3
  representative files each.

---

### `trelix telemetry`

#### Synopsis

```
trelix telemetry [<repo_path>] [--limit N]
```

#### Description

Reads the `query_telemetry` table in the index database and displays the most
recent queries with their latency, result count, and query intent
classification.

#### Options

| Option | Short | Type | Default | Description |
|--------|------|---------|---------|-------------|
| `--limit` | `-n` | integer | `20` | Number of rows to display. |

#### Examples

```bash
# Show the last 20 queries
trelix telemetry .

# Show the last 100 queries
trelix telemetry . --limit 100

# Explicit repo path
trelix telemetry /my/repo -n 50
```

#### Notes

- Telemetry is off by default. Enable with `TRELIX_TELEMETRY_ENABLED=true`.
  Without this setting, the table will be empty and the command prints a
  yellow warning.
- `<repo_path>` defaults to `.` (current directory) if omitted.

---

### `trelix eval`

#### Synopsis

```
trelix eval [<repo_path>] --golden <file>
```

#### Description

Evaluates retrieval quality by running every query in a golden JSONL file and
computing nDCG@10, Recall@10, and MRR against the expected relevant files.

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--golden` | `-g` | string | `.trelix/golden.jsonl` | Path to the golden JSONL file. |

#### Examples

```bash
# Evaluate with the default golden file
trelix eval .

# Use a custom golden file
trelix eval . --golden tests/golden_queries.jsonl
trelix eval /my/repo -g /shared/golden.jsonl
```

#### Golden file format

Each line is a JSON object:

```jsonl
{"query": "how does token refresh work", "relevant_files": ["src/auth.py"]}
{"query": "database connection pool", "relevant_files": ["src/db/pool.go", "src/db/connection.go"]}
```

#### Output

```
 Retrieval Evaluation Results
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric            ┃  Score ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ nDCG@10           │ 0.8721 │
│ Recall@10         │ 0.9143 │
│ MRR               │ 0.8934 │
│ Queries evaluated │     14 │
└───────────────────┴────────┘
```

#### Notes

- `<repo_path>` defaults to `.` if omitted.
- Exits with code 1 if the golden file does not exist, and prints instructions
  for creating one.

---

### `trelix eval-synthesis`

#### Synopsis

```
trelix eval-synthesis [<repo_path>] --golden <file>
```

#### Description

Evaluates LLM synthesis quality (not just retrieval) by running every query in
a golden JSONL file through the full retrieve-and-synthesize pipeline and
scoring the generated answer GroUSE-style: hallucination rate, completeness,
and faithfulness against the expected answer fragments and symbols.

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--golden` | `-g` | string | `.trelix/golden_synthesis.jsonl` | Path to the golden JSONL file. |

#### Examples

```bash
# Evaluate with the default golden file
trelix eval-synthesis .

# Use a custom golden file
trelix eval-synthesis . --golden tests/golden_synthesis_queries.jsonl
trelix eval-synthesis /my/repo -g /shared/golden_synthesis.jsonl
```

#### Golden file format

Each line is a JSON object — a superset of `trelix eval`'s golden format,
adding two optional fields:

```jsonl
{"query": "how does JWT validation work?", "relevant_files": ["src/auth/middleware.py"], "expected_answer_fragments": ["decode", "secret", "bearer"], "expected_symbols": ["AuthMiddleware.verify", "jwt.decode"]}
```

- `expected_answer_fragments` — substrings the synthesized answer should
  contain (case-insensitive). Optional.
- `expected_symbols` — qualified symbol names the answer should reference.
  Optional.
- Queries that omit both optional fields still contribute to `n_queries` with
  a score of `1.0`.

#### Output

```
    Synthesis Quality Results (GroUSE-style)     
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Metric             ┃  Score ┃ Direction       ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Hallucination rate │ 0.0500 │ lower = better  │
│ Completeness       │ 0.9100 │ higher = better │
│ Faithfulness       │ 0.9400 │ higher = better │
│ Overall            │ 0.9300 │ higher = better │
│ Queries evaluated  │     12 │                 │
└────────────────────┴────────┴─────────────────┘
```

#### Notes

- `<repo_path>` defaults to `.` if omitted.
- Requires a configured LLM provider (synthesis makes real LLM calls) —
  unlike `trelix eval`, which only exercises retrieval.
- **Unlike `trelix eval`**, a missing golden file does *not* raise an error
  or exit non-zero — it exits `0` and prints a table of all-zero scores with
  `Queries evaluated = 0`. Double-check the `--golden` path if you see an
  all-zero result; it usually means the file wasn't found, not that
  synthesis quality is actually zero.

---

### `trelix taint`

#### Synopsis

```
trelix taint [<repo_path>] [--tier TIER] [--severity SEVERITY] [--json]
```

#### Description

Runs Semgrep taint analysis on the repository and displays source-to-sink data
flows. Results are also persisted to the index database for later querying.

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--tier` | `-t` | string | `default` | Analysis tier: `default` \| `intrafile` \| `interfile`. |
| `--severity` | `-s` | string | `""` (all) | Filter output by severity: `ERROR` \| `WARNING` \| `INFO`. |
| `--json` | | flag | `false` | Output flows as JSON. |

#### Examples

```bash
# Run default taint analysis
trelix taint .

# Interfile analysis with ERROR-only output
trelix taint . --tier interfile --severity ERROR

# JSON output
trelix taint . --json

# Suppress WARNING and INFO, show only errors
trelix taint /my/repo --severity ERROR --json
```

#### JSON output schema

```json
[
  {
    "rule": "python.django.security.injection.tainted-sql-string",
    "severity": "ERROR",
    "source": "src/views.py:42",
    "sink": "src/db/query.py:107"
  }
]
```

#### Notes

- Requires Semgrep: `pip install trelix[taint]`.
- Rich table output is capped at 50 rows. Use `--json` for the full set.
- `<repo_path>` defaults to `.` if omitted.

---

### `trelix review`

#### Synopsis

```
trelix review [<repo_path>] [--diff FILE | --base REF --head REF] [--json] [--max-files N]
trelix review [<repo_path>] --pr OWNER/REPO#NUMBER [--post-comments] [--json]
```

#### Description

Performs retrieval-augmented code review on a git diff. trelix retrieves
context for each changed hunk and uses an LLM to generate structured review
comments with severity labels (`ERROR`, `WARN`, `INFO`).

Without `--pr`, uses a local git diff (from a file or by running `git diff`).
With `--pr`, fetches the diff directly from the GitHub API.

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--diff` | `-d` | string | — | Path to a `.diff` file. If omitted (and `--pr` not set), runs `git diff --base --head`. |
| `--base` | | string | `HEAD~1` | Base git ref for the diff. |
| `--head` | | string | `HEAD` | Head git ref for the diff. |
| `--json` | | flag | `false` | Output review comments as JSON. |
| `--max-files` | | integer | `10` | Maximum number of files to review from the diff. |
| `--pr` | | string | — | GitHub PR reference in the form `owner/repo#number`. Fetches the diff from the GitHub API. Requires `GITHUB_TOKEN`. **New in v2.4.0** |
| `--post-comments` | | flag | `false` | Post findings back to GitHub as a batched PR review. Requires `GITHUB_TOKEN` with `pull_requests:write`. **New in v2.4.0** |

#### Examples

```bash
# Review last commit
trelix review .

# Review a range of commits
trelix review . --base main --head feature/auth

# Review from a saved diff file
trelix review . --diff changes.patch

# JSON output
trelix review . --json

# Review a GitHub PR (v2.4.0)
GITHUB_TOKEN=$TOKEN trelix review . --pr acme/backend#142

# Review and post comments back to GitHub
GITHUB_TOKEN=$TOKEN trelix review . --pr acme/backend#142 --post-comments
```

#### JSON output schema

```json
[
  {
    "file": "src/auth.py",
    "lines": "42-56",
    "severity": "ERROR",
    "comment": "Token is logged before validation — potential secret leak."
  }
]
```

#### Notes

- `<repo_path>` defaults to `.` if omitted.
- `--pr` and `--diff`/`--base`/`--head` are mutually exclusive.
- Binary and oversized files from GitHub PRs are skipped automatically.
- PRs with more than 3,000 changed files will trigger a truncation warning.

---

### `trelix link-tickets`

#### Synopsis

```
trelix link-tickets <repo> [--max-commits N] [--since DATE] [--ticket-pattern REGEX]
```

#### Description

Walks the git history of an already-indexed repository, regex-matches ticket
IDs (e.g. Jira-style `PROJ-123`) in commit messages, and links every symbol in
each ticket-referencing commit's touched files to that ticket via a new
`generic_edges` row (`source_ref` format `ticket:<id>`). This feeds a
bidirectional edge into `rank_by_pagerank()`'s graph so cross-source
references influence PageRank.

Requires `<repo>` to already be indexed (`trelix index <repo>` must have been
run first) and to be a real git checkout. Non-git directories or repos with no
matching commits degrade gracefully to zero edges linked — the command never
raises for these cases. Re-running on the same repo is idempotent: a unique
index prevents duplicate edges from being created.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--max-commits` | integer | `5000` | Maximum number of commits to walk. Bounds cost on large repos with long histories. |
| `--since` | string | _(none)_ | Only walk commits after this date, e.g. `"90 days ago"`. Passed straight through to `git log --since`. |
| `--ticket-pattern` | string (regex) | `[A-Z]+-\d+` | Regex for matching ticket IDs in commit messages. The default matches Jira-style tickets (`PROJ-123`); override for other conventions, e.g. GitHub-issue style (`#\d+`). |

#### Examples

```bash
# Link tickets referenced in the full history (up to 5,000 commits)
trelix link-tickets ./my-repo

# Only walk the last 90 days of commits
trelix link-tickets ./my-repo --since "90 days ago"

# Bound the walk and use a GitHub-issue-style ticket pattern
trelix link-tickets ./my-repo --max-commits 1000 --ticket-pattern '#\d+'
```

#### Output

```
Linked 42 symbol-ticket edge(s).
```

Or, when no commits matched:

```
No ticket references linked. Either this isn't a git repo, no commits matched
the ticket pattern, or no touched files are indexed yet.
```

#### Notes

- Run `trelix index <repo>` before this command — it exits with code 1 if no
  index is found.
- Safe to re-run: existing symbol-ticket edges are never duplicated.
- This is a separate, slower pass from `trelix index` — it is not invoked
  automatically during indexing.

---

### `trelix connector sync`

#### Synopsis

```
trelix connector sync <repo> <jira|testrail|xray|linear> [--link/--no-link]
```

#### Description

Fetches artifacts (Jira tickets, TestRail test cases, Xray Cloud tests, or
Linear issues) from an external system via the connector's
`ArtifactSource.fetch()` and writes them to the `artifacts` table via
`upsert_artifact()`, keyed by source reference so re-syncing updates existing
rows rather than duplicating them.

Requires `<repo>` to already be indexed (checks that `.trelix/index.db`
exists before making any HTTP call). Required environment variables differ
per connector — see [CONFIGURATION.md](CONFIGURATION.md) for the full list of
`TRELIX_JIRA_*`, `TRELIX_TESTRAIL_*`, `TRELIX_XRAY_*`, and `TRELIX_LINEAR_*`
variables. Missing required configuration fails fast with an error, before
any network request is made.

By default (`--link`, the default), each successfully-synced artifact is
immediately linked into `generic_edges` via `ArtifactLinker` — it's reachable
from the code graph the moment this command returns, no separate
[`trelix link-artifacts`](#trelix-link-artifacts) pass required. Pass
`--no-link` to skip linking (e.g. when syncing many artifacts quickly, then
running [`trelix link-artifacts`](#trelix-link-artifacts) once as a batch
afterward).

#### Arguments

| Argument | Description |
|----------|-------------|
| `repo` | Path to the indexed repository. |
| `name` | Connector to sync: `jira`, `testrail`, `xray`, or `linear`. |

#### Options

| Option | Description |
|--------|-------------|
| `--link` / `--no-link` | Auto-link each synced artifact into `generic_edges` via `ArtifactLinker` (default: `--link`). |

#### Examples

```bash
# Sync Jira tickets (requires TRELIX_JIRA_* env vars)
TRELIX_JIRA_BASE_URL=https://acme.atlassian.net \
TRELIX_JIRA_EMAIL=bot@acme.com \
TRELIX_JIRA_API_TOKEN=$JIRA_TOKEN \
TRELIX_JIRA_PROJECT_KEY=PROJ \
trelix connector sync ./my-repo jira

# Sync TestRail cases (requires TRELIX_TESTRAIL_* env vars)
TRELIX_TESTRAIL_BASE_URL=https://acme.testrail.io \
TRELIX_TESTRAIL_USERNAME=bot@acme.com \
TRELIX_TESTRAIL_API_KEY=$TESTRAIL_KEY \
TRELIX_TESTRAIL_PROJECT_ID=7 \
trelix connector sync ./my-repo testrail

# Sync Xray Cloud tests (requires TRELIX_XRAY_* env vars)
TRELIX_XRAY_CLIENT_ID=$XRAY_CLIENT_ID \
TRELIX_XRAY_CLIENT_SECRET=$XRAY_CLIENT_SECRET \
TRELIX_XRAY_PROJECT_KEY=PROJ \
TRELIX_XRAY_JIRA_BASE_URL=https://acme.atlassian.net \
trelix connector sync ./my-repo xray

# Sync Linear issues (requires TRELIX_LINEAR_* env vars)
TRELIX_LINEAR_API_KEY=$LINEAR_API_KEY \
TRELIX_LINEAR_TEAM_KEY=ENG \
trelix connector sync ./my-repo linear
```

#### Output

```
Synced jira: fetched 84, wrote 84, errors 0, linked 79 edge(s)
```

#### Notes

- Exits with code 1 if `errors > 0`, or if required connector configuration
  is missing — configuration is validated before any HTTP call is made.
- Run `trelix index <repo>` before this command — it exits with code 1 if no
  index is found.
- Page size is configured via environment variable only; see
  [CONFIGURATION.md](CONFIGURATION.md). Defaults: `TRELIX_JIRA_PAGE_SIZE`
  (`100`, max `100`), `TRELIX_TESTRAIL_PAGE_SIZE` (`250`, max `250` —
  TestRail's own API ceiling), `TRELIX_XRAY_PAGE_SIZE` (`100`, max `100`),
  `TRELIX_LINEAR_PAGE_SIZE` (`100`, max `100` — not a confirmed Linear
  platform ceiling, chosen to stay well under its GraphQL query-complexity
  cap).

---

### `trelix link-artifacts`

#### Synopsis

```
trelix link-artifacts <repo> [--embedding-fallback] [--similarity-threshold FLOAT]
```

#### Description

Scans every artifact already synced into the `artifacts` table (via
`trelix connector sync`) for mentions of indexed symbol names/qualified
names, and links matches into `generic_edges` (`source_ref` matching the
artifact's own, `edge_kind="references_artifact"`) — the artifact-content
counterpart to `trelix link-tickets`'s git-commit-message matching. `trelix
connector sync` writes to the `artifacts` table only; it does not create
`generic_edges` rows on its own unless run with its default `--link` flag,
which calls this same linking logic per-artifact as it syncs. Run
`link-artifacts` standalone when you want a full re-link pass — e.g. after
syncing with `--no-link`, or after a schema/symbol change that could surface
new matches against already-synced artifacts.

Regex reference-extraction runs unconditionally (free, deterministic) and
always takes priority. An opt-in embedding-similarity fallback
(`--embedding-fallback`) runs only for artifacts where the regex pass found
zero matches — it costs one embed call per unmatched artifact, and produces
lower-confidence edges (`weight=0.5` vs. a regex hit's `weight=1.0`) so
fallback matches don't dominate PageRank mass.

Requires `<repo>` to already be indexed (`trelix index <repo>` must have been
run first). Re-running is idempotent — the same `generic_edges` unique index
that backs `link-tickets` prevents duplicate edges.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--embedding-fallback` | flag | off | For artifacts with no regex match, fall back to embedding similarity against indexed chunks. Costs one embed call per unmatched artifact. |
| `--similarity-threshold` | float | `0.75` | Minimum similarity score (0.0–1.0) for an embedding-fallback match to count. Ignored unless `--embedding-fallback` is set. |

Both options map to `ArtifactLinkerConfig`'s `TRELIX_ARTIFACT_LINKER_EMBEDDING_FALLBACK_ENABLED`/`TRELIX_ARTIFACT_LINKER_SIMILARITY_THRESHOLD` — note the `_ENABLED` suffix, since the field is `embedding_fallback_enabled` and carries no alias. See [CONFIGURATION.md](CONFIGURATION.md).

#### Examples

```bash
# Regex-only re-link pass over everything already synced
trelix link-artifacts ./my-repo

# Also fall back to embedding similarity for artifacts with no regex match
trelix link-artifacts ./my-repo --embedding-fallback --similarity-threshold 0.8
```

#### Output

```
Linked 17 symbol-artifact edge(s).
```

Or, when nothing matched:

```
No artifact references linked. Either no artifacts have been synced yet
(run `trelix connector sync`), or none mention an indexed symbol by name.
```

#### Notes

- Exits with code 1 if no index is found at `<repo>` — run `trelix index <repo>` first.
- `trelix connector sync`'s default `--link` behavior already calls this linker per-artifact as it syncs — most workflows never need to run `link-artifacts` directly; it exists for a standalone/batch re-link pass.

---

### `trelix search-all`

#### Synopsis

```
trelix search-all <query> [--k N] [--json] [--config PATH]
```

#### Description

Runs a hybrid search across all repositories registered in the federation
registry. Uses Reciprocal Rank Fusion (RRF) weighted by each repo's registered
weight to merge results. Displays results grouped by source repo.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--k` | integer | `10` | Number of top results to retrieve per registered repo. |
| `--json` | flag | `false` | Output results as JSON. |
| `--config` | string | `~/.config/trelix/repos.json` | Path to the federation registry JSON file. |

#### Examples

```bash
# Search all registered repos
trelix search-all "authentication middleware"

# Increase result count per repo
trelix search-all "database migrations" --k 20

# JSON output
trelix search-all "error handling" --json

# Custom registry
trelix search-all "rate limiter" --config /projects/.trelix/repos.json
```

#### JSON output schema

```json
[
  {
    "file": "src/auth.py",
    "symbol": "trelix.auth.middleware.check_token",
    "score": 0.9231,
    "source": "backend:sqlite://"
  }
]
```

#### Notes

- Register repos with `trelix federation add` before using this command.
- Rich table output is capped at 20 rows. Use `--json` for the full set.
- Federation caching (TTL 120 s, SHA-256 keyed) is active by default for
  repeated identical queries.

---

### `trelix federation add`

#### Synopsis

```
trelix federation add <alias> <path> [--weight WEIGHT] [--config PATH]
```

#### Description

Registers a repository in the federation registry so it participates in
`trelix search-all` and `trelix watch-all`. Persists the entry to the registry
JSON file.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--weight` | float | `1.0` | RRF score weight for this repo. Higher values up-rank results from this repo in federated search. |
| `--config` | string | `~/.config/trelix/repos.json` | Path to the federation registry JSON file. |

#### Examples

```bash
# Register the current directory
trelix federation add myapp .

# Register a remote path with a higher weight
trelix federation add backend /repos/backend --weight 1.5

# Use a project-local registry
trelix federation add frontend /repos/frontend \
  --config .trelix/federation.json
```

#### Notes

- Exits with code 1 if `<alias>` is already registered.
- `<path>` should be an absolute path to the repo root for reliable
  cross-directory usage.
- The registry file is created (including parent directories) if it does not
  exist.

---

### `trelix federation list`

#### Synopsis

```
trelix federation list [--config PATH]
```

#### Description

Lists all repositories registered in the federation registry, showing their
alias, path, and RRF weight.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | string | `~/.config/trelix/repos.json` | Path to the federation registry JSON file. |

#### Examples

```bash
# List all registered repos
trelix federation list

# List from a custom registry
trelix federation list --config .trelix/federation.json
```

#### Output example

```
      Registered Repos
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Alias    ┃ Path            ┃ Weight ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ backend  │ /repos/backend  │    1.5 │
│ frontend │ /repos/frontend │    1.0 │
└──────────┴─────────────────┴────────┘
```

---

### `trelix federation remove`

#### Synopsis

```
trelix federation remove <alias> [--config PATH]
```

#### Description

Unregisters a repo from the federation registry by alias. No-op (prints a
message, exits 0) if the alias isn't registered.

#### Examples

```bash
trelix federation remove backend
trelix federation remove backend --config .trelix/federation.json
```

---

### `trelix agent sessions list`

#### Synopsis

```
trelix agent sessions list <repo> [--limit N]
```

Lists persisted agentic (ReAct) sessions for a repo, most recently active
first. See [`trelix ask --agentic`](#trelix-ask) for how sessions are created.

### `trelix agent sessions show`

#### Synopsis

```
trelix agent sessions show <repo> <session_id>
```

Shows the full turn-by-turn (thought/action/observation) history for a
persisted session.

### `trelix agent sessions clear`

#### Synopsis

```
trelix agent sessions clear <repo> <session_id>
```

Deletes a persisted session and all its turns.

---

### `trelix audit list`

#### Synopsis

```
trelix audit list [--db PATH] [--limit N]
```

#### Description

Shows the most recent entries of the tamper-evident audit log, newest first, as
a Rich table (`id`, `ts`, `principal`, `action`, `resource`, `outcome`,
`status`).

The audit log lives in its own `audit.db` — never in the code index, which is
disposable and rebuilt at will. Written only when
`TRELIX_AUDIT_ENABLED=true` and only for requests through the HTTP API
(`trelix serve`). See [AUDIT.md](AUDIT.md) for the full event schema and the
integrity model.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--db` | string | `TRELIX_AUDIT_DB_PATH`, else `<cwd>/.trelix/audit.db` | Path to `audit.db`. |
| `--limit` / `-n` | integer | `50` | Number of most-recent rows to show. |

#### Examples

```bash
# Last 50 entries from the default location
trelix audit list

# Last 10 entries
trelix audit list -n 10

# An audit DB kept outside the repo
trelix audit list --db /var/log/trelix/audit.db
```

#### Output example

```
                   Audit Log (/var/log/trelix/audit.db)
┏━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ id ┃ ts           ┃ principal    ┃ action ┃ resource      ┃ outcome ┃ status ┃
┡━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│  3 │ 2026-08-13T… │ static-token │ admin  │ /graph/commu… │ denied  │    401 │
│  2 │ 2026-08-13T… │ a1b2c3d4@ht… │ ask    │ /ask          │ success │    200 │
│  1 │ 2026-08-13T… │ static-token │ search │ /search       │ success │    200 │
└────┴──────────────┴──────────────┴────────┴───────────────┴─────────┴────────┘
```

#### Notes

- Prints `No audit entries.` and exits 0 when the log is empty.
- Opening a path that does not exist **creates** an empty `audit.db` (parent
  directories included), so a typo in `--db` yields an empty table rather than
  an error.
- Long values are truncated to fit the terminal, and the two hash columns are
  not shown at all. Use [`trelix audit export`](#trelix-audit-export) when you
  need full values.
- Every cell is escaped before printing: audit rows record request-controlled
  strings (a URL path, a JWT `sub`), which Rich would otherwise interpret as
  console markup.

---

### `trelix audit verify`

#### Synopsis

```
trelix audit verify [--db PATH]
```

#### Description

Walks the whole hash chain and reports whether the audit log is intact. Detects
a mutated row (its recomputed `entry_hash` no longer matches), a
deleted/reordered row (the next row's `prev_hash` no longer links, or the `id`
sequence has a gap), and a deleted tail (the live row count / head hash no
longer match the in-DB anchor).

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--db` | string | `TRELIX_AUDIT_DB_PATH`, else `<cwd>/.trelix/audit.db` | Path to `audit.db`. |

#### Examples

```bash
# Verify the default audit DB
trelix audit verify

# Verify in CI / a cron job and fail the job on tamper
trelix audit verify --db /var/log/trelix/audit.db || echo "AUDIT TAMPERED"
```

#### Output

```
# intact (exit 0)
Audit chain intact.

# tampered (exit 1, on stderr)
Audit chain TAMPERED — first divergent entry id: 3
```

#### Notes

- **Exit code `0`** = chain intact, **`1`** = tamper detected. The failure
  message names the id of the *first* divergent entry, which is where to start
  an investigation — entries before it verified cleanly.
- An empty log verifies as intact (exit 0).
- This is tamper **evidence**, not tamper **proofing** — see
  [AUDIT.md](AUDIT.md#integrity-model--tamper-evident-not-tamper-proof) for
  exactly what it does and does not protect against.

---

### `trelix audit export`

#### Synopsis

```
trelix audit export [--db PATH] [--format ndjson]
```

#### Description

Writes every audit entry to stdout as NDJSON (one JSON object per line), in
append order (oldest first) so a downstream consumer can re-verify the chain.
Unlike `audit list`, the export includes every column — including `prev_hash`
and `entry_hash`.

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--db` | string | `TRELIX_AUDIT_DB_PATH`, else `<cwd>/.trelix/audit.db` | Path to `audit.db`. |
| `--format` | string | `ndjson` | Export format. `ndjson` is the only accepted value; anything else prints an error and exits 1. |

#### Examples

```bash
# Export to a file for SIEM ingestion
trelix audit export > /var/log/trelix/audit.ndjson

# Only denied requests
trelix audit export | jq -c 'select(.outcome == "denied")'

# Count requests per principal
trelix audit export | jq -r '.principal' | sort | uniq -c
```

#### NDJSON output schema

```json
{"id": 1, "ts": "2026-08-13T09:14:02.115331+00:00", "principal": "static-token", "action": "search", "resource": "/search", "outcome": "success", "status_code": 200, "client_ip": "127.0.0.1", "request_id": "4f1c9e2b8a7d4c10", "trace_id": null, "duration_ms": 87, "detail": null, "prev_hash": "0000000000000000000000000000000000000000000000000000000000000000", "entry_hash": "5d71295c36283eaac08929a0bac81f68b025a3fb62ac69c0659d6d5ff95c06ee"}
```

#### Notes

- NDJSON is the shipping format on purpose: Filebeat, Vector and Fluent Bit all
  tail newline-delimited JSON natively, so no bespoke HTTP transport is needed.
  See [AUDIT.md](AUDIT.md#shipping-to-a-siem) for ready-to-adapt shipper
  configs.
- The whole table is read into memory before the first line is printed. That is
  fine for the operational sizes this log is meant for, but it is not a
  streaming cursor over a multi-GB log.
- The export is a snapshot: entries appended while it runs are not included.

---

*End of CLI Reference — trelix v3.0.0*
