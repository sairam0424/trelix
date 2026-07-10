# trelix CLI Reference

**Version:** 2.7.1  
**Last updated:** 2026-07-10

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
   - [taint](#trelix-taint)
   - [review](#trelix-review)
   - [search-all](#trelix-search-all)
   - [federation add](#trelix-federation-add)
   - [federation list](#trelix-federation-list)

---

## Global flags

These flags are processed before any subcommand.

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--version` | `-V` | flag | Print the installed trelix version and exit. |
| `--help` | | flag | Show help text for the command or subcommand and exit. |

**Examples**

```bash
trelix --version        # trelix 2.7.1
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
| `TRELIX_RETRIEVAL_SPARSE` | `false` | Enable SPLADE-Code sparse retrieval leg |
| `TRELIX_RETRIEVAL_AGENTIC` | `false` | Enable multi-turn ReAct loop (also set by `--agentic`) |
| `TRELIX_RETRIEVAL_FLARE` | `false` | Enable FLARE confidence-gated re-retrieval |
| `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` | `1` | Max FLARE retries (range: 1–3). Replaces the old `TRELIX_RETRIEVAL_FLARE_MAX_ITER` (deprecated, removed in v3.0.0). Values > 3 raise `ValidationError` at startup. |
| `TRELIX_RETRIEVAL_PAGERANK_BOOST` | `false` | Boost results by PageRank symbol importance |
| `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING` | `true` | Apply per-language RRF score multipliers |
| `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_<LANG>` | varies | Per-language override, e.g. `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN=0.1` |

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

- Reranking is enabled by default (Cohere cross-encoder). Disable with
  `TRELIX_RETRIEVAL_RERANK_PROVIDER` or set `rerank=false`.
- The `--provider` flag affects the query embedding, not index scanning.
  Always use the same provider that was used for `trelix index`.

---

### `trelix ask`

#### Synopsis

```
trelix ask <repo_path> <question> [--provider PROVIDER] [--agentic]
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

#### Notes

- A `federation remove` CLI command does not exist as of v2.7.1. To remove a
  repo, edit the registry JSON file directly and delete the corresponding
  entry from the `"repos"` array. The default registry path is
  `~/.config/trelix/repos.json`.

---

*End of CLI Reference — trelix v2.7.1*
