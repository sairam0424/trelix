# Trelix v3.1.2 — Installation Guide

This guide covers every installation scenario for Trelix v3.1.2, from a quick
one-liner to Docker, standalone binaries, and virtual-environment setups.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Install (recommended)](#2-quick-install-recommended)
3. [Install Options by Use Case](#3-install-options-by-use-case)
4. [Standalone Binaries (no Python needed)](#4-standalone-binaries-no-python-needed)
5. [Virtual Environment](#5-virtual-environment-recommended-for-projects)
6. [uv (faster installs)](#6-uv-faster-installs)
7. [Docker (serve mode)](#7-docker-for-serve-mode)
8. [Verify Installation](#8-verify-installation)
9. [Environment Variables Reference](#9-environment-variables-reference)
10. [Upgrading from v2.3.x](#10-upgrading-from-v23x)

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.11, 3.12, or 3.13** | No known upper bound |
| **pip** or **uv** | pip ships with Python; uv is optional but significantly faster |
| **~500 MB free disk** | The local embedder model is downloaded on first use and cached by `sentence-transformers`/HuggingFace, i.e. under `$HF_HOME` (default `~/.cache/huggingface/hub/`). trelix passes no custom cache directory and has no `TRELIX_CACHE_DIR` setting. |
| `OPENAI_API_KEY` | Optional — enables OpenAI embeddings (higher quality, requires internet) |
| `AZURE_API_KEY` + `AZURE_ENDPOINT` | Optional — enables Azure OpenAI embeddings |
| `VOYAGE_API_KEY` | Optional — enables Voyage AI embeddings |

Check your Python version before installing:

```bash
python --version   # must be 3.11.x or 3.12.x
pip --version
```

---

## 2. Quick Install (recommended)

For most users the `local` extra bundles the offline sentence-transformer model
so no API key is required.

```bash
pip install "trelix[local]"
trelix --version   # prints "trelix <version>", e.g. trelix 3.1.2
```

On first use, Trelix downloads the embedder model (~420 MB) to
the HuggingFace cache (`$HF_HOME`, default `~/.cache/huggingface/hub/`). Subsequent runs
use the cached copy.

---

## 3. Install Options by Use Case

Pick the extras that match your workflow. Extras can be combined with commas
inside the brackets (e.g., `"trelix[local,rerank]"`).

### 3.1 Local-only (offline, no API key)

Best for air-gapped environments or when you do not want to send code to an
external service.

```bash
pip install "trelix[local]"
```

- Uses a bundled sentence-transformer model (all-MiniLM-L6-v2 by default).
- No network calls after the initial model download.

### 3.2 OpenAI embeddings (best quality)

Requires an OpenAI account and an active API key.

```bash
pip install trelix
export OPENAI_API_KEY="sk-..."
```

Set `TRELIX_EMBEDDER_PROVIDER=openai` (or pass `--provider openai`) to activate.

### 3.3 Azure OpenAI embeddings

Requires an Azure OpenAI resource with an embeddings deployment.

```bash
pip install trelix
export AZURE_API_KEY="..."
export AZURE_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_EMBEDDINGS_MODEL="text-embedding-3-large"   # your deployment name
```

Set `TRELIX_EMBEDDER_PROVIDER=azure` to activate.

### 3.4 MCP server (Claude Code / Cursor integration)

Exposes Trelix as an MCP tool your AI assistant can call directly.

```bash
pip install trelix-mcp
which trelix-mcp   # trelix-mcp takes no arguments — running it starts the stdio server
```

See [MCP_GUIDE.md](MCP_GUIDE.md) for the full Claude Code / Cursor setup.

### 3.5 LangChain integration

```bash
pip install trelix-langchain
```

Provides `TrelixRetriever` as a drop-in LangChain `BaseRetriever`.

### 3.6 LlamaIndex integration

```bash
pip install trelix-llama-index
```

Provides `TrelixQueryEngine` compatible with LlamaIndex query pipelines.

### 3.7 Real-time file watching

Automatically re-indexes files as they change on disk.

```bash
pip install "trelix[watch]"
trelix watch ./
```

Uses `watchdog` under the hood; supports Linux inotify, macOS FSEvents, and
Windows ReadDirectoryChangesW.

### 3.8 REST API server

Serve the Trelix index over HTTP for multi-user or CI environments.

```bash
pip install "trelix[serve]"
trelix serve ./ --port 8765
```

The OpenAPI spec is available at `http://localhost:8765/docs`.

### 3.9 Knowledge graph + visualization

Builds a relationship graph across files and renders it as an interactive
HTML/D3 diagram.

```bash
pip install "trelix[knowledge-graph]"
trelix graph ./  --output graph.html
```

### 3.10 Large-scale vector store (>100k chunks)

For very large monorepos or when SQLite becomes a bottleneck, use the LanceDB
backend.

```bash
pip install "trelix[lance]"
export TRELIX_STORE_BACKEND=lance
trelix index ./
```

LanceDB stores data at `.trelix/lance` inside the repo by default (override with
`LANCE_URI`; the table name is `LANCE_TABLE`, default `chunks`).

### 3.11 Reranking (Cohere)

Cross-encoder reranking improves precision on ambiguous queries.

```bash
pip install "trelix[rerank]"
export COHERE_API_KEY="..."
# Reranking is ON by default (TRELIX_RETRIEVAL_RERANK=true); set it to false to disable.
export TRELIX_RETRIEVAL_RERANK_PROVIDER=cross_encoder   # or cohere (default) / plaid / xtr
```

### 3.12 Everything (all extras)

```bash
pip install "trelix[local,rerank,voyage,qdrant,watch]"
```

This installs all first-party extras. Additional integrations
(`trelix-langchain`, `trelix-llama-index`, `trelix-mcp`) are separate packages
and must be installed independently.

---

## 4. Standalone Binaries (no Python needed)

Pre-compiled single-file binaries are published to the
[GitHub Releases](https://github.com/sairam0424/trelix/releases/latest)
page for each platform. No Python or pip required.

### macOS ARM64 (Apple Silicon)

```bash
curl -L https://github.com/sairam0424/trelix/releases/latest/download/trelix-macos-arm64 \
     -o /usr/local/bin/trelix
chmod +x /usr/local/bin/trelix
trelix --version
```

### Windows x64

Download `trelix-windows-x64.exe` from the Releases page and place it
somewhere on your `PATH`, or run it directly:

```powershell
.\trelix-windows-x64.exe --version
```

### Linux x64

```bash
curl -L https://github.com/sairam0424/trelix/releases/latest/download/trelix-linux-x64 \
     -o /usr/local/bin/trelix
chmod +x /usr/local/bin/trelix
trelix --version
```

### Linux ARM64

```bash
curl -L https://github.com/sairam0424/trelix/releases/latest/download/trelix-linux-arm64 \
     -o /usr/local/bin/trelix
chmod +x /usr/local/bin/trelix
trelix --version
```

Binaries are built with PyInstaller and include all Python dependencies. The
local embedder model is still downloaded to the HuggingFace cache
(`$HF_HOME`, default `~/.cache/huggingface/hub/`) on first use.

---

## 5. Virtual Environment (recommended for projects)

Isolating Trelix in a virtual environment prevents dependency conflicts with
other project packages.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install "trelix[local]"
trelix --version
```

### Windows (Command Prompt)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install "trelix[local]"
trelix --version
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install "trelix[local]"
trelix --version
```

Add `.venv/` to your `.gitignore` so the environment is not committed.

---

## 6. uv (faster installs)

[uv](https://github.com/astral-sh/uv) resolves and installs packages
significantly faster than pip and handles virtual environments automatically.

```bash
# Install uv (once, globally)
curl -Lf https://astral.sh/uv/install.sh | sh

# Add trelix to your project
uv add "trelix[local]"

# Or install without a project (global-like)
uv tool install "trelix[local]"
trelix --version
```

`uv add` pins the version in `pyproject.toml` / `uv.lock` so the install is
fully reproducible.

---

## 7. Docker (for serve mode)

The official image exposes the `trelix serve` REST API and can index any
directory you mount.

### Index a repository

```bash
docker run --rm \
  -v "$(pwd):/repo" \
  ghcr.io/sairam0424/trelix:latest \
  index /repo
```

The index is written to `/repo/.trelix/` inside the container (which maps to
`$(pwd)/.trelix/` on your host).

### Start the REST server

```bash
docker run --rm -p 8765:8765 \
  -v "$(pwd):/repo" \
  ghcr.io/sairam0424/trelix:latest \
  serve /repo --host 0.0.0.0 --port 8765
```

Then open `http://localhost:8765/docs` for the interactive API reference.

### Use with OpenAI embeddings

```bash
docker run --rm \
  -e OPENAI_API_KEY="sk-..." \
  -e TRELIX_EMBEDDER_PROVIDER=openai \
  -v "$(pwd):/repo" \
  ghcr.io/sairam0424/trelix:latest \
  index /repo
```

### Docker Compose example

A ready-to-run `docker-compose.yml` ships at the repo root:

```bash
OPENAI_API_KEY=sk-... docker compose up
```

By default it serves the current directory; set `REPO_PATH` to serve a
different repo:

```bash
REPO_PATH=/path/to/other/repo OPENAI_API_KEY=sk-... docker compose up
```

### Image variants

Two tags are published per release:

- `ghcr.io/sairam0424/trelix:X.Y.Z` — slim, API-embedder-only (OpenAI, Voyage,
  Cohere, Azure). Recommended default.
- `ghcr.io/sairam0424/trelix:X.Y.Z-local` — bundles `sentence-transformers`
  and `torch` for the local/offline embedder and cross-encoder reranker.
  Multi-gigabyte image; only pull this if you need `TRELIX_EMBEDDER_PROVIDER=local`
  inside the container.

---

## 8. Verify Installation

Run these commands after any installation method to confirm everything is
working correctly.

```bash
# Print version (must print the version you installed, e.g. 3.1.2)
trelix --version

# Print usage summary
trelix --help

# Index the current directory (creates .trelix/ index)
trelix index ./

# Show index statistics
trelix stats ./

# Run a test search query (REPO is a positional argument — there is no --repo flag)
trelix search ./ "hybrid search"
```

Expected output for `trelix stats ./`:

```
Trelix Index Stats
  Version   : 3.1.2
  Chunks    : <n>
  Embedder  : local (all-MiniLM-L6-v2)
  Backend   : sqlite
  Index path: ./.trelix/
```

If `trelix --version` prints nothing or fails, check that:

1. Your virtual environment is activated (if using one).
2. The Python executable that installed Trelix is on your `PATH`.
3. You are using Python 3.11, 3.12, or 3.13 (`python --version`).

---

## 9. Environment Variables Reference

Trelix's entire configuration surface is [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).
Variable names follow `TRELIX_<SECTION>_<FIELD>`, where `<SECTION>` is the config group
(`WALKER`, `PARSER`, `CHUNKER`, `EMBEDDER`, `STORE`, `RETRIEVAL`, `LLM`, `SPARSE`,
`INDEXER`) — plus a handful of conventional third-party names (`OPENAI_API_KEY`,
`QDRANT_URL`, …) that are read under their standard names. There is no config *file*:
`.env` in the current working directory and the process environment are the only two
sources. See [CONFIGURATION.md](CONFIGURATION.md) for the exhaustive list and the
per-group `.env` caveats, and [CLI_REFERENCE.md](CLI_REFERENCE.md) for which handful of
these are also exposed as CLI flags.

**`.env` is resolved relative to your current working directory, not the indexed repo.**
The `TRELIX_WALKER_*`, `TRELIX_PARSER_*`, `TRELIX_CHUNKER_*`, and `TRELIX_SPARSE_*` groups
are read from the process environment only and ignore `.env` entirely.

### Provider credentials

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | _(none)_ | OpenAI key; used for both embeddings and LLM synthesis |
| `AZURE_API_KEY` | _(none)_ | Azure OpenAI key |
| `AZURE_ENDPOINT` | _(none)_ | Azure OpenAI resource endpoint URL |
| `AZURE_API_VERSION` | `2025-04-01-preview` | Azure OpenAI API version |
| `VOYAGE_API_KEY` | _(none)_ | Voyage AI key; required for `--provider voyage` |
| `ANTHROPIC_API_KEY` | _(none)_ | Anthropic key for `TRELIX_LLM_PROVIDER=anthropic` |
| `COHERE_API_KEY` | _(none)_ | Cohere key; required by the default Cohere reranker |
| `GITHUB_TOKEN` | _(none)_ | Required by `trelix review --pr` and `--post-comments` |
| `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_PROFILE` | _(standard AWS defaults)_ | Bedrock embedding and LLM providers |

### Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `TRELIX_EMBEDDER_PROVIDER` | `local` | `local`, `openai`, `azure`, `voyage`, `local-code`, `bge-code`, `nomic-code`, `bedrock-titan`, `bedrock-cohere`. Same values as the `--provider` flag. |
| `TRELIX_EMBEDDER_LOCAL_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model for the local embedder |
| `TRELIX_EMBEDDER_OPENAI_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `AZURE_EMBEDDINGS_MODEL` | `text-embedding-3-large` | Azure embeddings *deployment* name |
| `TRELIX_EMBEDDER_VOYAGE_MODEL` | `voyage-code-3` | Voyage embedding model |
| `TRELIX_EMBEDDER_BATCH_SIZE` | `64` | Embeddings per model/API call |

### Storage and vector backend

| Variable | Default | Description |
|----------|---------|-------------|
| `TRELIX_STORE_DB_PATH` | `.trelix/index.db` | Index location, relative to the repo root |
| `TRELIX_STORE_BACKEND` | `sqlite` | Vector store: `sqlite`, `qdrant`, `lance` |
| `TRELIX_STORE_HNSW` | `true` | Use an HNSW ANN index for the vector leg |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint; used when `TRELIX_STORE_BACKEND=qdrant` |
| `QDRANT_API_KEY` | _(none)_ | Qdrant API key for cloud deployments |
| `QDRANT_COLLECTION` | `trelix` | Qdrant collection name |
| `LANCE_URI` | `.trelix/lance` | LanceDB path; used when `TRELIX_STORE_BACKEND=lance` |

### Chunking, indexing, retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `TRELIX_CHUNKER_MAX_TOKENS_PER_CHUNK` | `512` | Token ceiling per code chunk |
| `TRELIX_PARSE_WORKERS` | `4` | Parallel parse workers at index time |
| `TRELIX_INCREMENTAL` | `true` | Skip files whose content hash is unchanged |
| `TRELIX_RETRIEVAL_TOP_K_VECTOR` | `20` | Candidates from the vector leg |
| `TRELIX_RETRIEVAL_TOP_K_BM25` | `20` | Candidates from the BM25 leg |
| `TRELIX_RETRIEVAL_RRF_K` | `60` | Reciprocal Rank Fusion `k` constant |
| `TRELIX_RETRIEVAL_RERANK` | `true` | Reranking on/off |
| `TRELIX_RETRIEVAL_RERANK_PROVIDER` | `cohere` | `cohere`, `cross_encoder`, `plaid`, `xtr` (underscore — `cross-encoder` is rejected) |
| `TRELIX_RETRIEVAL_RERANK_TOP_N` | `15` | Results kept after reranking |
| `COHERE_MODEL_RERANK` | `Cohere-rerank-v4.0-pro` | Cohere rerank model name |
| `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET` | `12000` | Context tokens sent to the LLM |
| `TRELIX_RETRIEVAL_FLARE` | `false` | Enable FLARE confidence-gated re-retrieval |
| `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` | `1` | FLARE synthesis-call budget (min `1`, max `3`) |
| `TRELIX_LLM_PROVIDER` | `openai` | Synthesis provider |
| `TRELIX_LLM_MODEL` | `gpt-4o` | Synthesis model |
| `TRELIX_TELEMETRY_ENABLED` | `false` | Record every `retrieve()` to the `query_telemetry` table |

### Names that do NOT exist

Earlier revisions of this guide listed the variables below. None of them bind to
anything — setting one is silently ignored, so a command appears to accept your setting
and then uses the default. Use the replacement instead:

| Not a variable | Use instead |
|---|---|
| `TRELIX_EMBEDDER` | `TRELIX_EMBEDDER_PROVIDER`. Setting the bare `TRELIX_EMBEDDER` does not merely no-op — it makes `IndexConfig` raise a pydantic `SettingsError`, because pydantic tries to parse it as the nested `embedder` model. |
| `TRELIX_LOCAL_MODEL` | `TRELIX_EMBEDDER_LOCAL_MODEL` |
| `TRELIX_VECTOR_BACKEND` | `TRELIX_STORE_BACKEND` |
| `TRELIX_INDEX_PATH` | `TRELIX_STORE_DB_PATH` |
| `TRELIX_LANCE_PATH` | `LANCE_URI` |
| `TRELIX_CHUNK_SIZE` | `TRELIX_CHUNKER_MAX_TOKENS_PER_CHUNK` |
| `TRELIX_RERANK` | `TRELIX_RETRIEVAL_RERANK` |
| `TRELIX_RERANK_TOP_K` | `TRELIX_RETRIEVAL_RERANK_TOP_N` |
| `TRELIX_FLARE_MAX_RETRIES` | `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` |
| `OPENAI_EMBEDDING_MODEL` | `TRELIX_EMBEDDER_OPENAI_MODEL` |
| `AZURE_DEPLOYMENT` | `AZURE_EMBEDDINGS_MODEL` (embeddings) or `AZURE_CHAT_MODEL` (synthesis) |
| `VOYAGE_MODEL` | `TRELIX_EMBEDDER_VOYAGE_MODEL` |
| `COHERE_RERANK_MODEL` | `COHERE_MODEL_RERANK` |
| `TRELIX_CHUNK_OVERLAP` | nothing — chunking follows AST boundaries; there is no token-overlap setting |
| `TRELIX_HYBRID_ALPHA` | nothing — legs are fused by Reciprocal Rank Fusion, not a dense/sparse blend weight. The nearest knob is `TRELIX_RETRIEVAL_RRF_K`. |
| `TRELIX_CACHE_DIR` | nothing — model caches follow the underlying libraries (e.g. `HF_HOME`); the index lives at `<repo>/.trelix/index.db` |
| `TRELIX_LOG_LEVEL` | nothing — use `trelix index --verbose` / `-v` |
| `TRELIX_WATCH_DEBOUNCE_MS` | nothing — **there is no debounce flag or environment variable.** `trelix watch` uses `FileWatcher(debounce_ms=500)` and `trelix watch-all` uses `MultiRepoWatcher(debounce_ms=1600)`, both hardcoded at the call site. Change it only from Python by constructing the watcher yourself. |

---

## 10. Upgrading from v2.3.x

### Step 1 — Install the new version

```bash
pip install --upgrade "trelix[local]"   # or whatever extras you use
trelix --version   # confirm it matches the version you installed
```

### Step 2 — Review breaking changes

#### `search_code` now returns a dict envelope

In v2.3.x `search_code` returned a plain list of results:

```python
# v2.3.x
results = index.search_code("auth middleware")
for r in results:
    print(r.chunk)
```

In v2.4.0 it returns a dict envelope:

```python
# v2.4.0
response = index.search_code("auth middleware")
for r in response["results"]:
    print(r.chunk)
# response keys: "results", "query", "embedder", "took_ms", "total"
```

Update every call site that unpacks the return value directly as a list.

#### `flare_max_iterations` renamed to `flare_max_retries`

The field on `RetrievalConfig` is now `flare_max_retries`. There is **no** deprecation
shim: the old name is simply ignored, as are the unprefixed spellings
`TRELIX_FLARE_MAX_ITERATIONS` and `TRELIX_FLARE_MAX_RETRIES`. The only name that binds is
the fully prefixed one:

```bash
# Neither of these does anything (silently ignored — no warning is emitted)
export TRELIX_FLARE_MAX_ITERATIONS=3
export TRELIX_FLARE_MAX_RETRIES=3

# The real variable
export TRELIX_RETRIEVAL_FLARE_MAX_RETRIES=3   # allowed range: 1-3, default 1
```

FLARE itself is off by default; enable it with `TRELIX_RETRIEVAL_FLARE=true`.

If you set the value in code rather than via an environment variable:

```python
from trelix.core.config import RetrievalConfig

# Old
RetrievalConfig(flare_max_iterations=3)

# New
RetrievalConfig(flare_enabled=True, flare_max_retries=3)
```

### Step 3 — Re-indexing

**No re-indexing is required.** Existing `.trelix/` indexes created with v2.3.x
are forward-compatible with v2.4.0. Trelix migrates the schema automatically on
first access.

If you want to rebuild the index from scratch (for example, to pick up
improvements to the default chunking strategy):

```bash
rm -rf .trelix/
trelix index ./
```

### Step 4 — Changelog

See [CHANGELOG.md](../CHANGELOG.md) for the full list of changes, fixes, and
new features introduced in v2.4.0.

---

## Getting Help

- **Documentation**: `docs/` directory in this repository
- **Issues**: [github.com/sairam0424/trelix/issues](https://github.com/sairam0424/trelix/issues)
- **Discussions**: [github.com/sairam0424/trelix/discussions](https://github.com/sairam0424/trelix/discussions)

Run `trelix --help` or `trelix <subcommand> --help` for inline usage reference.
