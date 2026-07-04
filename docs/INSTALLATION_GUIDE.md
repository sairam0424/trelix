# Trelix v2.4.0 — Installation Guide

This guide covers every installation scenario for Trelix v2.4.0, from a quick
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
| **Python 3.11 or 3.12** | Python 3.13 is not yet tested and may have compatibility issues |
| **pip** or **uv** | pip ships with Python; uv is optional but significantly faster |
| **~500 MB free disk** | The local embedder model is downloaded on first use and cached in `~/.cache/trelix/` |
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
trelix --version   # should print 2.4.0
```

On first use, Trelix downloads the embedder model (~420 MB) to
`~/.cache/trelix/models/`. Subsequent runs use the cached copy.

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

Set `TRELIX_EMBEDDER=openai` (or pass `--embedder openai`) to activate.

### 3.3 Azure OpenAI embeddings

Requires an Azure OpenAI resource with an embeddings deployment.

```bash
pip install trelix
export AZURE_API_KEY="..."
export AZURE_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_DEPLOYMENT="text-embedding-3-large"   # your deployment name
```

Set `TRELIX_EMBEDDER=azure` to activate.

### 3.4 MCP server (Claude Code / Cursor integration)

Exposes Trelix as an MCP tool your AI assistant can call directly.

```bash
pip install trelix-mcp
trelix-mcp --help
```

See `docs/integrations/mcp.md` for the full Claude Code / Cursor setup.

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
trelix serve --port 8080 --repo ./
```

The OpenAPI spec is available at `http://localhost:8080/docs`.

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
export TRELIX_VECTOR_BACKEND=lance
trelix index ./
```

LanceDB stores data in `~/.cache/trelix/lance/` by default (override with
`TRELIX_LANCE_PATH`).

### 3.11 Reranking (Cohere)

Cross-encoder reranking improves precision on ambiguous queries.

```bash
pip install "trelix[rerank]"
export COHERE_API_KEY="..."
export TRELIX_RERANK=true
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
[GitHub Releases](https://github.com/sairam0424/trelix/releases/tag/v2.4.0)
page for each platform. No Python or pip required.

### macOS ARM64 (Apple Silicon)

```bash
curl -L https://github.com/sairam0424/trelix/releases/download/v2.4.0/trelix-macos-arm64 \
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
curl -L https://github.com/sairam0424/trelix/releases/download/v2.4.0/trelix-linux-x64 \
     -o /usr/local/bin/trelix
chmod +x /usr/local/bin/trelix
trelix --version
```

Binaries are built with PyInstaller and include all Python dependencies. The
local embedder model is still downloaded to `~/.cache/trelix/` on first use.

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
  ghcr.io/sairam0424/trelix:2.4.0 \
  index /repo
```

The index is written to `/repo/.trelix/` inside the container (which maps to
`$(pwd)/.trelix/` on your host).

### Start the REST server

```bash
docker run --rm -p 8080:8080 \
  -v "$(pwd):/repo" \
  ghcr.io/sairam0424/trelix:2.4.0 \
  serve --repo /repo --port 8080
```

Then open `http://localhost:8080/docs` for the interactive API reference.

### Use with OpenAI embeddings

```bash
docker run --rm \
  -e OPENAI_API_KEY="sk-..." \
  -e TRELIX_EMBEDDER=openai \
  -v "$(pwd):/repo" \
  ghcr.io/sairam0424/trelix:2.4.0 \
  index /repo
```

### Docker Compose example

```yaml
# docker-compose.yml
services:
  trelix:
    image: ghcr.io/sairam0424/trelix:2.4.0
    command: serve --repo /repo --port 8080
    ports:
      - "8080:8080"
    volumes:
      - .:/repo
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - TRELIX_EMBEDDER=openai
```

```bash
docker compose up
```

---

## 8. Verify Installation

Run these commands after any installation method to confirm everything is
working correctly.

```bash
# Print version (must show 2.4.0)
trelix --version

# Print usage summary
trelix --help

# Index the current directory (creates .trelix/ index)
trelix index ./

# Show index statistics
trelix stats ./

# Run a test search query
trelix search "hybrid search" --repo ./
```

Expected output for `trelix stats ./`:

```
Trelix Index Stats
  Version   : 2.4.0
  Chunks    : <n>
  Embedder  : local (all-MiniLM-L6-v2)
  Backend   : sqlite
  Index path: ./.trelix/
```

If `trelix --version` prints nothing or fails, check that:

1. Your virtual environment is activated (if using one).
2. The Python executable that installed Trelix is on your `PATH`.
3. You are using Python 3.11 or 3.12 (`python --version`).

---

## 9. Environment Variables Reference

All variables can be placed in a `.env` file at the repository root; Trelix
loads it automatically via `python-dotenv`.

| Variable | Default | Description |
|----------|---------|-------------|
| `TRELIX_EMBEDDER` | `local` | Embedder backend. Options: `local`, `openai`, `azure`, `voyage` |
| `TRELIX_LOCAL_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model name for the local embedder |
| `TRELIX_CACHE_DIR` | `~/.cache/trelix` | Directory for cached models and index data |
| `TRELIX_INDEX_PATH` | `.trelix/` | Path (relative to repo root) where the index is stored |
| `TRELIX_VECTOR_BACKEND` | `sqlite` | Vector store backend. Options: `sqlite`, `lance`, `qdrant` |
| `TRELIX_LANCE_PATH` | `~/.cache/trelix/lance` | Storage path when `TRELIX_VECTOR_BACKEND=lance` |
| `TRELIX_CHUNK_SIZE` | `512` | Token size for code chunk splitting |
| `TRELIX_CHUNK_OVERLAP` | `64` | Token overlap between consecutive chunks |
| `TRELIX_HYBRID_ALPHA` | `0.7` | Dense/sparse blend weight (1.0 = dense only, 0.0 = sparse only) |
| `TRELIX_RERANK` | `false` | Enable cross-encoder reranking. Set to `true` to activate |
| `TRELIX_RERANK_TOP_K` | `5` | Number of results to return after reranking |
| `TRELIX_FLARE_MAX_RETRIES` | `3` | Maximum FLARE loop iterations (renamed from `flare_max_iterations` in v2.4.0) |
| `TRELIX_LOG_LEVEL` | `WARNING` | Log verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `TRELIX_SERVE_HOST` | `0.0.0.0` | Host for the REST API server |
| `TRELIX_SERVE_PORT` | `8080` | Port for the REST API server |
| `TRELIX_WATCH_DEBOUNCE_MS` | `500` | Debounce delay (ms) for file-watch re-indexing |
| `OPENAI_API_KEY` | _(none)_ | OpenAI API key; required when `TRELIX_EMBEDDER=openai` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model name |
| `AZURE_API_KEY` | _(none)_ | Azure OpenAI API key; required when `TRELIX_EMBEDDER=azure` |
| `AZURE_ENDPOINT` | _(none)_ | Azure OpenAI resource endpoint URL |
| `AZURE_DEPLOYMENT` | _(none)_ | Azure OpenAI embeddings deployment name |
| `AZURE_API_VERSION` | `2024-02-01` | Azure OpenAI API version |
| `VOYAGE_API_KEY` | _(none)_ | Voyage AI API key; required when `TRELIX_EMBEDDER=voyage` |
| `VOYAGE_MODEL` | `voyage-code-2` | Voyage embedding model name |
| `COHERE_API_KEY` | _(none)_ | Cohere API key; required when `TRELIX_RERANK=true` |
| `COHERE_RERANK_MODEL` | `rerank-english-v3.0` | Cohere reranking model name |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint; required when `TRELIX_VECTOR_BACKEND=qdrant` |
| `QDRANT_API_KEY` | _(none)_ | Qdrant API key for cloud deployments |
| `QDRANT_COLLECTION` | `trelix` | Qdrant collection name |

---

## 10. Upgrading from v2.3.x

### Step 1 — Install the new version

```bash
pip install --upgrade "trelix[local]"   # or whatever extras you use
trelix --version   # confirm 2.4.0
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

The environment variable `TRELIX_FLARE_MAX_ITERATIONS` still works but emits a
`DeprecationWarning`. Migrate to the new name before v2.5.0:

```bash
# Old (still works in 2.4.0 with a warning)
export TRELIX_FLARE_MAX_ITERATIONS=5

# New (preferred)
export TRELIX_FLARE_MAX_RETRIES=5
```

If you set the value in code rather than via an environment variable:

```python
# Old
index = TrelixIndex(flare_max_iterations=5)

# New
index = TrelixIndex(flare_max_retries=5)
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
