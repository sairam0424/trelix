# Trelix v2.7.2 Troubleshooting Guide

This guide covers every common failure mode for trelix v2.7.2. Each entry follows the pattern: **Symptom → Cause → Fix**.

---

## Table of Contents

1. [Index Issues](#1-index-issues)
2. [Search Returns No Results](#2-search-returns-no-results)
3. [Embedding Provider Issues](#3-embedding-provider-issues)
4. [MCP Server Issues](#4-mcp-server-issues)
5. [GitHub PR Review Issues](#5-github-pr-review-issues)
6. [Federation Issues](#6-federation-issues)
7. [Performance Issues](#7-performance-issues)
8. [Python Version / Install Issues](#8-python-version--install-issues)

---

## 1. Index Issues

### "No such table: symbols"

**Symptom:** Any trelix command fails with `sqlite3.OperationalError: no such table: symbols` or a similar database schema error.

**Cause:** The `.trelix/` index directory does not exist or the database has never been initialized. trelix cannot run queries against an empty or missing index.

**Fix:**
```bash
# Run a full index of the current repository
trelix index

# Or index a specific path explicitly
trelix index /path/to/your/repo
```

---

### "DimensionMismatchError"

**Symptom:** Search or query commands fail with `DimensionMismatchError: expected 1536 dimensions, got 1024` (dimension numbers will vary by provider).

**Cause:** The index was originally built with one embedding provider (e.g., `openai` at 1536 dimensions) but the current configuration points to a different provider (e.g., `voyage` at 1024 dimensions). The stored vectors are incompatible with the new provider's output shape.

**Fix:**
```bash
# Reset all stored vectors and re-embed with the current provider
trelix migrate-vectors --reset

# If you want to switch providers permanently, set the provider first, then reset
trelix config set provider voyage
trelix migrate-vectors --reset
```

> **Note:** `--reset` discards all existing embeddings. The index metadata (file paths, symbols, AST data) is preserved — only the vector data is re-computed.

---

### Index is Stale

**Symptom:** Search results reference deleted functions or files, or newly added code is not found.

**Cause:** Files have changed on disk since the last `trelix index` run. The `.trelix/` database reflects an older snapshot of the repository.

**Fix:**

Option A — Incremental update (fast, updates only changed files):
```bash
trelix update-index
```

Option B — Continuous background watching (recommended for active development):
```bash
# Watch for file changes and update automatically
trelix watch
```

Option C — Full re-index (use when in doubt about index state):
```bash
trelix index --force
```

---

### Index DB Corrupted

**Symptom:** Commands fail with `sqlite3.DatabaseError: database disk image is malformed`, `database is locked`, or other SQLite integrity errors that persist after retrying.

**Cause:** The SQLite file inside `.trelix/` was corrupted — possible causes include interrupted writes, disk full events, or accidental truncation.

**Fix:**
```bash
# Delete the entire index directory and rebuild from scratch
rm -rf .trelix/
trelix index
```

> **Warning:** This deletes all cached embeddings and re-indexes from zero. On large repositories with a remote provider, this will consume API credits. Use `trelix index --provider local` for a cost-free baseline, then switch providers if needed.

---

## 2. Search Returns No Results

### Repo Not Indexed

**Symptom:** `trelix search "..."` returns `0 results` or `No index found for this repository`.

**Cause:** The repository has never been indexed. trelix requires an explicit indexing step before search is available.

**Fix:**
```bash
trelix index
```

Verify indexing completed successfully:
```bash
trelix stats
# Expected: shows file count, chunk count, symbol count
```

---

### Wrong Repo Path (Relative vs Absolute)

**Symptom:** Search returns no results even though you indexed the repo. `trelix stats` shows 0 files.

**Cause:** The index was built from one working directory but trelix is being invoked from a different directory, causing a path mismatch in the index key.

**Fix:**
```bash
# Always pass an absolute path to avoid ambiguity
trelix index /absolute/path/to/repo

# Then search from anywhere using the same path
trelix search "query" --repo /absolute/path/to/repo

# Or cd into the repo root before running trelix
cd /absolute/path/to/repo
trelix search "query"
```

---

### Provider Mismatch

**Symptom:** Search returns no results and `trelix stats` shows a non-zero chunk count. Logs may show `provider mismatch: index=local, active=azure`.

**Cause:** The index was built with one provider (e.g., `local`) but the active configuration specifies a different provider (e.g., `azure`). The query embedding has a different dimension or space than the stored vectors.

**Fix:**

Option A — Switch back to the provider used at index time:
```bash
trelix config set provider local
trelix search "query"
```

Option B — Re-index with the new provider:
```bash
trelix config set provider azure
trelix migrate-vectors --reset
trelix search "query"
```

Check which provider was used when the index was built:
```bash
trelix stats --verbose
# Shows: provider, dimensions, indexed-at timestamp
```

---

### File Type Not Supported

**Symptom:** Files exist in the repository but `trelix stats` shows a lower file count than expected. Searches for code in those files return nothing.

**Cause:** trelix only indexes supported file types (Python, TypeScript, JavaScript, Go, Rust, Java, C/C++, and others). Binary files, lock files, generated files, and unsupported languages are excluded.

**Fix:**
```bash
# Check what was actually indexed
trelix stats

# List all indexed file extensions
trelix stats --extensions

# If a supported language is missing, check .trelixignore or trelix.toml
cat .trelixignore
cat trelix.toml
```

To add a file pattern that was excluded by ignore rules:
```bash
# Edit trelix.toml to remove the exclusion
trelix config edit
```

---

## 3. Embedding Provider Issues

### OPENAI_API_KEY Not Set

**Symptom:** Indexing or search fails with `AuthenticationError: No API key provided` or `openai.AuthenticationError: Incorrect API key`.

**Cause:** The `OPENAI_API_KEY` environment variable is not exported in the current shell session.

**Fix:**
```bash
export OPENAI_API_KEY="sk-..."

# Verify it is set
echo $OPENAI_API_KEY

# For persistence, add to your shell profile
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc
source ~/.zshrc
```

---

### Azure Provider Missing Configuration

**Symptom:** Azure provider fails with `KeyError: 'AZURE_API_KEY'`, `ValueError: AZURE_ENDPOINT must be set`, or `AZURE_CHAT_MODEL is required`.

**Cause:** One or more required Azure OpenAI environment variables are missing.

**Required variables for the Azure provider:**

| Variable | Description | Example |
|---|---|---|
| `AZURE_API_KEY` | Azure OpenAI resource key | `abc123...` |
| `AZURE_ENDPOINT` | Azure OpenAI endpoint URL | `https://my-resource.openai.azure.com/` |
| `AZURE_CHAT_MODEL` | Deployment name for the chat/embedding model | `gpt-4o` |

**Fix:**
```bash
export AZURE_API_KEY="your-azure-key"
export AZURE_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_CHAT_MODEL="your-deployment-name"

# Verify
trelix config check --provider azure
```

---

### Voyage Missing VOYAGE_API_KEY

**Symptom:** Voyage provider fails with `voyage.AuthError: VOYAGE_API_KEY is not set`.

**Cause:** The `VOYAGE_API_KEY` environment variable is not exported.

**Fix:**
```bash
export VOYAGE_API_KEY="pa-..."

# Verify
echo $VOYAGE_API_KEY
```

---

### Bedrock: ValidationException on Inference Profile

**Symptom:** `ValidationException: Invocation of model ID anthropic.claude-sonnet-4-6 with on-demand throughput isn't supported`.

**Cause:** Bedrock requires **inference profile IDs** (`us.*` prefix), not bare model IDs, for on-demand throughput.

**Fix:**
```bash
TRELIX_LLM_BEDROCK_PRIMARY_MODEL=us.anthropic.claude-sonnet-4-6
TRELIX_LLM_BEDROCK_FALLBACK_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
```

---

### Bedrock Cohere Embeddings: ValidationException on Large Chunks

**Symptom:** `ValidationException: expected maxLength: 2048`.

**Cause:** Bedrock's Cohere endpoint rejects texts longer than 2048 characters before truncation occurs.

**Fix:** trelix pre-truncates automatically since v0.7.1. If you see this on an older version, upgrade:
```bash
pip install --upgrade "trelix[bedrock]"
```

---

### Local Provider Is Slow on First Run

**Symptom:** `trelix index --provider local` appears to hang or takes several minutes on the first invocation. CPU usage is high.

**Cause:** This is expected behavior. The local provider uses a small transformer model (e.g., `all-MiniLM-L6-v2`) which must be downloaded from Hugging Face on first use (~90 MB). Subsequent runs use the cached model and are significantly faster.

**Fix:** Wait for the download to complete. Progress will be shown in the terminal. After the first run the model is cached at `~/.cache/huggingface/` and startup time drops to under a second.

```bash
# Check cache size to confirm model is downloaded
du -sh ~/.cache/huggingface/
```

If the download stalls due to network issues:
```bash
# Set a mirror or proxy if needed
export HF_ENDPOINT="https://hf-mirror.com"
trelix index --provider local
```

---

### tree-sitter FutureWarning Spam

**Symptom:** Terminal flooded with messages like `FutureWarning: TreeSitter.Language` during indexing.

**Cause:** Language deprecation warnings from tree-sitter 0.21.x are not yet suppressed automatically.

**Fix:**
```bash
PYTHONWARNINGS=ignore::FutureWarning trelix index .
```

---

### HuggingFace Token Warning on Local Embedder

**Symptom:** A warning about a missing `HF_TOKEN` appears every time the local embedder runs.

**Cause:** The local embedder uses sentence-transformers, which checks for `HF_TOKEN` on every call. This is harmless — models are cached locally after first download and no token is required for public models.

**Fix:**
```bash
HF_HUB_DISABLE_SYMLINKS_WARNING=1 trelix index .
```

---

### DimensionGuard Firing After Provider Switch

**Symptom:** After changing `provider` in `trelix.toml` or via `trelix config set provider`, the next index or search raises `DimensionGuard: stored=1536, requested=1024. Run trelix migrate-vectors --reset`.

**Cause:** DimensionGuard is a safety mechanism that prevents silently mixing vectors from different embedding spaces. It fires whenever the new provider's output dimension does not match the dimension recorded in the index metadata.

**Fix:**
```bash
# Reset embeddings to match the new provider
trelix migrate-vectors --reset

# Confirm the guard is satisfied
trelix stats --verbose
# Look for: DimensionGuard: OK (provider=voyage, dims=1024)
```

---

## 4. MCP Server Issues

### "trelix-mcp command not found"

**Symptom:** Claude Code (or another MCP host) fails to start the server with `command not found: trelix-mcp` or `FileNotFoundError: [Errno 2] No such file or directory: 'trelix-mcp'`.

**Cause:** The `trelix-mcp` package is not installed in the Python environment that the MCP host uses to launch the server.

**Fix:**
```bash
# Install the MCP server package
pip install trelix-mcp

# If using a virtual environment, activate it first
source .venv/bin/activate
pip install trelix-mcp

# Verify the binary is on PATH
which trelix-mcp
trelix-mcp --version
```

If you are using `uv`:
```bash
uv pip install trelix-mcp
```

---

### Claude Code Not Picking Up MCP Server

**Symptom:** The `trelix-mcp` binary is installed and works standalone, but Claude Code does not show trelix tools in the available tool list.

**Cause:** The MCP server is either not registered with Claude Code, the registration is stale, or the config path is wrong.

**Fix:**
```bash
# Check currently registered MCP servers
claude mcp list

# If trelix is missing, re-register it
claude mcp add trelix-mcp trelix-mcp

# If it is listed but not working, remove and re-add
claude mcp remove trelix-mcp
claude mcp add trelix-mcp trelix-mcp

# Restart Claude Code after changes
```

For a custom binary path (e.g., inside a virtual environment):
```bash
claude mcp add trelix-mcp /path/to/.venv/bin/trelix-mcp
```

---

### stdout Pollution Breaking MCP Protocol

**Symptom:** MCP calls fail with JSON parse errors, `Unexpected token`, or `invalid JSON`. The MCP host logs show garbled output mixed into the JSON-RPC stream.

**Cause:** Something is writing to `stdout` — this could be a `print()` statement, a logging handler configured to output to stdout, or a dependency that writes startup banners. The MCP protocol uses `stdout` exclusively for JSON-RPC frames; any non-JSON bytes corrupt the transport.

**Diagnosis:**
```bash
# Run the server manually and watch for non-JSON output
trelix-mcp 2>/dev/null | head -5
# All output should be valid JSON-RPC frames, nothing else
```

**Fix:**

If you control the code emitting to stdout:
```python
# Wrong — breaks MCP
print("Debug info")

# Correct — write to stderr, which MCP ignores
import sys
print("Debug info", file=sys.stderr)
```

For logging configuration:
```python
import logging
import sys

# Direct all log output to stderr
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
```

If a third-party library is the source, redirect stderr and check:
```bash
trelix-mcp 2>/tmp/trelix-mcp-stderr.log
tail -f /tmp/trelix-mcp-stderr.log
```

---

### search_code Returns dict Not list (v2.4.0 Breaking Change)

**Symptom:** Callers that worked with trelix v2.3.x now receive a `dict` from `search_code` instead of a `list`, causing `TypeError: 'dict' object is not iterable` or attribute errors.

**Cause:** trelix v2.4.0 changed the `search_code` MCP tool response shape. The result is now a structured object:
```json
{
  "results": [...],
  "total": 42,
  "query_time_ms": 18
}
```
Prior versions returned a bare `list`.

**Fix:** Update all callers to access the `results` key:

```python
# v2.3.x — old pattern (broken in v2.4.0)
results = await mcp.call("search_code", {"query": "..."})
for item in results:
    print(item)

# v2.4.0 — new pattern
response = await mcp.call("search_code", {"query": "..."})
for item in response["results"]:
    print(item)
```

For Claude Code tool call handlers, update any skill or prompt that iterates the raw return value of `search_code`.

---

## 5. GitHub PR Review Issues

### "GITHUB_TOKEN not set"

**Symptom:** `trelix review pr <number>` fails with `EnvironmentError: GITHUB_TOKEN is not set` or `401 Unauthorized` from the GitHub API.

**Cause:** The `GITHUB_TOKEN` environment variable is not exported.

**Fix:**
```bash
export GITHUB_TOKEN="ghp_..."

# Verify
echo $GITHUB_TOKEN | cut -c1-8  # Should print "ghp_..." prefix only

# For persistence
echo 'export GITHUB_TOKEN="ghp_..."' >> ~/.zshrc
source ~/.zshrc
```

For CI/CD environments, set `GITHUB_TOKEN` as a secret in your pipeline configuration.

---

### 404 on PR

**Symptom:** `trelix review pr <number>` fails with `404 Not Found: Pull request not found`.

**Cause — Private repo, insufficient token scope:** The token was created without the `repo` scope (required for private repositories). Public repos only need `public_repo`.

**Cause — Wrong repo:** The repo was auto-detected from `git remote` but the remote points to a fork or a different repository than where the PR lives.

**Fix:**
```bash
# Verify token scopes (requires gh CLI)
gh auth status

# Check what scopes the token has
curl -s -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit \
  | grep -i "x-oauth-scopes" || true

# Explicitly specify the repo
trelix review pr <number> --repo owner/repo-name
```

To generate a token with the correct scope:
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens
2. Set **Repository access** to the target repo
3. Grant **Pull requests: Read** permission (and **Contents: Read** for private repos)

---

### 3000-File Truncation Warning

**Symptom:** `trelix review pr <number>` prints `Warning: PR contains 3000+ changed files. Results truncated. Review manually for full coverage.`

**Cause:** The GitHub API caps PR file listings at 3000 files. trelix enforces this ceiling and warns when it is hit. PRs of this size (e.g., large automated migrations, vendoring commits) cannot be fully reviewed programmatically.

**Fix:** There is no programmatic workaround for the GitHub API limit. Options:

1. Review the PR manually on GitHub.
2. Break the PR into smaller PRs with fewer than 3000 changed files each.
3. Use `trelix review pr <number> --paths "src/**"` to scope the review to a specific subdirectory.

---

## 6. Federation Issues

### "No repos registered"

**Symptom:** `trelix search-all "..."` or `trelix federation list` returns `No repos registered in federation`.

**Cause:** The federation registry is empty. Repos must be explicitly added before cross-repo search is available.

**Fix:**
```bash
# Register a repo with an alias
trelix federation add <alias> <path>

# Examples
trelix federation add backend /home/user/projects/backend
trelix federation add frontend /home/user/projects/frontend

# Verify
trelix federation list
```

Then index each repo:
```bash
trelix federation index-all
```

---

### DimensionMismatchError on search-all

**Symptom:** `trelix search-all "..."` fails with `DimensionMismatchError` or returns partial results with an error for some repos.

**Cause:** Different repos in the federation were indexed with different embedding providers, producing vectors of incompatible dimensions. Federation search requires all member repos to use the same provider and dimension.

**Fix:**
```bash
# Check which provider each repo used
trelix federation stats

# Re-index all repos with the same provider
trelix config set provider openai  # or your preferred provider
trelix federation index-all --reset

# Verify all repos show the same provider and dims
trelix federation stats --verbose
```

---

### watch-all Not Watching

**Symptom:** `trelix watch-all` exits immediately or prints `watchfiles not installed — file watching unavailable`.

**Cause:** The `watchfiles` package is an optional dependency. It is not installed by default and must be requested via the `watch` extra.

**Fix:**
```bash
pip install 'trelix[watch]'

# Or with uv
uv pip install 'trelix[watch]'

# Verify
python -c "import watchfiles; print(watchfiles.__version__)"

# Then retry
trelix watch-all
```

---

## 7. Performance Issues

### Indexing Is Slow

**Symptom:** `trelix index` takes several minutes for a medium-sized repository (e.g., 10k files).

**Cause — Remote provider rate limits:** OpenAI, Azure, and Voyage all have requests-per-minute and tokens-per-minute limits. Large repos hit these limits and the client backs off with delays.

**Cause — Low concurrency setting:** The default concurrency may be conservative for your network and API tier.

**Fix:**

Use the local provider for zero-latency, unlimited-throughput indexing:
```bash
trelix index --provider local
```

Increase embedding concurrency for remote providers:
```bash
export CC_EMBED_CONCURRENCY=16  # default is 4
trelix index
```

Check your API tier and increase limits if needed (OpenAI: Platform → Settings → Rate limits).

---

### Search Is Slow

**Symptom:** `trelix search "..."` takes more than 500 ms per query.

**Cause — Large index:** The default SQLite-based vector store performs a linear scan for approximate nearest neighbor search. For very large repos (>100k chunks), this becomes a bottleneck.

**Cause — Embedding latency:** If using a remote provider, each query requires an API round-trip to generate the query embedding.

**Fix:**

Check index size:
```bash
trelix stats
# If chunks > 100,000, consider switching to Qdrant
```

Switch to Qdrant for large indices (HNSW indexing, sub-millisecond queries):
```bash
# Start Qdrant locally
docker run -d -p 6333:6333 qdrant/qdrant

# Configure trelix to use Qdrant
trelix config set vector-store qdrant
trelix config set qdrant-url http://localhost:6333

# Re-index to populate Qdrant
trelix index --force
```

Switch to local provider to eliminate API round-trip latency:
```bash
trelix config set provider local
trelix migrate-vectors --reset
```

---

### watch Is Using Too Much CPU

**Symptom:** `trelix watch` causes sustained high CPU usage, especially in repos with many small files or high write frequency (e.g., build output directories).

**Cause:** By default, file-change events are debounced at 500 ms. If changes arrive faster than this (e.g., build tools writing many files in sequence), the debounce window may not be sufficient, leading to many rapid re-index cycles.

**Fix:**

Increase the debounce window:
```bash
trelix watch --debounce 2000  # 2 seconds, default is 500ms
```

Exclude noisy directories (build outputs, caches):
```bash
# Add to .trelixignore
echo "dist/" >> .trelixignore
echo "build/" >> .trelixignore
echo "__pycache__/" >> .trelixignore
echo "node_modules/" >> .trelixignore

trelix watch
```

Use `update-index` as a manual alternative if continuous watching is not required:
```bash
# Run once after a build instead of watching continuously
trelix update-index
```

---

## 8. Python Version / Install Issues

### Python Version Too Old

**Symptom:** Installation fails with `ERROR: Package 'trelix' requires a different Python: X.Y.Z not in '>=3.11'` or import errors about syntax features.

**Cause:** trelix requires Python 3.11 or later. Python 3.10 and older are not supported.

**Fix:**
```bash
# Check your current Python version
python --version
python3 --version

# Install Python 3.11+ via pyenv (recommended)
pyenv install 3.11.9
pyenv global 3.11.9

# Or via Homebrew on macOS
brew install python@3.11

# Verify
python3.11 --version
pip3.11 install trelix
```

---

### tree-sitter-languages Conflict

**Symptom:** Installation fails with a conflict like `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed` involving `tree-sitter-languages`, or import errors like `ImportError: cannot import name 'Language' from 'tree_sitter'`.

**Cause:** `tree-sitter-languages` has strict version requirements on the `tree-sitter` core package. If another package in your environment pinned a different version, there is a conflict.

**Fix:**
```bash
# Force reinstall tree-sitter-languages to resolve the conflict
pip install --force-reinstall tree-sitter-languages

# If that does not work, reinstall both packages cleanly
pip uninstall -y tree-sitter tree-sitter-languages
pip install tree-sitter-languages
pip install trelix
```

If the conflict persists, use a clean virtual environment (see below).

---

### sqlite-vec Not Loading (macOS)

**Symptom:** `ImportError: sqlite-vec requires SQLite ≥ 3.45 with loadable extensions`.

**Cause:** macOS ships with a system SQLite build that disables loadable extensions. sqlite-vec needs a SQLite build with extension loading enabled.

**Fix:**
```bash
brew install sqlite
# Then reinstall trelix against the Homebrew SQLite:
LDFLAGS="-L/opt/homebrew/opt/sqlite/lib" pip install --force-reinstall "trelix[local]"
```

---

### Virtual Environment Recommended

**Symptom:** trelix works on one machine but not another, or breaks after installing an unrelated package. Dependency conflicts are hard to diagnose.

**Cause:** Installing trelix into the system Python environment can cause version conflicts with other packages. Isolated environments prevent this entirely.

**Fix — Standard venv:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install trelix trelix-mcp
```

**Fix — uv (faster):**
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install trelix trelix-mcp
```

**Fix — pyenv + virtualenv:**
```bash
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install trelix
```

Verify the correct environment is active before running trelix:
```bash
which python     # Should point to .venv/bin/python
which trelix     # Should point to .venv/bin/trelix
trelix --version
```

---

## Quick Diagnostic Checklist

Run these commands to collect a baseline snapshot when something is wrong:

```bash
# System info
python --version
trelix --version
trelix-mcp --version 2>/dev/null || echo "trelix-mcp not installed"

# Index health
trelix stats --verbose

# Provider and config
trelix config show

# Environment variables (safe subset)
echo "OPENAI_API_KEY set: $([ -n "$OPENAI_API_KEY" ] && echo yes || echo NO)"
echo "AZURE_API_KEY set:  $([ -n "$AZURE_API_KEY" ] && echo yes || echo NO)"
echo "VOYAGE_API_KEY set: $([ -n "$VOYAGE_API_KEY" ] && echo yes || echo NO)"
echo "GITHUB_TOKEN set:   $([ -n "$GITHUB_TOKEN" ] && echo yes || echo NO)"

# MCP registration (requires Claude Code CLI)
claude mcp list 2>/dev/null || echo "claude CLI not available"
```

Paste this output when opening a bug report at https://github.com/your-org/trelix/issues.

---

## Still Stuck?

- Check the [GitHub Issues](https://github.com/your-org/trelix/issues) for known problems and workarounds.
- Run `trelix doctor` for an automated health check (available in v2.4.0+).
- Open a new issue with the output of the diagnostic checklist above.
