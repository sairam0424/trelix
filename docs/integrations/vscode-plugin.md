# trelix in VS Code

## Overview

A VS Code extension can use trelix as a bundled binary to provide code
intelligence features — indexing, semantic search, BM25 search, and
call-graph traversal — entirely on the developer's machine.

The binary is compiled as a single self-contained executable via PyInstaller
and embedded in the extension package. No Python installation is required on
the user's machine.

---

## Building the binary

From the trelix repo root:

```bash
make binary
```

This runs `scripts/build-binary.sh`, which:

1. Creates / activates `.venv` (Python 3.11+).
2. Installs `trelix[local,dev]` + PyInstaller.
3. Runs `pyinstaller trelix.spec --clean --noconfirm`.
4. Smoke-tests the result with `dist/trelix --help`.

**Output:** `dist/trelix` (macOS arm64 / Linux x64) or `dist/trelix.exe`
(Windows x64 when built on Windows).

---

## Embedding the binary in an extension

Place the compiled binary in your extension's assets directory, then resolve
it at runtime based on `process.platform`:

```typescript
const binName = process.platform === 'win32' ? 'trelix.exe' : 'trelix';
const binaryPath = path.join(context.extensionPath, 'src', 'assets', 'bin', binName);
```

---

## CLI commands

Two commands cover the full indexing and incremental-update lifecycle:

### Full index

```bash
trelix index <workspace-path> --provider <local|openai|azure|voyage|local-code|bedrock-titan|bedrock-cohere> -v
```

### Incremental file update

```bash
trelix update-index <workspace-path> <changed-file> --provider <provider>
```

---

## Environment variables

### Embedding providers

| Variable | Purpose |
|---|---|
| `TRELIX_EMBEDDER_PROVIDER` | Provider: `local` \| `openai` \| `azure` \| `voyage` \| `local-code` \| `bedrock-titan` \| `bedrock-cohere` |
| `OPENAI_API_KEY` | API key for the `openai` provider |
| `AZURE_API_KEY` / `AZURE_ENDPOINT` | Credentials for the `azure` provider |
| `VOYAGE_API_KEY` | API key for the `voyage` provider |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` | AWS credentials for `bedrock-titan` and `bedrock-cohere` providers |
| `TRELIX_STORE_DB_PATH` | Path where trelix writes the SQLite index |

### LLM / chat providers (v0.7.0)

| Variable | Purpose |
|---|---|
| `TRELIX_LLM_PROVIDER` | Chat provider: `openai` \| `azure` \| `anthropic` \| `bedrock` \| `vertex` \| `litellm` (default: `openai`) |
| `TRELIX_LLM_MODEL` | Model override (e.g. `gpt-4o`, `claude-sonnet-4-6`) |
| `ANTHROPIC_API_KEY` | API key for the `anthropic` provider |
| `TRELIX_LLM_BEDROCK_PRIMARY_MODEL` | Bedrock primary model (default: `us.anthropic.claude-sonnet-4-6`) |
| `TRELIX_LLM_BEDROCK_FALLBACK_MODEL` | Bedrock fallback model (default: `us.anthropic.claude-haiku-4-5-20251001-v1:0`) |
| `GOOGLE_CLOUD_PROJECT` / `GOOGLE_API_KEY` | Credentials for the `vertex` provider |

---

## Database schema

trelix writes a SQLite file (`.trelix/index.db` by default) with this schema:

| Table | Purpose |
|---|---|
| `symbols` | AST-extracted symbol records |
| `symbols_fts` | FTS5 full-text index |
| `chunks` | Text chunks with embedding vectors |
| `call_graph` | Caller → callee edges |
| `imports` | Module import relationships |
| `type_edges` | Type hierarchy edges |
| `vec0` | sqlite-vec virtual table for ANN search |

---

## Provider mapping

### Embedding providers

| `TRELIX_EMBEDDER_PROVIDER` | Embedding backend | Dims | Notes |
|---|---|---|---|
| `local` | sentence-transformers (`all-MiniLM-L6-v2`) | 384 | No network call, no API key required |
| `openai` | OpenAI Embeddings API (`text-embedding-3-large`) | 3072 | Requires `OPENAI_API_KEY` |
| `azure` | Azure OpenAI Embeddings | 3072 | Requires `AZURE_API_KEY` + `AZURE_ENDPOINT` |
| `voyage` | Voyage AI (`voyage-code-3`) | 1024 | Best API-based code model; requires `VOYAGE_API_KEY` |
| `local-code` | Salesforce `SFR-Embedding-Code-2B_R` | 4096 | No API key; requires ~8 GB RAM/GPU |
| `bedrock-titan` | AWS Bedrock Titan v2 (`amazon.titan-embed-text-v2:0`) | 256/512/1024 | Requires AWS credentials; `pip install trelix[bedrock]` |
| `bedrock-cohere` | AWS Bedrock Cohere English v3 (`cohere.embed-english-v3`) | 1024 | Asymmetric doc/query retrieval; requires AWS credentials; `pip install trelix[bedrock]` |

### LLM / chat providers (v0.7.0)

| `TRELIX_LLM_PROVIDER` | Backend | Notes |
|---|---|---|
| `openai` | OpenAI API | Default; requires `OPENAI_API_KEY` |
| `azure` | Azure OpenAI | Requires `AZURE_API_KEY` + `AZURE_ENDPOINT` |
| `anthropic` | Anthropic Claude direct | Requires `ANTHROPIC_API_KEY`; `pip install trelix[anthropic]` |
| `bedrock` | AWS Bedrock Converse API | Defaults to `us.anthropic.claude-sonnet-4-6` with auto-fallback to Haiku; requires AWS credentials; `pip install trelix[bedrock]` |
| `vertex` | Google Vertex AI / Gemini | Requires `GOOGLE_CLOUD_PROJECT` or `GOOGLE_API_KEY`; `pip install trelix[vertex]` |
| `litellm` | LiteLLM (100+ providers) | Model strings: `bedrock/claude-3-5-sonnet`, `gemini/gemini-2.0-flash`; `pip install trelix[litellm]` |
