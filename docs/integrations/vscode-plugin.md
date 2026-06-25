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
trelix index <workspace-path> --provider <local|openai|azure> -v
```

### Incremental file update

```bash
trelix update-index <workspace-path> <changed-file> --provider <provider>
```

---

## Environment variables

| Variable | Purpose |
|---|---|
| `TRELIX_EMBEDDER_PROVIDER` | Provider: `local` \| `openai` \| `azure` |
| `OPENAI_API_KEY` | API key for the `openai` provider |
| `AZURE_API_KEY` / `AZURE_ENDPOINT` | Credentials for the `azure` provider |
| `TRELIX_STORE_DB_PATH` | Path where trelix writes the SQLite index |

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

| `--provider` | Embedding backend |
|---|---|
| `local` | sentence-transformers — no network call, no token required |
| `openai` | OpenAI Embeddings API (`text-embedding-3-large`) |
| `azure` | Azure OpenAI Embeddings |
