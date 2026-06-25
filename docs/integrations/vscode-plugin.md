# trelix in VS Code (Drop-in for aava-core-vscode-ide-plugin)

## How it works

The `aava-core-vscode-ide-plugin` bundles a precompiled binary at:

```
src/assets/bin/codeindex        # macOS / Linux
src/assets/bin/codeindex.exe    # Windows
```

trelix compiles to the **exact same binary name and CLI interface** — it is a
drop-in replacement. No plugin source changes are required; you only swap the
binary.

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
4. Smoke-tests the result with `dist/codeindex --help`.

**Output:** `dist/codeindex` (macOS arm64 / Linux x64) or `dist/codeindex.exe`
(Windows x64 when built on Windows).

---

## Replacing the plugin binary

### macOS / Linux

```bash
cp dist/codeindex /path/to/aava-core-vscode-ide-plugin/src/assets/bin/codeindex
chmod +x /path/to/aava-core-vscode-ide-plugin/src/assets/bin/codeindex
```

Or install system-wide (macOS only):

```bash
make binary-install   # sudo-copies to /usr/local/bin/codeindex
```

### Windows

```bat
copy dist\codeindex.exe \path\to\aava-core-vscode-ide-plugin\src\assets\bin\codeindex.exe
```

The plugin selects the binary via a `process.platform` check in
`python-engine.ts`, so the platform-correct filename is picked automatically
at runtime.

---

## CLI interface the plugin uses

The plugin (`python-engine.ts`) issues exactly two commands:

### Full index

```bash
codeindex index <workspace-path> \
    --provider <aava|azure|openai|local> \
    -v
```

### Incremental file update

```bash
codeindex update-index <workspace-path> <changed-file> \
    --provider <provider>
```

Both commands read configuration from the environment (see below). trelix
accepts the identical flags and positional arguments.

---

## Environment variables

| Variable | Plugin role | trelix mapping |
|---|---|---|
| `EMBEDDING_BEARER_TOKEN` | Bearer token sent to the embedding endpoint | Passed directly to the `aava` provider HTTP client as the `Authorization: Bearer` header |
| `EMBEDDING_BASE_URL` | Base URL of the embedding service (e.g. `https://api.aava.ai/v1`) | Used by `aava`, `azure`, and `openai` providers as the API base URL |
| `CODEINDEX_STORE_DB_PATH` | Absolute path where the plugin expects the SQLite DB to be written | Forwarded internally as `TRELIX_STORE_DB_PATH` — **note this name difference**. If `CODEINDEX_STORE_DB_PATH` is set trelix reads it; set `TRELIX_STORE_DB_PATH` to override |

> **Compatibility note:** trelix reads `CODEINDEX_STORE_DB_PATH` as an alias
> for `TRELIX_STORE_DB_PATH` so the plugin works out of the box without any
> environment changes.

---

## Database schema compatibility

trelix writes the identical SQLite schema that the plugin expects:

| Table | Purpose |
|---|---|
| `symbols` | AST-extracted symbol records (name, kind, location, doc) |
| `symbols_fts` | FTS5 full-text index over `symbols` |
| `chunks` | Text chunks with embedding vectors |
| `call_graph` | Caller → callee edges |
| `imports` | Module import relationships |
| `type_edges` | Type hierarchy / usage edges |
| `vec0` | sqlite-vec virtual table for ANN search |

The plugin can open a database produced by trelix (and vice-versa) without
migration.

---

## Provider mapping

| `--provider` value | Embedding backend |
|---|---|
| `aava` | Aava platform embedding service (uses `EMBEDDING_BASE_URL` + `EMBEDDING_BEARER_TOKEN`) |
| `azure` | Azure OpenAI Embeddings (uses `EMBEDDING_BASE_URL` as the Azure endpoint) |
| `openai` | OpenAI Embeddings API (`text-embedding-3-small` by default) |
| `local` | Local sentence-transformers model — no network call, no token required |

The `local` provider is useful for offline development or CI environments where
no embedding service is available.
