# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.x.x   | ✅ Yes |
| 0.7.x   | ✅ Yes (security fixes only) |
| < 0.7   | ❌ No |

## Reporting a Vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.**

Please report security vulnerabilities by emailing:
**uggesairam0000@gmail.com**

Include in your report:
- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fix (optional)

We will:
- Acknowledge receipt within **48 hours**
- Provide a status update within **7 days**
- Work with you on a **coordinated disclosure timeline** (typically 90 days)
- Credit you in the release notes (unless you prefer anonymity)

## Scope

trelix processes local repository contents and makes network calls to configured LLM/embedding providers. Security-sensitive areas include:

- **Credential handling** — API keys are read from environment variables and never logged or written to disk
- **MCP server** (trelix-mcp) — executes as a subprocess; only binds to stdio transport (not network)
- **File system access** — reads only files under the indexed `repo_path`; does not follow symlinks outside the repo boundary
- **Tree-sitter parsing** — parses user code with C-extension parsers; malformed inputs are caught and logged

### REST API — /graph/visualize output path constraint

The `output` query parameter on `GET /graph/visualize` is validated server-side:
- The resolved output path must be inside `<repo>/.trelix/`
- Paths outside this directory are rejected with HTTP 400
- This prevents arbitrary file writes to sensitive locations

### MCP federation tools — config_path confinement (v2.8.1+)

The `config_path` parameter accepted by `federation_list_repos`,
`federation_add_repo`, `federation_remove_repo`, and `federation_search_all`
is validated server-side:
- The resolved path must be inside `~/.config/trelix/` or `<mcp-server-cwd>/.trelix/`
- Paths outside these roots are rejected with an `{"error": ...}` response
  (never raises — matches the "never raise" convention of every other
  federation tool)
- Unlike the `/graph/visualize` check above, this uses `Path.is_relative_to()`
  rather than a string-prefix comparison, which avoids a sibling-directory
  bypass (e.g. `~/.config/trelixevil/` would incorrectly pass a naive
  `startswith("~/.config/trelix")` check)
- Also caps the federation registry at `TRELIX_FEDERATION_MAX_REPOS`
  (default 50) entries and the number of repos actually queried per
  `federation_search_all` call, preventing a scripted/adversarial MCP
  client from growing the registry or fan-out unboundedly

## Out of Scope

- Vulnerabilities in third-party dependencies (report to upstream)
- Denial-of-service via extremely large repositories (use `--limit` flags)
- Issues requiring physical access to the machine

## v2.1.0 Security Notes

### Query Telemetry (`telemetry_enabled`)

- **Storage**: Telemetry data (query text, intent, elapsed_ms, result_count) is stored locally in `.trelix/index.db` only
- **No external transmission**: All telemetry is SQLite-only; no data leaves the machine or contacts external services
- **Sensitive query strings**: If your codebase contains secrets in symbol names or comments, those strings may appear in query telemetry logs
- **Mitigation**: Disable telemetry via environment variable `TRELIX_TELEMETRY_ENABLED=false` (default is enabled in v2.1.0)

### Eval Golden Files (`trelix eval --golden`)

- **File content**: Golden JSONL files may contain internal query strings and evaluation assertions
- **Treat as internal documentation**: Golden files contain test data and should not be committed to public repositories
- **Recommended storage**: Store golden files in `.trelix/` directory (already gitignored) rather than repo root to prevent accidental disclosure

### HyDE Synthetic Snippets

- **Transient generation**: HyDE generates synthetic code snippets via LLM calls; these snippets are never persisted to disk
- **LLM exposure**: Only the normal LLM provider data-transmission path applies (i.e., no additional sensitive data is sent beyond what standard similarity search already sends to your LLM provider)
- **No local storage**: Synthetic snippets are embedded transiently in memory for ranking; they are discarded after the search completes

## v2.2.0 Security Notes

### Agentic Loop Security
- AgentLoop executes `retrieve`, `grep`, and `get_symbol` actions only — no code execution, no shell commands
- All actions read from the indexed SQLite DB; no external network calls during agent turns
- `max_results` on grep is capped at 50 (enforced in loop.py._do_grep)
- Disable: `TRELIX_RETRIEVAL_AGENTIC=false` (default)

### Taint Analysis Security
- TaintAnalyzer runs the Semgrep CLI via subprocess with a 120-second timeout
- Semgrep operates on local files only; no data leaves the machine
- Rule files: use `--rules <path>` to restrict to known-good rules; default uses Semgrep registry (requires internet)
- Results stored in `taint_flows` SQLite table (local, not transmitted)
- Disable: `TRELIX_PARSER_TAINT=false` (default)

### Sparse Embeddings
- SPLADE-Code model weights loaded from HuggingFace at first use (internet required once, cached locally)
- Sparse vectors stored in `sparse_embeddings` SQLite table (local only)
- No query data sent externally when using local inference
