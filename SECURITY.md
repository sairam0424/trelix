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
