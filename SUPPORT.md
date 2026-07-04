# Getting Support

## Questions & General Help

For questions about using trelix, open a [GitHub Discussion](https://github.com/sairam0424/trelix/discussions). This is the best place for:

- "How do I set up hybrid search for my use case?"
- "What's the difference between sparse and dense retrieval?"
- "How do I connect trelix to an existing vector store?"
- "How do I tune relevance scoring?"

## Bug Reports

Found a bug? Open a [GitHub Issue](https://github.com/sairam0424/trelix/issues/new?template=bug_report.md) using the bug report template. Include:

- trelix version (`trelix --version`)
- Operating system and version
- Python version (`python --version`)
- Steps to reproduce (minimal, self-contained)
- `.trelix/index.db` size (`ls -lh .trelix/index.db` or equivalent)
- Full error message and stack trace
- Expected vs actual behavior

The index database size helps distinguish corruption issues from logic bugs — always include it.

## Feature Requests

Have an idea? Open a [GitHub Issue](https://github.com/sairam0424/trelix/issues/new?template=feature_request.md) using the feature request template. Describe the use case, not just the feature — what problem are you trying to solve and why does the current behavior fall short?

## Security Vulnerabilities

**Do not open a public issue for security vulnerabilities.** See [SECURITY.md](SECURITY.md) for how to report them privately. Security reports receive priority handling and acknowledgement within 2 business days.

## Self-Hosted Troubleshooting

Before opening an issue, work through this checklist:

1. **Index exists?**
   ```bash
   ls -lh .trelix/index.db
   ```
   If the file is missing, run `trelix index` to build or rebuild the index.

2. **Provider environment variables set?**
   Check that the embedding provider API key or endpoint is exported. For example:
   ```bash
   echo $OPENAI_API_KEY       # OpenAI embeddings
   echo $COHERE_API_KEY       # Cohere embeddings
   echo $TRELIX_PROVIDER      # active provider name
   ```
   trelix will error silently or fall back to a degraded mode if these are absent.

3. **Index not corrupted?**
   Run the built-in integrity check:
   ```bash
   trelix index --check
   ```
   If it reports errors, rebuild:
   ```bash
   trelix index --rebuild
   ```

4. **Search returning no results or wrong results?**
   - Confirm the index was built against the current corpus (`trelix status`).
   - Check that the query language/encoding matches the indexed content.
   - Try `trelix search --debug "<query>"` for per-stage scoring output.

5. **MCP server not responding?**
   Verify the server process is running and the port is not blocked:
   ```bash
   trelix mcp status
   ```

## Response Times

| Type | Expected Response |
|------|------------------|
| Security vulnerability | 2 business days acknowledgement |
| Bug report (critical — data loss / index corruption) | Best effort, prioritised |
| Bug report (general) | Best effort |
| Feature request | Best effort |
| Questions | Community-driven |

Response times are best-effort for an open-source project. The fastest path to a resolution is always a minimal reproduction case.

## Community

- **GitHub Discussions** — [github.com/sairam0424/trelix/discussions](https://github.com/sairam0424/trelix/discussions): questions, ideas, show-and-tell
- **GitHub Issues** — [github.com/sairam0424/trelix/issues](https://github.com/sairam0424/trelix/issues): confirmed bugs and tracked feature requests
- **Releases** — [github.com/sairam0424/trelix/releases](https://github.com/sairam0424/trelix/releases): changelogs and migration notes for each version
