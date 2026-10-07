# trelix: AGENTS.md snippet

Paste the block below into the AGENTS.md (or equivalent) of any agent that has the trelix MCP
tools. Tool names are shown bare; a Claude Code plugin user sees them as
`mcp__plugin_trelix_trelix__<tool>`.

````markdown
## Code search with trelix
- Every trelix tool takes `repo_path`: the absolute path of the project root.
- Index first: check that `<repo_path>/.trelix/index.db` exists. If not, ask before calling
  `index_codebase(repo_path)` (it embeds every symbol: minutes, and money with an API embedder).
- `search_code(query, repo_path)` for "where/how is X done" in plain language; pass
  `next_cursor` back as `cursor` for more results. `get_symbol(qualified_name, repo_path)` for a
  symbol's full source. `blast_radius(symbol_name, repo_path)` before changing a symbol.
  `graph_search_mcp(query, repo_path)` for call/import/type neighbours; `ask_agent` for
  multi-step questions (needs an LLM provider on the server; reuse its `session_id`).
- Fall back to grep when there is no index, when results look stale (a symbol you can see is
  missing), for exact identifiers or regular expressions, or when an error names
  `sentence-transformers`. Never report "not found" from an empty trelix result alone.
````
