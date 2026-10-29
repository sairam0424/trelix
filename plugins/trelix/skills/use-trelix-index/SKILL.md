---
name: use-trelix-index
description: Search this repository with the trelix MCP tools (hybrid semantic + BM25 + call-graph search over a local index) instead of grep when the question is where or how something is implemented, what calls a symbol, or what a change would affect. Use when the user asks to find, explain, trace callers of, or assess the blast radius of code, and the repository has a trelix index (a .trelix/index.db file at its root).
allowed-tools: mcp__plugin_trelix_trelix__search_code mcp__plugin_trelix_trelix__get_symbol mcp__plugin_trelix_trelix__blast_radius
---

# Search with the trelix index

trelix answers "where is X implemented", "what calls Y" and "what would changing Z affect"
from an index it keeps in `<repo_path>/.trelix/index.db`. Every trelix tool takes
`repo_path`: the absolute path of the project root (never `~/...`, never a relative path).

## Index first

1. Read the line in the session context that starts with `trelix:` (the plugin's SessionStart
   hook writes it on startup, resume and `/clear`). It has one of three shapes:
   `trelix: no index at <repo>/.trelix/index.db. Call index_codebase(...) before search_code;
   until then, use grep.`; `trelix: the index at <repo>/.trelix/index.db is empty (0 files).
   Call index_codebase(...) before search_code; until then, use grep.`; or `trelix: <repo> is
   indexed: N files, M symbols; built <when> from commit <sha> (<distance>); embedder
   <provider>. Pass repo_path="<repo>" to every trelix tool.` Only the third means search, and
   its `<repo>` is the `repo_path` to pass. Without such a line (the hook did not run), check
   that `<repo_path>/.trelix/index.db` exists (for example `ls <repo_path>/.trelix/index.db`).
   A file that version 3.4.3 left behind (step 3) exists but is empty and answers `[]`
   (the SessionStart line calls it `empty (0 files)`); when in doubt, grep first.
2. If it does not exist, do not search. Ask the user before calling
   `mcp__plugin_trelix_trelix__index_codebase(repo_path)`: it embeds every symbol in the
   repository, which takes minutes on a large tree and costs money with an API embedder.
   Until the index exists, use grep.
3. On this release of the server (trelix-mcp 3.5.1) a search on a repository WITHOUT an
   index fails with a tool error, `No index found at <repo>/.trelix/index.db. Run trelix
   index <repo> first.`, and writes nothing. Read it as "not indexed", never as "not found",
   and index with `index_codebase` (step 2). Version 3.4.3 answered empty (or, with no
   embedder, a `sentence-transformers` error) and left an empty `.trelix/index.db` behind, so such a
   file can still exist (step 1).

## Workflow

- `mcp__plugin_trelix_trelix__search_code(query, repo_path, k=10, cursor=0)`: a question in
  plain language ("where are sessions validated"). Each result has `file`, `symbol`,
  `lines`, `kind`, `language`, `score`, `source` and `body`. The response carries `next_cursor`;
  pass it back as `cursor` for the next page, and stop when it is `null`. With
  `detail="concise"` each result drops `body` and carries a one-line `signature` instead: scan
  with that, then read the few symbols that matter with `get_symbol`.
- `mcp__plugin_trelix_trelix__get_symbol(qualified_name, repo_path)`: the full source of a
  symbol once you know its name from a search result. A body over 20,000 characters is cut
  and `body_truncated` is `true`; `max_body_chars=0` returns the whole body.
- `mcp__plugin_trelix_trelix__blast_radius(symbol_name, repo_path)`: the files that call or
  import a symbol, one entry per file (`file`, `symbol`, `kind`, `line_start`, `language`). Run
  it before you change a function, class or constant. It returns at most 100 entries; when it
  cuts the list, a second text block says how many were left out and what cut it. `limit` (up
  to 500) asks for more, but on its own it helps only when that block says the `limit` argument
  alone cut the list, not the server's output size cap.
- `mcp__plugin_trelix_trelix__graph_search_mcp(query, repo_path, k=10)`: call, import and
  type neighbours of a search hit. It rebuilds the whole graph (a full Louvain pass and
  PageRank) and rewrites graph metadata into the index on EVERY call: use it once per
  question and prefer `search_code` / `blast_radius` for follow-ups.
- `mcp__plugin_trelix_trelix__build_knowledge_graph(repo_path)`: architecture clusters for
  the whole repository. Writes graph metadata into the index.
- `mcp__plugin_trelix_trelix__ask_agent(query, repo_path, session_id)`: a question that
  needs several searches. It needs an LLM provider configured on the server and spends
  money; pass the returned `session_id` back to continue the same conversation.

Only `search_code`, `get_symbol` and `blast_radius` are pre-approved by this skill. The
others prompt for permission because they always write to the index or spend money.

## When to fall back to grep

- The index does not exist (step 1 above), or the user declined to index.
- The index is stale: a symbol you can see in the working tree is missing from every
  result, or the last index predates the commits you are working on (the SessionStart line
  says `HEAD is N commits ahead`). Say so, and either ask to re-index (`index_codebase` is
  incremental on an existing index) or grep.
- A tool error names `sentence-transformers`: the server has no local embedding model.
  Fall back to grep and point the user at the plugin README's embeddings section.
- You need an exact identifier match, a regular expression, or a file-name pattern:
  grep and glob do that directly; trelix ranks by meaning.

Never report "not found" from an empty trelix result alone. Confirm with grep first.
