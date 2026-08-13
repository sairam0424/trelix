# Changelog

All notable changes to trelix are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **`trelix ask`, `search`, `query`, `call-graph`, `review`, `taint`,
  `agent sessions show` and six more commands crashed or silently corrupted
  their output when rendering ordinary indexed content.** The CLI's module-level
  `Console()` has Rich markup enabled, so every string reaching
  `console.print()`, `Table.add_row()`, a Table title or a Panel body was parsed
  for `[tag]` console markup — and most of what these commands render is
  arbitrary text: indexed file paths and symbol names, retrieved source code,
  LLM answers and agent observations quoting that code, Semgrep findings, GitHub
  PR filenames, and persisted queries.

  Two distinct failure modes, neither requiring an attacker. An unmatched
  closing tag raises `MarkupError`, so the command renders **nothing** and exits
  nonzero; a balanced-looking tag pair is silently swallowed, so the command
  exits 0 having **dropped characters from the value**. trelix's own source trips
  the first: `src/trelix/indexing/parser/extractors/rust.py` contains
  `re.sub(r"^//[/!]?\s?", ...)`, whose `[/!]` is an unmatched closing tag — so
  `trelix ask` against any repository holding a Rust comment-stripping regex,
  trelix included, died instead of showing results.

  Fixed by escaping the value at every markup-interpreting sink — 73 new
  `escape()` call sites across 22 sink groups — while leaving trelix's own
  `[red]…[/red]` markup unescaped so colouring still works. The worst sink was
  `agent sessions show`, which replays LLM thoughts and tool observations, and a
  tool observation *is* retrieved repository source. `trelix audit list`'s Table
  **title** was also still unsafe: that command's earlier fix escaped every row
  but not the title, so `--db '/tmp/a[/x].db'` still raised.

- **`trelix review --json`, `taint --json` and `graph --json` emitted
  unparseable JSON.** These payloads went out through `console.print()`, which is
  a markup sink like any other, so they inherited both failure modes plus a
  third: Rich hard-wraps at the console width, and a wrap landing inside a JSON
  *string* injects a raw newline that `json.loads` rejects as `Invalid control
  character`. One long unbroken token in an LLM review comment was enough.
  Whitespace *between* JSON tokens is semantically free, which is why the
  existing `--json` contract tests — all using short, bracket-free comments —
  never caught it.

  Escaping is the wrong fix for a machine-readable payload (it would write stray
  backslashes into consumers' parsed strings), so these seven sites now go
  through a `_print_json()` helper that disables Rich's markup parser, syntax
  highlighter and line wrapping instead of altering the value. Output is
  byte-identical for payloads that already worked.

- **Nine functions in `cli/main.py` carried a redundant function-local
  `from rich.markup import escape`.** A function-local import binds the name for
  the *entire* function scope, so any `escape()` call above the import line
  raises `UnboundLocalError` — a live trap for exactly the kind of change made
  above, and one `mypy` does not detect (`ruff`'s F823 does). All nine removed;
  the module-level import is now the single source. Four dead
  `import json as _json` locals went with them.

### Added

- 13 regression tests (`tests/unit/test_cli_markup_safety.py`, plus two in
  `tests/unit/test_review_pr_json.py`). Every test asserts the payload's
  **literal characters** appear in the output rather than merely that no
  exception was raised — a command that renders nothing also raises nothing, and
  the silent-swallow mode exits 0. The trigger line is read out of
  `extractors/rust.py` at test time rather than pasted, so the tests cannot
  drift from the code they pin. 11 of the 13 fail against the previous release;
  the other two are negative controls asserting `--json` output stays unescaped.

## [3.0.1] — 2026-08-13

### Overview
A correctness release. One defect, but a total one: every call edge, parent link, and type edge the Python parser produced was wrong in any file with a module docstring — 8,815 of 8,815 index references on trelix's own source. It shipped silently in v2.12.0 and v3.0.0 because the off-by-one always resolved to a valid (just incorrect) row, so nothing ever raised or logged. **A reindex is required to get a correct call graph.** Also restores 177 byte-identical backcompat assertions that had gone dark, and unblocks an integration suite that could not complete.

### Fixed

- **Every call edge, parent link, and type edge produced by the Python parser was
  wrong in any file with a module docstring.** `PythonParser` recorded local
  indices into its `symbols` list during the AST walk, then did
  `symbols.insert(0, <module>)` afterwards when the module had a docstring —
  shifting every element by one and silently invalidating all three index
  families at once: `Symbol.parent_id` (method → enclosing class),
  `CallEdge.caller_id` (call site → enclosing function), and
  `TypeEdge.from_symbol_id` (subclass → base). The module docstring claimed the
  Indexer remapped these; it does not. `indexer.py` builds `local_to_db` by
  enumerating the *final* list, so a pre-insert index resolved to a **valid but
  wrong** row. It never raised, never logged, never produced a null — which is
  why it shipped in v2.12.0 and again in v3.0.0.

  Measured across trelix's own 139 source files, resolving every index to a
  symbol name in both arms: **8,815 of 8,815 index references (100%) were
  wrong** — 7,190 call edges, 1,525 parent links, 100 type edges. 434 parent
  links named `<module>` instead of the declaring class, and 77 call edges were
  fabricated self-recursion. 100% is arithmetic rather than a measurement
  artifact: all 2,179 symbols live in the 131 docstring-bearing files, and the 8
  files without a docstring are zero-byte `__init__.py` that emit no symbols.

  Fixed by reserving `symbols[0]` for the `<module>` symbol *before* the walk, so
  every index recorded is already correct for the list's final layout. This
  removes the bug class rather than compensating for it — no remapping exists to
  get out of step later. Verified against an oracle derived from CPython's own
  `ast` module (0 violations across 22 cases, down from 69), with 5 of 5 mutants
  killed.

  **A reindex is required to obtain a correct call graph.** This corrects
  v3.0.0's release note that "upgrading from v2.12.0 needs no reindex and no
  migration" — true of the schema, but the graph built by the old parser is
  wrong. Expect call-graph expansion, blast radius, PageRank centrality, the
  knowledge graph, and symbol hierarchy to return different, accurate results
  afterwards: on trelix's own index, 100% of PageRank values and 95.1% of ranks
  change (e.g. `CachingPlanner.plan` moves from rank 1975 to 34).

### Test coverage

- **Restored 177 byte-identical backcompat assertions that had silently stopped
  running.** `tests/unit/test_assembler_backcompat_golden.py` proves that with
  compression disabled, `ContextAssembler` output is unchanged from pre-v3.0.0 —
  including an object-identity check on the result list, so no reordering can
  hide. It built its baseline from `git show HEAD:...assembler.py`, so it only
  worked while the change was uncommitted; once merged, HEAD contained the change
  and the module skipped, contributing 0 of 2,190 passing tests. Now pinned to the
  `v2.12.0` tag, which is permanent — preferred over recording golden fixtures
  because it preserves the identity assertion a snapshot cannot express. `ci.yml`
  gained `fetch-depth: 0` / `fetch-tags: true`, without which the tag is
  unreachable in CI and the suite would go dark again. Proven to bite: two
  injected mutations were caught (141 and 8 failures) while a byte-identical
  control passed all 177.
- **Unblocked the integration suite.** `test_graph_api_integration.py` indexed the
  entire worktree, exceeding 390s across three attempts, so 16 tests had no result
  at all; it is now scoped to a fixture tree and runs in 36s. Four `test_cli.py`
  failures came from the developer `.env` leaking an embedder provider into spawned
  subprocesses — `monkeypatch` isolation is in-process only and cannot reach a
  child. All 11 files now pass (99 passed, 1 skipped) in ~576s.
- Covered the remaining `trelix audit` CLI branches: `audit_list`'s empty-log path
  and `audit_export`'s format handling including an unknown format.

## [3.0.0] — 2026-08-13

### Overview
Four internal batches consolidated into one release — there were no public 3.0/3.1/3.2 tags, so everything below lands at once on top of v2.12.0. Six feature areas: Anthropic **extended thinking** as a per-call-site request parameter, a **model-aware context budget** derived from the LLM's real context window, an **actionable VS Code extension** (code lenses + a `@trelix` chat participant), a **hash-chained audit trail** in its own `audit.db`, **OIDC SSO** with an asymmetric-only algorithm allowlist, and **SeleCom query-conditioned context compression** that shrinks retrieved bodies so more results fit the same budget. Six classes of live defect were fixed along the way, each a real failure rather than a hypothetical: an `AttributeError` that crashed *every* Anthropic response containing a thinking block, a VS Code command that returned an interpolated prompt template as if it were the model's answer, four citation-fidelity bugs in compression rendering, an unauthenticated `MarkupError` DoS of `trelix audit list`, an unbounded JWKS read a hostile issuer could turn into ~660 MB of RSS in 0.2s, and a documented env setting (`TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null`) that raised `ValidationError` instead of working.

Everything here is additive and OFF by default. **No breaking changes: upgrading from v2.12.0 needs no reindex and no migration.** The index schema is untouched, every new flag defaults to exactly today's behavior, and the audit/SSO tables live in a brand-new, deliberately separate `audit.db` that is only created when auditing or SSO is switched on. 2,341 unit tests (2,340 pass in a local run; the one failure, `tests/unit/test_watcher.py::TestWatchdogImportError`, asserts the error raised when the optional `watchdog` package is *absent* and therefore cannot pass in an environment with the `watch` extra installed).

### Added
- **Extended thinking (Anthropic) — a request parameter, opted into per call site** — new `LLMConfig.thinking_enabled` (`TRELIX_LLM_THINKING_ENABLED`, default `False`) and `thinking_budget_tokens` (`TRELIX_LLM_THINKING_BUDGET_TOKENS`, default `4096`). Thinking is a **request parameter, not a model name**: `AnthropicBackend` sends `thinking={"type": "enabled", "budget_tokens": N}` on `messages.create`/`messages.stream`, so no model string anywhere in trelix changes and no new model entry is needed. `complete()` and `stream()` on every backend gained a `thinking: bool = False` keyword for signature parity, but only the Anthropic backend acts on it — OpenAI/Azure, Bedrock, Vertex, and LiteLLM accept and ignore it. `tool_call()` deliberately did **not** get the keyword, because extended thinking is incompatible with a forced `tool_choice`: there is no valid combination to expose. Opt-in is deliberately per call site, and exactly one call site opts in: `RetrievalSynthesizer` (`src/trelix/retrieval/synthesizer.py`, both the buffered and streaming paths) passes `thinking=self._llm_config.thinking_enabled`. The two index-time LLM sites — `ContextualChunker`'s per-symbol context call and `FileSummarizer`'s per-file summary — never pass it, on purpose: a single global flag would multiply an indexing run's LLM cost by roughly 5-10x to produce reasoning no retrieval path ever reads. `ChatResponse` gained `thinking: str | None`, `cache_read_tokens`, and `cache_write_tokens` (the two cache counters are read defensively via `hasattr`, so an older `anthropic` SDK whose `usage` object lacks them yields `0` rather than raising). The `temperature=1.0` requirement is enforced in code rather than discovered in production: the backend forces it and ignores any caller-supplied temperature while thinking is on. Thinking bills as **output** tokens — Anthropic exposes no separate thinking counter — so `ChatResponse.output_tokens` covers the visible answer and the thinking blocks together. Tests: `tests/unit/test_llm_thinking.py`.
- **Model-aware context budget** — new `src/trelix/llm/context_windows.py`: a longest-prefix-first `MODEL_WINDOWS` table (Gemini, Anthropic, OpenAI, Cohere, Mistral, Meta) plus `resolve_window(model)`, which prefix-matches so version suffixes resolve without a table entry each (`"gpt-4o-2024-11-20"` → 128,000; `"claude-sonnet-4-20250514"` → 200,000; `"gemini-2.5-pro-preview-0409"` → 1,000,000). `RetrievalConfig.context_token_budget` is now `int | None` and still defaults to `12_000` — the exact v2.12.0 number, so a default install packs context byte-identically. Set it to `null`/`none`/`auto`/`""` (or `None` from Python) and `Retriever._resolve_effective_budget()` derives `window × context_window_fraction` instead, where `context_window_fraction` (`TRELIX_RETRIEVAL_CONTEXT_WINDOW_FRACTION`) defaults to `0.5` and is bounded `0.1-0.9`; the resolved budget is logged at INFO so an operator can see which number actually applied. An unrecognized model logs a warning and falls back to 12,000 rather than guessing optimistically — and that fallback is reached more often than the table suggests, because `resolve_window` matches a bare prefix and therefore does **not** resolve provider-prefixed Bedrock ids (`us.anthropic.claude-sonnet-4-…` → `None` → 12,000). Documented as a known limitation, not silently papered over.
  **The budget was never the real ceiling, and the config now says so.** `rerank_top_n` (15) and `top_k_vector`/`top_k_bm25` (20/20) cap the candidate pool *before* the packer ever sees a budget, so raising only the budget changes very little in practice. New `scale_top_k_to_budget` (`TRELIX_RETRIEVAL_SCALE_TOP_K_TO_BUDGET`, default `False`) is the explicit opt-in that scales `top_k_vector` and `rerank_top_n` by `effective_budget / 12000` — and it raises per-query cost (more reranker work, a larger fusion pool, a bigger synthesis prompt), which is exactly why it stays off even when auto-derivation is on. Tests: `tests/unit/test_context_windows.py`, `tests/unit/test_model_aware_budget.py`.
- **VS Code extension: actionable code lenses** — new `TrelixCodeLensProvider` (`workspace-vscode/src/code-lens-provider.ts`) puts two lenses above every symbol in the open file: a static `$(search) Find similar` action and a count-bearing `N dependents` action. The performance contract is the whole design: `provideCodeLenses` does **zero** MCP work — it only asks VS Code's own `executeDocumentSymbolProvider` for symbols and returns the count lens *unresolved*, with no command attached. `resolveCodeLens` is the single place MCP is touched (`blastRadius()` → `$(references) N dependents`), and its result is cached against the exact `${uri}@${version}::${symbol}` the lens was produced for, so typing never fires a network call, repeated paints at one revision are free, and an edit bumps the document version and correctly misses the cache. A cancelled resolve is never cached, and it falls back to a plain `$(references) Blast radius` action rather than showing a wrong count. Degrades to zero lenses when no document-symbol provider is installed or it errors. Gated by the existing `trelix.codeLens.enabled` setting.
- **VS Code extension: `@trelix` chat participant** — new `workspace-vscode/src/chat-participant.ts` + `chat-handler.ts` register a `trelix.chat` participant: `@trelix <question>` runs the `ask_agent` MCP tool and renders the answer into VS Code's chat view as markdown (with a `stream.progress()` line first, since the ReAct loop is not instant), alongside `/search`, `/explain`, and `/impact` slash commands. Registration is runtime-guarded (`if (!vscode.chat?.createChatParticipant) return undefined`): the chat API landed in VS Code 1.95, so on 1.90-1.94 the participant silently no-ops and the extension still activates cleanly — `engines.vscode` stays `^1.90.0`, with no forced editor upgrade for existing users. Known limitation, documented rather than pretended away: the chat API exposes no stable thread/conversation id, so trelix agent sessions are keyed by a single constant and two chat threads open at the same time share one agent session. The only per-conversation signal available is `context.history` — empty history starts a fresh session, non-empty history resumes the stored one. Tests: `workspace-vscode/src/test/suite/code-lens-provider.test.ts`, `chat-handler.test.ts`.
- **Audit logging — hash-chained, append-only, in its own `audit.db`** — new `src/trelix/audit/` (`events.py`, `store.py`, `middleware.py`) and `AuditConfig` (`TRELIX_AUDIT_ENABLED` default `False`, `TRELIX_AUDIT_DB_PATH` default `<cwd>/.trelix/audit.db`, `TRELIX_AUDIT_LOG_QUERIES` default `False`, `TRELIX_AUDIT_FAIL_CLOSED` default `False`, `TRELIX_AUDIT_RETENTION_DAYS` default `365`). The trail lives in a **separate** SQLite file, never the index DB: the index is disposable and users delete and rebuild it, while an audit trail that vanishes on reindex is not an audit trail. Each row stores `prev_hash` and `entry_hash = sha256(prev_hash || canonical_json(content))` (genesis `prev_hash` = 64 zeros), and an `audit_meta` row keeps a running `count`/`head_hash` anchor updated atomically with every append — the chain alone cannot detect a deleted *tail*, since the survivors still form a valid chain. `AuditMiddleware` records exactly one event per HTTP request and is registered **outermost** in `create_app()`, so it observes the final status code including the 401s produced by the auth dependency and the 500s produced by an unhandled route error. It never persists a header value or token: `principal` comes from `request.state.principal` (defaulting to `"static-token"`), and for `search`/`ask` the query text in `detail` is stored as `sha256(detail)` unless `log_queries=True` is set explicitly. Failure contract, chosen deliberately: a write failure logs at WARNING and lets the request proceed (`fail_closed=False`) so a full disk cannot take down the API, while `fail_closed=True` re-raises for compliance deployments that need a durable trail more than they need uptime. New CLI: `trelix audit list` (Rich table, newest first), `trelix audit verify` (walks the chain, exits nonzero and names the first divergent id), and `trelix audit export --format ndjson` — NDJSON to stdout so a SIEM can be fed by Filebeat/Vector/Fluent Bit rather than by a bespoke HTTP transport trelix would have to maintain. Tests: `tests/unit/test_audit_store.py`, `tests/unit/test_api_audit.py`.
- **OIDC SSO (`pip install 'trelix[sso]'`)** — new `src/trelix/auth/` (`oidc.py`, `principal.py`, `store.py`) and `SSOConfig` (`TRELIX_OIDC_ENABLED` default `False`, `TRELIX_OIDC_ISSUER`, `TRELIX_OIDC_AUDIENCE`, `TRELIX_OIDC_ALGORITHMS` default `["RS256", "ES256"]`, `TRELIX_OIDC_JWKS_URI`, `TRELIX_OIDC_JWKS_TTL_SECONDS` default `3600`), backed by a new `sso` extra (`pyjwt[crypto]>=2.9`). `OidcVerifier` enforces an **asymmetric-only algorithm allowlist** at two gates — construction raises `ValueError` on any non-allowlisted entry, and verification rejects a disallowed `alg` header before decode — so `alg: none` and every `HS*` variant are unreachable. That is the direct defense against the classic algorithm-confusion forgery, where an attacker re-signs a token with HS256 using the server's own RSA *public* key as the HMAC secret. `iss`, `aud`, `exp`, and `nbf` are all verified. **Identity is the `(sub, iss)` pair, never the email** — emails are mutable and reassignable, which makes email-keyed identity an account-takeover primitive; `email`/`display_name`/`groups` ride along for display and coarse authorization only. A verified caller is JIT-provisioned into a `principals` table in the same `audit.db` (`first_seen` immutable, only `last_seen` advances), so identity survives a reindex alongside the trail it belongs to. Static-token auth (`TRELIX_API_AUTH_TOKEN`) still works exactly as before, and with SSO off the API behaves byte-identically to v2.12.0. Tests: `tests/unit/test_oidc.py`.
- **SeleCom query-conditioned context compression (opt-in)** — new `src/trelix/compression/` (`base.py`, `extractive.py`) plus `src/trelix/retrieval/context_compression.py`, wired behind `compression_enabled` (`TRELIX_RETRIEVAL_COMPRESSION`, default `False`), `compression_provider` (`TRELIX_RETRIEVAL_COMPRESSION_PROVIDER`, only `"extractive"` today), `compression_target_ratio` (`TRELIX_RETRIEVAL_COMPRESSION_RATIO`, default `0.45`, bounded `0.1-1.0`, used only for an unknown/absent intent), and `compression_min_tokens` (`TRELIX_RETRIEVAL_COMPRESSION_MIN_TOKENS`, default `120` — below that the per-span headers and elision markers cost more than the shrink saves). Two invariants carry the whole safety argument:
  - **Result-lossless.** Packing runs in two waves. Wave 1 is the exact uncompressed pack the assembler already did, with the compressor never called. Wave 2 compresses **only** the candidates wave 1 could not fit — the ones today silently drops — and re-offers them against the leftover budget, walking a ladder (budget-tightened target ratio → the compressor's signature+docstring floor → skip, recorded in the trace stats so a skip is never silent). The compressed selection is therefore always a superset of the uncompressed one, so Recall/MRR/nDCG cannot go down by construction, and a result that already fits is never touched, keeping its text byte-identical.
  - **Citation fidelity.** Kept spans render as separate `[Lines a-b] <qualified_name>` blocks with explicit `# ... N lines elided ...` markers between them, so no header ever claims lines its text does not contain.
  The default `"extractive"` provider does **zero query-time inference** — no embedding, API, or network call. Its `sub_chunk` path reuses sub-chunk vectors already stored at index time (`chunk_id = sub_chunk_id + 10_000_000`) and takes the query embedding from a *peek* at the retriever's existing `embed_query` LRU, never a fresh call; when sub-chunks are absent it falls back to a lexical splitter (blank-line + brace-balance segmentation scored by query-token overlap). The lexical path is the common one today, because sub-chunks are Python-only and `multi_granularity_enabled` (`TRELIX_CHUNKER_MULTI_GRANULARITY`) defaults to `False`. Per-intent ratios are baked into `RetrievalStrategy` and individually overridable via `TRELIX_RETRIEVAL_COMPRESSION_RATIO_<INTENT>`: `symbol_lookup`, `file_overview`, `project_overview`, and `config_lookup` are all `1.0` (off — for those the body *is* the answer), `feature_flow` `0.45`, `dependency_map` and `blast_radius` `0.30`, `comparison` `0.65`. A ratio of `1.0` short-circuits before the compressor is even constructed. Honest expectation, not a benchmark claim: roughly **30-60% fewer synthesis input tokens and 15-35% lower latency on a network-API synthesis path** — this is not a "4x faster" feature, and on a local model the token saving matters more than the wall clock. Tests: `tests/unit/test_compression_extractive.py`, `tests/unit/test_assembler_compression.py`, `tests/unit/test_compression_citation_adversarial.py`, `tests/unit/test_compression_citation_truncated_body.py`.
- **FTS5 declaration-boost ranking (opt-in)** — new `RetrievalConfig.declaration_boost_enabled`/`declaration_boost_weight` (`TRELIX_RETRIEVAL_DECLARATION_BOOST`/`_WEIGHT`), threaded into `Database.bm25_search()`'s previously-unweighted FTS5 query via an explicit `bm25(symbols_fts, ...)` call. Fixes a real ranking defect: unweighted BM25 can rank a symbol that only *mentions* a query term in its body/docstring above the symbol whose *name* actually matches it — reproduced live on trelix's own self-index, where `Database.bm25_search` (the method implementing the very feature being searched for) doesn't appear in the top 15 results for the query `"bm25_search"` under the old unweighted ranking, but reaches rank #9 with `declaration_boost_weight=5.0`. Default weight `1.0` is a verified no-op — byte-for-byte identical to today's unweighted ranking. No FTS5 schema change; the existing 5-column virtual table is reweighted, not rebuilt.
- **VS Code hover provider** — hovering over any identifier now shows its signature, docstring, and file/line location via the `get_symbol` MCP tool, with a per-session `word::repo` cache. No new backend plumbing — reuses the existing `get_symbol` tool and the extension's existing `ensureConnected()`/`getRepoPath()` helpers. Registered on an unrestricted `{ scheme: "file" }` selector rather than a per-language allowlist, since trelix indexes 20+ languages. Known limitation (documented, not fixed here): `get_symbol` falls back to an ambiguous bare-name lookup when the exact qualified name doesn't resolve — a hover over a name shared by multiple symbols may show the wrong one.
- **JetBrains IDE setup docs** — `docs/MCP_GUIDE.md` now documents connecting `trelix-mcp` via JetBrains IDEs' (2025.2+) first-party built-in MCP client, mirroring the existing Cursor/Windsurf/Continue.dev sections. Docs-only — no dedicated trelix JetBrains plugin exists or is planned by this change.
- **Markdown diagram-block tagging** — `Chunker` now prepends a `# Diagram: mermaid`/`# Diagram: plantuml` context header (mirroring the existing `# File:`/`# Class:` convention) when a chunked section's body contains a fenced mermaid/plantuml block. New `ChunkerConfig.include_diagram_tags` (default `True`). Implemented in the language-agnostic `Chunker`, not the markdown parser, so it works for any language whose symbol body happens to embed a fenced diagram block. Only the first diagram fence per section is tagged (documented v1 limitation, not a bug).

### Fixed
- **Anthropic backend crashed with `AttributeError` on any response containing a thinking block** — a latent crash, live the moment extended thinking was switched on: the backend read the answer as `response.content[0].text`, and Anthropic returns the thinking block *first*, so `content[0]` is a `ThinkingBlock` with no `.text` attribute. Fixed by a new `_split_content()` that walks every content block and joins `text` and `thinking` parts separately, returning `("", None)` for an empty content list. This is why `ChatResponse.thinking` exists as a real field instead of the reasoning being discarded or, worse, concatenated into the answer.
- **VS Code `ask()` returned an interpolated prompt template as if it were the model's answer** — `TrelixMcpClient.ask()` called `getPrompt("trelix-search")`, which by MCP's own semantics returns a *filled-in prompt template*, not a completion. The "trelix: Ask about Code" command therefore rendered prompt scaffolding to the user and had never actually answered a question. Fixed to `callTool({ name: "ask_agent", … })` — the multi-turn ReAct agentic loop — returning the parsed `{answer, session_id, turn_count}` envelope, mirroring `search()`'s existing `callTool` + `JSON.parse` pattern. The surfaced `session_id` is what makes chat follow-ups possible at all. Tests: `workspace-vscode/src/test/suite/mcp-client.test.ts`.
- **Four citation-fidelity defects in compressed-block rendering** — each one produced a `[Lines a-b]` header that lied about the text underneath it, which is the single worst failure mode for a retrieval system whose output an LLM will cite verbatim. All four are fixed in `src/trelix/retrieval/context_compression.py` with regression tests in `tests/unit/test_compression_citation_adversarial.py` and `tests/unit/test_compression_citation_truncated_body.py`:
  - A passthrough result claimed the symbol's **full declared span** even when the stored body had been truncated at index time (`body=…[:2000]` while the AST span stayed complete), so the header advertised lines the text never held. `_fallback_block()` now derives its `end` from the text *actually rendered* — the leading run of real source lines, stopping at the first elision marker — and never from the unmappable spans.
  - A kept span could escape past `line_end` when a parser prepends a synthetic line, making the body's line count and the symbol's declared range disagree. Spans are now filtered against `symbol.line_start <= a <= b <= symbol.line_start + len(body_lines) - 1`, i.e. against the body actually held rather than the declared range.
  - `_envelope()` could emit an **inverted** range such as `[Lines 9000-122]`, claiming a negative-length citation, whenever every kept span lay outside the symbol: the two ends were clamped independently, pairing `max(122, 9000)` with `min(122, 9000)`. It now falls back to the declared range when no span overlaps the symbol, and can no longer return an inverted pair.
  - The trailing `# ... N lines elided ...` count was computed against `symbol.line_end`, so a truncated stored body produced a **bogus** number for lines that were never ours to elide. The tail gap is now measured against `min(symbol.line_end, body_end)`, where `body_end` comes from the real body length.
- **`TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null` raised `ValidationError`** — the documented env route to the auto-derived budget did not work. Env values arrive as strings and an `int | None` field cannot coerce `"null"`; since the entire model-aware budget path is gated on `context_token_budget is None`, the feature was reachable only from the Python API, never from env, `.env`, or the CLI. Fixed with a `mode="before"` field validator that maps `""`/`"null"`/`"none"`/`"auto"`/`"~"` (case-insensitive, whitespace-stripped) to `None`. Verified end-to-end: `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null` now yields `context_token_budget=None` with `context_window_fraction=0.5`.
- **`IndexConfig` missing `populate_by_name=True`** — every other aliased-field
  config class in `core/config.py` (`EmbedderConfig`, `GitLinkerConfig`, and 9
  others) sets `populate_by_name=True` in `model_config`; `IndexConfig` was the
  sole exception, added when `file_summaries_enabled`/`telemetry_enabled` first
  gained their `alias=` (unlike every sibling class's identical addition).
  Without it, passing `file_summaries_enabled=`/`telemetry_enabled=` as a
  constructor kwarg by field name was silently ignored — the value fell back
  to the alias env var or default with no error, since `model_config` also
  sets `extra="ignore"`. Fixed by adding the missing `populate_by_name=True`,
  matching every sibling config class.

### Security
- **`trelix audit list` crashed with `MarkupError` on an attacker-controlled request path** — an unauthenticated DoS of the audit tooling, and of exactly the log a responder needs mid-incident. Audit rows store attacker-influenced values (the request path, and an OIDC `sub`), and Rich parses `[...]` in a cell as console markup: a single unauthenticated `GET /%5B/red%5D` persisted the resource `/[/red]`, after which every subsequent `trelix audit list` died before printing a single row. Fixed by passing every cell through `rich.markup.escape()` in `src/trelix/cli/main.py`. Note the asymmetry that made this worth fixing properly: the attacker needs no credentials to write the poisoned row, and the operator loses read access to the whole trail.
- **JWKS fetch had an unbounded read** — `_fetch_jwks()` called `response.read()` with no size cap, so a hostile or compromised issuer serving an endless body drove memory to roughly 660 MB RSS in 0.2s (the socket timeout is per-operation, not a total budget, so it does not bound the response size). Now capped at 1 MiB (`_MAX_JWKS_BYTES`): the reader asks for `cap + 1` bytes and refuses to parse anything larger, which is already absurdly generous for a document that is genuinely a few KB. The same resolver also enforces HTTPS-only, pins the JWKS host to the issuer host, refuses **every** redirect (a redirected JWKS fetch is a trust-boundary break / host-pinning bypass), and bounds the socket timeout.

### Deliberate scope decisions and known limitations
- **SAML is not implemented.** This release ships **OIDC only**, and nothing here should be read as SAML support. SAML's XML-signature and metadata handling is a materially larger and more error-prone attack surface than OIDC's JWS-over-JWKS, and shipping a half-verified SAML path would be worse than shipping none.
- **The audit trail is tamper-EVIDENT, not tamper-PROOF.** The hash chain plus the `count`/`head_hash` anchor reliably catch naive or accidental corruption — a stray `UPDATE`, a truncated file, a dropped tail, a reordered row. They do not stop a determined attacker with write access to `audit.db`, who can rewrite a row *and* recompute every subsequent `entry_hash` plus the anchor in one transaction, after which `verify_chain` passes. The anchor lives in the same file it is meant to protect and is bookkeeping, not an external root of trust. Hardening path if you need a stronger guarantee: sign the head hash with a key held outside the DB (HMAC or asymmetric) and/or anchor it to append-only/WORM storage.
- **Only the HTTP API surface is audited.** `AuditMiddleware` is HTTP middleware, so MCP tool calls and the internal agent loop are **not** on the audited path. A deployment that reaches trelix over MCP rather than over `trelix serve` produces no audit records at all.
- **`retention_days` is declarative only.** `TRELIX_AUDIT_RETENTION_DAYS` (default `365`) records the intended retention window for policy and tooling; no pruning job ships in this release, so the trail grows unbounded until an operator trims it.
- **`resolve_window()` does not resolve provider-prefixed model ids.** Prefix matching is anchored at the start of the string, so Bedrock-style ids (`us.anthropic.claude-…`, `anthropic.claude-3-5-sonnet-…`) fall through to the 12,000-token fallback with a warning rather than to their real 200,000-token window.
- **The one queued breaking change is still deferred.** `RetrievalConfig.flare_max_retries` continues to accept its legacy env alias `TRELIX_RETRIEVAL_FLARE_MAX_ITER` (deprecated in v2.4, still honored via `AliasChoices` with a `DeprecationWarning`). Dropping that alias was the single genuine breaking change reserved for a 3.x release, and it is deliberately **not** in this one — v3.0.0 is a major-version bump for the size of the feature surface, not for an incompatibility. See `docs/ROADMAP.md`.

## [2.12.0] — 2026-08-03

### Overview
Three-phase post-v2.11.1 optimization pass, scoped from a `/deep-research` pass (110 subagents, 27 primary sources) plus direct codebase exploration: a real call-graph/type-edge correctness fix, a per-leg RRF weight config activating an already-built-but-unwired fusion mechanism, and a bundle of four independent hardening/doc/CI items — including a previously-unknown packaging gap affecting every downstream consumer of the `trelix` PyPI package. All new surface is additive/opt-in; no breaking changes.

### Fixed
- **Call-graph/type-edge resolver silently picked the wrong symbol on a name collision** — `Indexer._store_call_edges()`/`_store_type_edges()` resolved a bare-name match by taking `matches[0]` unconditionally from `Database.get_symbol_by_name()`, with zero ambiguity check. When two classes defined a same-named method (e.g. `Retriever.retrieve` vs `FederatedRetriever.retrieve`), every call site anywhere in the codebase got wired to whichever symbol had the lower DB id, regardless of which one was actually called — violating a documented design invariant (`docs/architecture.md` §21 #5: "a wrong edge is worse than a missing edge"). The safe 4-priority cascade already existed in `Database.resolve_cross_file_calls()`, but that only runs for batches ≥5 files — a single-file watch re-index never reached it. Fixed by inlining the same priority cascade (exact qualified_name match, unique among candidates → type-hint-assisted match, unique among candidates → unique bare-name match → else unresolved) via a new `Indexer._resolve_symbol_match()` helper shared by both call- and type-edge resolution. Measured on trelix's own self-index: 56/65 incorrectly-resolved same-named-method calls → 4/61 correctly resolved (remainder correctly left unresolved); ~2,400 provably-wrong call edges removed codebase-wide.
- **`core/retry.py`: unbounded `Retry-After` value crashed the retry loop** — `_wait_retry_after_or_exponential` now clamps a server-supplied `Retry-After` value to `max_wait_seconds` before returning it. A hostile/malformed header (e.g. `Retry-After: 99999999999999999999999999999999`) previously reached `time.sleep()` unbounded, raising `OverflowError` and crashing the retry loop instead of retrying or failing cleanly — bypassing `stop_after_attempt` entirely. Affected every `@with_retry`-wrapped call site (all connectors, all LLM/embedder backends).
- **`indexing/connectors/jira.py`: unbounded ADF recursion crashed on deep descriptions** — `_adf_node_to_text()` now tracks recursion depth and truncates past `_ADF_MAX_DEPTH` (150 — empirically verified safe to 50,000+ nesting levels, since each ADF level costs more than one real Python stack frame). A sufficiently deep Jira description previously raised an uncaught `RecursionError`, aborting the entire connector sync run over one malformed document.
- **`trelix`'s own PyPI package shipped with no `py.typed` marker (PEP 561)** — without it, mypy silently treated every `trelix` import as `Any` once checked across the installed-package boundary, masking real type errors in every downstream package. Added `src/trelix/py.typed` (verified correctly bundled into a real built wheel) and fixed every newly-visible error in `trelix-mcp`/`trelix-langchain`/`trelix-llama-index` (missing return-type annotations, an untyped decorator chain, a genuine mypy list-invariance bug in `resources.py`'s symbol-lookup fallback).
- **`packages/trelix-mcp`'s `mcp`/`fastmcp` dependency floor was stale and would have been wrong if naively bumped** — the declared `mcp>=1.0.0` floor was stale, but bumping straight to latest (`mcp==2.0.0`) would have made `trelix-mcp` uninstallable alongside `fastmcp` (whose own latest release pins `mcp<2.0,>=1.24.0` — confirmed via a real `pip install` reproduction that fails with `ResolutionImpossible`). Set to `mcp>=1.24.0,<2.0` / `fastmcp>=3.4.0`, matching what's actually installable and was tested against.

### Added
- **Per-leg RRF weight config** — new `RetrievalConfig.leg_weights: dict[str, float]`, keyed by `vector`/`bm25`/`grep`/`summary`/`sub_chunk`/`sparse`, all default `1.0` (a no-op). Threaded through the primary single-repo `reciprocal_rank_fusion(...)` call site via the mechanism's pre-existing `list_weights=` param — already implemented, tested, and in production use by `FederatedRetriever` for per-repo weighting, just never wired up outside federation. Activates investigation of RRF's documented "weakest-link" failure mode (a weak retrieval leg fused with a strong one can drag the strong leg's score below what it would score alone — VLDB 2026, 11-dataset benchmark) without shipping any new default weights. Env var overrides: `TRELIX_RETRIEVAL_LEG_WEIGHT_<VECTOR|BM25|GREP|SUMMARY|SUB_CHUNK|SPARSE>`, mirroring the existing `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_<LANG>` pattern.
- **`docs/CLI_REFERENCE.md`: documented `trelix eval-synthesis`** — the command existed with zero documentation. New section covers Synopsis/Description/Options/Examples/Golden file format/Output/Notes, including a genuine discovered gotcha: unlike `trelix eval`, a missing golden file for `eval-synthesis` does not error or exit non-zero — it silently prints an all-zero-score table.
- **CI now lints and type-checks `packages/trelix-mcp`, `packages/trelix-langchain`, and `packages/trelix-llama-index`** — previously unchecked anywhere, CI or locally. This is what surfaced the `py.typed` gap and the dependency-floor landmine above.

## [2.11.1] — 2026-08-02

### Security
- **`/graph/visualize` output-path containment bypass via sibling directory** —
  the `output` query param's containment check used a raw string-prefix
  match (`str(requested).startswith(str(allowed))`) to confirm the resolved
  path stays inside `<repo>/.trelix/`. A sibling directory that merely
  starts with the same characters — e.g. `<repo>/.trelix-evil/x.html` —
  passed this check, letting a caller write an arbitrary-named HTML file
  outside the intended directory. `/parse` already fixed this exact bug
  class earlier in the same file (`src/trelix/api/app.py`, with a comment
  explicitly warning about the string-prefix pitfall), but the fix was
  never back-ported to `/graph/visualize`. Fixed by switching to
  `Path.is_relative_to()`, mirroring `/parse`'s existing correct check.
  Severity: MEDIUM — a file write of graph HTML, not arbitrary read or
  RCE, and only reachable when `TRELIX_API_AUTH_TOKEN` is unset (default)
  or the caller already has a valid API key. Found during a full
  production dry-run of v2.11.0's REST API surface (path-traversal probing
  against a live container). New regression tests in
  `tests/unit/test_api_graph.py::TestGraphVisualizeContainment` (4 cases,
  confirmed fail-before/pass-after against the reverted check).

## [2.11.0] — 2026-08-02

### Overview
Six backlog phases plus four follow-up fixes: connector-fetched artifacts now auto-link into `generic_edges` on sync, opt-in Personalized PageRank for cross-source ranking, a unified retry/backoff contract shared by every LLM backend/embedder/connector/reranker/PR client, structured JSON logging with trace correlation, a new Xray Cloud connector, and a new Linear connector — plus two real bugs found by live-testing the Jira connector against a production Jira Cloud site (ADF descriptions silently dropped to empty bodies; bad credentials silently reported as success), a `.env`-leakage test-isolation fix, and a Windows binary crash caught by this release's own pre-`main` CI run (structured logging's new `structlog` dependency triggering a `colorama`/Rich console conflict). All new surface is additive/opt-in; no breaking changes.

### Added
- **Connector-to-graph auto-linking** — new `ArtifactLinker` (`src/trelix/indexing/artifact_linker.py`) scans each connector-fetched artifact's title/body for symbol name/qualified-name mentions (regex-first, identical precedent to `GitLinker`'s commit-message matching) and writes `generic_edges` rows with `edge_kind="references_artifact"`, `weight=1.0`. An opt-in embedding-similarity fallback (`TRELIX_ARTIFACT_LINKER_EMBEDDING_FALLBACK`, default `False`) runs only for artifacts with zero regex matches, at `weight=0.5` so lower-confidence matches don't dominate PageRank mass. `ArtifactSource.sync()` gained an optional `linker: ArtifactLinker | None` param — when supplied, each successfully-written artifact is immediately passed through `linker.link_one()`, so a synced artifact is reachable from the code graph the moment `sync()` returns, with no separate pass required. New standalone CLI command `trelix link-artifacts <repo>` for a batch re-link (e.g. after a schema/symbol change). `trelix connector sync` gained a `--link/--no-link` flag (default `--link`) and now reports a `linked N edge(s)` count. `ConnectorSyncResult` gained an `edges_linked: int = 0` field. New `Database.get_all_symbol_names()`/`get_all_artifacts()` bulk queries backing the linker's per-`link()`-call name index (built once, not per-artifact, to avoid an O(artifacts × symbols) scan).
- **Personalized PageRank (opt-in)** — new `RetrievalConfig.pagerank_personalization_enabled: bool = Field(default=False, alias="TRELIX_RETRIEVAL_PAGERANK_PERSONALIZATION")`. Both `rank_by_pagerank()` (`retrieval/graph.py`, query-time) and `compute_pagerank()` (`graph/community.py`, index-time) now pass `personalization=` into `nx.pagerank()` when enabled: uniform mass `1/|T|` over every node with at least one `generic_edges`-derived artifact/ticket relationship, falling back to today's uniform teleportation when the flag is off or the seed set is empty — zero behavior change for anyone not opting in. Ships with a documented interaction-risk note against `pagerank_boost_enabled` (personalization's teleport mass isn't weighted by call-graph importance, so on a repo with few ticket-linked symbols, enabling both together can invert `get_top_central_symbols()`'s ranking).
- **Unified retry/backoff contract** — new `src/trelix/core/retry.py`, built on `tenacity`: `is_retryable_http_error()` classifies 429/5xx and connection-level failures as retryable (recognizing httpx/requests/boto3/openai/anthropic/google-genai/voyageai exception shapes without hard-importing any of them) and `with_retry(max_attempts=5, ...)` returns a `@retry` decorator honoring a server-supplied `Retry-After` header when present (parses both integer-seconds and HTTP-date forms per RFC 9110 §10.2.3), falling back to full-jitter exponential backoff otherwise. Wired into all 5 LLM backends (OpenAI/Anthropic/Bedrock/LiteLLM/Vertex), all remote embedder providers, `GitHubPRClient`, `reranker.py`'s Cohere reranker, and the Jira/TestRail connectors — replacing each connector's previous hand-rolled, non-jittered backoff loop. Also disables each SDK's own built-in retry (`max_retries=0` for OpenAI/Anthropic clients, `botocore.Config(retries={"max_attempts": 0})` for boto3-backed Bedrock) so the shared contract is the sole retry layer, avoiding multiplicative retry stacking.
- **Structured logging** — new `src/trelix/core/logging_setup.py`: `setup_console_logging()` (human-readable, used by every CLI command) and `setup_json_logging()` (`structlog.stdlib.ProcessorFormatter`-based JSON, used by `trelix serve`), both built on the same processor chain so the ~164 existing `logger.*` call sites across the codebase need no rewrite. When `TRELIX_OTEL_ENABLED=true`, a processor injects the active span's `trace_id`/`span_id` into every JSON log line, correlating logs with the existing OpenTelemetry traces. `trelix serve` now passes a matching `uvicorn_log_config()` to `uvicorn.run()` so access-log lines are JSON too, not half-JSON/half-default-colorized-text.
- **Xray Cloud connector** — fourth `ArtifactSource` implementation (`src/trelix/indexing/connectors/xray.py`), Cloud-only. Auth: `client_id`/`client_secret` (issued by a Jira admin in Xray's global settings, distinct from a personal Jira API token) exchanged via `POST /api/v2/authenticate` for a short-lived bearer JWT. Fetches test content via a single GraphQL `getTests` query per page (Jira issue fields and Xray-specific steps/definition both come back in one call — no separate Jira REST call needed), `limit`/`start` pagination. New `XrayConnectorConfig` (`TRELIX_XRAY_CLIENT_ID`/`CLIENT_SECRET`/`PROJECT_KEY`/`JIRA_BASE_URL`, `page_size` default 100).
- **Linear connector** — fifth `ArtifactSource` implementation (`src/trelix/indexing/connectors/linear.py`). Auth: personal API key sent as `Authorization: <key>` — **no Bearer prefix**, Linear's own documented scheme, distinct from every other connector in this codebase. GraphQL-only (single POST to `https://api.linear.app/graphql`, no official Python SDK exists), Relay-style cursor pagination (`first`/`after`/`pageInfo.hasNextPage`/`endCursor`), scoped to one team via `TRELIX_LINEAR_TEAM_KEY`. Rate-limit handling is deliberately connector-local rather than folded into the shared retry contract: Linear signals rate-limiting via HTTP 400 (not 429) with a GraphQL body error code `RATELIMITED` and no `Retry-After` header — instead reading `X-RateLimit-*-Reset` response headers (epoch-ms) — which the shared `is_retryable_http_error()` cannot and should not be taught, since that would couple a contract shared by every other connector/LLM backend to one connector's response shape. `source_ref` uses a `linear-issue:` prefix (not Jira's `ticket:`) to avoid a silent collision if a Jira project key and Linear team key ever coincided. New `LinearConnectorConfig` (`TRELIX_LINEAR_API_KEY`/`TEAM_KEY`, `page_size` default 100). Live-verified end-to-end against a real Linear workspace: fetch, cursor pagination, idempotent re-sync, `ArtifactLinker` auto-linking into `generic_edges`/`CodeGraph`/PageRank, `--no-link`, missing-config validation, and a real 401 all confirmed working.
- **`trelix connector sync`** now accepts `xray` and `linear` alongside `jira`/`testrail`; `docs/CLI_REFERENCE.md`/`docs/CONFIGURATION.md` updated for both (and backfilled the pre-existing gap where Xray was never documented after its own merge).

### Fixed
- **Jira ADF descriptions silently rendered as empty bodies, on every real ticket** — found by live-testing the Jira connector against a production Jira Cloud site (not previously caught, since it had only ever been exercised via mocked HTTP responses): Jira Cloud's v3 API always returns `description` as Atlassian Document Format (a nested JSON tree), never a plain string, and the connector's `isinstance(description, str)` check fell back to an empty body for every single ticket — not a rare edge case, the universal case. Fixed with a new `_adf_to_text()` renderer (`src/trelix/indexing/connectors/jira.py`) covering every node type confirmed against real descriptions: paragraphs, headings, bullet/ordered lists, code blocks, inline code, panels, expand sections, rules, links (rendered inline as `text (url)`), and embedded cards, with unrecognized future node types degrading to rendering their children rather than being dropped. Live-verified against a real project: body length went from 0 chars (all 5 tickets) to 497–6529 chars of real content each, and `ArtifactLinker`'s auto-link count on the same 5 tickets went from 2 edges (title-text matches only) to 5 — direct evidence real linking recall was silently lost, not just cosmetic body text.
- **Jira connector silently reported success on bad credentials** — also found live: `/rest/api/3/search/jql` returns HTTP 200 with an empty result set (`{"issues":[],"isLast":true}`) for a bad API token, no auth at all, *and* a genuinely empty/nonexistent project — all three byte-identical, meaning a misconfigured `TRELIX_JIRA_API_TOKEN` would report `fetched 0, wrote 0, errors 0` as success forever. Fixed by adding a pre-flight `_verify_auth()` call to `/rest/api/3/myself`, which correctly returns a real 401 for bad/missing credentials, before the search loop ever runs. Live-verified: a bad token now fails with `401 Unauthorized`/exit code 1 instead of silently succeeding; real credentials against the live project are unaffected.
- **`.env`-leakage test-isolation gaps for Linear/Jira/TestRail connector tests** — `tests/unit/conftest.py`'s isolation fixture only covered a handful of beast-mode flags; a developer's real `TRELIX_LINEAR_*`/`TRELIX_JIRA_*`/`TRELIX_TESTRAIL_*` credentials in `.env` (added for the live-testing above) silently broke each connector's "missing config" unit test, since `monkeypatch.delenv()` only clears the process environment while pydantic-settings' `.env`-file source reads the file directly and independently. Fixed by overriding those vars to `""` (or `"0"` for `TestRailConnectorConfig.project_id`, which is `int | None` and can't parse an empty string) — both falsy, so every `if not val` `validate_config()` check treats them identically to unset.
- **Windows binary crashed on every command once `stats` gained `_setup_logging()`** — found by this release's own pre-`main` CI run (the Windows binary-build job, not a unit test): `structlog`, added this cycle as a new direct dependency, ships a `ConsoleRenderer` whose `colors` argument auto-defaults to `True` on Windows whenever `colorama` is importable (already a transitive dependency here). Constructing it therefore called `colorama.init()`, which monkeypatches `sys.stdout`/`sys.stderr` into `AnsiToWin32`/`StreamWrapper` objects — confirmed to collide with Rich's own `legacy_windows_render` write path on the same stream once both are active, crashing with `OSError: [Errno 22] Invalid argument` the next time Rich printed (e.g. `stats`'s summary table). This codebase's actual color output has always come from Rich (`cli/main.py`'s `Console`/`Panel`/`Table`), never from structlog's renderer, so the coloring was pure incidental risk with no product benefit. Fixed by constructing `ConsoleRenderer(colors=False)` in `setup_console_logging()` (`src/trelix/core/logging_setup.py`). New regression test `test_console_renderer_never_requests_colors` (`tests/unit/test_structured_logging.py`) asserts the real constructor argument.

### Known non-blocking issues (found during pre-release security review, not fixed this cycle)
- `core/retry.py`'s `Retry-After` header handling has no upper bound — a hostile or unrealistically large value (e.g. `Retry-After: 99999999999999`) causes `time.sleep()` to raise an uncaught `OverflowError`, bypassing `stop_after_attempt` entirely; even a plausible large value (e.g. 6 hours) is honored verbatim past the configured `max_wait_seconds` ceiling. Affects every `@with_retry`-wrapped call site (all connectors, all LLM/embedder backends).
- `jira.py`'s `_adf_node_to_text()` recurses over nested ADF content with no depth limit — a sufficiently deep `description` (500+ levels) raises an uncaught `RecursionError`, aborting the sync run.
- Both require a malicious/misbehaving external API response to trigger (crash/hang, not credential leak or data corruption) and are tracked as fast-follow items, not release blockers.

## [2.10.0] — 2026-07-28

### Overview
Five backlog phases plus two follow-up fixes: REST API-key auth + HTTP-layer OpenTelemetry spans, a new `/parse` endpoint + opt-in per-source context budgeting, leg-level path filtering + intent hints for `/search`/`search_code`, cross-source graph edges via a git-log ticket linker, and Jira/TestRail artifact connectors — plus a path-traversal fix discovered in `/parse` during this cycle and a `generic_edges` dedup fix. All new surface is additive/opt-in; no breaking changes.

### Added
- **REST API-key auth (`TRELIX_API_AUTH_TOKEN`)** — new `_ApiAuthSettings(BaseSettings)` in `src/trelix/api/app.py` (single field `api_auth_token: str | None = None`, env alias `TRELIX_API_AUTH_TOKEN`, loaded from `.env`/environ once per `create_app()` call — this lives standalone in `app.py`, not folded into `core/config.py`/`IndexConfig`). New `verify_api_key` FastAPI dependency wired via `dependencies=auth` onto every route except `/health`: `/search`, `/ask`, `/index`, `/stats`, `/graph`, `/graph/communities`, `/graph/visualize`, `/graph/search`. Token unset (default) → fully open, identical to prior behavior. Token set + header missing → `401` (short-circuits on `x_trelix_api_key is None`, before any comparison). Token set + wrong header → `401` via `hmac.compare_digest` (timing-safe compare). All `401`s log a warning; `/health` is excluded from `dependencies` outright rather than merely bypassed in logic. Tests: `tests/unit/test_api.py::TestApiAuth` (4 cases) plus a `conftest.py` fixture that `delenv`s the token so unrelated tests stay unauthenticated.
- **OpenTelemetry spans for the HTTP layer** — the same 8 auth-gated routes now wrap their handler bodies in `pipeline_stage_span(config.retrieval, "http_<name>", ...)`, reusing v2.9.0's existing `retrieval/otel_tracing.py` helper: `http_search` (attrs `k`/`cursor`), `http_ask`, `http_index`, `http_stats`, `http_graph`, `http_graph_communities`, `http_graph_visualize`, `http_graph_search`. No new package import; still gated by `TRELIX_OTEL_ENABLED`/`is_enabled(cfg)` and a no-op when disabled.
- **Helm**: new `apiAuth.token` / `apiAuth.existingSecretName` / `apiAuth.existingSecretKey` in `values.yaml`, wired into `deployment.yaml` (an `existingSecretName` wins over the inline token) and `secret.yaml` (writes `stringData` only when an inline token is set and no existing secret is named).
- **`POST /parse`** — a new one-off parse endpoint (`ParseRequest`/`ParseResponse`, `src/trelix/api/app.py`). Request: `repo_path: str` (required), `file_path: str | None` (disk-backed, relative or absolute), `content: str | None` (inline text), `file_name: str | None` (required only when `content` is set). A `model_validator` rejects with `422` unless exactly one of `file_path`/`content` is set, and rejects with `422` when `content` is set without `file_name`. Response: `symbols: list[{name, qualified_name, kind, line_start, line_end, signature}]`, `call_edge_count`/`import_edge_count`/`type_edge_count`, `parse_errors`, and a `note` that explicitly states cross-file call/type resolution was skipped for this single-file parse (or reports "No parser available for language ..." for an unrecognized extension).
- **Per-source context budgeting** — new `RetrievalConfig.context_budget_per_source: bool = Field(default=False, alias="TRELIX_RETRIEVAL_CONTEXT_BUDGET_PER_SOURCE")` in `src/trelix/core/config.py`. Default `False` reproduces the prior single-pool greedy pack byte-for-byte. `ContextAssembler` gained a matching `per_source_budget` constructor param, wired through `Retriever`.
- **Leg-level path filtering** — new `SubQuery.path_filter: str | None = None` (`src/trelix/retrieval/planner/models.py`) restricts a leg's results to files whose `rel_path` starts with the given prefix, plus `RetrievalConfig.path_filter_oversample: int = 3` (`ge=1`, env `TRELIX_RETRIEVAL_PATH_FILTER_OVERSAMPLE`). Wired per leg: the grep leg passes `path_filter=sq.path_filter` straight into `grep_search()`; the BM25 leg pushes it into `db.bm25_search`'s SQL join (`f.rel_path LIKE ?`); the vector leg (`Retriever._vector_search`) over-fetches `k * path_filter_oversample`, post-filters by `rel_path.startswith(path_filter)`, then truncates back to `k`.
- **`intent_hint`/`hyde_snippet_hint` on `/search` and `search_code`** — identical new optional params added to REST `GET /search` (`src/trelix/api/app.py`) and MCP `search_code(...)` (`packages/trelix-mcp/src/trelix_mcp/server.py`). Both call `plan_from_intent_hint(query, intent_hint, hyde_snippet_hint)` when `intent_hint` is given and pass the result as `plan=plan` into `Retriever.retrieve()` — no changes needed inside `Retriever` itself. An unrecognized `intent_hint` never raises: `plan_from_intent_hint()` catches the `IntentType(intent_hint)` `ValueError` and also checks membership in `INTENT_STRATEGIES`, returning `None` either way so the query silently falls back to `Retriever`'s normal internal LLM-based intent classification.
- **Cross-source graph edges + git-log ticket linker** — new `generic_edges` table (`src/trelix/store/db.py`, via `_apply_migrations()`): `id`, `from_symbol_id` (FK → `symbols.id`, `ON DELETE CASCADE`), `source_ref` (free-form `"<type>:<ref>"`, e.g. `"ticket:PROJ-123"`), `edge_kind`, `weight REAL DEFAULT 1.0`, with indexes on `from_symbol_id` and `source_ref`. New CLI command `trelix link-tickets <repo>` (`src/trelix/cli/main.py`) with `--max-commits` (int, default 5000), `--since` (e.g. `"90 days ago"`, passed to `git log --since`), `--ticket-pattern` (regex, default `r"[A-Z]+-\d+"`, Jira-style). `GitLinkerConfig.enabled` defaults to `False`, and the linker is not wired into `Indexer.index()` — it's reachable only via `link-tickets`. `rank_by_pagerank()` (`src/trelix/retrieval/graph.py`) now also pulls each symbol's `generic_edges` targets via `db.get_generic_edge_targets(symbol_id)` and adds them as bidirectional `symbol<->source_ref` edges into the same PageRank graph as CALLS edges — one-way `symbol->ticket` edges would only boost the ticket node since PageRank propagates via incoming edges. Results are filtered to `isinstance(sid, int)` afterward to drop the synthetic ticket nodes before returning. The PageRank algorithm itself is unchanged — plain `nx.pagerank(G, alpha=0.85)` with uniform teleportation, no Personalized PageRank — only new edges were added to the existing graph. The same bidirectional pattern is mirrored in `CodeGraph._build()` (`graph/code_graph.py`).
- **Jira/TestRail artifact connectors** — new `ArtifactSource` ABC (`src/trelix/indexing/connectors/base.py`, abstract `validate_config()`/`fetch()`, concrete `sync(db_writer: ArtifactWriter) -> ConnectorSyncResult` orchestrating validate → fetch → `db_writer.upsert_artifact()` per item) and an `ArtifactWriter` `Protocol` (`upsert_artifact(artifact: Artifact) -> int`) — deliberately not built on `BaseParser`, since tickets/test cases have no file-id/line-span concept. New env-var-only config in `src/trelix/core/config.py`: `JiraConnectorConfig` (`base_url`/`email`/`api_token`/`project_key` via `TRELIX_JIRA_*`, `page_size` default 100) and `TestRailConnectorConfig` (`base_url`/`username`/`api_key`/`project_id` via `TRELIX_TESTRAIL_*`, `page_size` default 250, capped at 250). New CLI command `trelix connector sync <repo> <jira|testrail>` (new `connector` Typer sub-app, `src/trelix/cli/main.py`) — no REST route was added. This phase writes only to the new `artifacts` table via `db.upsert_artifact()`; it deliberately does **not** create `generic_edges` rows — artifact-to-code linkage is `trelix link-tickets`'s job (or a future writer's), so Jira/TestRail artifacts are indexed but not yet wired into the PageRank graph.
- **`tests/eval/cross_source_pagerank_eval.py`** — a standalone decision-support script (run manually via `python -m tests.eval.cross_source_pagerank_eval`, not wired into any `Makefile` target or CI workflow) measuring the effect of the new cross-source edges on retrieval and ranking. Against `TRELIX_SELF_CASES` (50 queries): recall@5 0.540→0.580 (+0.040), MRR 0.274→0.296 (+0.022), NDCG@10 0.362→0.393 (+0.031), with 4,011 edges inserted across 2,438 distinct symbols and 16 distinct `#NNN` tickets; of 25 seed symbols, 22 gained a `generic_edge`, and 95.45% of those (21/22) saw their PageRank rank position change (mean |delta| 2.77, median 2.00 positions). This is eval evidence that the new bidirectional edges measurably help ranking — it is **not** an implementation of Personalized PageRank, which remains unbuilt; the underlying algorithm is still plain uniform-teleportation `nx.pagerank`.

### Security
- **`/parse` path traversal (unauthenticated arbitrary file read)** — the disk-backed branch of `/parse` took `body.file_path` and joined it under `repo_path` only when it wasn't already absolute, without ever resolving or checking containment. An absolute `file_path` (e.g. `/etc/passwd`, or any file the server process could read) was read verbatim regardless of `repo_path`, and a relative path containing `../` segments could walk out of `repo_path` entirely — the response then leaked real file contents back to the caller as parsed symbols. Because `TRELIX_API_AUTH_TOKEN` is unset by default (see Added, above), this was exploitable with no authentication on any deployment that hadn't explicitly opted into the new API auth. Fixed by resolving both `repo_root` and the target `path` via `Path.resolve()` and rejecting with `400` unless `path.is_relative_to(repo_root)` — the same containment check `/graph/visualize` already used, deliberately `is_relative_to()` rather than a string-prefix match so a sibling directory like `<repo>-evil` can't falsely pass.

### Fixed
- **Re-running `trelix link-tickets` on the same repo duplicated `generic_edges` rows** — `GitLinker.link()`'s in-memory `seen` set only deduped within a single call, so a cron re-sync or simply re-running the CLI twice inserted a second identical row per `(symbol, ticket)` pair. Fixed by adding `CREATE UNIQUE INDEX idx_generic_edges_dedup ON generic_edges(from_symbol_id, source_ref, edge_kind)` and changing `insert_generic_edges()` from a plain `INSERT` to `INSERT OR IGNORE`, so a re-run is now a DB-level no-op. New regression test `test_rerunning_link_on_same_repo_does_not_duplicate_edges` (`tests/unit/test_git_linker.py`) fails at 2 rows without the fix and passes at 1 row with it.
- **`ghcr.io/sairam0424/trelix:X.Y.Z-local` Docker image failed to build/publish** —
  `Dockerfile`'s `pip install ".[serve,local]"` resolved PyPI's default `torch`
  wheel, which bundles the full CUDA/NVIDIA runtime (`nvidia-cublas`,
  `nvidia-cudnn`, `cuda-toolkit`, ...) even though this container never has
  GPU access. That bloat exhausted the GitHub-hosted runner's disk during the
  emulated `linux/arm64` build (`OSError: [Errno 28] No space left on device`,
  surfaced on the v2.9.0 release). Fixed by adding
  `--extra-index-url https://download.pytorch.org/whl/cpu`, so pip resolves
  torch's CPU-only wheel (~104MB vs. multi-GB with the CUDA stack) instead —
  a no-op for the slim (`EXTRAS=serve`) variant, which never installs torch.
  Also added a `workflow_dispatch` trigger to `docker-publish.yml` to backfill
  a version whose Docker publish failed without cutting a new release tag.
- **`docker-publish.yml`'s `workflow_dispatch` backfill never moved `:latest`/
  `:latest-local`**, even when backfilling the current release's own missing
  variant (e.g. `2.9.0-local` right after `2.9.0` shipped) — the guard added
  to prevent an *older*-release backfill from repointing `:latest` backwards
  was too broad and blocked the common case too. Added an explicit
  `move_latest` boolean input (default `false`) so backfilling the current
  release can opt in, while backfilling an older one still can't touch
  `:latest` by default.
- **Every CLI command and the MCP `index_codebase` tool silently overrode
  `TRELIX_EMBEDDER_PROVIDER`** — `--provider` always defaulted to the literal
  string `"local"` rather than `None`, and `EmbedderConfig` is a
  pydantic-settings model where an explicit constructor kwarg always wins
  over the env var. A user who set `TRELIX_EMBEDDER_PROVIDER` (the
  documented way to configure a default provider) and never touched
  `--provider` on a given invocation would silently index/search with the
  local embedder instead, with no warning — surfacing later as a confusing
  `DimensionMismatchError` once a real provider's vectors got mixed in.
  Fixed by defaulting `--provider`/`provider` to `None` everywhere and only
  passing it to `EmbedderConfig` when actually supplied, via a new
  `_build_embedder_config()` helper (`src/trelix/cli/main.py`) used by all
  7 CLI commands plus the MCP server's `index_codebase` tool.
- **CLI error messages containing literal square brackets rendered mangled
  or misleading** — e.g. the local-embedder-missing error's fix instruction,
  `pip install 'trelix[local]'`, rendered as `pip install 'trelix'` (the
  exact broken command the user had already run), because Rich interprets
  `[...]`-shaped substrings in `console.print()` calls as markup tags and
  silently strips/mangles unrecognized ones. Every `err_console.print(f"[red]...{exc}...")`
  call site in `src/trelix/cli/main.py` (34 sites) now routes through a new
  `_print_error()` helper that escapes the dynamic text via
  `rich.markup.escape()` before interpolation, preserving literal brackets
  while keeping the surrounding color markup intact.
- **macOS PyInstaller binary crashed on every real `index`/`search` call**
  with `'sqlite3.Connection' object has no attribute 'enable_load_extension'`
  — macOS's system-style Python (what `actions/setup-python` provisions on
  GitHub's macOS runners) builds its bundled `sqlite3` without
  `SQLITE_ENABLE_LOAD_EXTENSION`, which `sqlite-vec` requires to load its
  extension. Reproduced identically on both v2.8.1 and v2.9.0 binaries —
  pre-existing, not a v2.9.0 regression, but the binary distribution's core
  workflow was completely non-functional with zero warning to users.
  `build-binaries.yml`/`release.yml`'s macOS job now provisions Python via
  `astral-sh/setup-uv` instead (a `python-build-standalone` interpreter,
  confirmed to support `enable_load_extension`); Windows/Linux are
  unaffected and unchanged. All three platforms' binary-verify steps now
  also run a real `index --provider openai` smoke test (with a bogus API
  key — the bundled binary excludes `sentence-transformers`/`torch`, so
  `--provider local` can never work there; `SQLiteVectorStore`'s
  `sqlite-vec` load happens before the network call, so this still
  exercises the regression without needing real network egress) and
  asserts the crash string is absent plus `trelix stats` shows real
  progress, so a regression like this fails CI immediately instead of
  shipping silently.
- **`astral-sh/setup-uv` still didn't fix the macOS binary — `uv venv
  --python 3.11` picked up the runner image's pre-installed
  `/usr/local/bin/python3.11` (Homebrew) instead of downloading its own
  `python-build-standalone` interpreter**, silently defeating the previous
  fix; confirmed via a real CI run where the `enable_load_extension` crash
  came right back. `uv` prefers an already-discoverable interpreter over
  downloading a new one unless told otherwise. Added `--managed-python` to
  force `uv` to use its own downloaded interpreter.
- **Windows release binary crashed on every command that renders Rich's
  progress spinner** with `'charmap' codec can't encode character
  '⠋'` — Rich's default spinner uses Unicode braille glyphs, which
  Windows' legacy console codepage (`cp1252` etc.) can't encode. Surfaced
  by the new binary smoke test above (the old `--help`-only check never
  rendered a spinner). Pre-existing, not a v2.9.0 regression — any real
  Windows user in a non-UTF-8 terminal would hit this. Fixed by
  reconfiguring `sys.stdout`/`sys.stderr` to UTF-8 with
  `errors="replace"` at CLI startup, before any `rich.Console` is
  constructed — a no-op on terminals already using UTF-8 (macOS/Linux,
  or Windows Terminal with UTF-8 active).
- **The binary smoke test itself assumed `trelix index --provider openai`
  with a bogus key always exits 0** — it doesn't: whether Phase 2's
  per-file summary/sub-chunk embedding (both off by default) or Phase 3's
  batch embed hits the bad key first determines the exit code, and
  Phase 3's `embed_async()` call has no local `try`/`except`, so a real
  auth failure there propagates all the way to `typer.Exit(1)`. Wrapped
  the `index` invocation in a subshell with `|| true` so the smoke test's
  own exit code no longer depends on whether the deliberately-invalid
  API key happens to fail before or during the real embedding phase —
  only the crash-string check and the follow-up `stats` progress check
  matter.

## [2.9.0] — 2026-07-24

### Overview
Seven backlog items shipped: TypeScript SDK, typed REST API responses + `/search`
pagination, OpenTelemetry tracing, Python 3.13 support, Docker/Helm deployment,
VS Code extension hardening + search refinement, and GitHub App GA hardening.
1,643 unit tests passing, all features additive/opt-in — no breaking changes
(the `flare_max_iterations` removal originally slated for v3.0.0 remains
deliberately deferred; see `docs/ROADMAP.md`).

### Added
- **`@trelix/sdk` TypeScript client** (`packages/trelix-typescript`) — a hand-written
  HTTP client for the `trelix serve` REST API, with types generated from the
  live OpenAPI schema (`openapi-typescript`) and a thin, hand-glued
  `TrelixClient` class on top. Covers every route (`health`, `search`
  with cursor pagination, `index`, `stats`, `graph`, `graph/communities`,
  `graph/visualize`, `graph/search`) plus a separate `askStream()` async
  generator for `/ask`'s SSE stream (tokens, `[DONE]`, and a
  `TrelixAskError`-throwing `[ERROR: ...]` frame).
- **Typed REST API response models** — every `trelix serve` route now
  declares a real Pydantic response model (`SearchResponse`,
  `IndexResponse`, `StatsResponse`, `GraphStatsResponse`,
  `CommunitySummaryModel`, `GraphVisualizeResponse`,
  `GraphSearchResultModel`), so the auto-generated OpenAPI schema
  (`/openapi.json`) carries real per-field types instead of an untyped
  `object`/`array`. Groundwork for an OpenAPI-codegen'd TypeScript client.
- **`GET /search` cursor pagination** — gained a `cursor` query param and now
  returns `{results, next_cursor, total_available}` instead of a bare list,
  matching the MCP `search_code` tool's existing pagination contract
  exactly. This is the one deliberate, narrowly-scoped response-shape
  change in an otherwise additive pass — done now, before a TS SDK locks in
  against the old bare-list shape.
- **OpenTelemetry tracing for the retrieval pipeline** — opt-in via
  `pip install trelix[otel]` + `TRELIX_OTEL_ENABLED=true` (off by default,
  zero import cost and zero behavior change when disabled). Emits one
  `gen_ai.*`-conventions retrieval span per leg (vector, BM25, grep, sparse,
  sub-chunk, file-summary) via `opentelemetry-util-genai`'s
  `TelemetryHandler.retrieval()`, plus `trelix.*`-namespaced pipeline-stage
  spans (planner, fusion, expansion, rerank, pagerank boost, assembly).
  Correctly nests leg spans under the query's root span across the
  `ThreadPoolExecutor` boundary used for parallel sub-query execution (OTel's
  context is contextvars-based and does not cross thread pools on its own).
  New optional `OTEL_EXPORTER_OTLP_ENDPOINT` exports to any OTLP collector.
  See `docs/OBSERVABILITY.md` for the full span reference and a stability
  caveat (the `gen_ai.*` conventions are officially adopted but still
  "Development," not yet "Stable," upstream).
- **Python 3.13 support** — `requires-python` no longer caps at `<3.13`.
  The only blocker was `tree-sitter-languages` (abandoned upstream, no
  cp313 wheels); swapped for the actively-maintained, API-compatible
  `tree-sitter-language-pack` behind the single existing chokepoint,
  `src/trelix/indexing/parser/_grammar.py`. Bumped `tree-sitter>=0.23`
  and `pydantic>=2.8.0` (3.13-compatible floors). CI matrix now runs
  3.11/3.12/3.13.

### Changed
- **Tree-sitter grammar loading is now network-on-first-use, not bundled.**
  `tree-sitter-language-pack` fetches each language's compiled grammar
  over the network on first use and caches it locally (unlike the old
  `tree_sitter_languages`, which bundled every grammar in its wheel).
  Call `trelix.indexing.parser._grammar.prefetch_all()` once (e.g. during
  image build or CI setup) to warm the cache so indexing itself never
  needs network access — CI does this automatically now.

### Fixed
- **`docs/INSTALLATION_GUIDE.md`'s `trelix serve`/Docker examples used the
  wrong port** (8080) and a nonexistent `--repo` flag — `serve`'s actual
  CLI default is port 8765 with a positional `repo_path` argument. Also
  removed two fabricated env vars (`TRELIX_SERVE_HOST`, `TRELIX_SERVE_PORT`)
  that don't exist anywhere in source; there is no env-var override for the
  serve host/port, only the `--host`/`--port` CLI flags.
- **C# grammar naming**: `csharp.py` was requesting the language as
  `c_sharp`; the correct name is `csharp`. Silently broken until this
  release since `tree_sitter_languages` happened to accept both.
  Regression-tested via the existing `test_parser_csharp.py` suite.
- **Kotlin extractor rewritten for the new grammar's AST shape** —
  `class_declaration`/`function_declaration`/etc. no longer expose
  tree-sitter field names (`child_by_field_name` returned `None`
  everywhere), silently breaking all Kotlin class/interface/enum/
  function/property extraction. Rewritten to walk children positionally.
- **Python docstring/module-constant extraction** — the new grammar
  drops the `expression_statement` wrapper node entirely; assignments,
  calls, and bare strings (docstrings) now appear as direct children of
  their block. Updated `python.py`'s dispatch and `_get_docstring` to
  match the unwrapped shape.
- **Go interface methods**: `method_spec` renamed to `method_elem`.
- **TypeScript interface bodies**: `object_type` (for interface bodies
  specifically — type-alias object literals are unaffected) renamed to
  `interface_body`.
- **C# `using` alias imports**: the RHS type of `using X = Y.Z;` was
  wrapped in a `name_equals` node with an `alias` field in the old
  grammar; the new grammar flattens it to plain siblings around `=`.
  Updated `_extract_using` to take the first named type node after `=`.
- **`trelix.spec` (PyInstaller build) still imported the retired
  `tree_sitter_languages` package**, missed by the tree-sitter-language-pack
  migration since the binary build pipeline runs in its own workflow, not
  under `pytest`/mypy/ruff. Broke `Build Binaries` on every push to
  `develop`/`main` with `ModuleNotFoundError: No module named
  'tree_sitter_languages'`. Dropped the now-nonexistent package's `datas`/
  `hiddenimports` entries (it ships no bundled grammar data — grammars are
  fetched into an OS cache dir at runtime) and added `tree_sitter_language_pack`
  as a hidden import instead. Verified via a local `pyinstaller trelix.spec
  --clean --noconfirm` build + `dist/trelix --help`.

### Security
- **GitHub App: payload size limit and subprocess timeout** — the webhook
  route now caps request bodies at 25MB (GitHub's own documented webhook
  payload cap), rejecting oversized bodies with `413` during parsing
  rather than buffering an arbitrarily large request into memory; this
  matters because signature verification happens *after* body parsing, so
  the size limit is the only defense against a sender who doesn't know
  the webhook secret sending a deliberately huge payload.
  `runReviewCli`'s `trelix review` shell-out now passes a 5-minute
  `timeout`, so a hung/slow review (LLM synthesis latency, a huge diff, a
  stuck index) no longer ties up the process indefinitely — Node kills
  the child process (`SIGTERM`) and the call rejects. New tests exercise
  both with real subprocesses/payloads rather than mocks: a genuinely
  slow shell shim proves the timeout actually kills the process, and a
  real 26MB request body proves the size limit actually rejects.
- **GitHub App: webhook signature verification** — `infra/github-app/src/webhook.ts`
  now verifies `X-Hub-Signature-256` (HMAC-SHA256 over the raw request
  body, keyed by the webhook secret) via `@octokit/webhooks-methods`'s
  `verify()`, which compares using `crypto.timingSafeEqual` rather than a
  naive string compare (avoids leaking timing information about how many
  leading bytes matched). Requests with a missing, wrong-secret, or
  tampered-after-signing body are rejected with `401` before the route
  handler — and therefore `runReview`/the trelix CLI shell-out — ever sees
  the payload. New tests cover both the accept-valid and reject-tampered
  paths explicitly (the common real bug is only testing the happy path):
  no-header, wrong-secret, and tampered-body all assert `401` +
  `runReview` never called; a correctly-signed control case asserts `202`.

### Added
- **GitHub App: GA-readiness docs polish** — `infra/github-app/README.md`
  finalized (production deployment notes: HTTPS requirement, secret-manager
  guidance, runtime prerequisites) and its status upgraded from
  "skeleton"/"auth wired" to "installable and hardened" now that items
  6a-6c are complete. `docs/ROADMAP.md`'s "GitHub App GA" line explicitly
  states this App is installable and hardened, **not** Marketplace-listed
  — Marketplace paid-app listing has its own separate business/adoption
  requirements out of scope for this engineering work.
- **GitHub App: installation-token minting + Check-annotation posting**
  (`infra/github-app/src/auth.ts`, `src/review-runner.ts`) — completes the
  auth/posting work stubbed in item 6a. `getInstallationToken` uses
  `@octokit/auth-app` (App-ID+private-key JWT signing -> installation-token
  exchange), with one `AuthInterface` reused per `AppConfig` (a `WeakMap`)
  so the library's own expiry-aware cache actually has a chance to hit
  across calls — verified with mocked-transport tests proving a second
  call for the same config+installation makes zero additional HTTP
  requests, while distinct installations/configs never share a cached
  token. `runReview` now mints a token, fetches the PR's head SHA via
  `octokit.rest.pulls.get`, runs `trelix review --pr ... --json`, and
  posts a completed Check run with inline annotations via
  `octokit.rest.checks.create` (same conclusion logic as the existing
  `trelix-review.yml` workflow: any `failure`-level annotation ->
  `failure`, else `success`).
- **GitHub App skeleton** (`infra/github-app/`, `@trelix/github-app`) — the
  start of a standalone, webhook-driven GitHub App for zero-setup PR
  review (install the App, no workflow YAML needed in the installing
  repo), per the explicit architecture decision to build a standalone
  webhook-to-direct-execution service rather than a thin bridge to the
  existing Actions workflow. Ships `manifest.yml` (same
  `pull_requests`/`checks`/`contents` permissions and `pull_request`
  event the existing `trelix-review.yml` workflow already uses), an
  Express server with `/health` and `/webhooks/github`, webhook routing
  for `pull_request` `opened`/`synchronize`/`reopened` (mirroring the
  existing workflow's trigger), and a review-runner that shells out to
  `trelix review --pr ... --json` and maps findings to GitHub Check
  annotations (`toAnnotations` — a TypeScript port of the mapping logic
  fixed in the `trelix-review.yml` workflow). **Not yet wired for
  production use**: signature verification, installation-token minting,
  and Check-annotation posting are explicitly stubbed/unimplemented —
  land in item 6b. `infra/github-app/README.md` rewritten to cover both
  integration paths (the existing Actions workflow and this new App) so
  its previous "no App registration required" framing doesn't read as
  false now that a real App skeleton exists. New
  `.github/workflows/github-app-ci.yml` runs
  `npm ci && npm run typecheck && npm run build && npm test`, gated on
  `infra/github-app/**`.
- **VS Code extension: live-narrowing search + snippet preview** —
  `trelix.search` is now a debounced (250ms) search-as-you-type
  `QuickPick` instead of a one-shot `showInputBox` → static-list flow.
  Highlighting a result (arrow keys, not just accepting) shows a real
  snippet preview via `showTextDocument({preview: true})` against a new
  virtual-document `TextDocumentContentProvider`
  (`trelix-preview:` scheme, `src/preview.ts`) — this gets genuine VS Code
  syntax highlighting for free, since the virtual URI keeps the real
  file's extension. A `"Load more results…"` pseudo-item appears whenever
  `search_code`'s `next_cursor` is non-null (using the pagination fields
  PR #81/item 5a fixed), fetching and appending the next page without
  losing the current results or query. The debounce/cursor-pagination/
  stale-response-rejection state machine lives in a new, Extension-Host-
  independent `SearchController` class (`src/search-controller.ts`) —
  `search-controller.test.ts` uses an injectable fake-timer harness to
  simulate rapid keystrokes and prove `search()` fires exactly once per
  debounce window (not once per keystroke), that a stale in-flight
  response is discarded once a newer query has superseded it, and that
  `loadMore()` correctly appends via `next_cursor` and no-ops when there
  isn't one.
- **`docs/ROADMAP.md`**: logged the original Phase 3 plan's `@trelix` chat
  participant + hover providers (never actually delivered — only the 2
  QuickPick/Webview commands shipped) as an explicit v3.1.0 candidate,
  rather than silently dropping it again.

### Security
- **VS Code extension: XSS in the `trelix.ask` Webview** — `panel.webview.html`
  interpolated the raw, unescaped LLM answer string directly, with the
  Webview's `options` an empty `{}` (no CSP, no `enableScripts: false`, no
  `localResourceRoots` restriction at all). A crafted or adversarial answer
  could execute arbitrary script in the Webview's context. Now HTML-escapes
  the answer before interpolation and sets `enableScripts: false` plus an
  explicit `default-src 'none'` CSP meta tag.

### Fixed
- **VS Code extension: `search_code` results were silently mis-parsed** —
  `mcp-client.ts`'s `search()` read `symbol_name`/`file_path` off each
  result, but the real MCP `search_code` tool's response keys are
  `symbol`/`file` (confirmed against `packages/trelix-mcp/src/trelix_mcp/
  server.py`) — those two fields were always empty strings, and clicking a
  search result opened a broken/empty file URI. Also fixed: `next_cursor`/
  `total_available` were parsed off the response but discarded entirely
  (`search()` returned only `parsed.results`), and `kind`/`lines`/`source`/
  `language` were dropped from the parsed shape though the server already
  returns them. `search()` now returns the full `{results, nextCursor,
  totalAvailable}` shape with every field; `extension.ts` uses the newly
  available `lines` field ("start-end", 1-indexed) to jump to and highlight
  the matched symbol's line range on open, instead of just opening the file
  with no selection.
- **`trelix review --pr ... --json`'s stdout was never valid JSON** —
  `console.print(...)` status/progress messages (e.g. "Fetching PR diff
  from GitHub...") ran unconditionally to stdout even in `--json` mode,
  and `"No issues found."`/`"No textual changes..."` styled messages ran
  *instead of* an empty `[]` when there were zero comments. Combined with
  `.github/workflows/trelix-review.yml`'s `> file 2>&1` redirect, the
  review-posting Check's `JSON.parse()` has always thrown and been
  silently swallowed by a `try/catch` — meaning **the "trelix Code
  Review" Check has never posted a single real annotation** since this
  workflow shipped. All `--pr --json` status/progress messages now go to
  `err_console` (stderr); the workflow now redirects only stdout, keeping
  stderr in a separate log for debugging.
- **The same workflow's annotation-posting logic never matched trelix's
  real output shape even when parsing succeeded** — it read
  `data.findings || data.reviews || []` against `trelix review --json`'s
  real bare-array output (never matches, so `findings` was always `[]`
  regardless), and compared `f.severity === 'error'`/`'warning'`
  (lowercase) against the real values `"ERROR"`/`"WARN"`/`"INFO"`
  (uppercase — `'WARN' !== 'warning'` either way). Every annotation would
  have posted as `notice` severity even if the JSON had parsed. Now reads
  the real `{file, lines, severity, comment}` shape directly and maps
  `ERROR`→`failure`, `WARN`→`warning`, `INFO`→`notice`.
- New `tests/unit/test_review_pr_json.py` (4 tests) — regression-tests
  `--json` stdout purity for the has-comments, zero-comments, and
  no-textual-changes paths, plus confirms non-`--json` mode still prints
  status messages to stdout (the fix is `--json`-gated, not a blanket
  behavior change). Verified these tests actually fail against the
  pre-fix code (3/4 failed with the exact `JSONDecodeError` this bug
  produces) before confirming they pass against the fix.

### Changed
- **VS Code extension build/test infrastructure** — added `esbuild`
  (bundles `dist/extension.js`, `external: ["vscode"]`) instead of plain
  `tsc` emit, so the packaged `.vsix` no longer risks shipping unbundled
  `node_modules` (the extension's only runtime dependency,
  `@modelcontextprotocol/sdk`). `tsc --noEmit` remains a separate
  `typecheck` script since esbuild doesn't type-check. Added a
  `.vscodeignore` (previously absent) and a `@vscode/test-electron`+Mocha
  test harness (`src/test/runTest.ts`, `src/test/suite/`) — new
  `extension.test.ts` verifies activation and command registration;
  `mcp-client.test.ts` verifies the `search()`/`ask()` parsing fixes above
  against a mocked MCP transport. New
  `.github/workflows/vscode-extension-ci.yml` runs
  `npm ci && npm run typecheck && npm run build && xvfb-run -a npm test`,
  gated on `workspace-vscode/**` changes. Version bumped `0.1.0` → `0.2.0`
  (unchanged since the v2.7.0 scaffold).
- **`docs/integrations/vscode-plugin.md` full rewrite** — the previous
  version described a PyInstaller-binary-bundling architecture that was
  never actually built, and never once mentioned the real MCP-stdio-client
  architecture the extension actually ships with. Rewritten to describe
  the real `dist/extension.js` (esbuild bundle) → `trelix-mcp` (stdio
  child process) → trelix core data flow, the real `search_code` response
  shape, the security notes above, and the real build/test/package
  commands.

### Added
- **Helm chart** (`helm/trelix/`) for deploying `trelix serve` to Kubernetes —
  `Deployment`/`Service`/`PVC`/`Secret`/`Ingress` templates, `values.yaml`
  covering the full `StoreConfig` surface (`store.backend`: sqlite/qdrant/
  lance, HNSW tuning, BM25 read-pool size) plus embedder-provider
  credentials (OpenAI/Voyage/Cohere, either plaintext `apiKey` for dev or
  `existingSecretName`/`existingSecretKey` for shared clusters). Models
  `trelix serve`'s actual behavior directly: since `create_app()` takes zero
  arguments and every route re-derives its config from the request's own
  `repo` param, one Deployment is already multi-repo-capable — the chart's
  PVC (mounted at `/data` by default) is a *shared* data directory across
  every repo you index/serve, documented loudly in `NOTES.txt`/`README.md`
  since it's non-obvious. `ingress.enabled` defaults to `false`: `trelix
  serve` has zero auth middleware, so `NOTES.txt` warns explicitly about
  exposing `/index`/`/ask`/`/search` before enabling a public Ingress.
  Qdrant is treated strictly as an external, user-managed dependency — this
  chart only points `QDRANT_URL`/`QDRANT_API_KEY` at one, never deploys or
  operates Qdrant itself (its own chart states support is
  community-limited; self-hosted lacks zero-downtime upgrades and
  backup/DR). New `.github/workflows/helm-lint.yml` runs `helm lint` +
  `helm template` across all three `store.backend` values plus an
  ingress-enabled render, on every push/PR touching `helm/**`.
- **Official Docker image** — a multi-stage `Dockerfile` (root) publishes
  `ghcr.io/sairam0424/trelix` for `linux/amd64`+`linux/arm64` on every
  release tag, in two variants sharing one build (`EXTRAS` build arg):
  `:X.Y.Z` (slim, API-embedder-only — OpenAI/Voyage/Cohere/Azure) and
  `:X.Y.Z-local` (bundles `sentence-transformers`/`torch` for the
  local/offline embedder and cross-encoder reranker). Runs as a non-root
  `trelix` user, `ENTRYPOINT ["trelix"]` with `CMD ["serve", "/repo",
  "--host", "0.0.0.0", "--port", "8765"]` (overrides the CLI's
  `127.0.0.1` default, which isn't reachable from outside a container's
  network namespace), and a `HEALTHCHECK` hitting `/health`. New
  `docker-compose.yml` at the repo root is a runnable version of
  `docs/INSTALLATION_GUIDE.md`'s Docker Compose snippet. New
  `.github/workflows/docker-publish.yml` builds/pushes both variants on
  `v*` tags; CI gained a `docker-build` job that builds the slim image and
  runs `--help` against it on every push/PR, mirroring the existing
  per-OS binary `--help` smoke tests in `release.yml`.
- New Makefile targets: `docker-build`, `docker-build-local`, `docker-run`.

### Fixed
- **`TRELIX_EMBEDDER` was a silent no-op env var** — `docs/
  INSTALLATION_GUIDE.md` and `docker-compose.yml` both referenced
  `TRELIX_EMBEDDER`, but `EmbedderConfig`'s real env var is
  `TRELIX_EMBEDDER_PROVIDER` (confirmed empirically: setting
  `TRELIX_EMBEDDER=openai` in a clean environment left `provider` at its
  default `"local"`). On the slim Docker image this silently falls back to
  a provider that isn't installed and crashes, rather than erroring at the
  variable name. Also fixed a `--embedder` CLI flag reference in the same
  section — the real flag is `--provider`. Found while writing this
  chart's `values.yaml` example and wanting to confirm the var name against
  source before using it.
- **`docs/INSTALLATION_GUIDE.md`'s Docker Compose/serve examples used the
  wrong port** (8080) and a nonexistent `serve --repo` flag (`repo_path`
  is positional) — same class of bug already fixed for the `docker run`
  examples in PR #77, now fixed here too since this PR touches the same
  section.

## [2.8.1] — 2026-07-20

### Security
- **MCP federation `config_path` path confinement** — `federation_list_repos`/
  `federation_add_repo`/`federation_remove_repo`/`federation_search_all`
  previously passed a caller-supplied `config_path` straight into
  `RepoRegistry.load()`/`.save()` with no validation, letting an MCP client
  (including a prompt-injected agent) point registry I/O at an arbitrary
  path. Now confined to `~/.config/trelix/` or `<mcp-server-cwd>/.trelix/`
  via `Path.is_relative_to()` (not a naive string-prefix check, which would
  incorrectly also match a sibling directory like `~/.config/trelixevil/`).
  Found in the pre-push audit of v2.8.0 (issue #69).

### Added
- **Federation repo-count and fan-out caps** — `RepoRegistry.add()` gained an
  optional `max_repos` parameter (CLI callers remain unbounded by default;
  MCP's `federation_add_repo` now passes `TRELIX_FEDERATION_MAX_REPOS`,
  default 50). `FederatedRetriever` gained a `max_repos` constructor param
  capping how many registered repos are actually queried per call;
  `federation_search_all`'s response gained a `repos_skipped` field.
  Prevents a runaway/adversarial `federation_add_repo` loop from making
  every subsequent search scale linearly with an unbounded repo count.

### Fixed
- **`federation_search_all` pagination wasn't a stable slice** — previously
  requested `fed.retrieve(query, k=max(k+cursor, k))`, so the per-repo
  candidate pool feeding RRF fusion widened as `cursor` grew, meaning page 2
  could be fused from a differently-shaped pool than page 1 (items could
  shift rank, get deduped differently, or disappear between pages). Now
  fetches a fixed, cursor-independent width once and slices pages from the
  final fused list — mirrors `search_code`'s existing single-fetch-then-slice
  pattern.

### Changed
- All 4 federation MCP tools now consistently return an `"error": str|None`
  key on every response path (previously only present on failure paths for
  `federation_add_repo`), matching the convention already used by
  `ask_agent`/`agent_list_sessions`/`agent_clear_session`.

## [2.8.0] — 2026-07-20

### Added
- **Multi-repo support in MCP** — 4 new MCP tools (`federation_list_repos`,
  `federation_add_repo`, `federation_remove_repo`, `federation_search_all`)
  expose the existing `RepoRegistry`/`FederatedRetriever` CLI infrastructure
  (`trelix federation add/list`, `trelix search-all`) to MCP clients (Claude
  Desktop, Cursor, any IDE). Also added the missing `trelix federation remove`
  CLI command (the registry method existed but had no CLI entry point).
- **Persistent agent (ReAct loop) memory** — the agentic loop
  (`trelix ask --agentic`, `TRELIX_RETRIEVAL_AGENTIC=true`) now persists turn
  history to new `agent_sessions`/`agent_turns` tables in the per-repo
  `.trelix/index.db`, keyed by a client-supplied or auto-generated UUID4
  `session_id`. `AgentLoop.run()` now returns `(answer, session_id)` — pass
  the session_id back on a follow-up call to resume with full prior context.
  New CLI: `trelix ask --session <id>`, `trelix agent sessions list/show/clear`.
  New MCP tools: `ask_agent`, `agent_list_sessions`, `agent_clear_session`.
  Sessions auto-evict after `TRELIX_RETRIEVAL_AGENT_SESSION_MAX_AGE_SECONDS`
  of inactivity (default 7 days; `0` disables eviction).

### Fixed
- **Federated search lost repo provenance** — `FederatedRetriever` used to tag
  each result's `source` with `"{alias}:{leg}"` so callers could tell which
  repo a result came from, but this was silently dropped in a prior refactor.
  `trelix search-all`'s "Repo" column and `--json` output had been blank ever
  since, with no test catching it. Restored the tagging and added a
  regression test.
- **`RepoEntry.weight` was never applied** — settable via
  `trelix federation add --weight`, stored, and documented, but the fan-out
  fusion path never forwarded it into RRF, so per-repo weighting silently did
  nothing. `reciprocal_rank_fusion()` gained a new `list_weights` parameter
  (orthogonal to the existing per-language `weights` parameter; `None` is
  backward-compatible) and `FederatedRetriever` now passes each repo's weight
  through.
- **`agent_turns.turn_index` could silently collide on session resume** — found
  in pre-push audit. `AgentLoop.run()` used to compute the resume anchor from
  `len(prior_rows)` (a row-count snapshot), which drifts from reality after
  any persistence gap (a dropped turn) or a concurrent resume of the same
  `session_id`, silently producing duplicate `turn_index` rows with no error.
  `Database.insert_agent_turn()` now assigns `turn_index` atomically via
  `MAX(turn_index)+1` under the same lock as the insert, and `agent_turns`
  gained a `UNIQUE(session_id, turn_index)` index as defense-in-depth — any
  residual race now raises `IntegrityError` (caught and logged) instead of
  silently duplicating a row.

## [2.7.3] — 2026-07-13

### Changed
- **README.md end-to-end audit and rewrite** — fixed 15+ factual bugs (wrong
  env var names, fabricated pip extras, a broken Homebrew tap, a crash-causing
  `TRELIX_RETRIEVAL_RERANK_PROVIDER` value, wrong REST method/table names),
  rewrote the "How it works" diagram to show all 7 retrieval legs (was 3) plus
  the agentic/FLARE alternate synthesis modes, and consolidated duplicated
  content (3x REST API sections, Installation/Knowledge-Graph/Embedding-Providers
  duplicating `docs/`) into short pointers. 867 → 634 lines.
- **"What's New" and "Troubleshooting" moved out of README** — backfilled
  CHANGELOG.md's empty `[2.2.0]` entry with its 5 features (agentic ReAct loop,
  data-flow analysis, taint analysis, sparse+dense hybrid, multi-granularity
  indexing — previously undocumented anywhere else) and added README's 5
  Troubleshooting entries to `docs/TROUBLESHOOTING.md`'s existing sections,
  then trimmed both README sections to short pointers.

## [2.7.2] — 2026-07-12

### Added
- **Qdrant Cloud readiness** — `QdrantVectorStore` now accepts `prefer_grpc` and
  `timeout` options, wired through `StoreConfig.qdrant_prefer_grpc`
  (`QDRANT_PREFER_GRPC`, default `false`) and `StoreConfig.qdrant_timeout`
  (`QDRANT_TIMEOUT`, default `10.0`). Enables gRPC transport (port 6334) and
  longer request timeouts for Qdrant Cloud's higher network latency.
- **Incremental per-symbol embedding on partial re-index** — new
  `symbols.content_hash` column (`sha256(signature + body)`, backfilled via an
  `ALTER TABLE ... ADD COLUMN` migration guard). `Indexer._insert_one` now diffs
  each parsed symbol's `(qualified_name, content_hash)` against the stored row;
  unchanged symbols skip delete/re-chunk/re-embed entirely and keep their
  existing chunk rows and vectors. Only changed or new symbols flow through the
  delete → re-insert → chunk → embed path.
- **Opt-in parallel BM25 read pool** — new `ReadOnlyConnectionPool`
  (`src/trelix/store/read_pool.py`) opens N read-only SQLite connections
  (`mode=ro`, `PRAGMA query_only = ON`) for concurrent FTS5 reads.
  `TRELIX_STORE_BM25_READ_POOL_SIZE` (default `0`, disabled) — when set > 0,
  `Retriever.__init__` calls `Database.enable_bm25_read_pool()` automatically.
- **Linux ARM64 binary releases** — `build-binaries.yml` and `release.yml`
  matrices add `ubuntu-24.04-arm` (artifact `trelix-linux-arm64`);
  `docs/INSTALLATION_GUIDE.md` gained a "Linux ARM64" install section.

### Fixed
- **`SparseEmbedder` TOCTOU race** — `_load()` checked `self._model is not None`
  before acquiring the lock, so two threads could both pass the check and
  double-load the model concurrently. Fixed with double-checked locking:
  `self._model is not None` is re-checked again inside `self._lock`.
- **MCP stdout notification write race** — concurrent `send_resource_notification()`
  calls from different threads could interleave partial JSON-RPC lines on
  stdout. Added a module-level `_stdout_lock` guarding the `sys.stdout.write()` +
  `flush()` pair.
- **`SubscriptionRegistry` unbounded growth** — subscriptions were never capped
  or expired, so a misbehaving client could grow the registry indefinitely.
  Added `max_subscribers` (`TRELIX_MCP_MAX_SUBSCRIBERS`, default `1000`)
  enforced via a new `SubscriptionLimitExceeded` exception, and `ttl_seconds`
  (`TRELIX_MCP_SUBSCRIPTION_TTL_SECONDS`, default `3600`) swept by
  `_evict_expired_locked()` before every registry operation. The
  `subscribe_resource` tool now catches the limit error and returns a soft
  `{"subscribed": false, ...}` payload instead of raising.
- **Silent `parent_id`/`callee_id`/`type_edges` corruption on partial re-index** —
  `symbols.parent_id`, `calls.callee_id`, and `type_edges.to_symbol_id` are all
  `ON DELETE SET NULL`, so deleting a changed symbol's old row silently nulled
  these links on any other row (including unchanged ones) that pointed at it.
  Added `Database.get_children_with_stale_parent`/`repoint_parent_ids`,
  `get_calls_referencing_symbols`/`repoint_call_callee_ids`, and
  `get_type_edges_referencing_symbols`/`repoint_type_edge_targets`; the indexer
  snapshots stale links before the cascading delete and repoints them to the
  replacement symbol's new id afterward.
- **Incomplete BM25 concurrency lock** — `Database._conn`
  (`check_same_thread=False`) is not safe for concurrent statement execution
  from multiple threads despite that flag; the grep, sparse, and vector
  retrieval legs all hydrate through the same shared connection from sibling
  `ThreadPoolExecutor` threads. Added `self._conn_lock` and applied it to
  `bm25_search()`'s non-pool fallback, `get_symbol_with_file()`,
  `get_first_chunk_for_symbol()`, `get_chunk_with_context()`,
  `grep_search.py`'s `_name_search`/`_body_search`, and a new locked
  `Database.get_chunk_by_id()` helper for `sparse_search.py`'s raw chunk
  lookup. Verified via a 60-thread x 10-iteration x 3-leg stress test with
  zero errors.
- **`qdrant-client` 1.18 API migration** — `QdrantVectorStore` used the
  deprecated `search()` method; migrated to `query_points()`. Pinned
  `qdrant-client>=1.9.0,<2.0.0` in `pyproject.toml` to prevent an unguarded
  2.x upgrade from breaking the client again.

### Changed
- **Windows ARM64 binary intentionally not shipped** — `windows-11-arm` was
  briefly added to both binary-build matrices, then reverted:
  `tree-sitter-languages` and `sqlite-vec` publish no `win_arm64` wheel or
  sdist, so `pip install` fails before the build ever runs. Linux ARM64 ships;
  Windows ARM64 does not.

## [2.7.1] — 2026-07-10

### Fixed
- **Release pipeline asset collision** — `release.yml` referenced the macOS and Linux
  PyInstaller binaries by bare basename (both built as `dist/trelix`).
  `softprops/action-gh-release` uploads by basename, so the two identically-named
  binaries collided into a single GitHub Release asset. The published v2.7.0 release
  had only 2 binary assets instead of 3, and it was unknowable whether the surviving
  `trelix` asset was macOS or Linux. Each binary is now renamed to a unique filename
  (`trelix-macos-arm64` / `trelix-windows-x64.exe` / `trelix-linux-x64`) before upload.
- **No Linux binary in PR-time CI** — `build-binaries.yml` only built and verified
  macOS + Windows even though `release.yml` already builds Linux at tag time. Added
  the `ubuntu-latest` matrix entry and a Linux verify step.
- **Unjustified dependency-floor bumps reverted** — `trelix-mcp`, `trelix-langchain`,
  and `trelix-llama-index` had their `trelix>=X.Y.Z` floors raised to `>=2.7.0`/
  `>=2.4.0` in v2.7.0 based on an unverified assumption about API usage. Re-checked
  every import in all three packages — none use any Phase 1–3 v2.7.0 API. Reverted
  to `trelix>=0.4.0`.
- **`trelix-mcp` tests never ran in CI** — `ci.yml`'s test job never installed or
  executed `packages/trelix-mcp/tests/`. This let a real regression sit undetected:
  `test_four_tools_registered` asserted "exactly 6 tools" when the server has
  registered 8 since `subscribe_resource`/`unsubscribe_resource` shipped in v2.5.0.
  Fixed the test's expected set and wired `packages/trelix-mcp/tests/` into `ci.yml`.
- **Wrong env var name in docs** — `TRELIX_GRAPH_SEARCH_ENABLED` was incorrect in
  7 places across `docs/FAQ.md`, `docs/USER_GUIDE.md`, `CONTRIBUTING.md`. The real
  variable is `TRELIX_RETRIEVAL_GRAPH_SEARCH_ENABLED` (`graph_search_enabled` has
  no explicit alias override, so it inherits `RetrievalConfig`'s
  `env_prefix="TRELIX_RETRIEVAL_"`).
- **CHANGELOG footer link collision** — `[2.2.0]` was defined twice with conflicting
  URLs; markdown silently resolves to the last definition, making the first dead.
  `[2.3.0]`, `[1.1.0]`, `[0.7.1]`, `[0.7.0]`, `[0.6.0]` had no comparison link at all
  despite existing as dated release headers. Rebuilt the footer from scratch,
  cross-checked against `git tag -l`.

## [2.7.0] — 2026-07-09

### Added — Phase 1: Watch Bridge, DB Index, AdaptiveRouter Config Fix
- `FileWatcher._do_reindex` now fires `notify_file_changed()` after a successful
  re-index (not on hash-identical skips). MCP subscribers receive live
  `notifications/resources/updated` pushes when watched files change.
  Non-fatal when `trelix-mcp` is not installed.
- `idx_files_rel_path` index added to `files.rel_path` — eliminates full table
  scan on every `GraphUpdater.update_file()` call (`WHERE rel_path = ?`).
  `CREATE INDEX IF NOT EXISTS` — safe on existing databases.
- `AdaptiveRouter.__init__` now accepts `retrieval_config: RetrievalConfig | None = None`.
  When provided, it is used directly instead of constructing a new instance from env
  vars — fixes silent-ignore of programmatic config overrides.
- `Retriever` passes `config.retrieval` through `QueryPlanner → AdaptiveRouter`.

### Added — Phase 2: Cross-Repo Symbol Resolution, Semantic Diff Embeddings, Streaming Indexing
- `make_scip_symbol_id(package, version, qualified_name)` — stable SCIP-style
  cross-repo symbol ID using `||`-separated sha256[:16]. Unambiguous for scoped
  npm packages (`@scope/pkg`).
- `FederatedRetriever.record_exports(alias, repo_path)` — indexes all symbols from
  a trelix-indexed repo into an in-memory `federation_symbols` table.
- `FederatedRetriever.resolve_symbol(qualified_name)` — returns all repos that
  define a symbol. Supports exact match and suffix-LIKE (`%.verify`). Thread-safe
  via `threading.Lock` + `check_same_thread=False`.
- `DiffEmbedder` — CCRep-style before/after body pair embeddings for PR diff hunks
  (arXiv:2302.03924). `store_pr_diff()` caps at 500 hunks/PR; `search_similar_diffs()`
  finds historically similar changes by cosine similarity with NaN guard and
  dimension mismatch protection.
- `diff_chunks` SQLite table + `idx_diff_chunks_pr_ref` index added to schema.
- `TRELIX_INDEXER_STREAMING=true` — generator-based file processing pipeline.
  `_iter_files()` yields files lazily; `_index_streaming()` uses bounded
  `Queue(maxsize=64)` with `try/finally` producer sentinel guarantee.
  Default off — zero behavior change on default path.

### Added — Phase 3: VS Code Extension, GitHub App PR Review
- `workspace-vscode/` — VS Code extension scaffold (`trelix.search` and `trelix.ask`
  commands) using `TrelixMcpClient` over MCP stdio transport. Piggybacks on existing
  `trelix-mcp` package — no new Python backend.
- `.github/workflows/trelix-review.yml` — GitHub Actions workflow that runs
  `trelix review --pr N --json` on every pull request and posts findings as
  GitHub Check annotations with file+line references.
  Permissions: `checks: write`, `pull-requests: write`, `contents: read`.
  Index step has `continue-on-error: true` for CI environments without local models.
- `infra/github-app/README.md` — GitHub App integration setup guide.

## [2.6.0] — 2026-07-08

### Added — XTR Late-Interaction Reranker (Plan C, EXPERIMENTAL)
- `TRELIX_RETRIEVAL_RERANK_PROVIDER=xtr` — XTR reranker (NeurIPS 2023,
  arXiv:2304.01982). Scoring stage is 100–1000x cheaper than ColBERT/PLAID
  by reusing already-retrieved tokens instead of loading all document tokens.
- `TRELIX_RETRIEVAL_XTR_TOKENS=100` — candidate token count for XTR retrieval
  (range 10–1000).
- `trelix.retrieval.reranker_xtr` — pure-Python XTR scoring module
  (`xtr_score_documents`, `warn_experimental`).
- **EXPERIMENTAL:** XTR has not been benchmarked on code-specific retrieval
  (CoIR/CoREB evaluation pending). Emits `UserWarning` on first use. PLAID
  remains the production-validated late-interaction option.

### Added — GroUSE-Style Synthesis Quality Harness (Plan D)
- `trelix.eval.synthesis` — `SynthesisEvalHarness`, `evaluate_synthesis`,
  `score_hallucination`, `score_completeness`, `score_faithfulness`, `SynthesisResult`.
- `trelix eval-synthesis --golden <path>` — CLI command for synthesis quality evaluation.
- `eval/golden_synthesis_sample.jsonl` — sample golden file for getting started.
- Golden file format extends the existing eval harness with optional
  `expected_answer_fragments` and `expected_symbols` fields.
- Research basis: GroUSE (arXiv:2409.06595, COLING 2025) — 7 failure modes,
  144 unit tests. GPT-4 correlation is insufficient as a quality proxy.

### Added — Short-Query Lexical Fallback (Plan B)
- `TRELIX_RETRIEVAL_SHORT_QUERY_LEXICAL=true` — enables BM25+grep-only routing
  for queries with ≤ threshold meaningful tokens (default off).
- `TRELIX_RETRIEVAL_SHORT_QUERY_TOKENS` — sets the meaningful-token threshold
  (default 5, range 1–10).
- `is_short_query(query, threshold)` and `count_meaningful_tokens(query)` helpers
  in `trelix.retrieval.bm25`.
- `SubQuery.lexical_only: bool` — new field; when True, `_run_subquery_legs` skips
  vector ANN embedding entirely.
- Research basis: CoREB benchmark (arXiv:2605.04615) confirms all embedding models
  score 0.000–0.015 nDCG@10 on short keyword queries vs 0.45–0.58 on long queries.

### Added — Incremental Louvain Community Detection (Plan A)
- `detect_communities_incremental(cg, seed_nodes, prev_partition)` — DF Louvain
  frontier heuristic (arXiv:2404.19634). Reprocesses only the affected-vertex
  frontier instead of the full graph on file-change events.
- `compute_affected_frontier(G, seed_nodes, partition)` — computes the DF Louvain
  frontier: seed nodes + their neighbors + their community members.
- `GraphUpdater` now maintains `_prev_partition` across calls and uses incremental
  detection for subsequent updates. First run and large-frontier (>50% of nodes)
  fall back to full Louvain.
- `Database.get_symbol_ids_for_file(rel_path)` — returns symbol IDs for a file
  (used to seed the incremental frontier from a file-change event).

## [2.5.0] — 2026-07-06

### Overview
Phase A–C of the v2.5.0 backlog. Three independent subsystems shipped:
multi-query expansion wired into `_retrieve_standard`, DimensionGuard at
`FileWatcher.__init__`, and MCP resource subscriptions (capability declaration
+ subscription registry + file-change notification bridge). v3.0.0 deprecation
schedule documented and regression-tested.

### Added — Multi-Query Expansion Wiring (Phase A)
- `MultiQueryExpander` is now wired into `_retrieve_standard` via `ThreadPoolExecutor`
- Enable with `TRELIX_RETRIEVAL_MULTI_QUERY=true`, tune with `TRELIX_RETRIEVAL_MULTI_QUERY_COUNT=3`
- Variant queries run in parallel; results RRF-merged with k=60 before dedup
- `ExpandResult.llm_used` indicates whether LLM expansion ran or fell back to original

### Added — DimensionGuard at Watch Startup (Phase A)
- `FileWatcher.__init__` now calls `DimensionGuard.check()` at startup
- Raises `DimensionMismatchError` immediately if provider was changed since last index run
- Prevents silent embedding corruption from mismatched providers during watch

### Added — MCP Resource Subscriptions (Phase B)
- `trelix-mcp` now advertises `resources.subscribe=True` in server capabilities
- `SubscriptionRegistry` tracks subscription IDs per resource URI (thread-safe)
- `notify_file_changed()` fires `notifications/resources/updated` (URI-only, per MCP spec)
  over stdio for all active subscribers when watchfiles detects a change
- Wire protocol: `resources/subscribe` -> `notifications/resources/updated` -> `resources/read`

### Documentation
- `docs/BACKWARDS_COMPATIBILITY.md` — v3.0.0 breaking changes table with file:line refs
- Deprecation warning for `TRELIX_RETRIEVAL_FLARE_MAX_ITER` regression-tested

### Breaking Changes
None — all changes are additive or fail-fast safety improvements.

## [2.4.0] — 2026-07-04

### Overview
Six backlog items shipped across Plans A–F. 1,467 unit tests passing, all features default-ON or backward-compatible.

### ⚠️ BREAKING CHANGE — `search_code` MCP tool response envelope

**Before (v2.3.0):** `search_code` returned `list[dict]` directly.

**After (v2.4.0):** `search_code` returns a pagination envelope:
```json
{"results": [...], "next_cursor": 10, "total_available": 25}
```

**Migration:** Update any MCP client code that iterates `search_code(...)` directly:
```python
# Before
for result in search_code(query="auth", repo_path="/repo"):
    ...

# After
response = search_code(query="auth", repo_path="/repo")
for result in response["results"]:
    ...
# Paginate: pass response["next_cursor"] as cursor= for the next page
```

### Added — Config field rename: `flare_max_retries` (Plan A)
- **`flare_max_retries`** replaces `flare_max_iterations` in `RetrievalConfig`
- Both `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` (new) and `TRELIX_RETRIEVAL_FLARE_MAX_ITER` (old) accepted via `AliasChoices`
- Using the old env var emits `DeprecationWarning`; old name removed in v3.0.0
- **⚠️ Range constraint:** field enforces `ge=1, le=3`. If you previously set `TRELIX_RETRIEVAL_FLARE_MAX_ITER` to a value >3, lower it before upgrading or pydantic raises `ValidationError` at startup.

### Added — Multi-Query Expansion Observability (Plan B)
- **`ExpandResult`** dataclass — `(queries, llm_used, elapsed_ms)` returned by `MultiQueryExpander.expand()`
- Three new nullable columns in `query_telemetry`: `expansion_used`, `expansion_variants`, `expansion_elapsed_ms`
- `TelemetryWriter.record()` accepts optional `expansion_result=` to persist expansion metadata
- Migration: idempotent `ALTER TABLE ADD COLUMN` — existing DBs upgraded automatically

### Added — FederatedRetriever TTL Cache (Plan C)
- **`FederatedRetriever(registry, cache_ttl=120.0)`** — SHA-256-keyed in-memory cache, thread-safe via `threading.Lock`
- `cache_ttl=0` disables caching; `cache_stats()` returns hit/miss/size; `clear_cache()` for forced eviction
- Expected ~90% hit rate for typical debugging-session query patterns

### Added — GitHub PR API integration (Plan D)
- **`GitHubPRClient`** — fetch PR file diffs and post review comments via GitHub REST API
- **`trelix review --pr owner/repo#N`** — fetches PR diff from GitHub and runs `DiffReviewer`
- **`trelix review --pr owner/repo#N --post-comments`** — posts findings back as a single batched GitHub review
- Token from `GITHUB_TOKEN` env var only; handles all 7 file status values; 3,000-file truncation warning

### Added — Multi-repo file watching (Plan E)
- **`MultiRepoWatcher`** — single `watchfiles.awatch(*all_paths)` call watching all registered repos simultaneously
- Hash guard prevents re-index cascade loops; deleted files are removed from the SQLite index + vector store
- **`trelix watch-all`** — new CLI command; shows per-repo stats on exit; graceful Ctrl+C shutdown

### Added — MCP pagination + progress notifications (Plan F)
- **`search_code` pagination** — `cursor=` (offset) + `next_cursor` in response; MCP-spec-approved pattern for large payloads
- **`index_codebase` progress** — `ctx.report_progress()` sends `notifications/progress` during indexing stages (best-effort)

## [2.3.0] — 2026-07-02

### Overview
Five research-grounded intelligence and infrastructure upgrades. All features default **OFF** — zero regression when disabled. 42/42 e2e checks pass, 1458 unit tests, zero blockers.

### Added — Embedding Dimension Guard (Plan E)
- **`DimensionGuard`** — detects provider/dimension mismatch at `Retriever.__init__` startup; raises `DimensionMismatchError` with exact `trelix migrate-vectors --reset` recovery instruction
- **`index_metadata` SQLite table** — records embedding dimension after each successful index run
- **`trelix migrate-vectors --reset`** — clears `chunk_embeddings` + dimension metadata for fresh re-index after provider switch
- Prevents silent wrong-results bug when switching e.g. Azure (3072-dim) → local (384-dim)

### Added — Multi-Query Retrieval Wiring (Plan A)
- **`MultiQueryExpander` wired** into `_retrieve_standard` — the class already existed; this commit connects it to the live retrieval pipeline
- When `TRELIX_RETRIEVAL_MULTI_QUERY=true`, primary query expands to N variants, each runs all retrieval legs in parallel via `ThreadPoolExecutor`, results merge into `leg_results_list` before RRF fusion
- `variants[1:]` used (not `variants[:]`) — original query never runs twice
- Falls back gracefully (non-fatal `logger.warning`) when LLM unavailable

### Added — MCP Resources + Prompts (Plan B)
- **MCP Resources** (application-controlled URI-addressable data):
  - `trelix://index/stats` — aggregate index statistics
  - `trelix://repo/{repo_path}/manifest` — indexed file list
  - `trelix://repo/{repo_path}/symbols/{qualified_name}` — symbol source code
- **MCP Prompts** (reusable LLM interaction templates):
  - `trelix-search` — structured code search prompt
  - `trelix-explain` — symbol explanation prompt
  - `trelix-blast-radius` — impact analysis prompt
- All resource handlers return JSON even on error; stdout stays clean for MCP stdio protocol
- Research basis: MCP spec (5× 3-0 adversarial votes on Resources/Templates/Prompts primitives)

### Added — Semantic PR/Diff Review (Plan C)
- **`DiffParser`** — parses unified git diff into `DiffHunk` objects; `from_git(repo, base, head)` uses `subprocess.run` with `shell=False` (no injection risk); `to_search_query()` extracts identifiers for hybrid retrieval
- **`DiffReviewer(config).review(hunks)`** — retrieval-augmented review: each hunk → search query → retrieve context → LLM generates `ReviewComment` objects; crash-safe, never raises
- **`trelix review <repo> [--diff <file>] [--base] [--head] [--json]`** CLI command with Rich table output

### Added — Multi-Repo Federated Search (Plan D)
- **`RepoRegistry`** — load/save/manage `~/.config/trelix/repos.json`; `add(alias, path, weight)`, `remove`, `list`; raises `ValueError` on duplicate alias
- **`FederatedRetriever(registry, max_workers=4).retrieve(query, k)`** — parallel fan-out across registered repos via `ThreadPoolExecutor`; RRF merge; deduplicates by `(file_path, symbol_id)`; crash-safe (returns `[]` when all repos fail)
- **`trelix search-all <query>`** — federated search CLI
- **`trelix federation add/list`** — registry management CLI
- Config: `federation_enabled=False` (`TRELIX_FEDERATION_ENABLED`), `federation_max_workers=4`

### Breaking Changes
None — all new features are opt-in via config flags.

### v2.4.0 Backlog
- Multi-query expansion observability (log which mode: LLM-assisted vs fallback)
- MCP subscription/streaming (server-push on index changes)
- FederatedRetriever caching layer for repeated queries
- `trelix review` integration with GitHub PR API
- Real-time multi-repo watch (`trelix watch-all`)

---

## [2.2.0] — 2026-07-01

### Overview
Intelligence upgrades: an agentic ReAct retrieval loop, static analysis (data-flow
and taint), and two new hybrid-search legs (sparse SPLADE-Code, multi-granularity
chunking). All opt-in via config flags that default to `False` — zero regression
when disabled.

### Added
- **Agentic ReAct loop** (`agentic_enabled`, `TRELIX_RETRIEVAL_AGENTIC=true`) —
  multi-turn retrieve → observe → re-retrieve loop with self-correction, replacing
  the single-shot Retriever → Synthesizer chain when enabled.
- **Data-flow analysis** (`dataflow_enabled`, `TRELIX_PARSER_DATAFLOW=true`) —
  per-function def-use chains extracted via a tree-sitter AST walk, stored in the
  `def_use_edges` table.
- **Taint analysis** (`taint_enabled`; `pip install trelix[taint]` then
  `trelix taint .`) — Semgrep-backed source→sink flow detection, findings stored
  in `taint_flows`.
- **Multi-granularity indexing** (`multi_granularity_enabled`,
  `TRELIX_CHUNKER_MULTI_GRANULARITY=true`) — block- and statement-level
  sub-chunks indexed as a 6th RRF leg alongside symbol-level chunks.
- **Sparse+dense hybrid retrieval** (`sparse_enabled`, `TRELIX_RETRIEVAL_SPARSE=true`)
  — SPLADE-Code sparse embeddings as a 7th RRF leg alongside BM25, with a
  memoized, thread-safe `SparseEmbedder`.

## [2.1.0] — 2026-06-30

### Overview
Two major feature sets landing together. Phase A ships the Knowledge Graph layer (v2.0.0 development).
Phase B is the Beast-Mode Upgrade: seven research-grounded retrieval improvements, all opt-in via
config flags that default to `False` — zero regression when disabled.

### Added — Knowledge Graph (Phase A)
- **Knowledge Graph**: new `trelix/graph/` module unifying call/import/type edges into a traversable `CodeGraph` (NetworkX MultiDiGraph)
- **Community Detection**: Louvain algorithm clusters codebase into architectural modules; `trelix graph ./repo` CLI command shows top communities
- **Semantic Concepts**: `ConceptExtractor` — LLM-powered extraction of architectural concepts from symbol batches (crash-safe, returns `[]` on any failure)
- **Graph Visualization**: `GraphVisualizer.export_html()` — Pyvis interactive HTML with community coloring and edge-type coloring; `pip install trelix[knowledge-graph]`
- **4th Retrieval Leg**: `graph_search_enabled=True` in `RetrievalConfig` enables CodeGraph BFS as a 4th search leg after RRF fusion
- **REST API**: `GET /graph`, `GET /graph/communities`, `GET /graph/visualize`, `GET /graph/search` endpoints
- **MCP Tools**: `build_knowledge_graph` and `graph_search_mcp` tools in `trelix-mcp`
- **Graph Persistence**: `graph_metadata` SQLite table stores community and degree centrality per symbol
- **PageRank symbol boosting** (`pagerank_boost_enabled`) — scores symbols by import-graph centrality; boosts high-centrality symbols post-rerank
- **Incremental graph updater** — `GraphUpdater.update_file()` refreshes community + PageRank for a changed file; wired into `trelix watch`

### Added — Beast-Mode Retrieval (Phase B)
- **File-summary 5th retrieval leg** (`file_summary_leg_enabled`) — RAPTOR-style file-level embeddings used as a 5th RRF leg (arXiv:2401.18059); requires `TRELIX_FILE_SUMMARIES_ENABLED=true` at index time
- **HyDE fallback** (`hyde_fallback_enabled`) — Hypothetical Document Embeddings (arXiv:2212.10496): generates a synthetic code snippet, embeds it instead of the raw NL query
- **Multi-query expansion** (`multi_query_enabled`) — decomposes a query into N variants, retrieves independently, RRF-fuses for broader recall
- **FLARE re-retrieval loop** (`flare_enabled`) — confidence-gated iterative retrieval (arXiv:2305.06983): re-retrieves when synthesis output contains uncertainty phrases
- **Query telemetry** (`telemetry_enabled`) — `TelemetryWriter` writes per-query rows (latency, intent, result count) to `query_telemetry` SQLite table; `trelix telemetry` CLI shows recent queries
- **CoIR evaluation harness** — `trelix eval --golden <file>` reports nDCG@10, Recall@10, MRR (CoIR format, ACL 2025 arXiv:2407.02883); pure-Python `trelix.eval.ndcg` with no pandas dependency

### Breaking Changes
- **CLI**: `trelix graph` renamed to `trelix call-graph` (the old call-graph/callers display).
  The name `trelix graph` now refers to the knowledge graph build command.
  Update any scripts using `trelix graph <repo> <symbol>` to `trelix call-graph <repo> <symbol>`.

---

## [2.0.0] — 2026-06-28

### Overview
Major feature release spanning three research-grounded upgrade phases. Phase 1 delivers CoIR SOTA embedding models (BGE-Code-v1 at 81.77, Nomic CodeRankEmbed) and Voyage Matryoshka compact dimensions. Phase 2 adds RAPTOR-style multi-granularity file summaries, the PLAID ColBERT late-interaction reranker (7–45× faster than exact ColBERT), and live streaming synthesis for `trelix ask`. Phase 3 ships a LanceDB vector backend (3–5× faster insert at 100k+ chunks) and a production-ready REST API (`trelix serve`) with SSE streaming and full CRUD index management. An LLM-as-judge evaluator rounds out the quality measurement story.

### Added
- **BGE-Code-v1 embedder** (`bge-code` provider) — BAAI CoIR SOTA 2025, self-reported 81.77 avg. `pip install trelix[bge-code]`
- **Nomic CodeRankEmbed embedder** (`nomic-code` provider) — task-prefix asymmetric encoding, no new deps. `pip install trelix[local]`
- **Voyage Matryoshka support** — `TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=512` passes `output_dimension` to voyage-code-3 API for compact embeddings
- **LLM-as-judge eval scorer** — `LLMJudge.score()` rates semantic retrieval quality 0.0–1.0; `EvalReport.mean_judge_score` aggregate
- **Multi-granularity file summaries** — `TRELIX_FILE_SUMMARIES_ENABLED=true` generates LLM file-level summaries alongside symbol chunks (RAPTOR-inspired, arXiv 2401.18059). Enables "explain this codebase" queries.
- **PLAID late-interaction reranker** — `rerank_provider=plaid` via RAGatouille. 7–45× faster than exact ColBERT with equivalent quality. `pip install trelix[plaid]`
- **Streaming synthesis** — `trelix ask` streams tokens live to the terminal; `GET /ask` SSE endpoint for REST clients
- **LanceDB vector backend** — `TRELIX_STORE_BACKEND=lance` enables ARM-native HNSW with 3–5× faster vector insert at 100k+ chunks. `pip install trelix[lance]`
- **REST API** — `trelix serve ./repo --port 8765` exposes `/search`, `/ask` (SSE), `/index`, `/health` endpoints via FastAPI. `pip install trelix[serve]`

### Fixed
- **pathspec DeprecationWarning** — upgraded `PathSpec.from_patterns()` call site to current API; eliminates deprecation warnings in all indexing paths

---

## [1.1.0] — 2026-06-28

### Overview
Search quality and performance release — all four phases from the v1.0.0 stress test audit.

### Added
- **Phase 1b: QueryPlan LRU cache** — `CachingPlanner` caches the gpt-4o query planner call (~2–4s). Combined with Phase 1 embedding cache, warm P50 drops from ~4,500ms to **23ms** (170× speedup). `TRELIX_RETRIEVAL_PLAN_CACHE_SIZE=128` (default).
- **Phase 3: Public graph API** — `Retriever.get_callers(symbol)`, `get_callees(symbol)`, `get_importers(path)` expose the call/import graph. New `trelix graph <repo> <symbol>` CLI subcommand.

### Fixed
- **Phase 2: File-type weighting** — README/YAML no longer outranks source code in search results. Per-language RRF score multipliers: source `1.0×`, markdown `0.3×`, yaml/json `0.5×`, html/css `0.4×`. Fixes 4/6 recall misses from v1.0.0 stress test.
- **Phase 4: tree-sitter API upgrade** — All 20 parser extractors migrated from deprecated `Language(path, name)` to `get_language()`. Eliminates 439 FutureWarnings per test run.

### Test coverage
- 1197 unit tests (was 1148), 8 warnings (was 439)

---

## [1.0.0] — 2026-06-27

### Overview
First stable release of trelix. Public Python API stabilised, all hard blockers
resolved, coverage gate at 75%, full v1 stability guarantees in effect.

### Added
- Public Python API: `from trelix import IndexConfig, Indexer, Retriever, TrelixChatClient`
- `trelix --version` / `trelix -V` flag
- SECURITY.md with responsible disclosure policy
- Versioning & Stability Policy in CONTRIBUTING.md
- Troubleshooting section in README
- trelix-langchain README.md (PyPI listing)
- Unit tests for retriever, reranker, indexer, planner, CLI, and 6 parser extractors

### Fixed
- `trelix ask` with Anthropic/Bedrock/Vertex no longer silently falls back to OpenAI
- `grep_search._body_search` bounded — eliminates OOM on large repos
- Incremental watch: debounced cross-file resolution passes
- Raw pydantic ValidationError replaced with clean user-facing messages
- Ctrl+C during indexing shows "Indexing cancelled." cleanly
- Empty search results show "No results found." instead of blank table
- `bedrock-titan` and `bedrock-cohere` now selectable via `--provider` flag
- requires-python tightened to <3.13 (honest — cp313 tree-sitter-languages unavailable)

### Changed
- Development Status: 4 - Beta → 5 - Production/Stable
- Coverage gate: fail_under = 75
- `dist/` added to .gitignore

---

## [0.7.1] — 2026-06-27

### Fixed
- **`BedrockCohereEmbedder` chunk truncation** — Bedrock validates text length before
  applying `truncate="END"`, so texts >2048 characters raised `ValidationException` at
  the API level. Now pre-truncates client-side to 2048 chars before each `invoke_model`
  call. Found during live end-to-end indexing with default `max_tokens_per_chunk=512`
  (code chunks with docstrings routinely exceed 2048 characters).

### Added
- **Bedrock full-pipeline e2e tests** — `tests/integration/test_llm_e2e.py` now includes
  two tests that index a synthetic Python repo end-to-end (walk → parse → chunk → embed
  via Bedrock → store → search) for both `bedrock-cohere` and `bedrock-titan` providers.
- **`trelix-llama-index` README** — PyPI listing now shows description and usage examples.

---

## [0.7.0] — 2026-06-27

### Overview
Universal LLM client factory — all 5 chat call sites migrated to a provider-agnostic
`TrelixChatClient` ABC. Adding any new provider requires zero changes to business logic.

### Added
- **`src/trelix/llm/` package** — `TrelixChatClient` ABC, `ChatMessage`, `ChatResponse`,
  `ToolCallResponse` dataclasses, `build_chat_client()` factory
- **`LLMConfig`** — new config class for chat providers (separate from `EmbedderConfig`).
  Added as `IndexConfig.llm` field.
- **`OpenAIBackend`** — OpenAI + Azure. Auto-detects `max_completion_tokens` vs `max_tokens`
  based on model family (gpt-4o→max_completion_tokens; gpt-4/gpt-3.5→max_tokens)
- **`AnthropicBackend`** — Anthropic Claude direct. `max_tokens=`, `system=` separate param,
  `input_schema` tool format, `end_turn`→`stop` normalization. `pip install trelix[anthropic]`
- **`BedrockBackend`** — AWS Bedrock Converse API. `inferenceConfig.maxTokens` (nested camelCase),
  `system=[{"text":...}]` top-level, content always list-of-dicts, `{"auto":{}}` tool choice.
  `pip install trelix[bedrock]`
- **`VertexBackend`** — Google Vertex AI / Gemini via google-genai SDK. `max_output_tokens` in
  `GenerateContentConfig`, `system_instruction=` param. `pip install trelix[vertex]`
- **`LiteLLMBackend`** — universal delegate for 100+ providers. `drop_params=True` suppresses
  UnsupportedParamsError. Model strings: `"bedrock/claude-3-5-sonnet"`, `"gemini/gemini-2.0-flash"`.
  `pip install trelix[litellm]`
- New optional dep groups: `[anthropic]`, `[bedrock]`, `[vertex]`, `[litellm]`, `[llm-all]`

### Changed
- All 5 LLM call sites now use `TrelixChatClient` via factory — never import provider SDKs directly
- `ContextualChunker` accepts `TrelixChatClient` (new) or raw openai client (backward compat)

### Fixed
- `_token_limit_param()` in OpenAIBackend correctly routes legacy models to `max_tokens=`
  and modern models to `max_completion_tokens=` — eliminates the recurring parameter bug
- `BedrockBackend`: base64-encoded AWS credentials (stored in `.env`) decoded transparently
- `BedrockBackend`: bare model IDs rejected by Bedrock — now uses `us.*` inference profile IDs
- Unit test isolation: `test_llm_field_on_index_config` no longer leaks `.env` provider state

### Added (post-task additions)
- **`BedrockTitanEmbedder`** — `amazon.titan-embed-text-v2:0`, configurable 256/512/1024 dims,
  normalize=True. Set `TRELIX_EMBEDDER_PROVIDER=bedrock-titan`. `pip install trelix[bedrock]`
- **`BedrockCohereEmbedder`** — `cohere.embed-english-v3`, 1024 dims, asymmetric doc/query
  retrieval (`search_document` vs `search_query` input_type). `pip install trelix[bedrock]`
- **Bedrock model fallback** — `BedrockBackend` defaults to `us.anthropic.claude-sonnet-4-6`
  (primary) with transparent auto-fallback to `us.anthropic.claude-haiku-4-5-20251001-v1:0`
  on `ValidationException`. Override via `TRELIX_LLM_BEDROCK_PRIMARY_MODEL` /
  `TRELIX_LLM_BEDROCK_FALLBACK_MODEL`.
- **Live e2e tests** — `tests/integration/test_llm_e2e.py`: 16 tests covering Azure + Bedrock
  chat (complete/stream/tool_call) + Bedrock embeddings. Skip gracefully when creds absent.

---

## [0.6.0] — 2026-06-27

### Overview
Contextual chunking is now production-ready — the feature works end-to-end with verified context summaries stored in the database and indexed in BM25. Two bugs fixed that prevented contextual summaries from actually persisting.

### Fixed
- **Contextual chunking context_summary persistence:** `ContextualChunker.build_chunks()` sets `symbol.context_summary` but the DB insert in `Indexer._insert_one()` happened before chunking ran. Fixed by adding an `UPDATE symbols SET context_summary = ?` pass after `build_chunks()` for any symbols that received summaries. All 66 test symbols now have `context_summary IS NOT NULL`.
- **Contextual chunking LLM call:** `ContextualChunker._generate_summary()` used `max_tokens=` — unsupported by gpt-4o / newer Azure. Changed to `max_completion_tokens=` (consistent with synthesizer.py fix in v0.3.0).
- **Test updated:** `test_llm_called_with_correct_arguments` asserts `max_completion_tokens` instead of `max_tokens`.

### Verified
- 66/66 symbols receive LLM context summaries stored in `symbols.context_summary`
- Summaries indexed in `symbols_fts` — BM25 searches now include them
- Recall@5: 10/10 = 100% on mini_repo (baseline maintained)

### How to Enable Contextual Chunking

```bash
TRELIX_CHUNKER_CONTEXTUAL=true
TRELIX_CHUNKER_CONTEXTUAL_MODEL=gpt-4o-mini
TRELIX_EMBEDDER_PROVIDER=openai   # or azure
trelix index ./your-repo
```

---

## [0.5.1] — 2026-06-27

### Fixed
- `trelix-mcp` README: add `<!-- mcp-name: io.github.sairam0424/trelix -->` ownership verification tag required by the official MCP registry
- `trelix-mcp` server.json: shorten description to ≤100 chars to pass registry validation

---

## [0.5.0] — 2026-06-27

### Overview
Ecosystem discoverability release — trelix is now reachable across every major surface in the AI developer ecosystem. Three new PyPI packages, MCP registry listing, GitHub Action marketplace, Homebrew tap, and awesome list submissions.

### Added

#### New PyPI Packages
- **`trelix-mcp`** (`pip install trelix-mcp`) — MCP server exposing 4 tools via stdio transport. Works with Claude Code, Cursor, Windsurf, and Continue.dev. One-command setup: `claude mcp add trelix -- trelix-mcp`.
  - `search_code(query, repo_path, k=10)` — hybrid semantic + BM25 code search
  - `index_codebase(repo_path, provider="local")` — index a repository (run once)
  - `get_symbol(qualified_name, repo_path)` — get full source of any symbol
  - `blast_radius(symbol_name, repo_path)` — find everything that depends on a symbol
- **`trelix-langchain`** (`pip install trelix-langchain`) — `TrelixRetriever(BaseRetriever)` for LangChain RAG pipelines. Returns `list[Document]` with full metadata (file, symbol, language, score, lines).
- **`trelix-llama-index`** (`pip install trelix-llama-index`) — `TrelixIndexRetriever(BaseRetriever)` for LlamaIndex. Returns `list[NodeWithScore]` with file + symbol metadata.

#### Registry & Discovery
- **Official MCP Registry** — submitted via `mcp-publisher` CLI. Server ID: `io.github.sairam0424/trelix`. Pip ownership verified via `mcp-name` tag in README.
- **Glama.ai** — `glama.json` added to repo root for automatic Glama MCP directory indexing.
- **GitHub Actions Marketplace** — `trelix-index-action@v1` at `github.com/sairam0424/trelix-index-action`. Auto-indexes any repo on push with cached `.trelix/index.db`.
- **Homebrew tap** — `brew tap sairam0424/trelix && brew install trelix` via `github.com/sairam0424/homebrew-trelix`.
- **Awesome list submissions** — PRs submitted to awesome-mcp-servers (#8787), awesome-llm-apps (#903), awesome-langchain (#426).

#### PyPI Metadata
- 5 new Topic classifiers: `Scientific/Engineering :: Artificial Intelligence`, `Software Development :: Libraries :: Application Frameworks`, `Text Processing :: Indexing`, `Internet :: WWW/HTTP :: Indexing/Search`
- 21 keywords including `mcp`, `model-context-protocol`, `langchain`, `llama-index`, `code-assistant`, `static-analysis`
- 3 new README badges: MCP Compatible, LangChain retriever, Downloads

#### CI/CD
- `release.yml` now publishes all 4 packages (`trelix`, `trelix-mcp`, `trelix-langchain`, `trelix-llama-index`) to PyPI on `v*` tag
- PyPI OIDC trusted publisher configured for all 4 packages (no stored secrets for future releases)

#### Documentation
- `docs/discoverability/ECOSYSTEM-ROADMAP.md` — full ecosystem strategy with registry URLs, submission templates, priority stack
- `docs/discoverability/AWESOME-LIST-SUBMISSIONS.md` — ready-to-submit PR bodies for 3 awesome lists
- `packages/trelix-mcp/README.md` — install, Claude Code / Cursor / Windsurf / Continue.dev setup, tools table
- `packages/trelix-mcp/server.json` — official MCP registry schema for `mcp-publisher`

### Changed
- `pyproject.toml` version `0.4.0` → `0.5.0`; all sub-packages at `0.5.0` (trelix-mcp at `0.5.1`)
- `src/trelix/__init__.py` `__version__` updated to `0.5.0`
- README: added Integrations table (MCP, LangChain, LlamaIndex, GitHub Action, Homebrew), MCP Quick Setup block, LangChain code example, Homebrew install option, GitHub Action quick-start

### Fixed
- Package builds: `LICENSE` copied into each sub-package (hatchling resolves paths relative to package root, not repo root)
- `trelix-mcp/__init__.py`: added `__all__ = ["__version__"]` for parity with other packages
- `trelix-llama-index/retriever.py`: import ordering fix (ruff I001)
- Test files: removed unused `patch` imports from `trelix-langchain` and `trelix-llama-index` test suites

---

## [0.4.0] — 2026-06-26

### Overview
Beast-mode upgrade across three axes simultaneously: **retrieval quality** (+49% embedding quality, 67% failure-rate reduction), **scale** (HNSW index, Qdrant backend), and **speed** (4x async pipeline, real-time file watcher). Grounded in 6 adversarially-verified research findings from the CoIR benchmark, Anthropic contextual retrieval research, and VLDB/ACL 2025 proceedings.

### Added

#### Quality — Retrieval & Embeddings
- **Contextual Chunking (U1):** `ContextualChunker` prepends a 2-3 sentence LLM-generated summary to each chunk before embedding AND BM25 indexing. Reduces retrieval failure rate from 5.7% → 1.9% (67% reduction). Config-gated via `TRELIX_CHUNKER_CONTEXTUAL=false` — off by default.
- **Voyage Code Embedder (U2):** New `voyage` provider using `voyage-code-3` (1024-dim, 16k context). Scores 56.26 avg on CoIR benchmark vs Ada-002's 45.59 (+24%). `pip install trelix[voyage]`.
- **Local Code Embedder (U2):** New `local-code` provider using `Salesforce/SFR-Embedding-Code-2B_R` (4096-dim, 2B params). Scores 67.41 on CoIR — 49% quality gain over Ada-002. No API key required.

#### Scale — Vector Store
- **Filterable HNSW Index (U3):** O(log n) vector search via sqlite-vec HNSW. Falls back to flat scan on older versions.
- **Qdrant Optional Backend (U4):** `QdrantVectorStore` drop-in for >500k chunk deployments. `trelix migrate-vectors --to qdrant`. `pip install trelix[qdrant]`.

#### Speed — Indexing & Updates
- **Async Batch Embedding (U5):** Phase 3 runs up to 4 concurrent embed batches via `asyncio.gather`. ~3-4x speedup on large repos.
- **File Watcher (U6):** `trelix watch <repo>` — 500ms debounced auto-reindex on file save. `pip install trelix[watch]`.

#### Intelligence — Planning & Synthesis
- **Adaptive 3-Tier Query Router (U7):** Tier 1 (direct/skip retrieval) → Tier 2 (8-intent single-step) → Tier 3 (multi-step decomposition).
- **GraphRAG Map-Reduce Synthesis (U8):** For >20 results or >8k tokens, map-reduce synthesis handles arbitrarily large corpora.

#### Precision — Call Graph
- **Call Graph Precision (U9):** 3-priority callee resolution (qualified_name → type_hint+name → name-only). ~40% fewer false-positive cross-file edges.

#### Evaluation
- **Production Eval Harness (U10):** MRR, Recall@1/5/10, NDCG@10 on 50 trelix-self queries. `make eval-full`.

### Changed
- New optional dep groups: `[voyage]`, `[qdrant]`, `[watch]`
- `BaseVectorStore` ABC introduced; `VectorStore` → `SQLiteVectorStore`
- `QueryPlanner` → `AdaptiveRouter` (backward-compatible)

### Fixed
- `synthesizer.py`: `max_completion_tokens` for gpt-4o compatibility
- Test fixtures: removed synthetic passwords that triggered GitGuardian

---

## [0.3.0] — 2026-06-26

### Added
- Removed all internal origin watermarks (`aava`, `AavaPlatformEmbedder`, `CODEINDEX_*`, `codeindex` binary)
- PyInstaller binary renamed `codeindex` → `trelix`
- Fixed `synthesizer.py` `max_completion_tokens` for gpt-4o
- Restored correct `tree_sitter_languages.get_language()` in 4 parsers
- Updated `.gitignore` to exclude `.claude/`, `uv.lock`, `dist/`

---

## [0.2.0] — 2026-06-25

### Added
- Ruby parser — completes all 20 language extractors
- PyInstaller spec (`trelix.spec`) — `dist/trelix` single-file binary
- `scripts/build-binary.sh`, `make binary` / `make binary-clean` / `make binary-install`
- GitHub Actions `build-binaries.yml` — macOS arm64 + Windows x64 matrix
- Release workflow attaches binaries to GitHub Releases
- `docs/integrations/vscode-plugin.md`

---

## [0.1.0] — 2026-06-25

### Added
- Initial release — Tree-sitter AST indexing for 20+ languages
- Hybrid search: vector (ANN, sqlite-vec) + BM25 (FTS5) + grep via RRF
- RRF fusion + call-graph / import / type-edge expansion with PageRank
- 8-intent LLM query planner
- Cohere + cross-encoder reranker
- Intent-aware context assembler (greedy / breadth_first)
- LLM synthesis via OpenAI or Azure (`trelix ask`)
- CLI: `index`, `search`, `ask`, `query`, `stats`, `update-index`
- Providers: `local` (no API key), `openai`, `azure`
- Zero-infra store: single SQLite file with sqlite-vec + FTS5 BM25

[Unreleased]: https://github.com/sairam0424/trelix/compare/v3.0.1...HEAD
[3.0.1]: https://github.com/sairam0424/trelix/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/sairam0424/trelix/compare/v2.12.0...v3.0.0
[2.12.0]: https://github.com/sairam0424/trelix/compare/v2.11.1...v2.12.0
[2.11.1]: https://github.com/sairam0424/trelix/compare/v2.11.0...v2.11.1
[2.11.0]: https://github.com/sairam0424/trelix/compare/v2.10.0...v2.11.0
[2.10.0]: https://github.com/sairam0424/trelix/compare/v2.9.0...v2.10.0
[2.9.0]: https://github.com/sairam0424/trelix/compare/v2.8.1...v2.9.0
[2.8.1]: https://github.com/sairam0424/trelix/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/sairam0424/trelix/compare/v2.7.3...v2.8.0
[2.7.3]: https://github.com/sairam0424/trelix/compare/v2.7.2...v2.7.3
[2.7.2]: https://github.com/sairam0424/trelix/compare/v2.7.1...v2.7.2
[2.7.1]: https://github.com/sairam0424/trelix/compare/v2.7.0...v2.7.1
[2.7.0]: https://github.com/sairam0424/trelix/compare/v2.6.0...v2.7.0
[2.6.0]: https://github.com/sairam0424/trelix/compare/v2.5.0...v2.6.0
[2.5.0]: https://github.com/sairam0424/trelix/compare/v2.4.0...v2.5.0
[2.4.0]: https://github.com/sairam0424/trelix/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/sairam0424/trelix/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/sairam0424/trelix/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/sairam0424/trelix/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/sairam0424/trelix/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/sairam0424/trelix/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/sairam0424/trelix/compare/v0.7.1...v1.0.0
[0.7.1]: https://github.com/sairam0424/trelix/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/sairam0424/trelix/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/sairam0424/trelix/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/sairam0424/trelix/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/sairam0424/trelix/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/sairam0424/trelix/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/sairam0424/trelix/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/sairam0424/trelix/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sairam0424/trelix/releases/tag/v0.1.0
