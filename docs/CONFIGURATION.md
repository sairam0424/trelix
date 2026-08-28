# Trelix Configuration Reference — v3.2.3

Complete reference for all configuration options available in trelix.

---

## Configuration Methods

There are exactly two ways to configure trelix, plus the built-in defaults. Settings are
resolved in priority order (highest wins):

1. **Environment variables** — set in the shell or via CI/CD secrets
2. **`.env` file** — auto-loaded on startup
3. **Defaults** — built-in fallbacks documented in the tables below

Two things about the `.env` file are worth knowing before you rely on it:

- **It is resolved relative to the current working directory, not the indexed repo.**
  `trelix` loads `./.env` — the `.env` in whatever directory you launch the process from.
  A `.env` sitting inside the repo passed as the positional `REPO` argument (`repo_path`)
  is *not* read unless that directory also happens to be your cwd.
- **Not every setting group reads it.** The `TRELIX_WALKER_*`, `TRELIX_PARSER_*`,
  `TRELIX_CHUNKER_*`, and `TRELIX_SPARSE_*` groups are read from the process environment
  only; they ignore `.env` entirely.
- **Four dynamic, suffix-named settings also ignore `.env`.** These are read directly from
  the process environment (`os.environ`) rather than through the settings loader, because
  their names are not fixed fields — the suffix is part of the key:
  `TRELIX_RETRIEVAL_LEG_WEIGHT_<LEG>`, `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_<EXT>`,
  `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS` (the JSON form), and
  `TRELIX_RETRIEVAL_COMPRESSION_RATIO_<INTENT>`.
  Put them in your shell/systemd unit/container env, **not** in `.env` — a `.env` entry is
  silently ignored, which looks configured but is not.

There is **no per-project config file** — no TOML, YAML, or JSON config is read from
anywhere. See [Per-Project Configuration](#per-project-configuration).

---

## Environment Variables

> **Naming rule — read this before adding a row below.** These are `pydantic-settings`
> classes, and a field's env name comes from one of two mutually exclusive routes. A field
> with an explicit `Field(alias="X")` is settable as **`X` verbatim** — the class's
> `env_prefix` does **not** apply to it (e.g. `azure_embeddings_deployment` on
> `EmbedderConfig` is `AZURE_EMBEDDINGS_MODEL`, *not* `TRELIX_EMBEDDER_AZURE_...`). A field
> with **no** explicit alias is settable as `<env_prefix><FIELD_NAME_UPPER>` (e.g.
> `graph_search_enabled` on `RetrievalConfig` is `TRELIX_RETRIEVAL_GRAPH_SEARCH_ENABLED`).
> Concatenating the prefix onto an alias produces a name that is silently ignored. Always
> confirm a name against `src/trelix/core/config.py` — or by setting it and checking the
> field actually changed — before documenting it.

### Embedder

| Variable | Default | Description |
|---|---|---|
| `TRELIX_EMBEDDER_PROVIDER` | `local` | Embedding provider. One of: `local`, `openai`, `azure`, `voyage`, `local-code`, `bge-code`, `nomic-code`, `bedrock-titan`, `bedrock-cohere` (`bge-code` is **experimental** — pooling unverified) — see [PROVIDERS.md](PROVIDERS.md) |
| `TRELIX_EMBEDDER_OPENAI_MODEL` | `text-embedding-3-large` | OpenAI embedding model name |
| `AZURE_EMBEDDINGS_MODEL` | `text-embedding-3-large` | Azure deployment name for embeddings. **Not** `TRELIX_EMBEDDER_AZURE_DEPLOYMENT` — this field carries an explicit alias, so the `TRELIX_EMBEDDER_` prefix does not apply (see the naming rule above) |
| `TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS` | _(none)_ | Matryoshka output dimension for Voyage models. Accepted values: `256`, `512`, `1024`, `2048` |
| `OPENAI_API_KEY` | _(none)_ | API key — required when `TRELIX_EMBEDDER_PROVIDER=openai` |
| `AZURE_API_KEY` | _(none)_ | API key — required when `TRELIX_EMBEDDER_PROVIDER=azure` |
| `AZURE_ENDPOINT` | _(none)_ | Full Azure endpoint URL (e.g. `https://<name>.openai.azure.com/`) |
| `VOYAGE_API_KEY` | _(none)_ | API key — required when `TRELIX_EMBEDDER_PROVIDER=voyage` |
| `TRELIX_EMBEDDER_BATCH_SIZE` | `64` | Number of texts sent per embedding request. Raise for higher throughput against a generous API tier; lower it if you are hitting per-request size limits |
| `TRELIX_EMBEDDER_EMBED_MAX_TOKENS_PER_BATCH` | `100000` | Token ceiling per embedding batch. A batch is flushed early once this is reached, even if `TRELIX_EMBEDDER_BATCH_SIZE` is not yet full |
| `TRELIX_EMBEDDER_TPM_LIMIT` | `0` | Client-side tokens-per-minute cap for remote providers. `0` disables the limiter (correct for `local`, which has no rate limit) |

There is **no embedding concurrency setting.** Indexing throughput against a remote provider is
tuned via the three batching variables above, not by a worker/concurrency count.

### Retrieval

| Variable | Default | Description |
|---|---|---|
| `TRELIX_RETRIEVAL_MULTI_QUERY` | `false` | Enable multi-query expansion — generates additional query variants to improve recall |
| `TRELIX_RETRIEVAL_MULTI_QUERY_COUNT` | `2` | Number of query variants to generate when multi-query is enabled |
| `TRELIX_RETRIEVAL_SHORT_QUERY_LEXICAL` | `false` | Route short queries (≤threshold tokens) to BM25+grep only, skipping vector ANN |
| `TRELIX_RETRIEVAL_SHORT_QUERY_TOKENS` | `5` | Meaningful-token threshold for short-query classification (1–10) |
| `TRELIX_RETRIEVAL_PATH_FILTER_OVERSAMPLE` | `3` (min: `1`) | **Advanced/internal tuning knob.** When a `SubQuery.path_filter` is set, the vector leg over-fetches by this factor before post-filtering results by path prefix and truncating back to `k` — protects recall against the filter discarding raw ANN hits. There is currently no CLI, REST, or MCP parameter to set `path_filter` itself; it is only set programmatically. |
| `TRELIX_RETRIEVAL_CONTEXT_BUDGET_PER_SOURCE` | `false` | When enabled, the context-assembly token budget is split proportionally across each retrieval leg's result count instead of one shared pool, so a single noisy leg cannot crowd out the others. `false` reproduces the pre-existing single-pool greedy-pack behavior byte-for-byte. |
| `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET` | `12000` | Token budget for context assembly. An explicit integer is used verbatim and reproduces v2.12.0 behavior exactly. Set it to `null` — `none`, `auto`, `~`, and the empty string are also accepted, case-insensitively — to auto-derive the budget from the synthesis model's context window instead. See [Model-Aware Context Budget](#model-aware-context-budget). No range validation: `0` and negative integers are accepted and taken literally. |
| `TRELIX_RETRIEVAL_CONTEXT_WINDOW_FRACTION` | `0.5` | Fraction of the resolved model context window to spend on retrieved context when the budget is auto-derived (`budget = window × fraction`), range `0.1`–`0.9` — an out-of-range value fails validation at startup. **Only** consulted when `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET` is `null`; silently ignored when an explicit integer budget is set. |
| `TRELIX_RETRIEVAL_SCALE_TOP_K_TO_BUDGET` | `false` | Also scale the retrieval ceilings `top_k_vector` (default `20`) and `rerank_top_n` (default `15`) by `effective_budget / 12000` when the budget is auto-derived. Requires `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null` — a no-op with an explicit integer budget. Those ceilings, not the budget, are what actually limit how much context is assembled, so this is the flag that widens context; it also raises per-query reranker and synthesis cost. |
| `TRELIX_RETRIEVAL_COMPRESSION` | `false` | Master switch for SeleCom query-conditioned compression — shrinks retrieved symbol bodies so results that would not fit the budget are included in compressed form instead of dropped. `false` leaves the assembler without a compressor at all, reproducing the pre-existing assembled context byte-for-byte. See [Context Compression (SeleCom)](#context-compression-selecom). |
| `TRELIX_RETRIEVAL_COMPRESSION_PROVIDER` | `extractive` | Compression backend. `extractive` is the only accepted value — anything else fails validation at startup. It makes no embedding, API, or network call at query time. |
| `TRELIX_RETRIEVAL_COMPRESSION_RATIO` | `0.45` | **Fallback** target fraction of a body's tokens to keep, range `0.1`–`1.0`. Consulted only when the query's intent is unknown or absent — a recognised intent takes its ratio from the per-intent table in [Context Compression (SeleCom)](#context-compression-selecom) instead, so this value rarely applies. |
| `TRELIX_RETRIEVAL_COMPRESSION_RATIO_<INTENT>` | (per-intent defaults) | Override a single intent's compression ratio, e.g. `TRELIX_RETRIEVAL_COMPRESSION_RATIO_BLAST_RADIUS=0.5`. `<INTENT>` is the upper-case intent name: `SYMBOL_LOOKUP`, `FILE_OVERVIEW`, `FEATURE_FLOW`, `PROJECT_OVERVIEW`, `COMPARISON`, `CONFIG_LOOKUP`, `DEPENDENCY_MAP`, `BLAST_RADIUS`. A non-numeric value or one outside `0.1`–`1.0` is logged at WARNING and ignored in favour of the baked ratio — the planner never raises into the retrieval path. |
| `TRELIX_RETRIEVAL_COMPRESSION_MIN_TOKENS` | `120` (min: `0`) | Bodies below this token count are never compressed — the elision markers and per-span headers would cost more than the shrink saves. Such a result is handled exactly as before: kept if it fits, skipped if it does not. |
| `TRELIX_INDEXER_STREAMING` | `false` | Enable generator-based streaming indexing pipeline (bounded Queue, lazy file iteration). Default off — zero behavior change when unset. |
| `TRELIX_RETRIEVAL_RERANK_PROVIDER` | `cohere` | Reranker to apply after fusion. One of: `cross_encoder`, `cohere`, `plaid`, `xtr` (**experimental**). Reranking itself is gated separately by `TRELIX_RETRIEVAL_RERANK` (default `true`) |
| `TRELIX_RETRIEVAL_XTR_TOKENS` | `100` | **Inert.** `RetrievalConfig.xtr_candidate_tokens` is declared and range-validated but read nowhere in `src/`, deliberately: the `xtr` provider is degenerate (one synthetic query token, so every output score is bit-identical to its input) and a real candidate-token budget needs the ColBERT-style multi-vector token index trelix does not build. The knob is kept for that future embedder; the provider logs this on every call |
| `TRELIX_RETRIEVAL_FLARE` | `false` | Enable FLARE re-retrieval. Not the paper's token-log-probability method: after a synthesis completes, the answer is scanned for a fixed list of uncertainty phrases (`"i don't know"`, `"cannot find"`, …) and, on a hit, the query is enriched and re-synthesized. There is no probability threshold setting |
| `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` | `1` | Maximum FLARE iterations per query (min: 1, max: 3) |
| `TRELIX_RETRIEVAL_HYDE_FALLBACK` | `false` | Enable HyDE (Hypothetical Document Embeddings) fallback when standard retrieval returns weak results |
| `TRELIX_RETRIEVAL_FILE_SUMMARY_LEG` | `false` | Enable the file-summary retrieval leg — retrieves against LLM-generated file summaries in addition to raw chunks |
| `TRELIX_RETRIEVAL_PAGERANK_BOOST` | `false` | Enable PageRank-based symbol boosting — surfaces frequently referenced symbols higher in results |
| `TRELIX_RETRIEVAL_PAGERANK_PERSONALIZATION` | `false` | Enable Personalized PageRank: teleport mass is weighted toward symbols with a `generic_edges` connector-artifact/ticket relationship (uniform `1/\|T\|` over that seed set) instead of uniform teleportation across every node. Applies to both `rank_by_pagerank()` (query-time) and `compute_pagerank()` (index-time). Falls back to plain uniform-teleportation PageRank when disabled or when the seed set is empty — zero behavior change unless opted in. Interaction risk with `TRELIX_RETRIEVAL_PAGERANK_BOOST`: on a repo where only a few symbols have ever been referenced by a ticket/artifact, enabling both together can invert `get_top_central_symbols()`'s ranking — if boost results look off with both enabled, try disabling personalization first to isolate which flag is driving the change. |
| `TRELIX_RETRIEVAL_GRAPH_SEARCH_ENABLED` | `false` | Enable knowledge graph search leg — queries the code graph in addition to vector search. Note the `_ENABLED` suffix: this field has no explicit alias, so its name is the prefix plus the full field name (see the naming rule above). The shorter `TRELIX_RETRIEVAL_GRAPH_SEARCH` is **not** read |
| `TRELIX_RETRIEVAL_LEG_WEIGHT_<LEG>` | `1.0` per leg | Per-leg RRF score multiplier applied during fusion, before summing (not after, like file-type weights). `<LEG>` is one of `VECTOR`, `BM25`, `GREP`, `SUMMARY`, `SUB_CHUNK`, `SPARSE`, e.g. `TRELIX_RETRIEVAL_LEG_WEIGHT_BM25=0.7`. All-`1.0` (the default) is a no-op — byte-for-byte identical to unweighted fusion. |
| `TRELIX_RETRIEVAL_DECLARATION_BOOST` | `false` | Enable FTS5 declaration-boost ranking — boosts BM25 matches on a symbol's `name`/`qualified_name` over incidental matches in `docstring`/`body`/`context_summary`. Fixes cases where a symbol only *mentioning* a term outranks the symbol actually *named* that term. |
| `TRELIX_RETRIEVAL_DECLARATION_BOOST_WEIGHT` | `1.0` | Declaration-boost multiplier applied to the `name`/`qualified_name` FTS5 columns (range: 1.0–10.0). Default `1.0` is a no-op — byte-for-byte identical to unweighted BM25 ranking. Only takes effect when `TRELIX_RETRIEVAL_DECLARATION_BOOST=true`. |
| `TRELIX_RETRIEVAL_BREADTH_FLOOR` | `true` | When a direct-lookup intent (`file_overview`, `project_overview`, `config_lookup`) resolves only a thin result, also run standard retrieval and merge instead of returning it as if complete. Set `false` for the pre-3.1.2 all-or-nothing behaviour, where one matched file suppressed the vector, BM25 and grep legs entirely. |
| `TRELIX_RETRIEVAL_BREADTH_FLOOR_MIN_FILES` | `2` | The floor fires when the direct lookup resolves fewer than this many distinct files **and** fewer than `..._MIN_SYMBOLS` symbols — both conditions, so a single file rich in symbols is still treated as a real answer. |
| `TRELIX_RETRIEVAL_BREADTH_FLOOR_MIN_SYMBOLS` | `10` | See above. Chosen against one repository's golden set: the floor restored nDCG@10 to 0.6189/0.6217 from 0.6039/0.5791, at the cost of top-rank precision on exact-filename queries (Recall@10 stays 1.0000, nDCG 1.0000 -> 0.8253). |
| `TRELIX_TELEMETRY_ENABLED` | `false` | Record every `retrieve()` call to the `query_telemetry` table in the index DB. Zero overhead when disabled. This setting lives on the top-level index config, not on the retrieval config — so it is `TRELIX_TELEMETRY_ENABLED`, **not** `TRELIX_RETRIEVAL_TELEMETRY`, which is not read. For OpenTelemetry spans see `TRELIX_OTEL_ENABLED` under [Observability](#observability-opentelemetry) — a separate, independent switch |
| `TRELIX_FILE_SUMMARIES_ENABLED` | `false` | Generate LLM-powered file summaries at index time (requires a configured LLM provider) |

### Model-Aware Context Budget

By default `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET` is the fixed integer `12000`, which is exactly what v2.12.0 did. Setting it to `null` switches on auto-derivation: at `Retriever` construction time trelix resolves the context window of the model named by `TRELIX_LLM_MODEL` and computes `budget = int(window × TRELIX_RETRIEVAL_CONTEXT_WINDOW_FRACTION)`. The result is memoized for the session and logged at INFO.

| `TRELIX_LLM_MODEL` | Resolved window | Budget at fraction `0.5` |
|---|---|---|
| `gpt-4o` (and `gpt-4o-2024-11-20`) | 128,000 | 64,000 |
| `claude-sonnet-4-6` | 200,000 | 100,000 |
| `gemini-2.5-pro` | 1,000,000 | 500,000 |
| unrecognised (e.g. `my-finetune`) | — | 12,000 (fallback) |

Window lookup is **prefix** matching against a longest-prefix-first table, lower-cased on both sides, so version and date suffixes resolve correctly (`gpt-4o-2024-11-20` → `gpt-4o`). Matching is anchored at the **start** of the string: a provider-namespaced model id such as `us.anthropic.claude-sonnet-4-6`, `anthropic.claude-3-5-sonnet-...`, or `bedrock/claude-3-5-sonnet` does **not** match, and an unrecognised model falls back to a flat `12,000`-token budget (the fraction is not applied) with a WARNING in the log. Any failure resolving the window falls back the same way, so auto-derivation can never harden into a startup error.

**The budget is not the real ceiling.** `rerank_top_n` (default `15`), `top_k_vector` (default `20`), and `top_k_bm25` cap the candidate set *before* the budget is ever applied, so raising the budget alone usually changes very little — there simply aren't more candidates to pack. `TRELIX_RETRIEVAL_SCALE_TOP_K_TO_BUDGET=true` is what actually widens the assembled context: it scales `top_k_vector` and `rerank_top_n` by `effective_budget / 12000`. It requires `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null` and it raises per-query cost — more candidates through the reranker and more tokens into synthesis.

### Context Compression (SeleCom)

Off by default. When `TRELIX_RETRIEVAL_COMPRESSION=true`, context assembly runs in two waves. Wave 1 is the existing uncompressed pack, untouched. Wave 2 takes **only** the candidates wave 1 could not fit — the ones that were previously dropped — compresses them, and re-offers them against the leftover budget. The compressed selection is therefore always a superset of the uncompressed one, which is why ranking metrics (Recall/MRR/nDCG) cannot degrade by construction. A result that already fits is never touched, so its text stays byte-identical.

Per wave-2 candidate the packer walks a ladder and stops at the first rung that fits: the target ratio tightened to the leftover budget, then a floor of signature + docstring only, then skip. A skip is recorded in the per-query compression stats rather than happening silently.

**Citation fidelity** is the point of the design: kept spans are rendered as separate `[Lines a-b] <qualified_name>` blocks separated by explicit `# ... N lines elided ...` markers, so no line-range header ever claims lines the text below it does not contain.

The `extractive` provider does zero query-time inference and picks one of two scoring paths at runtime:

- **`sub_chunk`** — scores each stored sub-chunk by cosine similarity against the query embedding, reusing vectors already written at index time. No new embedding, API, or network call.
- **`lexical`** — splits the body on blank lines and brace balance and scores segments by query-token overlap. Works in any language.

**`lexical` is the common path**, for three compounding reasons: sub-chunk rows only exist when `TRELIX_CHUNKER_MULTI_GRANULARITY=true` at index time (it defaults to `false`); the multi-granularity chunker parses with the Python grammar only, so even when enabled the rows are Python-only; and the `sub_chunk` path additionally needs the query embedding to already be sitting in the `embed_query` LRU (`TRELIX_RETRIEVAL_QUERY_CACHE_SIZE`, default `256`, `0` disables it) — compression peeks that cache and never computes an embedding itself.

Ratios are per intent. `1.0` means the intent opts out entirely — the compressor is not even constructed for that query.

| Intent | Ratio | Why |
|---|---|---|
| `symbol_lookup` | `1.0` (off) | the body *is* the answer — never elide it |
| `file_overview` | `1.0` (off) | structural walk of one file — verbatim |
| `project_overview` | `1.0` (off) | already summary-level (module/README symbols) |
| `config_lookup` | `1.0` (off) | config values are the answer — never elide |
| `comparison` | `0.65` | both sides must fit, but detail still matters |
| `feature_flow` | `0.45` | many hops matter more than any one full body |
| `dependency_map` | `0.30` | breadth over depth — coverage is the answer |
| `blast_radius` | `0.30` | "what breaks" = how many callers, not their guts |

Unknown or absent intents fall back to `TRELIX_RETRIEVAL_COMPRESSION_RATIO` (`0.45`).

**Expected effect:** roughly a 30–60% reduction in synthesis input tokens and roughly a 15–35% latency reduction on a network-API synthesis path, where the token count dominates wall-clock time. These are expectations for that shape of workload, not a benchmark measured in this repo — a local or cached synthesis path will see far less. Compression does not make retrieval itself faster and is not a "4x faster" feature.

### LLM / Synthesis

| Variable | Default | Description |
|---|---|---|
| `TRELIX_LLM_PROVIDER` | `openai` | LLM provider used for answer synthesis. One of: `openai`, `azure`, `anthropic`, `bedrock`, `vertex`, `litellm` — see [PROVIDERS.md](PROVIDERS.md#llm-providers-for-trelix-ask) |
| `TRELIX_LLM_MODEL` | `gpt-4o` | Chat model for synthesis. Used verbatim by the `openai`, `anthropic`, and `vertex` backends, and it is the model name the auto-derived context budget resolves its window from — see [Model-Aware Context Budget](#model-aware-context-budget). |
| `AZURE_CHAT_MODEL` | `gpt-4o` | Azure chat deployment name — what the `azure` backend actually calls, instead of `TRELIX_LLM_MODEL` |
| `ANTHROPIC_API_KEY` | _(none)_ | Anthropic API key — required when `TRELIX_LLM_PROVIDER=anthropic` |
| `TRELIX_LLM_THINKING_ENABLED` | `false` | Opt the answer synthesizer into Anthropic extended thinking. Only has an effect when `TRELIX_LLM_PROVIDER=anthropic` — every other backend accepts the flag and ignores it. See [Extended Thinking (Anthropic)](#extended-thinking-anthropic). |
| `TRELIX_LLM_THINKING_BUDGET_TOKENS` | `4096` | Value sent as `thinking.budget_tokens` on the Anthropic Messages API request. Not range-validated locally — `0` or a negative value is accepted by config and forwarded to the API as-is. |
| `TRELIX_RETRIEVAL_AGENTIC` | `false` | Enable the agentic ReAct loop — the LLM iteratively issues retrieval calls before producing a final answer |

### Extended Thinking (Anthropic)

Extended thinking is a **request parameter**, not a model: enabling it sends `thinking={"type": "enabled", "budget_tokens": N}` alongside the messages. You do not change `TRELIX_LLM_MODEL` to turn it on, and there is no separate "thinking model" to select.

Opt-in is **per call site, and only the synthesizer opts in.** The index-time LLM call sites — contextual chunking (one call per symbol) and file summarization (one call per file) — deliberately never pass the flag. A single global switch would multiply indexing cost several-fold across thousands of calls for no retrieval-quality gain, so the flag is wired only where a human is waiting on one answer.

Behaviour to expect when it is on:

- **`temperature` is forced to `1.0`** for that request, overriding the temperature the synthesizer would otherwise use — the API requires it.
- **Incompatible with a forced `tool_choice`**; do not combine it with a call site that pins tool selection.
- **Thinking bills as output tokens.** There is no separate thinking counter — the tokens land in `output_tokens`, so raising `TRELIX_LLM_THINKING_BUDGET_TOKENS` directly raises the per-answer bill.
- Thinking blocks are split out of the response rather than concatenated into the answer: `ChatResponse.thinking` carries the reasoning text (`None` when absent), alongside the `cache_read_tokens` / `cache_write_tokens` counters.

### Agentic ReAct Loop — Persistent Sessions

The agentic loop (`trelix ask --agentic` / `TRELIX_RETRIEVAL_AGENTIC=true`) persists its turn history to `agent_sessions`/`agent_turns` tables in the repo's `.trelix/index.db`, keyed by a client-supplied or auto-generated `session_id`. Resume a session with `trelix ask --session <id>` (implies `--agentic`) or the MCP `ask_agent` tool's `session_id` argument.

| Variable | Default | Description |
|---|---|---|
| `TRELIX_RETRIEVAL_AGENT_MAX_TURNS` | `8` | Maximum ReAct turns per `ask_agent`/`--agentic` call (1–20) |
| `TRELIX_RETRIEVAL_AGENT_TOKEN_BUDGET` | `6000` | Token budget for `HistoryCompressor` when trimming turn history (minimum 1000) |
| `TRELIX_RETRIEVAL_AGENT_SESSION_MAX_AGE_SECONDS` | `604800` (7 days) | Idle time before a persisted agent session is auto-evicted. `0` disables eviction entirely. Use `trelix agent sessions clear <id>` (or the `agent_clear_session` MCP tool) to remove a session explicitly |

### File walker (which files get indexed)

This is the only mechanism trelix has for including/excluding paths. There is **no
`.trelixignore` and no ignore section in any config file** — exclusion is `.gitignore`
(repo-root *and* nested, as of v3.1.2) plus the three list variables below.

> **These variables are read from the process environment only.** `WalkerConfig` does not
> declare `env_file`, so a `TRELIX_WALKER_*` entry in `.env` is silently ignored.

| Variable | Default | Description |
|---|---|---|
| `TRELIX_WALKER_RESPECT_GITIGNORE` | `true` | Honour `.gitignore` via `pathspec`. **Since v3.1.2 nested `.gitignore` files are read too**, with git's own semantics: each file's patterns are matched relative to its own directory, and the `.gitignore` closest to a path wins (so a deeper `!keep.log` re-includes what its parent excluded). Before v3.1.2 only `<repo>/.gitignore` was read, which is why patterns previously had to be path-qualified in the root file. Set `false` to index everything git ignores |
| `TRELIX_WALKER_FOLLOW_SYMLINKS` | `true` | Whether the walk follows symlinks out of `repo_path`. **Default `true` is the historical behaviour**: a symlink whose target lives outside the repository is indexed, and `rel_path` is computed on the unresolved path so it is reported as though it sat inside (`linked_dir/secret.py`). Set `false` to confine the walk by resolved path. Opt-in because confining by default would silently drop files from repos that symlink to shared or vendored directories. Symlinks pointing *inside* the repo are indexed either way |
| `TRELIX_WALKER_MAX_FILE_SIZE_BYTES` | `500000` | Files larger than this are skipped entirely |
| `TRELIX_WALKER_LANGUAGES` | 26 languages (Python, JS, TS, TSX, Go, Rust, Java, Kotlin, Ruby, C++, C, C#, Razor, cshtml, csproj, Markdown, JSON, YAML, TOML, HTML, CSS, **shell, dockerfile, make, sql, proto**) | JSON array of language names to parse, e.g. `'["python","go"]'`. **This REPLACES the default list rather than adding to it** — pinning it means you do not get languages added in later versions. The five ops languages have no structural extractor yet and are parsed into line windows (see `parser/extractors/line_window.py`) |
| `TRELIX_WALKER_EXTRA_IGNORE_DIRS` | 30 entries (`.git`, `node_modules`, `__pycache__`, `venv`, `.venv`, `dist`, `build`, `target`, `.next`, `vendor`, `bin`, `obj`, `packages`, `.trelix`, …) | JSON array of directory names to skip. Matched on the exact directory basename — `Bin/` and `OBJ/` are **not** caught |
| `TRELIX_WALKER_EXTRA_IGNORE_FILENAMES` | `package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, `bun.lockb`, `angular.json` | JSON array of exact filenames to skip |
| `TRELIX_WALKER_EXTRA_IGNORE_EXTENSIONS` | 27 entries (`.pyc`, `.so`, `.dll`, `.png`, `.pdf`, `.zip`, `.min.js`, `.lock`, …) | JSON array of file extensions to skip |

> **Setting a list REPLACES the default, it does not append.** `TRELIX_WALKER_EXTRA_IGNORE_DIRS='["dist"]'`
> stops skipping `node_modules`, `.venv`, `.git`, and the other 27 defaults. To add one entry
> you must restate the whole list.

#### `bin`, `obj` and `packages` — and why your monorepo's source may be missing

Those three defaults are .NET build output (`packages/` is a NuGet restore, `bin/` and `obj/`
are MSBuild). Two of them are also *source* directories elsewhere: every pnpm/npm/yarn/lerna
monorepo keeps first-party code under `packages/`, and a Node CLI keeps real executables in
`bin/`. Because directory exclusion is enforced during traversal, the walk never descends and
the index simply contains none of it — while the run still reports `errors: 0`.

Measured across six repositories in one workspace, counted in **walk-units** — files trelix
would actually index, after the language, size, filename and `.gitignore` filters:

| repo | dir | declared first-party source hidden | indexed today | if that entry is removed |
|---|---|---|---|---|
| repo A | `packages` | 36 | 598 | 634 (+6%) |
| repo B | `packages` | 168 | 31 | 199 |
| repo C | `packages` | 137 | 200 | 337 |
| repo D | `packages` | 104 | 510 | 614 |
| repo E | `bin` | 189 (183 `.js`) | 2765 | 2972 |

Repo B is the extreme case: 31 of 212 tracked files indexed, with exactly one file in a code
language and 0 call edges. The right-hand column is what removing the entry costs — size it
with `--dry-run` first.

Two operations that are easy to conflate, because on four of these repos they coincide. Removing
the entry admits **every** directory of that name; flipping `index_conditional_dirs` admits only
the ones the probe can **prove**. Repos A–D measure the same either way. Repo E is where they
diverge: it holds a second `bin/` with no `package.json` beside it, so flipping the default admits
2954 while removing the entry admits 2972 — 18 more files of hand-written, git-tracked `.cjs`
tooling the probe cannot prove is first-party. **Removing the entry is always the wider of the
two**, which is why the column above is the removal figure: it is the action this section tells
you to take. `declared` in the third column carries the same caveat — repo E hides 207 first-party
files, of which 189 are ones the probe can demonstrate.

As of this release trelix **detects and reports** the case instead of hiding it. When a
`packages/` sits beside a workspace manifest (`pnpm-workspace.yaml`, `lerna.json`, `nx.json`,
`rush.json`, or a `package.json` with a `workspaces` key), or a `bin/` sits beside a
`package.json` whose `bin` field points into it, the walk logs one WARNING naming the
directory and the evidence file. **The walk itself is unchanged, so this release costs nothing
extra to run.** A sibling `*.sln`/`*.slnf`/`*.csproj`/`*.vcxproj`/`Directory.Build.props`/
`Directory.Build.targets`/`packages.config`/`packages.lock.json`/`NuGet.config` keeps the
directory excluded regardless — the .NET case wins any tie, and these are matched
case-insensitively (MSBuild and NuGet resolve them that way).

Customising the list does not silence the warning; only removing the name does — which is also
what stops the directory being hidden. So restate the list without that entry (remember: the
variable replaces all 30 defaults, a comma-separated value is rejected, and
`scripts/self-index.sh` is a working reference), then check the size of the change **before**
paying for it:

```bash
trelix index . --dry-run   # files walked + token estimate, no embedding calls
```

To exclude a path, the simplest route is `.gitignore` — it is honoured by default and needs
no env var:

```bash
echo "dist/" >> .gitignore
```

### Storage

| Variable | Default | Description |
|---|---|---|
| `TRELIX_STORE_BACKEND` | `sqlite` | Vector store backend. One of: `sqlite`, `qdrant`, `lance`. The default `sqlite` backend is the sqlite-vec-backed store; the accepted **value** is the bare string `sqlite` — passing `sqlite-vec` fails validation at startup with a `literal_error` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL — required when backend is `qdrant` |
| `QDRANT_API_KEY` | _(none)_ | Qdrant API key — required for authenticated Qdrant Cloud instances |
| `QDRANT_COLLECTION` | `trelix` | Qdrant collection name |
| `QDRANT_PREFER_GRPC` | `false` | Use Qdrant's gRPC port (6334) instead of REST (6333) — lower latency, recommended for Qdrant Cloud |
| `QDRANT_TIMEOUT` | `10.0` | Client request timeout in seconds — raise for Cloud deployments with higher network latency |
| `TRELIX_STORE_BM25_READ_POOL_SIZE` | `0` | Number of read-only SQLite connections to pool for parallel `bm25_search()` calls. `0` disables pooling (default — identical to the pre-existing single-connection behavior). When set, `Retriever` automatically calls `Database.enable_bm25_read_pool()` at construction time. |

### Federation

| Variable | Default | Description |
|---|---|---|
| `TRELIX_FEDERATION_ENABLED` | `false` | **Inert.** `RetrievalConfig.federation_enabled` is declared but read nowhere in `src/`. Federated search is reached by running `trelix search-all`, which federates whether this is set or not; setting it does not make `trelix ask`/`trelix search` federate |
| `TRELIX_FEDERATION_MAX_WORKERS` | `4` | **Inert.** `RetrievalConfig.federation_max_workers` is declared and range-validated (1–16) but read nowhere in `src/`. `search-all` constructs `FederatedRetriever(registry)` with no arguments, so the `ThreadPoolExecutor` size is always the constructor's own default of 4 — which happens to equal this documented default, so the only observable symptom is that raising it does nothing |
| `TRELIX_FEDERATION_MAX_REPOS` | `50` | Maximum number of registered repos actually queried per federated search call (1–500). Registered repos beyond this cap are skipped (reported via `repos_skipped` in the MCP `federation_search_all` response); prevents an unbounded `federation_add_repo` loop from making every subsequent query scale linearly. **No effect on `trelix search-all`** — `FederatedRetriever`'s `max_repos` parameter defaults to `None` (unbounded) and the CLI passes nothing, so only the separately-distributed `trelix-mcp` server applies this |

There is no environment variable for the federation registry file path. The registry JSON file location defaults to `~/.config/trelix/repos.json` and can be overridden per-call via the `--config` CLI option (`trelix search-all --config`, `trelix federation add/list/remove --config`) or the `config_path` argument on the corresponding MCP tools. For security, MCP callers may only point `config_path` at `~/.config/trelix/` or `<mcp-server-cwd>/.trelix/` — paths outside those roots are rejected.

### Git ticket linking

Configuration for [`trelix link-tickets`](CLI_REFERENCE.md#trelix-link-tickets), which walks git history to link code symbols to external ticket references found in commit messages. Off by default — requires the repo to actually be a git checkout, and is a separate, slower pass from the main index pipeline (invoked only via `trelix link-tickets`, never automatically from indexing).

| Variable | Default | Description |
|---|---|---|
| `TRELIX_GIT_LINKER_ENABLED` | `false` | Enable git-history ticket linking. Set to `true` automatically by `trelix link-tickets`; not something you typically set directly. |
| `TRELIX_GIT_LINKER_TICKET_PATTERN` | see `TICKET_PATTERN_DEFAULT` in `core/config.py` | Regex for matching ticket IDs in commit messages. Matches Jira/Linear-style keys (`PROJ-123`, `ENG-45`), including inside branch names in merge subjects (`feature/PROJ-456-thing`), while excluding technical constants that share the same shape (`UTF-8`, `SHA-256`, `HTTP-400`). Override for other conventions, e.g. GitHub-issue style (`#\d+`). |
| `TRELIX_GIT_LINKER_MAX_COMMITS` | `5000` (min: `1`) | Maximum number of commits to walk. Bounds cost on repos with 100k+ commit histories. |
| `TRELIX_GIT_LINKER_SINCE` | _(none)_ | Only walk commits after this date, e.g. `"90 days ago"`. Passed straight through to `git log --since`. |

### Connector credentials (Jira / TestRail / Xray / Linear)

Configuration for [`trelix connector sync`](CLI_REFERENCE.md#trelix-connector-sync), which fetches artifacts from an external system and writes them to trelix's `artifacts` table. Jira and TestRail use HTTP Basic auth; Xray Cloud exchanges a client_id/client_secret for a short-lived bearer JWT; Linear uses a personal API key sent directly in the `Authorization` header with no `Bearer` prefix. All required variables per connector must be set — missing any of them fails config validation before any HTTP call is made.

| Variable | Default | Description |
|---|---|---|
| `TRELIX_JIRA_BASE_URL` | _(none, required)_ | Base URL of the Jira Cloud instance, e.g. `https://acme.atlassian.net` |
| `TRELIX_JIRA_EMAIL` | _(none, required)_ | Email address used for HTTP Basic auth against the Jira REST API |
| `TRELIX_JIRA_API_TOKEN` | _(none, required)_ | Jira API token (paired with `TRELIX_JIRA_EMAIL` for Basic auth) |
| `TRELIX_JIRA_PROJECT_KEY` | _(none, required)_ | Jira project key to sync tickets from |
| `TRELIX_JIRA_PAGE_SIZE` | `100` (max `100`) | Page size for Jira API pagination |
| `TRELIX_TESTRAIL_BASE_URL` | _(none, required)_ | Base URL of the TestRail instance, e.g. `https://acme.testrail.io` |
| `TRELIX_TESTRAIL_USERNAME` | _(none, required)_ | Username used for HTTP Basic auth against the TestRail REST API |
| `TRELIX_TESTRAIL_API_KEY` | _(none, required)_ | TestRail API key (paired with `TRELIX_TESTRAIL_USERNAME` for Basic auth) |
| `TRELIX_TESTRAIL_PROJECT_ID` | _(none, required)_ | TestRail project ID to sync test cases from |
| `TRELIX_TESTRAIL_PAGE_SIZE` | `250` (max `250` — TestRail's own API ceiling) | Page size for TestRail API pagination |
| `TRELIX_XRAY_CLIENT_ID` | _(none, required)_ | Xray Cloud client ID, issued by a Jira admin in Xray's global settings (distinct from a user's own Jira API token) |
| `TRELIX_XRAY_CLIENT_SECRET` | _(none, required)_ | Xray Cloud client secret, paired with `TRELIX_XRAY_CLIENT_ID` and exchanged for a short-lived bearer JWT |
| `TRELIX_XRAY_PROJECT_KEY` | _(none, required)_ | Jira project key whose tests to sync (Xray Cloud tests are Jira issues under the hood) |
| `TRELIX_XRAY_JIRA_BASE_URL` | _(none, required)_ | Base URL of the Jira Cloud instance backing this Xray project, e.g. `https://acme.atlassian.net` |
| `TRELIX_XRAY_PAGE_SIZE` | `100` (max `100`) | Page size for Xray's GraphQL `getTests` pagination |
| `TRELIX_LINEAR_API_KEY` | _(none, required)_ | Linear personal API key — sent verbatim as `Authorization: <key>` (no `Bearer` prefix) |
| `TRELIX_LINEAR_TEAM_KEY` | _(none, required)_ | Linear team key to scope issue sync to, e.g. `ENG` |
| `TRELIX_LINEAR_PAGE_SIZE` | `100` (max `100`) | Page size for Linear's cursor-paginated `issues` query — not a confirmed Linear platform ceiling, chosen to stay well under its GraphQL query-complexity cap |

### REST API

| Variable | Default | Description |
|---|---|---|
| `TRELIX_API_AUTH_TOKEN` | _(none)_ | Shared secret for `trelix serve`'s REST API. Opt-in: unset (the default) leaves every route open, matching the same "off by default" pattern as `TRELIX_OTEL_ENABLED` and `TRELIX_TELEMETRY_ENABLED`. When set, every route except `GET /health` requires a matching `X-Trelix-Api-Key` header (checked with a constant-time comparison) — see [USER_GUIDE.md § API Quick Reference](USER_GUIDE.md#14-api-quick-reference). |

### MCP Server

| Variable | Default | Description |
|---|---|---|
| `TRELIX_MCP_MAX_SUBSCRIBERS` | `1000` | Maximum number of concurrent resource subscriptions `trelix-mcp` will accept. Re-subscribing an existing `subscription_id` never counts as growth. Once at capacity, `subscribe_resource` returns a soft error (`{"subscribed": false, ...}`) instead of raising. |
| `TRELIX_MCP_SUBSCRIPTION_TTL_SECONDS` | `3600` | Time-to-live (seconds) for an inactive resource subscription before it is evicted from the `SubscriptionRegistry`. Expired subscriptions are swept lazily on the next registry access. |

### Observability (OpenTelemetry)

Requires `pip install trelix[otel]`. See [OBSERVABILITY.md](OBSERVABILITY.md) for the full span/attribute reference.

| Variable | Default | Description |
|---|---|---|
| `TRELIX_OTEL_ENABLED` | `false` | Emit one OpenTelemetry span per retrieval leg (vector/BM25/grep/sparse/sub-chunk/file-summary) plus pipeline-stage spans (planner/fusion/expansion/rerank/pagerank/assembly). Zero import cost and zero behavior change when disabled. |
| `OTEL_SERVICE_NAME` | `trelix` | Service name attached to the installed `TracerProvider`'s resource attributes. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | _(none)_ | OTLP collector endpoint. If unset, spans are still created but have nowhere to export to unless a host application configures its own exporter/processor before trelix runs. |

---

## .env File Example

Copy this to `./.env` — the directory you run `trelix` **from**, not necessarily the repo you
are indexing (see [Configuration Methods](#configuration-methods)) — and fill in the values
relevant to your setup. Lines beginning with `#` are comments and are ignored.

```dotenv
# =============================================================================
# Trelix v3.1.5 — complete .env example
# Copy to .env and fill in values. Never commit this file.
# =============================================================================

# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

# Provider: local | openai | azure | voyage | nomic-code | bge-code (experimental)
TRELIX_EMBEDDER_PROVIDER=local

# OpenAI embeddings
# TRELIX_EMBEDDER_OPENAI_MODEL=text-embedding-3-large
# OPENAI_API_KEY=sk-...

# Azure embeddings — the deployment var is AZURE_EMBEDDINGS_MODEL (explicit
# alias, so no TRELIX_EMBEDDER_ prefix)
# AZURE_EMBEDDINGS_MODEL=text-embedding-3-large
# AZURE_API_KEY=...
# AZURE_ENDPOINT=https://<your-resource>.openai.azure.com/

# Voyage embeddings (Matryoshka dimension: 256 | 512 | 1024 | 2048)
# TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=1024
# VOYAGE_API_KEY=...

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Multi-query expansion
TRELIX_RETRIEVAL_MULTI_QUERY=false
TRELIX_RETRIEVAL_MULTI_QUERY_COUNT=2

# Streaming indexing
TRELIX_INDEXER_STREAMING=false

# FLARE iterative re-retrieval (max retries: 1-3)
TRELIX_RETRIEVAL_FLARE=false
TRELIX_RETRIEVAL_FLARE_MAX_RETRIES=1

# HyDE fallback
TRELIX_RETRIEVAL_HYDE_FALLBACK=false

# Extra retrieval legs
TRELIX_RETRIEVAL_FILE_SUMMARY_LEG=false
TRELIX_RETRIEVAL_PAGERANK_BOOST=false
TRELIX_RETRIEVAL_PAGERANK_PERSONALIZATION=false
TRELIX_RETRIEVAL_GRAPH_SEARCH_ENABLED=false

# Query telemetry -> query_telemetry table. Index-level, so no RETRIEVAL_ infix.
TRELIX_TELEMETRY_ENABLED=false

# Generate LLM file summaries at index time (requires LLM provider)
TRELIX_FILE_SUMMARIES_ENABLED=false

# Advanced retrieval tuning (internal knobs — no CLI/REST/MCP param yet)
# TRELIX_RETRIEVAL_PATH_FILTER_OVERSAMPLE=3
# TRELIX_RETRIEVAL_CONTEXT_BUDGET_PER_SOURCE=false

# Context budget: an explicit int (default 12000) or null/none/auto/"" to
# auto-derive from the synthesis model's context window.
TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=12000
# TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null
# Only read when the budget is null (range 0.1-0.9)
# TRELIX_RETRIEVAL_CONTEXT_WINDOW_FRACTION=0.5
# Scales top_k_vector/rerank_top_n too — needs the budget to be null, costs more
# TRELIX_RETRIEVAL_SCALE_TOP_K_TO_BUDGET=false

# Query-conditioned compression (SeleCom) — extractive is the only provider
TRELIX_RETRIEVAL_COMPRESSION=false
# TRELIX_RETRIEVAL_COMPRESSION_PROVIDER=extractive
# Fallback ratio for unknown intents only (range 0.1-1.0)
# TRELIX_RETRIEVAL_COMPRESSION_RATIO=0.45
# Per-intent override, e.g. BLAST_RADIUS defaults to 0.30 — NOTE: the per-intent
# suffix form is read from the process environment only and is IGNORED here in
# .env (see "Four dynamic, suffix-named settings" above). Export it in your
# shell/unit/container instead:
#   export TRELIX_RETRIEVAL_COMPRESSION_RATIO_BLAST_RADIUS=0.5
# TRELIX_RETRIEVAL_COMPRESSION_MIN_TOKENS=120

# ---------------------------------------------------------------------------
# LLM / Synthesis
# ---------------------------------------------------------------------------

# Provider: openai | azure | anthropic | bedrock | vertex | litellm
TRELIX_LLM_PROVIDER=openai

# Chat model. Also the name the auto-derived context budget resolves its
# window from, for every provider.
TRELIX_LLM_MODEL=gpt-4o
# OPENAI_API_KEY=sk-...  (shared with embedder if both use OpenAI)

# Azure chat — the azure backend calls this deployment, not TRELIX_LLM_MODEL
# AZURE_CHAT_MODEL=gpt-4o

# Anthropic
# ANTHROPIC_API_KEY=sk-ant-...

# Anthropic extended thinking — synthesizer only, forces temperature=1.0,
# and the budget bills as OUTPUT tokens. Ignored by non-Anthropic backends.
TRELIX_LLM_THINKING_ENABLED=false
# TRELIX_LLM_THINKING_BUDGET_TOKENS=4096

# Agentic ReAct loop
TRELIX_RETRIEVAL_AGENTIC=false

# Agentic ReAct loop — persistent session tuning
# TRELIX_RETRIEVAL_AGENT_MAX_TURNS=8
# TRELIX_RETRIEVAL_AGENT_TOKEN_BUDGET=6000
# TRELIX_RETRIEVAL_AGENT_SESSION_MAX_AGE_SECONDS=604800

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

# Backend: sqlite | qdrant | lance  ("sqlite" is the sqlite-vec store; the
# literal string "sqlite-vec" is NOT accepted and fails validation)
TRELIX_STORE_BACKEND=sqlite

# Qdrant (required when backend=qdrant)
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=...
# QDRANT_COLLECTION=trelix
# QDRANT_PREFER_GRPC=false
# QDRANT_TIMEOUT=10.0

# Parallel read-only BM25 connections (0 = disabled, default)
# TRELIX_STORE_BM25_READ_POOL_SIZE=4

# ---------------------------------------------------------------------------
# Federation
# ---------------------------------------------------------------------------

# INERT: both are declared in RetrievalConfig but read nowhere in src/. Run
# `trelix search-all` to federate; the pool size is always FederatedRetriever's
# own constructor default of 4. Listed only so the names are not mistaken for
# typos when you see them in config dumps.
# TRELIX_FEDERATION_ENABLED=false
# TRELIX_FEDERATION_MAX_WORKERS=4
# Applied only by the out-of-tree trelix-mcp server — the CLI leaves
# FederatedRetriever(max_repos=None), i.e. unbounded.
# TRELIX_FEDERATION_MAX_REPOS=50

# Federation registry file path has no env var override — use --config (CLI)
# or config_path (MCP tools) instead. Defaults to ~/.config/trelix/repos.json.

# ---------------------------------------------------------------------------
# Git ticket linking (trelix link-tickets)
# ---------------------------------------------------------------------------

# TRELIX_GIT_LINKER_ENABLED=false
# TRELIX_GIT_LINKER_TICKET_PATTERN=   # default excludes UTF-8/SHA-256-style noise
# TRELIX_GIT_LINKER_MAX_COMMITS=5000
# TRELIX_GIT_LINKER_SINCE=90 days ago

# ---------------------------------------------------------------------------
# Connector credentials (trelix connector sync)
# ---------------------------------------------------------------------------

# Jira Cloud (base_url/email/api_token/project_key all required to sync)
# TRELIX_JIRA_BASE_URL=https://acme.atlassian.net
# TRELIX_JIRA_EMAIL=bot@acme.com
# TRELIX_JIRA_API_TOKEN=...
# TRELIX_JIRA_PROJECT_KEY=PROJ
# TRELIX_JIRA_PAGE_SIZE=100

# TestRail (base_url/username/api_key/project_id all required to sync)
# TRELIX_TESTRAIL_BASE_URL=https://acme.testrail.io
# TRELIX_TESTRAIL_USERNAME=bot@acme.com
# TRELIX_TESTRAIL_API_KEY=...
# TRELIX_TESTRAIL_PROJECT_ID=7
# TRELIX_TESTRAIL_PAGE_SIZE=250

# Xray Cloud (client_id/client_secret/project_key/jira_base_url all required to sync)
# TRELIX_XRAY_CLIENT_ID=...
# TRELIX_XRAY_CLIENT_SECRET=...
# TRELIX_XRAY_PROJECT_KEY=PROJ
# TRELIX_XRAY_JIRA_BASE_URL=https://acme.atlassian.net
# TRELIX_XRAY_PAGE_SIZE=100

# Linear (api_key/team_key both required to sync)
# TRELIX_LINEAR_API_KEY=...
# TRELIX_LINEAR_TEAM_KEY=ENG
# TRELIX_LINEAR_PAGE_SIZE=100

# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------

# Opt-in auth for `trelix serve` — unset means every route stays open
# TRELIX_API_AUTH_TOKEN=...

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

# TRELIX_MCP_MAX_SUBSCRIBERS=1000
# TRELIX_MCP_SUBSCRIPTION_TTL_SECONDS=3600
```

---

## Per-Project Configuration

**Trelix has no per-project config file.** There is no TOML, YAML, or JSON settings file —
nothing is read from `.trelix/config.toml`, `trelix.toml`, `pyproject.toml`, or any other
file-based config source. Configuration comes from exactly two places, both of which are
process-wide rather than per-repo:

1. **Environment variables** — every setting in the tables above.
2. **A `.env` file** — loaded from the **current working directory** (`./.env`), which is a
   property of where you launch the process, not of the repo you pass as the positional
   `REPO` argument.

A `.trelix/` directory holds trelix's own generated data, never configuration: the index
lives at `<repo>/.trelix/index.db`, the optional audit log at `<cwd>/.trelix/audit.db`, and
trelix auto-writes a `.gitignore` alongside the index so watchers and git ignore it. The
file walker also skips `.trelix/` while indexing. Creating a `.trelix/config.toml` has no
effect whatsoever — the file is never opened.

> **Security note.** Because no config file is read, putting a setting like
> `TRELIX_API_AUTH_TOKEN` in a `.trelix/config.toml` would leave `trelix serve` running with
> **every route open** while looking configured. Auth, connector credentials, and API keys
> must be set as environment variables or in `.env`.

### Sharing settings across a team

Since no committed config file is consulted, share defaults the way the loading mechanism
actually supports:

- **Commit a `.env.example`** documenting the variables your project expects, and have each
  developer copy it to a git-ignored `.env`. Keep `.env` out of version control — it is
  where secrets live.
- **Set the variables in CI/CD** as pipeline environment variables or secrets.
- **Wrap invocations in a `Makefile` target or shell script** that exports the project's
  variables before calling `trelix`, so everyone gets the same settings regardless of cwd.

Remember that `.env` is resolved against the cwd: if contributors run `trelix` from
different directories, an env var exported by a wrapper script is more reliable than a
`.env` file.

---

## MCP Server

trelix ships a Model Context Protocol server (`trelix-mcp`) that exposes indexed repositories as MCP resources and tools, allowing MCP-compatible clients (e.g. Claude Desktop) to query trelix directly.

### Resource subscriptions (v2.5.0)

trelix-mcp v2.5.0 advertises `resources.subscribe = true` in its server capabilities and exposes two new tools:

| Tool | Parameters | Description |
|---|---|---|
| `subscribe_resource` | `uri`, `subscription_id` | Subscribe to change notifications for a `trelix://` resource URI |
| `unsubscribe_resource` | `subscription_id` | Cancel an active subscription |

**URI scheme:** `trelix://repo/{repo_path}/manifest`

**Wire protocol:**
1. Client calls `subscribe_resource(uri, subscription_id)` — the server registers the subscription.
2. When a watched file changes, trelix-mcp emits a `notifications/resources/updated` notification (URI only, with `subscriptionId` in `params._meta`).
3. Client calls `resources/read` to fetch the updated content.

Subscriptions are held in-memory (not persisted across server restarts). The subscription registry is thread-safe.
