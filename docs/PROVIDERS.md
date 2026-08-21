# trelix v3.1.2 — Providers Reference

Complete guide to all embedding providers and LLM providers supported by trelix.

---

## Embedding Providers

### Comparison Table

| Provider | Key Required | Dimensions | CoIR Score | Speed | Use Case |
|---|---|---|---|---|---|
| `local` | No | 384 | — | Medium (CPU) | Development, offline, privacy |
| `openai` | `OPENAI_API_KEY` | 1536 / 3072 | — | Fast (API) | Production, general use |
| `azure` | `AZURE_API_KEY` + `AZURE_ENDPOINT` | 1536 / 3072 | — | Fast (API) | Enterprise, Azure customers |
| `voyage` | `VOYAGE_API_KEY` | 1024 | 56.26 avg | Fast (API) | Best API-based code retrieval |
| `local-code` | No (HuggingFace) | 4096 | 67.41 avg | Slow (GPU rec.) | Highest offline accuracy |
| `bge-code` | No (HuggingFace) | 768 | — | Slow (GPU rec.) | Self-hosted, no API cost |
| `nomic-code` | No (HuggingFace) | 768 | — | Medium (CPU) | Self-hosted alternative |
| `bedrock-titan` | AWS credentials | 256 / 512 / 1024 | — | Fast (API) | AWS-native deployments |
| `bedrock-cohere` | AWS credentials | 1024 | — | Fast (API) | AWS + strong code retrieval |

CoIR = Code Information Retrieval benchmark (higher is better). `—` means not yet benchmarked on CoIR.

---

### local (sentence-transformers)

The **default provider**. No API key needed, no internet access required after the first run.

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **First run**: downloads ~80 MB model, cached locally afterwards
- **Install**: included in the base `pip install trelix`
- **Best for**: development, offline environments, privacy-sensitive repos

```bash
# Default — no configuration needed
trelix index ./my-repo
```

```env
TRELIX_EMBEDDER_PROVIDER=local
```

---

### openai

Production-quality embeddings via the OpenAI Embeddings API.

- **Default model**: `text-embedding-3-large` (3072 dims)
- **Alternate model**: `text-embedding-3-small` (1536 dims) — cheaper, slightly lower quality
- **Install**: `pip install trelix[openai]` (or base install — openai is a soft dep)

```bash
OPENAI_API_KEY=sk-... TRELIX_EMBEDDER_PROVIDER=openai trelix index ./my-repo
```

```env
TRELIX_EMBEDDER_PROVIDER=openai
OPENAI_API_KEY=sk-...
TRELIX_EMBEDDER_OPENAI_MODEL=text-embedding-3-large   # optional, this is the default
```

To use the smaller, cheaper model:

```env
TRELIX_EMBEDDER_OPENAI_MODEL=text-embedding-3-small
```

---

### azure

Azure OpenAI Service embeddings. Same quality as `openai` but routed through your Azure resource.

- **Dimensions**: 1536 or 3072 (depends on your deployment)
- **Required env**: `AZURE_API_KEY`, `AZURE_ENDPOINT`, `AZURE_EMBEDDINGS_MODEL`

```env
TRELIX_EMBEDDER_PROVIDER=azure
AZURE_API_KEY=...
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2025-04-01-preview
AZURE_EMBEDDINGS_MODEL=text-embedding-3-large
```

The `AZURE_CHAT_MODEL` variable controls which deployment is used for LLM calls (`trelix ask`); it is separate from the embeddings deployment.

---

### voyage

Best API-based code retrieval quality. **voyage-code-3** achieves 56.26 avg on the CoIR benchmark — the highest score among API-based providers.

- **Model**: `voyage-code-3`
- **Dimensions**: 1024 (Matryoshka — supports 256 / 512 / 1024 / 2048)
- **Context window**: 16k tokens per document
- **Install**: `pip install trelix[voyage]`

```bash
VOYAGE_API_KEY=pa-... TRELIX_EMBEDDER_PROVIDER=voyage trelix index ./my-repo
```

```env
TRELIX_EMBEDDER_PROVIDER=voyage
VOYAGE_API_KEY=pa-...
TRELIX_EMBEDDER_VOYAGE_MODEL=voyage-code-3   # default
```

**Matryoshka dimensions** (smaller = faster HNSW search, slightly lower quality):

```env
# Reduce output dimension — 512 is a good quality/speed trade-off
TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS=512
```

---

### local-code (SFR-Embedding-Code-2B_R)

The highest-accuracy offline option. **CoIR 67.41 avg** — top of the CoIR leaderboard as of 2025.

- **Model**: `Salesforce/SFR-Embedding-Code-2B_R`
- **Dimensions**: 4096
- **RAM**: ~8 GB GPU VRAM (or ~16 GB CPU RAM in slow mode)
- **Install**: `pip install trelix[local-code]`

```bash
TRELIX_EMBEDDER_PROVIDER=local-code trelix index ./my-repo
```

```env
TRELIX_EMBEDDER_PROVIDER=local-code
```

GPU is strongly recommended. CPU works but is significantly slower on large repos.

---

### bge-code (BAAI/BGE-Code-v1)

Self-hosted, no API cost, optimized for code. Uses the `FlagEmbedding` library.

- **Model**: `BAAI/bge-code-v1`
- **Dimensions**: 768
- **Install**: `pip install trelix[bge-code]`

```bash
TRELIX_EMBEDDER_PROVIDER=bge-code trelix index ./my-repo
```

```env
TRELIX_EMBEDDER_PROVIDER=bge-code
```

GPU recommended. CPU works for smaller repos.

---

### nomic-code (CodeRankEmbed)

Self-hosted alternative with a smaller footprint than `bge-code`. Uses `sentence-transformers`.

- **Model**: `nomic-ai/CodeRankEmbed`
- **Dimensions**: 768
- **Install**: `pip install trelix[local]`

```bash
TRELIX_EMBEDDER_PROVIDER=nomic-code trelix index ./my-repo
```

```env
TRELIX_EMBEDDER_PROVIDER=nomic-code
```

---

### bedrock-titan

Amazon Titan Embeddings V2 via AWS Bedrock. No separate API key — uses standard AWS credentials.

- **Model**: `amazon.titan-embed-text-v2:0`
- **Dimensions**: 256, 512, or 1024 (configurable)
- **Install**: `pip install trelix[bedrock]`

```env
TRELIX_EMBEDDER_PROVIDER=bedrock-titan
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

Dimension trade-off:

| Dimension | Quality | Storage |
|---|---|---|
| 1024 | Best (matches Voyage) | 4x of 256 |
| 512 | Good (recommended balance) | 2x of 256 |
| 256 | Lowest | Minimum |

```env
TRELIX_EMBEDDER_BEDROCK_TITAN_DIMENSIONS=512   # override default 1024
```

---

### bedrock-cohere

Cohere Embed English V3 via AWS Bedrock. Asymmetric retrieval (separate doc/query embeddings).

- **Model**: `cohere.embed-english-v3`
- **Dimensions**: 1024 (fixed)
- **Install**: `pip install trelix[bedrock]`

```env
TRELIX_EMBEDDER_PROVIDER=bedrock-cohere
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

---

### Switching Providers

**Important**: embedding vectors from different providers are not compatible — dimensions and spaces differ. When switching providers, reset the index first.

```bash
# 1. Clear old embeddings
trelix migrate-vectors --reset ./my-repo

# 2. Re-index with the new provider
TRELIX_EMBEDDER_PROVIDER=openai trelix index ./my-repo
```

The **DimensionGuard** built into trelix prevents silent mismatches: if the stored index dimension does not match the configured provider dimension, trelix will raise an error rather than silently produce wrong results.

---

## LLM Providers (for `trelix ask`)

LLM providers are configured separately from embedding providers. You can mix and match: for example, embed with `voyage` and synthesize answers with Anthropic.

### openai (default)

```env
TRELIX_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
TRELIX_LLM_MODEL=gpt-4o   # optional, gpt-4o is the default
```

### azure

```env
TRELIX_LLM_PROVIDER=azure
AZURE_API_KEY=...
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_CHAT_MODEL=gpt-4o
AZURE_API_VERSION=2025-04-01-preview
```

### anthropic

```bash
pip install "trelix[anthropic]"
```

```env
TRELIX_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
TRELIX_LLM_MODEL=claude-sonnet-4-6   # optional
```

### bedrock (AWS)

Primary model is tried first; if it returns a `ValidationException` (model unavailable in your region or throughput tier), trelix automatically retries with the fallback model.

```bash
pip install "trelix[bedrock]"
```

```env
TRELIX_LLM_PROVIDER=bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Optional: override inference profile IDs (us.* prefix required)
TRELIX_LLM_BEDROCK_PRIMARY_MODEL=us.anthropic.claude-sonnet-4-6
TRELIX_LLM_BEDROCK_FALLBACK_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
```

AWS IAM profile auth (alternative to key/secret):

```env
AWS_PROFILE=my-profile
```

### vertex (Google)

```bash
pip install "trelix[vertex]"
```

```env
TRELIX_LLM_PROVIDER=vertex
GOOGLE_CLOUD_PROJECT=my-project
GOOGLE_CLOUD_LOCATION=us-central1

# Or use an API key (AI Studio — simpler, no project needed)
GOOGLE_API_KEY=...
```

### litellm (100+ providers)

Route through [LiteLLM](https://github.com/BerriAI/litellm) to access any of its 100+ supported models and providers via a single proxy interface.

```bash
pip install "trelix[litellm]"
```

```env
TRELIX_LLM_PROVIDER=litellm
TRELIX_LLM_LITELLM_MODEL=bedrock/claude-3-5-sonnet   # any LiteLLM model string
```

Any environment variable expected by the underlying provider (e.g. `OPENAI_API_KEY`, `AWS_*`) must still be set — LiteLLM forwards them to the target provider.

### Extended thinking support

Extended thinking is an Anthropic-API request parameter (`thinking={"type": "enabled", "budget_tokens": N}`), not a model name. `TRELIX_LLM_THINKING_ENABLED=true` opts the answer synthesizer into it; see [CONFIGURATION.md § Extended Thinking (Anthropic)](CONFIGURATION.md#extended-thinking-anthropic) for the full behaviour contract.

| Provider | Extended thinking |
|---|---|
| `anthropic` | **Supported** — sends the `thinking` parameter, forces `temperature=1.0` for that call, and returns the reasoning text on `ChatResponse.thinking` |
| `openai` | Accepted and ignored |
| `azure` | Accepted and ignored |
| `bedrock` | Accepted and ignored — even for `us.anthropic.*` profiles |
| `vertex` | Accepted and ignored |
| `litellm` | Accepted and ignored |

"Accepted and ignored" is literal: every backend takes the flag in its `chat()`/`stream()` signature so the synthesizer needs no provider branch, but only the Anthropic backend acts on it. Setting the flag on another provider is a silent no-op, not an error — nothing is added to the request and `ChatResponse.thinking` stays `None`.

### Model-aware context budgets

Setting `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null` derives the context-assembly budget from the model's context window instead of the fixed `12000` default (see [CONFIGURATION.md § Model-Aware Context Budget](CONFIGURATION.md#model-aware-context-budget)). Two provider-specific facts matter:

**1. Resolution always reads `TRELIX_LLM_MODEL`** — never the provider-specific model field that is actually sent on the wire. So with `TRELIX_LLM_PROVIDER=bedrock`, the window is resolved from `TRELIX_LLM_MODEL` (default `gpt-4o` → 128,000), *not* from `TRELIX_LLM_BEDROCK_PRIMARY_MODEL`. The same gap applies to `AZURE_CHAT_MODEL` and `TRELIX_LLM_LITELLM_MODEL`. Set `TRELIX_LLM_MODEL` to a recognised name matching the model you actually call if you want the derived budget to be meaningful.

| Provider | Model actually called | Model used to resolve the window | Same? |
|---|---|---|---|
| `openai` | `TRELIX_LLM_MODEL` | `TRELIX_LLM_MODEL` | yes |
| `anthropic` | `TRELIX_LLM_MODEL` | `TRELIX_LLM_MODEL` | yes |
| `vertex` | `TRELIX_LLM_MODEL` | `TRELIX_LLM_MODEL` | yes |
| `azure` | `AZURE_CHAT_MODEL` (deployment name) | `TRELIX_LLM_MODEL` | **may differ** |
| `bedrock` | `TRELIX_LLM_BEDROCK_PRIMARY_MODEL` | `TRELIX_LLM_MODEL` | **may differ** |
| `litellm` | `TRELIX_LLM_LITELLM_MODEL` (falls back to `TRELIX_LLM_MODEL`) | `TRELIX_LLM_MODEL` | **may differ** |

**2. Matching is prefix-anchored, so namespaced ids do not resolve.** Lookup lower-cases both sides and walks a longest-prefix-first table, which handles version and date suffixes (`gpt-4o-2024-11-20` → `gpt-4o` → 128,000; `claude-sonnet-4-6` → 200,000). It does **not** strip a provider namespace, because the match is anchored at the start of the string:

| Model string | Resolves to |
|---|---|
| `gpt-4o-2024-11-20` | 128,000 |
| `claude-sonnet-4-6` | 200,000 |
| `gemini-2.5-pro` | 1,000,000 |
| `us.anthropic.claude-sonnet-4-6` | unrecognised |
| `anthropic.claude-3-5-sonnet-20240620-v1:0` | unrecognised |
| `bedrock/claude-3-5-sonnet` | unrecognised |

An unrecognised model logs a WARNING and falls back to a flat `12,000`-token budget — the window fraction is not applied. Auto-derivation therefore degrades to the v2.12.0 default rather than failing, but it also means Bedrock inference-profile ids and LiteLLM-prefixed model strings get no benefit from it unless `TRELIX_LLM_MODEL` is set to a bare, recognised model name.

---

## Vector Store Backends

Selected with `TRELIX_STORE_BACKEND` (`sqlite` default, `qdrant`, `lance`). They differ in how a *write* can fail, which is what the rest of this section is about — retrieval quality is identical, since all three store the same vectors.

| Backend | Upsert primitive | On a failed write |
|---|---|---|
| `sqlite` | `DELETE` + `INSERT` per chunk_id inside one transaction (sqlite-vec virtual tables reject `INSERT OR REPLACE`) | rolled back, then raised — nothing half-written |
| `qdrant` | server-side `upsert` by point id | raised by the client; with no delete step there is nothing to half-apply |
| `lance` | `checkout_latest` + `DELETE` + `add` — **two separate commits, no uniqueness constraint on chunk_id** | see below |

### LanceDB failure modes

Everything here is a consequence of one fact: LanceDB has no uniqueness constraint on `chunk_id`, so "replace" is a delete followed by an add, and an add whose delete did not happen is an **append**.

**A failed delete aborts the batch — and the index run.** `upsert_batch` logs at ERROR and re-raises instead of adding anyway. Adding anyway was measured to grow one chunk_id from 1 → 2 → 3 → 4 rows across three failed-delete upserts; every later `search()` then returns that chunk_id repeatedly, spending several of the `k` result slots on one chunk, and `count()` drifts above the SQLite `chunks` table. No subsequent upsert undoes that. Aborting instead leaves the previous single row in place (stale vector, still searchable).

The indexer treats that abort as a deliberate one rather than letting it escape as a bare traceback (this applies to any backend whose upsert raises, LanceDB is just the one that raises by design): the embedding phase stops, batches that had not started embedding are **not** embedded — a store rejecting one write is rarely done rejecting, and each batch is a paid API call — and the run fails with a `PartialIndexError` reporting how many batches failed, how many chunks got vectors, and the recovery path below. With a 50 ms embedding round trip and a store failing every write, a 40-batch run pays for 8 batches and skips 32 (3 runs, identical).

**A partial index stays partial until the next `trelix index`.** The chunk rows and the files' content hashes are committed in Phase 2, *before* embedding, and re-parsing alone will not find the damage: the incremental pre-filter skips those files on a matching hash, and even with `TRELIX_INCREMENTAL=false` the indexer only re-chunks symbols whose own signature+body hash changed. Until it is repaired, the chunks that missed their vectors stay unsearchable while `trelix stats` keeps counting them as chunks (it reads `COUNT(*) FROM chunks`, which is unaffected) — it does now report them separately as `Chunks missing vectors`. To recover, fix the cause and run `trelix index` again: it diffs the table's `stored_chunk_ids()` against the `chunks` table once per run and re-embeds exactly the chunks with no vector, so the batches that did land are not paid for twice. `trelix watch` will not do it — it only re-indexes files as they change, and never scans the store for holes.

**Duplicate rows are reported, not hidden.** Writes are serialised by a per-table in-process lock, and each write refreshes the handle first (a LanceDB table handle pins the version it opened at: measured on lancedb 0.33.0, a handle that has not refreshed sees `count() == 0` against a true 1, and its `DELETE` commits a new version while removing nothing). Neither mechanism reaches a **second OS process** on the same URI — the "multi-repo deployments sharing a vector store" case — so `upsert_batch` counts the rows it wrote and logs at ERROR when a chunk_id ends up with more than one. Those duplicates clear only on a later clean upsert of the same chunk_id, and the indexer embeds each chunk once per run, so in practice they survive the run. `Table.merge_insert`, LanceDB's own upsert primitive, is not a way out: 4 threads over 50 chunk_ids left 53–62 rows via `merge_insert` against 50–53 via delete+add, equally silently.

**Cost of the safety.** The refresh costs 0.21 ms on a 200k-row table (against 9.2 ms for one k=10 search) and the duplicate check one filtered `count_rows` (2.3 ms for 32 ids over 200k rows). The write lock reduces two concurrent writers to one — median 135 → 87 batches of 32 per second over 9 runs — which is below the embedder's ceiling unless the embedder is local and very fast.

---

## Environment Variables Reference

All variables trelix reads, with their defaults. Variables marked `(required)` have no default and must be set for the feature to work.

### Core

| Variable | Default | Description |
|---|---|---|
| `TRELIX_EMBEDDER_PROVIDER` | `local` | Embedding provider: `local`, `openai`, `azure`, `voyage`, `local-code`, `bge-code`, `nomic-code`, `bedrock-titan`, `bedrock-cohere` |
| `TRELIX_LLM_PROVIDER` | `openai` | LLM provider: `openai`, `azure`, `anthropic`, `bedrock`, `vertex`, `litellm` |
| `TRELIX_LLM_MODEL` | `gpt-4o` | LLM model name |
| `TRELIX_PARSE_WORKERS` | `4` | Parallel parse threads during indexing |
| `TRELIX_FILE_SUMMARIES_ENABLED` | `false` | Generate LLM file-level summaries (RAPTOR-style) at index time |
| `TRELIX_TELEMETRY_ENABLED` | `false` | Record every `retrieve()` call to the `query_telemetry` table |

### Embedding — OpenAI

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — (required) | OpenAI API key |
| `TRELIX_EMBEDDER_OPENAI_MODEL` | `text-embedding-3-large` | OpenAI embedding model |

### Embedding — Azure

| Variable | Default | Description |
|---|---|---|
| `AZURE_API_KEY` | — (required) | Azure OpenAI API key |
| `AZURE_ENDPOINT` | — (required) | Azure resource endpoint URL |
| `AZURE_API_VERSION` | `2025-04-01-preview` | Azure API version |
| `AZURE_EMBEDDINGS_MODEL` | `text-embedding-3-large` | Azure embeddings deployment name |
| `AZURE_CHAT_MODEL` | `gpt-4o` | Azure chat deployment name (for LLM) |

### Embedding — Voyage

| Variable | Default | Description |
|---|---|---|
| `VOYAGE_API_KEY` | — (required) | Voyage AI API key |
| `TRELIX_EMBEDDER_VOYAGE_MODEL` | `voyage-code-3` | Voyage model name |
| `TRELIX_EMBEDDER_VOYAGE_OUTPUT_DIMENSIONS` | — (none) | Matryoshka output dim: 256, 512, 1024, or 2048 |

### Embedding — AWS Bedrock

| Variable | Default | Description |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | — | AWS access key (or use `AWS_PROFILE`) |
| `AWS_SECRET_ACCESS_KEY` | — | AWS secret key (or use `AWS_PROFILE`) |
| `AWS_REGION` | `us-east-1` | AWS region |
| `AWS_PROFILE` | — | AWS named profile (alternative to key/secret) |
| `TRELIX_EMBEDDER_BEDROCK_TITAN_DIMENSIONS` | `1024` | Titan output dims: 256, 512, or 1024 |

### Embedding — Indexing Performance

| Variable | Default | Description |
|---|---|---|
| `TRELIX_EMBEDDER_EMBED_MAX_TOKENS_PER_BATCH` | `100000` | Max tokens per embedding batch |
| `TRELIX_EMBEDDER_TPM_LIMIT` | `0` | Tokens-per-minute rate limit (0 = unlimited) |

### LLM — Provider-specific

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — (required for anthropic) | Anthropic API key |
| `TRELIX_LLM_THINKING_ENABLED` | `false` | Anthropic extended thinking on the synthesizer's calls. Other backends accept and ignore it |
| `TRELIX_LLM_THINKING_BUDGET_TOKENS` | `4096` | `thinking.budget_tokens` sent to the Anthropic Messages API. Bills as output tokens |
| `TRELIX_LLM_BEDROCK_PRIMARY_MODEL` | `us.anthropic.claude-sonnet-4-6` | Bedrock primary inference profile |
| `TRELIX_LLM_BEDROCK_FALLBACK_MODEL` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock fallback profile |
| `GOOGLE_CLOUD_PROJECT` | — | GCP project (Vertex AI) |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | GCP region |
| `GOOGLE_API_KEY` | — | Google AI Studio API key (alternative to project) |
| `TRELIX_LLM_LITELLM_MODEL` | — | LiteLLM model string (e.g. `bedrock/claude-3-5-sonnet`) |

### Store / Vector DB

Write-failure behaviour differs per backend — see [Vector Store Backends](#vector-store-backends) before choosing `lance`.

| Variable | Default | Description |
|---|---|---|
| `TRELIX_STORE_BACKEND` | `sqlite` | Storage backend: `sqlite`, `qdrant`, `lance` |
| `TRELIX_STORE_HNSW` | `true` | Enable HNSW O(log n) index |
| `TRELIX_STORE_HNSW_M` | `16` | HNSW M parameter (graph connectivity) |
| `TRELIX_STORE_HNSW_EF_SEARCH` | `50` | HNSW ef_search (recall vs speed) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | — | Qdrant API key |
| `QDRANT_COLLECTION` | `trelix` | Qdrant collection name |
| `LANCE_URI` | `.trelix/lance` | LanceDB URI |
| `LANCE_TABLE` | `chunks` | LanceDB table name |

### Retrieval Tuning

| Variable | Default | Description |
|---|---|---|
| `TRELIX_RETRIEVAL_RERANK_PROVIDER` | `cohere` | Reranker: `cohere`, `cross_encoder`, `plaid` |
| `TRELIX_RETRIEVAL_RERANK_TOP_N` | `15` | Candidates passed to reranker |
| `TRELIX_RETRIEVAL_PLAID_MODEL` | `colbert-ir/colbertv2.0` | PLAID (ColBERT) model for late-interaction reranking |
| `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET` | `12000` | Max tokens in assembled context. An explicit int is a fixed budget; `null` (also `none`/`auto`/`~`/empty) auto-derives it from the model's context window — see [Model-aware context budgets](#model-aware-context-budgets) |
| `TRELIX_RETRIEVAL_CONTEXT_WINDOW_FRACTION` | `0.5` | Fraction of the model window to spend on context when the budget is `null` (range 0.1–0.9). Ignored with an explicit int budget |
| `TRELIX_RETRIEVAL_SCALE_TOP_K_TO_BUDGET` | `false` | Also scale `top_k_vector`/`rerank_top_n` by `effective_budget / 12000`. Needs the budget to be `null`; raises per-query cost |
| `TRELIX_RETRIEVAL_SYNTHESIS_MAX_TOKENS` | `12000` | Max tokens in LLM synthesis response |
| `TRELIX_RETRIEVAL_GRAPH_IMPORT_MAX_EXTRA` | `3` | **Inert.** `RetrievalConfig.graph_import_max_extra` is declared but read nowhere in `src/`. Import expansion takes its cap from the per-intent `RetrievalStrategy.import_max_extra` in `retrieval/planner/models.py` (0 for intents that skip import expansion, up to 30 for `blast_radius`) — that table is, per its own comment, "the ONLY place that controls retrieval behaviour per intent" |
| `TRELIX_RETRIEVAL_QUERY_CACHE_SIZE` | `256` | LRU cache for `embed_query()` (0 = disabled) |
| `TRELIX_RETRIEVAL_PLAN_CACHE_SIZE` | `128` | LRU cache for `QueryPlan` LLM calls (0 = disabled) |

### Retrieval — Optional Features (off by default)

| Variable | Default | Description |
|---|---|---|
| `TRELIX_RETRIEVAL_GRAPH_RAG` | `true` | GraphRAG map-reduce synthesis (auto-activates above token/result thresholds) |
| `TRELIX_RETRIEVAL_SPARSE` | `false` | Enable SPLADE-Code sparse+dense hybrid retrieval leg |
| `TRELIX_RETRIEVAL_SPARSE_TOP_K` | `20` | Top-K results from sparse retrieval |
| `TRELIX_RETRIEVAL_HYDE_FALLBACK` | `false` | HyDE: synthesize a code snippet before embedding the query |
| `TRELIX_RETRIEVAL_MULTI_QUERY` | `false` | Generate N query variants and merge results |
| `TRELIX_RETRIEVAL_MULTI_QUERY_COUNT` | `2` | Number of query variants (1–4) |
| `TRELIX_RETRIEVAL_FLARE` | `false` | FLARE-style confidence-gated re-retrieval |
| `TRELIX_RETRIEVAL_FLARE_MAX_RETRIES` | `1` | Max FLARE re-retrieval passes (1–3) |
| `TRELIX_RETRIEVAL_PAGERANK_BOOST` | `false` | Boost results by PageRank symbol importance |
| `TRELIX_RETRIEVAL_PAGERANK_BOOST_FACTOR` | `1.3` | PageRank score multiplier (1.0–3.0) |
| `TRELIX_RETRIEVAL_DECLARATION_BOOST` | `false` | Boost BM25 matches on a symbol's `name`/`qualified_name` over incidental `docstring`/`body` mentions |
| `TRELIX_RETRIEVAL_DECLARATION_BOOST_WEIGHT` | `1.0` | Declaration-boost multiplier (1.0–10.0) |
| `TRELIX_RETRIEVAL_AGENTIC` | `false` | Agentic ReAct multi-turn retrieve+observe+synthesize loop |
| `TRELIX_RETRIEVAL_AGENT_MAX_TURNS` | `8` | Max ReAct turns (1–20) |
| `TRELIX_RETRIEVAL_AGENT_TOKEN_BUDGET` | `6000` | Token budget per ReAct agent session |
| `TRELIX_RETRIEVAL_FILE_SUMMARY_LEG` | `false` | Enable file-summary retrieval leg (requires `TRELIX_FILE_SUMMARIES_ENABLED=true` at index time) |
| `TRELIX_RETRIEVAL_FILE_SUMMARY_TOP_K` | `5` | Top-K file summaries to retrieve |
| `TRELIX_RETRIEVAL_SUB_CHUNK` | `false` | Sub-chunk (block/statement) search leg (MGS3) |
| `TRELIX_RETRIEVAL_SUB_CHUNK_TOP_K` | `10` | Top-K sub-chunk results |
| `TRELIX_RETRIEVAL_COMPRESSION` | `false` | SeleCom query-conditioned compression of oversized bodies (provider-independent — no LLM call) |
| `TRELIX_RETRIEVAL_COMPRESSION_PROVIDER` | `extractive` | Compression backend; `extractive` is the only accepted value |
| `TRELIX_RETRIEVAL_COMPRESSION_RATIO` | `0.45` | Fallback keep-ratio for unknown intents only (0.1–1.0) |
| `TRELIX_RETRIEVAL_COMPRESSION_RATIO_<INTENT>` | (per-intent defaults) | Per-intent keep-ratio override, e.g. `..._BLAST_RADIUS=0.5` |
| `TRELIX_RETRIEVAL_COMPRESSION_MIN_TOKENS` | `120` | Bodies below this token count are never compressed |
| `TRELIX_FEDERATION_ENABLED` | `false` | **Inert.** Declared but read nowhere in `src/` — run `trelix search-all` to federate; it does not consult this |
| `TRELIX_FEDERATION_MAX_WORKERS` | `4` | **Inert.** Declared but read nowhere in `src/` — `search-all` calls `FederatedRetriever(registry)`, so the pool is always the constructor default of 4 |

### Retrieval — File-type Weighting

| Variable | Default | Description |
|---|---|---|
| `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING` | `true` | Apply per-language RRF score multipliers |
| `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS` | (JSON dict) | Full override as JSON, e.g. `{"markdown":0.1,"yaml":0.6}` |
| `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_<LANG>` | (per-language defaults) | Single-language override, e.g. `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN=0.1` |

Default weights: source code `1.0`, config/data `0.5`, HTML/CSS `0.4`, Markdown `0.3`, unknown `0.8`.

### Retrieval — Per-leg Weighting

| Variable | Default | Description |
|---|---|---|
| `TRELIX_RETRIEVAL_LEG_WEIGHT_<LEG>` | `1.0` per leg | Per-leg RRF score multiplier, applied during fusion (before summing, not after like file-type weights). `<LEG>` is one of `VECTOR`, `BM25`, `GREP`, `SUMMARY`, `SUB_CHUNK`, `SPARSE`, e.g. `TRELIX_RETRIEVAL_LEG_WEIGHT_BM25=0.7`. |

All-`1.0` (the default) is a no-op — byte-for-byte identical to unweighted fusion.

### Cohere Reranker

| Variable | Default | Description |
|---|---|---|
| `COHERE_API_KEY` | — (required for cohere rerank) | Cohere API key |
| `COHERE_ENDPOINT` | — | Azure-deployed Cohere endpoint URL |
| `COHERE_MODEL_RERANK` | `Cohere-rerank-v4.0-pro` | Cohere rerank model |

### Contextual Chunking

| Variable | Default | Description |
|---|---|---|
| `TRELIX_CHUNKER_CONTEXTUAL` | `false` | Generate LLM-based context summary per chunk at index time (67% better recall, costs tokens) |
| `TRELIX_CHUNKER_CONTEXTUAL_MODEL` | `gpt-4o-mini` | Model for contextual chunk summaries |
| `TRELIX_CHUNKER_CONTEXTUAL_MAX_TOKENS` | `100` | Max tokens per chunk summary |

### Multi-granularity Chunking (MGS3)

| Variable | Default | Description |
|---|---|---|
| `TRELIX_CHUNKER_MULTI_GRANULARITY` | `false` | Index block- and statement-level sub-symbols in addition to functions |
| `TRELIX_CHUNKER_GRANULARITY_LEVELS` | `["block","statement"]` | Granularity levels to index |

### Parser

| Variable | Default | Description |
|---|---|---|
| `TRELIX_PARSER_DATAFLOW` | `false` | Enable def-use chain data-flow analysis |
| `TRELIX_PARSER_TAINT` | `false` | **Inert.** `ParserConfig.taint_enabled` is declared but read nowhere in `src/`. Taint analysis runs only when you invoke `trelix taint`, which does not consult it — so this switch neither enables nor disables anything (`pip install trelix[taint]` is still required for the command itself) |

### Sparse Embeddings (SPLADE-Code)

| Variable | Default | Description |
|---|---|---|
| `TRELIX_SPARSE_MODEL` | `naver/splade-v3-distilbert` | SPLADE model. **Licensed CC BY-NC-SA-4.0 (non-commercial)** — trelix is MIT and does not redistribute the weights, but they are downloaded at runtime when the sparse leg is enabled, so a commercial deployment should set a permissively-licensed model here or leave `TRELIX_RETRIEVAL_SPARSE` off. Must be a **BERT-family MaskedLM** checkpoint: `embedder/sparse.py` loads it with `AutoModelForMaskedLM`, so the SPLADE-Code releases (`naver/splade-code-8B`, `naver/splade-code-06B`) will **not** load — they are `model_type=qwen3`, which is causal-LM. The previous default `naver-splab/splade-code-distil` did not exist on the Hub at all |
| `TRELIX_SPARSE_TOP_K_TOKENS` | `128` | Number of top tokens in sparse vector (16–512) |
| `TRELIX_SPARSE_BATCH_SIZE` | `16` | Batch size for sparse encoding |

---

## Quick Recipes

### Fastest local setup

```env
TRELIX_EMBEDDER_PROVIDER=local
```

No variables needed beyond `TRELIX_EMBEDDER_PROVIDER` (it is the default).

### Best retrieval quality (API-based)

```env
TRELIX_EMBEDDER_PROVIDER=voyage
VOYAGE_API_KEY=pa-...
TRELIX_RETRIEVAL_RERANK_PROVIDER=cohere
COHERE_API_KEY=...
```

### Best retrieval quality (fully offline)

```env
TRELIX_EMBEDDER_PROVIDER=local-code
TRELIX_RETRIEVAL_RERANK_PROVIDER=cross_encoder
```

### AWS-only deployment

```env
TRELIX_EMBEDDER_PROVIDER=bedrock-cohere
TRELIX_LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
AWS_PROFILE=my-profile
```

### Azure enterprise

```env
TRELIX_EMBEDDER_PROVIDER=azure
TRELIX_LLM_PROVIDER=azure
AZURE_API_KEY=...
AZURE_ENDPOINT=https://my-resource.openai.azure.com/
AZURE_EMBEDDINGS_MODEL=text-embedding-3-large
AZURE_CHAT_MODEL=gpt-4o
```
