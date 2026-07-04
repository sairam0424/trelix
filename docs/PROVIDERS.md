# trelix v2.4.0 — Providers Reference

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
| `TRELIX_LLM_BEDROCK_PRIMARY_MODEL` | `us.anthropic.claude-sonnet-4-6` | Bedrock primary inference profile |
| `TRELIX_LLM_BEDROCK_FALLBACK_MODEL` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock fallback profile |
| `GOOGLE_CLOUD_PROJECT` | — | GCP project (Vertex AI) |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | GCP region |
| `GOOGLE_API_KEY` | — | Google AI Studio API key (alternative to project) |
| `TRELIX_LLM_LITELLM_MODEL` | — | LiteLLM model string (e.g. `bedrock/claude-3-5-sonnet`) |

### Store / Vector DB

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
| `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET` | `12000` | Max tokens in assembled context window |
| `TRELIX_RETRIEVAL_SYNTHESIS_MAX_TOKENS` | `12000` | Max tokens in LLM synthesis response |
| `TRELIX_RETRIEVAL_GRAPH_IMPORT_MAX_EXTRA` | `3` | Extra symbols added via graph import expansion |
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
| `TRELIX_RETRIEVAL_AGENTIC` | `false` | Agentic ReAct multi-turn retrieve+observe+synthesize loop |
| `TRELIX_RETRIEVAL_AGENT_MAX_TURNS` | `8` | Max ReAct turns (1–20) |
| `TRELIX_RETRIEVAL_AGENT_TOKEN_BUDGET` | `6000` | Token budget per ReAct agent session |
| `TRELIX_RETRIEVAL_FILE_SUMMARY_LEG` | `false` | Enable file-summary retrieval leg (requires `TRELIX_FILE_SUMMARIES_ENABLED=true` at index time) |
| `TRELIX_RETRIEVAL_FILE_SUMMARY_TOP_K` | `5` | Top-K file summaries to retrieve |
| `TRELIX_RETRIEVAL_SUB_CHUNK` | `false` | Sub-chunk (block/statement) search leg (MGS3) |
| `TRELIX_RETRIEVAL_SUB_CHUNK_TOP_K` | `10` | Top-K sub-chunk results |
| `TRELIX_FEDERATION_ENABLED` | `false` | Multi-repo federated search |
| `TRELIX_FEDERATION_MAX_WORKERS` | `4` | Parallel workers for federated search (1–16) |

### Retrieval — File-type Weighting

| Variable | Default | Description |
|---|---|---|
| `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTING` | `true` | Apply per-language RRF score multipliers |
| `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHTS` | (JSON dict) | Full override as JSON, e.g. `{"markdown":0.1,"yaml":0.6}` |
| `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_<LANG>` | (per-language defaults) | Single-language override, e.g. `TRELIX_RETRIEVAL_FILE_TYPE_WEIGHT_MARKDOWN=0.1` |

Default weights: source code `1.0`, config/data `0.5`, HTML/CSS `0.4`, Markdown `0.3`, unknown `0.8`.

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
| `TRELIX_PARSER_TAINT` | `false` | Enable taint analysis (requires `pip install trelix[taint]`) |

### Sparse Embeddings (SPLADE-Code)

| Variable | Default | Description |
|---|---|---|
| `TRELIX_SPARSE_MODEL` | `naver-splab/splade-code-distil` | SPLADE model |
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
