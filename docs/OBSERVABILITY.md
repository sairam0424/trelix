# Trelix Observability — OpenTelemetry Tracing & Metrics

trelix can emit [OpenTelemetry](https://opentelemetry.io/) spans for every stage of the retrieval pipeline, and a small number of **metric counters** for embedding cost. Both are fully opt-in behind the same switch — disabled by default, zero import cost and zero behavior change when off.

The two signals do **not** have the same coverage. Tracing spans the whole retrieval pipeline; metrics cover embedding only. [What gets measured](#what-gets-measured-metrics) is explicit about where that line falls, because a partially-instrumented metrics surface that reads as complete is how you end up billing against a number that omits most of your spend.

---

## Enabling

```bash
pip install "trelix[otel]"
export TRELIX_OTEL_ENABLED=true
export OTEL_SERVICE_NAME=my-service          # optional, default "trelix"
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318  # optional; base or /v1/traces
```

These three switches cover **both** signals — spans and the embedding counters. A bare base
endpoint is the simplest thing that works for both; a `…/v1/traces` value also works, because
trelix swaps the suffix for the metrics signal (see [Exporting metrics](#exporting-metrics)).

See [CONFIGURATION.md](CONFIGURATION.md#observability-opentelemetry) for the full env var reference.

If `OTEL_EXPORTER_OTLP_ENDPOINT` is unset, spans are still created (visible to any exporter/processor a host application configures on its own `TracerProvider` before trelix runs) but have nowhere to export to on their own.

**Do not set a `TracerProvider` yourself and also let trelix install one** — trelix only installs its own provider when it detects the default `ProxyTracerProvider` (i.e. nothing has configured OTel yet). If your application already calls `trace.set_tracer_provider(...)` before constructing a `Retriever`, trelix reuses it and never overwrites it.

---

## What gets traced

One span per retrieval leg, using the official [`gen_ai.*` semantic conventions](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md#retrievals) via [`opentelemetry-util-genai`](https://github.com/open-telemetry/opentelemetry-python-genai)'s `TelemetryHandler.retrieval()`:

| Leg | `gen_ai.data_source.id` | Attributes set |
|---|---|---|
| Vector (dense ANN) | `vector` | `query_text`, `top_k`, `trelix.leg.result_count` |
| BM25 (FTS5) | `bm25` | same |
| Grep | `grep` | same |
| Sparse (SPLADE-Code, 7th leg) | `sparse` | same |
| Sub-chunk (MGS3, 6th leg) | `sub_chunk` | same |
| File-summary (RAPTOR-style, 5th leg) | `file_summary` | same |

Plus trelix-specific pipeline-stage spans (not `gen_ai.*` — these are trelix concepts, not GenAI operations), namespaced under `trelix.*`:

| Span name | Wraps |
|---|---|
| `trelix.retrieve` | The whole `Retriever.retrieve()` call (root span) |
| `trelix.planner` | Query planning (LLM intent classification, or `default_plan()`) |
| `trelix.fusion` | Reciprocal Rank Fusion across all leg result lists |
| `trelix.expansion` | Call-graph, import-graph, type-edge, and CodeGraph-BFS expansion |
| `trelix.rerank` | Cross-encoder/Cohere/PLAID/XTR reranking (only when `rerank_enabled` and not skipped by strategy) |
| `trelix.pagerank_boost` | PageRank centrality boost (only actually does work when `TRELIX_RETRIEVAL_PAGERANK_BOOST=true`) |
| `trelix.assembly` | Final context assembly within the token budget |

---

## What gets measured (metrics)

Metrics are newer than tracing and their scope is **much narrower**: they cover **embedding
only**. Read the tables below as exhaustive, not illustrative.

There is no separate switch. `TRELIX_OTEL_ENABLED`, `OTEL_SERVICE_NAME`, and
`OTEL_EXPORTER_OTLP_ENDPOINT` drive both signals off the same three config fields. One
wrinkle worth knowing: the embedder is not handed a `RetrievalConfig` (its construction
sites pass only `config.embedder`, which carries no `otel_*` fields), so the counters resolve
the flag through `RetrievalConfig` themselves, memoized once per process. That is deliberate
— it means `TRELIX_OTEL_ENABLED` set in `.env` is honoured on the metrics path exactly as it
is on the span path, instead of the two silently disagreeing.

### The four counters

All four are **monotonic counters** — use a `rate()`/`increase()` function in your backend;
they never go down.

| Instrument | Unit | Counts |
|---|---|---|
| `trelix.embedder.requests` | `{request}` | Provider calls — one per API request / model invocation, **not** one per batch of texts |
| `trelix.embedder.texts` | `{text}` | Texts (chunks at index time, queries at retrieval time) submitted for embedding |
| `trelix.embedder.characters` | `{character}` | Characters submitted — the volume proxy for providers that report no token usage |
| `trelix.embedder.tokens` | `{token}` | Provider-**reported** tokens — the billed quantity. Absent for providers that report none |

Each is attributed with `trelix.embedder.provider` and `gen_ai.request.model`. The provider
value is trelix's own selector string (`openai`, `bedrock-titan`, `local-code`, …) and is
deliberately **not** a `gen_ai.provider.name` enum member, so it keeps a `trelix.*` name
rather than pretending to conform; the model attribute uses the conventional key so these
counters join to the `gen_ai.*` spans above.

The counter names are trelix's own because the GenAI metric conventions cover chat token
usage, not embedding volume — there was nothing to borrow.

### Per-provider coverage — check your provider before building a cost panel

`requests`/`texts`/`characters` are recorded for every provider in the table; the `tokens`
series only exists where the provider actually reports a count.

| `trelix.embedder.provider` | Counted | `tokens` series | Token source |
|---|---|---|---|
| `openai` | yes | **yes** | `response.usage.total_tokens` |
| `azure` | yes | **yes** | `response.usage.total_tokens` |
| `voyage` | yes | **yes** | `response.total_tokens` |
| `bedrock-titan` | yes | **yes** | `inputTextTokenCount` on the response body |
| `bedrock-cohere` | yes | **no** | Bedrock's Cohere path reports none |
| `local` | yes | no | nothing billed |
| `local-code` | yes | no | nothing billed |
| `bge-code` | **no — not counted at all** | — | `BGECodeEmbedder` lives outside `embedder/base.py` and is uninstrumented |
| `nomic-code` | **no — not counted at all** | — | `NomicCodeEmbedder`, same reason |

Those last two rows are the ones to notice. They are 2 of the 9 values
`EmbedderConfig.provider` accepts, and `bge-code` is trelix's flagship v2.0 embedder, so a
bge-code deployment emits **no embedding counters at all** — a provider-filtered dashboard
will look broken rather than look free. Both run local models, so nothing is being billed
unmeasured; what is lost is the volume signal, not a cost signal.

The SPLADE sparse embedder (`embedder/sparse.py`) is also uninstrumented. It is a separate
subsystem rather than a `provider` value — sparse retrieval runs alongside a dense embedder,
not instead of one — so its absence does not show up as a missing series in the table above.

### Semantics worth knowing before you alert on these

- **Counted after the response lands.** A retried-then-failed attempt does not inflate
  `requests`, so the series measures successful provider calls, not attempts.
- **`tokens` is never estimated.** Where a provider reports nothing, the series is left
  untouched rather than filled with a `characters / 4` guess — a cost series that is
  silently a guess is worse than one that is visibly absent. Use `characters` there.
- **Cache hits do not increment anything.** `CachingEmbedder` short-circuits before the
  provider, so the counters track real spend rather than logical demand. A falling
  `requests` rate at constant query volume is the cache working.
- **Failure is loud, once.** If `TRELIX_OTEL_ENABLED` is set but OpenTelemetry metrics
  cannot initialise, trelix logs a **WARNING** (not a debug line, unlike the span helpers)
  naming the cause and saying counters will not be recorded. This is deliberate: an empty
  cost dashboard otherwise reads as "we spent nothing".

### Not instrumented

There are **no** metrics for any of the following. Each is a real cost or signal this
release does not count:

- **LLM tokens** — synthesis, query planning, contextual chunking, and file summarisation
  all call an LLM and none increments a counter. On a hosted model this is normally the
  largest bill trelix generates. `ChatResponse` carries `input_tokens` / `output_tokens` /
  `cache_read_tokens` / `cache_write_tokens` per call, but nothing aggregates them.
- **Retrieval latency or throughput** — no histogram for `retrieve()`, for any leg, or for
  fusion/rerank/assembly. That exists only as spans: per-query and sampled, not aggregated.
- **Reranker cost** — a Cohere or hosted cross-encoder rerank is a paid per-query call and
  is uncounted.
- **Cache effectiveness** — no hit/miss counters. `FederatedRetriever.cache_stats()`
  returns them on demand; nothing exports them.
- **Indexing** — no counters for files walked, symbols extracted, or chunks written.
- **HTTP API and MCP surfaces** — no request/error/duration counters. The HTTP layer has
  spans (v2.10.0) and audit rows (v3.0.0), but no metrics.

**Blunt consequence:** you cannot build a total-cost-of-trelix dashboard from these
counters. You can build an *embedding*-cost dashboard, and only for the providers marked
counted above. For LLM spend, read your provider's billing surface or the `gen_ai.*` spans.

### Exporting metrics

No extra configuration. The same `TRELIX_OTEL_ENABLED` switch and the same
`OTEL_EXPORTER_OTLP_ENDPOINT` cover both signals, and trelix maps the endpoint onto the
metrics path for you: OTLP/HTTP uses `/v1/traces` for spans and `/v1/metrics` for metrics,
so a configured `…:4318/v1/traces` has its suffix **swapped**, not reused. Posting metric
payloads to the traces route would be rejected by the collector, so this is handled in code
rather than left as a footgun — a bare `…:4318` works too.

As with tracing, trelix only installs its own `MeterProvider` when nothing has configured
OTel metrics yet (it checks for the API's unset `ProxyMeterProvider` placeholder). An
explicit `NoOpMeterProvider` is treated as a deliberate host choice and left alone.

---

## Stability caveat — read before building dashboards

The `gen_ai.*` semantic conventions this integration uses are officially part of OpenTelemetry, but marked **`Status: Development`**, not yet **`Stable`**, as of this writing. That means:

- Attribute names (`gen_ai.operation.name`, `gen_ai.data_source.id`, `gen_ai.request.top_k`, etc.) may still change in a future OTel semantic-conventions release.
- `opentelemetry-util-genai` itself ships pre-1.0 (`1.0b0` at time of writing) — its Python API surface could shift.

trelix deliberately adopted the official conventions now (rather than defining its own `trelix.retrieval.*` attribute set) to avoid a painful rename migration later, but this means dashboards/alerts built against `gen_ai.*` attributes should be revisited if you see them break after an `opentelemetry-util-genai` upgrade.

The `trelix.*`-namespaced pipeline-stage spans (fusion/expansion/rerank/etc.) are trelix's own naming and are not subject to this caveat — they won't change without a trelix version bump and a CHANGELOG entry.

---

## Relationship to existing (non-OTel) telemetry

This is additive — it does not replace either of trelix's existing telemetry mechanisms:

- **`TelemetryWriter`** (`TRELIX_TELEMETRY_ENABLED=true`) — writes one row per `retrieve()` call to the `query_telemetry` SQLite table (query text, intent, latency, result count, expansion columns) in the index DB. The only reader is the `trelix telemetry` CLI report. `trelix eval` does **not** read this table — it re-runs the queries in a golden JSONL file live through `Retriever` and computes nDCG@10 / recall@10 / MRR from those fresh results, so telemetry can be off and `eval` still works. (Earlier revisions of this doc claimed `eval` consumed the telemetry table; that was never true.)
- **Debug trace JSON** (always on unless commented out in `retriever.py`) — writes a structured `.trelix/debug/<ts>_<slug>.json` file per query with plan/legs/fusion/expansion/rerank/assembly data.

Use OTel tracing when you want to export spans to an existing observability stack (Jaeger, Grafana Tempo, Honeycomb, Datadog, etc. — anything that accepts OTLP). Use the other two when you want local-file or in-DB analysis without standing up a collector.

---

## Relationship to audit logging

The v3.0.0 audit trail (`TRELIX_AUDIT_ENABLED=true`, documented in [AUDIT.md](AUDIT.md)) is **not** a telemetry mechanism and does not replace any of the above. The distinction matters when deciding where to look during an incident:

| | Audit log | Query telemetry | OTel tracing |
|---|---|---|---|
| Question | Who did what, when, and was it allowed? | How did retrieval perform over time? | Where did the time go inside one query? |
| Unit | one row per **HTTP request** | one row per **`retrieve()` call** | one span per pipeline stage/leg |
| Records caller identity | **yes** (`principal` — `sub@iss` or `static-token`) | no | no |
| Records query text | no (deliberately) | **yes**, verbatim | as span attributes |
| Stored in | a separate `audit.db` that survives re-indexing | `query_telemetry` in the **disposable** index DB | your OTLP backend |
| Integrity | hash-chained, append-only, `trelix audit verify` | none — plain rows, deleted with the index | none — sampled, ephemeral |
| Covers | the HTTP API only (not MCP, not the agent loop, not the CLI) | every `retrieve()` regardless of caller | every `retrieve()` regardless of caller |

Two practical consequences:

- **Telemetry is not an audit trail.** It has no identity column, it lives in a DB that gets deleted on every re-index, and nothing detects modification of its rows. Do not use it for compliance questions.
- **The audit log is not a performance tool.** It records one coarse `duration_ms` per request and nothing about legs, fusion or reranking.

They do join in one place: when OTel tracing is enabled, each audit row carries the `trace_id` of the request that produced it, so a suspicious audit entry can be opened as a trace in your tracing backend.

---

## Cross-thread span nesting

`_retrieve_standard()`'s parallel sub-query execution runs inside a `ThreadPoolExecutor`. OpenTelemetry's context propagation is `contextvars`-based and does **not** automatically cross a thread-pool boundary — without explicit handling, each worker's leg spans would start as new, unparented traces instead of nesting under the query's root span.

trelix handles this internally (`with_current_context()` in `src/trelix/retrieval/otel_tracing.py`) — no action needed by callers. If you're instrumenting your own code that calls into `Retriever` from a thread pool, be aware of the same caveat for your own spans.
