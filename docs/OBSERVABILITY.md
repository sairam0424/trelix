# Trelix Observability — OpenTelemetry Tracing

trelix can emit [OpenTelemetry](https://opentelemetry.io/) spans for every stage of the retrieval pipeline. This is fully opt-in — disabled by default, zero import cost and zero behavior change when off.

---

## Enabling

```bash
pip install "trelix[otel]"
export TRELIX_OTEL_ENABLED=true
export OTEL_SERVICE_NAME=my-service          # optional, default "trelix"
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces  # optional
```

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

## Stability caveat — read before building dashboards

The `gen_ai.*` semantic conventions this integration uses are officially part of OpenTelemetry, but marked **`Status: Development`**, not yet **`Stable`**, as of this writing. That means:

- Attribute names (`gen_ai.operation.name`, `gen_ai.data_source.id`, `gen_ai.request.top_k`, etc.) may still change in a future OTel semantic-conventions release.
- `opentelemetry-util-genai` itself ships pre-1.0 (`1.0b0` at time of writing) — its Python API surface could shift.

trelix deliberately adopted the official conventions now (rather than defining its own `trelix.retrieval.*` attribute set) to avoid a painful rename migration later, but this means dashboards/alerts built against `gen_ai.*` attributes should be revisited if you see them break after an `opentelemetry-util-genai` upgrade.

The `trelix.*`-namespaced pipeline-stage spans (fusion/expansion/rerank/etc.) are trelix's own naming and are not subject to this caveat — they won't change without a trelix version bump and a CHANGELOG entry.

---

## Relationship to existing (non-OTel) telemetry

This is additive — it does not replace either of trelix's existing telemetry mechanisms:

- **`TelemetryWriter`** (`TRELIX_TELEMETRY_ENABLED=true`) — writes one row per `retrieve()` call to the `query_telemetry` SQLite table. Used by `trelix eval` and for offline analysis.
- **Debug trace JSON** (always on unless commented out in `retriever.py`) — writes a structured `.trelix/debug/<ts>_<slug>.json` file per query with plan/legs/fusion/expansion/rerank/assembly data.

Use OTel tracing when you want to export spans to an existing observability stack (Jaeger, Grafana Tempo, Honeycomb, Datadog, etc. — anything that accepts OTLP). Use the other two when you want local-file or in-DB analysis without standing up a collector.

---

## Cross-thread span nesting

`_retrieve_standard()`'s parallel sub-query execution runs inside a `ThreadPoolExecutor`. OpenTelemetry's context propagation is `contextvars`-based and does **not** automatically cross a thread-pool boundary — without explicit handling, each worker's leg spans would start as new, unparented traces instead of nesting under the query's root span.

trelix handles this internally (`with_current_context()` in `src/trelix/retrieval/otel_tracing.py`) — no action needed by callers. If you're instrumenting your own code that calls into `Retriever` from a thread pool, be aware of the same caveat for your own spans.
