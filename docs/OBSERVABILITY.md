# Trelix Observability — OpenTelemetry Tracing & Metrics

trelix can emit [OpenTelemetry](https://opentelemetry.io/) spans for every stage of the retrieval pipeline and for every LLM call, and a small number of **metric counters** for embedding cost. Both are fully opt-in behind the same switch — disabled by default, zero import cost and zero behavior change when off.

The two signals do **not** have the same coverage. Tracing spans the whole retrieval pipeline and every LLM call ([LLM chat spans](#llm-chat-spans)); trelix's own counters cover embedding only (the `opentelemetry-util-genai` library trelix builds its spans with also records a GenAI duration histogram per span and a token-usage histogram per chat span, see [Not instrumented](#not-instrumented)). [What gets measured](#what-gets-measured-metrics) is explicit about where that line falls, because a partially-instrumented metrics surface that reads as complete is how you end up billing against a number that omits most of your spend.

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

One span per retrieval leg, using the official [`gen_ai.*` semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/cb10b70c15c099ccab144e8316d934c9699da0fd/docs/gen-ai/gen-ai-spans.md#retrievals) via [`opentelemetry-util-genai`](https://github.com/open-telemetry/opentelemetry-python-genai/tree/opentelemetry-util-genai%3D%3D1.2b0/util/opentelemetry-util-genai)'s `TelemetryHandler.retrieval()`. The GenAI conventions moved out of the core `semantic-conventions` repository in 2026: its newest release, v1.44.0, ships the `docs/gen-ai/` pages as "Moved" stubs and marks every `gen_ai.*` registry row deprecated, and the new home, `open-telemetry/semantic-conventions-genai`, has no tags or releases, so the link above pins a commit. The library's 1.x line is developed and tagged in `opentelemetry-python-genai` (its PyPI metadata still names `opentelemetry-python-contrib`, whose copy stopped at `0.5b0.dev`); [the spike report](reports/otel-genai-semconv-spike-2026-10-07.md) records what was verified and how.

| Leg | `gen_ai.data_source.id` | Attributes set |
|---|---|---|
| Vector (dense ANN) | `vector` | `query_text` (content capture only, see below), `top_k`, `trelix.leg.result_count` |
| BM25 (FTS5) | `bm25` | same |
| Grep | `grep` | same |
| Sparse (SPLADE-Code, 7th leg) | `sparse` | same |
| Sub-chunk (MGS3, 6th leg) | `sub_chunk` | same |
| File-summary (RAPTOR-style, 5th leg) | `file_summary` | same |

`query_text` reaches a leg span only when **both** content switches are on (see [Content capture](#content-capture)): trelix hands it to the library only when `TRELIX_OTEL_CAPTURE_CONTENT=true` (default `false`), and `opentelemetry-util-genai` writes `gen_ai.retrieval.query.text` only when `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` is `SPAN_ONLY` or `SPAN_AND_EVENT`; in 1.0b0, 1.1b0 and 1.2b0 the default (`NO_CONTENT`, also what an unset or invalid value becomes) drops the text, so by default no query text reaches a span (measured on `opentelemetry-util-genai` 1.0b0 and 1.2b0; the 1.1b0 behaviour read in its extracted wheel). When the library reads its variable for the span attribute differs by version: 1.2b0 reads it once, when the handler is built on the first span, so a value exported into a running process has no effect until restart; 1.0b0 and 1.1b0 re-read it on every span, so exporting `SPAN_ONLY` into a running process with the trelix flag on puts the query text on every leg span from then on. See the [spike report](reports/otel-genai-semconv-spike-2026-10-07.md).

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

## LLM chat spans

With `TRELIX_OTEL_ENABLED=true`, `build_chat_client()` returns the provider backend wrapped in a `TracedChatClient` (`src/trelix/llm/otel.py`), and every `complete()`, `stream()` and `tool_call()` emits one GenAI inference span named `chat {request model}` (kind `CLIENT`) through the same `TelemetryHandler` as the retrieval legs, so the spans keep the library's instrumentation scope (`opentelemetry.util.genai.handler <version>`) and nest under whatever span is current on the calling thread: the planner's `tool_call()` runs inside `trelix.retrieve` / `trelix.planner`; the synthesizer's and the reviewer's calls run after `retrieve()` has returned and are roots unless the host has a span open (PR 4 of C-8 adds a per-hunk `trelix.review` parent). All 15 `build_chat_client` call sites in `src/` are covered: the reviewer, the synthesizer, the planner and its decomposition path, the agent loop, GraphRAG, query expansion, the concept graph, abstractive compression, contextual chunking, file and image summarisation, and the diagram connector. Requires `opentelemetry-util-genai>=1.2b0`, the `otel` extra's floor since this release.

| Attribute | On | Source |
|---|---|---|
| `gen_ai.operation.name` | every span | always `chat` (`tool_call()` is a forced function call on the chat API) |
| `gen_ai.provider.name` | every span | `TRELIX_LLM_PROVIDER` through the table below |
| `gen_ai.request.model` | every span | the backend's active model id: the Azure deployment, `TRELIX_LLM_LITELLM_MODEL`, Bedrock's active id (see below), else `TRELIX_LLM_MODEL`; also the span name's second word |
| `gen_ai.request.max_tokens` | every span | the effective cap the backend applies: the call's `max_tokens`, else `TRELIX_LLM_MAX_TOKENS` |
| `gen_ai.response.model` | `complete()` | `ChatResponse.model`: the response's model on OpenAI, Azure, Anthropic and LiteLLM; the request model on Vertex; on Bedrock `request["modelId"]`, the model that served the call after any fallback |
| `gen_ai.response.finish_reasons` | `complete()` | `[raw_finish_reason]`, the provider's own word (`end_turn`, `max_tokens`, `stop`, ...); absent when the provider reported none |
| `trelix.finish_reason` | `complete()` | trelix's normalised value: `stop`, `length`, `tool_calls`, `refusal`, `content_filter`, `paused`, `error` or `unknown` |
| `gen_ai.usage.input_tokens` | `complete()` | `input_tokens + cache_read_tokens + cache_write_tokens`, as the GenAI Anthropic conventions require (Anthropic's own `usage.input_tokens` is the non-cached part; the other backends report no cache counts, so there it equals `input_tokens`) |
| `gen_ai.usage.output_tokens` | `complete()` | `ChatResponse.output_tokens` |
| `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens` | `complete()` | the Anthropic prompt-cache counts; absent when zero (only the Anthropic backend fills them) |
| `error.type` | a failed call | the exception's class name, bare for builtins (`RuntimeError`) and module-qualified otherwise; the span status is `ERROR` with that same class name as its description, never `str(exc)`, because a provider error body can echo request details. The retrieval leg spans still record `str(exc)` (follow-up `fix/otel-leg-error-text`: align them) |
| `gen_ai.input.messages`, `gen_ai.system_instructions`, `gen_ai.output.messages` | content capture only | see [Content capture](#content-capture); `stream()` replies are never captured and images are never serialised |

`stream()` and `tool_call()` spans carry the request attributes only: `ToolCallResponse` has no usage, and a stream is not buffered to count it (follow-up `feat/llm-tool-call-usage` if the planner's per-query cost must be on spans). Nothing else from the request or reply reaches a span: no temperature, no `server.address`, no response id.

Provider mapping (`gen_ai.provider.name`, the well-known values of the GenAI registry where one exists):

| `TRELIX_LLM_PROVIDER` | `gen_ai.provider.name` |
|---|---|
| `openai` | `openai` |
| `azure` | `azure.ai.openai` |
| `anthropic` | `anthropic` |
| `bedrock` | `aws.bedrock` |
| `vertex` with `GOOGLE_API_KEY` | `gcp.gemini` |
| `vertex` with `GOOGLE_CLOUD_PROJECT` | `gcp.vertex_ai` |
| `litellm` | `litellm` (a custom value, which the conventions allow: the router hides the real provider) |

Before reading these spans:

- **Bedrock's request model is history-dependent.** `gen_ai.request.model` is the backend's active model id at call start, which the fallback logic may have swapped to the fallback model on an earlier call of the same process; `gen_ai.response.model` is the model that served this call. The two can differ within one process.
- **The unconfigured placeholder is a span too.** A backend without credentials returns `ChatResponse(model="none")` instead of calling a provider; that call is recorded like any reply, with `gen_ai.response.model="none"` and zero usage, so the calls that never left the process are visible rather than silently missing.
- **Stream lifecycle.** The wrapper is a generator: the span starts at the first `next()`, not at the call, and a generator that is never iterated produces no span. While the consumer drains the stream the span is detached from the current context (`suspend()`), so the consumer's own spans are not parented under it. A consumer that stops early closes the generator and the span ends `UNSET` with request attributes only; a backend error mid-stream ends it `ERROR` with `error.type`.
- **Metrics come with the spans.** Once a `MeterProvider` exists (see [Not instrumented](#not-instrumented) for when trelix installs one), `opentelemetry-util-genai` also records `gen_ai.client.operation.duration` for every chat span and `gen_ai.client.token.usage` (attributes `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.token.type=input|output`) for `complete()`. These are the library's, not trelix counters, and there is no switch for them.
- **Worker-thread spans are roots.** `FileSummarizer` runs `complete()` in the `trelix-summary` thread pool without context propagation, so those chat spans are unparented traces; the wrapper holds no per-call state, so concurrent calls on one instance are safe. Calls made on the thread that holds the `trelix.retrieve` span nest under it.
- **The switch is the environment's.** Chat spans follow `TRELIX_OTEL_ENABLED` as resolved from the process environment and `.env` (the factory holds only an `LLMConfig`, so it reads the flag the way the embedding counters do, memoised once per process); a `RetrievalConfig(otel_enabled=True)` constructed in code traces retrieval legs but not LLM calls. The off path returns the backend object unchanged (type-identical, no `opentelemetry` import) at the cost of that one memoised read.
- **Without the library, one WARNING.** `TRELIX_OTEL_ENABLED=true` with `opentelemetry-util-genai` not installed logs once per process and returns the bare backend; `pip install 'trelix[otel]'` resolves to 1.2b0 or newer. An install below the floor is not detected: on 1.0b0 or 1.1b0 the spans still appear, but a `stream()` span parents the consumer's spans, the cache-write count is missing and content capture is silently dropped, and on 1.0b0 a failed call's span also ends `UNSET` without `error.type` (that library's error step rejects the class-name string trelix passes; trelix logs the rejection at debug only).

  ```
  TRELIX_OTEL_ENABLED is set but OpenTelemetry GenAI spans are unavailable (<reason>) — LLM calls will NOT be traced. Install: pip install 'trelix[otel]'
  ```

---

## Content capture

Spans carry no prompt, reply or query text unless **two** switches are on, and each defaults to off:

| Switch | Default | Read by | What it decides |
|---|---|---|---|
| `TRELIX_OTEL_CAPTURE_CONTENT` | `false` | trelix | Whether trelix hands any text to the GenAI instrumentation at all. That text is the retrieval `query_text` of every leg span and, on the [chat spans](#llm-chat-spans), the prompt, the system instruction and the `complete()` reply (`gen_ai.input.messages`, `gen_ai.system_instructions`, `gen_ai.output.messages`; `stream()` replies are never captured). It includes repository code: retrieved context, diff hunks, the review system prompt. |
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `NO_CONTENT` | `opentelemetry-util-genai` (an OpenTelemetry name, not `TRELIX_`-prefixed) | Where the text trelix handed over goes: `SPAN_ONLY` puts it on span attributes (`gen_ai.retrieval.query.text` on the leg spans, the three message attributes on the chat spans), `EVENT_ONLY` puts it on the Logs signal (below), `SPAN_AND_EVENT` does both, `NO_CONTENT` (also what an unset or invalid value becomes) discards it. |

The trelix flag is an AND-gate in front of the OpenTelemetry one. A host process that opted another
library into content capture with `SPAN_ONLY` does not also receive trelix's query text, prompts and
repository code unless it sets `TRELIX_OTEL_CAPTURE_CONTENT=true` as well. The cost is
two switches to turn on, which is why the inert combination warns (below).

**The Logs signal.** `EVENT_ONLY`/`SPAN_AND_EVENT`, or `OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT=true`,
also emit one `gen_ai.client.inference.operation.details` log record per chat call on the Logs signal
(content-free under `NO_CONTENT`); trelix installs no `LoggerProvider`, so these go nowhere unless the
host configures one. Retrieval spans emit no such record (checked on `opentelemetry-util-genai` 1.0b0;
read in the 1.2b0 source), so `EVENT_ONLY` records the chat calls and nothing else from trelix.

**One WARNING when the trelix flag is inert.** With `TRELIX_OTEL_CAPTURE_CONTENT=true` and the upstream
mode `NO_CONTENT`, trelix logs once (best effort: parallel retrieval legs on that first query may
repeat the line), on the first span, leg or chat, that would have carried text:

```
TRELIX_OTEL_CAPTURE_CONTENT is true but OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT is NO_CONTENT (unset or invalid) — no prompt, reply or query text will be recorded. Set it to SPAN_ONLY, EVENT_ONLY or SPAN_AND_EVENT.
```

The decision is the memoised `TelemetryHandler`'s (`should_capture_content()`, fixed when the handler is
built on the first span), not a re-read of the environment, so a value exported into a running process
does not silence it until restart. On `opentelemetry-util-genai` 1.2b0, which reads the variable once,
that is also what the spans carry; on 1.0b0 and 1.1b0 the span attribute follows the live variable (see
the leg table above), so a late `SPAN_ONLY` export puts the query text on the spans from then on while
the one-time warning stands.

**Malformed values.** `TRELIX_OTEL_CAPTURE_CONTENT` is a pydantic boolean, the same parser as
`TRELIX_OTEL_ENABLED`: `true/1/yes/on` and `false/0/no/off` (case-insensitive, no surrounding spaces);
anything else, including a blank value, makes `RetrievalConfig()` raise. Where the embedding counters
resolve the flag from the environment, that is swallowed into "telemetry off, content off"; on every
`IndexConfig` construction it surfaces as the existing `Configuration error: TRELIX_OTEL_CAPTURE_CONTENT:
Input should be a valid boolean...` and exit 1 where the CLI catches it (`trelix search`), and as an
uncaught traceback (exit 1) in `trelix review`, identical to a malformed `TRELIX_OTEL_ENABLED` today. The
hosted GitHub App passes every `TRELIX_*` name to its review child, so a typo there fails every review.

---

## What gets measured (metrics)

Metrics are newer than tracing and their scope is **much narrower**: trelix's own counters cover
**embedding only**. Read the tables below as exhaustive for what trelix records itself; the
histograms that arrive from the span library without trelix code (operation duration for every
span, token usage for the chat spans) are described under [Not instrumented](#not-instrumented)
and [LLM chat spans](#llm-chat-spans).

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
`EmbedderConfig.provider` accepts, so a bge-code deployment emits **no embedding counters at
all** — a provider-filtered dashboard will look broken rather than look free. (`bge-code` was
described here as "trelix's flagship v2.0 embedder"; as of v3.1.7 it is marked **experimental**
— its pooling is unverified against BAAI's published `pooling_mode_lasttoken: true`.) Both run local models, so nothing is being billed
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

- **LLM tokens are per-call span attributes, not a trelix counter.** Synthesis, query planning,
  contextual chunking and file summarisation all call an LLM; each `complete()` carries
  `gen_ai.usage.*` on its [chat span](#llm-chat-spans), and `opentelemetry-util-genai` records them
  into its `gen_ai.client.token.usage` histogram once a `MeterProvider` exists. `stream()`
  (synthesis) and `tool_call()` (the planner's per-query call) carry no usage, so neither number is
  the whole bill, which on a hosted model is normally the largest trelix generates. Nothing in
  trelix aggregates them.
- **Retrieval latency or throughput** — trelix records no histogram for `retrieve()` or for
  fusion/rerank/assembly; those exist only as spans, per-query and sampled. The one exception is
  not trelix code: `opentelemetry-util-genai` records `gen_ai.client.operation.duration` (unit
  `s`, attribute `gen_ai.operation.name="retrieval"`, no leg attribute, so it aggregates across
  legs) for every retrieval leg span once a `MeterProvider` exists, and the same histogram with
  `gen_ai.operation.name="chat"` plus the provider and model attributes for every chat span. trelix installs its own on the
  first counted embedding provider call (every provider except `bge-code` and `nomic-code`, which
  never install one, and a `CachingEmbedder` hit is not a provider call; the vector leg embeds the
  query), a host may install one earlier, and a provider installed after the first span still
  receives the histogram. A `bge-code`/`nomic-code` deployment, or a process whose queries all hit
  the embedding cache, therefore exports the leg spans but records no duration unless the host
  installs a `MeterProvider`. Measured on `opentelemetry-util-genai` 1.0b0 and 1.2b0.
- **Reranker cost** — a Cohere or hosted cross-encoder rerank is a paid per-query call and
  is uncounted.
- **Cache effectiveness** — no hit/miss counters. `FederatedRetriever.cache_stats()`
  returns them on demand; nothing exports them.
- **Indexing** — no counters for files walked, symbols extracted, or chunks written.
- **HTTP API and MCP surfaces** — no request/error/duration counters. The HTTP layer has
  spans (v2.10.0) and audit rows (v3.0.0), but no metrics.

**Blunt consequence:** you cannot build a total-cost-of-trelix dashboard from these
counters. You can build an *embedding*-cost dashboard, and only for the providers marked
counted above. For LLM spend, the chat spans carry per-call usage for `complete()` and the library's
`gen_ai.client.token.usage` histogram sums it by model; `stream()` and `tool_call()` are not counted,
so your provider's billing surface stays the authority.

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

- Attribute names (`gen_ai.operation.name`, `gen_ai.data_source.id`, `gen_ai.retrieval.top_k`, etc.) may still change in a future OTel semantic-conventions release — and already have: the leg spans' `top_k` is emitted as `gen_ai.request.top_k` by `opentelemetry-util-genai` 1.0b0 and 1.1b0 and as `gen_ai.retrieval.top_k` (the name the conventions use today) by 1.2b0.
- `opentelemetry-util-genai` itself ships as pre-release betas (`1.2b0` is the newest as of 2026-10-07 and, since the chat spans, the `otel` extra's floor: `opentelemetry-util-genai>=1.2b0`) — and its Python API and attribute names do shift between betas: 1.2b0 renamed the cache-write attribute from `gen_ai.usage.cache_creation.input_tokens` (1.0b0, 1.1b0) to `gen_ai.usage.cache_write.input_tokens` and added `suspend()`/`activate()` on invocations, which is why the floor moved; trelix's spans carry the `cache_write` name.

trelix deliberately adopted the official conventions now (rather than defining its own `trelix.retrieval.*` attribute set) to avoid a painful rename migration later, but this means dashboards/alerts built against `gen_ai.*` attributes should be revisited if you see them break after an `opentelemetry-util-genai` upgrade.

The `trelix.*`-namespaced pipeline-stage spans (fusion/expansion/rerank/etc.) are trelix's own naming and are not subject to this caveat — they won't change without a trelix version bump and a CHANGELOG entry.

---

## Relationship to existing (non-OTel) telemetry

This is additive — it does not replace either of trelix's existing telemetry mechanisms:

- **`TelemetryWriter`** (`TRELIX_TELEMETRY_ENABLED=true`) — writes one row per `retrieve()` call to the `query_telemetry` SQLite table (query text, intent, latency, result count, expansion columns) in the index DB. The only reader is the `trelix telemetry` CLI report. `trelix eval` does **not** read this table — it re-runs the queries in a golden JSONL file live through `Retriever` and computes nDCG@10 / recall@10 / MRR from those fresh results, so telemetry can be off and `eval` still works. (Earlier revisions of this doc claimed `eval` consumed the telemetry table; that was never true.)
- **Debug trace JSON** (always on unless commented out in `retriever.py`) — writes a structured `debug/<ts>_<slug>.json` file per query, beside the index (`<repo>/.trelix/debug/` for the default `TRELIX_STORE_DB_PATH`; an index kept elsewhere, as a `trelix eval-suite` run keeps it, gets its traces there and the source tree stays untouched), with plan/legs/fusion/expansion/rerank/assembly data.

Use OTel tracing when you want to export spans to an existing observability stack (Jaeger, Grafana Tempo, Honeycomb, Datadog, etc. — anything that accepts OTLP). Use the other two when you want local-file or in-DB analysis without standing up a collector.

---

## Relationship to audit logging

The v3.0.0 audit trail (`TRELIX_AUDIT_ENABLED=true`, documented in [AUDIT.md](AUDIT.md)) is **not** a telemetry mechanism and does not replace any of the above. The distinction matters when deciding where to look during an incident:

| | Audit log | Query telemetry | OTel tracing |
|---|---|---|---|
| Question | Who did what, when, and was it allowed? | How did retrieval perform over time? | Where did the time go inside one query? |
| Unit | one row per **HTTP request** | one row per **`retrieve()` call** | one span per pipeline stage/leg |
| Records caller identity | **yes** (`principal` — `sub@iss` or `static-token`) | no | no |
| Records query text | no (deliberately) | **yes**, verbatim | only with `TRELIX_OTEL_CAPTURE_CONTENT=true` and a span content mode upstream (`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` = `SPAN_ONLY` or `SPAN_AND_EVENT`); both default off, so not recorded (the same two switches govern prompts and replies on the chat spans) |
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

trelix handles this internally (`with_current_context()` in `src/trelix/retrieval/otel_tracing.py`) — no action needed by callers. The one place trelix itself does not propagate is the `FileSummarizer` thread pool at index time, so its chat spans are unparented roots (see [LLM chat spans](#llm-chat-spans)). If you're instrumenting your own code that calls into `Retriever` from a thread pool, be aware of the same caveat for your own spans.
