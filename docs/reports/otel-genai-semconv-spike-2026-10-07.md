# OpenTelemetry GenAI semantic-conventions spike — 2026-10-07

Roadmap item C-8 (OpenTelemetry spans for LLM calls), requirement R-C6-01: before trelix emits a
single `chat` span, pin the conventions and the library it would build on and measure what they
do. This report is PR 1 of 4; it ships no package code, only one docs pin test. Every claim below names the source read or the
probe run on 2026-10-07. Tags: [SRC] a primary source fetched at a pinned version or commit,
[PROBE] executed in a scratch virtualenv against the published wheels, [CODE] read in this
repository at the commit this report was written against (`845590fe`, `develop`).

## 1. Sources and pins

| Source | Pin | Read at | How the pin was chosen |
|---|---|---|---|
| GenAI semantic conventions | `open-telemetry/semantic-conventions-genai` commit `cb10b70c15c099ccab144e8316d934c9699da0fd` (2026-10-05, "Clarify failed GenAI content capture (#459)") | `https://raw.githubusercontent.com/open-telemetry/semantic-conventions-genai/cb10b70c15c099ccab144e8316d934c9699da0fd/docs/gen-ai/{README,gen-ai-spans,gen-ai-metrics,gen-ai-token-metrics,anthropic,aws-bedrock,openai}.md` and `docs/registry/attributes/gen-ai.md` | The repository has **no tags and no releases** (`/repos/.../tags` and `/releases` both returned `[]`), so a commit is the only pin. Repository created 2026-05-05. [SRC] |
| Core semantic conventions | `open-telemetry/semantic-conventions` tag `v1.44.0` (published 2026-08-04; `/releases/latest` is the same tag) | `https://raw.githubusercontent.com/open-telemetry/semantic-conventions/v1.44.0/docs/gen-ai/gen-ai-spans.md`, `docs/gen-ai/README.md`, `docs/registry/attributes/gen-ai.md`, `docs/registry/attributes/error.md` | Newest release. Its `docs/gen-ai/` pages are "Moved" stubs pointing at the GenAI repository, and all 60 `gen_ai.*` rows of its registry carry the Deprecated badge: 50 say "Moved to the OpenTelemetry GenAI semantic conventions repository", 8 "Replaced by `…`, which has moved to" that repository, and 2 (`gen_ai.prompt`, `gen_ai.completion`) "Removed, no replacement at this time". `error.type` is Stable there. [SRC] |
| `opentelemetry-util-genai` | `1.2b0` (PyPI upload 2026-09-24, newest) | PyPI JSON `https://pypi.org/pypi/opentelemetry-util-genai/json`; wheels 1.0b0, 1.1b0, 1.2b0 downloaded and extracted; 1.2b0 installed in a scratch venv with `opentelemetry-sdk 1.45.1` (pulled `opentelemetry-api 1.45.1`, `opentelemetry-semantic-conventions 0.66b1`, `opentelemetry-instrumentation 0.66b1`) | Newest version. trelix's `otel` extra accepts `>=1.0b0` (`pyproject.toml:201-206`), and CI installs that extra unpinned (`.github/workflows/ci.yml:234`, `:383`, `:442`), so CI resolves to the newest version while an older developer environment may still hold 1.0b0. [SRC][PROBE][CODE] |
| `opentelemetry-util-genai` source | `open-telemetry/opentelemetry-python-genai` tag `opentelemetry-util-genai==1.2b0` (commit `14c76fee…`) | `https://raw.githubusercontent.com/open-telemetry/opentelemetry-python-genai/opentelemetry-util-genai%3D%3D1.2b0/util/opentelemetry-util-genai/src/opentelemetry/util/genai/version.py` (`__version__ = "1.2b0"`); `https://raw.githubusercontent.com/open-telemetry/opentelemetry-python-genai/opentelemetry-util-genai%3D%3D1.2b0/util/opentelemetry-util-genai/CHANGELOG.md` (368 lines; the line numbers cited in section 5 are of this file) | See section 10: the package's own metadata points elsewhere. [SRC] |
| `opentelemetry-semantic-conventions` (Python constants) | `0.66b1` (PyPI upload 2026-10-06, newest) | installed in the scratch venv; `opentelemetry.semconv._incubating.attributes.gen_ai_attributes` and `…metrics.gen_ai_metrics` | Newest version; what util-genai 1.2b0 resolves (`>=0.64b0,<1`). [PROBE] |

## 2. Status

- Every GenAI page at the pin is **Development** (`docs/gen-ai/README.md` and `gen-ai-spans.md`
  `**Status**: Development`). The GenAI registry at the pin has 79 attribute rows and 127
  stability badges, all `Development`. The only Stable attributes an inference span uses come
  from the core repository: `error.type`, `server.address`, `server.port`. [SRC]
- `OTEL_SEMCONV_STABILITY_OPT_IN` is mentioned nowhere in the pinned spans, metrics, registry,
  OpenAI, Anthropic or Bedrock pages (0 hits). The only opt-in mechanism the conventions name
  is content capture (section 4). [SRC]
- Consequence for dashboards: attribute names are not frozen. Two renames already happened
  between util-genai 1.1b0 and 1.2b0 (section 6).

## 3. Inference span at the pin

`gen-ai-spans.md` §Inference (lines 37-110 at the pin). **Span name** `{gen_ai.operation.name}
{gen_ai.request.model}` (line 53); **kind** `CLIENT` (`INTERNAL` allowed for in-process models,
lines 48-51); **status** per the core "Recording Errors" document (line 57). [SRC]

Attribute table, as printed from the pinned page (key | stability | requirement level), with the
trelix field that holds the value today ([CODE], `src/trelix/llm/client.py:67-86` unless noted):

| Attribute | Stability | Requirement level | trelix source today |
|---|---|---|---|
| `gen_ai.operation.name` | Development | Required | fixed `chat` |
| `gen_ai.provider.name` | Development | Required | `LLMConfig.provider` selector; mapping needed (section 3.1) |
| `error.type` | Stable | Conditionally Required if the operation ended in an error | exception class (util-genai derives it, section 5) |
| `gen_ai.request.model` | Development | Conditionally Required if available | the backend's `_model` (`providers/anthropic_backend.py:47`, `openai_backend.py:61` uses the Azure deployment name, `bedrock_backend.py:80` may swap to the fallback model, `litellm_backend.py:41`, `vertex_backend.py:52`) |
| `gen_ai.request.choice.count`, `.seed`, `.stream`, `.top_k`, `gen_ai.conversation.id`, `gen_ai.output.type`, `gen_ai.prompt.name`, `gen_ai.prompt.version`, `server.port` | Development (`server.port` Stable) | Conditionally Required, each under its own condition | not held by trelix |
| `gen_ai.request.max_tokens` | Development | Recommended | effective cap `max_tokens or self._config.max_tokens` (`anthropic_backend.py:254`) |
| `gen_ai.request.temperature`, `.top_p`, `.stop_sequences`, `.frequency_penalty`, `.presence_penalty`, `.previous_response.id`, `.reasoning.level`, `gen_ai.conversation.compacted` | Development | Recommended | not held by trelix |
| `gen_ai.response.finish_reasons` | Development | Recommended (`string[]`, "corresponding to each generation received") | `ChatResponse.raw_finish_reason` (the provider's word); `finish_reason` is trelix's normalised value |
| `gen_ai.response.model` | Development | Recommended | `ChatResponse.model`: the response model on Anthropic (`anthropic_backend.py:267`), OpenAI/Azure (`openai_backend.py:149`), LiteLLM (`litellm_backend.py:84`, falls back to the request model); the served model id on Bedrock (`bedrock_backend.py:442`); the request model on Vertex (`vertex_backend.py:131`) |
| `gen_ai.response.id`, `gen_ai.response.time_to_first_chunk` | Development | Recommended (the latter if streaming) | not held by trelix |
| `gen_ai.usage.input_tokens` | Development | Recommended; "SHOULD include all types of input tokens, including cached tokens" (footnote [28]) | `ChatResponse.input_tokens`, which on Anthropic is the **non-cached** count (section 3.2) |
| `gen_ai.usage.output_tokens` | Development | Recommended | `ChatResponse.output_tokens` |
| `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens` | Development | Recommended when applicable | `ChatResponse.cache_read_tokens` / `cache_write_tokens`, filled only by the Anthropic backend (`anthropic_backend.py:274-279`, from `cache_read_input_tokens` and `cache_creation_input_tokens`); Bedrock reads no cache usage (`bedrock_backend.py:445-446`) |
| `gen_ai.usage.reasoning.output_tokens`, `gen_ai.usage.{text,image,audio}.*` | Development | Recommended when applicable | not held by trelix |
| `server.address` | Stable | Recommended | not held by trelix |
| `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.definitions`, `gen_ai.prompt.variable` | Development | **Opt-In**, each flagged "likely to contain sensitive information" | the request messages, `system`, and `ChatResponse.content` |

### 3.1 `gen_ai.provider.name` well-known values

Registry at the pin: `anthropic`, `aws.bedrock`, `azure.ai.inference`, `azure.ai.openai`,
`cohere`, `deepseek`, `gcp.gemini` (the `generativelanguage` endpoint), `gcp.gen_ai` (any Google
endpoint), `gcp.vertex_ai` (the `aiplatform` endpoint), `groq`, `ibm.watsonx.ai`, `mistral_ai`,
`moonshot_ai`, `openai`, `perplexity`, `x_ai`; "otherwise, a custom value MAY be used". The
Python constants in `opentelemetry-semantic-conventions 0.66b1` (`GenAiProviderNameValues`)
carry the same list minus `moonshot_ai`. trelix's `LLMConfig.provider` values are `openai`,
`azure`, `anthropic`, `bedrock`, `vertex`, `litellm` (`src/trelix/core/config.py:1313`;
`src/trelix/llm/factory.py:14-34` dispatches on them); `litellm` has no well-known value.
[SRC][PROBE][CODE]

### 3.2 Provider pages

- Anthropic (`anthropic.md` at the pin, footnotes [22], [23], [27]): "Anthropic reports this
  separately from `input_tokens`. This value MUST be added to the Anthropic `input_tokens` to
  compute `gen_ai.usage.input_tokens`" for both cache counts; "`gen_ai.usage.input_tokens =
  input_tokens + cache_read_input_tokens + cache_write_input_tokens`". [SRC]
- Bedrock (`aws-bedrock.md` at the pin, footnotes [23], [24], [28]): cache read and write
  values "SHOULD be included in `gen_ai.usage.input_tokens`"; `gen_ai.usage.input_tokens`
  "SHOULD include all types of input tokens, including cached tokens". [SRC]
- trelix's `ChatResponse.input_tokens` is the provider's `usage.input_tokens` verbatim
  (`anthropic_backend.py:270`), so a span that copies it would under-report Anthropic input by the
  two cache counts. [CODE]

## 4. Content capture: the opt-in variable and its default

The conventions: "OpenTelemetry instrumentations SHOULD NOT capture them by default, but SHOULD
provide an option for users to opt in" (`gen-ai-spans.md` lines 1305-1306 at the pin); on spans
the structured value "SHOULD be serialized to JSON string" (lines 1350-1352). [SRC]

util-genai implements it with two environment variables (`environment_variables.py`, all three
versions): [PROBE]

| Variable | Values | Default | Read when |
|---|---|---|---|
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `NO_CONTENT`, `SPAN_ONLY`, `EVENT_ONLY`, `SPAN_AND_EVENT` (case-insensitive) | `NO_CONTENT`; unset, blank, or any other value becomes `NO_CONTENT` (an invalid value logs one WARNING: "`bogus` is not a valid option … Defaulting to `NO_CONTENT`") | **once, at `TelemetryHandler()` construction** (1.2b0, `handler.py:133`; changing the variable afterwards does not change that handler, a fresh handler sees the new value). 1.0b0 and 1.1b0 re-read it when each span finishes (`_retrieval_invocation.py:113` → `utils.py:80-82`): a handler built with the variable unset records the query text as soon as `SPAN_ONLY` is exported |
| `OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT` | `true` / `false` | unset: emit the event for `EVENT_ONLY` and `SPAN_AND_EVENT` only | per inference invocation (1.2b0 `_inference_invocation.py:139`, at construction; 1.0b0 line 214, at finish) |

Measured on 1.2b0 with an in-memory span exporter and an in-memory log exporter. The invocation
carried canary strings in `query_text` (retrieval), `input_messages`, `system_instruction` and
`output_messages` (inference): [PROBE]

| Mode | `gen_ai.retrieval.query.text` on the leg span | `gen_ai.input.messages` / `gen_ai.system_instructions` / `gen_ai.output.messages` on the chat span | `gen_ai.client.inference.operation.details` log record |
|---|---|---|---|
| unset (`NO_CONTENT`) | absent | absent | none |
| invalid value | absent (+ one WARNING) | absent | none |
| `SPAN_ONLY` | present | present, as JSON strings | none |
| `EVENT_ONLY` | absent | absent | **one record per chat call carrying all three message attributes with the canaries** |
| `SPAN_AND_EVENT` | present | present | one record per chat call, with content |
| `NO_CONTENT` + `EMIT_EVENT=true` | absent | absent | one record per chat call **without** the message attributes |

Three consequences for trelix:

1. **There is no content leak to fix today.** `retrieval_leg_span` sets `invocation.query_text`
   whenever a query is passed, with no content gate (`src/trelix/retrieval/otel_tracing.py:164-165`),
   but util-genai 1.0b0, 1.1b0 and 1.2b0 all drop it unless the variable selects a span mode
   (1.0b0: `_retrieval_invocation.py:111-117` guards on `should_capture_content_on_spans()`). The
   planned "fix the query_text default" change has no bug behind it. [PROBE][CODE]
2. **Content can leave through the Logs signal, not only through spans.** `EVENT_ONLY` puts the
   messages in a log record while the span stays clean. Any trelix-side gate must therefore
   stop handing text to the invocation, not merely watch span attributes; and a host that
   configures a `LoggerProvider` and sets `EVENT_ONLY` receives trelix's prompts. trelix installs
   no `LoggerProvider`, so by default these records go nowhere. [PROBE]
3. A trelix-side gate that reads the variable itself would disagree with the memoised handler
   (`otel_tracing.py:87-89` builds one handler per process); ask the handler
   (`TelemetryHandler.should_capture_content()`, present in all three versions) instead. [PROBE]

The JSON shapes 1.2b0 writes under `SPAN_ONLY` (literal span attribute values):
`gen_ai.input.messages` = `[{"role":"user","parts":[{"content":"…","type":"text"}],"name":null}]`,
`gen_ai.system_instructions` = `[{"content":"…","type":"text"}]`,
`gen_ai.output.messages` = `[{"role":"assistant","parts":[{"content":"…","type":"text"}],"finish_reason":null,"name":null}]`. [PROBE]

## 5. `opentelemetry-util-genai`: versions and API

PyPI uploads: 0.1b0 2025-09-25, 0.2b0 2025-10-15, 0.3b0 2026-02-20, 0.4b0 2026-05-01, **1.0b0
2026-07-09, 1.1b0 2026-08-20, 1.2b0 2026-09-24**. 1.2b0 requires `opentelemetry-api~=1.43`,
`opentelemetry-instrumentation>=0.64b0,<1`, `opentelemetry-semantic-conventions>=0.64b0,<1`,
`wrapt>=1.17.0,<3.0.0`, Python `>=3.10`. [SRC]

Inference API in 1.2b0 (`inspect.signature` on the installed wheel): [PROBE]

- `TelemetryHandler(tracer_provider=None, meter_provider=None, logger_provider=None,
  completion_hook=None, instrumentation_scope_name=None, instrumentation_scope_version=None)`.
  Default scope on every span and metric: `opentelemetry.util.genai.handler <version>`, schema
  URL `https://opentelemetry.io/schemas/1.37.0`. Passing a scope name changes the scope of
  **every** span the handler emits, including trelix's existing retrieval legs.
- `handler.inference(provider, *, request_model=None, server_address=None, server_port=None,
  operation_name=None, error_type_resolver=None, context=None, _attach_to_context=True,
  conversation_id=None) -> InferenceInvocation` starts the span (`chat {request_model}`, kind
  `CLIENT`) and attaches it to the current context.
- `handler.retrieval(*, data_source_id=None, provider=None, request_model=None,
  server_address=None, server_port=None, context=None, _attach_to_context=True)`.
- `InferenceInvocation` is a plain class (not a dataclass). Values are assigned as attributes
  (`max_tokens`, `input_messages`, `system_instruction`, `output_messages`,
  `response_model_name`, `input_tokens`, `output_tokens`, `cache_read_input_tokens`,
  `cache_write_input_tokens`, `finish_reasons`, `attributes` for custom keys) and the span is
  finished with `stop()` or `fail(exc)`. Methods: `activate`, `suspend`, `stop`, `fail`,
  `record_stream_chunk`, `set_input_tokens`, `set_output_tokens`, `set_cache_read_input_tokens`.
  Properties: `should_capture_content` (read-only, `_invocation.py:140-141`; the handler's
  `should_capture_content()` is the method) and `cache_creation_input_tokens` (alias).
- `fail(RuntimeError("boom"))` sets status `ERROR`, status description `boom` (the exception
  text, verbatim) and `error.type` = `RuntimeError`; no `gen_ai.usage.*` key is written unless
  assigned.
- Zero-valued cache counts are dropped in 1.2b0 (`… or None`, `_inference_invocation.py:311-315`):
  `cache_read_input_tokens = 0` and `cache_write_input_tokens = 0` produce no cache attribute,
  while `input_tokens = 10` and `output_tokens = 1` are written as given. 1.0b0 and 1.1b0 filter
  only `None` and write both zeros (`gen_ai.usage.cache_creation.input_tokens: 0`,
  `gen_ai.usage.cache_read.input_tokens: 0`); see the differences table below.

Differences 1.0b0 / 1.1b0 → 1.2b0, from the extracted wheels and the util-genai CHANGELOG at the
`opentelemetry-util-genai==1.2b0` tag (section 1; its 1.2b0 section is dated 2026-09-24): [PROBE][SRC]

| Surface | 1.0b0 | 1.1b0 | 1.2b0 |
|---|---|---|---|
| Cache-write field / attribute | `cache_creation_input_tokens` → `gen_ai.usage.cache_creation.input_tokens` | same | `cache_write_input_tokens` → `gen_ai.usage.cache_write.input_tokens`; `cache_creation_input_tokens` kept as a deprecated alias (CHANGELOG lines 47, 134-135) |
| Zero cache counts (`cache_read_input_tokens = 0`, cache-write `= 0`) | emitted as `0` | emitted as `0` | dropped |
| Retrieval `top_k` attribute | `gen_ai.request.top_k`, float | same | `gen_ai.retrieval.top_k`, int (CHANGELOG line 103) |
| `suspend()` / `activate()` (detach the span from the current context, e.g. while a stream is drained) | absent | absent | present (CHANGELOG line 63) |
| `TelemetryHandler(instrumentation_scope_name=, instrumentation_scope_version=)` | absent | absent | present (CHANGELOG lines 60-61) |
| Message part classes | `Text`, … | same | `TextPart`, …; `Text` kept as a deprecated alias (CHANGELOG line 119) |
| Streaming metrics | none | `gen_ai.client.operation.time_to_first_chunk` and `time_per_output_chunk` (CHANGELOG line 180) | plus `record_stream_chunk()` (CHANGELOG line 31) |
| Content gating | `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`, default `NO_CONTENT` | same | same |
| `gen_ai.client.inference.operation.details` event | present | present | present |

## 6. Cache-token attribute names, by source

| Source | Name for the cache-write count |
|---|---|
| GenAI conventions at the pin (registry, Anthropic and Bedrock pages) | `gen_ai.usage.cache_write.input_tokens`; `cache_creation` appears nowhere in the pinned registry (0 hits) |
| util-genai 1.0b0 and 1.1b0 (measured span attribute) | `gen_ai.usage.cache_creation.input_tokens` |
| util-genai 1.2b0 (measured span attribute) | `gen_ai.usage.cache_write.input_tokens` |
| `opentelemetry-semantic-conventions 0.66b1` constants | `GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"`; no `CACHE_WRITE` constant |

`gen_ai.usage.cache_read.input_tokens` is the same in all four. Which cache-write name a trelix
span carries therefore depends on the installed util-genai unless the `otel` extra's floor moves
to `1.2b0`. [SRC][PROBE]

## 7. Metrics util-genai already records

With a `MeterProvider` installed, util-genai 1.0b0, 1.1b0 and 1.2b0 all record, for every
invocation their handler finishes: [PROBE]

| Instrument | Recorded for | Attributes measured | Unit |
|---|---|---|---|
| `gen_ai.client.operation.duration` (histogram) | `retrieval` **and** `chat` | `gen_ai.operation.name`; for chat also `gen_ai.request.model`, `gen_ai.provider.name`, `gen_ai.response.model`. No `gen_ai.data_source.id`, so retrieval durations aggregate across legs | `s` |
| `gen_ai.client.token.usage` (histogram) | `chat` with `input_tokens` / `output_tokens` assigned | the chat attributes above plus `gen_ai.token.type` = `input` or `output` | `{token}` |

A `MeterProvider` installed **after** the handler was built still receives these points (the
API's proxy meter re-binds; measured on 1.0b0 and 1.2b0). In trelix the handler is built on the
first span (`otel_tracing.py:71-92`), the `trelix.retrieve` root span opens at
`src/trelix/retrieval/retriever.py:421`, the first query embedding happens inside the vector leg
(when the strategy includes it) at `retriever.py:1179-1182`, in the per-sub-query helper
`_run_subquery_legs` (the off-by-default file-summary and sub-chunk legs embed later, at
`665-668` and `680-683`), and that embedding call installs trelix's `MeterProvider` when none exists
(`otel_tracing.py:352-354`, reached only from `record_embedding_call`, `otel_tracing.py:422-457`) —
but only when it is a counted provider call: `record_embedding_call` is reached from
`_count_embed_call` (`embedder/base.py:78-93`, called by every provider in `base.py` and by
`embedder/cohere.py:140`); `embedder/bge_code.py` and `embedder/nomic_code.py` contain no call
(0 matches), and a `CachingEmbedder` hit returns before the provider (`embedder/cache.py:57-62`).
So with `TRELIX_OTEL_ENABLED=true` and the `otel` extra installed, **retrieval leg durations are
already recorded today without any trelix metrics code, once a counted embedding provider call has
happened or the host has installed a `MeterProvider`**; a `bge-code`/`nomic-code` deployment, or a
process whose queries all hit the embedding cache, exports the leg spans and records no duration.
[PROBE][CODE]

What the conventions define at the pin is different: `gen-ai-metrics.md` defines
`gen_ai.client.operation.duration`, `gen_ai.client.operation.time_to_first_chunk` and
`gen_ai.client.operation.time_per_output_chunk`; `gen-ai-token-metrics.md` defines the counters
`gen_ai.client.inference.usage.{input_tokens,output_tokens,cache_read.input_tokens,cache_write.input_tokens,reasoning.output_tokens}`
and the histograms `gen_ai.client.inference.operation.{input_tokens,output_tokens}`, broken down
by `gen_ai.token.modality`. **`gen_ai.client.token.usage` and `gen_ai.token.type` appear in none
of the eight pinned pages** (0 hits each); the Python constant `GEN_AI_CLIENT_TOKEN_USAGE` still
exists in `opentelemetry-semantic-conventions 0.66b1`, which is what util-genai 1.2b0 records.
A token dashboard built on `gen_ai.client.token.usage` is built on a name the conventions have
already moved away from. [SRC][PROBE]

## 8. Retrieval span at the pin, and what trelix emits

`gen-ai-spans.md` §Retrievals (lines 593-741): name `{gen_ai.operation.name}
{gen_ai.data_source.id}`, kind `CLIENT`; `gen_ai.operation.name` Required;
`gen_ai.data_source.id`, `gen_ai.provider.name`, `gen_ai.request.model`, `error.type` Conditionally
Required; `gen_ai.retrieval.top_k` Recommended; `gen_ai.retrieval.query.text` and
`gen_ai.retrieval.documents` **Opt-In** ("may contain sensitive information", footnote [9]). [SRC]

trelix's leg span (`otel_tracing.py:139-198`) calls `handler.retrieval(data_source_id=leg)` and
assigns `query_text` and `top_k` (as `float`). Measured result per installed version, default
environment: `{"gen_ai.operation.name": "retrieval", "gen_ai.data_source.id": "vector",
"gen_ai.request.top_k": 10.0}` on 1.0b0 and 1.1b0; `{"…", "gen_ai.retrieval.top_k": 10}` on
1.2b0; no query text on any of them. The existing tests assert only `gen_ai.operation.name` and
`gen_ai.data_source.id` (`tests/unit/test_otel_tracing.py:228-229`), so none of this is pinned.
[PROBE][CODE]

## 9. Where trelix would attach chat spans

All LLM traffic is built through one factory, `build_chat_client` (`src/trelix/llm/factory.py`,
39 lines). Call sites in `src/` (15, found with a Python regex over every file): `agent/loop.py:73`,
`compression/base.py:162`, `graph/concepts.py:57`, `indexing/connectors/diagram.py:117`,
`indexing/connectors/image.py:143`, `indexing/indexer.py:433` and `:462`,
`retrieval/graph_rag.py:106`, `retrieval/planner/agent.py:599` and `:632`,
`retrieval/query_expansion.py:64` and `:103`, `retrieval/synthesizer.py:142` and `:165`,
`review/reviewer.py:176`. The placeholder a backend returns without credentials has
`model == "none"` (`client.py:22`, `UNCONFIGURED_MODEL`). The OTel switch is
`RetrievalConfig.otel_enabled` (`TRELIX_OTEL_ENABLED`, `src/trelix/core/config.py:823-826`) with
`OTEL_SERVICE_NAME` and `OTEL_EXPORTER_OTLP_ENDPOINT` (`config.py:827-834`); no other `otel_*`
field exists. [CODE]

## 10. Provenance of util-genai 1.x (gap closed)

The package's PyPI metadata (`project_urls`, copied from its own `pyproject.toml`) names
`https://github.com/open-telemetry/opentelemetry-python-contrib` as Homepage and Repository.
That repository's `util/opentelemetry-util-genai/version.py` at `main` (read 2026-10-07) reads
`0.5b0.dev`, its CHANGELOG there stops at 0.4b0 (2026-05-01), and `git ls-remote` shows util-genai release
branches only up to `v0.5bx` plus a `v0.6b0` version-bump branch; no ref there matches 1.0b0,
1.1b0 or 1.2b0. The published 1.x line is in **`open-telemetry/opentelemetry-python-genai`**:
tags `opentelemetry-util-genai==1.0b0`, `==1.1b0`, `==1.2b0` exist, `version.py` at the 1.2b0 tag
reads `1.2b0`, `main` (read 2026-10-07) reads `1.3b0.dev`, and its CHANGELOG at the 1.2b0 tag
carries the 1.0b0-1.2b0 sections.
The link to `opentelemetry-python-genai` in `docs/OBSERVABILITY.md` was therefore right about
the repository; it now points at the `util/opentelemetry-util-genai` tree at the 1.2b0 tag.
What remains open is only that the package metadata is stale upstream. [SRC]

## 11. Corrections made to `docs/OBSERVABILITY.md` in this PR

| Claim in the document | Finding | Change |
|---|---|---|
| "metrics cover embedding only" (introduction and "What gets measured") | True of trelix's own counters; false of the process, which also records `gen_ai.client.operation.duration` per retrieval leg (section 7) | Says "trelix's own counters cover embedding only" and points at the histogram |
| Leg table: each leg sets `query_text` | Handed over, but recorded only under the upstream span modes; absent by default (section 4) | Footnote under the table |
| "Retrieval latency or throughput — no histogram for `retrieve()`, for any leg" | A per-leg duration histogram is recorded once a `MeterProvider` exists (section 7) | Bullet rewritten; the "LLM tokens" bullet stays, it is still true until chat spans exist |
| Link to `semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md` | A "Moved" stub since the conventions moved repositories (section 1) | Links the GenAI repository at commit `cb10b70c`, with the reason a commit is pinned |
| Link to `opentelemetry-python-genai` | Correct repository (section 10); the design for this work assumed python-contrib | Points at the util-genai tree at the 1.2b0 tag |
| "`opentelemetry-util-genai` … `1.0b0` at time of writing" | 1.2b0 since 2026-09-24; the extra accepts `>=1.0b0` | States both, and the two renames between them |
| Example attribute `gen_ai.request.top_k` | 1.0b0/1.1b0 name; 1.2b0 and the conventions use `gen_ai.retrieval.top_k` | Lists the rename as a demonstrated case |
| Audit table: query text "as span attributes" | Only under the upstream span modes | Row says so |

## 12. Open points for the implementation PRs

- The `otel` extra floor: `>=1.0b0` keeps `gen_ai.usage.cache_creation.input_tokens` and
  `gen_ai.request.top_k` possible on an installed system; `>=1.2b0` is the surface measured
  here and the only one with `suspend()`/`activate()`.
- `gen_ai.usage.input_tokens` must be `input_tokens + cache_read_tokens + cache_write_tokens`
  for Anthropic (section 3.2); `ChatResponse` carries all three.
- The Logs-signal path (section 4) means any trelix content gate must sit before the text is
  assigned on the invocation; the configuration text must name `EVENT_ONLY`.
- Scope: passing `instrumentation_scope_name` to `TelemetryHandler` would rename the scope of
  every existing retrieval span (section 5); leave it upstream's.
- `gen_ai.client.token.usage` is what will appear once chat spans exist; it is not the name the
  pinned conventions define (section 7). Document it as the library's behaviour, not as
  conformance.
