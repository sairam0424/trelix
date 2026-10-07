# Offline and local-model mode

`TRELIX_LLM_BASE_URL` points the `openai` backend at an OpenAI-compatible server (Ollama, llama-server,
a gateway) with no cloud key. This guide is the operator's recipe: what to set, why the context length
should be 64k, what the prompt-truncation check catches, what the 20B model-size floor means, which
calls every search makes once the variable is set, and the one-time prefetch an air-gapped machine
needs. The variable reference, with the full URL shape rules, stays in
[PROVIDERS.md](PROVIDERS.md#openai-with-a-local-openai-compatible-server); this guide agrees with it
and does not restate it.

## Purpose and limits

- Only the `openai` backend reads `TRELIX_LLM_BASE_URL`. Azure keeps `AZURE_ENDPOINT`; LiteLLM reads
  its own `api_base`/`OPENAI_API_BASE` settings, documented by LiteLLM, not here; Anthropic and
  Bedrock have no base URL in trelix. With any of those as `TRELIX_LLM_PROVIDER` the URL is ignored
  with one WARNING per LLM client built: `TRELIX_LLM_BASE_URL is set but TRELIX_LLM_PROVIDER=anthropic
  does not use it`.
- `trelix review` is guarded: a prompt the server cut is reported, per hunk, as unreviewed. `trelix
  ask` (a stream) and the query planner's tool call are not checked; they run against the server
  without the check.
- The openai SDK's own `OPENAI_BASE_URL` keeps working when `TRELIX_LLM_BASE_URL` is unset, exactly as
  before, and that route is unguarded: no keyless client, no `max_tokens` request shape, no
  prompt-truncation check, no model-size warning (openai-python reads `OPENAI_BASE_URL` itself).

## What you set

| Variable | Value | Notes |
|---|---|---|
| `TRELIX_LLM_PROVIDER` | `openai` | The only backend that reads the URL. |
| `TRELIX_LLM_BASE_URL` | `http://127.0.0.1:11434/v1` (Ollama), `http://127.0.0.1:8080/v1` (llama-server) | A scheme (`http://` or `https://`), a host, a valid port, no user name or password, no whitespace; blank is unset. A bad value is `Configuration error: llm -> base_url: ...`, exit 1; the message names the variable, never the value. Full shape rules: [PROVIDERS.md](PROVIDERS.md#openai-with-a-local-openai-compatible-server). |
| `TRELIX_LLM_MODEL` | the server's own tag, e.g. `qwen3-coder:30b-a3b` | The default `gpt-4o` is a 404 on Ollama (`exception:NotFoundError` on every hunk). |
| `OPENAI_API_KEY` | optional | Without it trelix sends the fixed bearer `trelix-local`: a public constant, not a secret. Set it when the server checks a key (llama-server `--api-key`, a gateway). |
| `TRELIX_LLM_LOCAL_CONTEXT_TOKENS` | the server's context length, e.g. `65536` | `1024` to `2000000`; needs `TRELIX_LLM_BASE_URL`; read only when `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null` (see [Context length and the retrieval budget](#context-length-and-the-retrieval-budget)). |

## Ollama

```bash
OLLAMA_CONTEXT_LENGTH=65536 ollama serve
ollama pull qwen3-coder:30b-a3b
export TRELIX_LLM_PROVIDER=openai
export TRELIX_LLM_BASE_URL=http://127.0.0.1:11434/v1
export TRELIX_LLM_MODEL=qwen3-coder:30b-a3b
export TRELIX_LLM_LOCAL_CONTEXT_TOKENS=65536
export TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null  # else the explicit 12000 budget wins
```

- Keep the `/v1`: Ollama serves the OpenAI API under it, and without it every call is a 404
  (`exception:NotFoundError` on every hunk).
- The context length is set server-side. Ollama's default is 4096 tokens (Ollama FAQ), and the
  `/v1` API has no per-request `num_ctx` (Ollama OpenAI-compatibility docs), so set
  `OLLAMA_CONTEXT_LENGTH` for the server or `PARAMETER num_ctx` in a Modelfile (Ollama FAQ). 4096 is
  too small for a review; the next section says why.
- The output cap goes out as `max_tokens`, because Ollama's `/v1` request has no
  `max_completion_tokens` field (`openai/openai.go` in ollama/ollama); a request carrying only that
  field runs unbounded.
- Use Ollama 0.31.0 or newer (ollama/ollama PR #16428): 0.30.x reported only the newly processed
  prompt tokens, so a repeated prefix (the system prompt is the same on every hunk) came back as a
  smaller count and trips the truncation check; the version is inferred from the merge date
  (2026-06-02) and the first tag after it (`v0.31.0`, 2026-06-29), not from release notes.

## llama-server

```bash
llama-server -m model.gguf -c 65536 --host 127.0.0.1 --port 8080
# optional: add --api-key <k> above and then: export OPENAI_API_KEY=<k>
export TRELIX_LLM_PROVIDER=openai
export TRELIX_LLM_BASE_URL=http://127.0.0.1:8080/v1
export TRELIX_LLM_MODEL=<model>          # sent as the request's "model" field
export TRELIX_LLM_LOCAL_CONTEXT_TOKENS=65536
export TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET=null  # else the explicit 12000 budget wins
```

- `--jinja` (chat templates) is on by default in current builds, so no flag for it (llama.cpp
  `tools/server`).
- A prompt that does not fit `-c` is an HTTP 400, not a silent cut (llama.cpp `tools/server`). trelix
  reports it per hunk as `error` with `detail: exception:BadRequestError`, so the truncation check
  has nothing to do there; raise `-c`.

## Context length and the retrieval budget

**Why 64k.** A review request is the system prompt, the hunk and up to 3,000 characters of retrieved
context. The reply cap is `TRELIX_REVIEW_MAX_TOKENS` (default `4096`), and a reply cut off at that cap
is retried once at four times the cap, capped at `16384`. A 4096-token window cannot hold the retry at
all; 64k holds the prompt, the retry and the planner call with room to spare.

**The retrieval budget.** An explicit `TRELIX_RETRIEVAL_CONTEXT_TOKEN_BUDGET` (default `12000`) is
used as is. With `null`, trelix derives the budget from the model's context window, and the
`context_windows` table knows no Ollama-style tag (`qwen3-coder:30b-a3b`, `gpt-oss:20b`,
`devstral-small-2:24b`, `llama3.1:8b`, `gemma3:270m` all miss; `mixtral-8x7b` happens to be listed at
32,000), so it logs the WARNING `Model 'qwen3-coder:30b-a3b' not recognized by context_windows —
falling back to 12,000 tokens` and uses 12,000 without the fraction. `TRELIX_LLM_LOCAL_CONTEXT_TOKENS`
supplies the window instead: with `TRELIX_LLM_LOCAL_CONTEXT_TOKENS=65536` and the default
`TRELIX_RETRIEVAL_CONTEXT_WINDOW_FRACTION` of `0.5` the budget is `int(65536 * 0.5) = 32768`, and the
log says `Context window 65536 from TRELIX_LLM_LOCAL_CONTEXT_TOKENS` (INFO). See
[CONFIGURATION.md](CONFIGURATION.md#model-aware-context-budget).

## What the truncation check catches and what it does not

- Ollama drops the head of a prompt longer than its context length (the system prompt goes first) and
  answers HTTP 200 with a normal finish reason. The only trace is `usage.prompt_tokens`. With
  `TRELIX_LLM_BASE_URL` set, `trelix review` compares that count with the cl100k_base count of what it
  sent: a reported count under `0.85` x the estimate marks the hunk `truncated` with
  `detail: prompt_truncated`; nothing is kept from the reply and the hunk is not retried (a larger
  output cap cannot help). Exactly at the floor is not truncated.
- What you see: the WARNING `Local server truncated the prompt: it reports 2100 prompt tokens, trelix
  sent about 6000 (cl100k_base); the hunk is reported as truncated` (the two numbers are the server's
  count and trelix's estimate; never the prompt), then on stderr
  `c.py:1 was not reviewed (truncated: prompt_truncated)` per hunk and, at the end,
  `4 of 5 hunks could not be fully reviewed`.
- What it misses: a server that fills its whole window hides an overflow under 15%. Current Ollama
  truncates to about half the window (ollama/ollama issue #17427), which is caught. The estimate uses
  cl100k_base; the Qwen and Llama tokenizers are a few percent more efficient on code, so the margin
  above `0.85` is thin, and there is no knob: the floor is a constant.
- A reply without `usage` is never marked truncated, and the WARNING `Local server reports no token
  usage; the prompt-truncation check is off for this run` is logged once per backend; a later reply
  that does carry `usage.prompt_tokens` is still checked.
- Hosted clients, and a server reached through the SDK's `OPENAI_BASE_URL` instead of
  `TRELIX_LLM_BASE_URL`, never get the signal.
- Exit codes: when every hunk was cut, nothing came of the review and the exit code is 3; otherwise
  `TRELIX_REVIEW_MAX_UNREVIEWED_FRACTION` (default `0.0`) decides between 0 and 4. Nothing is salvaged
  from a prompt-truncated reply: the model never saw the whole hunk.

## Model size: the 20B floor

Each LLM client built against the URL reads the model tag once and logs one WARNING when the largest
`<number>b` / `<number>m` token in the tag is under 20B:

```
Local model 'qwen2.5-coder:7b' is about 7B parameters, under the 20B floor docs/OFFLINE.md assumes for trelix review; expect more unreviewed hunks and weaker findings.
```

or when the tag has no readable size:

```
Local model 'glm-4.5-air': no parameter count in the tag, so trelix cannot tell whether it meets the 20B floor docs/OFFLINE.md assumes.
```

| Tag | Read as | Result |
|---|---|---|
| `qwen2.5-coder:7b` | 7B | warning (under the floor) |
| `qwen3-coder:30b-a3b` | 30B (the active-parameter `a3b` is not the size) | none |
| `gemma3:270m` | 0.27B | warning (under the floor) |
| `gpt-oss:20b` | 20B | none (at the floor is not under it) |
| `glm-4.5-air` | no size | warning (no readable size) |
| `mixtral-8x7b` | no size (an `NxMb` mixture tag is not read as N x M) | warning (no readable size) |

A `review` on an indexed repository builds two clients (the query planner's and the reviewer's), so
the line appears twice. The command runs regardless.

The floor is an assumption, not a measurement. R-C4-05, the bake-off on real hardware, sets the bar: a
model is "supported" iff unreviewed hunks are <= 10% and judged precision is within 15 points of the
hosted baseline on the golden PR set. Candidates nobody has measured yet: Qwen3-Coder-30B-A3B
(Q4_K_M), gpt-oss-20b and Devstral Small 2. None of them is "supported" until the bake-off says so.

## Every search also plans through the server

With `TRELIX_LLM_BASE_URL` set, the keyless client is a usable client, and the query planner uses it:
`trelix search`, `trelix query`, `trelix ask` and `trelix review` on an indexed repository send one
planner tool call per distinct query (`max_tokens=512`, a 30 s timeout, up to five attempts with
1-60 s full-jitter waits between them, that is four retries, before falling back to
`default_plan()`). A 7B model on a CPU that needs more than 30 s per plan pays the timeouts first.
`TRELIX_RETRIEVAL_PLAN_CACHE_FILE` records each plan once and replays it with zero calls (see
[FAQ.md](FAQ.md#what-is-trelix-query)), for a fixed query set only: a file that already holds a
plan when the command starts is replayed and nothing else, and a query not in it is refused with
`PlanCacheMissError` (the command fails) rather than planned, so the cache fits a CI job or a
golden set, not ad-hoc `ask`; delete the file to re-record it. A repository without an index
makes no planner call: `trelix review` on it reviews from the diff alone.

## Prefetch

Four one-time steps while online; the three warnings that mention this guide (`docs/OFFLINE.md
assumes`, `docs/OFFLINE.md: prefetch`) point here.

1. The embedder. `pip install "trelix[local]"`, then the command CI uses to warm the model cache, and
   `HF_HUB_OFFLINE=1` for every later run (Hugging Face Hub environment variable):

   ```bash
   pip install "trelix[local]"
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
   export HF_HUB_OFFLINE=1
   ```

2. The grammars. `tree-sitter-language-pack` does not bundle them in its wheel; each is downloaded on
   first use:

   ```bash
   python -c "from trelix.indexing.parser._grammar import prefetch_all; prefetch_all()"
   ```

3. tiktoken. The chunker, the context assembler and the review's prompt-truncation check all load the
   `cl100k_base` encoding, which tiktoken downloads on first use and caches as `TIKTOKEN_CACHE_DIR`,
   else `DATA_GYM_CACHE_DIR`, else `<tempdir>/data-gym-cache` (`tiktoken/load.py`); the temp directory
   is wiped. Prefetch once and keep `TIKTOKEN_CACHE_DIR` exported in every later shell:

   ```bash
   TIKTOKEN_CACHE_DIR=$HOME/.cache/trelix-tiktoken python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
   export TIKTOKEN_CACHE_DIR=$HOME/.cache/trelix-tiktoken
   ```

   Without the file, indexing fails, and the check logs `tiktoken cl100k_base is not available
   (<exception class>); the prompt-truncation check is off for this run (docs/OFFLINE.md: prefetch)`
   once and stays off for the run: fail-open, so a missing cache file does not turn every hunk into
   exit 3. This prefetch step is the control.

4. Reranking. The default provider is `cohere`; with `COHERE_API_KEY` unset it is skipped with
   `COHERE_API_KEY is not set; skipping Cohere reranking.` and retrieval still works. To rerank
   locally, `TRELIX_RETRIEVAL_RERANK_PROVIDER=cross_encoder` needs `sentence-transformers` (the
   `local` extra) and downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` once; do it before
   `HF_HUB_OFFLINE=1`:

   ```bash
   python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
   ```

## Network and secrets

- Plaintext `http://` to a non-loopback host sends repository code (hunks and retrieved context)
  unencrypted across the network. Loopback is fine; for a LAN inference box use `https://` or a
  tunnel.
- The URL, the bearer and the prompt are never logged: the size warnings carry the model tag, the
  truncation warning two integers, the no-usage and no-encoder warnings fixed text and an exception
  class name.
- A credential pasted into the URL (`http://user:pw@host/v1`) is refused at configuration time and
  never echoed: the error names the variable only.
- `trelix-local` is a public constant, not a secret; a server that checks keys needs
  `OPENAI_API_KEY`.

## Troubleshooting

| What you see | Cause | Fix |
|---|---|---|
| `exception:NotFoundError` on every hunk | Missing `/v1` in the URL, or a tag the server has not pulled (`TRELIX_LLM_MODEL` left at the default `gpt-4o`) | Add `/v1`; `ollama pull <tag>` and set `TRELIX_LLM_MODEL` to it |
| `exception:AuthenticationError` | The server wants a key | Set `OPENAI_API_KEY` to the key the server expects (llama-server `--api-key`, a gateway) |
| `prompt_truncated` on every hunk | The server's context length is too small | Raise it: Ollama `OLLAMA_CONTEXT_LENGTH` or a Modelfile `PARAMETER num_ctx`; a truncating proxy: its own setting. Then set `TRELIX_LLM_LOCAL_CONTEXT_TOKENS` to match |
| `exception:JSONDecodeError` | The server answered 200 with a body labelled `application/json` that is not parseable JSON (a gateway or proxy stub, a reply in another framing): not an OpenAI chat-completions endpoint | Point the URL at the chat-completions server itself |
| `exception:APIConnectionError`, and a run that takes minutes before failing | Server down or wrong port. Every hunk gets up to five attempts (four retries, 1-60 s full-jitter waits) before it is reported | Stop the run and fix the URL or start the server; do not wait it out |
| `exception:BadRequestError` on llama-server | The prompt exceeds `-c` | Start the server with a larger `-c` |
| `Configuration error: llm -> base_url: ...`, exit 1 | The URL shape: no scheme (`localhost:11434`), a credential, a bad port, whitespace or a trailing newline from a secret store | Use `http://host:port/v1`; the message names the variable, never the value |
| `TRELIX_LLM_BASE_URL is set but TRELIX_LLM_PROVIDER=... does not use it` | Only the `openai` backend reads the URL | Set `TRELIX_LLM_PROVIDER=openai`, or unset the URL |
| `tiktoken cl100k_base is not available (...)` | No cached encoding and no network | Run the tiktoken prefetch above and keep `TIKTOKEN_CACHE_DIR` exported |
| Every `search` or `ask` waits before showing results | The planner's tool call is slower than 30 s and is retried | For a fixed query set (CI, a golden set) set `TRELIX_RETRIEVAL_PLAN_CACHE_FILE` and record the plans once; a query not in the file is then refused (`PlanCacheMissError`), so for ad-hoc questions accept the planner latency |
