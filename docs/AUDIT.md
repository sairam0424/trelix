# Trelix Audit Logging — Hash-Chained Request Trail

trelix can record one append-only, hash-chained audit entry per HTTP API
request: who did what, to which resource, with what outcome. New in v3.0.0.
Fully opt-in — with `TRELIX_AUDIT_ENABLED` unset no middleware is registered,
no `audit.db` is created, and the API behaves exactly as it did in v2.12.0.

---

## Enabling

```bash
export TRELIX_AUDIT_ENABLED=true
export TRELIX_AUDIT_DB_PATH=/var/log/trelix/audit.db   # optional, see below
trelix serve ./my-repo          # REPO_PATH is a required positional argument
```

No extra install is needed beyond `pip install 'trelix[serve]'` — the audit
store is stdlib `sqlite3`, and the recorder is a Starlette middleware.

Inspect the result with [`trelix audit list`](CLI_REFERENCE.md#trelix-audit-list),
[`trelix audit verify`](CLI_REFERENCE.md#trelix-audit-verify) and
[`trelix audit export`](CLI_REFERENCE.md#trelix-audit-export).

---

## Scope — read this before you rely on it

**Only the HTTP API surface is audited.** The recorder is an ASGI middleware on
the app built by `trelix serve`. Everything that does not travel through that
app is invisible to the audit trail:

| Surface | Audited? |
|---|---|
| HTTP requests to `trelix serve` (`/search`, `/ask`, `/index`, `/graph/*`, `/health`, …) | **Yes** — one entry per request |
| MCP tool calls (the `trelix-mcp` server) | **No** — no audit integration exists in that package |
| The internal agentic (ReAct) loop — its per-turn tool calls | **No** — only the enclosing HTTP request is recorded |
| CLI commands (`trelix search`, `trelix ask`, `trelix index`, …) | **No** |
| Direct library use (`Retriever(config).retrieve(...)`) | **No** |

If you need coverage of those paths, front them with the HTTP API — or write to
`AuditStore` yourself; it is a public class (`trelix.audit.AuditStore`) and
`AuditEvent` is a plain frozen dataclass.

The middleware is registered as the **outermost** middleware, deliberately, so
it observes the *final* status code — including a `401` produced by the auth
dependency and a `500` produced by an unhandled route error. `/health` is not
gated by auth but is still audited.

---

## Integrity model — tamper-EVIDENT, not tamper-PROOF

> **The chain detects corruption. It does not prevent it.** The hash chain *and*
> the anchor it is checked against both live inside the same `audit.db`. Anyone
> with write access to that file can rewrite a row, recompute every subsequent
> `entry_hash`, and update the anchor in the same transaction — after which
> `trelix audit verify` reports the chain as intact. What this protects against
> is *accidental or naive* damage: a stray `UPDATE`, a truncated file, a deleted
> row, a dropped tail. It is not a defence against a determined attacker who
> already has write access to the host.

Two gates make that detection work:

1. **Hash chain.** Each row stores `prev_hash` (the previous row's `entry_hash`;
   the genesis value is 64 zeros) and
   `entry_hash = sha256(prev_hash || canonical_json(content))`, where `content`
   is the eleven logical columns (everything except `id`, `prev_hash` and
   `entry_hash`) serialized with sorted keys and compact separators. Mutating a
   row breaks its own recomputed hash; deleting or reordering a middle row
   breaks the next row's `prev_hash` linkage and leaves an `id` gap.
2. **In-DB count/head anchor.** A chain alone cannot detect a deleted *tail* —
   the survivors still form a valid chain. So an `audit_meta` table holds
   `count` and `head_hash`, updated atomically with every append, and
   `verify` compares the live row count and head against it. This is
   bookkeeping, not an external root of trust: it lives in the same file.

**Hardening path** (none of this ships today — it is what you would add):

- Sign the head hash with a key held **outside** the DB (HMAC or asymmetric) and
  verify the signature independently, so recomputing the chain is not enough.
- Anchor the head hash to append-only/WORM storage — an object-lock bucket, a
  write-once log service, or a transparency log.
- Ship entries off-box continuously (see
  [Shipping to a SIEM](#shipping-to-a-siem)) and treat the off-box copy as the
  evidence of record; the local `audit.db` then only has to be good enough to
  detect divergence from it.
- Restrict filesystem permissions on `audit.db` to the serving user, and keep it
  out of any directory a developer routinely deletes.

---

## Why a separate `audit.db`

The code index (`.trelix/index.db`) is **disposable** — it is deleted and
rebuilt whenever embeddings change, a provider is swapped, or a schema
migration is easier to skip than to write. An audit trail that lived there
would evaporate with it.

So the audit log gets its own SQLite file, defaulting to
`<cwd>/.trelix/audit.db` and overridable with `TRELIX_AUDIT_DB_PATH`. For
anything resembling a real deployment, point it **outside the repo** (e.g.
`/var/log/trelix/audit.db`) so that `rm -rf .trelix` — a routine act when
rebuilding an index — cannot touch it.

The `principals` table written by OIDC just-in-time provisioning lives in the
same `audit.db`, for the same reason: a verified identity and its audit records
belong together and must survive re-indexing. See [SSO.md](SSO.md).

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `TRELIX_AUDIT_ENABLED` | `false` | Register the audit middleware. When `false`, no middleware is added and no `audit.db` is created. |
| `TRELIX_AUDIT_DB_PATH` | _(none)_ → `<cwd>/.trelix/audit.db` | Path to the separate audit DB. Parent directories are created on first open. |
| `TRELIX_AUDIT_LOG_QUERIES` | `false` | Store raw query text for `search`/`ask` actions instead of `sha256:<hex>` of it. **Currently has no observable effect through the HTTP API** — see [Query text](#query-text-and-log_queries). |
| `TRELIX_AUDIT_FAIL_CLOSED` | `false` | `true` re-raises an audit-write failure into the request; `false` logs a WARNING and lets the request proceed. See [Failure contract](#failure-contract). |
| `TRELIX_AUDIT_RETENTION_DAYS` | `365` | **Not implemented.** Accepted and validated-free (a negative value is accepted), but nothing in trelix reads it — there is no pruning job. Rotate or prune `audit.db` yourself. |

---

## Event schema

One row per request in `audit_log`:

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key, autoincrement. Gapless `1..N` — a gap is itself tamper evidence. |
| `ts` | TEXT | ISO-8601 UTC timestamp, stamped by the middleware when the response is produced. |
| `principal` | TEXT | `static-token` in local/open mode; `sub@iss` for an OIDC-authenticated caller. Never an email, token, or header value. |
| `action` | TEXT | Coarse verb — `search`, `ask`, `index`, or `admin` (see below). |
| `resource` | TEXT | The request **path** only (`/search`, `/graph/communities`). The query string is deliberately not recorded. |
| `outcome` | TEXT | `success`, `failure`, `denied`, or `error` (see below). |
| `status_code` | INTEGER | Final HTTP status, after every inner layer has run. |
| `client_ip` | TEXT | Peer address as Starlette reports it, or `null`. Not `X-Forwarded-For`-aware — behind a proxy this is the proxy. |
| `request_id` | TEXT | The `X-Request-Id` request header if present, else a generated `uuid4().hex`. |
| `trace_id` | TEXT | The active OpenTelemetry trace id (32-hex) when tracing is on, else `null`. Lets an audit row join to a trace — see [OBSERVABILITY.md](OBSERVABILITY.md). |
| `duration_ms` | INTEGER | Wall-clock request duration, integer milliseconds. |
| `detail` | TEXT | Free-form context. **Always `null` from the HTTP middleware** — it never persists the request line, headers, or query text. |
| `prev_hash` | TEXT | Previous row's `entry_hash`; 64 zeros for the first entry. |
| `entry_hash` | TEXT | `sha256(prev_hash ‖ canonical_json(content))`. |

Indexed on `ts` and on `(principal, ts)` — "everything this principal did in
this window" is the query the schema is optimized for.

**`action` values.** Mapped from the request path by prefix (`path == prefix` or
`path.startswith(prefix + "/")`):

| Path prefix | `action` |
|---|---|
| `/search` | `search` |
| `/ask` | `ask` |
| `/index`, `/parse` | `index` |
| anything else (incl. `/health`, `/graph/*`) | `admin` |

`trelix.audit` also exports an `ACTION_AUTH` (`"auth"`) constant and the
`EVENT_TYPE_QUERY` / `EVENT_TYPE_MUTATION` / `EVENT_TYPE_SECURITY` groupings,
but the HTTP middleware never emits them — they exist for callers writing their
own events.

**`outcome` values.** Derived from the status code:

| Status | `outcome` |
|---|---|
| `401`, `403` | `denied` |
| `>= 500` | `error` |
| other `>= 400` | `failure` |
| everything else | `success` |

An unhandled route exception is recorded as `status_code 500` / `error` and then
re-raised, so the normal error-response path is unchanged.

---

## Verifying the chain

```bash
trelix audit verify                                  # default DB
trelix audit verify --db /var/log/trelix/audit.db
```

- Exit **0** — `Audit chain intact.`
- Exit **1** — `Audit chain TAMPERED — first divergent entry id: <id>` on stderr.
  Entries *before* that id verified cleanly, so it is the exact place to start
  an investigation.

What each failure mode looks like:

| Damage | How it is caught | Reported id |
|---|---|---|
| A row's content was edited | Recomputed `entry_hash` ≠ stored | that row's id |
| A middle row was deleted or reordered | `prev_hash` no longer links / `id` gap | the first id out of sequence |
| The tail was truncated | Live count/head ≠ `audit_meta` anchor | the first missing id |

An empty log verifies as intact (exit 0).

**One honest caveat about exit 0:** if the DB cannot be *opened* at all (path is
a directory, permissions denied), `AuditStore` logs
`AuditStore init failed for <path>; auditing disabled: ...` at WARNING and
`verify` then prints `Audit chain intact.` and exits 0 — it verified nothing.
When you wire `verify` into CI or a cron job, alert on that WARNING too; the
exit code alone is not sufficient proof that a chain was read.

**Single writer per `audit.db`.** The head-hash read and the append are guarded
by an in-process `threading.Lock` only. SQLite will keep the *file* consistent
across processes, but two `trelix serve` processes pointed at the same
`audit.db` can each read the same head and append rows claiming the same
`prev_hash` — a logical divergence that `verify` will correctly report as
tamper. Give each serving process its own `TRELIX_AUDIT_DB_PATH` (and merge
downstream in your SIEM), or run a single writer.

---

## Exporting

```bash
trelix audit export > /var/log/trelix/audit.ndjson
```

One JSON object per line, oldest first (chain order), every column included —
`prev_hash` and `entry_hash` too, so a downstream consumer can re-verify the
chain independently.

```json
{"id": 1, "ts": "2026-08-13T09:14:02.115331+00:00", "principal": "static-token", "action": "search", "resource": "/search", "outcome": "success", "status_code": 200, "client_ip": "127.0.0.1", "request_id": "4f1c9e2b8a7d4c10", "trace_id": null, "duration_ms": 87, "detail": null, "prev_hash": "0000000000000000000000000000000000000000000000000000000000000000", "entry_hash": "5d71295c36283eaac08929a0bac81f68b025a3fb62ac69c0659d6d5ff95c06ee"}
```

`export` is a **snapshot of the whole table** — it re-emits every entry each
time it runs, and it materializes all rows in memory before printing the first
line. For repeated shipping, export incrementally:

```bash
# Append only entries newer than the last shipped id.
STATE=/var/lib/trelix/last_audit_id
LAST=$(cat "$STATE" 2>/dev/null || echo 0)
trelix audit export | jq -c --argjson last "$LAST" 'select(.id > $last)' \
  >> /var/log/trelix/audit.ndjson
trelix audit export | tail -1 | jq -r '.id' > "$STATE"
```

---

## Shipping to a SIEM

NDJSON is the export format precisely so that no bespoke HTTP transport has to
exist in trelix: Filebeat, Vector and Fluent Bit all tail newline-delimited
JSON natively, handle retry/backpressure, and already have a sink for whatever
SIEM you run.

The three configs below are **starting sketches, not verified against a live
collector** — check them against the version of the shipper you actually
deploy, especially sink names and TLS options.

**Filebeat**

```yaml
filebeat.inputs:
  - type: filestream
    id: trelix-audit
    paths:
      - /var/log/trelix/audit.ndjson
    parsers:
      - ndjson:
          target: ""
          add_error_key: true

output.elasticsearch:
  hosts: ["https://siem.internal:9200"]
```

**Vector**

```toml
[sources.trelix_audit]
type = "file"
include = ["/var/log/trelix/audit.ndjson"]

[transforms.trelix_audit_json]
type = "remap"
inputs = ["trelix_audit"]
source = '. = parse_json!(.message)'

[sinks.siem]
type = "elasticsearch"     # or splunk_hec / loki / datadog_logs / ...
inputs = ["trelix_audit_json"]
endpoint = "https://siem.internal:9200"
```

**Fluent Bit**

```ini
[INPUT]
    Name    tail
    Path    /var/log/trelix/audit.ndjson
    Tag     trelix.audit
    Parser  json
    DB      /var/lib/fluent-bit/trelix-audit.db

[OUTPUT]
    Name    es
    Match   trelix.audit
    Host    siem.internal
    Port    9200
    tls     On
```

Two things worth doing on the SIEM side: **dedupe on `entry_hash`** (it is a
content hash, so a re-export produces identical values), and alert on
`outcome: "denied"` bursts and on any gap in the `id` sequence.

---

## Failure contract

**Default (`TRELIX_AUDIT_FAIL_CLOSED=false`) — log and continue.** A failed
audit write logs `Audit append failed (non-fatal): ...` at WARNING on the
`trelix.audit` logger and the request proceeds normally. This is a deliberate
choice: a full disk, a locked DB, or a bad path must not be able to take the API
down. `AuditStore` construction follows the same rule — an unopenable path logs
`AuditStore init failed ...; auditing disabled` and every subsequent append
returns `False`.

The cost of that choice is that **audit rows can be lost silently**. If the
trail matters, alert on the `trelix.audit` WARNING logger; it is the only signal
that entries are going missing.

**`TRELIX_AUDIT_FAIL_CLOSED=true` — reject.** The write failure is re-raised
inside the middleware, so the request fails (surfacing as a `500` from the outer
error middleware). Use this where "no audit record" must mean "no service" —
i.e. a compliance deployment. Note the implication honestly: if `audit.db`
becomes unwritable, *every* request starts failing. Monitor free disk space and
the audit path before turning this on.

---

## Query text and `log_queries`

`TRELIX_AUDIT_LOG_QUERIES` defaults to `false` because developers paste secrets
into search queries — tokens, customer identifiers, snippets of production
data. With it off, `AuditStore` stores `sha256:<hex>` of the `detail` field for
`search`/`ask` actions instead of the text, which still lets you correlate
repeated identical queries without retaining any of them.

**Honest limitation:** the HTTP middleware always sets `detail=None` — it
deliberately never persists the request line or query string — so today *no*
query text and *no* hash is written through the API, and setting
`TRELIX_AUDIT_LOG_QUERIES=true` changes nothing you can observe. The hashing
path only applies to code that calls `AuditStore.append()` with a `detail`
directly. Do not enable this expecting queries to appear in `audit.db`.

---

## Audit vs. query telemetry vs. OTel tracing

Three different mechanisms, three different questions. They are complementary,
not alternatives — see [OBSERVABILITY.md](OBSERVABILITY.md) for the other two.

| | Audit log | Query telemetry | OTel tracing |
|---|---|---|---|
| Question it answers | Who did what, when, and was it allowed? | How did retrieval perform? | Where did the time go inside one query? |
| Enabled by | `TRELIX_AUDIT_ENABLED` | `TRELIX_TELEMETRY_ENABLED` | `TRELIX_OTEL_ENABLED` |
| Written to | separate `audit.db` (survives re-indexing) | `query_telemetry` in the **disposable** index DB | an OTLP collector |
| Granularity | one row per HTTP request | one row per `retrieve()` call | one span per pipeline stage/leg |
| Records identity | **yes** (`principal`) | no | no |
| Records query text | no (see above) | **yes**, verbatim | as span attributes |
| Integrity | hash-chained, append-only | none — plain rows, dropped with the index | none — sampled, ephemeral |
| Read by | `trelix audit list/verify/export` | `trelix telemetry` | your tracing backend |

The one place they join: when OTel tracing is enabled, an audit row carries the
`trace_id` of the request that produced it.

---

## Known limitations, in one place

1. **Tamper-evident, not tamper-proof** — chain and anchor share `audit.db`.
2. **HTTP API surface only** — MCP tool calls, the internal agent loop, CLI
   commands and direct library use are not audited.
3. **`TRELIX_AUDIT_RETENTION_DAYS` is not implemented** — nothing prunes; the DB
   grows without bound.
4. **`TRELIX_AUDIT_LOG_QUERIES` is inert on the HTTP path** — `detail` is always
   `null` there.
5. **Coarse events** — `action` is one of four path-derived verbs; no request
   body, no query string, no response size.
6. **`client_ip` is the peer address**, not an `X-Forwarded-For`-resolved client.
7. **Single-writer** — one process per `audit.db`; concurrent writers can diverge
   the chain.
8. **`verify` exits 0 on an unopenable DB** — alert on the WARNING, not just the
   exit code.
9. **`export` is a whole-table snapshot** held in memory, not a streaming cursor.
