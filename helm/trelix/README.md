# trelix Helm chart

Deploys `trelix serve`'s REST API to Kubernetes.

```bash
helm install my-trelix helm/trelix
```

## Repo model — read this before deploying

`trelix serve`'s REST API is stateless per request: `create_app()` takes zero
arguments, and every route independently re-derives its config from the
request's own `repo` query param (confirmed directly against
`src/trelix/api/app.py` — the CLI's `serve <repo_path>` argument is never
passed into `create_app()`; it's only echoed in the startup log line).

This means **one Deployment can serve many repos**, not just one. This chart
models that directly: `persistence.enabled` (default `true`) mounts one PVC
at `persistence.mountPath` (default `/data`), shared across every repo you
index/serve. Pass `repo=/data/<name>` (an absolute path inside that volume)
on every REST call. See `templates/NOTES.txt` (also shown after
`helm install`/`helm upgrade`) for the exact workflow.

If you'd rather have `serve` genuinely pinned to one repo per Deployment,
that requires an application code change (wiring the CLI's `repo_path`
argument into `create_app()`), which is out of scope for this chart — it's a
tracked follow-up, not something to route around at the Helm layer.

## Security — no auth on the REST API

`trelix serve` has zero auth middleware. This chart defaults
`ingress.enabled: false` for exactly that reason — a public Ingress with no
fronting auth exposes:

- `POST /index` — arbitrary-path indexing
- `GET /ask` — LLM synthesis at your API key's cost
- `GET /search` — full source/body content of every indexed repo

Before setting `ingress.enabled: true`, put an authenticating layer in front
(an ingress-controller auth annotation, e.g. NGINX
`nginx.ingress.kubernetes.io/auth-*`, or an OAuth2-proxy sidecar/gateway).
This chart does not automate that setup — only documents the requirement.

## Qdrant is external, not managed by this chart

Setting `store.backend: qdrant` only configures trelix to *connect* to an
existing Qdrant instance (`store.qdrant.url`) — this chart never deploys or
operates Qdrant itself. Qdrant's own Helm chart states support for it is
"limited to community support," and self-hosted Qdrant lacks zero-downtime
upgrades, backup/DR, and shard rebalancing (Cloud/Enterprise-Operator-only
features per Qdrant's own comparison table). If you choose this backend,
Qdrant's availability and durability are entirely your operational
responsibility — use Qdrant Cloud or a self-managed cluster you're prepared
to run.

## Values

See `values.yaml` for the full list with defaults and comments. Highlights:

| Key | Default | Description |
|---|---|---|
| `image.tag` | `2.8.1` | Set to `X.Y.Z-local` to use the variant bundling `sentence-transformers`/torch for the local embedder |
| `store.backend` | `sqlite` | `sqlite` \| `qdrant` \| `lance` |
| `persistence.enabled` | `true` | Shared multi-repo PVC — see "Repo model" above |
| `persistence.mountPath` | `/data` | |
| `ingress.enabled` | `false` | See "Security" above before enabling |
| `store.qdrant.existingSecretName` | `""` | Prefer this over `store.qdrant.apiKey` in any shared cluster |
| `embedder.openai.existingSecretName` | `""` | Prefer this over `embedder.openai.apiKey` in any shared cluster |

Each of `store.qdrant`, `embedder.openai`, `embedder.voyage`,
`embedder.cohere` supports either a plaintext `apiKey` (dev/local
convenience — the chart creates one Secret from it) or
`existingSecretName`/`existingSecretKey` pointing at a Secret you manage
yourself. Set at most one of the two per credential.

## Linting and rendering locally

```bash
helm lint helm/trelix
helm template test helm/trelix
helm template test helm/trelix --set store.backend=qdrant --set store.qdrant.apiKey=x
helm template test helm/trelix --set store.backend=lance
helm template test helm/trelix --set ingress.enabled=true
```
