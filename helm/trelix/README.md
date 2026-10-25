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

`trelix serve` runs with no authentication unless you set `apiAuth.token`
(or `apiAuth.existingSecretName`), which then requires an `X-Trelix-Api-Key`
header on every route except `/health`. This chart defaults
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

## trelix-mcp over Streamable HTTP (optional)

`mcp.enabled: true` adds a second container, `trelix-mcp`, to the same pod. It shares the
PVC and serves the repositories under `persistence.mountPath` (its `--root`) over MCP's
Streamable HTTP transport, on Service port `mcp` (`mcp.port`). It is off by default, and
with it off the render is unchanged.

```bash
helm install my-trelix helm/trelix --set mcp.enabled=true --set mcp.auth.existingSecretName=my-mcp-token
kubectl port-forward svc/my-trelix 8766:8766
export TRELIX_MCP_AUTH_TOKEN=<the value stored in the my-mcp-token Secret>
claude mcp add --transport http trelix http://127.0.0.1:8766/mcp --header "Authorization: Bearer $TRELIX_MCP_AUTH_TOKEN"
```

- **Token (required).** Set `mcp.auth.token` or `mcp.auth.existingSecretName`. It is not
  `apiAuth`'s: the REST and MCP tokens are separate. Create the Secret with
  `kubectl create secret generic my-mcp-token --from-literal=TRELIX_MCP_AUTH_TOKEN=<token>`:
  a trailing newline makes the token non-printable and the container exits 2. Every request
  except `GET /health` must carry it.
- **Refusals.** `helm template` and `helm install` fail when `mcp.enabled` has no token, or
  when `persistence.enabled` is false (there is nothing to serve). `helm lint` reports it
  but exits 0.
- **Host check.** The container accepts the `Host` names `<fullname>`,
  `<fullname>.<namespace>` and `<fullname>.<namespace>.svc` (the release's Service names),
  plus `mcp.allowedHosts` and the loopback names (so a `kubectl port-forward` works); any other
  name answers 403. An Ingress host or the
  `.svc.cluster.local` form goes in `mcp.allowedHosts`, which is a list:
  `--set 'mcp.allowedHosts={trelix.example.com}'`. A plain string fails with
  `Cannot concat type string as list`. A `*` entry is dropped, so the chart
  cannot turn the check off, and so is any entry that is not one hostname or IP address (a
  URL, a wildcard): the container logs a warning (`kubectl logs deploy/<fullname> -c
  trelix-mcp`) and requests under that name answer 403. The probes use `GET /health`, which
  is exempt. The 403 text names the REST API's allow-list variable, which trelix-mcp reads
  only when its own is unset or blank, so edit `mcp.allowedHosts`, not `extraEnv`.
- **Image.** `image.tag` must be a release whose `trelix-mcp` has `--transport http`; on an
  older image the container exits at once on an argument error and crash-loops, and the
  whole pod, REST API included, stays NotReady and leaves the Service. Check an image with
  `docker run --rm --entrypoint trelix-mcp <image> --help`.
- **One replica.** Keep `replicaCount: 1`: the PVC is `ReadWriteOnce`, and both containers
  share one SQLite index and its single writer.
- **Ingress.** The chart's Ingress routes to the REST port only; point your own Ingress or
  Gateway at Service port `mcp`. Responses are `text/event-stream`: keep proxy buffering off
  (ingress-nginx's default; the annotation `nginx.ingress.kubernetes.io/proxy-buffering`
  turns it on) and set the read timeout to at least your longest `index_codebase`, e.g.
  `nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"` (seconds, unitless). TLS is the
  proxy's job. `service.type: LoadBalancer` or `NodePort` exposes the `mcp` port too. See
  [MCP_GUIDE section 19](../../docs/MCP_GUIDE.md#19-streamable-http-transport).

## Values

See `values.yaml` for the full list with defaults and comments. Highlights:

| Key | Default | Description |
|---|---|---|
| `image.tag` | `3.4.3` | Always equals `Chart.yaml`'s `appVersion`. Set to `X.Y.Z-local` to use the variant bundling `sentence-transformers`/torch for the local embedder |
| `mcp.enabled` | `false` | Add the `trelix-mcp` container (MCP over Streamable HTTP) to the pod — see "trelix-mcp over Streamable HTTP" above. Needs a token and `persistence.enabled` |
| `mcp.port` | `8766` | Container and Service port (`mcp`) of `trelix-mcp` |
| `mcp.allowedHosts` | `[]` | `Host` names `trelix-mcp` accepts besides the release's Service names (an Ingress host, the `.svc.cluster.local` form) |
| `mcp.auth.token` / `mcp.auth.existingSecretName` | `""` | The bearer token of `trelix-mcp`, which is not `apiAuth`'s; set one. Prefer `existingSecretName` in any shared cluster |
| `mcp.resources` | `{}` | Resources of the `trelix-mcp` container only |
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

## The two versions in `Chart.yaml`

`appVersion` is trelix's version and must equal `image.tag` and the release
tag — `release.yml`'s `verify-version` job fails the release if it does not.
`version` is the *chart's* version and moves independently, on its own SemVer
track, whenever the chart's templates or defaults change; nothing checks it,
because there is no chart to check against: the chart is installed from this
tree (`helm install my-trelix helm/trelix`, as above) and CI only lints and
templates it — `helm-lint.yml` never `helm package`s or pushes it to a repo.

The two are not, and should not become, lockstep. Bumping `version` in step
with `appVersion` would announce a chart change on every trelix release,
including the releases that do not touch `helm/` at all.

## Linting and rendering locally

```bash
helm lint helm/trelix
helm template test helm/trelix
helm template test helm/trelix --set store.backend=qdrant --set store.qdrant.apiKey=x
helm template test helm/trelix --set store.backend=lance
helm template test helm/trelix --set ingress.enabled=true
helm template test helm/trelix --set mcp.enabled=true --set mcp.auth.token=ci-test-placeholder
```
