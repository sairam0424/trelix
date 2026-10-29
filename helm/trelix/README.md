# trelix Helm chart

Deploys `trelix serve`'s REST API to Kubernetes.

```bash
helm install my-trelix helm/trelix -n trelix --create-namespace
```

The chart sets no `metadata.namespace`, so `-n` decides where it lands; the `default` namespace
is the one Kubernetes security baselines (CIS 5.7.4, Checkov's `CKV_K8S_21`) tell you not to use.

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
This chart does not automate that setup — only documents the requirement. The opt-in
NetworkPolicy under "Hardening" below narrows who can reach the port at all.

The container reads the token once, when it starts, so after a rotation the old
token stays valid until the pod restarts. A new `apiAuth.token` restarts the pod
on `helm upgrade`; after you change the Secret behind `apiAuth.existingSecretName`
run `kubectl -n <namespace> rollout restart deployment/<fullname>`.

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
helm install my-trelix helm/trelix -n trelix --create-namespace --set mcp.enabled=true --set mcp.auth.existingSecretName=my-mcp-token
kubectl -n trelix port-forward svc/my-trelix 8766:8766
export TRELIX_MCP_AUTH_TOKEN=<the value stored in the my-mcp-token Secret>
claude mcp add --transport http trelix http://127.0.0.1:8766/mcp --header "Authorization: Bearer $TRELIX_MCP_AUTH_TOKEN"
```

- **Token (required).** Set `mcp.auth.token` or `mcp.auth.existingSecretName`. It is not
  `apiAuth`'s: the REST and MCP tokens are separate. Create the Secret with
  `kubectl -n trelix create secret generic my-mcp-token --from-literal=TRELIX_MCP_AUTH_TOKEN=<token>`:
  a trailing newline makes the token non-printable and the container exits 2. Every request
  except `GET /health` must carry it.
- **Rotating the token.** The container reads the token once, when it starts, so the old
  token stays valid until the pod restarts. With `mcp.auth.token`, `helm upgrade` restarts
  the pod by itself: the chart hashes the Secret it owns into the pod template
  (`checksum/secret`). With `mcp.auth.existingSecretName` the chart cannot see the Secret,
  so after you change it run `kubectl -n <namespace> rollout restart deployment/<fullname>`.
- **Refusals.** `helm template` and `helm install` fail when `mcp.enabled` has no token, or
  when `persistence.enabled` is false (there is nothing to serve). `helm lint` reports it
  but exits 0.
- **Host check.** The container accepts the `Host` names `<fullname>`,
  `<fullname>.<namespace>` and `<fullname>.<namespace>.svc` (the release's Service names),
  plus `mcp.allowedHosts` and the loopback names (so a port-forward works); any other
  name answers 403. An Ingress host or the
  `.svc.cluster.local` form goes in `mcp.allowedHosts`, which is a list:
  `--set 'mcp.allowedHosts={trelix.example.com}'`. A plain string fails with
  `Cannot concat type string as list`. A `*` entry is dropped, so the chart
  cannot turn the check off, and so is any entry that is not one hostname or IP address (a
  URL, a wildcard): the container logs a warning (`kubectl -n <namespace> logs
  deploy/<fullname> -c trelix-mcp`) and requests under that name answer 403. The probes use `GET /health`, which
  is exempt. The Host/Origin 403 text names `TRELIX_MCP_ALLOWED_HOSTS`, which the chart sets from
  `mcp.allowedHosts`; trelix-mcp reads the REST API's allow-list variable only when that one is
  unset or blank, so edit `mcp.allowedHosts`, not `extraEnv`.
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

## Hardening

Chart 0.3.0 turns on the settings that cannot break a working install, and leaves the ones that
can break one off, each behind a single value. The chart needs Kubernetes 1.19 or newer
(`kubeVersion` in `Chart.yaml`).

### On by default

| Value | Default | Effect |
|---|---|---|
| `podSecurityContext.seccompProfile.type` | `RuntimeDefault` | The container runtime's default syscall filter (`fsGroup: 1000` is unchanged) |
| `securityContext` | `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]` | On every container of the pod, `trelix-mcp` included |
| `serviceAccount.automount` | `false` | No Kubernetes API token is mounted in the pod. trelix never calls the Kubernetes API, so the token is only something a compromised process could use |
| `resources` | requests `cpu: 250m`, `memory: 512Mi` | See "Sizing" below. No limits |

**Upgrading from chart 0.2.x.** The next `helm upgrade` changes the pod, and rolls it. It now has no
service-account token, a `RuntimeDefault` seccomp profile, no privilege escalation, no
capabilities and CPU and memory requests. A sidecar or injector that authenticates with the pod's
token (a Vault Agent injector, for example) stops working until you set
`serviceAccount.automount=true`. `helm upgrade --reuse-values` fails to render, as it did for
`mcp` in 0.2.7, because the old release's values have no `networkPolicy`: use
`--reset-then-reuse-values`. If you already set `resources.limits` below the new requests (under
`250m` CPU or `512Mi` memory) and no `resources.requests`, the API server rejects the Deployment,
since a request may not exceed its limit: set `resources.requests` at or below your limits.

**Going back.** In a values file, set `serviceAccount.automount: true`, `securityContext: null`,
`resources: null` and `podSecurityContext.seccompProfile: null` (on the command line,
`--set securityContext=null`). `null` deletes a default; an empty map (`securityContext: {}`)
does not, because Helm merges maps and the defaults stay.

### Sizing

`resources` asks for `250m` CPU and `512Mi` memory and sets no limit. The requests are a
scheduling floor, not a measurement. There is no default limit on purpose: a memory limit that
is too low kills the pod in the middle of indexing a large repository (see "Large repos may hit
memory limits" in [MCP_GUIDE](../../docs/MCP_GUIDE.md)), and the `-local` image also loads torch
and an embedding model; a CPU limit throttles the bursty parse and embed phases. Raise the
requests for the `-local` image or a big repository, and add `resources.limits` only after you
have measured the largest repository you index. `mcp.resources` sizes the `trelix-mcp` container
and has no default.

### Opt-in hardening

These are off by default because each one can break an install that works today. `helm-lint.yml`
renders each of them, but this chart has not been run against a live cluster with them on; try
them in a staging namespace first.

**Read-only root filesystem.** `--set securityContext.readOnlyRootFilesystem=true`. A read-only
root alone starts the server and passes its probes, and then fails as soon as indexing needs a
temporary file or the cache directory, so the chart does the rest when it sees the flag: it
mounts an `emptyDir` at `/tmp` and another at `/cache` in every container, and sets
`XDG_CACHE_HOME=/cache`. The index itself lives under `.trelix/` inside each repository on the
PVC, which stays writable. That needs the PVC: with `persistence.enabled=false` only `/tmp` and
`/cache` are mounted, so a read-only root leaves nowhere writable for the index.
Both `emptyDir`s are empty after a pod restart, so the tree-sitter grammars (downloaded from
github.com on first use) are fetched again, and so is the embedding cache if you switched it on
(`TRELIX_EMBEDDING_CACHE_ENABLED`). Point `TRELIX_EMBEDDING_CACHE_DIR` at a directory on the PVC
through `extraEnv` to keep it. With the `-local` image the Hugging Face model cache follows
`XDG_CACHE_HOME` too, so the model weights are downloaded again after every restart; set `HF_HOME`
to a directory on the PVC through `extraEnv` to keep them. Do not set `XDG_CACHE_HOME` in
`extraEnv` as well: the name would appear twice. Anything else that writes under the image's home directory or its working
directory, `/repo`, fails. Two are known, and the list may not be complete. The MCP tools
`federation_add_repo` and `federation_remove_repo` save `~/.config/trelix/repos.json`. The audit
trail (`TRELIX_AUDIT_ENABLED`) keeps `.trelix/audit.db` in the working directory by default, which
is on the read-only root: the file cannot be created, the server logs `AuditStore init failed ...
auditing disabled`, keeps serving and records nothing. Set `TRELIX_AUDIT_DB_PATH` to a file on the
PVC, for example `/data/audit.db`, through `extraEnv`. The SSO principal store
(`TRELIX_OIDC_ENABLED`) uses the same file.

**Non-root.** `--set securityContext.runAsNonRoot=true --set securityContext.runAsUser=999 --set
securityContext.runAsGroup=999`. Set all three: the image's `USER` is the name `trelix`, and the
kubelet refuses `runAsNonRoot` without a numeric user ("image has non-numeric user (trelix),
cannot verify user is non-root"). 999 is the id the published images use, for both variants;
the Dockerfile creates the user with `useradd --system` and does not pin the id, so check
the image you run with `docker run --rm --entrypoint id <image>` before you rely on it. On
OpenShift the restricted SCC assigns the UID itself: leave `runAsUser` and `runAsGroup` unset,
and expect to unset the fixed `podSecurityContext.fsGroup` as well
(`--set podSecurityContext.fsGroup=null`). This chart has not been tried on OpenShift, so the
default `RuntimeDefault` seccomp profile is untested there too: if the SCC refuses the pod over
it, `--set podSecurityContext.seccompProfile=null` removes it.

**NetworkPolicy.** `--set networkPolicy.enabled=true` adds a policy of type `Ingress` only. The
pod then accepts connections on the REST port (and the `trelix-mcp` port when that is on) from
pods in the release namespace and from the peers you list in `networkPolicy.ingressFrom`, a list
of standard `NetworkPolicyPeer` entries (`namespaceSelector`, `podSelector`, `ipBlock`):

```yaml
networkPolicy:
  enabled: true
  ingressFrom:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: ingress-nginx
```

Egress is not restricted: tree-sitter grammars come from github.com and the embedder and Qdrant
endpoints are outbound. With `ingress.enabled`, or a `service.type` other than `ClusterIP`, the
chart refuses to render the policy without `ingressFrom`, since the policy would drop the traffic
that Ingress or Service exists to carry; for a `LoadBalancer` or `NodePort`, the peer is
whatever source address your cloud presents. Anything else outside the namespace (Prometheus in
another namespace, a Gateway you run yourself in front of the `mcp` port) needs an entry too.
The policy is only enforced by a CNI that implements NetworkPolicy, and the kubelet's probes
reach the pod from its own node, which most CNIs let through; check yours.

## Values

See `values.yaml` for the full list with defaults and comments. Highlights:

| Key | Default | Description |
|---|---|---|
| `image.tag` | `3.5.0` | Always equals `Chart.yaml`'s `appVersion`. Set to `X.Y.Z-local` to use the variant bundling `sentence-transformers`/torch for the local embedder |
| `mcp.enabled` | `false` | Add the `trelix-mcp` container (MCP over Streamable HTTP) to the pod — see "trelix-mcp over Streamable HTTP" above. Needs a token and `persistence.enabled` |
| `mcp.port` | `8766` | Container and Service port (`mcp`) of `trelix-mcp` |
| `mcp.allowedHosts` | `[]` | `Host` names `trelix-mcp` accepts besides the release's Service names (an Ingress host, the `.svc.cluster.local` form) |
| `mcp.auth.token` / `mcp.auth.existingSecretName` | `""` | The bearer token of `trelix-mcp`, which is not `apiAuth`'s; set one. Prefer `existingSecretName` in any shared cluster |
| `mcp.resources` | `{}` | Resources of the `trelix-mcp` container only |
| `store.backend` | `sqlite` | `sqlite` \| `qdrant` \| `lance` |
| `persistence.enabled` | `true` | Shared multi-repo PVC — see "Repo model" above |
| `persistence.mountPath` | `/data` | |
| `ingress.enabled` | `false` | See "Security" above before enabling |
| `serviceAccount.automount` | `false` | Mount a Kubernetes API token in the pod. Needed by a Vault-agent style injector — see "Hardening" |
| `podSecurityContext` | `fsGroup: 1000`, `seccompProfile: RuntimeDefault` | See "Hardening" |
| `securityContext` | `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]` | Applies to every container. Also the switch for `readOnlyRootFilesystem` and `runAsNonRoot` / `runAsUser` / `runAsGroup` — see "Hardening" |
| `resources` | requests `cpu: 250m`, `memory: 512Mi` | No limits by default — see "Sizing" |
| `networkPolicy.enabled` | `false` | Ingress-only NetworkPolicy — see "Hardening" |
| `networkPolicy.ingressFrom` | `[]` | `NetworkPolicyPeer` entries admitted besides the release namespace; required with `ingress.enabled` or a non-`ClusterIP` Service |
| `store.qdrant.existingSecretName` | `""` | Prefer this over `store.qdrant.apiKey` in any shared cluster |
| `embedder.openai.existingSecretName` | `""` | Prefer this over `embedder.openai.apiKey` in any shared cluster |

Each of `store.qdrant`, `embedder.openai`, `embedder.voyage`,
`embedder.cohere` supports either a plaintext `apiKey` (dev/local
convenience — the chart creates one Secret from it) or
`existingSecretName`/`existingSecretKey` pointing at a Secret you manage
yourself. Set at most one of the two per credential. Containers read these
once, when they start: changing a plaintext `apiKey` restarts the pod on
`helm upgrade` (the chart hashes the Secret it owns into the pod template as
`checksum/secret`), while a Secret you manage needs
`kubectl -n <namespace> rollout restart deployment/<fullname>` after you change it. The same
goes for a Secret named in `extraSecretEnv`: the chart cannot see it either, so
changing it never rolls the pod by itself.

## The two versions in `Chart.yaml`

`appVersion` is trelix's version and must equal `image.tag` and the release
tag — `release.yml`'s `verify-version` job fails the release if it does not.
`version` is the *chart's* version and moves independently, on its own SemVer
track, whenever the chart's templates or defaults change; nothing checks it,
because there is no chart to check against: the chart is installed from this
tree (`helm install my-trelix helm/trelix -n trelix --create-namespace`, as above) and CI
only lints and templates it — `helm-lint.yml` never `helm package`s or pushes it to a repo.

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
helm template test helm/trelix --set securityContext.readOnlyRootFilesystem=true --set networkPolicy.enabled=true
```
