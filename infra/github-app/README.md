# trelix PR review — GitHub App and Actions workflow

Automatic PR review as a GitHub Check run with inline annotations on every
pull request. Two integration paths exist, covering the same review
capability:

| | GitHub Actions workflow | GitHub App (this directory) |
|---|---|---|
| Setup | Merge one YAML file into the repo | Install the App — no workflow file needed |
| Trust | Repo's own `GITHUB_TOKEN`, no third party | Grants a third-party App `pull_requests`/`checks`/`contents` access |
| Where reviews run | The installing repo's own Actions runners | This standalone service |
| Status | ✅ Shipped | ✅ Installable and hardened (see "Status" below) |

Pick the Actions workflow if you'd rather not install a third-party App.
Pick the App for zero-setup installability across many repos.

## Option 1: GitHub Actions workflow (shipped)

1. The workflow at `.github/workflows/trelix-review.yml` triggers on every
   `pull_request` event (`opened`/`synchronize`/`reopened`)
2. trelix indexes the repository (local embedder — no API key needed)
3. `trelix review --pr owner/repo#N --json` fetches the diff and reviews
   each changed hunk
4. Findings are posted as GitHub Check annotations with file + line
   references

### Quick setup

The workflow uses `GITHUB_TOKEN` (auto-provided by Actions) — no App
registration required.

1. Merge the PR that adds `.github/workflows/trelix-review.yml` to your repo
2. On the next pull request, the `trelix Code Review` check runs
   automatically

### Optional: richer reviews with an LLM provider

Set one of these repository secrets for LLM-powered synthesis:

| Secret | Provider |
|--------|---------|
| `OPENAI_API_KEY` | OpenAI GPT-4o |
| `ANTHROPIC_API_KEY` | Anthropic Claude |
| `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | AWS Bedrock |

Add under **Settings → Secrets and variables → Actions → Repository
secrets**, then update the workflow's `env:` block to pass the key.

### Behavior notes

- The index step has `continue-on-error: true` — if indexing fails
  (network-restricted CI, OOM), the workflow continues and posts an empty
  check rather than blocking the PR
- Review findings are capped at 50 annotations per PR (GitHub API limit)
- Works on private repos — `GITHUB_TOKEN` scopes are sufficient
- `trelix review` works without an LLM key (structural analysis only);
  synthesis requires a provider

### Permissions required

```yaml
permissions:
  pull-requests: write   # post PR comments
  checks: write           # post Check annotations
  contents: read          # checkout code
```

These are declared in the workflow YAML and require no manual
configuration.

---

## Option 2: standalone GitHub App (this directory's TypeScript service)

A webhook-driven App: install it on a repo and PR reviews happen
automatically, with **zero workflow YAML required in the installing
repository**. Its webhook handler runs `trelix review --pr` directly and
posts Check annotations via Octokit — no dependency on the installing repo
having any Actions workflow at all. This is the App's whole reason to
exist over the Actions workflow above: genuine zero-setup installability
across many repos, at the cost of installing a third-party App with real
permissions.

### Architecture

```
GitHub -- pull_request webhook -->  this service (Express)
                                       |
                                       v
                              trelix review --pr ... --json
                                       |
                                       v
                              GitHub Checks API (annotations)
```

### Status: installable, hardened, deployable on demand

- ✅ **Signature verification.** `src/webhook.ts` verifies
  `X-Hub-Signature-256` (HMAC-SHA256 over the raw request body, keyed by
  the webhook secret) via `@octokit/webhooks-methods`'s `verify()`, which
  compares using `crypto.timingSafeEqual` — not a naive string compare.
  Requests with a missing, wrong-secret, or body-tampered-after-signing
  signature are rejected with `401` before the route handler ever sees
  the payload.
- ✅ **Installation-token minting.** `src/auth.ts`'s `getInstallationToken`
  uses `@octokit/auth-app` (App-ID + private-key JWT signing ->
  installation-token exchange), with one `AuthInterface` reused per
  `AppConfig` so the library's own expiry-aware cache actually has a
  chance to hit across calls instead of re-minting on every request.
- ✅ **Clone-on-demand.** `src/repo-checkout.ts`'s `checkoutPullRequest`
  clones the *actual* PR being reviewed into a fresh, per-request temp
  workspace (`mkdtemp`, cleaned up in a `finally` block) using the minted
  installation token — there is no static, hardcoded repo path anymore.
  Fetches `refs/pull/<n>/head` against the base repo (never
  `--branch=<head.ref>`, which only exists on a fork's own repo for an
  external-contributor PR) and authenticates via a per-workspace
  `GIT_ASKPASS` script that reads the token from an env var — the token
  is never embedded in the remote URL, written to `.git/config`, or
  passed as a subprocess argument. Deliberately never passes
  `--recurse-submodules`: this service clones PRs from arbitrary external
  contributors, and an attacker-controlled `.gitmodules` is a real risk
  that diff-level review doesn't need to take on.
- ✅ **Check-annotation posting.** `runReview` mints a token, clones and
  indexes the PR's actual head, runs the CLI review against that clone,
  and posts a completed Check run with inline annotations via
  `octokit.rest.checks.create`.
- ✅ **Tolerant indexing.** `indexRepository` mirrors
  `trelix-review.yml`'s own `if ! trelix index .; then ::warning ...; fi`
  pattern — an indexing failure degrades findings to structural-only
  rather than blocking the review.
- ✅ **Redelivery backstop.** GitHub webhooks get no automatic retry past
  their own transient-failure window — a non-2xx response marks the
  delivery permanently failed. `src/scripts/redeliver-webhook-deliveries.ts`
  (via `getAppJwt`'s App-level JWT auth — distinct from every other call
  in this codebase, which uses an installation token) redelivers failed,
  sufficiently-aged deliveries. Runs on both a 6-hour schedule and
  `workflow_dispatch` — see "Deploying on Render" below.
- ✅ **Payload size limit.** The webhook route caps request bodies at 25MB
  — GitHub's own documented webhook payload cap — rejecting oversized
  bodies with `413` during parsing rather than buffering an arbitrarily
  large request into memory. This matters because signature verification
  happens *after* body parsing, so the size limit is the only defense
  against a sender who doesn't know the webhook secret sending a
  deliberately huge payload.
- ✅ **Subprocess timeout.** `runReviewCli` passes a 5-minute `timeout` to
  the `trelix review` shell-out; Node kills the child process (`SIGTERM`)
  and the call rejects if it hangs past that — a slow/stuck review no
  longer ties up server resources indefinitely.
- **Not claimed: GitHub Marketplace listing.** This App is installable
  and hardened, not Marketplace-verified — Marketplace paid-app listing
  has its own separate business/adoption requirements that are out of
  scope for this engineering work.

### Production deployment notes

- Run behind HTTPS (a reverse proxy or platform-provided TLS termination)
  — GitHub's webhook deliveries and the manifest's `hook_attributes.url`
  require it.
- `GITHUB_APP_PRIVATE_KEY`/`GITHUB_WEBHOOK_SECRET` must come from your
  platform's secret manager, never a committed file — `src/config.ts`
  reads them from env only and throws at startup if either is missing.
- `trelix` (the CLI) and a Python 3.12+ runtime must be present in the
  deployment image/environment — `review-runner.ts` shells out to it by
  name via `PATH`. `Dockerfile` (below) builds exactly this.
- Logs (`console.error`/`console.warn` on review/indexing failures)
  currently go to stdout/stderr only; wire your platform's log
  aggregation on top rather than expecting structured logging from this
  service directly.

### Files

- `manifest.yml` — GitHub App manifest for the [manifest registration
  flow](https://docs.github.com/en/apps/sharing-github-apps/registering-a-github-app-from-a-manifest).
  Declares the same three permissions the Actions workflow already uses
  (`pull_requests: write`, `checks: write`, `contents: read`) and
  subscribes to the `pull_request` event.
- `src/server.ts` — Express entry point (`/health`, `/webhooks/github`).
- `src/webhook.ts` — verifies `X-Hub-Signature-256`, then routes
  `pull_request` `opened`/`synchronize`/`reopened` deliveries (mirrors the
  Actions workflow's trigger), invokes the review runner.
- `src/review-runner.ts` — mints an installation token, clones the PR's
  actual head into a fresh workspace (`repo-checkout.ts`), indexes and
  reviews it via the `trelix` CLI, and posts the findings as a GitHub
  Check run (`toAnnotations`/`postCheckRun` — a TypeScript port of the
  same mapping logic in `trelix-review.yml`'s `github-script` step).
- `src/repo-checkout.ts` — clones a single PR's head into a per-request
  temp workspace via an installation token; see "Clone-on-demand" above
  for the security properties this enforces.
- `src/auth.ts` — installation-token minting (`getInstallationToken`) and
  App-level JWT minting (`getAppJwt`, used only by the redelivery script)
  via `@octokit/auth-app`, one cached `AuthInterface` per `AppConfig`.
- `src/scripts/redeliver-webhook-deliveries.ts` — the redelivery backstop;
  run via `npm run redeliver-failed-webhooks` or
  `.github/workflows/redeliver-failed-webhooks.yml`.
- `src/config.ts` — reads `GITHUB_APP_ID`/`GITHUB_APP_PRIVATE_KEY`/
  `GITHUB_WEBHOOK_SECRET` from env only, per this repo's "never hardcode
  secrets" convention.
- `Dockerfile` — builds this service + the `trelix` CLI it shells out to
  into one image; see "Deploying on Render" below.
- `render.yaml` — Render Blueprint for the free-tier deploy described
  below.

### Registering the App

1. Edit `manifest.yml`: replace the placeholder `https://trelix.example.com`
   URLs with your deployed service's real HTTPS origin.
2. Register via the [manifest flow](https://docs.github.com/en/apps/sharing-github-apps/registering-a-github-app-from-a-manifest)
   (create a temporary HTML form that POSTs the manifest JSON to
   `https://github.com/settings/apps/new`, or your organization's
   equivalent settings page).
3. GitHub redirects back with a one-time `code` (no `redirect_url`/
   `setup_url` is configured, so this is a plain query-string param on
   GitHub's own confirmation page, not a redirect into this service).
   Exchange it directly for the App's credentials:
   ```bash
   curl -X POST "https://api.github.com/app-manifests/${CODE}/conversions"
   ```
   The response includes the App ID, a generated private key, and (once
   you set a webhook secret in the App's settings) is where you'll also
   find the webhook secret.
4. Set `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY`, `GITHUB_WEBHOOK_SECRET` as
   environment variables (never commit them — see `.gitignore`'s `.env`
   entry).

### Deploying on Railway

Migrated off Render: Render fronts every service through Cloudflare
non-configurably, and Cloudflare's WAF blocked real GitHub `pull_request`
webhooks whose body contained code-like content (curl examples, code
fences) — a documented, unresolved false-positive class, not fixable from
our side. Railway has no default content-inspecting WAF, so this class of
failure can't recur there. `render.yaml`/the section this replaced is kept
only as a historical fallback reference until Render is fully decommissioned.

`Dockerfile` (same one, unchanged) builds this service and the `trelix`
CLI it shells out to into one image. Configure via the Railway dashboard
or CLI — **do not** add a `railway.json`/`railway.toml`; that config-as-code
format is deprecated with a 2026-12-01 cutoff in favor of a new
TypeScript/Python/Go IaC system, and our config needs are modest enough to
skip it entirely:

- **Root Directory:** repo root (blank/`.`).
- **Dockerfile path:** `infra/github-app/Dockerfile`, set via the
  `RAILWAY_DOCKERFILE_PATH` service variable (or the dashboard's
  Dockerfile-path field) — this also switches the effective builder to
  `DOCKERFILE` automatically; there's no separate "Docker" builder enum
  value to pick.
- **Branch:** `develop`, matching this repo's deploy branch elsewhere.
- **Health check:** path `/health`, generous timeout (300s) to cover a
  cold start plus a real `git clone` + `trelix index`/`review` burst.
- **Sleep (Serverless):** the Free plan *requires* `sleepApplication: true`
  for any service without a cron schedule — it cannot be disabled short of
  upgrading off Free. This means the same cold-start-drops-a-delivery risk
  Render's 15-minute spin-down had is unavoidable here too, so the same two
  mitigations apply, just repointed at the Railway URL:
  - **Keep-warm pinger — [cron-job.org](https://cron-job.org):** a free
    job hitting `https://<your-service>.up.railway.app/health` every
    10–14 minutes.
  - **Redelivery backstop:** `.github/workflows/redeliver-failed-webhooks.yml`
    (schedule + `workflow_dispatch`) — unchanged, hosting-agnostic. Needs
    `TRELIX_APP_ID`/`TRELIX_APP_PRIVATE_KEY` as repo secrets (Settings →
    Secrets and variables → Actions) — renamed from `GITHUB_APP_ID`/
    `GITHUB_APP_PRIVATE_KEY` in PR #337; the env var *names* `config.ts`
    itself reads are unchanged, only the GitHub Actions secret-store names
    changed.
- **Replica/Usage Limits:** set a generous (not aggressive) Replica Limit
  so a real review burst doesn't crash the instance, and only a *soft*
  (notify-only) Usage Limit — no hard billing cutoff that could kill the
  service unexpectedly. Configure via the dashboard; the underlying
  `usageLimitSet` mutation is billing-workspace-scoped, not project-scoped,
  so it's not something to script per-project.
- **Free-tier deploy freeze:** Railway blocks Free-plan deploys roughly
  8 AM–8 PM `America/Los_Angeles` daily, regardless of the service's own
  region. A push that would trigger an auto-deploy during that window
  simply fails until the window closes — worth knowing before assuming a
  merged PR redeployed.

Env vars (`GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY`, `GITHUB_WEBHOOK_SECRET`)
are the same shape as before. LLM synthesis uses `TRELIX_LLM_PROVIDER=azure`
plus `AZURE_API_KEY`/`AZURE_ENDPOINT` (reusing the main app's existing,
already-working Azure credentials) instead of a placeholder
`OPENAI_API_KEY` — no new credential was provisioned for this.

### Local development

```bash
npm install
npm run typecheck
npm run build
npm test

GITHUB_APP_ID=... GITHUB_APP_PRIVATE_KEY=... GITHUB_WEBHOOK_SECRET=... npm run dev
```

`trelix` (the CLI) must be installed and on `PATH` wherever this service
runs — `review-runner.ts` shells out to it directly.
