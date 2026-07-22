# trelix PR review — GitHub App and Actions workflow

Automatic PR review as a GitHub Check run with inline annotations on every
pull request. Two integration paths exist, covering the same review
capability:

| | GitHub Actions workflow | GitHub App (this directory) |
|---|---|---|
| Setup | Merge one YAML file into the repo | Install the App — no workflow file needed |
| Trust | Repo's own `GITHUB_TOKEN`, no third party | Grants a third-party App `pull_requests`/`checks`/`contents` access |
| Where reviews run | The installing repo's own Actions runners | This standalone service |
| Status | ✅ Shipped | 🚧 Skeleton only (item 6a) — not yet safe for production, see below |

Pick the Actions workflow if you'd rather not install a third-party App.
Pick the App once it's production-ready (6b/6c) for zero-setup
installability across many repos.

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

### Status: skeleton only (item 6a of the v3.0.0 roadmap)

This item ships the manifest, webhook routing, and review-runner shell-out
— **not yet wired for production use**:

- ❌ **No signature verification.** `src/webhook.ts` does not verify
  `X-Hub-Signature-256` against the webhook secret. **Do not expose this
  service on a public endpoint as-is.**
- ❌ **No installation-token minting.** `src/auth.ts`'s
  `getInstallationToken` is stubbed and throws.
- ❌ **No Check-annotation posting.** `runReview` returns parsed findings;
  nothing calls `github.rest.checks.create` yet.

All three land in item 6b (GitHub App auth + signature verification).

### Files

- `manifest.yml` — GitHub App manifest for the [manifest registration
  flow](https://docs.github.com/en/apps/sharing-github-apps/registering-a-github-app-from-a-manifest).
  Declares the same three permissions the Actions workflow already uses
  (`pull_requests: write`, `checks: write`, `contents: read`) and
  subscribes to the `pull_request` event.
- `src/server.ts` — Express entry point (`/health`, `/webhooks/github`).
- `src/webhook.ts` — routes `pull_request` `opened`/`synchronize`/
  `reopened` deliveries (mirrors the Actions workflow's trigger), invokes
  the review runner.
- `src/review-runner.ts` — shells out to `trelix review --pr ... --json`
  and maps findings to GitHub Check annotations (`toAnnotations`, a
  TypeScript port of the same mapping logic in `trelix-review.yml`'s
  `github-script` step).
- `src/auth.ts` — installation-token minting, **stubbed** (item 6b).
- `src/config.ts` — reads `GITHUB_APP_ID`/`GITHUB_APP_PRIVATE_KEY`/
  `GITHUB_WEBHOOK_SECRET` from env only, per this repo's "never hardcode
  secrets" convention.

### Registering the App

1. Edit `manifest.yml`: replace the placeholder `https://trelix.example.com`
   URLs with your deployed service's real HTTPS origin.
2. Register via the [manifest flow](https://docs.github.com/en/apps/sharing-github-apps/registering-a-github-app-from-a-manifest)
   (create a temporary HTML form that POSTs the manifest JSON to
   `https://github.com/settings/apps/new`, or your organization's
   equivalent settings page).
3. GitHub redirects back with a one-time `code`; exchange it for the App's
   credentials (App ID, a generated private key, and — once you set a
   webhook secret in the App settings — the webhook secret).
4. Set `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY`, `GITHUB_WEBHOOK_SECRET` as
   environment variables (never commit them — see `.gitignore`'s `.env`
   entry).

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
