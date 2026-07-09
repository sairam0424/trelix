# trelix GitHub App Integration

Automatic PR review as a GitHub Check run with inline annotations on every pull request.

## How it works

1. The workflow at `.github/workflows/trelix-review.yml` triggers on every `pull_request` event
2. trelix indexes the repository (local embedder — no API key needed)
3. `trelix review --pr owner/repo#N --json` fetches the diff and reviews each changed hunk
4. Findings are posted as GitHub Check annotations with file + line references

## Quick setup (GitHub Actions, no App registration needed)

The workflow uses `GITHUB_TOKEN` (auto-provided by Actions) — **no GitHub App registration required**.

1. Merge the PR that adds `.github/workflows/trelix-review.yml` to your repo
2. On the next pull request, the `trelix Code Review` check runs automatically

That's it. The workflow uses the built-in `GITHUB_TOKEN` with the permissions declared in the YAML.

## Optional: richer reviews with an LLM provider

Set one of these repository secrets for LLM-powered synthesis:

| Secret | Provider |
|--------|---------|
| `OPENAI_API_KEY` | OpenAI GPT-4o |
| `ANTHROPIC_API_KEY` | Anthropic Claude |
| `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` | AWS Bedrock |

Add under **Settings → Secrets and variables → Actions → Repository secrets**.

Then update the workflow `env:` block to pass the key:

```yaml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Behavior notes

- The index step has `continue-on-error: true` — if indexing fails (network-restricted CI, OOM), the workflow continues and posts an empty check rather than blocking the PR
- Review findings are capped at 50 annotations per PR (GitHub API limit)
- Works on private repos — `GITHUB_TOKEN` scopes are sufficient
- `trelix review` works without an LLM key (structural analysis only); synthesis requires a provider

## Permissions required

```yaml
permissions:
  pull-requests: write   # post PR comments
  checks: write          # post Check annotations
  contents: read         # checkout code
```

These are declared in the workflow YAML and require no manual configuration.
