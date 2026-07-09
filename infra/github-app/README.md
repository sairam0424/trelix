# trelix GitHub App Integration

This directory contains the GitHub App integration for automatic trelix
PR review as a Check run with inline annotations.

## Setup

1. The workflow at `.github/workflows/trelix-review.yml` handles everything
   automatically on every pull_request event.

2. Required permissions:
   - `pull-requests: write` — post comments
   - `checks: write` — post Check annotations
   - `contents: read` — checkout code

3. Optional: set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in repository
   secrets for LLM-powered synthesis (trelix review works without it
   but annotations are richer with an LLM provider configured).

## How it works

1. trelix indexes the PR's base branch
2. `trelix review --pr owner/repo#N --json` fetches the diff and reviews it
3. Findings are posted as GitHub Check annotations with file + line references
