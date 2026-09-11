# Migrating from trelix v3.x to v4.0.0

**Scope:** v3.2.5 → v4.0.0. v4.0.0 is trelix's only release allowed to make breaking
changes (see [BACKWARDS_COMPATIBILITY.md](../BACKWARDS_COMPATIBILITY.md)); this guide is
a living document, updated as each v4.0.0 change lands rather than written once at the
end. Check `docs/ROADMAP.md` for what's still in progress.

---

## Required: rename `TRELIX_RETRIEVAL_FLARE_MAX_ITER`

**What changed:** the env-var alias `TRELIX_RETRIEVAL_FLARE_MAX_ITER` — deprecated since
v2.4.0, still honoured with a `DeprecationWarning` through all of v3.x — is now removed
outright. Setting it has no effect: `flare_max_retries` falls back to its default (`1`)
instead of reading the old name, and no warning is emitted.

```bash
# Before (v3.x — still worked, warned)
export TRELIX_RETRIEVAL_FLARE_MAX_ITER=2

# After (v4.0.0 — the only name that binds)
export TRELIX_RETRIEVAL_FLARE_MAX_RETRIES=2
```

If you set this in Python rather than via the environment, nothing changes — the field
itself (`flare_max_retries`) was already renamed in v2.4.0 and is unaffected here:

```python
RetrievalConfig(flare_max_retries=2)  # unchanged
```

**Action required:** grep your deployment's environment/config for
`TRELIX_RETRIEVAL_FLARE_MAX_ITER` and rename it. If you already migrated per
[v2-to-v3.md](v2-to-v3.md)'s checklist, there is nothing to do.

---

## Not breaking: Python floor raised to `>=3.12`

trelix's `requires-python` moved from `>=3.11` to `>=3.12` (root package and all three
adapter packages: `trelix-mcp`, `trelix-langchain`, `trelix-llama-index`). This is a
packaging-level change, not an API change — CI has matrix-tested 3.11 through 3.14
identically for several releases, and the published Docker image has shipped
`python:3.14-slim` since before this bump. If you run trelix on Python 3.11, upgrade your
interpreter before upgrading the package; everything else about your integration is
unaffected.

---

## Not breaking: eight dependency floors bumped past their own breaking majors

`tree-sitter`, `tree-sitter-language-pack`, `numpy`, `pathspec`, `typer`, `rich`,
`pydantic`, and `tenacity` all had their own breaking major-version releases since
trelix's previous floors — each was individually confirmed (by reading trelix's actual
call sites, not by assumption) to touch none of the removed/changed APIs. Nothing in
trelix's public surface changes as a result. See
`docs/reports/v4-0-0-upgrade-research-2026-09-11.md` for the per-dependency evidence.

---

## Still to come

The rest of v4.0.0's scope (LLM provider abstraction changes, MCP protocol updates) is
tracked in `docs/ROADMAP.md` and not yet released. This guide will grow a section for
each as it ships.
