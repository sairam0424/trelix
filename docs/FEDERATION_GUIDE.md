# Federation Guide — trelix v2.8.1

Complete guide to searching across multiple independently-indexed repositories in one query.

---

## What is Federation?

Federation lets you search across multiple independently-indexed repositories in a single query. Each repository is indexed on its own schedule and stored separately; the federation layer merges results at query time using Reciprocal Rank Fusion (RRF).

---

## Setup

Register repositories by alias and path:

```bash
trelix federation add myapp /path/to/myapp
trelix federation add infra /path/to/infra
trelix federation list
```

Unregister a repository (v2.8.0):

```bash
trelix federation remove myapp
```

Each repository must be independently indexed before federation queries will return results for it:

```bash
trelix index /path/to/myapp
trelix index /path/to/infra
```

---

## Searching All Repos

```bash
trelix search-all "how does authentication work"
```

Returns merged, deduplicated results from every registered repository, ranked by RRF score.

---

## Python API

```python
from trelix.federation.retriever import FederatedRetriever
from trelix.federation.registry import RepoRegistry, RepoEntry
from pathlib import Path

registry = RepoRegistry.load()  # loads ~/.config/trelix/repos.json
fed = FederatedRetriever(
    registry,
    max_workers=4,      # parallel thread count
    cache_ttl=120.0,    # cache identical queries for 120s
    max_repos=50        # query at most 50 repos (None = unbounded)
)
results = fed.retrieve("authentication", k=10)
```

`results` is a list of `SearchResult` objects, each carrying its source repository alias alongside the standard `file_path`, `symbol_id`, `score`, and `snippet` fields.

To check how many repos were actually queried vs skipped:

```python
total_registered = len(registry.list())
repos_queried = fed.repos_queried_count(total_registered)
repos_skipped = total_registered - repos_queried
print(f"Queried {repos_queried} of {total_registered} repos ({repos_skipped} skipped)")
```

---

## MCP Tools (v2.8.0)

The `trelix-mcp` server exposes 4 MCP tools for federation, available to any MCP client (Claude Desktop, Cursor, IDEs):

| Tool | Purpose |
|------|---------|
| `federation_list_repos` | List all registered repositories |
| `federation_add_repo` | Register a repository by alias |
| `federation_remove_repo` | Unregister a repository by alias |
| `federation_search_all` | Search across all registered repos |

**Example usage in an MCP client:**

```python
# List registered repos
result = await client.call_tool("federation_list_repos")
# Returns: {"repos": [...], "count": int, "error": str|None}

# Register a new repo
result = await client.call_tool(
    "federation_add_repo",
    alias="myapp",
    path="/absolute/path/to/myapp",
    weight=1.0
)
# Returns: {"added": bool, "alias": str, "path": str, "error": str|None}

# Search across all repos
result = await client.call_tool(
    "federation_search_all",
    query="authentication flow",
    k=10,
    cursor=0
)
# Returns: {"results": [...], "next_cursor": int|None,
#           "total_available": int, "repos_searched": int,
#           "repos_skipped": int, "error": str|None}
```

### Security: `config_path` Confinement (v2.8.1)

All 4 federation MCP tools accept an optional `config_path` parameter to use a custom registry file instead of the default `~/.config/trelix/repos.json`. For security, `config_path` is confined to one of two allowed roots:

1. **`~/.config/trelix/`** — the default user config directory
2. **`<mcp-server-cwd>/.trelix/`** — the MCP server process's current working directory

Any `config_path` outside both roots is rejected with a `ConfigPathNotAllowedError`. This confinement prevents a malicious or prompt-injected MCP client from reading/writing arbitrary paths on the system.

Passing `config_path=None` (the default) uses `~/.config/trelix/repos.json`.

### Repository Cap: `TRELIX_FEDERATION_MAX_REPOS`

The registry enforced a cap on the total number of registered repositories to prevent unbounded growth from scripted or adversarial clients:

- **Default cap:** 50 repositories
- **Environment variable:** `TRELIX_FEDERATION_MAX_REPOS` (range: 1-500)
- **Effect on `federation_add_repo`:** Attempting to register a 51st repo (by default) returns `{"added": False, "error": "Registry is at capacity (50 repos) — remove a repo before adding another"}`
- **Effect on `federation_search_all`:** Only the first N registered repos are actually queried; the response includes `repos_searched` and `repos_skipped` fields

Example when 60 repos are registered but cap is 50:

```json
{
  "results": [...],
  "repos_searched": 50,
  "repos_skipped": 10,
  "error": null
}
```

The cap applies to MCP tool calls only; CLI `trelix federation add` has no cap (preserves backward compatibility).

---

## The TTL Cache (v2.4.0)

Repeated identical queries skip re-querying every registered repository:

| Property | Detail |
|----------|--------|
| Default TTL | 120 seconds |
| Cache key | SHA-256 over `(query, sorted_repo_paths, k)` |
| Thread safety | `threading.Lock` per `FederatedRetriever` instance |
| Disable cache | `cache_ttl=0` |

```python
# Inspect cache performance
stats = fed.cache_stats()   # {"hits": 14, "misses": 3, "size": 3}

# Invalidate after re-indexing a repo
fed.clear_cache()
```

---

## Watch-All (v2.4.0)

Watch all registered repositories for file changes simultaneously:

```bash
trelix watch-all
```

Requires the optional watch dependency:

```bash
pip install "trelix[watch]"
```

File-change events trigger incremental re-indexing for the affected repository only; cached federation results are invalidated automatically for queries that include that repository's path.

---

## How RRF Merge Works

Results from all repositories are merged via Reciprocal Rank Fusion:

1. Each repository returns its own top-k ranked list.
2. Every result `d` in every list receives a contribution score:

   ```
   RRF(d) = Σ  weight × (1 / (k + rank(d)))   for each repo list that contains d
   ```

   where `weight` is the per-repo weight from the registry (default 1.0).

3. Weighted contributions are summed across repositories and results are sorted descending.
4. Deduplication is applied on `(file_path, symbol_id)` — the highest-scoring occurrence is kept.

This approach is rank-based rather than score-based, so it tolerates repositories that use different embedding providers or scoring scales. Per-repo weights let you boost results from authoritative sources without re-indexing.

---

## Registry Location

```
~/.config/trelix/repos.json
```

Format:

```json
{
  "repos": [
    {"alias": "myapp", "path": "/path/to/myapp", "weight": 1.0},
    {"alias": "infra",  "path": "/path/to/infra",  "weight": 1.0}
  ]
}
```

The file is managed by `trelix federation add/remove` commands. Edit it directly only if you need bulk changes.

---

## Per-Repo Weight

Boost a repository's influence on the final ranking:

```bash
trelix federation add myapp /path --weight 1.5
```

Weight multiplies each RRF score contribution from that repository before the global merge. A weight of `1.0` (default) applies no adjustment.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `DimensionMismatchError` | Repositories were indexed with different embedding providers | Re-index all repositories with the same provider |
| No results | Repository not yet indexed | Run `trelix index /path/to/repo` for each registered repo |
| Slow queries | Cache miss on every call | Use the default `cache_ttl=120.0`; avoid setting `cache_ttl=0` in production |
| Stale results after re-index | TTL cache still holding old data | Call `fed.clear_cache()` after re-indexing |
| `ConfigPathNotAllowedError` (MCP only) | `config_path` argument resolves outside `~/.config/trelix/` and `<cwd>/.trelix/` | Use a path within one of the allowed roots, or omit `config_path` to use the default |
| Registry at capacity (MCP only) | Already have `TRELIX_FEDERATION_MAX_REPOS` repos registered | Remove an unused repo with `federation_remove_repo`, or increase the cap via env var |
