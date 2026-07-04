# Federation Guide — trelix v2.4.0

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
fed = FederatedRetriever(registry, cache_ttl=120.0)
results = fed.retrieve("authentication", k=10)
```

`results` is a list of `SearchResult` objects, each carrying its source repository alias alongside the standard `file_path`, `symbol_id`, `score`, and `snippet` fields.

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
   RRF(d) = Σ  1 / (k + rank(d))   for each repo list that contains d
   ```

3. Contributions are summed across repositories and results are sorted descending.
4. Deduplication is applied on `(file_path, symbol_id)` — the highest-scoring occurrence is kept.

This approach is rank-based rather than score-based, so it tolerates repositories that use different embedding providers or scoring scales.

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
