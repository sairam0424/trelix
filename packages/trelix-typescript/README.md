# @trelix/sdk

TypeScript client for the [trelix](https://github.com/sairam0424/trelix) REST API — hybrid code search, LLM synthesis, indexing, and knowledge-graph queries.

This is a thin hand-written HTTP client, not a fully generated one: types are generated from the API's OpenAPI schema (`src/generated/schema.ts`), but request/response glue is hand-written in `src/client.ts`/`src/sse.ts`. Full client codegen has a documented failure mode of dropping operations on non-trivial specs — hand-gluing a typed client on top of generated types avoids that risk.

## Install

```bash
npm install @trelix/sdk
```

Requires a running `trelix serve` instance (`pip install 'trelix[serve]' && trelix serve /path/to/repo --port 8765`).

## Usage

```ts
import { TrelixClient } from "@trelix/sdk";

const client = new TrelixClient("http://127.0.0.1:8765");

const { results, next_cursor, total_available } = await client.search({
  query: "how does authentication work?",
  repo: "/path/to/repo",
  k: 10,
});

for (const r of results) {
  console.log(`${r.file} :: ${r.symbol} (${r.score})`);
}

// Fetch the next page:
if (next_cursor !== null) {
  await client.search({ query: "...", repo: "/path/to/repo", cursor: next_cursor });
}
```

`search()` also takes `intentHint` (one of the server's IntentType values) and `hydeSnippetHint`. Setting a valid `intentHint` skips trelix's internal LLM intent classification and routes straight to that intent's strategy — for callers that already classified the query. An unrecognized value is not an error; the server falls back to normal classification.

Other methods: `health()`, `index(repoPath)`, `parseFile(request)`, `stats(repo)`, `graphStats(repo)`, `graphCommunities(repo, { minCommunitySize?, maxCommunities? })`, `graphVisualize(repo, output?)`, `graphSearch(repo, symbolId, depth?)`.

`parseFile()` parses a single file without persisting anything to the index — for editor / pre-commit callers that need structural info on unsaved or not-yet-indexed content. Pass exactly one content source, `file_path` (read fresh from disk) or `content` + `file_name` (inline text); the server rejects both/neither with a 422. Cross-file call and type resolution is skipped, so the returned edge counts reflect only what Tree-sitter could determine from that file in isolation.

```ts
const { symbols, parse_errors } = await client.parseFile({
  repo_path: "/path/to/repo",
  content: "def unsaved(): ...",
  file_name: "draft.py",
});
```

`graphCommunities()` returns the 50 largest communities of size >= 2 by default; pass `{ minCommunitySize: 1, maxCommunities: 0 }` for the uncapped list (on trelix's own index that is 6,497 entries / ~1.16 MB, 99.1% of them singletons).

### Streaming synthesis (`/ask`)

`/ask` is a Server-Sent Events endpoint, so it's a separate async generator rather than a `TrelixClient` method:

```ts
import { askStream, TrelixAskError } from "@trelix/sdk";

try {
  for await (const token of askStream("http://127.0.0.1:8765", {
    query: "how does authentication work?",
    repo: "/path/to/repo",
  })) {
    process.stdout.write(token);
  }
} catch (err) {
  if (err instanceof TrelixAskError) {
    console.error("synthesis failed:", err.message);
  }
}
```

## Error handling

Non-2xx HTTP responses throw `TrelixApiError` (with `.status` and the parsed error `.body`). `/ask` stream failures throw `TrelixAskError` instead of yielding an error token.

## Regenerating types

`src/generated/schema.ts` is checked into git, like a lockfile, so `npm run build`/`npm test` work without Python installed. After changing a route or response model in `src/trelix/api/app.py`, regenerate it:

```bash
pip install -e '.[serve]'   # from the repo root; fastapi is the only requirement
npm run codegen             # ~1.4s, no server needed
```

`npm run codegen` dumps the spec in-process via `create_app().openapi()` — no `trelix serve`, no port, no readiness race. This is what makes the drift gate possible: `.github/workflows/schema-drift.yml` reruns codegen on every push/PR and fails if the committed file differs. That gate did not exist while the schema sat 604 committed lines against 768 regenerated, hiding `POST /parse`, the `/search` intent hints and the `/graph/communities` caps from this client.

Set `TRELIX_OPENAPI_URL` to generate from an already-running instance instead (`http://host:8765/openapi.json`); both paths produce byte-identical output.

## Links

- [trelix on GitHub](https://github.com/sairam0424/trelix)
- [trelix on PyPI](https://pypi.org/project/trelix/)
