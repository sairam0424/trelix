# @trelix/sdk

TypeScript client for the [trelix](https://github.com/sairam0424/trelix) REST API — hybrid code search, LLM synthesis, indexing, and knowledge-graph queries.

This is a thin hand-written HTTP client, not a fully generated one: types are generated from the live OpenAPI schema (`src/generated/schema.ts`), but request/response glue is hand-written in `src/client.ts`/`src/sse.ts`. Full client codegen has a documented failure mode of dropping operations on non-trivial specs — hand-gluing a typed client on top of generated types avoids that risk.

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

Other methods: `health()`, `index(repoPath)`, `stats(repo)`, `graphStats(repo)`, `graphCommunities(repo)`, `graphVisualize(repo, output?)`, `graphSearch(repo, symbolId, depth?)`.

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

`src/generated/schema.ts` is checked into git, like a lockfile, so `npm run build`/`npm test` work in CI without a live server. After changing a route or response model in `src/trelix/api/app.py`, regenerate it by hand:

```bash
trelix serve /path/to/repo --port 8765 &
npm run codegen
```

## Links

- [trelix on GitHub](https://github.com/sairam0424/trelix)
- [trelix on PyPI](https://pypi.org/project/trelix/)
