export interface AskStreamOptions {
  fetch?: typeof fetch;
  /** Aborts the underlying request when triggered. */
  signal?: AbortSignal;
}

/** Thrown when the `/ask` stream emits a `data: [ERROR: ...]` frame. */
export class TrelixAskError extends Error {}

/**
 * Streams tokens from `/ask`, an SSE endpoint (see `ask()` in `src/trelix/api/app.py`).
 *
 * Frames are exactly one of:
 *   - `data: {token}\n\n`         — yielded as a token string
 *   - `data: [DONE]\n\n`          — stream ends normally
 *   - `data: [ERROR: {exc}]\n\n`  — stream ends by throwing TrelixAskError
 */
export async function* askStream(
  baseUrl: string,
  params: { query: string; repo: string },
  options: AskStreamOptions = {},
): AsyncGenerator<string, void, void> {
  const fetchImpl = options.fetch ?? fetch;
  const url = new URL(baseUrl.replace(/\/+$/, "") + "/ask");
  url.searchParams.set("query", params.query);
  url.searchParams.set("repo", params.repo);

  const res = await fetchImpl(url, { signal: options.signal });
  if (!res.ok || !res.body) {
    throw new TrelixAskError(`trelix /ask request failed: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let separatorIndex: number;
      while ((separatorIndex = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const result = parseFrame(frame);
        if (result === null) continue;
        if (result.done) return;
        yield result.token;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(frame: string): { done: false; token: string } | { done: true } | null {
  if (!frame.startsWith("data: ")) return null;
  const data = frame.slice("data: ".length);

  if (data === "[DONE]") return { done: true };

  const errorMatch = /^\[ERROR: ([\s\S]*)\]$/.exec(data);
  if (errorMatch) throw new TrelixAskError(errorMatch[1]);

  return { done: false, token: data };
}
