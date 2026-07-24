import * as assert from "assert";
import { SearchController } from "../../search-controller";
import { SearchPage } from "../../mcp-client";

function makePage(symbols: string[], nextCursor: number | null = null): SearchPage {
  return {
    results: symbols.map((s) => ({
      symbol: s,
      file: "a.py",
      kind: "function",
      lines: "1-2",
      score: 1,
      source: "vector",
      body: "",
      language: "python",
    })),
    nextCursor,
    totalAvailable: symbols.length,
  };
}

/** Fake timer harness: setTimeout/clearTimeout that only fire when flush() is called. */
function fakeTimers() {
  let nextId = 0;
  const pending = new Map<number, () => void>();
  return {
    setTimeoutFn: (fn: () => void) => {
      const id = nextId++;
      pending.set(id, fn);
      return id;
    },
    clearTimeoutFn: (handle: unknown) => {
      pending.delete(handle as number);
    },
    flush: () => {
      const fns = [...pending.values()];
      pending.clear();
      fns.forEach((fn) => fn());
    },
    pendingCount: () => pending.size,
  };
}

suite("SearchController debounce + re-search", () => {
  test("only the last onValueChange within the debounce window triggers a search", async () => {
    const timers = fakeTimers();
    let searchCalls = 0;
    let lastQuery = "";
    const items: unknown[] = [];

    const controller = new SearchController({
      debounceMs: 250,
      pageSize: 10,
      search: async (query) => {
        searchCalls++;
        lastQuery = query;
        return makePage(["result-for-" + query]);
      },
      onItems: (results) => items.push(...results),
      onError: () => {},
      onBusyChange: () => {},
      setTimeoutFn: timers.setTimeoutFn,
      clearTimeoutFn: timers.clearTimeoutFn,
    });

    // Simulate rapid keystrokes: "j", "jw", "jwt" fired faster than the debounce window.
    controller.onValueChange("j");
    controller.onValueChange("jw");
    controller.onValueChange("jwt");

    assert.strictEqual(timers.pendingCount(), 1, "each keystroke should cancel the prior pending timer, leaving exactly one scheduled");
    timers.flush();
    await new Promise((r) => setImmediate(r));

    assert.strictEqual(searchCalls, 1, "search() should be called exactly once, not once per keystroke");
    assert.strictEqual(lastQuery, "jwt");
  });

  test("clearing the query to empty shows no items and schedules no search", () => {
    const timers = fakeTimers();
    let searchCalls = 0;
    let lastItems: unknown[] | undefined;

    const controller = new SearchController({
      debounceMs: 250,
      pageSize: 10,
      search: async () => {
        searchCalls++;
        return makePage(["x"]);
      },
      onItems: (results) => {
        lastItems = results;
      },
      onError: () => {},
      onBusyChange: () => {},
      setTimeoutFn: timers.setTimeoutFn,
      clearTimeoutFn: timers.clearTimeoutFn,
    });

    controller.onValueChange("   ");

    assert.strictEqual(timers.pendingCount(), 0, "a blank query should not schedule a debounced search");
    assert.deepStrictEqual(lastItems, []);
    assert.strictEqual(searchCalls, 0);
  });

  test("a stale in-flight search response is discarded if a newer query has since superseded it", async () => {
    const timers = fakeTimers();
    const resolvers: Array<(page: SearchPage) => void> = [];
    const onItemsCalls: unknown[][] = [];

    const controller = new SearchController({
      debounceMs: 250,
      pageSize: 10,
      search: () =>
        new Promise<SearchPage>((resolve) => {
          resolvers.push(resolve);
        }),
      onItems: (results) => onItemsCalls.push(results),
      onError: () => {},
      onBusyChange: () => {},
      setTimeoutFn: timers.setTimeoutFn,
      clearTimeoutFn: timers.clearTimeoutFn,
    });

    controller.onValueChange("first");
    timers.flush(); // fires the debounced search("first", 0, gen=1) — now in flight

    controller.onValueChange("second");
    timers.flush(); // fires the debounced search("second", 0, gen=2) — now in flight

    // Resolve the STALE "first" request after the newer "second" request was already issued.
    resolvers[0](makePage(["stale-result"]));
    await new Promise((r) => setImmediate(r));
    assert.strictEqual(onItemsCalls.length, 0, "the stale response must not call onItems");

    resolvers[1](makePage(["fresh-result"]));
    await new Promise((r) => setImmediate(r));
    assert.strictEqual(onItemsCalls.length, 1);
    assert.deepStrictEqual(onItemsCalls[0], [{ symbol: "fresh-result", file: "a.py", kind: "function", lines: "1-2", score: 1, source: "vector", body: "", language: "python" }]);
  });
});

suite("SearchController pagination", () => {
  test("loadMore appends the next page to the accumulated results using next_cursor", async () => {
    const timers = fakeTimers();
    const cursorsRequested: number[] = [];
    let onItemsResult: { results: unknown[]; hasMore: boolean } | undefined;

    const controller = new SearchController({
      debounceMs: 250,
      pageSize: 2,
      search: async (_query, _k, cursor) => {
        cursorsRequested.push(cursor);
        return cursor === 0 ? makePage(["a", "b"], 2) : makePage(["c"], null);
      },
      onItems: (results, hasMore) => {
        onItemsResult = { results, hasMore };
      },
      onError: () => {},
      onBusyChange: () => {},
      setTimeoutFn: timers.setTimeoutFn,
      clearTimeoutFn: timers.clearTimeoutFn,
    });

    controller.onValueChange("query");
    timers.flush();
    await new Promise((r) => setImmediate(r));
    assert.strictEqual(onItemsResult?.hasMore, true);
    assert.strictEqual(onItemsResult?.results.length, 2);

    await controller.loadMore();

    assert.deepStrictEqual(cursorsRequested, [0, 2]);
    assert.strictEqual(onItemsResult?.hasMore, false, "no next_cursor on the second page means no more load-more affordance");
    assert.strictEqual(onItemsResult?.results.length, 3, "loadMore should append to, not replace, the accumulated results");
  });

  test("loadMore is a no-op when there is no next_cursor", async () => {
    const timers = fakeTimers();
    let searchCalls = 0;

    const controller = new SearchController({
      debounceMs: 250,
      pageSize: 10,
      search: async () => {
        searchCalls++;
        return makePage(["only-page"], null);
      },
      onItems: () => {},
      onError: () => {},
      onBusyChange: () => {},
      setTimeoutFn: timers.setTimeoutFn,
      clearTimeoutFn: timers.clearTimeoutFn,
    });

    controller.onValueChange("query");
    timers.flush();
    await new Promise((r) => setImmediate(r));
    assert.strictEqual(searchCalls, 1);

    await controller.loadMore();

    assert.strictEqual(searchCalls, 1, "loadMore must not call search() again when nextCursor is null");
  });
});
