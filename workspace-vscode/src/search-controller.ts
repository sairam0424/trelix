import { SearchResult, SearchPage } from "./mcp-client";

export interface SearchFn {
  (query: string, k: number, cursor: number): Promise<SearchPage>;
}

export interface SearchControllerOptions {
  debounceMs: number;
  pageSize: number;
  search: SearchFn;
  /** Called with the accumulated results + whether a "load more" affordance should show. */
  onItems: (results: SearchResult[], hasMore: boolean) => void;
  onError: (err: unknown) => void;
  onBusyChange: (busy: boolean) => void;
  setTimeoutFn?: (fn: () => void, ms: number) => unknown;
  clearTimeoutFn?: (handle: unknown) => void;
}

/**
 * Owns the debounce + cursor-pagination + stale-response-rejection state
 * machine behind trelix.search's live-narrowing QuickPick, decoupled from
 * vscode.QuickPick itself so it's testable without the Extension Host.
 */
export class SearchController {
  private readonly opts: SearchControllerOptions;
  private debounceHandle: unknown | undefined;
  private generation = 0;
  private accumulated: SearchResult[] = [];
  private nextCursor: number | null = null;
  private currentQuery = "";

  constructor(opts: SearchControllerOptions) {
    this.opts = opts;
  }

  /** Called on every QuickPick.onDidChangeValue — debounces then re-searches from page 0. */
  onValueChange(value: string): void {
    this.currentQuery = value;
    this.clearDebounce();
    this.generation++;
    const gen = this.generation;
    this.accumulated = [];
    this.nextCursor = null;

    if (!value.trim()) {
      this.opts.onItems([], false);
      return;
    }

    const setTimeoutFn = this.opts.setTimeoutFn ?? setTimeout;
    this.debounceHandle = setTimeoutFn(() => {
      void this.runSearch(value, 0, gen);
    }, this.opts.debounceMs);
  }

  /** Called when the user activates the "load more" item. */
  async loadMore(): Promise<void> {
    if (this.nextCursor === null) return;
    await this.runSearch(this.currentQuery, this.nextCursor, this.generation);
  }

  dispose(): void {
    this.clearDebounce();
  }

  private clearDebounce(): void {
    if (this.debounceHandle !== undefined) {
      const clear = this.opts.clearTimeoutFn ?? ((handle: unknown) => clearTimeout(handle as NodeJS.Timeout));
      clear(this.debounceHandle);
      this.debounceHandle = undefined;
    }
  }

  private async runSearch(query: string, cursor: number, gen: number): Promise<void> {
    this.opts.onBusyChange(true);
    try {
      const page = await this.opts.search(query, this.opts.pageSize, cursor);
      if (gen !== this.generation) return; // a newer keystroke superseded this request
      this.accumulated = cursor === 0 ? page.results : [...this.accumulated, ...page.results];
      this.nextCursor = page.nextCursor;
      this.opts.onItems(this.accumulated, this.nextCursor !== null);
    } catch (err) {
      if (gen === this.generation) this.opts.onError(err);
    } finally {
      if (gen === this.generation) this.opts.onBusyChange(false);
    }
  }
}
