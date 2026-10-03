/**
 * The dedupe memory of the review queue: which commits already have a review
 * queued, running or done.
 *
 * A claim is taken before the work is queued, in one synchronous call, so two
 * deliveries of the same commit cannot both pass. It then ends one of two ways:
 * `release` (the work failed or ended without a verdict: the next delivery may
 * try again) or `keep` (the work is done: the key is remembered for `keepMs`, so a
 * redelivery, or the redelivery backstop, does not pay for the same review twice).
 *
 * Bounds, so a flood of distinct keys cannot grow this without limit:
 * - claims in flight are only as many as the queue admits (its capacity plus its
 *   concurrency), because the queue releases a claim it then refuses;
 * - kept claims are capped at `maxKept`: past it the oldest is forgotten, which at
 *   worst allows one repeated review of a commit that was reviewed long ago;
 * - kept claims expire after `keepMs`, and expired ones are dropped on the next
 *   call. There is no timer: an idle process holds no timer for this.
 *
 * Nothing persists: a restart forgets every claim. The cost is one repeated
 * review, never a skipped one.
 */

/** Finished claims are remembered this long (24 hours). */
export const DEFAULT_CLAIM_KEEP_MS = 24 * 60 * 60 * 1000;

/** Most finished claims remembered at once; about a megabyte of keys. */
export const DEFAULT_MAX_KEPT_CLAIMS = 10_000;

export interface ClaimStoreOptions {
    /** How long a claim is remembered after `keep`. */
    readonly keepMs: number;
    /** Most finished claims remembered at once. */
    readonly maxKept: number;
    /** Milliseconds; injectable so a test does not wait a day. */
    readonly now: () => number;
}

export class ClaimStore {
    private readonly inFlight = new Set<string>();
    /** key -> time it expires. Entries are added in completion order, so the first is the oldest. */
    private readonly kept = new Map<string, number>();

    constructor(private readonly options: ClaimStoreOptions) {}

    /**
     * Takes the claim on `key`. Returns false, and changes nothing, when the key
     * is in flight or was kept less than `keepMs` ago. Synchronous, so the
     * check and the take cannot be interleaved with another caller.
     */
    claim(key: string): boolean {
        const now = this.options.now();
        this.dropExpired(now);
        if (this.inFlight.has(key)) {
            return false;
        }
        // Looked up by key, not trusted to dropExpired alone: if the clock ever steps
        // back, an expired entry can sit behind a live one.
        const expiresAt = this.kept.get(key);
        if (expiresAt !== undefined && expiresAt > now) {
            return false;
        }
        this.kept.delete(key);
        this.inFlight.add(key);
        return true;
    }

    /** Gives the claim up: the work did not finish, so the key may be claimed again at once. */
    release(key: string): void {
        this.inFlight.delete(key);
    }

    /** Ends the claim with the work done: `key` is refused for `keepMs` from now. */
    keep(key: string): void {
        this.inFlight.delete(key);
        // delete first: a key kept twice moves to the end, which keeps the Map in expiry order.
        this.kept.delete(key);
        this.kept.set(key, this.options.now() + this.options.keepMs);
        this.dropOverflow();
        this.dropExpired(this.options.now());
    }

    /** How many claims are held: in flight, and kept (expired ones not yet dropped included). */
    counts(): { readonly inFlight: number; readonly kept: number } {
        return { inFlight: this.inFlight.size, kept: this.kept.size };
    }

    private dropExpired(now: number): void {
        for (const [key, expiresAt] of this.kept) {
            if (expiresAt > now) {
                return;
            }
            this.kept.delete(key);
        }
    }

    private dropOverflow(): void {
        for (const key of this.kept.keys()) {
            if (this.kept.size <= this.options.maxKept) {
                return;
            }
            this.kept.delete(key);
        }
    }
}
