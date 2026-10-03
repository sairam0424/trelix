import { describe, expect, it } from "vitest";
import { ClaimStore } from "../src/claims.js";

const HOUR_MS = 60 * 60 * 1000;

/** A store over a clock the test moves by hand. */
function makeStore(keepMs = 24 * HOUR_MS, maxKept = 10_000) {
    const clock = { now: 1_000_000 };
    const store = new ClaimStore({
        keepMs,
        maxKept,
        now: () => clock.now,
    });
    return { store, clock };
}

describe("ClaimStore", () => {
    describe("claiming", () => {
        it("gives the claim to the first caller and refuses the second while it is in flight", () => {
            const { store } = makeStore();

            expect(store.claim("a")).toBe(true);
            expect(store.claim("a")).toBe(false);
        });

        it("treats different keys independently", () => {
            const { store } = makeStore();

            expect(store.claim("a")).toBe(true);
            expect(store.claim("b")).toBe(true);
        });
    });

    describe("release", () => {
        it("lets the key be claimed again at once", () => {
            const { store } = makeStore();
            store.claim("a");

            store.release("a");

            expect(store.claim("a")).toBe(true);
        });

        it("ignores a key that was never claimed", () => {
            const { store } = makeStore();

            store.release("never-claimed");

            expect(store.counts()).toEqual({ inFlight: 0, kept: 0 });
        });
    });

    describe("keep", () => {
        it("refuses the key for the whole keep period", () => {
            const { store, clock } = makeStore(24 * HOUR_MS);
            store.claim("a");
            store.keep("a");

            clock.now += 24 * HOUR_MS - 1;

            expect(store.claim("a")).toBe(false);
        });

        it("accepts the key again once the keep period has passed", () => {
            const { store, clock } = makeStore(24 * HOUR_MS);
            store.claim("a");
            store.keep("a");

            clock.now += 24 * HOUR_MS;

            expect(store.claim("a")).toBe(true);
        });

        it("counts the period from the moment of keep, not from the claim", () => {
            const { store, clock } = makeStore(HOUR_MS);
            store.claim("a");
            clock.now += 10 * HOUR_MS; // a long job
            store.keep("a");

            clock.now += HOUR_MS - 1;

            expect(store.claim("a")).toBe(false);
        });

        it("is no longer in flight after keep", () => {
            const { store } = makeStore();
            store.claim("a");

            store.keep("a");

            expect(store.counts()).toEqual({ inFlight: 0, kept: 1 });
        });

        it("drops expired entries on the next call, so an idle store does not grow", () => {
            const { store, clock } = makeStore(HOUR_MS);
            for (const key of ["a", "b", "c"]) {
                store.claim(key);
                store.keep(key);
            }
            clock.now += HOUR_MS;

            store.claim("d");

            expect(store.counts()).toEqual({ inFlight: 1, kept: 0 });
        });

        it("drops the expired entries at the front and keeps the live ones behind them", () => {
            const { store, clock } = makeStore(HOUR_MS);
            store.claim("first");
            store.keep("first"); // expires at start + 1 h
            clock.now += 30 * 60 * 1000;
            store.claim("second");
            store.keep("second"); // expires at start + 1.5 h

            clock.now += 31 * 60 * 1000; // start + 1 h 1 min

            expect(store.claim("first")).toBe(true);
            expect(store.claim("second")).toBe(false);
        });

        it("accepts an expired key even when a live one sits in front of it after the clock stepped back", () => {
            const { store, clock } = makeStore(HOUR_MS);
            store.claim("old");
            store.keep("old"); // expires at start + 1 h
            clock.now -= 2 * HOUR_MS; // the clock steps back two hours
            store.claim("young");
            store.keep("young"); // expires at start - 1 h: already in the past
            clock.now += 2 * HOUR_MS; // back to start

            expect(store.claim("young")).toBe(true);
            expect(store.claim("old")).toBe(false);
        });
    });

    describe("memory bound", () => {
        it("keeps at most maxKept finished claims and forgets the oldest first", () => {
            const { store } = makeStore(24 * HOUR_MS, 3);
            for (const key of ["k1", "k2", "k3", "k4", "k5"]) {
                store.claim(key);
                store.keep(key);
            }

            expect(store.counts().kept).toBe(3);
            expect(store.claim("k1")).toBe(true); // forgotten
            expect(store.claim("k2")).toBe(true); // forgotten
            expect(store.claim("k3")).toBe(false); // still kept
            expect(store.claim("k5")).toBe(false);
        });

        it("stays within the bound through a flood of distinct keys", () => {
            const { store } = makeStore(24 * HOUR_MS, 50);
            for (let i = 0; i < 5_000; i += 1) {
                const key = `installation:${i}`;
                store.claim(key);
                store.keep(key);
            }

            expect(store.counts()).toEqual({ inFlight: 0, kept: 50 });
        });

        it("does not grow when the same key is kept twice", () => {
            const { store } = makeStore(24 * HOUR_MS, 10);
            store.claim("a");
            store.keep("a");
            store.keep("a");

            expect(store.counts().kept).toBe(1);
        });

        it("moves a key kept again to the back of the line for eviction", () => {
            const { store } = makeStore(24 * HOUR_MS, 2);
            store.claim("a");
            store.keep("a");
            store.claim("b");
            store.keep("b");
            store.keep("a"); // "a" is now the newest

            store.claim("c");
            store.keep("c"); // over the bound: "b" is the oldest and goes

            expect(store.claim("b")).toBe(true);
            expect(store.claim("a")).toBe(false);
        });
    });
});
