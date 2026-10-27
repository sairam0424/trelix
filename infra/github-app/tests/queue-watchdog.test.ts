import { afterEach, describe, expect, it, vi } from "vitest";
import { JobTimeoutError } from "../src/queue.js";
import { makeHarness, settle } from "./support/queue-harness.js";

afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
});

describe("the watchdog", () => {
    const DEADLINE_MS = 1_000;

    it("takes back the slot of a job that never settles, reports a timeout by label, and starts the next job", async () => {
        vi.useFakeTimers();
        const h = makeHarness({
            concurrency: 1,
            capacity: 5,
            jobTimeoutMs: DEADLINE_MS,
        });
        h.queue.submit(h.hold("hung").job);
        h.queue.submit(h.hold("next").job);
        await vi.advanceTimersByTimeAsync(DEADLINE_MS - 1);
        expect(h.started).toEqual(["hung"]);
        expect(h.errors).toEqual([]);

        await vi.advanceTimersByTimeAsync(1);

        expect(h.started).toEqual(["hung", "next"]);
        expect(h.queue.counts()).toEqual({ waiting: 0, running: 1 });
        expect(h.errors).toHaveLength(1);
        expect(h.errors[0]?.label).toBe("hung");
        expect(h.errors[0]?.err).toBeInstanceOf(JobTimeoutError);
        expect(h.errors[0]?.err).toMatchObject({
            name: "JobTimeoutError",
            message:
                "the job did not finish within 1000 ms; its slot was released",
        });
    });

    it("counts the deadline from the moment the job starts, not from its submission", async () => {
        vi.useFakeTimers();
        const h = makeHarness({
            concurrency: 1,
            capacity: 5,
            jobTimeoutMs: DEADLINE_MS,
        });
        const first = h.hold("first");
        h.queue.submit(first.job);
        h.queue.submit(h.hold("second").job);
        await vi.advanceTimersByTimeAsync(700);
        first.gate.resolve();
        await vi.advanceTimersByTimeAsync(0);
        expect(h.started).toEqual(["first", "second"]);

        await vi.advanceTimersByTimeAsync(DEADLINE_MS - 1);
        expect(h.errors).toEqual([]);
        await vi.advanceTimersByTimeAsync(1);

        expect(h.errors.map((e) => e.label)).toEqual(["second"]);
    });

    it("releases the claim of the job it cuts off, so that key may be sent again", async () => {
        vi.useFakeTimers();
        const h = makeHarness({ jobTimeoutMs: DEADLINE_MS });
        h.queue.submit(h.hold("hung").job);
        await vi.advanceTimersByTimeAsync(DEADLINE_MS);

        expect(h.claims.counts()).toEqual({ inFlight: 0, kept: 0 });
        expect(h.queue.submit(h.hold("hung").job)).toBe("accepted");
    });

    it("takes back the group's slot too: the next job of the same group starts", async () => {
        vi.useFakeTimers();
        const h = makeHarness({
            concurrency: 2,
            capacity: 5,
            perGroupConcurrency: 1,
            jobTimeoutMs: DEADLINE_MS,
        });
        h.queue.submit(h.hold("hung", "G").job);
        h.queue.submit(h.hold("same-group", "G").job);
        await vi.advanceTimersByTimeAsync(DEADLINE_MS - 1);
        expect(h.started).toEqual(["hung"]);

        await vi.advanceTimersByTimeAsync(1);

        expect(h.started).toEqual(["hung", "same-group"]);
    });

    it("ignores the late settlement of a job it cut off: no second release, no second start", async () => {
        vi.useFakeTimers();
        const h = makeHarness({
            concurrency: 1,
            capacity: 5,
            jobTimeoutMs: DEADLINE_MS,
        });
        const hung = h.hold("hung");
        const second = h.hold("second");
        const third = h.hold("third");
        for (const { job } of [hung, second, third]) {
            h.queue.submit(job);
        }
        await vi.advanceTimersByTimeAsync(DEADLINE_MS);
        expect(h.started).toEqual(["hung", "second"]);

        hung.gate.resolve("done");
        await vi.advanceTimersByTimeAsync(0);

        expect(h.started).toEqual(["hung", "second"]);
        expect(h.queue.counts()).toEqual({ waiting: 1, running: 1 });
        // The late "done" keeps nothing: second and third hold the only claims.
        expect(h.claims.counts()).toEqual({ inFlight: 2, kept: 0 });
        expect(h.errors).toHaveLength(1);

        second.gate.resolve();
        await vi.advanceTimersByTimeAsync(0);
        expect(h.started).toEqual(["hung", "second", "third"]);
    });

    it("drops the late failure of a job it cut off: no second report, no unhandled rejection", async () => {
        vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
        const unhandled = vi.fn();
        process.on("unhandledRejection", unhandled);
        try {
            const h = makeHarness({ jobTimeoutMs: DEADLINE_MS });
            const hung = h.hold("hung");
            h.queue.submit(hung.job);
            await vi.advanceTimersByTimeAsync(DEADLINE_MS);
            expect(h.errors).toHaveLength(1);

            hung.gate.reject(new Error("failed after the deadline"));
            await settle();
            await new Promise((resolve) => setImmediate(resolve));

            expect(h.errors).toHaveLength(1);
            expect(unhandled).not.toHaveBeenCalled();
        } finally {
            process.off("unhandledRejection", unhandled);
        }
    });

    it("leaves a job that finishes just before the deadline alone", async () => {
        vi.useFakeTimers();
        const h = makeHarness({ jobTimeoutMs: DEADLINE_MS });
        const fine = h.hold("fine");
        h.queue.submit(fine.job);
        await vi.advanceTimersByTimeAsync(DEADLINE_MS - 1);

        fine.gate.resolve("done");
        await vi.advanceTimersByTimeAsync(0);

        expect(h.errors).toEqual([]);
        expect(h.claims.counts()).toEqual({ inFlight: 0, kept: 1 });
        expect(vi.getTimerCount()).toBe(0);
        await vi.advanceTimersByTimeAsync(10 * DEADLINE_MS);
        expect(h.errors).toEqual([]);
    });

    it("reports a job that throws as before, once, and arms nothing afterwards", async () => {
        vi.useFakeTimers();
        const h = makeHarness({ jobTimeoutMs: DEADLINE_MS });
        const bad = h.hold("bad");
        h.queue.submit(bad.job);
        await vi.advanceTimersByTimeAsync(DEADLINE_MS - 1);

        const failure = new Error("boom");
        bad.gate.reject(failure);
        await vi.advanceTimersByTimeAsync(0);

        expect(h.errors).toEqual([{ err: failure, label: "bad" }]);
        expect(h.claims.counts()).toEqual({ inFlight: 0, kept: 0 });
        expect(vi.getTimerCount()).toBe(0);
        await vi.advanceTimersByTimeAsync(10 * DEADLINE_MS);
        expect(h.errors).toHaveLength(1);
    });

    it("reports a job that throws before it returns a promise, and arms nothing", async () => {
        vi.useFakeTimers();
        const h = makeHarness({ jobTimeoutMs: DEADLINE_MS });
        const failure = new Error("sync failure");
        h.queue.submit({
            key: "bad",
            group: "g1",
            label: "bad",
            run: () => {
                throw failure;
            },
        });

        await vi.advanceTimersByTimeAsync(0);

        expect(h.errors).toEqual([{ err: failure, label: "bad" }]);
        expect(vi.getTimerCount()).toBe(0);
    });
});
