import { afterEach, describe, expect, it, vi } from "vitest";
import type { QueueLimits } from "../src/queue.js";
import { createReviewQueue } from "../src/review-intake.js";
import { CONFIG, DAY_MS } from "./support/webhook-harness.js";

/**
 * createReviewQueue is what server.ts and the webhook router build: the caps and
 * the per-job deadline come from the controls, and the dedupe memory keeps a
 * finished claim 4 days by the real clock, at most 10,000 of them.
 */
const LIMITS: QueueLimits = {
    capacity: 1,
    perGroupCapacity: 1,
    concurrency: 1,
    perGroupConcurrency: 1,
    jobTimeoutMs: 60_000,
};

async function settle(): Promise<void> {
    for (let i = 0; i < 25; i += 1) {
        await Promise.resolve();
    }
}

function quickJob(key: string) {
    return {
        key,
        group: "g",
        label: key,
        run: async () => "done" as const,
    };
}

afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
});

describe("createReviewQueue", () => {
    it("keeps a finished claim for 4 days of the real clock, then forgets it", async () => {
        vi.useFakeTimers({ toFake: ["Date"] });
        const queue = createReviewQueue(CONFIG, LIMITS);
        expect(queue.submit(quickJob("k"))).toBe("accepted");
        await settle();

        vi.setSystemTime(Date.now() + 4 * DAY_MS - 1);
        expect(queue.submit(quickJob("k"))).toBe("duplicate");

        vi.setSystemTime(Date.now() + 1);
        expect(queue.submit(quickJob("k"))).toBe("accepted");
    });

    it("remembers at most 10,000 finished claims, forgetting the oldest first", async () => {
        const queue = createReviewQueue(CONFIG, LIMITS);
        for (let i = 0; i <= 10_000; i += 1) {
            expect(queue.submit(quickJob(`key-${i}`))).toBe("accepted");
            await settle();
        }

        expect(queue.claimCounts()).toEqual({ inFlight: 0, kept: 10_000 });
        expect(queue.submit(quickJob("key-1"))).toBe("duplicate");
        expect(queue.submit(quickJob("key-0"))).toBe("accepted");
    });

    it("applies the caps it is given", async () => {
        const queue = createReviewQueue(CONFIG, LIMITS);
        const never = (key: string) => ({
            key,
            group: key,
            label: key,
            run: () => new Promise<never>(() => {}),
        });

        expect(queue.submit(never("a"))).toBe("accepted"); // runs
        expect(queue.submit(never("b"))).toBe("accepted"); // waits
        expect(queue.submit(never("c"))).toBe("full");
    });

    it("applies the share of the wait line it is given", async () => {
        const queue = createReviewQueue(CONFIG, {
            ...LIMITS,
            capacity: 3,
            perGroupCapacity: 1,
        });
        const never = (key: string, group: string) => ({
            key,
            group,
            label: key,
            run: () => new Promise<never>(() => {}),
        });

        expect(queue.submit(never("a1", "A"))).toBe("accepted"); // runs
        expect(queue.submit(never("a2", "A"))).toBe("accepted"); // waits
        expect(queue.submit(never("a3", "A"))).toBe("group_full");
        expect(queue.submit(never("b1", "B"))).toBe("accepted"); // waits
    });

    it("applies the per-job deadline it is given, and logs the cut-off job once", async () => {
        vi.useFakeTimers();
        const consoleError = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const queue = createReviewQueue(CONFIG, {
            ...LIMITS,
            capacity: 2,
            jobTimeoutMs: 60_000,
        });
        const hung = {
            key: "hung",
            group: "g1",
            label: "owner/repo#7",
            run: () => new Promise<never>(() => {}),
        };
        queue.submit(hung);
        queue.submit(quickJob("next"));
        await vi.advanceTimersByTimeAsync(59_999);
        expect(queue.counts()).toEqual({ waiting: 1, running: 1 });

        await vi.advanceTimersByTimeAsync(1);

        expect(queue.counts()).toEqual({ waiting: 0, running: 0 });
        expect(consoleError).toHaveBeenCalledTimes(1);
        const line = String(consoleError.mock.calls[0]?.[0]);
        expect(line).toContain("[webhook] review failed");
        expect(line).toContain('"job":"owner/repo#7"');
        expect(line).toContain('"error":"JobTimeoutError"');
    });
});
