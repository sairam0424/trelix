import { afterEach, describe, expect, it, vi } from "vitest";
import type { QueueLimits } from "../src/queue.js";
import { createReviewQueue } from "../src/review-intake.js";
import { CONFIG, DAY_MS } from "./support/webhook-harness.js";

/**
 * createReviewQueue is what server.ts and the webhook router build: the caps come
 * from the controls, and the dedupe memory keeps a finished claim 24 hours by the
 * real clock, at most 10,000 of them.
 */
const LIMITS: QueueLimits = {
    capacity: 1,
    concurrency: 1,
    perGroupConcurrency: 1,
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
});

describe("createReviewQueue", () => {
    it("keeps a finished claim for 24 hours of the real clock, then forgets it", async () => {
        vi.useFakeTimers({ toFake: ["Date"] });
        const queue = createReviewQueue(CONFIG, LIMITS);
        expect(queue.submit(quickJob("k"))).toBe("accepted");
        await settle();

        vi.setSystemTime(Date.now() + DAY_MS - 1);
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
});
