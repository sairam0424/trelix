import { Octokit } from "@octokit/rest";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { redeliverFailedDeliveries } from "../src/scripts/redeliver-webhook-deliveries.js";

interface FakeDelivery {
    id: number;
    guid: string;
    delivered_at: string;
    status_code: number;
    event: string;
}

/** Fakes @octokit/rest's own HTTP transport via its documented `hook.wrap("request", ...)` extension point. */
function buildFakeOctokit(deliveries: FakeDelivery[]) {
    const redeliveredIds: number[] = [];
    const octokit = new Octokit({});

    octokit.hook.wrap("request", async (_request, options) => {
        if (
            options.method === "GET" &&
            options.url === "/app/hook/deliveries"
        ) {
            return { status: 200, url: "", headers: {}, data: deliveries };
        }
        if (
            options.method === "POST" &&
            options.url === "/app/hook/deliveries/{delivery_id}/attempts"
        ) {
            redeliveredIds.push(options.delivery_id as number);
            return { status: 202, url: "", headers: {}, data: {} };
        }
        throw new Error(
            `unexpected octokit request in test: ${options.method} ${options.url}`,
        );
    });

    return { octokit, redeliveredIds };
}

const NOW = new Date("2026-09-20T12:00:00Z");

describe("redeliverFailedDeliveries", () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.setSystemTime(NOW);
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it("redelivers only failed deliveries old enough to be past GitHub's own retry window", async () => {
        const deliveries: FakeDelivery[] = [
            // Failed, well past the retry window -> should be redelivered.
            {
                id: 1,
                guid: "g1",
                delivered_at: "2026-09-20T11:00:00Z",
                status_code: 502,
                event: "pull_request",
            },
            // Failed, but delivered just moments ago -> too fresh, GitHub's own
            // retry may still be in flight, must NOT be redelivered.
            {
                id: 2,
                guid: "g2",
                delivered_at: "2026-09-20T11:59:30Z",
                status_code: 500,
                event: "pull_request",
            },
            // Succeeded -> must NOT be redelivered regardless of age.
            {
                id: 3,
                guid: "g3",
                delivered_at: "2026-09-20T10:00:00Z",
                status_code: 200,
                event: "pull_request",
            },
            // A 3xx redirect counts as delivered successfully -> must NOT be redelivered.
            {
                id: 4,
                guid: "g4",
                delivered_at: "2026-09-20T10:00:00Z",
                status_code: 301,
                event: "pull_request",
            },
        ];
        const { octokit, redeliveredIds } = buildFakeOctokit(deliveries);

        const count = await redeliverFailedDeliveries(octokit);

        expect(redeliveredIds).toEqual([1]);
        expect(count).toBe(1);
    });

    it("redelivers nothing when there are no failed deliveries", async () => {
        const deliveries: FakeDelivery[] = [
            {
                id: 1,
                guid: "g1",
                delivered_at: "2026-09-20T10:00:00Z",
                status_code: 200,
                event: "pull_request",
            },
        ];
        const { octokit, redeliveredIds } = buildFakeOctokit(deliveries);

        const count = await redeliverFailedDeliveries(octokit);

        expect(redeliveredIds).toEqual([]);
        expect(count).toBe(0);
    });

    it("redelivers every eligible failed delivery, not just the first", async () => {
        const deliveries: FakeDelivery[] = [
            {
                id: 10,
                guid: "g10",
                delivered_at: "2026-09-20T09:00:00Z",
                status_code: 404,
                event: "pull_request",
            },
            {
                id: 11,
                guid: "g11",
                delivered_at: "2026-09-20T09:05:00Z",
                status_code: 503,
                event: "pull_request",
            },
        ];
        const { octokit, redeliveredIds } = buildFakeOctokit(deliveries);

        const count = await redeliverFailedDeliveries(octokit);

        expect(redeliveredIds.sort()).toEqual([10, 11]);
        expect(count).toBe(2);
    });
});
