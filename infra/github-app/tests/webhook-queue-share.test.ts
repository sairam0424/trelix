import { afterEach, describe, expect, it, vi } from "vitest";
import {
    buildHarness,
    pullRequestPayload,
    SHA_B,
    send,
    tick,
} from "./support/webhook-harness.js";

afterEach(() => {
    vi.restoreAllMocks();
});

/** One review runs at a time; an installation may keep one review waiting; the line holds four. */
const LIMITS = { capacity: 4, perGroupCapacity: 1, concurrency: 1 };

/** Installation 1 runs one review and keeps another waiting: its whole share. */
async function useUpTheShareOfInstallation1(
    h: ReturnType<typeof buildHarness>,
) {
    await send(h.app, pullRequestPayload({ prNumber: 1, installationId: 1 })); // runs
    await send(h.app, pullRequestPayload({ prNumber: 2, installationId: 1 })); // waits
}

describe("an installation that holds its share of the wait line", () => {
    it("gets the answer a full line gets: 503, the same body, the same Retry-After", async () => {
        const h = buildHarness({ limits: LIMITS });
        await useUpTheShareOfInstallation1(h);

        const res = await send(
            h.app,
            pullRequestPayload({ prNumber: 3, installationId: 1 }),
        );

        expect(res.status).toBe(503);
        expect(res.headers["retry-after"]).toBe("60");
        expect(res.body).toEqual({ error: "review queue is full" });
    });

    it("is logged on one line that names the installation id and the pull request, and nothing of the payload", async () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const h = buildHarness({ limits: LIMITS });
        await useUpTheShareOfInstallation1(h);

        await send(
            h.app,
            pullRequestPayload({
                prNumber: 3,
                installationId: 1,
                repositoryId: 7_777_777,
                sha: SHA_B,
            }),
        );

        expect(warn.mock.calls).toEqual([
            [
                "[webhook] installation 1 already has its share of the review queue waiting: answering 503 for owner/repo#3",
            ],
        ]);
    });

    it("does not start a review for the refused delivery", async () => {
        const h = buildHarness({ limits: LIMITS });
        await useUpTheShareOfInstallation1(h);

        await send(
            h.app,
            pullRequestPayload({ prNumber: 3, installationId: 1 }),
        );
        await tick();

        expect(h.calls).toHaveLength(1);
        expect(h.queue.counts()).toEqual({ waiting: 1, running: 1 });
    });

    it("leaves the rest of the line to another installation", async () => {
        const h = buildHarness({ limits: LIMITS });
        await useUpTheShareOfInstallation1(h);

        const other = await send(
            h.app,
            pullRequestPayload({ prNumber: 1, installationId: 2 }),
        );

        expect(other.status).toBe(202);
        expect(other.body).toEqual({ accepted: true });
        expect(h.queue.counts()).toEqual({ waiting: 2, running: 1 });
    });

    it("is accepted again when the refused delivery is redelivered after a waiting review has started", async () => {
        const h = buildHarness({ limits: LIMITS });
        await useUpTheShareOfInstallation1(h);
        const refused = pullRequestPayload({ prNumber: 3, installationId: 1 });
        expect((await send(h.app, refused)).status).toBe(503);
        await tick();

        h.calls[0]?.gate.resolve([]); // the waiting review starts: its share is free
        await tick();
        const redelivered = await send(h.app, refused);

        expect(redelivered.status).toBe(202);
        expect(redelivered.body).toEqual({ accepted: true });
    });

    it("still gets the duplicate answer, not a 503, for a commit that is already queued", async () => {
        const h = buildHarness({ limits: LIMITS });
        await useUpTheShareOfInstallation1(h);

        const again = await send(
            h.app,
            pullRequestPayload({ prNumber: 2, installationId: 1 }),
        );

        expect(again.status).toBe(202);
        expect(again.body).toEqual({
            ignored: true,
            reason: "a review of this commit is already queued, running or done",
        });
    });
});
