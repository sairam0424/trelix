import { afterEach, describe, expect, it, vi } from "vitest";
import {
    buildHarness,
    DAY_MS,
    HOOK_CANARY,
    PEM_CANARY,
    pullRequestPayload,
    SHA_A,
    SHA_B,
    send,
    tick,
} from "./support/webhook-harness.js";

afterEach(() => {
    vi.restoreAllMocks();
});

describe("acknowledging a delivery", () => {
    it("answers 202 while the review is still running: the ack does not wait for the work", async () => {
        const h = buildHarness();
        const started = Date.now();

        const res = await send(h.app, pullRequestPayload());

        expect(res.status).toBe(202);
        expect(res.body).toEqual({ accepted: true });
        expect(Date.now() - started).toBeLessThan(2_000);
        await tick();
        expect(h.calls).toHaveLength(1); // started, and not finished
        expect(h.queue.counts()).toEqual({ waiting: 0, running: 1 });
    });

    it("runs the review with the request built from the delivery", async () => {
        const h = buildHarness();

        await send(h.app, pullRequestPayload({ prNumber: 7, sha: SHA_B }));
        await tick();

        expect(h.calls[0]?.request).toEqual({
            owner: "owner",
            repo: "repo",
            prNumber: 7,
            installationId: 999,
            repositoryId: 4242,
            headSha: SHA_B,
        });
    });
});

describe("dedupe", () => {
    it("runs one review when the same commit is delivered twice under different delivery GUIDs", async () => {
        const h = buildHarness();

        const first = await send(h.app, pullRequestPayload(), { guid: "g-1" });
        const second = await send(h.app, pullRequestPayload(), { guid: "g-2" });
        await tick();

        expect(first.body).toEqual({ accepted: true });
        expect(second.status).toBe(202);
        expect(second.body).toEqual({
            ignored: true,
            reason: "a review of this commit is already queued, running or done",
        });
        expect(h.calls).toHaveLength(1);
    });

    it("logs a refused duplicate on one line", async () => {
        const log = vi.spyOn(console, "log").mockImplementation(() => {});
        const h = buildHarness();
        await send(h.app, pullRequestPayload());

        await send(h.app, pullRequestPayload());

        expect(log.mock.calls).toEqual([
            [
                `[webhook] not queueing owner/repo#42 at ${SHA_A}: a review of this commit is already queued, running or done`,
            ],
        ]);
    });

    it("runs one review when a delivery is redelivered under the same GUID", async () => {
        const h = buildHarness();

        await send(h.app, pullRequestPayload(), { guid: "same-guid" });
        const redelivered = await send(h.app, pullRequestPayload(), {
            guid: "same-guid",
        });
        await tick();

        expect(redelivered.body).toMatchObject({ ignored: true });
        expect(h.calls).toHaveLength(1);
    });

    it("does not dedupe on the delivery GUID: a new commit under a reused GUID is reviewed", async () => {
        const h = buildHarness({
            limits: { concurrency: 4, perGroupConcurrency: 4 },
        });

        const first = await send(h.app, pullRequestPayload({ sha: SHA_A }), {
            guid: "reused-guid",
        });
        const second = await send(h.app, pullRequestPayload({ sha: SHA_B }), {
            guid: "reused-guid",
        });
        await tick();

        expect(first.body).toEqual({ accepted: true });
        expect(second.body).toEqual({ accepted: true });
        expect(h.calls.map((call) => call.request.headSha)).toEqual([
            SHA_A,
            SHA_B,
        ]);
    });

    it.each([
        ["another pull request", { prNumber: 43 }],
        ["another repository", { repositoryId: 4243 }],
        ["another installation", { installationId: 1000 }],
        ["another head commit", { sha: SHA_B }],
    ])("treats %s as a different review", async (_name, change) => {
        const h = buildHarness({
            limits: { concurrency: 4, perGroupConcurrency: 4 },
        });

        await send(h.app, pullRequestPayload());
        const other = await send(h.app, pullRequestPayload(change));
        await tick();

        expect(other.body).toEqual({ accepted: true });
        expect(h.calls).toHaveLength(2);
    });

    it("claims the commit before the job is queued: a duplicate that arrives while the first is still waiting is refused", async () => {
        const h = buildHarness({ limits: { concurrency: 1 } });
        await send(h.app, pullRequestPayload({ prNumber: 1 })); // runs
        await send(h.app, pullRequestPayload({ prNumber: 2 })); // waits

        const duplicate = await send(
            h.app,
            pullRequestPayload({ prNumber: 2 }),
        );

        expect(duplicate.body).toMatchObject({ ignored: true });
        expect(h.queue.counts()).toEqual({ waiting: 1, running: 1 });
    });

    it("keeps the claim 24 hours after a review that finished, then reviews the commit again", async () => {
        const h = buildHarness();
        await send(h.app, pullRequestPayload());
        await tick();
        h.calls[0]?.gate.resolve([]);
        await tick();

        const soon = await send(h.app, pullRequestPayload());
        expect(soon.body).toMatchObject({ ignored: true });

        h.clock.now += DAY_MS;
        const later = await send(h.app, pullRequestPayload());
        await tick();
        expect(later.body).toEqual({ accepted: true });
        expect(h.calls).toHaveLength(2);
    });

    it("releases the claim when the review fails, so the redelivery is reviewed", async () => {
        vi.spyOn(console, "error").mockImplementation(() => {});
        const h = buildHarness();
        await send(h.app, pullRequestPayload());
        await tick();
        h.calls[0]?.gate.reject(new Error("checkout failed"));
        await tick();

        const redelivered = await send(h.app, pullRequestPayload());
        await tick();

        expect(redelivered.body).toEqual({ accepted: true });
        expect(h.calls).toHaveLength(2);
    });

    it("releases the claim when the review ends without a verdict Check, so the commit can be reviewed again", async () => {
        const h = buildHarness();
        await send(h.app, pullRequestPayload());
        await tick();
        h.calls[0]?.options?.onNoVerdict?.();
        h.calls[0]?.gate.resolve([]);
        await tick();

        const redelivered = await send(h.app, pullRequestPayload());
        await tick();

        expect(redelivered.body).toEqual({ accepted: true });
        expect(h.calls).toHaveLength(2);
    });

    it("keeps the claim when the review ends with a verdict", async () => {
        const h = buildHarness();
        await send(h.app, pullRequestPayload());
        await tick();
        h.calls[0]?.gate.resolve([]);
        await tick();

        const redelivered = await send(h.app, pullRequestPayload());

        expect(redelivered.body).toMatchObject({ ignored: true });
    });
});

describe("a full queue", () => {
    async function fillQueue(h: ReturnType<typeof buildHarness>) {
        await send(
            h.app,
            pullRequestPayload({ prNumber: 1, installationId: 1 }),
        ); // runs
        await send(
            h.app,
            pullRequestPayload({ prNumber: 2, installationId: 1 }),
        ); // waits
    }

    it("answers 503 with a Retry-After header", async () => {
        const h = buildHarness({ limits: { capacity: 1, concurrency: 1 } });
        await fillQueue(h);

        const res = await send(
            h.app,
            pullRequestPayload({ prNumber: 3, installationId: 1 }),
        );

        expect(res.status).toBe(503);
        expect(res.headers["retry-after"]).toBe("60");
        expect(res.body).toEqual({ error: "review queue is full" });
    });

    it("logs the refusal on one line", async () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const h = buildHarness({ limits: { capacity: 1, concurrency: 1 } });
        await fillQueue(h);

        await send(
            h.app,
            pullRequestPayload({ prNumber: 3, installationId: 1 }),
        );

        expect(warn.mock.calls).toEqual([
            ["[webhook] review queue is full: answering 503 for owner/repo#3"],
        ]);
    });

    it("does not start a review for the refused delivery", async () => {
        const h = buildHarness({ limits: { capacity: 1, concurrency: 1 } });
        await fillQueue(h);

        await send(
            h.app,
            pullRequestPayload({ prNumber: 3, installationId: 1 }),
        );
        await tick();

        expect(h.calls).toHaveLength(1);
        expect(h.queue.counts()).toEqual({ waiting: 1, running: 1 });
    });

    it("accepts the refused delivery when it is redelivered after room has opened", async () => {
        const h = buildHarness({ limits: { capacity: 1, concurrency: 1 } });
        await fillQueue(h);
        const refused = pullRequestPayload({ prNumber: 3, installationId: 1 });
        expect((await send(h.app, refused)).status).toBe(503);
        await tick();

        h.calls[0]?.gate.resolve([]); // room opens: the waiting review starts
        await tick();
        const redelivered = await send(h.app, refused);

        expect(redelivered.status).toBe(202);
        expect(redelivered.body).toEqual({ accepted: true });
    });

    it("stays within its bounds under a flood of 500 distinct deliveries", async () => {
        const h = buildHarness({
            limits: { capacity: 20, concurrency: 2, perGroupConcurrency: 1 },
        });
        const statuses: number[] = [];

        for (let i = 1; i <= 500; i += 1) {
            const res = await send(
                h.app,
                pullRequestPayload({ prNumber: i, installationId: i }),
            );
            statuses.push(res.status);
        }
        await tick();

        expect(statuses.filter((s) => s === 202)).toHaveLength(22);
        expect(statuses.filter((s) => s === 503)).toHaveLength(478);
        expect(h.queue.counts()).toEqual({ waiting: 20, running: 2 });
        expect(h.calls).toHaveLength(2);
    }, 30_000);
});

describe("a queue that is shutting down", () => {
    it("answers 503 with a Retry-After header, so the redelivery backstop sends the delivery again", async () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const h = buildHarness();
        await h.queue.shutdown(1_000);

        const res = await send(h.app, pullRequestPayload());

        expect(res.status).toBe(503);
        expect(res.headers["retry-after"]).toBe("60");
        expect(res.body).toEqual({ error: "service is shutting down" });
        expect(h.calls).toHaveLength(0);
        expect(warn.mock.calls).toEqual([
            ["[webhook] shutting down: answering 503 for owner/repo#42"],
        ]);
    });
});

describe("concurrency caps", () => {
    it("runs the reviews of one installation one at a time", async () => {
        const h = buildHarness({ limits: { concurrency: 4 } });
        await send(
            h.app,
            pullRequestPayload({ prNumber: 1, installationId: 5 }),
        );
        await send(
            h.app,
            pullRequestPayload({ prNumber: 2, installationId: 5 }),
        );
        await tick();

        expect(h.calls.map((c) => c.request.prNumber)).toEqual([1]);

        h.calls[0]?.gate.resolve([]);
        await tick();
        expect(h.calls.map((c) => c.request.prNumber)).toEqual([1, 2]);
    });

    it("runs at most two reviews at once across installations", async () => {
        const h = buildHarness();
        for (const installationId of [1, 2, 3]) {
            await send(
                h.app,
                pullRequestPayload({
                    prNumber: installationId,
                    installationId,
                }),
            );
        }
        await tick();

        expect(h.calls.map((c) => c.request.installationId)).toEqual([1, 2]);

        h.calls[1]?.gate.resolve([]);
        await tick();
        expect(h.calls.map((c) => c.request.installationId)).toEqual([1, 2, 3]);
    });

    it("does not let one installation's waiting review hold up another installation", async () => {
        const h = buildHarness();
        await send(
            h.app,
            pullRequestPayload({ prNumber: 1, installationId: 1 }),
        );
        await send(
            h.app,
            pullRequestPayload({ prNumber: 2, installationId: 1 }),
        );
        await send(
            h.app,
            pullRequestPayload({ prNumber: 3, installationId: 2 }),
        );
        await tick();

        expect(h.calls.map((c) => c.request.prNumber)).toEqual([1, 3]);
    });
});

describe("a job that fails", () => {
    const SECRET_LEAK = `token leaked: ${HOOK_CANARY} and ${PEM_CANARY}`;

    it("is logged once, with the secrets removed, and does not change the 202", async () => {
        const consoleError = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const h = buildHarness();
        const res = await send(h.app, pullRequestPayload());
        await tick();

        h.calls[0]?.gate.reject(new Error(SECRET_LEAK));
        await tick();

        expect(res.status).toBe(202);
        expect(consoleError).toHaveBeenCalledTimes(1);
        const line = String(consoleError.mock.calls[0]?.[0]);
        expect(line.startsWith("[webhook] review failed {")).toBe(true);
        expect(line).toContain('"job":"owner/repo#42"');
        expect(line).toContain('"error":"Error"');
        expect(line).toContain("token leaked: [redacted] and [redacted]");
        expect(line).not.toContain(HOOK_CANARY);
        expect(line).not.toContain(PEM_CANARY);
        expect(line).not.toContain("\n");
    });

    it("is logged with the secrets removed when it rejects with a string", async () => {
        const consoleError = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const h = buildHarness();
        await send(h.app, pullRequestPayload());
        await tick();

        h.calls[0]?.gate.reject(SECRET_LEAK);
        await tick();

        const line = String(consoleError.mock.calls[0]?.[0]);
        expect(line).toContain("token leaked: [redacted] and [redacted]");
        expect(line).not.toContain(HOOK_CANARY);
    });

    it("does not stop the worker: the next delivery is reviewed", async () => {
        vi.spyOn(console, "error").mockImplementation(() => {});
        const h = buildHarness();
        await send(h.app, pullRequestPayload({ prNumber: 1 }));
        await tick();
        h.calls[0]?.gate.reject(new Error("boom"));
        await tick();

        await send(h.app, pullRequestPayload({ prNumber: 2 }));
        await tick();

        expect(h.calls.map((c) => c.request.prNumber)).toEqual([1, 2]);
    });

    it("is handled when the runner throws before it returns a promise", async () => {
        const consoleError = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        const h = buildHarness({
            runReview: () => {
                throw new Error("sync failure");
            },
        });

        const res = await send(h.app, pullRequestPayload());
        await tick();

        expect(res.status).toBe(202);
        expect(consoleError).toHaveBeenCalledTimes(1);
        expect(String(consoleError.mock.calls[0]?.[0])).toContain(
            "sync failure",
        );
    });

    it("raises no unhandled rejection", async () => {
        vi.spyOn(console, "error").mockImplementation(() => {});
        const unhandled = vi.fn();
        process.on("unhandledRejection", unhandled);
        try {
            const h = buildHarness();
            await send(h.app, pullRequestPayload());
            await tick();

            h.calls[0]?.gate.reject(new Error("boom"));
            await tick();
            await tick();

            expect(unhandled).not.toHaveBeenCalled();
        } finally {
            process.off("unhandledRejection", unhandled);
        }
    });
});
