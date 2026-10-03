import { afterEach, describe, expect, it, vi } from "vitest";
import type { InstallPolicy } from "../src/policy.js";
import {
    buildHarness,
    pullRequestPayload,
    send,
    tick,
} from "./support/webhook-harness.js";

afterEach(() => {
    vi.restoreAllMocks();
});

function allowlist(accounts: string[], installations: number[]): InstallPolicy {
    return {
        mode: "allowlist",
        accounts: new Set(accounts),
        installations: new Set(installations),
    };
}

describe("the kill switch", () => {
    function disabled() {
        return buildHarness({ controls: { reviewsEnabled: false } });
    }

    it("answers 202 {ignored:true} and starts no review", async () => {
        vi.spyOn(console, "warn").mockImplementation(() => {});
        const h = disabled();

        const res = await send(h.app, pullRequestPayload());
        await tick();

        expect(res.status).toBe(202);
        expect(res.text).toBe('{"ignored":true}');
        expect(h.calls).toHaveLength(0);
        expect(h.queue.counts()).toEqual({ waiting: 0, running: 0 });
    });

    it("logs one line for each delivery it ignores", async () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const h = disabled();

        await send(h.app, pullRequestPayload({ prNumber: 1 }));
        await send(h.app, pullRequestPayload({ prNumber: 2 }));

        expect(warn.mock.calls).toEqual([
            [
                "[webhook] reviews are disabled: ignoring pull_request delivery for owner/repo",
            ],
            [
                "[webhook] reviews are disabled: ignoring pull_request delivery for owner/repo",
            ],
        ]);
    });

    it("takes nothing from the dedupe memory: the same delivery is reviewed once it is switched back on", async () => {
        vi.spyOn(console, "warn").mockImplementation(() => {});
        const off = disabled();
        await send(off.app, pullRequestPayload());

        const on = buildHarness();
        const res = await send(on.app, pullRequestPayload());

        expect(res.body).toEqual({ accepted: true });
    });

    it("stays quiet for events that would not have been reviewed anyway", async () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const h = disabled();

        await send(h.app, pullRequestPayload({ action: "closed" }));
        await send(h.app, { zen: "ping" }, { event: "ping" });

        expect(warn).not.toHaveBeenCalled();
    });

    it("still rejects a bad signature with 401", async () => {
        const h = disabled();
        const supertest = (await import("supertest")).default;

        const res = await supertest(h.app)
            .post("/webhooks/github")
            .set("X-GitHub-Event", "pull_request")
            .set("Content-Type", "application/json")
            .send(JSON.stringify(pullRequestPayload()));

        expect(res.status).toBe(401);
    });

    it("lets reviews through when it is on", async () => {
        const h = buildHarness({ controls: { reviewsEnabled: true } });

        const res = await send(h.app, pullRequestPayload());

        expect(res.body).toEqual({ accepted: true });
    });
});

describe("the installation policy", () => {
    it("open: serves any account", async () => {
        const h = buildHarness();

        const res = await send(
            h.app,
            pullRequestPayload({ owner: "a-stranger", installationId: 31337 }),
        );

        expect(res.body).toEqual({ accepted: true });
    });

    it("allowlist: serves a listed account, whatever the case of the login", async () => {
        const h = buildHarness({
            controls: { installPolicy: allowlist(["sairam0424"], []) },
        });

        const res = await send(
            h.app,
            pullRequestPayload({ owner: "Sairam0424" }),
        );

        expect(res.body).toEqual({ accepted: true });
    });

    it("allowlist: serves a listed installation id under any account", async () => {
        const h = buildHarness({
            controls: { installPolicy: allowlist([], [999]) },
        });

        const res = await send(
            h.app,
            pullRequestPayload({ owner: "someone", installationId: 999 }),
        );

        expect(res.body).toEqual({ accepted: true });
    });

    describe("an installation outside the allow-list", () => {
        const policy = allowlist(["sairam0424"], [111]);

        it("gets a silent 202 {ignored:true}: the body is the kill switch's, and says nothing of a policy", async () => {
            vi.spyOn(console, "warn").mockImplementation(() => {});
            const h = buildHarness({ controls: { installPolicy: policy } });

            const res = await send(
                h.app,
                pullRequestPayload({ owner: "stranger", installationId: 222 }),
            );

            expect(res.status).toBe(202);
            expect(res.text).toBe('{"ignored":true}');
            expect(Object.keys(res.headers)).not.toContain("retry-after");
        });

        it("starts no review and leaves nothing in the queue or the dedupe memory", async () => {
            vi.spyOn(console, "warn").mockImplementation(() => {});
            const h = buildHarness({ controls: { installPolicy: policy } });

            await send(
                h.app,
                pullRequestPayload({ owner: "stranger", installationId: 222 }),
            );
            await tick();

            expect(h.calls).toHaveLength(0);
            expect(h.queue.counts()).toEqual({ waiting: 0, running: 0 });
            expect(h.queue.claimCounts()).toEqual({ inFlight: 0, kept: 0 });
        });

        it("is logged once, naming the installation", async () => {
            const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
            const h = buildHarness({ controls: { installPolicy: policy } });

            await send(
                h.app,
                pullRequestPayload({ owner: "stranger", installationId: 222 }),
            );

            expect(warn.mock.calls).toEqual([
                [
                    "[webhook] ignoring pull_request delivery for stranger/repo#42: installation 222 is not allowed by the install policy",
                ],
            ]);
        });

        it("is not refused because of a listed installation that belongs to someone else", async () => {
            const h = buildHarness({ controls: { installPolicy: policy } });

            const res = await send(
                h.app,
                pullRequestPayload({
                    owner: "sairam0424",
                    installationId: 222,
                }),
            );

            expect(res.body).toEqual({ accepted: true });
        });
    });

    it("allowlist with both lists empty: serves no one", async () => {
        vi.spyOn(console, "warn").mockImplementation(() => {});
        const h = buildHarness({
            controls: { installPolicy: allowlist([], []) },
        });

        const res = await send(h.app, pullRequestPayload());
        await tick();

        expect(res.body).toEqual({ ignored: true });
        expect(h.calls).toHaveLength(0);
    });
});

describe("a delivery the review cannot run for", () => {
    const warnText = (payload: object) =>
        `[webhook] ignoring pull_request delivery for ${
            (payload as { repository?: { full_name?: string } }).repository
                ?.full_name ?? "unknown"
        }: `;

    const USABLE_FIELDS_REASON =
        "pull_request payload has no usable owner, repository name, pull request number or installation id";

    it.each([
        ["no installation", pullRequestPayload({ installationId: null })],
        ["an installation id of 0", pullRequestPayload({ installationId: 0 })],
        [
            "an installation id sent as a string",
            pullRequestPayload({ installationId: "999" as unknown as number }),
        ],
        [
            "no pull request number",
            {
                ...pullRequestPayload(),
                pull_request: {
                    head: { sha: pullRequestPayload().pull_request.head.sha },
                },
            },
        ],
        ["a pull request number of 0", pullRequestPayload({ prNumber: 0 })],
        [
            "no owner",
            {
                ...pullRequestPayload(),
                repository: {
                    ...pullRequestPayload().repository,
                    owner: undefined,
                },
            },
        ],
        [
            "an owner login that is not text",
            pullRequestPayload({ owner: 5 as unknown as string }),
        ],
        ["an empty owner login", pullRequestPayload({ owner: "" })],
        ["an empty repository name", pullRequestPayload({ repo: "" })],
    ])(
        "is acknowledged and ignored, with no review, for %s",
        async (_name, payload) => {
            const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
            const h = buildHarness();

            const res = await send(h.app, payload);
            await tick();

            expect(res.status).toBe(202);
            expect(res.body).toEqual({
                ignored: true,
                reason: USABLE_FIELDS_REASON,
            });
            expect(h.calls).toHaveLength(0);
            expect(warn).toHaveBeenCalledTimes(1);
            expect(String(warn.mock.calls[0]?.[0])).toContain(
                warnText(payload) + USABLE_FIELDS_REASON,
            );
        },
    );

    it("does not let a hostile repository name reach the log unbounded or with control characters", async () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const h = buildHarness();
        const hostile = `evil\n\u001b[31mred${" ".repeat(50_000)}x`;
        const payload = {
            ...pullRequestPayload({ installationId: null }),
            repository: {
                ...pullRequestPayload().repository,
                full_name: hostile,
            },
        };
        const started = performance.now();

        const res = await send(h.app, payload);

        expect(res.status).toBe(202);
        expect(performance.now() - started).toBeLessThan(1_000);
        const line = String(warn.mock.calls[0]?.[0]);
        expect(line.length).toBeLessThan(400);
        expect(line).not.toContain("\n");
        expect(line).not.toContain("\u001b");
        expect(line).toContain("evil?");
    });

    it("cuts a hostile owner and repository name in the log line of a policy denial", async () => {
        const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
        const h = buildHarness({
            controls: { installPolicy: allowlist([], []) },
        });

        await send(
            h.app,
            pullRequestPayload({
                owner: `o\n${"y".repeat(50_000)}`,
                repo: `r\u0007${"z".repeat(50_000)}`,
            }),
        );

        const line = String(warn.mock.calls[0]?.[0]);
        expect(line.length).toBeLessThan(500);
        expect(line).not.toContain("\n");
        expect(line).not.toContain("\u0007");
    });
});
