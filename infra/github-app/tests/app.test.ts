import request from "supertest";
import { sign, verify } from "@octokit/webhooks-methods";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createApp } from "../src/app.js";
import type { AppConfig } from "../src/config.js";
import type { ReviewFinding, ReviewRequest } from "../src/review-runner.js";

// Keep the real verify() and let one test make it fail.
vi.mock("@octokit/webhooks-methods", async (importOriginal) => {
    const actual =
        await importOriginal<typeof import("@octokit/webhooks-methods")>();
    return { ...actual, verify: vi.fn(actual.verify) };
});

// Values the error handler must keep out of its log line.
const HOOK_CANARY = "canary-hook-s";
const PEM_CANARY = "canary-pem-body";

const config: AppConfig = {
    appId: "1",
    privateKey: PEM_CANARY,
    webhookSecret: HOOK_CANARY,
    port: 0,
};

type RunReviewFn = (
    config: AppConfig,
    request: ReviewRequest,
) => Promise<ReviewFinding[]>;

function buildApp() {
    const lines: string[] = [];
    const runReview = vi.fn<RunReviewFn>().mockResolvedValue([]);
    const app = createApp(config, {
        runReview,
        logError: (l) => lines.push(l),
    });
    return { app, lines, runReview };
}

function pullRequestPayload(action: string) {
    return {
        action,
        number: 42,
        repository: {
            id: 4242,
            full_name: "owner/repo",
            owner: { login: "owner" },
            name: "repo",
        },
        pull_request: {
            number: 42,
            head: { sha: "0123456789abcdef0123456789abcdef01234567" },
        },
        installation: { id: 999 },
    };
}

/**
 * Posts `sentBody`, signed as `signedBody` (the same bytes unless a test
 * wants the two to differ).
 */
async function postSigned(
    app: ReturnType<typeof createApp>,
    sentBody: string,
    signedBody: string = sentBody,
) {
    const signature = await sign(config.webhookSecret, signedBody);
    return request(app)
        .post("/webhooks/github")
        .set("X-GitHub-Event", "pull_request")
        .set("Content-Type", "application/json")
        .set("X-Hub-Signature-256", signature)
        .send(sentBody);
}

// Express decides how its default handler behaves from NODE_ENV when the app
// is built; unset is the case that used to leak a stack.
beforeEach(() => {
    vi.stubEnv("NODE_ENV", undefined);
});

afterEach(() => {
    vi.unstubAllEnvs();
    vi.mocked(verify).mockClear();
});

describe("createApp /health", () => {
    it("answers 200 with the same body as before", async () => {
        const { app } = buildApp();

        const res = await request(app).get("/health");

        expect(res.status).toBe(200);
        expect(res.text).toBe('{"status":"ok"}');
        expect(res.headers["content-type"]).toBe(
            "application/json; charset=utf-8",
        );
    });

    it("leaves an unknown path a plain 404, not an error", async () => {
        const { app, lines } = buildApp();

        const res = await request(app).get("/no-such-route");

        expect(res.status).toBe(404);
        expect(lines).toHaveLength(0);
    });
});

describe("createApp webhook route", () => {
    it("passes a correctly signed delivery to the injected runReview", async () => {
        const { app, runReview } = buildApp();

        const res = await postSigned(
            app,
            JSON.stringify(pullRequestPayload("opened")),
        );

        expect(res.status).toBe(202);
        expect(res.text).toBe('{"accepted":true}');
        await new Promise((r) => setTimeout(r, 0));
        expect(runReview).toHaveBeenCalledWith(
            config,
            {
                owner: "owner",
                repo: "repo",
                prNumber: 42,
                installationId: 999,
                repositoryId: 4242,
                headSha: "0123456789abcdef0123456789abcdef01234567",
            },
            { onNoVerdict: expect.any(Function) },
        );
    });

    it("rejects a missing signature with the same 401 body", async () => {
        const { app, runReview } = buildApp();

        const res = await request(app)
            .post("/webhooks/github")
            .set("X-GitHub-Event", "pull_request")
            .set("Content-Type", "application/json")
            .send(JSON.stringify(pullRequestPayload("opened")));

        expect(res.status).toBe(401);
        expect(res.text).toBe('{"error":"missing signature or body"}');
        expect(runReview).not.toHaveBeenCalled();
    });

    describe("the signature covers the raw bytes", () => {
        const compact = JSON.stringify(pullRequestPayload("opened"));

        it("accepts the exact bytes that were signed (control)", async () => {
            const { app, runReview } = buildApp();

            const res = await postSigned(app, compact);

            expect(res.status).toBe(202);
            await new Promise((r) => setTimeout(r, 0));
            expect(runReview).toHaveBeenCalledTimes(1);
        });

        it("rejects a body changed after signing", async () => {
            const { app, runReview } = buildApp();

            const res = await postSigned(
                app,
                JSON.stringify(pullRequestPayload("closed")),
                compact,
            );

            expect(res.status).toBe(401);
            expect(res.text).toBe('{"error":"signature verification failed"}');
            expect(runReview).not.toHaveBeenCalled();
        });

        // Both bodies parse to the same object, so a check made on the parsed
        // body (or on JSON.stringify of it) would let these through.
        it.each([
            [
                "re-indented",
                JSON.stringify(pullRequestPayload("opened"), null, 2),
            ],
            [
                "with an escaped character",
                compact.replace("owner", "\\u006fwner"),
            ],
            ["with a trailing newline", `${compact}\n`],
        ])("rejects the same JSON %s", async (_name, reformatted) => {
            const { app, runReview } = buildApp();

            const res = await postSigned(app, reformatted, compact);

            expect(res.status).toBe(401);
            expect(runReview).not.toHaveBeenCalled();
        });
    });
});

describe("createApp body errors", () => {
    it("answers malformed JSON with a fixed 400 that quotes nothing", async () => {
        const { app, lines, runReview } = buildApp();

        const res = await postSigned(app, '{"canary-body": ');

        expect(res.status).toBe(400);
        expect(res.text).toBe('{"error":"bad request"}');
        expect(runReview).not.toHaveBeenCalled();
        expect(lines).toHaveLength(1);
        expect(lines[0]).toContain('"status":400');
        expect(lines[0]).not.toContain("canary-body");
    });

    it("answers a body over the 25MB cap with a fixed 413", async () => {
        const { app, lines, runReview } = buildApp();
        const oversized = JSON.stringify({
            ...pullRequestPayload("opened"),
            padding: "x".repeat(26 * 1024 * 1024),
        });

        const res = await request(app)
            .post("/webhooks/github")
            .set("X-GitHub-Event", "pull_request")
            .set("Content-Type", "application/json")
            .send(oversized);

        expect(res.status).toBe(413);
        expect(res.text).toBe('{"error":"payload too large"}');
        expect(runReview).not.toHaveBeenCalled();
        expect(lines).toHaveLength(1);
    }, 30_000);

    // The injected logger is a public dependency. If it throws, Express's own
    // handler would take over and, with NODE_ENV unset, answer with the stack.
    it("still answers malformed JSON with the fixed 400 when the logger throws", async () => {
        const spy = vi.spyOn(console, "error").mockImplementation(() => {});
        const app = createApp(config, {
            runReview: vi.fn<RunReviewFn>().mockResolvedValue([]),
            logError: () => {
                throw new Error("canary log sink is down at /canary/sink.ts");
            },
        });

        try {
            const res = await postSigned(app, '{"canary-body": ');

            expect(res.status).toBe(400);
            expect(res.headers["content-type"]).toMatch(/^application\/json/);
            expect(res.text).toBe('{"error":"bad request"}');
            expect(spy.mock.calls).toEqual([
                ["[app] request failed; the error could not be logged"],
            ]);
        } finally {
            spy.mockRestore();
        }
    });

    // An async transport returns a promise. Nobody awaits it, so a rejection
    // must be handled the way a throw is, or it ends the process.
    it("still answers malformed JSON with the fixed 400 when the logger's promise rejects", async () => {
        const spy = vi.spyOn(console, "error").mockImplementation(() => {});
        const app = createApp(config, {
            runReview: vi.fn<RunReviewFn>().mockResolvedValue([]),
            logError: () =>
                Promise.reject(
                    new Error("canary log sink is down at /canary/sink.ts"),
                ),
        });

        try {
            const res = await postSigned(app, '{"canary-body": ');
            await new Promise((resolve) => setImmediate(resolve));

            expect(res.status).toBe(400);
            expect(res.text).toBe('{"error":"bad request"}');
            expect(spy.mock.calls).toEqual([
                ["[app] request failed; the error could not be logged"],
            ]);
        } finally {
            spy.mockRestore();
        }
    });

    it("answers an unsupported charset with a fixed 415", async () => {
        const { app } = buildApp();

        const res = await request(app)
            .post("/webhooks/github")
            .set("X-GitHub-Event", "pull_request")
            .set("Content-Type", "application/json; charset=canary-charset")
            .send("{}");

        expect(res.status).toBe(415);
        expect(res.text).toBe('{"error":"unsupported media type"}');
    });
});

describe("createApp unexpected errors", () => {
    const failure = () =>
        new Error(`verify failed with ${HOOK_CANARY} and ${PEM_CANARY}`);

    it("answers an error thrown in the route with the fixed 500 body and no stack", async () => {
        const { app, lines, runReview } = buildApp();
        vi.mocked(verify).mockRejectedValueOnce(failure());

        const res = await postSigned(
            app,
            JSON.stringify(pullRequestPayload("opened")),
        );

        expect(res.status).toBe(500);
        expect(res.text).toBe('{"error":"internal server error"}');
        expect(res.text).not.toContain("verify failed");
        expect(runReview).not.toHaveBeenCalled();
        expect(lines).toHaveLength(1);
        expect(lines[0]).toContain("verify failed with [redacted]");
    });

    it("keeps both configured secrets out of the log", async () => {
        const { app, lines } = buildApp();
        vi.mocked(verify).mockRejectedValueOnce(failure());

        await postSigned(app, JSON.stringify(pullRequestPayload("opened")));

        expect(lines[0]).not.toContain(HOOK_CANARY);
        expect(lines[0]).not.toContain(PEM_CANARY);
    });

    it("does not log the signature header or the request body", async () => {
        const { app, lines } = buildApp();
        vi.mocked(verify).mockRejectedValueOnce(failure());
        const body = JSON.stringify({
            ...pullRequestPayload("opened"),
            note: "canary-body",
        });
        const signature = await sign(config.webhookSecret, body);

        await postSigned(app, body);

        expect(lines[0]).not.toContain(signature);
        expect(lines[0]).not.toContain("canary-body");
    });

    it("writes the line to console.error when no logger is given", async () => {
        const spy = vi.spyOn(console, "error").mockImplementation(() => {});
        const app = createApp(config, {
            runReview: vi.fn<RunReviewFn>().mockResolvedValue([]),
        });
        vi.mocked(verify).mockRejectedValueOnce(failure());

        try {
            await postSigned(app, JSON.stringify(pullRequestPayload("opened")));

            expect(spy).toHaveBeenCalledTimes(1);
            expect(String(spy.mock.calls[0]?.[0])).toContain(
                "[app] request failed",
            );
        } finally {
            spy.mockRestore();
        }
    });
});
