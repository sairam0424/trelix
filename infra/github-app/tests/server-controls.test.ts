import express from "express";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
    pullRequestPayload,
    send,
    tick,
    HOOK_CANARY,
    PEM_CANARY,
} from "./support/webhook-harness.js";

/**
 * The deployed entry point reads the abuse controls from the environment and
 * hands them, with the queue it will drain on shutdown, to the app. These tests
 * import `server.ts` under different environments and drive the app it built,
 * so they fail if any link (env -> controls -> router -> queue -> shutdown) is cut.
 */

const mocks = vi.hoisted(() => ({
    sweepStaleWorkspaces: vi.fn(async () => undefined),
    installShutdownHandlers: vi.fn(),
    runReview: vi.fn(() => new Promise<never[]>(() => {})),
}));

vi.mock("../src/repo-checkout.js", async (importOriginal) => ({
    ...(await importOriginal<typeof import("../src/repo-checkout.js")>()),
    sweepStaleWorkspaces: mocks.sweepStaleWorkspaces,
}));
vi.mock("../src/shutdown.js", () => ({
    installShutdownHandlers: mocks.installShutdownHandlers,
}));
vi.mock("../src/review-runner.js", () => ({ runReview: mocks.runReview }));

type Env = Record<string, string>;

const CONTROL_VARIABLES = [
    "TRELIX_APP_REVIEWS_ENABLED",
    "TRELIX_APP_INSTALL_POLICY",
    "TRELIX_APP_ALLOWED_ACCOUNTS",
    "TRELIX_APP_ALLOWED_INSTALLATIONS",
    "TRELIX_APP_QUEUE_CAPACITY",
    "TRELIX_APP_CONCURRENCY",
    "TRELIX_APP_CONCURRENCY_PER_INSTALLATION",
];

let listen: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
    vi.resetModules();
    mocks.sweepStaleWorkspaces.mockClear();
    mocks.installShutdownHandlers.mockClear();
    mocks.runReview.mockClear();
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.stubEnv("NODE_ENV", undefined);
    vi.stubEnv("GITHUB_APP_ID", "1");
    vi.stubEnv("GITHUB_APP_PRIVATE_KEY", PEM_CANARY);
    vi.stubEnv("GITHUB_WEBHOOK_SECRET", HOOK_CANARY);
    vi.stubEnv("PORT", "4321");
    for (const name of CONTROL_VARIABLES) {
        vi.stubEnv(name, undefined);
    }
    listen = vi.spyOn(express.application, "listen").mockReturnThis();
});

afterEach(() => {
    listen.mockRestore();
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
});

/** Imports the entry point under `env` and returns the app it built. */
async function startServer(env: Env = {}): Promise<express.Express> {
    for (const [name, value] of Object.entries(env)) {
        vi.stubEnv(name, value);
    }
    await import("../src/server.js");
    return listen.mock.contexts[0] as express.Express;
}

describe("server.ts and the abuse controls", () => {
    it("refuses to start under an unknown install policy: nothing is swept and nothing listens", async () => {
        vi.stubEnv("TRELIX_APP_INSTALL_POLICY", "canary-policy");

        const start = import("../src/server.js");

        await expect(start).rejects.toThrow(
            'Invalid TRELIX_APP_INSTALL_POLICY: expected "open" or "allowlist"',
        );
        await expect(start).rejects.not.toThrow("canary-policy");
        expect(listen).not.toHaveBeenCalled();
        expect(mocks.sweepStaleWorkspaces).not.toHaveBeenCalled();
    });

    it("warns loudly at startup that the default policy is open", async () => {
        await startServer();

        expect(console.warn).toHaveBeenCalledWith(
            expect.stringContaining(
                "[config] WARNING: TRELIX_APP_INSTALL_POLICY is open",
            ),
        );
    });

    it("stays quiet about the policy when an allow-list is set", async () => {
        await startServer({
            TRELIX_APP_INSTALL_POLICY: "allowlist",
            TRELIX_APP_ALLOWED_ACCOUNTS: "owner",
        });

        expect(console.warn).not.toHaveBeenCalled();
    });

    it("hands the queue it built, the server and the process to the shutdown handlers", async () => {
        const app = await startServer();

        expect(mocks.installShutdownHandlers).toHaveBeenCalledTimes(1);
        const deps = mocks.installShutdownHandlers.mock.calls[0]?.[0] as {
            server: unknown;
            queue: { shutdown: unknown; counts: () => unknown };
            process: unknown;
        };
        expect(deps.server).toBe(app);
        expect(typeof deps.queue.shutdown).toBe("function");
        expect(deps.process).toBe(process);
    });

    it("the queue it drains is the queue the webhook runs reviews on", async () => {
        const app = await startServer();
        const deps = mocks.installShutdownHandlers.mock.calls[0]?.[0] as {
            queue: { counts: () => { waiting: number; running: number } };
        };

        const res = await send(app, pullRequestPayload());
        await tick();

        expect(res.body).toEqual({ accepted: true });
        expect(mocks.runReview).toHaveBeenCalledTimes(1);
        expect(deps.queue.counts()).toEqual({ waiting: 0, running: 1 });
    });

    it("TRELIX_APP_REVIEWS_ENABLED=false makes the webhook ignore every pull request", async () => {
        const app = await startServer({ TRELIX_APP_REVIEWS_ENABLED: "false" });

        const res = await send(app, pullRequestPayload());
        await tick();

        expect(res.status).toBe(202);
        expect(res.text).toBe('{"ignored":true}');
        expect(mocks.runReview).not.toHaveBeenCalled();
    });

    it("TRELIX_APP_INSTALL_POLICY=allowlist serves only the listed account", async () => {
        const app = await startServer({
            TRELIX_APP_INSTALL_POLICY: "allowlist",
            TRELIX_APP_ALLOWED_ACCOUNTS: " Owner ",
        });

        const allowed = await send(app, pullRequestPayload({ owner: "owner" }));
        const denied = await send(
            app,
            pullRequestPayload({ owner: "stranger", prNumber: 43 }),
        );
        await tick();

        expect(allowed.body).toEqual({ accepted: true });
        expect(denied.text).toBe('{"ignored":true}');
        expect(mocks.runReview).toHaveBeenCalledTimes(1);
    });

    it("TRELIX_APP_INSTALL_POLICY=allowlist serves a listed installation id", async () => {
        const app = await startServer({
            TRELIX_APP_INSTALL_POLICY: "allowlist",
            TRELIX_APP_ALLOWED_INSTALLATIONS: "12345",
        });

        const allowed = await send(
            app,
            pullRequestPayload({ owner: "anyone", installationId: 12345 }),
        );
        const denied = await send(
            app,
            pullRequestPayload({ owner: "anyone", installationId: 999 }),
        );

        expect(allowed.body).toEqual({ accepted: true });
        expect(denied.text).toBe('{"ignored":true}');
    });

    it("the queue caps come from the environment: capacity 1 and concurrency 1 refuse the third delivery with a 503", async () => {
        const app = await startServer({
            TRELIX_APP_QUEUE_CAPACITY: "1",
            TRELIX_APP_CONCURRENCY: "1",
        });

        const statuses: number[] = [];
        for (const installationId of [1, 2, 3]) {
            const res = await send(
                app,
                pullRequestPayload({
                    prNumber: installationId,
                    installationId,
                }),
            );
            statuses.push(res.status);
        }

        expect(statuses).toEqual([202, 202, 503]);
    });

    it("the default caps are 20 waiting and 2 running", async () => {
        const app = await startServer();
        const statuses: number[] = [];

        for (let i = 1; i <= 23; i += 1) {
            const res = await send(
                app,
                pullRequestPayload({ prNumber: i, installationId: i }),
            );
            statuses.push(res.status);
        }

        expect(statuses.filter((s) => s === 202)).toHaveLength(22);
        expect(statuses.at(-1)).toBe(503);
    }, 20_000);
});
