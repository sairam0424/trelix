import express from "express";
import request from "supertest";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

// The real sweep deletes trelix-review-* directories under the temp dir.
const mocks = vi.hoisted(() => ({
    sweepStaleWorkspaces: vi.fn(async () => undefined),
}));

vi.mock("../src/repo-checkout.js", async (importOriginal) => ({
    ...(await importOriginal<typeof import("../src/repo-checkout.js")>()),
    sweepStaleWorkspaces: mocks.sweepStaleWorkspaces,
}));

// Values the error handler must keep out of its log line.
const HOOK_CANARY = "canary-hook-s";
const PEM_CANARY = "canary-pem-body";

/**
 * `server.ts` is the only place the deployed app is assembled, and it listens
 * as soon as it is imported. These tests import it once, with `listen`
 * replaced, and drive the app it built: the one that answers real requests.
 * Without `createApp` it would be the pre-handler shape, where Express's own
 * handler answers an error with the stack.
 */
describe("server entry point", () => {
    let app: express.Express;
    let listen: ReturnType<typeof spyOnListen>;

    function spyOnListen() {
        return vi.spyOn(express.application, "listen").mockReturnThis();
    }

    beforeAll(async () => {
        // Express reads NODE_ENV when it builds the app.
        vi.stubEnv("NODE_ENV", undefined);
        vi.stubEnv("GITHUB_APP_ID", "1");
        vi.stubEnv("GITHUB_APP_PRIVATE_KEY", PEM_CANARY);
        vi.stubEnv("GITHUB_WEBHOOK_SECRET", HOOK_CANARY);
        vi.stubEnv("PORT", "4321");
        listen = spyOnListen();

        await import("../src/server.js");

        app = listen.mock.contexts[0] as express.Express;
    });

    afterAll(() => {
        listen.mockRestore();
        vi.unstubAllEnvs();
    });

    it("listens once, on the configured port", () => {
        expect(listen).toHaveBeenCalledTimes(1);
        expect(listen.mock.calls[0]?.[0]).toBe(4321);
    });

    it("announces the port once it is listening", () => {
        const log = vi.spyOn(console, "log").mockImplementation(() => {});
        try {
            const onListening = listen.mock.calls[0]?.[1] as () => void;

            onListening();

            expect(log.mock.calls).toEqual([
                ["trelix GitHub App listening on port 4321"],
            ]);
        } finally {
            log.mockRestore();
        }
    });

    it("sweeps stale workspaces before it listens", () => {
        expect(mocks.sweepStaleWorkspaces).toHaveBeenCalledTimes(1);
        const sweptAt = mocks.sweepStaleWorkspaces.mock.invocationCallOrder[0];
        const listenedAt = listen.mock.invocationCallOrder[0];
        expect(sweptAt).toBeLessThan(listenedAt ?? 0);
    });

    it("serves /health", async () => {
        const res = await request(app).get("/health");

        expect(res.status).toBe(200);
        expect(res.text).toBe('{"status":"ok"}');
    });

    it("answers a malformed webhook body with the fixed 400, not Express's error page", async () => {
        const consoleError = vi
            .spyOn(console, "error")
            .mockImplementation(() => {});
        try {
            const res = await request(app)
                .post("/webhooks/github")
                .set("X-GitHub-Event", "pull_request")
                .set("Content-Type", "application/json")
                .send('{"canary-body": ');

            expect(res.status).toBe(400);
            expect(res.headers["content-type"]).toMatch(/^application\/json/);
            expect(res.text).toBe('{"error":"bad request"}');
            expect(consoleError).toHaveBeenCalledTimes(1);
            expect(String(consoleError.mock.calls[0]?.[0])).toContain(
                "[app] request failed",
            );
        } finally {
            consoleError.mockRestore();
        }
    });
});
