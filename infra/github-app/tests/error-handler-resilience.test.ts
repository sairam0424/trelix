import express from "express";
import type { Express } from "express";
import request from "supertest";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createErrorHandler } from "../src/error-handler.js";

// Values the handler must keep out of what it writes.
const HOOK_CANARY = "canary-hook-s";
const PEM_CANARY = "canary-pem-body";
const MESSAGE = "canary error message";
const SINK_DOWN = "canary log sink is down";
const CONSOLE_DOWN = "canary console is down";

const FIXED_500 = '{"error":"internal server error"}';
const FIXED_400 = '{"error":"bad request"}';
const LOG_FAILURE_LINE = "[app] request failed; the error could not be logged";

/** A bare app: the routes under test, then the handler, and nothing else. */
function buildApp(
    configure: (app: Express) => void,
    logError: (line: string) => void,
): Express {
    const app = express();
    configure(app);
    app.use(
        createErrorHandler({
            secrets: [HOOK_CANARY, PEM_CANARY],
            logError,
        }),
    );
    return app;
}

function failingSink(): never {
    throw new Error(SINK_DOWN);
}

function collector() {
    const lines: string[] = [];
    return { lines, logError: (l: string) => void lines.push(l) };
}

function spyOnConsoleError() {
    return vi.spyOn(console, "error").mockImplementation(() => {});
}

let consoleError: ReturnType<typeof spyOnConsoleError>;

// Unset NODE_ENV is the case where Express's own handler answers with the stack.
beforeEach(() => {
    vi.stubEnv("NODE_ENV", undefined);
    consoleError = spyOnConsoleError();
});

afterEach(() => {
    vi.unstubAllEnvs();
    consoleError.mockRestore();
});

describe("when the configured logger throws", () => {
    it("still answers an unexpected error with the fixed 500, not Express's page", async () => {
        const app = buildApp(
            (a) =>
                a.get("/x", () => {
                    throw new Error(MESSAGE);
                }),
            failingSink,
        );

        const res = await request(app).get("/x");

        expect(res.status).toBe(500);
        expect(res.headers["content-type"]).toMatch(/^application\/json/);
        expect(res.text).toBe(FIXED_500);
        expect(res.text).not.toContain(SINK_DOWN);
        expect(res.text).not.toContain(MESSAGE);
    });

    it("still answers a body error with its own fixed 4xx", async () => {
        const app = buildApp(
            (a) =>
                a.get("/x", () => {
                    throw Object.assign(new SyntaxError(MESSAGE), {
                        status: 400,
                        expose: true,
                    });
                }),
            failingSink,
        );

        const res = await request(app).get("/x");

        expect(res.status).toBe(400);
        expect(res.text).toBe(FIXED_400);
    });

    it("reports the failure once on the console with a fixed line that quotes no error", async () => {
        const app = buildApp(
            (a) =>
                a.get("/x", () => {
                    throw new Error(`${MESSAGE} ${HOOK_CANARY}`);
                }),
            failingSink,
        );

        await request(app).get("/x");

        expect(consoleError.mock.calls).toEqual([[LOG_FAILURE_LINE]]);
    });

    it("still sends the fixed response when the console fails as well", async () => {
        consoleError.mockImplementation(() => {
            throw new Error(CONSOLE_DOWN);
        });
        const app = buildApp(
            (a) =>
                a.get("/x", () => {
                    throw new Error(MESSAGE);
                }),
            failingSink,
        );

        const res = await request(app).get("/x");

        expect(res.status).toBe(500);
        expect(res.text).toBe(FIXED_500);
        expect(consoleError).toHaveBeenCalledTimes(1);
    });

    it("does not touch the console when the logger works", async () => {
        const { lines, logError } = collector();
        const app = buildApp(
            (a) =>
                a.get("/x", () => {
                    throw new Error(MESSAGE);
                }),
            logError,
        );

        await request(app).get("/x");

        expect(lines).toHaveLength(1);
        expect(consoleError).not.toHaveBeenCalled();
    });
});

describe("an error object that throws when it is inspected", () => {
    it("is answered with the fixed 500 when reading its status throws", async () => {
        const { lines, logError } = collector();
        const hostile = Object.defineProperty({}, "status", {
            get() {
                throw new Error(MESSAGE);
            },
        });
        const app = buildApp(
            (a) =>
                a.get("/x", () => {
                    throw hostile;
                }),
            logError,
        );

        const res = await request(app).get("/x");

        expect(res.status).toBe(500);
        expect(res.text).toBe(FIXED_500);
        expect(lines).toHaveLength(1);
        expect(lines[0]).toContain('"status":500');
    });

    it.each(["stack", "name"])(
        "is answered with the fixed 500 when reading its %s throws",
        async (property) => {
            const { lines, logError } = collector();
            const hostile = Object.defineProperty(new Error("x"), property, {
                get() {
                    throw new Error(MESSAGE);
                },
            });
            const app = buildApp(
                (a) =>
                    a.get("/x", () => {
                        throw hostile;
                    }),
                logError,
            );

            const res = await request(app).get("/x");

            expect(res.status).toBe(500);
            expect(res.text).toBe(FIXED_500);
            expect(res.text).not.toContain(MESSAGE);
            expect(lines).toHaveLength(0);
            expect(consoleError.mock.calls).toEqual([[LOG_FAILURE_LINE]]);
        },
    );
});

describe("bounds on what is logged", () => {
    it("cuts a long request path", async () => {
        const { lines, logError } = collector();
        const app = buildApp(
            (a) =>
                a.use(() => {
                    throw new Error(MESSAGE);
                }),
            logError,
        );

        await request(app).get(`/x${"A".repeat(8_000)}`);

        expect(lines).toHaveLength(1);
        const record = JSON.parse(
            (lines[0] ?? "").slice("[app] request failed ".length),
        ) as { path: string };
        expect(record.path).toBe(`/x${"A".repeat(198)}…[truncated]`);
    });

    it("keeps a short request path whole", async () => {
        const { lines, logError } = collector();
        const app = buildApp(
            (a) =>
                a.use(() => {
                    throw new Error(MESSAGE);
                }),
            logError,
        );

        await request(app).get(`/x${"A".repeat(150)}`);

        expect(lines[0]).toContain(`"path":"/x${"A".repeat(150)}"`);
    });

    // The secret starts a little before the 4,000-character cut and ends after
    // it. Cutting first would leave its first characters in the line, where no
    // whole-secret match could find them.
    const straddlers: ReadonlyArray<readonly [string, () => unknown]> = [
        [
            "an Error",
            () =>
                new Error(
                    `${"x".repeat(3_980)}${PEM_CANARY}${"y".repeat(200)}`,
                ),
        ],
        [
            "a string",
            () => `${"x".repeat(3_990)}${PEM_CANARY}${"y".repeat(200)}`,
        ],
    ];

    it.each(straddlers)(
        "removes a secret that straddles the cut in %s",
        async (_name, make) => {
            const { lines, logError } = collector();
            const app = buildApp(
                (a) =>
                    a.get("/x", () => {
                        throw make();
                    }),
                logError,
            );

            await request(app).get("/x");

            expect(lines).toHaveLength(1);
            const line = lines[0] ?? "";
            expect(line).not.toContain(PEM_CANARY.slice(0, 8));
            expect(line).toContain("x[redacted]");
            expect(line).toContain("…[truncated]");
        },
    );
});
