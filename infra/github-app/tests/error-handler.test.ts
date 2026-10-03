import express from "express";
import type { Express, Request, Response } from "express";
import request from "supertest";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createErrorHandler, redactSecrets } from "../src/error-handler.js";

// Values the handler must keep out of its log line.
const HOOK_CANARY = "canary-hook-s";
const PEM_CANARY = "canary-pem-body";
const MESSAGE = "canary error message";

const FIXED_500 = '{"error":"internal server error"}';

interface Harness {
    readonly app: Express;
    readonly lines: string[];
}

/** A bare app: the routes under test, then the handler, and nothing else. */
function buildApp(
    configure: (app: Express) => void,
    hidden: readonly string[] = [HOOK_CANARY, PEM_CANARY],
): Harness {
    const lines: string[] = [];
    const app = express();
    configure(app);
    app.use(
        createErrorHandler({
            secrets: hidden,
            logError: (l) => lines.push(l),
        }),
    );
    return { app, lines };
}

// Express picks its default error handler from NODE_ENV when the app is
// built: unset means "development", which answers with the stack.
beforeEach(() => {
    vi.stubEnv("NODE_ENV", undefined);
});

afterEach(() => {
    vi.unstubAllEnvs();
});

describe("Express default handler (control)", () => {
    it("answers with the message and the stack when NODE_ENV is unset", async () => {
        const app = express();
        app.get("/x", () => {
            throw new Error(MESSAGE);
        });

        const res = await request(app).get("/x");

        expect(res.status).toBe(500);
        expect(res.text).toContain(MESSAGE);
        expect(res.text).toContain("error-handler.test.ts");
    });

    it("answers without them when NODE_ENV is production", async () => {
        vi.stubEnv("NODE_ENV", "production");
        const app = express();
        app.get("/x", () => {
            throw new Error(MESSAGE);
        });

        const res = await request(app).get("/x");

        expect(res.status).toBe(500);
        expect(res.text).not.toContain(MESSAGE);
    });
});

describe("error handler response", () => {
    const throwers: ReadonlyArray<readonly [string, (app: Express) => void]> = [
        [
            "a sync throw",
            (app) =>
                app.get("/x", () => {
                    throw new Error(MESSAGE);
                }),
        ],
        [
            "an async rejection",
            (app) =>
                app.get("/x", async () => {
                    throw new Error(MESSAGE);
                }),
        ],
        [
            "next(err)",
            (app) =>
                app.get("/x", (_req, _res, next) => {
                    next(new Error(MESSAGE));
                }),
        ],
        [
            "a thrown string",
            (app) =>
                app.get("/x", () => {
                    throw MESSAGE;
                }),
        ],
        [
            "a thrown plain object",
            (app) =>
                app.get("/x", () => {
                    throw { message: MESSAGE, stack: MESSAGE };
                }),
        ],
        [
            "a thrown null-prototype object",
            (app) =>
                app.get("/x", () => {
                    throw Object.create(null);
                }),
        ],
    ];

    it.each(throwers)(
        "answers %s with the fixed 500 body and logs once",
        async (_name, configure) => {
            const { app, lines } = buildApp(configure);

            const res = await request(app).get("/x");

            expect(res.status).toBe(500);
            expect(res.headers["content-type"]).toMatch(/^application\/json/);
            expect(res.text).toBe(FIXED_500);
            expect(res.text).not.toContain(MESSAGE);
            expect(lines).toHaveLength(1);
        },
    );

    const statusCases: ReadonlyArray<
        readonly [string, Record<string, unknown>, number, string]
    > = [
        [
            "400 that marks itself exposable",
            { status: 400, expose: true },
            400,
            '{"error":"bad request"}',
        ],
        [
            "413 that marks itself exposable",
            { status: 413, expose: true },
            413,
            '{"error":"payload too large"}',
        ],
        [
            "415 that marks itself exposable",
            { status: 415, expose: true },
            415,
            '{"error":"unsupported media type"}',
        ],
        [
            "400 that does not say it is safe to show",
            { status: 400 },
            500,
            FIXED_500,
        ],
        [
            "400 with expose false",
            { status: 400, expose: false },
            500,
            FIXED_500,
        ],
        [
            "401 even if exposable",
            { status: 401, expose: true },
            500,
            FIXED_500,
        ],
        [
            "403 even if exposable",
            { status: 403, expose: true },
            500,
            FIXED_500,
        ],
        [
            "404 even if exposable",
            { status: 404, expose: true },
            500,
            FIXED_500,
        ],
        [
            "503 even if exposable",
            { status: 503, expose: true },
            500,
            FIXED_500,
        ],
        [
            "a status given as text",
            { status: "400", expose: true },
            500,
            FIXED_500,
        ],
    ];

    it.each(statusCases)(
        "a %s is answered with the right status and a fixed body",
        async (_name, fields, expectedStatus, expectedBody) => {
            const { app } = buildApp((a) =>
                a.get("/x", () => {
                    throw Object.assign(new Error(MESSAGE), fields);
                }),
            );

            const res = await request(app).get("/x");

            expect(res.status).toBe(expectedStatus);
            expect(res.text).toBe(expectedBody);
            expect(res.text).not.toContain(MESSAGE);
        },
    );
});

describe("error handler log line", () => {
    it("logs the detail of a 5xx once, without the request's body, headers or query", async () => {
        const { app, lines } = buildApp((a) => {
            a.use(express.json());
            a.post("/x", () => {
                throw new Error("detail-marker boom");
            });
        });

        const res = await request(app)
            .post("/x?q=canary-query")
            .set("Content-Type", "application/json")
            .set("X-Hub-Signature-256", "canary-signature")
            .send('{"payload":"canary-body"}');

        expect(res.status).toBe(500);
        expect(lines).toHaveLength(1);
        const line = lines[0] ?? "";
        expect(line).toContain("detail-marker boom");
        expect(line).toContain('"method":"POST"');
        expect(line).toContain('"path":"/x"');
        expect(line).toContain('"status":500');
        expect(line).not.toContain("alreadySent");
        expect(line).not.toContain("canary-body");
        expect(line).not.toContain("canary-signature");
        expect(line).not.toContain("canary-query");
    });

    it("logs a 4xx by name and status, without its message", async () => {
        const { app, lines } = buildApp((a) =>
            a.get("/x", () => {
                throw Object.assign(new SyntaxError("canary-quoted-body"), {
                    status: 400,
                    expose: true,
                });
            }),
        );

        const res = await request(app).get("/x");

        expect(res.status).toBe(400);
        expect(lines).toHaveLength(1);
        const line = lines[0] ?? "";
        expect(line).toContain('"status":400');
        expect(line).toContain('"error":"SyntaxError"');
        expect(line).not.toContain("canary-quoted-body");
        expect(line).not.toContain("detail");
    });

    it("removes the configured secrets from the message and the stack", async () => {
        const { app, lines } = buildApp((a) =>
            a.get("/x", () => {
                throw new Error(
                    `failed with ${HOOK_CANARY} and ${PEM_CANARY}, again ${HOOK_CANARY}`,
                );
            }),
        );

        await request(app).get("/x");

        expect(lines).toHaveLength(1);
        const line = lines[0] ?? "";
        expect(line).not.toContain(HOOK_CANARY);
        expect(line).not.toContain(PEM_CANARY);
        expect(line).toContain("failed with [redacted] and [redacted]");
    });

    it("removes a secret from a thrown string too", async () => {
        const { app, lines } = buildApp((a) =>
            a.get("/x", () => {
                throw `leaked ${PEM_CANARY}`;
            }),
        );

        await request(app).get("/x");

        expect(lines[0]).not.toContain(PEM_CANARY);
        expect(lines[0]).toContain("leaked [redacted]");
    });

    it("cuts a very long message to a bounded line", async () => {
        const { app, lines } = buildApp((a) =>
            a.get("/x", () => {
                throw new Error("x".repeat(50_000));
            }),
        );

        await request(app).get("/x");

        expect(lines).toHaveLength(1);
        expect((lines[0] ?? "").length).toBeLessThan(4200);
        expect(lines[0]).toContain("…[truncated]");
    });
});

describe("redactSecrets", () => {
    it("replaces every occurrence", () => {
        expect(redactSecrets("k-k-k", ["k"])).toBe(
            "[redacted]-[redacted]-[redacted]",
        );
    });

    it("replaces the longest secret first, so a secret containing another is not left half visible", () => {
        expect(redactSecrets("xabx", ["ab", "xabx"])).toBe("[redacted]");
    });

    it("ignores an empty secret", () => {
        expect(redactSecrets("abc", [""])).toBe("abc");
    });

    it("leaves the list it was given alone", () => {
        const given = ["ab", "xabx"];

        redactSecrets("xabx", given);

        expect(given).toEqual(["ab", "xabx"]);
    });

    it("returns text without a secret unchanged", () => {
        expect(redactSecrets("nothing here", ["zzz"])).toBe("nothing here");
    });
});

describe("hostile input stays linear", () => {
    const LIMIT_MS = 2000;

    it("redacts 50,000 one-character matches in under 2 s", () => {
        const text = "x".repeat(50_000);
        const started = performance.now();

        const result = redactSecrets(text, ["x"]);

        expect(performance.now() - started).toBeLessThan(LIMIT_MS);
        expect(result.length).toBe(500_000);
    }, 60_000);

    // Half the text long and one character wrong at the end: a scan that
    // compares at every position does 25,000 x 25,000 comparisons (3.4 s when
    // measured), where split/join takes under 1 ms.
    it("scans 50,000 characters for a long near-miss secret in under 2 s", () => {
        const text = "a".repeat(50_000);
        const nearMiss = `${"a".repeat(24_999)}b`;
        const started = performance.now();

        const result = redactSecrets(text, [nearMiss]);

        expect(performance.now() - started).toBeLessThan(LIMIT_MS);
        expect(result).toBe(text);
    }, 60_000);

    it("handles an error with a 50,000-character message in under 2 s end to end", async () => {
        const { app, lines } = buildApp(
            (a) =>
                a.get("/x", () => {
                    throw new Error("x".repeat(50_000));
                }),
            ["x"],
        );
        const started = performance.now();

        const res = await request(app).get("/x");

        expect(performance.now() - started).toBeLessThan(LIMIT_MS);
        expect(res.text).toBe(FIXED_500);
        expect(lines).toHaveLength(1);
    }, 60_000);
});

describe("after the response has started", () => {
    function fakeResponse(headersSent: boolean, writableEnded: boolean) {
        return {
            headersSent,
            writableEnded,
            statusCode: 202,
            status: vi.fn().mockReturnThis(),
            json: vi.fn().mockReturnThis(),
            destroy: vi.fn(),
        };
    }

    function handle(res: ReturnType<typeof fakeResponse>) {
        const lines: string[] = [];
        const handler = createErrorHandler({
            secrets: [],
            logError: (l) => lines.push(l),
        });
        handler(
            new Error(MESSAGE),
            { method: "GET", path: "/x" } as Request,
            res as unknown as Response,
            vi.fn(),
        );
        return lines;
    }

    it("closes a half-written response and sends nothing more", () => {
        const res = fakeResponse(true, false);

        const lines = handle(res);

        expect(res.destroy).toHaveBeenCalledTimes(1);
        expect(res.status).not.toHaveBeenCalled();
        expect(res.json).not.toHaveBeenCalled();
        expect(lines).toHaveLength(1);
        expect(lines[0]).toContain('"status":202');
        expect(lines[0]).toContain('"alreadySent":true');
    });

    it("leaves a finished response alone", () => {
        const res = fakeResponse(true, true);

        const lines = handle(res);

        expect(res.destroy).not.toHaveBeenCalled();
        expect(res.status).not.toHaveBeenCalled();
        expect(lines).toHaveLength(1);
    });

    it("answers normally while nothing has been sent", () => {
        const res = fakeResponse(false, false);

        handle(res);

        expect(res.destroy).not.toHaveBeenCalled();
        expect(res.status).toHaveBeenCalledWith(500);
        expect(res.json).toHaveBeenCalledWith({
            error: "internal server error",
        });
    });

    it("keeps the response the client already got when a route fails after sending it", async () => {
        const { app, lines } = buildApp((a) =>
            a.get("/x", (_req, res, next) => {
                res.json({ ok: true });
                next(new Error(MESSAGE));
            }),
        );

        const res = await request(app).get("/x");

        expect(res.status).toBe(200);
        expect(res.text).toBe('{"ok":true}');
        expect(lines).toHaveLength(1);
        // The line says what the client got (200), not the 500 the error
        // would have been answered with, and still carries the detail.
        expect(lines[0]).toContain('"status":200');
        expect(lines[0]).toContain('"alreadySent":true');
        expect(lines[0]).toContain(MESSAGE);
    });

    it("drops the connection on a half-written response instead of finishing it with the error", async () => {
        const { app, lines } = buildApp((a) =>
            a.get("/x", (_req, res, next) => {
                res.status(200);
                res.write("partial");
                next(new Error(MESSAGE));
            }),
        );

        await expect(request(app).get("/x")).rejects.toBeDefined();
        expect(lines).toHaveLength(1);
        expect(lines[0]).toContain('"status":200');
        expect(lines[0]).toContain('"alreadySent":true');
    });
});
