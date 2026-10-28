import { createHmac } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import request from "supertest";
import { sign } from "@octokit/webhooks-methods";
import { describe, expect, it, vi } from "vitest";
import {
    buildHarness,
    CONFIG,
    pullRequestPayload,
    send,
    tick,
} from "./support/webhook-harness.js";

const WRONG_KEY = "test-k-wrong";
const JSON_TYPE = "application/json";

/** A body of about 3 MB that is cheap to sign and expensive to parse: 1,000,000 empty objects. */
const EXPENSIVE_BODY = `[${"{},".repeat(999_999)}{}]`;

function post(
    app: ReturnType<typeof buildHarness>["app"],
    body: string,
    headers: Record<string, string>,
) {
    return request(app)
        .post("/webhooks/github")
        .set("X-GitHub-Event", "pull_request")
        .set(headers)
        .send(body);
}

async function signedHeaders(body: string, contentType = JSON_TYPE) {
    return {
        "Content-Type": contentType,
        "X-Hub-Signature-256": await sign(CONFIG.webhookSecret, body),
    };
}

/**
 * A JSON.parse argument this long (a string or a Buffer) is the 3 MB body or a copy of it,
 * whatever was cut off it or however it was converted; nothing else in a request is near it.
 */
const LARGE_PARSE_LENGTH = 1_000_000;

/** The calls to JSON.parse that were handed something at least LARGE_PARSE_LENGTH long. */
function largeParses(parse: { mock: { calls: unknown[][] } }): number {
    return parse.mock.calls.filter(([input]) => {
        const length = (input as { length?: unknown } | null | undefined)
            ?.length;
        return typeof length === "number" && length >= LARGE_PARSE_LENGTH;
    }).length;
}

describe("a delivery is verified before its body is parsed", () => {
    const malformed = '{"canary-body": ';

    it("answers 401 'missing signature or body' for malformed JSON with no signature", async () => {
        const h = buildHarness();

        const res = await post(h.app, malformed, { "Content-Type": JSON_TYPE });

        expect(res.status).toBe(401);
        expect(res.body).toEqual({ error: "missing signature or body" });
        expect(h.runReview).not.toHaveBeenCalled();
    });

    it("answers 401 'signature verification failed' for malformed JSON with a wrong signature", async () => {
        const h = buildHarness();

        const res = await post(h.app, malformed, {
            "Content-Type": JSON_TYPE,
            "X-Hub-Signature-256": await sign(WRONG_KEY, malformed),
        });

        expect(res.status).toBe(401);
        expect(res.body).toEqual({ error: "signature verification failed" });
        expect(h.runReview).not.toHaveBeenCalled();
    });

    // verify() compares lengths first; a header too short to be a signature must still be a 401.
    it.each([["x"], ["sha256="]])(
        "answers 401 'signature verification failed' for the malformed signature header %j",
        async (header) => {
            const h = buildHarness();
            const body = JSON.stringify(pullRequestPayload());

            const res = await post(h.app, body, {
                "Content-Type": JSON_TYPE,
                "X-Hub-Signature-256": header,
            });

            expect(res.status).toBe(401);
            expect(res.body).toEqual({
                error: "signature verification failed",
            });
            expect(h.runReview).not.toHaveBeenCalled();
        },
    );

    it("answers 400 for malformed JSON that is correctly signed", async () => {
        const h = buildHarness();

        const res = await post(
            h.app,
            malformed,
            await signedHeaders(malformed),
        );

        expect(res.status).toBe(400);
        expect(h.runReview).not.toHaveBeenCalled();
    });

    it.each([
        ["no signature header", () => ({}) as Record<string, string>],
        [
            "a signature made with another key",
            async () => ({
                "X-Hub-Signature-256": await sign(WRONG_KEY, EXPENSIVE_BODY),
            }),
        ],
    ])("never parses a body that arrives with %s", async (_name, extra) => {
        const h = buildHarness();
        const parse = vi.spyOn(JSON, "parse");
        try {
            const res = await post(h.app, EXPENSIVE_BODY, {
                "Content-Type": JSON_TYPE,
                ...(await extra()),
            });

            expect(res.status).toBe(401);
            expect(largeParses(parse)).toBe(0);
        } finally {
            parse.mockRestore();
        }
    });

    it("does parse the same body once it is correctly signed (the spy sees a parse)", async () => {
        const h = buildHarness();
        const parse = vi.spyOn(JSON, "parse");
        try {
            const res = await post(
                h.app,
                EXPENSIVE_BODY,
                await signedHeaders(EXPENSIVE_BODY),
            );

            expect(res.status).toBe(202);
            expect(largeParses(parse)).toBe(1);
        } finally {
            parse.mockRestore();
        }
    });
});

describe("the size cap on a body read before the signature is checked", () => {
    const CAP_BYTES = 5_242_880; // 5 MiB, written out so the test does not read the module's own number

    /** A pull_request delivery of exactly `bytes` bytes (ASCII only, so characters are bytes). */
    function deliveryOfExactly(bytes: number): string {
        const empty = JSON.stringify({ ...pullRequestPayload(), padding: "" });
        const body = JSON.stringify({
            ...pullRequestPayload(),
            padding: "x".repeat(bytes - empty.length),
        });
        expect(Buffer.byteLength(body)).toBe(bytes);
        return body;
    }

    it("processes a correctly signed delivery of exactly the cap", async () => {
        const h = buildHarness();
        const body = deliveryOfExactly(CAP_BYTES);

        const res = await post(h.app, body, await signedHeaders(body));
        await tick();

        expect(res.status).toBe(202);
        expect(res.body).toEqual({ accepted: true });
        expect(h.calls).toHaveLength(1);
    }, 30_000);

    it.each([
        ["unsigned", false],
        ["correctly signed", true],
    ])(
        "answers 413 for a %s delivery one byte over the cap",
        async (_name, isSigned) => {
            const h = buildHarness();
            const body = deliveryOfExactly(CAP_BYTES + 1);
            const headers = isSigned
                ? await signedHeaders(body)
                : { "Content-Type": JSON_TYPE };

            const res = await post(h.app, body, headers);

            expect(res.status).toBe(413);
            expect(h.runReview).not.toHaveBeenCalled();
        },
        30_000,
    );
});

describe("what the route accepts, unchanged by reading the bytes first", () => {
    // A decision, not an accident: the charset parameter is not checked (the signature is over
    // the bytes, read as UTF-8). express.json() answered a signed latin1 delivery 415 and a
    // utf-16 one 400; GitHub sends neither, and this route no longer does either.
    it.each([
        ["application/json; charset=utf-8"],
        ["application/json; charset=latin1"],
        ["application/json; charset=utf-16"],
    ])(
        "accepts a correctly signed delivery whose content type is %s",
        async (contentType) => {
            const h = buildHarness();
            const body = JSON.stringify(pullRequestPayload());

            const res = await post(
                h.app,
                body,
                await signedHeaders(body, contentType),
            );
            await tick();

            expect(res.status).toBe(202);
            expect(res.body).toEqual({ accepted: true });
            expect(h.calls).toHaveLength(1);
        },
    );

    // express.json() dropped one leading byte order mark before parsing. The signature covers
    // the mark, so it can only be dropped after the check, and only one of them.
    it.each([
        ["one byte order mark", 202, "\uFEFF", 1],
        ["two byte order marks", 400, "\uFEFF\uFEFF", 0],
    ])(
        "answers a correctly signed delivery that starts with %s with %i",
        async (_name, status, prefix, reviews) => {
            const h = buildHarness();
            const body = `${prefix}${JSON.stringify(pullRequestPayload())}`;

            const res = await post(h.app, body, await signedHeaders(body));
            await tick();

            expect(res.status).toBe(status);
            expect(h.calls).toHaveLength(reviews);
        },
    );

    // The signature is over the bytes GitHub sent, which are UTF-8; reading them as any other
    // encoding before the check would turn a real delivery with a non-ASCII title into a 401.
    it("accepts a correctly signed delivery whose payload has non-ASCII text", async () => {
        const h = buildHarness();
        const body = JSON.stringify({
            ...pullRequestPayload(),
            body: "na\u00efve caf\u00e9 \u2713 \u65e5\u672c\u8a9e \u{1F600}",
        });
        expect(Buffer.byteLength(body)).toBeGreaterThan(body.length);

        const res = await post(h.app, body, await signedHeaders(body));
        await tick();

        expect(res.status).toBe(202);
        expect(res.body).toEqual({ accepted: true });
        expect(h.calls).toHaveLength(1);
    });

    it("still answers a signed delivery of another event type with the ignored body", async () => {
        const h = buildHarness();

        const res = await send(h.app, { zen: "canary" }, { event: "ping" });

        expect(res.status).toBe(202);
        expect(res.body).toEqual({
            ignored: true,
            reason: "unhandled event: ping",
        });
    });

    it("does not read a body whose content type is not JSON: 401 even when signed", async () => {
        const h = buildHarness();
        const body = JSON.stringify(pullRequestPayload());

        const res = await post(
            h.app,
            body,
            await signedHeaders(body, "text/plain"),
        );

        expect(res.status).toBe(401);
        expect(res.body).toEqual({ error: "missing signature or body" });
        expect(h.runReview).not.toHaveBeenCalled();
    });

    it.each([
        ["without a signature", {}],
        [
            "with the signature of an empty body",
            {
                "X-Hub-Signature-256": `sha256=${createHmac("sha256", CONFIG.webhookSecret).update("").digest("hex")}`,
            },
        ],
    ])(
        "answers 401 'missing signature or body' for an empty body %s",
        async (_name, extra) => {
            const h = buildHarness();

            const res = await post(h.app, "", {
                "Content-Type": JSON_TYPE,
                ...extra,
            });

            expect(res.status).toBe(401);
            expect(res.body).toEqual({ error: "missing signature or body" });
        },
    );

    // A parser in strict mode only takes an object or an array; `null` would otherwise
    // reach the route as a body with no `action` to read.
    it.each([
        ["null", "null"],
        ["a string", '"canary"'],
        ["a number", "123"],
        // express.json() read this as an empty object and the route answered 202.
        ["only a byte order mark", "\uFEFF"],
    ])(
        "answers 400 for a correctly signed body that is %s",
        async (_name, body) => {
            const h = buildHarness();

            const res = await post(h.app, body, await signedHeaders(body));

            expect(res.status).toBe(400);
            expect(h.runReview).not.toHaveBeenCalled();
        },
    );
});

// The README told operators the size limit was the only defense because the signature was
// checked after parsing. It has to say what the route does now.
describe("the README's account of the size cap", () => {
    const readme = readFileSync(
        fileURLToPath(new URL("../README.md", import.meta.url)),
        "utf8",
    );

    it("states the 5 MiB cap and no longer says the signature is checked after parsing", () => {
        expect(readme).toContain("reads at most 5 MiB of a");
        expect(readme).not.toContain("only defense");
        expect(readme).not.toContain("*after* body parsing");
    });

    // "never sees the payload" read as if the handler got no payload at all.
    it("says the handler only ever sees a verified payload", () => {
        expect(readme).toContain("only ever sees a verified payload");
        expect(readme).not.toContain("never sees the payload");
    });
});
