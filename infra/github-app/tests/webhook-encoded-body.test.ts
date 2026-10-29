import { readFileSync } from "node:fs";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import request from "supertest";
import { sign } from "@octokit/webhooks-methods";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createApp } from "../src/app.js";
import type { RunReviewFn } from "../src/review-intake.js";
import { CONFIG, pullRequestPayload, tick } from "./support/webhook-harness.js";

const CAP_BYTES = 5_242_880; // 5 MiB, written out so the test does not read the module's own number

type DecompressionFactory =
    | "createGunzip"
    | "createInflate"
    | "createBrotliDecompress"
    | "createZstdDecompress";

/** Every way the body parser could inflate a request, by the name a client sends. */
const ENCODINGS: ReadonlyArray<
    readonly [string, (bytes: Buffer) => Buffer, DecompressionFactory]
> = [
    ["gzip", (bytes) => zlib.gzipSync(bytes), "createGunzip"],
    ["deflate", (bytes) => zlib.deflateSync(bytes), "createInflate"],
    [
        "br",
        (bytes) =>
            zlib.brotliCompressSync(bytes, {
                params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 1 },
            }),
        "createBrotliDecompress",
    ],
    ["zstd", (bytes) => zlib.zstdCompressSync(bytes), "createZstdDecompress"],
];

function buildApp() {
    const lines: string[] = [];
    const runReview = vi.fn<RunReviewFn>().mockResolvedValue([]);
    const app = createApp(CONFIG, {
        runReview,
        logError: (line) => lines.push(line),
    });
    return { app, lines, runReview };
}

/** The decompression streams of node:zlib, which only an inflating body parser creates. */
function spyOnDecompression() {
    return ENCODINGS.map(([, , factory]) => vi.spyOn(zlib, factory));
}

function post(
    app: ReturnType<typeof createApp>,
    body: Buffer | string,
    headers: Record<string, string>,
) {
    return request(app)
        .post("/webhooks/github")
        .set("X-GitHub-Event", "pull_request")
        .set("Content-Type", "application/json")
        .set(headers)
        // superagent would otherwise JSON-encode a Buffer sent as application/json.
        .serialize((data) => data as string)
        .send(body);
}

afterEach(() => {
    vi.restoreAllMocks();
});

describe("a body with a Content-Encoding is refused before it is inflated", () => {
    it.each(ENCODINGS)(
        "answers 415 for a correctly signed %s body that inflates to a small valid delivery",
        async (encoding, compress) => {
            const { app, lines, runReview } = buildApp();
            const decompressors = spyOnDecompression();
            const text = JSON.stringify(pullRequestPayload());
            const signature = await sign(CONFIG.webhookSecret, text);

            const res = await post(app, compress(Buffer.from(text)), {
                "Content-Encoding": encoding,
                "X-Hub-Signature-256": signature,
            });
            await tick();

            expect(res.status).toBe(415);
            expect(res.text).toBe('{"error":"unsupported media type"}');
            expect(runReview).not.toHaveBeenCalled();
            expect(decompressors.map((spy) => spy.mock.calls.length)).toEqual([
                0, 0, 0, 0,
            ]);
            expect(lines).toHaveLength(1);
        },
    );

    it.each(ENCODINGS)(
        "answers 415, not 413, for an unsigned %s body that would inflate past the cap",
        async (encoding, compress) => {
            const { app, runReview } = buildApp();
            const decompressors = spyOnDecompression();
            const body = compress(Buffer.alloc(CAP_BYTES + 1, "a"));
            expect(body.length).toBeLessThan(100_000);

            const res = await post(app, body, {
                "Content-Encoding": encoding,
                "X-Hub-Signature-256": `sha256=${"0".repeat(64)}`,
            });

            expect(res.status).toBe(415);
            expect(res.text).toBe('{"error":"unsupported media type"}');
            expect(runReview).not.toHaveBeenCalled();
            expect(decompressors.map((spy) => spy.mock.calls.length)).toEqual([
                0, 0, 0, 0,
            ]);
        },
    );

    it("does not echo an encoding the sender made up", async () => {
        const { app, lines } = buildApp();

        const res = await post(app, "{}", {
            "Content-Encoding": "canary-encoding",
        });

        expect(res.status).toBe(415);
        expect(res.text).toBe('{"error":"unsupported media type"}');
        expect(lines.join("\n")).not.toContain("canary-encoding");
    });
});

describe("what is still accepted", () => {
    // The headers GitHub documents for a delivery (docs.github.com, "Webhook events and
    // payloads", "Delivery headers"): no Content-Encoding among them.
    it("processes a correctly signed delivery that has only the headers GitHub documents", async () => {
        const { app, runReview } = buildApp();
        const text = JSON.stringify(pullRequestPayload());

        const res = await post(app, text, {
            "User-Agent": "GitHub-Hookshot/044aadd",
            "X-GitHub-Delivery": "72d3162e-cc78-11e3-81ab-4c9367dc0958",
            "X-GitHub-Hook-ID": "292430182",
            "X-GitHub-Hook-Installation-Target-ID": "79929171",
            "X-GitHub-Hook-Installation-Target-Type": "integration",
            "X-Hub-Signature-256": await sign(CONFIG.webhookSecret, text),
        });
        await tick();

        expect(res.status).toBe(202);
        expect(res.body).toEqual({ accepted: true });
        expect(runReview).toHaveBeenCalledTimes(1);
    });

    // Only an encoding that changes the bytes is refused; the identity encoding is no encoding.
    it("processes a correctly signed delivery whose Content-Encoding is identity", async () => {
        const { app, runReview } = buildApp();
        const text = JSON.stringify(pullRequestPayload());

        const res = await post(app, text, {
            "Content-Encoding": "identity",
            "X-Hub-Signature-256": await sign(CONFIG.webhookSecret, text),
        });
        await tick();

        expect(res.status).toBe(202);
        expect(runReview).toHaveBeenCalledTimes(1);
    });
});

describe("the README's account of encoded bodies and the server limits", () => {
    // One line of text: a phrase may wrap onto the next line of the README.
    const readme = readFileSync(
        fileURLToPath(new URL("../README.md", import.meta.url)),
        "utf8",
    ).replace(/\s+/g, " ");

    it("says an encoded body is refused with 415 and why that is safe", () => {
        expect(readme).toContain(
            "declares a `Content-Encoding` other than `identity`",
        );
        expect(readme).toContain("`415`");
        expect(readme).toContain("no genuine delivery is refused");
    });

    it("states the timeouts and the cap on open connections", () => {
        expect(readme).toContain("more than 5 seconds");
        expect(readme).toContain("more than 15 seconds");
        expect(readme).toContain("once 40 connections are open");
        expect(readme).toContain("40 x 5 MiB = 200 MiB");
    });

    // The cap is global, not per client; the README must not read as a per-client limit.
    it("says the cap is one pool that a slow sender can fill, and what that trades", () => {
        expect(readme).toContain(
            "one pool shared by every sender, not a limit per client",
        );
        expect(readme).toContain(
            "refuse every other connection, GitHub's deliveries and `/health` included",
        );
        expect(readme).toContain("by opening a new connection each time");
        expect(readme).toContain("This trades a possible out-of-memory kill");
        expect(readme).toContain("for a possible refusal of service");
    });
});
