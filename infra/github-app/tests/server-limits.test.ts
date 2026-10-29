import http from "node:http";
import net from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { createApp } from "../src/app.js";
import { applyServerLimits, type ServerLimits } from "../src/server-limits.js";
import { CONFIG } from "./support/webhook-harness.js";

/** Node keeps the option as a property, which @types/node does not declare. */
function checkingInterval(server: http.Server): unknown {
    return (server as unknown as Record<string, unknown>)
        .connectionsCheckingInterval;
}

describe("applyServerLimits", () => {
    it("sets the limits the App runs with", () => {
        const server = http.createServer();

        applyServerLimits(server);

        expect(server.headersTimeout).toBe(5_000);
        expect(server.requestTimeout).toBe(15_000);
        expect(checkingInterval(server)).toBe(1_000);
        expect(server.maxConnections).toBe(40);
    });

    // Node refuses this pair in http.createServer() but not as properties, so nothing else would.
    it("keeps the request timeout above the headers timeout", () => {
        const server = http.createServer();

        applyServerLimits(server);

        expect(server.requestTimeout).toBeGreaterThan(server.headersTimeout);
    });

    // 40 connections each holding a body of the 5 MiB cap must stay under 40 percent of 512 MiB.
    it("keeps the worst case of full-size bodies under 40 percent of the container", () => {
        const server = http.createServer();

        applyServerLimits(server);

        const worstCaseBytes = server.maxConnections * 5 * 1024 * 1024;
        expect(worstCaseBytes).toBeLessThanOrEqual(0.4 * 512 * 1024 * 1024);
    });

    it("sets the limits it is given", () => {
        const server = http.createServer();

        applyServerLimits(server, {
            headersTimeoutMs: 11,
            requestTimeoutMs: 22,
            connectionsCheckingIntervalMs: 33,
            maxConnections: 44,
        });

        expect(server.headersTimeout).toBe(11);
        expect(server.requestTimeout).toBe(22);
        expect(checkingInterval(server)).toBe(33);
        expect(server.maxConnections).toBe(44);
    });
});

const open: http.Server[] = [];
const held: net.Socket[] = [];

afterEach(async () => {
    for (const socket of held.splice(0)) socket.destroy();
    for (const server of open.splice(0)) {
        server.closeAllConnections();
        await new Promise((resolve) => server.close(resolve));
    }
});

/** A listening App server; the short timeouts keep the tests fast. */
function listen(overrides: Partial<ServerLimits> = {}): Promise<http.Server> {
    // The error line of an upload that is cut off is not what these tests look at.
    const app = createApp(CONFIG, { logError: () => {} });
    const server = app.listen(0, "127.0.0.1");
    applyServerLimits(server, {
        headersTimeoutMs: 200,
        requestTimeoutMs: 400,
        connectionsCheckingIntervalMs: 25,
        maxConnections: 40,
        ...overrides,
    });
    open.push(server);
    return new Promise((resolve) =>
        server.on("listening", () => resolve(server)),
    );
}

// The checking interval is not in Node's documented properties: this is the test that fails if a
// Node release stops reading it, because the default check runs every 30 s.
describe("a request that does not arrive in time", () => {
    function stalledUpload(port: number) {
        return new Promise<{ elapsedMs: number; response: string }>(
            (resolve) => {
                const started = performance.now();
                let response = "";
                const socket = net.connect(port, "127.0.0.1");
                socket.on(
                    "data",
                    (chunk) => (response += chunk.toString("latin1")),
                );
                socket.on("close", () =>
                    resolve({
                        elapsedMs: performance.now() - started,
                        response,
                    }),
                );
                socket.on("connect", () =>
                    // 1,000 bytes promised, one sent.
                    socket.write(
                        "POST /webhooks/github HTTP/1.1\r\nHost: test\r\n" +
                            "Content-Type: application/json\r\nContent-Length: 1000\r\n\r\n{",
                    ),
                );
            },
        );
    }

    it("is answered 408 and closed at the request timeout", async () => {
        const server = await listen();
        const { port } = server.address() as net.AddressInfo;

        const { elapsedMs, response } = await stalledUpload(port);

        expect(response.split("\r\n")[0]).toBe("HTTP/1.1 408 Request Timeout");
        expect(elapsedMs).toBeGreaterThanOrEqual(300);
    }, 10_000);
});

// The cap is one pool for every sender, not a limit per client. The README and the comment in
// server-limits.ts say so, so this pins what a client sees: with the pool full, /health is refused.
describe("a connection above the cap", () => {
    /** Opens a connection and leaves its request unfinished. */
    function holdOpen(port: number): void {
        const socket = net.connect(port, "127.0.0.1");
        held.push(socket);
        socket.on("connect", () =>
            socket.write(
                "POST /webhooks/github HTTP/1.1\r\nHost: test\r\nContent-Length: 1000\r\n\r\n{",
            ),
        );
    }

    /** Waits until the server counts exactly `count` open connections. */
    async function untilConnections(
        server: http.Server,
        count: number,
    ): Promise<void> {
        for (let attempt = 0; attempt < 200; attempt += 1) {
            const current = await new Promise<number>((resolve, reject) =>
                server.getConnections((error, n) =>
                    error ? reject(error) : resolve(n),
                ),
            );
            if (current === count) return;
            await new Promise((resolve) => setTimeout(resolve, 10));
        }
        throw new Error(`the server never held exactly ${count} connections`);
    }

    /** The raw answer to GET /health; empty when the connection is closed without one. */
    function health(port: number): Promise<string> {
        return new Promise((resolve) => {
            let response = "";
            const socket = net.connect(port, "127.0.0.1");
            socket.on(
                "data",
                (chunk) => (response += chunk.toString("latin1")),
            );
            // A refused connection ends in a reset or in a plain close, depending on timing.
            // Either way `response` stays empty, which is what the test asserts.
            socket.on("error", () => {});
            socket.on("close", () => resolve(response));
            socket.on("connect", () =>
                socket.write(
                    "GET /health HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n",
                ),
            );
        });
    }

    it("is closed without an answer, /health included, once the connections below it are held open", async () => {
        const server = await listen({
            maxConnections: 2,
            headersTimeoutMs: 5_000,
            requestTimeoutMs: 10_000,
        });
        const { port } = server.address() as net.AddressInfo;

        holdOpen(port);
        await untilConnections(server, 1);
        const belowTheCap = await health(port);
        await untilConnections(server, 1);
        holdOpen(port);
        await untilConnections(server, 2);
        const atTheCap = await health(port);

        expect(belowTheCap.split("\r\n")[0]).toBe("HTTP/1.1 200 OK");
        expect(atTheCap).toBe("");
    }, 10_000);
});
