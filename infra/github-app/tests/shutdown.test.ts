import { describe, expect, it, vi } from "vitest";
import type { ShutdownReport } from "../src/queue.js";
import {
    drainAndReport,
    installShutdownHandlers,
    type ShutdownDeps,
} from "../src/shutdown.js";

function makeDeps(
    report: ShutdownReport | Error = { abandoned: 0, stillRunning: 0 },
) {
    const lines: string[] = [];
    const handlers = new Map<string, () => void>();
    const server = { close: vi.fn(), closeIdleConnections: vi.fn() };
    const queue = {
        shutdown: vi.fn(async (_graceMs: number) => {
            if (report instanceof Error) {
                throw report;
            }
            return report;
        }),
    };
    const proc = {
        on: vi.fn((signal: string, listener: () => void) => {
            handlers.set(signal, listener);
        }),
        exit: vi.fn(),
    };
    const deps: ShutdownDeps = {
        server,
        queue,
        process: proc,
        graceMs: 1234,
        log: (line) => lines.push(line),
    };
    return { deps, lines, handlers, server, queue, proc };
}

/** Lets the handler's promise chain finish. */
async function settle(): Promise<void> {
    for (let i = 0; i < 10; i += 1) {
        await Promise.resolve();
    }
}

describe("drainAndReport", () => {
    it("closes the listener, drains the queue for the grace period and exits 0 when nothing was lost", async () => {
        const h = makeDeps({ abandoned: 0, stillRunning: 0 });

        const status = await drainAndReport(h.deps, "SIGTERM");

        expect(status).toBe(0);
        expect(h.server.close).toHaveBeenCalledTimes(1);
        expect(h.server.closeIdleConnections).toHaveBeenCalledTimes(1);
        expect(h.queue.shutdown).toHaveBeenCalledWith(1234);
        expect(h.lines).toEqual([
            "[shutdown] SIGTERM received: closing the listener and draining the review queue",
            "[shutdown] drained: 0 queued review(s) dropped, 0 running review(s) cut off",
        ]);
    });

    it("exits 1 and says how many reviews were dropped", async () => {
        const h = makeDeps({ abandoned: 3, stillRunning: 0 });

        const status = await drainAndReport(h.deps, "SIGINT");

        expect(status).toBe(1);
        expect(h.lines[1]).toBe(
            "[shutdown] drained: 3 queued review(s) dropped, 0 running review(s) cut off",
        );
    });

    it("exits 1 when a running review was cut off", async () => {
        const h = makeDeps({ abandoned: 0, stillRunning: 2 });

        expect(await drainAndReport(h.deps, "SIGTERM")).toBe(1);
        expect(h.lines[1]).toBe(
            "[shutdown] drained: 0 queued review(s) dropped, 2 running review(s) cut off",
        );
    });

    it("uses a 25 second grace period by default", async () => {
        const h = makeDeps();

        await drainAndReport({ ...h.deps, graceMs: undefined }, "SIGTERM");

        expect(h.queue.shutdown).toHaveBeenCalledWith(25_000);
    });

    it("works with a server that has no closeIdleConnections", async () => {
        const h = makeDeps();
        const deps: ShutdownDeps = { ...h.deps, server: { close: vi.fn() } };

        expect(await drainAndReport(deps, "SIGTERM")).toBe(0);
    });

    it("never rejects: a failure is logged by name only and exits 1", async () => {
        const h = makeDeps(new TypeError("drain broke: canary-secret"));

        const status = await drainAndReport(h.deps, "SIGTERM");

        expect(status).toBe(1);
        expect(h.lines.at(-1)).toBe(
            "[shutdown] failed while draining: TypeError",
        );
        expect(h.lines.join("\n")).not.toContain("canary-secret");
    });

    it("logs to the console when no logger is given", async () => {
        const log = vi.spyOn(console, "log").mockImplementation(() => {});
        try {
            const h = makeDeps();

            await drainAndReport({ ...h.deps, log: undefined }, "SIGTERM");

            expect(log).toHaveBeenCalledTimes(2);
        } finally {
            log.mockRestore();
        }
    });
});

describe("installShutdownHandlers", () => {
    it("listens for SIGTERM and SIGINT", () => {
        const h = makeDeps();

        installShutdownHandlers(h.deps);

        expect([...h.handlers.keys()].sort()).toEqual(["SIGINT", "SIGTERM"]);
        expect(h.server.close).not.toHaveBeenCalled();
    });

    it("drains on the first signal and then exits with the status it chose", async () => {
        const h = makeDeps({ abandoned: 1, stillRunning: 0 });
        installShutdownHandlers(h.deps);

        h.handlers.get("SIGTERM")?.();
        await settle();

        expect(h.queue.shutdown).toHaveBeenCalledTimes(1);
        expect(h.proc.exit).toHaveBeenCalledTimes(1);
        expect(h.proc.exit).toHaveBeenCalledWith(1);
    });

    it("exits 0 after a clean drain", async () => {
        const h = makeDeps();
        installShutdownHandlers(h.deps);

        h.handlers.get("SIGINT")?.();
        await settle();

        expect(h.proc.exit).toHaveBeenCalledWith(0);
    });

    it("does not exit before the drain has finished", async () => {
        const h = makeDeps();
        let finish!: (report: ShutdownReport) => void;
        h.queue.shutdown.mockImplementation(
            () =>
                new Promise<ShutdownReport>((resolve) => {
                    finish = resolve;
                }),
        );
        installShutdownHandlers(h.deps);

        h.handlers.get("SIGTERM")?.();
        await settle();
        expect(h.proc.exit).not.toHaveBeenCalled();

        finish({ abandoned: 0, stillRunning: 0 });
        await settle();
        expect(h.proc.exit).toHaveBeenCalledWith(0);
    });

    it("ends the process at once with status 1 on a second signal while draining", async () => {
        const h = makeDeps();
        h.queue.shutdown.mockImplementation(() => new Promise(() => {}));
        installShutdownHandlers(h.deps);

        h.handlers.get("SIGTERM")?.();
        await settle();
        h.handlers.get("SIGINT")?.();

        expect(h.proc.exit).toHaveBeenCalledWith(1);
        expect(h.queue.shutdown).toHaveBeenCalledTimes(1);
    });
});
