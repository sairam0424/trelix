import type { ShutdownReport } from "./queue.js";

/**
 * Graceful shutdown for the deployed service: on SIGTERM or SIGINT it stops
 * listening, drops the reviews that have not started, gives the ones running
 * a short grace period, and exits.
 *
 * What is lost is stated, not hidden: a review that was queued but had not
 * started is dropped (its webhook was already answered 202, so GitHub does not
 * send it again), and a review still running when the grace period ends is cut
 * with the process. The exit status says which case it was: 0 when nothing
 * was lost, 1 when work was. A second signal while draining ends the process
 * at once with status 1.
 */

/** Longest a running review is given to finish after the signal. */
export const SHUTDOWN_GRACE_MS = 25_000;

export const SHUTDOWN_SIGNALS: readonly NodeJS.Signals[] = [
    "SIGTERM",
    "SIGINT",
];

export interface ClosableServer {
    close(): unknown;
    closeIdleConnections?(): void;
}

export interface DrainableQueue {
    shutdown(graceMs: number): Promise<ShutdownReport>;
}

/** The parts of `process` this module uses. */
export interface ShutdownProcess {
    on(signal: NodeJS.Signals, listener: () => void): unknown;
    exit(code: number): unknown;
}

export interface ShutdownDeps {
    readonly server: ClosableServer;
    readonly queue: DrainableQueue;
    readonly process: ShutdownProcess;
    readonly graceMs?: number;
    readonly log?: (line: string) => void;
}

function logToConsole(line: string): void {
    console.log(line);
}

function describeReport(report: ShutdownReport): string {
    return `${report.abandoned} queued review(s) dropped, ${report.stillRunning} running review(s) cut off`;
}

/**
 * The shutdown sequence, resolved with the exit status it chose. Never
 * rejects: anything that goes wrong on the way is logged and ends in status 1.
 */
export async function drainAndReport(
    deps: ShutdownDeps,
    signal: NodeJS.Signals,
): Promise<number> {
    const log = deps.log ?? logToConsole;
    log(
        `[shutdown] ${signal} received: closing the listener and draining the review queue`,
    );
    try {
        deps.server.close();
        deps.server.closeIdleConnections?.();
        const report = await deps.queue.shutdown(
            deps.graceMs ?? SHUTDOWN_GRACE_MS,
        );
        const lostWork = report.abandoned > 0 || report.stillRunning > 0;
        log(`[shutdown] drained: ${describeReport(report)}`);
        return lostWork ? 1 : 0;
    } catch (err) {
        log(
            `[shutdown] failed while draining: ${err instanceof Error ? err.name : typeof err}`,
        );
        return 1;
    }
}

/** Registers the handlers on SIGTERM and SIGINT. */
export function installShutdownHandlers(deps: ShutdownDeps): void {
    let isDraining = false;
    for (const signal of SHUTDOWN_SIGNALS) {
        deps.process.on(signal, () => {
            if (isDraining) {
                deps.process.exit(1);
                return;
            }
            isDraining = true;
            void drainAndReport(deps, signal).then((status) => {
                deps.process.exit(status);
            });
        });
    }
}
