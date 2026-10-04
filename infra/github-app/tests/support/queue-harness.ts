/**
 * Shared by queue.test.ts and queue-resilience.test.ts: a queue over a claim store with a
 * clock the test moves, and jobs that start, record themselves and finish when told to.
 */
import { ClaimStore } from "../../src/claims.js";
import {
    JobQueue,
    type Job,
    type JobOutcome,
    type QueueLimits,
} from "../../src/queue.js";

export const DAY_MS = 24 * 60 * 60 * 1000;

export interface Deferred {
    readonly promise: Promise<JobOutcome>;
    resolve(outcome?: JobOutcome): void;
    reject(reason: unknown): void;
}

export function deferred(): Deferred {
    let resolve!: (outcome: JobOutcome) => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<JobOutcome>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    return {
        promise,
        resolve: (outcome = "done") => resolve(outcome),
        reject,
    };
}

/** The queue only uses microtasks; this lets every pending one run. */
export async function settle(): Promise<void> {
    for (let i = 0; i < 25; i += 1) {
        await Promise.resolve();
    }
}

export interface Harness {
    readonly queue: JobQueue;
    readonly claims: ClaimStore;
    readonly clock: { now: number };
    readonly errors: Array<{ err: unknown; label: string }>;
    /** Labels of the jobs that have started, in order. */
    readonly started: string[];
    /** A job that starts, records itself, and finishes when the test says so. */
    hold(key: string, group?: string): { job: Job; gate: Deferred };
}

export function makeHarness(
    limits: Partial<QueueLimits> = {},
    onError?: (err: unknown, label: string) => void,
): Harness {
    const clock = { now: 1_000_000 };
    const claims = new ClaimStore({
        keepMs: DAY_MS,
        maxKept: 10_000,
        now: () => clock.now,
    });
    const errors: Array<{ err: unknown; label: string }> = [];
    const started: string[] = [];
    const queue = new JobQueue({
        capacity: 2,
        concurrency: 2,
        perGroupConcurrency: 1,
        ...limits,
        claims,
        onError:
            onError ??
            ((err, label) => {
                errors.push({ err, label });
            }),
    });
    return {
        queue,
        claims,
        clock,
        errors,
        started,
        hold(key, group = key) {
            const gate = deferred();
            const job: Job = {
                key,
                group,
                label: key,
                run: () => {
                    started.push(key);
                    return gate.promise;
                },
            };
            return { job, gate };
        },
    };
}
