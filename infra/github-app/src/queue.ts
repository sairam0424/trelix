import type { ClaimStore } from "./claims.js";
import { reportIfRejected } from "./error-handler.js";

/**
 * A bounded, in-process FIFO for review jobs. Every webhook delivery is
 * answered at once; the review itself runs afterwards, here, under these caps:
 *
 * - `capacity` jobs may wait for a slot (default 20);
 * - `perGroupCapacity` of those may belong to the same group, which the
 *   webhook sets to the installation (default: half of `capacity`, rounded
 *   up), so one installation cannot fill the wait line;
 * - `concurrency` jobs may run at the same time (default 2);
 * - `perGroupConcurrency` of them may belong to the same group (default 1),
 *   so one installation cannot take every slot;
 * - a job may hold its slot for `jobTimeoutMs` at most (default 17 minutes).
 *
 * Memory is bounded by construction: `waiting` never holds more than
 * `capacity` jobs, at most `concurrency` run, and the dedupe claims are
 * bounded by ClaimStore. The timers are the watchdog of each running job
 * (cleared when the job ends, and unref'd, so they never keep the process
 * alive) and the one `shutdown` arms. Nothing is persisted: a restart loses
 * the jobs that were waiting (the webhook had already answered 202, so GitHub
 * does not send them again).
 *
 * The worker cannot be killed by a job: a job that throws, rejects, or
 * returns something unexpected is reported once through `onError`, its claim
 * is released, and the next job starts. A job that never settles (a hung
 * request, a child that ignores its kill) is cut off by the watchdog:
 * `onError` gets a `JobTimeoutError`, the claim is released, the slot is
 * taken back and the next job starts. The job itself cannot be stopped; if it
 * settles later, that is ignored.
 */

/**
 * How a job ended. "done" keeps its dedupe claim (the same key is refused for a
 * day); "retry" releases it, because the job ended without a result worth
 * keeping and the same key may be sent again. A job that throws counts as
 * "retry".
 */
export type JobOutcome = "done" | "retry";

export interface Job {
    /** Dedupe key: a second job with the same key is refused while the first is queued, running or kept. */
    readonly key: string;
    /** Jobs of one group share `perGroupConcurrency`; the webhook uses the installation id. */
    readonly group: string;
    /** Names the job in logs: no secrets, no newlines. */
    readonly label: string;
    readonly run: () => Promise<JobOutcome>;
}

/**
 * What `submit` did with a job. Only "accepted" means the job will run:
 * "duplicate" (its key is claimed), "full" (the wait line is at capacity),
 * "group_full" (the job's group already has `perGroupCapacity` jobs waiting)
 * and "closed" (the queue is shutting down) leave no trace.
 */
export type SubmitResult =
    "accepted" | "duplicate" | "full" | "group_full" | "closed";

/** The part of the queue the webhook needs; tests substitute their own. */
export interface JobSubmitter {
    submit(job: Job): SubmitResult;
}

export interface QueueLimits {
    /** Most jobs waiting for a slot. A job that can start at once never waits. */
    readonly capacity: number;
    /** Most waiting jobs of one group: its share of the wait line. */
    readonly perGroupCapacity: number;
    /** Most jobs running at the same time. */
    readonly concurrency: number;
    /** Most running jobs of one group. */
    readonly perGroupConcurrency: number;
    /** Longest a job may hold a running slot, in milliseconds; then the queue takes the slot back. */
    readonly jobTimeoutMs: number;
}

export interface JobQueueOptions extends QueueLimits {
    readonly claims: ClaimStore;
    /**
     * Called once for every job that throws, rejects or outlives `jobTimeoutMs`;
     * must not be trusted with secrets in `err`. It may be async: nothing waits
     * for it, and a rejection is handled like a throw.
     */
    readonly onError: (err: unknown, label: string) => void;
}

export interface ShutdownReport {
    /** Jobs that were waiting and will never run. */
    readonly abandoned: number;
    /** Jobs still running when the grace period ended. */
    readonly stillRunning: number;
}

/**
 * What `onError` gets for a job that outlived its deadline. The name carries
 * the kind ("Timeout") into the log line; the message says the slot was taken
 * back, not that the job stopped: it cannot be stopped.
 */
export class JobTimeoutError extends Error {
    constructor(deadlineMs: number) {
        super(
            `the job did not finish within ${deadlineMs} ms; its slot was released`,
        );
        this.name = "JobTimeoutError";
    }
}

/** Written to the console when the failure of a job cannot be logged. It quotes nothing: what was thrown is not known to be free of secrets. */
const LOG_FAILURE_LINE = "[queue] a job failed; the error could not be logged";

function reportLogFailure(): void {
    try {
        console.error(LOG_FAILURE_LINE);
    } catch {
        // The console was the last place left to report to.
    }
}

export class JobQueue implements JobSubmitter {
    private readonly waiting: Job[] = [];
    private readonly runningByGroup = new Map<string, number>();
    private running = 0;
    private closed = false;
    private readonly idleWaiters = new Set<() => void>();

    constructor(private readonly options: JobQueueOptions) {}

    /**
     * Claims the job's key, then queues it. The claim comes first, in the same
     * synchronous call, so two deliveries of one commit cannot both be
     * accepted (and a duplicate is a duplicate even from a group whose share is
     * used up); a job that is then refused for lack of room gives its claim
     * back, so the redelivery of a 503 is not taken for a duplicate.
     */
    submit(job: Job): SubmitResult {
        if (this.closed) {
            return "closed";
        }
        if (!this.options.claims.claim(job.key)) {
            return "duplicate";
        }
        const refusal = this.canStartNow(job.group)
            ? null
            : this.refusalToWait(job.group);
        if (refusal !== null) {
            this.options.claims.release(job.key);
            return refusal;
        }
        this.waiting.push(job);
        this.pump();
        return "accepted";
    }

    /** Sizes, for logs and tests. */
    counts(): { readonly waiting: number; readonly running: number } {
        return { waiting: this.waiting.length, running: this.running };
    }

    /** How many dedupe claims are held, for logs and tests. */
    claimCounts(): { readonly inFlight: number; readonly kept: number } {
        return this.options.claims.counts();
    }

    /**
     * Stops taking jobs, drops the ones still waiting (their claims are
     * released) and waits up to `graceMs` for the running ones. Never rejects.
     * Jobs still running when it returns are not interrupted; the caller
     * decides what to do with the process.
     */
    async shutdown(graceMs: number): Promise<ShutdownReport> {
        this.closed = true;
        const abandoned = this.waiting.splice(0);
        for (const job of abandoned) {
            this.options.claims.release(job.key);
        }
        if (this.running > 0) {
            await this.untilIdle(graceMs);
        }
        return { abandoned: abandoned.length, stillRunning: this.running };
    }

    /** Why a job that would have to wait may not: the wait line is full, or its group holds its whole share of it. */
    private refusalToWait(group: string): "full" | "group_full" | null {
        if (this.waiting.length >= this.options.capacity) {
            return "full";
        }
        if (this.groupWaiting(group) >= this.options.perGroupCapacity) {
            return "group_full";
        }
        return null;
    }

    private groupWaiting(group: string): number {
        let count = 0;
        for (const job of this.waiting) {
            if (job.group === group) {
                count += 1;
            }
        }
        return count;
    }

    private canStartNow(group: string): boolean {
        return (
            this.running < this.options.concurrency &&
            this.groupRunning(group) < this.options.perGroupConcurrency
        );
    }

    private groupRunning(group: string): number {
        return this.runningByGroup.get(group) ?? 0;
    }

    /**
     * Starts every waiting job that may start, oldest first. A job blocked by
     * its group's cap is skipped, not waited for, so one busy installation
     * does not hold up the others; within a group the order is still FIFO.
     */
    private pump(): void {
        let index = 0;
        while (
            this.running < this.options.concurrency &&
            index < this.waiting.length
        ) {
            const job = this.waiting[index];
            if (
                job === undefined ||
                this.groupRunning(job.group) >= this.options.perGroupConcurrency
            ) {
                index += 1;
                continue;
            }
            this.waiting.splice(index, 1);
            this.start(job);
        }
    }

    private start(job: Job): void {
        this.running += 1;
        this.runningByGroup.set(job.group, this.groupRunning(job.group) + 1);
        // execute() catches everything a job can do; the .catch is for a bug in the
        // queue itself, which must not become an unhandled rejection and end the process.
        void this.execute(job).catch((err: unknown) => {
            this.report(err, job.label);
        });
    }

    private async execute(job: Job): Promise<void> {
        let outcome: JobOutcome = "retry";
        try {
            // Let the caller finish first: a webhook handler sends its answer before any work starts.
            await Promise.resolve();
            outcome = await this.runWithDeadline(job);
        } catch (err) {
            this.report(err, job.label);
        }
        this.finish(job, outcome);
    }

    /**
     * The job's outcome, or a `JobTimeoutError` once `jobTimeoutMs` has passed,
     * whichever comes first. `execute` awaits this once and then calls `finish`
     * once, so a job that settles after its deadline finds nothing waiting for
     * it: it cannot release the slot a second time or start another job. (The
     * race still subscribes to it, so a late rejection is not unhandled; it is
     * dropped, not reported, because the timeout was.)
     */
    private async runWithDeadline(job: Job): Promise<JobOutcome> {
        const { jobTimeoutMs } = this.options;
        let timer: NodeJS.Timeout | undefined;
        const deadline = new Promise<never>((_resolve, reject) => {
            timer = setTimeout(() => {
                reject(new JobTimeoutError(jobTimeoutMs));
            }, jobTimeoutMs);
            // The watchdog frees a slot; it is never a reason to keep the process alive.
            timer.unref();
        });
        try {
            return await Promise.race([job.run(), deadline]);
        } finally {
            clearTimeout(timer);
        }
    }

    private finish(job: Job, outcome: JobOutcome): void {
        this.running -= 1;
        const left = this.groupRunning(job.group) - 1;
        if (left > 0) {
            this.runningByGroup.set(job.group, left);
        } else {
            this.runningByGroup.delete(job.group);
        }
        if (outcome === "done") {
            this.options.claims.keep(job.key);
        } else {
            this.options.claims.release(job.key);
        }
        this.pump();
        if (this.running === 0) {
            this.wakeIdleWaiters();
        }
    }

    /** Reports to `onError`; if that throws or its promise rejects, says so on the console with a fixed line and nothing of the error. */
    private report(err: unknown, label: string): void {
        try {
            reportIfRejected(
                this.options.onError(err, label),
                reportLogFailure,
            );
        } catch {
            reportLogFailure();
        }
    }

    private untilIdle(graceMs: number): Promise<void> {
        return new Promise((resolve) => {
            const wake = (): void => {
                clearTimeout(timer);
                this.idleWaiters.delete(wake);
                resolve();
            };
            const timer = setTimeout(wake, graceMs);
            this.idleWaiters.add(wake);
        });
    }

    private wakeIdleWaiters(): void {
        for (const wake of [...this.idleWaiters]) {
            wake();
        }
    }
}
