import type { ClaimStore } from "./claims.js";

/**
 * A bounded, in-process FIFO for review jobs. Every webhook delivery is
 * answered at once; the review itself runs afterwards, here, under three caps:
 *
 * - `capacity` jobs may wait for a slot (default 20);
 * - `concurrency` jobs may run at the same time (default 2);
 * - `perGroupConcurrency` of them may belong to the same group, which the
 *   webhook sets to the installation (default 1), so one installation cannot
 *   take every slot.
 *
 * Memory is bounded by construction: `waiting` never holds more than
 * `capacity` jobs, at most `concurrency` run, and the dedupe claims are
 * bounded by ClaimStore. There is no timer except the one `shutdown` arms, and
 * nothing is persisted: a restart loses the jobs that were waiting (the
 * webhook had already answered 202, so GitHub does not send them again).
 *
 * The worker cannot be killed by a job: a job that throws, rejects, or
 * returns something unexpected is reported once through `onError`, its claim
 * is released, and the next job starts. The queue puts no time limit on a job;
 * `runReview` has its own (REVIEW_TIMEOUT_MS), which is unchanged.
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
 * "duplicate" (its key is claimed), "full" (the wait line is at capacity) and
 * "closed" (the queue is shutting down) leave no trace.
 */
export type SubmitResult = "accepted" | "duplicate" | "full" | "closed";

/** The part of the queue the webhook needs; tests substitute their own. */
export interface JobSubmitter {
    submit(job: Job): SubmitResult;
}

export interface QueueLimits {
    /** Most jobs waiting for a slot. A job that can start at once never waits. */
    readonly capacity: number;
    /** Most jobs running at the same time. */
    readonly concurrency: number;
    /** Most running jobs of one group. */
    readonly perGroupConcurrency: number;
}

export interface JobQueueOptions extends QueueLimits {
    readonly claims: ClaimStore;
    /** Called once for every job that throws or rejects; must not be trusted with secrets in `err`. */
    readonly onError: (err: unknown, label: string) => void;
}

export interface ShutdownReport {
    /** Jobs that were waiting and will never run. */
    readonly abandoned: number;
    /** Jobs still running when the grace period ended. */
    readonly stillRunning: number;
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
     * accepted; a job that is then refused for lack of room gives its claim
     * back, so the redelivery of a 503 is not taken for a duplicate.
     */
    submit(job: Job): SubmitResult {
        if (this.closed) {
            return "closed";
        }
        if (!this.options.claims.claim(job.key)) {
            return "duplicate";
        }
        if (!this.canStartNow(job.group) && this.isFull()) {
            this.options.claims.release(job.key);
            return "full";
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

    private isFull(): boolean {
        return this.waiting.length >= this.options.capacity;
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
            outcome = await job.run();
        } catch (err) {
            this.report(err, job.label);
        }
        this.finish(job, outcome);
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

    /** Reports to `onError`; if that throws, says so on the console with a fixed line and nothing of the error. */
    private report(err: unknown, label: string): void {
        try {
            this.options.onError(err, label);
        } catch {
            try {
                console.error(
                    "[queue] a job failed; the error could not be logged",
                );
            } catch {
                // The console was the last place left to report to.
            }
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
