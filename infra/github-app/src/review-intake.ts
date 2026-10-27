import type { AbuseControls } from "./abuse-controls.js";
import {
    ClaimStore,
    DEFAULT_CLAIM_KEEP_MS,
    DEFAULT_MAX_KEPT_CLAIMS,
} from "./claims.js";
import { isCommitId } from "./commit-id.js";
import type { AppConfig } from "./config.js";
import { createJobFailureLogger } from "./job-log.js";
import { isInstallationAllowed } from "./policy.js";
import { JobOutcome, JobQueue, JobSubmitter, QueueLimits } from "./queue.js";
import type {
    ReviewFinding,
    ReviewRequest,
    RunReviewOptions,
} from "./review-runner.js";

/**
 * What a `pull_request` delivery is read for. `repository.id`,
 * `pull_request.head.sha` and the rest are typed as present because GitHub
 * always sends them, but every one is checked before use: the installation
 * tokens are scoped to the repository id, the head sha decides which commit a
 * review may be posted on, and the installation id and the owner decide who
 * is served.
 */
export interface PullRequestPayload {
    action: string;
    number: number;
    repository: {
        id: number;
        full_name: string;
        owner: { login: string };
        name: string;
    };
    pull_request: { number: number; head: { sha: string } };
    installation?: { id: number };
}

/** The review runner as the intake calls it: `onNoVerdict` lets the queue release the claim on the commit. */
export type RunReviewFn = (
    config: AppConfig,
    request: ReviewRequest,
    options?: Pick<RunReviewOptions, "onNoVerdict">,
) => Promise<ReviewFinding[]>;

export interface IntakeLog {
    info(line: string): void;
    warn(line: string): void;
}

// Each call looks console up again, so a test can spy on it.
const consoleLog: IntakeLog = {
    info: (line) => console.log(line),
    warn: (line) => console.warn(line),
};

export interface IntakeDeps {
    readonly config: AppConfig;
    readonly controls: AbuseControls;
    readonly queue: JobSubmitter;
    readonly runReview: RunReviewFn;
    readonly log?: IntakeLog;
}

/** The answer to a delivery: a status and a JSON body; a 503 also says when to retry. */
export interface IntakeDecision {
    readonly status: 202 | 503;
    readonly body: Readonly<Record<string, unknown>>;
    readonly retryAfterSeconds?: number;
}

/**
 * How long a 503 asks the sender to wait. GitHub does not act on it; the
 * redelivery backstop (`redeliver-failed-webhooks.yml`, every 6 hours) is what
 * retries a delivery that got a 503. It is there for other senders.
 */
export const RETRY_AFTER_SECONDS = 60;

/** Longest owner, repository or account name copied into a log line. */
const MAX_LOGGED_NAME_CHARS = 100;

const IGNORED: IntakeDecision = { status: 202, body: { ignored: true } };

function isPositiveId(value: unknown): value is number {
    return (
        typeof value === "number" && Number.isSafeInteger(value) && value > 0
    );
}

function isNonEmptyText(value: unknown): value is string {
    return typeof value === "string" && value !== "";
}

/** Text from a payload, fit for a log line: cut first, then anything outside a name's usual characters becomes "?". Linear. */
function logSafe(value: unknown): string {
    if (typeof value !== "string") {
        return "unknown";
    }
    return value.slice(0, MAX_LOGGED_NAME_CHARS).replace(/[^\w.\-/#@]/g, "?");
}

/** The fields a review and the policy need, all checked; null when any is missing or unusable. */
interface Delivery {
    readonly owner: string;
    readonly repo: string;
    readonly prNumber: number;
    readonly installationId: number;
    /** `owner/repo#n` cut and cleaned for a log line. */
    readonly label: string;
}

function readDelivery(payload: PullRequestPayload): Delivery | null {
    const owner = payload.repository?.owner?.login;
    const repo = payload.repository?.name;
    const prNumber = payload.pull_request?.number;
    const installationId = payload.installation?.id;
    if (
        !isNonEmptyText(owner) ||
        !isNonEmptyText(repo) ||
        !isPositiveId(prNumber) ||
        !isPositiveId(installationId)
    ) {
        return null;
    }
    const label = `${logSafe(owner)}/${logSafe(repo)}#${prNumber}`;
    return { owner, repo, prNumber, installationId, label };
}

/** The warning for a delivery the queue refused: the cause, and the delivery as `owner/repo#n`. */
function refusalLogLine(
    result: "full" | "group_full" | "closed",
    delivery: Delivery,
): string {
    switch (result) {
        case "full":
            return `[webhook] review queue is full: answering 503 for ${delivery.label}`;
        case "group_full":
            return `[webhook] installation ${delivery.installationId} already has its share of the review queue waiting: answering 503 for ${delivery.label}`;
        case "closed":
            return `[webhook] shutting down: answering 503 for ${delivery.label}`;
    }
}

const NO_COMMIT_REASON =
    "pull_request payload has no usable repository id or head sha";
const NO_DELIVERY_FIELDS_REASON =
    "pull_request payload has no usable owner, repository name, pull request number or installation id";
const DUPLICATE_REASON =
    "a review of this commit is already queued, running or done";

/**
 * Decides what to do with a `pull_request` delivery whose event and action
 * the webhook already accepted, in this order, each step ending the delivery
 * with a 202 unless noted:
 *
 * 1. the kill switch (`reviewsEnabled` false): ignored, one log line;
 * 2. no usable repository id or head sha: ignored (a review needs both);
 * 3. no usable owner, repository name, pull request number or installation
 *    id: ignored (a review could not run, and nothing could be keyed);
 * 4. the installation policy: an installation outside the allow-list is
 *    ignored, with the same body as the kill switch, so a sender learns
 *    nothing about the policy from the answer;
 * 5. the queue: claimed and queued (202), already claimed (202, ignored), no
 *    room (503 with Retry-After, so the redelivery backstop retries it: the
 *    wait line is full, or the installation already has its share of it
 *    waiting; the sender gets the same answer for both) or shutting down (503).
 *
 * Nothing here waits for the review: the answer is ready as soon as the job
 * is queued, well inside GitHub's 10 seconds.
 */
class ReviewIntake {
    private readonly log: IntakeLog;

    constructor(private readonly deps: IntakeDeps) {
        this.log = deps.log ?? consoleLog;
    }

    handle(payload: PullRequestPayload): IntakeDecision {
        if (!this.deps.controls.reviewsEnabled) {
            this.log.warn(
                `[webhook] reviews are disabled: ignoring pull_request delivery for ${logSafe(payload.repository?.full_name)}`,
            );
            return IGNORED;
        }
        const repositoryId = payload.repository?.id;
        const headSha = payload.pull_request?.head?.sha;
        if (!isPositiveId(repositoryId) || !isCommitId(headSha)) {
            // A review needs both: the tokens are scoped to the repository and the Check
            // goes on the head commit. A real GitHub delivery always has them; this one is
            // ignored (not failed, so the redelivery backstop does not retry it for ever).
            return this.ignoreUnusable(payload, NO_COMMIT_REASON);
        }
        const delivery = readDelivery(payload);
        if (delivery === null) {
            return this.ignoreUnusable(payload, NO_DELIVERY_FIELDS_REASON);
        }
        const target = {
            account: delivery.owner,
            installationId: delivery.installationId,
        };
        if (!isInstallationAllowed(this.deps.controls.installPolicy, target)) {
            this.log.warn(
                `[webhook] ignoring pull_request delivery for ${delivery.label}: installation ${delivery.installationId} is not allowed by the install policy`,
            );
            return IGNORED;
        }
        return this.enqueue(delivery, repositoryId, headSha);
    }

    private ignoreUnusable(
        payload: PullRequestPayload,
        reason: string,
    ): IntakeDecision {
        this.log.warn(
            `[webhook] ignoring pull_request delivery for ${logSafe(payload.repository?.full_name)}: ${reason}`,
        );
        return { status: 202, body: { ignored: true, reason } };
    }

    private enqueue(
        delivery: Delivery,
        repositoryId: number,
        headSha: string,
    ): IntakeDecision {
        const request: ReviewRequest = {
            owner: delivery.owner,
            repo: delivery.repo,
            prNumber: delivery.prNumber,
            installationId: delivery.installationId,
            repositoryId,
            headSha,
        };
        const result = this.deps.queue.submit({
            key: `${delivery.installationId}:${repositoryId}:${delivery.prNumber}:${headSha}`,
            group: String(delivery.installationId),
            label: delivery.label,
            run: () => this.runJob(request),
        });
        if (result === "accepted") {
            // Acknowledge immediately: GitHub expects a fast response and will
            // retry/disable the hook on repeated timeouts. The review runs after.
            return { status: 202, body: { accepted: true } };
        }
        if (result === "duplicate") {
            this.log.info(
                `[webhook] not queueing ${delivery.label} at ${headSha}: ${DUPLICATE_REASON}`,
            );
            return {
                status: 202,
                body: { ignored: true, reason: DUPLICATE_REASON },
            };
        }
        return this.refuse(result, delivery);
    }

    /** Runs the review; "retry" when it ended without a verdict, so the commit may be sent again. */
    private async runJob(request: ReviewRequest): Promise<JobOutcome> {
        let outcome: JobOutcome = "done";
        await this.deps.runReview(this.deps.config, request, {
            onNoVerdict: () => {
                outcome = "retry";
            },
        });
        return outcome;
    }

    /**
     * The 503 for a job the queue could not take. An installation over its share
     * of the wait line is told exactly what a full line is told; only the log
     * line differs, and it names the installation id (not a secret), not the payload.
     */
    private refuse(
        result: "full" | "group_full" | "closed",
        delivery: Delivery,
    ): IntakeDecision {
        this.log.warn(refusalLogLine(result, delivery));
        return {
            status: 503,
            body: {
                error:
                    result === "closed"
                        ? "service is shutting down"
                        : "review queue is full",
            },
            retryAfterSeconds: RETRY_AFTER_SECONDS,
        };
    }
}

/** The intake as a function: payload in, decision out. */
export function createReviewIntake(
    deps: IntakeDeps,
): (payload: PullRequestPayload) => IntakeDecision {
    const intake = new ReviewIntake(deps);
    return (payload) => intake.handle(payload);
}

/**
 * The queue a deployment runs reviews on: the caps and the per-job deadline
 * from `limits`, dedupe claims kept 4 days (at most 10,000), and job failures
 * logged once, with the webhook secret and the private key removed. `now` is
 * injectable so a test does not wait days.
 */
export function createReviewQueue(
    config: AppConfig,
    limits: QueueLimits,
    now: () => number = () => Date.now(),
): JobQueue {
    return new JobQueue({
        ...limits,
        claims: new ClaimStore({
            keepMs: DEFAULT_CLAIM_KEEP_MS,
            maxKept: DEFAULT_MAX_KEPT_CLAIMS,
            now,
        }),
        onError: createJobFailureLogger(
            [config.webhookSecret, config.privateKey],
            "[webhook] review failed",
        ),
    });
}
