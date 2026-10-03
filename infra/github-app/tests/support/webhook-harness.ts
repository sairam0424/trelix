/**
 * A webhook router over an injected review runner, for the tests of the queue,
 * the kill switch and the installation policy. Nothing here touches the network:
 * the runner is a fake whose every call waits for the test to finish it.
 */
import express from "express";
import request from "supertest";
import { sign } from "@octokit/webhooks-methods";
import { vi } from "vitest";
import type { AbuseControls } from "../../src/abuse-controls.js";
import type { AppConfig } from "../../src/config.js";
import type { QueueLimits } from "../../src/queue.js";
import {
    createReviewQueue,
    type RunReviewFn,
} from "../../src/review-intake.js";
import { createWebhookRouter } from "../../src/webhook.js";
import type { ReviewFinding, ReviewRequest } from "../../src/review-runner.js";

/** Values the job failure log must keep out of its line. */
export const HOOK_CANARY = "canary-hook-s";
export const PEM_CANARY = "canary-pem-body";

export const CONFIG: AppConfig = {
    appId: "1",
    privateKey: PEM_CANARY,
    webhookSecret: HOOK_CANARY,
    port: 0,
};

export const SHA_A = "0123456789abcdef0123456789abcdef01234567";
export const SHA_B = "fedcba9876543210fedcba9876543210fedcba98";

export const DAY_MS = 24 * 60 * 60 * 1000;

export interface Deferred<T> {
    readonly promise: Promise<T>;
    resolve(value: T): void;
    reject(reason: unknown): void;
}

export function deferred<T>(): Deferred<T> {
    let resolve!: (value: T) => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<T>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    return { promise, resolve, reject };
}

export interface PayloadOptions {
    action?: string;
    sha?: string;
    prNumber?: number;
    repositoryId?: number;
    owner?: string;
    repo?: string;
    installationId?: number | null;
}

export function pullRequestPayload(options: PayloadOptions = {}) {
    const {
        action = "opened",
        sha = SHA_A,
        prNumber = 42,
        repositoryId = 4242,
        owner = "owner",
        repo = "repo",
        installationId = 999,
    } = options;
    return {
        action,
        number: prNumber,
        repository: {
            id: repositoryId,
            full_name: `${owner}/${repo}`,
            owner: { login: owner },
            name: repo,
        },
        pull_request: { number: prNumber, head: { sha } },
        ...(installationId === null
            ? {}
            : { installation: { id: installationId } }),
    };
}

/** One call to the fake review runner; the test ends it with `gate`. */
export interface ReviewCall {
    readonly request: ReviewRequest;
    readonly options: Parameters<RunReviewFn>[2];
    readonly gate: Deferred<ReviewFinding[]>;
}

export function controlledReview() {
    const calls: ReviewCall[] = [];
    const runReview = vi.fn<RunReviewFn>((_config, reviewRequest, options) => {
        const gate = deferred<ReviewFinding[]>();
        calls.push({ request: reviewRequest, options, gate });
        return gate.promise;
    });
    return { runReview, calls };
}

export interface HarnessOptions {
    controls?: Partial<AbuseControls>;
    limits?: Partial<QueueLimits>;
    runReview?: RunReviewFn;
}

export function buildHarness(options: HarnessOptions = {}) {
    const clock = { now: 5_000_000 };
    const review = controlledReview();
    const limits: QueueLimits = {
        capacity: 20,
        concurrency: 2,
        perGroupConcurrency: 1,
        ...options.limits,
    };
    // The queue a deployment runs, with a clock the test moves and the real redacting failure log.
    const queue = createReviewQueue(CONFIG, limits, () => clock.now);
    const controls: AbuseControls = {
        reviewsEnabled: true,
        installPolicy: {
            mode: "open",
            accounts: new Set(),
            installations: new Set(),
        },
        queue: limits,
        ...options.controls,
    };
    const app = express();
    app.use(
        "/webhooks/github",
        createWebhookRouter(CONFIG, {
            runReview: options.runReview ?? review.runReview,
            controls,
            queue,
        }),
    );
    return { app, queue, clock, ...review };
}

/** Sends a correctly signed delivery. The GUID header is what GitHub sets; the router must not depend on it. */
export async function send(
    app: express.Express,
    payload: unknown,
    options: { event?: string; guid?: string } = {},
) {
    const body = JSON.stringify(payload);
    const signature = await sign(CONFIG.webhookSecret, body);
    return request(app)
        .post("/webhooks/github")
        .set("X-GitHub-Event", options.event ?? "pull_request")
        .set("X-GitHub-Delivery", options.guid ?? "guid-default")
        .set("Content-Type", "application/json")
        .set("X-Hub-Signature-256", signature)
        .send(body);
}

/** Lets the macrotask queue, and so every pending microtask, run. */
export function tick(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
}
