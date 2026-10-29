import express, { Router, Request, Response, NextFunction } from "express";
import { verify } from "@octokit/webhooks-methods";
import { runReview } from "./review-runner.js";
import { AppConfig } from "./config.js";
import { AbuseControls, DEFAULT_ABUSE_CONTROLS } from "./abuse-controls.js";
import { JobSubmitter } from "./queue.js";
import {
  createReviewIntake,
  createReviewQueue,
  PullRequestPayload,
  RunReviewFn,
} from "./review-intake.js";

const HANDLED_ACTIONS = new Set(["opened", "synchronize", "reopened"]);

// The most bytes read from a request before its signature has verified.
// Anyone can send this many, so it bounds what an unsigned request costs
// (memory and one HMAC, never a parse). GitHub delivers payloads of up to
// 25MB, but the only event this App subscribes to, pull_request, is mostly
// its PR body (GitHub caps that at 65,536 characters), under 1 MiB.
// A larger body is answered 413 while it is read. server-limits.ts sizes its
// connection cap from this number.
export const MAX_WEBHOOK_PAYLOAD_BYTES = 5 * 1024 * 1024;

// express.json() drops one leading byte order mark before it parses, although the
// signature is over the bytes as sent, mark included. Dropping it here keeps a signed
// delivery that starts with one answered as it always was.
const BYTE_ORDER_MARK = "\uFEFF";

export interface WebhookRouterOptions {
  /** Injectable for tests — defaults to the real runReview (shells out to the trelix CLI). */
  runReview?: RunReviewFn;
  /**
   * Kill switch, installation policy and queue caps (see abuse-controls.ts). Defaults to
   * DEFAULT_ABUSE_CONTROLS: reviews on, every installation served.
   */
  controls?: AbuseControls;
  /**
   * Where accepted reviews run after the delivery is answered. Defaults to a new queue per
   * router, built from `controls.queue`; `server.ts` passes its own so it can drain it on
   * shutdown.
   */
  queue?: JobSubmitter;
}

/**
 * The error for a verified body that is not a JSON object or array. `status`
 * and `expose` are what make the error handler answer its fixed 400; the
 * message quotes nothing the sender wrote.
 */
function malformedBodyError(): Error {
  return Object.assign(new SyntaxError("webhook body is not a JSON object or array"), {
    status: 400,
    expose: true,
  });
}

/**
 * Parses like express.json() does in its default strict mode: one leading
 * byte order mark is dropped, and only an object or an array is a body (a
 * `null` would reach the route with no `action` to read). Returns undefined
 * for anything else, malformed JSON included; the parser's own error is
 * dropped because its text quotes the body.
 */
function parseJsonObject(text: string): object | undefined {
  let value: unknown;
  try {
    value = JSON.parse(
      text.startsWith(BYTE_ORDER_MARK) ? text.slice(BYTE_ORDER_MARK.length) : text,
    );
  } catch {
    return undefined;
  }
  return typeof value === "object" && value !== null ? value : undefined;
}

/**
 * Verifies X-Hub-Signature-256 (HMAC-SHA256 over the raw request body,
 * keyed by the webhook secret) using @octokit/webhooks-methods' verify(),
 * which compares via crypto.timingSafeEqual — never a naive string
 * compare, which would leak timing information about how many leading
 * bytes matched. Rejects with 401 before the route handler (and thus
 * runReview) ever sees an unverified payload.
 *
 * `req.body` is the raw bytes (express.raw below). verify() takes a string,
 * so the HMAC is over their UTF-8 decoding: the same bytes for valid UTF-8,
 * which every GitHub delivery is, and as before for anything else. Only once
 * the signature has verified is that same text parsed and put in `req.body`
 * for the route, so a sender without the secret can make the process hash a
 * body but never parse one.
 */
function verifyThenParse(config: AppConfig) {
  return async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    const signature = req.header("X-Hub-Signature-256");
    const bytes: unknown = req.body;
    if (!signature || !Buffer.isBuffer(bytes) || bytes.length === 0) {
      res.status(401).json({ error: "missing signature or body" });
      return;
    }

    const text = bytes.toString("utf8");
    const isValid = await verify(config.webhookSecret, text, signature);
    if (!isValid) {
      res.status(401).json({ error: "signature verification failed" });
      return;
    }

    const payload = parseJsonObject(text);
    if (payload === undefined) {
      next(malformedBodyError());
      return;
    }
    req.body = payload;
    next();
  };
}

/**
 * Routes `pull_request` webhook deliveries, mirroring the existing
 * .github/workflows/trelix-review.yml Actions workflow's trigger
 * (`types: [opened, synchronize, reopened]`).
 */
export function createWebhookRouter(config: AppConfig, options: WebhookRouterOptions = {}): Router {
  const router = Router();
  const controls = options.controls ?? DEFAULT_ABUSE_CONTROLS;
  const intake = createReviewIntake({
    config,
    controls,
    queue: options.queue ?? createReviewQueue(config, controls.queue),
    runReview: options.runReview ?? runReview,
  });

  // Raw bytes only: nothing is parsed until verifyThenParse has checked the signature.
  // A body of any other content type is not read at all and reaches it with no body (401).
  // `inflate: false` answers a body with a Content-Encoding other than identity 415 before any
  // of it is read. By default the limit counts the inflated bytes, so a few KB of gzip, deflate,
  // brotli or zstd could expand to the whole cap, and one HMAC, per unsigned request. GitHub's
  // documented delivery headers (docs.github.com, "Webhook events and payloads", "Delivery
  // headers") have no Content-Encoding, and its example delivery states a Content-Length.
  router.use(
    express.raw({ type: "application/json", limit: MAX_WEBHOOK_PAYLOAD_BYTES, inflate: false }),
  );
  router.use(verifyThenParse(config));

  router.post("/", (req: Request, res: Response) => {
    const event = req.header("X-GitHub-Event");

    if (event !== "pull_request") {
      res.status(202).json({ ignored: true, reason: `unhandled event: ${event}` });
      return;
    }

    const payload = req.body as PullRequestPayload;

    if (!HANDLED_ACTIONS.has(payload.action)) {
      res.status(202).json({ ignored: true, reason: `unhandled action: ${payload.action}` });
      return;
    }

    // Everything past the action filter is the intake's: the kill switch, the installation
    // policy, the dedupe claim and the queue. It answers at once; the review runs after.
    const decision = intake(payload);
    if (decision.retryAfterSeconds !== undefined) {
      res.set("Retry-After", String(decision.retryAfterSeconds));
    }
    res.status(decision.status).json(decision.body);
  });

  return router;
}
