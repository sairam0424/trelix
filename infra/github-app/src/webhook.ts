import express, { Router, Request, Response, NextFunction } from "express";
import { verify } from "@octokit/webhooks-methods";
import { runReview, ReviewRequest, ReviewFinding } from "./review-runner.js";
import { AppConfig } from "./config.js";

interface PullRequestWebhookPayload {
  action: string;
  number: number;
  repository: { full_name: string; owner: { login: string }; name: string };
  pull_request: { number: number };
  installation?: { id: number };
}

interface RequestWithRawBody extends Request {
  rawBody?: string;
}

const HANDLED_ACTIONS = new Set(["opened", "synchronize", "reopened"]);

// GitHub caps webhook payloads at 25MB (see docs.github.com/en/webhooks/
// webhook-events-and-payloads) — matching that cap here rejects an
// oversized body during parsing rather than buffering an arbitrarily
// large request into memory. Signature verification happens AFTER body
// parsing, so this limit is the only defense against a sender who
// doesn't know the webhook secret sending a deliberately huge payload.
const MAX_WEBHOOK_PAYLOAD_BYTES = "25mb";

export interface WebhookRouterOptions {
  /** Injectable for tests — defaults to the real runReview (shells out to the trelix CLI). */
  runReview?: (config: AppConfig, request: ReviewRequest) => Promise<ReviewFinding[]>;
}

/**
 * Verifies X-Hub-Signature-256 (HMAC-SHA256 over the raw request body,
 * keyed by the webhook secret) using @octokit/webhooks-methods' verify(),
 * which compares via crypto.timingSafeEqual — never a naive string
 * compare, which would leak timing information about how many leading
 * bytes matched. Rejects with 401 before the route handler (and thus
 * runReview) ever sees an unverified payload.
 */
function verifySignature(config: AppConfig) {
  return async (req: RequestWithRawBody, res: Response, next: NextFunction): Promise<void> => {
    const signature = req.header("X-Hub-Signature-256");
    if (!signature || !req.rawBody) {
      res.status(401).json({ error: "missing signature or body" });
      return;
    }

    const isValid = await verify(config.webhookSecret, req.rawBody, signature);
    if (!isValid) {
      res.status(401).json({ error: "signature verification failed" });
      return;
    }

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
  const runReviewFn = options.runReview ?? runReview;

  router.use(
    express.json({
      limit: MAX_WEBHOOK_PAYLOAD_BYTES,
      verify: (req: RequestWithRawBody, _res, buf) => {
        req.rawBody = buf.toString("utf8");
      },
    }),
  );
  router.use(verifySignature(config));

  router.post("/", async (req: Request, res: Response) => {
    const event = req.header("X-GitHub-Event");

    if (event !== "pull_request") {
      res.status(202).json({ ignored: true, reason: `unhandled event: ${event}` });
      return;
    }

    const payload = req.body as PullRequestWebhookPayload;

    if (!HANDLED_ACTIONS.has(payload.action)) {
      res.status(202).json({ ignored: true, reason: `unhandled action: ${payload.action}` });
      return;
    }

    // Acknowledge immediately — GitHub expects a fast response and will
    // retry/disable the hook on repeated timeouts. Review runs after.
    res.status(202).json({ accepted: true });

    try {
      await runReviewFn(config, {
        owner: payload.repository.owner.login,
        repo: payload.repository.name,
        prNumber: payload.pull_request.number,
        installationId: payload.installation?.id,
      });
    } catch (err) {
      console.error(
        `[webhook] review failed for ${payload.repository.full_name}#${payload.pull_request.number}:`,
        err,
      );
    }
  });

  return router;
}
