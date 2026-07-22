import { Router, Request, Response } from "express";
import { runReview, ReviewRequest, ReviewFinding } from "./review-runner.js";
import { AppConfig } from "./config.js";

interface PullRequestWebhookPayload {
  action: string;
  number: number;
  repository: { full_name: string; owner: { login: string }; name: string };
  pull_request: { number: number };
  installation?: { id: number };
}

const HANDLED_ACTIONS = new Set(["opened", "synchronize", "reopened"]);

export interface WebhookRouterOptions {
  /** Injectable for tests — defaults to the real runReview (shells out to the trelix CLI). */
  runReview?: (config: AppConfig, request: ReviewRequest) => Promise<ReviewFinding[]>;
}

/**
 * Routes `pull_request` webhook deliveries, mirroring the existing
 * .github/workflows/trelix-review.yml Actions workflow's trigger
 * (`types: [opened, synchronize, reopened]`).
 *
 * ⚠️ Signature verification (HMAC-SHA256 over the raw body against
 * X-Hub-Signature-256) is NOT implemented yet — see item 6b. This router
 * must not be exposed on a public endpoint until that lands; it is wired
 * here only so 6b has a real request-handling skeleton to add
 * verification to, not a stub function signature.
 */
export function createWebhookRouter(config: AppConfig, options: WebhookRouterOptions = {}): Router {
  const router = Router();
  const runReviewFn = options.runReview ?? runReview;

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
