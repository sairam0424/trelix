import express from "express";
import request from "supertest";
import { sign } from "@octokit/webhooks-methods";
import { describe, expect, it, vi } from "vitest";
import { createWebhookRouter } from "../src/webhook.js";
import { AppConfig } from "../src/config.js";
import { ReviewFinding, ReviewRequest } from "../src/review-runner.js";

const config: AppConfig = {
  appId: "1",
  privateKey: "fake",
  webhookSecret: "fake",
  port: 0,
  reviewRepoPath: ".",
};

type RunReviewFn = (config: AppConfig, request: ReviewRequest) => Promise<ReviewFinding[]>;

function buildApp(runReview: RunReviewFn) {
  const app = express();
  // No outer express.json() — createWebhookRouter owns body parsing itself
  // so it can capture the raw bytes for signature verification.
  app.use("/webhooks/github", createWebhookRouter(config, { runReview }));
  return app;
}

/** Sends a webhook request with a real, correctly-signed X-Hub-Signature-256 header. */
async function sendSigned(app: express.Express, event: string, payload: unknown) {
  const body = JSON.stringify(payload);
  const signature = await sign(config.webhookSecret, body);
  return request(app)
    .post("/webhooks/github")
    .set("X-GitHub-Event", event)
    .set("Content-Type", "application/json")
    .set("X-Hub-Signature-256", signature)
    .send(body);
}

function pullRequestPayload(action: string) {
  return {
    action,
    number: 42,
    repository: { full_name: "owner/repo", owner: { login: "owner" }, name: "repo" },
    pull_request: { number: 42 },
    installation: { id: 999 },
  };
}

describe("webhook router", () => {
  it("accepts and triggers a review for a pull_request 'opened' event", async () => {
    const runReview = vi.fn<RunReviewFn>().mockResolvedValue([]);
    const app = buildApp(runReview);

    const res = await sendSigned(app, "pull_request", pullRequestPayload("opened"));

    expect(res.status).toBe(202);
    // Review runs after the response is sent — wait a tick for the fire-and-forget call.
    await new Promise((r) => setTimeout(r, 0));
    expect(runReview).toHaveBeenCalledWith(config, {
      owner: "owner",
      repo: "repo",
      prNumber: 42,
      installationId: 999,
    });
  });

  it.each(["synchronize", "reopened"])("also triggers a review for '%s'", async (action) => {
    const runReview = vi.fn<RunReviewFn>().mockResolvedValue([]);
    const app = buildApp(runReview);

    await sendSigned(app, "pull_request", pullRequestPayload(action));

    await new Promise((r) => setTimeout(r, 0));
    expect(runReview).toHaveBeenCalledTimes(1);
  });

  it("ignores pull_request actions outside opened/synchronize/reopened", async () => {
    const runReview = vi.fn<RunReviewFn>();
    const app = buildApp(runReview);

    const res = await sendSigned(app, "pull_request", pullRequestPayload("closed"));

    expect(res.status).toBe(202);
    expect(res.body.ignored).toBe(true);
    expect(runReview).not.toHaveBeenCalled();
  });

  it("ignores non-pull_request event types entirely", async () => {
    const runReview = vi.fn<RunReviewFn>();
    const app = buildApp(runReview);

    const res = await sendSigned(app, "issues", { action: "opened" });

    expect(res.status).toBe(202);
    expect(res.body.ignored).toBe(true);
    expect(runReview).not.toHaveBeenCalled();
  });

  it("does not fail the HTTP response even if the review run throws", async () => {
    const runReview = vi.fn<RunReviewFn>().mockRejectedValue(new Error("boom"));
    const app = buildApp(runReview);

    const res = await sendSigned(app, "pull_request", pullRequestPayload("opened"));

    expect(res.status).toBe(202);
  });

  describe("signature verification", () => {
    it("rejects a request with no X-Hub-Signature-256 header", async () => {
      const runReview = vi.fn<RunReviewFn>();
      const app = buildApp(runReview);

      const res = await request(app)
        .post("/webhooks/github")
        .set("X-GitHub-Event", "pull_request")
        .set("Content-Type", "application/json")
        .send(JSON.stringify(pullRequestPayload("opened")));

      expect(res.status).toBe(401);
      expect(runReview).not.toHaveBeenCalled();
    });

    it("rejects a request with a signature computed from the wrong secret", async () => {
      const runReview = vi.fn<RunReviewFn>();
      const app = buildApp(runReview);
      const body = JSON.stringify(pullRequestPayload("opened"));
      const wrongSignature = await sign("not-the-real-secret", body);

      const res = await request(app)
        .post("/webhooks/github")
        .set("X-GitHub-Event", "pull_request")
        .set("Content-Type", "application/json")
        .set("X-Hub-Signature-256", wrongSignature)
        .send(body);

      expect(res.status).toBe(401);
      expect(runReview).not.toHaveBeenCalled();
    });

    it("rejects a request whose body was tampered with after signing", async () => {
      const runReview = vi.fn<RunReviewFn>();
      const app = buildApp(runReview);
      const originalBody = JSON.stringify(pullRequestPayload("opened"));
      const signature = await sign(config.webhookSecret, originalBody);
      const tamperedBody = JSON.stringify(pullRequestPayload("closed")); // signed for "opened", sent as "closed"

      const res = await request(app)
        .post("/webhooks/github")
        .set("X-GitHub-Event", "pull_request")
        .set("Content-Type", "application/json")
        .set("X-Hub-Signature-256", signature)
        .send(tamperedBody);

      expect(res.status).toBe(401);
      expect(runReview).not.toHaveBeenCalled();
    });

    it("accepts a request with a correctly signed body (control case)", async () => {
      const runReview = vi.fn<RunReviewFn>().mockResolvedValue([]);
      const app = buildApp(runReview);

      const res = await sendSigned(app, "pull_request", pullRequestPayload("opened"));

      expect(res.status).toBe(202);
    });
  });

  describe("payload size limit", () => {
    it(
      "rejects a body over GitHub's 25MB webhook payload cap with 413",
      async () => {
        const runReview = vi.fn<RunReviewFn>();
        const app = buildApp(runReview);
        const oversizedPayload = {
          ...pullRequestPayload("opened"),
          padding: "x".repeat(26 * 1024 * 1024),
        };

        const res = await request(app)
          .post("/webhooks/github")
          .set("X-GitHub-Event", "pull_request")
          .set("Content-Type", "application/json")
          .send(JSON.stringify(oversizedPayload));

        expect(res.status).toBe(413);
        expect(runReview).not.toHaveBeenCalled();
      },
      15_000,
    );
  });
});
