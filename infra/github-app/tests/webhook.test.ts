import express from "express";
import request from "supertest";
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
  app.use(express.json());
  app.use("/webhooks/github", createWebhookRouter(config, { runReview }));
  return app;
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

    const res = await request(app)
      .post("/webhooks/github")
      .set("X-GitHub-Event", "pull_request")
      .send(pullRequestPayload("opened"));

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

    await request(app)
      .post("/webhooks/github")
      .set("X-GitHub-Event", "pull_request")
      .send(pullRequestPayload(action));

    await new Promise((r) => setTimeout(r, 0));
    expect(runReview).toHaveBeenCalledTimes(1);
  });

  it("ignores pull_request actions outside opened/synchronize/reopened", async () => {
    const runReview = vi.fn<RunReviewFn>();
    const app = buildApp(runReview);

    const res = await request(app)
      .post("/webhooks/github")
      .set("X-GitHub-Event", "pull_request")
      .send(pullRequestPayload("closed"));

    expect(res.status).toBe(202);
    expect(res.body.ignored).toBe(true);
    expect(runReview).not.toHaveBeenCalled();
  });

  it("ignores non-pull_request event types entirely", async () => {
    const runReview = vi.fn<RunReviewFn>();
    const app = buildApp(runReview);

    const res = await request(app)
      .post("/webhooks/github")
      .set("X-GitHub-Event", "issues")
      .send({ action: "opened" });

    expect(res.status).toBe(202);
    expect(res.body.ignored).toBe(true);
    expect(runReview).not.toHaveBeenCalled();
  });

  it("does not fail the HTTP response even if the review run throws", async () => {
    const runReview = vi.fn<RunReviewFn>().mockRejectedValue(new Error("boom"));
    const app = buildApp(runReview);

    const res = await request(app)
      .post("/webhooks/github")
      .set("X-GitHub-Event", "pull_request")
      .send(pullRequestPayload("opened"));

    expect(res.status).toBe(202);
  });
});
