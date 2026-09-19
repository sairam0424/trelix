import express from "express";
import { loadConfig } from "./config.js";
import { sweepStaleWorkspaces } from "./repo-checkout.js";
import { createWebhookRouter } from "./webhook.js";

const config = loadConfig();
const app = express();

app.get("/health", (_req, res) => {
    res.status(200).json({ status: "ok" });
});

app.use("/webhooks/github", createWebhookRouter(config));

// Best-effort: removes any trelix-review-* workspace a previous instance
// leaked (e.g. via SIGKILL mid-checkout) before serving any webhook.
void sweepStaleWorkspaces();

app.listen(config.port, () => {
    console.log(`trelix GitHub App listening on port ${config.port}`);
});
