import express from "express";
import type { Express } from "express";
import type { AppConfig } from "./config.js";
import { createErrorHandler } from "./error-handler.js";
import { createWebhookRouter } from "./webhook.js";
import type { WebhookRouterOptions } from "./webhook.js";

/**
 * What a caller can replace. The webhook router's own options come through
 * as they are: a test injects `runReview`, and `server.ts` injects the
 * `controls` and the `queue`, without `createApp` knowing about them.
 */
export interface AppDeps extends WebhookRouterOptions {
    /**
     * Where the error handler writes its one line per error; defaults to the
     * console. It may be async: a rejected promise is handled like a throw.
     */
    readonly logError?: (line: string) => void;
}

// Looks up console.error on each call, so a test can spy on it.
function logToConsole(line: string): void {
    console.error(line);
}

/**
 * Builds the Express app without listening on a port, so a test can drive it
 * in process. `server.ts` does the rest: it loads the config and the abuse
 * controls, builds the review queue (so the shutdown handler can drain it),
 * calls `createApp`, sweeps stale workspaces, listens and installs the
 * shutdown handlers.
 *
 * Order matters: the body parser lives inside the webhook router (it keeps
 * the raw bytes the signature is computed over), and the error handler must
 * be the last thing registered to see what the routes throw.
 */
export function createApp(config: AppConfig, deps: AppDeps = {}): Express {
    const { logError = logToConsole, ...webhook } = deps;
    const app = express();

    app.get("/health", (_req, res) => {
        res.status(200).json({ status: "ok" });
    });

    app.use("/webhooks/github", createWebhookRouter(config, webhook));

    app.use(
        createErrorHandler({
            secrets: [config.webhookSecret, config.privateKey],
            logError,
        }),
    );

    return app;
}
