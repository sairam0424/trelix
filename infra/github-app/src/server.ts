import { loadAbuseControls } from "./abuse-controls.js";
import { createApp } from "./app.js";
import { loadConfig } from "./config.js";
import { sweepStaleWorkspaces } from "./repo-checkout.js";
import { createReviewQueue } from "./review-intake.js";
import { installShutdownHandlers } from "./shutdown.js";

const config = loadConfig();
// Throws, so the service does not start, when TRELIX_APP_INSTALL_POLICY names no policy.
const controls = loadAbuseControls();
// Built here, not inside the router, so the shutdown handler below can drain it.
const queue = createReviewQueue(config, controls.queue);
const app = createApp(config, { controls, queue });

// Best-effort: removes any trelix-review-* workspace a previous instance
// leaked (e.g. via SIGKILL mid-checkout) before serving any webhook.
void sweepStaleWorkspaces();

const server = app.listen(config.port, () => {
    console.log(`trelix GitHub App listening on port ${config.port}`);
});

installShutdownHandlers({ server, queue, process });
