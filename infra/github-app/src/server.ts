import { createApp } from "./app.js";
import { loadConfig } from "./config.js";
import { sweepStaleWorkspaces } from "./repo-checkout.js";

const config = loadConfig();
const app = createApp(config);

// Best-effort: removes any trelix-review-* workspace a previous instance
// leaked (e.g. via SIGKILL mid-checkout) before serving any webhook.
void sweepStaleWorkspaces();

app.listen(config.port, () => {
    console.log(`trelix GitHub App listening on port ${config.port}`);
});
