import { Octokit } from "@octokit/rest";
import { loadConfig } from "../config.js";
import { getAppJwt } from "../auth.js";

// GitHub delivers webhooks with automatic retry for its own transient
// failures, so redelivering something that's still within its natural
// retry window would just race the platform's own logic. This margin
// gives GitHub's own retries a chance to succeed first — only deliveries
// older than this are treated as needing a manual backstop redelivery.
const MIN_AGE_BEFORE_REDELIVER_MS = 15 * 60 * 1000;

function isFailedStatusCode(statusCode: number): boolean {
    return statusCode < 200 || statusCode >= 400;
}

function isOldEnoughToRedeliver(deliveredAt: string): boolean {
    return (
        Date.now() - new Date(deliveredAt).getTime() >=
        MIN_AGE_BEFORE_REDELIVER_MS
    );
}

/**
 * Redelivers any failed, sufficiently-aged webhook delivery for this App —
 * the safety net for GitHub's own "no auto-retry past its own transient
 * window" webhook contract. Uses JWT (App-level) auth: `/app/hook/deliveries*`
 * is JWT-only, distinct from the installation-token auth every other call
 * in this codebase uses.
 */
export async function redeliverFailedDeliveries(
    octokit: Octokit,
): Promise<number> {
    const { data: deliveries } = await octokit.rest.apps.listWebhookDeliveries({
        per_page: 100,
    });

    const candidates = deliveries.filter(
        (delivery) =>
            isFailedStatusCode(delivery.status_code) &&
            isOldEnoughToRedeliver(delivery.delivered_at),
    );

    for (const delivery of candidates) {
        await octokit.rest.apps.redeliverWebhookDelivery({
            delivery_id: delivery.id,
        });
        console.log(
            `[redeliver] redelivered delivery ${delivery.id} (guid ${delivery.guid}, ` +
                `event ${delivery.event}, original status ${delivery.status_code})`,
        );
    }

    console.log(
        `[redeliver] checked ${deliveries.length} delivery(ies), redelivered ${candidates.length}`,
    );
    return candidates.length;
}

async function main() {
    const config = loadConfig();
    const octokit = new Octokit({ auth: await getAppJwt(config) });
    await redeliverFailedDeliveries(octokit);
}

// Only run main() when this module is the actual entry point (not when
// imported by a test for redeliverFailedDeliveries).
if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch((err) => {
        console.error("[redeliver] failed:", err);
        process.exitCode = 1;
    });
}
