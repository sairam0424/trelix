import { AppConfig } from "./config.js";

/**
 * Installation-token minting — STUBBED in this item (6a: manifest +
 * webhook skeleton). Real implementation (App-ID+private-key JWT ->
 * installation-token exchange, with expiry-aware refresh caching) lands
 * in item 6b. Until then this throws, so the review pipeline visibly
 * fails closed rather than silently running unauthenticated.
 */
export async function getInstallationToken(
  _config: AppConfig,
  _installationId: number,
): Promise<string> {
  throw new Error(
    "getInstallationToken is not yet implemented — see item 6b (GitHub App auth + signature verification).",
  );
}
