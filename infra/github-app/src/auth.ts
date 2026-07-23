import { createAppAuth } from "@octokit/auth-app";
import type { RequestInterface } from "@octokit/types";
import { AppConfig } from "./config.js";

/**
 * One AuthInterface per AppConfig, reused across calls so
 * @octokit/auth-app's own internal cache (an LRU keyed by installationId)
 * actually has a chance to hit — creating a fresh createAppAuth() per call
 * would give each call its own empty cache, defeating expiry-aware reuse.
 * WeakMap keying means this never outlives the config object it's for and
 * never leaks memory across config reloads.
 */
const authInstances = new WeakMap<AppConfig, ReturnType<typeof createAppAuth>>();

function getAuthInstance(
  config: AppConfig,
  request?: RequestInterface,
): ReturnType<typeof createAppAuth> {
  let auth = authInstances.get(config);
  if (!auth) {
    auth = createAppAuth({ appId: config.appId, privateKey: config.privateKey, request });
    authInstances.set(config, auth);
  }
  return auth;
}

/**
 * Mints (or returns a cached, still-valid) installation access token.
 * @octokit/auth-app handles the JWT signing (App ID + private key) ->
 * installation-token exchange and expiry-aware refresh internally — this
 * function's job is only to reuse one AuthInterface per config so that
 * caching actually applies across calls.
 *
 * `request` is injectable for tests (a fake HTTP transport) — production
 * callers omit it and get @octokit/auth-app's real default.
 */
export async function getInstallationToken(
  config: AppConfig,
  installationId: number,
  request?: RequestInterface,
): Promise<string> {
  const auth = getAuthInstance(config, request);
  const authentication = await auth({ type: "installation", installationId });
  return authentication.token;
}
