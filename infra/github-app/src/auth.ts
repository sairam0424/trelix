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
const authInstances = new WeakMap<
    AppConfig,
    ReturnType<typeof createAppAuth>
>();

function getAuthInstance(
    config: AppConfig,
    request?: RequestInterface,
): ReturnType<typeof createAppAuth> {
    let auth = authInstances.get(config);
    if (!auth) {
        auth = createAppAuth({
            appId: config.appId,
            privateKey: config.privateKey,
            // Omit the key entirely when unset — @octokit/auth-app builds its
            // state via `Object.assign({request: <defaulted>}, options, ...)`,
            // and Object.assign copies a present key regardless of its value.
            // `request: undefined` here would overwrite that default with
            // undefined, crashing every real call (verified live: this was
            // happening on 100% of production webhooks, since only tests ever
            // pass a real `request` override).
            ...(request ? { request } : {}),
        });
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

/**
 * Mints a JWT authenticated as the App itself (App ID + private key,
 * RS256-signed, 10-minute expiry) rather than an installation access
 * token. This is the auth mode the redelivery script needs — GitHub's
 * `/app/hook/deliveries*` endpoints are JWT-only, distinct from every
 * other call in this codebase, which uses an installation token.
 *
 * Reuses the same `authInstances` cache as `getInstallationToken`, but
 * this does NOT mean the JWT itself is cached: `@octokit/auth-app`'s
 * `type: "app"` path (verified empirically — see auth.test.ts) mints a
 * fresh, locally-signed JWT reflecting the current time on every call; no
 * HTTP request and no expiry-aware reuse happens for this auth mode, only
 * for `type: "installation"`.
 */
export async function getAppJwt(
    config: AppConfig,
    request?: RequestInterface,
): Promise<string> {
    const auth = getAuthInstance(config, request);
    const authentication = await auth({ type: "app" });
    return authentication.token;
}
