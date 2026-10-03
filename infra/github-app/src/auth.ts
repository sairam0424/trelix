import { createAppAuth } from "@octokit/auth-app";
import type { RequestInterface } from "@octokit/types";
import { AppConfig } from "./config.js";

/**
 * One AuthInterface per AppConfig, reused across calls so
 * @octokit/auth-app's own internal cache (an LRU keyed by installation,
 * repository ids and permission set) actually has a chance to hit — creating a
 * fresh createAppAuth() per call would give each call its own empty cache,
 * defeating expiry-aware reuse.
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
 * What an installation token is minted for. Each purpose gets its own token, limited to
 * what that purpose does, because the three run in different places: the checkout token
 * reaches `git` (which runs over an outside author's PR), the review token reaches the
 * `trelix review` child (which also reads it, and the LLM key), and the poster token stays
 * in this process. A token that could do all three would let a compromised git or review
 * child forge Checks or read other code.
 */
export type TokenPurpose = "checkout" | "review" | "poster";

type PermissionName = "checks" | "contents" | "metadata" | "pull_requests";

type PurposePermissions = Readonly<
    Partial<Record<PermissionName, "read" | "write">>
>;

// `metadata: read` is the one permission every installation token carries; it is named so
// that each body says the whole of what the token may do.
//   checkout: `git fetch` of refs/pull/<n>/head (repo-checkout.ts).
//   review:   `trelix review --pr` lists the PR's files, GET /pulls/{n}/files (and, only
//             with --post-comments, which the App never passes, /pulls/{n}).
//   poster:   `checks.create`, the only write this service makes (createCheckRun).
const PURPOSE_PERMISSIONS: Readonly<Record<TokenPurpose, PurposePermissions>> =
    Object.freeze({
        checkout: Object.freeze({ contents: "read", metadata: "read" }),
        review: Object.freeze({ pull_requests: "read", metadata: "read" }),
        poster: Object.freeze({ checks: "write", metadata: "read" }),
    });

/**
 * Mints (or returns a cached, still-valid) installation access token for one purpose,
 * limited to one repository. @octokit/auth-app handles the JWT signing (App ID + private
 * key) -> installation-token exchange and expiry-aware refresh internally — this
 * function's job is to reuse one AuthInterface per config so that caching actually
 * applies across calls, and to never ask for an unscoped token.
 *
 * The library caches by installation, repository ids, repository names and the sorted
 * permission set (`optionsToCacheKey`), so two purposes never share a cached token;
 * tests/auth.test.ts asserts it on the POST bodies.
 *
 * Fails before any request when the repository id or the purpose is not usable: an
 * empty `repository_ids` or `permissions` is not a narrow token to GitHub.
 *
 * `request` is injectable for tests (a fake HTTP transport) — production
 * callers omit it and get @octokit/auth-app's real default.
 */
export async function getInstallationToken(
    config: AppConfig,
    installationId: number,
    repositoryId: number,
    purpose: TokenPurpose,
    request?: RequestInterface,
): Promise<string> {
    if (!Number.isSafeInteger(repositoryId) || repositoryId <= 0) {
        throw new Error(
            "an installation token needs the numeric id of one repository",
        );
    }
    // An own-property test: "constructor" or "__proto__" would otherwise find a
    // value on the object's prototype and be taken for a purpose.
    if (!Object.hasOwn(PURPOSE_PERMISSIONS, purpose)) {
        throw new Error(
            `unknown installation token purpose: ${String(purpose)}`,
        );
    }
    const permissions = PURPOSE_PERMISSIONS[purpose];
    const auth = getAuthInstance(config, request);
    const authentication = await auth({
        type: "installation",
        installationId,
        // New objects on every call: the library sorts `repositoryIds` in place.
        repositoryIds: [repositoryId],
        permissions: { ...permissions },
    });
    return authentication.token;
}

export interface PurposeTokens {
    checkout: string;
    review: string;
    poster: string;
}

/** One token per purpose for one review, all limited to `repositoryId`. */
export async function getPurposeTokens(
    config: AppConfig,
    installationId: number,
    repositoryId: number,
    request?: RequestInterface,
): Promise<PurposeTokens> {
    const [checkout, review, poster] = await Promise.all(
        (["checkout", "review", "poster"] as const).map((purpose) =>
            getInstallationToken(
                config,
                installationId,
                repositoryId,
                purpose,
                request,
            ),
        ),
    );
    return { checkout, review, poster };
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
