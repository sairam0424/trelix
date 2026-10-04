import { generateKeyPairSync, verify as verifySignature } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import {
    getAppJwt,
    getInstallationToken,
    getPurposeTokens,
} from "../src/auth.js";
import { AppConfig } from "../src/config.js";

// A real (test-only, never used against GitHub) RSA keypair — @octokit/auth-app
// signs a real JWT with it internally, so a syntactically valid PEM key is
// required even though no real GitHub API call happens in these tests.
// getAppJwt's own tests also verify the signed JWT against `publicKey`.
const { privateKey, publicKey } = generateKeyPairSync("rsa", {
    modulusLength: 2048,
    publicKeyEncoding: { type: "spki", format: "pem" },
    privateKeyEncoding: { type: "pkcs1", format: "pem" },
});

function makeConfig(): AppConfig {
    // A fresh object each call, since getInstallationToken's caching is
    // keyed by AppConfig identity (WeakMap) — reusing the same object
    // across tests would leak cache state between them.
    return {
        appId: "12345",
        privateKey,
        webhookSecret: "fake",
        port: 0,
    };
}

/** Fakes @octokit/auth-app's HTTP transport for the installation-token exchange. */
function fakeRequest(token: string, expiresAt: string) {
    return vi.fn(async () => ({
        data: {
            token,
            expires_at: expiresAt,
            permissions: {},
            repository_selection: "all",
        },
    })) as never;
}

describe("getInstallationToken", () => {
    it("returns the installation token from the (mocked) GitHub API", async () => {
        const config = makeConfig();
        const request = fakeRequest("ghs_faketoken1", "2099-01-01T00:00:00Z");

        const token = await getInstallationToken(
            config,
            999,
            4242,
            "checkout",
            request,
        );

        expect(token).toBe("ghs_faketoken1");
        expect(request).toHaveBeenCalledTimes(1);
        expect(request).toHaveBeenCalledWith(
            "POST /app/installations/{installation_id}/access_tokens",
            expect.objectContaining({ installation_id: 999 }),
        );
    });

    it("reuses the cached token on a second call for the same config+installation, making no second request", async () => {
        const config = makeConfig();
        const request = fakeRequest("ghs_cached_token", "2099-01-01T00:00:00Z");

        const first = await getInstallationToken(
            config,
            999,
            4242,
            "checkout",
            request,
        );
        const second = await getInstallationToken(
            config,
            999,
            4242,
            "checkout",
            request,
        );

        expect(first).toBe("ghs_cached_token");
        expect(second).toBe("ghs_cached_token");
        expect(request).toHaveBeenCalledTimes(1); // cache hit on the second call
    });

    it("mints a separate token per distinct installationId (no cross-installation cache collision)", async () => {
        const config = makeConfig();
        let callCount = 0;
        const request = vi.fn(
            async (_route: string, payload: { installation_id: number }) => {
                callCount++;
                return {
                    data: {
                        token: `ghs_token_for_${payload.installation_id}`,
                        expires_at: "2099-01-01T00:00:00Z",
                        permissions: {},
                        repository_selection: "all",
                    },
                };
            },
        ) as never;

        const tokenA = await getInstallationToken(
            config,
            111,
            4242,
            "checkout",
            request,
        );
        const tokenB = await getInstallationToken(
            config,
            222,
            4242,
            "checkout",
            request,
        );

        expect(tokenA).toBe("ghs_token_for_111");
        expect(tokenB).toBe("ghs_token_for_222");
        expect(callCount).toBe(2);
    });

    it("does not share cached tokens across two distinct AppConfig objects", async () => {
        const configA = makeConfig();
        const configB = makeConfig();
        const requestA = fakeRequest("ghs_token_a", "2099-01-01T00:00:00Z");
        const requestB = fakeRequest("ghs_token_b", "2099-01-01T00:00:00Z");

        const tokenA = await getInstallationToken(
            configA,
            999,
            4242,
            "checkout",
            requestA,
        );
        const tokenB = await getInstallationToken(
            configB,
            999,
            4242,
            "checkout",
            requestB,
        );

        expect(tokenA).toBe("ghs_token_a");
        expect(tokenB).toBe("ghs_token_b");
        expect(requestA).toHaveBeenCalledTimes(1);
        expect(requestB).toHaveBeenCalledTimes(1);
    });

    // Regression test — found live against real GitHub, not in review: every
    // other test above injects a fake `request`, so none of them exercise
    // the actual production call signature (no third argument at all).
    // createAppAuth({..., request}) with `request === undefined` still sets
    // the *key* `request: undefined` on the options object; @octokit/auth-app
    // builds its state via `Object.assign({request: <defaulted>}, options)`,
    // and Object.assign copies a present key regardless of its value — so
    // the explicit `undefined` silently overwrote the library's own default
    // transport, and every real webhook crashed with "request is not a
    // function" before ever reaching GitHub. Faking `globalThis.fetch`
    // (the real underlying transport @octokit/request's fetch-wrapper uses)
    // rather than injecting a `request` override is what makes this test
    // actually exercise the previously-broken path.
    it("mints a token via the real @octokit/request transport when no request override is passed at all", async () => {
        const config = makeConfig();
        const originalFetch = globalThis.fetch;
        const canaryTokenValue = "unittest-transport-canary-value";
        let capturedUrl: string | undefined;
        globalThis.fetch = vi.fn(async (url: string) => {
            capturedUrl = url;
            return new Response(
                JSON.stringify({
                    token: canaryTokenValue,
                    expires_at: "2099-01-01T00:00:00Z",
                    permissions: {},
                    repository_selection: "all",
                }),
                {
                    status: 201,
                    headers: { "content-type": "application/json" },
                },
            );
        }) as never;

        try {
            const token = await getInstallationToken(
                config,
                999,
                4242,
                "checkout",
            );
            expect(token).toBe(canaryTokenValue);
            expect(capturedUrl).toContain(
                "/app/installations/999/access_tokens",
            );
        } finally {
            globalThis.fetch = originalFetch;
        }
    });
});

/** One access_tokens request as the (fake) transport saw it, and what it was answered with. */
interface IssuedToken {
    token: string;
    body: Record<string, unknown>;
}

/**
 * A fake transport for the installation-token exchange that answers every request with a
 * distinct placeholder of its own and keeps the request, so a test can say which body got
 * which answer.
 */
function recordingRequest() {
    const issued: IssuedToken[] = [];
    const request = vi.fn(
        async (_route: string, payload: Record<string, unknown>) => {
            const answer = `canary-${issued.length + 1}`;
            issued.push({ token: answer, body: payload });
            return {
                data: {
                    token: answer,
                    expires_at: "2099-01-01T00:00:00Z",
                    permissions: {},
                    repository_selection: "selected",
                },
            };
        },
    ) as never;
    return { request, issued };
}

/** What a request asks GitHub for: the body without the App's own authorization header and media type. */
function scopeOf(issued: IssuedToken) {
    const { headers: _headers, mediaType: _mediaType, ...scope } = issued.body;
    return scope;
}

describe("installation tokens are scoped per purpose", () => {
    it.each([
        [
            "checkout",
            {
                installation_id: 999,
                repository_ids: [4242],
                permissions: { contents: "read", metadata: "read" },
            },
        ],
        [
            "review",
            {
                installation_id: 999,
                repository_ids: [4242],
                permissions: { pull_requests: "read", metadata: "read" },
            },
        ],
        [
            "poster",
            {
                installation_id: 999,
                repository_ids: [4242],
                permissions: { checks: "write", metadata: "read" },
            },
        ],
    ] as const)(
        "asks GitHub for the %s token with exactly one repository and exactly its permissions",
        async (purpose, expectedScope) => {
            const { request, issued } = recordingRequest();

            const got = await getInstallationToken(
                makeConfig(),
                999,
                4242,
                purpose,
                request,
            );

            expect(issued).toHaveLength(1);
            expect(got).toBe(issued[0].token);
            // toStrictEqual: an extra permission, a second repository or a
            // `repositories`/`permissions: {}` key would each fail here.
            expect(scopeOf(issued[0])).toStrictEqual(expectedScope);
            // The exchange is authenticated as the App itself (a JWT).
            expect(
                (issued[0].body.headers as { authorization: string })
                    .authorization,
            ).toMatch(/^bearer \S+\.\S+\.\S+$/);
        },
    );

    it("sends the scope on the wire: repository_ids and permissions in the JSON body of the real transport", async () => {
        const config = makeConfig();
        const originalFetch = globalThis.fetch;
        const sent: Array<{ url: string; body: unknown }> = [];
        globalThis.fetch = vi.fn(async (url: string, init?: RequestInit) => {
            sent.push({ url, body: JSON.parse(String(init?.body)) });
            return new Response(
                JSON.stringify({
                    token: "test-k",
                    expires_at: "2099-01-01T00:00:00Z",
                    permissions: {},
                    repository_selection: "selected",
                }),
                {
                    status: 201,
                    headers: { "content-type": "application/json" },
                },
            );
        }) as never;

        try {
            await getInstallationToken(config, 999, 4242, "poster");
        } finally {
            globalThis.fetch = originalFetch;
        }

        expect(sent).toHaveLength(1);
        expect(sent[0].url).toContain("/app/installations/999/access_tokens");
        expect(sent[0].body).toStrictEqual({
            repository_ids: [4242],
            permissions: { checks: "write", metadata: "read" },
        });
    });

    it("gives each purpose its own token, and never hands one purpose the token minted for another", async () => {
        const config = makeConfig();
        const { request, issued } = recordingRequest();

        const first = await getPurposeTokens(config, 999, 4242, request);

        expect(issued).toHaveLength(3);
        expect(new Set([first.checkout, first.review, first.poster]).size).toBe(
            3,
        );
        const permissionsOf = (answer: string) =>
            scopeOf(issued.find((i) => i.token === answer) as IssuedToken)
                .permissions;
        expect(permissionsOf(first.checkout)).toStrictEqual({
            contents: "read",
            metadata: "read",
        });
        expect(permissionsOf(first.review)).toStrictEqual({
            pull_requests: "read",
            metadata: "read",
        });
        expect(permissionsOf(first.poster)).toStrictEqual({
            checks: "write",
            metadata: "read",
        });

        // The cache answers the second round, purpose by purpose: no new request, and
        // each purpose gets its own token back again.
        const second = await getPurposeTokens(config, 999, 4242, request);
        expect(issued).toHaveLength(3);
        expect(second).toStrictEqual(first);
    });

    it("does not reuse a token minted for one repository for another repository", async () => {
        const config = makeConfig();
        const { request, issued } = recordingRequest();

        const forFirst = await getInstallationToken(
            config,
            999,
            4242,
            "checkout",
            request,
        );
        const forSecond = await getInstallationToken(
            config,
            999,
            4343,
            "checkout",
            request,
        );

        expect(forSecond).not.toBe(forFirst);
        expect(issued.map((i) => scopeOf(i).repository_ids)).toStrictEqual([
            [4242],
            [4343],
        ]);
    });

    it.each([
        ["zero", 0],
        ["negative", -1],
        ["fractional", 1.5],
        ["NaN", Number.NaN],
        ["infinite", Number.POSITIVE_INFINITY],
        ["past the safe-integer range", 2 ** 53],
    ])(
        "refuses a repository id that is %s, without asking GitHub for anything",
        async (_name, repositoryId) => {
            const { request, issued } = recordingRequest();

            await expect(
                getInstallationToken(
                    makeConfig(),
                    999,
                    repositoryId,
                    "checkout",
                    request,
                ),
            ).rejects.toThrow(
                "an installation token needs the numeric id of one repository",
            );

            expect(issued).toHaveLength(0);
        },
    );

    // "constructor" and "toString" are found on any object's prototype: an own-property
    // test is what keeps them from being taken for a purpose.
    it.each(["admin", "", "constructor", "toString", "hasOwnProperty"])(
        "refuses the purpose %j, which it does not know, without asking GitHub for anything",
        async (purpose) => {
            const { request, issued } = recordingRequest();

            await expect(
                getInstallationToken(
                    makeConfig(),
                    999,
                    4242,
                    purpose as never,
                    request,
                ),
            ).rejects.toThrow(`unknown installation token purpose: ${purpose}`);

            expect(issued).toHaveLength(0);
        },
    );
});

describe("getAppJwt", () => {
    it("mints a JWT whose signature verifies against the App's public key (pure local RS256 signing, no HTTP call)", async () => {
        const config = makeConfig();

        const jwt = await getAppJwt(config);
        const parts = jwt.split(".");
        expect(parts).toHaveLength(3);
        const [headerB64, payloadB64, signatureB64] = parts;

        const header = JSON.parse(
            Buffer.from(headerB64, "base64url").toString("utf8"),
        );
        const payload = JSON.parse(
            Buffer.from(payloadB64, "base64url").toString("utf8"),
        );
        expect(header.alg).toBe("RS256");
        expect(payload.iss).toBe(config.appId);

        const isValid = verifySignature(
            "RSA-SHA256",
            Buffer.from(`${headerB64}.${payloadB64}`),
            publicKey,
            Buffer.from(signatureB64, "base64url"),
        );
        expect(isValid).toBe(true);
    });

    it(
        "mints a fresh JWT reflecting the current time on every call — unlike installation " +
            "tokens, @octokit/auth-app never caches a 'type: app' JWT (verified empirically: " +
            "getAppAuthentication() always re-signs rather than consulting the shared cache)",
        async () => {
            const config = makeConfig();

            vi.useFakeTimers();
            try {
                vi.setSystemTime(new Date("2099-01-01T00:00:00Z"));
                const first = await getAppJwt(config);

                // Past the JWT's own 10-minute expiry — if this call reused a
                // cached token, it would be a stale, already-expired JWT.
                vi.setSystemTime(new Date("2099-01-01T00:20:00Z"));
                const second = await getAppJwt(config);

                expect(first).not.toBe(second);
            } finally {
                vi.useRealTimers();
            }
        },
    );

    it("mints a JWT authenticated as the App, not any specific installation (no installationId in the payload)", async () => {
        const config = makeConfig();

        const jwt = await getAppJwt(config);
        const payload = JSON.parse(
            Buffer.from(jwt.split(".")[1], "base64url").toString("utf8"),
        );

        expect(payload).not.toHaveProperty("installationId");
        expect(payload).not.toHaveProperty("installation_id");
    });
});
