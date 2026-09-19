import { generateKeyPairSync, verify as verifySignature } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import { getInstallationToken, getAppJwt } from "../src/auth.js";
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

        const token = await getInstallationToken(config, 999, request);

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

        const first = await getInstallationToken(config, 999, request);
        const second = await getInstallationToken(config, 999, request);

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

        const tokenA = await getInstallationToken(config, 111, request);
        const tokenB = await getInstallationToken(config, 222, request);

        expect(tokenA).toBe("ghs_token_for_111");
        expect(tokenB).toBe("ghs_token_for_222");
        expect(callCount).toBe(2);
    });

    it("does not share cached tokens across two distinct AppConfig objects", async () => {
        const configA = makeConfig();
        const configB = makeConfig();
        const requestA = fakeRequest("ghs_token_a", "2099-01-01T00:00:00Z");
        const requestB = fakeRequest("ghs_token_b", "2099-01-01T00:00:00Z");

        const tokenA = await getInstallationToken(configA, 999, requestA);
        const tokenB = await getInstallationToken(configB, 999, requestB);

        expect(tokenA).toBe("ghs_token_a");
        expect(tokenB).toBe("ghs_token_b");
        expect(requestA).toHaveBeenCalledTimes(1);
        expect(requestB).toHaveBeenCalledTimes(1);
    });
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
