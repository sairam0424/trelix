import { generateKeyPairSync } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import { getInstallationToken } from "../src/auth.js";
import { AppConfig } from "../src/config.js";

// A real (test-only, never used against GitHub) RSA keypair — @octokit/auth-app
// signs a real JWT with it internally, so a syntactically valid PEM key is
// required even though no real GitHub API call happens in these tests.
const { privateKey } = generateKeyPairSync("rsa", {
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
    reviewRepoPath: ".",
  };
}

/** Fakes @octokit/auth-app's HTTP transport for the installation-token exchange. */
function fakeRequest(token: string, expiresAt: string) {
  return vi.fn(async () => ({
    data: { token, expires_at: expiresAt, permissions: {}, repository_selection: "all" },
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
    const request = vi.fn(async (_route: string, payload: { installation_id: number }) => {
      callCount++;
      return {
        data: {
          token: `ghs_token_for_${payload.installation_id}`,
          expires_at: "2099-01-01T00:00:00Z",
          permissions: {},
          repository_selection: "all",
        },
      };
    }) as never;

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
