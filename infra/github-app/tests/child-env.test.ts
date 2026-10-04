import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
    buildGitChildEnv,
    buildIndexChildEnv,
    buildReviewChildEnv,
} from "../src/child-env.js";

/** Distinct, greppable stand-in for the value of an environment variable. */
const valueOf = (name: string) => `host-value-of-${name}`;

function hostEnv(names: readonly string[]): NodeJS.ProcessEnv {
    return Object.fromEntries(names.map((name) => [name, valueOf(name)]));
}

const GIT_CREDENTIAL = "installation-credential-for-git";
const REVIEW_CREDENTIAL = "installation-credential-for-review";
const HOST_GITHUB_CREDENTIAL = "credential-the-host-happens-to-hold";

// Held by the App's own process: its credentials, the platform's tokens, a
// GitHub token, and a name nobody listed. None may reach any child.
const FORBIDDEN_NAMES = [
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_WEBHOOK_SECRET",
    "GITHUB_APP_ID",
    "GITHUB_TOKEN",
    "RAILWAY_TOKEN",
    "RAILWAY_PROJECT_ID",
    "RAILWAY_ENVIRONMENT_ID",
    "RAILWAY_PUBLIC_DOMAIN",
    "SOME_UNLISTED_SECRET",
];

// Every non-TRELIX_ name trelix's config reads (CONFIG_NON_PREFIXED_ENV in
// tests/_env_isolation.py) plus three names a provider SDK reads for itself. The full SDK
// list is pinned against INSTALLED_SDK_PROVIDER_ENV by the Python contract test; the
// credential chains an operator on role-based AWS, Vertex or a gateway relies on follow.
const PROVIDER_NAMES = [
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_DEFAULT_REGION",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AZURE_API_KEY",
    "AZURE_API_VERSION",
    "AZURE_CHAT_MODEL",
    "AZURE_EMBEDDINGS_MODEL",
    "AZURE_ENDPOINT",
    "COHERE_API_KEY",
    "COHERE_ENDPOINT",
    "COHERE_MODEL_RERANK",
    "GOOGLE_API_KEY",
    "GOOGLE_CLOUD_LOCATION",
    "GOOGLE_CLOUD_PROJECT",
    "LANCE_TABLE",
    "LANCE_URI",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_SERVICE_NAME",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "QDRANT_PREFER_GRPC",
    "QDRANT_QUANTIZATION",
    "QDRANT_QUANTIZATION_RESCORE",
    "QDRANT_TIMEOUT",
    "QDRANT_URL",
    "VOYAGE_API_KEY",
];

const CREDENTIAL_CHAIN_NAMES = [
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_ROLE_ARN",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "GOOGLE_APPLICATION_CREDENTIALS",
];

// Not passed to any child: proxy and CA-bundle variables (a deployment that needs one
// adds it to child-env.ts), and provider-SDK names withheld on purpose because they verify
// a webhook, administer an organisation, or belong to a service trelix has no client for.
const NOT_PASSED_NAMES = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "AWS_CA_BUNDLE",
    "ANTHROPIC_WEBHOOK_SIGNING_KEY",
    "OPENAI_ADMIN_KEY",
    "OPENAI_WEBHOOK_SECRET",
    "AZURE_CLIENT_SECRET",
    "AWS_S3_ENCRYPTION_KEY_ID",
];

const gitInput = {
    token: GIT_CREDENTIAL,
    askpassPath: "/aux/askpass.sh",
    homeDir: "/aux/home",
};

type Builder = (base: NodeJS.ProcessEnv) => NodeJS.ProcessEnv;
const BUILDERS: Array<[string, Builder]> = [
    ["git", (base) => buildGitChildEnv(gitInput, base)],
    ["index", (base) => buildIndexChildEnv(base)],
    ["review", (base) => buildReviewChildEnv(REVIEW_CREDENTIAL, base)],
];

describe.each(BUILDERS)("the %s child env", (_kind, build) => {
    it("never carries the App's credentials, the platform's variables, a GitHub token from the host or an unlisted name", () => {
        const env = build({
            ...hostEnv(FORBIDDEN_NAMES),
            PATH: "/usr/bin",
        });

        for (const name of FORBIDDEN_NAMES) {
            expect(env[name]).not.toBe(valueOf(name));
        }
        expect(
            Object.keys(env).filter((name) => name.startsWith("RAILWAY_")),
        ).toEqual([]);
        expect(env).not.toHaveProperty("GITHUB_APP_PRIVATE_KEY");
        expect(env).not.toHaveProperty("GITHUB_WEBHOOK_SECRET");
        expect(env).not.toHaveProperty("SOME_UNLISTED_SECRET");
    });

    it("passes no proxy or CA-bundle variable and none of the provider names withheld on purpose", () => {
        const env = build({ ...hostEnv(NOT_PASSED_NAMES), PATH: "/usr/bin" });

        for (const name of NOT_PASSED_NAMES) {
            expect(env).not.toHaveProperty(name);
        }
    });

    it("returns a new object and leaves its input untouched", () => {
        const base = {
            ...hostEnv(FORBIDDEN_NAMES),
            PATH: "/usr/bin",
            TRELIX_WALKER_FOLLOW_SYMLINKS: "true",
        };
        const snapshot = { ...base };

        const env = build(base);

        expect(env).not.toBe(base);
        expect(base).toEqual(snapshot);
    });

    it("skips a name the host has not set rather than listing it as undefined", () => {
        const env = build({ PATH: "/usr/bin" });

        expect(Object.values(env)).not.toContain(undefined);
        expect(env).not.toHaveProperty("LANG");
    });
});

// ---------------------------------------------------------------------------
// Closed world. The tests above feed the builders the names we thought of; this one
// feeds them a host full of secrets nobody listed and requires every name in the output
// to come from the allow-list arrays in child-env.ts or from the builder's own fixed
// names. A name picked inline (`pick(base, ["DATABASE_URL"])`) is in neither.
// ---------------------------------------------------------------------------

/** The `const NAME: readonly string[] = [...]` literals of child-env.ts, by name. */
function allowLists(): Record<string, string[]> {
    const source = readFileSync(
        fileURLToPath(new URL("../src/child-env.ts", import.meta.url)),
        "utf8",
    )
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/\/\/[^\n]*/g, "");
    const lists: Record<string, string[]> = {};
    for (const [, name, body] of source.matchAll(
        /\bconst\s+([A-Z][A-Z0-9_]*)\s*:\s*readonly\s+string\[\]\s*=\s*\[([\s\S]*?)\]\s*;/g,
    )) {
        lists[name] = [...body.matchAll(/"([^"\\]*)"/g)].map((m) => m[1]);
    }
    return lists;
}

// Secret-looking names a deployment might hold for its own reasons. None is on any list.
const ARBITRARY_SECRET_NAMES = [
    "RAILWAY_API_TOKEN",
    "DATABASE_URL",
    "REDIS_URL",
    "STRIPE_SECRET_KEY",
    "AWS_SECRET_ACCESS_KEY_ID_2",
    "SOME_PASSWORD",
    "MY_API_KEY",
    "GH_TOKEN",
    "NPM_TOKEN",
    "SENTRY_DSN",
    "SLACK_BOT_TOKEN",
    "SENDGRID_API_KEY",
    "POSTGRES_PASSWORD",
    "JWT_SECRET",
    "SESSION_SECRET",
    "ENCRYPTION_KEY",
    "PRIVATE_KEY",
    "CLOUDFLARE_API_TOKEN",
    "VERCEL_TOKEN",
    "DOCKER_PASSWORD",
    "TWILIO_AUTH_TOKEN",
    "SSH_AUTH_SOCK",
];

// What each builder sets itself, whatever the host holds.
const GIT_FIXED_NAMES = [
    "HOME",
    "XDG_CONFIG_HOME",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_NOSYSTEM",
    "GIT_TERMINAL_PROMPT",
    "GIT_ALLOW_PROTOCOL",
    "GIT_ASKPASS",
    "TRELIX_GIT_TOKEN",
];
const TRELIX_FIXED_NAMES = [
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_KEY_0",
    "GIT_CONFIG_VALUE_0",
    "TRELIX_WALKER_FOLLOW_SYMLINKS",
];

// Names a builder is expected to pass through when the host has them, so the host below
// is not an empty one that would make "nothing unexpected" true by itself.
const PASSED_THROUGH_NAMES = [
    "PATH",
    "LANG",
    "HOME",
    "XDG_CONFIG_HOME",
    "AZURE_API_KEY",
    "AWS_ROLE_ARN",
    "TRELIX_LLM_PROVIDER",
];

describe("the allow-lists are extracted from child-env.ts", () => {
    it("finds the five arrays, none of them empty", () => {
        const lists = allowLists();

        expect(Object.keys(lists).sort()).toEqual([
            "CONFIG_PROVIDER_ENV",
            "GIT_INHERITED_ENV",
            "SDK_PROVIDER_ENV",
            "TRELIX_ENV_PREFIXES",
            "TRELIX_INHERITED_ENV",
        ]);
        for (const [name, entries] of Object.entries(lists)) {
            expect(entries.length, name).toBeGreaterThan(0);
        }
    });
});

describe("closed world: a host full of unlisted secrets", () => {
    const lists = allowLists();
    const trelixNames = new Set([
        ...lists.TRELIX_INHERITED_ENV,
        ...lists.CONFIG_PROVIDER_ENV,
        ...lists.SDK_PROVIDER_ENV,
        ...TRELIX_FIXED_NAMES,
    ]);
    const isTrelixOwned = (name: string) =>
        name !== "TRELIX_GIT_TOKEN" &&
        lists.TRELIX_ENV_PREFIXES.some((prefix) => name.startsWith(prefix));
    const CLOSED_WORLD: Array<[string, Builder, (name: string) => boolean]> = [
        [
            "git",
            (base) => buildGitChildEnv(gitInput, base),
            (name) =>
                lists.GIT_INHERITED_ENV.includes(name) ||
                GIT_FIXED_NAMES.includes(name),
        ],
        [
            "index",
            (base) => buildIndexChildEnv(base),
            (name) => trelixNames.has(name) || isTrelixOwned(name),
        ],
        [
            "review",
            (base) => buildReviewChildEnv(REVIEW_CREDENTIAL, base),
            (name) =>
                trelixNames.has(name) ||
                isTrelixOwned(name) ||
                name === "GITHUB_TOKEN",
        ],
    ];

    const host = hostEnv([
        ...FORBIDDEN_NAMES,
        ...ARBITRARY_SECRET_NAMES,
        ...PASSED_THROUGH_NAMES,
        "TRELIX_GIT_TOKEN",
    ]);

    it("holds at least twenty arbitrary secret-looking names besides the forbidden ones", () => {
        expect(ARBITRARY_SECRET_NAMES.length).toBeGreaterThanOrEqual(20);
        expect(new Set(ARBITRARY_SECRET_NAMES).size).toBe(
            ARBITRARY_SECRET_NAMES.length,
        );
        for (const name of ARBITRARY_SECRET_NAMES) {
            expect(Object.values(lists).flat()).not.toContain(name);
        }
    });

    it.each(CLOSED_WORLD)(
        "the %s child env has no name outside its allow-list and its own fixed names",
        (kind, build, isAllowed) => {
            const env = build(host);

            expect(Object.keys(env).filter((name) => !isAllowed(name))).toEqual(
                [],
            );
            // Control: the host really flowed through, so the check is not vacuous.
            expect(env.PATH).toBe(valueOf("PATH"));
            if (kind !== "git") {
                expect(env).toMatchObject({
                    AZURE_API_KEY: valueOf("AZURE_API_KEY"),
                    AWS_ROLE_ARN: valueOf("AWS_ROLE_ARN"),
                    TRELIX_LLM_PROVIDER: valueOf("TRELIX_LLM_PROVIDER"),
                });
            }
        },
    );

    it("the check itself flags a name that is not allowed", () => {
        const [, , isAllowed] = CLOSED_WORLD[1];

        expect(isAllowed("DATABASE_URL")).toBe(false);
        expect(isAllowed("TRELIX_GIT_TOKEN")).toBe(false);
        expect(isAllowed("TRELIX_SOMETHING_NEW")).toBe(true);
        expect(isAllowed("AWS_ROLE_ARN")).toBe(true);
    });
});

describe("buildGitChildEnv", () => {
    it("is exactly PATH, LANG, an empty HOME and the isolation set, whatever the host holds", () => {
        const base = {
            ...hostEnv(FORBIDDEN_NAMES),
            ...hostEnv(PROVIDER_NAMES),
            ...hostEnv(["TRELIX_LLM_PROVIDER", "TRELIX_GIT_TOKEN"]),
            PATH: "/usr/bin:/bin",
            LANG: "C.UTF-8",
            HOME: "/home/trelix",
            XDG_CONFIG_HOME: "/home/trelix/.config",
        };

        const env = buildGitChildEnv(gitInput, base);

        expect(env).toEqual({
            PATH: "/usr/bin:/bin",
            LANG: "C.UTF-8",
            HOME: "/aux/home",
            XDG_CONFIG_HOME: "/aux/home",
            GIT_CONFIG_GLOBAL: "/dev/null",
            GIT_CONFIG_SYSTEM: "/dev/null",
            GIT_CONFIG_NOSYSTEM: "1",
            GIT_TERMINAL_PROMPT: "0",
            GIT_ALLOW_PROTOCOL: "https",
            GIT_ASKPASS: "/aux/askpass.sh",
            TRELIX_GIT_TOKEN: GIT_CREDENTIAL,
        });
    });

    it("takes the allowed protocol from its parameter and never from the host environment", () => {
        const hostile = {
            GIT_ALLOW_PROTOCOL: "file:ssh:http",
            GIT_CONFIG_GLOBAL: "/home/trelix/.gitconfig",
            GIT_CONFIG_SYSTEM: "/etc/gitconfig",
            GIT_CONFIG_NOSYSTEM: "0",
            GIT_TERMINAL_PROMPT: "1",
            GIT_ASKPASS: "/usr/bin/evil",
        };

        const byDefault = buildGitChildEnv(gitInput, hostile);
        const overridden = buildGitChildEnv(
            { ...gitInput, allowProtocol: "file" },
            hostile,
        );

        expect(byDefault).toMatchObject({
            GIT_ALLOW_PROTOCOL: "https",
            GIT_CONFIG_GLOBAL: "/dev/null",
            GIT_CONFIG_SYSTEM: "/dev/null",
            GIT_CONFIG_NOSYSTEM: "1",
            GIT_TERMINAL_PROMPT: "0",
            GIT_ASKPASS: "/aux/askpass.sh",
        });
        expect(overridden.GIT_ALLOW_PROTOCOL).toBe("file");
    });
});

describe.each([
    ["index", (base: NodeJS.ProcessEnv) => buildIndexChildEnv(base)],
    [
        "review",
        (base: NodeJS.ProcessEnv) =>
            buildReviewChildEnv(REVIEW_CREDENTIAL, base),
    ],
])("the %s child env for trelix", (kind, build) => {
    it.each(PROVIDER_NAMES)("keeps the provider name %s", (name) => {
        const env = build(hostEnv([name]));

        expect(env[name]).toBe(valueOf(name));
    });

    it.each(CREDENTIAL_CHAIN_NAMES)(
        "keeps the credential-chain name %s",
        (name) => {
            const env = build(hostEnv([name]));

            expect(env[name]).toBe(valueOf(name));
        },
    );

    it("keeps PATH, LANG, HOME and XDG_CONFIG_HOME", () => {
        const env = build({
            PATH: "/usr/bin",
            LANG: "C.UTF-8",
            HOME: "/home/trelix",
            XDG_CONFIG_HOME: "/home/trelix/.config",
        });

        expect(env).toMatchObject({
            PATH: "/usr/bin",
            LANG: "C.UTF-8",
            HOME: "/home/trelix",
            XDG_CONFIG_HOME: "/home/trelix/.config",
        });
    });

    it("forces safe.bareRepository=explicit onto the git calls trelix makes itself, whatever the host passes", () => {
        const expected = {
            GIT_CONFIG_COUNT: "1",
            GIT_CONFIG_KEY_0: "safe.bareRepository",
            GIT_CONFIG_VALUE_0: "explicit",
        };

        expect(build({})).toMatchObject(expected);
        expect(
            build({
                GIT_CONFIG_COUNT: "0",
                GIT_CONFIG_KEY_0: "core.fsmonitor",
                GIT_CONFIG_VALUE_0: "/usr/bin/evil",
            }),
        ).toMatchObject(expected);
    });

    it("keeps every TRELIX_ name and forces the walker symlink flag off", () => {
        const env = build({
            TRELIX_LLM_PROVIDER: "azure",
            TRELIX_RETRIEVAL_FLARE: "1",
            TRELIX_WALKER_FOLLOW_SYMLINKS: "true",
        });

        expect(env).toMatchObject({
            TRELIX_LLM_PROVIDER: "azure",
            TRELIX_RETRIEVAL_FLARE: "1",
            TRELIX_WALKER_FOLLOW_SYMLINKS: "false",
        });
        expect(build({})).toMatchObject({
            TRELIX_WALKER_FOLLOW_SYMLINKS: "false",
        });
    });

    it("withholds the git token even though it starts with TRELIX_", () => {
        const env = build({ TRELIX_GIT_TOKEN: HOST_GITHUB_CREDENTIAL });

        expect(env).not.toHaveProperty("TRELIX_GIT_TOKEN");
    });

    it("withholds the service's own TRELIX_APP_ settings but not a neighbour", () => {
        const env = build({
            TRELIX_APP_ALLOWED_ACCOUNTS: "operator-login",
            TRELIX_APP_ALLOWED_INSTALLATIONS: "12345",
            TRELIX_APP_INSTALL_POLICY: "allowlist",
            TRELIX_APP_REVIEWS_ENABLED: "true",
            TRELIX_APP_QUEUE_CAPACITY: "20",
            TRELIX_APP_CONCURRENCY_PER_INSTALLATION: "1",
            TRELIX_APPLE: "1",
            TRELIX_APP: "1",
            TRELIX_LLM_PROVIDER: "azure",
        });

        expect(
            Object.keys(env).filter((name) => name.startsWith("TRELIX_APP_")),
        ).toEqual([]);
        expect(env).toMatchObject({
            TRELIX_APPLE: "1",
            TRELIX_APP: "1",
            TRELIX_LLM_PROVIDER: "azure",
        });
    });

    it("matches names exactly: a lowercase or embedded spelling is withheld", () => {
        const env = build({
            trelix_llm_provider: "azure",
            XTRELIX_FOO: "1",
            MY_TRELIX_FOO: "1",
            azure_endpoint: "lowercase",
            path: "/usr/bin",
        });

        const forced = [
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_KEY_0",
            "GIT_CONFIG_VALUE_0",
            "TRELIX_WALKER_FOLLOW_SYMLINKS",
        ];
        const expected =
            kind === "review" ? ["GITHUB_TOKEN", ...forced] : forced;
        expect(Object.keys(env).sort()).toEqual(expected);
    });
});

describe("GITHUB_TOKEN", () => {
    it("is given to the review child only, from the explicit argument and not from the host", () => {
        const base = { GITHUB_TOKEN: HOST_GITHUB_CREDENTIAL };

        expect(buildReviewChildEnv(REVIEW_CREDENTIAL, base).GITHUB_TOKEN).toBe(
            REVIEW_CREDENTIAL,
        );
        expect(buildIndexChildEnv(base)).not.toHaveProperty("GITHUB_TOKEN");
        expect(buildGitChildEnv(gitInput, base)).not.toHaveProperty(
            "GITHUB_TOKEN",
        );
    });

    it("reaches the review child even when the host has none", () => {
        expect(buildReviewChildEnv(REVIEW_CREDENTIAL, {}).GITHUB_TOKEN).toBe(
            REVIEW_CREDENTIAL,
        );
        expect(buildIndexChildEnv({})).not.toHaveProperty("GITHUB_TOKEN");
    });
});

describe("TRELIX_REVIEW_OUTCOME_FILE", () => {
    const OUTCOME_PATH = "/private/outcome-dir/outcome.json";

    it("is given to the review child, and only when the caller passes a path", () => {
        expect(
            buildReviewChildEnv(REVIEW_CREDENTIAL, {}, OUTCOME_PATH),
        ).toMatchObject({ TRELIX_REVIEW_OUTCOME_FILE: OUTCOME_PATH });
        expect(buildReviewChildEnv(REVIEW_CREDENTIAL, {})).not.toHaveProperty(
            "TRELIX_REVIEW_OUTCOME_FILE",
        );
    });

    it("is the caller's path even when the host sets another: the App reads only the one it chose", () => {
        const base = {
            TRELIX_REVIEW_OUTCOME_FILE: "/tmp/host-chose-this.json",
        };

        const env = buildReviewChildEnv(REVIEW_CREDENTIAL, base, OUTCOME_PATH);

        expect(env.TRELIX_REVIEW_OUTCOME_FILE).toBe(OUTCOME_PATH);
    });

    it("adds exactly one name to the review child env and changes nothing else", () => {
        const base = hostEnv(["PATH", "TRELIX_LLM_PROVIDER", "AZURE_API_KEY"]);

        const without = buildReviewChildEnv(REVIEW_CREDENTIAL, base);
        const withPath = buildReviewChildEnv(
            REVIEW_CREDENTIAL,
            base,
            OUTCOME_PATH,
        );

        expect({ ...withPath, TRELIX_REVIEW_OUTCOME_FILE: undefined }).toEqual({
            ...without,
            TRELIX_REVIEW_OUTCOME_FILE: undefined,
        });
    });

    it("is not given to the index child or the git child", () => {
        const base = { TRELIX_LLM_PROVIDER: "azure" };

        expect(buildIndexChildEnv(base)).not.toHaveProperty(
            "TRELIX_REVIEW_OUTCOME_FILE",
        );
        expect(buildGitChildEnv(gitInput, base)).not.toHaveProperty(
            "TRELIX_REVIEW_OUTCOME_FILE",
        );
    });
});
