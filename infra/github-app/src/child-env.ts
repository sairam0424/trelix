/**
 * Environments for the child processes the App spawns.
 *
 * Every child runs over content an outside PR author controls, and the App's own
 * process holds secrets (`GITHUB_APP_PRIVATE_KEY`, `GITHUB_WEBHOOK_SECRET`, the
 * platform's `RAILWAY_*` tokens) that none of them may see. Each builder therefore
 * starts from an empty object and copies in only the names listed below: a new
 * secret added to the host stays out of every child until someone lists it here,
 * which is the opposite of a denylist.
 *
 * tests/unit/test_github_app_child_env_contract.py reads the arrays below (keep each
 * one a `const NAME: readonly string[] = [...]` literal) and fails if a forbidden
 * name appears in one, or if a provider name trelix's config reads goes missing.
 */

// Copied from the host for a git child. HOME is NOT here: git gets an empty one.
const GIT_INHERITED_ENV: readonly string[] = ["PATH", "LANG"];

// Copied from the host for a `trelix index` / `trelix review` child. XDG_CONFIG_HOME is
// there because trelix looks for the operator's `trelix/env` file under it (else under
// $HOME/.config), see resolve_operator_env_file in src/trelix/core/config.py.
const TRELIX_INHERITED_ENV: readonly string[] = [
    "PATH",
    "LANG",
    "HOME",
    "XDG_CONFIG_HOME",
];

// Every env name trelix's config reads that is not under TRELIX_: the provider
// credentials and endpoints that keep the LLM and the embedder configured. Keep in
// step with CONFIG_NON_PREFIXED_ENV in tests/_env_isolation.py (the contract test
// enforces it). Matching is exact-case, so a lowercase spelling is withheld.
const CONFIG_PROVIDER_ENV: readonly string[] = [
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_SECRET_ACCESS_KEY",
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

// The common names a provider SDK reads for itself, which config.py has no field for: a
// role-based AWS login (web identity, ECS/EKS container credentials, a shared credentials
// file, a session token), a Vertex service-account file, an Anthropic identity token, a
// gateway base URL. Not every name the SDKs read: AWS_BEARER_TOKEN_BEDROCK (so Bedrock
// API-key users are not covered), OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_API_TYPE,
// OPENAI_CUSTOM_HEADERS, GOOGLE_GENAI_USE_VERTEXAI and AWS_EC2_METADATA_DISABLED are not
// here, and a deployment that needs one sees the review exit 3 (a neutral check).
// Keep in step with INSTALLED_SDK_PROVIDER_ENV in tests/_env_isolation.py: the contract
// test requires this list to be that one, minus the names in CONFIG_PROVIDER_ENV and
// minus a short, reasoned list of names withheld on purpose (secrets that verify or
// administer rather than call, clients and services trelix never uses, Azure
// service-principal secrets).
const SDK_PROVIDER_ENV: readonly string[] = [
    "ANTHROPIC_API_BASE",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_AWS_API_BASE",
    "ANTHROPIC_AWS_API_KEY",
    "ANTHROPIC_AWS_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "AWS_ACCOUNT_ID",
    "AWS_BEDROCK_BASE_URL",
    "AWS_BEDROCK_RUNTIME_ENDPOINT",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_DEFAULT_PROFILE",
    "AWS_DEFAULT_REGION",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT",
    "AWS_ENDPOINT_URL",
    "AWS_PROFILE_NAME",
    "AWS_REGION_NAME",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_SECURITY_TOKEN",
    "AWS_SESSION_NAME",
    "AWS_SESSION_TOKEN",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_USE_DUALSTACK_ENDPOINT",
    "AWS_USE_FIPS_ENDPOINT",
    "AWS_WEB_IDENTITY_TOKEN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AZURE_AD_TOKEN",
    "AZURE_AI_API_BASE",
    "AZURE_AI_API_KEY",
    "AZURE_AI_API_VERSION",
    "AZURE_API_BASE",
    "AZURE_DEFAULT_RESPONSES_API_VERSION",
    "AZURE_OPENAI_AD_TOKEN",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "COHERE_API_BASE",
    "CO_API_KEY",
    "CO_API_URL",
    "GEMINI_API_BASE",
    "GEMINI_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "OPENAI_API_BASE",
    "OPENAI_API_VERSION",
    "OPENAI_BASE_URL",
    "OPENAI_CHATGPT_API_BASE",
    "OPENAI_LIKE_API_BASE",
    "OPENAI_LIKE_API_KEY",
    "QDRANT_API_BASE",
    "VERTEXAI_API_BASE",
    "VERTEXAI_CREDENTIALS",
    "VERTEX_API_BASE",
    "VERTEX_CREDENTIALS",
    "VOYAGE_AI_API_KEY",
    "VOYAGE_AI_TOKEN",
    "VOYAGE_API_BASE",
    "VOYAGE_API_KEY_PATH",
];

// trelix's own settings: every TRELIX_* name passes (there are hundreds and new ones
// appear), except the git token and this service's own TRELIX_APP_* settings below.
const TRELIX_ENV_PREFIXES: readonly string[] = ["TRELIX_"];

// The token the git askpass helper prints. It belongs to the git child alone, so a
// host that happens to export the name does not leak it into a trelix child.
const GIT_TOKEN_ENV = "TRELIX_GIT_TOKEN";

// This service's own settings (kill switch, install allow-list, queue sizes). They are
// not secrets, but the review child reads text an outside author wrote and a prompt-
// injected reply could quote its environment into a Check, so the operator's allow-list
// stays out of it. trelix itself reads no name under this prefix (a contract test pins it).
const SERVICE_ENV_PREFIX = "TRELIX_APP_";

// Where `trelix review` writes the JSON record of which hunks it did not review. Set by
// the caller for the review child alone (see review-outcome.ts).
const REVIEW_OUTCOME_FILE_ENV = "TRELIX_REVIEW_OUTCOME_FILE";

const GIT_ALLOW_PROTOCOL_DEFAULT = "https";

// trelix runs `git diff`, `git log` and `git rev-parse` itself, with the checkout root as
// its working directory and none of the isolation the service's own git children get.
// Config handed to git through GIT_CONFIG_COUNT has command scope, so this one setting
// still applies to those calls: a bare repository the PR embedded in its tree is refused
// instead of adopted when git finds it implicitly. git older than 2.38 ignores it.
const TRELIX_GIT_ENV: Readonly<Record<string, string>> = {
    GIT_CONFIG_COUNT: "1",
    GIT_CONFIG_KEY_0: "safe.bareRepository",
    GIT_CONFIG_VALUE_0: "explicit",
};

/** Values of the listed `names` present in `base`, as a new object. */
function pick(
    base: NodeJS.ProcessEnv,
    names: readonly string[],
): Record<string, string> {
    const picked: Record<string, string> = {};
    for (const name of names) {
        const value = base[name];
        if (value !== undefined) picked[name] = value;
    }
    return picked;
}

export interface GitChildEnvInput {
    /** Installation token; reaches git only through the askpass helper. */
    token: string;
    /** The askpass script, kept outside the checkout. */
    askpassPath: string;
    /** An empty directory, used as both HOME and XDG_CONFIG_HOME. */
    homeDir: string;
    /** Tests only: a local bare repo needs "file". Never read from the environment. */
    allowProtocol?: string;
}

/**
 * Environment for a `git` child that fetches an outside author's PR. git reads no
 * user, system or repository-template configuration, never prompts, and may only
 * speak https. Returns a new object; `base` is never mutated.
 */
export function buildGitChildEnv(
    input: GitChildEnvInput,
    base: NodeJS.ProcessEnv = process.env,
): NodeJS.ProcessEnv {
    return {
        ...pick(base, GIT_INHERITED_ENV),
        HOME: input.homeDir,
        XDG_CONFIG_HOME: input.homeDir,
        GIT_CONFIG_GLOBAL: "/dev/null",
        GIT_CONFIG_SYSTEM: "/dev/null",
        GIT_CONFIG_NOSYSTEM: "1",
        GIT_TERMINAL_PROMPT: "0",
        GIT_ALLOW_PROTOCOL: input.allowProtocol ?? GIT_ALLOW_PROTOCOL_DEFAULT,
        GIT_ASKPASS: input.askpassPath,
        [GIT_TOKEN_ENV]: input.token,
    };
}

function buildTrelixChildEnv(base: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
    const trelixOwned = Object.entries(base).filter(
        ([name, value]) =>
            value !== undefined &&
            name !== GIT_TOKEN_ENV &&
            !name.startsWith(SERVICE_ENV_PREFIX) &&
            TRELIX_ENV_PREFIXES.some((prefix) => name.startsWith(prefix)),
    );
    return {
        ...pick(base, TRELIX_INHERITED_ENV),
        ...pick(base, CONFIG_PROVIDER_ENV),
        ...pick(base, SDK_PROVIDER_ENV),
        ...Object.fromEntries(trelixOwned),
        ...TRELIX_GIT_ENV,
        // trelix follows symlinks out of the repo by default and a PR can commit any
        // symlink it likes. Forced here, whatever the host passed in, so it stays
        // true even if the image or platform config drops the Dockerfile's setting.
        TRELIX_WALKER_FOLLOW_SYMLINKS: "false",
    };
}

/** Environment for `trelix index`. No GitHub credential of any kind. */
export function buildIndexChildEnv(
    base: NodeJS.ProcessEnv = process.env,
): NodeJS.ProcessEnv {
    return buildTrelixChildEnv(base);
}

/**
 * Environment for `trelix review --pr`, which fetches the PR diff through the
 * GitHub API and so needs the installation token. The token is passed in
 * explicitly and overrides any `GITHUB_TOKEN` the host has; it is never copied
 * from `base`. `outcomeFile`, when given, is where `trelix review` writes its
 * record of what it covered: it overrides the host's value of
 * `TRELIX_REVIEW_OUTCOME_FILE`, so the path is always one the caller chose.
 */
export function buildReviewChildEnv(
    token: string,
    base: NodeJS.ProcessEnv = process.env,
    outcomeFile?: string,
): NodeJS.ProcessEnv {
    return {
        ...buildTrelixChildEnv(base),
        GITHUB_TOKEN: token,
        ...(outcomeFile === undefined
            ? {}
            : { [REVIEW_OUTCOME_FILE_ENV]: outcomeFile }),
    };
}
