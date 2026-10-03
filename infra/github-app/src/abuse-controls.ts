import {
    InstallPolicy,
    OPEN_POLICY,
    parseAccountList,
    parseInstallPolicyMode,
    parseInstallationList,
} from "./policy.js";
import type { QueueLimits } from "./queue.js";

/**
 * The switches that limit what a flood of webhook deliveries can cost: the
 * kill switch, the installation policy and the review queue's caps. Read once
 * at startup, from env only. A change is a redeploy.
 *
 * Every parser fails closed. A number that cannot be read falls back to its
 * default and is logged; an unknown policy name throws, so the service does
 * not start under a policy nobody chose; a kill switch that cannot be read
 * is read as "off". Nothing is logged with the value: an env var can hold
 * anything, including a secret pasted into the wrong variable.
 */

export interface AbuseControls {
    /** False: every pull_request delivery is acknowledged and ignored. */
    readonly reviewsEnabled: boolean;
    readonly installPolicy: InstallPolicy;
    readonly queue: QueueLimits;
}

export const DEFAULT_QUEUE_LIMITS: QueueLimits = {
    capacity: 20,
    concurrency: 2,
    perGroupConcurrency: 1,
};

/** What a deployment gets with none of the variables set. */
export const DEFAULT_ABUSE_CONTROLS: AbuseControls = {
    reviewsEnabled: true,
    installPolicy: OPEN_POLICY,
    queue: DEFAULT_QUEUE_LIMITS,
};

export const REVIEWS_ENABLED_ENV = "TRELIX_APP_REVIEWS_ENABLED";
export const INSTALL_POLICY_ENV = "TRELIX_APP_INSTALL_POLICY";
export const ALLOWED_ACCOUNTS_ENV = "TRELIX_APP_ALLOWED_ACCOUNTS";
export const ALLOWED_INSTALLATIONS_ENV = "TRELIX_APP_ALLOWED_INSTALLATIONS";
export const QUEUE_CAPACITY_ENV = "TRELIX_APP_QUEUE_CAPACITY";
export const CONCURRENCY_ENV = "TRELIX_APP_CONCURRENCY";
export const CONCURRENCY_PER_INSTALLATION_ENV =
    "TRELIX_APP_CONCURRENCY_PER_INSTALLATION";

/** Upper bounds, so a typo cannot ask for an unbounded queue or an unbounded fan-out. */
const MAX_QUEUE_CAPACITY = 1000;
const MAX_CONCURRENCY = 16;

const TRUE_WORDS: readonly string[] = ["true", "1", "yes", "on"];
const FALSE_WORDS: readonly string[] = ["false", "0", "no", "off"];

/** A plain decimal: digits only, so "1e3", "0x10", "-1" and "2 cores" are all rejected. */
const WHOLE_NUMBER_PATTERN = /^[0-9]{1,9}$/;

type Warn = (line: string) => void;

function warnOnConsole(line: string): void {
    console.warn(line);
}

interface NumberRule {
    readonly name: string;
    readonly fallback: number;
    readonly min: number;
    readonly max: number;
}

/** A whole number within the rule's range; unset or blank is the fallback silently, anything else unreadable is the fallback with a warning. */
function readWholeNumber(
    env: NodeJS.ProcessEnv,
    rule: NumberRule,
    warn: Warn,
): number {
    const raw = (env[rule.name] ?? "").trim();
    if (raw === "") {
        return rule.fallback;
    }
    const value = WHOLE_NUMBER_PATTERN.test(raw) ? Number(raw) : NaN;
    if (!(value >= rule.min && value <= rule.max)) {
        warn(
            `[config] ${rule.name} is not a whole number from ${rule.min} to ${rule.max}; using the default ${rule.fallback}`,
        );
        return rule.fallback;
    }
    return value;
}

/** The kill switch. Unset or blank is on; a word that is neither true nor false is off, loudly. */
function readReviewsEnabled(env: NodeJS.ProcessEnv, warn: Warn): boolean {
    const word = (env[REVIEWS_ENABLED_ENV] ?? "").trim().toLowerCase();
    if (word === "" || TRUE_WORDS.includes(word)) {
        return true;
    }
    if (!FALSE_WORDS.includes(word)) {
        warn(
            `[config] WARNING: ${REVIEWS_ENABLED_ENV} is not true or false; reviews are OFF until it is fixed`,
        );
    }
    return false;
}

function readQueueLimits(env: NodeJS.ProcessEnv, warn: Warn): QueueLimits {
    return {
        capacity: readWholeNumber(
            env,
            {
                name: QUEUE_CAPACITY_ENV,
                fallback: DEFAULT_QUEUE_LIMITS.capacity,
                min: 1,
                max: MAX_QUEUE_CAPACITY,
            },
            warn,
        ),
        concurrency: readWholeNumber(
            env,
            {
                name: CONCURRENCY_ENV,
                fallback: DEFAULT_QUEUE_LIMITS.concurrency,
                min: 1,
                max: MAX_CONCURRENCY,
            },
            warn,
        ),
        perGroupConcurrency: readWholeNumber(
            env,
            {
                name: CONCURRENCY_PER_INSTALLATION_ENV,
                fallback: DEFAULT_QUEUE_LIMITS.perGroupConcurrency,
                min: 1,
                max: MAX_CONCURRENCY,
            },
            warn,
        ),
    };
}

function readInstallPolicy(env: NodeJS.ProcessEnv, warn: Warn): InstallPolicy {
    const mode = parseInstallPolicyMode(env[INSTALL_POLICY_ENV]);
    const accounts = parseAccountList(env[ALLOWED_ACCOUNTS_ENV]);
    const installations = parseInstallationList(env[ALLOWED_INSTALLATIONS_ENV]);
    if (installations.ignored > 0) {
        warn(
            `[config] ${installations.ignored} value(s) in ${ALLOWED_INSTALLATIONS_ENV} are not an installation id and allow nothing`,
        );
    }
    const hasList = accounts.size > 0 || installations.ids.size > 0;
    if (mode === "open") {
        warn(
            `[config] WARNING: ${INSTALL_POLICY_ENV} is open: any account that installs this App can spend this deployment's LLM quota. Set it to allowlist, with ${ALLOWED_ACCOUNTS_ENV} and/or ${ALLOWED_INSTALLATIONS_ENV}, to restrict who is served`,
        );
        if (hasList) {
            warn(
                `[config] WARNING: an allow-list is set but ${INSTALL_POLICY_ENV} is not allowlist, so the list is NOT applied`,
            );
        }
        return OPEN_POLICY;
    }
    if (!hasList) {
        warn(
            `[config] WARNING: ${INSTALL_POLICY_ENV} is allowlist but no account or installation is listed; every installation will be ignored`,
        );
    }
    return { mode, accounts, installations: installations.ids };
}

/**
 * Reads the controls from `env`. Throws, before the service listens, when
 * TRELIX_APP_INSTALL_POLICY names a policy that does not exist. Problems that
 * fail safe are reported through `warn` (the console by default).
 */
export function loadAbuseControls(
    env: NodeJS.ProcessEnv = process.env,
    warn: Warn = warnOnConsole,
): AbuseControls {
    const reviewsEnabled = readReviewsEnabled(env, warn);
    if (!reviewsEnabled) {
        warn(
            `[config] WARNING: reviews are disabled (${REVIEWS_ENABLED_ENV}); every pull_request delivery will be acknowledged and ignored`,
        );
    }
    return {
        reviewsEnabled,
        installPolicy: readInstallPolicy(env, warn),
        queue: readQueueLimits(env, warn),
    };
}
