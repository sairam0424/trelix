/**
 * Who may use this deployment of the App.
 *
 * Anyone can install a public GitHub App, and every review spends the
 * operator's LLM quota, so a deployment can restrict the installations it
 * serves. "open" serves every installation (the default, so a self-hosted
 * deployment keeps working after an upgrade); "allowlist" serves only the
 * accounts and installation ids listed.
 *
 * Parsing fails closed. An unknown policy name is an error (the service
 * refuses to start). In allowlist mode, a list entry that cannot be read is
 * dropped and an empty list allows nothing, so a typo narrows access instead
 * of widening it. Nothing here throws for a value that is merely wrong in a
 * list, and nothing echoes a value: an env var can hold anything.
 */

export type InstallPolicyMode = "open" | "allowlist";

export interface InstallPolicy {
    readonly mode: InstallPolicyMode;
    /** Lowercase GitHub logins; GitHub treats logins case-insensitively. */
    readonly accounts: ReadonlySet<string>;
    readonly installations: ReadonlySet<number>;
}

/** Serves every installation. */
export const OPEN_POLICY: InstallPolicy = {
    mode: "open",
    accounts: new Set(),
    installations: new Set(),
};

export interface InstallTarget {
    /** The account the App is installed on: the owner of the repository. */
    readonly account: string;
    readonly installationId: number;
}

const POLICY_MODES: readonly InstallPolicyMode[] = ["open", "allowlist"];

/** A decimal installation id: no sign, no leading zero, no exponent, at most 16 digits. */
const INSTALLATION_ID_PATTERN = /^[1-9][0-9]{0,15}$/;

/**
 * The mode named by `raw`, compared without case or surrounding space; unset
 * or blank means "open". Throws for any other text, without quoting it.
 */
export function parseInstallPolicyMode(
    raw: string | undefined,
): InstallPolicyMode {
    const name = (raw ?? "").trim().toLowerCase();
    if (name === "") {
        return "open";
    }
    const mode = POLICY_MODES.find((candidate) => candidate === name);
    if (mode === undefined) {
        throw new Error(
            `Invalid TRELIX_APP_INSTALL_POLICY: expected "open" or "allowlist"`,
        );
    }
    return mode;
}

/** Comma-separated entries: trimmed, blanks dropped, duplicates merged. Linear in the length of `raw`. */
function splitList(raw: string | undefined): string[] {
    if (raw === undefined) {
        return [];
    }
    return raw
        .split(",")
        .map((entry) => entry.trim())
        .filter((entry) => entry !== "");
}

/** GitHub logins from a comma-separated list, lowercased. */
export function parseAccountList(raw: string | undefined): Set<string> {
    return new Set(splitList(raw).map((entry) => entry.toLowerCase()));
}

export interface InstallationList {
    readonly ids: Set<number>;
    /** Entries that are not an installation id, and so allow nothing. */
    readonly ignored: number;
}

/** Installation ids from a comma-separated list; anything that is not a positive whole number is counted and dropped. */
export function parseInstallationList(
    raw: string | undefined,
): InstallationList {
    const ids = new Set<number>();
    let ignored = 0;
    for (const entry of splitList(raw)) {
        const id = INSTALLATION_ID_PATTERN.test(entry) ? Number(entry) : NaN;
        if (Number.isSafeInteger(id)) {
            ids.add(id);
        } else {
            ignored += 1;
        }
    }
    return { ids, ignored };
}

/**
 * Whether `target` may be served. In allowlist mode an installation is served
 * if its account is listed or its id is listed; an empty policy serves no
 * one, and so does any mode this function does not recognise.
 */
export function isInstallationAllowed(
    policy: InstallPolicy,
    target: InstallTarget,
): boolean {
    if (policy.mode === "open") {
        return true;
    }
    return (
        policy.accounts.has(target.account.toLowerCase()) ||
        policy.installations.has(target.installationId)
    );
}
